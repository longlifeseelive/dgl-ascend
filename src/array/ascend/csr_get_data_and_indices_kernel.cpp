#include "kernel_operator.h"
using namespace AscendC;

namespace {
constexpr uint32_t kBatch = 64;
constexpr uint32_t kRow = 128;
constexpr uint32_t kOutput = 128;
constexpr uint32_t kPtrLimit = 12288;

template <typename T>
__aicore__ inline void Load(LocalTensor<T> dst, GlobalTensor<T> src,
                            uint64_t pos, uint32_t n) {
  DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> pad{false, 0, 0, 0};
  DataCopyPad(dst, src[pos], cp, pad);
}
template <typename T>
__aicore__ inline void Store(GlobalTensor<T> dst, uint64_t pos,
                             LocalTensor<T> src, uint32_t n) {
  DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
  DataCopyPad(dst[pos], src, cp);
}
__aicore__ inline uint64_t Chunk(uint64_t n) {
  return ((n + GetBlockNum() - 1) / GetBlockNum() + 7) / 8 * 8;
}
__aicore__ inline uint64_t Popcount(uint64_t x) {
  x -= (x >> 1) & 0x5555555555555555ULL;
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
  return (x * 0x0101010101010101ULL) >> 56;
}

template <typename IdT>
class Masks {
 public:
  static constexpr uint32_t kWords = sizeof(IdT) / 4;
  static constexpr uint32_t kMasks = 2 * kWords;
  static constexpr uint32_t kSpan = 64 / kWords;
  __aicore__ inline void Init(TPipe* pipe, uint32_t slots) {
    pipe->InitBuffer(low_, slots * 32);
    pipe->InitBuffer(high_, sizeof(IdT) == 8 ? slots * 32 : 32);
  }
  __aicore__ inline void Compare(LocalTensor<IdT> input, uint32_t slot,
                                 IdT col, uint32_t take) {
    const uint32_t n = (take <= 64 ? 64 : 128) * kWords;
    CompareScalar(low_.Get<uint8_t>()[slot * 32],
                  input.template ReinterpretCast<int32_t>(),
                  static_cast<int32_t>(col), CMPMODE::EQ, n);
    if constexpr (sizeof(IdT) == 8) {
      CompareScalar(high_.Get<uint8_t>()[slot * 32],
                    input.template ReinterpretCast<int32_t>(),
                    static_cast<int32_t>(static_cast<uint64_t>(col) >> 32),
                    CMPMODE::EQ, n);
    }
  }
  __aicore__ inline uint64_t Get(uint32_t slot, uint32_t word, uint32_t take) {
    if (word * kSpan >= take) return 0;
    uint64_t bits = low_.Get<uint64_t>().GetValue(slot * 4 + word);
    if constexpr (sizeof(IdT) == 8) {
      bits &= (high_.Get<uint64_t>().GetValue(slot * 4 + word) >> 1)
              & 0x5555555555555555ULL;
    }
    const uint32_t valid = take - word * kSpan < kSpan ? take - word * kSpan : kSpan;
    if (valid * kWords < 64) bits &= (uint64_t(1) << (valid * kWords)) - 1;
    return bits;
  }
 private:
  TBuf<TPosition::VECCALC> low_, high_;
};

template <typename IdT>
class CountMatches {
 public:
  static constexpr uint32_t kMasks = Masks<IdT>::kMasks;
  static constexpr uint32_t kRecord = 2 + kMasks;
  __aicore__ inline void Init(GM_ADDR ptr, GM_ADDR idx, GM_ADDR rows,
      GM_ADDR cols, GM_ADDR records, GM_ADDR stats, uint64_t m, uint64_t n,
      uint64_t queries, uint32_t rs, uint32_t cs, uint32_t sorted, TPipe* pipe) {
    ptr_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(ptr));
    idx_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(idx));
    rows_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(rows));
    cols_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(cols));
    records_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(records));
    stats_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(stats));
    m_ = m; n_ = n; queries_ = queries; rs_ = rs; cs_ = cs; sorted_ = sorted;
    chunk_ = Chunk(queries);
    begin_ = GetBlockIdx() * chunk_;
    end_ = begin_ + chunk_ < queries ? begin_ + chunk_ : queries;
    cached_ = m < kPtrLimit && chunk_ >= kBatch;
    pipe->InitBuffer(ptr_buf_, cached_ ? ((m + 8) / 8 * 8) * sizeof(IdT) : 32);
    pipe->InitBuffer(row_buf_, kBatch * kRow * sizeof(IdT));
    pipe->InitBuffer(rows_buf_, kBatch * sizeof(IdT));
    pipe->InitBuffer(cols_buf_, kBatch * sizeof(IdT));
    pipe->InitBuffer(start_buf_, kBatch * sizeof(IdT));
    pipe->InitBuffer(end_buf_, kBatch * sizeof(IdT));
    pipe->InitBuffer(record_buf_, kBatch * kRecord * 8);
    pipe->InitBuffer(stat_buf_, 32);
    masks_.Init(pipe, kBatch);
  }
  __aicore__ inline void Process() {
    uint64_t total = 0, error = 0;
    auto pointers = ptr_buf_.Get<IdT>();
    auto rows = rows_buf_.Get<IdT>(); auto cols = cols_buf_.Get<IdT>();
    auto starts = start_buf_.Get<IdT>(); auto stops = end_buf_.Get<IdT>();
    auto data = row_buf_.Get<IdT>(); auto rec = record_buf_.Get<uint64_t>();
    if (cached_ && begin_ < end_) {
      Load(pointers, ptr_, 0, static_cast<uint32_t>(m_ + 1));
      SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
    }
    for (uint64_t base = begin_; base < end_; base += kBatch) {
      const uint32_t qn = end_ - base < kBatch ? end_ - base : kBatch;
      SetFlag<HardEvent::S_MTE2>(0); WaitFlag<HardEvent::S_MTE2>(0);
      Load(rows, rows_, rs_ ? base : 0, rs_ ? qn : 1);
      Load(cols, cols_, cs_ ? base : 0, cs_ ? qn : 1);
      SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
      for (uint32_t i = 0; i < qn; ++i) {
        const IdT r = rows.GetValue(rs_ ? i : 0), c = cols.GetValue(cs_ ? i : 0);
        const bool vr = r >= 0 && static_cast<uint64_t>(r) < m_;
        const bool vc = c >= 0 && static_cast<uint64_t>(c) < n_;
        IdT start = 0, stop = 0;
        if ((!vr || !vc) && !error) error = 2 * (base + i) + (vr ? 2 : 1);
        if (vr && vc) {
          start = cached_ ? pointers.GetValue(r) : ptr_.GetValue(r);
          stop = cached_ ? pointers.GetValue(r + 1) : ptr_.GetValue(r + 1);
        }
        starts.SetValue(i, start); stops.SetValue(i, stop);
        const uint64_t degree = static_cast<uint64_t>(stop - start);
        if (degree && !(sorted_ && degree > kRow)) {
          Load(data[i * kRow], idx_, start, degree < kRow ? degree : kRow);
        }
      }
      SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
      SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
      for (uint32_t i = 0; i < qn; ++i) {
        const uint64_t degree = static_cast<uint64_t>(stops.GetValue(i) - starts.GetValue(i));
        if (degree && !(sorted_ && degree > kRow))
          masks_.Compare(data[i * kRow], i, cols.GetValue(cs_ ? i : 0), degree < kRow ? degree : kRow);
      }
      SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
      for (uint32_t i = 0; i < qn; ++i) {
        const uint64_t start = starts.GetValue(i), stop = stops.GetValue(i);
        const uint64_t degree = stop - start;
        const IdT col = cols.GetValue(cs_ ? i : 0);
        rec.SetValue(i * kRecord, start); rec.SetValue(i * kRecord + 1, stop);
        uint64_t matches = 0;
        if (sorted_ && degree > kRow) {
          uint64_t lo = start, hi = stop;
          while (lo < hi) {
            const uint64_t mid = lo + (hi - lo) / 2;
            if (idx_.GetValue(mid) < col) lo = mid + 1; else hi = mid;
          }
          const uint64_t first = lo; hi = stop;
          while (lo < hi) {
            const uint64_t mid = lo + (hi - lo) / 2;
            if (idx_.GetValue(mid) <= col) lo = mid + 1; else hi = mid;
          }
          matches = lo - first;
          for (uint32_t w = 0; w < kMasks; ++w)
            rec.SetValue(i * kRecord + 2 + w, w == 0 ? first : w == 1 ? lo : 0);
        } else {
          const uint32_t take = degree < kRow ? degree : kRow;
          for (uint32_t w = 0; w < kMasks; ++w) {
            const uint64_t bits = masks_.Get(i, w, take);
            rec.SetValue(i * kRecord + 2 + w, bits); matches += Popcount(bits);
          }
          for (uint64_t tile = start + take; tile < stop; tile += kRow) {
            const uint32_t count = stop - tile < kRow ? stop - tile : kRow;
            SetFlag<HardEvent::S_MTE2>(0); WaitFlag<HardEvent::S_MTE2>(0);
            Load(data[i * kRow], idx_, tile, count);
            SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
            SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
            masks_.Compare(data[i * kRow], i, col, count);
            SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
            for (uint32_t w = 0; w < kMasks; ++w) matches += Popcount(masks_.Get(i, w, count));
          }
        }
        total += matches;
      }
      SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
      Store(records_, base * kRecord, rec, qn * kRecord);
      SetFlag<HardEvent::MTE3_S>(0); WaitFlag<HardEvent::MTE3_S>(0);
    }
    auto state = stat_buf_.Get<uint64_t>();
    state.SetValue(0, total); state.SetValue(1, error);
    state.SetValue(2, 0); state.SetValue(3, 0);
    SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
    Store(stats_, GetBlockIdx() * 4, state, 4);
    SetFlag<HardEvent::MTE3_S>(0); WaitFlag<HardEvent::MTE3_S>(0);
  }
 private:
  GlobalTensor<IdT> ptr_, idx_, rows_, cols_;
  GlobalTensor<uint64_t> records_, stats_;
  uint64_t m_, n_, queries_, chunk_, begin_, end_;
  uint32_t rs_, cs_, sorted_;
  bool cached_;
  Masks<IdT> masks_;
  TBuf<TPosition::VECCALC> ptr_buf_, row_buf_, rows_buf_, cols_buf_, start_buf_, end_buf_, record_buf_, stat_buf_;
};

template <typename IdT>
class EmitMatches {
 public:
  static constexpr uint32_t kWords = Masks<IdT>::kWords;
  static constexpr uint32_t kMasks = Masks<IdT>::kMasks;
  static constexpr uint32_t kRecord = 2 + kMasks;
  __aicore__ inline void Init(GM_ADDR idx, GM_ADDR data, GM_ADDR rows,
      GM_ADDR cols, GM_ADDR records, GM_ADDR stats, GM_ADDR out_rows,
      GM_ADDR out_cols, GM_ADDR out_ids, uint64_t queries, uint32_t rs,
      uint32_t cs, uint32_t has_data, uint32_t sorted, TPipe* pipe) {
    idx_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(idx));
    data_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(data));
    rows_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(rows));
    cols_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(cols));
    records_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(records));
    stats_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(stats));
    out_rows_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(out_rows));
    out_cols_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(out_cols));
    out_ids_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(out_ids));
    queries_ = queries; rs_ = rs; cs_ = cs; has_data_ = has_data; sorted_ = sorted;
    pipe->InitBuffer(records_buf_, kBatch * kRecord * 8);
    pipe->InitBuffer(stats_buf_, 32);
    pipe->InitBuffer(rows_buf_, kBatch * sizeof(IdT));
    pipe->InitBuffer(cols_buf_, kBatch * sizeof(IdT));
    pipe->InitBuffer(tile_buf_, kRow * sizeof(IdT));
    pipe->InitBuffer(pos_buf_, kOutput * sizeof(IdT));
    pipe->InitBuffer(out_row_buf_, kOutput * sizeof(IdT));
    pipe->InitBuffer(out_col_buf_, kOutput * sizeof(IdT));
    pipe->InitBuffer(out_id_buf_, kOutput * sizeof(IdT));
    pipe->InitBuffer(padded_buf_, kOutput * 32);
    pipe->InitBuffer(gather_buf_, kOutput * kWords * sizeof(uint32_t));
    masks_.Init(pipe, 1);
  }
  __aicore__ inline void Process() {
    const uint64_t chunk = Chunk(queries_);
    const uint64_t begin = GetBlockIdx() * chunk;
    const uint64_t end = begin + chunk < queries_ ? begin + chunk : queries_;
    if (begin >= end) return;
    auto state = stats_buf_.Get<uint64_t>();
    Load(state, stats_, GetBlockIdx() * 4, 4);
    SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
    cursor_ = state.GetValue(2);
    auto gather = gather_buf_.Get<uint32_t>();
    for (uint32_t i = 0; i < kOutput; ++i) {
      gather.SetValue(i * kWords, i * 32);
      if constexpr (kWords == 2) gather.SetValue(i * kWords + 1, i * 32 + 4);
    }
    auto rec = records_buf_.Get<uint64_t>();
    auto rows = rows_buf_.Get<IdT>(); auto cols = cols_buf_.Get<IdT>();
    for (uint64_t base = begin; base < end; base += kBatch) {
      const uint32_t qn = end - base < kBatch ? end - base : kBatch;
      SetFlag<HardEvent::S_MTE2>(0); WaitFlag<HardEvent::S_MTE2>(0);
      Load(rec, records_, base * kRecord, qn * kRecord);
      Load(rows, rows_, rs_ ? base : 0, rs_ ? qn : 1);
      Load(cols, cols_, cs_ ? base : 0, cs_ ? qn : 1);
      SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
      for (uint32_t i = 0; i < qn; ++i) {
        const IdT row = rows.GetValue(rs_ ? i : 0), col = cols.GetValue(cs_ ? i : 0);
        const uint64_t start = rec.GetValue(i * kRecord), stop = rec.GetValue(i * kRecord + 1);
        if (sorted_ && stop - start > kRow) {
          const uint64_t first = rec.GetValue(i * kRecord + 2), last = rec.GetValue(i * kRecord + 3);
          for (uint64_t edge = first; edge < last; ++edge) Append(edge, row, col);
        } else {
          for (uint32_t w = 0; w < kMasks; ++w)
            AppendBits(rec.GetValue(i * kRecord + 2 + w), w, start, row, col);
          for (uint64_t tile = start + kRow; tile < stop; tile += kRow) {
            const uint32_t take = stop - tile < kRow ? stop - tile : kRow;
            auto input = tile_buf_.Get<IdT>();
            SetFlag<HardEvent::S_MTE2>(0); WaitFlag<HardEvent::S_MTE2>(0);
            Load(input, idx_, tile, take);
            SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
            SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
            masks_.Compare(input, 0, col, take);
            SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
            for (uint32_t w = 0; w < kMasks; ++w) AppendBits(masks_.Get(0, w, take), w, tile, row, col);
          }
        }
      }
    }
    if (pending_) Flush();
  }
 private:
  __aicore__ inline void AppendBits(uint64_t bits, uint32_t word, uint64_t base, IdT row, IdT col) {
    while (bits) {
      const uint32_t bit = static_cast<uint32_t>(GetSFFValue<1>(bits));
      Append(base + word * Masks<IdT>::kSpan + bit / kWords, row, col);
      bits &= bits - 1;
    }
  }
  __aicore__ inline void Append(uint64_t edge, IdT row, IdT col) {
    pos_buf_.Get<IdT>().SetValue(pending_, static_cast<IdT>(edge));
    out_row_buf_.Get<IdT>().SetValue(pending_, row);
    out_col_buf_.Get<IdT>().SetValue(pending_, col);
    if (!has_data_) out_id_buf_.Get<IdT>().SetValue(pending_, static_cast<IdT>(edge));
    if (++pending_ == kOutput) Flush();
  }
  __aicore__ inline void Flush() {
    auto ids = out_id_buf_.Get<IdT>();
    if (has_data_) {
      auto positions = pos_buf_.Get<IdT>();
      const IdT first = positions.GetValue(0);
      bool consecutive = true;
      for (uint32_t i = 1; i < pending_; ++i)
        if (positions.GetValue(i) != first + i) { consecutive = false; break; }
      SetFlag<HardEvent::S_MTE2>(0); WaitFlag<HardEvent::S_MTE2>(0);
      if (consecutive) {
        Load(ids, data_, first, pending_);
        SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
      } else {
        auto padded = padded_buf_.Get<IdT>();
        for (uint32_t i = 0; i < pending_; ++i)
          Load(padded[i * (32 / sizeof(IdT))], data_, positions.GetValue(i), 1);
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
        Gather(ids.template ReinterpretCast<int32_t>(), padded.template ReinterpretCast<int32_t>(),
               gather_buf_.Get<uint32_t>(), 0, pending_ * kWords);
        SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
      }
    }
    SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
    Store(out_rows_, cursor_, out_row_buf_.Get<IdT>(), pending_);
    Store(out_cols_, cursor_, out_col_buf_.Get<IdT>(), pending_);
    Store(out_ids_, cursor_, ids, pending_);
    SetFlag<HardEvent::MTE3_S>(0); WaitFlag<HardEvent::MTE3_S>(0);
    cursor_ += pending_; pending_ = 0;
  }
  GlobalTensor<IdT> idx_, data_, rows_, cols_, out_rows_, out_cols_, out_ids_;
  GlobalTensor<uint64_t> records_, stats_;
  uint64_t queries_, cursor_ = 0;
  uint32_t rs_, cs_, has_data_, sorted_, pending_ = 0;
  Masks<IdT> masks_;
  TBuf<TPosition::VECCALC> records_buf_, stats_buf_, rows_buf_, cols_buf_, tile_buf_;
  TBuf<TPosition::VECCALC> pos_buf_, out_row_buf_, out_col_buf_, out_id_buf_, padded_buf_, gather_buf_;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_all_prefix(GM_ADDR stats_ptr, uint32_t cores) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> stats_buf, summary_buf;
  pipe.InitBuffer(stats_buf, 64 * 32); pipe.InitBuffer(summary_buf, 32);
  auto stats = stats_buf.Get<uint64_t>(); auto summary = summary_buf.Get<uint64_t>();
  GlobalTensor<uint64_t> gm; gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(stats_ptr));
  Load(stats, gm, 0, cores * 4);
  SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
  uint64_t total = 0, error = 0;
  for (uint32_t c = 0; c < cores; ++c) {
    stats.SetValue(c * 4 + 2, total); total += stats.GetValue(c * 4);
    const uint64_t code = stats.GetValue(c * 4 + 1);
    if (code && (!error || code < error)) error = code;
  }
  summary.SetValue(0, total); summary.SetValue(1, error);
  summary.SetValue(2, 0); summary.SetValue(3, 0);
  SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
  Store(gm, 0, stats, cores * 4); Store(gm, cores * 4, summary, 4);
  SetFlag<HardEvent::MTE3_S>(0); WaitFlag<HardEvent::MTE3_S>(0);
}

#define ALL_KERNELS(SUFFIX, ID) \
extern "C" __global__ __aicore__ void csr_all_count_##SUFFIX( \
    GM_ADDR ptr, GM_ADDR idx, GM_ADDR rows, GM_ADDR cols, GM_ADDR records, GM_ADDR stats, \
    uint64_t m, uint64_t n, uint64_t queries, uint32_t rs, uint32_t cs, uint32_t sorted) { \
  TPipe pipe; CountMatches<ID> op; \
  op.Init(ptr, idx, rows, cols, records, stats, m, n, queries, rs, cs, sorted, &pipe); op.Process(); \
} \
extern "C" __global__ __aicore__ void csr_all_emit_##SUFFIX( \
    GM_ADDR idx, GM_ADDR data, GM_ADDR rows, GM_ADDR cols, GM_ADDR records, GM_ADDR stats, \
    GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_ids, uint64_t queries, \
    uint32_t rs, uint32_t cs, uint32_t has_data, uint32_t sorted) { \
  TPipe pipe; EmitMatches<ID> op; \
  op.Init(idx, data, rows, cols, records, stats, out_rows, out_cols, out_ids, \
          queries, rs, cs, has_data, sorted, &pipe); op.Process(); \
}
ALL_KERNELS(int32, int32_t)
ALL_KERNELS(int64, int64_t)
#undef ALL_KERNELS
