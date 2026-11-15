#include "kernel_operator.h"
using namespace AscendC;

namespace {
// Prefetch a group of independent rows before waiting for MTE2. All query and
// result transfers are contiguous DMAs; no scalar GM output stores are used.
constexpr uint32_t kQueries = 512;
constexpr uint32_t kBatch = 64;
constexpr uint32_t kRow = 128;
constexpr uint32_t kPtrLimit = 12288;

template <typename T>
__aicore__ inline void Load(LocalTensor<T> dst, GlobalTensor<T> src,
                            uint64_t pos, uint32_t size) {
  DataCopyExtParams cp{1, static_cast<uint32_t>(size * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> pad{false, 0, 0, 0};
  DataCopyPad(dst, src[pos], cp, pad);
}

template <typename IdT, typename OutT, bool Weighted>
class KernelLookup {
 public:
  __aicore__ inline void Init(
      GM_ADDR ptr, GM_ADDR idx, GM_ADDR data, GM_ADDR rows, GM_ADDR cols,
      GM_ADDR weights, GM_ADDR out, uint32_t num_rows, uint64_t num_cols,
      uint64_t count, uint32_t row_stride, uint32_t col_stride,
      uint32_t has_data, uint32_t sorted, uint64_t filler, TPipe* pipe) {
    ptr_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(ptr));
    idx_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(idx));
    data_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(data));
    rows_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(rows));
    cols_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(cols));
    weights_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ OutT*>(weights));
    out_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ OutT*>(out));
    num_rows_ = num_rows; num_cols_ = num_cols; count_ = count;
    row_stride_ = row_stride; col_stride_ = col_stride;
    has_data_ = has_data; sorted_ = sorted; filler_ = static_cast<OutT>(filler);
    const uint64_t chunk = ((count + GetBlockNum() - 1) / GetBlockNum() + 7) / 8 * 8;
    begin_ = GetBlockIdx() * chunk;
    end_ = begin_ + chunk < count ? begin_ + chunk : count;
    cache_ptr_ = num_rows < kPtrLimit && chunk >= kBatch;
    pipe->InitBuffer(ptr_buf_, cache_ptr_ ? ((num_rows + 8) / 8 * 8) * sizeof(IdT) : 32);
    pipe->InitBuffer(rows_buf_, kQueries * sizeof(IdT));
    pipe->InitBuffer(cols_buf_, kQueries * sizeof(IdT));
    pipe->InitBuffer(out_buf_, kQueries * sizeof(OutT));
    pipe->InitBuffer(row_buf_, kBatch * kRow * sizeof(IdT));
    pipe->InitBuffer(start_buf_, kBatch * sizeof(IdT));
    pipe->InitBuffer(end_buf_, kBatch * sizeof(IdT));
    pipe->InitBuffer(mask_lo_buf_, kBatch * 32);
    pipe->InitBuffer(mask_hi_buf_, sizeof(IdT) == 8 ? kBatch * 32 : 32);
    pipe->InitBuffer(eid_buf_, kBatch * 32);
    pipe->InitBuffer(value_buf_, Weighted ? kBatch * 32 : 32);
    pipe->InitBuffer(field_offsets_buf_, kBatch * sizeof(OutT));
    if constexpr (sizeof(IdT) == 4) {
      pipe->InitBuffer(col_seed_buf_, kBatch * 8 * sizeof(int32_t));
      pipe->InitBuffer(batch_mask_buf_, kBatch * kRow / 8);
    }
  }

  __aicore__ inline void Process() {
    if (begin_ >= end_) return;
    auto pointers = ptr_buf_.Get<IdT>();
    auto rows = rows_buf_.Get<IdT>();
    auto cols = cols_buf_.Get<IdT>();
    auto result = out_buf_.Get<OutT>();
    auto row_data = row_buf_.Get<IdT>();
    auto starts = start_buf_.Get<IdT>();
    auto ends = end_buf_.Get<IdT>();
    auto eid_slots = eid_buf_.Get<IdT>();
    auto value_slots = value_buf_.Get<OutT>();
    auto offsets = field_offsets_buf_.Get<uint32_t>();
    constexpr uint32_t out_words = sizeof(OutT) / 4;
    for (uint32_t i = 0; i < kBatch; ++i) {
      offsets.SetValue(i * out_words, i * 32);
      if constexpr (out_words == 2) offsets.SetValue(i * out_words + 1, i * 32 + 4);
    }
    if (cache_ptr_) {
      Load(pointers, ptr_gm_, 0, num_rows_ + 1);
      SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
    }
    for (uint64_t base = begin_; base < end_; base += kQueries) {
      const uint32_t n = end_ - base < kQueries ? end_ - base : kQueries;
      SetFlag<HardEvent::S_MTE2>(0); WaitFlag<HardEvent::S_MTE2>(0);
      Load(rows, rows_gm_, row_stride_ ? base : 0, row_stride_ ? n : 1);
      Load(cols, cols_gm_, col_stride_ ? base : 0, col_stride_ ? n : 1);
      SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
      for (uint32_t batch = 0; batch < n; batch += kBatch) {
        const uint32_t qn = n - batch < kBatch ? n - batch : kBatch;
        SetFlag<HardEvent::S_MTE2>(0); WaitFlag<HardEvent::S_MTE2>(0);
        for (uint32_t i = 0; i < qn; ++i) {
          const IdT r = rows.GetValue(row_stride_ ? batch + i : 0);
          IdT start = 0, stop = 0;
          // Valid IDs are the public contract. Guard malformed device input
          // here to prevent an out-of-range device memory access.
          if (r >= 0 && static_cast<uint64_t>(r) < num_rows_) {
            start = cache_ptr_ ? pointers.GetValue(r) : ptr_gm_.GetValue(r);
            stop = cache_ptr_ ? pointers.GetValue(r + 1) : ptr_gm_.GetValue(r + 1);
          }
          starts.SetValue(i, start); ends.SetValue(i, stop);
          const uint64_t degree = static_cast<uint64_t>(stop - start);
          if (degree && !(sorted_ && degree > kRow)) {
            Load(row_data[i * kRow], idx_gm_, start,
                 degree < kRow ? static_cast<uint32_t>(degree) : kRow);
          }
        }
        SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
        if constexpr (sizeof(IdT) == 4) {
          auto seed = col_seed_buf_.Get<int32_t>();
          if (col_stride_) {
            Brcb(seed, cols[batch], static_cast<uint8_t>((qn + 7) / 8),
                 BrcbRepeatParams{1, 8});
          } else {
            Duplicate<int32_t>(seed, cols.GetValue(0), kBatch * 8);
          }
          PipeBarrier<PIPE_V>();
          BinaryRepeatParams repeat;
          repeat.src0BlkStride = 1; repeat.src1BlkStride = 0;
          repeat.src0RepStride = kRow / 8; repeat.src1RepStride = 1;
          auto mask = batch_mask_buf_.Get<uint8_t>();
          Compare(mask, row_data, seed, CMPMODE::EQ, uint64_t(64),
                  static_cast<uint8_t>(qn), repeat);
          Compare(mask[kBatch * 8], row_data[64], seed, CMPMODE::EQ,
                  uint64_t(64), static_cast<uint8_t>(qn), repeat);
        } else {
          for (uint32_t i = 0; i < qn; ++i) {
            const IdT degree = ends.GetValue(i) - starts.GetValue(i);
            if (degree > 0 && !(sorted_ && degree > kRow)) {
              MakeMask(i, cols.GetValue(col_stride_ ? batch + i : 0),
                       degree < kRow ? degree : kRow);
            }
          }
        }
        SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
        Duplicate<int32_t>(eid_buf_.Get<int32_t>(), -1, kBatch * 32 / 4);
        if constexpr (Weighted) FillValues(value_slots);
        SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
        SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
        for (uint32_t i = 0; i < qn; ++i) {
          const IdT start = starts.GetValue(i), stop = ends.GetValue(i);
          const IdT c = stop - start > kRow
              ? cols.GetValue(col_stride_ ? batch + i : 0) : IdT(0);
          IdT found = -1;
          if (sorted_ && static_cast<uint64_t>(stop - start) > kRow) {
            // lower_bound preserves the first occurrence in a multigraph.
            IdT lo = start, hi = stop;
            while (lo < hi) {
              const IdT mid = lo + (hi - lo) / 2;
              if (idx_gm_.GetValue(mid) < c) lo = mid + 1;
              else hi = mid;
            }
            if (lo < stop && idx_gm_.GetValue(lo) == c) found = lo;
          } else {
            for (IdT tile = start; tile < stop; tile += kRow) {
              const uint32_t take = stop - tile < kRow ? stop - tile : kRow;
              if (tile != start) {
                SetFlag<HardEvent::S_MTE2>(0); WaitFlag<HardEvent::S_MTE2>(0);
                Load(row_data[i * kRow], idx_gm_, tile, take);
                SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
                SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
                MakeMask(i, c, take);
                SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
              }
              int32_t match;
              if constexpr (sizeof(IdT) == 4) {
                match = tile == start ? FirstBatchMatch(i, take) : FirstMatch(i, take);
              } else {
                match = FirstMatch(i, take);
              }
              if (match >= 0) found = tile + match;
              if (found >= 0) break;
            }
          }
          if (found >= 0) {
            if (has_data_) {
              Load(eid_slots[i * (32 / sizeof(IdT))], data_gm_, found, 1);
            } else {
              eid_slots.SetValue(i * (32 / sizeof(IdT)), found);
            }
          }
        }
        SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
        if constexpr (Weighted) {
          for (uint32_t i = 0; i < qn; ++i) {
            const IdT eid = eid_slots.GetValue(i * (32 / sizeof(IdT)));
            if (eid >= 0) {
              Load(value_slots[i * (32 / sizeof(OutT))], weights_gm_, eid, 1);
            }
          }
        }
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
        if constexpr (Weighted) {
          Gather(result.template ReinterpretCast<int32_t>()[batch * out_words],
                 value_slots.template ReinterpretCast<int32_t>(), offsets, 0, qn * out_words);
        } else {
          Gather(result.template ReinterpretCast<int32_t>()[batch * out_words],
                 eid_slots.template ReinterpretCast<int32_t>(), offsets, 0, qn * out_words);
        }
        SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
      }
      SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
      DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(OutT)), 0, 0, 0};
      DataCopyPad(out_gm_[base], result, cp);
      SetFlag<HardEvent::MTE3_S>(0); WaitFlag<HardEvent::MTE3_S>(0);
    }
  }
 private:
  template <typename T>
  __aicore__ inline void FillValues(LocalTensor<T> buffer) {
    constexpr uint32_t nwords = kBatch * 32 / 4;
    auto words = buffer.template ReinterpretCast<int32_t>();
    Duplicate<int32_t>(words, static_cast<int32_t>(filler_), nwords);
    if constexpr (sizeof(T) == 8) {
      PipeBarrier<PIPE_V>();
      uint64_t mask[1] = {0xAAAAAAAAAAAAAAAAULL};
      Duplicate<int32_t>(words,
          static_cast<int32_t>(static_cast<uint64_t>(filler_) >> 32),
          mask, nwords / 64, 1, 8);
    }
  }
  __aicore__ inline int32_t FirstBatchMatch(uint32_t slot, uint32_t take) {
    auto mask = batch_mask_buf_.Get<uint64_t>();
    uint64_t bits = mask.GetValue(slot);
    if (take < 64) bits &= (uint64_t(1) << take) - 1;
    if (bits) return static_cast<int32_t>(GetSFFValue<1>(bits));
    if (take > 64) {
      bits = mask.GetValue(kBatch + slot);
      if (take < 128) bits &= (uint64_t(1) << (take - 64)) - 1;
      if (bits) return 64 + static_cast<int32_t>(GetSFFValue<1>(bits));
    }
    return -1;
  }
  template <uint32_t Width>
  __aicore__ inline void CompareRow(uint32_t slot, IdT col) {
    constexpr uint32_t words = sizeof(IdT) / 4;
    auto input = row_buf_.Get<int32_t>();
    CompareScalar(mask_lo_buf_.Get<uint8_t>()[slot * 32],
                  input[slot * kRow * words], static_cast<int32_t>(col),
                  CMPMODE::EQ, Width * words);
    if constexpr (sizeof(IdT) == 8) {
      CompareScalar(mask_hi_buf_.Get<uint8_t>()[slot * 32],
                    input[slot * kRow * words],
                    static_cast<int32_t>(static_cast<uint64_t>(col) >> 32),
                    CMPMODE::EQ, Width * words);
    }
  }
  __aicore__ inline void MakeMask(uint32_t slot, IdT col, uint32_t take) {
    if (take <= 64) CompareRow<64>(slot, col);
    else CompareRow<128>(slot, col);
  }
  __aicore__ inline int32_t FirstMatch(uint32_t slot, uint32_t take) {
    constexpr uint32_t words = sizeof(IdT) / 4;
    constexpr uint32_t span = 64 / words;
    auto low = mask_lo_buf_.Get<uint64_t>();
    auto high = mask_hi_buf_.Get<uint64_t>();
    for (uint32_t group = 0; group * span < take; ++group) {
      uint64_t bits = low.GetValue(slot * 4 + group);
      if constexpr (sizeof(IdT) == 8) {
        bits &= (high.GetValue(slot * 4 + group) >> 1) & 0x5555555555555555ULL;
      }
      const uint32_t valid = take - group * span < span ? take - group * span : span;
      if (valid * words < 64) bits &= (uint64_t(1) << (valid * words)) - 1;
      if (bits) return group * span + static_cast<uint32_t>(GetSFFValue<1>(bits)) / words;
    }
    return -1;
  }
  GlobalTensor<IdT> ptr_gm_, idx_gm_, data_gm_, rows_gm_, cols_gm_;
  GlobalTensor<OutT> weights_gm_, out_gm_;
  TBuf<TPosition::VECCALC> ptr_buf_, rows_buf_, cols_buf_, out_buf_, row_buf_;
  TBuf<TPosition::VECCALC> start_buf_, end_buf_;
  TBuf<TPosition::VECCALC> mask_lo_buf_, mask_hi_buf_;
  TBuf<TPosition::VECCALC> col_seed_buf_, batch_mask_buf_;
  TBuf<TPosition::VECCALC> eid_buf_, value_buf_, field_offsets_buf_;
  uint64_t num_cols_, count_, begin_, end_;
  uint32_t num_rows_, row_stride_, col_stride_, has_data_, sorted_;
  bool cache_ptr_;
  OutT filler_;
};
}  // namespace

// Floating values are transferred as integer bits. This supports float64
// without unsupported FP64 arithmetic, preserving filler/weight bit patterns.
#define GET_DATA_ENTRY(NAME, ID, OUT, WEIGHTED)                               \
extern "C" __global__ __aicore__ void NAME(                                  \
    GM_ADDR ptr, GM_ADDR idx, GM_ADDR data, GM_ADDR rows, GM_ADDR cols,         \
    GM_ADDR weights, GM_ADDR out, uint32_t num_rows, uint64_t num_cols,        \
    uint64_t count, uint32_t row_stride, uint32_t col_stride,                  \
    uint32_t has_data, uint32_t sorted, uint64_t filler) {                     \
  TPipe pipe;                                                                \
  KernelLookup<ID, OUT, WEIGHTED> op;                                         \
  op.Init(ptr, idx, data, rows, cols, weights, out, num_rows, num_cols, count,   \
          row_stride, col_stride, has_data, sorted, filler, &pipe);            \
  op.Process();                                                              \
}
GET_DATA_ENTRY(csr_get_data_int32, int32_t, int32_t, false)
GET_DATA_ENTRY(csr_get_data_int64, int64_t, int64_t, false)
GET_DATA_ENTRY(csr_get_data_weighted_int32, int32_t, int32_t, true)
GET_DATA_ENTRY(csr_get_data_weighted_int64, int64_t, int32_t, true)
GET_DATA_ENTRY(csr_get_data_weighted_f64_int32, int32_t, int64_t, true)
GET_DATA_ENTRY(csr_get_data_weighted_f64_int64, int64_t, int64_t, true)
#undef GET_DATA_ENTRY
