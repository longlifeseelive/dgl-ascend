#include "kernel_operator.h"
using namespace AscendC;

namespace {
constexpr uint32_t kTile = 1024;
constexpr uint32_t kBatch = 64;

// Every GM store is an exact-length DMA. In particular, independent cores may
// own different elements in the same 32-byte block: scalar GM writes or padded
// GM stores would lose updates at such boundaries.
template <typename T>
__aicore__ inline void Load(LocalTensor<T> dst, GlobalTensor<T> src,
                            uint64_t offset, uint32_t n) {
  DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> pad{false, 0, 0, 0};
  DataCopyPad(dst, src[offset], cp, pad);
}
template <typename T>
__aicore__ inline void Store(GlobalTensor<T> dst, uint64_t offset,
                             LocalTensor<T> src, uint32_t n) {
  DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
  DataCopyPad(dst[offset], src, cp);
}
__aicore__ inline uint64_t EdgeChunk(uint64_t nnz) {
  // Identical partitioning for count and scatter; 32-byte alignment for int64.
  return ((nnz + GetBlockNum() - 1) / GetBlockNum() + 7) / 8 * 8;
}

template <typename IdT, typename CountT>
__aicore__ inline void Count(GM_ADDR indices, GM_ADDR counts, uint64_t nnz,
                            uint64_t first, uint32_t cols) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> hist_buf, in_buf;
  const uint32_t stride = (cols + 7) / 8 * 8;
  pipe.InitBuffer(hist_buf, stride * sizeof(CountT));
  pipe.InitBuffer(in_buf, kTile * sizeof(IdT));
  auto hist = hist_buf.Get<CountT>();
  auto input = in_buf.Get<IdT>();
  GlobalTensor<IdT> idx;
  GlobalTensor<CountT> out;
  idx.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(indices));
  out.SetGlobalBuffer(reinterpret_cast<__gm__ CountT*>(counts));
  Duplicate<int32_t>(hist.template ReinterpretCast<int32_t>(), 0,
                     stride * sizeof(CountT) / 4);
  PipeBarrier<PIPE_ALL>();
  const uint64_t chunk = EdgeChunk(nnz);
  const uint64_t begin = chunk * GetBlockIdx();
  const uint64_t end = begin + chunk < nnz ? begin + chunk : nnz;
  for (uint64_t base = begin; base < end; base += kTile) {
    const uint32_t n = end - base < kTile ? end - base : kTile;
    SetFlag<HardEvent::S_MTE2>(0);
    WaitFlag<HardEvent::S_MTE2>(0);
    Load(input, idx, base, n);
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    for (uint32_t i = 0; i < n; ++i) {
      const uint64_t col = static_cast<uint64_t>(input.GetValue(i));
      if (col >= first && col - first < cols) {
        const uint32_t c = col - first;
        hist.SetValue(c, hist.GetValue(c) + 1);
      }
    }
  }
  SetFlag<HardEvent::S_MTE3>(0);
  WaitFlag<HardEvent::S_MTE3>(0);
  Store(out, GetBlockIdx() * stride, hist, stride);
  SetFlag<HardEvent::MTE3_S>(0);
  WaitFlag<HardEvent::MTE3_S>(0);
}

// One vector core reduces histograms and forms both the CSR prefix and
// per-shard scatter starts. Counts, prefix and the inter-band carry stay on NPU.
template <typename IdT, typename CountT>
__aicore__ inline void Prefix(GM_ADDR counts, GM_ADDR offsets, GM_ADDR outptr,
                             GM_ADDR carry, uint32_t blocks, uint64_t first,
                             uint32_t cols) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> sum_buf, tmp_buf, ptr_buf, carry_buf;
  const uint32_t stride = (cols + 7) / 8 * 8;
  pipe.InitBuffer(sum_buf, stride * sizeof(CountT));
  pipe.InitBuffer(tmp_buf, stride * sizeof(CountT));
  pipe.InitBuffer(ptr_buf, kTile * sizeof(IdT));
  pipe.InitBuffer(carry_buf, 8 * sizeof(CountT));
  auto sum = sum_buf.Get<CountT>();
  auto tmp = tmp_buf.Get<CountT>();
  auto ptr = ptr_buf.Get<IdT>();
  auto state = carry_buf.Get<CountT>();
  GlobalTensor<CountT> in, starts, state_gm;
  GlobalTensor<IdT> out;
  in.SetGlobalBuffer(reinterpret_cast<__gm__ CountT*>(counts));
  starts.SetGlobalBuffer(reinterpret_cast<__gm__ CountT*>(offsets));
  state_gm.SetGlobalBuffer(reinterpret_cast<__gm__ CountT*>(carry));
  out.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(outptr));
  Duplicate<int32_t>(sum.template ReinterpretCast<int32_t>(), 0,
                     stride * sizeof(CountT) / 4);
  PipeBarrier<PIPE_ALL>();
  for (uint32_t b = 0; b < blocks; ++b) {
    Load(tmp, in, b * stride, stride);
    PipeBarrier<PIPE_ALL>();
    if constexpr (sizeof(CountT) == 4) {
      Add<int32_t>(sum, sum, tmp, stride);
    } else {
      // A2 has no vector int64 Add. Scalar integer arithmetic remains on NPU
      // and retains the high word of large counts and absolute positions.
      for (uint32_t c = 0; c < cols; ++c)
        sum.SetValue(c, sum.GetValue(c) + tmp.GetValue(c));
    }
    PipeBarrier<PIPE_ALL>();
  }
  CountT position = 0;
  if (first) {
    Load(state, state_gm, 0, 8);
    PipeBarrier<PIPE_ALL>();
    position = state.GetValue(0);
  }
  uint32_t staged = 0;
  uint64_t out_base = first;
  for (uint32_t c = 0; c < cols; ++c) {
    const CountT count = sum.GetValue(c);
    sum.SetValue(c, position);
    ptr.SetValue(staged++, static_cast<IdT>(position));
    position += count;
    if (staged == kTile) {
      PipeBarrier<PIPE_ALL>();
      Store(out, out_base, ptr, staged);
      PipeBarrier<PIPE_ALL>();
      out_base += staged;
      staged = 0;
    }
  }
  ptr.SetValue(staged++, static_cast<IdT>(position));
  PipeBarrier<PIPE_ALL>();
  Store(out, out_base, ptr, staged);
  PipeBarrier<PIPE_ALL>();
  state.SetValue(0, position);
  PipeBarrier<PIPE_ALL>();
  Store(state_gm, 0, state, 8);
  PipeBarrier<PIPE_ALL>();
  // Each core gets the preceding cores' counts plus the column prefix.
  for (uint32_t b = 0; b < blocks; ++b) {
    Store(starts, b * stride, sum, stride);
    Load(tmp, in, b * stride, stride);
    PipeBarrier<PIPE_ALL>();
    if constexpr (sizeof(CountT) == 4) {
      Add<int32_t>(sum, sum, tmp, stride);
    } else {
      // A2 has no vector int64 Add. Scalar integer arithmetic remains on NPU
      // and retains the high word of large counts and absolute positions.
      for (uint32_t c = 0; c < cols; ++c)
        sum.SetValue(c, sum.GetValue(c) + tmp.GetValue(c));
    }
    PipeBarrier<PIPE_ALL>();
  }
}

template <typename IdT, typename CountT>
class KernelTransposeScatter {
 public:
  __aicore__ inline void Init(GM_ADDR indptr, GM_ADDR indices, GM_ADDR data,
                             GM_ADDR offsets, GM_ADDR outidx, GM_ADDR outdata,
                             uint64_t rows, uint64_t nnz, uint64_t first,
                             uint32_t cols, uint32_t has_data, TPipe* pipe) {
    ptr_p_ = reinterpret_cast<const __gm__ IdT*>(indptr);
    ptr_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(indptr));
    idx_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(indices));
    data_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(data));
    off_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ CountT*>(offsets));
    outidx_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(outidx));
    outdata_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(outdata));
    rows_ = rows; nnz_ = nnz; first_ = first; cols_ = cols; has_data_ = has_data;
    stride_ = (cols + 7) / 8 * 8;
    pipe->InitBuffer(off_buf_, stride_ * sizeof(CountT));
    pipe->InitBuffer(idx_buf_, kTile * sizeof(IdT));
    pipe->InitBuffer(data_buf_, kTile * sizeof(IdT));
    pipe->InitBuffer(ptr_buf_, (kPtrRows + 8) * sizeof(IdT));
    pipe->InitBuffer(pad_row_buf_, kBatch * 32);
    pipe->InitBuffer(pad_data_buf_, kBatch * 32);
    pipe->InitBuffer(row_buf_, kBatch * sizeof(IdT));
    pipe->InitBuffer(value_buf_, kBatch * sizeof(IdT));
    pipe->InitBuffer(target_buf_, kBatch * sizeof(CountT));
  }
  __aicore__ inline void Process() {
    const uint64_t chunk = EdgeChunk(nnz_);
    const uint64_t begin = chunk * GetBlockIdx();
    const uint64_t end = begin + chunk < nnz_ ? begin + chunk : nnz_;
    if (begin >= end) return;
    auto offsets = off_buf_.Get<CountT>();
    auto idx = idx_buf_.Get<IdT>();
    auto data = data_buf_.Get<IdT>();
    auto ptrs = ptr_buf_.Get<IdT>();
    auto padded_rows = pad_row_buf_.Get<IdT>();
    auto padded_data = pad_data_buf_.Get<IdT>();
    auto rows = row_buf_.Get<IdT>();
    auto values = value_buf_.Get<IdT>();
    auto targets = target_buf_.Get<CountT>();
    Load(offsets, off_gm_, GetBlockIdx() * stride_, stride_);
    PipeBarrier<PIPE_ALL>();
    uint64_t row = FindRow(begin), ptr_base = row;
    uint32_t ptr_count = LoadPointers(row);
    uint64_t row_end = static_cast<uint64_t>(ptrs.GetValue(1));
    uint32_t pending = 0;
    for (uint64_t base = begin; base < end; base += kTile) {
      const uint32_t n = end - base < kTile ? end - base : kTile;
      SetFlag<HardEvent::S_MTE2>(0);
      WaitFlag<HardEvent::S_MTE2>(0);
      Load(idx, idx_gm_, base, n);
      if (has_data_) Load(data, data_gm_, base, n);
      SetFlag<HardEvent::MTE2_S>(0);
      WaitFlag<HardEvent::MTE2_S>(0);
      for (uint32_t i = 0; i < n; ++i) {
        const uint64_t edge = base + i;
        if (edge >= row_end) {
          // Resident pointer tiles handle short rows. GM upper_bound skips
          // arbitrarily long empty row ranges without a linear scan.
          if (static_cast<uint64_t>(ptrs.GetValue(ptr_count - 1)) <= edge) {
            row = FindRow(edge);
            ptr_base = row;
            ptr_count = LoadPointers(row);
          } else {
            uint32_t lo = static_cast<uint32_t>(row + 1 - ptr_base), hi = ptr_count - 1;
            while (lo < hi) {
              const uint32_t mid = lo + (hi - lo) / 2;
              if (static_cast<uint64_t>(ptrs.GetValue(mid)) <= edge) lo = mid + 1;
              else hi = mid;
            }
            row = ptr_base + lo - 1;
          }
          row_end = static_cast<uint64_t>(ptrs.GetValue(static_cast<uint32_t>(row + 1 - ptr_base)));
        }
        const uint64_t col = static_cast<uint64_t>(idx.GetValue(i));
        if (col < first_ || col - first_ >= cols_) continue;
        const uint32_t c = col - first_;
        const CountT pos = offsets.GetValue(c);
        offsets.SetValue(c, pos + 1);
        const IdT value = has_data_ ? data.GetValue(i) : static_cast<IdT>(edge);
        targets.SetValue(pending, pos);
        rows.SetValue(pending, static_cast<IdT>(row));
        values.SetValue(pending, value);
        padded_rows.SetValue(pending * kAlign, static_cast<IdT>(row));
        padded_data.SetValue(pending * kAlign, value);
        if (++pending == kBatch) {
          Flush(pending);
          pending = 0;
        }
      }
    }
    if (pending) Flush(pending);
  }
 private:
  __aicore__ inline uint64_t FindRow(uint64_t edge) {
    uint64_t lo = 0, hi = rows_;
    while (lo < hi) {
      const uint64_t mid = lo + (hi - lo) / 2;
      if (static_cast<uint64_t>(ptr_p_[mid + 1]) <= edge) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
  __aicore__ inline uint32_t LoadPointers(uint64_t base) {
    const uint32_t count = rows_ - base < kPtrRows ? rows_ - base + 1 : kPtrRows + 1;
    SetFlag<HardEvent::S_MTE2>(0);
    WaitFlag<HardEvent::S_MTE2>(0);
    Load(ptr_buf_.Get<IdT>(), ptr_gm_, base, count);
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    return count;
  }
  __aicore__ inline void Flush(uint32_t n) {
    auto targets = target_buf_.Get<CountT>();
    const CountT first = targets.GetValue(0);
    bool consecutive = true;
    for (uint32_t i = 1; i < n; ++i) {
      if (targets.GetValue(i) != first + static_cast<CountT>(i)) {
        consecutive = false;
        break;
      }
    }
    SetFlag<HardEvent::S_MTE3>(0);
    WaitFlag<HardEvent::S_MTE3>(0);
    if (consecutive) {
      Store(outidx_gm_, first, row_buf_.Get<IdT>(), n);
      Store(outdata_gm_, first, value_buf_.Get<IdT>(), n);
    } else {
      auto rows = pad_row_buf_.Get<IdT>();
      auto data = pad_data_buf_.Get<IdT>();
      for (uint32_t i = 0; i < n; ++i) {
        const uint64_t pos = static_cast<uint64_t>(targets.GetValue(i));
        Store(outidx_gm_, pos, rows[i * kAlign], 1);
        Store(outdata_gm_, pos, data[i * kAlign], 1);
      }
    }
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
  }
  static constexpr uint32_t kPtrRows = 256;
  static constexpr uint32_t kAlign = 32 / sizeof(IdT);
  const __gm__ IdT* ptr_p_;
  GlobalTensor<IdT> ptr_gm_, idx_gm_, data_gm_, outidx_gm_, outdata_gm_;
  GlobalTensor<CountT> off_gm_;
  uint32_t cols_, has_data_, stride_;
  uint64_t rows_, first_, nnz_;
  TBuf<TPosition::VECCALC> off_buf_, idx_buf_, data_buf_, ptr_buf_;
  TBuf<TPosition::VECCALC> pad_row_buf_, pad_data_buf_, row_buf_, value_buf_, target_buf_;
};

template <typename IdT>
__aicore__ inline void Zero(GM_ADDR outptr, uint64_t size) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> buf;
  pipe.InitBuffer(buf, kTile * sizeof(IdT));
  auto staging = buf.Get<IdT>();
  Duplicate<int32_t>(staging.template ReinterpretCast<int32_t>(), 0,
                     kTile * sizeof(IdT) / 4);
  PipeBarrier<PIPE_ALL>();
  GlobalTensor<IdT> out;
  out.SetGlobalBuffer(reinterpret_cast<__gm__ IdT*>(outptr));
  const uint64_t chunk = ((size + GetBlockNum() - 1) / GetBlockNum() + 7) / 8 * 8;
  const uint64_t begin = GetBlockIdx() * chunk;
  const uint64_t end = begin + chunk < size ? begin + chunk : size;
  for (uint64_t base = begin; base < end; base += kTile) {
    const uint32_t n = end - base < kTile ? end - base : kTile;
    Store(out, base, staging, n);
  }
  PipeBarrier<PIPE_ALL>();
}
}  // namespace

#define TRANSPOSE_KERNELS(SUFFIX, ID_T, COUNT_T)                               \
extern "C" __global__ __aicore__ void csr_transpose_count_##SUFFIX(            \
    GM_ADDR idx, GM_ADDR counts, uint64_t nnz, uint64_t first, uint32_t cols) { \
  Count<ID_T, COUNT_T>(idx, counts, nnz, first, cols);                         \
}                                                                              \
extern "C" __global__ __aicore__ void csr_transpose_prefix_##SUFFIX(           \
    GM_ADDR counts, GM_ADDR offsets, GM_ADDR outptr, GM_ADDR carry,            \
    uint32_t blocks, uint64_t first, uint32_t cols) {                          \
  Prefix<ID_T, COUNT_T>(counts, offsets, outptr, carry, blocks, first, cols);  \
}                                                                              \
extern "C" __global__ __aicore__ void csr_transpose_scatter_##SUFFIX(          \
    GM_ADDR ptr, GM_ADDR idx, GM_ADDR data, GM_ADDR offsets, GM_ADDR outidx,   \
    GM_ADDR outdata, uint64_t rows, uint64_t nnz, uint64_t first,              \
    uint32_t cols, uint32_t has_data) {                                        \
  TPipe pipe;                                                                  \
  KernelTransposeScatter<ID_T, COUNT_T> op;                                    \
  op.Init(ptr, idx, data, offsets, outidx, outdata, rows, nnz, first, cols,    \
          has_data, &pipe);                                                    \
  op.Process();                                                                \
}                                                                              \
extern "C" __global__ __aicore__ void csr_transpose_zero_##SUFFIX(             \
    GM_ADDR outptr, uint64_t size) { Zero<ID_T>(outptr, size); }
TRANSPOSE_KERNELS(int32, int32_t, int32_t)
TRANSPOSE_KERNELS(int64, int64_t, int32_t)
TRANSPOSE_KERNELS(int64_wide, int64_t, int64_t)
#undef TRANSPOSE_KERNELS
