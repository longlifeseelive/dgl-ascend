#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;

// CSR -> COO row array.
//
// ret_row[pos] = row for every nonzero `pos` in [indptr[row], indptr[row+1]).
// Rows are partitioned across blocks; each block writes the contiguous run of
// output positions belonging to its rows.
//
// IMPORTANT: outputs are staged in a UB LocalTensor and flushed with
// DataCopyPad. Writing results with GlobalTensor::SetValue (per-word GM
// stores) is NOT reliable on this stack — writes from non-zero blocks are
// silently dropped (same failure documented in csr_get_row_nnz_kernel.cpp,
// and banned by the AscendC best-practices: GM access must go through
// DataCopy/DataCopyPad, never scalar SetValue).
template <typename IdxT>
class KernelCsrToCOO {
 public:
  __aicore__ inline KernelCsrToCOO() {}

  __aicore__ inline void Init(GM_ADDR indptr, GM_ADDR ret_row,
                              uint32_t num_rows, uint32_t nnz, TPipe* pipe) {
    indptr_p_ = (const __gm__ IdxT*)indptr;
    ret_row_gm_.SetGlobalBuffer((__gm__ IdxT*)ret_row, nnz);
    num_rows_ = num_rows;
    nnz_ = nnz;
    pipe->InitBuffer(row_buf_, kFlushElems * sizeof(IdxT));
  }

  __aicore__ inline void Process() {
    const uint32_t block_id = GetBlockIdx();
    const uint32_t block_num = GetBlockNum();
    const uint32_t chunk = (num_rows_ + block_num - 1) / block_num;
    const uint32_t start_row = block_id * chunk;
    const uint32_t end_row =
        (start_row + chunk > num_rows_) ? num_rows_ : start_row + chunk;
    if (start_row >= end_row) return;

    // Output positions for this block's rows form a contiguous run.
    uint32_t out_off = static_cast<uint32_t>(indptr_p_[start_row]);
    LocalTensor<IdxT> staging = row_buf_.Get<IdxT>();
    uint32_t staged = 0;
    for (uint32_t row = start_row; row < end_row; row++) {
      const IdxT seg_start = indptr_p_[row];
      const IdxT seg_end = indptr_p_[row + 1];

      uint32_t remaining = static_cast<uint32_t>(seg_end - seg_start);

      while(remaining > 0) {
        const uint32_t space = kFlushElems - staged;
        const uint32_t take = remaining < space ? remaining : space;

        if constexpr (std::is_same<IdxT, int32_t>::value){
          // int32: vectorized fill
          FillVectorInt32(staging, staged, take, static_cast<int32_t>(row));
        } else {
          // int64: keep origin scalar path
          for (uint32_t i = 0; i < take; ++i) {
            staging.SetValue(staged + i, static_cast<IdxT>(row));
          }
        }

        staged += take;
        remaining -= take;

        if (staged == kFlushElems) {
          FlushOut(staging, out_off, staged);
          out_off += staged;
          staged = 0;
        }
      }
    }
    if (staged > 0) FlushOut(staging, out_off, staged);
  }

 private:
  __aicore__ inline void FlushOut(LocalTensor<IdxT>& staging,
                                  uint32_t out_index, uint32_t count) {

    if constexpr (std::is_same<IdxT, int32_t>::value) {
      TQueSync<PIPE_V, PIPE_MTE3> sync;
      sync.SetFlag(0);
      sync.WaitFlag(0);
    }


    DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(IdxT)), 0, 0,
                         0};
    DataCopyPad(ret_row_gm_[out_index], staging, cp);

    if constexpr (std::is_same<IdxT, int32_t>::value) {
      TQueSync<PIPE_MTE3, PIPE_V> sync;
      sync.SetFlag(0);
      sync.WaitFlag(0);
    }
  }

  static constexpr uint32_t kFlushElems = 1024;

  const __gm__ IdxT* indptr_p_ = nullptr;
  GlobalTensor<IdxT> ret_row_gm_;
  uint32_t num_rows_;
  uint32_t nnz_;
  TBuf<TPosition::VECCALC> row_buf_;

  __aicore__ inline void FillVectorInt32(const LocalTensor<int32_t>& staging, uint32_t offset, uint32_t count, int32_t row) {
    constexpr uint32_t kVecElems = 64;

    while (count > 0) {
      const uint32_t aligned = offset & ~63U;
      const uint32_t lane = offset - aligned;
      const uint32_t capacity = kVecElems - lane;
      const uint32_t n = count < capacity ? count : capacity;
      uint64_t bits;

      if (n == 64 ){
        bits = ~uint64_t(0);
      } else {
        bits = ((uint64_t(1) << n) - 1ULL) << lane;
      }

      uint64_t mask[1] = {bits};

      Duplicate<int32_t>(staging[aligned], row, mask, 1, 1, 8);

      offset += n;
      count -= n;
    }
  }
};

extern "C" __global__ __aicore__ void csr_to_coo_int32(
    GM_ADDR indptr, GM_ADDR ret_row, GM_ADDR tiling_ptr) {
  // Struct-pointer tiling access: per-word GetValue on a shared GlobalTensor
  // is not consistently visible across blocks.
  const __gm__ uint32_t* tiling = (const __gm__ uint32_t*)tiling_ptr;
  const uint32_t num_rows = tiling[0];
  const uint32_t nnz = tiling[1];
  KernelCsrToCOO<int32_t> op;
  TPipe pipe;
  op.Init(indptr, ret_row, num_rows, nnz, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_to_coo_int64(
    GM_ADDR indptr, GM_ADDR ret_row, GM_ADDR tiling_ptr) {
  const __gm__ uint32_t* tiling = (const __gm__ uint32_t*)tiling_ptr;
  const uint32_t num_rows = tiling[0];
  const uint32_t nnz = tiling[1];
  KernelCsrToCOO<int64_t> op;
  TPipe pipe;
  op.Init(indptr, ret_row, num_rows, nnz, &pipe);
  op.Process();
}
