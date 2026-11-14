#include "kernel_operator.h"

using namespace AscendC;

// Sparse graphs use row tiles to avoid per-core binary searches. Few-row and
// high-average-degree graphs use weighted row/edge partitions so a single
// high-degree row can span vector cores. All output ranges are disjoint.
// Row pointers are read in UB batches; GM upper_bounds also skip long empty
// stretches. All output stores use MTE3, including the unaligned final tail.
template <typename IdxT>
class KernelCsrToCOO {
 public:
  __aicore__ inline void Init(GM_ADDR indptr, GM_ADDR ret_row,
                             uint32_t num_rows, uint64_t nnz, TPipe* pipe) {
    indptr_p_ = reinterpret_cast<const __gm__ IdxT*>(indptr);
    indptr_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdxT*>(indptr));
    ret_row_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdxT*>(ret_row));
    num_rows_ = num_rows;
    nnz_ = nnz;
    pipe->InitBuffer(row_buf_, kFlushElems * sizeof(IdxT));
    pipe->InitBuffer(ptr_buf_, (kPtrRows + kAlignElems) * sizeof(IdxT));
  }

  __aicore__ inline void Process() {
    uint64_t out_off, end;
    uint32_t row;
    if (num_rows_ >= GetBlockNum() &&
        nnz_ <= static_cast<uint64_t>(num_rows_) * 128) {
      const uint32_t rows_per_core =
          (static_cast<uint64_t>(num_rows_) + GetBlockNum() - 1) / GetBlockNum();
      row = GetBlockIdx() * rows_per_core;
      if (row >= num_rows_) return;
      const uint32_t end_row =
          num_rows_ - row < rows_per_core ? num_rows_ : row + rows_per_core;
      out_off = static_cast<uint64_t>(indptr_p_[row]);
      end = static_cast<uint64_t>(indptr_p_[end_row]);
    } else {
      const uint64_t work = nnz_ + static_cast<uint64_t>(num_rows_) * kRowWeight;
      const uint64_t chunk = (work + GetBlockNum() - 1) / GetBlockNum();
      out_off = EdgeBoundary(GetBlockIdx() * chunk);
      end = EdgeBoundary((GetBlockIdx() + 1) * chunk);
      // upper_bound(indptr, out_off) - 1 also skips leading empty rows.
      uint32_t lo = 0;
      uint32_t hi = num_rows_;
      while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (static_cast<uint64_t>(indptr_p_[mid + 1]) <= out_off)
          lo = mid + 1;
        else
          hi = mid;
      }
      row = lo;
    }
    if (out_off >= end) return;
    uint32_t ptr_base = row;
    uint32_t ptr_count = LoadPointers(ptr_base);
    auto ptrs = ptr_buf_.Get<IdxT>();
    auto staging = row_buf_.Get<IdxT>();
    // On A2, fill int64 by writing its low int32 words with a vector mask.
    // Row IDs are uint32; the high words stay zero for every buffer reuse.
    if constexpr (sizeof(IdxT) == 8) {
      Duplicate<int32_t>(staging.template ReinterpretCast<int32_t>(), 0,
                         kFlushElems * 2);
    }

    uint64_t pos = out_off;
    uint32_t staged = 0;
    while (pos < end) {
      if (row + 1 - ptr_base >= ptr_count) {
        ptr_base = row;
        ptr_count = LoadPointers(ptr_base);
      }
      const uint64_t row_end =
          static_cast<uint64_t>(ptrs.GetValue(row + 1 - ptr_base));
      if (row_end <= pos) {
        // Skip empty rows in the resident pointer tile. If the whole tail is
        // empty, upper_bound in GM jumps directly to the next occupied row;
        // do not scan arbitrarily long zero-degree stretches one row at a time.
        uint32_t lo = row + 1 - ptr_base;
        uint32_t hi = ptr_count - 1;
        if (static_cast<uint64_t>(ptrs.GetValue(hi)) > pos) {
          while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2;
            if (static_cast<uint64_t>(ptrs.GetValue(mid)) <= pos)
              lo = mid + 1;
            else
              hi = mid;
          }
          row = ptr_base + lo - 1;
        } else {
          lo = ptr_base + ptr_count - 1;
          hi = num_rows_;
          while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2;
            if (static_cast<uint64_t>(indptr_p_[mid + 1]) <= pos)
              lo = mid + 1;
            else
              hi = mid;
          }
          row = lo;
          ptr_base = row;
          ptr_count = LoadPointers(ptr_base);
        }
        continue;
      }
      const uint64_t stop = row_end < end ? row_end : end;
      while (pos < stop) {
        const uint32_t space = kFlushElems - staged;
        const uint32_t take = stop - pos < space ? stop - pos : space;
        Fill(staging, staged, take, row);
        pos += take;
        staged += take;
        if (staged == kFlushElems) {
          Flush(staging, out_off, staged);
          out_off += staged;
          staged = 0;
        }
      }
      ++row;
    }
    if (staged) Flush(staging, out_off, staged);
  }

 private:
  __aicore__ inline uint64_t EdgeBoundary(uint64_t work) {
    if (work == 0) return 0;
    uint32_t lo = 0, hi = num_rows_;
    while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      const uint64_t next = static_cast<uint64_t>(indptr_p_[mid + 1]) +
                            (static_cast<uint64_t>(mid) + 1) * kRowWeight;
      if (next <= work) lo = mid + 1;
      else hi = mid;
    }
    if (lo == num_rows_) return nnz_;
    const uint64_t first = static_cast<uint64_t>(indptr_p_[lo]);
    const uint64_t edge_work = first +
                              (static_cast<uint64_t>(lo) + 1) * kRowWeight;
    const uint64_t edge = first + (work > edge_work ? work - edge_work : 0);
    const uint64_t aligned = (edge + kAlignElems - 1) / kAlignElems * kAlignElems;
    return aligned < nnz_ ? aligned : nnz_;
  }

  __aicore__ inline uint32_t LoadPointers(uint32_t base) {
    const uint32_t count = num_rows_ - base < kPtrRows
                               ? num_rows_ - base + 1 : kPtrRows + 1;
    DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(IdxT)),
                         0, 0, 0};
    DataCopyPadExtParams<IdxT> pad{false, 0, 0, 0};
    auto ptrs = ptr_buf_.Get<IdxT>();
    // Finish scalar reads before reusing the indptr buffer for the next DMA.
    SetFlag<HardEvent::S_MTE2>(0);
    WaitFlag<HardEvent::S_MTE2>(0);
    DataCopyPad(ptrs, indptr_gm_[base], cp, pad);
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    return count;
  }

  __aicore__ inline void Fill(const LocalTensor<IdxT>& staging,
                             uint32_t offset, uint32_t count, uint32_t row) {
    constexpr uint32_t kWords = sizeof(IdxT) / sizeof(int32_t);
    constexpr uint32_t kLanes = 64 / kWords;
    auto words = staging.template ReinterpretCast<int32_t>();
    while (count) {
      const uint32_t lane = offset % kLanes;
      const uint32_t take = count < kLanes - lane ? count : kLanes - lane;
      const uint32_t width = take * kWords;
      uint64_t bits = width == 64 ? ~uint64_t(0)
                                 : (uint64_t(1) << width) - 1;
      bits <<= lane * kWords;
      if constexpr (sizeof(IdxT) == 8) bits &= 0x5555555555555555ULL;
      uint64_t mask[1] = {bits};
      Duplicate<int32_t>(words[(offset - lane) * kWords],
                         static_cast<int32_t>(row), mask, 1, 1, 8);
      offset += take;
      count -= take;
    }
  }

  __aicore__ inline void Flush(const LocalTensor<IdxT>& staging,
                              uint64_t out_index, uint32_t count) {
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(IdxT)),
                         0, 0, 0};
    DataCopyPad(ret_row_gm_[out_index], staging, cp);
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
  }

  static constexpr uint32_t kFlushElems = 2048;
  static constexpr uint32_t kPtrRows = 256;
  static constexpr uint32_t kRowWeight = 16;
  static constexpr uint32_t kAlignElems = 32 / sizeof(IdxT);
  const __gm__ IdxT* indptr_p_;
  GlobalTensor<IdxT> indptr_gm_, ret_row_gm_;
  uint32_t num_rows_;
  uint64_t nnz_;
  TBuf<TPosition::VECCALC> row_buf_, ptr_buf_;
};

extern "C" __global__ __aicore__ void csr_to_coo_int32(
    GM_ADDR indptr, GM_ADDR ret_row, uint32_t num_rows, uint64_t nnz) {
  TPipe pipe;
  KernelCsrToCOO<int32_t> op;
  op.Init(indptr, ret_row, num_rows, nnz, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_to_coo_int64(
    GM_ADDR indptr, GM_ADDR ret_row, uint32_t num_rows, uint64_t nnz) {
  TPipe pipe;
  KernelCsrToCOO<int64_t> op;
  op.Init(indptr, ret_row, num_rows, nnz, &pipe);
  op.Process();
}

// Apply the CSR edge-ID permutation entirely on AIV cores:
//   out_row[order[e]] = row[e], out_col[order[e]] = col[e].
// Consecutive ascending/descending IDs are coalesced into DMA runs. Other
// permutations use exact-size DMA stores, without scalar GM writes. All run
// detection, reversal and scatter work happens on the NPU, including fallback
// singleton runs. CSR data is required to be a valid permutation, as on CUDA.
template <typename IdxT>
class KernelCsrToCOOOrder {
 public:
  __aicore__ inline void Init(GM_ADDR row, GM_ADDR col, GM_ADDR order,
                             GM_ADDR out_row, GM_ADDR out_col,
                             uint64_t nnz, TPipe* pipe) {
    row_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdxT*>(row));
    col_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdxT*>(col));
    order_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdxT*>(order));
    out_row_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdxT*>(out_row));
    out_col_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdxT*>(out_col));
    nnz_ = nnz;
    pipe->InitBuffer(row_buf_, kTile * sizeof(IdxT));
    pipe->InitBuffer(col_buf_, kTile * sizeof(IdxT));
    pipe->InitBuffer(order_buf_, kTile * sizeof(IdxT));
    pipe->InitBuffer(out_row_buf_, kTile * sizeof(IdxT));
    pipe->InitBuffer(out_col_buf_, kTile * sizeof(IdxT));
    pipe->InitBuffer(store_meta_buf_, kPendingStores * 2 * sizeof(IdxT));
  }

  __aicore__ inline void Process() {
    constexpr uint32_t kAlign = 32 / sizeof(IdxT);
    static_assert(kPendingStores * 32 <= kTile * sizeof(IdxT),
                  "Pending DMA slots must fit in each output buffer");
    const uint64_t chunk =
        ((nnz_ + GetBlockNum() - 1) / GetBlockNum() + kAlign - 1) /
        kAlign * kAlign;
    const uint64_t begin = GetBlockIdx() * chunk;
    const uint64_t end = begin + chunk < nnz_ ? begin + chunk : nnz_;
    auto rows = row_buf_.Get<IdxT>();
    auto cols = col_buf_.Get<IdxT>();
    auto ids = order_buf_.Get<IdxT>();
    auto out_rows = out_row_buf_.Get<IdxT>();
    auto out_cols = out_col_buf_.Get<IdxT>();
    for (uint64_t base = begin; base < end; base += kTile) {
      const uint32_t count = end - base < kTile ? end - base : kTile;
      DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(IdxT)),
                           0, 0, 0};
      DataCopyPadExtParams<IdxT> pad{false, 0, 0, 0};
      SetFlag<HardEvent::S_MTE2>(0);
      WaitFlag<HardEvent::S_MTE2>(0);
      DataCopyPad(rows, row_gm_[base], cp, pad);
      DataCopyPad(cols, col_gm_[base], cp, pad);
      DataCopyPad(ids, order_gm_[base], cp, pad);
      SetFlag<HardEvent::MTE2_S>(0);
      WaitFlag<HardEvent::MTE2_S>(0);

      uint32_t cursor = 0;
      while (cursor < count) {
        const IdxT first = ids.GetValue(cursor);
        int32_t direction = 0;
        uint32_t run = 1;
        if (cursor + 1 < count) {
          const IdxT next = ids.GetValue(cursor + 1);
          if (next == first + 1) direction = 1;
          else if (next == first - 1) direction = -1;
          if (direction != 0) {
            run = 2;
            while (cursor + run < count &&
                   ids.GetValue(cursor + run) ==
                       first + static_cast<IdxT>(direction) * run) {
              ++run;
            }
          }
        }
        if (direction == 1 && cursor % kAlign == 0) {
          Store(static_cast<uint64_t>(first), rows[cursor], cols[cursor], run);
        } else if (run <= kAlign) {
          // Give each pending short run its own aligned UB slot. The slots
          // remain immutable until all DMA reads complete, allowing one pair
          // of S/MTE3 barriers per batch rather than per random edge.
          const uint32_t slot = pending_ * kAlign;
          for (uint32_t i = 0; i < run; ++i) {
            const uint32_t src = direction == -1
                                     ? cursor + run - 1 - i : cursor + i;
            out_rows.SetValue(slot + i, rows.GetValue(src));
            out_cols.SetValue(slot + i, cols.GetValue(src));
          }
          auto metadata = store_meta_buf_.Get<IdxT>();
          const IdxT target = direction == -1 ? first - (run - 1) : first;
          metadata.SetValue(pending_ * 2, target);
          metadata.SetValue(pending_ * 2 + 1, static_cast<IdxT>(run));
          if (++pending_ == kPendingStores) FlushPending();
        } else {
          FlushPending();
          for (uint32_t i = 0; i < run; ++i) {
            const uint32_t src = direction == -1
                                     ? cursor + run - 1 - i : cursor + i;
            out_rows.SetValue(i, rows.GetValue(src));
            out_cols.SetValue(i, cols.GetValue(src));
          }
          const IdxT target = direction == -1 ? first - (run - 1) : first;
          Store(static_cast<uint64_t>(target), out_rows, out_cols, run);
        }
        cursor += run;
      }
      FlushPending();
    }
  }

 private:
  __aicore__ inline void FlushPending() {
    if (pending_ == 0) return;
    constexpr uint32_t kAlign = 32 / sizeof(IdxT);
    auto rows = out_row_buf_.Get<IdxT>();
    auto cols = out_col_buf_.Get<IdxT>();
    auto metadata = store_meta_buf_.Get<IdxT>();
    SetFlag<HardEvent::S_MTE3>(0);
    WaitFlag<HardEvent::S_MTE3>(0);
    for (uint32_t i = 0; i < pending_; ++i) {
      const uint64_t offset = static_cast<uint64_t>(metadata.GetValue(i * 2));
      const uint32_t count = static_cast<uint32_t>(metadata.GetValue(i * 2 + 1));
      DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(IdxT)),
                           0, 0, 0};
      DataCopyPad(out_row_gm_[offset], rows[i * kAlign], cp);
      DataCopyPad(out_col_gm_[offset], cols[i * kAlign], cp);
    }
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
    pending_ = 0;
  }

  __aicore__ inline void Store(uint64_t offset,
                              const LocalTensor<IdxT>& rows,
                              const LocalTensor<IdxT>& cols, uint32_t count) {
    SetFlag<HardEvent::S_MTE3>(0);
    WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(IdxT)),
                         0, 0, 0};
    DataCopyPad(out_row_gm_[offset], rows, cp);
    DataCopyPad(out_col_gm_[offset], cols, cp);
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
  }

  static constexpr uint32_t kTile = 2048;
  static constexpr uint32_t kPendingStores = 64;
  uint64_t nnz_;
  uint32_t pending_ = 0;
  GlobalTensor<IdxT> row_gm_, col_gm_, order_gm_, out_row_gm_, out_col_gm_;
  TBuf<TPosition::VECCALC> row_buf_, col_buf_, order_buf_;
  TBuf<TPosition::VECCALC> out_row_buf_, out_col_buf_;
  TBuf<TPosition::VECCALC> store_meta_buf_;
};

extern "C" __global__ __aicore__ void csr_to_coo_order_int32(
    GM_ADDR row, GM_ADDR col, GM_ADDR order, GM_ADDR out_row, GM_ADDR out_col,
    uint64_t nnz) {
  TPipe pipe;
  KernelCsrToCOOOrder<int32_t> op;
  op.Init(row, col, order, out_row, out_col, nnz, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_to_coo_order_int64(
    GM_ADDR row, GM_ADDR col, GM_ADDR order, GM_ADDR out_row, GM_ADDR out_col,
    uint64_t nnz) {
  TPipe pipe;
  KernelCsrToCOOOrder<int64_t> op;
  op.Init(row, col, order, out_row, out_col, nnz, &pipe);
  op.Process();
}
