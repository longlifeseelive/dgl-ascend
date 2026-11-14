#ifdef DGL_USE_PYTORCH_NPU_STREAM
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#endif

#include <cstdint>
#include <memory>
#include <limits>
#include <tuple>
#include <type_traits>

#include <dgl/array.h>
#include "../array_op.h"
#ifdef DGL_USE_PYTORCH_NPU_STREAM
#include "../../graph/heterograph.h"
#endif

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
// torch_npu can include its older bundled ACL header first.
extern "C" aclError aclrtGetDeviceInfo(uint32_t, aclrtDevAttr, int64_t*);
#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e;              \
  }

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_csr_to_coo_int32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* ret_row, uint32_t num_rows, uint64_t nnz);

extern "C" uint32_t aclrtlaunch_csr_to_coo_int64(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* ret_row, uint32_t num_rows, uint64_t nnz);

extern "C" uint32_t aclrtlaunch_csr_to_coo_order_int32(
    uint32_t blockDim, aclrtStream stream, void* row, void* col, void* order,
    void* out_row, void* out_col, uint64_t nnz);
extern "C" uint32_t aclrtlaunch_csr_to_coo_order_int64(
    uint32_t blockDim, aclrtStream stream, void* row, void* col, void* order,
    void* out_row, void* out_col, uint64_t nnz);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}
}
#endif

namespace dgl {
namespace aten {
namespace impl {

namespace {
NDArray NewCOOArray(int64_t nnz, DGLDataType dtype, DGLContext ctx) {
#ifdef DGL_USE_PYTORCH_NPU_STREAM
  // Use the framework's caching allocator, retaining ownership until the last
  // DGL/PyTorch view is released. The kernel synchronizes before returning.
  // Preserve the 512-byte offset used by AscendDeviceAPI on DAV_2201.
  const auto device = c10::Device(c10::DeviceType::PrivateUse1, ctx.device_id);
  auto* allocator = c10_npu::NPUCachingAllocator::get();
  if (!allocator->initialized()) {
    // Also support callers that have only initialized ACL, not torch_npu.
    at::empty({0}, at::TensorOptions().device(device).dtype(at::kByte));
  }
  const c10_npu::NPUGuard guard(device);
  auto owner = std::make_unique<c10::DataPtr>(
      allocator->allocate(static_cast<size_t>(nnz) * (dtype.bits / 8) + 512));
  auto container = std::make_unique<NDArray::Container>();
  container->dl_tensor.data = static_cast<char*>(owner->get()) + 512;
  container->dl_tensor.ctx = ctx;
  container->dl_tensor.dtype = dtype;
  container->dl_tensor.ndim = 1;
  container->dl_tensor.shape = new int64_t[1]{nnz};
  container->manager_ctx = owner.release();
  container->deleter = [](NDArray::Container* self) {
    delete static_cast<c10::DataPtr*>(self->manager_ctx);
    delete[] self->dl_tensor.shape;
    delete self;
  };
  return NDArray(container.release());
#else
  return NDArray::Empty({nnz}, dtype, ctx);
#endif
}

#ifdef DGL_USE_ASCEND
uint32_t CsrBlockDim(DGLContext ctx, int64_t nnz) {
  // Partition edges across vector cores, including graphs with very few rows.
  static thread_local int cached_device = -1;
  static thread_local uint32_t cached_cores = 40;
  if (cached_device != ctx.device_id) {
    cached_cores = 40;
    int64_t core_num = 0;
    if (aclrtGetDeviceInfo(
            ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &core_num) ==
            ACL_SUCCESS &&
        core_num > 0 && core_num <= 4096) {
      cached_cores = static_cast<uint32_t>(core_num);
    }
    cached_device = ctx.device_id;
  }
  uint32_t block_dim = cached_cores;
  const uint64_t work_blocks = (static_cast<uint64_t>(nnz) + 255) / 256;
  if (block_dim > work_blocks) block_dim = static_cast<uint32_t>(work_blocks);

  return block_dim;
}

aclrtStream CsrStream(DGLContext ctx) {
  int32_t current_device = -1;
  ASCEND_CALL(aclrtGetDevice(&current_device));
  if (current_device != ctx.device_id) {
    ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  }
  return dgl::runtime::getCurrentAscendStream();
}

template <typename IdType>
void LaunchCsrRows(const CSRMatrix& csr, const NDArray& row,
                   aclrtStream stream, uint32_t block_dim) {
  CHECK_LE(csr.indptr->shape[0] - 1, static_cast<int64_t>(UINT32_MAX));
  const auto num_rows = static_cast<uint32_t>(csr.indptr->shape[0] - 1);
  const auto nnz = static_cast<uint64_t>(csr.indices->shape[0]);
  if (std::is_same<IdType, int32_t>::value) {
    ASCEND_CALL(aclrtlaunch_csr_to_coo_int32(
        block_dim, stream, csr.indptr->data, row->data, num_rows, nnz));
  } else {
    ASCEND_CALL(aclrtlaunch_csr_to_coo_int64(
        block_dim, stream, csr.indptr->data, row->data, num_rows, nnz));
  }
}
#endif
}  // namespace

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix csr) {
#ifdef DGL_USE_ASCEND
#ifdef DGL_USE_PYTORCH_NPU_STREAM
  RECORD_FUNCTION("dgl::CSRToCOO[Ascend]", std::vector<c10::IValue>());
#endif
  auto ctx = csr.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend)
      << "Expected Ascend device context for CSRToCOO";

  const int64_t nnz = csr.indices->shape[0];
  NDArray ret_row = NewCOOArray(nnz, csr.indices->dtype, ctx);
  if (nnz == 0) {
    // col and data are zero-copy views of csr.indices and csr.data
    return COOMatrix(csr.num_rows, csr.num_cols, ret_row, csr.indices,
                     csr.data, true, csr.sorted);
  }

  auto stream = CsrStream(ctx);

  CHECK_LE(csr.indptr->shape[0] - 1, static_cast<int64_t>(UINT32_MAX));
  const uint32_t num_rows =
      static_cast<uint32_t>(csr.indptr->shape[0] - 1);
  if (num_rows == 0) {
    return COOMatrix(csr.num_rows, csr.num_cols, ret_row, csr.indices,
                     csr.data, true, csr.sorted);
  }

  const uint32_t block_dim = CsrBlockDim(ctx, nnz);

  LaunchCsrRows<IdType>(csr, ret_row, stream, block_dim);

  ASCEND_CALL(aclrtSynchronizeStream(stream));

  // col and data are zero-copy views of csr.indices and csr.data
  return COOMatrix(csr.num_rows, csr.num_cols, ret_row, csr.indices,
                   csr.data, true, csr.sorted);
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

template COOMatrix CSRToCOO<kDGLAscend, int32_t>(CSRMatrix);
template COOMatrix CSRToCOO<kDGLAscend, int64_t>(CSRMatrix);


template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOODataAsOrder(CSRMatrix csr) {
#ifdef DGL_USE_ASCEND
#ifdef DGL_USE_PYTORCH_NPU_STREAM
  RECORD_FUNCTION("dgl::CSRToCOODataAsOrder[Ascend]", std::vector<c10::IValue>());
#endif
  // Both row expansion and permutation execute on the NPU. Only sizes and
  // pointers are handled on the host; no CSR array is copied to the CPU.
  const int64_t nnz = csr.indices->shape[0];
  if (!CSRHasData(csr) || nnz == 0) {
    COOMatrix coo = CSRToCOO<XPU, IdType>(csr);
    return COOMatrix(csr.num_rows, csr.num_cols, coo.row, coo.col);
  }
  const auto ctx = csr.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend);
  // Allocate the final row/col together, retaining the common owner in both
  // views. Padding keeps both device addresses 512-byte aligned for A2.
  CHECK_LE(nnz, (std::numeric_limits<int64_t>::max() - 1023) /
                    (2 * sizeof(IdType)));
  const int64_t bytes = nnz * sizeof(IdType);
  const int64_t stride_bytes = (bytes + 511) / 512 * 512;
  NDArray packed = NewCOOArray(2 * stride_bytes / sizeof(IdType),
                               csr.indices->dtype, ctx);
  NDArray row = packed.CreateView({nnz}, csr.indices->dtype);
  NDArray col = packed.CreateView({nnz}, csr.indices->dtype, stride_bytes);
  NDArray temporary_row = NewCOOArray(nnz, csr.indices->dtype, ctx);
  auto stream = CsrStream(ctx);
  const uint32_t block_dim = CsrBlockDim(ctx, nnz);
  // Both kernels run in order on the same stream. Keep the intermediate alive
  // and synchronize only after the permutation, avoiding a host round trip.
  LaunchCsrRows<IdType>(csr, temporary_row, stream, block_dim);
  if (std::is_same<IdType, int32_t>::value) {
    ASCEND_CALL(aclrtlaunch_csr_to_coo_order_int32(
        block_dim, stream, temporary_row->data, csr.indices->data, csr.data->data,
        row->data, col->data, static_cast<uint64_t>(nnz)));
  } else {
    ASCEND_CALL(aclrtlaunch_csr_to_coo_order_int64(
        block_dim, stream, temporary_row->data, csr.indices->data, csr.data->data,
        row->data, col->data, static_cast<uint64_t>(nnz)));
  }
  // Complete all DMA reads before the temporary CSR-order row array is freed.
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  return COOMatrix(csr.num_rows, csr.num_cols, row, col);
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

template COOMatrix CSRToCOODataAsOrder<kDGLAscend, int32_t>(CSRMatrix);
template COOMatrix CSRToCOODataAsOrder<kDGLAscend, int64_t>(CSRMatrix);

#ifdef DGL_USE_PYTORCH_NPU_STREAM
namespace {
// A synchronous borrow: the dispatcher retains the input tensors for this call
// and the conversion completes before returning. No tensor data is copied.
NDArray BorrowCsrTensor(const at::Tensor& tensor) {
  auto container = std::make_unique<NDArray::Container>();
  container->dl_tensor.data = tensor.data_ptr();
  container->dl_tensor.ctx = {kDGLAscend, tensor.get_device()};
  container->dl_tensor.dtype = {
      kDGLInt, static_cast<uint8_t>(tensor.element_size() * 8), 1};
  container->dl_tensor.ndim = 1;
  container->dl_tensor.shape = new int64_t[1]{tensor.numel()};
  container->deleter = [](NDArray::Container* self) {
    delete[] self->dl_tensor.shape;
    delete self;
  };
  return NDArray(container.release());
}

at::Tensor NewCsrTensor(int64_t size, const at::TensorOptions& options,
                         int64_t element_size) {
  // Keep the framework's real Storage/DataPtr, including its stream tracking.
  // A tensor view supplies the same 512-byte offset as the DGL allocation.
  const int64_t offset = 512 / element_size;
  TORCH_CHECK(size <= std::numeric_limits<int64_t>::max() - offset,
              "CSR output size overflow");
  auto storage = at::empty({size + offset}, options);
  return storage.narrow(0, offset, size);
}

at::Tensor NewCsrTensor(int64_t size, const at::Tensor& reference) {
  return NewCsrTensor(size, reference.options(), reference.element_size());
}

at::Tensor CopyGraphCsrArray(const NDArray& array, aclrtStream stream) {
  const auto dtype = array->dtype.bits == 32 ? at::kInt : at::kLong;
  const int64_t size = array->shape[0];
  auto options = at::TensorOptions().dtype(dtype);
  if (array->ctx.device_type == kDGLCPU && size == 0) {
    // Preserve the existing missing-data sentinel, including its CPU context.
    return at::empty({0}, options.device(at::kCPU));
  }
  options = options.device(c10::Device(c10::DeviceType::PrivateUse1,
                                     array->ctx.device_id));
  auto result = NewCsrTensor(size, options, array->dtype.bits / 8);
  if (size != 0) {
    const size_t bytes = size * (array->dtype.bits / 8);
    const void* source = static_cast<const char*>(array->data) +
                         array->byte_offset;
    ASCEND_CALL(aclrtMemcpyAsync(result.data_ptr(), bytes, source, bytes,
                                ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
  }
  return result;
}
}  // namespace

// Private bridge for HeteroGraphIndex's CSR conversion calls. The Python caller
// holds the DGL ObjectHandle during this synchronous operation; copy its shared
// owner before accessing graph metadata. Unsupported requests have no effects
// and return an empty list so the original DGL path handles them unchanged.
std::tuple<std::vector<at::Tensor>, int64_t, int64_t> GraphCsrToCOO(
    int64_t graph_handle, int64_t etype, bool materialize, bool transpose) {
  using Result = std::tuple<std::vector<at::Tensor>, int64_t, int64_t>;
  TORCH_CHECK(graph_handle != 0, "Null DGL graph handle");
  auto* holder = reinterpret_cast<std::shared_ptr<runtime::Object>*>(
      static_cast<uintptr_t>(graph_handle));
  auto graph = std::dynamic_pointer_cast<BaseHeteroGraph>(*holder);
  TORCH_CHECK(graph, "Expected a DGL heterograph handle");
  if (etype < 0 || etype >= graph->NumEdgeTypes() ||
      graph->Context().device_type != kDGLAscend) {
    return Result({}, 0, 0);
  }
  HeteroGraphPtr relation = graph;
  auto unit = std::dynamic_pointer_cast<UnitGraph>(relation);
  if (unit) {
    if (etype != 0) return Result({}, 0, 0);
  } else if (auto hetero = std::dynamic_pointer_cast<HeteroGraph>(graph)) {
    relation = hetero->GetRelationGraph(static_cast<dgl_type_t>(etype));
    unit = std::dynamic_pointer_cast<UnitGraph>(relation);
  }
  if (!unit) return Result({}, 0, 0);
  const auto created = unit->GetCreatedFormats();
  if (!(created & CSR_CODE) || (created & COO_CODE) ||
      (materialize && (created & CSC_CODE)) ||
      (materialize && !(unit->GetAllowedFormats() & COO_CODE)) ||
      (!materialize && unit->SelectFormat(0, ALL_CODE) != SparseFormat::kCSR)) {
    return Result({}, 0, 0);
  }
  auto csr = unit->GetCSRMatrix(0);
  if (csr.indptr->dtype.code != kDGLInt ||
      (csr.indptr->dtype.bits != 32 && csr.indptr->dtype.bits != 64) ||
      csr.indptr->byte_offset || csr.indices->byte_offset ||
      (csr.data.defined() && csr.data->byte_offset)) {
    return Result({}, 0, 0);
  }
  const auto ctx = csr.indptr->ctx;
  const c10_npu::NPUGuard guard(
      c10::Device(c10::DeviceType::PrivateUse1, ctx.device_id));
  TORCH_CHECK(c10_npu::npuSynchronizeDevice(true),
              "CSRToCOO graph producer synchronization failed");
  std::vector<NDArray> arrays;
  const auto endpoints = graph->GetEndpointTypes(static_cast<dgl_type_t>(etype));
  int64_t rows = graph->NumVertices(endpoints.first);
  int64_t cols = graph->NumVertices(endpoints.second);
  if (materialize) {
    // GetCOOMatrix uses the normal DGL materialization path, including format
    // restrictions, cached COO state, and the existing NPU CSR kernels.
    auto coo = unit->GetCOOMatrix(0);
    arrays = transpose ? std::vector<NDArray>{coo.col, coo.row}
                       : std::vector<NDArray>{coo.row, coo.col};
    if (transpose) std::swap(rows, cols);
  } else {
    auto edges = unit->Edges(0, "");
    arrays = {edges.src, edges.dst, edges.id};
  }
  auto stream = CsrStream(ctx);
  std::vector<at::Tensor> result;
  result.reserve(arrays.size());
  for (const auto& array : arrays) {
    result.push_back(CopyGraphCsrArray(array, stream));
  }
  // Arrays and graph owners stay alive until every asynchronous device copy
  // finishes. Returned tensors have genuine framework-owned Storage/DataPtr.
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  return Result(std::move(result), rows, cols);
}

// Optional tensor interface scoped to CSRToCOO. It reuses exactly the DGL
// kernels above, with framework-owned outputs and final completion. In contrast
// to graph.edges()/adj_tensors(), no generic DGL handle extraction is needed.
std::tuple<at::Tensor, at::Tensor, at::Tensor> NativeCsrToCOO(
    const at::Tensor& indptr, const at::Tensor& indices,
    const at::Tensor& data, int64_t num_cols, bool data_as_order) {
  TORCH_CHECK(indptr.device().type() == c10::DeviceType::PrivateUse1,
              "CSRToCOO requires NPU tensors");
  TORCH_CHECK(indices.device() == indptr.device() &&
                  data.device() == indptr.device(),
              "CSR tensors must be on the same NPU");
  TORCH_CHECK(indptr.scalar_type() == at::kInt ||
                  indptr.scalar_type() == at::kLong,
              "CSR IDs must be int32 or int64");
  TORCH_CHECK(indices.scalar_type() == indptr.scalar_type() &&
                  data.scalar_type() == indptr.scalar_type(),
              "CSR tensors must have identical ID types");
  TORCH_CHECK(indptr.dim() == 1 && indices.dim() == 1 && data.dim() == 1 &&
                  indptr.is_contiguous() && indices.is_contiguous() &&
                  data.is_contiguous(),
              "CSR tensors must be contiguous vectors");
  TORCH_CHECK(indptr.numel() >= 1 && num_cols >= 0 &&
                  (indptr.numel() > 1 || indices.numel() == 0) &&
                  (data.numel() == 0 || data.numel() == indices.numel()),
              "Invalid CSR metadata");
  TORCH_CHECK(indptr.numel() - 1 <= UINT32_MAX,
              "CSR row count exceeds the kernel's uint32 range");
  const c10_npu::NPUGuard guard(indptr.device());
  // Drain the framework task queue and all producer streams once. Submitting
  // AscendC work directly to a torch_npu task-queue stream is deliberately
  // avoided; DGL keeps its own stream and completes it before returning.
  TORCH_CHECK(c10_npu::npuSynchronizeDevice(true),
              "CSRToCOO producer synchronization failed");
  CSRMatrix csr(indptr.numel() - 1, num_cols, BorrowCsrTensor(indptr),
                BorrowCsrTensor(indices), BorrowCsrTensor(data));
  const int64_t nnz = indices.numel();
  const bool permute = data_as_order && data.numel() != 0 && nnz != 0;
  at::Tensor row = NewCsrTensor(nnz, indices);
  at::Tensor col = permute ? NewCsrTensor(nnz, indices) : indices;
  at::Tensor temporary_row = permute ? NewCsrTensor(nnz, indices) : row;
  at::Tensor ids = data_as_order ? at::empty({0}, data.options()) : data;
  if (nnz != 0) {
    const auto ctx = csr.indptr->ctx;
    auto stream = CsrStream(ctx);
    const uint32_t block_dim = CsrBlockDim(ctx, nnz);
    NDArray rows = BorrowCsrTensor(temporary_row);
    if (indptr.scalar_type() == at::kInt) {
      LaunchCsrRows<int32_t>(csr, rows, stream, block_dim);
      if (permute) {
        ASCEND_CALL(aclrtlaunch_csr_to_coo_order_int32(
            block_dim, stream, temporary_row.data_ptr(), indices.data_ptr(),
            data.data_ptr(), row.data_ptr(), col.data_ptr(), nnz));
      }
    } else {
      LaunchCsrRows<int64_t>(csr, rows, stream, block_dim);
      if (permute) {
        ASCEND_CALL(aclrtlaunch_csr_to_coo_order_int64(
            block_dim, stream, temporary_row.data_ptr(), indices.data_ptr(),
            data.data_ptr(), row.data_ptr(), col.data_ptr(), nnz));
      }
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));
  }
  return {std::move(row), std::move(col), std::move(ids)};
}
#endif

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#ifdef DGL_USE_PYTORCH_NPU_STREAM
TORCH_LIBRARY_FRAGMENT(dgl_ascend, m) {
  m.def("csr_to_coo(Tensor indptr, Tensor(a) indices, Tensor(b) data, "
        "int num_cols, bool data_as_order=False) -> "
        "(Tensor, Tensor(a), Tensor(b))");
  m.def(torch::schema(
            "_csr_to_coo_graph(int graph_handle, int etype, bool materialize, "
            "bool transpose=False) -> (Tensor[], int, int)",
            c10::AliasAnalysisKind::CONSERVATIVE),
        TORCH_FN(dgl::aten::impl::GraphCsrToCOO));
}
TORCH_LIBRARY_IMPL(dgl_ascend, PrivateUse1, m) {
  m.impl("csr_to_coo", TORCH_FN(dgl::aten::impl::NativeCsrToCOO));
}
#endif
