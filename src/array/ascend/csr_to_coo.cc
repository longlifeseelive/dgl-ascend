#ifdef DGL_USE_PYTORCH_NPU_STREAM
#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#endif

#include <cstdint>
#include <memory>
#include <type_traits>

#include <dgl/array.h>
#include "../array_op.h"

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
#endif
}  // namespace

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix csr) {
#ifdef DGL_USE_ASCEND
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

  int32_t current_device = -1;
  ASCEND_CALL(aclrtGetDevice(&current_device));
  if (current_device != ctx.device_id) {
    ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  }
  auto stream = dgl::runtime::getCurrentAscendStream();

  CHECK_LE(csr.indptr->shape[0] - 1, static_cast<int64_t>(UINT32_MAX));
  const uint32_t num_rows =
      static_cast<uint32_t>(csr.indptr->shape[0] - 1);
  if (num_rows == 0) {
    return COOMatrix(csr.num_rows, csr.num_cols, ret_row, csr.indices,
                     csr.data, true, csr.sorted);
  }

  const uint32_t block_dim = CsrBlockDim(ctx, nnz);

  if (std::is_same<IdType, int32_t>::value) {
    aclError err = aclrtlaunch_csr_to_coo_int32(
        block_dim, stream, csr.indptr->data, ret_row->data, num_rows, nnz);
    CHECK(err == ACL_SUCCESS) << "csr_to_coo_int32 launch failed: " << err;
  } else {
    aclError err = aclrtlaunch_csr_to_coo_int64(
        block_dim, stream, csr.indptr->data, ret_row->data, num_rows, nnz);
    CHECK(err == ACL_SUCCESS) << "csr_to_coo_int64 launch failed: " << err;
  }

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
  // Both row expansion and permutation execute on the NPU. Only sizes and
  // pointers are handled on the host; no CSR array is copied to the CPU.
  COOMatrix coo = CSRToCOO<XPU, IdType>(csr);
  const int64_t nnz = csr.indices->shape[0];
  if (!CSRHasData(csr) || nnz == 0) {
    return COOMatrix(csr.num_rows, csr.num_cols, coo.row, coo.col);
  }
  const auto ctx = csr.indptr->ctx;
  NDArray row = NewCOOArray(nnz, csr.indices->dtype, ctx);
  NDArray col = NewCOOArray(nnz, csr.indices->dtype, ctx);
  auto stream = dgl::runtime::getCurrentAscendStream();
  const uint32_t block_dim = CsrBlockDim(ctx, nnz);
  if (std::is_same<IdType, int32_t>::value) {
    ASCEND_CALL(aclrtlaunch_csr_to_coo_order_int32(
        block_dim, stream, coo.row->data, csr.indices->data, csr.data->data,
        row->data, col->data, static_cast<uint64_t>(nnz)));
  } else {
    ASCEND_CALL(aclrtlaunch_csr_to_coo_order_int64(
        block_dim, stream, coo.row->data, csr.indices->data, csr.data->data,
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

}  // namespace impl
}  // namespace aten
}  // namespace dgl
