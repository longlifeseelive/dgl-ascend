// Stable CSR transpose: private histograms, device prefix, device scatter.
#ifdef DGL_USE_PYTORCH_NPU_STREAM
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#endif
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <dgl/array.h>
#include "../array_op.h"

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
extern "C" aclError aclrtGetDeviceInfo(uint32_t, aclrtDevAttr, int64_t*);
#define TRANSPOSE_ACL(call) CHECK_EQ((call), ACL_SUCCESS)
#define DECLARE_TRANSPOSE(SUFFIX)                                             \
  extern "C" uint32_t aclrtlaunch_csr_transpose_count_##SUFFIX(               \
      uint32_t, aclrtStream, void*, void*, uint64_t, uint32_t, uint32_t);       \
  extern "C" uint32_t aclrtlaunch_csr_transpose_prefix_##SUFFIX(              \
      uint32_t, aclrtStream, void*, void*, void*, void*, uint32_t, uint32_t,    \
      uint32_t);                                                             \
  extern "C" uint32_t aclrtlaunch_csr_transpose_scatter_##SUFFIX(             \
      uint32_t, aclrtStream, void*, void*, void*, void*, void*, void*,         \
      uint32_t, uint64_t, uint32_t, uint32_t, uint32_t);                        \
  extern "C" uint32_t aclrtlaunch_csr_transpose_zero_##SUFFIX(                \
      uint32_t, aclrtStream, void*, uint64_t)
DECLARE_TRANSPOSE(int32);
DECLARE_TRANSPOSE(int64);
#undef DECLARE_TRANSPOSE
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
// Retain the caching allocator owner across DGL/PyTorch views, following the
// existing Ascend allocation convention. Converted results are never cached.
NDArray NewTransposeArray(int64_t size, DGLDataType dtype, DGLContext ctx) {
#ifdef DGL_USE_PYTORCH_NPU_STREAM
  const c10::Device device(c10::DeviceType::PrivateUse1, ctx.device_id);
  auto* allocator = c10_npu::NPUCachingAllocator::get();
  if (!allocator->initialized()) {
    at::empty({0}, at::TensorOptions().device(device).dtype(at::kByte));
  }
  const c10_npu::NPUGuard guard(device);
  auto owner = std::make_unique<c10::DataPtr>(
      allocator->allocate(static_cast<size_t>(size) * (dtype.bits / 8) + 512));
  auto container = std::make_unique<NDArray::Container>();
  container->dl_tensor.data = static_cast<char*>(owner->get()) + 512;
  container->dl_tensor.ctx = ctx;
  container->dl_tensor.dtype = dtype;
  container->dl_tensor.ndim = 1;
  container->dl_tensor.shape = new int64_t[1]{size};
  container->manager_ctx = owner.release();
  container->deleter = [](NDArray::Container* self) {
    delete static_cast<c10::DataPtr*>(self->manager_ctx);
    delete[] self->dl_tensor.shape;
    delete self;
  };
  return NDArray(container.release());
#else
  return NDArray::Empty({size}, dtype, ctx);
#endif
}
#ifdef DGL_USE_ASCEND
uint32_t TransposeCores(DGLContext ctx, uint64_t work) {
  static thread_local int cached_device = -1;
  static thread_local uint32_t cores = 40;
  if (cached_device != ctx.device_id) {
    int64_t available = 0;
    cores = 40;
    if (aclrtGetDeviceInfo(ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                         &available) == ACL_SUCCESS &&
        available > 0 && available <= 4096) {
      cores = static_cast<uint32_t>(available);
    }
    cached_device = ctx.device_id;
  }
  return static_cast<uint32_t>(std::min<uint64_t>(cores, (work + 2047) / 2048));
}
#endif

template <typename IdType>
CSRMatrix Transpose(CSRMatrix csr) {
#ifdef DGL_USE_ASCEND
#ifdef DGL_USE_PYTORCH_NPU_STREAM
  RECORD_FUNCTION("dgl::CSRTranspose[Ascend]", std::vector<c10::IValue>());
#endif
  const auto ctx = csr.indptr->ctx;
  CHECK_EQ(ctx.device_type, kDGLAscend);
  const int64_t nnz = csr.indices->shape[0];
  // A2 counts/absolute offsets use int32; ID and data payloads keep their width.
  CHECK_LE(nnz, static_cast<int64_t>(INT32_MAX));
  CHECK_GE(csr.num_rows, 0);
  CHECK_GE(csr.num_cols, 0);
  CHECK_LE(csr.num_rows, static_cast<int64_t>(UINT32_MAX));
  CHECK_LE(csr.num_cols, static_cast<int64_t>(UINT32_MAX));
  CHECK_LE(csr.num_rows, std::numeric_limits<IdType>::max());
  CHECK_LE(csr.num_cols, std::numeric_limits<IdType>::max());
  auto ptr = NewTransposeArray(csr.num_cols + 1, csr.indptr->dtype, ctx);
  auto idx = NewTransposeArray(nnz, csr.indices->dtype, ctx);
  auto data = NewTransposeArray(nnz, csr.indices->dtype, ctx);
  int32_t current = -1;
  TRANSPOSE_ACL(aclrtGetDevice(&current));
  if (current != ctx.device_id) TRANSPOSE_ACL(aclrtSetDevice(ctx.device_id));
  auto stream = runtime::getCurrentAscendStream();
  if (nnz == 0) {
    const uint32_t cores = TransposeCores(ctx, csr.num_cols + 1);
    if constexpr (sizeof(IdType) == 4) {
      TRANSPOSE_ACL(aclrtlaunch_csr_transpose_zero_int32(
          cores, stream, ptr->data, csr.num_cols + 1));
    } else {
      TRANSPOSE_ACL(aclrtlaunch_csr_transpose_zero_int64(
          cores, stream, ptr->data, csr.num_cols + 1));
    }
    TRANSPOSE_ACL(aclrtSynchronizeStream(stream));
    return CSRMatrix(csr.num_cols, csr.num_rows, ptr, idx, data, true);
  }
  CHECK_GT(csr.num_rows, 0);
  CHECK_GT(csr.num_cols, 0);
  const uint32_t cores = TransposeCores(ctx, nnz);
  // Bands bound UB usage for wide matrices. All inter-band prefix state is
  // device-resident; only shape metadata controls the host launch loop.
  constexpr uint32_t kBand = 16384;
  const uint32_t max_cols = std::min<int64_t>(csr.num_cols, kBand);
  const uint32_t max_stride = (max_cols + 7) / 8 * 8;
  DGLDataType count_type{kDGLInt, 32, 1};
  auto counts = NewTransposeArray(cores * max_stride, count_type, ctx);
  auto offsets = NewTransposeArray(cores * max_stride, count_type, ctx);
  auto carry = NewTransposeArray(8, count_type, ctx);
  for (uint64_t first = 0; first < static_cast<uint64_t>(csr.num_cols);
       first += kBand) {
    const uint32_t cols = std::min<uint64_t>(kBand, csr.num_cols - first);
    void* input_data = CSRHasData(csr) ? csr.data->data : csr.indices->data;
#define LAUNCH_TRANSPOSE(SUFFIX)                                              \
    TRANSPOSE_ACL(aclrtlaunch_csr_transpose_count_##SUFFIX(                   \
        cores, stream, csr.indices->data, counts->data, nnz, first, cols));    \
    TRANSPOSE_ACL(aclrtlaunch_csr_transpose_prefix_##SUFFIX(                  \
        1, stream, counts->data, offsets->data, ptr->data, carry->data,        \
        cores, first, cols));                                                \
    TRANSPOSE_ACL(aclrtlaunch_csr_transpose_scatter_##SUFFIX(                 \
        cores, stream, csr.indptr->data, csr.indices->data, input_data,        \
        offsets->data, idx->data, data->data, csr.num_rows, nnz, first, cols,   \
        CSRHasData(csr) ? 1 : 0))
    if constexpr (sizeof(IdType) == 4) {
      LAUNCH_TRANSPOSE(int32);
    } else {
      LAUNCH_TRANSPOSE(int64);
    }
#undef LAUNCH_TRANSPOSE
  }
  // Retain workspace until queued kernels complete (including the final band).
  TRANSPOSE_ACL(aclrtSynchronizeStream(stream));
  return CSRMatrix(csr.num_cols, csr.num_rows, ptr, idx, data, true);
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}
}  // namespace

template <>
CSRMatrix CSRTranspose<kDGLAscend, int32_t>(CSRMatrix csr) {
  return Transpose<int32_t>(csr);
}
template <>
CSRMatrix CSRTranspose<kDGLAscend, int64_t>(CSRMatrix csr) {
  return Transpose<int64_t>(csr);
}
}  // namespace impl
}  // namespace aten
}  // namespace dgl
