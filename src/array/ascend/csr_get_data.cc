#ifdef DGL_USE_PYTORCH_NPU_STREAM
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#endif
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <dgl/array.h>
#include <dgl/runtime/registry.h>
#include "../array_op.h"

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
extern "C" aclError aclrtGetDeviceInfo(uint32_t, aclrtDevAttr, int64_t*);
#define GET_DATA_ACL(call) CHECK_EQ((call), ACL_SUCCESS)
#define DECLARE_GET_DATA(NAME)                                               \
extern "C" uint32_t aclrtlaunch_##NAME(                                      \
    uint32_t, aclrtStream, void*, void*, void*, void*, void*, void*, void*,    \
    uint32_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t,      \
    uint64_t)
DECLARE_GET_DATA(csr_get_data_int32);
DECLARE_GET_DATA(csr_get_data_int64);
DECLARE_GET_DATA(csr_get_data_weighted_int32);
DECLARE_GET_DATA(csr_get_data_weighted_int64);
DECLARE_GET_DATA(csr_get_data_weighted_f64_int32);
DECLARE_GET_DATA(csr_get_data_weighted_f64_int64);
#undef DECLARE_GET_DATA
namespace dgl {
namespace runtime { aclrtStream getCurrentAscendStream(); }
}
#endif

namespace dgl {
namespace aten {
namespace impl {
namespace {
NDArray NewLookupArray(int64_t n, DGLDataType dtype, DGLContext ctx) {
#ifdef DGL_USE_PYTORCH_NPU_STREAM
  const c10::Device device(c10::DeviceType::PrivateUse1, ctx.device_id);
  auto* allocator = c10_npu::NPUCachingAllocator::get();
  if (!allocator->initialized()) {
    at::empty({0}, at::TensorOptions().device(device).dtype(at::kByte));
  }
  const c10_npu::NPUGuard guard(device);
  auto owner = std::make_unique<c10::DataPtr>(
      allocator->allocate(static_cast<size_t>(n) * (dtype.bits / 8) + 512));
  auto container = std::make_unique<NDArray::Container>();
  container->dl_tensor.data = static_cast<char*>(owner->get()) + 512;
  container->dl_tensor.ctx = ctx;
  container->dl_tensor.dtype = dtype;
  container->dl_tensor.ndim = 1;
  container->dl_tensor.shape = new int64_t[1]{n};
  container->manager_ctx = owner.release();
  container->deleter = [](NDArray::Container* self) {
    delete static_cast<c10::DataPtr*>(self->manager_ctx);
    delete[] self->dl_tensor.shape;
    delete self;
  };
  return NDArray(container.release());
#else
  return NDArray::Empty({n}, dtype, ctx);
#endif
}
#ifdef DGL_USE_ASCEND
uint32_t LookupCores(DGLContext ctx, uint64_t n) {
  static thread_local int device = -1;
  static thread_local uint32_t cores = 40;
  if (device != ctx.device_id) {
    int64_t available = 0;
    cores = 40;
    if (aclrtGetDeviceInfo(ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                           &available) == ACL_SUCCESS && available > 0 &&
        available <= 4096) cores = static_cast<uint32_t>(available);
    device = ctx.device_id;
  }
  return static_cast<uint32_t>(std::min<uint64_t>(cores, (n + 63) / 64));
}
#endif
}  // namespace

template <DGLDeviceType XPU, typename IdType, typename DType>
NDArray CSRGetData(CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids,
                  NDArray weights, DType filler) {
#ifdef DGL_USE_ASCEND
#ifdef DGL_USE_PYTORCH_NPU_STREAM
  RECORD_FUNCTION("dgl::CSRGetData[Ascend]", std::vector<c10::IValue>());
#endif
  const auto ctx = rows->ctx;
  CHECK_EQ(ctx.device_type, kDGLAscend);
  CHECK_EQ(csr.indptr->ctx, ctx);
  CHECK_EQ(csr.indices->ctx, ctx);
  CHECK_EQ(cols->ctx, ctx);
  CHECK_EQ(rows->ndim, 1); CHECK_EQ(cols->ndim, 1);
  CHECK_GE(csr.num_rows, 0);
  CHECK_LE(csr.num_rows, static_cast<int64_t>(UINT32_MAX) - 1);
  const int64_t rowlen = rows->shape[0], collen = cols->shape[0];
  CHECK(rowlen == collen || rowlen == 1 || collen == 1)
      << "Invalid row and col id array.";
  const int64_t n = std::max(rowlen, collen);
  CHECK(n == 0 || (rowlen > 0 && collen > 0));
  if (!return_eids) {
    CHECK_EQ(weights->ctx, ctx);
    CHECK_EQ(weights->ndim, 1);
    // CSR data may reference a larger feature vector (e.g. a CSR slice).
    // As on CPU, callers must provide weights covering the referenced EIDs.
  }
  auto out = NewLookupArray(n, return_eids ? csr.indices->dtype : weights->dtype, ctx);
  if (!n) return out;
  int32_t current = -1;
  GET_DATA_ACL(aclrtGetDevice(&current));
  if (current != ctx.device_id) GET_DATA_ACL(aclrtSetDevice(ctx.device_id));
  const auto stream = runtime::getCurrentAscendStream();
  uint64_t filler_bits = 0;
  if (return_eids) {
    // The public EID contract is -1 for misses. The existing Ascend array
    // dispatcher supplies zero here; normalize locally without changing it.
    filler_bits = ~uint64_t(0);
  } else {
    std::memcpy(&filler_bits, &filler, sizeof(DType));
  }
  const uint32_t cores = LookupCores(ctx, n);
  void* data = CSRHasData(csr) ? csr.data->data : csr.indices->data;
  void* weight_data = return_eids ? csr.indices->data : weights->data;
#define LAUNCH_LOOKUP(NAME)                                                  \
  GET_DATA_ACL(aclrtlaunch_##NAME(                                           \
      cores, stream, csr.indptr->data, csr.indices->data, data, rows->data,    \
      cols->data, weight_data, out->data, static_cast<uint32_t>(csr.num_rows),\
      static_cast<uint64_t>(csr.num_cols), static_cast<uint64_t>(n),          \
      rowlen == 1 ? 0 : 1, collen == 1 ? 0 : 1, CSRHasData(csr) ? 1 : 0,      \
      csr.sorted ? 1 : 0, filler_bits))
  if constexpr (sizeof(IdType) == 4) {
    if (return_eids) { LAUNCH_LOOKUP(csr_get_data_int32); }
    else if constexpr (sizeof(DType) == 8) { LAUNCH_LOOKUP(csr_get_data_weighted_f64_int32); }
    else { LAUNCH_LOOKUP(csr_get_data_weighted_int32); }
  } else {
    if (return_eids) { LAUNCH_LOOKUP(csr_get_data_int64); }
    else if constexpr (sizeof(DType) == 8) { LAUNCH_LOOKUP(csr_get_data_weighted_f64_int64); }
    else { LAUNCH_LOOKUP(csr_get_data_weighted_int64); }
  }
#undef LAUNCH_LOOKUP
  GET_DATA_ACL(aclrtSynchronizeStream(stream));
  return out;
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

#define INSTANTIATE_GET_DATA(ID, DTYPE)                                      \
template NDArray CSRGetData<kDGLAscend, ID, DTYPE>(                           \
    CSRMatrix, NDArray, NDArray, bool, NDArray, DTYPE)
INSTANTIATE_GET_DATA(int32_t, int32_t);
INSTANTIATE_GET_DATA(int64_t, int64_t);
INSTANTIATE_GET_DATA(int32_t, float);
INSTANTIATE_GET_DATA(int64_t, float);
INSTANTIATE_GET_DATA(int32_t, double);
INSTANTIATE_GET_DATA(int64_t, double);
#undef INSTANTIATE_GET_DATA
}  // namespace impl

// A scoped packed entry for directly using CSRGetData with a SparseMatrix,
// including sorted metadata, broadcasting and a custom weighted filler.
// Graph EID/CSRMask paths above use the same implementation unchanged.
DGL_REGISTER_GLOBAL("ascend._CAPI_DGLCSRGetData")
    .set_body([](runtime::DGLArgs args, runtime::DGLRetValue* rv) {
      SparseMatrixRef matrix = args[0];
      CHECK_EQ(matrix->format, static_cast<int32_t>(SparseFormat::kCSR));
      const CSRMatrix csr(matrix->num_rows, matrix->num_cols,
                          matrix->indices[0], matrix->indices[1],
                          matrix->indices[2], matrix->flags[0]);
      NDArray rows = args[1], cols = args[2], weights = args[4];
      const bool eids = args[3];
      const double filler = args[5];
      if (eids) *rv = aten::CSRGetData(csr, rows, cols);
      else *rv = aten::CSRGetFloatingData(csr, rows, cols, weights, filler);
    });
}  // namespace aten
}  // namespace dgl
