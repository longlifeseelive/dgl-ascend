#ifdef DGL_USE_PYTORCH_NPU_STREAM
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include "../../graph/heterograph.h"
#endif
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>
#include <optional>
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
    void*,                                                                  \
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
constexpr uint32_t kLookupMaxCores = 64;
struct LookupHostStatus {
  uint64_t* data = nullptr;
  LookupHostStatus() {
    GET_DATA_ACL(aclrtMallocHost(reinterpret_cast<void**>(&data), 512));
  }
  ~LookupHostStatus() { if (data) aclrtFreeHost(data); }
};

uint32_t LookupCores(DGLContext ctx, uint64_t n) {
  static thread_local int device = -1;
  static thread_local uint32_t cores = 40;
  if (device != ctx.device_id) {
    int64_t available = 0;
    cores = 40;
    if (aclrtGetDeviceInfo(ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                           &available) == ACL_SUCCESS && available > 0 &&
        available <= 4096)
      cores = static_cast<uint32_t>(std::min<int64_t>(available, kLookupMaxCores));
    device = ctx.device_id;
  }
  return static_cast<uint32_t>(std::min<uint64_t>(cores, (n + 63) / 64));
}
#endif
template <typename IdType, typename DType>
NDArray Lookup(CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids,
               NDArray weights, DType filler, NDArray out = NDArray()) {
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
  if (!out.defined()) {
    out = NewLookupArray(n, return_eids ? csr.indices->dtype : weights->dtype, ctx);
  }
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
  // Both allocation paths reserve the preceding 512 bytes. Each core writes
  // one exact-length uint64 status there, avoiding an extra device allocation.
  void* status = static_cast<char*>(out->data) - 512;
  static thread_local LookupHostStatus host_status;
  void* data = CSRHasData(csr) ? csr.data->data : csr.indices->data;
  void* weight_data = return_eids ? csr.indices->data : weights->data;
#define LAUNCH_LOOKUP(NAME)                                                  \
  GET_DATA_ACL(aclrtlaunch_##NAME(                                           \
      cores, stream, csr.indptr->data, csr.indices->data, data, rows->data,    \
      cols->data, weight_data, out->data, status,                             \
      static_cast<uint32_t>(csr.num_rows),                                   \
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
  GET_DATA_ACL(aclrtMemcpyAsync(host_status.data, 512, status,
      cores * sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST, stream));
  GET_DATA_ACL(aclrtSynchronizeStream(stream));
  uint64_t error = 0;
  for (uint32_t i = 0; i < cores; ++i) {
    const uint64_t code = host_status.data[i];
    if (code && (!error || code < error)) error = code;
  }
  if (error) {
    const bool bad_row = error & 1;
    const uint64_t query = (error - 1) / 2;
    const NDArray& ids = bad_row ? rows : cols;
    const uint64_t pos = ids->shape[0] == 1 ? 0 : query;
    IdType value = 0;
    GET_DATA_ACL(aclrtMemcpy(&value, sizeof(value),
        static_cast<const IdType*>(ids->data) + pos, sizeof(value),
        ACL_MEMCPY_DEVICE_TO_HOST));
    LOG(FATAL) << (bad_row ? "Invalid row index: " : "Invalid col index: ") << value;
  }
  return out;
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}
}  // namespace

template <DGLDeviceType XPU, typename IdType, typename DType>
NDArray CSRGetData(CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids,
                  NDArray weights, DType filler) {
  return Lookup<IdType, DType>(csr, rows, cols, return_eids, weights, filler);
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

#ifdef DGL_USE_PYTORCH_NPU_STREAM
namespace {
// Synchronous borrow: the dispatcher owns the tensors throughout the call,
// and Lookup completes every DMA before returning. No DLPack or data copy.
NDArray BorrowLookupTensor(const at::Tensor& tensor) {
  auto container = std::make_unique<NDArray::Container>();
  container->dl_tensor.data = tensor.data_ptr();
  container->dl_tensor.ctx = {kDGLAscend, tensor.device().index()};
  container->dl_tensor.dtype = {
      static_cast<uint8_t>(tensor.is_floating_point() ? kDGLFloat : kDGLInt),
      static_cast<uint8_t>(tensor.element_size() * 8), 1};
  container->dl_tensor.ndim = 1;
  container->dl_tensor.shape = new int64_t[1]{tensor.numel()};
  container->deleter = [](NDArray::Container* self) {
    delete[] self->dl_tensor.shape;
    delete self;
  };
  return NDArray(container.release());
}

at::Tensor LookupTensorResult(const CSRMatrix& csr, NDArray r, NDArray c,
                             const at::Tensor& weights, bool eids, double filler) {
  TORCH_CHECK(r->ndim == 1 && c->ndim == 1 &&
                  r->dtype.code == kDGLInt && c->dtype.code == kDGLInt &&
                  r->dtype.bits == csr.indptr->dtype.bits &&
                  c->dtype.bits == csr.indptr->dtype.bits,
              "CSR and queries require identical ID types");
  const int64_t nr = r->shape[0], nc = c->shape[0];
  TORCH_CHECK(nr == nc || nr == 1 || nc == 1, "Invalid row and col id array.");
  const int64_t n = std::max(nr, nc);
  TORCH_CHECK(n == 0 || (nr > 0 && nc > 0), "Invalid empty query broadcasting");
  const auto device = c10::Device(c10::DeviceType::PrivateUse1,
                                   csr.indptr->ctx.device_id);
  const c10_npu::NPUGuard guard(device);
  TORCH_CHECK(c10_npu::npuSynchronizeDevice(true),
              "CSRGetData producer synchronization failed");
  const auto options = eids
      ? at::TensorOptions().device(device).dtype(csr.indices->dtype.bits == 32 ? at::kInt : at::kLong)
      : weights.options();
  const int64_t bytes = eids ? csr.indices->dtype.bits / 8 : weights.element_size();
  const int64_t offset = 512 / bytes;
  TORCH_CHECK(n <= std::numeric_limits<int64_t>::max() - offset,
              "CSRGetData output size overflow");
  auto storage = at::empty({n + offset}, options.requires_grad(false));
  auto out = storage.narrow(0, offset, n);
  auto w = eids ? NDArray() : BorrowLookupTensor(weights);
  auto target = BorrowLookupTensor(out);
  if (csr.indptr->dtype.bits == 32) {
    if (eids) Lookup<int32_t, int32_t>(csr, r, c, true, w, -1, target);
    else if (weights.scalar_type() == at::kDouble)
      Lookup<int32_t, double>(csr, r, c, false, w, filler, target);
    else Lookup<int32_t, float>(csr, r, c, false, w, static_cast<float>(filler), target);
  } else {
    if (eids) Lookup<int64_t, int64_t>(csr, r, c, true, w, -1, target);
    else if (weights.scalar_type() == at::kDouble)
      Lookup<int64_t, double>(csr, r, c, false, w, filler, target);
    else Lookup<int64_t, float>(csr, r, c, false, w, static_cast<float>(filler), target);
  }
  return out;
}
}

// Operator-scoped graph entry. The caller retains the DGL graph handle; this
// call also takes a shared owner. Existing Python graph/CSRMask APIs remain
// unchanged. A materialized CSR is required; no format conversion is hidden.
at::Tensor GraphCSRGetData(int64_t graph_handle, int64_t etype,
                          const at::Tensor& rows, const at::Tensor& cols,
                          const at::Tensor& weights, bool return_eids,
                          double filler) {
  TORCH_CHECK(graph_handle != 0, "Null DGL graph handle");
  auto* holder = reinterpret_cast<std::shared_ptr<runtime::Object>*>(
      static_cast<uintptr_t>(graph_handle));
  auto graph = std::dynamic_pointer_cast<BaseHeteroGraph>(*holder);
  TORCH_CHECK(graph && etype >= 0 && etype < graph->NumEdgeTypes(),
              "Invalid DGL graph or edge type");
  HeteroGraphPtr relation = graph;
  auto unit = std::dynamic_pointer_cast<UnitGraph>(relation);
  if (!unit) {
    auto hetero = std::dynamic_pointer_cast<HeteroGraph>(graph);
    TORCH_CHECK(hetero, "Expected a DGL heterograph");
    relation = hetero->GetRelationGraph(static_cast<dgl_type_t>(etype));
    unit = std::dynamic_pointer_cast<UnitGraph>(relation);
  }
  TORCH_CHECK(unit && (unit->GetCreatedFormats() & CSR_CODE),
              "CSRGetData requires an already materialized CSR");
  auto csr = unit->GetCSRMatrix(0);
  TORCH_CHECK(rows.device().type() == c10::DeviceType::PrivateUse1 &&
                  cols.device() == rows.device() &&
                  csr.indptr->ctx.device_type == kDGLAscend &&
                  csr.indptr->ctx.device_id == rows.device().index(),
              "CSR and queries must be on the same NPU");
  TORCH_CHECK(rows.dim() == 1 && cols.dim() == 1 && rows.is_contiguous() &&
                  cols.is_contiguous(), "Queries must be contiguous vectors");
  TORCH_CHECK((rows.scalar_type() == at::kInt || rows.scalar_type() == at::kLong) &&
                  cols.scalar_type() == rows.scalar_type() &&
                  csr.indptr->dtype.bits == rows.element_size() * 8,
              "CSR and queries require the same int32/int64 ID type");
  if (!return_eids) {
    TORCH_CHECK(weights.device() == rows.device() && weights.dim() == 1 &&
                    weights.is_contiguous() &&
                    (weights.scalar_type() == at::kFloat || weights.scalar_type() == at::kDouble),
                "Weights must be a contiguous float32/float64 NPU vector");
    TORCH_CHECK(!weights.requires_grad(),
                "This entry is a forward primitive; use DGL CSRMask for autograd");
  }
  return LookupTensorResult(csr, BorrowLookupTensor(rows), BorrowLookupTensor(cols),
                            weights, return_eids, filler);
}

namespace {
HeteroGraphPtr LookupGraphOwner(int64_t handle) {
  if (!handle) return {};
  auto* holder = reinterpret_cast<std::shared_ptr<runtime::Object>*>(
      static_cast<uintptr_t>(handle));
  return std::dynamic_pointer_cast<BaseHeteroGraph>(*holder);
}
std::shared_ptr<UnitGraph> LookupRelation(const HeteroGraphPtr& graph,
                                         int64_t etype) {
  if (!graph || etype < 0 || etype >= graph->NumEdgeTypes()) return {};
  if (auto unit = std::dynamic_pointer_cast<UnitGraph>(graph)) return unit;
  if (auto hetero = std::dynamic_pointer_cast<HeteroGraph>(graph)) {
    return std::dynamic_pointer_cast<UnitGraph>(
        hetero->GetRelationGraph(static_cast<dgl_type_t>(etype)));
  }
  return {};
}
}

std::optional<at::Tensor> TryGraphCSRGetData(
    int64_t handle, int64_t etype, const at::Tensor& rows,
    const at::Tensor& cols, const at::Tensor& weights, bool eids, double filler) {
  if (rows.device().type() != c10::DeviceType::PrivateUse1 ||
      cols.device() != rows.device() || !rows.is_contiguous() ||
      !cols.is_contiguous() || rows.dim() != 1 || cols.dim() != 1) return {};
  auto owner = LookupGraphOwner(handle);
  auto unit = LookupRelation(owner, etype);
  if (!unit || owner->Context().device_type != kDGLAscend ||
      !(unit->GetCreatedFormats() & CSR_CODE) ||
      !(unit->GetAllowedFormats() & CSR_CODE)) return {};
  auto csr = unit->GetCSRMatrix(0);
  if ((rows.scalar_type() != at::kInt && rows.scalar_type() != at::kLong) ||
      cols.scalar_type() != rows.scalar_type() ||
      csr.indptr->dtype.bits != rows.element_size() * 8) return {};
  return GraphCSRGetData(handle, etype, rows, cols, weights, eids, filler);
}

std::optional<at::Tensor> TryGraphCSRMask(
    int64_t a_handle, const at::Tensor& weights, int64_t b_handle) {
  if (weights.device().type() != c10::DeviceType::PrivateUse1 ||
      weights.dim() != 1 || !weights.is_contiguous() ||
      (weights.scalar_type() != at::kFloat && weights.scalar_type() != at::kDouble) ||
      (weights.requires_grad() && at::GradMode::is_enabled())) return {};
  auto a = LookupGraphOwner(a_handle), b = LookupGraphOwner(b_handle);
  if (!a || !b || a->NumEdgeTypes() != 1 || b->NumEdgeTypes() != 1 ||
      a->Context().device_type != kDGLAscend ||
      b->Context().device_type != kDGLAscend) return {};
  auto au = LookupRelation(a, 0), bu = LookupRelation(b, 0);
  if (!au || !bu || !(au->GetCreatedFormats() & CSR_CODE) ||
      !(bu->GetCreatedFormats() & COO_CODE) ||
      !(au->GetAllowedFormats() & CSR_CODE) ||
      !(bu->GetAllowedFormats() & COO_CODE)) return {};
  auto csr = au->GetCSRMatrix(0);
  auto coo = bu->GetCOOMatrix(0);
  if (csr.num_rows != coo.num_rows || csr.num_cols != coo.num_cols ||
      csr.indptr->dtype.bits != coo.row->dtype.bits) return {};
  // CSRMask's existing autograd.Function owns backward; this is its forward
  // primitive. Direct differentiable calls fall back above rather than detach.
  return LookupTensorResult(csr, coo.row, coo.col, weights.detach(), false, 0.);
}
#endif
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

#ifdef DGL_USE_PYTORCH_NPU_STREAM
TORCH_LIBRARY_FRAGMENT(dgl_ascend, m) {
  m.def(torch::schema(
      "_csr_get_data_graph(int graph_handle, int etype, Tensor rows, Tensor cols, "
      "Tensor weights, bool return_eids, float filler=0.) -> Tensor",
      c10::AliasAnalysisKind::CONSERVATIVE),
      TORCH_FN(dgl::aten::impl::GraphCSRGetData));
  m.def(torch::schema(
      "_try_csr_get_data_graph(int graph_handle, int etype, Tensor rows, Tensor cols, "
      "Tensor weights, bool return_eids, float filler=0.) -> Tensor?",
      c10::AliasAnalysisKind::CONSERVATIVE),
      TORCH_FN(dgl::aten::impl::TryGraphCSRGetData));
  m.def(torch::schema(
      "_try_csr_get_data_mask(int a_handle, Tensor weights, int b_handle) -> Tensor?",
      c10::AliasAnalysisKind::CONSERVATIVE),
      TORCH_FN(dgl::aten::impl::TryGraphCSRMask));
}
#endif
