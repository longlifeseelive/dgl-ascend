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
#include <vector>
#include <dgl/array.h>
#include <dgl/runtime/registry.h>
#include "../array_op.h"

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
extern "C" aclError aclrtGetDeviceInfo(uint32_t, aclrtDevAttr, int64_t*);
#define ALL_ACL(call) CHECK_EQ((call), ACL_SUCCESS)
#define DECLARE_ALL(SUFFIX) \
extern "C" uint32_t aclrtlaunch_csr_all_count_##SUFFIX( \
    uint32_t, aclrtStream, void*, void*, void*, void*, void*, void*, \
    uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t); \
extern "C" uint32_t aclrtlaunch_csr_all_emit_##SUFFIX( \
    uint32_t, aclrtStream, void*, void*, void*, void*, void*, void*, void*, void*, void*, \
    uint64_t, uint32_t, uint32_t, uint32_t, uint32_t)
DECLARE_ALL(int32);
DECLARE_ALL(int64);
#undef DECLARE_ALL
extern "C" uint32_t aclrtlaunch_csr_all_prefix(uint32_t, aclrtStream, void*, uint32_t);
namespace dgl { namespace runtime { aclrtStream getCurrentAscendStream(); } }
#endif

namespace dgl {
namespace aten {
namespace impl {
namespace {
NDArray NewAllArray(int64_t n, DGLDataType dtype, DGLContext ctx) {
  CHECK_GE(n, 0);
  CHECK_LE(static_cast<uint64_t>(n),
           (std::numeric_limits<size_t>::max() - 512) / (dtype.bits / 8))
      << "CSRGetDataAndIndices allocation size overflow";
#ifdef DGL_USE_PYTORCH_NPU_STREAM
  const c10::Device device(c10::DeviceType::PrivateUse1, ctx.device_id);
  auto* allocator = c10_npu::NPUCachingAllocator::get();
  if (!allocator->initialized()) at::empty({0}, at::TensorOptions().device(device).dtype(at::kByte));
  const c10_npu::NPUGuard guard(device);
  auto owner = std::make_unique<c10::DataPtr>(
      allocator->allocate(static_cast<size_t>(n) * (dtype.bits / 8) + 512));
  auto container = std::make_unique<NDArray::Container>();
  container->dl_tensor.data = static_cast<char*>(owner->get()) + 512;
  container->dl_tensor.ctx = ctx; container->dl_tensor.dtype = dtype;
  container->dl_tensor.ndim = 1; container->dl_tensor.shape = new int64_t[1]{n};
  container->manager_ctx = owner.release();
  container->deleter = [](NDArray::Container* self) {
    delete static_cast<c10::DataPtr*>(self->manager_ctx);
    delete[] self->dl_tensor.shape; delete self;
  };
  return NDArray(container.release());
#else
  return NDArray::Empty({n}, dtype, ctx);
#endif
}
#ifdef DGL_USE_ASCEND
void* Data(const NDArray& a) { return static_cast<char*>(a->data) + a->byte_offset; }
uint32_t AllCores(DGLContext ctx, uint64_t queries) {
  static thread_local int device = -1;
  static thread_local uint32_t cores = 40;
  if (device != ctx.device_id) {
    int64_t available = 0; cores = 40;
    if (aclrtGetDeviceInfo(ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &available) == ACL_SUCCESS && available > 0)
      cores = static_cast<uint32_t>(std::min<int64_t>(available, 64));
    device = ctx.device_id;
  }
  return static_cast<uint32_t>(std::min<uint64_t>(cores, (queries + 63) / 64));
}
struct HostInfo {
  uint64_t* data = nullptr;
  HostInfo() { ALL_ACL(aclrtMallocHost(reinterpret_cast<void**>(&data), 32)); }
  ~HostInfo() { if (data) aclrtFreeHost(data); }
};
// Declare all device owners before this fence. Exception unwinding then waits
// before releasing memory referenced by an earlier successful kernel launch.
struct Completion {
  aclrtStream stream;
  bool pending = false;
  explicit Completion(aclrtStream s) : stream(s) {}
  ~Completion() { if (pending) aclrtSynchronizeStream(stream); }
  void Wait() {
    ALL_ACL(aclrtSynchronizeStream(stream)); pending = false;
  }
};
#endif
}  // namespace

template <DGLDeviceType XPU, typename IdType>
std::vector<NDArray> CSRGetDataAndIndices(CSRMatrix csr, NDArray rows, NDArray cols) {
#ifdef DGL_USE_ASCEND
#ifdef DGL_USE_PYTORCH_NPU_STREAM
  RECORD_FUNCTION("dgl::CSRGetDataAndIndices[Ascend]", std::vector<c10::IValue>());
#endif
  const auto ctx = rows->ctx;
  CHECK_EQ(ctx.device_type, kDGLAscend);
  CHECK_EQ(csr.indptr->ctx, ctx); CHECK_EQ(csr.indices->ctx, ctx); CHECK_EQ(cols->ctx, ctx);
  CHECK_EQ(rows->ndim, 1); CHECK_EQ(cols->ndim, 1);
  const int64_t nr = rows->shape[0], nc = cols->shape[0];
  CHECK(nr == nc || nr == 1 || nc == 1) << "Invalid row and col id array.";
  // The CPU loop also produces no queries when one broadcast operand is empty.
  const int64_t queries = nr == 0 || nc == 0 ? 0 : std::max(nr, nc);
  auto empty = [&]() {
    return std::vector<NDArray>{NewAllArray(0, rows->dtype, ctx),
                               NewAllArray(0, rows->dtype, ctx),
                               NewAllArray(0, rows->dtype, ctx)};
  };
  if (!queries) return empty();
  int32_t current = -1; ALL_ACL(aclrtGetDevice(&current));
  if (current != ctx.device_id) ALL_ACL(aclrtSetDevice(ctx.device_id));
  const auto stream = runtime::getCurrentAscendStream();
  const uint32_t cores = AllCores(ctx, queries);
  constexpr uint32_t record_words = 2 + 2 * (sizeof(IdType) / 4);
  CHECK_LE(queries, std::numeric_limits<int64_t>::max() / record_words)
      << "CSRGetDataAndIndices workspace size overflow";
  const DGLDataType u64{kDGLInt, 64, 1};
  auto records = NewAllArray(queries * record_words, u64, ctx);
  auto stats = NewAllArray((cores + 1) * 4, u64, ctx);
  std::vector<NDArray> result;
  Completion completion(stream);
  static thread_local HostInfo host;
  completion.pending = true;
#define COUNT(SUFFIX) \
  ALL_ACL(aclrtlaunch_csr_all_count_##SUFFIX(cores, stream, Data(csr.indptr), \
      Data(csr.indices), Data(rows), Data(cols), Data(records), Data(stats), \
      csr.num_rows, csr.num_cols, queries, nr == 1 ? 0 : 1, nc == 1 ? 0 : 1, csr.sorted ? 1 : 0))
  if constexpr (sizeof(IdType) == 4) { COUNT(int32); } else { COUNT(int64); }
#undef COUNT
  ALL_ACL(aclrtlaunch_csr_all_prefix(1, stream, Data(stats), cores));
  ALL_ACL(aclrtMemcpyAsync(host.data, 32,
      static_cast<uint64_t*>(Data(stats)) + cores * 4, 32,
      ACL_MEMCPY_DEVICE_TO_HOST, stream));
  completion.Wait();
  const uint64_t total = host.data[0], error = host.data[1];
  if (error) {
    const bool bad_row = error & 1;
    const NDArray& ids = bad_row ? rows : cols;
    const uint64_t query = (error - 1) / 2;
    IdType value = 0;
    ALL_ACL(aclrtMemcpy(&value, sizeof(value),
        static_cast<const IdType*>(Data(ids)) + (ids->shape[0] == 1 ? 0 : query),
        sizeof(value), ACL_MEMCPY_DEVICE_TO_HOST));
    LOG(FATAL) << (bad_row ? "Invalid row index: " : "Invalid col index: ") << value;
  }
  CHECK_LE(total, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      << "CSRGetDataAndIndices output size overflow";
  if (!total) return empty();
  for (int i = 0; i < 3; ++i) result.push_back(NewAllArray(total, rows->dtype, ctx));
  completion.pending = true;
#define EMIT(SUFFIX) \
  ALL_ACL(aclrtlaunch_csr_all_emit_##SUFFIX(cores, stream, Data(csr.indices), \
      CSRHasData(csr) ? Data(csr.data) : Data(csr.indices), Data(rows), Data(cols), \
      Data(records), Data(stats), Data(result[0]), Data(result[1]), Data(result[2]), \
      queries, nr == 1 ? 0 : 1, nc == 1 ? 0 : 1, CSRHasData(csr) ? 1 : 0, csr.sorted ? 1 : 0))
  if constexpr (sizeof(IdType) == 4) { EMIT(int32); } else { EMIT(int64); }
#undef EMIT
  completion.Wait();
  return result;
#else
  LOG(FATAL) << "Ascend support is not compiled."; return {};
#endif
}
template std::vector<NDArray> CSRGetDataAndIndices<kDGLAscend, int32_t>(CSRMatrix, NDArray, NDArray);
template std::vector<NDArray> CSRGetDataAndIndices<kDGLAscend, int64_t>(CSRMatrix, NDArray, NDArray);
}  // namespace impl

// Operator-scoped diagnostic entry for sorted metadata and standalone CSR
// arrays. Public graph APIs continue to call the normal DGL dispatch above.
DGL_REGISTER_GLOBAL("ascend._CAPI_DGLCSRGetDataAndIndices")
.set_body([](runtime::DGLArgs args, runtime::DGLRetValue* rv) {
  SparseMatrixRef matrix = args[0];
  CHECK_EQ(matrix->format, static_cast<int32_t>(SparseFormat::kCSR));
  CSRMatrix csr(matrix->num_rows, matrix->num_cols, matrix->indices[0],
                matrix->indices[1], matrix->indices[2], matrix->flags[0]);
  auto result = aten::CSRGetDataAndIndices(csr, args[1], args[2]);
  *rv = runtime::PackedFunc([result](runtime::DGLArgs item, runtime::DGLRetValue* out) {
    const int index = item[0]; CHECK_GE(index, 0); CHECK_LT(index, 3);
    *out = result[index];
  });
});
}  // namespace aten
}  // namespace dgl
