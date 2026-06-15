/**
 * @file runtime.cpp
 * @brief Implementation of the lite inference runtime
 */

#include "tenzor/lite/runtime.hpp"
#include "tenzor/lite/lite_graph.hpp"
#include "tenzor/lite/memory_planner.hpp"
#include "tenzor/lite/model_format.hpp"
#include "tenzor/lite/tensor_bridge.hpp"
#include <cerrno>

// Inf-E4: POSIX mmap for load_mmap(path). On non-POSIX platforms the
// implementation falls back to the heap-buffered load(path).
#if defined(__unix__) || defined(__APPLE__)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define TENZOR_HAS_POSIX_MMAP 1
#endif
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/core/tensor.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <optional>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace tenzor {
namespace lite {

// ============================================================================
// LiteTensor
// ============================================================================

LiteTensor::~LiteTensor() {
    if (owns_data && data) {
        std::free(data);
        data = nullptr;
    }
}

auto LiteTensor::numel() const -> int64_t {
    // Element count is the product of the dimension extents. A rank-0 scalar has
    // exactly one element (the empty product over zero dims is 1) — consistent
    // with to_lite_tensor() in tensor_bridge.cpp, which allocates real data for a
    // numel==1 scalar. Do NOT special-case ndim == 0 to 0: that under-sizes
    // nbytes() to zero and drops scalar values. A genuinely empty tensor is one
    // with a zero-sized dimension (product == 0).
    int64_t n = 1;
    for (int32_t i = 0; i < ndim; ++i) {
        n *= shape[i];
    }
    return n;
}

auto LiteTensor::nbytes() const -> int64_t {
    return numel() * dtype_size(dtype);
}

// ============================================================================
// LiteAllocator
// ============================================================================

LiteAllocator::LiteAllocator(const std::vector<size_t>& pool_sizes, size_t alignment)
    : pool_sizes_(pool_sizes), alignment_(alignment) {
    pools_.reserve(pool_sizes.size());
    for (auto size : pool_sizes) {
        void* ptr = nullptr;
        if (size > 0) {
            // L15 fix: capture errno from posix_memalign separately from
            // its return code so the caller can distinguish EINVAL
            // (bad alignment) from ENOMEM (out-of-memory).
            int errcode = 0;
#ifdef _WIN32
            ptr = _aligned_malloc(size, alignment);
            if (!ptr) errcode = errno;
#else
            errcode = posix_memalign(&ptr, alignment, size);
            if (errcode != 0) {
                ptr = nullptr;
            }
#endif
            if (!ptr) {
                // Clean up already allocated pools
                for (auto p : pools_) {
#ifdef _WIN32
                    _aligned_free(p);
#else
                    std::free(p);
#endif
                }
                const char* reason = (errcode == EINVAL)
                    ? "bad alignment (must be power of two and multiple of sizeof(void*))"
                    : (errcode == ENOMEM ? "out of memory" : "unknown allocator error");
                throw std::runtime_error(
                    std::string("LiteAllocator: failed to allocate ") +
                    std::to_string(size) + " bytes (alignment=" +
                    std::to_string(alignment) + "): " + reason);
            }
            std::memset(ptr, 0, size);
        }
        pools_.push_back(ptr);
        total_bytes_ += size;
    }
}

LiteAllocator::~LiteAllocator() {
    for (auto ptr : pools_) {
        if (ptr) {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
    }
}

auto LiteAllocator::get_buffer(size_t buffer_id, size_t offset) -> void* {
    if (buffer_id >= pools_.size()) {
        throw std::out_of_range("LiteAllocator: buffer_id out of range");
    }
    return static_cast<char*>(pools_[buffer_id]) + offset;
}

// ============================================================================
// LiteRuntime implementation
// ============================================================================

struct LiteRuntime::Impl {
    std::unique_ptr<LiteAllocator> allocator;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::unordered_map<std::string, std::string> metadata;

    LiteGraph graph;

    // Phase 2: parsed model artefacts.
    //   - tensor_values[tid] is the TVAL entry for tensor_id `tid`. Entries
    //     omitted from TVAL get a default Intermediate slot here.
    //   - weight_blob holds the raw bytes of the WGTS section (owned). Weight
    //     tensors are reconstructed as Tensor views over this buffer per
    //     forward() call. Phase 5 can swap this for an mmap-backed span.
    std::vector<TensorValue> tensor_values_by_id;  ///< indexed by tensor_id
    std::vector<uint8_t> weight_blob;

    // Inf-E1: parsed MMPL plan, retained so the loaded placement layout is
    // available to (and validated against) the runtime rather than computed
    // and discarded. Empty when the file shipped no MMPL section.
    std::optional<MmplPlan> memory_plan;
};

LiteRuntime::~LiteRuntime() = default;

auto LiteRuntime::load(const std::string& path) -> std::unique_ptr<LiteRuntime> {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("LiteRuntime: cannot open file: " + path);
    }

    // Read file into memory
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    // tellg() returns -1 on failure; converting that to size_t would request a
    // ~SIZE_MAX allocation, so fail cleanly instead.
    if (size < 0) {
        throw std::runtime_error("LiteRuntime: cannot determine file size: " +
                                 path);
    }

    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (size > 0 &&
        !file.read(reinterpret_cast<char*>(data.data()),
                   static_cast<std::streamsize>(size))) {
        throw std::runtime_error("LiteRuntime: cannot read file: " + path);
    }

    return load(data.data(), data.size());
}

namespace {

// Reshape a flat `LoadedModel::tensor_values` into an indexed table sized
// by max_tensor_id + 1, so the runtime can do O(1) lookup. Unfilled slots
// keep their default-constructed (Intermediate) TensorValue.
auto index_tensor_values(const std::vector<TensorValue>& tvs)
    -> std::vector<TensorValue> {
    // tensor_id is a signed int16_t read verbatim from an untrusted file.
    // A negative id would wrap the static_cast<size_t> below into a huge value
    // (OOB write); if every id were negative, max_id would stay -1, sizing the
    // table to 0 and making any write OOB. Validate up front.
    int16_t max_id = -1;
    for (const auto& tv : tvs) {
        if (tv.tensor_id < 0) {
            throw std::runtime_error(
                "LiteRuntime: negative tensor_id (" +
                std::to_string(tv.tensor_id) + ") in model");
        }
        max_id = std::max(max_id, tv.tensor_id);
    }
    if (max_id < 0) {
        // No tensor values: return an empty table.
        return {};
    }
    std::vector<TensorValue> out(static_cast<size_t>(max_id) + 1);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i].tensor_id = static_cast<int16_t>(i);
    }
    for (const auto& tv : tvs) {
        out[static_cast<size_t>(tv.tensor_id)] = tv;
    }
    return out;
}

// Walk the graph and verify every node's OpId is registered on the chosen
// backend. Throws with a deduplicated, actionable list of missing OpIds if
// any are absent. (Phase 5 backend coverage check — promoted from a future
// runtime crash into a clean load-time error.)
auto verify_op_coverage(const LiteGraph& graph, Device::Type device) -> void {
    std::unordered_set<uint16_t> missing;
    for (const auto& node : graph.nodes()) {
        if (!tenzor::is_op_supported(node.op, device)) {
            missing.insert(static_cast<uint16_t>(node.op));
        }
    }
    if (missing.empty()) return;

    std::vector<uint16_t> sorted(missing.begin(), missing.end());
    std::sort(sorted.begin(), sorted.end());

    std::ostringstream oss;
    oss << "LiteRuntime::load: " << sorted.size()
        << " OpId(s) used by this graph are not registered on backend "
        << tenzor::device_type_to_string(device) << ": [";
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (i) oss << ", ";
        oss << sorted[i];
    }
    oss << "]. Either pick a backend that supports them, or rebuild Tenzor "
           "with the missing kernels enabled.";
    throw std::runtime_error(oss.str());
}

}  // namespace

auto LiteRuntime::load(const void* data, size_t size) -> std::unique_ptr<LiteRuntime> {
    if (!data || size == 0) {
        throw std::runtime_error("LiteRuntime: empty model data");
    }

    auto runtime = std::unique_ptr<LiteRuntime>(new LiteRuntime());
    runtime->impl_ = std::make_unique<Impl>();

    auto model = TZLiteReader::load_full(data, size);

    // Phase 5: verify CPU coverage at load time so an unsupported op fails
    // here rather than mid-forward(). Multi-backend selection is the natural
    // next step (caller-supplied Device), gated behind a future API change.
    verify_op_coverage(*model.graph, Device::Type::CPU);

    runtime->impl_->graph              = std::move(*model.graph);
    runtime->impl_->tensor_values_by_id = index_tensor_values(model.tensor_values);
    runtime->impl_->weight_blob        = std::move(model.weight_blob);
    runtime->impl_->metadata           = std::move(model.metadata);

    // Populate input/output shape metadata from TVAL where available.
    for (int16_t tid : model.input_ids) {
        if (static_cast<size_t>(tid) < runtime->impl_->tensor_values_by_id.size()) {
            const auto& tv = runtime->impl_->tensor_values_by_id[tid];
            runtime->impl_->input_shapes.push_back(tv.shape);
        } else {
            runtime->impl_->input_shapes.emplace_back();
        }
    }
    // Output shapes are typically unknown without shape inference; Phase 3
    // populates them at export time.

    // Inf-E4: when the file ships with an MMPL section, size the arena
    // exactly from the precomputed plan. v1 files (no MMPL) keep the
    // pre-Inf-E behaviour: kernels allocate their own outputs from heap.
    //
    // NOTE: LiteGraph::execute does NOT yet place node outputs into this
    // arena — kernels still heap-allocate their own outputs via global
    // dispatch (a true arena-backed execution path requires the dispatch
    // layer to write into caller-provided buffers, tracked separately). To
    // ensure the plan is not silently computed-and-ignored, the parsed plan
    // is retained on Impl and validated here against the graph: every
    // placement must reference a real pool and lie within its bounds, so a
    // corrupt or mismatched MMPL fails loudly at load rather than being
    // dead weight.
    if (model.memory_plan) {
        const MmplPlan& plan = *model.memory_plan;
        for (const auto& p : plan.placements) {
            if (static_cast<size_t>(p.pool_index) >= plan.pool_sizes.size()) {
                throw std::runtime_error(
                    "LiteRuntime::load: MMPL placement for tensor_id " +
                    std::to_string(p.tensor_id) +
                    " references pool_index " + std::to_string(p.pool_index) +
                    " but the plan declares only " +
                    std::to_string(plan.pool_sizes.size()) + " pool(s)");
            }
            const uint64_t pool_bytes = plan.pool_sizes[p.pool_index];
            if (p.offset > pool_bytes) {
                throw std::runtime_error(
                    "LiteRuntime::load: MMPL placement for tensor_id " +
                    std::to_string(p.tensor_id) + " offset " +
                    std::to_string(p.offset) + " exceeds pool size " +
                    std::to_string(pool_bytes));
            }
        }

        std::vector<size_t> pool_sizes;
        pool_sizes.reserve(plan.pool_sizes.size());
        for (uint64_t s : plan.pool_sizes) {
            pool_sizes.push_back(static_cast<size_t>(s));
        }
        runtime->impl_->allocator = std::make_unique<LiteAllocator>(
            pool_sizes, static_cast<size_t>(plan.alignment));
        runtime->impl_->memory_plan = plan;
    } else {
        runtime->impl_->allocator = std::make_unique<LiteAllocator>(
            std::vector<size_t>{}, 64);
    }

    return runtime;
}

// Inf-E4: POSIX mmap variant. Maps the file into the address space,
// passes the data range to `load(data, size)` which constructs the
// runtime + copies WGTS bytes into the heap-backed `weight_blob`, then
// unmaps the source mapping.
//
// This implementation is behaviour-identical to `load(path)` — the mmap
// is short-lived because `load(data, size)` materialises a heap copy.
// True zero-copy WGTS (mmap-backed views feeding `Tensor::from_blob`
// with a no-op deleter) requires refactoring `LoadedModel::weight_blob`
// from `std::vector<uint8_t>` to a non-owning span, which is a deeper
// change tracked separately.
auto LiteRuntime::load_mmap(const std::string& path)
    -> std::unique_ptr<LiteRuntime> {
#ifdef TENZOR_HAS_POSIX_MMAP
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("LiteRuntime::load_mmap: cannot open file: " + path);
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        throw std::runtime_error("LiteRuntime::load_mmap: fstat failed on: " + path);
    }
    void* addr = ::mmap(nullptr, static_cast<size_t>(st.st_size),
                        PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // fd is no longer needed once mapped
    if (addr == MAP_FAILED) {
        // Fall back to heap-buffered load on mmap failure (e.g. EOVERFLOW
        // on 32-bit systems, or filesystem doesn't support mmap).
        return load(path);
    }
    std::unique_ptr<LiteRuntime> rt;
    try {
        rt = load(addr, static_cast<size_t>(st.st_size));
    } catch (...) {
        ::munmap(addr, static_cast<size_t>(st.st_size));
        throw;
    }
    ::munmap(addr, static_cast<size_t>(st.st_size));
    return rt;
#else
    // No POSIX mmap on this platform — fall back transparently.
    return load(path);
#endif
}

auto LiteRuntime::from_graph(LiteGraph graph) -> std::unique_ptr<LiteRuntime> {
    auto runtime = std::unique_ptr<LiteRuntime>(new LiteRuntime());
    runtime->impl_ = std::make_unique<Impl>();
    runtime->impl_->graph = std::move(graph);
    // Match declared graph I/O shape metadata if present (currently empty for
    // hand-built graphs — Phase 3 exporter fills these from JIT shape info).
    runtime->impl_->allocator = std::make_unique<LiteAllocator>(
        std::vector<size_t>{}, 64);
    return runtime;
}

auto LiteRuntime::forward(const std::vector<LiteTensor>& inputs) -> std::vector<LiteTensor> {
    if (!impl_) {
        throw std::runtime_error("LiteRuntime: forward() called on uninitialised runtime");
    }
    if (impl_->graph.num_nodes() == 0 && impl_->graph.input_ids().empty()) {
        throw std::runtime_error(
            "LiteRuntime::forward: graph is empty. "
            "Either build a runtime via LiteRuntime::from_graph(...) or load a "
            "real .tzlite file with at least one node.");
    }

    // Build non-owning Tensor views over the weight blob for every tensor_id
    // marked as a Weight in the TVAL table. The Tensor's Storage references
    // weight_blob with a no-op deleter, so the LiteRuntime's owned bytes
    // remain the source of truth.
    std::unordered_map<int16_t, Tensor> constants;
    if (!impl_->tensor_values_by_id.empty() && !impl_->weight_blob.empty()) {
        for (const auto& tv : impl_->tensor_values_by_id) {
            if (tv.source != TensorSource::Weight) continue;
            // Both operands are uint64_t read verbatim from an untrusted file;
            // the 64-bit add `offset + nbytes` can wrap and pass a naive check,
            // then `data() + offset` yields a wild OOB pointer. Use the
            // overflow-safe form (matching read_pod in model_format.cpp).
            const size_t blob_size = impl_->weight_blob.size();
            if (tv.weight_offset > blob_size ||
                tv.weight_nbytes > blob_size - tv.weight_offset) {
                throw std::runtime_error(
                    "LiteRuntime::forward: weight for tensor_id " +
                    std::to_string(tv.tensor_id) +
                    " refers to bytes past end of WGTS payload");
            }
            void* data = impl_->weight_blob.data() + tv.weight_offset;
            constants.emplace(
                tv.tensor_id,
                Tensor::from_blob(data,
                                  tv.shape,
                                  tv.dtype,
                                  Device::cpu(),
                                  /*deleter=*/[](void*) noexcept {}));
        }
    }
    return impl_->graph.execute(inputs, constants);
}

auto LiteRuntime::forward(const LiteTensor& input) -> LiteTensor {
    // Build a non-owning view of `input` so the per-call destructor of the
    // vector element doesn't free the user's buffer when ins goes out of
    // scope. (LiteTensor's destructor frees iff owns_data is true.)
    LiteTensor view;
    view.data    = input.data;
    view.shape   = input.shape;
    view.strides = input.strides;
    view.ndim    = input.ndim;
    view.dtype   = input.dtype;
    view.owns_data = false;
    std::vector<LiteTensor> ins;
    ins.push_back(std::move(view));

    auto outs = forward(ins);
    if (outs.size() != 1) {
        throw std::runtime_error(
            "LiteRuntime::forward(single): graph produced " +
            std::to_string(outs.size()) + " outputs");
    }
    return std::move(outs.front());
}

auto LiteRuntime::input_shapes() const -> std::vector<std::vector<int64_t>> {
    return impl_ ? impl_->input_shapes : std::vector<std::vector<int64_t>>{};
}

auto LiteRuntime::output_shapes() const -> std::vector<std::vector<int64_t>> {
    return impl_ ? impl_->output_shapes : std::vector<std::vector<int64_t>>{};
}

auto LiteRuntime::model_metadata(const std::string& key) const -> std::string {
    if (!impl_) return "";
    auto it = impl_->metadata.find(key);
    return it != impl_->metadata.end() ? it->second : "";
}

auto LiteRuntime::create_input(const std::vector<int64_t>& shape, DType dtype) -> LiteTensor {
    // Reject (don't silently truncate) shapes with more than kMaxDims dims: a
    // truncated tensor's numel no longer matches the caller's intent and would
    // under-size the buffer, letting downstream kernels read past it. Mirrors
    // to_lite_tensor()'s throw-on-too-many-dims contract.
    if (shape.size() > static_cast<size_t>(kMaxDims)) {
        throw std::invalid_argument(
            "LiteRuntime::create_input: shape has more dims than kMaxDims");
    }

    LiteTensor tensor;
    tensor.ndim = static_cast<int32_t>(shape.size());
    tensor.dtype = dtype;
    tensor.owns_data = true;

    int64_t numel = 1;
    for (int32_t i = 0; i < tensor.ndim; ++i) {
        tensor.shape[i] = shape[i];
        numel *= shape[i];
    }

    // Compute strides (row-major)
    for (int32_t i = tensor.ndim - 1; i >= 0; --i) {
        tensor.strides[i] = (i == tensor.ndim - 1) ? 1 : tensor.strides[i + 1] * tensor.shape[i + 1];
    }

    auto bytes = numel * dtype_size(dtype);
    tensor.data = std::calloc(1, static_cast<size_t>(bytes));
    // A failed/over-large allocation otherwise yields a LiteTensor with
    // data==nullptr and owns_data=true that callers write into (null deref).
    if (bytes > 0 && tensor.data == nullptr) {
        tensor.owns_data = false;  // nothing to free in the destructor
        throw std::bad_alloc{};
    }

    return tensor;
}

} // namespace lite
} // namespace tenzor
