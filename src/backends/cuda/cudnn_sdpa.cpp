/**
 * @file cudnn_sdpa.cpp
 * @brief cuDNN Graph API SDPA (Scaled Dot-Product Attention) wrapper
 *
 * Uses cuDNN Frontend library for optimized fused attention with Tensor Core support.
 * Requires cuDNN 9.0+ and cuDNN Frontend.
 */

#ifdef TENZOR_HAS_CUDNN_FRONTEND

#include <cudnn_frontend.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <list>
#include <mutex>

#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/cuda_caching_allocator.hpp"
#include "tenzor/backend/fused_ops.hpp"
#include "kernels/cuda_launch_utils.cuh"  // tenzor::cuda::CudaDeviceGuard

namespace fe = cudnn_frontend;

// cuDNN error checking macro
#ifndef CUDNN_CHECK
#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        throw std::runtime_error( \
            std::string("cuDNN error: ") + cudnnGetErrorString(status) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__) \
        ); \
    } \
} while(0)
#endif

// CUDA runtime error checking macro (local to this TU).
#ifndef TENZOR_SDPA_CUDA_CHECK
#define TENZOR_SDPA_CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error( \
            std::string("CUDA error: ") + cudaGetErrorString(err) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__) \
        ); \
    } \
} while(0)
#endif

namespace tenzor {
namespace cuda {

namespace {

// ============================================================================
// Singleton cuDNN Handle for SDPA operations
// ============================================================================

// A cuDNN handle is bound to the CUDA device that is current at cudnnCreate()
// time, and its internal scratch lives on that device. A single global handle
// created on device 0 is therefore unusable for SDPA that runs on device 1+.
// We keep one handle per device (lazily created with that device made current),
// so the handle, its workspace, and the SDPA tensors all share a device.
class SDPACuDNNHandle {
public:
    // Returns the cuDNN handle for `device`, creating it (with `device` made
    // current) on first use. The caller is responsible for restoring/leaving
    // the desired current device; cudnn_sdpa_forward sets the device explicitly.
    static cudnnHandle_t get(int device) {
        static thread_local std::unordered_map<int, SDPACuDNNHandle> handles;
        auto it = handles.find(device);
        if (it == handles.end()) {
            it = handles.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(device),
                                 std::forward_as_tuple(device)).first;
        }
        return it->second.handle_;
    }

    static void set_stream(int device, cudaStream_t stream) {
        CUDNN_CHECK(cudnnSetStream(get(device), stream));
    }

    SDPACuDNNHandle(const SDPACuDNNHandle&) = delete;
    SDPACuDNNHandle& operator=(const SDPACuDNNHandle&) = delete;
    SDPACuDNNHandle(SDPACuDNNHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    explicit SDPACuDNNHandle(int device) {
        int prev = 0;
        TENZOR_SDPA_CUDA_CHECK(cudaGetDevice(&prev));
        TENZOR_SDPA_CUDA_CHECK(cudaSetDevice(device));
        CUDNN_CHECK(cudnnCreate(&handle_));
        TENZOR_SDPA_CUDA_CHECK(cudaSetDevice(prev));
    }

    ~SDPACuDNNHandle() {
        if (handle_) {
            cudnnDestroy(handle_);
        }
    }

private:
    cudnnHandle_t handle_ = nullptr;
};

// ============================================================================
// Cache for SDPA graph plans to avoid rebuilding for same shapes
// ============================================================================
struct SDPACacheKey {
    int device;        // cuDNN frontend execution plans are bound to the handle/
                       // device they were built on; key per device so a graph
                       // built for device 0 is never executed on device 1's handle.
    int64_t batch;
    int64_t num_heads;
    int64_t seq_len_q;
    int64_t seq_len_k;
    int64_t head_dim;
    DType dtype;
    float scale;
    bool causal;       // M4: causal flag is part of the graph build, must key the cache.

    bool operator==(const SDPACacheKey& other) const {
        return device == other.device &&
               batch == other.batch && num_heads == other.num_heads &&
               seq_len_q == other.seq_len_q && seq_len_k == other.seq_len_k &&
               head_dim == other.head_dim && dtype == other.dtype &&
               scale == other.scale && causal == other.causal;
    }
};

struct SDPACacheKeyHash {
    size_t operator()(const SDPACacheKey& k) const {
        // Simple hash combining device, all dimensions, dtype, scale, and causal.
        size_t h = std::hash<int>{}(k.device);
        h ^= std::hash<int64_t>{}(k.batch) << 1;
        h ^= std::hash<int64_t>{}(k.num_heads) << 2;
        h ^= std::hash<int64_t>{}(k.seq_len_q) << 3;
        h ^= std::hash<int64_t>{}(k.seq_len_k) << 4;
        h ^= std::hash<int64_t>{}(k.head_dim) << 5;
        h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.dtype)) << 6;
        h ^= std::hash<float>{}(k.scale) << 7;
        h ^= std::hash<bool>{}(k.causal) << 8;
        return h;
    }
};

struct SDPACacheEntry {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace_size;
};

class SDPAGraphCache {
public:
    static SDPAGraphCache& instance() {
        static SDPAGraphCache cache;
        return cache;
    }

    bool get(const SDPACacheKey& key, SDPACacheEntry& entry) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Move to front of LRU list (most recently used)
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iter);
            entry.graph = it->second.graph;
            entry.workspace_size = it->second.workspace_size;
            return true;
        }
        return false;
    }

    void set(const SDPACacheKey& key, const SDPACacheEntry& entry) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Update existing entry and move to front
            it->second.graph = entry.graph;
            it->second.workspace_size = entry.workspace_size;
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iter);
            return;
        }
        // Evict LRU entry if at capacity
        if (cache_.size() >= kMaxCacheSize) {
            auto& lru_key = lru_list_.back();
            cache_.erase(lru_key);
            lru_list_.pop_back();
        }
        // Insert new entry at front of LRU list
        lru_list_.push_front(key);
        CacheValue val;
        val.graph = entry.graph;
        val.workspace_size = entry.workspace_size;
        val.lru_iter = lru_list_.begin();
        cache_[key] = std::move(val);
    }

private:
    SDPAGraphCache() = default;
    static constexpr size_t kMaxCacheSize = 64;

    struct CacheValue {
        std::shared_ptr<fe::graph::Graph> graph;
        int64_t workspace_size;
        std::list<SDPACacheKey>::iterator lru_iter;
    };

    std::unordered_map<SDPACacheKey, CacheValue, SDPACacheKeyHash> cache_;
    std::list<SDPACacheKey> lru_list_;
    std::mutex mutex_;
};

// Per-thread workspace buffer — each thread owns its own allocation,
// eliminating the data race where one thread frees the buffer while another
// thread's kernel is still using it. Routes through the caching allocator
// to avoid raw cudaMalloc/cudaFree overhead.
//
// The workspace MUST live on the same CUDA device as the SDPA tensors. The
// caching allocator defaults to device 0, so on a multi-GPU run where attention
// executes on a non-zero device, a device-0 workspace handed to cuDNN's
// graph->execute() causes an illegal access / silently wrong output. We
// therefore key the per-thread buffer by device and (re)allocate on the
// requested device whenever the device or size changes.
class SDPAWorkspace {
public:
    static void* get(size_t required_size, int device) {
        static thread_local SDPAWorkspace instance;
        if (device != instance.device_ || required_size > instance.size_) {
            instance.resize(required_size, device);
        }
        return instance.buffer_;
    }

private:
    SDPAWorkspace() : buffer_(nullptr), size_(0), device_(-1) {}
    ~SDPAWorkspace() {
        if (buffer_) {
            backend::CachingAllocator::get().free(buffer_, device_);
        }
    }

    void resize(size_t new_size, int device) {
        if (buffer_) {
            backend::CachingAllocator::get().free(buffer_, device_);
            buffer_ = nullptr;
        }
        device_ = device;
        size_ = new_size + (new_size / 4);  // 25% headroom
        buffer_ = backend::CachingAllocator::get().allocate(size_, device_);
    }

    void* buffer_;
    size_t size_;
    int device_;
};

// Unique IDs for graph tensors
enum class TensorUID : int64_t {
    Q = 1,
    K = 2,
    V = 3,
    O = 4,
    Stats = 5,
    Scale = 6
};

// ============================================================================
// Capability cache: remembers (dtype, head_dim, sm_arch) tuples that cuDNN
// reports as unsupported on the current device. Avoids paying the validate /
// build / check_support cost on every call when we know cuDNN won't accept
// the configuration (e.g. on a brand new GPU arch with limited SDPA tuning).
// ============================================================================
class SDPACapCache {
public:
    static SDPACapCache& instance() {
        static SDPACapCache cache;
        return cache;
    }

    // Returns true if this EXACT shape (full SDPACacheKey) is known unsupported.
    // The negative cache is keyed by the full shape/scale/causal key rather than
    // by the coarse {dtype, head_dim, sm} triple: create_sdpa_graph can throw for
    // reasons specific to a single shape (heuristic/build/check_support), and
    // poisoning every shape sharing the triple would needlessly force otherwise-
    // acceptable shapes onto the slower custom-flash fallback for the process
    // lifetime.
    bool is_known_unsupported(const SDPACacheKey& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = unsupported_.find(key);
        return it != unsupported_.end();
    }

    void mark_unsupported(const SDPACacheKey& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        unsupported_.insert(key);
    }

private:
    std::unordered_set<SDPACacheKey, SDPACacheKeyHash> unsupported_;
    std::mutex mutex_;
};

namespace {
// Compute capability (major*10 + minor) of a SPECIFIC device. The architecture
// gate below must reflect the GPU the tensors actually live on (Q.device().index),
// NOT the calling thread's current device — on a heterogeneous multi-GPU host
// (mixed Blackwell + non-Blackwell) the two can differ, which would mis-evaluate
// the gate (either spurious BMM fallback or reintroducing the Blackwell FP32
// illegal-memory-access). Callers must pass the tensor's device explicitly.
inline int sm_arch_of_device(int device) {
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, device);
    return props.major * 10 + props.minor;
}

inline fe::DataType_t to_cudnn_dtype(DType dt) {
    switch (dt) {
        case DType::Float16:  return fe::DataType_t::HALF;
        case DType::BFloat16: return fe::DataType_t::BFLOAT16;
        case DType::Float32:  return fe::DataType_t::FLOAT;
        default: throw std::runtime_error("cudnn_sdpa: unsupported dtype");
    }
}

// Fallback path for dtypes / shapes / archs cuDNN can't handle on this device.
// Reshapes 4D BHSD inputs to 3D (B*H, S, D) for the custom flash kernel.
// Honors causal so the fallback produces contract-equivalent output to the
// cuDNN graph path (audit C4 — cuDNN call had no causal, fallback didn't either).
std::pair<Tensor, Tensor> fused_attention_fallback(const Tensor& Q, const Tensor& K, const Tensor& V, float scale, bool causal) {
    auto q_shape = Q.shape();
    bool is_4d = (q_shape.size() == 4);
    Tensor Q3 = Q, K3 = K, V3 = V;
    if (is_4d) {
        int64_t b = q_shape[0], h = q_shape[1], sq = q_shape[2], d = q_shape[3];
        int64_t sk = K.shape()[2];
        Q3 = Q.reshape({b * h, sq, d});
        K3 = K.reshape({b * h, sk, d});
        V3 = V.reshape({b * h, sk, d});
    }
    auto [output, lse] = fused_attention_cuda(Q3, K3, V3, scale, causal, 0.0f, 0u);
    if (is_4d) {
        output = output.reshape({q_shape[0], q_shape[1], q_shape[2], q_shape[3]});
        lse = lse.reshape({q_shape[0], q_shape[1], q_shape[2]});
    }
    return {output, lse};
}
}  // namespace

auto create_sdpa_graph(
    cudnnHandle_t handle,
    int64_t batch,
    int64_t num_heads,
    int64_t seq_len_q,
    int64_t seq_len_k,
    int64_t head_dim,
    float attn_scale,
    fe::DataType_t io_dtype,
    bool causal
) -> std::pair<std::shared_ptr<fe::graph::Graph>, int64_t> {

    auto graph = std::make_shared<fe::graph::Graph>();

    // Set data types: caller-supplied I/O dtype (HALF / BFLOAT16 / FLOAT),
    // FP32 intermediate + compute. cuDNN runs SDPA with FP32 accumulation
    // regardless of I/O dtype, so numerical stability is the same across
    // dtypes -- only the memory bandwidth changes.
    graph->set_io_data_type(io_dtype)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // Define tensor dimensions [batch, num_heads, seq_len, head_dim]
    std::vector<int64_t> q_dim = {batch, num_heads, seq_len_q, head_dim};
    std::vector<int64_t> k_dim = {batch, num_heads, seq_len_k, head_dim};
    std::vector<int64_t> v_dim = {batch, num_heads, seq_len_k, head_dim};
    std::vector<int64_t> o_dim = {batch, num_heads, seq_len_q, head_dim};

    // Compute strides for BHSD layout (batch, heads, sequence, head_dim)
    auto compute_strides = [](const std::vector<int64_t>& dim) {
        std::vector<int64_t> strides(4);
        strides[3] = 1;                                     // head_dim stride
        strides[2] = dim[3];                                // seq stride = head_dim
        strides[1] = dim[2] * strides[2];                   // head stride = seq * head_dim
        strides[0] = dim[1] * strides[1];                   // batch stride
        return strides;
    };

    auto q_stride = compute_strides(q_dim);
    auto k_stride = compute_strides(k_dim);
    auto v_stride = compute_strides(v_dim);
    auto o_stride = compute_strides(o_dim);

    // Create Q, K, V input tensors
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("Q")
        .set_uid(static_cast<int64_t>(TensorUID::Q))
        .set_dim(q_dim)
        .set_stride(q_stride)
        .set_data_type(io_dtype));

    auto K = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("K")
        .set_uid(static_cast<int64_t>(TensorUID::K))
        .set_dim(k_dim)
        .set_stride(k_stride)
        .set_data_type(io_dtype));

    auto V = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("V")
        .set_uid(static_cast<int64_t>(TensorUID::V))
        .set_dim(v_dim)
        .set_stride(v_stride)
        .set_data_type(io_dtype));

    // Configure SDPA attributes. Per docs/internals/attention-contract.md, the
    // causal flag must apply *before* softmax (cuDNN does this internally via
    // its mask-fusion path). Previously this was hardcoded false, so any
    // caller asking for causal MHA on CUDA in eval mode silently got
    // non-causal output (audit C4/M5).
    //
    // set_generate_stats(true): per docs/internals/attention-contract.md,
    // FusedAttention's Outputs=(output,logsumexp) — Stats IS that logsumexp
    // tensor. It was previously disabled, silently violating the contract
    // and disabling the fast fused-kernel backward path for the cuDNN SDPA
    // path specifically.
    auto sdpa_options = fe::graph::SDPA_attributes()
        .set_name("flash_attention")
        .set_generate_stats(true)
        .set_attn_scale(attn_scale)
        .set_causal_mask(causal);

    // Create SDPA operation
    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Mark output tensor
    O->set_output(true)
     .set_uid(static_cast<int64_t>(TensorUID::O))
     .set_dim(o_dim)
     .set_stride(o_stride)
     .set_data_type(io_dtype);

    // Mark Stats (logsumexp) tensor. cuDNN's SDPA Stats output is shaped
    // [batch, num_heads, seq_len_q, 1] and always FP32 regardless of I/O
    // dtype (the softmax accumulation dtype).
    std::vector<int64_t> stats_dim = {batch, num_heads, seq_len_q, 1};
    std::vector<int64_t> stats_stride = {num_heads * seq_len_q, seq_len_q, 1, 1};
    Stats->set_output(true)
         .set_uid(static_cast<int64_t>(TensorUID::Stats))
         .set_dim(stats_dim)
         .set_stride(stats_stride)
         .set_data_type(fe::DataType_t::FLOAT);

    // Build the graph with heuristic mode A (fastest)
    auto status = graph->validate();
    if (!status.is_good()) {
        throw std::runtime_error(std::string("SDPA graph validation failed: ") +
                                 status.get_message());
    }

    status = graph->build_operation_graph(handle);
    if (!status.is_good()) {
        throw std::runtime_error(std::string("SDPA graph build failed: ") +
                                 status.get_message());
    }

    status = graph->create_execution_plans({fe::HeurMode_t::A});
    if (!status.is_good()) {
        throw std::runtime_error(std::string("SDPA execution plan creation failed: ") +
                                 status.get_message());
    }

    status = graph->check_support(handle);
    if (!status.is_good()) {
        throw std::runtime_error(std::string("SDPA not supported on this device: ") +
                                 status.get_message());
    }

    status = graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
        throw std::runtime_error(std::string("SDPA plan build failed: ") +
                                 status.get_message());
    }

    int64_t workspace_size = graph->get_workspace_size();

    return {graph, workspace_size};
}

}  // anonymous namespace

/**
 * @brief cuDNN SDPA forward pass using Flash Attention with Tensor Cores
 *
 * @param Q Query tensor [batch, num_heads, seq_len_q, head_dim] or [batch*heads, seq_len_q, head_dim]
 * @param K Key tensor [batch, num_heads, seq_len_k, head_dim] or [batch*heads, seq_len_k, head_dim]
 * @param V Value tensor [batch, num_heads, seq_len_k, head_dim] or [batch*heads, seq_len_k, head_dim]
 * @param scale Attention scale factor (1/sqrt(head_dim))
 * @return Output tensor with same shape as Q
 */
auto cudnn_sdpa_forward(
    const Tensor& Q_in,
    const Tensor& K_in,
    const Tensor& V_in,
    float scale,
    bool causal,
    cudaStream_t stream
) -> std::pair<Tensor, Tensor> {
    // Per docs/internals/attention-contract.md, the host helper must enforce
    // contiguity at entry — cuDNN's stride descriptors are computed assuming
    // contiguous BHSD layout in compute_strides() below; a permuted view of
    // [B,S,H,D] would silently read garbage (audit C3/H6).
    const Tensor& Q = Q_in.is_contiguous() ? Q_in : Q_in.contiguous();
    const Tensor& K = K_in.is_contiguous() ? K_in : K_in.contiguous();
    const Tensor& V = V_in.is_contiguous() ? V_in : V_in.contiguous();
    // Determine I/O dtype for cuDNN. Anything outside the FP16 / BF16 / FP32
    // set goes straight to the custom flash kernel — cuDNN frontend does not
    // expose other types for SDPA.
    //
    // Architecture gate for FP32: cuDNN frontend reports check_support OK
    // for FP32 SDPA on Blackwell (sm >= 100), but execute() crashes with an
    // illegal memory access — upstream tuning gap on the newest arch as of
    // cuDNN 9.x. We throw here so attention.cpp's outer try/catch falls
    // through to the BMM path (which on these shapes is faster than the
    // custom flash kernel anyway). FP16 / BF16 on Blackwell are unaffected.
    fe::DataType_t io_dtype;
    switch (Q.dtype()) {
        case DType::Float16:  io_dtype = fe::DataType_t::HALF; break;
        case DType::BFloat16: io_dtype = fe::DataType_t::BFLOAT16; break;
        case DType::Float32:
            // Gate on the compute capability of the tensor's own device, not the
            // calling thread's current device (see sm_arch_of_device above).
            if (sm_arch_of_device(Q.device().index) >= 100) {
                // cuDNN FP32 SDPA is broken on Blackwell (sm_120): enabling it
                // triggers an illegal memory access (verified). Throw so MHA
                // falls back to the materialized BMM path, which is correct and
                // — measured — faster than the custom FP32 flash kernel here.
                throw std::runtime_error(
                    "cuDNN FP32 SDPA not stable on Blackwell — falling through to BMM");
            }
            io_dtype = fe::DataType_t::FLOAT;
            break;
        default:
            return fused_attention_fallback(Q, K, V, scale, causal);
    }

    // Determine dimensions
    auto q_shape = Q.shape();
    int64_t batch, num_heads, seq_len_q, head_dim;
    int64_t seq_len_k;
    bool is_3d = (q_shape.size() == 3);

    if (is_3d) {
        // Input is [batch*heads, seq_len, head_dim] — fold batch into heads.
        // For proper batch=heads attribution, callers should provide 4D tensors.
        int64_t batch_heads = q_shape[0];
        seq_len_q = q_shape[1];
        head_dim = q_shape[2];
        seq_len_k = K.shape()[1];
        batch = 1;
        num_heads = batch_heads;
    } else {
        // Input is [batch, num_heads, seq_len, head_dim]
        batch = q_shape[0];
        num_heads = q_shape[1];
        seq_len_q = q_shape[2];
        head_dim = q_shape[3];
        seq_len_k = K.shape()[2];
    }

    // Key for this exact shape + dtype + scale + causal. The causal flag changes
    // the cuDNN graph (mask op fused into softmax), so it must key the cache to
    // avoid silently reusing a non-causal graph for a causal call.
    // Include the device index: cuDNN frontend execution plans (and the
    // check_support/build negative-cache result, which can differ per arch) are
    // tied to the handle/device they were built on, so an identical shape on a
    // different GPU must get its own cache entry.
    SDPACacheKey cache_key{Q.device().index, batch, num_heads, seq_len_q, seq_len_k, head_dim, Q.dtype(), scale, causal};

    // Negative cache: if cuDNN previously failed validation/build for THIS exact
    // shape, skip the (expensive) graph build and fall back to the custom kernel.
    // Keyed by the full shape (not the coarse {dtype, head_dim, sm} triple) so a
    // shape-specific build failure does not poison every shape sharing the triple.
    if (SDPACapCache::instance().is_known_unsupported(cache_key)) {
        return fused_attention_fallback(Q, K, V, scale, causal);
    }

    // Run the cuDNN graph on the SAME device as the tensors. Make that device
    // current (cuDNN allocates internal scratch on the current device at create
    // time and references the current device at execute time), fetch the
    // per-device handle, and restore the previous current device on exit.
    const int sdpa_device = Q.device().index;
    // Shared RAII guard (one implementation in cuda_launch_utils.cuh): make the
    // tensor's device current for the cuDNN graph build/execute and restore the
    // previous device on scope exit.
    tenzor::cuda::CudaDeviceGuard device_guard(sdpa_device);

    cudnnHandle_t handle = SDPACuDNNHandle::get(sdpa_device);
    // Bind the requested stream to the handle so the SDPA work and the output's
    // allocation/consumers stay ordered on one stream (avoids a read-before-write
    // race when scheduled on a pooled non-default stream). nullptr selects the
    // default stream, matching prior behaviour.
    CUDNN_CHECK(cudnnSetStream(handle, stream));
    std::vector<int64_t> q_shape_vec(q_shape.begin(), q_shape.end());

    SDPACacheEntry cache_entry;

    if (!SDPAGraphCache::instance().get(cache_key, cache_entry)) {
        // Build new graph. Wrap in try/catch so any cuDNN validation/build
        // failure records the negative-cache miss for this shape and falls back
        // instead of repeatedly retrying an expensive build that cannot succeed.
        try {
            auto [graph, workspace_size] = create_sdpa_graph(
                handle, batch, num_heads, seq_len_q, seq_len_k, head_dim, scale, io_dtype, causal);
            cache_entry.graph = graph;
            cache_entry.workspace_size = workspace_size;
            SDPAGraphCache::instance().set(cache_key, cache_entry);
        } catch (const std::exception&) {
            SDPACapCache::instance().mark_unsupported(cache_key);
            // Composed fallback below uses fused_attention_cuda which now also
            // honors causal; passing it through preserves contract.
            return fused_attention_fallback(Q, K, V, scale, causal);
        }
    }

    void* workspace = SDPAWorkspace::get(cache_entry.workspace_size, sdpa_device);

    // Output dtype matches input (cuDNN frontend honors set_io_data_type).
    Tensor o_output(q_shape_vec, Q.dtype(), Q.device());
    // Stats (logsumexp) is always FP32, shaped [batch, num_heads, seq_len_q]
    // (the trailing size-1 dim from the cuDNN descriptor is dropped here —
    // matches the custom flash kernel's own lse shape convention).
    Tensor stats_output(std::vector<int64_t>{batch, num_heads, seq_len_q}, DType::Float32, Q.device());

    std::unordered_map<int64_t, void*> variant_pack;
    variant_pack[static_cast<int64_t>(TensorUID::Q)] = const_cast<void*>(Q.data_ptr());
    variant_pack[static_cast<int64_t>(TensorUID::K)] = const_cast<void*>(K.data_ptr());
    variant_pack[static_cast<int64_t>(TensorUID::V)] = const_cast<void*>(V.data_ptr());
    variant_pack[static_cast<int64_t>(TensorUID::O)] = o_output.data_ptr();
    variant_pack[static_cast<int64_t>(TensorUID::Stats)] = stats_output.data_ptr();

    auto status = cache_entry.graph->execute(handle, variant_pack, workspace);
    if (!status.is_good()) {
        // Execution-time failures aren't necessarily permanent (could be
        // workspace sizing etc.) but we still want to fall back gracefully
        // rather than throwing into the dispatch site.
        return fused_attention_fallback(Q, K, V, scale, causal);
    }

    return {o_output, stats_output};
}

/**
 * @brief Check if cuDNN SDPA is supported for the given configuration
 */
auto cudnn_sdpa_supported(
    int64_t batch,
    int64_t num_heads,
    int64_t seq_len_q,
    int64_t seq_len_k,
    int64_t head_dim,
    int device_index
) -> bool {
    // cuDNN SDPA requirements:
    // - Head dim must be 64 or 128 (some versions support 32, 256)
    // - Ampere or newer GPU (sm_80+)

    // Check head dim
    if (head_dim != 32 && head_dim != 64 && head_dim != 128 && head_dim != 256) {
        return false;
    }

    // Check GPU architecture (requires Ampere or newer). Query the tensors'
    // device, not just the calling thread's current device — on a multi-GPU
    // host they can differ. -1 means "use current device".
    int device = device_index;
    if (device < 0) {
        cudaGetDevice(&device);
    }
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int sm_version = props.major * 10 + props.minor;

    if (sm_version < 80) {  // Ampere = sm_80
        return false;
    }

    return true;
}

}  // namespace cuda
}  // namespace tenzor

#endif  // TENZOR_HAS_CUDNN_FRONTEND
