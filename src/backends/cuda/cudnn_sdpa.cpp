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
#include <utility>
#include <list>
#include <mutex>

#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "tenzor/backend/fused_ops.hpp"

namespace fe = cudnn_frontend;

// cuDNN error checking macro
#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        throw std::runtime_error( \
            std::string("cuDNN error: ") + cudnnGetErrorString(status) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__) \
        ); \
    } \
} while(0)

namespace tenzor {
namespace cuda {

// Forward declaration: FP32 flash attention kernel (defined in fused_ops.cu)
auto fused_attention_cuda(const Tensor& Q, const Tensor& K, const Tensor& V, float scale) -> std::pair<Tensor, Tensor>;

namespace {

// ============================================================================
// Singleton cuDNN Handle for SDPA operations
// ============================================================================

class SDPACuDNNHandle {
public:
    static cudnnHandle_t get() {
        static SDPACuDNNHandle instance;
        return instance.handle_;
    }

    static void set_stream(cudaStream_t stream) {
        CUDNN_CHECK(cudnnSetStream(get(), stream));
    }

    SDPACuDNNHandle(const SDPACuDNNHandle&) = delete;
    SDPACuDNNHandle& operator=(const SDPACuDNNHandle&) = delete;

private:
    SDPACuDNNHandle() {
        CUDNN_CHECK(cudnnCreate(&handle_));
    }

    ~SDPACuDNNHandle() {
        if (handle_) {
            cudnnDestroy(handle_);
        }
    }

    cudnnHandle_t handle_ = nullptr;
};

// ============================================================================
// Cache for SDPA graph plans to avoid rebuilding for same shapes
// ============================================================================
struct SDPACacheKey {
    int64_t batch;
    int64_t num_heads;
    int64_t seq_len_q;
    int64_t seq_len_k;
    int64_t head_dim;
    DType dtype;
    float scale;

    bool operator==(const SDPACacheKey& other) const {
        return batch == other.batch && num_heads == other.num_heads &&
               seq_len_q == other.seq_len_q && seq_len_k == other.seq_len_k &&
               head_dim == other.head_dim && dtype == other.dtype &&
               scale == other.scale;
    }
};

struct SDPACacheKeyHash {
    size_t operator()(const SDPACacheKey& k) const {
        // Simple hash combining all dimensions, dtype, and scale
        size_t h = std::hash<int64_t>{}(k.batch);
        h ^= std::hash<int64_t>{}(k.num_heads) << 1;
        h ^= std::hash<int64_t>{}(k.seq_len_q) << 2;
        h ^= std::hash<int64_t>{}(k.seq_len_k) << 3;
        h ^= std::hash<int64_t>{}(k.head_dim) << 4;
        h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.dtype)) << 5;
        h ^= std::hash<float>{}(k.scale) << 6;
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
class SDPAWorkspace {
public:
    static void* get(size_t required_size) {
        static thread_local SDPAWorkspace instance;
        if (required_size > instance.size_) {
            instance.resize(required_size);
        }
        return instance.buffer_;
    }

private:
    SDPAWorkspace() : buffer_(nullptr), size_(0) {}
    ~SDPAWorkspace() {
        if (buffer_) {
            backend::CachingAllocator::get().free(buffer_);
        }
    }

    void resize(size_t new_size) {
        if (buffer_) {
            backend::CachingAllocator::get().free(buffer_);
        }
        size_ = new_size + (new_size / 4);  // 25% headroom
        buffer_ = backend::CachingAllocator::get().allocate(size_);
    }

    void* buffer_;
    size_t size_;
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

auto create_sdpa_graph(
    cudnnHandle_t handle,
    int64_t batch,
    int64_t num_heads,
    int64_t seq_len_q,
    int64_t seq_len_k,
    int64_t head_dim,
    float attn_scale
) -> std::pair<std::shared_ptr<fe::graph::Graph>, int64_t> {

    auto graph = std::make_shared<fe::graph::Graph>();

    // Set data types: FP16 for I/O, FP32 for compute
    graph->set_io_data_type(fe::DataType_t::HALF)
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
        .set_data_type(fe::DataType_t::HALF));

    auto K = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("K")
        .set_uid(static_cast<int64_t>(TensorUID::K))
        .set_dim(k_dim)
        .set_stride(k_stride)
        .set_data_type(fe::DataType_t::HALF));

    auto V = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("V")
        .set_uid(static_cast<int64_t>(TensorUID::V))
        .set_dim(v_dim)
        .set_stride(v_stride)
        .set_data_type(fe::DataType_t::HALF));

    // Configure SDPA attributes
    auto sdpa_options = fe::graph::SDPA_attributes()
        .set_name("flash_attention")
        .set_is_inference(true)
        .set_attn_scale(attn_scale);

    // Create SDPA operation
    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Mark output tensor
    O->set_output(true)
     .set_uid(static_cast<int64_t>(TensorUID::O))
     .set_dim(o_dim)
     .set_stride(o_stride)
     .set_data_type(fe::DataType_t::HALF);

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
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    float scale
) -> Tensor {
    // FP32/BF16 bypass: route to custom flash attention kernel
    // BF16 is handled by fused_attention_cuda via upcast to FP32
    if (Q.dtype() == DType::Float32 || Q.dtype() == DType::BFloat16) {
        auto [output, lse] = fused_attention_cuda(Q, K, V, scale);
        return output;
    }

    // Get singleton cuDNN handle (avoids create/destroy overhead)
    cudnnHandle_t handle = SDPACuDNNHandle::get();

    // Determine dimensions
    auto q_shape = Q.shape();
    int64_t batch, num_heads, seq_len_q, head_dim;
    int64_t seq_len_k;
    bool is_3d = (q_shape.size() == 3);

    if (is_3d) {
        // Input is [batch*heads, seq_len, head_dim]
        // We need to infer batch and num_heads - assume batch=1 for simplicity
        // or caller should provide 4D tensors
        int64_t batch_heads = q_shape[0];
        seq_len_q = q_shape[1];
        head_dim = q_shape[2];
        seq_len_k = K.shape()[1];

        // Assume square batch*heads factorization - this is a limitation
        // For proper support, caller should provide 4D tensors
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

    // Convert span to vector for Tensor construction
    std::vector<int64_t> q_shape_vec(q_shape.begin(), q_shape.end());

    // Check cache for pre-built graph
    SDPACacheKey cache_key{batch, num_heads, seq_len_q, seq_len_k, head_dim, Q.dtype(), scale};
    SDPACacheEntry cache_entry;

    if (!SDPAGraphCache::instance().get(cache_key, cache_entry)) {
        // Build new graph
        auto [graph, workspace_size] = create_sdpa_graph(
            handle, batch, num_heads, seq_len_q, seq_len_k, head_dim, scale);
        cache_entry.graph = graph;
        cache_entry.workspace_size = workspace_size;
        SDPAGraphCache::instance().set(cache_key, cache_entry);
    }

    // Get workspace
    void* workspace = SDPAWorkspace::get(cache_entry.workspace_size);

    // Only FP16 reaches here (FP32 is handled by fused_attention_cuda above)
    Tensor o_output(q_shape_vec, DType::Float16, Q.device());

    // Create variant pack (const_cast needed: cuDNN FE takes void* but does not modify inputs)
    std::unordered_map<int64_t, void*> variant_pack;
    variant_pack[static_cast<int64_t>(TensorUID::Q)] = const_cast<void*>(Q.data_ptr());
    variant_pack[static_cast<int64_t>(TensorUID::K)] = const_cast<void*>(K.data_ptr());
    variant_pack[static_cast<int64_t>(TensorUID::V)] = const_cast<void*>(V.data_ptr());
    variant_pack[static_cast<int64_t>(TensorUID::O)] = o_output.data_ptr();

    // Execute
    auto status = cache_entry.graph->execute(handle, variant_pack, workspace);
    if (!status.is_good()) {
        throw std::runtime_error(std::string("SDPA execution failed: ") +
                                 status.get_message());
    }

    return o_output;
}

/**
 * @brief Check if cuDNN SDPA is supported for the given configuration
 */
auto cudnn_sdpa_supported(
    int64_t batch,
    int64_t num_heads,
    int64_t seq_len_q,
    int64_t seq_len_k,
    int64_t head_dim
) -> bool {
    // cuDNN SDPA requirements:
    // - Head dim must be 64 or 128 (some versions support 32, 256)
    // - Ampere or newer GPU (sm_80+)

    // Check head dim
    if (head_dim != 32 && head_dim != 64 && head_dim != 128 && head_dim != 256) {
        return false;
    }

    // Check GPU architecture (requires Ampere or newer)
    int device;
    cudaGetDevice(&device);
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
