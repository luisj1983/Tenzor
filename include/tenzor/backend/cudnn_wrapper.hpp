#pragma once

#ifdef TENZOR_HAS_CUDNN

#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"

namespace tenzor {
namespace cuda {

// ============================================================================
// cuDNN Error Checking
// ============================================================================

#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        throw std::runtime_error( \
            std::string("cuDNN error: ") + cudnnGetErrorString(status) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__) \
        ); \
    } \
} while(0)

// ============================================================================
// cuDNN Handle Manager (Singleton for performance)
// ============================================================================

class CuDNNHandle {
public:
    // Get the singleton cuDNN handle
    static cudnnHandle_t get() {
        static CuDNNHandle instance;
        return instance.handle_;
    }

    // Set stream on the singleton handle
    static void set_stream(cudaStream_t stream) {
        CUDNN_CHECK(cudnnSetStream(get(), stream));
    }

    // Non-copyable, non-movable singleton
    CuDNNHandle(const CuDNNHandle&) = delete;
    CuDNNHandle& operator=(const CuDNNHandle&) = delete;
    CuDNNHandle(CuDNNHandle&&) = delete;
    CuDNNHandle& operator=(CuDNNHandle&&) = delete;

private:
    CuDNNHandle() {
        CUDNN_CHECK(cudnnCreate(&handle_));
    }

    ~CuDNNHandle() {
        if (handle_) {
            cudnnDestroy(handle_);
        }
    }

    cudnnHandle_t handle_ = nullptr;
};

// ============================================================================
// Workspace Manager (Persistent buffer for cuDNN operations)
// ============================================================================

class CuDNNWorkspace {
public:
    static void* get(size_t required_size) {
        static CuDNNWorkspace instance;
        if (required_size > instance.size_) {
            instance.resize(required_size);
        }
        return instance.buffer_;
    }

    static size_t current_size() {
        static CuDNNWorkspace instance;
        return instance.size_;
    }

    // Get maximum workspace size for algorithm search
    // Dynamic based on available memory, but conservative to leave room for tensors
    static size_t max_workspace_size() {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        // Use 10% of free memory for workspace, leaving 90% for model tensors
        // This allows good algorithm selection while avoiding OOM
        size_t dynamic_max = free_mem / 10;

        // Cap at 1GB - larger workspaces rarely provide better algorithms
        constexpr size_t kMaxCap = 1024ULL * 1024 * 1024;  // 1 GB
        // Minimum 64MB to ensure basic algorithms work
        constexpr size_t kMinWorkspace = 64ULL * 1024 * 1024;  // 64 MB

        return std::max(std::min(dynamic_max, kMaxCap), kMinWorkspace);
    }

    CuDNNWorkspace(const CuDNNWorkspace&) = delete;
    CuDNNWorkspace& operator=(const CuDNNWorkspace&) = delete;

private:
    CuDNNWorkspace() : buffer_(nullptr), size_(0) {}

    ~CuDNNWorkspace() {
        if (buffer_) {
            cudaFree(buffer_);
        }
    }

    void resize(size_t new_size) {
        if (buffer_) {
            cudaFree(buffer_);
        }
        // Allocate with some extra headroom to reduce reallocations
        size_ = new_size + (new_size / 4);  // 25% extra
        cudaMalloc(&buffer_, size_);
    }

    void* buffer_;
    size_t size_;
};

// ============================================================================
// Algorithm Cache for Conv2d (avoids per-call algorithm selection)
// ============================================================================

// Tensor format enum for cache key
enum class TensorFormat {
    NCHW = 0,
    NHWC = 1
};

// Key for conv2d algorithm cache
struct Conv2dCacheKey {
    int64_t batch;
    int64_t in_channels;
    int64_t height;
    int64_t width;
    int64_t out_channels;
    int64_t kernel_h;
    int64_t kernel_w;
    int64_t stride;
    int64_t padding;
    int64_t dilation;
    int64_t groups;
    cudnnDataType_t dtype;
    TensorFormat format;  // NCHW or NHWC

    bool operator==(const Conv2dCacheKey& other) const {
        return batch == other.batch &&
               in_channels == other.in_channels &&
               height == other.height &&
               width == other.width &&
               out_channels == other.out_channels &&
               kernel_h == other.kernel_h &&
               kernel_w == other.kernel_w &&
               stride == other.stride &&
               padding == other.padding &&
               dilation == other.dilation &&
               groups == other.groups &&
               dtype == other.dtype &&
               format == other.format;
    }
};

// Hash function for Conv2dCacheKey
struct Conv2dCacheKeyHash {
    size_t operator()(const Conv2dCacheKey& k) const {
        // Combine all fields into a single hash
        size_t h = 0;
        auto hash_combine = [&h](auto val) {
            h ^= std::hash<decltype(val)>{}(val) + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        hash_combine(k.batch);
        hash_combine(k.in_channels);
        hash_combine(k.height);
        hash_combine(k.width);
        hash_combine(k.out_channels);
        hash_combine(k.kernel_h);
        hash_combine(k.kernel_w);
        hash_combine(k.stride);
        hash_combine(k.padding);
        hash_combine(k.dilation);
        hash_combine(k.groups);
        hash_combine(static_cast<int>(k.dtype));
        hash_combine(static_cast<int>(k.format));
        return h;
    }
};

// Cached algorithm info
struct CachedFwdAlgo {
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_size;
};

struct CachedBwdDataAlgo {
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t workspace_size;
};

struct CachedBwdFilterAlgo {
    cudnnConvolutionBwdFilterAlgo_t algo;
    size_t workspace_size;
};

// Thread-safe algorithm cache
class Conv2dAlgoCache {
public:
    static Conv2dAlgoCache& instance() {
        static Conv2dAlgoCache cache;
        return cache;
    }

    // Forward algorithm cache
    bool get_fwd(const Conv2dCacheKey& key, CachedFwdAlgo& result) {
        std::lock_guard<std::mutex> lock(fwd_mutex_);
        auto it = fwd_cache_.find(key);
        if (it != fwd_cache_.end()) {
            result = it->second;
            return true;
        }
        return false;
    }

    void set_fwd(const Conv2dCacheKey& key, const CachedFwdAlgo& algo) {
        std::lock_guard<std::mutex> lock(fwd_mutex_);
        fwd_cache_[key] = algo;
    }

    // Backward data algorithm cache
    bool get_bwd_data(const Conv2dCacheKey& key, CachedBwdDataAlgo& result) {
        std::lock_guard<std::mutex> lock(bwd_data_mutex_);
        auto it = bwd_data_cache_.find(key);
        if (it != bwd_data_cache_.end()) {
            result = it->second;
            return true;
        }
        return false;
    }

    void set_bwd_data(const Conv2dCacheKey& key, const CachedBwdDataAlgo& algo) {
        std::lock_guard<std::mutex> lock(bwd_data_mutex_);
        bwd_data_cache_[key] = algo;
    }

    // Backward filter algorithm cache
    bool get_bwd_filter(const Conv2dCacheKey& key, CachedBwdFilterAlgo& result) {
        std::lock_guard<std::mutex> lock(bwd_filter_mutex_);
        auto it = bwd_filter_cache_.find(key);
        if (it != bwd_filter_cache_.end()) {
            result = it->second;
            return true;
        }
        return false;
    }

    void set_bwd_filter(const Conv2dCacheKey& key, const CachedBwdFilterAlgo& algo) {
        std::lock_guard<std::mutex> lock(bwd_filter_mutex_);
        bwd_filter_cache_[key] = algo;
    }

    // Clear all caches (useful for benchmarking)
    void clear() {
        {
            std::lock_guard<std::mutex> lock(fwd_mutex_);
            fwd_cache_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(bwd_data_mutex_);
            bwd_data_cache_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(bwd_filter_mutex_);
            bwd_filter_cache_.clear();
        }
    }

private:
    Conv2dAlgoCache() = default;

    std::unordered_map<Conv2dCacheKey, CachedFwdAlgo, Conv2dCacheKeyHash> fwd_cache_;
    std::mutex fwd_mutex_;

    std::unordered_map<Conv2dCacheKey, CachedBwdDataAlgo, Conv2dCacheKeyHash> bwd_data_cache_;
    std::mutex bwd_data_mutex_;

    std::unordered_map<Conv2dCacheKey, CachedBwdFilterAlgo, Conv2dCacheKeyHash> bwd_filter_cache_;
    std::mutex bwd_filter_mutex_;
};

// ============================================================================
// cuDNN Descriptor Wrappers (RAII)
// ============================================================================

class TensorDescriptor {
public:
    TensorDescriptor() {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    }

    ~TensorDescriptor() {
        if (desc_) {
            cudnnDestroyTensorDescriptor(desc_);
        }
    }

    TensorDescriptor(const TensorDescriptor&) = delete;
    TensorDescriptor& operator=(const TensorDescriptor&) = delete;

    cudnnTensorDescriptor_t get() const { return desc_; }

    // Set tensor descriptor in NCHW format (default)
    void set(cudnnDataType_t dtype, int n, int c, int h, int w) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            desc_,
            CUDNN_TENSOR_NCHW,
            dtype,
            n, c, h, w
        ));
    }

    // Set tensor descriptor in NHWC format (optimized for Tensor Cores)
    void set_nhwc(cudnnDataType_t dtype, int n, int c, int h, int w) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            desc_,
            CUDNN_TENSOR_NHWC,
            dtype,
            n, c, h, w
        ));
    }

    // Set tensor descriptor with explicit format
    void set_format(cudnnDataType_t dtype, int n, int c, int h, int w, TensorFormat format) {
        cudnnTensorFormat_t cudnn_format = (format == TensorFormat::NHWC)
            ? CUDNN_TENSOR_NHWC
            : CUDNN_TENSOR_NCHW;
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            desc_,
            cudnn_format,
            dtype,
            n, c, h, w
        ));
    }

    void set_from_tensor(const Tensor& tensor) {
        auto shape = tensor.shape();
        cudnnDataType_t dtype = dtype_to_cudnn(tensor.dtype());

        if (shape.size() == 4) {
            set(dtype, shape[0], shape[1], shape[2], shape[3]);
        } else if (shape.size() == 2) {
            set(dtype, shape[0], shape[1], 1, 1);
        } else {
            throw std::runtime_error("TensorDescriptor: unsupported shape dimension");
        }
    }

    // Helper to convert DType to cuDNN dtype
    static cudnnDataType_t dtype_to_cudnn(DType dtype) {
        switch (dtype) {
            case DType::Float32: return CUDNN_DATA_FLOAT;
            case DType::Float64: return CUDNN_DATA_DOUBLE;
            case DType::Float16: return CUDNN_DATA_HALF;
            case DType::Int32: return CUDNN_DATA_INT32;
            default:
                throw std::runtime_error("Unsupported dtype for cuDNN");
        }
    }

private:
    cudnnTensorDescriptor_t desc_ = nullptr;
};

class FilterDescriptor {
public:
    FilterDescriptor() {
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc_));
    }

    ~FilterDescriptor() {
        if (desc_) {
            cudnnDestroyFilterDescriptor(desc_);
        }
    }

    FilterDescriptor(const FilterDescriptor&) = delete;
    FilterDescriptor& operator=(const FilterDescriptor&) = delete;

    cudnnFilterDescriptor_t get() const { return desc_; }

    // Set filter descriptor in NCHW format (default)
    void set(cudnnDataType_t dtype, int k, int c, int h, int w) {
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            desc_,
            dtype,
            CUDNN_TENSOR_NCHW,
            k, c, h, w
        ));
    }

    // Set filter descriptor in NHWC format (optimized for Tensor Cores)
    void set_nhwc(cudnnDataType_t dtype, int k, int c, int h, int w) {
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            desc_,
            dtype,
            CUDNN_TENSOR_NHWC,
            k, c, h, w
        ));
    }

    // Set filter descriptor with explicit format
    void set_format(cudnnDataType_t dtype, int k, int c, int h, int w, TensorFormat format) {
        cudnnTensorFormat_t cudnn_format = (format == TensorFormat::NHWC)
            ? CUDNN_TENSOR_NHWC
            : CUDNN_TENSOR_NCHW;
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            desc_,
            dtype,
            cudnn_format,
            k, c, h, w
        ));
    }

private:
    cudnnFilterDescriptor_t desc_ = nullptr;
};

class ConvolutionDescriptor {
public:
    ConvolutionDescriptor() {
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc_));
    }

    ~ConvolutionDescriptor() {
        if (desc_) {
            cudnnDestroyConvolutionDescriptor(desc_);
        }
    }

    ConvolutionDescriptor(const ConvolutionDescriptor&) = delete;
    ConvolutionDescriptor& operator=(const ConvolutionDescriptor&) = delete;

    cudnnConvolutionDescriptor_t get() const { return desc_; }

    void set(int pad_h, int pad_w, int stride_h, int stride_w,
             int dilation_h, int dilation_w, cudnnDataType_t dtype) {
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            desc_,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            CUDNN_CROSS_CORRELATION,
            dtype
        ));

        // Enable tensor core acceleration (TF32 for FP32, FP16 tensor cores for FP16)
        // CUDNN_DEFAULT_MATH automatically uses TF32 on Ampere+ GPUs (like PyTorch)
        // For FP16, explicitly request tensor core math
        #ifdef TENZOR_HAS_TENSOR_CORES
        if (dtype == CUDNN_DATA_HALF) {
            CUDNN_CHECK(cudnnSetConvolutionMathType(desc_, CUDNN_TENSOR_OP_MATH));
        }
        // For FP32, use default math which enables TF32 on compatible GPUs
        // No need to set explicitly as CUDNN_DEFAULT_MATH is the default
        #endif
    }

    void set_group_count(int groups) {
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(desc_, groups));
    }

private:
    cudnnConvolutionDescriptor_t desc_ = nullptr;
};

class PoolingDescriptor {
public:
    PoolingDescriptor() {
        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&desc_));
    }

    ~PoolingDescriptor() {
        if (desc_) {
            cudnnDestroyPoolingDescriptor(desc_);
        }
    }

    PoolingDescriptor(const PoolingDescriptor&) = delete;
    PoolingDescriptor& operator=(const PoolingDescriptor&) = delete;

    cudnnPoolingDescriptor_t get() const { return desc_; }

    void set_maxpool(int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w) {
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(
            desc_,
            CUDNN_POOLING_MAX,
            CUDNN_NOT_PROPAGATE_NAN,
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w
        ));
    }

    void set_avgpool(int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, bool count_include_pad = false) {
        cudnnPoolingMode_t mode = count_include_pad ?
            CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING :
            CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(
            desc_,
            mode,
            CUDNN_NOT_PROPAGATE_NAN,
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w
        ));
    }

private:
    cudnnPoolingDescriptor_t desc_ = nullptr;
};

class RNNDescriptor {
public:
    RNNDescriptor() {
        CUDNN_CHECK(cudnnCreateRNNDescriptor(&desc_));
    }

    ~RNNDescriptor() {
        if (desc_) {
            cudnnDestroyRNNDescriptor(desc_);
        }
    }

    RNNDescriptor(const RNNDescriptor&) = delete;
    RNNDescriptor& operator=(const RNNDescriptor&) = delete;

    cudnnRNNDescriptor_t get() const { return desc_; }

    void set_lstm(cudnnHandle_t handle, int hidden_size, int num_layers,
                  cudnnDropoutDescriptor_t dropout_desc, cudnnRNNInputMode_t input_mode,
                  cudnnDirectionMode_t direction, cudnnDataType_t dtype) {
        // Note: cuDNN 9.x+ uses different RNN API (cudnnRNNForward_v8, etc.)
        // For now, we'll comment this out and focus on Conv2d operations
        // Full RNN support would require using the new cudnnRNNForward_v8 API

        // Legacy API (cuDNN < 8):
        // cudnnSetRNNDescriptor(...);

        // Modern API (cuDNN 8+):
        // Would need cudnnSetRNNDescriptor_v8 or similar

        throw std::runtime_error("cuDNN RNN operations not yet implemented for cuDNN 9.x+ (Conv2d is supported)");
    }

private:
    cudnnRNNDescriptor_t desc_ = nullptr;
};

class DropoutDescriptor {
public:
    DropoutDescriptor() {
        CUDNN_CHECK(cudnnCreateDropoutDescriptor(&desc_));
    }

    ~DropoutDescriptor() {
        if (desc_) {
            cudnnDestroyDropoutDescriptor(desc_);
        }
    }

    DropoutDescriptor(const DropoutDescriptor&) = delete;
    DropoutDescriptor& operator=(const DropoutDescriptor&) = delete;

    cudnnDropoutDescriptor_t get() const { return desc_; }

    void set(cudnnHandle_t handle, float dropout, void* states, size_t state_size, unsigned long long seed) {
        CUDNN_CHECK(cudnnSetDropoutDescriptor(
            desc_,
            handle,
            dropout,
            states,
            state_size,
            seed
        ));
    }

private:
    cudnnDropoutDescriptor_t desc_ = nullptr;
};

// ============================================================================
// NCHW <-> NHWC Format Conversion (GPU kernels)
// ============================================================================

/**
 * @brief Convert tensor from NCHW to NHWC format
 *
 * NHWC format is optimized for Tensor Cores on modern NVIDIA GPUs.
 * Memory layout: [batch, height, width, channels]
 */
auto nchw_to_nhwc(const Tensor& input, cudaStream_t stream = nullptr) -> Tensor;

/**
 * @brief Convert tensor from NHWC to NCHW format
 *
 * NCHW format is the standard PyTorch/Tenzor layout.
 * Memory layout: [batch, channels, height, width]
 */
auto nhwc_to_nchw(const Tensor& input, int64_t channels, cudaStream_t stream = nullptr) -> Tensor;

// ============================================================================
// cuDNN Convolution Operations
// ============================================================================

auto cudnn_conv2d_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor;

/**
 * @brief NHWC-optimized Conv2D forward using cuDNN
 *
 * Uses NHWC tensor format internally which enables:
 * - Better Tensor Core utilization (2-3x faster on RTX 30xx/40xx/50xx)
 * - Improved memory coalescing for channel dimension
 * - Faster implicit GEMM algorithms in cuDNN
 *
 * Handles NCHW<->NHWC conversion internally so the API remains unchanged.
 * For small convolutions, may fall back to NCHW if conversion overhead exceeds benefit.
 *
 * @param input Input tensor in NCHW format [N, C, H, W]
 * @param weight Filter tensor in NCHW format [K, C/groups, kH, kW]
 * @param bias Optional bias tensor [K]
 * @param stride Convolution stride
 * @param padding Convolution padding
 * @param dilation Convolution dilation
 * @param groups Number of groups for grouped convolution
 * @param stream CUDA stream
 * @return Output tensor in NCHW format [N, K, oH, oW]
 */
auto cudnn_conv2d_forward_nhwc(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor;

auto cudnn_conv2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool compute_grad_input,
    bool compute_grad_weight,
    bool compute_grad_bias,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor>;

/**
 * @brief NHWC-optimized Conv2D backward
 */
auto cudnn_conv2d_backward_nhwc(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool compute_grad_input,
    bool compute_grad_weight,
    bool compute_grad_bias,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor>;

// ============================================================================
// cuDNN RNN/LSTM Operations
// ============================================================================

auto cudnn_lstm_forward(
    const Tensor& input,         // (seq_len, batch, input_size)
    const Tensor& hx,            // (num_layers, batch, hidden_size)
    const Tensor& cx,            // (num_layers, batch, hidden_size)
    const Tensor& weights,       // Packed weights
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor>;

auto cudnn_lstm_backward(
    const Tensor& grad_output,
    const Tensor& grad_hy,
    const Tensor& grad_cy,
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& output,
    const Tensor& hy,
    const Tensor& cy,
    const Tensor& weights,
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor, Tensor>;

// ============================================================================
// cuDNN Pooling Operations
// ============================================================================

auto cudnn_maxpool2d_forward(
    const Tensor& input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    cudaStream_t stream
) -> std::pair<Tensor, Tensor>;  // Returns (output, indices)

auto cudnn_maxpool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& output,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    cudaStream_t stream
) -> Tensor;

auto cudnn_avgpool2d_forward(
    const Tensor& input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    cudaStream_t stream
) -> Tensor;

auto cudnn_avgpool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    cudaStream_t stream
) -> Tensor;

// ============================================================================
// cuDNN Softmax Operations
// ============================================================================

auto cudnn_softmax_forward(
    const Tensor& input,
    int64_t dim,
    cudaStream_t stream
) -> Tensor;

auto cudnn_softmax_backward(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    cudaStream_t stream
) -> Tensor;

auto cudnn_log_softmax_forward(
    const Tensor& input,
    int64_t dim,
    cudaStream_t stream
) -> Tensor;

auto cudnn_log_softmax_backward(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    cudaStream_t stream
) -> Tensor;

// ============================================================================
// cuDNN BatchNorm Operations (Optimized)
// ============================================================================

/**
 * @brief cuDNN-optimized BatchNorm inference forward
 *
 * @param input Input tensor (N, C, H, W)
 * @param running_mean Running mean (C,)
 * @param running_var Running variance (C,)
 * @param gamma Scale parameter (C,)
 * @param beta Bias parameter (C,)
 * @param epsilon Epsilon for numerical stability
 * @param stream CUDA stream
 * @return Output tensor (N, C, H, W)
 */
auto cudnn_batchnorm2d_forward_inference(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& gamma,
    const Tensor& beta,
    float epsilon,
    cudaStream_t stream = nullptr
) -> Tensor;

/**
 * @brief cuDNN-optimized BatchNorm training forward
 *
 * @param input Input tensor (N, C, H, W)
 * @param running_mean Running mean to update (C,)
 * @param running_var Running variance to update (C,)
 * @param gamma Scale parameter (C,)
 * @param beta Bias parameter (C,)
 * @param momentum Momentum for running statistics
 * @param epsilon Epsilon for numerical stability
 * @param stream CUDA stream
 * @return Tuple of (output, saved_mean, saved_inv_variance)
 */
auto cudnn_batchnorm2d_forward_training(
    const Tensor& input,
    Tensor& running_mean,
    Tensor& running_var,
    const Tensor& gamma,
    const Tensor& beta,
    float momentum,
    float epsilon,
    cudaStream_t stream = nullptr
) -> std::tuple<Tensor, Tensor, Tensor>;

/**
 * @brief cuDNN-optimized BatchNorm backward
 *
 * @param grad_output Gradient from next layer (N, C, H, W)
 * @param input Original input (N, C, H, W)
 * @param gamma Scale parameter (C,)
 * @param saved_mean Saved mean from forward (C,)
 * @param saved_inv_var Saved inverse variance from forward (C,)
 * @param epsilon Epsilon used in forward
 * @param stream CUDA stream
 * @return Tuple of (grad_input, grad_gamma, grad_beta)
 */
auto cudnn_batchnorm2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& saved_mean,
    const Tensor& saved_inv_var,
    float epsilon,
    cudaStream_t stream = nullptr
) -> std::tuple<Tensor, Tensor, Tensor>;

/**
 * @brief Wrapper matching existing batchnorm2d_forward_affine signature
 */
auto cudnn_batchnorm2d_forward_affine_wrapper(
    const Tensor& input,
    const Tensor& mean,
    const Tensor& variance,
    const Tensor& gamma,
    const Tensor& beta,
    float epsilon,
    cudaStream_t stream = nullptr
) -> Tensor;

// ============================================================================
// cuDNN LayerNorm Operations (Optimized with warp shuffles)
// ============================================================================

/**
 * @brief Optimized LayerNorm forward using warp-level primitives
 *
 * @param input Input tensor
 * @param normalized_shape Shape of normalized dimensions
 * @param weight Gamma parameter
 * @param bias Beta parameter
 * @param eps Epsilon for numerical stability
 * @param stream CUDA stream
 * @return Tuple of (output, mean, inv_std)
 */
auto cudnn_layer_norm_forward(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps,
    cudaStream_t stream = nullptr
) -> std::tuple<Tensor, Tensor, Tensor>;

/**
 * @brief Optimized LayerNorm backward using warp-level primitives
 *
 * @param grad_output Gradient from next layer
 * @param input Original input
 * @param weight Gamma parameter
 * @param mean Saved mean from forward
 * @param inv_std Saved inverse std from forward
 * @param normalized_shape Shape of normalized dimensions
 * @param stream CUDA stream
 * @return Tuple of (grad_input, grad_weight, grad_bias)
 */
auto cudnn_layer_norm_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& mean,
    const Tensor& inv_std,
    const std::vector<int64_t>& normalized_shape,
    cudaStream_t stream = nullptr
) -> std::tuple<Tensor, Tensor, Tensor>;

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_HAS_CUDNN
