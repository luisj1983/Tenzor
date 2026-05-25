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
#include <vector>
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
    // Audit-6 AA.12: `get()` previously called `resize()` under no lock, so
    // two streams hitting cuDNN concurrently could race: stream A's
    // `cudaFree(buffer_)` would invalidate the pointer stream B's kernel
    // was about to read. The cudaMalloc return code was also dropped, so
    // OOM silently produced a nullptr that crashed deep in cuDNN.
    //
    // The immediate fix is a single mutex serialising both `get()` and
    // `resize()` plus checked cudaMalloc. Stream-scoped workspaces and a
    // proper retire-after-stream-sync are tracked as a follow-up; the
    // mutex closes the wrong-math hazard now.
    static void* get(size_t required_size) {
        static CuDNNWorkspace instance;
        std::lock_guard<std::mutex> lock(instance.mu_);
        if (required_size > instance.size_) {
            instance.resize_locked(required_size);
        }
        return instance.buffer_;
    }

    static size_t current_size() {
        static CuDNNWorkspace instance;
        std::lock_guard<std::mutex> lock(instance.mu_);
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

    // Caller must hold `mu_`.
    void resize_locked(size_t new_size) {
        if (buffer_) {
            cudaFree(buffer_);
            buffer_ = nullptr;
        }
        // Allocate with some extra headroom to reduce reallocations
        size_ = new_size + (new_size / 4);  // 25% extra
        cudaError_t err = cudaMalloc(&buffer_, size_);
        if (err != cudaSuccess) {
            // Reset state so a later, smaller allocation can succeed
            buffer_ = nullptr;
            size_t failed_size = size_;
            size_ = 0;
            throw std::runtime_error(
                std::string("CuDNNWorkspace: cudaMalloc(") +
                std::to_string(failed_size) + ") failed: " +
                cudaGetErrorString(err));
        }
    }

    void* buffer_;
    size_t size_;
    std::mutex mu_;
};

// ============================================================================
// Algorithm Cache for Conv2d (avoids per-call algorithm selection)
// ============================================================================

// Tensor format enum for cache key
enum class TensorFormat {
    NCHW = 0,
    NHWC = 1
};

// Key for conv2d algorithm cache. Audit E1: per-axis stride/padding/dilation
// so asymmetric convolutions get distinct cache entries.
struct Conv2dCacheKey {
    int64_t batch;
    int64_t in_channels;
    int64_t height;
    int64_t width;
    int64_t out_channels;
    int64_t kernel_h;
    int64_t kernel_w;
    int64_t stride_h;
    int64_t stride_w;
    int64_t pad_h;
    int64_t pad_w;
    int64_t dil_h;
    int64_t dil_w;
    int64_t groups;
    cudnnDataType_t dtype;
    TensorFormat format;  // NCHW or NHWC
    // Track whether the cached algo was picked under the "prefer precise
    // FP32" policy (TF32 disabled). A Winograd algo cached when TF32 was
    // allowed must not be reused when the caller has since disabled TF32,
    // and vice versa.
    bool prefer_precise_f32 = false;

    bool operator==(const Conv2dCacheKey& other) const {
        return batch == other.batch &&
               in_channels == other.in_channels &&
               height == other.height &&
               width == other.width &&
               out_channels == other.out_channels &&
               kernel_h == other.kernel_h &&
               kernel_w == other.kernel_w &&
               stride_h == other.stride_h &&
               stride_w == other.stride_w &&
               pad_h == other.pad_h &&
               pad_w == other.pad_w &&
               dil_h == other.dil_h &&
               dil_w == other.dil_w &&
               groups == other.groups &&
               dtype == other.dtype &&
               format == other.format &&
               prefer_precise_f32 == other.prefer_precise_f32;
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
        hash_combine(k.stride_h);
        hash_combine(k.stride_w);
        hash_combine(k.pad_h);
        hash_combine(k.pad_w);
        hash_combine(k.dil_h);
        hash_combine(k.dil_w);
        hash_combine(k.groups);
        hash_combine(static_cast<int>(k.dtype));
        hash_combine(static_cast<int>(k.format));
        hash_combine(static_cast<int>(k.prefer_precise_f32));
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
        // For FP16 inputs, use FP32 compute type to prevent intermediate product
        // overflow (FP16 × FP16 can exceed 65504 and produce Inf in tensor cores).
        // This matches CPU behavior where Float32 products are used in GEMM.
        // For FP16/BF16 inputs, use FP32 compute type to prevent intermediate
        // product overflow and maintain precision (cuDNN requires FP32 compute
        // for half-precision data types)
        cudnnDataType_t compute_type = dtype;
        if (dtype == CUDNN_DATA_HALF || dtype == CUDNN_DATA_BFLOAT16) {
            compute_type = CUDNN_DATA_FLOAT;
        }

        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            desc_,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            CUDNN_CROSS_CORRELATION,
            compute_type
        ));

        // Enable tensor core acceleration for half-precision types
        // TENSOR_OP_MATH_ALLOW_CONVERSION allows cuDNN to use tensor cores
        // (FP16/BF16 I/O with FP32 accumulate) while maintaining FP32 compute
        // precision for intermediate operations.
        #ifdef TENZOR_HAS_TENSOR_CORES
        if (dtype == CUDNN_DATA_HALF || dtype == CUDNN_DATA_BFLOAT16) {
            CUDNN_CHECK(cudnnSetConvolutionMathType(desc_, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
        }
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

class ActivationDescriptor {
public:
    ActivationDescriptor() {
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&desc_));
    }

    ~ActivationDescriptor() {
        if (desc_) {
            cudnnDestroyActivationDescriptor(desc_);
        }
    }

    ActivationDescriptor(const ActivationDescriptor&) = delete;
    ActivationDescriptor& operator=(const ActivationDescriptor&) = delete;

    cudnnActivationDescriptor_t get() const { return desc_; }

    void set(cudnnActivationMode_t mode, double coeff = 0.0) {
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            desc_, mode, CUDNN_NOT_PROPAGATE_NAN, coeff));
    }

    void set_relu() {
        set(CUDNN_ACTIVATION_RELU);
    }

    void set_sigmoid() {
        set(CUDNN_ACTIVATION_SIGMOID);
    }

    void set_tanh() {
        set(CUDNN_ACTIVATION_TANH);
    }

    void set_elu(double alpha = 1.0) {
        set(CUDNN_ACTIVATION_ELU, alpha);
    }

    void set_swish(double beta = 1.0) {
        set(CUDNN_ACTIVATION_SWISH, beta);
    }

    void set_identity() {
        set(CUDNN_ACTIVATION_IDENTITY);
    }

private:
    cudnnActivationDescriptor_t desc_ = nullptr;
};

/**
 * @brief RAII wrapper around `cudnnRNNDescriptor_t`.
 *
 * The descriptor is configured via the cuDNN v8 RNN API
 * (`cudnnSetRNNDescriptor_v8`), which is the documented entry point for both
 * cuDNN 8.x and 9.x. The legacy v6 API (`cudnnSetRNNDescriptor_v6`) is offered
 * as a fallback when building against cuDNN < 8 — in that case it omits the
 * features introduced in v8 (recurrent projection, math-type selection, the
 * `auxFlags` bitmap) since they don't exist in the older library.
 */
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

    /**
     * @brief Configure as an LSTM via cuDNN v8 RNN API.
     *
     * @param handle        Active cuDNN handle (only used for the v6 fallback
     *                      so cuDNN can allocate persistent buffers).
     * @param input_size    Number of features in each input timestep.
     * @param hidden_size   Number of features in the hidden state.
     * @param proj_size     Output projection dimension (LSTMP). Pass 0 (or
     *                      `hidden_size`) for no projection.
     * @param num_layers    Number of stacked layers.
     * @param bidirectional Build a bidirectional LSTM if true.
     * @param dropout_desc  Inter-layer dropout descriptor (must be
     *                      initialised; pass a descriptor configured with
     *                      `dropout=0` when no dropout is desired).
     * @param dtype         Element dtype of input / output / weights.
     * @param math_prec     Compute precision. Defaults to Float32.
     * @param math_type     Tensor-Core hint. Defaults to `CUDNN_DEFAULT_MATH`.
     */
    void set_lstm(cudnnHandle_t handle,
                  int input_size,
                  int hidden_size,
                  int proj_size,
                  int num_layers,
                  bool bidirectional,
                  cudnnDropoutDescriptor_t dropout_desc,
                  cudnnDataType_t dtype,
                  cudnnDataType_t math_prec = CUDNN_DATA_FLOAT,
                  cudnnMathType_t math_type = CUDNN_DEFAULT_MATH) {
        configure(handle,
                  CUDNN_LSTM,
                  CUDNN_RNN_DOUBLE_BIAS,
                  input_size, hidden_size, proj_size, num_layers,
                  bidirectional, dropout_desc,
                  dtype, math_prec, math_type);
    }

    /**
     * @brief Configure as a GRU. cuDNN does not support recurrent projection
     * for GRU, so `proj_size` is implicitly 0.
     */
    void set_gru(cudnnHandle_t handle,
                 int input_size,
                 int hidden_size,
                 int num_layers,
                 bool bidirectional,
                 cudnnDropoutDescriptor_t dropout_desc,
                 cudnnDataType_t dtype,
                 cudnnDataType_t math_prec = CUDNN_DATA_FLOAT,
                 cudnnMathType_t math_type = CUDNN_DEFAULT_MATH) {
        configure(handle,
                  CUDNN_GRU,
                  CUDNN_RNN_DOUBLE_BIAS,
                  input_size, hidden_size, /*proj_size=*/0, num_layers,
                  bidirectional, dropout_desc,
                  dtype, math_prec, math_type);
    }

    /**
     * @brief Configure as a basic Elman RNN with the chosen pointwise
     * activation (tanh by default; ReLU also supported).
     */
    void set_rnn(cudnnHandle_t handle,
                 int input_size,
                 int hidden_size,
                 int num_layers,
                 bool bidirectional,
                 cudnnDropoutDescriptor_t dropout_desc,
                 cudnnDataType_t dtype,
                 cudnnRNNMode_t cell_mode = CUDNN_RNN_TANH,
                 cudnnDataType_t math_prec = CUDNN_DATA_FLOAT,
                 cudnnMathType_t math_type = CUDNN_DEFAULT_MATH) {
        if (cell_mode != CUDNN_RNN_TANH && cell_mode != CUDNN_RNN_RELU) {
            throw std::runtime_error(
                "RNNDescriptor::set_rnn: cell_mode must be CUDNN_RNN_TANH or CUDNN_RNN_RELU");
        }
        configure(handle,
                  cell_mode,
                  CUDNN_RNN_DOUBLE_BIAS,
                  input_size, hidden_size, /*proj_size=*/0, num_layers,
                  bidirectional, dropout_desc,
                  dtype, math_prec, math_type);
    }

private:
    void configure(cudnnHandle_t handle,
                   cudnnRNNMode_t cell_mode,
                   cudnnRNNBiasMode_t bias_mode,
                   int input_size,
                   int hidden_size,
                   int proj_size,
                   int num_layers,
                   bool bidirectional,
                   cudnnDropoutDescriptor_t dropout_desc,
                   cudnnDataType_t dtype,
                   cudnnDataType_t math_prec,
                   cudnnMathType_t math_type) {
        const auto direction = bidirectional
            ? CUDNN_BIDIRECTIONAL
            : CUDNN_UNIDIRECTIONAL;
        const auto input_mode = CUDNN_LINEAR_INPUT;

#if defined(CUDNN_MAJOR) && CUDNN_MAJOR >= 8
        // cuDNN 8.x and 9.x: a single v8 setter accepts every option.
        CUDNN_CHECK(cudnnSetRNNDescriptor_v8(
            desc_,
            CUDNN_RNN_ALGO_STANDARD,
            cell_mode,
            bias_mode,
            direction,
            input_mode,
            dtype,
            math_prec,
            math_type,
            input_size,
            hidden_size,
            proj_size,
            num_layers,
            dropout_desc,
            /*auxFlags=*/CUDNN_RNN_PADDED_IO_DISABLED));
        (void)handle;
#else
        // cuDNN < 8 fallback. The v6 API lacks projection, math-type, and the
        // padded-IO flag; reject configurations that depend on those features
        // instead of silently dropping them.
        if (proj_size != 0 && proj_size != hidden_size) {
            throw std::runtime_error(
                "RNNDescriptor: recurrent projection requires cuDNN >= 8");
        }
        if (math_type != CUDNN_DEFAULT_MATH) {
            throw std::runtime_error(
                "RNNDescriptor: math_type selection requires cuDNN >= 8");
        }
        CUDNN_CHECK(cudnnSetRNNDescriptor_v6(
            handle,
            desc_,
            hidden_size,
            num_layers,
            dropout_desc,
            input_mode,
            direction,
            cell_mode,
            CUDNN_RNN_ALGO_STANDARD,
            math_prec));
        (void)bias_mode;
        (void)input_size;
        (void)dtype;
#endif
    }

    cudnnRNNDescriptor_t desc_ = nullptr;
};

/**
 * @brief RAII wrapper around `cudnnRNNDataDescriptor_t`.
 *
 * cuDNN distinguishes the *RNN* descriptor (which fixes the cell shape) from
 * the *RNN data* descriptor (which fixes the sequence layout: time-major vs
 * batch-major, max seq length, per-sample length array, padding fill). This
 * wrapper handles construction / destruction and exposes a setter for each
 * layout; per-sample sequence lengths are stored on the host side and
 * forwarded to `cudnnSetRNNDataDescriptor`. The companion device-side length
 * array required by `cudnnRNNForward` is the caller's responsibility because
 * its lifetime needs to span the cuDNN call but not necessarily the
 * descriptor's.
 */
class RNNDataDescriptor {
public:
    RNNDataDescriptor() {
        CUDNN_CHECK(cudnnCreateRNNDataDescriptor(&desc_));
    }

    ~RNNDataDescriptor() {
        if (desc_) {
            cudnnDestroyRNNDataDescriptor(desc_);
        }
    }

    RNNDataDescriptor(const RNNDataDescriptor&) = delete;
    RNNDataDescriptor& operator=(const RNNDataDescriptor&) = delete;

    cudnnRNNDataDescriptor_t get() const { return desc_; }

    /**
     * @brief Configure a time-major (seq-major) RNN data descriptor.
     *
     * Layout: `[max_seq_length, batch_size, vector_size]`, contiguous in
     * vector. Per-sample lengths default to `max_seq_length` (no truncation).
     */
    void set_seq_major(cudnnDataType_t dtype,
                       int max_seq_length,
                       int batch_size,
                       int vector_size,
                       const int* seq_lengths = nullptr,
                       void* padding_fill = nullptr) {
        set(dtype,
            CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
            max_seq_length, batch_size, vector_size,
            seq_lengths, padding_fill);
    }

    /**
     * @brief Configure a batch-major RNN data descriptor.
     *
     * Layout: `[batch_size, max_seq_length, vector_size]`. Matches PyTorch's
     * `batch_first=True` convention.
     */
    void set_batch_major(cudnnDataType_t dtype,
                         int max_seq_length,
                         int batch_size,
                         int vector_size,
                         const int* seq_lengths = nullptr,
                         void* padding_fill = nullptr) {
        set(dtype,
            CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
            max_seq_length, batch_size, vector_size,
            seq_lengths, padding_fill);
    }

    /**
     * @brief Configure a packed (variable-length, no padding) descriptor.
     * Sequences must be sorted in descending length order.
     */
    void set_seq_major_packed(cudnnDataType_t dtype,
                              int max_seq_length,
                              int batch_size,
                              int vector_size,
                              const int* seq_lengths,
                              void* padding_fill = nullptr) {
        if (!seq_lengths) {
            throw std::runtime_error(
                "RNNDataDescriptor::set_seq_major_packed: seq_lengths is required");
        }
        set(dtype,
            CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
            max_seq_length, batch_size, vector_size,
            seq_lengths, padding_fill);
    }

private:
    void set(cudnnDataType_t dtype,
             cudnnRNNDataLayout_t layout,
             int max_seq_length,
             int batch_size,
             int vector_size,
             const int* seq_lengths,
             void* padding_fill) {
        // cudnnSetRNNDataDescriptor requires a length-per-sample array, even
        // when every sequence is the same length. Materialise a temporary
        // filled with max_seq_length when the caller doesn't supply one.
        std::vector<int> default_lengths;
        const int* lens = seq_lengths;
        if (!lens) {
            default_lengths.assign(static_cast<size_t>(batch_size), max_seq_length);
            lens = default_lengths.data();
        }
        CUDNN_CHECK(cudnnSetRNNDataDescriptor(
            desc_,
            dtype,
            layout,
            max_seq_length,
            batch_size,
            vector_size,
            lens,
            padding_fill));
    }

    cudnnRNNDataDescriptor_t desc_ = nullptr;
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
 * @brief Per-axis Conv2D forward (audit E1).
 *
 * Accepts separate `stride_h`/`stride_w`, `pad_h`/`pad_w`, `dil_h`/`dil_w`,
 * which cuDNN's `cudnnSetConvolution2dDescriptor` supports natively. The
 * scalar overload above delegates to this one with duplicated values.
 *
 * Removes the "asymmetric stride/padding/dilation not supported" gate that
 * existed on GPU backends.
 */
auto cudnn_conv2d_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
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
    void* stream
) -> Tensor;

/**
 * @brief Fused Conv2D + Bias + ReLU using cudnnConvolutionBiasActivationForward
 *
 * Single cuDNN call that fuses convolution, bias addition, and ReLU activation.
 */
auto cudnn_fused_conv2d_activation_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudnnActivationMode_t activation_mode,
    double activation_coeff,
    cudaStream_t stream
) -> Tensor;

/// Per-axis fused Conv2d + Bias + Activation (Phase 2.1).
auto cudnn_fused_conv2d_activation_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups,
    cudnnActivationMode_t activation_mode,
    double activation_coeff,
    cudaStream_t stream
) -> Tensor;

auto cudnn_fused_conv2d_relu_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor;

/// Per-axis overload (Phase 2.1).
auto cudnn_fused_conv2d_relu_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups,
    cudaStream_t stream
) -> Tensor;

auto cudnn_fused_conv2d_sigmoid_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor;

/// Per-axis overload (Phase 2.1).
auto cudnn_fused_conv2d_sigmoid_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups,
    cudaStream_t stream
) -> Tensor;

auto cudnn_fused_conv2d_tanh_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor;

/// Per-axis overload (Phase 2.1).
auto cudnn_fused_conv2d_tanh_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups,
    cudaStream_t stream
) -> Tensor;

auto cudnn_fused_conv2d_swish_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor;

/// Per-axis overload (Phase 2.1).
auto cudnn_fused_conv2d_swish_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
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
 * @brief Per-axis Conv2D backward (audit E1). See per-axis forward above.
 */
auto cudnn_conv2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
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
    void* stream
) -> std::tuple<Tensor, Tensor, Tensor>;

// ============================================================================
// cuDNN RNN/LSTM/GRU Operations (cuDNN v8 RNN API)
// ============================================================================

/**
 * @brief Result bundle returned from `cudnn_lstm_forward`.
 *
 * `output`        — (seq_len, batch, hidden * num_directions).
 * `hy`            — (num_layers * num_directions, batch, hidden_or_proj).
 * `cy`            — (num_layers * num_directions, batch, hidden).
 * `reserve_space` — opaque buffer needed by `cudnn_lstm_backward`. Empty
 *                   tensor when `fwd_mode == CUDNN_FWD_MODE_INFERENCE`.
 * `weight_space`  — packed weight buffer that cuDNN consumed. Required by
 *                   the backward pass so it can read the same packing.
 */
struct CudnnLSTMOutputs {
    Tensor output;
    Tensor hy;
    Tensor cy;
    Tensor reserve_space;
    Tensor weight_space;
};

struct CudnnRNNOutputs {
    Tensor output;
    Tensor hy;
    Tensor reserve_space;
    Tensor weight_space;
};

/**
 * @brief Full-sequence LSTM forward via cuDNN's v8 RNN API.
 *
 * Accepts the per-layer / per-direction weights in PyTorch / Tenzor layout
 * and internally packs them into cuDNN's weight space using
 * `cudnnGetRNNWeightParams`. The `weight_space` returned in the result must
 * be passed verbatim to the backward call.
 *
 * @param input            (seq_len, batch, input_size), time-major.
 * @param hx               (num_layers * num_directions, batch, hidden_or_proj).
 *                         Empty tensor zero-initialises the hidden state.
 * @param cx               (num_layers * num_directions, batch, hidden_size).
 *                         Empty tensor zero-initialises the cell state.
 * @param weights_ih       Per layer / direction. shape (4*hidden, input_size_l).
 * @param weights_hh       Per layer / direction. shape (4*hidden, hidden_or_proj).
 * @param biases_ih        Per layer / direction. shape (4*hidden). Empty tensor
 *                         entries treated as zero.
 * @param biases_hh        Same, for the recurrent bias.
 * @param weights_hr       Optional recurrent projection weight per layer (one
 *                         per direction). Shape (proj_size, hidden_size). Pass
 *                         empty vector when `proj_size == 0`.
 * @param hidden_size      LSTM cell hidden width.
 * @param proj_size        LSTMP projection width, or 0 if unused.
 * @param num_layers       Number of stacked layers.
 * @param bidirectional    Bidirectional LSTM if true.
 * @param dropout          Inter-layer dropout probability. Ignored when
 *                         `fwd_mode == CUDNN_FWD_MODE_INFERENCE`.
 * @param fwd_mode         INFERENCE skips the reserve-space allocation;
 *                         TRAINING populates it for the backward call.
 * @param stream           CUDA stream.
 */
auto cudnn_lstm_forward(
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,
    const std::vector<Tensor>& weights_ih,
    const std::vector<Tensor>& weights_hh,
    const std::vector<Tensor>& biases_ih,
    const std::vector<Tensor>& biases_hh,
    const std::vector<Tensor>& weights_hr,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudnnForwardMode_t fwd_mode,
    cudaStream_t stream
) -> CudnnLSTMOutputs;

/**
 * @brief Bundle returned from `cudnn_lstm_backward`.
 *
 * `grad_input`       — gradient w.r.t. `input`.
 * `grad_hx`          — gradient w.r.t. initial hidden state.
 * `grad_cx`          — gradient w.r.t. initial cell state.
 * `grad_weight_space` — packed weight-gradient buffer. Caller is expected to
 *                       split it back into per-layer tensors via
 *                       `cudnn_lstm_unpack_weight_grads`.
 */
struct CudnnLSTMGrads {
    Tensor grad_input;
    Tensor grad_hx;
    Tensor grad_cx;
    Tensor grad_weight_space;
};

struct CudnnRNNGrads {
    Tensor grad_input;
    Tensor grad_hx;
    Tensor grad_weight_space;
};

/**
 * @brief Full-sequence LSTM backward via `cudnnRNNBackwardData_v8` +
 * `cudnnRNNBackwardWeights_v8`.
 *
 * Requires the forward pass to have been executed with
 * `fwd_mode == CUDNN_FWD_MODE_TRAINING`; the `reserve_space` and
 * `weight_space` from `CudnnLSTMOutputs` must be forwarded verbatim.
 *
 * @param grad_output      Gradient w.r.t. the per-step output. Shape matches
 *                         `output`.
 * @param grad_hy          Gradient w.r.t. final hidden state. Empty tensor =
 *                         no contribution.
 * @param grad_cy          Gradient w.r.t. final cell state. Empty tensor =
 *                         no contribution.
 * @param input / hx / cx  Original forward inputs (cuDNN needs them for both
 *                         data and weight gradients).
 * @param output           Forward output tensor (cuDNN requires re-reading
 *                         the activations alongside the reserve space).
 * @param weight_space     The same buffer returned by the forward call.
 * @param reserve_space    The same buffer returned by the forward call.
 * @param hidden_size / proj_size / num_layers / bidirectional / dropout —
 *                         must match the forward configuration.
 * @param stream           CUDA stream.
 */
auto cudnn_lstm_backward(
    const Tensor& grad_output,
    const Tensor& grad_hy,
    const Tensor& grad_cy,
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& output,
    const Tensor& weight_space,
    const Tensor& reserve_space,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudaStream_t stream
) -> CudnnLSTMGrads;

/**
 * @brief Split a packed weight gradient buffer back into the
 * `(W_ih, W_hh, b_ih, b_hh, W_hr)` layout used by the LSTM layer.
 *
 * The returned vectors are sized `num_layers * num_directions`, indexed as
 * `[layer * num_directions + dir]`. `out_W_hr` is empty when `proj_size == 0`.
 */
void cudnn_lstm_unpack_weight_grads(
    const Tensor& weight_space_grad,
    int64_t input_size,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool bidirectional,
    std::vector<Tensor>& out_W_ih,
    std::vector<Tensor>& out_W_hh,
    std::vector<Tensor>& out_b_ih,
    std::vector<Tensor>& out_b_hh,
    std::vector<Tensor>& out_W_hr,
    cudaStream_t stream
);

/**
 * @brief Full-sequence GRU forward via cuDNN's v8 RNN API. Mirrors the LSTM
 * variant but omits the cell state and the recurrent projection.
 */
auto cudnn_gru_forward(
    const Tensor& input,
    const Tensor& hx,
    const std::vector<Tensor>& weights_ih,
    const std::vector<Tensor>& weights_hh,
    const std::vector<Tensor>& biases_ih,
    const std::vector<Tensor>& biases_hh,
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudnnForwardMode_t fwd_mode,
    cudaStream_t stream
) -> CudnnRNNOutputs;

auto cudnn_gru_backward(
    const Tensor& grad_output,
    const Tensor& grad_hy,
    const Tensor& input,
    const Tensor& hx,
    const Tensor& output,
    const Tensor& weight_space,
    const Tensor& reserve_space,
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudaStream_t stream
) -> CudnnRNNGrads;

/**
 * @brief Full-sequence Elman RNN (tanh / ReLU) forward via cuDNN's v8 RNN API.
 *
 * `cell_mode` selects the pointwise activation:
 *   - `CUDNN_RNN_TANH` — `h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)`
 *   - `CUDNN_RNN_RELU` — same with ReLU instead of tanh.
 */
auto cudnn_rnn_forward(
    const Tensor& input,
    const Tensor& hx,
    const std::vector<Tensor>& weights_ih,
    const std::vector<Tensor>& weights_hh,
    const std::vector<Tensor>& biases_ih,
    const std::vector<Tensor>& biases_hh,
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudnnRNNMode_t cell_mode,
    cudnnForwardMode_t fwd_mode,
    cudaStream_t stream
) -> CudnnRNNOutputs;

auto cudnn_rnn_backward(
    const Tensor& grad_output,
    const Tensor& grad_hy,
    const Tensor& input,
    const Tensor& hx,
    const Tensor& output,
    const Tensor& weight_space,
    const Tensor& reserve_space,
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudnnRNNMode_t cell_mode,
    cudaStream_t stream
) -> CudnnRNNGrads;

/**
 * @brief Split a packed GRU / RNN weight gradient back into per-layer
 * `(W_ih, W_hh, b_ih, b_hh)` tensors. `gates_per_cell` is 3 for GRU and 1 for
 * Elman RNN; the LSTM variant has its own helper.
 */
void cudnn_rnn_unpack_weight_grads(
    const Tensor& weight_space_grad,
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    cudnnRNNMode_t cell_mode,
    std::vector<Tensor>& out_W_ih,
    std::vector<Tensor>& out_W_hh,
    std::vector<Tensor>& out_b_ih,
    std::vector<Tensor>& out_b_hh,
    cudaStream_t stream
);

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

/**
 * @brief Per-axis MaxPool2d forward (Phase 2.1).
 *
 * Underlying cuDNN PoolingDescriptor accepts separate `(kernel_h, kernel_w,
 * pad_h, pad_w, stride_h, stride_w)`; the scalar overload above delegates to
 * this one with duplicated values.
 */
auto cudnn_maxpool2d_forward(
    const Tensor& input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    cudaStream_t stream
) -> std::pair<Tensor, Tensor>;

auto cudnn_maxpool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& output,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    cudaStream_t stream
) -> Tensor;

/// Per-axis MaxPool2d backward (Phase 2.1).
auto cudnn_maxpool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& output,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    cudaStream_t stream
) -> Tensor;

auto cudnn_avgpool2d_forward(
    const Tensor& input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    cudaStream_t stream
) -> Tensor;

/// Per-axis AvgPool2d forward (Phase 2.1).
auto cudnn_avgpool2d_forward(
    const Tensor& input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
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

/// Per-axis AvgPool2d backward (Phase 2.1).
auto cudnn_avgpool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
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
