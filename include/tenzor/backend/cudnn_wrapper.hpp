#pragma once

#ifdef TENZOR_HAS_CUDNN

#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <memory>
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
// cuDNN Handle Manager (RAII)
// ============================================================================

class CuDNNHandle {
public:
    CuDNNHandle() {
        CUDNN_CHECK(cudnnCreate(&handle_));
    }

    ~CuDNNHandle() {
        if (handle_) {
            cudnnDestroy(handle_);
        }
    }

    // No copy
    CuDNNHandle(const CuDNNHandle&) = delete;
    CuDNNHandle& operator=(const CuDNNHandle&) = delete;

    // Move support
    CuDNNHandle(CuDNNHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    CuDNNHandle& operator=(CuDNNHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                cudnnDestroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    cudnnHandle_t get() const { return handle_; }

    void set_stream(cudaStream_t stream) {
        CUDNN_CHECK(cudnnSetStream(handle_, stream));
    }

private:
    cudnnHandle_t handle_ = nullptr;
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

    void set(cudnnDataType_t dtype, int n, int c, int h, int w) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            desc_,
            CUDNN_TENSOR_NCHW,
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

private:
    cudnnTensorDescriptor_t desc_ = nullptr;

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

    void set(cudnnDataType_t dtype, int k, int c, int h, int w) {
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            desc_,
            dtype,
            CUDNN_TENSOR_NCHW,
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

        // Enable Tensor Cores for FP16/FP32 on compatible GPUs
        #ifdef TENZOR_HAS_TENSOR_CORES
        if (dtype == CUDNN_DATA_FLOAT || dtype == CUDNN_DATA_HALF) {
            CUDNN_CHECK(cudnnSetConvolutionMathType(desc_, CUDNN_TENSOR_OP_MATH));
        }
        #endif
    }

    void set_group_count(int groups) {
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(desc_, groups));
    }

private:
    cudnnConvolutionDescriptor_t desc_ = nullptr;
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

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_HAS_CUDNN
