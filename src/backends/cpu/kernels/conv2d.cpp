#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// Float16 Arithmetic Helper Functions
// ============================================================================
// These inline helpers allow Float16 to work with template code that uses
// arithmetic operators. Operations are performed in Float32 precision.

inline Float16 operator+(const Float16& a, const Float16& b) {
    return Float16(static_cast<float>(a) + static_cast<float>(b));
}

inline Float16 operator-(const Float16& a, const Float16& b) {
    return Float16(static_cast<float>(a) - static_cast<float>(b));
}

inline Float16 operator*(const Float16& a, const Float16& b) {
    return Float16(static_cast<float>(a) * static_cast<float>(b));
}

inline Float16 operator/(const Float16& a, const Float16& b) {
    return Float16(static_cast<float>(a) / static_cast<float>(b));
}

inline Float16& operator+=(Float16& a, const Float16& b) {
    a = Float16(static_cast<float>(a) + static_cast<float>(b));
    return a;
}

inline Float16& operator-=(Float16& a, const Float16& b) {
    a = Float16(static_cast<float>(a) - static_cast<float>(b));
    return a;
}

inline Float16& operator*=(Float16& a, const Float16& b) {
    a = Float16(static_cast<float>(a) * static_cast<float>(b));
    return a;
}

inline Float16& operator/=(Float16& a, const Float16& b) {
    a = Float16(static_cast<float>(a) / static_cast<float>(b));
    return a;
}

// ============================================================================
// Helper Functions
// ============================================================================

// Calculate output size for convolution
inline int64_t calculate_output_size(int64_t input_size, int64_t kernel_size,
                                     int64_t stride, int64_t padding, int64_t dilation) {
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

// ============================================================================
// im2col CPU Implementation
// ============================================================================

// im2col: Convert 4D input (N,C,H,W) to 2D matrix for convolution
// Input: (batch, in_channels, height, width)
// Output: (batch * out_h * out_w, kernel_h * kernel_w * in_channels)
template<typename T>
void im2col_cpu(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t out_h,
    int64_t out_w
) {
    int64_t total_elements = batch * out_h * out_w * channels * kernel_h * kernel_w;

    #pragma omp parallel for if(total_elements > 10000)
    for (int64_t idx = 0; idx < total_elements; ++idx) {
        // Decode flat index to (b, oh, ow, c, kh, kw)
        int64_t temp = idx;
        int64_t kw = temp % kernel_w; temp /= kernel_w;
        int64_t kh = temp % kernel_h; temp /= kernel_h;
        int64_t c = temp % channels; temp /= channels;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t b = temp;

        // Calculate input position with padding and dilation
        int64_t ih = oh * stride - padding + kh * dilation;
        int64_t iw = ow * stride - padding + kw * dilation;

        // Output index in col matrix
        // Shape: (batch * out_h * out_w, channels * kernel_h * kernel_w)
        int64_t out_row = b * out_h * out_w + oh * out_w + ow;
        int64_t out_col = c * kernel_h * kernel_w + kh * kernel_w + kw;
        int64_t out_idx = out_row * (channels * kernel_h * kernel_w) + out_col;

        // Check bounds and apply padding
        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            int64_t input_idx = b * (channels * height * width) +
                               c * (height * width) +
                               ih * width + iw;
            output[out_idx] = input[input_idx];
        } else {
            output[out_idx] = T(0.0f);  // Padding with zeros
        }
    }
}

// ============================================================================
// col2im CPU Implementation
// ============================================================================

// col2im: Reverse of im2col for gradient computation
// Input: (batch * out_h * out_w, kernel_h * kernel_w * in_channels)
// Output: (batch, in_channels, height, width)
// Note: This accumulates gradients for overlapping regions
template<typename T>
void col2im_cpu(
    const T* col,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t out_h,
    int64_t out_w
) {
    // Zero initialize output
    int64_t output_size = batch * channels * height * width;
    std::memset(output, 0, output_size * sizeof(T));

    // Process each output element and accumulate from all contributing col positions
    // This avoids race conditions compared to the col-centric approach
    #pragma omp parallel for collapse(4) if(output_size > 10000)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t ih = 0; ih < height; ++ih) {
                for (int64_t iw = 0; iw < width; ++iw) {
                    // Accumulate from all kernel positions that contribute to this output
                    T sum = T(0.0f);

                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            // Reverse the mapping: given (ih, iw) and (kh, kw), find (oh, ow)
                            int64_t ih_shifted = ih + padding - kh * dilation;
                            int64_t iw_shifted = iw + padding - kw * dilation;

                            // Check if this maps to a valid output position
                            if (ih_shifted % stride == 0 && iw_shifted % stride == 0) {
                                int64_t oh = ih_shifted / stride;
                                int64_t ow = iw_shifted / stride;

                                if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                    // This kernel position contributes to our output
                                    int64_t col_row = b * out_h * out_w + oh * out_w + ow;
                                    int64_t col_col = c * kernel_h * kernel_w + kh * kernel_w + kw;
                                    int64_t col_idx = col_row * (channels * kernel_h * kernel_w) + col_col;

                                    sum += col[col_idx];
                                }
                            }
                        }
                    }

                    // Write accumulated value
                    int64_t output_idx = b * (channels * height * width) +
                                        c * (height * width) +
                                        ih * width + iw;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

// ============================================================================
// Matrix Multiplication Helper (Row-major GEMM)
// ============================================================================

// Simple blocked matrix multiplication for C = A @ B^T
// A: (M, K) row-major
// B: (N, K) row-major (will be transposed)
// C: (M, N) row-major
template<typename T>
void gemm_cpu(
    const T* A, const T* B, T* C,
    int64_t M, int64_t N, int64_t K,
    bool transpose_B = true
) {
    #pragma omp parallel for collapse(2) if(M * N > 1000)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            T sum = T(0.0f);
            if (transpose_B) {
                // B is (N, K) row-major, access as B[j][k]
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[j * K + k];
                }
            } else {
                // B is (K, N) row-major, access as B[k][j]
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
            }
            C[i * N + j] = sum;
        }
    }
}

// Matrix multiplication for C = A^T @ B
// A: (K, M) row-major (will be transposed)
// B: (K, N) row-major
// C: (M, N) row-major
template<typename T>
void gemm_transA_cpu(
    const T* A, const T* B, T* C,
    int64_t M, int64_t N, int64_t K
) {
    #pragma omp parallel for collapse(2) if(M * N > 1000)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            T sum = T(0.0f);
            // A is (K, M) row-major, access as A[k][i]
            for (int64_t k = 0; k < K; ++k) {
                sum += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Specialized version for Float16 (uses Float32 accumulation)
template<>
void gemm_transA_cpu<Float16>(
    const Float16* A, const Float16* B, Float16* C,
    int64_t M, int64_t N, int64_t K
) {
    #pragma omp parallel for collapse(2) if(M * N > 1000)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            // A is (K, M) row-major, access as A[k][i]
            for (int64_t k = 0; k < K; ++k) {
                float a_val = static_cast<float>(A[k * M + i]);
                float b_val = static_cast<float>(B[k * N + j]);
                sum += a_val * b_val;
            }
            C[i * N + j] = Float16(sum);
        }
    }
}

// ============================================================================
// Conv2d Forward CPU Implementation
// ============================================================================

// Template helper for dtype-generic conv2d forward
template<typename T>
void conv2d_forward_impl(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& output,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t in_channels_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    auto out_shape = output.shape();
    int64_t out_h = out_shape[2];
    int64_t out_w = out_shape[3];

    // Initialize output to zeros
    std::memset(output.data<T>(), 0, output.numel() * sizeof(T));

    // Process each group separately
    int64_t out_channels_per_group = out_channels / groups;

    for (int64_t g = 0; g < groups; ++g) {
        // Calculate channel offsets
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Allocate im2col buffer for this group
        int64_t col_rows = batch * out_h * out_w;
        int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;
        std::vector<T> col_buffer(col_rows * col_cols);

        // Apply im2col transformation for this group's input channels
        const T* input_ptr = input.data<T>() + in_start * height * width;
        im2col_cpu(
            input_ptr,
            col_buffer.data(),
            batch,
            in_channels_per_group,
            height,
            width,
            kernel_h,
            kernel_w,
            stride,
            padding,
            dilation,
            out_h,
            out_w
        );

        // Matrix multiplication
        int64_t M = col_rows;
        int64_t K = col_cols;
        int64_t N = out_channels_per_group;

        const T* weight_ptr = weight.data<T>() + out_start * in_channels_per_group * kernel_h * kernel_w;
        T* output_ptr = output.data<T>() + out_start * out_h * out_w;

        // Perform GEMM: C = A @ B^T
        gemm_cpu(
            col_buffer.data(),  // A: (M, K)
            weight_ptr,         // B: (N, K) - will be transposed
            output_ptr,         // C: (M, N)
            M, N, K,
            true  // transpose B
        );
    }

    // Add bias if present
    if (bias != nullptr) {
        const T* bias_data = bias->data<T>();
        T* output_data = output.data<T>();

        #pragma omp parallel for collapse(4)
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < out_channels; ++c) {
                for (int64_t h = 0; h < out_h; ++h) {
                    for (int64_t w = 0; w < out_w; ++w) {
                        int64_t idx = b * (out_channels * out_h * out_w) +
                                     c * (out_h * out_w) +
                                     h * out_w + w;
                        output_data[idx] += bias_data[c];
                    }
                }
            }
        }
    }
}

auto conv2d_forward_kernel(
    const Tensor& input,         // (batch, in_channels, height, width)
    const Tensor& weight,        // (out_channels, in_channels, kernel_h, kernel_w)
    const Tensor* bias,          // (out_channels) or nullptr
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) -> Tensor {
    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t out_channels = weight_shape[0];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions
    int64_t out_h = calculate_output_size(height, kernel_h, stride, padding, dilation);
    int64_t out_w = calculate_output_size(width, kernel_w, stride, padding, dilation);

    // Create output tensor with correct dtype
    std::vector<int64_t> output_shape = {batch, out_channels, out_h, out_w};
    Tensor output(output_shape, input.dtype(), input.device());

    // Dispatch based on dtype
    if (input.dtype() == DType::Float32) {
        conv2d_forward_impl<float>(input, weight, bias, output, stride, padding, dilation, groups);
    } else if (input.dtype() == DType::Float64) {
        conv2d_forward_impl<double>(input, weight, bias, output, stride, padding, dilation, groups);
    } else if (input.dtype() == DType::Float16) {
        conv2d_forward_impl<Float16>(input, weight, bias, output, stride, padding, dilation, groups);
    } else {
        throw std::runtime_error("Unsupported dtype for conv2d_forward");
    }

    return output;
}

// ============================================================================
// Conv2d Backward Input CPU Implementation
// ============================================================================

// Template helper for dtype-generic conv2d backward input
template<typename T>
void conv2d_backward_input_impl(
    const Tensor& grad_output,
    const Tensor& weight,
    Tensor& grad_input,
    const std::vector<int64_t>& input_shape,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto weight_shape = weight.shape();
    auto grad_shape = grad_output.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t in_channels_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h = grad_shape[2];
    int64_t out_w = grad_shape[3];

    int64_t out_channels_per_group = out_channels / groups;
    int64_t col_rows = batch * out_h * out_w;
    int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;

    // Zero-initialize gradient input
    std::memset(grad_input.data<T>(), 0, grad_input.numel() * sizeof(T));

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Allocate col buffer
        std::vector<T> grad_col(col_rows * col_cols);

        int64_t M = col_rows;
        int64_t K = out_channels_per_group;
        int64_t N = col_cols;

        const T* grad_out_ptr = grad_output.data<T>() + out_start * out_h * out_w;
        const T* weight_ptr = weight.data<T>() + out_start * in_channels_per_group * kernel_h * kernel_w;

        // Perform GEMM: C = A @ B (no transpose)
        gemm_cpu<T>(
            grad_out_ptr,       // A: (M, K)
            weight_ptr,         // B: (K, N) - already in correct orientation
            grad_col.data(),    // C: (M, N)
            M, N, K,
            false  // don't transpose B
        );

        // Apply col2im to accumulate gradients
        T* grad_input_ptr = grad_input.data<T>() + in_start * height * width;
        col2im_cpu<T>(
            grad_col.data(),
            grad_input_ptr,
            batch,
            in_channels_per_group,
            height,
            width,
            kernel_h,
            kernel_w,
            stride,
            padding,
            dilation,
            out_h,
            out_w
        );
    }
}

auto conv2d_backward_input_kernel(
    const Tensor& grad_output,   // (batch, out_channels, out_h, out_w)
    const Tensor& weight,        // (out_channels, in_channels, kernel_h, kernel_w)
    const std::vector<int64_t>& input_shape,  // (batch, in_channels, height, width)
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) -> Tensor {
    // Initialize gradient w.r.t input with correct dtype
    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    // Dispatch based on dtype
    if (grad_output.dtype() == DType::Float32) {
        conv2d_backward_input_impl<float>(grad_output, weight, grad_input, input_shape, stride, padding, dilation, groups);
    } else if (grad_output.dtype() == DType::Float64) {
        conv2d_backward_input_impl<double>(grad_output, weight, grad_input, input_shape, stride, padding, dilation, groups);
    } else if (grad_output.dtype() == DType::Float16) {
        conv2d_backward_input_impl<Float16>(grad_output, weight, grad_input, input_shape, stride, padding, dilation, groups);
    } else {
        throw std::runtime_error("Unsupported dtype for conv2d_backward_input");
    }

    return grad_input;
}

// ============================================================================
// Conv2d Backward Weight CPU Implementation
// ============================================================================

// Template helper for dtype-generic conv2d backward weight
template<typename T>
void conv2d_backward_weight_impl(
    const Tensor& grad_output,
    const Tensor& input,
    Tensor& grad_weight,
    const std::vector<int64_t>& weight_shape,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto input_shape = input.shape();
    auto grad_shape = grad_output.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t in_channels_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h = grad_shape[2];
    int64_t out_w = grad_shape[3];

    int64_t out_channels_per_group = out_channels / groups;
    int64_t col_rows = batch * out_h * out_w;
    int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;

    // Zero-initialize gradient weight
    std::memset(grad_weight.data<T>(), 0, grad_weight.numel() * sizeof(T));

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Apply im2col to input
        std::vector<T> input_col(col_rows * col_cols);

        const T* input_ptr = input.data<T>() + in_start * height * width;
        im2col_cpu<T>(
            input_ptr,
            input_col.data(),
            batch,
            in_channels_per_group,
            height,
            width,
            kernel_h,
            kernel_w,
            stride,
            padding,
            dilation,
            out_h,
            out_w
        );

        int64_t M = out_channels_per_group;
        int64_t K = col_rows;
        int64_t N = col_cols;

        const T* grad_out_ptr = grad_output.data<T>() + out_start * out_h * out_w;
        T* grad_weight_ptr = grad_weight.data<T>() + out_start * in_channels_per_group * kernel_h * kernel_w;

        // Perform GEMM: C = A^T @ B
        gemm_transA_cpu<T>(
            grad_out_ptr,       // A: (K, M) - will be transposed
            input_col.data(),   // B: (K, N)
            grad_weight_ptr,    // C: (M, N)
            M, N, K
        );
    }
}

auto conv2d_backward_weight_kernel(
    const Tensor& grad_output,   // (batch, out_channels, out_h, out_w)
    const Tensor& input,         // (batch, in_channels, height, width)
    const std::vector<int64_t>& weight_shape,  // (out_channels, in_channels, kernel_h, kernel_w)
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) -> Tensor {
    // Initialize gradient w.r.t weight
    Tensor grad_weight(weight_shape, grad_output.dtype(), grad_output.device());

    // Dispatch based on dtype
    if (grad_output.dtype() == DType::Float32) {
        conv2d_backward_weight_impl<float>(grad_output, input, grad_weight, weight_shape, stride, padding, dilation, groups);
    } else if (grad_output.dtype() == DType::Float64) {
        conv2d_backward_weight_impl<double>(grad_output, input, grad_weight, weight_shape, stride, padding, dilation, groups);
    } else if (grad_output.dtype() == DType::Float16) {
        conv2d_backward_weight_impl<Float16>(grad_output, input, grad_weight, weight_shape, stride, padding, dilation, groups);
    } else {
        throw std::runtime_error("Unsupported dtype for conv2d_backward_weight");
    }

    return grad_weight;
}

// ============================================================================
// Conv2d Backward Bias CPU Implementation
// ============================================================================

// Template helper for dtype-generic conv2d backward bias
template<typename T>
void conv2d_backward_bias_impl(
    const Tensor& grad_output,
    Tensor& grad_bias
) {
    auto grad_shape = grad_output.shape();
    int64_t batch = grad_shape[0];
    int64_t out_channels = grad_shape[1];
    int64_t out_h = grad_shape[2];
    int64_t out_w = grad_shape[3];

    T* grad_bias_data = grad_bias.data<T>();
    std::memset(grad_bias_data, 0, out_channels * sizeof(T));

    const T* grad_out_data = grad_output.data<T>();

    // Sum over batch, height, width dimensions
    #pragma omp parallel for
    for (int64_t c = 0; c < out_channels; ++c) {
        T sum = T(0.0f);
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < out_h; ++h) {
                for (int64_t w = 0; w < out_w; ++w) {
                    int64_t idx = b * (out_channels * out_h * out_w) +
                                 c * (out_h * out_w) +
                                 h * out_w + w;
                    sum += grad_out_data[idx];
                }
            }
        }
        grad_bias_data[c] = sum;
    }
}

auto conv2d_backward_bias_kernel(
    const Tensor& grad_output    // (batch, out_channels, out_h, out_w)
) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t out_channels = grad_shape[1];

    // Initialize gradient w.r.t bias
    Tensor grad_bias({out_channels}, grad_output.dtype(), grad_output.device());

    // Dispatch based on dtype
    if (grad_output.dtype() == DType::Float32) {
        conv2d_backward_bias_impl<float>(grad_output, grad_bias);
    } else if (grad_output.dtype() == DType::Float64) {
        conv2d_backward_bias_impl<double>(grad_output, grad_bias);
    } else if (grad_output.dtype() == DType::Float16) {
        conv2d_backward_bias_impl<Float16>(grad_output, grad_bias);
    } else {
        throw std::runtime_error("Unsupported dtype for conv2d_backward_bias");
    }

    return grad_bias;
}

} // namespace cpu
} // namespace tenzor
