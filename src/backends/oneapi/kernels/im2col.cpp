#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <type_traits>  // FF.7: std::is_same_v in col2im atomic_ref guard

namespace tenzor {
namespace oneapi {

// Kernel class declarations for im2col/col2im operations (separate classes per dtype)
struct Im2colKernelFloat32 {};
struct Im2colKernelFloat64 {};
struct Im2colKernelFloat16 {};
struct Im2colKernelBFloat16 {};
struct Col2imKernelFloat32 {};
struct Col2imKernelFloat64 {};
struct Col2imKernelFloat16 {};
struct Col2imKernelBFloat16 {};

// Compile-time kernel-tag selector for im2col/col2im over scalar T.
template<typename T> struct Im2colTag;
template<> struct Im2colTag<float>      { using type = Im2colKernelFloat32; };
template<> struct Im2colTag<double>     { using type = Im2colKernelFloat64; };
template<> struct Im2colTag<sycl::half> { using type = Im2colKernelFloat16; };
// BFloat16 is represented as uint16_t at storage; im2col is a pure data
// shuffle (no arithmetic) so we template directly on uint16_t.
template<> struct Im2colTag<uint16_t>   { using type = Im2colKernelBFloat16; };

template<typename T> struct Col2imTag;
template<> struct Col2imTag<float>      { using type = Col2imKernelFloat32; };
template<> struct Col2imTag<double>     { using type = Col2imKernelFloat64; };
template<> struct Col2imTag<sycl::half> { using type = Col2imKernelFloat16; };

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

/**
 * @brief Im2col transformation kernel implementation.
 *
 * Transforms an image tensor into a column matrix suitable for convolution via GEMM.
 * Each column in the output represents the receptive field for one output position.
 *
 * @tparam T Data type (float or double)
 * @param data_im Input image data
 * @param channels Number of input channels
 * @param height Input height
 * @param width Input width
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param pad Padding size
 * @param stride Stride size
 * @param dilation Dilation size
 * @param data_col Output column matrix
 * @param queue SYCL queue for execution
 */
template<typename T>
void im2col_kernel_impl(const T* data_im, int64_t channels, int64_t height, int64_t width,
                        int64_t kernel_h, int64_t kernel_w, int64_t pad, int64_t stride,
                        int64_t dilation, T* data_col, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int64_t output_w = (width + 2 * pad - dilation * (kernel_w - 1) - 1) / stride + 1;
    const int64_t col_size = channels * kernel_h * kernel_w * output_h * output_w;

    using KernelClass = typename Im2colTag<T>::type;
    queue.parallel_for<KernelClass>(sycl::range<1>(col_size), [=](sycl::id<1> index) {
        int64_t w_out = index % output_w;
        int64_t idx = index / output_w;
        int64_t h_out = idx % output_h;
        idx /= output_h;
        int64_t kw = idx % kernel_w;
        idx /= kernel_w;
        int64_t kh = idx % kernel_h;
        int64_t c = idx / kernel_h;

        int64_t h_in = h_out * stride - pad + kh * dilation;
        int64_t w_in = w_out * stride - pad + kw * dilation;

        data_col[index] = (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) ?
            data_im[(c * height + h_in) * width + w_in] : T(0);
    });
}

/**
 * @brief Col2im transformation kernel implementation.
 *
 * Inverse of im2col - transforms a column matrix back into an image tensor.
 * Used for backpropagation through convolution operations.
 *
 * @tparam T Data type (float or double)
 * @param data_col Input column matrix
 * @param channels Number of input channels
 * @param height Output height
 * @param width Output width
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param pad Padding size
 * @param stride Stride size
 * @param dilation Dilation size
 * @param data_im Output image data
 * @param queue SYCL queue for execution
 */
template<typename T>
void col2im_kernel_impl(const T* data_col, int64_t channels, int64_t height, int64_t width,
                        int64_t kernel_h, int64_t kernel_w, int64_t pad, int64_t stride,
                        int64_t dilation, T* data_im, sycl::queue& queue) {
    // FF.7: sycl::atomic_ref<T, ...> is only specified for 32-/64-bit IEEE
    // floats (and integers).  Half / bfloat16 instantiations either fail to
    // compile or silently fall back to non-atomic — gate to F32/F64 so the
    // host dispatcher widens lower precision through the F32 path.
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
        "im2col atomic_ref requires F32/F64");
    const int64_t output_h = (height + 2 * pad - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int64_t output_w = (width + 2 * pad - dilation * (kernel_w - 1) - 1) / stride + 1;
    const int64_t im_size = channels * height * width;

    // Initialize grad_input to zero
    queue.fill(data_im, T(0), im_size);

    // Accumulate gradients from col buffer
    using KernelClass = typename Col2imTag<T>::type;
    queue.parallel_for<KernelClass>(sycl::range<1>(channels * kernel_h * kernel_w * output_h * output_w),
                      [=](sycl::id<1> index) {
        int64_t w_out = index % output_w;
        int64_t idx = index / output_w;
        int64_t h_out = idx % output_h;
        idx /= output_h;
        int64_t kw = idx % kernel_w;
        idx /= kernel_w;
        int64_t kh = idx % kernel_h;
        int64_t c = idx / kernel_h;

        int64_t h_in = h_out * stride - pad + kh * dilation;
        int64_t w_in = w_out * stride - pad + kw * dilation;

        if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
            int64_t im_idx = (c * height + h_in) * width + w_in;
            // Atomic add for thread safety
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
                atomic_val(data_im[im_idx]);
            atomic_val.fetch_add(data_col[index]);
        }
    });
}

/**
 * @brief Im2col operation for convolution.
 *
 * Converts an input tensor into a column matrix format suitable for convolution
 * via matrix multiplication. This is a standard technique for implementing
 * convolution efficiently using BLAS operations.
 *
 * Expected input shape: [batch, channels, height, width]
 * Output shape: [batch, channels * kernel_h * kernel_w, output_h * output_w]
 *
 * @param input Input tensor (4D: batch, channels, height, width)
 * @param attrs Operation attributes containing:
 *   - kernel_size: int64_t - Size of the convolution kernel (assumed square)
 *   - stride: int64_t - Stride of the convolution (default: 1)
 *   - padding: int64_t - Padding size (default: 0)
 *   - dilation: int64_t - Dilation factor (default: 1)
 * @param queue SYCL queue for execution
 * @return Tensor Column matrix representation
 */
auto im2col_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    // Validate input
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("im2col: input must be 4D tensor (batch, channels, height, width)");
    }

    // Extract parameters from attributes
    if (!attrs.has(AttrKey::KernelSize)) {
        throw std::invalid_argument("im2col: 'kernel_size' attribute is required");
    }

    int64_t kernel_size = attrs.get_int(AttrKey::KernelSize);
    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);

    const int64_t N = input_shape[0];        // Batch size
    const int64_t C = input_shape[1];        // Input channels
    const int64_t H = input_shape[2];        // Input height
    const int64_t W = input_shape[3];        // Input width
    const int64_t K_h = kernel_size;         // Kernel height
    const int64_t K_w = kernel_size;         // Kernel width

    // Calculate output dimensions
    const int64_t H_out = (H + 2 * padding - dilation * (K_h - 1) - 1) / stride + 1;
    const int64_t W_out = (W + 2 * padding - dilation * (K_w - 1) - 1) / stride + 1;

    // Create output tensor
    // Shape: [batch, channels * kernel_h * kernel_w, output_h * output_w]
    Tensor output({N, C * K_h * K_w, H_out * W_out}, input.dtype(), input.device());

    // Dispatch based on dtype
    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr<const float>(input);
        float* output_ptr = get_data_ptr<float>(output);

        // Process each batch
        for (int64_t n = 0; n < N; ++n) {
            im2col_kernel_impl<float>(
                input_ptr + n * C * H * W,
                C, H, W, K_h, K_w, padding, stride, dilation,
                output_ptr + n * C * K_h * K_w * H_out * W_out,
                queue
            );
        }
    }
    else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr<const double>(input);
        double* output_ptr = get_data_ptr<double>(output);

        for (int64_t n = 0; n < N; ++n) {
            im2col_kernel_impl<double>(
                input_ptr + n * C * H * W,
                C, H, W, K_h, K_w, padding, stride, dilation,
                output_ptr + n * C * K_h * K_w * H_out * W_out,
                queue
            );
        }
    }
    else if (input.dtype() == DType::Float16) {
        // Wave F2: native F16 — im2col is a pure data shuffle so no F32 widen
        // is needed; the SYCL half type is opaque to the algorithm.
        const sycl::half* input_ptr =
            reinterpret_cast<const sycl::half*>(input.data_ptr());
        sycl::half* output_ptr =
            reinterpret_cast<sycl::half*>(const_cast<void*>(output.data_ptr()));
        for (int64_t n = 0; n < N; ++n) {
            im2col_kernel_impl<sycl::half>(
                input_ptr + n * C * H * W,
                C, H, W, K_h, K_w, padding, stride, dilation,
                output_ptr + n * C * K_h * K_w * H_out * W_out,
                queue
            );
        }
    }
    else if (input.dtype() == DType::BFloat16) {
        // Wave F2: BF16 storage is uint16_t; im2col is a bit-pattern copy
        // (T(0) = 0 in BF16, same as uint16_t(0)) so we template directly on
        // uint16_t storage. No F32 widen, no tensor-wide cast.
        const uint16_t* input_ptr =
            reinterpret_cast<const uint16_t*>(input.data_ptr());
        uint16_t* output_ptr =
            reinterpret_cast<uint16_t*>(const_cast<void*>(output.data_ptr()));
        for (int64_t n = 0; n < N; ++n) {
            im2col_kernel_impl<uint16_t>(
                input_ptr + n * C * H * W,
                C, H, W, K_h, K_w, padding, stride, dilation,
                output_ptr + n * C * K_h * K_w * H_out * W_out,
                queue
            );
        }
    }
    else {
        throw std::runtime_error("im2col: Unsupported data type (only Float32/Float64/Float16/BFloat16 supported)");
    }

    return output;
}

/**
 * @brief Col2im operation (inverse of im2col).
 *
 * Converts a column matrix back into an image tensor. This is used in the
 * backward pass of convolution operations to compute gradients with respect
 * to the input.
 *
 * Expected input shape: [batch, channels * kernel_h * kernel_w, output_h * output_w]
 * Output shape: [batch, channels, height, width]
 *
 * @param input Input tensor (column matrix format)
 * @param attrs Operation attributes containing:
 *   - kernel_size: int64_t - Size of the convolution kernel (assumed square)
 *   - stride: int64_t - Stride of the convolution (default: 1)
 *   - padding: int64_t - Padding size (default: 0)
 *   - dilation: int64_t - Dilation factor (default: 1)
 *   - output_height: int64_t - Output tensor height
 *   - output_width: int64_t - Output tensor width
 * @param queue SYCL queue for execution
 * @return Tensor Image tensor
 */
auto col2im_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    // Validate input
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("col2im: input must be 3D tensor (batch, channels * kernel_h * kernel_w, spatial)");
    }

    // Extract parameters from attributes
    if (!attrs.has(AttrKey::KernelSize) || !attrs.has(AttrKey::OutputHeight) || !attrs.has(AttrKey::OutputWidth)) {
        throw std::invalid_argument("col2im: 'kernel_size', 'output_height', and 'output_width' attributes are required");
    }

    int64_t kernel_size = attrs.get_int(AttrKey::KernelSize);
    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
    int64_t output_height = attrs.get_int(AttrKey::OutputHeight);
    int64_t output_width = attrs.get_int(AttrKey::OutputWidth);

    const int64_t N = input_shape[0];                    // Batch size
    const int64_t K_h = kernel_size;                     // Kernel height
    const int64_t K_w = kernel_size;                     // Kernel width
    const int64_t spatial_size = input_shape[2];         // Output_h * Output_w

    // Infer number of channels
    const int64_t C = input_shape[1] / (K_h * K_w);
    if (input_shape[1] != C * K_h * K_w) {
        throw std::invalid_argument("col2im: input dimension mismatch - channels * kernel_h * kernel_w");
    }

    const int64_t H = output_height;
    const int64_t W = output_width;

    // Calculate expected spatial size
    const int64_t H_out = (H + 2 * padding - dilation * (K_h - 1) - 1) / stride + 1;
    const int64_t W_out = (W + 2 * padding - dilation * (K_w - 1) - 1) / stride + 1;

    if (spatial_size != H_out * W_out) {
        throw std::invalid_argument("col2im: spatial dimension mismatch");
    }

    // Create output tensor
    // Shape: [batch, channels, height, width]
    Tensor output({N, C, H, W}, input.dtype(), input.device());

    // Dispatch based on dtype
    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr<const float>(input);
        float* output_ptr = get_data_ptr<float>(output);

        // Process each batch
        for (int64_t n = 0; n < N; ++n) {
            col2im_kernel_impl<float>(
                input_ptr + n * C * K_h * K_w * H_out * W_out,
                C, H, W, K_h, K_w, padding, stride, dilation,
                output_ptr + n * C * H * W,
                queue
            );
        }
    }
    else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr<const double>(input);
        double* output_ptr = get_data_ptr<double>(output);

        // Process each batch
        for (int64_t n = 0; n < N; ++n) {
            col2im_kernel_impl<double>(
                input_ptr + n * C * K_h * K_w * H_out * W_out,
                C, H, W, K_h, K_w, padding, stride, dilation,
                output_ptr + n * C * H * W,
                queue
            );
        }
    }
    else {
        throw std::runtime_error("col2im: Unsupported data type (only Float32 and Float64 supported)");
    }

    return output;
}

} // namespace oneapi
} // namespace tenzor
