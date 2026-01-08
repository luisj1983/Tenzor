#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <stdexcept>
#include <vector>
#include <cmath>

namespace tenzor {
namespace rocm {

// ============================================================================
// HIP Error Checking
// ============================================================================

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)

// ============================================================================
// Kernel Launch Helpers
// ============================================================================

constexpr int BLOCK_SIZE = 256;

inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    return std::min(static_cast<int64_t>((n + block_size - 1) / block_size), static_cast<int64_t>(65535));
}

#define HIP_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// ============================================================================
// Unfold HIP Kernel (im2col)
// ============================================================================

template<typename T>
__global__ void unfold_kernel_hip(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t out_h,
    int64_t out_w
) {
    int64_t num_blocks = out_h * out_w;
    int64_t total_elements = batch * channels * kernel_size * kernel_size * num_blocks;

    HIP_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, c, kh, kw, block_idx)
        int64_t temp = idx;
        int64_t block_idx = temp % num_blocks; temp /= num_blocks;
        int64_t kw = temp % kernel_size; temp /= kernel_size;
        int64_t kh = temp % kernel_size; temp /= kernel_size;
        int64_t c = temp % channels; temp /= channels;
        int64_t b = temp;

        // Calculate output position from block_idx
        int64_t oh = block_idx / out_w;
        int64_t ow = block_idx % out_w;

        // Calculate input position with padding and dilation
        int64_t ih = oh * stride - padding + kh * dilation;
        int64_t iw = ow * stride - padding + kw * dilation;

        // Column index in output: (c * K * K + kh * K + kw)
        int64_t col_c = c * kernel_size * kernel_size + kh * kernel_size + kw;

        // Output index: (b, col_c, block_idx)
        int64_t output_idx = b * (channels * kernel_size * kernel_size * num_blocks) +
                            col_c * num_blocks +
                            block_idx;

        // Check bounds and apply padding
        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            int64_t input_idx = b * (channels * height * width) +
                               c * (height * width) +
                               ih * width + iw;
            output[output_idx] = input[input_idx];
        } else {
            output[output_idx] = T(0);  // Padding with zeros
        }
    }
}

// ============================================================================
// Fold HIP Kernel (col2im with atomic accumulation)
// ============================================================================

template<typename T>
__global__ void fold_kernel_hip(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t out_h,
    int64_t out_w
) {
    int64_t num_blocks = out_h * out_w;
    int64_t col_channels = channels * kernel_size * kernel_size;
    int64_t total_elements = batch * col_channels * num_blocks;

    HIP_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, col_c, block_idx)
        int64_t temp = idx;
        int64_t block_idx = temp % num_blocks; temp /= num_blocks;
        int64_t col_c = temp % col_channels; temp /= col_channels;
        int64_t b = temp;

        // Decode col_c to (c, kh, kw)
        int64_t kw = col_c % kernel_size;
        int64_t kh = (col_c / kernel_size) % kernel_size;
        int64_t c = col_c / (kernel_size * kernel_size);

        // Calculate output position from block_idx
        int64_t oh = block_idx / out_w;
        int64_t ow = block_idx % out_w;

        // Calculate output position in image
        int64_t ih = oh * stride - padding + kh * dilation;
        int64_t iw = ow * stride - padding + kw * dilation;

        // Check bounds
        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            int64_t input_idx = b * (col_channels * num_blocks) +
                               col_c * num_blocks +
                               block_idx;

            int64_t output_idx = b * (channels * height * width) +
                                c * (height * width) +
                                ih * width + iw;

            // Accumulate (sum overlapping values)
            atomicAdd(&output[output_idx], input[input_idx]);
        }
    }
}

// ============================================================================
// Interpolation HIP Kernels
// ============================================================================

// Nearest neighbor interpolation kernel
template<typename T>
__global__ void interpolate_nearest_kernel_hip(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_h,
    int64_t out_w
) {
    int64_t total_elements = batch * channels * out_h * out_w;

    HIP_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, c, oh, ow)
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t c = temp % channels; temp /= channels;
        int64_t b = temp;

        // Calculate source position
        float scale_h = static_cast<float>(in_h) / out_h;
        float scale_w = static_cast<float>(in_w) / out_w;

        int64_t ih = static_cast<int64_t>(oh * scale_h);
        int64_t iw = static_cast<int64_t>(ow * scale_w);

        // Clamp to valid range
        ih = min(max(ih, int64_t(0)), in_h - 1);
        iw = min(max(iw, int64_t(0)), in_w - 1);

        int64_t in_idx = b * (channels * in_h * in_w) +
                        c * (in_h * in_w) +
                        ih * in_w + iw;

        output[idx] = input[in_idx];
    }
}

// Bilinear interpolation kernel
template<typename T>
__global__ void interpolate_bilinear_kernel_hip(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_h,
    int64_t out_w,
    bool align_corners
) {
    int64_t total_elements = batch * channels * out_h * out_w;

    HIP_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, c, oh, ow)
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t c = temp % channels; temp /= channels;
        int64_t b = temp;

        // Calculate source position (floating point)
        float y, x;
        if (align_corners) {
            // Align corners: map [0, out-1] to [0, in-1]
            y = (out_h > 1) ? oh * static_cast<float>(in_h - 1) / (out_h - 1) : 0.0f;
            x = (out_w > 1) ? ow * static_cast<float>(in_w - 1) / (out_w - 1) : 0.0f;
        } else {
            // Half-pixel centers: pixels are unit squares
            float scale_h = static_cast<float>(in_h) / out_h;
            float scale_w = static_cast<float>(in_w) / out_w;
            y = (oh + 0.5f) * scale_h - 0.5f;
            x = (ow + 0.5f) * scale_w - 0.5f;
        }

        // Clamp to valid range
        y = fmaxf(0.0f, fminf(y, static_cast<float>(in_h - 1)));
        x = fmaxf(0.0f, fminf(x, static_cast<float>(in_w - 1)));

        // Get integer and fractional parts
        int64_t y0 = static_cast<int64_t>(y);
        int64_t x0 = static_cast<int64_t>(x);
        int64_t y1 = min(y0 + 1, in_h - 1);
        int64_t x1 = min(x0 + 1, in_w - 1);

        float fy = y - y0;
        float fx = x - x0;

        // Bilinear interpolation weights
        float w00 = (1.0f - fy) * (1.0f - fx);
        float w01 = (1.0f - fy) * fx;
        float w10 = fy * (1.0f - fx);
        float w11 = fy * fx;

        // Get pixel values
        int64_t base_idx = b * (channels * in_h * in_w) + c * (in_h * in_w);
        T v00 = input[base_idx + y0 * in_w + x0];
        T v01 = input[base_idx + y0 * in_w + x1];
        T v10 = input[base_idx + y1 * in_w + x0];
        T v11 = input[base_idx + y1 * in_w + x1];

        // Interpolate
        output[idx] = static_cast<T>(w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11);
    }
}

// Bicubic interpolation helper function
__device__ inline float cubic_interp1d(float x) {
    float abs_x = fabsf(x);
    if (abs_x <= 1.0f) {
        return 1.5f * abs_x * abs_x * abs_x - 2.5f * abs_x * abs_x + 1.0f;
    } else if (abs_x < 2.0f) {
        return -0.5f * abs_x * abs_x * abs_x + 2.5f * abs_x * abs_x - 4.0f * abs_x + 2.0f;
    }
    return 0.0f;
}

// Bicubic interpolation kernel
template<typename T>
__global__ void interpolate_bicubic_kernel_hip(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_h,
    int64_t out_w,
    bool align_corners
) {
    int64_t total_elements = batch * channels * out_h * out_w;

    HIP_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, c, oh, ow)
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t c = temp % channels; temp /= channels;
        int64_t b = temp;

        // Calculate source position (floating point)
        float y, x;
        if (align_corners) {
            y = (out_h > 1) ? oh * static_cast<float>(in_h - 1) / (out_h - 1) : 0.0f;
            x = (out_w > 1) ? ow * static_cast<float>(in_w - 1) / (out_w - 1) : 0.0f;
        } else {
            float scale_h = static_cast<float>(in_h) / out_h;
            float scale_w = static_cast<float>(in_w) / out_w;
            y = (oh + 0.5f) * scale_h - 0.5f;
            x = (ow + 0.5f) * scale_w - 0.5f;
        }

        // Clamp to valid range
        y = fmaxf(0.0f, fminf(y, static_cast<float>(in_h - 1)));
        x = fmaxf(0.0f, fminf(x, static_cast<float>(in_w - 1)));

        int64_t y_int = static_cast<int64_t>(y);
        int64_t x_int = static_cast<int64_t>(x);

        // Bicubic interpolation using 4x4 neighborhood
        float sum = 0.0f;
        for (int64_t dy = -1; dy <= 2; ++dy) {
            for (int64_t dx = -1; dx <= 2; ++dx) {
                int64_t iy = y_int + dy;
                int64_t ix = x_int + dx;

                // Clamp indices
                iy = max(int64_t(0), min(iy, in_h - 1));
                ix = max(int64_t(0), min(ix, in_w - 1));

                float weight_y = cubic_interp1d(y - (y_int + dy));
                float weight_x = cubic_interp1d(x - (x_int + dx));
                float weight = weight_y * weight_x;

                int64_t in_idx = b * (channels * in_h * in_w) +
                                c * (in_h * in_w) +
                                iy * in_w + ix;

                sum += weight * static_cast<float>(input[in_idx]);
            }
        }

        output[idx] = static_cast<T>(sum);
    }
}

// ============================================================================
// Host Functions
// ============================================================================

// Unfold host function
auto unfold_kernel(const Tensor& input,
                   int64_t kernel_size,
                   int64_t stride,
                   int64_t padding,
                   int64_t dilation,
                   hipStream_t stream) -> Tensor {
    auto shape = input.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t height = shape[2];
    int64_t width = shape[3];

    // Calculate output dimensions
    int64_t out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t num_blocks = out_h * out_w;

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, channels * kernel_size * kernel_size, num_blocks};
    Tensor output(output_shape, input.dtype(), input.device());

    // Launch kernel
    int64_t total_elements = batch * channels * kernel_size * kernel_size * num_blocks;
    int num_blocks_kernel = get_num_blocks(total_elements);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(unfold_kernel_hip<float>,
            dim3(num_blocks_kernel), dim3(BLOCK_SIZE), 0, stream,
            input.data<float>(),
            output.data<float>(),
            batch, channels, height, width,
            kernel_size, stride, padding, dilation,
            out_h, out_w
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(unfold_kernel_hip<double>,
            dim3(num_blocks_kernel), dim3(BLOCK_SIZE), 0, stream,
            input.data<double>(),
            output.data<double>(),
            batch, channels, height, width,
            kernel_size, stride, padding, dilation,
            out_h, out_w
        );
    } else {
        throw std::runtime_error("unfold_kernel: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// Fold host function
auto fold_kernel(const Tensor& input,
                 const std::vector<int64_t>& output_size,
                 int64_t kernel_size,
                 int64_t stride,
                 int64_t padding,
                 int64_t dilation,
                 hipStream_t stream) -> Tensor {
    auto shape = input.shape();
    int64_t batch = shape[0];
    int64_t col_channels = shape[1];
    int64_t num_blocks = shape[2];

    int64_t channels = col_channels / (kernel_size * kernel_size);
    int64_t height = output_size[0];
    int64_t width = output_size[1];

    // Calculate expected dimensions
    int64_t out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Create output tensor (initialized to zero)
    std::vector<int64_t> output_shape = {batch, channels, height, width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Initialize to zero
    HIP_CHECK(hipMemsetAsync(output.data<void>(), 0,
        output.numel() * dtype_size(input.dtype()), stream));

    // Launch kernel
    int64_t total_elements = batch * col_channels * num_blocks;
    int num_blocks_kernel = get_num_blocks(total_elements);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fold_kernel_hip<float>,
            dim3(num_blocks_kernel), dim3(BLOCK_SIZE), 0, stream,
            input.data<float>(),
            output.data<float>(),
            batch, channels, height, width,
            kernel_size, stride, padding, dilation,
            out_h, out_w
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(fold_kernel_hip<double>,
            dim3(num_blocks_kernel), dim3(BLOCK_SIZE), 0, stream,
            input.data<double>(),
            output.data<double>(),
            batch, channels, height, width,
            kernel_size, stride, padding, dilation,
            out_h, out_w
        );
    } else {
        throw std::runtime_error("fold_kernel: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// Interpolate host function
auto interpolate_kernel(const Tensor& input,
                        const std::vector<int64_t>& size,
                        const std::string& mode,
                        bool align_corners,
                        hipStream_t stream) -> Tensor {
    auto shape = input.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t in_h = shape[2];
    int64_t in_w = shape[3];
    int64_t out_h = size[0];
    int64_t out_w = size[1];

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, channels, out_h, out_w};
    Tensor output(output_shape, input.dtype(), input.device());

    // Launch kernel
    int64_t total_elements = batch * channels * out_h * out_w;
    int num_blocks = get_num_blocks(total_elements);

    if (mode == "nearest") {
        if (input.dtype() == DType::Float32) {
            hipLaunchKernelGGL(interpolate_nearest_kernel_hip<float>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                input.data<float>(),
                output.data<float>(),
                batch, channels, in_h, in_w, out_h, out_w
            );
        } else if (input.dtype() == DType::Float64) {
            hipLaunchKernelGGL(interpolate_nearest_kernel_hip<double>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                input.data<double>(),
                output.data<double>(),
                batch, channels, in_h, in_w, out_h, out_w
            );
        } else {
            throw std::runtime_error("interpolate_kernel: Unsupported dtype");
        }
    } else if (mode == "bilinear") {
        if (input.dtype() == DType::Float32) {
            hipLaunchKernelGGL(interpolate_bilinear_kernel_hip<float>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                input.data<float>(),
                output.data<float>(),
                batch, channels, in_h, in_w, out_h, out_w,
                align_corners
            );
        } else if (input.dtype() == DType::Float64) {
            hipLaunchKernelGGL(interpolate_bilinear_kernel_hip<double>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                input.data<double>(),
                output.data<double>(),
                batch, channels, in_h, in_w, out_h, out_w,
                align_corners
            );
        } else {
            throw std::runtime_error("interpolate_kernel: Unsupported dtype");
        }
    } else if (mode == "bicubic") {
        if (input.dtype() == DType::Float32) {
            hipLaunchKernelGGL(interpolate_bicubic_kernel_hip<float>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                input.data<float>(),
                output.data<float>(),
                batch, channels, in_h, in_w, out_h, out_w,
                align_corners
            );
        } else if (input.dtype() == DType::Float64) {
            hipLaunchKernelGGL(interpolate_bicubic_kernel_hip<double>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                input.data<double>(),
                output.data<double>(),
                batch, channels, in_h, in_w, out_h, out_w,
                align_corners
            );
        } else {
            throw std::runtime_error("interpolate_kernel: Unsupported dtype");
        }
    } else {
        throw std::runtime_error("interpolate_kernel: Unsupported mode: " + mode);
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

} // namespace rocm
} // namespace tenzor
