#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/dtype_dispatch.hpp"
#include "cuda_common.cuh"
#include "cuda_launch_utils.cuh"
#include "launch_config.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace cuda {

// ============================================================================
// Kernel Launch Helpers
// ============================================================================

// Delegates to compute_grid_size() from cuda_launch_utils.cuh to avoid
// duplicating the block-size logic. For per-kernel occupancy-optimized
// launches use optimal_launch_config() or the LAUNCH_KERNEL macro instead.
inline void compute_launch_config_1d(int64_t n, dim3& grid, dim3& block) {
    constexpr int block_size = 256;
    block = dim3(block_size, 1, 1);
    int num_blocks = compute_grid_size(n, block_size);
    grid = dim3(static_cast<unsigned int>(num_blocks), 1, 1);
}

#define CUDA_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// ============================================================================
// Unfold CUDA Kernel
// ============================================================================

template<typename T>
__global__ void unfold_kernel(
    const T* input,
    T* output,
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

    CUDA_KERNEL_LOOP(idx, total_elements) {
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
// Fold CUDA Kernel (col2im with atomic accumulation)
// ============================================================================

template<typename T>
__global__ void fold_kernel(
    const T* input,
    T* output,
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

    CUDA_KERNEL_LOOP(idx, total_elements) {
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
// Interpolation CUDA Kernels
// ============================================================================

// Nearest neighbor interpolation kernel
template<typename T>
__global__ void interpolate_nearest_kernel(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_h,
    int64_t out_w
) {
    int64_t total_elements = batch * channels * out_h * out_w;

    CUDA_KERNEL_LOOP(idx, total_elements) {
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
__global__ void interpolate_bilinear_kernel(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_h,
    int64_t out_w,
    bool align_corners
) {
    int64_t total_elements = batch * channels * out_h * out_w;

    CUDA_KERNEL_LOOP(idx, total_elements) {
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

        // Get pixel values and convert to float for interpolation
        int64_t base_idx = b * (channels * in_h * in_w) + c * (in_h * in_w);
        float v00 = static_cast<float>(input[base_idx + y0 * in_w + x0]);
        float v01 = static_cast<float>(input[base_idx + y0 * in_w + x1]);
        float v10 = static_cast<float>(input[base_idx + y1 * in_w + x0]);
        float v11 = static_cast<float>(input[base_idx + y1 * in_w + x1]);

        // Interpolate in float, then convert back
        float result = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
        output[idx] = static_cast<T>(result);
    }
}

// Bicubic interpolation kernel
template<typename T>
__global__ void interpolate_bicubic_kernel(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_h,
    int64_t out_w,
    bool align_corners
) {
    int64_t total_elements = batch * channels * out_h * out_w;

    // Cubic interpolation coefficient function
    auto cubic_interp1d = [](float x) -> float {
        float abs_x = fabsf(x);
        if (abs_x <= 1.0f) {
            return 1.5f * abs_x * abs_x * abs_x - 2.5f * abs_x * abs_x + 1.0f;
        } else if (abs_x < 2.0f) {
            return -0.5f * abs_x * abs_x * abs_x + 2.5f * abs_x * abs_x - 4.0f * abs_x + 2.0f;
        }
        return 0.0f;
    };

    CUDA_KERNEL_LOOP(idx, total_elements) {
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
// 5D Interpolation Kernels (trilinear, nearest-5d)
// ============================================================================

template<typename T>
__global__ void interpolate_trilinear_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t in_d,
    int64_t in_h,
    int64_t in_w,
    int64_t out_d,
    int64_t out_h,
    int64_t out_w,
    bool align_corners
) {
    int64_t total = batch * channels * out_d * out_h * out_w;
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t od = temp % out_d; temp /= out_d;
        int64_t c  = temp % channels; temp /= channels;
        int64_t b  = temp;

        float z, y, x;
        if (align_corners) {
            z = (out_d > 1) ? static_cast<float>(od) * (in_d - 1) / (out_d - 1) : 0.0f;
            y = (out_h > 1) ? static_cast<float>(oh) * (in_h - 1) / (out_h - 1) : 0.0f;
            x = (out_w > 1) ? static_cast<float>(ow) * (in_w - 1) / (out_w - 1) : 0.0f;
        } else {
            float scale_d = static_cast<float>(in_d) / out_d;
            float scale_h = static_cast<float>(in_h) / out_h;
            float scale_w = static_cast<float>(in_w) / out_w;
            z = (od + 0.5f) * scale_d - 0.5f;
            y = (oh + 0.5f) * scale_h - 0.5f;
            x = (ow + 0.5f) * scale_w - 0.5f;
        }

        z = fmaxf(0.0f, fminf(z, static_cast<float>(in_d - 1)));
        y = fmaxf(0.0f, fminf(y, static_cast<float>(in_h - 1)));
        x = fmaxf(0.0f, fminf(x, static_cast<float>(in_w - 1)));

        int64_t z0 = static_cast<int64_t>(z);
        int64_t y0 = static_cast<int64_t>(y);
        int64_t x0 = static_cast<int64_t>(x);
        int64_t z1 = min(z0 + 1, in_d - 1);
        int64_t y1 = min(y0 + 1, in_h - 1);
        int64_t x1 = min(x0 + 1, in_w - 1);

        float fz = z - z0;
        float fy = y - y0;
        float fx = x - x0;

        int64_t base = (b * channels + c) * in_d * in_h * in_w;

        float v000 = static_cast<float>(input[base + z0 * in_h * in_w + y0 * in_w + x0]);
        float v001 = static_cast<float>(input[base + z0 * in_h * in_w + y0 * in_w + x1]);
        float v010 = static_cast<float>(input[base + z0 * in_h * in_w + y1 * in_w + x0]);
        float v011 = static_cast<float>(input[base + z0 * in_h * in_w + y1 * in_w + x1]);
        float v100 = static_cast<float>(input[base + z1 * in_h * in_w + y0 * in_w + x0]);
        float v101 = static_cast<float>(input[base + z1 * in_h * in_w + y0 * in_w + x1]);
        float v110 = static_cast<float>(input[base + z1 * in_h * in_w + y1 * in_w + x0]);
        float v111 = static_cast<float>(input[base + z1 * in_h * in_w + y1 * in_w + x1]);

        float result =
            v000 * (1 - fz) * (1 - fy) * (1 - fx) +
            v001 * (1 - fz) * (1 - fy) * fx +
            v010 * (1 - fz) * fy * (1 - fx) +
            v011 * (1 - fz) * fy * fx +
            v100 * fz * (1 - fy) * (1 - fx) +
            v101 * fz * (1 - fy) * fx +
            v110 * fz * fy * (1 - fx) +
            v111 * fz * fy * fx;

        output[idx] = static_cast<T>(result);
    }
}

template<typename T>
__global__ void interpolate_nearest_5d_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t in_d,
    int64_t in_h,
    int64_t in_w,
    int64_t out_d,
    int64_t out_h,
    int64_t out_w
) {
    float scale_d = static_cast<float>(in_d) / out_d;
    float scale_h = static_cast<float>(in_h) / out_h;
    float scale_w = static_cast<float>(in_w) / out_w;
    int64_t total = batch * channels * out_d * out_h * out_w;

    CUDA_KERNEL_LOOP(idx, total) {
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t od = temp % out_d; temp /= out_d;
        int64_t c  = temp % channels; temp /= channels;
        int64_t b  = temp;

        int64_t id = min(static_cast<int64_t>(od * scale_d), in_d - 1);
        int64_t ih = min(static_cast<int64_t>(oh * scale_h), in_h - 1);
        int64_t iw = min(static_cast<int64_t>(ow * scale_w), in_w - 1);

        int64_t in_idx = ((b * channels + c) * in_d + id) * in_h * in_w + ih * in_w + iw;
        output[idx] = input[in_idx];
    }
}

// ============================================================================
// Host Functions
// ============================================================================

// Unfold host function
auto unfold_cuda(const Tensor& input,
                 int64_t kernel_size,
                 int64_t stride,
                 int64_t padding,
                 int64_t dilation) -> Tensor {
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
    dim3 grid, block;
    compute_launch_config_1d(total_elements, grid, block);

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "unfold_cuda", [&]() {
        unfold_kernel<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch, channels, height, width,
            kernel_size, stride, padding, dilation,
            out_h, out_w
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    });

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return output;
}

// Fold host function
auto fold_cuda(const Tensor& input,
               const std::vector<int64_t>& output_size,
               int64_t kernel_size,
               int64_t stride,
               int64_t padding,
               int64_t dilation) -> Tensor {
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
    TENZOR_CUDA_CHECK(cudaMemset(output.data_ptr(), 0, output.numel() * output.dtype_size()));

    // Launch kernel
    int64_t total_elements = batch * col_channels * num_blocks;
    dim3 grid, block;
    compute_launch_config_1d(total_elements, grid, block);

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "fold_cuda", [&]() {
        fold_kernel<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch, channels, height, width,
            kernel_size, stride, padding, dilation,
            out_h, out_w
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    });

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return output;
}

// Interpolate host function
auto interpolate_cuda(const Tensor& input,
                      const std::vector<int64_t>& size,
                      const std::string& mode,
                      bool align_corners) -> Tensor {
    auto shape = input.shape();

    // Handle 5D input (trilinear / nearest-5d)
    if (shape.size() == 5) {
        int64_t batch = shape[0], channels = shape[1];
        int64_t in_d = shape[2], in_h = shape[3], in_w = shape[4];
        int64_t out_d = size[0], out_h = size[1], out_w = size[2];

        Tensor output({batch, channels, out_d, out_h, out_w}, input.dtype(), input.device());
        int64_t total = batch * channels * out_d * out_h * out_w;
        dim3 grid, block;
        compute_launch_config_1d(total, grid, block);

        if (mode == "trilinear") {
            TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "interpolate_trilinear", [&]() {
                interpolate_trilinear_kernel<scalar_t><<<grid, block>>>(
                    input.data<scalar_t>(), output.data<scalar_t>(),
                    batch, channels, in_d, in_h, in_w, out_d, out_h, out_w, align_corners);
                TENZOR_CUDA_POST_LAUNCH_CHECK();
            });
        } else {
            TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "interpolate_nearest_5d", [&]() {
                interpolate_nearest_5d_kernel<scalar_t><<<grid, block>>>(
                    input.data<scalar_t>(), output.data<scalar_t>(),
                    batch, channels, in_d, in_h, in_w, out_d, out_h, out_w);
                TENZOR_CUDA_POST_LAUNCH_CHECK();
            });
        }
        return output;
    }

    // 4D path
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
    dim3 grid, block;
    compute_launch_config_1d(total_elements, grid, block);

    if (mode == "nearest") {
        if (input.dtype() == DType::Float32) {
            interpolate_nearest_kernel<float><<<grid, block>>>(
                input.data<float>(),
                output.data<float>(),
                batch, channels, in_h, in_w, out_h, out_w
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else if (input.dtype() == DType::Float64) {
            interpolate_nearest_kernel<double><<<grid, block>>>(
                input.data<double>(),
                output.data<double>(),
                batch, channels, in_h, in_w, out_h, out_w
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else if (input.dtype() == DType::Float16) {
            interpolate_nearest_kernel<__half><<<grid, block>>>(
                reinterpret_cast<const __half*>(input.data_ptr()),
                reinterpret_cast<__half*>(output.data_ptr()),
                batch, channels, in_h, in_w, out_h, out_w
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else {
            throw std::runtime_error("interpolate_cuda: Unsupported dtype");
        }
    } else if (mode == "bilinear") {
        if (input.dtype() == DType::Float32) {
            interpolate_bilinear_kernel<float><<<grid, block>>>(
                input.data<float>(),
                output.data<float>(),
                batch, channels, in_h, in_w, out_h, out_w,
                align_corners
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else if (input.dtype() == DType::Float64) {
            interpolate_bilinear_kernel<double><<<grid, block>>>(
                input.data<double>(),
                output.data<double>(),
                batch, channels, in_h, in_w, out_h, out_w,
                align_corners
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else if (input.dtype() == DType::Float16) {
            interpolate_bilinear_kernel<__half><<<grid, block>>>(
                reinterpret_cast<const __half*>(input.data_ptr()),
                reinterpret_cast<__half*>(output.data_ptr()),
                batch, channels, in_h, in_w, out_h, out_w,
                align_corners
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else {
            throw std::runtime_error("interpolate_cuda: Unsupported dtype");
        }
    } else if (mode == "bicubic") {
        if (input.dtype() == DType::Float32) {
            interpolate_bicubic_kernel<float><<<grid, block>>>(
                input.data<float>(),
                output.data<float>(),
                batch, channels, in_h, in_w, out_h, out_w,
                align_corners
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else if (input.dtype() == DType::Float64) {
            interpolate_bicubic_kernel<double><<<grid, block>>>(
                input.data<double>(),
                output.data<double>(),
                batch, channels, in_h, in_w, out_h, out_w,
                align_corners
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else if (input.dtype() == DType::Float16) {
            interpolate_bicubic_kernel<__half><<<grid, block>>>(
                reinterpret_cast<const __half*>(input.data_ptr()),
                reinterpret_cast<__half*>(output.data_ptr()),
                batch, channels, in_h, in_w, out_h, out_w,
                align_corners
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else {
            throw std::runtime_error("interpolate_cuda: Unsupported dtype");
        }
    } else {
        throw std::runtime_error("interpolate_cuda: Unsupported mode: " + mode);
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return output;
}

// ============================================================================
// BoxIoU CUDA Kernel — pairwise IoU matrix
// ============================================================================

template<typename T>
__global__ void box_iou_kernel(
    const T* __restrict__ boxes1,  // [N, 4]
    const T* __restrict__ boxes2,  // [M, 4]
    T* __restrict__ output,        // [N, M]
    int64_t N,
    int64_t M,
    int iou_type  // 0 = IoU, 1 = GIoU
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * M;
    if (idx >= total) return;

    int64_t i = idx / M;
    int64_t j = idx % M;

    // Box format: [x1, y1, x2, y2]
    T x1_1 = boxes1[i * 4 + 0];
    T y1_1 = boxes1[i * 4 + 1];
    T x2_1 = boxes1[i * 4 + 2];
    T y2_1 = boxes1[i * 4 + 3];

    T x1_2 = boxes2[j * 4 + 0];
    T y1_2 = boxes2[j * 4 + 1];
    T x2_2 = boxes2[j * 4 + 2];
    T y2_2 = boxes2[j * 4 + 3];

    // Intersection
    T inter_x1 = max(x1_1, x1_2);
    T inter_y1 = max(y1_1, y1_2);
    T inter_x2 = min(x2_1, x2_2);
    T inter_y2 = min(y2_1, y2_2);

    T inter_w = max(static_cast<T>(0), inter_x2 - inter_x1);
    T inter_h = max(static_cast<T>(0), inter_y2 - inter_y1);
    T inter_area = inter_w * inter_h;

    // Areas
    T area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    T area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    T union_area = area1 + area2 - inter_area;

    T iou = inter_area / (union_area + static_cast<T>(1e-7));

    if (iou_type == 1) {
        // GIoU: IoU - (enclosing_area - union_area) / enclosing_area
        T enc_x1 = min(x1_1, x1_2);
        T enc_y1 = min(y1_1, y1_2);
        T enc_x2 = max(x2_1, x2_2);
        T enc_y2 = max(y2_1, y2_2);
        T enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1);
        iou = iou - (enc_area - union_area) / (enc_area + static_cast<T>(1e-7));
    }

    output[i * M + j] = iou;
}

auto box_iou_cuda(const Tensor& boxes1, const Tensor& boxes2, int iou_type) -> Tensor {
    int64_t N = boxes1.shape()[0];
    int64_t M = boxes2.shape()[0];

    Tensor output({N, M}, boxes1.dtype(), boxes1.device());

    int64_t total = N * M;
    if (total == 0) return output;

    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    if (boxes1.dtype() == DType::Float32) {
        box_iou_kernel<float><<<grid, block>>>(
            boxes1.data<float>(), boxes2.data<float>(), output.data<float>(),
            N, M, iou_type);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (boxes1.dtype() == DType::Float64) {
        box_iou_kernel<double><<<grid, block>>>(
            boxes1.data<double>(), boxes2.data<double>(), output.data<double>(),
            N, M, iou_type);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("box_iou_cuda: unsupported dtype");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return output;
}

} // namespace cuda
} // namespace tenzor
