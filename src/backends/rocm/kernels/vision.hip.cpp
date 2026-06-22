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
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t out_h,
    int64_t out_w
) {
    int64_t num_blocks = out_h * out_w;
    int64_t total_elements = batch * channels * kernel_h * kernel_w * num_blocks;

    HIP_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, c, kh, kw, block_idx)
        int64_t temp = idx;
        int64_t block_idx = temp % num_blocks; temp /= num_blocks;
        int64_t kw = temp % kernel_w; temp /= kernel_w;
        int64_t kh = temp % kernel_h; temp /= kernel_h;
        int64_t c = temp % channels; temp /= channels;
        int64_t b = temp;

        // Calculate output position from block_idx
        int64_t oh = block_idx / out_w;
        int64_t ow = block_idx % out_w;

        // Calculate input position with padding and dilation
        int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
        int64_t iw = ow * stride_w - padding_w + kw * dilation_w;

        // Column index in output: (c * Kh * Kw + kh * Kw + kw)
        int64_t col_c = c * kernel_h * kernel_w + kh * kernel_w + kw;

        // Output index: (b, col_c, block_idx)
        int64_t output_idx = b * (channels * kernel_h * kernel_w * num_blocks) +
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
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t out_h,
    int64_t out_w
) {
    int64_t num_blocks = out_h * out_w;
    int64_t col_channels = channels * kernel_h * kernel_w;
    int64_t total_elements = batch * col_channels * num_blocks;

    HIP_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, col_c, block_idx)
        int64_t temp = idx;
        int64_t block_idx = temp % num_blocks; temp /= num_blocks;
        int64_t col_c = temp % col_channels; temp /= col_channels;
        int64_t b = temp;

        // Decode col_c to (c, kh, kw)
        int64_t kw = col_c % kernel_w;
        int64_t kh = (col_c / kernel_w) % kernel_h;
        int64_t c = col_c / (kernel_h * kernel_w);

        // Calculate output position from block_idx
        int64_t oh = block_idx / out_w;
        int64_t ow = block_idx % out_w;

        // Calculate output position in image
        int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
        int64_t iw = ow * stride_w - padding_w + kw * dilation_w;

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

// D3-followup ROCm: bilinear backward via atomicAdd scatter — mirror of
// `interpolate_bilinear_backward_kernel` in src/backends/cuda/kernels/vision.cu.
// HIP supports atomicAdd for float and double (gfx9+ via builtin); the kernel
// itself is line-for-line equivalent to the CUDA version.
template<typename T>
__global__ void interpolate_bilinear_backward_kernel_hip(
    const T* __restrict__ grad_out,
    T* __restrict__ grad_in,
    int64_t batch, int64_t channels,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    bool align_corners)
{
    int64_t total = batch * channels * out_h * out_w;
    HIP_KERNEL_LOOP(idx, total) {
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t c  = temp % channels; temp /= channels;
        int64_t b  = temp;

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
        y = fmaxf(0.0f, fminf(y, static_cast<float>(in_h - 1)));
        x = fmaxf(0.0f, fminf(x, static_cast<float>(in_w - 1)));

        int64_t y0 = static_cast<int64_t>(y);
        int64_t x0 = static_cast<int64_t>(x);
        int64_t y1 = min(y0 + 1, in_h - 1);
        int64_t x1 = min(x0 + 1, in_w - 1);
        float fy = y - y0;
        float fx = x - x0;

        float w00 = (1.0f - fy) * (1.0f - fx);
        float w01 = (1.0f - fy) * fx;
        float w10 = fy * (1.0f - fx);
        float w11 = fy * fx;

        float g = static_cast<float>(grad_out[idx]);
        int64_t base_idx = b * (channels * in_h * in_w) + c * (in_h * in_w);
        atomicAdd(&grad_in[base_idx + y0 * in_w + x0], static_cast<T>(w00 * g));
        atomicAdd(&grad_in[base_idx + y0 * in_w + x1], static_cast<T>(w01 * g));
        atomicAdd(&grad_in[base_idx + y1 * in_w + x0], static_cast<T>(w10 * g));
        atomicAdd(&grad_in[base_idx + y1 * in_w + x1], static_cast<T>(w11 * g));
    }
}

// Wave H4: nearest-neighbour interpolate backward — scatter each output
// gradient back to the single nearest input pixel via atomicAdd. Multiple
// output positions can map to the same input pixel; atomicAdd accumulates
// safely. Mirrors the bilinear backward shape with one destination per output
// element instead of four bilinear corners.
template<typename T>
__global__ void interpolate_nearest_backward_kernel_hip(
    const T* __restrict__ grad_out,
    T* __restrict__ grad_in,
    int64_t batch, int64_t channels,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w)
{
    int64_t total = batch * channels * out_h * out_w;
    HIP_KERNEL_LOOP(idx, total) {
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t c  = temp % channels; temp /= channels;
        int64_t b  = temp;

        // PyTorch nearest-mode forward maps `oh -> floor(oh * scale_h)` using a
        // float scale (no half-pixel offset, no align_corners — the nearest
        // mode in PyTorch ignores align_corners). The backward MUST mirror the
        // forward's float computation exactly: integer `(oh * in_h) / out_h`
        // disagrees with `(int64_t)(oh * (float)in_h / out_h)` at integer
        // boundaries due to float rounding, which would scatter gradient to a
        // different input pixel than the forward gathered from.
        float scale_h = static_cast<float>(in_h) / out_h;
        float scale_w = static_cast<float>(in_w) / out_w;
        int64_t y = static_cast<int64_t>(oh * scale_h);
        int64_t x = static_cast<int64_t>(ow * scale_w);
        y = min(max(y, int64_t(0)), in_h - 1);
        x = min(max(x, int64_t(0)), in_w - 1);

        int64_t base_idx = b * (channels * in_h * in_w) + c * (in_h * in_w);
        atomicAdd(&grad_in[base_idx + y * in_w + x], grad_out[idx]);
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

// Unfold host function (LL.3: per-axis kernel/stride/padding/dilation)
auto unfold_kernel(const Tensor& input,
                   int64_t kernel_h,
                   int64_t kernel_w,
                   int64_t stride_h,
                   int64_t stride_w,
                   int64_t padding_h,
                   int64_t padding_w,
                   int64_t dilation_h,
                   int64_t dilation_w,
                   hipStream_t stream) -> Tensor {
    // Float16 upcast: convert to Float32, compute, convert back
    if (input.dtype() == DType::Float16) {
        return unfold_kernel(input.to(DType::Float32),
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w, stream)
            .to(DType::Float16);
    }

    // BFloat16 upcast: convert to Float32, compute, convert back
    if (input.dtype() == DType::BFloat16) {
        return unfold_kernel(input.to(DType::Float32),
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w, stream)
            .to(DType::BFloat16);
    }

    // The kernels index with dense NCHW offsets, so a non-contiguous view
    // (transpose/slice/permute/channels-last) would read the wrong elements.
    Tensor in = input.is_contiguous() ? input : input.contiguous();

    auto shape = in.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t height = shape[2];
    int64_t width = shape[3];

    // Calculate output dimensions
    int64_t out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int64_t num_blocks = out_h * out_w;

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, channels * kernel_h * kernel_w, num_blocks};
    Tensor output(output_shape, in.dtype(), in.device());

    // Launch kernel
    int64_t total_elements = batch * channels * kernel_h * kernel_w * num_blocks;
    int num_blocks_kernel = get_num_blocks(total_elements);

    if (in.dtype() == DType::Float32) {
        hipLaunchKernelGGL(unfold_kernel_hip<float>,
            dim3(num_blocks_kernel), dim3(BLOCK_SIZE), 0, stream,
            in.data<float>(),
            output.data<float>(),
            batch, channels, height, width,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w,
            out_h, out_w
        );
    } else if (in.dtype() == DType::Float64) {
        hipLaunchKernelGGL(unfold_kernel_hip<double>,
            dim3(num_blocks_kernel), dim3(BLOCK_SIZE), 0, stream,
            in.data<double>(),
            output.data<double>(),
            batch, channels, height, width,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w,
            out_h, out_w
        );
    } else {
        throw std::runtime_error("unfold_kernel: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// Fold host function (LL.3: per-axis kernel/stride/padding/dilation)
auto fold_kernel(const Tensor& input,
                 const std::vector<int64_t>& output_size,
                 int64_t kernel_h,
                 int64_t kernel_w,
                 int64_t stride_h,
                 int64_t stride_w,
                 int64_t padding_h,
                 int64_t padding_w,
                 int64_t dilation_h,
                 int64_t dilation_w,
                 hipStream_t stream) -> Tensor {
    // Float16 upcast: convert to Float32, compute, convert back
    if (input.dtype() == DType::Float16) {
        return fold_kernel(input.to(DType::Float32), output_size,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w, stream)
            .to(DType::Float16);
    }

    // BFloat16 upcast: convert to Float32, compute, convert back
    if (input.dtype() == DType::BFloat16) {
        return fold_kernel(input.to(DType::Float32), output_size,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w, stream)
            .to(DType::BFloat16);
    }

    // Kernels index with dense offsets; materialize a contiguous input first.
    Tensor in = input.is_contiguous() ? input : input.contiguous();

    auto shape = in.shape();
    int64_t batch = shape[0];
    int64_t col_channels = shape[1];
    int64_t num_blocks = shape[2];

    int64_t channels = col_channels / (kernel_h * kernel_w);
    int64_t height = output_size[0];
    int64_t width = output_size[1];

    // Calculate expected dimensions
    int64_t out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Create output tensor (initialized to zero)
    std::vector<int64_t> output_shape = {batch, channels, height, width};
    Tensor output(output_shape, in.dtype(), in.device());

    // Initialize to zero
    HIP_CHECK(hipMemsetAsync(output.data_ptr(), 0,
        output.numel() * dtype_size(in.dtype()), stream));

    // Launch kernel
    int64_t total_elements = batch * col_channels * num_blocks;
    int num_blocks_kernel = get_num_blocks(total_elements);

    if (in.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fold_kernel_hip<float>,
            dim3(num_blocks_kernel), dim3(BLOCK_SIZE), 0, stream,
            in.data<float>(),
            output.data<float>(),
            batch, channels, height, width,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w,
            out_h, out_w
        );
    } else if (in.dtype() == DType::Float64) {
        hipLaunchKernelGGL(fold_kernel_hip<double>,
            dim3(num_blocks_kernel), dim3(BLOCK_SIZE), 0, stream,
            in.data<double>(),
            output.data<double>(),
            batch, channels, height, width,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w,
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
    // Float16 upcast: convert to Float32, compute, convert back
    if (input.dtype() == DType::Float16) {
        return interpolate_kernel(input.to(DType::Float32), size, mode, align_corners, stream)
            .to(DType::Float16);
    }

    // BFloat16 upcast: convert to Float32, compute, convert back
    if (input.dtype() == DType::BFloat16) {
        return interpolate_kernel(input.to(DType::Float32), size, mode, align_corners, stream)
            .to(DType::BFloat16);
    }

    // Kernels index with dense NCHW offsets; materialize contiguous input.
    Tensor in = input.is_contiguous() ? input : input.contiguous();

    auto shape = in.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t in_h = shape[2];
    int64_t in_w = shape[3];
    int64_t out_h = size[0];
    int64_t out_w = size[1];

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, channels, out_h, out_w};
    Tensor output(output_shape, in.dtype(), in.device());

    // Launch kernel
    int64_t total_elements = batch * channels * out_h * out_w;
    int num_blocks = get_num_blocks(total_elements);

    if (mode == "nearest") {
        if (in.dtype() == DType::Float32) {
            hipLaunchKernelGGL(interpolate_nearest_kernel_hip<float>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                in.data<float>(),
                output.data<float>(),
                batch, channels, in_h, in_w, out_h, out_w
            );
        } else if (in.dtype() == DType::Float64) {
            hipLaunchKernelGGL(interpolate_nearest_kernel_hip<double>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                in.data<double>(),
                output.data<double>(),
                batch, channels, in_h, in_w, out_h, out_w
            );
        } else {
            throw std::runtime_error("interpolate_kernel: Unsupported dtype");
        }
    } else if (mode == "bilinear") {
        if (in.dtype() == DType::Float32) {
            hipLaunchKernelGGL(interpolate_bilinear_kernel_hip<float>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                in.data<float>(),
                output.data<float>(),
                batch, channels, in_h, in_w, out_h, out_w,
                align_corners
            );
        } else if (in.dtype() == DType::Float64) {
            hipLaunchKernelGGL(interpolate_bilinear_kernel_hip<double>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                in.data<double>(),
                output.data<double>(),
                batch, channels, in_h, in_w, out_h, out_w,
                align_corners
            );
        } else {
            throw std::runtime_error("interpolate_kernel: Unsupported dtype");
        }
    } else if (mode == "bicubic") {
        if (in.dtype() == DType::Float32) {
            hipLaunchKernelGGL(interpolate_bicubic_kernel_hip<float>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                in.data<float>(),
                output.data<float>(),
                batch, channels, in_h, in_w, out_h, out_w,
                align_corners
            );
        } else if (in.dtype() == DType::Float64) {
            hipLaunchKernelGGL(interpolate_bicubic_kernel_hip<double>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                in.data<double>(),
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

// ============================================================================
// Interpolate backward (D3-followup ROCm) — bilinear-only, atomicAdd scatter.
// Mirrors the CUDA host function 1:1; nearest-mode backward routes through
// the bilinear path with integer fractional parts (the same shape simplification
// the CUDA host function makes).
// ============================================================================
// Catmull-Rom cubic convolution weight (a=-0.5); matches CPU cubic_interp_coeff.
__device__ __forceinline__ float tz_bicubic_coeff_hip(float x) {
    float a = fabsf(x);
    if (a <= 1.0f) return 1.5f * a * a * a - 2.5f * a * a + 1.0f;
    if (a < 2.0f)  return -0.5f * a * a * a + 2.5f * a * a - 4.0f * a + 2.0f;
    return 0.0f;
}

// Bicubic backward (4D): scatter each output gradient to its 4x4 neighborhood (clamped).
template<typename T>
__global__ void interpolate_bicubic_backward_kernel_hip(
    const T* __restrict__ grad_out, T* __restrict__ grad_in,
    int64_t batch, int64_t channels, int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w, bool align_corners)
{
    int64_t total = batch * channels * out_h * out_w;
    HIP_KERNEL_LOOP(idx, total) {
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t c  = temp % channels; temp /= channels;
        int64_t b  = temp;
        float scale_h = (align_corners && out_h > 1) ? static_cast<float>(in_h - 1) / (out_h - 1) : static_cast<float>(in_h) / out_h;
        float scale_w = (align_corners && out_w > 1) ? static_cast<float>(in_w - 1) / (out_w - 1) : static_cast<float>(in_w) / out_w;
        float src_h = align_corners ? oh * scale_h : (oh + 0.5f) * scale_h - 0.5f;
        float src_w = align_corners ? ow * scale_w : (ow + 0.5f) * scale_w - 0.5f;
        // Mirror the forward kernel EXACTLY: clamp the continuous source coord to
        // [0,in-1] BEFORE taking the integer base, so the 4x4 neighborhood and
        // fractional offsets match the forward (and CPU reference) at the borders.
        // Without this clamp, non-align_corners border pixels (src<0) floor to -1
        // and scatter to a different neighborhood -> parity divergence at edges.
        src_h = fmaxf(0.0f, fminf(src_h, static_cast<float>(in_h - 1)));
        src_w = fmaxf(0.0f, fminf(src_w, static_cast<float>(in_w - 1)));
        int64_t hi = static_cast<int64_t>(floorf(src_h));
        int64_t wi = static_cast<int64_t>(floorf(src_w));
        float g = static_cast<float>(grad_out[idx]);
        int64_t base = b * (channels * in_h * in_w) + c * (in_h * in_w);
        for (int dy = -1; dy <= 2; ++dy) {
            int64_t iy = min(max(static_cast<int64_t>(0), hi + dy), in_h - 1);
            float wy = tz_bicubic_coeff_hip(src_h - static_cast<float>(hi + dy));
            for (int dx = -1; dx <= 2; ++dx) {
                int64_t ix = min(max(static_cast<int64_t>(0), wi + dx), in_w - 1);
                float wx = tz_bicubic_coeff_hip(src_w - static_cast<float>(wi + dx));
                atomicAdd(&grad_in[base + iy * in_w + ix], static_cast<T>(wy * wx * g));
            }
        }
    }
}

// Trilinear backward (5D): scatter to 8 corners; out-of-bounds corners skipped (CPU semantics).
template<typename T>
__global__ void interpolate_trilinear_backward_kernel_hip(
    const T* __restrict__ grad_out, T* __restrict__ grad_in,
    int64_t batch, int64_t channels, int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_d, int64_t out_h, int64_t out_w, bool align_corners)
{
    int64_t total = batch * channels * out_d * out_h * out_w;
    HIP_KERNEL_LOOP(idx, total) {
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t od = temp % out_d; temp /= out_d;
        int64_t c  = temp % channels; temp /= channels;
        int64_t b  = temp;
        float scd = (align_corners && out_d > 1) ? static_cast<float>(in_d - 1) / (out_d - 1) : static_cast<float>(in_d) / out_d;
        float sch = (align_corners && out_h > 1) ? static_cast<float>(in_h - 1) / (out_h - 1) : static_cast<float>(in_h) / out_h;
        float scw = (align_corners && out_w > 1) ? static_cast<float>(in_w - 1) / (out_w - 1) : static_cast<float>(in_w) / out_w;
        float sd = align_corners ? od * scd : (od + 0.5f) * scd - 0.5f;
        float sh = align_corners ? oh * sch : (oh + 0.5f) * sch - 0.5f;
        float sw = align_corners ? ow * scw : (ow + 0.5f) * scw - 0.5f;
        // Clamp the source coord to [0, in-1] BEFORE flooring so the backward is
        // the exact transpose of the (clamping) forward at the borders; otherwise
        // a negative src floors to -1, its (1-f) tap is dropped, and the border
        // plane loses gradient mass (CPU parity — see cpu/kernels/vision.cpp).
        sd = fminf(fmaxf(sd, 0.0f), static_cast<float>(in_d - 1));
        sh = fminf(fmaxf(sh, 0.0f), static_cast<float>(in_h - 1));
        sw = fminf(fmaxf(sw, 0.0f), static_cast<float>(in_w - 1));
        int64_t d0 = static_cast<int64_t>(floorf(sd)), h0 = static_cast<int64_t>(floorf(sh)), w0 = static_cast<int64_t>(floorf(sw));
        int64_t d1 = (d0 + 1 < in_d) ? d0 + 1 : in_d - 1;
        int64_t h1 = (h0 + 1 < in_h) ? h0 + 1 : in_h - 1;
        int64_t w1 = (w0 + 1 < in_w) ? w0 + 1 : in_w - 1;
        float fd = sd - d0, fh = sh - h0, fw = sw - w0;
        float g = static_cast<float>(grad_out[idx]);
        int64_t base = b * (channels * in_d * in_h * in_w) + c * (in_d * in_h * in_w);
#define TZ_TRI_ADD_HIP(dd, hh, ww, wgt) do { \
        if ((dd) >= 0 && (dd) < in_d && (hh) >= 0 && (hh) < in_h && (ww) >= 0 && (ww) < in_w) \
            atomicAdd(&grad_in[base + (dd) * in_h * in_w + (hh) * in_w + (ww)], static_cast<T>((wgt) * g)); \
    } while (0)
        TZ_TRI_ADD_HIP(d0, h0, w0, (1.0f - fd) * (1.0f - fh) * (1.0f - fw));
        TZ_TRI_ADD_HIP(d0, h0, w1, (1.0f - fd) * (1.0f - fh) * fw);
        TZ_TRI_ADD_HIP(d0, h1, w0, (1.0f - fd) * fh * (1.0f - fw));
        TZ_TRI_ADD_HIP(d0, h1, w1, (1.0f - fd) * fh * fw);
        TZ_TRI_ADD_HIP(d1, h0, w0, fd * (1.0f - fh) * (1.0f - fw));
        TZ_TRI_ADD_HIP(d1, h0, w1, fd * (1.0f - fh) * fw);
        TZ_TRI_ADD_HIP(d1, h1, w0, fd * fh * (1.0f - fw));
        TZ_TRI_ADD_HIP(d1, h1, w1, fd * fh * fw);
#undef TZ_TRI_ADD_HIP
    }
}

auto interpolate_backward_kernel(const Tensor& grad_output,
                                  const std::vector<int64_t>& input_size,
                                  const std::string& mode,
                                  bool align_corners,
                                  hipStream_t stream) -> Tensor {
    // The scatter backward uses HIP atomicAdd, which has no Float16/BFloat16
    // overload. Compute the gradient in Float32 and narrow back so half-precision
    // inputs round-trip (matches the CPU/CUDA dtype-preserving behavior).
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        auto g32 = interpolate_backward_kernel(grad_output.to(DType::Float32),
                                               input_size, mode, align_corners, stream);
        return g32.to(grad_output.dtype());
    }
    auto shape = grad_output.shape();

    // Trilinear backward operates on 5D (N, C, D, H, W).
    if (mode == "trilinear") {
        if (shape.size() != 5)
            throw std::runtime_error("interpolate_backward (ROCm): trilinear requires 5D (N,C,D,H,W).");
        if (input_size.size() != 3)
            throw std::runtime_error("interpolate_backward (ROCm): trilinear input_size must be [in_d,in_h,in_w].");
        const int64_t N = shape[0], C = shape[1], out_d = shape[2], out_h = shape[3], out_w = shape[4];
        const int64_t in_d = input_size[0], in_h = input_size[1], in_w = input_size[2];
        Tensor grad_input({N, C, in_d, in_h, in_w}, grad_output.dtype(), grad_output.device());
        HIP_CHECK(hipMemsetAsync(grad_input.data_ptr(), 0,
                                  static_cast<size_t>(grad_input.numel()) * dtype_size(grad_input.dtype()), stream));
        const int64_t total = N * C * out_d * out_h * out_w;
        const int threads = 256;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        if (grad_output.dtype() == DType::Float32) {
            hipLaunchKernelGGL(interpolate_trilinear_backward_kernel_hip<float>, dim3(blocks), dim3(threads), 0, stream,
                grad_output.data<float>(), grad_input.data<float>(), N, C, in_d, in_h, in_w, out_d, out_h, out_w, align_corners);
        } else if (grad_output.dtype() == DType::Float64) {
            hipLaunchKernelGGL(interpolate_trilinear_backward_kernel_hip<double>, dim3(blocks), dim3(threads), 0, stream,
                grad_output.data<double>(), grad_input.data<double>(), N, C, in_d, in_h, in_w, out_d, out_h, out_w, align_corners);
        } else {
            throw std::runtime_error("interpolate_backward trilinear (ROCm): only Float32/Float64 supported.");
        }
        HIP_CHECK(hipGetLastError());
        return grad_input;
    }

    if (mode != "bilinear" && mode != "nearest" && mode != "bicubic") {
        throw std::runtime_error("interpolate_backward (ROCm): mode '" + mode +
                                  "' not supported. Use 'bilinear'/'nearest'/'bicubic' (4D) or 'trilinear' (5D).");
    }
    if (shape.size() != 4) {
        throw std::runtime_error("interpolate_backward (ROCm): bilinear/nearest/bicubic require 4D (N,C,H,W).");
    }
    if (input_size.size() != 2) {
        throw std::runtime_error("interpolate_backward (ROCm): input_size must be [in_h, in_w].");
    }
    const int64_t N     = shape[0];
    const int64_t C     = shape[1];
    const int64_t out_h = shape[2];
    const int64_t out_w = shape[3];
    const int64_t in_h  = input_size[0];
    const int64_t in_w  = input_size[1];

    Tensor grad_input({N, C, in_h, in_w}, grad_output.dtype(), grad_output.device());
    HIP_CHECK(hipMemsetAsync(grad_input.data_ptr(), 0,
                              static_cast<size_t>(grad_input.numel()) * dtype_size(grad_input.dtype()),
                              stream));

    const int64_t total = N * C * out_h * out_w;
    const int threads = 256;
    const int blocks  = static_cast<int>((total + threads - 1) / threads);

    // Wave H4: mode-aware dispatch — bilinear vs nearest both supported
    // natively now via dedicated scatter kernels.
    if (mode == "bilinear") {
        if (grad_output.dtype() == DType::Float32) {
            hipLaunchKernelGGL(interpolate_bilinear_backward_kernel_hip<float>,
                dim3(blocks), dim3(threads), 0, stream,
                grad_output.data<float>(), grad_input.data<float>(),
                N, C, in_h, in_w, out_h, out_w, align_corners);
        } else if (grad_output.dtype() == DType::Float64) {
            hipLaunchKernelGGL(interpolate_bilinear_backward_kernel_hip<double>,
                dim3(blocks), dim3(threads), 0, stream,
                grad_output.data<double>(), grad_input.data<double>(),
                N, C, in_h, in_w, out_h, out_w, align_corners);
        } else {
            throw std::runtime_error(
                "interpolate_backward bilinear (ROCm): unsupported dtype " +
                std::string(dtype_name(grad_output.dtype())) +
                ". Only Float32 and Float64 are supported (HIP atomicAdd availability).");
        }
    } else if (mode == "bicubic") {
        if (grad_output.dtype() == DType::Float32) {
            hipLaunchKernelGGL(interpolate_bicubic_backward_kernel_hip<float>,
                dim3(blocks), dim3(threads), 0, stream,
                grad_output.data<float>(), grad_input.data<float>(),
                N, C, in_h, in_w, out_h, out_w, align_corners);
        } else if (grad_output.dtype() == DType::Float64) {
            hipLaunchKernelGGL(interpolate_bicubic_backward_kernel_hip<double>,
                dim3(blocks), dim3(threads), 0, stream,
                grad_output.data<double>(), grad_input.data<double>(),
                N, C, in_h, in_w, out_h, out_w, align_corners);
        } else {
            throw std::runtime_error("interpolate_backward bicubic (ROCm): only Float32/Float64 supported.");
        }
    } else {  // "nearest"
        if (grad_output.dtype() == DType::Float32) {
            hipLaunchKernelGGL(interpolate_nearest_backward_kernel_hip<float>,
                dim3(blocks), dim3(threads), 0, stream,
                grad_output.data<float>(), grad_input.data<float>(),
                N, C, in_h, in_w, out_h, out_w);
        } else if (grad_output.dtype() == DType::Float64) {
            hipLaunchKernelGGL(interpolate_nearest_backward_kernel_hip<double>,
                dim3(blocks), dim3(threads), 0, stream,
                grad_output.data<double>(), grad_input.data<double>(),
                N, C, in_h, in_w, out_h, out_w);
        } else {
            throw std::runtime_error(
                "interpolate_backward nearest (ROCm): unsupported dtype " +
                std::string(dtype_name(grad_output.dtype())) +
                ". Only Float32 and Float64 are supported (HIP atomicAdd availability).");
        }
    }

    HIP_CHECK(hipGetLastError());
    return grad_input;
}

// ============================================================================
// BoxIoU kernel — compute IoU/GIoU between two sets of boxes
// ============================================================================

#define HIP_GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

template<typename T>
__global__ void box_iou_kernel(
    const T* boxes1, const T* boxes2, T* output,
    int64_t N, int64_t M, int iou_type)
{
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

    // GIoU(1) / DIoU(2) / CIoU(3) all need the smallest enclosing box.
    if (iou_type >= 1) {
        T enc_x1 = min(x1_1, x1_2);
        T enc_y1 = min(y1_1, y1_2);
        T enc_x2 = max(x2_1, x2_2);
        T enc_y2 = max(y2_1, y2_2);
        T enc_w = enc_x2 - enc_x1;
        T enc_h = enc_y2 - enc_y1;

        if (iou_type == 1) {
            // GIoU = IoU - (enclose_area - union_area) / enclose_area
            T enc_area = enc_w * enc_h;
            iou = iou - (enc_area - union_area) / (enc_area + static_cast<T>(1e-7));
        } else {
            // Center-distance penalty (shared by DIoU and CIoU).
            T cx1 = (x1_1 + x2_1) * static_cast<T>(0.5);
            T cy1 = (y1_1 + y2_1) * static_cast<T>(0.5);
            T cx2 = (x1_2 + x2_2) * static_cast<T>(0.5);
            T cy2 = (y1_2 + y2_2) * static_cast<T>(0.5);
            T center_dist_sq = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);
            T diag_dist_sq = enc_w * enc_w + enc_h * enc_h;
            T dist_penalty = center_dist_sq / (diag_dist_sq + static_cast<T>(1e-7));

            if (iou_type == 2) {
                // DIoU = IoU - center_dist^2 / diag_dist^2
                iou = iou - dist_penalty;
            } else {
                // CIoU = IoU - center_dist^2/diag^2 - alpha*v, with
                // v = (4/pi^2)(atan(w1/h1) - atan(w2/h2))^2 and
                // alpha = v / (1 - IoU + v). alpha/v use the ORIGINAL IoU.
                double w1 = static_cast<double>(x2_1 - x1_1);
                double h1 = static_cast<double>(y2_1 - y1_1);
                double w2 = static_cast<double>(x2_2 - x1_2);
                double h2 = static_cast<double>(y2_2 - y1_2);
                const double four_over_pi_sq =
                    4.0 / (3.14159265358979323846 * 3.14159265358979323846);
                double ar_diff = atan(w1 / (h1 + 1e-7)) - atan(w2 / (h2 + 1e-7));
                double v = ar_diff * ar_diff * four_over_pi_sq;
                double iou_d = static_cast<double>(iou);
                double alpha = v / (1.0 - iou_d + v + 1e-7);
                iou = static_cast<T>(iou_d - static_cast<double>(dist_penalty) - alpha * v);
            }
        }
    }

    output[i * M + j] = iou;
}

auto box_iou_hip(const Tensor& boxes1, const Tensor& boxes2, int iou_type,
                 hipStream_t stream) -> Tensor {
    // Float16 upcast: convert to Float32, compute, convert back
    if (boxes1.dtype() == DType::Float16) {
        return box_iou_hip(boxes1.to(DType::Float32), boxes2.to(DType::Float32), iou_type, stream)
            .to(DType::Float16);
    }

    // BFloat16 upcast: convert to Float32, compute, convert back
    if (boxes1.dtype() == DType::BFloat16) {
        return box_iou_hip(boxes1.to(DType::Float32), boxes2.to(DType::Float32), iou_type, stream)
            .to(DType::BFloat16);
    }

    // Flat-index kernel requires contiguous storage; materialize views.
    Tensor b1 = boxes1.is_contiguous() ? boxes1 : boxes1.contiguous();
    Tensor b2 = boxes2.is_contiguous() ? boxes2 : boxes2.contiguous();

    int64_t N = b1.shape()[0];
    int64_t M = b2.shape()[0];

    Tensor output({N, M}, b1.dtype(), b1.device());

    int64_t total = N * M;
    if (total == 0) return output;

    int num_blocks = get_num_blocks(total);

    if (b1.dtype() == DType::Float32) {
        hipLaunchKernelGGL(box_iou_kernel<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            b1.data<float>(), b2.data<float>(), output.data<float>(),
            N, M, iou_type);
    } else if (b1.dtype() == DType::Float64) {
        hipLaunchKernelGGL(box_iou_kernel<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            b1.data<double>(), b2.data<double>(), output.data<double>(),
            N, M, iou_type);
    } else {
        throw std::runtime_error("box_iou_hip: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

} // namespace rocm
} // namespace tenzor
