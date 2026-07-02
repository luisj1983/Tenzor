#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/dtype_dispatch.hpp"
#include "cuda_common.cuh"
#include "cuda_launch_utils.cuh"
#include "cuda_nan_helpers.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tenzor {
namespace cuda {

// Conversion kernels for half-type upcast/downcast
template<typename HalfT>
__global__ void half_to_float_kernel(const HalfT* __restrict__ in, float* __restrict__ out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = static_cast<float>(in[idx]);
}

template<typename HalfT>
__global__ void float_to_half_kernel(const float* __restrict__ in, HalfT* __restrict__ out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = static_cast<HalfT>(in[idx]);
}

// ============================================================================
// Kernel Launch Helpers
// ============================================================================

// compute_launch_config_1d() is now in cuda_launch_utils.cuh

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

    TENZOR_CUDA_KERNEL_LOOP(idx, total_elements) {
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
            // Zero-initialize in a way that works for fp16/bf16 too (which
            // may lack an int-taking constructor across all CUDA versions).
            if constexpr (std::is_same_v<T, __half>) {
                output[output_idx] = __float2half(0.0f);
            } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                output[output_idx] = __float2bfloat16(0.0f);
            } else {
                output[output_idx] = T(0);  // Padding with zeros
            }
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

    TENZOR_CUDA_KERNEL_LOOP(idx, total_elements) {
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
// Fold FP16 Kernel — arch-guarded atomicAdd for __half
// ============================================================================

__global__ void fold_kernel_fp16(
    const __half* input,
    __half* output,
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

    TENZOR_CUDA_KERNEL_LOOP(idx, total_elements) {
        int64_t temp = idx;
        int64_t block_idx = temp % num_blocks; temp /= num_blocks;
        int64_t col_c = temp % col_channels; temp /= col_channels;
        int64_t b = temp;

        int64_t kw = col_c % kernel_w;
        int64_t kh = (col_c / kernel_w) % kernel_h;
        int64_t c = col_c / (kernel_h * kernel_w);

        int64_t oh = block_idx / out_w;
        int64_t ow = block_idx % out_w;

        int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
        int64_t iw = ow * stride_w - padding_w + kw * dilation_w;

        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            int64_t input_idx = b * (col_channels * num_blocks) +
                               col_c * num_blocks +
                               block_idx;

            int64_t output_idx = b * (channels * height * width) +
                                c * (height * width) +
                                ih * width + iw;

#if __CUDA_ARCH__ >= 700
            atomicAdd(&output[output_idx], input[input_idx]);
#else
            // CAS-based fallback for SM < 70
            float val = __half2float(input[input_idx]);
            unsigned int* addr = reinterpret_cast<unsigned int*>(
                &output[output_idx & ~int64_t(1)]);
            unsigned int old_val, new_val;
            int lane = output_idx & 1;
            do {
                old_val = atomicCAS(addr, 0u, 0u);
                __half* h = reinterpret_cast<__half*>(&old_val);
                // W.7: NaN-preserving F32→F16 conversion.
                __half result = ::tenzor::cuda::safe_f2half(::tenzor::cuda::safe_half2f(h[lane]) + val);
                new_val = old_val;
                reinterpret_cast<__half*>(&new_val)[lane] = result;
            } while (atomicCAS(addr, old_val, new_val) != old_val);
#endif
        }
    }
}

// ============================================================================
// Fold BF16 Kernel — arch-guarded atomicAdd for __nv_bfloat16
// ============================================================================

__global__ void fold_kernel_bf16(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
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

    TENZOR_CUDA_KERNEL_LOOP(idx, total_elements) {
        int64_t temp = idx;
        int64_t block_idx = temp % num_blocks; temp /= num_blocks;
        int64_t col_c = temp % col_channels; temp /= col_channels;
        int64_t b = temp;

        int64_t kw = col_c % kernel_w;
        int64_t kh = (col_c / kernel_w) % kernel_h;
        int64_t c = col_c / (kernel_h * kernel_w);

        int64_t oh = block_idx / out_w;
        int64_t ow = block_idx % out_w;

        int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
        int64_t iw = ow * stride_w - padding_w + kw * dilation_w;

        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            int64_t input_idx = b * (col_channels * num_blocks) +
                               col_c * num_blocks +
                               block_idx;

            int64_t output_idx = b * (channels * height * width) +
                                c * (height * width) +
                                ih * width + iw;

#if __CUDA_ARCH__ >= 800
            atomicAdd(&output[output_idx], input[input_idx]);
#else
            // CAS-based fallback for SM < 80
            float val = __bfloat162float(input[input_idx]);
            unsigned int* addr = reinterpret_cast<unsigned int*>(
                &output[output_idx & ~int64_t(1)]);
            unsigned int old_val, new_val;
            int lane = output_idx & 1;
            do {
                old_val = atomicCAS(addr, 0u, 0u);
                __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&old_val);
                // W.7: NaN-preserving F32→BF16 conversion.
                __nv_bfloat16 result = ::tenzor::cuda::safe_f2bf16(::tenzor::cuda::safe_bf162f(h[lane]) + val);
                new_val = old_val;
                reinterpret_cast<__nv_bfloat16*>(&new_val)[lane] = result;
            } while (atomicCAS(addr, old_val, new_val) != old_val);
#endif
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

    TENZOR_CUDA_KERNEL_LOOP(idx, total_elements) {
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
    // Compute in double for Float64 inputs to preserve FP64 precision/parity
    // with the CPU reference; float otherwise.
    using Compute = typename std::conditional<std::is_same<T, double>::value, double, float>::type;
    int64_t total_elements = batch * channels * out_h * out_w;

    TENZOR_CUDA_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, c, oh, ow)
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t c = temp % channels; temp /= channels;
        int64_t b = temp;

        // Calculate source position (floating point)
        Compute y, x;
        if (align_corners) {
            // Align corners: map [0, out-1] to [0, in-1]
            y = (out_h > 1) ? oh * static_cast<Compute>(in_h - 1) / (out_h - 1) : Compute(0);
            x = (out_w > 1) ? ow * static_cast<Compute>(in_w - 1) / (out_w - 1) : Compute(0);
        } else {
            // Half-pixel centers: pixels are unit squares
            Compute scale_h = static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
            Compute scale_w = static_cast<Compute>(in_w) / static_cast<Compute>(out_w);
            y = (oh + Compute(0.5)) * scale_h - Compute(0.5);
            x = (ow + Compute(0.5)) * scale_w - Compute(0.5);
        }

        // Clamp to valid range
        const Compute yhi = static_cast<Compute>(in_h - 1);
        const Compute xhi = static_cast<Compute>(in_w - 1);
        y = y < Compute(0) ? Compute(0) : (y > yhi ? yhi : y);
        x = x < Compute(0) ? Compute(0) : (x > xhi ? xhi : x);

        // Get integer and fractional parts
        int64_t y0 = static_cast<int64_t>(y);
        int64_t x0 = static_cast<int64_t>(x);
        int64_t y1 = min(y0 + 1, in_h - 1);
        int64_t x1 = min(x0 + 1, in_w - 1);

        Compute fy = y - y0;
        Compute fx = x - x0;

        // Bilinear interpolation weights
        Compute w00 = (Compute(1) - fy) * (Compute(1) - fx);
        Compute w01 = (Compute(1) - fy) * fx;
        Compute w10 = fy * (Compute(1) - fx);
        Compute w11 = fy * fx;

        // Get pixel values and convert to compute type for interpolation
        int64_t base_idx = b * (channels * in_h * in_w) + c * (in_h * in_w);
        Compute v00 = static_cast<Compute>(input[base_idx + y0 * in_w + x0]);
        Compute v01 = static_cast<Compute>(input[base_idx + y0 * in_w + x1]);
        Compute v10 = static_cast<Compute>(input[base_idx + y1 * in_w + x0]);
        Compute v11 = static_cast<Compute>(input[base_idx + y1 * in_w + x1]);

        // Interpolate, then convert back
        Compute result = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
        output[idx] = static_cast<T>(result);
    }
}

// Cubic convolution kernel weight (a = -0.75), matches CPU cubic_interp_coeff
// and PyTorch upsample_bicubic2d. Templated on the compute type so Float64
// preserves FP64 precision. Shared by the bicubic forward and backward kernels
// so the backward stays the exact transpose of the forward.
template <typename Compute>
__device__ __forceinline__ Compute tz_bicubic_coeff(Compute x) {
    Compute a = x < Compute(0) ? -x : x;
    if (a <= Compute(1)) return Compute(1.25) * a * a * a - Compute(2.25) * a * a + Compute(1);
    if (a < Compute(2))  return Compute(-0.75) * a * a * a + Compute(3.75) * a * a - Compute(6) * a + Compute(3);
    return Compute(0);
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
    // Compute in double for Float64 inputs to preserve FP64 precision/parity
    // with the CPU reference; float otherwise.
    using Compute = typename std::conditional<std::is_same<T, double>::value, double, float>::type;
    int64_t total_elements = batch * channels * out_h * out_w;

    TENZOR_CUDA_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, c, oh, ow)
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t c = temp % channels; temp /= channels;
        int64_t b = temp;

        // Calculate source position (floating point)
        Compute y, x;
        if (align_corners) {
            y = (out_h > 1) ? oh * static_cast<Compute>(in_h - 1) / (out_h - 1) : Compute(0);
            x = (out_w > 1) ? ow * static_cast<Compute>(in_w - 1) / (out_w - 1) : Compute(0);
        } else {
            Compute scale_h = static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
            Compute scale_w = static_cast<Compute>(in_w) / static_cast<Compute>(out_w);
            y = (oh + Compute(0.5)) * scale_h - Compute(0.5);
            x = (ow + Compute(0.5)) * scale_w - Compute(0.5);
        }

        // Clamp to valid range
        const Compute yhi = static_cast<Compute>(in_h - 1);
        const Compute xhi = static_cast<Compute>(in_w - 1);
        y = y < Compute(0) ? Compute(0) : (y > yhi ? yhi : y);
        x = x < Compute(0) ? Compute(0) : (x > xhi ? xhi : x);

        int64_t y_int = static_cast<int64_t>(y);
        int64_t x_int = static_cast<int64_t>(x);

        // Bicubic interpolation using 4x4 neighborhood
        Compute sum = Compute(0);
        for (int64_t dy = -1; dy <= 2; ++dy) {
            for (int64_t dx = -1; dx <= 2; ++dx) {
                int64_t iy = y_int + dy;
                int64_t ix = x_int + dx;

                // Clamp indices
                iy = max(int64_t(0), min(iy, in_h - 1));
                ix = max(int64_t(0), min(ix, in_w - 1));

                Compute weight_y = tz_bicubic_coeff<Compute>(y - (y_int + dy));
                Compute weight_x = tz_bicubic_coeff<Compute>(x - (x_int + dx));
                Compute weight = weight_y * weight_x;

                int64_t in_idx = b * (channels * in_h * in_w) +
                                c * (in_h * in_w) +
                                iy * in_w + ix;

                sum += weight * static_cast<Compute>(input[in_idx]);
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
    // Compute in double for Float64 inputs to preserve FP64 precision/parity
    // with the CPU reference; float otherwise.
    using Compute = typename std::conditional<std::is_same<T, double>::value, double, float>::type;
    int64_t total = batch * channels * out_d * out_h * out_w;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t od = temp % out_d; temp /= out_d;
        int64_t c  = temp % channels; temp /= channels;
        int64_t b  = temp;

        Compute z, y, x;
        if (align_corners) {
            z = (out_d > 1) ? static_cast<Compute>(od) * (in_d - 1) / (out_d - 1) : Compute(0);
            y = (out_h > 1) ? static_cast<Compute>(oh) * (in_h - 1) / (out_h - 1) : Compute(0);
            x = (out_w > 1) ? static_cast<Compute>(ow) * (in_w - 1) / (out_w - 1) : Compute(0);
        } else {
            Compute scale_d = static_cast<Compute>(in_d) / static_cast<Compute>(out_d);
            Compute scale_h = static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
            Compute scale_w = static_cast<Compute>(in_w) / static_cast<Compute>(out_w);
            z = (od + Compute(0.5)) * scale_d - Compute(0.5);
            y = (oh + Compute(0.5)) * scale_h - Compute(0.5);
            x = (ow + Compute(0.5)) * scale_w - Compute(0.5);
        }

        const Compute zhi = static_cast<Compute>(in_d - 1);
        const Compute yhi = static_cast<Compute>(in_h - 1);
        const Compute xhi = static_cast<Compute>(in_w - 1);
        z = z < Compute(0) ? Compute(0) : (z > zhi ? zhi : z);
        y = y < Compute(0) ? Compute(0) : (y > yhi ? yhi : y);
        x = x < Compute(0) ? Compute(0) : (x > xhi ? xhi : x);

        int64_t z0 = static_cast<int64_t>(z);
        int64_t y0 = static_cast<int64_t>(y);
        int64_t x0 = static_cast<int64_t>(x);
        int64_t z1 = min(z0 + 1, in_d - 1);
        int64_t y1 = min(y0 + 1, in_h - 1);
        int64_t x1 = min(x0 + 1, in_w - 1);

        Compute fz = z - z0;
        Compute fy = y - y0;
        Compute fx = x - x0;

        int64_t base = (b * channels + c) * in_d * in_h * in_w;

        Compute v000 = static_cast<Compute>(input[base + z0 * in_h * in_w + y0 * in_w + x0]);
        Compute v001 = static_cast<Compute>(input[base + z0 * in_h * in_w + y0 * in_w + x1]);
        Compute v010 = static_cast<Compute>(input[base + z0 * in_h * in_w + y1 * in_w + x0]);
        Compute v011 = static_cast<Compute>(input[base + z0 * in_h * in_w + y1 * in_w + x1]);
        Compute v100 = static_cast<Compute>(input[base + z1 * in_h * in_w + y0 * in_w + x0]);
        Compute v101 = static_cast<Compute>(input[base + z1 * in_h * in_w + y0 * in_w + x1]);
        Compute v110 = static_cast<Compute>(input[base + z1 * in_h * in_w + y1 * in_w + x0]);
        Compute v111 = static_cast<Compute>(input[base + z1 * in_h * in_w + y1 * in_w + x1]);

        Compute result =
            v000 * (Compute(1) - fz) * (Compute(1) - fy) * (Compute(1) - fx) +
            v001 * (Compute(1) - fz) * (Compute(1) - fy) * fx +
            v010 * (Compute(1) - fz) * fy * (Compute(1) - fx) +
            v011 * (Compute(1) - fz) * fy * fx +
            v100 * fz * (Compute(1) - fy) * (Compute(1) - fx) +
            v101 * fz * (Compute(1) - fy) * fx +
            v110 * fz * fy * (Compute(1) - fx) +
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

    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
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

// Unfold host function (LL.3: per-axis kernel/stride/padding/dilation)
auto unfold_cuda(const Tensor& input,
                 int64_t kernel_h,
                 int64_t kernel_w,
                 int64_t stride_h,
                 int64_t stride_w,
                 int64_t padding_h,
                 int64_t padding_w,
                 int64_t dilation_h,
                 int64_t dilation_w,
                 cudaStream_t stream) -> Tensor {
    // The unfold kernel indexes input with flat contiguous-NCHW offsets, so a
    // non-contiguous (sliced/permuted/channels-last) view must be materialized.
    auto input_c = input.contiguous();
    auto shape = input_c.shape();
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
    Tensor output(output_shape, input_c.dtype(), input_c.device());

    // Launch kernel
    int64_t total_elements = batch * channels * kernel_h * kernel_w * num_blocks;
    dim3 grid, block;
    compute_launch_config_1d(total_elements, grid, block);

    if (input_c.dtype() == DType::Float16) {
        unfold_kernel<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input_c.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            batch, channels, height, width,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w,
            out_h, out_w);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input_c.dtype() == DType::BFloat16) {
        unfold_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input_c.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            batch, channels, height, width,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w,
            out_h, out_w);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        TENZOR_DISPATCH_FLOATING_TYPES(input_c.dtype(), "unfold_cuda", [&]() {
            unfold_kernel<scalar_t><<<grid, block, 0, stream>>>(
                input_c.data<scalar_t>(),
                output.data<scalar_t>(),
                batch, channels, height, width,
                kernel_h, kernel_w, stride_h, stride_w,
                padding_h, padding_w, dilation_h, dilation_w,
                out_h, out_w
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        });
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return output;
}

// Fold host function (LL.3: per-axis kernel/stride/padding/dilation)
auto fold_cuda(const Tensor& input,
               const std::vector<int64_t>& output_size,
               int64_t kernel_h,
               int64_t kernel_w,
               int64_t stride_h,
               int64_t stride_w,
               int64_t padding_h,
               int64_t padding_w,
               int64_t dilation_h,
               int64_t dilation_w,
               cudaStream_t stream) -> Tensor {
    // The fold kernel reads the column buffer with flat contiguous offsets.
    auto input_c = input.contiguous();
    auto shape = input_c.shape();
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
    Tensor output(output_shape, input_c.dtype(), input_c.device());

    // Initialize to zero
    TENZOR_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, output.numel() * output.dtype_size(), stream));

    // Launch kernel
    int64_t total_elements = batch * col_channels * num_blocks;
    dim3 grid, block;
    compute_launch_config_1d(total_elements, grid, block);

    if (input_c.dtype() == DType::Float16) {
        fold_kernel_fp16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input_c.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            batch, channels, height, width,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w,
            out_h, out_w);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    } else if (input_c.dtype() == DType::BFloat16) {
        fold_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input_c.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            batch, channels, height, width,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w,
            out_h, out_w);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    } else {
        TENZOR_DISPATCH_FLOATING_TYPES(input_c.dtype(), "fold_cuda", [&]() {
            fold_kernel<scalar_t><<<grid, block, 0, stream>>>(
                input_c.data<scalar_t>(),
                output.data<scalar_t>(),
                batch, channels, height, width,
                kernel_h, kernel_w, stride_h, stride_w,
                padding_h, padding_w, dilation_h, dilation_w,
                out_h, out_w
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        });
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return output;
}

// M11 fix: nearest-mode interpolate backward. Each output pixel scatters
// its gradient to the single nearest input pixel via atomicAdd. Multiple
// output positions can map to the same input pixel; atomicAdd accumulates
// safely. Mirrors `interpolate_nearest_backward_kernel_hip` from ROCm
// (Wave H4) — same PyTorch nearest-mode convention (floor mapping, no
// half-pixel, no align_corners).
template<typename T>
__global__ void interpolate_nearest_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ grad_in,
    int64_t batch, int64_t channels,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w)
{
    int64_t total = batch * channels * out_h * out_w;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t c  = temp % channels; temp /= channels;
        int64_t b  = temp;

        // Use the IDENTICAL source-pixel mapping as the forward kernel
        // (float scale then truncate); float `oh*scale` and integer floor of
        // `oh*in_h/out_h` can disagree at integer boundaries, which would land
        // the gradient on the wrong input pixel and break parity.
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

template<typename T>
__global__ void interpolate_bilinear_backward_kernel(
    const T* grad_out, T* grad_in,
    int64_t batch, int64_t channels,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    bool align_corners
) {
    // Compute in double for Float64 inputs to preserve FP64 precision/parity
    // with the CPU reference; float otherwise.
    using Compute = typename std::conditional<std::is_same<T, double>::value, double, float>::type;
    int64_t total = batch * channels * out_h * out_w;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t c  = temp % channels; temp /= channels;
        int64_t b  = temp;

        Compute y, x;
        if (align_corners) {
            y = (out_h > 1) ? oh * static_cast<Compute>(in_h - 1) / (out_h - 1) : Compute(0);
            x = (out_w > 1) ? ow * static_cast<Compute>(in_w - 1) / (out_w - 1) : Compute(0);
        } else {
            Compute scale_h = static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
            Compute scale_w = static_cast<Compute>(in_w) / static_cast<Compute>(out_w);
            y = (oh + Compute(0.5)) * scale_h - Compute(0.5);
            x = (ow + Compute(0.5)) * scale_w - Compute(0.5);
        }
        const Compute yhi = static_cast<Compute>(in_h - 1);
        const Compute xhi = static_cast<Compute>(in_w - 1);
        y = y < Compute(0) ? Compute(0) : (y > yhi ? yhi : y);
        x = x < Compute(0) ? Compute(0) : (x > xhi ? xhi : x);

        int64_t y0 = static_cast<int64_t>(y);
        int64_t x0 = static_cast<int64_t>(x);
        int64_t y1 = min(y0 + 1, in_h - 1);
        int64_t x1 = min(x0 + 1, in_w - 1);
        Compute fy = y - y0;
        Compute fx = x - x0;

        Compute w00 = (Compute(1) - fy) * (Compute(1) - fx);
        Compute w01 = (Compute(1) - fy) * fx;
        Compute w10 = fy * (Compute(1) - fx);
        Compute w11 = fy * fx;

        Compute g = static_cast<Compute>(grad_out[idx]);
        int64_t base_idx = b * (channels * in_h * in_w) + c * (in_h * in_w);
        // Typed atomicAdd: CUDA provides overloads for float (compute 2.0+),
        // double (6.0+), __half (7.0+), and __nv_bfloat16 (8.0+).
        atomicAdd(&grad_in[base_idx + y0 * in_w + x0], static_cast<T>(w00 * g));
        atomicAdd(&grad_in[base_idx + y0 * in_w + x1], static_cast<T>(w01 * g));
        atomicAdd(&grad_in[base_idx + y1 * in_w + x0], static_cast<T>(w10 * g));
        atomicAdd(&grad_in[base_idx + y1 * in_w + x1], static_cast<T>(w11 * g));
    }
}

// Bicubic backward (4D): scatter each output gradient to its 4x4 input neighborhood.
template <typename T>
__global__ void interpolate_bicubic_backward_kernel(
    const T* grad_out, T* grad_in,
    int64_t batch, int64_t channels, int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w, bool align_corners)
{
    // Compute in double for Float64 inputs to preserve FP64 precision/parity
    // with the CPU reference; float otherwise.
    using Compute = typename std::conditional<std::is_same<T, double>::value, double, float>::type;
    int64_t total = batch * channels * out_h * out_w;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t c  = temp % channels; temp /= channels;
        int64_t b  = temp;

        // Must use the EXACT coordinate convention of the forward kernel so the
        // analytic gradient matches: forward clamps the continuous coord to
        // [0,in-1] BEFORE flooring (here truncating, identical for the clamped
        // non-negative value) and derives the 4x4 neighborhood from that. Without
        // the clamp, non-align_corners border pixels (src<0) floored to -1 scatter
        // to a different neighborhood with different fractional offsets — gradcheck
        // divergence at edges.
        Compute src_h, src_w;
        if (align_corners) {
            src_h = (out_h > 1) ? oh * static_cast<Compute>(in_h - 1) / (out_h - 1) : Compute(0);
            src_w = (out_w > 1) ? ow * static_cast<Compute>(in_w - 1) / (out_w - 1) : Compute(0);
        } else {
            Compute scale_h = static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
            Compute scale_w = static_cast<Compute>(in_w) / static_cast<Compute>(out_w);
            src_h = (oh + Compute(0.5)) * scale_h - Compute(0.5);
            src_w = (ow + Compute(0.5)) * scale_w - Compute(0.5);
        }
        const Compute hhi = static_cast<Compute>(in_h - 1);
        const Compute whi = static_cast<Compute>(in_w - 1);
        src_h = src_h < Compute(0) ? Compute(0) : (src_h > hhi ? hhi : src_h);
        src_w = src_w < Compute(0) ? Compute(0) : (src_w > whi ? whi : src_w);
        int64_t hi = static_cast<int64_t>(src_h);
        int64_t wi = static_cast<int64_t>(src_w);

        Compute g = static_cast<Compute>(grad_out[idx]);
        int64_t base = b * (channels * in_h * in_w) + c * (in_h * in_w);
        for (int dy = -1; dy <= 2; ++dy) {
            int64_t iy = min(max(static_cast<int64_t>(0), hi + dy), in_h - 1);
            Compute wy = tz_bicubic_coeff<Compute>(src_h - static_cast<Compute>(hi + dy));
            for (int dx = -1; dx <= 2; ++dx) {
                int64_t ix = min(max(static_cast<int64_t>(0), wi + dx), in_w - 1);
                Compute wx = tz_bicubic_coeff<Compute>(src_w - static_cast<Compute>(wi + dx));
                atomicAdd(&grad_in[base + iy * in_w + ix], static_cast<T>(wy * wx * g));
            }
        }
    }
}

// Trilinear backward (5D): scatter each output gradient to its 8 input corners.
template <typename T>
__global__ void interpolate_trilinear_backward_kernel(
    const T* grad_out, T* grad_in,
    int64_t batch, int64_t channels, int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_d, int64_t out_h, int64_t out_w, bool align_corners)
{
    // Compute in double for Float64 inputs to preserve FP64 precision/parity
    // with the CPU reference; float otherwise.
    using Compute = typename std::conditional<std::is_same<T, double>::value, double, float>::type;
    int64_t total = batch * channels * out_d * out_h * out_w;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t od = temp % out_d; temp /= out_d;
        int64_t c  = temp % channels; temp /= channels;
        int64_t b  = temp;

        Compute scd = (align_corners && out_d > 1) ? static_cast<Compute>(in_d - 1) / (out_d - 1) : static_cast<Compute>(in_d) / static_cast<Compute>(out_d);
        Compute sch = (align_corners && out_h > 1) ? static_cast<Compute>(in_h - 1) / (out_h - 1) : static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
        Compute scw = (align_corners && out_w > 1) ? static_cast<Compute>(in_w - 1) / (out_w - 1) : static_cast<Compute>(in_w) / static_cast<Compute>(out_w);
        // Mirror the trilinear FORWARD (vision.cu interpolate_trilinear_kernel):
        // clamp the continuous source coord to [0, in-1] BEFORE flooring, then
        // derive the neighbor pair as {d0, min(d0+1, in_d-1)} with fractional
        // weight fd = src - d0. This makes the backward the EXACT transpose of
        // the clamped forward at non-align_corners borders. Without the clamp,
        // a border output voxel whose src < 0 floors to -1, scattering the
        // (1-fd) mass that the forward assigns to the boundary voxel into a
        // dropped out-of-range tap — a forward/backward inconsistency that fails
        // gradcheck at edges for non-align_corners upsampling.
        Compute sd = align_corners ? od * scd : (od + Compute(0.5)) * scd - Compute(0.5);
        Compute sh = align_corners ? oh * sch : (oh + Compute(0.5)) * sch - Compute(0.5);
        Compute sw = align_corners ? ow * scw : (ow + Compute(0.5)) * scw - Compute(0.5);
        const Compute dhi = static_cast<Compute>(in_d - 1);
        const Compute hhi = static_cast<Compute>(in_h - 1);
        const Compute whi = static_cast<Compute>(in_w - 1);
        sd = sd < Compute(0) ? Compute(0) : (sd > dhi ? dhi : sd);
        sh = sh < Compute(0) ? Compute(0) : (sh > hhi ? hhi : sh);
        sw = sw < Compute(0) ? Compute(0) : (sw > whi ? whi : sw);
        int64_t d0 = static_cast<int64_t>(sd), h0 = static_cast<int64_t>(sh), w0 = static_cast<int64_t>(sw);
        int64_t d1 = min(d0 + 1, in_d - 1), h1 = min(h0 + 1, in_h - 1), w1 = min(w0 + 1, in_w - 1);
        Compute fd = sd - d0, fh = sh - h0, fw = sw - w0;
        Compute g = static_cast<Compute>(grad_out[idx]);
        int64_t base = b * (channels * in_d * in_h * in_w) + c * (in_d * in_h * in_w);
#define TZ_TRI_ADD(dd, hh, ww, wgt) do { \
        if ((dd) >= 0 && (dd) < in_d && (hh) >= 0 && (hh) < in_h && (ww) >= 0 && (ww) < in_w) \
            atomicAdd(&grad_in[base + (dd) * in_h * in_w + (hh) * in_w + (ww)], static_cast<T>((wgt) * g)); \
    } while (0)
        TZ_TRI_ADD(d0, h0, w0, (Compute(1) - fd) * (Compute(1) - fh) * (Compute(1) - fw));
        TZ_TRI_ADD(d0, h0, w1, (Compute(1) - fd) * (Compute(1) - fh) * fw);
        TZ_TRI_ADD(d0, h1, w0, (Compute(1) - fd) * fh * (Compute(1) - fw));
        TZ_TRI_ADD(d0, h1, w1, (Compute(1) - fd) * fh * fw);
        TZ_TRI_ADD(d1, h0, w0, fd * (Compute(1) - fh) * (Compute(1) - fw));
        TZ_TRI_ADD(d1, h0, w1, fd * (Compute(1) - fh) * fw);
        TZ_TRI_ADD(d1, h1, w0, fd * fh * (Compute(1) - fw));
        TZ_TRI_ADD(d1, h1, w1, fd * fh * fw);
#undef TZ_TRI_ADD
    }
}

// Nearest backward (5D): each output voxel scatters its gradient to the single
// nearest input voxel via atomicAdd. Uses the IDENTICAL source-voxel mapping as
// the 5D nearest FORWARD kernel (interpolate_nearest_5d_kernel: float scale =
// in/out, multiply, truncate, clamp to in-1) so the backward is the exact
// transpose. Multiple output voxels can map to the same input voxel; atomicAdd
// accumulates safely. Mirrors interpolate_nearest_backward_kernel (4D) and the
// CPU nearest_backward_axis_scatter (rank-generic). No half-pixel, no
// align_corners — PyTorch nearest convention.
template<typename T>
__global__ void interpolate_nearest_5d_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ grad_in,
    int64_t batch, int64_t channels,
    int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_d, int64_t out_h, int64_t out_w)
{
    float scale_d = static_cast<float>(in_d) / out_d;
    float scale_h = static_cast<float>(in_h) / out_h;
    float scale_w = static_cast<float>(in_w) / out_w;
    int64_t total = batch * channels * out_d * out_h * out_w;

    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
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
        atomicAdd(&grad_in[in_idx], grad_out[idx]);
    }
}

// Interpolate host function
auto interpolate_cuda(const Tensor& input_in,
                      const std::vector<int64_t>& size,
                      const std::string& mode,
                      bool align_corners) -> Tensor {
    // The interpolate kernels index input with flat contiguous-NCHW/NCDHW
    // offsets, so a non-contiguous view must be materialized first.
    auto input_c = input_in.contiguous();
    const Tensor& input = input_c;
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
        } else if (input.dtype() == DType::BFloat16) {
            interpolate_nearest_kernel<__nv_bfloat16><<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
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
        } else if (input.dtype() == DType::BFloat16) {
            interpolate_bilinear_kernel<__nv_bfloat16><<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
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
        } else if (input.dtype() == DType::BFloat16) {
            interpolate_bicubic_kernel<__nv_bfloat16><<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
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
// Audit D3: bilinear backward host dispatcher (device-resident scatter).
// ============================================================================
auto interpolate_backward_cuda(const Tensor& grad_output,
                                const std::vector<int64_t>& input_size,
                                const std::string& mode,
                                bool align_corners) -> Tensor {
    auto shape = grad_output.shape();

    // Float16/BFloat16 have no atomicAdd; the scatter-add backward kernels are
    // float-only (TENZOR_DISPATCH_FLOATING_TYPES). Widen to Float32, compute the
    // gradient, then narrow back — mirrors the oneAPI/Vulkan fix for the same op.
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        const DType odt = grad_output.dtype();
        return interpolate_backward_cuda(grad_output.to(DType::Float32), input_size, mode, align_corners).to(odt);
    }

    // Trilinear backward operates on 5D (N, C, D, H, W).
    if (mode == "trilinear") {
        if (shape.size() != 5)
            throw std::runtime_error("interpolate_backward_cuda: trilinear requires 5D (N,C,D,H,W).");
        if (input_size.size() != 3)
            throw std::runtime_error("interpolate_backward_cuda: trilinear input_size must be [in_d, in_h, in_w].");
        const int64_t N = shape[0], C = shape[1], out_d = shape[2], out_h = shape[3], out_w = shape[4];
        const int64_t in_d = input_size[0], in_h = input_size[1], in_w = input_size[2];
        Tensor grad_input({N, C, in_d, in_h, in_w}, grad_output.dtype(), grad_output.device());
        TENZOR_CUDA_CHECK(cudaMemset(grad_input.data_ptr(), 0,
                                      static_cast<size_t>(grad_input.numel()) * dtype_size(grad_input.dtype())));
        int64_t total = N * C * out_d * out_h * out_w;
        int threads = 256;
        int blocks  = static_cast<int>((total + threads - 1) / threads);
        TENZOR_DISPATCH_FLOATING_TYPES(grad_output.dtype(), "interpolate_trilinear_backward", [&]() {
            interpolate_trilinear_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad_output.data<scalar_t>(), grad_input.data<scalar_t>(),
                N, C, in_d, in_h, in_w, out_d, out_h, out_w, align_corners);
        });
        TENZOR_CUDA_POST_LAUNCH_CHECK();
        return grad_input;
    }

    // 5D nearest backward: forward supports 5D nearest (interpolate_nearest_5d_kernel),
    // so the backward must too (CPU routes nearest to the rank-generic
    // nearest_backward_axis_scatter). Scatter-add each output voxel's gradient
    // to its single nearest input voxel.
    if (mode == "nearest" && shape.size() == 5) {
        if (input_size.size() != 3)
            throw std::runtime_error("interpolate_backward_cuda: 5D nearest input_size must be [in_d, in_h, in_w].");
        const int64_t N = shape[0], C = shape[1], out_d = shape[2], out_h = shape[3], out_w = shape[4];
        const int64_t in_d = input_size[0], in_h = input_size[1], in_w = input_size[2];
        Tensor grad_input({N, C, in_d, in_h, in_w}, grad_output.dtype(), grad_output.device());
        TENZOR_CUDA_CHECK(cudaMemset(grad_input.data_ptr(), 0,
                                      static_cast<size_t>(grad_input.numel()) * dtype_size(grad_input.dtype())));
        int64_t total = N * C * out_d * out_h * out_w;
        int threads = 256;
        int blocks  = static_cast<int>((total + threads - 1) / threads);
        TENZOR_DISPATCH_FLOATING_TYPES(grad_output.dtype(), "interpolate_nearest_5d_backward", [&]() {
            interpolate_nearest_5d_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad_output.data<scalar_t>(), grad_input.data<scalar_t>(),
                N, C, in_d, in_h, in_w, out_d, out_h, out_w);
        });
        TENZOR_CUDA_POST_LAUNCH_CHECK();
        return grad_input;
    }

    if (mode != "bilinear" && mode != "nearest" && mode != "bicubic") {
        throw std::runtime_error("interpolate_backward_cuda: mode '" + mode +
            "' not supported. Use 'bilinear'/'nearest'/'bicubic' (4D), 'nearest' (5D), or 'trilinear' (5D).");
    }
    if (shape.size() != 4) {
        throw std::runtime_error("interpolate_backward_cuda: bilinear/nearest/bicubic require 4D (N,C,H,W).");
    }
    if (input_size.size() != 2) {
        throw std::runtime_error("interpolate_backward_cuda: input_size must be [in_h, in_w].");
    }
    const int64_t N     = shape[0];
    const int64_t C     = shape[1];
    const int64_t out_h = shape[2];
    const int64_t out_w = shape[3];
    const int64_t in_h  = input_size[0];
    const int64_t in_w  = input_size[1];

    Tensor grad_input({N, C, in_h, in_w}, grad_output.dtype(), grad_output.device());
    TENZOR_CUDA_CHECK(cudaMemset(grad_input.data_ptr(), 0,
                                  static_cast<size_t>(grad_input.numel()) * dtype_size(grad_input.dtype())));

    int64_t total = N * C * out_h * out_w;
    int threads = 256;
    int blocks  = static_cast<int>((total + threads - 1) / threads);

    if (mode == "bilinear") {
        TENZOR_DISPATCH_FLOATING_TYPES(grad_output.dtype(), "interpolate_bilinear_backward", [&]() {
            interpolate_bilinear_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad_output.data<scalar_t>(), grad_input.data<scalar_t>(),
                N, C, in_h, in_w, out_h, out_w, align_corners);
        });
    } else if (mode == "bicubic") {
        TENZOR_DISPATCH_FLOATING_TYPES(grad_output.dtype(), "interpolate_bicubic_backward", [&]() {
            interpolate_bicubic_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad_output.data<scalar_t>(), grad_input.data<scalar_t>(),
                N, C, in_h, in_w, out_h, out_w, align_corners);
        });
    } else {
        // Native nearest-mode backward via atomicAdd scatter (each output pixel
        // writes its gradient to the single nearest input pixel).
        TENZOR_DISPATCH_FLOATING_TYPES(grad_output.dtype(), "interpolate_nearest_backward", [&]() {
            interpolate_nearest_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad_output.data<scalar_t>(), grad_input.data<scalar_t>(),
                N, C, in_h, in_w, out_h, out_w);
        });
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return grad_input;
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

    // Match the CPU reference (src/backends/cpu/kernels/vision.cpp): plain IoU
    // is 0 for a zero/degenerate union rather than inter_area/1e-7 (which blows
    // up tiny numerical-noise intersections or yields a negative ratio for
    // inverted boxes).
    T iou = (union_area > static_cast<T>(0)) ? (inter_area / union_area) : static_cast<T>(0);

    if (iou_type == 1) {
        // GIoU: IoU - (enclosing_area - union_area) / enclosing_area
        T enc_x1 = min(x1_1, x1_2);
        T enc_y1 = min(y1_1, y1_2);
        T enc_x2 = max(x2_1, x2_2);
        T enc_y2 = max(y2_1, y2_2);
        T enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1);
        // Divide by max(enc_area, 1e-7), matching CPU std::max(enclose_area, 1e-7f).
        iou = iou - (enc_area - union_area) / max(enc_area, static_cast<T>(1e-7));
    } else if (iou_type == 2 || iou_type == 3) {
        // DIoU (2) / CIoU (3): subtract the normalized squared center distance
        // (and, for CIoU, the aspect-ratio penalty). Previously these fell
        // through and returned plain IoU — off by the penalty term(s).
        T cx1 = (x1_1 + x2_1) / static_cast<T>(2);
        T cy1 = (y1_1 + y2_1) / static_cast<T>(2);
        T cx2 = (x1_2 + x2_2) / static_cast<T>(2);
        T cy2 = (y1_2 + y2_2) / static_cast<T>(2);
        T center_dist_sq = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);

        T enc_x1 = min(x1_1, x1_2);
        T enc_y1 = min(y1_1, y1_2);
        T enc_x2 = max(x2_1, x2_2);
        T enc_y2 = max(y2_1, y2_2);
        T enc_w = enc_x2 - enc_x1;
        T enc_h = enc_y2 - enc_y1;
        T diag_dist_sq = enc_w * enc_w + enc_h * enc_h;

        T result = iou - center_dist_sq / (diag_dist_sq + static_cast<T>(1e-7));
        if (iou_type == 3) {
            T w1 = x2_1 - x1_1, h1 = y2_1 - y1_1;
            T w2 = x2_2 - x1_2, h2 = y2_2 - y1_2;
            const T four_over_pi_sq =
                static_cast<T>(4.0 / (3.14159265358979323846 * 3.14159265358979323846));
            T diff = atan(w2 / (h2 + static_cast<T>(1e-7))) -
                     atan(w1 / (h1 + static_cast<T>(1e-7)));
            T v = four_over_pi_sq * diff * diff;
            T alpha = v / (static_cast<T>(1) - iou + v + static_cast<T>(1e-7));  // original IoU
            result = result - alpha * v;
        }
        iou = result;
    }

    output[i * M + j] = iou;
}

auto box_iou_cuda(const Tensor& boxes1_in, const Tensor& boxes2_in, int iou_type) -> Tensor {
    // box_iou_kernel indexes each box row flat (i*4+k); materialize contiguous
    // copies so a sliced/permuted boxes view is not read with wrong strides.
    auto boxes1_c = boxes1_in.contiguous();
    auto boxes2_c = boxes2_in.contiguous();
    const Tensor& boxes1 = boxes1_c;
    const Tensor& boxes2 = boxes2_c;
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
    } else if (boxes1.dtype() == DType::Float16 || boxes1.dtype() == DType::BFloat16) {
        // Upcast to Float32 for computation
        Tensor output_f32({N, M}, DType::Float32, boxes1.device());
        // Cast inputs to Float32
        Tensor b1_f32({boxes1.shape()[0], 4}, DType::Float32, boxes1.device());
        Tensor b2_f32({boxes2.shape()[0], 4}, DType::Float32, boxes2.device());
        int64_t n1 = boxes1.numel(), n2 = boxes2.numel();
        int cvt_block = 256;
        if (boxes1.dtype() == DType::Float16) {
            half_to_float_kernel<<<(n1+cvt_block-1)/cvt_block, cvt_block>>>(
                reinterpret_cast<const __half*>(boxes1.data_ptr()), b1_f32.data<float>(), n1);
            half_to_float_kernel<<<(n2+cvt_block-1)/cvt_block, cvt_block>>>(
                reinterpret_cast<const __half*>(boxes2.data_ptr()), b2_f32.data<float>(), n2);
        } else {
            half_to_float_kernel<<<(n1+cvt_block-1)/cvt_block, cvt_block>>>(
                reinterpret_cast<const __nv_bfloat16*>(boxes1.data_ptr()), b1_f32.data<float>(), n1);
            half_to_float_kernel<<<(n2+cvt_block-1)/cvt_block, cvt_block>>>(
                reinterpret_cast<const __nv_bfloat16*>(boxes2.data_ptr()), b2_f32.data<float>(), n2);
        }
        TENZOR_CUDA_POST_LAUNCH_CHECK();
        box_iou_kernel<float><<<grid, block>>>(
            b1_f32.data<float>(), b2_f32.data<float>(), output_f32.data<float>(),
            N, M, iou_type);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
        // Convert output back
        int64_t out_n = N * M;
        if (boxes1.dtype() == DType::Float16) {
            float_to_half_kernel<<<(out_n+cvt_block-1)/cvt_block, cvt_block>>>(
                output_f32.data<float>(), reinterpret_cast<__half*>(output.data_ptr()), out_n);
        } else {
            float_to_half_kernel<<<(out_n+cvt_block-1)/cvt_block, cvt_block>>>(
                output_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), out_n);
        }
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("box_iou_cuda: unsupported dtype");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return output;
}

} // namespace cuda
} // namespace tenzor
