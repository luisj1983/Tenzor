/**
 * @file grid_sample.cu
 * @brief CUDA kernel for grid_sample and affine_grid operations.
 *
 * Implements bilinear/nearest/bicubic grid sampling with zeros/border/reflection
 * padding on GPU. Uses per-thread element processing for simplicity and correctness.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "cuda_common.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace tenzor {
namespace cuda {

// ============================================================================
// Device helpers
// ============================================================================

// E.1: templated coordinate helpers — instantiate for both float and
// double so grid_sample_*_kernel<double> can compute natively without
// widen-narrow precision loss.
template <typename T>
__device__ __forceinline__ T denormalize_dev(T coord, int size, bool align_corners) {
    if (align_corners) {
        return (coord + T(1)) * T(0.5) * static_cast<T>(size - 1);
    } else {
        return ((coord + T(1)) * static_cast<T>(size) - T(1)) * T(0.5);
    }
}

template <typename T>
__device__ __forceinline__ T reflect_coord(T coord, int size) {
    if (size <= 1) return T(0);
    T max_val = static_cast<T>(size - 1);
    if constexpr (std::is_same_v<T, float>)  coord = fabsf(coord);
    else                                     coord = fabs(coord);
    T period = T(2) * max_val;
    if constexpr (std::is_same_v<T, float>)  coord = fmodf(coord, period);
    else                                     coord = fmod(coord, period);
    if (coord > max_val) coord = period - coord;
    return coord;
}

// ============================================================================
// Bilinear grid_sample kernel
// ============================================================================

// E.1: templated bilinear kernel — instantiated for float (FP32 path) and
// double (native FP64 path; no widen-narrow). The non-templated dispatcher
// below picks the right specialization by input dtype.
template <typename T>
__global__ void grid_sample_bilinear_kernel(
    const T* __restrict__ input,
    const T* __restrict__ grid,
    T* __restrict__ output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int padding_mode,  // 0=zeros, 1=border, 2=reflection
    bool align_corners
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (C * H_out * W_out);

    int grid_idx = ((n * H_out + h) * W_out + w) * 2;
    T gx = grid[grid_idx];
    T gy = grid[grid_idx + 1];

    T ix = denormalize_dev<T>(gx, W_in, align_corners);
    T iy = denormalize_dev<T>(gy, H_in, align_corners);

    if (padding_mode == 1) {  // border
        if constexpr (std::is_same_v<T, float>) {
            ix = fminf(fmaxf(ix, 0.0f), static_cast<float>(W_in - 1));
            iy = fminf(fmaxf(iy, 0.0f), static_cast<float>(H_in - 1));
        } else {
            ix = fmin(fmax(ix, 0.0), static_cast<double>(W_in - 1));
            iy = fmin(fmax(iy, 0.0), static_cast<double>(H_in - 1));
        }
    } else if (padding_mode == 2) {  // reflection
        ix = reflect_coord<T>(ix, W_in);
        iy = reflect_coord<T>(iy, H_in);
    }

    int x0, y0;
    if constexpr (std::is_same_v<T, float>) {
        x0 = static_cast<int>(floorf(ix));
        y0 = static_cast<int>(floorf(iy));
    } else {
        x0 = static_cast<int>(floor(ix));
        y0 = static_cast<int>(floor(iy));
    }
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    T wx1 = ix - static_cast<T>(x0);
    T wy1 = iy - static_cast<T>(y0);
    T wx0 = T(1) - wx1;
    T wy0 = T(1) - wy1;

    auto safe_get = [&](int y, int x) -> T {
        if (y >= 0 && y < H_in && x >= 0 && x < W_in)
            return input[((n * C + c) * H_in + y) * W_in + x];
        return T(0);
    };

    T val = wy0 * wx0 * safe_get(y0, x0) +
            wy0 * wx1 * safe_get(y0, x1) +
            wy1 * wx0 * safe_get(y1, x0) +
            wy1 * wx1 * safe_get(y1, x1);

    output[((n * C + c) * H_out + h) * W_out + w] = val;
}

// ============================================================================
// Bicubic grid_sample kernel — Phase P0 / Fix 7
// ============================================================================
//
// 4x4 neighbourhood with Catmull-Rom basis (a = -0.5), matching PyTorch's
// `grid_sample(mode='bicubic')` and Tenzor's own interpolate(mode='bicubic').
//
// E.1: templated cubic-weights helper for native FP32 / FP64 bicubic.
template <typename T>
__device__ inline void cubic_weights_dev(T t, T w[4]) {
    constexpr T a = T(-0.5);
    const T t2 = t * t;
    const T t3 = t2 * t;
    w[0] = ((a * t - T(2) * a) * t + a) * t;
    w[1] = ((a + T(2)) * t3 - (a + T(3)) * t2 + T(1));
    const T u = T(1) - t;
    const T u2 = u * u;
    const T u3 = u2 * u;
    w[2] = ((a + T(2)) * u3 - (a + T(3)) * u2 + T(1));
    w[3] = ((a * u - T(2) * a) * u + a) * u;
}

template <typename T>
__global__ void grid_sample_bicubic_kernel(
    const T* __restrict__ input,
    const T* __restrict__ grid,
    T* __restrict__ output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int padding_mode,  // 0=zeros, 1=border, 2=reflection
    bool align_corners
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (C * H_out * W_out);

    int grid_idx = ((n * H_out + h) * W_out + w) * 2;
    T gx = grid[grid_idx];
    T gy = grid[grid_idx + 1];

    T ix = denormalize_dev<T>(gx, W_in, align_corners);
    T iy = denormalize_dev<T>(gy, H_in, align_corners);

    if (padding_mode == 1) {  // border
        if constexpr (std::is_same_v<T, float>) {
            ix = fminf(fmaxf(ix, 0.0f), static_cast<float>(W_in - 1));
            iy = fminf(fmaxf(iy, 0.0f), static_cast<float>(H_in - 1));
        } else {
            ix = fmin(fmax(ix, 0.0), static_cast<double>(W_in - 1));
            iy = fmin(fmax(iy, 0.0), static_cast<double>(H_in - 1));
        }
    } else if (padding_mode == 2) {  // reflection
        ix = reflect_coord<T>(ix, W_in);
        iy = reflect_coord<T>(iy, H_in);
    }

    int ix_floor, iy_floor;
    if constexpr (std::is_same_v<T, float>) {
        ix_floor = static_cast<int>(floorf(ix));
        iy_floor = static_cast<int>(floorf(iy));
    } else {
        ix_floor = static_cast<int>(floor(ix));
        iy_floor = static_cast<int>(floor(iy));
    }
    const T tx = ix - static_cast<T>(ix_floor);
    const T ty = iy - static_cast<T>(iy_floor);
    T wx[4], wy[4];
    cubic_weights_dev<T>(tx, wx);
    cubic_weights_dev<T>(ty, wy);

    auto safe_get = [&](int y, int x) -> T {
        if (padding_mode == 0) {  // zeros
            if (y < 0 || y >= H_in || x < 0 || x >= W_in) return T(0);
            return input[((n * C + c) * H_in + y) * W_in + x];
        }
        // border / reflection: clamp neighbours past the edge.
        y = max(0, min(y, H_in - 1));
        x = max(0, min(x, W_in - 1));
        return input[((n * C + c) * H_in + y) * W_in + x];
    };

    T val = T(0);
    #pragma unroll
    for (int dy = -1; dy <= 2; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 2; ++dx) {
            val += wy[dy + 1] * wx[dx + 1] * safe_get(iy_floor + dy, ix_floor + dx);
        }
    }

    output[((n * C + c) * H_out + h) * W_out + w] = val;
}

// ============================================================================
// Nearest grid_sample kernel
// ============================================================================

template <typename T>
__global__ void grid_sample_nearest_kernel(
    const T* __restrict__ input,
    const T* __restrict__ grid,
    T* __restrict__ output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int padding_mode, bool align_corners
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (C * H_out * W_out);

    int grid_idx = ((n * H_out + h) * W_out + w) * 2;
    T gx = grid[grid_idx];
    T gy = grid[grid_idx + 1];

    T ix = denormalize_dev<T>(gx, W_in, align_corners);
    T iy = denormalize_dev<T>(gy, H_in, align_corners);

    if (padding_mode == 1) {
        if constexpr (std::is_same_v<T, float>) {
            ix = fminf(fmaxf(ix, 0.0f), static_cast<float>(W_in - 1));
            iy = fminf(fmaxf(iy, 0.0f), static_cast<float>(H_in - 1));
        } else {
            ix = fmin(fmax(ix, 0.0), static_cast<double>(W_in - 1));
            iy = fmin(fmax(iy, 0.0), static_cast<double>(H_in - 1));
        }
    } else if (padding_mode == 2) {
        ix = reflect_coord<T>(ix, W_in);
        iy = reflect_coord<T>(iy, H_in);
    }

    int nx, ny;
    if constexpr (std::is_same_v<T, float>) {
        nx = static_cast<int>(roundf(ix));
        ny = static_cast<int>(roundf(iy));
    } else {
        nx = static_cast<int>(round(ix));
        ny = static_cast<int>(round(iy));
    }

    T val = T(0);
    if (ny >= 0 && ny < H_in && nx >= 0 && nx < W_in) {
        val = input[((n * C + c) * H_in + ny) * W_in + nx];
    }

    output[((n * C + c) * H_out + h) * W_out + w] = val;
}

// ============================================================================
// Affine grid kernel
// ============================================================================

__global__ void affine_grid_kernel(
    const float* __restrict__ theta,
    float* __restrict__ grid,
    int N, int H, int W, bool align_corners
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int n = idx / (H * W);

    float x_norm, y_norm;
    if (align_corners) {
        x_norm = (W > 1) ? (2.0f * static_cast<float>(w) / static_cast<float>(W - 1) - 1.0f) : 0.0f;
        y_norm = (H > 1) ? (2.0f * static_cast<float>(h) / static_cast<float>(H - 1) - 1.0f) : 0.0f;
    } else {
        x_norm = (2.0f * static_cast<float>(w) + 1.0f) / static_cast<float>(W) - 1.0f;
        y_norm = (2.0f * static_cast<float>(h) + 1.0f) / static_cast<float>(H) - 1.0f;
    }

    const float* t = theta + n * 6;
    float x_out = t[0] * x_norm + t[1] * y_norm + t[2];
    float y_out = t[3] * x_norm + t[4] * y_norm + t[5];

    int out_idx = ((n * H + h) * W + w) * 2;
    grid[out_idx] = x_out;
    grid[out_idx + 1] = y_out;
}

// ============================================================================
// Host API
// ============================================================================

auto grid_sample_cuda(const Tensor& input, const Tensor& grid,
                      const std::string& mode, const std::string& padding_mode,
                      bool align_corners) -> Tensor {
    auto in_shape = input.shape();
    auto grid_shape = grid.shape();

    int N = static_cast<int>(in_shape[0]);
    int C = static_cast<int>(in_shape[1]);
    int H_in = static_cast<int>(in_shape[2]);
    int W_in = static_cast<int>(in_shape[3]);
    int H_out = static_cast<int>(grid_shape[1]);
    int W_out = static_cast<int>(grid_shape[2]);

    int pad_mode = 0;
    if (padding_mode == "border") pad_mode = 1;
    else if (padding_mode == "reflection") pad_mode = 2;

    int total = N * C * H_out * W_out;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    // E.1: native FP64 path for ALL modes (nearest, bilinear, bicubic).
    // Templated kernel instantiation, no widen-narrow.
    if (input.dtype() == DType::Float64) {
        Tensor input_f64 = input.contiguous();
        Tensor grid_f64  = grid.to(DType::Float64).contiguous();
        Tensor output_f64({N, C, H_out, W_out}, DType::Float64, input.device());
        if (mode == "nearest") {
            grid_sample_nearest_kernel<double><<<grid_size, block_size>>>(
                input_f64.data<double>(), grid_f64.data<double>(),
                output_f64.data<double>(),
                N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
        } else if (mode == "bilinear") {
            grid_sample_bilinear_kernel<double><<<grid_size, block_size>>>(
                input_f64.data<double>(), grid_f64.data<double>(),
                output_f64.data<double>(),
                N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
        } else if (mode == "bicubic") {
            grid_sample_bicubic_kernel<double><<<grid_size, block_size>>>(
                input_f64.data<double>(), grid_f64.data<double>(),
                output_f64.data<double>(),
                N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
        } else {
            throw std::invalid_argument(
                "grid_sample_cuda: unknown mode '" + mode +
                "'. Supported: bilinear, nearest, bicubic.");
        }
        TENZOR_CUDA_POST_LAUNCH_CHECK();
        return output_f64;
    }

    // Float32 path. FP16/BF16 promote to Float32 via Tensor::to() — those
    // dtypes have fewer mantissa bits than Float32 so this is mathematically
    // lossless at element granularity (no widen of the whole tensor; each
    // load/store is the natural cast).
    Tensor input_f32 = input.to(DType::Float32);
    Tensor grid_f32 = grid.to(DType::Float32);
    Tensor output_f32({N, C, H_out, W_out}, DType::Float32, input.device());

    if (mode == "nearest") {
        grid_sample_nearest_kernel<float><<<grid_size, block_size>>>(
            input_f32.data<float>(), grid_f32.data<float>(),
            output_f32.data<float>(),
            N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
    } else if (mode == "bilinear") {
        grid_sample_bilinear_kernel<float><<<grid_size, block_size>>>(
            input_f32.data<float>(), grid_f32.data<float>(),
            output_f32.data<float>(),
            N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
    } else if (mode == "bicubic") {
        grid_sample_bicubic_kernel<float><<<grid_size, block_size>>>(
            input_f32.data<float>(), grid_f32.data<float>(),
            output_f32.data<float>(),
            N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
    } else {
        throw std::invalid_argument(
            "grid_sample_cuda: unknown mode '" + mode +
            "'. Supported: bilinear, nearest, bicubic.");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return output_f32.to(input.dtype());
}

auto affine_grid_cuda(const Tensor& theta, const std::vector<int64_t>& size,
                      bool align_corners) -> Tensor {
    int N = static_cast<int>(size[0]);
    int H = static_cast<int>(size[2]);
    int W = static_cast<int>(size[3]);

    Tensor theta_f32 = theta.to(DType::Float32);
    Tensor grid({N, H, W, 2}, DType::Float32, theta.device());

    int total = N * H * W;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    affine_grid_kernel<<<grid_size, block_size>>>(
        theta_f32.data<float>(), grid.data<float>(),
        N, H, W, align_corners);

    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return grid;
}

} // namespace cuda
} // namespace tenzor
