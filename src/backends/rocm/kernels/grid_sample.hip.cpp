/**
 * @file grid_sample.hip.cpp
 * @brief HIP/ROCm port of grid_sample and affine_grid kernels.
 *
 * Mirrors src/backends/cuda/kernels/grid_sample.cu line-for-line; replaces the
 * previous CPU-roundtrip fallback in rocm_kernel_registry.cpp.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {
namespace rocm {

// Forward decls (defined in math.hip.cpp)
#ifndef HIP_CHECK
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)
#endif

// =========================================================================
// Device helpers
// =========================================================================

__device__ __forceinline__ float gs_denormalize_dev(float coord, int size, bool align_corners) {
    if (align_corners) {
        return (coord + 1.0f) * 0.5f * static_cast<float>(size - 1);
    } else {
        return ((coord + 1.0f) * static_cast<float>(size) - 1.0f) * 0.5f;
    }
}

__device__ __forceinline__ float gs_reflect_coord(float coord, int size) {
    if (size <= 1) return 0.0f;
    float max_val = static_cast<float>(size - 1);
    coord = fabsf(coord);
    float period = 2.0f * max_val;
    coord = fmodf(coord, period);
    if (coord > max_val) coord = period - coord;
    return coord;
}

// =========================================================================
// Bilinear grid_sample kernel
// =========================================================================

__global__ void grid_sample_bilinear_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grid,
    float* __restrict__ output,
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
    float gx = grid[grid_idx];
    float gy = grid[grid_idx + 1];

    float ix = gs_denormalize_dev(gx, W_in, align_corners);
    float iy = gs_denormalize_dev(gy, H_in, align_corners);

    if (padding_mode == 1) {
        ix = fminf(fmaxf(ix, 0.0f), static_cast<float>(W_in - 1));
        iy = fminf(fmaxf(iy, 0.0f), static_cast<float>(H_in - 1));
    } else if (padding_mode == 2) {
        ix = gs_reflect_coord(ix, W_in);
        iy = gs_reflect_coord(iy, H_in);
    }

    int x0 = static_cast<int>(floorf(ix));
    int y0 = static_cast<int>(floorf(iy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wx1 = ix - static_cast<float>(x0);
    float wy1 = iy - static_cast<float>(y0);
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;

    auto safe_get = [&](int y, int x) -> float {
        if (y >= 0 && y < H_in && x >= 0 && x < W_in)
            return input[((n * C + c) * H_in + y) * W_in + x];
        return 0.0f;
    };

    float val = wy0 * wx0 * safe_get(y0, x0) +
                wy0 * wx1 * safe_get(y0, x1) +
                wy1 * wx0 * safe_get(y1, x0) +
                wy1 * wx1 * safe_get(y1, x1);

    output[((n * C + c) * H_out + h) * W_out + w] = val;
}

// =========================================================================
// Nearest grid_sample kernel
// =========================================================================

__global__ void grid_sample_nearest_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grid,
    float* __restrict__ output,
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
    float gx = grid[grid_idx];
    float gy = grid[grid_idx + 1];

    float ix = gs_denormalize_dev(gx, W_in, align_corners);
    float iy = gs_denormalize_dev(gy, H_in, align_corners);

    if (padding_mode == 1) {
        ix = fminf(fmaxf(ix, 0.0f), static_cast<float>(W_in - 1));
        iy = fminf(fmaxf(iy, 0.0f), static_cast<float>(H_in - 1));
    } else if (padding_mode == 2) {
        ix = gs_reflect_coord(ix, W_in);
        iy = gs_reflect_coord(iy, H_in);
    }

    int nx = static_cast<int>(roundf(ix));
    int ny = static_cast<int>(roundf(iy));

    float val = 0.0f;
    if (ny >= 0 && ny < H_in && nx >= 0 && nx < W_in) {
        val = input[((n * C + c) * H_in + ny) * W_in + nx];
    }

    output[((n * C + c) * H_out + h) * W_out + w] = val;
}

// =========================================================================
// Affine grid kernel
// =========================================================================

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

// =========================================================================
// Host API
// =========================================================================

auto grid_sample_kernel(const Tensor& input, const Tensor& grid,
                        const std::string& mode, const std::string& padding_mode,
                        bool align_corners, hipStream_t stream) -> Tensor {
    auto in_shape = input.shape();
    auto grid_shape = grid.shape();

    int N = static_cast<int>(in_shape[0]);
    int C = static_cast<int>(in_shape[1]);
    int H_in = static_cast<int>(in_shape[2]);
    int W_in = static_cast<int>(in_shape[3]);
    int H_out = static_cast<int>(grid_shape[1]);
    int W_out = static_cast<int>(grid_shape[2]);

    Tensor input_f32 = input.to(DType::Float32);
    Tensor grid_f32 = grid.to(DType::Float32);

    Tensor output_f32({N, C, H_out, W_out}, DType::Float32, input.device());

    int pad_mode = 0;
    if (padding_mode == "border") pad_mode = 1;
    else if (padding_mode == "reflection") pad_mode = 2;

    int total = N * C * H_out * W_out;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    if (mode == "nearest") {
        hipLaunchKernelGGL(grid_sample_nearest_kernel,
            dim3(grid_size), dim3(block_size), 0, stream,
            input_f32.data<float>(), grid_f32.data<float>(),
            output_f32.data<float>(),
            N, C, H_in, W_in, H_out, W_out,
            pad_mode, align_corners);
    } else {
        hipLaunchKernelGGL(grid_sample_bilinear_kernel,
            dim3(grid_size), dim3(block_size), 0, stream,
            input_f32.data<float>(), grid_f32.data<float>(),
            output_f32.data<float>(),
            N, C, H_in, W_in, H_out, W_out,
            pad_mode, align_corners);
    }

    HIP_CHECK(hipGetLastError());
    return output_f32.to(input.dtype());
}

auto affine_grid_kernel_host(const Tensor& theta, const std::vector<int64_t>& size,
                              bool align_corners, hipStream_t stream) -> Tensor {
    int N = static_cast<int>(size[0]);
    int H = static_cast<int>(size[2]);
    int W = static_cast<int>(size[3]);

    Tensor theta_f32 = theta.to(DType::Float32);
    Tensor grid({N, H, W, 2}, DType::Float32, theta.device());

    int total = N * H * W;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    hipLaunchKernelGGL(affine_grid_kernel,
        dim3(grid_size), dim3(block_size), 0, stream,
        theta_f32.data<float>(), grid.data<float>(),
        N, H, W, align_corners);

    HIP_CHECK(hipGetLastError());
    return grid;
}

}  // namespace rocm
}  // namespace tenzor
