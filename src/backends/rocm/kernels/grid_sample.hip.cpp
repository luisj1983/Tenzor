/**
 * @file grid_sample.hip.cpp
 * @brief HIP/ROCm port of grid_sample and affine_grid kernels.
 *
 * Mirrors src/backends/cuda/kernels/grid_sample.cu: templated on the scalar
 * type so Float32 and Float64 both run natively (no widen-narrow), with
 * bilinear / nearest / bicubic sampling and zeros/border/reflection padding.
 * Float16/BFloat16 promote to Float32 (each load/store is the natural cast).
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tenzor {
namespace rocm {

#ifndef HIP_CHECK
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)
#endif

// =========================================================================
// Device helpers (templated for native FP32 / FP64)
// =========================================================================

template <typename T>
__device__ __forceinline__ T gs_denormalize_dev(T coord, int size, bool align_corners) {
    if (align_corners) {
        return (coord + T(1)) * T(0.5) * static_cast<T>(size - 1);
    } else {
        return ((coord + T(1)) * static_cast<T>(size) - T(1)) * T(0.5);
    }
}

template <typename T>
__device__ __forceinline__ T gs_reflect_coord(T coord, int size, bool align_corners) {
    if (size <= 1) return T(0);
    // align_corners-aware reflection matching the CPU reflect_coord_impl
    // convention (the previous hardcoded span was only correct for
    // align_corners=true).
    T twice_low  = align_corners ? T(0) : T(-1);
    T twice_high = align_corners ? static_cast<T>(2 * (size - 1))
                                 : static_cast<T>(2 * size - 1);
    T mn = twice_low / T(2);
    T span = (twice_high - twice_low) / T(2);
    T c = fabs(coord - mn);
    T extra = fmod(c, span);
    long long flips = static_cast<long long>(floor(c / span));
    T reflected = ((flips % 2) == 0) ? (extra + mn) : (span - extra + mn);
    return fmin(fmax(reflected, T(0)), static_cast<T>(size - 1));
}

// d(gs_reflect_coord)/d(coord): +1 or -1 fold-sign (align_corners-aware).
template <typename T>
__device__ __forceinline__ T gs_reflect_coord_grad(T coord, int size, bool align_corners) {
    if (size <= 1) return T(0);
    T twice_low  = align_corners ? T(0) : T(-1);
    T twice_high = align_corners ? static_cast<T>(2 * (size - 1))
                                 : static_cast<T>(2 * size - 1);
    T mn = twice_low / T(2);
    T span = (twice_high - twice_low) / T(2);
    T d = coord - mn;
    T sign = (d < T(0)) ? T(-1) : T(1);
    long long flips = static_cast<long long>(floor(fabs(d) / span));
    if (flips % 2 != 0) sign = -sign;
    return sign;
}

template <typename T>
__device__ __forceinline__ int gs_floor_int(T v) {
    if constexpr (std::is_same_v<T, float>) return static_cast<int>(floorf(v));
    else                                    return static_cast<int>(floor(v));
}

template <typename T>
__device__ __forceinline__ T gs_clamp_coord(T v, int size) {
    if constexpr (std::is_same_v<T, float>) return fminf(fmaxf(v, 0.0f), static_cast<float>(size - 1));
    else                                    return fmin(fmax(v, 0.0), static_cast<double>(size - 1));
}

// cubic convolution weights (Catmull-Rom, a = -0.5)
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
__device__ inline void cubic_dweights_dev(T t, T dw[4]) {
    constexpr T a = T(-0.5);
    const T u = T(1) - t;
    dw[0] = (T(3) * a * t * t - T(4) * a * t + a);
    dw[1] = (T(3) * (a + T(2)) * t * t - T(2) * (a + T(3)) * t);
    dw[2] = -(T(3) * (a + T(2)) * u * u - T(2) * (a + T(3)) * u);
    dw[3] = -(T(3) * a * u * u - T(4) * a * u + a);
}

// =========================================================================
// Forward kernels
// =========================================================================

template <typename T>
__global__ void grid_sample_bilinear_kernel(
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
    T ix = gs_denormalize_dev<T>(grid[grid_idx], W_in, align_corners);
    T iy = gs_denormalize_dev<T>(grid[grid_idx + 1], H_in, align_corners);

    if (padding_mode == 1) {
        ix = gs_clamp_coord<T>(ix, W_in);
        iy = gs_clamp_coord<T>(iy, H_in);
    } else if (padding_mode == 2) {
        ix = gs_reflect_coord<T>(ix, W_in, align_corners);
        iy = gs_reflect_coord<T>(iy, H_in, align_corners);
    }

    int x0 = gs_floor_int<T>(ix);
    int y0 = gs_floor_int<T>(iy);
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
    T ix = gs_denormalize_dev<T>(grid[grid_idx], W_in, align_corners);
    T iy = gs_denormalize_dev<T>(grid[grid_idx + 1], H_in, align_corners);

    if (padding_mode == 1) {
        ix = gs_clamp_coord<T>(ix, W_in);
        iy = gs_clamp_coord<T>(iy, H_in);
    } else if (padding_mode == 2) {
        ix = gs_reflect_coord<T>(ix, W_in, align_corners);
        iy = gs_reflect_coord<T>(iy, H_in, align_corners);
    }

    // L5: rint/rintf = round-half-to-even, matching CPU std::nearbyint / CUDA
    // rint / PyTorch-ATen grid_sampler nearest (roundf/round are half-away-
    // from-zero and diverge from every other backend at exact .5 offsets,
    // e.g. an aligned grid landing exactly between two pixels).
    int nx, ny;
    if constexpr (std::is_same_v<T, float>) { nx = static_cast<int>(rintf(ix)); ny = static_cast<int>(rintf(iy)); }
    else                                    { nx = static_cast<int>(rint(ix));  ny = static_cast<int>(rint(iy)); }

    T val = T(0);
    if (ny >= 0 && ny < H_in && nx >= 0 && nx < W_in) {
        val = input[((n * C + c) * H_in + ny) * W_in + nx];
    }
    output[((n * C + c) * H_out + h) * W_out + w] = val;
}

template <typename T>
__global__ void grid_sample_bicubic_kernel(
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
    T ix = gs_denormalize_dev<T>(grid[grid_idx], W_in, align_corners);
    T iy = gs_denormalize_dev<T>(grid[grid_idx + 1], H_in, align_corners);

    if (padding_mode == 1) {
        ix = gs_clamp_coord<T>(ix, W_in);
        iy = gs_clamp_coord<T>(iy, H_in);
    } else if (padding_mode == 2) {
        ix = gs_reflect_coord<T>(ix, W_in, align_corners);
        iy = gs_reflect_coord<T>(iy, H_in, align_corners);
    }

    int ix_floor = gs_floor_int<T>(ix);
    int iy_floor = gs_floor_int<T>(iy);
    const T tx = ix - static_cast<T>(ix_floor);
    const T ty = iy - static_cast<T>(iy_floor);
    T wx[4], wy[4];
    cubic_weights_dev<T>(tx, wx);
    cubic_weights_dev<T>(ty, wy);

    auto safe_get = [&](int y, int x) -> T {
        if (padding_mode == 0) {
            if (y < 0 || y >= H_in || x < 0 || x >= W_in) return T(0);
            return input[((n * C + c) * H_in + y) * W_in + x];
        }
        if (H_in == 0 || W_in == 0) return T(0);
        if (padding_mode == 2) {
            // reflection: true-reflect this out-of-range 4x4 neighbour back
            // into [0, size-1] (PyTorch reflection semantics), matching
            // CPU's safe_get instead of edge-clamping.
            int ry = static_cast<int>(gs_reflect_coord<T>(static_cast<T>(y), H_in, align_corners));
            int rx = static_cast<int>(gs_reflect_coord<T>(static_cast<T>(x), W_in, align_corners));
            ry = max(0, min(ry, H_in - 1));
            rx = max(0, min(rx, W_in - 1));
            return input[((n * C + c) * H_in + ry) * W_in + rx];
        }
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

// =========================================================================
// Affine grid kernel (Float32 only — unchanged)
// =========================================================================

template <typename T>
__global__ void affine_grid_kernel(
    const T* __restrict__ theta,
    T* __restrict__ grid,
    int N, int H, int W, bool align_corners
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int n = idx / (H * W);

    T x_norm, y_norm;
    if (align_corners) {
        x_norm = (W > 1) ? (T(2) * static_cast<T>(w) / static_cast<T>(W - 1) - T(1)) : T(0);
        y_norm = (H > 1) ? (T(2) * static_cast<T>(h) / static_cast<T>(H - 1) - T(1)) : T(0);
    } else {
        x_norm = (T(2) * static_cast<T>(w) + T(1)) / static_cast<T>(W) - T(1);
        y_norm = (T(2) * static_cast<T>(h) + T(1)) / static_cast<T>(H) - T(1);
    }

    const T* t = theta + n * 6;
    T x_out = t[0] * x_norm + t[1] * y_norm + t[2];
    T y_out = t[3] * x_norm + t[4] * y_norm + t[5];

    int out_idx = ((n * H + h) * W + w) * 2;
    grid[out_idx] = x_out;
    grid[out_idx + 1] = y_out;
}

// =========================================================================
// Forward host API
// =========================================================================

namespace {
template <typename T>
void launch_grid_sample_forward(const std::string& mode, int grid_size, int block_size,
                                hipStream_t stream,
                                const T* in, const T* grid, T* out,
                                int N, int C, int H_in, int W_in, int H_out, int W_out,
                                int pad_mode, bool align_corners) {
    if (mode == "nearest") {
        hipLaunchKernelGGL(grid_sample_nearest_kernel<T>, dim3(grid_size), dim3(block_size), 0, stream,
            in, grid, out, N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
    } else if (mode == "bilinear") {
        hipLaunchKernelGGL(grid_sample_bilinear_kernel<T>, dim3(grid_size), dim3(block_size), 0, stream,
            in, grid, out, N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
    } else if (mode == "bicubic") {
        hipLaunchKernelGGL(grid_sample_bicubic_kernel<T>, dim3(grid_size), dim3(block_size), 0, stream,
            in, grid, out, N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
    } else {
        throw std::invalid_argument(
            "grid_sample (ROCm): unknown mode '" + mode +
            "'. Supported: bilinear, nearest, bicubic.");
    }
}
}  // namespace

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

    int pad_mode = 0;
    if (padding_mode == "border") pad_mode = 1;
    else if (padding_mode == "reflection") pad_mode = 2;

    int total = N * C * H_out * W_out;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    // Native FP64 path for all modes (no widen-narrow).
    if (input.dtype() == DType::Float64) {
        Tensor input_f64 = input.contiguous();
        Tensor grid_f64  = grid.to(DType::Float64).contiguous();
        Tensor output_f64({N, C, H_out, W_out}, DType::Float64, input.device());
        launch_grid_sample_forward<double>(mode, grid_size, block_size, stream,
            input_f64.data<double>(), grid_f64.data<double>(), output_f64.data<double>(),
            N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
        HIP_CHECK(hipGetLastError());
        return output_f64;
    }

    // Float32 path (Float16/BFloat16 promote losslessly per element).
    // .contiguous() is required: Tensor::to(dtype) returns *this unchanged when
    // already Float32, so a permuted/sliced Float32 input/grid would be read
    // with dense NCHW offsets and pick the wrong elements.
    Tensor input_f32 = input.to(DType::Float32).contiguous();
    Tensor grid_f32  = grid.to(DType::Float32).contiguous();
    Tensor output_f32({N, C, H_out, W_out}, DType::Float32, input.device());
    launch_grid_sample_forward<float>(mode, grid_size, block_size, stream,
        input_f32.data<float>(), grid_f32.data<float>(), output_f32.data<float>(),
        N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
    HIP_CHECK(hipGetLastError());
    return output_f32.to(input.dtype());
}

auto affine_grid_kernel_host(const Tensor& theta, const std::vector<int64_t>& size,
                              bool align_corners, hipStream_t stream) -> Tensor {
    int N = static_cast<int>(size[0]);
    int H = static_cast<int>(size[2]);
    int W = static_cast<int>(size[3]);

    // Native double compute for Float64 theta (matches CUDA/OneAPI/CPU and this
    // file's OWN affine_grid_backward_kernel_dev<T>, which already computes
    // natively in double) -- the previous unconditional .to(Float32) computed
    // in single precision and only restored the OUTPUT dtype label afterward,
    // silently losing precision instead of preserving it end-to-end. Float16/
    // BFloat16 widen to Float32, compute, narrow back (the kernel has no
    // half-precision instantiation).
    if (theta.dtype() == DType::Float16 || theta.dtype() == DType::BFloat16) {
        DType orig = theta.dtype();
        Tensor grid_f32 = affine_grid_kernel_host(theta.to(DType::Float32), size, align_corners, stream);
        return grid_f32.to(orig);
    }

    // .contiguous(): the kernel reads theta with dense strides, so a non-contiguous
    // theta view would be read at the wrong offsets (matches grid_sample_kernel).
    Tensor theta_c = theta.is_contiguous() ? theta : theta.contiguous();
    Tensor grid({N, H, W, 2}, theta.dtype(), theta.device());

    int total = N * H * W;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    if (theta.dtype() == DType::Float64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(affine_grid_kernel<double>),
            dim3(grid_size), dim3(block_size), 0, stream,
            theta_c.data<double>(), grid.data<double>(),
            N, H, W, align_corners);
    } else if (theta.dtype() == DType::Float32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(affine_grid_kernel<float>),
            dim3(grid_size), dim3(block_size), 0, stream,
            theta_c.data<float>(), grid.data<float>(),
            N, H, W, align_corners);
    } else {
        throw std::runtime_error("affine_grid (ROCm): unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return grid;
}

// =========================================================================
// Backward kernels (templated FP32/FP64)
// =========================================================================

template <typename T>
__global__ void grid_sample_bilinear_backward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ grid,
    const T* __restrict__ grad_output,
    T* __restrict__ grad_input,
    T* __restrict__ grad_grid,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int padding_mode, bool align_corners)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W_out;
    if (idx >= total) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int n = idx / (H_out * W_out);

    int grid_idx = ((n * H_out + h) * W_out + w) * 2;
    T ix = gs_denormalize_dev<T>(grid[grid_idx], W_in, align_corners);
    T iy = gs_denormalize_dev<T>(grid[grid_idx + 1], H_in, align_corners);

    bool in_bounds_ix = (ix >= T(0) && ix <= static_cast<T>(W_in - 1));
    bool in_bounds_iy = (iy >= T(0) && iy <= static_cast<T>(H_in - 1));

    // Reflection contributes a ±1 fold-sign to the coordinate gradient; capture
    // it from the PRE-reflection coordinate before ix/iy are overwritten.
    T refl_sign_x = T(1), refl_sign_y = T(1);
    if (padding_mode == 1) {
        ix = gs_clamp_coord<T>(ix, W_in);
        iy = gs_clamp_coord<T>(iy, H_in);
    } else if (padding_mode == 2) {
        refl_sign_x = gs_reflect_coord_grad<T>(ix, W_in, align_corners);
        refl_sign_y = gs_reflect_coord_grad<T>(iy, H_in, align_corners);
        ix = gs_reflect_coord<T>(ix, W_in, align_corners);
        iy = gs_reflect_coord<T>(iy, H_in, align_corners);
    }

    T dix_dgx, diy_dgy;
    if (align_corners) {
        dix_dgx = T(0.5) * static_cast<T>(W_in - 1);
        diy_dgy = T(0.5) * static_cast<T>(H_in - 1);
    } else {
        dix_dgx = T(0.5) * static_cast<T>(W_in);
        diy_dgy = T(0.5) * static_cast<T>(H_in);
    }
    dix_dgx *= refl_sign_x;
    diy_dgy *= refl_sign_y;

    int x0 = gs_floor_int<T>(ix);
    int y0 = gs_floor_int<T>(iy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    T wx1 = ix - static_cast<T>(x0);
    T wy1 = iy - static_cast<T>(y0);
    T wx0 = T(1) - wx1;
    T wy0 = T(1) - wy1;

    T sum_dx = T(0), sum_dy = T(0);
    for (int c = 0; c < C; ++c) {
        const T go = grad_output[((n * C + c) * H_out + h) * W_out + w];
        T* ch_gi = grad_input + (n * C + c) * H_in * W_in;
        const T* ch_in = input + (n * C + c) * H_in * W_in;

        auto safe_scatter = [&](int y, int x, T weight) {
            if (y >= 0 && y < H_in && x >= 0 && x < W_in) {
                atomicAdd(&ch_gi[y * W_in + x], go * weight);
            }
        };
        auto safe_get = [&](int y, int x) -> T {
            if (y >= 0 && y < H_in && x >= 0 && x < W_in) return ch_in[y * W_in + x];
            return T(0);
        };
        safe_scatter(y0, x0, wy0 * wx0);
        safe_scatter(y0, x1, wy0 * wx1);
        safe_scatter(y1, x0, wy1 * wx0);
        safe_scatter(y1, x1, wy1 * wx1);

        sum_dx += go * (wy0 * (-safe_get(y0, x0) + safe_get(y0, x1)) +
                        wy1 * (-safe_get(y1, x0) + safe_get(y1, x1)));
        sum_dy += go * (wx0 * (-safe_get(y0, x0) + safe_get(y1, x0)) +
                        wx1 * (-safe_get(y0, x1) + safe_get(y1, x1)));
    }
    // H9 (was mislabeled F069): zero grad_grid where the pre-clamp coord is
    // out of range ONLY for border (1) padding — under border the clamp
    // derivative is 0 there, so a clamped sample must not produce a spurious
    // grad_grid. 'zeros' must NOT be gated: safe_get()/safe_scatter() above
    // already return/skip 0 for out-of-bounds bilinear corners, so sum_dx/
    // sum_dy already contain only in-bounds contributions, and the bands
    // ix in [-1,0) / (W-1,W) (likewise iy) have exactly one in-bounds
    // neighbour with a genuine non-zero gradient that a 'zeros' gate here
    // would incorrectly drop to 0 (matches CPU/CUDA).
    bool oob_gate = (padding_mode == 1);
    T scale_x = (oob_gate && !in_bounds_ix) ? T(0) : dix_dgx;
    T scale_y = (oob_gate && !in_bounds_iy) ? T(0) : diy_dgy;
    grad_grid[grid_idx]     = sum_dx * scale_x;
    grad_grid[grid_idx + 1] = sum_dy * scale_y;
}

template <typename T>
__global__ void grid_sample_nearest_backward_kernel(
    const T* __restrict__ /*input*/,
    const T* __restrict__ grid,
    const T* __restrict__ grad_output,
    T* __restrict__ grad_input,
    T* __restrict__ grad_grid,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int padding_mode, bool align_corners)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W_out;
    if (idx >= total) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int n = idx / (H_out * W_out);

    int grid_idx = ((n * H_out + h) * W_out + w) * 2;
    T ix = gs_denormalize_dev<T>(grid[grid_idx], W_in, align_corners);
    T iy = gs_denormalize_dev<T>(grid[grid_idx + 1], H_in, align_corners);

    if (padding_mode == 1) {
        ix = gs_clamp_coord<T>(ix, W_in);
        iy = gs_clamp_coord<T>(iy, H_in);
    } else if (padding_mode == 2) {
        ix = gs_reflect_coord<T>(ix, W_in, align_corners);
        iy = gs_reflect_coord<T>(iy, H_in, align_corners);
    }

    // L5: rint/rintf = round-half-to-even, matching CPU std::nearbyint / CUDA
    // rint / PyTorch-ATen grid_sampler nearest (roundf/round are half-away-
    // from-zero and diverge from every other backend at exact .5 offsets,
    // e.g. an aligned grid landing exactly between two pixels).
    int nx, ny;
    if constexpr (std::is_same_v<T, float>) { nx = static_cast<int>(rintf(ix)); ny = static_cast<int>(rintf(iy)); }
    else                                    { nx = static_cast<int>(rint(ix));  ny = static_cast<int>(rint(iy)); }

    grad_grid[grid_idx]     = T(0);
    grad_grid[grid_idx + 1] = T(0);

    if (ny >= 0 && ny < H_in && nx >= 0 && nx < W_in) {
        for (int c = 0; c < C; ++c) {
            const T go = grad_output[((n * C + c) * H_out + h) * W_out + w];
            atomicAdd(&grad_input[((n * C + c) * H_in + ny) * W_in + nx], go);
        }
    }
}

template <typename T>
__global__ void grid_sample_bicubic_backward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ grid,
    const T* __restrict__ grad_output,
    T* __restrict__ grad_input,
    T* __restrict__ grad_grid,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int padding_mode, bool align_corners)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W_out;
    if (idx >= total) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int n = idx / (H_out * W_out);

    int grid_idx = ((n * H_out + h) * W_out + w) * 2;
    T ix = gs_denormalize_dev<T>(grid[grid_idx], W_in, align_corners);
    T iy = gs_denormalize_dev<T>(grid[grid_idx + 1], H_in, align_corners);

    // F069: capture pre-clamp in-bounds BEFORE the border clamp overwrites ix/iy,
    // so a border-clamped sample gets zero grad_grid (clamp derivative is 0).
    bool in_bounds_ix = (ix >= T(0) && ix <= static_cast<T>(W_in - 1));
    bool in_bounds_iy = (iy >= T(0) && iy <= static_cast<T>(H_in - 1));

    // Reflection contributes a ±1 fold-sign to the coordinate gradient; capture
    // it from the PRE-reflection coordinate before ix/iy are overwritten.
    T refl_sign_x = T(1), refl_sign_y = T(1);
    if (padding_mode == 1) {
        ix = gs_clamp_coord<T>(ix, W_in);
        iy = gs_clamp_coord<T>(iy, H_in);
    } else if (padding_mode == 2) {
        refl_sign_x = gs_reflect_coord_grad<T>(ix, W_in, align_corners);
        refl_sign_y = gs_reflect_coord_grad<T>(iy, H_in, align_corners);
        ix = gs_reflect_coord<T>(ix, W_in, align_corners);
        iy = gs_reflect_coord<T>(iy, H_in, align_corners);
    }

    T dix_dgx, diy_dgy;
    if (align_corners) {
        dix_dgx = T(0.5) * static_cast<T>(W_in - 1);
        diy_dgy = T(0.5) * static_cast<T>(H_in - 1);
    } else {
        dix_dgx = T(0.5) * static_cast<T>(W_in);
        diy_dgy = T(0.5) * static_cast<T>(H_in);
    }
    dix_dgx *= refl_sign_x;
    diy_dgy *= refl_sign_y;

    int ix_floor = gs_floor_int<T>(ix);
    int iy_floor = gs_floor_int<T>(iy);
    T tx = ix - static_cast<T>(ix_floor);
    T ty = iy - static_cast<T>(iy_floor);
    T wx[4], wy[4], dwx[4], dwy[4];
    cubic_weights_dev<T>(tx, wx);
    cubic_weights_dev<T>(ty, wy);
    cubic_dweights_dev<T>(tx, dwx);
    cubic_dweights_dev<T>(ty, dwy);

    T sum_dx = T(0), sum_dy = T(0);
    for (int c = 0; c < C; ++c) {
        const T go = grad_output[((n * C + c) * H_out + h) * W_out + w];
        T dval_dix = T(0), dval_diy = T(0);
        #pragma unroll
        for (int dy = -1; dy <= 2; ++dy) {
            #pragma unroll
            for (int dx = -1; dx <= 2; ++dx) {
                const int yy = iy_floor + dy;
                const int xx = ix_floor + dx;
                const T weight = wy[dy + 1] * wx[dx + 1];

                int yy_s = yy, xx_s = xx;
                bool valid_s = true;
                if (padding_mode == 0) {
                    if (yy < 0 || yy >= H_in || xx < 0 || xx >= W_in) valid_s = false;
                } else if (padding_mode == 2) {
                    // reflection: true-reflect (matches CPU backward
                    // scatter), not edge-clamp.
                    yy_s = static_cast<int>(gs_reflect_coord<T>(static_cast<T>(yy), H_in, align_corners));
                    xx_s = static_cast<int>(gs_reflect_coord<T>(static_cast<T>(xx), W_in, align_corners));
                    yy_s = max(0, min(yy_s, H_in - 1));
                    xx_s = max(0, min(xx_s, W_in - 1));
                } else {
                    yy_s = max(0, min(yy, H_in - 1));
                    xx_s = max(0, min(xx, W_in - 1));
                }
                if (valid_s) {
                    atomicAdd(&grad_input[((n * C + c) * H_in + yy_s) * W_in + xx_s], go * weight);
                }

                T v = T(0);
                if (padding_mode == 0) {
                    if (yy >= 0 && yy < H_in && xx >= 0 && xx < W_in) {
                        v = input[((n * C + c) * H_in + yy) * W_in + xx];
                    }
                } else if (padding_mode == 2) {
                    int yy_f = static_cast<int>(gs_reflect_coord<T>(static_cast<T>(yy), H_in, align_corners));
                    int xx_f = static_cast<int>(gs_reflect_coord<T>(static_cast<T>(xx), W_in, align_corners));
                    yy_f = max(0, min(yy_f, H_in - 1));
                    xx_f = max(0, min(xx_f, W_in - 1));
                    v = input[((n * C + c) * H_in + yy_f) * W_in + xx_f];
                } else {
                    int yy_f = max(0, min(yy, H_in - 1));
                    int xx_f = max(0, min(xx, W_in - 1));
                    v = input[((n * C + c) * H_in + yy_f) * W_in + xx_f];
                }
                dval_dix += wy[dy + 1] * dwx[dx + 1] * v;
                dval_diy += dwy[dy + 1] * wx[dx + 1] * v;
            }
        }
        sum_dx += go * dval_dix;
        sum_dy += go * dval_diy;
    }

    // F069: under border padding a clamped coord has zero clamp derivative, so
    // zero grad_grid (zeros padding is already handled by the per-corner v=0
    // checks above, so it needs no whole-point gate here).
    T scale_x = (padding_mode == 1 && !in_bounds_ix) ? T(0) : dix_dgx;
    T scale_y = (padding_mode == 1 && !in_bounds_iy) ? T(0) : diy_dgy;
    grad_grid[grid_idx]     = sum_dx * scale_x;
    grad_grid[grid_idx + 1] = sum_dy * scale_y;
}

template <typename T>
__global__ void affine_grid_backward_kernel_dev(
    const T* __restrict__ grad_grid,
    T* __restrict__ grad_theta,
    int N, int H, int W, bool align_corners)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int n = idx / (H * W);

    T x_norm, y_norm;
    if (align_corners) {
        x_norm = (W > 1) ? (T(2) * static_cast<T>(w) / static_cast<T>(W - 1) - T(1)) : T(0);
        y_norm = (H > 1) ? (T(2) * static_cast<T>(h) / static_cast<T>(H - 1) - T(1)) : T(0);
    } else {
        x_norm = (T(2) * static_cast<T>(w) + T(1)) / static_cast<T>(W) - T(1);
        y_norm = (T(2) * static_cast<T>(h) + T(1)) / static_cast<T>(H) - T(1);
    }

    int gg_idx = ((n * H + h) * W + w) * 2;
    T dg_x = grad_grid[gg_idx];
    T dg_y = grad_grid[gg_idx + 1];

    T* t = grad_theta + n * 6;
    atomicAdd(&t[0], dg_x * x_norm);
    atomicAdd(&t[1], dg_x * y_norm);
    atomicAdd(&t[2], dg_x);
    atomicAdd(&t[3], dg_y * x_norm);
    atomicAdd(&t[4], dg_y * y_norm);
    atomicAdd(&t[5], dg_y);
}

namespace {
template <typename T>
void launch_grid_sample_backward(const std::string& mode, int gs, int block_size,
                                 hipStream_t stream,
                                 const T* in, const T* grid, const T* go,
                                 T* gi, T* gg,
                                 int N, int C, int H_in, int W_in, int H_out, int W_out,
                                 int pad_mode, bool align_corners) {
    if (mode == "bilinear") {
        hipLaunchKernelGGL(grid_sample_bilinear_backward_kernel<T>, dim3(gs), dim3(block_size), 0, stream,
            in, grid, go, gi, gg, N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
    } else if (mode == "nearest") {
        hipLaunchKernelGGL(grid_sample_nearest_backward_kernel<T>, dim3(gs), dim3(block_size), 0, stream,
            in, grid, go, gi, gg, N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
    } else if (mode == "bicubic") {
        hipLaunchKernelGGL(grid_sample_bicubic_backward_kernel<T>, dim3(gs), dim3(block_size), 0, stream,
            in, grid, go, gi, gg, N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
    } else {
        throw std::invalid_argument(
            "grid_sample_backward (ROCm): unknown mode '" + mode + "'");
    }
}
}  // namespace

auto grid_sample_backward_kernel_host(const Tensor& grad_output,
                                      const Tensor& input, const Tensor& grid,
                                      const std::string& mode,
                                      const std::string& padding_mode,
                                      bool align_corners, hipStream_t stream)
    -> std::pair<Tensor, Tensor>
{
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

    int total = N * H_out * W_out;
    int block_size = 256;
    int gs = (total + block_size - 1) / block_size;

    // Native FP64 path when EITHER input or grid is Float64 (matches CUDA/CPU),
    // so grad_grid is computed in double whenever the grid is double.
    if (input.dtype() == DType::Float64 || grid.dtype() == DType::Float64) {
        Tensor input_f64 = input.to(DType::Float64).contiguous();
        Tensor grid_f64  = grid.to(DType::Float64).contiguous();
        Tensor go_f64    = grad_output.to(DType::Float64).contiguous();
        Tensor gi_f64({N, C, H_in, W_in},  DType::Float64, input.device());
        Tensor gg_f64({N, H_out, W_out, 2}, DType::Float64, grid.device());
        HIP_CHECK(hipMemsetAsync(gi_f64.data_ptr(), 0, gi_f64.numel() * sizeof(double), stream));
        launch_grid_sample_backward<double>(mode, gs, block_size, stream,
            input_f64.data<double>(), grid_f64.data<double>(), go_f64.data<double>(),
            gi_f64.data<double>(), gg_f64.data<double>(),
            N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
        HIP_CHECK(hipGetLastError());
        return {gi_f64.to(input.dtype()), gg_f64.to(grid.dtype())};
    }

    DType in_dt = input.dtype();
    DType gr_dt = grid.dtype();
    // .contiguous() is required: to(Float32) returns *this when already Float32,
    // so a non-contiguous input/grid/grad_output would be misread by the dense
    // NCHW-indexed kernel (matches the FP64 branch above).
    Tensor input_f32 = input.to(DType::Float32).contiguous();
    Tensor grid_f32  = grid.to(DType::Float32).contiguous();
    Tensor go_f32    = grad_output.to(DType::Float32).contiguous();
    Tensor gi_f32({N, C, H_in, W_in},  DType::Float32, input.device());
    Tensor gg_f32({N, H_out, W_out, 2}, DType::Float32, grid.device());
    HIP_CHECK(hipMemsetAsync(gi_f32.data_ptr(), 0, gi_f32.numel() * sizeof(float), stream));

    launch_grid_sample_backward<float>(mode, gs, block_size, stream,
        input_f32.data<float>(), grid_f32.data<float>(), go_f32.data<float>(),
        gi_f32.data<float>(), gg_f32.data<float>(),
        N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
    HIP_CHECK(hipGetLastError());
    return {gi_f32.to(in_dt), gg_f32.to(gr_dt)};
}

auto affine_grid_backward_kernel_host(const Tensor& grad_grid,
                                      const std::vector<int64_t>& size,
                                      bool align_corners, hipStream_t stream) -> Tensor
{
    int N = static_cast<int>(size[0]);
    int H = static_cast<int>(size[2]);
    int W = static_cast<int>(size[3]);

    DType gr_dt = grad_grid.dtype();
    // M9: native FP64 compute path (matches CUDA sibling's affine_grid_backward_cuda),
    // instead of always downcasting to Float32 and losing a Float64 grad_grid's precision.
    DType compute = (gr_dt == DType::Float64) ? DType::Float64 : DType::Float32;

    Tensor gg_c = grad_grid.to(compute).contiguous();
    Tensor gt_c({N, 2, 3}, compute, grad_grid.device());
    HIP_CHECK(hipMemsetAsync(gt_c.data_ptr(), 0,
        gt_c.numel() * dtype_size(compute), stream));

    int total = N * H * W;
    int block_size = 256;
    int gs = (total + block_size - 1) / block_size;

    if (compute == DType::Float64) {
        hipLaunchKernelGGL(affine_grid_backward_kernel_dev<double>,
            dim3(gs), dim3(block_size), 0, stream,
            gg_c.data<double>(), gt_c.data<double>(),
            N, H, W, align_corners);
    } else {
        hipLaunchKernelGGL(affine_grid_backward_kernel_dev<float>,
            dim3(gs), dim3(block_size), 0, stream,
            gg_c.data<float>(), gt_c.data<float>(),
            N, H, W, align_corners);
    }
    HIP_CHECK(hipGetLastError());
    return gt_c.to(gr_dt);
}

}  // namespace rocm
}  // namespace tenzor
