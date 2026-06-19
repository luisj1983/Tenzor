/**
 * @file grid_sample.cpp
 * @brief CPU kernel implementations for grid_sample and affine_grid operations.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <type_traits>
#include "tenzor/backend/omp_thresholds.hpp"

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace tenzor {
namespace cpu {

namespace {

// Denormalize grid coordinate from [-1, 1] to [0, size-1]
template <typename T>
inline T denormalize(T coord, int64_t size, bool align_corners) {
    if (align_corners) {
        // [-1, 1] -> [0, size-1]
        return (coord + T(1)) * T(0.5) * static_cast<T>(size - 1);
    } else {
        // [-1, 1] -> [-0.5, size-0.5]
        return ((coord + T(1)) * static_cast<T>(size) - T(1)) * T(0.5);
    }
}

// Reflect a coordinate into the valid range using PyTorch's twice_low/twice_high
// convention. align_corners=true reflects about [0, size-1]; align_corners=false
// reflects about [-0.5, size-0.5] (period differs), matching reflect_coordinates.
template <typename T>
inline T reflect_coord_impl(T coord, int64_t size, bool align_corners) {
    if (size <= 1) return T(0);
    T twice_low  = align_corners ? T(0) : T(-1);
    T twice_high = align_corners ? static_cast<T>(2 * (size - 1))
                                 : static_cast<T>(2 * size - 1);
    T mn = twice_low / T(2);
    T span = (twice_high - twice_low) / T(2);
    T c = std::fabs(coord - mn);
    T extra = std::fmod(c, span);
    int64_t flips = static_cast<int64_t>(std::floor(c / span));
    T reflected = (flips % 2 == 0) ? (extra + mn) : (span - extra + mn);
    // Clip to the sampleable range to guard FP edge cases.
    return std::clamp(reflected, T(0), static_cast<T>(size - 1));
}

// Apply padding mode to compute effective coordinate
template <typename T>
inline T apply_padding(T coord, int64_t size, const std::string& padding_mode, bool align_corners) {
    if (padding_mode == "border") {
        coord = std::clamp(coord, T(0), static_cast<T>(size - 1));
    } else if (padding_mode == "reflection") {
        coord = reflect_coord_impl<T>(coord, size, align_corners);
    }
    // "zeros" mode: out-of-bound coordinates will use zero values (handled in sampling)
    return coord;
}

template <typename T>
inline bool is_in_bounds(T y, T x, int64_t H, int64_t W) {
    return y >= 0 && y < H && x >= 0 && x < W;
}

// Catmull-Rom cubic interpolation weights with a = -0.5. Matches the
// `cubic_interp1d` helper used by interpolate(mode='bicubic') in vision.cpp.
// Returns the four weights w[-1..2] that combine with the 4 neighbour values.
template <typename T>
inline void cubic_weights(T t, T w[4]) {
    constexpr T a = T(-0.5);
    const T t2 = t * t;
    const T t3 = t2 * t;
    w[0] = ((a * t - T(2) * a) * t + a) * t;                  // |x| = 1+t
    w[1] = ((a + T(2)) * t3 - (a + T(3)) * t2 + T(1));        // |x| = t
    w[2] = ((a + T(2)) * (T(1) - t) * (T(1) - t) * (T(1) - t)
           - (a + T(3)) * (T(1) - t) * (T(1) - t) + T(1));    // |x| = 1-t
    w[3] = ((a * (T(1) - t) - T(2) * a) * (T(1) - t) + a) * (T(1) - t); // |x| = 2-t
}

// Forward grid_sample computed natively in type T (float or double). The AVX2
// fast path is float-only and selected with `if constexpr`; the double
// instantiation uses the scalar path, preserving full Float64 precision.
template <typename T>
auto grid_sample_forward_impl(const Tensor& input, const Tensor& grid,
                              const std::string& mode,
                              const std::string& padding_mode,
                              bool align_corners) -> Tensor {
    auto in_shape = input.shape();
    auto grid_shape = grid.shape();

    int64_t N = in_shape[0];
    int64_t C = in_shape[1];
    int64_t H_in = in_shape[2];
    int64_t W_in = in_shape[3];
    int64_t H_out = grid_shape[1];
    int64_t W_out = grid_shape[2];

    const T* in_data = input.data<T>();
    const T* grid_data = grid.data<T>();

    constexpr DType out_dt = std::is_same_v<T, double> ? DType::Float64 : DType::Float32;
    Tensor output({N, C, H_out, W_out}, out_dt, input.device());
    T* out_data = output.data<T>();

    int64_t spatial_size = H_in * W_in;

    #pragma omp parallel for collapse(2) if(N * H_out * W_out > ::tenzor::OmpThresholds::complex())
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t hw = 0; hw < H_out * W_out; ++hw) {
            int64_t h = hw / W_out;
            int64_t w = hw % W_out;

            // Grid coordinates: (x, y) at grid[n, h, w, :]
            int64_t grid_idx = ((n * H_out + h) * W_out + w) * 2;
            T gx = grid_data[grid_idx];
            T gy = grid_data[grid_idx + 1];

            T ix = denormalize<T>(gx, W_in, align_corners);
            T iy = denormalize<T>(gy, H_in, align_corners);

            if (mode == "bilinear") {
                T px = apply_padding<T>(ix, W_in, padding_mode, align_corners);
                T py = apply_padding<T>(iy, H_in, padding_mode, align_corners);

                int64_t x0 = static_cast<int64_t>(std::floor(px));
                int64_t y0 = static_cast<int64_t>(std::floor(py));
                int64_t x1 = x0 + 1;
                int64_t y1 = y0 + 1;

                T wx1 = px - static_cast<T>(x0);
                T wy1 = py - static_cast<T>(y0);
                T wx0 = T(1) - wx1;
                T wy0 = T(1) - wy1;

                // Compute the four corner weights once, reuse across all channels
                T w00 = wy0 * wx0;
                T w01 = wy0 * wx1;
                T w10 = wy1 * wx0;
                T w11 = wy1 * wx1;

                // Bounds check (shared across all channels)
                bool b00 = (y0 >= 0 && y0 < H_in && x0 >= 0 && x0 < W_in);
                bool b01 = (y0 >= 0 && y0 < H_in && x1 >= 0 && x1 < W_in);
                bool b10 = (y1 >= 0 && y1 < H_in && x0 >= 0 && x0 < W_in);
                bool b11 = (y1 >= 0 && y1 < H_in && x1 >= 0 && x1 < W_in);

                // Pixel offsets within a single channel plane (stride = spatial_size per channel)
                int64_t off00 = y0 * W_in + x0;
                int64_t off01 = y0 * W_in + x1;
                int64_t off10 = y1 * W_in + x0;
                int64_t off11 = y1 * W_in + x1;

                int64_t c = 0;

#if defined(__AVX2__)
                if constexpr (std::is_same_v<T, float>) {
                    // Vectorize across channels: process 8 channels at a time
                    __m256 vw00 = _mm256_set1_ps(w00);
                    __m256 vw01 = _mm256_set1_ps(w01);
                    __m256 vw10 = _mm256_set1_ps(w10);
                    __m256 vw11 = _mm256_set1_ps(w11);
                    __m256 vzero = _mm256_setzero_ps();

                    for (; c + 8 <= C; c += 8) {
                        const float* base = in_data + n * C * spatial_size + c * spatial_size;
                        float* out_base = out_data + n * C * H_out * W_out + c * H_out * W_out;

                        __m256 v00 = vzero, v01 = vzero, v10 = vzero, v11 = vzero;
                        if (b00) {
                            alignas(32) float tmp[8];
                            for (int k = 0; k < 8; ++k) tmp[k] = base[k * spatial_size + off00];
                            v00 = _mm256_load_ps(tmp);
                        }
                        if (b01) {
                            alignas(32) float tmp[8];
                            for (int k = 0; k < 8; ++k) tmp[k] = base[k * spatial_size + off01];
                            v01 = _mm256_load_ps(tmp);
                        }
                        if (b10) {
                            alignas(32) float tmp[8];
                            for (int k = 0; k < 8; ++k) tmp[k] = base[k * spatial_size + off10];
                            v10 = _mm256_load_ps(tmp);
                        }
                        if (b11) {
                            alignas(32) float tmp[8];
                            for (int k = 0; k < 8; ++k) tmp[k] = base[k * spatial_size + off11];
                            v11 = _mm256_load_ps(tmp);
                        }

                        __m256 result = _mm256_add_ps(
                            _mm256_add_ps(_mm256_mul_ps(vw00, v00), _mm256_mul_ps(vw01, v01)),
                            _mm256_add_ps(_mm256_mul_ps(vw10, v10), _mm256_mul_ps(vw11, v11)));

                        alignas(32) float out_tmp[8];
                        _mm256_store_ps(out_tmp, result);
                        int64_t out_hw = h * W_out + w;
                        for (int k = 0; k < 8; ++k) {
                            out_base[k * H_out * W_out + out_hw] = out_tmp[k];
                        }
                    }
                }
#endif
                // Scalar tail for remaining channels (all channels when T == double)
                for (; c < C; ++c) {
                    T val = T(0);
                    const T* ch_data = in_data + (n * C + c) * spatial_size;
                    if (b00) val += w00 * ch_data[off00];
                    if (b01) val += w01 * ch_data[off01];
                    if (b10) val += w10 * ch_data[off10];
                    if (b11) val += w11 * ch_data[off11];
                    out_data[((n * C + c) * H_out + h) * W_out + w] = val;
                }
            } else if (mode == "nearest") {
                int64_t nx = static_cast<int64_t>(std::round(
                    apply_padding<T>(ix, W_in, padding_mode, align_corners)));
                int64_t ny = static_cast<int64_t>(std::round(
                    apply_padding<T>(iy, H_in, padding_mode, align_corners)));

                bool in_bounds = is_in_bounds<T>(static_cast<T>(ny), static_cast<T>(nx), H_in, W_in);
                for (int64_t c = 0; c < C; ++c) {
                    T val = T(0);
                    if (in_bounds) {
                        val = in_data[((n * C + c) * H_in + ny) * W_in + nx];
                    }
                    out_data[((n * C + c) * H_out + h) * W_out + w] = val;
                }
            } else if (mode == "bicubic") {
                // Real bicubic interpolation: 4x4 neighbourhood with Catmull-Rom
                // (a = -0.5) basis, matching PyTorch and the existing
                // interpolate(mode='bicubic') kernel in vision.cpp. Padding is
                // applied per-neighbour so all three modes work as in bilinear.
                const T ix_pad = apply_padding<T>(ix, W_in, padding_mode, align_corners);
                const T iy_pad = apply_padding<T>(iy, H_in, padding_mode, align_corners);
                const int64_t ix_floor = static_cast<int64_t>(std::floor(ix_pad));
                const int64_t iy_floor = static_cast<int64_t>(std::floor(iy_pad));
                const T tx = ix_pad - static_cast<T>(ix_floor);
                const T ty = iy_pad - static_cast<T>(iy_floor);
                T wx[4], wy[4];
                cubic_weights<T>(tx, wx);
                cubic_weights<T>(ty, wy);

                for (int64_t c = 0; c < C; ++c) {
                    const T* ch_data = in_data + (n * C + c) * spatial_size;
                    auto safe_get = [&](int64_t y, int64_t x) -> T {
                        if (padding_mode == "zeros") {
                            if (y < 0 || y >= H_in || x < 0 || x >= W_in) return T(0);
                            return ch_data[y * W_in + x];
                        }
                        y = std::clamp(y, int64_t(0), H_in - 1);
                        x = std::clamp(x, int64_t(0), W_in - 1);
                        return ch_data[y * W_in + x];
                    };
                    T val = T(0);
                    for (int dy = -1; dy <= 2; ++dy) {
                        for (int dx = -1; dx <= 2; ++dx) {
                            val += wy[dy + 1] * wx[dx + 1] *
                                   safe_get(iy_floor + dy, ix_floor + dx);
                        }
                    }
                    out_data[((n * C + c) * H_out + h) * W_out + w] = val;
                }
            } else {
                // Unknown mode — fail loud rather than silently falling back.
                throw std::invalid_argument(
                    "grid_sample: unknown mode '" + mode +
                    "'. Supported: bilinear, nearest, bicubic.");
            }
        }
    }

    return output;
}

// affine_grid computed natively in type T.
template <typename T>
auto affine_grid_forward_impl(const Tensor& theta,
                              const std::vector<int64_t>& size,
                              bool align_corners) -> Tensor {
    int64_t N = size[0];
    int64_t H = size[2];
    int64_t W = size[3];

    const T* t_data = theta.data<T>();

    constexpr DType out_dt = std::is_same_v<T, double> ? DType::Float64 : DType::Float32;
    Tensor grid({N, H, W, 2}, out_dt, theta.device());
    T* g_data = grid.data<T>();

    for (int64_t n = 0; n < N; ++n) {
        const T* t = t_data + n * 6;  // 2x3 matrix flattened

        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                T y_norm, x_norm;
                if (align_corners) {
                    x_norm = (W > 1) ? (T(2) * static_cast<T>(w) / static_cast<T>(W - 1) - T(1)) : T(0);
                    y_norm = (H > 1) ? (T(2) * static_cast<T>(h) / static_cast<T>(H - 1) - T(1)) : T(0);
                } else {
                    x_norm = (T(2) * static_cast<T>(w) + T(1)) / static_cast<T>(W) - T(1);
                    y_norm = (T(2) * static_cast<T>(h) + T(1)) / static_cast<T>(H) - T(1);
                }

                // Apply affine transformation: [x', y'] = theta * [x_norm, y_norm, 1]^T
                T x_out = t[0] * x_norm + t[1] * y_norm + t[2];
                T y_out = t[3] * x_norm + t[4] * y_norm + t[5];

                int64_t idx = ((n * H + h) * W + w) * 2;
                g_data[idx] = x_out;
                g_data[idx + 1] = y_out;
            }
        }
    }

    return grid;
}

} // anonymous namespace

auto grid_sample_kernel(const Tensor& input, const Tensor& grid,
                        const std::string& mode,
                        const std::string& padding_mode,
                        bool align_corners) -> Tensor {
    // input: (N, C, H_in, W_in), grid: (N, H_out, W_out, 2).
    // Pick the compute dtype the same way the backward does: Float64 when
    // either input or grid is Float64, else Float32. This keeps forward and
    // backward at matching precision (Float64 gradcheck) instead of silently
    // downcasting an FP64 forward to FP32. Float16/BFloat16 widen to Float32.
    const DType in_dt = input.dtype();
    const DType gr_dt = grid.dtype();
    const DType compute = (in_dt == DType::Float64 || gr_dt == DType::Float64)
        ? DType::Float64 : DType::Float32;

    Tensor input_c = input.to(compute);
    Tensor grid_c = grid.to(compute);

    Tensor out_c = (compute == DType::Float64)
        ? grid_sample_forward_impl<double>(input_c, grid_c, mode, padding_mode, align_corners)
        : grid_sample_forward_impl<float>(input_c, grid_c, mode, padding_mode, align_corners);

    return out_c.to(in_dt);
}

auto affine_grid_kernel(const Tensor& theta,
                        const std::vector<int64_t>& size,
                        bool align_corners) -> Tensor {
    // theta: (N, 2, 3), size: {N, C, H, W}. Compute in Float64 when theta is
    // Float64, else Float32; return the grid in theta's dtype.
    const DType compute = (theta.dtype() == DType::Float64)
        ? DType::Float64 : DType::Float32;
    Tensor theta_c = theta.to(compute);

    Tensor grid = (compute == DType::Float64)
        ? affine_grid_forward_impl<double>(theta_c, size, align_corners)
        : affine_grid_forward_impl<float>(theta_c, size, align_corners);

    return grid.to(theta.dtype());
}

// ============================================================================
// Backward kernels (audit Q.4)
// ============================================================================

namespace {

// Templated cubic weights + derivatives, kept private to this TU.
template <typename T>
inline void cubic_weights_T(T t, T w[4]) {
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
inline void cubic_dweights_T(T t, T dw[4]) {
    constexpr T a = T(-0.5);
    const T u = T(1) - t;
    dw[0] = (T(3) * a * t * t - T(4) * a * t + a);
    dw[1] = (T(3) * (a + T(2)) * t * t - T(2) * (a + T(3)) * t);
    dw[2] = -(T(3) * (a + T(2)) * u * u - T(2) * (a + T(3)) * u);
    dw[3] = -(T(3) * a * u * u - T(4) * a * u + a);
}

// Padding helpers used for bicubic neighbour fetch (border/reflection clamp
// past the edge, zeros leaves the neighbour out of bounds).
template <typename T>
inline T denormalize_T(T coord, int64_t size, bool align_corners) {
    if (align_corners) {
        return (coord + T(1)) * T(0.5) * static_cast<T>(size - 1);
    }
    return ((coord + T(1)) * static_cast<T>(size) - T(1)) * T(0.5);
}

template <typename T>
inline T reflect_coord_T(T coord, int64_t size, bool align_corners) {
    return reflect_coord_impl<T>(coord, size, align_corners);
}

// d(reflect_coord_impl)/d(coord) = +1 or -1, depending on which leg of the
// reflection fold the coordinate lands on. The backward must chain this factor
// into grad_grid for reflection padding; omitting it (as the CPU path used to)
// drops the sign whenever a sample lands in an odd reflection region, diverging
// from PyTorch and the CUDA backend.
template <typename T>
inline T reflect_coord_grad_T(T coord, int64_t size, bool align_corners) {
    if (size <= 1) return T(0);
    T twice_low  = align_corners ? T(0) : T(-1);
    T twice_high = align_corners ? static_cast<T>(2 * (size - 1))
                                 : static_cast<T>(2 * size - 1);
    T mn = twice_low / T(2);
    T span = (twice_high - twice_low) / T(2);
    T d = coord - mn;
    T sign = (d < T(0)) ? T(-1) : T(1);
    int64_t flips = static_cast<int64_t>(std::floor(std::fabs(d) / span));
    if (flips % 2 != 0) sign = -sign;
    return sign;
}

// Full backward implementation for one dtype. Computes grad_input AND
// grad_grid in a single scalar pass — both have the same access pattern,
// just differing in how they accumulate.
//
// For the 'zeros' / 'border' / 'reflection' padding modes the math
// matches PyTorch's CPU reference: padding affects the coordinate
// transform but only `zeros` zeroes out the grad_grid where the
// transformed sample landed out of bounds.
template <typename T>
void grid_sample_backward_impl(
    const Tensor& grad_output, const Tensor& input, const Tensor& grid,
    const std::string& mode, const std::string& padding_mode, bool align_corners,
    Tensor& grad_input, Tensor& grad_grid)
{
    auto in_shape = input.shape();
    auto grid_shape = grid.shape();
    const int64_t N = in_shape[0];
    const int64_t C = in_shape[1];
    const int64_t H_in = in_shape[2];
    const int64_t W_in = in_shape[3];
    const int64_t H_out = grid_shape[1];
    const int64_t W_out = grid_shape[2];

    const T* in_data   = input.data<T>();
    const T* grid_data = grid.data<T>();
    const T* go_data   = grad_output.data<T>();
    T* gi_data         = grad_input.data<T>();
    T* gg_data         = grad_grid.data<T>();

    // Zero outputs.
    std::fill(gi_data, gi_data + grad_input.numel(), T(0));
    std::fill(gg_data, gg_data + grad_grid.numel(),  T(0));

    auto clamp_idx = [](int64_t v, int64_t lo, int64_t hi) -> int64_t {
        return std::max(lo, std::min(v, hi));
    };

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H_out; ++h) {
            for (int64_t w = 0; w < W_out; ++w) {
                const int64_t grid_idx = ((n * H_out + h) * W_out + w) * 2;
                const T gx = grid_data[grid_idx];
                const T gy = grid_data[grid_idx + 1];

                T ix = denormalize_T<T>(gx, W_in, align_corners);
                T iy = denormalize_T<T>(gy, H_in, align_corners);

                // For bilinear/nearest we track whether the sample fell
                // inside the *unpadded* domain (so grad_grid is zero
                // outside under 'zeros' padding — matches PyTorch).
                bool in_bounds_ix = (ix >= T(0) && ix <= static_cast<T>(W_in - 1));
                bool in_bounds_iy = (iy >= T(0) && iy <= static_cast<T>(H_in - 1));

                // Reflection contributes a ±1 fold-sign to the coordinate
                // gradient; capture it from the PRE-reflection coordinate before
                // ix/iy are overwritten below.
                T refl_sign_x = T(1);
                T refl_sign_y = T(1);
                if (padding_mode == "border") {
                    ix = std::min(std::max(ix, T(0)), static_cast<T>(W_in - 1));
                    iy = std::min(std::max(iy, T(0)), static_cast<T>(H_in - 1));
                } else if (padding_mode == "reflection") {
                    refl_sign_x = reflect_coord_grad_T<T>(ix, W_in, align_corners);
                    refl_sign_y = reflect_coord_grad_T<T>(iy, H_in, align_corners);
                    ix = reflect_coord_T<T>(ix, W_in, align_corners);
                    iy = reflect_coord_T<T>(iy, H_in, align_corners);
                }

                // d(px)/d(gx), d(py)/d(gy) — chain the reflection fold-sign
                // through the denormalisation factor back to the normalised grid.
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

                if (mode == "bilinear") {
                    const int64_t x0 = static_cast<int64_t>(std::floor(ix));
                    const int64_t y0 = static_cast<int64_t>(std::floor(iy));
                    const int64_t x1 = x0 + 1;
                    const int64_t y1 = y0 + 1;

                    const T wx1 = ix - static_cast<T>(x0);
                    const T wy1 = iy - static_cast<T>(y0);
                    const T wx0 = T(1) - wx1;
                    const T wy0 = T(1) - wy1;

                    T sum_dx = T(0);
                    T sum_dy = T(0);
                    for (int64_t c = 0; c < C; ++c) {
                        const T go = go_data[((n * C + c) * H_out + h) * W_out + w];
                        const T* ch_in = in_data + (n * C + c) * H_in * W_in;
                        T* ch_gi       = gi_data + (n * C + c) * H_in * W_in;

                        auto scatter = [&](int64_t y, int64_t x, T weight) {
                            if (y >= 0 && y < H_in && x >= 0 && x < W_in) {
                                ch_gi[y * W_in + x] += go * weight;
                            }
                        };
                        auto fetch = [&](int64_t y, int64_t x) -> T {
                            if (y >= 0 && y < H_in && x >= 0 && x < W_in) {
                                return ch_in[y * W_in + x];
                            }
                            return T(0);
                        };

                        scatter(y0, x0, wy0 * wx0);
                        scatter(y0, x1, wy0 * wx1);
                        scatter(y1, x0, wy1 * wx0);
                        scatter(y1, x1, wy1 * wx1);

                        const T dx = go * (wy0 * (-fetch(y0, x0) + fetch(y0, x1)) +
                                           wy1 * (-fetch(y1, x0) + fetch(y1, x1)));
                        const T dy = go * (wx0 * (-fetch(y0, x0) + fetch(y1, x0)) +
                                           wx1 * (-fetch(y0, x1) + fetch(y1, x1)));
                        sum_dx += dx;
                        sum_dy += dy;
                    }

                    // 'zeros' padding zeros grad_grid where the (un-clamped)
                    // sample fell outside, since the value was 0 independent
                    // of grid coordinate.
                    T scale_x = (padding_mode == "zeros" && !in_bounds_ix) ? T(0) : dix_dgx;
                    T scale_y = (padding_mode == "zeros" && !in_bounds_iy) ? T(0) : diy_dgy;
                    gg_data[grid_idx]     += sum_dx * scale_x;
                    gg_data[grid_idx + 1] += sum_dy * scale_y;
                } else if (mode == "nearest") {
                    const int64_t nx = static_cast<int64_t>(std::round(ix));
                    const int64_t ny = static_cast<int64_t>(std::round(iy));
                    if (ny >= 0 && ny < H_in && nx >= 0 && nx < W_in) {
                        for (int64_t c = 0; c < C; ++c) {
                            const T go = go_data[((n * C + c) * H_out + h) * W_out + w];
                            gi_data[((n * C + c) * H_in + ny) * W_in + nx] += go;
                        }
                    }
                    // nearest: grad w.r.t. grid is 0 (non-differentiable).
                } else if (mode == "bicubic") {
                    const int64_t ix_floor = static_cast<int64_t>(std::floor(ix));
                    const int64_t iy_floor = static_cast<int64_t>(std::floor(iy));
                    const T tx = ix - static_cast<T>(ix_floor);
                    const T ty = iy - static_cast<T>(iy_floor);
                    T wx[4], wy[4], dwx[4], dwy[4];
                    cubic_weights_T<T>(tx, wx);
                    cubic_weights_T<T>(ty, wy);
                    cubic_dweights_T<T>(tx, dwx);
                    cubic_dweights_T<T>(ty, dwy);

                    auto fetch = [&](int64_t c, int64_t y, int64_t x) -> T {
                        if (padding_mode == "zeros") {
                            if (y < 0 || y >= H_in || x < 0 || x >= W_in) return T(0);
                            return in_data[((n * C + c) * H_in + y) * W_in + x];
                        }
                        y = clamp_idx(y, 0, H_in - 1);
                        x = clamp_idx(x, 0, W_in - 1);
                        return in_data[((n * C + c) * H_in + y) * W_in + x];
                    };
                    auto scatter = [&](int64_t c, int64_t y, int64_t x, T weight) {
                        if (padding_mode == "zeros") {
                            if (y >= 0 && y < H_in && x >= 0 && x < W_in) {
                                gi_data[((n * C + c) * H_in + y) * W_in + x] += weight;
                            }
                            return;
                        }
                        y = clamp_idx(y, 0, H_in - 1);
                        x = clamp_idx(x, 0, W_in - 1);
                        gi_data[((n * C + c) * H_in + y) * W_in + x] += weight;
                    };

                    T sum_dx = T(0);
                    T sum_dy = T(0);
                    for (int64_t c = 0; c < C; ++c) {
                        const T go = go_data[((n * C + c) * H_out + h) * W_out + w];
                        T dval_dix = T(0);
                        T dval_diy = T(0);
                        for (int dy = -1; dy <= 2; ++dy) {
                            for (int dx = -1; dx <= 2; ++dx) {
                                const T weight = wy[dy + 1] * wx[dx + 1];
                                scatter(c, iy_floor + dy, ix_floor + dx, go * weight);
                                const T v = fetch(c, iy_floor + dy, ix_floor + dx);
                                dval_dix += wy[dy + 1] * dwx[dx + 1] * v;
                                dval_diy += dwy[dy + 1] * wx[dx + 1] * v;
                            }
                        }
                        sum_dx += go * dval_dix;
                        sum_dy += go * dval_diy;
                    }
                    gg_data[grid_idx]     += sum_dx * dix_dgx;
                    gg_data[grid_idx + 1] += sum_dy * diy_dgy;
                } else {
                    throw std::invalid_argument(
                        "grid_sample_backward: unknown mode '" + mode +
                        "'. Supported: bilinear, nearest, bicubic.");
                }
            }
        }
    }
}

template <typename T>
void affine_grid_backward_impl(const Tensor& grad_grid,
                               int64_t N, int64_t H, int64_t W,
                               bool align_corners, Tensor& grad_theta) {
    const T* gg_data = grad_grid.data<T>();
    T* gt_data = grad_theta.data<T>();
    std::fill(gt_data, gt_data + grad_theta.numel(), T(0));

    for (int64_t n = 0; n < N; ++n) {
        T* t = gt_data + n * 6;
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                T x_norm, y_norm;
                if (align_corners) {
                    x_norm = (W > 1) ? (T(2) * static_cast<T>(w) / static_cast<T>(W - 1) - T(1)) : T(0);
                    y_norm = (H > 1) ? (T(2) * static_cast<T>(h) / static_cast<T>(H - 1) - T(1)) : T(0);
                } else {
                    x_norm = (T(2) * static_cast<T>(w) + T(1)) / static_cast<T>(W) - T(1);
                    y_norm = (T(2) * static_cast<T>(h) + T(1)) / static_cast<T>(H) - T(1);
                }
                const int64_t idx = ((n * H + h) * W + w) * 2;
                const T dg_x = gg_data[idx];
                const T dg_y = gg_data[idx + 1];
                t[0] += dg_x * x_norm;
                t[1] += dg_x * y_norm;
                t[2] += dg_x;
                t[3] += dg_y * x_norm;
                t[4] += dg_y * y_norm;
                t[5] += dg_y;
            }
        }
    }
}

} // anonymous namespace

auto grid_sample_backward_kernel(const Tensor& grad_output,
                                 const Tensor& input, const Tensor& grid,
                                 const std::string& mode,
                                 const std::string& padding_mode,
                                 bool align_corners)
    -> std::pair<Tensor, Tensor>
{
    auto in_shape_span = input.shape();
    auto gr_shape_span = grid.shape();
    std::vector<int64_t> in_shape_v(in_shape_span.begin(), in_shape_span.end());
    std::vector<int64_t> gr_shape_v(gr_shape_span.begin(), gr_shape_span.end());

    // Promote inputs to the common compute dtype. CPU kernel supports
    // native F32 and F64; F16/BF16 widen to F32 then narrow back.
    const DType in_dt = input.dtype();
    const DType gr_dt = grid.dtype();
    DType compute = (in_dt == DType::Float64 || gr_dt == DType::Float64)
        ? DType::Float64 : DType::Float32;

    Tensor inp_c = input.to(compute);
    Tensor grid_c = grid.to(compute);
    Tensor go_c = grad_output.to(compute);
    Tensor gi_c(in_shape_v, compute, input.device());
    Tensor gg_c(gr_shape_v, compute, grid.device());

    if (compute == DType::Float64) {
        grid_sample_backward_impl<double>(go_c, inp_c, grid_c,
            mode, padding_mode, align_corners, gi_c, gg_c);
    } else {
        grid_sample_backward_impl<float>(go_c, inp_c, grid_c,
            mode, padding_mode, align_corners, gi_c, gg_c);
    }

    return {gi_c.to(in_dt), gg_c.to(gr_dt)};
}

auto affine_grid_backward_kernel(const Tensor& grad_grid,
                                 const std::vector<int64_t>& size,
                                 bool align_corners) -> Tensor
{
    const int64_t N = size[0];
    const int64_t H = size[2];
    const int64_t W = size[3];

    DType gr_dt = grad_grid.dtype();
    DType compute = (gr_dt == DType::Float64) ? DType::Float64 : DType::Float32;

    Tensor gg_c = grad_grid.to(compute);
    Tensor gt_c({N, 2, 3}, compute, grad_grid.device());

    if (compute == DType::Float64) {
        affine_grid_backward_impl<double>(gg_c, N, H, W, align_corners, gt_c);
    } else {
        affine_grid_backward_impl<float>(gg_c, N, H, W, align_corners, gt_c);
    }
    return gt_c.to(gr_dt);
}

} // namespace cpu
} // namespace tenzor
