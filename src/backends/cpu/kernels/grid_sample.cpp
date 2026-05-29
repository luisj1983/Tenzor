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
inline float denormalize(float coord, int64_t size, bool align_corners) {
    if (align_corners) {
        // [-1, 1] -> [0, size-1]
        return (coord + 1.0f) * 0.5f * static_cast<float>(size - 1);
    } else {
        // [-1, 1] -> [-0.5, size-0.5]
        return ((coord + 1.0f) * static_cast<float>(size) - 1.0f) * 0.5f;
    }
}

// Apply padding mode to compute effective coordinate
inline float apply_padding(float coord, int64_t size, const std::string& padding_mode) {
    if (padding_mode == "border") {
        coord = std::clamp(coord, 0.0f, static_cast<float>(size - 1));
    } else if (padding_mode == "reflection") {
        // Reflect coordinates to stay in [0, size-1]
        if (size > 1) {
            float max_val = static_cast<float>(size - 1);
            // Map to [0, 2*max_val]
            coord = std::fabs(coord);
            float period = 2.0f * max_val;
            coord = std::fmod(coord, period);
            if (coord > max_val) {
                coord = period - coord;
            }
        } else {
            coord = 0.0f;
        }
    }
    // "zeros" mode: out-of-bound coordinates will use zero values (handled in sampling)
    return coord;
}

inline bool is_in_bounds(float y, float x, int64_t H, int64_t W) {
    return y >= 0 && y < H && x >= 0 && x < W;
}

// Catmull-Rom cubic interpolation weights with a = -0.5. Matches the
// `cubic_interp1d` helper used by interpolate(mode='bicubic') in vision.cpp.
// Returns the four weights w[-1..2] that combine with the 4 neighbour values.
inline void cubic_weights(float t, float w[4]) {
    constexpr float a = -0.5f;
    const float t2 = t * t;
    const float t3 = t2 * t;
    w[0] = ((a * t - 2.0f * a) * t + a) * t;                // t = 1+t   -> |x|=1+t
    w[1] = ((a + 2.0f) * t3 - (a + 3.0f) * t2 + 1.0f);      // t = t     -> |x|=t
    w[2] = ((a + 2.0f) * (1.0f - t) * (1.0f - t) * (1.0f - t)
           - (a + 3.0f) * (1.0f - t) * (1.0f - t) + 1.0f);   // t = 1-t   -> |x|=1-t
    w[3] = ((a * (1.0f - t) - 2.0f * a) * (1.0f - t) + a) * (1.0f - t); // |x|=2-t
}

} // anonymous namespace

auto grid_sample_kernel(const Tensor& input, const Tensor& grid,
                        const std::string& mode,
                        const std::string& padding_mode,
                        bool align_corners) -> Tensor {
    // input: (N, C, H_in, W_in), grid: (N, H_out, W_out, 2)
    auto in_shape = input.shape();
    auto grid_shape = grid.shape();

    int64_t N = in_shape[0];
    int64_t C = in_shape[1];
    int64_t H_in = in_shape[2];
    int64_t W_in = in_shape[3];
    int64_t H_out = grid_shape[1];
    int64_t W_out = grid_shape[2];

    // Convert to Float32 for computation
    Tensor input_f32 = input.to(DType::Float32);
    Tensor grid_f32 = grid.to(DType::Float32);
    const float* in_data = input_f32.data<float>();
    const float* grid_data = grid_f32.data<float>();

    Tensor output_f32({N, C, H_out, W_out}, DType::Float32, input.device());
    float* out_data = output_f32.data<float>();

    int64_t spatial_size = H_in * W_in;

    #pragma omp parallel for collapse(2) if(N * H_out * W_out > ::tenzor::OmpThresholds::complex())
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t hw = 0; hw < H_out * W_out; ++hw) {
            int64_t h = hw / W_out;
            int64_t w = hw % W_out;

            // Grid coordinates: (x, y) at grid[n, h, w, :]
            int64_t grid_idx = ((n * H_out + h) * W_out + w) * 2;
            float gx = grid_data[grid_idx];
            float gy = grid_data[grid_idx + 1];

            float ix = denormalize(gx, W_in, align_corners);
            float iy = denormalize(gy, H_in, align_corners);

            if (mode == "bilinear") {
                float px = apply_padding(ix, W_in, padding_mode);
                float py = apply_padding(iy, H_in, padding_mode);

                int64_t x0 = static_cast<int64_t>(std::floor(px));
                int64_t y0 = static_cast<int64_t>(std::floor(py));
                int64_t x1 = x0 + 1;
                int64_t y1 = y0 + 1;

                float wx1 = px - static_cast<float>(x0);
                float wy1 = py - static_cast<float>(y0);
                float wx0 = 1.0f - wx1;
                float wy0 = 1.0f - wy1;

                // Compute the four corner weights once, reuse across all channels
                float w00 = wy0 * wx0;
                float w01 = wy0 * wx1;
                float w10 = wy1 * wx0;
                float w11 = wy1 * wx1;

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
                // Vectorize across channels: process 8 channels at a time
                __m256 vw00 = _mm256_set1_ps(w00);
                __m256 vw01 = _mm256_set1_ps(w01);
                __m256 vw10 = _mm256_set1_ps(w10);
                __m256 vw11 = _mm256_set1_ps(w11);
                __m256 vzero = _mm256_setzero_ps();

                for (; c + 8 <= C; c += 8) {
                    // Load 4 corner values for 8 consecutive channels
                    // Channel c data starts at in_data[n*C*spatial + c*spatial]
                    const float* base = in_data + n * C * spatial_size + c * spatial_size;
                    float* out_base = out_data + n * C * H_out * W_out + c * H_out * W_out;

                    // Gather from 8 channels at the same spatial offset
                    // Channels are contiguous blocks of spatial_size floats
                    __m256 v00 = vzero, v01 = vzero, v10 = vzero, v11 = vzero;
                    if (b00) {
                        // Load base[0*spatial+off00], base[1*spatial+off00], ..., base[7*spatial+off00]
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

                    // Bilinear interpolation: w00*v00 + w01*v01 + w10*v10 + w11*v11
                    __m256 result = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(vw00, v00), _mm256_mul_ps(vw01, v01)),
                        _mm256_add_ps(_mm256_mul_ps(vw10, v10), _mm256_mul_ps(vw11, v11)));

                    // Store 8 channels of output
                    alignas(32) float out_tmp[8];
                    _mm256_store_ps(out_tmp, result);
                    int64_t out_hw = h * W_out + w;
                    for (int k = 0; k < 8; ++k) {
                        out_base[k * H_out * W_out + out_hw] = out_tmp[k];
                    }
                }
#endif
                // Scalar tail for remaining channels
                for (; c < C; ++c) {
                    float val = 0.0f;
                    const float* ch_data = in_data + (n * C + c) * spatial_size;
                    if (b00) val += w00 * ch_data[off00];
                    if (b01) val += w01 * ch_data[off01];
                    if (b10) val += w10 * ch_data[off10];
                    if (b11) val += w11 * ch_data[off11];
                    out_data[((n * C + c) * H_out + h) * W_out + w] = val;
                }
            } else if (mode == "nearest") {
                int64_t nx = static_cast<int64_t>(std::round(
                    apply_padding(ix, W_in, padding_mode)));
                int64_t ny = static_cast<int64_t>(std::round(
                    apply_padding(iy, H_in, padding_mode)));

                bool in_bounds = is_in_bounds(static_cast<float>(ny), static_cast<float>(nx), H_in, W_in);
                for (int64_t c = 0; c < C; ++c) {
                    float val = 0.0f;
                    if (in_bounds) {
                        val = in_data[((n * C + c) * H_in + ny) * W_in + nx];
                    }
                    out_data[((n * C + c) * H_out + h) * W_out + w] = val;
                }
            } else if (mode == "bicubic") {
                // Phase P0 / Fix 7: real bicubic interpolation. 4x4 neighbourhood
                // with Catmull-Rom (a = -0.5) basis, matching PyTorch and the
                // existing interpolate(mode='bicubic') kernel in vision.cpp.
                // Padding is applied per-neighbour so all three modes (zeros,
                // border, reflection) work the same as bilinear.
                const float ix_pad = apply_padding(ix, W_in, padding_mode);
                const float iy_pad = apply_padding(iy, H_in, padding_mode);
                const int64_t ix_floor = static_cast<int64_t>(std::floor(ix_pad));
                const int64_t iy_floor = static_cast<int64_t>(std::floor(iy_pad));
                const float tx = ix_pad - static_cast<float>(ix_floor);
                const float ty = iy_pad - static_cast<float>(iy_floor);
                float wx[4], wy[4];
                cubic_weights(tx, wx);
                cubic_weights(ty, wy);

                for (int64_t c = 0; c < C; ++c) {
                    const float* ch_data = in_data + (n * C + c) * spatial_size;
                    auto safe_get = [&](int64_t y, int64_t x) -> float {
                        if (padding_mode == "zeros") {
                            if (y < 0 || y >= H_in || x < 0 || x >= W_in) return 0.0f;
                            return ch_data[y * W_in + x];
                        }
                        // border / reflection: clamp since apply_padding already
                        // mapped the center; neighbours past the edge clamp.
                        y = std::clamp(y, int64_t(0), H_in - 1);
                        x = std::clamp(x, int64_t(0), W_in - 1);
                        return ch_data[y * W_in + x];
                    };
                    float val = 0.0f;
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

    return output_f32.to(input.dtype());
}

auto affine_grid_kernel(const Tensor& theta,
                        const std::vector<int64_t>& size,
                        bool align_corners) -> Tensor {
    // theta: (N, 2, 3), size: {N, C, H, W}
    int64_t N = size[0];
    int64_t H = size[2];
    int64_t W = size[3];

    Tensor theta_f32 = theta.to(DType::Float32);
    const float* t_data = theta_f32.data<float>();

    // Output: (N, H, W, 2)
    Tensor grid({N, H, W, 2}, DType::Float32, theta.device());
    float* g_data = grid.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        const float* t = t_data + n * 6;  // 2x3 matrix flattened

        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                // Compute normalized coordinates for this pixel
                float y_norm, x_norm;
                if (align_corners) {
                    x_norm = (W > 1) ? (2.0f * static_cast<float>(w) / static_cast<float>(W - 1) - 1.0f) : 0.0f;
                    y_norm = (H > 1) ? (2.0f * static_cast<float>(h) / static_cast<float>(H - 1) - 1.0f) : 0.0f;
                } else {
                    x_norm = (2.0f * static_cast<float>(w) + 1.0f) / static_cast<float>(W) - 1.0f;
                    y_norm = (2.0f * static_cast<float>(h) + 1.0f) / static_cast<float>(H) - 1.0f;
                }

                // Apply affine transformation: [x', y'] = theta * [x_norm, y_norm, 1]^T
                float x_out = t[0] * x_norm + t[1] * y_norm + t[2];
                float y_out = t[3] * x_norm + t[4] * y_norm + t[5];

                int64_t idx = ((n * H + h) * W + w) * 2;
                g_data[idx] = x_out;
                g_data[idx + 1] = y_out;
            }
        }
    }

    return grid;
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
inline T reflect_coord_T(T coord, int64_t size) {
    if (size <= 1) return T(0);
    T max_val = static_cast<T>(size - 1);
    coord = std::fabs(coord);
    T period = T(2) * max_val;
    coord = std::fmod(coord, period);
    if (coord > max_val) coord = period - coord;
    return coord;
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

                if (padding_mode == "border") {
                    ix = std::min(std::max(ix, T(0)), static_cast<T>(W_in - 1));
                    iy = std::min(std::max(iy, T(0)), static_cast<T>(H_in - 1));
                } else if (padding_mode == "reflection") {
                    ix = reflect_coord_T<T>(ix, W_in);
                    iy = reflect_coord_T<T>(iy, H_in);
                }

                // d(ix)/d(gx), d(iy)/d(gy) — chain back to normalised grid.
                T dix_dgx, diy_dgy;
                if (align_corners) {
                    dix_dgx = T(0.5) * static_cast<T>(W_in - 1);
                    diy_dgy = T(0.5) * static_cast<T>(H_in - 1);
                } else {
                    dix_dgx = T(0.5) * static_cast<T>(W_in);
                    diy_dgy = T(0.5) * static_cast<T>(H_in);
                }

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
