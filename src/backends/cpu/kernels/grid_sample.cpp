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

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());

    // Convert to Float32 for computation
    Tensor input_f32 = input.to(DType::Float32);
    Tensor grid_f32 = grid.to(DType::Float32);
    const float* in_data = input_f32.data<float>();
    const float* grid_data = grid_f32.data<float>();

    Tensor output_f32({N, C, H_out, W_out}, DType::Float32, input.device());
    float* out_data = output_f32.data<float>();

    int64_t spatial_size = H_in * W_in;

    #pragma omp parallel for collapse(2) if(N * H_out * W_out > 4096)
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

} // namespace cpu
} // namespace tenzor
