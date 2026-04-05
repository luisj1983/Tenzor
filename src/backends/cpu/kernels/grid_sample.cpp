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

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H_out; ++h) {
            for (int64_t w = 0; w < W_out; ++w) {
                // Grid coordinates: (x, y) at grid[n, h, w, :]
                int64_t grid_idx = ((n * H_out + h) * W_out + w) * 2;
                float gx = grid_data[grid_idx];      // x coordinate
                float gy = grid_data[grid_idx + 1];   // y coordinate

                // Denormalize from [-1, 1] to pixel coordinates
                float ix = denormalize(gx, W_in, align_corners);
                float iy = denormalize(gy, H_in, align_corners);

                for (int64_t c = 0; c < C; ++c) {
                    float val = 0.0f;

                    if (mode == "nearest") {
                        int64_t nx = static_cast<int64_t>(std::round(
                            apply_padding(ix, W_in, padding_mode)));
                        int64_t ny = static_cast<int64_t>(std::round(
                            apply_padding(iy, H_in, padding_mode)));

                        if (is_in_bounds(static_cast<float>(ny), static_cast<float>(nx), H_in, W_in)) {
                            val = in_data[((n * C + c) * H_in + ny) * W_in + nx];
                        }
                    } else if (mode == "bilinear") {
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

                        auto safe_get = [&](int64_t y, int64_t x) -> float {
                            if (y >= 0 && y < H_in && x >= 0 && x < W_in) {
                                return in_data[((n * C + c) * H_in + y) * W_in + x];
                            }
                            return 0.0f;  // zeros padding
                        };

                        val = wy0 * wx0 * safe_get(y0, x0) +
                              wy0 * wx1 * safe_get(y0, x1) +
                              wy1 * wx0 * safe_get(y1, x0) +
                              wy1 * wx1 * safe_get(y1, x1);
                    } else {
                        // bicubic - simplified: use bilinear for now, can extend later
                        // Full bicubic requires 16-point interpolation with cubic weights
                        float px = apply_padding(ix, W_in, padding_mode);
                        float py = apply_padding(iy, H_in, padding_mode);

                        int64_t x0 = static_cast<int64_t>(std::floor(px));
                        int64_t y0 = static_cast<int64_t>(std::floor(py));

                        float wx1 = px - static_cast<float>(x0);
                        float wy1 = py - static_cast<float>(y0);
                        float wx0 = 1.0f - wx1;
                        float wy0 = 1.0f - wy1;

                        auto safe_get = [&](int64_t y, int64_t x) -> float {
                            y = std::clamp(y, int64_t(0), H_in - 1);
                            x = std::clamp(x, int64_t(0), W_in - 1);
                            return in_data[((n * C + c) * H_in + y) * W_in + x];
                        };

                        val = wy0 * wx0 * safe_get(y0, x0) +
                              wy0 * wx1 * safe_get(y0, x0 + 1) +
                              wy1 * wx0 * safe_get(y0 + 1, x0) +
                              wy1 * wx1 * safe_get(y0 + 1, x0 + 1);
                    }

                    out_data[((n * C + c) * H_out + h) * W_out + w] = val;
                }
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
