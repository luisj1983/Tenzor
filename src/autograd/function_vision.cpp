/**
 * @file function_vision.cpp
 * @brief Autograd backward functions for vision operations (grid_sample, affine_grid).
 */

#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>
#include <algorithm>

namespace tenzor {

// ============================================================================
// GridSampleBackward
// ============================================================================

auto GridSampleBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor(), inputs[1].tensor()});

    auto result = ops::grid_sample(inputs[0].tensor(), inputs[1].tensor(),
                                   mode_, padding_mode_, align_corners_);
    return {Variable(result, inputs[0].requires_grad() || inputs[1].requires_grad())};
}

auto GridSampleBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const Tensor& input = saved_tensors_[0];
    const Tensor& grid = saved_tensors_[1];
    const Tensor& grad_output = grad_outputs[0];

    auto in_shape = input.shape();
    auto grid_shape = grid.shape();

    int64_t N = in_shape[0];
    int64_t C = in_shape[1];
    int64_t H_in = in_shape[2];
    int64_t W_in = in_shape[3];
    int64_t H_out = grid_shape[1];
    int64_t W_out = grid_shape[2];

    // Compute gradient w.r.t. input using scatter of grad_output weighted by
    // bilinear interpolation weights. This is the transpose of the forward sampling.
    Tensor grad_input = zeros(std::vector<int64_t>{N, C, H_in, W_in},
                              input.dtype(), input.device());

    // Compute gradient w.r.t. grid by differentiating the bilinear interpolation
    // formula w.r.t. grid coordinates.
    Tensor grad_grid = zeros(std::vector<int64_t>{N, H_out, W_out, 2},
                             grid.dtype(), grid.device());

    // For simplicity and correctness, compute on CPU in Float32
    Tensor input_f32 = input.to(DType::Float32).cpu();
    Tensor grid_f32 = grid.to(DType::Float32).cpu();
    Tensor grad_out_f32 = grad_output.to(DType::Float32).cpu();

    Tensor gi_f32 = zeros(std::vector<int64_t>{N, C, H_in, W_in}, DType::Float32, Device::cpu());
    Tensor gg_f32 = zeros(std::vector<int64_t>{N, H_out, W_out, 2}, DType::Float32, Device::cpu());

    const float* in_data = input_f32.data<float>();
    const float* grid_data = grid_f32.data<float>();
    const float* go_data = grad_out_f32.data<float>();
    float* gi_data = gi_f32.data<float>();
    float* gg_data = gg_f32.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H_out; ++h) {
            for (int64_t w = 0; w < W_out; ++w) {
                int64_t grid_idx = ((n * H_out + h) * W_out + w) * 2;
                float gx = grid_data[grid_idx];
                float gy = grid_data[grid_idx + 1];

                // Denormalize
                float ix, iy;
                if (align_corners_) {
                    ix = (gx + 1.0f) * 0.5f * static_cast<float>(W_in - 1);
                    iy = (gy + 1.0f) * 0.5f * static_cast<float>(H_in - 1);
                } else {
                    ix = ((gx + 1.0f) * static_cast<float>(W_in) - 1.0f) * 0.5f;
                    iy = ((gy + 1.0f) * static_cast<float>(H_in) - 1.0f) * 0.5f;
                }

                if (mode_ == "bicubic") {
                    // Phase P0 / Fix 7: real bicubic backward. 4x4 stencil
                    // mirroring the forward (a = -0.5 Catmull-Rom). For each
                    // (dy, dx) ∈ [-1, 2], scatter grad_output weighted by
                    // wy[dy+1] * wx[dx+1] into grad_input, and accumulate the
                    // analytic derivative of (wy * wx) w.r.t. ix / iy into
                    // grad_grid.
                    const int64_t ix_floor = static_cast<int64_t>(std::floor(ix));
                    const int64_t iy_floor = static_cast<int64_t>(std::floor(iy));
                    const float tx = ix - static_cast<float>(ix_floor);
                    const float ty = iy - static_cast<float>(iy_floor);

                    constexpr float a = -0.5f;
                    auto cubic_w = [](float t, float w[4]) {
                        const float t2 = t * t;
                        const float t3 = t2 * t;
                        w[0] = ((a * t - 2.0f * a) * t + a) * t;
                        w[1] = ((a + 2.0f) * t3 - (a + 3.0f) * t2 + 1.0f);
                        const float u = 1.0f - t;
                        const float u2 = u * u;
                        const float u3 = u2 * u;
                        w[2] = ((a + 2.0f) * u3 - (a + 3.0f) * u2 + 1.0f);
                        w[3] = ((a * u - 2.0f * a) * u + a) * u;
                    };
                    // d w[k] / d t. The chain through 1-t flips the sign for
                    // w[2] and w[3].
                    auto cubic_dw = [](float t, float dw[4]) {
                        dw[0] = (3.0f * a * t * t - 4.0f * a * t + a);
                        dw[1] = (3.0f * (a + 2.0f) * t * t - 2.0f * (a + 3.0f) * t);
                        const float u = 1.0f - t;
                        dw[2] = -(3.0f * (a + 2.0f) * u * u - 2.0f * (a + 3.0f) * u);
                        dw[3] = -(3.0f * a * u * u - 4.0f * a * u + a);
                    };
                    float wx[4], wy[4], dwx[4], dwy[4];
                    cubic_w(tx, wx);
                    cubic_w(ty, wy);
                    cubic_dw(tx, dwx);
                    cubic_dw(ty, dwy);

                    float dix_dgx_bc, diy_dgy_bc;
                    if (align_corners_) {
                        dix_dgx_bc = 0.5f * static_cast<float>(W_in - 1);
                        diy_dgy_bc = 0.5f * static_cast<float>(H_in - 1);
                    } else {
                        dix_dgx_bc = 0.5f * static_cast<float>(W_in);
                        diy_dgy_bc = 0.5f * static_cast<float>(H_in);
                    }

                    for (int64_t c = 0; c < C; ++c) {
                        float go = go_data[((n * C + c) * H_out + h) * W_out + w];
                        auto scatter_add = [&](int64_t y, int64_t x, float weight) {
                            if (y >= 0 && y < H_in && x >= 0 && x < W_in) {
                                gi_data[((n * C + c) * H_in + y) * W_in + x] += go * weight;
                            }
                        };
                        auto safe_get = [&](int64_t y, int64_t x) -> float {
                            if (y >= 0 && y < H_in && x >= 0 && x < W_in)
                                return in_data[((n * C + c) * H_in + y) * W_in + x];
                            return 0.0f;
                        };
                        float dval_dix = 0.0f;
                        float dval_diy = 0.0f;
                        for (int dy = -1; dy <= 2; ++dy) {
                            for (int dx = -1; dx <= 2; ++dx) {
                                const float weight = wy[dy + 1] * wx[dx + 1];
                                scatter_add(iy_floor + dy, ix_floor + dx, weight);
                                const float v = safe_get(iy_floor + dy, ix_floor + dx);
                                dval_dix += wy[dy + 1] * dwx[dx + 1] * v;
                                dval_diy += dwy[dy + 1] * wx[dx + 1] * v;
                            }
                        }
                        gg_data[grid_idx]     += go * dval_dix * dix_dgx_bc;
                        gg_data[grid_idx + 1] += go * dval_diy * diy_dgy_bc;
                    }
                } else if (mode_ == "bilinear") {
                    int64_t x0 = static_cast<int64_t>(std::floor(ix));
                    int64_t y0 = static_cast<int64_t>(std::floor(iy));
                    int64_t x1 = x0 + 1;
                    int64_t y1 = y0 + 1;

                    float wx1 = ix - static_cast<float>(x0);
                    float wy1 = iy - static_cast<float>(y0);
                    float wx0 = 1.0f - wx1;
                    float wy0 = 1.0f - wy1;

                    for (int64_t c = 0; c < C; ++c) {
                        float go = go_data[((n * C + c) * H_out + h) * W_out + w];

                        // Gradient w.r.t. input (scatter)
                        auto scatter_add = [&](int64_t y, int64_t x, float weight) {
                            if (y >= 0 && y < H_in && x >= 0 && x < W_in) {
                                gi_data[((n * C + c) * H_in + y) * W_in + x] += go * weight;
                            }
                        };
                        scatter_add(y0, x0, wy0 * wx0);
                        scatter_add(y0, x1, wy0 * wx1);
                        scatter_add(y1, x0, wy1 * wx0);
                        scatter_add(y1, x1, wy1 * wx1);

                        // Gradient w.r.t. grid
                        auto safe_get = [&](int64_t y, int64_t x) -> float {
                            if (y >= 0 && y < H_in && x >= 0 && x < W_in)
                                return in_data[((n * C + c) * H_in + y) * W_in + x];
                            return 0.0f;
                        };

                        // d/d(ix): differentiate bilinear w.r.t. x coordinate
                        float dx = go * (wy0 * (-safe_get(y0, x0) + safe_get(y0, x1)) +
                                         wy1 * (-safe_get(y1, x0) + safe_get(y1, x1)));

                        // d/d(iy): differentiate bilinear w.r.t. y coordinate
                        float dy = go * (wx0 * (-safe_get(y0, x0) + safe_get(y1, x0)) +
                                         wx1 * (-safe_get(y0, x1) + safe_get(y1, x1)));

                        // Chain rule: d/d(gx) = d/d(ix) * d(ix)/d(gx)
                        float dix_dgx, diy_dgy;
                        if (align_corners_) {
                            dix_dgx = 0.5f * static_cast<float>(W_in - 1);
                            diy_dgy = 0.5f * static_cast<float>(H_in - 1);
                        } else {
                            dix_dgx = 0.5f * static_cast<float>(W_in);
                            diy_dgy = 0.5f * static_cast<float>(H_in);
                        }

                        gg_data[grid_idx] += dx * dix_dgx;
                        gg_data[grid_idx + 1] += dy * diy_dgy;
                    }
                }
                // nearest mode has zero gradients w.r.t. grid (non-differentiable)
            }
        }
    }

    // Move back to original device
    grad_input = gi_f32.to(input.dtype()).to(input.device());
    grad_grid = gg_f32.to(grid.dtype()).to(grid.device());

    return {grad_input, grad_grid};
}

// ============================================================================
// AffineGridBackward
// ============================================================================

auto AffineGridBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});

    auto result = ops::affine_grid(inputs[0].tensor(), size_, align_corners_);
    return {Variable(result, inputs[0].requires_grad())};
}

auto AffineGridBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const Tensor& grad_grid = grad_outputs[0];

    int64_t N = size_[0];
    int64_t H = size_[2];
    int64_t W = size_[3];

    // grad_grid: (N, H, W, 2)
    // theta: (N, 2, 3)
    // grid[n,h,w,:] = theta[n] @ [x_norm, y_norm, 1]^T
    // So d(loss)/d(theta) = sum_h,w  d(loss)/d(grid[n,h,w,:]) @ [x_norm, y_norm, 1]

    Tensor grad_theta = zeros(std::vector<int64_t>{N, 2, 3}, DType::Float32, grad_grid.device());

    Tensor gg_f32 = grad_grid.to(DType::Float32).cpu();
    Tensor gt_f32 = zeros(std::vector<int64_t>{N, 2, 3}, DType::Float32, Device::cpu());

    const float* gg_data = gg_f32.data<float>();
    float* gt_data = gt_f32.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                float x_norm, y_norm;
                if (align_corners_) {
                    x_norm = (W > 1) ? (2.0f * static_cast<float>(w) / static_cast<float>(W - 1) - 1.0f) : 0.0f;
                    y_norm = (H > 1) ? (2.0f * static_cast<float>(h) / static_cast<float>(H - 1) - 1.0f) : 0.0f;
                } else {
                    x_norm = (2.0f * static_cast<float>(w) + 1.0f) / static_cast<float>(W) - 1.0f;
                    y_norm = (2.0f * static_cast<float>(h) + 1.0f) / static_cast<float>(H) - 1.0f;
                }

                int64_t gg_idx = ((n * H + h) * W + w) * 2;
                float dg_x = gg_data[gg_idx];
                float dg_y = gg_data[gg_idx + 1];

                // d/d(theta[n, 0, :]) = dg_x * [x_norm, y_norm, 1]
                // d/d(theta[n, 1, :]) = dg_y * [x_norm, y_norm, 1]
                float* t = gt_data + n * 6;
                t[0] += dg_x * x_norm;
                t[1] += dg_x * y_norm;
                t[2] += dg_x;
                t[3] += dg_y * x_norm;
                t[4] += dg_y * y_norm;
                t[5] += dg_y;
            }
        }
    }

    return {gt_f32.to(grad_grid.dtype()).to(grad_grid.device())};
}

auto GridSampleBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Delegate to tensor-level backward and wrap results
    // GridSample backward is complex (bilinear interpolation gradients);
    // for higher-order, we wrap the tensor results as Variables
    auto result_tensors = backward({grad_outputs[0].tensor()});
    std::vector<Variable> results;
    results.reserve(result_tensors.size());
    for (auto& t : result_tensors) {
        results.emplace_back(std::move(t), grad_outputs[0].requires_grad());
    }
    return results;
}

auto AffineGridBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result_tensors = backward({grad_outputs[0].tensor()});
    std::vector<Variable> results;
    results.reserve(result_tensors.size());
    for (auto& t : result_tensors) {
        results.emplace_back(std::move(t), grad_outputs[0].requires_grad());
    }
    return results;
}

} // namespace tenzor
