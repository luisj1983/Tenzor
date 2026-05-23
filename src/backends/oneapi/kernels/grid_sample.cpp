/**
 * @file grid_sample.cpp
 * @brief OneAPI/SYCL port of grid_sample and affine_grid kernels.
 *
 * Mirrors src/backends/cuda/kernels/grid_sample.cu; replaces the previous
 * CPU-roundtrip fallback in oneapi_kernel_registry.cpp.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {
namespace oneapi {

namespace {

template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

inline float gs_denormalize(float coord, int size, bool align_corners) {
    if (align_corners) {
        return (coord + 1.0f) * 0.5f * static_cast<float>(size - 1);
    }
    return ((coord + 1.0f) * static_cast<float>(size) - 1.0f) * 0.5f;
}

inline float gs_reflect_coord(float coord, int size) {
    if (size <= 1) return 0.0f;
    float max_val = static_cast<float>(size - 1);
    coord = sycl::fabs(coord);
    float period = 2.0f * max_val;
    coord = sycl::fmod(coord, period);
    if (coord > max_val) coord = period - coord;
    return coord;
}

struct GridSampleBilinearKernel {};
struct GridSampleNearestKernel {};
struct AffineGridKernel {};

}  // namespace

auto grid_sample_kernel(const Tensor& input, const Tensor& grid,
                        const std::string& mode, const std::string& padding_mode,
                        bool align_corners, sycl::queue& queue) -> Tensor {
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

    Tensor output_f32(std::vector<int64_t>{N, C, H_out, W_out},
                      DType::Float32, input.device());

    int pad_mode = 0;
    if (padding_mode == "border") pad_mode = 1;
    else if (padding_mode == "reflection") pad_mode = 2;

    int total = N * C * H_out * W_out;
    if (total == 0) return output_f32.to(input.dtype());

    const float* input_ptr = get_data_ptr<const float>(input_f32);
    const float* grid_ptr = get_data_ptr<const float>(grid_f32);
    float* output_ptr = get_data_ptr<float>(output_f32);

    if (mode == "nearest") {
        queue.parallel_for<GridSampleNearestKernel>(sycl::range<1>(total),
            [=](sycl::id<1> idx_) {
                int idx = static_cast<int>(idx_);
                int w = idx % W_out;
                int h = (idx / W_out) % H_out;
                int c = (idx / (W_out * H_out)) % C;
                int n = idx / (C * H_out * W_out);

                int grid_idx = ((n * H_out + h) * W_out + w) * 2;
                float gx = grid_ptr[grid_idx];
                float gy = grid_ptr[grid_idx + 1];

                float ix = gs_denormalize(gx, W_in, align_corners);
                float iy = gs_denormalize(gy, H_in, align_corners);

                if (pad_mode == 1) {
                    ix = sycl::fmin(sycl::fmax(ix, 0.0f), static_cast<float>(W_in - 1));
                    iy = sycl::fmin(sycl::fmax(iy, 0.0f), static_cast<float>(H_in - 1));
                } else if (pad_mode == 2) {
                    ix = gs_reflect_coord(ix, W_in);
                    iy = gs_reflect_coord(iy, H_in);
                }

                int nx = static_cast<int>(sycl::round(ix));
                int ny = static_cast<int>(sycl::round(iy));

                float val = 0.0f;
                if (ny >= 0 && ny < H_in && nx >= 0 && nx < W_in) {
                    val = input_ptr[((n * C + c) * H_in + ny) * W_in + nx];
                }
                output_ptr[((n * C + c) * H_out + h) * W_out + w] = val;
            });
    } else {
        queue.parallel_for<GridSampleBilinearKernel>(sycl::range<1>(total),
            [=](sycl::id<1> idx_) {
                int idx = static_cast<int>(idx_);
                int w = idx % W_out;
                int h = (idx / W_out) % H_out;
                int c = (idx / (W_out * H_out)) % C;
                int n = idx / (C * H_out * W_out);

                int grid_idx = ((n * H_out + h) * W_out + w) * 2;
                float gx = grid_ptr[grid_idx];
                float gy = grid_ptr[grid_idx + 1];

                float ix = gs_denormalize(gx, W_in, align_corners);
                float iy = gs_denormalize(gy, H_in, align_corners);

                if (pad_mode == 1) {
                    ix = sycl::fmin(sycl::fmax(ix, 0.0f), static_cast<float>(W_in - 1));
                    iy = sycl::fmin(sycl::fmax(iy, 0.0f), static_cast<float>(H_in - 1));
                } else if (pad_mode == 2) {
                    ix = gs_reflect_coord(ix, W_in);
                    iy = gs_reflect_coord(iy, H_in);
                }

                int x0 = static_cast<int>(sycl::floor(ix));
                int y0 = static_cast<int>(sycl::floor(iy));
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                float wx1 = ix - static_cast<float>(x0);
                float wy1 = iy - static_cast<float>(y0);
                float wx0 = 1.0f - wx1;
                float wy0 = 1.0f - wy1;

                auto safe_get = [&](int y, int x) -> float {
                    if (y >= 0 && y < H_in && x >= 0 && x < W_in)
                        return input_ptr[((n * C + c) * H_in + y) * W_in + x];
                    return 0.0f;
                };

                float val = wy0 * wx0 * safe_get(y0, x0) +
                            wy0 * wx1 * safe_get(y0, x1) +
                            wy1 * wx0 * safe_get(y1, x0) +
                            wy1 * wx1 * safe_get(y1, x1);
                output_ptr[((n * C + c) * H_out + h) * W_out + w] = val;
            });
    }
    queue.wait();
    return output_f32.to(input.dtype());
}

// =========================================================================
// Backward kernels (audit Q.4) — F32 only, matching forward dtype coverage.
// =========================================================================

namespace {
struct GridSampleBilinearBackwardKernel {};
struct GridSampleNearestBackwardKernel {};
struct AffineGridBackwardKernel {};
}

auto grid_sample_backward_kernel(const Tensor& grad_output,
                                 const Tensor& input, const Tensor& grid,
                                 const std::string& mode,
                                 const std::string& padding_mode,
                                 bool align_corners, sycl::queue& queue)
    -> std::pair<Tensor, Tensor>
{
    if (mode == "bicubic") {
        throw std::runtime_error(
            "grid_sample_backward (OneAPI): mode='bicubic' is not implemented "
            "on this backend because the forward kernel only covers 'bilinear' "
            "and 'nearest'. Move the tensor to CPU/CUDA, or open a backend ticket.");
    }
    auto in_shape = input.shape();
    auto grid_shape = grid.shape();
    int N = static_cast<int>(in_shape[0]);
    int C = static_cast<int>(in_shape[1]);
    int H_in = static_cast<int>(in_shape[2]);
    int W_in = static_cast<int>(in_shape[3]);
    int H_out = static_cast<int>(grid_shape[1]);
    int W_out = static_cast<int>(grid_shape[2]);

    DType in_dt = input.dtype();
    DType gr_dt = grid.dtype();

    Tensor input_f32 = input.to(DType::Float32);
    Tensor grid_f32  = grid.to(DType::Float32);
    Tensor go_f32    = grad_output.to(DType::Float32);

    Tensor gi_f32(std::vector<int64_t>{N, C, H_in, W_in},
                  DType::Float32, input.device());
    Tensor gg_f32(std::vector<int64_t>{N, H_out, W_out, 2},
                  DType::Float32, grid.device());

    // Zero grad_input (atomic accumulator). For bilinear/nearest grad_grid is
    // written via direct store so it doesn't need pre-zeroing.
    queue.memset(gi_f32.data_ptr(), 0, gi_f32.numel() * sizeof(float)).wait();

    int pad_mode = 0;
    if (padding_mode == "border") pad_mode = 1;
    else if (padding_mode == "reflection") pad_mode = 2;

    int total = N * H_out * W_out;
    if (total == 0) return {gi_f32.to(in_dt), gg_f32.to(gr_dt)};

    const float* input_ptr = get_data_ptr<const float>(input_f32);
    const float* grid_ptr  = get_data_ptr<const float>(grid_f32);
    const float* go_ptr    = get_data_ptr<const float>(go_f32);
    float* gi_ptr          = get_data_ptr<float>(gi_f32);
    float* gg_ptr          = get_data_ptr<float>(gg_f32);

    if (mode == "bilinear") {
        queue.parallel_for<GridSampleBilinearBackwardKernel>(sycl::range<1>(total),
            [=](sycl::id<1> idx_) {
                int idx = static_cast<int>(idx_);
                int w = idx % W_out;
                int h = (idx / W_out) % H_out;
                int n = idx / (H_out * W_out);

                int grid_idx = ((n * H_out + h) * W_out + w) * 2;
                float gx = grid_ptr[grid_idx];
                float gy = grid_ptr[grid_idx + 1];
                float ix = gs_denormalize(gx, W_in, align_corners);
                float iy = gs_denormalize(gy, H_in, align_corners);

                bool in_bounds_ix = (ix >= 0.0f && ix <= static_cast<float>(W_in - 1));
                bool in_bounds_iy = (iy >= 0.0f && iy <= static_cast<float>(H_in - 1));

                if (pad_mode == 1) {
                    ix = sycl::fmin(sycl::fmax(ix, 0.0f), static_cast<float>(W_in - 1));
                    iy = sycl::fmin(sycl::fmax(iy, 0.0f), static_cast<float>(H_in - 1));
                } else if (pad_mode == 2) {
                    ix = gs_reflect_coord(ix, W_in);
                    iy = gs_reflect_coord(iy, H_in);
                }

                float dix_dgx, diy_dgy;
                if (align_corners) {
                    dix_dgx = 0.5f * static_cast<float>(W_in - 1);
                    diy_dgy = 0.5f * static_cast<float>(H_in - 1);
                } else {
                    dix_dgx = 0.5f * static_cast<float>(W_in);
                    diy_dgy = 0.5f * static_cast<float>(H_in);
                }

                int x0 = static_cast<int>(sycl::floor(ix));
                int y0 = static_cast<int>(sycl::floor(iy));
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                float wx1 = ix - static_cast<float>(x0);
                float wy1 = iy - static_cast<float>(y0);
                float wx0 = 1.0f - wx1;
                float wy0 = 1.0f - wy1;

                float sum_dx = 0.0f, sum_dy = 0.0f;
                for (int c = 0; c < C; ++c) {
                    const float go = go_ptr[((n * C + c) * H_out + h) * W_out + w];
                    auto scatter = [&](int y, int x, float weight) {
                        if (y >= 0 && y < H_in && x >= 0 && x < W_in) {
                            sycl::atomic_ref<float,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                                ref(gi_ptr[((n * C + c) * H_in + y) * W_in + x]);
                            ref.fetch_add(go * weight);
                        }
                    };
                    auto fetch = [&](int y, int x) -> float {
                        if (y >= 0 && y < H_in && x >= 0 && x < W_in) {
                            return input_ptr[((n * C + c) * H_in + y) * W_in + x];
                        }
                        return 0.0f;
                    };
                    scatter(y0, x0, wy0 * wx0);
                    scatter(y0, x1, wy0 * wx1);
                    scatter(y1, x0, wy1 * wx0);
                    scatter(y1, x1, wy1 * wx1);
                    sum_dx += go * (wy0 * (-fetch(y0, x0) + fetch(y0, x1)) +
                                    wy1 * (-fetch(y1, x0) + fetch(y1, x1)));
                    sum_dy += go * (wx0 * (-fetch(y0, x0) + fetch(y1, x0)) +
                                    wx1 * (-fetch(y0, x1) + fetch(y1, x1)));
                }
                float sx = (pad_mode == 0 && !in_bounds_ix) ? 0.0f : dix_dgx;
                float sy = (pad_mode == 0 && !in_bounds_iy) ? 0.0f : diy_dgy;
                gg_ptr[grid_idx]     = sum_dx * sx;
                gg_ptr[grid_idx + 1] = sum_dy * sy;
            });
    } else if (mode == "nearest") {
        queue.parallel_for<GridSampleNearestBackwardKernel>(sycl::range<1>(total),
            [=](sycl::id<1> idx_) {
                int idx = static_cast<int>(idx_);
                int w = idx % W_out;
                int h = (idx / W_out) % H_out;
                int n = idx / (H_out * W_out);

                int grid_idx = ((n * H_out + h) * W_out + w) * 2;
                float gx = grid_ptr[grid_idx];
                float gy = grid_ptr[grid_idx + 1];
                float ix = gs_denormalize(gx, W_in, align_corners);
                float iy = gs_denormalize(gy, H_in, align_corners);

                if (pad_mode == 1) {
                    ix = sycl::fmin(sycl::fmax(ix, 0.0f), static_cast<float>(W_in - 1));
                    iy = sycl::fmin(sycl::fmax(iy, 0.0f), static_cast<float>(H_in - 1));
                } else if (pad_mode == 2) {
                    ix = gs_reflect_coord(ix, W_in);
                    iy = gs_reflect_coord(iy, H_in);
                }
                int nx = static_cast<int>(sycl::round(ix));
                int ny = static_cast<int>(sycl::round(iy));

                gg_ptr[grid_idx]     = 0.0f;
                gg_ptr[grid_idx + 1] = 0.0f;

                if (ny >= 0 && ny < H_in && nx >= 0 && nx < W_in) {
                    for (int c = 0; c < C; ++c) {
                        const float go = go_ptr[((n * C + c) * H_out + h) * W_out + w];
                        sycl::atomic_ref<float,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            ref(gi_ptr[((n * C + c) * H_in + ny) * W_in + nx]);
                        ref.fetch_add(go);
                    }
                }
            });
    } else {
        throw std::invalid_argument(
            "grid_sample_backward (OneAPI): unknown mode '" + mode + "'");
    }
    queue.wait();
    return {gi_f32.to(in_dt), gg_f32.to(gr_dt)};
}

auto affine_grid_backward_kernel(const Tensor& grad_grid,
                                 const std::vector<int64_t>& size,
                                 bool align_corners, sycl::queue& queue) -> Tensor
{
    int N = static_cast<int>(size[0]);
    int H = static_cast<int>(size[2]);
    int W = static_cast<int>(size[3]);

    DType gr_dt = grad_grid.dtype();
    Tensor gg_f32 = grad_grid.to(DType::Float32);
    Tensor gt_f32(std::vector<int64_t>{N, 2, 3}, DType::Float32, grad_grid.device());

    queue.memset(gt_f32.data_ptr(), 0, gt_f32.numel() * sizeof(float)).wait();

    int total = N * H * W;
    if (total == 0) return gt_f32.to(gr_dt);

    const float* gg_ptr = get_data_ptr<const float>(gg_f32);
    float* gt_ptr       = get_data_ptr<float>(gt_f32);

    queue.parallel_for<AffineGridBackwardKernel>(sycl::range<1>(total),
        [=](sycl::id<1> idx_) {
            int idx = static_cast<int>(idx_);
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
            int gg_idx = ((n * H + h) * W + w) * 2;
            float dg_x = gg_ptr[gg_idx];
            float dg_y = gg_ptr[gg_idx + 1];

            float* t = gt_ptr + n * 6;
            auto add = [&](int i, float v) {
                sycl::atomic_ref<float,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
                    ref(t[i]);
                ref.fetch_add(v);
            };
            add(0, dg_x * x_norm);
            add(1, dg_x * y_norm);
            add(2, dg_x);
            add(3, dg_y * x_norm);
            add(4, dg_y * y_norm);
            add(5, dg_y);
        });
    queue.wait();
    return gt_f32.to(gr_dt);
}

auto affine_grid_kernel(const Tensor& theta, const std::vector<int64_t>& size,
                        bool align_corners, sycl::queue& queue) -> Tensor {
    int N = static_cast<int>(size[0]);
    int H = static_cast<int>(size[2]);
    int W = static_cast<int>(size[3]);

    Tensor theta_f32 = theta.to(DType::Float32);
    Tensor grid(std::vector<int64_t>{N, H, W, 2}, DType::Float32, theta.device());

    int total = N * H * W;
    if (total == 0) return grid;

    const float* theta_ptr = get_data_ptr<const float>(theta_f32);
    float* grid_ptr = get_data_ptr<float>(grid);

    queue.parallel_for<AffineGridKernel>(sycl::range<1>(total),
        [=](sycl::id<1> idx_) {
            int idx = static_cast<int>(idx_);
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

            const float* t = theta_ptr + n * 6;
            float x_out = t[0] * x_norm + t[1] * y_norm + t[2];
            float y_out = t[3] * x_norm + t[4] * y_norm + t[5];

            int out_idx = ((n * H + h) * W + w) * 2;
            grid_ptr[out_idx] = x_out;
            grid_ptr[out_idx + 1] = y_out;
        });
    queue.wait();
    return grid;
}

}  // namespace oneapi
}  // namespace tenzor
