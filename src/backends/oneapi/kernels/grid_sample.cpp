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
