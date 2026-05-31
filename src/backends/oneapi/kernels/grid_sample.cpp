/**
 * @file grid_sample.cpp
 * @brief OneAPI/SYCL port of grid_sample and affine_grid kernels.
 *
 * Mirrors src/backends/cuda/kernels/grid_sample.cu: templated on the scalar
 * type so Float32 and Float64 both run natively (no widen-narrow), with
 * bilinear / nearest / bicubic sampling and zeros/border/reflection padding.
 * Float16/BFloat16 promote to Float32 (each load/store is the natural cast).
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "oneapi_kernel_utils.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {
namespace oneapi {

namespace {

template <typename T>
inline T gs_denormalize(T coord, int size, bool align_corners) {
    if (align_corners) {
        return (coord + T(1)) * T(0.5) * static_cast<T>(size - 1);
    }
    return ((coord + T(1)) * static_cast<T>(size) - T(1)) * T(0.5);
}

template <typename T>
inline T gs_reflect_coord(T coord, int size) {
    if (size <= 1) return T(0);
    T max_val = static_cast<T>(size - 1);
    coord = sycl::fabs(coord);
    T period = T(2) * max_val;
    coord = sycl::fmod(coord, period);
    if (coord > max_val) coord = period - coord;
    return coord;
}

template <typename T>
inline void cubic_weights(T t, T w[4]) {
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
inline void cubic_dweights(T t, T dw[4]) {
    constexpr T a = T(-0.5);
    const T u = T(1) - t;
    dw[0] = (T(3) * a * t * t - T(4) * a * t + a);
    dw[1] = (T(3) * (a + T(2)) * t * t - T(2) * (a + T(3)) * t);
    dw[2] = -(T(3) * (a + T(2)) * u * u - T(2) * (a + T(3)) * u);
    dw[3] = -(T(3) * a * u * u - T(4) * a * u + a);
}

// Templated SYCL kernel-name tags (one instantiation per scalar type).
template <typename T> struct GSNearestK {};
template <typename T> struct GSBilinearK {};
template <typename T> struct GSBicubicK {};
template <typename T> struct GSNearestBwdK {};
template <typename T> struct GSBilinearBwdK {};
template <typename T> struct GSBicubicBwdK {};
struct AffineGridKernel {};
struct AffineGridBackwardKernel {};

template <typename T>
sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                 sycl::access::address_space::global_space>
make_atomic(T& ref) {
    return sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(ref);
}

// Forward dispatch for a single scalar type.
template <typename T>
void run_grid_sample_forward(sycl::queue& queue, const std::string& mode,
                             const T* input_ptr, const T* grid_ptr, T* output_ptr,
                             int N, int C, int H_in, int W_in, int H_out, int W_out,
                             int pad_mode, bool align_corners) {
    int total = N * C * H_out * W_out;
    if (total == 0) return;

    if (mode == "nearest") {
        queue.parallel_for<GSNearestK<T>>(sycl::range<1>(total), [=](sycl::id<1> idx_) {
            int idx = static_cast<int>(idx_);
            int w = idx % W_out;
            int h = (idx / W_out) % H_out;
            int c = (idx / (W_out * H_out)) % C;
            int n = idx / (C * H_out * W_out);
            int grid_idx = ((n * H_out + h) * W_out + w) * 2;
            T ix = gs_denormalize<T>(grid_ptr[grid_idx], W_in, align_corners);
            T iy = gs_denormalize<T>(grid_ptr[grid_idx + 1], H_in, align_corners);
            if (pad_mode == 1) {
                ix = sycl::fmin(sycl::fmax(ix, T(0)), static_cast<T>(W_in - 1));
                iy = sycl::fmin(sycl::fmax(iy, T(0)), static_cast<T>(H_in - 1));
            } else if (pad_mode == 2) {
                ix = gs_reflect_coord<T>(ix, W_in);
                iy = gs_reflect_coord<T>(iy, H_in);
            }
            int nx = static_cast<int>(sycl::round(ix));
            int ny = static_cast<int>(sycl::round(iy));
            T val = T(0);
            if (ny >= 0 && ny < H_in && nx >= 0 && nx < W_in) {
                val = input_ptr[((n * C + c) * H_in + ny) * W_in + nx];
            }
            output_ptr[((n * C + c) * H_out + h) * W_out + w] = val;
        });
    } else if (mode == "bilinear") {
        queue.parallel_for<GSBilinearK<T>>(sycl::range<1>(total), [=](sycl::id<1> idx_) {
            int idx = static_cast<int>(idx_);
            int w = idx % W_out;
            int h = (idx / W_out) % H_out;
            int c = (idx / (W_out * H_out)) % C;
            int n = idx / (C * H_out * W_out);
            int grid_idx = ((n * H_out + h) * W_out + w) * 2;
            T ix = gs_denormalize<T>(grid_ptr[grid_idx], W_in, align_corners);
            T iy = gs_denormalize<T>(grid_ptr[grid_idx + 1], H_in, align_corners);
            if (pad_mode == 1) {
                ix = sycl::fmin(sycl::fmax(ix, T(0)), static_cast<T>(W_in - 1));
                iy = sycl::fmin(sycl::fmax(iy, T(0)), static_cast<T>(H_in - 1));
            } else if (pad_mode == 2) {
                ix = gs_reflect_coord<T>(ix, W_in);
                iy = gs_reflect_coord<T>(iy, H_in);
            }
            int x0 = static_cast<int>(sycl::floor(ix));
            int y0 = static_cast<int>(sycl::floor(iy));
            int x1 = x0 + 1, y1 = y0 + 1;
            T wx1 = ix - static_cast<T>(x0);
            T wy1 = iy - static_cast<T>(y0);
            T wx0 = T(1) - wx1, wy0 = T(1) - wy1;
            auto safe_get = [&](int y, int x) -> T {
                if (y >= 0 && y < H_in && x >= 0 && x < W_in)
                    return input_ptr[((n * C + c) * H_in + y) * W_in + x];
                return T(0);
            };
            T val = wy0 * wx0 * safe_get(y0, x0) +
                    wy0 * wx1 * safe_get(y0, x1) +
                    wy1 * wx0 * safe_get(y1, x0) +
                    wy1 * wx1 * safe_get(y1, x1);
            output_ptr[((n * C + c) * H_out + h) * W_out + w] = val;
        });
    } else if (mode == "bicubic") {
        queue.parallel_for<GSBicubicK<T>>(sycl::range<1>(total), [=](sycl::id<1> idx_) {
            int idx = static_cast<int>(idx_);
            int w = idx % W_out;
            int h = (idx / W_out) % H_out;
            int c = (idx / (W_out * H_out)) % C;
            int n = idx / (C * H_out * W_out);
            int grid_idx = ((n * H_out + h) * W_out + w) * 2;
            T ix = gs_denormalize<T>(grid_ptr[grid_idx], W_in, align_corners);
            T iy = gs_denormalize<T>(grid_ptr[grid_idx + 1], H_in, align_corners);
            if (pad_mode == 1) {
                ix = sycl::fmin(sycl::fmax(ix, T(0)), static_cast<T>(W_in - 1));
                iy = sycl::fmin(sycl::fmax(iy, T(0)), static_cast<T>(H_in - 1));
            } else if (pad_mode == 2) {
                ix = gs_reflect_coord<T>(ix, W_in);
                iy = gs_reflect_coord<T>(iy, H_in);
            }
            int ix_floor = static_cast<int>(sycl::floor(ix));
            int iy_floor = static_cast<int>(sycl::floor(iy));
            T tx = ix - static_cast<T>(ix_floor);
            T ty = iy - static_cast<T>(iy_floor);
            T wx[4], wy[4];
            cubic_weights<T>(tx, wx);
            cubic_weights<T>(ty, wy);
            auto safe_get = [&](int y, int x) -> T {
                if (pad_mode == 0) {
                    if (y < 0 || y >= H_in || x < 0 || x >= W_in) return T(0);
                    return input_ptr[((n * C + c) * H_in + y) * W_in + x];
                }
                y = sycl::max(0, sycl::min(y, H_in - 1));
                x = sycl::max(0, sycl::min(x, W_in - 1));
                return input_ptr[((n * C + c) * H_in + y) * W_in + x];
            };
            T val = T(0);
            for (int dy = -1; dy <= 2; ++dy) {
                for (int dx = -1; dx <= 2; ++dx) {
                    val += wy[dy + 1] * wx[dx + 1] * safe_get(iy_floor + dy, ix_floor + dx);
                }
            }
            output_ptr[((n * C + c) * H_out + h) * W_out + w] = val;
        });
    } else {
        throw std::invalid_argument(
            "grid_sample (OneAPI): unknown mode '" + mode +
            "'. Supported: bilinear, nearest, bicubic.");
    }
    queue.wait_and_throw();
}

// Backward dispatch for a single scalar type. gi/gg must be pre-zeroed where
// required (gi is an atomic accumulator).
template <typename T>
void run_grid_sample_backward(sycl::queue& queue, const std::string& mode,
                              const T* input_ptr, const T* grid_ptr, const T* go_ptr,
                              T* gi_ptr, T* gg_ptr,
                              int N, int C, int H_in, int W_in, int H_out, int W_out,
                              int pad_mode, bool align_corners) {
    int total = N * H_out * W_out;
    if (total == 0) return;

    if (mode == "bilinear") {
        queue.parallel_for<GSBilinearBwdK<T>>(sycl::range<1>(total), [=](sycl::id<1> idx_) {
            int idx = static_cast<int>(idx_);
            int w = idx % W_out;
            int h = (idx / W_out) % H_out;
            int n = idx / (H_out * W_out);
            int grid_idx = ((n * H_out + h) * W_out + w) * 2;
            T ix = gs_denormalize<T>(grid_ptr[grid_idx], W_in, align_corners);
            T iy = gs_denormalize<T>(grid_ptr[grid_idx + 1], H_in, align_corners);
            bool in_bounds_ix = (ix >= T(0) && ix <= static_cast<T>(W_in - 1));
            bool in_bounds_iy = (iy >= T(0) && iy <= static_cast<T>(H_in - 1));
            if (pad_mode == 1) {
                ix = sycl::fmin(sycl::fmax(ix, T(0)), static_cast<T>(W_in - 1));
                iy = sycl::fmin(sycl::fmax(iy, T(0)), static_cast<T>(H_in - 1));
            } else if (pad_mode == 2) {
                ix = gs_reflect_coord<T>(ix, W_in);
                iy = gs_reflect_coord<T>(iy, H_in);
            }
            T dix_dgx, diy_dgy;
            if (align_corners) {
                dix_dgx = T(0.5) * static_cast<T>(W_in - 1);
                diy_dgy = T(0.5) * static_cast<T>(H_in - 1);
            } else {
                dix_dgx = T(0.5) * static_cast<T>(W_in);
                diy_dgy = T(0.5) * static_cast<T>(H_in);
            }
            int x0 = static_cast<int>(sycl::floor(ix));
            int y0 = static_cast<int>(sycl::floor(iy));
            int x1 = x0 + 1, y1 = y0 + 1;
            T wx1 = ix - static_cast<T>(x0);
            T wy1 = iy - static_cast<T>(y0);
            T wx0 = T(1) - wx1, wy0 = T(1) - wy1;
            T sum_dx = T(0), sum_dy = T(0);
            for (int c = 0; c < C; ++c) {
                const T go = go_ptr[((n * C + c) * H_out + h) * W_out + w];
                auto scatter = [&](int y, int x, T weight) {
                    if (y >= 0 && y < H_in && x >= 0 && x < W_in) {
                        make_atomic<T>(gi_ptr[((n * C + c) * H_in + y) * W_in + x]).fetch_add(go * weight);
                    }
                };
                auto fetch = [&](int y, int x) -> T {
                    if (y >= 0 && y < H_in && x >= 0 && x < W_in)
                        return input_ptr[((n * C + c) * H_in + y) * W_in + x];
                    return T(0);
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
            T sx = (pad_mode == 0 && !in_bounds_ix) ? T(0) : dix_dgx;
            T sy = (pad_mode == 0 && !in_bounds_iy) ? T(0) : diy_dgy;
            gg_ptr[grid_idx]     = sum_dx * sx;
            gg_ptr[grid_idx + 1] = sum_dy * sy;
        });
    } else if (mode == "nearest") {
        queue.parallel_for<GSNearestBwdK<T>>(sycl::range<1>(total), [=](sycl::id<1> idx_) {
            int idx = static_cast<int>(idx_);
            int w = idx % W_out;
            int h = (idx / W_out) % H_out;
            int n = idx / (H_out * W_out);
            int grid_idx = ((n * H_out + h) * W_out + w) * 2;
            T ix = gs_denormalize<T>(grid_ptr[grid_idx], W_in, align_corners);
            T iy = gs_denormalize<T>(grid_ptr[grid_idx + 1], H_in, align_corners);
            if (pad_mode == 1) {
                ix = sycl::fmin(sycl::fmax(ix, T(0)), static_cast<T>(W_in - 1));
                iy = sycl::fmin(sycl::fmax(iy, T(0)), static_cast<T>(H_in - 1));
            } else if (pad_mode == 2) {
                ix = gs_reflect_coord<T>(ix, W_in);
                iy = gs_reflect_coord<T>(iy, H_in);
            }
            int nx = static_cast<int>(sycl::round(ix));
            int ny = static_cast<int>(sycl::round(iy));
            gg_ptr[grid_idx]     = T(0);
            gg_ptr[grid_idx + 1] = T(0);
            if (ny >= 0 && ny < H_in && nx >= 0 && nx < W_in) {
                for (int c = 0; c < C; ++c) {
                    const T go = go_ptr[((n * C + c) * H_out + h) * W_out + w];
                    make_atomic<T>(gi_ptr[((n * C + c) * H_in + ny) * W_in + nx]).fetch_add(go);
                }
            }
        });
    } else if (mode == "bicubic") {
        queue.parallel_for<GSBicubicBwdK<T>>(sycl::range<1>(total), [=](sycl::id<1> idx_) {
            int idx = static_cast<int>(idx_);
            int w = idx % W_out;
            int h = (idx / W_out) % H_out;
            int n = idx / (H_out * W_out);
            int grid_idx = ((n * H_out + h) * W_out + w) * 2;
            T ix = gs_denormalize<T>(grid_ptr[grid_idx], W_in, align_corners);
            T iy = gs_denormalize<T>(grid_ptr[grid_idx + 1], H_in, align_corners);
            if (pad_mode == 1) {
                ix = sycl::fmin(sycl::fmax(ix, T(0)), static_cast<T>(W_in - 1));
                iy = sycl::fmin(sycl::fmax(iy, T(0)), static_cast<T>(H_in - 1));
            } else if (pad_mode == 2) {
                ix = gs_reflect_coord<T>(ix, W_in);
                iy = gs_reflect_coord<T>(iy, H_in);
            }
            T dix_dgx, diy_dgy;
            if (align_corners) {
                dix_dgx = T(0.5) * static_cast<T>(W_in - 1);
                diy_dgy = T(0.5) * static_cast<T>(H_in - 1);
            } else {
                dix_dgx = T(0.5) * static_cast<T>(W_in);
                diy_dgy = T(0.5) * static_cast<T>(H_in);
            }
            int ix_floor = static_cast<int>(sycl::floor(ix));
            int iy_floor = static_cast<int>(sycl::floor(iy));
            T tx = ix - static_cast<T>(ix_floor);
            T ty = iy - static_cast<T>(iy_floor);
            T wx[4], wy[4], dwx[4], dwy[4];
            cubic_weights<T>(tx, wx);
            cubic_weights<T>(ty, wy);
            cubic_dweights<T>(tx, dwx);
            cubic_dweights<T>(ty, dwy);
            T sum_dx = T(0), sum_dy = T(0);
            for (int c = 0; c < C; ++c) {
                const T go = go_ptr[((n * C + c) * H_out + h) * W_out + w];
                T dval_dix = T(0), dval_diy = T(0);
                for (int dy = -1; dy <= 2; ++dy) {
                    for (int dx = -1; dx <= 2; ++dx) {
                        const int yy = iy_floor + dy;
                        const int xx = ix_floor + dx;
                        const T weight = wy[dy + 1] * wx[dx + 1];
                        int yy_s = yy, xx_s = xx;
                        bool valid_s = true;
                        if (pad_mode == 0) {
                            if (yy < 0 || yy >= H_in || xx < 0 || xx >= W_in) valid_s = false;
                        } else {
                            yy_s = sycl::max(0, sycl::min(yy, H_in - 1));
                            xx_s = sycl::max(0, sycl::min(xx, W_in - 1));
                        }
                        if (valid_s) {
                            make_atomic<T>(gi_ptr[((n * C + c) * H_in + yy_s) * W_in + xx_s]).fetch_add(go * weight);
                        }
                        T v = T(0);
                        if (pad_mode == 0) {
                            if (yy >= 0 && yy < H_in && xx >= 0 && xx < W_in) {
                                v = input_ptr[((n * C + c) * H_in + yy) * W_in + xx];
                            }
                        } else {
                            int yy_f = sycl::max(0, sycl::min(yy, H_in - 1));
                            int xx_f = sycl::max(0, sycl::min(xx, W_in - 1));
                            v = input_ptr[((n * C + c) * H_in + yy_f) * W_in + xx_f];
                        }
                        dval_dix += wy[dy + 1] * dwx[dx + 1] * v;
                        dval_diy += dwy[dy + 1] * wx[dx + 1] * v;
                    }
                }
                sum_dx += go * dval_dix;
                sum_dy += go * dval_diy;
            }
            gg_ptr[grid_idx]     = sum_dx * dix_dgx;
            gg_ptr[grid_idx + 1] = sum_dy * diy_dgy;
        });
    } else {
        throw std::invalid_argument(
            "grid_sample_backward (OneAPI): unknown mode '" + mode + "'");
    }
    queue.wait_and_throw();
}

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

    int pad_mode = 0;
    if (padding_mode == "border") pad_mode = 1;
    else if (padding_mode == "reflection") pad_mode = 2;

    // Native FP64 path for all modes (no widen-narrow).
    if (input.dtype() == DType::Float64) {
        Tensor input_f64 = input.contiguous();
        Tensor grid_f64  = grid.to(DType::Float64).contiguous();
        Tensor output_f64(std::vector<int64_t>{N, C, H_out, W_out}, DType::Float64, input.device());
        run_grid_sample_forward<double>(queue, mode,
            get_data_ptr<const double>(input_f64), get_data_ptr<const double>(grid_f64),
            get_data_ptr<double>(output_f64),
            N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
        return output_f64;
    }

    Tensor input_f32 = input.to(DType::Float32);
    Tensor grid_f32  = grid.to(DType::Float32);
    Tensor output_f32(std::vector<int64_t>{N, C, H_out, W_out}, DType::Float32, input.device());
    run_grid_sample_forward<float>(queue, mode,
        get_data_ptr<const float>(input_f32), get_data_ptr<const float>(grid_f32),
        get_data_ptr<float>(output_f32),
        N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
    return output_f32.to(input.dtype());
}

auto grid_sample_backward_kernel(const Tensor& grad_output,
                                 const Tensor& input, const Tensor& grid,
                                 const std::string& mode,
                                 const std::string& padding_mode,
                                 bool align_corners, sycl::queue& queue)
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

    // Native FP64 path when EITHER input or grid is Float64 (matches CUDA/CPU),
    // so grad_grid is computed in double whenever the grid is double.
    if (input.dtype() == DType::Float64 || grid.dtype() == DType::Float64) {
        Tensor input_f64 = input.to(DType::Float64).contiguous();
        Tensor grid_f64  = grid.to(DType::Float64).contiguous();
        Tensor go_f64    = grad_output.to(DType::Float64).contiguous();
        Tensor gi_f64(std::vector<int64_t>{N, C, H_in, W_in},  DType::Float64, input.device());
        Tensor gg_f64(std::vector<int64_t>{N, H_out, W_out, 2}, DType::Float64, grid.device());
        queue.memset(gi_f64.data_ptr(), 0, gi_f64.numel() * sizeof(double)).wait();
        run_grid_sample_backward<double>(queue, mode,
            get_data_ptr<const double>(input_f64), get_data_ptr<const double>(grid_f64),
            get_data_ptr<const double>(go_f64),
            get_data_ptr<double>(gi_f64), get_data_ptr<double>(gg_f64),
            N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
        return {gi_f64.to(input.dtype()), gg_f64.to(grid.dtype())};
    }

    DType in_dt = input.dtype();
    DType gr_dt = grid.dtype();
    Tensor input_f32 = input.to(DType::Float32);
    Tensor grid_f32  = grid.to(DType::Float32);
    Tensor go_f32    = grad_output.to(DType::Float32);
    Tensor gi_f32(std::vector<int64_t>{N, C, H_in, W_in},  DType::Float32, input.device());
    Tensor gg_f32(std::vector<int64_t>{N, H_out, W_out, 2}, DType::Float32, grid.device());
    queue.memset(gi_f32.data_ptr(), 0, gi_f32.numel() * sizeof(float)).wait();
    run_grid_sample_backward<float>(queue, mode,
        get_data_ptr<const float>(input_f32), get_data_ptr<const float>(grid_f32),
        get_data_ptr<const float>(go_f32),
        get_data_ptr<float>(gi_f32), get_data_ptr<float>(gg_f32),
        N, C, H_in, W_in, H_out, W_out, pad_mode, align_corners);
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
            auto add = [&](int i, float v) { make_atomic<float>(t[i]).fetch_add(v); };
            add(0, dg_x * x_norm);
            add(1, dg_x * y_norm);
            add(2, dg_x);
            add(3, dg_y * x_norm);
            add(4, dg_y * y_norm);
            add(5, dg_y);
        });
    queue.wait_and_throw();
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
    queue.wait_and_throw();
    return grid;
}

}  // namespace oneapi
}  // namespace tenzor
