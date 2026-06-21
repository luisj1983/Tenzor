#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "gemm_optimized.hpp"
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#include "tenzor/backend/omp_thresholds.hpp"
#endif

// Intel MKL BLAS for the Float64 im2col-GEMM path (cblas_dgemm); the float path
// uses SIMD micro-kernels and without MKL Float64 falls back to gemm_local<double>
// (the scalar triple loop). That fallback is CORRECT — it accumulates in genuine
// `double` — only ~10x slower; it is not a precision regression.
#ifdef TENZOR_USE_MKL
#include <mkl.h>
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// Helper: Calculate output size for convolution dimension
// ============================================================================
static inline int64_t calc_out_size(int64_t in, int64_t kernel, int64_t stride,
                                     int64_t padding, int64_t dilation) {
    return (in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

// ============================================================================
// im3col: Convert 5D input (N,C,D,H,W) to 2D matrix for Conv3d
// Output: (batch * out_d * out_h * out_w, C * kD * kH * kW)
//
// Audit I5: per-axis stride/padding/dilation. Previously stride/padding/
// dilation were single scalars applied to all three spatial axes — making
// anisotropic Conv3d (e.g., temporal/spatial-asymmetric volumetric models)
// impossible to express on the CPU backend.
// ============================================================================
template<typename T>
static void im3col_cpu(
    const T* input,
    T* output,
    int64_t batch, int64_t channels,
    int64_t depth, int64_t height, int64_t width,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t out_d, int64_t out_h, int64_t out_w
) {
    // Overflow-checked int64 product: an unchecked product can wrap negative for
    // pathological-but-legal 5D shapes, truncating the loop below and leaving the
    // col buffer (and thus the conv output) partially uninitialized rather than
    // throwing. Guard each multiplicative step, mirroring calculate_output_size.
    auto checked_mul = [](int64_t a, int64_t b) -> int64_t {
        if (a != 0 && b != 0 &&
            (a > std::numeric_limits<int64_t>::max() / b ||
             a < std::numeric_limits<int64_t>::min() / b)) {
            throw std::invalid_argument("im3col: index/size product overflows int64");
        }
        return a * b;
    };
    int64_t col_cols = checked_mul(checked_mul(checked_mul(channels, kD), kH), kW);
    int64_t total = checked_mul(
        checked_mul(checked_mul(checked_mul(batch, out_d), out_h), out_w), col_cols);

    #pragma omp parallel for if(total > ::tenzor::OmpThresholds::medium())
    for (int64_t idx = 0; idx < total; ++idx) {
        int64_t tmp = idx;
        int64_t kw = tmp % kW; tmp /= kW;
        int64_t kh = tmp % kH; tmp /= kH;
        int64_t kd = tmp % kD; tmp /= kD;
        int64_t c  = tmp % channels; tmp /= channels;
        int64_t ow = tmp % out_w; tmp /= out_w;
        int64_t oh = tmp % out_h; tmp /= out_h;
        int64_t od = tmp % out_d; tmp /= out_d;
        int64_t b  = tmp;

        int64_t id = od * sD - pD + kd * dD;
        int64_t ih = oh * sH - pH + kh * dH;
        int64_t iw = ow * sW - pW + kw * dW;

        int64_t out_row = b * out_d * out_h * out_w + od * out_h * out_w + oh * out_w + ow;
        int64_t out_col = c * kD * kH * kW + kd * kH * kW + kh * kW + kw;
        int64_t out_idx = out_row * col_cols + out_col;

        if (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width) {
            int64_t in_idx = b * (channels * depth * height * width) +
                             c * (depth * height * width) +
                             id * (height * width) + ih * width + iw;
            output[out_idx] = input[in_idx];
        } else {
            output[out_idx] = T(0);
        }
    }
}

// ============================================================================
// col3im: Reverse of im3col for backward pass (audit I5: per-axis).
// ============================================================================
template<typename T>
static void col3im_cpu(
    const T* col,
    T* output,
    int64_t batch, int64_t channels,
    int64_t depth, int64_t height, int64_t width,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t out_d, int64_t out_h, int64_t out_w
) {
    int64_t output_size = batch * channels * depth * height * width;
    std::memset(output, 0, output_size * sizeof(T));

    int64_t col_cols = channels * kD * kH * kW;

    #pragma omp parallel for collapse(5) if(output_size > ::tenzor::OmpThresholds::medium())
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t id = 0; id < depth; ++id) {
                for (int64_t ih = 0; ih < height; ++ih) {
                    for (int64_t iw = 0; iw < width; ++iw) {
                        T sum = T(0);
                        for (int64_t kd = 0; kd < kD; ++kd) {
                            for (int64_t kh = 0; kh < kH; ++kh) {
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    int64_t id_s = id + pD - kd * dD;
                                    int64_t ih_s = ih + pH - kh * dH;
                                    int64_t iw_s = iw + pW - kw * dW;
                                    if (id_s % sD == 0 && ih_s % sH == 0 && iw_s % sW == 0) {
                                        int64_t od = id_s / sD;
                                        int64_t oh = ih_s / sH;
                                        int64_t ow = iw_s / sW;
                                        if (od >= 0 && od < out_d && oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                            int64_t col_row = b * out_d * out_h * out_w + od * out_h * out_w + oh * out_w + ow;
                                            int64_t col_col = c * kD * kH * kW + kd * kH * kW + kh * kW + kw;
                                            sum += col[col_row * col_cols + col_col];
                                        }
                                    }
                                }
                            }
                        }
                        int64_t out_idx = b * (channels * depth * height * width) +
                                          c * (depth * height * width) +
                                          id * (height * width) + ih * width + iw;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }
}

// ============================================================================
// GEMM helpers (reuse from conv2d.cpp via template)
// ============================================================================
template<typename T>
static void gemm_local(const T* A, const T* B, T* C,
                        int64_t M, int64_t N, int64_t K, bool transB) {
    #pragma omp parallel for collapse(2) if(M * N > ::tenzor::OmpThresholds::matmul())
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            T sum = T(0);
            if (transB) {
                for (int64_t k = 0; k < K; ++k)
                    sum += A[i * K + k] * B[j * K + k];
            } else {
                for (int64_t k = 0; k < K; ++k)
                    sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Float32 specialization using optimized GEMM
template<>
void gemm_local<float>(const float* A, const float* B, float* C,
                        int64_t M, int64_t N, int64_t K, bool transB) {
    if (transB) {
        gemm::gemm_transB_optimized(A, B, C, M, N, K);
    } else {
        gemm::gemm_optimized(A, B, C, M, N, K);
    }
}

#ifdef TENZOR_USE_MKL
// Float64 specialization using MKL cblas_dgemm. A is (M, K) row-major; for
// transB, B is (N, K) and we compute A @ B^T, else B is (K, N) and A @ B.
template<>
void gemm_local<double>(const double* A, const double* B, double* C,
                         int64_t M, int64_t N, int64_t K, bool transB) {
    // The LP64 MKL interface takes int dimensions/leading-dims. If any of
    // M/N/K (which double as the leading dimensions here) exceeds INT_MAX the
    // static_cast<int> would silently truncate, producing a wrong-shaped GEMM
    // with out-of-bounds access. Fall back to an int64-indexed scalar GEMM in
    // that (extreme, multi-GB 3D volume) case so the result stays correct.
    // Mirrors conv2d.cpp's gemm_cpu<double> guard.
    constexpr int64_t kIntMax = static_cast<int64_t>(std::numeric_limits<int>::max());
    if (M > kIntMax || N > kIntMax || K > kIntMax) {
        #pragma omp parallel for collapse(2) if(M * N > ::tenzor::OmpThresholds::matmul())
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                double sum = 0.0;
                if (transB) {
                    for (int64_t k = 0; k < K; ++k) {
                        sum += A[i * K + k] * B[j * K + k];
                    }
                } else {
                    for (int64_t k = 0; k < K; ++k) {
                        sum += A[i * K + k] * B[k * N + j];
                    }
                }
                C[i * N + j] = sum;
            }
        }
        return;
    }
    if (transB) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0, A, static_cast<int>(K), B, static_cast<int>(K),
                    0.0, C, static_cast<int>(N));
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0, A, static_cast<int>(K), B, static_cast<int>(N),
                    0.0, C, static_cast<int>(N));
    }
}
#endif // TENZOR_USE_MKL

// ============================================================================
// Conv3d Forward Implementation (template) — audit I5: per-axis.
// ============================================================================
template<typename T>
static void conv3d_forward_impl(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    Tensor& output,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups
) {
    auto is = input.shape();   // (N, C_in, D, H, W)
    auto ws = weight.shape();  // (C_out, C_in/g, kD, kH, kW)
    auto os = output.shape();  // (N, C_out, oD, oH, oW)

    int64_t batch = is[0];
    int64_t in_channels = is[1];
    int64_t depth = is[2], height = is[3], width = is[4];
    int64_t out_channels = ws[0];
    int64_t ic_per_g = ws[1];
    int64_t kD = ws[2], kH = ws[3], kW = ws[4];
    int64_t out_d = os[2], out_h = os[3], out_w = os[4];
    int64_t oc_per_g = out_channels / groups;

    std::memset(output.data<T>(), 0, output.numel() * sizeof(T));

    // 1x1x1 fast path
    if (kD == 1 && kH == 1 && kW == 1 &&
        sD == 1 && sH == 1 && sW == 1 &&
        pD == 0 && pH == 0 && pW == 0 &&
        dD == 1 && dH == 1 && dW == 1) {
        const T* in_data = input.data<T>();
        const T* w_data = weight.data<T>();
        T* out_data = output.data<T>();
        int64_t spatial = depth * height * width;

        for (int64_t g = 0; g < groups; ++g) {
            int64_t in_start = g * ic_per_g;
            int64_t out_start = g * oc_per_g;
            #pragma omp parallel for if(batch * spatial > ::tenzor::OmpThresholds::medium())
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t s = 0; s < spatial; ++s) {
                    for (int64_t oc = 0; oc < oc_per_g; ++oc) {
                        T sum{};
                        for (int64_t ic = 0; ic < ic_per_g; ++ic) {
                            sum += in_data[b * in_channels * spatial + (in_start + ic) * spatial + s]
                                 * w_data[(out_start + oc) * ic_per_g + ic];
                        }
                        out_data[b * out_channels * spatial + (out_start + oc) * spatial + s] = sum;
                    }
                }
            }
        }
    } else {
        // General path: im3col + GEMM
        for (int64_t g = 0; g < groups; ++g) {
            int64_t in_start = g * ic_per_g;
            int64_t out_start = g * oc_per_g;

            int64_t col_rows = batch * out_d * out_h * out_w;
            int64_t col_cols = ic_per_g * kD * kH * kW;
            std::vector<T> col_buf(col_rows * col_cols);

            int64_t col_per_batch = out_d * out_h * out_w * col_cols;
            for (int64_t b = 0; b < batch; ++b) {
                const T* in_ptr = input.data<T>() + (b * in_channels + in_start) * depth * height * width;
                im3col_cpu(in_ptr, col_buf.data() + b * col_per_batch,
                           1, ic_per_g, depth, height, width,
                           kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW,
                           out_d, out_h, out_w);
            }

            // GEMM: col_buf (M x K) @ weight^T (N x K) -> gemm_out (M x N)
            int64_t M = col_rows, K = col_cols, N = oc_per_g;
            const T* w_ptr = weight.data<T>() + out_start * ic_per_g * kD * kH * kW;
            std::vector<T> gemm_out(M * N);
            gemm_local(col_buf.data(), w_ptr, gemm_out.data(), M, N, K, true);

            // Scatter from (batch * oD * oH * oW, oc_per_g) to NCDHW
            T* out_data = output.data<T>();
            #pragma omp parallel for collapse(5) if(batch * oc_per_g * out_d * out_h * out_w > ::tenzor::OmpThresholds::medium())
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t c = 0; c < oc_per_g; ++c) {
                    for (int64_t od = 0; od < out_d; ++od) {
                        for (int64_t oh = 0; oh < out_h; ++oh) {
                            for (int64_t ow = 0; ow < out_w; ++ow) {
                                int64_t gemm_row = b * out_d * out_h * out_w + od * out_h * out_w + oh * out_w + ow;
                                int64_t gemm_idx = gemm_row * oc_per_g + c;
                                int64_t ncdhw = b * out_channels * out_d * out_h * out_w +
                                                (c + out_start) * out_d * out_h * out_w +
                                                od * out_h * out_w + oh * out_w + ow;
                                out_data[ncdhw] = gemm_out[gemm_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias
    if (bias) {
        const T* bias_data = bias->data<T>();
        T* out_data = output.data<T>();
        int64_t spatial = out_d * out_h * out_w;
        #pragma omp parallel for collapse(3) if(batch * out_channels * spatial > ::tenzor::OmpThresholds::medium())
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < out_channels; ++c) {
                for (int64_t s = 0; s < spatial; ++s) {
                    out_data[b * out_channels * spatial + c * spatial + s] += bias_data[c];
                }
            }
        }
    }
}

// ============================================================================
// Conv3d Backward Input Implementation (audit I5: per-axis)
// ============================================================================
template<typename T>
static void conv3d_backward_input_impl(
    const Tensor& grad_output, const Tensor& weight, Tensor& grad_input,
    const std::vector<int64_t>& input_shape,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups
) {
    auto ws = weight.shape();
    auto gs = grad_output.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t depth = input_shape[2], height = input_shape[3], width = input_shape[4];
    int64_t out_channels = ws[0], ic_per_g = ws[1];
    int64_t kD = ws[2], kH = ws[3], kW = ws[4];
    int64_t out_d = gs[2], out_h = gs[3], out_w = gs[4];
    int64_t oc_per_g = out_channels / groups;

    std::memset(grad_input.data<T>(), 0, grad_input.numel() * sizeof(T));

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * ic_per_g;
        int64_t out_start = g * oc_per_g;

        // Pack grad_output for this group: (batch * oD * oH * oW, oc_per_g)
        int64_t M = batch * out_d * out_h * out_w;
        int64_t K = oc_per_g;
        int64_t N = ic_per_g * kD * kH * kW;

        std::vector<T> grad_col(M * K);
        const T* go_data = grad_output.data<T>();
        #pragma omp parallel for if(M * K > ::tenzor::OmpThresholds::medium())
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t od = 0; od < out_d; ++od) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        int64_t row = b * out_d * out_h * out_w + od * out_h * out_w + oh * out_w + ow;
                        for (int64_t c = 0; c < oc_per_g; ++c) {
                            grad_col[row * K + c] = go_data[b * out_channels * out_d * out_h * out_w +
                                                            (c + out_start) * out_d * out_h * out_w +
                                                            od * out_h * out_w + oh * out_w + ow];
                        }
                    }
                }
            }
        }

        // GEMM: grad_col (M x K) @ weight (K x N) -> col_buf (M x N)
        // weight is (oc_per_g, ic_per_g * kD * kH * kW), no transpose
        const T* w_ptr = weight.data<T>() + out_start * ic_per_g * kD * kH * kW;
        std::vector<T> col_buf(M * N);
        gemm_local(grad_col.data(), w_ptr, col_buf.data(), M, N, K, false);

        // col3im to accumulate into grad_input
        int64_t col_per_batch = out_d * out_h * out_w * N;
        for (int64_t b = 0; b < batch; ++b) {
            T* gi_ptr = grad_input.data<T>() + (b * in_channels + in_start) * depth * height * width;
            // Accumulate from col_buf for this batch
            std::vector<T> tmp(ic_per_g * depth * height * width, T(0));
            col3im_cpu(col_buf.data() + b * col_per_batch, tmp.data(),
                       1, ic_per_g, depth, height, width,
                       kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW,
                       out_d, out_h, out_w);
            // Add to grad_input
            int64_t n = ic_per_g * depth * height * width;
            for (int64_t i = 0; i < n; ++i) {
                gi_ptr[i] += tmp[i];
            }
        }
    }
}

// ============================================================================
// Conv3d Backward Weight Implementation (audit I5: per-axis)
// ============================================================================
template<typename T>
static void conv3d_backward_weight_impl(
    const Tensor& grad_output, const Tensor& input, Tensor& grad_weight,
    const std::vector<int64_t>& weight_shape,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups
) {
    auto is = input.shape();
    auto gs = grad_output.shape();

    int64_t batch = is[0], in_channels = is[1];
    int64_t depth = is[2], height = is[3], width = is[4];
    int64_t out_channels = weight_shape[0], ic_per_g = weight_shape[1];
    int64_t kD = weight_shape[2], kH = weight_shape[3], kW = weight_shape[4];
    int64_t out_d = gs[2], out_h = gs[3], out_w = gs[4];
    int64_t oc_per_g = out_channels / groups;

    std::memset(grad_weight.data<T>(), 0, grad_weight.numel() * sizeof(T));

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * ic_per_g;
        int64_t out_start = g * oc_per_g;

        // im3col the input
        int64_t col_rows = batch * out_d * out_h * out_w;
        int64_t col_cols = ic_per_g * kD * kH * kW;
        std::vector<T> col_buf(col_rows * col_cols);

        int64_t col_per_batch = out_d * out_h * out_w * col_cols;
        for (int64_t b = 0; b < batch; ++b) {
            const T* in_ptr = input.data<T>() + (b * in_channels + in_start) * depth * height * width;
            im3col_cpu(in_ptr, col_buf.data() + b * col_per_batch,
                       1, ic_per_g, depth, height, width,
                       kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW,
                       out_d, out_h, out_w);
        }

        // Pack grad_output: (batch * oD * oH * oW, oc_per_g)
        int64_t M = col_rows;
        std::vector<T> grad_col(M * oc_per_g);
        const T* go_data = grad_output.data<T>();
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t od = 0; od < out_d; ++od) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        int64_t row = b * out_d * out_h * out_w + od * out_h * out_w + oh * out_w + ow;
                        for (int64_t c = 0; c < oc_per_g; ++c) {
                            grad_col[row * oc_per_g + c] =
                                go_data[b * out_channels * out_d * out_h * out_w +
                                        (c + out_start) * out_d * out_h * out_w +
                                        od * out_h * out_w + oh * out_w + ow];
                        }
                    }
                }
            }
        }

        // GEMM: grad_col^T (oc_per_g x M) @ col_buf (M x col_cols) -> grad_w_chunk (oc_per_g x col_cols)
        // Use transA: A=(M, oc_per_g), B=(M, col_cols), C = A^T @ B
        std::vector<T> gw_chunk(oc_per_g * col_cols, T(0));
        // Manual transA GEMM
        #pragma omp parallel for collapse(2) if(oc_per_g * col_cols > ::tenzor::OmpThresholds::matmul())
        for (int64_t i = 0; i < oc_per_g; ++i) {
            for (int64_t j = 0; j < col_cols; ++j) {
                T sum = T(0);
                for (int64_t k = 0; k < M; ++k) {
                    sum += grad_col[k * oc_per_g + i] * col_buf[k * col_cols + j];
                }
                gw_chunk[i * col_cols + j] = sum;
            }
        }

        // Copy to grad_weight
        T* gw_data = grad_weight.data<T>() + out_start * ic_per_g * kD * kH * kW;
        std::memcpy(gw_data, gw_chunk.data(), oc_per_g * col_cols * sizeof(T));
    }
}

// ============================================================================
// Public Kernel Functions
// ============================================================================

// Audit I5: per-axis public overload. Scalar overload delegates by passing
// the same value for all three spatial dims.
auto conv3d_forward_kernel(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups
) -> Tensor {
    auto is = input.shape();
    auto ws = weight.shape();

    int64_t out_d = calc_out_size(is[2], ws[2], sD, pD, dD);
    int64_t out_h = calc_out_size(is[3], ws[3], sH, pH, dH);
    int64_t out_w = calc_out_size(is[4], ws[4], sW, pW, dW);

    if (out_d <= 0 || out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument(
            "Invalid Conv3d configuration: output dimensions are non-positive (out_d=" +
            std::to_string(out_d) + ", out_h=" + std::to_string(out_h) +
            ", out_w=" + std::to_string(out_w) +
            "); kernel/dilation too large for the padded input");
    }

    Tensor output({is[0], ws[0], out_d, out_h, out_w}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        conv3d_forward_impl<float>(input, weight, bias, output, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
    } else if (input.dtype() == DType::Float64) {
        conv3d_forward_impl<double>(input, weight, bias, output, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto in_f32 = input.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        Tensor b_f32;
        const Tensor* b_ptr = nullptr;
        if (bias) { b_f32 = bias->to(DType::Float32); b_ptr = &b_f32; }
        Tensor out_f32({is[0], ws[0], out_d, out_h, out_w}, DType::Float32, input.device());
        conv3d_forward_impl<float>(in_f32, w_f32, b_ptr, out_f32, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
        output = out_f32.to(orig);
    } else {
        throw std::runtime_error("Unsupported dtype for conv3d_forward");
    }
    return output;
}

// Scalar overload — delegates to per-axis with identical D/H/W values.
auto conv3d_forward_kernel(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups
) -> Tensor {
    return conv3d_forward_kernel(input, weight, bias,
        stride, stride, stride, padding, padding, padding,
        dilation, dilation, dilation, groups);
}

auto conv3d_backward_input_kernel(
    const Tensor& grad_output, const Tensor& weight,
    const std::vector<int64_t>& input_shape,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups
) -> Tensor {
    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        conv3d_backward_input_impl<float>(grad_output, weight, grad_input, input_shape, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
    } else if (grad_output.dtype() == DType::Float64) {
        conv3d_backward_input_impl<double>(grad_output, weight, grad_input, input_shape, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        DType orig = grad_output.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        Tensor gi_f32(input_shape, DType::Float32, grad_output.device());
        conv3d_backward_input_impl<float>(go_f32, w_f32, gi_f32, input_shape, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
        grad_input = gi_f32.to(orig);
    } else {
        throw std::runtime_error("Unsupported dtype for conv3d_backward_input");
    }
    return grad_input;
}

auto conv3d_backward_input_kernel(
    const Tensor& grad_output, const Tensor& weight,
    const std::vector<int64_t>& input_shape,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups
) -> Tensor {
    return conv3d_backward_input_kernel(grad_output, weight, input_shape,
        stride, stride, stride, padding, padding, padding,
        dilation, dilation, dilation, groups);
}

auto conv3d_backward_weight_kernel(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& weight_shape,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups
) -> Tensor {
    Tensor grad_weight(weight_shape, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        conv3d_backward_weight_impl<float>(grad_output, input, grad_weight, weight_shape, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
    } else if (grad_output.dtype() == DType::Float64) {
        conv3d_backward_weight_impl<double>(grad_output, input, grad_weight, weight_shape, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        DType orig = grad_output.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        Tensor gw_f32(weight_shape, DType::Float32, grad_output.device());
        conv3d_backward_weight_impl<float>(go_f32, in_f32, gw_f32, weight_shape, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
        grad_weight = gw_f32.to(orig);
    } else {
        throw std::runtime_error("Unsupported dtype for conv3d_backward_weight");
    }
    return grad_weight;
}

auto conv3d_backward_weight_kernel(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& weight_shape,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups
) -> Tensor {
    return conv3d_backward_weight_kernel(grad_output, input, weight_shape,
        stride, stride, stride, padding, padding, padding,
        dilation, dilation, dilation, groups);
}

auto conv3d_backward_bias_kernel(const Tensor& grad_output) -> Tensor {
    auto shape = grad_output.shape();
    int64_t batch = shape[0], out_channels = shape[1];
    int64_t spatial = shape[2] * shape[3] * shape[4];

    Tensor grad_bias({out_channels}, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        const float* data = grad_output.data<float>();
        float* gb = grad_bias.data<float>();
        for (int64_t c = 0; c < out_channels; ++c) {
            float sum = 0.0f;
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t s = 0; s < spatial; ++s) {
                    sum += data[b * out_channels * spatial + c * spatial + s];
                }
            }
            gb[c] = sum;
        }
    } else if (grad_output.dtype() == DType::Float64) {
        const double* data = grad_output.data<double>();
        double* gb = grad_bias.data<double>();
        for (int64_t c = 0; c < out_channels; ++c) {
            double sum = 0.0;
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t s = 0; s < spatial; ++s) {
                    sum += data[b * out_channels * spatial + c * spatial + s];
                }
            }
            gb[c] = sum;
        }
    } else {
        // Half-precision: compute in Float32
        auto go_f32 = grad_output.to(DType::Float32);
        auto gb_f32 = conv3d_backward_bias_kernel(go_f32);
        grad_bias = gb_f32.to(grad_output.dtype());
    }
    return grad_bias;
}

// ============================================================================
// ConvTranspose3d: Transposed 3D convolution (deconvolution)
// Forward: col3im(GEMM(weight^T, input_col)) + bias
// ============================================================================

static inline int64_t calc_transpose_out_size(int64_t in, int64_t kernel, int64_t stride,
                                               int64_t padding, int64_t output_padding,
                                               int64_t dilation) {
    return (in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1;
}

template<typename T>
static void conv_transpose3d_forward_impl(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    Tensor& output,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    [[maybe_unused]] int64_t opD, [[maybe_unused]] int64_t opH, [[maybe_unused]] int64_t opW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups)
{
    // Audit I5: per-axis stride/padding/output_padding/dilation.
    auto in_shape = input.shape();
    auto w_shape = weight.shape();
    int64_t batch = in_shape[0];
    int64_t in_channels = in_shape[1];
    int64_t in_d = in_shape[2], in_h = in_shape[3], in_w = in_shape[4];
    int64_t out_channels_per_group = w_shape[1];
    int64_t kD = w_shape[2], kH = w_shape[3], kW = w_shape[4];
    int64_t out_channels = out_channels_per_group * groups;

    auto out_shape = output.shape();
    int64_t out_d = out_shape[2], out_h = out_shape[3], out_w = out_shape[4];

    int64_t in_channels_per_group = in_channels / groups;
    int64_t col_h = out_channels_per_group * kD * kH * kW;
    int64_t col_w = in_d * in_h * in_w;

    const T* in_ptr = input.data<T>();
    const T* w_ptr = weight.data<T>();
    T* out_ptr = output.data<T>();

    std::memset(out_ptr, 0, batch * out_channels * out_d * out_h * out_w * sizeof(T));

    int64_t in_spatial = in_d * in_h * in_w;
    int64_t out_spatial = out_d * out_h * out_w;

    std::vector<T> col_buf(col_h * col_w);
    std::vector<T> wt_buf(col_h * in_channels_per_group);

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t g = 0; g < groups; ++g) {
            const T* in_g = in_ptr + b * in_channels * in_spatial + g * in_channels_per_group * in_spatial;
            const T* w_g = w_ptr + g * in_channels_per_group * out_channels_per_group * kD * kH * kW;

            for (int64_t i = 0; i < in_channels_per_group; ++i)
                for (int64_t j = 0; j < col_h; ++j)
                    wt_buf[j * in_channels_per_group + i] = w_g[i * col_h + j];

            gemm_local<T>(wt_buf.data(), in_g, col_buf.data(), col_h, col_w, in_channels_per_group, false);

            T* out_g = out_ptr + b * out_channels * out_spatial + g * out_channels_per_group * out_spatial;
            for (int64_t c = 0; c < out_channels_per_group; ++c) {
                for (int64_t kd = 0; kd < kD; ++kd) {
                    for (int64_t kh = 0; kh < kH; ++kh) {
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t col_row = ((c * kD + kd) * kH + kh) * kW + kw;
                            for (int64_t id = 0; id < in_d; ++id) {
                                int64_t od = id * sD - pD + kd * dD;
                                if (od < 0 || od >= out_d) continue;
                                for (int64_t ih = 0; ih < in_h; ++ih) {
                                    int64_t oh = ih * sH - pH + kh * dH;
                                    if (oh < 0 || oh >= out_h) continue;
                                    for (int64_t iw = 0; iw < in_w; ++iw) {
                                        int64_t ow = iw * sW - pW + kw * dW;
                                        if (ow < 0 || ow >= out_w) continue;
                                        int64_t col_col = (id * in_h + ih) * in_w + iw;
                                        out_g[c * out_spatial + (od * out_h + oh) * out_w + ow] +=
                                            col_buf[col_row * col_w + col_col];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (bias) {
        const T* bias_ptr = bias->data<T>();
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < out_channels; ++c) {
                T bv = bias_ptr[c];
                T* dst = out_ptr + b * out_channels * out_spatial + c * out_spatial;
                for (int64_t s = 0; s < out_spatial; ++s) dst[s] += bv;
            }
        }
    }
}

// ConvTranspose3d backward input: the "forward" of a regular conv3d
template<typename T>
static void conv_transpose3d_backward_input_impl(
    const Tensor& grad_output, const Tensor& weight,
    Tensor& grad_input,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    [[maybe_unused]] int64_t opD, [[maybe_unused]] int64_t opH, [[maybe_unused]] int64_t opW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups)
{
    auto go_shape = grad_output.shape();
    auto w_shape = weight.shape();
    auto gi_shape = grad_input.shape();

    int64_t batch = go_shape[0];
    int64_t out_channels = go_shape[1];
    int64_t out_d = go_shape[2], out_h = go_shape[3], out_w = go_shape[4];

    int64_t in_channels = gi_shape[1];
    int64_t in_d = gi_shape[2], in_h = gi_shape[3], in_w = gi_shape[4];

    int64_t in_channels_per_group = in_channels / groups;
    int64_t out_channels_per_group = out_channels / groups;
    int64_t kD = w_shape[2], kH = w_shape[3], kW = w_shape[4];

    int64_t in_spatial = in_d * in_h * in_w;
    int64_t out_spatial = out_d * out_h * out_w;
    int64_t col_h = out_channels_per_group * kD * kH * kW;
    int64_t col_w = in_d * in_h * in_w;

    const T* go_ptr = grad_output.data<T>();
    const T* w_ptr = weight.data<T>();
    T* gi_ptr = grad_input.data<T>();

    std::memset(gi_ptr, 0, batch * in_channels * in_spatial * sizeof(T));

    // Backward of transposed conv = forward conv
    // For each (batch, group): im3col(grad_output) then GEMM with weight
    std::vector<T> col_buf(col_h * col_w);

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t g = 0; g < groups; ++g) {
            // im3col of grad_output w.r.t. output spatial dims
            const T* go_g = go_ptr + b * out_channels * out_spatial + g * out_channels_per_group * out_spatial;

            // Build columns from grad_output
            for (int64_t c = 0; c < out_channels_per_group; ++c) {
                for (int64_t kd = 0; kd < kD; ++kd) {
                    for (int64_t kh = 0; kh < kH; ++kh) {
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t col_row = ((c * kD + kd) * kH + kh) * kW + kw;
                            for (int64_t id = 0; id < in_d; ++id) {
                                int64_t od = id * sD - pD + kd * dD;
                                for (int64_t ih = 0; ih < in_h; ++ih) {
                                    int64_t oh = ih * sH - pH + kh * dH;
                                    for (int64_t iw = 0; iw < in_w; ++iw) {
                                        int64_t ow = iw * sW - pW + kw * dW;
                                        int64_t col_col = (id * in_h + ih) * in_w + iw;
                                        if (od >= 0 && od < out_d && oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                            col_buf[col_row * col_w + col_col] =
                                                go_g[c * out_spatial + (od * out_h + oh) * out_w + ow];
                                        } else {
                                            col_buf[col_row * col_w + col_col] = T(0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // GEMM: weight (in_ch_per_g, col_h) * col_buf (col_h, col_w) → grad_input_g (in_ch_per_g, col_w)
            const T* w_g = w_ptr + g * in_channels_per_group * out_channels_per_group * kD * kH * kW;
            T* gi_g = gi_ptr + b * in_channels * in_spatial + g * in_channels_per_group * in_spatial;

            gemm_local<T>(w_g, col_buf.data(), gi_g, in_channels_per_group, col_w, col_h, false);
        }
    }
}

// ConvTranspose3d backward weight (audit I5: per-axis)
template<typename T>
static void conv_transpose3d_backward_weight_impl(
    const Tensor& grad_output, const Tensor& input,
    Tensor& grad_weight,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    [[maybe_unused]] int64_t opD, [[maybe_unused]] int64_t opH, [[maybe_unused]] int64_t opW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups)
{
    auto go_shape = grad_output.shape();
    auto in_shape = input.shape();
    auto w_shape = grad_weight.shape();

    int64_t batch = go_shape[0];
    int64_t out_channels = go_shape[1];
    int64_t out_d = go_shape[2], out_h = go_shape[3], out_w = go_shape[4];

    int64_t in_channels = in_shape[1];
    int64_t in_d = in_shape[2], in_h = in_shape[3], in_w = in_shape[4];

    int64_t in_channels_per_group = in_channels / groups;
    int64_t out_channels_per_group = out_channels / groups;
    int64_t kD = w_shape[2], kH = w_shape[3], kW = w_shape[4];

    int64_t in_spatial = in_d * in_h * in_w;
    int64_t out_spatial = out_d * out_h * out_w;
    int64_t col_h = out_channels_per_group * kD * kH * kW;
    int64_t col_w = in_d * in_h * in_w;

    const T* go_ptr = grad_output.data<T>();
    const T* in_ptr = input.data<T>();
    T* gw_ptr = grad_weight.data<T>();

    std::memset(gw_ptr, 0, in_channels * out_channels_per_group * kD * kH * kW * sizeof(T));

    std::vector<T> col_buf(col_h * col_w);

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t g = 0; g < groups; ++g) {
            // Build columns from grad_output (same as backward input)
            const T* go_g = go_ptr + b * out_channels * out_spatial + g * out_channels_per_group * out_spatial;

            for (int64_t c = 0; c < out_channels_per_group; ++c) {
                for (int64_t kd = 0; kd < kD; ++kd) {
                    for (int64_t kh = 0; kh < kH; ++kh) {
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t col_row = ((c * kD + kd) * kH + kh) * kW + kw;
                            for (int64_t id = 0; id < in_d; ++id) {
                                int64_t od = id * sD - pD + kd * dD;
                                for (int64_t ih = 0; ih < in_h; ++ih) {
                                    int64_t oh = ih * sH - pH + kh * dH;
                                    for (int64_t iw = 0; iw < in_w; ++iw) {
                                        int64_t ow = iw * sW - pW + kw * dW;
                                        int64_t col_col = (id * in_h + ih) * in_w + iw;
                                        if (od >= 0 && od < out_d && oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                            col_buf[col_row * col_w + col_col] =
                                                go_g[c * out_spatial + (od * out_h + oh) * out_w + ow];
                                        } else {
                                            col_buf[col_row * col_w + col_col] = T(0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // GEMM: input_g (in_ch_per_g, in_spatial) * col_buf^T (in_spatial, col_h) → temp (in_ch_per_g, col_h)
            const T* in_g = in_ptr + b * in_channels * in_spatial + g * in_channels_per_group * in_spatial;
            T* gw_g = gw_ptr + g * in_channels_per_group * out_channels_per_group * kD * kH * kW;

            // transB=true: col_buf is (col_h, col_w) but we need (col_w, col_h) = col_buf^T
            std::vector<T> temp_gw(in_channels_per_group * col_h);
            gemm_local<T>(in_g, col_buf.data(), temp_gw.data(), in_channels_per_group, col_h, col_w, true);

            // Accumulate into grad_weight
            int64_t gw_size = in_channels_per_group * col_h;
            for (int64_t i = 0; i < gw_size; ++i)
                gw_g[i] += temp_gw[i];
        }
    }
}

// Public kernel functions for ConvTranspose3d

// Audit I5: per-axis public kernel. Scalar overload (further down) delegates.
auto conv_transpose3d_forward_kernel(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t opD, int64_t opH, int64_t opW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups
) -> Tensor {
    auto in_shape = input.shape();
    auto w_shape = weight.shape();

    int64_t batch = in_shape[0];
    int64_t out_channels_per_group = w_shape[1];
    int64_t out_channels = out_channels_per_group * groups;
    int64_t kD = w_shape[2], kH = w_shape[3], kW = w_shape[4];

    int64_t out_d = calc_transpose_out_size(in_shape[2], kD, sD, pD, opD, dD);
    int64_t out_h = calc_transpose_out_size(in_shape[3], kH, sH, pH, opH, dH);
    int64_t out_w = calc_transpose_out_size(in_shape[4], kW, sW, pW, opW, dW);

    Tensor output({batch, out_channels, out_d, out_h, out_w}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        conv_transpose3d_forward_impl<float>(input, weight, bias, output, sD, sH, sW, pD, pH, pW, opD, opH, opW, dD, dH, dW, groups);
    } else if (input.dtype() == DType::Float64) {
        conv_transpose3d_forward_impl<double>(input, weight, bias, output, sD, sH, sW, pD, pH, pW, opD, opH, opW, dD, dH, dW, groups);
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto in_f32 = input.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        const Tensor* b_f32_ptr = nullptr;
        Tensor b_f32;
        if (bias) { b_f32 = bias->to(DType::Float32); b_f32_ptr = &b_f32; }
        Tensor out_f32({batch, out_channels, out_d, out_h, out_w}, DType::Float32, input.device());
        conv_transpose3d_forward_impl<float>(in_f32, w_f32, b_f32_ptr, out_f32, sD, sH, sW, pD, pH, pW, opD, opH, opW, dD, dH, dW, groups);
        output = out_f32.to(orig);
    } else {
        throw std::runtime_error("Unsupported dtype for conv_transpose3d_forward");
    }
    return output;
}

// Scalar overload — delegates with identical D/H/W values.
auto conv_transpose3d_forward_kernel(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    int64_t stride, int64_t padding, int64_t output_padding,
    int64_t dilation, int64_t groups
) -> Tensor {
    return conv_transpose3d_forward_kernel(input, weight, bias,
        stride, stride, stride, padding, padding, padding,
        output_padding, output_padding, output_padding,
        dilation, dilation, dilation, groups);
}

auto conv_transpose3d_backward_input_kernel(
    const Tensor& grad_output, const Tensor& weight,
    const std::vector<int64_t>& input_shape,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t opD, int64_t opH, int64_t opW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups
) -> Tensor {
    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        conv_transpose3d_backward_input_impl<float>(grad_output, weight, grad_input, sD, sH, sW, pD, pH, pW, opD, opH, opW, dD, dH, dW, groups);
    } else if (grad_output.dtype() == DType::Float64) {
        conv_transpose3d_backward_input_impl<double>(grad_output, weight, grad_input, sD, sH, sW, pD, pH, pW, opD, opH, opW, dD, dH, dW, groups);
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        DType orig = grad_output.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        Tensor gi_f32(input_shape, DType::Float32, grad_output.device());
        conv_transpose3d_backward_input_impl<float>(go_f32, w_f32, gi_f32, sD, sH, sW, pD, pH, pW, opD, opH, opW, dD, dH, dW, groups);
        grad_input = gi_f32.to(orig);
    } else {
        throw std::runtime_error("Unsupported dtype for conv_transpose3d_backward_input");
    }
    return grad_input;
}

auto conv_transpose3d_backward_input_kernel(
    const Tensor& grad_output, const Tensor& weight,
    const std::vector<int64_t>& input_shape,
    int64_t stride, int64_t padding, int64_t output_padding,
    int64_t dilation, int64_t groups
) -> Tensor {
    return conv_transpose3d_backward_input_kernel(grad_output, weight, input_shape,
        stride, stride, stride, padding, padding, padding,
        output_padding, output_padding, output_padding,
        dilation, dilation, dilation, groups);
}

auto conv_transpose3d_backward_weight_kernel(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& weight_shape,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t opD, int64_t opH, int64_t opW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups
) -> Tensor {
    Tensor grad_weight(weight_shape, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        conv_transpose3d_backward_weight_impl<float>(grad_output, input, grad_weight, sD, sH, sW, pD, pH, pW, opD, opH, opW, dD, dH, dW, groups);
    } else if (grad_output.dtype() == DType::Float64) {
        conv_transpose3d_backward_weight_impl<double>(grad_output, input, grad_weight, sD, sH, sW, pD, pH, pW, opD, opH, opW, dD, dH, dW, groups);
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        DType orig = grad_output.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        Tensor gw_f32(weight_shape, DType::Float32, grad_output.device());
        conv_transpose3d_backward_weight_impl<float>(go_f32, in_f32, gw_f32, sD, sH, sW, pD, pH, pW, opD, opH, opW, dD, dH, dW, groups);
        grad_weight = gw_f32.to(orig);
    } else {
        throw std::runtime_error("Unsupported dtype for conv_transpose3d_backward_weight");
    }
    return grad_weight;
}

auto conv_transpose3d_backward_weight_kernel(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& weight_shape,
    int64_t stride, int64_t padding, int64_t output_padding,
    int64_t dilation, int64_t groups
) -> Tensor {
    return conv_transpose3d_backward_weight_kernel(grad_output, input, weight_shape,
        stride, stride, stride, padding, padding, padding,
        output_padding, output_padding, output_padding,
        dilation, dilation, dilation, groups);
}

} // namespace cpu
} // namespace tenzor
