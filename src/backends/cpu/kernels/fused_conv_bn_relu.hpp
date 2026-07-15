/**
 * @file fused_conv_bn_relu.hpp
 * @brief Fused Conv2d + BatchNorm + ReLU kernel
 *
 * Key optimizations:
 * - Single-pass computation reduces memory bandwidth by ~3x
 * - BatchNorm parameters folded into conv weights (inference mode)
 * - SIMD-accelerated computation
 * - Uses buffer pool for temporary allocations
 *
 * Math for weight/bias folding:
 *   y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta
 *     = gamma / sqrt(var + eps) * conv(x) + (beta - gamma * mean / sqrt(var + eps))
 *     = w_fold * conv(x) + b_fold
 *
 * Where:
 *   w_fold = weight * gamma / sqrt(var + eps)  [per output channel]
 *   b_fold = bias * gamma / sqrt(var + eps) + beta - gamma * mean / sqrt(var + eps)
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <type_traits>

#include "buffer_pool.hpp"
#include "gemm_optimized.hpp"

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #if defined(__AVX512F__)
        #define TENZOR_FUSED_AVX512
    #endif
    #if defined(__AVX2__)
        #define TENZOR_FUSED_AVX2
    #endif
#endif

#ifdef _OPENMP
#include <omp.h>
#include "tenzor/backend/omp_thresholds.hpp"
#endif

namespace tenzor {
namespace cpu {
namespace fused {

// ============================================================================
// BatchNorm Parameter Folding
// ============================================================================

/**
 * @brief Pre-compute folded weights and biases for Conv+BN fusion
 *
 * This is done once before inference, not during forward pass.
 */
// Templated on T so Float64 inputs get a genuine double-precision fold
// instead of being widened/narrowed through float (F059). The AVX2 SIMD
// path is only valid for T=float (the intrinsics are 32-bit-lane only); the
// T=double instantiation takes the scalar branch and keeps every multiply
// in native double precision — no narrowing anywhere in this function.
template <typename T>
inline void fold_bn_params(
    T* weight,           // [out_channels, in_channels, kH, kW] - modified in place
    T* bias,             // [out_channels] - modified in place (or created if null)
    const T* bn_gamma,   // [out_channels]
    const T* bn_beta,    // [out_channels]
    const T* bn_mean,    // [out_channels]
    const T* bn_var,     // [out_channels]
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    T eps = static_cast<T>(1e-5)
) {
    const int64_t weight_per_oc = in_channels * kernel_h * kernel_w;

    // BN folding always produces a nonzero folded bias (beta - gamma*mean/std)
    // that must be stored somewhere. A null bias buffer would silently drop that
    // shift term, turning the result into relu(scale*conv(x)) instead of
    // relu(scale*conv(x) + folded_bias) — a hard-to-spot correctness bug.
    // Validate the precondition here, OUTSIDE the parallel region (we cannot
    // throw from inside an OpenMP parallel for). Callers with a bias-less conv
    // must pass a zero-initialised [out_channels] buffer.
    if (bias == nullptr) {
        throw std::invalid_argument(
            "fold_bn_params: bias buffer must be non-null (BN folding produces a "
            "folded bias that must be stored; pass a zero-initialised buffer for "
            "a bias-less conv)");
    }

    #pragma omp parallel for if(out_channels > 32)
    for (int64_t oc = 0; oc < out_channels; ++oc) {
        // Compute scale factor: gamma / sqrt(var + eps)
        T scale = bn_gamma[oc] / std::sqrt(bn_var[oc] + eps);

        // Fold scale into weights
        T* w_ptr = weight + oc * weight_per_oc;

        if constexpr (std::is_same_v<T, float>) {
#ifdef TENZOR_FUSED_AVX2
            int64_t i = 0;
            __m256 vscale = _mm256_set1_ps(scale);
            for (; i + 8 <= weight_per_oc; i += 8) {
                __m256 vw = _mm256_loadu_ps(w_ptr + i);
                vw = _mm256_mul_ps(vw, vscale);
                _mm256_storeu_ps(w_ptr + i, vw);
            }
            for (; i < weight_per_oc; ++i) {
                w_ptr[i] *= scale;
            }
#else
            for (int64_t i = 0; i < weight_per_oc; ++i) {
                w_ptr[i] *= scale;
            }
#endif
        } else {
            // Genuine double-precision path (F059): plain scalar loop, no
            // float SIMD narrowing.
            for (int64_t i = 0; i < weight_per_oc; ++i) {
                w_ptr[i] *= scale;
            }
        }

        // Fold into bias: bias * scale + beta - gamma * mean / sqrt(var + eps).
        // `bias` is guaranteed non-null by the precondition check above.
        T bias_fold = bn_beta[oc] - bn_gamma[oc] * bn_mean[oc] / std::sqrt(bn_var[oc] + eps);
        bias[oc] = bias[oc] * scale + bias_fold;
    }
}

// ============================================================================
// im2col with SIMD optimization
// ============================================================================

/**
 * @brief Optimized im2col for Conv2d
 */
template <typename T>
inline void im2col_optimized(
    const T* input,
    T* col,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t out_h,
    int64_t out_w
) {
    const int64_t col_width = channels * kernel_h * kernel_w;

    #pragma omp parallel for collapse(2) if(batch * out_h * out_w > ::tenzor::OmpThresholds::medium())
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
            for (int64_t ow = 0; ow < out_w; ++ow) {
                T* col_ptr = col + (b * out_h * out_w + oh * out_w + ow) * col_width;
                int64_t col_idx = 0;

                for (int64_t c = 0; c < channels; ++c) {
                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        int64_t ih = oh * stride_h - padding_h + kh * dilation_h;

                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t iw = ow * stride_w - padding_w + kw * dilation_w;

                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                col_ptr[col_idx] = input[
                                    b * (channels * height * width) +
                                    c * (height * width) +
                                    ih * width + iw
                                ];
                            } else {
                                col_ptr[col_idx] = T(0);
                            }
                            col_idx++;
                        }
                    }
                }
            }
        }
    }
}

// GEMM dispatch for the fused Conv+BN+ReLU path: C = A @ B^T (A is MxK,
// B is NxK). float routes to the hand-tuned SIMD micro-kernel in
// gemm_optimized.hpp. That micro-kernel is float-only (gemm_transB_optimized
// takes `const float*`), so T=double takes a plain scalar loop that
// accumulates in T (double) throughout — genuinely full double precision,
// not a narrowed float GEMM upcast back to double afterwards (F059). This
// mirrors conv2d.cpp's gemm_cpu<double> scalar fallback, which the codebase
// already treats as "correct, just slower", not a precision regression.
template <typename T>
inline void gemm_transB_generic(
    const T* A, const T* B, T* C, int64_t M, int64_t N, int64_t K
) {
    if constexpr (std::is_same_v<T, float>) {
        gemm::gemm_transB_optimized(A, B, C, M, N, K);
    } else {
        #pragma omp parallel for collapse(2) if(M * N > ::tenzor::OmpThresholds::medium())
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                T sum = T(0);
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[j * K + k];
                }
                C[i * N + j] = sum;
            }
        }
    }
}

// ============================================================================
// Fused Conv2d + BatchNorm + ReLU Forward
// ============================================================================

/**
 * @brief Fused Conv2d + BatchNorm + ReLU (inference mode with folded params)
 *
 * Assumes BatchNorm parameters have been folded into weights/biases.
 * This is the fastest inference path.
 */
// `groups` (F061) defaults to 1 for source compatibility. weight is expected
// in the standard grouped-conv layout [C_out, C_in/groups, kH, kW] (same
// layout plain Conv2d uses — see conv2d.cpp's conv2d_forward_impl). For
// groups>1 the convolution (im2col+GEMM) is computed per group, with each
// group's input/output channel window offset the same way
// conv2d_forward_impl does it; BN fold/bias/ReLU stay purely per-output-
// -channel so they need no group awareness beyond the channel offset
// already baked into `bias`/`output`.
template <typename T>
inline void conv_bn_relu_folded(
    const T* input,      // [N, C_in, H, W]
    const T* weight,     // [C_out, C_in/groups, kH, kW] - folded weights
    const T* bias,       // [C_out] - folded bias
    T* output,           // [N, C_out, H_out, W_out]
    int64_t batch,
    int64_t in_channels,
    int64_t height,
    int64_t width,
    int64_t out_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t out_h,
    int64_t out_w,
    int64_t groups = 1
) {
    // Per-batch im2col + GEMM so the GEMM output lands directly in NCHW
    // (out_channels, out_h*out_w) for each batch element.  Producing the
    // GEMM result as weight @ col^T (rather than col @ weight^T) keeps the
    // channel dimension as the outer one, which matches the per-channel
    // bias-add + ReLU pass below and the canonical [N, C, H, W] output
    // tensor layout.  (Previous implementation called col @ weight^T whose
    // (col_rows, out_channels) NHWC-flat layout did not match the post-GEMM
    // NCHW indexing — see audit item A.1.)
    const int64_t in_channels_per_group = in_channels / groups;
    const int64_t out_channels_per_group = out_channels / groups;
    const int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;
    const int64_t spatial_size = out_h * out_w;

    auto col_buffer = acquire_buffer<T>(spatial_size * col_cols);
    T* col = col_buffer.data();

    for (int64_t g = 0; g < groups; ++g) {
        const int64_t in_start = g * in_channels_per_group;
        const int64_t out_start = g * out_channels_per_group;
        // Rows [out_start, out_start+out_channels_per_group) of `weight`
        // are contiguous in the [C_out, C_in/groups, kH, kW] layout.
        const T* weight_g = weight + out_start * col_cols;

        for (int64_t b = 0; b < batch; ++b) {
            const T* input_b = input + b * in_channels * height * width
                                      + in_start * height * width;
            T* output_b = output + b * out_channels * spatial_size
                                  + out_start * spatial_size;

            // im2col for a single batch element (batch=1), this group's
            // input-channel window only.
            im2col_optimized(
                input_b, col,
                /*batch=*/1, in_channels_per_group, height, width,
                kernel_h, kernel_w,
                stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
                out_h, out_w
            );

            // GEMM: output_b = weight_g @ col^T, i.e. C(M,N) = A(M,K) * B(N,K)^T.
            //   weight_g is (out_channels_per_group=M, col_cols=K)
            //   col      is (spatial_size=N, col_cols=K)
            //   output_b is (M, N) = (out_channels_per_group, out_h * out_w),
            //   landing at this group's channel offset in the NCHW output.
            gemm_transB_generic<T>(
                weight_g, col, output_b,
                out_channels_per_group, spatial_size, col_cols
            );

            // Add bias + ReLU per output channel in this group. The AVX
            // float micro-kernels below are only valid for T=float; T=double
            // (F059) takes the plain scalar branch and keeps the add/max in
            // native double precision.
            if constexpr (std::is_same_v<T, float>) {
#ifdef TENZOR_FUSED_AVX512
                #pragma omp parallel for if(out_channels_per_group > 32)
                for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                    float* out_ptr = output_b + oc * spatial_size;
                    __m512 vbias = _mm512_set1_ps(bias[out_start + oc]);
                    __m512 vzero = _mm512_setzero_ps();

                    int64_t s = 0;
                    for (; s + 16 <= spatial_size; s += 16) {
                        __m512 v = _mm512_loadu_ps(out_ptr + s);
                        v = _mm512_add_ps(v, vbias);
                        v = _mm512_max_ps(v, vzero);
                        _mm512_storeu_ps(out_ptr + s, v);
                    }
                    for (; s < spatial_size; ++s) {
                        out_ptr[s] = std::max(0.0f, out_ptr[s] + bias[out_start + oc]);
                    }
                }

#elif defined(TENZOR_FUSED_AVX2)
                #pragma omp parallel for if(out_channels_per_group > 32)
                for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                    float* out_ptr = output_b + oc * spatial_size;
                    __m256 vbias = _mm256_set1_ps(bias[out_start + oc]);
                    __m256 vzero = _mm256_setzero_ps();

                    int64_t s = 0;
                    for (; s + 8 <= spatial_size; s += 8) {
                        __m256 v = _mm256_loadu_ps(out_ptr + s);
                        v = _mm256_add_ps(v, vbias);
                        v = _mm256_max_ps(v, vzero);
                        _mm256_storeu_ps(out_ptr + s, v);
                    }
                    for (; s < spatial_size; ++s) {
                        out_ptr[s] = std::max(0.0f, out_ptr[s] + bias[out_start + oc]);
                    }
                }

#else
                #pragma omp parallel for
                for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                    float b_val = bias[out_start + oc];
                    float* out_ptr = output_b + oc * spatial_size;
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        out_ptr[s] = std::max(0.0f, out_ptr[s] + b_val);
                    }
                }
#endif
            } else {
                #pragma omp parallel for
                for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                    T b_val = bias[out_start + oc];
                    T* out_ptr = output_b + oc * spatial_size;
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        out_ptr[s] = std::max(T(0), out_ptr[s] + b_val);
                    }
                }
            }
        }
    }
}

/**
 * @brief Fused Conv2d + BatchNorm + ReLU (training mode - no folding)
 *
 * For training, we need separate BN statistics updates.
 * This version computes BN inline without folding.
 */
// Templated on T (F059): T=double takes the scalar GEMM/SIMD-free branches
// throughout, so a Float64 input's convolution, batch-statistics reduction,
// and normalize+scale+ReLU pass all stay in genuine double precision end to
// end — matching CUDA's `launch_fused_conv2d_bn_relu_training<double>` which
// never narrows through float either.
template <typename T>
inline void conv_bn_relu_training(
    const T* input,      // [N, C_in, H, W]
    const T* weight,     // [C_out, C_in/groups, kH, kW]
    const T* conv_bias,  // [C_out] or nullptr
    const T* bn_gamma,   // [C_out]
    const T* bn_beta,    // [C_out]
    T* running_mean,     // [C_out] - updated
    T* running_var,      // [C_out] - updated
    T* output,           // [N, C_out, H_out, W_out]
    int64_t batch,
    int64_t in_channels,
    int64_t height,
    int64_t width,
    int64_t out_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t out_h,
    int64_t out_w,
    int64_t groups = 1,
    T momentum = static_cast<T>(0.1),
    T eps = static_cast<T>(1e-5)
) {
    // Step 1: Convolution.  Per-batch, per-group im2col + GEMM keeps the
    // output in canonical NCHW layout — see comment in conv_bn_relu_folded
    // for the reason this is required (audit item A.1) and for the
    // group channel-offset scheme (F061), mirrored here identically. BN
    // statistics/normalize/ReLU below operate purely per-output-channel, so
    // they are correct as written regardless of which group produced a
    // given output channel — no further group awareness needed past this
    // step.
    const int64_t in_channels_per_group = in_channels / groups;
    const int64_t out_channels_per_group = out_channels / groups;
    const int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;
    const int64_t spatial_size = out_h * out_w;

    auto col_buffer = acquire_buffer<T>(spatial_size * col_cols);
    T* col = col_buffer.data();

    auto conv_out_buffer = acquire_buffer<T>(batch * out_channels * spatial_size);
    T* conv_out = conv_out_buffer.data();

    for (int64_t g = 0; g < groups; ++g) {
        const int64_t in_start = g * in_channels_per_group;
        const int64_t out_start = g * out_channels_per_group;
        const T* weight_g = weight + out_start * col_cols;

        for (int64_t b = 0; b < batch; ++b) {
            const T* input_b = input + b * in_channels * height * width
                                      + in_start * height * width;
            T* conv_out_b = conv_out + b * out_channels * spatial_size
                                      + out_start * spatial_size;

            im2col_optimized(
                input_b, col,
                /*batch=*/1, in_channels_per_group, height, width,
                kernel_h, kernel_w,
                stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
                out_h, out_w
            );

            // conv_out_b (out_channels_per_group, out_h*out_w) = weight_g @ col^T  ⇒ NCHW.
            gemm_transB_generic<T>(
                weight_g, col, conv_out_b,
                out_channels_per_group, spatial_size, col_cols
            );
        }
    }

    // Add conv bias if present
    if (conv_bias) {
        #pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t oc = 0; oc < out_channels; ++oc) {
                T bias_val = conv_bias[oc];
                for (int64_t s = 0; s < spatial_size; ++s) {
                    conv_out[b * out_channels * spatial_size + oc * spatial_size + s] += bias_val;
                }
            }
        }
    }

    // Step 2: BatchNorm (compute batch statistics + normalize + ReLU)
    const int64_t samples_per_channel = batch * spatial_size;

    // Allocate temporary storage for batch mean/var
    auto batch_mean = acquire_buffer<T>(out_channels);
    auto batch_var = acquire_buffer<T>(out_channels);

    // Compute batch statistics
    #pragma omp parallel for if(out_channels > 16)
    for (int64_t oc = 0; oc < out_channels; ++oc) {
        // Compute mean. Accumulate in double: a float accumulator over
        // batch*spatial_size (tens of thousands of) elements loses precision,
        // diverging from the double-accumulated reference used by the other
        // BatchNorm normalization paths. For T=double this is simply native
        // precision throughout (no narrowing at the final static_cast<T>).
        double sum = 0.0;
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < spatial_size; ++s) {
                sum += static_cast<double>(conv_out[b * out_channels * spatial_size + oc * spatial_size + s]);
            }
        }
        double mean_d = sum / static_cast<double>(samples_per_channel);
        T mean = static_cast<T>(mean_d);
        batch_mean[oc] = mean;

        // Compute variance (double accumulator, two-pass).
        double var_sum = 0.0;
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < spatial_size; ++s) {
                double diff = static_cast<double>(conv_out[b * out_channels * spatial_size + oc * spatial_size + s]) - mean_d;
                var_sum += diff * diff;
            }
        }
        double var_biased_d = var_sum / static_cast<double>(samples_per_channel);
        T var = static_cast<T>(var_biased_d);  // biased: used for normalization
        batch_var[oc] = var;

        // Running variance uses the unbiased (Bessel-corrected) estimate to
        // match PyTorch and this codebase's BatchNorm; the biased var above is
        // still used for the in-batch normalization.
        double n_d = static_cast<double>(samples_per_channel);
        T var_unbiased = (samples_per_channel > 1)
            ? static_cast<T>(var_biased_d * n_d / (n_d - 1.0))
            : var;

        // Update running statistics
        running_mean[oc] = (T(1) - momentum) * running_mean[oc] + momentum * mean;
        running_var[oc] = (T(1) - momentum) * running_var[oc] + momentum * var_unbiased;
    }

    // Normalize + scale + ReLU
    #pragma omp parallel for collapse(2) if(batch * out_channels > 32)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t oc = 0; oc < out_channels; ++oc) {
            T mean = batch_mean[oc];
            T var = batch_var[oc];
            T inv_std = T(1) / std::sqrt(var + eps);
            T gamma = bn_gamma[oc];
            T beta = bn_beta[oc];

            T* in_ptr = conv_out + b * out_channels * spatial_size + oc * spatial_size;
            T* out_ptr = output + b * out_channels * spatial_size + oc * spatial_size;

            if constexpr (std::is_same_v<T, float>) {
#ifdef TENZOR_FUSED_AVX2
                __m256 vmean = _mm256_set1_ps(mean);
                __m256 vinv_std = _mm256_set1_ps(inv_std);
                __m256 vgamma = _mm256_set1_ps(gamma);
                __m256 vbeta = _mm256_set1_ps(beta);
                __m256 vzero = _mm256_setzero_ps();

                int64_t s = 0;
                for (; s + 8 <= spatial_size; s += 8) {
                    __m256 v = _mm256_loadu_ps(in_ptr + s);
                    v = _mm256_sub_ps(v, vmean);
                    v = _mm256_mul_ps(v, vinv_std);
                    v = _mm256_fmadd_ps(v, vgamma, vbeta);
                    v = _mm256_max_ps(v, vzero);  // ReLU
                    _mm256_storeu_ps(out_ptr + s, v);
                }
                for (; s < spatial_size; ++s) {
                    float normalized = (in_ptr[s] - mean) * inv_std;
                    float scaled = normalized * gamma + beta;
                    out_ptr[s] = std::max(0.0f, scaled);
                }
#else
                for (int64_t s = 0; s < spatial_size; ++s) {
                    float normalized = (in_ptr[s] - mean) * inv_std;
                    float scaled = normalized * gamma + beta;
                    out_ptr[s] = std::max(0.0f, scaled);
                }
#endif
            } else {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    T normalized = (in_ptr[s] - mean) * inv_std;
                    T scaled = normalized * gamma + beta;
                    out_ptr[s] = std::max(T(0), scaled);
                }
            }
        }
    }
}

// NOTE (audit E): the standalone `conv_relu` helper was removed. It was never
// wired into any dispatch path (only conv_bn_relu_folded / conv_bn_relu_training
// are used) and was broken — it ran gemm_transB into an [N*out_h*out_w,
// out_channels] (NHWC-flat) buffer, then applied bias/ReLU with NCHW indexing,
// the exact layout mismatch the two live functions were fixed for.

} // namespace fused
} // namespace cpu
} // namespace tenzor
