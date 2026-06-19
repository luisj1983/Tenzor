// =====================================================================
// CPU DepthwiseConv3d forward kernel (Stream S18).
//
// Specialised 3-D depthwise convolution: groups = in_channels = out_channels.
// Each output channel `c` is the 3-D convolution of input channel `c` with
// the weight slice `weight[c, 0, :, :, :]`. No cross-channel reduction →
// the per-channel work decomposes cleanly across OpenMP threads and the
// innermost (W) spatial loop is vectorisable.
//
// Contract (matches the NN-layer dispatch in src/nn/layers/conv.cpp):
//   - input:  [N, C, D, H, W]
//   - weight: [C, 1, kD, kH, kW]
//   - attrs:  StrideD/H/W (with scalar Stride fallback),
//             PaddingD/H/W (with scalar Padding fallback),
//             DilationD/H/W (with scalar Dilation fallback).
//
// Mirrors the 2-D depthwise pattern in kernels/conv2d.cpp.
// =====================================================================

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/omp_thresholds.hpp"

#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #if defined(__AVX2__)
        #define TENZOR_DWCONV3D_AVX2
    #endif
#endif

#include "half_operators.hpp"

namespace tenzor {
namespace cpu {

// ---------------------------------------------------------------------
// Scalar Kahan-summing reference implementation.
//
// Layout: input [N, C, D, H, W], weight [C, 1, kD, kH, kW] (collapsed to
// [C, kD, kH, kW] for indexing). bias indexed by output channel == input
// channel. Output [N, C, D_out, H_out, W_out].
// ---------------------------------------------------------------------
template <typename T>
static void depthwise_conv3d_impl(const T* in_data,
                                  const T* w_data,
                                  const T* b_data,
                                  T* out_data,
                                  int64_t N, int64_t C,
                                  int64_t D, int64_t H, int64_t W,
                                  int64_t kD, int64_t kH, int64_t kW,
                                  int64_t D_out, int64_t H_out, int64_t W_out,
                                  int64_t stride_d, int64_t stride_h, int64_t stride_w,
                                  int64_t pad_d,    int64_t pad_h,    int64_t pad_w,
                                  int64_t dil_d,    int64_t dil_h,    int64_t dil_w) {
    using AccumT = std::conditional_t<std::is_same_v<T, double>, double, float>;
    const int64_t total = N * C * D_out * H_out * W_out;
    const int64_t HW       = H * W;
    const int64_t DHW      = D * H * W;
    const int64_t kHW      = kH * kW;
    const int64_t kDHW     = kD * kH * kW;
    const int64_t HW_out   = H_out * W_out;
    const int64_t DHW_out  = D_out * H_out * W_out;

    // collapse(4) over N/C/D_out/H_out — leaves the W_out inner loop sequential
    // (and contiguous in memory) so vectorisation has a clean access pattern.
    #pragma omp parallel for collapse(4) if(total > OmpThresholds::medium())
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t od = 0; od < D_out; ++od) {
                for (int64_t oh = 0; oh < H_out; ++oh) {
                    const T* in_ch  = in_data  + (n * C + c) * DHW;
                    const T* wt_ch  = w_data   + c * kDHW;
                    T*       out_ch = out_data + (n * C + c) * DHW_out;

                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        AccumT sum = AccumT(0);
                        AccumT compensation = AccumT(0);

                        for (int64_t kd = 0; kd < kD; ++kd) {
                            const int64_t id = od * stride_d - pad_d + kd * dil_d;
                            if (id < 0 || id >= D) continue;
                            for (int64_t kh = 0; kh < kH; ++kh) {
                                const int64_t ih = oh * stride_h - pad_h + kh * dil_h;
                                if (ih < 0 || ih >= H) continue;
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    const int64_t iw = ow * stride_w - pad_w + kw * dil_w;
                                    if (iw < 0 || iw >= W) continue;

                                    AccumT prod = static_cast<AccumT>(in_ch[id * HW + ih * W + iw]) *
                                                  static_cast<AccumT>(wt_ch[kd * kHW + kh * kW + kw]);
                                    AccumT y = prod - compensation;
                                    AccumT t = sum + y;
                                    compensation = (t - sum) - y;
                                    sum = t;
                                }
                            }
                        }

                        if (b_data) {
                            AccumT y = static_cast<AccumT>(b_data[c]) - compensation;
                            sum = sum + y;
                        }

                        out_ch[od * HW_out + oh * W_out + ow] = static_cast<T>(sum);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// AVX2 Float32 fast path. Vectorises along the innermost (W) spatial
// output axis, 8 lanes at a time. Requires stride==1, dilation==1 on
// all three axes; per-axis padding is fine (only shifts the iw_start
// arithmetic). The matching 2-D fast path is `depthwise_conv2d_avx2_f32`
// in conv2d.cpp — we mirror its boundary-handling strategy verbatim.
// ---------------------------------------------------------------------
#ifdef TENZOR_DWCONV3D_AVX2
static void depthwise_conv3d_avx2_f32(const float* __restrict__ in_data,
                                       const float* __restrict__ w_data,
                                       const float* __restrict__ b_data,
                                       float* __restrict__ out_data,
                                       int64_t N, int64_t C,
                                       int64_t D, int64_t H, int64_t W,
                                       int64_t kD, int64_t kH, int64_t kW,
                                       int64_t D_out, int64_t H_out, int64_t W_out,
                                       int64_t pad_d, int64_t pad_h, int64_t pad_w) {
    const int64_t HW       = H * W;
    const int64_t DHW      = D * H * W;
    const int64_t kHW      = kH * kW;
    const int64_t kDHW     = kD * kH * kW;
    const int64_t HW_out   = H_out * W_out;
    const int64_t DHW_out  = D_out * H_out * W_out;

    #pragma omp parallel for collapse(4) if(N * C * D_out * H_out > 1)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t od = 0; od < D_out; ++od) {
                for (int64_t oh = 0; oh < H_out; ++oh) {
                    const float* in_ch  = in_data  + (n * C + c) * DHW;
                    const float* filter = w_data   + c * kDHW;
                    float*       out_row = out_data + (n * C + c) * DHW_out
                                          + od * HW_out + oh * W_out;

                    int64_t ow = 0;
                    for (; ow + 8 <= W_out; ow += 8) {
                        __m256 v_sum = _mm256_setzero_ps();

                        for (int64_t kd = 0; kd < kD; ++kd) {
                            const int64_t id = od - pad_d + kd;
                            if (id < 0 || id >= D) continue;
                            for (int64_t kh = 0; kh < kH; ++kh) {
                                const int64_t ih = oh - pad_h + kh;
                                if (ih < 0 || ih >= H) continue;

                                const float* in_row = in_ch + id * HW + ih * W;

                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    const int64_t iw_start = ow - pad_w + kw;
                                    __m256 v_w = _mm256_set1_ps(filter[kd * kHW + kh * kW + kw]);

                                    if (iw_start >= 0 && iw_start + 8 <= W) {
                                        __m256 v_in = _mm256_loadu_ps(in_row + iw_start);
                                        v_sum = _mm256_fmadd_ps(v_in, v_w, v_sum);
                                    } else {
                                        alignas(32) float tmp[8];
                                        for (int i = 0; i < 8; ++i) {
                                            const int64_t iw = iw_start + i;
                                            tmp[i] = (iw >= 0 && iw < W) ? in_row[iw] : 0.0f;
                                        }
                                        __m256 v_in = _mm256_load_ps(tmp);
                                        v_sum = _mm256_fmadd_ps(v_in, v_w, v_sum);
                                    }
                                }
                            }
                        }

                        if (b_data) {
                            __m256 v_bias = _mm256_set1_ps(b_data[c]);
                            v_sum = _mm256_add_ps(v_sum, v_bias);
                        }

                        _mm256_storeu_ps(out_row + ow, v_sum);
                    }

                    // Scalar tail.
                    for (; ow < W_out; ++ow) {
                        float sum = 0.0f;
                        for (int64_t kd = 0; kd < kD; ++kd) {
                            const int64_t id = od - pad_d + kd;
                            if (id < 0 || id >= D) continue;
                            for (int64_t kh = 0; kh < kH; ++kh) {
                                const int64_t ih = oh - pad_h + kh;
                                if (ih < 0 || ih >= H) continue;
                                const float* in_row = in_ch + id * HW + ih * W;
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    const int64_t iw = ow - pad_w + kw;
                                    if (iw >= 0 && iw < W) {
                                        sum += in_row[iw] * filter[kd * kHW + kh * kW + kw];
                                    }
                                }
                            }
                        }
                        if (b_data) sum += b_data[c];
                        out_row[ow] = sum;
                    }
                }
            }
        }
    }
}
#endif  // TENZOR_DWCONV3D_AVX2

// ---------------------------------------------------------------------
// Public kernel.
// ---------------------------------------------------------------------
auto depthwise_conv3d_kernel(const Tensor& input,
                              const Tensor& weight,
                              const Tensor* bias,
                              int64_t stride_d, int64_t stride_h, int64_t stride_w,
                              int64_t pad_d,    int64_t pad_h,    int64_t pad_w,
                              int64_t dil_d,    int64_t dil_h,    int64_t dil_w) -> Tensor {
    const auto in_shape = input.shape();
    const auto w_shape  = weight.shape();
    if (in_shape.size() != 5 || w_shape.size() != 5) {
        throw std::runtime_error(
            "depthwise_conv3d_kernel: expected 5-D input/weight ([N,C,D,H,W] and "
            "[C,1,kD,kH,kW]).");
    }

    const int64_t N  = in_shape[0];
    const int64_t C  = in_shape[1];
    const int64_t D  = in_shape[2];
    const int64_t H  = in_shape[3];
    const int64_t W  = in_shape[4];
    const int64_t kD = w_shape[2];
    const int64_t kH = w_shape[3];
    const int64_t kW = w_shape[4];

    const int64_t D_out = (D + 2 * pad_d - dil_d * (kD - 1) - 1) / stride_d + 1;
    const int64_t H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int64_t W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (D_out <= 0 || H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("depthwise_conv3d_kernel: non-positive output extent "
                                 "(check padding / dilation / kernel size).");
    }

    auto output = Tensor::empty_uninitialized({N, C, D_out, H_out, W_out},
                                              input.dtype(), input.device());

    // The impl/AVX2 paths index input/weight as flat packed buffers
    // ([N,C,D,H,W] / [C,1,kD,kH,kW]); a non-contiguous view (permuted/sliced
    // channels) would be walked in the wrong order, silently producing wrong
    // results. Materialize contiguous copies once (no-op when already packed).
    // The F16/BF16 branches below already repack via .to(Float32).
    const Tensor in_c = input.is_contiguous() ? input : input.contiguous();
    const Tensor w_c  = weight.is_contiguous() ? weight : weight.contiguous();
    Tensor bias_c_storage;
    const Tensor* bias_c = bias;
    if (bias && !bias->is_contiguous()) {
        bias_c_storage = bias->contiguous();
        bias_c = &bias_c_storage;
    }

    const DType dt = input.dtype();
    if (dt == DType::Float32) {
#ifdef TENZOR_DWCONV3D_AVX2
        if (stride_d == 1 && stride_h == 1 && stride_w == 1 &&
            dil_d == 1 && dil_h == 1 && dil_w == 1) {
            depthwise_conv3d_avx2_f32(in_c.data<float>(), w_c.data<float>(),
                bias_c ? bias_c->data<float>() : nullptr, output.data<float>(),
                N, C, D, H, W, kD, kH, kW, D_out, H_out, W_out,
                pad_d, pad_h, pad_w);
        } else
#endif
        {
            depthwise_conv3d_impl<float>(in_c.data<float>(), w_c.data<float>(),
                bias_c ? bias_c->data<float>() : nullptr, output.data<float>(),
                N, C, D, H, W, kD, kH, kW, D_out, H_out, W_out,
                stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dil_d, dil_h, dil_w);
        }
    } else if (dt == DType::Float64) {
        depthwise_conv3d_impl<double>(in_c.data<double>(), w_c.data<double>(),
            bias_c ? bias_c->data<double>() : nullptr, output.data<double>(),
            N, C, D, H, W, kD, kH, kW, D_out, H_out, W_out,
            stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dil_d, dil_h, dil_w);
    } else if (dt == DType::Float16) {
        auto in_f32 = input.to(DType::Float32);
        auto w_f32  = weight.to(DType::Float32);
        Tensor bias_f32_storage;
        const float* bias_f32_ptr = nullptr;
        if (bias) { bias_f32_storage = bias->to(DType::Float32); bias_f32_ptr = bias_f32_storage.data<float>(); }
        auto out_f32 = Tensor::empty_uninitialized({N, C, D_out, H_out, W_out},
                                                   DType::Float32, input.device());
#ifdef TENZOR_DWCONV3D_AVX2
        if (stride_d == 1 && stride_h == 1 && stride_w == 1 &&
            dil_d == 1 && dil_h == 1 && dil_w == 1) {
            depthwise_conv3d_avx2_f32(in_f32.data<float>(), w_f32.data<float>(),
                bias_f32_ptr, out_f32.data<float>(),
                N, C, D, H, W, kD, kH, kW, D_out, H_out, W_out, pad_d, pad_h, pad_w);
        } else
#endif
        {
            depthwise_conv3d_impl<float>(in_f32.data<float>(), w_f32.data<float>(),
                bias_f32_ptr, out_f32.data<float>(),
                N, C, D, H, W, kD, kH, kW, D_out, H_out, W_out,
                stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dil_d, dil_h, dil_w);
        }
        output = out_f32.to(DType::Float16);
    } else if (dt == DType::BFloat16) {
        auto in_f32 = input.to(DType::Float32);
        auto w_f32  = weight.to(DType::Float32);
        Tensor bias_f32_storage;
        const float* bias_f32_ptr = nullptr;
        if (bias) { bias_f32_storage = bias->to(DType::Float32); bias_f32_ptr = bias_f32_storage.data<float>(); }
        auto out_f32 = Tensor::empty_uninitialized({N, C, D_out, H_out, W_out},
                                                   DType::Float32, input.device());
#ifdef TENZOR_DWCONV3D_AVX2
        if (stride_d == 1 && stride_h == 1 && stride_w == 1 &&
            dil_d == 1 && dil_h == 1 && dil_w == 1) {
            depthwise_conv3d_avx2_f32(in_f32.data<float>(), w_f32.data<float>(),
                bias_f32_ptr, out_f32.data<float>(),
                N, C, D, H, W, kD, kH, kW, D_out, H_out, W_out, pad_d, pad_h, pad_w);
        } else
#endif
        {
            depthwise_conv3d_impl<float>(in_f32.data<float>(), w_f32.data<float>(),
                bias_f32_ptr, out_f32.data<float>(),
                N, C, D, H, W, kD, kH, kW, D_out, H_out, W_out,
                stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dil_d, dil_h, dil_w);
        }
        output = out_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("depthwise_conv3d_kernel: unsupported dtype "
                                 "(supports Float32, Float64, Float16, BFloat16).");
    }

    return output;
}

}  // namespace cpu
}  // namespace tenzor
