// =====================================================================
// CPU DepthwiseConv1d forward kernel (Stream S18).
//
// Specialised 1-D depthwise convolution: groups = in_channels = out_channels.
// Each output channel `c` is the 1-D convolution of input channel `c` with
// the weight slice `weight[c, 0, :]`. Because there is no cross-channel
// reduction, the per-output-channel work decomposes cleanly across OpenMP
// threads and the spatial inner loop is vectorisable.
//
// Contract (matches the NN-layer dispatch in src/nn/layers/conv.cpp):
//   - The Conv1d::forward_impl path manually applies 1-D padding to the
//     length axis BEFORE calling dispatch(...). It then unsqueezes the
//     input from [N, C, L]  -> [N, C, 1, L_padded]
//     and the weight from   [C, 1, kL] -> [C, 1, 1, kL].
//     `AttrKey::Padding` is set to 0 in the attrs, while Stride/Dilation/
//     Groups reflect the actual layer config.
//   - So when our kernel runs, the input is 4-D with H == 1, kH == 1, and
//     padding_h == 0. We treat the W axis as the only spatial axis.
//   - Output is [N, C, 1, L_out]; the NN layer squeezes the H=1 axis off
//     before returning.
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
        #define TENZOR_DWCONV1D_AVX2
    #endif
#endif

#include "half_operators.hpp"

namespace tenzor {
namespace cpu {

// ---------------------------------------------------------------------
// Scalar Kahan-summing reference implementation.
//
// Layout: input [N, C, L], weight [C, 1, kL] (collapsed to [C, kL] for
// indexing purposes), output [N, C, L_out]. bias is optional and indexed
// by output channel == input channel for depthwise.
// ---------------------------------------------------------------------
template <typename T>
static void depthwise_conv1d_impl(const T* in_data,
                                  const T* w_data,
                                  const T* b_data,
                                  T* out_data,
                                  int64_t N, int64_t C,
                                  int64_t L, int64_t kL, int64_t L_out,
                                  int64_t stride_l, int64_t padding_l,
                                  int64_t dilation_l) {
    using AccumT = std::conditional_t<std::is_same_v<T, double>, double, float>;
    const int64_t total = N * C * L_out;

    #pragma omp parallel for collapse(3) if(total > OmpThresholds::medium())
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t ol = 0; ol < L_out; ++ol) {
                AccumT sum = AccumT(0);
                AccumT compensation = AccumT(0);

                for (int64_t kl = 0; kl < kL; ++kl) {
                    const int64_t il = ol * stride_l - padding_l + kl * dilation_l;
                    if (il < 0 || il >= L) continue;

                    AccumT prod = static_cast<AccumT>(in_data[(n * C + c) * L + il]) *
                                  static_cast<AccumT>(w_data[c * kL + kl]);
                    AccumT y = prod - compensation;
                    AccumT t = sum + y;
                    compensation = (t - sum) - y;
                    sum = t;
                }

                if (b_data) {
                    AccumT y = static_cast<AccumT>(b_data[c]) - compensation;
                    sum = sum + y;
                }

                out_data[(n * C + c) * L_out + ol] = static_cast<T>(sum);
            }
        }
    }
}

// ---------------------------------------------------------------------
// AVX2 Float32 fast path. Vectorises along the SPATIAL (length) output
// dimension — 8 output positions at a time. Stride==1, dilation==1 only;
// arbitrary padding allowed (only affects the per-tap iw_start arithmetic).
// ---------------------------------------------------------------------
#ifdef TENZOR_DWCONV1D_AVX2
static void depthwise_conv1d_avx2_f32(const float* __restrict in_data,
                                       const float* __restrict w_data,
                                       const float* __restrict b_data,
                                       float* __restrict out_data,
                                       int64_t N, int64_t C,
                                       int64_t L, int64_t kL, int64_t L_out,
                                       int64_t padding_l) {
    // Caller guarantees stride == 1 and dilation == 1.
    #pragma omp parallel for collapse(2) if(N * C > 1)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* filter = w_data + c * kL;
            const float* in_row = in_data + (n * C + c) * L;
            float*       out_row = out_data + (n * C + c) * L_out;

            int64_t ol = 0;
            for (; ol + 8 <= L_out; ol += 8) {
                __m256 v_sum = _mm256_setzero_ps();

                for (int64_t kl = 0; kl < kL; ++kl) {
                    const int64_t iw_start = ol - padding_l + kl;
                    __m256 v_w = _mm256_set1_ps(filter[kl]);

                    if (iw_start >= 0 && iw_start + 8 <= L) {
                        __m256 v_in = _mm256_loadu_ps(in_row + iw_start);
                        v_sum = _mm256_fmadd_ps(v_in, v_w, v_sum);
                    } else {
                        alignas(32) float tmp[8];
                        for (int i = 0; i < 8; ++i) {
                            const int64_t iw = iw_start + i;
                            tmp[i] = (iw >= 0 && iw < L) ? in_row[iw] : 0.0f;
                        }
                        __m256 v_in = _mm256_load_ps(tmp);
                        v_sum = _mm256_fmadd_ps(v_in, v_w, v_sum);
                    }
                }

                if (b_data) {
                    __m256 v_bias = _mm256_set1_ps(b_data[c]);
                    v_sum = _mm256_add_ps(v_sum, v_bias);
                }

                _mm256_storeu_ps(out_row + ol, v_sum);
            }

            // Scalar tail.
            for (; ol < L_out; ++ol) {
                float sum = 0.0f;
                for (int64_t kl = 0; kl < kL; ++kl) {
                    const int64_t iw = ol - padding_l + kl;
                    if (iw >= 0 && iw < L) {
                        sum += in_row[iw] * filter[kl];
                    }
                }
                if (b_data) sum += b_data[c];
                out_row[ol] = sum;
            }
        }
    }
}
#endif  // TENZOR_DWCONV1D_AVX2

// ---------------------------------------------------------------------
// Public kernel. The NN layer hands us 4-D tensors (H == 1) created via
// unsqueeze(2). We collapse the H axis here so the underlying maths stays
// 1-D.
// ---------------------------------------------------------------------
auto depthwise_conv1d_kernel(const Tensor& input_4d,
                              const Tensor& weight_4d,
                              const Tensor* bias,
                              int64_t stride_l,
                              int64_t padding_l,
                              int64_t dilation_l) -> Tensor {
    // input_4d: [N, C, 1, L_padded]   weight_4d: [C, 1, 1, kL]
    const auto in_shape = input_4d.shape();
    const auto w_shape  = weight_4d.shape();
    if (in_shape.size() != 4 || w_shape.size() != 4 ||
        in_shape[2] != 1 || w_shape[2] != 1) {
        throw std::runtime_error(
            "depthwise_conv1d_kernel: expected 4-D input/weight with H==1 "
            "(unsqueezed 1-D from the NN layer).");
    }

    const int64_t N  = in_shape[0];
    const int64_t C  = in_shape[1];
    const int64_t L  = in_shape[3];
    const int64_t kL = w_shape[3];

    // Effective (dilated) kernel extent must fit within the padded input,
    // mirroring PyTorch's explicit size check. Validating here avoids relying
    // on the L_out<=0 guard, which misses numerators in (-stride_l, 0) that
    // truncate to 0 under C++ integer division and yield a misleading 1-wide
    // output.
    const int64_t effective_kernel = dilation_l * (kL - 1) + 1;
    if (effective_kernel > L + 2 * padding_l) {
        throw std::runtime_error(
            "depthwise_conv1d_kernel: effective kernel size "
            "(dilation*(kernel-1)+1) exceeds padded input length "
            "(check padding / dilation / kernel size).");
    }

    const int64_t L_out = (L + 2 * padding_l - dilation_l * (kL - 1) - 1) / stride_l + 1;
    if (L_out <= 0) {
        throw std::runtime_error("depthwise_conv1d_kernel: non-positive output length "
                                 "(check padding / dilation / kernel size).");
    }

    auto output = Tensor::empty_uninitialized({N, C, 1, L_out}, input_4d.dtype(), input_4d.device());

    // The impl/AVX2 paths index input/weight as flat packed [N,C,1,L] buffers.
    // When the Conv1d layer runs with padding==0 it forwards padded.unsqueeze(2)
    // of the raw (possibly non-contiguous) input view, which stays
    // non-contiguous — walking it as flat would silently produce wrong results.
    // Materialize contiguous copies once (no-op when already packed). The
    // F16/BF16 branches below already repack via .to(Float32).
    const Tensor in_c = input_4d.is_contiguous() ? input_4d : input_4d.contiguous();
    const Tensor w_c  = weight_4d.is_contiguous() ? weight_4d : weight_4d.contiguous();
    Tensor bias_c_storage;
    const Tensor* bias_c = bias;
    if (bias && !bias->is_contiguous()) {
        bias_c_storage = bias->contiguous();
        bias_c = &bias_c_storage;
    }

    const DType dt = input_4d.dtype();
    if (dt == DType::Float32) {
#ifdef TENZOR_DWCONV1D_AVX2
        if (stride_l == 1 && dilation_l == 1) {
            depthwise_conv1d_avx2_f32(in_c.data<float>(), w_c.data<float>(),
                bias_c ? bias_c->data<float>() : nullptr, output.data<float>(),
                N, C, L, kL, L_out, padding_l);
        } else
#endif
        {
            depthwise_conv1d_impl<float>(in_c.data<float>(), w_c.data<float>(),
                bias_c ? bias_c->data<float>() : nullptr, output.data<float>(),
                N, C, L, kL, L_out, stride_l, padding_l, dilation_l);
        }
    } else if (dt == DType::Float64) {
        depthwise_conv1d_impl<double>(in_c.data<double>(), w_c.data<double>(),
            bias_c ? bias_c->data<double>() : nullptr, output.data<double>(),
            N, C, L, kL, L_out, stride_l, padding_l, dilation_l);
    } else if (dt == DType::Float16) {
        // Widen-narrow. Same pattern as depthwise_conv2d_impl<Float16>: the
        // template already accumulates in float, but we must also widen the
        // BUFFERS to Float32 so cache traffic and intermediate stores stay in
        // float — otherwise the per-tap conversion cost dominates. Easiest
        // route: cast inputs once, run the f32 path, narrow the result.
        auto in_f32  = in_c.to(DType::Float32);   // widen from the contiguous copy
        auto w_f32   = w_c.to(DType::Float32);
        Tensor bias_f32_storage;
        const float* bias_f32_ptr = nullptr;
        if (bias) { bias_f32_storage = bias->to(DType::Float32); bias_f32_ptr = bias_f32_storage.data<float>(); }
        auto out_f32 = Tensor::empty_uninitialized({N, C, 1, L_out}, DType::Float32, input_4d.device());
#ifdef TENZOR_DWCONV1D_AVX2
        if (stride_l == 1 && dilation_l == 1) {
            depthwise_conv1d_avx2_f32(in_f32.data<float>(), w_f32.data<float>(),
                bias_f32_ptr, out_f32.data<float>(),
                N, C, L, kL, L_out, padding_l);
        } else
#endif
        {
            depthwise_conv1d_impl<float>(in_f32.data<float>(), w_f32.data<float>(),
                bias_f32_ptr, out_f32.data<float>(),
                N, C, L, kL, L_out, stride_l, padding_l, dilation_l);
        }
        output = out_f32.to(DType::Float16);
    } else if (dt == DType::BFloat16) {
        auto in_f32  = in_c.to(DType::Float32);   // widen from the contiguous copy
        auto w_f32   = w_c.to(DType::Float32);
        Tensor bias_f32_storage;
        const float* bias_f32_ptr = nullptr;
        if (bias) { bias_f32_storage = bias->to(DType::Float32); bias_f32_ptr = bias_f32_storage.data<float>(); }
        auto out_f32 = Tensor::empty_uninitialized({N, C, 1, L_out}, DType::Float32, input_4d.device());
#ifdef TENZOR_DWCONV1D_AVX2
        if (stride_l == 1 && dilation_l == 1) {
            depthwise_conv1d_avx2_f32(in_f32.data<float>(), w_f32.data<float>(),
                bias_f32_ptr, out_f32.data<float>(),
                N, C, L, kL, L_out, padding_l);
        } else
#endif
        {
            depthwise_conv1d_impl<float>(in_f32.data<float>(), w_f32.data<float>(),
                bias_f32_ptr, out_f32.data<float>(),
                N, C, L, kL, L_out, stride_l, padding_l, dilation_l);
        }
        output = out_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("depthwise_conv1d_kernel: unsupported dtype "
                                 "(supports Float32, Float64, Float16, BFloat16).");
    }

    return output;
}

}  // namespace cpu
}  // namespace tenzor
