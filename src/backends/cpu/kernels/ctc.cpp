/**
 * @file ctc.cpp
 * @brief CPU kernel for CTC (Connectionist Temporal Classification) loss.
 *
 * Implements the same forward-backward DP that previously lived inline in
 * src/nn/loss/losses_advanced.cpp, now exposed as a backend kernel via
 * OpId::CTCLossForward. The losses_advanced.cpp layer now dispatches to
 * this kernel for CPU tensors and to the GPU backends (CUDA, etc.) for
 * device tensors, eliminating the CPU round-trip in the GPU path.
 *
 * Output contract:
 *   inputs:  [log_probs (T, N, C) Float32,
 *             targets (N, S_max) Int32,
 *             input_lengths (N,) Int32,
 *             target_lengths (N,) Int32]
 *   attrs:   Blank (int, default 0), ZeroInfinity (bool, default false)
 *   outputs: [loss_per_sample (N,) Float32, raw_grad (T, N, C) Float32]
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/backend/backend.hpp"  // OpAttributes alias

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {
namespace cpu {

namespace {

constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

inline float log_add(float a, float b) {
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    float m = std::max(a, b);
    return m + std::log1p(std::exp(-std::fabs(a - b)));
}

// Compute CTC forward-backward for one batch element.
//   log_probs: (T_n, C) per-timestep log-probabilities (contiguous)
//   target:    (S_n,) target labels
// Returns (loss, grad) where loss = -logZ and grad is (T_n, C) with
//   grad[t, c] = exp(log_probs[t, c])
//              - sum_{s: ext_label[s] == c} exp(alpha[t,s] + beta[t,s] - logZ)
auto ctc_single(
    const float* log_probs, int64_t lp_t_stride,  // stride along T (in elements)
    const int32_t* target,
    int64_t T, int64_t S, int64_t C,
    int64_t blank
) -> std::pair<float, std::vector<float>> {
    int64_t L = 2 * S + 1;
    std::vector<int32_t> ext_label(L);
    for (int64_t i = 0; i < L; ++i) {
        ext_label[i] = (i % 2 == 0) ? static_cast<int32_t>(blank) : target[i / 2];
    }

    std::vector<float> alpha(T * L, NEG_INF);
    alpha[0 * L + 0] = log_probs[0 * lp_t_stride + ext_label[0]];
    if (L > 1) {
        alpha[0 * L + 1] = log_probs[0 * lp_t_stride + ext_label[1]];
    }

    for (int64_t t = 1; t < T; ++t) {
        for (int64_t s = 0; s < L; ++s) {
            float prev = alpha[(t - 1) * L + s];
            if (s > 0) {
                prev = log_add(prev, alpha[(t - 1) * L + (s - 1)]);
            }
            if (s > 1 && ext_label[s] != blank && ext_label[s] != ext_label[s - 2]) {
                prev = log_add(prev, alpha[(t - 1) * L + (s - 2)]);
            }
            alpha[t * L + s] = prev + log_probs[t * lp_t_stride + ext_label[s]];
        }
    }

    float logZ = log_add(
        alpha[(T - 1) * L + (L - 1)],
        (L > 1) ? alpha[(T - 1) * L + (L - 2)] : NEG_INF);

    std::vector<float> beta(T * L, NEG_INF);
    beta[(T - 1) * L + (L - 1)] = 0.0f;
    if (L > 1) {
        beta[(T - 1) * L + (L - 2)] = 0.0f;
    }

    for (int64_t t = T - 2; t >= 0; --t) {
        for (int64_t s = 0; s < L; ++s) {
            float next = beta[(t + 1) * L + s]
                       + log_probs[(t + 1) * lp_t_stride + ext_label[s]];
            if (s < L - 1) {
                next = log_add(next,
                    beta[(t + 1) * L + (s + 1)]
                  + log_probs[(t + 1) * lp_t_stride + ext_label[s + 1]]);
            }
            if (s < L - 2 && ext_label[s] != blank && ext_label[s] != ext_label[s + 2]) {
                next = log_add(next,
                    beta[(t + 1) * L + (s + 2)]
                  + log_probs[(t + 1) * lp_t_stride + ext_label[s + 2]]);
            }
            beta[t * L + s] = next;
        }
    }

    std::vector<float> grad(T * C, 0.0f);

    for (int64_t t = 0; t < T; ++t) {
        // Accumulate log-posterior per class via log-add over ext label
        // positions mapping to each c. We use the grad buffer as scratch
        // for log-space accumulation; mark "no contribution" by leaving
        // it at 0.0 then converting in the second pass.
        for (int64_t s = 0; s < L; ++s) {
            float posterior = alpha[t * L + s] + beta[t * L + s];
            if (posterior > NEG_INF + 1.0f) {
                int32_t c = ext_label[s];
                float old = grad[t * C + c];
                grad[t * C + c] = (old == 0.0f)
                    ? posterior
                    : log_add(old, posterior);
            }
        }
        for (int64_t c = 0; c < C; ++c) {
            float lp = log_probs[t * lp_t_stride + c];
            float& slot = grad[t * C + c];
            if (slot != 0.0f) {
                slot = std::exp(lp) - std::exp(slot - logZ);
            } else {
                slot = std::exp(lp);
            }
        }
    }

    return {-logZ, std::move(grad)};
}

} // anonymous namespace

// Public kernel.
auto ctc_loss_forward_kernel(
    std::span<const Tensor> inputs,
    const OpAttributes& attrs
) -> std::vector<Tensor> {
    if (inputs.size() != 4) {
        throw std::invalid_argument(
            "ctc_loss_forward (CPU): expected 4 inputs "
            "[log_probs, targets, input_lengths, target_lengths]");
    }
    const Tensor& log_probs = inputs[0];
    const Tensor& targets = inputs[1];
    const Tensor& input_lengths = inputs[2];
    const Tensor& target_lengths = inputs[3];

    if (log_probs.shape().size() != 3) {
        throw std::invalid_argument(
            "ctc_loss_forward (CPU): log_probs must be 3D (T, N, C)");
    }
    int64_t T_max = log_probs.shape()[0];
    int64_t N     = log_probs.shape()[1];
    int64_t C     = log_probs.shape()[2];

    // Ensure Float32 / Int32 and contiguous.
    Tensor lp_f32  = log_probs.dtype() == DType::Float32
                     ? log_probs.contiguous()
                     : log_probs.to(DType::Float32).contiguous();
    Tensor tgt_i32 = targets.dtype() == DType::Int32
                     ? targets.contiguous()
                     : targets.to(DType::Int32).contiguous();
    Tensor il_i32  = input_lengths.dtype() == DType::Int32
                     ? input_lengths.contiguous()
                     : input_lengths.to(DType::Int32).contiguous();
    Tensor tl_i32  = target_lengths.dtype() == DType::Int32
                     ? target_lengths.contiguous()
                     : target_lengths.to(DType::Int32).contiguous();

    const float*   lp_data  = lp_f32.data<float>();
    const int32_t* tgt_data = tgt_i32.data<int32_t>();
    const int32_t* il_data  = il_i32.data<int32_t>();
    const int32_t* tl_data  = tl_i32.data<int32_t>();

    auto tgt_shape = tgt_i32.shape();
    int64_t S_max = tgt_shape.size() > 1 ? tgt_shape[1]
                  : tgt_shape.size() == 1 ? tgt_shape[0] : 0;

    int64_t blank = attrs.get_int(AttrKey::Blank, 0);
    bool zero_infinity = attrs.get_bool(AttrKey::ZeroInfinity, false);

    Tensor loss_out({N}, DType::Float32, log_probs.device());
    Tensor grad_out({T_max, N, C}, DType::Float32, log_probs.device());
    std::memset(loss_out.data<float>(), 0, sizeof(float) * static_cast<size_t>(N));
    std::memset(grad_out.data<float>(), 0,
                sizeof(float) * static_cast<size_t>(T_max * N * C));

    #pragma omp parallel for if(N > 4)
    for (int64_t n = 0; n < N; ++n) {
        int64_t T_n = il_data[n];
        int64_t S_n = tl_data[n];
        if (T_n <= 0 || S_n <= 0 || T_n > T_max || S_n > S_max) {
            continue;
        }

        // Extract per-batch log_probs into a contiguous (T_n, C) buffer.
        // Source layout is (T, N, C), so the t-stride is N*C and we pick
        // batch index n.
        std::vector<float> lp_n(T_n * C);
        for (int64_t t = 0; t < T_n; ++t) {
            for (int64_t c = 0; c < C; ++c) {
                lp_n[t * C + c] = lp_data[t * N * C + n * C + c];
            }
        }
        const int32_t* tgt_n = tgt_data + n * S_max;

        auto [loss, grad] = ctc_single(lp_n.data(), C, tgt_n, T_n, S_n, C, blank);

        if (zero_infinity && std::isinf(loss)) {
            loss = 0.0f;
            std::fill(grad.begin(), grad.end(), 0.0f);
        }

        loss_out.data<float>()[n] = loss;
        float* grad_dst = grad_out.data<float>();
        for (int64_t t = 0; t < T_n; ++t) {
            for (int64_t c = 0; c < C; ++c) {
                grad_dst[t * N * C + n * C + c] = grad[t * C + c];
            }
        }
    }

    return {loss_out, grad_out};
}

} // namespace cpu
} // namespace tenzor
