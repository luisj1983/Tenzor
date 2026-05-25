/**
 * @file ctc.cu
 * @brief CUDA kernels for CTC (Connectionist Temporal Classification) loss.
 *
 * Implements the forward-backward dynamic programming algorithm for CTC
 * loss + gradient computation entirely on device, eliminating the prior
 * CPU round-trip in src/nn/loss/losses_advanced.cpp.
 *
 * Algorithm: log-domain forward-backward over an extended label sequence
 *   ext_label[i] = blank        if i even
 *                = target[i/2]   if i odd     (length L = 2*S + 1)
 *
 * Forward variable alpha[t, s] = log P(emit ext_label[0..s] up to time t)
 * Backward variable beta[t, s] = log P(emit ext_label[s..L-1] from time t)
 *
 * Total log-prob = log_sum_exp(alpha[T-1, L-1], alpha[T-1, L-2])
 *
 * Gradient w.r.t. log_probs:
 *   grad[t, c] = exp(log_probs[t, c])
 *              - sum_{s: ext_label[s] == c} exp(alpha[t,s] + beta[t,s] - logZ)
 *
 * Parallelisation strategy: one CUDA block per batch element. Threads
 * within the block parallelise over the L extended-label positions for
 * each timestep. T iterations are sequential (state recurrence).
 *
 * Reference: Graves et al. (2006), PyTorch aten/src/ATen/native/cuda/LossCTC.cu.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include <limits>
#include <stdexcept>

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "cuda_common.cuh"

namespace tenzor {
namespace cuda {

namespace {

// Maximum supported extended-label length per batch element (covers
// targets up to 1023 tokens; alpha/beta are kept in global memory so
// there's no shared-memory constraint, but we still bound the launch).
// For sequences longer than this, the kernel falls back to one-thread-per-
// batch-element loops (each thread iterates over s within its t).
constexpr int CTC_THREADS_PER_BLOCK = 128;

__device__ __forceinline__ float log_add(float a, float b) {
    constexpr float NEG_INF = -INFINITY;
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    float m = fmaxf(a, b);
    return m + log1pf(expf(-fabsf(a - b)));
}

// Forward DP: fills alpha[t, s] for one batch element n.
// Each thread handles a stride of positions s within each timestep.
// Block-wide __syncthreads() separates timesteps.
__global__ void ctc_forward_backward_kernel(
    const float* __restrict__ log_probs,   // (T_max, N, C)
    const int32_t* __restrict__ targets,   // (N, S_max)
    const int32_t* __restrict__ input_lengths,   // (N,)
    const int32_t* __restrict__ target_lengths,  // (N,)
    float* __restrict__ alpha_buf,         // (N, T_max, L_max) workspace
    float* __restrict__ beta_buf,          // (N, T_max, L_max) workspace
    float* __restrict__ post_scratch,      // (N, T_max, C) per-block log-posterior scratch
    float* __restrict__ loss_out,          // (N,)
    float* __restrict__ grad_out,          // (T_max, N, C)
    int64_t T_max,
    int64_t N,
    int64_t C,
    int64_t S_max,
    int64_t L_max,
    int64_t blank,
    bool zero_infinity
) {
    const int n = blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int32_t T_n = input_lengths[n];
    const int32_t S_n = target_lengths[n];
    const int64_t L_n = 2 * S_n + 1;

    constexpr float NEG_INF = -INFINITY;

    // Compute per-batch base pointers.
    float* alpha = alpha_buf + n * T_max * L_max;
    float* beta  = beta_buf  + n * T_max * L_max;
    const int32_t* tgt_n = targets + n * S_max;

    // Helper: ext_label[s].
    auto ext_label = [&] __device__ (int64_t s) -> int32_t {
        return (s % 2 == 0) ? static_cast<int32_t>(blank) : tgt_n[s / 2];
    };

    if (T_n <= 0 || S_n <= 0 || L_n > L_max) {
        // Degenerate case: zero loss, zero grad.
        if (tid == 0) loss_out[n] = 0.0f;
        for (int64_t idx = tid; idx < T_max * C; idx += nthreads) {
            int64_t t = idx / C;
            int64_t c = idx % C;
            grad_out[t * N * C + n * C + c] = 0.0f;
        }
        return;
    }

    // ----------------------------------------------------------------
    // Forward pass: alpha[t, s]
    // ----------------------------------------------------------------

    // Initialise alpha at t=0.
    for (int64_t s = tid; s < L_n; s += nthreads) {
        if (s == 0) {
            alpha[0 * L_max + 0] = log_probs[0 * N * C + n * C + ext_label(0)];
        } else if (s == 1 && L_n > 1) {
            alpha[0 * L_max + 1] = log_probs[0 * N * C + n * C + ext_label(1)];
        } else {
            alpha[0 * L_max + s] = NEG_INF;
        }
    }
    __syncthreads();

    // Recurrence over t.
    for (int64_t t = 1; t < T_n; ++t) {
        for (int64_t s = tid; s < L_n; s += nthreads) {
            float a = alpha[(t - 1) * L_max + s];
            if (s > 0) {
                a = log_add(a, alpha[(t - 1) * L_max + (s - 1)]);
            }
            int32_t c_s = ext_label(s);
            if (s > 1 && c_s != blank && c_s != ext_label(s - 2)) {
                a = log_add(a, alpha[(t - 1) * L_max + (s - 2)]);
            }
            alpha[t * L_max + s] = a + log_probs[t * N * C + n * C + c_s];
        }
        __syncthreads();
    }

    // Total log-probability.
    __shared__ float s_logZ;
    if (tid == 0) {
        float term1 = alpha[(T_n - 1) * L_max + (L_n - 1)];
        float term2 = (L_n > 1) ? alpha[(T_n - 1) * L_max + (L_n - 2)] : NEG_INF;
        s_logZ = log_add(term1, term2);
    }
    __syncthreads();
    float logZ = s_logZ;

    // ----------------------------------------------------------------
    // Backward pass: beta[t, s]
    // ----------------------------------------------------------------

    // Initialise beta at t = T_n - 1.
    for (int64_t s = tid; s < L_n; s += nthreads) {
        if (s == L_n - 1 || (s == L_n - 2 && L_n > 1)) {
            beta[(T_n - 1) * L_max + s] = 0.0f;  // log(1)
        } else {
            beta[(T_n - 1) * L_max + s] = NEG_INF;
        }
    }
    __syncthreads();

    for (int64_t t = T_n - 2; t >= 0; --t) {
        for (int64_t s = tid; s < L_n; s += nthreads) {
            int32_t c_s = ext_label(s);
            float b = beta[(t + 1) * L_max + s] + log_probs[(t + 1) * N * C + n * C + c_s];
            if (s < L_n - 1) {
                int32_t c_s1 = ext_label(s + 1);
                float term = beta[(t + 1) * L_max + (s + 1)]
                           + log_probs[(t + 1) * N * C + n * C + c_s1];
                b = log_add(b, term);
            }
            if (s < L_n - 2 && c_s != blank && c_s != ext_label(s + 2)) {
                int32_t c_s2 = ext_label(s + 2);
                float term = beta[(t + 1) * L_max + (s + 2)]
                           + log_probs[(t + 1) * N * C + n * C + c_s2];
                b = log_add(b, term);
            }
            beta[t * L_max + s] = b;
        }
        __syncthreads();
    }

    // ----------------------------------------------------------------
    // Loss
    // ----------------------------------------------------------------

    // PyTorch reports per-sample loss as -logZ. zero_infinity replaces
    // infinite losses (which arise when no alignment exists, e.g. T < S)
    // by 0 and zeroes their gradients.
    float per_sample_loss = -logZ;
    bool is_inf = !isfinite(per_sample_loss);
    if (zero_infinity && is_inf) {
        per_sample_loss = 0.0f;
    }
    if (tid == 0) {
        loss_out[n] = per_sample_loss;
    }

    // ----------------------------------------------------------------
    // Gradient w.r.t. log_probs[:, n, :]
    //
    //   grad[t, c] = exp(log_probs[t, c])
    //              - sum_{s : ext_label[s] == c} exp(alpha[t,s] + beta[t,s] - logZ)
    //
    // AA.10 fix: accumulate log-posteriors into a dedicated per-block
    // scratch tile (post_scratch[(n, t, c)]) for *all* t first; only after
    // every t has been processed do we read log_probs and convert to the
    // final gradient.  The previous in-place implementation aliased the
    // global grad_out cell as both the log-space accumulator (initialised
    // to NEG_INF) and the final exp-space gradient — partial writes from
    // iteration t leaked into iteration t+1 readers on backends that
    // cache global memory across __syncthreads(), producing +inf grads.
    // The CPU reference avoids this by using a private std::vector per
    // batch element; the scratch tile is the GPU equivalent.
    // ----------------------------------------------------------------

    if (zero_infinity && is_inf) {
        // Zero out the entire (T_max, n, :) row of grad and return.
        for (int64_t idx = tid; idx < T_max * C; idx += nthreads) {
            int64_t t = idx / C;
            int64_t c = idx % C;
            grad_out[t * N * C + n * C + c] = 0.0f;
        }
        return;
    }

    // Per-block scratch slice [(t, c)] inside post_scratch[n, *, *].
    float* post_n = post_scratch + n * T_max * C;

    // First, zero the grad slice for this batch element (handles t >= T_n
    // padding) and seed the scratch with NEG_INF so log_add picks up the
    // first posterior cleanly.
    for (int64_t idx = tid; idx < T_max * C; idx += nthreads) {
        int64_t t = idx / C;
        int64_t c = idx % C;
        grad_out[t * N * C + n * C + c] = 0.0f;
        post_n[t * C + c] = NEG_INF;
    }
    __syncthreads();

    // Pass 1: accumulate log-posteriors per (t, c) in private scratch.
    // Serialised per t over L_n positions (which is small), parallelised
    // over t when L is small enough that no thread is the bottleneck.
    // For simplicity we keep the proven single-thread accumulation over s
    // (matches CPU reference exactly).
    if (tid == 0) {
        for (int64_t t = 0; t < T_n; ++t) {
            for (int64_t s = 0; s < L_n; ++s) {
                int32_t c = ext_label(s);
                float posterior = alpha[t * L_max + s] + beta[t * L_max + s];
                float& slot = post_n[t * C + c];
                slot = log_add(slot, posterior);
            }
        }
    }
    __syncthreads();

    // Pass 2: convert log-posterior to gradient by combining with
    // exp(log_probs) and subtracting. All threads parallelise over (t, c).
    for (int64_t idx = tid; idx < T_n * C; idx += nthreads) {
        int64_t t = idx / C;
        int64_t c = idx % C;
        float lp = log_probs[t * N * C + n * C + c];
        float post = post_n[t * C + c];
        float prob = expf(lp);
        float post_prob = (post == NEG_INF) ? 0.0f : expf(post - logZ);
        grad_out[t * N * C + n * C + c] = prob - post_prob;
    }

    // Pad: t in [T_n, T_max) is already zeroed above.
}

} // anonymous namespace

// Public entry. Returns {loss_per_sample (N,), raw_grad (T, N, C)}.
auto ctc_loss_forward_kernel(
    const Tensor& log_probs,        // (T, N, C) Float32
    const Tensor& targets,          // (N, S_max) Int32
    const Tensor& input_lengths,    // (N,) Int32
    const Tensor& target_lengths,   // (N,) Int32
    int64_t blank,
    bool zero_infinity,
    cudaStream_t stream
) -> std::vector<Tensor> {
    if (log_probs.dtype() != DType::Float32) {
        throw std::invalid_argument(
            "ctc_loss_forward (CUDA): log_probs must be Float32");
    }
    if (targets.dtype() != DType::Int32 ||
        input_lengths.dtype() != DType::Int32 ||
        target_lengths.dtype() != DType::Int32) {
        throw std::invalid_argument(
            "ctc_loss_forward (CUDA): targets / input_lengths / target_lengths "
            "must be Int32");
    }
    auto lp_shape = log_probs.shape();
    if (lp_shape.size() != 3) {
        throw std::invalid_argument(
            "ctc_loss_forward (CUDA): log_probs must be 3D (T, N, C)");
    }
    int64_t T_max = lp_shape[0];
    int64_t N = lp_shape[1];
    int64_t C = lp_shape[2];

    auto tgt_shape = targets.shape();
    int64_t S_max = (tgt_shape.size() >= 2) ? tgt_shape[1]
                  : (tgt_shape.size() == 1) ? tgt_shape[0] : 0;
    int64_t L_max = 2 * S_max + 1;

    // Outputs.
    Tensor loss_out({N}, DType::Float32, log_probs.device());
    Tensor grad_out({T_max, N, C}, DType::Float32, log_probs.device());

    // Workspace.
    // alpha and beta are (N, T_max, L_max) Float32. For T*N*L_max large
    // this can be substantial, but matches the algorithm's natural
    // working set.
    int64_t alpha_elems = N * T_max * L_max;
    Tensor alpha_buf({alpha_elems}, DType::Float32, log_probs.device());
    Tensor beta_buf({alpha_elems}, DType::Float32, log_probs.device());
    // AA.10: per-block (T_max, C) scratch for log-space posterior
    // accumulation — must not alias grad_out, or partial writes from
    // iteration t leak into iteration t+1's readers.
    int64_t post_elems = N * T_max * C;
    Tensor post_scratch({post_elems}, DType::Float32, log_probs.device());

    if (N == 0 || T_max == 0 || C == 0) {
        // Zero-sized tensors: nothing to do. Zero out outputs for safety.
        if (loss_out.numel() > 0) {
            TENZOR_CUDA_CHECK(cudaMemsetAsync(
                loss_out.data_ptr(), 0,
                loss_out.numel() * sizeof(float), stream));
        }
        if (grad_out.numel() > 0) {
            TENZOR_CUDA_CHECK(cudaMemsetAsync(
                grad_out.data_ptr(), 0,
                grad_out.numel() * sizeof(float), stream));
        }
        return {loss_out, grad_out};
    }

    // Launch: one block per batch element.
    dim3 grid(static_cast<unsigned>(N));
    dim3 block(CTC_THREADS_PER_BLOCK);

    ctc_forward_backward_kernel<<<grid, block, 0, stream>>>(
        log_probs.data<float>(),
        targets.data<int32_t>(),
        input_lengths.data<int32_t>(),
        target_lengths.data<int32_t>(),
        alpha_buf.data<float>(),
        beta_buf.data<float>(),
        post_scratch.data<float>(),
        loss_out.data<float>(),
        grad_out.data<float>(),
        T_max, N, C, S_max, L_max,
        blank, zero_infinity
    );
    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return {loss_out, grad_out};
}

} // namespace cuda
} // namespace tenzor
