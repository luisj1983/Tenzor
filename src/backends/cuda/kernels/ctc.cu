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

// Numerically-stable log-domain addition. Templated on the compute type so the
// Float64 path keeps full double precision (matching the CPU reference, which
// runs the DP natively in double for Float64 inputs) while Float32/half inputs
// use single precision.
template<typename T>
__device__ __forceinline__ T ctc_log_add(T a, T b) {
    const T NEG_INF = -std::numeric_limits<T>::infinity();
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    T m = a > b ? a : b;
    return m + ::log1p(::exp(-::fabs(a - b)));
}

// Forward DP: fills alpha[t, s] for one batch element n.
// Each thread handles a stride of positions s within each timestep.
// Block-wide __syncthreads() separates timesteps.
template<typename T>
__global__ void ctc_forward_backward_kernel(
    const T* __restrict__ log_probs,       // (T_max, N, C)
    const int32_t* __restrict__ targets,   // (N, S_max)
    const int32_t* __restrict__ input_lengths,   // (N,)
    const int32_t* __restrict__ target_lengths,  // (N,)
    T* __restrict__ alpha_buf,             // (N, T_max, L_max) workspace
    T* __restrict__ beta_buf,              // (N, T_max, L_max) workspace
    T* __restrict__ post_scratch,          // (N, T_max, C) per-block log-posterior scratch
    T* __restrict__ loss_out,              // (N,)
    T* __restrict__ grad_out,              // (T_max, N, C)
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

    const T NEG_INF = -std::numeric_limits<T>::infinity();

    // Compute per-batch base pointers.
    T* alpha = alpha_buf + n * T_max * L_max;
    T* beta  = beta_buf  + n * T_max * L_max;
    const int32_t* tgt_n = targets + n * S_max;

    // Helper: ext_label[s].
    auto ext_label = [&] __device__ (int64_t s) -> int32_t {
        return (s % 2 == 0) ? static_cast<int32_t>(blank) : tgt_n[s / 2];
    };

    if (T_n <= 0 || S_n <= 0 || L_n > L_max) {
        // Degenerate case: zero loss, zero grad.
        if (tid == 0) loss_out[n] = T(0);
        for (int64_t idx = tid; idx < T_max * C; idx += nthreads) {
            int64_t t = idx / C;
            int64_t c = idx % C;
            grad_out[t * N * C + n * C + c] = T(0);
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
            T a = alpha[(t - 1) * L_max + s];
            if (s > 0) {
                a = ctc_log_add(a, alpha[(t - 1) * L_max + (s - 1)]);
            }
            int32_t c_s = ext_label(s);
            if (s > 1 && c_s != blank && c_s != ext_label(s - 2)) {
                a = ctc_log_add(a, alpha[(t - 1) * L_max + (s - 2)]);
            }
            alpha[t * L_max + s] = a + log_probs[t * N * C + n * C + c_s];
        }
        __syncthreads();
    }

    // Total log-probability.
    __shared__ T s_logZ;
    if (tid == 0) {
        T term1 = alpha[(T_n - 1) * L_max + (L_n - 1)];
        T term2 = (L_n > 1) ? alpha[(T_n - 1) * L_max + (L_n - 2)] : NEG_INF;
        s_logZ = ctc_log_add(term1, term2);
    }
    __syncthreads();
    T logZ = s_logZ;

    // ----------------------------------------------------------------
    // Backward pass: beta[t, s]
    // ----------------------------------------------------------------

    // Initialise beta at t = T_n - 1.
    for (int64_t s = tid; s < L_n; s += nthreads) {
        if (s == L_n - 1 || (s == L_n - 2 && L_n > 1)) {
            beta[(T_n - 1) * L_max + s] = T(0);  // log(1)
        } else {
            beta[(T_n - 1) * L_max + s] = NEG_INF;
        }
    }
    __syncthreads();

    for (int64_t t = T_n - 2; t >= 0; --t) {
        for (int64_t s = tid; s < L_n; s += nthreads) {
            int32_t c_s = ext_label(s);
            T b = beta[(t + 1) * L_max + s] + log_probs[(t + 1) * N * C + n * C + c_s];
            if (s < L_n - 1) {
                int32_t c_s1 = ext_label(s + 1);
                T term = beta[(t + 1) * L_max + (s + 1)]
                       + log_probs[(t + 1) * N * C + n * C + c_s1];
                b = ctc_log_add(b, term);
            }
            if (s < L_n - 2 && c_s != blank && c_s != ext_label(s + 2)) {
                int32_t c_s2 = ext_label(s + 2);
                T term = beta[(t + 1) * L_max + (s + 2)]
                       + log_probs[(t + 1) * N * C + n * C + c_s2];
                b = ctc_log_add(b, term);
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
    T per_sample_loss = -logZ;
    bool is_inf = !isfinite(static_cast<double>(per_sample_loss));
    if (zero_infinity && is_inf) {
        per_sample_loss = T(0);
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
            grad_out[t * N * C + n * C + c] = T(0);
        }
        return;
    }

    // Per-block scratch slice [(t, c)] inside post_scratch[n, *, *].
    T* post_n = post_scratch + n * T_max * C;

    // First, zero the grad slice for this batch element (handles t >= T_n
    // padding) and seed the scratch with NEG_INF so log_add picks up the
    // first posterior cleanly.
    for (int64_t idx = tid; idx < T_max * C; idx += nthreads) {
        int64_t t = idx / C;
        int64_t c = idx % C;
        grad_out[t * N * C + n * C + c] = T(0);
        post_n[t * C + c] = NEG_INF;
    }
    __syncthreads();

    // Pass 1: accumulate log-posteriors per (t, c) in private scratch.
    // Parallelised over t across the block's threads: each t owns its own
    // post_n[t, *] row, so distinct threads never touch the same slot and no
    // synchronisation is needed within the loop. The inner s loop stays serial
    // per t because several s can map to the same c (their log_add must be
    // ordered), exactly matching the CPU reference's per-t accumulation.
    for (int64_t t = tid; t < T_n; t += nthreads) {
        T* post_t = post_n + t * C;
        const T* alpha_t = alpha + t * L_max;
        const T* beta_t = beta + t * L_max;
        for (int64_t s = 0; s < L_n; ++s) {
            int32_t c = ext_label(s);
            T posterior = alpha_t[s] + beta_t[s];
            post_t[c] = ctc_log_add(post_t[c], posterior);
        }
    }
    __syncthreads();

    // Pass 2: convert log-posterior to gradient by combining with
    // exp(log_probs) and subtracting. All threads parallelise over (t, c).
    for (int64_t idx = tid; idx < T_n * C; idx += nthreads) {
        int64_t t = idx / C;
        int64_t c = idx % C;
        T lp = log_probs[t * N * C + n * C + c];
        T post = post_n[t * C + c];
        T prob = ::exp(lp);
        T post_prob = (post == NEG_INF) ? T(0) : ::exp(post - logZ);
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
    // Compute dtype follows the CPU CTC reference (S13 dtype-preservation):
    //   Float32: native single precision.
    //   Float64: native DOUBLE precision DP (no silent downcast) — the kernel is
    //            instantiated with T=double below.
    //   Float16/BFloat16/other: widen to Float32 (CTC is precision-sensitive and
    //            half-precision logZ saturates quickly), compute, narrow outputs.
    if (log_probs.dtype() != DType::Float32 && log_probs.dtype() != DType::Float64) {
        const DType orig = log_probs.dtype();
        auto outs = ctc_loss_forward_kernel(
            log_probs.to(DType::Float32), targets, input_lengths, target_lengths,
            blank, zero_infinity, stream);
        for (auto& t : outs) { t = t.to(orig); }
        return outs;
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

    // Native compute dtype: Float64 stays double, Float32 stays float.
    const DType compute_dtype = log_probs.dtype();
    const bool native_f64 = (compute_dtype == DType::Float64);

    // ----------------------------------------------------------------
    // Validate the blank label and every active target label up front.
    // ext_label[i] (== blank or a target label) is used directly as a channel
    // index into log_probs / grad inside the kernel; an out-of-range value
    // would cause an OOB device read of log_probs or an OOB device write into
    // grad. The CPU reference performs the same validation host-side; mirror it
    // here by copying the (small) Int32 targets / length tensors to host. This
    // matches CPU parity and prevents user-data-driven memory corruption.
    // ----------------------------------------------------------------
    if (blank < 0 || blank >= C) {
        throw std::invalid_argument(
            "ctc_loss_forward (CUDA): blank index " + std::to_string(blank) +
            " out of range [0, " + std::to_string(C) + ")");
    }
    if (N > 0 && S_max > 0) {
        Tensor tgt_cpu = targets.to(Device::cpu()).contiguous();
        Tensor il_cpu  = input_lengths.to(Device::cpu()).contiguous();
        Tensor tl_cpu  = target_lengths.to(Device::cpu()).contiguous();
        const int32_t* tgt_data = tgt_cpu.data<int32_t>();
        const int32_t* il_data  = il_cpu.data<int32_t>();
        const int32_t* tl_data  = tl_cpu.data<int32_t>();
        for (int64_t n = 0; n < N; ++n) {
            int64_t T_n = il_data[n];
            int64_t S_n = tl_data[n];
            if (T_n <= 0 || S_n <= 0 || T_n > T_max || S_n > S_max) {
                continue;  // inactive sample; skipped by the compute kernel too
            }
            const int32_t* tgt_n = tgt_data + n * S_max;
            for (int64_t s = 0; s < S_n; ++s) {
                const int64_t label = static_cast<int64_t>(tgt_n[s]);
                if (label < 0 || label >= C) {
                    throw std::invalid_argument(
                        "ctc_loss_forward (CUDA): target label " +
                        std::to_string(label) + " out of range [0, " +
                        std::to_string(C) + ")");
                }
            }
        }
    }

    // Outputs (in the native compute dtype).
    Tensor loss_out({N}, compute_dtype, log_probs.device());
    Tensor grad_out({T_max, N, C}, compute_dtype, log_probs.device());

    // Workspace.
    // alpha and beta are (N, T_max, L_max) in the compute dtype. For T*N*L_max
    // large this can be substantial, but matches the algorithm's natural
    // working set.
    int64_t alpha_elems = N * T_max * L_max;
    Tensor alpha_buf({alpha_elems}, compute_dtype, log_probs.device());
    Tensor beta_buf({alpha_elems}, compute_dtype, log_probs.device());
    // AA.10: per-block (T_max, C) scratch for log-space posterior
    // accumulation — must not alias grad_out, or partial writes from
    // iteration t leak into iteration t+1's readers.
    int64_t post_elems = N * T_max * C;
    Tensor post_scratch({post_elems}, compute_dtype, log_probs.device());

    if (N == 0 || T_max == 0 || C == 0) {
        // Zero-sized tensors: nothing to do. Zero out outputs for safety.
        if (loss_out.numel() > 0) {
            TENZOR_CUDA_CHECK(cudaMemsetAsync(
                loss_out.data_ptr(), 0,
                loss_out.numel() * dtype_size(compute_dtype), stream));
        }
        if (grad_out.numel() > 0) {
            TENZOR_CUDA_CHECK(cudaMemsetAsync(
                grad_out.data_ptr(), 0,
                grad_out.numel() * dtype_size(compute_dtype), stream));
        }
        return {loss_out, grad_out};
    }

    // Launch: one block per batch element.
    dim3 grid(static_cast<unsigned>(N));
    dim3 block(CTC_THREADS_PER_BLOCK);

    if (native_f64) {
        ctc_forward_backward_kernel<double><<<grid, block, 0, stream>>>(
            log_probs.data<double>(),
            targets.data<int32_t>(),
            input_lengths.data<int32_t>(),
            target_lengths.data<int32_t>(),
            alpha_buf.data<double>(),
            beta_buf.data<double>(),
            post_scratch.data<double>(),
            loss_out.data<double>(),
            grad_out.data<double>(),
            T_max, N, C, S_max, L_max,
            blank, zero_infinity
        );
    } else {
        ctc_forward_backward_kernel<float><<<grid, block, 0, stream>>>(
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
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return {loss_out, grad_out};
}

} // namespace cuda
} // namespace tenzor
