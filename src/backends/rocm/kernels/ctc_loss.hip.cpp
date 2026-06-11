/**
 * @file ctc_loss.hip.cpp
 * @brief HIP/ROCm port of the CTC (Connectionist Temporal Classification)
 *        forward-backward DP kernel.
 *
 * Mirrors src/backends/cuda/kernels/ctc.cu line-for-line: one HIP block per
 * batch element, log-domain alpha/beta over extended labels, gradient
 * computed in-place.
 *
 *   inputs:  [log_probs (T, N, C) Float32,
 *             targets (N, S_max) Int32,
 *             input_lengths (N,) Int32,
 *             target_lengths (N,) Int32]
 *   attrs:   Blank (int, default 0), ZeroInfinity (bool, default false)
 *   outputs: [loss_per_sample (N,) Float32, raw_grad (T, N, C) Float32]
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {
namespace rocm {

#ifndef HIP_CHECK
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)
#endif

namespace {

constexpr int CTC_THREADS_PER_BLOCK = 128;

__device__ __forceinline__ float log_add(float a, float b) {
    // -fno-fast-math is forced on this file via the ROCm CMakeLists so
    // -INFINITY semantics are reliable; the global -ffast-math flag would
    // otherwise let the compiler eliminate -inf-comparison branches.
    constexpr float NEG_INF = -INFINITY;
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    float m = fmaxf(a, b);
    return m + log1pf(expf(-fabsf(a - b)));
}

__global__ void ctc_forward_backward_kernel(
    const float* __restrict__ log_probs,
    const int32_t* __restrict__ targets,
    const int32_t* __restrict__ input_lengths,
    const int32_t* __restrict__ target_lengths,
    float* __restrict__ alpha_buf,
    float* __restrict__ beta_buf,
    float* __restrict__ post_scratch,  // (N, T_max, C) log-posterior scratch
    float* __restrict__ loss_out,
    float* __restrict__ grad_out,
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

    // -fno-fast-math is forced on this file via the ROCm CMakeLists so
    // -INFINITY semantics are reliable; the global -ffast-math flag would
    // otherwise let the compiler eliminate -inf-comparison branches.
    constexpr float NEG_INF = -INFINITY;

    float* alpha = alpha_buf + n * T_max * L_max;
    float* beta  = beta_buf  + n * T_max * L_max;
    const int32_t* tgt_n = targets + n * S_max;

    auto ext_label = [&] __device__ (int64_t s) -> int32_t {
        return (s % 2 == 0) ? static_cast<int32_t>(blank) : tgt_n[s / 2];
    };

    if (T_n <= 0 || S_n <= 0 || L_n > L_max) {
        if (tid == 0) loss_out[n] = 0.0f;
        for (int64_t idx = tid; idx < T_max * C; idx += nthreads) {
            int64_t t = idx / C;
            int64_t c = idx % C;
            grad_out[t * N * C + n * C + c] = 0.0f;
        }
        return;
    }

    // Forward DP initialisation.
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

    __shared__ float s_logZ;
    if (tid == 0) {
        float term1 = alpha[(T_n - 1) * L_max + (L_n - 1)];
        float term2 = (L_n > 1) ? alpha[(T_n - 1) * L_max + (L_n - 2)] : NEG_INF;
        s_logZ = log_add(term1, term2);
    }
    __syncthreads();
    float logZ = s_logZ;

    // Backward DP initialisation.
    for (int64_t s = tid; s < L_n; s += nthreads) {
        if (s == L_n - 1 || (s == L_n - 2 && L_n > 1)) {
            beta[(T_n - 1) * L_max + s] = 0.0f;
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

    float per_sample_loss = -logZ;
    bool is_inf = !isfinite(per_sample_loss);
    if (zero_infinity && is_inf) {
        per_sample_loss = 0.0f;
    }
    if (tid == 0) {
        loss_out[n] = per_sample_loss;
    }

    if (zero_infinity && is_inf) {
        for (int64_t idx = tid; idx < T_max * C; idx += nthreads) {
            int64_t t = idx / C;
            int64_t c = idx % C;
            grad_out[t * N * C + n * C + c] = 0.0f;
        }
        return;
    }

    // AA.10 fix: accumulate log-posteriors into a dedicated per-block
    // scratch tile (post_scratch[n, t, c]) for all t first; only after
    // every t has been processed do we read log_probs and convert to the
    // final gradient. The previous in-place implementation aliased the
    // global grad_out cell as both log-space accumulator and final
    // exp-space gradient — partial writes from iteration t leaked into
    // iteration t+1 readers, producing +inf grads. CPU reference uses a
    // private std::vector per batch element; this is the GPU equivalent.
    float* post_n = post_scratch + n * T_max * C;

    for (int64_t idx = tid; idx < T_max * C; idx += nthreads) {
        int64_t t = idx / C;
        int64_t c = idx % C;
        grad_out[t * N * C + n * C + c] = 0.0f;
        post_n[t * C + c] = NEG_INF;
    }
    __syncthreads();

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

    for (int64_t idx = tid; idx < T_n * C; idx += nthreads) {
        int64_t t = idx / C;
        int64_t c = idx % C;
        float lp = log_probs[t * N * C + n * C + c];
        float post = post_n[t * C + c];
        float prob = expf(lp);
        float post_prob = (post == NEG_INF) ? 0.0f : expf(post - logZ);
        grad_out[t * N * C + n * C + c] = prob - post_prob;
    }
}

} // anonymous namespace

auto ctc_loss_forward_kernel(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t blank,
    bool zero_infinity,
    hipStream_t stream
) -> std::vector<Tensor> {
    // Float64: compute in Float32 on-device (the alpha/beta recursion runs in
    // log-space where single precision is ample), then narrow loss and grad
    // back to Float64 so the output dtype matches the input. Stays GPU-resident.
    if (log_probs.dtype() == DType::Float64) {
        auto lp32 = log_probs.to(DType::Float32);
        auto results = ctc_loss_forward_kernel(lp32, targets, input_lengths,
                                               target_lengths, blank, zero_infinity, stream);
        for (auto& t : results) t = t.to(DType::Float64);
        return results;
    }
    if (log_probs.dtype() != DType::Float32) {
        throw std::invalid_argument(
            "ctc_loss_forward (ROCm): log_probs must be Float32");
    }
    if (targets.dtype() != DType::Int32 ||
        input_lengths.dtype() != DType::Int32 ||
        target_lengths.dtype() != DType::Int32) {
        throw std::invalid_argument(
            "ctc_loss_forward (ROCm): targets / input_lengths / target_lengths "
            "must be Int32");
    }
    auto lp_shape = log_probs.shape();
    if (lp_shape.size() != 3) {
        throw std::invalid_argument(
            "ctc_loss_forward (ROCm): log_probs must be 3D (T, N, C)");
    }
    int64_t T_max = lp_shape[0];
    int64_t N = lp_shape[1];
    int64_t C = lp_shape[2];

    auto tgt_shape = targets.shape();
    int64_t S_max = (tgt_shape.size() >= 2) ? tgt_shape[1]
                  : (tgt_shape.size() == 1) ? tgt_shape[0] : 0;
    int64_t L_max = 2 * S_max + 1;

    Tensor loss_out({N}, DType::Float32, log_probs.device());
    Tensor grad_out({T_max, N, C}, DType::Float32, log_probs.device());

    int64_t alpha_elems = N * T_max * L_max;
    Tensor alpha_buf({alpha_elems}, DType::Float32, log_probs.device());
    Tensor beta_buf({alpha_elems}, DType::Float32, log_probs.device());
    // AA.10: per-block (T_max, C) scratch for log-space posterior
    // accumulation — must not alias grad_out.
    int64_t post_elems = N * T_max * C;
    Tensor post_scratch({post_elems}, DType::Float32, log_probs.device());

    if (N == 0 || T_max == 0 || C == 0) {
        if (loss_out.numel() > 0) {
            HIP_CHECK(hipMemsetAsync(loss_out.data_ptr(), 0,
                                     loss_out.numel() * sizeof(float), stream));
        }
        if (grad_out.numel() > 0) {
            HIP_CHECK(hipMemsetAsync(grad_out.data_ptr(), 0,
                                     grad_out.numel() * sizeof(float), stream));
        }
        return {loss_out, grad_out};
    }

    dim3 grid(static_cast<unsigned>(N));
    dim3 block(CTC_THREADS_PER_BLOCK);

    hipLaunchKernelGGL(ctc_forward_backward_kernel,
                       grid, block, 0, stream,
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
                       blank, zero_infinity);
    HIP_CHECK(hipGetLastError());

    return {loss_out, grad_out};
}

} // namespace rocm
} // namespace tenzor
