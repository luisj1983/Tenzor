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

// Templated on the compute type: Float64 inputs keep full double precision
// (matching the CPU reference and the CUDA path, which run the DP natively in
// double for Float64 inputs); Float32/half inputs use single precision.
template<typename T>
__device__ __forceinline__ T log_add(T a, T b) {
    // -fno-fast-math is forced on this file via the ROCm CMakeLists so
    // -INFINITY semantics are reliable; the global -ffast-math flag would
    // otherwise let the compiler eliminate -inf-comparison branches.
    const T NEG_INF = -std::numeric_limits<T>::infinity();
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    T m = a > b ? a : b;
    return m + ::log1p(::exp(-::fabs(a - b)));
}

template<typename T>
__global__ void ctc_forward_backward_kernel(
    const T* __restrict__ log_probs,
    const int32_t* __restrict__ targets,
    const int32_t* __restrict__ input_lengths,
    const int32_t* __restrict__ target_lengths,
    T* __restrict__ alpha_buf,
    T* __restrict__ beta_buf,
    T* __restrict__ post_scratch,  // (N, T_max, C) log-posterior scratch
    T* __restrict__ loss_out,
    T* __restrict__ grad_out,
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
    const T NEG_INF = -std::numeric_limits<T>::infinity();

    T* alpha = alpha_buf + n * T_max * L_max;
    T* beta  = beta_buf  + n * T_max * L_max;
    const int32_t* tgt_n = targets + n * S_max;

    auto ext_label = [&] __device__ (int64_t s) -> int32_t {
        return (s % 2 == 0) ? static_cast<int32_t>(blank) : tgt_n[s / 2];
    };

    // F116: include T_n > T_max — otherwise input_lengths[n] > T_max drives the
    // forward recurrence to index alpha/beta (sized [T_max, L_max]) past their
    // allocation and writes grad_out out of bounds (device heap corruption).
    if (T_n <= 0 || S_n <= 0 || L_n > L_max || T_n > T_max) {
        if (tid == 0) loss_out[n] = T(0);
        for (int64_t idx = tid; idx < T_max * C; idx += nthreads) {
            int64_t t = idx / C;
            int64_t c = idx % C;
            grad_out[t * N * C + n * C + c] = T(0);
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
            T a = alpha[(t - 1) * L_max + s];
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

    __shared__ T s_logZ;
    if (tid == 0) {
        T term1 = alpha[(T_n - 1) * L_max + (L_n - 1)];
        T term2 = (L_n > 1) ? alpha[(T_n - 1) * L_max + (L_n - 2)] : NEG_INF;
        s_logZ = log_add(term1, term2);
    }
    __syncthreads();
    T logZ = s_logZ;

    // Backward DP initialisation.
    for (int64_t s = tid; s < L_n; s += nthreads) {
        if (s == L_n - 1 || (s == L_n - 2 && L_n > 1)) {
            beta[(T_n - 1) * L_max + s] = T(0);
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
                b = log_add(b, term);
            }
            if (s < L_n - 2 && c_s != blank && c_s != ext_label(s + 2)) {
                int32_t c_s2 = ext_label(s + 2);
                T term = beta[(t + 1) * L_max + (s + 2)]
                           + log_probs[(t + 1) * N * C + n * C + c_s2];
                b = log_add(b, term);
            }
            beta[t * L_max + s] = b;
        }
        __syncthreads();
    }

    T per_sample_loss = -logZ;
    // F114: zero_infinity zeroes only +/-inf losses (matches CPU's std::isinf); a
    // NaN loss must propagate. The old !isfinite also swallowed NaN, diverging
    // from CPU/CUDA. Double-cast for reliable isinf semantics (as CUDA does).
    bool is_inf = isinf(static_cast<double>(per_sample_loss));
    if (zero_infinity && is_inf) {
        per_sample_loss = T(0);
    }
    if (tid == 0) {
        loss_out[n] = per_sample_loss;
    }

    if (zero_infinity && is_inf) {
        for (int64_t idx = tid; idx < T_max * C; idx += nthreads) {
            int64_t t = idx / C;
            int64_t c = idx % C;
            grad_out[t * N * C + n * C + c] = T(0);
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
    T* post_n = post_scratch + n * T_max * C;

    for (int64_t idx = tid; idx < T_max * C; idx += nthreads) {
        int64_t t = idx / C;
        int64_t c = idx % C;
        grad_out[t * N * C + n * C + c] = T(0);
        post_n[t * C + c] = NEG_INF;
    }
    __syncthreads();

    // Parallelize the log-posterior accumulation over t: each thread owns one
    // or more distinct t rows of post_n[t,*]. Different t rows touch disjoint
    // slots, so no atomics/sync are needed; the inner log_add across s stays
    // thread-private within each t row (multiple s may map to the same c, but a
    // single thread serializes those updates for its own row).
    for (int64_t t = tid; t < T_n; t += nthreads) {
        for (int64_t s = 0; s < L_n; ++s) {
            int32_t c = ext_label(s);
            T posterior = alpha[t * L_max + s] + beta[t * L_max + s];
            T& slot = post_n[t * C + c];
            slot = log_add(slot, posterior);
        }
    }
    __syncthreads();

    for (int64_t idx = tid; idx < T_n * C; idx += nthreads) {
        int64_t t = idx / C;
        int64_t c = idx % C;
        T lp = log_probs[t * N * C + n * C + c];
        T post = post_n[t * C + c];
        T prob = ::exp(lp);
        T post_prob = (post == NEG_INF) ? T(0) : ::exp(post - logZ);
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
    // Compute dtype follows the CPU/CUDA CTC reference (dtype-preservation):
    //   Float32: native single precision.
    //   Float64: native DOUBLE precision DP (no silent downcast) — the kernel is
    //            instantiated with T=double below, matching CPU+CUDA so
    //            dispatch<CTCLossForward> on Float64 is consistent across
    //            backends. The alpha/beta recursion runs in double.
    //   Float16/BFloat16/other: widen to Float32 (CTC is precision-sensitive and
    //            half-precision logZ saturates quickly), compute, narrow outputs.
    if (log_probs.dtype() != DType::Float32 && log_probs.dtype() != DType::Float64) {
        const DType orig = log_probs.dtype();
        auto outs = ctc_loss_forward_kernel(
            log_probs.to(DType::Float32), targets, input_lengths, target_lengths,
            blank, zero_infinity, stream);
        for (auto& t : outs) t = t.to(orig);
        return outs;
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

    // F118: normalize a 1-D concatenated (PyTorch-style) targets tensor into a
    // padded 2-D [N, S_pad] layout so the (n * S_max) indexing in host validation
    // and in the device kernel is correct. Without this the kernel read
    // targets + n*(total concatenated length) — out of bounds for n >= 1 (matches
    // CUDA's host normalization and the CPU backend's prefix-sum handling).
    Tensor targets2d = targets;
    if (targets.shape().size() == 1 && N > 0) {
        Tensor tl_cpu = target_lengths.to(Device::cpu()).contiguous();
        Tensor tg_cpu = targets.to(Device::cpu()).contiguous();
        const int32_t* tl = tl_cpu.data<int32_t>();
        const int32_t* tg = tg_cpu.data<int32_t>();
        const int64_t total = static_cast<int64_t>(tg_cpu.numel());
        int64_t S_pad = 1;
        for (int64_t n = 0; n < N; ++n) if (tl[n] > S_pad) S_pad = tl[n];
        Tensor padded({N, S_pad}, DType::Int32, Device::cpu());
        int32_t* pd = padded.data<int32_t>();
        for (int64_t i = 0; i < N * S_pad; ++i) pd[i] = 0;  // pad positions (never read past S_n)
        int64_t acc = 0;
        for (int64_t n = 0; n < N; ++n) {
            int64_t Sn = tl[n];
            for (int64_t s = 0; s < Sn && (acc + s) < total; ++s)
                pd[n * S_pad + s] = tg[acc + s];
            if (Sn > 0) acc += Sn;
        }
        if (acc > total) {
            throw std::invalid_argument(
                "ctc_loss_forward (ROCm): sum(target_lengths) exceeds the "
                "flattened targets length");
        }
        targets2d = padded.to(targets.device());
    }

    auto tgt_shape = targets2d.shape();
    int64_t S_max = (tgt_shape.size() >= 2) ? tgt_shape[1]
                  : (tgt_shape.size() == 1) ? tgt_shape[0] : 0;
    int64_t L_max = 2 * S_max + 1;

    // Native compute dtype: Float64 stays double, Float32 stays float.
    const DType compute_dtype = log_probs.dtype();
    const bool native_f64 = (compute_dtype == DType::Float64);
    const size_t compute_elem_size = native_f64 ? sizeof(double) : sizeof(float);

    Tensor loss_out({N}, compute_dtype, log_probs.device());
    Tensor grad_out({T_max, N, C}, compute_dtype, log_probs.device());

    int64_t alpha_elems = N * T_max * L_max;
    Tensor alpha_buf({alpha_elems}, compute_dtype, log_probs.device());
    Tensor beta_buf({alpha_elems}, compute_dtype, log_probs.device());
    // AA.10: per-block (T_max, C) scratch for log-space posterior
    // accumulation — must not alias grad_out.
    int64_t post_elems = N * T_max * C;
    Tensor post_scratch({post_elems}, compute_dtype, log_probs.device());

    if (N == 0 || T_max == 0 || C == 0) {
        if (loss_out.numel() > 0) {
            HIP_CHECK(hipMemsetAsync(loss_out.data_ptr(), 0,
                                     loss_out.numel() * compute_elem_size, stream));
        }
        if (grad_out.numel() > 0) {
            HIP_CHECK(hipMemsetAsync(grad_out.data_ptr(), 0,
                                     grad_out.numel() * compute_elem_size, stream));
        }
        return {loss_out, grad_out};
    }

    // Validate the blank label and every active target label up front, mirroring
    // the CPU reference (src/backends/cpu/kernels/ctc.cpp). ext_label(s) (== blank
    // or a target label) is used directly as a channel index into log_probs /
    // post_n; an out-of-range value would cause an OOB device read of log_probs
    // or an OOB write into post_n. Targets/lengths live on-device, so copy them
    // to host for validation (same std::invalid_argument exceptions as CPU).
    if (blank < 0 || blank >= C) {
        throw std::invalid_argument(
            "ctc_loss_forward (ROCm): blank index " + std::to_string(blank) +
            " out of range [0, " + std::to_string(C) + ")");
    }
    {
        std::vector<int32_t> il_host(static_cast<size_t>(N));
        std::vector<int32_t> tl_host(static_cast<size_t>(N));
        HIP_CHECK(hipMemcpy(il_host.data(), input_lengths.data<int32_t>(),
                            static_cast<size_t>(N) * sizeof(int32_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(tl_host.data(), target_lengths.data<int32_t>(),
                            static_cast<size_t>(N) * sizeof(int32_t), hipMemcpyDeviceToHost));

        std::vector<int32_t> tgt_host(static_cast<size_t>(N * S_max));
        if (N * S_max > 0) {
            HIP_CHECK(hipMemcpy(tgt_host.data(), targets2d.data<int32_t>(),
                                static_cast<size_t>(N * S_max) * sizeof(int32_t),
                                hipMemcpyDeviceToHost));
        }

        for (int64_t n = 0; n < N; ++n) {
            int64_t T_n = il_host[static_cast<size_t>(n)];
            int64_t S_n = tl_host[static_cast<size_t>(n)];
            if (T_n <= 0 || S_n <= 0 || T_n > T_max || S_n > S_max) {
                continue;  // inactive sample; skipped by the compute loops too
            }
            const int32_t* tgt_n = tgt_host.data() + n * S_max;
            for (int64_t s = 0; s < S_n; ++s) {
                const int64_t label = static_cast<int64_t>(tgt_n[s]);
                if (label < 0 || label >= C) {
                    throw std::invalid_argument(
                        "ctc_loss_forward (ROCm): target label " +
                        std::to_string(label) + " out of range [0, " +
                        std::to_string(C) + ")");
                }
            }
        }
    }

    dim3 grid(static_cast<unsigned>(N));
    dim3 block(CTC_THREADS_PER_BLOCK);

    if (native_f64) {
        hipLaunchKernelGGL(ctc_forward_backward_kernel<double>,
                           grid, block, 0, stream,
                           log_probs.data<double>(),
                           targets2d.data<int32_t>(),
                           input_lengths.data<int32_t>(),
                           target_lengths.data<int32_t>(),
                           alpha_buf.data<double>(),
                           beta_buf.data<double>(),
                           post_scratch.data<double>(),
                           loss_out.data<double>(),
                           grad_out.data<double>(),
                           T_max, N, C, S_max, L_max,
                           blank, zero_infinity);
    } else {
        hipLaunchKernelGGL(ctc_forward_backward_kernel<float>,
                           grid, block, 0, stream,
                           log_probs.data<float>(),
                           targets2d.data<int32_t>(),
                           input_lengths.data<int32_t>(),
                           target_lengths.data<int32_t>(),
                           alpha_buf.data<float>(),
                           beta_buf.data<float>(),
                           post_scratch.data<float>(),
                           loss_out.data<float>(),
                           grad_out.data<float>(),
                           T_max, N, C, S_max, L_max,
                           blank, zero_infinity);
    }
    HIP_CHECK(hipGetLastError());

    return {loss_out, grad_out};
}

} // namespace rocm
} // namespace tenzor
