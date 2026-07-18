/**
 * @file ctc_loss.cpp
 * @brief OneAPI/SYCL port of the CTC forward-backward DP kernel.
 *
 * Mirrors src/backends/cuda/kernels/ctc.cu / rocm/kernels/ctc_loss.hip.cpp:
 * one workgroup per batch element, log-domain alpha/beta over extended
 * labels, gradient computed in-place.
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

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {
namespace oneapi {

namespace {

constexpr int CTC_THREADS_PER_BLOCK = 128;

template <typename S>
inline S ctc_log_add(S a, S b) {
    constexpr S NEG_INF = -std::numeric_limits<S>::infinity();
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    S m = sycl::fmax(a, b);
    return m + sycl::log1p(sycl::exp(-sycl::fabs(a - b)));
}

// Scalar-generic CTC forward-backward implementation. Float32 instantiates as
// float, Float64 natively as double (so precision is preserved rather than
// narrowed through Float32). Half formats are widened to Float32 by the public
// dispatcher below before reaching here.
template <typename Scalar>
std::vector<Tensor> ctc_loss_forward_impl(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t blank,
    bool zero_infinity,
    sycl::queue& queue
) {
    constexpr DType SCALAR_DTYPE =
        std::is_same_v<Scalar, double> ? DType::Float64 : DType::Float32;
    if (targets.dtype() != DType::Int32 ||
        input_lengths.dtype() != DType::Int32 ||
        target_lengths.dtype() != DType::Int32) {
        throw std::invalid_argument(
            "ctc_loss_forward (OneAPI): targets / input_lengths / target_lengths "
            "must be Int32");
    }
    auto lp_shape = log_probs.shape();
    if (lp_shape.size() != 3) {
        throw std::invalid_argument(
            "ctc_loss_forward (OneAPI): log_probs must be 3D (T, N, C)");
    }
    int64_t T_max = lp_shape[0];
    int64_t N = lp_shape[1];
    int64_t C = lp_shape[2];

    auto tgt_shape = targets.shape();
    int64_t S_max = (tgt_shape.size() >= 2) ? tgt_shape[1]
                  : (tgt_shape.size() == 1) ? tgt_shape[0] : 0;
    int64_t L_max = 2 * S_max + 1;

    Tensor loss_out({N}, SCALAR_DTYPE, log_probs.device());
    Tensor grad_out({T_max, N, C}, SCALAR_DTYPE, log_probs.device());

    int64_t alpha_elems = N * T_max * L_max;
    Tensor alpha_buf({alpha_elems}, SCALAR_DTYPE, log_probs.device());
    Tensor beta_buf({alpha_elems}, SCALAR_DTYPE, log_probs.device());
    // AA.10: per-block (T_max, C) scratch for log-space posterior
    // accumulation — must not alias grad_out, or partial writes leak
    // across iterations and produce +inf grads.
    int64_t post_elems = N * T_max * C;
    Tensor post_scratch({post_elems}, SCALAR_DTYPE, log_probs.device());

    if (N == 0 || T_max == 0 || C == 0) {
        if (loss_out.numel() > 0) {
            queue.fill<Scalar>(static_cast<Scalar*>(loss_out.data_ptr()), Scalar(0),
                              static_cast<size_t>(loss_out.numel())).wait();
        }
        if (grad_out.numel() > 0) {
            queue.fill<Scalar>(static_cast<Scalar*>(grad_out.data_ptr()), Scalar(0),
                              static_cast<size_t>(grad_out.numel())).wait();
        }
        return {loss_out, grad_out};
    }

    // The kernel reads log_probs/targets/lengths via shape-derived flat offsets,
    // so a non-contiguous view (e.g. a transposed (N,T,C)->(T,N,C) log_probs, a
    // common way to produce CTC inputs) would misread. Materialize contiguous,
    // matching the CPU reference which calls .contiguous() on every input.
    Tensor log_probs_c = log_probs.is_contiguous() ? log_probs : log_probs.contiguous();
    Tensor targets_c = targets.is_contiguous() ? targets : targets.contiguous();
    Tensor input_lengths_c = input_lengths.is_contiguous() ? input_lengths : input_lengths.contiguous();
    Tensor target_lengths_c = target_lengths.is_contiguous() ? target_lengths : target_lengths.contiguous();

    const Scalar* lp_data = static_cast<const Scalar*>(log_probs_c.data_ptr());
    const int32_t* tgt_data = static_cast<const int32_t*>(targets_c.data_ptr());
    const int32_t* il_data = static_cast<const int32_t*>(input_lengths_c.data_ptr());
    const int32_t* tl_data = static_cast<const int32_t*>(target_lengths_c.data_ptr());
    Scalar* alpha_data = static_cast<Scalar*>(alpha_buf.data_ptr());
    Scalar* beta_data = static_cast<Scalar*>(beta_buf.data_ptr());
    Scalar* post_data = static_cast<Scalar*>(post_scratch.data_ptr());
    Scalar* loss_data = static_cast<Scalar*>(loss_out.data_ptr());
    Scalar* grad_data = static_cast<Scalar*>(grad_out.data_ptr());

    // Validate blank and target labels host-side before they are used (unguarded)
    // as channel indices into lp_data/grad_data on the device. An out-of-range
    // blank or target label would otherwise index out of bounds (heap corruption).
    // Mirrors the CPU reference (cpu/kernels/ctc.cpp).
    if (blank < 0 || blank >= C) {
        throw std::out_of_range("ctc_loss: blank index " + std::to_string(blank) +
            " out of range [0, " + std::to_string(C) + ")");
    }
    {
        // Pull targets + target_lengths to host (USM-shared but use an explicit
        // copy so this is correct regardless of allocation kind) and check that
        // every active label is in [0, C).
        const int64_t tgt_numel = targets_c.numel();
        std::vector<int32_t> host_tgt(static_cast<size_t>(tgt_numel));
        std::vector<int32_t> host_tl(static_cast<size_t>(N));
        if (tgt_numel > 0) {
            queue.memcpy(host_tgt.data(), tgt_data,
                         static_cast<size_t>(tgt_numel) * sizeof(int32_t)).wait();
        }
        queue.memcpy(host_tl.data(), tl_data,
                     static_cast<size_t>(N) * sizeof(int32_t)).wait();
        // Targets may be either 2D padded (N, S_max) — sample n at row offset
        // n*S_max — or 1D concatenated (sum of target_lengths) — sample n at the
        // running prefix-sum offset. Derive the per-sample base identically to the
        // device kernel so validation and computation address the same elements.
        const bool targets_2d = (targets_c.shape().size() == 2);
        int64_t concat_off = 0;  // running offset for 1D concatenated targets
        for (int64_t n = 0; n < N; ++n) {
            int64_t len = host_tl[n];
            int64_t base = targets_2d ? (n * S_max) : concat_off;
            for (int64_t s = 0; s < len; ++s) {
                int64_t off = base + s;
                if (off < 0 || off >= tgt_numel) {
                    // Matches CPU/CUDA/ROCm/Vulkan: silently `continue`-ing past
                    // an out-of-range offset let a malformed target_lengths sum
                    // slip through undetected instead of raising a clean,
                    // catchable error (and risked an OOB read below if some
                    // caller ever reached this branch with off still in-range
                    // for host_tgt but not for the logical target layout).
                    throw std::invalid_argument(
                        "ctc_loss_forward (OneAPI): sum(target_lengths) exceeds the "
                        "flattened targets length");
                }
                int32_t lbl = host_tgt[off];
                if (lbl < 0 || lbl >= C) {
                    throw std::out_of_range("ctc_loss: target label " + std::to_string(lbl) +
                        " out of range [0, " + std::to_string(C) + ")");
                }
            }
            concat_off += len;
        }
    }

    constexpr Scalar NEG_INF = -std::numeric_limits<Scalar>::infinity();
    const int local_size = CTC_THREADS_PER_BLOCK;
    const int global_size = static_cast<int>(N) * local_size;

    sycl::nd_range<1> launch_range{sycl::range<1>(global_size), sycl::range<1>(local_size)};

    auto event = queue.submit([&](sycl::handler& h) {
        // Local accessor for s_logZ.
        sycl::local_accessor<Scalar, 1> s_logZ_local(sycl::range<1>(1), h);

        const int64_t T_max_c = T_max;
        const int64_t N_c = N;
        const int64_t C_c = C;
        const int64_t S_max_c = S_max;
        const int64_t L_max_c = L_max;
        const int64_t blank_c = blank;
        const bool zero_infinity_c = zero_infinity;
        const bool targets_2d_c = (targets_c.shape().size() == 2);

        h.parallel_for(launch_range, [=](sycl::nd_item<1> item) {
            const int n = static_cast<int>(item.get_group(0));
            if (n >= N_c) return;
            const int tid = static_cast<int>(item.get_local_id(0));
            const int nthreads = static_cast<int>(item.get_local_range(0));

            const int32_t T_n = il_data[n];
            const int32_t S_n = tl_data[n];
            const int64_t L_n = 2 * S_n + 1;

            Scalar* alpha = alpha_data + n * T_max_c * L_max_c;
            Scalar* beta  = beta_data  + n * T_max_c * L_max_c;
            // Per-sample target base: 2D padded uses row offset n*S_max; 1D
            // concatenated uses the prefix sum of target_lengths (matches the
            // host validator). Deriving both here keeps 1D targets with N>1 in
            // bounds instead of always assuming the 2D-padded layout.
            int64_t tgt_off = n * S_max_c;
            if (!targets_2d_c) {
                tgt_off = 0;
                for (int k = 0; k < n; ++k) tgt_off += tl_data[k];
            }
            const int32_t* tgt_n = tgt_data + tgt_off;

            auto ext_label = [&](int64_t s) -> int32_t {
                return (s % 2 == 0) ? static_cast<int32_t>(blank_c) : tgt_n[s / 2];
            };

            if (T_n <= 0 || S_n <= 0 || T_n > T_max_c || L_n > L_max_c) {
                if (tid == 0) loss_data[n] = Scalar(0);
                for (int64_t idx = tid; idx < T_max_c * C_c; idx += nthreads) {
                    int64_t t = idx / C_c;
                    int64_t c = idx % C_c;
                    grad_data[t * N_c * C_c + n * C_c + c] = Scalar(0);
                }
                return;
            }

            // Forward DP initialisation at t=0.
            for (int64_t s = tid; s < L_n; s += nthreads) {
                if (s == 0) {
                    alpha[0 * L_max_c + 0] = lp_data[0 * N_c * C_c + n * C_c + ext_label(0)];
                } else if (s == 1 && L_n > 1) {
                    alpha[0 * L_max_c + 1] = lp_data[0 * N_c * C_c + n * C_c + ext_label(1)];
                } else {
                    alpha[0 * L_max_c + s] = NEG_INF;
                }
            }
            item.barrier(sycl::access::fence_space::global_space);

            for (int64_t t = 1; t < T_n; ++t) {
                for (int64_t s = tid; s < L_n; s += nthreads) {
                    Scalar a = alpha[(t - 1) * L_max_c + s];
                    if (s > 0) {
                        a = ctc_log_add(a, alpha[(t - 1) * L_max_c + (s - 1)]);
                    }
                    int32_t c_s = ext_label(s);
                    if (s > 1 && c_s != blank_c && c_s != ext_label(s - 2)) {
                        a = ctc_log_add(a, alpha[(t - 1) * L_max_c + (s - 2)]);
                    }
                    alpha[t * L_max_c + s] = a + lp_data[t * N_c * C_c + n * C_c + c_s];
                }
                item.barrier(sycl::access::fence_space::global_space);
            }

            if (tid == 0) {
                Scalar term1 = alpha[(T_n - 1) * L_max_c + (L_n - 1)];
                Scalar term2 = (L_n > 1) ? alpha[(T_n - 1) * L_max_c + (L_n - 2)] : NEG_INF;
                s_logZ_local[0] = ctc_log_add(term1, term2);
            }
            item.barrier(sycl::access::fence_space::local_space);
            Scalar logZ = s_logZ_local[0];

            // Backward DP initialisation at t = T_n - 1.
            for (int64_t s = tid; s < L_n; s += nthreads) {
                if (s == L_n - 1 || (s == L_n - 2 && L_n > 1)) {
                    beta[(T_n - 1) * L_max_c + s] = Scalar(0);
                } else {
                    beta[(T_n - 1) * L_max_c + s] = NEG_INF;
                }
            }
            item.barrier(sycl::access::fence_space::global_space);

            for (int64_t t = T_n - 2; t >= 0; --t) {
                for (int64_t s = tid; s < L_n; s += nthreads) {
                    int32_t c_s = ext_label(s);
                    Scalar b = beta[(t + 1) * L_max_c + s] + lp_data[(t + 1) * N_c * C_c + n * C_c + c_s];
                    if (s < L_n - 1) {
                        int32_t c_s1 = ext_label(s + 1);
                        Scalar term = beta[(t + 1) * L_max_c + (s + 1)]
                                   + lp_data[(t + 1) * N_c * C_c + n * C_c + c_s1];
                        b = ctc_log_add(b, term);
                    }
                    if (s < L_n - 2 && c_s != blank_c && c_s != ext_label(s + 2)) {
                        int32_t c_s2 = ext_label(s + 2);
                        Scalar term = beta[(t + 1) * L_max_c + (s + 2)]
                                   + lp_data[(t + 1) * N_c * C_c + n * C_c + c_s2];
                        b = ctc_log_add(b, term);
                    }
                    beta[t * L_max_c + s] = b;
                }
                item.barrier(sycl::access::fence_space::global_space);
            }

            Scalar per_sample_loss = -logZ;
            bool is_inf = sycl::isinf(per_sample_loss);
            if (zero_infinity_c && is_inf) {
                per_sample_loss = Scalar(0);
            }
            if (tid == 0) {
                loss_data[n] = per_sample_loss;
            }

            if (zero_infinity_c && is_inf) {
                for (int64_t idx = tid; idx < T_max_c * C_c; idx += nthreads) {
                    int64_t t = idx / C_c;
                    int64_t c = idx % C_c;
                    grad_data[t * N_c * C_c + n * C_c + c] = 0.0f;
                }
                return;
            }

            // AA.10 fix: accumulate log-posteriors into a dedicated
            // per-block scratch tile (post_data[n, t, c]) for all t first;
            // only after every t has been processed do we read log_probs
            // and convert to the final gradient. The previous in-place
            // version aliased the global grad cell as both log-space
            // accumulator and final exp-space gradient — partial writes
            // leaked across iterations, producing +inf grads.
            Scalar* post_n = post_data + n * T_max_c * C_c;

            for (int64_t idx = tid; idx < T_max_c * C_c; idx += nthreads) {
                int64_t t = idx / C_c;
                int64_t c = idx % C_c;
                grad_data[t * N_c * C_c + n * C_c + c] = 0.0f;
                post_n[t * C_c + c] = NEG_INF;
            }
            item.barrier(sycl::access::fence_space::global_space);

            // Parallelize over t across threads: each thread owns a disjoint set
            // of t rows, so for a fixed t the ctc_log_add into post_n[t,c] is
            // single-owner even when multiple s map to the same extended label c
            // (no cross-thread race). This replaces the previous tid==0 serial
            // loop that idled the rest of the work-group.
            for (int64_t t = tid; t < T_n; t += nthreads) {
                for (int64_t s = 0; s < L_n; ++s) {
                    int32_t c = ext_label(s);
                    Scalar posterior = alpha[t * L_max_c + s] + beta[t * L_max_c + s];
                    Scalar& slot = post_n[t * C_c + c];
                    slot = ctc_log_add(slot, posterior);
                }
            }
            item.barrier(sycl::access::fence_space::global_space);

            for (int64_t idx = tid; idx < T_n * C_c; idx += nthreads) {
                int64_t t = idx / C_c;
                int64_t c = idx % C_c;
                Scalar lp = lp_data[t * N_c * C_c + n * C_c + c];
                Scalar post = post_n[t * C_c + c];
                Scalar prob = sycl::exp(lp);
                Scalar post_prob = (post == NEG_INF) ? Scalar(0) : sycl::exp(post - logZ);
                grad_data[t * N_c * C_c + n * C_c + c] = prob - post_prob;
            }
        });
    });
    event.wait_and_throw();

    return {loss_out, grad_out};
}

} // anonymous namespace

// Public entry point: dispatch on dtype. Float32/Float64 run natively (double
// preserves precision instead of narrowing through Float32); Float16/BFloat16
// widen to Float32, run, then narrow both outputs (loss + raw grad) back so the
// op preserves precision metadata (matches the CPU backend).
auto ctc_loss_forward_kernel(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t blank,
    bool zero_infinity,
    sycl::queue& queue
) -> std::vector<Tensor> {
    const DType dt = log_probs.dtype();
    if (dt == DType::Float32) {
        return ctc_loss_forward_impl<float>(log_probs, targets, input_lengths,
                                            target_lengths, blank, zero_infinity, queue);
    }
    if (dt == DType::Float64) {
        return ctc_loss_forward_impl<double>(log_probs, targets, input_lengths,
                                             target_lengths, blank, zero_infinity, queue);
    }
    if (dt == DType::Float16 || dt == DType::BFloat16) {
        auto out = ctc_loss_forward_impl<float>(log_probs.to(DType::Float32), targets,
                                                input_lengths, target_lengths, blank,
                                                zero_infinity, queue);
        for (auto& t : out) t = t.to(dt);
        return out;
    }
    throw std::invalid_argument(
        "ctc_loss_forward (OneAPI): log_probs must be Float16/BFloat16/Float32/Float64");
}

} // namespace oneapi
} // namespace tenzor
