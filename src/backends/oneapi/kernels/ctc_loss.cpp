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

inline float ctc_log_add(float a, float b) {
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    float m = sycl::fmax(a, b);
    return m + sycl::log1p(sycl::exp(-sycl::fabs(a - b)));
}

} // anonymous namespace

auto ctc_loss_forward_kernel(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t blank,
    bool zero_infinity,
    sycl::queue& queue
) -> std::vector<Tensor> {
    if (log_probs.dtype() != DType::Float32) {
        throw std::invalid_argument(
            "ctc_loss_forward (OneAPI): log_probs must be Float32");
    }
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

    Tensor loss_out({N}, DType::Float32, log_probs.device());
    Tensor grad_out({T_max, N, C}, DType::Float32, log_probs.device());

    int64_t alpha_elems = N * T_max * L_max;
    Tensor alpha_buf({alpha_elems}, DType::Float32, log_probs.device());
    Tensor beta_buf({alpha_elems}, DType::Float32, log_probs.device());
    // AA.10: per-block (T_max, C) scratch for log-space posterior
    // accumulation — must not alias grad_out, or partial writes leak
    // across iterations and produce +inf grads.
    int64_t post_elems = N * T_max * C;
    Tensor post_scratch({post_elems}, DType::Float32, log_probs.device());

    if (N == 0 || T_max == 0 || C == 0) {
        if (loss_out.numel() > 0) {
            queue.fill<float>(static_cast<float*>(loss_out.data_ptr()), 0.0f,
                              static_cast<size_t>(loss_out.numel())).wait();
        }
        if (grad_out.numel() > 0) {
            queue.fill<float>(static_cast<float*>(grad_out.data_ptr()), 0.0f,
                              static_cast<size_t>(grad_out.numel())).wait();
        }
        return {loss_out, grad_out};
    }

    const float* lp_data = static_cast<const float*>(log_probs.data_ptr());
    const int32_t* tgt_data = static_cast<const int32_t*>(targets.data_ptr());
    const int32_t* il_data = static_cast<const int32_t*>(input_lengths.data_ptr());
    const int32_t* tl_data = static_cast<const int32_t*>(target_lengths.data_ptr());
    float* alpha_data = static_cast<float*>(alpha_buf.data_ptr());
    float* beta_data = static_cast<float*>(beta_buf.data_ptr());
    float* post_data = static_cast<float*>(post_scratch.data_ptr());
    float* loss_data = static_cast<float*>(loss_out.data_ptr());
    float* grad_data = static_cast<float*>(grad_out.data_ptr());

    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
    const int local_size = CTC_THREADS_PER_BLOCK;
    const int global_size = static_cast<int>(N) * local_size;

    sycl::nd_range<1> launch_range{sycl::range<1>(global_size), sycl::range<1>(local_size)};

    auto event = queue.submit([&](sycl::handler& h) {
        // Local accessor for s_logZ.
        sycl::local_accessor<float, 1> s_logZ_local(sycl::range<1>(1), h);

        const int64_t T_max_c = T_max;
        const int64_t N_c = N;
        const int64_t C_c = C;
        const int64_t S_max_c = S_max;
        const int64_t L_max_c = L_max;
        const int64_t blank_c = blank;
        const bool zero_infinity_c = zero_infinity;

        h.parallel_for(launch_range, [=](sycl::nd_item<1> item) {
            const int n = static_cast<int>(item.get_group(0));
            if (n >= N_c) return;
            const int tid = static_cast<int>(item.get_local_id(0));
            const int nthreads = static_cast<int>(item.get_local_range(0));

            const int32_t T_n = il_data[n];
            const int32_t S_n = tl_data[n];
            const int64_t L_n = 2 * S_n + 1;

            float* alpha = alpha_data + n * T_max_c * L_max_c;
            float* beta  = beta_data  + n * T_max_c * L_max_c;
            const int32_t* tgt_n = tgt_data + n * S_max_c;

            auto ext_label = [&](int64_t s) -> int32_t {
                return (s % 2 == 0) ? static_cast<int32_t>(blank_c) : tgt_n[s / 2];
            };

            if (T_n <= 0 || S_n <= 0 || L_n > L_max_c) {
                if (tid == 0) loss_data[n] = 0.0f;
                for (int64_t idx = tid; idx < T_max_c * C_c; idx += nthreads) {
                    int64_t t = idx / C_c;
                    int64_t c = idx % C_c;
                    grad_data[t * N_c * C_c + n * C_c + c] = 0.0f;
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
                    float a = alpha[(t - 1) * L_max_c + s];
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
                float term1 = alpha[(T_n - 1) * L_max_c + (L_n - 1)];
                float term2 = (L_n > 1) ? alpha[(T_n - 1) * L_max_c + (L_n - 2)] : NEG_INF;
                s_logZ_local[0] = ctc_log_add(term1, term2);
            }
            item.barrier(sycl::access::fence_space::local_space);
            float logZ = s_logZ_local[0];

            // Backward DP initialisation at t = T_n - 1.
            for (int64_t s = tid; s < L_n; s += nthreads) {
                if (s == L_n - 1 || (s == L_n - 2 && L_n > 1)) {
                    beta[(T_n - 1) * L_max_c + s] = 0.0f;
                } else {
                    beta[(T_n - 1) * L_max_c + s] = NEG_INF;
                }
            }
            item.barrier(sycl::access::fence_space::global_space);

            for (int64_t t = T_n - 2; t >= 0; --t) {
                for (int64_t s = tid; s < L_n; s += nthreads) {
                    int32_t c_s = ext_label(s);
                    float b = beta[(t + 1) * L_max_c + s] + lp_data[(t + 1) * N_c * C_c + n * C_c + c_s];
                    if (s < L_n - 1) {
                        int32_t c_s1 = ext_label(s + 1);
                        float term = beta[(t + 1) * L_max_c + (s + 1)]
                                   + lp_data[(t + 1) * N_c * C_c + n * C_c + c_s1];
                        b = ctc_log_add(b, term);
                    }
                    if (s < L_n - 2 && c_s != blank_c && c_s != ext_label(s + 2)) {
                        int32_t c_s2 = ext_label(s + 2);
                        float term = beta[(t + 1) * L_max_c + (s + 2)]
                                   + lp_data[(t + 1) * N_c * C_c + n * C_c + c_s2];
                        b = ctc_log_add(b, term);
                    }
                    beta[t * L_max_c + s] = b;
                }
                item.barrier(sycl::access::fence_space::global_space);
            }

            float per_sample_loss = -logZ;
            bool is_inf = !sycl::isfinite(per_sample_loss);
            if (zero_infinity_c && is_inf) {
                per_sample_loss = 0.0f;
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
            float* post_n = post_data + n * T_max_c * C_c;

            for (int64_t idx = tid; idx < T_max_c * C_c; idx += nthreads) {
                int64_t t = idx / C_c;
                int64_t c = idx % C_c;
                grad_data[t * N_c * C_c + n * C_c + c] = 0.0f;
                post_n[t * C_c + c] = NEG_INF;
            }
            item.barrier(sycl::access::fence_space::global_space);

            if (tid == 0) {
                for (int64_t t = 0; t < T_n; ++t) {
                    for (int64_t s = 0; s < L_n; ++s) {
                        int32_t c = ext_label(s);
                        float posterior = alpha[t * L_max_c + s] + beta[t * L_max_c + s];
                        float& slot = post_n[t * C_c + c];
                        slot = ctc_log_add(slot, posterior);
                    }
                }
            }
            item.barrier(sycl::access::fence_space::global_space);

            for (int64_t idx = tid; idx < T_n * C_c; idx += nthreads) {
                int64_t t = idx / C_c;
                int64_t c = idx % C_c;
                float lp = lp_data[t * N_c * C_c + n * C_c + c];
                float post = post_n[t * C_c + c];
                float prob = sycl::exp(lp);
                float post_prob = (post == NEG_INF) ? 0.0f : sycl::exp(post - logZ);
                grad_data[t * N_c * C_c + n * C_c + c] = prob - post_prob;
            }
        });
    });
    event.wait_and_throw();

    return {loss_out, grad_out};
}

} // namespace oneapi
} // namespace tenzor
