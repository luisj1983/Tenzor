/**
 * @file ctc_loss.metal
 * @brief Metal shader for CTC (Connectionist Temporal Classification) loss.
 *
 * Mirrors src/backends/cuda/kernels/ctc.cu. One threadgroup per batch
 * element; within the group, thread 0 runs the sequential forward-backward
 * DP and all threads cooperate on the per-(t, c) gradient conversion.
 *
 * Buffer layout:
 *   0: log_probs   (T_max * N * C)        float
 *   1: targets     (N * S_max)            int32
 *   2: in_lengths  (N)                    int32
 *   3: tgt_lengths (N)                    int32
 *   4: alpha       (N * T_max * L_max)    float (workspace)
 *   5: beta        (N * T_max * L_max)    float (workspace)
 *   6: loss        (N)                    float
 *   7: grad        (T_max * N * C)        float
 *   8: params      uniform                uint x 8 (T_max, N, C, S_max, L_max,
 *                                                   blank, zero_infinity, _pad)
 *
 * NOTE: the host code paths for MPS CTC on Linux are unbuildable; this
 * shader is provided per the audit-4 W.3 spec for use when an Apple
 * build target is enabled.
 */

#include <metal_stdlib>
using namespace metal;

inline float ctc_log_add(float a, float b) {
    constexpr float NEG_INF = -INFINITY;
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    float m = fmax(a, b);
    return m + log1p(exp(-fabs(a - b)));
}

struct CTCParams {
    int32_t T_max;
    int32_t N;
    int32_t C;
    int32_t S_max;
    int32_t L_max;
    int32_t blank;
    int32_t zero_infinity;
    int32_t _pad;
};

kernel void ctc_forward_backward_kernel(
    device const float*    log_probs    [[buffer(0)]],
    device const int32_t*  targets      [[buffer(1)]],
    device const int32_t*  in_lengths   [[buffer(2)]],
    device const int32_t*  tgt_lengths  [[buffer(3)]],
    device float*          alpha_buf    [[buffer(4)]],
    device float*          beta_buf     [[buffer(5)]],
    device float*          loss_out     [[buffer(6)]],
    device float*          grad_out     [[buffer(7)]],
    constant CTCParams&    params       [[buffer(8)]],
    uint2 gid                            [[threadgroup_position_in_grid]],
    uint  tid                            [[thread_position_in_threadgroup]],
    uint  nthreads                       [[threads_per_threadgroup]]
) {
    const int n = int(gid.x);
    if (n >= params.N) return;

    const int T_max = params.T_max;
    const int N = params.N;
    const int C = params.C;
    const int S_max = params.S_max;
    const int L_max = params.L_max;
    const int blank = params.blank;
    const bool zero_infinity = (params.zero_infinity != 0);

    const int T_n = in_lengths[n];
    const int S_n = tgt_lengths[n];
    const int L_n = 2 * S_n + 1;

    constexpr float NEG_INF = -INFINITY;

    device float* alpha = alpha_buf + n * T_max * L_max;
    device float* beta  = beta_buf  + n * T_max * L_max;
    device const int32_t* tgt_n = targets + n * S_max;

    threadgroup float s_logZ;

    // Degenerate guard.
    if (T_n <= 0 || S_n <= 0 || L_n > L_max) {
        if (tid == 0) loss_out[n] = 0.0;
        for (int idx = int(tid); idx < T_max * C; idx += int(nthreads)) {
            int t = idx / C;
            int c = idx % C;
            grad_out[t * N * C + n * C + c] = 0.0;
        }
        return;
    }

    // Initialise alpha at t=0.
    for (int s = int(tid); s < L_n; s += int(nthreads)) {
        int c0 = (s % 2 == 0) ? blank : tgt_n[s / 2];
        if (s == 0) {
            alpha[0 * L_max + 0] = log_probs[0 * N * C + n * C + c0];
        } else if (s == 1 && L_n > 1) {
            alpha[0 * L_max + 1] = log_probs[0 * N * C + n * C + c0];
        } else {
            alpha[0 * L_max + s] = NEG_INF;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (int t = 1; t < T_n; ++t) {
        for (int s = int(tid); s < L_n; s += int(nthreads)) {
            float a = alpha[(t - 1) * L_max + s];
            if (s > 0) {
                a = ctc_log_add(a, alpha[(t - 1) * L_max + (s - 1)]);
            }
            int c_s = (s % 2 == 0) ? blank : tgt_n[s / 2];
            int c_sm2 = (s >= 2) ? ((s - 2) % 2 == 0 ? blank : tgt_n[(s - 2) / 2]) : -1;
            if (s > 1 && c_s != blank && c_s != c_sm2) {
                a = ctc_log_add(a, alpha[(t - 1) * L_max + (s - 2)]);
            }
            alpha[t * L_max + s] = a + log_probs[t * N * C + n * C + c_s];
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    if (tid == 0) {
        float term1 = alpha[(T_n - 1) * L_max + (L_n - 1)];
        float term2 = (L_n > 1) ? alpha[(T_n - 1) * L_max + (L_n - 2)] : NEG_INF;
        s_logZ = ctc_log_add(term1, term2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float logZ = s_logZ;

    // Init beta at t = T_n - 1.
    for (int s = int(tid); s < L_n; s += int(nthreads)) {
        if (s == L_n - 1 || (s == L_n - 2 && L_n > 1)) {
            beta[(T_n - 1) * L_max + s] = 0.0;
        } else {
            beta[(T_n - 1) * L_max + s] = NEG_INF;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (int t = T_n - 2; t >= 0; --t) {
        for (int s = int(tid); s < L_n; s += int(nthreads)) {
            int c_s = (s % 2 == 0) ? blank : tgt_n[s / 2];
            float b = beta[(t + 1) * L_max + s] + log_probs[(t + 1) * N * C + n * C + c_s];
            if (s < L_n - 1) {
                int c_s1 = ((s + 1) % 2 == 0) ? blank : tgt_n[(s + 1) / 2];
                float term = beta[(t + 1) * L_max + (s + 1)]
                           + log_probs[(t + 1) * N * C + n * C + c_s1];
                b = ctc_log_add(b, term);
            }
            int c_sp2 = (s + 2 < L_n) ? ((s + 2) % 2 == 0 ? blank : tgt_n[(s + 2) / 2]) : -1;
            if (s < L_n - 2 && c_s != blank && c_s != c_sp2) {
                float term = beta[(t + 1) * L_max + (s + 2)]
                           + log_probs[(t + 1) * N * C + n * C + c_sp2];
                b = ctc_log_add(b, term);
            }
            beta[t * L_max + s] = b;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    float per_sample_loss = -logZ;
    bool is_inf = !isfinite(per_sample_loss);
    if (zero_infinity && is_inf) {
        per_sample_loss = 0.0;
    }
    if (tid == 0) {
        loss_out[n] = per_sample_loss;
    }

    if (zero_infinity && is_inf) {
        for (int idx = int(tid); idx < T_max * C; idx += int(nthreads)) {
            int t = idx / C;
            int c = idx % C;
            grad_out[t * N * C + n * C + c] = 0.0;
        }
        return;
    }

    for (int idx = int(tid); idx < T_max * C; idx += int(nthreads)) {
        int t = idx / C;
        int c = idx % C;
        grad_out[t * N * C + n * C + c] = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (int t = 0; t < T_n; ++t) {
        for (int c = int(tid); c < C; c += int(nthreads)) {
            grad_out[t * N * C + n * C + c] = NEG_INF;
        }
        threadgroup_barrier(mem_flags::mem_device);

        if (tid == 0) {
            for (int s = 0; s < L_n; ++s) {
                int c = (s % 2 == 0) ? blank : tgt_n[s / 2];
                float posterior = alpha[t * L_max + s] + beta[t * L_max + s];
                float slot = grad_out[t * N * C + n * C + c];
                grad_out[t * N * C + n * C + c] = ctc_log_add(slot, posterior);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        for (int c = int(tid); c < C; c += int(nthreads)) {
            float lp = log_probs[t * N * C + n * C + c];
            float post = grad_out[t * N * C + n * C + c];
            float prob = exp(lp);
            float post_prob = (post == NEG_INF) ? 0.0 : exp(post - logZ);
            grad_out[t * N * C + n * C + c] = prob - post_prob;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}
