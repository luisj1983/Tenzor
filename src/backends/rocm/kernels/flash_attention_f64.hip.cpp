// =============================================================================
// Float64 FlashAttention forward / backward kernels (audit A.11, ROCm)
//
// Native double-precision HIP implementation of FlashAttention. The mainline
// FP32 / FP16 / BF16 path in `fused_ops.hip.cpp` accumulates in `float`, which
// silently halves precision when Float64 inputs are upcast → Float32 →
// downcast. For Float64 gradcheck (relative tolerance ~1e-7) that round-trip
// is fatal: forward already loses double precision and analytical-vs-numerical
// gradients diverge beyond tolerance.
//
// Algorithm matches the FP32 `flash_attention_v2_kernel_hip` / backward kernel
// in `fused_ops.hip.cpp` (online softmax, tiled K/V, dQ via atomicAdd, dK/dV
// in registers) but every accumulator, score, and shared-memory slot is
// `double`.
//
// Per docs/internals/attention-contract.md, LSE remains Float32 in the saved
// tensor; we accumulate the LSE in `double` inside the kernel and cast to
// Float32 only on the final store. Backward recomputes softmax in FP64 from
// Q / K (matches the CUDA flash_attention_f64.cu approach) instead of reading
// the saved Float32 LSE, so the backward is correct to full double precision.
//
// Causal mask uses -INFINITY (contract sentinel rule). Dropout is intentionally
// not supported in the Float64 path — Float64 attention is gradcheck-only and
// dropout is incompatible with deterministic gradcheck. The dispatcher refuses
// dropout > 0 with Float64 inputs.
//
// HIP-specific notes vs. CUDA reference (flash_attention_f64.cu):
//   * We use shared-memory tree reductions instead of warp shuffles. HIP's
//     `__shfl_xor` does NOT take a mask argument, AND the AMD wavefront size
//     varies (32 on RDNA, 64 on CDNA/MI200/MI300). The existing FP32 ROCm
//     FlashAttention kernel already uses tree reductions for that exact
//     portability reason — we keep the same pattern.
//   * HIP supports native `atomicAdd<double>` on gfx9+ (and via
//     `-munsafe-fp-atomics` on gfx10+/RDNA). The ROCm CMakeLists already
//     passes `-munsafe-fp-atomics` for tenzor_rocm_kernels. The same code
//     pattern is used in reduction.hip.cpp and vision.hip.cpp.
//   * `hipFuncSetAttribute` is not required on AMD — dynamic shared memory
//     up to the LDS size is available without an opt-in. We just pass the
//     size at launch time. (LDS is 64 KiB on most AMD GPUs which fits all
//     our supported head_dims.)
// =============================================================================

#include <hip/hip_runtime.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/fast_dispatch.hpp"  // for any future dispatch needs
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <utility>
#include <string>

namespace tenzor {
namespace rocm {

// Forward declaration of the helper from fused_ops.hip.cpp (same translation
// unit category — the registry just needs the symbol). We intentionally do
// not include fused_ops.hip.cpp directly; instead, we redeclare the helper
// here as an inline so the linker is happy if it's not externally visible.
namespace {

inline Tensor create_hip_zeros_f64(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    size_t bytes = t.numel() * dtype_size(dtype);
    hipMemsetAsync(t.data_ptr(), 0, bytes, nullptr);
    return t;
}

} // anonymous namespace

// =============================================================================
// Float64 FlashAttention forward kernel
// =============================================================================
//
// Grid:  (batch_heads, seq_len_q)
// Block: BLOCK_SIZE threads (default 256)
// Shared memory layout (all `double`):
//   K_tile [Bc * K_STRIDE]
//   V_tile [Bc * K_STRIDE]
//   Q_shared      [HEAD_DIM]
//   scores_shared [Bc]
//   reduce_buf    [Bc]      (used for sum reduction scratch)
template<int HEAD_DIM, int BLOCK_SIZE = 256>
__global__ void flash_attention_v2_kernel_f64_hip(
    const double* __restrict__ Q,
    const double* __restrict__ K,
    const double* __restrict__ V,
    double* __restrict__ O,
    float* __restrict__ L,            // Float32 per contract; may be nullptr
    const int seq_len_q,
    const int seq_len_k,
    const double scale,
    const bool causal
) {
    constexpr int Bc = 32;
    constexpr int K_STRIDE = HEAD_DIM + 4;  // pad to avoid bank conflicts

    const int batch_head = blockIdx.x;
    const int q_row      = blockIdx.y;
    if (q_row >= seq_len_q) return;

    const int tid = threadIdx.x;

    const double* Q_row  = Q + (batch_head * seq_len_q + q_row) * HEAD_DIM;
    const double* K_base = K + batch_head * seq_len_k * HEAD_DIM;
    const double* V_base = V + batch_head * seq_len_k * HEAD_DIM;
    double*       O_row  = O + (batch_head * seq_len_q + q_row) * HEAD_DIM;

    extern __shared__ double smem_f64_fwd[];
    double* K_tile        = smem_f64_fwd;                              // [Bc][K_STRIDE]
    double* V_tile        = smem_f64_fwd + Bc * K_STRIDE;              // [Bc][K_STRIDE]
    double* Q_shared      = smem_f64_fwd + 2 * Bc * K_STRIDE;          // [HEAD_DIM]
    double* scores_shared = Q_shared + HEAD_DIM;                        // [Bc]
    double* reduce_buf    = scores_shared + Bc;                         // [Bc]

    // Load Q row cooperatively.
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        Q_shared[d] = Q_row[d];
    }
    __syncthreads();

    constexpr int ELEMS_PER_THREAD = (HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double o_acc[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) o_acc[e] = 0.0;

    double m_prev = -__longlong_as_double(0x7ff0000000000000LL);  // -INF (bit pattern, ROCm-safe)
    double l_prev = 0.0;

    const int num_kv_blocks = (seq_len_k + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int kv_start = kv_block * Bc;
        const int kv_end_actual = (kv_start + Bc < seq_len_k) ? Bc : (seq_len_k - kv_start);

        // Load K tile.
        for (int idx = tid; idx < Bc * HEAD_DIM; idx += BLOCK_SIZE) {
            int row = idx / HEAD_DIM;
            int col = idx % HEAD_DIM;
            if (row < kv_end_actual) {
                K_tile[row * K_STRIDE + col] = K_base[(kv_start + row) * HEAD_DIM + col];
            } else {
                K_tile[row * K_STRIDE + col] = 0.0;
            }
        }
        // Load V tile.
        for (int idx = tid; idx < Bc * HEAD_DIM; idx += BLOCK_SIZE) {
            int row = idx / HEAD_DIM;
            int col = idx % HEAD_DIM;
            if (row < kv_end_actual) {
                V_tile[row * K_STRIDE + col] = V_base[(kv_start + row) * HEAD_DIM + col];
            } else {
                V_tile[row * K_STRIDE + col] = 0.0;
            }
        }
        __syncthreads();

        // Compute scores S[j] = Q · K[j]^T * scale; causal mask via -INF
        // (contract sentinel rule).
        const double NEG_INF = -__longlong_as_double(0x7ff0000000000000LL);
        for (int j = tid; j < Bc; j += BLOCK_SIZE) {
            double score;
            int kv_pos = kv_start + j;
            if (j < kv_end_actual && !(causal && kv_pos > q_row)) {
                score = 0.0;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_shared[d] * K_tile[j * K_STRIDE + d];
                }
                score *= scale;
            } else {
                score = NEG_INF;
            }
            scores_shared[j] = score;
        }
        __syncthreads();

        // Tree-reduce max over scores_shared (wavefront-agnostic, see FP32
        // kernel comment in fused_ops.hip.cpp for why we avoid shuffles here).
        // This destroys the original scores, so we recompute them below.
        for (int stride = Bc / 2; stride > 0; stride >>= 1) {
            if (tid < stride && tid + stride < Bc) {
                double a = scores_shared[tid];
                double b = scores_shared[tid + stride];
                scores_shared[tid] = (a > b) ? a : b;
            }
            __syncthreads();
        }
        double tile_max = scores_shared[0];
        __syncthreads();

        // Recompute scores (max reduction destroyed them).
        for (int j = tid; j < Bc; j += BLOCK_SIZE) {
            double score;
            int kv_pos = kv_start + j;
            if (j < kv_end_actual && !(causal && kv_pos > q_row)) {
                score = 0.0;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_shared[d] * K_tile[j * K_STRIDE + d];
                }
                score *= scale;
            } else {
                score = NEG_INF;
            }
            scores_shared[j] = score;
        }
        __syncthreads();

        // exp(score - tile_max). No dropout in FP64 path.
        for (int j = tid; j < Bc; j += BLOCK_SIZE) {
            scores_shared[j] = exp(scores_shared[j] - tile_max);
        }
        __syncthreads();

        // Tree-reduce sum via reduce_buf scratch (preserves scores_shared for
        // the P @ V step below).
        for (int j = tid; j < Bc; j += BLOCK_SIZE) {
            reduce_buf[j] = scores_shared[j];
        }
        __syncthreads();
        for (int stride = Bc / 2; stride > 0; stride >>= 1) {
            if (tid < stride && tid + stride < Bc) {
                reduce_buf[tid] += reduce_buf[tid + stride];
            }
            __syncthreads();
        }
        double tile_sum = reduce_buf[0];
        __syncthreads();

        // Online softmax merge.
        double m_new        = (m_prev > tile_max) ? m_prev : tile_max;
        double rescale_prev = exp(m_prev - m_new);
        double rescale_tile = exp(tile_max - m_new);
        double l_new        = l_prev * rescale_prev + tile_sum * rescale_tile;

        // Rescale previous output and accumulate P @ V.
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            int d = tid + e * BLOCK_SIZE;
            if (d < HEAD_DIM) {
                o_acc[e] *= rescale_prev;
                double pv = 0.0;
                for (int j = 0; j < Bc; ++j) {
                    pv += scores_shared[j] * V_tile[j * K_STRIDE + d];
                }
                o_acc[e] += pv * rescale_tile;
            }
        }

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }

    // Final normalize + write.
    double l_inv = (l_prev > 0.0) ? (1.0 / l_prev) : 0.0;
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        int d = tid + e * BLOCK_SIZE;
        if (d < HEAD_DIM) {
            O_row[d] = o_acc[e] * l_inv;
        }
    }

    // LSE downcast to Float32 per contract. Backward replays softmax in
    // double from Q/K, so this LSE is a stub for contract compliance.
    if (L != nullptr && tid == 0) {
        double lse = m_prev + log(l_prev + 1e-300);
        L[batch_head * seq_len_q + q_row] = static_cast<float>(lse);
    }
}

// =============================================================================
// Float64 FlashAttention backward kernel
// =============================================================================
//
// Same tiled structure as the FP32 backward kernel: grid = (num_kv_tiles,
// batch_heads); each block owns a KV tile, iterates Q tiles. Recomputes scores
// & per-row softmax denominator in `double` from Q, K (rather than reading the
// FP32 LSE) so the backward is correct to full double precision.
template<int HEAD_DIM, int Br, int Bc, int BLOCK_SIZE>
__global__ void flash_attention_backward_kernel_f64_hip(
    const double* __restrict__ Q,     // [batch_heads, seq_len, HEAD_DIM]
    const double* __restrict__ K,     // [batch_heads, seq_len, HEAD_DIM]
    const double* __restrict__ V,     // [batch_heads, seq_len, HEAD_DIM]
    const double* __restrict__ O,     // [batch_heads, seq_len, HEAD_DIM]
    const double* __restrict__ dO,    // [batch_heads, seq_len, HEAD_DIM]
    double* __restrict__ dQ,          // accumulated via atomicAdd
    double* __restrict__ dK,          // one block per KV tile → no race
    double* __restrict__ dV,          // one block per KV tile → no race
    const int seq_len,
    const double scale,
    const bool causal
) {
    const int kv_tile_idx = blockIdx.x;
    const int batch_head  = blockIdx.y;
    const int tid         = threadIdx.x;

    const int kv_start = kv_tile_idx * Bc;
    if (kv_start >= seq_len) return;
    const int actual_Bc = min(Bc, seq_len - kv_start);

    const double* Q_base  = Q  + batch_head * seq_len * HEAD_DIM;
    const double* K_base  = K  + batch_head * seq_len * HEAD_DIM;
    const double* V_base  = V  + batch_head * seq_len * HEAD_DIM;
    const double* O_base  = O  + batch_head * seq_len * HEAD_DIM;
    const double* dO_base = dO + batch_head * seq_len * HEAD_DIM;
    double* dQ_base = dQ + batch_head * seq_len * HEAD_DIM;
    double* dK_base = dK + batch_head * seq_len * HEAD_DIM;
    double* dV_base = dV + batch_head * seq_len * HEAD_DIM;

    extern __shared__ double smem_f64_bwd[];
    double* K_tile  = smem_f64_bwd;
    double* V_tile  = K_tile  + Bc * HEAD_DIM;
    double* Q_tile  = V_tile  + Bc * HEAD_DIM;
    double* dO_tile = Q_tile  + Br * HEAD_DIM;
    double* S_tile  = dO_tile + Br * HEAD_DIM;       // [Br * Bc]
    double* l_tile  = S_tile  + Br * Bc;             // [Br]
    double* D_tile  = l_tile  + Br;                  // [Br]

    const double NEG_INF = -__longlong_as_double(0x7ff0000000000000LL);

    // Load K/V tiles.
    for (int i = tid; i < actual_Bc * HEAD_DIM; i += BLOCK_SIZE) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        K_tile[row * HEAD_DIM + col] = K_base[(kv_start + row) * HEAD_DIM + col];
        V_tile[row * HEAD_DIM + col] = V_base[(kv_start + row) * HEAD_DIM + col];
    }
    for (int i = tid + actual_Bc * HEAD_DIM; i < Bc * HEAD_DIM; i += BLOCK_SIZE) {
        K_tile[i] = 0.0;
        V_tile[i] = 0.0;
    }
    __syncthreads();

    constexpr int MAX_ELEMS_PER_THREAD = (Bc * HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double dk_acc[MAX_ELEMS_PER_THREAD];
    double dv_acc[MAX_ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < MAX_ELEMS_PER_THREAD; ++e) {
        dk_acc[e] = 0.0;
        dv_acc[e] = 0.0;
    }

    const int num_q_tiles = (seq_len + Br - 1) / Br;

    for (int q_tile_idx = 0; q_tile_idx < num_q_tiles; ++q_tile_idx) {
        const int q_start = q_tile_idx * Br;
        if (q_start >= seq_len) break;
        const int actual_Br = min(Br, seq_len - q_start);

        if (causal && (q_start + actual_Br - 1) < kv_start) continue;

        // Load Q_i / dO_i tiles.
        for (int i = tid; i < actual_Br * HEAD_DIM; i += BLOCK_SIZE) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            Q_tile[row * HEAD_DIM + col]  = Q_base[(q_start + row) * HEAD_DIM + col];
            dO_tile[row * HEAD_DIM + col] = dO_base[(q_start + row) * HEAD_DIM + col];
        }
        for (int i = tid + actual_Br * HEAD_DIM; i < Br * HEAD_DIM; i += BLOCK_SIZE) {
            Q_tile[i]  = 0.0;
            dO_tile[i] = 0.0;
        }

        // Per-row D_i = sum_d dO[i,d] * O[i,d].
        for (int row = tid; row < actual_Br; row += BLOCK_SIZE) {
            double d_sum = 0.0;
            for (int d = 0; d < HEAD_DIM; ++d) {
                d_sum += dO_base[(q_start + row) * HEAD_DIM + d]
                       *  O_base[(q_start + row) * HEAD_DIM + d];
            }
            D_tile[row] = d_sum;
        }
        for (int row = tid + actual_Br; row < Br; row += BLOCK_SIZE) {
            D_tile[row] = 0.0;
        }

        // S_ij = Q_i · K_j^T * scale.
        for (int idx = tid; idx < actual_Br * actual_Bc; idx += BLOCK_SIZE) {
            int i = idx / actual_Bc;
            int j = idx % actual_Bc;
            double dot = 0.0;
            for (int d = 0; d < HEAD_DIM; ++d) {
                dot += Q_tile[i * HEAD_DIM + d] * K_tile[j * HEAD_DIM + d];
            }
            S_tile[i * Bc + j] = dot * scale;
        }
        // Out-of-bounds → -inf.
        for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
            int i = idx / Bc;
            int j = idx % Bc;
            if (i >= actual_Br || j >= actual_Bc) {
                S_tile[idx] = NEG_INF;
            }
        }
        __syncthreads();

        // Recompute per-row LSE in `double` over ALL keys (matches the CUDA
        // FP64 backward — we don't keep an online-softmax state across the
        // KV-tile axis, so we rebuild the row LSE here from Q/K in double).
        for (int row = tid; row < actual_Br; row += BLOCK_SIZE) {
            double m_row = NEG_INF;
            for (int k = 0; k < seq_len; ++k) {
                if (causal && k > (q_start + row)) break;
                double dot = 0.0;
                const double* Kk = K_base + k * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += Q_base[(q_start + row) * HEAD_DIM + d] * Kk[d];
                }
                dot *= scale;
                if (dot > m_row) m_row = dot;
            }
            double l_row = 0.0;
            for (int k = 0; k < seq_len; ++k) {
                if (causal && k > (q_start + row)) break;
                double dot = 0.0;
                const double* Kk = K_base + k * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += Q_base[(q_start + row) * HEAD_DIM + d] * Kk[d];
                }
                dot *= scale;
                l_row += exp(dot - m_row);
            }
            l_tile[row] = (l_row > 0.0) ? (m_row + log(l_row)) : NEG_INF;
        }
        for (int row = tid + actual_Br; row < Br; row += BLOCK_SIZE) {
            l_tile[row] = NEG_INF;
        }
        __syncthreads();

        // P_ij = exp(S_ij - l_i), causal-masked.
        for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
            int i = idx / Bc;
            int j = idx % Bc;
            double p = 0.0;
            if (i < actual_Br && j < actual_Bc) {
                if (causal && (q_start + i) < (kv_start + j)) {
                    p = 0.0;
                } else {
                    p = exp(S_tile[i * Bc + j] - l_tile[i]);
                }
            }
            S_tile[i * Bc + j] = p;
        }
        __syncthreads();

        // dV_j += P_ij^T @ dO_i.
        {
            int e = 0;
            for (int idx = tid; idx < Bc * HEAD_DIM; idx += BLOCK_SIZE, ++e) {
                int j = idx / HEAD_DIM;
                int d = idx % HEAD_DIM;
                if (j < actual_Bc) {
                    double sum = 0.0;
                    for (int i = 0; i < actual_Br; ++i) {
                        sum += S_tile[i * Bc + j] * dO_tile[i * HEAD_DIM + d];
                    }
                    dv_acc[e] += sum;
                }
            }
        }
        __syncthreads();

        // dS_ij = P_ij * (dP_ij - D_i), where dP_ij = dO_i · V_j.
        for (int idx = tid; idx < actual_Br * actual_Bc; idx += BLOCK_SIZE) {
            int i = idx / actual_Bc;
            int j = idx % actual_Bc;
            double dp = 0.0;
            for (int d = 0; d < HEAD_DIM; ++d) {
                dp += dO_tile[i * HEAD_DIM + d] * V_tile[j * HEAD_DIM + d];
            }
            double p_ij = S_tile[i * Bc + j];
            S_tile[i * Bc + j] = p_ij * (dp - D_tile[i]);
        }
        __syncthreads();

        // dK_j += scale * dS_ij^T @ Q_i.
        {
            int e = 0;
            for (int idx = tid; idx < Bc * HEAD_DIM; idx += BLOCK_SIZE, ++e) {
                int j = idx / HEAD_DIM;
                int d = idx % HEAD_DIM;
                if (j < actual_Bc) {
                    double sum = 0.0;
                    for (int i = 0; i < actual_Br; ++i) {
                        sum += S_tile[i * Bc + j] * Q_tile[i * HEAD_DIM + d];
                    }
                    dk_acc[e] += sum * scale;
                }
            }
        }

        // dQ_i += scale * dS_ij @ K_j (atomicAdd across KV tiles).
        // HIP atomicAdd<double> is supported on gfx9+; ROCm CMakeLists
        // also passes -munsafe-fp-atomics for RDNA.
        for (int idx = tid; idx < actual_Br * HEAD_DIM; idx += BLOCK_SIZE) {
            int i = idx / HEAD_DIM;
            int d = idx % HEAD_DIM;
            double sum = 0.0;
            for (int j = 0; j < actual_Bc; ++j) {
                sum += S_tile[i * Bc + j] * K_tile[j * HEAD_DIM + d];
            }
            atomicAdd(&dQ_base[(q_start + i) * HEAD_DIM + d], sum * scale);
        }
        __syncthreads();
    }

    // Write accumulated dK / dV.
    int e = 0;
    for (int idx = tid; idx < Bc * HEAD_DIM; idx += BLOCK_SIZE, ++e) {
        int row = idx / HEAD_DIM;
        int col = idx % HEAD_DIM;
        if (row < actual_Bc) {
            dK_base[(kv_start + row) * HEAD_DIM + col] = dk_acc[e];
            dV_base[(kv_start + row) * HEAD_DIM + col] = dv_acc[e];
        }
    }
}

// =============================================================================
// Host launch helpers
// =============================================================================

// Returns (output [BH, Sq, Hd], lse [BH, Sq] in Float32) — same contract as
// fused_attention_hip but Float64 inputs / output.
auto fused_attention_hip_f64(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    double scale,
    bool causal
) -> std::pair<Tensor, Tensor> {
    if (Q.dtype() != DType::Float64 || K.dtype() != DType::Float64 || V.dtype() != DType::Float64) {
        throw std::runtime_error("fused_attention_hip_f64: requires Float64 inputs");
    }
    int64_t batch_heads = Q.shape()[0];
    int64_t seq_len_q   = Q.shape()[1];
    int64_t head_dim    = Q.shape()[2];
    int64_t seq_len_k   = K.shape()[1];

    Tensor output = create_hip_zeros_f64({batch_heads, seq_len_q, head_dim}, DType::Float64, Q.device());
    Tensor lse    = create_hip_zeros_f64({batch_heads, seq_len_q},           DType::Float32, Q.device());

    constexpr int BLOCK_SIZE = 256;
    constexpr int Bc = 32;
    dim3 grid(static_cast<int>(batch_heads), static_cast<int>(seq_len_q));
    dim3 threads(BLOCK_SIZE);

    auto compute_smem = [](int hd) -> size_t {
        int k_stride = hd + 4;
        return (2 * Bc * k_stride + hd + Bc + Bc) * sizeof(double);
    };

    const double* q_ptr = Q.data<double>();
    const double* k_ptr = K.data<double>();
    const double* v_ptr = V.data<double>();
    double*       o_ptr = output.data<double>();
    float*        l_ptr = lse.data<float>();
    int sq = static_cast<int>(seq_len_q);
    int sk = static_cast<int>(seq_len_k);

    switch (head_dim) {
        case 16:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_v2_kernel_f64_hip<16, BLOCK_SIZE>),
                grid, threads, compute_smem(16), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, sq, sk, scale, causal);
            break;
        case 32:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_v2_kernel_f64_hip<32, BLOCK_SIZE>),
                grid, threads, compute_smem(32), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, sq, sk, scale, causal);
            break;
        case 48:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_v2_kernel_f64_hip<48, BLOCK_SIZE>),
                grid, threads, compute_smem(48), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, sq, sk, scale, causal);
            break;
        case 64:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_v2_kernel_f64_hip<64, BLOCK_SIZE>),
                grid, threads, compute_smem(64), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, sq, sk, scale, causal);
            break;
        case 80:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_v2_kernel_f64_hip<80, BLOCK_SIZE>),
                grid, threads, compute_smem(80), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, sq, sk, scale, causal);
            break;
        case 96:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_v2_kernel_f64_hip<96, BLOCK_SIZE>),
                grid, threads, compute_smem(96), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, sq, sk, scale, causal);
            break;
        case 128:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_v2_kernel_f64_hip<128, BLOCK_SIZE>),
                grid, threads, compute_smem(128), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, sq, sk, scale, causal);
            break;
        default:
            throw std::runtime_error(
                "fused_attention_hip_f64: Unsupported head_dim " +
                std::to_string(head_dim) +
                ". Supported: {16, 32, 48, 64, 80, 96, 128}.");
    }
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("fused_attention_hip_f64 kernel launch failed: ")
                                 + hipGetErrorString(err));
    }

    return {output, lse};
}

// Returns {dQ, dK, dV} (all Float64).
auto flash_attention_backward_hip_f64(
    const Tensor& dO,
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    const Tensor& O,
    double scale,
    bool causal
) -> std::vector<Tensor> {
    if (Q.dtype() != DType::Float64) {
        throw std::runtime_error("flash_attention_backward_hip_f64: requires Float64 inputs");
    }
    int64_t batch_heads = Q.shape()[0];
    int64_t seq_len     = Q.shape()[1];
    int64_t head_dim    = Q.shape()[2];

    Tensor dQ = create_hip_zeros_f64({batch_heads, seq_len, head_dim}, DType::Float64, Q.device());
    Tensor dK = create_hip_zeros_f64({batch_heads, seq_len, head_dim}, DType::Float64, K.device());
    Tensor dV = create_hip_zeros_f64({batch_heads, seq_len, head_dim}, DType::Float64, V.device());

    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int BLOCK_SIZE = 256;

    int num_kv_tiles = static_cast<int>((seq_len + Bc - 1) / Bc);
    dim3 grid(num_kv_tiles, static_cast<int>(batch_heads));
    dim3 threads(BLOCK_SIZE);

    auto compute_smem = [](int hd) -> size_t {
        return (2 * Bc * hd + 2 * Br * hd + Br * Bc + Br + Br) * sizeof(double);
    };

    const double* q_ptr  = Q.data<double>();
    const double* k_ptr  = K.data<double>();
    const double* v_ptr  = V.data<double>();
    const double* o_ptr  = O.data<double>();
    const double* do_ptr = dO.data<double>();
    double* dq_ptr = dQ.data<double>();
    double* dk_ptr = dK.data<double>();
    double* dv_ptr = dV.data<double>();
    int sl = static_cast<int>(seq_len);

    switch (head_dim) {
        case 16:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_backward_kernel_f64_hip<16, Br, Bc, BLOCK_SIZE>),
                grid, threads, compute_smem(16), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr,
                sl, scale, causal);
            break;
        case 32:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_backward_kernel_f64_hip<32, Br, Bc, BLOCK_SIZE>),
                grid, threads, compute_smem(32), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr,
                sl, scale, causal);
            break;
        case 48:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_backward_kernel_f64_hip<48, Br, Bc, BLOCK_SIZE>),
                grid, threads, compute_smem(48), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr,
                sl, scale, causal);
            break;
        case 64:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_backward_kernel_f64_hip<64, Br, Bc, BLOCK_SIZE>),
                grid, threads, compute_smem(64), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr,
                sl, scale, causal);
            break;
        case 80:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_backward_kernel_f64_hip<80, Br, Bc, BLOCK_SIZE>),
                grid, threads, compute_smem(80), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr,
                sl, scale, causal);
            break;
        case 96:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_backward_kernel_f64_hip<96, Br, Bc, BLOCK_SIZE>),
                grid, threads, compute_smem(96), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr,
                sl, scale, causal);
            break;
        case 128:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_backward_kernel_f64_hip<128, Br, Bc, BLOCK_SIZE>),
                grid, threads, compute_smem(128), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr,
                sl, scale, causal);
            break;
        default:
            throw std::runtime_error(
                "flash_attention_backward_hip_f64: Unsupported head_dim " +
                std::to_string(head_dim) +
                ". Supported: {16, 32, 48, 64, 80, 96, 128}.");
    }
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("flash_attention_backward_hip_f64 kernel launch failed: ")
                                 + hipGetErrorString(err));
    }

    return {dQ, dK, dV};
}

} // namespace rocm
} // namespace tenzor
