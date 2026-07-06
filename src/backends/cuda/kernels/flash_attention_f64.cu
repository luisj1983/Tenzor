// =============================================================================
// Float64 FlashAttention forward / backward kernels (audit A.11)
//
// Native double-precision CUDA implementation of FlashAttention. The mainline
// FP32 / FP16 / BF16 kernels in `fused_ops.cu` accumulate in `float`, which
// silently halves precision when Float64 inputs are upcast → Float32 →
// downcast. For Float64 gradcheck (relative tolerance ~1e-7) that round-trip
// is fatal: forward already loses double precision and analytical-vs-numerical
// gradients diverge beyond tolerance.
//
// Algorithm matches the FP32 `flash_attention_v2_kernel` / backward kernel in
// `fused_ops.cu` (online softmax, tiled K/V, dQ via atomicAdd, dK/dV in
// registers) but every accumulator, score, and shared-memory slot is `double`.
// Per docs/internals/attention-contract.md, LSE remains Float32 in the saved
// tensor; we accumulate the LSE in `double` inside the kernel and cast to
// Float32 only on the final store, so backward replays softmax in FP64 from Q
// and K rather than reading the saved LSE.
//
// Causal mask uses -INFINITY (contract sentinel rule). Dropout is intentionally
// not supported in the Float64 path — Float64 attention is gradcheck-only and
// dropout is incompatible with deterministic gradcheck. The dispatcher refuses
// dropout > 0 with Float64 inputs.
// =============================================================================

#ifdef TENZOR_CUDA_AVAILABLE

#include <cuda_runtime.h>
#include "tenzor/core/tensor.hpp"
#include "cuda_common.cuh"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>
#include <limits>

namespace tenzor {
namespace cuda {

namespace {

// Create a zero-filled CUDA tensor — local mirror of the helper in fused_ops.cu
// (file-scope so we don't depend on fused_ops.cu internals).
inline Tensor make_cuda_zeros_f64(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    size_t bytes = t.numel() * dtype_size(dtype);
    TENZOR_CUDA_CHECK(cudaMemsetAsync(t.data_ptr(), 0, bytes, nullptr));
    return t;
}

inline void maybe_set_max_smem(const void* func, size_t smem_bytes) {
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
    }
}

} // anonymous namespace

// Device-side warp reductions for double via 32-bit-pair shuffles.
// __shfl_xor_sync supports 64-bit operands directly on sm_30+, accessed via
// the `long long` overload (or by reinterpreting bits). We use the bit-cast
// path for maximum portability across CUDA toolkits.
__device__ __forceinline__ double warp_max_reduce_f64(double v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        long long bits = __double_as_longlong(v);
        long long peer = __shfl_xor_sync(0xffffffff, bits, offset);
        double pv = __longlong_as_double(peer);
        if (pv > v) v = pv;
    }
    return v;
}

__device__ __forceinline__ double warp_sum_reduce_f64(double v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        long long bits = __double_as_longlong(v);
        long long peer = __shfl_xor_sync(0xffffffff, bits, offset);
        v += __longlong_as_double(peer);
    }
    return v;
}

// =============================================================================
// Float64 FlashAttention forward kernel
// =============================================================================
//
// Grid:  (batch_heads, seq_len_q)
// Block: BLOCK_SIZE threads (default 256)
// Shared memory layout (all `double`):
//   K_tile  [Bc * (HEAD_DIM + 4)]
//   V_tile  [Bc * (HEAD_DIM + 4)]
//   Q_shared[HEAD_DIM]
//   scores  [Bc]
//   reduce  [num_warps]
template<int HEAD_DIM, int BLOCK_SIZE = 256>
__global__ void flash_attention_v2_kernel_f64(
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

    const int batch_head = blockIdx.x;
    const int query_idx  = blockIdx.y;
    if (query_idx >= seq_len_q) return;

    const int tid       = threadIdx.x;
    const int warp_id   = tid / 32;
    const int lane_id   = tid % 32;
    const int num_warps = BLOCK_SIZE / 32;

    const double* Q_row  = Q + batch_head * seq_len_q * HEAD_DIM + query_idx * HEAD_DIM;
    const double* K_base = K + batch_head * seq_len_k * HEAD_DIM;
    const double* V_base = V + batch_head * seq_len_k * HEAD_DIM;
    double*       O_row  = O + batch_head * seq_len_q * HEAD_DIM + query_idx * HEAD_DIM;

    constexpr int K_STRIDE = HEAD_DIM + 4;
    extern __shared__ double smem_f64[];
    double* K_tile        = smem_f64;
    double* V_tile        = smem_f64 + Bc * K_STRIDE;
    double* Q_shared      = smem_f64 + 2 * Bc * K_STRIDE;
    double* scores_shared = Q_shared + HEAD_DIM;
    double* reduce_buf    = scores_shared + Bc;

    // Load Q row, pre-scaled.
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        Q_shared[d] = Q_row[d] * scale;
    }
    __syncthreads();

    // Per-thread output accumulator — each thread owns up to ceil(HEAD_DIM/BLOCK_SIZE) elements.
    // For HEAD_DIM <= 1024 and BLOCK_SIZE = 256, 4 slots cover everything we register.
    double o_local[4] = {0.0, 0.0, 0.0, 0.0};

    double m_prev = -INFINITY;
    double l_prev = 0.0;

    const int num_kv_blocks = (seq_len_k + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int k_start   = kv_block * Bc;
        const int actual_Bc = min(Bc, seq_len_k - k_start);

        for (int idx = tid; idx < actual_Bc * HEAD_DIM; idx += BLOCK_SIZE) {
            int row = idx / HEAD_DIM;
            int col = idx % HEAD_DIM;
            K_tile[row * K_STRIDE + col] = K_base[(k_start + row) * HEAD_DIM + col];
            V_tile[row * K_STRIDE + col] = V_base[(k_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        // Step 1: Q·K scores + per-tile max.
        double local_max = -INFINITY;
        for (int j = tid; j < actual_Bc; j += BLOCK_SIZE) {
            int kv_pos = k_start + j;
            double score;
            // Bottom-right causal alignment (matches FP32 + Vulkan-F64): a query
            // at absolute row `query_idx` attends keys up to
            // query_idx + (seq_len_k - seq_len_q).
            if (causal && kv_pos > query_idx + (seq_len_k - seq_len_q)) {
                score = -INFINITY;
            } else {
                score = 0.0;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_shared[d] * K_tile[j * K_STRIDE + d];
                }
            }
            scores_shared[j] = score;
            if (score > local_max) local_max = score;
        }

        // Warp + cross-warp max reduction (see warp_max_reduce_f64 above).
        local_max = warp_max_reduce_f64(local_max);
        if (lane_id == 0) reduce_buf[warp_id] = local_max;
        __syncthreads();

        if (warp_id == 0) {
            double val = (lane_id < num_warps) ? reduce_buf[lane_id] : -INFINITY;
            val = warp_max_reduce_f64(val);
            if (lane_id == 0) reduce_buf[0] = val;
        }
        __syncthreads();
        double block_max = reduce_buf[0];

        // All-masked tile (fully causal-masked): skip the softmax/accumulate
        // steps but DO NOT `continue` — falling through guarantees the loop
        // body's terminal __syncthreads() at the bottom always executes, so the
        // next iteration's K_tile/V_tile overwrite is correctly ordered against
        // any prior smem use regardless of which branch ran. (Every thread sees
        // the same block_max broadcast via reduce_buf[0], so the branch is
        // warp-uniform — no divergence.) The previous `__syncthreads(); continue;`
        // dropped the terminal sync on this path, a latent smem WAR race.
        if (block_max != -INFINITY) {
            // Step 2: exp(score - max) + per-tile sum.
            double local_sum = 0.0;
            for (int j = tid; j < actual_Bc; j += BLOCK_SIZE) {
                double exp_score = exp(scores_shared[j] - block_max);
                scores_shared[j] = exp_score;
                local_sum += exp_score;
            }
            __syncthreads();

            local_sum = warp_sum_reduce_f64(local_sum);
            if (lane_id == 0) reduce_buf[warp_id] = local_sum;
            __syncthreads();

            if (warp_id == 0) {
                double val = (lane_id < num_warps) ? reduce_buf[lane_id] : 0.0;
                val = warp_sum_reduce_f64(val);
                if (lane_id == 0) reduce_buf[0] = val;
            }
            __syncthreads();
            double block_sum = reduce_buf[0];

            // Step 3: online-softmax merge.
            double m_new    = (m_prev > block_max) ? m_prev : block_max;
            double exp_prev = exp(m_prev - m_new);
            double exp_curr = exp(block_max - m_new);
            double l_new    = exp_prev * l_prev + exp_curr * block_sum;

            // Step 4: rescale previous output and accumulate P @ V for this tile.
            for (int i = 0; i < 4; ++i) {
                int d = tid + i * BLOCK_SIZE;
                if (d < HEAD_DIM) {
                    o_local[i] *= exp_prev;
                    double pv_sum = 0.0;
                    for (int j = 0; j < actual_Bc; ++j) {
                        pv_sum += scores_shared[j] * V_tile[j * K_STRIDE + d];
                    }
                    o_local[i] += exp_curr * pv_sum;
                }
            }

            m_prev = m_new;
            l_prev = l_new;
        }
        __syncthreads();
    }

    // Final normalize + write.
    double l_inv = (l_prev > 0.0) ? (1.0 / l_prev) : 0.0;
    for (int i = 0; i < 4; ++i) {
        int d = tid + i * BLOCK_SIZE;
        if (d < HEAD_DIM) {
            O_row[d] = o_local[i] * l_inv;
        }
    }

    // LSE in Float32 per contract (downcast on store). Backward path for FP64
    // recomputes softmax in double from Q/K anyway, so this is just a stub for
    // the FlashAttention 4-output contract.
    if (L != nullptr && tid == 0) {
        double lse = m_prev + log(fmax(l_prev, 1e-300));
        L[batch_head * seq_len_q + query_idx] = static_cast<float>(lse);
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
__global__ void flash_attention_backward_kernel_f64(
    const double* __restrict__ Q,
    const double* __restrict__ K,
    const double* __restrict__ V,
    const double* __restrict__ O,
    const double* __restrict__ dO,
    double* __restrict__ dQ,          // accumulated via atomicAdd
    double* __restrict__ dK,          // one block per KV tile → no race
    double* __restrict__ dV,          // one block per KV tile → no race
    const int seq_len_q,
    const int seq_len_k,
    const double scale,
    const bool causal
) {
    const int kv_tile_idx = blockIdx.x;
    const int batch_head  = blockIdx.y;
    const int tid         = threadIdx.x;

    // Bottom-right causal alignment (matches the forward kernel + FP32 path):
    // query row r attends keys up to r + (seq_len_k - seq_len_q).
    const int causal_offset = seq_len_k - seq_len_q;

    const int kv_start = kv_tile_idx * Bc;
    if (kv_start >= seq_len_k) return;
    const int actual_Bc = min(Bc, seq_len_k - kv_start);

    // Q/O/dO/dQ stride by seq_len_q; K/V/dK/dV stride by seq_len_k.
    const double* Q_base  = Q  + batch_head * seq_len_q * HEAD_DIM;
    const double* K_base  = K  + batch_head * seq_len_k * HEAD_DIM;
    const double* V_base  = V  + batch_head * seq_len_k * HEAD_DIM;
    const double* O_base  = O  + batch_head * seq_len_q * HEAD_DIM;
    const double* dO_base = dO + batch_head * seq_len_q * HEAD_DIM;
    double* dQ_base = dQ + batch_head * seq_len_q * HEAD_DIM;
    double* dK_base = dK + batch_head * seq_len_k * HEAD_DIM;
    double* dV_base = dV + batch_head * seq_len_k * HEAD_DIM;

    extern __shared__ double smem_bwd_f64[];
    double* K_tile  = smem_bwd_f64;
    double* V_tile  = K_tile  + Bc * HEAD_DIM;
    double* Q_tile  = V_tile  + Bc * HEAD_DIM;
    double* dO_tile = Q_tile  + Br * HEAD_DIM;
    double* S_tile  = dO_tile + Br * HEAD_DIM;       // [Br * Bc]
    double* l_tile  = S_tile  + Br * Bc;             // [Br]  per-row LSE (recomputed)
    double* D_tile  = l_tile  + Br;                  // [Br]  per-row rowsum(dO·O)

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

    const int num_q_tiles = (seq_len_q + Br - 1) / Br;

    for (int q_tile_idx = 0; q_tile_idx < num_q_tiles; ++q_tile_idx) {
        const int q_start = q_tile_idx * Br;
        if (q_start >= seq_len_q) break;
        const int actual_Br = min(Br, seq_len_q - q_start);

        if (causal && (q_start + actual_Br - 1) + causal_offset < kv_start) continue;

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

        // Per-row D_i = sum_d dO[i,d] * O[i,d]. Recomputed here in double.
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
        // Q_tile/dO_tile are written cooperatively above and read below by other
        // threads; without this barrier the S_ij reads race the tile stores,
        // corrupting FP64 gradients. Mirrors the FP32 kernel (fused_ops.cu).
        __syncthreads();

        // S_ij = Q_i @ K_j^T * scale  (rebuilt every Q-tile pass).
        for (int idx = tid; idx < actual_Br * actual_Bc; idx += BLOCK_SIZE) {
            int i = idx / actual_Bc;
            int j = idx % actual_Bc;
            double dot = 0.0;
            for (int d = 0; d < HEAD_DIM; ++d) {
                dot += Q_tile[i * HEAD_DIM + d] * K_tile[j * HEAD_DIM + d];
            }
            S_tile[i * Bc + j] = dot * scale;
        }
        // Out-of-bounds → -inf so they get exp() = 0.
        for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
            int i = idx / Bc;
            int j = idx % Bc;
            if (i >= actual_Br || j >= actual_Bc) {
                S_tile[idx] = -INFINITY;
            }
        }
        __syncthreads();

        // Recompute per-row LSE in `double` over ALL keys (not just this tile)
        // — necessary because we don't keep an online-softmax state across the
        // KV-tile axis (this block owns one KV tile and iterates Q tiles).
        //
        // For each query row in this tile, scan all keys (in chunks of Bc by
        // re-reading K from global) and compute m_row + log(sum exp). Do this
        // once per Q-tile pass; cost is O(seq_len * HEAD_DIM) per row, which
        // dominates only when seq_len >> Bc; acceptable for an FP64 gradcheck
        // kernel.
        for (int row = tid; row < actual_Br; row += BLOCK_SIZE) {
            double m_row = -INFINITY;
            // First pass: find max.
            for (int k = 0; k < seq_len_k; ++k) {
                if (causal && k > (q_start + row) + causal_offset) break;  // upper-tri zeroed below
                double dot = 0.0;
                const double* Kk = K_base + k * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += Q_base[(q_start + row) * HEAD_DIM + d] * Kk[d];
                }
                dot *= scale;
                if (dot > m_row) m_row = dot;
            }
            // Second pass: sum of exp(score - m).
            double l_row = 0.0;
            for (int k = 0; k < seq_len_k; ++k) {
                if (causal && k > (q_start + row) + causal_offset) break;
                double dot = 0.0;
                const double* Kk = K_base + k * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += Q_base[(q_start + row) * HEAD_DIM + d] * Kk[d];
                }
                dot *= scale;
                l_row += exp(dot - m_row);
            }
            l_tile[row] = (l_row > 0.0) ? (m_row + log(l_row)) : -INFINITY;
        }
        for (int row = tid + actual_Br; row < Br; row += BLOCK_SIZE) {
            l_tile[row] = -INFINITY;
        }
        __syncthreads();

        // P_ij = exp(S_ij - l_i), causal-masked.
        for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
            int i = idx / Bc;
            int j = idx % Bc;
            double p = 0.0;
            if (i < actual_Br && j < actual_Bc) {
                if (causal && (q_start + i) + causal_offset < (kv_start + j)) {
                    p = 0.0;
                } else {
                    p = exp(S_tile[i * Bc + j] - l_tile[i]);
                }
            }
            S_tile[i * Bc + j] = p;
        }
        __syncthreads();

        // dV_j += P_ij^T @ dO_i  →  [Bc, HEAD_DIM]
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

        // dK_j += scale * dS_ij^T @ Q_i
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

        // dQ_i += scale * dS_ij @ K_j  (atomicAdd across KV tiles).
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

// Returns (output [BH, Sq, Hd], lse [BH, Sq] in Float32).
auto fused_attention_cuda_f64(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    double scale,
    bool causal
) -> std::pair<Tensor, Tensor> {
    if (Q.dtype() != DType::Float64 || K.dtype() != DType::Float64 || V.dtype() != DType::Float64) {
        throw std::runtime_error("fused_attention_cuda_f64: requires Float64 inputs");
    }
    int64_t batch_heads = Q.shape()[0];
    int64_t seq_len_q   = Q.shape()[1];
    int64_t head_dim    = Q.shape()[2];
    int64_t seq_len_k   = K.shape()[1];

    Tensor output = make_cuda_zeros_f64({batch_heads, seq_len_q, head_dim}, DType::Float64, Q.device());
    Tensor lse    = make_cuda_zeros_f64({batch_heads, seq_len_q},          DType::Float32, Q.device());

    constexpr int BLOCK_SIZE = 256;
    constexpr int Bc = 32;
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(batch_heads, seq_len_q);

    auto compute_smem = [](int hd) -> size_t {
        int k_stride = hd + 4;
        return (2 * Bc * k_stride + hd + Bc + 8) * sizeof(double);
    };

    const double* q_ptr = Q.data<double>();
    const double* k_ptr = K.data<double>();
    const double* v_ptr = V.data<double>();
    double*       o_ptr = output.data<double>();
    float*        l_ptr = lse.data<float>();
    int sq = static_cast<int>(seq_len_q);
    int sk = static_cast<int>(seq_len_k);

    auto launch = [&](auto kernel_fn, int hd) {
        size_t smem = compute_smem(hd);
        maybe_set_max_smem(reinterpret_cast<const void*>(kernel_fn), smem);
        kernel_fn<<<blocks, threads, smem>>>(q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
                                              sq, sk, scale, causal);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    };

    switch (head_dim) {
        case 16:  launch(flash_attention_v2_kernel_f64<16,  BLOCK_SIZE>, 16);  break;
        case 32:  launch(flash_attention_v2_kernel_f64<32,  BLOCK_SIZE>, 32);  break;
        case 48:  launch(flash_attention_v2_kernel_f64<48,  BLOCK_SIZE>, 48);  break;
        case 64:  launch(flash_attention_v2_kernel_f64<64,  BLOCK_SIZE>, 64);  break;
        case 80:  launch(flash_attention_v2_kernel_f64<80,  BLOCK_SIZE>, 80);  break;
        case 96:  launch(flash_attention_v2_kernel_f64<96,  BLOCK_SIZE>, 96);  break;
        case 128: launch(flash_attention_v2_kernel_f64<128, BLOCK_SIZE>, 128); break;
        default:
            throw std::runtime_error(
                "fused_attention_cuda_f64: Unsupported head_dim " +
                std::to_string(head_dim) +
                ". Supported: {16, 32, 48, 64, 80, 96, 128}.");
    }

    return {output, lse};
}

// Returns {dQ, dK, dV} (all Float64).
auto flash_attention_backward_cuda_f64(
    const Tensor& dO,
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    const Tensor& O,
    double scale,
    bool causal
) -> std::vector<Tensor> {
    if (Q.dtype() != DType::Float64) {
        throw std::runtime_error("flash_attention_backward_cuda_f64: requires Float64 inputs");
    }
    int64_t batch_heads = Q.shape()[0];
    int64_t seq_len_q   = Q.shape()[1];
    int64_t head_dim    = Q.shape()[2];
    int64_t seq_len_k   = K.shape()[1];

    Tensor dQ = make_cuda_zeros_f64({batch_heads, seq_len_q, head_dim}, DType::Float64, Q.device());
    Tensor dK = make_cuda_zeros_f64({batch_heads, seq_len_k, head_dim}, DType::Float64, K.device());
    Tensor dV = make_cuda_zeros_f64({batch_heads, seq_len_k, head_dim}, DType::Float64, V.device());

    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int BLOCK_SIZE = 256;

    int num_kv_tiles = static_cast<int>((seq_len_k + Bc - 1) / Bc);
    dim3 grid(num_kv_tiles, batch_heads);
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
    int sq = static_cast<int>(seq_len_q);
    int sk = static_cast<int>(seq_len_k);

    auto launch = [&](auto kernel_fn, int hd) {
        size_t smem = compute_smem(hd);
        maybe_set_max_smem(reinterpret_cast<const void*>(kernel_fn), smem);
        kernel_fn<<<grid, threads, smem>>>(q_ptr, k_ptr, v_ptr, o_ptr, do_ptr,
                                            dq_ptr, dk_ptr, dv_ptr,
                                            sq, sk, scale, causal);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    };

    switch (head_dim) {
        case 16:  launch(flash_attention_backward_kernel_f64<16,  Br, Bc, BLOCK_SIZE>, 16);  break;
        case 32:  launch(flash_attention_backward_kernel_f64<32,  Br, Bc, BLOCK_SIZE>, 32);  break;
        case 48:  launch(flash_attention_backward_kernel_f64<48,  Br, Bc, BLOCK_SIZE>, 48);  break;
        case 64:  launch(flash_attention_backward_kernel_f64<64,  Br, Bc, BLOCK_SIZE>, 64);  break;
        case 80:  launch(flash_attention_backward_kernel_f64<80,  Br, Bc, BLOCK_SIZE>, 80);  break;
        case 96:  launch(flash_attention_backward_kernel_f64<96,  Br, Bc, BLOCK_SIZE>, 96);  break;
        case 128: launch(flash_attention_backward_kernel_f64<128, Br, Bc, BLOCK_SIZE>, 128); break;
        default:
            throw std::runtime_error(
                "flash_attention_backward_cuda_f64: Unsupported head_dim " +
                std::to_string(head_dim) +
                ". Supported: {16, 32, 48, 64, 80, 96, 128}.");
    }

    return {dQ, dK, dV};
}

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_CUDA_AVAILABLE
