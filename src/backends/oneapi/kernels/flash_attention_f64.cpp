// =============================================================================
// Float64 FlashAttention forward / backward kernels (audit A.11, OneAPI/SYCL)
//
// Native double-precision SYCL implementation of FlashAttention. The mainline
// OneAPI FP32 path in `fused_ops.cpp` does already template on ComputeT so a
// Float64 specialisation accumulates in `double`, but A.11 wants a dedicated
// FP64 kernel that exactly mirrors the CUDA/ROCm reference (separate compile
// unit, native FP64 backward, head_dim coverage {16, 32, 48, 64, 80, 96, 128}).
// This file is that kernel.
//
// Algorithm matches the CUDA reference in
// `src/backends/cuda/kernels/flash_attention_f64.cu` and the ROCm port in
// `src/backends/rocm/kernels/flash_attention_f64.hip.cpp` (online softmax,
// tiled K/V, dQ via atomic_ref<double>, dK/dV in registers, recompute LSE
// during backward in `double` instead of reading the saved Float32 LSE).
//
// SYCL specifics vs the CUDA / ROCm reference:
//   * `__global__` kernel → `queue.submit` + `cgh.parallel_for` over an
//     `nd_range<2>` with one work-group per (batch_head, query_row) /
//     (batch_head, kv_tile). This matches the existing FP32 FlashAttention
//     OneAPI kernel style — `parallel_for_work_group` is not used elsewhere
//     in this codebase.
//   * `__shfl_xor_sync` → `sycl::reduce_over_group(item.get_group(), v, op)`.
//     This collective has an implicit work-group barrier, so we can drop the
//     explicit cross-warp reduction scratch buffer used by the CUDA kernel.
//   * `__shared__` → `sycl::local_accessor<double, 1>`. Pointer is recovered
//     via `local_mem.get_multi_ptr<sycl::access::decorated::no>().get()` —
//     same idiom the FP32 kernel uses.
//   * `atomicAdd<double>` → `sycl::atomic_ref<double, relaxed, device,
//     global_space>` (already used in indexing.cpp / pooling.cpp / vision.cpp).
//   * `__double_as_longlong` → `sycl::bit_cast<int64_t, double>` is not needed:
//     we just use `-std::numeric_limits<double>::infinity()` directly. SYCL
//     guarantees IEEE-754 representation for -inf and `sycl::exp(-inf)` → 0.
//
// Causal mask uses -INFINITY (contract sentinel rule). Dropout is intentionally
// not supported in the Float64 path — Float64 attention is gradcheck-only and
// dropout is incompatible with deterministic gradcheck. The dispatcher refuses
// dropout > 0 with Float64 inputs in `function_attention.cpp`.
// =============================================================================

#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tenzor {
namespace oneapi {

// SYCL kernel name classes — one per (head_dim, direction) to keep the JIT
// name graph deterministic and let the runtime cache specialised kernels.
namespace fa_f64_names {
template<int HEAD_DIM> struct ForwardF64 {};
template<int HEAD_DIM> struct BackwardF64 {};
}  // namespace fa_f64_names

namespace {

template<typename T>
inline T* get_dp(const Tensor& t) {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

inline Tensor make_zeros_f64(const std::vector<int64_t>& shape, DType dtype,
                              Device device, sycl::queue& queue) {
    Tensor t(shape, dtype, device);
    size_t bytes = t.numel() * dtype_size(dtype);
    if (bytes > 0) {
        queue.memset(t.data_ptr(), 0, bytes).wait();
    }
    return t;
}

}  // anonymous namespace

// =============================================================================
// Float64 FlashAttention forward kernel
// =============================================================================
//
// nd_range<2>:
//   global = (batch_heads, seq_len_q * BLOCK_SIZE)
//   local  = (1,           BLOCK_SIZE)
// Each work-group owns one (batch_head, query_row).
//
// Local memory layout (all `double`):
//   K_tile        [Bc * K_STRIDE]
//   V_tile        [Bc * K_STRIDE]
//   Q_shared      [HEAD_DIM]
//   scores_shared [Bc]
template<int HEAD_DIM>
static void launch_flash_attention_f64_forward(
    const double* q_ptr, const double* k_ptr, const double* v_ptr,
    double* o_ptr, float* l_ptr,
    int64_t batch_heads, int64_t seq_len_q, int64_t seq_len_k,
    double scale, bool causal, sycl::queue& queue)
{
    constexpr int Bc = 32;
    constexpr int K_STRIDE = HEAD_DIM + 4;  // pad to dodge bank conflicts
    constexpr int BLOCK_SIZE = 128;
    constexpr int MAX_D_PER_THREAD = (HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const size_t local_doubles =
        static_cast<size_t>(2 * Bc * K_STRIDE + HEAD_DIM + Bc);

    const int hd  = HEAD_DIM;
    const int slq = static_cast<int>(seq_len_q);
    const int slk = static_cast<int>(seq_len_k);
    const int ks  = K_STRIDE;
    const double sc = scale;
    const bool  cs = causal;

    queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<double, 1> local_mem(local_doubles, cgh);

        cgh.parallel_for<fa_f64_names::ForwardF64<HEAD_DIM>>(
            sycl::nd_range<2>(
                sycl::range<2>(static_cast<size_t>(batch_heads),
                               static_cast<size_t>(seq_len_q) * BLOCK_SIZE),
                sycl::range<2>(1, BLOCK_SIZE)),
            [=](sycl::nd_item<2> item) {
                const int batch_head = static_cast<int>(item.get_global_id(0));
                const int query_idx  = static_cast<int>(item.get_global_id(1)) / BLOCK_SIZE;
                const int tid        = static_cast<int>(item.get_local_id(1));

                if (query_idx >= slq) return;

                double* lmem = local_mem.template get_multi_ptr<sycl::access::decorated::no>().get();
                double* K_tile        = lmem;                              // [Bc][ks]
                double* V_tile        = lmem + Bc * ks;                    // [Bc][ks]
                double* Q_shared      = lmem + 2 * Bc * ks;                // [hd]
                double* scores_shared = Q_shared + hd;                     // [Bc]

                const double* Q_row  = q_ptr + batch_head * slq * hd + query_idx * hd;
                const double* K_base = k_ptr + batch_head * slk * hd;
                const double* V_base = v_ptr + batch_head * slk * hd;
                double*       O_row  = o_ptr + batch_head * slq * hd + query_idx * hd;

                // Load Q row, pre-scaled.
                for (int d = tid; d < hd; d += BLOCK_SIZE) {
                    Q_shared[d] = Q_row[d] * sc;
                }
                sycl::group_barrier(item.get_group());

                double o_local[MAX_D_PER_THREAD];
                for (int i = 0; i < MAX_D_PER_THREAD; ++i) o_local[i] = 0.0;

                double m_prev = -std::numeric_limits<double>::infinity();
                double l_prev = 0.0;

                const int num_kv_blocks = (slk + Bc - 1) / Bc;

                for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
                    const int k_start = kv_block * Bc;

                    // Causal early-exit: skip blocks entirely past the boundary.
                    if (cs && k_start > query_idx) break;

                    const int actual_Bc = sycl::min(Bc, slk - k_start);

                    // Load K/V tile cooperatively into local memory.
                    for (int idx = tid; idx < actual_Bc * hd; idx += BLOCK_SIZE) {
                        int row = idx / hd;
                        int col = idx % hd;
                        K_tile[row * ks + col] = K_base[(k_start + row) * hd + col];
                        V_tile[row * ks + col] = V_base[(k_start + row) * hd + col];
                    }
                    sycl::group_barrier(item.get_group());

                    // Step 1: Q · K scores + per-tile max.
                    double local_max = -std::numeric_limits<double>::infinity();
                    for (int j = tid; j < actual_Bc; j += BLOCK_SIZE) {
                        int kv_pos = k_start + j;
                        double score;
                        if (cs && kv_pos > query_idx) {
                            score = -std::numeric_limits<double>::infinity();
                        } else {
                            score = 0.0;
                            for (int d = 0; d < hd; ++d) {
                                score += Q_shared[d] * K_tile[j * ks + d];
                            }
                        }
                        scores_shared[j] = score;
                        local_max = sycl::fmax(local_max, score);
                    }

                    // reduce_over_group has an implicit barrier so the writes
                    // to scores_shared above are visible to all threads before
                    // step 2 reads them back.
                    double block_max = sycl::reduce_over_group(
                        item.get_group(), local_max, sycl::maximum<double>());
                    sycl::group_barrier(item.get_group());

                    // All-masked tile early-out (matches FP32 / CUDA kernels).
                    if (block_max == -std::numeric_limits<double>::infinity()) {
                        sycl::group_barrier(item.get_group());
                        continue;
                    }

                    // Step 2: exp(score - max) + per-tile sum.
                    double local_sum = 0.0;
                    for (int j = tid; j < actual_Bc; j += BLOCK_SIZE) {
                        double exp_score = sycl::exp(scores_shared[j] - block_max);
                        scores_shared[j] = exp_score;
                        local_sum += exp_score;
                    }
                    sycl::group_barrier(item.get_group());

                    double block_sum = sycl::reduce_over_group(
                        item.get_group(), local_sum, sycl::plus<double>());

                    // Step 3: online-softmax merge.
                    double m_new    = sycl::fmax(m_prev, block_max);
                    double exp_prev = sycl::exp(m_prev - m_new);
                    double exp_curr = sycl::exp(block_max - m_new);
                    double l_new    = exp_prev * l_prev + exp_curr * block_sum;

                    // Step 4: rescale previous output and accumulate P @ V.
                    for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
                        int d = tid + i * BLOCK_SIZE;
                        if (d < hd) {
                            o_local[i] *= exp_prev;
                            double pv_sum = 0.0;
                            for (int j = 0; j < actual_Bc; ++j) {
                                pv_sum += scores_shared[j] * V_tile[j * ks + d];
                            }
                            o_local[i] += exp_curr * pv_sum;
                        }
                    }

                    m_prev = m_new;
                    l_prev = l_new;
                    sycl::group_barrier(item.get_group());
                }

                // Final normalize + write.
                double l_inv = (l_prev > 0.0) ? (1.0 / l_prev) : 0.0;
                for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
                    int d = tid + i * BLOCK_SIZE;
                    if (d < hd) {
                        O_row[d] = o_local[i] * l_inv;
                    }
                }

                // LSE in Float32 per contract (downcast on store). Backward
                // replays softmax in double from Q/K anyway, so this is just a
                // stub for the FlashAttention 4-output contract.
                if (l_ptr != nullptr && tid == 0) {
                    double lse;
                    if (l_prev > 0.0) {
                        lse = m_prev + sycl::log(l_prev);
                    } else {
                        lse = -std::numeric_limits<double>::infinity();
                    }
                    l_ptr[batch_head * slq + query_idx] = static_cast<float>(lse);
                }
            });
    });
}

// =============================================================================
// Float64 FlashAttention backward kernel
// =============================================================================
//
// Same tiled structure as the FP32 backward kernel: nd_range<2> with one
// work-group per (batch_head, kv_tile). Recomputes scores & per-row softmax
// denominator in `double` from Q, K (rather than reading the FP32 LSE) so the
// backward is correct to full double precision.
template<int HEAD_DIM>
static void launch_flash_attention_f64_backward(
    const double* q_ptr, const double* k_ptr, const double* v_ptr,
    const double* o_ptr, const double* do_ptr,
    double* dq_ptr, double* dk_ptr, double* dv_ptr,
    int64_t batch_heads, int64_t seq_len,
    double scale, bool causal, sycl::queue& queue)
{
    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int BLOCK_SIZE = 128;
    constexpr int MAX_ELEMS_PER_THREAD = (Bc * HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Local memory: K [Bc·D] | V [Bc·D] | Q [Br·D] | dO [Br·D] | S [Br·Bc] | l [Br] | D [Br]
    const size_t local_doubles = static_cast<size_t>(
        2 * Bc * HEAD_DIM + 2 * Br * HEAD_DIM + Br * Bc + Br + Br);

    const int hd = HEAD_DIM;
    const int sl = static_cast<int>(seq_len);
    const double sc = scale;
    const bool cs = causal;

    const int num_kv_tiles = (sl + Bc - 1) / Bc;
    const int num_q_tiles  = (sl + Br - 1) / Br;

    queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<double, 1> local_mem(local_doubles, cgh);

        cgh.parallel_for<fa_f64_names::BackwardF64<HEAD_DIM>>(
            sycl::nd_range<2>(
                sycl::range<2>(static_cast<size_t>(batch_heads),
                               static_cast<size_t>(num_kv_tiles) * BLOCK_SIZE),
                sycl::range<2>(1, BLOCK_SIZE)),
            [=](sycl::nd_item<2> item) {
                const int batch_head  = static_cast<int>(item.get_global_id(0));
                const int kv_tile_idx = static_cast<int>(item.get_global_id(1)) / BLOCK_SIZE;
                const int tid         = static_cast<int>(item.get_local_id(1));

                const int kv_start = kv_tile_idx * Bc;
                if (kv_start >= sl) return;
                const int actual_Bc = sycl::min(Bc, sl - kv_start);

                const double* Q_base  = q_ptr  + batch_head * sl * hd;
                const double* K_base  = k_ptr  + batch_head * sl * hd;
                const double* V_base  = v_ptr  + batch_head * sl * hd;
                const double* O_base  = o_ptr  + batch_head * sl * hd;
                const double* dO_base = do_ptr + batch_head * sl * hd;
                double* dQ_base = dq_ptr + batch_head * sl * hd;
                double* dK_base = dk_ptr + batch_head * sl * hd;
                double* dV_base = dv_ptr + batch_head * sl * hd;

                double* lmem = local_mem.template get_multi_ptr<sycl::access::decorated::no>().get();
                double* K_tile  = lmem;
                double* V_tile  = K_tile  + Bc * hd;
                double* Q_tile  = V_tile  + Bc * hd;
                double* dO_tile = Q_tile  + Br * hd;
                double* S_tile  = dO_tile + Br * hd;       // [Br * Bc]
                double* l_tile  = S_tile  + Br * Bc;       // [Br]
                double* D_tile  = l_tile  + Br;            // [Br]

                // Load K/V tiles.
                for (int i = tid; i < actual_Bc * hd; i += BLOCK_SIZE) {
                    int row = i / hd;
                    int col = i % hd;
                    K_tile[row * hd + col] = K_base[(kv_start + row) * hd + col];
                    V_tile[row * hd + col] = V_base[(kv_start + row) * hd + col];
                }
                for (int i = tid + actual_Bc * hd; i < Bc * hd; i += BLOCK_SIZE) {
                    K_tile[i] = 0.0;
                    V_tile[i] = 0.0;
                }
                sycl::group_barrier(item.get_group());

                // Per-thread dK / dV accumulators (one block per KV tile → no race).
                double dk_acc[MAX_ELEMS_PER_THREAD];
                double dv_acc[MAX_ELEMS_PER_THREAD];
                for (int e = 0; e < MAX_ELEMS_PER_THREAD; ++e) {
                    dk_acc[e] = 0.0;
                    dv_acc[e] = 0.0;
                }

                for (int q_tile_idx = 0; q_tile_idx < num_q_tiles; ++q_tile_idx) {
                    const int q_start = q_tile_idx * Br;
                    if (q_start >= sl) break;
                    const int actual_Br = sycl::min(Br, sl - q_start);

                    if (cs && (q_start + actual_Br - 1) < kv_start) continue;

                    // Load Q_i / dO_i tiles.
                    for (int i = tid; i < actual_Br * hd; i += BLOCK_SIZE) {
                        int row = i / hd;
                        int col = i % hd;
                        Q_tile[row * hd + col]  = Q_base[(q_start + row) * hd + col];
                        dO_tile[row * hd + col] = dO_base[(q_start + row) * hd + col];
                    }
                    for (int i = tid + actual_Br * hd; i < Br * hd; i += BLOCK_SIZE) {
                        Q_tile[i]  = 0.0;
                        dO_tile[i] = 0.0;
                    }

                    // Per-row D_i = sum_d dO[i,d] * O[i,d]. Recomputed in double.
                    for (int row = tid; row < actual_Br; row += BLOCK_SIZE) {
                        double d_sum = 0.0;
                        for (int d = 0; d < hd; ++d) {
                            d_sum += dO_base[(q_start + row) * hd + d]
                                   *  O_base[(q_start + row) * hd + d];
                        }
                        D_tile[row] = d_sum;
                    }
                    for (int row = tid + actual_Br; row < Br; row += BLOCK_SIZE) {
                        D_tile[row] = 0.0;
                    }

                    // S_ij = Q_i @ K_j^T * scale.
                    for (int idx = tid; idx < actual_Br * actual_Bc; idx += BLOCK_SIZE) {
                        int i = idx / actual_Bc;
                        int j = idx % actual_Bc;
                        double dot = 0.0;
                        for (int d = 0; d < hd; ++d) {
                            dot += Q_tile[i * hd + d] * K_tile[j * hd + d];
                        }
                        S_tile[i * Bc + j] = dot * sc;
                    }
                    // Out-of-bounds → -inf so they get exp() = 0.
                    for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
                        int i = idx / Bc;
                        int j = idx % Bc;
                        if (i >= actual_Br || j >= actual_Bc) {
                            S_tile[idx] = -std::numeric_limits<double>::infinity();
                        }
                    }
                    sycl::group_barrier(item.get_group());

                    // Recompute per-row LSE in `double` over ALL keys (matches
                    // CUDA/ROCm FP64 backward — we don't keep an online-softmax
                    // state across the KV-tile axis, so we rebuild the row LSE
                    // here from Q/K in double).
                    for (int row = tid; row < actual_Br; row += BLOCK_SIZE) {
                        double m_row = -std::numeric_limits<double>::infinity();
                        for (int k = 0; k < sl; ++k) {
                            if (cs && k > (q_start + row)) break;
                            double dot = 0.0;
                            const double* Kk = K_base + k * hd;
                            for (int d = 0; d < hd; ++d) {
                                dot += Q_base[(q_start + row) * hd + d] * Kk[d];
                            }
                            dot *= sc;
                            if (dot > m_row) m_row = dot;
                        }
                        double l_row = 0.0;
                        for (int k = 0; k < sl; ++k) {
                            if (cs && k > (q_start + row)) break;
                            double dot = 0.0;
                            const double* Kk = K_base + k * hd;
                            for (int d = 0; d < hd; ++d) {
                                dot += Q_base[(q_start + row) * hd + d] * Kk[d];
                            }
                            dot *= sc;
                            l_row += sycl::exp(dot - m_row);
                        }
                        l_tile[row] = (l_row > 0.0)
                            ? (m_row + sycl::log(l_row))
                            : -std::numeric_limits<double>::infinity();
                    }
                    for (int row = tid + actual_Br; row < Br; row += BLOCK_SIZE) {
                        l_tile[row] = -std::numeric_limits<double>::infinity();
                    }
                    sycl::group_barrier(item.get_group());

                    // P_ij = exp(S_ij - l_i), causal-masked.
                    for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
                        int i = idx / Bc;
                        int j = idx % Bc;
                        double p = 0.0;
                        if (i < actual_Br && j < actual_Bc) {
                            if (cs && (q_start + i) < (kv_start + j)) {
                                p = 0.0;
                            } else {
                                p = sycl::exp(S_tile[i * Bc + j] - l_tile[i]);
                            }
                        }
                        S_tile[i * Bc + j] = p;
                    }
                    sycl::group_barrier(item.get_group());

                    // dV_j += P_ij^T @ dO_i.
                    {
                        int e = 0;
                        for (int idx = tid; idx < Bc * hd; idx += BLOCK_SIZE, ++e) {
                            int j = idx / hd;
                            int d = idx % hd;
                            if (j < actual_Bc) {
                                double sum = 0.0;
                                for (int i = 0; i < actual_Br; ++i) {
                                    sum += S_tile[i * Bc + j] * dO_tile[i * hd + d];
                                }
                                dv_acc[e] += sum;
                            }
                        }
                    }
                    sycl::group_barrier(item.get_group());

                    // dS_ij = P_ij * (dP_ij - D_i), where dP_ij = dO_i · V_j.
                    for (int idx = tid; idx < actual_Br * actual_Bc; idx += BLOCK_SIZE) {
                        int i = idx / actual_Bc;
                        int j = idx % actual_Bc;
                        double dp = 0.0;
                        for (int d = 0; d < hd; ++d) {
                            dp += dO_tile[i * hd + d] * V_tile[j * hd + d];
                        }
                        double p_ij = S_tile[i * Bc + j];
                        S_tile[i * Bc + j] = p_ij * (dp - D_tile[i]);
                    }
                    sycl::group_barrier(item.get_group());

                    // dK_j += scale * dS_ij^T @ Q_i.
                    {
                        int e = 0;
                        for (int idx = tid; idx < Bc * hd; idx += BLOCK_SIZE, ++e) {
                            int j = idx / hd;
                            int d = idx % hd;
                            if (j < actual_Bc) {
                                double sum = 0.0;
                                for (int i = 0; i < actual_Br; ++i) {
                                    sum += S_tile[i * Bc + j] * Q_tile[i * hd + d];
                                }
                                dk_acc[e] += sum * sc;
                            }
                        }
                    }

                    // dQ_i += scale * dS_ij @ K_j  (atomic_ref across KV tiles).
                    for (int idx = tid; idx < actual_Br * hd; idx += BLOCK_SIZE) {
                        int i = idx / hd;
                        int d = idx % hd;
                        double sum = 0.0;
                        for (int j = 0; j < actual_Bc; ++j) {
                            sum += S_tile[i * Bc + j] * K_tile[j * hd + d];
                        }
                        sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_dq(dQ_base[(q_start + i) * hd + d]);
                        atomic_dq.fetch_add(sum * sc);
                    }
                    sycl::group_barrier(item.get_group());
                }

                // Write accumulated dK / dV.
                int e = 0;
                for (int idx = tid; idx < Bc * hd; idx += BLOCK_SIZE, ++e) {
                    int row = idx / hd;
                    int col = idx % hd;
                    if (row < actual_Bc) {
                        dK_base[(kv_start + row) * hd + col] = dk_acc[e];
                        dV_base[(kv_start + row) * hd + col] = dv_acc[e];
                    }
                }
            });
    });
}

// =============================================================================
// Host launch helpers (public symbols, registered by oneapi_kernel_registry.cpp)
// =============================================================================

// Returns (output [BH, Sq, Hd], lse [BH, Sq] in Float32) — same contract as
// `flash_attention_kernel_with_lse` but Float64 inputs / output.
auto fused_attention_oneapi_f64(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    double scale,
    bool causal,
    sycl::queue& queue
) -> std::pair<Tensor, Tensor> {
    if (Q.dtype() != DType::Float64 || K.dtype() != DType::Float64 || V.dtype() != DType::Float64) {
        throw std::runtime_error("fused_attention_oneapi_f64: requires Float64 inputs");
    }
    const int64_t batch_heads = Q.shape()[0];
    const int64_t seq_len_q   = Q.shape()[1];
    const int64_t head_dim    = Q.shape()[2];
    const int64_t seq_len_k   = K.shape()[1];

    Tensor output = make_zeros_f64({batch_heads, seq_len_q, head_dim},
                                    DType::Float64, Q.device(), queue);
    Tensor lse    = make_zeros_f64({batch_heads, seq_len_q},
                                    DType::Float32, Q.device(), queue);

    const double* q_ptr = get_dp<const double>(Q);
    const double* k_ptr = get_dp<const double>(K);
    const double* v_ptr = get_dp<const double>(V);
    double*       o_ptr = get_dp<double>(output);
    float*        l_ptr = get_dp<float>(lse);

    switch (head_dim) {
        case 16:  launch_flash_attention_f64_forward<16> (q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, batch_heads, seq_len_q, seq_len_k, scale, causal, queue); break;
        case 32:  launch_flash_attention_f64_forward<32> (q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, batch_heads, seq_len_q, seq_len_k, scale, causal, queue); break;
        case 48:  launch_flash_attention_f64_forward<48> (q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, batch_heads, seq_len_q, seq_len_k, scale, causal, queue); break;
        case 64:  launch_flash_attention_f64_forward<64> (q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, batch_heads, seq_len_q, seq_len_k, scale, causal, queue); break;
        case 80:  launch_flash_attention_f64_forward<80> (q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, batch_heads, seq_len_q, seq_len_k, scale, causal, queue); break;
        case 96:  launch_flash_attention_f64_forward<96> (q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, batch_heads, seq_len_q, seq_len_k, scale, causal, queue); break;
        case 128: launch_flash_attention_f64_forward<128>(q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, batch_heads, seq_len_q, seq_len_k, scale, causal, queue); break;
        default:
            throw std::runtime_error(
                "fused_attention_oneapi_f64: Unsupported head_dim " +
                std::to_string(head_dim) +
                ". Supported: {16, 32, 48, 64, 80, 96, 128}.");
    }
    queue.wait_and_throw();
    return {output, lse};
}

// Returns {dQ, dK, dV} (all Float64).
auto flash_attention_backward_oneapi_f64(
    const Tensor& dO,
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    const Tensor& O,
    double scale,
    bool causal,
    sycl::queue& queue
) -> std::vector<Tensor> {
    if (Q.dtype() != DType::Float64) {
        throw std::runtime_error("flash_attention_backward_oneapi_f64: requires Float64 inputs");
    }
    const int64_t batch_heads = Q.shape()[0];
    const int64_t seq_len     = Q.shape()[1];
    const int64_t head_dim    = Q.shape()[2];

    Tensor dQ = make_zeros_f64({batch_heads, seq_len, head_dim}, DType::Float64, Q.device(), queue);
    Tensor dK = make_zeros_f64({batch_heads, seq_len, head_dim}, DType::Float64, K.device(), queue);
    Tensor dV = make_zeros_f64({batch_heads, seq_len, head_dim}, DType::Float64, V.device(), queue);

    const double* q_ptr  = get_dp<const double>(Q);
    const double* k_ptr  = get_dp<const double>(K);
    const double* v_ptr  = get_dp<const double>(V);
    const double* o_ptr  = get_dp<const double>(O);
    const double* do_ptr = get_dp<const double>(dO);
    double* dq_ptr = get_dp<double>(dQ);
    double* dk_ptr = get_dp<double>(dK);
    double* dv_ptr = get_dp<double>(dV);

    switch (head_dim) {
        case 16:  launch_flash_attention_f64_backward<16> (q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr, batch_heads, seq_len, scale, causal, queue); break;
        case 32:  launch_flash_attention_f64_backward<32> (q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr, batch_heads, seq_len, scale, causal, queue); break;
        case 48:  launch_flash_attention_f64_backward<48> (q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr, batch_heads, seq_len, scale, causal, queue); break;
        case 64:  launch_flash_attention_f64_backward<64> (q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr, batch_heads, seq_len, scale, causal, queue); break;
        case 80:  launch_flash_attention_f64_backward<80> (q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr, batch_heads, seq_len, scale, causal, queue); break;
        case 96:  launch_flash_attention_f64_backward<96> (q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr, batch_heads, seq_len, scale, causal, queue); break;
        case 128: launch_flash_attention_f64_backward<128>(q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr, batch_heads, seq_len, scale, causal, queue); break;
        default:
            throw std::runtime_error(
                "flash_attention_backward_oneapi_f64: Unsupported head_dim " +
                std::to_string(head_dim) +
                ". Supported: {16, 32, 48, 64, 80, 96, 128}.");
    }
    queue.wait_and_throw();
    return {dQ, dK, dV};
}

}  // namespace oneapi
}  // namespace tenzor
