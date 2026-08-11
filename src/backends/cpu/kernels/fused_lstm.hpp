/**
 * @file fused_lstm.hpp
 * @brief Optimized fused LSTM operations
 *
 * Key features:
 * - Fused gate computations (GEMM + activations)
 * - SIMD-accelerated sigmoid and tanh
 * - Efficient memory access patterns
 * - OpenMP parallelization for batch dimension
 *
 * LSTM equations:
 *   i_t = sigmoid(W_ii @ x_t + W_hi @ h_{t-1} + b_i)
 *   f_t = sigmoid(W_if @ x_t + W_hf @ h_{t-1} + b_f)
 *   g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)
 *   o_t = sigmoid(W_io @ x_t + W_ho @ h_{t-1} + b_o)
 *   c_t = f_t * c_{t-1} + i_t * g_t
 *   h_t = o_t * tanh(c_t)
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <atomic>
#include <optional>
#include <stdexcept>
#include <string>

#include "buffer_pool.hpp"
#include "tenzor/utils/log.hpp"
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #if defined(__AVX512F__)
        #define TENZOR_LSTM_AVX512
    #endif
    #if defined(__AVX2__)
        #define TENZOR_LSTM_AVX2
    #endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// Include MKL header for threading control (even when using oneDNN for GEMM)
#ifdef TENZOR_USE_MKL
#include <mkl.h>
#endif

// LSTM GEMM Strategy:
// MKL's cblas_sgemm is highly optimized and provides best LSTM performance.
// We prefer MKL when available, with oneDNN as fallback.
// Note: The original MKL/ROCm interaction issue was caused by single-threaded MKL
// forcing, which we've removed. Multi-threaded MKL works correctly.

#ifdef TENZOR_USE_MKL
// Use MKL's optimized BLAS - best performance for LSTM
using gemm_int = MKL_INT;
#define TENZOR_LSTM_USE_MKL_GEMM 1
#else
// No MKL - check for oneDNN or fall back to generic CBLAS
#ifdef TENZOR_USE_ONEDNN
#include <dnnl.hpp>
#include "onednn_cache.hpp"
#define TENZOR_LSTM_USE_ONEDNN_GEMM 1
// oneDNN takes int64_t dimensions natively, so there is no truncation risk on
// this path. Define gemm_int = int64_t so the shared check_gemm_dims guard
// compiles and behaves uniformly across all GEMM backends.
using gemm_int = int64_t;

// Use shared per-thread oneDNN engine/stream from onednn_cache.hpp
// (eliminates a duplicate engine per thread).
namespace {

inline void onednn_sgemm_nt(
    int64_t M, int64_t N, int64_t K,
    float alpha, const float* A, int64_t lda,
    const float* B, int64_t ldb,
    float beta, float* C, int64_t ldc
) {
    // Audit item C.2: honour alpha and beta via oneDNN post-ops rather
    // than silently ignoring them.  Equivalent to BLAS sgemm:
    //   C = alpha * (A @ B^T) + beta * C
    // - alpha: eltwise_linear post-op with (alpha, 0)
    // - beta : sum post-op with scale=beta (reads the current C value
    //          before the write).
    // The order matters: append the sum first so beta*C is added BEFORE
    // alpha is applied; then eltwise_linear scales the combined result.
    // We instead append eltwise_linear FIRST (scales matmul output) and
    // sum AFTER with scale=beta (adds beta*C to the scaled product).
    // That yields  C = alpha * (A @ B^T) + beta * C  — the BLAS contract.
    auto& engine = tenzor::cpu::get_onednn_engine();
    auto& stream = tenzor::cpu::get_onednn_stream();

    try {
        dnnl::memory::dims a_dims = {M, K};
        dnnl::memory::dims bt_dims = {K, N};
        dnnl::memory::dims c_dims = {M, N};

        dnnl::memory::desc a_md(a_dims, dnnl::memory::data_type::f32,
                                dnnl::memory::dims{lda, 1});
        dnnl::memory::desc bt_md(bt_dims, dnnl::memory::data_type::f32,
                                 dnnl::memory::dims{1, ldb});
        dnnl::memory::desc c_md(c_dims, dnnl::memory::data_type::f32,
                                dnnl::memory::dims{ldc, 1});

        dnnl::primitive_attr attr;
        const bool need_alpha = (alpha != 1.0f);
        const bool need_beta  = (beta  != 0.0f);
        if (need_alpha || need_beta) {
            dnnl::post_ops po;
            if (need_alpha) {
                // eltwise_linear(alpha, 0) ⇒ y = alpha * x
                po.append_eltwise(dnnl::algorithm::eltwise_linear,
                                  /*alpha=*/alpha, /*beta=*/0.0f);
            }
            if (need_beta) {
                // sum(scale=beta) ⇒ y += beta * existing(C)
                po.append_sum(beta);
            }
            attr.set_post_ops(po);
        }
        auto matmul_pd = dnnl::matmul::primitive_desc(
            engine, a_md, bt_md, c_md, attr);
        auto matmul_prim = dnnl::matmul(matmul_pd);

        auto a_mem = dnnl::memory(a_md, engine, const_cast<float*>(A));
        auto bt_mem = dnnl::memory(bt_md, engine, const_cast<float*>(B));
        auto c_mem = dnnl::memory(c_md, engine, C);

        matmul_prim.execute(stream, {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, bt_mem},
            {DNNL_ARG_DST, c_mem}
        });
        stream.wait();
    } catch (const dnnl::error& e) {
        // TENZOR_STRICT_BACKEND=1 promotes oneDNN failures to hard errors so
        // silent fallbacks are auditable. Otherwise we log a WARN and fall
        // back to the scalar matmul below.
        if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
            throw std::runtime_error(
                std::string("[LSTM] oneDNN matmul failed (TENZOR_STRICT_BACKEND=1): ") +
                e.what());
        }
        TENZOR_LOG_WARN("[LSTM] oneDNN matmul failed ({}); using scalar fallback", e.what());
        // Scalar fallback honours alpha and beta (audit item C.2).
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * lda + k] * B[j * ldb + k];
                }
                const float prev = (beta != 0.0f) ? C[i * ldc + j] : 0.0f;
                C[i * ldc + j] = alpha * sum + beta * prev;
            }
        }
    }
}
} // anonymous namespace
#else
// Generic CBLAS fallback
extern "C" {
    void cblas_sgemm(int, int, int, int, int, int,
                     float, const float*, int, const float*, int,
                     float, float*, int);
    #define CblasRowMajor 101
    #define CblasNoTrans 111
    #define CblasTrans 112
}
using gemm_int = int;
#define TENZOR_LSTM_USE_GENERIC_GEMM 1
#endif
#endif

// GEMM dimension guard.
//
// In the default MKL LP64 build, gemm_int == MKL_INT == 32-bit. The fused
// LSTM/GRU input-projection GEMM passes M = seq_len * batch, which is computed
// in int64_t. For pathological-but-reachable shapes (seq_len * batch, or any of
// M/N/K/lda/ldb/ldc) exceeding the range of gemm_int, the static_cast<gemm_int>
// in LSTM_SGEMM_NT would silently truncate the row/col/stride count while the
// underlying buffers are sized in size_t/int64 — producing a partial/wrong GEMM
// with no diagnostic. Validate every dimension fits in gemm_int and hard-error
// otherwise (the only correct option: a 32-bit BLAS cannot express the op, and
// silently computing the wrong thing is worse than failing loudly).
namespace tenzor {
namespace cpu {
namespace lstm {
namespace detail {

inline void check_gemm_dims(int64_t M, int64_t N, int64_t K,
                            int64_t lda, int64_t ldb, int64_t ldc) {
    constexpr int64_t kMax = static_cast<int64_t>(std::numeric_limits<gemm_int>::max());
    const bool ok =
        M >= 0 && M <= kMax &&
        N >= 0 && N <= kMax &&
        K >= 0 && K <= kMax &&
        lda >= 0 && lda <= kMax &&
        ldb >= 0 && ldb <= kMax &&
        ldc >= 0 && ldc <= kMax;
    if (!ok) {
        throw std::runtime_error(
            std::string("[LSTM] GEMM dimension exceeds BLAS integer range (") +
            "max=" + std::to_string(kMax) + "): " +
            "M=" + std::to_string(M) +
            " N=" + std::to_string(N) +
            " K=" + std::to_string(K) +
            " lda=" + std::to_string(lda) +
            " ldb=" + std::to_string(ldb) +
            " ldc=" + std::to_string(ldc) +
            ". Rebuild against an ILP64 (64-bit integer) BLAS for shapes this large.");
    }
}

} // namespace detail
} // namespace lstm
} // namespace cpu
} // namespace tenzor

// LSTM GEMM macro - selects best available implementation.
// Every variant first validates that all dimensions fit in gemm_int (see
// detail::check_gemm_dims) so the static_cast below cannot silently truncate.
#ifdef TENZOR_LSTM_USE_MKL_GEMM
// MKL: Best performance
#define LSTM_SGEMM_NT(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) \
    (::tenzor::cpu::lstm::detail::check_gemm_dims((M), (N), (K), (lda), (ldb), (ldc)), \
     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, \
                static_cast<gemm_int>(M), static_cast<gemm_int>(N), static_cast<gemm_int>(K), \
                alpha, A, static_cast<gemm_int>(lda), \
                B, static_cast<gemm_int>(ldb), \
                beta, C, static_cast<gemm_int>(ldc)))
#elif defined(TENZOR_LSTM_USE_ONEDNN_GEMM)
// oneDNN fallback (uses int64_t throughout; check is a cheap, harmless guard
// that also documents the contract and keeps behaviour uniform across builds).
#define LSTM_SGEMM_NT(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) \
    (::tenzor::cpu::lstm::detail::check_gemm_dims((M), (N), (K), (lda), (ldb), (ldc)), \
     onednn_sgemm_nt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc))
#else
// Generic CBLAS fallback
#define LSTM_SGEMM_NT(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) \
    (::tenzor::cpu::lstm::detail::check_gemm_dims((M), (N), (K), (lda), (ldb), (ldc)), \
     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, \
                static_cast<gemm_int>(M), static_cast<gemm_int>(N), static_cast<gemm_int>(K), \
                alpha, A, static_cast<gemm_int>(lda), \
                B, static_cast<gemm_int>(ldb), \
                beta, C, static_cast<gemm_int>(ldc)))
#endif

namespace tenzor {
namespace cpu {
namespace lstm {

// ============================================================================
// SIMD Activation Functions
// ============================================================================

#ifdef TENZOR_LSTM_AVX2

/**
 * @brief SIMD sigmoid for the fused LSTM/GRU gate path.
 *
 * Precision note: the previous implementation used a coarse degree-5
 * polynomial exp approximation, which diverged from the cell-level reference
 * (lstm_cell_forward_kernel / gru_cell_forward_kernel, the Float64 full-sequence
 * paths, and PyTorch) — all of which use exact std::exp/std::tanh. Worse, the
 * scalar remainder loop of the SAME fused kernel uses exact sigmoid_scalar /
 * tanh_scalar, so different hidden lanes of one hidden state were computed with
 * different accuracy. These are hot but precision-sensitive paths, so we
 * evaluate the activations exactly per lane via the scalar reference
 * (std::exp / std::tanh). This makes the SIMD lanes bit-identical to the scalar
 * remainder and to the cross-backend cell reference, eliminating the divergence
 * that previously forced loosened gradcheck/parity tolerances.
 */
inline __m256 sigmoid_avx2(__m256 x) {
    alignas(32) float buf[8];
    _mm256_store_ps(buf, x);
    for (int i = 0; i < 8; ++i) {
        float v = std::max(-20.0f, std::min(20.0f, buf[i]));
        buf[i] = 1.0f / (1.0f + std::exp(-v));
    }
    return _mm256_load_ps(buf);
}

/**
 * @brief SIMD tanh for the fused LSTM/GRU gate path (exact, per-lane std::tanh).
 *        See sigmoid_avx2 for the precision rationale.
 */
inline __m256 tanh_avx2(__m256 x) {
    alignas(32) float buf[8];
    _mm256_store_ps(buf, x);
    for (int i = 0; i < 8; ++i) {
        buf[i] = std::tanh(buf[i]);
    }
    return _mm256_load_ps(buf);
}

#endif // TENZOR_LSTM_AVX2

#ifdef TENZOR_LSTM_AVX512

// Exact per-lane sigmoid (std::exp), bit-identical to sigmoid_scalar and the
// cell-level reference. See sigmoid_avx2 for the precision rationale.
inline __m512 sigmoid_avx512(__m512 x) {
    alignas(64) float buf[16];
    _mm512_store_ps(buf, x);
    for (int i = 0; i < 16; ++i) {
        float v = std::max(-20.0f, std::min(20.0f, buf[i]));
        buf[i] = 1.0f / (1.0f + std::exp(-v));
    }
    return _mm512_load_ps(buf);
}

// Exact per-lane tanh (std::tanh). See sigmoid_avx2 for the precision rationale.
inline __m512 tanh_avx512(__m512 x) {
    alignas(64) float buf[16];
    _mm512_store_ps(buf, x);
    for (int i = 0; i < 16; ++i) {
        buf[i] = std::tanh(buf[i]);
    }
    return _mm512_load_ps(buf);
}

#endif // TENZOR_LSTM_AVX512

// Scalar fallbacks
inline float sigmoid_scalar(float x) {
    x = std::max(-20.0f, std::min(20.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

inline float tanh_scalar(float x) {
    return std::tanh(x);
}

// ============================================================================
// LSTM Gate Computation
// ============================================================================

/**
 * @brief Compute LSTM gates for a single timestep with SIMD
 *
 * gates = W_ih @ x + W_hh @ h + bias
 * Then apply activations: i,f,o = sigmoid, g = tanh
 *
 * @param gates_ih Pre-computed W_ih @ x (batch, 4*hidden)
 * @param h Previous hidden state (batch, hidden)
 * @param c Previous cell state (batch, hidden)
 * @param W_hh Hidden-to-hidden weights (4*hidden, hidden)
 * @param bias Bias terms (4*hidden) or nullptr
 * @param h_out New hidden state output
 * @param c_out New cell state output
 * @param batch Batch size
 * @param hidden Hidden size
 */
inline void lstm_cell_fused(
    const float* gates_ih,  // (batch, 4*hidden) - pre-computed input gates
    const float* h,         // (batch, hidden) - previous hidden
    const float* c,         // (batch, hidden) - previous cell
    const float* W_hh,      // (4*hidden, hidden) - hidden weights
    const float* bias,      // (4*hidden) or nullptr
    float* h_out,           // (batch, hidden) - new hidden
    float* c_out,           // (batch, hidden) - new cell
    float* workspace,       // (batch, 4*hidden) - temp buffer for gates_hh
    int64_t batch,
    int64_t hidden,
    float* i_reserve = nullptr,      // (batch, hidden) post-activation i, or nullptr
    float* f_reserve = nullptr,      // (batch, hidden) post-activation f, or nullptr
    float* g_reserve = nullptr,      // (batch, hidden) post-activation g, or nullptr
    float* o_reserve = nullptr,      // (batch, hidden) post-activation o, or nullptr
    float* tanh_c_reserve = nullptr  // (batch, hidden) tanh(c_out), or nullptr
) {
    int64_t gate_size = 4 * hidden;

    // Step 1: Compute gates_hh = h @ W_hh^T using BLAS
    // W_hh is (4*hidden, hidden), h is (batch, hidden)
    // Result is (batch, 4*hidden)
    // Uses oneDNN when available to avoid MKL/ROCm interaction crashes
    LSTM_SGEMM_NT(batch, gate_size, hidden,
                  1.0f, h, hidden,
                  W_hh, hidden,
                  0.0f, workspace, gate_size);

    // Step 2: Fused gate computation with SIMD
    // gates = gates_ih + gates_hh + bias
    // i = sigmoid(gates[0:H]), f = sigmoid(gates[H:2H])
    // g = tanh(gates[2H:3H]), o = sigmoid(gates[3H:4H])
    // c_new = f * c + i * g
    // h_new = o * tanh(c_new)

    // NOTE: Avoid OMP here to prevent nested parallelism with MKL's internal threads
    for (int64_t b = 0; b < batch; ++b) {
        const float* g_ih = gates_ih + b * gate_size;
        const float* g_hh = workspace + b * gate_size;
        const float* c_prev = c + b * hidden;
        float* h_new = h_out + b * hidden;
        float* c_new = c_out + b * hidden;

        int64_t d = 0;

#ifdef TENZOR_LSTM_AVX512
        // Process 16 elements at a time with AVX512
        for (; d + 16 <= hidden; d += 16) {
            // Load pre-computed input gates
            __m512 i_gate = _mm512_loadu_ps(g_ih + d);
            __m512 f_gate = _mm512_loadu_ps(g_ih + hidden + d);
            __m512 g_gate = _mm512_loadu_ps(g_ih + 2 * hidden + d);
            __m512 o_gate = _mm512_loadu_ps(g_ih + 3 * hidden + d);

            // Add hidden-to-hidden gates
            i_gate = _mm512_add_ps(i_gate, _mm512_loadu_ps(g_hh + d));
            f_gate = _mm512_add_ps(f_gate, _mm512_loadu_ps(g_hh + hidden + d));
            g_gate = _mm512_add_ps(g_gate, _mm512_loadu_ps(g_hh + 2 * hidden + d));
            o_gate = _mm512_add_ps(o_gate, _mm512_loadu_ps(g_hh + 3 * hidden + d));

            // Add bias if provided
            if (bias) {
                i_gate = _mm512_add_ps(i_gate, _mm512_loadu_ps(bias + d));
                f_gate = _mm512_add_ps(f_gate, _mm512_loadu_ps(bias + hidden + d));
                g_gate = _mm512_add_ps(g_gate, _mm512_loadu_ps(bias + 2 * hidden + d));
                o_gate = _mm512_add_ps(o_gate, _mm512_loadu_ps(bias + 3 * hidden + d));
            }

            // Apply activations
            __m512 i = sigmoid_avx512(i_gate);
            __m512 f = sigmoid_avx512(f_gate);
            __m512 g = tanh_avx512(g_gate);
            __m512 o = sigmoid_avx512(o_gate);

            // Load previous states
            __m512 c_p = _mm512_loadu_ps(c_prev + d);

            // c_new = f * c_prev + i * g
            __m512 c_n = _mm512_fmadd_ps(f, c_p, _mm512_mul_ps(i, g));
            __m512 tanh_cn = tanh_avx512(c_n);

            // h_new = o * tanh(c_new)
            __m512 h_n = _mm512_mul_ps(o, tanh_cn);

            _mm512_storeu_ps(c_new + d, c_n);
            _mm512_storeu_ps(h_new + d, h_n);

            if (i_reserve) _mm512_storeu_ps(i_reserve + b * hidden + d, i);
            if (f_reserve) _mm512_storeu_ps(f_reserve + b * hidden + d, f);
            if (g_reserve) _mm512_storeu_ps(g_reserve + b * hidden + d, g);
            if (o_reserve) _mm512_storeu_ps(o_reserve + b * hidden + d, o);
            if (tanh_c_reserve) _mm512_storeu_ps(tanh_c_reserve + b * hidden + d, tanh_cn);
        }
#endif

#ifdef TENZOR_LSTM_AVX2
        // Process 8 elements at a time with AVX2
        for (; d + 8 <= hidden; d += 8) {
            // Load pre-computed input gates
            __m256 i_gate = _mm256_loadu_ps(g_ih + d);
            __m256 f_gate = _mm256_loadu_ps(g_ih + hidden + d);
            __m256 g_gate = _mm256_loadu_ps(g_ih + 2 * hidden + d);
            __m256 o_gate = _mm256_loadu_ps(g_ih + 3 * hidden + d);

            // Add hidden-to-hidden gates
            i_gate = _mm256_add_ps(i_gate, _mm256_loadu_ps(g_hh + d));
            f_gate = _mm256_add_ps(f_gate, _mm256_loadu_ps(g_hh + hidden + d));
            g_gate = _mm256_add_ps(g_gate, _mm256_loadu_ps(g_hh + 2 * hidden + d));
            o_gate = _mm256_add_ps(o_gate, _mm256_loadu_ps(g_hh + 3 * hidden + d));

            // Add bias if provided
            if (bias) {
                i_gate = _mm256_add_ps(i_gate, _mm256_loadu_ps(bias + d));
                f_gate = _mm256_add_ps(f_gate, _mm256_loadu_ps(bias + hidden + d));
                g_gate = _mm256_add_ps(g_gate, _mm256_loadu_ps(bias + 2 * hidden + d));
                o_gate = _mm256_add_ps(o_gate, _mm256_loadu_ps(bias + 3 * hidden + d));
            }

            // Apply activations
            __m256 i = sigmoid_avx2(i_gate);
            __m256 f = sigmoid_avx2(f_gate);
            __m256 g = tanh_avx2(g_gate);
            __m256 o = sigmoid_avx2(o_gate);

            // Load previous states
            __m256 c_p = _mm256_loadu_ps(c_prev + d);

            // c_new = f * c_prev + i * g
            __m256 c_n = _mm256_fmadd_ps(f, c_p, _mm256_mul_ps(i, g));
            __m256 tanh_cn = tanh_avx2(c_n);

            // h_new = o * tanh(c_new)
            __m256 h_n = _mm256_mul_ps(o, tanh_cn);

            _mm256_storeu_ps(c_new + d, c_n);
            _mm256_storeu_ps(h_new + d, h_n);

            if (i_reserve) _mm256_storeu_ps(i_reserve + b * hidden + d, i);
            if (f_reserve) _mm256_storeu_ps(f_reserve + b * hidden + d, f);
            if (g_reserve) _mm256_storeu_ps(g_reserve + b * hidden + d, g);
            if (o_reserve) _mm256_storeu_ps(o_reserve + b * hidden + d, o);
            if (tanh_c_reserve) _mm256_storeu_ps(tanh_c_reserve + b * hidden + d, tanh_cn);
        }
#endif

        // Scalar fallback for remaining elements
        for (; d < hidden; ++d) {
            float i_gate = g_ih[d] + g_hh[d];
            float f_gate = g_ih[hidden + d] + g_hh[hidden + d];
            float g_gate = g_ih[2 * hidden + d] + g_hh[2 * hidden + d];
            float o_gate = g_ih[3 * hidden + d] + g_hh[3 * hidden + d];

            if (bias) {
                i_gate += bias[d];
                f_gate += bias[hidden + d];
                g_gate += bias[2 * hidden + d];
                o_gate += bias[3 * hidden + d];
            }

            float i = sigmoid_scalar(i_gate);
            float f = sigmoid_scalar(f_gate);
            float g = tanh_scalar(g_gate);
            float o = sigmoid_scalar(o_gate);

            float c_n = f * c_prev[d] + i * g;
            float tanh_cn = tanh_scalar(c_n);
            float h_n = o * tanh_cn;

            c_new[d] = c_n;
            h_new[d] = h_n;

            if (i_reserve) i_reserve[b * hidden + d] = i;
            if (f_reserve) f_reserve[b * hidden + d] = f;
            if (g_reserve) g_reserve[b * hidden + d] = g;
            if (o_reserve) o_reserve[b * hidden + d] = o;
            if (tanh_c_reserve) tanh_c_reserve[b * hidden + d] = tanh_cn;
        }
    }
}

// ============================================================================
// Full LSTM Forward Pass
// ============================================================================

/**
 * @brief Fused LSTM forward pass for entire sequence
 *
 * Processes all timesteps efficiently with:
 * 1. Batched input transformation (one GEMM for all timesteps)
 * 2. Fused gate computations with SIMD
 * 3. Efficient memory reuse
 *
 * @param input Input sequence (seq_len, batch, input_size)
 * @param W_ih Input-to-hidden weights (4*hidden, input_size)
 * @param W_hh Hidden-to-hidden weights (4*hidden, hidden)
 * @param bias Bias terms (4*hidden) or nullptr
 * @param h0 Initial hidden state (batch, hidden)
 * @param c0 Initial cell state (batch, hidden)
 * @param output Output sequence (seq_len, batch, hidden)
 * @param h_n Final hidden state (batch, hidden)
 * @param c_n Final cell state (batch, hidden)
 */
inline void lstm_forward(
    const float* input,     // (seq_len, batch, input_size)
    const float* W_ih,      // (4*hidden, input_size)
    const float* W_hh,      // (4*hidden, hidden)
    const float* bias,      // (4*hidden) or nullptr
    const float* h0,        // (batch, hidden)
    const float* c0,        // (batch, hidden)
    float* output,          // (seq_len, batch, hidden)
    float* h_n,             // (batch, hidden)
    float* c_n,             // (batch, hidden)
    int64_t seq_len,
    int64_t batch,
    int64_t input_size,
    int64_t hidden
) {
    int64_t gate_size = 4 * hidden;

    // Validate input pointers
    if (!input || !W_ih || !W_hh || !h0 || !c0 || !output || !h_n || !c_n) {
        fprintf(stderr, "[LSTM] ERROR: NULL pointer detected!\n");
        fprintf(stderr, "  input=%p W_ih=%p W_hh=%p h0=%p c0=%p\n",
                (void*)input, (void*)W_ih, (void*)W_hh, (void*)h0, (void*)c0);
        fprintf(stderr, "  output=%p h_n=%p c_n=%p\n",
                (void*)output, (void*)h_n, (void*)c_n);
        return;
    }

    // Memory fence to ensure all prior operations are complete
    std::atomic_thread_fence(std::memory_order_seq_cst);

    // Validate memory is accessible by reading from each buffer.
    // This forces page faults before MKL threads access the memory.
    // Guard every read on a positive element count: a representable empty
    // tensor (any dimension == 0) would otherwise index [-1] on a zero-length
    // buffer (negative OOB read / UB).
    const int64_t input_count = seq_len * batch * input_size;
    const int64_t wih_count   = gate_size * input_size;
    const int64_t whh_count   = gate_size * hidden;
    const int64_t state_count = batch * hidden;
    if (input_count > 0) {
        volatile float check_input = input[0];
        volatile float check_input_last = input[input_count - 1];
        (void)check_input; (void)check_input_last;
    }
    if (wih_count > 0) {
        volatile float check_wih = W_ih[0];
        volatile float check_wih_last = W_ih[wih_count - 1];
        (void)check_wih; (void)check_wih_last;
    }
    if (whh_count > 0) {
        volatile float check_whh = W_hh[0];
        volatile float check_whh_last = W_hh[whh_count - 1];
        (void)check_whh; (void)check_whh_last;
    }
    if (state_count > 0) {
        volatile float check_h0 = h0[0];
        volatile float check_h0_last = h0[state_count - 1];
        volatile float check_c0 = c0[0];
        volatile float check_c0_last = c0[state_count - 1];
        (void)check_h0; (void)check_h0_last;
        (void)check_c0; (void)check_c0_last;
    }

    // Thread-local cached buffers for LSTM workspace
    // This avoids repeated allocation overhead in the hot path
    struct LSTMWorkspace {
        std::vector<float> gates_ih;
        std::vector<float> gates_hh;
        std::vector<float> h_curr;
        std::vector<float> c_curr;
    };
    static thread_local LSTMWorkspace workspace;

    // Resize buffers if needed (only reallocates if size increases)
    size_t gates_ih_count = seq_len * batch * gate_size;
    size_t gates_hh_count = batch * gate_size;
    size_t h_temp_count = batch * hidden;
    size_t c_temp_count = batch * hidden;

    if (workspace.gates_ih.size() < gates_ih_count) workspace.gates_ih.resize(gates_ih_count);
    if (workspace.gates_hh.size() < gates_hh_count) workspace.gates_hh.resize(gates_hh_count);
    if (workspace.h_curr.size() < h_temp_count) workspace.h_curr.resize(h_temp_count);
    if (workspace.c_curr.size() < c_temp_count) workspace.c_curr.resize(c_temp_count);

    float* gates_ih = workspace.gates_ih.data();
    float* gates_hh = workspace.gates_hh.data();
    float* h_curr = workspace.h_curr.data();
    float* c_curr = workspace.c_curr.data();

    // Use MKL's default threading (multi-threaded) for best GEMM performance
    // The GEMM operations dominate LSTM computation, so we want full parallelism
#ifdef TENZOR_USE_MKL
    // Ensure MKL uses optimal thread count (don't restrict to 1 thread)
    int mkl_threads = mkl_get_max_threads();
    (void)mkl_threads;  // Use default - don't override
#endif

    // Step 1: Pre-compute ALL input gates in one batched GEMM
    // Uses oneDNN when available to avoid MKL/ROCm interaction crashes
    LSTM_SGEMM_NT(seq_len * batch, gate_size, input_size,
                  1.0f, input, input_size,
                  W_ih, input_size,
                  0.0f, gates_ih, gate_size);

    // Initialize current states from h0, c0
    std::memcpy(h_curr, h0, batch * hidden * sizeof(float));
    std::memcpy(c_curr, c0, batch * hidden * sizeof(float));

    // Step 2: Process each timestep
    for (int64_t t = 0; t < seq_len; ++t) {
        const float* gates_ih_t = gates_ih + t * batch * gate_size;
        float* output_t = output + t * batch * hidden;

        lstm_cell_fused(
            gates_ih_t, h_curr, c_curr, W_hh, bias,
            output_t, c_curr, gates_hh, batch, hidden
        );

        std::memcpy(h_curr, output_t, batch * hidden * sizeof(float));
    }

    // Copy final states
    std::memcpy(h_n, h_curr, batch * hidden * sizeof(float));
    std::memcpy(c_n, c_curr, batch * hidden * sizeof(float));

    // Workspace buffers are cached in thread-local storage, no cleanup needed
}

/**
 * @brief Fused LSTM forward pass for entire sequence, TRAINING variant.
 *
 * Identical computation to lstm_forward (same batched input-gate GEMM, same
 * per-timestep lstm_cell_fused call), but additionally captures the
 * per-timestep post-activation gates and cell states needed by
 * lstm_backward_training for analytic BPTT.
 */
inline void lstm_forward_training(
    const float* input,     // (seq_len, batch, input_size)
    const float* W_ih,      // (4*hidden, input_size)
    const float* W_hh,      // (4*hidden, hidden)
    const float* bias,      // (4*hidden) or nullptr
    const float* h0,        // (batch, hidden)
    const float* c0,        // (batch, hidden)
    float* output,          // (seq_len, batch, hidden)
    float* h_n,             // (batch, hidden)
    float* c_n,             // (batch, hidden)
    float* gates_reserve,      // (4, seq_len, batch, hidden): i,f,g,o back-to-back planes
    float* cell_reserve,       // (seq_len, batch, hidden): c_t per timestep
    float* tanh_cell_reserve,  // (seq_len, batch, hidden): tanh(c_t) per timestep
    int64_t seq_len,
    int64_t batch,
    int64_t input_size,
    int64_t hidden
) {
    int64_t gate_size = 4 * hidden;

    std::vector<float> gates_ih(static_cast<size_t>(seq_len * batch * gate_size));
    std::vector<float> gates_hh_workspace(static_cast<size_t>(batch * gate_size));
    std::vector<float> h_curr(static_cast<size_t>(batch * hidden));
    std::vector<float> c_curr(static_cast<size_t>(batch * hidden));

    LSTM_SGEMM_NT(seq_len * batch, gate_size, input_size,
                  1.0f, input, input_size,
                  W_ih, input_size,
                  0.0f, gates_ih.data(), gate_size);

    std::memcpy(h_curr.data(), h0, batch * hidden * sizeof(float));
    std::memcpy(c_curr.data(), c0, batch * hidden * sizeof(float));

    // gates_reserve layout: 4 back-to-back (seq_len, batch, hidden) planes,
    // one per gate, in i,f,g,o order — NOT interleaved per-timestep. This
    // matches how lstm_cell_fused indexes each reserve pointer independently
    // as base + b*hidden + d; a column-sliced pointer into an interleaved
    // (batch, 4*hidden) buffer would read/write the wrong batch rows for
    // b >= 1, since the cell's per-batch stride is `hidden`, not `4*hidden`.
    const int64_t plane = seq_len * batch * hidden;
    float* i_plane = gates_reserve;
    float* f_plane = gates_reserve + plane;
    float* g_plane = gates_reserve + 2 * plane;
    float* o_plane = gates_reserve + 3 * plane;

    for (int64_t t = 0; t < seq_len; ++t) {
        const float* gates_ih_t = gates_ih.data() + t * batch * gate_size;
        float* output_t = output + t * batch * hidden;
        float* cell_reserve_t = cell_reserve + t * batch * hidden;
        float* tanh_cell_reserve_t = tanh_cell_reserve + t * batch * hidden;

        lstm_cell_fused(
            gates_ih_t, h_curr.data(), c_curr.data(), W_hh, bias,
            output_t, cell_reserve_t, gates_hh_workspace.data(), batch, hidden,
            /*i_reserve=*/i_plane + t * batch * hidden,
            /*f_reserve=*/f_plane + t * batch * hidden,
            /*g_reserve=*/g_plane + t * batch * hidden,
            /*o_reserve=*/o_plane + t * batch * hidden,
            /*tanh_c_reserve=*/tanh_cell_reserve_t);

        std::memcpy(c_curr.data(), cell_reserve_t, batch * hidden * sizeof(float));
        std::memcpy(h_curr.data(), output_t, batch * hidden * sizeof(float));
    }

    std::memcpy(h_n, h_curr.data(), batch * hidden * sizeof(float));
    std::memcpy(c_n, c_curr.data(), batch * hidden * sizeof(float));
}

/**
 * @brief Analytic LSTM BPTT backward for the entire sequence, TRAINING variant.
 *
 * Two phases:
 *  1. Sequential (unavoidable): walk t = seq_len-1 .. 0, using the reserve
 *     buffers from lstm_forward_training to compute each timestep's
 *     pre-activation gate gradients (d_gates_all), propagating dh/dc through
 *     W_hh. Same per-step cost as the forward recurrence.
 *  2. Batched (parallel): once d_gates_all is fully populated, grad_W_ih,
 *     grad_W_hh, grad_bias, and grad_input are each one large GEMM/reduction
 *     across all timesteps instead of per-timestep ones.
 *
 * Gate order throughout is i, f, g, o (see file header) — the same order
 * lstm_forward_training's reserve buffers use, so no format translation.
 *
 * Forward reference (per timestep t, batch row b, unit d):
 *   pre = W_ih @ x_t + W_hh @ h_{t-1} + bias   (row layout [i|f|g|o], 4*hidden)
 *   i,f,o = sigmoid(pre), g = tanh(pre)
 *   c_t = f*c_{t-1} + i*g,  h_t = o*tanh(c_t)
 * so with dh_t = dL/dh_t (external grad_output plus the recurrent term):
 *   dc_t   += dh_t * o * (1 - tanh(c_t)^2)
 *   di      = dc_t * g,        df = dc_t * c_{t-1}
 *   dg      = dc_t * i,        do = dh_t * tanh(c_t)
 *   d_pre_i = di * i*(1-i),    d_pre_f = df * f*(1-f)
 *   d_pre_g = dg * (1 - g^2),  d_pre_o = do * o*(1-o)
 *   dc_{t-1} = dc_t * f,       dh_{t-1} = d_pre @ W_hh
 */
inline void lstm_backward_training(
    const float* grad_output,   // (seq_len, batch, hidden): dL/d(output_t) for every t
    const float* grad_cy,       // (batch, hidden): dL/d(c_n)
    const float* input,         // (seq_len, batch, input_size)
    const float* h0,            // (batch, hidden)
    const float* c0,            // (batch, hidden)
    const float* W_ih,          // (4*hidden, input_size)
    const float* W_hh,          // (4*hidden, hidden)
    const float* gates_reserve,      // (4, seq_len, batch, hidden) i,f,g,o planes (see lstm_forward_training)
    const float* cell_reserve,       // (seq_len, batch, hidden)
    const float* tanh_cell_reserve,  // (seq_len, batch, hidden)
    float* grad_input,   // (seq_len, batch, input_size) OUT
    float* grad_h0,      // (batch, hidden) OUT
    float* grad_c0,      // (batch, hidden) OUT
    float* grad_W_ih,    // (4*hidden, input_size) OUT
    float* grad_W_hh,    // (4*hidden, hidden) OUT
    float* grad_bias,    // (4*hidden) OUT (shared by b_ih and b_hh by the caller)
    int64_t seq_len,
    int64_t batch,
    int64_t input_size,
    int64_t hidden
) {
    const int64_t gate_size = 4 * hidden;
    const int64_t plane = seq_len * batch * hidden;
    const float* i_plane = gates_reserve;
    const float* f_plane = gates_reserve + plane;
    const float* g_plane = gates_reserve + 2 * plane;
    const float* o_plane = gates_reserve + 3 * plane;

    // Phase 1: sequential BPTT, accumulating d_gates_all (seq_len, batch, 4*hidden)
    // in the SAME i,f,g,o column layout LSTM_SGEMM_NT/Linear expect (matches
    // LSTMCell's gates layout, so grad_W_ih/grad_W_hh below need no reordering).
    std::vector<float> d_gates_all(static_cast<size_t>(seq_len * batch * gate_size));
    std::vector<float> dh_next(static_cast<size_t>(batch * hidden), 0.0f);
    std::vector<float> dc_next(static_cast<size_t>(batch * hidden));
    std::memcpy(dc_next.data(), grad_cy, batch * hidden * sizeof(float));

    for (int64_t t = seq_len - 1; t >= 0; --t) {
        const float* i_t = i_plane + t * batch * hidden;
        const float* f_t = f_plane + t * batch * hidden;
        const float* g_t = g_plane + t * batch * hidden;
        const float* o_t = o_plane + t * batch * hidden;
        const float* tanh_c_t = tanh_cell_reserve + t * batch * hidden;
        const float* c_prev_t = (t == 0) ? c0 : (cell_reserve + (t - 1) * batch * hidden);
        const float* grad_out_t = grad_output + t * batch * hidden;
        float* d_gates_t = d_gates_all.data() + t * batch * gate_size;

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t d = 0; d < hidden; ++d) {
                int64_t bh = b * hidden + d;
                float dh_t = grad_out_t[bh] + dh_next[bh];
                float i = i_t[bh], f = f_t[bh], g = g_t[bh], o = o_t[bh];
                float tanh_c = tanh_c_t[bh];
                float dc_t = dc_next[bh] + dh_t * o * (1.0f - tanh_c * tanh_c);

                float di = dc_t * g;
                float df = dc_t * c_prev_t[bh];
                float dg = dc_t * i;
                float do_ = dh_t * tanh_c;

                float d_i_pre = di * i * (1.0f - i);
                float d_f_pre = df * f * (1.0f - f);
                float d_g_pre = dg * (1.0f - g * g);
                float d_o_pre = do_ * o * (1.0f - o);

                int64_t base = b * gate_size;
                d_gates_t[base + d] = d_i_pre;
                d_gates_t[base + hidden + d] = d_f_pre;
                d_gates_t[base + 2 * hidden + d] = d_g_pre;
                d_gates_t[base + 3 * hidden + d] = d_o_pre;

                dc_next[bh] = dc_t * f;  // feeds t-1 (or becomes grad_c0 at t==0)
            }
        }

        // dh_from_this_timestep = d_gates_t @ W_hh  -> (batch, hidden), feeds t-1
        // (or becomes grad_h0 at t==0). W_hh is (4*hidden, hidden); this is a
        // plain A @ B (no transpose) with A=(batch,4H), B=(4H,hidden).
        detail::check_gemm_dims(batch, hidden, gate_size, gate_size, hidden, hidden);
#ifdef TENZOR_LSTM_USE_MKL_GEMM
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<gemm_int>(batch), static_cast<gemm_int>(hidden), static_cast<gemm_int>(gate_size),
                    1.0f, d_gates_t, static_cast<gemm_int>(gate_size),
                    W_hh, static_cast<gemm_int>(hidden),
                    0.0f, dh_next.data(), static_cast<gemm_int>(hidden));
#else
        // PERF: no oneDNN/generic-CBLAS NN/TN GEMM macro exists yet for this
        // orientation (only LSTM_SGEMM_NT does); this scalar fallback is
        // numerically correct but O(N) slower than the MKL path. On non-MKL
        // builds this may reduce or eliminate the fused path's speed
        // advantage over the per-timestep loop it replaces. Adding
        // LSTM_SGEMM_NN/LSTM_SGEMM_TN macros (mirroring LSTM_SGEMM_NT's
        // three-backend-variant structure) is tracked as follow-up work, not
        // required for correctness.
        std::fill(dh_next.begin(), dh_next.end(), 0.0f);
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t k = 0; k < gate_size; ++k) {
                float dgv = d_gates_t[b * gate_size + k];
                if (dgv == 0.0f) continue;
                for (int64_t d = 0; d < hidden; ++d) {
                    dh_next[b * hidden + d] += dgv * W_hh[k * hidden + d];
                }
            }
        }
#endif
    }

    std::memcpy(grad_h0, dh_next.data(), batch * hidden * sizeof(float));
    std::memcpy(grad_c0, dc_next.data(), batch * hidden * sizeof(float));

    // Phase 2: batched reductions across all timesteps.
    const int64_t total_rows = seq_len * batch;

    // grad_bias = column_sum(d_gates_all) over all (seq*batch) rows.
    std::fill(grad_bias, grad_bias + gate_size, 0.0f);
    for (int64_t r = 0; r < total_rows; ++r) {
        const float* row = d_gates_all.data() + r * gate_size;
        for (int64_t k = 0; k < gate_size; ++k) grad_bias[k] += row[k];
    }

    // grad_W_ih = d_gates_all^T @ input_all -> (4*hidden, input_size)
    // grad_input = d_gates_all   @ W_ih     -> (seq*batch, input_size)
    // (both use the flat (seq*batch, ...) row view of the sequence tensors,
    //  since the input projection is timestep-independent.)
    detail::check_gemm_dims(gate_size, input_size, total_rows, gate_size, input_size, input_size);
#ifdef TENZOR_LSTM_USE_MKL_GEMM
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                static_cast<gemm_int>(gate_size), static_cast<gemm_int>(input_size), static_cast<gemm_int>(total_rows),
                1.0f, d_gates_all.data(), static_cast<gemm_int>(gate_size),
                input, static_cast<gemm_int>(input_size),
                0.0f, grad_W_ih, static_cast<gemm_int>(input_size));

    detail::check_gemm_dims(total_rows, input_size, gate_size, gate_size, input_size, input_size);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<gemm_int>(total_rows), static_cast<gemm_int>(input_size), static_cast<gemm_int>(gate_size),
                1.0f, d_gates_all.data(), static_cast<gemm_int>(gate_size),
                W_ih, static_cast<gemm_int>(input_size),
                0.0f, grad_input, static_cast<gemm_int>(input_size));
#else
    // PERF: no oneDNN/generic-CBLAS NN/TN GEMM macro exists yet for this
    // orientation (only LSTM_SGEMM_NT does); this scalar fallback is
    // numerically correct but O(N) slower than the MKL path. On non-MKL
    // builds this may reduce or eliminate the fused path's speed advantage
    // over the per-timestep loop it replaces. Adding LSTM_SGEMM_NN/
    // LSTM_SGEMM_TN macros (mirroring LSTM_SGEMM_NT's three-backend-variant
    // structure) is tracked as follow-up work, not required for correctness.
    // (Covers both the grad_W_ih [TN] and grad_input [NN] products below.)
    std::fill(grad_W_ih, grad_W_ih + gate_size * input_size, 0.0f);
    for (int64_t r = 0; r < total_rows; ++r) {
        const float* dg_row = d_gates_all.data() + r * gate_size;
        const float* in_row = input + r * input_size;
        for (int64_t k = 0; k < gate_size; ++k) {
            float dgv = dg_row[k];
            if (dgv == 0.0f) continue;
            for (int64_t f = 0; f < input_size; ++f) grad_W_ih[k * input_size + f] += dgv * in_row[f];
        }
    }
    std::fill(grad_input, grad_input + total_rows * input_size, 0.0f);
    for (int64_t r = 0; r < total_rows; ++r) {
        const float* dg_row = d_gates_all.data() + r * gate_size;
        float* gi_row = grad_input + r * input_size;
        for (int64_t k = 0; k < gate_size; ++k) {
            float dgv = dg_row[k];
            if (dgv == 0.0f) continue;
            for (int64_t f = 0; f < input_size; ++f) gi_row[f] += dgv * W_ih[k * input_size + f];
        }
    }
#endif

    // grad_W_hh = d_gates_all^T @ h_prev_all, where h_prev_all is the sequence
    // of hidden states FED INTO each timestep: [h0, h_0, h_1, ..., h_{T-2}].
    // This function is not given the h-sequence buffer, but it can reconstruct
    // it exactly from the reserves: lstm_forward_training defines
    // h_t = o_t * tanh(c_t), and both o_t (gates_reserve's o plane) and
    // tanh(c_t) (tanh_cell_reserve) are stored per timestep. So
    //   t == 0 -> h0
    //   t >  0 -> o_{t-1} * tanh_cell_reserve[t-1]   (== output[t-1])
    // Because h_prev must be materialised per timestep, this is accumulated as
    // a per-timestep rank-`batch` update rather than one batched GEMM.
    std::fill(grad_W_hh, grad_W_hh + gate_size * hidden, 0.0f);
    std::vector<float> h_prev(static_cast<size_t>(batch * hidden));
    for (int64_t t = 0; t < seq_len; ++t) {
        if (t == 0) {
            std::memcpy(h_prev.data(), h0, batch * hidden * sizeof(float));
        } else {
            const float* o_prev = o_plane + (t - 1) * batch * hidden;
            const float* tanh_c_prev = tanh_cell_reserve + (t - 1) * batch * hidden;
            for (int64_t i = 0; i < batch * hidden; ++i) h_prev[i] = o_prev[i] * tanh_c_prev[i];
        }
        const float* d_gates_t = d_gates_all.data() + t * batch * gate_size;
        for (int64_t b = 0; b < batch; ++b) {
            const float* dg_row = d_gates_t + b * gate_size;
            const float* h_row = h_prev.data() + b * hidden;
            for (int64_t k = 0; k < gate_size; ++k) {
                float dgv = dg_row[k];
                if (dgv == 0.0f) continue;
                for (int64_t d = 0; d < hidden; ++d) grad_W_hh[k * hidden + d] += dgv * h_row[d];
            }
        }
    }
}

// RAII guard that sets MKL's per-thread count to `n` for the duration of its
// lifetime and restores the previous value on destruction. Used to prevent
// 2*N MKL over-subscription when two lstm_forward calls run concurrently in
// #pragma omp parallel sections.
#ifdef TENZOR_USE_MKL
struct MklLocalThreads {
    int saved;
    explicit MklLocalThreads(int n) : saved(mkl_get_max_threads()) {
        mkl_set_num_threads_local(n);
    }
    ~MklLocalThreads() { mkl_set_num_threads_local(saved); }
};
#else
// No-op when MKL is not present
struct MklLocalThreads {
    explicit MklLocalThreads(int) {}
};
#endif

/**
 * @brief Bidirectional LSTM forward pass
 */
inline void lstm_forward_bidirectional(
    const float* input,
    const float* W_ih_fwd, const float* W_hh_fwd, const float* bias_fwd,
    const float* W_ih_bwd, const float* W_hh_bwd, const float* bias_bwd,
    const float* h0_fwd, const float* c0_fwd,
    const float* h0_bwd, const float* c0_bwd,
    float* output,          // (seq_len, batch, 2*hidden)
    float* h_n_fwd, float* c_n_fwd,
    float* h_n_bwd, float* c_n_bwd,
    int64_t seq_len,
    int64_t batch,
    int64_t input_size,
    int64_t hidden
) {
    // Allocate buffers for forward and backward outputs
    auto fwd_output_buf = acquire_buffer<float>(seq_len * batch * hidden);
    auto bwd_output_buf = acquire_buffer<float>(seq_len * batch * hidden);
    float* fwd_output = fwd_output_buf.data();
    float* bwd_output = bwd_output_buf.data();

    // Run forward and backward LSTMs in parallel.
    // Each section calls lstm_forward which internally uses MKL GEMM.  Without
    // throttling, two concurrent sections each use mkl_get_max_threads() worker
    // threads, creating 2*N oversubscription.  Restrict each MKL call to 1
    // thread for the duration of the parallel region so total = 2 * 1 = 2 threads
    // (the two OMP sections) instead of 2 * N.
    {
        // Only throttle MKL when the two sections actually run concurrently
        // (batch >= 2). For batch == 1 the sections run serially, so pinning MKL
        // to 1 thread would needlessly single-thread the dominant GEMMs — leave
        // MKL at its default thread count in that case.
        //
        // mkl_set_num_threads_local affects ONLY the calling thread, so the guard
        // must be constructed inside each omp section (i.e. on the worker thread
        // that actually issues the GEMMs); a guard built on the master thread
        // before the parallel region has no effect on the section workers.
        const bool throttle = (batch >= 2);
    #pragma omp parallel sections if(batch >= 2)
    {
        #pragma omp section
        {
            std::optional<MklLocalThreads> _mkl_guard;
            if (throttle) _mkl_guard.emplace(1);
            lstm_forward(input, W_ih_fwd, W_hh_fwd, bias_fwd,
                        h0_fwd, c0_fwd, fwd_output, h_n_fwd, c_n_fwd,
                        seq_len, batch, input_size, hidden);
        }
        #pragma omp section
        {
            std::optional<MklLocalThreads> _mkl_guard;
            if (throttle) _mkl_guard.emplace(1);
            // Backward: reverse input order
            auto input_rev_buf = acquire_buffer<float>(seq_len * batch * input_size);
            float* input_rev = input_rev_buf.data();

            // Reverse sequence
            for (int64_t t = 0; t < seq_len; ++t) {
                std::memcpy(input_rev + t * batch * input_size,
                           input + (seq_len - 1 - t) * batch * input_size,
                           batch * input_size * sizeof(float));
            }

            lstm_forward(input_rev, W_ih_bwd, W_hh_bwd, bias_bwd,
                        h0_bwd, c0_bwd, bwd_output, h_n_bwd, c_n_bwd,
                        seq_len, batch, input_size, hidden);
        }
    }
    } // end throttle scope

    // Concatenate forward and backward outputs
    #pragma omp parallel for if(seq_len * batch > 16)
    for (int64_t t = 0; t < seq_len; ++t) {
        for (int64_t b = 0; b < batch; ++b) {
            float* out = output + (t * batch + b) * 2 * hidden;
            const float* fwd = fwd_output + (t * batch + b) * hidden;
            // Backward output is reversed
            const float* bwd = bwd_output + ((seq_len - 1 - t) * batch + b) * hidden;

            std::memcpy(out, fwd, hidden * sizeof(float));
            std::memcpy(out + hidden, bwd, hidden * sizeof(float));
        }
    }
}

// ============================================================================
// GRU Support (Similar structure)
// ============================================================================

/**
 * @brief Fused GRU cell computation
 *
 * GRU equations:
 *   r = sigmoid(W_ir @ x + W_hr @ h + b_r)
 *   z = sigmoid(W_iz @ x + W_hz @ h + b_z)
 *   n = tanh(W_in @ x + r * (W_hn @ h + b_hn) + b_n)
 *   h_new = (1 - z) * n + z * h
 */
// F.2: PyTorch-faithful GRU cell. Takes bias_ih and bias_hh SEPARATELY so
// the n-gate's `r * (W_hh @ h + b_hh)` term is computed exactly. Calling
// with `bias_ih` and `bias_hh = nullptr` reproduces the legacy single-bias
// semantics for back-compat.
//   r = sigmoid(W_ih @ x + b_ih_r + W_hh @ h + b_hh_r)
//   z = sigmoid(W_ih @ x + b_ih_z + W_hh @ h + b_hh_z)
//   n = tanh((W_ih @ x + b_ih_n) + r * (W_hh @ h + b_hh_n))
//   h_new = (1 - z) * n + z * h
inline void gru_cell_fused(
    const float* gates_ih,  // (batch, 3*hidden) - pre-computed input gates [r_i, z_i, n_i]
    const float* h,         // (batch, hidden) - previous hidden
    const float* W_hh,      // (3*hidden, hidden) - hidden weights
    const float* bias_ih,   // (3*hidden) or nullptr — input-side bias
    const float* bias_hh,   // (3*hidden) or nullptr — hidden-side bias
    float* h_out,           // (batch, hidden) - new hidden
    float* workspace,       // (batch, 3*hidden) - temp buffer
    int64_t batch,
    int64_t hidden
) {
    int64_t gate_size = 3 * hidden;

    // Compute gates_hh = h @ W_hh^T
    // Uses oneDNN when available to avoid MKL/ROCm interaction crashes
    LSTM_SGEMM_NT(batch, gate_size, hidden,
                  1.0f, h, hidden,
                  W_hh, hidden,
                  0.0f, workspace, gate_size);

    // NOTE: Avoid OMP here to prevent nested parallelism with BLAS threads
    for (int64_t b = 0; b < batch; ++b) {
        const float* g_ih = gates_ih + b * gate_size;
        const float* g_hh = workspace + b * gate_size;
        const float* h_prev = h + b * hidden;
        float* h_new = h_out + b * hidden;

        int64_t d = 0;

#ifdef TENZOR_LSTM_AVX2
        for (; d + 8 <= hidden; d += 8) {
            __m256 r_gate = _mm256_add_ps(_mm256_loadu_ps(g_ih + d),
                                          _mm256_loadu_ps(g_hh + d));
            __m256 z_gate = _mm256_add_ps(_mm256_loadu_ps(g_ih + hidden + d),
                                          _mm256_loadu_ps(g_hh + hidden + d));
            __m256 n_ih = _mm256_loadu_ps(g_ih + 2 * hidden + d);
            __m256 n_hh = _mm256_loadu_ps(g_hh + 2 * hidden + d);

            // F.2: apply PyTorch GRU bias convention. b_ih is summed into
            // every gate's i-side; b_hh is summed into the r/z gates'
            // h-side (combined into the gate sum) and — critically — into
            // the n gate's h-side BEFORE the r multiplication.
            if (bias_ih) {
                r_gate = _mm256_add_ps(r_gate, _mm256_loadu_ps(bias_ih + d));
                z_gate = _mm256_add_ps(z_gate, _mm256_loadu_ps(bias_ih + hidden + d));
                n_ih   = _mm256_add_ps(n_ih,   _mm256_loadu_ps(bias_ih + 2 * hidden + d));
            }
            if (bias_hh) {
                r_gate = _mm256_add_ps(r_gate, _mm256_loadu_ps(bias_hh + d));
                z_gate = _mm256_add_ps(z_gate, _mm256_loadu_ps(bias_hh + hidden + d));
                n_hh   = _mm256_add_ps(n_hh,   _mm256_loadu_ps(bias_hh + 2 * hidden + d));
            }

            __m256 r = sigmoid_avx2(r_gate);
            __m256 z = sigmoid_avx2(z_gate);

            // n = tanh(n_ih + r * (n_hh + b_hh_n))
            __m256 n_gate = _mm256_fmadd_ps(r, n_hh, n_ih);
            __m256 n = tanh_avx2(n_gate);

            // h_new = (1 - z) * n + z * h
            __m256 h_p = _mm256_loadu_ps(h_prev + d);
            __m256 one = _mm256_set1_ps(1.0f);
            __m256 one_minus_z = _mm256_sub_ps(one, z);
            __m256 h_n = _mm256_fmadd_ps(one_minus_z, n, _mm256_mul_ps(z, h_p));

            _mm256_storeu_ps(h_new + d, h_n);
        }
#endif

        for (; d < hidden; ++d) {
            float r_gate = g_ih[d] + g_hh[d];
            float z_gate = g_ih[hidden + d] + g_hh[hidden + d];
            float n_ih = g_ih[2 * hidden + d];
            float n_hh = g_hh[2 * hidden + d];

            if (bias_ih) {
                r_gate += bias_ih[d];
                z_gate += bias_ih[hidden + d];
                n_ih   += bias_ih[2 * hidden + d];
            }
            if (bias_hh) {
                r_gate += bias_hh[d];
                z_gate += bias_hh[hidden + d];
                n_hh   += bias_hh[2 * hidden + d];
            }

            float r = sigmoid_scalar(r_gate);
            float z = sigmoid_scalar(z_gate);
            float n = tanh_scalar(n_ih + r * n_hh);

            h_new[d] = (1.0f - z) * n + z * h_prev[d];
        }
    }
}

/**
 * @brief Fused GRU forward pass
 */
inline void gru_forward(
    const float* input,
    const float* W_ih,
    const float* W_hh,
    const float* bias_ih,   // F.2: PyTorch convention — two biases.
    const float* bias_hh,   //       Either may be nullptr.
    const float* h0,
    float* output,
    float* h_n,
    int64_t seq_len,
    int64_t batch,
    int64_t input_size,
    int64_t hidden
) {
    int64_t gate_size = 3 * hidden;

    auto gates_ih_buf = acquire_buffer<float>(seq_len * batch * gate_size);
    auto gates_hh_buf = acquire_buffer<float>(batch * gate_size);
    auto h_temp_buf = acquire_buffer<float>(batch * hidden);

    float* gates_ih = gates_ih_buf.data();
    float* gates_hh = gates_hh_buf.data();
    float* h_curr = h_temp_buf.data();

    // Pre-compute all input gates
    // Uses oneDNN when available to avoid MKL/ROCm interaction crashes
    LSTM_SGEMM_NT(seq_len * batch, gate_size, input_size,
                  1.0f, input, input_size,
                  W_ih, input_size,
                  0.0f, gates_ih, gate_size);

    std::memcpy(h_curr, h0, batch * hidden * sizeof(float));

    for (int64_t t = 0; t < seq_len; ++t) {
        const float* gates_ih_t = gates_ih + t * batch * gate_size;
        float* output_t = output + t * batch * hidden;

        gru_cell_fused(gates_ih_t, h_curr, W_hh, bias_ih, bias_hh,
                       output_t, gates_hh, batch, hidden);
        std::memcpy(h_curr, output_t, batch * hidden * sizeof(float));
    }

    std::memcpy(h_n, h_curr, batch * hidden * sizeof(float));
}

} // namespace lstm
} // namespace cpu
} // namespace tenzor
