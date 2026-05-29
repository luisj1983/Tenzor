/**
 * @file rnn_onednn.hpp
 * @brief oneDNN-accelerated LSTM and GRU primitives
 *
 * Optimized for maximum performance with:
 * - Fast single-entry cache for common inference pattern (same dimensions)
 * - Pre-allocated workspace buffers to eliminate per-call allocations
 * - Efficient weight reordering with SIMD
 * - Lazy initialization to avoid static init crashes
 */

#pragma once

#ifdef TENZOR_USE_ONEDNN

#include <dnnl.hpp>
#include <iostream>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif
#include "../cpu_thread_config.hpp"
#include "onednn_cache.hpp"
#include "tenzor/utils/log.hpp"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace tenzor {
namespace cpu {
namespace rnn_onednn {

// ============================================================================
// oneDNN engine and stream — delegate to single per-thread instance
// ============================================================================

/// Returns the shared thread-local oneDNN engine (from onednn_cache.hpp).
inline dnnl::engine& get_engine() {
    // Ensure OMP thread count is configured before first use.
    static thread_local bool threads_configured = false;
    if (!threads_configured) {
        threads_configured = true;
        tenzor::backends::cpu::configure_omp_threads();
    }
    return tenzor::cpu::get_onednn_engine();
}

/// Returns the shared thread-local oneDNN stream (from onednn_cache.hpp).
inline dnnl::stream& get_stream() {
    return tenzor::cpu::get_onednn_stream();
}

// ============================================================================
// Optimized LSTM Primitive Cache
// Uses single-entry fast path for common case (same dimensions across calls)
// ============================================================================

struct LSTMCachedPrimitive {
    dnnl::lstm_forward prim;
    dnnl::lstm_forward::primitive_desc pd;

    // Memory descriptors
    dnnl::memory::desc src_layer_md;
    dnnl::memory::desc src_iter_md;
    dnnl::memory::desc src_iter_c_md;
    dnnl::memory::desc dst_layer_md;
    dnnl::memory::desc dst_iter_md;
    dnnl::memory::desc dst_iter_c_md;

    // Pre-allocated memories with optimal format
    dnnl::memory weights_layer_mem;
    dnnl::memory weights_iter_mem;
    dnnl::memory bias_mem;
    dnnl::memory scratchpad_mem;

    // Pre-allocated input/output memories (reused across calls)
    dnnl::memory src_layer_mem;
    dnnl::memory src_iter_mem;
    dnnl::memory src_iter_c_mem;
    dnnl::memory dst_layer_mem;
    dnnl::memory dst_iter_mem;
    dnnl::memory dst_iter_c_mem;

    // Pre-computed execution args (avoids map allocation on every call)
    std::unordered_map<int, dnnl::memory> args;

    // Dimensions
    int64_t seq_len;
    int64_t batch;
    int64_t input_size;
    int64_t hidden_size;
    bool has_bias;
    const void* w_ih_ptr;
    const void* w_hh_ptr;
    // Content fingerprint of the weights at the time the reordered primitive
    // was built. Pointer identity alone is unsafe: an in-place optimizer update
    // (same storage, new values) or a realloc at the same address would reuse
    // STALE reordered weights (reordering happens only on rebuild).
    uint64_t weight_fp{0};

    // Full matching including weight pointers (like GRU)
    bool matches(int64_t s, int64_t b, int64_t i, int64_t h, bool bias,
                 const void* wih, const void* whh) const {
        return seq_len == s && batch == b && input_size == i &&
               hidden_size == h && has_bias == bias &&
               w_ih_ptr == wih && w_hh_ptr == whh;
    }
};

// Fast weight fingerprint computation (samples key elements for quick hash)
inline uint64_t compute_weight_fingerprint(const float* W_ih, const float* W_hh,
                                            int64_t ih_size, int64_t hh_size) {
    uint64_t hash = 0;
    // Sample first, middle, and last elements from each weight matrix
    if (ih_size > 0) {
        hash ^= *reinterpret_cast<const uint32_t*>(W_ih);
        hash ^= *reinterpret_cast<const uint32_t*>(W_ih + ih_size/2) << 16;
        hash ^= *reinterpret_cast<const uint32_t*>(W_ih + ih_size - 1) << 8;
    }
    if (hh_size > 0) {
        hash ^= static_cast<uint64_t>(*reinterpret_cast<const uint32_t*>(W_hh)) << 32;
        hash ^= static_cast<uint64_t>(*reinterpret_cast<const uint32_t*>(W_hh + hh_size/2)) << 48;
        hash ^= static_cast<uint64_t>(*reinterpret_cast<const uint32_t*>(W_hh + hh_size - 1)) << 40;
    }
    return hash;
}

// Fast single-entry cache - optimal for inference where dimensions don't change
inline std::shared_ptr<LSTMCachedPrimitive>& get_lstm_cached() {
    static thread_local std::shared_ptr<LSTMCachedPrimitive> cached;
    return cached;
}

// Separate cache for BiLSTM backward direction (avoids thrashing main cache)
inline std::shared_ptr<LSTMCachedPrimitive>& get_lstm_backward_cached() {
    static thread_local std::shared_ptr<LSTMCachedPrimitive> cached;
    return cached;
}

/**
 * @brief Reorder weights from PyTorch format to oneDNN ldigo format
 *
 * PyTorch LSTM weights: (4*hidden, input_size) row-major
 * oneDNN LSTM weights: ldigo format (1, 1, input_size, 4, hidden_size)
 *
 * Optimized with block tiling for cache-friendly access (same pattern as GRU).
 */
inline void reorder_weights_to_ldigo(
    const float* src,           // PyTorch format: (4*hidden, in_features)
    float* dst,                 // oneDNN ldigo: (1, 1, in_features, 4, hidden)
    int64_t in_features,
    int64_t hidden_size
) {
    // ldigo layout: dst[i * 4 * hidden + g * hidden + h] = src[(g * hidden + h) * in_features + i]
    // Use block tiling for cache-friendly access (matches GRU optimization)
    constexpr int64_t BLOCK = 32;

    for (int64_t i_block = 0; i_block < in_features; i_block += BLOCK) {
        int64_t i_end = std::min(i_block + BLOCK, in_features);

        for (int64_t g = 0; g < 4; ++g) {
            const float* src_gate = src + g * hidden_size * in_features;

            for (int64_t h = 0; h < hidden_size; ++h) {
                const float* src_row = src_gate + h * in_features;

                for (int64_t i = i_block; i < i_end; ++i) {
                    dst[i * 4 * hidden_size + g * hidden_size + h] = src_row[i];
                }
            }
        }
    }
}

/**
 * @brief oneDNN-accelerated LSTM forward pass
 *
 * Optimized implementation with:
 * - Single-entry cache for fast path (no hash table lookup)
 * - Pre-allocated memory objects reused across calls
 * - Minimal per-call overhead
 */
inline bool lstm_forward_onednn(
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
    int64_t hidden_size
) {
    try {
        auto& engine = get_engine();
        auto& stream = get_stream();
        auto& cached = get_lstm_cached();

        bool has_bias = (bias != nullptr);

        // LSTM has 4 gates: W_ih is (4*hidden, input), W_hh is (4*hidden, hidden).
        const uint64_t weight_fp = compute_weight_fingerprint(
            W_ih, W_hh, 4 * hidden_size * input_size, 4 * hidden_size * hidden_size);

        // Reuse the cached primitive only if dims AND weight content match;
        // pointer-only matching would serve stale reordered weights after an
        // in-place update or realloc-at-same-address (audit B6).
        bool need_rebuild = !cached ||
                           !cached->matches(seq_len, batch, input_size, hidden_size,
                                           has_bias, W_ih, W_hh) ||
                           cached->weight_fp != weight_fp;

        if (need_rebuild) {
            cached = std::make_shared<LSTMCachedPrimitive>();
            cached->seq_len = seq_len;
            cached->batch = batch;
            cached->input_size = input_size;
            cached->hidden_size = hidden_size;
            cached->has_bias = has_bias;
            cached->w_ih_ptr = W_ih;
            cached->w_hh_ptr = W_hh;
            cached->weight_fp = weight_fp;

            // Define dimensions
            dnnl::memory::dims src_layer_dims = {seq_len, batch, input_size};
            dnnl::memory::dims src_iter_dims = {1, 1, batch, hidden_size};
            dnnl::memory::dims weights_layer_dims = {1, 1, input_size, 4, hidden_size};
            dnnl::memory::dims weights_iter_dims = {1, 1, hidden_size, 4, hidden_size};
            dnnl::memory::dims bias_dims = {1, 1, 4, hidden_size};
            dnnl::memory::dims dst_layer_dims = {seq_len, batch, hidden_size};
            dnnl::memory::dims dst_iter_dims = {1, 1, batch, hidden_size};

            // Memory descriptors
            cached->src_layer_md = dnnl::memory::desc(src_layer_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::tnc);
            cached->src_iter_md = dnnl::memory::desc(src_iter_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::ldnc);
            cached->src_iter_c_md = dnnl::memory::desc(src_iter_dims, dnnl::memory::data_type::f32,
                                                        dnnl::memory::format_tag::ldnc);
            cached->dst_layer_md = dnnl::memory::desc(dst_layer_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::tnc);
            cached->dst_iter_md = dnnl::memory::desc(dst_iter_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::ldnc);
            cached->dst_iter_c_md = dnnl::memory::desc(dst_iter_dims, dnnl::memory::data_type::f32,
                                                        dnnl::memory::format_tag::ldnc);

            // Use format_tag::any to let oneDNN choose optimal blocked format (e.g., abdEc32e)
            // This enables brgemm:avx512_core implementation instead of ref
            auto weights_layer_md = dnnl::memory::desc(weights_layer_dims, dnnl::memory::data_type::f32,
                                                        dnnl::memory::format_tag::any);
            auto weights_iter_md = dnnl::memory::desc(weights_iter_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::any);
            auto bias_md = dnnl::memory::desc(bias_dims, dnnl::memory::data_type::f32,
                                               dnnl::memory::format_tag::ldgo);

            // Create primitive descriptor
            cached->pd = dnnl::lstm_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                dnnl::rnn_direction::unidirectional_left2right,
                cached->src_layer_md,
                cached->src_iter_md,
                cached->src_iter_c_md,
                weights_layer_md,
                weights_iter_md,
                bias_md,
                cached->dst_layer_md,
                cached->dst_iter_md,
                cached->dst_iter_c_md
            );

            // Create primitive
            cached->prim = dnnl::lstm_forward(cached->pd);

            // Allocate weight memories with optimal format
            cached->weights_layer_mem = dnnl::memory(cached->pd.weights_layer_desc(), engine);
            cached->weights_iter_mem = dnnl::memory(cached->pd.weights_iter_desc(), engine);
            cached->bias_mem = dnnl::memory(cached->pd.bias_desc(), engine);
            cached->scratchpad_mem = dnnl::memory(cached->pd.scratchpad_desc(), engine);

            // Pre-allocate input/output memories
            cached->src_layer_mem = dnnl::memory(cached->src_layer_md, engine);
            cached->src_iter_mem = dnnl::memory(cached->src_iter_md, engine);
            cached->src_iter_c_mem = dnnl::memory(cached->src_iter_c_md, engine);
            cached->dst_layer_mem = dnnl::memory(cached->dst_layer_md, engine);
            cached->dst_iter_mem = dnnl::memory(cached->dst_iter_md, engine);
            cached->dst_iter_c_mem = dnnl::memory(cached->dst_iter_c_md, engine);

            // Reorder weights to optimal format
            std::vector<float> w_layer_ldigo(input_size * 4 * hidden_size);
            std::vector<float> w_iter_ldigo(hidden_size * 4 * hidden_size);

            reorder_weights_to_ldigo(W_ih, w_layer_ldigo.data(), input_size, hidden_size);
            reorder_weights_to_ldigo(W_hh, w_iter_ldigo.data(), hidden_size, hidden_size);

            // Copy to oneDNN memory (using optimal format)
            auto user_weights_layer_md = dnnl::memory::desc(weights_layer_dims, dnnl::memory::data_type::f32,
                                                             dnnl::memory::format_tag::ldigo);
            auto user_weights_iter_md = dnnl::memory::desc(weights_iter_dims, dnnl::memory::data_type::f32,
                                                            dnnl::memory::format_tag::ldigo);

            if (cached->pd.weights_layer_desc() != user_weights_layer_md) {
                auto user_mem = dnnl::memory(user_weights_layer_md, engine, w_layer_ldigo.data());
                dnnl::reorder(user_mem, cached->weights_layer_mem).execute(stream, user_mem, cached->weights_layer_mem);
            } else {
                std::memcpy(cached->weights_layer_mem.get_data_handle(),
                           w_layer_ldigo.data(), w_layer_ldigo.size() * sizeof(float));
            }

            if (cached->pd.weights_iter_desc() != user_weights_iter_md) {
                auto user_mem = dnnl::memory(user_weights_iter_md, engine, w_iter_ldigo.data());
                dnnl::reorder(user_mem, cached->weights_iter_mem).execute(stream, user_mem, cached->weights_iter_mem);
            } else {
                std::memcpy(cached->weights_iter_mem.get_data_handle(),
                           w_iter_ldigo.data(), w_iter_ldigo.size() * sizeof(float));
            }

            // Handle bias
            if (has_bias) {
                std::memcpy(cached->bias_mem.get_data_handle(), bias, 4 * hidden_size * sizeof(float));
            } else {
                std::memset(cached->bias_mem.get_data_handle(), 0, 4 * hidden_size * sizeof(float));
            }

            // Pre-build args map to avoid allocation on every execute
            cached->args = {
                {DNNL_ARG_SRC_LAYER, cached->src_layer_mem},
                {DNNL_ARG_SRC_ITER, cached->src_iter_mem},
                {DNNL_ARG_SRC_ITER_C, cached->src_iter_c_mem},
                {DNNL_ARG_WEIGHTS_LAYER, cached->weights_layer_mem},
                {DNNL_ARG_WEIGHTS_ITER, cached->weights_iter_mem},
                {DNNL_ARG_BIAS, cached->bias_mem},
                {DNNL_ARG_DST_LAYER, cached->dst_layer_mem},
                {DNNL_ARG_DST_ITER, cached->dst_iter_mem},
                {DNNL_ARG_DST_ITER_C, cached->dst_iter_c_mem},
                {DNNL_ARG_SCRATCHPAD, cached->scratchpad_mem}
            };

            stream.wait();
        }

        // Fast path: update data pointers and execute
        // For h0/c0, since ldnc with L=D=1 has same layout as (batch, hidden), we can use directly
        cached->src_layer_mem.set_data_handle(const_cast<float*>(input));
        cached->src_iter_mem.set_data_handle(const_cast<float*>(h0));
        cached->src_iter_c_mem.set_data_handle(const_cast<float*>(c0));
        cached->dst_layer_mem.set_data_handle(output);
        cached->dst_iter_mem.set_data_handle(h_n);
        cached->dst_iter_c_mem.set_data_handle(c_n);

        // Execute with pre-built args (no allocation)
        cached->prim.execute(stream, cached->args);
        stream.wait();
        return true;

    } catch (const dnnl::error& e) {
        // TENZOR_STRICT_BACKEND=1 promotes oneDNN failures to hard errors so
        // silent fallbacks are auditable. Otherwise log a WARN and return
        // false so the caller can fall back to the scalar path.
        if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
            throw std::runtime_error(
                std::string("[LSTM single-layer oneDNN] failed (TENZOR_STRICT_BACKEND=1): ") +
                e.what());
        }
        TENZOR_LOG_WARN("[LSTM single-layer oneDNN] forward failed ({}); using scalar fallback",
                        e.what());
        return false;
    } catch (const std::exception& e) {
        // Audit L.3: surface unexpected non-dnnl exceptions with type+message.
        // The outer dnnl::error catch above already handles primitive-
        // descriptor failures with a documented scalar fallback;
        // anything reaching this branch is a non-oneDNN exception
        // (bad_alloc, logic_error from our wrappers, etc.) and
        // should not be silently swallowed.
        TENZOR_LOG_ERROR("[LSTM single-layer oneDNN] non-dnnl exception: type={} msg={}",
                         typeid(e).name(), e.what());
        throw;
    } catch (...) {
        // Audit L.3: unknown exception type (not derived from std::exception).
        TENZOR_LOG_ERROR("[LSTM single-layer oneDNN] unknown exception type in fast path; rethrowing");
        throw;
    }
}

// ============================================================================
// Internal LSTM function with explicit cache (for BiLSTM)
// ============================================================================

/**
 * @brief Internal LSTM forward with explicit cache selection
 *
 * Same as lstm_forward_onednn but allows specifying which cache to use.
 * Used by BiLSTM to maintain separate caches for forward/backward directions.
 */
inline bool lstm_forward_onednn_with_cache(
    std::shared_ptr<LSTMCachedPrimitive>& cached,
    const float* input, const float* W_ih, const float* W_hh, const float* bias,
    const float* h0, const float* c0,
    float* output, float* h_n, float* c_n,
    int64_t seq_len, int64_t batch, int64_t input_size, int64_t hidden_size
) {
    try {
        auto& engine = get_engine();
        auto& stream = get_stream();

        bool has_bias = (bias != nullptr);

        const uint64_t weight_fp = compute_weight_fingerprint(
            W_ih, W_hh, 4 * hidden_size * input_size, 4 * hidden_size * hidden_size);

        bool need_rebuild = !cached ||
                           !cached->matches(seq_len, batch, input_size, hidden_size,
                                           has_bias, W_ih, W_hh) ||
                           cached->weight_fp != weight_fp;

        if (need_rebuild) {
            cached = std::make_shared<LSTMCachedPrimitive>();
            cached->seq_len = seq_len;
            cached->batch = batch;
            cached->input_size = input_size;
            cached->hidden_size = hidden_size;
            cached->has_bias = has_bias;
            cached->weight_fp = weight_fp;
            cached->w_ih_ptr = W_ih;
            cached->w_hh_ptr = W_hh;

            dnnl::memory::dims src_layer_dims = {seq_len, batch, input_size};
            dnnl::memory::dims src_iter_dims = {1, 1, batch, hidden_size};
            dnnl::memory::dims weights_layer_dims = {1, 1, input_size, 4, hidden_size};
            dnnl::memory::dims weights_iter_dims = {1, 1, hidden_size, 4, hidden_size};
            dnnl::memory::dims bias_dims = {1, 1, 4, hidden_size};
            dnnl::memory::dims dst_layer_dims = {seq_len, batch, hidden_size};
            dnnl::memory::dims dst_iter_dims = {1, 1, batch, hidden_size};

            cached->src_layer_md = dnnl::memory::desc(src_layer_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::tnc);
            cached->src_iter_md = dnnl::memory::desc(src_iter_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::ldnc);
            cached->src_iter_c_md = dnnl::memory::desc(src_iter_dims, dnnl::memory::data_type::f32,
                                                        dnnl::memory::format_tag::ldnc);
            cached->dst_layer_md = dnnl::memory::desc(dst_layer_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::tnc);
            cached->dst_iter_md = dnnl::memory::desc(dst_iter_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::ldnc);
            cached->dst_iter_c_md = dnnl::memory::desc(dst_iter_dims, dnnl::memory::data_type::f32,
                                                        dnnl::memory::format_tag::ldnc);

            auto weights_layer_md = dnnl::memory::desc(weights_layer_dims, dnnl::memory::data_type::f32,
                                                        dnnl::memory::format_tag::any);
            auto weights_iter_md = dnnl::memory::desc(weights_iter_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::any);
            auto bias_md = dnnl::memory::desc(bias_dims, dnnl::memory::data_type::f32,
                                               dnnl::memory::format_tag::ldgo);

            cached->pd = dnnl::lstm_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                dnnl::rnn_direction::unidirectional_left2right,
                cached->src_layer_md, cached->src_iter_md, cached->src_iter_c_md,
                weights_layer_md, weights_iter_md, bias_md,
                cached->dst_layer_md, cached->dst_iter_md, cached->dst_iter_c_md
            );

            cached->prim = dnnl::lstm_forward(cached->pd);

            cached->weights_layer_mem = dnnl::memory(cached->pd.weights_layer_desc(), engine);
            cached->weights_iter_mem = dnnl::memory(cached->pd.weights_iter_desc(), engine);
            cached->bias_mem = dnnl::memory(cached->pd.bias_desc(), engine);
            cached->scratchpad_mem = dnnl::memory(cached->pd.scratchpad_desc(), engine);

            cached->src_layer_mem = dnnl::memory(cached->src_layer_md, engine);
            cached->src_iter_mem = dnnl::memory(cached->src_iter_md, engine);
            cached->src_iter_c_mem = dnnl::memory(cached->src_iter_c_md, engine);
            cached->dst_layer_mem = dnnl::memory(cached->dst_layer_md, engine);
            cached->dst_iter_mem = dnnl::memory(cached->dst_iter_md, engine);
            cached->dst_iter_c_mem = dnnl::memory(cached->dst_iter_c_md, engine);

            std::vector<float> w_layer_ldigo(input_size * 4 * hidden_size);
            std::vector<float> w_iter_ldigo(hidden_size * 4 * hidden_size);

            reorder_weights_to_ldigo(W_ih, w_layer_ldigo.data(), input_size, hidden_size);
            reorder_weights_to_ldigo(W_hh, w_iter_ldigo.data(), hidden_size, hidden_size);

            auto user_weights_layer_md = dnnl::memory::desc(weights_layer_dims, dnnl::memory::data_type::f32,
                                                             dnnl::memory::format_tag::ldigo);
            auto user_weights_iter_md = dnnl::memory::desc(weights_iter_dims, dnnl::memory::data_type::f32,
                                                            dnnl::memory::format_tag::ldigo);

            if (cached->pd.weights_layer_desc() != user_weights_layer_md) {
                dnnl::memory user_weights_layer(user_weights_layer_md, engine, w_layer_ldigo.data());
                dnnl::reorder(user_weights_layer, cached->weights_layer_mem).execute(stream, user_weights_layer, cached->weights_layer_mem);
            } else {
                std::memcpy(cached->weights_layer_mem.get_data_handle(), w_layer_ldigo.data(),
                           input_size * 4 * hidden_size * sizeof(float));
            }

            if (cached->pd.weights_iter_desc() != user_weights_iter_md) {
                dnnl::memory user_weights_iter(user_weights_iter_md, engine, w_iter_ldigo.data());
                dnnl::reorder(user_weights_iter, cached->weights_iter_mem).execute(stream, user_weights_iter, cached->weights_iter_mem);
            } else {
                std::memcpy(cached->weights_iter_mem.get_data_handle(), w_iter_ldigo.data(),
                           hidden_size * 4 * hidden_size * sizeof(float));
            }

            if (has_bias) {
                std::memcpy(cached->bias_mem.get_data_handle(), bias, 4 * hidden_size * sizeof(float));
            } else {
                std::memset(cached->bias_mem.get_data_handle(), 0, 4 * hidden_size * sizeof(float));
            }

            // Pre-build args map
            cached->args = {
                {DNNL_ARG_SRC_LAYER, cached->src_layer_mem},
                {DNNL_ARG_SRC_ITER, cached->src_iter_mem},
                {DNNL_ARG_SRC_ITER_C, cached->src_iter_c_mem},
                {DNNL_ARG_WEIGHTS_LAYER, cached->weights_layer_mem},
                {DNNL_ARG_WEIGHTS_ITER, cached->weights_iter_mem},
                {DNNL_ARG_BIAS, cached->bias_mem},
                {DNNL_ARG_DST_LAYER, cached->dst_layer_mem},
                {DNNL_ARG_DST_ITER, cached->dst_iter_mem},
                {DNNL_ARG_DST_ITER_C, cached->dst_iter_c_mem},
                {DNNL_ARG_SCRATCHPAD, cached->scratchpad_mem}
            };

            stream.wait();
        }

        cached->src_layer_mem.set_data_handle(const_cast<float*>(input));
        cached->src_iter_mem.set_data_handle(const_cast<float*>(h0));
        cached->src_iter_c_mem.set_data_handle(const_cast<float*>(c0));
        cached->dst_layer_mem.set_data_handle(output);
        cached->dst_iter_mem.set_data_handle(h_n);
        cached->dst_iter_c_mem.set_data_handle(c_n);

        cached->prim.execute(stream, cached->args);

        stream.wait();
        return true;

    } catch (const std::exception& e) {
        // Audit L.3: surface unexpected exceptions with type+message instead
        // of silently returning false; the caller's fallback should kick in
        // only on a documented dnnl::error path.
        TENZOR_LOG_ERROR("[LSTM oneDNN cached-primitive] non-dnnl exception: type={} msg={}",
                         typeid(e).name(), e.what());
        throw;
    } catch (...) {
        // Audit L.3: unknown exception type (not derived from std::exception).
        TENZOR_LOG_ERROR("[LSTM oneDNN cached-primitive] unknown exception type; rethrowing");
        throw;
    }
}

// ============================================================================
// Bidirectional LSTM - runs forward and backward directions separately
// ============================================================================

/**
 * @brief oneDNN-accelerated Bidirectional LSTM forward pass
 *
 * Runs two LSTM primitives (forward and backward directions) and concatenates outputs.
 * Uses separate caches for each direction to avoid cache thrashing.
 * This matches PyTorch's implementation strategy.
 *
 * @param input Input tensor (seq_len, batch, input_size)
 * @param W_ih_fwd Forward direction input-hidden weights (4*hidden, input_size)
 * @param W_hh_fwd Forward direction hidden-hidden weights (4*hidden, hidden)
 * @param bias_fwd Forward direction bias (4*hidden) or nullptr
 * @param W_ih_bwd Backward direction input-hidden weights (4*hidden, input_size)
 * @param W_hh_bwd Backward direction hidden-hidden weights (4*hidden, hidden)
 * @param bias_bwd Backward direction bias (4*hidden) or nullptr
 * @param h0_fwd Forward initial hidden state (batch, hidden)
 * @param c0_fwd Forward initial cell state (batch, hidden)
 * @param h0_bwd Backward initial hidden state (batch, hidden)
 * @param c0_bwd Backward initial cell state (batch, hidden)
 * @param output Output tensor (seq_len, batch, 2*hidden) - concatenated
 * @param h_n_fwd Forward final hidden state (batch, hidden)
 * @param c_n_fwd Forward final cell state (batch, hidden)
 * @param h_n_bwd Backward final hidden state (batch, hidden)
 * @param c_n_bwd Backward final cell state (batch, hidden)
 */
inline bool bilstm_forward_onednn(
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
    int64_t hidden_size
) {
    try {
        // Use separate caches for forward and backward directions to avoid thrashing
        auto& fwd_cache = get_lstm_cached();
        auto& bwd_cache = get_lstm_backward_cached();

        // Allocate temporary buffers for individual direction outputs
        std::vector<float> output_fwd(seq_len * batch * hidden_size);
        std::vector<float> output_bwd(seq_len * batch * hidden_size);

        // Run forward direction LSTM using forward cache
        bool fwd_ok = lstm_forward_onednn_with_cache(
            fwd_cache,
            input, W_ih_fwd, W_hh_fwd, bias_fwd,
            h0_fwd, c0_fwd,
            output_fwd.data(), h_n_fwd, c_n_fwd,
            seq_len, batch, input_size, hidden_size
        );
        if (!fwd_ok) return false;

        // For backward direction, we need to reverse input, run LSTM, then reverse output
        // Create reversed input
        std::vector<float> input_reversed(seq_len * batch * input_size);
        for (int64_t t = 0; t < seq_len; ++t) {
            std::memcpy(
                input_reversed.data() + t * batch * input_size,
                input + (seq_len - 1 - t) * batch * input_size,
                batch * input_size * sizeof(float)
            );
        }

        // Run backward direction LSTM on reversed input using backward cache
        std::vector<float> output_bwd_reversed(seq_len * batch * hidden_size);
        bool bwd_ok = lstm_forward_onednn_with_cache(
            bwd_cache,
            input_reversed.data(), W_ih_bwd, W_hh_bwd, bias_bwd,
            h0_bwd, c0_bwd,
            output_bwd_reversed.data(), h_n_bwd, c_n_bwd,
            seq_len, batch, input_size, hidden_size
        );
        if (!bwd_ok) return false;

        // Reverse the backward output
        for (int64_t t = 0; t < seq_len; ++t) {
            std::memcpy(
                output_bwd.data() + t * batch * hidden_size,
                output_bwd_reversed.data() + (seq_len - 1 - t) * batch * hidden_size,
                batch * hidden_size * sizeof(float)
            );
        }

        // Concatenate forward and backward outputs along hidden dimension
        // output[t, b, :] = [output_fwd[t, b, :], output_bwd[t, b, :]]
        #pragma omp parallel for collapse(2)
        for (int64_t t = 0; t < seq_len; ++t) {
            for (int64_t b = 0; b < batch; ++b) {
                float* dst = output + (t * batch + b) * 2 * hidden_size;
                const float* src_fwd = output_fwd.data() + (t * batch + b) * hidden_size;
                const float* src_bwd = output_bwd.data() + (t * batch + b) * hidden_size;

                std::memcpy(dst, src_fwd, hidden_size * sizeof(float));
                std::memcpy(dst + hidden_size, src_bwd, hidden_size * sizeof(float));
            }
        }

        return true;

    } catch (const std::exception& e) {
        // Audit L.3: surface unexpected exceptions in the BiLSTM stitch
        // path with type+message. The fwd/bwd direction calls each handle
        // their own dnnl::error fallback; reaching this catch means
        // something else broke.
        TENZOR_LOG_ERROR("[BiLSTM oneDNN stitch] non-dnnl exception: type={} msg={}",
                         typeid(e).name(), e.what());
        throw;
    } catch (...) {
        // Audit L.3: unknown exception type (not derived from std::exception).
        TENZOR_LOG_ERROR("[BiLSTM oneDNN stitch] unknown exception type; rethrowing");
        throw;
    }
}

// ============================================================================
// Optimized GRU Primitive Cache (single-entry for fast inference)
// ============================================================================

struct GRUCachedPrimitive {
    dnnl::gru_forward prim;
    dnnl::gru_forward::primitive_desc pd;

    dnnl::memory::desc src_layer_md;
    dnnl::memory::desc src_iter_md;
    dnnl::memory::desc dst_layer_md;
    dnnl::memory::desc dst_iter_md;

    dnnl::memory weights_layer_mem;
    dnnl::memory weights_iter_mem;
    dnnl::memory bias_mem;
    dnnl::memory scratchpad_mem;

    // Pre-allocated input/output memories
    dnnl::memory src_layer_mem;
    dnnl::memory src_iter_mem;
    dnnl::memory dst_layer_mem;
    dnnl::memory dst_iter_mem;

    // Dimensions
    int64_t seq_len;
    int64_t batch;
    int64_t input_size;
    int64_t hidden_size;
    bool has_bias;
    const void* w_ih_ptr;
    const void* w_hh_ptr;
    uint64_t weight_fp{0};  // content fingerprint (audit B6: in-place update / realloc)

    bool matches(int64_t s, int64_t b, int64_t i, int64_t h, bool bias,
                 const void* wih, const void* whh) const {
        return seq_len == s && batch == b && input_size == i &&
               hidden_size == h && has_bias == bias &&
               w_ih_ptr == wih && w_hh_ptr == whh;
    }
};

inline std::shared_ptr<GRUCachedPrimitive>& get_gru_cached() {
    static thread_local std::shared_ptr<GRUCachedPrimitive> cached;
    return cached;
}

/**
 * @brief Reorder GRU weights from PyTorch format to oneDNN ldigo format
 */
inline void reorder_gru_weights_to_ldigo(
    const float* src,
    float* dst,
    int64_t in_features,
    int64_t hidden_size
) {
    constexpr int64_t BLOCK = 32;
    for (int64_t i_block = 0; i_block < in_features; i_block += BLOCK) {
        int64_t i_end = std::min(i_block + BLOCK, in_features);
        for (int64_t g = 0; g < 3; ++g) {
            const float* src_gate = src + g * hidden_size * in_features;
            for (int64_t h = 0; h < hidden_size; ++h) {
                const float* src_row = src_gate + h * in_features;
                for (int64_t i = i_block; i < i_end; ++i) {
                    dst[i * 3 * hidden_size + g * hidden_size + h] = src_row[i];
                }
            }
        }
    }
}

/**
 * @brief Optimized oneDNN GRU forward pass
 */
inline bool gru_forward_onednn(
    const float* input,     // (seq_len, batch, input_size)
    const float* W_ih,      // (3*hidden, input_size)
    const float* W_hh,      // (3*hidden, hidden)
    const float* bias,      // (3*hidden) or nullptr
    const float* h0,        // (batch, hidden)
    float* output,          // (seq_len, batch, hidden)
    float* h_n,             // (batch, hidden)
    int64_t seq_len,
    int64_t batch,
    int64_t input_size,
    int64_t hidden_size
) {
    try {
        auto& engine = get_engine();
        auto& stream = get_stream();
        auto& cached = get_gru_cached();

        bool has_bias = (bias != nullptr);

        // GRU has 3 gates: W_ih is (3*hidden, input), W_hh is (3*hidden, hidden).
        const uint64_t weight_fp = compute_weight_fingerprint(
            W_ih, W_hh, 3 * hidden_size * input_size, 3 * hidden_size * hidden_size);

        bool need_rebuild = !cached ||
                           !cached->matches(seq_len, batch, input_size, hidden_size,
                                           has_bias, W_ih, W_hh) ||
                           cached->weight_fp != weight_fp;

        if (need_rebuild) {
            cached = std::make_shared<GRUCachedPrimitive>();
            cached->weight_fp = weight_fp;
            cached->seq_len = seq_len;
            cached->batch = batch;
            cached->input_size = input_size;
            cached->hidden_size = hidden_size;
            cached->has_bias = has_bias;
            cached->w_ih_ptr = W_ih;
            cached->w_hh_ptr = W_hh;

            dnnl::memory::dims src_layer_dims = {seq_len, batch, input_size};
            dnnl::memory::dims src_iter_dims = {1, 1, batch, hidden_size};
            dnnl::memory::dims weights_layer_dims = {1, 1, input_size, 3, hidden_size};
            dnnl::memory::dims weights_iter_dims = {1, 1, hidden_size, 3, hidden_size};
            dnnl::memory::dims bias_dims = {1, 1, 3, hidden_size};
            dnnl::memory::dims dst_layer_dims = {seq_len, batch, hidden_size};
            dnnl::memory::dims dst_iter_dims = {1, 1, batch, hidden_size};

            cached->src_layer_md = dnnl::memory::desc(src_layer_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::tnc);
            cached->src_iter_md = dnnl::memory::desc(src_iter_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::ldnc);
            cached->dst_layer_md = dnnl::memory::desc(dst_layer_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::tnc);
            cached->dst_iter_md = dnnl::memory::desc(dst_iter_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::ldnc);

            // Use format_tag::any to let oneDNN choose optimal blocked format
            auto weights_layer_md = dnnl::memory::desc(weights_layer_dims, dnnl::memory::data_type::f32,
                                                        dnnl::memory::format_tag::any);
            auto weights_iter_md = dnnl::memory::desc(weights_iter_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::any);
            auto bias_md = dnnl::memory::desc(bias_dims, dnnl::memory::data_type::f32,
                                               dnnl::memory::format_tag::ldgo);

            cached->pd = dnnl::gru_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                dnnl::rnn_direction::unidirectional_left2right,
                cached->src_layer_md,
                cached->src_iter_md,
                weights_layer_md,
                weights_iter_md,
                bias_md,
                cached->dst_layer_md,
                cached->dst_iter_md
            );

            cached->prim = dnnl::gru_forward(cached->pd);

            cached->weights_layer_mem = dnnl::memory(cached->pd.weights_layer_desc(), engine);
            cached->weights_iter_mem = dnnl::memory(cached->pd.weights_iter_desc(), engine);
            cached->bias_mem = dnnl::memory(cached->pd.bias_desc(), engine);
            cached->scratchpad_mem = dnnl::memory(cached->pd.scratchpad_desc(), engine);

            // Pre-allocate input/output memories
            cached->src_layer_mem = dnnl::memory(cached->src_layer_md, engine);
            cached->src_iter_mem = dnnl::memory(cached->src_iter_md, engine);
            cached->dst_layer_mem = dnnl::memory(cached->dst_layer_md, engine);
            cached->dst_iter_mem = dnnl::memory(cached->dst_iter_md, engine);

            // Reorder weights
            std::vector<float> w_layer_ldigo(input_size * 3 * hidden_size);
            std::vector<float> w_iter_ldigo(hidden_size * 3 * hidden_size);

            reorder_gru_weights_to_ldigo(W_ih, w_layer_ldigo.data(), input_size, hidden_size);
            reorder_gru_weights_to_ldigo(W_hh, w_iter_ldigo.data(), hidden_size, hidden_size);

            auto user_weights_layer_md = dnnl::memory::desc(weights_layer_dims, dnnl::memory::data_type::f32,
                                                             dnnl::memory::format_tag::ldigo);
            auto user_weights_iter_md = dnnl::memory::desc(weights_iter_dims, dnnl::memory::data_type::f32,
                                                            dnnl::memory::format_tag::ldigo);

            if (cached->pd.weights_layer_desc() != user_weights_layer_md) {
                auto user_mem = dnnl::memory(user_weights_layer_md, engine, w_layer_ldigo.data());
                dnnl::reorder(user_mem, cached->weights_layer_mem).execute(stream, user_mem, cached->weights_layer_mem);
            } else {
                std::memcpy(cached->weights_layer_mem.get_data_handle(),
                           w_layer_ldigo.data(), w_layer_ldigo.size() * sizeof(float));
            }

            if (cached->pd.weights_iter_desc() != user_weights_iter_md) {
                auto user_mem = dnnl::memory(user_weights_iter_md, engine, w_iter_ldigo.data());
                dnnl::reorder(user_mem, cached->weights_iter_mem).execute(stream, user_mem, cached->weights_iter_mem);
            } else {
                std::memcpy(cached->weights_iter_mem.get_data_handle(),
                           w_iter_ldigo.data(), w_iter_ldigo.size() * sizeof(float));
            }

            if (has_bias) {
                std::memcpy(cached->bias_mem.get_data_handle(), bias, 3 * hidden_size * sizeof(float));
            } else {
                std::memset(cached->bias_mem.get_data_handle(), 0, 3 * hidden_size * sizeof(float));
            }

            stream.wait();
        }

        // Fast path: update data pointers and execute
        cached->src_layer_mem.set_data_handle(const_cast<float*>(input));
        cached->src_iter_mem.set_data_handle(const_cast<float*>(h0));
        cached->dst_layer_mem.set_data_handle(output);
        cached->dst_iter_mem.set_data_handle(h_n);

        cached->prim.execute(stream, {
            {DNNL_ARG_SRC_LAYER, cached->src_layer_mem},
            {DNNL_ARG_SRC_ITER, cached->src_iter_mem},
            {DNNL_ARG_WEIGHTS_LAYER, cached->weights_layer_mem},
            {DNNL_ARG_WEIGHTS_ITER, cached->weights_iter_mem},
            {DNNL_ARG_BIAS, cached->bias_mem},
            {DNNL_ARG_DST_LAYER, cached->dst_layer_mem},
            {DNNL_ARG_DST_ITER, cached->dst_iter_mem},
            {DNNL_ARG_SCRATCHPAD, cached->scratchpad_mem}
        });

        stream.wait();
        return true;

    } catch (const dnnl::error& e) {
        if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
            throw std::runtime_error(
                std::string("[GRU single-layer oneDNN] failed (TENZOR_STRICT_BACKEND=1): ") +
                e.what());
        }
        TENZOR_LOG_WARN("[GRU single-layer oneDNN] forward failed ({}); using scalar fallback",
                        e.what());
        return false;
    } catch (const std::exception& e) {
        // Audit L.3: surface unexpected non-dnnl exceptions in the GRU
        // single-layer fast path with type+message.
        TENZOR_LOG_ERROR("[GRU single-layer oneDNN] non-dnnl exception: type={} msg={}",
                         typeid(e).name(), e.what());
        throw;
    } catch (...) {
        // Audit L.3: unknown exception type (not derived from std::exception).
        TENZOR_LOG_ERROR("[GRU single-layer oneDNN] unknown exception type in fast path; rethrowing");
        throw;
    }
}

// ============================================================================
// Fused Multi-Layer LSTM Primitive
// ============================================================================

struct LSTMMultiLayerCachedPrimitive {
    // For layer 0 (potentially different input_size)
    std::shared_ptr<LSTMCachedPrimitive> layer0_cache;

    // For layers 1+ (fused, all have hidden_size as input)
    dnnl::lstm_forward prim;
    dnnl::lstm_forward::primitive_desc pd;
    bool has_fused_layers = false;

    dnnl::memory weights_layer_mem;
    dnnl::memory weights_iter_mem;
    dnnl::memory bias_mem;
    dnnl::memory scratchpad_mem;

    dnnl::memory src_layer_mem;
    dnnl::memory src_iter_mem;
    dnnl::memory src_iter_c_mem;
    dnnl::memory dst_layer_mem;
    dnnl::memory dst_iter_mem;
    dnnl::memory dst_iter_c_mem;

    // Configuration
    int64_t num_layers;
    int64_t seq_len;
    int64_t batch;
    int64_t input_size;
    int64_t hidden_size;
    bool has_bias;

    // Weight fingerprint for cache validation
    uint64_t weight_fingerprint;

    // Dimension-only matching for fast cache check
    bool matches_dims(int64_t nl, int64_t s, int64_t b, int64_t i, int64_t h, bool bias) const {
        return num_layers == nl && seq_len == s && batch == b &&
               input_size == i && hidden_size == h && has_bias == bias;
    }

    // Full matching including weight fingerprint
    bool matches_full(int64_t nl, int64_t s, int64_t b, int64_t i, int64_t h, bool bias,
                      uint64_t fingerprint) const {
        return matches_dims(nl, s, b, i, h, bias) && weight_fingerprint == fingerprint;
    }
};

// Compute fingerprint for multi-layer weights
inline uint64_t compute_multilayer_fingerprint(
    const std::vector<const float*>& W_ih_list,
    const std::vector<const float*>& W_hh_list,
    int64_t input_size, int64_t hidden_size
) {
    uint64_t hash = 0;
    for (size_t l = 0; l < W_ih_list.size(); ++l) {
        int64_t ih_size = (l == 0 ? input_size : hidden_size) * 4 * hidden_size;
        int64_t hh_size = hidden_size * 4 * hidden_size;
        hash ^= compute_weight_fingerprint(W_ih_list[l], W_hh_list[l], ih_size, hh_size);
        hash = (hash << 7) | (hash >> 57);  // Rotate to mix layer contributions
    }
    return hash;
}

inline std::shared_ptr<LSTMMultiLayerCachedPrimitive>& get_lstm_multilayer_cached() {
    static thread_local std::shared_ptr<LSTMMultiLayerCachedPrimitive> cached;
    return cached;
}

/**
 * @brief Fused multi-layer LSTM using oneDNN
 *
 * Handles the common case where input_size != hidden_size by:
 * 1. Processing layer 0 separately (single-layer primitive)
 * 2. Fusing layers 1 to num_layers-1 into a single multi-layer primitive
 *
 * For the case where input_size == hidden_size, all layers are fused.
 *
 * @param input Input sequence (seq_len, batch, input_size)
 * @param W_ih_list Input-to-hidden weights for each layer
 * @param W_hh_list Hidden-to-hidden weights for each layer
 * @param bias_list Bias for each layer (or nullptr)
 * @param h0 Initial hidden states (num_layers, batch, hidden)
 * @param c0 Initial cell states (num_layers, batch, hidden)
 * @param output Output sequence (seq_len, batch, hidden)
 * @param h_n Final hidden states (num_layers, batch, hidden)
 * @param c_n Final cell states (num_layers, batch, hidden)
 */
inline bool lstm_multilayer_forward_onednn(
    const float* input,
    const std::vector<const float*>& W_ih_list,
    const std::vector<const float*>& W_hh_list,
    const std::vector<const float*>& bias_list,
    const float* h0,
    const float* c0,
    float* output,
    float* h_n,
    float* c_n,
    int64_t num_layers,
    int64_t seq_len,
    int64_t batch,
    int64_t input_size,
    int64_t hidden_size
) {
    if (num_layers < 1 || W_ih_list.size() != static_cast<size_t>(num_layers)) {
        return false;
    }

    // For single layer, delegate to optimized single-layer kernel
    if (num_layers == 1) {
        return lstm_forward_onednn(
            input, W_ih_list[0], W_hh_list[0], bias_list.empty() ? nullptr : bias_list[0],
            h0, c0, output, h_n, c_n, seq_len, batch, input_size, hidden_size
        );
    }

    // For multi-layer, use sequential single-layer primitives like PyTorch
    // This is faster than fused multi-layer due to better brgemm utilization
    // Use persistent thread-local buffers and per-layer caches to avoid overhead
    try {
        // Thread-local persistent buffers - only reallocate if size increases
        static thread_local std::vector<float> layer_input_buf;
        static thread_local std::vector<float> layer_output_buf;
        static thread_local int64_t cached_buffer_size = 0;

        const int64_t required_size = seq_len * batch * hidden_size;
        if (required_size > cached_buffer_size) {
            layer_input_buf.resize(required_size);
            layer_output_buf.resize(required_size);
            cached_buffer_size = required_size;
        }

        const float* layer_input = input;
        float* layer_output = layer_output_buf.data();

        // Per-layer caches to avoid cache thrashing between layers
        // Each layer has different weight pointers, so sharing a cache causes rebuilds
        static thread_local std::vector<std::shared_ptr<LSTMCachedPrimitive>> layer_caches;
        if (layer_caches.size() < static_cast<size_t>(num_layers)) {
            layer_caches.resize(num_layers);
        }

        for (int64_t l = 0; l < num_layers; ++l) {
            int64_t layer_input_size = (l == 0) ? input_size : hidden_size;

            // Get h0/c0 for this layer
            const float* h0_layer = h0 + l * batch * hidden_size;
            const float* c0_layer = c0 + l * batch * hidden_size;

            // Get output locations for this layer's final states
            float* h_n_layer = h_n + l * batch * hidden_size;
            float* c_n_layer = c_n + l * batch * hidden_size;

            // Output goes to final buffer for last layer, temp buffer otherwise
            float* current_output = (l == num_layers - 1) ? output : layer_output;

            // Use per-layer cache to avoid rebuilds when weight pointers differ
            bool ok = lstm_forward_onednn_with_cache(
                layer_caches[l],
                layer_input,
                W_ih_list[l], W_hh_list[l],
                bias_list.empty() ? nullptr : bias_list[l],
                h0_layer, c0_layer,
                current_output, h_n_layer, c_n_layer,
                seq_len, batch, layer_input_size, hidden_size
            );

            if (!ok) return false;

            // For next layer: input = current output, output = other buffer
            if (l < num_layers - 1) {
                layer_input = current_output;
                layer_output = (current_output == layer_output_buf.data()) ?
                               layer_input_buf.data() : layer_output_buf.data();
            }
        }

        return true;
    } catch (const std::exception& e) {
        // Audit L.3: surface unexpected exceptions in the multi-layer
        // LSTM fast path with type+message. Per-layer fast paths already
        // handle dnnl::error with documented scalar fallbacks; anything
        // reaching this catch is a non-dnnl exception worth surfacing.
        TENZOR_LOG_ERROR("[LSTM multi-layer oneDNN] non-dnnl exception: type={} msg={}",
                         typeid(e).name(), e.what());
        throw;
    } catch (...) {
        // Audit L.3: unknown exception type (not derived from std::exception).
        TENZOR_LOG_ERROR("[LSTM multi-layer oneDNN] unknown exception type; rethrowing");
        throw;
    }

}

// ============================================================================
// Fused Multi-Layer GRU Primitive
// ============================================================================

struct GRUMultiLayerCachedPrimitive {
    // For layer 0 (potentially different input_size)
    std::shared_ptr<GRUCachedPrimitive> layer0_cache;

    // For layers 1+ (fused, all have hidden_size as input)
    dnnl::gru_forward prim;
    dnnl::gru_forward::primitive_desc pd;
    bool has_fused_layers = false;

    dnnl::memory weights_layer_mem;
    dnnl::memory weights_iter_mem;
    dnnl::memory bias_mem;
    dnnl::memory scratchpad_mem;

    dnnl::memory src_layer_mem;
    dnnl::memory src_iter_mem;
    dnnl::memory dst_layer_mem;
    dnnl::memory dst_iter_mem;

    // Configuration
    int64_t num_layers;
    int64_t seq_len;
    int64_t batch;
    int64_t input_size;
    int64_t hidden_size;
    bool has_bias;

    // Weight tracking for cache validation
    std::vector<const void*> w_ih_ptrs;
    std::vector<const void*> w_hh_ptrs;

    bool matches(int64_t nl, int64_t s, int64_t b, int64_t i, int64_t h, bool bias,
                 const std::vector<const float*>& wih_list,
                 const std::vector<const float*>& whh_list) const {
        if (num_layers != nl || seq_len != s || batch != b ||
            input_size != i || hidden_size != h || has_bias != bias) {
            return false;
        }
        if (w_ih_ptrs.size() != wih_list.size() || w_hh_ptrs.size() != whh_list.size()) {
            return false;
        }
        for (size_t l = 0; l < wih_list.size(); ++l) {
            if (w_ih_ptrs[l] != wih_list[l] || w_hh_ptrs[l] != whh_list[l]) {
                return false;
            }
        }
        return true;
    }
};

inline std::shared_ptr<GRUMultiLayerCachedPrimitive>& get_gru_multilayer_cached() {
    static thread_local std::shared_ptr<GRUMultiLayerCachedPrimitive> cached;
    return cached;
}

/**
 * @brief Fused multi-layer GRU using oneDNN
 *
 * Handles the common case where input_size != hidden_size by:
 * 1. Processing layer 0 separately (single-layer primitive)
 * 2. Fusing layers 1 to num_layers-1 into a single multi-layer primitive
 *
 * For the case where input_size == hidden_size, all layers are fused.
 *
 * @param input Input sequence (seq_len, batch, input_size)
 * @param W_ih_list Input-to-hidden weights for each layer
 * @param W_hh_list Hidden-to-hidden weights for each layer
 * @param bias_list Bias for each layer (or nullptr)
 * @param h0 Initial hidden states (num_layers, batch, hidden)
 * @param output Output sequence (seq_len, batch, hidden)
 * @param h_n Final hidden states (num_layers, batch, hidden)
 */
inline bool gru_multilayer_forward_onednn(
    const float* input,
    const std::vector<const float*>& W_ih_list,
    const std::vector<const float*>& W_hh_list,
    const std::vector<const float*>& bias_list,
    const float* h0,
    float* output,
    float* h_n,
    int64_t num_layers,
    int64_t seq_len,
    int64_t batch,
    int64_t input_size,
    int64_t hidden_size
) {
    if (num_layers < 1 || W_ih_list.size() != static_cast<size_t>(num_layers)) {
        return false;
    }

    // For single layer, delegate to optimized single-layer kernel
    if (num_layers == 1) {
        return gru_forward_onednn(
            input, W_ih_list[0], W_hh_list[0], bias_list.empty() ? nullptr : bias_list[0],
            h0, output, h_n, seq_len, batch, input_size, hidden_size
        );
    }

    try {
        auto& engine = get_engine();
        auto& stream = get_stream();
        auto& cached = get_gru_multilayer_cached();

        bool has_bias = !bias_list.empty() && bias_list[0] != nullptr;

        // Check cache validity
        bool need_rebuild = !cached ||
                           !cached->matches(num_layers, seq_len, batch, input_size,
                                           hidden_size, has_bias, W_ih_list, W_hh_list);

        if (need_rebuild) {
            cached = std::make_shared<GRUMultiLayerCachedPrimitive>();
            cached->num_layers = num_layers;
            cached->seq_len = seq_len;
            cached->batch = batch;
            cached->input_size = input_size;
            cached->hidden_size = hidden_size;
            cached->has_bias = has_bias;

            // Store weight pointers for cache validation
            cached->w_ih_ptrs.clear();
            cached->w_hh_ptrs.clear();
            for (size_t l = 0; l < W_ih_list.size(); ++l) {
                cached->w_ih_ptrs.push_back(W_ih_list[l]);
                cached->w_hh_ptrs.push_back(W_hh_list[l]);
            }

            // Determine if we can fuse all layers
            bool can_fuse_all = (input_size == hidden_size);
            int64_t fused_layers = can_fuse_all ? num_layers : (num_layers - 1);

            // Create layer 0 cache if needed (when input_size != hidden_size)
            if (!can_fuse_all) {
                cached->layer0_cache = std::make_shared<GRUCachedPrimitive>();
            }

            // Create fused multi-layer primitive for layers 1+ (or all if can_fuse_all)
            if (fused_layers > 0) {
                cached->has_fused_layers = true;

                // Dimensions for fused layers
                int64_t fused_input_size = can_fuse_all ? input_size : hidden_size;

                dnnl::memory::dims src_layer_dims = {seq_len, batch, fused_input_size};
                dnnl::memory::dims src_iter_dims = {fused_layers, 1, batch, hidden_size};
                dnnl::memory::dims weights_layer_dims = {fused_layers, 1, fused_input_size, 3, hidden_size};
                dnnl::memory::dims weights_iter_dims = {fused_layers, 1, hidden_size, 3, hidden_size};
                dnnl::memory::dims bias_dims = {fused_layers, 1, 3, hidden_size};
                dnnl::memory::dims dst_layer_dims = {seq_len, batch, hidden_size};
                dnnl::memory::dims dst_iter_dims = {fused_layers, 1, batch, hidden_size};

                auto src_layer_md = dnnl::memory::desc(src_layer_dims, dnnl::memory::data_type::f32,
                                                        dnnl::memory::format_tag::tnc);
                auto src_iter_md = dnnl::memory::desc(src_iter_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::ldnc);
                // Use format_tag::any for weights to enable optimized implementation
                auto weights_layer_md = dnnl::memory::desc(weights_layer_dims, dnnl::memory::data_type::f32,
                                                            dnnl::memory::format_tag::any);
                auto weights_iter_md = dnnl::memory::desc(weights_iter_dims, dnnl::memory::data_type::f32,
                                                           dnnl::memory::format_tag::any);
                auto bias_md = dnnl::memory::desc(bias_dims, dnnl::memory::data_type::f32,
                                                   dnnl::memory::format_tag::ldgo);
                auto dst_layer_md = dnnl::memory::desc(dst_layer_dims, dnnl::memory::data_type::f32,
                                                        dnnl::memory::format_tag::tnc);
                auto dst_iter_md = dnnl::memory::desc(dst_iter_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::ldnc);

                cached->pd = dnnl::gru_forward::primitive_desc(
                    engine,
                    dnnl::prop_kind::forward_inference,
                    dnnl::rnn_direction::unidirectional_left2right,
                    src_layer_md, src_iter_md,
                    weights_layer_md, weights_iter_md, bias_md,
                    dst_layer_md, dst_iter_md
                );

                cached->prim = dnnl::gru_forward(cached->pd);

                // Allocate memories
                cached->weights_layer_mem = dnnl::memory(cached->pd.weights_layer_desc(), engine);
                cached->weights_iter_mem = dnnl::memory(cached->pd.weights_iter_desc(), engine);
                cached->bias_mem = dnnl::memory(cached->pd.bias_desc(), engine);
                cached->scratchpad_mem = dnnl::memory(cached->pd.scratchpad_desc(), engine);

                cached->src_layer_mem = dnnl::memory(src_layer_md, engine);
                cached->src_iter_mem = dnnl::memory(src_iter_md, engine);
                cached->dst_layer_mem = dnnl::memory(dst_layer_md, engine);
                cached->dst_iter_mem = dnnl::memory(dst_iter_md, engine);

                // Pack weights for all fused layers
                int64_t start_layer = can_fuse_all ? 0 : 1;
                size_t w_layer_size = fused_input_size * 3 * hidden_size;
                size_t w_iter_size = hidden_size * 3 * hidden_size;

                std::vector<float> w_layer_packed(fused_layers * w_layer_size);
                std::vector<float> w_iter_packed(fused_layers * w_iter_size);
                std::vector<float> bias_packed(fused_layers * 3 * hidden_size);

                for (int64_t l = 0; l < fused_layers; ++l) {
                    int64_t src_layer = start_layer + l;
                    int64_t layer_input = (src_layer == 0) ? input_size : hidden_size;

                    // Reorder weights to ldigo format for this layer
                    reorder_gru_weights_to_ldigo(
                        W_ih_list[src_layer],
                        w_layer_packed.data() + l * w_layer_size,
                        layer_input, hidden_size
                    );
                    reorder_gru_weights_to_ldigo(
                        W_hh_list[src_layer],
                        w_iter_packed.data() + l * w_iter_size,
                        hidden_size, hidden_size
                    );

                    // Pack bias
                    if (has_bias) {
                        std::memcpy(bias_packed.data() + l * 3 * hidden_size,
                                   bias_list[src_layer], 3 * hidden_size * sizeof(float));
                    } else {
                        std::memset(bias_packed.data() + l * 3 * hidden_size,
                                   0, 3 * hidden_size * sizeof(float));
                    }
                }

                // Copy packed weights to oneDNN memory with potential reorder
                auto user_weights_layer_md = dnnl::memory::desc(weights_layer_dims, dnnl::memory::data_type::f32,
                                                                 dnnl::memory::format_tag::ldigo);
                auto user_weights_iter_md = dnnl::memory::desc(weights_iter_dims, dnnl::memory::data_type::f32,
                                                                dnnl::memory::format_tag::ldigo);

                if (cached->pd.weights_layer_desc() != user_weights_layer_md) {
                    auto user_mem = dnnl::memory(user_weights_layer_md, engine, w_layer_packed.data());
                    dnnl::reorder(user_mem, cached->weights_layer_mem).execute(stream, user_mem, cached->weights_layer_mem);
                } else {
                    std::memcpy(cached->weights_layer_mem.get_data_handle(),
                               w_layer_packed.data(), w_layer_packed.size() * sizeof(float));
                }

                if (cached->pd.weights_iter_desc() != user_weights_iter_md) {
                    auto user_mem = dnnl::memory(user_weights_iter_md, engine, w_iter_packed.data());
                    dnnl::reorder(user_mem, cached->weights_iter_mem).execute(stream, user_mem, cached->weights_iter_mem);
                } else {
                    std::memcpy(cached->weights_iter_mem.get_data_handle(),
                               w_iter_packed.data(), w_iter_packed.size() * sizeof(float));
                }

                std::memcpy(cached->bias_mem.get_data_handle(),
                           bias_packed.data(), bias_packed.size() * sizeof(float));

                stream.wait();
            }
        }

        bool can_fuse_all = (input_size == hidden_size);
        int64_t fused_layers = can_fuse_all ? num_layers : (num_layers - 1);

        // Working buffers for intermediate results
        std::vector<float> intermediate_output;
        std::vector<float> intermediate_h;

        const float* fused_input = input;
        const float* fused_h0 = h0;

        // Process layer 0 separately if input_size != hidden_size
        if (!can_fuse_all) {
            intermediate_output.resize(seq_len * batch * hidden_size);
            intermediate_h.resize(batch * hidden_size);

            // Run single-layer GRU for layer 0
            bool layer0_ok = gru_forward_onednn(
                input,
                W_ih_list[0], W_hh_list[0],
                has_bias ? bias_list[0] : nullptr,
                h0,  // First layer's h0
                intermediate_output.data(),
                intermediate_h.data(),
                seq_len, batch, input_size, hidden_size
            );

            if (!layer0_ok) {
                return false;
            }

            // Copy layer 0's final state to output
            std::memcpy(h_n, intermediate_h.data(), batch * hidden_size * sizeof(float));

            // Set up input for fused layers
            fused_input = intermediate_output.data();
            fused_h0 = h0 + batch * hidden_size;  // h0 for layers 1+
        }

        // Execute fused multi-layer primitive
        if (cached->has_fused_layers && fused_layers > 0) {
            // Set data handles
            cached->src_layer_mem.set_data_handle(const_cast<float*>(fused_input));
            cached->src_iter_mem.set_data_handle(const_cast<float*>(fused_h0));
            cached->dst_layer_mem.set_data_handle(output);

            // For output states, we need to handle the offset
            if (can_fuse_all) {
                cached->dst_iter_mem.set_data_handle(h_n);
            } else {
                // States for layers 1+ go after layer 0's state
                cached->dst_iter_mem.set_data_handle(h_n + batch * hidden_size);
            }

            cached->prim.execute(stream, {
                {DNNL_ARG_SRC_LAYER, cached->src_layer_mem},
                {DNNL_ARG_SRC_ITER, cached->src_iter_mem},
                {DNNL_ARG_WEIGHTS_LAYER, cached->weights_layer_mem},
                {DNNL_ARG_WEIGHTS_ITER, cached->weights_iter_mem},
                {DNNL_ARG_BIAS, cached->bias_mem},
                {DNNL_ARG_DST_LAYER, cached->dst_layer_mem},
                {DNNL_ARG_DST_ITER, cached->dst_iter_mem},
                {DNNL_ARG_SCRATCHPAD, cached->scratchpad_mem}
            });

            stream.wait();
        } else if (!can_fuse_all && fused_layers == 0) {
            // Single layer case where layer 0 was processed above
            // Just copy the output
            std::memcpy(output, intermediate_output.data(),
                       seq_len * batch * hidden_size * sizeof(float));
        }

        return true;

    } catch (const dnnl::error& e) {
        if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
            throw std::runtime_error(
                std::string("[GRU multi-layer oneDNN] failed (TENZOR_STRICT_BACKEND=1): ") +
                e.what());
        }
        TENZOR_LOG_WARN("[GRU multi-layer oneDNN] forward failed ({}); using scalar fallback",
                        e.what());
        return false;
    } catch (const std::exception& e) {
        // Audit L.3: surface unexpected non-dnnl exceptions in the GRU
        // multi-layer fast path with type+message.
        TENZOR_LOG_ERROR("[GRU multi-layer oneDNN] non-dnnl exception: type={} msg={}",
                         typeid(e).name(), e.what());
        throw;
    } catch (...) {
        // Audit L.3: unknown exception type (not derived from std::exception).
        TENZOR_LOG_ERROR("[GRU multi-layer oneDNN] unknown exception type in fast path; rethrowing");
        throw;
    }
}

} // namespace rnn_onednn
} // namespace cpu
} // namespace tenzor

#endif // TENZOR_USE_ONEDNN
