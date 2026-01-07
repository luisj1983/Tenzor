/**
 * @file rnn_onednn.hpp
 * @brief oneDNN-accelerated LSTM and GRU primitives
 *
 * Uses oneDNN's highly optimized RNN primitives with LRU caching
 * to match PyTorch's MKL-DNN performance.
 */

#pragma once

#ifdef TENZOR_USE_ONEDNN

#include <dnnl.hpp>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <list>
#include <vector>

namespace tenzor {
namespace cpu {
namespace rnn_onednn {

// Thread-local oneDNN engine and stream
static thread_local dnnl::engine g_rnn_engine(dnnl::engine::kind::cpu, 0);
static thread_local dnnl::stream g_rnn_stream(g_rnn_engine);

// ============================================================================
// LSTM Primitive Cache
// ============================================================================

struct LSTMCacheKey {
    int64_t seq_len;
    int64_t batch;
    int64_t input_size;
    int64_t hidden_size;
    bool has_bias;

    bool operator==(const LSTMCacheKey& other) const {
        return seq_len == other.seq_len && batch == other.batch &&
               input_size == other.input_size && hidden_size == other.hidden_size &&
               has_bias == other.has_bias;
    }
};

struct LSTMCacheKeyHash {
    size_t operator()(const LSTMCacheKey& k) const {
        size_t h = std::hash<int64_t>{}(k.seq_len);
        h ^= std::hash<int64_t>{}(k.batch) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.input_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.hidden_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(k.has_bias) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct LSTMCachedPrimitive {
    dnnl::lstm_forward prim;
    dnnl::lstm_forward::primitive_desc pd;

    // Memory descriptors - user format
    dnnl::memory::desc src_layer_md;
    dnnl::memory::desc src_iter_md;
    dnnl::memory::desc src_iter_c_md;
    dnnl::memory::desc dst_layer_md;
    dnnl::memory::desc dst_iter_md;
    dnnl::memory::desc dst_iter_c_md;

    // Weights in oneDNN's preferred format
    dnnl::memory weights_layer_mem;
    dnnl::memory weights_iter_mem;
    dnnl::memory bias_mem;

    // Scratchpad
    dnnl::memory scratchpad_mem;

    // Dimensions for verification
    int64_t input_size;
    int64_t hidden_size;

    bool weights_cached = false;
};

static constexpr size_t LSTM_CACHE_SIZE = 16;

class LSTMPrimitiveCache {
public:
    std::shared_ptr<LSTMCachedPrimitive> get(const LSTMCacheKey& key) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            lru_list_.remove(key);
            lru_list_.push_front(key);
            return it->second;
        }
        return nullptr;
    }

    void put(const LSTMCacheKey& key, std::shared_ptr<LSTMCachedPrimitive> value) {
        if (cache_.size() >= LSTM_CACHE_SIZE) {
            auto evict_key = lru_list_.back();
            lru_list_.pop_back();
            cache_.erase(evict_key);
        }
        cache_[key] = value;
        lru_list_.push_front(key);
    }

private:
    std::unordered_map<LSTMCacheKey, std::shared_ptr<LSTMCachedPrimitive>, LSTMCacheKeyHash> cache_;
    std::list<LSTMCacheKey> lru_list_;
};

static thread_local LSTMPrimitiveCache g_lstm_cache;

/**
 * @brief Reorder weights from PyTorch format to oneDNN ldigo format
 *
 * PyTorch LSTM weights: (4*hidden, input_size) with gate order [i, f, g, o]
 * oneDNN LSTM weights: ldigo format (layers, directions, input, gates, output)
 *
 * The key insight is that PyTorch stores weights as:
 *   W[gate * hidden + h, i] = weight for gate 'gate', hidden unit 'h', input 'i'
 *
 * oneDNN ldigo stores as:
 *   W[l, d, i, g, o] where l=layer, d=direction, i=input, g=gate, o=output(hidden)
 */
inline void reorder_weights_to_ldigo(
    const float* src,           // PyTorch format: (4*hidden, in_features)
    float* dst,                 // oneDNN ldigo: (1, 1, in_features, 4, hidden)
    int64_t in_features,
    int64_t hidden_size
) {
    // ldigo layout: dst[i * 4 * hidden + g * hidden + h] = src[(g * hidden + h) * in_features + i]
    for (int64_t i = 0; i < in_features; ++i) {
        for (int64_t g = 0; g < 4; ++g) {
            for (int64_t h = 0; h < hidden_size; ++h) {
                int64_t src_idx = (g * hidden_size + h) * in_features + i;
                int64_t dst_idx = i * 4 * hidden_size + g * hidden_size + h;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

/**
 * @brief oneDNN-accelerated LSTM forward pass
 *
 * NOTE: Currently disabled due to performance issues. The oneDNN LSTM primitive
 * is running ~30x slower than expected, likely due to memory object creation
 * overhead on each call. Using SIMD fallback until this is optimized.
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
    // Disabled: oneDNN LSTM currently ~30x slower than SIMD implementation
    // TODO: Investigate memory object creation overhead and optimize
    (void)input; (void)W_ih; (void)W_hh; (void)bias;
    (void)h0; (void)c0; (void)output; (void)h_n; (void)c_n;
    (void)seq_len; (void)batch; (void)input_size; (void)hidden_size;
    return false;

#if 0  // Disabled implementation
    try {
        auto& engine = g_rnn_engine;
        auto& stream = g_rnn_stream;

        // Create cache key
        LSTMCacheKey key{seq_len, batch, input_size, hidden_size, bias != nullptr};

        // Check cache
        auto cached = g_lstm_cache.get(key);

        if (!cached) {
            cached = std::make_shared<LSTMCachedPrimitive>();
            cached->input_size = input_size;
            cached->hidden_size = hidden_size;

            // Define dimensions
            dnnl::memory::dims src_layer_dims = {seq_len, batch, input_size};
            dnnl::memory::dims src_iter_dims = {1, 1, batch, hidden_size};
            dnnl::memory::dims src_iter_c_dims = {1, 1, batch, hidden_size};
            dnnl::memory::dims weights_layer_dims = {1, 1, input_size, 4, hidden_size};
            dnnl::memory::dims weights_iter_dims = {1, 1, hidden_size, 4, hidden_size};
            dnnl::memory::dims bias_dims = {1, 1, 4, hidden_size};
            dnnl::memory::dims dst_layer_dims = {seq_len, batch, hidden_size};
            dnnl::memory::dims dst_iter_dims = {1, 1, batch, hidden_size};
            dnnl::memory::dims dst_iter_c_dims = {1, 1, batch, hidden_size};

            // User-side memory descriptors
            cached->src_layer_md = dnnl::memory::desc(src_layer_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::tnc);
            cached->src_iter_md = dnnl::memory::desc(src_iter_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::ldnc);
            cached->src_iter_c_md = dnnl::memory::desc(src_iter_c_dims, dnnl::memory::data_type::f32,
                                                        dnnl::memory::format_tag::ldnc);
            cached->dst_layer_md = dnnl::memory::desc(dst_layer_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::tnc);
            cached->dst_iter_md = dnnl::memory::desc(dst_iter_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::ldnc);
            cached->dst_iter_c_md = dnnl::memory::desc(dst_iter_c_dims, dnnl::memory::data_type::f32,
                                                        dnnl::memory::format_tag::ldnc);

            // Weight descriptors - use ldigo format
            auto weights_layer_md = dnnl::memory::desc(weights_layer_dims, dnnl::memory::data_type::f32,
                                                        dnnl::memory::format_tag::ldigo);
            auto weights_iter_md = dnnl::memory::desc(weights_iter_dims, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::ldigo);
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

            // Allocate weight memories using primitive's preferred format
            cached->weights_layer_mem = dnnl::memory(cached->pd.weights_layer_desc(), engine);
            cached->weights_iter_mem = dnnl::memory(cached->pd.weights_iter_desc(), engine);
            cached->bias_mem = dnnl::memory(cached->pd.bias_desc(), engine);

            // Allocate scratchpad
            cached->scratchpad_mem = dnnl::memory(cached->pd.scratchpad_desc(), engine);

            g_lstm_cache.put(key, cached);
        }

        // Reorder weights if not cached
        if (!cached->weights_cached) {
            // Allocate temporary buffers for ldigo format
            std::vector<float> w_layer_ldigo(input_size * 4 * hidden_size);
            std::vector<float> w_iter_ldigo(hidden_size * 4 * hidden_size);
            std::vector<float> bias_ldgo(4 * hidden_size, 0.0f);

            // Reorder W_ih to ldigo
            reorder_weights_to_ldigo(W_ih, w_layer_ldigo.data(), input_size, hidden_size);

            // Reorder W_hh to ldigo
            reorder_weights_to_ldigo(W_hh, w_iter_ldigo.data(), hidden_size, hidden_size);

            // Copy bias (already in correct format: 4*hidden)
            if (bias != nullptr) {
                std::memcpy(bias_ldgo.data(), bias, 4 * hidden_size * sizeof(float));
            }

            // Create user memory objects with ldigo format
            dnnl::memory::dims weights_layer_dims = {1, 1, input_size, 4, hidden_size};
            dnnl::memory::dims weights_iter_dims = {1, 1, hidden_size, 4, hidden_size};
            dnnl::memory::dims bias_dims = {1, 1, 4, hidden_size};

            auto user_weights_layer_md = dnnl::memory::desc(weights_layer_dims, dnnl::memory::data_type::f32,
                                                             dnnl::memory::format_tag::ldigo);
            auto user_weights_iter_md = dnnl::memory::desc(weights_iter_dims, dnnl::memory::data_type::f32,
                                                            dnnl::memory::format_tag::ldigo);
            auto user_bias_md = dnnl::memory::desc(bias_dims, dnnl::memory::data_type::f32,
                                                    dnnl::memory::format_tag::ldgo);

            auto user_weights_layer_mem = dnnl::memory(user_weights_layer_md, engine, w_layer_ldigo.data());
            auto user_weights_iter_mem = dnnl::memory(user_weights_iter_md, engine, w_iter_ldigo.data());
            auto user_bias_mem = dnnl::memory(user_bias_md, engine, bias_ldgo.data());

            // Reorder to primitive's preferred format if needed
            if (cached->pd.weights_layer_desc() != user_weights_layer_md) {
                dnnl::reorder(user_weights_layer_mem, cached->weights_layer_mem)
                    .execute(stream, user_weights_layer_mem, cached->weights_layer_mem);
            } else {
                std::memcpy(cached->weights_layer_mem.get_data_handle(),
                           w_layer_ldigo.data(), w_layer_ldigo.size() * sizeof(float));
            }

            if (cached->pd.weights_iter_desc() != user_weights_iter_md) {
                dnnl::reorder(user_weights_iter_mem, cached->weights_iter_mem)
                    .execute(stream, user_weights_iter_mem, cached->weights_iter_mem);
            } else {
                std::memcpy(cached->weights_iter_mem.get_data_handle(),
                           w_iter_ldigo.data(), w_iter_ldigo.size() * sizeof(float));
            }

            if (cached->pd.bias_desc() != user_bias_md) {
                dnnl::reorder(user_bias_mem, cached->bias_mem)
                    .execute(stream, user_bias_mem, cached->bias_mem);
            } else {
                std::memcpy(cached->bias_mem.get_data_handle(),
                           bias_ldgo.data(), bias_ldgo.size() * sizeof(float));
            }

            stream.wait();
            cached->weights_cached = true;
        }

        // Create input/output memories
        auto src_layer_mem = dnnl::memory(cached->src_layer_md, engine, const_cast<float*>(input));
        auto dst_layer_mem = dnnl::memory(cached->dst_layer_md, engine, output);

        // Initial states
        auto src_iter_mem = dnnl::memory(cached->src_iter_md, engine, const_cast<float*>(h0));
        auto src_iter_c_mem = dnnl::memory(cached->src_iter_c_md, engine, const_cast<float*>(c0));

        // Final states
        auto dst_iter_mem = dnnl::memory(cached->dst_iter_md, engine, h_n);
        auto dst_iter_c_mem = dnnl::memory(cached->dst_iter_c_md, engine, c_n);

        // Execute
        cached->prim.execute(stream, {
            {DNNL_ARG_SRC_LAYER, src_layer_mem},
            {DNNL_ARG_SRC_ITER, src_iter_mem},
            {DNNL_ARG_SRC_ITER_C, src_iter_c_mem},
            {DNNL_ARG_WEIGHTS_LAYER, cached->weights_layer_mem},
            {DNNL_ARG_WEIGHTS_ITER, cached->weights_iter_mem},
            {DNNL_ARG_BIAS, cached->bias_mem},
            {DNNL_ARG_DST_LAYER, dst_layer_mem},
            {DNNL_ARG_DST_ITER, dst_iter_mem},
            {DNNL_ARG_DST_ITER_C, dst_iter_c_mem},
            {DNNL_ARG_SCRATCHPAD, cached->scratchpad_mem}
        });

        stream.wait();
        return true;

    } catch (...) {
        return false;
    }
#endif  // Disabled implementation
}

/**
 * @brief oneDNN-accelerated GRU forward pass
 */
inline bool gru_forward_onednn(
    const float* input,
    const float* W_ih,
    const float* W_hh,
    const float* bias,
    const float* h0,
    float* output,
    float* h_n,
    int64_t seq_len,
    int64_t batch,
    int64_t input_size,
    int64_t hidden_size
) {
    // GRU oneDNN implementation - TODO
    // For now, fall back to SIMD
    return false;
}

} // namespace rnn_onednn
} // namespace cpu
} // namespace tenzor

#endif // TENZOR_USE_ONEDNN
