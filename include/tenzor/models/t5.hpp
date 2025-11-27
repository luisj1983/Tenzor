/**
 * @file t5.hpp
 * @brief T5 (Text-to-Text Transfer Transformer) model implementation
 *
 * T5 is a full encoder-decoder transformer with key innovations:
 * 1. Relative position bias instead of absolute position embeddings
 * 2. Pre-layer normalization (norm before sublayer)
 * 3. Unified text-to-text framework for all NLP tasks
 * 4. Shared token embeddings between encoder and decoder
 *
 * Reference: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
 * (Raffel et al., 2019)
 */

#pragma once

#include <memory>
#include <string>
#include <cmath>
#include "../nn/module.hpp"
#include "../nn/layers/embedding.hpp"
#include "../nn/layers/linear.hpp"
#include "../nn/layers/dropout.hpp"
#include "../nn/layers/normalization.hpp"
#include "../core/tensor.hpp"
#include "../autograd/variable.hpp"

namespace tenzor {
namespace models {

/**
 * @brief Configuration for T5 models
 */
struct T5Config {
    int64_t vocab_size = 32128;              ///< Vocabulary size (SentencePiece)
    int64_t d_model = 512;                   ///< Hidden size
    int64_t d_kv = 64;                       ///< Key/value dimension per head
    int64_t d_ff = 2048;                     ///< FFN intermediate size
    int64_t num_layers = 6;                  ///< Number of layers (encoder + decoder each)
    int64_t num_heads = 8;                   ///< Number of attention heads
    int64_t relative_attention_num_buckets = 32;  ///< Number of relative position buckets
    int64_t relative_attention_max_distance = 128; ///< Maximum relative distance
    double dropout_rate = 0.1;               ///< Dropout probability
    double layer_norm_epsilon = 1e-6;        ///< Layer norm epsilon
    std::string dense_act_fn = "relu";       ///< Activation function (T5 uses ReLU)
    bool is_gated_act = false;               ///< Use gated activation (T5.1.1)
    bool use_checkpoint = false;             ///< Enable gradient checkpointing for memory savings

    /**
     * @brief Create T5-small configuration (60M params)
     */
    static T5Config small() {
        T5Config config;
        config.d_model = 512;
        config.d_kv = 64;
        config.d_ff = 2048;
        config.num_layers = 6;
        config.num_heads = 8;
        return config;
    }

    /**
     * @brief Create T5-base configuration (220M params)
     */
    static T5Config base() {
        T5Config config;
        config.d_model = 768;
        config.d_kv = 64;
        config.d_ff = 3072;
        config.num_layers = 12;
        config.num_heads = 12;
        return config;
    }

    /**
     * @brief Create T5-large configuration (770M params)
     */
    static T5Config large() {
        T5Config config;
        config.d_model = 1024;
        config.d_kv = 64;
        config.d_ff = 4096;
        config.num_layers = 24;
        config.num_heads = 16;
        return config;
    }

    /**
     * @brief Create T5-xl configuration (3B params)
     */
    static T5Config xl() {
        T5Config config;
        config.d_model = 2048;
        config.d_kv = 128;
        config.d_ff = 5120;
        config.num_layers = 24;
        config.num_heads = 32;
        return config;
    }

    /**
     * @brief Create T5-xxl configuration (11B params)
     */
    static T5Config xxl() {
        T5Config config;
        config.d_model = 4096;
        config.d_kv = 128;
        config.d_ff = 10240;
        config.num_layers = 24;
        config.num_heads = 64;
        return config;
    }
};

/**
 * @brief T5 Attention with relative position bias
 *
 * Key differences from standard attention:
 * 1. No absolute position embeddings
 * 2. Relative position bias added to attention scores
 * 3. First layer computes bias, subsequent layers reuse it
 * 4. Supports both self-attention and cross-attention
 */
class T5Attention : public nn::Module {
public:
    /**
     * @brief Construct T5 attention layer
     *
     * @param config T5 configuration
     * @param has_relative_attention_bias If true, computes relative position bias (first layer only)
     */
    T5Attention(const T5Config& config, bool has_relative_attention_bias);

    /**
     * @brief Forward pass through attention
     *
     * @param hidden_states Query input [batch, seq_len, d_model]
     * @param key_value_states Key/value input for cross-attention (optional)
     * @param attention_mask Attention mask (optional)
     * @param position_bias Pre-computed position bias (optional, reused from first layer)
     * @return Tuple of (attention_output, position_bias)
     */
    auto forward(const Variable& hidden_states,
                const Variable& key_value_states = Variable{},
                const Tensor& attention_mask = Tensor{},
                const Tensor& position_bias = Tensor{})
        -> std::tuple<Variable, Tensor>;

    /**
     * @brief Required by Module base class
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    T5Config config_;
    bool has_relative_attention_bias_;
    std::shared_ptr<nn::Linear> q_proj_;
    std::shared_ptr<nn::Linear> k_proj_;
    std::shared_ptr<nn::Linear> v_proj_;
    std::shared_ptr<nn::Linear> o_proj_;
    std::shared_ptr<nn::Embedding> relative_attention_bias_;  ///< Only in first layer

    /**
     * @brief Compute relative position bucket for a relative position
     */
    auto relative_position_bucket(int64_t relative_position,
                                  int64_t num_buckets,
                                  int64_t max_distance) -> int64_t;

    /**
     * @brief Compute relative position bias matrix
     */
    auto compute_bias(int64_t query_length, int64_t key_length) -> Tensor;
};

/**
 * @brief T5 Layer normalization wrapper
 *
 * T5 uses RMSNorm in some variants, but standard LayerNorm in base T5
 */
class T5LayerNorm : public nn::Module {
public:
    /**
     * @brief Construct T5 layer norm
     *
     * @param d_model Hidden dimension
     * @param eps Epsilon value
     */
    T5LayerNorm(int64_t d_model, double eps);

    /**
     * @brief Forward pass (required by Module base class)
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<nn::LayerNorm> layer_norm_;
};

/**
 * @brief T5 Feed-forward network
 *
 * Standard 2-layer MLP with activation in between
 */
class T5DenseActDense : public nn::Module {
public:
    /**
     * @brief Construct feed-forward network
     *
     * @param config T5 configuration
     */
    explicit T5DenseActDense(const T5Config& config);

    /**
     * @brief Forward pass (required by Module base class)
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    T5Config config_;
    std::shared_ptr<nn::Linear> wi_;  ///< Input projection
    std::shared_ptr<nn::Linear> wo_;  ///< Output projection
    std::shared_ptr<nn::Dropout> dropout_;
};

/**
 * @brief T5 Layer (self-attention + optional cross-attention + feed-forward)
 *
 * Uses pre-layer normalization: norm before sublayer, then add residual
 */
class T5Block : public nn::Module {
public:
    /**
     * @brief Construct T5 block
     *
     * @param config T5 configuration
     * @param has_relative_attention_bias Compute relative position bias (first layer only)
     * @param is_decoder If true, includes cross-attention
     */
    T5Block(const T5Config& config, bool has_relative_attention_bias, bool is_decoder);

    /**
     * @brief Forward pass through block
     *
     * @param hidden_states Input [batch, seq_len, d_model]
     * @param encoder_hidden_states Encoder outputs for cross-attention (decoder only)
     * @param attention_mask Self-attention mask
     * @param encoder_attention_mask Cross-attention mask
     * @param position_bias Self-attention position bias (reused from first layer)
     * @param encoder_position_bias Cross-attention position bias (reused from first layer)
     * @return Tuple of (output, self_attn_position_bias, cross_attn_position_bias)
     */
    auto forward(const Variable& hidden_states,
                const Variable& encoder_hidden_states = Variable{},
                const Tensor& attention_mask = Tensor{},
                const Tensor& encoder_attention_mask = Tensor{},
                const Tensor& position_bias = Tensor{},
                const Tensor& encoder_position_bias = Tensor{})
        -> std::tuple<Variable, Tensor, Tensor>;

    /**
     * @brief Required by Module base class
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    T5Config config_;
    bool is_decoder_;
    std::shared_ptr<T5LayerNorm> layer_norm_1_;
    std::shared_ptr<T5Attention> self_attention_;
    std::shared_ptr<T5LayerNorm> layer_norm_2_;  ///< Cross-attention norm (decoder only)
    std::shared_ptr<T5Attention> cross_attention_; ///< Cross-attention (decoder only)
    std::shared_ptr<T5LayerNorm> layer_norm_ffn_;
    std::shared_ptr<T5DenseActDense> ffn_;
    std::shared_ptr<nn::Dropout> dropout_;
};

/**
 * @brief T5 Encoder stack
 */
class T5Encoder : public nn::Module {
public:
    /**
     * @brief Construct T5 encoder
     *
     * @param config T5 configuration
     * @param shared_embeddings Shared token embeddings
     */
    T5Encoder(const T5Config& config, std::shared_ptr<nn::Embedding> shared_embeddings);

    /**
     * @brief Forward pass through encoder
     *
     * @param input_ids Token IDs [batch, seq_len]
     * @param attention_mask Attention mask [batch, seq_len]
     * @return Encoded sequence [batch, seq_len, d_model]
     */
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{}) -> Variable;

    /**
     * @brief Required by Module base class
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    T5Config config_;
    std::shared_ptr<nn::Embedding> shared_embeddings_;
    std::vector<std::shared_ptr<T5Block>> blocks_;
    std::shared_ptr<T5LayerNorm> final_layer_norm_;
    std::shared_ptr<nn::Dropout> dropout_;
};

/**
 * @brief T5 Decoder stack
 */
class T5Decoder : public nn::Module {
public:
    /**
     * @brief Construct T5 decoder
     *
     * @param config T5 configuration
     * @param shared_embeddings Shared token embeddings
     */
    T5Decoder(const T5Config& config, std::shared_ptr<nn::Embedding> shared_embeddings);

    /**
     * @brief Forward pass through decoder
     *
     * @param decoder_input_ids Decoder token IDs [batch, target_seq_len]
     * @param encoder_hidden_states Encoder outputs [batch, source_seq_len, d_model]
     * @param decoder_attention_mask Decoder attention mask (causal)
     * @param encoder_attention_mask Encoder attention mask
     * @return Decoded sequence [batch, target_seq_len, d_model]
     */
    auto forward(const Variable& decoder_input_ids,
                const Variable& encoder_hidden_states,
                const Tensor& decoder_attention_mask = Tensor{},
                const Tensor& encoder_attention_mask = Tensor{}) -> Variable;

    /**
     * @brief Required by Module base class
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    T5Config config_;
    std::shared_ptr<nn::Embedding> shared_embeddings_;
    std::vector<std::shared_ptr<T5Block>> blocks_;
    std::shared_ptr<T5LayerNorm> final_layer_norm_;
    std::shared_ptr<nn::Dropout> dropout_;

    /**
     * @brief Create causal attention mask for decoder
     */
    auto create_causal_mask(int64_t seq_len, Device device) -> Tensor;
};

/**
 * @brief Output structure for T5 models
 */
struct T5Output {
    Variable encoder_output;  ///< Encoder output [batch, source_seq_len, d_model]
    Variable decoder_output;  ///< Decoder output [batch, target_seq_len, d_model]
};

/**
 * @brief Base T5 model (encoder + decoder)
 *
 * Full text-to-text transformer with shared embeddings.
 *
 * Example usage:
 * ```
 * auto config = T5Config::base();
 * auto t5 = T5Model(config);
 *
 * Variable input_ids;         // [batch, source_seq_len]
 * Variable decoder_input_ids; // [batch, target_seq_len]
 * auto outputs = t5.forward(input_ids, decoder_input_ids);
 * auto encoder_output = outputs.encoder_output;  // [batch, source_seq_len, 768]
 * auto decoder_output = outputs.decoder_output;  // [batch, target_seq_len, 768]
 * ```
 */
class T5Model : public nn::Module {
public:
    /**
     * @brief Construct T5 model
     *
     * @param config T5 configuration
     */
    explicit T5Model(const T5Config& config);

    /**
     * @brief Forward pass through T5
     *
     * @param input_ids Encoder input IDs [batch, source_seq_len]
     * @param decoder_input_ids Decoder input IDs [batch, target_seq_len]
     * @param attention_mask Encoder attention mask
     * @param decoder_attention_mask Decoder attention mask
     * @return T5Output with encoder and decoder outputs
     */
    auto forward(const Variable& input_ids,
                const Variable& decoder_input_ids,
                const Tensor& attention_mask = Tensor{},
                const Tensor& decoder_attention_mask = Tensor{}) -> T5Output;

    /**
     * @brief Required by Module base class (encoder-only mode)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Get model configuration
     */
    auto config() const -> const T5Config& { return config_; }

private:
    T5Config config_;
    std::shared_ptr<nn::Embedding> shared_embeddings_;
    std::shared_ptr<T5Encoder> encoder_;
    std::shared_ptr<T5Decoder> decoder_;
};

/**
 * @brief T5 for conditional generation (text-to-text)
 *
 * T5 model with language modeling head for generation tasks:
 * - Translation
 * - Summarization
 * - Question answering
 * - Classification (via text generation)
 */
class T5ForConditionalGeneration : public nn::Module {
public:
    /**
     * @brief Construct T5 conditional generation model
     *
     * @param config T5 configuration
     */
    explicit T5ForConditionalGeneration(const T5Config& config);

    /**
     * @brief Forward pass
     *
     * @param input_ids Encoder input IDs [batch, source_seq_len]
     * @param decoder_input_ids Decoder input IDs [batch, target_seq_len]
     * @param attention_mask Encoder attention mask
     * @param decoder_attention_mask Decoder attention mask
     * @return Logits [batch, target_seq_len, vocab_size]
     */
    auto forward(const Variable& input_ids,
                const Variable& decoder_input_ids,
                const Tensor& attention_mask = Tensor{},
                const Tensor& decoder_attention_mask = Tensor{}) -> Variable;

    /**
     * @brief Required by Module base class
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Generate text autoregressively
     *
     * @param input_ids Encoder input [batch, source_seq_len]
     * @param max_length Maximum generation length
     * @param temperature Sampling temperature (default: 1.0)
     * @return Generated token IDs [batch, max_length]
     */
    auto generate(const Variable& input_ids,
                 int64_t max_length,
                 double temperature = 1.0) -> Tensor;

private:
    T5Config config_;
    std::shared_ptr<T5Model> t5_;
    std::shared_ptr<nn::Linear> lm_head_;
};

} // namespace models
} // namespace tenzor
