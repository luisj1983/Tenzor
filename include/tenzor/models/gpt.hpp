/**
 * @file gpt.hpp
 * @brief GPT (Generative Pre-trained Transformer) models for language modeling
 *
 * Implements GPT-2 and GPT-3 architectures with causal (autoregressive) attention.
 * Includes text generation utilities with various sampling strategies.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <random>
#include "../nn/module.hpp"
#include "../nn/layers/linear.hpp"
#include "../nn/layers/embedding.hpp"
#include "../nn/layers/dropout.hpp"
#include "../nn/layers/normalization.hpp"
#include "../nn/layers/attention.hpp"

namespace tenzor {
namespace models {

/**
 * @brief Configuration for GPT-2 model.
 *
 * Contains all hyperparameters needed to construct a GPT-2 model.
 * Default values match GPT-2 small (117M parameters).
 */
struct GPT2Config {
    int64_t vocab_size = 50257;          ///< Size of vocabulary
    int64_t n_positions = 1024;          ///< Maximum sequence length (context length)
    int64_t n_embd = 768;                ///< Hidden size (embedding dimension)
    int64_t n_layer = 12;                ///< Number of transformer layers
    int64_t n_head = 12;                 ///< Number of attention heads
    int64_t n_inner = 3072;              ///< FFN intermediate size (typically 4 * n_embd)
    double attn_pdrop = 0.1;             ///< Attention dropout probability
    double embd_pdrop = 0.1;             ///< Embedding dropout probability
    double resid_pdrop = 0.1;            ///< Residual dropout probability
    double layer_norm_epsilon = 1e-5;    ///< Layer norm epsilon
    std::string activation = "gelu";      ///< Activation function ("gelu" or "relu")

    /**
     * @brief Predefined configurations for different GPT-2 sizes.
     */
    static auto gpt2_small() -> GPT2Config {
        return GPT2Config{};  // 117M params
    }

    static auto gpt2_medium() -> GPT2Config {
        GPT2Config config;
        config.n_embd = 1024;
        config.n_layer = 24;
        config.n_head = 16;
        config.n_inner = 4096;
        return config;  // 345M params
    }

    static auto gpt2_large() -> GPT2Config {
        GPT2Config config;
        config.n_embd = 1280;
        config.n_layer = 36;
        config.n_head = 20;
        config.n_inner = 5120;
        return config;  // 774M params
    }

    static auto gpt2_xl() -> GPT2Config {
        GPT2Config config;
        config.n_embd = 1600;
        config.n_layer = 48;
        config.n_head = 25;
        config.n_inner = 6400;
        return config;  // 1.5B params
    }
};

/**
 * @brief Configuration for GPT-3 model.
 *
 * Extends GPT2Config with GPT-3 specific parameters.
 */
struct GPT3Config : public GPT2Config {
    /**
     * @brief Predefined configurations for different GPT-3 sizes.
     */
    static auto gpt3_small() -> GPT3Config {
        GPT3Config config;
        config.n_embd = 768;
        config.n_layer = 12;
        config.n_head = 12;
        config.n_inner = 3072;
        return config;  // 125M params
    }

    static auto gpt3_medium() -> GPT3Config {
        GPT3Config config;
        config.n_embd = 1024;
        config.n_layer = 24;
        config.n_head = 16;
        config.n_inner = 4096;
        return config;  // 350M params
    }

    static auto gpt3_large() -> GPT3Config {
        GPT3Config config;
        config.n_embd = 1536;
        config.n_layer = 24;
        config.n_head = 16;
        config.n_inner = 6144;
        return config;  // 760M params
    }

    static auto gpt3_xl() -> GPT3Config {
        GPT3Config config;
        config.n_embd = 2048;
        config.n_layer = 24;
        config.n_head = 24;
        config.n_inner = 8192;
        return config;  // 1.3B params
    }

    static auto gpt3_2_7b() -> GPT3Config {
        GPT3Config config;
        config.n_embd = 2560;
        config.n_layer = 32;
        config.n_head = 32;
        config.n_inner = 10240;
        return config;  // 2.7B params
    }

    static auto gpt3_6_7b() -> GPT3Config {
        GPT3Config config;
        config.n_embd = 4096;
        config.n_layer = 32;
        config.n_head = 32;
        config.n_inner = 16384;
        return config;  // 6.7B params
    }

    static auto gpt3_13b() -> GPT3Config {
        GPT3Config config;
        config.n_embd = 5120;
        config.n_layer = 40;
        config.n_head = 40;
        config.n_inner = 20480;
        return config;  // 13B params
    }

    static auto gpt3_175b() -> GPT3Config {
        GPT3Config config;
        config.n_embd = 12288;
        config.n_layer = 96;
        config.n_head = 96;
        config.n_inner = 49152;
        return config;  // 175B params
    }
};

/**
 * @brief GPT embeddings layer.
 *
 * Combines token embeddings and learned positional embeddings.
 * Applies dropout to the sum.
 */
class GPTEmbeddings : public nn::Module {
public:
    /**
     * @brief Construct GPT embeddings.
     *
     * @param config GPT configuration
     */
    explicit GPTEmbeddings(const GPT2Config& config);

    /**
     * @brief Forward pass through embeddings.
     *
     * @param input_ids Token IDs of shape (batch, seq_len)
     * @param position_ids Optional position IDs of shape (batch, seq_len)
     *                     If not provided, uses sequential positions 0, 1, 2, ...
     * @return Embedded tokens of shape (batch, seq_len, n_embd)
     */
    auto forward(const Variable& input_ids, const Variable& position_ids = Variable{}) -> Variable;

    /**
     * @brief Default forward (uses input_ids only).
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /// @brief Access the token embedding submodule (used to tie the LM head).
    auto token_embedding() const -> std::shared_ptr<nn::Embedding> { return token_embedding_; }

private:
    std::shared_ptr<nn::Embedding> token_embedding_;      ///< Token embeddings
    std::shared_ptr<nn::Embedding> position_embedding_;   ///< Position embeddings (learned)
    std::shared_ptr<nn::Dropout> dropout_;                ///< Embedding dropout
    int64_t n_positions_;                                  ///< Maximum positions
};

/**
 * @brief GPT decoder layer (transformer block).
 *
 * Implements a single GPT transformer block with:
 * 1. Layer normalization (pre-norm)
 * 2. Causal self-attention
 * 3. Residual connection
 * 4. Layer normalization (pre-norm)
 * 5. Feed-forward network
 * 6. Residual connection
 *
 * Uses pre-norm architecture (normalize before attention/FFN) as in GPT-2/3.
 */
class GPTDecoderLayer : public nn::Module {
public:
    /**
     * @brief Construct GPT decoder layer.
     *
     * @param config GPT configuration
     */
    explicit GPTDecoderLayer(const GPT2Config& config);

    /**
     * @brief Forward pass through decoder layer.
     *
     * @param hidden_states Hidden states of shape (batch, seq_len, n_embd)
     * @param attention_mask Optional attention mask (causal mask)
     * @param use_cache If true, return attention cache for generation
     * @return Output hidden states of shape (batch, seq_len, n_embd)
     */
    auto forward(const Variable& hidden_states,
                const Tensor& attention_mask = Tensor{},
                bool use_cache = false) -> Variable;

    /**
     * @brief Default forward.
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<nn::LayerNorm> ln_1_;           ///< First layer norm (before attention)
    std::shared_ptr<nn::MultiheadAttention> attn_;  ///< Causal self-attention
    std::shared_ptr<nn::LayerNorm> ln_2_;           ///< Second layer norm (before FFN)
    std::shared_ptr<nn::Linear> mlp_fc_;            ///< FFN first linear layer
    std::shared_ptr<nn::Linear> mlp_proj_;          ///< FFN second linear layer
    std::shared_ptr<nn::Dropout> dropout_;          ///< Residual dropout
    std::string activation_;                         ///< Activation function name

    /**
     * @brief Apply activation function (GELU or ReLU).
     */
    auto apply_activation(const Variable& x) const -> Variable;
};

/**
 * @brief GPT-2 base model (returns hidden states).
 *
 * The core GPT-2 transformer model without the language modeling head.
 * Useful for feature extraction or as a base for custom heads.
 */
class GPT2Model : public nn::Module {
public:
    /**
     * @brief Construct GPT-2 model.
     *
     * @param config GPT-2 configuration
     */
    explicit GPT2Model(const GPT2Config& config);

    /**
     * @brief Forward pass through GPT-2.
     *
     * @param input_ids Token IDs of shape (batch, seq_len)
     * @param position_ids Optional position IDs of shape (batch, seq_len)
     * @param attention_mask Optional attention mask for padding
     * @return Hidden states of shape (batch, seq_len, n_embd)
     */
    auto forward(const Variable& input_ids,
                const Variable& position_ids = Variable{},
                const Tensor& attention_mask = Tensor{}) -> Variable;

    /**
     * @brief Default forward (uses input_ids only).
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Get model configuration.
     */
    auto config() const -> const GPT2Config& { return config_; }

    /// Load pretrained weights via ModelHub (audit H4). See AlbertModel.
    auto load_pretrained(const std::string& path, bool strict = true) -> void;

protected:
    GPT2Config config_;                                           ///< Model configuration
    std::shared_ptr<GPTEmbeddings> embeddings_;                  ///< Token + position embeddings
    std::vector<std::shared_ptr<GPTDecoderLayer>> layers_;       ///< Stack of decoder layers
    std::shared_ptr<nn::LayerNorm> ln_f_;                        ///< Final layer norm
    std::shared_ptr<nn::Dropout> dropout_;                       ///< Output dropout

    /**
     * @brief Create causal attention mask for autoregressive generation.
     *
     * @param seq_len Sequence length
     * @param device Device to create mask on
     * @param dtype Data type for the mask (should match hidden states dtype)
     * @return Causal mask of shape (seq_len, seq_len)
     */
    auto create_causal_attention_mask(int64_t seq_len, Device device, DType dtype) const -> Tensor;
};

/**
 * @brief GPT-2 Language Model Head.
 *
 * GPT-2 model with a linear layer projecting to vocabulary logits.
 * Used for text generation and language modeling.
 */
class GPT2LMHeadModel : public GPT2Model {
public:
    /**
     * @brief Construct GPT-2 LM head model.
     *
     * @param config GPT-2 configuration
     */
    explicit GPT2LMHeadModel(const GPT2Config& config);

    /**
     * @brief Forward pass returning logits over vocabulary.
     *
     * @param input_ids Token IDs of shape (batch, seq_len)
     * @param position_ids Optional position IDs
     * @param attention_mask Optional attention mask
     * @return Logits of shape (batch, seq_len, vocab_size)
     */
    auto forward(const Variable& input_ids,
                const Variable& position_ids = Variable{},
                const Tensor& attention_mask = Tensor{}) -> Variable;

    /**
     * @brief Default forward.
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<nn::Linear> lm_head_;  ///< Linear layer to vocabulary
};

/**
 * @brief GPT-3 model (architecture similar to GPT-2, scaled up).
 *
 * GPT-3 uses the same architecture as GPT-2 but with larger scales.
 * This class is primarily for architecture compatibility; training requires
 * massive compute and data.
 */
class GPT3Model : public GPT2Model {
public:
    /**
     * @brief Construct GPT-3 model.
     *
     * @param config GPT-3 configuration
     */
    explicit GPT3Model(const GPT3Config& config);
};

/**
 * @brief GPT-3 Language Model Head.
 */
class GPT3LMHeadModel : public GPT2LMHeadModel {
public:
    /**
     * @brief Construct GPT-3 LM head model.
     *
     * @param config GPT-3 configuration
     */
    explicit GPT3LMHeadModel(const GPT3Config& config);
};

/**
 * @brief Text generation strategy.
 */
enum class GenerationStrategy {
    Greedy,      ///< Always select highest probability token
    TopK,        ///< Sample from top K tokens
    TopP,        ///< Nucleus sampling (sample from smallest set with cumulative prob >= p)
    BeamSearch   ///< Beam search decoding
};

/**
 * @brief Configuration for text generation.
 */
struct GenerationConfig {
    int64_t max_length = 50;                     ///< Maximum generation length
    int64_t min_length = 0;                      ///< Minimum generation length
    GenerationStrategy strategy = GenerationStrategy::Greedy;
    double temperature = 1.0;                     ///< Softmax temperature (higher = more random)
    int64_t top_k = 50;                          ///< Top-K sampling parameter
    double top_p = 0.9;                          ///< Top-P (nucleus) sampling parameter
    int64_t num_beams = 4;                       ///< Number of beams for beam search
    int64_t eos_token_id = -1;                   ///< End-of-sequence token ID (-1 to disable)
    int64_t pad_token_id = -1;                   ///< Padding token ID
    bool do_sample = true;                       ///< Whether to use sampling (vs deterministic)
    int64_t seed = 42;                           ///< Random seed for reproducibility
};

/**
 * @brief Text generator with various sampling strategies.
 *
 * Provides utilities for generating text from GPT models using different
 * decoding strategies (greedy, top-k, top-p, beam search).
 */
class TextGenerator {
public:
    /**
     * @brief Construct text generator.
     *
     * @param model GPT language model
     * @param config Generation configuration
     */
    explicit TextGenerator(GPT2LMHeadModel& model, const GenerationConfig& config = GenerationConfig{});

    /**
     * @brief Generate text from input token IDs.
     *
     * @param input_ids Starting token IDs (prompt) of shape (batch, seq_len)
     * @return Generated token IDs of shape (batch, max_length)
     */
    auto generate(const Tensor& input_ids) -> Tensor;

    /**
     * @brief Greedy decoding: always select highest probability token.
     *
     * @param input_ids Starting token IDs
     * @return Generated token IDs
     */
    auto greedy_search(const Tensor& input_ids) -> Tensor;

    /**
     * @brief Top-K sampling: sample from top K most likely tokens.
     *
     * @param input_ids Starting token IDs
     * @param top_k Number of top tokens to consider
     * @param temperature Softmax temperature
     * @return Generated token IDs
     */
    auto top_k_sampling(const Tensor& input_ids, int64_t top_k, double temperature = 1.0) -> Tensor;

    /**
     * @brief Top-P (nucleus) sampling: sample from smallest set with cumulative prob >= p.
     *
     * @param input_ids Starting token IDs
     * @param top_p Cumulative probability threshold
     * @param temperature Softmax temperature
     * @return Generated token IDs
     */
    auto top_p_sampling(const Tensor& input_ids, double top_p, double temperature = 1.0) -> Tensor;

    /**
     * @brief Beam search decoding: maintain top-k hypotheses at each step.
     *
     * @param input_ids Starting token IDs
     * @param num_beams Number of beams to maintain
     * @return Generated token IDs (best hypothesis)
     */
    auto beam_search(const Tensor& input_ids, int64_t num_beams) -> Tensor;

private:
    GPT2LMHeadModel& model_;                    ///< Reference to GPT model
    GenerationConfig config_;                    ///< Generation configuration
    std::mt19937 rng_;                          ///< Random number generator

    /**
     * @brief Apply temperature scaling to logits.
     */
    auto apply_temperature(const Variable& logits, double temperature) const -> Variable;

    /**
     * @brief Convert logits to probabilities with temperature.
     */
    auto logits_to_probs(const Variable& logits, double temperature) const -> Variable;

    /**
     * @brief Sample from categorical distribution.
     */
    auto sample_from_probs(const Tensor& probs) -> int64_t;

    /**
     * @brief Get top-K indices and values.
     */
    auto top_k_filter(const Tensor& logits, int64_t k) const -> std::pair<Tensor, Tensor>;

    /**
     * @brief Get top-P (nucleus) indices and values.
     */
    auto top_p_filter(const Tensor& probs, double p) const -> std::pair<Tensor, Tensor>;
};

} // namespace models
} // namespace tenzor
