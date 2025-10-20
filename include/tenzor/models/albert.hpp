/**
 * @file albert.hpp
 * @brief ALBERT (A Lite BERT) model implementation
 *
 * ALBERT achieves parameter reduction through:
 * 1. Factorized embedding parameterization (V×E + E×H instead of V×H)
 * 2. Cross-layer parameter sharing (reuse same transformer layer)
 * 3. Sentence-Order Prediction (SOP) instead of Next Sentence Prediction (NSP)
 *
 * Reference: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"
 * (Lan et al., 2019)
 */

#pragma once

#include <memory>
#include <string>
#include "../nn/module.hpp"
#include "../nn/layers/embedding.hpp"
#include "../nn/layers/linear.hpp"
#include "../nn/layers/dropout.hpp"
#include "../nn/layers/normalization.hpp"
#include "../nn/layers/transformer.hpp"
#include "../core/tensor.hpp"
#include "../autograd/variable.hpp"

namespace tenzor {
namespace models {

/**
 * @brief Configuration for ALBERT models
 */
struct AlbertConfig {
    int64_t vocab_size = 30000;              ///< Vocabulary size
    int64_t embedding_size = 128;            ///< Factorized embedding dimension (E)
    int64_t hidden_size = 768;               ///< Hidden dimension (H), typically >> embedding_size
    int64_t num_hidden_layers = 12;          ///< Number of layers (reuses same layer)
    int64_t num_attention_heads = 12;        ///< Number of attention heads
    int64_t intermediate_size = 3072;        ///< FFN intermediate dimension
    double hidden_dropout_prob = 0.1;        ///< Hidden layer dropout
    double attention_probs_dropout_prob = 0.1; ///< Attention dropout
    int64_t max_position_embeddings = 512;   ///< Maximum sequence length
    int64_t type_vocab_size = 2;             ///< Number of token types (segments)
    double layer_norm_eps = 1e-12;           ///< Layer norm epsilon
    std::string hidden_act = "gelu";         ///< Activation function

    /**
     * @brief Create ALBERT-base configuration (12M params)
     */
    static AlbertConfig base() {
        return AlbertConfig{};  // Default values are ALBERT-base
    }

    /**
     * @brief Create ALBERT-large configuration (18M params)
     */
    static AlbertConfig large() {
        AlbertConfig config;
        config.hidden_size = 1024;
        config.num_hidden_layers = 24;
        config.num_attention_heads = 16;
        config.intermediate_size = 4096;
        return config;
    }

    /**
     * @brief Create ALBERT-xlarge configuration (60M params)
     */
    static AlbertConfig xlarge() {
        AlbertConfig config;
        config.hidden_size = 2048;
        config.num_hidden_layers = 24;
        config.num_attention_heads = 16;
        config.intermediate_size = 8192;
        return config;
    }

    /**
     * @brief Create ALBERT-xxlarge configuration (233M params)
     */
    static AlbertConfig xxlarge() {
        AlbertConfig config;
        config.hidden_size = 4096;
        config.num_hidden_layers = 12;
        config.num_attention_heads = 64;
        config.intermediate_size = 16384;
        return config;
    }
};

/**
 * @brief ALBERT embeddings with factorized parameterization
 *
 * Instead of V×H embedding matrix, uses:
 * 1. V×E embedding matrix (small)
 * 2. E×H projection matrix
 * This reduces parameters by ~83% when E << H
 *
 * Example: vocab_size=30000, E=128, H=768
 * - BERT: 30000 × 768 = 23.04M parameters
 * - ALBERT: (30000 × 128) + (128 × 768) = 3.94M parameters
 */
class AlbertEmbeddings : public nn::Module {
public:
    /**
     * @brief Construct ALBERT embeddings layer
     *
     * @param config ALBERT configuration
     */
    explicit AlbertEmbeddings(const AlbertConfig& config);

    /**
     * @brief Forward pass through embeddings
     *
     * @param input_ids Token IDs [batch, seq_len]
     * @param token_type_ids Segment IDs [batch, seq_len] (default: all zeros)
     * @param position_ids Position IDs [batch, seq_len] (default: 0, 1, 2, ...)
     * @return Embedded representations [batch, seq_len, hidden_size]
     */
    auto forward(const Variable& input_ids,
                const Variable& token_type_ids = Variable{},
                const Variable& position_ids = Variable{}) -> Variable;

    /**
     * @brief Required by Module base class
     */
    auto forward(const Variable& input) -> Variable override;

private:
    AlbertConfig config_;
    std::shared_ptr<nn::Embedding> word_embeddings_;      ///< V × E (small)
    std::shared_ptr<nn::Embedding> position_embeddings_;   ///< P × E
    std::shared_ptr<nn::Embedding> token_type_embeddings_; ///< T × E
    std::shared_ptr<nn::Linear> embedding_projection_;     ///< E × H (project to hidden size)
    std::shared_ptr<nn::LayerNorm> layer_norm_;
    std::shared_ptr<nn::Dropout> dropout_;

    /**
     * @brief Create default position IDs [0, 1, 2, ..., seq_len-1]
     */
    auto create_position_ids(const Tensor& input_ids) -> Tensor;
};

/**
 * @brief ALBERT encoder with cross-layer parameter sharing
 *
 * Key innovation: Creates ONE transformer layer and applies it N times.
 * This reduces encoder parameters by ~91% compared to BERT.
 *
 * Example: 12 layers
 * - BERT: 12 unique layers × 7M params/layer = 84M params
 * - ALBERT: 1 shared layer × 7M params = 7M params
 */
class AlbertEncoder : public nn::Module {
public:
    /**
     * @brief Construct ALBERT encoder
     *
     * @param config ALBERT configuration
     */
    explicit AlbertEncoder(const AlbertConfig& config);

    /**
     * @brief Forward pass through encoder
     *
     * Applies the same transformer layer multiple times (parameter sharing)
     *
     * @param hidden_states Input embeddings [batch, seq_len, hidden_size]
     * @param attention_mask Attention mask [batch, seq_len] (1 = attend, 0 = ignore)
     * @return Encoded sequence [batch, seq_len, hidden_size]
     */
    auto forward(const Variable& hidden_states,
                const Tensor& attention_mask = Tensor{}) -> Variable;

    /**
     * @brief Required by Module base class
     */
    auto forward(const Variable& input) -> Variable override;

private:
    AlbertConfig config_;
    std::shared_ptr<nn::TransformerEncoderLayer> shared_layer_;  ///< Single shared layer
};

/**
 * @brief ALBERT pooler layer (same as BERT)
 *
 * Pools the [CLS] token representation and applies a linear transformation
 * with tanh activation.
 */
class AlbertPooler : public nn::Module {
public:
    /**
     * @brief Construct ALBERT pooler
     *
     * @param config ALBERT configuration
     */
    explicit AlbertPooler(const AlbertConfig& config);

    /**
     * @brief Forward pass through pooler (required by Module base class)
     *
     * @param hidden_states Encoder output [batch, seq_len, hidden_size]
     * @return Pooled output [batch, hidden_size]
     */
    auto forward(const Variable& hidden_states) -> Variable override;

private:
    std::shared_ptr<nn::Linear> dense_;
};

/**
 * @brief Output structure for ALBERT models
 */
struct AlbertOutput {
    Variable sequence_output;  ///< Token-level representations [batch, seq_len, hidden_size]
    Variable pooled_output;    ///< Sentence representation [batch, hidden_size]
};

/**
 * @brief Base ALBERT model
 *
 * The core ALBERT model that outputs both sequence-level and pooled representations.
 * Uses factorized embeddings and cross-layer parameter sharing.
 *
 * Example usage:
 * ```
 * auto config = AlbertConfig::base();
 * auto albert = AlbertModel(config);
 *
 * Variable input_ids;  // [batch, seq_len]
 * auto outputs = albert.forward(input_ids);
 * auto sequence_output = outputs.sequence_output;  // [batch, seq_len, 768]
 * auto pooled_output = outputs.pooled_output;      // [batch, 768]
 * ```
 */
class AlbertModel : public nn::Module {
public:
    /**
     * @brief Construct ALBERT model
     *
     * @param config ALBERT configuration
     */
    explicit AlbertModel(const AlbertConfig& config);

    /**
     * @brief Forward pass through ALBERT
     *
     * @param input_ids Token IDs [batch, seq_len]
     * @param attention_mask Attention mask [batch, seq_len] (optional)
     * @param token_type_ids Segment IDs [batch, seq_len] (optional)
     * @param position_ids Position IDs [batch, seq_len] (optional)
     * @return AlbertOutput with sequence_output and pooled_output
     */
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{},
                const Variable& token_type_ids = Variable{},
                const Variable& position_ids = Variable{}) -> AlbertOutput;

    /**
     * @brief Required by Module base class (returns sequence_output only)
     */
    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Get model configuration
     */
    auto config() const -> const AlbertConfig& { return config_; }

private:
    AlbertConfig config_;
    std::shared_ptr<AlbertEmbeddings> embeddings_;
    std::shared_ptr<AlbertEncoder> encoder_;
    std::shared_ptr<AlbertPooler> pooler_;
};

/**
 * @brief ALBERT for sequence classification
 *
 * ALBERT model with a classification head on top of the pooled output.
 */
class AlbertForSequenceClassification : public nn::Module {
public:
    /**
     * @brief Construct ALBERT sequence classifier
     *
     * @param config ALBERT configuration
     * @param num_labels Number of classes
     */
    AlbertForSequenceClassification(const AlbertConfig& config, int64_t num_labels);

    /**
     * @brief Forward pass
     *
     * @param input_ids Token IDs [batch, seq_len]
     * @param attention_mask Attention mask [batch, seq_len] (optional)
     * @param token_type_ids Segment IDs [batch, seq_len] (optional)
     * @return Classification logits [batch, num_labels]
     */
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{},
                const Variable& token_type_ids = Variable{}) -> Variable;

    /**
     * @brief Required by Module base class
     */
    auto forward(const Variable& input) -> Variable override;

private:
    AlbertConfig config_;
    int64_t num_labels_;
    std::shared_ptr<AlbertModel> albert_;
    std::shared_ptr<nn::Dropout> dropout_;
    std::shared_ptr<nn::Linear> classifier_;
};

/**
 * @brief ALBERT for token classification
 *
 * ALBERT model with a token-level classification head.
 * Suitable for NER, POS tagging, etc.
 */
class AlbertForTokenClassification : public nn::Module {
public:
    /**
     * @brief Construct ALBERT token classifier
     *
     * @param config ALBERT configuration
     * @param num_labels Number of token labels
     */
    AlbertForTokenClassification(const AlbertConfig& config, int64_t num_labels);

    /**
     * @brief Forward pass
     *
     * @param input_ids Token IDs [batch, seq_len]
     * @param attention_mask Attention mask [batch, seq_len] (optional)
     * @param token_type_ids Segment IDs [batch, seq_len] (optional)
     * @return Token classification logits [batch, seq_len, num_labels]
     */
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{},
                const Variable& token_type_ids = Variable{}) -> Variable;

    /**
     * @brief Required by Module base class
     */
    auto forward(const Variable& input) -> Variable override;

private:
    AlbertConfig config_;
    int64_t num_labels_;
    std::shared_ptr<AlbertModel> albert_;
    std::shared_ptr<nn::Dropout> dropout_;
    std::shared_ptr<nn::Linear> classifier_;
};

/**
 * @brief ALBERT for pre-training with MLM and SOP objectives
 *
 * ALBERT pre-training model with:
 * 1. Masked Language Model (MLM) head
 * 2. Sentence-Order Prediction (SOP) head
 *
 * SOP is harder than NSP: predicts whether two sentences are in correct order
 * (positive: original order, negative: swapped order from same document)
 */
class AlbertForPreTraining : public nn::Module {
public:
    /**
     * @brief Construct ALBERT pre-training model
     *
     * @param config ALBERT configuration
     */
    explicit AlbertForPreTraining(const AlbertConfig& config);

    /**
     * @brief Forward pass
     *
     * @param input_ids Token IDs [batch, seq_len]
     * @param attention_mask Attention mask [batch, seq_len] (optional)
     * @param token_type_ids Segment IDs [batch, seq_len] (optional)
     * @return Tuple of (mlm_logits, sop_logits)
     *         mlm_logits: [batch, seq_len, vocab_size]
     *         sop_logits: [batch, 2] - binary classification
     */
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{},
                const Variable& token_type_ids = Variable{})
        -> std::tuple<Variable, Variable>;

    /**
     * @brief Required by Module base class (returns mlm_logits only)
     */
    auto forward(const Variable& input) -> Variable override;

private:
    AlbertConfig config_;
    std::shared_ptr<AlbertModel> albert_;
    std::shared_ptr<nn::Linear> mlm_head_;     ///< MLM prediction head
    std::shared_ptr<nn::Linear> sop_classifier_; ///< SOP binary classifier
};

} // namespace models
} // namespace tenzor
