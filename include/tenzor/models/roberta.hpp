/**
 * @file roberta.hpp
 * @brief RoBERTa (Robustly Optimized BERT) model family
 *
 * Implements RoBERTa architecture which improves upon BERT through:
 * - Removal of Next Sentence Prediction (NSP) objective
 * - Dynamic masking (performed during data loading, not model architecture)
 * - Larger byte-level BPE vocabulary (50265 tokens)
 * - Training on full sentences without segment embeddings
 * - Different layer normalization epsilon
 *
 * Reference: "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
 * (Liu et al., 2019)
 */

#pragma once

#include <memory>
#include <string>
#include "bert.hpp"
#include "../nn/module.hpp"
#include "../core/tensor.hpp"
#include "../autograd/variable.hpp"

namespace tenzor {
namespace models {

/**
 * @brief Configuration for RoBERTa models
 *
 * Key differences from BERT:
 * - vocab_size: 50265 (byte-level BPE)
 * - type_vocab_size: 1 (no segment embeddings)
 * - layer_norm_eps: 1e-5 (vs BERT's 1e-12)
 * - max_position_embeddings: 514 (slightly larger)
 */
struct RobertaConfig {
    int64_t vocab_size = 50265;              ///< Byte-level BPE vocabulary
    int64_t hidden_size = 768;               ///< Hidden dimension
    int64_t num_hidden_layers = 12;          ///< Number of encoder layers
    int64_t num_attention_heads = 12;        ///< Number of attention heads
    int64_t intermediate_size = 3072;        ///< FFN intermediate dimension
    double hidden_dropout_prob = 0.1;        ///< Hidden layer dropout
    double attention_probs_dropout_prob = 0.1; ///< Attention dropout
    int64_t max_position_embeddings = 514;   ///< Maximum sequence length (512 + 2 special tokens)
    int64_t type_vocab_size = 1;             ///< Single segment type (no NSP)
    double layer_norm_eps = 1e-5;            ///< Layer norm epsilon (different from BERT)
    std::string hidden_act = "gelu";         ///< Activation function

    /**
     * @brief Create RoBERTa-base configuration (125M parameters)
     */
    static RobertaConfig base() {
        return RobertaConfig{};  // Default values are RoBERTa-base
    }

    /**
     * @brief Create RoBERTa-large configuration (355M parameters)
     */
    static RobertaConfig large() {
        RobertaConfig config;
        config.hidden_size = 1024;
        config.num_hidden_layers = 24;
        config.num_attention_heads = 16;
        config.intermediate_size = 4096;
        return config;
    }

    /**
     * @brief Convert to BertConfig for component reuse
     */
    BertConfig to_bert_config() const {
        BertConfig config;
        config.vocab_size = vocab_size;
        config.hidden_size = hidden_size;
        config.num_hidden_layers = num_hidden_layers;
        config.num_attention_heads = num_attention_heads;
        config.intermediate_size = intermediate_size;
        config.hidden_dropout_prob = hidden_dropout_prob;
        config.attention_probs_dropout_prob = attention_probs_dropout_prob;
        config.max_position_embeddings = max_position_embeddings;
        config.type_vocab_size = type_vocab_size;
        config.layer_norm_eps = layer_norm_eps;
        config.hidden_act = hidden_act;
        return config;
    }
};

/**
 * @brief RoBERTa embeddings layer
 *
 * Identical to BERT embeddings but with:
 * - Single token type (no segment embeddings in practice)
 * - Larger vocabulary for byte-level BPE
 * - Different layer norm epsilon
 *
 * Reuses BertEmbeddings implementation with modified config.
 */
class RobertaEmbeddings : public BertEmbeddings {
public:
    /**
     * @brief Construct RoBERTa embeddings layer
     *
     * @param config RoBERTa configuration
     */
    explicit RobertaEmbeddings(const RobertaConfig& config);

    /**
     * @brief RoBERTa position IDs: padding_idx+1 plus the cumulative count of
     * non-padding tokens, matching HF create_position_ids_from_input_ids
     * (padding positions map to padding_idx). RoBERTa pad_token_id = 1.
     */
    auto create_position_ids(const Tensor& input_ids) -> Tensor override;
};

/**
 * @brief RoBERTa encoder (stack of transformer encoder layers)
 *
 * Identical to BertEncoder - reuses implementation directly.
 */
class RobertaEncoder : public BertEncoder {
public:
    /**
     * @brief Construct RoBERTa encoder
     *
     * @param config RoBERTa configuration
     */
    explicit RobertaEncoder(const RobertaConfig& config);
};

/**
 * @brief RoBERTa pooler layer
 *
 * Identical to BertPooler - reuses implementation directly.
 */
class RobertaPooler : public BertPooler {
public:
    /**
     * @brief Construct RoBERTa pooler
     *
     * @param config RoBERTa configuration
     */
    explicit RobertaPooler(const RobertaConfig& config);
};

/**
 * @brief Output structure for RoBERTa models
 */
struct RobertaOutput {
    Variable sequence_output;  ///< Token-level representations [batch, seq_len, hidden_size]
    Variable pooled_output;    ///< Sentence representation [batch, hidden_size]
};

/**
 * @brief Base RoBERTa model
 *
 * The core RoBERTa model that outputs both sequence-level and pooled representations.
 * Architecture is identical to BERT, but trained differently:
 * - No Next Sentence Prediction (NSP) task
 * - Dynamic masking (different masks each epoch)
 * - Full sentences from single documents
 * - Byte-level BPE tokenization
 *
 * Example usage:
 * ```
 * auto config = RobertaConfig::base();
 * auto roberta = RobertaModel(config);
 *
 * Variable input_ids;  // [batch, seq_len]
 * auto outputs = roberta.forward(input_ids);
 * auto sequence_output = outputs.sequence_output;  // [batch, seq_len, 768]
 * auto pooled_output = outputs.pooled_output;      // [batch, 768]
 * ```
 */
class RobertaModel : public nn::Module {
public:
    /**
     * @brief Construct RoBERTa model
     *
     * @param config RoBERTa configuration
     */
    explicit RobertaModel(const RobertaConfig& config);

    /**
     * @brief Forward pass through RoBERTa
     *
     * @param input_ids Token IDs [batch, seq_len]
     * @param attention_mask Attention mask [batch, seq_len] (optional)
     * @param token_type_ids Segment IDs [batch, seq_len] (optional, usually all zeros)
     * @param position_ids Position IDs [batch, seq_len] (optional)
     * @return RobertaOutput with sequence_output and pooled_output
     */
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{},
                const Variable& token_type_ids = Variable{},
                const Variable& position_ids = Variable{}) -> RobertaOutput;

    /**
     * @brief Required by Module base class (returns sequence_output only)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Get model configuration
     */
    auto config() const -> const RobertaConfig& { return config_; }

    /// Load pretrained weights via ModelHub. See AlbertModel for details.
    auto load_pretrained(const std::string& path, bool strict = true) -> void;

private:
    RobertaConfig config_;
    std::shared_ptr<RobertaEmbeddings> embeddings_;
    std::shared_ptr<RobertaEncoder> encoder_;
    std::shared_ptr<RobertaPooler> pooler_;
};

/**
 * @brief RoBERTa for sequence classification
 *
 * RoBERTa model with a classification head on top of the pooled output.
 * Identical architecture to BertForSequenceClassification but uses RoBERTa base model.
 *
 * Example usage:
 * ```
 * auto config = RobertaConfig::base();
 * auto classifier = RobertaForSequenceClassification(config, 2);  // Binary classification
 *
 * Variable input_ids;  // [batch, seq_len]
 * Variable logits = classifier.forward(input_ids);  // [batch, 2]
 * ```
 */
class RobertaForSequenceClassification : public nn::Module {
public:
    /**
     * @brief Construct RoBERTa sequence classifier
     *
     * @param config RoBERTa configuration
     * @param num_labels Number of classes
     */
    RobertaForSequenceClassification(const RobertaConfig& config, int64_t num_labels);

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
    auto forward_impl(const Variable& input) -> Variable override;

private:
    RobertaConfig config_;
    int64_t num_labels_;
    std::shared_ptr<RobertaModel> roberta_;
    std::shared_ptr<nn::Dropout> dropout_;
    std::shared_ptr<nn::Linear> classifier_;
};

/**
 * @brief RoBERTa for token classification
 *
 * RoBERTa model with a token-level classification head.
 * Suitable for NER, POS tagging, and other token-level tasks.
 *
 * Example usage:
 * ```
 * auto config = RobertaConfig::base();
 * auto tagger = RobertaForTokenClassification(config, 9);  // 9 NER tags
 *
 * Variable input_ids;  // [batch, seq_len]
 * Variable logits = tagger.forward(input_ids);  // [batch, seq_len, 9]
 * ```
 */
class RobertaForTokenClassification : public nn::Module {
public:
    /**
     * @brief Construct RoBERTa token classifier
     *
     * @param config RoBERTa configuration
     * @param num_labels Number of token labels
     */
    RobertaForTokenClassification(const RobertaConfig& config, int64_t num_labels);

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
    auto forward_impl(const Variable& input) -> Variable override;

private:
    RobertaConfig config_;
    int64_t num_labels_;
    std::shared_ptr<RobertaModel> roberta_;
    std::shared_ptr<nn::Dropout> dropout_;
    std::shared_ptr<nn::Linear> classifier_;
};

/**
 * @brief Output structure for question answering
 */
struct RobertaQAOutput {
    Variable start_logits;  ///< Start position logits [batch, seq_len]
    Variable end_logits;    ///< End position logits [batch, seq_len]
};

/**
 * @brief RoBERTa for question answering
 *
 * RoBERTa model with start and end span prediction heads for extractive QA.
 * Suitable for SQuAD-style question answering.
 *
 * Example usage:
 * ```
 * auto config = RobertaConfig::base();
 * auto qa_model = RobertaForQuestionAnswering(config);
 *
 * Variable input_ids;  // [batch, seq_len], format: <s> question </s></s> context </s>
 * auto outputs = qa_model.forward(input_ids);
 * auto start_logits = outputs.start_logits;  // [batch, seq_len]
 * auto end_logits = outputs.end_logits;      // [batch, seq_len]
 * ```
 */
class RobertaForQuestionAnswering : public nn::Module {
public:
    /**
     * @brief Construct RoBERTa QA model
     *
     * @param config RoBERTa configuration
     */
    explicit RobertaForQuestionAnswering(const RobertaConfig& config);

    /**
     * @brief Forward pass
     *
     * @param input_ids Token IDs [batch, seq_len]
     * @param attention_mask Attention mask [batch, seq_len] (optional)
     * @param token_type_ids Segment IDs [batch, seq_len] (optional)
     * @return RobertaQAOutput with start_logits and end_logits
     */
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{},
                const Variable& token_type_ids = Variable{}) -> RobertaQAOutput;

    /**
     * @brief Required by Module base class (returns start_logits only)
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    RobertaConfig config_;
    std::shared_ptr<RobertaModel> roberta_;
    std::shared_ptr<nn::Linear> qa_outputs_;
};

} // namespace models
} // namespace tenzor
