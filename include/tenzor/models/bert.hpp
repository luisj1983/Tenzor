/**
 * @file bert.hpp
 * @brief BERT (Bidirectional Encoder Representations from Transformers) model family
 *
 * Implements BERT architecture and its variants for NLP tasks including:
 * - Base BERT model for feature extraction
 * - Sequence classification (text classification, sentiment analysis)
 * - Token classification (NER, POS tagging)
 * - Question answering (SQuAD-style)
 *
 * Reference: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
 * (Devlin et al., 2018)
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
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
 * @brief Configuration for BERT models
 */
struct BertConfig {
    int64_t vocab_size = 30522;              ///< Vocabulary size
    int64_t hidden_size = 768;               ///< Hidden dimension
    int64_t num_hidden_layers = 12;          ///< Number of encoder layers
    int64_t num_attention_heads = 12;        ///< Number of attention heads
    int64_t intermediate_size = 3072;        ///< FFN intermediate dimension
    double hidden_dropout_prob = 0.1;        ///< Hidden layer dropout
    double attention_probs_dropout_prob = 0.1; ///< Attention dropout
    int64_t max_position_embeddings = 512;   ///< Maximum sequence length
    int64_t type_vocab_size = 2;             ///< Number of token types (segments)
    double layer_norm_eps = 1e-12;           ///< Layer norm epsilon
    std::string hidden_act = "gelu";         ///< Activation function

    /**
     * @brief Create BERT-base configuration
     */
    static BertConfig base() {
        return BertConfig{};  // Default values are BERT-base
    }

    /**
     * @brief Create BERT-large configuration
     */
    static BertConfig large() {
        BertConfig config;
        config.hidden_size = 1024;
        config.num_hidden_layers = 24;
        config.num_attention_heads = 16;
        config.intermediate_size = 4096;
        return config;
    }
};

/**
 * @brief BERT embeddings layer
 *
 * Combines token embeddings, position embeddings, and token type embeddings,
 * followed by layer normalization and dropout.
 *
 * Architecture:
 * @code
 *   token_ids ──> TokenEmbedding ───┐
 *   positions ──> PositionEmbedding ─┼──> Add ──> LayerNorm ──> Dropout ──> Output
 *   token_types > TypeEmbedding ────┘
 * @endcode
 */
class BertEmbeddings : public nn::Module {
public:
    /**
     * @brief Construct BERT embeddings layer
     *
     * @param config BERT configuration
     */
    explicit BertEmbeddings(const BertConfig& config);

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
    auto forward_impl(const Variable& input) -> Variable override;

    /// @brief Access the word (token) embedding table.
    auto word_embeddings() const -> std::shared_ptr<nn::Embedding> { return word_embeddings_; }

    /**
     * @brief Replace the word embedding table with an externally-owned shared
     *        one (e.g. for ELECTRA generator/discriminator embedding tying).
     *
     * The shared table is intentionally NOT registered as a submodule here so
     * the owner can register it exactly once (avoiding double-counting in
     * parameters()). If `shared_dim` differs from this module's hidden size, a
     * linear projection is added so the summed embeddings stay at hidden_size.
     *
     * @param shared     Shared embedding table [vocab, shared_dim]
     * @param shared_dim Embedding dimension of `shared`
     */
    auto set_shared_word_embeddings(std::shared_ptr<nn::Embedding> shared,
                                    int64_t shared_dim) -> void;

private:
    BertConfig config_;
    std::shared_ptr<nn::Embedding> word_embeddings_;
    std::shared_ptr<nn::Embedding> position_embeddings_;
    std::shared_ptr<nn::Embedding> token_type_embeddings_;
    std::shared_ptr<nn::Linear> word_embeddings_project_;  ///< Optional embed_dim -> hidden_size projection (shared/factorized embeddings)
    std::shared_ptr<nn::LayerNorm> layer_norm_;
    std::shared_ptr<nn::Dropout> dropout_;

    /**
     * @brief Create default position IDs [0, 1, 2, ..., seq_len-1]
     */
    virtual auto create_position_ids(const Tensor& input_ids) -> Tensor;
};

/**
 * @brief BERT encoder (stack of transformer encoder layers)
 *
 * Wraps TransformerEncoder with BERT-specific configuration.
 */
class BertEncoder : public nn::Module {
public:
    /**
     * @brief Construct BERT encoder
     *
     * @param config BERT configuration
     */
    explicit BertEncoder(const BertConfig& config);

    /**
     * @brief Forward pass through encoder
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
    auto forward_impl(const Variable& input) -> Variable override;

private:
    BertConfig config_;
    std::shared_ptr<nn::TransformerEncoder> encoder_;

    /**
     * @brief Convert attention mask to transformer format
     * Transforms [batch, seq_len] with 0/1 to [seq_len, seq_len] with -inf/0
     */
    auto prepare_attention_mask(const Tensor& mask, int64_t seq_len, DType compute_dtype = DType::Float32) -> Tensor;
};

/**
 * @brief BERT pooler layer
 *
 * Pools the [CLS] token representation and applies a linear transformation
 * with tanh activation.
 */
class BertPooler : public nn::Module {
public:
    /**
     * @brief Construct BERT pooler
     *
     * @param config BERT configuration
     */
    explicit BertPooler(const BertConfig& config);

    /**
     * @brief Forward pass through pooler (required by Module base class)
     *
     * @param hidden_states Encoder output [batch, seq_len, hidden_size]
     * @return Pooled output [batch, hidden_size]
     */
    auto forward_impl(const Variable& hidden_states) -> Variable override;

private:
    std::shared_ptr<nn::Linear> dense_;
};

/**
 * @brief Output structure for BERT models
 */
struct BertOutput {
    Variable sequence_output;  ///< Token-level representations [batch, seq_len, hidden_size]
    Variable pooled_output;    ///< Sentence representation [batch, hidden_size]
};

/**
 * @brief Base BERT model
 *
 * The core BERT model that outputs both sequence-level and pooled representations.
 * Can be used for feature extraction or as a base for task-specific models.
 *
 * Example usage:
 * ```
 * auto config = BertConfig::base();
 * auto bert = BertModel(config);
 *
 * Variable input_ids;  // [batch, seq_len]
 * auto outputs = bert.forward(input_ids);
 * auto sequence_output = outputs.sequence_output;  // [batch, seq_len, 768]
 * auto pooled_output = outputs.pooled_output;      // [batch, 768]
 * ```
 */
class BertModel : public nn::Module {
public:
    /**
     * @brief Construct BERT model
     *
     * @param config BERT configuration
     * @param add_pooling_layer Whether to include pooling layer (default: true)
     */
    explicit BertModel(const BertConfig& config, bool add_pooling_layer = true);

    /**
     * @brief Forward pass through BERT
     *
     * @param input_ids Token IDs [batch, seq_len]
     * @param attention_mask Attention mask [batch, seq_len] (optional)
     * @param token_type_ids Segment IDs [batch, seq_len] (optional)
     * @param position_ids Position IDs [batch, seq_len] (optional)
     * @return BertOutput with sequence_output and pooled_output
     */
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{},
                const Variable& token_type_ids = Variable{},
                const Variable& position_ids = Variable{}) -> BertOutput;

    /**
     * @brief Required by Module base class (returns sequence_output only)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Get model configuration
     */
    auto config() const -> const BertConfig& { return config_; }

    /// @brief Access the embeddings submodule (used for ELECTRA embedding tying).
    auto embeddings() const -> std::shared_ptr<BertEmbeddings> { return embeddings_; }

private:
    BertConfig config_;
    std::shared_ptr<BertEmbeddings> embeddings_;
    std::shared_ptr<BertEncoder> encoder_;
    std::shared_ptr<BertPooler> pooler_;  // nullptr if add_pooling_layer=false
};

/**
 * @brief BERT for sequence classification
 *
 * BERT model with a classification head on top of the pooled output.
 * Suitable for:
 * - Text classification
 * - Sentiment analysis
 * - Intent detection
 * - Sentence pair classification (NLI, paraphrase detection)
 *
 * Example usage:
 * ```
 * auto config = BertConfig::base();
 * auto classifier = BertForSequenceClassification(config, 2);  // Binary classification
 *
 * Variable input_ids;  // [batch, seq_len]
 * Variable logits = classifier.forward(input_ids);  // [batch, 2]
 * ```
 */
class BertForSequenceClassification : public nn::Module {
public:
    /**
     * @brief Construct BERT sequence classifier
     *
     * @param config BERT configuration
     * @param num_labels Number of classes
     */
    BertForSequenceClassification(const BertConfig& config, int64_t num_labels);

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
    BertConfig config_;
    int64_t num_labels_;
    std::shared_ptr<BertModel> bert_;
    std::shared_ptr<nn::Dropout> dropout_;
    std::shared_ptr<nn::Linear> classifier_;
};

/**
 * @brief BERT for token classification
 *
 * BERT model with a token-level classification head.
 * Suitable for:
 * - Named Entity Recognition (NER)
 * - Part-of-Speech (POS) tagging
 * - Slot filling
 * - Word-level classification tasks
 *
 * Example usage:
 * ```
 * auto config = BertConfig::base();
 * auto tagger = BertForTokenClassification(config, 9);  // 9 NER tags
 *
 * Variable input_ids;  // [batch, seq_len]
 * Variable logits = tagger.forward(input_ids);  // [batch, seq_len, 9]
 * ```
 */
class BertForTokenClassification : public nn::Module {
public:
    /**
     * @brief Construct BERT token classifier
     *
     * @param config BERT configuration
     * @param num_labels Number of token labels
     */
    BertForTokenClassification(const BertConfig& config, int64_t num_labels);

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
    BertConfig config_;
    int64_t num_labels_;
    std::shared_ptr<BertModel> bert_;
    std::shared_ptr<nn::Dropout> dropout_;
    std::shared_ptr<nn::Linear> classifier_;
};

/**
 * @brief Output structure for question answering
 */
struct BertQAOutput {
    Variable start_logits;  ///< Start position logits [batch, seq_len]
    Variable end_logits;    ///< End position logits [batch, seq_len]
};

/**
 * @brief BERT for question answering
 *
 * BERT model with start and end span prediction heads for extractive QA.
 * Suitable for SQuAD-style question answering where the answer is a span
 * within the context.
 *
 * Example usage:
 * ```
 * auto config = BertConfig::base();
 * auto qa_model = BertForQuestionAnswering(config);
 *
 * Variable input_ids;  // [batch, seq_len], format: [CLS] question [SEP] context [SEP]
 * auto outputs = qa_model.forward(input_ids);
 * auto start_logits = outputs.start_logits;  // [batch, seq_len]
 * auto end_logits = outputs.end_logits;      // [batch, seq_len]
 * ```
 */
class BertForQuestionAnswering : public nn::Module {
public:
    /**
     * @brief Construct BERT QA model
     *
     * @param config BERT configuration
     */
    explicit BertForQuestionAnswering(const BertConfig& config);

    /**
     * @brief Forward pass
     *
     * @param input_ids Token IDs [batch, seq_len]
     * @param attention_mask Attention mask [batch, seq_len] (optional)
     * @param token_type_ids Segment IDs [batch, seq_len] (optional)
     * @return BertQAOutput with start_logits and end_logits
     */
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{},
                const Variable& token_type_ids = Variable{}) -> BertQAOutput;

    /**
     * @brief Required by Module base class (returns start_logits only)
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    BertConfig config_;
    std::shared_ptr<BertModel> bert_;
    std::shared_ptr<nn::Linear> qa_outputs_;
};

/**
 * @brief Model hub utilities for loading pretrained models
 */
class BertModelHub {
public:
    /**
     * @brief Load pretrained BERT weights from Hugging Face format
     *
     * @param model BERT model to load weights into
     * @param model_name Model name (e.g., "bert-base-uncased")
     * @param cache_dir Directory to cache downloaded models (default: ~/.cache/tenzor)
     *
     * @throws std::runtime_error if model not found or incompatible
     *
     * @code
     * auto bert = BertModel(BertConfig::base());
     * ModelHub::load_pretrained_weights(bert, "bert-base-uncased");
     * @endcode
     */
    static auto load_pretrained_weights(nn::Module& model,
                                       const std::string& model_name,
                                       const std::string& cache_dir = "") -> void;

    /**
     * @brief Download and cache pretrained model checkpoint
     *
     * @param model_name Model name from Hugging Face
     * @param cache_dir Cache directory
     * @return Path to downloaded checkpoint
     */
    static auto download_model(const std::string& model_name,
                              const std::string& cache_dir = "") -> std::string;

    /**
     * @brief Map Hugging Face parameter names to Tenzor names
     *
     * @param hf_name Hugging Face parameter name
     * @return Tenzor parameter name
     */
    static auto map_parameter_name(const std::string& hf_name) -> std::string;

    /**
     * @brief Load PyTorch checkpoint file (.bin or .pt)
     *
     * @param checkpoint_path Path to checkpoint file
     * @return Map of parameter names to tensors
     */
    static auto load_pytorch_checkpoint(const std::string& checkpoint_path)
        -> std::unordered_map<std::string, Tensor>;

    /**
     * @brief Verify checkpoint is compatible with model architecture
     *
     * @param model Model to check
     * @param checkpoint Loaded checkpoint
     * @throws std::runtime_error if incompatible
     */
    static auto verify_checkpoint_compatibility(const nn::Module& model,
                                               const std::unordered_map<std::string, Tensor>& checkpoint) -> void;
};

} // namespace models
} // namespace tenzor
