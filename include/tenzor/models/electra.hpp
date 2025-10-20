/**
 * @file electra.hpp
 * @brief ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)
 *
 * Implements ELECTRA's generator-discriminator architecture for efficient pre-training.
 * Uses a small generator to replace tokens and a full-sized discriminator to detect them.
 * More sample-efficient than BERT's MLM, training on all tokens rather than just masked ones.
 *
 * Architecture:
 * - Generator: Small BERT-like model (1/3 to 1/4 size of discriminator)
 * - Discriminator: Full BERT-sized model
 * - Training: Generator creates fake tokens, discriminator detects them
 * - Fine-tuning: Only discriminator is used
 *
 * Reference: "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators"
 * (Clark et al., 2020)
 */

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include "bert.hpp"
#include "../nn/module.hpp"
#include "../core/tensor.hpp"
#include "../autograd/variable.hpp"

namespace tenzor {
namespace models {

/**
 * @brief Configuration for ELECTRA models
 */
struct ElectraConfig {
    // Vocabulary
    int64_t vocab_size = 30522;

    // Discriminator (main model, same as BERT)
    int64_t hidden_size = 768;
    int64_t num_hidden_layers = 12;
    int64_t num_attention_heads = 12;
    int64_t intermediate_size = 3072;

    // Generator (smaller model)
    int64_t generator_hidden_size = 256;     ///< Typically 1/3 to 1/4 of discriminator
    int64_t generator_layers = 12;           ///< Same depth as discriminator
    int64_t generator_heads = 4;             ///< Fewer heads
    int64_t generator_intermediate_size = 1024;  ///< 4 * generator_hidden_size

    // Training
    double gen_loss_weight = 1.0;            ///< Weight for generator loss
    double disc_loss_weight = 50.0;          ///< Weight for discriminator loss
    double mask_prob = 0.15;                 ///< Masking probability
    bool tie_embeddings = true;              ///< Share token embeddings

    // Regularization
    double hidden_dropout_prob = 0.1;
    double attention_probs_dropout_prob = 0.1;

    // Position & Segments
    int64_t max_position_embeddings = 512;
    int64_t type_vocab_size = 2;

    // Layer Norm
    double layer_norm_eps = 1e-12;

    // Activation
    std::string hidden_act = "gelu";

    /**
     * @brief Create ELECTRA-Small configuration (17M parameters total)
     */
    static ElectraConfig small() {
        ElectraConfig config;
        config.hidden_size = 256;
        config.num_hidden_layers = 12;
        config.num_attention_heads = 4;
        config.intermediate_size = 1024;
        config.generator_hidden_size = 128;
        config.generator_heads = 2;
        config.generator_intermediate_size = 512;
        return config;  // 14M discriminator + 3M generator
    }

    /**
     * @brief Create ELECTRA-Base configuration (143M parameters total)
     */
    static ElectraConfig base() {
        return ElectraConfig{};  // 110M discriminator + 33M generator
    }

    /**
     * @brief Create ELECTRA-Large configuration (415M parameters total)
     */
    static ElectraConfig large() {
        ElectraConfig config;
        config.hidden_size = 1024;
        config.num_hidden_layers = 24;
        config.num_attention_heads = 16;
        config.intermediate_size = 4096;
        config.generator_hidden_size = 256;
        config.generator_heads = 4;
        config.generator_intermediate_size = 1024;
        return config;  // 335M discriminator + 80M generator
    }

    /**
     * @brief Convert discriminator config to BertConfig
     */
    BertConfig to_discriminator_config() const {
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

    /**
     * @brief Convert generator config to BertConfig
     */
    BertConfig to_generator_config() const {
        BertConfig config;
        config.vocab_size = vocab_size;
        config.hidden_size = generator_hidden_size;
        config.num_hidden_layers = generator_layers;
        config.num_attention_heads = generator_heads;
        config.intermediate_size = generator_intermediate_size;
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
 * @brief ELECTRA Generator
 *
 * Small BERT-like model that predicts masked tokens.
 * Used during pre-training to generate plausible token replacements.
 */
class ElectraGenerator : public nn::Module {
public:
    /**
     * @brief Construct ELECTRA generator
     *
     * @param config ELECTRA configuration
     */
    explicit ElectraGenerator(const ElectraConfig& config);

    /**
     * @brief Forward pass through generator
     *
     * @param input_ids Token IDs [batch, seq_len] with masked tokens
     * @param attention_mask Attention mask [batch, seq_len] (optional)
     * @param token_type_ids Segment IDs [batch, seq_len] (optional)
     * @return Logits for all tokens [batch, seq_len, vocab_size]
     */
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{},
                const Variable& token_type_ids = Variable{}) -> Variable;

    /**
     * @brief Required by Module base class
     */
    auto forward(const Variable& input) -> Variable override;

private:
    ElectraConfig config_;
    std::shared_ptr<BertModel> generator_;
    std::shared_ptr<nn::Linear> lm_head_;
};

/**
 * @brief ELECTRA Discriminator
 *
 * Full BERT-sized model that classifies each token as real or replaced.
 * Used during pre-training to detect generator's token replacements.
 * This is the model used for downstream tasks after pre-training.
 */
class ElectraDiscriminator : public nn::Module {
public:
    /**
     * @brief Construct ELECTRA discriminator
     *
     * @param config ELECTRA configuration
     */
    explicit ElectraDiscriminator(const ElectraConfig& config);

    /**
     * @brief Forward pass through discriminator
     *
     * @param input_ids Token IDs [batch, seq_len] (possibly with replaced tokens)
     * @param attention_mask Attention mask [batch, seq_len] (optional)
     * @param token_type_ids Segment IDs [batch, seq_len] (optional)
     * @return Binary logits for each token [batch, seq_len] (real vs replaced)
     */
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{},
                const Variable& token_type_ids = Variable{}) -> Variable;

    /**
     * @brief Required by Module base class
     */
    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Get the underlying BERT model (for downstream tasks)
     */
    auto get_bert_model() -> std::shared_ptr<BertModel> { return discriminator_; }

private:
    ElectraConfig config_;
    std::shared_ptr<BertModel> discriminator_;
    std::shared_ptr<nn::Linear> classifier_;
};

/**
 * @brief Output structure for ELECTRA pre-training
 */
struct ElectraPreTrainingOutput {
    Variable gen_logits;      ///< Generator predictions [batch, seq_len, vocab_size]
    Variable disc_logits;     ///< Discriminator predictions [batch, seq_len]
    Tensor is_replaced;       ///< Binary mask of replaced tokens [batch, seq_len]
};

/**
 * @brief ELECTRA for pre-training
 *
 * Combines generator and discriminator for ELECTRA pre-training.
 * Training process:
 * 1. Mask some input tokens
 * 2. Generator predicts masked tokens
 * 3. Sample from generator to create corrupted sequence
 * 4. Discriminator detects which tokens are real vs replaced
 * 5. Train both models jointly
 *
 * Example usage:
 * ```
 * auto config = ElectraConfig::base();
 * auto electra = ElectraForPreTraining(config);
 *
 * Variable input_ids;  // [batch, seq_len]
 * Tensor masked_positions;  // [batch, seq_len] - 1 for masked, 0 for real
 * Tensor original_tokens;   // [batch, seq_len] - original token IDs
 *
 * auto outputs = electra.forward(input_ids, masked_positions, original_tokens);
 * // outputs contains generator and discriminator predictions
 * ```
 */
class ElectraForPreTraining : public nn::Module {
public:
    /**
     * @brief Construct ELECTRA pre-training model
     *
     * @param config ELECTRA configuration
     */
    explicit ElectraForPreTraining(const ElectraConfig& config);

    /**
     * @brief Forward pass through ELECTRA (for pre-training)
     *
     * @param input_ids Token IDs with masked positions [batch, seq_len]
     * @param masked_positions Binary mask indicating masked positions [batch, seq_len]
     * @param original_tokens Original tokens before masking [batch, seq_len]
     * @param attention_mask Attention mask [batch, seq_len] (optional)
     * @return ElectraPreTrainingOutput with generator and discriminator predictions
     */
    auto forward(const Variable& input_ids,
                const Tensor& masked_positions,
                const Tensor& original_tokens,
                const Tensor& attention_mask = Tensor{}) -> ElectraPreTrainingOutput;

    /**
     * @brief Required by Module base class (simplified interface)
     */
    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Compute combined loss for pre-training
     *
     * @param gen_logits Generator predictions
     * @param disc_logits Discriminator predictions
     * @param is_replaced Binary mask of replaced tokens
     * @param masked_positions Binary mask of masked positions
     * @param original_tokens Original tokens
     * @return Combined loss (weighted sum of generator and discriminator losses)
     */
    auto compute_loss(const Variable& gen_logits,
                     const Variable& disc_logits,
                     const Tensor& is_replaced,
                     const Tensor& masked_positions,
                     const Tensor& original_tokens) -> Variable;

    /**
     * @brief Get generator (useful for debugging)
     */
    auto get_generator() -> std::shared_ptr<ElectraGenerator> { return generator_; }

    /**
     * @brief Get discriminator (used for fine-tuning)
     */
    auto get_discriminator() -> std::shared_ptr<ElectraDiscriminator> { return discriminator_; }

private:
    ElectraConfig config_;
    std::shared_ptr<ElectraGenerator> generator_;
    std::shared_ptr<ElectraDiscriminator> discriminator_;

    /**
     * @brief Sample token from probability distribution
     *
     * @param probs Probability distribution [vocab_size]
     * @param size Vocabulary size
     * @return Sampled token ID
     */
    auto sample_from_distribution(const float* probs, int64_t size) -> int64_t;
};

/**
 * @brief ELECTRA for sequence classification
 *
 * Uses only the discriminator for downstream sequence classification tasks.
 * The generator is discarded after pre-training.
 *
 * Example usage:
 * ```
 * auto config = ElectraConfig::base();
 * auto classifier = ElectraForSequenceClassification(config, 2);  // Binary classification
 *
 * Variable input_ids;  // [batch, seq_len]
 * Variable logits = classifier.forward(input_ids);  // [batch, 2]
 * ```
 */
class ElectraForSequenceClassification : public nn::Module {
public:
    /**
     * @brief Construct ELECTRA sequence classifier
     *
     * @param config ELECTRA configuration
     * @param num_labels Number of classes
     */
    ElectraForSequenceClassification(const ElectraConfig& config, int64_t num_labels);

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
    ElectraConfig config_;
    int64_t num_labels_;
    std::shared_ptr<BertModel> bert_;  // Uses discriminator's BERT model
    std::shared_ptr<nn::Dropout> dropout_;
    std::shared_ptr<nn::Linear> classifier_;
};

/**
 * @brief ELECTRA for token classification
 *
 * Uses only the discriminator for downstream token classification tasks.
 *
 * Example usage:
 * ```
 * auto config = ElectraConfig::base();
 * auto tagger = ElectraForTokenClassification(config, 9);  // 9 NER tags
 *
 * Variable input_ids;  // [batch, seq_len]
 * Variable logits = tagger.forward(input_ids);  // [batch, seq_len, 9]
 * ```
 */
class ElectraForTokenClassification : public nn::Module {
public:
    /**
     * @brief Construct ELECTRA token classifier
     *
     * @param config ELECTRA configuration
     * @param num_labels Number of token labels
     */
    ElectraForTokenClassification(const ElectraConfig& config, int64_t num_labels);

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
    ElectraConfig config_;
    int64_t num_labels_;
    std::shared_ptr<BertModel> bert_;  // Uses discriminator's BERT model
    std::shared_ptr<nn::Dropout> dropout_;
    std::shared_ptr<nn::Linear> classifier_;
};

/**
 * @brief Output structure for question answering
 */
struct ElectraQAOutput {
    Variable start_logits;  ///< Start position logits [batch, seq_len]
    Variable end_logits;    ///< End position logits [batch, seq_len]
};

/**
 * @brief ELECTRA for question answering
 *
 * Uses only the discriminator for extractive QA tasks.
 *
 * Example usage:
 * ```
 * auto config = ElectraConfig::base();
 * auto qa_model = ElectraForQuestionAnswering(config);
 *
 * Variable input_ids;  // [batch, seq_len]
 * auto outputs = qa_model.forward(input_ids);
 * auto start_logits = outputs.start_logits;  // [batch, seq_len]
 * auto end_logits = outputs.end_logits;      // [batch, seq_len]
 * ```
 */
class ElectraForQuestionAnswering : public nn::Module {
public:
    /**
     * @brief Construct ELECTRA QA model
     *
     * @param config ELECTRA configuration
     */
    explicit ElectraForQuestionAnswering(const ElectraConfig& config);

    /**
     * @brief Forward pass
     *
     * @param input_ids Token IDs [batch, seq_len]
     * @param attention_mask Attention mask [batch, seq_len] (optional)
     * @param token_type_ids Segment IDs [batch, seq_len] (optional)
     * @return ElectraQAOutput with start_logits and end_logits
     */
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{},
                const Variable& token_type_ids = Variable{}) -> ElectraQAOutput;

    /**
     * @brief Required by Module base class (returns start_logits only)
     */
    auto forward(const Variable& input) -> Variable override;

private:
    ElectraConfig config_;
    std::shared_ptr<BertModel> bert_;  // Uses discriminator's BERT model
    std::shared_ptr<nn::Linear> qa_outputs_;
};

} // namespace models
} // namespace tenzor
