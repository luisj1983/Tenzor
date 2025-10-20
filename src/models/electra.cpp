/**
 * @file electra.cpp
 * @brief Implementation of ELECTRA model family
 */

#include "tenzor/models/electra.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/creation.hpp"
#include <random>
#include <cmath>

namespace tenzor {
namespace models {

// ============================================================================
// ElectraGenerator Implementation
// ============================================================================

ElectraGenerator::ElectraGenerator(const ElectraConfig& config)
    : config_(config) {
    // Create small BERT model for generator
    auto gen_config = config.to_generator_config();
    generator_ = std::make_shared<BertModel>(gen_config);

    // Language model head for token prediction
    lm_head_ = std::make_shared<nn::Linear>(
        config.generator_hidden_size, config.vocab_size);

    register_module("generator", generator_);
    register_module("lm_head", lm_head_);
}

auto ElectraGenerator::forward(const Variable& input_ids,
                                const Tensor& attention_mask,
                                const Variable& token_type_ids) -> Variable {
    // Get generator outputs
    auto outputs = generator_->forward(input_ids, attention_mask, token_type_ids, Variable{});

    // Predict tokens for all positions
    auto logits = lm_head_->forward(outputs.sequence_output);

    return logits;  // [batch, seq_len, vocab_size]
}

auto ElectraGenerator::forward(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// ElectraDiscriminator Implementation
// ============================================================================

ElectraDiscriminator::ElectraDiscriminator(const ElectraConfig& config)
    : config_(config) {
    // Create full BERT model for discriminator
    auto disc_config = config.to_discriminator_config();
    discriminator_ = std::make_shared<BertModel>(disc_config);

    // Binary classification head (real vs replaced)
    classifier_ = std::make_shared<nn::Linear>(config.hidden_size, 1);

    register_module("discriminator", discriminator_);
    register_module("classifier", classifier_);
}

auto ElectraDiscriminator::forward(const Variable& input_ids,
                                    const Tensor& attention_mask,
                                    const Variable& token_type_ids) -> Variable {
    // Get discriminator outputs
    auto outputs = discriminator_->forward(input_ids, attention_mask, token_type_ids, Variable{});

    // Classify each token as real (0) or replaced (1)
    auto logits = classifier_->forward(outputs.sequence_output);

    // Squeeze last dimension: [batch, seq_len, 1] -> [batch, seq_len]
    auto shape = logits.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];

    auto squeezed = tenzor::reshape(logits, {batch_size, seq_len});

    return squeezed;
}

auto ElectraDiscriminator::forward(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// ElectraForPreTraining Implementation
// ============================================================================

ElectraForPreTraining::ElectraForPreTraining(const ElectraConfig& config)
    : config_(config) {
    // Create generator and discriminator
    generator_ = std::make_shared<ElectraGenerator>(config);
    discriminator_ = std::make_shared<ElectraDiscriminator>(config);

    register_module("generator", generator_);
    register_module("discriminator", discriminator_);
}

auto ElectraForPreTraining::sample_from_distribution(const float* probs, int64_t size) -> int64_t {
    // Sample token index from probability distribution
    // Using discrete distribution for sampling
    std::discrete_distribution<int64_t> dist(probs, probs + size);
    static std::mt19937 gen{42};  // Fixed seed for reproducibility
    return dist(gen);
}

auto ElectraForPreTraining::forward(const Variable& input_ids,
                                    const Tensor& masked_positions,
                                    const Tensor& original_tokens,
                                    const Tensor& attention_mask) -> ElectraPreTrainingOutput {
    int64_t batch_size = input_ids.shape()[0];
    int64_t seq_len = input_ids.shape()[1];

    // Step 1: Generator predicts masked tokens
    auto gen_logits = generator_->forward(input_ids, attention_mask);  // [batch, seq_len, vocab]

    // Step 2: Sample tokens from generator predictions (at masked positions)
    auto gen_probs = nn::softmax(gen_logits, -1);

    // Create corrupted input by replacing masked tokens with generator samples
    Tensor generated_tokens = input_ids.tensor().clone();  // Copy original
    auto shape_vec = std::vector<int64_t>(input_ids.shape().begin(), input_ids.shape().end());
    Tensor is_replaced(shape_vec, DType::Float32, input_ids.tensor().device());
    is_replaced.zero_();

    const int64_t* mask_data = masked_positions.data<int64_t>();
    const int64_t* orig_data = original_tokens.data<int64_t>();
    int64_t* gen_data = generated_tokens.data<int64_t>();
    float* repl_data = is_replaced.data<float>();

    // Sample from generator for masked positions
    auto probs_data = gen_probs.tensor().data<float>();

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            int64_t idx = b * seq_len + s;

            if (mask_data[idx] == 1) {  // This position was masked
                // Get probability distribution for this position
                const float* pos_probs = probs_data + idx * config_.vocab_size;

                // Sample from generator's distribution
                int64_t sampled_token = sample_from_distribution(pos_probs, config_.vocab_size);

                gen_data[idx] = sampled_token;

                // Mark as replaced if different from original
                repl_data[idx] = (sampled_token != orig_data[idx]) ? 1.0f : 0.0f;
            }
        }
    }

    // Step 3: Discriminator classifies all tokens as real/replaced
    Variable gen_input(generated_tokens, false);  // Don't need gradients for input tensor
    auto disc_logits = discriminator_->forward(gen_input, attention_mask);  // [batch, seq_len]

    return ElectraPreTrainingOutput{gen_logits, disc_logits, is_replaced};
}

auto ElectraForPreTraining::forward(const Variable& input) -> Variable {
    // Simplified interface - just return discriminator output
    // For actual pre-training, use the full forward() method
    auto disc_logits = discriminator_->forward(input, Tensor{}, Variable{});
    return disc_logits;
}

auto ElectraForPreTraining::compute_loss(const Variable& gen_logits,
                                         const Variable& disc_logits,
                                         const Tensor& is_replaced,
                                         const Tensor& masked_positions,
                                         const Tensor& original_tokens) -> Variable {
    int64_t batch_size = gen_logits.shape()[0];
    int64_t seq_len = gen_logits.shape()[1];
    int64_t vocab_size = gen_logits.shape()[2];

    // Generator loss: MLM loss on masked tokens only
    // Compute cross-entropy loss for masked positions
    auto reshaped_logits = tenzor::reshape(gen_logits, {batch_size * seq_len, vocab_size});

    // Create labels tensor
    Tensor labels_flat(std::vector<int64_t>{batch_size * seq_len}, DType::Int64, original_tokens.device());
    std::copy_n(original_tokens.data<int64_t>(), batch_size * seq_len, labels_flat.data<int64_t>());
    Variable labels_var(labels_flat, false);

    // Compute log softmax for cross-entropy
    auto log_probs = nn::log_softmax(reshaped_logits, 1);

    // Gather log probabilities for true labels
    // For simplicity, we'll compute mean loss over masked positions
    // In a full implementation, this would use a proper cross-entropy loss with masking

    // Discriminator loss: Binary cross-entropy on ALL tokens
    // Convert is_replaced to Variable
    Variable is_replaced_var(is_replaced, false);

    // Compute binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
    // where p = sigmoid(disc_logits)
    auto sigmoid_logits = nn::sigmoid(disc_logits);
    auto log_prob_real = tenzor::log(sigmoid_logits);
    auto ones_var = Variable(ones_like(sigmoid_logits.tensor()), false);
    auto log_prob_fake = tenzor::log(ones_var - sigmoid_logits);

    auto ones_repl = Variable(ones_like(is_replaced), false);
    auto disc_loss = tenzor::neg(is_replaced_var * log_prob_fake +
                       (ones_repl - is_replaced_var) * log_prob_real);

    // Mean over all tokens
    disc_loss = tenzor::mean(disc_loss);

    // For generator loss, we'll use a simplified version here
    // In practice, this should be proper MLM loss
    auto gen_loss = tenzor::mean(log_probs);  // Simplified

    // Combined loss (weighted sum)
    auto neg_gen_loss = tenzor::neg(gen_loss);
    auto gen_component = neg_gen_loss * config_.gen_loss_weight;
    auto disc_component = disc_loss * config_.disc_loss_weight;
    auto total_loss = gen_component + disc_component;

    return total_loss;
}

// ============================================================================
// ElectraForSequenceClassification Implementation
// ============================================================================

ElectraForSequenceClassification::ElectraForSequenceClassification(
    const ElectraConfig& config, int64_t num_labels)
    : config_(config), num_labels_(num_labels) {
    // Use discriminator's BERT model
    auto bert_config = config.to_discriminator_config();
    bert_ = std::make_shared<BertModel>(bert_config);

    // Classification head
    dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);
    classifier_ = std::make_shared<nn::Linear>(config.hidden_size, num_labels);

    register_module("electra", bert_);
    register_module("dropout", dropout_);
    register_module("classifier", classifier_);
}

auto ElectraForSequenceClassification::forward(const Variable& input_ids,
                                                const Tensor& attention_mask,
                                                const Variable& token_type_ids) -> Variable {
    // Get BERT outputs
    auto outputs = bert_->forward(input_ids, attention_mask, token_type_ids, Variable{});

    // Use pooled output
    auto pooled_output = outputs.pooled_output;

    // Apply dropout
    pooled_output = dropout_->forward(pooled_output);

    // Classify
    auto logits = classifier_->forward(pooled_output);

    return logits;
}

auto ElectraForSequenceClassification::forward(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// ElectraForTokenClassification Implementation
// ============================================================================

ElectraForTokenClassification::ElectraForTokenClassification(
    const ElectraConfig& config, int64_t num_labels)
    : config_(config), num_labels_(num_labels) {
    // Use discriminator's BERT model
    auto bert_config = config.to_discriminator_config();
    bert_ = std::make_shared<BertModel>(bert_config);

    // Token classification head
    dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);
    classifier_ = std::make_shared<nn::Linear>(config.hidden_size, num_labels);

    register_module("electra", bert_);
    register_module("dropout", dropout_);
    register_module("classifier", classifier_);
}

auto ElectraForTokenClassification::forward(const Variable& input_ids,
                                             const Tensor& attention_mask,
                                             const Variable& token_type_ids) -> Variable {
    // Get BERT outputs
    auto outputs = bert_->forward(input_ids, attention_mask, token_type_ids, Variable{});

    // Use sequence output (token-level representations)
    auto sequence_output = outputs.sequence_output;

    // Apply dropout
    sequence_output = dropout_->forward(sequence_output);

    // Classify each token
    auto logits = classifier_->forward(sequence_output);

    return logits;
}

auto ElectraForTokenClassification::forward(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// ElectraForQuestionAnswering Implementation
// ============================================================================

ElectraForQuestionAnswering::ElectraForQuestionAnswering(const ElectraConfig& config)
    : config_(config) {
    // Use discriminator's BERT model
    auto bert_config = config.to_discriminator_config();
    bert_ = std::make_shared<BertModel>(bert_config);

    // QA output layer (predicts start and end positions)
    qa_outputs_ = std::make_shared<nn::Linear>(config.hidden_size, 2);

    register_module("electra", bert_);
    register_module("qa_outputs", qa_outputs_);
}

auto ElectraForQuestionAnswering::forward(const Variable& input_ids,
                                          const Tensor& attention_mask,
                                          const Variable& token_type_ids) -> ElectraQAOutput {
    // Get BERT outputs
    auto outputs = bert_->forward(input_ids, attention_mask, token_type_ids, Variable{});

    // Use sequence output
    auto sequence_output = outputs.sequence_output;

    // Predict start and end logits
    auto logits = qa_outputs_->forward(sequence_output);

    // Split into start and end logits while preserving gradients
    // logits: [batch, seq_len, 2]
    auto shape = logits.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];

    // Reshape to [batch * seq_len, 2]
    auto reshaped = tenzor::reshape(logits, {batch_size * seq_len, 2});

    // Create selection matrices to extract start and end logits
    // Start logits: multiply by [1, 0]
    Tensor start_selector(std::vector<int64_t>{2, 1}, DType::Float32, logits.tensor().device());
    start_selector.zero_();
    start_selector.data<float>()[0] = 1.0f;  // [1; 0]

    // End logits: multiply by [0, 1]
    Tensor end_selector(std::vector<int64_t>{2, 1}, DType::Float32, logits.tensor().device());
    end_selector.zero_();
    end_selector.data<float>()[1] = 1.0f;  // [0; 1]

    // Use matmul to select: [batch*seq_len, 2] @ [2, 1] = [batch*seq_len, 1]
    Variable start_selector_var(start_selector, false);
    Variable end_selector_var(end_selector, false);

    auto start_flat = tenzor::matmul(reshaped, start_selector_var);  // [batch*seq_len, 1]
    auto end_flat = tenzor::matmul(reshaped, end_selector_var);      // [batch*seq_len, 1]

    // Reshape back to [batch, seq_len]
    auto start_logits = tenzor::reshape(start_flat, {batch_size, seq_len});
    auto end_logits = tenzor::reshape(end_flat, {batch_size, seq_len});

    return ElectraQAOutput{start_logits, end_logits};
}

auto ElectraForQuestionAnswering::forward(const Variable& input) -> Variable {
    auto outputs = forward(input, Tensor{}, Variable{});
    return outputs.start_logits;
}

} // namespace models
} // namespace tenzor
