/**
 * @file roberta.cpp
 * @brief Implementation of RoBERTa model family
 */

#include "tenzor/models/roberta.hpp"
#include "tenzor/models/hub.hpp"  // Audit H4
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"

namespace tenzor {
namespace models {

// ============================================================================
// RobertaEmbeddings Implementation
// ============================================================================

RobertaEmbeddings::RobertaEmbeddings(const RobertaConfig& config)
    : BertEmbeddings(config.to_bert_config()) {
    // Reuses BertEmbeddings with RoBERTa-specific config
    // Key differences:
    // - vocab_size: 50265 (byte-level BPE)
    // - type_vocab_size: 1 (no segment embeddings)
    // - layer_norm_eps: 1e-5 (vs BERT's 1e-12)
}

// ============================================================================
// RobertaEncoder Implementation
// ============================================================================

RobertaEncoder::RobertaEncoder(const RobertaConfig& config)
    : BertEncoder(config.to_bert_config()) {
    // Reuses BertEncoder - identical architecture
}

// ============================================================================
// RobertaPooler Implementation
// ============================================================================

RobertaPooler::RobertaPooler(const RobertaConfig& config)
    : BertPooler(config.to_bert_config()) {
    // Reuses BertPooler - identical implementation
}

// ============================================================================
// RobertaModel Implementation
// ============================================================================

RobertaModel::RobertaModel(const RobertaConfig& config)
    : config_(config) {
    // Initialize components (reuse BERT components with RoBERTa config)
    embeddings_ = std::make_shared<RobertaEmbeddings>(config);
    encoder_ = std::make_shared<RobertaEncoder>(config);
    pooler_ = std::make_shared<RobertaPooler>(config);

    register_module("embeddings", embeddings_);
    register_module("encoder", encoder_);
    register_module("pooler", pooler_);
}

auto RobertaModel::forward(const Variable& input_ids,
                           const Tensor& attention_mask,
                           const Variable& token_type_ids,
                           const Variable& position_ids) -> RobertaOutput {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    // Get embeddings
    auto embedding_output = embeddings_->forward(input_ids, token_type_ids, position_ids);

    // Encode
    auto sequence_output = encoder_->forward(embedding_output, attention_mask);

    // Pool
    auto pooled_output = pooler_->forward(sequence_output);

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return RobertaOutput{sequence_output, pooled_output};
}

auto RobertaModel::forward_impl(const Variable& input) -> Variable {
    auto outputs = forward(input, Tensor{}, Variable{}, Variable{});
    return outputs.sequence_output;
}

auto RobertaModel::load_pretrained(const std::string& path, bool strict) -> void {
    // Audit H4. See AlbertModel::load_pretrained for details.
    ModelHub::load_pretrained_weights(*this, path, strict);
}

// ============================================================================
// RobertaForSequenceClassification Implementation
// ============================================================================

RobertaForSequenceClassification::RobertaForSequenceClassification(
    const RobertaConfig& config, int64_t num_labels)
    : config_(config), num_labels_(num_labels) {
    // Base RoBERTa model
    roberta_ = std::make_shared<RobertaModel>(config);

    // Classification head
    dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);
    classifier_ = std::make_shared<nn::Linear>(config.hidden_size, num_labels);

    register_module("roberta", roberta_);
    register_module("dropout", dropout_);
    register_module("classifier", classifier_);
}

auto RobertaForSequenceClassification::forward(const Variable& input_ids,
                                                const Tensor& attention_mask,
                                                const Variable& token_type_ids) -> Variable {
    // Call forward pre-hooks (enables CPU-start offloading)
    call_forward_pre_hooks();

    // Get RoBERTa outputs
    auto outputs = roberta_->forward(input_ids, attention_mask, token_type_ids, Variable{});

    // Use pooled output
    auto pooled_output = outputs.pooled_output;

    // Apply dropout
    pooled_output = dropout_->forward(pooled_output);

    // Classify
    auto logits = classifier_->forward(pooled_output);

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return logits;
}

auto RobertaForSequenceClassification::forward_impl(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// RobertaForTokenClassification Implementation
// ============================================================================

RobertaForTokenClassification::RobertaForTokenClassification(
    const RobertaConfig& config, int64_t num_labels)
    : config_(config), num_labels_(num_labels) {
    // Base RoBERTa model
    roberta_ = std::make_shared<RobertaModel>(config);

    // Token classification head
    dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);
    classifier_ = std::make_shared<nn::Linear>(config.hidden_size, num_labels);

    register_module("roberta", roberta_);
    register_module("dropout", dropout_);
    register_module("classifier", classifier_);
}

auto RobertaForTokenClassification::forward(const Variable& input_ids,
                                             const Tensor& attention_mask,
                                             const Variable& token_type_ids) -> Variable {
    // Call forward pre-hooks (enables CPU-start offloading)
    call_forward_pre_hooks();

    // Get RoBERTa outputs
    auto outputs = roberta_->forward(input_ids, attention_mask, token_type_ids, Variable{});

    // Use sequence output (token-level representations)
    auto sequence_output = outputs.sequence_output;

    // Apply dropout
    sequence_output = dropout_->forward(sequence_output);

    // Classify each token
    auto logits = classifier_->forward(sequence_output);

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return logits;
}

auto RobertaForTokenClassification::forward_impl(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// RobertaForQuestionAnswering Implementation
// ============================================================================

RobertaForQuestionAnswering::RobertaForQuestionAnswering(const RobertaConfig& config)
    : config_(config) {
    // Base RoBERTa model
    roberta_ = std::make_shared<RobertaModel>(config);

    // QA output layer (predicts start and end positions)
    qa_outputs_ = std::make_shared<nn::Linear>(config.hidden_size, 2);

    register_module("roberta", roberta_);
    register_module("qa_outputs", qa_outputs_);
}

auto RobertaForQuestionAnswering::forward(const Variable& input_ids,
                                          const Tensor& attention_mask,
                                          const Variable& token_type_ids) -> RobertaQAOutput {
    // Call forward pre-hooks (enables CPU-start offloading)
    call_forward_pre_hooks();

    // Get RoBERTa outputs
    auto outputs = roberta_->forward(input_ids, attention_mask, token_type_ids, Variable{});

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
    // Use the same dtype as logits for consistency
    auto dtype = logits.tensor().dtype();

    // Start logits: multiply by [1, 0]
    // Create selector in Float32 then convert to target dtype for Float16 compatibility
    Tensor start_selector(std::vector<int64_t>{2, 1}, DType::Float32, Device::cpu());
    start_selector.zero_();
    start_selector.data<float>()[0] = 1.0f;
    if (dtype != DType::Float32) {
        start_selector = start_selector.to(dtype);
    }
    start_selector = start_selector.to(logits.tensor().device());

    // End logits: multiply by [0, 1]
    Tensor end_selector(std::vector<int64_t>{2, 1}, DType::Float32, Device::cpu());
    end_selector.zero_();
    end_selector.data<float>()[1] = 1.0f;
    if (dtype != DType::Float32) {
        end_selector = end_selector.to(dtype);
    }
    end_selector = end_selector.to(logits.tensor().device());

    // Use matmul to select: [batch*seq_len, 2] @ [2, 1] = [batch*seq_len, 1]
    Variable start_selector_var(start_selector, false);
    Variable end_selector_var(end_selector, false);

    auto start_flat = tenzor::matmul(reshaped, start_selector_var);  // [batch*seq_len, 1]
    auto end_flat = tenzor::matmul(reshaped, end_selector_var);      // [batch*seq_len, 1]

    // Reshape back to [batch, seq_len]
    auto start_logits = tenzor::reshape(start_flat, {batch_size, seq_len});
    auto end_logits = tenzor::reshape(end_flat, {batch_size, seq_len});

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return RobertaQAOutput{start_logits, end_logits};
}

auto RobertaForQuestionAnswering::forward_impl(const Variable& input) -> Variable {
    auto outputs = forward(input, Tensor{}, Variable{});
    return outputs.start_logits;
}

} // namespace models
} // namespace tenzor
