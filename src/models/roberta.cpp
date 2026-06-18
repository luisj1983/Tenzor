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

auto RobertaEmbeddings::create_position_ids(const Tensor& input_ids) -> Tensor {
    // RoBERTa derives position IDs from the input token IDs: each non-padding
    // token gets position padding_idx + (its 1-based index among non-padding
    // tokens), and padding tokens map to padding_idx. This differs from BERT's
    // plain [0..seq_len-1] and matches HF create_position_ids_from_input_ids.
    constexpr int64_t padding_idx = 1;  // RoBERTa pad_token_id

    auto ids_cpu = input_ids.to(Device::cpu()).to(DType::Int64);
    auto shape = ids_cpu.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];
    const int64_t* in = ids_cpu.data<int64_t>();

    Tensor pos_cpu(std::vector<int64_t>{batch_size, seq_len}, DType::Int64, Device::cpu());
    int64_t* out = pos_cpu.data<int64_t>();
    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t cumulative = 0;
        for (int64_t t = 0; t < seq_len; ++t) {
            if (in[b * seq_len + t] != padding_idx) {
                cumulative += 1;
                out[b * seq_len + t] = cumulative + padding_idx;
            } else {
                out[b * seq_len + t] = padding_idx;
            }
        }
    }

    return (input_ids.device() == Device::cpu()) ? pos_cpu : pos_cpu.to(input_ids.device());
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

    // Split into start and end logits while preserving gradients.
    // logits: [batch, seq_len, 2]. Extract channel 0 (start) and channel 1
    // (end) via autograd-aware slice + squeeze on the last dim. This preserves
    // the grad_fn chain and sidesteps the old selection-matrix path (which only
    // filled Float32/Float64/Float16 selectors, produced an all-zero selector
    // for BFloat16, and added an unnecessary matmul) — matching the sibling
    // BertForQuestionAnswering / ElectraForQuestionAnswering heads.
    auto start_logits = tenzor::squeeze(tenzor::slice(logits, 2, 0, 1), 2);  // [batch, seq_len]
    auto end_logits   = tenzor::squeeze(tenzor::slice(logits, 2, 1, 2), 2);  // [batch, seq_len]

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
