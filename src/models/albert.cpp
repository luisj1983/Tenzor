/**
 * @file albert.cpp
 * @brief Implementation of ALBERT model family
 */

#include "tenzor/models/albert.hpp"
#include "tenzor/models/hub.hpp"  // Audit H4: ModelHub::load_pretrained_weights
#include "tenzor/core/tensor.hpp"
#include "embedding_utils.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor {
namespace models {

// ============================================================================
// AlbertEmbeddings Implementation
// ============================================================================

AlbertEmbeddings::AlbertEmbeddings(const AlbertConfig& config)
    : config_(config) {
    // Initialize small embedding layers (V×E, P×E, T×E)
    word_embeddings_ = std::make_shared<nn::Embedding>(
        config.vocab_size, config.embedding_size);
    position_embeddings_ = std::make_shared<nn::Embedding>(
        config.max_position_embeddings, config.embedding_size);
    token_type_embeddings_ = std::make_shared<nn::Embedding>(
        config.type_vocab_size, config.embedding_size);

    // Projection layer to expand from E to H
    embedding_projection_ = std::make_shared<nn::Linear>(
        config.embedding_size, config.hidden_size);

    // Layer normalization and dropout (applied after projection)
    layer_norm_ = std::make_shared<nn::LayerNorm>(
        std::vector<int64_t>{config.hidden_size}, config.layer_norm_eps);
    dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);

    // Register modules
    register_module("word_embeddings", word_embeddings_);
    register_module("position_embeddings", position_embeddings_);
    register_module("token_type_embeddings", token_type_embeddings_);
    register_module("embedding_projection", embedding_projection_);
    register_module("layer_norm", layer_norm_);
    register_module("dropout", dropout_);
}

auto AlbertEmbeddings::create_position_ids(const Tensor& input_ids) -> Tensor {
    return make_sequential_position_ids(input_ids);
}

auto AlbertEmbeddings::forward(const Variable& input_ids,
                                const Variable& token_type_ids,
                                const Variable& position_ids) -> Variable {
    auto shape = input_ids.tensor().shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];

    // Get word embeddings (V×E)
    auto embeddings = word_embeddings_->forward(input_ids);

    // Get position embeddings (P×E)
    Variable pos_ids = position_ids;
    if (!position_ids.is_initialized() || position_ids.tensor().numel() == 0) {
        auto pos_tensor = create_position_ids(input_ids.tensor());
        pos_ids = Variable(pos_tensor, false);
    }
    auto position_embeddings = position_embeddings_->forward(pos_ids);

    // Get token type embeddings (T×E)
    Variable type_ids = token_type_ids;
    if (!token_type_ids.is_initialized() || token_type_ids.tensor().numel() == 0) {
        auto target_device = input_ids.tensor().device();
        // Create zeros on CPU then move to device
        Tensor zeros_cpu({batch_size, seq_len}, DType::Int64, Device::cpu());
        zeros_cpu.zero_();
        auto zeros_tensor = (target_device == Device::cpu()) ? zeros_cpu : zeros_cpu.to(target_device);
        type_ids = Variable(zeros_tensor, false);
    }
    auto token_type_embeddings = token_type_embeddings_->forward(type_ids);

    // Combine embeddings (still dimension E)
    embeddings = embeddings + position_embeddings + token_type_embeddings;

    // Project from E to H (factorized embedding key innovation)
    embeddings = embedding_projection_->forward(embeddings);

    // Apply layer normalization and dropout
    embeddings = layer_norm_->forward(embeddings);
    embeddings = dropout_->forward(embeddings);

    return embeddings;
}

auto AlbertEmbeddings::forward_impl(const Variable& input) -> Variable {
    return forward(input, Variable{}, Variable{});
}

// ============================================================================
// AlbertEncoder Implementation
// ============================================================================

AlbertEncoder::AlbertEncoder(const AlbertConfig& config)
    : config_(config) {
    // Create a SINGLE transformer layer that will be shared across all layers
    // This is the key innovation: cross-layer parameter sharing
    shared_layer_ = std::make_shared<nn::TransformerEncoderLayer>(
        config.hidden_size,
        config.num_attention_heads,
        config.intermediate_size,
        config.attention_probs_dropout_prob,
        config.hidden_act,
        true  // batch_first
    );

    register_module("shared_layer", shared_layer_);
}

auto AlbertEncoder::forward(const Variable& hidden_states,
                             const Tensor& attention_mask) -> Variable {
    auto output = hidden_states;

    // Apply the SAME layer multiple times (parameter sharing)
    // This reduces parameters by ~91% compared to BERT
    for (int64_t i = 0; i < config_.num_hidden_layers; ++i) {
        output = shared_layer_->forward(output, attention_mask, Tensor{});
    }

    return output;
}

auto AlbertEncoder::forward_impl(const Variable& input) -> Variable {
    return forward(input, Tensor{});
}

// ============================================================================
// AlbertPooler Implementation
// ============================================================================

AlbertPooler::AlbertPooler(const AlbertConfig& config) {
    // Linear transformation for pooling
    dense_ = std::make_shared<nn::Linear>(config.hidden_size, config.hidden_size);
    register_module("dense", dense_);
}

auto AlbertPooler::forward_impl(const Variable& hidden_states) -> Variable {
    // Extract [CLS] token (first token) representation
    // hidden_states: [batch, seq_len, hidden_size]
    // Use tensor slicing instead of manual memory copies for device compatibility

    // Slice first token: [batch, seq_len, hidden_size] -> [batch, 1, hidden_size] -> [batch, hidden_size]
    auto first_token_tensor = hidden_states.tensor().slice(1, 0, 1).squeeze(1);
    Variable first_token(first_token_tensor, hidden_states.requires_grad());

    // Apply linear transformation and tanh activation
    auto pooled = dense_->forward(first_token);
    pooled = nn::tanh(pooled);

    return pooled;
}

// ============================================================================
// AlbertModel Implementation
// ============================================================================

AlbertModel::AlbertModel(const AlbertConfig& config)
    : config_(config) {
    embeddings_ = std::make_shared<AlbertEmbeddings>(config);
    encoder_ = std::make_shared<AlbertEncoder>(config);
    pooler_ = std::make_shared<AlbertPooler>(config);

    register_module("embeddings", embeddings_);
    register_module("encoder", encoder_);
    register_module("pooler", pooler_);
}

auto AlbertModel::forward(const Variable& input_ids,
                           const Tensor& attention_mask,
                           const Variable& token_type_ids,
                           const Variable& position_ids) -> AlbertOutput {
    // Get embeddings
    auto embedding_output = embeddings_->forward(input_ids, token_type_ids, position_ids);

    // Encode
    auto sequence_output = encoder_->forward(embedding_output, attention_mask);

    // Pool
    auto pooled_output = pooler_->forward(sequence_output);

    return AlbertOutput{sequence_output, pooled_output};
}

auto AlbertModel::forward_impl(const Variable& input) -> Variable {
    auto outputs = forward(input, Tensor{}, Variable{}, Variable{});
    return outputs.sequence_output;
}

auto AlbertModel::load_pretrained(const std::string& path, bool strict) -> void {
    // Audit H4: thin wrapper around ModelHub's format-dispatching loader.
    // Accepts SafeTensors and native Tenzor checkpoints; .pth throws a clear
    // error pointing to the SafeTensors variant (H2-followup for pickle).
    ModelHub::load_pretrained_weights(*this, path, strict);
}

// ============================================================================
// AlbertForSequenceClassification Implementation
// ============================================================================

AlbertForSequenceClassification::AlbertForSequenceClassification(
    const AlbertConfig& config, int64_t num_labels)
    : config_(config), num_labels_(num_labels) {
    albert_ = std::make_shared<AlbertModel>(config);
    dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);
    classifier_ = std::make_shared<nn::Linear>(config.hidden_size, num_labels);

    register_module("albert", albert_);
    register_module("dropout", dropout_);
    register_module("classifier", classifier_);
}

auto AlbertForSequenceClassification::forward(const Variable& input_ids,
                                                const Tensor& attention_mask,
                                                const Variable& token_type_ids) -> Variable {
    auto outputs = albert_->forward(input_ids, attention_mask, token_type_ids);
    auto pooled_output = outputs.pooled_output;

    pooled_output = dropout_->forward(pooled_output);
    auto logits = classifier_->forward(pooled_output);

    return logits;
}

auto AlbertForSequenceClassification::forward_impl(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// AlbertForTokenClassification Implementation
// ============================================================================

AlbertForTokenClassification::AlbertForTokenClassification(
    const AlbertConfig& config, int64_t num_labels)
    : config_(config), num_labels_(num_labels) {
    albert_ = std::make_shared<AlbertModel>(config);
    dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);
    classifier_ = std::make_shared<nn::Linear>(config.hidden_size, num_labels);

    register_module("albert", albert_);
    register_module("dropout", dropout_);
    register_module("classifier", classifier_);
}

auto AlbertForTokenClassification::forward(const Variable& input_ids,
                                             const Tensor& attention_mask,
                                             const Variable& token_type_ids) -> Variable {
    auto outputs = albert_->forward(input_ids, attention_mask, token_type_ids);
    auto sequence_output = outputs.sequence_output;

    sequence_output = dropout_->forward(sequence_output);
    auto logits = classifier_->forward(sequence_output);

    return logits;
}

auto AlbertForTokenClassification::forward_impl(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// AlbertForPreTraining Implementation
// ============================================================================

AlbertForPreTraining::AlbertForPreTraining(const AlbertConfig& config)
    : config_(config) {
    albert_ = std::make_shared<AlbertModel>(config);

    // MLM head: project to vocabulary size
    mlm_head_ = std::make_shared<nn::Linear>(config.hidden_size, config.vocab_size);

    // SOP classifier: binary classification (correct order vs swapped)
    sop_classifier_ = std::make_shared<nn::Linear>(config.hidden_size, 2);

    register_module("albert", albert_);
    register_module("mlm_head", mlm_head_);
    register_module("sop_classifier", sop_classifier_);
}

auto AlbertForPreTraining::forward(const Variable& input_ids,
                                     const Tensor& attention_mask,
                                     const Variable& token_type_ids)
    -> std::tuple<Variable, Variable> {
    auto outputs = albert_->forward(input_ids, attention_mask, token_type_ids);

    // MLM predictions on all tokens
    auto mlm_logits = mlm_head_->forward(outputs.sequence_output);

    // SOP predictions on pooled [CLS] representation
    auto sop_logits = sop_classifier_->forward(outputs.pooled_output);

    return {mlm_logits, sop_logits};
}

auto AlbertForPreTraining::forward_impl(const Variable& input) -> Variable {
    auto [mlm_logits, sop_logits] = forward(input, Tensor{}, Variable{});
    return mlm_logits;  // Return MLM logits as default
}

} // namespace models
} // namespace tenzor
