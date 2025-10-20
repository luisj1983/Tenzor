/**
 * @file gpt.cpp
 * @brief Implementation of GPT models
 */

#include "../../include/tenzor/models/gpt.hpp"
#include "../../include/tenzor/autograd/variable.hpp"
#include "../../include/tenzor/autograd/ops.hpp"
#include "../../include/tenzor/nn/activations/activations.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <chrono>

namespace tenzor {
namespace models {

// ============================================================================
// GPTEmbeddings Implementation
// ============================================================================

GPTEmbeddings::GPTEmbeddings(const GPT2Config& config)
    : n_positions_(config.n_positions) {
    // Token embeddings
    token_embedding_ = std::make_shared<nn::Embedding>(
        config.vocab_size,
        config.n_embd
    );

    // Position embeddings (learned)
    position_embedding_ = std::make_shared<nn::Embedding>(
        config.n_positions,
        config.n_embd
    );

    // Dropout
    dropout_ = std::make_shared<nn::Dropout>(config.embd_pdrop);

    // Register submodules
    register_module("token_embedding", token_embedding_);
    register_module("position_embedding", position_embedding_);
    register_module("dropout", dropout_);
}

auto GPTEmbeddings::forward(const Variable& input_ids, const Variable& position_ids) -> Variable {
    auto batch_size = input_ids.tensor().shape()[0];
    auto seq_len = input_ids.tensor().shape()[1];

    // Get token embeddings
    auto token_embeds = token_embedding_->forward(input_ids);

    // Create or use provided position IDs
    Variable pos_ids = position_ids;
    if (!position_ids.is_initialized() || position_ids.tensor().numel() == 0) {
        // Create sequential position IDs [0, 1, 2, ..., seq_len-1]
        std::vector<int64_t> pos_data(seq_len);
        std::iota(pos_data.begin(), pos_data.end(), 0);

        Tensor pos_tensor(std::vector<int64_t>{1, seq_len}, DType::Int64, input_ids.tensor().device());
        std::copy(pos_data.begin(), pos_data.end(), pos_tensor.data<int64_t>());

        pos_ids = Variable(pos_tensor, false);
    }

    // Get position embeddings
    auto position_embeds = position_embedding_->forward(pos_ids);

    // Add token and position embeddings
    auto embeddings = token_embeds + position_embeds;

    // Apply dropout
    return dropout_->forward(embeddings);
}

auto GPTEmbeddings::forward(const Variable& input) -> Variable {
    return forward(input, Variable{});
}

// ============================================================================
// GPTDecoderLayer Implementation
// ============================================================================

GPTDecoderLayer::GPTDecoderLayer(const GPT2Config& config)
    : activation_(config.activation) {
    // Layer normalization (pre-norm)
    ln_1_ = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.n_embd}, config.layer_norm_epsilon);
    ln_2_ = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.n_embd}, config.layer_norm_epsilon);

    // Causal self-attention
    attn_ = std::make_shared<nn::MultiheadAttention>(
        config.n_embd,
        config.n_head,
        config.attn_pdrop,
        true,   // bias
        false,  // add_bias_kv
        false,  // add_zero_attn
        0,      // kdim (0 = use embed_dim)
        0,      // vdim (0 = use embed_dim)
        true    // batch_first
    );

    // Feed-forward network (MLP)
    mlp_fc_ = std::make_shared<nn::Linear>(config.n_embd, config.n_inner);
    mlp_proj_ = std::make_shared<nn::Linear>(config.n_inner, config.n_embd);

    // Dropout
    dropout_ = std::make_shared<nn::Dropout>(config.resid_pdrop);

    // Register submodules
    register_module("ln_1", ln_1_);
    register_module("ln_2", ln_2_);
    register_module("attn", attn_);
    register_module("mlp_fc", mlp_fc_);
    register_module("mlp_proj", mlp_proj_);
    register_module("dropout", dropout_);
}

auto GPTDecoderLayer::apply_activation(const Variable& x) const -> Variable {
    if (activation_ == "gelu") {
        return nn::gelu(x);
    } else if (activation_ == "relu") {
        return nn::relu(x);
    } else {
        throw std::runtime_error("Unknown activation: " + activation_);
    }
}

auto GPTDecoderLayer::forward(const Variable& hidden_states,
                              const Tensor& attention_mask,
                              bool use_cache) -> Variable {
    // Self-attention with pre-norm (GPT-2 style)
    auto residual = hidden_states;
    auto x = ln_1_->forward(hidden_states);

    // Apply causal self-attention
    auto [attn_output, _] = attn_->forward(x, x, x, Tensor{}, attention_mask, false);

    // Residual connection with dropout
    x = residual + dropout_->forward(attn_output);

    // Feed-forward network with pre-norm
    residual = x;
    x = ln_2_->forward(x);

    // MLP: Linear -> Activation -> Linear
    x = mlp_fc_->forward(x);
    x = apply_activation(x);
    x = mlp_proj_->forward(x);

    // Residual connection with dropout
    x = residual + dropout_->forward(x);

    return x;
}

auto GPTDecoderLayer::forward(const Variable& input) -> Variable {
    return forward(input, Tensor{}, false);
}

// ============================================================================
// GPT2Model Implementation
// ============================================================================

GPT2Model::GPT2Model(const GPT2Config& config)
    : config_(config) {
    // Embeddings
    embeddings_ = std::make_shared<GPTEmbeddings>(config);

    // Decoder layers
    for (int64_t i = 0; i < config.n_layer; ++i) {
        auto layer = std::make_shared<GPTDecoderLayer>(config);
        layers_.push_back(layer);
        register_module("layer_" + std::to_string(i), layer);
    }

    // Final layer norm
    ln_f_ = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.n_embd}, config.layer_norm_epsilon);

    // Dropout
    dropout_ = std::make_shared<nn::Dropout>(config.resid_pdrop);

    // Register modules
    register_module("embeddings", embeddings_);
    register_module("ln_f", ln_f_);
    register_module("dropout", dropout_);
}

auto GPT2Model::create_causal_attention_mask(int64_t seq_len, Device device) const -> Tensor {
    return nn::create_causal_mask(seq_len, device);
}

auto GPT2Model::forward(const Variable& input_ids,
                       const Variable& position_ids,
                       const Tensor& attention_mask) -> Variable {
    auto seq_len = input_ids.tensor().shape()[1];

    // Get embeddings
    auto hidden_states = embeddings_->forward(input_ids, position_ids);

    // Create causal mask if not provided
    Tensor causal_mask = attention_mask;
    if (attention_mask.numel() == 0) {
        causal_mask = create_causal_attention_mask(seq_len, input_ids.tensor().device());
    }

    // Pass through decoder layers
    for (auto& layer : layers_) {
        hidden_states = layer->forward(hidden_states, causal_mask);
    }

    // Final layer norm
    hidden_states = ln_f_->forward(hidden_states);

    return hidden_states;
}

auto GPT2Model::forward(const Variable& input) -> Variable {
    return forward(input, Variable{}, Tensor{});
}

// ============================================================================
// GPT2LMHeadModel Implementation
// ============================================================================

GPT2LMHeadModel::GPT2LMHeadModel(const GPT2Config& config)
    : GPT2Model(config) {
    // Language modeling head (linear layer to vocabulary)
    lm_head_ = std::make_shared<nn::Linear>(config.n_embd, config.vocab_size, false);

    // Register module
    register_module("lm_head", lm_head_);
}

auto GPT2LMHeadModel::forward(const Variable& input_ids,
                              const Variable& position_ids,
                              const Tensor& attention_mask) -> Variable {
    // Get hidden states from base model
    auto hidden_states = GPT2Model::forward(input_ids, position_ids, attention_mask);

    // Project to vocabulary logits
    auto logits = lm_head_->forward(hidden_states);

    return logits;
}

auto GPT2LMHeadModel::forward(const Variable& input) -> Variable {
    return forward(input, Variable{}, Tensor{});
}

// ============================================================================
// GPT3Model Implementation
// ============================================================================

GPT3Model::GPT3Model(const GPT3Config& config)
    : GPT2Model(config) {
    // GPT-3 uses same architecture as GPT-2, just scaled up
}

// ============================================================================
// GPT3LMHeadModel Implementation
// ============================================================================

GPT3LMHeadModel::GPT3LMHeadModel(const GPT3Config& config)
    : GPT2LMHeadModel(config) {
    // GPT-3 LM head is same as GPT-2, just with larger dimensions
}

// ============================================================================
// TextGenerator Implementation
// ============================================================================

TextGenerator::TextGenerator(GPT2LMHeadModel& model, const GenerationConfig& config)
    : model_(model), config_(config), rng_(config.seed) {
}

auto TextGenerator::apply_temperature(const Variable& logits, double temperature) const -> Variable {
    if (temperature == 1.0) {
        return logits;
    }
    // Scale logits by dividing by temperature
    auto temp_tensor = Tensor(std::vector<int64_t>{1}, DType::Float32, logits.tensor().device());
    temp_tensor.data<float>()[0] = static_cast<float>(temperature);
    Variable temp_var(temp_tensor, false);
    return logits / temp_var;
}

auto TextGenerator::logits_to_probs(const Variable& logits, double temperature) const -> Variable {
    auto scaled_logits = apply_temperature(logits, temperature);
    return nn::softmax(scaled_logits, -1);
}

auto TextGenerator::sample_from_probs(const Tensor& probs) -> int64_t {
    // Sample from categorical distribution
    const float* probs_data = probs.data<float>();
    auto vocab_size = probs.shape()[probs.ndim() - 1];

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float sample = dist(rng_);

    float cumsum = 0.0f;
    for (int64_t i = 0; i < vocab_size; ++i) {
        cumsum += probs_data[i];
        if (sample < cumsum) {
            return i;
        }
    }
    return vocab_size - 1;  // Fallback
}

auto TextGenerator::top_k_filter(const Tensor& logits, int64_t k) const -> std::pair<Tensor, Tensor> {
    auto vocab_size = logits.shape()[logits.ndim() - 1];

    // Get logits for last position
    std::vector<std::pair<float, int64_t>> logit_pairs;
    const float* logits_data = logits.data<float>();

    for (int64_t i = 0; i < vocab_size; ++i) {
        logit_pairs.push_back({logits_data[i], i});
    }

    // Partial sort to get top k
    std::partial_sort(logit_pairs.begin(), logit_pairs.begin() + k, logit_pairs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    // Create filtered tensors
    Tensor filtered_logits(std::vector<int64_t>{k}, DType::Float32, logits.device());
    Tensor indices(std::vector<int64_t>{k}, DType::Int64, logits.device());

    float* filtered_data = filtered_logits.data<float>();
    int64_t* indices_data = indices.data<int64_t>();

    for (int64_t i = 0; i < k; ++i) {
        filtered_data[i] = logit_pairs[i].first;
        indices_data[i] = logit_pairs[i].second;
    }

    return {filtered_logits, indices};
}

auto TextGenerator::top_p_filter(const Tensor& probs, double p) const -> std::pair<Tensor, Tensor> {
    auto vocab_size = probs.shape()[probs.ndim() - 1];

    // Get probs for last position
    std::vector<std::pair<float, int64_t>> prob_pairs;
    const float* probs_data = probs.data<float>();

    for (int64_t i = 0; i < vocab_size; ++i) {
        prob_pairs.push_back({probs_data[i], i});
    }

    // Sort by probability (descending)
    std::sort(prob_pairs.begin(), prob_pairs.end(),
             [](const auto& a, const auto& b) { return a.first > b.first; });

    // Find nucleus (smallest set with cumulative prob >= p)
    float cumsum = 0.0f;
    int64_t nucleus_size = 0;

    for (size_t i = 0; i < prob_pairs.size(); ++i) {
        cumsum += prob_pairs[i].first;
        nucleus_size++;
        if (cumsum >= p) {
            break;
        }
    }

    // Create filtered tensors
    Tensor filtered_probs(std::vector<int64_t>{nucleus_size}, DType::Float32, probs.device());
    Tensor indices(std::vector<int64_t>{nucleus_size}, DType::Int64, probs.device());

    float* filtered_data = filtered_probs.data<float>();
    int64_t* indices_data = indices.data<int64_t>();

    // Renormalize probabilities
    float renorm_sum = 0.0f;
    for (int64_t i = 0; i < nucleus_size; ++i) {
        renorm_sum += prob_pairs[i].first;
    }

    for (int64_t i = 0; i < nucleus_size; ++i) {
        filtered_data[i] = prob_pairs[i].first / renorm_sum;
        indices_data[i] = prob_pairs[i].second;
    }

    return {filtered_probs, indices};
}

auto TextGenerator::generate(const Tensor& input_ids) -> Tensor {
    switch (config_.strategy) {
        case GenerationStrategy::Greedy:
            return greedy_search(input_ids);
        case GenerationStrategy::TopK:
            return top_k_sampling(input_ids, config_.top_k, config_.temperature);
        case GenerationStrategy::TopP:
            return top_p_sampling(input_ids, config_.top_p, config_.temperature);
        case GenerationStrategy::BeamSearch:
            return beam_search(input_ids, config_.num_beams);
        default:
            throw std::runtime_error("Unknown generation strategy");
    }
}

auto TextGenerator::greedy_search(const Tensor& input_ids) -> Tensor {
    model_.eval();

    auto batch_size = input_ids.shape()[0];
    auto current_len = input_ids.shape()[1];

    // Copy input to output buffer
    std::vector<int64_t> output_shape = {batch_size, config_.max_length};
    Tensor output(output_shape, DType::Int64, input_ids.device());

    // Copy input_ids to output
    auto input_data = input_ids.data<int64_t>();
    auto output_data = output.data<int64_t>();

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < current_len; ++i) {
            output_data[b * config_.max_length + i] = input_data[b * current_len + i];
        }
        // Pad rest with 0 (will be filled during generation)
        for (int64_t i = current_len; i < config_.max_length; ++i) {
            output_data[b * config_.max_length + i] = 0;
        }
    }

    // Generate tokens autoregressively
    for (int64_t step = current_len; step < config_.max_length; ++step) {
        // Get current sequence
        Tensor current_seq = output.slice(1, 0, step, 1);
        Variable current_var(current_seq, false);

        // Forward pass - explicitly use the 3-argument overload
        auto logits = model_.forward(current_var, Variable{}, Tensor{});

        // Get logits for last position [batch, vocab_size]
        auto last_logits = logits.tensor().slice(1, step - 1, step, 1).squeeze(1);

        // Get argmax (greedy)
        const float* logits_data = last_logits.data<float>();
        auto vocab_size = last_logits.shape()[last_logits.ndim() - 1];

        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t max_idx = 0;
            float max_val = logits_data[b * vocab_size];

            for (int64_t v = 1; v < vocab_size; ++v) {
                float val = logits_data[b * vocab_size + v];
                if (val > max_val) {
                    max_val = val;
                    max_idx = v;
                }
            }

            output_data[b * config_.max_length + step] = max_idx;

            // Check for EOS token
            if (config_.eos_token_id >= 0 && max_idx == config_.eos_token_id) {
                // Could implement early stopping here
            }
        }
    }

    return output;
}

auto TextGenerator::top_k_sampling(const Tensor& input_ids, int64_t top_k, double temperature) -> Tensor {
    model_.eval();

    auto batch_size = input_ids.shape()[0];
    auto current_len = input_ids.shape()[1];

    // Copy input to output buffer
    std::vector<int64_t> output_shape = {batch_size, config_.max_length};
    Tensor output(output_shape, DType::Int64, input_ids.device());

    auto input_data = input_ids.data<int64_t>();
    auto output_data = output.data<int64_t>();

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < current_len; ++i) {
            output_data[b * config_.max_length + i] = input_data[b * current_len + i];
        }
        for (int64_t i = current_len; i < config_.max_length; ++i) {
            output_data[b * config_.max_length + i] = 0;
        }
    }

    // Generate tokens with top-k sampling
    for (int64_t step = current_len; step < config_.max_length; ++step) {
        Tensor current_seq = output.slice(1, 0, step, 1);
        Variable current_var(current_seq, false);

        auto logits = model_.forward(current_var, Variable{}, Tensor{});
        auto last_logits = logits.tensor().slice(1, step - 1, step, 1).squeeze(1);

        for (int64_t b = 0; b < batch_size; ++b) {
            // Get logits for this batch item
            auto batch_start = b * last_logits.shape()[last_logits.ndim() - 1];
            Tensor batch_logits(std::vector<int64_t>{last_logits.shape()[last_logits.ndim() - 1]},
                               DType::Float32, input_ids.device());

            const float* src_data = last_logits.data<float>() + batch_start;
            float* dst_data = batch_logits.data<float>();
            std::copy(src_data, src_data + batch_logits.numel(), dst_data);

            // Apply top-k filter
            auto [filtered_logits, indices] = top_k_filter(batch_logits, top_k);

            // Convert to probabilities
            Variable filtered_var(filtered_logits, false);
            auto probs = logits_to_probs(filtered_var, temperature);

            // Sample from filtered distribution
            int64_t sampled_idx = sample_from_probs(probs.tensor());
            int64_t token_id = indices.data<int64_t>()[sampled_idx];

            output_data[b * config_.max_length + step] = token_id;
        }
    }

    return output;
}

auto TextGenerator::top_p_sampling(const Tensor& input_ids, double top_p, double temperature) -> Tensor {
    model_.eval();

    auto batch_size = input_ids.shape()[0];
    auto current_len = input_ids.shape()[1];

    // Copy input to output buffer
    std::vector<int64_t> output_shape = {batch_size, config_.max_length};
    Tensor output(output_shape, DType::Int64, input_ids.device());

    auto input_data = input_ids.data<int64_t>();
    auto output_data = output.data<int64_t>();

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < current_len; ++i) {
            output_data[b * config_.max_length + i] = input_data[b * current_len + i];
        }
        for (int64_t i = current_len; i < config_.max_length; ++i) {
            output_data[b * config_.max_length + i] = 0;
        }
    }

    // Generate tokens with top-p (nucleus) sampling
    for (int64_t step = current_len; step < config_.max_length; ++step) {
        Tensor current_seq = output.slice(1, 0, step, 1);
        Variable current_var(current_seq, false);

        auto logits = model_.forward(current_var, Variable{}, Tensor{});
        auto last_logits = logits.tensor().slice(1, step - 1, step, 1).squeeze(1);

        for (int64_t b = 0; b < batch_size; ++b) {
            // Get logits for this batch item
            auto batch_start = b * last_logits.shape()[last_logits.ndim() - 1];
            Tensor batch_logits(std::vector<int64_t>{last_logits.shape()[last_logits.ndim() - 1]},
                               DType::Float32, input_ids.device());

            const float* src_data = last_logits.data<float>() + batch_start;
            float* dst_data = batch_logits.data<float>();
            std::copy(src_data, src_data + batch_logits.numel(), dst_data);

            // Convert to probabilities
            Variable batch_var(batch_logits, false);
            auto probs = logits_to_probs(batch_var, temperature);

            // Apply top-p filter
            auto [filtered_probs, indices] = top_p_filter(probs.tensor(), top_p);

            // Sample from filtered distribution
            int64_t sampled_idx = sample_from_probs(filtered_probs);
            int64_t token_id = indices.data<int64_t>()[sampled_idx];

            output_data[b * config_.max_length + step] = token_id;
        }
    }

    return output;
}

auto TextGenerator::beam_search(const Tensor& input_ids, int64_t num_beams) -> Tensor {
    model_.eval();

    auto batch_size = input_ids.shape()[0];
    auto current_len = input_ids.shape()[1];

    if (batch_size != 1) {
        throw std::runtime_error("Beam search currently only supports batch_size=1");
    }

    // Initialize beams: each beam is (sequence, score)
    struct Beam {
        std::vector<int64_t> tokens;
        float score;
    };

    std::vector<Beam> beams(num_beams);

    // Initialize with input sequence
    auto input_data = input_ids.data<int64_t>();
    for (int64_t b = 0; b < num_beams; ++b) {
        beams[b].tokens.resize(current_len);
        for (int64_t i = 0; i < current_len; ++i) {
            beams[b].tokens[i] = input_data[i];
        }
        beams[b].score = 0.0f;
    }

    // Beam search
    for (int64_t step = current_len; step < config_.max_length; ++step) {
        std::vector<Beam> candidates;

        for (int64_t b = 0; b < num_beams; ++b) {
            // Create tensor from beam tokens
            Tensor beam_tensor(std::vector<int64_t>{1, static_cast<int64_t>(beams[b].tokens.size())},
                              DType::Int64, input_ids.device());
            int64_t* beam_data = beam_tensor.data<int64_t>();
            std::copy(beams[b].tokens.begin(), beams[b].tokens.end(), beam_data);

            // Forward pass
            Variable beam_var(beam_tensor, false);
            auto logits = model_.forward(beam_var, Variable{}, Tensor{});

            // Get log probabilities for last position
            auto last_logits = logits.tensor().slice(1, step - 1, step, 1).squeeze();
            Variable last_var(last_logits, false);
            auto log_probs_var = nn::log_softmax(last_var, -1);
            auto log_probs = log_probs_var.tensor();

            const float* log_probs_data = log_probs.data<float>();
            auto vocab_size = log_probs.numel();

            // Create candidates by extending current beam with each token
            for (int64_t v = 0; v < vocab_size; ++v) {
                Beam candidate;
                candidate.tokens = beams[b].tokens;
                candidate.tokens.push_back(v);
                candidate.score = beams[b].score + log_probs_data[v];
                candidates.push_back(candidate);
            }
        }

        // Select top num_beams candidates
        std::partial_sort(candidates.begin(), candidates.begin() + num_beams, candidates.end(),
                         [](const Beam& a, const Beam& b) { return a.score > b.score; });

        // Update beams
        for (int64_t b = 0; b < num_beams; ++b) {
            beams[b] = candidates[b];
        }
    }

    // Return best beam
    Tensor output(std::vector<int64_t>{1, config_.max_length}, DType::Int64, input_ids.device());
    int64_t* output_data = output.data<int64_t>();

    for (int64_t i = 0; i < config_.max_length && i < static_cast<int64_t>(beams[0].tokens.size()); ++i) {
        output_data[i] = beams[0].tokens[i];
    }

    return output;
}

} // namespace models
} // namespace tenzor
