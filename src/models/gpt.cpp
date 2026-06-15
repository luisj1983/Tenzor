/**
 * @file gpt.cpp
 * @brief Implementation of GPT models
 */

#include "../../include/tenzor/models/gpt.hpp"
#include "../../include/tenzor/models/hub.hpp"  // Audit H4
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
    auto seq_len = input_ids.tensor().shape()[1];

    // Get token embeddings
    auto token_embeds = token_embedding_->forward(input_ids);

    // Create or use provided position IDs
    Variable pos_ids = position_ids;
    if (!position_ids.is_initialized() || position_ids.tensor().numel() == 0) {
        // Create sequential position IDs [0, 1, 2, ..., seq_len-1]
        std::vector<int64_t> pos_data(seq_len);
        std::iota(pos_data.begin(), pos_data.end(), 0);

        // Create on CPU first, then move to target device
        Tensor pos_tensor_cpu(std::vector<int64_t>{1, seq_len}, DType::Int64, Device::cpu());
        std::copy(pos_data.begin(), pos_data.end(), pos_tensor_cpu.data<int64_t>());
        Tensor pos_tensor = pos_tensor_cpu.to(input_ids.tensor().device());

        pos_ids = Variable(pos_tensor, false);
    }

    // Get position embeddings
    auto position_embeds = position_embedding_->forward(pos_ids);

    // Add token and position embeddings
    auto embeddings = token_embeds + position_embeds;

    // Apply dropout
    return dropout_->forward(embeddings);
}

auto GPTEmbeddings::forward_impl(const Variable& input) -> Variable {
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
                              [[maybe_unused]] bool use_cache) -> Variable {
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

auto GPTDecoderLayer::forward_impl(const Variable& input) -> Variable {
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

auto GPT2Model::create_causal_attention_mask(int64_t seq_len, Device device, DType dtype) const -> Tensor {
    return nn::create_causal_mask(seq_len, device, dtype);
}

auto GPT2Model::forward(const Variable& input_ids,
                       const Variable& position_ids,
                       const Tensor& attention_mask) -> Variable {
    auto seq_len = input_ids.tensor().shape()[1];

    // Get embeddings
    auto hidden_states = embeddings_->forward(input_ids, position_ids);

    // Create causal mask if not provided
    // Use hidden_states dtype to ensure dtype compatibility
    Tensor causal_mask = attention_mask;
    if (!attention_mask.is_valid() || attention_mask.numel() == 0) {
        causal_mask = create_causal_attention_mask(seq_len, input_ids.tensor().device(),
                                                    hidden_states.tensor().dtype());
    }

    // Pass through decoder layers
    for (auto& layer : layers_) {
        hidden_states = layer->forward(hidden_states, causal_mask);
    }

    // Final layer norm
    hidden_states = ln_f_->forward(hidden_states);

    return hidden_states;
}

auto GPT2Model::forward_impl(const Variable& input) -> Variable {
    return forward(input, Variable{}, Tensor{});
}

auto GPT2Model::load_pretrained(const std::string& path, bool strict) -> void {
    // Audit H4. See AlbertModel::load_pretrained.
    ModelHub::load_pretrained_weights(*this, path, strict);
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

auto GPT2LMHeadModel::forward_impl(const Variable& input) -> Variable {
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
    // Create temperature tensor with same dtype as logits
    auto dtype = logits.tensor().dtype();
    auto temp_tensor_cpu = Tensor(std::vector<int64_t>{1}, DType::Float32, Device::cpu());
    temp_tensor_cpu.data<float>()[0] = static_cast<float>(temperature);
    // Convert to logits dtype if needed
    if (dtype != DType::Float32) {
        temp_tensor_cpu = temp_tensor_cpu.to(dtype);
    }
    auto temp_tensor = temp_tensor_cpu.to(logits.tensor().device());
    Variable temp_var(temp_tensor, false);
    return logits / temp_var;
}

auto TextGenerator::logits_to_probs(const Variable& logits, double temperature) const -> Variable {
    auto scaled_logits = apply_temperature(logits, temperature);
    return nn::softmax(scaled_logits, -1);
}

auto TextGenerator::sample_from_probs(const Tensor& probs) -> int64_t {
    // Sample from categorical distribution (move to CPU and convert to Float32 for data access)
    Tensor probs_cpu = probs.to(Device::cpu());
    if (probs_cpu.dtype() != DType::Float32) {
        probs_cpu = probs_cpu.to(DType::Float32);
    }
    const float* probs_data = probs_cpu.data<float>();
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
    auto original_device = logits.device();

    // Clamp k into [1, vocab_size]; otherwise partial_sort(begin()+k) and the copy
    // loop read past the vocab_size-element logit_pairs buffer (UB for small vocabs).
    k = std::max<int64_t>(1, std::min<int64_t>(k, vocab_size));

    // Move to CPU and convert to Float32 for processing
    Tensor logits_cpu = logits.to(Device::cpu());
    if (logits_cpu.dtype() != DType::Float32) {
        logits_cpu = logits_cpu.to(DType::Float32);
    }
    const float* logits_data = logits_cpu.data<float>();

    // Get logits for last position
    std::vector<std::pair<float, int64_t>> logit_pairs;
    for (int64_t i = 0; i < vocab_size; ++i) {
        logit_pairs.push_back({logits_data[i], i});
    }

    // Partial sort to get top k
    std::partial_sort(logit_pairs.begin(), logit_pairs.begin() + k, logit_pairs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    // Create filtered tensors on CPU
    Tensor filtered_logits_cpu(std::vector<int64_t>{k}, DType::Float32, Device::cpu());
    Tensor indices_cpu(std::vector<int64_t>{k}, DType::Int64, Device::cpu());

    float* filtered_data = filtered_logits_cpu.data<float>();
    int64_t* indices_data = indices_cpu.data<int64_t>();

    for (int64_t i = 0; i < k; ++i) {
        filtered_data[i] = logit_pairs[i].first;
        indices_data[i] = logit_pairs[i].second;
    }

    // Move back to original device
    return {filtered_logits_cpu.to(original_device), indices_cpu.to(original_device)};
}

auto TextGenerator::top_p_filter(const Tensor& probs, double p) const -> std::pair<Tensor, Tensor> {
    auto vocab_size = probs.shape()[probs.ndim() - 1];
    auto original_device = probs.device();

    // Move to CPU and convert to Float32 for processing
    Tensor probs_cpu = probs.to(Device::cpu());
    if (probs_cpu.dtype() != DType::Float32) {
        probs_cpu = probs_cpu.to(DType::Float32);
    }
    const float* probs_data = probs_cpu.data<float>();

    // Get probs for last position
    std::vector<std::pair<float, int64_t>> prob_pairs;
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

    // Create filtered tensors on CPU
    Tensor filtered_probs_cpu(std::vector<int64_t>{nucleus_size}, DType::Float32, Device::cpu());
    Tensor indices_cpu(std::vector<int64_t>{nucleus_size}, DType::Int64, Device::cpu());

    float* filtered_data = filtered_probs_cpu.data<float>();
    int64_t* indices_data = indices_cpu.data<int64_t>();

    // Renormalize probabilities
    float renorm_sum = 0.0f;
    for (int64_t i = 0; i < nucleus_size; ++i) {
        renorm_sum += prob_pairs[i].first;
    }

    for (int64_t i = 0; i < nucleus_size; ++i) {
        filtered_data[i] = prob_pairs[i].first / renorm_sum;
        indices_data[i] = prob_pairs[i].second;
    }

    // Move back to original device
    return {filtered_probs_cpu.to(original_device), indices_cpu.to(original_device)};
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

    auto original_device = input_ids.device();
    auto batch_size = input_ids.shape()[0];
    auto current_len = input_ids.shape()[1];

    // Work on CPU for data manipulation
    Tensor input_cpu = input_ids.to(Device::cpu());

    // Copy input to output buffer (on CPU)
    std::vector<int64_t> output_shape = {batch_size, config_.max_length};
    Tensor output_cpu(output_shape, DType::Int64, Device::cpu());

    // Copy input_ids to output
    auto input_data = input_cpu.data<int64_t>();
    auto output_data = output_cpu.data<int64_t>();

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < current_len; ++i) {
            output_data[b * config_.max_length + i] = input_data[b * current_len + i];
        }
        // Pad rest with 0 (will be filled during generation)
        for (int64_t i = current_len; i < config_.max_length; ++i) {
            output_data[b * config_.max_length + i] = 0;
        }
    }

    // Per-row finished flags for EOS early-stop: once a row emits
    // eos_token_id we stop updating it (pad with eos) and break the whole
    // loop once every row has finished.
    std::vector<bool> finished(batch_size, false);

    // Generate tokens autoregressively
    for (int64_t step = current_len; step < config_.max_length; ++step) {
        // Get current sequence and move to GPU for forward pass
        Tensor current_seq_cpu = output_cpu.slice(1, 0, step, 1);
        Tensor current_seq = current_seq_cpu.to(original_device);
        Variable current_var(current_seq, false);

        // Forward pass - explicitly use the 3-argument overload
        auto logits = model_.forward(current_var, Variable{}, Tensor{});

        // Get logits for last position [batch, vocab_size], move to CPU and convert to Float32
        auto last_logits = logits.tensor().slice(1, step - 1, step, 1).squeeze(1);
        Tensor last_logits_cpu = last_logits.to(Device::cpu());
        if (last_logits_cpu.dtype() != DType::Float32) {
            last_logits_cpu = last_logits_cpu.to(DType::Float32);
        }

        // Get argmax (greedy)
        const float* logits_data = last_logits_cpu.data<float>();
        auto vocab_size = last_logits.shape()[last_logits.ndim() - 1];

        for (int64_t b = 0; b < batch_size; ++b) {
            if (finished[b]) {
                // Row already emitted EOS — keep it stable by repeating the
                // EOS token rather than decoding garbage past the end.
                output_data[b * config_.max_length + step] = config_.eos_token_id;
                continue;
            }

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

            // EOS early-stop: mark this row finished once it emits eos_token_id.
            if (config_.eos_token_id >= 0 && max_idx == config_.eos_token_id) {
                finished[b] = true;
            }
        }

        // Stop generating entirely once every row has emitted EOS.
        if (config_.eos_token_id >= 0 &&
            std::all_of(finished.begin(), finished.end(), [](bool f) { return f; })) {
            break;
        }
    }

    // Return output on original device
    return output_cpu.to(original_device);
}

auto TextGenerator::top_k_sampling(const Tensor& input_ids, int64_t top_k, double temperature) -> Tensor {
    model_.eval();

    auto original_device = input_ids.device();
    auto batch_size = input_ids.shape()[0];
    auto current_len = input_ids.shape()[1];

    // Work on CPU for data manipulation
    Tensor input_cpu = input_ids.to(Device::cpu());

    // Copy input to output buffer (on CPU)
    std::vector<int64_t> output_shape = {batch_size, config_.max_length};
    Tensor output_cpu(output_shape, DType::Int64, Device::cpu());

    auto input_data = input_cpu.data<int64_t>();
    auto output_data = output_cpu.data<int64_t>();

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < current_len; ++i) {
            output_data[b * config_.max_length + i] = input_data[b * current_len + i];
        }
        for (int64_t i = current_len; i < config_.max_length; ++i) {
            output_data[b * config_.max_length + i] = 0;
        }
    }

    // Per-row finished flags for EOS early-stop (see greedy_search).
    std::vector<bool> finished(batch_size, false);

    // Generate tokens with top-k sampling
    for (int64_t step = current_len; step < config_.max_length; ++step) {
        // Move current sequence to GPU for forward pass
        Tensor current_seq_cpu = output_cpu.slice(1, 0, step, 1);
        Tensor current_seq = current_seq_cpu.to(original_device);
        Variable current_var(current_seq, false);

        auto logits = model_.forward(current_var, Variable{}, Tensor{});
        auto last_logits = logits.tensor().slice(1, step - 1, step, 1).squeeze(1);
        Tensor last_logits_cpu = last_logits.to(Device::cpu());
        // Convert to Float32 for processing
        if (last_logits_cpu.dtype() != DType::Float32) {
            last_logits_cpu = last_logits_cpu.to(DType::Float32);
        }

        for (int64_t b = 0; b < batch_size; ++b) {
            if (finished[b]) {
                output_data[b * config_.max_length + step] = config_.eos_token_id;
                continue;
            }

            // Get logits for this batch item (on CPU)
            auto batch_start = b * last_logits.shape()[last_logits.ndim() - 1];
            Tensor batch_logits(std::vector<int64_t>{last_logits.shape()[last_logits.ndim() - 1]},
                               DType::Float32, Device::cpu());

            const float* src_data = last_logits_cpu.data<float>() + batch_start;
            float* dst_data = batch_logits.data<float>();
            std::copy(src_data, src_data + batch_logits.numel(), dst_data);

            // Apply top-k filter (works on CPU internally)
            auto [filtered_logits, indices] = top_k_filter(batch_logits, top_k);

            // Convert to probabilities
            Variable filtered_var(filtered_logits, false);
            auto probs = logits_to_probs(filtered_var, temperature);

            // Sample from filtered distribution (works on CPU internally)
            int64_t sampled_idx = sample_from_probs(probs.tensor());
            Tensor indices_cpu = indices.to(Device::cpu());
            int64_t token_id = indices_cpu.data<int64_t>()[sampled_idx];

            output_data[b * config_.max_length + step] = token_id;

            // EOS early-stop: mark this row finished once it emits eos_token_id.
            if (config_.eos_token_id >= 0 && token_id == config_.eos_token_id) {
                finished[b] = true;
            }
        }

        // Stop generating entirely once every row has emitted EOS.
        if (config_.eos_token_id >= 0 &&
            std::all_of(finished.begin(), finished.end(), [](bool f) { return f; })) {
            break;
        }
    }

    // Return output on original device
    return output_cpu.to(original_device);
}

auto TextGenerator::top_p_sampling(const Tensor& input_ids, double top_p, double temperature) -> Tensor {
    model_.eval();

    auto original_device = input_ids.device();
    auto batch_size = input_ids.shape()[0];
    auto current_len = input_ids.shape()[1];

    // Work on CPU for data manipulation
    Tensor input_cpu = input_ids.to(Device::cpu());

    // Copy input to output buffer (on CPU)
    std::vector<int64_t> output_shape = {batch_size, config_.max_length};
    Tensor output_cpu(output_shape, DType::Int64, Device::cpu());

    auto input_data = input_cpu.data<int64_t>();
    auto output_data = output_cpu.data<int64_t>();

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < current_len; ++i) {
            output_data[b * config_.max_length + i] = input_data[b * current_len + i];
        }
        for (int64_t i = current_len; i < config_.max_length; ++i) {
            output_data[b * config_.max_length + i] = 0;
        }
    }

    // Per-row finished flags for EOS early-stop (see greedy_search).
    std::vector<bool> finished(batch_size, false);

    // Generate tokens with top-p (nucleus) sampling
    for (int64_t step = current_len; step < config_.max_length; ++step) {
        // Move current sequence to GPU for forward pass
        Tensor current_seq_cpu = output_cpu.slice(1, 0, step, 1);
        Tensor current_seq = current_seq_cpu.to(original_device);
        Variable current_var(current_seq, false);

        auto logits = model_.forward(current_var, Variable{}, Tensor{});
        auto last_logits = logits.tensor().slice(1, step - 1, step, 1).squeeze(1);
        Tensor last_logits_cpu = last_logits.to(Device::cpu());
        // Convert to Float32 for processing
        if (last_logits_cpu.dtype() != DType::Float32) {
            last_logits_cpu = last_logits_cpu.to(DType::Float32);
        }

        for (int64_t b = 0; b < batch_size; ++b) {
            if (finished[b]) {
                output_data[b * config_.max_length + step] = config_.eos_token_id;
                continue;
            }

            // Get logits for this batch item (on CPU)
            auto batch_start = b * last_logits.shape()[last_logits.ndim() - 1];
            Tensor batch_logits(std::vector<int64_t>{last_logits.shape()[last_logits.ndim() - 1]},
                               DType::Float32, Device::cpu());

            const float* src_data = last_logits_cpu.data<float>() + batch_start;
            float* dst_data = batch_logits.data<float>();
            std::copy(src_data, src_data + batch_logits.numel(), dst_data);

            // Convert to probabilities
            Variable batch_var(batch_logits, false);
            auto probs = logits_to_probs(batch_var, temperature);

            // Apply top-p filter (works on CPU internally)
            auto [filtered_probs, indices] = top_p_filter(probs.tensor(), top_p);

            // Sample from filtered distribution (works on CPU internally)
            int64_t sampled_idx = sample_from_probs(filtered_probs);
            Tensor indices_cpu = indices.to(Device::cpu());
            int64_t token_id = indices_cpu.data<int64_t>()[sampled_idx];

            output_data[b * config_.max_length + step] = token_id;

            // EOS early-stop: mark this row finished once it emits eos_token_id.
            if (config_.eos_token_id >= 0 && token_id == config_.eos_token_id) {
                finished[b] = true;
            }
        }

        // Stop generating entirely once every row has emitted EOS.
        if (config_.eos_token_id >= 0 &&
            std::all_of(finished.begin(), finished.end(), [](bool f) { return f; })) {
            break;
        }
    }

    // Return output on original device
    return output_cpu.to(original_device);
}

auto TextGenerator::beam_search(const Tensor& input_ids, int64_t num_beams) -> Tensor {
    model_.eval();

    // Audit G16: batched beam search.
    //
    // Previously hard-coded to batch_size=1 (the function threw otherwise).
    // Real beam search across a batch of inputs maintains B × num_beams beams
    // total, with each batch's top-num_beams kept independently. We pack all
    // beams into a single (B × num_beams, T) forward call per step rather
    // than looping per-beam — the model sees a virtual batch of size
    // B × num_beams, so latency is amortized across beams and batch items.
    auto original_device = input_ids.device();
    auto batch_size = input_ids.shape()[0];
    auto current_len = input_ids.shape()[1];
    const int64_t flat_n = batch_size * num_beams;

    // Beams stored as variable-length token vectors plus a parallel score
    // vector indexed as `bat * num_beams + b`.
    std::vector<std::vector<int64_t>> beam_tokens(flat_n);
    std::vector<float> beam_scores(flat_n, 0.0f);

    Tensor input_cpu = input_ids.to(Device::cpu());
    const int64_t* input_data = input_cpu.data<int64_t>();
    for (int64_t bat = 0; bat < batch_size; ++bat) {
        for (int64_t b = 0; b < num_beams; ++b) {
            auto& v = beam_tokens[bat * num_beams + b];
            v.resize(current_len);
            for (int64_t i = 0; i < current_len; ++i) {
                v[i] = input_data[bat * current_len + i];
            }
        }
    }

    struct Cand {
        int64_t src_beam;  // index within the batch's num_beams
        int64_t token;
        float score;
    };

    for (int64_t step = current_len; step < config_.max_length; ++step) {
        const int64_t cur_T = step;  // all beams currently this long

        // Pack all B × num_beams beams into one (flat_n, cur_T) tensor.
        Tensor stacked_cpu({flat_n, cur_T}, DType::Int64, Device::cpu());
        int64_t* sp = stacked_cpu.data<int64_t>();
        for (int64_t i = 0; i < flat_n; ++i) {
            std::copy_n(beam_tokens[i].data(), cur_T, sp + i * cur_T);
        }
        Tensor stacked = (original_device == Device::cpu())
            ? stacked_cpu : stacked_cpu.to(original_device);
        Variable stacked_var(stacked, false);

        // Single batched forward — the model treats flat_n as the batch dim.
        auto logits = model_.forward(stacked_var, Variable{}, Tensor{});
        auto last = logits.tensor().slice(1, cur_T - 1, cur_T, 1).squeeze(1);  // (flat_n, V)
        Variable last_var(last, false);
        auto log_probs_t = nn::log_softmax(last_var, -1).tensor();
        Tensor lp_cpu = log_probs_t.to(Device::cpu());
        if (lp_cpu.dtype() != DType::Float32) lp_cpu = lp_cpu.to(DType::Float32);
        const float* lp = lp_cpu.data<float>();
        const int64_t vocab_size = lp_cpu.shape()[1];

        // Per-batch top-k selection. New beams replace old in a separate
        // buffer so within-step swaps don't corrupt the source token vectors.
        std::vector<std::vector<int64_t>> new_beam_tokens(flat_n);
        std::vector<float> new_beam_scores(flat_n);

        for (int64_t bat = 0; bat < batch_size; ++bat) {
            std::vector<Cand> candidates;
            candidates.reserve(static_cast<size_t>(num_beams * vocab_size));
            for (int64_t b = 0; b < num_beams; ++b) {
                const int64_t flat_idx = bat * num_beams + b;
                const float prev = beam_scores[flat_idx];
                const float* row = lp + flat_idx * vocab_size;
                for (int64_t v = 0; v < vocab_size; ++v) {
                    candidates.push_back({b, v, prev + row[v]});
                }
            }
            std::partial_sort(candidates.begin(),
                              candidates.begin() + num_beams,
                              candidates.end(),
                              [](const Cand& a, const Cand& b) { return a.score > b.score; });
            for (int64_t b = 0; b < num_beams; ++b) {
                const int64_t dst = bat * num_beams + b;
                const int64_t src = bat * num_beams + candidates[b].src_beam;
                new_beam_tokens[dst] = beam_tokens[src];
                new_beam_tokens[dst].push_back(candidates[b].token);
                new_beam_scores[dst] = candidates[b].score;
            }
        }
        beam_tokens = std::move(new_beam_tokens);
        beam_scores = std::move(new_beam_scores);
    }

    // Return the best (highest-score) beam per batch.
    Tensor output_cpu({batch_size, config_.max_length}, DType::Int64, Device::cpu());
    int64_t* od = output_cpu.data<int64_t>();
    for (int64_t bat = 0; bat < batch_size; ++bat) {
        const auto& best = beam_tokens[bat * num_beams + 0];
        const int64_t n = std::min<int64_t>(config_.max_length, static_cast<int64_t>(best.size()));
        for (int64_t i = 0; i < n; ++i) {
            od[bat * config_.max_length + i] = best[i];
        }
        // Tail (if best is shorter than max_length) stays zero-initialized.
        for (int64_t i = n; i < config_.max_length; ++i) {
            od[bat * config_.max_length + i] = 0;
        }
    }
    return (original_device == Device::cpu()) ? output_cpu : output_cpu.to(original_device);
}

} // namespace models
} // namespace tenzor
