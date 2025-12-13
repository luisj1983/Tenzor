/**
 * @file hrm.cpp
 * @brief Hierarchical Reasoning Model implementation
 */

#include "tenzor/nn/layers/hrm.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include <cmath>
#include <numeric>

namespace tenzor {
namespace nn {

// ============================================================================
// RMSNorm Implementation
// ============================================================================

RMSNorm::RMSNorm(int64_t normalized_shape, double eps)
    : normalized_shape_(normalized_shape), eps_(eps) {

    // Initialize weight (gamma) to ones
    Tensor weight_data = ones({normalized_shape}, DType::Float32, Device::cpu());
    weight_ = std::make_shared<Variable>(weight_data, true);
    register_parameter("weight", *weight_);
}

auto RMSNorm::forward_impl(const Variable& input) -> Variable {
    // RMSNorm: x / sqrt(mean(x^2) + eps) * gamma
    auto x = input;

    // Compute RMS: sqrt(mean(x^2))
    auto x_sq = x * x;

    // Mean over last dimension
    auto mean_sq = tenzor::mean(x_sq.tensor(), {-1}, true);
    auto rms = tenzor::sqrt(mean_sq + eps_);

    // Normalize
    auto x_norm = x.tensor() / rms;

    // Scale by learned weight
    auto result = x_norm * weight_->tensor();

    return Variable(result, input.requires_grad());
}

// ============================================================================
// GatedLinearUnit Implementation
// ============================================================================

GatedLinearUnit::GatedLinearUnit(int64_t in_features, int64_t hidden_features,
                                  bool use_silu, bool bias)
    : use_silu_(use_silu) {

    // Gate and up projections: in_features -> hidden_features
    gate_proj_ = std::make_shared<Linear>(in_features, hidden_features, bias);
    up_proj_ = std::make_shared<Linear>(in_features, hidden_features, bias);

    // Down projection: hidden_features -> in_features
    down_proj_ = std::make_shared<Linear>(hidden_features, in_features, bias);

    register_module("gate_proj", gate_proj_);
    register_module("up_proj", up_proj_);
    register_module("down_proj", down_proj_);
}

auto GatedLinearUnit::forward_impl(const Variable& input) -> Variable {
    // GLU: gate(x) * up(x), then down projection
    auto gate = gate_proj_->forward(input);
    auto up = up_proj_->forward(input);

    // Apply activation to gate
    Variable activated_gate;
    if (use_silu_) {
        // SiLU: x * sigmoid(x)
        auto sigmoid_gate = sigmoid(gate);
        activated_gate = Variable(gate.tensor() * sigmoid_gate.tensor(), gate.requires_grad());
    } else {
        // Standard sigmoid
        activated_gate = sigmoid(gate);
    }

    // Element-wise multiply
    auto gated = Variable(activated_gate.tensor() * up.tensor(), input.requires_grad());

    // Down projection
    return down_proj_->forward(gated);
}

// ============================================================================
// RotaryPositionEmbedding Implementation
// ============================================================================

RotaryPositionEmbedding::RotaryPositionEmbedding(int64_t dim, int64_t max_seq_len,
                                                  double base)
    : dim_(dim), max_seq_len_(max_seq_len), base_(base) {

    if (dim % 2 != 0) {
        throw std::invalid_argument("RoPE dimension must be even");
    }

    precompute_freqs();
}

void RotaryPositionEmbedding::precompute_freqs() {
    // Compute inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
    std::vector<float> inv_freq(dim_ / 2);
    for (int64_t i = 0; i < dim_ / 2; ++i) {
        inv_freq[i] = 1.0f / std::pow(base_, 2.0 * i / dim_);
    }

    // Create position indices
    std::vector<float> positions(max_seq_len_);
    for (int64_t i = 0; i < max_seq_len_; ++i) {
        positions[i] = static_cast<float>(i);
    }

    // Compute freqs: outer product of positions and inv_freq
    // Shape: (max_seq_len, dim/2)
    std::vector<float> cos_data(max_seq_len_ * dim_ / 2);
    std::vector<float> sin_data(max_seq_len_ * dim_ / 2);

    for (int64_t pos = 0; pos < max_seq_len_; ++pos) {
        for (int64_t i = 0; i < dim_ / 2; ++i) {
            float freq = positions[pos] * inv_freq[i];
            cos_data[pos * (dim_ / 2) + i] = std::cos(freq);
            sin_data[pos * (dim_ / 2) + i] = std::sin(freq);
        }
    }

    // Create tensors
    cos_cache_ = Tensor({max_seq_len_, dim_ / 2}, DType::Float32, Device::cpu());
    sin_cache_ = Tensor({max_seq_len_, dim_ / 2}, DType::Float32, Device::cpu());

    std::memcpy(cos_cache_.data_ptr(), cos_data.data(),
                cos_data.size() * sizeof(float));
    std::memcpy(sin_cache_.data_ptr(), sin_data.data(),
                sin_data.size() * sizeof(float));
}

auto RotaryPositionEmbedding::forward(const Variable& x, int64_t seq_offset) -> Variable {
    // x shape: (batch, seq_len, n_heads, head_dim)
    auto shape = x.shape();
    int64_t batch = shape[0];
    int64_t seq_len = shape[1];
    int64_t n_heads = shape[2];
    int64_t head_dim = shape[3];

    if (head_dim != dim_) {
        throw std::runtime_error("RoPE: head_dim mismatch");
    }

    // Move cache to same device as input
    auto device = x.tensor().device();
    auto cos_slice = cos_cache_.slice(0, seq_offset, seq_offset + seq_len).to(device);
    auto sin_slice = sin_cache_.slice(0, seq_offset, seq_offset + seq_len).to(device);

    // Split x into two halves for rotation
    // x1, x2 = x[..., :dim/2], x[..., dim/2:]
    auto x_data = x.tensor();

    // Reshape for rotation: treat last dim as (dim/2, 2)
    auto x_reshaped = x_data.view({batch, seq_len, n_heads, head_dim / 2, 2});

    // Get x1 (even indices) and x2 (odd indices)
    // Using tenzor::select function instead of member function
    auto x1 = tenzor::select(x_reshaped, -1, 0);  // [..., 0]
    auto x2 = tenzor::select(x_reshaped, -1, 1);  // [..., 1]

    // Broadcast cos/sin to match x shape
    // cos_slice: (seq_len, dim/2) -> (1, seq_len, 1, dim/2)
    auto cos_broadcast = tenzor::unsqueeze(tenzor::unsqueeze(cos_slice, 0), 2);
    auto sin_broadcast = tenzor::unsqueeze(tenzor::unsqueeze(sin_slice, 0), 2);

    // Apply rotation:
    // x_rotated[..., 0] = x1 * cos - x2 * sin
    // x_rotated[..., 1] = x1 * sin + x2 * cos
    auto rot1 = x1 * cos_broadcast - x2 * sin_broadcast;
    auto rot2 = x1 * sin_broadcast + x2 * cos_broadcast;

    // Stack back together
    std::vector<Tensor> to_stack = {rot1, rot2};
    auto rotated = tenzor::stack(to_stack, -1);
    auto result = rotated.view({batch, seq_len, n_heads, head_dim});

    return Variable(result, x.requires_grad());
}

auto RotaryPositionEmbedding::forward_impl(const Variable& input) -> Variable {
    return forward(input, 0);
}

// ============================================================================
// HRMBlock Implementation
// ============================================================================

HRMBlock::HRMBlock(int64_t d_model, int64_t n_heads, int64_t d_feedforward,
                   double dropout, bool use_post_norm, int64_t max_seq_len)
    : d_model_(d_model), n_heads_(n_heads), use_post_norm_(use_post_norm) {

    // Self-attention
    self_attn_ = std::make_shared<MultiheadAttention>(
        d_model, n_heads, dropout, true, false, false, 0, 0, true);

    // Cross-attention (for L reading from H)
    cross_attn_ = std::make_shared<MultiheadAttention>(
        d_model, n_heads, dropout, true, false, false, 0, 0, true);

    // Feed-forward with GLU
    ffn_ = std::make_shared<GatedLinearUnit>(d_model, d_feedforward, true, false);

    // RMS normalization layers
    norm1_ = std::make_shared<RMSNorm>(d_model);
    norm2_ = std::make_shared<RMSNorm>(d_model);
    norm3_ = std::make_shared<RMSNorm>(d_model);

    // Dropout
    dropout_ = std::make_shared<Dropout>(dropout);

    // Rotary position embeddings
    int64_t head_dim = d_model / n_heads;
    rope_ = std::make_shared<RotaryPositionEmbedding>(head_dim, max_seq_len);

    // Register submodules
    register_module("self_attn", self_attn_);
    register_module("cross_attn", cross_attn_);
    register_module("ffn", ffn_);
    register_module("norm1", norm1_);
    register_module("norm2", norm2_);
    register_module("norm3", norm3_);
    register_module("dropout", dropout_);
    register_module("rope", rope_);
}

auto HRMBlock::forward(const Variable& x, const Variable& context,
                       const Tensor& mask) -> Variable {
    Variable out;

    // Check if context is provided (non-empty Variable)
    bool has_context = static_cast<bool>(context);

    if (use_post_norm_) {
        // Post-norm: attention -> residual -> norm

        // Self-attention
        auto [attn_out, _] = self_attn_->forward(x, x, x, Tensor{}, mask, false);
        auto residual1 = x.tensor() + dropout_->forward(attn_out).tensor();
        out = norm1_->forward(Variable(residual1, x.requires_grad()));

        // Cross-attention (if context provided)
        if (has_context) {
            auto [cross_out, __] = cross_attn_->forward(out, context, context,
                                                         Tensor{}, Tensor{}, false);
            auto residual2 = out.tensor() + dropout_->forward(cross_out).tensor();
            out = norm3_->forward(Variable(residual2, x.requires_grad()));
        }

        // Feed-forward
        auto ffn_out = ffn_->forward(out);
        auto residual3 = out.tensor() + dropout_->forward(ffn_out).tensor();
        out = norm2_->forward(Variable(residual3, x.requires_grad()));

    } else {
        // Pre-norm: norm -> attention -> residual

        // Self-attention
        auto normed = norm1_->forward(x);
        auto [attn_out, _] = self_attn_->forward(normed, normed, normed,
                                                  Tensor{}, mask, false);
        out = Variable(x.tensor() + dropout_->forward(attn_out).tensor(), x.requires_grad());

        // Cross-attention (if context provided)
        if (has_context) {
            auto normed_cross = norm3_->forward(out);
            auto [cross_out, __] = cross_attn_->forward(normed_cross, context, context,
                                                         Tensor{}, Tensor{}, false);
            out = Variable(out.tensor() + dropout_->forward(cross_out).tensor(), x.requires_grad());
        }

        // Feed-forward
        auto normed_ffn = norm2_->forward(out);
        auto ffn_out = ffn_->forward(normed_ffn);
        out = Variable(out.tensor() + dropout_->forward(ffn_out).tensor(), x.requires_grad());
    }

    return out;
}

auto HRMBlock::forward_impl(const Variable& input) -> Variable {
    return forward(input, Variable{}, Tensor{});
}

// ============================================================================
// AdaptiveComputationalTime Implementation
// ============================================================================

AdaptiveComputationalTime::AdaptiveComputationalTime(int64_t d_model,
                                                      int64_t max_steps,
                                                      double threshold)
    : threshold_(threshold), max_steps_(max_steps) {

    // Project hidden state to halting probability
    halt_proj_ = std::make_shared<Linear>(d_model, 1, true);
    register_module("halt_proj", halt_proj_);
}

auto AdaptiveComputationalTime::compute_halt_prob(const Variable& state) -> Variable {
    // state: (batch, seq_len, d_model)
    // Output: (batch, seq_len) halting probability

    auto logits = halt_proj_->forward(state);  // (batch, seq_len, 1)
    // Squeeze last dimension
    auto squeezed = Variable(tenzor::squeeze(logits.tensor(), -1), logits.requires_grad());
    auto probs = sigmoid(squeezed);  // (batch, seq_len)

    return probs;
}

auto AdaptiveComputationalTime::should_halt(const Variable& cumulative_prob) -> bool {
    // Check if all positions have cumulative probability >= threshold
    auto min_prob = tenzor::min(cumulative_prob.tensor());
    return min_prob.item<float>() >= threshold_;
}

auto AdaptiveComputationalTime::forward_impl(const Variable& input) -> Variable {
    return compute_halt_prob(input);
}

// ============================================================================
// HRM Implementation
// ============================================================================

HRM::HRM(const HRMConfig& config) : config_(config) {
    // Validate config
    if (config.d_model % config.n_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by n_heads");
    }

    // Embedding layer (optional)
    if (config.vocab_size > 0) {
        embedding_ = std::make_shared<Linear>(config.vocab_size, config.d_model, false);
        register_module("embedding", embedding_);
    }

    // H-module (high-level, slow updates)
    h_module_ = std::make_shared<HRMBlock>(
        config.d_model, config.n_heads, config.d_feedforward,
        config.dropout, config.use_post_norm, config.max_seq_len);

    // L-module (low-level, fast updates)
    l_module_ = std::make_shared<HRMBlock>(
        config.d_model, config.n_heads, config.d_feedforward,
        config.dropout, config.use_post_norm, config.max_seq_len);

    // State initialization projections
    h_init_proj_ = std::make_shared<Linear>(config.d_model, config.d_model, true);
    l_init_proj_ = std::make_shared<Linear>(config.d_model, config.d_model, true);

    // Output layers
    output_norm_ = std::make_shared<RMSNorm>(config.d_model);

    if (config.num_classes > 0) {
        output_proj_ = std::make_shared<Linear>(config.d_model, config.num_classes, false);
        register_module("output_proj", output_proj_);
    }

    // Adaptive Computational Time (optional)
    if (config.use_act) {
        act_ = std::make_shared<AdaptiveComputationalTime>(
            config.d_model, config.n_high_cycles, config.act_threshold);
        register_module("act", act_);
    }

    // Register modules
    register_module("h_module", h_module_);
    register_module("l_module", l_module_);
    register_module("h_init_proj", h_init_proj_);
    register_module("l_init_proj", l_init_proj_);
    register_module("output_norm", output_norm_);

    // Initialize statistics
    stats_ = {0, 0, 0.0, 0.0};
}

auto HRM::init_states(const Variable& x)
    -> std::pair<Variable, Variable> {
    // Initialize H and L states from input
    // x: (batch, seq_len, d_model)

    auto h_state = h_init_proj_->forward(x);
    auto l_state = l_init_proj_->forward(x);

    return {h_state, l_state};
}

auto HRM::run_h_cycle(Variable& h_state, Variable& l_state,
                      const Tensor& mask) -> Variable {
    // Run T low-level iterations
    // L converges to local equilibrium with H as context
    for (int64_t t = 0; t < config_.t_low_steps; ++t) {
        l_state = l_module_->forward(l_state, h_state, mask);
    }

    // Update H based on converged L
    h_state = h_module_->forward(h_state, l_state, mask);

    // Return current output
    auto normed = output_norm_->forward(h_state);
    if (output_proj_) {
        return output_proj_->forward(normed);
    }
    return normed;
}

auto HRM::forward_impl(const Variable& input) -> Variable {
    auto [output, _] = forward_with_aux(input);
    return output;
}

auto HRM::forward_with_aux(const Variable& input, const Tensor& mask)
    -> std::pair<Variable, std::vector<Variable>> {

    Variable x = input;

    // Apply embedding if needed
    if (embedding_ && input.tensor().dtype() == DType::Int64) {
        // One-hot encode and embed
        // For simplicity, assuming input is already embedded if Float type
        x = embedding_->forward(input);
    }

    // Initialize states
    auto [h_state, l_state] = init_states(x);

    std::vector<Variable> aux_outputs;
    int64_t actual_cycles = 0;

    // ACT state
    Variable cumulative_halt_prob;
    if (act_) {
        auto shape = h_state.shape();
        cumulative_halt_prob = Variable(
            zeros({shape[0], shape[1]}, DType::Float32, h_state.tensor().device()),
            false);
    }

    // Run N high-level cycles
    for (int64_t n = 0; n < config_.n_high_cycles; ++n) {
        // Run one H cycle
        auto output = run_h_cycle(h_state, l_state, mask);
        actual_cycles++;

        // Collect intermediate output for deep supervision
        if (config_.deep_supervision) {
            aux_outputs.push_back(output);
        }

        // Check ACT halting condition
        if (act_) {
            auto halt_prob = act_->compute_halt_prob(h_state);
            cumulative_halt_prob = Variable(
                cumulative_halt_prob.tensor() + halt_prob.tensor(),
                false);

            if (act_->should_halt(cumulative_halt_prob)) {
                break;
            }
        }

        // CRITICAL: Detach hidden states for approximate gradient (O(1) memory)
        // This prevents gradient flow through the recurrence
        h_state = h_state.detach();
        l_state = l_state.detach();

        // Re-enable gradient tracking for next iteration
        h_state.set_requires_grad(true);
        l_state.set_requires_grad(true);
    }

    // Update statistics
    stats_.actual_high_cycles = actual_cycles;
    stats_.actual_low_steps = actual_cycles * config_.t_low_steps;
    stats_.h_participation_ratio = compute_participation_ratio(h_state);
    stats_.l_participation_ratio = compute_participation_ratio(l_state);

    // Final output
    Variable final_output;
    if (aux_outputs.empty()) {
        final_output = run_h_cycle(h_state, l_state, mask);
    } else {
        final_output = aux_outputs.back();
    }

    return {final_output, aux_outputs};
}

auto HRM::compute_participation_ratio(const Variable& state) -> double {
    // Participation ratio: measure of effective dimensionality
    // Simplified approximation using activation statistics

    auto x = state.tensor();
    auto shape = x.shape();

    // Flatten to (batch * seq_len, d_model)
    auto flat = x.view({-1, shape.back()});

    // Simple approximation: use mean squared over sum squared
    // This approximates how "spread out" the activations are
    auto x_sq = flat * flat;
    auto sum_sq = tenzor::sum(x_sq);
    auto mean_sq = tenzor::mean(x_sq);

    float ss = sum_sq.item<float>();
    float ms = mean_sq.item<float>();

    if (ss < 1e-10f) return 0.0;

    // Number of elements
    int64_t n = flat.numel();

    // Approximation of participation ratio
    return (ms * ms * n) / ss;
}

auto HRM::num_parameters() const -> int64_t {
    int64_t total = 0;
    for (const auto& param : const_cast<HRM*>(this)->parameters()) {
        total += param->tensor().numel();
    }
    return total;
}

// ============================================================================
// Deep Supervision Loss
// ============================================================================

auto hrm_deep_supervision_loss(
    const std::vector<Variable>& outputs,
    const Variable& targets,
    std::function<Variable(const Variable&, const Variable&)> loss_fn,
    double weight_decay) -> Variable {

    if (outputs.empty()) {
        throw std::invalid_argument("No outputs for deep supervision");
    }

    // Compute weighted sum of losses
    // Later outputs get higher weight
    int64_t n_outputs = outputs.size();
    double total_weight = 0.0;
    Variable total_loss;

    for (int64_t i = 0; i < n_outputs; ++i) {
        // Exponentially increasing weight
        double weight = std::pow(1.0 / weight_decay, i);
        total_weight += weight;

        auto loss = loss_fn(outputs[i], targets);
        auto weighted_loss = Variable(loss.tensor() * static_cast<float>(weight),
                                       loss.requires_grad());

        if (i == 0) {
            total_loss = weighted_loss;
        } else {
            total_loss = Variable(total_loss.tensor() + weighted_loss.tensor(),
                                   total_loss.requires_grad());
        }
    }

    // Normalize by total weight
    return Variable(total_loss.tensor() / static_cast<float>(total_weight),
                    total_loss.requires_grad());
}

} // namespace nn
} // namespace tenzor
