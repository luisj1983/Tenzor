/**
 * @file hrm.cpp
 * @brief Hierarchical Reasoning Model implementation
 *
 * Complete implementation matching the HRM paper (Wang et al., 2025)
 * including Q-learning ACT, stablemax, and proper initialization.
 */

#include "tenzor/nn/layers/hrm.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>

namespace tenzor {
namespace nn {

// ============================================================================
// Initialization Utilities Implementation
// ============================================================================

void lecun_normal_init(Tensor& tensor, int64_t fan_in) {
    // LeCun normal: N(0, 1/fan_in)
    double std = 1.0 / std::sqrt(static_cast<double>(fan_in));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, static_cast<float>(std));

    float* data = tensor.data<float>();
    int64_t numel = tensor.numel();

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(gen);
    }
}

void lecun_uniform_init(Tensor& tensor, int64_t fan_in) {
    // LeCun uniform: U(-limit, limit) where limit = sqrt(3/fan_in)
    double limit = std::sqrt(3.0 / static_cast<double>(fan_in));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(static_cast<float>(-limit),
                                                static_cast<float>(limit));

    float* data = tensor.data<float>();
    int64_t numel = tensor.numel();

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(gen);
    }
}

void truncated_normal_init(Tensor& tensor, double mean, double std,
                           double a, double b) {
    // Truncated normal: sample from N(mean, std) but reject values outside [mean + a*std, mean + b*std]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(static_cast<float>(mean),
                                          static_cast<float>(std));

    float lower = static_cast<float>(mean + a * std);
    float upper = static_cast<float>(mean + b * std);

    float* data = tensor.data<float>();
    int64_t numel = tensor.numel();

    for (int64_t i = 0; i < numel; ++i) {
        float sample;
        int max_attempts = 100;
        int attempts = 0;

        do {
            sample = dist(gen);
            attempts++;
        } while ((sample < lower || sample > upper) && attempts < max_attempts);

        // If we couldn't get a valid sample, clamp it
        if (sample < lower) sample = lower;
        if (sample > upper) sample = upper;

        data[i] = sample;
    }
}

// ============================================================================
// Stablemax Implementation
// ============================================================================

Variable stablemax(const Variable& input, int64_t dim, double eps) {
    // Stablemax: exp(x - max(x)) / (sum(exp(x - max(x))) + eps)
    // The eps term in denominator prevents division by very small numbers.
    //
    // Implementation runs entirely on Variable-level ops so backward() flows
    // through to `input`. The previous implementation extracted
    // `input.tensor()` up front and returned a fresh Variable with no
    // grad_fn, silently severing the graph.

    if (dim < 0) {
        dim = static_cast<int64_t>(input.shape().size()) + dim;
    }

    auto x_max = ::tenzor::max(input, dim, /*keepdim=*/true);
    auto x_shifted = input - x_max;
    auto exp_x = ::tenzor::exp(x_shifted);
    auto sum_exp = ::tenzor::sum(exp_x, dim, /*keepdim=*/true) + static_cast<float>(eps);
    return exp_x / sum_exp;
}

Variable stablemax_cross_entropy(const Variable& input, const Variable& target,
                                  double eps) {
    // Compute stablemax cross-entropy loss
    // input: (batch, num_classes) logits
    // target: (batch,) class indices

    auto batch_size = input.shape()[0];
    auto num_classes = input.shape()[1];

    // Compute stablemax probabilities
    auto probs = stablemax(input, -1, eps);

    // Gather the probabilities for correct classes
    // For each sample, get prob[target[i]]
    auto target_tensor = target.tensor();
    auto probs_tensor = probs.tensor();

    // Compute negative log likelihood
    std::vector<float> losses(batch_size);

    for (int64_t i = 0; i < batch_size; ++i) {
        int64_t cls = static_cast<int64_t>(target_tensor.data<float>()[i]);
        if (target_tensor.dtype() == DType::Int64) {
            cls = target_tensor.data<int64_t>()[i];
        } else if (target_tensor.dtype() == DType::Int32) {
            cls = static_cast<int64_t>(target_tensor.data<int32_t>()[i]);
        }

        // Get probability for this class
        float prob = probs_tensor.data<float>()[i * num_classes + cls];

        // Negative log likelihood with numerical stability
        losses[i] = -std::log(std::max(prob, static_cast<float>(eps)));
    }

    // Create loss tensor and compute mean
    Tensor loss_tensor({batch_size}, DType::Float32, input.tensor().device());
    std::memcpy(loss_tensor.data<float>(), losses.data(), batch_size * sizeof(float));

    auto mean_loss = tenzor::mean(loss_tensor);

    return Variable(mean_loss, input.requires_grad());
}

// RMSNorm is now implemented in normalization.cpp

// ============================================================================
// GatedLinearUnit Implementation
// ============================================================================

GatedLinearUnit::GatedLinearUnit(int64_t in_features, int64_t hidden_features,
                                  GateType gate_type, bool bias)
    : gate_type_(gate_type) {

    // Gate and up projections: in_features -> hidden_features
    gate_proj_ = std::make_shared<Linear>(in_features, hidden_features, bias);
    up_proj_ = std::make_shared<Linear>(in_features, hidden_features, bias);

    // Down projection: hidden_features -> in_features
    down_proj_ = std::make_shared<Linear>(hidden_features, in_features, bias);

    register_module("gate_proj", gate_proj_);
    register_module("up_proj", up_proj_);
    register_module("down_proj", down_proj_);
}

GatedLinearUnit::GatedLinearUnit(int64_t in_features, int64_t hidden_features,
                                  bool use_silu, bool bias)
    : GatedLinearUnit(in_features, hidden_features,
                       use_silu ? GateType::SiLU : GateType::Sigmoid, bias) {}

auto GatedLinearUnit::forward_impl(const Variable& input) -> Variable {
    // GLU: gate(x) * up(x), then down projection
    auto gate = gate_proj_->forward(input);
    auto up = up_proj_->forward(input);

    // Apply activation to gate
    Variable activated_gate;
    switch (gate_type_) {
        case GateType::SiLU: {
            // SiLU: x * nn::sigmoid(x)
            auto sigmoid_gate = nn::sigmoid(gate);
            activated_gate = Variable(gate.tensor() * sigmoid_gate.tensor(), gate.requires_grad());
            break;
        }
        case GateType::GELU:
            activated_gate = nn::gelu(gate);
            break;
        case GateType::ReLU:
            activated_gate = nn::relu(gate);
            break;
        case GateType::Sigmoid:
        default:
            activated_gate = nn::sigmoid(gate);
            break;
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
    auto probs = nn::sigmoid(squeezed);  // (batch, seq_len)

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
// QLearningACT Implementation
// ============================================================================

QLearningACT::QLearningACT(int64_t d_model, int64_t max_segments,
                           double epsilon, double gamma, double lr)
    : epsilon_(epsilon), gamma_(gamma), lr_(lr), max_segments_(max_segments) {

    // Q-head projects pooled state to 2 Q-values: [Q_halt, Q_continue]
    q_head_ = std::make_shared<Linear>(d_model, 2, true);
    register_module("q_head", q_head_);

    // Initialize statistics
    stats_ = {0.0, 0.0, epsilon, 0, 0};

    // Initialize RNG
    std::random_device rd;
    rng_ = std::mt19937(rd());
}

auto QLearningACT::pool_state(const Variable& state) -> Variable {
    // Mean pooling across sequence dimension
    // state: (batch, seq_len, d_model) -> (batch, d_model)
    auto pooled = tenzor::mean(state.tensor(), {1}, false);
    return Variable(pooled, state.requires_grad());
}

auto QLearningACT::compute_q_values(const Variable& state)
    -> std::pair<Variable, Variable> {
    // Pool state and compute Q-values
    auto pooled = pool_state(state);

    // Project to Q-values: (batch, 2)
    auto q_values = q_head_->forward(pooled);

    // Split into Q_halt and Q_continue
    auto q_tensor = q_values.tensor();

    // Extract Q_halt (index 0) and Q_continue (index 1)
    auto q_halt = tenzor::select(q_tensor, -1, 0);      // (batch,)
    auto q_continue = tenzor::select(q_tensor, -1, 1);  // (batch,)

    // Apply sigmoid to bound Q-values to [0, 1] (since reward is 0 or 1)
    // Use nn::sigmoid for Variable inputs
    auto q_halt_var = Variable(q_halt, q_values.requires_grad());
    auto q_continue_var = Variable(q_continue, q_values.requires_grad());
    auto q_halt_bounded = nn::sigmoid(q_halt_var);
    auto q_continue_bounded = nn::sigmoid(q_continue_var);

    return std::make_pair(q_halt_bounded, q_continue_bounded);
}

auto QLearningACT::select_action(const Variable& q_halt, const Variable& q_continue,
                                  bool training) -> bool {
    stats_.total_decisions++;

    // Get mean Q-values across batch for decision
    float mean_q_halt = tenzor::mean(q_halt.tensor()).item<float>();
    float mean_q_continue = tenzor::mean(q_continue.tensor()).item<float>();

    // Update statistics
    stats_.avg_q_halt = 0.9 * stats_.avg_q_halt + 0.1 * mean_q_halt;
    stats_.avg_q_continue = 0.9 * stats_.avg_q_continue + 0.1 * mean_q_continue;

    // Epsilon-greedy exploration (only during training)
    if (training) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng_) < epsilon_) {
            // Random action
            bool halt = dist(rng_) < 0.5;
            if (halt) stats_.halt_decisions++;
            return halt;
        }
    }

    // Greedy action: halt if Q_halt > Q_continue
    bool halt = mean_q_halt > mean_q_continue;
    if (halt) stats_.halt_decisions++;
    return halt;
}

auto QLearningACT::update_q_values(const Variable& state, bool action, double reward,
                                    const Variable& next_state, bool done) -> double {
    // Compute current Q-values
    auto [q_halt, q_continue] = compute_q_values(state);

    float current_q;
    if (action) {  // halt
        current_q = tenzor::mean(q_halt.tensor()).item<float>();
    } else {  // continue
        current_q = tenzor::mean(q_continue.tensor()).item<float>();
    }

    // Compute target Q-value
    float target_q;
    if (done || action) {
        // Terminal state or halt action: Q = reward
        target_q = static_cast<float>(reward);
    } else {
        // Non-terminal continue: Q = max(Q_halt', Q_continue')
        auto [next_q_halt, next_q_continue] = compute_q_values(next_state);
        float max_next_q = std::max(
            tenzor::mean(next_q_halt.tensor()).item<float>(),
            tenzor::mean(next_q_continue.tensor()).item<float>()
        );
        target_q = static_cast<float>(gamma_) * max_next_q;
    }

    // TD error
    double td_error = target_q - current_q;

    return td_error;
}

auto QLearningACT::compute_loss(const std::vector<Variable>& states,
                                 const std::vector<bool>& actions,
                                 const std::vector<double>& rewards,
                                 const std::vector<Variable>& next_states,
                                 const std::vector<bool>& dones) -> Variable {
    if (states.empty()) {
        // Return zero loss
        return Variable(zeros({1}, DType::Float32, Device::cpu()), true);
    }

    // Compute Q-learning loss (MSE between predicted and target Q-values)
    float total_loss = 0.0f;

    for (size_t i = 0; i < states.size(); ++i) {
        auto [q_halt, q_continue] = compute_q_values(states[i]);

        // Get predicted Q for taken action
        float predicted_q;
        if (actions[i]) {  // halt
            predicted_q = tenzor::mean(q_halt.tensor()).item<float>();
        } else {  // continue
            predicted_q = tenzor::mean(q_continue.tensor()).item<float>();
        }

        // Compute target
        float target_q;
        if (dones[i] || actions[i]) {
            target_q = static_cast<float>(rewards[i]);
        } else {
            auto [next_q_halt, next_q_continue] = compute_q_values(next_states[i]);
            float max_next_q = std::max(
                tenzor::mean(next_q_halt.tensor()).item<float>(),
                tenzor::mean(next_q_continue.tensor()).item<float>()
            );
            target_q = static_cast<float>(gamma_) * max_next_q;
        }

        // Squared error
        float error = predicted_q - target_q;
        total_loss += error * error;
    }

    // Mean squared error
    total_loss /= static_cast<float>(states.size());

    Tensor loss_tensor({1}, DType::Float32, Device::cpu());
    loss_tensor.data<float>()[0] = total_loss;

    return Variable(loss_tensor, true);
}

auto QLearningACT::decay_epsilon(double decay_rate, double min_epsilon) -> void {
    epsilon_ = std::max(epsilon_ * decay_rate, min_epsilon);
    stats_.exploration_rate = epsilon_;
}

auto QLearningACT::forward_impl(const Variable& input) -> Variable {
    auto [q_halt, q_continue] = compute_q_values(input);
    // Return Q_halt as the halting signal
    return q_halt;
}

// ============================================================================
// HRM Implementation
// ============================================================================

HRM::HRM(const HRMConfig& config) : config_(config) {
    // Validate config
    if (config.d_model % config.n_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by n_heads");
    }

    // Embedding layer (optional) - use proper Embedding instead of Linear
    if (config.vocab_size > 0) {
        embedding_ = std::make_shared<Embedding>(config.vocab_size, config.d_model);
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

    // Initialize fixed hidden state templates with truncated normal
    // These are kept fixed during training (as per HRM paper)
    h_init_state_ = zeros({1, 1, config.d_model}, DType::Float32, Device::cpu());
    l_init_state_ = zeros({1, 1, config.d_model}, DType::Float32, Device::cpu());

    if (config.use_truncated_normal) {
        truncated_normal_init(h_init_state_, 0.0, config.init_std,
                              config.truncated_a, config.truncated_b);
        truncated_normal_init(l_init_state_, 0.0, config.init_std,
                              config.truncated_a, config.truncated_b);
    }

    // Output layers
    output_norm_ = std::make_shared<RMSNorm>(config.d_model);

    if (config.num_classes > 0) {
        output_proj_ = std::make_shared<Linear>(config.d_model, config.num_classes, false);
        register_module("output_proj", output_proj_);
    }

    // Adaptive Computational Time (optional)
    if (config.use_act) {
        if (config.use_qlearning_act) {
            // Q-learning based ACT
            qlearning_act_ = std::make_shared<QLearningACT>(
                config.d_model, config.max_segments,
                config.act_epsilon, config.act_gamma, config.act_lr);
            register_module("qlearning_act", qlearning_act_);
        } else {
            // Simple halting probability ACT
            act_ = std::make_shared<AdaptiveComputationalTime>(
                config.d_model, config.n_high_cycles, config.act_threshold);
            register_module("act", act_);
        }
    }

    // Register modules
    register_module("h_module", h_module_);
    register_module("l_module", l_module_);
    register_module("h_init_proj", h_init_proj_);
    register_module("l_init_proj", l_init_proj_);
    register_module("output_norm", output_norm_);

    // Initialize statistics
    stats_ = {0, 0, 0, 0.0, 0.0, 0.0, 0.0};

    // Apply LeCun initialization if requested
    if (config.use_lecun_init) {
        apply_hrm_initialization();
    }
}

auto HRM::apply_hrm_initialization() -> void {
    // Apply LeCun initialization to all linear layers
    apply_lecun_init_to_module(h_module_.get());
    apply_lecun_init_to_module(l_module_.get());
    apply_lecun_init_to_module(h_init_proj_.get());
    apply_lecun_init_to_module(l_init_proj_.get());

    if (output_proj_) {
        apply_lecun_init_to_module(output_proj_.get());
    }
}

auto HRM::apply_lecun_init_to_module(Module* module) -> void {
    // Apply LeCun initialization to the module's parameters
    auto params = module->parameters();
    for (auto& param : params) {
        auto shape = param->shape();
        if (shape.size() >= 2) {
            // For weight matrices: shape is typically (out_features, in_features)
            int64_t fan_in = shape[shape.size() - 1];  // Last dimension is usually fan_in
            lecun_normal_init(param->tensor(), fan_in);
        }
    }
}

auto HRM::apply_output_activation(const Variable& logits) -> Variable {
    if (config_.num_classes <= 0) {
        // No output projection, return as-is
        return logits;
    }

    if (config_.use_stablemax) {
        return stablemax(logits, -1, config_.stablemax_eps);
    } else {
        // Standard softmax
        return nn::softmax(logits, -1);
    }
}

auto HRM::init_states(const Variable& x)
    -> std::pair<Variable, Variable> {
    // Initialize H and L states
    // As per HRM paper: initial hidden states are sampled from truncated normal
    // and kept FIXED during training

    auto shape = x.shape();
    int64_t batch = shape[0];
    int64_t seq_len = shape[1];
    auto device = x.tensor().device();

    // Broadcast fixed initial states to match input batch/seq dimensions
    // h_init_state_ and l_init_state_ are (1, 1, d_model)
    // We manually broadcast by repeating the tensor

    // Create expanded tensors by tiling
    auto h_device = h_init_state_.to(device);
    auto l_device = l_init_state_.to(device);

    // Broadcast via repeat: (1, 1, d_model) -> (batch, seq_len, d_model)
    auto h_expanded = tenzor::repeat(h_device, {batch, seq_len, 1});
    auto l_expanded = tenzor::repeat(l_device, {batch, seq_len, 1});

    // Project through initialization layers
    // This allows the model to learn how to transform the fixed initial states
    auto h_state = h_init_proj_->forward(Variable(h_expanded, false));
    auto l_state = l_init_proj_->forward(Variable(l_expanded, false));

    // Add input information to initial states
    // This helps the model condition on the input while maintaining fixed initialization
    h_state = Variable(h_state.tensor() + x.tensor() * 0.1f, x.requires_grad());
    l_state = Variable(l_state.tensor() + x.tensor() * 0.1f, x.requires_grad());

    return std::make_pair(h_state, l_state);
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
        x = embedding_->forward(input);
    }

    // Initialize states
    auto [h_state, l_state] = init_states(x);

    std::vector<Variable> aux_outputs;
    int64_t actual_cycles = 0;
    bool is_training = this->training_;

    // ACT state for simple halting
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
        if (qlearning_act_) {
            // Q-learning based ACT
            auto [q_halt, q_continue] = qlearning_act_->compute_q_values(h_state);
            if (qlearning_act_->select_action(q_halt, q_continue, is_training)) {
                break;  // Halt
            }
        } else if (act_) {
            // Simple halting probability ACT
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
    stats_.actual_segments = 1;  // Single segment in forward_with_aux
    stats_.h_participation_ratio = compute_participation_ratio(h_state);
    stats_.l_participation_ratio = compute_participation_ratio(l_state);

    // Q-learning stats
    if (qlearning_act_) {
        auto q_stats = qlearning_act_->stats();
        stats_.avg_q_halt = q_stats.avg_q_halt;
        stats_.avg_q_continue = q_stats.avg_q_continue;
    }

    // Final output
    Variable final_output;
    if (aux_outputs.empty()) {
        final_output = run_h_cycle(h_state, l_state, mask);
    } else {
        final_output = aux_outputs.back();
    }

    return {final_output, aux_outputs};
}

auto HRM::forward_with_segments(const Variable& input,
                                 const Variable& targets,
                                 const Tensor& mask)
    -> std::tuple<Variable, std::vector<Variable>, Variable> {
    // Segment-based training as described in HRM paper
    // Multiple forward passes ("segments") with deep supervision

    Variable x = input;

    // Apply embedding if needed
    if (embedding_ && input.tensor().dtype() == DType::Int64) {
        x = embedding_->forward(input);
    }

    std::vector<Variable> segment_outputs;
    std::vector<Variable> q_states;
    std::vector<bool> q_actions;
    std::vector<double> q_rewards;
    std::vector<Variable> q_next_states;
    std::vector<bool> q_dones;

    int64_t total_cycles = 0;
    int64_t actual_segments = 0;
    bool is_training = this->training_;

    // Initialize states
    auto [h_state, l_state] = init_states(x);

    // Run multiple segments
    for (int64_t seg = 0; seg < config_.max_segments; ++seg) {
        actual_segments++;

        // Run N high-level cycles within this segment
        for (int64_t n = 0; n < config_.n_high_cycles; ++n) {
            auto output = run_h_cycle(h_state, l_state, mask);
            total_cycles++;

            // CRITICAL: Detach for approximate gradient
            h_state = h_state.detach();
            l_state = l_state.detach();
            h_state.set_requires_grad(true);
            l_state.set_requires_grad(true);
        }

        // Compute output for this segment
        auto normed = output_norm_->forward(h_state);
        Variable segment_output;
        if (output_proj_) {
            segment_output = output_proj_->forward(normed);
        } else {
            segment_output = normed;
        }
        segment_outputs.push_back(segment_output);

        // Q-learning ACT decision
        if (qlearning_act_) {
            q_states.push_back(h_state);

            auto [q_halt, q_continue] = qlearning_act_->compute_q_values(h_state);
            bool should_halt = qlearning_act_->select_action(q_halt, q_continue, is_training);
            q_actions.push_back(should_halt);

            if (should_halt) {
                // Compute reward: 1 if prediction is correct, 0 otherwise
                double reward = 0.0;
                if (static_cast<bool>(targets) && targets.tensor().numel() > 0) {
                    // Check if prediction matches target
                    auto pred = tenzor::argmax(segment_output.tensor(), -1);
                    auto target_tensor = targets.tensor();

                    // Simple accuracy check
                    // Convert boolean equality result to float for sum
                    auto eq_result = tenzor::eq(pred, target_tensor);
                    auto eq_float = eq_result.to(DType::Float32);
                    auto correct = tenzor::sum(eq_float).item<float>();
                    auto total = static_cast<float>(pred.numel());
                    reward = (correct / total > 0.5) ? 1.0 : 0.0;
                }
                q_rewards.push_back(reward);
                q_next_states.push_back(h_state);  // Dummy for terminal state
                q_dones.push_back(true);
                break;  // Halt
            } else {
                q_rewards.push_back(0.0);  // No reward for continue
                q_next_states.push_back(h_state);  // Will be updated next iteration
                q_dones.push_back(false);
            }
        }
    }

    // Update next_states for Q-learning (shift by one)
    for (size_t i = 0; i + 1 < q_next_states.size(); ++i) {
        q_next_states[i] = q_states[i + 1];
    }

    // Compute Q-learning loss
    Variable q_loss;
    if (qlearning_act_ && !q_states.empty()) {
        q_loss = qlearning_act_->compute_loss(q_states, q_actions, q_rewards,
                                               q_next_states, q_dones);
    } else {
        q_loss = Variable(zeros({1}, DType::Float32, x.tensor().device()), false);
    }

    // Update statistics
    stats_.actual_high_cycles = total_cycles;
    stats_.actual_low_steps = total_cycles * config_.t_low_steps;
    stats_.actual_segments = actual_segments;
    stats_.h_participation_ratio = compute_participation_ratio(h_state);
    stats_.l_participation_ratio = compute_participation_ratio(l_state);

    if (qlearning_act_) {
        auto q_stats = qlearning_act_->stats();
        stats_.avg_q_halt = q_stats.avg_q_halt;
        stats_.avg_q_continue = q_stats.avg_q_continue;
    }

    // Final output is last segment output
    Variable final_output = segment_outputs.empty() ?
        run_h_cycle(h_state, l_state, mask) : segment_outputs.back();

    return {final_output, segment_outputs, q_loss};
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
