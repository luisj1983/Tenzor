/**
 * @file hrm.cpp
 * @brief Hierarchical Reasoning Model implementation
 *
 * Complete implementation matching the HRM paper (Wang et al., 2025)
 * including Q-learning ACT, stablemax, and proper initialization.
 */

#include "tenzor/nn/layers/hrm.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/op_id.hpp"
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

// Phase 22-followup #36 fix: these initializers used to assume Float32
// (calling `tensor.data<float>()` directly), which threw "Type mismatch"
// when the model ran in Float16/BFloat16/Float64. Sample into a Float32
// scratch tensor on CPU, then copy/cast into the actual tensor — works
// regardless of the tensor's dtype or device.
namespace {
void copy_f32_into(const std::vector<float>& src, Tensor& dst) {
    Tensor scratch({static_cast<int64_t>(src.size())},
                   DType::Float32, Device::cpu());
    std::memcpy(scratch.data<float>(), src.data(), src.size() * sizeof(float));
    // Reshape to dst shape, cast to dst dtype, move to dst device, then
    // assign storage. tensor::reshape returns a view; we materialize a
    // contiguous tensor via to(.).
    auto shape = dst.shape();
    std::vector<int64_t> shape_v(shape.begin(), shape.end());
    Tensor casted = scratch.reshape(shape_v).to(dst.dtype()).to(dst.device());
    dst = casted;
}
}  // namespace

void lecun_normal_init(Tensor& tensor, int64_t fan_in) {
    // LeCun normal: N(0, 1/fan_in)
    double std = 1.0 / std::sqrt(static_cast<double>(fan_in));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, static_cast<float>(std));

    int64_t numel = tensor.numel();
    std::vector<float> samples(numel);
    for (int64_t i = 0; i < numel; ++i) samples[i] = dist(gen);
    copy_f32_into(samples, tensor);
}

void lecun_uniform_init(Tensor& tensor, int64_t fan_in) {
    // LeCun uniform: U(-limit, limit) where limit = sqrt(3/fan_in)
    double limit = std::sqrt(3.0 / static_cast<double>(fan_in));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(static_cast<float>(-limit),
                                                static_cast<float>(limit));

    int64_t numel = tensor.numel();
    std::vector<float> samples(numel);
    for (int64_t i = 0; i < numel; ++i) samples[i] = dist(gen);
    copy_f32_into(samples, tensor);
}

void truncated_normal_init(Tensor& tensor, double mean, double std,
                           double a, double b) {
    // Truncated normal: sample from N(mean, std) but reject values outside
    // [mean + a*std, mean + b*std].
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(static_cast<float>(mean),
                                          static_cast<float>(std));

    float lower = static_cast<float>(mean + a * std);
    float upper = static_cast<float>(mean + b * std);

    int64_t numel = tensor.numel();
    std::vector<float> samples(numel);
    for (int64_t i = 0; i < numel; ++i) {
        float sample;
        int max_attempts = 100;
        int attempts = 0;
        do {
            sample = dist(gen);
            attempts++;
        } while ((sample < lower || sample > upper) && attempts < max_attempts);
        if (sample < lower) sample = lower;
        if (sample > upper) sample = upper;
        samples[i] = sample;
    }
    copy_f32_into(samples, tensor);
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
    // Stablemax cross-entropy loss.
    //   input:  (batch, num_classes) logits, autograd-enabled
    //   target: (batch,) class indices (Int32/Int64), or (batch, num_classes)
    //           one-hot / soft labels (float).
    //
    // Previous implementation extracted `probs.tensor()` and built the loss
    // on the host in a vector<float>, wrapping the mean in a fresh Variable
    // with no grad_fn — which silently zeroed `input.grad()`. Matches the
    // pattern in feedback_raw_tensor_op_bug.md.
    //
    // Fixed: everything runs on Variable-level ops so backward() flows back
    // through `stablemax` to `input`. The one-hot construction reuses the
    // OneHot dispatch path from CrossEntropyLoss::forward, keeping the
    // computation on the input's device.

    const auto num_classes = input.shape()[1];

    // stablemax already preserves the graph; log() is autograd-aware.
    auto probs = stablemax(input, -1, eps);
    auto log_probs = ::tenzor::log(probs + static_cast<float>(eps));  // [B, C]

    // Build a [B, C] one-hot / soft-label Variable that doesn't require grad.
    const auto& target_tensor_raw = target.tensor();
    const bool is_float_target =
        (target_tensor_raw.dtype() == DType::Float32 ||
         target_tensor_raw.dtype() == DType::Float64 ||
         target_tensor_raw.dtype() == DType::Float16) &&
        target_tensor_raw.ndim() == 2;

    Variable one_hot_var;
    if (is_float_target) {
        Tensor t = target_tensor_raw;
        if (t.device() != input.tensor().device()) {
            t = t.to(input.tensor().device());
        }
        if (t.dtype() != input.tensor().dtype()) {
            t = t.to(input.tensor().dtype());
        }
        one_hot_var = Variable(t, false);
    } else {
        NewOpAttributes oh_attrs;
        oh_attrs.set(AttrKey::NumClasses, num_classes);
        Tensor target_dev = (target_tensor_raw.device() == input.tensor().device())
            ? target_tensor_raw
            : target_tensor_raw.to(input.tensor().device());
        std::vector<Tensor> oh_inputs = {target_dev};
        auto oh_results = dispatch(OpId::OneHot, oh_inputs, oh_attrs);
        Tensor one_hot = oh_results[0];
        if (one_hot.dtype() != input.tensor().dtype()) {
            one_hot = one_hot.to(input.tensor().dtype());
        }
        one_hot_var = Variable(one_hot, false);
    }

    // NLL = -mean( sum(log_probs * one_hot, dim=1) ); keeps graph intact.
    auto weighted = log_probs * one_hot_var;                // [B, C]
    auto per_sample = ::tenzor::sum(weighted, 1, false);    // [B]
    return ::tenzor::mean(::tenzor::neg(per_sample));       // scalar
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

auto GatedLinearUnit::forward_impl(const Variable& input) -> Variable {
    // GLU: gate(x) * up(x), then down projection
    auto gate = gate_proj_->forward(input);
    auto up = up_proj_->forward(input);

    // Apply activation to gate. CRITICAL: must use Variable-level ops so the
    // autograd graph survives. Wrapping `gate.tensor() * other.tensor()` in a
    // new Variable (the previous implementation) produced a no-grad_fn node
    // that silently dropped input/parameter gradients on backward.
    Variable activated_gate;
    switch (gate_type_) {
        case GateType::SiLU: {
            // SiLU: x * nn::sigmoid(x) — keep both operands as Variables so
            // the multiplication appears on the autograd graph.
            auto sigmoid_gate = nn::sigmoid(gate);
            activated_gate = gate * sigmoid_gate;
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

    // Element-wise multiply through Variable operator* — preserves graph.
    auto gated = activated_gate * up;

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

    // Build the entire rotation on the autograd Variable path so grad_fn is
    // preserved end-to-end. cos/sin are constant (no-grad) Variables.
    const int64_t half = head_dim / 2;

    // Reshape for rotation: treat last dim as (dim/2, 2) — autograd-aware.
    auto x_reshaped = tenzor::reshape(x, {batch, seq_len, n_heads, half, 2});

    // Get x1 (index 0) and x2 (index 1) along the trailing pair axis (dim 4),
    // keeping grad_fn via autograd slice + squeeze.
    auto x1 = tenzor::squeeze(tenzor::slice(x_reshaped, 4, 0, 1), 4);  // (b,s,h,half)
    auto x2 = tenzor::squeeze(tenzor::slice(x_reshaped, 4, 1, 2), 4);  // (b,s,h,half)

    // Broadcast cos/sin to match x shape:
    // cos_slice: (seq_len, half) -> (1, seq_len, 1, half)
    Variable cos_v(cos_slice, /*requires_grad=*/false);
    Variable sin_v(sin_slice, /*requires_grad=*/false);
    auto cos_broadcast = tenzor::unsqueeze(tenzor::unsqueeze(cos_v, 0), 2);
    auto sin_broadcast = tenzor::unsqueeze(tenzor::unsqueeze(sin_v, 0), 2);

    // Apply rotation (Variable arithmetic, autograd-aware):
    // x_rotated[..., 0] = x1 * cos - x2 * sin
    // x_rotated[..., 1] = x1 * sin + x2 * cos
    auto rot1 = x1 * cos_broadcast - x2 * sin_broadcast;  // (b,s,h,half)
    auto rot2 = x1 * sin_broadcast + x2 * cos_broadcast;  // (b,s,h,half)

    // Stack back together along a new trailing pair axis, then flatten to
    // head_dim. cat of two (b,s,h,half,1) tensors along dim 4 -> (b,s,h,half,2).
    auto rotated = tenzor::cat({tenzor::unsqueeze(rot1, 4), tenzor::unsqueeze(rot2, 4)}, 4);
    return tenzor::reshape(rotated, {batch, seq_len, n_heads, head_dim});
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
    ffn_ = std::make_shared<GatedLinearUnit>(d_model, d_feedforward, GateType::SiLU, false);

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

    // Phase 22-followup #32 fix: residual additions previously used raw
    // tensor + and rewrapped in Variable(.,.) which dropped the autograd
    // graph between input and output. Use Variable operator+ throughout
    // (and never extract .tensor() until the final hand-off).
    if (use_post_norm_) {
        // Post-norm: attention -> residual -> norm
        auto [attn_out, _] = self_attn_->forward(x, x, x, Tensor{}, mask, false);
        out = norm1_->forward(x + dropout_->forward(attn_out));

        if (has_context) {
            auto [cross_out, __] = cross_attn_->forward(out, context, context,
                                                         Tensor{}, Tensor{}, false);
            out = norm3_->forward(out + dropout_->forward(cross_out));
        }

        auto ffn_out = ffn_->forward(out);
        out = norm2_->forward(out + dropout_->forward(ffn_out));

    } else {
        // Pre-norm: norm -> attention -> residual
        auto normed = norm1_->forward(x);
        auto [attn_out, _] = self_attn_->forward(normed, normed, normed,
                                                  Tensor{}, mask, false);
        out = x + dropout_->forward(attn_out);

        if (has_context) {
            auto normed_cross = norm3_->forward(out);
            auto [cross_out, __] = cross_attn_->forward(normed_cross, context, context,
                                                         Tensor{}, Tensor{}, false);
            out = out + dropout_->forward(cross_out);
        }

        auto normed_ffn = norm2_->forward(out);
        auto ffn_out = ffn_->forward(normed_ffn);
        out = out + dropout_->forward(ffn_out);
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
    // Squeeze last dimension — use the Variable-aware tenzor::squeeze so
    // the halt-probability gradient flows back to halt_proj_. The previous
    //   Variable(tenzor::squeeze(logits.tensor(), -1), logits.requires_grad())
    // discarded logits' grad_fn, silently zeroing gradients to halt_proj_.
    auto squeezed = ::tenzor::squeeze(logits, -1);
    auto probs = nn::sigmoid(squeezed);  // (batch, seq_len)

    return probs;
}

auto AdaptiveComputationalTime::should_halt(const Variable& cumulative_prob) -> bool {
    // Check if all positions have cumulative probability >= threshold.
    // Cast to Float32 before extracting via item<float> — `cumulative_prob`
    // can be Float64 when the model runs in double precision and item<>
    // throws on dtype mismatch.
    auto min_prob = tenzor::min(cumulative_prob.tensor()).to(DType::Float32);
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
    // Variable-level mean keeps the autograd chain so gradient from the
    // Q-loss reaches the H-module state as well as q_head_'s own params.
    return ::tenzor::mean(state, /*dim=*/1, /*keepdim=*/false);
}

auto QLearningACT::compute_q_values(const Variable& state)
    -> std::pair<Variable, Variable> {
    // Pool state and compute Q-values
    auto pooled = pool_state(state);

    // Project to Q-values: (batch, 2)
    auto q_values = q_head_->forward(pooled);

    // Split into Q_halt and Q_continue along the class dim. Use the
    // autograd-aware `narrow` so backprop from the Q-loss flows back
    // through q_head_'s weights — `tenzor::select` on the raw tensor
    // here previously severed that chain. Shape is (batch, 1) instead
    // of (batch,); downstream consumers all reduce via mean so the
    // extra trailing dim is harmless.
    auto q_halt = ::tenzor::narrow(q_values, /*dim=*/-1, /*start=*/0, /*length=*/1);
    auto q_continue = ::tenzor::narrow(q_values, /*dim=*/-1, /*start=*/1, /*length=*/1);

    // Apply sigmoid to bound Q-values to [0, 1] (since reward is 0 or 1)
    auto q_halt_bounded = nn::sigmoid(q_halt);
    auto q_continue_bounded = nn::sigmoid(q_continue);

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
    auto device = states.empty() ? Device::cpu() : states[0].tensor().device();
    auto dtype  = states.empty() ? DType::Float32 : states[0].tensor().dtype();

    if (states.empty()) {
        return Variable(zeros({1}, dtype, device), false);
    }

    // MSE between predicted Q for the taken action and the Bellman target.
    // Predicted Q stays inside the autograd graph (so gradient reaches
    // q_head_ and, via pool_state, the H-module). Target Q is built as a
    // no-grad scalar — standard target-network treatment in DQN-style
    // Q-learning. Tensors are constructed on the input's device so the
    // returned loss can be added to a GPU-side supervision loss without a
    // device-mismatch crash.
    Variable total_loss;
    bool first = true;

    for (size_t i = 0; i < states.size(); ++i) {
        auto [q_halt, q_continue] = compute_q_values(states[i]);

        Variable predicted_q = actions[i] ? ::tenzor::mean(q_halt)
                                          : ::tenzor::mean(q_continue);

        float target_val;
        if (dones[i] || actions[i]) {
            target_val = static_cast<float>(rewards[i]);
        } else {
            auto [next_q_halt, next_q_continue] = compute_q_values(next_states[i]);
            float max_next_q = std::max(
                ::tenzor::mean(next_q_halt.tensor()).item<float>(),
                ::tenzor::mean(next_q_continue.tensor()).item<float>()
            );
            target_val = static_cast<float>(gamma_) * max_next_q;
        }
        Variable target_q(::tenzor::full({1}, target_val, dtype, device),
                          /*requires_grad=*/false);

        Variable err = predicted_q - target_q;
        Variable sq  = err * err;

        if (first) { total_loss = sq; first = false; }
        else       { total_loss = total_loss + sq; }
    }

    Variable inv_n(::tenzor::full({1},
                                   1.0f / static_cast<float>(states.size()),
                                   dtype, device),
                   /*requires_grad=*/false);
    return total_loss * inv_n;
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
    // These are kept fixed during training (as per HRM paper). Register as
    // buffers so they're surfaced through named_buffers() / state-dict and
    // get moved to the right device by Module::to(...) — without this,
    // backend-parity tests that copy only `parameters()` see different random
    // h/l_init values on each fresh HRM(cfg) instance, producing huge
    // CPU-vs-GPU divergence in HRM forward output.
    h_init_state_ = zeros({1, 1, config.d_model}, DType::Float32, Device::cpu());
    l_init_state_ = zeros({1, 1, config.d_model}, DType::Float32, Device::cpu());

    if (config.use_truncated_normal) {
        truncated_normal_init(h_init_state_, 0.0, config.init_std,
                              config.truncated_a, config.truncated_b);
        truncated_normal_init(l_init_state_, 0.0, config.init_std,
                              config.truncated_a, config.truncated_b);
    }

    register_buffer("h_init_state", Variable(h_init_state_, false));
    register_buffer("l_init_state", Variable(l_init_state_, false));

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

    // Broadcast fixed initial states to match input batch/seq dimensions.
    // h_init_state_ / l_init_state_ are registered as buffers so they get
    // moved by Module::to(device); fetch the buffer's current Tensor here
    // instead of using the (possibly stale-device) cpp members. The buffer's
    // tensor is also what backend-parity tests copy from the CPU reference
    // model, so this is the single source of truth for the random init.
    auto h_buf = this->get_buffer("h_init_state");
    auto l_buf = this->get_buffer("l_init_state");
    auto h_device = (h_buf && h_buf->tensor().device() == device)
        ? h_buf->tensor()
        : (h_buf ? h_buf->tensor().to(device) : h_init_state_.to(device));
    auto l_device = (l_buf && l_buf->tensor().device() == device)
        ? l_buf->tensor()
        : (l_buf ? l_buf->tensor().to(device) : l_init_state_.to(device));

    // Broadcast via repeat: (1, 1, d_model) -> (batch, seq_len, d_model)
    auto h_expanded = tenzor::repeat(h_device, {batch, seq_len, 1});
    auto l_expanded = tenzor::repeat(l_device, {batch, seq_len, 1});

    // Project through initialization layers
    // This allows the model to learn how to transform the fixed initial states
    auto h_state = h_init_proj_->forward(Variable(h_expanded, false));
    auto l_state = l_init_proj_->forward(Variable(l_expanded, false));

    // Add input information to initial states. Phase 22-followup #32 fix:
    // raw `x.tensor() * 0.1f` previously dropped the autograd graph from x
    // into the initial states. Use Variable-level scalar multiplication
    // (broadcast against a no-grad scale tensor) so backward flows.
    auto scale = Variable(
        ::tenzor::full({1}, 0.1f, x.tensor().dtype(), x.tensor().device()),
        /*requires_grad=*/false);
    h_state = h_state + (x * scale);
    l_state = l_state + (x * scale);

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
            // Simple halting probability ACT.
            // audit-10 OO.3: previously `cumulative_halt_prob = Variable(
            //     cumulative_halt_prob.tensor() + halt_prob.tensor(), false)`
            // discarded `halt_prob`'s grad_fn, severing autograd through the
            // ACT halt-probability head.  Use Variable + Variable so the
            // accumulator preserves the graph.  `should_halt` only inspects
            // values (min + item<float>) so it doesn't need a graph; mirrors
            // the MoE pattern from GG.4.
            auto halt_prob = act_->compute_halt_prob(h_state);
            cumulative_halt_prob = cumulative_halt_prob + halt_prob;

            // Value-only halt decision: read the running min on host so the
            // boolean break does not need to participate in autograd.
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

    // Final output.
    //
    // Always recompute the final output with a fresh run_h_cycle on the
    // (detached, grad-enabled) post-loop state, regardless of deep_supervision.
    // Previously the deep_supervision branch returned aux_outputs.back() — the
    // intermediate produced *inside* the last loop iteration, before that
    // iteration's detach — while the non-supervision branch ran a fresh
    // run_h_cycle here. Those are two different computations (the detach breaks
    // the recurrence graph between them), so the returned "final" output and the
    // gradient path differed depending on the flag. Unifying on the fresh
    // run_h_cycle makes the final output consistent; the per-cycle intermediates
    // remain available separately in aux_outputs for the deep-supervision loss.
    Variable final_output = run_h_cycle(h_state, l_state, mask);

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
    // S27 fix: previously this returned ~`mean(x^2)/sum(x^2) * numel`, which
    // is just `1` (constant in activations) — a meaningless diagnostic.
    //
    // True participation ratio for an activation matrix A (N samples, D
    // features) is PR(C) = (sum lambda_i)^2 / sum(lambda_i^2) where lambda_i
    // are the eigenvalues of the empirical covariance C = (A - mean)^T (A -
    // mean) / (N - 1). PR equals D when C is isotropic, 1 when one
    // eigenvalue dominates.
    //
    // We avoid a full eigendecomposition (heavy + lacks a backend symeig
    // contract guarantee for every device path) and use the diagonal-variance
    // proxy: PR_diag = (sum sigma_d^2)^2 / sum(sigma_d^4) where sigma_d^2 is
    // the per-feature variance. This equals PR exactly when off-diagonal
    // covariance is zero and is a tight lower bound otherwise — it still
    // varies with rank structure, which is what the diagnostic is for.

    auto x = state.tensor();
    auto shape = x.shape();
    if (shape.empty()) {
        return 0.0;
    }

    // Flatten leading dims so we have (N, D) with N = sample count.
    auto flat = x.view({-1, shape.back()});
    const int64_t N = flat.shape()[0];
    const int64_t D = flat.shape()[1];
    if (N <= 1 || D <= 0) {
        // PR is undefined for a single sample (variance has 0 degrees of
        // freedom). Return 0.0 to signal "not measurable" rather than NaN.
        return 0.0;
    }

    // Per-feature variance: shape [D]. Unbiased (N-1) denominator matches
    // the standard PR definition; both numerator and denominator below are
    // scaled by the same (N-1)^2 so the choice is irrelevant for PR, but
    // unbiased=true matches the textbook formula.
    auto variances = tenzor::var(flat, /*dim=*/0, /*keepdim=*/false, /*unbiased=*/true);

    // Promote to Float64 for the ratio so we don't lose precision when
    // feature variances are tiny (early-training scenarios produce values
    // near 0, and squaring compresses them further).
    auto v64 = variances.to(DType::Float64);
    auto v_sq = v64 * v64;

    auto sum_var = tenzor::sum(v64);
    auto sum_var_sq = tenzor::sum(v_sq);

    double s  = sum_var.item<double>();
    double s2 = sum_var_sq.item<double>();

    if (s2 < 1e-30) {
        // All-zero activations: no information; report PR = 0.
        return 0.0;
    }
    return (s * s) / s2;
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
        // Use Variable-level operators to keep the autograd chain. The
        // previous `Variable(loss.tensor() * scalar, rg)` re-wrap silently
        // severed the link from per-output loss back to the model — gradient
        // never reached the model parameters.
        Variable weight_var(::tenzor::full({1}, static_cast<float>(weight),
                                            loss.tensor().dtype(), loss.tensor().device()), false);
        auto weighted_loss = loss * weight_var;

        if (i == 0) {
            total_loss = weighted_loss;
        } else {
            total_loss = total_loss + weighted_loss;
        }
    }

    // Normalize by total weight
    Variable inv_total_weight(::tenzor::full({1}, static_cast<float>(1.0 / total_weight),
                                              total_loss.tensor().dtype(), total_loss.tensor().device()), false);
    return total_loss * inv_total_weight;
}

} // namespace nn
} // namespace tenzor
