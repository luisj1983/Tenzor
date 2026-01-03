/**
 * @file hrm.hpp
 * @brief Hierarchical Reasoning Model (HRM) implementation
 *
 * Implements the brain-inspired recurrent architecture from:
 * "Hierarchical Reasoning Model" (Wang et al., 2025)
 * https://arxiv.org/abs/2506.21734
 *
 * Key features:
 * - Two interdependent recurrent modules (H and L) at different timescales
 * - H-module: Abstract, deliberate planning (slow updates)
 * - L-module: Rapid, detailed computations (fast updates)
 * - Approximate gradient method for O(1) memory training
 * - Deep supervision at multiple forward passes
 * - Q-Learning based Adaptive Computational Time (ACT)
 * - Stablemax output for numerical stability
 * - LeCun initialization with truncated normal for hidden states
 */

#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <random>
#include <functional>
#include "../module.hpp"
#include "linear.hpp"
#include "dropout.hpp"
#include "normalization.hpp"
#include "attention.hpp"
#include "embedding.hpp"

namespace tenzor {
namespace nn {

// ============================================================================
// Initialization Utilities
// ============================================================================

/**
 * @brief LeCun normal initialization
 *
 * Initializes weights from N(0, 1/fan_in) for better gradient flow.
 * Recommended for networks with SELU activation or self-normalizing networks.
 *
 * @param tensor Tensor to initialize
 * @param fan_in Number of input features
 */
void lecun_normal_init(Tensor& tensor, int64_t fan_in);

/**
 * @brief LeCun uniform initialization
 *
 * Initializes weights from U(-limit, limit) where limit = sqrt(3/fan_in).
 *
 * @param tensor Tensor to initialize
 * @param fan_in Number of input features
 */
void lecun_uniform_init(Tensor& tensor, int64_t fan_in);

/**
 * @brief Truncated normal initialization
 *
 * Samples from a normal distribution but rejects values outside [a, b].
 * Used in the HRM paper for hidden state initialization.
 *
 * @param tensor Tensor to initialize
 * @param mean Mean of the distribution
 * @param std Standard deviation
 * @param a Lower bound (default: -2*std)
 * @param b Upper bound (default: 2*std)
 */
void truncated_normal_init(Tensor& tensor, double mean = 0.0, double std = 1.0,
                           double a = -2.0, double b = 2.0);

// ============================================================================
// Stablemax Activation
// ============================================================================

/**
 * @brief Stablemax activation function
 *
 * A numerically stable variant of softmax that works better with small samples.
 * From the HRM paper, this provides better stability than standard softmax.
 *
 * stablemax(x)_i = exp(x_i - max(x)) / (sum_j exp(x_j - max(x)) + eps)
 *
 * The key difference from softmax is the eps term in the denominator which
 * prevents division by very small numbers.
 *
 * @param input Input tensor
 * @param dim Dimension along which to compute stablemax
 * @param eps Epsilon for numerical stability (default: 1e-12)
 * @return Normalized probabilities
 */
Variable stablemax(const Variable& input, int64_t dim = -1, double eps = 1e-12);

/**
 * @brief Stablemax cross-entropy loss
 *
 * Cross-entropy loss using stablemax instead of softmax for improved
 * numerical stability with small sample sizes.
 *
 * @param input Logits (batch, num_classes)
 * @param target Target class indices (batch,)
 * @param eps Epsilon for numerical stability
 * @return Scalar loss
 */
Variable stablemax_cross_entropy(const Variable& input, const Variable& target,
                                  double eps = 1e-12);

/**
 * @brief Configuration for Hierarchical Reasoning Model
 */
struct HRMConfig {
    int64_t d_model = 256;          ///< Model dimension
    int64_t n_heads = 8;            ///< Number of attention heads
    int64_t d_feedforward = 1024;   ///< Feed-forward hidden dimension
    int64_t n_high_cycles = 4;      ///< Number of high-level cycles (N)
    int64_t t_low_steps = 8;        ///< Low-level steps per high cycle (T)
    double dropout = 0.1;           ///< Dropout probability
    bool use_post_norm = true;      ///< Use post-norm (HRM default) vs pre-norm
    bool deep_supervision = true;   ///< Enable deep supervision at each H cycle
    int64_t max_seq_len = 512;      ///< Maximum sequence length

    // Vocab/output settings
    int64_t vocab_size = 0;         ///< Vocabulary size (0 = no embedding/output layers)
    int64_t num_classes = 0;        ///< Number of output classes (0 = use d_model output)

    // Output activation
    bool use_stablemax = true;      ///< Use stablemax instead of softmax for output
    double stablemax_eps = 1e-12;   ///< Epsilon for stablemax stability

    // Adaptive Computational Time (ACT) settings
    bool use_act = false;           ///< Enable Adaptive Computational Time
    bool use_qlearning_act = true;  ///< Use Q-learning ACT (true) vs simple halting (false)
    double act_threshold = 0.99;    ///< ACT halting threshold (for simple ACT)
    double act_epsilon = 0.1;       ///< Exploration rate for Q-learning ACT
    double act_gamma = 0.99;        ///< Discount factor for Q-learning
    double act_lr = 0.01;           ///< Learning rate for Q-value updates
    int64_t max_segments = 16;      ///< Maximum segments for inference-time scaling

    // Initialization settings
    bool use_lecun_init = true;     ///< Use LeCun initialization for weights
    bool use_truncated_normal = true; ///< Use truncated normal for hidden state init
    double init_std = 0.02;         ///< Standard deviation for initialization
    double truncated_a = -2.0;      ///< Lower bound for truncated normal (in std units)
    double truncated_b = 2.0;       ///< Upper bound for truncated normal (in std units)
};

// RMSNorm is now defined in normalization.hpp and included above

/**
 * @brief Gated Linear Unit (GLU) activation
 *
 * GLU(x) = x[:, :d] * sigmoid(x[:, d:])
 * or with SiLU: SwiGLU(x) = x[:, :d] * silu(x[:, d:])
 *
 * Used in modern transformers like LLaMA, PaLM.
 */
class GatedLinearUnit : public Module {
public:
    /**
     * @brief Construct GLU layer
     *
     * @param in_features Input dimension
     * @param hidden_features Hidden dimension (will output hidden_features)
     * @param use_silu Use SiLU instead of sigmoid (SwiGLU)
     * @param bias Use bias in linear layers
     */
    GatedLinearUnit(int64_t in_features, int64_t hidden_features,
                    bool use_silu = true, bool bias = false);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<Linear> gate_proj_;
    std::shared_ptr<Linear> up_proj_;
    std::shared_ptr<Linear> down_proj_;
    bool use_silu_;
};

/**
 * @brief Rotary Position Embedding (RoPE)
 *
 * Applies rotary position embeddings to queries and keys.
 * From "RoFormer: Enhanced Transformer with Rotary Position Embedding"
 */
class RotaryPositionEmbedding : public Module {
public:
    /**
     * @brief Construct RoPE layer
     *
     * @param dim Head dimension (must be even)
     * @param max_seq_len Maximum sequence length
     * @param base Base for frequency computation (default: 10000)
     */
    RotaryPositionEmbedding(int64_t dim, int64_t max_seq_len = 2048,
                            double base = 10000.0);

    /**
     * @brief Apply rotary embeddings to input
     *
     * @param x Input tensor of shape (batch, seq_len, n_heads, head_dim)
     * @param seq_offset Starting position in sequence (for caching)
     * @return Tensor with rotary embeddings applied
     */
    auto forward(const Variable& x, int64_t seq_offset = 0) -> Variable;

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t dim_;
    int64_t max_seq_len_;
    double base_;
    Tensor cos_cache_;  ///< Precomputed cosines
    Tensor sin_cache_;  ///< Precomputed sines

    void precompute_freqs();
};

/**
 * @brief HRM Transformer Block with Post-Norm and optional RoPE
 *
 * A single transformer block used in both H and L modules.
 * Architecture:
 *   x -> MultiheadAttention -> Dropout -> x + residual -> Norm
 *     -> FFN (GLU) -> Dropout -> x + residual -> Norm
 */
class HRMBlock : public Module {
public:
    /**
     * @brief Construct HRM transformer block
     *
     * @param d_model Model dimension
     * @param n_heads Number of attention heads
     * @param d_feedforward Feed-forward dimension
     * @param dropout Dropout probability
     * @param use_post_norm Use post-norm (true) or pre-norm (false)
     * @param max_seq_len Maximum sequence length for RoPE
     */
    HRMBlock(int64_t d_model, int64_t n_heads, int64_t d_feedforward,
             double dropout = 0.1, bool use_post_norm = true,
             int64_t max_seq_len = 512);

    /**
     * @brief Forward pass through transformer block
     *
     * @param x Input tensor
     * @param context Optional context from other module (for cross-attention)
     * @param mask Optional attention mask
     * @return Output tensor
     */
    auto forward(const Variable& x,
                 const Variable& context = Variable{},
                 const Tensor& mask = Tensor{}) -> Variable;

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t d_model_;
    int64_t n_heads_;
    bool use_post_norm_;

    std::shared_ptr<MultiheadAttention> self_attn_;
    std::shared_ptr<MultiheadAttention> cross_attn_;  ///< For L reading from H
    std::shared_ptr<GatedLinearUnit> ffn_;
    std::shared_ptr<RMSNorm> norm1_;
    std::shared_ptr<RMSNorm> norm2_;
    std::shared_ptr<RMSNorm> norm3_;  ///< For cross-attention
    std::shared_ptr<Dropout> dropout_;
    std::shared_ptr<RotaryPositionEmbedding> rope_;
};

/**
 * @brief Adaptive Computational Time (ACT) module
 *
 * Dynamically determines the number of computation steps based on
 * input complexity. Uses a learned halting probability.
 */
class AdaptiveComputationalTime : public Module {
public:
    /**
     * @brief Construct ACT module
     *
     * @param d_model Model dimension
     * @param max_steps Maximum number of steps
     * @param threshold Halting threshold (default: 0.99)
     */
    AdaptiveComputationalTime(int64_t d_model, int64_t max_steps,
                              double threshold = 0.99);

    /**
     * @brief Compute halting probability for current state
     *
     * @param state Current hidden state
     * @return Halting probability (scalar per batch element)
     */
    auto compute_halt_prob(const Variable& state) -> Variable;

    /**
     * @brief Check if computation should halt
     *
     * @param cumulative_prob Cumulative halting probability
     * @return True if all batch elements should halt
     */
    auto should_halt(const Variable& cumulative_prob) -> bool;

    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<Linear> halt_proj_;
    double threshold_;
    int64_t max_steps_;
};

/**
 * @brief Q-Learning based Adaptive Computational Time
 *
 * Implements the Q-learning ACT mechanism from the HRM paper.
 * Uses a learned Q-function to decide whether to halt or continue
 * computation, treating the decision as an episodic MDP.
 *
 * Actions:
 * - halt: Stop computation and output current result
 * - continue: Perform another computation cycle
 *
 * Reward:
 * - +1 if prediction is correct after halting
 * - 0 for continuation steps
 *
 * The Q-values are updated using temporal difference learning:
 * Q(s, halt) <- reward
 * Q(s, continue) <- max(Q(s', halt), Q(s', continue))
 */
class QLearningACT : public Module {
public:
    /**
     * @brief Construct Q-Learning ACT module
     *
     * @param d_model Model dimension
     * @param max_segments Maximum number of segments
     * @param epsilon Exploration rate (epsilon-greedy)
     * @param gamma Discount factor
     * @param lr Learning rate for Q-value updates
     */
    QLearningACT(int64_t d_model, int64_t max_segments,
                 double epsilon = 0.1, double gamma = 0.99, double lr = 0.01);

    /**
     * @brief Compute Q-values for halt and continue actions
     *
     * @param state Current H-module state (batch, seq_len, d_model)
     * @return Pair of (Q_halt, Q_continue) tensors, each (batch, seq_len)
     */
    auto compute_q_values(const Variable& state) -> std::pair<Variable, Variable>;

    /**
     * @brief Select action using epsilon-greedy policy
     *
     * @param q_halt Q-value for halt action
     * @param q_continue Q-value for continue action
     * @param training Whether in training mode (enables exploration)
     * @return True if should halt, false if should continue
     */
    auto select_action(const Variable& q_halt, const Variable& q_continue,
                       bool training = true) -> bool;

    /**
     * @brief Update Q-values based on observed reward
     *
     * @param state State where action was taken
     * @param action Action taken (true = halt, false = continue)
     * @param reward Observed reward (1.0 for correct, 0.0 otherwise)
     * @param next_state Next state (for continue action)
     * @param done Whether episode is done
     * @return TD error for monitoring
     */
    auto update_q_values(const Variable& state, bool action, double reward,
                         const Variable& next_state, bool done) -> double;

    /**
     * @brief Compute Q-learning loss for training
     *
     * @param states Batch of states
     * @param actions Batch of actions taken
     * @param rewards Batch of rewards
     * @param next_states Batch of next states
     * @param dones Batch of done flags
     * @return Q-learning loss (BCE between predicted and target Q-values)
     */
    auto compute_loss(const std::vector<Variable>& states,
                      const std::vector<bool>& actions,
                      const std::vector<double>& rewards,
                      const std::vector<Variable>& next_states,
                      const std::vector<bool>& dones) -> Variable;

    /**
     * @brief Get statistics from Q-learning
     */
    struct QLearningStats {
        double avg_q_halt;      ///< Average Q-value for halt
        double avg_q_continue;  ///< Average Q-value for continue
        double exploration_rate; ///< Current exploration rate
        int64_t total_decisions; ///< Total number of decisions made
        int64_t halt_decisions;  ///< Number of halt decisions
    };
    auto stats() const -> const QLearningStats& { return stats_; }

    /**
     * @brief Decay exploration rate
     *
     * @param decay_rate Multiplicative decay factor
     * @param min_epsilon Minimum exploration rate
     */
    auto decay_epsilon(double decay_rate = 0.995, double min_epsilon = 0.01) -> void;

    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<Linear> q_head_;  ///< Projects state to Q-values [halt, continue]
    double epsilon_;                   ///< Exploration rate
    double gamma_;                     ///< Discount factor
    double lr_;                        ///< Learning rate for Q-updates
    int64_t max_segments_;

    QLearningStats stats_;
    std::mt19937 rng_;  ///< Random number generator for exploration

    /**
     * @brief Pool state across sequence for Q-value computation
     *
     * @param state State tensor (batch, seq_len, d_model)
     * @return Pooled state (batch, d_model)
     */
    auto pool_state(const Variable& state) -> Variable;
};

/**
 * @brief Hierarchical Reasoning Model (HRM)
 *
 * A brain-inspired recurrent architecture with two interdependent modules:
 * - H-module: High-level, slow, abstract planning
 * - L-module: Low-level, fast, detailed computation
 *
 * The model processes through N high-level cycles, each containing T
 * low-level timesteps. This enables computational depth while maintaining
 * training stability through the approximate gradient method.
 *
 * Key innovations:
 * 1. Hierarchical cycling: L converges before H updates
 * 2. Approximate gradient: O(1) memory via hidden state detachment
 * 3. Deep supervision: Loss at each H cycle for stable training
 * 4. Q-Learning ACT: Dynamic compute based on task complexity
 * 5. Stablemax: Numerically stable output activation
 * 6. LeCun initialization: Better gradient flow
 *
 * @code
 * HRMConfig config;
 * config.d_model = 256;
 * config.n_high_cycles = 4;
 * config.t_low_steps = 8;
 * config.vocab_size = 10000;
 * config.use_qlearning_act = true;
 * config.use_stablemax = true;
 *
 * HRM model(config);
 * model.to(Device::vulkan());
 *
 * Variable input(Tensor({batch, seq_len}, DType::Int64, Device::vulkan()), false);
 * auto [output, aux_outputs] = model.forward_with_aux(input);
 * @endcode
 */
class HRM : public Module {
public:
    /**
     * @brief Construct HRM from configuration
     *
     * @param config HRM configuration
     */
    explicit HRM(const HRMConfig& config);

    /**
     * @brief Forward pass returning final output only
     *
     * @param input Input tensor (batch, seq_len) for token IDs or
     *              (batch, seq_len, d_model) for embeddings
     * @param mask Optional attention mask
     * @return Output tensor (batch, seq_len, num_classes) or (batch, seq_len, d_model)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Forward pass with auxiliary outputs for deep supervision
     *
     * @param input Input tensor
     * @param mask Optional attention mask
     * @return Pair of (final_output, vector of intermediate outputs)
     */
    auto forward_with_aux(const Variable& input,
                          const Tensor& mask = Tensor{})
        -> std::pair<Variable, std::vector<Variable>>;

    /**
     * @brief Forward pass with segment-based training (for deep supervision)
     *
     * Runs multiple segments, each with its own forward pass and gradient.
     * This is the training mode described in the HRM paper.
     *
     * @param input Input tensor
     * @param targets Target tensor for computing reward (for Q-learning ACT)
     * @param mask Optional attention mask
     * @return Tuple of (final_output, segment_outputs, q_learning_loss)
     */
    auto forward_with_segments(const Variable& input,
                               const Variable& targets = Variable{},
                               const Tensor& mask = Tensor{})
        -> std::tuple<Variable, std::vector<Variable>, Variable>;

    /**
     * @brief Get configuration
     */
    auto config() const -> const HRMConfig& { return config_; }

    /**
     * @brief Get number of parameters
     */
    auto num_parameters() const -> int64_t;

    /**
     * @brief Get statistics from last forward pass
     */
    struct ForwardStats {
        int64_t actual_high_cycles;   ///< Actual H cycles used (may differ with ACT)
        int64_t actual_low_steps;     ///< Actual L steps used
        int64_t actual_segments;      ///< Actual segments used
        double h_participation_ratio; ///< Dimensionality measure for H
        double l_participation_ratio; ///< Dimensionality measure for L
        double avg_q_halt;            ///< Average Q-value for halt action
        double avg_q_continue;        ///< Average Q-value for continue action
    };
    auto last_forward_stats() const -> const ForwardStats& { return stats_; }

    /**
     * @brief Get Q-Learning ACT module (for external training/monitoring)
     */
    auto get_qlearning_act() -> std::shared_ptr<QLearningACT> { return qlearning_act_; }

    /**
     * @brief Apply proper initialization (LeCun + truncated normal)
     *
     * Should be called after construction if using custom initialization.
     */
    auto apply_hrm_initialization() -> void;

private:
    HRMConfig config_;
    ForwardStats stats_;

    // Embedding layer (optional, based on vocab_size)
    std::shared_ptr<Embedding> embedding_;  ///< Proper embedding layer (not Linear)

    // H-module (high-level, slow)
    std::shared_ptr<HRMBlock> h_module_;

    // L-module (low-level, fast)
    std::shared_ptr<HRMBlock> l_module_;

    // State initialization
    std::shared_ptr<Linear> h_init_proj_;
    std::shared_ptr<Linear> l_init_proj_;

    // Fixed initial hidden states (sampled from truncated normal)
    Tensor h_init_state_;  ///< Initial H state template
    Tensor l_init_state_;  ///< Initial L state template

    // Output projection (optional, based on num_classes)
    std::shared_ptr<Linear> output_proj_;
    std::shared_ptr<RMSNorm> output_norm_;

    // Adaptive Computational Time (mutually exclusive)
    std::shared_ptr<AdaptiveComputationalTime> act_;  ///< Simple halting ACT
    std::shared_ptr<QLearningACT> qlearning_act_;     ///< Q-learning based ACT

    /**
     * @brief Initialize hidden states from input
     *
     * Uses truncated normal initialization as specified in the HRM paper.
     */
    auto init_states(const Variable& x)
        -> std::pair<Variable, Variable>;

    /**
     * @brief Run one complete H cycle (T low-level steps + H update)
     */
    auto run_h_cycle(Variable& h_state, Variable& l_state,
                     const Tensor& mask) -> Variable;

    /**
     * @brief Apply output activation (stablemax or softmax)
     */
    auto apply_output_activation(const Variable& logits) -> Variable;

    /**
     * @brief Compute participation ratio for analysis
     */
    auto compute_participation_ratio(const Variable& state) -> double;

    /**
     * @brief Initialize weights with LeCun initialization
     */
    auto apply_lecun_init_to_module(Module* module) -> void;
};

/**
 * @brief Deep supervision loss for HRM training
 *
 * Applies loss at each H cycle output with exponentially increasing weights.
 * This provides frequent feedback to the H-module during training.
 *
 * @param outputs Vector of outputs from each H cycle
 * @param targets Target tensor
 * @param loss_fn Base loss function to use
 * @param weight_decay Decay factor for earlier outputs (default: 0.5)
 * @return Weighted sum of losses
 */
auto hrm_deep_supervision_loss(
    const std::vector<Variable>& outputs,
    const Variable& targets,
    std::function<Variable(const Variable&, const Variable&)> loss_fn,
    double weight_decay = 0.5) -> Variable;

} // namespace nn
} // namespace tenzor
