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
 * - Optional Adaptive Computational Time (ACT)
 */

#pragma once

#include <memory>
#include <vector>
#include <optional>
#include "../module.hpp"
#include "linear.hpp"
#include "dropout.hpp"
#include "normalization.hpp"
#include "attention.hpp"

namespace tenzor {
namespace nn {

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
    bool use_act = false;           ///< Enable Adaptive Computational Time
    double act_threshold = 0.99;    ///< ACT halting threshold
    int64_t max_seq_len = 512;      ///< Maximum sequence length

    // Vocab/output settings
    int64_t vocab_size = 0;         ///< Vocabulary size (0 = no embedding/output layers)
    int64_t num_classes = 0;        ///< Number of output classes (0 = use d_model output)
};

/**
 * @brief RMS Layer Normalization (used in modern transformers)
 *
 * RMSNorm(x) = x / RMS(x) * gamma
 * where RMS(x) = sqrt(mean(x^2) + eps)
 *
 * More efficient than LayerNorm as it doesn't require mean subtraction.
 */
class RMSNorm : public Module {
public:
    /**
     * @brief Construct RMS normalization layer
     *
     * @param normalized_shape Dimensions to normalize over
     * @param eps Epsilon for numerical stability
     */
    explicit RMSNorm(int64_t normalized_shape, double eps = 1e-6);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t normalized_shape_;
    double eps_;
    std::shared_ptr<Variable> weight_;  ///< Learnable scale parameter (gamma)
};

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
 * 4. Optional ACT: Dynamic compute based on task complexity
 *
 * @code
 * HRMConfig config;
 * config.d_model = 256;
 * config.n_high_cycles = 4;
 * config.t_low_steps = 8;
 * config.vocab_size = 10000;
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
        double h_participation_ratio; ///< Dimensionality measure for H
        double l_participation_ratio; ///< Dimensionality measure for L
    };
    auto last_forward_stats() const -> const ForwardStats& { return stats_; }

private:
    HRMConfig config_;
    ForwardStats stats_;

    // Embedding layers (optional, based on vocab_size)
    std::shared_ptr<Linear> embedding_;

    // H-module (high-level, slow)
    std::shared_ptr<HRMBlock> h_module_;

    // L-module (low-level, fast)
    std::shared_ptr<HRMBlock> l_module_;

    // State initialization
    std::shared_ptr<Linear> h_init_proj_;
    std::shared_ptr<Linear> l_init_proj_;

    // Output projection (optional, based on num_classes)
    std::shared_ptr<Linear> output_proj_;
    std::shared_ptr<RMSNorm> output_norm_;

    // Adaptive Computational Time (optional)
    std::shared_ptr<AdaptiveComputationalTime> act_;

    /**
     * @brief Initialize hidden states from input
     */
    auto init_states(const Variable& x)
        -> std::pair<Variable, Variable>;

    /**
     * @brief Run one complete H cycle (T low-level steps + H update)
     */
    auto run_h_cycle(Variable& h_state, Variable& l_state,
                     const Tensor& mask) -> Variable;

    /**
     * @brief Compute participation ratio for analysis
     */
    auto compute_participation_ratio(const Variable& state) -> double;
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
