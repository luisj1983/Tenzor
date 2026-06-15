/**
 * @file rnn.hpp
 * @brief Recurrent Neural Network (RNN) layers
 *
 * Implements vanilla RNN, LSTM, and GRU recurrent layers with
 * support for multi-layer stacking, bidirectional processing,
 * and dropout regularization.
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include "../module.hpp"
#include "linear.hpp"
#include "dropout.hpp"

namespace tenzor {
namespace nn {

// ============================================================================
// RNN (Vanilla Recurrent Neural Network)
// ============================================================================

/**
 * @brief Single RNN cell.
 *
 * Applies a single recurrent step:
 *   h_t = activation(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
 *
 * Supports both tanh and ReLU activations.
 *
 * Shape:
 * - Input: (batch, input_size)
 * - Hidden: (batch, hidden_size)
 * - Output: (batch, hidden_size)
 *
 * @code
 * RNNCell cell(128, 256, "tanh");
 * Variable x(Tensor({32, 128}, DType::Float32, Device::cpu()), true);
 * Variable h(Tensor({32, 256}, DType::Float32, Device::cpu()), true);
 * Variable h_next = cell.forward(x, h);
 * @endcode
 */
class RNNCell : public Module {
public:
    // Bring base class forward into scope (avoid hiding by 2-param forward)
    using Module::forward;

    /**
     * @brief Construct RNN cell.
     *
     * @param input_size Size of input features
     * @param hidden_size Size of hidden state
     * @param nonlinearity Activation function ("tanh" or "relu", default: "tanh")
     * @param bias If true, add learnable bias (default: true)
     */
    RNNCell(int64_t input_size, int64_t hidden_size,
            const std::string& nonlinearity = "tanh",
            bool bias = true);

    /**
     * @brief Forward pass through RNN cell.
     *
     * @param input Input tensor of shape (batch, input_size)
     * @param hx Hidden state of shape (batch, hidden_size). If empty, zero-initialized.
     * @return New hidden state of shape (batch, hidden_size)
     */
    auto forward(const Variable& input, const Variable& hx) -> Variable;

    /**
     * @brief Override base Module forward (single parameter).
     */
    auto forward_impl(const Variable& input) -> Variable override {
        return forward(input, Variable{});
    }

    /**
     * @brief Forward pass with pre-computed input gates (for batched optimization).
     *
     * Used by RNN to avoid redundant input->hidden computations when processing
     * multiple timesteps. Instead of calling input_layer_->forward() per timestep,
     * RNN pre-computes all input gates at once and passes them here.
     *
     * @param precomputed_ih Pre-computed input-to-hidden output of shape (batch, hidden_size)
     * @param hx Hidden state of shape (batch, hidden_size)
     * @return New hidden state of shape (batch, hidden_size)
     */
    auto forward_with_precomputed_ih(const Tensor& precomputed_ih, const Variable& hx) -> Variable;

    /** @brief Get input-to-hidden weight layer for batched optimization */
    auto weight_ih() const -> std::shared_ptr<Linear> { return input_layer_; }

    /** @brief Get hidden size */
    auto hidden_size() const -> int64_t { return hidden_size_; }

private:
    int64_t input_size_;
    int64_t hidden_size_;
    std::string nonlinearity_;
    std::shared_ptr<Linear> input_layer_;   ///< W_ih and b_ih
    std::shared_ptr<Linear> hidden_layer_;  ///< W_hh and b_hh
};

/**
 * @brief Multi-layer RNN.
 *
 * Applies a multi-layer Elman RNN with tanh or ReLU non-linearity to an
 * input sequence. For each element in the sequence:
 *   h_t = activation(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
 *
 * Supports:
 * - Multi-layer stacking with dropout between layers
 * - Bidirectional processing (processes sequence forward and backward)
 * - Batch-first or sequence-first input format
 *
 * Shape:
 * - Input: (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
 * - Hidden: (num_layers * num_directions, batch, hidden_size)
 * - Output: (seq_len, batch, num_directions * hidden_size) or (batch, seq_len, ...) if batch_first
 * - Final hidden: (num_layers * num_directions, batch, hidden_size)
 *
 * @code
 * RNN rnn(128, 256, 2, "tanh", true, false, 0.5, true);  // 2 layers, bidirectional
 * Variable x(Tensor({10, 32, 128}, DType::Float32, Device::cpu()), true);
 * auto [output, h_n] = rnn.forward(x);
 * // output: (10, 32, 512), h_n: (4, 32, 256)
 * @endcode
 */
class RNN : public Module {
public:
    // Bring base class forward into scope (avoid hiding by 2-param forward)
    using Module::forward;

    /**
     * @brief Construct multi-layer RNN.
     *
     * @param input_size Size of input features
     * @param hidden_size Size of hidden state
     * @param num_layers Number of stacked RNN layers (default: 1)
     * @param nonlinearity Activation function ("tanh" or "relu", default: "tanh")
     * @param bias If true, use bias (default: true)
     * @param batch_first If true, input is (batch, seq_len, features) (default: false)
     * @param dropout Dropout probability between layers (default: 0.0)
     * @param bidirectional If true, process sequence in both directions (default: false)
     */
    RNN(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1,
        const std::string& nonlinearity = "tanh",
        bool bias = true, bool batch_first = false, double dropout = 0.0,
        bool bidirectional = false,
        // Optional separate activation for backward-direction cells when
        // bidirectional. Empty string → reuse `nonlinearity`. ONNX RNN
        // exports may specify a different activation per direction.
        const std::string& nonlinearity_bwd = "");

    /**
     * @brief Forward pass through RNN.
     *
     * @param input Input sequence tensor
     * @param hx Initial hidden state. If empty, zero-initialized.
     * @return Tuple of (output, h_n) where:
     *   - output: All hidden states for each time step
     *   - h_n: Final hidden state for each layer
     */
    auto forward(const Variable& input, const Variable& hx) -> std::pair<Variable, Variable>;

    /**
     * @brief Override base Module forward (single parameter).
     */
    auto forward_impl(const Variable& input) -> Variable override {
        return forward(input, Variable{}).first;
    }

private:
    int64_t input_size_;
    int64_t hidden_size_;
    int64_t num_layers_;
    std::string nonlinearity_;
    bool batch_first_;
    bool bidirectional_;
    double dropout_p_;

    std::vector<std::shared_ptr<RNNCell>> forward_cells_;
    std::vector<std::shared_ptr<RNNCell>> backward_cells_;  // For bidirectional
    std::shared_ptr<Dropout> dropout_;
};

// ============================================================================
// LSTM (Long Short-Term Memory)
// ============================================================================

/**
 * @brief Single LSTM cell.
 *
 * Applies a single LSTM step:
 *   i_t = σ(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)  # Input gate
 *   f_t = σ(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)  # Forget gate
 *   g_t = tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)  # Cell gate
 *   o_t = σ(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)  # Output gate
 *   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t  # Cell state
 *   h_t = o_t ⊙ tanh(c_t)  # Hidden state
 *
 * Shape:
 * - Input: (batch, input_size)
 * - Hidden: (batch, hidden_size)
 * - Cell: (batch, hidden_size)
 * - Output: ((batch, hidden_size), (batch, hidden_size))
 *
 * @code
 * LSTMCell cell(128, 256);
 * Variable x(Tensor({32, 128}, DType::Float32, Device::cpu()), true);
 * Variable h(Tensor({32, 256}, DType::Float32, Device::cpu()), true);
 * Variable c(Tensor({32, 256}, DType::Float32, Device::cpu()), true);
 * auto [h_next, c_next] = cell.forward(x, h, c);
 * @endcode
 */
class LSTMCell : public Module {
public:
    // Bring base class forward into scope (avoid hiding by 3-param forward)
    using Module::forward;

    /**
     * @brief Construct LSTM cell.
     *
     * @param input_size Size of input features
     * @param hidden_size Size of hidden state
     * @param bias If true, add learnable bias (default: true)
     */
    LSTMCell(int64_t input_size, int64_t hidden_size, bool bias = true);

    /**
     * @brief Construct LSTM cell with explicit recurrent dimension (audit G1).
     *
     * Used by LSTM with `proj_size > 0` (PyTorch LSTMP): the recurrent
     * connection weight `W_hh` has shape (4*hidden_size, recurrent_size)
     * instead of the default (4*hidden_size, hidden_size). The cell still
     * produces an output of `hidden_size` per step — the LSTM module
     * applies the hidden→proj projection externally.
     *
     * @param input_size Size of input features
     * @param hidden_size Size of cell hidden state (always)
     * @param recurrent_size Size of the recurrent input (== proj_size when
     *        used inside an LSTM with projection)
     * @param bias If true, add learnable bias on weight_ih (no bias on weight_hh)
     */
    LSTMCell(int64_t input_size, int64_t hidden_size,
             int64_t recurrent_size, bool bias);

    /**
     * @brief Forward pass through LSTM cell.
     *
     * @param input Input tensor of shape (batch, input_size)
     * @param hx Hidden state of shape (batch, hidden_size). If empty, zero-initialized.
     * @param cx Cell state of shape (batch, hidden_size). If empty, zero-initialized.
     * @return Tuple of (h_next, c_next) both of shape (batch, hidden_size)
     */
    auto forward(const Variable& input, const Variable& hx, const Variable& cx)
        -> std::pair<Variable, Variable>;

    /**
     * @brief Forward pass with pre-computed input gates (for batched optimization).
     *
     * This method is used by LSTM to avoid redundant input->hidden computations
     * when processing multiple timesteps. Instead of calling weight_ih->forward()
     * for each timestep, LSTM pre-computes all input gates at once and passes them here.
     *
     * @param gates_ih Pre-computed input gates of shape (batch, 4*hidden_size)
     * @param hx Hidden state of shape (batch, hidden_size)
     * @param cx Cell state of shape (batch, hidden_size)
     * @return Tuple of (h_next, c_next) both of shape (batch, hidden_size)
     */
    auto forward_with_precomputed_ih(const Tensor& gates_ih, const Variable& hx, const Variable& cx)
        -> std::pair<Variable, Variable>;

    /**
     * @brief Override base Module forward (single parameter).
     */
    auto forward_impl(const Variable& input) -> Variable override {
        return forward(input, Variable{}, Variable{}).first;
    }

    /** @brief Get input-to-hidden weight matrix for batched optimization */
    std::shared_ptr<Linear> weight_ih() const { return weight_ih_; }

    /** @brief Get hidden-to-hidden weight matrix for batched optimization */
    std::shared_ptr<Linear> weight_hh() const { return weight_hh_; }

    /** @brief Get hidden size */
    int64_t hidden_size() const { return hidden_size_; }

private:
    int64_t input_size_;
    int64_t hidden_size_;
    /// Audit G1: dim of the recurrent input. Equals hidden_size by default;
    /// equals LSTM's proj_size when used inside an LSTMP.
    int64_t recurrent_size_;
    std::shared_ptr<Linear> weight_ih_;   ///< Input-to-hidden weights (4*hidden_size, input_size)
    std::shared_ptr<Linear> weight_hh_;   ///< Hidden-to-hidden weights (4*hidden_size, recurrent_size)
};

/**
 * @brief Multi-layer LSTM.
 *
 * Applies a multi-layer Long Short-Term Memory (LSTM) RNN to an input sequence.
 *
 * Supports:
 * - Multi-layer stacking with dropout between layers
 * - Bidirectional processing
 * - Batch-first or sequence-first input format
 * - Optional projection (proj_size > 0 projects hidden state before output)
 *
 * Shape:
 * - Input: (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
 * - Hidden: (num_layers * num_directions, batch, hidden_size or proj_size)
 * - Cell: (num_layers * num_directions, batch, hidden_size)
 * - Output: (seq_len, batch, num_directions * (proj_size or hidden_size))
 *
 * @code
 * LSTM lstm(128, 256, 2, true, false, 0.5, true);  // 2 layers, bidirectional
 * Variable x(Tensor({10, 32, 128}, DType::Float32, Device::cpu()), true);
 * auto [output, hidden_state] = lstm.forward(x);
 * auto [h_n, c_n] = hidden_state;
 * @endcode
 */
class LSTM : public Module {
public:
    // Bring base class forward into scope (avoid hiding by 2-param forward)
    using Module::forward;

    /**
     * @brief Construct multi-layer LSTM.
     *
     * @param input_size Size of input features
     * @param hidden_size Size of hidden state
     * @param num_layers Number of stacked LSTM layers (default: 1)
     * @param bias If true, use bias (default: true)
     * @param batch_first If true, input is (batch, seq_len, features) (default: false)
     * @param dropout Dropout probability between layers (default: 0.0)
     * @param bidirectional If true, process sequence in both directions (default: false)
     * @param proj_size If > 0, project hidden state to this size (default: 0)
     */
    LSTM(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1,
         bool bias = true, bool batch_first = false, double dropout = 0.0,
         bool bidirectional = false, int64_t proj_size = 0);

    /**
     * @brief Forward pass through LSTM.
     *
     * @param input Input sequence tensor
     * @param hx Tuple of (h_0, c_0) initial states. If empty, zero-initialized.
     * @return Tuple of (output, (h_n, c_n)) where:
     *   - output: All hidden states for each time step
     *   - h_n: Final hidden state for each layer
     *   - c_n: Final cell state for each layer
     */
    auto forward(const Variable& input, const std::pair<Variable, Variable>& hx,
                 const Tensor& lengths = {})
        -> std::pair<Variable, std::pair<Variable, Variable>>;

    /**
     * @brief Override base Module forward (single parameter).
     */
    auto forward_impl(const Variable& input) -> Variable override {
        return forward(input, {Variable{}, Variable{}}).first;
    }

    auto extra_repr() const -> std::string override {
        return "input_size=" + std::to_string(input_size_) +
               ", hidden_size=" + std::to_string(hidden_size_) +
               ", num_layers=" + std::to_string(num_layers_) +
               ", bidirectional=" + std::string(bidirectional_ ? "True" : "False");
    }

private:
    int64_t input_size_;
    int64_t hidden_size_;
    int64_t num_layers_;
    bool batch_first_;
    bool bidirectional_;
    double dropout_p_;
    int64_t proj_size_;

    std::vector<std::shared_ptr<LSTMCell>> forward_cells_;
    std::vector<std::shared_ptr<LSTMCell>> backward_cells_;  // For bidirectional
    // Audit G1: per-layer hidden→proj projection (no bias) for proj_size > 0.
    // Empty when proj_size == 0.
    std::vector<std::shared_ptr<Linear>> forward_projections_;
    std::vector<std::shared_ptr<Linear>> backward_projections_;
    std::shared_ptr<Dropout> dropout_;
};

// ============================================================================
// GRU (Gated Recurrent Unit)
// ============================================================================

/**
 * @brief Single GRU cell.
 *
 * Applies a single GRU step:
 *   r_t = σ(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)  # Reset gate
 *   z_t = σ(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)  # Update gate
 *   n_t = tanh(W_in * x_t + b_in + r_t ⊙ (W_hn * h_{t-1} + b_hn))  # New gate
 *   h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}  # Hidden state
 *
 * Shape:
 * - Input: (batch, input_size)
 * - Hidden: (batch, hidden_size)
 * - Output: (batch, hidden_size)
 *
 * @code
 * GRUCell cell(128, 256);
 * Variable x(Tensor({32, 128}, DType::Float32, Device::cpu()), true);
 * Variable h(Tensor({32, 256}, DType::Float32, Device::cpu()), true);
 * Variable h_next = cell.forward(x, h);
 * @endcode
 */
class GRUCell : public Module {
public:
    // Bring base class forward into scope (avoid hiding by 2-param forward)
    using Module::forward;

    /**
     * @brief Construct GRU cell.
     *
     * @param input_size Size of input features
     * @param hidden_size Size of hidden state
     * @param bias If true, add learnable bias (default: true)
     * @param linear_before_reset Selects the new-gate ("n"/"h") formulation:
     *   - true (default, preserves historical Tenzor behavior == ONNX
     *     linear_before_reset=1): the reset gate multiplies the recurrent
     *     matmul RESULT, n_t = tanh(Xt·Wn + Wbn + r_t ⊙ (Ht-1·Rn + Rbn)).
     *   - false (== ONNX default linear_before_reset=0): the reset gate
     *     multiplies the hidden state BEFORE the recurrent matmul,
     *     n_t = tanh(Xt·Wn + Wbn + (r_t ⊙ Ht-1)·Rn + Rbn).
     *   The z and r gates are identical in both modes.
     */
    GRUCell(int64_t input_size, int64_t hidden_size, bool bias = true,
            bool linear_before_reset = true);

    /**
     * @brief Forward pass through GRU cell.
     *
     * @param input Input tensor of shape (batch, input_size)
     * @param hx Hidden state of shape (batch, hidden_size). If empty, zero-initialized.
     * @return New hidden state of shape (batch, hidden_size)
     */
    auto forward(const Variable& input, const Variable& hx) -> Variable;

    /**
     * @brief Forward pass with pre-computed input gates (for batched optimization).
     *
     * This method is used by GRU to avoid redundant input->hidden computations
     * when processing multiple timesteps. Instead of calling weight_ih->forward()
     * for each timestep, GRU pre-computes all input gates at once and passes them here.
     *
     * @param gates_ih Pre-computed input gates of shape (batch, 3*hidden_size)
     * @param hx Hidden state of shape (batch, hidden_size)
     * @return New hidden state of shape (batch, hidden_size)
     */
    auto forward_with_precomputed_ih(const Tensor& gates_ih, const Variable& hx) -> Variable;

    /**
     * @brief Override base Module forward (single parameter).
     */
    auto forward_impl(const Variable& input) -> Variable override {
        return forward(input, Variable{});
    }

    /** @brief Get input-to-hidden weight matrix for batched optimization */
    std::shared_ptr<Linear> weight_ih() const { return weight_ih_; }

    /** @brief Get hidden-to-hidden weight matrix for batched optimization */
    std::shared_ptr<Linear> weight_hh() const { return weight_hh_; }

    /** @brief Get hidden size */
    int64_t hidden_size() const { return hidden_size_; }

    /** @brief Whether the cell uses the linear_before_reset=1 new-gate form. */
    bool linear_before_reset() const { return linear_before_reset_; }

private:
    int64_t input_size_;
    int64_t hidden_size_;
    bool linear_before_reset_;  ///< true == mode 1 (default), false == ONNX mode 0
    // Combined weight matrices for all 3 gates (reset, update, new)
    // More efficient: 2 linear layers instead of 6
    std::shared_ptr<Linear> weight_ih_;    ///< (3*hidden, input) for all gates
    std::shared_ptr<Linear> weight_hh_;    ///< (3*hidden, hidden) for all gates
};

/**
 * @brief Multi-layer GRU.
 *
 * Applies a multi-layer Gated Recurrent Unit (GRU) RNN to an input sequence.
 * GRU is similar to LSTM but with fewer parameters and no separate cell state.
 *
 * Supports:
 * - Multi-layer stacking with dropout between layers
 * - Bidirectional processing
 * - Batch-first or sequence-first input format
 *
 * Shape:
 * - Input: (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
 * - Hidden: (num_layers * num_directions, batch, hidden_size)
 * - Output: (seq_len, batch, num_directions * hidden_size)
 *
 * @code
 * GRU gru(128, 256, 2, true, false, 0.5, true);  // 2 layers, bidirectional
 * Variable x(Tensor({10, 32, 128}, DType::Float32, Device::cpu()), true);
 * auto [output, h_n] = gru.forward(x);
 * @endcode
 */
class GRU : public Module {
public:
    // Bring base class forward into scope (avoid hiding by 2-param forward)
    using Module::forward;

    /**
     * @brief Construct multi-layer GRU.
     *
     * @param input_size Size of input features
     * @param hidden_size Size of hidden state
     * @param num_layers Number of stacked GRU layers (default: 1)
     * @param bias If true, use bias (default: true)
     * @param batch_first If true, input is (batch, seq_len, features) (default: false)
     * @param dropout Dropout probability between layers (default: 0.0)
     * @param bidirectional If true, process sequence in both directions (default: false)
     * @param linear_before_reset Selects the new-gate formulation for every
     *   cell (see GRUCell). true (default) preserves historical Tenzor
     *   behavior (== ONNX linear_before_reset=1); false selects the ONNX
     *   default (linear_before_reset=0). Note: the fused inference / cuDNN
     *   training fast paths only implement mode 1; when mode 0 is selected the
     *   GRU forces the per-timestep autograd path.
     */
    GRU(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1,
        bool bias = true, bool batch_first = false, double dropout = 0.0,
        bool bidirectional = false, bool linear_before_reset = true);

    /**
     * @brief Forward pass through GRU.
     *
     * @param input Input sequence tensor
     * @param hx Initial hidden state. If empty, zero-initialized.
     * @param lengths Optional tensor of sequence lengths per batch element for masking.
     * @return Tuple of (output, h_n) where:
     *   - output: All hidden states for each time step
     *   - h_n: Final hidden state for each layer
     */
    auto forward(const Variable& input, const Variable& hx,
                 const Tensor& lengths = {}) -> std::pair<Variable, Variable>;

    /**
     * @brief Override base Module forward (single parameter).
     */
    auto forward_impl(const Variable& input) -> Variable override {
        return forward(input, Variable{}).first;
    }

    auto extra_repr() const -> std::string override {
        return "input_size=" + std::to_string(input_size_) +
               ", hidden_size=" + std::to_string(hidden_size_) +
               ", num_layers=" + std::to_string(num_layers_) +
               ", bidirectional=" + std::string(bidirectional_ ? "True" : "False");
    }

private:
    int64_t input_size_;
    int64_t hidden_size_;
    int64_t num_layers_;
    bool batch_first_;
    bool bidirectional_;
    double dropout_p_;
    bool linear_before_reset_;  ///< true == mode 1 (default), false == ONNX mode 0

    std::vector<std::shared_ptr<GRUCell>> forward_cells_;
    std::vector<std::shared_ptr<GRUCell>> backward_cells_;  // For bidirectional
    std::shared_ptr<Dropout> dropout_;
};

} // namespace nn
} // namespace tenzor
