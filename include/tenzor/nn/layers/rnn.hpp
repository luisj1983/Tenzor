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
        bool bidirectional = false);

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
     * @brief Override base Module forward (single parameter).
     */
    auto forward_impl(const Variable& input) -> Variable override {
        return forward(input, Variable{}, Variable{}).first;
    }

private:
    int64_t input_size_;
    int64_t hidden_size_;
    std::shared_ptr<Linear> weight_ih_;   ///< Input-to-hidden weights (4*hidden_size, input_size)
    std::shared_ptr<Linear> weight_hh_;   ///< Hidden-to-hidden weights (4*hidden_size, hidden_size)
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
    auto forward(const Variable& input, const std::pair<Variable, Variable>& hx)
        -> std::pair<Variable, std::pair<Variable, Variable>>;

    /**
     * @brief Override base Module forward (single parameter).
     */
    auto forward_impl(const Variable& input) -> Variable override {
        return forward(input, {Variable{}, Variable{}}).first;
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
     */
    GRUCell(int64_t input_size, int64_t hidden_size, bool bias = true);

    /**
     * @brief Forward pass through GRU cell.
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

private:
    int64_t input_size_;
    int64_t hidden_size_;
    std::shared_ptr<Linear> reset_gate_input_;    ///< W_ir, b_ir
    std::shared_ptr<Linear> reset_gate_hidden_;   ///< W_hr, b_hr
    std::shared_ptr<Linear> update_gate_input_;   ///< W_iz, b_iz
    std::shared_ptr<Linear> update_gate_hidden_;  ///< W_hz, b_hz
    std::shared_ptr<Linear> new_gate_input_;      ///< W_in, b_in
    std::shared_ptr<Linear> new_gate_hidden_;     ///< W_hn, b_hn
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
     */
    GRU(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1,
        bool bias = true, bool batch_first = false, double dropout = 0.0,
        bool bidirectional = false);

    /**
     * @brief Forward pass through GRU.
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
    bool batch_first_;
    bool bidirectional_;
    double dropout_p_;

    std::vector<std::shared_ptr<GRUCell>> forward_cells_;
    std::vector<std::shared_ptr<GRUCell>> backward_cells_;  // For bidirectional
    std::shared_ptr<Dropout> dropout_;
};

} // namespace nn
} // namespace tenzor
