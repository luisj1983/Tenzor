#include "tenzor/nn/layers/rnn.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/function.hpp"
#include <stdexcept>
#include <sstream>

namespace tenzor::nn {

// ============================================================================
// GRUCell Implementation
// ============================================================================

GRUCell::GRUCell(int64_t input_size, int64_t hidden_size, bool bias)
    : input_size_(input_size), hidden_size_(hidden_size) {

    // GRU has 3 gates:
    // 1. Reset gate: r_t = σ(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
    // 2. Update gate: z_t = σ(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
    // 3. New gate: n_t = tanh(W_in @ x_t + b_in + r_t ⊙ (W_hn @ h_{t-1} + b_hn))

    // Reset gate layers
    reset_gate_input_ = std::make_shared<Linear>(input_size, hidden_size, bias);
    reset_gate_hidden_ = std::make_shared<Linear>(hidden_size, hidden_size, bias);

    // Update gate layers
    update_gate_input_ = std::make_shared<Linear>(input_size, hidden_size, bias);
    update_gate_hidden_ = std::make_shared<Linear>(hidden_size, hidden_size, bias);

    // New gate layers
    new_gate_input_ = std::make_shared<Linear>(input_size, hidden_size, bias);
    new_gate_hidden_ = std::make_shared<Linear>(hidden_size, hidden_size, bias);

    // Register as submodules
    register_module("reset_gate_input", reset_gate_input_);
    register_module("reset_gate_hidden", reset_gate_hidden_);
    register_module("update_gate_input", update_gate_input_);
    register_module("update_gate_hidden", update_gate_hidden_);
    register_module("new_gate_input", new_gate_input_);
    register_module("new_gate_hidden", new_gate_hidden_);
}

auto GRUCell::forward(const Variable& input, const Variable& hx) -> Variable {
    // Validate input shape
    auto input_shape = input.shape();
    if (input_shape.size() != 2) {
        throw std::runtime_error("GRUCell: input must be 2D (batch, input_size)");
    }
    if (input_shape[1] != input_size_) {
        std::ostringstream oss;
        oss << "GRUCell: expected input size " << input_size_
            << " but got " << input_shape[1];
        throw std::runtime_error(oss.str());
    }

    int64_t batch_size = input_shape[0];

    // Initialize hidden state if not provided
    Variable h = hx;
    if (h.tensor().numel() == 0) {
        h = Variable(zeros({batch_size, hidden_size_},
                          input.dtype(), input.device()), false);
    }

    // Validate hidden state shape
    auto h_shape = h.shape();
    if (h_shape.size() != 2 || h_shape[0] != batch_size || h_shape[1] != hidden_size_) {
        throw std::runtime_error("GRUCell: invalid hidden state shape");
    }

    // Compute reset gate: r_t = σ(W_ir @ x_t + W_hr @ h_{t-1})
    auto r_i = reset_gate_input_->forward(input);
    auto r_h = reset_gate_hidden_->forward(h);
    auto r_combined = Variable(r_i.tensor() + r_h.tensor(), true);
    auto r_t = nn::sigmoid(r_combined);

    // Compute update gate: z_t = σ(W_iz @ x_t + W_hz @ h_{t-1})
    auto z_i = update_gate_input_->forward(input);
    auto z_h = update_gate_hidden_->forward(h);
    auto z_combined = Variable(z_i.tensor() + z_h.tensor(), true);
    auto z_t = nn::sigmoid(z_combined);

    // Compute new gate: n_t = tanh(W_in @ x_t + r_t ⊙ (W_hn @ h_{t-1}))
    auto n_i = new_gate_input_->forward(input);
    auto n_h = new_gate_hidden_->forward(h);
    // Apply reset gate to hidden transformation
    auto n_h_reset = Variable(r_t.tensor() * n_h.tensor(), true);
    auto n_combined = Variable(n_i.tensor() + n_h_reset.tensor(), true);
    auto n_t = nn::tanh(n_combined);

    // Compute new hidden state: h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
    // h_t = z_t * h_{t-1} + (1 - z_t) * n_t
    auto z_h_prev = Variable(z_t.tensor() * h.tensor(), true);

    // Compute (1 - z_t)
    auto ones_tensor = ones_like(z_t.tensor());
    auto one_minus_z = Variable(ones_tensor - z_t.tensor(), true);

    auto one_minus_z_n = Variable(one_minus_z.tensor() * n_t.tensor(), true);

    auto h_new = Variable(z_h_prev.tensor() + one_minus_z_n.tensor(), true);

    return h_new;
}

// ============================================================================
// GRU Implementation
// ============================================================================

GRU::GRU(int64_t input_size, int64_t hidden_size, int64_t num_layers,
         bool bias, bool batch_first, double dropout, bool bidirectional)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      num_layers_(num_layers),
      batch_first_(batch_first),
      bidirectional_(bidirectional),
      dropout_p_(dropout) {

    if (num_layers < 1) {
        throw std::invalid_argument("GRU: num_layers must be >= 1");
    }
    if (dropout < 0.0 || dropout > 1.0) {
        throw std::invalid_argument("GRU: dropout must be in [0, 1]");
    }

    // Create forward cells
    for (int64_t i = 0; i < num_layers; ++i) {
        int64_t layer_input_size = (i == 0) ? input_size : hidden_size * (bidirectional_ ? 2 : 1);
        auto cell = std::make_shared<GRUCell>(layer_input_size, hidden_size, bias);
        forward_cells_.push_back(cell);
        register_module("forward_cell_" + std::to_string(i), cell);
    }

    // Create backward cells for bidirectional GRU
    if (bidirectional_) {
        for (int64_t i = 0; i < num_layers; ++i) {
            int64_t layer_input_size = (i == 0) ? input_size : hidden_size * 2;
            auto cell = std::make_shared<GRUCell>(layer_input_size, hidden_size, bias);
            backward_cells_.push_back(cell);
            register_module("backward_cell_" + std::to_string(i), cell);
        }
    }

    // Create dropout layer
    if (dropout_p_ > 0.0 && num_layers > 1) {
        dropout_ = std::make_shared<Dropout>(dropout_p_);
        register_module("dropout", dropout_);
    }
}

auto GRU::forward(const Variable& input, const Variable& hx)
    -> std::pair<Variable, Variable> {

    auto input_shape = input.shape();

    if (input_shape.size() != 3) {
        throw std::runtime_error("GRU: input must be 3D");
    }

    // Determine dimensions
    int64_t seq_len, batch_size, feat_size;
    Variable x = input;

    if (batch_first_) {
        batch_size = input_shape[0];
        seq_len = input_shape[1];
        feat_size = input_shape[2];
        x = Variable(x.tensor().transpose(0, 1), x.requires_grad());
    } else {
        seq_len = input_shape[0];
        batch_size = input_shape[1];
        feat_size = input_shape[2];
    }

    if (feat_size != input_size_) {
        throw std::runtime_error("GRU: input feature size mismatch");
    }

    int64_t num_directions = bidirectional_ ? 2 : 1;

    // Initialize hidden state
    Variable h = hx;
    if (h.tensor().numel() == 0) {
        h = Variable(zeros({num_layers_ * num_directions, batch_size, hidden_size_},
                          input.dtype(), input.device()), false);
    }

    // Validate shape
    auto h_shape = h.shape();
    if (h_shape.size() != 3 || h_shape[0] != num_layers_ * num_directions ||
        h_shape[1] != batch_size || h_shape[2] != hidden_size_) {
        throw std::runtime_error("GRU: invalid hidden state shape");
    }

    // Split states by layer
    std::vector<Variable> h_layers;
    for (int64_t i = 0; i < num_layers_ * num_directions; ++i) {
        auto h_tensor = h.tensor();
        auto h_layer = h_tensor.slice(0, i, i + 1).reshape({batch_size, hidden_size_});
        h_layers.push_back(Variable(h_layer, false));
    }

    // Process through layers
    Variable layer_input = x;
    std::vector<Variable> final_h_states;

    for (int64_t layer = 0; layer < num_layers_; ++layer) {
        auto& forward_cell = forward_cells_[layer];
        Variable forward_h = h_layers[layer * num_directions];

        std::vector<Variable> forward_outputs;

        // Forward pass
        for (int64_t t = 0; t < seq_len; ++t) {
            auto x_tensor = layer_input.tensor();
            int64_t layer_feat_size = (layer == 0) ? input_size_ : (hidden_size_ * num_directions);
            auto x_t = x_tensor.slice(0, t, t + 1).reshape({batch_size, layer_feat_size});
            Variable x_t_var(x_t, layer_input.requires_grad());

            forward_h = forward_cell->forward(x_t_var, forward_h);
            forward_outputs.push_back(forward_h);
        }

        final_h_states.push_back(forward_h);

        Variable layer_output = layer_input;

        if (bidirectional_) {
            auto& backward_cell = backward_cells_[layer];
            Variable backward_h = h_layers[layer * num_directions + 1];

            std::vector<Variable> backward_outputs;

            // Backward pass
            for (int64_t t = seq_len - 1; t >= 0; --t) {
                auto x_tensor = layer_input.tensor();
                int64_t layer_feat_size = (layer == 0) ? input_size_ : (hidden_size_ * 2);
                auto x_t = x_tensor.slice(0, t, t + 1).reshape({batch_size, layer_feat_size});
                Variable x_t_var(x_t, layer_input.requires_grad());

                backward_h = backward_cell->forward(x_t_var, backward_h);
                backward_outputs.push_back(backward_h);
            }

            std::reverse(backward_outputs.begin(), backward_outputs.end());
            final_h_states.push_back(backward_h);

            // Concatenate forward and backward
            std::vector<Tensor> output_tensors;
            for (int64_t t = 0; t < seq_len; ++t) {
                std::vector<Tensor> tensors_to_concat = {forward_outputs[t].tensor(), backward_outputs[t].tensor()};
                auto concatenated = cat(tensors_to_concat, 1);
                output_tensors.push_back(concatenated);
            }
            layer_output = Variable(stack(output_tensors, 0), true);
        } else {
            std::vector<Tensor> output_tensors;
            for (const auto& out : forward_outputs) {
                output_tensors.push_back(out.tensor());
            }
            layer_output = Variable(stack(output_tensors, 0), true);
        }

        // Apply dropout
        if (dropout_ && layer < num_layers_ - 1) {
            layer_output = dropout_->forward(layer_output);
        }

        layer_input = layer_output;
    }

    Variable output = layer_input;

    if (batch_first_) {
        output = Variable(output.tensor().transpose(0, 1), output.requires_grad());
    }

    // Stack final states
    // Each hidden state is (batch, hidden_size), stack to (num_layers * num_directions, batch, hidden_size)
    std::vector<Tensor> h_final_tensors;
    for (const auto& h_state : final_h_states) {
        h_final_tensors.push_back(h_state.tensor());
    }

    Variable h_final(stack(std::span<const Tensor>(h_final_tensors), 0), false);

    return {output, h_final};
}

} // namespace tenzor::nn
