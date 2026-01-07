#include "tenzor/nn/layers/rnn.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include <stdexcept>
#include <sstream>

namespace tenzor::nn {

// ============================================================================
// LSTMCell Implementation
// ============================================================================

LSTMCell::LSTMCell(int64_t input_size, int64_t hidden_size, bool bias)
    : input_size_(input_size), hidden_size_(hidden_size) {

    // PyTorch-style LSTM with separate weight matrices for input and hidden
    // All 4 gates are combined into single weight matrices for efficiency
    // weight_ih: (4 * hidden_size, input_size) - maps input to all 4 gates
    // weight_hh: (4 * hidden_size, hidden_size) - maps hidden state to all 4 gates

    // Input-to-hidden transformation for all 4 gates
    weight_ih_ = std::make_shared<Linear>(input_size, 4 * hidden_size, bias);
    register_module("weight_ih", weight_ih_);

    // Hidden-to-hidden transformation for all 4 gates
    weight_hh_ = std::make_shared<Linear>(hidden_size, 4 * hidden_size, false);  // No bias for hh
    register_module("weight_hh", weight_hh_);
}

auto LSTMCell::forward(const Variable& input, const Variable& hx, const Variable& cx)
    -> std::pair<Variable, Variable> {

    // Validate input shape
    auto input_shape = input.shape();
    if (input_shape.size() != 2) {
        throw std::runtime_error("LSTMCell: input must be 2D (batch, input_size)");
    }
    if (input_shape[1] != input_size_) {
        std::ostringstream oss;
        oss << "LSTMCell: expected input size " << input_size_
            << " but got " << input_shape[1];
        throw std::runtime_error(oss.str());
    }

    int64_t batch_size = input_shape[0];

    // Ensure module parameters are on the same device as input
    auto input_device = input.device();
    weight_ih_->to(input_device);
    weight_hh_->to(input_device);

    // Initialize hidden and cell states if not provided
    Variable h = hx;
    Variable c = cx;

    if (!h.is_initialized() || h.tensor().numel() == 0) {
        h = Variable(zeros({batch_size, hidden_size_},
                          input.dtype(), input.device()), false);
    }
    if (!c.is_initialized() || c.tensor().numel() == 0) {
        c = Variable(zeros({batch_size, hidden_size_},
                          input.dtype(), input.device()), false);
    }

    // Validate hidden and cell state shapes
    auto h_shape = h.shape();
    auto c_shape = c.shape();
    if (h_shape.size() != 2 || h_shape[0] != batch_size || h_shape[1] != hidden_size_) {
        throw std::runtime_error("LSTMCell: invalid hidden state shape");
    }
    if (c_shape.size() != 2 || c_shape[0] != batch_size || c_shape[1] != hidden_size_) {
        throw std::runtime_error("LSTMCell: invalid cell state shape");
    }

    // Compute combined gates: gates = W_ih @ input + W_hh @ hidden
    // gates shape: (batch, 4 * hidden_size)
    // gates = [input_gate | forget_gate | cell_gate | output_gate]
    auto gates_ih = weight_ih_->forward(input);    // (batch, 4*hidden_size)
    auto gates_hh = weight_hh_->forward(h);         // (batch, 4*hidden_size)

    // Add the two gate tensors and create a new variable
    auto gates_ih_t = gates_ih.tensor();
    auto gates_hh_t = gates_hh.tensor();

    // Debug: Check shapes
    auto gh_shape = gates_ih_t.shape();
    auto ghh_shape = gates_hh_t.shape();
    if (gh_shape.size() != 2 || gh_shape[0] != batch_size || gh_shape[1] != 4 * hidden_size_) {
        std::ostringstream oss;
        oss << "gates_ih has wrong shape: (" << gh_shape[0] << ", " << gh_shape[1]
            << "), expected (" << batch_size << ", " << (4 * hidden_size_) << ")";
        throw std::runtime_error(oss.str());
    }
    if (ghh_shape.size() != 2 || ghh_shape[0] != batch_size || ghh_shape[1] != 4 * hidden_size_) {
        std::ostringstream oss;
        oss << "gates_hh has wrong shape: (" << ghh_shape[0] << ", " << ghh_shape[1]
            << "), expected (" << batch_size << ", " << (4 * hidden_size_) << ")";
        throw std::runtime_error(oss.str());
    }

    auto gates = Variable(gates_ih_t + gates_hh_t, true);
    auto gates_tensor = gates.tensor().contiguous();  // (batch, 4*hidden_size)

    // Debug: Check gates_tensor shape
    auto gt_shape = gates_tensor.shape();
    if (gt_shape.size() != 2 || gt_shape[0] != batch_size || gt_shape[1] != 4 * hidden_size_) {
        std::ostringstream oss;
        oss << "gates_tensor has wrong shape: (" << gt_shape[0] << ", " << gt_shape[1]
            << "), expected (" << batch_size << ", " << (4 * hidden_size_) << ")";
        throw std::runtime_error(oss.str());
    }

    // Split gates into 4 chunks along dimension 1
    // PyTorch uses chunk(4, 1) which splits evenly into 4 pieces
    auto gate_chunks = chunk(gates_tensor, 4, 1);
    if (gate_chunks.size() != 4) {
        throw std::runtime_error("Expected 4 gate chunks");
    }

    auto i_gate_tensor = gate_chunks[0];  // Input gate
    auto f_gate_tensor = gate_chunks[1];  // Forget gate
    auto g_gate_tensor = gate_chunks[2];  // Cell gate
    auto o_gate_tensor = gate_chunks[3];  // Output gate

    // Apply activations
    auto i_t = nn::sigmoid(Variable(i_gate_tensor, true));
    auto f_t = nn::sigmoid(Variable(f_gate_tensor, true));
    auto g_t = nn::tanh(Variable(g_gate_tensor, true));
    auto o_t = nn::sigmoid(Variable(o_gate_tensor, true));

    // Debug: Check activation output shapes
    auto it_shape = i_t.shape();
    auto ft_shape = f_t.shape();
    if (it_shape.size() != 2 || it_shape[0] != batch_size || it_shape[1] != hidden_size_) {
        std::ostringstream oss;
        oss << "i_t has wrong shape: (" << it_shape[0] << ", " << it_shape[1]
            << "), expected (" << batch_size << ", " << hidden_size_ << ")";
        throw std::runtime_error(oss.str());
    }

    // Update cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    auto f_c = Variable(f_t.tensor() * c.tensor(), true);
    auto i_g = Variable(i_t.tensor() * g_t.tensor(), true);
    auto c_new = Variable(f_c.tensor() + i_g.tensor(), true);

    // Update hidden state: h_t = o_t ⊙ tanh(c_t)
    auto c_tanh = nn::tanh(c_new);
    auto h_new = Variable(o_t.tensor() * c_tanh.tensor(), true);

    return {h_new, c_new};
}

auto LSTMCell::forward_with_precomputed_ih(const Tensor& gates_ih, const Variable& hx, const Variable& cx)
    -> std::pair<Variable, Variable> {

    int64_t batch_size = gates_ih.shape()[0];

    // Initialize hidden and cell states if not provided
    Variable h = hx;
    Variable c = cx;

    if (!h.is_initialized() || h.tensor().numel() == 0) {
        h = Variable(zeros({batch_size, hidden_size_},
                          gates_ih.dtype(), gates_ih.device()), false);
    }
    if (!c.is_initialized() || c.tensor().numel() == 0) {
        c = Variable(zeros({batch_size, hidden_size_},
                          gates_ih.dtype(), gates_ih.device()), false);
    }

    // Compute hidden-to-hidden gates
    auto gates_hh = weight_hh_->forward(h);

    // Combine gates: gates = gates_ih + gates_hh
    auto gates_tensor = (gates_ih + gates_hh.tensor()).contiguous();

    // Split gates into 4 chunks along dimension 1
    auto gate_chunks = chunk(gates_tensor, 4, 1);

    auto i_gate_tensor = gate_chunks[0];  // Input gate
    auto f_gate_tensor = gate_chunks[1];  // Forget gate
    auto g_gate_tensor = gate_chunks[2];  // Cell gate
    auto o_gate_tensor = gate_chunks[3];  // Output gate

    // Apply activations
    auto i_t = nn::sigmoid(Variable(i_gate_tensor, true));
    auto f_t = nn::sigmoid(Variable(f_gate_tensor, true));
    auto g_t = nn::tanh(Variable(g_gate_tensor, true));
    auto o_t = nn::sigmoid(Variable(o_gate_tensor, true));

    // Update cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    auto f_c = Variable(f_t.tensor() * c.tensor(), true);
    auto i_g = Variable(i_t.tensor() * g_t.tensor(), true);
    auto c_new = Variable(f_c.tensor() + i_g.tensor(), true);

    // Update hidden state: h_t = o_t ⊙ tanh(c_t)
    auto c_tanh = nn::tanh(c_new);
    auto h_new = Variable(o_t.tensor() * c_tanh.tensor(), true);

    return {h_new, c_new};
}

// ============================================================================
// LSTM Implementation
// ============================================================================

LSTM::LSTM(int64_t input_size, int64_t hidden_size, int64_t num_layers,
           bool bias, bool batch_first, double dropout, bool bidirectional,
           int64_t proj_size)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      num_layers_(num_layers),
      batch_first_(batch_first),
      bidirectional_(bidirectional),
      dropout_p_(dropout),
      proj_size_(proj_size) {

    if (num_layers < 1) {
        throw std::invalid_argument("LSTM: num_layers must be >= 1");
    }
    if (dropout < 0.0 || dropout > 1.0) {
        throw std::invalid_argument("LSTM: dropout must be in [0, 1]");
    }
    if (proj_size < 0) {
        throw std::invalid_argument("LSTM: proj_size must be >= 0");
    }
    if (proj_size > 0) {
        throw std::runtime_error("LSTM: projection not yet implemented");
    }

    // Create forward cells for each layer
    for (int64_t i = 0; i < num_layers; ++i) {
        int64_t layer_input_size = (i == 0) ? input_size : hidden_size * (bidirectional_ ? 2 : 1);
        auto cell = std::make_shared<LSTMCell>(layer_input_size, hidden_size, bias);
        forward_cells_.push_back(cell);
        register_module("forward_cell_" + std::to_string(i), cell);
    }

    // Create backward cells for bidirectional LSTM
    if (bidirectional_) {
        for (int64_t i = 0; i < num_layers; ++i) {
            int64_t layer_input_size = (i == 0) ? input_size : hidden_size * 2;
            auto cell = std::make_shared<LSTMCell>(layer_input_size, hidden_size, bias);
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

auto LSTM::forward(const Variable& input, const std::pair<Variable, Variable>& hx)
    -> std::pair<Variable, std::pair<Variable, Variable>> {

    auto input_shape = input.shape();

    if (input_shape.size() != 3) {
        throw std::runtime_error("LSTM: input must be 3D");
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
        throw std::runtime_error("LSTM: input feature size mismatch");
    }

    int64_t num_directions = bidirectional_ ? 2 : 1;

    // Initialize hidden and cell states
    auto [h0, c0] = hx;
    Variable h = h0;
    Variable c = c0;

    if (!h.is_initialized() || h.tensor().numel() == 0) {
        h = Variable(zeros({num_layers_ * num_directions, batch_size, hidden_size_},
                          input.dtype(), input.device()), false);
    }
    if (!c.is_initialized() || c.tensor().numel() == 0) {
        c = Variable(zeros({num_layers_ * num_directions, batch_size, hidden_size_},
                          input.dtype(), input.device()), false);
    }

    // Validate shapes
    auto h_shape = h.shape();
    auto c_shape = c.shape();
    if (h_shape.size() != 3 || h_shape[0] != num_layers_ * num_directions ||
        h_shape[1] != batch_size || h_shape[2] != hidden_size_) {
        throw std::runtime_error("LSTM: invalid hidden state shape");
    }
    if (c_shape.size() != 3 || c_shape[0] != num_layers_ * num_directions ||
        c_shape[1] != batch_size || c_shape[2] != hidden_size_) {
        throw std::runtime_error("LSTM: invalid cell state shape");
    }

    // =========================================================================
    // FAST PATH: Use fused CPU backend kernel for inference
    // Conditions: CPU, Float32, not training, single layer, not bidirectional
    // =========================================================================
    bool can_use_fused = input.device().type == Device::Type::CPU &&
                         input.dtype() == DType::Float32 &&
                         !is_training() &&
                         num_layers_ == 1 &&
                         !bidirectional_;

    if (can_use_fused) {
        // Get weight matrices from the cell
        auto& cell = forward_cells_[0];
        auto W_ih_linear = cell->weight_ih();
        auto W_hh_var = cell->parameters()[2];  // weight_hh weight

        // Get weight tensors (4*hidden, input_size) and (4*hidden, hidden_size)
        auto W_ih_params = W_ih_linear->parameters();
        Tensor W_ih_tensor = W_ih_params[0]->tensor().contiguous();
        Tensor W_hh_tensor = W_hh_var->tensor().contiguous();

        // Get bias (4*hidden) or empty tensor
        Tensor bias_tensor;
        if (W_ih_params.size() > 1) {
            bias_tensor = W_ih_params[1]->tensor().contiguous();
        } else {
            bias_tensor = empty({0}, DType::Float32, input.device());
        }

        // Make input contiguous (seq, batch, input_size)
        Tensor input_tensor = x.tensor().contiguous();

        // Get initial states (batch, hidden)
        Tensor h0_tensor = h.tensor().slice(0, 0, 1).reshape({batch_size, hidden_size_}).contiguous();
        Tensor c0_tensor = c.tensor().slice(0, 0, 1).reshape({batch_size, hidden_size_}).contiguous();

        // Call fused kernel via dispatch
        // inputs: [input, W_ih, W_hh, bias, h0, c0]
        std::vector<Tensor> inputs = {input_tensor, W_ih_tensor, W_hh_tensor,
                                       bias_tensor, h0_tensor, c0_tensor};
        auto outputs = dispatch<OpId::LSTMForward>(inputs);
        // outputs: [output, h_n, c_n]

        // Reshape output
        Variable output(outputs[0], false);
        if (batch_first_) {
            output = Variable(output.tensor().transpose(0, 1), false);
        }

        // Reshape final states to (1, batch, hidden)
        Variable h_final(outputs[1].reshape({1, batch_size, hidden_size_}), false);
        Variable c_final(outputs[2].reshape({1, batch_size, hidden_size_}), false);

        return {output, {h_final, c_final}};
    }

    // =========================================================================
    // STANDARD PATH: Use autograd operations (required for training/gradients)
    // =========================================================================

    // Split states by layer
    std::vector<Variable> h_layers;
    std::vector<Variable> c_layers;

    for (int64_t i = 0; i < num_layers_ * num_directions; ++i) {
        auto h_tensor = h.tensor();
        auto c_tensor = c.tensor();

        auto h_layer = h_tensor.slice(0, i, i + 1).reshape({batch_size, hidden_size_});
        auto c_layer = c_tensor.slice(0, i, i + 1).reshape({batch_size, hidden_size_});

        h_layers.push_back(Variable(h_layer, false));
        c_layers.push_back(Variable(c_layer, false));
    }

    // Process through layers
    Variable layer_input = x;
    std::vector<Variable> final_h_states;
    std::vector<Variable> final_c_states;

    for (int64_t layer = 0; layer < num_layers_; ++layer) {
        auto& forward_cell = forward_cells_[layer];
        Variable forward_h = h_layers[layer * num_directions];
        Variable forward_c = c_layers[layer * num_directions];

        int64_t layer_feat_size = (layer == 0) ? input_size_ : (hidden_size_ * num_directions);

        // OPTIMIZATION: Pre-compute all input-to-hidden gates at once
        // Instead of calling weight_ih->forward() seq_len times, we do it ONCE
        // Reshape from (seq, batch, feat) to (seq*batch, feat)
        auto x_tensor = layer_input.tensor().contiguous();
        auto x_flat = x_tensor.reshape({seq_len * batch_size, layer_feat_size});

        // Compute all input gates at once: (seq*batch, 4*hidden)
        auto all_gates_ih = forward_cell->weight_ih()->forward(Variable(x_flat, false));

        // Reshape to (seq, batch, 4*hidden)
        auto gates_ih_tensor = all_gates_ih.tensor().reshape({seq_len, batch_size, 4 * hidden_size_});

        std::vector<Variable> forward_outputs;

        // Forward pass with pre-computed input gates
        for (int64_t t = 0; t < seq_len; ++t) {
            // Extract pre-computed input gates for this timestep
            auto gates_ih_t = gates_ih_tensor.slice(0, t, t + 1).reshape({batch_size, 4 * hidden_size_});

            auto [h_next, c_next] = forward_cell->forward_with_precomputed_ih(gates_ih_t, forward_h, forward_c);
            forward_h = h_next;
            forward_c = c_next;
            forward_outputs.push_back(h_next);
        }

        final_h_states.push_back(forward_h);
        final_c_states.push_back(forward_c);

        Variable layer_output = layer_input;

        if (bidirectional_) {
            auto& backward_cell = backward_cells_[layer];
            Variable backward_h = h_layers[layer * num_directions + 1];
            Variable backward_c = c_layers[layer * num_directions + 1];

            // Pre-compute all backward input gates
            auto all_gates_ih_bwd = backward_cell->weight_ih()->forward(Variable(x_flat, false));
            auto gates_ih_bwd_tensor = all_gates_ih_bwd.tensor().reshape({seq_len, batch_size, 4 * hidden_size_});

            std::vector<Variable> backward_outputs;

            // Backward pass with pre-computed input gates
            for (int64_t t = seq_len - 1; t >= 0; --t) {
                auto gates_ih_t = gates_ih_bwd_tensor.slice(0, t, t + 1).reshape({batch_size, 4 * hidden_size_});

                auto [h_next, c_next] = backward_cell->forward_with_precomputed_ih(gates_ih_t, backward_h, backward_c);
                backward_h = h_next;
                backward_c = c_next;
                backward_outputs.push_back(h_next);
            }

            std::reverse(backward_outputs.begin(), backward_outputs.end());
            final_h_states.push_back(backward_h);
            final_c_states.push_back(backward_c);

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
    // Each state is (batch, hidden_size), stack to (num_layers * num_directions, batch, hidden_size)
    std::vector<Tensor> h_final_tensors, c_final_tensors;
    for (const auto& h_state : final_h_states) {
        h_final_tensors.push_back(h_state.tensor());
    }
    for (const auto& c_state : final_c_states) {
        c_final_tensors.push_back(c_state.tensor());
    }

    Variable h_final(stack(std::span<const Tensor>(h_final_tensors), 0), false);
    Variable c_final(stack(std::span<const Tensor>(c_final_tensors), 0), false);

    return {output, {h_final, c_final}};
}

} // namespace tenzor::nn
