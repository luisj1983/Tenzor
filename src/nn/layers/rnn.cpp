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
// RNNCell Implementation
// ============================================================================

RNNCell::RNNCell(int64_t input_size, int64_t hidden_size,
                 const std::string& nonlinearity, bool bias)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      nonlinearity_(nonlinearity) {

    if (nonlinearity != "tanh" && nonlinearity != "relu") {
        throw std::invalid_argument("RNNCell: nonlinearity must be 'tanh' or 'relu'");
    }

    // Create linear layers for input-to-hidden and hidden-to-hidden transformations
    input_layer_ = std::make_shared<Linear>(input_size, hidden_size, bias);
    hidden_layer_ = std::make_shared<Linear>(hidden_size, hidden_size, bias);

    // Register as submodules
    register_module("input_layer", input_layer_);
    register_module("hidden_layer", hidden_layer_);
}

auto RNNCell::forward(const Variable& input, const Variable& hx) -> Variable {
    // Validate input shape
    auto input_shape = input.shape();
    if (input_shape.size() != 2) {
        throw std::runtime_error("RNNCell: input must be 2D (batch, input_size)");
    }
    if (input_shape[1] != input_size_) {
        std::ostringstream oss;
        oss << "RNNCell: expected input size " << input_size_
            << " but got " << input_shape[1];
        throw std::runtime_error(oss.str());
    }

    int64_t batch_size = input_shape[0];

    // Initialize hidden state if not provided
    Variable h = hx;
    if (!h.is_initialized() || h.tensor().numel() == 0) {
        h = Variable(zeros({batch_size, hidden_size_},
                          input.dtype(), input.device()), false);
    }

    // Validate hidden state shape
    auto h_shape = h.shape();
    if (h_shape.size() != 2 || h_shape[0] != batch_size || h_shape[1] != hidden_size_) {
        throw std::runtime_error("RNNCell: invalid hidden state shape");
    }

    // Compute: h_new = activation(W_ih @ x + W_hh @ h + b)
    auto i_out = input_layer_->forward(input);
    auto h_out = hidden_layer_->forward(h);
    auto combined = i_out + h_out;

    // Apply activation
    if (nonlinearity_ == "tanh") {
        return nn::tanh(combined);
    } else { // relu
        return nn::relu(combined);
    }
}

auto RNNCell::forward_with_precomputed_ih(const Tensor& precomputed_ih, const Variable& hx) -> Variable {
    int64_t batch_size = precomputed_ih.shape()[0];

    // Initialize hidden state if not provided
    Variable h = hx;
    if (!h.is_initialized() || h.tensor().numel() == 0) {
        h = Variable(zeros({batch_size, hidden_size_},
                          precomputed_ih.dtype(), precomputed_ih.device()), false);
    }

    // Compute hidden-to-hidden transformation
    auto h_out = hidden_layer_->forward(h);

    // Combine: precomputed_ih + W_hh @ h
    auto combined = Variable(precomputed_ih + h_out.tensor(), true);

    // Apply activation
    if (nonlinearity_ == "tanh") {
        return nn::tanh(combined);
    } else { // relu
        return nn::relu(combined);
    }
}

// ============================================================================
// RNN Implementation
// ============================================================================

RNN::RNN(int64_t input_size, int64_t hidden_size, int64_t num_layers,
         const std::string& nonlinearity, bool bias, bool batch_first,
         double dropout, bool bidirectional)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      num_layers_(num_layers),
      nonlinearity_(nonlinearity),
      batch_first_(batch_first),
      bidirectional_(bidirectional),
      dropout_p_(dropout) {

    if (num_layers < 1) {
        throw std::invalid_argument("RNN: num_layers must be >= 1");
    }
    if (dropout < 0.0 || dropout > 1.0) {
        throw std::invalid_argument("RNN: dropout must be in [0, 1]");
    }

    // Create forward cells for each layer
    for (int64_t i = 0; i < num_layers; ++i) {
        int64_t layer_input_size = (i == 0) ? input_size : hidden_size * (bidirectional_ ? 2 : 1);
        auto cell = std::make_shared<RNNCell>(layer_input_size, hidden_size, nonlinearity, bias);
        forward_cells_.push_back(cell);
        register_module("forward_cell_" + std::to_string(i), cell);
    }

    // Create backward cells for bidirectional RNN
    if (bidirectional_) {
        for (int64_t i = 0; i < num_layers; ++i) {
            int64_t layer_input_size = (i == 0) ? input_size : hidden_size * 2;
            auto cell = std::make_shared<RNNCell>(layer_input_size, hidden_size, nonlinearity, bias);
            backward_cells_.push_back(cell);
            register_module("backward_cell_" + std::to_string(i), cell);
        }
    }

    // Create dropout layer (applied between layers, not after last layer)
    if (dropout_p_ > 0.0 && num_layers > 1) {
        dropout_ = std::make_shared<Dropout>(dropout_p_);
        register_module("dropout", dropout_);
    }
}

auto RNN::forward(const Variable& input, const Variable& hx) -> std::pair<Variable, Variable> {
    auto input_shape = input.shape();

    if (input_shape.size() != 3) {
        throw std::runtime_error("RNN: input must be 3D (seq_len, batch, input_size) or (batch, seq_len, input_size)");
    }

    // Determine dimensions based on batch_first flag
    int64_t seq_len, batch_size, feat_size;
    Variable x = input;

    if (batch_first_) {
        // Input: (batch, seq_len, input_size)
        batch_size = input_shape[0];
        seq_len = input_shape[1];
        feat_size = input_shape[2];
        // Transpose to (seq_len, batch, input_size)
        x = Variable(x.tensor().transpose(0, 1), x.requires_grad());
    } else {
        // Input: (seq_len, batch, input_size)
        seq_len = input_shape[0];
        batch_size = input_shape[1];
        feat_size = input_shape[2];
    }

    if (feat_size != input_size_) {
        throw std::runtime_error("RNN: input feature size mismatch");
    }

    int64_t num_directions = bidirectional_ ? 2 : 1;

    // Initialize hidden state if not provided
    Variable h = hx;
    if (!h.is_initialized() || h.tensor().numel() == 0) {
        h = Variable(zeros({num_layers_ * num_directions, batch_size, hidden_size_},
                          input.dtype(), input.device()), false);
    }

    // Validate hidden state shape
    auto h_shape = h.shape();
    if (h_shape.size() != 3 ||
        h_shape[0] != num_layers_ * num_directions ||
        h_shape[1] != batch_size ||
        h_shape[2] != hidden_size_) {
        throw std::runtime_error("RNN: invalid hidden state shape");
    }

    // Split hidden state by layer and direction
    std::vector<Variable> h_layers;
    for (int64_t i = 0; i < num_layers_ * num_directions; ++i) {
        // Extract h[i, :, :]
        auto h_tensor = h.tensor();
        auto h_layer_tensor = h_tensor.slice(0, i, i + 1);
        // Squeeze first dimension
        auto h_layer_squeezed = h_layer_tensor.reshape({batch_size, hidden_size_});
        h_layers.push_back(Variable(h_layer_squeezed, false));
    }

    // Process through layers
    Variable layer_input = x;
    std::vector<Variable> final_hidden_states;

    for (int64_t layer = 0; layer < num_layers_; ++layer) {
        int64_t layer_feat_size = (layer == 0) ? input_size_ : (hidden_size_ * num_directions);

        // Forward direction — batch the input-to-hidden matmul across all timesteps
        auto& forward_cell = forward_cells_[layer];
        Variable forward_h = h_layers[layer * num_directions];

        // Flatten (seq_len, batch, features) -> (seq_len * batch, features)
        auto x_flat = layer_input.tensor().reshape({seq_len * batch_size, layer_feat_size});
        // Batch matmul: all_ih = W_ih @ x_flat  -> (seq_len * batch, hidden_size)
        auto all_ih = forward_cell->weight_ih()->forward(Variable(x_flat, false));
        // Reshape back: (seq_len, batch, hidden_size)
        auto ih_3d = all_ih.tensor().reshape({seq_len, batch_size, hidden_size_});

        std::vector<Variable> forward_outputs;
        for (int64_t t = 0; t < seq_len; ++t) {
            auto ih_t = ih_3d.slice(0, t, t + 1).reshape({batch_size, hidden_size_});
            forward_h = forward_cell->forward_with_precomputed_ih(ih_t, forward_h);
            forward_outputs.push_back(forward_h);
        }

        final_hidden_states.push_back(forward_h);

        // Backward direction (if bidirectional) — same batched optimization
        Variable layer_output = layer_input;
        if (bidirectional_) {
            auto& backward_cell = backward_cells_[layer];
            Variable backward_h = h_layers[layer * num_directions + 1];

            // Batch the input-to-hidden matmul for backward direction
            auto all_ih_bwd = backward_cell->weight_ih()->forward(Variable(x_flat, false));
            auto ih_3d_bwd = all_ih_bwd.tensor().reshape({seq_len, batch_size, hidden_size_});

            std::vector<Variable> backward_outputs;
            for (int64_t t = seq_len - 1; t >= 0; --t) {
                auto ih_t = ih_3d_bwd.slice(0, t, t + 1).reshape({batch_size, hidden_size_});
                backward_h = backward_cell->forward_with_precomputed_ih(ih_t, backward_h);
                backward_outputs.push_back(backward_h);
            }

            // Reverse backward outputs
            std::reverse(backward_outputs.begin(), backward_outputs.end());
            final_hidden_states.push_back(backward_h);

            // Concatenate forward and backward outputs
            std::vector<Tensor> output_tensors;
            for (int64_t t = 0; t < seq_len; ++t) {
                auto fwd = forward_outputs[t].tensor();
                auto bwd = backward_outputs[t].tensor();
                std::vector<Tensor> tensors_to_concat = {fwd, bwd};
                auto concatenated = cat(tensors_to_concat, 1);
                output_tensors.push_back(concatenated);
            }

            // Stack along time dimension
            layer_output = Variable(stack(output_tensors, 0), true);
        } else {
            // Stack forward outputs
            std::vector<Tensor> output_tensors;
            for (const auto& out : forward_outputs) {
                output_tensors.push_back(out.tensor());
            }
            layer_output = Variable(stack(output_tensors, 0), true);
        }

        // Apply dropout (except after last layer)
        if (dropout_ && layer < num_layers_ - 1) {
            layer_output = dropout_->forward(layer_output);
        }

        layer_input = layer_output;
    }

    // Prepare final output
    Variable output = layer_input;

    if (batch_first_) {
        // Transpose back to (batch, seq_len, hidden_size * num_directions)
        output = Variable(output.tensor().transpose(0, 1), output.requires_grad());
    }

    // Stack final hidden states
    // Each hidden state is (batch, hidden_size), stack to (num_layers * num_directions, batch, hidden_size)
    std::vector<Tensor> h_final_tensors;
    for (const auto& h_state : final_hidden_states) {
        h_final_tensors.push_back(h_state.tensor());
    }
    Variable h_final = Variable(stack(std::span<const Tensor>(h_final_tensors), 0), false);

    return {output, h_final};
}

} // namespace tenzor::nn
