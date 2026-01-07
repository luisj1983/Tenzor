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
// GRUCell Implementation
// ============================================================================

GRUCell::GRUCell(int64_t input_size, int64_t hidden_size, bool bias)
    : input_size_(input_size), hidden_size_(hidden_size) {

    // GRU has 3 gates (reset, update, new)
    // PyTorch-style GRU with combined weight matrices for efficiency:
    // weight_ih: (3 * hidden_size, input_size) - maps input to all 3 gates
    // weight_hh: (3 * hidden_size, hidden_size) - maps hidden to all 3 gates
    // Gate order: [reset | update | new]

    // Input-to-hidden transformation for all 3 gates (6 -> 2 linear layers)
    weight_ih_ = std::make_shared<Linear>(input_size, 3 * hidden_size, bias);
    register_module("weight_ih", weight_ih_);

    // Hidden-to-hidden transformation for all 3 gates
    weight_hh_ = std::make_shared<Linear>(hidden_size, 3 * hidden_size, bias);
    register_module("weight_hh", weight_hh_);
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

    // Ensure module parameters are on the same device as input
    auto input_device = input.device();
    weight_ih_->to(input_device);
    weight_hh_->to(input_device);

    // Initialize hidden state if not provided
    Variable h = hx;
    if (!h.is_initialized() || h.tensor().numel() == 0) {
        h = Variable(zeros({batch_size, hidden_size_},
                          input.dtype(), input.device()), false);
    }

    // Validate hidden state shape
    auto h_shape = h.shape();
    if (h_shape.size() != 2 || h_shape[0] != batch_size || h_shape[1] != hidden_size_) {
        throw std::runtime_error("GRUCell: invalid hidden state shape");
    }

    // Compute combined gates efficiently (2 linear calls instead of 6)
    // gates_ih: (batch, 3 * hidden_size) = [r_i | z_i | n_i]
    // gates_hh: (batch, 3 * hidden_size) = [r_h | z_h | n_h]
    auto gates_ih = weight_ih_->forward(input);    // (batch, 3*hidden_size)
    auto gates_hh = weight_hh_->forward(h);        // (batch, 3*hidden_size)

    auto gates_ih_t = gates_ih.tensor().contiguous();
    auto gates_hh_t = gates_hh.tensor().contiguous();

    // Split gates into 3 chunks
    auto ih_chunks = chunk(gates_ih_t, 3, 1);  // [r_i, z_i, n_i]
    auto hh_chunks = chunk(gates_hh_t, 3, 1);  // [r_h, z_h, n_h]

    auto r_i = ih_chunks[0];
    auto z_i = ih_chunks[1];
    auto n_i = ih_chunks[2];
    auto r_h = hh_chunks[0];
    auto z_h = hh_chunks[1];
    auto n_h = hh_chunks[2];

    // Compute reset gate: r_t = σ(r_i + r_h)
    auto r_t = nn::sigmoid(Variable(r_i + r_h, true));

    // Compute update gate: z_t = σ(z_i + z_h)
    auto z_t = nn::sigmoid(Variable(z_i + z_h, true));

    // Compute new gate: n_t = tanh(n_i + r_t ⊙ n_h)
    auto n_h_reset = r_t.tensor() * n_h;
    auto n_t = nn::tanh(Variable(n_i + n_h_reset, true));

    // Compute new hidden state: h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
    auto z_tensor = z_t.tensor();
    auto n_tensor = n_t.tensor();
    auto h_tensor = h.tensor();

    // h_new = z_t * h + (1 - z_t) * n_t = z_t * h + n_t - z_t * n_t
    //       = n_t + z_t * (h - n_t)
    auto h_minus_n = h_tensor - n_tensor;
    auto h_new_tensor = n_tensor + z_tensor * h_minus_n;

    return Variable(h_new_tensor, true);
}

auto GRUCell::forward_with_precomputed_ih(const Tensor& gates_ih, const Variable& hx) -> Variable {
    int64_t batch_size = gates_ih.shape()[0];

    // Initialize hidden state if not provided
    Variable h = hx;
    if (!h.is_initialized() || h.tensor().numel() == 0) {
        h = Variable(zeros({batch_size, hidden_size_},
                          gates_ih.dtype(), gates_ih.device()), false);
    }

    // Compute hidden-to-hidden gates
    auto gates_hh = weight_hh_->forward(h);
    auto gates_hh_t = gates_hh.tensor().contiguous();

    // Split gates into 3 chunks
    auto ih_chunks = chunk(gates_ih.contiguous(), 3, 1);  // [r_i, z_i, n_i]
    auto hh_chunks = chunk(gates_hh_t, 3, 1);             // [r_h, z_h, n_h]

    auto r_i = ih_chunks[0];
    auto z_i = ih_chunks[1];
    auto n_i = ih_chunks[2];
    auto r_h = hh_chunks[0];
    auto z_h = hh_chunks[1];
    auto n_h = hh_chunks[2];

    // Compute reset gate: r_t = σ(r_i + r_h)
    auto r_t = nn::sigmoid(Variable(r_i + r_h, true));

    // Compute update gate: z_t = σ(z_i + z_h)
    auto z_t = nn::sigmoid(Variable(z_i + z_h, true));

    // Compute new gate: n_t = tanh(n_i + r_t ⊙ n_h)
    auto n_h_reset = r_t.tensor() * n_h;
    auto n_t = nn::tanh(Variable(n_i + n_h_reset, true));

    // Compute new hidden state: h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
    auto z_tensor = z_t.tensor();
    auto n_tensor = n_t.tensor();
    auto h_tensor = h.tensor();

    // h_new = n_t + z_t * (h - n_t)
    auto h_minus_n = h_tensor - n_tensor;
    auto h_new_tensor = n_tensor + z_tensor * h_minus_n;

    return Variable(h_new_tensor, true);
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
    if (!h.is_initialized() || h.tensor().numel() == 0) {
        h = Variable(zeros({num_layers_ * num_directions, batch_size, hidden_size_},
                          input.dtype(), input.device()), false);
    }

    // Validate shape
    auto h_shape = h.shape();
    if (h_shape.size() != 3 || h_shape[0] != num_layers_ * num_directions ||
        h_shape[1] != batch_size || h_shape[2] != hidden_size_) {
        throw std::runtime_error("GRU: invalid hidden state shape");
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

        // Get weight tensors (3*hidden, input_size) and (3*hidden, hidden_size)
        auto W_ih_params = W_ih_linear->parameters();
        Tensor W_ih_tensor = W_ih_params[0]->tensor().contiguous();

        // Get W_hh from cell's registered modules
        auto cell_params = cell->parameters();
        Tensor W_hh_tensor = cell_params[2]->tensor().contiguous();

        // Get bias (3*hidden) or empty tensor
        Tensor bias_tensor;
        if (W_ih_params.size() > 1) {
            bias_tensor = W_ih_params[1]->tensor().contiguous();
        } else {
            bias_tensor = empty({0}, DType::Float32, input.device());
        }

        // Make input contiguous (seq, batch, input_size)
        Tensor input_tensor = x.tensor().contiguous();

        // Get initial state (batch, hidden)
        Tensor h0_tensor = h.tensor().slice(0, 0, 1).reshape({batch_size, hidden_size_}).contiguous();

        // Call fused kernel via dispatch
        // inputs: [input, W_ih, W_hh, bias, h0]
        std::vector<Tensor> inputs = {input_tensor, W_ih_tensor, W_hh_tensor,
                                       bias_tensor, h0_tensor};
        auto outputs = dispatch<OpId::GRUForward>(inputs);
        // outputs: [output, h_n]

        // Reshape output
        Variable output(outputs[0], false);
        if (batch_first_) {
            output = Variable(output.tensor().transpose(0, 1), false);
        }

        // Reshape final state to (1, batch, hidden)
        Variable h_final(outputs[1].reshape({1, batch_size, hidden_size_}), false);

        return {output, h_final};
    }

    // =========================================================================
    // STANDARD PATH: Use autograd operations (required for training/gradients)
    // =========================================================================

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

        int64_t layer_feat_size = (layer == 0) ? input_size_ : (hidden_size_ * num_directions);

        // OPTIMIZATION: Pre-compute all input-to-hidden gates at once
        // Instead of calling weight_ih->forward() seq_len times, we do it ONCE
        // Reshape from (seq, batch, feat) to (seq*batch, feat)
        auto x_tensor = layer_input.tensor().contiguous();
        auto x_flat = x_tensor.reshape({seq_len * batch_size, layer_feat_size});

        // Compute all input gates at once: (seq*batch, 3*hidden)
        auto all_gates_ih = forward_cell->weight_ih()->forward(Variable(x_flat, false));

        // Reshape to (seq, batch, 3*hidden)
        auto gates_ih_tensor = all_gates_ih.tensor().reshape({seq_len, batch_size, 3 * hidden_size_});

        std::vector<Variable> forward_outputs;

        // Forward pass with pre-computed input gates
        for (int64_t t = 0; t < seq_len; ++t) {
            // Extract pre-computed input gates for this timestep
            auto gates_ih_t = gates_ih_tensor.slice(0, t, t + 1).reshape({batch_size, 3 * hidden_size_});

            forward_h = forward_cell->forward_with_precomputed_ih(gates_ih_t, forward_h);
            forward_outputs.push_back(forward_h);
        }

        final_h_states.push_back(forward_h);

        Variable layer_output = layer_input;

        if (bidirectional_) {
            auto& backward_cell = backward_cells_[layer];
            Variable backward_h = h_layers[layer * num_directions + 1];

            // Pre-compute all backward input gates
            auto all_gates_ih_bwd = backward_cell->weight_ih()->forward(Variable(x_flat, false));
            auto gates_ih_bwd_tensor = all_gates_ih_bwd.tensor().reshape({seq_len, batch_size, 3 * hidden_size_});

            std::vector<Variable> backward_outputs;

            // Backward pass with pre-computed input gates
            for (int64_t t = seq_len - 1; t >= 0; --t) {
                auto gates_ih_t = gates_ih_bwd_tensor.slice(0, t, t + 1).reshape({batch_size, 3 * hidden_size_});

                backward_h = backward_cell->forward_with_precomputed_ih(gates_ih_t, backward_h);
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
