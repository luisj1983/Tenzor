#include "tenzor/nn/layers/rnn.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include <stdexcept>
#include <sstream>

namespace tenzor::nn {

// ============================================================================
// RNNCell Implementation
// ============================================================================

namespace {

// Apply the named activation to a Variable. Recognises the full ONNX RNN
// activation alphabet (https://onnx.ai/onnx/operators/onnx__RNN.html) so the
// importer can pass `sigmoid`, `leaky_relu`, `elu`, etc. through without
// rewriting the recurrence per-cell.
auto apply_rnn_activation(const std::string& name, const Variable& x) -> Variable {
    if (name == "tanh")          return nn::tanh(x);
    if (name == "relu")          return nn::relu(x);
    if (name == "sigmoid")       return nn::sigmoid(x);
    if (name == "leaky_relu")    return nn::leaky_relu(x, /*negative_slope=*/0.01);
    if (name == "elu")           return nn::elu(x, /*alpha=*/1.0);
    if (name == "hardsigmoid" || name == "hard_sigmoid") return nn::hardsigmoid(x);
    if (name == "hardtanh"    || name == "hard_tanh")    return nn::hardtanh(x);
    if (name == "softsign")      return nn::softsign(x);
    if (name == "affine" || name == "identity" || name == "linear") return x;
    if (name == "scaledtanh" || name == "scaled_tanh")
        // LeCun scaled tanh: 1.7159 * tanh((2/3) * x), the canonical "scaled
        // tanh" (LeCun 1998) chosen so the output has ~unit variance for
        // normalized inputs. Previously this silently returned plain tanh(x),
        // dropping the characteristic scaling entirely.
        return nn::tanh(x * (2.0f / 3.0f)) * 1.7159f;
    throw std::invalid_argument(
        "RNNCell: unsupported activation '" + name + "'. Supported: tanh, relu, "
        "sigmoid, leaky_relu, elu, hardsigmoid, hardtanh, softsign, affine, "
        "scaledtanh.");
}

bool is_known_rnn_activation(const std::string& name) {
    return name == "tanh" || name == "relu" || name == "sigmoid" ||
           name == "leaky_relu" || name == "elu" || name == "hardsigmoid" ||
           name == "hard_sigmoid" || name == "hardtanh" || name == "hard_tanh" ||
           name == "softsign" || name == "affine" || name == "identity" ||
           name == "linear" || name == "scaledtanh" || name == "scaled_tanh";
}

}  // namespace

RNNCell::RNNCell(int64_t input_size, int64_t hidden_size,
                 const std::string& nonlinearity, bool bias)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      nonlinearity_(nonlinearity) {

    if (!is_known_rnn_activation(nonlinearity)) {
        throw std::invalid_argument(
            "RNNCell: unsupported nonlinearity '" + nonlinearity +
            "'. See apply_rnn_activation in src/nn/layers/rnn.cpp for the "
            "supported set.");
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

    // For Float16/BFloat16, run the whole recurrence (gate matmuls + activation)
    // in Float32, then cast the new hidden state back to the original dtype.
    // The per-timestep cell math (sigmoid/tanh/state update) loses precision in
    // half and some elementwise kernels lack half dispatch. Widening input and
    // hidden to Float32 makes the Linear layers cast their weights to Float32 to
    // match (Linear uses input.dtype() as the compute dtype), so the entire
    // recurrence executes at Float32. Mirrors the attention half-widening.
    const DType orig_dtype = input.dtype();
    const bool needs_upcast =
        (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Variable x_c = needs_upcast ? variable_cast(input, DType::Float32) : input;
    Variable h_c = needs_upcast ? variable_cast(h, DType::Float32) : h;

    // Compute: h_new = activation(W_ih @ x + W_hh @ h + b)
    auto i_out = input_layer_->forward(x_c);
    auto h_out = hidden_layer_->forward(h_c);
    auto combined = i_out + h_out;

    // Apply activation (supports the full ONNX RNN activation set).
    auto h_new = apply_rnn_activation(nonlinearity_, combined);
    return needs_upcast ? variable_cast(h_new, orig_dtype) : h_new;
}

auto RNNCell::forward_with_precomputed_ih(const Tensor& precomputed_ih, const Variable& hx) -> Variable {
    // `precomputed_ih` is a raw Tensor: the input-gate contribution (W_ih @ input
    // + b_ih) computed by the caller. The caller has already decided not to
    // retain autograd through the input side — this signature takes Tensor, not
    // Variable. The recurrent / hidden-side grad chain through `hx` and
    // `hidden_layer_`'s weights must still be preserved, though.
    //
    // Previously: `Variable(precomputed_ih + h_out.tensor(), true)` did a raw
    // Tensor add and wrapped in a fresh Variable with no grad_fn — silently
    // severing the hidden-side backward path. Rewritten to use Variable-level
    // addition so the grad_fn chain is intact.
    int64_t batch_size = precomputed_ih.shape()[0];

    Variable h = hx;
    if (!h.is_initialized() || h.tensor().numel() == 0) {
        h = Variable(zeros({batch_size, hidden_size_},
                          precomputed_ih.dtype(), precomputed_ih.device()), false);
    }

    auto h_out = hidden_layer_->forward(h);  // Variable with grad_fn

    // Wrap the precomputed input-side contribution as a non-autograd constant;
    // Variable+Variable addition keeps h_out's grad_fn on the result.
    Variable precomputed_ih_var(precomputed_ih, false);
    auto combined = precomputed_ih_var + h_out;

    return apply_rnn_activation(nonlinearity_, combined);
}

// ============================================================================
// RNN Implementation
// ============================================================================

RNN::RNN(int64_t input_size, int64_t hidden_size, int64_t num_layers,
         const std::string& nonlinearity, bool bias, bool batch_first,
         double dropout, bool bidirectional,
         const std::string& nonlinearity_bwd)
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

    // Backward activation defaults to the forward activation. ONNX RNN
    // exports can specify different activations per direction; this lets
    // the importer pass both without a wrapper module.
    const std::string& backward_nonlin =
        nonlinearity_bwd.empty() ? nonlinearity : nonlinearity_bwd;

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
            auto cell = std::make_shared<RNNCell>(layer_input_size, hidden_size,
                                                  backward_nonlin, bias);
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
        // Transpose to (seq_len, batch, input_size) using autograd-aware op
        x = ::tenzor::transpose(x, 0, 1);
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

    // =========================================================================
    // STANDARD PATH: Autograd-correct per-timestep forward.
    // =========================================================================
    // This path keeps every operation on the Variable/autograd graph so that
    // backward() can propagate gradients all the way back to `input` (and
    // through each cell's parameters), enabling higher-order gradients.
    //
    // The previous implementation pre-computed input gates via
    // `Variable(x_flat, false)` and called `forward_with_precomputed_ih`,
    // which silently detached the input from the graph and produced empty
    // input gradients.
    //
    // We instead:
    //   1. Slice the h0 per-layer states with autograd-aware `slice` +
    //      `squeeze` so intermediate states carry grad history.
    //   2. For each timestep, slice `layer_input` at t along dim=0, then
    //      call the cell's full `forward(input_var, h_var)`.
    //   3. Accumulate outputs via autograd-aware `unsqueeze` + `cat`.

    // Split hidden state by layer and direction using autograd-aware ops
    auto split_states_per_layer = [&](const Variable& stacked_states,
                                      std::vector<Variable>& out) {
        for (int64_t i = 0; i < num_layers_ * num_directions; ++i) {
            auto sliced = ::tenzor::slice(stacked_states, 0, i, i + 1);
            auto sq = ::tenzor::squeeze(sliced, 0);  // (batch, hidden)
            out.push_back(sq);
        }
    };

    std::vector<Variable> h_layers;
    split_states_per_layer(h, h_layers);

    // Process through layers
    Variable layer_input = x;  // shape: (seq, batch, feat)
    std::vector<Variable> final_hidden_states;

    for (int64_t layer = 0; layer < num_layers_; ++layer) {
        auto& forward_cell = forward_cells_[layer];
        Variable forward_h = h_layers[layer * num_directions];

        std::vector<Variable> forward_outputs;
        forward_outputs.reserve(static_cast<size_t>(seq_len));

        // Per-timestep forward pass through the full cell.
        for (int64_t t = 0; t < seq_len; ++t) {
            auto x_t_raw = ::tenzor::slice(layer_input, 0, t, t + 1);  // (1, batch, feat)
            auto x_t = ::tenzor::squeeze(x_t_raw, 0);                  // (batch, feat)

            forward_h = forward_cell->forward(x_t, forward_h);
            forward_outputs.push_back(forward_h);
        }

        final_hidden_states.push_back(forward_h);

        // Helper to stack timestep outputs along dim 0 using autograd ops
        auto stack_timesteps = [](const std::vector<Variable>& step_outputs) {
            std::vector<Variable> expanded;
            expanded.reserve(step_outputs.size());
            for (const auto& v : step_outputs) {
                expanded.push_back(::tenzor::unsqueeze(v, 0));
            }
            return ::tenzor::cat(expanded, 0);
        };

        Variable layer_output;
        if (bidirectional_) {
            auto& backward_cell = backward_cells_[layer];
            Variable backward_h = h_layers[layer * num_directions + 1];

            std::vector<Variable> backward_outputs;
            backward_outputs.reserve(static_cast<size_t>(seq_len));

            for (int64_t t = seq_len - 1; t >= 0; --t) {
                auto x_t_raw = ::tenzor::slice(layer_input, 0, t, t + 1);
                auto x_t = ::tenzor::squeeze(x_t_raw, 0);

                backward_h = backward_cell->forward(x_t, backward_h);
                backward_outputs.push_back(backward_h);
            }

            // Reverse backward outputs to align with forward time order
            std::reverse(backward_outputs.begin(), backward_outputs.end());
            final_hidden_states.push_back(backward_h);

            // Concatenate forward and backward outputs per timestep
            std::vector<Variable> concat_per_t;
            concat_per_t.reserve(static_cast<size_t>(seq_len));
            for (int64_t t = 0; t < seq_len; ++t) {
                std::vector<Variable> pair = {forward_outputs[t], backward_outputs[t]};
                concat_per_t.push_back(::tenzor::cat(pair, 1));
            }
            layer_output = stack_timesteps(concat_per_t);
        } else {
            layer_output = stack_timesteps(forward_outputs);
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
        output = ::tenzor::transpose(output, 0, 1);
    }

    // Stack final hidden states using autograd-aware ops
    auto stack_layer_states = [](const std::vector<Variable>& states) {
        std::vector<Variable> expanded;
        expanded.reserve(states.size());
        for (const auto& s : states) {
            expanded.push_back(::tenzor::unsqueeze(s, 0));
        }
        return ::tenzor::cat(expanded, 0);
    };

    Variable h_final = stack_layer_states(final_hidden_states);

    return {output, h_final};
}

} // namespace tenzor::nn
