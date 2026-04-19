#include "tenzor/nn/layers/rnn.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
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
    //
    // IMPORTANT: stay on the autograd-aware Variable path. Extracting
    // .tensor() and re-wrapping with `Variable(t, true)` would create orphan
    // Variables with no grad_fn, so backward() couldn't reach `input`/`hx`.
    auto gates_ih = weight_ih_->forward(input);  // (batch, 3*hidden_size)
    auto gates_hh = weight_hh_->forward(h);      // (batch, 3*hidden_size)

    // Slice each gates tensor into 3 sub-variables (r, z, n) along dim=1.
    // (autograd::chunk does not exist yet — use slice which is autograd-aware.)
    const int64_t H = hidden_size_;
    auto r_i = ::tenzor::slice(gates_ih, 1, 0,     H);
    auto z_i = ::tenzor::slice(gates_ih, 1, H,   2*H);
    auto n_i = ::tenzor::slice(gates_ih, 1, 2*H, 3*H);
    auto r_h = ::tenzor::slice(gates_hh, 1, 0,     H);
    auto z_h = ::tenzor::slice(gates_hh, 1, H,   2*H);
    auto n_h = ::tenzor::slice(gates_hh, 1, 2*H, 3*H);

    // Reset gate: r_t = σ(r_i + r_h)
    auto r_t = ::tenzor::sigmoid(r_i + r_h);

    // Update gate: z_t = σ(z_i + z_h)
    auto z_t = ::tenzor::sigmoid(z_i + z_h);

    // New gate: n_t = tanh(n_i + r_t ⊙ n_h)
    auto n_t = ::tenzor::tanh(n_i + r_t * n_h);

    // h_new = (1 - z_t) ⊙ n_t + z_t ⊙ h
    //       = n_t + z_t ⊙ (h - n_t)
    // (this formulation only needs Variable - Variable / Variable * Variable
    //  / Variable + Variable, all of which are autograd-aware)
    auto h_new = n_t + z_t * (h - n_t);

    return h_new;
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

auto GRU::forward(const Variable& input, const Variable& hx,
                  const Tensor& lengths)
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
        // Use Variable-level transpose so backward propagates through the
        // batch_first→sequence_first reshape (the raw tensor re-wrap
        // silently severed the chain to the user's input).
        x = ::tenzor::transpose(x, 0, 1);
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
    // Conditions: CPU, Float32, not training, not bidirectional
    // Uses oneDNN fused multi-layer GRU primitive for optimal performance
    // =========================================================================
    bool can_use_fused = is_op_supported(OpId::GRUForward, input.device().type) &&
                         input.dtype() == DType::Float32 &&
                         !is_training() &&
                         !bidirectional_;

    if (can_use_fused) {
        // Prepare input tensor for kernel (batch_first format for optimal oneDNN performance)
        Tensor layer_input = x.tensor();
        if (batch_first_) {
            layer_input = layer_input.transpose(0, 1);
        }
        layer_input = layer_input.contiguous();

        // For multi-layer GRU without dropout, use fused kernel
        // Dropout requires per-layer execution
        bool use_multilayer_fused = (num_layers_ > 1) && (!dropout_ || dropout_p_ == 0.0);

        if (use_multilayer_fused) {
            // Collect weights from all layers
            // Input format: [input, h0, W_ih_0, W_hh_0, bias_0, W_ih_1, ...]
            std::vector<Tensor> kernel_inputs;
            kernel_inputs.push_back(layer_input);
            kernel_inputs.push_back(h.tensor().contiguous());

            for (int64_t layer = 0; layer < num_layers_; ++layer) {
                auto& cell = forward_cells_[layer];
                auto W_ih_linear = cell->weight_ih();
                auto cell_params = cell->parameters();

                auto W_ih_params = W_ih_linear->parameters();
                kernel_inputs.push_back(W_ih_params[0]->tensor());  // W_ih
                kernel_inputs.push_back(cell_params[2]->tensor());  // W_hh

                // Bias or empty tensor
                if (W_ih_params.size() > 1) {
                    kernel_inputs.push_back(W_ih_params[1]->tensor());
                } else {
                    kernel_inputs.push_back(empty({0}, DType::Float32, input.device()));
                }
            }

            // Call fused multi-layer kernel
            OpAttributes attrs;
            attrs.set(AttrKey::NumLayers, num_layers_);
            auto outputs = dispatch<OpId::GRUMultiLayerForward>(kernel_inputs, attrs);

            // Reshape output
            Variable output(outputs[0], false);
            if (batch_first_) {
                output = Variable(output.tensor().transpose(0, 1), false);
            }

            Variable h_final(outputs[1], false);

            return {output, h_final};
        }

        // Single-layer or dropout case: process layers sequentially using fused kernels
        std::vector<Tensor> final_h_states;

        for (int64_t layer = 0; layer < num_layers_; ++layer) {
            auto& cell = forward_cells_[layer];
            auto W_ih_linear = cell->weight_ih();
            auto cell_params = cell->parameters();

            // Get weight tensors
            auto W_ih_params = W_ih_linear->parameters();
            Tensor W_ih_tensor = W_ih_params[0]->tensor().contiguous();
            Tensor W_hh_tensor = cell_params[2]->tensor().contiguous();

            // Get bias or empty tensor
            Tensor bias_tensor;
            if (W_ih_params.size() > 1) {
                bias_tensor = W_ih_params[1]->tensor().contiguous();
            } else {
                bias_tensor = empty({0}, DType::Float32, input.device());
            }

            // Get initial state for this layer
            Tensor h0_layer = h.tensor().slice(0, layer, layer + 1)
                                .reshape({batch_size, hidden_size_}).contiguous();

            // Call fused kernel
            std::vector<Tensor> inputs = {layer_input, W_ih_tensor, W_hh_tensor,
                                           bias_tensor, h0_layer};
            auto outputs = dispatch<OpId::GRUForward>(inputs);

            // Store final state
            final_h_states.push_back(outputs[1]);

            // Output becomes input for next layer
            layer_input = outputs[0];

            // Apply dropout between layers (not after last layer)
            if (dropout_ && layer < num_layers_ - 1) {
                layer_input = dropout_->forward(Variable(layer_input, false)).tensor();
            }
        }

        // Reshape output
        Variable output(layer_input, false);
        if (batch_first_) {
            output = Variable(output.tensor().transpose(0, 1), false);
        }

        // Stack final states: (num_layers, batch, hidden)
        Variable h_final(stack(std::span<const Tensor>(final_h_states), 0), false);

        return {output, h_final};
    }

    // =========================================================================
    // STANDARD PATH: Autograd-correct per-timestep forward.
    // Mirror LSTM standard path — see lstm.cpp for rationale.
    // =========================================================================
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

    Variable layer_input = x;  // (seq, batch, feat)
    std::vector<Variable> final_h_states;

    for (int64_t layer = 0; layer < num_layers_; ++layer) {
        auto& forward_cell = forward_cells_[layer];
        Variable forward_h = h_layers[layer * num_directions];

        const bool have_lengths = lengths.is_valid() && lengths.numel() > 0;
        Tensor lengths_dev;
        if (have_lengths) {
            lengths_dev = lengths.to(input.device()).to(DType::Float32);
        }

        std::vector<Variable> forward_outputs;
        forward_outputs.reserve(static_cast<size_t>(seq_len));

        for (int64_t t = 0; t < seq_len; ++t) {
            auto x_t_raw = ::tenzor::slice(layer_input, 0, t, t + 1);
            auto x_t = ::tenzor::squeeze(x_t_raw, 0);

            auto h_next = forward_cell->forward(x_t, forward_h);

            if (have_lengths) {
                auto t_scalar = full({batch_size}, static_cast<float>(t), DType::Float32, input.device());
                auto mask_t = gt(lengths_dev, t_scalar).to(input.dtype()).reshape({batch_size, 1});
                auto one_minus_mask_t =
                    full({batch_size, 1}, 1.0f, input.dtype(), input.device()) - mask_t;
                Variable mask_v(mask_t, false);
                Variable one_minus_mask_v(one_minus_mask_t, false);

                forward_h = mask_v * h_next + one_minus_mask_v * forward_h;
            } else {
                forward_h = h_next;
            }
            forward_outputs.push_back(forward_h);
        }

        final_h_states.push_back(forward_h);

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

                auto h_next = backward_cell->forward(x_t, backward_h);

                if (have_lengths) {
                    auto t_scalar = full({batch_size}, static_cast<float>(t), DType::Float32, input.device());
                    auto mask_t = gt(lengths_dev, t_scalar).to(input.dtype()).reshape({batch_size, 1});
                    auto one_minus_mask_t =
                        full({batch_size, 1}, 1.0f, input.dtype(), input.device()) - mask_t;
                    Variable mask_v(mask_t, false);
                    Variable one_minus_mask_v(one_minus_mask_t, false);

                    backward_h = mask_v * h_next + one_minus_mask_v * backward_h;
                } else {
                    backward_h = h_next;
                }
                backward_outputs.push_back(backward_h);
            }

            std::reverse(backward_outputs.begin(), backward_outputs.end());
            final_h_states.push_back(backward_h);

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

        if (dropout_ && layer < num_layers_ - 1) {
            layer_output = dropout_->forward(layer_output);
        }

        layer_input = layer_output;
    }

    Variable output = layer_input;

    if (batch_first_) {
        output = ::tenzor::transpose(output, 0, 1);
    }

    auto stack_layer_states = [](const std::vector<Variable>& states) {
        std::vector<Variable> expanded;
        expanded.reserve(states.size());
        for (const auto& s : states) {
            expanded.push_back(::tenzor::unsqueeze(s, 0));
        }
        return ::tenzor::cat(expanded, 0);
    };

    Variable h_final = stack_layer_states(final_h_states);

    return {output, h_final};
}

} // namespace tenzor::nn
