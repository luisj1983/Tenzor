#include "tenzor/nn/layers/rnn.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/utils/log.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <mutex>

namespace tenzor::nn {

// ============================================================================
// LSTMCell Implementation
// ============================================================================

LSTMCell::LSTMCell(int64_t input_size, int64_t hidden_size, bool bias)
    : LSTMCell(input_size, hidden_size, /*recurrent_size=*/hidden_size, bias) {}

// Audit G1: explicit-recurrent-size constructor for LSTM with projection.
LSTMCell::LSTMCell(int64_t input_size, int64_t hidden_size,
                   int64_t recurrent_size, bool bias)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      recurrent_size_(recurrent_size) {

    // PyTorch-style LSTM with separate weight matrices for input and hidden
    // All 4 gates are combined into single weight matrices for efficiency
    // weight_ih: (4 * hidden_size, input_size) - maps input to all 4 gates
    // weight_hh: (4 * hidden_size, recurrent_size) - maps hidden state to all 4 gates
    //            (recurrent_size == hidden_size by default; == proj_size for LSTMP)

    weight_ih_ = std::make_shared<Linear>(input_size, 4 * hidden_size, bias);
    register_module("weight_ih", weight_ih_);

    weight_hh_ = std::make_shared<Linear>(recurrent_size, 4 * hidden_size, false);  // No bias for hh
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
        h = Variable(zeros({batch_size, recurrent_size_},
                          input.dtype(), input.device()), false);
    }
    if (!c.is_initialized() || c.tensor().numel() == 0) {
        c = Variable(zeros({batch_size, hidden_size_},
                          input.dtype(), input.device()), false);
    }

    // Validate hidden and cell state shapes. Audit G1: h has recurrent_size
    // (== hidden_size unless this cell is part of an LSTM with projection).
    auto h_shape = h.shape();
    auto c_shape = c.shape();
    if (h_shape.size() != 2 || h_shape[0] != batch_size || h_shape[1] != recurrent_size_) {
        throw std::runtime_error("LSTMCell: invalid hidden state shape");
    }
    if (c_shape.size() != 2 || c_shape[0] != batch_size || c_shape[1] != hidden_size_) {
        throw std::runtime_error("LSTMCell: invalid cell state shape");
    }

    // Compute combined gates: gates = W_ih @ input + W_hh @ hidden
    // gates shape: (batch, 4 * hidden_size)
    // gates = [input_gate | forget_gate | cell_gate | output_gate]
    //
    // IMPORTANT: every operation here must stay on the autograd-aware
    // Variable path. Extracting .tensor() and re-wrapping with
    // `Variable(t, true)` would create orphan Variables with no grad_fn,
    // breaking the chain back to `input`/`hx`/`cx` so backward() wouldn't
    // populate any input gradient.
    auto gates_ih = weight_ih_->forward(input);  // (batch, 4*hidden_size)
    auto gates_hh = weight_hh_->forward(h);      // (batch, 4*hidden_size)
    auto gates = gates_ih + gates_hh;            // Variable + Variable -> Variable

    // Sanity check shape
    {
        auto gs = gates.shape();
        if (gs.size() != 2 || gs[0] != batch_size || gs[1] != 4 * hidden_size_) {
            std::ostringstream oss;
            oss << "LSTMCell: gates has wrong shape: (" << gs[0] << ", " << gs[1]
                << "), expected (" << batch_size << ", " << (4 * hidden_size_) << ")";
            throw std::runtime_error(oss.str());
        }
    }

    // Split gates into 4 sub-variables along dim=1, each of width hidden_size_.
    // We use slice (autograd-aware) instead of chunk because there is no
    // autograd::chunk overload yet.
    const int64_t H = hidden_size_;
    auto i_gate = ::tenzor::slice(gates, 1, 0,       H);
    auto f_gate = ::tenzor::slice(gates, 1, H,     2*H);
    auto g_gate = ::tenzor::slice(gates, 1, 2*H,   3*H);
    auto o_gate = ::tenzor::slice(gates, 1, 3*H,   4*H);

    auto i_t = ::tenzor::sigmoid(i_gate);
    auto f_t = ::tenzor::sigmoid(f_gate);
    auto g_t = ::tenzor::tanh(g_gate);
    auto o_t = ::tenzor::sigmoid(o_gate);

    // Update cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    auto c_new = (f_t * c) + (i_t * g_t);

    // Update hidden state: h_t = o_t ⊙ tanh(c_t)
    auto h_new = o_t * ::tenzor::tanh(c_new);

    return {h_new, c_new};
}

auto LSTMCell::forward_with_precomputed_ih(const Tensor& gates_ih, const Variable& hx, const Variable& cx)
    -> std::pair<Variable, Variable> {
    // `gates_ih` is a raw Tensor: the input-gate contributions for all four
    // LSTM gates, concatenated along dim=1. The caller opted out of input-side
    // autograd (Tensor signature), but the hidden-side / cell-state / weight_hh
    // grad chain through `hx`/`cx`/`weight_hh_` must still be preserved.
    //
    // Previous implementation fixed only the c_new/h_new ops at the bottom,
    // but still ran `gates_ih + gates_hh.tensor()` (raw Tensor add, strips
    // grad_fn) and then `chunk(gates_tensor, 4, 1)` (raw Tensor chunk),
    // severing the chain back to `h`/`weight_hh_`. Gradients on those were
    // silently zero. Rewritten to keep the gate combination on the Variable
    // path; c_new/h_new were already correct and are untouched.
    int64_t batch_size = gates_ih.shape()[0];

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

    auto gates_hh = weight_hh_->forward(h);  // Variable with grad_fn

    // Non-autograd constant for the input-side contribution.
    Variable gates_ih_var(gates_ih, false);
    auto gates = gates_ih_var + gates_hh;                 // Variable + Variable

    // No autograd-aware chunk yet; use slice (which is autograd-aware) to
    // split the 4*H-wide gate tensor into four H-wide gates. Same pattern
    // as the standard RNN per-timestep path.
    const int64_t H = hidden_size_;
    auto i_t = nn::sigmoid(::tenzor::slice(gates, 1, 0,       H));
    auto f_t = nn::sigmoid(::tenzor::slice(gates, 1, H,     2*H));
    auto g_t = nn::tanh   (::tenzor::slice(gates, 1, 2*H,   3*H));
    auto o_t = nn::sigmoid(::tenzor::slice(gates, 1, 3*H,   4*H));

    // Update cell state and hidden state; Variable-level arithmetic preserves
    // grad_fn back to f_t/i_t/g_t/o_t → gates → gates_hh → weight_hh_/h.
    auto c_new = f_t * c + i_t * g_t;
    auto h_new = o_t * nn::tanh(c_new);
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
    // Audit G1: PyTorch convention — proj_size must be strictly less than
    // hidden_size when projection is enabled (the projection compresses).
    if (proj_size > 0 && proj_size >= hidden_size) {
        throw std::invalid_argument(
            "LSTM: proj_size (" + std::to_string(proj_size) +
            ") must be less than hidden_size (" + std::to_string(hidden_size) +
            ") when projection is enabled.");
    }

    // With proj_size > 0, the inter-layer input AND the cell's recurrent
    // input are the projected hidden state (PyTorch LSTMP convention).
    // With proj_size == 0 the full hidden_size feeds both.
    const int64_t inter_layer_dim = (proj_size > 0 ? proj_size : hidden_size);
    const int64_t recurrent_dim   = inter_layer_dim;

    // Create forward cells for each layer
    for (int64_t i = 0; i < num_layers; ++i) {
        int64_t layer_input_size = (i == 0)
            ? input_size
            : inter_layer_dim * (bidirectional_ ? 2 : 1);
        auto cell = std::make_shared<LSTMCell>(layer_input_size, hidden_size,
                                                 recurrent_dim, bias);
        forward_cells_.push_back(cell);
        register_module("forward_cell_" + std::to_string(i), cell);
    }

    // Create backward cells for bidirectional LSTM
    if (bidirectional_) {
        for (int64_t i = 0; i < num_layers; ++i) {
            int64_t layer_input_size = (i == 0)
                ? input_size
                : inter_layer_dim * 2;
            auto cell = std::make_shared<LSTMCell>(layer_input_size, hidden_size,
                                                     recurrent_dim, bias);
            backward_cells_.push_back(cell);
            register_module("backward_cell_" + std::to_string(i), cell);
        }
    }

    // Audit G1: per-layer projection Linear(hidden_size → proj_size, no bias).
    // Mirrors PyTorch's `weight_hr_l[k]`. Applied after each cell forward to
    // produce the inter-layer input (and the final h_n).
    if (proj_size_ > 0) {
        for (int64_t i = 0; i < num_layers; ++i) {
            auto proj = std::make_shared<Linear>(hidden_size, proj_size_, /*bias=*/false);
            forward_projections_.push_back(proj);
            register_module("forward_projection_" + std::to_string(i), proj);
        }
        if (bidirectional_) {
            for (int64_t i = 0; i < num_layers; ++i) {
                auto proj = std::make_shared<Linear>(hidden_size, proj_size_, /*bias=*/false);
                backward_projections_.push_back(proj);
                register_module("backward_projection_" + std::to_string(i), proj);
            }
        }
    }

    // Create dropout layer
    if (dropout_p_ > 0.0 && num_layers > 1) {
        dropout_ = std::make_shared<Dropout>(dropout_p_);
        register_module("dropout", dropout_);
    }
}

auto LSTM::forward(const Variable& input, const std::pair<Variable, Variable>& hx,
                   const Tensor& lengths)
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
        // Use Variable-level transpose so backward propagates through the
        // batch_first→sequence_first reshape. Previously the raw-tensor
        // re-wrap silently severed the chain to the user's input.
        x = ::tenzor::transpose(x, 0, 1);
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

    // Audit G1: with proj_size > 0, the externally-visible h state has
    // proj_size as the trailing dim (PyTorch convention). The cell state
    // always has hidden_size as the trailing dim.
    const int64_t h_trailing = (proj_size_ > 0 ? proj_size_ : hidden_size_);

    if (!h.is_initialized() || h.tensor().numel() == 0) {
        h = Variable(zeros({num_layers_ * num_directions, batch_size, h_trailing},
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
        h_shape[1] != batch_size || h_shape[2] != h_trailing) {
        throw std::runtime_error("LSTM: invalid hidden state shape");
    }
    if (c_shape.size() != 3 || c_shape[0] != num_layers_ * num_directions ||
        c_shape[1] != batch_size || c_shape[2] != hidden_size_) {
        throw std::runtime_error("LSTM: invalid cell state shape");
    }

    // =========================================================================
    // FAST PATH: Use fused backend kernel for inference
    // CPU: Uses oneDNN fused LSTM primitives for optimal performance
    // CUDA: cuDNN LSTM forward/backward available in cudnn_ops.cu (enable #if 0 → #if 1)
    //        Requires reserve_space from forward to be passed to backward for correct gradients
    // =========================================================================
    // Audit G1: the fused-kernel LSTM forward paths below do not yet
    // implement the per-layer hidden→proj projection (weight_hr). When the
    // user has enabled projection, fall through to the standard autograd
    // path which composes the projection at the Variable level. This is a
    // perf cliff but produces correct math; the fused path with projection
    // is tracked as G1-followup (cuDNN/oneDNN both support projection in
    // their LSTM primitives).
    bool can_use_fused = is_op_supported(OpId::LSTMForward, input.device().type) &&
                         input.dtype() == DType::Float32 &&
                         !is_training() &&
                         proj_size_ == 0;

    // Special fast path for single-layer bidirectional LSTM
    if (can_use_fused && bidirectional_ && num_layers_ == 1) {
        Tensor layer_input = x.tensor().contiguous();

        // Get forward direction weights
        auto& fwd_cell = forward_cells_[0];
        auto W_ih_fwd_linear = fwd_cell->weight_ih();
        auto W_hh_fwd_linear = fwd_cell->weight_hh();
        auto W_ih_fwd_params = W_ih_fwd_linear->parameters();
        auto W_hh_fwd_params = W_hh_fwd_linear->parameters();

        Tensor W_ih_fwd = W_ih_fwd_params[0]->tensor().contiguous();
        Tensor W_hh_fwd = W_hh_fwd_params[0]->tensor().contiguous();
        Tensor bias_ih_fwd = W_ih_fwd_params.size() > 1 ?
            W_ih_fwd_params[1]->tensor() : empty({0}, DType::Float32, input.device());
        Tensor bias_hh_fwd = W_hh_fwd_params.size() > 1 ?
            W_hh_fwd_params[1]->tensor() : empty({0}, DType::Float32, input.device());

        // Get backward direction weights
        auto& bwd_cell = backward_cells_[0];
        auto W_ih_bwd_linear = bwd_cell->weight_ih();
        auto W_hh_bwd_linear = bwd_cell->weight_hh();
        auto W_ih_bwd_params = W_ih_bwd_linear->parameters();
        auto W_hh_bwd_params = W_hh_bwd_linear->parameters();

        Tensor W_ih_bwd = W_ih_bwd_params[0]->tensor().contiguous();
        Tensor W_hh_bwd = W_hh_bwd_params[0]->tensor().contiguous();
        Tensor bias_ih_bwd = W_ih_bwd_params.size() > 1 ?
            W_ih_bwd_params[1]->tensor() : empty({0}, DType::Float32, input.device());
        Tensor bias_hh_bwd = W_hh_bwd_params.size() > 1 ?
            W_hh_bwd_params[1]->tensor() : empty({0}, DType::Float32, input.device());

        // h0, c0 are (num_layers * num_directions, batch, hidden) = (2, batch, hidden) for single-layer BiLSTM
        Tensor h0_tensor = h.tensor().contiguous();
        Tensor c0_tensor = c.tensor().contiguous();

        std::vector<Tensor> inputs = {
            layer_input, h0_tensor, c0_tensor,
            W_ih_fwd, W_hh_fwd, bias_ih_fwd, bias_hh_fwd,
            W_ih_bwd, W_hh_bwd, bias_ih_bwd, bias_hh_bwd
        };

        auto outputs = dispatch<OpId::BiLSTMForward>(inputs);

        Variable output(outputs[0], false);
        if (batch_first_) {
            output = Variable(output.tensor().transpose(0, 1), false);
        }

        return {output, {Variable(outputs[1], false), Variable(outputs[2], false)}};
    }

    // Multi-layer bidirectional LSTM fast path
    if (can_use_fused && bidirectional_ && num_layers_ > 1) {
        Tensor layer_input = x.tensor().contiguous();
        int64_t kernel_batch = layer_input.shape()[1];

        std::vector<Tensor> final_h_states;
        std::vector<Tensor> final_c_states;

        for (int64_t layer = 0; layer < num_layers_; ++layer) {
            // Get forward direction weights for this layer
            auto& fwd_cell = forward_cells_[layer];
            auto W_ih_fwd_linear = fwd_cell->weight_ih();
            auto W_hh_fwd_linear = fwd_cell->weight_hh();
            auto W_ih_fwd_params = W_ih_fwd_linear->parameters();
            auto W_hh_fwd_params = W_hh_fwd_linear->parameters();

            Tensor W_ih_fwd = W_ih_fwd_params[0]->tensor().contiguous();
            Tensor W_hh_fwd = W_hh_fwd_params[0]->tensor().contiguous();
            Tensor bias_ih_fwd = W_ih_fwd_params.size() > 1 ?
                W_ih_fwd_params[1]->tensor() : empty({0}, DType::Float32, input.device());
            Tensor bias_hh_fwd = W_hh_fwd_params.size() > 1 ?
                W_hh_fwd_params[1]->tensor() : empty({0}, DType::Float32, input.device());

            // Get backward direction weights for this layer
            auto& bwd_cell = backward_cells_[layer];
            auto W_ih_bwd_linear = bwd_cell->weight_ih();
            auto W_hh_bwd_linear = bwd_cell->weight_hh();
            auto W_ih_bwd_params = W_ih_bwd_linear->parameters();
            auto W_hh_bwd_params = W_hh_bwd_linear->parameters();

            Tensor W_ih_bwd = W_ih_bwd_params[0]->tensor().contiguous();
            Tensor W_hh_bwd = W_hh_bwd_params[0]->tensor().contiguous();
            Tensor bias_ih_bwd = W_ih_bwd_params.size() > 1 ?
                W_ih_bwd_params[1]->tensor() : empty({0}, DType::Float32, input.device());
            Tensor bias_hh_bwd = W_hh_bwd_params.size() > 1 ?
                W_hh_bwd_params[1]->tensor() : empty({0}, DType::Float32, input.device());

            // h0, c0 for this layer: extract from (num_layers*2, batch, hidden)
            // Layout: [layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd, ...]
            int64_t fwd_idx = layer * 2;
            int64_t bwd_idx = layer * 2 + 1;

            Tensor h0_fwd = h.tensor().slice(0, fwd_idx, fwd_idx + 1)
                            .reshape({kernel_batch, hidden_size_}).contiguous();
            Tensor c0_fwd = c.tensor().slice(0, fwd_idx, fwd_idx + 1)
                            .reshape({kernel_batch, hidden_size_}).contiguous();
            Tensor h0_bwd = h.tensor().slice(0, bwd_idx, bwd_idx + 1)
                            .reshape({kernel_batch, hidden_size_}).contiguous();
            Tensor c0_bwd = c.tensor().slice(0, bwd_idx, bwd_idx + 1)
                            .reshape({kernel_batch, hidden_size_}).contiguous();

            // Stack h0 and c0 for BiLSTM kernel: (2, batch, hidden)
            std::vector<Tensor> h0_list = {h0_fwd, h0_bwd};
            std::vector<Tensor> c0_list = {c0_fwd, c0_bwd};
            Tensor h0_stacked = stack(std::span<const Tensor>(h0_list), 0);
            Tensor c0_stacked = stack(std::span<const Tensor>(c0_list), 0);

            std::vector<Tensor> inputs = {
                layer_input, h0_stacked, c0_stacked,
                W_ih_fwd, W_hh_fwd, bias_ih_fwd, bias_hh_fwd,
                W_ih_bwd, W_hh_bwd, bias_ih_bwd, bias_hh_bwd
            };

            auto outputs = dispatch<OpId::BiLSTMForward>(inputs);

            // outputs[0]: (seq, batch, 2*hidden)
            // outputs[1]: h_n (2, batch, hidden)
            // outputs[2]: c_n (2, batch, hidden)

            // Store final states for this layer
            final_h_states.push_back(outputs[1].slice(0, 0, 1).reshape({1, kernel_batch, hidden_size_}));
            final_h_states.push_back(outputs[1].slice(0, 1, 2).reshape({1, kernel_batch, hidden_size_}));
            final_c_states.push_back(outputs[2].slice(0, 0, 1).reshape({1, kernel_batch, hidden_size_}));
            final_c_states.push_back(outputs[2].slice(0, 1, 2).reshape({1, kernel_batch, hidden_size_}));

            // Output becomes input for next layer
            layer_input = outputs[0].contiguous();

            // Apply dropout between layers (but not after last layer)
            if (dropout_ && layer < num_layers_ - 1) {
                layer_input = dropout_->forward(Variable(layer_input, false)).tensor();
            }
        }

        Variable output(layer_input, false);
        if (batch_first_) {
            output = Variable(output.tensor().transpose(0, 1), false);
        }

        // Stack all hidden states: (num_layers*2, batch, hidden)
        Variable h_final(cat(std::span<const Tensor>(final_h_states), 0), false);
        Variable c_final(cat(std::span<const Tensor>(final_c_states), 0), false);

        return {output, {h_final, c_final}};
    }

    // For unidirectional LSTM, continue with existing fast path
    can_use_fused = can_use_fused && !bidirectional_;

    if (can_use_fused) {
        // x is already transposed to (seq, batch, input) format above (line 279)
        // Use tensor directly if already contiguous (avoid copy)
        const Tensor& x_tensor = x.tensor();
        Tensor layer_input = x_tensor.is_contiguous() ? x_tensor : x_tensor.contiguous();

        // SINGLE-LAYER: Direct dispatch (most efficient)
        if (num_layers_ == 1) {
            auto& cell = forward_cells_[0];

            // Access weights directly from Linear layers (cached in module)
            const Tensor& W_ih_tensor = cell->weight_ih()->weight()->tensor();
            const Tensor& W_hh_tensor = cell->weight_hh()->weight()->tensor();
            Tensor bias_ih_tensor = cell->weight_ih()->has_bias() ?
                cell->weight_ih()->bias()->tensor() : empty({0}, DType::Float32, input.device());
            Tensor bias_hh_tensor = cell->weight_hh()->has_bias() ?
                cell->weight_hh()->bias()->tensor() : empty({0}, DType::Float32, input.device());

            // For single-layer, h0/c0 shape is (1, batch, hidden)
            // We can squeeze the first dim instead of slice+reshape if contiguous
            const Tensor& h_tensor = h.tensor();
            const Tensor& c_tensor = c.tensor();
            int64_t kernel_batch = layer_input.shape()[1];

            // Fast path: if h0/c0 are contiguous and shape is (1, batch, hidden),
            // the data layout is already (batch, hidden) - just view it
            Tensor h0_tensor, c0_tensor;
            if (h_tensor.is_contiguous() && h_tensor.shape()[0] == 1) {
                h0_tensor = h_tensor.reshape({kernel_batch, hidden_size_});
            } else {
                h0_tensor = h_tensor.slice(0, 0, 1).reshape({kernel_batch, hidden_size_}).contiguous();
            }
            if (c_tensor.is_contiguous() && c_tensor.shape()[0] == 1) {
                c0_tensor = c_tensor.reshape({kernel_batch, hidden_size_});
            } else {
                c0_tensor = c_tensor.slice(0, 0, 1).reshape({kernel_batch, hidden_size_}).contiguous();
            }

            // Pass both bias_ih and bias_hh for kernel to combine during cache setup
            std::vector<Tensor> inputs = {layer_input, W_ih_tensor, W_hh_tensor,
                                           bias_ih_tensor, bias_hh_tensor, h0_tensor, c0_tensor};
            auto outputs = dispatch<OpId::LSTMForward>(inputs);

            Variable output(outputs[0], false);
            if (batch_first_) {
                output = Variable(output.tensor().transpose(0, 1), false);
            }

            // outputs[1] is (batch, hidden) - unsqueeze to (1, batch, hidden)
            Variable h_final(outputs[1].unsqueeze(0), false);
            Variable c_final(outputs[2].unsqueeze(0), false);

            return {output, {h_final, c_final}};
        }

        // MULTI-LAYER without dropout: Use fused multi-layer kernel
        if (!dropout_ || dropout_p_ == 0.0) {
            std::vector<Tensor> kernel_inputs;
            kernel_inputs.push_back(layer_input);
            kernel_inputs.push_back(h.tensor().contiguous());
            kernel_inputs.push_back(c.tensor().contiguous());

            for (int64_t layer = 0; layer < num_layers_; ++layer) {
                auto& cell = forward_cells_[layer];
                auto W_ih_linear = cell->weight_ih();
                auto W_hh_linear = cell->weight_hh();

                auto W_ih_params = W_ih_linear->parameters();
                auto W_hh_params = W_hh_linear->parameters();
                kernel_inputs.push_back(W_ih_params[0]->tensor());
                kernel_inputs.push_back(W_hh_params[0]->tensor());

                if (W_ih_params.size() > 1) {
                    kernel_inputs.push_back(W_ih_params[1]->tensor());
                } else {
                    kernel_inputs.push_back(empty({0}, DType::Float32, input.device()));
                }
            }

            OpAttributes attrs;
            attrs.set(AttrKey::NumLayers, num_layers_);
            auto outputs = dispatch<OpId::LSTMMultiLayerForward>(kernel_inputs, attrs);

            Variable output(outputs[0], false);
            if (batch_first_) {
                output = Variable(output.tensor().transpose(0, 1), false);
            }

            return {output, {Variable(outputs[1], false), Variable(outputs[2], false)}};
        }

        // MULTI-LAYER with dropout: Process layers sequentially
        std::vector<Tensor> final_h_states;
        std::vector<Tensor> final_c_states;

        for (int64_t layer = 0; layer < num_layers_; ++layer) {
            auto& cell = forward_cells_[layer];
            auto W_ih_linear = cell->weight_ih();
            auto W_hh_linear = cell->weight_hh();

            auto W_ih_params = W_ih_linear->parameters();
            auto W_hh_params = W_hh_linear->parameters();
            const Tensor& W_ih_tensor = W_ih_params[0]->tensor();
            const Tensor& W_hh_tensor = W_hh_params[0]->tensor();

            // Get bias tensors - pass both to kernel for combining during cache setup
            Tensor bias_ih_tensor, bias_hh_tensor;
            if (W_ih_params.size() > 1) {
                bias_ih_tensor = W_ih_params[1]->tensor();
            } else {
                bias_ih_tensor = empty({0}, DType::Float32, input.device());
            }
            if (W_hh_params.size() > 1) {
                bias_hh_tensor = W_hh_params[1]->tensor();
            } else {
                bias_hh_tensor = empty({0}, DType::Float32, input.device());
            }

            Tensor h0_layer = h.tensor().slice(0, layer, layer + 1)
                                .reshape({batch_size, hidden_size_}).contiguous();
            Tensor c0_layer = c.tensor().slice(0, layer, layer + 1)
                                .reshape({batch_size, hidden_size_}).contiguous();

            std::vector<Tensor> inputs = {layer_input, W_ih_tensor, W_hh_tensor,
                                           bias_ih_tensor, bias_hh_tensor, h0_layer, c0_layer};
            auto outputs = dispatch<OpId::LSTMForward>(inputs);

            final_h_states.push_back(outputs[1]);
            final_c_states.push_back(outputs[2]);
            layer_input = outputs[0];

            if (dropout_ && layer < num_layers_ - 1) {
                layer_input = dropout_->forward(Variable(layer_input, false)).tensor();
            }
        }

        Variable output(layer_input, false);
        if (batch_first_) {
            output = Variable(output.tensor().transpose(0, 1), false);
        }

        Variable h_final(stack(std::span<const Tensor>(final_h_states), 0), false);
        Variable c_final(stack(std::span<const Tensor>(final_c_states), 0), false);

        return {output, {h_final, c_final}};
    }

    // =========================================================================
    // STANDARD PATH: Autograd-correct per-timestep forward.
    // =========================================================================
    // This path must keep every operation on the Variable/autograd graph so
    // that backward() can propagate gradients all the way back to `input`
    // (and through each cell's parameters). The previous implementation
    // pre-computed input gates via `Variable(x_flat, false)` and called
    // `forward_with_precomputed_ih`, which silently detached the input from
    // the graph and produced empty input gradients.
    //
    // We instead:
    //   1. Slice the (h0, c0) per-layer states with the autograd-aware
    //      `slice` + `squeeze` so intermediate states carry grad history.
    //   2. For each timestep, slice `layer_input` at t along dim=0, then
    //      call the cell's full `forward(input_var, h_var, c_var)`.
    //   3. Accumulate outputs via autograd-aware `unsqueeze` + `cat`.
    //
    // The inference-mode fused path above is preserved for speed when
    // `!is_training()`.
    //
    // Known v0.1 limitation: this path is materially slower than PyTorch's
    // cuDNN RNN training kernel (~50x on a CUDA RTX 5070 at the published
    // benchmark shapes). cuDNN RNN integration is on the v0.2 roadmap;
    // until then, warn once per process so users notice the perf cliff
    // before they wait an hour for an epoch.
    if (input.device().type == Device::Type::CUDA) {
        static std::once_flag warned;
        std::call_once(warned, []() {
            // Audit I.4: unified logger so TENZOR_LOG_LEVEL applies.
            TENZOR_LOG_WARN("CUDA LSTM training uses the per-timestep autograd "
                            "path and is significantly slower than PyTorch's "
                            "cuDNN RNN. See known issues in CHANGELOG. Use "
                            "eval-mode forward, train on CPU, or wait for v0.2.");
        });
    }
    auto split_states_per_layer = [&](const Variable& stacked_states,
                                      std::vector<Variable>& out) {
        for (int64_t i = 0; i < num_layers_ * num_directions; ++i) {
            auto sliced = ::tenzor::slice(stacked_states, 0, i, i + 1);
            auto sq = ::tenzor::squeeze(sliced, 0);  // (batch, hidden)
            out.push_back(sq);
        }
    };

    std::vector<Variable> h_layers;
    std::vector<Variable> c_layers;
    split_states_per_layer(h, h_layers);
    split_states_per_layer(c, c_layers);

    // Layer-by-layer processing.
    Variable layer_input = x;  // shape: (seq, batch, feat)
    std::vector<Variable> final_h_states;
    std::vector<Variable> final_c_states;

    for (int64_t layer = 0; layer < num_layers_; ++layer) {
        auto& forward_cell = forward_cells_[layer];
        Variable forward_h = h_layers[layer * num_directions];
        Variable forward_c = c_layers[layer * num_directions];

        const bool have_lengths = lengths.is_valid() && lengths.numel() > 0;
        Tensor lengths_dev;
        if (have_lengths) {
            lengths_dev = lengths.to(input.device()).to(DType::Float32);
        }

        std::vector<Variable> forward_outputs;  // each: (batch, hidden)
        forward_outputs.reserve(static_cast<size_t>(seq_len));

        // Per-timestep forward pass through the full cell.
        // Audit G1: with proj_size > 0, project h_next via the layer's
        // forward_projections_[layer] before storing/recurring. The cell
        // produces an `hidden_size`-shaped h_next; the projection compresses
        // to `proj_size`, which is what feeds the next timestep recurrence
        // (cell's W_hh has proj_size columns when LSTMP) and also what
        // ultimately becomes the layer's output and h_n.
        for (int64_t t = 0; t < seq_len; ++t) {
            auto x_t_raw = ::tenzor::slice(layer_input, 0, t, t + 1);  // (1, batch, feat)
            auto x_t = ::tenzor::squeeze(x_t_raw, 0);                  // (batch, feat)

            auto [h_next, c_next] = forward_cell->forward(x_t, forward_h, forward_c);

            if (proj_size_ > 0) {
                h_next = forward_projections_[layer]->forward(h_next);
            }

            if (have_lengths) {
                // Preserve old state for padded positions. mask and
                // one_minus_mask are built on raw Tensors; wrapping them in
                // requires_grad=false Variables is correct since they are
                // gating constants (no gradient flows through them).
                auto t_scalar = full({batch_size}, static_cast<float>(t), DType::Float32, input.device());
                auto mask_t = gt(lengths_dev, t_scalar).to(input.dtype()).reshape({batch_size, 1});
                auto one_minus_mask_t =
                    full({batch_size, 1}, 1.0f, input.dtype(), input.device()) - mask_t;
                Variable mask_v(mask_t, false);
                Variable one_minus_mask_v(one_minus_mask_t, false);

                forward_h = mask_v * h_next + one_minus_mask_v * forward_h;
                forward_c = mask_v * c_next + one_minus_mask_v * forward_c;
            } else {
                forward_h = h_next;
                forward_c = c_next;
            }

            forward_outputs.push_back(forward_h);
        }

        final_h_states.push_back(forward_h);
        final_c_states.push_back(forward_c);

        // Build layer_output as (seq, batch, hidden[*2]) on the autograd path.
        auto stack_timesteps = [](const std::vector<Variable>& step_outputs) {
            std::vector<Variable> expanded;
            expanded.reserve(step_outputs.size());
            for (const auto& v : step_outputs) {
                expanded.push_back(::tenzor::unsqueeze(v, 0));  // (1, batch, hidden)
            }
            return ::tenzor::cat(expanded, 0);  // (seq, batch, hidden)
        };

        Variable layer_output;
        if (bidirectional_) {
            auto& backward_cell = backward_cells_[layer];
            Variable backward_h = h_layers[layer * num_directions + 1];
            Variable backward_c = c_layers[layer * num_directions + 1];

            std::vector<Variable> backward_outputs;
            backward_outputs.reserve(static_cast<size_t>(seq_len));

            for (int64_t t = seq_len - 1; t >= 0; --t) {
                auto x_t_raw = ::tenzor::slice(layer_input, 0, t, t + 1);
                auto x_t = ::tenzor::squeeze(x_t_raw, 0);

                auto [h_next, c_next] = backward_cell->forward(x_t, backward_h, backward_c);

                // Audit G1: project the backward direction's hidden too.
                if (proj_size_ > 0) {
                    h_next = backward_projections_[layer]->forward(h_next);
                }

                if (have_lengths) {
                    auto t_scalar = full({batch_size}, static_cast<float>(t), DType::Float32, input.device());
                    auto mask_t = gt(lengths_dev, t_scalar).to(input.dtype()).reshape({batch_size, 1});
                    auto one_minus_mask_t =
                        full({batch_size, 1}, 1.0f, input.dtype(), input.device()) - mask_t;
                    Variable mask_v(mask_t, false);
                    Variable one_minus_mask_v(one_minus_mask_t, false);

                    backward_h = mask_v * h_next + one_minus_mask_v * backward_h;
                    backward_c = mask_v * c_next + one_minus_mask_v * backward_c;
                } else {
                    backward_h = h_next;
                    backward_c = c_next;
                }
                backward_outputs.push_back(backward_h);
            }

            std::reverse(backward_outputs.begin(), backward_outputs.end());
            final_h_states.push_back(backward_h);
            final_c_states.push_back(backward_c);

            // Concatenate forward and backward outputs along dim=1 (hidden).
            // Build per-timestep concatenated Variables, then stack.
            std::vector<Variable> concat_per_t;
            concat_per_t.reserve(static_cast<size_t>(seq_len));
            for (int64_t t = 0; t < seq_len; ++t) {
                std::vector<Variable> pair = {forward_outputs[t], backward_outputs[t]};
                concat_per_t.push_back(::tenzor::cat(pair, 1));  // (batch, 2*hidden)
            }
            layer_output = stack_timesteps(concat_per_t);
        } else {
            layer_output = stack_timesteps(forward_outputs);
        }

        // Apply dropout between layers (but not after last layer).
        if (dropout_ && layer < num_layers_ - 1) {
            layer_output = dropout_->forward(layer_output);
        }

        layer_input = layer_output;
    }

    Variable output = layer_input;  // (seq, batch, hidden[*2]), on autograd graph

    if (batch_first_) {
        // Autograd-aware transpose.
        output = ::tenzor::transpose(output, 0, 1);
    }

    // Stack per-layer final states along dim=0: list of (batch, hidden)
    //   -> (num_layers * num_directions, batch, hidden)
    auto stack_layer_states = [](const std::vector<Variable>& states) {
        std::vector<Variable> expanded;
        expanded.reserve(states.size());
        for (const auto& s : states) {
            expanded.push_back(::tenzor::unsqueeze(s, 0));
        }
        return ::tenzor::cat(expanded, 0);
    };

    Variable h_final = stack_layer_states(final_h_states);
    Variable c_final = stack_layer_states(final_c_states);

    return {output, {h_final, c_final}};
}

} // namespace tenzor::nn
