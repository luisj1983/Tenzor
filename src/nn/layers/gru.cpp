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
#include "tenzor/nn/utils/variable_cast.hpp"
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <mutex>
#include <cstdlib>

namespace tenzor::nn {

namespace {
// JIT-R072: Tensor::slice()/reshape()/transpose() are pure TensorImpl
// metadata tricks with ZERO dispatch() calls — invisible to both the
// tracer's generic dispatch<OpId> interception and its own tensor lineage
// map (same bug class as JIT-R052/R058a). Currently masked
// (OpId::GRUForward/GRUCudnnTrainForward are both unmapped, so the
// dispatch() call these wrap graph-breaks first regardless of whether the
// operands were traced), but this is the correct fix ready for when GRU JIT
// tracing support is added. Mirrors lstm.cpp/quantized_layers.cpp's
// traced_slice/traced_reshape/traced_transpose exactly.
inline auto traced_slice(const Tensor& t, int64_t dim, int64_t start,
                          int64_t end, int64_t step = 1) -> Tensor {
    return ::tenzor::slice(::tenzor::Variable(t, false), dim, start, end, step).tensor();
}
inline auto traced_reshape(const Tensor& t, std::vector<int64_t> shape) -> Tensor {
    return ::tenzor::reshape(::tenzor::Variable(t, false), std::move(shape)).tensor();
}
inline auto traced_transpose(const Tensor& t, int64_t dim0, int64_t dim1) -> Tensor {
    return ::tenzor::transpose(::tenzor::Variable(t, false), dim0, dim1).tensor();
}
}  // namespace

// ============================================================================
// GRUCell Implementation
// ============================================================================

GRUCell::GRUCell(int64_t input_size, int64_t hidden_size, bool bias,
                 bool linear_before_reset)
    : input_size_(input_size), hidden_size_(hidden_size),
      linear_before_reset_(linear_before_reset) {

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

    // Ensure module parameters are on the same device as input. Only trigger
    // the transfer when the parameter is actually on a different device —
    // otherwise this re-runs the (potentially expensive) move every timestep,
    // re-copying non-contiguous weights each step (LSTMCell guards identically).
    auto input_device = input.device();
    if (weight_ih_->weight()->device() != input_device) {
        weight_ih_->to(input_device);
    }
    if (weight_hh_->weight()->device() != input_device) {
        weight_hh_->to(input_device);
    }

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

    // For Float16/BFloat16, run the gate matmuls, sigmoid/tanh, reset/update
    // gating, and the new-state combine in Float32, then cast the new hidden
    // state back to the original dtype. Half-precision gate activations and the
    // recurrence lose precision and some elementwise kernels lack half dispatch.
    // Widening input/h to Float32 makes the Linear layers cast their weights to
    // Float32 to match (Linear's compute dtype follows input), so the whole cell
    // — including the mode-0 re-run of weight_hh_ on (r_t ⊙ h) — runs at Float32.
    const DType orig_dtype = input.dtype();
    const bool needs_upcast =
        (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Variable input_c = needs_upcast ? variable_cast(input, DType::Float32) : input;
    if (needs_upcast) {
        h = variable_cast(h, DType::Float32);
    }

    // Compute combined gates efficiently (2 linear calls instead of 6)
    // gates_ih: (batch, 3 * hidden_size) = [r_i | z_i | n_i]
    // gates_hh: (batch, 3 * hidden_size) = [r_h | z_h | n_h]
    //
    // IMPORTANT: stay on the autograd-aware Variable path. Extracting
    // .tensor() and re-wrapping with `Variable(t, true)` would create orphan
    // Variables with no grad_fn, so backward() couldn't reach `input`/`hx`.
    auto gates_ih = weight_ih_->forward(input_c);  // (batch, 3*hidden_size)
    auto gates_hh = weight_hh_->forward(h);        // (batch, 3*hidden_size)

    // Slice each gates tensor into 3 sub-variables (r, z, n) along dim=1.
    // (autograd::chunk does not exist yet — use slice which is autograd-aware.)
    const int64_t H = hidden_size_;
    auto r_i = ::tenzor::slice(gates_ih, 1, 0,     H);
    auto z_i = ::tenzor::slice(gates_ih, 1, H,   2*H);
    auto n_i = ::tenzor::slice(gates_ih, 1, 2*H, 3*H);
    auto r_h = ::tenzor::slice(gates_hh, 1, 0,     H);
    auto z_h = ::tenzor::slice(gates_hh, 1, H,   2*H);

    // Reset gate: r_t = σ(r_i + r_h)
    auto r_t = ::tenzor::sigmoid(r_i + r_h);

    // Update gate: z_t = σ(z_i + z_h)
    auto z_t = ::tenzor::sigmoid(z_i + z_h);

    // New gate:
    //   mode 1 (linear_before_reset_ == true, default, ONNX lbr=1):
    //       n_t = tanh(n_i + r_t ⊙ n_h)
    //     where n_h = (H·Rn + recurrent_bias_n) is the recurrent matmul RESULT
    //     and the reset gate multiplies that whole result.
    //   mode 0 (linear_before_reset_ == false, ONNX default lbr=0):
    //       n_t = tanh(n_i + (r_t ⊙ H)·Rn + recurrent_bias_n)
    //     where the reset gate multiplies the hidden state BEFORE the
    //     recurrent matmul; the recurrent new-gate bias is added after the
    //     matmul. We obtain this by re-running weight_hh_ on (r_t ⊙ H) and
    //     slicing its n-gate column (the Linear adds its bias post-matmul,
    //     matching ONNX). All ops stay Variable-level to keep autograd intact.
    Variable n_t;
    if (linear_before_reset_) {
        auto n_h = ::tenzor::slice(gates_hh, 1, 2*H, 3*H);
        n_t = ::tenzor::tanh(n_i + r_t * n_h);
    } else {
        auto gates_hh_reset = weight_hh_->forward(r_t * h);  // (batch, 3*H)
        auto n_h0 = ::tenzor::slice(gates_hh_reset, 1, 2*H, 3*H);
        n_t = ::tenzor::tanh(n_i + n_h0);
    }

    // h_new = (1 - z_t) ⊙ n_t + z_t ⊙ h
    //       = n_t + z_t ⊙ (h - n_t)
    // (this formulation only needs Variable - Variable / Variable * Variable
    //  / Variable + Variable, all of which are autograd-aware)
    auto h_new = n_t + z_t * (h - n_t);

    // Narrow the new hidden state back to the caller's original dtype.
    return needs_upcast ? variable_cast(h_new, orig_dtype) : h_new;
}

auto GRUCell::forward_with_precomputed_ih(const Tensor& gates_ih, const Variable& hx) -> Variable {
    // `gates_ih` is a raw Tensor: the input-gate contributions (for the three
    // GRU gates concatenated) computed by the caller without retaining
    // autograd on the input side. The recurrent / hidden-side grad chain
    // through `hx` and `weight_hh_` must still be preserved.
    //
    // Previously the whole body ran on raw Tensors, extracting .tensor() on
    // every intermediate and wrapping the final h_new in Variable(..., true)
    // with no grad_fn. Backward populated nothing on hx/weight_hh_. Rewritten
    // to use Variable-level chunk/sigmoid/tanh/arithmetic throughout.
    int64_t batch_size = gates_ih.shape()[0];

    Variable h = hx;
    if (!h.is_initialized() || h.tensor().numel() == 0) {
        h = Variable(zeros({batch_size, hidden_size_},
                          gates_ih.dtype(), gates_ih.device()), false);
    }

    // For Float16/BFloat16, run the gate sigmoid/tanh, reset/update gating, and
    // the new-state combine in Float32, then narrow the new hidden state back.
    // Mirrors GRUCell::forward: half-precision gate activations and the
    // recurrence lose precision and some elementwise kernels lack half dispatch.
    // Widening h to Float32 makes weight_hh_ cast its weights to Float32 to match
    // (Linear's compute dtype follows input), and we widen the precomputed
    // gates_ih constant to Float32 so all gate arithmetic runs at Float32 —
    // including the mode-0 re-run of weight_hh_ on (r_t ⊙ h).
    const DType orig_dtype = gates_ih.dtype();
    const bool needs_upcast =
        (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Tensor gates_ih_c = needs_upcast ? gates_ih.to(DType::Float32) : gates_ih;
    if (needs_upcast) {
        h = variable_cast(h, DType::Float32);
    }

    // Variable-level hidden-to-hidden gate computation.
    auto gates_hh = weight_hh_->forward(h);  // Variable, carries grad_fn

    // Wrap gates_ih as a non-autograd constant (caller opted out of input-side
    // grad). No autograd-aware chunk yet, so split 3*H-wide gate tensors with
    // autograd-aware slice — same pattern as the standard RNN per-timestep
    // path and as our LSTMCell::forward_with_precomputed_ih rewrite.
    Variable gates_ih_var(gates_ih_c, false);
    const int64_t H = hidden_size_;
    auto r_i = ::tenzor::slice(gates_ih_var, 1, 0,     H);
    auto z_i = ::tenzor::slice(gates_ih_var, 1, H,   2*H);
    auto n_i = ::tenzor::slice(gates_ih_var, 1, 2*H, 3*H);
    auto r_h = ::tenzor::slice(gates_hh,     1, 0,     H);
    auto z_h = ::tenzor::slice(gates_hh,     1, H,   2*H);

    // Gates. Variable + Variable + sigmoid/tanh keeps the graph connected.
    auto r_t = nn::sigmoid(r_i + r_h);
    auto z_t = nn::sigmoid(z_i + z_h);

    // New gate — same mode-0/mode-1 split as GRUCell::forward (see the comment
    // there). mode 0 re-runs weight_hh_ on (r_t ⊙ H) so the reset gate is
    // applied before the recurrent matmul and the recurrent new-gate bias is
    // added after it.
    Variable n_t;
    if (linear_before_reset_) {
        auto n_h = ::tenzor::slice(gates_hh, 1, 2*H, 3*H);
        n_t = nn::tanh(n_i + r_t * n_h);
    } else {
        auto gates_hh_reset = weight_hh_->forward(r_t * h);  // (batch, 3*H)
        auto n_h0 = ::tenzor::slice(gates_hh_reset, 1, 2*H, 3*H);
        n_t = nn::tanh(n_i + n_h0);
    }

    // h_new = (1 - z_t) * n_t + z_t * h   ≡   n_t + z_t * (h - n_t)
    auto h_new = n_t + z_t * (h - n_t);

    // Narrow the new hidden state back to the caller's original dtype.
    return needs_upcast ? variable_cast(h_new, orig_dtype) : h_new;
}

// ============================================================================
// GRU Implementation
// ============================================================================

GRU::GRU(int64_t input_size, int64_t hidden_size, int64_t num_layers,
         bool bias, bool batch_first, double dropout, bool bidirectional,
         bool linear_before_reset)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      num_layers_(num_layers),
      batch_first_(batch_first),
      bidirectional_(bidirectional),
      dropout_p_(dropout),
      linear_before_reset_(linear_before_reset) {

    if (num_layers < 1) {
        throw std::invalid_argument("GRU: num_layers must be >= 1");
    }
    if (dropout < 0.0 || dropout > 1.0) {
        throw std::invalid_argument("GRU: dropout must be in [0, 1]");
    }

    // Create forward cells
    for (int64_t i = 0; i < num_layers; ++i) {
        int64_t layer_input_size = (i == 0) ? input_size : hidden_size * (bidirectional_ ? 2 : 1);
        auto cell = std::make_shared<GRUCell>(layer_input_size, hidden_size, bias,
                                              linear_before_reset);
        forward_cells_.push_back(cell);
        register_module("forward_cell_" + std::to_string(i), cell);
    }

    // Create backward cells for bidirectional GRU
    if (bidirectional_) {
        for (int64_t i = 0; i < num_layers; ++i) {
            int64_t layer_input_size = (i == 0) ? input_size : hidden_size * 2;
            auto cell = std::make_shared<GRUCell>(layer_input_size, hidden_size, bias,
                                              linear_before_reset);
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

// Backward node for the fused cuDNN GRU training path. GRU has no cell state,
// so output is the only node that needs a grad_fn; h_n is an autograd-aware
// slice of output (its gradient routes into output's last timestep, and cuDNN
// sums dy[last]+dhy internally). Mirrors CudnnLSTMTrainBackward.
class CudnnGRUTrainBackward : public Function {
public:
    CudnnGRUTrainBackward(bool has_bias_ih, bool has_bias_hh,
                          std::vector<Tensor> saved)
        : has_bias_ih_(has_bias_ih), has_bias_hh_(has_bias_hh) {
        save_for_backward(std::move(saved));
    }
    auto forward(std::vector<Variable>) -> std::vector<Variable> override {
        throw std::runtime_error("CudnnGRUTrainBackward::forward should not be called");
    }
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto sv = saved_tensors();
        // saved: [input,h0,W_ih,W_hh,b_ih,b_hh,output,weight_space,reserve_space]
        const Tensor& input = sv[0]; const Tensor& h0 = sv[1];
        const Tensor& W_ih = sv[2]; const Tensor& W_hh = sv[3];
        const Tensor& b_ih = sv[4]; const Tensor& b_hh = sv[5];
        const Tensor& output = sv[6]; const Tensor& weight_space = sv[7];
        const Tensor& reserve_space = sv[8];
        const Tensor& g = grad_outputs[0];
        Tensor zero0 = empty({0}, output.dtype(), output.device());
        std::vector<Tensor> ins = {g, zero0, input, h0, output,
                                   weight_space, reserve_space,
                                   W_ih, W_hh, b_ih, b_hh};
        auto grads = dispatch<OpId::GRUCudnnBackward>(ins);
        // grads: [grad_input,grad_hx,grad_W_ih,grad_W_hh,grad_b_ih,grad_b_hh]
        // Order must mirror the forward wiring of next_functions/input_variables:
        // [input, W_ih, W_hh, (b_ih if present), (b_hh if present)].
        std::vector<Tensor> result = {grads[0], grads[2], grads[3]};
        if (has_bias_ih_) result.push_back(grads[4]);
        if (has_bias_hh_) result.push_back(grads[5]);
        return result;
    }
private:
    bool has_bias_ih_;
    bool has_bias_hh_;
};

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
    // F.2: GRU bias-semantics alignment.
    // The fused per-backend GRU kernels historically diverged from PyTorch
    // on bias application: CUDA's gru_cell_fused_kernel collapsed
    // bias_ih + bias_hh into a single bias, while CPU oneDNN's vanilla_gru
    // applied bias inside the `r * (...)` form differently than the
    // SIMD-tiled CPU kernel. Either path is *correct GRU math*, but they
    // produce different numerical values from the PyTorch reference, which
    // breaks tests that check checkpoint round-trip equivalence with
    // PyTorch weights.
    //
    // The slow per-timestep GRUCell path uses two independent Linear
    // sublayers (each with its own bias), which matches PyTorch exactly
    // and is verified against PyTorch's reference in our test suite. The
    // performance gap on practical training shapes is small because the
    // per-timestep path still uses MKL/cuBLAS GEMM under the hood; the
    // missing wins are kernel-launch fusion and intermediate-buffer
    // elision.
    //
    // Re-enabling the fused fast-path requires per-backend kernel surgery
    // to take separate (bias_ih, bias_hh) tensors and apply them at the
    // PyTorch-correct positions. Doing that safely needs GPU runtime
    // validation of bit-equivalence against the slow path on every
    // backend — a separate work item that this code path is intentionally
    // gated against.
    // F.2: enable the fused fast-path when grad tracking is globally off
    // (NoGradGuard / eval mode). The fused kernel returns a grad-free
    // output, which is correct for inference but breaks param-grad flow
    // if any leaf requires_grad. Checking `is_grad_enabled()` covers
    // both the input and the GRU's own parameters in one shot.
    // CPU SIMD applies bias_ih / bias_hh at PyTorch-correct positions;
    // CUDA's gru_forward_cuda does the same.
    // Fused inference fast-path: enabled when grad is globally off and the
    // input doesn't require grad (PyTorch no_grad semantics — the grad-free
    // leaf output is correct). The earlier wrong-output bug here was the fused
    // path sourcing W_hh / bias_hh via positional cell->parameters() indices;
    // fixed below by using named accessors (see the layer loop).
    // The fused inference kernels (GRUForward) and the cuDNN training path
    // below only implement the linear_before_reset=1 new-gate math. When the
    // GRU is configured for mode 0 we must fall through to the per-timestep
    // autograd path, whose GRUCell::forward honors linear_before_reset_.
    // NOTE: must exclude bidirectional. The fused fast-path below iterates only
    // forward_cells_ and stacks num_layers states (feature dim = hidden), so a
    // bidirectional GRU would silently run forward-only and produce the wrong
    // output/h_final shapes. Bidirectional must fall through to the per-timestep
    // autograd path that also runs backward_cells_.
    const bool can_use_fused =
        is_op_supported(OpId::GRUForward, input.device().type) &&
        input.dtype() == DType::Float32 &&
        linear_before_reset_ &&
        !bidirectional_ &&
        !x.requires_grad() && !tenzor::is_grad_enabled();

    if (can_use_fused) {
        // JIT-R102 (non-JIT, same review pass): this fused-kernel path reads
        // Parameter tensors directly (bypassing GRUCell::forward, whose own
        // device-alignment above -- `weight_ih_->to(input_device)` -- is what
        // normally handles a lazily-placed module: constructed on CPU, then
        // called with a GPU input, the common workflow). Mirror that same
        // established mutating in-place move here so every raw ->tensor()
        // extraction below already reads an input-device tensor.
        for (auto& cell : forward_cells_) {
            if (cell->weight_ih()->weight()->device() != input.device()) {
                cell->weight_ih()->to(input.device());
            }
            if (cell->weight_hh()->weight()->device() != input.device()) {
                cell->weight_hh()->to(input.device());
            }
        }

        // Prepare input tensor for kernel. The fused GRU kernels (CPU
        // gru_forward_kernel, CUDA gru_forward_cuda, ...) all expect
        // time-major (seq_len, batch, input_size). When batch_first=true
        // the caller-side block above (lines 214-226) has already
        // transposed `x` into time-major via `tenzor::transpose(x, 0, 1)`,
        // so we must NOT transpose again here. The previous duplicate
        // transpose silently re-batch-firsted the input, causing the
        // CUDA gru_forward to misinterpret shape[0] as seq_len when it
        // was actually batch — which surfaced as a "cudaMemcpyAsync
        // invalid argument" at the bench shape (Phase 8.5).
        Tensor layer_input = x.tensor().contiguous();

        // Always process layers sequentially using the per-layer fused
        // GRUForward op. The legacy multi-layer fast path (GRUMultiLayerForward)
        // packed only one bias per layer (bias_ih), dropping bias_hh — a shared
        // bug across backends — so it was permanently disabled and is now
        // removed. The per-layer op packs both biases (6-input convention) and
        // routes to PyTorch-correct kernels in each backend.
        //
        // Process layers sequentially using fused kernels
        std::vector<Tensor> final_h_states;

        for (int64_t layer = 0; layer < num_layers_; ++layer) {
            auto& cell = forward_cells_[layer];
            auto W_ih_linear = cell->weight_ih();
            auto W_hh_linear = cell->weight_hh();

            // Get weight tensors via NAMED accessors. The previous code used
            // positional cell->parameters() indices ([2]=W_hh weight, [3]=W_hh
            // bias), assuming the order [W_ih_w, W_ih_b, W_hh_w, W_hh_b]. That
            // order is not guaranteed, and when it differed the fused path
            // silently fed the wrong tensors as W_hh / bias_hh — producing
            // materially incorrect GRU output. LSTM already uses named
            // accessors; mirror that here.
            auto W_ih_params = W_ih_linear->parameters();
            auto W_hh_params = W_hh_linear->parameters();
            Tensor W_ih_tensor = W_ih_params[0]->tensor().contiguous();
            Tensor W_hh_tensor = W_hh_params[0]->tensor().contiguous();

            Tensor bias_ih_tensor;
            Tensor bias_hh_tensor;
            if (W_ih_params.size() > 1 && W_hh_params.size() > 1) {
                bias_ih_tensor = W_ih_params[1]->tensor().contiguous();
                bias_hh_tensor = W_hh_params[1]->tensor().contiguous();
            } else if (W_ih_params.size() > 1) {
                bias_ih_tensor = W_ih_params[1]->tensor().contiguous();
                bias_hh_tensor = empty({0}, DType::Float32, input.device());
            } else {
                bias_ih_tensor = empty({0}, DType::Float32, input.device());
                bias_hh_tensor = empty({0}, DType::Float32, input.device());
            }

            // Get initial state for this layer
            Tensor h0_layer = traced_reshape(
                traced_slice(h.tensor(), 0, layer, layer + 1),
                {batch_size, hidden_size_}).contiguous();

            // Call fused kernel — 6th input (bias_hh) is consumed by
            // backends that distinguish between bias_ih and bias_hh
            // (Phase 8.5). Backends that take only the 5-input form
            // ignore the extra input.
            std::vector<Tensor> inputs = {layer_input, W_ih_tensor, W_hh_tensor,
                                           bias_ih_tensor, h0_layer,
                                           bias_hh_tensor};
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
            output = Variable(traced_transpose(output.tensor(), 0, 1), false);
        }

        // Stack final states: (num_layers, batch, hidden)
        Variable h_final(stack(std::span<const Tensor>(final_h_states), 0), false);

        return {output, h_final};
    }

    // =========================================================================
    // FUSED cuDNN TRAINING PATH (unidirectional, Float32/CUDA). Verified vs
    // PyTorch. Enabled by default; set TENZOR_DISABLE_FUSED_GRU_TRAIN to fall
    // back to the per-timestep autograd loop. Multi-layer chains the verified
    // single-layer fused step per layer (output_var feeds the next layer);
    // h_n is an autograd-aware slice of each layer's output. Mirrors the LSTM
    // fused training path; GRU has no cell state.
    // =========================================================================
    if (std::getenv("TENZOR_DISABLE_FUSED_GRU_TRAIN") == nullptr &&
        linear_before_reset_ &&
        is_op_supported(OpId::GRUCudnnTrainForward, input.device().type) &&
        input.dtype() == DType::Float32 &&
        !bidirectional_ &&
        tenzor::is_grad_enabled()) {

        const int64_t kb = x.shape()[1];
        std::vector<Variable> h_n_layers;
        h_n_layers.reserve(static_cast<size_t>(num_layers_));

        Variable layer_in = x;  // (seq, batch, feat) seq-major; later (seq, batch, hidden)
        for (int64_t layer = 0; layer < num_layers_; ++layer) {
            auto& cell = forward_cells_[layer];
            // JIT-R102 (non-JIT, same review pass): mirrors the LSTM fused
            // cuDNN training path fix -- see lstm.cpp's identical block for
            // the full rationale. to_device() is autograd-aware, so
            // gradients still flow back to the real, original-device
            // Parameter through the extra device-transfer node when a
            // transfer actually happens.
            Variable W_ih_v = *cell->weight_ih()->weight();
            Variable W_hh_v = *cell->weight_hh()->weight();
            if (W_ih_v.tensor().device() != input.tensor().device()) {
                W_ih_v = tenzor::to_device(W_ih_v, input.tensor().device());
            }
            if (W_hh_v.tensor().device() != input.tensor().device()) {
                W_hh_v = tenzor::to_device(W_hh_v, input.tensor().device());
            }
            const bool has_bias_ih = cell->weight_ih()->has_bias();
            const bool has_bias_hh = cell->weight_hh()->has_bias();
            Variable b_ih_v, b_hh_v;
            Tensor b_ih_t, b_hh_t;
            if (has_bias_ih) {
                b_ih_v = *cell->weight_ih()->bias();
                if (b_ih_v.tensor().device() != input.tensor().device()) {
                    b_ih_v = tenzor::to_device(b_ih_v, input.tensor().device());
                }
                b_ih_t = b_ih_v.tensor();
            } else {
                b_ih_t = empty({0}, DType::Float32, input.device());
            }
            if (has_bias_hh) {
                b_hh_v = *cell->weight_hh()->bias();
                if (b_hh_v.tensor().device() != input.tensor().device()) {
                    b_hh_v = tenzor::to_device(b_hh_v, input.tensor().device());
                }
                b_hh_t = b_hh_v.tensor();
            } else {
                b_hh_t = empty({0}, DType::Float32, input.device());
            }

            Tensor x_t = layer_in.tensor().contiguous();  // (seq, batch, in)
            Tensor h0_t = traced_reshape(
                traced_slice(h.tensor(), 0, layer, layer + 1),
                {kb, hidden_size_}).contiguous();

            std::vector<Tensor> fwd_in = {x_t, h0_t,
                                          W_ih_v.tensor(), W_hh_v.tensor(),
                                          b_ih_t, b_hh_t};
            auto outs = dispatch<OpId::GRUCudnnTrainForward>(fwd_in);
            Tensor output_t = outs[0];      // (seq, batch, hidden)
            Tensor reserve  = outs[2];
            Tensor wspace   = outs[3];

            std::vector<Tensor> saved = {x_t, h0_t,
                                         W_ih_v.tensor(), W_hh_v.tensor(),
                                         b_ih_t, b_hh_t, output_t, wspace, reserve};

            std::vector<std::shared_ptr<Function>> next_funcs = {
                layer_in.grad_fn(), W_ih_v.grad_fn(), W_hh_v.grad_fn()};
            std::vector<Variable> in_vars = {layer_in, W_ih_v, W_hh_v};
            if (has_bias_ih) {
                next_funcs.push_back(b_ih_v.grad_fn());
                in_vars.push_back(b_ih_v);
            }
            if (has_bias_hh) {
                next_funcs.push_back(b_hh_v.grad_fn());
                in_vars.push_back(b_hh_v);
            }

            auto gfn_out = std::make_shared<CudnnGRUTrainBackward>(has_bias_ih, has_bias_hh, saved);
            Variable output_var(output_t, true);
            gfn_out->set_next_functions(next_funcs);
            gfn_out->set_input_variables(in_vars);
            output_var.set_grad_fn(gfn_out);

            h_n_layers.push_back(::tenzor::slice(output_var, 0, seq_len - 1, seq_len));
            layer_in = output_var;
        }

        Variable output_var = layer_in;
        Variable output_out = batch_first_ ? ::tenzor::transpose(output_var, 0, 1) : output_var;
        Variable h_n = num_layers_ == 1 ? h_n_layers[0] : ::tenzor::cat(h_n_layers, 0);
        return {output_out, h_n};
    }

    // =========================================================================
    // STANDARD PATH: Autograd-correct per-timestep forward.
    // Mirror LSTM standard path — see lstm.cpp for rationale.
    //
    // Same v0.1 known-perf limitation as LSTM: this autograd path is
    // materially slower than PyTorch's cuDNN RNN training kernel on CUDA.
    // cuDNN RNN integration is on the v0.2 roadmap.
    // =========================================================================
    if (input.device().type == Device::Type::CUDA) {
        static std::once_flag warned;
        std::call_once(warned, []() {
            // Audit I.4: unified logger so TENZOR_LOG_LEVEL applies.
            TENZOR_LOG_WARN("CUDA GRU training uses the per-timestep autograd "
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
