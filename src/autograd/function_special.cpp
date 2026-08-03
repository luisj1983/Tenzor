#include "tenzor/autograd/function.hpp"
#include <cassert>
#include "tenzor/autograd/ops.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/safe_math.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <typeinfo>
#include <unordered_set>
#ifdef __GNUC__
#include <cxxabi.h>
#include <cstdlib>
#endif

namespace tenzor {

// ============================================================================
// Fused Linear+ReLU backward implementation
// ============================================================================

auto FusedLinearReLUBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("FusedLinearReLUBackward::forward() should not be called directly");
}

auto FusedLinearReLUBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    validate_saved_tensors();

    // saved_tensors_[0] = input (x), saved_tensors_[1] = weight (W)
    require_saved_tensors(2);
    auto& input = saved_tensors_[0];
    auto& weight = saved_tensors_[1];

    auto& grad_output = grad_outputs[0];

    // Apply ReLU mask: grad_output * (relu_output > 0)
    auto relu_shape = std::vector<int64_t>(relu_output_.shape().begin(), relu_output_.shape().end());
    auto zero_tensor = full(relu_shape, 0.0f, relu_output_.dtype(), relu_output_.device());
    auto relu_mask = gt(relu_output_, zero_tensor);
    auto masked_grad = mul(grad_output, relu_mask.to(grad_output.dtype()));

    // grad_input = masked_grad @ W (for the input gradient).
    // The fused node replaces MatMul(out = x @ W^T) -> ReLU, where the weight
    // W is stored in the PyTorch nn.Linear convention W=[OUT,IN] (see
    // fused_linear_relu() in src/ops/fused_ops.cpp: out_features=W.shape()[0],
    // and the forward is out = x @ W^T). For out = x @ W^T the input gradient
    // is grad_x = grad_out @ (W^T)^T = grad_out @ W. The previous code
    // transposed W (computing grad_out @ W^T), which is a shape mismatch for
    // non-square W and a silently-wrong gradient for square W.
    auto grad_input = matmul(masked_grad, weight);

    // grad_weight = masked_grad^T @ x (for the weight gradient).
    // For out = x @ W^T, grad_{W^T} = x^T @ grad_out, so
    // grad_W = (x^T @ grad_out)^T = grad_out^T @ x = [OUT,B]@[B,IN] = [OUT,IN],
    // matching W's shape.
    auto masked_grad_t = masked_grad.transpose(masked_grad.ndim() - 2, masked_grad.ndim() - 1);
    auto grad_weight = matmul(masked_grad_t, input);

    std::vector<Tensor> result;
    result.push_back(std::move(grad_input));
    result.push_back(std::move(grad_weight));
    return result;
}

auto FusedLinearReLUBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Audit D1: real Variable-level backward. The previous body called the
    // tensor backward then rewrapped results as `Variable(t, true)` with no
    // grad_fn — silently severing the graph for `create_graph=true`. Now we
    // compose with autograd-level ops so higher-order grads flow through.
    //
    // Forward:  z = ReLU(x @ W)
    // Backward (linear in grad_out):
    //   masked_grad = grad_out * (z > 0)
    //   grad_input  = masked_grad @ W^T   (input gradient for out = x @ W)
    //   grad_weight = x^T @ masked_grad
    //
    // The ReLU mask `(z > 0)` is a non-differentiable constant w.r.t. the
    // graph, so we materialize it at tensor level and wrap as a non-grad
    // Variable. The arithmetic flows through `Variable::operator*` /
    // `autograd::matmul` and preserves grad_fn for higher-order.
    validate_saved_tensors();
    require_saved_tensors(2);

    const auto& grad_out = grad_outputs[0];
    const Tensor& grad_out_t = grad_out.tensor();

    // Build the ReLU mask in grad_out's dtype/device. Non-differentiable.
    auto relu_shape = std::vector<int64_t>(relu_output_.shape().begin(),
                                            relu_output_.shape().end());
    auto zero_tensor = full(relu_shape, 0.0f, relu_output_.dtype(),
                             relu_output_.device());
    auto mask_t = gt(relu_output_, zero_tensor).to(grad_out_t.dtype());
    Variable mask_var(mask_t, /*requires_grad=*/false);

    // Recover x and W as non-grad Variables. saved_tensors_[0] = input (x),
    // saved_tensors_[1] = weight (W).
    Variable x_var(saved_tensors_[0], /*requires_grad=*/false);
    Variable W_var(saved_tensors_[1], /*requires_grad=*/false);

    // masked_grad has grad_fn through grad_out (mask is a constant Variable).
    Variable masked_grad = grad_out * mask_var;

    // grad_input = masked_grad @ W. The forward is out = x @ W^T with W stored
    // in the nn.Linear convention W=[OUT,IN] (see fused_linear_relu() in
    // src/ops/fused_ops.cpp), so grad_x = grad_out @ (W^T)^T = grad_out @ W.
    // Done at the Variable level so the grad_fn chain is preserved for
    // higher-order (create_graph=true). (The prior code transposed W here,
    // which mismatched shapes for non-square W and was silently wrong for
    // square W.)
    Variable grad_input = matmul(masked_grad, W_var);

    // grad_weight = masked_grad^T @ x = [OUT,B]@[B,IN] = [OUT,IN], matching W.
    int64_t mnd = static_cast<int64_t>(masked_grad.shape().size());
    Variable masked_grad_t = transpose(masked_grad, mnd - 2, mnd - 1);
    Variable grad_weight = matmul(masked_grad_t, x_var);

    return {grad_input, grad_weight};
}

// ============================================================================
// Custom Op backward implementation
// ============================================================================

auto CustomOpBackward::forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> {
    // CustomOpBackward is only used as a grad_fn — forward() is never called
    // through this class. The actual forward is dispatched via CustomOpRegistry.
    throw std::runtime_error("CustomOpBackward::forward() should not be called directly");
}

auto CustomOpBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    validate_saved_tensors();
    return backward_fn_(saved_tensors_, grad_outputs);
}

auto CustomOpBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Audit D2: honest higher-order handling.
    //
    // If the user registered a Variable-level backward, call it directly so
    // the user's gradient computation composes with the autograd graph and
    // higher-order derivatives flow correctly.
    //
    // If only a tensor-level backward was registered, we cannot synthesize a
    // Variable-level graph from a raw-tensor callback — the prior code did
    // `Variable(t, /*requires_grad=*/true)` without a `grad_fn`, silently
    // severing the graph. Now we still run the tensor backward (so first-
    // order grads keep working) but report `is_higher_order_stub() = true`
    // so the engine's Warn/Error mode fires correctly. Results are returned
    // as non-grad Variables, matching the engine's contract for stubs.
    validate_saved_tensors();

    if (var_backward_fn_) {
        // Wrap saved tensors as non-grad Variables for the user's backward.
        std::vector<Variable> saved_vars;
        saved_vars.reserve(saved_tensors_.size());
        for (const auto& t : saved_tensors_) {
            saved_vars.emplace_back(t, /*requires_grad=*/false);
        }
        return var_backward_fn_(saved_vars, grad_outputs);
    }

    // Legacy / tensor-only path: invoke the raw-tensor backward, return
    // results without a grad_fn. The engine's stub-detection path will
    // increment its disconnection counter for this op.
    std::vector<Tensor> tensor_grads;
    tensor_grads.reserve(grad_outputs.size());
    for (auto& var : grad_outputs) {
        tensor_grads.push_back(var.tensor());
    }
    auto result_tensors = backward_fn_(saved_tensors_, tensor_grads);

    std::vector<Variable> result_vars;
    result_vars.reserve(result_tensors.size());
    for (auto& t : result_tensors) {
        result_vars.emplace_back(t, /*requires_grad=*/false);
    }
    return result_vars;
}

} // namespace tenzor
