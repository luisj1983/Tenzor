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

    // grad_input = masked_grad @ W (for the input gradient)
    auto grad_input = matmul(masked_grad, weight);

    // grad_weight = input^T @ masked_grad (for the weight gradient)
    auto input_t = input.transpose(input.ndim() - 2, input.ndim() - 1);
    auto grad_weight = matmul(input_t, masked_grad);

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
    // Forward:  z = ReLU(x @ W.T + b)  (PyTorch convention: W shape [out, in])
    //           but this op stores W as [out, in] and computes
    //           grad_input = (grad_out * mask) @ W   directly.
    // Backward (linear in grad_out):
    //   masked_grad = grad_out * (z > 0)
    //   grad_input  = masked_grad @ W
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

    // grad_input = masked_grad @ W
    Variable grad_input = matmul(masked_grad, W_var);

    // grad_weight = x^T @ masked_grad
    int64_t xnd = static_cast<int64_t>(x_var.shape().size());
    Variable x_t = transpose(x_var, xnd - 2, xnd - 1);
    Variable grad_weight = matmul(x_t, masked_grad);

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
