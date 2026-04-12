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
    std::vector<Tensor> tensor_grads;
    tensor_grads.reserve(grad_outputs.size());
    for (auto& var : grad_outputs) {
        tensor_grads.push_back(var.tensor());
    }

    auto result_tensors = backward(tensor_grads);

    std::vector<Variable> result_vars;
    result_vars.reserve(result_tensors.size());
    for (auto& t : result_tensors) {
        result_vars.emplace_back(t, true);
    }
    return result_vars;
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
    // Extract tensors for the user's backward function
    std::vector<Tensor> tensor_grads;
    tensor_grads.reserve(grad_outputs.size());
    for (auto& var : grad_outputs) {
        tensor_grads.push_back(var.tensor());
    }

    validate_saved_tensors();
    auto result_tensors = backward_fn_(saved_tensors_, tensor_grads);

    // Wrap results as Variables with requires_grad=true so the gradient
    // graph continues through the custom op for higher-order derivatives
    std::vector<Variable> result_vars;
    result_vars.reserve(result_tensors.size());
    for (auto& t : result_tensors) {
        result_vars.emplace_back(t, true);
    }
    return result_vars;
}

} // namespace tenzor
