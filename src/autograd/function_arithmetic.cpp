#include "tenzor/autograd/function.hpp"
#include "function_helpers.hpp"
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

// AddBackward implementation
auto AddBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Save input shapes for broadcasting-aware backward pass
    input_shape_a_ = std::vector<int64_t>(inputs[0].shape().begin(), inputs[0].shape().end());
    input_shape_b_ = std::vector<int64_t>(inputs[1].shape().begin(), inputs[1].shape().end());

    auto result = add(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto AddBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // Fast path: same shape (common case for residual connections)
    // Avoids function call overhead and vector comparisons
    if (input_shape_a_ == input_shape_b_) {
        auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
        if (grad_shape == input_shape_a_) {
            // Both inputs have same shape as gradient - just return gradient twice
            return {grad, grad};
        }
    }

    // Slow path: handle broadcasting
    auto grad_a = reduce_grad_for_broadcasting(grad, input_shape_a_);
    auto grad_b = reduce_grad_for_broadcasting(grad, input_shape_b_);
    return {grad_a, grad_b};
}

auto AddBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    const auto& grad = grad_outputs[0];

    // For add: gradients are just the incoming gradient (possibly with broadcast reduction)
    // Since these are identity operations on the gradient, the Variable grad already has
    // its grad_fn set, so higher-order gradients flow through naturally
    if (input_shape_a_ == input_shape_b_) {
        auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
        if (grad_shape == input_shape_a_) {
            return {grad, grad};
        }
    }

    auto grad_a = reduce_grad_var_for_broadcasting(grad, input_shape_a_);
    auto grad_b = reduce_grad_var_for_broadcasting(grad, input_shape_b_);
    return {grad_a, grad_b};
}

// SubBackward implementation
auto SubBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Save input shapes for broadcasting-aware backward pass
    input_shape_a_ = std::vector<int64_t>(inputs[0].shape().begin(), inputs[0].shape().end());
    input_shape_b_ = std::vector<int64_t>(inputs[1].shape().begin(), inputs[1].shape().end());

    auto result = sub(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto SubBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(a-b)/da = 1, d(a-b)/db = -1
    // Handle broadcasting
    auto grad_a = reduce_grad_for_broadcasting(grad_outputs[0], input_shape_a_);
    auto grad_b_unreduced = neg(grad_outputs[0]);
    auto grad_b = reduce_grad_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

auto SubBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(a-b)/da = 1, d(a-b)/db = -1
    // Use Variable operations so negation is tracked for higher-order gradients
    auto grad_a = reduce_grad_var_for_broadcasting(grad_outputs[0], input_shape_a_);
    auto grad_b_unreduced = tenzor::neg(grad_outputs[0]);  // Variable neg - tracked by autograd
    auto grad_b = reduce_grad_var_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

// MulBackward implementation
auto MulBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor(), inputs[1].tensor()});
    // Save input shapes for broadcasting-aware backward pass
    input_shape_a_ = std::vector<int64_t>(inputs[0].shape().begin(), inputs[0].shape().end());
    input_shape_b_ = std::vector<int64_t>(inputs[1].shape().begin(), inputs[1].shape().end());

    auto result = mul(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto MulBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    // d(a*b)/da = b, d(a*b)/db = a
    // Handle broadcasting
    auto grad_a_unreduced = mul(grad_outputs[0], saved_tensors_[1]);
    auto grad_b_unreduced = mul(grad_outputs[0], saved_tensors_[0]);

    auto grad_a = reduce_grad_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_for_broadcasting(grad_b_unreduced, input_shape_b_);

    return {grad_a, grad_b};
}

auto MulBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(a*b)/da = b, d(a*b)/db = a
    // Use saved Variables if available, otherwise wrap saved Tensors
    Variable saved_a, saved_b;
    if (has_saved_variables()) {
        require_saved_variables(2);
        saved_a = saved_variables_[0];
        saved_b = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        saved_a = Variable(saved_tensors_[0], false);
        saved_b = Variable(saved_tensors_[1], false);
    }

    // Use Variable operations so multiplication is tracked for higher-order gradients
    auto grad_a_unreduced = grad_outputs[0] * saved_b;
    auto grad_b_unreduced = grad_outputs[0] * saved_a;

    auto grad_a = reduce_grad_var_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_var_for_broadcasting(grad_b_unreduced, input_shape_b_);

    return {grad_a, grad_b};
}

// DivBackward implementation
auto DivBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor(), inputs[1].tensor()});
    // Save input shapes for broadcasting-aware backward pass
    input_shape_a_ = std::vector<int64_t>(inputs[0].shape().begin(), inputs[0].shape().end());
    input_shape_b_ = std::vector<int64_t>(inputs[1].shape().begin(), inputs[1].shape().end());

    auto result = div(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto DivBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    // d(a/b)/da = 1/b, d(a/b)/db = -a/(b^2)
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];

    // Zero-safe: replace zero denominator with epsilon to avoid NaN/Inf
    auto zero_b = zeros(std::vector<int64_t>(b.shape().begin(), b.shape().end()),
                        b.dtype(), b.device());
    auto eps_b = full(std::vector<int64_t>(b.shape().begin(), b.shape().end()),
                      detail::dtype_epsilon(b.dtype()), b.dtype(), b.device());
    auto safe_b = where(eq(b, zero_b), eps_b, b);

    auto grad_a_unreduced = div(grad_outputs[0], safe_b);
    // grad_b = -a / (b^2) * grad_output = -(a * grad_output) / (b * b)
    auto grad_b_unreduced = neg(div(mul(a, grad_outputs[0]), mul(safe_b, safe_b)));

    auto grad_a = reduce_grad_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

auto DivBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(a/b)/da = 1/b, d(a/b)/db = -a/(b^2)
    Variable saved_a, saved_b;
    if (has_saved_variables()) {
        require_saved_variables(2);
        saved_a = saved_variables_[0];
        saved_b = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        saved_a = Variable(saved_tensors_[0], false);
        saved_b = Variable(saved_tensors_[1], false);
    }

    // Zero-safe: replace zero denominator with epsilon (same guard as backward())
    auto b_tensor = saved_b.tensor();
    auto b_shape = std::vector<int64_t>(b_tensor.shape().begin(), b_tensor.shape().end());
    auto zero_b = zeros(b_shape, b_tensor.dtype(), b_tensor.device());
    auto eps_b = full(b_shape, detail::dtype_epsilon(b_tensor.dtype()),
                      b_tensor.dtype(), b_tensor.device());
    auto safe_b_tensor = where(eq(b_tensor, zero_b), eps_b, b_tensor);
    Variable safe_b(safe_b_tensor, false);

    // Use Variable operations for higher-order gradient tracking
    auto grad_a_unreduced = grad_outputs[0] / safe_b;
    // grad_b = -(a * grad_output) / (b * b)
    auto grad_b_unreduced = tenzor::neg((saved_a * grad_outputs[0]) / (safe_b * safe_b));

    auto grad_a = reduce_grad_var_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_var_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

// MatMulBackward implementation
auto MatMulBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor(), inputs[1].tensor()});
    auto result = matmul(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto MatMulBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    // For C = A @ B:
    // dL/dA = dL/dC @ B.T
    // dL/dB = A.T @ dL/dC
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& grad_out = grad_outputs[0];

    // Get the number of dimensions
    auto a_ndim = a.shape().size();
    auto b_ndim = b.shape().size();

    // For 2D matrices: grad_a = grad_out @ b.T, grad_b = a.T @ grad_out
    auto b_t = transpose(b, b_ndim - 2, b_ndim - 1);
    auto a_t = transpose(a, a_ndim - 2, a_ndim - 1);

    auto grad_a = matmul(grad_out, b_t);
    auto grad_b = matmul(a_t, grad_out);

    return {grad_a, grad_b};
}

auto MatMulBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // For C = A @ B:
    // dL/dA = dL/dC @ B.T
    // dL/dB = A.T @ dL/dC
    Variable saved_a, saved_b;
    if (has_saved_variables()) {
        require_saved_variables(2);
        saved_a = saved_variables_[0];
        saved_b = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        saved_a = Variable(saved_tensors_[0], false);
        saved_b = Variable(saved_tensors_[1], false);
    }

    const auto& grad_out = grad_outputs[0];

    // Use Variable operations for higher-order gradient tracking
    auto a_ndim = saved_a.shape().size();
    auto b_ndim = saved_b.shape().size();

    auto b_t = tenzor::transpose(saved_b, b_ndim - 2, b_ndim - 1);
    auto a_t = tenzor::transpose(saved_a, a_ndim - 2, a_ndim - 1);

    auto grad_a = tenzor::matmul(grad_out, b_t);
    auto grad_b = tenzor::matmul(a_t, grad_out);

    return {grad_a, grad_b};
}

// LinearBackward implementation
auto LinearBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // This is typically not called - autograd::linear() handles forward directly
    // But implement for completeness
    // inputs[0] = x (batch_size, in_features)
    // inputs[1] = W (out_features, in_features)
    // inputs[2] = b (out_features)
    save_for_backward({inputs[0].tensor(), inputs[1].tensor(), inputs[2].tensor()});

    // Compute: y = x @ W.T + b
    auto x = inputs[0].tensor();
    auto w = inputs[1].tensor();
    auto b = inputs[2].tensor();

    // Transpose weight: (out_features, in_features) -> (in_features, out_features)
    auto w_t = transpose(w, 0, 1);

    // Matrix multiplication: (batch, in) @ (in, out) -> (batch, out)
    auto matmul_result = matmul(x, w_t);

    // Add bias (broadcasts automatically)
    auto result = add(matmul_result, b);

    return {Variable(result, true)};
}

auto LinearBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(3);
    // For y = x @ W.T + b:
    // dL/dx = dL/dy @ W          -> (batch, out) @ (out, in) = (batch, in)
    // dL/dW = dL/dy.T @ x        -> (out, batch) @ (batch, in) = (out, in)
    // dL/db = sum(dL/dy, dim=0)  -> reduce batch dimension

    const auto& x = saved_tensors_[0];       // (batch, in)
    const auto& w = saved_tensors_[1];       // (out, in)
    const auto& grad_out = grad_outputs[0];  // (batch, out)

    // Use optimized LinearBackward kernel (supports Float32, Float64, Float16, BFloat16)
    bool is_gpu = (grad_out.device().type != Device::Type::CPU);
    if (is_gpu ||
        grad_out.dtype() == DType::Float32 || grad_out.dtype() == DType::Float64) {
        // For Float16/BFloat16 on GPU backends, upcast to Float32 for computation to
        // prevent gradient overflow. FP16 gemm can't represent values > 65504,
        // causing Inf→NaN propagation in larger models.
        DType orig_dt = grad_out.dtype();
        bool needs_upcast = (is_gpu &&
                            (orig_dt == DType::Float16 || orig_dt == DType::BFloat16));
        if (needs_upcast) {
            std::vector<Tensor> inputs = {
                grad_out.to(DType::Float32),
                x.to(DType::Float32),
                w.to(DType::Float32)
            };
            auto results = dispatch<OpId::LinearBackward>(inputs);
            for (auto& r : results) r = r.to(orig_dt);
            return results;
        }
        // Ensure all inputs match grad_out dtype (mixed precision: e.g. Float64
        // input with Float32 weight) — CUDA kernels select kernel by first tensor dtype
        auto x_cast = (x.dtype() != orig_dt) ? x.to(orig_dt) : x;
        auto w_cast = (w.dtype() != orig_dt) ? w.to(orig_dt) : w;
        std::vector<Tensor> inputs = {grad_out, x_cast, w_cast};
        return dispatch<OpId::LinearBackward>(inputs);
    }

    // Fallback for other backends/types using tensor operations
    // grad_input = grad_out @ W
    auto grad_x = matmul(grad_out, w);

    // grad_weight = grad_out.T @ x
    auto grad_out_t = transpose(grad_out, 0, 1);  // (out, batch)
    auto grad_w = matmul(grad_out_t, x);          // (out, in)

    // grad_bias = sum(grad_out, dim=0)
    auto grad_b = tenzor::sum(grad_out, 0, false);  // (out,)

    return {grad_x, grad_w, grad_b};
}

auto LinearBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // For y = x @ W.T + b:
    // dL/dx = dL/dy @ W
    // dL/dW = dL/dy.T @ x
    // dL/db = sum(dL/dy, dim=0)
    Variable saved_x, saved_w;
    if (has_saved_variables()) {
        require_saved_variables(2);
        saved_x = saved_variables_[0];
        saved_w = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        saved_x = Variable(saved_tensors_[0], false);
        saved_w = Variable(saved_tensors_[1], false);
    }
    const auto& grad_out = grad_outputs[0];

    // For Float16/BFloat16 on GPU backends, upcast to Float32 to prevent
    // gradient overflow (FP16 max ~65504, easily exceeded in backward).
    DType orig_dt = grad_out.tensor().dtype();
    bool is_gpu = (grad_out.tensor().device().type != Device::Type::CPU);
    bool needs_upcast = is_gpu && (orig_dt == DType::Float16 || orig_dt == DType::BFloat16);

    Variable go = needs_upcast ? Variable(grad_out.tensor().to(DType::Float32), false) : grad_out;
    Variable sx = needs_upcast ? Variable(saved_x.tensor().to(DType::Float32), false) : saved_x;
    Variable sw = needs_upcast ? Variable(saved_w.tensor().to(DType::Float32), false) : saved_w;

    auto grad_x = tenzor::matmul(go, sw);
    auto grad_out_t = tenzor::transpose(go, 0, 1);
    auto grad_w = tenzor::matmul(grad_out_t, sx);
    auto grad_b = tenzor::sum(go, 0, false);

    if (needs_upcast) {
        grad_x = Variable(grad_x.tensor().to(orig_dt), grad_x.requires_grad());
        grad_w = Variable(grad_w.tensor().to(orig_dt), grad_w.requires_grad());
        grad_b = Variable(grad_b.tensor().to(orig_dt), grad_b.requires_grad());
    }
    return {grad_x, grad_w, grad_b};
}

} // namespace tenzor
