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
#include "tenzor/nn/utils/variable_cast.hpp"
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
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    Tensor grad_a_unreduced, grad_b_unreduced;
    if (a.is_complex() || b.is_complex()) {
        // Wirtinger derivative: d/d(conj(a)) (a*b) = conj(b) * grad
        grad_a_unreduced = mul(grad, conj(b));
        grad_b_unreduced = mul(grad, conj(a));
    } else {
        grad_a_unreduced = mul(grad, b);
        grad_b_unreduced = mul(grad, a);
    }

    auto grad_a = reduce_grad_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_for_broadcasting(grad_b_unreduced, input_shape_b_);

    return {grad_a, grad_b};
}

auto MulBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
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

    Variable grad_a_unreduced, grad_b_unreduced;
    if (saved_a.tensor().is_complex() || saved_b.tensor().is_complex()) {
        // Wirtinger: route conj through the Variable-level overload so the
        // grad_fn carried on saved_a / saved_b (when has_saved_variables())
        // survives. Raw `Variable(conj(saved_b.tensor()), false)` severed
        // the chain — audit-5 X.5 / Y.9.
        auto conj_b = tenzor::conj(saved_b);
        auto conj_a = tenzor::conj(saved_a);
        grad_a_unreduced = grad_outputs[0] * conj_b;
        grad_b_unreduced = grad_outputs[0] * conj_a;
    } else {
        grad_a_unreduced = grad_outputs[0] * saved_b;
        grad_b_unreduced = grad_outputs[0] * saved_a;
    }

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
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    // Zero-safe: replace zero denominator with epsilon to avoid NaN/Inf
    auto zero_b = zeros(std::vector<int64_t>(b.shape().begin(), b.shape().end()),
                        b.dtype(), b.device());
    auto eps_b = full(std::vector<int64_t>(b.shape().begin(), b.shape().end()),
                      detail::dtype_epsilon(b.dtype()), b.dtype(), b.device());
    auto safe_b = where(eq(b, zero_b), eps_b, b);

    Tensor grad_a_unreduced, grad_b_unreduced;
    if (a.is_complex() || b.is_complex()) {
        // Wirtinger: d/d(conj(a)) (a/b) = 1/conj(b)
        //            d/d(conj(b)) (a/b) = -conj(a)/conj(b)^2
        auto conj_b = conj(safe_b);
        grad_a_unreduced = div(grad, conj_b);
        grad_b_unreduced = neg(div(mul(conj(a), grad), mul(conj_b, conj_b)));
    } else {
        grad_a_unreduced = div(grad, safe_b);
        grad_b_unreduced = neg(div(mul(a, grad), mul(safe_b, safe_b)));
    }

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

    // Zero-safe: replace zero denominator with epsilon (same guard as backward()).
    // FF.4: construct `safe_b` via Variable-level ops so the grad_fn carried on
    // `saved_b` (when `has_saved_variables()` is populated) survives into the
    // backward graph — mirrors the audit-5 X.5 / Y.9 fix that was already
    // applied to the complex Wirtinger paths. The previous implementation built
    // `safe_b` from raw Tensor ops and wrapped with `Variable(..., false)`,
    // severing the chain even at b != 0 (real and complex paths alike).
    auto b_tensor_view = saved_b.tensor();
    auto b_shape = std::vector<int64_t>(b_tensor_view.shape().begin(), b_tensor_view.shape().end());
    auto zero_b_t = zeros(b_shape, b_tensor_view.dtype(), b_tensor_view.device());
    auto eps_b_t = full(b_shape, detail::dtype_epsilon(b_tensor_view.dtype()),
                        b_tensor_view.dtype(), b_tensor_view.device());
    Variable zero_b_var(zero_b_t, false);
    Variable eps_b_var(eps_b_t, false);
    auto b_is_zero = ::tenzor::eq(saved_b, zero_b_var);
    auto safe_b = ::tenzor::where(b_is_zero, eps_b_var, saved_b);

    Variable grad_a_unreduced, grad_b_unreduced;
    if (saved_a.tensor().is_complex() || saved_b.tensor().is_complex()) {
        // Wirtinger: route conj through the Variable-level overload so the
        // grad_fn carried on saved_a (and the zero-safe `safe_b`) survives.
        // Raw `Variable(conj(t), false)` severed the chain — audit-5 X.5.
        auto conj_b = tenzor::conj(safe_b);
        auto conj_a = tenzor::conj(saved_a);
        grad_a_unreduced = grad_outputs[0] / conj_b;
        grad_b_unreduced = tenzor::neg((conj_a * grad_outputs[0]) / (conj_b * conj_b));
    } else {
        grad_a_unreduced = grad_outputs[0] / safe_b;
        grad_b_unreduced = tenzor::neg((saved_a * grad_outputs[0]) / (safe_b * safe_b));
    }

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
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& grad_out = grad_outputs[0];

    // Audit-6 AA.8: cast to signed so `ndim - 2` doesn't wrap when ndim == 1.
    // PyTorch's matmul has four broadcast modes that all funnel through this
    // backward — handle them explicitly:
    //   * 1D @ 1D → scalar dot product. grad_a = grad * b, grad_b = grad * a.
    //   * 1D @ ND → treat 1D as a row vector (unsqueeze axis 0), compute the
    //               ND backward, then squeeze axis 0 from grad_a.
    //   * ND @ 1D → treat 1D as a column vector (unsqueeze axis -1), compute
    //               the ND backward, then squeeze axis -1 from grad_b.
    //   * ND @ ND (both rank >= 2) → standard transpose-on-last-two-axes form.
    const int64_t a_ndim = static_cast<int64_t>(a.shape().size());
    const int64_t b_ndim = static_cast<int64_t>(b.shape().size());

    auto conj_if_complex = [](const Tensor& t) {
        return t.is_complex() ? conj(t) : t;
    };

    Tensor grad_a, grad_b;

    if (a_ndim == 1 && b_ndim == 1) {
        // Dot product: y = a · b is a scalar. grad shape matches the scalar.
        // dL/da = grad * b, dL/db = grad * a. (Conjugate `a`/`b` for complex
        // Wirtinger to match the rank-2 branch's `conj(transpose(...))` form.)
        grad_a = mul(grad_out, conj_if_complex(b));
        grad_b = mul(grad_out, conj_if_complex(a));
    } else if (a_ndim == 1) {
        // 1D @ ND. unsqueeze a to (1, K), compute via standard ND form, then
        // squeeze axis -2 from the resulting grad_a.
        auto a2 = unsqueeze(a, 0);                          // (1, K)
        auto grad_out_2 = unsqueeze(grad_out, b_ndim - 2);  // (..., 1, N)
        if (a.is_complex() || b.is_complex()) {
            auto b_ct = conj(transpose(b, b_ndim - 2, b_ndim - 1));
            auto a_ct = conj(transpose(a2, 0, 1));
            auto grad_a_2 = matmul(grad_out_2, b_ct);       // (..., 1, K)
            grad_b = matmul(a_ct, grad_out_2);              // (..., K, N)
            grad_a = squeeze(grad_a_2, b_ndim - 2);
        } else {
            auto b_t = transpose(b, b_ndim - 2, b_ndim - 1);
            auto a_t = transpose(a2, 0, 1);
            auto grad_a_2 = matmul(grad_out_2, b_t);
            grad_b = matmul(a_t, grad_out_2);
            grad_a = squeeze(grad_a_2, b_ndim - 2);
        }
    } else if (b_ndim == 1) {
        // ND @ 1D. unsqueeze b to (K, 1), compute via standard ND form, then
        // squeeze the trailing axis from grad_b.
        auto b2 = unsqueeze(b, 1);                          // (K, 1)
        auto grad_out_2 = unsqueeze(grad_out, a_ndim - 1);  // (..., M, 1)
        if (a.is_complex() || b.is_complex()) {
            auto b_ct = conj(transpose(b2, 0, 1));
            auto a_ct = conj(transpose(a, a_ndim - 2, a_ndim - 1));
            grad_a = matmul(grad_out_2, b_ct);              // (..., M, K)
            auto grad_b_2 = matmul(a_ct, grad_out_2);       // (..., K, 1)
            grad_b = squeeze(grad_b_2, static_cast<int64_t>(grad_b_2.shape().size()) - 1);
        } else {
            auto b_t = transpose(b2, 0, 1);
            auto a_t = transpose(a, a_ndim - 2, a_ndim - 1);
            grad_a = matmul(grad_out_2, b_t);
            auto grad_b_2 = matmul(a_t, grad_out_2);
            grad_b = squeeze(grad_b_2, static_cast<int64_t>(grad_b_2.shape().size()) - 1);
        }
    } else {
        // Both rank >= 2: standard form. `a_ndim - 2` is now safe (signed).
        if (a.is_complex() || b.is_complex()) {
            auto b_ct = conj(transpose(b, b_ndim - 2, b_ndim - 1));
            auto a_ct = conj(transpose(a, a_ndim - 2, a_ndim - 1));
            grad_a = matmul(grad_out, b_ct);
            grad_b = matmul(a_ct, grad_out);
        } else {
            auto b_t = transpose(b, b_ndim - 2, b_ndim - 1);
            auto a_t = transpose(a, a_ndim - 2, a_ndim - 1);
            grad_a = matmul(grad_out, b_t);
            grad_b = matmul(a_t, grad_out);
        }
    }

    // audit-10 MM.1: reduce broadcasted batch axes back to operand shapes.
    // Batched matmul like (B,M,K) @ (K,N) produces grad_b shaped (B,K,N)
    // before reduction.  Every other arithmetic backward already does this;
    // MatMul was the lone holdout.  Skip when input_shape_*_ is empty
    // (legacy call paths that didn't go through the autograd::matmul
    // forward wrapper) so the existing wrong-shape error surfaces as
    // before.
    if (!input_shape_a_.empty()) {
        grad_a = reduce_grad_for_broadcasting(grad_a, input_shape_a_);
    }
    if (!input_shape_b_.empty()) {
        grad_b = reduce_grad_for_broadcasting(grad_b, input_shape_b_);
    }

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

    // Audit-6 AA.8: signed dims + 1D-input special cases, mirroring the
    // Tensor-level backward above.
    const int64_t a_ndim = static_cast<int64_t>(saved_a.shape().size());
    const int64_t b_ndim = static_cast<int64_t>(saved_b.shape().size());
    const bool is_complex = saved_a.tensor().is_complex() ||
                            saved_b.tensor().is_complex();

    Variable grad_a, grad_b;

    if (a_ndim == 1 && b_ndim == 1) {
        // 1D · 1D dot product. grad_a = grad * conj(b), grad_b = grad * conj(a)
        // (Variable * Variable propagates grad_fn for higher-order).
        auto b_for = is_complex ? tenzor::conj(saved_b) : saved_b;
        auto a_for = is_complex ? tenzor::conj(saved_a) : saved_a;
        grad_a = grad_out * b_for;
        grad_b = grad_out * a_for;
    } else if (a_ndim == 1) {
        auto a2 = tenzor::unsqueeze(saved_a, 0);            // (1, K)
        auto grad_out_2 = tenzor::unsqueeze(grad_out, b_ndim - 2);  // (..., 1, N)
        Variable b_t, a_t;
        if (is_complex) {
            b_t = tenzor::conj(tenzor::transpose(saved_b, b_ndim - 2, b_ndim - 1));
            a_t = tenzor::conj(tenzor::transpose(a2, 0, 1));
        } else {
            b_t = tenzor::transpose(saved_b, b_ndim - 2, b_ndim - 1);
            a_t = tenzor::transpose(a2, 0, 1);
        }
        auto grad_a_2 = tenzor::matmul(grad_out_2, b_t);    // (..., 1, K)
        grad_b = tenzor::matmul(a_t, grad_out_2);           // (..., K, N)
        grad_a = tenzor::squeeze(grad_a_2, b_ndim - 2);
    } else if (b_ndim == 1) {
        auto b2 = tenzor::unsqueeze(saved_b, 1);            // (K, 1)
        auto grad_out_2 = tenzor::unsqueeze(grad_out, a_ndim - 1);  // (..., M, 1)
        Variable b_t, a_t;
        if (is_complex) {
            b_t = tenzor::conj(tenzor::transpose(b2, 0, 1));
            a_t = tenzor::conj(tenzor::transpose(saved_a, a_ndim - 2, a_ndim - 1));
        } else {
            b_t = tenzor::transpose(b2, 0, 1);
            a_t = tenzor::transpose(saved_a, a_ndim - 2, a_ndim - 1);
        }
        grad_a = tenzor::matmul(grad_out_2, b_t);           // (..., M, K)
        auto grad_b_2 = tenzor::matmul(a_t, grad_out_2);    // (..., K, 1)
        grad_b = tenzor::squeeze(grad_b_2,
                                 static_cast<int64_t>(grad_b_2.shape().size()) - 1);
    } else if (is_complex) {
        // Wirtinger ND @ ND. Audit-5 X.5 / Y.9: route through Variable
        // overloads so grad_fn on saved inputs survives the second-order
        // pass.
        auto b_t = tenzor::transpose(saved_b, b_ndim - 2, b_ndim - 1);
        auto a_t = tenzor::transpose(saved_a, a_ndim - 2, a_ndim - 1);
        auto b_ct = tenzor::conj(b_t);
        auto a_ct = tenzor::conj(a_t);
        grad_a = tenzor::matmul(grad_out, b_ct);
        grad_b = tenzor::matmul(a_ct, grad_out);
    } else {
        auto b_t = tenzor::transpose(saved_b, b_ndim - 2, b_ndim - 1);
        auto a_t = tenzor::transpose(saved_a, a_ndim - 2, a_ndim - 1);
        grad_a = tenzor::matmul(grad_out, b_t);
        grad_b = tenzor::matmul(a_t, grad_out);
    }

    // audit-10 MM.1: reduce broadcasted batch axes back to operand shapes.
    // See the Tensor-level backward above for the rationale.
    if (!input_shape_a_.empty()) {
        grad_a = reduce_grad_var_for_broadcasting(grad_a, input_shape_a_);
    }
    if (!input_shape_b_.empty()) {
        grad_b = reduce_grad_var_for_broadcasting(grad_b, input_shape_b_);
    }

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

    // Route the upcast through the autograd-aware `variable_cast` so the
    // input grad_fn chains survive into the F32 compute path (mirrors the
    // post-compute narrow-back below and the audit-3 Q.2 fix). Raw
    // `Variable(t.to(F32), false)` severed the chain — second-order grads
    // through mixed-precision Linear lost contributions.
    Variable go = needs_upcast ? tenzor::nn::variable_cast(grad_out, DType::Float32) : grad_out;
    Variable sx = needs_upcast ? tenzor::nn::variable_cast(saved_x, DType::Float32) : saved_x;
    Variable sw = needs_upcast ? tenzor::nn::variable_cast(saved_w, DType::Float32) : saved_w;

    auto grad_x = tenzor::matmul(go, sw);
    auto grad_out_t = tenzor::transpose(go, 0, 1);
    auto grad_w = tenzor::matmul(grad_out_t, sx);
    auto grad_b = tenzor::sum(go, 0, false);

    if (needs_upcast) {
        // Use autograd-aware cast (TypeCastBackward) so the Variable graph
        // stays intact for higher-order derivatives. The raw
        // `Variable(t.to(orig_dt), requires_grad)` rewrap severed grad_fn.
        grad_x = tenzor::nn::variable_cast(grad_x, orig_dt);
        grad_w = tenzor::nn::variable_cast(grad_w, orig_dt);
        grad_b = tenzor::nn::variable_cast(grad_b, orig_dt);
    }
    return {grad_x, grad_w, grad_b};
}

} // namespace tenzor
