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

// SumBackward implementation
auto SumBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});
    auto result = sum(inputs[0].tensor(), dim_, keepdim_);
    return {Variable(result, true)};
}

auto SumBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "SumBackward: cannot compute gradient of sum over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Reduced all dimensions - broadcast scalar tensor back to original shape
        // Use pure tensor operations (no CPU transfers) - backend agnostic!
        auto grad = grad_output;

        // Ensure grad is a 0-d tensor (may be 1-element tensor from some reductions)
        if (grad.ndim() > 0) {
            grad = reshape(grad, {});
        }

        // Use expand() to broadcast the scalar to input shape natively on device
        auto result = expand(grad, input_shape_vec);

        return {result};
    } else {
        // Dimension-specific reduction backward using unsqueeze + expand
        // expand() now uses native CUDA implementation - no device transfers!
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
        }

        return {expand(grad, input_shape_vec)};
    }
}

auto SumBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Sum backward is just expanding the gradient back to input shape.
    // Use the Variable-level autograd overloads of reshape/unsqueeze/expand so
    // the returned gradient keeps a grad_fn chaining back to grad_outputs[0]'s
    // producers. The adjoint of expand (a reduction) must propagate in
    // double-backward/HVP where grad_output depends on parameters; using the
    // Tensor-level ops here silently dropped the second-order contribution.
    const auto& input = saved_tensors_[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "SumBackward: cannot compute gradient of sum over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        Variable grad = grad_outputs[0];
        if (grad.tensor().ndim() > 0) {
            grad = tenzor::reshape(grad, {});
        }
        return {tenzor::expand(grad, input_shape_vec)};
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        Variable grad = grad_outputs[0];
        if (!keepdim_) {
            grad = tenzor::unsqueeze(grad, dim);
        }
        return {tenzor::expand(grad, input_shape_vec)};
    }
}

// MeanBackward implementation
auto MeanBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});
    auto result = mean(inputs[0].tensor(), dim_, keepdim_);
    return {Variable(result, true)};
}

auto MeanBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    // Calculate the number of elements that were averaged.
    // Normalize negative dim before indexing into shape — the dim-specific
    // branch below already does this (line ~157), but the n_elements compute
    // previously read raw `dim_.value()` and tripped a libstdc++ span
    // out-of-bounds assertion when dim < 0. (audit-2026-05-03 bug #5)
    int64_t n_elements = 1;
    if (dim_.has_value()) {
        int64_t dim_pos = dim_.value();
        if (dim_pos < 0) dim_pos += static_cast<int64_t>(input.shape().size());
        n_elements = input.shape()[dim_pos];
    } else {
        n_elements = input.numel();
    }

    TENZOR_CHECK_SHAPE(n_elements > 0,
        "MeanBackward: cannot compute mean of empty tensor (0 elements)");

    // Use double for scale calculation to preserve precision for Float64 tensors
    double scale = 1.0 / static_cast<double>(n_elements);
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Reduced all dimensions - broadcast scalar tensor back to original shape
        // Use pure tensor operations (no CPU transfers) - backend agnostic!
        auto grad = grad_output;

        // Ensure grad is a 0-d tensor (may be 1-element tensor from some reductions)
        if (grad.ndim() > 0) {
            grad = reshape(grad, {});
        }

        // Expand the scalar to input shape natively on device
        auto expanded = expand(grad, input_shape_vec);

        // Scale by 1/N using backend-agnostic tensor multiplication
        // Create scalar tensor with same dtype and device as expanded gradient
        // Use double overload of full() to preserve precision for Float64
        auto scale_tensor = full({}, scale, expanded.dtype(), expanded.device());

        auto result = mul(expanded, scale_tensor);

        return {result};
    } else {
        // Dimension-specific reduction backward
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
        }

        // Expand to input shape - works on CPU, transfers back if needed
        auto expanded = expand(grad, input_shape_vec);

        // Scale the expanded gradient using native Tensor multiplication
        // This now uses CUDA broadcasting automatically
        // Use expanded.dtype() to ensure dtypes match for element-wise operations
        // Use double overload of full() to preserve precision for Float64
        auto scale_tensor = full(input_shape_vec, scale, expanded.dtype(), expanded.device());
        return {mul(expanded, scale_tensor)};
    }
}

auto MeanBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Mean backward: expand gradient and scale by 1/N.
    // The scaling by 1/N uses Variable::operator*(double) which IS tracked by autograd
    // for higher-order gradient support.
    const auto& input = saved_tensors_[0];

    // Normalize negative dim before indexing into shape (audit-2026-05-03 bug #5).
    int64_t n_elements = 1;
    if (dim_.has_value()) {
        int64_t dim_pos = dim_.value();
        if (dim_pos < 0) dim_pos += static_cast<int64_t>(input.shape().size());
        n_elements = input.shape()[dim_pos];
    } else {
        n_elements = input.numel();
    }

    TENZOR_CHECK_SHAPE(n_elements > 0,
        "MeanBackward: cannot compute mean of empty tensor (0 elements)");

    double scale = 1.0 / static_cast<double>(n_elements);
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Scale the gradient Variable - this uses Variable::operator*(double) which
    // builds autograd graph when create_graph is active
    auto scaled_grad = grad_outputs[0] * scale;

    // Expand to input shape via the Variable-level autograd overloads so the
    // grad_fn chains through (expand's adjoint must propagate in double-backward
    // when grad_output depends on parameters). Tensor-level expand here severed
    // the second-order contribution.
    if (!dim_.has_value()) {
        Variable grad = scaled_grad;
        if (grad.tensor().ndim() > 0) {
            grad = tenzor::reshape(grad, {});
        }
        return {tenzor::expand(grad, input_shape_vec)};
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        Variable grad = scaled_grad;
        if (!keepdim_) {
            grad = tenzor::unsqueeze(grad, dim);
        }
        return {tenzor::expand(grad, input_shape_vec)};
    }
}

// LogBackward implementation
auto LogBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});
    auto result = log(inputs[0].tensor());
    return {Variable(result, true)};
}

auto LogBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(log(x))/dx = 1/x, with zero-safe clamping to prevent NaN
    const auto& input = saved_tensors_[0];
    if (input.is_complex()) {
        // Wirtinger: d/d(conj(z)) log(z) = grad / conj(z)
        auto grad_input = div(grad_outputs[0], conj(input));
        return {grad_input};
    }
    auto zero = zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
    auto eps = full(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                    detail::dtype_epsilon(input.dtype()),
                    input.dtype(), input.device());
    auto safe_input = where(eq(input, zero), eps, input);
    auto grad_input = div(grad_outputs[0], safe_input);
    return {grad_input};
}

auto LogBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(log(x))/dx = 1/x
    // GG.1: prefer the saved input Variable so the graph chains back through
    // the upstream forward; fall back to a non-grad Variable wrap when
    // create_graph was not active during forward.
    Variable saved_input = has_saved_variables() ? saved_variables_[0]
                                                  : Variable(saved_tensors_[0], false);
    return {grad_outputs[0] / saved_input};
}

// ExpBackward implementation
auto ExpBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = exp(inputs[0].tensor());
    save_for_backward({result});  // Save output for backward
    return {Variable(result, true)};
}

auto ExpBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(exp(x))/dx = exp(x)
    const auto& output = saved_tensors_[0];
    if (output.is_complex()) {
        // Wirtinger: d/d(conj(z)) exp(z) = conj(exp(z)) * grad
        auto grad_input = mul(grad_outputs[0], conj(output));
        return {grad_input};
    }
    auto grad_input = mul(grad_outputs[0], output);
    return {grad_input};
}

auto ExpBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(exp(x))/dx = exp(x).
    // GG.1: recompute exp from saved input Variable on the higher-order path
    // so the graph chains back to the original input. Fall back to the saved
    // output Tensor when create_graph was not active during forward.
    Variable saved_output;
    if (has_saved_variables()) {
        saved_output = tenzor::exp(saved_variables_[0]);
    } else {
        saved_output = Variable(saved_tensors_[0], false);
    }
    return {grad_outputs[0] * saved_output};
}

// NegBackward implementation
auto NegBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = neg(inputs[0].tensor());
    return {Variable(result, true)};
}

auto NegBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(-x)/dx = -1
    return {neg(grad_outputs[0])};
}

auto NegBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(-x)/dx = -1
    // Use Variable neg for higher-order gradient tracking
    return {tenzor::neg(grad_outputs[0])};
}

// LogSoftmaxBackward implementation
auto LogSoftmaxBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim_);
    std::vector<Tensor> input_tensors = {inputs[0].tensor()};
    auto result = dispatch(OpId::LogSoftmax, input_tensors, attrs)[0];

    // Save output for backward
    save_for_backward({result});

    return {Variable(result, true)};
}

auto LogSoftmaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Use backend's log_softmax_backward kernel
    const auto& output = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim_);
    std::vector<Tensor> inputs = {grad_output, output};
    auto grad_input = dispatch(OpId::LogSoftmaxBackward, inputs, attrs)[0];

    return {grad_input};
}

auto LogSoftmaxBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dx_i = dL/dy_i - exp(y_i) * sum_j(dL/dy_j)
    // GG.1: recompute log_softmax(input) on the live graph when create_graph
    // was active; fall back to the saved output Tensor otherwise.
    Variable output_var;
    if (has_saved_variables()) {
        output_var = tenzor::log_softmax(saved_variables_[0], dim_);
    } else {
        output_var = Variable(saved_tensors_[0], false);
    }
    auto grad_sum = tenzor::sum(grad_outputs[0], dim_, true);
    auto softmax_output = tenzor::exp(output_var);
    auto grad_input = grad_outputs[0] - softmax_output * grad_sum;
    return {grad_input};
}

// SoftmaxBackward implementation
auto SoftmaxBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim_);
    std::vector<Tensor> input_tensors = {inputs[0].tensor()};
    auto result = dispatch(OpId::Softmax, input_tensors, attrs)[0];

    // Save output for backward
    save_for_backward({result});

    return {Variable(result, true)};
}

auto SoftmaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Use backend-optimized softmax_backward kernel via dispatch
    const auto& output = saved_tensors_[0];  // y = softmax(x)
    const auto& grad_output = grad_outputs[0];  // dL/dy

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim_);
    std::vector<Tensor> inputs = {grad_output, output};
    auto grad_input = dispatch(OpId::SoftmaxBackward, inputs, attrs)[0];

    return {grad_input};
}

auto SoftmaxBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
    // GG.1: recompute softmax(input) on the live graph when create_graph was
    // active; fall back to the saved output Tensor otherwise.
    Variable output_var;
    if (has_saved_variables()) {
        output_var = tenzor::softmax(saved_variables_[0], dim_);
    } else {
        output_var = Variable(saved_tensors_[0], false);
    }
    auto dot_product = tenzor::sum(grad_outputs[0] * output_var, dim_, true);
    auto grad_input = output_var * (grad_outputs[0] - dot_product);
    return {grad_input};
}

// AbsBackward implementation
auto AbsBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});
    auto result = abs(inputs[0].tensor());
    return {Variable(result, true)};
}

auto AbsBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    if (input.is_complex()) {
        // Wirtinger (no-1/2 convention, matching LogBackward/ExpBackward and
        // PyTorch's grad * self.sgn()): d/d(conj(z)) |z| = z / |z| = sgn(z).
        auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
        auto abs_input = tenzor::abs(input);
        double eps = 1e-7;
        auto eps_tensor = full(input_shape_vec, eps, abs_input.dtype(), input.device());
        auto safe_abs = tenzor::where(gt(abs_input, eps_tensor), abs_input, eps_tensor);
        // grad * z / |z|
        auto scale = div(input, safe_abs);
        return {mul(grad.to(input.dtype()), scale)};
    }

    // Real path: d(abs(x))/dx = sign(x), with epsilon guard at x=0
    double eps = 1e-7;
    if (input.dtype() == DType::Float64) eps = 1e-15;
    else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) eps = 1e-3;

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto abs_input = tenzor::abs(input);
    auto eps_tensor = full(input_shape_vec, eps, input.dtype(), input.device());
    auto mask = gt(abs_input, eps_tensor);
    auto safe_sign = tenzor::where(mask,
        sign(input),
        zeros(input_shape_vec, input.dtype(), input.device()));
    return {mul(grad, safe_sign)};
}

auto AbsBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(abs(x))/dx = sign(x), with epsilon guard to avoid NaN at x=0.
    // sign is non-differentiable, so compute it at Tensor level.
    // GG.1: the sign factor is a constant Variable (sign/gt/where are not
    // differentiable); only grad_outputs[0]'s chain matters for higher-order
    // autograd here. We still source the input Tensor from saved_variables_
    // when available so the wrapped Tensor is the live one rather than the
    // potentially-offloaded saved copy.
    const auto& input = has_saved_variables() ? saved_variables_[0].tensor()
                                              : saved_tensors_[0];

    double eps = 1e-7;
    if (input.dtype() == DType::Float64) eps = 1e-15;
    else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) eps = 1e-3;

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto abs_input = tenzor::abs(input);
    auto eps_tensor = full(input_shape_vec, eps, input.dtype(), input.device());
    auto mask = gt(abs_input, eps_tensor);
    auto safe_sign = tenzor::where(mask,
        sign(input),
        zeros(input_shape_vec, input.dtype(), input.device()));
    Variable sign_var(safe_sign, false);
    return {grad_outputs[0] * sign_var};
}

// ClampBackward implementation
auto ClampBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});
    auto result = clamp(inputs[0].tensor(), min_, max_);
    return {Variable(result, true)};
}

auto ClampBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(clamp(x, min, max))/dx = 1 if min <= x <= max else 0
    const auto& input = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    // Create mask: 1 where min <= x <= max, 0 otherwise
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
    auto zeros_tensor = zeros(input_shape_vec, input.dtype(), input.device());

    // Check if input >= min
    auto min_tensor = full(input_shape_vec, min_, input.dtype(), input.device());
    auto max_tensor = full(input_shape_vec, max_, input.dtype(), input.device());

    // Mask = (input >= min) & (input <= max) — inclusive at the boundary,
    // matching PyTorch's clamp_backward convention. The formula
    //     mask = 1 - |sign(input - clamp(input, min, max))|
    // is exact (not an approximation): sign(0) == 0 gives mask = 1 in the
    // interior AND at exact boundary points; sign(±) == ±1 gives mask = 0
    // outside the clamp range. Re-using `clamp` is the cleanest way to
    // express the inclusive boundary check across all backends.
    auto clamped   = clamp(input, min_, max_);
    auto diff      = sub(input, clamped);
    auto diff_sign = abs(sign(diff));
    auto mask      = sub(ones_tensor, diff_sign);

    return {mul(grad_output, mask)};
}

auto ClampBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(clamp(x, min, max))/dx = 1 if min <= x <= max, else 0
    // The mask is non-differentiable, compute at Tensor level.
    // GG.1: see AbsBackward comment.
    const auto& input = has_saved_variables() ? saved_variables_[0].tensor()
                                              : saved_tensors_[0];
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
    auto clamped = clamp(input, min_, max_);
    auto diff = sub(input, clamped);
    auto diff_sign = abs(sign(diff));
    auto mask = sub(ones_tensor, diff_sign);
    Variable mask_var(mask, false);
    return {grad_outputs[0] * mask_var};
}

// =========================================================================
// Complex number backward functions (Wirtinger derivatives)
// =========================================================================

// ConjBackward: conj(z) -> grad = conj(grad)
auto ConjBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = conj(inputs[0].tensor());
    return {Variable(result, true)};
}

auto ConjBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // audit-6 BB.4: X.5 routes real-Variable conj() through this class so
    // grad_fn is preserved. For real dtypes, conj(z) == z and the gradient is
    // simply identity — and calling conj() on a real tensor would throw if
    // OpId::Conj isn't registered for that backend. Short-circuit here.
    const auto& grad = grad_outputs[0];
    if (!grad.is_complex()) {
        return {grad};
    }
    return {conj(grad)};
}

// RealBackward: real(z) -> grad_z = 0.5 * grad (with zero imaginary part)
auto RealBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    input_dtype_ = inputs[0].tensor().dtype();
    auto result = real(inputs[0].tensor());
    return {Variable(result, true)};
}

auto RealBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // audit-7 DD.1: PyTorch contract for real(z) backward is
    // grad_z = complex(grad_real, 0). No 0.5 factor. Build the complex
    // tensor by stacking [grad, zeros] along a new trailing axis and
    // viewing the result as complex.
    auto zero = zeros_like(grad);
    std::array<Tensor, 2> parts = {grad, zero};
    std::span<const Tensor> parts_span(parts.data(), parts.size());
    auto stacked = ::tenzor::stack(parts_span, /*dim=*/-1).contiguous();
    auto result = ::tenzor::view_as_complex(stacked);
    return {result};
}

// ImagBackward: imag(z) -> grad_z = -0.5j * grad
auto ImagBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    input_dtype_ = inputs[0].tensor().dtype();
    auto result = imag(inputs[0].tensor());
    return {Variable(result, true)};
}

auto ImagBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // audit-7 DD.2: PyTorch contract for imag(z) backward is
    // grad_z = 0 + j*grad (real=0, imag=grad). Build via stack
    // [zeros, grad] along a new trailing axis and view as complex.
    auto zero = zeros_like(grad);
    std::array<Tensor, 2> parts = {zero, grad};
    std::span<const Tensor> parts_span(parts.data(), parts.size());
    auto stacked = ::tenzor::stack(parts_span, /*dim=*/-1).contiguous();
    auto result = ::tenzor::view_as_complex(stacked);
    return {result};
}

// Audit-7 EE.5: higher-order (create_graph=True) backwards for the complex
// triplet must stay in Variable land. The previous implementations called
// backward({grad.tensor()}) and rewrapped, which severed the autograd graph
// after the first .backward(create_graph=True) call so second derivatives
// would be zero.
auto ConjBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Mirror the raw-Tensor backward: real dtype → identity, complex → conj.
    // Both use the Variable-level autograd::conj which preserves the graph.
    const auto& g = grad_outputs[0];
    if (!g.tensor().is_complex()) {
        return {g};
    }
    return {tenzor::conj(g)};
}

auto RealBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // PyTorch contract: grad_z = complex(grad_real, 0).
    // Build it with Variable ops so the graph survives create_graph=True:
    // [grad, zeros] -> unsqueeze each on dim -1 -> cat on dim -1 -> view_as_complex.
    const auto& g = grad_outputs[0];
    Variable zero_v(zeros_like(g.tensor()), false);
    auto g_u = tenzor::unsqueeze(g, -1);
    auto z_u = tenzor::unsqueeze(zero_v, -1);
    auto stacked = tenzor::cat(std::vector<Variable>{g_u, z_u}, /*dim=*/-1);
    return {tenzor::view_as_complex(stacked)};
}

auto ImagBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // PyTorch contract: grad_z = 0 + j*grad_imag.
    // Same construction as RealBackward but with (zeros, grad) order.
    const auto& g = grad_outputs[0];
    Variable zero_v(zeros_like(g.tensor()), false);
    auto z_u = tenzor::unsqueeze(zero_v, -1);
    auto g_u = tenzor::unsqueeze(g, -1);
    auto stacked = tenzor::cat(std::vector<Variable>{z_u, g_u}, /*dim=*/-1);
    return {tenzor::view_as_complex(stacked)};
}

} // namespace tenzor
