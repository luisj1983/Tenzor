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
// CumSum backward
// ============================================================================

auto CumSumBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CumSumBackward::forward should not be called");
}

auto CumSumBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // dL/dx = flip(cumsum(flip(grad, dim), dim), dim)
    auto flipped = flip(grad, {dim_});
    auto cum = cumsum(flipped, dim_);
    return {flip(cum, {dim_})};
}

auto CumSumBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // cumsum backward: flip(cumsum(flip(grad, dim), dim), dim)
    auto flipped = tenzor::flip(grad_outputs[0], {dim_});
    auto cum = tenzor::cumsum(flipped, dim_);
    return {tenzor::flip(cum, {dim_})};
}

// ============================================================================
// CumProd backward
// ============================================================================

auto CumProdBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CumProdBackward::forward should not be called");
}

auto CumProdBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];

    // dL/dx = flip(cumsum(flip(output * grad, dim), dim), dim) / input
    // With zero-safe division
    auto prod_grad = mul(output, grad);
    auto flipped = flip(prod_grad, {dim_});
    auto cum = cumsum(flipped, dim_);
    auto rev_cum = flip(cum, {dim_});

    // Zero-safe: where input == 0, use 0 gradient
    auto zero_t = zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                        input.dtype(), input.device());
    auto eps = full(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                    detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    auto zero_mask = eq(input, zero_t);
    auto safe_input = where(zero_mask, eps, input);
    auto result = div(rev_cum, safe_input);

    // Zero out positions where input was zero
    return {where(zero_mask, zero_t, result)};
}

auto CumProdBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // cumprod backward: flip(cumsum(flip(output * grad, dim), dim), dim) / input
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    Variable output_var(output, false);

    auto prod_grad = output_var * grad_outputs[0];
    auto flipped = tenzor::flip(prod_grad, {dim_});
    auto cum = tenzor::cumsum(flipped, dim_);
    auto rev_cum = tenzor::flip(cum, {dim_});

    // Zero-safe division at Tensor level (input is constant)
    auto zero_t = zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                        input.dtype(), input.device());
    auto eps = full(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                    detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    auto zero_mask = eq(input, zero_t);
    auto safe_input = where(zero_mask, eps, input);
    Variable safe_input_var(safe_input, false);

    auto result = rev_cum / safe_input_var;

    // Zero out positions where input was zero
    Variable zero_var(zero_t, false);
    Variable mask_var(zero_mask, false);
    return {tenzor::where(mask_var, zero_var, result)};
}

// ============================================================================
// TopK backward
// ============================================================================

auto TopKBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TopKBackward::forward should not be called");
}

auto TopKBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // saved_tensors_[0] = original shape as 1D Int64 tensor
    // saved_tensors_[1] = indices from topk
    const auto& shape_tensor = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];

    auto shape_ptr = shape_tensor.data<int64_t>();
    auto orig_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    // Create zeros with original shape and scatter grad at index positions
    auto result = zeros(orig_shape, grad.dtype(), grad.device());
    return {scatter_add(result, dim_, indices, grad)};
}

auto TopKBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Indices and original shape are constants; scatter_add threads grad Variable
    // back to original positions preserving the graph for create_graph=true.
    const auto& shape_tensor = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];
    const auto& grad_var = grad_outputs[0];

    auto shape_ptr = shape_tensor.data<int64_t>();
    auto orig_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    auto zeros_t = zeros(orig_shape, grad_var.tensor().dtype(), grad_var.tensor().device());
    Variable zeros_var(zeros_t, false);
    return {tenzor::scatter_add(zeros_var, dim_, indices, grad_var)};
}

// ============================================================================
// Sort backward
// ============================================================================

auto SortBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SortBackward::forward should not be called");
}

auto SortBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // saved_tensors_[0] = original shape as 1D Int64 tensor
    // saved_tensors_[1] = sort indices
    const auto& shape_tensor = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];

    auto shape_ptr = shape_tensor.data<int64_t>();
    auto orig_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    // Scatter grad back using inverse permutation (same as scatter)
    auto result = zeros(orig_shape, grad.dtype(), grad.device());
    return {scatter(result, dim_, indices, grad)};
}

auto SortBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Sort indices are a constant permutation; scatter threads grad Variable
    // back through the inverse permutation preserving the graph.
    const auto& shape_tensor = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];
    const auto& grad_var = grad_outputs[0];

    auto shape_ptr = shape_tensor.data<int64_t>();
    auto orig_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    auto zeros_t = zeros(orig_shape, grad_var.tensor().dtype(), grad_var.tensor().device());
    Variable zeros_var(zeros_t, false);
    return {tenzor::scatter(zeros_var, dim_, indices, grad_var)};
}

// ============================================================================
// Diag backward
// ============================================================================

auto DiagBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("DiagBackward::forward should not be called");
}

auto DiagBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // diag() is its own "transpose": applying diag to the grad reverses the operation
    return {diag(grad, diagonal_)};
}

auto DiagBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // diag backward: diag(grad, diagonal) reverses the operation
    return {tenzor::diag(grad_outputs[0], diagonal_)};
}

// ============================================================================
// Trace backward
// ============================================================================

auto TraceBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TraceBackward::forward should not be called");
}

auto TraceBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // saved_tensors_[0] holds dtype/device info from the original input
    const auto& input = saved_tensors_[0];
    // dL/dA = grad_scalar * eye(n)
    auto identity = eye(n_, std::nullopt, input.dtype(), input.device());
    // grad is scalar — expand it
    return {mul(identity, grad)};
}

auto TraceBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // trace backward: dL/dA = grad_scalar * eye(n)
    const auto& input = saved_tensors_[0];
    auto identity = eye(n_, std::nullopt, input.dtype(), input.device());
    Variable eye_var(identity, false);
    // grad is scalar; multiply broadcasts
    return {grad_outputs[0] * eye_var};
}

// ============================================================================
// Triu backward
// ============================================================================

auto TriuBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TriuBackward::forward should not be called");
}

auto TriuBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {triu(grad_outputs[0], diagonal_)};
}

auto TriuBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {tenzor::triu(grad_outputs[0], diagonal_)};
}

// ============================================================================
// Tril backward
// ============================================================================

auto TrilBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TrilBackward::forward should not be called");
}

auto TrilBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {tril(grad_outputs[0], diagonal_)};
}

auto TrilBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {tenzor::tril(grad_outputs[0], diagonal_)};
}

// ============================================================================
// FFT backward
// ============================================================================
// Normalization inversion. For y = FFT(x) with norm ν the Jacobian is
// the matrix A_ν; its Hermitian adjoint A_ν^H — which is what the chain
// rule needs — is the IFFT under the *opposite* scaling convention.
// Concretely: "backward" (unscaled fft, scaled ifft) pairs with "forward"
// (scaled fft, unscaled ifft); "ortho" is self-adjoint. Using the same
// norm on both branches (as the old code did) left grad_x short by a
// factor of N and broke numerical-gradient parity.
static std::string invert_fft_norm(const std::string& norm) {
    if (norm == "backward") return "forward";
    if (norm == "forward") return "backward";
    return norm;  // "ortho" is self-adjoint
}

auto FFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("FFTBackward::forward should not be called");
}

auto FFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {fft::ifft(grad_outputs[0], n_, dim_, invert_fft_norm(norm_))};
}

auto FFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {fft_autograd::ifft(grad_outputs[0], n_, dim_, invert_fft_norm(norm_))};
}

// ============================================================================
// IFFT backward
// ============================================================================

auto IFFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IFFTBackward::forward should not be called");
}

auto IFFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {fft::fft(grad_outputs[0], n_, dim_, invert_fft_norm(norm_))};
}

auto IFFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {fft_autograd::fft(grad_outputs[0], n_, dim_, invert_fft_norm(norm_))};
}

// ============================================================================
// RFFT backward
// ============================================================================

auto RFFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("RFFTBackward::forward should not be called");
}

auto RFFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // irfft needs the original signal length to reconstruct
    return {fft::irfft(grad_outputs[0], signal_length_, dim_, invert_fft_norm(norm_))};
}

auto RFFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {fft_autograd::irfft(grad_outputs[0], signal_length_, dim_, invert_fft_norm(norm_))};
}

// ============================================================================
// IRFFT backward
// ============================================================================

auto IRFFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IRFFTBackward::forward should not be called");
}

auto IRFFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {fft::rfft(grad_outputs[0], std::nullopt, dim_, invert_fft_norm(norm_))};
}

auto IRFFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {fft_autograd::rfft(grad_outputs[0], std::nullopt, dim_, invert_fft_norm(norm_))};
}

} // namespace tenzor
