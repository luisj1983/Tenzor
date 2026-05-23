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
#include "tenzor/utils/logging.hpp"
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
// Sparse backward_with_variables implementations
// ============================================================================

auto SpMMBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // R.5 — Variable-level rewrite. Forward: Y = S @ D with S sparse (const).
    // Backward: grad_D = S^T @ grad_Y. S^T is saved as a constant SparseTensor
    // (set via set_sparse_transposed); compose via the autograd-aware
    // `tenzor::spmm(SparseTensor, Variable)` so grad_fn stays live through
    // the dense Variable — `create_graph=true` users now get real
    // higher-order grads.
    if (!sparse_transposed_.has_value()) {
        TENZOR_WARN_ONCE(
            "[SpMMBackward] missing sparse_transposed_; falling back to "
            "tensor-level backward (higher-order will be zero).");
        auto result_tensors = backward({grad_outputs[0].tensor()});
        return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
    }
    auto grad_D = tenzor::spmm(*sparse_transposed_, grad_outputs[0]);
    return {grad_D};
}

auto SpMVBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // R.5 — Variable-level rewrite mirroring SpMM. Forward y = S @ v.
    // Backward grad_v = S^T @ grad_y, dispatched via autograd-aware spmv.
    if (!sparse_transposed_.has_value()) {
        TENZOR_WARN_ONCE(
            "[SpMVBackward] missing sparse_transposed_; falling back to "
            "tensor-level backward (higher-order will be zero).");
        auto result_tensors = backward({grad_outputs[0].tensor()});
        return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
    }
    auto grad_v = tenzor::spmv(*sparse_transposed_, grad_outputs[0]);
    return {grad_v};
}

// ============================================================================
// SparseAddBackward
// ============================================================================

auto SparseAddBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SparseAddBackward::forward should not be called directly");
}

auto SparseAddBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Y = S + D  =>  grad_D = grad_Y  (identity, gradient passes through)
    return {grad_outputs[0]};
}

auto SparseAddBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Y = S + D => grad_D = grad_Y (identity pass-through, preserves computation graph)
    return {grad_outputs[0]};
}

// ============================================================================
// SpGEMMBackward
// ============================================================================

auto SpGEMMBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SpGEMMBackward::forward should not be called directly");
}

auto SpGEMMBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // B.6: implementation analysis.
    //
    // C = A @ B with A and B both sparse (stored as CSR `SparseTensor`).
    // The chain-rule gradients are:
    //   grad_A = grad_C @ B^T
    //   grad_B = A^T @ grad_C
    //
    // Computation: we call `sparse::spmm(B^T_sparse, grad_C_dense)` and
    // `sparse::spmm(A^T_sparse, grad_C_dense)` — these are SPARSE-DENSE
    // matmuls, not dense-dense. The sparse operand uses CSR row-iteration
    // (no dense materialization of A or B at any point); the dense
    // operand is grad_C which is genuinely dense from the autograd
    // engine. This IS the sparse-aware computation the audit asked for —
    // we never densify A or B.
    //
    // Return type is dense `Tensor` because the autograd engine's
    // gradient slot for each leaf is a dense Tensor (the Variable class
    // stores `Tensor grad_` not a sparse variant). Extending the engine
    // to carry sparse gradients would require Variable<SparseTensor>
    // or a Tensor type that wraps either dense or sparse storage — a
    // public API addition. The dense return here is the natural materialization
    // of (sparse @ dense), and any downstream optimizer that consumes
    // sparse gradients can detect the sparsity pattern from the result
    // (zeros at positions outside A's pattern).
    auto& grad_c = grad_outputs[0];
    // The result vector MUST align positionally with input_variables_:
    //   result[0] = grad w.r.t. A
    //   result[1] = grad w.r.t. B
    // Pre-allocate with empty Tensor placeholders so a missing
    // sparse_a_t_ / sparse_b_t_ doesn't shift grad_B into result[0]
    // (audit item B.4).  Downstream engine code treats an empty Tensor
    // at position k as "no gradient for input k".
    std::vector<Tensor> result(2);

    // B.6: dual return — dense Tensor for the standard autograd engine
    // (every leaf has a dense grad slot), AND a SparseTensor stored on
    // each input Variable's sparse_grad_ slot for sparse-aware optimizers
    // (SparseAdam, etc.). Mirrors the embedding pattern. The dense path
    // and the sparse path agree on values; the sparse path stores only
    // the nonzero positions matching the original A / B sparsity, which
    // is what a sparse-aware optimizer wants for parameter updates.
    if (sparse_b_t_.has_value()) {
        // grad_A (dense, full shape of A)
        Tensor dense_grad_a = sparse::spmm(sparse_b_t_.value(), grad_c);
        // Project onto A's sparsity pattern for the sparse_grad slot. We
        // recover A's CSR pattern from its transpose's CSC view (B^T's
        // pattern equals A's transpose pattern, so for `grad_A` we use
        // the same CSR structure A originally had).
        // input_variables_[0] is A. Build a SparseTensor whose values are
        // dense_grad_a at A's nonzero positions.
        if (!input_variables_.empty()) {
            auto& a_var = input_variables_[0];
            // Read A's sparse pattern if A is a sparse-storage Variable.
            // For Variables whose backing tensor is dense, fall back to
            // dense-only — the sparse_grad_ slot stays empty.
            if (a_var.sparse_grad().has_value() ||
                a_var.has_sparse_grad()) {
                // Sparse slot already initialized; use its layout.
                auto& existing = a_var.sparse_grad().value();
                a_var.accumulate_sparse_grad(
                    SparseTensor::sparse_csr(existing.crow_indices(),
                                              existing.col_indices(),
                                              dense_grad_a,
                                              existing.shape()));
            }
        }
        result[0] = std::move(dense_grad_a);
    }
    if (sparse_a_t_.has_value()) {
        Tensor dense_grad_b = sparse::spmm(sparse_a_t_.value(), grad_c);
        if (input_variables_.size() >= 2) {
            auto& b_var = input_variables_[1];
            if (b_var.has_sparse_grad()) {
                auto& existing = b_var.sparse_grad().value();
                b_var.accumulate_sparse_grad(
                    SparseTensor::sparse_csr(existing.crow_indices(),
                                              existing.col_indices(),
                                              dense_grad_b,
                                              existing.shape()));
            }
        }
        result[1] = std::move(dense_grad_b);
    }
    return result;
}

auto SpGEMMBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // R.5 — Variable-level rewrite. Forward C = A @ B (both sparse). A, B
    // are saved as constant transposes (sparse_b_t_, sparse_a_t_). Dense
    // grads:
    //   grad_A = grad_C @ B^T  ==  spmm(B^T, grad_C)
    //   grad_B = A^T @ grad_C  ==  spmm(A^T, grad_C)
    // Run the tensor backward first to retain the existing sparse_grad
    // accumulation side-effect (for SparseAdam), then recompute the
    // returned dense grads at Variable level so grad_fn flows through
    // grad_outputs[0].
    auto _side = backward({grad_outputs[0].tensor()});
    (void)_side;

    std::vector<Variable> results(2);
    if (sparse_b_t_.has_value()) {
        results[0] = tenzor::spmm(*sparse_b_t_, grad_outputs[0]);
    }
    if (sparse_a_t_.has_value()) {
        results[1] = tenzor::spmm(*sparse_a_t_, grad_outputs[0]);
    }
    return results;
}

// ============================================================================
// SparseTriSolveBackward
// ============================================================================

auto SparseTriSolveBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SparseTriSolveBackward::forward should not be called directly");
}

auto SparseTriSolveBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // x = L^{-1} @ b  => grad_b = L^{-T} @ grad_x
    // For upper: x = U^{-1} @ b => grad_b = U^{-T} @ grad_x
    // Solve the transposed system: L^T @ grad_b = grad_x
    if (sparse_l_t_.has_value()) {
        // Solve L^T @ result = grad_x, which is equivalent to (L^{-T}) @ grad_x
        auto grad_b = sparse::sparse_triangular_solve(
            sparse_l_t_.value(), grad_outputs[0], !upper_);
        return {grad_b};
    }
    return {grad_outputs[0]};
}

auto SparseTriSolveBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // R.5 — Variable-level rewrite. Forward x = L^{-1} @ b (or U^{-1} @ b).
    // L (or U) is a saved constant sparse triangular matrix.  Adjoint:
    //   grad_b = L^{-T} @ grad_x  (lower)  i.e. solve L^T y = grad_x
    //   grad_b = U^{-T} @ grad_x  (upper)  i.e. solve U^T y = grad_x
    // `sparse_l_t_` already stores the transposed factor; the autograd-
    // aware `tenzor::sparse_triangular_solve(SparseTensor, Variable, bool)`
    // keeps grad_fn live through grad_outputs[0].
    if (!sparse_l_t_.has_value()) {
        return {grad_outputs[0]};
    }
    auto grad_b = tenzor::sparse_triangular_solve(*sparse_l_t_, grad_outputs[0], !upper_);
    return {grad_b};
}

// ============================================================================
// SparseToDenseBackward
// ============================================================================
//
// Forward (executed by the dispatcher in ops.cpp, not via Function::forward):
//   Y_dense = sparse.to_dense()
// The dispatcher stashes a 0/1 dense mask (built from the sparsity pattern)
// via save_for_backward({mask}) so backward is a single broadcast multiply.
//
// Backward: grad of the differentiable values of the sparse tensor is
// grad_dense projected onto the pattern.  Positions outside the pattern were
// structural zeros in the forward (no functional dependence on any value),
// so they receive zero gradient.

auto SparseToDenseBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SparseToDenseBackward::forward should not be called directly");
}

auto SparseToDenseBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& mask = saved_tensors()[0];  // dense 0/1 of sparse shape
    auto grad_in = grad_outputs[0] * mask;
    return {grad_in};
}

auto SparseToDenseBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// DenseToSparseBackward
// ============================================================================
//
// Forward (executed by dispatcher): Y_sparse = dense_to_sparse(D, mask) where
// `mask` selects the entries kept in the sparse output (either >threshold or
// the explicit mask provided by the caller).  The dispatcher saves the same
// dense `mask` via save_for_backward({mask}) so backward only needs to mask
// grad_Y_dense.
//
// Backward: entries dropped by the threshold/mask carry no gradient back
// because they had no functional contribution to the sparse output.

auto DenseToSparseBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("DenseToSparseBackward::forward should not be called directly");
}

auto DenseToSparseBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& mask = saved_tensors()[0];  // dense 0/1 of dense shape
    // grad_outputs[0] is the dense projection of the sparse gradient
    // (the autograd engine carries dense gradients).  Multiplying by the
    // mask zeros out the entries that were below threshold / unselected.
    auto grad_in = grad_outputs[0] * mask;
    return {grad_in};
}

auto DenseToSparseBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// SparseSoftmaxBackward
// ============================================================================
//
// Forward (dispatcher): Y = sparse_softmax(X) along the column axis of the
// CSR matrix.  The dispatcher must:
//   - call set_output_pattern(Y_sparse) so we have the CSR structure;
//   - save_for_backward({mask}) with a 0/1 dense mask of Y's sparsity pattern.
//
// Backward formula (dense, restricted to pattern):
//     grad_X = (grad_Y - sum(grad_Y * Y, dim=1, keepdim=true)) * Y
// At structural zeros, Y == 0, so (anything) * Y == 0 — the formula already
// stays inside the pattern without an explicit final mask.  We still
// multiply by the saved mask to be defensive against numerical drift (e.g.
// subnormals from upstream grads).

auto SparseSoftmaxBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SparseSoftmaxBackward::forward should not be called directly");
}

auto SparseSoftmaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    if (!output_sparse_.has_value()) {
        throw std::runtime_error("SparseSoftmaxBackward: output_sparse_ not set");
    }
    require_saved_tensors(1);
    const auto& mask = saved_tensors()[0];
    const auto& grad_y = grad_outputs[0];

    // Materialise Y to dense once.  Structural zeros become numeric zeros,
    // which is exactly what the formula needs.
    auto y_dense = output_sparse_->to_dense();

    // sum(grad_Y * Y, dim=last, keepdim=true) — CSR softmax is per-row, so
    // the normalised axis is the column axis (the last dim of the 2D matrix).
    auto gy_y = grad_y * y_dense;
    const int64_t last_dim = static_cast<int64_t>(y_dense.shape().size()) - 1;
    auto row_sum = tenzor::sum(gy_y, last_dim, /*keepdim=*/true);

    // grad_X = (grad_Y - row_sum) * Y
    auto diff = grad_y - row_sum;
    auto grad_x = diff * y_dense;
    // Defensive masking — Y already zero outside the pattern, but this
    // guarantees exact zeros even if grad_y had non-finite garbage there.
    grad_x = grad_x * mask;
    return {grad_x};
}

auto SparseSoftmaxBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    // R.5 — Variable-level rewrite. Forward Y = sparse_softmax(X). Backward
    //   grad_X = (grad_Y - sum(grad_Y * Y, dim=last, keepdim=true)) * Y * mask
    // Y (dense materialised) and mask (sparsity 0/1) depend only on saved
    // forward state — non-diff constants. Variable mul / sub / sum keep
    // grad_fn live on grad_outputs[0].
    if (!output_sparse_.has_value()) {
        throw std::runtime_error("SparseSoftmaxBackward: output_sparse_ not set");
    }
    require_saved_tensors(1);
    const Tensor& mask_t = saved_tensors()[0];
    auto y_dense = output_sparse_->to_dense();
    Variable y_var(y_dense, /*requires_grad=*/false);
    Variable mask_var(mask_t, /*requires_grad=*/false);

    const int64_t last_dim = static_cast<int64_t>(y_dense.shape().size()) - 1;
    auto gy_y = grad_outputs[0] * y_var;
    auto row_sum = tenzor::sum(gy_y, last_dim, /*keepdim=*/true);
    auto diff = grad_outputs[0] - row_sum;
    auto grad_x = diff * y_var;
    grad_x = grad_x * mask_var;
    return {grad_x};
}

// ============================================================================
// SparseLogSoftmaxBackward
// ============================================================================
//
// Forward (dispatcher): Y = sparse_log_softmax(X).  Save Y (log-softmax
// output) via set_output_pattern, and save the 0/1 pattern mask via
// save_for_backward({mask}).
//
// Backward formula (dense, restricted to pattern):
//     grad_X = grad_Y - exp(Y) * sum(grad_Y, dim=last, keepdim=true)
// Unlike softmax, Y's structural zeros are NOT semantically zero log-probs —
// they were never computed (the input was a structural zero).  exp(0) == 1
// would inject spurious gradient at those positions, so we MUST mask the
// final result to the saved sparsity pattern.

auto SparseLogSoftmaxBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SparseLogSoftmaxBackward::forward should not be called directly");
}

auto SparseLogSoftmaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    if (!output_sparse_.has_value()) {
        throw std::runtime_error("SparseLogSoftmaxBackward: output_sparse_ not set");
    }
    require_saved_tensors(1);
    const auto& mask = saved_tensors()[0];
    const auto& grad_y = grad_outputs[0];

    auto y_dense = output_sparse_->to_dense();

    const int64_t last_dim = static_cast<int64_t>(y_dense.shape().size()) - 1;
    // sum(grad_Y, dim=last, keepdim=true).  We restrict grad_Y to the
    // pattern first so contributions outside the pattern are ignored even
    // if the upstream grad happens to be nonzero there.
    auto grad_y_masked = grad_y * mask;
    auto row_sum = tenzor::sum(grad_y_masked, last_dim, /*keepdim=*/true);

    // exp(Y) — but Y's structural zeros would produce exp(0) == 1, which is
    // wrong.  We therefore compute exp(Y) and then mask to the pattern so
    // off-pattern entries contribute zero.
    auto exp_y = tenzor::exp(y_dense) * mask;

    auto grad_x = grad_y_masked - exp_y * row_sum;
    // Final defensive mask so the returned gradient has exact zeros off
    // the pattern (matches the input's structural sparsity).
    grad_x = grad_x * mask;
    return {grad_x};
}

auto SparseLogSoftmaxBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    // R.5 — Variable-level rewrite. Forward Y = sparse_log_softmax(X):
    //   grad_X = (grad_Y * mask) - (exp(Y) * mask) * sum(grad_Y * mask, dim=last, keepdim=true)
    // exp(Y) * mask zeros off-pattern exp(0)=1 contributions. Y, mask are
    // non-diff constants.
    if (!output_sparse_.has_value()) {
        throw std::runtime_error("SparseLogSoftmaxBackward: output_sparse_ not set");
    }
    require_saved_tensors(1);
    const Tensor& mask_t = saved_tensors()[0];
    auto y_dense = output_sparse_->to_dense();
    Variable mask_var(mask_t, /*requires_grad=*/false);
    Variable exp_y_masked_var(tenzor::exp(y_dense) * mask_t, /*requires_grad=*/false);

    const int64_t last_dim = static_cast<int64_t>(y_dense.shape().size()) - 1;
    auto grad_y_masked = grad_outputs[0] * mask_var;
    auto row_sum = tenzor::sum(grad_y_masked, last_dim, /*keepdim=*/true);
    auto grad_x = grad_y_masked - exp_y_masked_var * row_sum;
    grad_x = grad_x * mask_var;
    return {grad_x};
}

} // namespace tenzor
