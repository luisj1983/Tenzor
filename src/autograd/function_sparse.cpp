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
// Sparse backward_with_variables implementations
// ============================================================================

auto SpMMBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Compute gradient at Tensor level (sparse matrix is a constant, no graph needed through it)
    // but wrap result as a Variable that preserves requires_grad for graph continuity
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
}

auto SpMVBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Same pattern as SpMM — sparse matrix is constant, gradient flows through dense input only
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
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
    // C = A @ B (sparse-sparse)
    // grad_A = grad_C @ B^T  => use spmm with B^T
    // grad_B = A^T @ grad_C  => use spmm with A^T
    // Both A and B are constant sparse matrices; gradients are dense.
    auto& grad_c = grad_outputs[0];
    std::vector<Tensor> result;

    if (sparse_b_t_.has_value()) {
        // grad_A (dense) = grad_C @ B^T
        result.push_back(sparse::spmm(sparse_b_t_.value(), grad_c));
    }
    if (sparse_a_t_.has_value()) {
        // grad_B (dense) = A^T @ grad_C
        result.push_back(sparse::spmm(sparse_a_t_.value(), grad_c));
    }
    return result;
}

auto SpGEMMBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result_tensors = backward({grad_outputs[0].tensor()});
    std::vector<Variable> results;
    results.reserve(result_tensors.size());
    for (auto& t : result_tensors) {
        results.emplace_back(t, grad_outputs[0].requires_grad());
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
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
}

} // namespace tenzor
