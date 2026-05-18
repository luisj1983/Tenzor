/**
 * @file foreach.hpp
 * @brief Multi-tensor (foreach) optimizer ops
 *
 * PyTorch-style _foreach_* ops that apply the same element-wise operation
 * to every tensor in a list in a single C++ call. Significantly faster than
 * a Python for-loop over per-tensor ops because:
 *   (a) one C++ call boundary instead of N,
 *   (b) OMP-parallel across tensors for lists of 4 or more,
 *   (c) amortized tensor metadata access.
 *
 * Initial implementation loops over existing per-tensor kernels (correctness
 * first). Fused single-pass kernels are a follow-up perf optimization.
 *
 * All functions assert that all input lists have the same size. Mismatched
 * sizes throw std::invalid_argument.
 */

#pragma once

#include <vector>
#include "tenzor/core/tensor.hpp"

namespace tenzor {

// =============================================================================
// Out-of-place binary ops (list, list) -> list
// =============================================================================

/** @brief Element-wise a[i] + b[i] for each tensor pair. */
auto foreach_add(const std::vector<Tensor>& a, const std::vector<Tensor>& b)
    -> std::vector<Tensor>;

/** @brief Element-wise a[i] - b[i] for each tensor pair. */
auto foreach_sub(const std::vector<Tensor>& a, const std::vector<Tensor>& b)
    -> std::vector<Tensor>;

/** @brief Element-wise a[i] * b[i] for each tensor pair. */
auto foreach_mul(const std::vector<Tensor>& a, const std::vector<Tensor>& b)
    -> std::vector<Tensor>;

/** @brief Element-wise a[i] / b[i] for each tensor pair. */
auto foreach_div(const std::vector<Tensor>& a, const std::vector<Tensor>& b)
    -> std::vector<Tensor>;

// =============================================================================
// Out-of-place unary ops (list) -> list
// =============================================================================

/** @brief Element-wise -a[i] for each tensor. */
auto foreach_neg(const std::vector<Tensor>& a) -> std::vector<Tensor>;

/** @brief Element-wise |a[i]| for each tensor. */
auto foreach_abs(const std::vector<Tensor>& a) -> std::vector<Tensor>;

/** @brief Element-wise sqrt(a[i]) for each tensor. */
auto foreach_sqrt(const std::vector<Tensor>& a) -> std::vector<Tensor>;

/** @brief Return deep copies of all tensors in the list. */
auto foreach_copy(const std::vector<Tensor>& src) -> std::vector<Tensor>;

// =============================================================================
// In-place binary ops (list, list) -> void
// =============================================================================

/** @brief a[i] += b[i] for each tensor pair. */
void foreach_add_(std::vector<Tensor>& a, const std::vector<Tensor>& b);

/** @brief a[i] -= b[i] for each tensor pair. */
void foreach_sub_(std::vector<Tensor>& a, const std::vector<Tensor>& b);

/** @brief a[i] *= b[i] for each tensor pair. */
void foreach_mul_(std::vector<Tensor>& a, const std::vector<Tensor>& b);

/** @brief a[i] /= b[i] for each tensor pair. */
void foreach_div_(std::vector<Tensor>& a, const std::vector<Tensor>& b);

// =============================================================================
// In-place unary ops (list) -> void
// =============================================================================

/** @brief a[i] = -a[i] for each tensor (in-place). */
void foreach_neg_(std::vector<Tensor>& a);

/** @brief a[i] = |a[i]| for each tensor (in-place). */
void foreach_abs_(std::vector<Tensor>& a);

/** @brief a[i] = sqrt(a[i]) for each tensor (in-place). */
void foreach_sqrt_(std::vector<Tensor>& a);

/** @brief a[i] = 0 for each tensor (zero-fill in place). */
void foreach_zero_(std::vector<Tensor>& a);

// =============================================================================
// Ternary fused ops
// =============================================================================

/**
 * @brief self[i] += scalar * a[i] / b[i]  (in-place, addcdiv)
 *
 * Equivalent to PyTorch _foreach_addcdiv_.
 */
void foreach_addcdiv_(std::vector<Tensor>& self,
                      const std::vector<Tensor>& a,
                      const std::vector<Tensor>& b,
                      double scalar = 1.0);

/**
 * @brief self[i] += scalar * a[i] * b[i]  (in-place, addcmul)
 *
 * Equivalent to PyTorch _foreach_addcmul_.
 */
void foreach_addcmul_(std::vector<Tensor>& self,
                      const std::vector<Tensor>& a,
                      const std::vector<Tensor>& b,
                      double scalar = 1.0);

/**
 * @brief self[i] = self[i] + scalar * (b[i] - self[i])  (in-place lerp)
 *
 * Equivalent to: self[i] = (1 - scalar) * self[i] + scalar * b[i].
 * Equivalent to PyTorch _foreach_lerp_.
 */
void foreach_lerp_(std::vector<Tensor>& self,
                   const std::vector<Tensor>& b,
                   double scalar);

// =============================================================================
// Reduction
// =============================================================================

/**
 * @brief Compute p-norm of each tensor in the list.
 *
 * Returns a list of scalar tensors, one per input tensor.
 * Equivalent to PyTorch _foreach_norm with ord=p.
 *
 * @param a Input tensor list.
 * @param p Norm order (default 2.0 = L2 norm).
 */
auto foreach_norm(const std::vector<Tensor>& a, double p = 2.0) -> std::vector<Tensor>;

} // namespace tenzor
