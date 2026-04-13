/**
 * @file nested_ops.hpp
 * @brief Autograd-aware functional API for nested tensor operations.
 *
 * These functions wrap the raw nested tensor operations with Variable-based
 * gradient tracking. Element-wise ops on NestedTensor's values_ tensor
 * piggyback on the existing Variable autograd mechanism; only offset-aware
 * ops (softmax, layer norm, attention, reduction) need custom Function
 * subclasses defined in nested_autograd_ops.cpp.
 */

#pragma once

#include "variable.hpp"
#include "../core/tensor.hpp"

namespace tenzor::autograd {

/**
 * @brief Segmented softmax along a dimension within each ragged sequence.
 *
 * @param values  Variable wrapping the packed values tensor [total_len, D]
 * @param offsets Offset tensor [B+1] delimiting sequences in values
 * @param dim     Dimension along which to compute softmax (within each segment)
 * @return Variable with softmax applied per-segment
 */
auto nested_softmax(const Variable& values, const Tensor& offsets, int64_t dim) -> Variable;

/**
 * @brief Segmented layer normalization.
 *
 * Computes mean and variance within each segment, normalizes, then applies
 * affine transform (weight * normalized + bias).
 *
 * @param values  Variable wrapping packed values [total_len, D]
 * @param offsets Offset tensor [B+1]
 * @param weight  Learnable scale parameter [D]
 * @param bias    Learnable bias parameter [D]
 * @param eps     Small constant for numerical stability
 * @return Normalized Variable
 */
auto nested_layer_norm(const Variable& values, const Tensor& offsets,
                       const Variable& weight, const Variable& bias, double eps) -> Variable;

/**
 * @brief Linear projection on packed nested values.
 *
 * Equivalent to values @ weight^T + bias, applied identically to every row
 * regardless of segment boundaries. Because this is a standard matmul on
 * the packed values, the existing matmul autograd handles gradients
 * automatically. This wrapper simply performs the dispatch.
 *
 * @param values  Variable wrapping packed values [total_len, D_in]
 * @param offsets Offset tensor [B+1]
 * @param weight  Weight matrix [D_out, D_in]
 * @param bias    Optional bias [D_out] (nullptr to omit)
 * @return Projected Variable [total_len, D_out]
 */
auto nested_linear(const Variable& values, const Tensor& offsets,
                   const Variable& weight, const Variable* bias) -> Variable;

/**
 * @brief Variable-length self/cross-attention.
 *
 * Computes scaled dot-product attention per batch element, where query and
 * key/value sequences may have different lengths (cross-attention).
 *
 * @param q_values   Packed query values [total_q_len, head_dim]
 * @param k_values   Packed key values   [total_kv_len, head_dim]
 * @param v_values   Packed value values  [total_kv_len, head_dim]
 * @param q_offsets  Query offsets [B+1]
 * @param kv_offsets Key/value offsets [B+1]
 * @param scale      Attention scale factor (typically 1/sqrt(head_dim))
 * @param causal     Whether to apply causal masking
 * @return Attention output Variable [total_q_len, head_dim]
 */
auto nested_attention(const Variable& q_values, const Variable& k_values,
                      const Variable& v_values, const Tensor& q_offsets,
                      const Tensor& kv_offsets, double scale, bool causal) -> Variable;

/**
 * @brief Segmented sum reduction.
 *
 * Reduces each segment by summing along the ragged dimension, producing
 * a dense [B, D] output.
 *
 * @param values  Variable wrapping packed values [total_len, D]
 * @param offsets Offset tensor [B+1]
 * @return Dense Variable [B, D]
 */
auto nested_sum(const Variable& values, const Tensor& offsets) -> Variable;

/**
 * @brief Segmented mean reduction.
 *
 * Reduces each segment by averaging along the ragged dimension.
 *
 * @param values  Variable wrapping packed values [total_len, D]
 * @param offsets Offset tensor [B+1]
 * @return Dense Variable [B, D]
 */
auto nested_mean(const Variable& values, const Tensor& offsets) -> Variable;

} // namespace tenzor::autograd
