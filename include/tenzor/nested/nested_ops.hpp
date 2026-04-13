/**
 * @file nested_ops.hpp
 * @brief Operations on NestedTensor
 *
 * Element-wise operations act directly on the contiguous values buffer.
 * Offset-aware operations (softmax along ragged dim, layer norm, attention)
 * dispatch through the kernel system for backend-specific implementations.
 */

#pragma once

#include "nested_tensor.hpp"

namespace tenzor {

// =========================================================================
// Element-wise Operations (operate on values_ directly)
// =========================================================================

auto nested_add(const NestedTensor& a, const NestedTensor& b) -> NestedTensor;
auto nested_sub(const NestedTensor& a, const NestedTensor& b) -> NestedTensor;
auto nested_mul(const NestedTensor& a, const NestedTensor& b) -> NestedTensor;
auto nested_div(const NestedTensor& a, const NestedTensor& b) -> NestedTensor;
auto nested_neg(const NestedTensor& a) -> NestedTensor;
auto nested_relu(const NestedTensor& a) -> NestedTensor;
auto nested_gelu(const NestedTensor& a) -> NestedTensor;
auto nested_sigmoid(const NestedTensor& a) -> NestedTensor;
auto nested_tanh(const NestedTensor& a) -> NestedTensor;
auto nested_abs(const NestedTensor& a) -> NestedTensor;

// =========================================================================
// Scalar Operations
// =========================================================================

auto nested_add_scalar(const NestedTensor& a, double scalar) -> NestedTensor;
auto nested_mul_scalar(const NestedTensor& a, double scalar) -> NestedTensor;

// =========================================================================
// Offset-aware Operations (need per-segment logic)
// =========================================================================

auto nested_softmax(const NestedTensor& input, int64_t dim) -> NestedTensor;
auto nested_log_softmax(const NestedTensor& input, int64_t dim) -> NestedTensor;
auto nested_layer_norm(const NestedTensor& input, const Tensor& weight,
                       const Tensor& bias, double eps = 1e-5) -> NestedTensor;
auto nested_sum(const NestedTensor& input, int64_t dim,
                bool keepdim = false) -> NestedTensor;
auto nested_mean(const NestedTensor& input, int64_t dim,
                 bool keepdim = false) -> NestedTensor;

// =========================================================================
// Compound Operations
// =========================================================================

/**
 * @brief Linear projection on nested tensor values.
 *
 * Since values_ is [total_len, D], standard matmul(values, weight^T)
 * works directly without needing offset awareness.
 */
auto nested_linear(const NestedTensor& input, const Tensor& weight,
                   const Tensor* bias = nullptr) -> NestedTensor;

/**
 * @brief Matrix multiply nested tensor values by a dense matrix.
 */
auto nested_matmul(const NestedTensor& a, const Tensor& b) -> NestedTensor;

/**
 * @brief Scaled dot-product attention on nested tensors.
 *
 * Query, key, value must have same batch structure (offsets).
 * Supports optional causal masking.
 *
 * @param scale Attention scale factor; negative means 1/sqrt(head_dim)
 * @param causal Apply causal (lower-triangular) mask
 */
auto nested_attention(const NestedTensor& query, const NestedTensor& key,
                      const NestedTensor& value, double scale = -1.0,
                      bool causal = false) -> NestedTensor;

// =========================================================================
// Manipulation
// =========================================================================

/**
 * @brief Concatenate nested tensors along the batch dimension.
 */
auto nested_cat(std::span<const NestedTensor> tensors, int64_t dim = 0) -> NestedTensor;

/**
 * @brief Apply dropout to nested tensor values.
 */
auto nested_dropout(const NestedTensor& input, double p, bool training) -> NestedTensor;

} // namespace tenzor
