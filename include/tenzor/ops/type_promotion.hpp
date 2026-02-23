/**
 * @file type_promotion.hpp
 * @brief NumPy/PyTorch-compatible automatic type promotion for binary operations
 */

#pragma once

#include "../core/dtype.hpp"
#include "../core/tensor.hpp"

namespace tenzor {

/**
 * @brief Determine the promoted dtype for two dtypes.
 *
 * Follows NumPy/PyTorch promotion rules:
 * - Float64 > Float32 > Float16/BFloat16
 * - Int64 > Int32 > Int16 > Int8/UInt8
 * - Float wins over Int (Int32 + Float32 -> Float32)
 * - Bool promotes to anything
 * - Complex128 > Complex64, complex wins over real
 */
auto promote_types(DType a, DType b) -> DType;

/**
 * @brief Determine the result dtype for a binary operation on two tensors.
 */
auto result_type(const Tensor& a, const Tensor& b) -> DType;

/**
 * @brief Promote two tensors to a common dtype, converting if necessary.
 * @return Pair of tensors with matching dtype (may be the originals if already matching)
 */
auto promote_inputs(const Tensor& a, const Tensor& b) -> std::pair<Tensor, Tensor>;

} // namespace tenzor
