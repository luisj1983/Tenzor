/**
 * @file jvp_rules.hpp
 * @brief JVP (Jacobian-Vector Product) rules for forward-mode AD
 *
 * Each rule takes dual inputs and returns a dual output, propagating
 * tangent vectors through the operation using the chain rule.
 */

#pragma once

#include "dual.hpp"
#include <optional>
#include <vector>
#include <cstdint>

namespace tenzor {

/// @name Arithmetic JVP Rules
/// @{
auto jvp_add(const DualTensor& a, const DualTensor& b) -> DualTensor;
auto jvp_sub(const DualTensor& a, const DualTensor& b) -> DualTensor;
auto jvp_mul(const DualTensor& a, const DualTensor& b) -> DualTensor;
auto jvp_div(const DualTensor& a, const DualTensor& b) -> DualTensor;
auto jvp_neg(const DualTensor& a) -> DualTensor;
/// @}

/// @name Matrix JVP Rules
/// @{
auto jvp_matmul(const DualTensor& a, const DualTensor& b) -> DualTensor;
/// @}

/// @name Activation JVP Rules
/// @{
auto jvp_relu(const DualTensor& x) -> DualTensor;
auto jvp_sigmoid(const DualTensor& x) -> DualTensor;
auto jvp_tanh(const DualTensor& x) -> DualTensor;
auto jvp_gelu(const DualTensor& x) -> DualTensor;
/// @}

/// @name Math JVP Rules
/// @{
auto jvp_exp(const DualTensor& x) -> DualTensor;
auto jvp_log(const DualTensor& x) -> DualTensor;
auto jvp_sqrt(const DualTensor& x) -> DualTensor;
auto jvp_pow(const DualTensor& x, double exponent) -> DualTensor;
auto jvp_sin(const DualTensor& x) -> DualTensor;
auto jvp_cos(const DualTensor& x) -> DualTensor;
auto jvp_abs(const DualTensor& x) -> DualTensor;
/// @}

/// @name Reduction JVP Rules
/// @{
auto jvp_sum(const DualTensor& x,
             std::optional<int64_t> dim = std::nullopt,
             bool keepdim = false) -> DualTensor;
auto jvp_mean(const DualTensor& x,
              std::optional<int64_t> dim = std::nullopt,
              bool keepdim = false) -> DualTensor;
/// @}

/// @name Shape JVP Rules
/// @{
auto jvp_reshape(const DualTensor& x, std::vector<int64_t> shape) -> DualTensor;
auto jvp_transpose(const DualTensor& x, int64_t dim0, int64_t dim1) -> DualTensor;
/// @}

} // namespace tenzor
