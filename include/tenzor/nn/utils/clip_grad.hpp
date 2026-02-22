/**
 * @file clip_grad.hpp
 * @brief Gradient clipping utilities for training stability
 *
 * Provides gradient clipping by norm and by value, compatible with
 * PyTorch's torch.nn.utils.clip_grad_norm_ and clip_grad_value_.
 */

#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include "../../autograd/variable.hpp"
#include "../../ops/reduction.hpp"
#include "../../ops/math.hpp"

namespace tenzor::nn::utils {

/**
 * @brief Clip gradients by global norm (PyTorch-compatible).
 *
 * Computes the total norm of all parameter gradients, then scales them
 * so the total norm does not exceed max_norm.
 *
 * @param parameters Parameters whose gradients will be clipped
 * @param max_norm Maximum allowed total norm
 * @param norm_type Type of the p-norm (default: 2.0 for L2 norm)
 * @return The total norm of the gradients before clipping
 */
auto clip_grad_norm_(std::vector<std::shared_ptr<Variable>> parameters,
                     double max_norm,
                     double norm_type = 2.0) -> double;

/**
 * @brief Clip gradients by value.
 *
 * Clamps each gradient element to [-clip_value, clip_value].
 *
 * @param parameters Parameters whose gradients will be clipped
 * @param clip_value Maximum absolute value for gradient elements
 */
void clip_grad_value_(std::vector<std::shared_ptr<Variable>> parameters,
                      double clip_value);

} // namespace tenzor::nn::utils
