/**
 * @file gradient_utils.hpp
 * @brief Utility functions for gradient operations in ZeRO Stage 2
 *
 * Provides functions for gradient flattening, bucketing, and memory management
 * needed by ZeRO Stage 2 gradient partitioning.
 */

#pragma once

#include "../../core/tensor.hpp"
#include "../../autograd/variable.hpp"
#include <vector>
#include <cstddef>
#include <memory>

namespace tenzor {
namespace optim {

/**
 * @brief Flatten a vector of tensors into a single contiguous tensor
 *
 * Combines multiple tensors with potentially different shapes into a single
 * 1D tensor by concatenating their data in memory order.
 *
 * All input tensors must:
 * - Have the same dtype
 * - Be on the same device
 * - Be contiguous in memory (call .contiguous() if needed)
 *
 * @param tensors Vector of tensors to flatten
 * @return Single 1D tensor containing all data from input tensors
 * @throws std::invalid_argument if tensors have different dtypes or devices
 * @throws std::invalid_argument if input is empty
 *
 * @code
 * std::vector<Tensor> grads = {
 *     Tensor({10, 20}, DType::Float32, Device::cpu()),
 *     Tensor({5, 5}, DType::Float32, Device::cpu())
 * };
 * Tensor flat = flatten_tensors(grads);  // Shape: {225} (200 + 25)
 * @endcode
 */
auto flatten_tensors(const std::vector<Tensor>& tensors) -> Tensor;

/**
 * @brief Flatten gradients from a vector of Variables
 *
 * Convenience overload that extracts and flattens gradients from Variables.
 * Skips Variables with null gradients.
 *
 * @param variables Vector of Variables with gradients
 * @return Single 1D tensor containing all gradient data
 * @throws std::invalid_argument if no Variables have gradients
 *
 * @code
 * std::vector<std::shared_ptr<Variable>> params = model.parameters();
 * Tensor flat_grad = flatten_tensors(params);
 * @endcode
 */
auto flatten_tensors(const std::vector<std::shared_ptr<Variable>>& variables) -> Tensor;

/**
 * @brief Unflatten a contiguous tensor back into a vector of tensors
 *
 * Splits a flattened 1D tensor back into multiple tensors with specified shapes.
 * This is the inverse operation of flatten_tensors().
 *
 * The sum of elements in shapes must equal the number of elements in the input tensor.
 *
 * @param flat_tensor Flattened 1D tensor
 * @param shapes Vector of shapes for output tensors
 * @return Vector of tensors with specified shapes
 * @throws std::invalid_argument if shapes don't match flat_tensor size
 * @throws std::invalid_argument if flat_tensor is not 1D
 *
 * @code
 * Tensor flat = Tensor({225}, DType::Float32, Device::cpu());
 * std::vector<std::vector<int64_t>> shapes = {{10, 20}, {5, 5}};
 * auto tensors = unflatten_into(flat, shapes);  // 2 tensors: {10,20} and {5,5}
 * @endcode
 */
auto unflatten_into(const Tensor& flat_tensor,
                   const std::vector<std::vector<int64_t>>& shapes) -> std::vector<Tensor>;

/**
 * @brief Unflatten a tensor and copy into existing tensor views
 *
 * Splits a flattened tensor and copies data into provided output tensors.
 * Output tensors must already be allocated with correct shapes and dtypes.
 *
 * This is more efficient than unflatten_into() when output tensors are already allocated,
 * as it avoids creating new tensors.
 *
 * @param flat_tensor Flattened 1D tensor
 * @param output_tensors Pre-allocated output tensors to copy into
 * @throws std::invalid_argument if output tensors don't match flat_tensor size
 * @throws std::invalid_argument if dtypes don't match
 *
 * @code
 * Tensor flat = flatten_tensors(gradients);
 * // ... communicate flat tensor ...
 * unflatten_into(flat, gradients);  // Copy back into original gradients
 * @endcode
 */
auto unflatten_into(const Tensor& flat_tensor,
                   std::vector<Tensor>& output_tensors) -> void;

} // namespace optim
} // namespace tenzor
