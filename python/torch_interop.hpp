/**
 * @file torch_interop.hpp
 * @brief PyTorch tensor interoperability for Tenzor
 *
 * Provides zero-copy conversion between PyTorch tensors and Tenzor tensors
 * when memory layout and device allow it.
 */

#pragma once

#include <tenzor/core/tensor.hpp>
#include <tenzor/core/dtype.hpp>
#include <tenzor/core/device.hpp>
#include <tenzor/autograd/variable.hpp>  // tenzor::Variable (variable_* / sync_gradients)

// Modern libtorch (>=2.x) defines torch::Tensor and torch::autograd::Variable
// as type aliases (using-declarations over at::Tensor), not forward-declarable
// classes. Forward-declaring `class Variable` here conflicts with torch's own
// `using Variable = at::Tensor` (a hard "conflicting declaration" error that
// also corrupts downstream template/std lookup), so this header must include
// the real torch headers instead. It is only ever included by translation
// units built with TENZOR_HAS_TORCH (torch_interop.cpp and the guarded block
// in bindings.cpp), so the libtorch dependency is unconditional here.
#include <torch/torch.h>

namespace tenzor {
namespace torch_interop {

/**
 * @brief Check if zero-copy conversion from Tenzor to PyTorch is possible
 *
 * Zero-copy is possible when:
 * - Tensor is contiguous
 * - Device is compatible (CPU-CPU, CUDA-CUDA)
 * - Data type is supported by PyTorch
 *
 * @param tensor Tenzor tensor
 * @return true if zero-copy is possible
 */
auto can_zero_copy_to_torch(const Tensor& tensor) -> bool;

/**
 * @brief Check if zero-copy conversion from PyTorch to Tenzor is possible
 *
 * @param torch_tensor PyTorch tensor
 * @return true if zero-copy is possible
 */
auto can_zero_copy_from_torch(const torch::Tensor& torch_tensor) -> bool;

/**
 * @brief Convert Tenzor tensor to PyTorch tensor
 *
 * Performs zero-copy conversion when possible, otherwise copies data.
 * For CUDA tensors, uses the same device pointer.
 *
 * @param tensor Tenzor tensor
 * @param requires_grad Whether PyTorch tensor should track gradients
 * @return PyTorch tensor
 * @throws std::runtime_error if conversion fails
 */
auto tensor_to_torch(const Tensor& tensor, bool requires_grad = false) -> torch::Tensor;

/**
 * @brief Convert PyTorch tensor to Tenzor tensor
 *
 * Performs zero-copy conversion when possible, otherwise copies data.
 *
 * @param torch_tensor PyTorch tensor
 * @param device Target device (if not specified, uses PyTorch tensor's device)
 * @return Tenzor tensor
 * @throws std::runtime_error if conversion fails
 */
auto tensor_from_torch(const torch::Tensor& torch_tensor,
                       std::optional<Device> device = std::nullopt) -> Tensor;

/**
 * @brief Convert Tenzor Variable to PyTorch Variable (with autograd)
 *
 * Converts gradient-tracking Variable to PyTorch Variable.
 * If both have gradients, they share gradient storage when possible.
 *
 * @param variable Tenzor Variable
 * @return PyTorch Variable (autograd::Variable)
 */
auto variable_to_torch(const Variable& variable) -> torch::autograd::Variable;

/**
 * @brief Convert PyTorch Variable to Tenzor Variable
 *
 * @param torch_variable PyTorch Variable
 * @return Tenzor Variable
 */
auto variable_from_torch(const torch::autograd::Variable& torch_variable) -> Variable;

/**
 * @brief Map Tenzor DType to PyTorch ScalarType
 *
 * @param dtype Tenzor data type
 * @return PyTorch scalar type
 * @throws std::runtime_error if dtype not supported by PyTorch
 */
auto dtype_to_torch(DType dtype) -> int; // Returns torch::ScalarType as int

/**
 * @brief Map PyTorch ScalarType to Tenzor DType
 *
 * @param torch_dtype PyTorch scalar type (as int)
 * @return Tenzor data type
 * @throws std::runtime_error if type not supported
 */
auto dtype_from_torch(int torch_dtype) -> DType;

/**
 * @brief Map Tenzor Device to PyTorch Device
 *
 * @param device Tenzor device
 * @return PyTorch device (as string like "cpu" or "cuda:0")
 */
auto device_to_torch_string(const Device& device) -> std::string;

/**
 * @brief Map PyTorch Device to Tenzor Device
 *
 * @param device_str PyTorch device string ("cpu", "cuda", "cuda:0", etc.)
 * @return Tenzor device
 * @throws std::runtime_error if device string invalid
 */
auto device_from_torch_string(const std::string& device_str) -> Device;

/**
 * @brief Synchronize gradient storage between Tenzor and PyTorch
 *
 * After backward pass, synchronize gradients so both frameworks
 * see the same gradient values.
 *
 * @param tenzor_var Tenzor Variable
 * @param torch_var PyTorch Variable
 * @param direction true = Tenzor -> PyTorch, false = PyTorch -> Tenzor
 */
auto sync_gradients(Variable& tenzor_var,
                   torch::autograd::Variable& torch_var,
                   bool tenzor_to_torch = true) -> void;

} // namespace torch_interop
} // namespace tenzor
