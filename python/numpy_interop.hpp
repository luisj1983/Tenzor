#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tenzor/core/tensor.hpp>
#include <tenzor/core/dtype.hpp>

namespace py = pybind11;

namespace tenzor {
namespace numpy {

/**
 * Convert Tenzor DType to NumPy dtype string
 * @param dtype The Tenzor DType
 * @return NumPy format string
 */
auto dtype_to_numpy_format(DType dtype) -> std::string;

/**
 * Convert NumPy dtype to Tenzor DType
 * @param arr NumPy array
 * @return Tenzor DType
 * @throws std::runtime_error if dtype is unsupported
 */
auto numpy_dtype_to_tenzor(const py::array& arr) -> DType;

/**
 * Get NumPy dtype size
 * @param arr NumPy array
 * @return Size in bytes
 */
auto get_numpy_itemsize(const py::array& arr) -> size_t;

/**
 * Convert Tensor to NumPy array
 * Zero-copy when tensor is on CPU (supports strided views)
 * Always copies when tensor is on non-CPU devices (CUDA, Vulkan, ROCm, OneAPI)
 *
 * @param tensor The input tensor
 * @return NumPy array (may share memory with CPU tensor)
 */
auto tensor_to_numpy(const Tensor& tensor) -> py::array;

/**
 * Convert NumPy array to Tensor
 * Zero-copy when possible (aligned, contiguous, CPU device)
 * Copies when necessary (non-contiguous, misaligned, or CUDA requested)
 *
 * @param arr NumPy array
 * @param device Target device (default: CPU)
 * @return Tensor (may share memory with NumPy array)
 */
auto numpy_to_tensor(py::array arr, Device device = Device::cpu()) -> Tensor;

/**
 * Check if zero-copy is possible for numpy to tensor conversion
 * @param arr NumPy array
 * @return true if zero-copy is possible
 */
auto can_zero_copy_numpy_to_tensor(const py::array& arr) -> bool;

/**
 * Check if zero-copy is possible for tensor to numpy conversion
 * @param tensor Input tensor
 * @return true if zero-copy is possible
 */
auto can_zero_copy_tensor_to_numpy(const Tensor& tensor) -> bool;

} // namespace numpy
} // namespace tenzor
