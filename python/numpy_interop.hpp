#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tenzor/core/tensor.hpp>
#include <tenzor/core/dtype.hpp>

namespace py = pybind11;

#ifdef _WIN32
// ssize_t is a POSIX typedef (<sys/types.h>) with no MSVC equivalent;
// pybind11 itself already defines py::ssize_t (aliasing Py_ssize_t) for
// exactly this reason. numpy_interop.cpp uses the bare, unqualified name.
using ssize_t = py::ssize_t;
#endif

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
 * Prepare tensor for NumPy conversion (pure C++, no Python objects).
 * Transfers non-CPU tensors to CPU and makes them contiguous if needed.
 * Can be called without the GIL held.
 *
 * @param tensor The input tensor
 * @return CPU tensor ready for NumPy array creation
 */
auto prepare_tensor_for_numpy(const Tensor& tensor) -> Tensor;

/**
 * Create NumPy array from a CPU tensor (requires GIL).
 * The tensor MUST be on CPU. Use prepare_tensor_for_numpy() first for non-CPU tensors.
 *
 * @param tensor CPU tensor
 * @param original_dtype Original dtype (for BFloat16 handling)
 * @param want_no_copy Y.25: when ``true`` and the strided view exceeds storage
 *        bounds (i.e. the function would otherwise fall back to a contiguous
 *        copy), throw ``py::value_error`` instead of warning and copying. The
 *        caller is honouring the NumPy 2.0 ``__array__(copy=False)`` contract.
 *        Default ``false`` preserves the legacy warn-and-copy behaviour.
 * @return NumPy array (may share memory with CPU tensor)
 */
auto create_numpy_array(const Tensor& tensor, DType original_dtype,
                        bool want_no_copy = false) -> py::array;

/**
 * Convert Tensor to NumPy array
 * Zero-copy when tensor is on CPU (supports strided views)
 * Always copies when tensor is on non-CPU devices (CUDA, Vulkan, ROCm, OneAPI)
 *
 * BFloat16 tensors (audit-5 Z.19): exposing a BFloat16 tensor to NumPy
 * requires the ``ml_dtypes`` package (``pip install ml_dtypes``). Without
 * it this function raises ``ValueError`` rather than silently returning a
 * raw uint16 buffer that would round-trip back as ``UInt16``.
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
