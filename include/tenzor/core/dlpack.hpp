/**
 * @file dlpack.hpp
 * @brief Zero-copy tensor exchange with NumPy 2.0+, JAX, PyTorch, CuPy, etc.
 *
 * DLPack is the open in-memory tensor exchange protocol. These helpers let
 * Tenzor tensors cross library boundaries without a data copy:
 *
 *   1. Tenzor -> DLPack:  to_dlpack(tensor)
 *      Returns a heap-allocated `DLManagedTensor*`. The returned object is
 *      owned by the caller (or the Python capsule holding it). Its
 *      `deleter` releases a reference on the original Tensor's storage so
 *      the data stays alive until the DLPack consumer is done.
 *
 *   2. DLPack -> Tenzor:  from_dlpack(managed)
 *      Takes ownership of the incoming DLManagedTensor (calls its deleter
 *      when the resulting Tensor's storage is released) and wraps its data
 *      as a Tenzor with the appropriate dtype / device / shape.
 *
 * Constraints:
 *   - Import currently requires contiguous DLPack tensors (strides == NULL
 *     or standard C-contiguous strides). Non-contiguous imports throw.
 *   - Only dtypes Tenzor natively supports are accepted. Complex exports
 *     use dtype code kDLComplex with bits=64 or 128.
 *   - CUDA, ROCm, Vulkan, OneAPI, CPU devices are all mapped. Unsupported
 *     device types throw on both directions.
 */

#pragma once

#include "tenzor/core/tensor.hpp"
#include "tenzor/external/dlpack/dlpack.h"

namespace tenzor {

/**
 * @brief Export a Tenzor as a DLPack `DLManagedTensor*`.
 *
 * The returned pointer is heap-allocated; the caller (or the Python
 * capsule consumer) must invoke `managed->deleter(managed)` exactly once
 * to release the held reference on the source tensor's storage.
 *
 * The tensor must be contiguous OR have well-defined strides — strided
 * views are exported with their actual strides so the consumer can
 * honor the memory layout.
 *
 * @param tensor Source tensor (kept alive until deleter runs)
 * @return Newly allocated DLManagedTensor
 * @throws std::runtime_error if dtype or device isn't DLPack-representable
 */
auto to_dlpack(const Tensor& tensor) -> DLManagedTensor*;

/**
 * @brief Import a DLPack `DLManagedTensor*` as a Tenzor.
 *
 * Transfers ownership of `managed` to the resulting Tenzor: when the
 * Tenzor's storage is released, `managed->deleter(managed)` is invoked
 * to release the producer's resources. The caller must not touch
 * `managed` after this call succeeds.
 *
 * @param managed Incoming DLManagedTensor (ownership transferred on success)
 * @return A Tenzor wrapping `managed->dl_tensor.data` (no copy)
 * @throws std::runtime_error on dtype / device / layout mismatch. On
 *         error, ownership is NOT transferred and the caller must still
 *         delete `managed` itself.
 */
auto from_dlpack(DLManagedTensor* managed) -> Tensor;

} // namespace tenzor
