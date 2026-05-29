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

// ---------------------------------------------------------------------------
// DLPack v1.0 "versioned" ABI.
//
// The bundled dlpack.h is v0.8 (DLPACK_VERSION 80) and predates the versioned
// container (`DLManagedTensorVersioned`) introduced in DLPack 1.0. Consumers
// that call ``__dlpack__(max_version=(1, x))`` (NumPy >= 1.23, PyTorch, JAX)
// can accept either the legacy unversioned ``"dltensor"`` capsule or the new
// ``"dltensor_versioned"`` capsule. We define the v1.0 structs here (matching
// the standardised byte layout) so the producer can emit a versioned capsule
// when negotiated. Guarded so a future header upgrade that defines them wins.
#ifndef DLPACK_MAJOR_VERSION
#define DLPACK_MAJOR_VERSION 1
#define DLPACK_MINOR_VERSION 0

/// Bit in DLManagedTensorVersioned::flags marking the data as read-only.
#define DLPACK_FLAG_BITMASK_READ_ONLY (1UL << 0)
/// Bit marking the tensor as a fresh copy the consumer may freely mutate.
#define DLPACK_FLAG_BITMASK_IS_COPIED (1UL << 1)

typedef struct {
    uint32_t major;  ///< DLPack major version of the producer.
    uint32_t minor;  ///< DLPack minor version of the producer.
} DLPackVersion;

/// Versioned DLPack container (DLPack >= 1.0). Field order/layout is part of
/// the cross-library ABI and must not be reordered.
typedef struct DLManagedTensorVersioned {
    DLPackVersion version;  ///< Producer's DLPack version.
    void* manager_ctx;      ///< Producer-owned context passed to deleter.
    /// Releases manager_ctx; called exactly once by whoever owns the capsule.
    void (*deleter)(struct DLManagedTensorVersioned* self);
    uint64_t flags;         ///< OR of DLPACK_FLAG_BITMASK_* values.
    DLTensor dl_tensor;     ///< The tensor being exported.
} DLManagedTensorVersioned;
#endif  // DLPACK_MAJOR_VERSION

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
 * @brief Export a Tenzor as a DLPack v1.0 `DLManagedTensorVersioned*`.
 *
 * Identical ownership semantics to ::to_dlpack (the returned pointer is
 * heap-allocated; `managed->deleter(managed)` must be invoked exactly once),
 * but wraps the tensor in the versioned container negotiated via
 * `__dlpack__(max_version=(1, x))`. The `version` field advertises
 * (DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION); `flags` is 0 (writable,
 * not a copy).
 *
 * @param tensor Source tensor (kept alive until deleter runs)
 * @return Newly allocated DLManagedTensorVersioned
 * @throws std::runtime_error if dtype or device isn't DLPack-representable
 */
auto to_dlpack_versioned(const Tensor& tensor) -> DLManagedTensorVersioned*;

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
