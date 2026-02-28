/**
 * @file oneapi_internal.hpp
 * @brief Internal interface for OneAPI kernel registry queue access
 *
 * Provides a mechanism for the kernel registry (which cannot include
 * the OneAPIBackend class definition) to access SYCL queues by device ID.
 * The backend sets itself as the queue provider during construction.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace tenzor::oneapi_internal {

/// Type-erased queue getter function pointer.
/// Signature: sycl::queue& (void* backend, int32_t device_id)
using QueueGetter = sycl::queue& (*)(void*, int32_t);

/// Set by OneAPIBackend constructor so kernel registry lambdas can obtain queues.
/// @param backend Pointer to the OneAPIBackend instance (type-erased to avoid
///                exposing the class definition outside oneapi_backend.cpp).
void set_backend_queue_provider(void* backend);

/// Set the queue getter callback. Called by OneAPIBackend constructor.
void set_queue_getter(QueueGetter fn);

/// Retrieve the SYCL queue for the given device ID.
/// @param device_id Zero-based device index
/// @return Reference to the device's SYCL queue
sycl::queue& get_queue(int32_t device_id);

} // namespace tenzor::oneapi_internal
