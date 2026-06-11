#pragma once

#include <cstddef>

// Opaque host-side interface to ROCm (HIP) CPU<->GPU DMA.
//
// Why this exists: tenzor_core is a host (g++) library compiled with the CUDA
// backend enabled, so its translation units include <cuda_runtime.h>. Including
// <hip/hip_runtime.h> in the same TU collides with CUDA on the make_*N vector
// helper definitions (CUDA vector_functions.hpp vs HIP amd_hip_vector_types.h),
// which is why the ROCm path in transfer_engine.cpp was compiled out and offload
// silently returned zero-filled tensors. Rather than mix the headers, the actual
// HIP calls live in rocm_transfer.hip.cpp (compiled by hipcc) and are reached
// through these HIP-type-free signatures — no HIP headers leak into core host TUs.
namespace tenzor {
namespace rocm_transfer {

/// True when the real HIP-backed implementation is linked in (ROCm build).
auto available() -> bool;

/// Asynchronous host->device copy on a pooled, non-blocking HIP stream.
/// Returns an opaque event handle that completes when the copy finishes
/// (nullptr for a zero-byte copy or when ROCm is unavailable).
auto h2d_async(void* dst, const void* src, std::size_t bytes, int device) -> void*;

/// Asynchronous device->host copy. See h2d_async.
auto d2h_async(void* dst, const void* src, std::size_t bytes, int device) -> void*;

/// Block until the event completes, then return it to the internal pool.
/// Safe to call with nullptr (no-op). Each event handle must be passed here
/// exactly once.
auto event_sync(void* event) -> void;

/// Non-blocking completion query. Returns true if complete (or event==nullptr).
/// Does NOT release the event — call event_sync() to release.
auto event_ready(void* event) -> bool;

}  // namespace rocm_transfer
}  // namespace tenzor
