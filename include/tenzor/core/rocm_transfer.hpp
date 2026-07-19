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

// ---------------------------------------------------------------------------
// Generic stream/event primitives for GPU communication/compute overlap
// (DDP bucket all-reduce, FSDP, ZeRO optimizer comm streams — see FINDING 60
// in findings.txt). These mirror the CUDA stream/event API 1:1 but are
// backed by real HIP calls in rocm_transfer.hip.cpp, reached through this
// HIP-type-free header so callers never need <hip/hip_runtime.h>.
// ---------------------------------------------------------------------------

/// Create a non-blocking HIP stream. Returns nullptr on failure or when ROCm
/// is unavailable.
auto stream_create() -> void*;

/// Destroy a stream created by stream_create(). Safe to call with nullptr.
auto stream_destroy(void* stream) -> void;

/// Create a timing-disabled HIP event. Returns nullptr on failure or when
/// ROCm is unavailable.
auto event_create() -> void*;

/// Destroy an event created by event_create() (NOT one returned by
/// h2d_async/d2h_async — those are pool-managed and released via
/// event_sync()). Safe to call with nullptr.
auto event_destroy(void* event) -> void;

/// Record `event` on `stream`. `stream` may be nullptr (the default stream).
auto event_record(void* event, void* stream) -> void;

/// Make `stream` wait (device-side, non-blocking) for `event`. `stream` may
/// be nullptr (the default stream).
auto stream_wait_event(void* stream, void* event) -> void;

/// Block the calling host thread until `stream` finishes all queued work.
/// Safe to call with nullptr (no-op).
auto stream_synchronize(void* stream) -> void;

/// Query free/total device memory for `device`. Returns false on failure or
/// when ROCm is unavailable (outputs left untouched).
auto mem_get_info(int device, std::size_t* free_bytes, std::size_t* total_bytes) -> bool;

}  // namespace rocm_transfer
}  // namespace tenzor
