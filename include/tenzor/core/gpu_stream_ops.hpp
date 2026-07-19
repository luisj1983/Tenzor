#pragma once

#include "tenzor/core/device.hpp"
#include <cstddef>

// Backend-agnostic GPU stream/event primitives for communication/compute
// overlap (DDP bucket all-reduce, FSDP, ZeRO optimizer comm streams).
//
// Why this exists (see FINDING 60 in findings.txt): ddp.cpp/fsdp.cpp/
// zero_optimizer.cpp used to pick their stream/event API at COMPILE time via
// `#if defined(TENZOR_USE_CUDA) #elif defined(TENZOR_USE_ROCM)`, directly
// #include-ing <hip/hip_runtime.h> for the ROCm branch. On this project's
// default combined CUDA+ROCm build that's broken two separate ways:
//   1. TENZOR_USE_CUDA is defined tenzor_core-wide, so the #elif ROCm branch
//      can never be reached no matter what else is defined for that TU.
//   2. Even scoping TENZOR_USE_ROCM to just these files wouldn't help. and
//      would additionally be unsafe: <hip/hip_runtime.h> in a TU that also
//      sees <cuda_runtime.h> collides on ~30 make_*N/dim3-style vector-type
//      redefinitions (proved by actually trying it against
//      transfer_engine.hpp — see that header's FINDING 60 comment).
//
// This header instead dispatches at RUNTIME on the actual Device::Type of
// the tensors being communicated: CUDA calls go straight through (always
// compiled — TENZOR_USE_CUDA is project-wide), ROCm calls route through the
// isolated rocm_transfer.hip.cpp TU (compiled by hipcc) via its opaque
// void*-based API — the same pattern already proven for async CPU<->GPU
// transfers in transfer_engine.cpp.
namespace tenzor::core::gpu_stream {

/// Create a non-blocking stream for `type`. Returns nullptr for a
/// non-GPU/unsupported device type or if that backend isn't built in.
auto create_stream(Device::Type type) -> void*;

/// Destroy a stream created by create_stream(). Safe to call with nullptr.
auto destroy_stream(void* stream, Device::Type type) -> void;

/// Create a timing-disabled event for `type`. Returns nullptr for a
/// non-GPU/unsupported device type or if that backend isn't built in.
auto create_event(Device::Type type) -> void*;

/// Destroy an event created by create_event(). Safe to call with nullptr.
auto destroy_event(void* event, Device::Type type) -> void;

/// Record `event` on `stream`. `stream` may be nullptr (the default stream).
auto record_event(void* event, void* stream, Device::Type type) -> void;

/// Make `stream` wait (device-side, non-blocking) for `event`. `stream` may
/// be nullptr (the default stream).
auto stream_wait_event(void* stream, void* event, Device::Type type) -> void;

/// Block the calling host thread until `stream` finishes all queued work.
/// Safe to call with nullptr (no-op).
auto synchronize_stream(void* stream, Device::Type type) -> void;

/// Query free/total memory for `device`. Returns false (outputs untouched)
/// for a non-GPU/unsupported device type, if that backend isn't built in,
/// or on a runtime query failure.
auto mem_get_info(Device device, std::size_t* free_bytes, std::size_t* total_bytes) -> bool;

}  // namespace tenzor::core::gpu_stream
