// Real HIP-backed implementation of the opaque ROCm transfer API declared in
// rocm_transfer.hpp. Compiled by hipcc (LANGUAGE HIP), so it is the ONLY core
// TU that includes <hip/hip_runtime.h> — keeping HIP's vector-type definitions
// out of the host TUs that also include CUDA headers (see the header comment).
#include "tenzor/core/rocm_transfer.hpp"

#include <hip/hip_runtime.h>

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {
namespace rocm_transfer {
namespace {

constexpr int kNumStreams = 4;

std::once_flag g_init_flag;
std::vector<hipStream_t> g_streams;
std::atomic<unsigned> g_round_robin{0};

std::mutex g_event_mutex;
std::vector<hipEvent_t> g_event_pool;

void init_streams() {
    g_streams.resize(kNumStreams);
    for (auto& s : g_streams) {
        // Non-blocking streams so transfers overlap with compute on the default stream.
        (void)hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    }
}

auto acquire_event() -> hipEvent_t {
    std::lock_guard<std::mutex> lock(g_event_mutex);
    if (!g_event_pool.empty()) {
        hipEvent_t e = g_event_pool.back();
        g_event_pool.pop_back();
        return e;
    }
    hipEvent_t e = nullptr;
    (void)hipEventCreateWithFlags(&e, hipEventDisableTiming);
    return e;
}

void release_event(hipEvent_t e) {
    std::lock_guard<std::mutex> lock(g_event_mutex);
    g_event_pool.push_back(e);
}

auto do_copy(void* dst, const void* src, std::size_t bytes, int device, hipMemcpyKind kind) -> void* {
    if (bytes == 0) return nullptr;
    std::call_once(g_init_flag, init_streams);
    (void)hipSetDevice(device);
    hipStream_t stream = g_streams[g_round_robin.fetch_add(1, std::memory_order_relaxed) % kNumStreams];

    // FINDING 25: g_streams are created hipStreamNonBlocking specifically so
    // transfers overlap with compute on the default stream -- but that
    // explicitly OPTS OUT of the implicit ordering the (legacy) default
    // stream would otherwise give for free. For a DeviceToHost copy whose
    // device `src` was JUST produced by a regular tensor-op kernel (dispatched
    // on the default stream by the normal ROCm kernel-dispatch path, not
    // through this isolated TU), nothing here ever waited for that producer
    // kernel to reach the GPU before the DMA on g_streams[i] started -- a
    // genuine missing cross-stream dependency, not just a theoretical race:
    // reproduced via OffloadContext's Int8-with-scale parameter offload
    // (quantize kernels write `src` on the default stream, then
    // gpu_to_cpu_async's ROCm path lands here), where later parameters in a
    // multi-tensor offload loop non-deterministically read stale/partially-
    // written GPU memory, corrupting the dequantized result.
    //
    // The original fix was a full hipDeviceSynchronize() before every copy.
    // That is correct but blunt: it waits for ALL streams, including other
    // in-flight DMAs on g_streams, which fully serialises concurrent async
    // transfers and destroys the overlap these streams exist to provide (the
    // async-overlap benchmarks measured ~0.6x -- parallel slower than serial).
    //
    // The targeted fix: only D2H can have a cross-stream producer (the device
    // `src` may have been written by a default-stream kernel). H2D's `src` is
    // host memory (always stable to read) and its `dst` is a freshly allocated
    // GPU buffer with no prior writer, so H2D needs no pre-dependency. For D2H
    // we establish the dependency with a NON-BLOCKING default-stream event:
    // record an event capturing all work already submitted to stream 0, then
    // have this DMA's g_stream wait on it. This orders the DMA after the
    // producer kernel without waiting for unrelated DMAs on other g_streams,
    // so concurrent D2H transfers still overlap. Once hipStreamWaitEvent
    // returns the dependency is recorded in the stream and the event may be
    // destroyed, so there is no per-transfer event leak.
    if (kind == hipMemcpyDeviceToHost) {
        hipEvent_t dep = nullptr;
        (void)hipEventCreateWithFlags(&dep, hipEventDisableTiming);
        (void)hipEventRecord(dep, /*stream=*/0);
        (void)hipStreamWaitEvent(stream, dep, 0);
        (void)hipEventDestroy(dep);
    }

    // A nullptr return means "no event / nothing to wait on" and callers treat
    // it as already-completed (see transfer_engine cpu_to_gpu_async). A failed
    // DMA must therefore NOT share that sentinel, or the never-filled
    // destination tensor would silently be reported as a completed transfer.
    if (hipError_t err = hipMemcpyAsync(dst, src, bytes, kind, stream);
        err != hipSuccess) {
        throw std::runtime_error(
            std::string("rocm_transfer::do_copy: hipMemcpyAsync failed: ") +
            hipGetErrorString(err));
    }
    hipEvent_t event = acquire_event();
    (void)hipEventRecord(event, stream);
    return static_cast<void*>(event);
}

}  // namespace

auto available() -> bool { return true; }

auto h2d_async(void* dst, const void* src, std::size_t bytes, int device) -> void* {
    return do_copy(dst, src, bytes, device, hipMemcpyHostToDevice);
}

auto d2h_async(void* dst, const void* src, std::size_t bytes, int device) -> void* {
    return do_copy(dst, src, bytes, device, hipMemcpyDeviceToHost);
}

auto event_sync(void* event) -> void {
    if (event == nullptr) return;
    hipEvent_t e = static_cast<hipEvent_t>(event);
    (void)hipEventSynchronize(e);
    release_event(e);
}

auto event_ready(void* event) -> bool {
    if (event == nullptr) return true;
    return hipEventQuery(static_cast<hipEvent_t>(event)) == hipSuccess;
}

// ---------------------------------------------------------------------------
// Generic stream/event primitives (FINDING 60 — DDP/FSDP/ZeRO comm overlap).
// Unlike h2d_async/d2h_async's pooled streams/events, these are owned
// 1:1 by the caller (mirrors cudaStreamCreate/cudaEventCreate lifecycle).
// ---------------------------------------------------------------------------

auto stream_create() -> void* {
    hipStream_t stream = nullptr;
    if (hipError_t err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        err != hipSuccess) {
        throw std::runtime_error(
            std::string("rocm_transfer::stream_create: ") + hipGetErrorString(err));
    }
    return static_cast<void*>(stream);
}

auto stream_destroy(void* stream) -> void {
    if (stream == nullptr) return;
    (void)hipStreamDestroy(static_cast<hipStream_t>(stream));
}

auto event_create() -> void* {
    hipEvent_t event = nullptr;
    if (hipError_t err = hipEventCreateWithFlags(&event, hipEventDisableTiming);
        err != hipSuccess) {
        throw std::runtime_error(
            std::string("rocm_transfer::event_create: ") + hipGetErrorString(err));
    }
    return static_cast<void*>(event);
}

auto event_destroy(void* event) -> void {
    if (event == nullptr) return;
    (void)hipEventDestroy(static_cast<hipEvent_t>(event));
}

auto event_record(void* event, void* stream) -> void {
    if (event == nullptr) return;
    if (hipError_t err = hipEventRecord(static_cast<hipEvent_t>(event),
                                         static_cast<hipStream_t>(stream));
        err != hipSuccess) {
        throw std::runtime_error(
            std::string("rocm_transfer::event_record: ") + hipGetErrorString(err));
    }
}

auto stream_wait_event(void* stream, void* event) -> void {
    if (event == nullptr) return;
    if (hipError_t err = hipStreamWaitEvent(static_cast<hipStream_t>(stream),
                                             static_cast<hipEvent_t>(event), 0);
        err != hipSuccess) {
        throw std::runtime_error(
            std::string("rocm_transfer::stream_wait_event: ") + hipGetErrorString(err));
    }
}

auto stream_synchronize(void* stream) -> void {
    if (stream == nullptr) return;
    if (hipError_t err = hipStreamSynchronize(static_cast<hipStream_t>(stream));
        err != hipSuccess) {
        throw std::runtime_error(
            std::string("rocm_transfer::stream_synchronize: ") + hipGetErrorString(err));
    }
}

auto mem_get_info(int device, std::size_t* free_bytes, std::size_t* total_bytes) -> bool {
    (void)hipSetDevice(device);
    return hipMemGetInfo(free_bytes, total_bytes) == hipSuccess;
}

auto host_register(void* ptr, std::size_t size) -> bool {
    if (ptr == nullptr || size == 0) return false;
    hipError_t err = hipHostRegister(ptr, size, hipHostRegisterDefault);
    if (err == hipSuccess) return true;
    // Clear the error so it doesn't propagate to the next HIP call, mirroring
    // the CUDA cudaHostRegister failure path in storage.cpp.
    (void)hipGetLastError();
    return false;
}

auto host_unregister(void* ptr) -> void {
    if (ptr == nullptr) return;
    (void)hipHostUnregister(ptr);
}

auto host_malloc(std::size_t size) -> void* {
    if (size == 0) return nullptr;
    void* ptr = nullptr;
    if (hipError_t err = hipHostMalloc(&ptr, size, hipHostMallocPortable); err != hipSuccess) {
        return nullptr;
    }
    return ptr;
}

auto host_free(void* ptr) -> void {
    if (ptr == nullptr) return;
    (void)hipHostFree(ptr);
}

}  // namespace rocm_transfer
}  // namespace tenzor
