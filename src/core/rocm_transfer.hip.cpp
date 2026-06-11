// Real HIP-backed implementation of the opaque ROCm transfer API declared in
// rocm_transfer.hpp. Compiled by hipcc (LANGUAGE HIP), so it is the ONLY core
// TU that includes <hip/hip_runtime.h> — keeping HIP's vector-type definitions
// out of the host TUs that also include CUDA headers (see the header comment).
#include "tenzor/core/rocm_transfer.hpp"

#include <hip/hip_runtime.h>

#include <atomic>
#include <mutex>
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
    if (hipMemcpyAsync(dst, src, bytes, kind, stream) != hipSuccess) {
        return nullptr;
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

}  // namespace rocm_transfer
}  // namespace tenzor
