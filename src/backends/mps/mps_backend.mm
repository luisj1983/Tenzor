/**
 * @file mps_backend.mm
 * @brief Apple Metal/MPS backend implementation (Objective-C++)
 *
 * Uses Metal framework for GPU compute on Apple Silicon.
 * MTLResourceStorageModeShared for unified memory (zero-copy CPU/GPU access).
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>

#include "mps_backend.hpp"
#include "tenzor/core/tensor.hpp"
#include <unordered_map>
#include <mutex>
#include <stdexcept>

namespace tenzor {
namespace mps {

struct MPSBackend::Impl {
    id<MTLDevice> device{nil};
    id<MTLCommandQueue> command_queue{nil};

    // Buffer tracking: void* -> MTLBuffer
    std::unordered_map<void*, id<MTLBuffer>> buffer_map;
    std::mutex buffer_mutex;

    // Thread-local device index
    static thread_local int current_device_id;
};

thread_local int MPSBackend::Impl::current_device_id = 0;

// Active backend Impl for cross-TU buffer reuse. There is a single MPS backend
// instance; the element-wise kernels (a separate translation unit) use this to
// fetch the allocator's MTLBuffer for a tensor instead of wrapping a throwaway
// newBufferWithBytesNoCopy buffer per operand per op.
static MPSBackend::Impl* g_active_mps_impl = nullptr;

// Returns the allocator's MTLBuffer whose contents pointer is exactly `ptr`
// (i.e. a tensor that owns its allocation — contiguous tensors), or nil. Views
// (ptr offset into an allocation) miss and the caller falls back to a no-copy
// wrapper. Buffers are released from buffer_map on deallocate(), so a returned
// buffer is always live for an in-use tensor.
id<MTLBuffer> pooled_buffer_for(void* ptr) {
    if (!g_active_mps_impl || !ptr) return nil;
    std::lock_guard<std::mutex> lock(g_active_mps_impl->buffer_mutex);
    auto it = g_active_mps_impl->buffer_map.find(ptr);
    return (it != g_active_mps_impl->buffer_map.end()) ? it->second : nil;
}

MPSBackend::MPSBackend() : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            throw std::runtime_error("MPS: No Metal device available");
        }
        impl_->command_queue = [impl_->device newCommandQueue];
        if (!impl_->command_queue) {
            throw std::runtime_error("MPS: Failed to create command queue");
        }
    }
    // Publish this instance's allocator map for cross-TU buffer reuse.
    g_active_mps_impl = impl_.get();
}

MPSBackend::~MPSBackend() {
    @autoreleasepool {
        // Each kernel commits and waits on its own command buffer, so there is
        // no pending batched work to flush here.
        // Release all tracked buffers
        impl_->buffer_map.clear();
    }
    if (g_active_mps_impl == impl_.get()) {
        g_active_mps_impl = nullptr;
    }
}

auto MPSBackend::device_count() const -> int32_t {
    return 1; // Apple Silicon has one GPU
}

auto MPSBackend::is_available() const -> bool {
    return impl_->device != nil;
}

auto MPSBackend::get_device_info([[maybe_unused]] int32_t device_id) const -> DeviceInfo {
    DeviceInfo info;
    @autoreleasepool {
        info.name = [[impl_->device name] UTF8String];
        info.vendor = "Apple";
        info.total_memory = [impl_->device recommendedMaxWorkingSetSize];
        info.available_memory = info.total_memory - [impl_->device currentAllocatedSize];
        info.compute_units = 0; // Not exposed by Metal API
        info.max_threads_per_block = static_cast<int>([impl_->device maxThreadsPerThreadgroup].width);
        info.warp_size = 32; // SIMD width on Apple Silicon
        info.supports_fp16 = true;
        info.supports_fp64 = false; // Metal does not support FP64
        info.supports_int8 = true;
        info.is_integrated = true; // Apple Silicon is always integrated
        info.is_discrete = false;
    }
    return info;
}

auto MPSBackend::allocate(size_t size_bytes, [[maybe_unused]] int32_t device_id) -> void* {
    @autoreleasepool {
        if (size_bytes == 0) size_bytes = 1; // Metal requires non-zero
        id<MTLBuffer> buffer = [impl_->device
            newBufferWithLength:size_bytes
            options:MTLResourceStorageModeShared];
        if (!buffer) {
            throw std::runtime_error("MPS: Failed to allocate " +
                                     std::to_string(size_bytes) + " bytes");
        }
        void* ptr = [buffer contents];
        std::lock_guard lock(impl_->buffer_mutex);
        impl_->buffer_map[ptr] = buffer;
        return ptr;
    }
}

auto MPSBackend::deallocate(void* ptr) -> void {
    if (!ptr) return;
    std::lock_guard lock(impl_->buffer_mutex);
    impl_->buffer_map.erase(ptr);
    // ARC releases the MTLBuffer when erased from the map
}

auto MPSBackend::copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void {
    // On Apple Silicon with StorageModeShared, all memory is unified.
    // CPU and GPU share the same address space, so memcpy works for all cases.
    std::memcpy(dst, src, bytes);
}

auto MPSBackend::memset(void* ptr, int value, size_t bytes,
                        [[maybe_unused]] int32_t device_id) -> void {
    std::memset(ptr, value, bytes);
}

auto MPSBackend::synchronize([[maybe_unused]] int32_t device_id) -> void {
    // Each kernel creates, commits, and waits on its own command buffer
    // (per-op synchronous execution), so all submitted work is already
    // complete by the time control returns to the caller. Nothing to flush.
}

auto MPSBackend::set_device(int32_t device_id) -> void {
    Impl::current_device_id = device_id;
}

auto MPSBackend::get_current_device() const -> int32_t {
    return Impl::current_device_id;
}

// MPS has a single implicit command queue; there is one logical stream. The
// stream API is satisfied with a single-stream model: create returns the
// default (null) handle, destroy is a no-op, and synchronizing a stream flushes
// the pending command buffer (same as device synchronize).
auto MPSBackend::create_stream([[maybe_unused]] int32_t device_id) -> StreamHandle {
    return nullptr;
}

auto MPSBackend::destroy_stream([[maybe_unused]] StreamHandle stream) -> void {
    // No-op: the single implicit command queue is owned by the backend.
}

auto MPSBackend::synchronize_stream([[maybe_unused]] StreamHandle stream) -> void {
    synchronize(0);
}

auto MPSBackend::register_kernels(BackendDispatchTable& table) -> void {
    register_mps_kernels(table);
}

} // namespace mps
} // namespace tenzor

// Export function for dynamic loading
extern "C" {
    auto create_backend() -> std::unique_ptr<tenzor::Backend> {
        return std::make_unique<tenzor::mps::MPSBackend>();
    }
}
