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
#include "mps_cmd_check.h"

namespace tenzor {
namespace mps {

struct MPSBackend::Impl {
    id<MTLDevice> device{nil};
    id<MTLCommandQueue> command_queue{nil};
    id<MTLCommandBuffer> current_command_buffer{nil};
    size_t operations_in_batch{0};

    // Buffer tracking: void* -> MTLBuffer
    std::unordered_map<void*, id<MTLBuffer>> buffer_map;
    std::mutex buffer_mutex;

    // Thread-local device index
    static thread_local int current_device_id;
};

thread_local int MPSBackend::Impl::current_device_id = 0;

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
}

MPSBackend::~MPSBackend() {
    @autoreleasepool {
        // Synchronize any pending work
        if (impl_->current_command_buffer) {
            [impl_->current_command_buffer commit];
            [impl_->current_command_buffer waitUntilCompleted];
            ::tenzor::mps::mps_cmd_check(impl_->current_command_buffer, __func__);
        }
        // Release all tracked buffers
        impl_->buffer_map.clear();
    }
}

auto MPSBackend::device_count() const -> int {
    return 1; // Apple Silicon has one GPU
}

auto MPSBackend::is_available() const -> bool {
    return impl_->device != nil;
}

auto MPSBackend::get_device_info(int device_id) const -> DeviceInfo {
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

auto MPSBackend::allocate(size_t size_bytes, int device_id) -> void* {
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

auto MPSBackend::memset(void* ptr, int value, size_t bytes) -> void {
    std::memset(ptr, value, bytes);
}

auto MPSBackend::synchronize(int device_id) -> void {
    @autoreleasepool {
        if (impl_->current_command_buffer) {
            [impl_->current_command_buffer commit];
            [impl_->current_command_buffer waitUntilCompleted];
            ::tenzor::mps::mps_cmd_check(impl_->current_command_buffer, __func__);
            impl_->current_command_buffer = nil;
            impl_->operations_in_batch = 0;
        }
    }
}

auto MPSBackend::set_device(int device_id) -> void {
    Impl::current_device_id = device_id;
}

auto MPSBackend::get_current_device() const -> int {
    return Impl::current_device_id;
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
