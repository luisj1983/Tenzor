/**
 * @file vulkan_memory.cpp
 * @brief Vulkan backend memory management: allocate, deallocate, copy, staging buffers
 *
 * Performance optimizations:
 * - Per-device mutexes instead of global lock (9C)
 * - Deferred free list to avoid GPU sync on deallocate (9B)
 * - Staging buffer pool for concurrent H2D/D2H transfers (9D)
 */

#include "vulkan_helpers.hpp"
#include "tenzor/backend/vulkan_caching_allocator.hpp"
#include <cassert>
#include <cstdint>
#include <vector>
#ifdef TENZOR_HAS_VMA
// Emit the Vulkan Memory Allocator implementation in exactly this translation
// unit. VMA is header-only; without VMA_IMPLEMENTATION the vma* symbols
// (vmaCreateAllocator, ...) are undefined and the .so fails to load.
#define VMA_IMPLEMENTATION
#include "tenzor/backend/vulkan_vma_allocator.hpp"
#endif

#include <cstring>
#include <stdexcept>

namespace tenzor {

namespace {

// Staging buffers serve BOTH directions: CPU->GPU writes (HostToDevice) and
// GPU->CPU reads (DeviceToHost, via memcpy out of the mapped pointer in
// VulkanBackend::copy's DeviceToHost case). VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
// | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT alone, with no HOST_CACHED bit, was
// the only property set ever requested here -- that combination is
// satisfied by write-combined (uncached) system memory on most discrete-GPU
// drivers (fast sequential CPU writes, but every CPU *read* misses cache and
// pays a full round trip). That made GPU->CPU staging readback measured at
// ~0.4 GB/s versus ~4-10 GB/s for CUDA/ROCm/OneAPI on the same host doing the
// same 100MB transfer (see FINDING 25 / tests/core/test_transfer_engine.cpp's
// BandwidthMeasurement_GPUToCPU) -- roughly a 10-20x self-inflicted penalty
// for every Vulkan-backed ZeRO-offload readback, not a fundamentally slower
// backend. Prefer an also-HOST_CACHED memory type (fine for writes too --
// the GPU-side DMA read performance is independent of the host's cache
// attributes) and fall back to the coherent-only combination on hardware
// that doesn't expose a cached host-visible type (e.g. some unified-memory
// integrated GPUs, where COHERENT-without-CACHED is often already about as
// fast as it gets).
auto makeStagingBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size)
        -> std::unique_ptr<vulkan::VulkanBuffer> {
    constexpr VkBufferUsageFlags kUsage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    constexpr VkMemoryPropertyFlags kPreferred =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    constexpr VkMemoryPropertyFlags kFallback =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    try {
        return std::make_unique<vulkan::VulkanBuffer>(device, physicalDevice, size, kUsage, kPreferred);
    } catch (const std::exception&) {
        return std::make_unique<vulkan::VulkanBuffer>(device, physicalDevice, size, kUsage, kFallback);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Staging buffer pool implementation (9D)
// ---------------------------------------------------------------------------

size_t VulkanBackend::StagingBufferPool::acquire(
        int32_t /*device_id*/, size_t size, const DeviceContext& ctx) {
    std::lock_guard<std::mutex> lock(*mutex);

    uint64_t current_tick = ++tick_counter;

    // Try to find an existing buffer that is not in use and large enough
    for (size_t i = 0; i < buffers.size(); ++i) {
        if (!buffers[i].in_use && buffers[i].buffer && buffers[i].size >= size) {
            buffers[i].in_use = true;
            buffers[i].last_use_tick = current_tick;
            return i;
        }
    }

    // Try to find a slot that is not in use but needs resizing (or is empty)
    for (size_t i = 0; i < buffers.size(); ++i) {
        if (!buffers[i].in_use) {
            buffers[i].buffer = makeStagingBuffer(ctx.device, ctx.physicalDevice, size);
            buffers[i].size = size;
            buffers[i].in_use = true;
            buffers[i].last_use_tick = current_tick;
            return i;
        }
    }

    // Evict oldest unused buffers if pool exceeds max size
    if (buffers.size() >= kMaxPoolSize) {
        size_t oldest_idx = SIZE_MAX;
        uint64_t oldest_tick = UINT64_MAX;
        for (size_t i = 0; i < buffers.size(); ++i) {
            if (!buffers[i].in_use && buffers[i].last_use_tick < oldest_tick) {
                oldest_tick = buffers[i].last_use_tick;
                oldest_idx = i;
            }
        }
        if (oldest_idx != SIZE_MAX) {
            buffers[oldest_idx].buffer = makeStagingBuffer(ctx.device, ctx.physicalDevice, size);
            buffers[oldest_idx].size = size;
            buffers[oldest_idx].in_use = true;
            buffers[oldest_idx].last_use_tick = current_tick;
            return oldest_idx;
        }
    }

    // All slots are in use -- create a new one
    size_t idx = buffers.size();
    auto& sb = buffers.emplace_back();
    sb.buffer = makeStagingBuffer(ctx.device, ctx.physicalDevice, size);
    sb.size = size;
    sb.in_use = true;
    sb.last_use_tick = current_tick;
    return idx;
}

void VulkanBackend::StagingBufferPool::release(size_t index) {
    std::lock_guard<std::mutex> lock(*mutex);
    if (index < buffers.size()) {
        buffers[index].in_use = false;
    }
}

// ---------------------------------------------------------------------------
// Allocate / Deallocate (9B deferred frees, 9C per-device mutexes)
// ---------------------------------------------------------------------------

auto VulkanBackend::allocate(size_t bytes, int32_t device_id) -> void* {
    // Allocate minimum 4 bytes for all requests, and round every allocation
    // up to a multiple of 4. This ensures:
    // 1. Empty tensors always have valid, tracked Vulkan buffers (no null data_ptr crashes)
    // 2. Float16/BFloat16 tensors with odd element counts have enough space for
    //    uint32 shader access (half-precision shaders pack 2 elements per uint32,
    //    so a 9-element Float16 tensor is 18 bytes logically but needs 20 bytes
    //    of storage to let the shader read/write the trailing uint32 word safely —
    //    without rounding the CAS-based packed writes corrupt the last element.)
    size_t alloc_bytes = std::max(bytes, static_cast<size_t>(4));
    alloc_bytes = (alloc_bytes + 3) & ~size_t(3);

    if (device_id < 0 || device_id >= device_count()) {
        throw std::invalid_argument("Invalid device ID");
    }

    // Lock only the target device for the actual Vulkan allocation
    std::lock_guard<std::recursive_mutex> dev_lock(devices_[device_id].mutex);
    void* ptr = nullptr;
    try {
        ptr = allocateDeviceMemory(alloc_bytes, device_id);
    } catch (const std::exception&) {
        // OOM: freed tensors sit in the per-device deferred-free list (they are
        // not returned to the allocator until a synchronize(), to avoid a GPU
        // sync on every deallocate). A long compute-only loop never syncs, so
        // the deferred list grows until the allocator exhausts device memory
        // even though most of it is reclaimable. Drain the GPU + flush the
        // deferred frees back into the allocator cache, then retry once.
        // (Mirrors the CUDA caching allocator's free-cache-and-retry on OOM.)
        synchronize(device_id);
        ptr = allocateDeviceMemory(alloc_bytes, device_id);
    }

    // Lock the allocations map briefly to insert the tracking entry
    {
        std::lock_guard<std::mutex> alloc_lock(allocations_mutex_);
        allocations_[ptr] = {alloc_bytes, device_id};
    }
    return ptr;
}

void* VulkanBackend::allocateDeviceMemory(size_t bytes, int32_t device_id) {
#ifdef TENZOR_HAS_VMA
    // When VMA is available, use it for allocation (better suballocation + defrag)
    auto& vma_alloc = backend::VulkanVMAAllocator::get();
    void* ptr = vma_alloc.allocate(
        bytes, device_id,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    return ptr;
#else
    // Use custom caching allocator for efficient memory reuse
    auto& allocator = backend::VulkanCachingAllocator::get();

    // Allocate device-local memory for compute buffers.
    // Data transfers use staging buffers (HOST_VISIBLE), so compute tensors
    // only need DEVICE_LOCAL. This avoids per-allocation mapping overhead and
    // gives the driver maximum flexibility for memory placement.
    void* ptr = allocator.allocate(
        bytes, device_id,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    return ptr;
#endif
}

auto VulkanBackend::deallocate(void* ptr) -> void {
    if (ptr == nullptr) {
        return;
    }

    // Look up the allocation under the allocations lock, then remove it.
    size_t bytes = 0;
    int32_t device_id = -1;
    {
        std::lock_guard<std::mutex> alloc_lock(allocations_mutex_);
        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            return;  // Unknown pointer -- nothing to do
        }
        bytes = it->second.first;
        device_id = it->second.second;
        allocations_.erase(it);
    }

    // Instead of force-syncing the GPU (which drains the pipeline), defer the
    // actual free until the next synchronize() call for this device.
    std::lock_guard<std::recursive_mutex> dev_lock(devices_[device_id].mutex);

    // If batching is enabled, force-submit the current batch so the buffer is
    // no longer referenced by an un-submitted command buffer. This is cheap
    // compared to a full device wait.
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        submitBatchIfNeeded(device_id, true);
    }

    // Add to deferred free list -- actual vkFreeMemory happens at synchronize()
    deferred_frees_[device_id].push_back({ptr, bytes, device_id});
}

void VulkanBackend::freeDeviceMemory(void* ptr, int32_t device_id) {
#ifdef TENZOR_HAS_VMA
    backend::VulkanVMAAllocator::get().free(ptr, device_id);
#else
    // Return memory to caching allocator for reuse
    backend::VulkanCachingAllocator::get().free(ptr, device_id);
#endif
}

void VulkanBackend::flush_deferred_frees(int32_t device_id) {
    // Called from synchronize() after all GPU work is guaranteed complete.
    // The caller already holds devices_[device_id].mutex.
    auto& frees = deferred_frees_[device_id];
    for (auto& entry : frees) {
        freeDeviceMemory(entry.ptr, entry.device_id);
    }
    frees.clear();
}

// ---------------------------------------------------------------------------
// Copy with staging buffer pool (9D)
// ---------------------------------------------------------------------------

auto VulkanBackend::copy(void* dst, const void* src, size_t bytes,
                        CopyKind kind) -> void {
    if (bytes == 0) {
        return;
    }

    // Resolve the owning Vulkan device of a device pointer. A pointer may be the
    // exact allocation base or a view into the middle of one (sliced / in-place
    // target), so fall back to a range scan. Throws (never silently uses device
    // 0) so a multi-GPU host can't submit on the wrong device's queue.
    auto resolve_device = [this](const void* device_ptr) -> int32_t {
        std::lock_guard<std::mutex> alloc_lock(allocations_mutex_);
        auto it = allocations_.find(const_cast<void*>(device_ptr));
        if (it != allocations_.end()) {
            return it->second.second;
        }
        const auto* view_ptr = static_cast<const char*>(device_ptr);
        for (const auto& [alloc_base, info] : allocations_) {
            const auto* base = static_cast<const char*>(alloc_base);
            if (view_ptr >= base && view_ptr < base + info.first) {
                return info.second;
            }
        }
        throw std::runtime_error(
            "VulkanBackend::copy: device pointer not found in "
            "allocations; cannot determine the owning device");
    };

    // Cross-device DeviceToDevice must route through a host staging bounce: a
    // single command buffer / vkCmdCopyBuffer may only reference buffers that
    // live on the SAME VkDevice (a buffer can only be used on its owning
    // device), so binding src (device A) and dst (device B) into one copy is
    // undefined behaviour. Detect it BEFORE taking any per-device lock so the
    // recursive D2H/H2D calls each lock only their own device (no two-device
    // lock held at once → no cross-device lock-ordering deadlock).
    if (kind == CopyKind::DeviceToDevice) {
        const int32_t src_device = resolve_device(src);
        const int32_t dst_device = resolve_device(dst);
        if (src_device != dst_device) {
            std::vector<uint8_t> host_staging(bytes);
            copy(host_staging.data(), src, bytes, CopyKind::DeviceToHost);
            copy(dst, host_staging.data(), bytes, CopyKind::HostToDevice);
            return;
        }
    }

    // Determine device ID for the single-device path.
    // For HostToDevice, dst is on device; for DeviceToHost, src is on device;
    // for (same-device) DeviceToDevice, either works.
    const void* device_ptr = (kind == CopyKind::DeviceToHost) ? src : dst;
    int32_t device_id = resolve_device(device_ptr);

    // Lock per-device mutex for GPU command submission
    std::lock_guard<std::recursive_mutex> dev_lock(devices_[device_id].mutex);

    switch (kind) {
        case CopyKind::HostToDevice: {
            size_t staging_idx = acquireStagingBuffer(device_id, bytes);
            // Release the staging slot on EVERY exit path, including an exception
            // thrown by the submit/map/wait calls below. Without this, a throw
            // between acquire and release left the slot permanently in_use.
            struct StagingGuard {
                VulkanBackend* self; int32_t dev; size_t idx;
                ~StagingGuard() { self->releaseStagingBuffer(dev, idx); }
            } staging_guard{this, device_id, staging_idx};
            auto& pool = stagingPools_[device_id];
            auto& staging = pool.buffers[staging_idx];

            void* mapped = staging.buffer->map();
            std::memcpy(mapped, src, bytes);
            staging.buffer->unmap();

            // Copy from staging to device
            auto [dst_buffer, dst_offset] = getVulkanBufferAndOffset(dst);
            VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;  // Staging buffer starts at 0
            copyRegion.dstOffset = dst_offset;
            copyRegion.size = bytes;
            vkCmdCopyBuffer(cmdBuffer, staging.buffer->buffer(),
                          dst_buffer, 1, &copyRegion);
            // Insert barrier for subsequent compute ops that may read this buffer
            insertTransferToComputeBarrier(cmdBuffer);
            endSingleTimeCommands(cmdBuffer, device_id);

            // Force submit now to ensure staging buffer content is copied to the
            // device before the staging buffer can be reused. The wait must be
            // unconditional: when batching is disabled, endSingleTimeCommands has
            // already submitted asynchronously, and the StagingGuard destructor
            // would otherwise free the slot while the GPU is still reading it,
            // racing a subsequent HostToDevice transfer that re-acquires the slot.
            if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
                submitBatchIfNeeded(device_id, true);  // Force submit
            }
            ensurePendingWorkComplete(device_id);   // Wait for copy to complete

            // staging slot released by StagingGuard on scope exit.
            break;
        }
        case CopyKind::DeviceToHost: {
            // Flush any batched commands before reading back data
            if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
                submitBatchIfNeeded(device_id, true);  // Force submit
            }
            // Ensure any pending GPU compute work is complete before copying
            ensurePendingWorkComplete(device_id);

            size_t staging_idx = acquireStagingBuffer(device_id, bytes);
            // Release the staging slot on EVERY exit path, including an exception
            // thrown by the submit/map/wait calls below. Without this, a throw
            // between acquire and release left the slot permanently in_use.
            struct StagingGuard {
                VulkanBackend* self; int32_t dev; size_t idx;
                ~StagingGuard() { self->releaseStagingBuffer(dev, idx); }
            } staging_guard{this, device_id, staging_idx};
            auto& pool = stagingPools_[device_id];
            auto& staging = pool.buffers[staging_idx];

            // Flush any pending (batched) compute work first: the source buffer
            // may have just been written by a compute shader still queued in the
            // batch command buffer. Without this, the immediate transfer below
            // could read src before that compute completes (no compute->transfer
            // dependency existed across the separate command buffers).
            ensurePendingWorkComplete(device_id);

            // Copy from device to staging - MUST use immediate execution, not batching
            // because we need the data available right after this call
            auto [src_buffer, src_offset] = getVulkanBufferAndOffset(src);
            VkCommandBuffer cmdBuffer = acquireCommandBuffer(device_id);  // Bypass batching
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = src_offset;
            copyRegion.dstOffset = 0;  // Staging buffer starts at 0
            copyRegion.size = bytes;
            vkCmdCopyBuffer(cmdBuffer, src_buffer,
                          staging.buffer->buffer(), 1, &copyRegion);
            endSingleTimeCommandsAsync(cmdBuffer, device_id);  // Submit immediately

            // Ensure copy is complete before reading from staging buffer
            ensurePendingWorkComplete(device_id);

            void* mapped = staging.buffer->map();
            std::memcpy(dst, mapped, bytes);
            staging.buffer->unmap();

            // staging slot released by StagingGuard on scope exit.
            break;
        }
        case CopyKind::DeviceToDevice: {
            // Same-device D2D only — the cross-device case was handled above via
            // a host staging bounce before any per-device lock was taken.
            auto [src_buffer, src_offset] = getVulkanBufferAndOffset(src);
            auto [dst_buffer, dst_offset] = getVulkanBufferAndOffset(dst);
            VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = src_offset;
            copyRegion.dstOffset = dst_offset;
            copyRegion.size = bytes;
            vkCmdCopyBuffer(cmdBuffer, src_buffer,
                          dst_buffer, 1, &copyRegion);
            // Insert barrier for subsequent compute ops that may read this buffer
            insertTransferToComputeBarrier(cmdBuffer);
            endSingleTimeCommands(cmdBuffer, device_id);
            break;
        }
        case CopyKind::HostToHost: {
            std::memcpy(dst, src, bytes);
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// memset via vkCmdFillBuffer
// ---------------------------------------------------------------------------

auto VulkanBackend::memset(void* ptr, int value, size_t bytes, int32_t device_id) -> void {
    if (bytes == 0) return;

    std::lock_guard<std::recursive_mutex> dev_lock(devices_[device_id].mutex);

    auto [buffer, offset] = getVulkanBufferAndOffset(ptr);

    // vkCmdFillBuffer fills with a 32-bit value, size must be multiple of 4
    // Extend the fill pattern across all 4 bytes
    uint32_t fill_value = 0;
    if (value != 0) {
        uint8_t byte_val = static_cast<uint8_t>(value);
        fill_value = (static_cast<uint32_t>(byte_val) << 24) |
                     (static_cast<uint32_t>(byte_val) << 16) |
                     (static_cast<uint32_t>(byte_val) << 8)  |
                     static_cast<uint32_t>(byte_val);
    }

    // vkCmdFillBuffer requires offset to be a multiple of 4
    if (offset % 4 != 0) {
        throw std::invalid_argument("vkCmdFillBuffer: offset (" + std::to_string(offset) +
                                    ") is not 4-byte aligned");
    }

    // Round size up to 4-byte alignment for vkCmdFillBuffer
    size_t aligned_bytes = (bytes + 3) & ~size_t(3);

    // Verify aligned fill does not exceed the owning allocation.
    //
    // allocations_ is keyed by *base* allocation pointer, so allocations_.find(ptr)
    // misses for a view pointer (base + offset, e.g. a sliced tensor). Resolve the
    // owning allocation by locating the tracked range [base, base + size) that
    // contains ptr, then clamp against (alloc_size - view_offset) where view_offset
    // is measured from that base. This guards the round-up overrun for views as
    // well as base pointers — vkCmdFillBuffer with aligned_bytes = (bytes+3)&~3
    // could otherwise write up to 3 bytes past the underlying allocation when the
    // view ends at the allocation boundary.
    {
        std::lock_guard<std::mutex> alloc_lock(allocations_mutex_);
        const auto* view_ptr = static_cast<const char*>(ptr);
        size_t avail = 0;
        bool found = false;
        for (const auto& [alloc_base, info] : allocations_) {
            const auto* base = static_cast<const char*>(alloc_base);
            size_t alloc_size = info.first;
            if (view_ptr >= base && view_ptr < base + alloc_size) {
                size_t view_offset = static_cast<size_t>(view_ptr - base);
                avail = alloc_size - view_offset;
                found = true;
                break;
            }
        }
        if (found && aligned_bytes > avail) {
            // Clamp to the owning allocation boundary — the caller only needs
            // 'bytes' filled, but rounding up could overrun the buffer.
            //
            // Correctness invariant: the allocator rounds every allocation up to
            // a multiple of 4 (allocate()), and views are sub-ranges of such
            // allocations, so `avail` is always a multiple of 4 and the
            // `avail & ~3` re-align below is a no-op that never drops requested
            // fill bytes. Assert it so a future sub-4B-padded allocation path
            // cannot silently start truncating the tail of a memset.
            assert((avail % 4 == 0) &&
                   "Vulkan memset: owning allocation size must be 4-byte aligned; "
                   "otherwise the clamp would drop the last 1-3 fill bytes");
            aligned_bytes = avail;
            // Re-align down to 4-byte boundary (vkCmdFillBuffer requirement)
            aligned_bytes &= ~size_t(3);
            if (aligned_bytes == 0) {
                return;  // Nothing left to fill within alignment constraints
            }
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdFillBuffer(cmdBuffer, buffer, offset, aligned_bytes, fill_value);
    insertTransferToComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);
}

// ---------------------------------------------------------------------------
// Staging buffer pool helpers
// ---------------------------------------------------------------------------

size_t VulkanBackend::acquireStagingBuffer(int32_t device_id, size_t size) {
    return stagingPools_[device_id].acquire(device_id, size, devices_[device_id]);
}

void VulkanBackend::releaseStagingBuffer(int32_t device_id, size_t index) {
    stagingPools_[device_id].release(index);
}

} // namespace tenzor
