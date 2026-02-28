/**
 * @file vulkan_memory.cpp
 * @brief Vulkan backend memory management: allocate, deallocate, copy, staging buffers
 */

#include "vulkan_helpers.hpp"
#include "tenzor/backend/vulkan_caching_allocator.hpp"

#include <cstring>
#include <stdexcept>

namespace tenzor {

auto VulkanBackend::allocate(size_t bytes, int32_t device_id) -> void* {
    // Use global mutex here because allocations_ map is shared across devices.
    std::lock_guard<std::recursive_mutex> lock(dispatch_mutex_);
    // Allocate minimum 4 bytes for all requests. This ensures:
    // 1. Empty tensors always have valid, tracked Vulkan buffers (no null data_ptr crashes)
    // 2. Float16 tensors with odd element counts have enough space for uint32 shader access
    //    (Float16 shaders pack 2 elements per uint32, so a single-element Float16 tensor
    //    needs 4 bytes, not just 2, to avoid out-of-bounds shader writes)
    size_t alloc_bytes = std::max(bytes, static_cast<size_t>(4));

    if (device_id < 0 || device_id >= device_count()) {
        throw std::invalid_argument("Invalid device ID");
    }

    void* ptr = allocateDeviceMemory(alloc_bytes, device_id);
    allocations_[ptr] = {alloc_bytes, device_id};
    return ptr;
}

void* VulkanBackend::allocateDeviceMemory(size_t bytes, int32_t device_id) {
    // Use caching allocator for efficient memory reuse
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
}

auto VulkanBackend::deallocate(void* ptr) -> void {
    std::lock_guard<std::recursive_mutex> lock(dispatch_mutex_);
    if (ptr == nullptr) {
        return;
    }

    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        auto [bytes, device_id] = it->second;
        // CRITICAL: When batching is enabled, the buffer may be referenced by
        // a command buffer that hasn't been submitted yet. We must force-submit
        // the batch and wait for it to complete before freeing the buffer.
        if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
            submitBatchIfNeeded(device_id, true);  // Force submit any pending batch
        }
        // Ensure any pending async GPU work completes before freeing memory
        ensurePendingWorkComplete(device_id);
        freeDeviceMemory(ptr, device_id);
        allocations_.erase(it);
    }
}

void VulkanBackend::freeDeviceMemory(void* ptr, int32_t device_id) {
    // Return memory to caching allocator for reuse
    backend::VulkanCachingAllocator::get().free(ptr, device_id);
}

auto VulkanBackend::copy(void* dst, const void* src, size_t bytes,
                        CopyKind kind) -> void {
    std::lock_guard<std::recursive_mutex> lock(dispatch_mutex_);
    if (bytes == 0) {
        return;
    }

    // Determine device ID from allocations
    int32_t device_id = 0;

    switch (kind) {
        case CopyKind::HostToDevice: {
            auto& staging = getStagingBuffer(device_id, bytes);
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

            // CRITICAL: With batching enabled, force submit now to ensure staging buffer
            // content is copied to device before staging buffer can be reused.
            // Without this, a subsequent HostToDevice copy could overwrite staging buffer
            // before our copy command is actually submitted.
            if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
                submitBatchIfNeeded(device_id, true);  // Force submit
                ensurePendingWorkComplete(device_id);   // Wait for copy to complete
            }
            break;
        }
        case CopyKind::DeviceToHost: {
            // Flush any batched commands before reading back data
            if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
                submitBatchIfNeeded(device_id, true);  // Force submit
            }
            // Ensure any pending GPU compute work is complete before copying
            ensurePendingWorkComplete(device_id);

            auto& staging = getStagingBuffer(device_id, bytes);

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
            break;
        }
        case CopyKind::DeviceToDevice: {
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

VulkanBackend::StagingBuffer& VulkanBackend::getStagingBuffer(int32_t device_id, size_t size) {
    auto& staging = stagingBuffers_[device_id];

    if (!staging.buffer || staging.size < size) {
        auto& ctx = devices_[device_id];
        staging.buffer = std::make_unique<vulkan::VulkanBuffer>(
            ctx.device, ctx.physicalDevice, size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        staging.size = size;
    }

    return staging;
}

} // namespace tenzor
