/**
 * @file vulkan_commands.cpp
 * @brief Vulkan backend command pool, batching, fences, barriers, synchronization
 */

#include "vulkan_helpers.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {

VkCommandBuffer VulkanBackend::beginSingleTimeCommands(int32_t device_id) {
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        // Use batched command buffer for reduced submission overhead
        return getOrCreateBatchCommandBuffer(device_id);
    } else {
        // Legacy path: individual command buffer per operation
        return acquireCommandBuffer(device_id);
    }
}

void VulkanBackend::endSingleTimeCommands(VkCommandBuffer commandBuffer, int32_t device_id) {
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        // Record operation to batch - don't submit yet
        recordOperationToBatch(device_id);
    } else {
        // Legacy path: submit immediately with fence tracking
        endSingleTimeCommandsAsync(commandBuffer, device_id);
    }
}

void VulkanBackend::initCommandBufferPool(DeviceContext& ctx) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.commandBufferCount = static_cast<uint32_t>(DeviceContext::COMMAND_BUFFER_POOL_SIZE);

    ctx.commandBufferPool.resize(DeviceContext::COMMAND_BUFFER_POOL_SIZE);
    vulkan::checkVk(vkAllocateCommandBuffers(ctx.device, &allocInfo, ctx.commandBufferPool.data()),
                   "Failed to allocate command buffer pool");
    ctx.nextCommandBufferIndex = 0;
}

VkCommandBuffer VulkanBackend::acquireCommandBuffer(int32_t device_id) {
    auto& ctx = devices_[device_id];

    // If we've used all buffers in the pool, wait for ALL GPU work and reset
    if (ctx.nextCommandBufferIndex >= ctx.commandBufferPool.size()) {
        // Use vkDeviceWaitIdle instead of fence-only wait to ensure ALL GPU work
        // is complete before resetting descriptor pool. Fence-only wait may miss
        // in-flight commands that reference descriptor sets from this pool.
        vkDeviceWaitIdle(ctx.device);
        vkResetCommandPool(ctx.device, ctx.commandPool, 0);
        ctx.nextCommandBufferIndex = 0;
        ctx.submittedFrames = 0;
        ctx.currentFrame = 0;
        ctx.hasPendingWork = false;

        // Safe to reset descriptor pool now — all GPU work is complete
        if (ctx.descriptorPool) {
            ctx.descriptorPool->reset();
        }
    }

    VkCommandBuffer cmdBuffer = ctx.commandBufferPool[ctx.nextCommandBufferIndex++];

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmdBuffer, &beginInfo);
    return cmdBuffer;
}

void VulkanBackend::releaseCommandBuffer(VkCommandBuffer cmdBuffer, int32_t device_id) {
    // Command buffers are pooled and reset together, no individual release needed
    (void)cmdBuffer;
    (void)device_id;
}

void VulkanBackend::ensurePendingWorkComplete(int32_t device_id) {
    auto& ctx = devices_[device_id];

    // Wait on all frame fences that have been submitted
    if (ctx.submittedFrames > 0) {
        // Collect all fences that might have pending work
        std::vector<VkFence> fencesToWait;
        for (size_t i = 0; i < DeviceContext::MAX_FRAMES_IN_FLIGHT; ++i) {
            if (ctx.frameFences[i] != VK_NULL_HANDLE) {
                // Check if fence is actually signaled (has pending work)
                VkResult status = vkGetFenceStatus(ctx.device, ctx.frameFences[i]);
                if (status == VK_NOT_READY) {
                    fencesToWait.push_back(ctx.frameFences[i]);
                } else if (status == VK_ERROR_DEVICE_LOST) {
                    throw std::runtime_error("Device lost before fence wait (fence status check)");
                }
            }
        }

        if (!fencesToWait.empty()) {
            // Use 30 second timeout to detect GPU hangs (often caused by memory pressure)
            constexpr uint64_t FENCE_TIMEOUT_NS = 30'000'000'000ULL;  // 30 seconds
            VkResult result = vkWaitForFences(ctx.device,
                                              static_cast<uint32_t>(fencesToWait.size()),
                                              fencesToWait.data(), VK_TRUE, FENCE_TIMEOUT_NS);
            if (result == VK_TIMEOUT) {
                throw std::runtime_error("GPU fence wait timed out after 30 seconds. "
                    "This often indicates memory pressure or a shader hang. "
                    "Try reducing batch size or model size, or use a smaller dtype.");
            }
            if (result != VK_SUCCESS) {
                std::string error_msg = "Failed to wait for fences: " + std::to_string(result);
                if (result == VK_ERROR_DEVICE_LOST) {
                    error_msg += " (VK_ERROR_DEVICE_LOST - GPU crash or timeout)";
                }
                throw std::runtime_error(error_msg);
            }
        }

        // Reset state - all work is now complete
        ctx.submittedFrames = 0;
        ctx.currentFrame = 0;
    }

    ctx.hasPendingWork = false;
}

void VulkanBackend::endSingleTimeCommandsAsync(VkCommandBuffer commandBuffer, int32_t device_id) {
    auto& ctx = devices_[device_id];

    VkResult result = vkEndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to end command buffer: " + std::to_string(result));
    }

    // Use ring buffer of fences for true async execution
    // Only wait if all frames in flight are occupied
    size_t targetFrame = ctx.currentFrame;

    // Check if we need to wait for the target frame's fence
    // We only need to wait if we've submitted MAX_FRAMES_IN_FLIGHT work already
    if (ctx.submittedFrames >= DeviceContext::MAX_FRAMES_IN_FLIGHT) {
        // Wait for the oldest frame to complete before reusing its fence
        // Use 30 second timeout to detect GPU hangs
        constexpr uint64_t FENCE_TIMEOUT_NS = 30'000'000'000ULL;  // 30 seconds
        VkFence fenceToWait = ctx.frameFences[targetFrame];
        VkResult waitResult = vkWaitForFences(ctx.device, 1, &fenceToWait, VK_TRUE, FENCE_TIMEOUT_NS);
        if (waitResult == VK_TIMEOUT) {
            throw std::runtime_error("GPU fence wait timed out after 30 seconds. "
                "This often indicates memory pressure or a shader hang.");
        }
        if (waitResult != VK_SUCCESS) {
            throw std::runtime_error("Failed to wait for frame fence: " + std::to_string(waitResult));
        }
        // Frame completed — decrement in-flight count so submittedFrames
        // accurately tracks the number of actually pending submissions
        ctx.submittedFrames--;
    }

    // Get the fence for this submission and reset it
    VkFence fence = ctx.frameFences[targetFrame];
    vkResetFences(ctx.device, 1, &fence);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    result = vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, fence);
    if (result != VK_SUCCESS) {
        std::string error_msg = "Failed to submit queue with fence: " + std::to_string(result);
        if (result == VK_ERROR_DEVICE_LOST) {
            error_msg += " (VK_ERROR_DEVICE_LOST)";
        }
        throw std::runtime_error(error_msg);
    }

    // Move to next frame slot (ring buffer)
    ctx.currentFrame = (ctx.currentFrame + 1) % DeviceContext::MAX_FRAMES_IN_FLIGHT;
    ctx.submittedFrames++;

    // Also mark pendingFence for legacy code paths
    ctx.hasPendingWork = true;
}

auto VulkanBackend::synchronize(int32_t device_id) -> void {
    if (device_id < 0 || device_id >= device_count()) {
        throw std::invalid_argument("Invalid device ID");
    }
    auto& ctx = devices_[device_id];
    std::lock_guard<std::recursive_mutex> lock(ctx.mutex);

    // Submit any pending batched commands before synchronizing
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        submitBatchIfNeeded(device_id, true);  // Force submit any pending work
    }

    // First ensure any fence-tracked work is complete (legacy pendingFence)
    ensurePendingWorkComplete(device_id);

    // Wait for all frame fences in the ring buffer
    if (ctx.submittedFrames > 0) {
        std::vector<VkFence> fencesToWait;
        for (size_t i = 0; i < DeviceContext::MAX_FRAMES_IN_FLIGHT; ++i) {
            if (ctx.frameFences[i] != VK_NULL_HANDLE) {
                fencesToWait.push_back(ctx.frameFences[i]);
            }
        }
        if (!fencesToWait.empty()) {
            constexpr uint64_t FENCE_TIMEOUT_NS = 30'000'000'000ULL;  // 30 seconds
            VkResult result = vkWaitForFences(ctx.device, static_cast<uint32_t>(fencesToWait.size()),
                           fencesToWait.data(), VK_TRUE, FENCE_TIMEOUT_NS);
            if (result == VK_TIMEOUT) {
                throw std::runtime_error("GPU sync fence wait timed out after 30 seconds. "
                    "This often indicates memory pressure or a shader hang.");
            }
        }
        ctx.submittedFrames = 0;
        ctx.currentFrame = 0;
    }

    // Then wait for all device operations (belt and suspenders)
    vkDeviceWaitIdle(ctx.device);

    // Reset command pool and pool index
    vkResetCommandPool(ctx.device, ctx.commandPool, 0);
    ctx.nextCommandBufferIndex = 0;

    // Reset batching state
    ctx.activeCommandBuffer = VK_NULL_HANDLE;
    ctx.operationsInBatch = 0;

    // Reset descriptor pool to reclaim descriptor sets
    // This is safe because all GPU work is complete after vkDeviceWaitIdle
    if (ctx.descriptorPool) {
        ctx.descriptorPool->reset();
    }
}

// ============================================================================
// Batched Command Execution for Performance
// ============================================================================

void VulkanBackend::initFrameFences(DeviceContext& ctx) {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Start signaled so first wait doesn't block

    for (size_t i = 0; i < DeviceContext::MAX_FRAMES_IN_FLIGHT; ++i) {
        vulkan::checkVk(vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.frameFences[i]),
                       "Failed to create frame fence");
    }

    // Pre-allocate command buffers for each frame
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.commandBufferCount = static_cast<uint32_t>(DeviceContext::MAX_FRAMES_IN_FLIGHT);

    vulkan::checkVk(vkAllocateCommandBuffers(ctx.device, &allocInfo, ctx.frameCommandBuffers.data()),
                   "Failed to allocate frame command buffers");
}

VkCommandBuffer VulkanBackend::getOrCreateBatchCommandBuffer(int32_t device_id) {
    auto& ctx = devices_[device_id];

    // If no active batch, start one
    if (ctx.activeCommandBuffer == VK_NULL_HANDLE) {
        // Wait for this frame's fence if it has pending work
        if (ctx.submittedFrames > 0) {
            waitForFrame(device_id, ctx.currentFrame);
        }

        ctx.activeCommandBuffer = ctx.frameCommandBuffers[ctx.currentFrame];

        // Reset and begin the command buffer
        vkResetCommandBuffer(ctx.activeCommandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx.activeCommandBuffer, &beginInfo);

        ctx.operationsInBatch = 0;
    }

    return ctx.activeCommandBuffer;
}

void VulkanBackend::recordOperationToBatch(int32_t device_id) {
    auto& ctx = devices_[device_id];
    ctx.operationsInBatch++;

    // Auto-submit if batch is full (use config threshold)
    if (ctx.operationsInBatch >= vulkan_config::BATCH_SIZE_THRESHOLD) {
        submitBatchIfNeeded(device_id, true);
    }
}

void VulkanBackend::submitBatchIfNeeded(int32_t device_id, bool force) {
    auto& ctx = devices_[device_id];

    if (ctx.activeCommandBuffer == VK_NULL_HANDLE || ctx.operationsInBatch == 0) {
        return;  // Nothing to submit
    }

    if (!force && ctx.operationsInBatch < DeviceContext::MAX_OPERATIONS_PER_BATCH / 2) {
        return;  // Let more operations accumulate
    }

    // End and submit the command buffer
    vkEndCommandBuffer(ctx.activeCommandBuffer);

    VkFence fence = ctx.frameFences[ctx.currentFrame];

    // Reset fence before use
    vkResetFences(ctx.device, 1, &fence);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &ctx.activeCommandBuffer;

    VkResult result = vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, fence);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit batch command buffer: " + std::to_string(result));
    }

    // Move to next frame slot
    ctx.activeCommandBuffer = VK_NULL_HANDLE;
    ctx.operationsInBatch = 0;
    ctx.currentFrame = (ctx.currentFrame + 1) % DeviceContext::MAX_FRAMES_IN_FLIGHT;
    ctx.submittedFrames++;
}

void VulkanBackend::waitForFrame(int32_t device_id, size_t frameIndex) {
    auto& ctx = devices_[device_id];
    VkFence fence = ctx.frameFences[frameIndex];

    constexpr uint64_t FENCE_TIMEOUT_NS = 30'000'000'000ULL;  // 30 seconds
    VkResult result = vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, FENCE_TIMEOUT_NS);
    if (result == VK_TIMEOUT) {
        throw std::runtime_error("GPU frame fence wait timed out after 30 seconds. "
            "This often indicates memory pressure or a shader hang.");
    }
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to wait for frame fence: " + std::to_string(result));
    }
}

} // namespace tenzor
