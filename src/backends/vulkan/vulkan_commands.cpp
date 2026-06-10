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
    auto& ctx = devices_[device_id];
    // Lock the device mutex for the entire command recording sequence.
    // This prevents two threads from interleaving vkCmd* calls on the
    // same command buffer (undefined behavior in Vulkan).
    // The recursive_mutex allows re-entrant calls from the same thread
    // (e.g., dispatchContiguous called within dispatchMatmul).
    // Unlocked in endSingleTimeCommands.
    ctx.mutex.lock();

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

    // Unlock the device mutex acquired in beginSingleTimeCommands.
    auto& ctx = devices_[device_id];
    ctx.mutex.unlock();
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

    // If we've used all buffers in the pool, wait for pending GPU work and reset
    if (ctx.nextCommandBufferIndex >= ctx.commandBufferPool.size()) {
        // If a batch command buffer is active, submit it first so the
        // pending work is tracked by a fence before we wait.
        if (ctx.activeCommandBuffer != VK_NULL_HANDLE) {
            submitBatchIfNeeded(device_id, true);
        }

        // Wait for all submitted frame fences instead of vkDeviceWaitIdle.
        // This is more targeted — only waits for our own submissions, not
        // all device work (which would serialize unrelated queues).
        ensurePendingWorkComplete(device_id);
        vulkan::checkVk(vkResetCommandPool(ctx.device, ctx.commandPool, 0),
                        "Failed to reset command pool");
        ctx.nextCommandBufferIndex = 0;
        ctx.submittedFrames = 0;
        ctx.currentFrame = 0;
        ctx.hasPendingWork = false;

        // Invalidate batch state — command pool reset makes all buffers invalid
        ctx.activeCommandBuffer = VK_NULL_HANDLE;
        ctx.operationsInBatch = 0;

        // Safe to reset descriptor pool now — all submitted work is complete
        if (ctx.descriptorPool) {
            ctx.descriptorPool->reset();
        }
    }

    VkCommandBuffer cmdBuffer = ctx.commandBufferPool[ctx.nextCommandBufferIndex++];

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vulkan::checkVk(vkBeginCommandBuffer(cmdBuffer, &beginInfo),
                    "Failed to begin command buffer");
    return cmdBuffer;
}

void VulkanBackend::releaseCommandBuffer(VkCommandBuffer cmdBuffer, int32_t device_id) {
    // Command buffers are pooled and reset together, no individual release needed
    (void)cmdBuffer;
    (void)device_id;
}

void VulkanBackend::ensurePendingWorkComplete(int32_t device_id) {
    auto& ctx = devices_[device_id];

    if (ctx.device_lost) {
        // Attempt one-time recovery before giving up
        if (!try_reset_device(device_id)) {
            throw std::runtime_error("Vulkan device lost — cannot wait for work on a lost device");
        }
        // Recovery succeeded — no pending work remains
        return;
    }

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
                    ctx.device_lost = true;
                    throw std::runtime_error("Vulkan device lost (fence status check)");
                } else if (status != VK_SUCCESS) {
                    throw std::runtime_error("Vulkan fence status check failed with VkResult " +
                                             std::to_string(static_cast<int>(status)));
                }
            }
        }

        if (!fencesToWait.empty()) {
            VkResult result = vkWaitForFences(ctx.device,
                                              static_cast<uint32_t>(fencesToWait.size()),
                                              fencesToWait.data(), VK_TRUE, ctx.fence_timeout_ns);
            if (result == VK_TIMEOUT) {
                throw std::runtime_error("GPU fence wait timed out after " +
                    std::to_string(ctx.fence_timeout_ns / 1'000'000'000ULL) + " seconds. "
                    "This often indicates memory pressure or a shader hang. "
                    "Try reducing batch size or model size, or set TENZOR_VULKAN_FENCE_TIMEOUT_S.");
            }
            if (result == VK_ERROR_DEVICE_LOST) {
                ctx.device_lost = true;
                throw std::runtime_error("Vulkan device lost (fence wait) — GPU crash or timeout");
            }
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Failed to wait for fences: " + std::to_string(result));
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

    if (ctx.device_lost) {
        throw std::runtime_error("Vulkan device lost — cannot submit work to a lost device");
    }

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
        VkFence fenceToWait = ctx.frameFences[targetFrame];
        VkResult waitResult = vkWaitForFences(ctx.device, 1, &fenceToWait, VK_TRUE, ctx.fence_timeout_ns);
        if (waitResult == VK_TIMEOUT) {
            throw std::runtime_error("GPU fence wait timed out after " +
                std::to_string(ctx.fence_timeout_ns / 1'000'000'000ULL) + " seconds. "
                "This often indicates memory pressure or a shader hang.");
        }
        if (waitResult == VK_ERROR_DEVICE_LOST) {
            ctx.device_lost = true;
            throw std::runtime_error("Vulkan device lost during frame fence wait");
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
    VkResult resetResult = vkResetFences(ctx.device, 1, &fence);
    if (resetResult != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset frame fence: " + std::to_string(resetResult));
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    result = vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, fence);
    if (result != VK_SUCCESS) {
        if (result == VK_ERROR_DEVICE_LOST) {
            ctx.device_lost = true;
            throw std::runtime_error("Vulkan device lost during queue submit — GPU crash or timeout");
        }
        throw std::runtime_error("Failed to submit queue with fence: " + std::to_string(result));
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

    // Phase 8.3 fast-path: if the device has no pending work, skip the
    // full vkDeviceWaitIdle/command-pool-reset cycle. Common case in
    // tight inference loops where each op already calls
    // endSingleTimeCommands which submits and waits on its own fence.
    // This eliminates dozens of redundant device-wide barriers per
    // forward pass.
    if (!ctx.hasPendingWork && ctx.submittedFrames == 0 &&
        ctx.activeCommandBuffer == VK_NULL_HANDLE) {
        // No in-flight GPU work, but freed buffers may still be parked in the
        // deferred-free list. The GPU is idle here, so they are safe to return
        // to the allocator now — without this, a compute-only loop that never
        // takes the slow path below accumulates deferred frees until OOM.
        flush_deferred_frees(device_id);
        return;
    }

    // Submit any pending batched commands before synchronizing
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        submitBatchIfNeeded(device_id, true);  // Force submit any pending work
    }

    // Phase 8.3: replace `vkDeviceWaitIdle` with a per-submission fence wait.
    // Every submission this backend makes goes through `endSingleTimeCommandsAsync`,
    // which already attaches one of `frameFences[MAX_FRAMES_IN_FLIGHT]` to the
    // submission. `ensurePendingWorkComplete` walks those fences and waits only
    // on the ones that are actually pending (`vkGetFenceStatus == VK_NOT_READY`).
    //
    // Functionally equivalent to vkDeviceWaitIdle for our single-compute-queue
    // backend, but does not stall on unrelated queue activity (transfer queue
    // when one is added later, presentation queue if a swapchain is integrated).
    // It is also a step toward a timeline-semaphore-based design — the per-queue
    // timeline counter is just a single-fence collapse of `frameFences[]`.
    //
    // The original vkDeviceWaitIdle is preserved as a fallback in
    // `try_reset_device()` for the device-lost recovery path, where the queue
    // state is already corrupted and a hardware-level barrier is needed.
    ensurePendingWorkComplete(device_id);

    // Reset command pool and pool index — safe because all submitted command
    // buffers have completed (per-fence wait above is equivalent to idle for
    // this device's compute queue).
    vulkan::checkVk(vkResetCommandPool(ctx.device, ctx.commandPool, 0),
                    "Failed to reset command pool during synchronize");
    ctx.nextCommandBufferIndex = 0;

    // Reset batching state
    ctx.activeCommandBuffer = VK_NULL_HANDLE;
    ctx.operationsInBatch = 0;

    // Flush deferred frees now that all GPU work is complete
    flush_deferred_frees(device_id);

    // Phase 8.3: do NOT reset the descriptor pool on every synchronize().
    // The pool grows on demand (`grow()` doubles capacity on
    // `VK_ERROR_OUT_OF_POOL_MEMORY`); resetting on every synchronize() defeats
    // that growth and makes pool exhaustion paths chatty in tight inference
    // loops. The pool's ~64K-set ceiling is reset only at backend destruction.
    // If a future workload genuinely needs per-step reclamation, the user can
    // call `descriptorPool->reset()` explicitly.
}

auto VulkanBackend::is_device_lost(int32_t device_id) const -> bool {
    if (device_id < 0 || device_id >= device_count()) return true;
    return devices_[device_id].device_lost.load(std::memory_order_acquire);
}

auto VulkanBackend::try_reset_device(int32_t device_id) -> bool {
    if (device_id < 0 || device_id >= device_count()) return false;
    auto& ctx = devices_[device_id];
    std::lock_guard<std::recursive_mutex> lock(ctx.mutex);

    if (!ctx.device_lost.load(std::memory_order_acquire)) {
        return true;  // Not lost — nothing to do
    }

    // Attempt recovery: wait for device idle (the GPU may have recovered
    // from a transient TDR timeout after the driver resets it)
    VkResult result = vkDeviceWaitIdle(ctx.device);
    if (result == VK_ERROR_DEVICE_LOST) {
        // Truly lost — hardware failure or unrecoverable driver state
        return false;
    }
    if (result != VK_SUCCESS) {
        return false;
    }

    // Device responded — reset all state
    ctx.device_lost.store(false, std::memory_order_release);
    ctx.submittedFrames = 0;
    ctx.currentFrame = 0;
    ctx.hasPendingWork = false;
    ctx.activeCommandBuffer = VK_NULL_HANDLE;
    ctx.operationsInBatch = 0;

    // Reset command pool (all prior command buffers are now invalid)
    {
        VkResult resetResult = vkResetCommandPool(ctx.device, ctx.commandPool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
        if (resetResult == VK_ERROR_DEVICE_LOST) {
            // Can't recover — leave device_lost set and bail
            return false;
        }
        // Other errors during recovery are non-fatal — we already checked vkDeviceWaitIdle
    }
    ctx.nextCommandBufferIndex = 0;

    // Reinitialize command buffer pool
    initCommandBufferPool(ctx);

    // Reset descriptor pool
    if (ctx.descriptorPool) {
        ctx.descriptorPool->reset();
    }

    // Flush deferred frees
    flush_deferred_frees(device_id);

    return true;
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
        vulkan::checkVk(vkResetCommandBuffer(ctx.activeCommandBuffer, 0),
                        "Failed to reset batch command buffer");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vulkan::checkVk(vkBeginCommandBuffer(ctx.activeCommandBuffer, &beginInfo),
                        "Failed to begin batch command buffer");

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
    vulkan::checkVk(vkEndCommandBuffer(ctx.activeCommandBuffer),
                    "Failed to end batch command buffer");

    VkFence fence = ctx.frameFences[ctx.currentFrame];

    // Wait for the previous submission that used this frame slot before
    // resetting its fence (same guard as endSingleTimeCommandsAsync).
    // Resetting a fence whose submission is still executing is undefined
    // behaviour: the slot's completion signal is destroyed, so the command
    // buffer pool / deferred frees could recycle resources the GPU was still
    // using — observed as nondeterministic corruption in multi-step loops
    // (e.g. sequence-GRU diverging from t>=3 on RADV).
    if (ctx.submittedFrames >= DeviceContext::MAX_FRAMES_IN_FLIGHT) {
        waitForFrame(device_id, ctx.currentFrame);
    }

    // Reset fence before use
    VkResult resetResult = vkResetFences(ctx.device, 1, &fence);
    if (resetResult != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset batch fence: " + std::to_string(resetResult));
    }

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

    VkResult result = vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, ctx.fence_timeout_ns);
    if (result == VK_ERROR_DEVICE_LOST) {
        ctx.device_lost.store(true, std::memory_order_release);
        throw std::runtime_error("GPU device lost while waiting for frame fence. "
            "The GPU may have crashed or been reset.");
    }
    if (result == VK_TIMEOUT) {
        throw std::runtime_error("GPU frame fence wait timed out after " +
            std::to_string(ctx.fence_timeout_ns / 1'000'000'000ULL) + " seconds. "
            "This often indicates memory pressure or a shader hang.");
    }
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to wait for frame fence: " + std::to_string(result));
    }
}

} // namespace tenzor
