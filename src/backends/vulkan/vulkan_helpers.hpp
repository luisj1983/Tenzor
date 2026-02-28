/**
 * @file vulkan_helpers.hpp
 * @brief Shared inline helper functions for the Vulkan backend implementation files.
 *
 * These helpers are used across multiple translation units (vulkan_memory.cpp,
 * vulkan_commands.cpp, vulkan_dispatch.cpp, vulkan_ops.cpp, vulkan_backend.cpp).
 */

#pragma once

#include "vulkan_backend.hpp"
#include <vulkan/vulkan.h>

// Undefine Xlib Bool macro that conflicts with DType::Bool
// (Xlib.h is included via vulkan_xlib.h when VK_USE_PLATFORM_XLIB_KHR is defined)
#ifdef Bool
#undef Bool
#endif

#include <span>
#include <vector>
#include <cstdint>

namespace tenzor {

// Helper function to insert transfer-to-compute barrier
// Required when a compute shader reads from a buffer that was just written by a transfer op
inline void insertTransferToComputeBarrier(VkCommandBuffer cmdBuffer) {
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmdBuffer,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    }
    // When batching is disabled, each operation is submitted separately so no barrier needed
}

// Helper function to insert a pre-read barrier
// Required BEFORE a compute shader that reads from a buffer that may have pending writes
// from a previous compute operation in the same batch
inline void insertPreReadBarrier(VkCommandBuffer cmdBuffer) {
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmdBuffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    }
}

// Helper function to insert compute shader memory barrier.
// Always inserts a compute-to-compute barrier (RAW hazard between consecutive
// compute dispatches). When batching is disabled, also inserts a transfer/host
// barrier for immediate readback.
inline void insertComputeBarrier(VkCommandBuffer cmdBuffer) {
    // Compute-to-compute barrier (needed in both modes for RAW hazard safety)
    VkMemoryBarrier computeBarrier{};
    computeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    computeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    computeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmdBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &computeBarrier, 0, nullptr, 0, nullptr);

    if constexpr (!vulkan_config::USE_COMMAND_BATCHING) {
        // Non-batching mode: also add transfer/host barrier for immediate readback
        VkMemoryBarrier transferBarrier{};
        transferBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        transferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        transferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmdBuffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                            0, 1, &transferBarrier, 0, nullptr, 0, nullptr);
    }
}

// Helper: Check if two shapes are broadcastable
static inline bool are_broadcastable(std::span<const int64_t> shape_a,
                               std::span<const int64_t> shape_b) {
    size_t max_ndim = std::max(shape_a.size(), shape_b.size());

    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t dim_a = i < shape_a.size() ? shape_a[shape_a.size() - 1 - i] : 1;
        int64_t dim_b = i < shape_b.size() ? shape_b[shape_b.size() - 1 - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }

    return true;
}

// Helper: Compute the broadcasted output shape
static inline std::vector<int64_t> compute_broadcast_shape(std::span<const int64_t> shape_a,
                                                     std::span<const int64_t> shape_b) {
    size_t max_ndim = std::max(shape_a.size(), shape_b.size());
    std::vector<int64_t> result(max_ndim);

    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t dim_a = i < shape_a.size() ? shape_a[shape_a.size() - 1 - i] : 1;
        int64_t dim_b = i < shape_b.size() ? shape_b[shape_b.size() - 1 - i] : 1;

        if (dim_a == dim_b || dim_a == 1 || dim_b == 1) {
            result[max_ndim - 1 - i] = std::max(dim_a, dim_b);
        } else {
            throw std::runtime_error("Shapes are not broadcastable");
        }
    }

    return result;
}

// Helper: Compute strides for broadcasting
static inline std::vector<uint32_t> compute_broadcast_strides(std::span<const int64_t> shape,
                                                        std::span<const int64_t> broadcast_shape) {
    std::vector<uint32_t> strides(broadcast_shape.size(), 0);

    // Compute normal strides for the original shape
    std::vector<int64_t> original_strides(shape.size());
    if (!shape.empty()) {
        original_strides.back() = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            original_strides[i] = original_strides[i + 1] * shape[i + 1];
        }
    }

    // Map to broadcast strides
    int64_t offset = static_cast<int64_t>(broadcast_shape.size()) - static_cast<int64_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 1) {
            strides[offset + i] = 0;  // Broadcasting dimension
        } else {
            strides[offset + i] = static_cast<uint32_t>(original_strides[i]);
        }
    }

    return strides;
}

// Helper to check if a 2D tensor is a simple transpose of a contiguous tensor
// Returns true if the tensor's strides indicate a simple row-column swap
static inline bool isSimpleTranspose2D(const Tensor& t) {
    if (t.ndim() != 2) return false;
    auto strides = t.strides();
    auto shape = t.shape();
    // A transposed 2D contiguous tensor has strides [1, rows] where rows = shape[0]
    // (original was [cols, 1] before transpose)
    return strides[0] == 1 && strides[1] == static_cast<int64_t>(shape[0]);
}

} // namespace tenzor
