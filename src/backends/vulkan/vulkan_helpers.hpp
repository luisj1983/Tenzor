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

// ============================================================================
// Typed barrier API
// ============================================================================

/// Barrier types for common Vulkan synchronization patterns.
enum class BarrierType {
    TransferToCompute,  ///< Transfer write → compute read (e.g., after vkCmdCopyBuffer)
    ComputeToCompute,   ///< Compute write → compute read (RAW between dispatches)
    ComputeToHost       ///< Compute write → transfer/host read (for readback)
};

/// Insert a typed pipeline barrier.
inline void insertBarrier(VkCommandBuffer cmdBuffer, BarrierType type) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

    switch (type) {
    case BarrierType::TransferToCompute:
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmdBuffer,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &barrier, 0, nullptr, 0, nullptr);
        break;

    case BarrierType::ComputeToCompute:
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmdBuffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &barrier, 0, nullptr, 0, nullptr);
        break;

    case BarrierType::ComputeToHost:
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmdBuffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                            0, 1, &barrier, 0, nullptr, 0, nullptr);
        break;
    }
}

// ============================================================================
// Legacy barrier API (delegates to typed API)
// ============================================================================

/// Transfer → compute barrier (after buffer copy/fill).
inline void insertTransferToComputeBarrier(VkCommandBuffer cmdBuffer) {
    insertBarrier(cmdBuffer, BarrierType::TransferToCompute);
}

/// Pre-read compute barrier for RAW (Read-After-Write) dependencies between dispatches.
/// Always inserted regardless of command batching mode — without batching, consecutive
/// dispatches within a single command buffer still need barriers to ensure write
/// visibility before the next read.
inline void insertPreReadBarrier(VkCommandBuffer cmdBuffer) {
    insertBarrier(cmdBuffer, BarrierType::ComputeToCompute);
}

/// Post-dispatch compute barrier for GPU-only workloads.
/// Only ensures SHADER_WRITE → SHADER_READ visibility between dispatches.
/// Use this when the next operation is another compute dispatch (no host readback).
inline void insertComputeOnlyBarrier(VkCommandBuffer cmdBuffer) {
    insertBarrier(cmdBuffer, BarrierType::ComputeToCompute);
}

/// Post-dispatch compute barrier with host readback visibility.
/// Ensures both SHADER_WRITE → SHADER_READ and compute → host/transfer visibility.
/// Use this when a host readback (synchronize, copy-to-host) may follow.
inline void insertComputeBarrier(VkCommandBuffer cmdBuffer) {
    insertBarrier(cmdBuffer, BarrierType::ComputeToCompute);
    insertBarrier(cmdBuffer, BarrierType::ComputeToHost);
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

/// Compute number of workgroups for a given element count and workgroup size.
/// Equivalent to ceil(n / wg_size).
inline uint32_t div_wg(uint64_t n, uint32_t wg_size) {
    return static_cast<uint32_t>((n + wg_size - 1) / wg_size);
}

} // namespace tenzor
