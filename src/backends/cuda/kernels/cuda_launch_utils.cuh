#pragma once

/**
 * @file cuda_launch_utils.cuh
 * @brief CUDA kernel launch utilities with occupancy-based block sizing
 *
 * Provides optimal_launch_config() to replace hardcoded BLOCK_SIZE = 256
 * with cudaOccupancyMaxPotentialBlockSize for better GPU utilization
 * across different architectures.
 */

#include <cuda_runtime.h>
#include <algorithm>
#include <unordered_map>
#include <utility>

namespace tenzor {
namespace cuda {

/**
 * @brief Compute optimal grid and block dimensions using CUDA occupancy API.
 *
 * Uses cudaOccupancyMaxPotentialBlockSize to determine the block size
 * that maximizes occupancy for a given kernel. Results are cached per
 * kernel function pointer to avoid repeated API calls.
 *
 * @tparam KernelFunc CUDA kernel function type
 * @param kernel Pointer to the CUDA kernel function
 * @param num_elements Total number of elements to process
 * @param dynamic_smem Dynamic shared memory per block (bytes)
 * @return Pair of (num_blocks, block_size)
 */
template<typename KernelFunc>
inline std::pair<int, int> optimal_launch_config(
    KernelFunc kernel, int64_t num_elements, size_t dynamic_smem = 0)
{
    // Cache block size per kernel function pointer to avoid repeated
    // cudaOccupancyMaxPotentialBlockSize calls (~10-100us each)
    static thread_local std::unordered_map<const void*, int> block_cache;
    const void* key = reinterpret_cast<const void*>(kernel);

    int block_size;
    auto it = block_cache.find(key);
    if (it != block_cache.end()) {
        block_size = it->second;
    } else {
        int min_grid_size = 0;
        block_size = 256;  // fallback
        cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &block_size, kernel, dynamic_smem, 0);
        if (err != cudaSuccess) {
            block_size = 256;  // fallback on error
        }
        block_cache[key] = block_size;
    }

    int num_blocks = static_cast<int>(
        std::min(static_cast<int64_t>((num_elements + block_size - 1) / block_size),
                 static_cast<int64_t>(2147483647)));  // CUDA x-dim max: 2^31-1
    num_blocks = std::max(1, num_blocks);

    return {num_blocks, block_size};
}

/**
 * @brief Compute grid size for a given block size and element count.
 *
 * Simple utility when block size is already known (e.g., from a previous
 * optimal_launch_config call or when using a fixed block size).
 *
 * @param num_elements Total number of elements to process
 * @param block_size Threads per block
 * @return Number of blocks to launch
 */
inline int compute_grid_size(int64_t num_elements, int block_size = 256) {
    int num_blocks = static_cast<int>(
        (num_elements + block_size - 1) / block_size);
    return std::max(1, std::min(num_blocks, 2147483647));  // CUDA x-dim max: 2^31-1
}

} // namespace cuda
} // namespace tenzor
