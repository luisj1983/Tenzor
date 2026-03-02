#pragma once

/**
 * @file launch_config.cuh
 * @brief Dynamic occupancy-based CUDA kernel launch configuration
 *
 * Wraps cudaOccupancyMaxPotentialBlockSize to avoid hardcoded block sizes
 * and maximize GPU occupancy across different architectures.
 *
 * This header provides a convenient LaunchConfig struct and optimal_launch()
 * helpers. For the full occupancy API with caching, see cuda_launch_utils.cuh.
 */

#ifdef TENZOR_HAS_CUDA
#include <cuda_runtime.h>
#include "cuda_launch_utils.cuh"

namespace tenzor::cuda {

/**
 * @brief Simple struct holding grid and block dimensions for kernel launches.
 */
struct LaunchConfig {
    int grid_size;
    int block_size;
};

/**
 * @brief Compute optimal launch configuration using cudaOccupancyMaxPotentialBlockSize.
 *
 * This avoids hardcoded block sizes and maximizes GPU occupancy.
 * Results are cached per kernel function pointer via optimal_launch_config().
 *
 * @tparam KernelFunc CUDA kernel function type
 * @param kernel Pointer to the CUDA kernel function
 * @param num_elements Total number of elements to process
 * @param shared_mem Dynamic shared memory per block (bytes)
 * @return LaunchConfig with optimal grid and block sizes
 */
template<typename KernelFunc>
LaunchConfig optimal_launch(KernelFunc kernel, int num_elements, int shared_mem = 0) {
    auto [grid, block] = optimal_launch_config(kernel, static_cast<int64_t>(num_elements),
                                                static_cast<size_t>(shared_mem));
    return {grid, block};
}

/**
 * @brief Simpler overload for element-wise kernels with no shared memory.
 */
template<typename KernelFunc>
LaunchConfig optimal_launch(KernelFunc kernel, int64_t num_elements) {
    auto [grid, block] = optimal_launch_config(kernel, num_elements, 0);
    return {grid, block};
}

} // namespace tenzor::cuda

#endif // TENZOR_HAS_CUDA
