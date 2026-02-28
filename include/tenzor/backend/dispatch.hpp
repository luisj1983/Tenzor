/**
 * @file dispatch.hpp
 * @brief Central dispatcher for routing operations to appropriate backends
 *
 * Provides automatic backend selection and device compatibility checking
 * for tensor operations. Routes operations to CPU, CUDA, ROCm, or OneAPI
 * backends based on tensor device placement.
 */

#pragma once

#include <span>
#include <vector>
#include "../core/tensor.hpp"
#include "backend.hpp"

namespace tenzor {

/**
 * @brief Central operation dispatcher for backend routing.
 *
 * The Dispatcher class provides static methods for automatically routing
 * tensor operations to the appropriate backend implementation based on
 * device placement. It handles device compatibility checking and backend
 * selection logic.
 *
 * All operations in Tenzor funnel through this dispatcher, which ensures
 * that operations are executed on the correct hardware backend.
 *
 * @note For dispatching operations, use the OpId-based dispatch in
 *       fast_dispatch.hpp instead (e.g., dispatch<OpId::Add>({a, b})).
 * @note This class contains only static methods (utility class pattern).
 * @see Backend for backend interface
 * @see BackendDispatchTable for OpId-based kernel registration
 */
class Dispatcher {
public:
    /**
     * @brief Get appropriate backend for tensor set.
     *
     * Determines which backend should execute operations for the given
     * tensors. All tensors should be on the same device type.
     *
     * @param tensors Tensors to determine backend for
     * @return Pointer to appropriate backend
     * @throws std::runtime_error if tensors are on incompatible devices
     *
     * @code
     * Backend* backend = Dispatcher::get_backend({tensor_a, tensor_b});
     * std::cout << "Using backend: " << backend->name() << std::endl;
     * @endcode
     */
    static auto get_backend(std::span<const Tensor> tensors) -> Backend*;

    /**
     * @brief Check if tensors are on compatible devices.
     *
     * Verifies that all tensors are on the same device type and index,
     * which is required for most operations.
     *
     * @param tensors Tensors to check
     * @return true if all tensors are on the same device
     *
     * @code
     * if (!Dispatcher::check_device_compatibility({a, b, c})) {
     *     throw std::runtime_error("Tensors must be on same device");
     * }
     * @endcode
     */
    static auto check_device_compatibility(std::span<const Tensor> tensors) -> bool;
};

} // namespace tenzor
