/**
 * @file registry.hpp
 * @brief Operation kernel registration and dispatch system
 *
 * Provides a thread-safe registry for mapping operation names and device
 * types to kernel implementations. Enables extensible operation support
 * through dynamic kernel registration.
 */

#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <span>
#include <shared_mutex>
#include "../core/tensor.hpp"
#include "../core/device.hpp"
#include "backend.hpp"

namespace tenzor {

/**
 * @brief Kernel function signature.
 *
 * Standard signature for all operation kernels. Kernels receive input
 * tensors and operation attributes, then return output tensor(s).
 *
 * @param inputs Span of input tensors
 * @param attrs Operation-specific attributes
 * @return Vector of output tensors
 *
 * @code
 * auto add_kernel(std::span<const Tensor> inputs,
 *                 const OpAttributes& attrs) -> std::vector<Tensor> {
 *     return {inputs[0] + inputs[1]};
 * }
 * @endcode
 */
using KernelFunction = std::function<
    std::vector<Tensor>(std::span<const Tensor>, const OpAttributes&)
>;

/**
 * @brief Thread-safe registry for operation kernels.
 *
 * Maintains a mapping from (operation_name, device_type) pairs to kernel
 * implementations. Supports dynamic registration and lookup of kernels at
 * runtime.
 *
 * The registry uses a two-level map:
 * - First level: operation name -> device map
 * - Second level: device type -> kernel function
 *
 * This allows efficient lookup and supports multiple device implementations
 * for each operation.
 *
 * @code
 * OperationRegistry& registry = operation_registry();
 *
 * // Register CPU implementation
 * registry.register_kernel("add", Device::Type::CPU, cpu_add_kernel);
 *
 * // Register CUDA implementation
 * registry.register_kernel("add", Device::Type::CUDA, cuda_add_kernel);
 *
 * // Dispatch based on tensor device
 * auto result = registry.dispatch("add", {tensor_a, tensor_b}, {});
 * @endcode
 *
 * @note Thread-safe for concurrent registration and dispatch.
 * @see KernelFunction for kernel signature
 * @see operation_registry() for global instance
 */
class OperationRegistry {
public:
    OperationRegistry() = default;

    /**
     * @brief Register kernel implementation for operation and device.
     *
     * Associates a kernel function with a specific operation name and
     * device type. Overwrites any existing registration for the same
     * (op_name, device_type) pair.
     *
     * @param op_name Operation identifier (e.g., "add", "matmul", "conv2d")
     * @param device_type Target device type (CPU, CUDA, etc.)
     * @param kernel Kernel function to register
     *
     * @code
     * registry.register_kernel("matmul", Device::Type::CUDA,
     *     [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
     *         return {cublas_matmul(inputs[0], inputs[1])};
     *     }
     * );
     * @endcode
     *
     * @note Thread-safe with exclusive locking.
     */
    auto register_kernel(std::string_view op_name,
                        Device::Type device_type,
                        KernelFunction kernel) -> void;

    /**
     * @brief Dispatch operation to appropriate kernel.
     *
     * Looks up and executes the kernel for the given operation and device
     * type (inferred from input tensors). Returns the kernel's output.
     *
     * @param op_name Operation identifier
     * @param inputs Input tensors (device type determines kernel selection)
     * @param attrs Operation-specific attributes
     * @return Vector of output tensors
     * @throws std::runtime_error if operation not registered for device type
     *
     * @code
     * Tensor a({3, 4}, DType::Float32, Device::cuda(0));
     * Tensor b({3, 4}, DType::Float32, Device::cuda(0));
     * auto result = registry.dispatch("add", {a, b}, {});
     * @endcode
     *
     * @note Thread-safe with shared locking for read access.
     */
    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor>;

    /**
     * @brief Check if kernel is registered.
     *
     * @param op_name Operation name to check
     * @param device_type Device type to check
     * @return true if kernel exists for (op_name, device_type)
     *
     * @code
     * if (registry.has_kernel("conv2d", Device::Type::CUDA)) {
     *     std::cout << "CUDA convolution available" << std::endl;
     * }
     * @endcode
     */
    auto has_kernel(std::string_view op_name, Device::Type device_type) const -> bool;

    /**
     * @brief Clear all registered kernels.
     *
     * Removes all kernel registrations. Must be called before backends
     * are destroyed to prevent dangling function pointers captured in lambdas.
     *
     * @note Thread-safe with exclusive locking.
     */
    auto clear() -> void {
        std::unique_lock lock(mutex_);
        kernels_.clear();
    }

    /**
     * @brief Get list of all registered operation names.
     *
     * @return Vector of unique operation names
     *
     * @code
     * auto ops = registry.registered_operations();
     * for (const auto& op : ops) {
     *     std::cout << "Operation: " << op << std::endl;
     * }
     * @endcode
     */
    auto registered_operations() const -> std::vector<std::string>;

private:
    mutable std::shared_mutex mutex_;  ///< Reader-writer lock for thread safety
    std::unordered_map<
        std::string,
        std::unordered_map<Device::Type, KernelFunction>
    > kernels_;  ///< Two-level map: operation -> device -> kernel
};

/**
 * @brief Get global operation registry singleton.
 *
 * Thread-safe singleton providing access to the global kernel registry.
 * All kernel registrations and dispatches go through this instance.
 *
 * @return Reference to global OperationRegistry
 *
 * @code
 * auto& registry = operation_registry();
 * registry.register_kernel("my_op", Device::Type::CPU, my_kernel);
 * @endcode
 *
 * @note Thread-safe (uses static local initialization).
 */
auto operation_registry() -> OperationRegistry&;

} // namespace tenzor
