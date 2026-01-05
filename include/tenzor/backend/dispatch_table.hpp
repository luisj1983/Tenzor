/**
 * @file dispatch_table.hpp
 * @brief O(1) dispatch table infrastructure for direct kernel lookup
 *
 * Provides cache-aligned function pointer tables for single-dispatch O(1)
 * kernel lookup. Replaces the string-based double dispatch with a direct
 * array lookup indexed by OpId.
 *
 * Design:
 * - Single dispatch: table[device_type].kernels[op_id] -> kernel function
 * - No string comparisons, no hash lookups, no virtual calls
 * - ~10-20ns dispatch vs 100-1000ns for string-based dispatch
 */

#pragma once

#include <array>
#include <span>
#include <vector>
#include <stdexcept>
#include <string>
#include "../core/tensor.hpp"
#include "../core/device.hpp"
#include "../ops/op_id.hpp"
#include "backend.hpp"  // For OpAttributes

namespace tenzor {

// Forward declaration
class Backend;

/**
 * @brief Kernel function pointer type for direct dispatch.
 *
 * Raw function pointer (not std::function) for maximum performance.
 * All kernels conform to this signature.
 *
 * @param inputs Input tensors for the operation
 * @param attrs Operation-specific attributes (stride, algorithm hints, etc.)
 * @return Output tensor(s) produced by the kernel
 */
using KernelFn = std::vector<Tensor>(*)(std::span<const Tensor>, const OpAttributes&);

/**
 * @brief Single-output kernel function pointer type for optimized dispatch.
 *
 * For operations that produce exactly one output tensor, this avoids
 * the overhead of vector allocation. Most ops (add, mul, matmul, linear, etc.)
 * are single-output and benefit from this optimization.
 *
 * @param inputs Input tensors for the operation
 * @param attrs Operation-specific attributes
 * @return Single output tensor (no heap allocation for container)
 */
using SingleOutputKernelFn = Tensor(*)(std::span<const Tensor>, const OpAttributes&);

/**
 * @brief Number of device types in the dispatch system.
 *
 * Must match Device::Type enum count.
 */
inline constexpr size_t DEVICE_TYPE_COUNT = 7;

/**
 * @brief Cache-aligned dispatch table for a single backend.
 *
 * Contains direct function pointers to kernel implementations indexed by OpId.
 * The array is cache-aligned to minimize cache misses during dispatch.
 *
 * @code
 * BackendDispatchTable table;
 * table.device_type = Device::Type::CPU;
 *
 * // Register kernel
 * table.register_kernel(OpId::Add, cpu_add_kernel);
 *
 * // Dispatch (O(1) array lookup)
 * auto result = table.dispatch(OpId::Add, {a, b}, {});
 * @endcode
 */
struct alignas(64) BackendDispatchTable {
    /// Function pointer array indexed by OpId (nullptr = not supported)
    std::array<KernelFn, OP_COUNT> kernels{};

    /// Single-output kernel array for optimized dispatch (no vector allocation)
    std::array<SingleOutputKernelFn, OP_COUNT> single_output_kernels{};

    /// Device type this table serves
    Device::Type device_type{Device::Type::CPU};

    /// Backend instance (for memory operations, synchronization)
    Backend* backend{nullptr};

    /**
     * @brief Register a kernel for an operation.
     *
     * @param op Operation identifier
     * @param fn Kernel function pointer
     */
    void register_kernel(OpId op, KernelFn fn) noexcept {
        kernels[static_cast<size_t>(op)] = fn;
    }

    /**
     * @brief Register a single-output kernel for optimized dispatch.
     *
     * Use this for operations that always produce exactly one output tensor.
     * This avoids vector allocation overhead on every dispatch call.
     *
     * @param op Operation identifier
     * @param fn Single-output kernel function pointer
     */
    void register_single_output_kernel(OpId op, SingleOutputKernelFn fn) noexcept {
        single_output_kernels[static_cast<size_t>(op)] = fn;
    }

    /**
     * @brief Check if a kernel is registered for an operation.
     *
     * @param op Operation identifier
     * @return true if kernel exists
     */
    [[nodiscard]] bool has_kernel(OpId op) const noexcept {
        return kernels[static_cast<size_t>(op)] != nullptr ||
               single_output_kernels[static_cast<size_t>(op)] != nullptr;
    }

    /**
     * @brief Check if a single-output kernel is registered.
     *
     * @param op Operation identifier
     * @return true if single-output kernel exists
     */
    [[nodiscard]] bool has_single_output_kernel(OpId op) const noexcept {
        return single_output_kernels[static_cast<size_t>(op)] != nullptr;
    }

    /**
     * @brief Get kernel function pointer for an operation.
     *
     * @param op Operation identifier
     * @return Kernel function pointer (nullptr if not registered)
     */
    [[nodiscard]] KernelFn get_kernel(OpId op) const noexcept {
        return kernels[static_cast<size_t>(op)];
    }

    /**
     * @brief Get single-output kernel function pointer.
     *
     * @param op Operation identifier
     * @return Single-output kernel function pointer (nullptr if not registered)
     */
    [[nodiscard]] SingleOutputKernelFn get_single_output_kernel(OpId op) const noexcept {
        return single_output_kernels[static_cast<size_t>(op)];
    }

    /**
     * @brief Dispatch operation to kernel.
     *
     * Single O(1) lookup: array[op_id] -> direct function call.
     *
     * @param op Operation identifier
     * @param inputs Input tensors
     * @param attrs Operation attributes
     * @return Output tensors
     * @throws std::runtime_error if operation not supported
     */
    [[nodiscard]] std::vector<Tensor> dispatch(
        OpId op,
        std::span<const Tensor> inputs,
        const OpAttributes& attrs) const
    {
        auto fn = kernels[static_cast<size_t>(op)];
        if (!fn) [[unlikely]] {
            throw_unsupported(op);
        }
        return fn(inputs, attrs);
    }

    /**
     * @brief Optimized dispatch for single-output operations.
     *
     * Returns Tensor directly without vector allocation overhead.
     * Use dispatch_single<OpId>() from fast_dispatch.hpp for convenience.
     *
     * @param op Operation identifier
     * @param inputs Input tensors
     * @param attrs Operation attributes
     * @return Single output tensor
     * @throws std::runtime_error if operation not supported
     */
    [[nodiscard]] Tensor dispatch_single(
        OpId op,
        std::span<const Tensor> inputs,
        const OpAttributes& attrs = {}) const
    {
        auto fn = single_output_kernels[static_cast<size_t>(op)];
        if (fn) [[likely]] {
            return fn(inputs, attrs);
        }
        // Fall back to multi-output dispatch if no single-output kernel
        auto multi_fn = kernels[static_cast<size_t>(op)];
        if (!multi_fn) [[unlikely]] {
            throw_unsupported(op);
        }
        return multi_fn(inputs, attrs)[0];
    }

private:
    [[noreturn]] void throw_unsupported(OpId op) const;
};

/**
 * @brief Global registry of dispatch tables indexed by device type.
 *
 * Provides O(1) access to backend dispatch tables. The registry is
 * a simple array lookup: tables_[device_type].
 *
 * @code
 * // Get CPU dispatch table
 * auto& table = DispatchTableRegistry::get_table(Device::Type::CPU);
 *
 * // Dispatch operation
 * auto result = table.dispatch(OpId::MatMul, {a, b}, {});
 * @endcode
 */
class DispatchTableRegistry {
public:
    /**
     * @brief Get dispatch table for a device type.
     *
     * O(1) array lookup by device type enum value.
     *
     * @param type Device type
     * @return Reference to dispatch table
     */
    static BackendDispatchTable& get_table(Device::Type type) noexcept {
        return tables_[static_cast<size_t>(type)];
    }

    /**
     * @brief Get dispatch table for a device type (const).
     *
     * @param type Device type
     * @return Const reference to dispatch table
     */
    static const BackendDispatchTable& get_table_const(Device::Type type) noexcept {
        return tables_[static_cast<size_t>(type)];
    }

    /**
     * @brief Initialize dispatch table for a backend.
     *
     * @param type Device type
     * @param backend Backend instance
     */
    static void register_backend(Device::Type type, Backend* backend) noexcept {
        auto& table = tables_[static_cast<size_t>(type)];
        table.device_type = type;
        table.backend = backend;
    }

    /**
     * @brief Check if a backend is registered.
     *
     * @param type Device type
     * @return true if backend is registered
     */
    static bool has_backend(Device::Type type) noexcept {
        return tables_[static_cast<size_t>(type)].backend != nullptr;
    }

    /**
     * @brief Get backend instance for device type.
     *
     * @param type Device type
     * @return Backend pointer (nullptr if not registered)
     */
    static Backend* get_backend(Device::Type type) noexcept {
        return tables_[static_cast<size_t>(type)].backend;
    }

private:
    /// Dispatch tables indexed by Device::Type
    static std::array<BackendDispatchTable, DEVICE_TYPE_COUNT> tables_;
};

/**
 * @brief Convert device type to string for error messages.
 *
 * @param type Device type
 * @return String representation
 */
inline const char* device_type_to_string(Device::Type type) noexcept {
    switch (type) {
        case Device::Type::CPU:    return "CPU";
        case Device::Type::CUDA:   return "CUDA";
        case Device::Type::ROCm:   return "ROCm";
        case Device::Type::OneAPI: return "OneAPI";
        case Device::Type::Vulkan: return "Vulkan";
        case Device::Type::Metal:  return "Metal";
        case Device::Type::WebGPU: return "WebGPU";
    }
    return "Unknown";
}

} // namespace tenzor
