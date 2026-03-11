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
#include <atomic>
#include <span>
#include <vector>
#include <cstdio>
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
 * @brief Inplace kernel function pointer type for operations that modify tensors in-place.
 *
 * For operations like add_inplace, sub_inplace, etc. that modify the first tensor
 * in-place rather than creating a new output. This avoids the unnecessary tensor
 * copy that occurs when passing through std::span<const Tensor>.
 *
 * @param target The tensor to modify in-place (first input)
 * @param others Additional input tensors (remaining inputs)
 * @param attrs Operation-specific attributes
 * @return Reference to the modified target tensor
 */
using InplaceKernelFn = Tensor&(*)(Tensor&, std::span<const Tensor>, const OpAttributes&);

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

    /// Inplace kernel array for operations that modify tensors in-place
    std::array<InplaceKernelFn, OP_COUNT> inplace_kernels{};

    /// Device type this table serves
    Device::Type device_type{Device::Type::CPU};

    /// Backend instance (for memory operations, synchronization)
    Backend* backend{nullptr};

    /// Set to true after all kernels are registered; dispatch checks this
    std::atomic<bool> ready{false};

    /**
     * @brief Register a kernel for an operation.
     *
     * @param op Operation identifier
     * @param fn Kernel function pointer
     */
    void register_kernel(OpId op, KernelFn fn) noexcept {
#ifndef NDEBUG
        auto idx = static_cast<size_t>(op);
        if (kernels[idx] != nullptr) {
            fprintf(stderr, "[dispatch_table] WARNING: overwriting kernel for OpId %zu\n", idx);
        }
        if (single_output_kernels[idx] != nullptr || inplace_kernels[idx] != nullptr) {
            fprintf(stderr, "[dispatch_table] WARNING: OpId %zu already registered in a different kernel array\n", idx);
        }
#endif
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
#ifndef NDEBUG
        auto idx = static_cast<size_t>(op);
        if (single_output_kernels[idx] != nullptr) {
            fprintf(stderr, "[dispatch_table] WARNING: overwriting single_output kernel for OpId %zu\n", idx);
        }
        if (kernels[idx] != nullptr || inplace_kernels[idx] != nullptr) {
            fprintf(stderr, "[dispatch_table] WARNING: OpId %zu already registered in a different kernel array\n", idx);
        }
#endif
        single_output_kernels[static_cast<size_t>(op)] = fn;
    }

    /**
     * @brief Register an inplace kernel for operations that modify tensors in-place.
     *
     * Use this for operations like add_inplace, sub_inplace, etc. that modify
     * the first input tensor rather than creating a new output.
     *
     * @param op Operation identifier (should be an inplace op like OpId::AddInplace)
     * @param fn Inplace kernel function pointer
     */
    void register_inplace_kernel(OpId op, InplaceKernelFn fn) noexcept {
#ifndef NDEBUG
        auto idx = static_cast<size_t>(op);
        if (inplace_kernels[idx] != nullptr) {
            fprintf(stderr, "[dispatch_table] WARNING: overwriting inplace kernel for OpId %zu\n", idx);
        }
        if (kernels[idx] != nullptr || single_output_kernels[idx] != nullptr) {
            fprintf(stderr, "[dispatch_table] WARNING: OpId %zu already registered in a different kernel array\n", idx);
        }
#endif
        inplace_kernels[static_cast<size_t>(op)] = fn;
    }

    /**
     * @brief Check if a kernel is registered for an operation.
     *
     * @param op Operation identifier
     * @return true if kernel exists
     */
    [[nodiscard]] bool has_kernel(OpId op) const noexcept {
        return kernels[static_cast<size_t>(op)] != nullptr ||
               single_output_kernels[static_cast<size_t>(op)] != nullptr ||
               inplace_kernels[static_cast<size_t>(op)] != nullptr;
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
     * @brief Check if an inplace kernel is registered.
     *
     * @param op Operation identifier
     * @return true if inplace kernel exists
     */
    [[nodiscard]] bool has_inplace_kernel(OpId op) const noexcept {
        return inplace_kernels[static_cast<size_t>(op)] != nullptr;
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
     * @brief Get inplace kernel function pointer.
     *
     * @param op Operation identifier
     * @return Inplace kernel function pointer (nullptr if not registered)
     */
    [[nodiscard]] InplaceKernelFn get_inplace_kernel(OpId op) const noexcept {
        return inplace_kernels[static_cast<size_t>(op)];
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
        if (!ready.load(std::memory_order_acquire)) [[unlikely]] {
            throw std::runtime_error("Backend dispatch table not ready (backend not initialized)");
        }
        auto idx = static_cast<size_t>(op);
        if (idx >= kernels.size()) [[unlikely]] {
            throw_unsupported(op);
        }
        auto fn = kernels[idx];
        if (fn) [[likely]] {
            return fn(inputs, attrs);
        }
        // Fallback to single-output kernel (wrap result in vector)
        auto single_fn = single_output_kernels[idx];
        if (single_fn) {
            return {single_fn(inputs, attrs)};
        }
        // Inplace kernels must be called explicitly via dispatch_inplace()
        // to avoid const_cast UB. Do not fall through to inplace kernels here.
        throw_unsupported(op);
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
        if (!ready.load(std::memory_order_acquire)) [[unlikely]] {
            throw std::runtime_error("Backend dispatch table not ready (backend not initialized)");
        }
        auto idx = static_cast<size_t>(op);
        if (idx >= single_output_kernels.size()) [[unlikely]] {
            throw_unsupported(op);
        }
        auto fn = single_output_kernels[idx];
        if (fn) [[likely]] {
            return fn(inputs, attrs);
        }
        // Fall back to multi-output dispatch if no single-output kernel
        auto multi_fn = kernels[idx];
        if (!multi_fn) [[unlikely]] {
            throw_unsupported(op);
        }
        return multi_fn(inputs, attrs)[0];
    }

    /**
     * @brief Dispatch for inplace operations.
     *
     * Modifies the target tensor in-place without unnecessary copies.
     *
     * @param op Operation identifier (should be an inplace op)
     * @param target The tensor to modify in-place
     * @param others Additional input tensors
     * @param attrs Operation attributes
     * @return Reference to the modified target tensor
     * @throws std::runtime_error if operation not supported
     */
    Tensor& dispatch_inplace(
        OpId op,
        Tensor& target,
        std::span<const Tensor> others,
        const OpAttributes& attrs = {}) const
    {
        if (!ready.load(std::memory_order_acquire)) [[unlikely]] {
            throw std::runtime_error("Backend dispatch table not ready (backend not initialized)");
        }
        auto idx = static_cast<size_t>(op);
        if (idx >= inplace_kernels.size()) [[unlikely]] {
            throw_unsupported(op);
        }
        auto fn = inplace_kernels[idx];
        if (!fn) [[unlikely]] {
            throw_unsupported(op);
        }
        auto& result = fn(target, others, attrs);
        // Auto-bump version counter so kernel authors don't need to remember
        result.bump_version();
        return result;
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
        table.ready.store(false, std::memory_order_release);
        table.device_type = type;
        table.backend = backend;
    }

    /**
     * @brief Mark a backend's dispatch table as ready for use.
     *
     * Must be called after register_kernels() completes. Dispatch checks
     * this flag to avoid reading partially-initialized kernel tables.
     *
     * @param type Device type to mark as ready
     */
    static void mark_ready(Device::Type type) noexcept {
        tables_[static_cast<size_t>(type)].ready.store(true, std::memory_order_release);
    }

    /**
     * @brief Clear all dispatch tables.
     *
     * Zeroes out all kernel function pointers and backend pointers across
     * all device types. Must be called before backends are destroyed to
     * prevent dangling function pointers.
     */
    static void clear() noexcept {
        for (auto& table : tables_) {
            table.ready.store(false, std::memory_order_release);
            table.kernels.fill(nullptr);
            table.single_output_kernels.fill(nullptr);
            table.inplace_kernels.fill(nullptr);
            table.backend = nullptr;
        }
    }

    /**
     * @brief Clear a single backend's dispatch table.
     *
     * Used by unload_backend() to invalidate a specific device type's
     * kernel registrations without affecting other backends.
     *
     * @param type Device type to clear
     */
    static void clear_backend(Device::Type type) noexcept {
        auto& table = tables_[static_cast<size_t>(type)];
        table.ready.store(false, std::memory_order_release);
        table.kernels.fill(nullptr);
        table.single_output_kernels.fill(nullptr);
        table.inplace_kernels.fill(nullptr);
        table.backend = nullptr;
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
        case Device::Type::WebGPU: return "WebGPU";
    }
    return "Unknown";
}

} // namespace tenzor
