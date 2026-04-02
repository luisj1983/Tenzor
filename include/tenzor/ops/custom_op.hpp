/**
 * @file custom_op.hpp
 * @brief Runtime custom operation registration API
 *
 * Allows users to register custom operations at runtime (after initialization),
 * extending the dispatch system beyond the built-in OpId enum.
 *
 * Built-in ops continue to use O(1) array dispatch (zero performance change).
 * Custom ops use an overflow hash map (O(1) amortized, behind a shared_mutex).
 *
 * @code
 * // Register a custom op on CPU
 * auto my_op = tenzor::register_custom_op("my_namespace::custom_relu",
 *     Device::Type::CPU,
 *     [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
 *         auto& input = inputs[0];
 *         return input.clone();  // placeholder
 *     });
 *
 * // Dispatch it
 * auto result = tenzor::dispatch_custom_op(my_op, {tensor});
 * @endcode
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <span>

#include "../core/tensor.hpp"
#include "../core/device.hpp"
#include "../backend/backend.hpp"
#include "op_id.hpp"

namespace tenzor {

/**
 * @brief Opaque handle for custom operations.
 *
 * Stores a uint32_t that is either a built-in OpId (< OP_COUNT) or a
 * dynamically assigned custom op ID (>= OP_COUNT).
 */
struct CustomOpId {
    uint32_t value;

    /// Implicit conversion from built-in OpId
    CustomOpId(OpId id) : value(static_cast<uint32_t>(id)) {}

    /// Explicit construction from raw value
    explicit CustomOpId(uint32_t v) : value(v) {}

    /// Check if this is a built-in operation
    bool is_builtin() const { return value < OP_COUNT; }

    /// Convert to built-in OpId (only valid if is_builtin())
    OpId as_builtin() const {
        assert(is_builtin());
        return static_cast<OpId>(value);
    }

    bool operator==(const CustomOpId& other) const { return value == other.value; }
    bool operator!=(const CustomOpId& other) const { return value != other.value; }
};

/// Custom kernel function type (uses std::function to support closures/lambdas)
using CustomKernelFn = std::function<Tensor(std::span<const Tensor>, const OpAttributes&)>;

/**
 * @brief Singleton registry for custom operations.
 *
 * Thread-safe via shared_mutex. Registration (write) is exclusive;
 * name lookup (read) is shared.
 */
class CustomOpRegistry {
public:
    static auto instance() -> CustomOpRegistry&;

    /**
     * @brief Register a new custom op by name.
     *
     * If the name already exists, returns the existing ID.
     * Thread-safe (takes exclusive lock).
     *
     * @param name Unique operation name (e.g., "my_namespace::my_op")
     * @return CustomOpId for the operation
     */
    auto register_op(const std::string& name) -> CustomOpId;

    /**
     * @brief Look up a custom op by name.
     *
     * Thread-safe (takes shared lock).
     *
     * @param name Operation name
     * @return CustomOpId if found, nullopt otherwise
     */
    auto find_op(std::string_view name) const -> std::optional<CustomOpId>;

    /**
     * @brief Get the name of a custom op.
     *
     * @param id Custom op ID (must be >= OP_COUNT)
     * @return Operation name, or empty string if not found
     */
    auto op_name(CustomOpId id) const -> std::string_view;

    /**
     * @brief Register a kernel for a custom op on a specific device.
     *
     * Thread-safe (takes exclusive lock on the device's custom kernel map).
     *
     * @param id Custom op ID
     * @param device_type Target device type
     * @param kernel Kernel function (std::function, supports closures)
     */
    void register_kernel(CustomOpId id, Device::Type device_type, CustomKernelFn kernel);

    /**
     * @brief Dispatch a custom op.
     *
     * Thread-safe (takes shared lock on the device's custom kernel map).
     *
     * @param id Custom op ID
     * @param device_type Device type to dispatch on
     * @param inputs Input tensors
     * @param attrs Operation attributes
     * @return Output tensor
     * @throws std::runtime_error if no kernel registered for this op/device
     */
    auto dispatch(CustomOpId id, Device::Type device_type,
                  std::span<const Tensor> inputs,
                  const OpAttributes& attrs) const -> Tensor;

    /**
     * @brief Check if a kernel is registered for a custom op on a device.
     */
    bool has_kernel(CustomOpId id, Device::Type device_type) const;

private:
    CustomOpRegistry() = default;

    mutable std::shared_mutex registry_mutex_;
    std::unordered_map<std::string, uint32_t> name_to_id_;
    std::vector<std::string> id_to_name_;
    uint32_t next_id_ = OP_COUNT;

    // Per-device custom kernel maps, protected by their own mutexes
    struct DeviceKernels {
        mutable std::shared_mutex mutex;
        std::unordered_map<uint32_t, CustomKernelFn> kernels;
    };
    std::array<DeviceKernels, static_cast<size_t>(Device::Type::COUNT)> device_kernels_;
};

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Register a custom operation with a kernel on a single device.
 *
 * @param name Unique operation name
 * @param device_type Device type for the kernel
 * @param kernel Kernel function
 * @return CustomOpId for dispatching the operation
 */
auto register_custom_op(const std::string& name,
                        Device::Type device_type,
                        CustomKernelFn kernel) -> CustomOpId;

/**
 * @brief Register a custom operation with kernels on multiple devices.
 *
 * @param name Unique operation name
 * @param kernels List of (device_type, kernel) pairs
 * @return CustomOpId for dispatching the operation
 */
auto register_custom_op(const std::string& name,
                        std::initializer_list<std::pair<Device::Type, CustomKernelFn>> kernels)
    -> CustomOpId;

/**
 * @brief Dispatch a custom operation.
 *
 * @param id Custom op ID returned by register_custom_op()
 * @param inputs Input tensors (device type inferred from first input)
 * @param attrs Operation attributes
 * @return Output tensor
 */
auto dispatch_custom_op(CustomOpId id,
                        std::span<const Tensor> inputs,
                        const OpAttributes& attrs = OpAttributes{}) -> Tensor;

} // namespace tenzor
