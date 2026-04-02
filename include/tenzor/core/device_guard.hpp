/**
 * @file device_guard.hpp
 * @brief RAII device context management for multi-GPU programming
 *
 * Provides DeviceGuard and OptionalDeviceGuard for automatic device
 * switching and restoration, eliminating manual cudaSetDevice/hipSetDevice
 * save-restore patterns.
 *
 * @code
 * {
 *     DeviceGuard guard(Device::cuda(1));
 *     // All operations here execute on cuda:1
 *     auto tensor = Tensor({256, 256}, DType::Float32, Device::cuda(1));
 * }
 * // Previous device automatically restored
 * @endcode
 *
 * Thread safety:
 * - Device state is thread-local; guards in different threads are independent.
 * - RAII nesting provides implicit stack semantics via saved prev_index_.
 */

#pragma once

#include <optional>
#include "device.hpp"

namespace tenzor {

// Forward declaration
class Backend;

namespace detail {

/**
 * @brief Get/set the thread-local current device index for a backend type.
 *
 * Each thread maintains its own device index per backend type, initialized to 0.
 * This is the source of truth for "which device is active" in a given thread.
 *
 * @param type Backend type (CPU, CUDA, ROCm, OneAPI, Vulkan)
 * @return Mutable reference to the thread-local device index
 */
auto current_device_index(Device::Type type) -> int32_t&;

/**
 * @brief Switch the active device for a backend type.
 *
 * Delegates to the backend's set_device() method and updates thread-local state.
 * No-op if the requested device is already active or type is CPU.
 *
 * @param type Backend type
 * @param device_id Target device index
 */
void switch_device(Device::Type type, int32_t device_id);

} // namespace detail

/**
 * @brief RAII guard that sets a device on construction and restores the previous device on destruction.
 *
 * Replaces the error-prone manual pattern of:
 * @code
 * int prev; cudaGetDevice(&prev);
 * cudaSetDevice(target);
 * // ... work ...
 * cudaSetDevice(prev);  // Easy to forget on exceptions!
 * @endcode
 *
 * With the safe:
 * @code
 * {
 *     DeviceGuard guard(Device::cuda(target));
 *     // ... work ...
 * }  // Automatically restores previous device, even on exception
 * @endcode
 *
 * @note CPU devices are always index 0; DeviceGuard for CPU is a no-op.
 * @note Nesting DeviceGuards works correctly — each saves/restores independently.
 */
class DeviceGuard {
public:
    /**
     * @brief Construct guard and switch to the specified device.
     *
     * @param device Target device to switch to
     */
    explicit DeviceGuard(Device device);

    /**
     * @brief Destroy guard and restore previous device.
     */
    ~DeviceGuard();

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
    DeviceGuard(DeviceGuard&&) = delete;
    DeviceGuard& operator=(DeviceGuard&&) = delete;

    /**
     * @brief Get the device this guard switched to.
     */
    auto device() const -> Device { return device_; }

    /**
     * @brief Get the device index that was active before this guard.
     */
    auto previous_device_index() const -> int32_t { return prev_index_; }

private:
    Device device_;
    int32_t prev_index_;
};

/**
 * @brief Optional RAII device guard that is a no-op when device is nullopt or CPU.
 *
 * Useful when a function accepts an optional device parameter:
 * @code
 * void my_function(std::optional<Device> device = std::nullopt) {
 *     OptionalDeviceGuard guard(device);
 *     // If device was provided and is a GPU, we're now on that device
 * }
 * @endcode
 */
class OptionalDeviceGuard {
public:
    /**
     * @brief Construct from optional device.
     *
     * If device is nullopt or CPU, no device switch occurs.
     *
     * @param device Optional target device
     */
    explicit OptionalDeviceGuard(std::optional<Device> device = std::nullopt);

    /**
     * @brief Construct from device, treating CPU as no-op.
     *
     * @param device Target device (no-op if CPU)
     */
    explicit OptionalDeviceGuard(Device device);

    ~OptionalDeviceGuard() = default;

    OptionalDeviceGuard(const OptionalDeviceGuard&) = delete;
    OptionalDeviceGuard& operator=(const OptionalDeviceGuard&) = delete;

    /**
     * @brief Check if this guard is actively managing a device switch.
     */
    auto has_value() const -> bool { return guard_.has_value(); }

private:
    std::optional<DeviceGuard> guard_;
};

/**
 * @brief Get the current device for a backend type in this thread.
 *
 * @param type Backend type
 * @return Currently active device for the given backend type
 */
inline auto current_device(Device::Type type) -> Device {
    return Device{type, detail::current_device_index(type)};
}

} // namespace tenzor
