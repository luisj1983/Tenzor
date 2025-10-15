/**
 * @file autocast.hpp
 * @brief Automatic mixed precision (AMP) context manager
 *
 * Provides a context manager for automatic casting of operations to lower precision
 * (Float16 or BFloat16) for performance while maintaining numerical stability.
 */

#pragma once

#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include <atomic>
#include <optional>

namespace tenzor {
namespace nn {
namespace amp {

/**
 * @brief Autocast mode for automatic mixed precision
 */
class Autocast {
public:
    /**
     * @brief Constructor for Autocast context
     * @param enabled Whether autocast is enabled
     * @param dtype Target dtype for autocasting (Float16 or BFloat16)
     * @param device_type Device type to apply autocast (CUDA only by default)
     */
    explicit Autocast(
        bool enabled = true,
        DType dtype = DType::Float16,
        Device::Type device_type = Device::Type::CUDA
    );

    /**
     * @brief Destructor - restores previous autocast state
     */
    ~Autocast();

    // Delete copy constructor and assignment
    Autocast(const Autocast&) = delete;
    auto operator=(const Autocast&) -> Autocast& = delete;

    /**
     * @brief Check if autocast is currently enabled
     */
    static auto is_enabled() -> bool;

    /**
     * @brief Get the current autocast dtype
     */
    static auto get_dtype() -> std::optional<DType>;

    /**
     * @brief Get the current autocast device type
     */
    static auto get_device_type() -> std::optional<Device::Type>;

    /**
     * @brief Determine if a given operation should be autocast
     * @param op_name Operation name
     * @param device Device of operation
     */
    static auto should_autocast(const std::string& op_name, const Device& device) -> bool;

    /**
     * @brief Get the target dtype for a given operation
     * @param op_name Operation name
     * @param input_dtype Input dtype
     */
    static auto get_autocast_dtype(const std::string& op_name, DType input_dtype) -> DType;

private:
    bool prev_enabled_;
    std::optional<DType> prev_dtype_;
    std::optional<Device::Type> prev_device_type_;

    // Thread-local state
    static thread_local bool enabled_;
    static thread_local std::optional<DType> dtype_;
    static thread_local std::optional<Device::Type> device_type_;

    /**
     * @brief Check if operation should use lower precision
     * Operations like matmul, conv2d benefit from lower precision
     */
    static auto is_compute_heavy_op(const std::string& op_name) -> bool;

    /**
     * @brief Check if operation should stay in higher precision
     * Operations like softmax, layer_norm need higher precision for stability
     */
    static auto is_stability_critical_op(const std::string& op_name) -> bool;
};

/**
 * @brief RAII helper for autocast context
 *
 * Usage:
 * @code
 * {
 *     AutocastGuard guard(true, DType::Float16);
 *     auto output = model.forward(input);  // Operations auto-cast to FP16
 * }
 * // Autocast disabled after scope
 * @endcode
 */
class AutocastGuard {
public:
    explicit AutocastGuard(bool enabled, DType dtype = DType::Float16)
        : autocast_(enabled, dtype) {}

private:
    Autocast autocast_;
};

/**
 * @brief Decorator for disabling autocast in a scope
 *
 * Usage:
 * @code
 * {
 *     AutocastDisabled guard;
 *     auto loss = criterion(output, target);  // Computed in FP32
 * }
 * @endcode
 */
class AutocastDisabled {
public:
    AutocastDisabled() : autocast_(false) {}

private:
    Autocast autocast_;
};

/**
 * @brief Helper function to create autocast context
 */
inline auto autocast(
    bool enabled = true,
    DType dtype = DType::Float16,
    Device::Type device_type = Device::Type::CUDA
) -> Autocast {
    return Autocast(enabled, dtype, device_type);
}

} // namespace amp
} // namespace nn
} // namespace tenzor
