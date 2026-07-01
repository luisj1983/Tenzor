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
#include <shared_mutex>
#include <string>
#include <unordered_set>

namespace tenzor {
namespace nn {
namespace amp {

/**
 * @brief Registry for autocast op categorization policies.
 *
 * Thread-safe singleton that classifies operations as compute-heavy
 * (cast to lower precision) or stability-critical (keep in FP32).
 * Initialized with default lists matching PyTorch's autocast behavior.
 * Users can register/unregister ops to customize autocast policies.
 *
 * @code
 * auto& registry = AutocastPolicyRegistry::instance();
 * registry.register_compute_heavy("my_custom_matmul");
 * registry.unregister("sum");  // Remove from stability-critical
 * @endcode
 */
class AutocastPolicyRegistry {
public:
    static auto instance() -> AutocastPolicyRegistry&;

    /// Register an op as compute-heavy (will be cast to lower precision)
    auto register_compute_heavy(const std::string& op_name) -> void;

    /// Register an op as stability-critical (will be kept in FP32)
    auto register_stability_critical(const std::string& op_name) -> void;

    /// Remove an op from both compute-heavy and stability-critical sets
    auto unregister(const std::string& op_name) -> void;

    /// Check if op is compute-heavy
    auto is_compute_heavy(const std::string& op_name) const -> bool;

    /// Check if op is stability-critical
    auto is_stability_critical(const std::string& op_name) const -> bool;

    /// Lock-free fast path: true iff any custom override has been registered.
    /// Lets the dispatch hot path skip the per-op string construction and the
    /// two locked set lookups when no policy override exists (the common case).
    auto has_overrides() const noexcept -> bool {
        return override_count_.load(std::memory_order_acquire) > 0;
    }

private:
    AutocastPolicyRegistry();

    std::unordered_set<std::string> compute_heavy_ops_;
    std::unordered_set<std::string> stability_critical_ops_;
    std::atomic<size_t> override_count_{0};
    mutable std::shared_mutex mutex_;
};

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
    bool pushed_interceptor_ = false;  ///< Whether we pushed an autocast interceptor

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
