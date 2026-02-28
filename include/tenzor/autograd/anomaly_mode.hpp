/**
 * @file anomaly_mode.hpp
 * @brief Runtime anomaly detection for automatic differentiation
 *
 * Provides AnomalyMode RAII guard that enables NaN/Inf checking
 * during backward passes. When enabled, the backward engine checks
 * each computed gradient for anomalous values and throws a descriptive
 * error identifying the responsible autograd function.
 *
 * Zero overhead when disabled (thread-local bool check only).
 *
 * @code
 * {
 *     AnomalyMode guard;  // Enable anomaly detection
 *     auto loss = model.forward(input);
 *     loss.backward();    // Throws if any gradient is NaN/Inf
 * }
 * // Anomaly detection disabled again here
 * @endcode
 */

#pragma once

namespace tenzor {

/// Check if anomaly detection mode is active (thread-local).
auto is_anomaly_detection_enabled() -> bool;

/// Set anomaly detection mode (thread-local).
auto set_anomaly_detection(bool enabled) -> void;

/**
 * @brief RAII guard for scoped anomaly detection.
 *
 * Enables NaN/Inf checking in the backward engine for the
 * lifetime of this guard. Restores previous state on destruction.
 *
 * Follows the same pattern as NoGradGuard and CreateGraphGuard.
 */
class AnomalyMode {
public:
    /// Enable anomaly detection for this scope.
    explicit AnomalyMode(bool enabled = true);

    /// Restore previous anomaly detection state.
    ~AnomalyMode();

    AnomalyMode(const AnomalyMode&) = delete;
    AnomalyMode& operator=(const AnomalyMode&) = delete;

private:
    bool prev_state_;
};

} // namespace tenzor
