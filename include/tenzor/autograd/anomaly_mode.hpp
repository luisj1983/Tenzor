/**
 * @file anomaly_mode.hpp
 * @brief Runtime anomaly detection for automatic differentiation
 *
 * Provides AnomalyMode RAII guard that enables NaN/Inf checking
 * during backward passes. When enabled, the backward engine checks
 * each computed gradient for anomalous values and throws a descriptive
 * error identifying the responsible autograd function.
 *
 * When anomaly mode is on, forward-pass operations also record creation
 * metadata on output Variables, providing a traceback to the forward op
 * that produced the tensor when a backward anomaly is detected.
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

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace tenzor {

/// Check if anomaly detection mode is active (thread-local).
auto is_anomaly_detection_enabled() -> bool;

/// Set anomaly detection mode (thread-local).
auto set_anomaly_detection(bool enabled) -> void;

/**
 * @brief Metadata recorded during forward pass when anomaly detection is on.
 *
 * Captures the operation name, input/output shapes at the time of the forward
 * pass. Attached to output Variables so the backward engine can include this
 * information in anomaly error messages.
 *
 * Zero overhead when anomaly detection is disabled (no metadata allocated).
 */
struct AnomalyMetadata {
    /// Name of the autograd function that created this tensor
    std::string function_name;

    /// Shapes of input tensors at forward time
    std::vector<std::vector<int64_t>> input_shapes;

    /// Shapes of output tensors at forward time
    std::vector<std::vector<int64_t>> output_shapes;

    /// Parent metadata (for chained creation tracebacks)
    std::shared_ptr<AnomalyMetadata> parent;

    /// Format as human-readable traceback string
    auto to_string() const -> std::string {
        std::string result;
        result += "  Created by: " + function_name;
        auto append_shapes = [&result](const char* label,
                                        const std::vector<std::vector<int64_t>>& shapes) {
            if (shapes.empty()) return;
            result += label;
            for (size_t i = 0; i < shapes.size(); ++i) {
                if (i > 0) result += ", ";
                result += "[";
                for (size_t j = 0; j < shapes[i].size(); ++j) {
                    if (j > 0) result += ", ";
                    result += std::to_string(shapes[i][j]);
                }
                result += "]";
            }
        };
        append_shapes("\n    Inputs: ", input_shapes);
        append_shapes("\n    Outputs: ", output_shapes);
        if (parent) {
            result += "\n" + parent->to_string();
        }
        return result;
    }
};

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
