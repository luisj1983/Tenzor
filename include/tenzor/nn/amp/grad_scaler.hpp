/**
 * @file grad_scaler.hpp
 * @brief Automatic Mixed Precision (AMP) gradient scaler for loss scaling
 *
 * Provides GradScaler class for gradient scaling to prevent underflow in FP16 training.
 * Implements dynamic loss scaling with automatic adjustment based on gradient overflow detection.
 */

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include "../../autograd/variable.hpp"
#include "../optim/optimizer.hpp"
#include "../utils/clip_grad.hpp"

namespace tenzor {
namespace nn {
namespace amp {

/**
 * @brief Gradient scaler for automatic mixed precision training
 *
 * GradScaler helps prevent gradient underflow when training with FP16/mixed precision
 * by scaling the loss before backward() and unscaling gradients before optimizer.step().
 *
 * **Algorithm:**
 * 1. Scale loss by a large factor before backward()
 * 2. Compute scaled gradients via backward()
 * 3. Unscale gradients before optimizer update
 * 4. Check for inf/nan in gradients
 * 5. Skip optimizer step if overflow detected
 * 6. Dynamically adjust scale factor based on overflow history
 *
 * **Dynamic Scaling:**
 * - Scale increases by growth_factor every growth_interval successful iterations
 * - Scale decreases by backoff_factor when overflow detected
 * - Prevents gradients from becoming too large or too small
 *
 * **Typical Usage:**
 * @code
 * GradScaler scaler(65536.0f, 2.0f, 0.5f, 2000);
 *
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     optimizer.zero_grad();
 *
 *     auto output = model.forward(input);
 *     auto loss = criterion(output, target);
 *
 *     // Scale loss and compute gradients
 *     auto scaled_loss = scaler.scale(loss);
 *     scaled_loss.backward();
 *
 *     // Unscale gradients and update if no overflow
 *     scaler.step(optimizer);
 *
 *     // Update scale factor for next iteration
 *     scaler.update();
 * }
 * @endcode
 *
 * **Parameters:**
 * - init_scale: Initial loss scale (default: 65536.0 = 2^16)
 * - growth_factor: Scale multiplier on success (default: 2.0)
 * - backoff_factor: Scale multiplier on overflow (default: 0.5)
 * - growth_interval: Iterations before growing scale (default: 2000)
 *
 * @see https://pytorch.org/docs/stable/amp.html#gradient-scaling
 */
class GradScaler {
public:
    /**
     * @brief Construct gradient scaler with specified parameters
     *
     * @param init_scale Initial loss scale factor (default: 65536.0)
     * @param growth_factor Scale multiplier for growth (default: 2.0)
     * @param backoff_factor Scale multiplier for backoff (default: 0.5)
     * @param growth_interval Steps before attempting scale growth (default: 2000)
     *
     * @code
     * // Default configuration
     * GradScaler scaler;
     *
     * // Conservative scaling
     * GradScaler conservative(1024.0f, 1.5f, 0.75f, 5000);
     *
     * // Aggressive scaling
     * GradScaler aggressive(131072.0f, 2.5f, 0.25f, 1000);
     * @endcode
     */
    explicit GradScaler(float init_scale = 65536.0f,
                       float growth_factor = 2.0f,
                       float backoff_factor = 0.5f,
                       int growth_interval = 2000);

    /**
     * @brief Scale loss by current scale factor
     *
     * Multiplies loss by the current scale factor before backward pass.
     * This prevents gradient underflow in FP16 training.
     *
     * @param loss Unscaled loss variable
     * @return Scaled loss variable
     *
     * @par Complexity
     * O(1) - Simple scalar multiplication
     *
     * @code
     * auto loss = criterion(output, target);
     * auto scaled_loss = scaler.scale(loss);
     * scaled_loss.backward();  // Compute scaled gradients
     * @endcode
     */
    auto scale(const Variable& loss) -> Variable;

    /**
     * @brief Unscale gradients in optimizer parameters
     *
     * Divides all gradients by the current scale factor. Must be called
     * after backward() and before checking for inf/nan or updating parameters.
     *
     * @param optimizer Optimizer containing parameters with gradients
     *
     * @par Complexity
     * O(P) where P is total number of parameters
     *
     * @code
     * scaled_loss.backward();
     * scaler.unscale_(optimizer);  // Restore true gradient magnitudes
     *
     * // Now safe to clip gradients or check norms
     * clip_grad_norm_(model.parameters(), max_norm);
     * @endcode
     *
     * @warning This method modifies gradients in-place
     * @note The trailing underscore follows PyTorch convention for in-place operations
     */
    auto unscale_(optim::Optimizer& optimizer) -> void;

    /**
     * @brief Execute optimizer step with overflow detection
     *
     * Unscales gradients, checks for inf/nan, and conditionally updates parameters.
     * If overflow detected, skips the update and marks scale for adjustment.
     *
     * @param optimizer Optimizer to step
     * @return true if update performed, false if skipped due to overflow
     *
     * @par Complexity
     * O(P) where P is total number of parameters
     *
     * @code
     * scaled_loss.backward();
     *
     * if (scaler.step(optimizer)) {
     *     // Successful update
     *     std::cout << "Step performed\n";
     * } else {
     *     // Overflow detected, step skipped
     *     std::cout << "Step skipped due to overflow\n";
     * }
     * @endcode
     *
     * @note Automatically calls unscale_() if not already called
     */
    auto step(optim::Optimizer& optimizer) -> bool;

    /**
     * @brief Update scale factor based on overflow history
     *
     * Adjusts the scale factor for the next iteration:
     * - If overflow detected: multiply by backoff_factor (decrease)
     * - If growth_interval steps without overflow: multiply by growth_factor (increase)
     *
     * Should be called once per training iteration after step().
     *
     * @par Complexity
     * O(1)
     *
     * @code
     * scaler.step(optimizer);
     * scaler.update();  // Adjust scale for next iteration
     * @endcode
     *
     * @note This does not affect the current iteration, only future ones
     */
    auto update() -> void;

    /**
     * @brief Get current scale factor
     *
     * @return Current loss scale value
     *
     * @code
     * std::cout << "Current scale: " << scaler.get_scale() << "\n";
     * @endcode
     */
    auto get_scale() const -> float;

    /**
     * @brief Get number of consecutive successful iterations
     *
     * @return Count of iterations since last overflow
     *
     * @code
     * if (scaler.get_growth_tracker() >= 2000) {
     *     std::cout << "Scale will grow on next update\n";
     * }
     * @endcode
     */
    auto get_growth_tracker() const -> int;

    /**
     * @brief Check if overflow was detected in last step
     *
     * @return true if inf/nan detected in last step()
     */
    auto found_inf_nan() const -> bool;

    /**
     * @brief Reset scaler to initial state
     *
     * Resets scale factor to init_scale and clears overflow history.
     * Useful when changing training regime.
     *
     * @code
     * scaler.reset();  // Start fresh after warmup phase
     * @endcode
     */
    auto reset() -> void;

    /**
     * @brief Get scaler state for serialization
     *
     * @return Dictionary containing scale, growth_tracker, and parameters
     */
    auto state_dict() const -> std::unordered_map<std::string, float>;

    /**
     * @brief Load scaler state from dictionary
     *
     * @param state State dictionary to load
     */
    auto load_state_dict(const std::unordered_map<std::string, float>& state) -> void;

    /**
     * @brief Clip gradients by global norm after unscaling
     *
     * Convenience method that unscales gradients (if not already done) and then
     * clips them by global norm. Equivalent to calling unscale_() followed by
     * nn::utils::clip_grad_norm_().
     *
     * @param optimizer Optimizer containing parameters with gradients
     * @param max_norm Maximum allowed total norm
     * @param norm_type Type of the p-norm (default: 2.0 for L2 norm)
     * @return The total norm of the gradients before clipping
     *
     * @code
     * scaled_loss.backward();
     * double total_norm = scaler.clip_grad_norm_(optimizer, 1.0);
     * scaler.step(optimizer);
     * scaler.update();
     * @endcode
     */
    auto clip_grad_norm_(optim::Optimizer& optimizer, double max_norm, double norm_type = 2.0) -> double;

    /**
     * @brief Clip gradients by value after unscaling
     *
     * Convenience method that unscales gradients (if not already done) and then
     * clamps each gradient element to [-clip_value, clip_value]. Equivalent to
     * calling unscale_() followed by nn::utils::clip_grad_value_().
     *
     * @param optimizer Optimizer containing parameters with gradients
     * @param clip_value Maximum absolute value for gradient elements
     *
     * @code
     * scaled_loss.backward();
     * scaler.clip_grad_value_(optimizer, 0.5);
     * scaler.step(optimizer);
     * scaler.update();
     * @endcode
     */
    auto clip_grad_value_(optim::Optimizer& optimizer, double clip_value) -> void;

private:
    float scale_;                 ///< Current loss scale factor
    float init_scale_;            ///< Initial loss scale factor (for reset)
    float growth_factor_;         ///< Multiplier for scale growth
    float backoff_factor_;        ///< Multiplier for scale backoff
    int growth_interval_;         ///< Steps before attempting growth
    int growth_tracker_;          ///< Iterations since last overflow
    bool found_inf_nan_;          ///< Whether overflow detected in last step
    bool has_unscaled_;           ///< Whether gradients have been unscaled

    /// Y.32: side-table holding the F32 unscaled gradient for params whose
    /// stored grad dtype is F16/BF16. ``unscale_`` populates this with the
    /// pre-cast-back F32 representation; ``check_inf_nan_`` reads from here
    /// so overflow detection sees the value that fits in F32, not the
    /// potentially-saturated half-precision cast-back stored on the
    /// Variable. Cleared at ``step()`` time after the check.
    std::unordered_map<Variable*, Tensor> f32_unscaled_grads_;

    /**
     * @brief Check if any parameter gradients contain inf or nan
     *
     * @param optimizer Optimizer containing parameters to check
     * @return true if inf or nan found in any gradient
     */
    auto check_inf_nan_(const optim::Optimizer& optimizer) const -> bool;
};

} // namespace amp
} // namespace nn
} // namespace tenzor
