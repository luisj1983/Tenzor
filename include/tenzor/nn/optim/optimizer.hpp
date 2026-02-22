/**
 * @file optimizer.hpp
 * @brief Base class for all optimizers
 *
 * Provides the foundation for gradient-based optimization algorithms.
 * Optimizers update model parameters based on computed gradients to minimize loss.
 */

#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include "../../autograd/variable.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief Parameter group with individual learning rate
 *
 * Allows different learning rates and weight decay for different parameter groups.
 * Useful for transfer learning or fine-tuning specific layers.
 *
 * @code
 * // Different learning rates for different layers
 * ParamGroup group1{backbone_params, 0.001, 0.0001};
 * ParamGroup group2{head_params, 0.01, 0.0001};
 * @endcode
 */
struct ParamGroup {
    std::vector<std::shared_ptr<Variable>> params;  ///< Parameters in this group
    double lr;                      ///< Learning rate for this group
    double weight_decay{0.0};       ///< Weight decay (L2 regularization) for this group
};

/**
 * @brief Gradient clipping mode for optimizer integration.
 */
enum class ClipMode {
    None,   ///< No gradient clipping
    Norm,   ///< Clip by global norm (L2)
    Value   ///< Clip by value (element-wise clamping)
};

/**
 * @brief Configuration for automatic gradient clipping in optimizers.
 *
 * When set on an optimizer, gradients are clipped automatically
 * before each parameter update in step().
 *
 * @code
 * // Clip gradients by norm (max_norm = 1.0)
 * ClipConfig clip{ClipMode::Norm, 1.0};
 * optimizer.set_clip_config(clip);
 *
 * // Clip gradients by value (clamp to [-0.5, 0.5])
 * ClipConfig clip{ClipMode::Value, 0.5};
 * optimizer.set_clip_config(clip);
 *
 * // Disable clipping
 * optimizer.set_clip_config(ClipConfig{});
 * @endcode
 */
struct ClipConfig {
    ClipMode mode{ClipMode::None};  ///< Clipping mode
    double max_norm{1.0};           ///< Maximum norm for Norm mode, or max absolute value for Value mode
    double norm_type{2.0};          ///< Norm type for Norm mode (default: L2)
};

/**
 * @brief Abstract base class for all optimizers
 *
 * Optimizers perform gradient-based parameter updates during training.
 * Common workflow:
 * 1. Forward pass: Compute predictions
 * 2. Loss calculation: Compare predictions to targets
 * 3. Backward pass: Compute gradients via loss.backward()
 * 4. optimizer.step(): Clip gradients (if configured) then update parameters
 * 5. optimizer.zero_grad(): Clear gradients for next iteration
 *
 * **Gradient Clipping:**
 * - Configure via set_clip_config() to automatically clip gradients before updates
 * - Supports clipping by global norm (ClipMode::Norm) or by value (ClipMode::Value)
 * - step() calls clip_gradients_() then step_impl() internally
 *
 * **State Management:**
 * - Maintains references to model parameters
 * - Stores optimizer state (momentum buffers, etc.)
 * - Supports serialization via state_dict()
 *
 * **Derived Classes:**
 * - SGD: Stochastic Gradient Descent with momentum
 * - Adam: Adaptive Moment Estimation
 * - AdamW: Adam with decoupled weight decay
 *
 * @par Thread Safety
 * Not thread-safe. Use separate optimizer instances for parallel training.
 *
 * @code
 * // Typical training loop with gradient clipping
 * auto optimizer = SGD(model.parameters(), 0.01);
 * optimizer.set_clip_config({ClipMode::Norm, 1.0});  // Clip grad norm to 1.0
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     optimizer.zero_grad();           // Clear previous gradients
 *     auto output = model.forward(input);
 *     auto loss = criterion(output, targets);
 *     loss.backward();                 // Compute gradients
 *     optimizer.step();                // Clips gradients, then updates parameters
 * }
 * @endcode
 *
 * @see SGD, Adam, AdamW
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;

    /**
     * @brief Perform single optimization step with optional gradient clipping.
     *
     * Applies gradient clipping (if configured via set_clip_config()),
     * then delegates to step_impl() for the actual parameter update.
     *
     * @pre Gradients must be computed via backward()
     * @post Parameters are updated according to optimizer's algorithm
     */
    auto step() -> void;

    /**
     * @brief Implementation of the parameter update step.
     *
     * Must be implemented by derived classes to perform the actual optimization
     * algorithm (SGD, Adam, etc.). Called by step() after gradient clipping.
     */
    virtual auto step_impl() -> void = 0;

    /**
     * @brief Zero out all parameter gradients
     *
     * Clears gradients from previous iteration. Must be called before each backward pass
     * to prevent gradient accumulation.
     *
     * @par Complexity
     * O(P) where P is the total number of parameters
     *
     * @code
     * optimizer.zero_grad();  // Clear old gradients
     * loss.backward();        // Compute new gradients
     * optimizer.step();       // Update parameters
     * @endcode
     */
    auto zero_grad() -> void;

    /**
     * @brief Get list of parameters being optimized
     * @return Const reference to parameter vector
     */
    auto parameters() const -> const std::vector<std::shared_ptr<Variable>>&;

    /**
     * @brief Get optimizer state as dictionary
     *
     * Returns internal optimizer state (momentum buffers, learning rates, etc.)
     * for serialization. Must be implemented by derived classes.
     *
     * @return Map of state variable names to tensors
     * @see load_state_dict()
     */
    virtual auto state_dict() const -> std::unordered_map<std::string, Tensor> = 0;

    /**
     * @brief Load optimizer state from dictionary
     *
     * Restores optimizer state from previously saved state_dict().
     * Used for checkpoint restoration.
     *
     * @param state State dictionary to load
     * @see state_dict()
     */
    virtual auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void = 0;

    /**
     * @brief Save optimizer state to file
     * @param path File path to save to
     */
    auto save_state(const std::string& path) const -> void;

    /**
     * @brief Load optimizer state from file
     * @param path File path to load from
     */
    auto load_state(const std::string& path) -> void;

    /**
     * @brief Add a parameter group with custom hyperparameters.
     *
     * @param group Parameter group to add
     */
    auto add_param_group(ParamGroup group) -> void;

    /**
     * @brief Get all parameter groups.
     * @return Reference to the vector of parameter groups
     */
    auto param_groups() -> std::vector<ParamGroup>&;

    /**
     * @brief Get all parameter groups (const).
     * @return Const reference to the vector of parameter groups
     */
    auto param_groups() const -> const std::vector<ParamGroup>&;

    /**
     * @brief Set gradient clipping configuration.
     *
     * When configured, gradients are automatically clipped before each
     * parameter update in step().
     *
     * @param config Clipping configuration (default-constructed disables clipping)
     */
    auto set_clip_config(const ClipConfig& config) -> void;

    /**
     * @brief Get current gradient clipping configuration.
     * @return Current clip configuration
     */
    auto clip_config() const -> const ClipConfig&;

    /**
     * @brief Set learning rate on the optimizer.
     *
     * Base implementation throws. Derived classes should override.
     * Used by schedulers to adjust the learning rate.
     *
     * @param lr New learning rate
     */
    virtual auto set_lr(double lr) -> void;

    /**
     * @brief Get current learning rate from the optimizer.
     *
     * Base implementation throws. Derived classes should override.
     *
     * @return Current learning rate
     */
    virtual auto get_lr() const -> double;

protected:
    /**
     * @brief Construct optimizer with parameters to optimize
     * @param params Vector of shared pointers to model parameters
     */
    explicit Optimizer(std::vector<std::shared_ptr<Variable>> params);

    /**
     * @brief Construct optimizer with parameter groups
     * @param groups Vector of parameter groups with individual hyperparameters
     */
    explicit Optimizer(std::vector<ParamGroup> groups);

    /**
     * @brief Apply gradient clipping based on the current clip configuration.
     *
     * Called internally by step() before step_impl(). Does nothing if
     * clip_config_.mode is ClipMode::None.
     */
    auto clip_gradients_() -> void;

    std::vector<std::shared_ptr<Variable>> parameters_;  ///< All parameters (flattened from groups)
    std::vector<ParamGroup> param_groups_;  ///< Parameter groups with individual hyperparams
    ClipConfig clip_config_;  ///< Gradient clipping configuration
};

} // namespace optim
} // namespace tenzor
