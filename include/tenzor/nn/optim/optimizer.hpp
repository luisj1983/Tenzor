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
 * @brief Abstract base class for all optimizers
 *
 * Optimizers perform gradient-based parameter updates during training.
 * Common workflow:
 * 1. Forward pass: Compute predictions
 * 2. Loss calculation: Compare predictions to targets
 * 3. Backward pass: Compute gradients via loss.backward()
 * 4. optimizer.step(): Update parameters based on gradients
 * 5. optimizer.zero_grad(): Clear gradients for next iteration
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
 * // Typical training loop
 * auto optimizer = SGD(model.parameters(), 0.01);
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     optimizer.zero_grad();           // Clear previous gradients
 *     auto output = model.forward(input);
 *     auto loss = criterion(output, targets);
 *     loss.backward();                 // Compute gradients
 *     optimizer.step();                // Update parameters
 * }
 * @endcode
 *
 * @see SGD, Adam, AdamW
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;

    /**
     * @brief Perform single optimization step (parameter update)
     *
     * Updates all parameters based on their gradients. Must be implemented by derived classes.
     * Called after loss.backward() has computed gradients.
     *
     * @pre Gradients must be computed via backward()
     * @post Parameters are updated according to optimizer's algorithm
     */
    virtual auto step() -> void = 0;

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

protected:
    /**
     * @brief Construct optimizer with parameters to optimize
     * @param params Vector of shared pointers to model parameters
     */
    explicit Optimizer(std::vector<std::shared_ptr<Variable>> params);

    std::vector<std::shared_ptr<Variable>> parameters_;  ///< Parameters being optimized
};

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

} // namespace optim
} // namespace tenzor
