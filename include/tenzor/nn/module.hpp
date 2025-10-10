/**
 * @file module.hpp
 * @brief Neural network module base class
 *
 * Provides Module base class for building neural networks with
 * parameter management, training/eval modes, and serialization.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "../autograd/variable.hpp"
#include "../core/device.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Base class for all neural network modules.
 *
 * Module provides the foundation for building neural networks with:
 * - Parameter and buffer management
 * - Training/evaluation mode switching
 * - Device management (CPU/GPU)
 * - Model serialization/deserialization
 * - Hierarchical module composition
 *
 * All neural network layers (Linear, Conv2d, etc.) inherit from Module.
 *
 * @code
 * class MyNetwork : public Module {
 * public:
 *     MyNetwork() {
 *         fc1 = std::make_shared<Linear>(784, 128);
 *         fc2 = std::make_shared<Linear>(128, 10);
 *         register_module("fc1", fc1);
 *         register_module("fc2", fc2);
 *     }
 *
 *     auto forward(const Variable& x) -> Variable override {
 *         auto h = fc1->forward(x).relu();
 *         return fc2->forward(h);
 *     }
 *
 * private:
 *     std::shared_ptr<Linear> fc1, fc2;
 * };
 * @endcode
 */
class Module {
public:
    virtual ~Module() = default;

    /**
     * @brief Forward pass computation (must be implemented by derived classes).
     *
     * @param input Input variable
     * @return Output variable
     */
    virtual auto forward(const Variable& input) -> Variable = 0;

    /**
     * @brief Convenience operator for forward pass.
     *
     * Allows calling module like a function: output = module(input)
     *
     * @param input Input variable
     * @return Output variable
     */
    auto operator()(const Variable& input) -> Variable {
        return forward(input);
    }

    // ============================================================================
    // Parameter Management
    // ============================================================================

    /**
     * @brief Get all parameters (weights, biases, etc.).
     *
     * Recursively collects parameters from this module and all submodules.
     *
     * @return Vector of pointers to all parameters
     *
     * @code
     * auto params = model.parameters();
     * for (auto* param : params) {
     *     param->zero_grad();
     * }
     * @endcode
     */
    virtual auto parameters() -> std::vector<Variable*>;

    /**
     * @brief Get all parameters with names.
     *
     * @return Vector of (name, parameter) pairs
     *
     * @code
     * for (auto& [name, param] : model.named_parameters()) {
     *     std::cout << name << ": " << param->shape() << "\n";
     * }
     * @endcode
     */
    virtual auto named_parameters() -> std::vector<std::pair<std::string, Variable*>>;

    /**
     * @brief Get all buffers (non-trainable tensors).
     *
     * Buffers are tensors that are part of module state but not optimized.
     * Examples: running statistics in BatchNorm.
     *
     * @return Vector of pointers to all buffers
     */
    auto buffers() -> std::vector<Variable*>;

    /**
     * @brief Get all buffers with names.
     *
     * @return Vector of (name, buffer) pairs
     */
    auto named_buffers() -> std::vector<std::pair<std::string, Variable*>>;

    // ============================================================================
    // Training Mode
    // ============================================================================

    /**
     * @brief Set module to training mode.
     *
     * Enables training-specific behaviors like dropout and batch normalization
     * statistics updates. Affects all submodules recursively.
     *
     * @param mode Training mode (default: true)
     *
     * @code
     * model.train();  // Enable training
     * @endcode
     */
    auto train(bool mode = true) -> void;

    /**
     * @brief Set module to evaluation mode.
     *
     * Disables training-specific behaviors. Equivalent to train(false).
     *
     * @code
     * model.eval();  // Disable training for inference
     * @endcode
     */
    auto eval() -> void;

    /**
     * @brief Check if module is in training mode.
     *
     * @return true if in training mode
     */
    auto is_training() const -> bool { return training_; }

    // ============================================================================
    // Device Management
    // ============================================================================

    /**
     * @brief Move module to specified device.
     *
     * Moves all parameters, buffers, and submodules to the device.
     *
     * @param device Target device
     *
     * @code
     * model.to(Device::cuda(0));  // Move to GPU 0
     * @endcode
     */
    auto to(Device device) -> void;

    /**
     * @brief Move module to CUDA device.
     *
     * @param device_id GPU device index (default: 0)
     */
    auto cuda(int device_id = 0) -> void;

    /**
     * @brief Move module to CPU.
     */
    auto cpu() -> void;

    // ============================================================================
    // Gradient Management
    // ============================================================================

    /**
     * @brief Zero all parameter gradients.
     *
     * Clears gradients for all parameters in this module and submodules.
     * Call before each backward pass.
     *
     * @code
     * model.zero_grad();
     * loss.backward();
     * optimizer.step();
     * @endcode
     */
    auto zero_grad() -> void;

    // ============================================================================
    // Serialization
    // ============================================================================

    /**
     * @brief Get module state as dictionary.
     *
     * Returns all parameters and buffers as a map for serialization.
     *
     * @return Map of (name -> tensor) for all state
     */
    virtual auto state_dict() const -> std::unordered_map<std::string, Tensor>;

    /**
     * @brief Load module state from dictionary.
     *
     * Restores parameters and buffers from a state dictionary.
     *
     * @param state Map of (name -> tensor) with saved state
     * @throws std::runtime_error if state doesn't match module structure
     */
    virtual auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void;

    /**
     * @brief Save module to file.
     *
     * @param path File path for saved model
     */
    auto save(const std::string& path) const -> void;

    /**
     * @brief Load module from file.
     *
     * @param path File path to load model from
     */
    auto load(const std::string& path) -> void;

protected:
    /**
     * @brief Register a parameter.
     *
     * Adds a trainable parameter to this module.
     *
     * @param name Parameter name
     * @param param Parameter variable
     */
    auto register_parameter(std::string name, Variable param) -> void;

    /**
     * @brief Register a buffer.
     *
     * Adds a non-trainable tensor to this module.
     *
     * @param name Buffer name
     * @param buffer Buffer variable
     */
    auto register_buffer(std::string name, Variable buffer) -> void;

    /**
     * @brief Register a submodule.
     *
     * Adds a child module to this module's hierarchy.
     *
     * @param name Submodule name
     * @param module Submodule to register
     */
    auto register_module(std::string name, std::shared_ptr<Module> module) -> void;

    bool training_{true};                                              ///< Training mode flag
    std::unordered_map<std::string, Variable> parameters_;             ///< Named parameters
    std::unordered_map<std::string, Variable> buffers_;                ///< Named buffers
    std::unordered_map<std::string, std::shared_ptr<Module>> submodules_;  ///< Named submodules
};

/**
 * @brief Sequential container for chaining modules.
 *
 * Sequential chains multiple modules together, passing the output of each
 * module as input to the next. Useful for building feed-forward networks.
 *
 * @code
 * auto model = std::make_shared<Sequential>(
 *     std::make_shared<Linear>(784, 128),
 *     std::make_shared<ReLU>(),
 *     std::make_shared<Linear>(128, 10)
 * );
 *
 * Variable output = model->forward(input);
 * @endcode
 */
class Sequential : public Module {
public:
    /**
     * @brief Default constructor.
     */
    Sequential() = default;

    /**
     * @brief Variadic constructor for multiple modules.
     *
     * @tparam Modules Module types
     * @param modules Modules to chain sequentially
     *
     * @code
     * Sequential net(
     *     std::make_shared<Linear>(10, 20),
     *     std::make_shared<ReLU>()
     * );
     * @endcode
     */
    template<typename... Modules>
    explicit Sequential(std::shared_ptr<Modules>... modules) {
        (add_module(modules), ...);
    }

    /**
     * @brief Add a module to the sequence.
     *
     * @param module Module to append
     * @return Reference to this Sequential for chaining
     *
     * @code
     * Sequential net;
     * net.add_module(std::make_shared<Linear>(10, 20))
     *    .add_module(std::make_shared<ReLU>());
     * @endcode
     */
    auto add_module(std::shared_ptr<Module> module) -> Sequential&;

    /**
     * @brief Forward pass through all modules sequentially.
     *
     * @param input Input variable
     * @return Output after passing through all modules
     */
    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Get all parameters preserving module order.
     *
     * @return Vector of parameter pointers
     */
    auto parameters() -> std::vector<Variable*> override;

    /**
     * @brief Get named parameters preserving module order.
     *
     * @return Vector of (name, parameter) pairs
     */
    auto named_parameters() -> std::vector<std::pair<std::string, Variable*>> override;

    /**
     * @brief Get state dictionary.
     *
     * @return Map of parameter/buffer names to tensors
     */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /**
     * @brief Load state from dictionary.
     *
     * @param state State dictionary to load
     */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    std::vector<std::shared_ptr<Module>> modules_;  ///< Ordered list of modules
};

} // namespace nn
} // namespace tenzor
