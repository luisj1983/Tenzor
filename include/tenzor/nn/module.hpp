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
     * @brief Forward pass computation.
     *
     * This method automatically calls forward pre-hooks and post-hooks
     * if any are registered on this module. Derived classes should override
     * forward_impl() to provide their computation logic.
     *
     * @param input Input variable
     * @return Output variable
     */
    auto forward(const Variable& input) -> Variable {
        // Only call hooks if this specific module has hooks registered
        // This avoids overhead for modules without hooks
        if (has_forward_hooks_) {
            call_own_forward_pre_hooks();
            auto output = forward_impl(input);
            call_own_forward_post_hooks();
            return output;
        }
        return forward_impl(input);
    }

    /**
     * @brief Implementation of forward pass (must be implemented by derived classes).
     *
     * @param input Input variable
     * @return Output variable
     */
    virtual auto forward_impl(const Variable& input) -> Variable = 0;

    /**
     * @brief Convenience operator for forward pass.
     *
     * Equivalent to calling forward() - provided for functional syntax.
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
     * @return Vector of shared pointers to all parameters
     *
     * @code
     * auto params = model.parameters();
     * for (auto& param : params) {
     *     param->zero_grad();
     * }
     * @endcode
     */
    virtual auto parameters() -> std::vector<std::shared_ptr<Variable>>;

    /**
     * @brief Get only this module's direct parameters (not submodules').
     *
     * Used by offload hooks to load/offload only the current layer's parameters.
     *
     * @return Vector of shared pointers to this module's own parameters only
     */
    virtual auto own_parameters() -> std::vector<std::shared_ptr<Variable>>;

    /**
     * @brief Get all parameters with names.
     *
     * @return Vector of (name, shared_ptr to parameter) pairs
     *
     * @code
     * for (auto& [name, param] : model.named_parameters()) {
     *     std::cout << name << ": " << param->shape() << "\n";
     * }
     * @endcode
     */
    virtual auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>>;

    /**
     * @brief Get all buffers (non-trainable tensors).
     *
     * Buffers are tensors that are part of module state but not optimized.
     * Examples: running statistics in BatchNorm.
     *
     * @return Vector of shared pointers to all buffers
     */
    auto buffers() -> std::vector<std::shared_ptr<Variable>>;

    /**
     * @brief Get this module's own buffers (not submodules').
     *
     * @return Vector of shared pointers to this module's direct buffers only
     */
    auto own_buffers() -> std::vector<std::shared_ptr<Variable>>;

    /**
     * @brief Get all buffers with names.
     *
     * @return Vector of (name, shared_ptr to buffer) pairs
     */
    auto named_buffers() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>>;

    /**
     * @brief Get all direct submodules.
     *
     * @return Map of (name -> submodule) pairs
     */
    auto get_submodules() const -> const std::unordered_map<std::string, std::shared_ptr<Module>>& {
        return submodules_;
    }

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
    // Hook System (for Phase 2 Offload Support)
    // ============================================================================

    /**
     * @brief Hook function types for forward and backward passes.
     */
    using ForwardPreHook = std::function<void(Module*)>;
    using ForwardPostHook = std::function<void(Module*)>;
    using BackwardPreHook = std::function<void(Module*)>;
    using BackwardPostHook = std::function<void(Module*)>;

    /**
     * @brief Register a forward pre-hook.
     *
     * Hook is called before forward() executes. Useful for parameter prefetching.
     *
     * @param hook Function to call before forward pass
     * @return Hook ID for later removal
     */
    auto register_forward_pre_hook(ForwardPreHook hook) -> size_t;

    /**
     * @brief Register a forward post-hook.
     *
     * Hook is called after forward() executes. Useful for parameter offloading.
     *
     * @param hook Function to call after forward pass
     * @return Hook ID for later removal
     */
    auto register_forward_post_hook(ForwardPostHook hook) -> size_t;

    /**
     * @brief Register a backward pre-hook.
     *
     * Hook is called before backward pass. Useful for gradient prefetching.
     *
     * @param hook Function to call before backward pass
     * @return Hook ID for later removal
     */
    auto register_backward_pre_hook(BackwardPreHook hook) -> size_t;

    /**
     * @brief Register a backward post-hook.
     *
     * Hook is called after backward pass. Useful for gradient offloading.
     *
     * @param hook Function to call after backward pass
     * @return Hook ID for later removal
     */
    auto register_backward_post_hook(BackwardPostHook hook) -> size_t;

    /**
     * @brief Remove a registered hook by ID.
     *
     * @param hook_id ID returned from register_*_hook()
     */
    auto remove_hook(size_t hook_id) -> void;

    /**
     * @brief Call all registered forward pre-hooks.
     *
     * Called internally before forward() execution.
     */
    auto call_forward_pre_hooks() -> void;

    /**
     * @brief Call all registered forward post-hooks.
     *
     * Called internally after forward() execution.
     */
    auto call_forward_post_hooks() -> void;

    /**
     * @brief Call all registered backward pre-hooks.
     *
     * Called before backward pass execution.
     */
    auto call_backward_pre_hooks() -> void;

    /**
     * @brief Call all registered backward post-hooks.
     *
     * Called after backward pass execution.
     */
    auto call_backward_post_hooks() -> void;

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
    auto to(DType dtype) -> void;

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
     * Parameters are stored as shared_ptr to ensure stable addresses for autograd.
     *
     * @param name Parameter name
     * @param param Parameter variable (will be wrapped in shared_ptr)
     */
    auto register_parameter(std::string name, Variable param) -> void;

    /**
     * @brief Register a buffer.
     *
     * Adds a non-trainable tensor to this module.
     *
     * @param name Buffer name
     * @param buffer Buffer variable (will be wrapped in shared_ptr)
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

    bool training_{true};                                                         ///< Training mode flag
    std::unordered_map<std::string, std::shared_ptr<Variable>> parameters_;       ///< Named parameters (stable addresses)
    std::unordered_map<std::string, std::shared_ptr<Variable>> buffers_;          ///< Named buffers (stable addresses)
    std::unordered_map<std::string, std::shared_ptr<Module>> submodules_;         ///< Named submodules

    // Hook system storage
    std::vector<ForwardPreHook> forward_pre_hooks_;                               ///< Forward pre-hooks
    std::vector<ForwardPostHook> forward_post_hooks_;                             ///< Forward post-hooks
    std::vector<BackwardPreHook> backward_pre_hooks_;                             ///< Backward pre-hooks
    std::vector<BackwardPostHook> backward_post_hooks_;                           ///< Backward post-hooks
    size_t next_hook_id_{0};                                                      ///< Next hook ID for tracking
    bool has_forward_hooks_{false};                                               ///< True if this module has forward hooks

    /**
     * @brief Call only this module's forward pre-hooks (no recursion).
     */
    auto call_own_forward_pre_hooks() -> void {
        for (auto& hook : forward_pre_hooks_) {
            hook(this);
        }
    }

    /**
     * @brief Call only this module's forward post-hooks (no recursion).
     */
    auto call_own_forward_post_hooks() -> void {
        for (auto& hook : forward_post_hooks_) {
            hook(this);
        }
    }
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
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Get all parameters preserving module order.
     *
     * @return Vector of shared pointers to parameters
     */
    auto parameters() -> std::vector<std::shared_ptr<Variable>> override;

    /**
     * @brief Get named parameters preserving module order.
     *
     * @return Vector of (name, shared_ptr to parameter) pairs
     */
    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override;

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
