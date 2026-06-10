/**
 * @file module.hpp
 * @brief Neural network module base class
 *
 * Provides Module base class for building neural networks with
 * parameter management, training/eval modes, and serialization.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include "../autograd/variable.hpp"
#include "../autograd/function.hpp"
#include "../core/device.hpp"

namespace tenzor {
namespace nn {

// Forward declarations
class Module;
class ModuleHookFunction;

// Forward declaration of wrap function (defined after Module class)
auto wrap_with_backward_hooks(Module* module, const Variable& input, Variable output) -> Variable;

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
    friend class ModuleHookFunction;  // Allow access to backward hook vectors

public:
    virtual ~Module();;

    Module();;
    Module(Module&& other) noexcept
        : id_(other.id_)
        , training_(other.training_)
        , parameters_(std::move(other.parameters_))
        , buffers_(std::move(other.buffers_))
        , submodules_(std::move(other.submodules_))
        , forward_pre_hooks_(std::move(other.forward_pre_hooks_))
        , forward_post_hooks_(std::move(other.forward_post_hooks_))
        , forward_pre_hooks_multi_(std::move(other.forward_pre_hooks_multi_))
        , forward_post_hooks_multi_(std::move(other.forward_post_hooks_multi_))
        , backward_pre_hooks_(std::move(other.backward_pre_hooks_))
        , backward_post_hooks_(std::move(other.backward_post_hooks_))
        , next_hook_id_(other.next_hook_id_.load())
        , has_forward_hooks_(other.has_forward_hooks_)
        , has_backward_hooks_(other.has_backward_hooks_)
    {}
    Module& operator=(Module&&) = delete;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

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
        if (has_forward_hooks_ || has_backward_hooks_) {
            if (has_forward_hooks_) {
                call_own_forward_pre_hooks(input);
            }
            auto output = forward_impl(input);
            if (has_forward_hooks_) {
                call_own_forward_post_hooks(input, output);
            }
            // Wrap output with backward hook function if backward hooks are registered
            if (has_backward_hooks_) {
                output = wrap_with_backward_hooks(this, input, std::move(output));
            }
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
     * @brief Extra representation string for __repr__.
     *
     * Override in subclasses to show constructor args (e.g. "in_features=10, out_features=5").
     * @return Extra string to display inside ClassName(...)
     */
    virtual auto extra_repr() const -> std::string { return ""; }

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
     * @brief Get a single parameter by name.
     *
     * @param name Parameter name (e.g. "weight", "bias")
     * @return Shared pointer to the parameter Variable
     * @throws std::out_of_range if parameter not found
     */
    auto get_parameter(const std::string& name) const -> std::shared_ptr<Variable>;

    /**
     * @brief Get a single buffer by name.
     *
     * @param name Buffer name (e.g. "running_mean", "running_var")
     * @return Shared pointer to the buffer Variable
     * @throws std::out_of_range if buffer not found
     */
    auto get_buffer(const std::string& name) const -> std::shared_ptr<Variable>;

    /**
     * @brief Unregister a parameter by name.
     *
     * @param name Parameter name to remove
     * @throws std::out_of_range if parameter not found
     */
    auto unregister_parameter(const std::string& name) -> void;

    /**
     * @brief Unregister a buffer by name.
     *
     * @param name Buffer name to remove
     * @throws std::out_of_range if buffer not found
     */
    auto unregister_buffer(const std::string& name) -> void;

    /**
     * @brief Unregister a submodule by name.
     *
     * @param name Submodule name to remove
     * @throws std::out_of_range if submodule not found
     */
    auto unregister_module(const std::string& name) -> void;

    /**
     * @brief Get all direct submodules.
     *
     * @return Map of (name -> submodule) pairs
     */
    auto get_submodules() const -> const std::unordered_map<std::string, std::shared_ptr<Module>>& {
        return submodules_;
    }

    /**
     * @brief Enable/disable activation (gradient) checkpointing recursively.
     *
     * The base implementation simply forwards the request to every registered
     * submodule, so a single call on a top-level model reaches whatever
     * checkpoint-capable containers (Sequential, TransformerEncoder) live inside
     * it — including those behind type-erased `Module` pointers (e.g. a
     * detection model's backbone). Containers override this to set their own
     * flag; leaf modules inherit the recursing default. Off by default.
     */
    virtual auto set_gradient_checkpointing(bool enabled) -> void {
        for (auto& [name, sub] : submodules_) {
            if (sub) sub->set_gradient_checkpointing(enabled);
        }
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

    /**
     * @brief Get the unique 64-bit identifier for this Module instance.
     *
     * Each Module is assigned a process-monotonic UID at construction time.
     * Used by external registries (e.g. nn::utils::parametrize) that need
     * a key that stays stable across moves and never collides with a
     * destroyed-then-recreated Module at the same heap address.
     *
     * @return Unique identifier (>=1; never 0).
     */
    auto id() const -> uint64_t { return id_; }

    // ============================================================================
    // Hook System (for Phase 2 Offload Support)
    // ============================================================================

    /**
     * @brief Hook function types for forward and backward passes.
     *
     * ForwardPreHook receives: (module, input)
     * ForwardPostHook receives: (module, input, output)
     * BackwardPreHook receives: (module, grad_output)
     * BackwardPostHook receives: (module, grad_input, grad_output)
     */
    using ForwardPreHook = std::function<void(Module*, const Variable& input)>;
    using ForwardPostHook = std::function<void(Module*, const Variable& input, const Variable& output)>;
    using BackwardPreHook = std::function<void(Module*, const Variable& grad_output)>;
    using BackwardPostHook = std::function<void(Module*, const Variable& grad_input, const Variable& grad_output)>;

    /// Multi-input forward pre-hook: receives vector of all inputs
    using ForwardPreHookMulti = std::function<void(Module*, const std::vector<Variable>& inputs)>;
    /// Multi-input forward post-hook: receives vector of inputs and vector of outputs
    using ForwardPostHookMulti = std::function<void(Module*, const std::vector<Variable>& inputs,
                                                     const std::vector<Variable>& outputs)>;

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
     * @brief Register a multi-input forward pre-hook.
     *
     * Called before forward with all inputs as a vector. Use for modules
     * that accept multiple Variable inputs (e.g., attention with q, k, v).
     *
     * @param hook Function to call before forward pass
     * @return Hook ID for later removal
     */
    auto register_forward_pre_hook_multi(ForwardPreHookMulti hook) -> size_t;

    /**
     * @brief Register a multi-input forward post-hook.
     *
     * Called after forward with all inputs and outputs as vectors.
     *
     * @param hook Function to call after forward pass
     * @return Hook ID for later removal
     */
    auto register_forward_post_hook_multi(ForwardPostHookMulti hook) -> size_t;

    /**
     * @brief Call all registered multi-input forward pre-hooks.
     *
     * @param inputs All input variables
     */
    auto call_forward_pre_hooks_multi(const std::vector<Variable>& inputs) -> void;

    /**
     * @brief Call all registered multi-input forward post-hooks.
     *
     * @param inputs All input variables
     * @param outputs All output variables
     */
    auto call_forward_post_hooks_multi(const std::vector<Variable>& inputs,
                                        const std::vector<Variable>& outputs) -> void;

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
     * @param input The input variable being passed to forward
     */
    auto call_forward_pre_hooks(const Variable& input) -> void;

    /**
     * @brief Call all registered forward pre-hooks (no-argument version).
     *
     * For modules with multiple inputs that manually call hooks.
     * Passes an empty Variable to hooks.
     */
    auto call_forward_pre_hooks() -> void;

    /**
     * @brief Call all registered forward post-hooks.
     *
     * Called internally after forward() execution.
     * @param input The input variable passed to forward
     * @param output The output variable produced by forward
     */
    auto call_forward_post_hooks(const Variable& input, const Variable& output) -> void;

    /**
     * @brief Call all registered forward post-hooks (no-argument version).
     *
     * For modules with multiple inputs that manually call hooks.
     * Passes empty Variables to hooks.
     */
    auto call_forward_post_hooks() -> void;

    /**
     * @brief Call all registered backward pre-hooks.
     *
     * Called before backward pass execution.
     * @param grad_output The gradient w.r.t. module output
     */
    auto call_backward_pre_hooks(const Variable& grad_output) -> void;

    /**
     * @brief Call all registered backward post-hooks.
     *
     * Called after backward pass execution.
     * @param grad_input The gradient w.r.t. module input
     * @param grad_output The gradient w.r.t. module output
     */
    auto call_backward_post_hooks(const Variable& grad_input, const Variable& grad_output) -> void;

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
    // V.29: virtual so LazyLinear (and any other lazily-materialised module)
    // can intercept `to(DType)` / `to(Device)` calls that arrive before the
    // first forward pass — without an override, a `model.to(BFloat16)` on a
    // not-yet-materialised LazyLinear is silently dropped (parameters_ /
    // buffers_ are empty), then materialise() hardcodes Float32 / the input
    // device and the requested dtype/device never re-fires.
    virtual auto to(Device device) -> void;
    virtual auto to(DType dtype) -> void;

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
     * @brief Load state dictionary with optional strict mode.
     *
     * @param state Map of parameter names to tensors
     * @param strict If true (default), throws on missing/unexpected keys.
     *               If false, silently ignores mismatches.
     */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state, bool strict) -> void;

    /**
     * @brief Audit-4 W.17: load_state_dict variant that reports missing
     *        and unexpected keys via out-parameters instead of throwing.
     *
     * Mirrors PyTorch's ``Module.load_state_dict`` which returns an
     * ``_IncompatibleKeys(missing_keys, unexpected_keys)`` named tuple.
     * With @p strict=true the legacy throw-on-mismatch path is taken
     * for back-compat, and the out-params are still populated before
     * the throw (caller can introspect inside an except: block).
     *
     * @param state            Map of (name -> tensor) with saved state.
     * @param strict           When true, throw on mismatch (for back-compat).
     * @param missing_keys     [out] Keys expected by this module but
     *                          absent from @p state.
     * @param unexpected_keys  [out] Keys present in @p state but not
     *                          consumed by this module.
     */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state,
                         bool strict,
                         std::vector<std::string>& missing_keys,
                         std::vector<std::string>& unexpected_keys) -> void;

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

    /**
     * @brief Register an existing shared Variable as a parameter (public hook).
     *
     * Unlike the protected register_parameter(name, Variable), this overload
     * accepts the shared_ptr directly so external callers (e.g.
     * `WeightNorm` / `SpectralNorm` reparameterisations) can register a
     * Variable whose identity must remain stable across the call. The exact
     * shared_ptr is stored, so the caller and the module share gradient state.
     *
     * @param name Parameter name
     * @param param Existing parameter shared_ptr (must be non-null)
     */
    auto register_parameter_shared(std::string name, std::shared_ptr<Variable> param) -> void;

    /**
     * @brief Register an existing shared Variable as a buffer (public hook).
     *
     * Companion to register_parameter_shared() for non-trainable state
     * tracked by reparameterisations (e.g. SpectralNorm power-iteration
     * vectors `u`/`v`).
     *
     * @param name Buffer name
     * @param buffer Existing buffer shared_ptr (must be non-null)
     */
    auto register_buffer_shared(std::string name, std::shared_ptr<Variable> buffer) -> void;

protected:
    /**
     * @brief Register a parameter.
     *
     * Adds a trainable parameter to this module.
     * Parameters are stored as shared_ptr to ensure stable addresses for autograd.
     * Thread safety is automatically enabled (make_thread_safe()) for safe
     * concurrent gradient accumulation in multi-threaded training.
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

    uint64_t id_{0};                                                              ///< Unique per-instance identifier (assigned in ctor)
    bool training_{true};                                                         ///< Training mode flag
    std::unordered_map<std::string, std::shared_ptr<Variable>> parameters_;       ///< Named parameters (stable addresses)
    std::unordered_map<std::string, std::shared_ptr<Variable>> buffers_;          ///< Named buffers (stable addresses)
    std::unordered_map<std::string, std::shared_ptr<Module>> submodules_;         ///< Named submodules

    // Hook system storage (keyed by hook ID — std::map for deterministic registration-order iteration)
    std::map<size_t, ForwardPreHook> forward_pre_hooks_;                ///< Forward pre-hooks
    std::map<size_t, ForwardPostHook> forward_post_hooks_;              ///< Forward post-hooks
    std::map<size_t, ForwardPreHookMulti> forward_pre_hooks_multi_;     ///< Multi-input forward pre-hooks
    std::map<size_t, ForwardPostHookMulti> forward_post_hooks_multi_;   ///< Multi-input forward post-hooks
    std::map<size_t, BackwardPreHook> backward_pre_hooks_;              ///< Backward pre-hooks
    std::map<size_t, BackwardPostHook> backward_post_hooks_;            ///< Backward post-hooks
    std::atomic<size_t> next_hook_id_{0};                                           ///< Next hook ID for tracking
    bool has_forward_hooks_{false};                                               ///< True if this module has forward hooks
    bool has_backward_hooks_{false};                                              ///< True if this module has backward hooks

    /**
     * @brief Call only this module's forward pre-hooks (no recursion).
     * @param input The input variable being passed to forward
     */
    auto call_own_forward_pre_hooks(const Variable& input) -> void {
        for (auto& [id, hook] : forward_pre_hooks_) {
            hook(this, input);
        }
    }

    /**
     * @brief Call only this module's forward post-hooks (no recursion).
     * @param input The input variable passed to forward
     * @param output The output variable produced by forward
     */
    auto call_own_forward_post_hooks(const Variable& input, const Variable& output) -> void {
        for (auto& [id, hook] : forward_post_hooks_) {
            hook(this, input, output);
        }
    }

    /**
     * @brief Validate that input is compatible with this module's parameters.
     *
     * Checks that the input is on the same device as the module's first parameter.
     * Layers can optionally call this at the start of forward_impl() for
     * better error messages on device/dtype mismatches.
     *
     * @param input The input variable to validate
     * @throws std::runtime_error if device mismatch detected
     */
    auto validate_input_compat(const Variable& input) const -> void {
        if (parameters_.empty()) return;
        const auto& first_param = parameters_.begin()->second;
        if (!first_param) return;
        auto param_device = first_param->tensor().device();
        auto input_device = input.tensor().device();
        if (param_device != input_device) {
            throw std::runtime_error(
                "Input tensor is on " + input_device.to_string() +
                " but module parameters are on " + param_device.to_string() +
                ". Use module.to(device) or input.to(device) to match devices.");
        }
    }
};

/**
 * @brief Autograd function for module backward hook integration.
 *
 * This function is inserted into the computation graph when a module has
 * backward hooks registered. It acts as an identity function on forward
 * (passing through the tensor unchanged) but triggers the module's backward
 * hooks during the backward pass.
 *
 * This enables PyTorch-compatible backward hooks that are called automatically
 * during loss.backward().
 */
class ModuleHookFunction : public Function {
public:
    /**
     * @brief Construct with module reference and input variable.
     *
     * @param module The module whose backward hooks to call
     * @param input The input that was passed to the module's forward
     */
    ModuleHookFunction(Module* module, Variable input)
        : module_(module), input_(std::move(input)) {}

    /**
     * @brief Forward pass - identity function.
     *
     * Simply returns the input unchanged. The function exists only to
     * insert a node in the computation graph for backward hook triggering.
     */
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;

    /**
     * @brief Backward pass - triggers module hooks and passes gradient through.
     *
     * Calls the module's backward pre-hooks and post-hooks, then returns
     * the gradient unchanged (identity gradient).
     */
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;

private:
    Module* module_;  ///< Module whose hooks to call (non-owning)
    Variable input_;  ///< Original input for hook context
};

/**
 * @brief Wrap a module's output with backward hook support.
 *
 * If the module has backward hooks registered, wraps the output variable
 * with a ModuleHookFunction so backward hooks are triggered during backprop.
 *
 * @param module The module
 * @param input The input passed to forward
 * @param output The output from forward_impl
 * @return Output wrapped with hook function if needed, otherwise unchanged
 */
auto wrap_with_backward_hooks(Module* module, const Variable& input, Variable output) -> Variable;

/**
 * @brief List container for modules (like PyTorch's nn.ModuleList).
 *
 * ModuleList holds modules in a list and properly registers them as
 * submodules so that parameter traversal works correctly. Unlike
 * Sequential, ModuleList does NOT implement forward — it is a
 * container only.
 *
 * @code
 * auto layers = std::make_shared<ModuleList>();
 * layers->append(std::make_shared<Linear>(10, 20));
 * layers->append(std::make_shared<Linear>(20, 30));
 *
 * // Iterate manually
 * Variable x = input;
 * for (size_t i = 0; i < layers->size(); ++i) {
 *     x = layers->at(i)->forward(x);
 * }
 * @endcode
 */
class ModuleList : public Module {
public:
    ModuleList() = default;

    /**
     * @brief Append a module to the list.
     *
     * @param module Module to append
     * @return Reference to this ModuleList for chaining
     */
    auto append(std::shared_ptr<Module> module) -> ModuleList&;

    /**
     * @brief Get module at index.
     *
     * @param idx Index (0-based)
     * @return Shared pointer to module
     * @throws std::out_of_range if index is out of bounds
     */
    auto at(size_t idx) const -> std::shared_ptr<Module>;

    /**
     * @brief Get number of modules.
     */
    auto size() const -> size_t { return modules_.size(); }

    auto begin() { return modules_.begin(); }
    auto end() { return modules_.end(); }
    auto begin() const { return modules_.begin(); }
    auto end() const { return modules_.end(); }

    /**
     * @brief ModuleList does not implement forward.
     * @throws std::runtime_error always
     */
    auto forward_impl(const Variable& input) -> Variable override;

    auto parameters() -> std::vector<std::shared_ptr<Variable>> override;
    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override;
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

/**
 * @brief Dictionary container for modules (like PyTorch's nn.ModuleDict).
 *
 * ModuleDict holds modules in a dictionary keyed by string and properly
 * registers them as submodules. Maintains insertion order for iteration.
 * Does NOT implement forward.
 *
 * @code
 * auto blocks = std::make_shared<ModuleDict>();
 * blocks->insert("encoder", std::make_shared<Linear>(784, 256));
 * blocks->insert("decoder", std::make_shared<Linear>(256, 784));
 *
 * auto encoded = blocks->at("encoder")->forward(input);
 * auto decoded = blocks->at("decoder")->forward(encoded);
 * @endcode
 */
class ModuleDict : public Module {
public:
    ModuleDict() = default;

    /**
     * @brief Insert or replace a module with the given key.
     *
     * @param key String key for the module
     * @param module Module to insert
     * @return Reference to this ModuleDict for chaining
     */
    auto insert(const std::string& key, std::shared_ptr<Module> module) -> ModuleDict&;

    /**
     * @brief Get module by key.
     *
     * @param key Module key
     * @return Shared pointer to module
     * @throws std::out_of_range if key not found
     */
    auto at(const std::string& key) const -> std::shared_ptr<Module>;

    /**
     * @brief Check if key exists.
     */
    auto contains(const std::string& key) const -> bool;

    /**
     * @brief Remove module by key.
     *
     * @param key Module key
     * @throws std::out_of_range if key not found
     */
    auto erase(const std::string& key) -> void;

    /**
     * @brief Get number of modules.
     */
    auto size() const -> size_t { return order_.size(); }

    /**
     * @brief Get keys in insertion order.
     */
    auto keys() const -> std::vector<std::string> { return order_; }

    /**
     * @brief Get values in insertion order.
     */
    auto values() const -> std::vector<std::shared_ptr<Module>>;

    /**
     * @brief Get key-value pairs in insertion order.
     */
    auto items() const -> std::vector<std::pair<std::string, std::shared_ptr<Module>>>;

    /**
     * @brief ModuleDict does not implement forward.
     * @throws std::runtime_error always
     */
    auto forward_impl(const Variable& input) -> Variable override;

    auto parameters() -> std::vector<std::shared_ptr<Variable>> override;
    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override;
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    std::unordered_map<std::string, std::shared_ptr<Module>> modules_;
    std::vector<std::string> order_;  ///< Insertion-order keys
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

    /**
     * @brief Get read-only access to the ordered list of modules.
     *
     * @return Const reference to the internal module list
     */
    auto modules() const -> const std::vector<std::shared_ptr<Module>>& { return modules_; }

    /**
     * @brief Enable/disable activation (gradient) checkpointing.
     *
     * When enabled, each contained module's forward is wrapped in
     * autograd::checkpoint(): its activations are recomputed during backward
     * instead of being retained, cutting peak memory by ~O(num_modules) at the
     * cost of one extra forward. Gradients are identical (RNG is captured and
     * replayed). Off by default. Lets very deep Sequential stacks (e.g.
     * EfficientNet-B7 stages, ResNet backbones) train within tight GPU memory.
     */
    auto set_gradient_checkpointing(bool enabled) -> void { gradient_checkpointing_ = enabled; }
    auto gradient_checkpointing() const -> bool { return gradient_checkpointing_; }

private:
    std::vector<std::shared_ptr<Module>> modules_;  ///< Ordered list of modules
    bool gradient_checkpointing_{false};
};

/**
 * @brief Ordered list container for parameters (like PyTorch's nn.ParameterList).
 *
 * ParameterList holds parameters in an ordered list and properly registers them.
 * Does NOT implement forward.
 *
 * @code
 * auto params = std::make_shared<ParameterList>();
 * params->append(Variable(Tensor({3, 3}, DType::Float32, Device::cpu()), true));
 * params->append(Variable(Tensor({3}, DType::Float32, Device::cpu()), true));
 * auto p = params->at(0);  // First parameter
 * @endcode
 */
class ParameterList : public Module {
public:
    ParameterList() = default;

    auto append(Variable param) -> ParameterList&;
    auto at(size_t idx) const -> std::shared_ptr<Variable>;
    auto size() const -> size_t { return params_.size(); }

    auto begin() { return params_.begin(); }
    auto end() { return params_.end(); }
    auto begin() const { return params_.begin(); }
    auto end() const { return params_.end(); }

    auto forward_impl(const Variable& input) -> Variable override;
    auto parameters() -> std::vector<std::shared_ptr<Variable>> override;
    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override;
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    std::vector<std::shared_ptr<Variable>> params_;
};

/**
 * @brief Dictionary container for parameters (like PyTorch's nn.ParameterDict).
 *
 * ParameterDict holds parameters in a dictionary keyed by string and properly
 * registers them. Maintains insertion order for iteration.
 * Does NOT implement forward.
 *
 * @code
 * auto params = std::make_shared<ParameterDict>();
 * params->insert("weight", Variable(Tensor({3, 3}, DType::Float32, Device::cpu()), true));
 * params->insert("bias", Variable(Tensor({3}, DType::Float32, Device::cpu()), true));
 * auto w = params->at("weight");
 * @endcode
 */
class ParameterDict : public Module {
public:
    ParameterDict() = default;

    auto insert(const std::string& key, Variable param) -> ParameterDict&;
    auto at(const std::string& key) const -> std::shared_ptr<Variable>;
    auto contains(const std::string& key) const -> bool;
    auto erase(const std::string& key) -> void;
    auto size() const -> size_t { return order_.size(); }
    auto keys() const -> std::vector<std::string> { return order_; }

    auto forward_impl(const Variable& input) -> Variable override;
    auto parameters() -> std::vector<std::shared_ptr<Variable>> override;
    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override;
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    std::unordered_map<std::string, std::shared_ptr<Variable>> params_;
    std::vector<std::string> order_;
};

} // namespace nn
} // namespace tenzor
