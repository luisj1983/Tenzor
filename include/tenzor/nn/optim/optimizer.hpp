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
#include <functional>
#include <optional>
#include <atomic>
#include <cstdint>
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

    // Audit D.4: per-group hyperparameter overrides.  Each `std::optional`
    // field, when set, overrides the corresponding member on the
    // optimizer for *just this group's parameters*.  Optimizers read
    // these via the `read_*` accessors below so adding a new field is a
    // single-line change in the relevant step_impl().
    //
    // PyTorch's torch.optim uses a free-form dict for this; we use
    // named optionals to keep the type-checker honest at the C++ side.
    // Hyperparams that no optimizer actually consumes can be left as
    // std::nullopt — the optimizer falls back to its own member default.
    std::optional<double> momentum;
    std::optional<double> dampening;
    std::optional<bool>   nesterov;
    std::optional<double> beta1;
    std::optional<double> beta2;
    std::optional<double> eps;
    std::optional<bool>   amsgrad;      ///< Adam / AdamW / AdamAtan2
    std::optional<bool>   centered;     ///< RMSprop
    std::optional<double> alpha;        ///< RMSprop / ASGD
    std::optional<double> rho;          ///< Adadelta
    std::optional<double> lr_decay;     ///< Adagrad
    std::optional<double> initial_accumulator_value;  ///< Adagrad

    /**
     * @brief Read a per-group hyperparam with optimizer-member fallback.
     *
     * Used inside step_impl(): the loop over param_groups_ picks each
     * group's lr / weight_decay / etc. from the group's stored optional
     * (if set) or from the optimizer's own default member.  See SGD /
     * Adam / etc. step_impl() bodies for the pattern.
     */
    template <typename T>
    static T or_else(const std::optional<T>& opt, T fallback) {
        return opt.has_value() ? opt.value() : fallback;
    }
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
     * @brief Perform optimization step with a closure that recomputes the loss.
     *
     * Calls the closure to (re-)evaluate the model and compute gradients,
     * then applies the optimizer step. Useful for optimizers like LBFGS
     * that require multiple function evaluations.
     *
     * @param closure Callable that clears gradients, computes forward+backward,
     *                and returns the loss as a Variable
     * @return The loss value returned by the closure
     *
     * @code
     * auto loss = optimizer.step([&]() {
     *     optimizer.zero_grad();
     *     auto output = model.forward(input);
     *     auto loss = criterion(output, target);
     *     loss.backward();
     *     return loss;
     * });
     * @endcode
     */
    // Virtual so derived classes that need a different control flow
    // (e.g. SAM, which requires two forward+backward passes around a
    // weight perturbation) can override the whole step(closure) path,
    // not just step_impl().
    virtual auto step(std::function<Variable()> closure) -> Variable;

    /**
     * @brief Implementation of the parameter update step.
     *
     * Must be implemented by derived classes to perform the actual optimization
     * algorithm (SGD, Adam, etc.). Called by step() after gradient clipping.
     */
    virtual auto step_impl() -> void = 0;

    /**
     * @brief Register a post-step hook (audit G.10).
     *
     * Hooks fire in registration order at the end of every successful
     * `step()` / `step(closure)` invocation, after the parameter update
     * has been applied.  Used by:
     *
     * - Pruning utilities to re-apply masks after each gradient step
     *   (audit G.10) so the optimizer can't silently undo the mask.
     * - User code that needs an after-step callback for logging,
     *   weight clipping, etc.
     *
     * Hooks see the optimizer's own parameter list via `parameters()`.
     * Exceptions thrown by a hook propagate.
     *
     * @param hook Callable invoked with no arguments at the end of step().
     * @return A handle that can be passed to `remove_post_step_hook` to
     *         deregister.  Handles are stable across hook insertions/
     *         removals.
     */
    using PostStepHook = std::function<void()>;
    auto register_post_step_hook(PostStepHook hook) -> uint64_t;

    /**
     * @brief Deregister a previously registered post-step hook.
     *
     * @param hook_id Handle returned by `register_post_step_hook`.
     * @return true if a hook was found and removed; false otherwise.
     */
    auto remove_post_step_hook(uint64_t hook_id) -> bool;

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
     * @brief Replace the parameters being optimized.
     *
     * Used by MasterWeightManager to swap model parameters with FP32 master
     * copies so the optimizer operates in full precision.
     *
     * @param new_params New parameter vector (must be same size as current)
     */
    auto replace_parameters(std::vector<std::shared_ptr<Variable>> new_params) -> void;

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
     * @brief Hook called by add_param_group() after parameters have been
     *        appended to parameters_ and the new ParamGroup has been
     *        recorded.  Derived optimisers MUST override this to extend
     *        their state buffers (exp_avg_, exp_avg_sq_, momentum_buffer_,
     *        sum_, acc_delta_, square_avg_, etc.) so the next step_impl()
     *        does not OOB-read on the newly-added param indices.
     *
     * @param old_count  parameters_.size() before the append.
     * @param new_count  parameters_.size() after the append.
     *
     * Default implementation throws std::runtime_error.  This is
     * deliberately not a silent no-op — every optimizer is required to
     * extend its state, and a default that silently did nothing would
     * reintroduce the K.1 OOB bug (audit 2026-05-23).
     */
    virtual auto on_parameters_appended_(size_t old_count, size_t new_count) -> void;

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
     * Base implementation writes @p lr into every ParamGroup, which is the
     * correct behaviour for any optimizer that uses the standard
     * param_groups_ container. Optimizers that maintain LR outside of
     * param_groups_ should override.
     *
     * Used by LR schedulers.
     *
     * @param lr New learning rate (applied to all parameter groups)
     */
    virtual auto set_lr(double lr) -> void;

    /**
     * @brief Get current learning rate from the optimizer.
     *
     * Base implementation returns the LR of the first ParamGroup
     * (matches PyTorch's `param_groups[0]['lr']` convention). Callers that
     * need per-group LRs should use param_groups() directly.
     *
     * @return Current learning rate of the first parameter group
     */
    virtual auto get_lr() const -> double;

    /**
     * @brief Canonical hyperparameter map for state-dict serialization.
     *
     * Mirrors PyTorch's `Optimizer.defaults`. Returns a flattened key/value
     * map of every scalar hyperparameter the optimizer cares about — at the
     * minimum {"lr", "weight_decay"} from the first ParamGroup. Concrete
     * optimizers override to expose their own params (Adam adds beta1,
     * beta2, eps; SGD adds momentum, dampening, nesterov; etc.). Tuple-typed
     * hyperparameters (e.g. Adam's betas) are flattened to scalar keys
     * (beta1, beta2) so the map fits a uniform serialization format.
     *
     * @return Map of hyperparameter name to scalar value
     */
    virtual auto defaults() const -> std::unordered_map<std::string, double>;

    /**
     * @brief Total number of completed step() invocations.
     *
     * Incremented monotonically by the base step() implementation after
     * step_impl() returns and post-step hooks fire. Used by features that
     * need a stable per-iteration key across post-step hooks (S.14: pruning
     * mask reapplication idempotence keyed on step_count() rather than a
     * hook-local atomic counter).
     *
     * @return Monotonic count of completed steps (starts at 0).
     */
    auto step_count() const -> uint64_t;

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

    // Audit G.10: post-step hook storage. Vector of (id, callable) so we
    // can preserve registration order *and* remove a specific hook by id.
    // The `next_hook_id_` counter is incremented atomically so handles
    // are stable across insertions / removals.
    std::vector<std::pair<uint64_t, PostStepHook>> post_step_hooks_;
    std::atomic<uint64_t> next_hook_id_{1};

    // S.14: monotonic per-optimizer step counter so post-step hooks
    // (e.g. pruning mask reapplication) can key idempotence on a stable
    // global step number rather than a hook-local atomic that ticks once
    // per invocation (which the threaded DataParallel hand-out breaks).
    std::atomic<uint64_t> step_count_total_{0};

    /**
     * @brief Fire every registered post-step hook in registration order.
     *
     * Called by `step()` and `step(closure)` after the underlying
     * step_impl() has updated the parameters. Exceptions thrown by a
     * hook propagate to the caller of step().
     */
    auto fire_post_step_hooks_() -> void;

    /**
     * @brief Resolve the ParamGroup that owns `parameters_[i]`, if any.
     *
     * Audit D.4 helper: when a derived optimizer's step_impl() needs to
     * read per-group hyperparameters (lr, weight_decay, momentum, …),
     * it calls this to find the matching group.  Returns nullptr if
     * the optimizer was constructed from a flat parameter list (no
     * groups) — the caller should fall back to its own defaults.
     *
     * O(n_groups × n_params_per_group) worst case; with the typical 1–5
     * groups this is fine.
     */
    auto find_group_for_param(size_t param_index) const -> const ParamGroup*;
};

} // namespace optim
} // namespace tenzor
