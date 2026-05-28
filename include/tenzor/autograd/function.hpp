/**
 * @file function.hpp
 * @brief Autograd function interface and built-in operations
 *
 * Defines the Function base class for autograd operations and
 * provides implementations for common differentiable operations.
 */

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <atomic>
#include "../core/tensor.hpp"
#include "../ops/op_id.hpp"
#include "../backend/op_attributes.hpp"
#include "../sparse/sparse_tensor.hpp"
#include "variable.hpp"

namespace tenzor {

/**
 * @brief Controls behavior when create_graph=true but the operation
 * doesn't implement backward_with_variables().
 */
enum class HigherOrderGradMode : uint8_t {
    Error,   ///< Throw std::runtime_error (default, safe)
    Warn,    ///< Log warning per disconnection, fall through with disconnected graph
};

/// Set the higher-order gradient fallback mode programmatically.
/// This takes precedence over the TENZOR_HIGHER_ORDER_GRAD env var.
void set_higher_order_grad_mode(HigherOrderGradMode mode);

/// Get the current higher-order gradient fallback mode.
auto get_higher_order_grad_mode() -> HigherOrderGradMode;

/// Returns the number of times higher-order gradient graph was disconnected
/// due to an operation not supporting backward_with_variables() (Warn/Silent mode).
/// Useful for programmatic detection in tests and debugging.
auto higher_order_disconnection_count() -> uint64_t;

/// Reset the disconnection counter to zero.
void reset_higher_order_disconnection_count();

namespace detail {
/// Internal: increment the disconnection counter (used by engine for stub ops)
void increment_higher_order_disconnection_count();
} // namespace detail

/// Enable/disable activation offloading to CPU for GPU-resident saved tensors.
/// When enabled, save_for_backward() moves GPU tensors to CPU RAM to reduce
/// GPU memory pressure, and saved_tensors() loads them back on access.
void set_activation_offload(bool enabled);
bool activation_offload_enabled();

/**
 * @brief Inject the three-method higher-order stub into a Function subclass.
 *
 * Pastes `backward_with_variables()`, `supports_higher_order()`, and
 * `is_higher_order_stub()` overrides that forward to
 * `Function::passthrough_stub_backward()`. Use from inside a Function
 * subclass whose forward is linear or piecewise-linear (and therefore
 * whose second derivative is structurally zero) — pooling, dropout,
 * flatten, embedding, type-cast, bilinear upsample, ReLU, LeakyReLU.
 *
 * Example:
 * @code
 * class AvgPool2dBackward : public Function {
 * public:
 *     auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
 *     TENZOR_HIGHER_ORDER_STRUCTURAL_ZERO_STUB()
 * };
 * @endcode
 *
 * Do NOT use for genuinely non-linear ops (sigmoid, tanh, GeLU, conv,
 * batch/layer norm) — those need a real `backward_with_variables`
 * implementation that builds a gradient graph via Variable ops.
 */
#define TENZOR_HIGHER_ORDER_STRUCTURAL_ZERO_STUB()                               \
    auto backward_with_variables(std::vector<Variable> grad_outputs)             \
        -> std::vector<Variable> override {                                      \
        return passthrough_stub_backward(std::move(grad_outputs));               \
    }                                                                            \
    auto supports_higher_order() const -> bool override { return true; }         \
    auto is_higher_order_stub() const -> bool override { return true; }

/**
 * @brief Per-function offload policy for activation offloading.
 *
 * Controls whether a specific Function offloads saved tensors to CPU,
 * overriding the global thread-local setting.
 */
enum class OffloadPolicy : uint8_t {
    Inherit,  ///< Use the global thread-local setting (default)
    Always,   ///< Always offload saved tensors to CPU (regardless of global setting)
    Never     ///< Never offload (regardless of global setting)
};

// Forward declaration
class Variable;

/**
 * @brief Base class for autograd differentiable functions.
 *
 * Function represents a differentiable operation in the computation graph.
 * Each operation implements forward() and backward() to support automatic
 * differentiation.
 *
 * To create custom differentiable operations:
 * 1. Inherit from Function
 * 2. Implement forward() to compute output
 * 3. Implement backward() to compute gradients
 * 4. Use save_for_backward() to store tensors needed for gradient computation
 *
 * @code
 * class MyCustomOp : public Function {
 * public:
 *     auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
 *         // Save inputs if needed for backward
 *         save_for_backward({inputs[0].tensor()});
 *
 *         // Compute output
 *         Tensor result = compute(inputs[0].tensor());
 *         return {Variable(result)};
 *     }
 *
 *     auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
 *         // Retrieve saved tensors
 *         auto& input = saved_tensors()[0];
 *
 *         // Compute input gradient from output gradient
 *         Tensor grad_input = compute_gradient(input, grad_outputs[0]);
 *         return {grad_input};
 *     }
 * };
 * @endcode
 */
class Function : public std::enable_shared_from_this<Function> {
    friend class Variable;  // Allow Variable to access saved_tensors_

public:
    Function() : id_(next_id_.fetch_add(1, std::memory_order_relaxed)) {}
    virtual ~Function() = default;

    /**
     * @brief Unique identifier for this Function instance.
     *
     * Used as a stable key in gradient accumulation maps instead of raw
     * pointers, avoiding potential issues with address reuse after free.
     */
    auto id() const noexcept -> uint64_t { return id_; }

    /**
     * @brief Forward pass computation.
     *
     * Computes the function output from inputs. Should save any tensors
     * needed for the backward pass using save_for_backward().
     *
     * @param inputs Input variables
     * @return Output variables
     */
    virtual auto forward(std::vector<Variable> inputs) -> std::vector<Variable> = 0;

    /**
     * @brief Backward pass gradient computation.
     *
     * Computes gradients of inputs given gradients of outputs.
     * Uses saved tensors from forward pass.
     *
     * @param grad_outputs Gradients with respect to outputs
     * @return Gradients with respect to inputs
     */
    virtual auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> = 0;

    /**
     * @brief Higher-order backward pass using Variables for graph creation.
     *
     * When create_graph=true during backward(), this method is called instead
     * of backward(). It receives and returns Variables so that the gradient
     * computation itself is tracked by autograd, enabling higher-order gradients.
     *
     * The default implementation throws an error if any grad_output requires grad,
     * since the gradient graph cannot be built without a proper implementation.
     * Subclasses that support higher-order gradients must override this method.
     *
     * **create_graph coverage:** Currently the following ops have full
     * higher-order gradient support via backward_with_variables():
     *   - AddBackward, SubBackward, MulBackward, DivBackward, MatMulBackward
     *   - LinearBackward, SumBackward, MeanBackward, LogBackward, ExpBackward, NegBackward
     *   - SigmoidBackward_AG, TanhBackward_AG, GeluBackward, EluBackward, SeluBackward
     *   - MishBackward, LeakyReluBackward, SoftplusBackward, ReLUBackward
     *   - SinBackward, CosBackward, TanBackward, AsinBackward, AcosBackward, AtanBackward
     *   - SinhBackward, CoshBackward
     *   - ErfBackward, ErfcBackward, Log2Backward, Log10Backward, Log1pBackward
     *   - Exp2Backward, Expm1Backward, Atan2Backward
     *   - SqrtBackward, PowBackward, ReciprocalBackward, AbsBackward, ClampBackward
     *   - VarBackward, StdBackward, ProdBackward, LogSumExpBackward
     *   - ReshapeBackward, TransposeBackward, PermuteBackward, SqueezeBackward
     *   - UnsqueezeBackward, ExpandBackward, FlattenBackward, RollBackward
     *   - BmmBackward, CatBackward, SliceBackward, WhereBackward
     *   - FlipBackward, RepeatBackward, CumSumBackward, CumProdBackward
     *   - DiagBackward, TraceBackward, TriuBackward, TrilBackward
     *   - FFTBackward, IFFTBackward, RFFTBackward, IRFFTBackward
     *   - LogSoftmaxBackward, SoftmaxBackward
     *   - DetBackward, InvBackward, SolveBackward, NormBackward_Linalg, SlogdetBackward
     *   - CholeskyBackward, SvdBackward, QrBackward, EighBackward, EigvalshBackward
     *   - SpMMBackward, SpMVBackward, SparseAddBackward
     *   - MedianBackward, ModeBackward
     *   - LogAddExpBackward, LogAddExp2Backward, XLogYBackward
     *   - I0eBackward, I1eBackward, EntrBackward, SphericalBesselJ0Backward
     *   - NdtrBackward, LogNdtrBackward, MultigammalnBackward
     *   - CosineSimilarityBackward, RenormBackward
     *   - CholeskyInverseBackward, LinalgLDLSolveBackward
     *   - TensorInvBackward, TensorSolveBackward
     *   - LinalgVectorNormBackward, LinalgMatrixNormBackward, LinalgVecdotBackward
     *
     * Ops with full backward_with_variables using Variable-level scatter_add:
     *   - GatherBackward, IndexSelectBackward, IndexBackward, ScatterAddBackward
     *   - MaxBackward, MinBackward, TopKBackward, SortBackward
     *   - ScatterBackward, NarrowBackward
     *
     * Ops with passthrough stubs (is_higher_order_stub() returns true):
     *   - UpsampleBilinearBackward / UpsampleBackward — fixed interpolation
     *     weights; structurally zero 2nd derivative.
     *   - Pooling (MaxPool/AvgPool/AdaptivePool 1D/2D/3D) — piecewise-linear
     *     selection/averaging; structurally zero 2nd derivative.
     *   - Flatten/View/Reshape/Dropout — either linear or masked-identity;
     *     structurally zero 2nd derivative.
     *   - Embedding — lookup table; grad is scatter_add of incoming grad.
     *   - Attention softmax-backward (non-flash path) flagged stub.
     *   - Conv{1,2,3}dBackward, ConvTranspose{1,2,3}dBackward (P4.2e) — real
     *     2nd-order conv requires a differentiable conv_transpose + weight-
     *     backward re-expressed as primal convs (multi-week per backend;
     *     tracked as a future RFC). Currently flagged as stubs so Warn mode
     *     continues to work while Error mode surfaces the gap.
     *   - BatchNorm/LayerNorm/GroupNorm/InstanceNorm/RMSNorm/SyncBatchNorm
     *     (P4.2f) — normalization 2nd-order needs Variable-level
     *     mean/var/rsqrt compositions through the statistics path; flagged
     *     as stubs pending a dedicated branch.
     *   - RNN/LSTM/GRU (P4.2g) — recurrent 2nd-order requires per-timestep
     *     Variable-level gate recomputation; flagged as stubs pending a
     *     dedicated branch.
     *   - LinalgLDLFactorBackward — complex structured symmetric backprop;
     *     returns zeros through factorization (use solve path for gradients).
     *   - LinalgHouseholderBackward — complex Householder product backward;
     *     returns zeros (rarely needed in gradient flows).
     *   - AsStridedBackward — scatter_add inverse stride mapping; flagged stub.
     *
     * The "structural-zero" cases are mathematically correct. The Conv/Norm/
     * RNN cases are pragmatic correctness compromises that Warn mode accepts
     * (see HigherOrderGradMode). Set HigherOrderGradMode::Error to surface
     * them as hard failures at runtime.
     *
     * @param grad_outputs Gradient Variables with respect to outputs
     * @return Gradient Variables with respect to inputs (with grad_fn set)
     * @throws std::runtime_error if create_graph=true and op doesn't support it
     */
    virtual auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable>;

    /**
     * @brief Query whether this function supports higher-order gradients.
     *
     * Returns true if the subclass has a proper backward_with_variables()
     * implementation that builds a gradient graph. The default returns false.
     * Subclasses that override backward_with_variables() should also override
     * this to return true.
     *
     * @note In practice, if a subclass overrides backward_with_variables(),
     * the base class fallback (which checks this flag) is never reached.
     * This method is primarily useful for introspection and pre-flight checks.
     */
    virtual auto supports_higher_order() const -> bool { return false; }

    /**
     * @brief Query whether this function's backward_with_variables() is a stub.
     *
     * Returns true if the subclass has a passthrough stub that delegates to
     * backward() and wraps results without building a gradient graph. This
     * means higher-order gradients through this operation will be zero.
     *
     * Stubs are appropriate for operations whose forward is linear or
     * piecewise-linear (pooling, dropout, flatten, embedding, type-cast,
     * bilinear upsample, ReLU/LeakyReLU), because the mathematical second
     * derivative is structurally zero and the passthrough produces the
     * correct result. Ops with genuinely non-linear forwards (sigmoid,
     * tanh, GeLU, conv, normalization, ...) need real backward_with_variables
     * implementations, not stubs.
     *
     * @return true if backward_with_variables() is a passthrough stub
     */
    virtual auto is_higher_order_stub() const -> bool { return false; }

    /**
     * @brief Query whether a saved tensor at the given index is actually
     *        needed for backward computation.
     *
     * Returns true by default (conservative: assume all saved tensors are
     * needed). Subclasses may override to return false for specific indices
     * whose saved tensors are not read during backward(), enabling in-place
     * modification of those tensors between forward and backward without
     * triggering a version check error.
     *
     * For example, ReLUBackward saves the output but doesn't need the input,
     * so it overrides this for index 0 to return false.
     *
     * @param index Index into the saved_tensors vector
     * @return true if the tensor at this index must not be modified in-place
     */
    virtual auto needs_saved_tensor(size_t index) const -> bool {
        (void)index;
        return true;
    }

    /**
     * @brief Shared passthrough used by `is_higher_order_stub()`-style
     *        subclasses whose second derivative is structurally zero.
     *
     * Unwraps the incoming Variables to their tensors, calls the
     * subclass's raw `backward()`, and returns Variables that carry
     * no grad_fn. Autograd treats the resulting gradient graph as
     * disconnected at this op (higher-order gradients become zero).
     *
     * Intended to be called from a subclass's `backward_with_variables`
     * override; normally paired with the
     * `TENZOR_HIGHER_ORDER_STRUCTURAL_ZERO_STUB()` helper macro below.
     */
    auto passthrough_stub_backward(std::vector<Variable> grad_outputs)
        -> std::vector<Variable>;

    /**
     * @brief Get readable name for this backward function.
     *
     * Used in error messages and debugging output. Default returns
     * demangled C++ type name. Subclasses should override for readability.
     *
     * @return Human-readable function name (e.g., "AddBackward", "MatMulBackward")
     */
    virtual auto name() const -> std::string;

    /**
     * @brief Get the canonical forward `OpId` this Function differentiates.
     *
     * Audit A.2: enables `Graph::replace_nodes` / fusion-pattern matchers /
     * vmap-rule registries to identify a Function by its forward op instead
     * of RTTI-substring or name() string matching, which historically led
     * to silently-broken pattern matches when class names diverged from
     * the matcher's expected string.
     *
     * Subclasses *should* override and return the OpId of the matching
     * forward op. The default returns `OpId::Unknown`, which is treated by
     * the pattern matchers as "do not match" — opt-in, so subclasses that
     * don't override remain pattern-invisible (the previous fall-through
     * behaviour) rather than silently mis-matching.
     *
     * @return The forward OpId this Function differentiates, or
     *         `OpId::Unknown` if the subclass hasn't opted in.
     */
    virtual auto op_id() const -> OpId { return OpId::Unknown; }

    /**
     * @brief Reconstruct an `OpAttributes` snapshot of this Function's
     *        attribute-bearing fields, for forward-mode (JVP) dispatch.
     *
     * Audit A.4 (multi-op JVP graph traversal): the JVP dispatch table is
     * keyed by `OpId` and parametrised by `OpAttributes`. The autograd
     * `Function` graph stores attributes in Function-specific fields
     * (`dim_`, `dims_`, `dim0_/dim1_`, `min_/max_`, scalar params kept in
     * `saved_tensors_[1]`, etc.), not in an `OpAttributes` object. The
     * `functional::jvp` graph walker calls this method on each node to
     * recover the attributes needed by `dispatch_jvp` without having to
     * know the concrete subclass.
     *
     * Default returns an empty `OpAttributes`, which is appropriate for
     * ops whose JVP rules take no attributes (e.g. Add, Mul, ReLU, MatMul).
     * Subclasses with attribute-bearing JVP rules override to populate
     * the relevant `AttrKey` entries from their saved fields. See
     * `src/autograd/jvp_rules.cpp` for the per-op attribute contract.
     */
    virtual auto saved_attributes() const -> OpAttributes { return {}; }

    /**
     * @brief Repack the walker's (primals, tangents) into the contract the
     *        JVP rule for this op expects, for the A.4 multi-output walker
     *        integration.
     *
     * The default forward-graph walker assembles primals/tangents from
     * `input_variables()` parallel-paired with `next_functions()`. That
     * works when the Function's `input_variables` exactly mirror the
     * registered JVP rule's `primals` contract. Some ops (notably
     * `LayerNormBackward` / `BatchNorm2dBackward`) only retain
     * gradient-requiring inputs in `input_variables` while the multi-output
     * JVP rule expects a fixed-arity tuple (e.g. LayerNorm needs (x, gamma,
     * beta) even if gamma/beta are constants in this trace).
     *
     * Overrides return the (primals, tangents) the JVP rule should be
     * called with. The default returns `nullopt`, meaning the walker uses
     * its standard input_variables-based assembly.
     *
     * Tangents for inputs that don't participate in the seed (e.g.
     * constant gamma/beta) are zero tensors of matching shape/dtype/device.
     *
     * @param walker_primals  Primals the walker assembled from input_variables.
     * @param walker_tangents Tangents the walker resolved (one per walker_primal).
     * @return Optional (primals, tangents) pair for the JVP rule. nullopt
     *         keeps the walker's default behaviour.
     */
    virtual auto jvp_pack_inputs_for_walker(
            const std::vector<Tensor>& walker_primals,
            const std::vector<Tensor>& walker_tangents) const
        -> std::optional<std::pair<std::vector<Tensor>, std::vector<Tensor>>> {
        (void)walker_primals; (void)walker_tangents;
        return std::nullopt;
    }

    /**
     * @brief Map a `saved_tensors()` index back to a forward output_idx,
     *        for the multi-output JVP walker.
     *
     * When the walker resolves a consumer's input back to a multi-output
     * producer, it matches the consumer's input data_ptr against the
     * producer's saved tensors and calls this method to translate the
     * matched saved-tensor index into the corresponding output_idx of the
     * forward op. Default returns 0 — appropriate for Functions whose
     * saved_tensors are all internal (e.g. LayerNormBackward saves
     * {x, mean, rstd, gamma}, where only mean (idx 1 → out 1) and rstd
     * (idx 2 → out 2) correspond to forward outputs). Subclasses whose
     * saved layout maps 1:1 to outputs (e.g. EighBackward saves {W, V} ↔
     * outputs (0, 1)) override accordingly.
     */
    virtual auto jvp_saved_tensor_to_output_idx(std::size_t saved_idx) const
        -> std::size_t {
        (void)saved_idx;
        return 0;
    }

    /**
     * @brief Set next functions in computation graph.
     *
     * Links this function to preceding functions for backpropagation.
     *
     * @param funcs Vector of gradient functions to chain to
     */
    auto set_next_functions(std::vector<std::shared_ptr<Function>> funcs) -> void;

    /**
     * @brief Get next functions in computation graph.
     *
     * @return Vector of chained gradient functions
     */
    auto next_functions() const -> const std::vector<std::shared_ptr<Function>>&;

    /**
     * @brief Set input variables for gradient accumulation.
     *
     * Stores Variables by value for gradient accumulation during backward pass.
     * The Variables' shared_ptr<VariableImpl> keeps the data alive even if the
     * Variable handle (temporary) is destroyed.
     *
     * @param inputs Vector of input Variables to track
     */
    auto set_input_variables(std::vector<Variable> inputs) -> void;

    /**
     * @brief Get input variables for gradient accumulation.
     *
     * @return Vector of tracked Variables
     */
    auto input_variables() const -> const std::vector<Variable>&;
    auto input_variables() -> std::vector<Variable>&;

    /**
     * @brief Reload offloaded saved tensors back to GPU.
     *
     * If activation offloading moved saved tensors to CPU during forward,
     * this reloads them to the original GPU device before backward.
     */
    void reload_saved_tensors() const;

private:
    /// Lockless implementation body for reload_saved_tensors(). Caller
    /// must hold offload_mutex_. Separated out so that saved_tensors(),
    /// which already holds the mutex, can reload without recursively
    /// locking the same non-recursive std::mutex.
    void reload_saved_tensors_locked() const;

public:

    /**
     * @brief Release saved tensors to free memory.
     *
     * Called after backward() to release GPU memory held by saved tensors.
     */
    void release_saved_tensors() {
        std::lock_guard lock(offload_mutex_);
        saved_tensors_.clear();
        saved_versions_.clear();
        // HH.5: also clear the offload-baseline vectors so a Function reused
        // after release_saved_tensors() doesn't inherit stale state.
        pre_offload_versions_.clear();
        was_offloaded_ = false;
        tensors_offloaded_.store(false, std::memory_order_relaxed);
    }

    /**
     * @brief Release saved Variables (graph-carrying) to free memory.
     *
     * audit-9 JJ.2: with `create_graph=true, retain_graph=false`, the engine
     * cleanup loop clears `input_variables_` / `next_functions_` but
     * previously left `saved_variables_` populated.  Each Function pinned
     * live Variables whose grad_fn chains across the entire higher-order
     * graph, so the second-order graph was never released even though the
     * first-order graph was.  Engine cleanup now calls this immediately
     * after `release_saved_tensors()` on the same Function.
     */
    void clear_saved_variables() {
        saved_variables_.clear();
    }

    /**
     * @brief Release op-specific ad-hoc state held by subclasses.
     *
     * audit-10 NN.6: `release_saved_tensors()` and `clear_saved_variables()`
     * only clear the common `saved_tensors_` / `saved_variables_` buckets.
     * Subclasses that hold ad-hoc Tensor / SparseTensor / Variable members
     * (e.g. CheckpointFunction's `cached_recompute_outputs_`,
     * FusedLinearReLUBackward's `relu_output_`, sparse Backwards' transposed
     * SparseTensor caches) used to keep that state alive until the Function
     * destructor ran.  Engine cleanup now calls this hook so per-Function
     * ad-hoc state is dropped together with the common buckets.
     *
     * Default impl: nothing.  Subclasses override.
     */
    virtual void release_op_specific_state() {}

    /**
     * @brief Validate that saved tensors have not been modified in-place.
     *
     * Checks version counters recorded by save_for_backward() against
     * current tensor versions. Throws if any tensor was modified in-place
     * after the forward pass.
     */
    void validate_saved_tensors() const;

    /// Assert that at least @p count tensors were saved. Throws if not.
    void require_saved_tensors(size_t count) const;

    /// Assert that at least @p count variables were saved. Throws if not.
    void require_saved_variables(size_t count) const;

    /**
     * @brief Get number of saved tensors.
     *
     * @return Count of tensors saved for backward pass
     */
    auto num_saved_tensors() const -> size_t { return saved_tensors_.size(); }

    /**
     * @brief Save tensors for backward pass.
     *
     * Stores tensors that will be needed to compute gradients.
     * Call this during forward pass.
     *
     * @param tensors Tensors to save for gradient computation
     */
    auto save_for_backward(std::vector<Tensor> tensors) -> void;

    /**
     * @brief Get saved tensors.
     *
     * Retrieves tensors saved during forward pass for use in backward pass.
     * Validates that saved tensors have not been modified in-place since save.
     *
     * @return Reference to saved tensors vector
     */
    auto saved_tensors() const -> const std::vector<Tensor>&;

    /**
     * @brief Save Variables for backward pass (preserves graph connections).
     *
     * When create_graph=true, forward operations should save Variables instead
     * of raw Tensors so that the backward pass can use Variable operations
     * and build a higher-order gradient graph.
     *
     * @param variables Variables to save for gradient computation
     */
    auto save_variables_for_backward(std::vector<Variable> variables) -> void;

    /**
     * @brief Get saved Variables.
     *
     * Retrieves Variables saved during forward pass. If no Variables were saved
     * but Tensors were saved, wraps the Tensors as Variables without grad tracking.
     *
     * @return Vector of saved Variables
     */
    auto saved_variables() const -> const std::vector<Variable>&;

    /**
     * @brief Check if Variables were saved for backward.
     *
     * @return true if save_variables_for_backward() was called
     */
    auto has_saved_variables() const -> bool { return !saved_variables_.empty(); }

    /**
     * @brief Set per-function offload policy.
     *
     * @param policy Offload policy (Inherit/Always/Never)
     * @param min_bytes Minimum tensor size in bytes to offload (0 = offload all)
     */
    void set_offload_policy(OffloadPolicy policy, size_t min_bytes = 0) {
        offload_policy_ = policy;
        offload_min_bytes_ = min_bytes;
    }

    /// Get the current offload policy
    auto offload_policy() const -> OffloadPolicy { return offload_policy_; }

    /// Check if a specific tensor should be offloaded based on policy and size
    auto should_offload(const Tensor& t) const -> bool;

protected:
    mutable std::vector<Tensor> saved_tensors_;                     ///< Tensors saved for backward
    mutable std::vector<uint64_t> saved_versions_;                  ///< Tensor versions at save time (for in-place detection)
    mutable std::vector<uint64_t> saved_view_base_versions_;       ///< View base versions at save time (0 if not a view)
    /// HH.5: pre-offload version baselines for the in-place detector.
    /// When tensors are offloaded to CPU and later reloaded with .to(),
    /// the post-reload Tensor has a fresh version counter. Storing the
    /// PRE-offload version here keeps the version-tracking assertion
    /// meaningful on the non-offloaded path; on the offloaded path it
    /// is documented as a best-effort signal (the GPU-side offload may
    /// legitimately bump the version through reload_saved_tensors_locked).
    mutable std::vector<uint64_t> pre_offload_versions_;
    mutable bool was_offloaded_{false};                             ///< HH.5: latch — true once any tensor in this Function has been offloaded
    mutable std::vector<Variable> saved_variables_;                 ///< Variables saved for backward (preserves graph for create_graph)
    mutable Device offloaded_device_{Device::cpu()};                ///< Original device of the first offloaded tensor (legacy; reload uses offloaded_devices_)
    mutable std::vector<Device> offloaded_devices_;                 ///< V.5: per-tensor original device; reload sends each tensor back to its own source device
    mutable std::atomic<bool> tensors_offloaded_{false};            ///< Whether saved tensors are on CPU due to offloading
    mutable std::mutex offload_mutex_;                              ///< Guards offload/reload of saved tensors
    std::vector<std::shared_ptr<Function>> next_functions_;         ///< Chained gradient functions
    std::vector<Variable> input_variables_;                          ///< Input variables for gradient accumulation (stored by value)
    OffloadPolicy offload_policy_{OffloadPolicy::Inherit};           ///< Per-function offload policy
    size_t offload_min_bytes_{0};                                    ///< Minimum tensor size to offload (0=all)

private:
    uint64_t id_;                                                    ///< Unique identifier (stable key for gradient accumulators)
    static std::atomic<uint64_t> next_id_;                           ///< Global counter for unique IDs
};

// ============================================================================
// Built-in Autograd Functions
// ============================================================================

/**
 * @brief Addition gradient function.
 *
 * Implements forward and backward for element-wise addition with broadcasting support.
 *
 * Forward: C = A + B (with broadcasting)
 * Backward: dL/dA = sum_reduce(dL/dC), dL/dB = sum_reduce(dL/dC)
 *
 * @note Gradients are sum-reduced along broadcasted dimensions to match input shapes.
 *
 * @code
 * Variable a(Tensor({2, 3}, DType::Float32, Device::cpu()), true);
 * Variable b(Tensor({3}, DType::Float32, Device::cpu()), true);
 * Variable c = a + b;  // Broadcasting: {2,3} + {3} -> {2,3}
 * // Backward: grad_b summed from {2,3} to {3}
 * @endcode
 */
class AddBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    // Audit A.2: opt-in to OpId-based pattern matching.
    auto op_id() const -> OpId override { return OpId::Add; }

    // Public for direct access from Variable operators
    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
};

/**
 * @brief Subtraction gradient function.
 *
 * Implements forward and backward for element-wise subtraction with broadcasting support.
 *
 * Forward: C = A - B (with broadcasting)
 * Backward: dL/dA = sum_reduce(dL/dC), dL/dB = -sum_reduce(dL/dC)
 *
 * @note Gradients are sum-reduced along broadcasted dimensions to match input shapes.
 */
class SubBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Sub; }

    // Public for direct access from Variable operators
    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
};

/**
 * @brief Multiplication gradient function.
 *
 * Implements forward and backward for element-wise multiplication with broadcasting support.
 *
 * Forward: C = A * B (with broadcasting)
 * Backward: dL/dA = sum_reduce(dL/dC * B), dL/dB = sum_reduce(dL/dC * A)
 *
 * @note Uses product rule for differentiation. Input tensors and shapes are saved for backward pass.
 */
class MulBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Mul; }

    // Public for direct access from Variable operators
    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
};

/**
 * @brief Division gradient function.
 *
 * Implements forward and backward for element-wise division.
 *
 * Forward: C = A / B
 * Backward: dL/dA = dL/dC / B, dL/dB = -dL/dC * A / (B^2)
 *
 * @note Uses quotient rule for differentiation. Input tensors are saved for backward pass.
 */
class DivBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Div; }

    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
};

/**
 * @brief Matrix multiplication gradient function.
 *
 * Implements forward and backward for matrix multiplication.
 *
 * Forward: C = A @ B
 * Backward: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
 *
 * @note Input matrices are saved for backward pass. Supports batched matrix multiplication.
 *
 * @code
 * Variable A(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable B(Tensor({4, 5}, DType::Float32, Device::cpu()), true);
 * Variable C = matmul(A, B);  // Shape: {3, 5}
 * @endcode
 */
class MatMulBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::MatMul; }

    // audit-10 MM.1: batched matmul broadcasts (e.g. (B,M,K) @ (K,N) =>
    // grad_b has shape (B,K,N) but actual b is (K,N)) — every other
    // arithmetic backward saves input shapes and reduces the broadcasted
    // grad; MatMul was exempt and produced wrong-shaped grad_b.  Captured
    // by the autograd::matmul forward wrapper before the Function is
    // attached.
    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
};

/**
 * @brief Fused linear layer gradient function.
 *
 * Implements fused forward and backward for y = x @ W.T + b
 * More efficient than separate matmul + add operations.
 *
 * Forward: y = x @ W.T + b (via linear_kernel)
 * Backward:
 *   dL/dx = dL/dy @ W
 *   dL/dW = dL/dy.T @ x
 *   dL/db = sum(dL/dy, dim=0)
 *
 * @note All three inputs (x, W, b) are saved for backward pass.
 *       Uses optimized MKL kernels internally.
 */
class LinearBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Linear; }
};

// Note: ReLUBackward is implemented in src/nn/activations/activations.cpp
// with full higher-order gradient support. No separate declaration needed here.

/**
 * @brief Sum reduction gradient function.
 *
 * Implements forward and backward for sum reduction operation.
 *
 * Forward: y = sum(x, dim, keepdim)
 * Backward: dL/dx = broadcast(dL/dy, original_shape)
 *
 * @note Gradient is broadcast back to original input shape. Original shape is saved.
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = x.sum(1);  // Sum along dimension 1, shape: {3}
 * // Gradient will broadcast back to {3, 4}
 * @endcode
 */
class SumBackward : public Function {
public:
    SumBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Sum; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        if (dim_.has_value()) attrs.set(AttrKey::Dim, *dim_);
        attrs.set(AttrKey::Keepdim, keepdim_);
        return attrs;
    }
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};;

/**
 * @brief Mean reduction gradient function.
 *
 * Implements forward and backward for mean reduction operation.
 *
 * Forward: y = mean(x, dim, keepdim)
 * Backward: dL/dx = broadcast(dL/dy / count, original_shape)
 *
 * @note Gradient is divided by element count then broadcast to original shape.
 */
class MeanBackward : public Function {
public:
    MeanBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Mean; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        if (dim_.has_value()) attrs.set(AttrKey::Dim, *dim_);
        attrs.set(AttrKey::Keepdim, keepdim_);
        return attrs;
    }
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};;

/**
 * @brief Natural logarithm gradient function.
 *
 * Forward: y = log(x)
 * Backward: dL/dx = dL/dy / x
 *
 * @note Input is saved for gradient computation. Undefined for x <= 0.
 */
class LogBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Log; }
};

/**
 * @brief Exponential gradient function.
 *
 * Forward: y = exp(x)
 * Backward: dL/dx = dL/dy * exp(x) = dL/dy * y
 *
 * @note Output is saved for efficient gradient computation.
 */
class ExpBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Exp; }
};

/**
 * @brief Complex conjugate gradient function.
 *
 * Forward: y = conj(z)
 * Backward (Wirtinger): dL/d(conj(z)) = conj(dL/dy)
 */
class ConjBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    // Conj is linear in its input (y = conj(z)); the second derivative is
    // structurally zero. backward_with_variables wraps without a grad_fn,
    // which is mathematically correct; declare it explicitly so the engine's
    // higher-order stub counter reports it accurately.
    auto is_higher_order_stub() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Conj; }
};

/**
 * @brief Real part gradient function.
 *
 * Forward: y = real(z)
 * Backward (Wirtinger): dL/d(conj(z)) = 0.5 * dL/dy (broadcast to complex)
 */
class RealBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    // Real-part projection is linear (grad propagates as 0.5 * grad cast to
    // the complex dtype); second derivative is structurally zero. Declare
    // the higher-order wrapping as a stub explicitly.
    auto is_higher_order_stub() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Real; }
    DType input_dtype_;
};

/**
 * @brief Imaginary part gradient function.
 *
 * Forward: y = imag(z)
 * Backward (Wirtinger): dL/d(conj(z)) = -0.5j * dL/dy (broadcast to complex)
 */
class ImagBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    // Imag-part projection is linear (grad propagates as a scaled-and-negated
    // cast into the complex dtype); second derivative is structurally zero.
    // Declare the higher-order wrapping as a stub explicitly.
    auto is_higher_order_stub() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Imag; }
    DType input_dtype_;
};

/**
 * @brief Negation gradient function.
 *
 * Forward: y = -x
 * Backward: dL/dx = -dL/dy
 *
 * @note Gradient is simply negated.
 */
class NegBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Neg; }
};

/**
 * @brief Log-softmax gradient function.
 *
 * Implements numerically stable log-softmax operation.
 *
 * Forward: y_i = x_i - log(sum(exp(x_j)))
 * Backward: dL/dx_i = dL/dy_i - exp(y_i) * sum(dL/dy_j)
 *
 * @note Output is saved for gradient computation. More numerically stable than log(softmax(x)).
 *
 * @code
 * Variable logits(Tensor({batch_size, num_classes}, DType::Float32, Device::cpu()), true);
 * Variable log_probs = log_softmax(logits, 1);  // Compute along class dimension
 * @endcode
 */
class LogSoftmaxBackward : public Function {
public:
    LogSoftmaxBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::LogSoftmax; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim_);
        return attrs;
    }
private:
    int64_t dim_;
};;

/**
 * @brief Softmax gradient function.
 *
 * Implements softmax activation with autograd support.
 *
 * Forward: y_i = exp(x_i) / sum(exp(x_j))
 * Backward: dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
 *
 * @note Output is saved for gradient computation.
 */
class SoftmaxBackward : public Function {
public:
    SoftmaxBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Softmax; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim_);
        return attrs;
    }
private:
    int64_t dim_;
};;

/**
 * @brief Absolute value gradient function.
 *
 * Forward: y = |x|
 * Backward: dL/dx = dL/dy * sign(x)
 *
 * @note Input is saved to compute sign. Gradient is undefined at x=0.
 */
class AbsBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Abs; }
};

/**
 * @brief Clamp gradient function.
 *
 * Forward: y = clamp(x, min, max) = max(min, min(x, max))
 * Backward: dL/dx = dL/dy * (min < x < max)
 *
 * @note Gradient passes through only for elements within bounds. Input is saved.
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = clamp(x, -1.0f, 1.0f);  // Clamp to [-1, 1]
 * @endcode
 */
class ClampBackward : public Function {
public:
    ClampBackward(double min, double max) : min_(min), max_(max) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Clamp; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Min, min_);
        attrs.set(AttrKey::Max, max_);
        return attrs;
    }
private:
    double min_;
    double max_;
};;

/**
 * @brief Max reduction gradient function.
 *
 * Forward: y = max(x, dim, keepdim)
 * Backward: dL/dx_i = dL/dy if x_i == max, else 0
 *
 * @note Gradient flows only to maximum elements. Input and indices are saved.
 * If multiple elements are tied for max, gradient is split among them.
 */
class MaxBackward : public Function {
public:
    MaxBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Max; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        if (dim_.has_value()) attrs.set(AttrKey::Dim, *dim_);
        attrs.set(AttrKey::Keepdim, keepdim_);
        return attrs;
    }
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};;

/**
 * @brief Median reduction gradient function.
 *
 * Forward: y = median(x, dim, keepdim)
 * Backward: dL/dx_i = dL/dy if x_i is the median element, else 0
 */
class MedianBackward : public Function {
public:
    MedianBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Median; }
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};

/**
 * @brief Mode reduction gradient function.
 *
 * Forward: y = mode(x, dim, keepdim)
 * Backward: dL/dx_i = dL/dy if x_i is the first occurrence of mode value, else 0
 */
class ModeBackward : public Function {
public:
    ModeBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Mode; }
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};

/**
 * @brief Reshape gradient function.
 *
 * Forward: y = reshape(x, shape)
 * Backward: dL/dx = reshape(dL/dy, input_shape)
 *
 * @note Original input shape is saved for gradient reshaping.
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = reshape(x, {12});  // Forward: {3, 4} -> {12}
 * // Backward: gradient reshaped from {12} back to {3, 4}
 * @endcode
 */
class ReshapeBackward : public Function {
public:
    /**
     * @param input_shape The forward input's shape (used by `backward` to
     *                    reshape the incoming gradient back to the original
     *                    shape).
     * @param output_shape The forward output's shape. Optional for
     *                     backward-mode AD (which doesn't need it), but
     *                     required by `saved_attributes()` so the
     *                     forward-mode (JVP) dispatch rule can be invoked
     *                     through this grad_fn (A.4 multi-op JVP traversal).
     *                     Falls back to empty if omitted.
     */
    ReshapeBackward(std::vector<int64_t> input_shape,
                    std::vector<int64_t> output_shape = {})
        : input_shape_(std::move(input_shape)),
          output_shape_(std::move(output_shape)) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Reshape; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        if (!output_shape_.empty()) {
            std::string s;
            for (size_t i = 0; i < output_shape_.size(); ++i) {
                if (i) s += ',';
                s += std::to_string(output_shape_[i]);
            }
            attrs.set(AttrKey::Shape, s);
        }
        return attrs;
    }
private:
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
};;

/**
 * @brief Permute gradient function.
 *
 * Forward: y = permute(x, dims)
 * Backward: dL/dx = permute(dL/dy, inverse_dims)
 *
 * @note Inverse permutation is computed and saved for gradient computation.
 *
 * @code
 * Variable x(Tensor({2, 3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = permute(x, {2, 0, 1});  // Forward: {2, 3, 4} -> {4, 2, 3}
 * // Backward: gradient permuted from {4, 2, 3} back to {2, 3, 4}
 * @endcode
 */
class PermuteBackward : public Function {
public:
    PermuteBackward(std::vector<int64_t> dims) : dims_(std::move(dims)) {
        // Compute inverse permutation
        inv_dims_.resize(dims_.size());
        for (size_t i = 0; i < dims_.size(); ++i) {
            inv_dims_[dims_[i]] = static_cast<int64_t>(i);
        }
    }
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Permute; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        std::string s;
        for (size_t i = 0; i < dims_.size(); ++i) {
            if (i) s += ',';
            s += std::to_string(dims_[i]);
        }
        attrs.set(AttrKey::Dims, s);
        return attrs;
    }
private:
    std::vector<int64_t> dims_;
    std::vector<int64_t> inv_dims_;
};;

/**
 * @brief Transpose gradient function.
 *
 * Forward: y = transpose(x, dim0, dim1)
 * Backward: dL/dx = transpose(dL/dy, dim0, dim1)
 *
 * @note Transpose is its own inverse, so backward uses same dimensions.
 */
class TransposeBackward : public Function {
public:
    TransposeBackward(int64_t dim0, int64_t dim1) : dim0_(dim0), dim1_(dim1) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Transpose; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim0, dim0_);
        attrs.set(AttrKey::Dim1, dim1_);
        return attrs;
    }
private:
    int64_t dim0_;
    int64_t dim1_;
};;

/**
 * @brief Roll gradient function.
 *
 * Forward: y = roll(x, shifts, dim)
 * Backward: dL/dx = roll(dL/dy, -shifts, dim)
 *
 * @note Rolling is reversed by rolling in the opposite direction.
 */
class RollBackward : public Function {
public:
    RollBackward(int64_t shifts, int64_t dim) : shifts_(shifts), dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "RollBackward"; }
    auto op_id() const -> OpId override { return OpId::Roll; }
    // V.9: expose Dim + Shift so vmap's dim_shifted_passthrough can
    // re-dispatch OpId::Roll on the batched tensor with the shifted
    // axis. Without this override, the passthrough_rule applied roll to
    // the batched tensor with the unbatched-rank `dim` — silently rolling
    // along the batch axis for batch_dim != 0.
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim_);
        attrs.set(AttrKey::Shift, shifts_);
        return attrs;
    }
private:
    int64_t shifts_;
    int64_t dim_;
};

/**
 * @brief Squeeze gradient function.
 *
 * Forward: y = squeeze(x, dim)
 * Backward: dL/dx = unsqueeze(dL/dy, dim)
 *
 * @note Squeezing dimension is saved for unsqueezing in backward pass.
 */
class SqueezeBackward : public Function {
public:
    /**
     * @param dim Dimension that was squeezed in the forward pass.
     * @param input_ndim Rank of the (pre-squeeze) input tensor. Used to
     *        normalise a negative @p dim into a non-negative index at
     *        construction so that both `backward` (Tensor path) and
     *        `backward_with_variables` (Variable path) operate on the
     *        same positive index. Pass 0 (default) when the caller is
     *        certain @p dim is already non-negative.
     */
    SqueezeBackward(int64_t dim, int64_t input_ndim = 0) {
        if (dim < 0 && input_ndim > 0) {
            dim_ = dim + input_ndim;
        } else {
            dim_ = dim;
        }
    }
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Squeeze; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim_);
        return attrs;
    }
private:
    int64_t dim_;
};;

/**
 * @brief Batch matrix multiplication gradient function.
 *
 * Implements forward and backward for batched matrix multiplication.
 *
 * Forward: C = bmm(A, B) where A: (batch, n, m), B: (batch, m, p), C: (batch, n, p)
 * Backward:
 *   dL/dA = bmm(dL/dC, permute(B, {0, 2, 1}))
 *   dL/dB = bmm(permute(A, {0, 2, 1}), dL/dC)
 *
 * @note Input tensors are saved for gradient computation using transposition.
 *
 * @code
 * Variable a(Tensor({32, 10, 20}, DType::Float32, Device::cpu()), true);
 * Variable b(Tensor({32, 20, 30}, DType::Float32, Device::cpu()), true);
 * Variable c = bmm(a, b);  // Uses BmmBackward internally
 * c.backward();  // Computes gradients w.r.t. a and b
 * @endcode
 */
class BmmBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Bmm; }

    // audit-11 RR.2: MM.1 sibling. If forward `bmm` ever permits a
    // broadcasted batch dim (e.g. (1,M,K) @ (B,K,N) -> (B,M,N)), the
    // chain-rule produces grads shaped as the broadcast output and must
    // be reduced back to each operand's saved shape — same pattern as
    // MatMulBackward's input_shape_a_/input_shape_b_.  Captured by the
    // autograd::bmm forward wrapper before the Function is attached.
    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
};;

/**
 * @brief Concatenation gradient function.
 *
 * Implements forward and backward for tensor concatenation.
 *
 * Forward: y = cat([x1, x2, ..., xn], dim)
 * Backward: Split dL/dy back to [dL/dx1, dL/dx2, ..., dL/dxn] along dim
 *
 * @note Split sizes and concatenation dimension are saved for gradient splitting.
 *
 * @code
 * Variable x1(Tensor({2, 3}, DType::Float32, Device::cpu()), true);
 * Variable x2(Tensor({2, 5}, DType::Float32, Device::cpu()), true);
 * Variable y = cat({x1, x2}, 1);  // Forward: {2,3} + {2,5} -> {2,8}
 * // Backward: gradient split from {2,8} back to {2,3} and {2,5}
 * @endcode
 */
class CatBackward : public Function {
public:
    CatBackward(std::vector<int64_t> split_sizes, int64_t dim)
        : split_sizes_(std::move(split_sizes)), dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Cat; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim_);
        return attrs;
    }
private:
    std::vector<int64_t> split_sizes_;  ///< Size of each input along concat dimension
    int64_t dim_;                        ///< Concatenation dimension
};;

/**
 * @brief Slice gradient function.
 *
 * Implements forward and backward for tensor slicing.
 *
 * Forward: y = slice(x, dim, start, end, step)
 * Backward: Scatter dL/dy back to positions in dL/dx, zeros elsewhere
 *
 * @note Original input shape and slice parameters are saved for gradient scattering.
 *
 * @code
 * Variable x(Tensor({10, 20}, DType::Float32, Device::cpu()), true);
 * Variable y = slice(x, 1, 5, 15, 2);  // Shape: {10, 5} - every 2nd element from 5 to 15
 * // Backward: gradient scattered back to original positions in {10, 20}
 * @endcode
 */
class SliceBackward : public Function {
public:
    SliceBackward(std::vector<int64_t> input_shape, int64_t dim, int64_t start, int64_t end, int64_t step)
        : input_shape_(std::move(input_shape)), dim_(dim), start_(start), end_(end), step_(step) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Slice; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim_);
        attrs.set(AttrKey::Start, start_);
        attrs.set(AttrKey::End, end_);
        attrs.set(AttrKey::Step, step_);
        return attrs;
    }
private:
    std::vector<int64_t> input_shape_;  ///< Original input shape
    int64_t dim_;                        ///< Slice dimension
    int64_t start_;                      ///< Start index
    int64_t end_;                        ///< End index (exclusive)
    int64_t step_;                       ///< Step size
};;

/**
 * @brief Bilinear upsample gradient function.
 *
 * Implements the linear adjoint of bilinear upsampling.
 *
 * Forward: y = upsample_bilinear(x, target_h, target_w)
 *   (handled by `nn::upsample_bilinear` via `OpId::Interpolate` dispatch.)
 * Backward: distribute dL/dy back to input pixels using bilinear weights —
 *   each output pixel contributes to its four nearest input pixels in
 *   proportion to its fractional source coordinates.
 *
 * Audit D3: the tensor-level `backward` dispatches `OpId::InterpolateBackward`
 * (CPU + CUDA device-resident scatter; ROCm/OneAPI/Vulkan throw honestly
 * until their kernels land) — no `.to(cpu)` round-trip. The Variable-level
 * `backward_with_variables` attaches an `UpsampleBilinearForwardAdjoint`
 * `grad_fn` so `create_graph=true` produces a real 2nd-order graph that
 * dispatches `OpId::Interpolate` for the next-level backward.
 */
class UpsampleBilinearBackward : public Function {
public:
    UpsampleBilinearBackward(int64_t input_h, int64_t input_w, int64_t output_h, int64_t output_w)
        : input_h_(input_h), input_w_(input_w), output_h_(output_h), output_w_(output_w) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    // Audit D3: no longer a stub — `backward_with_variables` builds a real
    // adjoint Function (`UpsampleBilinearForwardAdjoint`) that dispatches
    // `OpId::Interpolate` for higher-order grads.
    auto is_higher_order_stub() const -> bool override { return false; }
private:
    int64_t input_h_;   ///< Input height
    int64_t input_w_;   ///< Input width
    int64_t output_h_;  ///< Output height
    int64_t output_w_;  ///< Output width
};

/**
 * @brief Adjoint of `UpsampleBilinearBackward` — audit D3 higher-order.
 *
 * The bilinear-upsample backward is a linear scatter A^T (where A is the
 * forward bilinear upsample). When the engine asks for the backward of the
 * scatter (i.e. 2nd-order grads), the answer is the forward bilinear
 * upsample applied to the next-level gradient — exactly what
 * `OpId::Interpolate` already implements on every backend.
 *
 * This Function is created on the fly inside
 * `UpsampleBilinearBackward::backward_with_variables` and never directly
 * exercised from forward code, so its own `forward` throws.
 */
class UpsampleBilinearForwardAdjoint : public Function {
public:
    UpsampleBilinearForwardAdjoint(int64_t input_h, int64_t input_w,
                                    int64_t output_h, int64_t output_w)
        : input_h_(input_h), input_w_(input_w),
          output_h_(output_h), output_w_(output_w) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }
    auto name() const -> std::string override { return "UpsampleBilinearForwardAdjoint"; }
private:
    int64_t input_h_;
    int64_t input_w_;
    int64_t output_h_;
    int64_t output_w_;
};

// =========================================================================
// Activation Backward Functions
// =========================================================================

class SigmoidBackward_AG : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class TanhBackward_AG : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class GeluBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Gelu; }
};

class EluBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Elu; }
    auto saved_attributes() const -> OpAttributes override;
};;

class SeluBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Selu; }
};

class MishBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Mish; }
};

class LeakyReluBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::LeakyReLU; }
    auto saved_attributes() const -> OpAttributes override;
};;

class SoftplusBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Softplus; }
    auto saved_attributes() const -> OpAttributes override;
};;

// =========================================================================
// Element-wise Math Backward Functions
// =========================================================================

class SqrtBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Sqrt; }
};

class PowBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Pow; }
    auto saved_attributes() const -> OpAttributes override;
};;

class ReciprocalBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Reciprocal; }
};

class SinBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Sin; }
};

class CosBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Cos; }
};

class TanBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Tan; }
};

class AsinBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Asin; }
};

class AcosBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Acos; }
};

class AtanBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Atan; }
};

class SinhBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Sinh; }
};

class CoshBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Cosh; }
};

// =========================================================================
// Extended Math Backward Functions
// =========================================================================

class ErfBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Erf; }
};

class ErfcBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Erfc; }
};

class ErfInvBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::ErfInv; }
};

class GammaBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Gamma; }
};

class LgammaBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Lgamma; }
};

class DigammaBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Digamma; }
};

class BesselI0Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::BesselI0; }
};

class BesselI1Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::BesselI1; }
};

class SincBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Sinc; }
};

class Log2Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Log2; }
};

class Log10Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Log10; }
};

class Log1pBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Log1p; }
};

class Exp2Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Exp2; }
};

class Expm1Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Expm1; }
};

class Atan2Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }

    std::vector<int64_t> input_shape_y_;
    std::vector<int64_t> input_shape_x_;
    auto op_id() const -> OpId override { return OpId::Atan2; }
};

// =========================================================================
// Reduction Backward Functions
// =========================================================================

class MinBackward : public Function {
public:
    MinBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    std::optional<int64_t> dim_;
    bool keepdim_;
    auto op_id() const -> OpId override { return OpId::Min; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        if (dim_.has_value()) attrs.set(AttrKey::Dim, *dim_);
        attrs.set(AttrKey::Keepdim, keepdim_);
        return attrs;
    }
};;

class StdBackward : public Function {
public:
    StdBackward(std::optional<int64_t> dim, bool keepdim, bool unbiased = true)
        : dim_(dim), keepdim_(keepdim), unbiased_(unbiased) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    std::optional<int64_t> dim_;
    bool keepdim_;
    bool unbiased_;  // R.3: branch denom between N (false) and N-1 (true) in backward.
    auto op_id() const -> OpId override { return OpId::Std; }
    auto saved_attributes() const -> OpAttributes override {
        // R.8: emit Unbiased so vmap dim-shift / JVP-adapter see the real
        // forward semantics instead of defaulting to true silently.
        OpAttributes attrs;
        if (dim_.has_value()) attrs.set(AttrKey::Dim, *dim_);
        attrs.set(AttrKey::Keepdim, keepdim_);
        attrs.set(AttrKey::Unbiased, unbiased_);
        return attrs;
    }
};;

class VarBackward : public Function {
public:
    VarBackward(std::optional<int64_t> dim, bool keepdim, bool unbiased = true)
        : dim_(dim), keepdim_(keepdim), unbiased_(unbiased) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    std::optional<int64_t> dim_;
    bool keepdim_;
    bool unbiased_;  // R.3: branch denom between N (false) and N-1 (true) in backward.
    auto op_id() const -> OpId override { return OpId::Var; }
    auto saved_attributes() const -> OpAttributes override {
        // R.8: emit Unbiased so vmap dim-shift / JVP-adapter see the real
        // forward semantics instead of defaulting to true silently.
        OpAttributes attrs;
        if (dim_.has_value()) attrs.set(AttrKey::Dim, *dim_);
        attrs.set(AttrKey::Keepdim, keepdim_);
        attrs.set(AttrKey::Unbiased, unbiased_);
        return attrs;
    }
};;

class ProdBackward : public Function {
public:
    ProdBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    std::optional<int64_t> dim_;
    bool keepdim_;
    auto op_id() const -> OpId override { return OpId::Prod; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        if (dim_.has_value()) attrs.set(AttrKey::Dim, *dim_);
        attrs.set(AttrKey::Keepdim, keepdim_);
        return attrs;
    }
};;

class LogSumExpBackward : public Function {
public:
    LogSumExpBackward(int64_t dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    int64_t dim_;
    bool keepdim_;
    auto op_id() const -> OpId override { return OpId::LogSumExp; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim_);
        attrs.set(AttrKey::Keepdim, keepdim_);
        return attrs;
    }
};;

// =========================================================================
// Shape/Indexing Backward Functions
// =========================================================================

class UnsqueezeBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Unsqueeze; }
    auto saved_attributes() const -> OpAttributes override;
};;

class ExpandBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Expand; }
};

class DeviceTransferBackward : public Function {
public:
    Device source_device;  // Device to transfer gradients back to
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    // Device transfer is the identity on values (only the residency changes);
    // it is linear and the second derivative is structurally zero. The
    // backward_with_variables override wraps without a grad_fn — declare the
    // higher-order wrapping as a stub explicitly.
    auto is_higher_order_stub() const -> bool override { return true; }
};

class FlattenBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Flatten; }
};

class WhereBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Where; }

    // Saved at forward() time so the backward can broadcast-reduce the
    // unreduced grad back to the user's original (un-broadcasted) input shapes
    // (audit-5 Y.4). `tenzor::where` materialises the common broadcast shape;
    // the backward needs to sum across the broadcast axes to land at x.shape()
    // and y.shape() respectively, mirroring the BinaryOp backwards.
    std::vector<int64_t> input_shape_x_;
    std::vector<int64_t> input_shape_y_;
};

class GatherBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Gather; }
    auto saved_attributes() const -> OpAttributes override;
};;

class ScatterBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Scatter; }
    auto saved_attributes() const -> OpAttributes override;
};;

/**
 * @brief Backward for scatter_add: output[dim][index[i]] += src[i]
 *
 * Forward:  y = scatter_add(x, dim, index, src)
 * Backward: grad_x = grad_y (identity), grad_src = gather(grad_y, dim, index)
 *
 * Saved tensors:
 *   [0] = dim (scalar Float32, cast to int64_t)
 *   [1] = index tensor
 */
class ScatterAddBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::ScatterAdd; }
    // V.11: emit normalised non-negative dim for vmap consumers.
    auto saved_attributes() const -> OpAttributes override;
};

class IndexSelectBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::IndexSelect; }
    auto saved_attributes() const -> OpAttributes override;
};;

/**
 * @brief Backward for advanced (fancy) indexing.
 *
 * Forward:  y = x[idx0, idx1, ...]   (gather)
 * Backward: grad_x = zeros_like(x); grad_x.index_put(indices, grad_y, accumulate=true)
 *
 * Saved tensors:
 *   [0] = input shape (1D Int64)
 *   [1..N] = index tensors (0-element sentinel for null dims)
 *   N stored in num_indices_.
 */
class IndexBackward : public Function {
public:
    void set_num_indices(int64_t n) { num_indices_ = n; }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    // R.5: advanced (fancy) indexing backward uses `index_put` with
    // accumulation, for which we lack a Variable-level multi-tensor overload
    // whose backward closes the loop. Honest stub until that primitive
    // lands — engine counter + WARN_ONCE in backward_with_variables.
    auto is_higher_order_stub() const -> bool override { return true; }
    auto name() const -> std::string override { return "IndexBackward"; }

private:
    int64_t num_indices_ = 0;
};

class NarrowBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "NarrowBackward"; }
    // V.9: expose the saved `dim` so vmap can route NarrowBackward through
    // a per-slice loop fallback (no forward OpId::Narrow exists for the
    // generic dim_shifted_passthrough re-dispatch). The vmap rule reads
    // the dim out of saved_attributes() to decide how to invoke the
    // captured user function; without this override the dim was lost and
    // the passthrough_rule applied narrow to the batched tensor along
    // the unbatched dim — silently wrong for batch_dim != 0.
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        if (!saved_tensors_.empty()) {
            attrs.set(AttrKey::Dim, saved_tensors_[0].data<int64_t>()[0]);
        }
        return attrs;
    }
};

class FlipBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Flip; }
};

class RepeatBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto op_id() const -> OpId override { return OpId::Repeat; }
};

// =========================================================================
// Cumulative, Sorting, and Triangular Backward Functions
// =========================================================================

/**
 * @brief Cumulative sum gradient function.
 *
 * Forward: y = cumsum(x, dim)
 * Backward: dL/dx = flip(cumsum(flip(dL/dy, dim), dim), dim)
 *   i.e. reverse cumulative sum of the gradient
 */
class CumSumBackward : public Function {
public:
    explicit CumSumBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "CumSumBackward"; }
    auto op_id() const -> OpId override { return OpId::CumSum; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim_);
        return attrs;
    }
private:
    int64_t dim_;
};;

/**
 * @brief Cumulative product gradient function.
 *
 * Forward: y = cumprod(x, dim)
 * Backward: dL/dx = flip(cumsum(flip(y * dL/dy, dim), dim), dim) / x
 *   with zero-safe division using nan_to_num
 *
 * @note Saves input and output for backward.
 */
class CumProdBackward : public Function {
public:
    explicit CumProdBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "CumProdBackward"; }
    auto op_id() const -> OpId override { return OpId::CumProd; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim_);
        return attrs;
    }
private:
    int64_t dim_;
};;

/**
 * @brief TopK gradient function.
 *
 * Forward: (values, indices) = topk(x, k, dim)
 * Backward: scatter grad into zeros at saved indices positions
 *
 * @note Only values receive gradients; indices are non-differentiable.
 */
class TopKBackward : public Function {
public:
    TopKBackward(int64_t k, int64_t dim) : k_(k), dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "TopKBackward"; }
    auto op_id() const -> OpId override { return OpId::TopK; }
    // Audit J.1: expose k/dim so the vmap dim-shifted passthrough rule
    // can rebuild the forward attrs with a shifted dim.
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::K, k_);
        attrs.set(AttrKey::Dim, dim_);
        return attrs;
    }
private:
    int64_t k_;
    int64_t dim_;
};

/**
 * @brief Sort gradient function.
 *
 * Forward: (sorted_values, indices) = sort(x, dim)
 * Backward: scatter grad using inverse permutation of sort indices
 *
 * @note Only sorted values receive gradients; indices are non-differentiable.
 */
class SortBackward : public Function {
public:
    explicit SortBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SortBackward"; }
    auto op_id() const -> OpId override { return OpId::Sort; }
    // Audit J.1: expose dim so the vmap dim-shifted passthrough rule
    // can rebuild the forward attrs with a shifted dim.
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim_);
        return attrs;
    }
private:
    int64_t dim_;
};

/**
 * @brief Diag gradient function.
 *
 * Forward: y = diag(x, k)
 * Backward:
 *   If input was 1D (output 2D): dL/dx = diag(dL/dy, k) — extract diagonal
 *   If input was 2D (output 1D): dL/dx = diag(dL/dy, k) — construct diagonal matrix
 *
 * @note Saves input ndim to select correct backward path.
 */
class DiagBackward : public Function {
public:
    DiagBackward(int64_t input_ndim, int64_t diagonal) : input_ndim_(input_ndim), diagonal_(diagonal) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "DiagBackward"; }
    auto op_id() const -> OpId override { return OpId::Diag; }
    // V.9: expose Diagonal so vmap can route OpId::Diag (which has no
    // axis-shift semantics; diag flattens the last two axes regardless
    // of batch_dim) via a per-slice loop. Diag has no `dim` argument, so
    // dim_shifted_passthrough is not used — the saved_attributes only
    // serve to keep the OpId / attrs pair consistent with the forward.
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Diagonal, diagonal_);
        return attrs;
    }
private:
    int64_t input_ndim_;
    int64_t diagonal_;
};

/**
 * @brief Trace gradient function.
 *
 * Forward: y = trace(A)  (sum of diagonal elements)
 * Backward: dL/dA = dL/dy * eye(n)
 */
class TraceBackward : public Function {
public:
    explicit TraceBackward(int64_t n) : n_(n) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "TraceBackward"; }
    auto op_id() const -> OpId override { return OpId::Trace; }
private:
    int64_t n_;
};

/**
 * @brief Upper triangular gradient function.
 *
 * Forward: y = triu(x, k)
 * Backward: dL/dx = triu(dL/dy, k)
 */
class TriuBackward : public Function {
public:
    explicit TriuBackward(int64_t diagonal) : diagonal_(diagonal) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "TriuBackward"; }
    auto op_id() const -> OpId override { return OpId::Triu; }
    // V.9: expose Diagonal so vmap can re-dispatch OpId::Triu on the
    // batched tensor. Triu operates on the last two dims (no axis shift
    // is needed for batch_dim < ndim-2), so a plain re-dispatch with the
    // same diagonal value gives the correct per-slice result.
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Diagonal, diagonal_);
        return attrs;
    }
private:
    int64_t diagonal_;
};

/**
 * @brief Lower triangular gradient function.
 *
 * Forward: y = tril(x, k)
 * Backward: dL/dx = tril(dL/dy, k)
 */
class TrilBackward : public Function {
public:
    explicit TrilBackward(int64_t diagonal) : diagonal_(diagonal) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "TrilBackward"; }
    auto op_id() const -> OpId override { return OpId::Tril; }
    // V.9: expose Diagonal (see TriuBackward for rationale).
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Diagonal, diagonal_);
        return attrs;
    }
private:
    int64_t diagonal_;
};

// =========================================================================
// FFT Backward Functions
// =========================================================================

/**
 * @brief FFT gradient function.
 *
 * Forward: y = fft(x, n, dim, norm)
 * Backward: dL/dx = ifft(dL/dy, n, dim, norm)
 */
class FFTBackward : public Function {
public:
    FFTBackward(std::optional<int64_t> n, int64_t dim, std::string norm)
        : n_(n), dim_(dim), norm_(std::move(norm)) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "FFTBackward"; }
    auto op_id() const -> OpId override { return OpId::FFT; }
    // A.4 multi-op JVP walker: surface the forward op's attributes so the
    // registered JVP rule can rebuild the same op on the input tangent.
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        if (n_.has_value()) attrs.set(AttrKey::N, n_.value());
        attrs.set(AttrKey::Dim, dim_);
        attrs.set(AttrKey::Norm, std::string_view(norm_));
        return attrs;
    }
private:
    std::optional<int64_t> n_;
    int64_t dim_;
    std::string norm_;
};;

/**
 * @brief Inverse FFT gradient function.
 *
 * Forward: y = ifft(x, n, dim, norm)
 * Backward: dL/dx = fft(dL/dy, n, dim, norm)
 */
class IFFTBackward : public Function {
public:
    IFFTBackward(std::optional<int64_t> n, int64_t dim, std::string norm)
        : n_(n), dim_(dim), norm_(std::move(norm)) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "IFFTBackward"; }
    auto op_id() const -> OpId override { return OpId::IFFT; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        if (n_.has_value()) attrs.set(AttrKey::N, n_.value());
        attrs.set(AttrKey::Dim, dim_);
        attrs.set(AttrKey::Norm, std::string_view(norm_));
        return attrs;
    }
private:
    std::optional<int64_t> n_;
    int64_t dim_;
    std::string norm_;
};;

/**
 * @brief Real FFT gradient function.
 *
 * Forward: y = rfft(x, n, dim, norm)
 * Backward: dL/dx = irfft(dL/dy, n=signal_length, dim, norm)
 *
 * @note Saves original signal length for irfft reconstruction.
 */
class RFFTBackward : public Function {
public:
    RFFTBackward(int64_t signal_length, int64_t dim, std::string norm)
        : signal_length_(signal_length), dim_(dim), norm_(std::move(norm)) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "RFFTBackward"; }
    auto op_id() const -> OpId override { return OpId::RFFT; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        // signal_length_ is the saved-for-backward `n` (forward `n` value
        // post-default). Surface it as AttrKey::N so the JVP rule can
        // reproduce the same RFFT on the input tangent.
        attrs.set(AttrKey::N, signal_length_);
        attrs.set(AttrKey::Dim, dim_);
        attrs.set(AttrKey::Norm, std::string_view(norm_));
        return attrs;
    }
private:
    int64_t signal_length_;
    int64_t dim_;
    std::string norm_;
};;

/**
 * @brief Inverse real FFT gradient function.
 *
 * Forward: y = irfft(x, n, dim, norm)
 * Backward: dL/dx = rfft(dL/dy, dim, norm)
 */
class IRFFTBackward : public Function {
public:
    // R.7: n_orig is the frequency-bin count of the original irfft input
    // saved at forward time (= rfft input X.shape[dim] pre-irfft). The
    // adjoint of irfft(X, n) is rfft(grad, n_orig), and recomputing n_orig
    // from the time-domain output length is only correct when the user did
    // not pad/truncate (i.e. n was the default). Use -1 to mean "unset":
    // the backward then passes nullopt so rfft picks the natural default.
    IRFFTBackward(int64_t dim, std::string norm, int64_t n_orig = -1)
        : dim_(dim), norm_(std::move(norm)), n_orig_(n_orig) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "IRFFTBackward"; }
    auto op_id() const -> OpId override { return OpId::IRFFT; }
    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim_);
        attrs.set(AttrKey::Norm, std::string_view(norm_));
        if (n_orig_ >= 0) attrs.set(AttrKey::N, n_orig_);
        return attrs;
    }
    int64_t n_orig() const { return n_orig_; }
private:
    int64_t dim_;
    std::string norm_;
    int64_t n_orig_;  // R.7: original frequency-bin count of the rfft input.
};;


// =====================================================================
// Phase A.3 — STFT / ISTFT autograd. STFT and ISTFT are mutual
// adjoint-inverse linear operators (when the window satisfies COLA),
// so the gradient of one flows through the other with the same params.
// =====================================================================

class STFTBackward : public Function {
public:
    // Audit-7 EE.4: `input_dtype` records the forward STFT input's real dtype
    // (Float32 / Float64 / Float16 / BFloat16). The adjoint emits the
    // time-domain gradient in that same dtype instead of unconditionally F32.
    STFTBackward(int64_t n_fft, int64_t hop_length, int64_t win_length,
                 Tensor window, bool center, bool normalized, bool onesided,
                 int64_t signal_length, DType input_dtype)
        : n_fft_(n_fft), hop_length_(hop_length), win_length_(win_length),
          window_(std::move(window)), center_(center), normalized_(normalized),
          onesided_(onesided), signal_length_(signal_length),
          input_dtype_(input_dtype) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "STFTBackward"; }
    auto op_id() const -> OpId override { return OpId::STFT; }
private:
    int64_t n_fft_;
    int64_t hop_length_;
    int64_t win_length_;
    Tensor window_;
    bool center_;
    bool normalized_;
    bool onesided_;
    int64_t signal_length_;
    DType input_dtype_;  ///< Audit-7 EE.4: real-dtype of forward input.
};

class ISTFTBackward : public Function {
public:
    // Audit-7 EE.4: `input_dtype` records the forward ISTFT input's complex
    // dtype (Complex64 / Complex128). The adjoint emits the freq-domain
    // gradient in that same dtype instead of unconditionally Complex64.
    ISTFTBackward(int64_t n_fft, int64_t hop_length, int64_t win_length,
                  Tensor window, bool center, bool normalized, bool onesided,
                  DType input_dtype)
        : n_fft_(n_fft), hop_length_(hop_length), win_length_(win_length),
          window_(std::move(window)), center_(center), normalized_(normalized),
          onesided_(onesided), input_dtype_(input_dtype) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "ISTFTBackward"; }
    auto op_id() const -> OpId override { return OpId::ISTFT; }
private:
    int64_t n_fft_;
    int64_t hop_length_;
    int64_t win_length_;
    Tensor window_;
    bool center_;
    bool normalized_;
    bool onesided_;
    DType input_dtype_;  ///< Audit-7 EE.4: complex-dtype of forward input.
};

// =========================================================================
// Linear Algebra Backward Functions
// =========================================================================

/**
 * @brief Determinant gradient function.
 *
 * Forward: y = det(A)
 * Backward: dL/dA = dL/dy * det(A) * A^{-T}
 *
 * @note Saves det result and inverse for backward computation.
 */
class DetBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "DetBackward"; }
};

/**
 * @brief Matrix inverse gradient function.
 *
 * Forward: Y = A^{-1}
 * Backward: dL/dA = -Y^T @ dL/dY @ Y^T
 *
 * @note Saves inverse result for backward computation.
 */
class InvBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "InvBackward"; }
};

/**
 * @brief Linear solve gradient function.
 *
 * Forward: X = solve(A, B) where AX = B
 * Backward:
 *   dL/dB = solve(A^T, dL/dX)
 *   dL/dA = -dL/dB @ X^T
 *
 * @note Saves A and solution X for backward computation.
 */
class SolveBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SolveBackward"; }
};

/**
 * @brief LU-solve gradient function (audit-2026-05-03 Phase 8).
 * Treats LU and pivots as fixed; differentiates only w.r.t. B.
 */
class LUSolveBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "LUSolveBackward"; }
};

/**
 * @brief Non-symmetric eigendecomposition gradient (audit-2026-05-03 Phase 8).
 * Backward only supports the real-eigenvalue path.
 */
class EigBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "EigBackward"; }
};

/**
 * @brief Cholesky decomposition gradient function.
 *
 * Forward: L = cholesky(A) where A = L @ L^T
 * Backward: dL/dA = L^{-T} @ phi(L^T @ dL/dL) @ L^{-1}
 *           where phi(X) = tril(X) with diagonal halved
 *
 * @note Saves L for backward computation.
 */
/**
 * @brief Cholesky solve gradient function.
 *
 * Forward: X = cholesky_solve(B, L) where A = L @ L^T, A @ X = B
 * Backward: grad_B = cholesky_solve(grad_X, L)
 *           grad_L = -tril(L^{-T} @ (grad_X @ X^T + X @ grad_X^T) @ L^{-1})
 *
 * @note Saves X and L for backward computation.
 */
class CholeskySolveBackward : public Function {
public:
    CholeskySolveBackward(bool upper = false) : upper_(upper) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    // Audit B.3: real Variable-level higher-order backward.
    // The closed-form backward (grad_B = cholesky_solve(grad_X, L),
    // grad_L = -tril(cholesky_solve(grad_X @ X^T + X @ grad_X^T, L)))
    // is composed entirely of Variable-level ops, so reverse-mode
    // autograd through these gives the correct second-order grad.
    auto backward_with_variables(std::vector<Variable> grad_outputs)
        -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "CholeskySolveBackward"; }
private:
    bool upper_;
};

class CholeskyBackward : public Function {
public:
    CholeskyBackward(bool upper = false) : upper_(upper) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "CholeskyBackward"; }
private:
    bool upper_;
};

/**
 * @brief LU decomposition gradient function.
 *
 * Forward: (L, U, pivots) = lu(A)  where P @ L @ U = A
 * Backward: dA = P^T @ (dL @ U + L @ dU)
 *
 * @note Saves L, U for backward computation. Pivots are non-differentiable.
 */
class LUBackward : public Function {
public:
    // audit-2026-05-03 — output_slot identifies which of {L=0, U=1} this
    // backward instance accumulates a gradient for. tenzor's autograd engine
    // collapses all per-output gradients of a multi-output function into a
    // single accumulator entry; with L and U sharing the (N, N) shape that
    // collapse erases per-output information. Using one Function instance
    // per output means each accumulator slot stays distinct.
    explicit LUBackward(int output_slot = -1) : output_slot_(output_slot) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "LUBackward"; }
private:
    int output_slot_;  // 0=L, 1=U; -1 = legacy combined (kept for compat)
};;

/**
 * @brief SVD gradient function.
 *
 * Forward: (U, S, Vh) = svd(A)
 * Backward: Uses the SVD backward formula involving F matrix
 *
 * @note Saves U, S, Vh for backward computation. Full backward is complex.
 */
class SvdBackward : public Function {
public:
    // audit-2026-05-03 — output_slot identifies which of {U=0, S=1, Vh=2}
    // this backward instance handles. The engine collapses per-output
    // accumulator entries, mixing U/S/Vh contributions; per-output Function
    // instances keep them distinct.
    SvdBackward(bool full_matrices = true, int output_slot = -1)
        : full_matrices_(full_matrices), output_slot_(output_slot) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SvdBackward"; }
private:
    bool full_matrices_;
    int output_slot_;
};

/**
 * @brief QR decomposition gradient function.
 *
 * Forward: (Q, R) = qr(A)
 * Backward: Uses the QR backward formula
 *   dL/dA = (dL/dQ + Q @ copyltu(Q^T @ dL/dQ - dL/dR^T @ R^{-T})) @ R^{-T}
 *
 * @note Saves Q and R for backward computation.
 */
class QrBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "QrBackward"; }
};

/**
 * @brief Symmetric eigendecomposition gradient function.
 *
 * Forward: (W, V) = eigh(A) where A = V @ diag(W) @ V^T
 * Backward:
 *   F_{ij} = 1/(w_j - w_i) for i != j, 0 on diagonal
 *   dL/dA = V @ (F * (V^T @ dL/dV) + diag(dL/dW)) @ V^T
 *
 * @note Saves eigenvalues W and eigenvectors V for backward computation.
 */
class EighBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "EighBackward"; }
    // A.4 multi-output JVP walker integration. eigh returns (W, V); the
    // OpId here points at the matching forward op so the walker can look
    // up the multi-output JVP rule.
    auto op_id() const -> OpId override { return OpId::LinalgEigh; }

    // EighBackward saves {W, V} (see `eigh` in src/autograd/ops.cpp), which
    // maps 1:1 to outputs (0, 1) of the forward op. Walker uses this to
    // route downstream consumers of V to output_idx=1 (so they get dV, not
    // the default dW).
    auto jvp_saved_tensor_to_output_idx(std::size_t saved_idx) const
        -> std::size_t override {
        // saved[0] = W → output 0; saved[1] = V → output 1.
        return saved_idx < 2 ? saved_idx : 0;
    }
};;;

/**
 * @brief Eigenvalues-only gradient function for symmetric matrices.
 *
 * Forward: W = eigvalsh(A)
 * Backward: dL/dA = V @ diag(dL/dW) @ V^T
 *           where V is computed via eigh during forward
 *
 * @note Saves eigenvectors V for backward computation (computed during forward).
 */
class EigvalshBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "EigvalshBackward"; }
};

/**
 * @brief Matrix norm gradient function.
 *
 * Forward: y = norm(A, ord)
 * Backward (Frobenius): dL/dA = dL/dy * A / norm(A)
 *
 * @note Saves input and norm result for backward computation.
 *       Currently only supports Frobenius norm backward.
 */
class NormBackward_Linalg : public Function {
public:
    NormBackward_Linalg(const std::string& ord = "fro") : ord_(ord) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "NormBackward_Linalg"; }
private:
    std::string ord_;
};

/**
 * @brief Slogdet gradient function.
 *
 * Forward: (sign, logabsdet) = slogdet(A)
 * Backward: dL/dA = dL/d(logabsdet) * A^{-T}
 *           (sign gradient is zero since sign is discrete)
 *
 * @note Saves inverse for backward computation.
 */
class SlogdetBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SlogdetBackward"; }
};

// =========================================================================
// Sparse Backward Functions
// =========================================================================

/**
 * @brief Backward function for sparse-dense matrix multiplication (spmm).
 *
 * For Y = S @ D where S is sparse (M,K) and D is dense (K,N):
 * - grad_D = S^T @ grad_Y  (uses sparse::spmm with transposed sparse matrix)
 * - The sparse matrix S is not differentiated (treated as constant).
 *
 * The transposed sparse matrix S^T is stored directly as a SparseTensor,
 * avoiding the memory cost of converting to dense.
 */
class SpMMBackward : public Function {
public:
    /// Set the transposed sparse matrix for backward computation.
    void set_sparse_transposed(SparseTensor sparse_t) {
        sparse_transposed_.emplace(std::move(sparse_t));
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SpMMBackward"; }

    // audit-10 NN.6: drop ad-hoc transposed-CSR cache on cleanup.
    void release_op_specific_state() override { sparse_transposed_.reset(); }

private:
    std::optional<SparseTensor> sparse_transposed_;  ///< S^T stored in sparse format
};

/**
 * @brief Backward function for sparse-dense matrix-vector multiplication (spmv).
 *
 * For y = S @ v where S is sparse (M,K) and v is dense (K,):
 * - grad_v = S^T @ grad_y  (uses sparse::spmv with transposed sparse matrix)
 * - The sparse matrix S is not differentiated (treated as constant).
 *
 * The transposed sparse matrix S^T is stored directly as a SparseTensor,
 * avoiding the memory cost of converting to dense.
 */
class SpMVBackward : public Function {
public:
    /// Set the transposed sparse matrix for backward computation.
    void set_sparse_transposed(SparseTensor sparse_t) {
        sparse_transposed_.emplace(std::move(sparse_t));
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SpMVBackward"; }

    // audit-10 NN.6: drop ad-hoc transposed-CSR cache on cleanup.
    void release_op_specific_state() override { sparse_transposed_.reset(); }

private:
    std::optional<SparseTensor> sparse_transposed_;  ///< S^T stored in sparse format
};

/**
 * @brief Backward function for sparse-dense addition.
 *
 * For Y = S + D (sparse + dense = dense):
 * - grad_D = grad_Y  (gradient passes through to the dense input)
 * - The sparse matrix S is not differentiated (treated as constant).
 */
class SparseAddBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SparseAddBackward"; }
    auto op_id() const -> OpId override { return OpId::SparseAdd; }
};

/**
 * @brief Backward function for sparse-sparse matrix multiplication (SpGEMM).
 *
 * For C = A @ B where A and B are sparse:
 * - grad_A = grad_C @ B^T (sparse)
 * - grad_B = A^T @ grad_C (sparse)
 * Both A and B are treated as constants (sparse matrices are not differentiable
 * through the sparsity pattern), so gradients flow through dense conversions.
 */
class SpGEMMBackward : public Function {
public:
    void set_sparse_a_transposed(SparseTensor at) { sparse_a_t_.emplace(std::move(at)); }
    void set_sparse_b_transposed(SparseTensor bt) { sparse_b_t_.emplace(std::move(bt)); }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SpGEMMBackward"; }

    // audit-10 NN.6: drop ad-hoc sparse transposed CSR caches on cleanup.
    void release_op_specific_state() override {
        sparse_a_t_.reset();
        sparse_b_t_.reset();
        sparse_grad_accumulated_ = false;
    }

private:
    /// audit-10 NN.5: accumulate the sparse-grad slot side-effect once.
    /// `backward_with_variables` previously delegated to `backward()` for the
    /// SparseAdam accumulation, then rebuilt Variable-level dense grads — a
    /// second `backward(retain_graph=true)` re-entered and double-accumulated.
    auto accumulate_sparse_into_inputs(const Tensor& grad_c) -> void;

    std::optional<SparseTensor> sparse_a_t_;  ///< A^T in sparse format
    std::optional<SparseTensor> sparse_b_t_;  ///< B^T in sparse format
    bool sparse_grad_accumulated_{false};
};

/**
 * @brief Backward function for sparse triangular solve.
 *
 * For x = L^{-1} @ b (lower triangular solve):
 * - grad_b = L^{-T} @ grad_x
 * - L is not differentiated (treated as constant sparse matrix).
 */
class SparseTriSolveBackward : public Function {
public:
    void set_sparse_l_transposed(SparseTensor lt) { sparse_l_t_.emplace(std::move(lt)); }
    void set_upper(bool u) { upper_ = u; }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SparseTriSolveBackward"; }

    // audit-10 NN.6: drop ad-hoc transposed-CSR cache on cleanup.
    void release_op_specific_state() override { sparse_l_t_.reset(); }

private:
    std::optional<SparseTensor> sparse_l_t_;  ///< L^T (or U^T) in sparse format
    bool upper_{false};
};


/**
 * @brief Backward function for SparseTensor -> dense conversion.
 *
 * For Y = sparse.to_dense() where the sparse input has a fixed CSR pattern:
 * - The gradient with respect to the (conceptually differentiable) values of
 *   the sparse tensor is grad_dense projected onto the original sparsity
 *   pattern: positions outside the pattern contribute no gradient because
 *   they were structural zeros in the forward.
 * - Returns the gradient as a dense Tensor (same shape as the sparse tensor),
 *   with entries outside the sparsity pattern zeroed.
 *
 * The sparsity pattern is captured at forward time as a 0/1 mask (saved via
 * save_for_backward()) so the backward is a single elementwise multiply.
 */
class SparseToDenseBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    // sparse_to_dense is linear in the sparse values; the backward formula
    // is grad_input = grad_output * mask, also linear. Second derivative is
    // structurally zero, so the no-grad_fn wrap in backward_with_variables
    // is mathematically correct — declare the higher-order stub explicitly.
    auto is_higher_order_stub() const -> bool override { return true; }
    auto name() const -> std::string override { return "SparseToDenseBackward"; }
    auto op_id() const -> OpId override { return OpId::SparseToDense; }
};

/**
 * @brief Backward function for dense -> SparseTensor conversion.
 *
 * For Y = dense_to_sparse(D, mask) (or by threshold), only entries selected
 * by the mask (i.e. originally nonzero / above threshold) contribute to the
 * sparse output. The gradient w.r.t. D therefore zeros out entries that
 * were discarded:
 *     grad_D = grad_Y_dense * mask
 * where grad_Y_dense is the dense projection of the sparse gradient and
 * `mask` is the 0/1 selection tensor saved at forward time.
 */
class DenseToSparseBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    // dense_to_sparse is linear in the dense input (mask-selection); the
    // backward is grad_input = grad_output_dense * mask, also linear. Second
    // derivative is structurally zero, so the no-grad_fn wrap in
    // backward_with_variables is mathematically correct — declare the
    // higher-order stub explicitly.
    auto is_higher_order_stub() const -> bool override { return true; }
    auto name() const -> std::string override { return "DenseToSparseBackward"; }
    auto op_id() const -> OpId override { return OpId::DenseToSparse; }
};

/**
 * @brief Backward function for sparse softmax (CSR, per-row).
 *
 * Forward: Y = sparse_softmax(X) along the column axis of a CSR matrix;
 * zeros stay zero (structural zeros are treated as -inf inputs and never
 * participate in normalisation).
 *
 * Backward: standard softmax backward, restricted to the sparsity pattern:
 *     grad_X = (grad_Y - sum(grad_Y * Y, dim=col, keepdim=true)) * Y
 * Since both grad_Y and Y are zero outside the pattern, this naturally
 * stays inside the pattern. We save Y (the forward output) and the CSR
 * structure so the per-row reduction can be performed on the dense view.
 */
class SparseSoftmaxBackward : public Function {
public:
    /// Save the CSR structure (crow + col + shape) of the output for backward.
    void set_output_pattern(SparseTensor out) { output_sparse_.emplace(std::move(out)); }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SparseSoftmaxBackward"; }
    auto op_id() const -> OpId override { return OpId::SparseSoftmax; }

    // audit-10 NN.6: drop ad-hoc sparse output cache on cleanup.
    void release_op_specific_state() override { output_sparse_.reset(); }

private:
    std::optional<SparseTensor> output_sparse_;  ///< Y = softmax(X) saved for backward.
};

/**
 * @brief Backward function for sparse log-softmax (CSR, per-row).
 *
 * Forward: Y = sparse_log_softmax(X). Like SparseSoftmax, zeros remain zero
 * structurally and do not enter the per-row normalisation.
 *
 * Backward: standard log-softmax backward, restricted to the sparsity
 * pattern:
 *     grad_X = grad_Y - exp(Y) * sum(grad_Y, dim=col, keepdim=true)
 * We save Y (log-softmax output) and the CSR structure.
 */
class SparseLogSoftmaxBackward : public Function {
public:
    /// Save the CSR structure (crow + col + shape) of the output for backward.
    void set_output_pattern(SparseTensor out) { output_sparse_.emplace(std::move(out)); }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SparseLogSoftmaxBackward"; }
    auto op_id() const -> OpId override { return OpId::SparseLogSoftmax; }

    // audit-10 NN.6: drop ad-hoc sparse output cache on cleanup.
    void release_op_specific_state() override { output_sparse_.reset(); }

private:
    std::optional<SparseTensor> output_sparse_;  ///< Y = log_softmax(X) saved for backward.
};

// ============================================================================
// Custom Op Autograd
// ============================================================================

/// Forward declaration of custom op backward function type
using CustomBackwardFn = std::function<std::vector<Tensor>(
    std::span<const Tensor> saved_tensors,
    std::span<const Tensor> grad_outputs)>;

/// Audit D2: Variable-level custom-op backward.
/// Must match the definition in `tenzor/ops/custom_op.hpp` — declared here
/// too so `CustomOpBackward` below can use it without a circular include.
using CustomBackwardVariableFn = std::function<std::vector<Variable>(
    std::span<const Variable> saved_variables,
    std::span<const Variable> grad_outputs)>;

/**
 * @brief Autograd function for user-registered custom operations.
 *
 * Created by autograd::dispatch_custom_op() when inputs require gradients.
 * Stores the user-provided backward function and calls it during backpropagation.
 *
 * @see register_custom_op_with_backward()
 */
/**
 * @brief Fused Linear+ReLU backward pass.
 *
 * Combines the backward passes of MatMul and ReLU into a single operation.
 * Given forward: z = ReLU(W @ x + b), the backward is:
 *   grad_input = W^T @ (grad_output * (z > 0))
 * where the ReLU mask is computed from the forward output.
 *
 * This saves one intermediate tensor (the pre-ReLU activation) and
 * one kernel launch compared to separate MatMul + ReLU backward.
 */
class FusedLinearReLUBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "FusedLinearReLUBackward"; }

    void set_relu_output(Tensor output) { relu_output_ = std::move(output); }
    auto op_id() const -> OpId override { return OpId::FusedLinearReLU; }

    // audit-10 NN.6: drop the ad-hoc relu mask source on engine cleanup.
    void release_op_specific_state() override { relu_output_ = Tensor(); }

private:
    Tensor relu_output_;  // Output of ReLU (for computing mask in backward)
};

class CustomOpBackward : public Function {
public:
    explicit CustomOpBackward(CustomBackwardFn backward_fn)
        : backward_fn_(std::move(backward_fn)) {}

    /// Audit D2: variant that also accepts a Variable-level backward. When
    /// `var_backward_fn` is non-null, `backward_with_variables` invokes it
    /// and preserves the autograd graph; otherwise the op honestly reports
    /// `is_higher_order_stub() = true` and the engine's disconnection
    /// machinery fires under Warn/Error mode.
    CustomOpBackward(CustomBackwardFn backward_fn,
                     CustomBackwardVariableFn var_backward_fn)
        : backward_fn_(std::move(backward_fn)),
          var_backward_fn_(std::move(var_backward_fn)) {}

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    /// D2: stub iff no Variable-level backward was registered.
    auto is_higher_order_stub() const -> bool override {
        return !static_cast<bool>(var_backward_fn_);
    }
    auto name() const -> std::string override { return "CustomOpBackward"; }

private:
    CustomBackwardFn backward_fn_;
    CustomBackwardVariableFn var_backward_fn_;  // may be empty
};

/**
 * @brief Grid sample gradient function.
 *
 * Computes gradients for grid_sample w.r.t. both input and grid.
 * Uses bilinear interpolation gradient formulas.
 */
class GridSampleBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    // R.5: first-order backward is an opaque per-backend kernel; no
    // Variable-level composition exists, so higher-order is an honest
    // stub. WARN_ONCE fires in backward_with_variables.
    auto is_higher_order_stub() const -> bool override { return true; }
    auto name() const -> std::string override { return "GridSampleBackward"; }

    std::string mode_;
    std::string padding_mode_;
    bool align_corners_{false};
    auto op_id() const -> OpId override { return OpId::GridSample; }
};

/**
 * @brief Affine grid gradient function.
 *
 * Computes gradient for affine_grid w.r.t. theta.
 */
class AffineGridBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    // R.5: first-order backward is an opaque per-backend kernel; honest
    // higher-order stub. WARN_ONCE fires in backward_with_variables.
    auto is_higher_order_stub() const -> bool override { return true; }
    auto name() const -> std::string override { return "AffineGridBackward"; }

    std::vector<int64_t> size_;
    bool align_corners_{false};
    auto op_id() const -> OpId override { return OpId::AffineGrid; }
};

// =========================================================================
// New Op Backward Functions (Phase 7)
// =========================================================================

// --- Element-wise binary ops ---

class LogAddExpBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "LogAddExpBackward"; }

    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
    auto op_id() const -> OpId override { return OpId::LogAddExp; }
};

class LogAddExp2Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "LogAddExp2Backward"; }

    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
    auto op_id() const -> OpId override { return OpId::LogAddExp2; }
};

class XLogYBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "XLogYBackward"; }

    std::vector<int64_t> input_shape_x_;
    std::vector<int64_t> input_shape_y_;
    auto op_id() const -> OpId override { return OpId::XLogY; }
};

// --- Element-wise unary ops ---

class I0eBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "I0eBackward"; }
    auto op_id() const -> OpId override { return OpId::I0e; }
};

class I1eBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "I1eBackward"; }
    auto op_id() const -> OpId override { return OpId::I1e; }
};

class EntrBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "EntrBackward"; }
    auto op_id() const -> OpId override { return OpId::Entr; }
};

class SphericalBesselJ0Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SphericalBesselJ0Backward"; }
    auto op_id() const -> OpId override { return OpId::SphericalBesselJ0; }
};


// Phase 12 (audit-2026-05-03) — Bessel J0/J1/Y0/Y1 and Zeta backwards.
// These add Variable-level autograd to ops that previously had Tensor-only forward.
class BesselJ0Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "BesselJ0Backward"; }
    auto op_id() const -> OpId override { return OpId::BesselJ0; }
};

class BesselJ1Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "BesselJ1Backward"; }
    auto op_id() const -> OpId override { return OpId::BesselJ1; }
};

class BesselY0Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "BesselY0Backward"; }
    auto op_id() const -> OpId override { return OpId::BesselY0; }
};

class BesselY1Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "BesselY1Backward"; }
    auto op_id() const -> OpId override { return OpId::BesselY1; }
};

// Zeta(s, q): differentiable wrt q via d/dq zeta(s,q) = -s * zeta(s+1, q).
// s is treated as a non-differentiable parameter (its derivative requires
// the digamma-zeta identity which has no closed form).
class ZetaBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "ZetaBackward"; }
    auto op_id() const -> OpId override { return OpId::Zeta; }
};


// BetaInc(a, b, x) = I_x(a, b) regularised incomplete beta function.
// Differentiable wrt x: dI/dx = x^(a-1) (1-x)^(b-1) / B(a, b).
// a, b receive zero gradients (their derivatives need digamma/polylog terms
// which are out of scope here and a, b are usually fixed parameters).
class BetaIncBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "BetaIncBackward"; }
    auto op_id() const -> OpId override { return OpId::BetaInc; }
};

// --- Statistical/special ops ---

class NdtrBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "NdtrBackward"; }
    auto op_id() const -> OpId override { return OpId::Ndtr; }
};

class LogNdtrBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "LogNdtrBackward"; }
    auto op_id() const -> OpId override { return OpId::LogNdtr; }
};

class MultigammalnBackward : public Function {
public:
    explicit MultigammalnBackward(int64_t p) : p_(p) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "MultigammalnBackward"; }
    auto op_id() const -> OpId override { return OpId::Multigammaln; }
private:
    int64_t p_;
};

// --- Reduction ops ---

class CosineSimilarityBackward : public Function {
public:
    explicit CosineSimilarityBackward(int64_t dim = 1, double eps = 1e-8)
        : dim_(dim), eps_(eps) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "CosineSimilarityBackward"; }
    auto op_id() const -> OpId override { return OpId::CosineSimilarity; }
private:
    int64_t dim_;
    double eps_;
};

class RenormBackward : public Function {
public:
    RenormBackward(double p, int64_t dim, double maxnorm)
        : p_(p), dim_(dim), maxnorm_(maxnorm) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "RenormBackward"; }
    auto op_id() const -> OpId override { return OpId::Renorm; }
private:
    double p_;
    int64_t dim_;
    double maxnorm_;
};

// --- Linalg ops ---

class CholeskyInverseBackward : public Function {
public:
    explicit CholeskyInverseBackward(bool upper = false) : upper_(upper) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "CholeskyInverseBackward"; }
    auto op_id() const -> OpId override { return OpId::CholeskyInverse; }
private:
    bool upper_;
};

class LinalgLDLFactorBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    // Audit B.3: real Variable-level higher-order backward.
    // The Smith (1995) closed-form L^{-T} (S + R) L^{-1} is rewritten
    // via `linalg::inv(L)` so the entire backward composes from
    // Variable-level ops (tril, transpose, matmul, diag, diag_embed,
    // inv) and reverse-mode autograd produces correct 2nd-order grads.
    auto backward_with_variables(std::vector<Variable> grad_outputs)
        -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "LinalgLDLFactorBackward"; }
    auto op_id() const -> OpId override { return OpId::LinalgLDLFactor; }
};

class LinalgLDLSolveBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "LinalgLDLSolveBackward"; }
    auto op_id() const -> OpId override { return OpId::LinalgLDLSolve; }
};

class LinalgHouseholderBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    // Audit item B.3: previously declared STRUCTURAL_ZERO_STUB, which
    // claimed `is_higher_order_stub() == true` even though the first-order
    // backward (function_new_ops.cpp:807) is a real closed-form
    // implementation.  The stub macro was a lie that also installed a
    // passthrough `backward_with_variables` — wrong for a non-linear op.
    // Removed; create_graph=true through Householder now raises a clear
    // "higher-order not supported" error from the base class default,
    // which is the honest contract for an op that has not yet shipped
    // its second-order backward.
    auto name() const -> std::string override { return "LinalgHouseholderBackward"; }
    auto op_id() const -> OpId override { return OpId::LinalgHouseholder; }
};

class TensorInvBackward : public Function {
public:
    explicit TensorInvBackward(int64_t ind = 2) : ind_(ind) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "TensorInvBackward"; }
    auto op_id() const -> OpId override { return OpId::TensorInv; }
private:
    int64_t ind_;
};

class TensorSolveBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "TensorSolveBackward"; }
    auto op_id() const -> OpId override { return OpId::TensorSolve; }
};

class LinalgVectorNormBackward : public Function {
public:
    LinalgVectorNormBackward(double ord = 2.0, std::vector<int64_t> dim = {}, bool keepdim = false)
        : ord_(ord), dim_(std::move(dim)), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "LinalgVectorNormBackward"; }
    auto op_id() const -> OpId override { return OpId::LinalgVectorNorm; }
private:
    double ord_;
    std::vector<int64_t> dim_;
    bool keepdim_;
};

class LinalgMatrixNormBackward : public Function {
public:
    explicit LinalgMatrixNormBackward(double ord = 2.0) : ord_(ord) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "LinalgMatrixNormBackward"; }
    auto op_id() const -> OpId override { return OpId::LinalgMatrixNorm; }
private:
    double ord_;
};

class LinalgVecdotBackward : public Function {
public:
    explicit LinalgVecdotBackward(int64_t dim = -1) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "LinalgVecdotBackward"; }
    auto op_id() const -> OpId override { return OpId::LinalgVecdot; }
private:
    int64_t dim_;
};

class AsStridedBackward : public Function {
public:
    AsStridedBackward(std::vector<int64_t> input_shape, std::vector<int64_t> size,
                      std::vector<int64_t> stride, std::optional<int64_t> storage_offset)
        : input_shape_(std::move(input_shape)), size_(std::move(size)),
          stride_(std::move(stride)), storage_offset_(storage_offset) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    // Audit B.3: AsStrided is a linear strided gather; its backward is
    // a linear scatter-add. The higher-order backward of that scatter
    // is the same strided gather applied to the second-order grad,
    // which is exactly what `tenzor::as_strided` (Variable overload)
    // does — and that overload installs its own grad_fn so further
    // differentiation keeps composing. No structural-zero stub.
    auto backward_with_variables(std::vector<Variable> grad_outputs)
        -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }
    auto name() const -> std::string override { return "AsStridedBackward"; }
    auto op_id() const -> OpId override { return OpId::AsStrided; }
private:
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> size_;
    std::vector<int64_t> stride_;
    std::optional<int64_t> storage_offset_;
};

/**
 * @brief ViewAsReal gradient function.
 *
 * Forward: y = view_as_real(x)   [Complex -> Real with trailing dim 2]
 * Backward: dL/dx = view_as_complex(dL/dy)
 */
class ViewAsRealBackward : public Function {
public:
    ViewAsRealBackward() = default;
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    // Audit B.3: ViewAsReal is a linear isomorphism with a closed-form
    // higher-order backward (view_as_complex on the Variable). No stub.
    auto backward_with_variables(std::vector<Variable> grad_outputs)
        -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }
    auto name() const -> std::string override { return "ViewAsRealBackward"; }
};

/**
 * @brief ViewAsComplex gradient function.
 *
 * Forward: y = view_as_complex(x)   [Real with trailing dim 2 -> Complex]
 * Backward: dL/dx = view_as_real(dL/dy)
 */
class ViewAsComplexBackward : public Function {
public:
    ViewAsComplexBackward() = default;
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    // Audit B.3: ViewAsComplex is a linear isomorphism with a closed-form
    // higher-order backward (view_as_real on the Variable). No stub.
    auto backward_with_variables(std::vector<Variable> grad_outputs)
        -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }
    auto name() const -> std::string override { return "ViewAsComplexBackward"; }
};

// ============================================================================
// Attention autograd Functions
// ============================================================================
//
// Per docs/internals/attention-contract.md:
//   - FlashAttention forward returns (output, lse, seed, offset). Backward
//     consumes (dO, Q, K, V, O, L, seed, offset) and the same scalar context
//     (scale, causal, dropout_p) the forward used. The Function persists L,
//     seed, offset across the forward/backward boundary so the user-facing
//     output is just the differentiable Variable.
//   - FusedAttention is the same shape but no dropout; it shares the same
//     backward kernel so a separate backward class is unnecessary.
//   - FlexAttention carries a serializable score_mod OpId and an optional
//     block_mask; its backward dispatches FlexAttentionBackward.
//
// Backends that don't yet honor the 4-tuple contract (M3-M7 work) trigger the
// composed-ops fallback in backward via the saved-tensor count check.

/**
 * @brief FlashAttention gradient function.
 *
 * Forward: (output, lse, seed, offset) = flash_attention(Q, K, V; scale, causal, dropout_p, is_training)
 * Backward: (dQ, dK, dV) via OpId::FlashAttentionBackward dispatch using saved (Q, K, V, O, L, seed, offset).
 */
class FlashAttentionBackward : public Function {
public:
    FlashAttentionBackward(double scale, bool causal, float dropout_p, bool is_training)
        : scale_(scale), causal_(causal), dropout_p_(dropout_p), is_training_(is_training) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    // Audit B.3: real Variable-level higher-order backward.
    // Mirrors the existing Float64 composed-ops fast path in
    // `flash_attention()` — softmax-attention backward expressed as
    // matmul/softmax/transpose on Variables, so reverse-mode autograd
    // through these gives the correct second-order grad. Dropout is
    // unsupported for higher-order (training-time stochasticity is
    // not differentiable); the override falls through to the
    // structural-zero stub when dropout_p > 0.
    auto backward_with_variables(std::vector<Variable> grad_outputs)
        -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "FlashAttentionBackward"; }
    auto op_id() const -> OpId override { return OpId::FlashAttention; }
private:
    double scale_;
    bool causal_;
    float dropout_p_;
    bool is_training_;
};

/**
 * @brief FusedAttention gradient function.
 *
 * Forward: (output, lse) = fused_attention(Q, K, V; scale, causal, use_cudnn_sdpa)
 * Backward: shares the FlashAttentionBackward kernel since the math is identical
 *           (no dropout in fused). Saves Q, K, V, O, L for backward.
 */
class FusedAttentionBackward : public Function {
public:
    FusedAttentionBackward(double scale, bool causal, bool use_cudnn_sdpa)
        : scale_(scale), causal_(causal), use_cudnn_sdpa_(use_cudnn_sdpa) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    // Audit B.3: real Variable-level higher-order backward.
    // Same closed form as FlashAttentionBackward (math is identical,
    // no dropout). Reverse-mode autograd through the composed
    // matmul/softmax/transpose chain produces correct 2nd-order grads.
    auto backward_with_variables(std::vector<Variable> grad_outputs)
        -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "FusedAttentionBackward"; }
    auto op_id() const -> OpId override { return OpId::FusedAttention; }
private:
    double scale_;
    bool causal_;
    bool use_cudnn_sdpa_;
};

/**
 * @brief FlexAttention gradient function.
 *
 * Forward: (output, lse) = flex_attention(Q, K, V; score_mod_id, block_mask?)
 * Backward: (dQ, dK, dV) via OpId::FlexAttentionBackward dispatch using saved (Q, K, V, O, L).
 */
class FlexAttentionBackward : public Function {
public:
    FlexAttentionBackward(double scale, int64_t score_mod_id, bool has_block_mask)
        : scale_(scale), score_mod_id_(score_mod_id), has_block_mask_(has_block_mask) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    // Audit B.3: real Variable-level higher-order backward.
    // Same closed form as FlashAttentionBackward with score_mod applied
    // to scores before softmax. Only score_mod_id == 0 (identity, i.e.
    // standard attention) is supported for higher-order; other score
    // mods are user-supplied OpIds without a Variable-level pipeline.
    auto backward_with_variables(std::vector<Variable> grad_outputs)
        -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "FlexAttentionBackward"; }
    auto op_id() const -> OpId override { return OpId::FlexAttention; }
private:
    double scale_;
    int64_t score_mod_id_;
    bool has_block_mask_;
};

// ============================================================================
// Audit E.7 — typed non-differentiable Function wrappers.
//
// These ops have a well-defined forward but are *intrinsically*
// non-differentiable in their primary input (histogram counts,
// bin assignments, sort positions, discrete samples). Their Function
// wrappers exist so the autograd graph carries the dependency (no
// silent un-tracked Variable returned) and so any caller who hits
// backward() through one of these gets a typed `NonDifferentiable`
// exception with the op name — instead of the previous mystery
// "Function 'unknown' has no backward".
// ============================================================================

class HistcBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "HistcBackward"; }
    auto op_id() const -> OpId override { return OpId::Histc; }
};

class BincountBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "BincountBackward"; }
    auto op_id() const -> OpId override { return OpId::Bincount; }
};

class SearchSortedBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "SearchSortedBackward"; }
    auto op_id() const -> OpId override { return OpId::SearchSorted; }
};

/**
 * @brief Multinomial (weighted-index) sampling. Returns int64 indices
 *        whose gradient w.r.t. probs is the delta-of-Diracs distribution —
 *        intrinsically non-differentiable.  Use Gumbel-softmax or a
 *        straight-through estimator for differentiable approximations.
 */
class MultinomialSampleBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "MultinomialSampleBackward"; }
    auto op_id() const -> OpId override { return OpId::Multinomial; }
};

/**
 * @brief Bernoulli sampling. Returns {0, 1} draws whose Jacobian w.r.t.
 *        the input probability is undefined (the output value doesn't
 *        change continuously with `probs`).  Use STE or a relaxed
 *        Bernoulli (Concrete distribution) for differentiable variants.
 */
class BernoulliSampleBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "BernoulliSampleBackward"; }
    auto op_id() const -> OpId override { return OpId::Bernoulli; }
};

/**
 * @brief argmax / argmin. Returns integer indices of extrema; the
 *        Jacobian w.r.t. the input tensor is the delta-of-Diracs at
 *        the maximiser (zero everywhere except at the argmax, where
 *        it's undefined).  Use soft-argmax (softmax-weighted indices)
 *        for a differentiable approximation.
 */
class ArgmaxBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "ArgmaxBackward"; }
    auto op_id() const -> OpId override { return OpId::ArgMax; }
};

class ArgminBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "ArgminBackward"; }
    auto op_id() const -> OpId override { return OpId::ArgMin; }
};

/**
 * @brief Bucketize. Returns bucket-index Tensor (Int64) of input values
 *        in sorted-boundaries via binary search; non-differentiable
 *        for the same reason as searchsorted (integer position output).
 */
class BucketizeBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "BucketizeBackward"; }
    auto op_id() const -> OpId override { return OpId::Bucketize; }
};

/**
 * @brief argsort. Returns the permutation indices that would sort the
 *        input. Integer output ⇒ no Jacobian w.r.t. the input values.
 *        For gradient-aware sorting, use `sort` (which returns a
 *        differentiable values+indices tuple — only the values branch
 *        carries gradient via gather-then-scatter).
 */
class ArgSortBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "ArgSortBackward"; }
    auto op_id() const -> OpId override { return OpId::ArgSort; }
};

// NOTE: ModeBackward is already declared earlier in this file with a real
// "scatter to first-occurrence-of-mode-value" implementation; that one
// stands. We do not re-add a NonDifferentiable stub for mode().

// ============================================================================
// Audit E.7 continuation: additional Function wrappers for OpIds that
// previously dispatched through the kernel registry without a corresponding
// autograd Function. The differentiable ones below have closed-form backward;
// the non-differentiable ones throw tenzor::NonDifferentiable from backward()
// with a message that names the offending op.
// ============================================================================

/**
 * @brief square(x) = x * x. Backward: grad * 2 * x.
 *        Saves input for backward.
 */
class SquareBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "SquareBackward"; }
    auto op_id() const -> OpId override { return OpId::Square; }
};

/**
 * @brief rsqrt(x) = 1/sqrt(x). Backward: grad * (-0.5 * y^3),
 *        where y = rsqrt(x). Saves output for backward.
 */
class RsqrtBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "RsqrtBackward"; }
    auto op_id() const -> OpId override { return OpId::Rsqrt; }
};

/**
 * @brief deg2rad(x) = x * (pi / 180). Backward: grad * (pi / 180).
 *        No saved tensors — slope is a constant.
 */
class Deg2RadBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "Deg2RadBackward"; }
    auto op_id() const -> OpId override { return OpId::Deg2Rad; }
};

/**
 * @brief rad2deg(x) = x * (180 / pi). Backward: grad * (180 / pi).
 *        No saved tensors — slope is a constant.
 */
class Rad2DegBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "Rad2DegBackward"; }
    auto op_id() const -> OpId override { return OpId::Rad2Deg; }
};

/**
 * @brief logit(x) = log(x / (1 - x)). Backward: grad / (x * (1 - x)).
 *        Saves input for backward. Undefined outside (0, 1).
 */
class LogitBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "LogitBackward"; }
    auto op_id() const -> OpId override { return OpId::Logit; }
};

/**
 * @brief nan_to_num replaces NaN/+Inf/-Inf with finite scalars.
 *        Wherever the input was finite, the output is the input value, so
 *        the local Jacobian is 1; wherever the input was NaN/Inf, the
 *        output is a constant and the local Jacobian is 0.
 *        Backward: grad * isfinite(input).cast(input.dtype).
 */
class NanToNumBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "NanToNumBackward"; }
    auto op_id() const -> OpId override { return OpId::NanToNum; }
};

/**
 * @brief Heaviside step function. Piecewise constant ⇒ derivative is the
 *        Dirac delta at the jump and zero elsewhere. Non-differentiable.
 *        Use a smooth surrogate (sigmoid scaled by temperature) if you
 *        need gradients.
 */
class HeavisideBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "HeavisideBackward"; }
    auto op_id() const -> OpId override { return OpId::Heaviside; }
};

/**
 * @brief signbit returns a Bool tensor (true for negative values, including
 *        -0.0). Output is discrete ⇒ non-differentiable.
 */
class SignbitBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "SignbitBackward"; }
    auto op_id() const -> OpId override { return OpId::Signbit; }
};

/**
 * @brief frexp decomposes x = mantissa * 2^exponent. The exponent branch
 *        is integer (non-differentiable); the mantissa branch is piecewise
 *        constant in exponent intervals, so its derivative is also a sum
 *        of Diracs at the dyadic boundaries. Treat as non-differentiable.
 */
class FrexpBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "FrexpBackward"; }
    auto op_id() const -> OpId override { return OpId::Frexp; }
};

/**
 * @brief histogram (fixed-bin counts + edges). Integer count tensor is
 *        non-differentiable in the input values, same reason as histc.
 */
class HistogramBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "HistogramBackward"; }
    auto op_id() const -> OpId override { return OpId::Histogram; }
};

// ============================================================================
// Audit E.7 continuation (batch 2): another 10 OpIds.
// Pattern matches the first batch (SquareBackward / LogitBackward /
// HeavisideBackward). Differentiable Functions have closed-form backward;
// non-differentiable Functions throw tenzor::NonDifferentiable from
// backward() with a message that names the op and explains the reason.
// ============================================================================

/**
 * @brief sign(x). Gradient is zero almost everywhere (delta at the origin
 *        is ignored). We return a zero tensor of the same shape/dtype as
 *        the input, matching the standard treatment used by PyTorch / JAX
 *        so that downstream graph traversal still receives a well-typed
 *        gradient buffer instead of an exception.
 */
class SignBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "SignBackward"; }
    auto op_id() const -> OpId override { return OpId::Sign; }
};

/**
 * @brief hypot(x, y) = sqrt(x*x + y*y). Backward:
 *        d/dx = x / hypot(x, y) * grad
 *        d/dy = y / hypot(x, y) * grad
 *        Saves both inputs and the output to avoid recomputing the norm.
 */
class HypotBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "HypotBackward"; }
    auto op_id() const -> OpId override { return OpId::Hypot; }

    std::vector<int64_t> input_shape_x_;
    std::vector<int64_t> input_shape_y_;
};

/**
 * @brief copysign(magnitude, sign_src) returns |magnitude| * sign(sign_src).
 *        d/d(magnitude) = sign(sign_src). d/d(sign_src) = 0 almost
 *        everywhere. Saves the sign source for backward.
 */
class CopysignBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "CopysignBackward"; }
    auto op_id() const -> OpId override { return OpId::Copysign; }

    std::vector<int64_t> input_shape_mag_;
    std::vector<int64_t> input_shape_sign_;
};

/**
 * @brief xlog1py(x, y) = x * log1p(y), with x*log1p(y)=0 when x=0
 *        regardless of y. Backward:
 *        d/dx = log1p(y) * grad
 *        d/dy = x / (1 + y) * grad
 *        Saves both inputs.
 */
class Xlog1pyBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "Xlog1pyBackward"; }
    auto op_id() const -> OpId override { return OpId::Xlog1py; }

    std::vector<int64_t> input_shape_x_;
    std::vector<int64_t> input_shape_y_;
};

/**
 * @brief addcmul(a, b, c, value) = a + value * b * c. Backward:
 *        d/da = grad
 *        d/db = value * c * grad
 *        d/dc = value * b * grad
 *        Saves b, c and the scalar value.
 */
class AddcmulBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "AddcmulBackward"; }
    auto op_id() const -> OpId override { return OpId::Addcmul; }

    double value_ = 1.0;
    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
    std::vector<int64_t> input_shape_c_;
};

/**
 * @brief addcdiv(a, b, c, value) = a + value * b / c. Backward:
 *        d/da = grad
 *        d/db = (value / c) * grad
 *        d/dc = -(value * b / (c * c)) * grad
 *        Saves b, c and the scalar value.
 */
class AddcdivBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "AddcdivBackward"; }
    auto op_id() const -> OpId override { return OpId::Addcdiv; }

    double value_ = 1.0;
    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
    std::vector<int64_t> input_shape_c_;
};

/**
 * @brief floor(x). Piecewise-constant ⇒ derivative is zero except at the
 *        integer jumps where it is a Dirac comb. Non-differentiable.
 */
class FloorBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "FloorBackward"; }
    auto op_id() const -> OpId override { return OpId::Floor; }
};

/**
 * @brief ceil(x). Piecewise-constant ⇒ derivative is zero except at the
 *        integer jumps where it is a Dirac comb. Non-differentiable.
 */
class CeilBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "CeilBackward"; }
    auto op_id() const -> OpId override { return OpId::Ceil; }
};

/**
 * @brief isnan(x). Returns a Bool tensor — discrete output. Non-differentiable.
 */
class IsNanBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "IsNanBackward"; }
    auto op_id() const -> OpId override { return OpId::IsNan; }
};

/**
 * @brief logical_and(a, b). Bool inputs, Bool output ⇒ non-differentiable.
 */
class LogicalAndBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "LogicalAndBackward"; }
    auto op_id() const -> OpId override { return OpId::LogicalAnd; }
};

// ============================================================================
// Audit E.7 continuation (batch 3): another 10 OpIds.
// Pattern matches the first two batches. Differentiable Functions have
// closed-form backward; non-differentiable Functions throw
// tenzor::NonDifferentiable from backward() with a message that names the
// op and explains why it is not differentiable.
// ============================================================================

/**
 * @brief addmm(input, mat1, mat2, beta, alpha) = beta*input + alpha*(mat1 @ mat2).
 *        d/d(input) = beta * grad  (reduced for broadcast)
 *        d/d(mat1)  = alpha * grad @ mat2^T
 *        d/d(mat2)  = alpha * mat1^T @ grad
 *        Saves mat1, mat2 plus the scalar coefficients.
 */
class AddmmBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "AddmmBackward"; }
    auto op_id() const -> OpId override { return OpId::Addmm; }

    double beta_ = 1.0;
    double alpha_ = 1.0;
    std::vector<int64_t> input_shape_input_;
    std::vector<int64_t> input_shape_mat1_;
    std::vector<int64_t> input_shape_mat2_;
};

/**
 * @brief addmv(input, mat, vec, beta, alpha) = beta*input + alpha*(mat @ vec).
 *        d/d(input) = beta * grad  (reduced for broadcast)
 *        d/d(mat)   = alpha * outer(grad, vec)
 *        d/d(vec)   = alpha * mat^T @ grad
 *        Saves mat, vec plus the scalar coefficients.
 */
class AddmvBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "AddmvBackward"; }
    auto op_id() const -> OpId override { return OpId::Addmv; }

    double beta_ = 1.0;
    double alpha_ = 1.0;
    std::vector<int64_t> input_shape_input_;
    std::vector<int64_t> input_shape_mat_;
    std::vector<int64_t> input_shape_vec_;
};

/**
 * @brief baddbmm(input, batch1, batch2, beta, alpha)
 *          = beta*input + alpha*(batch1 @ batch2)  (batched).
 *        d/d(input)  = beta * grad  (reduced for broadcast)
 *        d/d(batch1) = alpha * grad @ batch2^T   (per-batch transpose)
 *        d/d(batch2) = alpha * batch1^T @ grad
 *        Saves batch1, batch2 plus the scalar coefficients.
 */
class BaddbmmBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "BaddbmmBackward"; }
    auto op_id() const -> OpId override { return OpId::Baddbmm; }

    double beta_ = 1.0;
    double alpha_ = 1.0;
    std::vector<int64_t> input_shape_input_;
    std::vector<int64_t> input_shape_batch1_;
    std::vector<int64_t> input_shape_batch2_;
};

/**
 * @brief nansum(x, dim, keepdim): sum treating NaN as zero. Backward is
 *        the standard sum backward (broadcast grad back to input shape)
 *        with NaN positions masked to zero so the synthetic zero
 *        contribution does not produce a non-zero local Jacobian. Saves
 *        the input to recompute the NaN mask without storing it.
 */
class NansumBackward : public Function {
public:
    NansumBackward(std::optional<int64_t> dim, bool keepdim)
        : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "NansumBackward"; }
    auto op_id() const -> OpId override { return OpId::Nansum; }
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};

/**
 * @brief tile(input, reps): tile input along each dim (right-aligned reps,
 *        padded with 1s on the left). Backward splits the output's tiled
 *        dims into (reps[i], orig_shape[i]) interleaved pairs and sums
 *        over the reps[i] axes, mirroring repeat's backward. Saves the
 *        input's original (unpadded) shape and the user-supplied reps.
 */
class TileBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "TileBackward"; }
    auto op_id() const -> OpId override { return OpId::Tile; }

    std::vector<int64_t> original_shape_;
    std::vector<int64_t> reps_;
};

/**
 * @brief count_nonzero(x). Integer output (a count) — non-differentiable.
 */
class CountNonzeroBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "CountNonzeroBackward"; }
    auto op_id() const -> OpId override { return OpId::CountNonzero; }
};

/**
 * @brief isinf(x). Bool output — metadata, non-differentiable.
 */
class IsInfBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "IsInfBackward"; }
    auto op_id() const -> OpId override { return OpId::IsInf; }
};

/**
 * @brief bitwise_and(a, b). Integer/bool inputs and output — discrete.
 *        Non-differentiable.
 */
class BitwiseAndBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "BitwiseAndBackward"; }
    auto op_id() const -> OpId override { return OpId::BitwiseAnd; }
};

/**
 * @brief round(x). Piecewise-constant — gradient is zero almost everywhere
 *        with Dirac jumps at half-integers. Non-differentiable.
 */
class RoundBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "RoundBackward"; }
    auto op_id() const -> OpId override { return OpId::Round; }
};

/**
 * @brief eq(a, b). Bool output (a == b) — discrete, non-differentiable.
 */
class EqBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "EqBackward"; }
    auto op_id() const -> OpId override { return OpId::Eq; }
};

// ============================================================================
// Audit E.7 continuation (batch 4): another 10 OpIds. Knocks out the rest of
// the boolean-comparison family (ne / lt / le / gt / ge), three integer bit
// ops (bitwise_or / bitwise_xor / bitwise_not), one Bool introspection op
// (isfinite) and one differentiable op (logcumsumexp).
//
// Same pattern as batches 1–3: differentiable Functions compute a closed-form
// backward; non-differentiable Functions throw tenzor::NonDifferentiable with
// a message that names the op and explains why no smooth gradient exists.
// The five comparisons share a single reason string referencing the full
// eq/ne/lt/le/gt/ge family.
// ============================================================================

/**
 * @brief ne(a, b). Bool output (a != b) — discrete comparison, non-differentiable.
 *        See EqBackward for the shared comparison-family reason.
 */
class NeBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "NeBackward"; }
    auto op_id() const -> OpId override { return OpId::Ne; }
};

/**
 * @brief lt(a, b). Bool output (a < b) — discrete, non-differentiable.
 */
class LtBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "LtBackward"; }
    auto op_id() const -> OpId override { return OpId::Lt; }
};

/**
 * @brief le(a, b). Bool output (a <= b) — discrete, non-differentiable.
 */
class LeBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "LeBackward"; }
    auto op_id() const -> OpId override { return OpId::Le; }
};

/**
 * @brief gt(a, b). Bool output (a > b) — discrete, non-differentiable.
 */
class GtBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "GtBackward"; }
    auto op_id() const -> OpId override { return OpId::Gt; }
};

/**
 * @brief ge(a, b). Bool output (a >= b) — discrete, non-differentiable.
 */
class GeBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "GeBackward"; }
    auto op_id() const -> OpId override { return OpId::Ge; }
};

/**
 * @brief bitwise_or(a, b). Integer / bool inputs and output — discrete bit-level op.
 *        Non-differentiable.
 */
class BitwiseOrBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "BitwiseOrBackward"; }
    auto op_id() const -> OpId override { return OpId::BitwiseOr; }
};

/**
 * @brief bitwise_xor(a, b). Integer / bool inputs and output. Non-differentiable.
 */
class BitwiseXorBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "BitwiseXorBackward"; }
    auto op_id() const -> OpId override { return OpId::BitwiseXor; }
};

/**
 * @brief bitwise_not(x). Unary integer / bool bit complement. Non-differentiable.
 */
class BitwiseNotBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "BitwiseNotBackward"; }
    auto op_id() const -> OpId override { return OpId::BitwiseNot; }
};

/**
 * @brief isfinite(x). Bool output classifying x as finite vs. (±inf, NaN).
 *        Pure metadata — non-differentiable.
 */
class IsFiniteBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "IsFiniteBackward"; }
    auto op_id() const -> OpId override { return OpId::IsFinite; }
};

/**
 * @brief logcumsumexp(x, dim): y = log(cumsum(exp(x), dim)).
 *        Backward: grad_x[i] = sum_{j >= i} grad_y[j] * exp(x[i] - y[j])
 *                = exp(x) * reverse_cumsum_along_dim(grad_y * exp(-y))
 *        where reverse_cumsum_along_dim(z) = flip(cumsum(flip(z, dim), dim), dim).
 *        Saves x and y to recompute the per-position weights without storing
 *        the full softmax window. dim is normalised at backward-call time
 *        against the saved x's rank.
 */
class LogcumsumexpBackward : public Function {
public:
    explicit LogcumsumexpBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "LogcumsumexpBackward"; }
    auto op_id() const -> OpId override { return OpId::Logcumsumexp; }
private:
    int64_t dim_;
};

// ============================================================================
// Audit E.7 continuation (batch 5): 10 more OpIds.
//   non-differentiable: IsPosInf, IsNegInf, Trunc, Any, All, HasInfNan
//   differentiable:     Nanmean, MaskedFill, MaskedSelect, MaskedScatter
// IndexSelect / Nansum already covered in earlier batches and are skipped.
// ============================================================================

/**
 * @brief isposinf(x). Bool tensor flagging +inf positions. Non-differentiable.
 */
class IsPosInfBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "IsPosInfBackward"; }
    auto op_id() const -> OpId override { return OpId::IsPosInf; }
};

/**
 * @brief isneginf(x). Bool tensor flagging -inf positions. Non-differentiable.
 */
class IsNegInfBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "IsNegInfBackward"; }
    auto op_id() const -> OpId override { return OpId::IsNegInf; }
};

/**
 * @brief trunc(x): round toward zero. Piecewise-constant, non-differentiable.
 */
class TruncBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "TruncBackward"; }
    auto op_id() const -> OpId override { return OpId::Trunc; }
};

/**
 * @brief any(x, dim, keepdim). Bool reduction. Non-differentiable.
 */
class AnyBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "AnyBackward"; }
    auto op_id() const -> OpId override { return OpId::Any; }
};

/**
 * @brief all(x, dim, keepdim). Bool reduction. Non-differentiable.
 */
class AllBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "AllBackward"; }
    auto op_id() const -> OpId override { return OpId::All; }
};

/**
 * @brief has_inf_nan(x). Bool scalar tensor flagging any inf/NaN element.
 *        Non-differentiable.
 */
class HasInfNanBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "HasInfNanBackward"; }
    auto op_id() const -> OpId override { return OpId::HasInfNan; }
};

/**
 * @brief nanmean(x, dim, keepdim) — mean ignoring NaN entries.
 *        Forward sums non-NaN values and divides by the per-output count of
 *        non-NaN entries. Backward distributes grad_y / count to non-NaN
 *        positions and zero to NaN positions:
 *            grad_x = where(isnan(x), 0, grad_y_broadcast / count_non_nan)
 *        Saves x; the mask and count are recomputed at backward time.
 */
class NanmeanBackward : public Function {
public:
    NanmeanBackward(std::optional<int64_t> dim, bool keepdim)
        : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "NanmeanBackward"; }
    auto op_id() const -> OpId override { return OpId::Nanmean; }
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};

/**
 * @brief masked_fill(x, mask, value) = x with mask=true positions overwritten
 *        by the scalar value. Backward:
 *            grad_x = where(mask, 0, grad_y)
 *        The value is a non-differentiable scalar. Saves the mask.
 */
class MaskedFillBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "MaskedFillBackward"; }
    auto op_id() const -> OpId override { return OpId::MaskedFill; }
};

/**
 * @brief masked_select(x, mask) — flattened gather of x at mask=true positions.
 *        Backward scatters grad_y back into a zeros_like(x) at the masked
 *        positions:  grad_x = masked_scatter(zeros_like(x), mask, grad_y).
 *        Saves the mask and the input's shape.
 */
class MaskedSelectBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "MaskedSelectBackward"; }
    auto op_id() const -> OpId override { return OpId::MaskedSelect; }
};

/**
 * @brief masked_scatter(x, mask, source) — x with mask=true positions
 *        overwritten by leading elements of source (in mask-iteration order).
 *        Backward:
 *            grad_x      = where(mask, 0, grad_y)
 *            grad_source = pad_to_source_shape(masked_select(grad_y, mask))
 *        Saves the mask and source.shape so the source grad can be padded
 *        with zeros for the trailing unused elements of source.
 */
class MaskedScatterBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "MaskedScatterBackward"; }
    auto op_id() const -> OpId override { return OpId::MaskedScatter; }
    std::vector<int64_t> source_shape_;
};

// ============================================================================
// Audit E.7 batch 6 — special-math closed forms + view/index ops
// ============================================================================

/**
 * @brief igamma(a, x) = P(a, x) = gamma_lower(a, x) / Gamma(a), the
 *        regularised lower incomplete gamma function. Differentiable wrt
 *        x with closed form:
 *            dP/dx = x^(a-1) * exp(-x) / Gamma(a)
 *                  = exp((a-1) * log(x) - x - lgamma(a))
 *        The derivative wrt a involves the derivative of the regularised
 *        series and has no elementary closed form. PyTorch mirrors this:
 *        d/dx is implemented, d/da raises NotImplemented. We follow suit
 *        and return a zero gradient for `a` (which is typically a fixed
 *        shape parameter rather than an optimisable Variable).
 */
class IgammaBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "IgammaBackward"; }
    auto op_id() const -> OpId override { return OpId::Igamma; }
};

/**
 * @brief igammac(a, x) = Q(a, x) = Gamma_upper(a, x) / Gamma(a), the
 *        regularised upper incomplete gamma function. Same structure as
 *        IgammaBackward but with the opposite sign on x:
 *            dQ/dx = -x^(a-1) * exp(-x) / Gamma(a)
 *        d/da is non-implemented (no elementary closed form); a receives
 *        a zero gradient.
 */
class IgammacBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "IgammacBackward"; }
    auto op_id() const -> OpId override { return OpId::Igammac; }
};

/**
 * @brief beta(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b). Closed-form
 *        gradients via digamma psi:
 *            dB/da = B(a, b) * (psi(a) - psi(a + b))
 *            dB/db = B(a, b) * (psi(b) - psi(a + b))
 *        Both inputs are differentiable. Broadcasting handled by the
 *        saved input shapes + reduce_grad_for_broadcasting.
 */
class BetaBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "BetaBackward"; }
    auto op_id() const -> OpId override { return OpId::Beta; }
    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
};

/**
 * @brief pairwise_distance(x1, x2, p) — per-row L_p distance for x1, x2
 *        of shape (B, D); output shape (B,). Closed-form backward:
 *            d := x1 - x2
 *            y[b] = (sum_d |d[b,d]|^p)^(1/p)
 *            dy[b]/dx1[b,d] = sign(d[b,d]) * |d[b,d]|^(p-1) * y[b]^(1-p)
 *            dy[b]/dx2[b,d] = -dy[b]/dx1[b,d]
 *        For p=2 the form simplifies to d / y. The y=0 degenerate case
 *        is regularised with a small epsilon so 0^(1-p) does not blow up.
 */
class PairwiseDistanceBackward : public Function {
public:
    explicit PairwiseDistanceBackward(double p) : p_(p) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "PairwiseDistanceBackward"; }
    auto op_id() const -> OpId override { return OpId::PairwiseDistance; }
private:
    double p_;
};

/**
 * @brief pdist(input, p) — pairwise L_p distances within a single batch.
 *        Input shape (N, D), output shape (N*(N-1)/2,) in row-major
 *        upper-triangular order. The per-pair closed-form derivative
 *        exists, but mapping it back into the (N, D) input in pure
 *        Variable ops requires materialising a dense (N, N, D)
 *        intermediate and scatter-summing pair gradients across pairs.
 *        That is O(N^2 * D) memory and needs a dedicated backward kernel
 *        for correctness/perf; the project does not yet provide one
 *        (PyTorch ships a custom `pdist_backward` op for this reason).
 *        Throw NonDifferentiable until such a kernel lands.
 */
class PdistBackward : public Function {
public:
    explicit PdistBackward(double p) : p_(p) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "PdistBackward"; }
    auto op_id() const -> OpId override { return OpId::Pdist; }
private:
    double p_;
};

/**
 * @brief cdist(x1, x2, p) — pairwise L_p distance between two sets,
 *        x1: (N, D), x2: (M, D); output (N, M). Same situation as
 *        PdistBackward: per-pair derivative is closed-form, but the
 *        scatter step needs a dedicated backward kernel which is not
 *        yet present in the project. Throw NonDifferentiable until one
 *        exists.
 */
class CDistBackward : public Function {
public:
    explicit CDistBackward(double p) : p_(p) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "CDistBackward"; }
    auto op_id() const -> OpId override { return OpId::CDist; }
private:
    double p_;
};

/**
 * @brief Advanced indexing: y = x[indices] (NumPy-style multi-tensor
 *        index). The mathematically correct backward is scatter-add of
 *        grad_y into a zero-filled source-shaped tensor at the indexed
 *        positions (duplicate indices must accumulate). The project's
 *        current AdvancedIndexPut kernels do not plumb an `accumulate`
 *        flag and `tenzor::index_put` is overwrite-only, so an
 *        autograd-level implementation today would silently drop
 *        duplicate-index gradient contributions.
 *
 *        We mark this NON-DIFFERENTIABLE with a pointer to that gap so
 *        callers fail loudly. A follow-up should land an accumulating
 *        multi-dim scatter kernel and then turn this into a real
 *        gradient.
 */
class AdvancedIndexBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "AdvancedIndexBackward"; }
    auto op_id() const -> OpId override { return OpId::AdvancedIndex; }
};

/**
 * @brief One-hot encoding: integer indices -> one-hot tensor.
 *        NON-DIFFERENTIABLE — the input is an integer index tensor and
 *        has no meaningful gradient (the output is a discrete encoding,
 *        not a smooth function of the indices).
 */
class OneHotBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "OneHotBackward"; }
    auto op_id() const -> OpId override { return OpId::OneHot; }
};

/**
 * @brief lerp(start, end, weight) = start + weight * (end - start).
 *        Tensor-weight overload: all three inputs are differentiable.
 *            d/dstart  = (1 - weight) * grad
 *            d/dend    = weight * grad
 *            d/dweight = (end - start) * grad
 *        Scalar-weight overload (`weight: double`): only start, end are
 *        differentiable; weight is constant.
 */
class LerpBackward : public Function {
public:
    LerpBackward() : has_weight_tensor_(true), weight_scalar_(0.0) {}
    explicit LerpBackward(double w) : has_weight_tensor_(false), weight_scalar_(w) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "LerpBackward"; }
    auto op_id() const -> OpId override { return OpId::Lerp; }
    bool has_weight_tensor_;
    double weight_scalar_;
    std::vector<int64_t> input_shape_start_;
    std::vector<int64_t> input_shape_end_;
    std::vector<int64_t> input_shape_weight_;
};

/**
 * @brief cross(a, b, dim) = a x b along the given dim (length 3).
 *        Closed-form backward using the antisymmetry of the cross
 *        product:
 *            grad_a = cross(b, grad, dim) = b x grad
 *            grad_b = cross(grad, a, dim) = grad x a
 *        Both inputs differentiable. No saved scalar state besides the
 *        dim used by the forward.
 */
class CrossBackward : public Function {
public:
    explicit CrossBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "CrossBackward"; }
    auto op_id() const -> OpId override { return OpId::Cross; }
private:
    int64_t dim_;
};


// ============================================================================
// Audit E.7 batch 7 — index/scatter/view ops
// ============================================================================

/**
 * @brief index_add(input, dim, index, source) — self[index[i]] += source[i]
 *        along dim. (Note: the project's underlying tensor op has no `alpha`
 *        argument; alpha is implicitly 1.0.)
 *
 *        Backward:
 *            grad_input  = grad_y   (the +=source step leaves input gradient
 *                                    untouched along non-indexed positions,
 *                                    and the indexed positions also pass
 *                                    grad through 1:1)
 *            grad_source = index_select(grad_y, dim, index)
 *        index is integer, non-differentiable.
 */
class IndexAddBackward : public Function {
public:
    explicit IndexAddBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "IndexAddBackward"; }
    auto op_id() const -> OpId override { return OpId::IndexAdd; }
private:
    int64_t dim_;
};

/**
 * @brief index_copy(input, dim, index, source) — self[index[i]] = source[i]
 *        along dim. Backward:
 *            grad_input  = index_fill(grad_y, dim, index, 0)
 *                          (indexed positions of input were overwritten)
 *            grad_source = index_select(grad_y, dim, index)
 *        Behaviour assumes indices are unique along dim; the underlying op
 *        is overwrite (not accumulate), so duplicate indices would already
 *        be undefined at the forward level.
 */
class IndexCopyBackward : public Function {
public:
    explicit IndexCopyBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "IndexCopyBackward"; }
    auto op_id() const -> OpId override { return OpId::IndexCopy; }
private:
    int64_t dim_;
};

/**
 * @brief index_fill(input, dim, index, value) — self[index[i]] = value along
 *        dim. `value` is a non-differentiable scalar.
 *        Backward:
 *            grad_input = index_fill(grad_y, dim, index, 0)
 */
class IndexFillBackward : public Function {
public:
    explicit IndexFillBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "IndexFillBackward"; }
    auto op_id() const -> OpId override { return OpId::IndexFill; }
private:
    int64_t dim_;
};

/**
 * @brief select_scatter(input, src, dim, index) — copy of input with src
 *        written into the select(dim, index) slice. Backward:
 *            grad_input = select_scatter(grad_y, zeros_like(slice), dim, index)
 *            grad_src   = select(grad_y, dim, index)
 */
class SelectScatterBackward : public Function {
public:
    SelectScatterBackward(int64_t dim, int64_t index) : dim_(dim), index_(index) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "SelectScatterBackward"; }
    auto op_id() const -> OpId override { return OpId::SelectScatter; }
private:
    int64_t dim_;
    int64_t index_;
};

/**
 * @brief slice_scatter(input, src, dim, start, end, step) — copy of input
 *        with src written into the slice(dim, start, end, step) region.
 *        Backward:
 *            grad_input = slice_scatter(grad_y, zeros_like(slice), ...)
 *            grad_src   = slice(grad_y, dim, start, end, step).contiguous()
 *        Saves src.shape so the zero-slice can be built without `slice` on
 *        the gradient first.
 */
class SliceScatterBackward : public Function {
public:
    SliceScatterBackward(int64_t dim, int64_t start, int64_t end, int64_t step)
        : dim_(dim), start_(start), end_(end), step_(step) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "SliceScatterBackward"; }
    auto op_id() const -> OpId override { return OpId::SliceScatter; }
    std::vector<int64_t> src_shape_;
private:
    int64_t dim_;
    int64_t start_;
    int64_t end_;
    int64_t step_;
};

/**
 * @brief diagonal_scatter(input, src, offset, dim1, dim2) — copy of input
 *        with src placed on the specified diagonal. The mathematically
 *        correct backward is:
 *            grad_input = diagonal_scatter(grad_y, zeros_like(src), ...)
 *            grad_src   = diagonal(grad_y, offset, dim1, dim2)
 *        The project ships `diag(...)` (1D <-> 2D shortcut) but not a
 *        general N-D `diagonal(offset, dim1, dim2)` extractor needed to
 *        gather grad_src for arbitrary (dim1, dim2). Without it, only the
 *        2D-on-the-main-axes special case is implementable, and silently
 *        restricting the backward to that case would surprise callers.
 *
 *        Mark NonDifferentiable until a general `diagonal(...)` extractor
 *        lands; this matches the project policy of failing loudly rather
 *        than faking gradients.
 */
class DiagonalScatterBackward : public Function {
public:
    DiagonalScatterBackward(int64_t offset, int64_t dim1, int64_t dim2)
        : offset_(offset), dim1_(dim1), dim2_(dim2) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "DiagonalScatterBackward"; }
    auto op_id() const -> OpId override { return OpId::DiagonalScatter; }
private:
    int64_t offset_;
    int64_t dim1_;
    int64_t dim2_;
};

/**
 * @brief repeat_interleave(input, repeats: int, dim) — uniform integer-
 *        repeats overload. Each element along `dim` is repeated `repeats`
 *        times consecutively. Output shape multiplies `dim`'s extent by
 *        `repeats`. When dim is nullopt the input is flattened first and
 *        the output is 1D.
 *
 *        Backward: reshape the output gradient so the repeated axis is
 *        split into (orig_size, repeats), sum along the repeats axis, then
 *        reshape back to the input shape.
 *
 *        The tensor-valued repeats overload (`Tensor repeats`) is *not*
 *        handled here — it requires a scatter-add against a variable-
 *        length expansion which has no closed form without the per-element
 *        repeat counts and an accumulating scatter; we route it through a
 *        separate non-differentiable wrapper.
 */
class RepeatInterleaveBackward : public Function {
public:
    // CC.2: `was_flattened_` distinguishes the nullopt-dim path (forward
    // flattens N-D input to 1D before repeating) from the explicit-dim path
    // (forward preserves rank). Saving only `dim_` is ambiguous when the
    // original input happened to be 1D: the backward needs to know whether
    // the output gradient still has the original rank or has been collapsed
    // to a single axis. Setting this flag in the wrapper based on
    // `dim == nullopt` is the only signal we have at backward time.
    RepeatInterleaveBackward(int64_t repeats, std::optional<int64_t> dim,
                             std::vector<int64_t> input_shape,
                             bool was_flattened)
        : repeats_(repeats), dim_(dim), input_shape_(std::move(input_shape)),
          was_flattened_(was_flattened) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "RepeatInterleaveBackward"; }
    auto op_id() const -> OpId override { return OpId::RepeatInterleave; }
private:
    int64_t repeats_;
    std::optional<int64_t> dim_;
    std::vector<int64_t> input_shape_;
    bool was_flattened_;
};

/**
 * @brief unfold(input, kernel_size, stride, padding, dilation) — image-style
 *        sliding-window patch extraction (im2col). Input (N, C, H, W), output
 *        (N, C*K*K, L). Backward is the linear adjoint, which is exactly the
 *        scatter-add `fold(grad_y, (H, W), kernel, stride, padding, dilation)`.
 *        Saves the input spatial size to reconstruct the output_size arg.
 */
class UnfoldBackward : public Function {
public:
    UnfoldBackward(int64_t kernel_size, int64_t stride, int64_t padding,
                   int64_t dilation, int64_t H, int64_t W)
        : kernel_size_(kernel_size), stride_(stride), padding_(padding),
          dilation_(dilation), H_(H), W_(W) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "UnfoldBackward"; }
    auto op_id() const -> OpId override { return OpId::Unfold; }
private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t H_;
    int64_t W_;
};

/**
 * @brief nonzero(x) — returns Int64 indices of nonzero entries.
 *        NON-DIFFERENTIABLE: the output is an integer index tensor and is
 *        not a smooth function of the input.
 */
class NonzeroBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "NonzeroBackward"; }
    auto op_id() const -> OpId override { return OpId::Nonzero; }
};

/**
 * @brief unique(x) — returns sorted unique values (plus optional inverse and
 *        counts). Forward is a sorting/deduplication step whose dependence
 *        on x is discontinuous (the set of selected positions jumps under
 *        small perturbations when ties exist). No well-defined gradient.
 *        NON-DIFFERENTIABLE (strict; matches Sign-family policy).
 */
class UniqueBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "UniqueBackward"; }
    auto op_id() const -> OpId override { return OpId::Unique; }
};

// ============================================================================
// Audit E.7 batch 8 — order-statistic, integration, segment reductions
// ============================================================================

/**
 * @brief aminmax(x, dim, keepdim) — simultaneous min + max along dim,
 *        returning (min_values, max_values).
 *
 *        Project autograd Function instances own a single logical output
 *        Variable, so the wrapper produces two Variables each with its own
 *        AminmaxBackward (one configured for the "min" route, one for the
 *        "max" route). Each backward scatters its incoming grad onto the
 *        value-matching positions in the input (tie-normalised) and routes
 *        zero everywhere else. The autograd engine sums the two
 *        contributions when the same input reaches both branches.
 *
 *        Saves per instance: input, this branch's value tensor (min or max).
 */
class AminmaxBackward : public Function {
public:
    AminmaxBackward(std::optional<int64_t> dim, bool keepdim, bool is_max)
        : dim_(dim), keepdim_(keepdim), is_max_(is_max) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "AminmaxBackward"; }
    auto op_id() const -> OpId override { return OpId::Aminmax; }
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
    bool is_max_;
};

/**
 * @brief kthvalue(x, k, dim, keepdim) — k-th smallest value and its index along
 *        dim. Gradient: scatter dL/dy at the k-th-position index (single
 *        position per reduction row). Saves the index tensor returned by the
 *        forward to drive scatter_add at backward time.
 */
class KthvalueBackward : public Function {
public:
    KthvalueBackward(int64_t k, int64_t dim, bool keepdim)
        : k_(k), dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "KthvalueBackward"; }
    auto op_id() const -> OpId override { return OpId::Kthvalue; }
private:
    int64_t k_;
    int64_t dim_;
    bool keepdim_;
};

/**
 * @brief quantile(x, q, dim, keepdim) — interpolated q-th quantile along dim.
 *        The backward is the linear adjoint of a two-position interpolation
 *        between the two flanking order statistics for each reduction row,
 *        which requires reconstructing the per-row sort permutation at
 *        backward time. The project's order-statistic primitives don't expose
 *        a stable per-row argsort with interpolation weights without doing
 *        the sort twice (once in forward, once in backward); the inputs/
 *        outputs alone are not enough to recover the two flanking positions
 *        for every (..., dim, ...) slice.
 *
 *        Marking NonDifferentiable until a stable per-row argsort surface
 *        lands — fail loudly rather than fake gradients. See policy in
 *        DiagonalScatterBackward / repeat_interleave(Tensor) for the same
 *        pattern.
 */
class QuantileBackward : public Function {
public:
    QuantileBackward(double q, std::optional<int64_t> dim, bool keepdim)
        : q_(q), dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "QuantileBackward"; }
    auto op_id() const -> OpId override { return OpId::Quantile; }
private:
    double q_;
    std::optional<int64_t> dim_;
    bool keepdim_;
};

/**
 * @brief nanmedian(x, dim) — median along dim with NaN entries skipped.
 *        Backward: scatter dL/dy onto positions equal to the saved median
 *        value, with NaN positions excluded from the tie-mask normalisation.
 *        Equivalent to MedianBackward but the mask is multiplied by
 *        `!isnan(x)` before tie-count normalisation.
 */
class NanmedianBackward : public Function {
public:
    explicit NanmedianBackward(std::optional<int64_t> dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "NanmedianBackward"; }
    auto op_id() const -> OpId override { return OpId::Nanmedian; }
private:
    std::optional<int64_t> dim_;
};

/**
 * @brief trapezoid(y, dx | x, dim) — trapezoidal rule integration.
 *
 *   I = sum_{i=0..N-2} 0.5 * (y[i] + y[i+1]) * dx[i]
 *
 *   * Uniform spacing (constant dx):
 *         dI/dy[i] = dx                 for interior i (0 < i < N-1)
 *         dI/dy[0] = dI/dy[N-1] = dx/2  for the two endpoints
 *   * Non-uniform spacing (x given):
 *         dI/dy[i]  = 0.5 * (dx[i-1] + dx[i])  for interior
 *         dI/dy[0]  = 0.5 * dx[0]
 *         dI/dy[-1] = 0.5 * dx[-1]
 *         dI/dx[i]  is 0 at the endpoints in dx-space; we only differentiate
 *                   wrt y (matching PyTorch's `torch.trapezoid` backward).
 *
 *   has_x_=false ⇒ uniform-dx overload. dim_ is normalised to non-negative
 *   inside the backward.
 */
class TrapezoidBackward : public Function {
public:
    TrapezoidBackward(int64_t dim, double dx, bool has_x)
        : dim_(dim), dx_(dx), has_x_(has_x) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "TrapezoidBackward"; }
    auto op_id() const -> OpId override { return OpId::Trapezoid; }
private:
    int64_t dim_;
    double dx_;
    bool has_x_;
};

/**
 * @brief cumulative_trapezoid(y, dx | x, dim) — cumulative trapezoidal
 *        integration. Output has dim's extent reduced by 1.
 *
 *   For uniform dx, out[k] = sum_{i=0..k-1} 0.5*(y[i] + y[i+1])*dx, so
 *   d out[k] / d y[i] is dx if 1 <= i <= k-1 (interior) or dx/2 at edge i=0
 *   or i=k. Equivalently, grad_y[i] = dx/2 * (G_i + G_{i-1}) where
 *   G_j = sum_{k>=j} grad_out[k] is the *reverse cumsum* of grad_out, with
 *   G_N = 0. We compute this by reverse-cumsum, pad-shift, and average.
 *
 *   Non-uniform x case uses dx[i] = x[i+1] - x[i] in place of constant dx
 *   (dx[i-1] + dx[i] etc.). Only grad wrt y is produced.
 */
class CumulativeTrapezoidBackward : public Function {
public:
    CumulativeTrapezoidBackward(int64_t dim, double dx, bool has_x)
        : dim_(dim), dx_(dx), has_x_(has_x) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "CumulativeTrapezoidBackward"; }
    auto op_id() const -> OpId override { return OpId::CumulativeTrapezoid; }
private:
    int64_t dim_;
    double dx_;
    bool has_x_;
};

/**
 * @brief segment_reduce(data, offsets, reduce, axis) — segmented reduction.
 *        Backward is well-defined for "sum" and "mean" (route grad to each
 *        segment uniformly, optionally divided by segment length) but for
 *        "max"/"min" it requires argmax/argmin per segment, which the kernel
 *        does not currently expose, and for "prod" it requires the per-
 *        segment product chain which suffers numerically when any element is
 *        zero. Implementing only sum/mean would silently break callers using
 *        the other modes (no signal that no gradient flows).
 *
 *        Project policy: NonDifferentiable until the kernel returns per-
 *        segment argmax/argmin indices and a numerically-safe prod backward
 *        is added. Matches the policy used for Nonzero/Unique/DiagonalScatter.
 */
class SegmentReduceBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "SegmentReduceBackward"; }
    auto op_id() const -> OpId override { return OpId::SegmentReduce; }
};

/**
 * @brief gumbel_softmax(logits, tau, hard, dim) — Gumbel-Softmax with optional
 *        straight-through estimator. The standard "soft" backward is the
 *        SoftmaxBackward of the noise-perturbed logits, and the hard=true
 *        variant uses a straight-through estimator that re-routes the
 *        backward through the soft sample. Implementing this correctly
 *        requires saving the *exact* Gumbel noise drawn in the forward (so
 *        the backward is reproducible across calls and the soft surrogate
 *        matches what was sampled) — the current forward does not return or
 *        save that noise, so any backward we add here would either re-sample
 *        and silently produce wrong gradients, or fall back to a STE that
 *        ignores the temperature `tau` scaling.
 *
 *        Project policy: NonDifferentiable until the forward saves the
 *        Gumbel noise and the STE variant is wired through SoftmaxBackward.
 *        Users that need the gradient should compose softmax(logits / tau)
 *        directly and apply their own STE wrapper.
 */
class GumbelSoftmaxBackward : public Function {
public:
    GumbelSoftmaxBackward(double tau, bool hard, int64_t dim)
        : tau_(tau), hard_(hard), dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    // audit-11 QQ.4: implement straight-through estimator (PyTorch parity).
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "GumbelSoftmaxBackward"; }
    auto op_id() const -> OpId override { return OpId::GumbelSoftmax; }
private:
    double tau_;
    bool hard_;
    int64_t dim_;
};

/**
 * @brief cummax(x, dim) — cumulative maximum along dim, returning (values,
 *        indices). Forward: out[k] = max(x[0..k]); indices[k] = argmax(x[0..k]).
 *        Backward: scatter_add of grad_values along dim using the saved
 *        indices. Equivalent to gather's adjoint. Only grad wrt x (values
 *        slot) is produced; indices are non-differentiable.
 */
class CumMaxBackward : public Function {
public:
    explicit CumMaxBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    // audit-11 QQ.5: higher-order — default would throw under II.2.
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "CumMaxBackward"; }
    auto op_id() const -> OpId override { return OpId::CumMax; }
private:
    int64_t dim_;
};

/**
 * @brief cummin(x, dim) — cumulative minimum, returning (values, indices).
 *        Backward symmetric to CumMaxBackward.
 */
class CumMinBackward : public Function {
public:
    explicit CumMinBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    // audit-11 QQ.5: higher-order — default would throw under II.2.
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "CumMinBackward"; }
    auto op_id() const -> OpId override { return OpId::CumMin; }
private:
    int64_t dim_;
};

// ============================================================================
// Audit E.7 batch 9 — indexing/scatter reductions, audio/vision composites,
// integer-only binary ops.
//
// Differentiable: ScatterReduce / IndexReduce (sum, mean), EmbeddingBag.
// Non-differentiable (typed): ROIAlign (kernel needs features for adjoint;
// modelled at Module layer instead — keeping a typed stub for direct OpId
// users), DeformableConv2d (split per-input kernels), MelScale / DCT / MFCC
// (linear in principle; adjoint requires recomputing fixed transforms with
// matching norm conventions which the kernels do not export), Gcd / Lcm
// (integer-domain operations whose Jacobian is zero a.e.).
// ============================================================================

/**
 * @brief scatter_reduce(input, dim, index, src, reduce, include_self)
 *        — scatter src into input at positions in index along `dim`,
 *        combining colliding writes with `reduce`.
 *
 *        Differentiable for `reduce = "sum"` and `reduce = "mean"`:
 *          grad_src   = gather(grad_out, dim, index) / (cnt if mean else 1)
 *          grad_input = grad_out * (include_self ? 1 : 0)  (mean dilutes too)
 *
 *        For "amax" / "amin" the backward needs an argmax/argmin tie mask
 *        which scatter_reduce does not return; for "prod" the backward
 *        divides by zero when any contributing element is zero. Both raise
 *        NonDifferentiable to fail loudly.
 */
class ScatterReduceBackward : public Function {
public:
    ScatterReduceBackward(int64_t dim, std::string reduce, bool include_self)
        : dim_(dim), reduce_(std::move(reduce)), include_self_(include_self) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "ScatterReduceBackward"; }
    auto op_id() const -> OpId override { return OpId::ScatterReduce; }
private:
    int64_t dim_;
    std::string reduce_;
    bool include_self_;
};

/**
 * @brief index_reduce(input, dim, index, src, reduce, include_self)
 *        — like scatter_reduce but `index` is 1-D over `dim`.
 *
 *        Differentiable for "sum" / "mean" via index_select of grad_out
 *        along `dim`. NonDifferentiable for "amax" / "amin" / "prod"
 *        for the same reasons as ScatterReduceBackward.
 */
class IndexReduceBackward : public Function {
public:
    IndexReduceBackward(int64_t dim, std::string reduce, bool include_self)
        : dim_(dim), reduce_(std::move(reduce)), include_self_(include_self) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "IndexReduceBackward"; }
    // index_reduce is a thin wrapper around scatter_reduce (see
    // src/ops/indexing.cpp); reuse its OpId for graph identity.
    auto op_id() const -> OpId override { return OpId::ScatterReduce; }
private:
    int64_t dim_;
    std::string reduce_;
    bool include_self_;
};

/**
 * @brief embedding_bag — reduces an embedding lookup per bag (sum / mean /
 *        max). Differentiable in `weight` only (indices/offsets are integer-
 *        typed). Backward dispatches to OpId::EmbeddingBagBackward which
 *        applies the mode-specific scatter (sum: scatter-add; mean: scatter-
 *        add divided by bag size; max: scatter at argmax indices).
 *
 *        Wired here so any caller that bypasses nn::EmbeddingBag (e.g. a
 *        direct OpId::EmbeddingBagForward dispatch) still gets a typed
 *        Function instead of "Function 'unknown' has no backward".
 */
class EmbeddingBagBackward : public Function {
public:
    EmbeddingBagBackward(std::string mode, int64_t padding_idx,
                         int64_t num_embeddings)
        : mode_(std::move(mode)),
          padding_idx_(padding_idx),
          num_embeddings_(num_embeddings) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "EmbeddingBagBackward"; }
    auto op_id() const -> OpId override { return OpId::EmbeddingBagForward; }
private:
    std::string mode_;
    int64_t padding_idx_;
    int64_t num_embeddings_;
};

/**
 * @brief ROIAlign forward. Differentiable in `features` via
 *        OpId::ROIAlignBackward, but the project routes ROI Align through
 *        the nn::detection::ROIAlign Module which manages the saved
 *        feature shape and ROI tensor explicitly.
 *
 *        This stub exists for graph hygiene when a caller wires
 *        OpId::ROIAlignForward directly without the Module — backward
 *        in that path is undefined without a saved features tensor, so
 *        we fail loudly. Use nn::detection::ROIAlign for autograd-
 *        aware ROI alignment.
 */
class ROIAlignBackward : public Function {
public:
    ROIAlignBackward() = default;
    ROIAlignBackward(int64_t output_h, int64_t output_w,
                     double spatial_scale, int64_t sampling_ratio, bool aligned)
        : output_h_(output_h), output_w_(output_w),
          spatial_scale_(spatial_scale), sampling_ratio_(sampling_ratio),
          aligned_(aligned), configured_(true) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "ROIAlignBackward"; }
    auto op_id() const -> OpId override { return OpId::ROIAlignForward; }
private:
    int64_t output_h_ = 7;
    int64_t output_w_ = 7;
    double spatial_scale_ = 1.0 / 16.0;
    int64_t sampling_ratio_ = 0;
    bool aligned_ = true;
    bool configured_ = false;
};

/**
 * @brief DeformableConv2d (DCNv2) forward. Backward is split across three
 *        kernels — OpId::DeformableConv2dBackwardInput,
 *        OpId::DeformableConv2dBackwardWeight, and
 *        OpId::DeformableConv2dBackwardBias — each returning a different
 *        gradient. A unified Function-level adjoint that re-dispatches all
 *        three and threads them back to the right input Variables is not
 *        currently wired, so we fail loudly.
 *
 *        For autograd-aware deformable conv use the nn::DeformableConv2d
 *        Module which owns weights/bias and constructs the right gradient
 *        routing per input.
 */
class DeformableConv2dBackward : public Function {
public:
    DeformableConv2dBackward() = default;
    DeformableConv2dBackward(int64_t stride_h, int64_t stride_w,
                             int64_t pad_h, int64_t pad_w,
                             int64_t dilation_h, int64_t dilation_w,
                             int64_t groups, int64_t offset_groups,
                             bool use_mask, bool has_bias)
        : stride_h_(stride_h), stride_w_(stride_w),
          pad_h_(pad_h), pad_w_(pad_w),
          dilation_h_(dilation_h), dilation_w_(dilation_w),
          groups_(groups), offset_groups_(offset_groups),
          use_mask_(use_mask), has_bias_(has_bias), configured_(true) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "DeformableConv2dBackward"; }
    auto op_id() const -> OpId override { return OpId::DeformableConv2dForward; }
private:
    int64_t stride_h_ = 1, stride_w_ = 1;
    int64_t pad_h_ = 0, pad_w_ = 0;
    int64_t dilation_h_ = 1, dilation_w_ = 1;
    int64_t groups_ = 1, offset_groups_ = 1;
    bool use_mask_ = false;
    bool has_bias_ = false;
    bool configured_ = false;
};

/**
 * @brief mel_scale — applies a fixed triangular mel filterbank to a
 *        magnitude/power spectrogram. In principle differentiable
 *        (linear matmul against a constant matrix), but the filterbank
 *        is built inside the op and not exposed as a Tensor; reconstructing
 *        it at backward time would duplicate the (sample_rate, n_mels,
 *        f_min, f_max)-dependent generation logic in autograd. Marked
 *        NonDifferentiable until the filterbank is hoisted into a saved
 *        tensor on the forward path.
 */
class MelScaleBackward : public Function {
public:
    MelScaleBackward(int64_t n_mels, double f_min, double f_max,
                     int64_t sample_rate)
        : n_mels_(n_mels), f_min_(f_min), f_max_(f_max),
          sample_rate_(sample_rate) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "MelScaleBackward"; }
    auto op_id() const -> OpId override { return OpId::MelScale; }
private:
    int64_t n_mels_;
    double f_min_;
    double f_max_;
    int64_t sample_rate_;
};

/**
 * @brief dct(input, type, n, dim, norm) — Discrete Cosine Transform.
 *        Linear and therefore differentiable (the adjoint of DCT-II is
 *        DCT-III with matching norm and vice versa). The exact adjoint
 *        depends on the (`type`, `norm`) pair and on whether `n` truncated
 *        or padded the signal; getting this wrong silently distorts the
 *        gradient, so we mark it NonDifferentiable until the
 *        type/norm/length matrix is implemented.
 */
class DCTBackward : public Function {
public:
    DCTBackward(int type, int64_t n, int64_t dim, std::string norm)
        : type_(type), n_(n), dim_(dim), norm_(std::move(norm)) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "DCTBackward"; }
    auto op_id() const -> OpId override { return OpId::DCT; }
private:
    int type_;
    int64_t n_;
    int64_t dim_;
    std::string norm_;
};

/**
 * @brief mfcc(waveform, sample_rate, n_mfcc, n_mels, n_fft, hop_length,
 *        f_min, f_max) — Mel-Frequency Cepstral Coefficients.
 *
 *        MFCC = DCT( log( MelScale( |STFT(x)|^2 ) ) ). The composition is
 *        differentiable in principle but the forward op fuses STFT, mel,
 *        log and DCT into one kernel without exposing intermediates, so
 *        no closed-form adjoint is reachable from the autograd layer.
 *        Use the explicit pipeline (stft → abs → square → mel_scale → log
 *        → dct) if you need gradients through MFCC.
 */
class MFCCBackward : public Function {
public:
    MFCCBackward() = default;
    MFCCBackward(int64_t sample_rate, int64_t n_mfcc, int64_t n_mels,
                 int64_t n_fft, int64_t hop_length, double f_min, double f_max)
        : sample_rate_(sample_rate), n_mfcc_(n_mfcc), n_mels_(n_mels),
          n_fft_(n_fft), hop_length_(hop_length),
          f_min_(f_min), f_max_(f_max), configured_(true) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "MFCCBackward"; }
    auto op_id() const -> OpId override { return OpId::MFCC; }
private:
    int64_t sample_rate_ = 16000;
    int64_t n_mfcc_ = 40;
    int64_t n_mels_ = 128;
    int64_t n_fft_ = 400;
    int64_t hop_length_ = 160;
    double f_min_ = 0.0;
    double f_max_ = 0.0;
    bool configured_ = false;
};

/**
 * @brief gcd(a, b) — greatest common divisor. Integer-typed inputs; the
 *        Jacobian is zero a.e. (and the operation isn't well-defined for
 *        non-integer reals). NonDifferentiable.
 */
class GcdBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "GcdBackward"; }
    auto op_id() const -> OpId override { return OpId::Gcd; }
};

/**
 * @brief lcm(a, b) — least common multiple. Integer-typed inputs; same
 *        rationale as GcdBackward. NonDifferentiable.
 */
class LcmBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "LcmBackward"; }
    auto op_id() const -> OpId override { return OpId::Lcm; }
};

// ============================================================================
// Audit E.7 batch 10 — RNN kernel-level forwards + small math/transform ops
//
// RNN cell + multi-step forwards (LSTMCellForward, GRUCellForward, LSTMForward,
// GRUForward, BiLSTMForward, LSTMMultiLayerForward, GRUMultiLayerForward) are
// kernel-level OpIds. The autograd-aware path in this project is the
// nn::LSTMCell / nn::GRUCell / nn::LSTM / nn::GRU Modules, which build the
// computation graph by composing Variable-level Linear/sigmoid/tanh/slice
// (see src/nn/layers/lstm.cpp, gru.cpp, rnn.cpp). The kernel forwards exist
// for fused-inference paths and have matching kernel-level *Backward kernels
// (OpId::LSTMCellBackward, OpId::GRUCellBackward); a Function-layer adjoint
// that re-dispatches those kernels and threads gradients back to the right
// input Variables is not currently wired, so we install typed
// NonDifferentiable stubs to fail loudly with a pointer to the Module-level
// autograd path.
//
// NonDifferentiable: NextafterBackward (discrete float step), IDCTBackward
// (mirror of DCTBackward — linear in principle but the inverse transform's
// adjoint depends on the (type, norm, n) tuple which is not exposed).
// Differentiable: LdexpBackward — d/dx ldexp(x, n) = 2^n (the integer
// exponent n is non-differentiable; we only route the gradient back to x).
// ============================================================================

/**
 * @brief LSTMCellForward kernel op — fused single-step LSTM cell forward
 *        (returns [h_new, c_new]).
 *
 *        Autograd-aware path: nn::LSTMCell (see src/nn/layers/lstm.cpp)
 *        which decomposes the cell into Variable-level Linear / sigmoid /
 *        tanh / slice / add / mul. The composed graph carries gradients to
 *        input, hx, cx and the cell's weight Modules with no Function-layer
 *        bookkeeping required here.
 *
 *        A unified Function-layer adjoint that re-dispatches the kernel
 *        OpId::LSTMCellBackward (which exists in every backend's registry)
 *        and threads grad_input, grad_hx, grad_cx, grad_w_ih, grad_w_hh,
 *        grad_b_ih, grad_b_hh back to the right input Variables is not
 *        currently wired, so direct OpId::LSTMCellForward dispatch routed
 *        through autograd raises NonDifferentiable. Use nn::LSTMCell.
 */
class LSTMCellForwardBackward : public Function {
public:
    LSTMCellForwardBackward(int64_t input_size, int64_t hidden_size)
        : input_size_(input_size), hidden_size_(hidden_size) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "LSTMCellForwardBackward"; }
    auto op_id() const -> OpId override { return OpId::LSTMCellForward; }
private:
    int64_t input_size_;
    int64_t hidden_size_;
};

/**
 * @brief GRUCellForward kernel op — fused single-step GRU cell forward.
 *        Three gates (r, z, n). Same rationale as LSTMCellForwardBackward:
 *        the autograd-aware path is nn::GRUCell (src/nn/layers/gru.cpp)
 *        which composes the cell from Variable ops. Direct kernel dispatch
 *        through autograd raises NonDifferentiable.
 */
class GRUCellForwardBackward : public Function {
public:
    GRUCellForwardBackward(int64_t input_size, int64_t hidden_size)
        : input_size_(input_size), hidden_size_(hidden_size) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "GRUCellForwardBackward"; }
    auto op_id() const -> OpId override { return OpId::GRUCellForward; }
private:
    int64_t input_size_;
    int64_t hidden_size_;
};

/**
 * @brief LSTMForward kernel op — fused single-layer multi-step LSTM forward.
 *        The autograd-aware path is nn::LSTM (src/nn/layers/lstm.cpp) which
 *        unrolls per-timestep LSTMCell::forward calls so each step's
 *        Variable graph composes into the sequence graph. Direct kernel
 *        dispatch through autograd raises NonDifferentiable.
 */
class LSTMForwardBackward : public Function {
public:
    LSTMForwardBackward(int64_t input_size, int64_t hidden_size,
                        bool batch_first)
        : input_size_(input_size), hidden_size_(hidden_size),
          batch_first_(batch_first) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "LSTMForwardBackward"; }
    auto op_id() const -> OpId override { return OpId::LSTMForward; }
private:
    int64_t input_size_;
    int64_t hidden_size_;
    bool batch_first_;
};

/**
 * @brief GRUForward kernel op — fused single-layer multi-step GRU forward.
 *        Same rationale as LSTMForwardBackward: nn::GRU is the autograd-
 *        aware path. Direct kernel dispatch through autograd raises
 *        NonDifferentiable.
 */
class GRUForwardBackward : public Function {
public:
    GRUForwardBackward(int64_t input_size, int64_t hidden_size,
                       bool batch_first)
        : input_size_(input_size), hidden_size_(hidden_size),
          batch_first_(batch_first) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "GRUForwardBackward"; }
    auto op_id() const -> OpId override { return OpId::GRUForward; }
private:
    int64_t input_size_;
    int64_t hidden_size_;
    bool batch_first_;
};

/**
 * @brief LSTMMultiLayerForward kernel op — stacked-layer fused LSTM forward.
 *        The autograd-aware path is nn::LSTM (with num_layers > 1) which
 *        loops nn::LSTMCell forwards per layer per timestep, so the graph
 *        carries gradients through every layer's Linear weights and the
 *        per-step hidden/cell states. Direct kernel dispatch through
 *        autograd raises NonDifferentiable.
 */
class LSTMMultiLayerForwardBackward : public Function {
public:
    LSTMMultiLayerForwardBackward(int64_t input_size, int64_t hidden_size,
                                   int64_t num_layers, bool batch_first)
        : input_size_(input_size), hidden_size_(hidden_size),
          num_layers_(num_layers), batch_first_(batch_first) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "LSTMMultiLayerForwardBackward"; }
    auto op_id() const -> OpId override { return OpId::LSTMMultiLayerForward; }
private:
    int64_t input_size_;
    int64_t hidden_size_;
    int64_t num_layers_;
    bool batch_first_;
};

/**
 * @brief GRUMultiLayerForward kernel op — stacked-layer fused GRU forward.
 *        Same rationale as LSTMMultiLayerForwardBackward. Use nn::GRU
 *        with num_layers > 1 for autograd-aware multi-layer GRU.
 */
class GRUMultiLayerForwardBackward : public Function {
public:
    GRUMultiLayerForwardBackward(int64_t input_size, int64_t hidden_size,
                                  int64_t num_layers, bool batch_first)
        : input_size_(input_size), hidden_size_(hidden_size),
          num_layers_(num_layers), batch_first_(batch_first) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "GRUMultiLayerForwardBackward"; }
    auto op_id() const -> OpId override { return OpId::GRUMultiLayerForward; }
private:
    int64_t input_size_;
    int64_t hidden_size_;
    int64_t num_layers_;
    bool batch_first_;
};

/**
 * @brief BiLSTMForward kernel op — single-layer bidirectional LSTM forward.
 *        The autograd-aware path is nn::LSTM(bidirectional=true) which
 *        wires forward and reverse LSTMCell sweeps and concatenates the
 *        per-step hidden outputs. Direct kernel dispatch through autograd
 *        raises NonDifferentiable.
 */
class BiLSTMForwardBackward : public Function {
public:
    BiLSTMForwardBackward(int64_t input_size, int64_t hidden_size,
                          bool batch_first)
        : input_size_(input_size), hidden_size_(hidden_size),
          batch_first_(batch_first) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "BiLSTMForwardBackward"; }
    auto op_id() const -> OpId override { return OpId::BiLSTMForward; }
private:
    int64_t input_size_;
    int64_t hidden_size_;
    bool batch_first_;
};

/**
 * @brief nextafter(from, to) — returns the next representable floating-
 *        point value after `from` in the direction of `to`. The output
 *        jumps to the next ULP, so it is piecewise-constant in `from`
 *        (Jacobian zero a.e. with a measure-zero jump at every
 *        representable boundary) and discrete in `to` (sign-only
 *        dependence). Marked NonDifferentiable to fail loudly instead
 *        of returning the deceptive zero gradient that "Mul by 1.0"
 *        would produce.
 */
class NextafterBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "NextafterBackward"; }
    auto op_id() const -> OpId override { return OpId::Nextafter; }
};

/**
 * @brief ldexp(x, n) = x * 2^n. Differentiable in `x`: d/dx ldexp(x, n)
 *        = 2^n, so grad_x = ldexp(grad_out, n). The exponent `n` is
 *        integer-typed (it is interpreted as an exact bit-shift of the
 *        binary exponent field), so we route a zero gradient to it
 *        rather than throwing — same convention the project uses for
 *        integer-indexed Variable inputs elsewhere (Gather index,
 *        IndexSelect index, etc.).
 *
 *        Saves: the exponent tensor `n` (needed to recompute 2^n at
 *        backward time).
 */
class LdexpBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "LdexpBackward"; }
    auto op_id() const -> OpId override { return OpId::Ldexp; }
private:
    std::vector<int64_t> input_shape_x_;
    std::vector<int64_t> input_shape_n_;
};

/**
 * @brief idct(input, type, n, dim, norm) — Inverse Discrete Cosine
 *        Transform. Linear and therefore differentiable in principle
 *        (the adjoint of IDCT-{II,III,IV} is its matching DCT under the
 *        same `norm`), but the exact adjoint depends on the (type, norm,
 *        n) tuple in the same way DCTBackward does. The kernels do not
 *        export a parameter-matched DCT helper to compose the adjoint
 *        from, so we mirror DCTBackward and mark NonDifferentiable until
 *        the type/norm/length matrix is implemented. Use norm="ortho"
 *        plus an explicit DCT call if you need gradients through IDCT.
 */
class IDCTBackward : public Function {
public:
    IDCTBackward(int type, int64_t n, int64_t dim, std::string norm)
        : type_(type), n_(n), dim_(dim), norm_(std::move(norm)) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "IDCTBackward"; }
    auto op_id() const -> OpId override { return OpId::IDCT; }
private:
    int type_;
    int64_t n_;
    int64_t dim_;
    std::string norm_;
};

// ============================================================================
// Audit E.7 batch 11 — final residual OpIds: nested-tensor conversion /
// log-softmax / linear OpId-level routes + integer-style binary mod ops +
// fused quantized inference ops.
// ============================================================================

/**
 * @brief remainder(a, b) — Python-style modulo: a - floor(a/b) * b. The
 *        output is piecewise-linear with discrete jumps of size `b` at every
 *        multiple of `b`, so the Jacobian is the constant {1, -floor(a/b)}
 *        between those jumps and undefined at them. We mark this
 *        NonDifferentiable to fail loudly: the "piecewise-constant per
 *        segment" gradient that PyTorch returns silently composes into a
 *        wrong gradient any time `a/b` crosses an integer during training,
 *        and the project policy is to refuse rather than return a deceptive
 *        adjoint at the discontinuity. Users who need a smooth surrogate
 *        should compose `a - b * floor(a / b)` explicitly with their own
 *        STE.
 */
class RemainderBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "RemainderBackward"; }
    auto op_id() const -> OpId override { return OpId::Remainder; }
};

/**
 * @brief fmod(a, b) — C-style modulo: a - trunc(a/b) * b. Same rationale
 *        as RemainderBackward but with `trunc` instead of `floor`; the
 *        jumps now straddle zero, so the discontinuity set is the same
 *        measure-zero lattice of `a/b` integer crossings. NonDifferentiable.
 */
class FmodBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "FmodBackward"; }
    auto op_id() const -> OpId override { return OpId::Fmod; }
};

/**
 * @brief nested_log_softmax(values, offsets, dim) — segmented log-softmax
 *        along the ragged dim of a packed nested tensor. The autograd path
 *        in this project is the explicit `tenzor::log(nested_softmax(...))`
 *        composition (NestedSoftmaxBackward + LogBackward), which carries
 *        gradients via the existing per-segment dot-correction adjoint of
 *        NestedSoftmax. A direct Function-layer adjoint for the fused
 *        log-softmax would have to (a) reconstruct the per-segment grad
 *        sum across the same offsets and (b) widen Float16/BFloat16 inputs
 *        through the segment reduction the same way NestedSoftmaxBackward
 *        does. That wiring is not present here. NonDifferentiable until the
 *        per-segment adjoint is implemented; use
 *        `tenzor::log(autograd::nested_softmax(values, offsets, dim))` if
 *        you need gradients through nested log-softmax.
 */
class NestedLogSoftmaxBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "NestedLogSoftmaxBackward"; }
    auto op_id() const -> OpId override { return OpId::NestedLogSoftmax; }
};

/**
 * @brief nested_to_padded(values, offsets, padding_value) — densify a
 *        packed nested tensor into a padded [B, max_len, ...] dense tensor.
 *        The forward is a per-segment scatter into padded slots. The
 *        adjoint is the inverse scatter (gather the unpadded slots back
 *        into packed values), which requires both the offsets and the
 *        per-batch lengths the forward saw — the forward currently does
 *        not expose those as saved tensors, and the autograd-aware path
 *        for nested ↔ padded round-trips is to use the Variable-level
 *        scatter/gather composition in nn::NestedSequence (or to keep
 *        the computation in packed-values form and use the per-op nested
 *        autograd Functions). NonDifferentiable until the forward saves
 *        the {offsets, max_len, batch_lengths} triple needed for the
 *        gather adjoint.
 */
class NestedToPaddedBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "NestedToPaddedBackward"; }
    auto op_id() const -> OpId override { return OpId::NestedToPadded; }
};

/**
 * @brief nested_from_padded(padded, offsets) — inverse of nested_to_padded:
 *        gather the unpadded slots of a [B, max_len, ...] dense tensor
 *        into packed nested values. The adjoint is the scatter (the same
 *        kernel as nested_to_padded), but routing the gradient back
 *        requires the per-batch lengths and pad slots the forward saw,
 *        which are not saved here. NonDifferentiable until the forward
 *        saves the {offsets, max_len, batch_lengths} triple; same
 *        rationale as NestedToPaddedBackward.
 */
class NestedFromPaddedBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "NestedFromPaddedBackward"; }
    auto op_id() const -> OpId override { return OpId::NestedFromPadded; }
};

/**
 * @brief nested_linear(values, weight, bias) — Linear projection on the
 *        packed values of a nested tensor. The autograd-aware path is the
 *        Variable-level matmul + add composition in
 *        autograd::nested_linear (see src/autograd/nested_autograd_ops.cpp)
 *        which builds the graph from existing MatMul / Add Function nodes
 *        and therefore carries gradients to `weight` and `bias` through
 *        the standard linear adjoint. A Function-layer adjoint for the
 *        fused OpId::NestedLinear is redundant when the values are
 *        layout-equivalent to a dense [N, in_features] matrix (which is
 *        the only layout the fused kernel supports), so direct OpId
 *        dispatch routed through autograd raises NonDifferentiable; use
 *        autograd::nested_linear for autograd-aware nested linear.
 */
class NestedLinearBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "NestedLinearBackward"; }
    auto op_id() const -> OpId override { return OpId::NestedLinear; }
};

/**
 * @brief quantized_linear(input, packed_weight, bias, scale, zero_point)
 *        — fused INT8/INT4 quantized inference linear kernel. The forward
 *        dequantizes the packed weight, performs the matmul in the
 *        accumulator dtype, requantizes, and clamps to the output dtype's
 *        representable range. Quantized inference operates on a frozen
 *        weight (and frozen scale / zero_point), so the standard PyTorch
 *        convention is that this op is not differentiable in any input:
 *        the weight is integer-packed (no gradient w.r.t. an integer
 *        lattice), and the activation Jacobian goes through the
 *        clamp-round step which is piecewise-constant a.e. The
 *        quantization-aware-training (QAT) path in this project is the
 *        Variable-level fake-quant composition in nn::FakeQuantize
 *        (`x + StopGradient(quantize_dequantize(x) - x)` STE), which
 *        threads gradients through the activation only. Direct OpId
 *        dispatch routed through autograd raises NonDifferentiable; use
 *        nn::FakeQuantize + nn::Linear for QAT-aware quantized linear.
 */
class QuantizedLinearBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "QuantizedLinearBackward"; }
    auto op_id() const -> OpId override { return OpId::QuantizedLinear; }
};

/**
 * @brief quantized_conv2d(input, packed_weight, bias, scale, zero_point,
 *        stride, padding, dilation, groups) — fused INT8/INT4 quantized
 *        inference 2-D convolution. Same rationale as
 *        QuantizedLinearBackward: integer-packed frozen weight + clamp/
 *        round requantization step => not differentiable through the
 *        fused kernel. The QAT path is nn::FakeQuantize +
 *        nn::Conv2d (with weight fake-quant applied at the parameter,
 *        not the input). NonDifferentiable; use the Variable-level
 *        composition for QAT-aware quantized conv.
 */
class QuantizedConv2dBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto name() const -> std::string override { return "QuantizedConv2dBackward"; }
    auto op_id() const -> OpId override { return OpId::QuantizedConv2d; }
};

} // namespace tenzor
