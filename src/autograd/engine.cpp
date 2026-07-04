#include "tenzor/autograd/engine.hpp"
#include "tenzor/autograd/anomaly_mode.hpp"
#include "tenzor/autograd/profiler.hpp"
#include "tenzor/utils/log.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/type_promotion.hpp"
#include "tenzor/utils/error.hpp"
#include <unordered_set>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <typeinfo>
#include <cassert>

#ifdef __GNUC__
#include <cxxabi.h>
#endif

namespace tenzor {

// Debug-mode guard to verify the single-lock invariant: the backward engine
// must never hold two grad_mutex_ locks simultaneously.  This makes deadlock
// structurally impossible (see engine.hpp thread-safety documentation).
#ifndef NDEBUG
namespace {
thread_local int grad_mutex_hold_count_ = 0;

struct GradMutexDebugGuard {
    GradMutexDebugGuard() {
        assert(grad_mutex_hold_count_ == 0 &&
               "BackwardEngine: holding >1 grad_mutex_ simultaneously — "
               "this violates the single-lock invariant and risks deadlock");
        ++grad_mutex_hold_count_;
    }
    ~GradMutexDebugGuard() { --grad_mutex_hold_count_; }
    GradMutexDebugGuard(const GradMutexDebugGuard&) = delete;
    GradMutexDebugGuard& operator=(const GradMutexDebugGuard&) = delete;
};
} // anonymous namespace
#endif

// RAII scope guard for exception-safe cleanup
namespace {
struct ScopeGuard {
    std::function<void()> fn;
    bool dismissed = false;
    ~ScopeGuard() { if (!dismissed && fn) fn(); }
    void dismiss() { dismissed = true; }
};
} // anonymous namespace

// Check computed gradients for NaN/Inf when anomaly detection is enabled.
// Zero overhead when disabled (thread-local bool early-return).
static void check_for_anomaly(const std::vector<Tensor>& grads,
                              const Function* func) {
    if (!is_anomaly_detection_enabled()) return;

    for (size_t i = 0; i < grads.size(); ++i) {
        const auto& grad = grads[i];
        if (grad.numel() == 0) continue;

        // HH.3: complex grads (Complex64/Complex128) carry NaN/Inf in either
        // the real or imaginary part. Decompose via real()/imag() and OR the
        // masks together so the count + diagnostic reflects all anomalies.
        Tensor probe;
        if (grad.is_complex()) {
            // real()/imag() return real-typed views; combine masks across
            // both components so a NaN in either part is detected.
            Tensor re = tenzor::real(grad);
            Tensor im = tenzor::imag(grad);
            auto nan_mask = logical_or(isnan(re), isnan(im));
            auto inf_mask = logical_or(isinf(re), isinf(im));
            auto nan_count = sum(nan_mask.to(DType::Int64));
            auto inf_count = sum(inf_mask.to(DType::Int64));
            bool has_nan = nan_count.item<int64_t>() > 0;
            bool has_inf = inf_count.item<int64_t>() > 0;
            if (has_nan || has_inf) {
                std::string func_name = typeid(*func).name();
#ifdef __GNUC__
                int status = 0;
                char* demangled = abi::__cxa_demangle(func_name.c_str(), nullptr, nullptr, &status);
                if (status == 0 && demangled) {
                    func_name = demangled;
                    free(demangled);
                }
#endif
                std::string anomaly;
                if (has_nan && has_inf) anomaly = "NaN and Inf";
                else if (has_nan) anomaly = "NaN";
                else anomaly = "Inf";
                std::string detail;
                detail += "\n  Gradient shape: [";
                for (size_t d = 0; d < grad.shape().size(); ++d) {
                    if (d > 0) detail += ", ";
                    detail += std::to_string(grad.shape()[d]);
                }
                detail += "]";
                detail += "\n  NaN count: " + std::to_string(nan_count.item<int64_t>());
                detail += "\n  Inf count: " + std::to_string(inf_count.item<int64_t>());
                detail += "\n  (complex grad — counts merged over real+imag components)";
                throw AutogradException(
                    "Anomaly detected: gradient output " + std::to_string(i) +
                    " contains " + anomaly + " values in backward of '" +
                    func_name + "'" + detail);
            }
            continue;
        }

        // Only check floating-point gradients (integer grads can't be NaN/Inf)
        if (grad.dtype() != DType::Float32 && grad.dtype() != DType::Float64 &&
            grad.dtype() != DType::Float16 && grad.dtype() != DType::BFloat16) {
            continue;
        }

        auto nan_mask = isnan(grad);
        auto inf_mask = isinf(grad);

        // S.3 — sum mask in Int64 so the count is exact for tensors above
        // 2^24 elements. Float32 sum silently saturates beyond that, making
        // the reported "NaN count: N" diagnostic wrong for 16M+ element
        // grads. Int64 is also the natural dtype for an integer count.
        auto nan_count = sum(nan_mask.to(DType::Int64));
        auto inf_count = sum(inf_mask.to(DType::Int64));

        bool has_nan = nan_count.item<int64_t>() > 0;
        bool has_inf = inf_count.item<int64_t>() > 0;

        if (has_nan || has_inf) {
            // Get demangled function name for readability
            std::string func_name = typeid(*func).name();
#ifdef __GNUC__
            int status = 0;
            char* demangled = abi::__cxa_demangle(func_name.c_str(), nullptr, nullptr, &status);
            if (status == 0 && demangled) {
                func_name = demangled;
                free(demangled);
            }
#endif
            std::string anomaly;
            if (has_nan && has_inf) anomaly = "NaN and Inf";
            else if (has_nan) anomaly = "NaN";
            else anomaly = "Inf";

            // Build detailed diagnostic (only computed on error path)
            std::string detail;

            // Gradient shape
            detail += "\n  Gradient shape: [";
            for (size_t d = 0; d < grad.shape().size(); ++d) {
                if (d > 0) detail += ", ";
                detail += std::to_string(grad.shape()[d]);
            }
            detail += "]";

            // Anomaly counts (S.3 — Int64 to avoid Float32 saturation beyond 2^24).
            detail += "\n  NaN count: " + std::to_string(nan_count.item<int64_t>());
            detail += "\n  Inf count: " + std::to_string(inf_count.item<int64_t>());

            // Finite value statistics — count in Int64 for the same reason.
            auto finite_mask = logical_not(logical_or(nan_mask, inf_mask));
            Tensor finite_count_t = sum(finite_mask.to(DType::Int64));
            int64_t finite_count_val = finite_count_t.item<int64_t>();
            if (finite_count_val > 0) {
                auto grad_shape_vec = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
                // S.3 (finite-stat widening): reduce the diagnostic min/max/mean
                // in Float64 (not Float32). A Float32 sum() saturates beyond 2^24,
                // so Float64 gradients were being narrowed in the reported mean.
                // This is purely diagnostic — no gradient VALUE is affected.
                Tensor grad_f64 = (grad.dtype() != DType::Float64) ? grad.to(DType::Float64) : grad;
                Tensor zero_t = zeros(grad_shape_vec, DType::Float64, grad.device());
                Tensor finite_vals = where(finite_mask, grad_f64, zero_t);
                Tensor sum_t = sum(finite_vals);
                double finite_sum = sum_t.item<double>();
                // For min: replace non-finite with +max so they don't affect min
                Tensor max_fill = full(grad_shape_vec, std::numeric_limits<double>::max(), DType::Float64, grad.device());
                Tensor for_min = where(finite_mask, grad_f64, max_fill);
                Tensor min_t = min(for_min);
                double min_val = min_t.item<double>();
                // For max: replace non-finite with -max so they don't affect max
                Tensor min_fill = full(grad_shape_vec, std::numeric_limits<double>::lowest(), DType::Float64, grad.device());
                Tensor for_max = where(finite_mask, grad_f64, min_fill);
                Tensor max_t = max(for_max);
                double max_val = max_t.item<double>();
                detail += "\n  Finite values: min=" + std::to_string(min_val)
                        + " max=" + std::to_string(max_val)
                        + " mean=" + std::to_string(finite_sum / finite_count_val);
            }

            // Saved tensor shapes (catch exceptions since tensors may have been modified)
            try {
                const auto& saved = func->saved_tensors();
                if (!saved.empty()) {
                    detail += "\n  Saved tensor shapes:";
                    for (size_t j = 0; j < saved.size(); ++j) {
                        detail += " [" + std::to_string(j) + "]=(";
                        for (size_t d = 0; d < saved[j].shape().size(); ++d) {
                            if (d > 0) detail += ", ";
                            detail += std::to_string(saved[j].shape()[d]);
                        }
                        detail += ")";
                    }
                }
            } catch (...) {
                // Ignore errors from saved tensor validation
            }

            // Include forward-pass creation metadata if available
            const auto& input_vars = func->input_variables();
            for (size_t v = 0; v < input_vars.size(); ++v) {
                const auto& meta = input_vars[v].creation_metadata();
                if (meta) {
                    detail += "\n  Forward traceback for input " + std::to_string(v) + ":";
                    detail += "\n" + meta->to_string();
                }
            }

            throw AutogradException(
                "Anomaly detected: gradient output " + std::to_string(i) +
                " contains " + anomaly + " values in backward of '" +
                func_name + "'" + detail);
        }
    }
}

// Use the shared promote_types() for gradient dtype promotion.
// This replaces the old local promote_types() that duplicated type_promotion.cpp logic.

// HH.1: shared shape-check used by both single-root (execute) and multi-root
// (execute_multi) accumulation loops. Previously only the single-root path
// validated; multi-root silently let a mismatched grad through and crashed
// later inside accumulate_unlocked or in the user's hook.
static void validate_grad_shape_or_throw(const Variable& var,
                                         const Tensor& grad,
                                         const std::string& func_name,
                                         size_t input_index) {
    auto grad_shape = grad.shape();
    auto var_shape = var.tensor().shape();
    if (grad_shape.size() == var_shape.size() &&
        std::equal(grad_shape.begin(), grad_shape.end(), var_shape.begin())) {
        return;
    }
    std::string expected, got;
    for (size_t s = 0; s < var_shape.size(); ++s) {
        if (s > 0) expected += ",";
        expected += std::to_string(var_shape[s]);
    }
    for (size_t s = 0; s < grad_shape.size(); ++s) {
        if (s > 0) got += ",";
        got += std::to_string(grad_shape[s]);
    }
    throw AutogradException(
        "In " + func_name + ".backward(): gradient for input " +
        std::to_string(input_index) + " has shape [" + got +
        "], expected [" + expected + "]");
}

auto BackwardEngine::synthesize_or_validate_root_grad(const Variable& root,
                                                      std::optional<Tensor> user_grad) -> Tensor {
    if (!user_grad.has_value()) {
        if (root.tensor().numel() != 1) {
            // PyTorch message wording: keep parity with torch.autograd.
            throw AutogradException(
                "grad can be implicitly created only for scalar outputs "
                "(tensor has " + std::to_string(root.tensor().numel()) + " elements)");
        }
        return ones_like(root.tensor());
    }

    auto grad_shape = user_grad->shape();
    auto root_shape = root.tensor().shape();
    if (grad_shape.size() != root_shape.size() ||
        !std::equal(grad_shape.begin(), grad_shape.end(), root_shape.begin())) {
        auto fmt = [](std::span<const int64_t> s) {
            std::string r = "[";
            for (size_t i = 0; i < s.size(); ++i) {
                if (i > 0) r += ", ";
                r += std::to_string(s[i]);
            }
            return r + "]";
        };
        throw AutogradException(
            "User-supplied gradient shape mismatch: expected " +
            fmt(root_shape) + " got " + fmt(grad_shape));
    }
    // Audit-7 EE.2: also reject dtype/device mismatch — silent .to() coercion
    // hides bugs (e.g. F64 seed on F32 root, or CPU seed on CUDA root).
    if (user_grad->dtype() != root.tensor().dtype() ||
        user_grad->device() != root.tensor().device()) {
        throw AutogradException(
            std::string("Mismatched grad dtype/device: expected ") +
            std::string(dtype_name(root.tensor().dtype())) + " on " +
            root.tensor().device().to_string() + ", got " +
            std::string(dtype_name(user_grad->dtype())) + " on " +
            user_grad->device().to_string());
    }
    // audit-11 QQ.6: reject non-floating-point / non-complex seed dtypes.
    // Integer/Bool seeds silently accumulated into Float param grads as
    // garbage; PyTorch raises RuntimeError ("only Tensors of floating point
    // and complex dtype can require gradients" / "grad can be created only
    // for floating-point Tensors"). Mirror that wording.
    if (!user_grad->is_floating_point() && !user_grad->is_complex()) {
        throw AutogradException(
            std::string("User-supplied gradient must be floating-point or "
                        "complex; got ") +
            std::string(dtype_name(user_grad->dtype())));
    }
    return std::move(*user_grad);
}

auto BackwardEngine::execute(Variable& root, std::optional<Tensor> gradient,
                             bool retain_graph, bool create_graph) -> void {
    if (!root.requires_grad()) {
        return;
    }

    // audit-6 BB.2: route through shared helper so single-root and multi-root
    // paths agree on scalar-implicit-ones and shape-mismatch behaviour.
    Tensor seed = synthesize_or_validate_root_grad(root, std::move(gradient));

    // PyTorch semantics: a non-leaf root does not retain .grad (it stays None)
    // unless retain_grad() was requested; only a leaf root's own .grad is the
    // result. Storing the seed on a non-leaf root polluted loss.grad() with
    // ones_like(loss) and defeated the retains_grad diagnostic. The backward
    // traversal is seeded from the local `seed` below, so it is unaffected.
    if (!root.grad_fn() || root.is_leaf() || root.retains_grad()) {
        root.set_grad(seed);
    }

    // If no grad_fn, this is a leaf variable, nothing to backprop
    if (!root.grad_fn()) {
        return;
    }

    // Save and clear grad_accumulators_ for re-entrancy safety.
    // Nested backward calls (from checkpointing) must use independent
    // accumulator maps to avoid corrupting the outer call's state.
    auto saved_accumulators = std::move(grad_accumulators_);
    grad_accumulators_.clear();
    auto saved_accumulators_var = std::move(grad_accumulators_var_);
    grad_accumulators_var_.clear();

    // Install the restore guards IMMEDIATELY after moving the maps out, before
    // any code that can throw (topological_sort throws on a detected cycle). If
    // they were installed later, a throw in between would destroy the moved-out
    // maps unrestored, permanently wiping an OUTER backward's accumulators
    // (matches the fix already applied in execute_multi()).
    ScopeGuard accum_guard{[&]{ grad_accumulators_ = std::move(saved_accumulators); }};
    ScopeGuard accum_guard_var{[&]{ grad_accumulators_var_ = std::move(saved_accumulators_var); }};

    // Topological sort from root
    // Use a local variable (not the instance cache) to be re-entrant safe.
    // Nested backward calls (e.g. from gradient checkpointing) invoke
    // execute() on the same thread-local engine; a shared cache would be
    // overwritten, invalidating the outer call's iterators.
    auto sorted = topological_sort(root.grad_fn());

    // Set the create_graph flag so backward functions know to use Variable ops
    // Use RAII guard to ensure flag is restored even on exception
    std::optional<CreateGraphGuard> graph_guard;
    if (create_graph) {
        graph_guard.emplace();
    }

    // Helper to clean up computation graph (breaks circular references)
    auto cleanup_graph = [&]() {
        if (!retain_graph) {
            for (auto& func : sorted) {
                if (func) {
                    // Clear grad_fn on intermediate (non-leaf) input variables
                    // to break reference cycles that would otherwise leak memory.
                    for (auto& var : func->input_variables()) {
                        if (var.grad_fn() && !var.is_leaf()) {
                            var.set_grad_fn(nullptr);
                        }
                    }
                    func->set_input_variables({});
                    func->set_next_functions({});
                }
            }
            sorted.clear();
            if (root.grad_fn() && !root.is_leaf()) {
                root.set_grad_fn(nullptr);
            }
        }
    };

    // RAII guard for exception-safe graph cleanup (accumulator-restore guards
    // are installed earlier, right after the maps are moved out).
    ScopeGuard cleanup_guard{[&]{ clear_gradients(); cleanup_graph(); }};

    // HH.4: track leaves that actually received an accumulation during this
    // backward so the dtype-downcast finalization loop visits them all.
    // The pre-existing loop iterated `func->input_variables()` which misses
    // leaves reached purely through the `next_funcs` chain (a leaf whose
    // accumulator was seeded by accumulate_grad on a downstream Function
    // never appears as an `input_variables_` entry of that Function).
    std::unordered_set<VariableImpl*> touched_leaves;

    // Seed root gradient into accumulator so the root function is handled
    // uniformly — no fragile "empty accumulators = root" assumption.
    accumulate_grad(root.grad_fn().get(), seed);
    if (create_graph) {
        // Seed the graph-carrying accumulator. requires_grad=true mirrors the
        // legacy `Variable(g, true)` wrap so that downstream
        // backward_with_variables products keep building grad_fn (a detached
        // seed would make `grad * saved` non-requiring and sever the second-
        // order graph). The seed itself is a leaf (no grad_fn), so forward-mode
        // JVP / further differentiation correctly treat it as a constant.
        accumulate_grad_var(root.grad_fn().get(),
                            Variable(seed, /*requires_grad=*/true));
    }

    {
        // Execute backward in reverse topological order
        for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
            // Check if shared_ptr is valid before dereferencing
            if (!*it) {
                continue;
            }

            auto& function = *it;

            // Get the gradient for this function's output
            std::vector<Tensor> grad_outputs;

            const auto& accum_grads = get_accumulated_grads(function.get());
            if (accum_grads.empty()) {
                continue;  // No gradient flows to this function
            }

            // Sum all accumulated gradients with dtype promotion.
            // Pre-compute target dtype to avoid repeated conversions.
            DType target_dtype = accum_grads[0].dtype();
            for (size_t i = 1; i < accum_grads.size(); ++i) {
                target_dtype = promote_types(target_dtype, accum_grads[i].dtype());
            }
            Tensor total_grad = (accum_grads[0].dtype() == target_dtype)
                ? accum_grads[0]
                : accum_grads[0].to(target_dtype);
            for (size_t i = 1; i < accum_grads.size(); ++i) {
                const Tensor& gi = accum_grads[i];
                total_grad = total_grad + (gi.dtype() == target_dtype ? gi : gi.to(target_dtype));
            }
            grad_outputs.push_back(total_grad);

            // Reload offloaded saved tensors back to GPU before backward
            function->reload_saved_tensors();

            // Validate saved tensors haven't been modified in-place since forward
            function->validate_saved_tensors();

            // Check incoming gradients for NaN/Inf before calling backward
            check_for_anomaly(grad_outputs, function.get());

            // Compute gradients for inputs
            std::vector<Tensor> input_grads;

            std::vector<Variable> var_input_grads;  // only populated when create_graph
            {
                // Optional profiling: time the backward function call
                std::optional<BackwardTimer> _profiler_timer;
                if (AutogradProfiler::instance().is_enabled()) {
                    _profiler_timer.emplace(function->name());
                }

                if (create_graph) {
                    // Higher-order gradient path: use backward_with_variables
                    // which operates on Variables so the backward computation itself
                    // is tracked by autograd (enabling gradients of gradients).
                    std::vector<Variable> var_grad_outputs =
                        build_var_grad_outputs(function.get(), grad_outputs);

                    var_input_grads = function->backward_with_variables(var_grad_outputs);

                    if (function->is_higher_order_stub()) {
                        auto mode = get_higher_order_grad_mode();
                        if (mode == HigherOrderGradMode::Error) {
                            throw std::runtime_error(
                                "create_graph=true but '" + function->name() +
                                "' is a higher-order stub (second derivatives are zero). "
                                "Use set_higher_order_grad_mode(Warn) to allow this.");
                        }
                        detail::increment_higher_order_disconnection_count();
                        // Audit I.4: unified logger so TENZOR_LOG_LEVEL filter applies.
                        TENZOR_LOG_WARN("[tenzor::autograd] '{}' is a higher-order stub — "
                                        "second derivatives through it will be zero. "
                                        "(disconnection #{})",
                                        function->name(),
                                        higher_order_disconnection_count());
                    }

                    // Extract raw tensors for the standard accumulation path (unchanged),
                    // but keep var_input_grads alive so we can store the graph-carrying
                    // Variable onto leaves below.
                    input_grads.reserve(var_input_grads.size());
                    for (auto& vg : var_input_grads) {
                        input_grads.push_back(vg.tensor());
                    }
                } else {
                    // Standard backward path: raw Tensor operations
                    input_grads = function->backward(grad_outputs);
                }
            } // _profiler_timer destroyed here, records elapsed time

            // Check for NaN/Inf in computed gradients when anomaly detection is on
            check_for_anomaly(input_grads, function.get());

            // Validate gradient dtypes are floating-point or complex
            for (size_t i = 0; i < input_grads.size(); ++i) {
                const auto& g = input_grads[i];
                if (g.is_valid() && g.numel() > 0 &&
                    !g.is_floating_point() && !g.is_complex()) {
                    throw std::runtime_error(
                        std::string("Backward function ") + function->name() +
                        " returned non-floating-point gradient (dtype=" +
                        std::string(dtype_name(g.dtype())) + " at index " +
                        std::to_string(i) +
                        "). Gradients must be floating-point or complex.");
                }
            }

            // Accumulate gradients to input variables
            auto& input_vars = function->input_variables();

            if (input_grads.size() < input_vars.size()) {
                // Backward returned fewer gradients than inputs — check if any
                // skipped inputs actually require grad (likely a bug in backward())
                size_t skipped_requiring_grad = 0;
                for (size_t i = input_grads.size(); i < input_vars.size(); ++i) {
                    if (input_vars[i].requires_grad()) {
                        ++skipped_requiring_grad;
                    }
                }
                if (skipped_requiring_grad > 0) {
                    // Audit I.4: unified logger.
                    TENZOR_LOG_WARN("[tenzor::autograd] backward function '{}' returned "
                                    "{} gradients but has {} inputs ({} skipped inputs "
                                    "require grad)",
                                    function->name(),
                                    input_grads.size(),
                                    input_vars.size(),
                                    skipped_requiring_grad);
                }
            }

            for (size_t i = 0; i < input_vars.size() && i < input_grads.size(); ++i) {
                Variable& var = input_vars[i];

                // Skip Variables without gradients (default-constructed with requires_grad=false)
                if (!var.requires_grad()) {
                    continue;
                }

                Tensor grad_to_apply = input_grads[i];

                // HH.1: validate gradient shape matches variable shape via the
                // shared helper used by both single-root and multi-root paths.
                validate_grad_shape_or_throw(var, grad_to_apply, function->name(), i);

                // Apply hooks (access through impl_ for handle pattern)
                // Take shared_lock for thread-safe iteration, then copy hooks
                // to local before iterating — a hook may register/unregister
                // Capture the hooks at this wider scope so the create_graph path
                // below can apply the SAME transform to the graph-carrying gradient.
                std::map<size_t, std::function<Tensor(const Tensor&)>> applied_hooks;
                if (var.impl_) {
                    {
                        std::shared_lock lock(var.impl_->hooks_mutex_);
                        if (!var.impl_->hooks_.empty()) {
                            applied_hooks = var.impl_->hooks_;
                        }
                    }
                    for (auto& [id, hook] : applied_hooks) {
                        grad_to_apply = hook(grad_to_apply);
                    }
                }

                // PyTorch semantics: a backward hook transforms the gradient
                // flowing THROUGH this variable, so the hooked gradient must
                // also propagate to downstream Functions (next_functions),
                // not just leaf accumulation. Write the transformed gradient
                // back now — before accumulate_unlocked may promote its dtype
                // for leaf storage — so the propagation loop below forwards the
                // hooked value. (For a non-leaf this is the whole point of a
                // hook, e.g. gradient reversal / clipping.)
                if (!applied_hooks.empty()) {
                    input_grads[i] = grad_to_apply;
                    // Mirror the hook transform onto the graph-carrying cotangent
                    // so BOTH the leaf grad_with_graph store and the
                    // next_functions propagation below forward the hooked
                    // Variable. Previously accumulate_grad_var() (create_graph)
                    // forwarded an un-hooked var_input_grads[i] while the
                    // raw/value path (input_grads[i]) above was hooked —
                    // inconsistent gradients under backward hooks.
                    if (create_graph && i < var_input_grads.size()) {
                        Tensor hv = var_input_grads[i].tensor();
                        for (auto& [id, hook] : applied_hooks) hv = hook(hv);
                        var_input_grads[i] =
                            Variable(hv, var_input_grads[i].requires_grad());
                    }
                }

                // Accumulate gradient to leaf variables.
                // When thread_safe_ is enabled, lock to prevent concurrent
                // accumulation from corrupting the gradient tensor.
                if (var.is_leaf() || var.retains_grad()) {
                    // Access impl_->grad_ directly under the lock to avoid
                    // re-entrant locking (has_grad/set_grad also lock grad_mutex_
                    // when thread_safe_ is true, causing self-deadlock).
                    auto accumulate_unlocked = [&]() {
                        // V.4: Accumulate in the promoted dtype of (existing,
                        // incoming) so summing F32 grads into an F16 leaf does
                        // not downcast each incoming grad to F16 before the
                        // add. Downcasting per-step truncates ~3 bits of
                        // mantissa per accumulation; PyTorch keeps the
                        // gradient at the upstream dtype throughout backward
                        // and downcasts only on the user-facing .grad() read.
                        if (var.impl_->grad_.has_value()) {
                            auto existing_grad = var.impl_->grad_.value();
                            DType target = promote_types(existing_grad.dtype(),
                                                          grad_to_apply.dtype());
                            if (existing_grad.dtype() != target) {
                                existing_grad = existing_grad.to(target);
                            }
                            if (grad_to_apply.dtype() != target) {
                                grad_to_apply = grad_to_apply.to(target);
                            }
                            var.impl_->grad_ = existing_grad + grad_to_apply;
                        } else {
                            var.impl_->grad_ = grad_to_apply;
                        }
                    };

                    if (var.impl_) {
                        // NN.4: snapshot existing graph impl under the lock,
                        // compute the Variable sum OUTSIDE the lock (some
                        // AddBackward.forward paths can construct Variables
                        // that touch grad_mutex_ — running the sum under the
                        // lock risked a recursive lock for checkpoint-recovered
                        // leaves), then re-acquire briefly to store the result.
                        std::shared_ptr<VariableImpl> existing_graph_impl;
                        bool need_graph_update = create_graph && i < var_input_grads.size();
                        {
#ifndef NDEBUG
                            GradMutexDebugGuard single_lock_check;
#endif
                            std::lock_guard lock(*var.impl_->grad_mutex_);
                            accumulate_unlocked();
                            // HH.4: record the leaf as touched so the finalization
                            // loop downcasts its grad to its own dtype if mismatched.
                            touched_leaves.insert(var.impl_.get());
                            if (need_graph_update) {
                                existing_graph_impl = var.impl_->grad_with_graph_impl_;
                            }
                        }

                        // GG.8: when a leaf receives gradients via multiple paths
                        // (e.g. loss = x*x + sin(x)), the FIRST accumulation
                        // stored only var_input_grads[i] and subsequent
                        // contributions were added at the raw-Tensor level —
                        // the Variable graph reflected only the first path
                        // and second-derivative through branchy higher-order
                        // forwards dropped a path. Fix: sum at the Variable
                        // level so grad_fn chains through all contributing
                        // paths.
                        if (need_graph_update) {
                            // Apply the SAME backward hooks to the graph-carrying
                            // gradient so grad_variable() (consumed by second-order
                            // diff: Hessian/HVP, WGAN-GP, MAML) reflects the hook,
                            // not just .grad(). Hooks are opaque Tensor->Tensor
                            // maps, so the second derivative does not flow THROUGH
                            // the hook (inherent to Tensor hooks), but the gradient
                            // VALUE is now consistent between .grad() and grad_variable().
                            // var_input_grads[i] was already hook-transformed
                            // above (in lockstep with input_grads[i]), so the
                            // graph store, leaf .grad(), and next_functions
                            // propagation all see the same hooked cotangent.
                            Variable graph_grad = var_input_grads[i];
                            // E-01 (lost-update race): the read-modify-write of
                            // grad_with_graph_impl_ is split across the unlock
                            // window (the Variable sum is computed OUTSIDE the
                            // lock to avoid a re-entrant grad_mutex_ lock). Two
                            // concurrent create_graph backwards could snapshot
                            // the same existing impl and the second store would
                            // clobber the first. Use a compare-and-retry loop:
                            // recompute the combine from a fresh snapshot and
                            // only commit if grad_with_graph_impl_ is still the
                            // value we based our sum on. The value-grad
                            // accumulation (accumulate_unlocked) already ran
                            // exactly once above; ONLY this graph-grad combine
                            // retries. In the single-threaded / non-contended
                            // path the CAS succeeds on the first iteration, so
                            // behavior is unchanged.
                            std::shared_ptr<VariableImpl> snapshot = existing_graph_impl;
                            while (true) {
                                std::shared_ptr<VariableImpl> new_impl;
                                if (!snapshot) {
                                    new_impl = graph_grad.impl_;
                                } else {
                                    Variable existing_var;
                                    existing_var.impl_ = snapshot;
                                    new_impl =
                                        (existing_var + graph_grad).impl_;
                                }
                                {
#ifndef NDEBUG
                                    GradMutexDebugGuard single_lock_check;
#endif
                                    std::lock_guard lock(*var.impl_->grad_mutex_);
                                    if (var.impl_->grad_with_graph_impl_ == snapshot) {
                                        // Unchanged since our snapshot → commit.
                                        var.impl_->grad_with_graph_impl_ = new_impl;
                                        var.impl_->grad_with_graph_cache_storage_.reset();
                                        break;
                                    }
                                    // Another thread updated it; reload the
                                    // current value and retry the combine so its
                                    // contribution is not lost.
                                    snapshot = var.impl_->grad_with_graph_impl_;
                                }
                            }
                        }
                    } else {
                        var.set_grad(grad_to_apply);
                    }
                }
            }

            // Also accumulate to next functions for non-leaf variables
            const auto& next_funcs = function->next_functions();

            for (size_t i = 0; i < next_funcs.size() && i < input_grads.size(); ++i) {
                if (next_funcs[i]) {
                    accumulate_grad(next_funcs[i].get(), input_grads[i]);
                    // Under create_graph, also forward the graph-carrying
                    // gradient Variable so the downstream Function receives a
                    // connected cotangent (keeps the second-order graph intact
                    // across this edge). var_input_grads is index-aligned with
                    // input_grads (both derived from the same backward call).
                    if (create_graph && i < var_input_grads.size()) {
                        accumulate_grad_var(next_funcs[i].get(), var_input_grads[i]);
                    }
                }
            }

            // Release saved tensors after all gradient accumulation and hook
            // execution is complete. This must happen after hooks since hooks
            // may access saved tensors via closures.
            if (!retain_graph) {
                function->release_saved_tensors();
                // audit-9 JJ.2: also drop saved_variables_ so the higher-order
                // graph released here can actually be freed.  Each Function
                // whose backward_with_variables saved Variables holds grad_fn
                // chains across the second-order graph; without this clear,
                // create_graph=true workloads leak the entire higher-order
                // graph even when retain_graph is false.
                function->clear_saved_variables();
                // audit-10 NN.6: release per-Function ad-hoc state (checkpoint
                // recompute caches, FusedLinearReLUBackward::relu_output_,
                // sparse transposed-CSR caches, etc.) so they don't survive
                // until the Function destructs.
                function->release_op_specific_state();
            }
        }
    }

    // Audit-6 AA.7: V.4 accumulation preserves the upstream (promoted) dtype
    // throughout backward to avoid per-step half-precision rounding. The
    // user-facing `leaf.grad` contract is `leaf.grad.dtype == leaf.dtype`
    // (PyTorch); downcast every leaf that ended up with a promoted gradient.
    // This is a no-op for the common F32/F32 case.
    //
    // Audit-7 EE.1: opt-out via Variable::set_preserve_grad_dtype(true) for
    // the standard AMP "fp32 master weights" pattern (F16/BF16 params with
    // F32 grads). When set, we leave the promoted gradient as-is.
    //
    // HH.4: iterate the touched_leaves set captured at every accumulation
    // site so leaves reached purely through next_funcs accumulation chain
    // (not present in any sorted function's input_variables()) are still
    // downcast. The pre-existing `for (auto& func : sorted)` loop missed
    // those, leaving such leaves' .grad in the promoted dtype.
    for (auto* leaf_impl : touched_leaves) {
        if (!leaf_impl) continue;
        std::lock_guard lock(*leaf_impl->grad_mutex_);
        if (!leaf_impl->grad_.has_value()) continue;
        const auto leaf_dt = leaf_impl->data_.dtype();
        if (leaf_impl->grad_->dtype() != leaf_dt &&
            !leaf_impl->preserve_grad_dtype_.load(std::memory_order_acquire)) {
            leaf_impl->grad_ = leaf_impl->grad_->to(leaf_dt);
        }
    }
    // Also handle the root itself when it is a leaf with grad (the root
    // gradient was assigned in user-supplied dtype but accumulate paths may
    // have promoted it). Same dtype-equality fast path.
    if (root.requires_grad() && (root.is_leaf() || root.retains_grad()) &&
        root.impl_) {
        std::lock_guard lock(*root.impl_->grad_mutex_);
        if (root.impl_->grad_.has_value()) {
            const auto root_dt = root.dtype();
            if (root.impl_->grad_->dtype() != root_dt &&
                !root.impl_->preserve_grad_dtype_.load(std::memory_order_acquire)) {
                root.impl_->grad_ = root.impl_->grad_->to(root_dt);
            }
        }
    }
    // ScopeGuards handle cleanup in both normal and exception paths
}

auto BackwardEngine::topological_sort(std::shared_ptr<Function> root)
    -> std::vector<std::shared_ptr<Function>> {
    std::vector<std::shared_ptr<Function>> sorted;
    std::unordered_set<Function*> visited;
    std::unordered_set<Function*> recursion_stack;

    // Iterative DFS-based topological sort (avoids stack overflow on deep graphs)
    struct Frame {
        std::shared_ptr<Function> node;
        size_t child_idx;  // which child to visit next
    };
    std::vector<Frame> stack;
    stack.push_back({root, 0});
    visited.insert(root.get());
    recursion_stack.insert(root.get());

    while (!stack.empty()) {
        auto& frame = stack.back();
        auto& node = frame.node;

        if (!node) {
            stack.pop_back();
            continue;
        }

        const auto& children = node->next_functions();

        if (frame.child_idx < children.size()) {
            // Process next child
            const auto& child = children[frame.child_idx++];
            if (!child) continue;

            if (recursion_stack.count(child.get())) {
                throw AutogradException("Cycle detected in computation graph");
            }
            if (visited.count(child.get())) {
                continue;
            }

            visited.insert(child.get());
            recursion_stack.insert(child.get());
            stack.push_back({child, 0});
        } else {
            // All children visited — post-order: add to sorted
            recursion_stack.erase(node.get());
            sorted.push_back(std::move(node));
            stack.pop_back();
        }
    }

    return sorted;
}

auto BackwardEngine::clear_gradients() -> void {
    grad_accumulators_.clear();
}

auto BackwardEngine::accumulate_grad(Function* func, Tensor grad) -> void {
    grad_accumulators_[func->id()].push_back(std::move(grad));
}

auto BackwardEngine::accumulate_grad_var(Function* func, Variable grad) -> void {
    grad_accumulators_var_[func->id()].push_back(std::move(grad));
}

auto BackwardEngine::build_var_grad_outputs(Function* func,
                                            const std::vector<Tensor>& raw_grad_outputs)
    -> std::vector<Variable> {
    // Build the graph-carrying grad_outputs handed to backward_with_variables.
    // Prefer the per-Function Variable accumulator (summed at the Variable
    // level) so the second-order graph stays connected across this Function's
    // input edge. Only when no Variable contribution was recorded do we fall
    // back to a detached wrap of the raw total (defensive — should not happen
    // for a reachable Function once the root var-accumulator is seeded).
    std::vector<Variable> var_grad_outputs;
    auto it = grad_accumulators_var_.find(func->id());
    if (it != grad_accumulators_var_.end() && !it->second.empty()) {
        const auto& var_grads = it->second;
        Variable summed = var_grads[0];
        for (size_t k = 1; k < var_grads.size(); ++k) {
            summed = summed + var_grads[k];
        }
        var_grad_outputs.push_back(std::move(summed));
    } else {
        var_grad_outputs.reserve(raw_grad_outputs.size());
        for (const auto& g : raw_grad_outputs) {
            var_grad_outputs.emplace_back(g, true);
        }
    }
    return var_grad_outputs;
}

auto BackwardEngine::get_accumulated_grads(Function* func) -> const std::vector<Tensor>& {
    static const std::vector<Tensor> empty;
    auto it = grad_accumulators_.find(func->id());
    if (it == grad_accumulators_.end()) {
        return empty;
    }
    return it->second;
}

auto BackwardEngine::execute_multi(std::vector<Variable*> roots,
                                   std::vector<Tensor> gradients,
                                   bool retain_graph,
                                   bool create_graph) -> void {
    if (roots.empty()) {
        return;
    }

    if (roots.size() != gradients.size()) {
        throw AutogradException(
            "execute_multi: number of roots (" + std::to_string(roots.size()) +
            ") must match number of gradients (" + std::to_string(gradients.size()) + ")");
    }

    // Save and clear grad_accumulators_ for re-entrancy safety.
    // Nested backward calls (from checkpointing) must use independent
    // accumulator maps to avoid corrupting the outer call's state.
    //
    // V.3: install the restore guard IMMEDIATELY after the std::move, before
    // we run any seeding code that could throw. Previously the guard was
    // installed below (after topo sort + accumulate_grad calls), so any
    // exception from `set_grad` / `accumulate_grad` between the move and
    // the guard install would leak the saved accumulators forever (the
    // outer-thread state would be permanently empty).
    auto saved_accumulators = std::move(grad_accumulators_);
    grad_accumulators_.clear();
    ScopeGuard accum_guard_multi{[&]{ grad_accumulators_ = std::move(saved_accumulators); }};
    auto saved_accumulators_var_multi = std::move(grad_accumulators_var_);
    grad_accumulators_var_.clear();
    ScopeGuard accum_guard_var_multi{[&]{ grad_accumulators_var_ = std::move(saved_accumulators_var_multi); }};

    // Set the create_graph flag so backward functions know to use Variable ops.
    // Use RAII guard to ensure flag is restored even on exception. Same
    // pattern as the single-root execute() above; V.2 brings the multi-root
    // path to parity (torch.autograd.backward([roots], create_graph=True)
    // previously silently degraded to first-order-only).
    std::optional<CreateGraphGuard> graph_guard;
    if (create_graph) {
        graph_guard.emplace();
    }

    // audit-6 BB.2 + BB.3: validate / synthesize each root gradient and merge
    // duplicate roots (last-writer-wins previously dropped contributions when
    // the same Variable appeared multiple times, e.g. autograd.grad((y, y), x)).
    // Sum the gradients for each unique root identity (VariableImpl*) before
    // seeding, so we set_grad / accumulate_grad exactly once per unique root.
    std::unordered_map<VariableImpl*, Tensor> root_seeds;
    std::vector<VariableImpl*> root_order;  // preserve first-seen order
    for (size_t i = 0; i < roots.size(); ++i) {
        if (!roots[i] || !roots[i]->requires_grad()) {
            continue;
        }
        Tensor seed = synthesize_or_validate_root_grad(
            *roots[i],
            std::optional<Tensor>(std::move(gradients[i])));
        auto* key = roots[i]->impl_.get();
        auto it = root_seeds.find(key);
        if (it == root_seeds.end()) {
            root_seeds.emplace(key, std::move(seed));
            root_order.push_back(key);
        } else {
            // Duplicate root: sum gradient contributions (matches the
            // PyTorch semantics of grad_outputs being added together).
            it->second = add(it->second, seed);
        }
    }

    // Now seed each unique root exactly once.
    // We need fast lookup from VariableImpl* back to a Variable* in `roots` —
    // first-occurrence wins so set_grad is observable on the canonical handle.
    std::unordered_map<VariableImpl*, Variable*> first_root_handle;
    for (auto* r : roots) {
        if (!r) continue;
        auto* key = r->impl_.get();
        first_root_handle.emplace(key, r);  // emplace = no-op if already present
    }
    for (auto* key : root_order) {
        Variable* r = first_root_handle[key];
        // Only leaf / retains_grad roots keep .grad (PyTorch semantics; see
        // execute()); a non-leaf root stays None. Backprop is seeded from
        // root_seeds below, so gating this does not affect the traversal.
        if (!r->grad_fn() || r->is_leaf() || r->retains_grad()) {
            r->set_grad(root_seeds[key]);
        }
    }

    // Build combined topological sort from all roots using iterative DFS
    // (iterative to avoid stack overflow on deep computation graphs)
    std::vector<std::shared_ptr<Function>> sorted;
    std::unordered_set<Function*> visited;
    std::unordered_set<Function*> on_stack;

    struct DFSFrame {
        std::shared_ptr<Function> node;
        size_t child_idx;
    };
    std::vector<DFSFrame> stack;

    for (auto* root : roots) {
        if (!root || !root->grad_fn()) continue;
        auto root_fn = root->grad_fn();
        if (visited.count(root_fn.get())) continue;

        stack.push_back({root_fn, 0});
        visited.insert(root_fn.get());
        on_stack.insert(root_fn.get());

        while (!stack.empty()) {
            auto& frame = stack.back();
            const auto& children = frame.node->next_functions();

            if (frame.child_idx < children.size()) {
                auto& child = children[frame.child_idx];
                frame.child_idx++;

                if (!child) continue;
                if (on_stack.count(child.get())) {
                    throw AutogradException("Cycle detected in computation graph");
                }
                if (visited.count(child.get())) continue;

                visited.insert(child.get());
                on_stack.insert(child.get());
                stack.push_back({child, 0});
            } else {
                // All children processed — post-order visit
                on_stack.erase(frame.node.get());
                sorted.push_back(frame.node);
                stack.pop_back();
            }
        }
    }

    // Seed gradient accumulators for root grad_fns. audit-6 BB.3: use the
    // deduped per-unique-root cumulative seed so duplicate roots contribute
    // their gradients exactly once (summed), not last-writer-wins.
    for (auto* key : root_order) {
        Variable* r = first_root_handle[key];
        if (r && r->grad_fn()) {
            accumulate_grad(r->grad_fn().get(), root_seeds[key]);
            if (create_graph) {
                accumulate_grad_var(r->grad_fn().get(),
                                    Variable(root_seeds[key], /*requires_grad=*/true));
            }
        }
    }

    auto cleanup_graph = [&]() {
        if (!retain_graph) {
            for (auto& func : sorted) {
                if (func) {
                    // Clear grad_fn on intermediate (non-leaf) input variables
                    // to break reference cycles that would otherwise leak memory.
                    for (auto& var : func->input_variables()) {
                        if (var.grad_fn() && !var.is_leaf()) {
                            var.set_grad_fn(nullptr);
                        }
                    }
                    func->set_input_variables({});
                    func->set_next_functions({});
                }
            }
            for (auto* root : roots) {
                if (root && root->grad_fn() && !root->is_leaf()) {
                    root->set_grad_fn(nullptr);
                }
            }
        }
    };

    // V.3: accum_guard_multi was moved up to immediately after the
    // std::move (above). Only the graph-cleanup guard is installed here.
    ScopeGuard cleanup_guard_multi{[&]{ clear_gradients(); cleanup_graph(); }};

    // HH.4: track leaves that actually received an accumulation; mirrors the
    // single-root path so leaves reached purely through next_funcs are also
    // downcast in the finalization loop below.
    std::unordered_set<VariableImpl*> touched_leaves;

    {
        // Execute backward in reverse topological order
        for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
            if (!*it) continue;
            auto& function = *it;

            std::vector<Tensor> grad_outputs;
            const auto& accum_grads = get_accumulated_grads(function.get());
            if (accum_grads.empty()) {
                continue;  // No gradient flows to this function
            }

            // Sum all accumulated gradients with dtype promotion.
            // Pre-compute target dtype to avoid repeated conversions.
            DType target_dtype = accum_grads[0].dtype();
            for (size_t i = 1; i < accum_grads.size(); ++i) {
                target_dtype = promote_types(target_dtype, accum_grads[i].dtype());
            }
            Tensor total_grad = (accum_grads[0].dtype() == target_dtype)
                ? accum_grads[0]
                : accum_grads[0].to(target_dtype);
            for (size_t i = 1; i < accum_grads.size(); ++i) {
                const Tensor& gi = accum_grads[i];
                total_grad = total_grad + (gi.dtype() == target_dtype ? gi : gi.to(target_dtype));
            }
            grad_outputs.push_back(total_grad);

            // Check incoming gradients for NaN/Inf before calling backward
            check_for_anomaly(grad_outputs, function.get());

            function->reload_saved_tensors();
            function->validate_saved_tensors();

            // V.2: dual-path block matching the single-root execute() —
            // honours `create_graph` by invoking `backward_with_variables`
            // and tracking higher-order stub disconnections. Previously
            // the multi-root path always called the raw-Tensor `backward`,
            // so torch.autograd.backward([roots], create_graph=True)
            // silently degraded to first-order-only.
            std::vector<Tensor> input_grads;
            std::vector<Variable> var_input_grads;
            {
                std::optional<BackwardTimer> _profiler_timer;
                if (AutogradProfiler::instance().is_enabled()) {
                    _profiler_timer.emplace(function->name());
                }

                if (create_graph) {
                    std::vector<Variable> var_grad_outputs =
                        build_var_grad_outputs(function.get(), grad_outputs);

                    var_input_grads = function->backward_with_variables(var_grad_outputs);

                    if (function->is_higher_order_stub()) {
                        auto mode = get_higher_order_grad_mode();
                        if (mode == HigherOrderGradMode::Error) {
                            throw std::runtime_error(
                                "create_graph=true but '" + function->name() +
                                "' is a higher-order stub (second derivatives are zero). "
                                "Use set_higher_order_grad_mode(Warn) to allow this.");
                        }
                        detail::increment_higher_order_disconnection_count();
                        TENZOR_LOG_WARN("[tenzor::autograd] '{}' is a higher-order stub — "
                                        "second derivatives through it will be zero. "
                                        "(disconnection #{})",
                                        function->name(),
                                        higher_order_disconnection_count());
                    }

                    input_grads.reserve(var_input_grads.size());
                    for (auto& vg : var_input_grads) {
                        input_grads.push_back(vg.tensor());
                    }
                } else {
                    input_grads = function->backward(grad_outputs);
                }
            }

            // Check for NaN/Inf in computed gradients when anomaly detection is on
            check_for_anomaly(input_grads, function.get());

            // Validate gradient dtypes are floating-point or complex
            for (size_t i = 0; i < input_grads.size(); ++i) {
                const auto& g = input_grads[i];
                if (g.is_valid() && g.numel() > 0 &&
                    !g.is_floating_point() && !g.is_complex()) {
                    throw std::runtime_error(
                        std::string("Backward function ") + function->name() +
                        " returned non-floating-point gradient (dtype=" +
                        std::string(dtype_name(g.dtype())) + " at index " +
                        std::to_string(i) +
                        "). Gradients must be floating-point or complex.");
                }
            }

            // HH.2: release_saved_tensors() was previously called HERE,
            // before hook execution and accumulation. That broke hooks that
            // capture saved tensors by reference (the single-root path
            // releases AFTER hooks+accumulate; multi-root must match).
            // Move below, after the accumulation loop completes.

            // Accumulate gradients to input variables
            auto& input_vars = function->input_variables();
            for (size_t i = 0; i < input_vars.size() && i < input_grads.size(); ++i) {
                Variable& var = input_vars[i];
                if (!var.requires_grad()) continue;

                Tensor grad_to_apply = input_grads[i];

                // HH.1: validate gradient shape matches variable shape via
                // the shared helper. Previously only the single-root path
                // checked; multi-root let mismatched shapes through and
                // crashed inside the accumulation or in a user hook.
                validate_grad_shape_or_throw(var, grad_to_apply, function->name(), i);

                // Capture the hooks at this wider scope so the create_graph path
                // below can apply the SAME transform to the graph-carrying gradient.
                std::map<size_t, std::function<Tensor(const Tensor&)>> applied_hooks;
                if (var.impl_) {
                    {
                        std::shared_lock lock(var.impl_->hooks_mutex_);
                        if (!var.impl_->hooks_.empty()) {
                            applied_hooks = var.impl_->hooks_;
                        }
                    }
                    for (auto& [id, hook] : applied_hooks) {
                        grad_to_apply = hook(grad_to_apply);
                    }
                }

                // PyTorch semantics: the hooked gradient must propagate to
                // downstream Functions (next_functions), not only to leaf
                // accumulation. Write it back before the leaf dtype promotion
                // so the propagation loop below forwards the transformed value.
                if (!applied_hooks.empty()) {
                    input_grads[i] = grad_to_apply;
                    // Mirror the hook transform onto the graph-carrying cotangent
                    // so the leaf grad_with_graph store and the next_functions
                    // propagation below both forward the hooked Variable
                    // (consistent with the raw input_grads[i] path).
                    if (create_graph && i < var_input_grads.size()) {
                        Tensor hv = var_input_grads[i].tensor();
                        for (auto& [id, hook] : applied_hooks) hv = hook(hv);
                        var_input_grads[i] =
                            Variable(hv, var_input_grads[i].requires_grad());
                    }
                }

                if (var.is_leaf() || var.retains_grad()) {
                    // V.4: accumulate at the promoted dtype of (existing,
                    // incoming) — never downcast incoming to a half-precision
                    // leaf dtype before the add.
                    auto accumulate_unlocked = [&]() {
                        if (var.impl_->grad_.has_value()) {
                            auto existing_grad = var.impl_->grad_.value();
                            DType target = promote_types(existing_grad.dtype(),
                                                         grad_to_apply.dtype());
                            if (existing_grad.dtype() != target) {
                                existing_grad = existing_grad.to(target);
                            }
                            if (grad_to_apply.dtype() != target) {
                                grad_to_apply = grad_to_apply.to(target);
                            }
                            var.impl_->grad_ = existing_grad + grad_to_apply;
                        } else {
                            var.impl_->grad_ = grad_to_apply;
                        }
                    };

                    if (var.impl_) {
                        // NN.4: snapshot the existing graph impl under the
                        // lock, drop the lock, run the Variable-level sum,
                        // then re-acquire to store.
                        std::shared_ptr<VariableImpl> existing_graph_impl;
                        bool need_graph_update = create_graph && i < var_input_grads.size();
                        {
#ifndef NDEBUG
                            GradMutexDebugGuard single_lock_check;
#endif
                            std::lock_guard lock(*var.impl_->grad_mutex_);
                            accumulate_unlocked();
                            // HH.4: record touched leaf so the finalization
                            // dtype-downcast loop visits it even when the leaf
                            // is reached only through next_funcs chains.
                            touched_leaves.insert(var.impl_.get());
                            if (need_graph_update) {
                                existing_graph_impl = var.impl_->grad_with_graph_impl_;
                            }
                        }

                        // V.2 / GG.8: mirror the single-root path — capture
                        // the Variable form of the gradient on the leaf when
                        // create_graph is set; sum at the Variable level so
                        // grad_fn chains through every contributing path.
                        if (need_graph_update) {
                            // Apply the SAME backward hooks to the graph-carrying
                            // gradient so grad_variable() (consumed by second-order
                            // diff: Hessian/HVP, WGAN-GP, MAML) reflects the hook,
                            // not just .grad(). Hooks are opaque Tensor->Tensor
                            // maps, so the second derivative does not flow THROUGH
                            // the hook (inherent to Tensor hooks), but the gradient
                            // VALUE is now consistent between .grad() and grad_variable().
                            // var_input_grads[i] was already hook-transformed
                            // above (in lockstep with input_grads[i]), so the
                            // graph store, leaf .grad(), and next_functions
                            // propagation all see the same hooked cotangent.
                            Variable graph_grad = var_input_grads[i];
                            // E-01 (lost-update race): the read-modify-write of
                            // grad_with_graph_impl_ is split across the unlock
                            // window (the Variable sum is computed OUTSIDE the
                            // lock to avoid a re-entrant grad_mutex_ lock). Two
                            // concurrent create_graph backwards could snapshot
                            // the same existing impl and the second store would
                            // clobber the first. Use a compare-and-retry loop:
                            // recompute the combine from a fresh snapshot and
                            // only commit if grad_with_graph_impl_ is still the
                            // value we based our sum on. The value-grad
                            // accumulation (accumulate_unlocked) already ran
                            // exactly once above; ONLY this graph-grad combine
                            // retries. In the single-threaded / non-contended
                            // path the CAS succeeds on the first iteration, so
                            // behavior is unchanged.
                            std::shared_ptr<VariableImpl> snapshot = existing_graph_impl;
                            while (true) {
                                std::shared_ptr<VariableImpl> new_impl;
                                if (!snapshot) {
                                    new_impl = graph_grad.impl_;
                                } else {
                                    Variable existing_var;
                                    existing_var.impl_ = snapshot;
                                    new_impl =
                                        (existing_var + graph_grad).impl_;
                                }
                                {
#ifndef NDEBUG
                                    GradMutexDebugGuard single_lock_check;
#endif
                                    std::lock_guard lock(*var.impl_->grad_mutex_);
                                    if (var.impl_->grad_with_graph_impl_ == snapshot) {
                                        // Unchanged since our snapshot → commit.
                                        var.impl_->grad_with_graph_impl_ = new_impl;
                                        var.impl_->grad_with_graph_cache_storage_.reset();
                                        break;
                                    }
                                    // Another thread updated it; reload the
                                    // current value and retry the combine so its
                                    // contribution is not lost.
                                    snapshot = var.impl_->grad_with_graph_impl_;
                                }
                            }
                        }
                    } else {
                        var.set_grad(grad_to_apply);
                    }
                }
            }

            const auto& next_funcs = function->next_functions();
            for (size_t i = 0; i < next_funcs.size() && i < input_grads.size(); ++i) {
                if (next_funcs[i]) {
                    accumulate_grad(next_funcs[i].get(), input_grads[i]);
                    // Under create_graph, also forward the graph-carrying
                    // gradient Variable so the downstream Function receives a
                    // connected cotangent (keeps the second-order graph intact
                    // across this edge). var_input_grads is index-aligned with
                    // input_grads (both derived from the same backward call).
                    if (create_graph && i < var_input_grads.size()) {
                        accumulate_grad_var(next_funcs[i].get(), var_input_grads[i]);
                    }
                }
            }

            // HH.2: release saved tensors AFTER hooks + accumulation, matching
            // the single-root execute() path. Hooks may capture saved tensors
            // via closures, so they must remain alive through the accumulation
            // loop above.
            if (!retain_graph) {
                function->release_saved_tensors();
                // audit-9 JJ.2: see single-root execute() at L630 — drop
                // saved_variables_ to release the second-order graph.
                function->clear_saved_variables();
                // audit-10 NN.6: release per-Function ad-hoc state.
                function->release_op_specific_state();
            }
        }
    }

    // Audit-6 AA.7: mirror the single-root downcast — restore PyTorch's
    // `leaf.grad.dtype == leaf.dtype` contract by casting any promoted
    // accumulator back to the leaf's declared dtype.
    // Audit-7 EE.1: opt-out via Variable::set_preserve_grad_dtype(true)
    // for AMP fp32-master-weights.
    //
    // HH.4: iterate touched_leaves so leaves reached only via next_funcs
    // accumulation (not present in any sorted function's input_variables())
    // are also downcast.
    for (auto* leaf_impl : touched_leaves) {
        if (!leaf_impl) continue;
        std::lock_guard lock(*leaf_impl->grad_mutex_);
        if (!leaf_impl->grad_.has_value()) continue;
        const auto leaf_dt = leaf_impl->data_.dtype();
        if (leaf_impl->grad_->dtype() != leaf_dt &&
            !leaf_impl->preserve_grad_dtype_.load(std::memory_order_acquire)) {
            leaf_impl->grad_ = leaf_impl->grad_->to(leaf_dt);
        }
    }
    for (auto* root : roots) {
        if (!root) continue;
        if (!root->requires_grad()) continue;
        if (!root->is_leaf() && !root->retains_grad()) continue;
        if (!root->impl_) continue;
        std::lock_guard lock(*root->impl_->grad_mutex_);
        if (!root->impl_->grad_.has_value()) continue;
        const auto root_dt = root->dtype();
        if (root->impl_->grad_->dtype() != root_dt &&
            !root->impl_->preserve_grad_dtype_.load(std::memory_order_acquire)) {
            root->impl_->grad_ = root->impl_->grad_->to(root_dt);
        }
    }
    // ScopeGuards handle cleanup in both normal and exception paths
}

// Thread-local engine -- each thread gets its own instance so concurrent
// backward passes don't corrupt the shared grad_accumulators_ map.
auto backward_engine() -> BackwardEngine& {
    static thread_local BackwardEngine engine;
    return engine;
}

} // namespace tenzor
