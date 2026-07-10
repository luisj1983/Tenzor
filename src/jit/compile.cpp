/**
 * @file compile.cpp
 * @brief Implementation of automatic graph capture and compilation
 */

#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/compiler.hpp"
#include "tenzor/backend/dispatch_interceptor.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/core/jit_hooks.hpp"
#include "tenzor/utils/log.hpp"

#ifdef TENZOR_HAS_MLIR_JIT
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#endif

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <unordered_map>

namespace tenzor {
namespace jit {

// ============================================================================
// MLIR backend cache (Pimpl)
// ============================================================================

namespace mlir_detail {

#ifdef TENZOR_HAS_MLIR_JIT
struct MlirInvokerCache {
    std::unordered_map<std::string,
                       std::unique_ptr<::tenzor::jit::mlir_jit::IreeInvoker>>
        invokers;
};
#else
struct MlirInvokerCache {};
#endif

}  // namespace mlir_detail

namespace {
// Strict mode is on if the per-call config requests it OR the global
// TENZOR_JIT_STRICT env var is set to a non-empty, non-"0" value. Centralized so
// every JIT fallback site — compile failure, graph break / empty trace, and
// replay failure, on both the nvrtc and mlir backends — applies the identical
// rule. A graph break that escaped this check would silently degrade to eager on
// one path while another path throws: an inconsistent strict contract across
// backends and between inference and training.
auto jit_strict_enabled(bool config_strict) -> bool {
    if (config_strict) return true;
    const char* s = std::getenv("TENZOR_JIT_STRICT");
    return s && *s && *s != '0';
}
}  // namespace

// ============================================================================
// CompiledFunction
// ============================================================================

CompiledFunction::CompiledFunction(FnType fn, CompileConfig config)
    : CompiledFunction(
          FnTypeN([fn = std::move(fn)](std::span<const Variable> inputs) {
              if (inputs.size() != 1) {
                  throw std::invalid_argument(
                      "CompiledFunction: this function was constructed from a "
                      "single-input callable but was invoked with " +
                      std::to_string(inputs.size()) + " inputs");
              }
              return fn(inputs[0]);
          }),
          std::move(config)) {}

CompiledFunction::CompiledFunction(FnTypeN fn, CompileConfig config)
    : fn_(std::move(fn)), config_(std::move(config)) {
    if (config_.mode != "default" && config_.mode != "reduce-overhead" && config_.mode != "max-autotune") {
        throw std::invalid_argument("Unknown compilation mode: " + config_.mode +
            ". Valid modes: \"default\", \"reduce-overhead\", \"max-autotune\"");
    }
    if (config_.backend != "nvrtc" && config_.backend != "mlir") {
        throw std::invalid_argument(
            "Unknown compile backend: " + config_.backend +
            R"_(. Valid backends: "nvrtc", "mlir")_");
    }
    if (config_.backend == "mlir") {
        // NOTE: the IREE target string is deliberately NOT rejected here. An
        // unknown/mis-cased target is validated at the compile boundary
        // (mj::compile_mlir throws a clear JitCompileError listing the valid
        // canonical names), and CompiledFunction treats an unmappable target
        // string uniformly with an unmappable DEVICE (OneAPI/MPS): under strict
        // mode the error rethrows; under non-strict it degrades to eager with a
        // WARN (see mlir_invoke's C1/C2 safety net). Throwing at construction
        // would break that uniformity — an OneAPI device (target="auto") degrades
        // gracefully, so a bogus target string must too — and is redundant with
        // the compile-boundary validation.
#ifndef TENZOR_HAS_MLIR_JIT
        throw std::runtime_error(
            "CompiledFunction: backend=\"mlir\" requested but Tenzor was "
            "built without TENZOR_USE_MLIR_JIT=ON");
#else
        mlir_cache_ = std::make_unique<mlir_detail::MlirInvokerCache>();
#endif
    }
}

CompiledFunction::CompiledFunction(FnTypeN fn,
                                   std::vector<std::shared_ptr<Variable>> params,
                                   CompileConfig config)
    : CompiledFunction(std::move(fn), std::move(config)) {
    parameters_ = std::move(params);
}

CompiledFunction::~CompiledFunction() = default;

auto CompiledFunction::set_parameters(
    std::vector<std::shared_ptr<Variable>> params) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    parameters_ = std::move(params);
    // The cached graphs' parameter-leaf tables were built for the previous
    // parameter set (or none); drop them so the next call re-traces with the
    // new parameters bound.
    cache_.clear();
    // The MLIR inference path freezes captured parameters as constants inside
    // each IREE invoker, so those must be dropped too or an mlir-backend call
    // would keep returning results computed with the old parameter values.
    if (mlir_cache_) {
        mlir_cache_->invokers.clear();
    }
    warmup_counts_.clear();  // JIT-F056: counters tracked the dropped modules
}

auto CompiledFunction::with_parameters(
    std::vector<std::shared_ptr<Variable>> params) -> CompiledFunction& {
    set_parameters(std::move(params));
    return *this;
}

auto CompiledFunction::any_parameter_requires_grad() const -> bool {
    for (const auto& p : parameters_) {
        if (p && p->requires_grad()) return true;
    }
    return false;
}

auto CompiledFunction::operator()(const Variable& input) -> Variable {
    const Variable arr[1] = {input};
    return (*this)(std::span<const Variable>(arr, 1));
}

auto CompiledFunction::operator()(std::span<const Variable> inputs) -> Variable {
    if (inputs.empty()) {
        throw std::invalid_argument(
            "CompiledFunction::operator(): expected at least one input "
            "Variable, got zero");
    }

    // Re-entrancy guard: if a trace is already active on this thread, this
    // compiled function is being invoked from *inside* another function that is
    // currently being traced. Starting our own trace here would clear() the
    // shared thread-local Tracer and silently drop the outer trace's ops. Run
    // eagerly instead so our ops are recorded by the outer trace's interceptor
    // and inlined into the outer graph (both nvrtc and mlir trace paths set the
    // tracer's tracing_ flag around fn_).
    if (Tracer::get_instance().is_tracing()) {
        return fn_(inputs);
    }

    // Training-through-JIT: when any input requires grad, execute the CAPTURED
    // graph differentiably (grad_invoke) so .backward() produces input/parameter
    // gradients that match eager autograd — instead of the old unconditional
    // eager short-circuit. The grad variant is compiled WITHOUT fusion and
    // replayed through the autograd-aware executor, so no backward-less fused
    // kernel is ever hit. grad_invoke replays through the autograd-aware
    // INTERPRETER (Graph::forward grad_mode) and never uses the IREE runtime, so
    // it is backend-agnostic: an MLIR-configured function trains through the same
    // compiled graph (incl. captured-parameter leaves) as an nvrtc-configured one
    // — only its INFERENCE forward uses IREE. This gives training parity across
    // backends. grad_invoke degrades to eager autograd on the same device if the
    // graph can't be built/replayed.
    bool any_requires_grad = false;
    for (const auto& v : inputs) {
        if (v.requires_grad()) { any_requires_grad = true; break; }
    }
    // Closure-captured training: the explicit inputs (the batch) need not
    // require grad, but the declared parameters do. Take the differentiable
    // path whenever EITHER an input or a captured parameter requires grad, so
    // .backward() reaches param->grad().
    if (!any_requires_grad) {
        std::lock_guard<std::mutex> lock(mutex_);
        any_requires_grad = any_parameter_requires_grad();
    }
    if (any_requires_grad) {
        return grad_invoke(inputs);
    }

    if (config_.backend == "mlir") {
        return mlir_invoke(inputs);
    }

    auto key = shape_key(inputs, /*grad_variant=*/false);

    // Fast path: look up the compiled module under the lock, copy the handle
    // out, then RELEASE the lock before executing. Previously mutex_ was held
    // across forward()/replay/capture, serializing every concurrent call on
    // this CompiledFunction (despite the "without lock" comment).
    std::shared_ptr<CompiledModule> compiled_module;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            compiled_module = it->second;
        }
    }
    if (compiled_module) {
            // reduce-overhead mode: capture and replay CUDA graphs. This path
            // mutates shared CUDA-graph state and warmup_counts_; serialize it.
            // CUDA-graph capture only applies to CUDA/ROCm; for CPU/Vulkan/OneAPI
            // fall through to normal compiled execution on that same backend
            // (running an empty CUDA graph would return stale results / crash).
            const bool graph_capable =
                !inputs.empty() &&
                (inputs[0].tensor().device().type == Device::Type::CUDA ||
                 inputs[0].tensor().device().type == Device::Type::ROCm);
            // reduce-overhead requested but the device can't capture a graph
            // (Vulkan/OneAPI/CPU): warn once so it isn't a silent no-op, matching
            // the MLIR path's behavior (JIT-035).
            if (config_.mode == "reduce-overhead" && !graph_capable) {
                static std::once_flag warned;
                std::call_once(warned, [] {
                    TENZOR_LOG_WARN(
                        "JIT reduce-overhead mode has no effect on this backend "
                        "(CUDA-graph capture is CUDA/ROCm only); running normally.");
                });
            }
            if (config_.mode == "reduce-overhead" && graph_capable) {
                std::lock_guard<std::mutex> lock(mutex_);
                std::vector<Tensor> input_tensors;
                input_tensors.reserve(inputs.size());
                for (const auto& v : inputs) input_tensors.push_back(v.tensor());

                if (compiled_module->has_cuda_graph()) {
                    // Replay the captured graph: it copies the fresh inputs into
                    // the captured device buffers, replays the recorded GPU work,
                    // and leaves the results in captured_outputs_. Return THOSE
                    // (via replay_cuda_graph_outputs) — re-running forward() here
                    // would discard the replayed result and defeat the capture.
                    if (compiled_module->replay_cuda_graph(input_tensors)) {
                        auto outs = compiled_module->replay_cuda_graph_outputs();
                        if (outs.empty()) {
                            throw std::runtime_error(
                                "CompiledFunction: CUDA graph replay produced no outputs");
                        }
                        return Variable(outs[0], /*requires_grad=*/false);
                    }
                    // Replay failed — fall back to a single eager forward.
                    if (inputs.size() == 1) {
                        return compiled_module->forward(inputs[0]);
                    }
                    std::vector<Variable> input_vars(inputs.begin(), inputs.end());
                    auto outs = compiled_module->forward(input_vars);
                    if (outs.empty()) {
                        throw std::runtime_error(
                            "CompiledFunction: compiled graph produced no outputs");
                    }
                    return outs[0];
                }
                int& warmup = warmup_counts_[key];
                ++warmup;
                if (warmup >= kReduceOverheadWarmupCalls) {
                    // Capture CUDA graph on this execution
                    if (inputs.size() == 1) {
                        auto result = compiled_module->forward(inputs[0]);
                        compiled_module->capture_cuda_graph(input_tensors);
                        return result;
                    }
                    std::vector<Variable> input_vars(inputs.begin(), inputs.end());
                    auto outs = compiled_module->forward(input_vars);
                    compiled_module->capture_cuda_graph(input_tensors);
                    if (outs.empty()) {
                        throw std::runtime_error(
                            "CompiledFunction: compiled graph produced no outputs");
                    }
                    return outs[0];
                }
            }

            // Cache hit: execute compiled graph. Prefer the single-input
            // `forward(Variable)` overload when arity is 1 — it carries
            // additional dtype/device-mismatch retrace machinery that the
            // vector overload does not.
            if (inputs.size() == 1) {
                return compiled_module->forward(inputs[0]);
            }
            std::vector<Variable> input_vars(inputs.begin(), inputs.end());
            auto outs = compiled_module->forward(input_vars);
            if (outs.empty()) {
                throw std::runtime_error(
                    "CompiledFunction: compiled graph produced no outputs");
            }
            return outs[0];
    }

    // Cache miss: trace, compile, and cache. Trace ONCE — fn_() runs a single
    // time under the interceptor and yields the eager result via the out-params
    // below. Running fn_() a second time here would apply a side-effecting
    // closure's effect twice on a cache miss (JIT-008).
    Variable result;
    bool fn_ran = false;

    // JIT-033: the nvrtc backend has no native codegen for Vulkan/OneAPI, so the
    // traced graph runs through the interpreter there — correct, but unaccelerated
    // and previously silent (unlike the MLIR path). Warn once so it is visible.
    if (config_.backend == "nvrtc" && !inputs.empty()) {
        const auto dt = inputs[0].tensor().device().type;
        if (dt == Device::Type::Vulkan || dt == Device::Type::OneAPI) {
            // Per-compiled-function (JIT-F038): each such function warns once, not
            // just the first one in the process.
            if (!warned_no_accel_.exchange(true)) {
                TENZOR_LOG_WARN(
                    "JIT nvrtc backend has no GPU codegen on Vulkan/OneAPI; the "
                    "traced graph runs via the interpreter (correct but not "
                    "accelerated). Use backend=\"mlir\" for acceleration.");
            }
        }
    }

    // Attempt compilation in the background. Two distinct "no compiled graph"
    // outcomes must both honour strict mode: (a) trace_and_compile THROWS on a
    // pipeline failure, and (b) it returns nullptr on a graph break / empty
    // trace. Case (b) is NOT an exception, so it must be handled explicitly —
    // otherwise a graph break silently degrades to eager on this nvrtc inference
    // path while the mlir path (mlir_invoke_impl) throws, an inconsistent strict
    // contract across backends.
    std::shared_ptr<CompiledModule> compiled;
    bool compile_threw = false;
    try {
        compiled = trace_and_compile(inputs, /*grad_mode=*/false,
                                     &result, &fn_ran);
    } catch (const std::exception& e) {
        // If fn_ itself threw (before it finished), that is a genuine closure
        // error — propagate it exactly as a plain eager call would, regardless
        // of strict mode. Only a COMPILE failure (fn_ already ran) is eligible
        // for the eager fallback, using the result captured above.
        if (!fn_ran) throw;
        // Compilation failed. The eager result is already correct, but we do not
        // want this to be silent: it would hide regressions in the JIT pipeline.
        // In non-strict mode log a WARN every time so callers can see exactly
        // which graphs fail.
        if (jit_strict_enabled(config_.strict)) {
            throw;
        }
        TENZOR_LOG_WARN("JIT compilation failed ({} backend, target={}): {}. "
                        "Falling back to eager execution.",
                        config_.backend, config_.target, e.what());
        compile_threw = true;
    }

    if (!compile_threw) {
        if (compiled) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (cache_.size() < static_cast<size_t>(config_.max_retraces)) {
                cache_[key] = compiled;
            }
        } else if (config_.fullgraph || jit_strict_enabled(config_.strict)) {
            // fullgraph=True must error on ANY graph break — including the
            // .item()/in-place breaks that now surface via the trace hooks
            // (JIT-F054), not only unmapped-op breaks caught by the interceptor.
            throw std::runtime_error(
                "JIT trace produced no compiled graph (graph break or empty "
                "trace) and fullgraph/strict mode is enabled (" + config_.backend +
                " backend, config.fullgraph / config.strict / TENZOR_JIT_STRICT)");
        } else {
            TENZOR_LOG_WARN("JIT trace produced no compiled graph ({} backend): "
                            "graph break or empty trace. Falling back to eager "
                            "execution.", config_.backend);
        }
    }

    return result;
}

auto CompiledFunction::num_cached() const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    // On the mlir backend inference specializations live in mlir_cache_, not in
    // cache_; count both so introspection is consistent across backends.
    return cache_.size() + (mlir_cache_ ? mlir_cache_->invokers.size() : 0);
}

auto CompiledFunction::num_grad_forwards() const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return grad_forward_count_;
}

auto CompiledFunction::grad_invoke(std::span<const Variable> inputs) -> Variable {
    // shape_key's grad-variant marker (|G1) puts the differentiable graph in a
    // DISTINCT cache entry from the inference (|G0) variant for the same
    // shape/dtype/device — one never serves the other. The explicit-input _g
    // flags are not enough here: training closes over parameters that require
    // grad while the batch inputs do not.
    auto key = shape_key(inputs, /*grad_variant=*/true);

    std::shared_ptr<CompiledModule> compiled_module;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            compiled_module = it->second;
        }
    }

    // Capture the single traced fn_ execution's (grad-enabled) result AND whether
    // fn_ ran during THIS call. Hoisted to function scope (not just the cache-miss
    // block) so the differentiable-REPLAY fallback further down can reuse this
    // result instead of running fn_ a SECOND time when this same call traced it —
    // a side-effecting closure would otherwise double-apply on a cache-miss whose
    // replay then fails (the JIT-008 hazard, grad replay path — JIT-071). On a
    // cache HIT fn_ never ran here, so fn_ran_this_call stays false and the eager
    // fallback correctly runs fn_ exactly once.
    Variable grad_result;
    bool fn_ran_this_call = false;

    // Cache miss: build the differentiable (un-fused) variant now. Unlike the
    // inference path we do NOT run an eager warmup first — we want THIS call to
    // go through the compiled graph so grads provably come from the replay, not
    // fn_. If tracing/compilation can't produce a graph (graph break, empty
    // trace), fall back to eager autograd on the same backend (correct grads).
    if (!compiled_module) {
        try {
            compiled_module = trace_and_compile(inputs, /*grad_mode=*/true,
                                                &grad_result, &fn_ran_this_call);
        } catch (const std::exception& e) {
            // fn_ itself threw (before finishing): a genuine closure error —
            // propagate it exactly as a plain eager call would.
            if (!fn_ran_this_call) throw;
            bool strict = config_.strict;
            if (!strict) {
                if (const char* s = std::getenv("TENZOR_JIT_STRICT"); s && *s && *s != '0') {
                    strict = true;
                }
            }
            if (strict) throw;
            TENZOR_LOG_WARN("JIT grad compile failed: {}. Falling back to eager "
                            "autograd on the input's backend.", e.what());
            return grad_result;  // fn_ already ran once (grad-enabled result)
        }
        if (!compiled_module) {
            // Graph break / empty trace (nullptr, NOT an exception). Honour
            // strict mode here too: otherwise training silently degrades to eager
            // autograd while the same config would throw on the mlir inference
            // path — an inconsistent strict contract between inference and
            // training. In non-strict mode WARN so the fallback is visible.
            if (config_.fullgraph || jit_strict_enabled(config_.strict)) {
                throw std::runtime_error(
                    "JIT grad trace produced no compiled graph (graph break or "
                    "empty trace) and fullgraph/strict mode is enabled");
            }
            TENZOR_LOG_WARN("JIT grad trace produced no compiled graph: graph "
                            "break or empty trace. Falling back to eager autograd "
                            "on the input's backend.");
            return grad_result;  // fn_ already ran once during the trace
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (cache_.size() < static_cast<size_t>(config_.max_retraces)) {
            cache_[key] = compiled_module;
        }
    }

    // Replay the captured graph differentiably. If an op in the graph isn't
    // wired for differentiable replay it throws (see Graph::execute_node); catch
    // it and fall back to eager autograd on the SAME backend so gradients are
    // still correct (requirement: dynamic/unsupported cases degrade, never crash
    // or silently sever grad).
    std::vector<Variable> input_vars(inputs.begin(), inputs.end());
    std::vector<Variable> outs;
    try {
        outs = compiled_module->forward_grad(input_vars);
    } catch (const std::exception& e) {
        bool strict = config_.strict;
        if (!strict) {
            if (const char* s = std::getenv("TENZOR_JIT_STRICT"); s && *s && *s != '0') {
                strict = true;
            }
        }
        if (strict) throw;
        TENZOR_LOG_WARN("JIT grad replay fell back to eager autograd: {}", e.what());
        // If THIS call traced fn_ (cache miss above), reuse that grad-enabled
        // result — re-running fn_ here would double-execute a side-effecting
        // closure (JIT-071). On a cache HIT fn_ has not run yet this call, so an
        // eager call is correct and runs it exactly once.
        if (fn_ran_this_call) return grad_result;
        return fn_(inputs);
    }
    if (outs.empty()) {
        throw std::runtime_error(
            "CompiledFunction: differentiable compiled graph produced no outputs");
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++grad_forward_count_;
    }
    return outs[0];
}
auto CompiledFunction::clear_cache() -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    // The MLIR/IREE inference specializations live in a separate cache whose
    // invokers bake trace-time weights/constants into the compiled artifact.
    // Clearing only cache_ would leave stale invokers serving results on the
    // mlir backend after the user forces a re-trace.
    if (mlir_cache_) {
        mlir_cache_->invokers.clear();
    }
    // The per-shape warmup counters tracked the modules just dropped; reset them
    // so a re-traced module re-warms before CUDA-graph capture (JIT-F056).
    warmup_counts_.clear();
}

auto CompiledFunction::shape_key(const Variable& input, bool grad_variant)
    -> std::string {
    const Variable arr[1] = {input};
    return shape_key(std::span<const Variable>(arr, 1), grad_variant);
}

auto CompiledFunction::shape_key(std::span<const Variable> inputs,
                                 bool grad_variant) -> std::string {
    std::ostringstream ss;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i > 0) ss << "|";
        auto shape = inputs[i].tensor().shape();
        for (size_t j = 0; j < shape.size(); ++j) {
            if (j > 0) ss << "x";
            ss << shape[j];
        }
        ss << "_" << static_cast<int>(inputs[i].tensor().dtype());
        // Encode the FULL device (type AND ordinal index), not just the type.
        // On a multi-GPU box a module captured on cuda:0 must not be reused for
        // a cuda:1 input under the same shape/dtype key — that returns a graph
        // whose NVRTC module / CUDA-graph capture / cached device pointers are
        // bound to GPU 0, causing an illegal cross-device access on GPU 1.
        ss << "_" << inputs[i].tensor().device().to_string();
        // Differentiability is part of the contract: a requires_grad input takes
        // the eager (differentiable) path, a non-grad input the compiled path.
        // Keep them in separate cache entries so one never serves the other.
        ss << (inputs[i].requires_grad() ? "_g1" : "_g0");
    }
    // Global grad-variant marker. The per-input `_g` flags above key on each
    // input's own requires_grad, which is NOT sufficient when the grad path is
    // driven by captured parameters while the explicit inputs do not require
    // grad (training: x is a plain batch, only the weights need grads). Without
    // this the differentiable graph and the inference graph would collide on the
    // same key. `grad_variant` reflects the path actually taken by the caller.
    ss << (grad_variant ? "|G1" : "|G0");
    return ss.str();
}

namespace {

// RAII installer for the two side-channel trace hooks that the DispatchInterceptor
// alone does NOT catch (JIT-F027):
//   * the graph-break hook — data-dependent leaf reads (Tensor::item()) notify
//     the tracer through this hook, not the dispatch stack;
//   * the in-place op hook — in-place kernels dispatch through dispatch_inplace,
//     which intentionally bypasses the DispatchInterceptorStack, so without this
//     hook their mutations are invisible to the trace and later reads replay the
//     PRE-mutation value (silently wrong numerics, no eager fallback).
// TracingGuard (jit.trace) installs both; the CompiledFunction compile paths
// previously installed only the interceptor. `on_break` runs in addition to
// tracer.record_graph_break so the caller flips its trace-local break flag and
// discards / falls back like an unmapped-op break.
struct ScopedTraceHooks {
    ScopedTraceHooks(Tracer& tracer, std::function<void()> on_break) {
        tenzor::detail::set_graph_break_hook(
            [&tracer, on_break = std::move(on_break)](const std::string& reason) {
                if (on_break) on_break();
                tracer.record_graph_break(reason);
            });
        tenzor::detail::set_inplace_op_hook(
            [&tracer](OpId op, Tensor& target, const Tensor* others,
                      std::size_t num_others, const OpAttributes& attrs,
                      const Tensor* pre_snapshot) {
                tracer.record_inplace(
                    op, target,
                    std::span<const Tensor>(others, num_others), attrs,
                    pre_snapshot);
            });
    }
    ~ScopedTraceHooks() {
        tenzor::detail::set_graph_break_hook(nullptr);
        tenzor::detail::set_inplace_op_hook(nullptr);
    }
    ScopedTraceHooks(const ScopedTraceHooks&) = delete;
    ScopedTraceHooks& operator=(const ScopedTraceHooks&) = delete;
};

}  // namespace

auto CompiledFunction::trace_and_compile(std::span<const Variable> inputs,
                                         bool grad_mode,
                                         Variable* out_result,
                                         bool* out_fn_ran)
    -> std::shared_ptr<CompiledModule> {
    // Trace-local graph-break flag. The Tracer is thread_local but this
    // CompiledFunction (and its had_graph_break_ member) is shared across
    // threads; keying the decision off a stack-local bool ensures concurrent
    // different-shape traces cannot clobber each other's flag and cache a graph
    // that silently froze an unrepresented op as a constant.
    bool graph_break = false;
    size_t traced_op_count = 0;

    auto& tracer = Tracer::get_instance();
    tracer.start_trace();

    // Install the in-place / graph-break side-channel hooks for the duration of
    // the trace so in-place mutations are recorded (not silently dropped) and a
    // data-dependent .item() read breaks the trace (JIT-F027).
    ScopedTraceHooks trace_hooks(tracer, [&graph_break] { graph_break = true; });

    // Declare the closure-captured trainable parameters (if any) BEFORE running
    // the traced forward, so end_trace() can classify each captured parameter as
    // a live parameter leaf instead of freezing it as a constant. start_trace()
    // above already cleared any parameters from a prior trace on this
    // thread-local tracer. Copy under the lock — parameters_ is mutable state.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!parameters_.empty()) {
            tracer.set_parameters(parameters_);
        }
    }

    // Push a tracing interceptor onto the dispatch stack
    auto interceptor = make_tracing_interceptor(tracer,
        [this, &traced_op_count, &graph_break](OpId op) {
            graph_break = true;
            if (config_.fullgraph) {
                throw std::runtime_error(
                    "tenzor.compile(fullgraph=True): graph break at OpId "
                    + std::to_string(static_cast<int>(op)));
            }
        });

    DispatchInterceptorStack::push(std::move(interceptor));

    // Execute the function (ops are recorded by the interceptor)
    Variable output;
    try {
        output = fn_(inputs);
    } catch (...) {
        DispatchInterceptorStack::pop();
        tracer.clear();
        throw;
    }

    DispatchInterceptorStack::pop();

    // fn_() has completed exactly once (under the interceptor). Publish its
    // result so the caller reuses it instead of running fn_() a second time
    // (JIT-008), and flag that the closure itself ran (so a later compile
    // failure is distinguishable from a closure error).
    if (out_fn_ran) *out_fn_ran = true;
    if (out_result) *out_result = output;

    // A graph break means an op that could not be represented in the IR ran
    // during the trace. Its output has no producing node, so any later consumer
    // resolves it to the *trace-time* tensor and end_trace() bakes that value in
    // as a frozen constant (set_constant) — silently ignoring the runtime input
    // for every subsequent replay. There is no way to distinguish such a frozen
    // intermediate from a legitimate parameter constant after the fact, so we
    // must NOT build/cache a compiled graph when a break occurred. Returning
    // nullptr leaves the cache empty; CompiledFunction::operator() already ran
    // fn_() eagerly for this (cache-miss) call and returns that correct result,
    // and future calls re-trace and fall back to eager again. This is the only
    // correct behavior short of true graph segmentation. (In fullgraph mode the
    // interceptor callback has already thrown before reaching here.)
    // Publish for the had_graph_break() introspection accessor (best-effort);
    // the decision below reads the race-free trace-local flag.
    had_graph_break_.store(graph_break);
    if (graph_break) {
        tracer.clear();
        return nullptr;
    }

    // Build graph from traced ops
    std::vector<Variable> input_vec(inputs.begin(), inputs.end());
    auto graph = tracer.end_trace(input_vec, {output});
    if (!graph || graph->num_nodes() == 0) {
        return nullptr;
    }

    // Optimize the graph.
    //
    // Differentiable variant (grad_mode): skip ALL fusion/layout/dtype passes.
    // The fused GPU nodes (softmax/layernorm/rmsnorm/gemm-epilogue/small-MLP/
    // reduction) have no backward, and DType downcasting to Float16 would change
    // the numerics vs eager. The un-fused graph is replayed through the
    // autograd-aware executor so .backward() matches eager exactly. Inference
    // (grad_mode == false) keeps the full fast fused path unchanged.
    if (grad_mode) {
        return [&] {
            auto compiled = std::make_shared<CompiledModule>();
            compiled->set_graph(graph);
            return compiled;
        }();
    }

    if (config_.enable_fusion) {
        Compiler compiler(true);
        compiler.optimize(*graph);
    }

    // Mode-specific optimizations
    if (config_.mode == "max-autotune") {
        // Layout optimization only. Do NOT force a Float16 downcast here:
        // max-autotune tunes kernel/layout selection and must not silently
        // change numerics vs eager or across backends. Precision reduction is
        // opt-in via explicit autocast / mixed precision, never implied by mode.
        Compiler autotune_compiler(false);  // no default passes
        autotune_compiler.add_pass(std::make_unique<LayoutOptimizationPass>());
        autotune_compiler.optimize(*graph, 1);
    }

    // Wrap in a CompiledModule
    auto compiled = std::make_shared<CompiledModule>();
    compiled->set_graph(graph);

    return compiled;
}

// ============================================================================
// MLIR backend invoke
// ============================================================================

namespace {

/// Build the Graph for a single-input function by running the tracer.
/// Mirrors `CompiledFunction::trace_and_compile` (the NVRTC path): pushes
/// a tracing interceptor onto the dispatch stack so every backend dispatch
/// is recorded as a TracedOp. Without the interceptor, the tracer sees no
/// ops and `end_trace` returns an empty graph.
auto trace_single_input_graph(CompiledFunction::FnTypeN& fn,
                              std::span<const Variable> inputs,
                              Variable* out_result = nullptr,
                              bool* out_fn_attempted = nullptr,
                              bool* out_fn_ok = nullptr,
                              bool* out_had_break = nullptr)
    -> std::shared_ptr<Graph> {
    auto& tracer = Tracer::get_instance();
    tracer.start_trace();
    bool had_break = false;
    // Install the in-place / graph-break side-channel hooks (JIT-F027) so
    // in-place mutations are recorded and .item() reads break the trace.
    ScopedTraceHooks trace_hooks(tracer, [&had_break] { had_break = true; });
    auto interceptor = make_tracing_interceptor(tracer, [&had_break](OpId /*op*/) {
        // A break means an unmappable op ran; its output would be frozen as a
        // constant in the built graph (see trace_and_compile for the full
        // rationale). Record it so we discard the graph below and let
        // mlir_invoke() degrade to eager for this invocation.
        had_break = true;
    });
    DispatchInterceptorStack::push(std::move(interceptor));

    // fn is about to run (exactly once, under the interceptor). Mark it
    // attempted BEFORE the call so a caller can tell "fn threw" apart from "fn
    // never ran" on the fallback path — the two demand opposite handling
    // (propagate vs. eager re-run) and must never be conflated (JIT-008-class
    // double-exec on the MLIR path).
    if (out_fn_attempted) *out_fn_attempted = true;
    Variable output;
    try {
        output = fn(inputs);
    } catch (...) {
        DispatchInterceptorStack::pop();
        tracer.clear();
        throw;
    }
    DispatchInterceptorStack::pop();
    // fn completed successfully. Publish its result + the ok flag so the caller
    // reuses it on any post-trace fallback (graph break, lower/compile failure)
    // instead of running fn a SECOND time.
    if (out_fn_ok) *out_fn_ok = true;
    if (out_result) *out_result = output;
    if (out_had_break) *out_had_break = had_break;  // JIT-F055 (publish for accessor)
    if (had_break) {
        // Returning nullptr routes mlir_invoke() into its "trace produced no
        // graph -> fall back to eager (or throw in strict mode)" path instead
        // of executing a graph with a baked-in frozen constant.
        tracer.clear();
        return nullptr;
    }
    std::vector<Variable> input_vec(inputs.begin(), inputs.end());
    return tracer.end_trace(input_vec, {output});
}

}  // namespace

// ============================================================================
// Debug / introspection APIs (Group F.1 — show_graph)
// ============================================================================

auto CompiledFunction::dump_graph(const Variable& input) -> std::string {
    const Variable arr[1] = {input};
    return dump_graph(std::span<const Variable>(arr, 1));
}

auto CompiledFunction::dump_graph(std::span<const Variable> inputs) -> std::string {
    auto graph = trace_single_input_graph(fn_, inputs);
    if (!graph || graph->num_nodes() == 0) {
        return "<empty graph>\n";
    }
    if (config_.enable_fusion) {
        Compiler compiler(true);
        compiler.optimize(*graph);
    }
    return graph->to_string();
}

#ifdef TENZOR_HAS_MLIR_JIT
namespace {

/// The native IREE HAL target for a device, or nullopt when the device has no
/// IREE target at all (C1). IREE ships no Level-Zero/SYCL HAL for Intel
/// OneAPI, and no Metal HAL for Apple MPS in this build — so a graph on those
/// devices cannot be compiled natively and must run eagerly on that same
/// backend. Everything with a HAL (CPU/CUDA/ROCm/Vulkan) maps 1:1.
auto auto_target_for_device(const ::tenzor::Device& dev)
    -> std::optional<std::string> {
    switch (dev.type) {
        case ::tenzor::Device::Type::CPU:    return std::string("llvm-cpu");
        case ::tenzor::Device::Type::CUDA:   return std::string("cuda");
        case ::tenzor::Device::Type::ROCm:   return std::string("rocm");
        case ::tenzor::Device::Type::Vulkan: return std::string("vulkan-spirv");
        default:                             return std::nullopt;
    }
}

/// Resolve the effective IREE target string. Explicit (non-"auto") targets are
/// normalized for documented aliases (H6: "vulkan" → "vulkan-spirv", which is
/// the name IREE actually requires); "auto" derives from the device. Only
/// reached for devices that have a native target — mlir_invoke() short-circuits
/// to eager for OneAPI/MPS before this is called.
auto resolve_target(const std::string& cfg_target,
                    const ::tenzor::Device& dev) -> std::string {
    if (cfg_target != "auto") {
        std::string resolved = cfg_target;
        if (cfg_target == "vulkan") resolved = "vulkan-spirv";
        else if (cfg_target == "hip") resolved = "rocm";
        // JIT-032: an explicit target whose device family differs from the input
        // tensor's runs the graph on a DIFFERENT backend and host-round-trips the
        // result back — a silent, easy-to-hit device migration. Warn once so it
        // is visible rather than silent.
        if (auto in_t = auto_target_for_device(dev); in_t && *in_t != resolved) {
            static std::once_flag warned;
            std::call_once(warned, [&] {
                TENZOR_LOG_WARN(
                    "JIT: explicit MLIR target '{}' does not match the input "
                    "tensor's device ({}); the graph runs on '{}' and the result "
                    "is copied back to the input device — verify this is intended.",
                    resolved, dev.to_string(), resolved);
            });
        }
        return resolved;
    }
    if (auto t = auto_target_for_device(dev)) return *t;
    throw std::runtime_error(
        "MLIR backend: no IREE target mapped for device '" + dev.to_string() +
        "' (expected eager fallback to have handled this).");
}

}  // namespace

// C1 + C2 wrapper: decide eager-vs-compile up front, then run the compile path
// under a strict-aware eager safety net. This mirrors the NVRTC path's
// try→eager degradation so an unmappable device (OneAPI/MPS), a lowering gap,
// an unsupported-target throw, or a driver-load error degrades to eager on the
// SAME backend instead of crashing — unless strict mode is on.
// Map the actual Vulkan device's vendor to an IREE `--iree-vulkan-target` arch
// that enables the F16 / 16-bit-storage SPIR-V capabilities. TENZOR_VULKAN_TARGET
// overrides. Returns "" if the vendor is unknown / no Vulkan backend, letting
// compile_mlir fall back to the build default (or a bare target).
static auto detect_vulkan_iree_target(const ::tenzor::Device& in_dev)
    -> std::string {
    if (const char* e = std::getenv("TENZOR_VULKAN_TARGET"); e && *e) {
        return e;
    }
    auto* be = ::tenzor::try_get_backend(::tenzor::Device::Type::Vulkan);
    if (be == nullptr) return {};
    int idx = (in_dev.type == ::tenzor::Device::Type::Vulkan && in_dev.index >= 0)
                  ? in_dev.index
                  : 0;
    try {
        auto info = be->get_device_info(idx);
        std::string v = info.vendor;
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (v.find("nvidia") != std::string::npos) return "ampere";
        if (v.find("amd") != std::string::npos ||
            v.find("radeon") != std::string::npos) return "rdna3";
        if (v.find("intel") != std::string::npos) return "arc";
    } catch (...) {
        // Best-effort: fall through to the build default.
    }
    return {};
}

auto CompiledFunction::mlir_invoke(std::span<const Variable> inputs) -> Variable {
    if (inputs.empty()) {
        throw std::invalid_argument(
            "CompiledFunction::mlir_invoke: expected at least one input");
    }

    const ::tenzor::Device dev = inputs[0].tensor().device();

    // C1: a device with no IREE HAL target (OneAPI/MPS) can never compile
    // natively. Under target="auto" this is a capability limit, not a
    // failure — run eagerly on that same backend (never route to CPU) and
    // warn once. (An explicit target on such a device is honoured below and,
    // if it fails, caught by the C2 net.)
    if (config_.target == "auto" && !auto_target_for_device(dev)) {
        // In strict mode a missing IREE target is a hard error, exactly like a
        // lowering gap on CUDA/ROCm — the documented strict contract is "JIT
        // coverage gaps throw loudly". Without this, identical strict code threw
        // on a CUDA lowering gap but silently ran eager on OneAPI/MPS.
        bool strict = config_.strict;
        if (!strict) {
            if (const char* s = std::getenv("TENZOR_JIT_STRICT");
                s && *s && *s != '0') {
                strict = true;
            }
        }
        if (strict) {
            throw std::runtime_error(
                "MLIR JIT (strict): device '" + dev.to_string() +
                "' has no IREE target backend (Intel OneAPI / Apple MPS have no "
                "IREE HAL); cannot compile. Use backend=\"nvrtc\" or a device "
                "with an IREE target (CPU/CUDA/ROCm/Vulkan) to enable JIT.");
        }
        // Per-compiled-function (JIT-F038): warn once for EACH function that runs
        // unaccelerated here, not once for the whole process.
        if (!warned_no_accel_.exchange(true)) {
            TENZOR_LOG_WARN(
                "MLIR JIT: device '{}' has no IREE target backend (Intel "
                "OneAPI / Apple MPS have no IREE HAL); executing eagerly on "
                "that backend. Use backend=\"nvrtc\" or a device with an IREE "
                "target (CPU/CUDA/ROCm/Vulkan) to enable JIT.",
                dev.to_string());
        }
        return fn_(inputs);
    }

    // C2: strict-aware eager fallback around the entire lower/compile/resolve/
    // run section. Thread three signals out of the impl so the fallback runs
    // fn_ AT MOST ONCE across the whole invocation (JIT-008-class double-exec):
    //   fn_ok        — fn ran to completion during the trace; traced_result is
    //                  valid. A later JIT stage (lower/compile/invoke) failed →
    //                  REUSE traced_result, never re-run fn_.
    //   fn_attempted — fn was entered but threw. Re-running would double-execute
    //                  its side effects and throw again → propagate the original.
    //   neither      — fn never ran (e.g. a cached invoker failed at runtime) →
    //                  eager fallback fn_(inputs) is safe and is the documented
    //                  C2 behaviour.
    Variable traced_result;
    bool fn_attempted = false;
    bool fn_ok = false;
    try {
        return mlir_invoke_impl(inputs, &traced_result, &fn_attempted, &fn_ok);
    } catch (const std::exception& e) {
        bool strict = config_.strict;
        if (!strict) {
            if (const char* s = std::getenv("TENZOR_JIT_STRICT");
                s && *s && *s != '0') {
                strict = true;
            }
        }
        if (strict) throw;
        if (fn_ok) {
            TENZOR_LOG_WARN(
                "MLIR JIT: compile/run failed (target={}): {}. Reusing the "
                "traced eager result (fn already ran).",
                config_.target, e.what());
            return traced_result;
        }
        if (fn_attempted) {
            // The user's function itself threw during the trace — surface that
            // error unchanged rather than silently re-running it.
            throw;
        }
        TENZOR_LOG_WARN(
            "MLIR JIT: compile/run failed (target={}): {}. Falling back to "
            "eager execution on the input's backend.",
            config_.target, e.what());
        return fn_(inputs);
    }
}

auto CompiledFunction::mlir_invoke_impl(std::span<const Variable> inputs,
                                        Variable* out_result,
                                        bool* out_fn_attempted,
                                        bool* out_fn_ok)
    -> Variable {
    namespace mj = ::tenzor::jit::mlir_jit;

    const auto key = shape_key(inputs);

    // Materialize input tensors once; reused on hit path and after compile.
    std::vector<::tenzor::Tensor> input_tensors;
    input_tensors.reserve(inputs.size());
    for (const auto& v : inputs) input_tensors.push_back(v.tensor());

    bool any_requires_grad = false;
    for (const auto& v : inputs) {
        if (v.requires_grad()) { any_requires_grad = true; break; }
    }

    // Cache-hit fast path.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = mlir_cache_->invokers.find(key);
        if (it != mlir_cache_->invokers.end()) {
            mj::internal::record_cache_hit();
            auto outs = it->second->invoke(input_tensors);
            if (outs.size() != 1) {
                throw std::runtime_error(
                    "MLIR backend: expected exactly 1 output tensor from "
                    "@main, got " +
                    std::to_string(outs.size()));
            }
            // Place the host-staged output on the input's device (see the
            // cache-miss path below for rationale).
            ::tenzor::Tensor out0 = std::move(outs[0]);
            const auto out_dev = inputs[0].tensor().device();
            if (out0.device() != out_dev) out0 = out0.to(out_dev);
            return Variable(std::move(out0), any_requires_grad);
        }
        // Miss. If we've already compiled something else for a different
        // shape, this counts as a retrace (a new shape forces a fresh trace).
        mj::internal::record_cache_miss();
        if (!mlir_cache_->invokers.empty()) {
            mj::internal::record_retrace();
        }
    }

    // Cache miss: trace → lower → compile → load invoker → cache. Thread the
    // out-params so the trace runs fn EXACTLY ONCE; every fallback below reuses
    // that result rather than re-invoking fn_ (JIT-008-class double-exec).
    bool had_break = false;
    auto graph = trace_single_input_graph(fn_, inputs, out_result,
                                          out_fn_attempted, out_fn_ok, &had_break);
    had_graph_break_.store(had_break);  // JIT-F055: publish for the introspection accessor
    if (!graph || graph->num_nodes() == 0) {
        // Trace produced nothing (graph break in fullgraph mode, or the
        // function only ran ops the interceptor doesn't capture). Either
        // surface a hard error (strict) or log + degrade to eager.
        bool strict = config_.strict;
        if (!strict) {
            if (const char* s = std::getenv("TENZOR_JIT_STRICT"); s && *s && *s != '0') {
                strict = true;
            }
        }
        // fullgraph=True must also error on a graph break here (JIT-F054), not
        // only in strict mode.
        if (strict || config_.fullgraph) {
            throw std::runtime_error(
                "CompiledFunction::mlir_invoke: trace produced no graph "
                "(graph break; config.fullgraph / TENZOR_JIT_STRICT).");
        }
        // fn already ran (successfully) during the trace above — reuse its
        // result instead of running it a SECOND time.
        TENZOR_LOG_WARN("MLIR JIT: trace produced no graph; reusing the traced "
                        "eager result for this invocation.");
        if (out_result && out_fn_ok && *out_fn_ok) return *out_result;
        return fn_(inputs);
    }

    if (config_.enable_fusion) {
        Compiler compiler(true);
        compiler.optimize(*graph);
    }

    // M4: honour config_.mode on the MLIR path (previously a silent no-op —
    // only enable_fusion ran). max-autotune applies the same Layout/DType
    // passes the NVRTC path uses; reduce-overhead has no CUDA-graph capture on
    // the IREE runtime here, so warn once rather than silently ignoring it.
    if (config_.mode == "max-autotune") {
        // F014: LayoutOptimizationPass inserts OpType::LayoutConvert nodes with a
        // memory_format tag intended for the cuDNN/MIOpen NATIVE kernels. The
        // StableHLO lowerer has no LayoutConvert case — it would throw, forcing a
        // silent eager fallback (non-strict) or a hard compile error (strict) on
        // an otherwise valid conv graph — and IREE already performs its own
        // layout/tiling optimization during codegen. Running the pass here is
        // therefore both redundant and harmful, so the MLIR path skips it and
        // lets IREE handle layout. (No forced Float16 downcast either: max-autotune
        // must not silently change numerics vs eager or across IREE targets.)
    } else if (config_.mode == "reduce-overhead") {
        static std::once_flag warned_reduce_overhead;
        std::call_once(warned_reduce_overhead, [] {
            TENZOR_LOG_WARN(
                "MLIR JIT: mode=\"reduce-overhead\" has no CUDA-graph capture "
                "on the IREE runtime path; executing without it (results are "
                "correct, latency is unchanged). Use backend=\"nvrtc\" for "
                "CUDA-graph replay.");
        });
    }

    mj::GraphToMLIR lowerer;
    // Always lower the 4 dialect ops (flash_attention/gqa/rope/rms_norm) to pure
    // StableHLO (the "expand" form) rather than `call @tenzor_plugin.<op>`. The
    // plugin custom-calls resolve via a VM native module whose marshalling always
    // materializes CPU tensors (buffer_view_to_tensor), so they ran their CPU
    // kernels even on a GPU target — a hidden host round-trip AND a numeric
    // divergence from the StableHLO expansion used on GPU-HAL-less runtimes.
    // Expanding uniformly makes these ops execute on the compiled HAL target on
    // EVERY backend, with identical numerics regardless of how libIREERuntime was
    // built. (Graphs without these ops are unaffected — the expansion is a no-op
    // for them.)
    lowerer.set_plugin_enabled(false);
    const std::string mlir_text = lowerer.lower(*graph);

    const ::tenzor::Device in_dev = inputs[0].tensor().device();

    mj::CompileOptions opts;
    opts.target          = resolve_target(config_.target, in_dev);
    opts.plugin_enabled  = false;  // H2: always the device-correct expand form
    // M1: derive the ROCm GPU ISA from the actual device instead of a
    // build-time constant. Ordinal is the input's Device::index when the
    // tensor lives on a ROCm device; otherwise (e.g. a CPU-staged input with
    // an explicit rocm target) default to the first ROCm device. An empty
    // result makes compile_mlir fall back to the build default arch.
    if (opts.target == "rocm") {
        const int rocm_ordinal =
            (in_dev.type == ::tenzor::Device::Type::ROCm) ? in_dev.index : 0;
        opts.rocm_arch = mj::detect_rocm_gfx_arch(rocm_ordinal);
    }
    // F032: derive an IREE Vulkan target from the actual device's vendor so the
    // emitted SPIR-V environment enables shaderFloat16 / 16-bit storage. Without
    // it, IREE's conservative default SPIR-V env lacks F16, so F16/BF16 graphs
    // silently produce garbage/NaN on the GPU (F32 works). TENZOR_VULKAN_TARGET
    // overrides. Empty result => compile_mlir falls back to the build default.
    if (opts.target == "vulkan-spirv" || opts.target == "vulkan") {
        opts.vulkan_arch = detect_vulkan_iree_target(in_dev);
    }
    // F004: derive the CUDA arch (sm_XX) from the actual device's compute
    // capability instead of only env/build-default/sm_80, so device-specific
    // codegen and sm_90+-only ops are available (matching the device-derived
    // rocm/vulkan arch). Best-effort: any failure (no CUDA runtime) leaves
    // cuda_arch empty and compile_mlir falls back to TENZOR_CUDA_TARGET / build
    // default / sm_80.
    if (opts.target == "cuda") {
        try {
            const ::tenzor::Device cuda_dev =
                (in_dev.type == ::tenzor::Device::Type::CUDA)
                    ? in_dev
                    : ::tenzor::Device::cuda(0);
            const auto props = ::tenzor::get_device_properties(cuda_dev);
            if (props.major_version > 0) {
                opts.cuda_arch = "sm_" +
                    std::to_string(props.major_version) +
                    std::to_string(props.minor_version);
            }
        } catch (...) {
            opts.cuda_arch.clear();
        }
    }

    auto artifact = mj::compile_mlir(mlir_text, opts);
    // Mode::InProcess registers the tenzor_plugin VM module before loading
    // the bytecode — required whenever the compiled .vmfb references any
    // tenzor_plugin.<op> import. For targets where the linked runtime lacks
    // the HAL driver (e.g. distributions without cuda/vulkan compiled in)
    // we fall back to the subprocess driver, which carries its own driver
    // set; this preserves the GPU end-to-end path without a CPU fallback.
    // M3: thread the input's device ordinal so the IREE HAL device matches the
    // requested ordinal (cuda:1 runs on GPU 1, not the driver default).
    const int hal_ordinal = in_dev.index < 0 ? 0 : in_dev.index;
    // Test/diagnostic hook: force the subprocess path (iree-run-module) even when
    // the in-process HAL driver is linked, so the subprocess I/O path can be
    // exercised on a CPU-only build. Does not affect default behavior.
    bool force_subprocess = false;
    if (const char* s = std::getenv("TENZOR_MLIR_FORCE_SUBPROCESS");
        s && *s && *s != '0') {
        force_subprocess = true;
    }
    std::unique_ptr<mj::IreeInvoker> invoker;
    try {
        invoker = mj::IreeInvoker::load(
            artifact,
            force_subprocess ? mj::IreeInvoker::Mode::Subprocess
                             : mj::IreeInvoker::Mode::InProcess,
            hal_ordinal);
    } catch (const mj::JitInvokeError& e) {
        const std::string what = e.what();
        // A missing/unregistered in-process HAL driver can surface under several
        // IREE status codes, not just NOT_FOUND (JIT-036). Anchor on "driver" and
        // accept the driver-unavailable statuses so the subprocess fallback isn't
        // skipped just because the wording differs (which would drop us to eager).
        const bool driver_mentioned = what.find("driver") != std::string::npos;
        const bool unavailable =
            what.find("NOT_FOUND")   != std::string::npos ||
            what.find("UNAVAILABLE") != std::string::npos ||
            what.find("not registered") != std::string::npos ||
            what.find("no driver")   != std::string::npos;
        if (driver_mentioned && unavailable) {
            // The in-process HAL driver for this target isn't linked, so we must
            // run via the iree-run-module subprocess. That subprocess canNOT
            // register the tenzor_plugin VM native module, so any
            // `call @tenzor_plugin.<op>` import in the plugin-form vmfb would
            // fail to load (this made flash_attention/rope/rms_norm/gqa
            // effectively CPU-only). Re-lower WITHOUT the plugin — expanding
            // those custom ops to pure StableHLO — and recompile so the
            // subprocess vmfb is fully self-contained. (Plugin-free graphs are
            // unaffected: the expanded form is identical for them.)
            mj::GraphToMLIR sh_lowerer;
            sh_lowerer.set_plugin_enabled(false);
            const std::string sh_text = sh_lowerer.lower(*graph);
            mj::CompileOptions sh_opts = opts;
            sh_opts.plugin_enabled = false;
            auto sh_artifact = mj::compile_mlir(sh_text, sh_opts);
            invoker = mj::IreeInvoker::load(sh_artifact,
                                            mj::IreeInvoker::Mode::Subprocess,
                                            hal_ordinal);
        } else {
            throw;
        }
    }

    // Optional artifact dumping: when TENZOR_JIT_DUMP=<dir> is set, write the
    // full pipeline (graph text + MLIR + expanded StableHLO + iree-compile
    // pipeline dump + path to the vmfb) into a hash-keyed subdirectory.
    if (const char* dump_dir = std::getenv("TENZOR_JIT_DUMP"); dump_dir && *dump_dir) {
        try {
            namespace fs = std::filesystem;
            const std::string key = mj::compute_cache_key(mlir_text, opts.target);
            const fs::path subdir = fs::path(dump_dir) / key;
            fs::create_directories(subdir);

            {
                std::ofstream f(subdir / "graph.txt", std::ios::trunc);
                f << graph->to_string();
            }
            {
                std::ofstream f(subdir / "mlir.txt",
                                std::ios::binary | std::ios::trunc);
                f.write(mlir_text.data(),
                        static_cast<std::streamsize>(mlir_text.size()));
            }
            {
                // Re-lower with plugin_enabled=false to dump the deploy form.
                mj::GraphToMLIR sh_lowerer;
                sh_lowerer.set_plugin_enabled(false);
                const std::string sh_text = sh_lowerer.lower(*graph);
                std::ofstream f(subdir / "stablehlo.txt",
                                std::ios::binary | std::ios::trunc);
                f.write(sh_text.data(),
                        static_cast<std::streamsize>(sh_text.size()));
            }
            {
                // Best-effort iree-compile pipeline dump. Shares the same
                // logic as dump_iree() but inlined so a single compile
                // produces the complete dump set without an extra trace.
                // Use the same discovery chain compile_mlir() uses so the
                // dump always lines up with what actually ran.
                const std::string& iree_compile =
                    ::tenzor::jit::mlir_jit::resolve_iree_compile();
                const fs::path tmp_in =
                    fs::temp_directory_path() /
                    ("tz_jit_dump_iree_" +
                     std::to_string(::getpid()) + "_" + key.substr(0, 8) +
                     ".mlir");
                {
                    std::ofstream f(tmp_in,
                                    std::ios::binary | std::ios::trunc);
                    f.write(mlir_text.data(),
                            static_cast<std::streamsize>(mlir_text.size()));
                }
                // Run iree-compile directly (no shell) capturing combined
                // stdout+stderr; the pipeline IR is printed to stderr.
                // --iree-input-type=auto picks up StableHLO on both old
                // (3.0–3.10) and new (3.11+) IREE. Older accepted =stablehlo
                // but 3.11+ rejects it; "auto" is the safe common value.
                const std::string captured =
                    ::tenzor::jit::mlir_jit::exec_capture(
                        iree_compile,
                        {"--iree-input-type=auto",
                         "--iree-hal-target-backends=" + opts.target,
                         "--mlir-disable-threading",
                         "--mlir-print-ir-after-all",
                         "--compile-to=vm",
                         "-o", "/dev/null",
                         tmp_in.string()},
                        /*capture_stderr=*/true);
                std::ofstream f(subdir / "iree.log",
                                std::ios::binary | std::ios::trunc);
                f.write(captured.data(),
                        static_cast<std::streamsize>(captured.size()));
                std::error_code _ec;
                fs::remove(tmp_in, _ec);
            }
            {
                // Record where the cached vmfb lives so the dump dir is
                // self-describing without copying the bytecode itself.
                std::ofstream f(subdir / "vmfb.path", std::ios::trunc);
                f << artifact.vmfb_path.string() << "\n";
            }
        } catch (const std::exception&) {
            // Dumping is best-effort: never fail compilation because of it.
        }
    }

    // F003: tell the invoker how many results @main returns so the subprocess
    // path writes one --output=@file per result and returns ALL of them (matching
    // the in-process path), rather than only the first. Set before the invoker is
    // cached below so the cached instance keeps the value.
    invoker->set_expected_outputs(static_cast<int>(graph->outputs().size()));
    auto outs = invoker->invoke(input_tensors);
    if (outs.size() != 1) {
        throw std::runtime_error(
            "MLIR backend: expected exactly 1 output tensor from @main, got " +
            std::to_string(outs.size()));
    }
    // IREE marshalling host-stages I/O, so the output comes back on CPU
    // regardless of input device. Place it on the input's device (type AND
    // ordinal) so the JIT output's placement matches eager — a GPU model yields
    // a GPU output on the same GPU, not silently on CPU / device 0.
    ::tenzor::Tensor out0 = std::move(outs[0]);
    const auto out_dev = inputs[0].tensor().device();
    if (out0.device() != out_dev) out0 = out0.to(out_dev);
    Variable result(std::move(out0), any_requires_grad);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (mlir_cache_->invokers.size() <
            static_cast<size_t>(config_.max_retraces)) {
            mlir_cache_->invokers.emplace(key, std::move(invoker));
        } else {
            // Hit the per-function cache ceiling — count this insertion
            // attempt as an eviction (the new shape's invoker is dropped on
            // the floor, so the next call for it will re-trace).
            mj::internal::record_eviction();
        }
    }
    return result;
}

// ============================================================================
// Debug / introspection APIs (Group F.2 / F.3 — show_mlir / show_stablehlo)
// ============================================================================

auto CompiledFunction::dump_mlir(const Variable& input) -> std::string {
    const Variable arr[1] = {input};
    return dump_mlir(std::span<const Variable>(arr, 1));
}

auto CompiledFunction::dump_mlir(std::span<const Variable> inputs) -> std::string {
    namespace mj = ::tenzor::jit::mlir_jit;
    auto graph = trace_single_input_graph(fn_, inputs);
    if (!graph || graph->num_nodes() == 0) {
        return "<empty graph: nothing to lower>\n";
    }
    if (config_.enable_fusion) {
        Compiler compiler(true);
        compiler.optimize(*graph);
    }
    mj::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(true);
    return lowerer.lower(*graph);
}

auto CompiledFunction::dump_stablehlo(const Variable& input) -> std::string {
    const Variable arr[1] = {input};
    return dump_stablehlo(std::span<const Variable>(arr, 1));
}

auto CompiledFunction::dump_stablehlo(std::span<const Variable> inputs) -> std::string {
    namespace mj = ::tenzor::jit::mlir_jit;
    auto graph = trace_single_input_graph(fn_, inputs);
    if (!graph || graph->num_nodes() == 0) {
        return "<empty graph: nothing to lower>\n";
    }
    if (config_.enable_fusion) {
        Compiler compiler(true);
        compiler.optimize(*graph);
    }
    mj::GraphToMLIR lowerer;
    // The defining difference vs. dump_mlir: expand all `tenzor.*` custom_call
    // ops into pure StableHLO primitives. The text returned here is the form
    // shipped to deploy targets that do not link the Tenzor runtime plugin.
    lowerer.set_plugin_enabled(false);
    return lowerer.lower(*graph);
}

auto CompiledFunction::dump_iree(const Variable& input) -> std::string {
    const Variable arr[1] = {input};
    return dump_iree(std::span<const Variable>(arr, 1));
}

auto CompiledFunction::dump_iree(std::span<const Variable> inputs) -> std::string {
    if (inputs.empty()) {
        throw std::invalid_argument(
            "CompiledFunction::dump_iree: expected at least one input");
    }
    const std::string mlir_text = dump_mlir(inputs);
    if (mlir_text.rfind("<empty", 0) == 0) {
        return mlir_text;
    }

    // Resolve the IREE target the same way mlir_invoke does so the dump
    // exactly mirrors what the runtime compiles.
    const std::string target =
        resolve_target(config_.target, inputs[0].tensor().device());

    // Write the MLIR text to a temp file and shell out to iree-compile with
    // --compile-to=vm (the fast pipeline that produces no .vmfb, just runs
    // up to the VM stage) plus --mlir-print-ir-after-all. The full pipeline
    // IR is printed to stderr; we capture both streams via 2>&1.
    namespace fs = std::filesystem;
    const fs::path tmp_in =
        fs::temp_directory_path() /
        ("tz_jit_dump_iree_" + std::to_string(::getpid()) + ".mlir");
    {
        std::ofstream f(tmp_in, std::ios::binary | std::ios::trunc);
        f.write(mlir_text.data(),
                static_cast<std::streamsize>(mlir_text.size()));
    }

    // Use the same discovery chain compile_mlir() uses (env override → pip
    // venv → CMake-found dist → $PATH). show_iree must dump the IR for the
    // exact compiler that actually ran during compile_mlir(), so any
    // divergence here would be a bug.
    const std::string& iree_compile =
        ::tenzor::jit::mlir_jit::resolve_iree_compile();

    // Run iree-compile directly (no shell) capturing combined stdout+stderr;
    // the pipeline IR is printed to stderr. --iree-input-type=auto picks up
    // StableHLO on both old and new IREE (3.0–3.10 also accepted =stablehlo;
    // 3.11+ replaced it).
    std::string captured =
        ::tenzor::jit::mlir_jit::exec_capture(
            iree_compile,
            {"--iree-input-type=auto",
             "--iree-hal-target-backends=" + target,
             "--mlir-disable-threading",
             "--mlir-print-ir-after-all",
             "--compile-to=vm",
             "-o", "/dev/null",
             tmp_in.string()},
            /*capture_stderr=*/true);
    if (captured.empty()) {
        captured = "<failed to spawn iree-compile>\n";
    }
    std::error_code _ec;
    fs::remove(tmp_in, _ec);
    return captured;
}

#else  // TENZOR_HAS_MLIR_JIT

auto CompiledFunction::mlir_invoke(std::span<const Variable> /*inputs*/) -> Variable {
    throw std::runtime_error(
        "CompiledFunction::mlir_invoke: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_mlir(const Variable& /*input*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_mlir: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_mlir(std::span<const Variable> /*inputs*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_mlir: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_stablehlo(const Variable& /*input*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_stablehlo: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_stablehlo(std::span<const Variable> /*inputs*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_stablehlo: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_iree(const Variable& /*input*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_iree: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_iree(std::span<const Variable> /*inputs*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_iree: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

#endif  // TENZOR_HAS_MLIR_JIT

// ============================================================================
// Free function
// ============================================================================

auto compile(CompiledFunction::FnType fn, CompileConfig config) -> CompiledFunction {
    return CompiledFunction(std::move(fn), std::move(config));
}

auto compile(CompiledFunction::FnTypeN fn, CompileConfig config) -> CompiledFunction {
    return CompiledFunction(std::move(fn), std::move(config));
}

auto compile(CompiledFunction::FnTypeN fn,
             std::vector<std::shared_ptr<Variable>> params,
             CompileConfig config) -> CompiledFunction {
    return CompiledFunction(std::move(fn), std::move(params), std::move(config));
}

} // namespace jit
} // namespace tenzor
