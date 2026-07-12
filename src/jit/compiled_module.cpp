/**
 * @file compiled_module.cpp
 * @brief Implementation of CompiledModule and convenience functions
 *
 * Provides the high-level interface for traced JIT modules, including
 * tracing, execution, optimization, serialization, and metadata management.
 */

#include "../../include/tenzor/jit/compiler.hpp"
#include "../../include/tenzor/jit/tracer.hpp"
#include "../../include/tenzor/jit/serialization.hpp"
#include "../../include/tenzor/core/device.hpp"
#include "../../include/tenzor/backend/loader.hpp"   // backend_registry()/get_backend()
#include "../../include/tenzor/backend/backend.hpp"  // Backend::copy / CopyKind
#include <algorithm>
#include <cstdio>
#include <stdexcept>

#ifdef TENZOR_HAS_ROCM
#include "../backends/rocm/hip_graph.hpp"
#include "../backends/rocm/rocm_error.hpp"
#include <hip/hip_runtime.h>
#endif

namespace tenzor {
namespace jit {

// ============================================================================
// CompiledModule Implementation
// ============================================================================

CompiledModule::~CompiledModule() {
    invalidate_cuda_graph();
}

CompiledModule::CompiledModule(std::shared_ptr<Graph> graph)
    : graph_(std::move(graph)) {}

auto CompiledModule::trace(std::shared_ptr<nn::Module> module,
                            const Variable& example_input) -> std::shared_ptr<CompiledModule> {
    if (!module) {
        throw std::runtime_error("Cannot trace null module");
    }

    // Use the existing jit::trace function to build the graph
    auto graph = jit::trace(module, example_input);

    auto compiled = std::make_shared<CompiledModule>(graph);
    // Record the device / dtype the graph was traced with so forward() can
    // trigger a retrace when a mismatched input is supplied later.
    compiled->traced_device_ = example_input.tensor().device();
    compiled->traced_dtype_  = example_input.tensor().dtype();
    compiled->traced_shape_key_ = compiled->compute_shape_key(example_input);
    // Retain the source module so the device/dtype-mismatch and shape-guard
    // retrace paths in forward() are actually reachable for traced modules
    // (they are all guarded by `source_module_ != nullptr`). Without this the
    // retrace is dead code and a cross-device call throws at dispatch instead.
    compiled->set_source_module(module);
    return compiled;
}

auto CompiledModule::trace(std::shared_ptr<nn::Module> module,
                            const Tensor& example_input) -> std::shared_ptr<CompiledModule> {
    return trace(module, Variable(example_input, false));
}

auto CompiledModule::compute_shape_key(const Variable& input) -> std::string {
    std::string key;
    auto shape = input.tensor().shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) key += ',';
        key += std::to_string(shape[i]);
    }
    // Include device+dtype so retracing distinguishes between runs on different
    // backends / precisions (e.g. CPU Float32 vs CUDA Float64 with identical
    // shapes). Without this, a compiled graph traced on CPU Float32 silently
    // replays on CUDA tensors and hits device-mismatch errors at dispatch time.
    key += '@';
    key += input.tensor().device().to_string();
    key += ':';
    key += std::to_string(static_cast<int>(input.tensor().dtype()));
    return key;
}

auto CompiledModule::compute_shape_key(const std::vector<Variable>& inputs) -> std::string {
    std::string key;
    for (size_t j = 0; j < inputs.size(); ++j) {
        if (j > 0) key += '|';
        auto shape = inputs[j].tensor().shape();
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) key += ',';
            key += std::to_string(shape[i]);
        }
        key += '@';
        key += inputs[j].tensor().device().to_string();
        key += ':';
        key += std::to_string(static_cast<int>(inputs[j].tensor().dtype()));
    }
    return key;
}

auto CompiledModule::throw_if_loaded_shape_mismatch(
    const std::vector<std::vector<int64_t>>& call_shapes,
    const std::vector<int>& call_dtypes,
    const std::vector<Device>& call_devices) const -> void {
    // Only loaded modules that cannot retrace are guarded. If dynamic dims were
    // configured (via mark_dynamic_dims after load) or a retrace path exists, the
    // module can adapt and this guard does not apply.
    if (!loaded_ || source_module_ || retrace_fn_ || !dynamic_dims_.empty()) {
        return;
    }
    if (call_shapes == loaded_input_shapes_ &&
        call_dtypes == loaded_input_dtypes_ &&
        call_devices == loaded_input_devices_) {
        return;
    }
    // A loaded module bakes its constants (weights, folded biases) at the
    // trace-time DEVICE and DTYPE; the loaded path never consults
    // compute_shape_key's device/dtype components, so a same-shape call from a
    // different device or dtype would replay e.g. CUDA-resident/Float32
    // constants against a CPU/Float64 input — a confusing downstream dispatch
    // error, or (for an unfused graph, whose nodes/constants get silently
    // relocated to the caller's device by Graph::forward's device-portability
    // path) a loaded module quietly ignoring which device it was specialized
    // for. Guard shape, dtype, AND device (JIT-F030, R1-02).
    throw std::runtime_error(
        "CompiledModule: input shape, dtype, or device differs from the "
        "serialized trace and this module was loaded from disk, which cannot "
        "retrace (the source module is not serialized). Replaying would "
        "silently use the baked trace-time shapes/dtypes/device. Re-trace/"
        "re-export the module for the new shape/dtype/device, or call "
        "mark_dynamic_dims() on the loaded module before forwarding.");
}

auto CompiledModule::set_traced_signature(const std::vector<Variable>& example_inputs)
    -> void {
    traced_shape_key_ = compute_shape_key(example_inputs);
    if (!example_inputs.empty()) {
        traced_device_ = example_inputs[0].tensor().device();
        traced_dtype_  = example_inputs[0].tensor().dtype();
    }
}

auto CompiledModule::forward(const Variable& input) -> Variable {
    // Serialise the whole call: forward() reassigns graph_, inserts into
    // shape_cache_, and overwrites traced_device_/traced_dtype_ on the retrace
    // paths, and graph_->forward() must not run on a graph another thread is
    // swapping out. Without this a module shared across inference threads races.
    std::lock_guard<std::recursive_mutex> guard(forward_mutex_);

    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph");
    }

    // JIT-R004: route to the differentiable replay when the input requires
    // grad, exactly mirroring CompiledFunction::operator()'s any_requires_grad
    // gate (compile.cpp). Without this, a fused CUDA/ROCm subgraph's
    // execute_fused_gpu_node unconditionally returns a detached
    // (requires_grad=false) Variable ("generated kernels have NO backward") —
    // model.optimize_for_inference() followed by a call with a
    // requires_grad-True input would silently sever autograd instead of
    // either computing real grads or throwing.
    if (input.requires_grad() && is_grad_enabled()) {
        auto outs = forward_grad({input});
        if (outs.empty()) {
            throw std::runtime_error("CompiledModule produced no outputs");
        }
        return outs[0];
    }

    // If dynamic shapes are configured, bind symbolic dims to actual values
    if (!dynamic_dims_.empty()) {
        SymbolicShapeEnvironment env;
        for (const auto& dd : dynamic_dims_) {
            if (dd.input_idx == 0) {
                auto shape = input.tensor().shape();
                if (dd.dim >= 0 &&
                    static_cast<size_t>(dd.dim) < shape.size()) {
                    env.bind(dd.name, shape[static_cast<size_t>(dd.dim)]);
                }
            }
        }
        graph_->bind_symbolic_shapes(env);
    }

    // A loaded (non-retraceable) module must fail loudly on a shape, dtype, or
    // device change rather than replay a graph baked at the serialized trace.
    {
        auto s = input.tensor().shape();
        throw_if_loaded_shape_mismatch(
            {std::vector<int64_t>(s.begin(), s.end())},
            {static_cast<int>(input.tensor().dtype())},
            {input.tensor().device()});
    }

    // Device/dtype mismatch retrace: if the current graph was traced with a
    // different device or dtype than the incoming input, retrace using the
    // actual input. Cached by shape-key (which now encodes device+dtype).
    auto key = compute_shape_key(input);
    if (source_module_ && key != traced_shape_key_) {
        auto it = shape_cache_.find(key);
        if (it != shape_cache_.end()) {
            // R3-02: a CUDA/HIP graph captured against the pre-retrace
            // graph_ must not keep replaying once graph_ is swapped out from
            // under it -- has_cuda_graph()/replay_cuda_graph() would
            // otherwise keep reporting/replaying a graph that no longer
            // matches the module's actual computation.
            invalidate_cuda_graph();
            graph_ = it->second;
        } else {
            // Retrace on ANY shape/device/dtype change (compute_shape_key encodes
            // all three). A device/dtype miss must retrace (frozen constants carry
            // the old dtype; fused nodes are pinned to the old device). A
            // SHAPE-only change must ALSO retrace: creation ops (zeros/arange) and
            // other shape-derived values were baked at the trace shape, so
            // replaying the old graph on a new shape silently uses stale shapes.
            // Cache only while under the capacity bound; past it, use the fresh
            // graph for this call without growing the cache.
            auto retraced = CompiledModule::trace(source_module_, input);
            // Mirror THIS module's optimization state onto the retraced
            // replacement (JIT-R008): unconditionally optimizing here made
            // a never-optimized module start silently fusing (and thus
            // computing different numerics -- fusion is not numerically
            // neutral) the instant any input triggered a retrace, while a
            // module the caller HAD optimized stayed unfused on this path.
            if (was_optimized_) retraced->optimize_for_inference();
            // JIT-R015: propagate the dynamic-dims config and user metadata_
            // onto the retraced replacement too, mirroring was_optimized_
            // just above. Without this, a device/dtype/ShapeGuard-driven
            // retrace silently drops mark_dynamic_dims()'s symbolic-shape
            // structure — confirmed perf-only (compute_shape_key() already
            // retraces unconditionally on every shape change regardless, so
            // this was never a correctness bug), but it degrades a caller's
            // "reuse one dynamic-shape graph" configuration into "retrace on
            // every shape change" for the remaining lifetime of the module.
            if (!dynamic_dims_.empty()) retraced->mark_dynamic_dims(dynamic_dims_);
            retraced->metadata_ = metadata_;
            if (static_cast<int>(shape_cache_.size()) < MAX_RETRACES) {
                shape_cache_[key] = retraced->graph_;
            }
            invalidate_cuda_graph();  // R3-02: see comment above
            graph_ = retraced->graph_;
        }
        traced_device_ = input.tensor().device();
        traced_dtype_  = input.tensor().dtype();
        traced_shape_key_ = key;
    }

    auto results = graph_->forward({input});

    // Check if a ShapeGuard triggered a retrace request
    if (graph_->needs_retrace() && source_module_) {
        graph_->reset_retrace();

        auto key = compute_shape_key(input);

        // Check shape cache first
        auto it = shape_cache_.find(key);
        if (it != shape_cache_.end()) {
            invalidate_cuda_graph();  // R3-02: see comment above
            graph_ = it->second;
        } else if (static_cast<int>(shape_cache_.size()) < MAX_RETRACES) {
            // Re-trace with the new input shape and cache
            auto retraced = CompiledModule::trace(source_module_, input);
            // Mirror THIS module's optimization state onto the retraced
            // replacement (JIT-R008): unconditionally optimizing here made
            // a never-optimized module start silently fusing (and thus
            // computing different numerics -- fusion is not numerically
            // neutral) the instant any input triggered a retrace, while a
            // module the caller HAD optimized stayed unfused on this path.
            if (was_optimized_) retraced->optimize_for_inference();
            // JIT-R015: propagate the dynamic-dims config and user metadata_
            // onto the retraced replacement too, mirroring was_optimized_
            // just above. Without this, a device/dtype/ShapeGuard-driven
            // retrace silently drops mark_dynamic_dims()'s symbolic-shape
            // structure — confirmed perf-only (compute_shape_key() already
            // retraces unconditionally on every shape change regardless, so
            // this was never a correctness bug), but it degrades a caller's
            // "reuse one dynamic-shape graph" configuration into "retrace on
            // every shape change" for the remaining lifetime of the module.
            if (!dynamic_dims_.empty()) retraced->mark_dynamic_dims(dynamic_dims_);
            retraced->metadata_ = metadata_;
            shape_cache_[key] = retraced->graph_;
            invalidate_cuda_graph();  // R3-02: see comment above
            graph_ = retraced->graph_;
        }
        // else: too many distinct shapes, stay on current graph

        // Re-execute with new/cached graph
        results = graph_->forward({input});
    } else if (graph_->needs_retrace()) {
        // JIT-R015: a ShapeGuard mismatch was detected but this module has no
        // source_module_ to retrace with (e.g. loaded from serialization
        // without its originating nn::Module). Graph::forward() now stops as
        // soon as the guard trips rather than completing the run on
        // known-mismatched data, so `results` here is whatever was computed
        // before the trip (possibly empty) -- fail loudly and specifically
        // rather than either silently returning a partial/stale result or
        // falling through to the generic "produced no outputs" message below,
        // which wouldn't explain WHY.
        graph_->reset_retrace();
        throw std::runtime_error(
            "CompiledModule: input shape mismatch detected by a ShapeGuard, "
            "but no source module is available to retrace with (this "
            "CompiledModule was likely loaded from a serialized graph without "
            "its originating nn::Module). Cannot safely continue.");
    }

    if (results.empty()) {
        throw std::runtime_error("CompiledModule produced no outputs");
    }
    return results[0];
}

auto CompiledModule::forward(const Tensor& input) -> Variable {
    return forward(Variable(input, false));
}

auto CompiledModule::forward(const std::vector<Variable>& inputs) -> std::vector<Variable> {
    // Serialise: same mutable state (graph_, shape_cache_, traced_*) as the
    // single-input overload. See that overload for rationale.
    std::lock_guard<std::recursive_mutex> guard(forward_mutex_);

    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph");
    }

    // JIT-R004: same any_requires_grad gate as the single-input overload
    // above — see its comment for the fused-GPU-kernel-severs-autograd
    // rationale.
    {
        bool any_requires_grad = false;
        for (const auto& v : inputs) {
            if (v.requires_grad()) { any_requires_grad = true; break; }
        }
        if (any_requires_grad && is_grad_enabled()) {
            return forward_grad(inputs);
        }
    }

    // If dynamic shapes are configured, bind symbolic dims to actual values
    if (!dynamic_dims_.empty()) {
        SymbolicShapeEnvironment env;
        for (const auto& dd : dynamic_dims_) {
            if (dd.input_idx >= 0 &&
                static_cast<size_t>(dd.input_idx) < inputs.size()) {
                auto shape = inputs[static_cast<size_t>(dd.input_idx)].tensor().shape();
                if (dd.dim >= 0 &&
                    static_cast<size_t>(dd.dim) < shape.size()) {
                    env.bind(dd.name, shape[static_cast<size_t>(dd.dim)]);
                }
            }
        }
        graph_->bind_symbolic_shapes(env);
    }

    // Device/dtype mismatch retrace: mirror the single-input forward() so a
    // multi-input CompiledModule traced on one backend/precision retraces when
    // called with mismatched tensors instead of replaying the wrong-device
    // graph and hitting a device-mismatch error at dispatch. Keyed off
    // inputs[0] (the example_input used for tracing); cached by shape-key
    // (which encodes device+dtype).
    // Retrace on ANY shape/device/dtype change across the inputs. compute_shape_key
    // encodes every input's shape+device+dtype, so comparing it against the key
    // the current graph_ was specialized for catches all three (a device/dtype
    // change on any input, AND a shape-only change that would otherwise replay a
    // graph with baked trace-shape creation ops).
    // A loaded (non-retraceable) module must fail loudly on a shape change
    // rather than replay a graph baked at the serialized trace shape.
    {
        std::vector<std::vector<int64_t>> call_shapes;
        std::vector<int> call_dtypes;
        std::vector<Device> call_devices;
        call_shapes.reserve(inputs.size());
        call_dtypes.reserve(inputs.size());
        call_devices.reserve(inputs.size());
        for (const auto& v : inputs) {
            auto s = v.tensor().shape();
            call_shapes.emplace_back(s.begin(), s.end());
            call_dtypes.push_back(static_cast<int>(v.tensor().dtype()));
            call_devices.push_back(v.tensor().device());
        }
        throw_if_loaded_shape_mismatch(call_shapes, call_dtypes, call_devices);
    }

    auto key = compute_shape_key(inputs);
    if ((source_module_ || retrace_fn_) && !inputs.empty() && key != traced_shape_key_) {
        auto it = shape_cache_.find(key);
        if (it != shape_cache_.end()) {
            invalidate_cuda_graph();  // R3-02: see single-input overload's comment
            graph_ = it->second;
        } else {
            // Prefer the multi-input retrace closure when present (e.g. a
            // multi-argument script): the single-input source_module_ trace would
            // drop all but inputs[0] and throw an argument-count mismatch.
            auto retraced = retrace_fn_ ? retrace_fn_(inputs)
                                        : CompiledModule::trace(source_module_, inputs[0]);
            // Mirror THIS module's optimization state onto the retraced
            // replacement (JIT-R008): unconditionally optimizing here made
            // a never-optimized module start silently fusing (and thus
            // computing different numerics -- fusion is not numerically
            // neutral) the instant any input triggered a retrace, while a
            // module the caller HAD optimized stayed unfused on this path.
            if (was_optimized_) retraced->optimize_for_inference();
            // JIT-R015: propagate the dynamic-dims config and user metadata_
            // onto the retraced replacement too, mirroring was_optimized_
            // just above. Without this, a device/dtype/ShapeGuard-driven
            // retrace silently drops mark_dynamic_dims()'s symbolic-shape
            // structure — confirmed perf-only (compute_shape_key() already
            // retraces unconditionally on every shape change regardless, so
            // this was never a correctness bug), but it degrades a caller's
            // "reuse one dynamic-shape graph" configuration into "retrace on
            // every shape change" for the remaining lifetime of the module.
            if (!dynamic_dims_.empty()) retraced->mark_dynamic_dims(dynamic_dims_);
            retraced->metadata_ = metadata_;
            if (static_cast<int>(shape_cache_.size()) < MAX_RETRACES) {
                shape_cache_[key] = retraced->graph_;
            }
            invalidate_cuda_graph();  // R3-02: see single-input overload's comment
            graph_ = retraced->graph_;
        }
        traced_device_ = inputs[0].tensor().device();
        traced_dtype_  = inputs[0].tensor().dtype();
        traced_shape_key_ = key;
    }

    auto results = graph_->forward(inputs);

    // Check if a ShapeGuard triggered a retrace request
    if (graph_->needs_retrace() && (source_module_ || retrace_fn_) && !inputs.empty()) {
        graph_->reset_retrace();

        auto key = compute_shape_key(inputs);

        auto it = shape_cache_.find(key);
        if (it != shape_cache_.end()) {
            invalidate_cuda_graph();  // R3-02: see single-input overload's comment
            graph_ = it->second;
        } else if (static_cast<int>(shape_cache_.size()) < MAX_RETRACES) {
            auto retraced = retrace_fn_ ? retrace_fn_(inputs)
                                        : CompiledModule::trace(source_module_, inputs[0]);
            // Mirror THIS module's optimization state onto the retraced
            // replacement (JIT-R008): unconditionally optimizing here made
            // a never-optimized module start silently fusing (and thus
            // computing different numerics -- fusion is not numerically
            // neutral) the instant any input triggered a retrace, while a
            // module the caller HAD optimized stayed unfused on this path.
            if (was_optimized_) retraced->optimize_for_inference();
            // JIT-R015: propagate the dynamic-dims config and user metadata_
            // onto the retraced replacement too, mirroring was_optimized_
            // just above. Without this, a device/dtype/ShapeGuard-driven
            // retrace silently drops mark_dynamic_dims()'s symbolic-shape
            // structure — confirmed perf-only (compute_shape_key() already
            // retraces unconditionally on every shape change regardless, so
            // this was never a correctness bug), but it degrades a caller's
            // "reuse one dynamic-shape graph" configuration into "retrace on
            // every shape change" for the remaining lifetime of the module.
            if (!dynamic_dims_.empty()) retraced->mark_dynamic_dims(dynamic_dims_);
            retraced->metadata_ = metadata_;
            shape_cache_[key] = retraced->graph_;
            invalidate_cuda_graph();  // R3-02: see single-input overload's comment
            graph_ = retraced->graph_;
        }

        results = graph_->forward(inputs);
    } else if (graph_->needs_retrace()) {
        // JIT-R015: mirrors the single-input overload above -- a ShapeGuard
        // mismatch fired but there's no way to retrace (no source_module_/
        // retrace_fn_, or a niladic call with an empty inputs list).
        // Graph::forward() now stops as soon as the guard trips, so `results`
        // here may be empty/incomplete rather than the old fully-computed
        // (but already-known-wrong) run. Fail loudly instead of silently
        // returning a partial result to the caller.
        graph_->reset_retrace();
        throw std::runtime_error(
            "CompiledModule: input shape mismatch detected by a ShapeGuard, "
            "but no source module/retrace closure is available to retrace "
            "with. Cannot safely continue.");
    }

    return results;
}

auto CompiledModule::forward_grad(const std::vector<Variable>& inputs)
    -> std::vector<Variable> {
    // Serialise access to graph_ like the inference forwards do.
    std::lock_guard<std::recursive_mutex> guard(forward_mutex_);

    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph");
    }

    // Bind dynamic dims if configured (same as inference forward).
    if (!dynamic_dims_.empty()) {
        SymbolicShapeEnvironment env;
        for (const auto& dd : dynamic_dims_) {
            if (dd.input_idx >= 0 &&
                static_cast<size_t>(dd.input_idx) < inputs.size()) {
                auto shape = inputs[static_cast<size_t>(dd.input_idx)].tensor().shape();
                if (dd.dim >= 0 &&
                    static_cast<size_t>(dd.dim) < shape.size()) {
                    env.bind(dd.name, shape[static_cast<size_t>(dd.dim)]);
                }
            }
        }
        graph_->bind_symbolic_shapes(env);
    }

    // Differentiable replay. No CUDA-graph capture and no fused-kernel path is
    // reachable here: the grad variant was compiled without fusion, and
    // Graph::execute_node throws on any fusion node in grad mode.
    auto results = graph_->forward(inputs, /*grad_mode=*/true);

    // JIT-R030: unlike the two inference forward() overloads, this graph_
    // has no source_module_/retrace_fn_ counterpart that produces a
    // DIFFERENTIABLE (unfused) replacement graph — those closures only ever
    // retrace in inference mode (CompiledModule::trace / retrace_fn_), so
    // silently reusing them here would swap in a non-differentiable fused
    // graph, defeating the whole point of forward_grad. There is therefore
    // no safe self-heal available at this layer. Per the documented
    // contract in graph.cpp ("the caller is required to check
    // needs_retrace() before trusting this result"), forward_grad was the
    // one caller that skipped this check entirely: Graph::forward() stops
    // executing as soon as a baked ShapeGuard trips on a call at a
    // different shape than this graph was traced/retraced for, so
    // `results` here can be empty or a partial prefix of the real output
    // list — silently returning it let a misleadingly-worded, UNCAUGHT
    // "produced no outputs" error surface deep in
    // CompiledFunction::grad_invoke instead. Throw HERE instead: this
    // propagates through grad_invoke's existing try/catch around its
    // `compiled_module->forward_grad(...)` call, which already correctly
    // falls back to eager autograd on the input's backend (the same
    // recovery inference already gets via retrace) — this just makes sure
    // grad_invoke's ALREADY-CORRECT fallback actually triggers instead of
    // being bypassed by a silently-wrong result.
    if (graph_->needs_retrace()) {
        graph_->reset_retrace();
        throw std::runtime_error(
            "CompiledModule::forward_grad: input shape (or device/dtype) "
            "differs from the shape this differentiable graph was traced "
            "for, tripping a ShapeGuard mid-replay. The grad-mode graph "
            "cannot self-heal via retrace at this layer (only inference "
            "mode caches a retrace closure) — the caller must retrace via "
            "jit.compile()'s cache-miss path instead (this exception is "
            "expected to be caught there and trigger an eager-autograd "
            "fallback).");
    }

    return results;
}

auto CompiledModule::optimize_for_inference() -> int {
    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph to optimize");
    }

    // Use Compiler directly so we can capture the memory plan
    Compiler compiler(true);
    int result = compiler.optimize(*graph_);
    memory_plan_ = compiler.memory_plan();
    was_optimized_ = true;
    return result;
}

auto CompiledModule::mark_dynamic_dims(const std::vector<DynamicDimSpec>& dynamic_dims) -> void {
    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph");
    }

    dynamic_dims_ = dynamic_dims;

    // Create and run a SymbolicTracePass to propagate symbolic shapes
    SymbolicTracePass trace_pass;
    for (const auto& dd : dynamic_dims_) {
        trace_pass.mark_dynamic(dd.input_idx, dd.dim, dd.name);
    }
    trace_pass.run(*graph_);
}

auto CompiledModule::save(const std::string& path) const -> void {
    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph to save");
    }
    // Push the module's user KV metadata onto the graph so it is serialized
    // (the graph is the unit save_graph persists). Without this, save/load was
    // lossy for metadata — a reloaded module had an empty metadata map.
    for (const auto& [key, value] : metadata_) {
        graph_->set_string_metadata(key, value);
    }
    // F031: serialize the dynamic-dim configuration so a module made dynamic via
    // mark_dynamic_dims() loads back DYNAMIC. Previously neither the Value
    // symbolic_shape_ nor dynamic_dims_ was serialized, so a dynamic module
    // reloaded as a STATIC graph locked to the trace-time shape — and, with the
    // loaded-module exact-shape guard (F030), silently rejected the variable
    // shapes it was configured to accept. Encoded as a reserved metadata KV
    // (`input_idx:dim:name` entries joined by ';') via the existing string-metadata
    // round-trip (no binary-format change); load() re-applies mark_dynamic_dims,
    // which re-runs SymbolicTracePass to rebuild the symbolic shapes.
    if (!dynamic_dims_.empty()) {
        std::string encoded;
        for (const auto& dd : dynamic_dims_) {
            if (!encoded.empty()) encoded += ';';
            encoded += std::to_string(dd.input_idx) + ':' +
                       std::to_string(dd.dim) + ':' + dd.name;
        }
        graph_->set_string_metadata("__tenzor_dynamic_dims__", encoded);
    }
    // R1-02: serialize each input's trace-time device the same side-channel way
    // F031 preserves dynamic dims, since GraphWriter::write_values always emits
    // Device::Type::CPU for portability (a deliberate, separate design choice —
    // it lets a graph traced on one backend replay on another) and so drops the
    // device this in-memory graph_ was actually traced/frozen on. Without this,
    // a loaded module has no record of its original device, and the loaded-
    // module compatibility guard (throw_if_loaded_shape_mismatch) cannot detect
    // a same-shape/same-dtype call arriving on a different device than the one
    // this module's frozen constants were specialized for.
    {
        std::string encoded;
        for (const auto& in : graph_->inputs()) {
            if (!encoded.empty()) encoded += ';';
            if (in) encoded += in->device().to_string();
        }
        graph_->set_string_metadata("__tenzor_input_devices__", encoded);
    }
    // Save the graph using existing serialization
    save_graph(*graph_, path);
}

auto CompiledModule::load(const std::string& path) -> std::shared_ptr<CompiledModule> {
    auto graph = load_graph(path);
    if (!graph) {
        throw std::runtime_error("Failed to load graph from: " + path);
    }
    auto module = std::make_shared<CompiledModule>(graph);
    // Restore user KV metadata that was serialized with the graph (skipping the
    // internal dynamic-dims / input-devices keys, which are not user metadata).
    for (const auto& [key, value] : graph->string_metadata()) {
        if (key == "__tenzor_dynamic_dims__" || key == "__tenzor_input_devices__") {
            continue;
        }
        module->add_metadata(key, value);
    }
    // R1-02: parse back the per-input trace-time devices saved as a side-channel
    // (see the matching comment in save()). Falls back to an empty list (later
    // defaulted to CPU per input) for a file saved before this fix existed, or on
    // a malformed entry -- CPU matches the only device such an older file could
    // have been meaningfully used from anyway, since nothing preserved the real
    // device before this change.
    std::vector<Device> saved_input_devices;
    {
        const auto& md = graph->string_metadata();
        auto it = md.find("__tenzor_input_devices__");
        if (it != md.end() && !it->second.empty()) {
            std::stringstream ss(it->second);
            std::string entry;
            while (std::getline(ss, entry, ';')) {
                try {
                    saved_input_devices.push_back(
                        entry.empty() ? Device::cpu() : Device::from_string(entry));
                } catch (const std::exception&) {
                    saved_input_devices.push_back(Device::cpu());
                }
            }
        }
    }
    // F031: re-apply the serialized dynamic-dim configuration so the module loads
    // back DYNAMIC (matching what mark_dynamic_dims() configured before save),
    // instead of a static graph locked to the trace-time shape.
    {
        const auto& md = graph->string_metadata();
        auto it = md.find("__tenzor_dynamic_dims__");
        if (it != md.end() && !it->second.empty()) {
            std::vector<DynamicDimSpec> specs;
            std::stringstream ss(it->second);
            std::string entry;
            while (std::getline(ss, entry, ';')) {
                const auto p1 = entry.find(':');
                if (p1 == std::string::npos) continue;
                const auto p2 = entry.find(':', p1 + 1);
                if (p2 == std::string::npos) continue;
                try {
                    DynamicDimSpec dd;
                    dd.input_idx = std::stoi(entry.substr(0, p1));
                    dd.dim = std::stoi(entry.substr(p1 + 1, p2 - p1 - 1));
                    dd.name = entry.substr(p2 + 1);
                    specs.push_back(std::move(dd));
                } catch (const std::exception&) {
                    // Malformed entry — skip it rather than fail the load.
                }
            }
            if (!specs.empty()) {
                module->mark_dynamic_dims(specs);  // re-runs SymbolicTracePass
            }
        }
    }
    // A loaded module has no source module / retrace closure (neither is
    // serialized), so it cannot retrace. Record that fact and the graph's baked
    // input shapes so forward() fails loudly on a shape change instead of
    // silently replaying the trace-time-baked graph (JIT-F011).
    module->loaded_ = true;
    const auto& inputs = module->graph_->inputs();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& in = inputs[i];
        if (in) {
            auto s = in->shape();
            module->loaded_input_shapes_.emplace_back(s.begin(), s.end());
            module->loaded_input_dtypes_.push_back(static_cast<int>(in->dtype()));
        } else {
            module->loaded_input_shapes_.emplace_back();
            module->loaded_input_dtypes_.push_back(-1);
        }
        module->loaded_input_devices_.push_back(
            i < saved_input_devices.size() ? saved_input_devices[i] : Device::cpu());
    }
    return module;
}

auto CompiledModule::add_metadata(const std::string& key, const std::string& value) -> void {
    metadata_[key] = value;
}

auto CompiledModule::get_metadata(const std::string& key) const -> std::string {
    auto it = metadata_.find(key);
    return it != metadata_.end() ? it->second : "";
}

auto CompiledModule::has_metadata(const std::string& key) const -> bool {
    return metadata_.find(key) != metadata_.end();
}

auto CompiledModule::all_metadata() const -> const std::unordered_map<std::string, std::string>& {
    return metadata_;
}

// ============================================================================
// ROCm Graph Adapter (wraps HIPGraph behind the CUDAGraph interface)
// ============================================================================

#ifdef TENZOR_HAS_ROCM
namespace {

class ROCmGraphAdapter : public CUDAGraph {
public:
    explicit ROCmGraphAdapter(int32_t device_id) : device_id_(device_id) {
        HIP_CHECK(hipSetDevice(device_id_));
        auto err = hipStreamCreate(&stream_);
        if (err != hipSuccess) {
            throw std::runtime_error(
                std::string("ROCmGraphAdapter: failed to create stream: ") +
                hipGetErrorString(err));
        }
    }

    ~ROCmGraphAdapter() override {
        if (hip_graph_.is_capturing()) {
            // Abort capture to leave stream in valid state
            hipGraph_t dummy = nullptr;
            (void)hipStreamEndCapture(stream_, &dummy);
            if (dummy) (void)hipGraphDestroy(dummy);
        }
        if (stream_) {
            (void)hipStreamDestroy(stream_);
        }
    }

    void prepare_capture_stream() override {
        HIP_CHECK(hipSetDevice(device_id_));
    }

    void begin_capture() override {
        HIP_CHECK(hipSetDevice(device_id_));
        hip_graph_.begin_capture(stream_);
    }

    void end_capture() override {
        hip_graph_.end_capture(stream_);
    }

    void replay() override {
        HIP_CHECK(hipSetDevice(device_id_));
        hip_graph_.replay(stream_);
    }

    bool is_ready() const override {
        return hip_graph_.is_compiled();
    }

private:
    int32_t device_id_;
    hipStream_t stream_ = nullptr;
    rocm::HIPGraph hip_graph_;
};

} // anonymous namespace
#endif // TENZOR_HAS_ROCM

// ============================================================================
// CUDA / ROCm Graph Capture/Replay
// ============================================================================

auto CompiledModule::capture_cuda_graph(std::vector<Tensor> sample_inputs) -> void {
    // Guard cuda_graph_/captured_shapes_ against concurrent forward()/replay on
    // a shared module. Recursive mutex so the invalidate_cuda_graph() call below
    // (which re-locks) does not self-deadlock.
    std::lock_guard<std::recursive_mutex> guard(forward_mutex_);

    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph");
    }

    // Invalidate any existing captured graph
    invalidate_cuda_graph();

    // Determine device ID and type from the first input tensor
    int32_t device_id = 0;
    Device::Type device_type = Device::Type::CPU;
    if (!sample_inputs.empty()) {
        device_id = sample_inputs[0].device().index;
        device_type = sample_inputs[0].device().type;
    }

    // Create the graph capture object via the backend-registered factory for
    // BOTH CUDA and ROCm. Backends are dlopen'd RTLD_LOCAL, so the real
    // CUDA/HIP graph implementation cannot be linked directly — the CUDA backend
    // registers a CUDA-graph factory and the ROCm backend a HIP-graph factory
    // (both stream-aware: they route the captured forward() onto the capture
    // stream). Capture is a CUDA/ROCm-only optimization; any other device type
    // is refused rather than recording an empty graph over non-CUDA work.
    {
        if (device_type != Device::Type::CUDA &&
            device_type != Device::Type::ROCm) {
            throw std::runtime_error(
                "capture_cuda_graph: CUDA-graph capture is only supported on "
                "CUDA/ROCm devices; got " +
                Device{device_type, device_id}.to_string());
        }
        cuda_graph_ =
            CUDAGraph::create_for(static_cast<int>(device_type), device_id);
        if (!cuda_graph_) {
            throw std::runtime_error(
                "GPU is not available; cannot capture graph (device type: " +
                Device{device_type, device_id}.to_string() + ")");
        }
    }

    // Record input shapes for validation during replay
    captured_shapes_.clear();
    captured_shapes_.reserve(sample_inputs.size());
    for (const auto& t : sample_inputs) {
        auto s = t.shape();
        captured_shapes_.emplace_back(s.begin(), s.end());
    }

    // Retain the EXACT input buffers the captured graph will hard-code so that
    // replay() can copy fresh inputs into them (device-to-device) instead of
    // re-running over stale capture-time data. The same Tensor objects are
    // wrapped into the Variables fed to forward(), so the graph captures these
    // very buffers.
    //
    // clone() (not just contiguous()) is REQUIRED: contiguous() returns the
    // caller's tensor unchanged when it is already contiguous, so the retained
    // buffer would alias the caller's sample-input storage. replay() then does a
    // device-to-device copy of fresh data INTO these buffers — which would
    // silently overwrite the caller's still-live capture-time tensor. A deep
    // copy gives the module private buffers that only replay() writes to.
    captured_inputs_.clear();
    captured_inputs_.reserve(sample_inputs.size());
    for (auto& t : sample_inputs) {
        captured_inputs_.push_back(t.contiguous().clone());
    }

    // Wrap the RETAINED contiguous inputs as Variables for the capture forward
    // pass — capturing the buffers we hold in captured_inputs_, which replay
    // then overwrites with fresh data.
    std::vector<Variable> vars;
    vars.reserve(captured_inputs_.size());
    for (auto& t : captured_inputs_) {
        vars.emplace_back(t, false);
    }

    // Capture: all GPU work submitted between begin/end is recorded. Retain the
    // output Variables' tensors: their device storage IS the buffer the captured
    // graph's terminal nodes write into, so after a replay() these tensors hold
    // the FRESH results. Exposing them via replay_cuda_graph_outputs() makes the
    // replay path usable (and verifiable) — without this the caller has no way
    // to read a replay's output.
    captured_outputs_.clear();
    auto reset_capture_state = [&]() {
        // Order matters: free captured tensors BEFORE destroying cuda_graph_.
        // See invalidate_cuda_graph()'s comment -- captured_intermediates_ can
        // hold buffers the CachingAllocator tagged with the capture stream;
        // freeing them after cuda_graph_.reset() has cudaStreamDestroy()'d
        // that stream is a use-after-free of the stream handle.
        captured_shapes_.clear();
        captured_inputs_.clear();
        captured_outputs_.clear();
        captured_intermediates_.clear();
        cuda_graph_.reset();
    };
    // Warm up: run the forward once OUTSIDE capture so the caching allocator has
    // every intermediate/output buffer cached. CUDA/HIP forbid cudaMalloc while a
    // stream is capturing, so allocations during capture must be served from the
    // cache (no driver allocation). The warmup populates that cache.
    //
    // Route the warmup onto the SAME dedicated stream begin_capture() will use
    // (without actually starting capture) so every buffer the warmup allocates
    // is tagged with that stream. Otherwise warmup runs on the default/legacy
    // stream, and the real captured pass below reusing those cached blocks from
    // a DIFFERENT (capture) stream is a cross-stream reuse: the allocator must
    // insert a stream-to-stream wait to stay safe, which CUDA/HIP reject while
    // the target stream is actively capturing ("operation would make the legacy
    // stream depend on a capturing blocking stream"). Same-stream reuse needs no
    // such wait.
    cuda_graph_->prepare_capture_stream();
    try {
        auto warm = graph_->forward(vars);
        (void)warm;
    } catch (...) {
        reset_capture_state();
        throw;
    }
    // Drain the warmup and any prior async work so it is not swept into the
    // capture, and so the capture starts from a quiescent device.
    if (!captured_inputs_.empty()) {
        captured_inputs_.front().device().synchronize();
    }
    cuda_graph_->begin_capture();
    try {
        // Collect every Variable this (capture-time) forward() call touches —
        // not just the declared outputs_ — so every intermediate node result's
        // device buffer can be pinned below. See captured_intermediates_'s doc
        // comment: an intermediate whose Tensor is allowed to go out of scope
        // here would be freed back to the shared allocator's free list even
        // though the graph just captured is about to reference that exact
        // address on every future replay.
        std::vector<Variable> all_vars;
        auto out_vars = graph_->forward(vars, /*grad_mode=*/false, &all_vars);
        captured_outputs_.reserve(out_vars.size());
        for (auto& ov : out_vars) {
            captured_outputs_.push_back(ov.tensor());
        }
        captured_intermediates_.clear();
        captured_intermediates_.reserve(all_vars.size());
        for (auto& v : all_vars) {
            captured_intermediates_.push_back(v.tensor());
        }
    } catch (...) {
        // If forward fails during capture, we must still end capture to
        // leave the stream in a valid state. The graph will be unusable.
        try {
            cuda_graph_->end_capture();
        } catch (...) {
            // Ignore end_capture errors during cleanup
        }
        reset_capture_state();
        throw;
    }
    // end_capture can FAIL if the forward work was not actually recorded onto the
    // capture stream (e.g. a backend that launches on the legacy default stream
    // during a global-mode capture). Rather than instantiate/replay a graph that
    // is empty or invalid — which would silently return stale capture-time
    // outputs — reset all capture state so has_cuda_graph() is false and the
    // caller transparently falls back to the correct normal forward() path.
    try {
        cuda_graph_->end_capture();
    } catch (...) {
        reset_capture_state();
        return;
    }
    if (!cuda_graph_ || !cuda_graph_->is_ready()) {
        reset_capture_state();
    }
}

auto CompiledModule::replay_cuda_graph_outputs() const -> std::vector<Tensor> {
    std::lock_guard<std::recursive_mutex> guard(forward_mutex_);
    // Clone the static capture buffers (JIT-005). The next replay() overwrites
    // captured_outputs_ IN PLACE, so returning them directly would alias
    // consecutive results — `y1 = f(x1); y2 = f(x2);` would leave y1 sharing
    // y2's buffer. Cloning gives each caller an independent output.
    std::vector<Tensor> out;
    out.reserve(captured_outputs_.size());
    for (const auto& t : captured_outputs_) out.push_back(t.clone());
    return out;
}

auto CompiledModule::replay_cuda_graph(std::vector<Tensor>& inputs) -> bool {
    std::lock_guard<std::recursive_mutex> guard(forward_mutex_);

    if (!cuda_graph_ || !cuda_graph_->is_ready()) {
        return false;
    }

    // Validate input shapes match captured shapes
    if (inputs.size() != captured_shapes_.size()) {
        throw std::runtime_error(
            "CUDA graph replay: expected " +
            std::to_string(captured_shapes_.size()) + " inputs, got " +
            std::to_string(inputs.size()));
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto in_shape = inputs[i].shape();
        const auto& cap_shape = captured_shapes_[i];
        if (in_shape.size() != cap_shape.size() ||
            !std::equal(in_shape.begin(), in_shape.end(), cap_shape.begin())) {
            throw std::runtime_error(
                "CUDA graph replay: input " + std::to_string(i) +
                " shape mismatch (graph was captured with different shapes)");
        }
    }

    // ------------------------------------------------------------------
    // Copy fresh inputs into the captured buffers (device-to-device).
    //
    // A captured CUDA/HIP graph re-runs verbatim over the device pointers it
    // recorded at capture time (captured_inputs_) — it has no API to rebind
    // inputs. Without this copy-in, replay() ignores `inputs` entirely and
    // returns the stale capture-time result whenever the caller passes a
    // different (or mutated) input tensor. The shape check above is necessary
    // but NOT sufficient; we must actually move the new data into the captured
    // buffers before replaying.
    //
    // captured_inputs_[i] is contiguous (forced at capture). We make each fresh
    // input contiguous + dtype/device-matched, then issue a flat
    // DeviceToDevice copy through the backend's copy primitive
    // (cudaMemcpyAsync / hipMemcpyAsync under the hood). The copy is enqueued on
    // the default stream, which the captured graph's replay also uses, so the
    // copy is ordered before the replayed kernels read the buffers.
    // ------------------------------------------------------------------
    if (captured_inputs_.size() != inputs.size()) {
        // capture/replay invariant violated — refuse rather than read stale data
        throw std::runtime_error(
            "CUDA graph replay: captured input count (" +
            std::to_string(captured_inputs_.size()) +
            ") does not match replay input count (" +
            std::to_string(inputs.size()) + ")");
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        // The captured graph hard-codes the device (type + ordinal) and dtype of
        // its buffers. A replay input on a different device ordinal or dtype must
        // NOT be silently moved onto the captured device — that would migrate the
        // caller's data across GPUs and run on the wrong one. Refuse replay so
        // the caller falls back to a normal forward on the input's own device.
        if (inputs[i].device() != captured_inputs_[i].device() ||
            inputs[i].dtype() != captured_inputs_[i].dtype()) {
            return false;
        }
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor& dst = captured_inputs_[i];
        // Match dtype/device to the captured buffer, then force contiguous so a
        // flat byte copy is valid. (.to() is a no-op when already matching;
        // .contiguous() is a no-op when already contiguous.)
        Tensor src = inputs[i];
        if (src.dtype() != dst.dtype()) {
            throw std::runtime_error(
                "CUDA graph replay: input " + std::to_string(i) +
                " dtype mismatch with captured buffer");
        }
        if (src.device() != dst.device()) {
            src = src.to(dst.device());
        }
        src = src.contiguous();

        const size_t bytes =
            static_cast<size_t>(dst.numel()) * dst.dtype_size();
        if (bytes == 0) {
            continue;
        }

        auto* backend = backend_registry().get_backend(dst.device().type);
        if (backend == nullptr) {
            throw std::runtime_error(
                "CUDA graph replay: no backend available for device " +
                dst.device().to_string());
        }
        // Both buffers live on the GPU; this is a device-to-device transfer.
        backend->copy(dst.data_ptr(), src.data_ptr(), bytes,
                      CopyKind::DeviceToDevice);
    }

    cuda_graph_->replay();
    return true;
}

auto CompiledModule::invalidate_cuda_graph() -> void {
    std::lock_guard<std::recursive_mutex> guard(forward_mutex_);
    // Free the captured tensors BEFORE destroying cuda_graph_ (and with it the
    // private capture stream). captured_intermediates_ holds buffers that were
    // allocated DURING capture, so the CachingAllocator tagged their blocks
    // with the capture stream; freeing one records a CUDA event against that
    // same stream (CachingAllocator::free -> record_free_event ->
    // cudaEventRecord) so a later cross-stream reuse can wait for it. Resetting
    // cuda_graph_ first calls cudaStreamDestroy() on that stream, so the event
    // record above would run against an already-destroyed stream handle --
    // undefined behavior, observed as a SIGSEGV inside libcuda.so.
    captured_shapes_.clear();
    captured_inputs_.clear();
    captured_outputs_.clear();
    captured_intermediates_.clear();
    cuda_graph_.reset();
}

auto CompiledModule::has_cuda_graph() const -> bool {
    return cuda_graph_ && cuda_graph_->is_ready();
}

// ============================================================================
// Free function convenience wrappers
// ============================================================================

auto optimize_for_inference(std::shared_ptr<CompiledModule> module) -> int {
    if (!module) {
        throw std::runtime_error("Cannot optimize null module");
    }
    return module->optimize_for_inference();
}

auto save(const std::shared_ptr<CompiledModule>& module, const std::string& path) -> void {
    if (!module) {
        throw std::runtime_error("Cannot save null module");
    }
    module->save(path);
}

auto load(const std::string& path) -> std::shared_ptr<CompiledModule> {
    return CompiledModule::load(path);
}

auto add_metadata(const std::shared_ptr<CompiledModule>& module,
                  const std::string& key, const std::string& value) -> void {
    if (!module) {
        throw std::runtime_error("Cannot add metadata to null module");
    }
    module->add_metadata(key, value);
}

auto get_metadata(const std::shared_ptr<CompiledModule>& module,
                  const std::string& key) -> std::string {
    if (!module) {
        throw std::runtime_error("Cannot get metadata from null module");
    }
    return module->get_metadata(key);
}

} // namespace jit
} // namespace tenzor
