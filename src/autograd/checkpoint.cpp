/**
 * @file checkpoint.cpp
 * @brief Complete implementation of gradient checkpointing
 */

#include "../../include/tenzor/autograd/checkpoint.hpp"
#include "../../include/tenzor/nn/module.hpp"
#include "../../include/tenzor/ops/creation.hpp"
#include "../../include/tenzor/ops/math.hpp"
#include "../../include/tenzor/ops/reduction.hpp"
#include "../../include/tenzor/utils/log.hpp"
#include "../../include/tenzor/autograd/ops.hpp"
#include "../../include/tenzor/nn/utils/variable_cast.hpp"
#include "../../include/tenzor/backend/loader.hpp"
#include "../../include/tenzor/backend/backend.hpp"
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <functional>

namespace tenzor {
namespace autograd {

// ============================================================================
// Global Statistics
// ============================================================================

// Thread-local storage for checkpoint statistics
thread_local CheckpointStats global_checkpoint_stats;
thread_local bool checkpoint_enabled = true;

auto get_checkpoint_stats() -> CheckpointStats& {
    return global_checkpoint_stats;
}

auto reset_checkpoint_stats() -> void {
    global_checkpoint_stats = CheckpointStats{};
}

auto is_checkpoint_enabled() -> bool {
    return checkpoint_enabled;
}

auto set_checkpoint_enabled(bool enabled) -> void {
    checkpoint_enabled = enabled;
}

// ============================================================================
// CheckpointFunction Implementation
// ============================================================================

CheckpointFunction::CheckpointFunction(
    std::function<std::vector<Variable>(const std::vector<Variable>&)> forward_fn,
    bool allow_caching,
    bool verify
) : forward_fn_(std::move(forward_fn)),
    allow_caching_(allow_caching),
    verify_(verify),
    verification_done_(false),
    recompute_count_(0),
    estimated_activation_memory_(0),
    has_cached_outputs_(false) {}

auto CheckpointFunction::forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> {
    // NOTE: This function is NOT used by the checkpoint() free function.
    // It's here for completeness but should not be called directly because
    // `inputs` are passed by value, making input_variables_ pointers invalid.
    // The checkpoint() free function sets up the graph manually instead.
    throw std::runtime_error("CheckpointFunction::forward should not be called directly. Use checkpoint() free function.");
}

auto CheckpointFunction::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Audit D.3: a CheckpointFunction is only ever installed as a grad_fn
    // when checkpointing was enabled at forward time (see
    // checkpoint_impl_shared early-exit). If the user toggles
    // is_checkpoint_enabled() to false between forward and backward, we
    // must still produce correct gradients by recomputing the saved
    // forward and running a normal autograd backward through it — not
    // return zeros, which would silently corrupt training. The previous
    // zero-grad branch is therefore intentionally removed.

    // Time the recomputation
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create fresh Variables with gradient tracking enabled for recomputation
    // These form an independent autograd graph that can handle nested checkpoints
    cached_recompute_inputs_.clear();
    cached_recompute_inputs_.reserve(saved_tensors().size());
    for (const auto& tensor : saved_tensors()) {
        cached_recompute_inputs_.emplace_back(tensor, true);
    }

    // Recompute forward pass - nested checkpoints work naturally in this fresh graph
    auto recomputed_outputs = recompute_forward(cached_recompute_inputs_);

    // Validate output count
    if (recomputed_outputs.size() != grad_outputs.size()) {
        throw std::runtime_error("Checkpoint backward: output count mismatch");
    }

    // Use standard autograd backward - handles all complexity including nested checkpoints
    for (size_t i = 0; i < recomputed_outputs.size(); ++i) {
        if (recomputed_outputs[i].requires_grad() && recomputed_outputs[i].grad_fn()) {
            // Match the seed's dtype to the recomputed output if the inner
            // forward upcast (e.g. FP16 -> FP32); a mismatched seed dtype would
            // throw or mis-accumulate in the recompute backward. Mirrors the
            // dtype match in backward_with_variables below.
            Tensor seed = grad_outputs[i];
            if (seed.dtype() != recomputed_outputs[i].tensor().dtype()) {
                seed = seed.to(recomputed_outputs[i].tensor().dtype());
            }
            // retain_graph=true allows multiple outputs and nested checkpoints
            recomputed_outputs[i].backward(seed, /*retain_graph=*/true);
        }
    }

    // Extract gradients from the recomputed inputs
    // Return gradients directly for the engine to accumulate through shared impl_
    std::vector<Tensor> input_grads;
    input_grads.reserve(cached_recompute_inputs_.size());

    for (size_t i = 0; i < cached_recompute_inputs_.size(); ++i) {
        if (cached_recompute_inputs_[i].has_grad()) {
            input_grads.push_back(cached_recompute_inputs_[i].grad().value());
        } else {
            // No gradient computed - return zero
            input_grads.push_back(Tensor::zeros_like(cached_recompute_inputs_[i].tensor()));
        }
    }

    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    auto& stats = get_checkpoint_stats();
    stats.num_recomputations++;
    stats.total_recompute_time_ms += duration.count() / 1000.0;
    recompute_count_++;

    // Invalidate cached recomputation outputs after backward to avoid stale data
    // if backward is called again (e.g., with retain_graph=true)
    has_cached_outputs_ = false;
    cached_recompute_outputs_.clear();

    return input_grads;
}

auto CheckpointFunction::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    // audit-10 NN.3: higher-order checkpoint backward. Recompute the forward
    // with gradient tracking, then call the inner backward with
    // create_graph=true so the inner ops build a second-order graph; harvest
    // graph-carrying grads via grad_variable() on the recomputed inputs.

    auto start_time = std::chrono::high_resolution_clock::now();

    cached_recompute_inputs_.clear();
    cached_recompute_inputs_.reserve(saved_tensors().size());
    for (const auto& tensor : saved_tensors()) {
        cached_recompute_inputs_.emplace_back(tensor, true);
    }

    auto recomputed_outputs = recompute_forward(cached_recompute_inputs_);

    if (recomputed_outputs.size() != grad_outputs.size()) {
        throw std::runtime_error("Checkpoint backward_with_variables: output count mismatch");
    }

    // audit-11 QQ.2: previously called recomputed_outputs[i].backward(
    // grad_outputs[i].tensor(), ...) — .tensor() strips grad_fn from the
    // outer Variable seed, so any third-order derivative through checkpoint
    // was wrong. The Variable engine entry point only accepts a Tensor seed,
    // so we build a scalar surrogate loss = sum_i sum(recomputed_outputs[i] *
    // grad_outputs[i]) and back-propagate from THAT (standard VJP identity:
    // d/dx <go, f(x)> == J(f)^T go). This keeps grad_outputs[i]'s Variable
    // chain wired into the recompute graph for higher-order grads.
    std::optional<Variable> surrogate;
    for (size_t i = 0; i < recomputed_outputs.size(); ++i) {
        if (!(recomputed_outputs[i].requires_grad() && recomputed_outputs[i].grad_fn())) {
            continue;
        }
        Variable go = grad_outputs[i];
        // Match dtype/device of the recomputed output if the inner forward
        // upcast (e.g. FP16 → FP32) — otherwise the multiply would throw.
        if (go.tensor().dtype() != recomputed_outputs[i].tensor().dtype()) {
            go = tenzor::nn::variable_cast(go, recomputed_outputs[i].tensor().dtype());
        }
        auto prod = recomputed_outputs[i] * go;
        auto term = tenzor::sum(prod);
        if (!surrogate.has_value()) {
            surrogate = term;
        } else {
            surrogate = *surrogate + term;
        }
    }
    if (surrogate.has_value()) {
        // retain_graph=true + create_graph=true so the second-order graph
        // survives the inner backward.
        surrogate->backward(std::nullopt,
                            /*retain_graph=*/true,
                            /*create_graph=*/true);
    }

    std::vector<Variable> input_grads;
    input_grads.reserve(cached_recompute_inputs_.size());

    for (size_t i = 0; i < cached_recompute_inputs_.size(); ++i) {
        const auto& gv = cached_recompute_inputs_[i].grad_variable();
        if (gv.has_value()) {
            input_grads.push_back(*gv);
        } else if (cached_recompute_inputs_[i].has_grad()) {
            // Fall back to a non-graph-carrying Variable for the tensor grad.
            input_grads.emplace_back(
                cached_recompute_inputs_[i].grad().value(), false);
        } else {
            // No gradient computed - return a zero Variable (no graph).
            input_grads.emplace_back(
                Tensor::zeros_like(cached_recompute_inputs_[i].tensor()), false);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    auto& stats = get_checkpoint_stats();
    stats.num_recomputations++;
    stats.total_recompute_time_ms += duration.count() / 1000.0;
    recompute_count_++;

    has_cached_outputs_ = false;
    cached_recompute_outputs_.clear();

    return input_grads;
}

void CheckpointFunction::release_op_specific_state() {
    // audit-10 NN.6: drop ad-hoc per-checkpoint state.
    cached_recompute_outputs_.clear();
    cached_recompute_inputs_.clear();
    recomputed_intermediates_.clear();
    has_cached_outputs_ = false;
}

auto CheckpointFunction::get_memory_savings() const -> size_t {
    return estimated_activation_memory_;
}

auto CheckpointFunction::save_rng_state(const std::vector<Device>& input_devices) -> void {
    saved_global_rng_ = save_global_rng_state();

    // Build the set of devices whose default Generator we should snapshot:
    // every distinct input device plus, conservatively, CPU (host-side ops
    // and CPU-driven kernels share the CPU default generator).
    std::vector<Device> devs;
    devs.reserve(input_devices.size() + 1);
    auto already_have = [&](const Device& d) {
        for (const auto& e : devs) {
            if (e == d) return true;
        }
        return false;
    };
    for (const auto& d : input_devices) {
        if (!already_have(d)) devs.push_back(d);
    }
    Device cpu_dev = Device::cpu();
    if (!already_have(cpu_dev)) devs.push_back(cpu_dev);

    // V-06: also snapshot every available backend device's default generator.
    // A stochastic op inside forward_fn_ may draw on a device that is neither an
    // input device nor CPU (e.g. inputs on CPU but a parameter/op on CUDA); if
    // that device's generator is not restored on recompute, the replayed random
    // draw (dropout mask, etc.) diverges from the original forward and produces
    // gradients that disagree with the non-checkpointed path — and differently
    // across backends. Enumerate defensively; never let RNG bookkeeping throw.
    if (is_backend_registry_alive()) {
        try {
            auto& reg = backend_registry();
            for (const auto& name : reg.available_backends()) {
                Backend* b = reg.get_backend(name);
                if (b == nullptr) continue;
                const int32_t count = b->device_count();
                for (int32_t i = 0; i < count; ++i) {
                    Device d = Device::from_string(name + ":" + std::to_string(i));
                    if (!already_have(d)) devs.push_back(d);
                }
            }
        } catch (const std::exception&) {
            // Best-effort: fall back to the input-device + CPU set already built.
        }
    }

    saved_generator_states_.clear();
    saved_generator_states_.reserve(devs.size());
    for (const auto& d : devs) {
        saved_generator_states_.emplace_back(d, default_generator(d).get_state());
    }
    rng_state_saved_ = true;
}

auto CheckpointFunction::recompute_forward(const std::vector<Variable>& inputs) -> std::vector<Variable> {
    // Check if we have cached outputs
    if (allow_caching_ && has_cached_outputs_) {
        return cached_recompute_outputs_;
    }

    // Phase E (E2): notify the recompute-begin hook before forward_fn_ runs.
    // ZeROStage3Optimizer registers a hook that re-gathers all currently-
    // partitioned parameters so the recomputed forward sees full-shape weights
    // instead of 1-D partition slices.
    auto hooks = get_recompute_hooks();
    if (hooks.on_begin) {
        hooks.on_begin(this);
    }

    // Audit D.3: replay the RNG state captured at original-forward time so
    // any stochastic op inside forward_fn_ (Dropout, BatchNorm noise,
    // multinomial/poisson/etc.) draws the same samples it drew the first
    // time. Save the *current* RNG state first so we can restore it after
    // the recompute and not perturb the outer training loop's stream.
    GlobalRngState prev_global_rng;
    std::vector<std::pair<Device, GeneratorState>> prev_gen_states;
    if (rng_state_saved_) {
        prev_global_rng = save_global_rng_state();
        prev_gen_states.reserve(saved_generator_states_.size());
        for (const auto& [dev, _] : saved_generator_states_) {
            prev_gen_states.emplace_back(dev, default_generator(dev).get_state());
        }
        restore_global_rng_state(saved_global_rng_);
        for (const auto& [dev, st] : saved_generator_states_) {
            default_generator(dev).set_state(st);
        }
    }

    // Recompute forward function with gradient tracking enabled
    auto outputs = forward_fn_(inputs);

    // Audit D.3: restore the outer RNG state so the recompute is invisible
    // to whatever code runs after backward() returns.
    if (rng_state_saved_) {
        restore_global_rng_state(prev_global_rng);
        for (const auto& [dev, st] : prev_gen_states) {
            default_generator(dev).set_state(st);
        }
    }

    // Phase E (E2): notify the recompute-end hook so the consumer can release
    // whatever it gathered in on_begin (Stage-3 frees the gathered buffers
    // and restores partitions).
    if (hooks.on_end) {
        hooks.on_end(this);
    }

    // Verify determinism on first recomputation
    if (verify_ && !verification_done_) {
        verification_done_ = true;
        // Create detached inputs for verification run
        std::vector<Variable> verify_inputs;
        verify_inputs.reserve(inputs.size());
        for (const auto& inp : inputs) {
            verify_inputs.emplace_back(inp.tensor(), false);
        }
        // Audit D.3: replay saved RNG so verify_outputs uses the same draws
        // as the just-executed recompute; otherwise determinism check would
        // false-positive on legitimately deterministic-after-replay code.
        GlobalRngState verify_prev_global;
        std::vector<std::pair<Device, GeneratorState>> verify_prev_gens;
        if (rng_state_saved_) {
            verify_prev_global = save_global_rng_state();
            verify_prev_gens.reserve(saved_generator_states_.size());
            for (const auto& [dev, _] : saved_generator_states_) {
                verify_prev_gens.emplace_back(dev, default_generator(dev).get_state());
            }
            restore_global_rng_state(saved_global_rng_);
            for (const auto& [dev, st] : saved_generator_states_) {
                default_generator(dev).set_state(st);
            }
        }
        auto verify_outputs = forward_fn_(verify_inputs);
        if (rng_state_saved_) {
            restore_global_rng_state(verify_prev_global);
            for (const auto& [dev, st] : verify_prev_gens) {
                default_generator(dev).set_state(st);
            }
        }

        if (verify_outputs.size() != outputs.size()) {
            // Audit I.4: unified logger so TENZOR_LOG_LEVEL filter applies.
            TENZOR_LOG_WARN("Checkpoint: non-deterministic function detected "
                            "(different output count: {} vs {})",
                            outputs.size(), verify_outputs.size());
        } else {
            for (size_t i = 0; i < outputs.size(); ++i) {
                auto diff = tenzor::abs(outputs[i].tensor().to(DType::Float64) -
                            verify_outputs[i].tensor().to(DType::Float64));
                auto max_diff_t = tenzor::max(diff);
                // max_diff_t lives on the (possibly GPU) device of the checkpointed
                // outputs; copy to host before dereferencing its data pointer.
                auto max_diff_host = max_diff_t.to(Device::cpu());
                double max_diff = max_diff_host.data<double>()[0];
                if (max_diff > 1e-6) {
                    // Audit I.4: unified logger.
                    TENZOR_LOG_WARN("Checkpoint: non-deterministic function detected "
                                    "(output {} max diff: {}). Gradients may be incorrect.",
                                    i, max_diff);
                }
            }
        }
    }

    // Cache outputs if allowed
    if (allow_caching_) {
        cached_recompute_outputs_ = outputs;
        has_cached_outputs_ = true;
    }

    return outputs;
}

auto CheckpointFunction::estimate_memory(const std::vector<Variable>& vars) const -> size_t {
    size_t total = 0;
    for (const auto& var : vars) {
        const auto& tensor = var.tensor();
        total += tensor.numel() * tensor.dtype_size();
    }
    return total;
}

// ============================================================================
// Checkpoint Free Functions
// ============================================================================

// Internal shared_ptr-based implementation
// This is the core implementation that eliminates all Variable copying
static auto checkpoint_impl_shared(
    std::function<std::vector<Variable>(const std::vector<Variable>&)> fn,
    std::vector<std::shared_ptr<Variable>> input_ptrs,
    const std::vector<Variable*>& original_inputs
) -> std::vector<Variable> {
    // Check if any input requires gradients
    bool requires_grad = false;
    for (const auto& ptr : input_ptrs) {
        if (ptr->requires_grad()) {
            requires_grad = true;
            break;
        }
    }

    if (!is_checkpoint_enabled() || !requires_grad || !is_grad_enabled()) {
        // If checkpointing is disabled or no gradients needed, execute function normally
        std::vector<Variable> inputs_for_call;
        inputs_for_call.reserve(input_ptrs.size());
        for (const auto& ptr : input_ptrs) {
            inputs_for_call.push_back(*ptr);
        }
        return fn(inputs_for_call);
    }

    // Create checkpoint function with the user's function directly (no wrapping!)
    // The shared_ptrs keep Variables alive long enough to extract Tensors
    auto checkpoint_fn = std::make_shared<CheckpointFunction>(fn, true);

    // Save input tensors for recomputation
    std::vector<Tensor> input_tensors;
    input_tensors.reserve(input_ptrs.size());
    std::vector<Device> input_devices;
    input_devices.reserve(input_ptrs.size());
    for (const auto& ptr : input_ptrs) {
        input_tensors.push_back(ptr->tensor());
        input_devices.push_back(ptr->tensor().device());
    }
    checkpoint_fn->save_for_backward(std::move(input_tensors));

    // Audit D.3: snapshot the RNG state *before* running the original
    // forward so that the recompute on backward sees the same random draws
    // for any stochastic op inside fn (Dropout, BatchNorm noise,
    // multinomial sampling, etc.). Must precede the fn() call below.
    checkpoint_fn->save_rng_state(input_devices);

    // Set up backward graph connections to original inputs
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.reserve(input_ptrs.size());
    for (const auto& ptr : input_ptrs) {
        next_funcs.push_back(ptr->grad_fn());  // nullptr if input is leaf
    }
    checkpoint_fn->set_next_functions(next_funcs);

    // Store Variables on heap to keep them alive
    std::vector<Variable> inputs_for_storage;
    inputs_for_storage.reserve(input_ptrs.size());
    for (const auto& ptr : input_ptrs) {
        inputs_for_storage.push_back(*ptr);
    }
    checkpoint_fn->store_input_copies(inputs_for_storage);

    // Convert raw pointers to Variables for set_input_variables
    auto input_copy_ptrs = checkpoint_fn->get_input_copy_pointers();
    std::vector<Variable> input_vars;
    input_vars.reserve(input_copy_ptrs.size());
    for (auto* ptr : input_copy_ptrs) {
        input_vars.push_back(*ptr);
    }
    checkpoint_fn->set_input_variables(input_vars);

    // Store pointers to original leaf Variables for gradient accumulation
    if (!original_inputs.empty()) {
        checkpoint_fn->store_original_inputs(original_inputs);
    } else {
        // No originals provided - use the heap-allocated copies
        checkpoint_fn->store_original_inputs(checkpoint_fn->get_input_copy_pointers());
    }

    // Execute forward function using the user's function directly
    std::vector<Variable> inputs_for_execution;
    inputs_for_execution.reserve(input_ptrs.size());
    for (const auto& ptr : input_ptrs) {
        inputs_for_execution.push_back(*ptr);
    }

    // Whether gradients should flow through this checkpoint (decided BEFORE the
    // NoGradGuard flips the thread-local grad state).
    const bool grad_enabled_outer = is_grad_enabled();
    bool any_input_requires_grad = false;
    for (const auto& ptr : input_ptrs) {
        if (ptr->requires_grad()) { any_input_requires_grad = true; break; }
    }

    // Run the checkpointed forward WITHOUT building an autograd graph — that is
    // the whole point of activation checkpointing (trade compute for memory).
    // Previously the forward ran with grad enabled, building a full internal
    // graph that was immediately discarded when checkpoint_fn was attached
    // below, so peak forward memory equalled an un-checkpointed forward. The
    // gradient is instead produced by re-running the forward during backward
    // (checkpoint_fn), so no internal graph is needed here.
    std::vector<Variable> outputs;
    {
        NoGradGuard no_grad;
        outputs = fn(inputs_for_execution);
    }

    // Attach the checkpoint grad_fn so gradients flow via recomputation. Under
    // NoGrad the outputs don't track grad, so re-mark the grad-capable
    // (floating/complex) outputs as requiring grad when the checkpoint should
    // participate in autograd (mirrors PyTorch's non-reentrant checkpoint).
    if (any_input_requires_grad && grad_enabled_outer) {
        for (auto& output : outputs) {
            if (output.tensor().is_floating_point() || output.tensor().is_complex()) {
                Variable tracked(output.tensor(), /*requires_grad=*/true);
                tracked.set_grad_fn(checkpoint_fn);
                output = tracked;
            }
        }
    }

    // Update statistics
    auto& stats = get_checkpoint_stats();
    stats.num_checkpoints++;
    size_t estimated_memory = 0;
    for (const auto& ptr : input_ptrs) {
        const auto& tensor = ptr->tensor();
        estimated_memory += tensor.numel() * tensor.dtype_size();
    }
    stats.saved_memory_bytes += estimated_memory;

    return outputs;
}

// Public API wrappers that convert to shared_ptr-based implementation
auto checkpoint(
    std::function<std::vector<Variable>(const std::vector<Variable>&)> fn,
    const std::vector<Variable>& inputs  // Changed to const reference to avoid unnecessary copy
) -> std::vector<Variable> {
    // Convert inputs to shared_ptrs immediately
    std::vector<std::shared_ptr<Variable>> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const auto& input : inputs) {
        input_ptrs.push_back(std::make_shared<Variable>(input));
    }

    // Use empty original_inputs
    static const std::vector<Variable*> empty_originals;
    return checkpoint_impl_shared(fn, input_ptrs, empty_originals);
}

auto checkpoint_with_originals(
    std::function<std::vector<Variable>(const std::vector<Variable>&)> fn,
    const std::vector<Variable>& inputs,  // Changed to const reference to avoid unnecessary copy
    const std::vector<Variable*>& original_inputs  // Also const reference for consistency
) -> std::vector<Variable> {
    // Convert inputs to shared_ptrs immediately
    std::vector<std::shared_ptr<Variable>> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const auto& input : inputs) {
        input_ptrs.push_back(std::make_shared<Variable>(input));
    }

    return checkpoint_impl_shared(fn, input_ptrs, original_inputs);
}

auto checkpoint(
    std::function<Variable(const Variable&)> fn,
    const Variable& input
) -> Variable {
    // Wrap single-input function as multi-input
    auto multi_fn = [fn](const std::vector<Variable>& inputs) -> std::vector<Variable> {
        if (inputs.size() != 1) {
            throw std::runtime_error("Checkpoint wrapper: expected single input");
        }
        return {fn(inputs[0])};
    };

    // CRITICAL: Create vector with input to avoid extra copy
    std::vector<Variable> inputs_vec;
    inputs_vec.push_back(input);
    auto outputs = checkpoint(multi_fn, inputs_vec);

    if (outputs.size() != 1) {
        throw std::runtime_error("Checkpoint wrapper: expected single output");
    }

    return outputs[0];
}

auto checkpoint_with_original(
    std::function<Variable(const Variable&)> fn,
    const Variable& input,
    Variable* original_input
) -> Variable {
    // Wrap single-input function as multi-input
    auto multi_fn = [fn](const std::vector<Variable>& inputs) -> std::vector<Variable> {
        if (inputs.size() != 1) {
            throw std::runtime_error("Checkpoint wrapper: expected single input");
        }
        return {fn(inputs[0])};
    };

    // Create vectors - this does copy Variable but necessary for API compatibility
    std::vector<Variable> inputs_vec;
    inputs_vec.push_back(input);
    std::vector<Variable*> originals_vec;
    originals_vec.push_back(original_input);
    auto outputs = checkpoint_with_originals(multi_fn, inputs_vec, originals_vec);

    if (outputs.size() != 1) {
        throw std::runtime_error("Checkpoint wrapper: expected single output");
    }

    return outputs[0];
}

// ============================================================================
// CheckpointContext Implementation
// ============================================================================


CheckpointContext::CheckpointContext(bool enabled)
    : enabled_(enabled),
      prev_enabled_(is_checkpoint_enabled()),
      initial_stats_(get_checkpoint_stats()) {
    set_checkpoint_enabled(enabled);
}

CheckpointContext::~CheckpointContext() {
    // Restore previous state
    set_checkpoint_enabled(prev_enabled_);
}

auto CheckpointContext::get_stats() const -> CheckpointStats {
    // Return difference from initial stats
    auto current_stats = get_checkpoint_stats();
    CheckpointStats diff;

    diff.num_checkpoints = current_stats.num_checkpoints - initial_stats_.num_checkpoints;
    diff.num_recomputations = current_stats.num_recomputations - initial_stats_.num_recomputations;
    diff.saved_memory_bytes = current_stats.saved_memory_bytes - initial_stats_.saved_memory_bytes;
    diff.peak_memory_bytes = current_stats.peak_memory_bytes;
    diff.total_recompute_time_ms = current_stats.total_recompute_time_ms - initial_stats_.total_recompute_time_ms;

    return diff;
}

// ============================================================================
// MemoryTracker Implementation
// ============================================================================

thread_local bool MemoryTracker::tracking_enabled_ = false;
thread_local size_t MemoryTracker::current_memory_ = 0;
thread_local size_t MemoryTracker::peak_memory_ = 0;

auto MemoryTracker::start_tracking() -> void {
    tracking_enabled_ = true;
    current_memory_ = 0;
    peak_memory_ = 0;
}

auto MemoryTracker::stop_tracking() -> void {
    tracking_enabled_ = false;
}

auto MemoryTracker::current_memory() -> size_t {
    return current_memory_;
}

auto MemoryTracker::peak_memory() -> size_t {
    return peak_memory_;
}

auto MemoryTracker::reset() -> void {
    current_memory_ = 0;
    peak_memory_ = 0;
}

// ============================================================================
// CheckpointSegment Implementation
// ============================================================================

CheckpointSegment::CheckpointSegment(std::string name, int level)
    : name_(std::move(name)),
      nesting_level_(level) {}

auto CheckpointSegment::execute(
    std::function<std::vector<Variable>(const std::vector<Variable>&)> fn,
    std::vector<Variable> inputs
) -> std::vector<Variable> {
    // Execute function with checkpointing
    return checkpoint(fn, inputs);
}

// ============================================================================
// AutoCheckpointPolicy Implementation
// ============================================================================

auto AutoCheckpointPolicy::should_checkpoint(int index, int total) const -> bool {
    switch (strategy_) {
        case CheckpointStrategy::None:
            return false;
        case CheckpointStrategy::EveryN:
            return every_n_ > 0 && (index % every_n_) == 0 && index > 0;
        case CheckpointStrategy::SqrtN: {
            if (total <= 1) return false;
            int interval = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(total))));
            return (index % interval) == 0 && index > 0;
        }
        case CheckpointStrategy::MemoryBudget: {
            // Distribute checkpoints so each segment stays within budget.
            // Estimate: place a checkpoint every N submodules, where N is
            // chosen so each segment's parameter memory fits the budget.
            // This is computed in apply() and stored as every_n_.
            // Fall through to EveryN logic with the computed interval.
            return every_n_ > 0 && (index % every_n_) == 0 && index > 0;
        }
    }
    return false;
}

auto AutoCheckpointPolicy::apply(nn::Module& module) -> void {
    // Remove any existing hooks first
    remove(module);

    if (strategy_ == CheckpointStrategy::None) return;

    // Collect direct submodules
    auto& submodules = module.get_submodules();
    int total = static_cast<int>(submodules.size());

    // For MemoryBudget: compute every_n_ based on parameter sizes
    if (strategy_ == CheckpointStrategy::MemoryBudget && memory_budget_bytes_ > 0) {
        // Sum total parameter bytes across all submodules
        size_t total_param_bytes = 0;
        for (auto& [name, sub] : submodules) {
            for (auto& p : sub->parameters()) {
                total_param_bytes += static_cast<size_t>(p->tensor().numel()) *
                                    dtype_size(p->tensor().dtype());
            }
        }
        // Compute how many segments we need so each fits in budget
        if (total_param_bytes > 0 && total > 0) {
            int num_segments = std::max(1, static_cast<int>(
                (total_param_bytes + memory_budget_bytes_ - 1) / memory_budget_bytes_));
            every_n_ = std::max(1, total / num_segments);
        } else {
            every_n_ = total; // no params -> no checkpointing needed
        }
    }

    int index = 0;
    for (auto& [name, submodule] : submodules) {
        if (should_checkpoint(index, total)) {
            // Register a forward pre-hook that wraps the submodule's forward
            // with checkpointing by enabling the checkpoint context
            auto hook_id = submodule->register_forward_pre_hook(
                [](nn::Module*, [[maybe_unused]] const Variable& input) {
                    set_checkpoint_enabled(true);
                }
            );
            registered_hooks_.emplace_back(submodule.get(), hook_id);

            auto post_hook_id = submodule->register_forward_post_hook(
                [](nn::Module*, const Variable&, const Variable&) {
                    set_checkpoint_enabled(false);
                }
            );
            registered_hooks_.emplace_back(submodule.get(), post_hook_id);
        }
        ++index;
    }
}

auto AutoCheckpointPolicy::remove(nn::Module& /* module */) -> void {
    // Audit D.3: actually unregister hooks instead of leaving them resident
    // on the modules as silent no-ops (which would still toggle
    // set_checkpoint_enabled on later forwards even after the policy was
    // dropped). Module::remove_hook handles ID lookup across both the
    // single- and multi-input hook maps.
    for (auto& [mod, hook_id] : registered_hooks_) {
        if (mod) {
            mod->remove_hook(hook_id);
        }
    }
    registered_hooks_.clear();
}

auto enable_auto_checkpoint(
    nn::Module& module,
    CheckpointStrategy strategy,
    int every_n,
    size_t memory_budget_bytes) -> std::shared_ptr<AutoCheckpointPolicy>
{
    auto policy = std::make_shared<AutoCheckpointPolicy>(
        strategy, every_n, memory_budget_bytes);
    policy->apply(module);
    return policy;
}

// Phase E (E2): global recompute hooks registry. Single owner expected
// (the active Stage-3 optimizer). Plain global with mutex; the alternative
// (passing the optimizer through every checkpoint() lambda capture) would
// break the existing std::function-based API.
namespace {
    std::mutex& recompute_hooks_mutex() {
        static std::mutex m;
        return m;
    }
    RecomputeHooks& recompute_hooks_storage() {
        static RecomputeHooks hooks;
        return hooks;
    }
}

auto set_recompute_hooks(RecomputeHooks hooks) -> RecomputeHooks {
    std::lock_guard<std::mutex> lock(recompute_hooks_mutex());
    RecomputeHooks prev = recompute_hooks_storage();
    recompute_hooks_storage() = std::move(hooks);
    return prev;
}

auto get_recompute_hooks() -> RecomputeHooks {
    // Return a COPY taken under the same mutex set_recompute_hooks() uses.
    // Returning an unlocked reference to the global std::functions raced with a
    // concurrent set_recompute_hooks() (e.g. optimizer teardown): the caller
    // could invoke on_begin while its target was being reassigned/destroyed
    // underneath it (UB). A by-value snapshot keeps the target alive for the
    // duration of the call regardless of subsequent mutations.
    std::lock_guard<std::mutex> lock(recompute_hooks_mutex());
    return recompute_hooks_storage();
}

} // namespace autograd
} // namespace tenzor
