/**
 * @file compile.cpp
 * @brief Implementation of automatic graph capture and compilation
 */

#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/compiler.hpp"
#include "tenzor/jit/autotune.hpp"  // R1-11: AutotuneModeGuard
#include "tenzor/backend/dispatch_interceptor.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
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

// M3/JIT-055: resolve the IREE HAL device ordinal for `target` given the
// input tensor's device. Returns `in_dev.index` only when in_dev's backend
// family actually matches `target` (cuda/rocm/vulkan-spirv/llvm-cpu); an
// explicit cross-family target override (JIT-032, e.g. target="rocm" on a
// cuda:1 input) must not carry the wrong family's device ordinal over to the
// HAL device URI -- that would pair an arch correctly derived for e.g. ROCm
// device 0 with HAL device index 1, silently targeting the wrong physical
// GPU. Mirrors the family-aware fallback rocm_ordinal/cuda_dev already use
// for arch derivation. Pure/host-only (no IREE headers needed) so it's
// compiled and testable unconditionally, independent of TENZOR_HAS_MLIR_JIT.
auto resolve_hal_ordinal(const std::string& target, const Device& in_dev) -> int {
    const bool family_matches =
        (target == "llvm-cpu" && in_dev.type == Device::Type::CPU) ||
        (target == "cuda" && in_dev.type == Device::Type::CUDA) ||
        (target == "rocm" && in_dev.type == Device::Type::ROCm) ||
        (target == "vulkan-spirv" && in_dev.type == Device::Type::Vulkan);
    if (!family_matches) return 0;
    return in_dev.index < 0 ? 0 : in_dev.index;
}

// JIT-R108/JIT-R131: pure vendor+product-name -> --iree-vulkan-target string
// classifier, extracted from detect_vulkan_iree_target() (which does the
// actual hardware query) purely so it can be unit tested directly with
// synthetic vendor/name strings, without needing every GPU generation's real
// hardware on hand. `vendor_lower`/`name_lower` must already be lowercased by
// the caller (matching detect_vulkan_iree_target's own preprocessing).
auto classify_vulkan_target(const std::string& vendor_lower,
                            const std::string& name_lower) -> std::string {
    if (vendor_lower.find("nvidia") != std::string::npos) {
        // JIT-R108: a single hardcoded "ampere" for every NVIDIA GPU was
        // confirmed (via a standalone iree-run-module repro built while
        // investigating JIT-R106) to fail to even initialize
        // (VK_ERROR_INITIALIZATION_FAILED / SIGSEGV) on Blackwell
        // (RTX 50-series) hardware, where "ada" compiles and runs cleanly
        // instead. Parse the reported product name for a generation marker
        // rather than guessing a single fixed value. "ada"/"ampere" are the
        // only two target strings empirically verified against this IREE
        // build; unrecognized/future NVIDIA generations fall back to the
        // generic default profile below rather than a confidently wrong
        // specific target — this IREE build does not yet recognize a
        // dedicated "blackwell" target, so Blackwell-generation cards use
        // "ada" as the newest available, empirically-working profile.
        if (name_lower.find("rtx 50") != std::string::npos ||
            name_lower.find("rtx pro 6000") != std::string::npos) return "ada";
        if (name_lower.find("rtx 40") != std::string::npos) return "ada";
        if (name_lower.find("rtx 30") != std::string::npos ||
            name_lower.find("a100") != std::string::npos) return "ampere";
        return {};
    }
    if (vendor_lower.find("amd") != std::string::npos ||
        vendor_lower.find("radeon") != std::string::npos) {
        // JIT-R131: a single hardcoded "rdna3" for every AMD GPU would be as
        // stale for RDNA4 (RX 9000-series, which predates this fix) as the
        // pre-fix NVIDIA "ampere"-for-everything default was for Blackwell.
        // Verified against this IREE build: "rdna1"/"rdna2"/"rdna3"/"rdna4"
        // all compile as valid, distinct --iree-vulkan-target profiles (a
        // genuinely unrecognized string is rejected outright with "Unknown
        // Vulkan target"). Parse the product name for a generation marker,
        // mirroring the NVIDIA pattern above.
        //
        // JIT-R112 fix: a plain "rx 5" substring collides with the pre-RDNA
        // Polaris/GCN4 "RX 500-series" (RX 580/570/560/550/590 -- a 3-digit
        // marketing number) as well as the intended RDNA1 "RX 5000-series"
        // (RX 5700/5600/5500/5300 XT -- a 4-digit number). Require a full
        // 4-digit number after the generation digit so Polaris/GCN4 cards
        // don't get silently misrouted to an RDNA1 target they don't
        // support. RDNA2/3/4's RX 6000/7000/9000-series numbers have no
        // legacy 3-digit family to collide with, but the same digit-count
        // check is applied uniformly for consistency.
        auto matches_rdna_rx_series = [&](char gen_digit) -> bool {
            const std::string needle = std::string("rx ") + gen_digit;
            const auto pos = name_lower.find(needle);
            if (pos == std::string::npos) return false;
            std::size_t p = pos + needle.size();
            int extra_digits = 0;
            while (p < name_lower.size() &&
                   name_lower[p] >= '0' && name_lower[p] <= '9') {
                ++extra_digits;
                ++p;
            }
            // gen_digit itself is the 1st digit; need >=3 more for a full
            // 4-digit RDNA-generation number (e.g. "5" + "700" = "5700").
            return extra_digits >= 3;
        };
        if (matches_rdna_rx_series('9')) return "rdna4";
        if (matches_rdna_rx_series('7')) return "rdna3";
        if (matches_rdna_rx_series('6')) return "rdna2";
        if (matches_rdna_rx_series('5')) return "rdna1";
        // CDNA/Instinct datacenter GPUs (MI100/MI200/MI300) are an entirely
        // different architecture family from consumer RDNA -- plausible for
        // Tenzor ML workloads, and must never silently receive a consumer
        // RDNA target guess.
        if (name_lower.find("instinct") != std::string::npos ||
            name_lower.find(" mi100") != std::string::npos ||
            name_lower.find(" mi2") != std::string::npos ||
            name_lower.find(" mi3") != std::string::npos) {
            return {};
        }
        // No RX-series or Instinct marketing number matched at all -- this
        // is the case an integrated APU (e.g. "AMD Radeon 890M Graphics",
        // which carries no RX number) falls into. Fall back to "rdna3",
        // empirically verified working on this review's actual RDNA3.5
        // (gfx1150) integrated GPU, so it isn't left with no signal at all.
        // This default intentionally does NOT apply to the Polaris/GCN4 or
        // CDNA/Instinct cases above, nor to any OTHER unrecognized-but-RX-
        // numbered card (handled just below) -- only to genuinely RX/
        // Instinct-number-less AMD hardware.
        if (name_lower.find("rx ") == std::string::npos) {
            return "rdna3";
        }
        // An "rx" token IS present (so this isn't the no-signal iGPU case
        // above) but didn't match any known 4-digit RDNA generation -- e.g.
        // the 3-digit Polaris/GCN4 "RX 500-series", or any other RX-numbered
        // but unrecognized generation. Prefer the safe empty default over a
        // confidently wrong guess, matching the NVIDIA/Intel pattern.
        return {};
    }
    if (vendor_lower.find("intel") != std::string::npos) {
        // JIT-R131: "arc" -- the previously hardcoded value -- is not merely
        // stale, it is not a valid --iree-vulkan-target string in this IREE
        // build at all: verified it CRASHES iree-compile outright (SIGABRT
        // inside ExecutableTargetAttr::getBackend(), exit code 245, no vmfb
        // produced), the same way any genuinely unrecognized target does but
        // with an actual crash instead of a clean diagnostic. Every other
        // plausible Intel Vulkan target string tried against this IREE
        // build (arc/xe/xe2/battlemage/alchemist/dg2/a770/a750/a380) was
        // rejected identically — this IREE build has no working
        // Intel-specific Vulkan target profile at all. Return the safe
        // default (empty) so an Intel Vulkan compile falls through to
        // IREE's conservative default SPIR-V profile (correct for F32;
        // F16/BF16 already refuse with a clear "shaderFloat16" error rather
        // than silently compiling wrong, per the guard on the
        // empty-vulkan_arch path in iree_compile.cpp) instead of a target
        // string guaranteed to crash compilation outright.
        return {};
    }
    return {};
}

#ifdef TENZOR_HAS_MLIR_JIT
// JIT-R109: a cached invoker is paired with the leaf-argument layout the
// compiled module actually expects (GraphToMLIR::leaf_args(), captured at
// the trace that produced it) — parameter/buffer leaves are extra @main
// arguments, not baked constants, so every invocation (cache hit or miss)
// must gather their CURRENT values, in this exact order, and append them
// after the regular inputs.
struct MlirCacheEntry {
    std::shared_ptr<::tenzor::jit::mlir_jit::IreeInvoker> invoker;
    std::vector<::tenzor::jit::mlir_jit::ParamBufferLeafArg> leaf_args;
};

struct MlirInvokerCache {
    // JIT-R014: shared_ptr (not unique_ptr) so the cache-hit fast path can
    // copy a handle out under mutex_ and invoke it AFTER releasing the lock
    // — mirrors cache_'s (the nvrtc path's) identical shared_ptr<CompiledModule>
    // pattern, fixing the same "mutex_ held across the full invocation,
    // serializing every concurrent call" anti-pattern already fixed there.
    std::unordered_map<std::string, MlirCacheEntry> invokers;
};

// Gather the CURRENT tensor value for each leaf argument, placed on
// target_device, in leaf_args order (JIT-R109). Pure/host-only; called on
// every mlir_invoke_impl invocation (cache hit AND miss) so a captured
// parameter/buffer is never stale.
auto gather_leaf_tensors(
    const std::vector<::tenzor::jit::mlir_jit::ParamBufferLeafArg>& leaf_args,
    const std::vector<std::shared_ptr<Variable>>& parameters,
    const std::vector<std::shared_ptr<Variable>>& buffers,
    const Device& target_device) -> std::vector<Tensor> {
    std::vector<Tensor> out;
    out.reserve(leaf_args.size());
    for (const auto& leaf : leaf_args) {
        const auto& src = leaf.is_buffer ? buffers : parameters;
        if (leaf.index >= src.size() || !src[leaf.index]) {
            throw std::runtime_error(
                "JIT: MLIR-compiled module expects " +
                std::string(leaf.is_buffer ? "buffer" : "parameter") +
                " leaf index " + std::to_string(leaf.index) +
                " but only " + std::to_string(src.size()) +
                (leaf.is_buffer ? " buffers" : " parameters") +
                " are currently declared -- with_parameters()/with_buffers() "
                "must not shrink or reorder relative to the trace that "
                "compiled this module.");
        }
        const Tensor& t = src[leaf.index]->tensor();
        out.push_back(t.device() == target_device ? t : t.to(target_device));
    }
    return out;
}
#else
struct MlirInvokerCache {};
#endif

}  // namespace mlir_detail

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
    // JIT-R109: the compiled MLIR module's extra leaf @main arguments are
    // bound by parameter INDEX (GraphToMLIR::leaf_args()), so a different
    // parameter list (different count/shapes/order) invalidates the cached
    // invoker's argument contract, not just its values -- drop it too.
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

// JIT-R005 fix: mirrors set_parameters()/with_parameters() exactly, for
// non-trainable buffers (BatchNorm/InstanceNorm running_mean_/running_var_).
auto CompiledFunction::set_buffers(
    std::vector<std::shared_ptr<Variable>> buffers) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_ = std::move(buffers);
    cache_.clear();
    // JIT-R109: see set_parameters()'s identical comment -- a different
    // buffer list invalidates the cached invoker's leaf-argument contract.
    if (mlir_cache_) {
        mlir_cache_->invokers.clear();
    }
    warmup_counts_.clear();
}

auto CompiledFunction::with_buffers(
    std::vector<std::shared_ptr<Variable>> buffers) -> CompiledFunction& {
    set_buffers(std::move(buffers));
    return *this;
}

auto CompiledFunction::snapshot_params_buffers() const
    -> std::pair<std::vector<std::shared_ptr<Variable>>,
                std::vector<std::shared_ptr<Variable>>> {
    std::lock_guard<std::mutex> lock(mutex_);
    return {parameters_, buffers_};
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

    // JIT-R019: IREE target/arch/HAL-ordinal (and, on the nvrtc path, the
    // launch device) are derived exclusively from inputs[0]'s device further
    // down this call -- with no check that every input actually shares that
    // device, a multi-argument call with mixed-device inputs would silently
    // compile/dispatch for only the first input's device while other inputs
    // are read from a different one, diverging from eager dispatch (which
    // get_dispatch_device already rejects outright, see fast_dispatch.hpp).
    // Reuse that exact same validation here so the JIT path rejects a
    // mixed-device call at the same point eager would, instead of silently
    // mis-targeting.
    {
        std::vector<Tensor> input_tensors;
        input_tensors.reserve(inputs.size());
        for (const auto& v : inputs) input_tensors.push_back(v.tensor());
        (void)get_dispatch_device(std::span<const Tensor>(input_tensors));
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

    // R1-11: enable autotuned kernel-launch selection (CompiledKernel::launch,
    // src/jit/codegen.cpp) for the duration of this call when the user
    // requested max-autotune mode. Scoped via RAII so every return path below
    // (the cache-hit fast path, incl. its reduce-overhead/CUDA-graph
    // branches, and the cache-miss path further down) is covered without
    // threading a flag through Graph/Node/FusionGroup down to the kernel
    // launch. Two notes on current reach: (1) it is a no-op on the cache-miss
    // path below, which only ever returns the eager trace result (see the
    // JIT-008 comment further down), never invoking a compiled kernel launch;
    // (2) today's ExtendedFusionPass-driven replay executes fused GPU nodes
    // via execute_extended_fused/launch_raw (fixed launch geometry driven by
    // the per-kind cost model in extended_codegen.cpp), not the plain
    // FusionGroup/execute_fused/CompiledKernel::launch path this guard feeds
    // -- launch_raw's geometry is tied to each kernel's generated shared-
    // memory layout and is intentionally NOT autotuned here (that would need
    // per-kind kernel-source verification, not just a launch-geometry
    // change). This guard is still the correct, forward-compatible hook: any
    // direct caller of execute_fused/CompiledKernel::launch (and any future
    // plain-elementwise fusion pass built on FusionGroup) gets real
    // autotuning for free the moment it runs under this scope.
    AutotuneModeGuard autotune_guard(config_.mode == "max-autotune");

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
        // JIT-R120: the entire cache-hit path (below, including its reduce-
        // overhead/CUDA-graph replay/capture sub-branch) used to have no
        // exception safety net at all -- any runtime failure (a ShapeGuard
        // mismatch with no source_module_ to retrace, CUDA OOM, an internal
        // dispatch error) propagated uncaught regardless of config_.strict,
        // silently violating the documented non-strict "compile-or-eager"
        // contract that both the cache-MISS branch below and the mlir
        // backend's cache-hit path (mlir_invoke's C2 wrapper) already honor.
        // fn_ is guaranteed NOT to have run yet on a cache hit (only
        // compiled_module methods are called below), so falling back to
        // fn_(inputs) here can never double-execute a side-effecting
        // closure.
        try {
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
        } catch (const std::exception& e) {
            if (jit_strict_mode_enabled(config_.strict)) throw;
            TENZOR_LOG_WARN(
                "JIT cache-hit replay failed ({} backend, target={}): {}. "
                "Falling back to eager execution.",
                config_.backend, config_.target, e.what());
            return fn_(inputs);
        }
    }

    // Cache miss: trace, compile, and cache. Trace ONCE — fn_() runs a single
    // time under the interceptor and yields the eager result via the out-params
    // below. Running fn_() a second time here would apply a side-effecting
    // closure's effect twice on a cache miss (JIT-008).
    Variable result;
    bool fn_ran = false;

    // JIT-033: the nvrtc backend has no native codegen for CPU/Vulkan/OneAPI/MPS
    // — extended/native codegen (codegen.cpp/extended_codegen.cpp) is gated to
    // CUDA/ROCm only — so the traced graph runs through the interpreter on any
    // of those four device types, correct but unaccelerated and previously
    // silent for all but Vulkan/OneAPI (unlike the MLIR path, which already
    // warns for OneAPI/MPS). Warn once so it is visible for every device type
    // that actually takes this unaccelerated path, not just two of the four.
    if (config_.backend == "nvrtc" && !inputs.empty()) {
        const auto dt = inputs[0].tensor().device().type;
        if (dt == Device::Type::CPU || dt == Device::Type::Vulkan ||
            dt == Device::Type::OneAPI || dt == Device::Type::MPS) {
            // Per-compiled-function (JIT-F038): each such function warns once, not
            // just the first one in the process.
            if (!warned_no_accel_.exchange(true)) {
                TENZOR_LOG_WARN(
                    "JIT nvrtc backend has no GPU codegen on CPU/Vulkan/OneAPI/"
                    "MPS; the traced graph runs via the interpreter (correct but "
                    "not accelerated). Use backend=\"mlir\" for acceleration.");
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
        if (jit_strict_mode_enabled(config_.strict)) {
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
        } else if (config_.fullgraph || jit_strict_mode_enabled(config_.strict)) {
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
            if (jit_strict_mode_enabled(config_.strict)) throw;
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
            if (config_.fullgraph || jit_strict_mode_enabled(config_.strict)) {
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
        if (jit_strict_mode_enabled(config_.strict)) throw;
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
        // JIT-R005 fix: same declaration for non-trainable buffers, so
        // end_trace() classifies a captured buffer (e.g. BatchNorm's
        // running_mean_/running_var_) as a live buffer leaf instead of
        // freezing it as a constant.
        if (!buffers_.empty()) {
            tracer.set_buffers(buffers_);
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
                              const std::vector<std::shared_ptr<Variable>>& parameters,
                              const std::vector<std::shared_ptr<Variable>>& buffers,
                              Variable* out_result = nullptr,
                              bool* out_fn_attempted = nullptr,
                              bool* out_fn_ok = nullptr,
                              bool* out_had_break = nullptr)
    -> std::shared_ptr<Graph> {
    auto& tracer = Tracer::get_instance();
    tracer.start_trace();
    // JIT-R109: declare the closure-captured parameters/buffers BEFORE
    // running the traced forward, mirroring trace_and_compile's (the nvrtc
    // path's) identical declaration -- so end_trace() classifies each
    // captured parameter/buffer as a live leaf (rebound to its current value
    // on every replay) instead of freezing it as an opaque constant. Without
    // this, every MLIR-backend invocation (mlir_invoke_impl, dump_graph,
    // dump_mlir, dump_stablehlo) baked captured parameters/buffers -- e.g. a
    // BatchNorm's running_mean_/running_var_, or even a Linear layer's
    // weight/bias -- as literal constants in the compiled module, silently
    // stale after the first trace no matter how the source Variable changed.
    if (!parameters.empty()) tracer.set_parameters(parameters);
    if (!buffers.empty()) tracer.set_buffers(buffers);
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
    auto [params_snapshot, buffers_snapshot] = snapshot_params_buffers();
    auto graph = trace_single_input_graph(fn_, inputs, params_snapshot, buffers_snapshot);
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
        std::string name = info.name;
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return mlir_detail::classify_vulkan_target(v, name);
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
        if (jit_strict_mode_enabled(config_.strict)) {
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
        ::tenzor::jit::mlir_jit::internal::record_eager_fallback();
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
        if (jit_strict_mode_enabled(config_.strict)) throw;
        if (fn_ok) {
            TENZOR_LOG_WARN(
                "MLIR JIT: compile/run failed (target={}): {}. Reusing the "
                "traced eager result (fn already ran).",
                config_.target, e.what());
            ::tenzor::jit::mlir_jit::internal::record_eager_fallback();
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
        ::tenzor::jit::mlir_jit::internal::record_eager_fallback();
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

    // Cache-hit fast path. JIT-R014: copy the invoker's shared_ptr handle
    // out under the lock, then RELEASE it before calling invoke() — mirrors
    // the nvrtc cache-hit fix above (see that comment for the full
    // rationale). Previously mutex_ was held across the entire IREE
    // invocation, serializing every concurrent call to this
    // backend="mlir" CompiledFunction, unlike the identical nvrtc path.
    std::shared_ptr<mj::IreeInvoker> hit_invoker;
    std::vector<mj::ParamBufferLeafArg> hit_leaf_args;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = mlir_cache_->invokers.find(key);
        if (it != mlir_cache_->invokers.end()) {
            hit_invoker = it->second.invoker;
            hit_leaf_args = it->second.leaf_args;
        } else {
            // Miss. If we've already compiled something else for a different
            // shape, this counts as a retrace (a new shape forces a fresh trace).
            mj::internal::record_cache_miss();
            if (!mlir_cache_->invokers.empty()) {
                mj::internal::record_retrace();
            }
        }
    }
    if (hit_invoker) {
        mj::internal::record_cache_hit();
        // JIT-R109: append the CURRENT value of each parameter/buffer leaf
        // the compiled module expects, in the order it was compiled with --
        // these are extra @main arguments, never baked as constants, so a
        // cache hit must re-gather them on every call the same as a miss.
        if (!hit_leaf_args.empty()) {
            auto [params_snapshot, buffers_snapshot] = snapshot_params_buffers();
            const auto in_dev = inputs[0].tensor().device();
            auto leaf_tensors = mlir_detail::gather_leaf_tensors(
                hit_leaf_args, params_snapshot, buffers_snapshot, in_dev);
            input_tensors.insert(input_tensors.end(),
                                 std::make_move_iterator(leaf_tensors.begin()),
                                 std::make_move_iterator(leaf_tensors.end()));
        }
        auto outs = hit_invoker->invoke(input_tensors);
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

    // Cache miss: trace → lower → compile → load invoker → cache. Thread the
    // out-params so the trace runs fn EXACTLY ONCE; every fallback below reuses
    // that result rather than re-invoking fn_ (JIT-008-class double-exec).
    bool had_break = false;
    auto [params_snapshot, buffers_snapshot] = snapshot_params_buffers();
    auto graph = trace_single_input_graph(fn_, inputs, params_snapshot, buffers_snapshot,
                                          out_result, out_fn_attempted, out_fn_ok, &had_break);
    had_graph_break_.store(had_break);  // JIT-F055: publish for the introspection accessor
    if (!graph || graph->num_nodes() == 0) {
        // Trace produced nothing (graph break in fullgraph mode, or the
        // function only ran ops the interceptor doesn't capture). Either
        // surface a hard error (strict) or log + degrade to eager.
        // fullgraph=True must also error on a graph break here (JIT-F054), not
        // only in strict mode.
        if (jit_strict_mode_enabled(config_.strict) || config_.fullgraph) {
            throw std::runtime_error(
                "CompiledFunction::mlir_invoke: trace produced no graph "
                "(graph break; config.fullgraph / TENZOR_JIT_STRICT).");
        }
        // fn already ran (successfully) during the trace above — reuse its
        // result instead of running it a SECOND time.
        TENZOR_LOG_WARN("MLIR JIT: trace produced no graph; reusing the traced "
                        "eager result for this invocation.");
        ::tenzor::jit::mlir_jit::internal::record_eager_fallback();
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
    // JIT-R109: the graph's parameter/buffer leaves (if any) were bound as
    // extra @main arguments by lower() above, in this order -- captured here
    // so the invoke call below (and the cached entry) know to append their
    // current values after the regular inputs on every call.
    const std::vector<mj::ParamBufferLeafArg> leaf_args = lowerer.leaf_args();

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
    // M3/JIT-055: thread the input's device ordinal so the IREE HAL device
    // matches the requested ordinal (cuda:1 runs on GPU 1, not the driver
    // default) -- family-guarded, see resolve_hal_ordinal's doc comment.
    const int hal_ordinal = mlir_detail::resolve_hal_ordinal(opts.target, in_dev);
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
    // JIT-R109: append the current value of each parameter/buffer leaf lower()
    // bound as an extra @main argument, in that exact order (params_snapshot/
    // buffers_snapshot were captured before the trace above, so they reflect
    // the same declaration the graph was traced against).
    if (!leaf_args.empty()) {
        auto leaf_tensors = mlir_detail::gather_leaf_tensors(
            leaf_args, params_snapshot, buffers_snapshot, in_dev);
        input_tensors.insert(input_tensors.end(),
                             std::make_move_iterator(leaf_tensors.begin()),
                             std::make_move_iterator(leaf_tensors.end()));
    }
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
            mlir_cache_->invokers.emplace(
                key, mlir_detail::MlirCacheEntry{std::move(invoker), leaf_args});
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
    auto [params_snapshot, buffers_snapshot] = snapshot_params_buffers();
    auto graph = trace_single_input_graph(fn_, inputs, params_snapshot, buffers_snapshot);
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
    auto [params_snapshot, buffers_snapshot] = snapshot_params_buffers();
    auto graph = trace_single_input_graph(fn_, inputs, params_snapshot, buffers_snapshot);
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
