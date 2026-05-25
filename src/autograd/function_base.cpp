#include "tenzor/autograd/function.hpp"
#include <cassert>
#include "tenzor/autograd/ops.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/log.hpp"
#include "tenzor/utils/safe_math.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <typeinfo>
#include <unordered_set>
#ifdef __GNUC__
#include <cxxabi.h>
#include <cstdlib>
#endif

namespace tenzor {

// Global unique ID counter for Function instances. Relaxed ordering is
// sufficient — atomicity alone guarantees uniqueness.
std::atomic<uint64_t> Function::next_id_{1};

// Higher-order gradient fallback mode. Programmatic setting takes precedence
// over the TENZOR_HIGHER_ORDER_GRAD env var.
static std::atomic<int> g_higher_order_mode{-1}; // -1 = not set (check env var)

// Counter for higher-order gradient disconnections (Warn mode).
static std::atomic<uint64_t> g_higher_order_disconnection_count{0};

void set_higher_order_grad_mode(HigherOrderGradMode mode) {
    g_higher_order_mode.store(static_cast<int>(mode), std::memory_order_relaxed);
}

auto get_higher_order_grad_mode() -> HigherOrderGradMode {
    int mode = g_higher_order_mode.load(std::memory_order_relaxed);
    if (mode >= 0) return static_cast<HigherOrderGradMode>(mode);

    // Fall back to env var for backward compatibility
    static int env_mode = []() {
        if (const char* env = std::getenv("TENZOR_HIGHER_ORDER_GRAD")) {
            if (std::string(env) == "warn") return static_cast<int>(HigherOrderGradMode::Warn);
        }
        return static_cast<int>(HigherOrderGradMode::Error);
    }();
    return static_cast<HigherOrderGradMode>(env_mode);
}

auto higher_order_disconnection_count() -> uint64_t {
    return g_higher_order_disconnection_count.load(std::memory_order_relaxed);
}

void reset_higher_order_disconnection_count() {
    g_higher_order_disconnection_count.store(0, std::memory_order_relaxed);
}

namespace detail {
void increment_higher_order_disconnection_count() {
    g_higher_order_disconnection_count.fetch_add(1, std::memory_order_relaxed);
}
} // namespace detail

// Per-thread activation offload flag (thread-local instead of global atomic
// so one thread enabling offload doesn't affect other threads' backward passes)
static thread_local bool g_activation_offload_enabled = false;

void set_activation_offload(bool enabled) {
    g_activation_offload_enabled = enabled;
}

bool activation_offload_enabled() {
    return g_activation_offload_enabled;
}

auto Function::should_offload(const Tensor& t) const -> bool {
    // Check device - only offload GPU tensors
    if (t.device().type == Device::Type::CPU) return false;

    // Check per-function policy
    switch (offload_policy_) {
        case OffloadPolicy::Always: break;  // Always offload (if GPU)
        case OffloadPolicy::Never: return false;
        case OffloadPolicy::Inherit:
            if (!activation_offload_enabled()) return false;
            break;
    }

    // Check size threshold
    if (offload_min_bytes_ > 0) {
        size_t tensor_bytes = static_cast<size_t>(t.numel()) * dtype_size(t.dtype());
        if (tensor_bytes < offload_min_bytes_) return false;
    }

    return true;
}

auto Function::set_next_functions(std::vector<std::shared_ptr<Function>> funcs) -> void {
    next_functions_ = std::move(funcs);
}

auto Function::next_functions() const -> const std::vector<std::shared_ptr<Function>>& {
    return next_functions_;
}

auto Function::set_input_variables(std::vector<Variable> inputs) -> void {
    input_variables_ = std::move(inputs);
}

auto Function::input_variables() const -> const std::vector<Variable>& {
    return input_variables_;
}

auto Function::input_variables() -> std::vector<Variable>& {
    return input_variables_;
}

auto Function::save_for_backward(std::vector<Tensor> tensors) -> void {
    // Per-tensor offload check first (respects per-function policy +
    // size threshold). When a tensor is offloaded we lose access to the
    // original GPU tensor, so version tracking on the POST-offload CPU
    // copy is what we actually store. For non-offloaded tensors, the
    // version recorded below is the live one and the existing in-place
    // detection still applies.
    // V.5: record the source device per tensor so multi-device saves
    // (rare but legal — e.g. a custom Function that captures activations
    // from two different GPUs) round-trip correctly. The legacy single
    // `offloaded_device_` was only set on the first iteration, so any
    // subsequent tensor from a different device would be reloaded to the
    // *first* tensor's device, silently relocating it.
    offloaded_devices_.clear();
    offloaded_devices_.reserve(tensors.size());
    if (!tensors.empty()) {
        bool any_offloaded = false;
        for (auto& t : tensors) {
            if (should_offload(t)) {
                Device src = t.device();
                offloaded_devices_.push_back(src);
                if (!any_offloaded) {
                    offloaded_device_ = src;  // legacy field — first offload source
                    any_offloaded = true;
                }
                t = t.to(Device::cpu());
            } else {
                // Not offloaded — record CPU as a no-op sentinel so the
                // index stays aligned with saved_tensors_.
                offloaded_devices_.push_back(Device::cpu());
            }
        }
        if (any_offloaded) {
            tensors_offloaded_.store(true, std::memory_order_release);
        }
    }

    // Record version counters for in-place modification detection.
    // For offloaded tensors these are versions of private CPU copies
    // the Function owns — no external code has a reference, so the
    // version should never change until we reload.
    saved_versions_.clear();
    saved_versions_.reserve(tensors.size());
    saved_view_base_versions_.clear();
    saved_view_base_versions_.reserve(tensors.size());
    for (const auto& t : tensors) {
        saved_versions_.push_back(t.version());
        // Also track view base version for view safety detection.
        // Offloaded tensors are fresh copies with no view base, so this
        // stays 0 for them — which is the correct "not a view" marker.
        if (t.is_view() && t._view_base()) {
            saved_view_base_versions_.push_back(t._view_base()->version());
        } else {
            saved_view_base_versions_.push_back(0);
        }
    }

    saved_tensors_ = std::move(tensors);
}

auto Function::saved_tensors() const -> const std::vector<Tensor>& {
    if (tensors_offloaded_.load(std::memory_order_acquire)) {
        // Take the offload mutex, re-check under the lock (a concurrent
        // reload may already have completed), and reload without taking
        // the mutex again — reload_saved_tensors_locked does the work
        // assuming the caller holds offload_mutex_. Previously this
        // called the public reload_saved_tensors() which took the same
        // std::mutex a second time and deadlocked on the same thread.
        std::lock_guard lock(offload_mutex_);
        if (tensors_offloaded_.load(std::memory_order_relaxed)) {
            reload_saved_tensors_locked();
        }
        validate_saved_tensors();
        return saved_tensors_;
    }
    // Check that saved tensors have not been modified in-place since save
    validate_saved_tensors();
    return saved_tensors_;
}

void Function::validate_saved_tensors() const {
    if (saved_tensors_.size() != saved_versions_.size()) {
        throw std::runtime_error(
            "saved_tensors_/saved_versions_ size mismatch: " +
            std::to_string(saved_tensors_.size()) + " tensors vs " +
            std::to_string(saved_versions_.size()) + " versions in " + name());
    }
    for (size_t i = 0; i < saved_tensors_.size(); ++i) {
        // Skip version check for saved tensors the backward function doesn't
        // actually need.  This enables in-place modification of those tensors
        // between forward and backward (matching PyTorch's behavior).
        if (!needs_saved_tensor(i)) {
            continue;
        }
        if (saved_tensors_[i].version() != saved_versions_[i]) {
            throw std::runtime_error(
                std::string("one of the variables needed for gradient computation has been "
                "modified by an in-place operation: saved tensor ") + std::to_string(i) +
                " in " + name() + ". Use .clone() before in-place ops on tensors used in "
                "autograd computation, or check that the backward function overrides "
                "needs_saved_tensor() for inputs it doesn't read.");
        }
        // Check if view base was modified (in-place op on the base invalidates the view)
        if (i < saved_view_base_versions_.size() && saved_view_base_versions_[i] != 0) {
            auto* base = saved_tensors_[i]._view_base();
            if (base) {
                auto current_base_ver = base->version();
                if (current_base_ver != saved_view_base_versions_[i]) {
                    throw std::runtime_error(
                        "a view's base tensor has been modified by an in-place operation after "
                        "the forward pass. This invalidates the saved view. Use .clone() on "
                        "the view before the in-place modification.");
                }
            }
        }
    }
}

void Function::require_saved_tensors(size_t count) const {
    if (saved_tensors_.size() < count) {
        throw std::runtime_error(
            std::string(name()) + "::backward() expected " + std::to_string(count) +
            " saved tensors but got " + std::to_string(saved_tensors_.size()) +
            ". This is an internal autograd bug — forward() did not save the expected tensors.");
    }
}

void Function::require_saved_variables(size_t count) const {
    if (saved_variables_.size() < count) {
        throw std::runtime_error(
            std::string(name()) + "::backward_with_variables() expected " + std::to_string(count) +
            " saved variables but got " + std::to_string(saved_variables_.size()) +
            ". This is an internal autograd bug.");
    }
}

void Function::reload_saved_tensors() const {
    if (!tensors_offloaded_.load(std::memory_order_acquire)) return;
    std::lock_guard lock(offload_mutex_);
    if (!tensors_offloaded_.load(std::memory_order_relaxed)) return;
    reload_saved_tensors_locked();
}

void Function::reload_saved_tensors_locked() const {
    // Caller must hold offload_mutex_. Refresh saved_versions_ after
    // the move back so that validate_saved_tensors doesn't falsely
    // report an in-place modification — .to() produces a new Tensor
    // with a fresh version counter each time, so the version the
    // original save_for_backward() recorded is no longer meaningful
    // after the offload/reload round trip. We record the *current*
    // post-reload version as the new baseline.
    for (size_t i = 0; i < saved_tensors_.size(); ++i) {
        // V.5: use per-tensor source device; fall back to the legacy
        // single device for entries that pre-date the per-tensor tracking
        // (e.g. a Function that bypassed save_for_backward to populate
        // saved_tensors_ directly).
        Device target = (i < offloaded_devices_.size())
            ? offloaded_devices_[i]
            : offloaded_device_;
        saved_tensors_[i] = saved_tensors_[i].to(target);
        if (i < saved_versions_.size()) {
            saved_versions_[i] = saved_tensors_[i].version();
        }
    }
    tensors_offloaded_.store(false, std::memory_order_release);
}

auto Function::save_variables_for_backward(std::vector<Variable> variables) -> void {
    saved_variables_ = std::move(variables);
}

auto Function::saved_variables() const -> const std::vector<Variable>& {
    return saved_variables_;
}

auto Function::passthrough_stub_backward(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    // Strip the Variable wrappers, call the subclass's raw backward(),
    // and return Variables without grad_fn. Used by Function subclasses
    // whose 2nd derivative is structurally zero (linear or piecewise-
    // linear forwards); pair with is_higher_order_stub() = true so the
    // engine's Warn-mode counter reflects the disconnection.
    std::vector<Tensor> tensor_grads;
    tensor_grads.reserve(grad_outputs.size());
    for (auto& var : grad_outputs) {
        tensor_grads.push_back(var.tensor());
    }
    auto result_tensors = backward(std::move(tensor_grads));
    std::vector<Variable> result_vars;
    result_vars.reserve(result_tensors.size());
    for (auto& t : result_tensors) {
        result_vars.emplace_back(std::move(t), /*requires_grad=*/false);
    }
    return result_vars;
}

auto Function::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Check if any grad_output requires grad (i.e., create_graph=true was used)
    bool any_requires_grad = false;
    for (auto& var : grad_outputs) {
        if (var.requires_grad()) { any_requires_grad = true; break; }
    }

    if (any_requires_grad) {
        // Higher-order gradients requested but this op doesn't support them.
        auto mode = get_higher_order_grad_mode();
        auto op_name = name();

        switch (mode) {
        case HigherOrderGradMode::Error:
            throw std::runtime_error(
                "create_graph=true requires higher-order gradient support, but '" +
                op_name + "' does not implement backward_with_variables(). "
                "Either use create_graph=false, or call "
                "set_higher_order_grad_mode(HigherOrderGradMode::Warn) "
                "to fall through with disconnected gradient graph.");
        case HigherOrderGradMode::Warn: {
            auto count = g_higher_order_disconnection_count.fetch_add(1, std::memory_order_relaxed) + 1;
            // CC.7: rate-limit the warning to avoid unbounded log spam when a
            // training loop hits a non-higher-order op every step (which is
            // typical — Warn-mode is meant to be a soft signal, not a flood).
            // The atomic counter itself is untouched so `higher_order_disconnection_count()`
            // still reports the true total; we only throttle the log emission.
            // Emit the first occurrence (count == 1) and then every 100th, so
            // bursts in tight loops collapse to ~1% of their original volume.
            // Audit I.4: unified logger so TENZOR_LOG_LEVEL can silence
            // these warnings entirely in production.
            if (count == 1 || (count % 100) == 0) {
                TENZOR_LOG_WARN("[tenzor::autograd] '{}' does not support higher-order "
                                "gradients{}. The gradient graph is disconnected at "
                                "this operation — second-order derivatives through "
                                "it will be zero. (disconnection #{})",
                                op_name,
                                is_higher_order_stub() ? " (passthrough stub)" : "",
                                count);
            }
            break;
        }
        }
    }

    // Fallback: extract Tensors, call backward(), wrap results without grad tracking
    std::vector<Tensor> tensor_grads;
    tensor_grads.reserve(grad_outputs.size());
    for (auto& var : grad_outputs) {
        tensor_grads.push_back(var.tensor());
    }

    auto result_tensors = backward(tensor_grads);

    std::vector<Variable> result_vars;
    result_vars.reserve(result_tensors.size());
    for (auto& t : result_tensors) {
        result_vars.emplace_back(t, false);
    }
    return result_vars;
}

auto Function::name() const -> std::string {
#ifdef __GNUC__
    int status = 0;
    char* demangled = abi::__cxa_demangle(typeid(*this).name(), nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        std::string result(demangled);
        free(demangled);
        // Strip namespace prefixes for readability (e.g., "tenzor::nn::ReLUBackward" → "ReLUBackward")
        auto pos = result.rfind("::");
        if (pos != std::string::npos) {
            return result.substr(pos + 2);
        }
        return result;
    }
#endif
    return typeid(*this).name();
}

// Helper functions moved to function_helpers.hpp


} // namespace tenzor
