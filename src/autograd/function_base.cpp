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

// Counter for higher-order gradient disconnections (Warn/Silent mode).
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
            if (std::string(env) == "silent") return static_cast<int>(HigherOrderGradMode::Silent);
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
    // Record version counters for in-place modification detection
    saved_versions_.clear();
    saved_versions_.reserve(tensors.size());
    for (const auto& t : tensors) {
        saved_versions_.push_back(t.version());
    }

    if (activation_offload_enabled() && !tensors.empty()) {
        // Check if any tensor is on a GPU device
        auto dev = tensors[0].device();
        if (dev.type != Device::Type::CPU) {
            offloaded_device_ = dev;
            tensors_offloaded_.store(true, std::memory_order_release);
            for (auto& t : tensors) {
                t = t.to(Device::cpu());
            }
        }
    }
    saved_tensors_ = std::move(tensors);
}

auto Function::saved_tensors() const -> const std::vector<Tensor>& {
    if (tensors_offloaded_.load(std::memory_order_acquire)) {
        std::lock_guard lock(offload_mutex_);
        if (tensors_offloaded_.load(std::memory_order_relaxed)) {
            reload_saved_tensors();
        }
        // Check and return while still holding the lock to prevent
        // concurrent offload from modifying saved_tensors_ under us
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
        if (saved_tensors_[i].version() != saved_versions_[i]) {
            throw std::runtime_error(
                "one of the variables needed for gradient computation has been modified by an "
                "in-place operation after the forward pass. Use .clone() before in-place ops "
                "on tensors used in autograd computation.");
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
    for (auto& t : saved_tensors_) {
        t = t.to(offloaded_device_);
    }
    tensors_offloaded_.store(false, std::memory_order_release);
}

auto Function::save_variables_for_backward(std::vector<Variable> variables) -> void {
    saved_variables_ = std::move(variables);
}

auto Function::saved_variables() const -> const std::vector<Variable>& {
    return saved_variables_;
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
            g_higher_order_disconnection_count.fetch_add(1, std::memory_order_relaxed);
            static thread_local std::unordered_set<std::string> warned_ops;
            if (warned_ops.find(op_name) == warned_ops.end()) {
                warned_ops.insert(op_name);
                std::cerr << "[tenzor::autograd] Warning: '" << op_name
                          << "' does not support higher-order gradients"
                          << (is_higher_order_stub() ? " (passthrough stub)" : "")
                          << ". The gradient graph will be disconnected at this "
                          << "operation — second-order derivatives through it will "
                          << "be zero. Use set_higher_order_grad_mode(Error) to "
                          << "make this a hard error, or check "
                          << "higher_order_disconnection_count() programmatically.\n";
            }
            break;
        }
        case HigherOrderGradMode::Silent:
            g_higher_order_disconnection_count.fetch_add(1, std::memory_order_relaxed);
            break;
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
