#include "tenzor/autograd/function.hpp"
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
    for (size_t i = 0; i < saved_tensors_.size() && i < saved_versions_.size(); ++i) {
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
        // Behavior controlled by env var:
        //   TENZOR_HIGHER_ORDER_GRAD=error  → throw (default, safe)
        //   TENZOR_HIGHER_ORDER_GRAD=warn   → warn and fall through (backward compat)
        static int mode = []() {
            if (const char* env = std::getenv("TENZOR_HIGHER_ORDER_GRAD")) {
                if (std::string(env) == "warn") return 1;
                if (std::string(env) == "silent") return 2;
            }
            return 1; // warn by default (allows higher-order grad to work with disconnected graph)
        }();

        auto op_name = name();
        if (mode == 0) {
            throw std::runtime_error(
                "create_graph=true requires higher-order gradient support, but '" +
                op_name + "' does not implement backward_with_variables(). "
                "Either use create_graph=false, or set TENZOR_HIGHER_ORDER_GRAD=warn "
                "to fall through with disconnected gradient graph.");
        } else if (mode == 1) {
            static thread_local std::unordered_set<std::string> warned_ops;
            if (warned_ops.find(op_name) == warned_ops.end()) {
                warned_ops.insert(op_name);
                std::cerr << "[tenzor::autograd] Warning: " << op_name
                          << " does not support higher-order gradients. "
                          << "Gradient graph will be disconnected at this operation.\n";
            }
        }
        // mode == 2: silent fallthrough
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
        result_vars.emplace_back(t, any_requires_grad && t.is_valid());
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

// Helper function to reduce gradient Variable along broadcasted dimensions (for create_graph)
static auto reduce_grad_var_for_broadcasting(const Variable& grad, const std::vector<int64_t>& target_shape) -> Variable {
    auto grad_shape_vec = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());

    // If shapes match, no reduction needed
    if (grad_shape_vec == target_shape) {
        return grad;
    }

    auto result = grad;

    // Handle size difference (prepended dimensions in grad)
    int64_t ndim_diff = static_cast<int64_t>(grad_shape_vec.size()) - static_cast<int64_t>(target_shape.size());

    if (ndim_diff > 0) {
        // grad has MORE dimensions than target - sum along prepended dimensions
        for (int64_t i = 0; i < ndim_diff; ++i) {
            result = tenzor::sum(result, 0, false);
        }
    } else if (ndim_diff < 0) {
        throw std::runtime_error(
            "Autograd bug: gradient has fewer dimensions (" +
            std::to_string(grad_shape_vec.size()) + ") than target shape (" +
            std::to_string(target_shape.size()) + ")");
    }

    // Now result and target should have same ndim
    // Sum along dimensions that were broadcasted (size 1 in target but > 1 in result)
    auto result_shape_vec = std::vector<int64_t>(result.shape().begin(), result.shape().end());
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == 1 && result_shape_vec[i] > 1) {
            result = tenzor::sum(result, static_cast<int64_t>(i), true);
            result_shape_vec = std::vector<int64_t>(result.shape().begin(), result.shape().end());
        }
    }

    // Final reshape to exact target shape
    if (result_shape_vec != target_shape) {
        result = tenzor::reshape(result, target_shape);
    }

    return result;
}

// Helper function to reduce gradient along broadcasted dimensions
static auto reduce_grad_for_broadcasting(const Tensor& grad, const std::vector<int64_t>& target_shape) -> Tensor {
    auto grad_shape_vec = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());

    // If shapes match, no reduction needed
    if (grad_shape_vec == target_shape) {
        return grad;
    }

    auto result = grad;

    // Handle size difference (prepended dimensions in grad)
    int64_t ndim_diff = static_cast<int64_t>(grad_shape_vec.size()) - static_cast<int64_t>(target_shape.size());

    if (ndim_diff > 0) {
        // grad has MORE dimensions than target - sum along prepended dimensions
        for (int64_t i = 0; i < ndim_diff; ++i) {
            result = tenzor::sum(result, 0, false);  // Sum and remove dimension
        }
    } else if (ndim_diff < 0) {
        // grad has FEWER dimensions than target — this indicates an autograd graph bug
        // (gradient should never have fewer dimensions than the variable it flows to)
        throw std::runtime_error(
            "Autograd bug: gradient has fewer dimensions (" +
            std::to_string(grad_shape_vec.size()) + ") than target shape (" +
            std::to_string(target_shape.size()) + ")");
    }

    // Now result and target should have same ndim
    // Sum along dimensions that were broadcasted (size 1 in target but > 1 in result)
    auto result_shape_vec = std::vector<int64_t>(result.shape().begin(), result.shape().end());
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == 1 && result_shape_vec[i] > 1) {
            result = tenzor::sum(result, static_cast<int64_t>(i), true);  // Keep dim as size 1
            result_shape_vec = std::vector<int64_t>(result.shape().begin(), result.shape().end());
        }
    }

    // Final reshape to exact target shape (handle keepdim=true above)
    if (result_shape_vec != target_shape) {
        result = reshape(result, target_shape);
    }

    return result;
}

// AddBackward implementation
auto AddBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Save input shapes for broadcasting-aware backward pass
    input_shape_a_ = std::vector<int64_t>(inputs[0].shape().begin(), inputs[0].shape().end());
    input_shape_b_ = std::vector<int64_t>(inputs[1].shape().begin(), inputs[1].shape().end());

    auto result = add(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto AddBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // Fast path: same shape (common case for residual connections)
    // Avoids function call overhead and vector comparisons
    if (input_shape_a_ == input_shape_b_) {
        auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
        if (grad_shape == input_shape_a_) {
            // Both inputs have same shape as gradient - just return gradient twice
            return {grad, grad};
        }
    }

    // Slow path: handle broadcasting
    auto grad_a = reduce_grad_for_broadcasting(grad, input_shape_a_);
    auto grad_b = reduce_grad_for_broadcasting(grad, input_shape_b_);
    return {grad_a, grad_b};
}

auto AddBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    const auto& grad = grad_outputs[0];

    // For add: gradients are just the incoming gradient (possibly with broadcast reduction)
    // Since these are identity operations on the gradient, the Variable grad already has
    // its grad_fn set, so higher-order gradients flow through naturally
    if (input_shape_a_ == input_shape_b_) {
        auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
        if (grad_shape == input_shape_a_) {
            return {grad, grad};
        }
    }

    auto grad_a = reduce_grad_var_for_broadcasting(grad, input_shape_a_);
    auto grad_b = reduce_grad_var_for_broadcasting(grad, input_shape_b_);
    return {grad_a, grad_b};
}

// SubBackward implementation
auto SubBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Save input shapes for broadcasting-aware backward pass
    input_shape_a_ = std::vector<int64_t>(inputs[0].shape().begin(), inputs[0].shape().end());
    input_shape_b_ = std::vector<int64_t>(inputs[1].shape().begin(), inputs[1].shape().end());

    auto result = sub(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto SubBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(a-b)/da = 1, d(a-b)/db = -1
    // Handle broadcasting
    auto grad_a = reduce_grad_for_broadcasting(grad_outputs[0], input_shape_a_);
    auto grad_b_unreduced = neg(grad_outputs[0]);
    auto grad_b = reduce_grad_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

auto SubBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(a-b)/da = 1, d(a-b)/db = -1
    // Use Variable operations so negation is tracked for higher-order gradients
    auto grad_a = reduce_grad_var_for_broadcasting(grad_outputs[0], input_shape_a_);
    auto grad_b_unreduced = tenzor::neg(grad_outputs[0]);  // Variable neg - tracked by autograd
    auto grad_b = reduce_grad_var_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

// MulBackward implementation
auto MulBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor(), inputs[1].tensor()});
    // Save input shapes for broadcasting-aware backward pass
    input_shape_a_ = std::vector<int64_t>(inputs[0].shape().begin(), inputs[0].shape().end());
    input_shape_b_ = std::vector<int64_t>(inputs[1].shape().begin(), inputs[1].shape().end());

    auto result = mul(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto MulBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    // d(a*b)/da = b, d(a*b)/db = a
    // Handle broadcasting
    auto grad_a_unreduced = mul(grad_outputs[0], saved_tensors_[1]);
    auto grad_b_unreduced = mul(grad_outputs[0], saved_tensors_[0]);

    auto grad_a = reduce_grad_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_for_broadcasting(grad_b_unreduced, input_shape_b_);

    return {grad_a, grad_b};
}

auto MulBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(a*b)/da = b, d(a*b)/db = a
    // Use saved Variables if available, otherwise wrap saved Tensors
    Variable saved_a, saved_b;
    if (has_saved_variables()) {
        require_saved_variables(2);
        saved_a = saved_variables_[0];
        saved_b = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        saved_a = Variable(saved_tensors_[0], false);
        saved_b = Variable(saved_tensors_[1], false);
    }

    // Use Variable operations so multiplication is tracked for higher-order gradients
    auto grad_a_unreduced = grad_outputs[0] * saved_b;
    auto grad_b_unreduced = grad_outputs[0] * saved_a;

    auto grad_a = reduce_grad_var_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_var_for_broadcasting(grad_b_unreduced, input_shape_b_);

    return {grad_a, grad_b};
}

// DivBackward implementation
auto DivBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor(), inputs[1].tensor()});
    // Save input shapes for broadcasting-aware backward pass
    input_shape_a_ = std::vector<int64_t>(inputs[0].shape().begin(), inputs[0].shape().end());
    input_shape_b_ = std::vector<int64_t>(inputs[1].shape().begin(), inputs[1].shape().end());

    auto result = div(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto DivBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    // d(a/b)/da = 1/b, d(a/b)/db = -a/(b^2)
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];

    auto grad_a_unreduced = div(grad_outputs[0], b);
    // grad_b = -a / (b^2) * grad_output = -(a * grad_output) / (b * b)
    auto grad_b_unreduced = neg(div(mul(a, grad_outputs[0]), mul(b, b)));

    auto grad_a = reduce_grad_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

auto DivBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(a/b)/da = 1/b, d(a/b)/db = -a/(b^2)
    Variable saved_a, saved_b;
    if (has_saved_variables()) {
        require_saved_variables(2);
        saved_a = saved_variables_[0];
        saved_b = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        saved_a = Variable(saved_tensors_[0], false);
        saved_b = Variable(saved_tensors_[1], false);
    }

    // Use Variable operations for higher-order gradient tracking
    auto grad_a_unreduced = grad_outputs[0] / saved_b;
    // grad_b = -(a * grad_output) / (b * b)
    auto grad_b_unreduced = tenzor::neg((saved_a * grad_outputs[0]) / (saved_b * saved_b));

    auto grad_a = reduce_grad_var_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_var_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

// MatMulBackward implementation
auto MatMulBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor(), inputs[1].tensor()});
    auto result = matmul(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto MatMulBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    // For C = A @ B:
    // dL/dA = dL/dC @ B.T
    // dL/dB = A.T @ dL/dC
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& grad_out = grad_outputs[0];

    // Get the number of dimensions
    auto a_ndim = a.shape().size();
    auto b_ndim = b.shape().size();

    // For 2D matrices: grad_a = grad_out @ b.T, grad_b = a.T @ grad_out
    auto b_t = transpose(b, b_ndim - 2, b_ndim - 1);
    auto a_t = transpose(a, a_ndim - 2, a_ndim - 1);

    auto grad_a = matmul(grad_out, b_t);
    auto grad_b = matmul(a_t, grad_out);

    return {grad_a, grad_b};
}

auto MatMulBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // For C = A @ B:
    // dL/dA = dL/dC @ B.T
    // dL/dB = A.T @ dL/dC
    Variable saved_a, saved_b;
    if (has_saved_variables()) {
        require_saved_variables(2);
        saved_a = saved_variables_[0];
        saved_b = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        saved_a = Variable(saved_tensors_[0], false);
        saved_b = Variable(saved_tensors_[1], false);
    }

    const auto& grad_out = grad_outputs[0];

    // Use Variable operations for higher-order gradient tracking
    auto a_ndim = saved_a.shape().size();
    auto b_ndim = saved_b.shape().size();

    auto b_t = tenzor::transpose(saved_b, b_ndim - 2, b_ndim - 1);
    auto a_t = tenzor::transpose(saved_a, a_ndim - 2, a_ndim - 1);

    auto grad_a = tenzor::matmul(grad_out, b_t);
    auto grad_b = tenzor::matmul(a_t, grad_out);

    return {grad_a, grad_b};
}

// LinearBackward implementation
auto LinearBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // This is typically not called - autograd::linear() handles forward directly
    // But implement for completeness
    // inputs[0] = x (batch_size, in_features)
    // inputs[1] = W (out_features, in_features)
    // inputs[2] = b (out_features)
    save_for_backward({inputs[0].tensor(), inputs[1].tensor(), inputs[2].tensor()});

    // Compute: y = x @ W.T + b
    auto x = inputs[0].tensor();
    auto w = inputs[1].tensor();
    auto b = inputs[2].tensor();

    // Transpose weight: (out_features, in_features) -> (in_features, out_features)
    auto w_t = transpose(w, 0, 1);

    // Matrix multiplication: (batch, in) @ (in, out) -> (batch, out)
    auto matmul_result = matmul(x, w_t);

    // Add bias (broadcasts automatically)
    auto result = add(matmul_result, b);

    return {Variable(result, true)};
}

auto LinearBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    // For y = x @ W.T + b:
    // dL/dx = dL/dy @ W          -> (batch, out) @ (out, in) = (batch, in)
    // dL/dW = dL/dy.T @ x        -> (out, batch) @ (batch, in) = (out, in)
    // dL/db = sum(dL/dy, dim=0)  -> reduce batch dimension

    const auto& x = saved_tensors_[0];       // (batch, in)
    const auto& w = saved_tensors_[1];       // (out, in)
    const auto& grad_out = grad_outputs[0];  // (batch, out)

    // Use optimized LinearBackward kernel (supports Float32, Float64, Float16, BFloat16)
    if (grad_out.device().type == Device::Type::CUDA ||
        grad_out.dtype() == DType::Float32 || grad_out.dtype() == DType::Float64) {
        // For Float16/BFloat16 on CUDA, upcast to Float32 for computation to
        // prevent gradient overflow. cuBLAS GemmEx outputs Float16 which can't
        // represent values > 65504, causing Inf in larger models.
        DType orig_dt = grad_out.dtype();
        bool needs_upcast = (grad_out.device().type == Device::Type::CUDA &&
                            (orig_dt == DType::Float16 || orig_dt == DType::BFloat16));
        if (needs_upcast) {
            std::vector<Tensor> inputs = {
                grad_out.to(DType::Float32),
                x.to(DType::Float32),
                w.to(DType::Float32)
            };
            auto results = dispatch<OpId::LinearBackward>(inputs);
            for (auto& r : results) r = r.to(orig_dt);
            return results;
        }
        // Ensure all inputs match grad_out dtype (mixed precision: e.g. Float64
        // input with Float32 weight) — CUDA kernels select kernel by first tensor dtype
        auto x_cast = (x.dtype() != orig_dt) ? x.to(orig_dt) : x;
        auto w_cast = (w.dtype() != orig_dt) ? w.to(orig_dt) : w;
        std::vector<Tensor> inputs = {grad_out, x_cast, w_cast};
        return dispatch<OpId::LinearBackward>(inputs);
    }

    // Fallback for other backends/types using tensor operations
    // grad_input = grad_out @ W
    auto grad_x = matmul(grad_out, w);

    // grad_weight = grad_out.T @ x
    auto grad_out_t = transpose(grad_out, 0, 1);  // (out, batch)
    auto grad_w = matmul(grad_out_t, x);          // (out, in)

    // grad_bias = sum(grad_out, dim=0)
    auto grad_b = tenzor::sum(grad_out, 0, false);  // (out,)

    return {grad_x, grad_w, grad_b};
}

auto LinearBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // For y = x @ W.T + b:
    // dL/dx = dL/dy @ W
    // dL/dW = dL/dy.T @ x
    // dL/db = sum(dL/dy, dim=0)
    Variable saved_x, saved_w;
    if (has_saved_variables()) {
        require_saved_variables(2);
        saved_x = saved_variables_[0];
        saved_w = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        saved_x = Variable(saved_tensors_[0], false);
        saved_w = Variable(saved_tensors_[1], false);
    }
    const auto& grad_out = grad_outputs[0];

    auto grad_x = tenzor::matmul(grad_out, saved_w);
    auto grad_out_t = tenzor::transpose(grad_out, 0, 1);
    auto grad_w = tenzor::matmul(grad_out_t, saved_x);
    auto grad_b = tenzor::sum(grad_out, 0, false);
    return {grad_x, grad_w, grad_b};
}

// SumBackward implementation
auto SumBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});
    auto result = sum(inputs[0].tensor(), dim_, keepdim_);
    return {Variable(result, true)};
}

auto SumBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "SumBackward: cannot compute gradient of sum over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Reduced all dimensions - broadcast scalar tensor back to original shape
        // Use pure tensor operations (no CPU transfers) - backend agnostic!
        auto grad = grad_output;

        // Ensure grad is a 0-d tensor (may be 1-element tensor from some reductions)
        if (grad.ndim() > 0) {
            grad = reshape(grad, {});
        }

        // Use expand() to broadcast the scalar to input shape natively on device
        auto result = expand(grad, input_shape_vec);

        return {result};
    } else {
        // Dimension-specific reduction backward using unsqueeze + expand
        // expand() now uses native CUDA implementation - no device transfers!
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
        }

        return {expand(grad, input_shape_vec)};
    }
}

auto SumBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Sum backward is just expanding the gradient back to input shape.
    // This operation doesn't depend on saved inputs, so we can use Tensor-level
    // expand/reshape and wrap the result. The gradient Variable itself carries
    // its computation graph for higher-order differentiation.
    const auto& input = saved_tensors_[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "SumBackward: cannot compute gradient of sum over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto grad_tensor = grad_outputs[0].tensor();

    if (!dim_.has_value()) {
        if (grad_tensor.ndim() > 0) {
            grad_tensor = reshape(grad_tensor, {});
        }
        auto result = expand(grad_tensor, input_shape_vec);
        // Wrap as Variable preserving requires_grad from the incoming gradient
        return {Variable(result, grad_outputs[0].requires_grad())};
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_tensor;
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
        }
        auto result = expand(grad, input_shape_vec);
        return {Variable(result, grad_outputs[0].requires_grad())};
    }
}

// MeanBackward implementation
auto MeanBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});
    auto result = mean(inputs[0].tensor(), dim_, keepdim_);
    return {Variable(result, true)};
}

auto MeanBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    // Calculate the number of elements that were averaged
    int64_t n_elements = 1;
    if (dim_.has_value()) {
        n_elements = input.shape()[dim_.value()];
    } else {
        n_elements = input.numel();
    }

    TENZOR_CHECK_SHAPE(n_elements > 0,
        "MeanBackward: cannot compute mean of empty tensor (0 elements)");

    // Use double for scale calculation to preserve precision for Float64 tensors
    double scale = 1.0 / static_cast<double>(n_elements);
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Reduced all dimensions - broadcast scalar tensor back to original shape
        // Use pure tensor operations (no CPU transfers) - backend agnostic!
        auto grad = grad_output;

        // Ensure grad is a 0-d tensor (may be 1-element tensor from some reductions)
        if (grad.ndim() > 0) {
            grad = reshape(grad, {});
        }

        // Expand the scalar to input shape natively on device
        auto expanded = expand(grad, input_shape_vec);

        // Scale by 1/N using backend-agnostic tensor multiplication
        // Create scalar tensor with same dtype and device as expanded gradient
        // Use double overload of full() to preserve precision for Float64
        auto scale_tensor = full({}, scale, expanded.dtype(), expanded.device());

        auto result = mul(expanded, scale_tensor);

        return {result};
    } else {
        // Dimension-specific reduction backward
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
        }

        // Expand to input shape - works on CPU, transfers back if needed
        auto expanded = expand(grad, input_shape_vec);

        // Scale the expanded gradient using native Tensor multiplication
        // This now uses CUDA broadcasting automatically
        // Use expanded.dtype() to ensure dtypes match for element-wise operations
        // Use double overload of full() to preserve precision for Float64
        auto scale_tensor = full(input_shape_vec, scale, expanded.dtype(), expanded.device());
        return {mul(expanded, scale_tensor)};
    }
}

auto MeanBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Mean backward: expand gradient and scale by 1/N.
    // The scaling by 1/N uses Variable::operator*(double) which IS tracked by autograd
    // for higher-order gradient support.
    const auto& input = saved_tensors_[0];

    int64_t n_elements = 1;
    if (dim_.has_value()) {
        n_elements = input.shape()[dim_.value()];
    } else {
        n_elements = input.numel();
    }

    TENZOR_CHECK_SHAPE(n_elements > 0,
        "MeanBackward: cannot compute mean of empty tensor (0 elements)");

    double scale = 1.0 / static_cast<double>(n_elements);
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Scale the gradient Variable - this uses Variable::operator*(double) which
    // builds autograd graph when create_graph is active
    auto scaled_grad = grad_outputs[0] * scale;

    // Now expand to input shape using Tensor-level operations
    auto grad_tensor = scaled_grad.tensor();

    if (!dim_.has_value()) {
        if (grad_tensor.ndim() > 0) {
            grad_tensor = reshape(grad_tensor, {});
        }
        auto result = expand(grad_tensor, input_shape_vec);
        return {Variable(result, scaled_grad.requires_grad())};
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_tensor;
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
        }
        auto result = expand(grad, input_shape_vec);
        return {Variable(result, scaled_grad.requires_grad())};
    }
}

// LogBackward implementation
auto LogBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});
    auto result = log(inputs[0].tensor());
    return {Variable(result, true)};
}

auto LogBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(log(x))/dx = 1/x, with zero-safe clamping to prevent NaN
    const auto& input = saved_tensors_[0];
    auto zero = zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
    auto eps = full(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                    static_cast<double>(std::numeric_limits<float>::min()),
                    input.dtype(), input.device());
    auto safe_input = where(eq(input, zero), eps, input);
    auto grad_input = div(grad_outputs[0], safe_input);
    return {grad_input};
}

auto LogBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(log(x))/dx = 1/x
    // Use Variable division for higher-order gradient tracking
    Variable saved_input(saved_tensors_[0], false);
    return {grad_outputs[0] / saved_input};
}

// ExpBackward implementation
auto ExpBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = exp(inputs[0].tensor());
    save_for_backward({result});  // Save output for backward
    return {Variable(result, true)};
}

auto ExpBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(exp(x))/dx = exp(x)
    const auto& output = saved_tensors_[0];
    auto grad_input = mul(grad_outputs[0], output);
    return {grad_input};
}

auto ExpBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(exp(x))/dx = exp(x) = saved output
    // Use Variable multiplication for higher-order gradient tracking
    Variable saved_output(saved_tensors_[0], false);
    return {grad_outputs[0] * saved_output};
}

// NegBackward implementation
auto NegBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = neg(inputs[0].tensor());
    return {Variable(result, true)};
}

auto NegBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(-x)/dx = -1
    return {neg(grad_outputs[0])};
}

auto NegBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(-x)/dx = -1
    // Use Variable neg for higher-order gradient tracking
    return {tenzor::neg(grad_outputs[0])};
}

// LogSoftmaxBackward implementation
auto LogSoftmaxBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim_);
    std::vector<Tensor> input_tensors = {inputs[0].tensor()};
    auto result = dispatch(OpId::LogSoftmax, input_tensors, attrs)[0];

    // Save output for backward
    save_for_backward({result});

    return {Variable(result, true)};
}

auto LogSoftmaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Use backend's log_softmax_backward kernel
    const auto& output = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim_);
    std::vector<Tensor> inputs = {grad_output, output};
    auto grad_input = dispatch(OpId::LogSoftmaxBackward, inputs, attrs)[0];

    return {grad_input};
}

auto LogSoftmaxBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dx_i = dL/dy_i - exp(y_i) * sum_j(dL/dy_j)
    // Use Variable operations for higher-order gradient tracking
    Variable output_var(saved_tensors_[0], false);
    auto grad_sum = tenzor::sum(grad_outputs[0], dim_, true);
    auto softmax_output = tenzor::exp(output_var);
    auto grad_input = grad_outputs[0] - softmax_output * grad_sum;
    return {grad_input};
}

// SoftmaxBackward implementation
auto SoftmaxBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim_);
    std::vector<Tensor> input_tensors = {inputs[0].tensor()};
    auto result = dispatch(OpId::Softmax, input_tensors, attrs)[0];

    // Save output for backward
    save_for_backward({result});

    return {Variable(result, true)};
}

auto SoftmaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Use backend-optimized softmax_backward kernel via dispatch
    const auto& output = saved_tensors_[0];  // y = softmax(x)
    const auto& grad_output = grad_outputs[0];  // dL/dy

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim_);
    std::vector<Tensor> inputs = {grad_output, output};
    auto grad_input = dispatch(OpId::SoftmaxBackward, inputs, attrs)[0];

    return {grad_input};
}

auto SoftmaxBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
    // Use Variable operations for higher-order gradient tracking
    Variable output_var(saved_tensors_[0], false);
    auto dot_product = tenzor::sum(grad_outputs[0] * output_var, dim_, true);
    auto grad_input = output_var * (grad_outputs[0] - dot_product);
    return {grad_input};
}

// AbsBackward implementation
auto AbsBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});
    auto result = abs(inputs[0].tensor());
    return {Variable(result, true)};
}

auto AbsBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(abs(x))/dx = sign(x), but sign(0) is 0 which gives NaN gradient.
    // Use epsilon guard: where(|x| > eps, sign(x), 0) to avoid NaN at x=0.
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    float eps = 1e-7f; // default for Float32
    if (input.dtype() == DType::Float64) eps = 1e-15f;
    else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) eps = 1e-3f;

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto abs_input = tenzor::abs(input);
    auto eps_tensor = full(input_shape_vec, eps, input.dtype(), input.device());
    auto mask = gt(abs_input, eps_tensor);
    auto safe_sign = tenzor::where(mask,
        sign(input),
        zeros(input_shape_vec, input.dtype(), input.device()));
    return {mul(grad, safe_sign)};
}

auto AbsBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(abs(x))/dx = sign(x), with epsilon guard to avoid NaN at x=0.
    // sign is non-differentiable, so compute it at Tensor level.
    const auto& input = saved_tensors_[0];

    float eps = 1e-7f;
    if (input.dtype() == DType::Float64) eps = 1e-15f;
    else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) eps = 1e-3f;

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto abs_input = tenzor::abs(input);
    auto eps_tensor = full(input_shape_vec, eps, input.dtype(), input.device());
    auto mask = gt(abs_input, eps_tensor);
    auto safe_sign = tenzor::where(mask,
        sign(input),
        zeros(input_shape_vec, input.dtype(), input.device()));
    Variable sign_var(safe_sign, false);
    return {grad_outputs[0] * sign_var};
}

// ClampBackward implementation
auto ClampBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});
    auto result = clamp(inputs[0].tensor(), min_, max_);
    return {Variable(result, true)};
}

auto ClampBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(clamp(x, min, max))/dx = 1 if min <= x <= max else 0
    const auto& input = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    // Create mask: 1 where min <= x <= max, 0 otherwise
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
    auto zeros_tensor = zeros(input_shape_vec, input.dtype(), input.device());

    // Check if input >= min
    auto min_tensor = full(input_shape_vec, min_, input.dtype(), input.device());
    auto max_tensor = full(input_shape_vec, max_, input.dtype(), input.device());

    // Mask = (input >= min) & (input <= max)
    // For now, use clamp and compare approach
    auto clamped = clamp(input, min_, max_);

    // grad = grad_output where input == clamped else 0
    // This is approximately: mask = 1 - abs(sign(input - clamped))
    auto diff = sub(input, clamped);
    auto diff_sign = abs(sign(diff));
    auto mask = sub(ones_tensor, diff_sign);

    return {mul(grad_output, mask)};
}

auto ClampBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(clamp(x, min, max))/dx = 1 if min <= x <= max, else 0
    // The mask is non-differentiable, compute at Tensor level
    const auto& input = saved_tensors_[0];
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
    auto clamped = clamp(input, min_, max_);
    auto diff = sub(input, clamped);
    auto diff_sign = abs(sign(diff));
    auto mask = sub(ones_tensor, diff_sign);
    Variable mask_var(mask, false);
    return {grad_outputs[0] * mask_var};
}

// MaxBackward implementation
auto MaxBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = max(inputs[0].tensor(), dim_, keepdim_);
    // Save both input and output for backward
    save_for_backward({inputs[0].tensor(), result});
    return {Variable(result, true)};
}

auto MaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "MaxBackward: cannot compute gradient of max over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Global max: gradient flows only to the maximum element
        // Create mask where input == output (broadcasted)

        // Reshape scalar output to match input dimensions before expanding
        auto output_reshaped = output;
        if (output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            output_reshaped = reshape(output, ones_shape);
        }
        auto output_expanded = expand(output_reshaped, input_shape_vec);

        // Create mask where input == output (within epsilon)
        auto diff = sub(input, output_expanded);
        auto abs_diff = abs(diff);
        // Select epsilon appropriate for the tensor's precision
        double eps_val;
        switch (input.dtype()) {
            case DType::Float64:  eps_val = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val = 1e-3; break;
            default:              eps_val = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val, input.dtype(), input.device());
        auto mask_bool = lt(abs_diff, epsilon);
        // Convert boolean mask to float for gradient computation
        auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
        auto zeros_tensor = zeros(input_shape_vec, input.dtype(), input.device());
        auto mask = where(mask_bool, ones_tensor, zeros_tensor);

        // Normalize mask by tie count so gradient is split among tied elements
        auto tie_count = sum(mask);
        mask = div(mask, tie_count);

        // Broadcast grad_output to input shape
        // FIX: grad_output is also scalar, need to reshape before expanding
        auto grad_reshaped = grad_output;
        if (grad_output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad_reshaped = reshape(grad_output, ones_shape);
        }
        auto grad_broadcasted = expand(grad_reshaped, input_shape_vec);

        return {mul(grad_broadcasted, mask)};
    } else {
        // Dimension-specific max
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        auto out = output;

        // Unsqueeze if needed
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
            out = unsqueeze(out, dim);
        }

        // Expand to input shape
        auto out_expanded = expand(out, input_shape_vec);
        auto grad_expanded = expand(grad, input_shape_vec);

        // Create mask where input == max_value
        auto diff = sub(input, out_expanded);
        auto abs_diff = abs(diff);

        // Select epsilon appropriate for the tensor's precision
        double eps_val2;
        switch (input.dtype()) {
            case DType::Float64:  eps_val2 = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val2 = 1e-3; break;
            default:              eps_val2 = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val2, input.dtype(), input.device());
        auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
        auto scaled_diff = div(abs_diff, epsilon);
        auto clamped = clamp(scaled_diff, 0.0f, 1.0f);
        auto mask = sub(ones_tensor, clamped);

        // Normalize mask by tie count along dim so gradient is split among tied elements
        auto tie_count = sum(mask, dim, /*keepdim=*/true);
        mask = div(mask, tie_count);

        return {mul(grad_expanded, mask)};
    }
}

auto MaxBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Max backward: gradient flows only to max element(s), masked by position.
    // The mask is a step function (non-differentiable), so compute at Tensor level.
    // For higher-order gradients, the second derivative of max is zero everywhere
    // except at tie points, which have measure zero.
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// MedianBackward implementation
auto MedianBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    int64_t dim = dim_.value_or(-1);
    auto [values, indices] = ::tenzor::median(inputs[0].tensor(), dim, keepdim_);
    save_for_backward({inputs[0].tensor(), values});
    return {Variable(values, true)};
}

auto MedianBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "MedianBackward: cannot compute gradient of median over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Global median: gradient flows only to the median element
        auto output_reshaped = output;
        if (output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            output_reshaped = reshape(output, ones_shape);
        }
        auto output_expanded = expand(output_reshaped, input_shape_vec);

        // Create mask where input == median value (within epsilon)
        auto diff = sub(input, output_expanded);
        auto abs_diff = abs(diff);
        double eps_val;
        switch (input.dtype()) {
            case DType::Float64:  eps_val = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val = 1e-3; break;
            default:              eps_val = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val, input.dtype(), input.device());
        auto mask_bool = lt(abs_diff, epsilon);
        auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
        auto zeros_tensor = zeros(input_shape_vec, input.dtype(), input.device());
        auto mask = where(mask_bool, ones_tensor, zeros_tensor);

        // Normalize mask by tie count so gradient is split among tied elements
        auto tie_count = sum(mask);
        mask = div(mask, tie_count);

        // Broadcast grad_output to input shape
        auto grad_reshaped = grad_output;
        if (grad_output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad_reshaped = reshape(grad_output, ones_shape);
        }
        auto grad_broadcasted = expand(grad_reshaped, input_shape_vec);

        return {mul(grad_broadcasted, mask)};
    } else {
        // Dimension-specific median
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        auto out = output;

        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
            out = unsqueeze(out, dim);
        }

        auto out_expanded = expand(out, input_shape_vec);
        auto grad_expanded = expand(grad, input_shape_vec);

        // Create mask where input == median value
        auto diff = sub(input, out_expanded);
        auto abs_diff = abs(diff);

        double eps_val2;
        switch (input.dtype()) {
            case DType::Float64:  eps_val2 = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val2 = 1e-3; break;
            default:              eps_val2 = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val2, input.dtype(), input.device());
        auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
        auto scaled_diff = div(abs_diff, epsilon);
        auto clamped = clamp(scaled_diff, 0.0f, 1.0f);
        auto mask = sub(ones_tensor, clamped);

        // Normalize mask by tie count along dim
        auto tie_count = sum(mask, dim, /*keepdim=*/true);
        mask = div(mask, tie_count);

        return {mul(grad_expanded, mask)};
    }
}

auto MedianBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ModeBackward implementation
auto ModeBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    int64_t dim = dim_.value_or(-1);
    auto [values, indices] = ::tenzor::mode(inputs[0].tensor(), dim, keepdim_);
    save_for_backward({inputs[0].tensor(), values});
    return {Variable(values, true)};
}

auto ModeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "ModeBackward: cannot compute gradient of mode over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Global mode: gradient flows only to the mode element(s)
        auto output_reshaped = output;
        if (output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            output_reshaped = reshape(output, ones_shape);
        }
        auto output_expanded = expand(output_reshaped, input_shape_vec);

        // Create mask where input == mode value (within epsilon)
        auto diff = sub(input, output_expanded);
        auto abs_diff = abs(diff);
        double eps_val;
        switch (input.dtype()) {
            case DType::Float64:  eps_val = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val = 1e-3; break;
            default:              eps_val = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val, input.dtype(), input.device());
        auto mask_bool = lt(abs_diff, epsilon);
        auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
        auto zeros_tensor = zeros(input_shape_vec, input.dtype(), input.device());
        auto mask = where(mask_bool, ones_tensor, zeros_tensor);

        // Normalize mask by tie count so gradient is split among tied elements
        auto tie_count = sum(mask);
        mask = div(mask, tie_count);

        // Broadcast grad_output to input shape
        auto grad_reshaped = grad_output;
        if (grad_output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad_reshaped = reshape(grad_output, ones_shape);
        }
        auto grad_broadcasted = expand(grad_reshaped, input_shape_vec);

        return {mul(grad_broadcasted, mask)};
    } else {
        // Dimension-specific mode
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        auto out = output;

        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
            out = unsqueeze(out, dim);
        }

        auto out_expanded = expand(out, input_shape_vec);
        auto grad_expanded = expand(grad, input_shape_vec);

        // Create mask where input == mode value
        auto diff = sub(input, out_expanded);
        auto abs_diff = abs(diff);

        double eps_val2;
        switch (input.dtype()) {
            case DType::Float64:  eps_val2 = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val2 = 1e-3; break;
            default:              eps_val2 = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val2, input.dtype(), input.device());
        auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
        auto scaled_diff = div(abs_diff, epsilon);
        auto clamped = clamp(scaled_diff, 0.0f, 1.0f);
        auto mask = sub(ones_tensor, clamped);

        // Normalize mask by tie count along dim
        auto tie_count = sum(mask, dim, /*keepdim=*/true);
        mask = div(mask, tie_count);

        return {mul(grad_expanded, mask)};
    }
}

auto ModeBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ReshapeBackward implementation
auto ReshapeBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("ReshapeBackward::forward should not be called");
}

auto ReshapeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Reshape gradient back to input shape and ensure contiguity
    // Reshape may create non-contiguous views, which can cause issues in element-wise operations
    auto grad_input = reshape(grad_outputs[0], input_shape_).contiguous();
    return {grad_input};
}

auto ReshapeBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {reshape(grad_outputs[0], input_shape_)};
}

// PermuteBackward implementation
auto PermuteBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("PermuteBackward::forward should not be called");
}

auto PermuteBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Apply inverse permutation to gradient and ensure contiguity
    // Permute creates non-contiguous views, which can cause issues in element-wise operations
    auto grad_input = permute(grad_outputs[0], inv_dims_).contiguous();
    return {grad_input};
}

auto PermuteBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {permute(grad_outputs[0], inv_dims_)};
}

// TransposeBackward implementation
auto TransposeBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("TransposeBackward::forward should not be called");
}

auto TransposeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Transpose is its own inverse, so apply same transpose to gradient
    auto grad_input = transpose(grad_outputs[0], dim0_, dim1_).contiguous();
    return {grad_input};
}

auto TransposeBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {transpose(grad_outputs[0], dim0_, dim1_)};
}

// RollBackward implementation
auto RollBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("RollBackward::forward should not be called");
}

auto RollBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Roll backward is roll with negative shift
    auto grad_input = roll(grad_outputs[0], -shifts_, dim_);
    return {grad_input};
}

auto RollBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {roll(grad_outputs[0], -shifts_, dim_)};
}

// SqueezeBackward implementation
auto SqueezeBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("SqueezeBackward::forward should not be called");
}

auto SqueezeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Unsqueeze gradient back to original shape
    auto grad_input = unsqueeze(grad_outputs[0], dim_);
    return {grad_input};
}

auto SqueezeBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Use Variable-level reshape to unsqueeze back to original shape
    // This preserves the computation graph for higher-order gradients
    auto grad = grad_outputs[0];
    auto target_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
    target_shape.insert(target_shape.begin() + dim_, 1);
    return {reshape(grad, target_shape)};
}

// BmmBackward implementation
auto BmmBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor(), inputs[1].tensor()});
    auto result = bmm(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto BmmBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // For C = bmm(A, B):
    // A: (batch, n, m), B: (batch, m, p), C: (batch, n, p)
    // grad_output: (batch, n, p)
    //
    // Backward gradients:
    // grad_a = grad_output @ B^T = (batch, n, p) @ (batch, p, m) = (batch, n, m)
    // grad_b = A^T @ grad_output = (batch, m, n) @ (batch, n, p) = (batch, m, p)

    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& grad_output = grad_outputs[0];

    // Transpose last two dimensions: (batch, m, p) -> (batch, p, m)
    auto b_transposed = permute(b, {0, 2, 1});

    // grad_a = grad_output @ b^T
    auto grad_a = bmm(grad_output, b_transposed);

    // Transpose a: (batch, n, m) -> (batch, m, n)
    auto a_transposed = permute(a, {0, 2, 1});

    // grad_b = a^T @ grad_output
    auto grad_b = bmm(a_transposed, grad_output);

    return {grad_a, grad_b};
}

auto BmmBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // For C = bmm(A, B):
    // grad_a = grad_output @ B^T
    // grad_b = A^T @ grad_output
    Variable saved_a, saved_b;
    if (has_saved_variables()) {
        saved_a = saved_variables_[0];
        saved_b = saved_variables_[1];
    } else {
        saved_a = Variable(saved_tensors_[0], false);
        saved_b = Variable(saved_tensors_[1], false);
    }
    const auto& grad_out = grad_outputs[0];
    auto b_t = tenzor::transpose(saved_b, saved_b.shape().size() - 2, saved_b.shape().size() - 1);
    auto a_t = tenzor::transpose(saved_a, saved_a.shape().size() - 2, saved_a.shape().size() - 1);
    auto grad_a = tenzor::bmm(grad_out, b_t);
    auto grad_b = tenzor::bmm(a_t, grad_out);
    return {grad_a, grad_b};
}

// CatBackward implementation
auto CatBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Convert Variables to Tensors for concatenation
    std::vector<Tensor> tensors;
    tensors.reserve(inputs.size());
    for (const auto& var : inputs) {
        tensors.push_back(var.tensor());
    }

    auto result = cat(tensors, dim_);
    return {Variable(result, true)};
}

auto CatBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Split gradient back along concatenation dimension
    // grad_output shape: [..., sum(split_sizes), ...]
    // Need to split into gradients of shape [..., split_sizes[i], ...]

    const auto& grad_output = grad_outputs[0];
    std::vector<Tensor> grad_inputs;
    grad_inputs.reserve(split_sizes_.size());

    int64_t offset = 0;
    for (int64_t split_size : split_sizes_) {
        // Slice grad_output from offset to offset+split_size along dim_
        auto grad_slice = slice(grad_output, dim_, offset, offset + split_size);
        grad_inputs.push_back(grad_slice);
        offset += split_size;
    }

    return grad_inputs;
}

auto CatBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    const auto& grad_output = grad_outputs[0];
    std::vector<Variable> grad_inputs;
    grad_inputs.reserve(split_sizes_.size());
    int64_t offset = 0;
    for (int64_t split_size : split_sizes_) {
        // Use Variable-level slice to preserve computation graph for higher-order gradients
        grad_inputs.push_back(slice(grad_output, dim_, offset, offset + split_size));
        offset += split_size;
    }
    return grad_inputs;
}

// SliceBackward implementation
auto SliceBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = slice(inputs[0].tensor(), dim_, start_, end_, step_);
    return {Variable(result, true)};
}

auto SliceBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad_output = grad_outputs[0];

    // Create zero gradient tensor with original input shape
    auto grad_input = zeros(input_shape_, grad_output.dtype(), grad_output.device());

    // Build index tensor for scatter operation
    // Index tensor must have same shape as grad_output
    int64_t slice_size = grad_output.shape()[dim_];
    int64_t total_elements = grad_output.numel();

    // Create index tensor with same shape as grad_output
    auto index_shape = std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end());
    auto index = zeros(index_shape, DType::Int64, Device::cpu());

    // Fill index tensor on CPU
    int64_t* index_ptr = index.data<int64_t>();

    // Calculate stride for the sliced dimension
    int64_t dim_stride = 1;
    for (int64_t d = dim_ + 1; d < grad_output.ndim(); ++d) {
        dim_stride *= grad_output.shape()[d];
    }

    // Fill index tensor: each element along dim_ gets mapped to (start_ + pos * step_)
    for (int64_t i = 0; i < total_elements; ++i) {
        int64_t pos_in_dim = (i / dim_stride) % slice_size;
        index_ptr[i] = start_ + pos_in_dim * step_;
    }

    // Transfer to target device if needed
    if (grad_output.device() != Device::cpu()) {
        index = index.to(grad_output.device());
    }

    // Use scatter to place gradients - dispatches to appropriate backend
    grad_input = scatter(grad_input, dim_, index, grad_output);

    return {grad_input};
}

auto SliceBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Same as tensor-level backward but wrapping with requires_grad=true
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
}

// UpsampleBilinearBackward implementation
auto UpsampleBilinearBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Save input tensor for backward pass
    save_for_backward({inputs[0].tensor()});

    // Forward computation is done externally in the wrapper function
    // This method is not typically called directly
    throw std::runtime_error("UpsampleBilinearBackward::forward should not be called directly");
}

auto UpsampleBilinearBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Distribute gradients from upsampled output back to input size
    // For nearest neighbor upsampling: each output pixel's gradient goes to its source input pixel

    const auto& grad_output_orig = grad_outputs[0];
    const auto& shape = grad_output_orig.shape();

    if (shape.size() != 4) {
        throw std::runtime_error("UpsampleBilinearBackward: Expected 4D gradient tensor (N, C, H, W)");
    }

    // Remember original dtype and device for output conversion
    DType original_dtype = grad_output_orig.dtype();
    Device original_device = grad_output_orig.device();

    // Convert to Float32 on CPU for computation
    Tensor grad_output = grad_output_orig.to(Device::cpu());
    if (grad_output.dtype() != DType::Float32) {
        grad_output = grad_output.to(DType::Float32);
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H_out = shape[2];
    int64_t W_out = shape[3];

    // Create gradient tensor for input (all zeros initially) in Float32
    auto grad_input = zeros({N, C, input_h_, input_w_}, DType::Float32, Device::cpu());

    // Calculate scaling factors (align_corners=false convention)
    float scale_h = static_cast<float>(input_h_) / static_cast<float>(output_h_);
    float scale_w = static_cast<float>(input_w_) / static_cast<float>(output_w_);

    // Distribute gradients using bilinear interpolation weights
    auto* grad_in_ptr = grad_input.data<float>();
    const auto* grad_out_ptr = grad_output.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H_out; ++h) {
                for (int64_t w = 0; w < W_out; ++w) {
                    // Map output pixel to input coordinate (align_corners=false)
                    float src_h = (h + 0.5f) * scale_h - 0.5f;
                    float src_w = (w + 0.5f) * scale_w - 0.5f;

                    // Bounding input pixels
                    int64_t h0 = static_cast<int64_t>(std::floor(src_h));
                    int64_t w0 = static_cast<int64_t>(std::floor(src_w));
                    int64_t h1 = h0 + 1;
                    int64_t w1 = w0 + 1;

                    // Interpolation weights from fractional part
                    float fh = src_h - h0;
                    float fw = src_w - w0;

                    float grad_val = grad_out_ptr[((n * C + c) * H_out + h) * W_out + w];
                    int64_t base = (n * C + c) * input_h_;

                    // Accumulate weighted gradient to each of the 4 neighbors
                    if (h0 >= 0 && h0 < input_h_ && w0 >= 0 && w0 < input_w_)
                        grad_in_ptr[(base + h0) * input_w_ + w0] += grad_val * (1.0f - fh) * (1.0f - fw);
                    if (h0 >= 0 && h0 < input_h_ && w1 >= 0 && w1 < input_w_)
                        grad_in_ptr[(base + h0) * input_w_ + w1] += grad_val * (1.0f - fh) * fw;
                    if (h1 >= 0 && h1 < input_h_ && w0 >= 0 && w0 < input_w_)
                        grad_in_ptr[(base + h1) * input_w_ + w0] += grad_val * fh * (1.0f - fw);
                    if (h1 >= 0 && h1 < input_h_ && w1 >= 0 && w1 < input_w_)
                        grad_in_ptr[(base + h1) * input_w_ + w1] += grad_val * fh * fw;
                }
            }
        }
    }

    // Convert back to original dtype and device
    if (grad_input.dtype() != original_dtype) {
        grad_input = grad_input.to(original_dtype);
    }
    if (grad_input.device() != original_device) {
        grad_input = grad_input.to(original_device);
    }

    return {grad_input};
}

auto UpsampleBilinearBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Upsample backward is a linear operation (weighted accumulation), so its
    // second derivative is constant. Compute at Tensor level since the bilinear
    // weights don't depend on the input values.
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// =========================================================================
// Activation Backward Functions
// =========================================================================

// SigmoidBackward_AG implementation
// Saves output. backward: grad * output * (1 - output)
auto SigmoidBackward_AG::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SigmoidBackward_AG::forward should not be called");
}

auto SigmoidBackward_AG::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& output = saved_tensors_[0];  // sigmoid output
    // grad * output * (1 - output)
    auto one_minus_out = sub(ones(std::vector<int64_t>(output.shape().begin(), output.shape().end()),
                                  output.dtype(), output.device()), output);
    return {mul(grad, mul(output, one_minus_out))};
}

auto SigmoidBackward_AG::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable output_var(saved_tensors_[0], false);
    auto one_tensor = ones(std::vector<int64_t>(saved_tensors_[0].shape().begin(), saved_tensors_[0].shape().end()),
                           saved_tensors_[0].dtype(), saved_tensors_[0].device());
    Variable one_var(one_tensor, false);
    return {grad_outputs[0] * output_var * (one_var - output_var)};
}

// TanhBackward_AG implementation
// Saves output. backward: grad * (1 - output * output)
auto TanhBackward_AG::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TanhBackward_AG::forward should not be called");
}

auto TanhBackward_AG::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& output = saved_tensors_[0];  // tanh output
    // grad * (1 - output^2)
    auto out_sq = mul(output, output);
    auto one_minus_sq = sub(ones(std::vector<int64_t>(output.shape().begin(), output.shape().end()),
                                  output.dtype(), output.device()), out_sq);
    return {mul(grad, one_minus_sq)};
}

auto TanhBackward_AG::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable output_var(saved_tensors_[0], false);
    auto one_tensor = ones(std::vector<int64_t>(saved_tensors_[0].shape().begin(), saved_tensors_[0].shape().end()),
                           saved_tensors_[0].dtype(), saved_tensors_[0].device());
    Variable one_var(one_tensor, false);
    return {grad_outputs[0] * (one_var - output_var * output_var)};
}

// GeluBackward implementation
// Saves input. backward: grad * (0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2))
auto GeluBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("GeluBackward::forward should not be called");
}

auto GeluBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Constants
    constexpr double sqrt2 = 1.4142135623730951;
    constexpr double inv_sqrt_2pi = 0.3989422804014327;  // 1/sqrt(2*pi)

    // cdf = 0.5 * (1 + erf(x / sqrt(2)))
    auto x_over_sqrt2 = mul(input, 1.0 / sqrt2);
    auto erf_val = erf(x_over_sqrt2);
    auto cdf = mul(add(erf_val, 1.0), 0.5);

    // pdf = (1/sqrt(2*pi)) * exp(-x^2/2)
    auto x_sq = mul(input, input);
    auto neg_half_x_sq = mul(x_sq, -0.5);
    auto pdf = mul(exp(neg_half_x_sq), inv_sqrt_2pi);

    // grad_input = grad * (cdf + x * pdf)
    auto x_times_pdf = mul(input, pdf);
    auto result = mul(grad, add(cdf, x_times_pdf));
    return {result};
}

auto GeluBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// EluBackward implementation
// Saves input and alpha (as saved_tensors_[1]). backward: grad * where(input > 0, 1, alpha * exp(input))
auto EluBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("EluBackward::forward should not be called");
}

auto EluBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& alpha_tensor = saved_tensors_[1];  // scalar tensor holding alpha
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Extract alpha value
    float alpha_val = alpha_tensor.data<float>()[0];

    // mask = input > 0
    auto zero_tensor = zeros(shape_vec, input.dtype(), input.device());
    auto mask = gt(input, zero_tensor);

    // positive path: gradient is 1
    auto ones_tensor = ones(shape_vec, input.dtype(), input.device());

    // negative path: gradient is alpha * exp(input)
    auto neg_grad = mul(exp(input), static_cast<double>(alpha_val));

    // where(input > 0, 1, alpha * exp(input))
    auto grad_factor = where(mask, ones_tensor, neg_grad);

    return {mul(grad, grad_factor)};
}

auto EluBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// SeluBackward implementation
// Saves input. lambda=1.0507, alpha=1.6733. backward: grad * where(input > 0, lambda, lambda * alpha * exp(input))
auto SeluBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SeluBackward::forward should not be called");
}

auto SeluBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    constexpr double lambda = 1.0507009873554804934193349852946;
    constexpr double alpha = 1.6732632423543772848170429916717;

    // mask = input > 0
    auto zero_tensor = zeros(shape_vec, input.dtype(), input.device());
    auto mask = gt(input, zero_tensor);

    // positive path: lambda
    auto pos_grad = full(shape_vec, lambda, input.dtype(), input.device());

    // negative path: lambda * alpha * exp(input)
    auto neg_grad = mul(exp(input), lambda * alpha);

    auto grad_factor = where(mask, pos_grad, neg_grad);
    return {mul(grad, grad_factor)};
}

auto SeluBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// MishBackward implementation
// Saves input. backward: grad * (tanh(sp) + x * sigmoid(x) * (1 - tanh(sp)^2)) where sp = softplus(x) = log(1 + exp(x))
auto MishBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("MishBackward::forward should not be called");
}

auto MishBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    // softplus(x) = log(1 + exp(x))
    auto exp_x = exp(input);
    auto sp = log(add(exp_x, 1.0));  // log(1 + exp(x))

    // tanh_sp = tanh(sp)
    auto tanh_sp = tanh(sp);

    // sigmoid(x) = 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))
    auto sig_x = sigmoid(input);

    // 1 - tanh(sp)^2
    auto tanh_sp_sq = mul(tanh_sp, tanh_sp);
    auto one_minus_tanh_sq = sub(ones(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                                       input.dtype(), input.device()), tanh_sp_sq);

    // x * sigmoid(x) * (1 - tanh(sp)^2)
    auto second_term = mul(mul(input, sig_x), one_minus_tanh_sq);

    // grad * (tanh_sp + second_term)
    return {mul(grad, add(tanh_sp, second_term))};
}

auto MishBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// LeakyReluBackward implementation
// Saves input and negative_slope. backward: grad * where(input > 0, 1, negative_slope)
auto LeakyReluBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LeakyReluBackward::forward should not be called");
}

auto LeakyReluBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& slope_tensor = saved_tensors_[1];  // scalar tensor holding negative_slope
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    float slope_val = slope_tensor.data<float>()[0];

    auto zero_tensor = zeros(shape_vec, input.dtype(), input.device());
    auto mask = gt(input, zero_tensor);

    auto ones_tensor = ones(shape_vec, input.dtype(), input.device());
    auto slope_full = full(shape_vec, static_cast<double>(slope_val), input.dtype(), input.device());

    auto grad_factor = where(mask, ones_tensor, slope_full);
    return {mul(grad, grad_factor)};
}

auto LeakyReluBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// SoftplusBackward implementation
// Saves input and beta. backward: grad * sigmoid(beta * input)
auto SoftplusBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SoftplusBackward::forward should not be called");
}

auto SoftplusBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& beta_tensor = saved_tensors_[1];  // scalar tensor holding beta

    float beta_val = beta_tensor.data<float>()[0];

    // sigmoid(beta * input)
    auto beta_x = mul(input, static_cast<double>(beta_val));
    auto sig = sigmoid(beta_x);

    return {mul(grad, sig)};
}

auto SoftplusBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// =========================================================================
// Element-wise Math Backward Functions
// =========================================================================

// SqrtBackward implementation
// Saves output. backward: grad / (2 * output)
auto SqrtBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SqrtBackward::forward should not be called");
}

auto SqrtBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& output = saved_tensors_[0];  // sqrt(x)
    // grad / (2 * output)
    auto two_output = mul(output, 2.0);
    return {div(grad, two_output)};
}

auto SqrtBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable output_var(saved_tensors_[0], false);
    auto two_output = output_var * 2.0;
    return {grad_outputs[0] / two_output};
}

// PowBackward implementation
// Saves input and exponent. backward: grad * exponent * pow(input, exponent - 1)
auto PowBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("PowBackward::forward should not be called");
}

auto PowBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& exp_tensor = saved_tensors_[1];  // scalar tensor holding exponent

    float exp_val = exp_tensor.data<float>()[0];

    // grad * exponent * pow(input, exponent - 1)
    auto pow_term = pow(input, exp_val - 1.0f);
    auto scaled = mul(pow_term, static_cast<double>(exp_val));
    return {mul(grad, scaled)};
}

auto PowBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ReciprocalBackward implementation
// Saves output. backward: grad * (-output * output)
auto ReciprocalBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ReciprocalBackward::forward should not be called");
}

auto ReciprocalBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& output = saved_tensors_[0];  // 1/x
    // grad * (-output^2)
    auto neg_out_sq = neg(mul(output, output));
    return {mul(grad, neg_out_sq)};
}

auto ReciprocalBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable output_var(saved_tensors_[0], false);
    return {grad_outputs[0] * tenzor::neg(output_var * output_var)};
}

// SinBackward implementation
// Saves input. backward: grad * cos(input)
auto SinBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SinBackward::forward should not be called");
}

auto SinBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    return {mul(grad, cos(input))};
}

auto SinBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable input_var(saved_tensors_[0], false);
    auto cos_val = Variable(cos(saved_tensors_[0]), false);
    return {grad_outputs[0] * cos_val};
}

// CosBackward implementation
// Saves input. backward: grad * (-sin(input))
auto CosBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CosBackward::forward should not be called");
}

auto CosBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    return {mul(grad, neg(sin(input)))};
}

auto CosBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto neg_sin_val = Variable(neg(sin(saved_tensors_[0])), false);
    return {grad_outputs[0] * neg_sin_val};
}

// TanBackward implementation
// Saves output. backward: grad * (1 + output * output)
auto TanBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TanBackward::forward should not be called");
}

auto TanBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& output = saved_tensors_[0];  // tan(x)
    // 1 + tan^2(x) = sec^2(x)
    auto out_sq = mul(output, output);
    auto sec_sq = add(out_sq, 1.0);
    return {mul(grad, sec_sq)};
}

auto TanBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable output_var(saved_tensors_[0], false);
    auto sec_sq = Variable(add(mul(saved_tensors_[0], saved_tensors_[0]), 1.0), false);
    return {grad_outputs[0] * sec_sq};
}

// AsinBackward implementation
// Saves input. backward: grad / sqrt(1 - input * input)
auto AsinBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("AsinBackward::forward should not be called");
}

auto AsinBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    // 1 / sqrt(1 - x^2)
    auto x_sq = mul(input, input);
    auto one_minus_sq = sub(ones(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                                  input.dtype(), input.device()), x_sq);
    auto denom = sqrt(one_minus_sq);
    return {div(grad, denom)};
}

auto AsinBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// AcosBackward implementation
// Saves input. backward: -grad / sqrt(1 - input * input)
auto AcosBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("AcosBackward::forward should not be called");
}

auto AcosBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    // -1 / sqrt(1 - x^2)
    auto x_sq = mul(input, input);
    auto one_minus_sq = sub(ones(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                                  input.dtype(), input.device()), x_sq);
    auto denom = sqrt(one_minus_sq);
    return {neg(div(grad, denom))};
}

auto AcosBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// AtanBackward implementation
// Saves input. backward: grad / (1 + input * input)
auto AtanBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("AtanBackward::forward should not be called");
}

auto AtanBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    // 1 / (1 + x^2)
    auto x_sq = mul(input, input);
    auto denom = add(x_sq, 1.0);
    return {div(grad, denom)};
}

auto AtanBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable input_var(saved_tensors_[0], false);
    auto denom = Variable(add(mul(saved_tensors_[0], saved_tensors_[0]), 1.0), false);
    return {grad_outputs[0] / denom};
}

// SinhBackward implementation
// Saves input. backward: grad * cosh(input)
auto SinhBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SinhBackward::forward should not be called");
}

auto SinhBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    return {mul(grad, cosh(input))};
}

auto SinhBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto cosh_val = Variable(cosh(saved_tensors_[0]), false);
    return {grad_outputs[0] * cosh_val};
}

// CoshBackward implementation
// Saves input. backward: grad * sinh(input)
auto CoshBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CoshBackward::forward should not be called");
}

auto CoshBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    return {mul(grad, sinh(input))};
}

auto CoshBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto sinh_val = Variable(sinh(saved_tensors_[0]), false);
    return {grad_outputs[0] * sinh_val};
}

// =========================================================================
// Extended Math Backward Functions
// =========================================================================

// ErfBackward implementation
// Saves input. backward: grad * (2/sqrt(pi)) * exp(-input^2)
auto ErfBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ErfBackward::forward should not be called");
}

auto ErfBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    constexpr double two_over_sqrt_pi = 1.1283791670955126;  // 2/sqrt(pi)

    auto neg_x_sq = neg(mul(input, input));
    auto exp_term = exp(neg_x_sq);
    auto factor = mul(exp_term, two_over_sqrt_pi);
    return {mul(grad, factor)};
}

auto ErfBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ErfcBackward implementation
// Saves input. backward: grad * (-2/sqrt(pi)) * exp(-input^2)
auto ErfcBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ErfcBackward::forward should not be called");
}

auto ErfcBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    constexpr double neg_two_over_sqrt_pi = -1.1283791670955126;  // -2/sqrt(pi)

    auto neg_x_sq = neg(mul(input, input));
    auto exp_term = exp(neg_x_sq);
    auto factor = mul(exp_term, neg_two_over_sqrt_pi);
    return {mul(grad, factor)};
}

auto ErfcBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// Log2Backward implementation
// Saves input. backward: grad / (input * log(2))
auto Log2Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("Log2Backward::forward should not be called");
}

auto Log2Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    constexpr double ln2 = 0.6931471805599453;  // log(2)

    auto denom = mul(input, ln2);
    return {div(grad, denom)};
}

auto Log2Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    constexpr double ln2 = 0.6931471805599453;
    Variable input_var(saved_tensors_[0], false);
    return {grad_outputs[0] / (input_var * ln2)};
}

// Log10Backward implementation
// Saves input. backward: grad / (input * log(10))
auto Log10Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("Log10Backward::forward should not be called");
}

auto Log10Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    constexpr double ln10 = 2.302585092994046;  // log(10)

    auto denom = mul(input, ln10);
    return {div(grad, denom)};
}

auto Log10Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    constexpr double ln10 = 2.302585092994046;
    Variable input_var(saved_tensors_[0], false);
    return {grad_outputs[0] / (input_var * ln10)};
}

// Log1pBackward implementation
// Saves input. backward: grad / (1 + input)
auto Log1pBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("Log1pBackward::forward should not be called");
}

auto Log1pBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    auto denom = add(input, 1.0);
    return {div(grad, denom)};
}

auto Log1pBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable input_var(saved_tensors_[0], false);
    auto denom = Variable(add(saved_tensors_[0], 1.0), false);
    return {grad_outputs[0] / denom};
}

// Exp2Backward implementation
// Saves output. backward: grad * output * log(2)
auto Exp2Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("Exp2Backward::forward should not be called");
}

auto Exp2Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& output = saved_tensors_[0];  // 2^x

    constexpr double ln2 = 0.6931471805599453;  // log(2)

    auto factor = mul(output, ln2);
    return {mul(grad, factor)};
}

auto Exp2Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    constexpr double ln2 = 0.6931471805599453;
    Variable output_var(saved_tensors_[0], false);
    return {grad_outputs[0] * (output_var * ln2)};
}

// Expm1Backward implementation
// Saves input. backward: grad * exp(input)
auto Expm1Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("Expm1Backward::forward should not be called");
}

auto Expm1Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    return {mul(grad, exp(input))};
}

auto Expm1Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto exp_val = Variable(exp(saved_tensors_[0]), false);
    return {grad_outputs[0] * exp_val};
}

// Atan2Backward implementation
// Saves inputs (y, x). backward: grad_y = grad * x / (x^2 + y^2), grad_x = grad * (-y) / (x^2 + y^2)
auto Atan2Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("Atan2Backward::forward should not be called");
}

auto Atan2Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& y = saved_tensors_[0];
    const auto& x = saved_tensors_[1];

    // denom = x^2 + y^2
    auto denom = add(mul(x, x), mul(y, y));

    // grad_y = grad * x / denom
    auto grad_y = div(mul(grad, x), denom);

    // grad_x = grad * (-y) / denom
    auto grad_x = div(mul(grad, neg(y)), denom);

    return {grad_y, grad_x};
}

auto Atan2Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable y_var(saved_tensors_[0], false);
    Variable x_var(saved_tensors_[1], false);
    auto denom = Variable(add(mul(saved_tensors_[1], saved_tensors_[1]),
                              mul(saved_tensors_[0], saved_tensors_[0])), false);
    auto grad_y = grad_outputs[0] * x_var / denom;
    auto grad_x = grad_outputs[0] * tenzor::neg(y_var) / denom;
    return {grad_y, grad_x};
}

// =========================================================================
// Reduction Backward Functions
// =========================================================================

// MinBackward implementation
// Same pattern as MaxBackward. Save input+output. backward: mask where input == min_val
auto MinBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("MinBackward::forward should not be called");
}

auto MinBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];  // min values
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "MinBackward: cannot compute gradient of min over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // The saved_tensors_[2] holds dim as a scalar Int64 tensor (or not present for global min)
    bool has_dim = saved_tensors_.size() > 2;

    if (!has_dim) {
        // Global min: gradient flows only to the minimum element
        auto output_reshaped = output;
        if (output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            output_reshaped = reshape(output, ones_shape);
        }
        auto output_expanded = expand(output_reshaped, input_shape_vec);

        // Create mask where input == output (within epsilon)
        auto diff = sub(input, output_expanded);
        auto abs_diff = abs(diff);
        double eps_val;
        switch (input.dtype()) {
            case DType::Float64:  eps_val = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val = 1e-3; break;
            default:              eps_val = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val, input.dtype(), input.device());
        auto mask_bool = lt(abs_diff, epsilon);
        auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
        auto zeros_tensor = zeros(input_shape_vec, input.dtype(), input.device());
        auto mask = where(mask_bool, ones_tensor, zeros_tensor);

        // Normalize mask by tie count
        auto tie_count = sum(mask);
        mask = div(mask, tie_count);

        // Broadcast grad_output to input shape
        auto grad_reshaped = grad_output;
        if (grad_output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad_reshaped = reshape(grad_output, ones_shape);
        }
        auto grad_broadcasted = expand(grad_reshaped, input_shape_vec);

        return {mul(grad_broadcasted, mask)};
    } else {
        // Dimension-specific min
        int64_t dim = saved_tensors_[2].data<int64_t>()[0];
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        auto out = output;

        // Check if keepdim was used by comparing shapes
        bool keepdim = (output.ndim() == input.ndim());

        if (!keepdim) {
            grad = unsqueeze(grad, dim);
            out = unsqueeze(out, dim);
        }

        // Expand to input shape
        auto out_expanded = expand(out, input_shape_vec);
        auto grad_expanded = expand(grad, input_shape_vec);

        // Create mask where input == min_value
        auto diff = sub(input, out_expanded);
        auto abs_diff = abs(diff);
        double eps_val2;
        switch (input.dtype()) {
            case DType::Float64:  eps_val2 = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val2 = 1e-3; break;
            default:              eps_val2 = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val2, input.dtype(), input.device());
        auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
        auto scaled_diff = div(abs_diff, epsilon);
        auto clamped = clamp(scaled_diff, 0.0f, 1.0f);
        auto mask = sub(ones_tensor, clamped);

        // Normalize mask by tie count along dim
        auto tie_count = sum(mask, dim, /*keepdim=*/true);
        mask = div(mask, tie_count);

        return {mul(grad_expanded, mask)};
    }
}

auto MinBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// StdBackward implementation
// Saves input and output. backward: grad * (input - mean) / (N * output)
auto StdBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("StdBackward::forward should not be called");
}

auto StdBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& std_out = saved_tensors_[1];  // std(x)

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Determine dim and N from saved_tensors_[2] if present
    bool has_dim = saved_tensors_.size() > 2;
    std::optional<int64_t> dim_opt;
    int64_t N;
    bool keepdim;

    if (has_dim) {
        int64_t dim = saved_tensors_[2].data<int64_t>()[0];
        if (dim < 0) dim += input.shape().size();
        dim_opt = dim;
        N = input.shape()[dim];
        keepdim = (std_out.ndim() == input.ndim());
    } else {
        N = input.numel();
        keepdim = false;
    }

    // Compute mean of input
    auto input_mean = mean(input, dim_opt, true);

    // (input - mean)
    auto diff = sub(input, expand(input_mean, input_shape_vec));

    // Expand std and grad to input shape
    auto std_expanded = std_out;
    auto grad_expanded = grad;
    if (dim_opt.has_value() && !keepdim) {
        std_expanded = unsqueeze(std_out, dim_opt.value());
        grad_expanded = unsqueeze(grad, dim_opt.value());
    } else if (!dim_opt.has_value()) {
        if (std_out.ndim() > 0) {
            std_expanded = reshape(std_out, std::vector<int64_t>(input_shape_vec.size(), 1));
        } else {
            std_expanded = reshape(std_out, std::vector<int64_t>(input_shape_vec.size(), 1));
        }
        if (grad.ndim() > 0) {
            grad_expanded = reshape(grad, std::vector<int64_t>(input_shape_vec.size(), 1));
        } else {
            grad_expanded = reshape(grad, std::vector<int64_t>(input_shape_vec.size(), 1));
        }
    }

    std_expanded = expand(std_expanded, input_shape_vec);
    grad_expanded = expand(grad_expanded, input_shape_vec);

    // grad_input = grad * (input - mean) / (N * std)
    auto n_std = mul(std_expanded, static_cast<double>(N));
    auto grad_input = div(mul(grad_expanded, diff), n_std);

    return {grad_input};
}

auto StdBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// VarBackward implementation
// Saves input. backward: grad * 2 * (input - mean) / N
auto VarBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("VarBackward::forward should not be called");
}

auto VarBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& var_out = saved_tensors_[1];

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Determine dim and N from saved_tensors_[2] if present
    bool has_dim = saved_tensors_.size() > 2;
    std::optional<int64_t> dim_opt;
    int64_t N;
    bool keepdim;

    if (has_dim) {
        int64_t dim = saved_tensors_[2].data<int64_t>()[0];
        if (dim < 0) dim += input.shape().size();
        dim_opt = dim;
        N = input.shape()[dim];
        keepdim = (var_out.ndim() == input.ndim());
    } else {
        N = input.numel();
        keepdim = false;
    }

    // Compute mean of input
    auto input_mean = mean(input, dim_opt, true);

    // (input - mean)
    auto diff = sub(input, expand(input_mean, input_shape_vec));

    // Expand grad to input shape
    auto grad_expanded = grad;
    if (dim_opt.has_value() && !keepdim) {
        grad_expanded = unsqueeze(grad, dim_opt.value());
    } else if (!dim_opt.has_value()) {
        if (grad.ndim() > 0) {
            grad_expanded = reshape(grad, std::vector<int64_t>(input_shape_vec.size(), 1));
        } else {
            grad_expanded = reshape(grad, std::vector<int64_t>(input_shape_vec.size(), 1));
        }
    }
    grad_expanded = expand(grad_expanded, input_shape_vec);

    // grad_input = grad * 2 * (input - mean) / N
    auto scale = 2.0 / static_cast<double>(N);
    auto grad_input = mul(mul(grad_expanded, diff), scale);

    return {grad_input};
}

auto VarBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ProdBackward implementation
// Saves input and output. backward: grad * output / input (with zero handling)
auto ProdBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ProdBackward::forward should not be called");
}

auto ProdBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& prod_out = saved_tensors_[1];

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    bool has_dim = saved_tensors_.size() > 2;
    std::optional<int64_t> dim_opt;
    bool keepdim;

    if (has_dim) {
        int64_t dim = saved_tensors_[2].data<int64_t>()[0];
        if (dim < 0) dim += input.shape().size();
        dim_opt = dim;
        keepdim = (prod_out.ndim() == input.ndim());
    } else {
        keepdim = false;
    }

    // Expand prod_out and grad to input shape
    auto prod_expanded = prod_out;
    auto grad_expanded = grad;
    if (dim_opt.has_value() && !keepdim) {
        prod_expanded = unsqueeze(prod_out, dim_opt.value());
        grad_expanded = unsqueeze(grad, dim_opt.value());
    } else if (!dim_opt.has_value()) {
        prod_expanded = reshape(prod_out, std::vector<int64_t>(input_shape_vec.size(), 1));
        grad_expanded = reshape(grad, std::vector<int64_t>(input_shape_vec.size(), 1));
    }
    prod_expanded = expand(prod_expanded, input_shape_vec);
    grad_expanded = expand(grad_expanded, input_shape_vec);

    // grad_input = grad * prod / input
    // Handle zeros: where input == 0, the gradient is the product of all other elements
    // For simplicity, use the formula: grad * prod / input, with input clamped away from zero
    auto eps_val = full(input_shape_vec, 1e-12, input.dtype(), input.device());
    auto zero_tensor = zeros(input_shape_vec, input.dtype(), input.device());
    auto mask_zero = eq(input, zero_tensor);

    // Replace zeros with ones for safe division
    auto safe_input = where(mask_zero, ones(input_shape_vec, input.dtype(), input.device()), input);
    auto grad_input = mul(grad_expanded, div(prod_expanded, safe_input));

    // For zero elements, need to compute product of non-zero elements
    // This is expensive; for now use the approximation which is correct when at most one zero exists
    return {grad_input};
}

auto ProdBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// LogSumExpBackward implementation
// Saves input and output. backward: grad * softmax(input, dim)
auto LogSumExpBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LogSumExpBackward::forward should not be called");
}

auto LogSumExpBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& lse_out = saved_tensors_[1];

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Determine dim from saved_tensors_[2]
    int64_t dim = saved_tensors_[2].data<int64_t>()[0];
    if (dim < 0) dim += input.shape().size();

    bool keepdim = (lse_out.ndim() == input.ndim());

    // softmax(input, dim) = exp(input - logsumexp(input, dim))
    auto lse_expanded = lse_out;
    auto grad_expanded = grad;
    if (!keepdim) {
        lse_expanded = unsqueeze(lse_out, dim);
        grad_expanded = unsqueeze(grad, dim);
    }
    lse_expanded = expand(lse_expanded, input_shape_vec);
    grad_expanded = expand(grad_expanded, input_shape_vec);

    auto softmax_val = exp(sub(input, lse_expanded));

    return {mul(grad_expanded, softmax_val)};
}

auto LogSumExpBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// =========================================================================
// Shape/Indexing Backward Functions
// =========================================================================

// UnsqueezeBackward implementation
// Saves dim. backward: squeeze(grad, dim)
auto UnsqueezeBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("UnsqueezeBackward::forward should not be called");
}

auto UnsqueezeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    int64_t dim = saved_tensors_[0].data<int64_t>()[0];
    return {squeeze(grad, dim)};
}

auto UnsqueezeBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    int64_t dim = saved_tensors_[0].data<int64_t>()[0];
    return {tenzor::squeeze(grad_outputs[0], dim)};
}

// ExpandBackward implementation
// Saves original shape. backward: sum_to(grad, original_shape)
auto ExpandBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ExpandBackward::forward should not be called");
}

auto ExpandBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // Original shape is saved in saved_tensors_[0] as a 1D Int64 tensor
    const auto& shape_tensor = saved_tensors_[0];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto original_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    return {reduce_grad_for_broadcasting(grad, original_shape)};
}

auto ExpandBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    const auto& shape_tensor = saved_tensors_[0];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto original_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    return {reduce_grad_var_for_broadcasting(grad_outputs[0], original_shape)};
}

// FlattenBackward implementation
// Saves original shape. backward: reshape(grad, original_shape)
auto FlattenBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("FlattenBackward::forward should not be called");
}

auto FlattenBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // Original shape is saved in saved_tensors_[0] as a 1D Int64 tensor
    const auto& shape_tensor = saved_tensors_[0];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto original_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    return {reshape(grad, original_shape)};
}

auto FlattenBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    const auto& shape_tensor = saved_tensors_[0];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto original_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    return {tenzor::reshape(grad_outputs[0], original_shape)};
}

// WhereBackward implementation
// Saves condition. backward: grad_x = grad * condition, grad_y = grad * !condition
auto WhereBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("WhereBackward::forward should not be called");
}

auto WhereBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& condition = saved_tensors_[0];  // Bool tensor

    auto shape_vec = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
    auto zeros_tensor = zeros(shape_vec, grad.dtype(), grad.device());

    // grad_x = where(condition, grad, 0)
    auto grad_x = where(condition, grad, zeros_tensor);

    // grad_y = where(condition, 0, grad) = where(!condition, grad, 0)
    auto grad_y = where(condition, zeros_tensor, grad);

    return {grad_x, grad_y};
}

auto WhereBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad()),
            Variable(result[1], grad_outputs[0].requires_grad())};
}

// GatherBackward implementation
// Saves dim, index, input_shape. backward: scatter_add(zeros(input_shape), dim, index, grad)
auto GatherBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("GatherBackward::forward should not be called");
}

auto GatherBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // saved_tensors_[0] = dim (scalar Int64)
    // saved_tensors_[1] = index
    // saved_tensors_[2] = input shape (1D Int64)
    int64_t dim = saved_tensors_[0].data<int64_t>()[0];
    const auto& index = saved_tensors_[1];
    const auto& shape_tensor = saved_tensors_[2];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto input_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    // scatter_add zeros with grad at index positions
    auto grad_input = zeros(input_shape, grad.dtype(), grad.device());
    grad_input = scatter_add(grad_input, dim, index, grad);

    return {grad_input};
}

auto GatherBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ScatterBackward implementation
// Saves dim, index. backward: grad_input = scatter(grad, dim, index, zeros), grad_src = gather(grad, dim, index)
auto ScatterBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ScatterBackward::forward should not be called");
}

auto ScatterBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // saved_tensors_[0] = dim (scalar Int64)
    // saved_tensors_[1] = index
    int64_t dim = saved_tensors_[0].data<int64_t>()[0];
    const auto& index = saved_tensors_[1];

    // grad_input: zero out the scattered positions
    auto index_shape_vec = std::vector<int64_t>(index.shape().begin(), index.shape().end());
    auto zeros_src = zeros(index_shape_vec, grad.dtype(), grad.device());
    auto grad_input = scatter(grad, dim, index, zeros_src);

    // grad_src: gather from grad at the index positions
    auto grad_src = gather(grad, dim, index);

    return {grad_input, grad_src};
}

auto ScatterBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad()),
            Variable(result[1], grad_outputs[0].requires_grad())};
}

// IndexSelectBackward implementation
// Saves dim, index, input_shape. backward: create zeros, scatter_add grad at index positions
auto IndexSelectBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IndexSelectBackward::forward should not be called");
}

auto IndexSelectBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // saved_tensors_[0] = dim (scalar Int64)
    // saved_tensors_[1] = index (1D Int64)
    // saved_tensors_[2] = input shape (1D Int64)
    int64_t dim = saved_tensors_[0].data<int64_t>()[0];
    const auto& index = saved_tensors_[1];
    const auto& shape_tensor = saved_tensors_[2];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto input_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    // Create zeros of input shape
    auto grad_input = zeros(input_shape, grad.dtype(), grad.device());

    // Build a full index tensor matching grad shape for scatter_add along dim
    auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
    auto full_index = zeros(grad_shape, DType::Int64, Device::cpu());
    auto* idx_ptr = full_index.data<int64_t>();
    auto* src_idx_ptr = index.to(Device::cpu()).data<int64_t>();

    int64_t total = grad.numel();
    int64_t dim_size = grad_shape[dim];
    int64_t dim_stride = 1;
    for (int64_t d = dim + 1; d < static_cast<int64_t>(grad_shape.size()); ++d) {
        dim_stride *= grad_shape[d];
    }

    for (int64_t i = 0; i < total; ++i) {
        int64_t pos_in_dim = (i / dim_stride) % dim_size;
        idx_ptr[i] = src_idx_ptr[pos_in_dim];
    }

    if (grad.device() != Device::cpu()) {
        full_index = full_index.to(grad.device());
    }

    grad_input = scatter_add(grad_input, dim, full_index, grad);

    return {grad_input};
}

auto IndexSelectBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// NarrowBackward implementation
// Saves dim, start, original_shape. backward: zero-pad grad to original shape
auto NarrowBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("NarrowBackward::forward should not be called");
}

auto NarrowBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // saved_tensors_[0] = dim (scalar Int64)
    // saved_tensors_[1] = start (scalar Int64)
    // saved_tensors_[2] = original shape (1D Int64)
    int64_t dim = saved_tensors_[0].data<int64_t>()[0];
    int64_t start = saved_tensors_[1].data<int64_t>()[0];
    const auto& shape_tensor = saved_tensors_[2];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto original_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    // Create zero tensor of original shape
    auto grad_input = zeros(original_shape, grad.dtype(), grad.device());

    // Use scatter to place grad values at the correct positions
    // Build index tensor
    auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
    int64_t narrow_len = grad_shape[dim];
    auto index = zeros(grad_shape, DType::Int64, Device::cpu());
    auto* idx_ptr = index.data<int64_t>();
    int64_t total = grad.numel();
    int64_t dim_stride = 1;
    for (int64_t d = dim + 1; d < static_cast<int64_t>(grad_shape.size()); ++d) {
        dim_stride *= grad_shape[d];
    }

    for (int64_t i = 0; i < total; ++i) {
        int64_t pos_in_dim = (i / dim_stride) % narrow_len;
        idx_ptr[i] = start + pos_in_dim;
    }

    if (grad.device() != Device::cpu()) {
        index = index.to(grad.device());
    }

    grad_input = scatter(grad_input, dim, index, grad);

    return {grad_input};
}

auto NarrowBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// FlipBackward implementation
// Saves dims. backward: flip(grad, dims)
auto FlipBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("FlipBackward::forward should not be called");
}

auto FlipBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // saved_tensors_[0] holds the dims as a 1D Int64 tensor
    const auto& dims_tensor = saved_tensors_[0];
    auto dims_ptr = dims_tensor.data<int64_t>();
    auto dims = std::vector<int64_t>(dims_ptr, dims_ptr + dims_tensor.numel());
    return {flip(grad, dims)};
}

auto FlipBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// RepeatBackward implementation
// Saves original_shape and repeats. backward: sum grad over repeated dimensions
auto RepeatBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("RepeatBackward::forward should not be called");
}

auto RepeatBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // saved_tensors_[0] = original shape (1D Int64)
    // saved_tensors_[1] = repeats (1D Int64)
    const auto& shape_tensor = saved_tensors_[0];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto original_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    const auto& repeats_tensor = saved_tensors_[1];
    auto repeats_ptr = repeats_tensor.data<int64_t>();
    auto repeats = std::vector<int64_t>(repeats_ptr, repeats_ptr + repeats_tensor.numel());

    // To compute gradient: reshape grad so each repeated dimension is split into
    // (repeat_count, original_dim_size), then sum over the repeat_count dimension.
    //
    // For each dimension i:
    //   grad_shape[i] = repeats[i] * original_shape[i]
    // We reshape to interleave repeat and original dims, then sum.

    auto ndim = original_shape.size();

    // Build reshape: [repeats[0], orig[0], repeats[1], orig[1], ...]
    std::vector<int64_t> expanded_shape;
    expanded_shape.reserve(2 * ndim);
    for (size_t i = 0; i < ndim; ++i) {
        expanded_shape.push_back(repeats[i]);
        expanded_shape.push_back(original_shape[i]);
    }

    auto grad_reshaped = reshape(grad, expanded_shape);

    // Sum over the repeat dimensions (dims 0, 2, 4, ...)
    // We need to sum from the highest dim first to avoid shifting indices
    auto result = grad_reshaped;
    for (int64_t i = static_cast<int64_t>(ndim) - 1; i >= 0; --i) {
        int64_t repeat_dim = 2 * i;  // The repeat count dimension
        result = tenzor::sum(result, repeat_dim, false);
    }

    return {result};
}

auto RepeatBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// =========================================================================
// Linear Algebra Backward Functions
// =========================================================================

// DetBackward implementation
// Forward: y = det(A)
// Backward: dL/dA = dL/dy * det(A) * A^{-T}
auto DetBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("DetBackward::forward should not be called directly");
}

auto DetBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];          // dL/dy, scalar or (...,)
    const auto& det_val = saved_tensors_[0];      // det(A), same shape as grad
    const auto& inv_A = saved_tensors_[1];        // A^{-1}, (..., N, N)

    auto ndim = inv_A.ndim();
    // A^{-T} = transpose of inverse
    auto inv_At = transpose(inv_A, ndim - 2, ndim - 1);

    // grad * det(A) is scalar (per batch), need to broadcast to matrix shape
    auto grad_det = mul(grad, det_val);  // (...,)

    // Reshape grad_det to broadcast with inv_At: add two trailing dims
    auto gd_shape = std::vector<int64_t>(grad_det.shape().begin(), grad_det.shape().end());
    gd_shape.push_back(1);
    gd_shape.push_back(1);
    auto grad_det_expanded = reshape(grad_det, gd_shape);

    auto grad_A = mul(grad_det_expanded, inv_At);
    return {grad_A};
}

// InvBackward implementation
// Forward: Y = A^{-1}
// Backward: dL/dA = -Y^T @ dL/dY @ Y^T
auto InvBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("InvBackward::forward should not be called directly");
}

auto InvBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];       // dL/dY, (..., N, N)
    const auto& inv_A = saved_tensors_[0];    // Y = A^{-1}, (..., N, N)

    auto ndim = inv_A.ndim();
    auto inv_At = transpose(inv_A, ndim - 2, ndim - 1);

    // -Y^T @ dL/dY @ Y^T
    auto temp = matmul(inv_At, grad);
    auto result = matmul(temp, inv_At);
    auto grad_A = neg(result);

    return {grad_A};
}

// SolveBackward implementation
// Forward: X = solve(A, B) where AX = B
// Backward:
//   dL/dB = solve(A^T, dL/dX)
//   dL/dA = -dL/dB @ X^T
auto SolveBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SolveBackward::forward should not be called directly");
}

auto SolveBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];    // dL/dX, (..., N, K)
    const auto& A = saved_tensors_[0];     // A, (..., N, N)
    const auto& X = saved_tensors_[1];     // solution X, (..., N, K)

    auto ndim = A.ndim();
    auto At = transpose(A, ndim - 2, ndim - 1);

    // dL/dB = solve(A^T, dL/dX)
    auto grad_B = tenzor::linalg::solve(At, grad);

    // dL/dA = -grad_B @ X^T
    auto x_ndim = X.ndim();
    auto Xt = transpose(X, x_ndim - 2, x_ndim - 1);
    auto grad_A = neg(matmul(grad_B, Xt));

    return {grad_A, grad_B};
}

// CholeskyBackward implementation
// Forward: L = cholesky(A)  where A = L @ L^T
// Backward: Uses the formula from Murray 2016 / PyTorch:
//   S = L^T @ dL/dL
//   S = tril(S) with diagonal halved: phi(S)
//   dL/dA = L^{-T} @ phi(S + S^T) @ L^{-1}
//   Simplified: dL/dA = solve(L^T, phi(L^T @ grad_L)) then symmetrize
auto CholeskyBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CholeskyBackward::forward should not be called directly");
}

auto CholeskyBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    auto grad_L = grad_outputs[0];          // dL/dL, (..., N, N)
    const auto& L = saved_tensors_[0];      // Cholesky factor, (..., N, N)

    if (upper_) {
        // If upper triangular was returned, transpose to work with lower
        auto ndim = L.ndim();
        grad_L = transpose(grad_L, ndim - 2, ndim - 1);
        // L is actually U, transpose it
        auto L_lower = transpose(L, ndim - 2, ndim - 1);

        auto Lt = transpose(L_lower, ndim - 2, ndim - 1);
        auto S = matmul(Lt, grad_L);

        // phi(S): take lower triangle and halve diagonal
        auto S_tril = tril(S);
        // Halve diagonal: S_tril = S_tril - 0.5 * diag(diag(S_tril)) -- approximate via element ops
        // A simpler approach: S_sym = tril(S) with diagonal * 0.5
        // We use: phi(S) = tril(S, -1) + 0.5 * diag(diag(S))
        auto S_strict_lower = tril(S, -1);
        auto S_diag = mul(tril(S), 0.5);  // This isn't quite right, let's use a different approach

        // Actually: phi(X) = tril(X) but with diagonal halved
        // phi(X) = tril(X, -1) + 0.5 * (tril(X, 0) - tril(X, -1))
        // = tril(X, -1) + 0.5 * diag_matrix(diag(X))
        // Simpler: just use tril(S) and subtract 0.5 * diag elements
        // Let's compute: phi(S) where phi takes tril and halves diagonal
        auto phi_S = add(S_strict_lower, sub(S_tril, S_strict_lower) * 0.5);

        // dL/dA = L^{-T} @ phi_S @ L^{-1}
        // = solve(L^T, phi_S) then solve(L, result^T)^T
        // Simpler: use solve twice
        auto temp = tenzor::linalg::solve(Lt, phi_S);
        auto grad_A = tenzor::linalg::solve(Lt, transpose(temp, ndim - 2, ndim - 1));
        grad_A = transpose(grad_A, ndim - 2, ndim - 1);

        // Symmetrize: dL/dA = 0.5 * (dL/dA + dL/dA^T)
        auto grad_At = transpose(grad_A, ndim - 2, ndim - 1);
        grad_A = mul(add(grad_A, grad_At), 0.5);

        return {grad_A};
    }

    auto ndim = L.ndim();
    auto Lt = transpose(L, ndim - 2, ndim - 1);

    // S = L^T @ grad_L
    auto S = matmul(Lt, grad_L);

    // phi(S): tril(S) with diagonal halved
    auto S_tril = tril(S);
    auto S_strict_lower = tril(S, -1);
    auto phi_S = add(S_strict_lower, mul(sub(S_tril, S_strict_lower), 0.5));

    // dL/dA = L^{-T} @ phi_S @ L^{-1}
    auto temp = tenzor::linalg::solve(Lt, phi_S);
    auto grad_A = tenzor::linalg::solve(Lt, transpose(temp, ndim - 2, ndim - 1));
    grad_A = transpose(grad_A, ndim - 2, ndim - 1);

    // Symmetrize: dL/dA = 0.5 * (dL/dA + dL/dA^T)
    auto grad_At = transpose(grad_A, ndim - 2, ndim - 1);
    grad_A = mul(add(grad_A, grad_At), 0.5);

    return {grad_A};
}

// NormBackward_Linalg implementation
// Forward: y = norm(A, ord)
// Backward (Frobenius): dL/dA = dL/dy * A / norm(A)
auto NormBackward_Linalg::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("NormBackward_Linalg::forward should not be called directly");
}

auto NormBackward_Linalg::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];       // dL/dy, scalar
    const auto& input = saved_tensors_[0];    // A
    const auto& norm_val = saved_tensors_[1]; // norm(A), scalar

    if (ord_ == "fro") {
        // dL/dA = dL/dy * A / norm(A)
        // Reshape grad and norm_val to broadcast with A
        auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());

        // grad / norm_val is a scalar ratio
        auto scale = div(grad, norm_val);

        // Expand scale to match input shape
        auto scale_shape = std::vector<int64_t>(scale.shape().begin(), scale.shape().end());
        while (scale_shape.size() < input_shape.size()) {
            scale_shape.push_back(1);
        }
        auto scale_expanded = reshape(scale, scale_shape);

        auto grad_A = mul(scale_expanded, input);
        return {grad_A};
    }

    // For non-Frobenius norms, return zeros (unsupported)
    auto grad_A = zeros_like(input);
    return {grad_A};
}

// SlogdetBackward implementation
// Forward: (sign, logabsdet) = slogdet(A)
// Backward: dL/dA = dL/d(logabsdet) * A^{-T}
// sign gradient is zero (discrete)
auto SlogdetBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SlogdetBackward::forward should not be called directly");
}

auto SlogdetBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // grad_outputs[0] = dL/d(sign) -- ignored (zero gradient)
    // grad_outputs[1] = dL/d(logabsdet)
    const auto& grad_logabsdet = grad_outputs[1];
    const auto& inv_A = saved_tensors_[0];     // A^{-1}, (..., N, N)

    auto ndim = inv_A.ndim();
    auto inv_At = transpose(inv_A, ndim - 2, ndim - 1);

    // Reshape grad_logabsdet to broadcast with inv_At
    auto gd_shape = std::vector<int64_t>(grad_logabsdet.shape().begin(), grad_logabsdet.shape().end());
    while (gd_shape.size() < static_cast<size_t>(ndim)) {
        gd_shape.push_back(1);
    }
    auto grad_expanded = reshape(grad_logabsdet, gd_shape);

    auto grad_A = mul(grad_expanded, inv_At);
    return {grad_A};
}

// SvdBackward implementation
// Forward: (U, S, Vh) = svd(A)
// Backward: complex formula using F matrix
// For A = U @ diag(S) @ Vh:
//   F_{ij} = 1/(s_j^2 - s_i^2) for i != j, 0 on diagonal
//   dL/dA = U @ (diag(dL/dS) + (F * (U^T @ dL/dU - dL/dVh^T @ Vh^T @ diag(S))) @ ... ) @ Vh
// Simplified approach for thin SVD:
auto SvdBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SvdBackward::forward should not be called directly");
}

auto SvdBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad_U = grad_outputs[0];     // dL/dU, (..., M, K)
    const auto& grad_S = grad_outputs[1];     // dL/dS, (..., K)
    const auto& grad_Vh = grad_outputs[2];    // dL/dVh, (..., K, N)

    const auto& U = saved_tensors_[0];        // U, (..., M, K)
    const auto& S = saved_tensors_[1];        // S, (..., K)
    const auto& Vh = saved_tensors_[2];       // Vh, (..., K, N)

    auto ndim = U.ndim();
    auto K = S.shape()[S.ndim() - 1];

    // Construct diagonal matrix from S
    auto S_diag = diag(S);  // (..., K, K)

    // U^T and V (= Vh^T)
    auto Ut = transpose(U, ndim - 2, ndim - 1);
    auto V = transpose(Vh, Vh.ndim() - 2, Vh.ndim() - 1);

    // Construct F matrix: F_{ij} = 1/(s_j^2 - s_i^2) for i != j
    // F is K x K
    auto S_sq = mul(S, S);  // (..., K)

    // Build F matrix element-wise
    // For simplicity, compute on CPU
    auto s_data = S.contiguous();
    auto f_shape = std::vector<int64_t>(S.shape().begin(), S.shape().end());
    f_shape.back() = K;
    f_shape.push_back(K);

    // Create F as zeros
    // Simplified: we need batch-aware F construction
    // For now, compute F for last two dims
    auto F_tensor = zeros({K, K}, S.dtype(), S.device());

    // Fill F: this requires element access, do it with a simpler approach
    // Use outer products of S
    // s_j^2 - s_i^2 = (s_j - s_i)(s_j + s_i)
    // Reshape S for broadcasting: S_row (1, K), S_col (K, 1)
    auto S_row = reshape(S_sq, {1, K});
    auto S_col = reshape(S_sq, {K, 1});
    auto diffs = sub(S_row, S_col);  // (K, K): diffs[i][j] = s_j^2 - s_i^2

    // Replace zeros on diagonal with 1 to avoid division by zero
    auto mask = eye(K, std::nullopt, S.dtype(), S.device());
    auto safe_diffs = add(diffs, mask);  // diagonal becomes 1 instead of 0

    // F = 1/safe_diffs, then zero out diagonal
    auto F = reciprocal(safe_diffs);
    auto anti_mask = sub(ones({K, K}, S.dtype(), S.device()), mask);
    F = mul(F, anti_mask);  // Zero out diagonal

    // Core SVD backward formula:
    // dL/dA = U @ (diag(dL/dS) + F * (Ut @ dL/dU) @ S_diag + S_diag @ F * (dL/dVh @ V)) @ Vh
    //       + (I - U @ Ut) @ dL/dU @ diag(1/S) @ Vh
    //       + U @ diag(1/S) @ dL/dVh @ (I - V @ Vt)

    // Simplified formula for full-rank case:
    // dA = U @ (diag(grad_S) + F * (Ut @ grad_U - (grad_Vh @ V)^T) @ S_diag) @ Vh
    //    + ... (projector terms for non-square)

    // Compute Ut @ grad_U
    auto UtgU = matmul(Ut, grad_U);  // (K, K)

    // Compute grad_Vh @ V
    auto gVhV = matmul(grad_Vh, V);  // (K, K)
    auto gVhVt = transpose(gVhV, gVhV.ndim() - 2, gVhV.ndim() - 1);

    // Symmetric part: F * (Ut @ grad_U - (grad_Vh @ V)^T) = F * (UtgU - gVhVt)
    auto skew = sub(UtgU, gVhVt);
    auto F_skew = mul(F, skew);  // (K, K)

    // F_skew @ S_diag
    auto term = matmul(F_skew, S_diag);

    // Add S_diag @ F * (gVhVt - UtgU)^T ... actually let's use the symmetric formulation
    // The correct formula from Ionescu et al. 2015:
    // dA = U @ (diag(grad_S) + (F * (Ut@gU)) @ S + S @ (F * (gVh@V))) @ Vh

    auto F_UtgU = mul(F, UtgU);
    auto F_gVhV = mul(F, gVhV);

    auto term1 = diag(grad_S);             // diag(grad_S), (K, K)
    auto term2 = matmul(F_UtgU, S_diag);   // F*(Ut@gU) @ S
    auto term3 = matmul(S_diag, F_gVhV);   // S @ F*(gVh@V)

    auto middle = add(add(term1, term2), term3);  // (K, K)

    auto grad_A = matmul(matmul(U, middle), Vh);  // (..., M, N)

    return {grad_A};
}

// QrBackward implementation
// Forward: (Q, R) = qr(A) where A = Q @ R
// Backward formula (from Seeger et al.):
//   M = R @ grad_R^T - grad_Q^T @ Q
//   copyltu(M) = tril(M) + tril(M, -1)^T   (symmetrize lower triangle)
//   dL/dA = (grad_Q + Q @ copyltu(M)) @ R^{-T}
auto QrBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("QrBackward::forward should not be called directly");
}

auto QrBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad_Q = grad_outputs[0];   // dL/dQ, (..., M, N)
    const auto& grad_R = grad_outputs[1];   // dL/dR, (..., N, N)

    const auto& Q = saved_tensors_[0];      // Q, (..., M, N)
    const auto& R = saved_tensors_[1];      // R, (..., N, N)

    auto ndim = R.ndim();
    auto Rt = transpose(R, ndim - 2, ndim - 1);
    auto Qt = transpose(Q, Q.ndim() - 2, Q.ndim() - 1);

    // M = R @ grad_R^T - grad_Q^T @ Q
    auto grad_Rt = transpose(grad_R, ndim - 2, ndim - 1);
    auto M = sub(matmul(R, grad_Rt), matmul(Qt, grad_Q));

    // copyltu(M) = tril(M) + tril(M, -1)^T
    auto M_tril = tril(M);
    auto M_strict_lower = tril(M, -1);
    auto M_strict_lower_t = transpose(M_strict_lower, ndim - 2, ndim - 1);
    auto copyltu_M = add(M_tril, M_strict_lower_t);

    // dL/dA = (grad_Q + Q @ copyltu_M) @ R^{-T}
    auto Q_copyltu = matmul(Q, copyltu_M);
    auto rhs = add(grad_Q, Q_copyltu);

    // Solve R^T @ X^T = rhs^T  =>  X = (R^{-T} @ rhs^T)^T...
    // Actually: rhs @ R^{-T} = solve(R, rhs^T)^T
    auto rhs_t = transpose(rhs, rhs.ndim() - 2, rhs.ndim() - 1);
    auto solve_result = tenzor::linalg::solve(R, rhs_t);
    auto grad_A = transpose(solve_result, solve_result.ndim() - 2, solve_result.ndim() - 1);

    return {grad_A};
}

// EighBackward implementation
// Forward: (W, V) = eigh(A) where A = V @ diag(W) @ V^T
// Backward:
//   F_{ij} = 1/(w_j - w_i) for i != j, 0 on diagonal
//   dL/dA = V @ (F * (V^T @ dL/dV) + diag(dL/dW)) @ V^T
auto EighBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("EighBackward::forward should not be called directly");
}

auto EighBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad_W = grad_outputs[0];   // dL/dW, (..., N)
    const auto& grad_V = grad_outputs[1];   // dL/dV, (..., N, N)

    const auto& W = saved_tensors_[0];      // eigenvalues, (..., N)
    const auto& V = saved_tensors_[1];      // eigenvectors, (..., N, N)

    auto N = W.shape()[W.ndim() - 1];

    // Construct F matrix: F_{ij} = 1/(w_j - w_i) for i != j, 0 on diagonal
    auto W_row = reshape(W, {1, N});
    auto W_col = reshape(W, {N, 1});
    auto diffs = sub(W_row, W_col);  // (N, N): diffs[i][j] = w_j - w_i

    // Replace zeros on diagonal with 1 to avoid division by zero
    auto mask = eye(N, std::nullopt, W.dtype(), W.device());
    auto safe_diffs = add(diffs, mask);

    auto F = reciprocal(safe_diffs);
    auto anti_mask = sub(ones({N, N}, W.dtype(), W.device()), mask);
    F = mul(F, anti_mask);  // Zero out diagonal

    auto Vt = transpose(V, V.ndim() - 2, V.ndim() - 1);

    // V^T @ dL/dV
    auto VtgV = matmul(Vt, grad_V);  // (N, N)

    // F * (V^T @ dL/dV) + diag(dL/dW)
    auto middle = add(mul(F, VtgV), diag(grad_W));  // (N, N)

    // dL/dA = V @ middle @ V^T
    auto grad_A = matmul(matmul(V, middle), Vt);

    // Symmetrize the result (since A is symmetric)
    auto grad_At = transpose(grad_A, grad_A.ndim() - 2, grad_A.ndim() - 1);
    grad_A = mul(add(grad_A, grad_At), 0.5);

    return {grad_A};
}

// EigvalshBackward implementation
// Forward: W = eigvalsh(A)
// Backward: dL/dA = V @ diag(dL/dW) @ V^T
//           where V was computed via eigh during forward
auto EigvalshBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("EigvalshBackward::forward should not be called directly");
}

auto EigvalshBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad_W = grad_outputs[0];   // dL/dW, (..., N)
    const auto& V = saved_tensors_[0];      // eigenvectors, (..., N, N)

    auto Vt = transpose(V, V.ndim() - 2, V.ndim() - 1);

    // dL/dA = V @ diag(dL/dW) @ V^T
    auto grad_diag = diag(grad_W);  // (N, N)
    auto grad_A = matmul(matmul(V, grad_diag), Vt);

    // Symmetrize (since A is symmetric)
    auto grad_At = transpose(grad_A, grad_A.ndim() - 2, grad_A.ndim() - 1);
    grad_A = mul(add(grad_A, grad_At), 0.5);

    return {grad_A};
}

// ============================================================================
// SpMMBackward
// ============================================================================

auto SpMMBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SpMMBackward::forward should not be called directly");
}

auto SpMMBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Y = S @ D  =>  grad_D = S^T @ grad_Y
    // sparse_transposed_ = S^T stored as SparseTensor (K, M)
    if (!sparse_transposed_.has_value()) {
        throw std::runtime_error("SpMMBackward: sparse_transposed_ not set");
    }
    const auto& grad_output = grad_outputs[0];  // shape (M, N)

    // grad_D = S^T @ grad_Y using sparse::spmm, shape (K, N)
    auto grad_dense = sparse::spmm(*sparse_transposed_, grad_output);

    return {grad_dense};
}

// ============================================================================
// SpMVBackward
// ============================================================================

auto SpMVBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SpMVBackward::forward should not be called directly");
}

auto SpMVBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // y = S @ v  =>  grad_v = S^T @ grad_y
    // sparse_transposed_ = S^T stored as SparseTensor (K, M)
    if (!sparse_transposed_.has_value()) {
        throw std::runtime_error("SpMVBackward: sparse_transposed_ not set");
    }
    const auto& grad_y = grad_outputs[0];  // shape (M,)

    // grad_v = S^T @ grad_y using sparse::spmv, shape (K,)
    auto grad_v = sparse::spmv(*sparse_transposed_, grad_y);

    return {grad_v};
}

// ============================================================================
// Linalg backward_with_variables implementations (higher-order gradients)
// ============================================================================

auto DetBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dA = dL/dy * det(A) * A^{-T}
    // Use saved Variables if available, otherwise wrap saved Tensors
    Variable det_val, inv_A;
    if (has_saved_variables()) {
        require_saved_variables(2);
        det_val = saved_variables_[0];
        inv_A = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        det_val = Variable(saved_tensors_[0], false);
        inv_A = Variable(saved_tensors_[1], false);
    }

    auto ndim = inv_A.tensor().ndim();
    auto inv_At = tenzor::transpose(inv_A, ndim - 2, ndim - 1);

    // grad * det(A) — expand for broadcasting with matrix shape
    auto grad_det = grad_outputs[0] * det_val;
    auto gd_shape = std::vector<int64_t>(grad_det.shape().begin(), grad_det.shape().end());
    gd_shape.push_back(1);
    gd_shape.push_back(1);
    auto grad_det_expanded = tenzor::reshape(grad_det, gd_shape);

    auto grad_A = grad_det_expanded * inv_At;
    return {grad_A};
}

auto InvBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dA = -Y^T @ dL/dY @ Y^T  where Y = A^{-1}
    Variable inv_A;
    if (has_saved_variables()) {
        require_saved_variables(1);
        inv_A = saved_variables_[0];
    } else {
        require_saved_tensors(1);
        inv_A = Variable(saved_tensors_[0], false);
    }

    auto ndim = inv_A.tensor().ndim();
    auto inv_At = tenzor::transpose(inv_A, ndim - 2, ndim - 1);

    auto temp = tenzor::matmul(inv_At, grad_outputs[0]);
    auto result = tenzor::matmul(temp, inv_At);
    auto grad_A = tenzor::neg(result);

    return {grad_A};
}

auto SolveBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dB = solve(A^T, dL/dX), dL/dA = -dL/dB @ X^T
    Variable A, X;
    if (has_saved_variables()) {
        require_saved_variables(2);
        A = saved_variables_[0];
        X = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        A = Variable(saved_tensors_[0], false);
        X = Variable(saved_tensors_[1], false);
    }

    auto ndim = A.tensor().ndim();
    auto At = tenzor::transpose(A, ndim - 2, ndim - 1);

    auto grad_B = tenzor::solve(At, grad_outputs[0]);

    auto x_ndim = X.tensor().ndim();
    auto Xt = tenzor::transpose(X, x_ndim - 2, x_ndim - 1);
    auto grad_A = tenzor::neg(tenzor::matmul(grad_B, Xt));

    return {grad_A, grad_B};
}

auto NormBackward_Linalg::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Frobenius: dL/dA = dL/dy * A / norm(A)
    Variable input, norm_val;
    if (has_saved_variables()) {
        require_saved_variables(2);
        input = saved_variables_[0];
        norm_val = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        input = Variable(saved_tensors_[0], false);
        norm_val = Variable(saved_tensors_[1], false);
    }

    if (ord_ == "fro") {
        auto scale = grad_outputs[0] * tenzor::reciprocal(norm_val);
        auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
        auto scale_shape = std::vector<int64_t>(scale.shape().begin(), scale.shape().end());
        while (scale_shape.size() < input_shape.size()) {
            scale_shape.push_back(1);
        }
        auto scale_expanded = tenzor::reshape(scale, scale_shape);
        return {scale_expanded * input};
    }

    // Unsupported norm order — return zeros (no gradient)
    return {Variable(zeros_like(input.tensor()), false)};
}

auto SlogdetBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dA = dL/d(logabsdet) * A^{-T} (sign gradient is zero)
    Variable inv_A;
    if (has_saved_variables()) {
        require_saved_variables(1);
        inv_A = saved_variables_[0];
    } else {
        require_saved_tensors(1);
        inv_A = Variable(saved_tensors_[0], false);
    }

    auto ndim = inv_A.tensor().ndim();
    auto inv_At = tenzor::transpose(inv_A, ndim - 2, ndim - 1);

    auto gd_shape = std::vector<int64_t>(grad_outputs[1].shape().begin(), grad_outputs[1].shape().end());
    while (gd_shape.size() < static_cast<size_t>(ndim)) {
        gd_shape.push_back(1);
    }
    auto grad_expanded = tenzor::reshape(grad_outputs[1], gd_shape);

    return {grad_expanded * inv_At};
}

auto EigvalshBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dA = V @ diag(dL/dW) @ V^T — delegates to backward() since diag() has no Variable-level op
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
}

// Complex linalg ops — delegate to backward() and wrap (diag/tril/eye have no Variable-level ops)

auto CholeskyBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
}

auto SvdBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    std::vector<Tensor> tensor_grads;
    tensor_grads.reserve(grad_outputs.size());
    for (auto& var : grad_outputs) {
        tensor_grads.push_back(var.tensor());
    }
    auto result_tensors = backward(tensor_grads);
    bool any_rg = false;
    for (auto& var : grad_outputs) { if (var.requires_grad()) { any_rg = true; break; } }
    return {Variable(result_tensors[0], any_rg)};
}

auto QrBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result_tensors = backward({grad_outputs[0].tensor(), grad_outputs[1].tensor()});
    bool any_rg = grad_outputs[0].requires_grad() || grad_outputs[1].requires_grad();
    return {Variable(result_tensors[0], any_rg)};
}

auto EighBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result_tensors = backward({grad_outputs[0].tensor(), grad_outputs[1].tensor()});
    bool any_rg = grad_outputs[0].requires_grad() || grad_outputs[1].requires_grad();
    return {Variable(result_tensors[0], any_rg)};
}

// ============================================================================
// CumSum backward
// ============================================================================

auto CumSumBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CumSumBackward::forward should not be called");
}

auto CumSumBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // dL/dx = flip(cumsum(flip(grad, dim), dim), dim)
    auto flipped = flip(grad, {dim_});
    auto cum = cumsum(flipped, dim_);
    return {flip(cum, {dim_})};
}

auto CumSumBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// CumProd backward
// ============================================================================

auto CumProdBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CumProdBackward::forward should not be called");
}

auto CumProdBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];

    // dL/dx = flip(cumsum(flip(output * grad, dim), dim), dim) / input
    // With zero-safe division
    auto prod_grad = mul(output, grad);
    auto flipped = flip(prod_grad, {dim_});
    auto cum = cumsum(flipped, dim_);
    auto rev_cum = flip(cum, {dim_});

    // Zero-safe: where input == 0, use 0 gradient
    auto eps = full(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                    1e-30, input.dtype(), input.device());
    auto safe_input = where(eq(input, zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                                            input.dtype(), input.device())),
                           eps, input);
    auto result = div(rev_cum, safe_input);

    // Zero out positions where input was zero
    auto zero_mask = eq(input, zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                                     input.dtype(), input.device()));
    auto zero_tensor = zeros(std::vector<int64_t>(result.shape().begin(), result.shape().end()),
                            result.dtype(), result.device());
    return {where(zero_mask, zero_tensor, result)};
}

auto CumProdBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// TopK backward
// ============================================================================

auto TopKBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TopKBackward::forward should not be called");
}

auto TopKBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // saved_tensors_[0] = original shape as 1D Int64 tensor
    // saved_tensors_[1] = indices from topk
    const auto& shape_tensor = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];

    auto shape_ptr = shape_tensor.data<int64_t>();
    auto orig_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    // Create zeros with original shape and scatter grad at index positions
    auto result = zeros(orig_shape, grad.dtype(), grad.device());
    return {scatter_add(result, dim_, indices, grad)};
}

auto TopKBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// Sort backward
// ============================================================================

auto SortBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SortBackward::forward should not be called");
}

auto SortBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // saved_tensors_[0] = original shape as 1D Int64 tensor
    // saved_tensors_[1] = sort indices
    const auto& shape_tensor = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];

    auto shape_ptr = shape_tensor.data<int64_t>();
    auto orig_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    // Scatter grad back using inverse permutation (same as scatter)
    auto result = zeros(orig_shape, grad.dtype(), grad.device());
    return {scatter(result, dim_, indices, grad)};
}

auto SortBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// Diag backward
// ============================================================================

auto DiagBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("DiagBackward::forward should not be called");
}

auto DiagBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // diag() is its own "transpose": applying diag to the grad reverses the operation
    return {diag(grad, diagonal_)};
}

auto DiagBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// Trace backward
// ============================================================================

auto TraceBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TraceBackward::forward should not be called");
}

auto TraceBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // saved_tensors_[0] holds dtype/device info from the original input
    const auto& input = saved_tensors_[0];
    // dL/dA = grad_scalar * eye(n)
    auto identity = eye(n_, std::nullopt, input.dtype(), input.device());
    // grad is scalar — expand it
    return {mul(identity, grad)};
}

auto TraceBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// Triu backward
// ============================================================================

auto TriuBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TriuBackward::forward should not be called");
}

auto TriuBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {triu(grad_outputs[0], diagonal_)};
}

auto TriuBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// Tril backward
// ============================================================================

auto TrilBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TrilBackward::forward should not be called");
}

auto TrilBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {tril(grad_outputs[0], diagonal_)};
}

auto TrilBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// FFT backward
// ============================================================================

auto FFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("FFTBackward::forward should not be called");
}

auto FFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {fft::ifft(grad_outputs[0], n_, dim_, norm_)};
}

auto FFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// IFFT backward
// ============================================================================

auto IFFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IFFTBackward::forward should not be called");
}

auto IFFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {fft::fft(grad_outputs[0], n_, dim_, norm_)};
}

auto IFFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// RFFT backward
// ============================================================================

auto RFFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("RFFTBackward::forward should not be called");
}

auto RFFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // irfft needs the original signal length to reconstruct
    return {fft::irfft(grad_outputs[0], signal_length_, dim_, norm_)};
}

auto RFFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// IRFFT backward
// ============================================================================

auto IRFFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IRFFTBackward::forward should not be called");
}

auto IRFFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {fft::rfft(grad_outputs[0], std::nullopt, dim_, norm_)};
}

auto IRFFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ============================================================================
// Sparse backward_with_variables implementations
// ============================================================================

auto SpMMBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Sparse ops have no Variable-level equivalents — delegate to backward()
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
}

auto SpMVBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
}

} // namespace tenzor
