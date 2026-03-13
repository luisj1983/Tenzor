#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/anomaly_mode.hpp"
#include "tenzor/autograd/engine.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include <iostream>
#include <vector>

namespace tenzor {

// Thread-local gradient state — each thread has independent grad tracking
static thread_local bool grad_enabled{true};

// Thread-local create_graph state for higher-order gradients
static thread_local bool creating_graph{false};

// Thread-local inference mode state
static thread_local bool inference_mode_enabled{false};

// Thread-local anomaly detection state
static thread_local bool anomaly_detection_enabled{false};

Variable::Variable(Tensor data, bool requires_grad)
    : impl_(std::make_shared<VariableImpl>(std::move(data), requires_grad)) {
}

auto Variable::is_initialized() const -> bool {
    return impl_ != nullptr;
}

Variable::operator bool() const {
    return is_initialized();
}

auto Variable::tensor() const -> const Tensor& {
    if (!impl_) {
        throw std::runtime_error("Cannot access tensor of uninitialized Variable");
    }
    return impl_->data_;
}

auto Variable::tensor() -> Tensor& {
    if (!impl_) {
        throw std::runtime_error("Cannot access tensor of uninitialized Variable");
    }
    return impl_->data_;
}

auto Variable::grad() const -> const std::optional<Tensor>& {
    if (!impl_) {
        throw std::runtime_error("Cannot access grad of uninitialized Variable");
    }
    return impl_->grad_;
}

auto Variable::mutable_grad() -> std::optional<Tensor>& {
    if (!impl_) {
        throw std::runtime_error("Cannot access grad of uninitialized Variable");
    }
    return impl_->grad_;
}

auto Variable::has_grad() const -> bool {
    if (!impl_) return false;
    if (impl_->thread_safe_.load(std::memory_order_acquire)) {
        std::lock_guard lock(*impl_->grad_mutex_);
        return impl_->grad_.has_value();
    }
    return impl_->grad_.has_value();
}

auto Variable::set_grad(Tensor gradient) -> void {
    if (!impl_) {
        throw std::runtime_error("Cannot set grad of uninitialized Variable");
    }
    if (impl_->thread_safe_.load(std::memory_order_acquire)) {
        std::lock_guard lock(*impl_->grad_mutex_);
        impl_->grad_ = std::move(gradient);
    } else {
        impl_->grad_ = std::move(gradient);
    }
}

auto Variable::backward(std::optional<Tensor> gradient, bool retain_graph, bool create_graph) -> void {
    if (!impl_) {
        throw std::runtime_error("Cannot call backward on uninitialized Variable");
    }
    // create_graph implies retain_graph (we need the graph to differentiate through it)
    backward_engine().execute(*this, gradient, retain_graph || create_graph, create_graph);
}

auto Variable::register_hook(std::function<Tensor(const Tensor&)> hook) -> size_t {
    if (!impl_) {
        throw std::runtime_error("Cannot register hook on uninitialized Variable");
    }
    size_t id = impl_->next_hook_id_.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock lock(impl_->hooks_mutex_);
    impl_->hooks_[id] = std::move(hook);
    return id;
}

auto Variable::unregister_hook(size_t hook_id) -> bool {
    if (!impl_) {
        throw std::runtime_error("Cannot unregister hook on uninitialized Variable");
    }
    std::unique_lock lock(impl_->hooks_mutex_);
    return impl_->hooks_.erase(hook_id) > 0;
}

auto Variable::clear_hooks() -> void {
    if (!impl_) return;
    std::unique_lock lock(impl_->hooks_mutex_);
    impl_->hooks_.clear();
}

auto Variable::make_thread_safe() -> void {
    if (!impl_) {
        throw std::runtime_error("Cannot make uninitialized Variable thread-safe");
    }
    impl_->thread_safe_.store(true, std::memory_order_release);
}

auto Variable::is_thread_safe() const -> bool {
    return impl_ && impl_->thread_safe_.load(std::memory_order_acquire);
}

auto Variable::retain_grad() -> void {
    if (!impl_) {
        throw std::runtime_error("Cannot retain grad of uninitialized Variable");
    }
    impl_->retain_grad_ = true;
}

auto Variable::retains_grad() const -> bool {
    return impl_ && impl_->retain_grad_;
}

auto Variable::zero_grad() -> void {
    if (impl_) {
        if (impl_->thread_safe_.load(std::memory_order_acquire)) {
            std::lock_guard lock(*impl_->grad_mutex_);
            impl_->grad_.reset();
        } else {
            impl_->grad_.reset();
        }
    }
}

auto Variable::detach() -> Variable {
    if (!impl_) {
        throw std::runtime_error("Cannot detach uninitialized Variable");
    }
    return Variable(impl_->data_.detach(), false);
}

auto Variable::requires_grad() const -> bool {
    return impl_ && impl_->requires_grad_;
}

auto Variable::set_requires_grad(bool requires_grad) -> void {
    if (!impl_) {
        throw std::runtime_error("Cannot set requires_grad on uninitialized Variable");
    }
    impl_->requires_grad_ = requires_grad;
}

auto Variable::is_leaf() const -> bool {
    return !impl_ || !impl_->grad_fn_;
}

auto Variable::set_grad_fn(std::shared_ptr<Function> fn) -> void {
    if (!impl_) {
        throw std::runtime_error("Cannot set grad_fn on uninitialized Variable");
    }
    impl_->grad_fn_ = std::move(fn);
}

auto Variable::grad_fn() const -> std::shared_ptr<Function> {
    return impl_ ? impl_->grad_fn_ : nullptr;
}

auto Variable::shape() const -> std::span<const int64_t> {
    if (!impl_) {
        throw std::runtime_error("Cannot access shape of uninitialized Variable");
    }
    return impl_->data_.shape();
}

auto Variable::dtype() const -> DType {
    if (!impl_) {
        throw std::runtime_error("Cannot access dtype of uninitialized Variable");
    }
    return impl_->data_.dtype();
}

auto Variable::device() const -> const Device& {
    if (!impl_) {
        throw std::runtime_error("Cannot access device of uninitialized Variable");
    }
    return impl_->data_.device();
}

// NoGradGuard implementation
NoGradGuard::NoGradGuard() : prev_state_(is_grad_enabled()) {
    set_grad_enabled(false);
}

NoGradGuard::~NoGradGuard() {
    set_grad_enabled(prev_state_);
}

InferenceModeGuard::InferenceModeGuard()
    : prev_grad_state_(is_grad_enabled()),
      prev_inference_state_(inference_mode_enabled) {
    set_grad_enabled(false);
    inference_mode_enabled = true;
}

InferenceModeGuard::~InferenceModeGuard() {
    set_grad_enabled(prev_grad_state_);
    inference_mode_enabled = prev_inference_state_;
}

auto is_inference_mode_enabled() -> bool {
    return inference_mode_enabled;
}

// Global functions
auto is_grad_enabled() -> bool {
    return grad_enabled;
}

auto set_grad_enabled(bool enabled) -> void {
    grad_enabled = enabled;
}

// Higher-order gradient graph creation state
auto is_creating_graph() -> bool {
    return creating_graph;
}

auto set_creating_graph(bool creating) -> void {
    creating_graph = creating;
}

// CreateGraphGuard implementation
CreateGraphGuard::CreateGraphGuard() : prev_state_(is_creating_graph()) {
    set_creating_graph(true);
}

CreateGraphGuard::~CreateGraphGuard() {
    set_creating_graph(prev_state_);
}

// Anomaly detection state
auto is_anomaly_detection_enabled() -> bool {
    return anomaly_detection_enabled;
}

auto set_anomaly_detection(bool enabled) -> void {
    anomaly_detection_enabled = enabled;
}

AnomalyMode::AnomalyMode(bool enabled) : prev_state_(is_anomaly_detection_enabled()) {
    set_anomaly_detection(enabled);
}

AnomalyMode::~AnomalyMode() {
    set_anomaly_detection(prev_state_);
}

// Arithmetic operators
auto Variable::operator+(const Variable& other) const -> Variable {
    if (!impl_) {
        throw std::runtime_error("Cannot add with uninitialized Variable (lhs)");
    }
    if (!other.impl_) {
        throw std::runtime_error("Cannot add with uninitialized Variable (rhs)");
    }
    // Compute result
    auto result = impl_->data_ + other.impl_->data_;
    bool needs_grad = impl_->requires_grad_ || other.impl_->requires_grad_;
    Variable output(result, needs_grad);

    if (is_grad_enabled() && needs_grad) {
        auto grad_fn = std::make_shared<AddBackward>();

        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.push_back(impl_->grad_fn_);
        next_funcs.push_back(other.impl_->grad_fn_);
        grad_fn->set_next_functions(next_funcs);
        grad_fn->set_input_variables({*this, other});

        grad_fn->input_shape_a_ = std::vector<int64_t>(impl_->data_.shape().begin(), impl_->data_.shape().end());
        grad_fn->input_shape_b_ = std::vector<int64_t>(other.impl_->data_.shape().begin(), other.impl_->data_.shape().end());

        output.set_grad_fn(grad_fn);
    }

    return output;
}

auto Variable::operator-(const Variable& other) const -> Variable {
    if (!impl_) {
        throw std::runtime_error("Cannot subtract with uninitialized Variable (lhs)");
    }
    if (!other.impl_) {
        throw std::runtime_error("Cannot subtract with uninitialized Variable (rhs)");
    }
    // Compute result
    auto result = impl_->data_ - other.impl_->data_;
    bool needs_grad = impl_->requires_grad_ || other.impl_->requires_grad_;
    Variable output(result, needs_grad);

    if (is_grad_enabled() && needs_grad) {
        auto grad_fn = std::make_shared<SubBackward>();

        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.push_back(impl_->grad_fn_);
        next_funcs.push_back(other.impl_->grad_fn_);
        grad_fn->set_next_functions(next_funcs);
        grad_fn->set_input_variables({*this, other});

        grad_fn->input_shape_a_ = std::vector<int64_t>(impl_->data_.shape().begin(), impl_->data_.shape().end());
        grad_fn->input_shape_b_ = std::vector<int64_t>(other.impl_->data_.shape().begin(), other.impl_->data_.shape().end());

        output.set_grad_fn(grad_fn);
    }

    return output;
}

auto Variable::operator*(const Variable& other) const -> Variable {
    if (!impl_) {
        throw std::runtime_error("Cannot multiply with uninitialized Variable (lhs)");
    }
    if (!other.impl_) {
        throw std::runtime_error("Cannot multiply with uninitialized Variable (rhs)");
    }
    // Compute result
    auto result = impl_->data_ * other.impl_->data_;
    bool needs_grad = impl_->requires_grad_ || other.impl_->requires_grad_;
    Variable output(result, needs_grad);

    if (is_grad_enabled() && needs_grad) {
        auto grad_fn = std::make_shared<MulBackward>();

        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.push_back(impl_->grad_fn_);
        next_funcs.push_back(other.impl_->grad_fn_);
        grad_fn->set_next_functions(next_funcs);
        grad_fn->set_input_variables({*this, other});

        grad_fn->save_for_backward({impl_->data_, other.impl_->data_});
        grad_fn->input_shape_a_ = std::vector<int64_t>(impl_->data_.shape().begin(), impl_->data_.shape().end());
        grad_fn->input_shape_b_ = std::vector<int64_t>(other.impl_->data_.shape().begin(), other.impl_->data_.shape().end());

        if (is_creating_graph()) {
            grad_fn->save_variables_for_backward({*this, other});
        }

        output.set_grad_fn(grad_fn);
    }

    return output;
}

auto Variable::operator/(const Variable& other) const -> Variable {
    if (!impl_) {
        throw std::runtime_error("Cannot divide with uninitialized Variable (lhs)");
    }
    if (!other.impl_) {
        throw std::runtime_error("Cannot divide with uninitialized Variable (rhs)");
    }
    // Compute result
    auto result = impl_->data_ / other.impl_->data_;
    bool needs_grad = impl_->requires_grad_ || other.impl_->requires_grad_;
    Variable output(result, needs_grad);

    if (is_grad_enabled() && needs_grad) {
        auto grad_fn = std::make_shared<DivBackward>();

        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.push_back(impl_->grad_fn_);
        next_funcs.push_back(other.impl_->grad_fn_);
        grad_fn->set_next_functions(next_funcs);
        grad_fn->set_input_variables({*this, other});

        grad_fn->save_for_backward({impl_->data_, other.impl_->data_});
        grad_fn->input_shape_a_ = std::vector<int64_t>(impl_->data_.shape().begin(), impl_->data_.shape().end());
        grad_fn->input_shape_b_ = std::vector<int64_t>(other.impl_->data_.shape().begin(), other.impl_->data_.shape().end());

        if (is_creating_graph()) {
            grad_fn->save_variables_for_backward({*this, other});
        }

        output.set_grad_fn(grad_fn);
    }

    return output;
}

// Helper: create scalar tensor directly on target device via full() dispatch.
// Avoids the old 2-step CPU alloc + .to(device) pattern for GPU tensors.
static auto make_scalar_var(double value, DType dtype, Device device) -> Variable {
    Tensor t = full({}, value, dtype, device);
    return Variable(t, false);
}

// Scalar operations
auto Variable::operator+(float scalar) const -> Variable {
    if (!impl_) throw std::runtime_error("Cannot add scalar to uninitialized Variable");
    return *this + make_scalar_var(scalar, impl_->data_.dtype(), impl_->data_.device());
}

auto Variable::operator+(double scalar) const -> Variable {
    if (!impl_) throw std::runtime_error("Cannot add scalar to uninitialized Variable");
    return *this + make_scalar_var(scalar, impl_->data_.dtype(), impl_->data_.device());
}

auto Variable::operator*(float scalar) const -> Variable {
    if (!impl_) throw std::runtime_error("Cannot multiply scalar with uninitialized Variable");
    return *this * make_scalar_var(scalar, impl_->data_.dtype(), impl_->data_.device());
}

auto Variable::operator*(double scalar) const -> Variable {
    if (!impl_) throw std::runtime_error("Cannot multiply scalar with uninitialized Variable");
    return *this * make_scalar_var(scalar, impl_->data_.dtype(), impl_->data_.device());
}

// Scalar subtraction operators
auto Variable::operator-(float scalar) const -> Variable {
    return *this - static_cast<double>(scalar);
}

auto Variable::operator-(double scalar) const -> Variable {
    return *this - make_scalar_var(scalar, dtype(), device());
}

// Scalar division operators
auto Variable::operator/(float scalar) const -> Variable {
    return *this / static_cast<double>(scalar);
}

auto Variable::operator/(double scalar) const -> Variable {
    return *this / make_scalar_var(scalar, dtype(), device());
}

// Shape transformation methods
auto Variable::reshape(std::vector<int64_t> shape) const -> Variable {
    return tenzor::reshape(*this, std::move(shape));
}

auto Variable::transpose(int64_t dim0, int64_t dim1) const -> Variable {
    return tenzor::transpose(*this, dim0, dim1);
}

auto Variable::permute(std::vector<int64_t> dims) const -> Variable {
    return tenzor::permute(*this, std::move(dims));
}

auto Variable::matmul(const Variable& other) const -> Variable {
    return tenzor::matmul(*this, other);
}

auto Variable::squeeze(int64_t dim) const -> Variable {
    return tenzor::squeeze(*this, dim);
}

} // namespace tenzor

