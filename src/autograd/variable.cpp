#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/anomaly_mode.hpp"
#include "tenzor/autograd/engine.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/utils/logging.hpp"
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

auto Variable::set_data_view(Tensor data) -> void {
    if (!impl_) {
        throw std::runtime_error("Cannot set data view on uninitialized Variable");
    }
    // Replace only the data_ field. grad_fn_, grad_, hooks_, requires_grad_,
    // retain_grad_, was_non_leaf_, and creation_metadata_ are all left
    // untouched so any autograd graph that already references this Variable
    // (and any code that holds a stable shared_ptr<Variable> to this slot)
    // keeps working. This is the storage swap requested by parametrize's
    // pre-forward hook to redirect a parameter at the chain's output tensor
    // without overwriting the parameter buffer in place.
    impl_->data_ = std::move(data);
}

auto Variable::grad() const -> const std::optional<Tensor>& {
    if (!impl_) {
        throw std::runtime_error("Cannot access grad of uninitialized Variable");
    }
    // Match PyTorch: warn at most once when .grad is accessed on a
    // non-leaf Variable that has no stored gradient and was not marked
    // with retain_grad(). Use was_non_leaf_ rather than the live
    // grad_fn_ pointer so the check still fires after the backward
    // engine has cleared grad_fn_. The empty-grad check lets internal
    // consumers (engine seeding root, checkpointing, functional autograd
    // helpers) read the grad field without spurious warnings — those
    // paths always populate grad_ before reading it.
    const bool is_non_leaf = impl_->was_non_leaf_.load(std::memory_order_acquire);
    const bool wants_grad = impl_->requires_grad_;
    const bool retained = impl_->retain_grad_.load(std::memory_order_acquire);
    const bool grad_empty = !impl_->grad_.has_value();
    if (is_non_leaf && wants_grad && !retained && grad_empty) {
        TENZOR_WARN_ONCE(
            "The .grad attribute of a Variable that is not a leaf Variable "
            "is being accessed. Its .grad attribute won't be populated "
            "during autograd.backward(). If you indeed want the .grad field "
            "to be populated for a non-leaf Variable, use .retain_grad() on "
            "the non-leaf Variable. If you access the non-leaf Variable by "
            "mistake, make sure you access the leaf Variable instead.");
    }
    return impl_->grad_;
}

auto Variable::mutable_grad() -> std::optional<Tensor>& {
    if (!impl_) {
        throw std::runtime_error("Cannot access grad of uninitialized Variable");
    }
    return impl_->grad_;
}

auto Variable::grad_variable() const -> const std::optional<Variable>& {
    if (!impl_) {
        throw std::runtime_error("Cannot access grad_variable of uninitialized Variable");
    }
    using OptVar = std::optional<Variable>;
    if (!impl_->grad_with_graph_cache_storage_) {
        impl_->grad_with_graph_cache_storage_.reset(
            reinterpret_cast<char*>(new OptVar{}));
    }
    auto* cache = reinterpret_cast<OptVar*>(
        impl_->grad_with_graph_cache_storage_.get());
    if (impl_->grad_with_graph_impl_) {
        if (!cache->has_value() ||
            (*cache)->impl_ != impl_->grad_with_graph_impl_) {
            Variable v;
            v.impl_ = impl_->grad_with_graph_impl_;
            *cache = std::move(v);
        }
    } else {
        cache->reset();
    }
    return *cache;
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

    // When anomaly detection is enabled, record creation metadata for tracebacks
    if (is_anomaly_detection_enabled() && fn) {
        auto meta = std::make_shared<AnomalyMetadata>();
        meta->function_name = fn->name();

        // Record input shapes from the function's input variables
        for (const auto& input_var : fn->input_variables()) {
            auto s = input_var.shape();
            meta->input_shapes.emplace_back(s.begin(), s.end());
        }

        // Record output shape (this variable)
        auto out_shape = impl_->data_.shape();
        meta->output_shapes.emplace_back(out_shape.begin(), out_shape.end());

        // Link parent metadata from first input for chained tracebacks
        const auto& inputs = fn->input_variables();
        if (!inputs.empty()) {
            meta->parent = inputs[0].creation_metadata();
        }

        impl_->creation_metadata_ = std::move(meta);
    }

    // Latch "was created as non-leaf" before moving fn into the impl.
    // The engine later clears grad_fn_ during backward cleanup, but this
    // flag stays set so post-backward diagnostics (e.g. the retain_grad
    // warning) can still tell whether this Variable originated from an
    // operation. Only transitions false → true; a later set_grad_fn(nullptr)
    // does not unlatch it.
    if (fn) {
        impl_->was_non_leaf_.store(true, std::memory_order_release);
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

auto Variable::accumulate_sparse_grad(SparseTensor sg) -> void {
    if (!impl_) return;
    auto do_accumulate = [&] {
        if (impl_->sparse_grad_.has_value()) {
            impl_->sparse_grad_ =
                sparse::add(impl_->sparse_grad_.value(), sg).coalesce();
        } else {
            impl_->sparse_grad_ = std::move(sg);
        }
    };
    if (impl_->thread_safe_.load(std::memory_order_acquire)) {
        std::lock_guard lock(*impl_->grad_mutex_);
        do_accumulate();
    } else {
        do_accumulate();
    }
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

auto Variable::creation_metadata() const -> const std::shared_ptr<AnomalyMetadata>& {
    static const std::shared_ptr<AnomalyMetadata> empty;
    if (!impl_) return empty;
    return impl_->creation_metadata_;
}

auto Variable::set_creation_metadata(std::shared_ptr<AnomalyMetadata> meta) -> void {
    if (impl_) {
        impl_->creation_metadata_ = std::move(meta);
    }
}

} // namespace tenzor

