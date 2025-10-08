#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/ops/math.hpp"
#include <numeric>
#include <algorithm>

namespace tenzor {

// TensorImpl implementation
TensorImpl::TensorImpl(std::vector<int64_t> shape_, DType dtype_, Device device_)
    : shape(std::move(shape_)), dtype(dtype_), device(device_) {

    // Compute strides
    strides = compute_strides(this->shape);

    // Allocate storage
    size_t size_bytes = numel() * dtype_size(dtype);

    if (device.type == Device::Type::CPU) {
        storage = std::make_shared<CPUStorage>(size_bytes);
    } else {
        // Device storage will be allocated by backend
        // For now, just create empty storage
        storage = std::make_shared<CPUStorage>(size_bytes);
    }
}

auto TensorImpl::numel() const -> int64_t {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                          std::multiplies<int64_t>{});
}

auto TensorImpl::is_contiguous() const -> bool {
    auto expected_strides = compute_strides(shape);
    return strides == expected_strides;
}

// Tensor implementation
Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device device)
    : impl_(std::make_shared<TensorImpl>(std::move(shape), dtype, device)) {}

auto Tensor::shape() const noexcept -> std::span<const int64_t> {
    if (!impl_) return {};
    return impl_->shape;
}

auto Tensor::strides() const noexcept -> std::span<const int64_t> {
    if (!impl_) return {};
    return impl_->strides;
}

auto Tensor::ndim() const noexcept -> int64_t {
    if (!impl_) return 0;
    return static_cast<int64_t>(impl_->shape.size());
}

auto Tensor::numel() const noexcept -> int64_t {
    if (!impl_) return 0;
    return impl_->numel();
}

auto Tensor::dtype() const noexcept -> DType {
    if (!impl_) return DType::Float32;
    return impl_->dtype;
}

auto Tensor::device() const noexcept -> const Device& {
    static const Device default_device = Device::cpu();
    if (!impl_) return default_device;
    return impl_->device;
}

auto Tensor::requires_grad() const noexcept -> bool {
    if (!impl_) return false;
    return impl_->requires_grad;
}

auto Tensor::is_contiguous() const noexcept -> bool {
    if (!impl_) return true;
    return impl_->is_contiguous();
}

// Template instantiations for common types
template<> auto Tensor::data<float>() -> float* {
    return static_cast<float*>(impl_->storage->data());
}

template<> auto Tensor::data<float>() const -> const float* {
    return static_cast<const float*>(impl_->storage->data());
}

template<> auto Tensor::data<double>() -> double* {
    return static_cast<double*>(impl_->storage->data());
}

template<> auto Tensor::data<double>() const -> const double* {
    return static_cast<const double*>(impl_->storage->data());
}

template<> auto Tensor::data<int32_t>() -> int32_t* {
    return static_cast<int32_t*>(impl_->storage->data());
}

template<> auto Tensor::data<int32_t>() const -> const int32_t* {
    return static_cast<const int32_t*>(impl_->storage->data());
}

// Stub implementations for operations
auto Tensor::to(Device device) const -> Tensor {
    // TODO: Implement device transfer
    return *this;
}

auto Tensor::to(DType dtype) const -> Tensor {
    // TODO: Implement dtype conversion
    return *this;
}

auto Tensor::cuda(int32_t device_id) const -> Tensor {
    return to(Device::cuda(device_id));
}

auto Tensor::cpu() const -> Tensor {
    return to(Device::cpu());
}

auto Tensor::clone() const -> Tensor {
    // TODO: Implement deep copy
    return *this;
}

auto Tensor::detach() const -> Tensor {
    auto result = clone();
    result.impl_->requires_grad = false;
    return result;
}

auto Tensor::contiguous() const -> Tensor {
    if (is_contiguous()) {
        return *this;
    }
    // TODO: Implement contiguous copy
    return *this;
}

// Arithmetic operators
auto Tensor::operator+(const Tensor& other) const -> Tensor {
    return tenzor::add(*this, other);
}

auto Tensor::operator-(const Tensor& other) const -> Tensor {
    return tenzor::sub(*this, other);
}

auto Tensor::operator*(const Tensor& other) const -> Tensor {
    return tenzor::mul(*this, other);
}

auto Tensor::operator/(const Tensor& other) const -> Tensor {
    return tenzor::div(*this, other);
}

// Scalar operations
auto Tensor::operator+(float scalar) const -> Tensor {
    // TODO: Implement scalar addition
    return *this;
}

auto Tensor::operator-(float scalar) const -> Tensor {
    // TODO: Implement scalar subtraction
    return *this;
}

auto Tensor::operator*(float scalar) const -> Tensor {
    // TODO: Implement scalar multiplication
    return *this;
}

auto Tensor::operator/(float scalar) const -> Tensor {
    // TODO: Implement scalar division
    return *this;
}

// In-place operations
auto Tensor::operator+=(const Tensor& other) -> Tensor& {
    *this = *this + other;
    return *this;
}

auto Tensor::operator-=(const Tensor& other) -> Tensor& {
    *this = *this - other;
    return *this;
}

auto Tensor::operator*=(const Tensor& other) -> Tensor& {
    *this = *this * other;
    return *this;
}

auto Tensor::operator/=(const Tensor& other) -> Tensor& {
    *this = *this / other;
    return *this;
}

auto Tensor::fill_(float value) -> Tensor& {
    // TODO: Implement fill
    return *this;
}

auto Tensor::zero_() -> Tensor& {
    return fill_(0.0f);
}

// Shape operations
auto Tensor::reshape(std::vector<int64_t> new_shape) const -> Tensor {
    // TODO: Implement reshape
    return *this;
}

auto Tensor::view(std::vector<int64_t> new_shape) const -> Tensor {
    // TODO: Implement view (zero-copy)
    return *this;
}

auto Tensor::transpose(int64_t dim0, int64_t dim1) const -> Tensor {
    // TODO: Implement transpose
    return *this;
}

auto Tensor::permute(std::vector<int64_t> dims) const -> Tensor {
    // TODO: Implement permute
    return *this;
}

auto Tensor::squeeze(std::optional<int64_t> dim) const -> Tensor {
    // TODO: Implement squeeze
    return *this;
}

auto Tensor::unsqueeze(int64_t dim) const -> Tensor {
    // TODO: Implement unsqueeze
    return *this;
}

auto Tensor::flatten(int64_t start_dim, int64_t end_dim) const -> Tensor {
    // TODO: Implement flatten
    return *this;
}

// Indexing
auto Tensor::operator[](int64_t idx) const -> Tensor {
    // TODO: Implement indexing
    return *this;
}

auto Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const -> Tensor {
    // TODO: Implement slice
    return *this;
}

// Comparison
auto Tensor::operator==(const Tensor& other) const -> Tensor {
    // TODO: Implement element-wise comparison
    return *this;
}

auto Tensor::operator!=(const Tensor& other) const -> Tensor {
    // TODO: Implement element-wise comparison
    return *this;
}

auto Tensor::operator<(const Tensor& other) const -> Tensor {
    // TODO: Implement element-wise comparison
    return *this;
}

auto Tensor::operator>(const Tensor& other) const -> Tensor {
    // TODO: Implement element-wise comparison
    return *this;
}

} // namespace tenzor
