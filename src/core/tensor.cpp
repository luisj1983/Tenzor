#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/backend/dispatch.hpp"
#include <numeric>
#include <algorithm>
#include <cstring>
#include <iostream>

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
        // Allocate device storage using backend
        auto* backend = backend_registry().get_backend(device.type);
        if (!backend) {
            throw std::runtime_error("Backend not available for device: " + device.to_string());
        }

        // Allocate device memory
        void* device_ptr = backend->allocate(size_bytes, device.index);
        if (!device_ptr && size_bytes > 0) {
            // Only throw error if allocation failed for non-empty tensor
            throw std::runtime_error("Failed to allocate device memory");
        }

        // Create device storage (device_ptr can be nullptr for empty tensors)
        storage = std::make_shared<DeviceStorage>(device_ptr, size_bytes, device, backend);
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

template<> auto Tensor::data<int64_t>() -> int64_t* {
    return static_cast<int64_t*>(impl_->storage->data());
}

template<> auto Tensor::data<int64_t>() const -> const int64_t* {
    return static_cast<const int64_t*>(impl_->storage->data());
}

template<> auto Tensor::data<uint8_t>() -> uint8_t* {
    return static_cast<uint8_t*>(impl_->storage->data());
}

template<> auto Tensor::data<uint8_t>() const -> const uint8_t* {
    return static_cast<const uint8_t*>(impl_->storage->data());
}

template<> auto Tensor::data<bool>() -> bool* {
    return static_cast<bool*>(impl_->storage->data());
}

template<> auto Tensor::data<bool>() const -> const bool* {
    return static_cast<const bool*>(impl_->storage->data());
}

// Stub implementations for operations
auto Tensor::to(Device device) const -> Tensor {
    if (!impl_) {
        return *this;
    }

    // If already on the target device and contiguous, just return
    if (impl_->device == device && is_contiguous()) {
        return *this;
    }

    // For same-device non-contiguous GPU tensors, we need special handling
    // Fall through to the code below that handles non-contiguous GPU tensors

    // Special handling for non-contiguous GPU tensors:
    // Transfer to CPU first (with stride handling), then transfer to target device
    if (!is_contiguous() && impl_->device.type != Device::Type::CPU) {
        // Step 1: Create CPU tensor
        Tensor cpu_result(impl_->shape, impl_->dtype, Device::cpu());
        cpu_result.impl_->requires_grad = impl_->requires_grad;

        // Step 2: Copy from GPU to CPU with stride handling
        // We need to copy element-by-element due to non-contiguous layout
        const size_t size_bytes = numel() * dtype_size(impl_->dtype);
        std::vector<uint8_t> temp_buffer(size_bytes);

        // Get backend
        auto* src_backend = backend_registry().get_backend(impl_->device.type);
        if (!src_backend) {
            throw std::runtime_error("Backend not available for source device");
        }

        // Copy entire GPU buffer to temp (includes padding/non-contiguous data)
        const size_t total_bytes = impl_->storage->size_bytes();
        std::vector<uint8_t> gpu_buffer(total_bytes);
        src_backend->copy(gpu_buffer.data(), impl_->storage->data(),
                         total_bytes, CopyKind::DeviceToHost);

        // Now rearrange into contiguous layout on CPU
        const int64_t ndims = ndim();
        const size_t element_size = dtype_size(impl_->dtype);
        std::vector<int64_t> indices(ndims, 0);
        int64_t dst_offset = 0;

        for (int64_t i = 0; i < numel(); ++i) {
            int64_t src_offset = impl_->offset;
            for (int64_t dim = 0; dim < ndims; ++dim) {
                src_offset += indices[dim] * impl_->strides[dim];
            }

            std::memcpy(temp_buffer.data() + dst_offset * element_size,
                       gpu_buffer.data() + src_offset * element_size,
                       element_size);

            ++dst_offset;

            for (int64_t dim = ndims - 1; dim >= 0; --dim) {
                if (++indices[dim] < impl_->shape[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }

        // Copy contiguous data to CPU tensor
        std::memcpy(cpu_result.impl_->storage->data(), temp_buffer.data(), size_bytes);

        // Step 3: If target is also CPU, we're done
        if (device.type == Device::Type::CPU) {
            return cpu_result;
        }

        // Step 4: Transfer contiguous CPU tensor to target device
        Tensor result(cpu_result.impl_->shape, cpu_result.impl_->dtype, device);
        result.impl_->requires_grad = cpu_result.impl_->requires_grad;

        auto* dst_backend = backend_registry().get_backend(device.type);
        if (!dst_backend) {
            throw std::runtime_error("Backend not available for target device");
        }

        dst_backend->copy(result.impl_->storage->data(),
                         cpu_result.impl_->storage->data(),
                         size_bytes,
                         CopyKind::HostToDevice);

        return result;
    }

    // Normal path: tensor is contiguous or on CPU
    auto cont = is_contiguous() ? *this : contiguous();

    // Create new tensor on target device
    Tensor result(cont.impl_->shape, cont.impl_->dtype, device);
    result.impl_->requires_grad = cont.impl_->requires_grad;

    // Copy data using backend
    const size_t size_bytes = cont.numel() * dtype_size(cont.dtype());

    // Determine copy kind
    CopyKind copy_kind;
    if (cont.impl_->device.type == Device::Type::CPU && device.type == Device::Type::CPU) {
        copy_kind = CopyKind::HostToHost;
    } else if (cont.impl_->device.type == Device::Type::CPU && device.type != Device::Type::CPU) {
        copy_kind = CopyKind::HostToDevice;
    } else if (cont.impl_->device.type != Device::Type::CPU && device.type == Device::Type::CPU) {
        copy_kind = CopyKind::DeviceToHost;
    } else {
        copy_kind = CopyKind::DeviceToDevice;
    }

    // Get backend for the copy operation
    auto* backend = (copy_kind == CopyKind::HostToDevice || copy_kind == CopyKind::DeviceToDevice)
                    ? backend_registry().get_backend(device.type)
                    : backend_registry().get_backend(cont.impl_->device.type);

    if (!backend) {
        throw std::runtime_error("Backend not available for device transfer");
    }

    // Perform the copy
    backend->copy(result.impl_->storage->data(),
                  cont.impl_->storage->data(),
                  size_bytes,
                  copy_kind);

    return result;
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
    if (!impl_) {
        return *this;
    }

    // Make tensor contiguous if needed
    Tensor cont;
    if (is_contiguous()) {
        cont = *this;
    } else if (impl_->device.type == Device::Type::CPU) {
        cont = contiguous();
    } else {
        // For non-contiguous GPU tensors, use .to() which handles stride conversion
        cont = to(impl_->device);
    }

    // Create new tensor with same shape, dtype, device
    Tensor result(cont.impl_->shape, cont.impl_->dtype, cont.impl_->device);
    result.impl_->requires_grad = cont.impl_->requires_grad;

    // Deep copy the data using backend
    const size_t size_bytes = cont.numel() * dtype_size(cont.dtype());

    if (cont.impl_->device.type == Device::Type::CPU) {
        // CPU: direct memcpy
        std::memcpy(result.impl_->storage->data(),
                    cont.impl_->storage->data(),
                    size_bytes);
    } else {
        // GPU: use backend copy
        auto* backend = backend_registry().get_backend(cont.impl_->device.type);
        if (!backend) {
            throw std::runtime_error("Backend not available for clone");
        }
        backend->copy(result.impl_->storage->data(),
                     cont.impl_->storage->data(),
                     size_bytes,
                     CopyKind::DeviceToDevice);
    }

    return result;
}

auto Tensor::detach() const -> Tensor {
    auto result = clone();
    result.impl_->requires_grad = false;
    return result;
}

auto Tensor::contiguous() const -> Tensor {
    if (!impl_) {
        return *this;
    }

    if (is_contiguous()) {
        return *this;
    }

    // Dispatch to backend for contiguous operation
    // This properly handles both CPU and CUDA tensors
    std::vector<Tensor> inputs = {*this};
    return Dispatcher::dispatch("contiguous", inputs)[0];
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

// Scalar operations - use backend dispatch for device-agnostic execution
auto Tensor::operator+(float scalar) const -> Tensor {
    if (!impl_) return *this;

    // For CPU tensors, use fast direct access
    if (impl_->device.type == Device::Type::CPU) {
        auto result = clone();
        auto* data_ptr = result.data<float>();
        const int64_t n = numel();
        for (int64_t i = 0; i < n; ++i) {
            data_ptr[i] += scalar;
        }
        return result;
    }

    // For GPU tensors, create scalar tensor and use element-wise add
    auto scalar_tensor = full(std::vector<int64_t>(impl_->shape.begin(), impl_->shape.end()),
                             scalar, impl_->dtype, impl_->device);
    return *this + scalar_tensor;
}

auto Tensor::operator-(float scalar) const -> Tensor {
    if (!impl_) return *this;

    // For CPU tensors, use fast direct access
    if (impl_->device.type == Device::Type::CPU) {
        auto result = clone();
        auto* data_ptr = result.data<float>();
        const int64_t n = numel();
        for (int64_t i = 0; i < n; ++i) {
            data_ptr[i] -= scalar;
        }
        return result;
    }

    // For GPU tensors, create scalar tensor and use element-wise sub
    auto scalar_tensor = full(std::vector<int64_t>(impl_->shape.begin(), impl_->shape.end()),
                             scalar, impl_->dtype, impl_->device);
    return *this - scalar_tensor;
}

auto Tensor::operator*(float scalar) const -> Tensor {
    if (!impl_) return *this;

    // For CPU tensors, use fast direct access
    if (impl_->device.type == Device::Type::CPU) {
        auto result = clone();
        auto* data_ptr = result.data<float>();
        const int64_t n = numel();
        for (int64_t i = 0; i < n; ++i) {
            data_ptr[i] *= scalar;
        }
        return result;
    }

    // For GPU tensors, create scalar tensor and use element-wise mul
    auto scalar_tensor = full(std::vector<int64_t>(impl_->shape.begin(), impl_->shape.end()),
                             scalar, impl_->dtype, impl_->device);
    return *this * scalar_tensor;
}

auto Tensor::operator/(float scalar) const -> Tensor {
    if (!impl_) return *this;

    // For CPU tensors, use fast direct access
    if (impl_->device.type == Device::Type::CPU) {
        auto result = clone();
        auto* data_ptr = result.data<float>();
        const int64_t n = numel();
        for (int64_t i = 0; i < n; ++i) {
            data_ptr[i] /= scalar;
        }
        return result;
    }

    // For GPU tensors, create scalar tensor and use element-wise div
    auto scalar_tensor = full(std::vector<int64_t>(impl_->shape.begin(), impl_->shape.end()),
                             scalar, impl_->dtype, impl_->device);
    return *this / scalar_tensor;
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
    if (!impl_) {
        return *this;
    }
    auto* data_ptr = data<float>();
    const int64_t n = numel();
    for (int64_t i = 0; i < n; ++i) {
        data_ptr[i] = value;
    }
    return *this;
}

auto Tensor::zero_() -> Tensor& {
    return fill_(0.0f);
}

// Shape operations
auto Tensor::reshape(std::vector<int64_t> new_shape) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("Cannot reshape null tensor");
    }

    // Handle -1 inference (one dimension can be inferred)
    int64_t infer_dim = -1;
    int64_t total = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (infer_dim != -1) {
                throw std::invalid_argument("Only one dimension can be inferred");
            }
            infer_dim = static_cast<int64_t>(i);
        } else if (new_shape[i] <= 0) {
            throw std::invalid_argument("Invalid shape dimension");
        } else {
            total *= new_shape[i];
        }
    }

    if (infer_dim != -1) {
        if (numel() % total != 0) {
            throw std::invalid_argument("Cannot infer dimension");
        }
        new_shape[infer_dim] = numel() / total;
        total = numel();
    }

    // Validate total elements match
    if (total != numel()) {
        throw std::invalid_argument("Shape incompatible with number of elements");
    }

    // Try view first (zero-copy if contiguous)
    if (is_contiguous()) {
        return view(std::move(new_shape));
    }

    // Otherwise need to make contiguous first
    return contiguous().view(std::move(new_shape));
}

auto Tensor::view(std::vector<int64_t> new_shape) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("Cannot view null tensor");
    }

    if (!is_contiguous()) {
        throw std::runtime_error("View requires contiguous tensor. Use reshape() or contiguous() first.");
    }

    // Validate total elements
    int64_t total = 1;
    for (auto dim : new_shape) {
        total *= dim;
    }
    if (total != numel()) {
        throw std::invalid_argument("View shape incompatible with number of elements");
    }

    // Create new tensor sharing storage
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*impl_);
    result.impl_->shape = std::move(new_shape);
    result.impl_->strides = compute_strides(result.impl_->shape);

    return result;
}

auto Tensor::transpose(int64_t dim0, int64_t dim1) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("Cannot transpose null tensor");
    }

    const int64_t ndims = ndim();

    // Handle negative dimensions
    if (dim0 < 0) dim0 += ndims;
    if (dim1 < 0) dim1 += ndims;

    if (dim0 < 0 || dim0 >= ndims || dim1 < 0 || dim1 >= ndims) {
        throw std::out_of_range("Dimension out of range");
    }

    // For 2D tensors, default is transpose(0, 1)
    if (ndims == 2 && dim0 == 0 && dim1 == 1) {
        // Simple transpose - just swap shape and strides
        Tensor result;
        result.impl_ = std::make_shared<TensorImpl>(*impl_);
        std::swap(result.impl_->shape[0], result.impl_->shape[1]);
        std::swap(result.impl_->strides[0], result.impl_->strides[1]);
        return result;
    }

    // General case - swap specified dimensions
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*impl_);
    std::swap(result.impl_->shape[dim0], result.impl_->shape[dim1]);
    std::swap(result.impl_->strides[dim0], result.impl_->strides[dim1]);

    return result;
}

auto Tensor::permute(std::vector<int64_t> dims) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("Cannot permute null tensor");
    }

    const int64_t ndims = ndim();

    if (static_cast<int64_t>(dims.size()) != ndims) {
        throw std::invalid_argument("Permutation must have same number of dimensions as tensor");
    }

    // Validate dimensions and handle negative indices
    std::vector<bool> seen(ndims, false);
    for (size_t i = 0; i < dims.size(); ++i) {
        int64_t dim = dims[i];
        if (dim < 0) dim += ndims;

        if (dim < 0 || dim >= ndims) {
            throw std::out_of_range("Dimension out of range in permutation");
        }
        if (seen[dim]) {
            throw std::invalid_argument("Duplicate dimension in permutation");
        }
        seen[dim] = true;
        dims[i] = dim;  // Update with normalized value
    }

    // Create permuted tensor
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*impl_);

    std::vector<int64_t> new_shape(ndims);
    std::vector<int64_t> new_strides(ndims);

    for (int64_t i = 0; i < ndims; ++i) {
        new_shape[i] = impl_->shape[dims[i]];
        new_strides[i] = impl_->strides[dims[i]];
    }

    result.impl_->shape = std::move(new_shape);
    result.impl_->strides = std::move(new_strides);

    return result;
}

auto Tensor::squeeze(std::optional<int64_t> dim) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("Cannot squeeze null tensor");
    }

    const int64_t ndims = ndim();

    if (dim.has_value()) {
        // Squeeze specific dimension
        int64_t d = dim.value();
        if (d < 0) d += ndims;

        if (d < 0 || d >= ndims) {
            throw std::out_of_range("Dimension out of range");
        }

        if (impl_->shape[d] != 1) {
            throw std::runtime_error("Cannot squeeze dimension with size != 1");
        }

        // Remove the dimension
        Tensor result;
        result.impl_ = std::make_shared<TensorImpl>(*impl_);
        result.impl_->shape.erase(result.impl_->shape.begin() + d);
        result.impl_->strides.erase(result.impl_->strides.begin() + d);

        return result;
    } else {
        // Squeeze all dimensions with size 1
        std::vector<int64_t> new_shape;
        std::vector<int64_t> new_strides;

        for (int64_t i = 0; i < ndims; ++i) {
            if (impl_->shape[i] != 1) {
                new_shape.push_back(impl_->shape[i]);
                new_strides.push_back(impl_->strides[i]);
            }
        }

        // If all dimensions were 1, keep at least one
        if (new_shape.empty()) {
            new_shape.push_back(1);
            new_strides.push_back(1);
        }

        Tensor result;
        result.impl_ = std::make_shared<TensorImpl>(*impl_);
        result.impl_->shape = std::move(new_shape);
        result.impl_->strides = std::move(new_strides);

        return result;
    }
}

auto Tensor::unsqueeze(int64_t dim) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("Cannot unsqueeze null tensor");
    }

    const int64_t ndims = ndim();
    const int64_t new_ndims = ndims + 1;

    // Handle negative dimension (can be from -new_ndims to new_ndims-1)
    if (dim < 0) dim += new_ndims;

    if (dim < 0 || dim >= new_ndims) {
        throw std::out_of_range("Dimension out of range for unsqueeze");
    }

    // Insert dimension of size 1
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*impl_);

    result.impl_->shape.insert(result.impl_->shape.begin() + dim, 1);

    // Compute stride for new dimension (should be product of all following dims)
    int64_t new_stride = (dim < ndims) ? impl_->strides[dim] : 1;
    result.impl_->strides.insert(result.impl_->strides.begin() + dim, new_stride);

    return result;
}

auto Tensor::flatten(int64_t start_dim, int64_t end_dim) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("Cannot flatten null tensor");
    }

    const int64_t ndims = ndim();

    if (ndims == 0) {
        // Scalar tensor becomes 1D with size 1
        return view({1});
    }

    // Handle negative dimensions
    if (start_dim < 0) start_dim += ndims;
    if (end_dim < 0) end_dim += ndims;

    if (start_dim < 0 || start_dim >= ndims) {
        throw std::out_of_range("start_dim out of range");
    }
    if (end_dim < 0 || end_dim >= ndims) {
        throw std::out_of_range("end_dim out of range");
    }
    if (start_dim > end_dim) {
        throw std::invalid_argument("start_dim must be <= end_dim");
    }

    // Build new shape
    std::vector<int64_t> new_shape;

    // Dimensions before start_dim
    for (int64_t i = 0; i < start_dim; ++i) {
        new_shape.push_back(impl_->shape[i]);
    }

    // Flattened dimension
    int64_t flat_size = 1;
    for (int64_t i = start_dim; i <= end_dim; ++i) {
        flat_size *= impl_->shape[i];
    }
    new_shape.push_back(flat_size);

    // Dimensions after end_dim
    for (int64_t i = end_dim + 1; i < ndims; ++i) {
        new_shape.push_back(impl_->shape[i]);
    }

    return reshape(std::move(new_shape));
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
