#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/ops/indexing.hpp"
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
template<typename T>
auto Tensor::data() -> T* {
    return static_cast<T*>(impl_->storage->data()) + impl_->offset;
}

template<typename T>
auto Tensor::data() const -> const T* {
    return static_cast<const T*>(impl_->storage->data()) + impl_->offset;
}

// Explicit instantiations
template auto Tensor::data<float>() -> float*;
template auto Tensor::data<float>() const -> const float*;
template auto Tensor::data<double>() -> double*;
template auto Tensor::data<double>() const -> const double*;
template auto Tensor::data<Float16>() -> Float16*;
template auto Tensor::data<Float16>() const -> const Float16*;
template auto Tensor::data<BFloat16>() -> BFloat16*;
template auto Tensor::data<BFloat16>() const -> const BFloat16*;
template auto Tensor::data<int8_t>() -> int8_t*;
template auto Tensor::data<int8_t>() const -> const int8_t*;
template auto Tensor::data<int32_t>() -> int32_t*;
template auto Tensor::data<int32_t>() const -> const int32_t*;
template auto Tensor::data<int64_t>() -> int64_t*;
template auto Tensor::data<int64_t>() const -> const int64_t*;
template auto Tensor::data<uint8_t>() -> uint8_t*;
template auto Tensor::data<uint8_t>() const -> const uint8_t*;
template auto Tensor::data<bool>() -> bool*;
template auto Tensor::data<bool>() const -> const bool*;

// Additional instantiations for const-qualified template parameters (used by quantization)
template auto Tensor::data<const float>() -> const float*;
template auto Tensor::data<const float>() const -> const float*;
template auto Tensor::data<const int>() const -> const int*;
template auto Tensor::data<const unsigned char>() const -> const unsigned char*;
template auto Tensor::data<const signed char>() const -> const signed char*;

// Template instantiations for item<T>() - extract scalar from single-element tensor
template<> auto Tensor::item<float>() const -> float {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    if (dtype() != DType::Float32) {
        throw std::runtime_error("Type mismatch: tensor dtype is not Float32");
    }
    // For GPU tensors, copy to CPU first
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *cpu_tensor.data<float>();
    }
    return *data<float>();
}

template<> auto Tensor::item<double>() const -> double {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    if (dtype() != DType::Float64) {
        throw std::runtime_error("Type mismatch: tensor dtype is not Float64");
    }
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *cpu_tensor.data<double>();
    }
    return *data<double>();
}

template<> auto Tensor::item<int32_t>() const -> int32_t {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    if (dtype() != DType::Int32) {
        throw std::runtime_error("Type mismatch: tensor dtype is not Int32");
    }
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *cpu_tensor.data<int32_t>();
    }
    return *data<int32_t>();
}

template<> auto Tensor::item<int64_t>() const -> int64_t {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    if (dtype() != DType::Int64) {
        throw std::runtime_error("Type mismatch: tensor dtype is not Int64");
    }
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *cpu_tensor.data<int64_t>();
    }
    return *data<int64_t>();
}

template<> auto Tensor::item<int16_t>() const -> int16_t {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    if (dtype() != DType::Int16) {
        throw std::runtime_error("Type mismatch: tensor dtype is not Int16");
    }
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *reinterpret_cast<const int16_t*>(cpu_tensor.impl_->storage->data());
    }
    return *reinterpret_cast<const int16_t*>(impl_->storage->data());
}

template<> auto Tensor::item<int8_t>() const -> int8_t {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    if (dtype() != DType::Int8) {
        throw std::runtime_error("Type mismatch: tensor dtype is not Int8");
    }
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *reinterpret_cast<const int8_t*>(cpu_tensor.impl_->storage->data());
    }
    return *reinterpret_cast<const int8_t*>(impl_->storage->data());
}

template<> auto Tensor::item<uint8_t>() const -> uint8_t {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    if (dtype() != DType::UInt8) {
        throw std::runtime_error("Type mismatch: tensor dtype is not UInt8");
    }
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *cpu_tensor.data<uint8_t>();
    }
    return *data<uint8_t>();
}

template<> auto Tensor::item<uint16_t>() const -> uint16_t {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    if (dtype() != DType::UInt16) {
        throw std::runtime_error("Type mismatch: tensor dtype is not UInt16");
    }
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *reinterpret_cast<const uint16_t*>(cpu_tensor.impl_->storage->data());
    }
    return *reinterpret_cast<const uint16_t*>(impl_->storage->data());
}

template<> auto Tensor::item<uint32_t>() const -> uint32_t {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    if (dtype() != DType::UInt32) {
        throw std::runtime_error("Type mismatch: tensor dtype is not UInt32");
    }
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *reinterpret_cast<const uint32_t*>(cpu_tensor.impl_->storage->data());
    }
    return *reinterpret_cast<const uint32_t*>(impl_->storage->data());
}

template<> auto Tensor::item<uint64_t>() const -> uint64_t {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    if (dtype() != DType::UInt64) {
        throw std::runtime_error("Type mismatch: tensor dtype is not UInt64");
    }
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *reinterpret_cast<const uint64_t*>(cpu_tensor.impl_->storage->data());
    }
    return *reinterpret_cast<const uint64_t*>(impl_->storage->data());
}

template<> auto Tensor::item<bool>() const -> bool {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    if (dtype() != DType::Bool) {
        throw std::runtime_error("Type mismatch: tensor dtype is not Bool");
    }
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *cpu_tensor.data<bool>();
    }
    return *data<bool>();
}

template<> auto Tensor::item<std::complex<float>>() const -> std::complex<float> {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    if (dtype() != DType::Complex64) {
        throw std::runtime_error("Type mismatch: tensor dtype is not Complex64");
    }
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *reinterpret_cast<const std::complex<float>*>(cpu_tensor.impl_->storage->data());
    }
    return *reinterpret_cast<const std::complex<float>*>(impl_->storage->data());
}

template<> auto Tensor::item<std::complex<double>>() const -> std::complex<double> {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    if (dtype() != DType::Complex128) {
        throw std::runtime_error("Type mismatch: tensor dtype is not Complex128");
    }
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *reinterpret_cast<const std::complex<double>*>(cpu_tensor.impl_->storage->data());
    }
    return *reinterpret_cast<const std::complex<double>*>(impl_->storage->data());
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
        const size_t size_bytes = numel() * dtype_size();
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
        const size_t element_size = dtype_size();
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
    const size_t size_bytes = cont.numel() * cont.dtype_size();

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
    if (!impl_) {
        return *this;
    }

    // If already the target dtype, return as-is
    if (impl_->dtype == dtype) {
        return *this;
    }

    // For dtype conversion, we need to work on CPU for element-wise conversion
    // If tensor is on GPU, move to CPU first
    const bool was_on_gpu = impl_->device.type != Device::Type::CPU;
    Tensor cpu_tensor = was_on_gpu ? cpu() : *this;

    // Ensure contiguous layout for efficient conversion
    if (!cpu_tensor.is_contiguous()) {
        cpu_tensor = cpu_tensor.contiguous();
    }

    // Create output tensor with target dtype on CPU
    Tensor result(cpu_tensor.impl_->shape, dtype, Device::cpu());
    result.impl_->requires_grad = cpu_tensor.impl_->requires_grad;

    const int64_t n = cpu_tensor.numel();
    const DType src_dtype = cpu_tensor.impl_->dtype;

    // Dispatch based on source and destination dtypes
    // Use template helper to avoid code duplication
    auto convert_elements = [&]<typename SrcT, typename DstT>() {
        const SrcT* src_ptr = cpu_tensor.data<SrcT>();
        DstT* dst_ptr = result.data<DstT>();

        for (int64_t i = 0; i < n; ++i) {
            if constexpr (std::is_same_v<SrcT, DstT>) {
                dst_ptr[i] = src_ptr[i];
            } else if constexpr (std::is_same_v<SrcT, Float16> || std::is_same_v<SrcT, BFloat16>) {
                // Convert half-precision to float, then to target
                float intermediate = static_cast<float>(src_ptr[i]);
                if constexpr (std::is_same_v<DstT, Float16> || std::is_same_v<DstT, BFloat16>) {
                    dst_ptr[i] = DstT(intermediate);
                } else if constexpr (std::is_same_v<DstT, std::complex<float>>) {
                    dst_ptr[i] = std::complex<float>(intermediate, 0.0f);
                } else if constexpr (std::is_same_v<DstT, std::complex<double>>) {
                    dst_ptr[i] = std::complex<double>(static_cast<double>(intermediate), 0.0);
                } else {
                    dst_ptr[i] = static_cast<DstT>(intermediate);
                }
            } else if constexpr (std::is_same_v<DstT, Float16> || std::is_same_v<DstT, BFloat16>) {
                // Convert source to float, then to half-precision
                float intermediate;
                if constexpr (std::is_same_v<SrcT, std::complex<float>>) {
                    intermediate = src_ptr[i].real();
                } else if constexpr (std::is_same_v<SrcT, std::complex<double>>) {
                    intermediate = static_cast<float>(src_ptr[i].real());
                } else {
                    intermediate = static_cast<float>(src_ptr[i]);
                }
                dst_ptr[i] = DstT(intermediate);
            } else if constexpr (std::is_same_v<SrcT, std::complex<float>> || std::is_same_v<SrcT, std::complex<double>>) {
                // Convert from complex (take real part)
                if constexpr (std::is_same_v<DstT, std::complex<float>>) {
                    dst_ptr[i] = std::complex<float>(src_ptr[i]);
                } else if constexpr (std::is_same_v<DstT, std::complex<double>>) {
                    dst_ptr[i] = std::complex<double>(src_ptr[i]);
                } else {
                    dst_ptr[i] = static_cast<DstT>(src_ptr[i].real());
                }
            } else if constexpr (std::is_same_v<DstT, std::complex<float>> || std::is_same_v<DstT, std::complex<double>>) {
                // Convert to complex (set imaginary to 0)
                using RealType = typename DstT::value_type;
                dst_ptr[i] = DstT(static_cast<RealType>(src_ptr[i]), RealType{0});
            } else {
                // Standard numeric conversion
                dst_ptr[i] = static_cast<DstT>(src_ptr[i]);
            }
        }
    };

    // Macro to reduce code duplication for source type dispatch
    #define DISPATCH_SRC_DTYPE(src_dtype, SrcT) \
        case src_dtype: { \
            switch (dtype) { \
                case DType::Float32: convert_elements.template operator()<SrcT, float>(); break; \
                case DType::Float64: convert_elements.template operator()<SrcT, double>(); break; \
                case DType::Float16: convert_elements.template operator()<SrcT, Float16>(); break; \
                case DType::BFloat16: convert_elements.template operator()<SrcT, BFloat16>(); break; \
                case DType::Int8: convert_elements.template operator()<SrcT, int8_t>(); break; \
                case DType::Int16: convert_elements.template operator()<SrcT, int16_t>(); break; \
                case DType::Int32: convert_elements.template operator()<SrcT, int32_t>(); break; \
                case DType::Int64: convert_elements.template operator()<SrcT, int64_t>(); break; \
                case DType::UInt8: convert_elements.template operator()<SrcT, uint8_t>(); break; \
                case DType::UInt16: convert_elements.template operator()<SrcT, uint16_t>(); break; \
                case DType::UInt32: convert_elements.template operator()<SrcT, uint32_t>(); break; \
                case DType::UInt64: convert_elements.template operator()<SrcT, uint64_t>(); break; \
                case DType::Bool: convert_elements.template operator()<SrcT, bool>(); break; \
                case DType::Complex64: convert_elements.template operator()<SrcT, std::complex<float>>(); break; \
                case DType::Complex128: convert_elements.template operator()<SrcT, std::complex<double>>(); break; \
            } \
            break; \
        }

    // Dispatch based on source dtype
    switch (src_dtype) {
        DISPATCH_SRC_DTYPE(DType::Float32, float)
        DISPATCH_SRC_DTYPE(DType::Float64, double)
        DISPATCH_SRC_DTYPE(DType::Float16, Float16)
        DISPATCH_SRC_DTYPE(DType::BFloat16, BFloat16)
        DISPATCH_SRC_DTYPE(DType::Int8, int8_t)
        DISPATCH_SRC_DTYPE(DType::Int16, int16_t)
        DISPATCH_SRC_DTYPE(DType::Int32, int32_t)
        DISPATCH_SRC_DTYPE(DType::Int64, int64_t)
        DISPATCH_SRC_DTYPE(DType::UInt8, uint8_t)
        DISPATCH_SRC_DTYPE(DType::UInt16, uint16_t)
        DISPATCH_SRC_DTYPE(DType::UInt32, uint32_t)
        DISPATCH_SRC_DTYPE(DType::UInt64, uint64_t)
        DISPATCH_SRC_DTYPE(DType::Bool, bool)
        DISPATCH_SRC_DTYPE(DType::Complex64, std::complex<float>)
        DISPATCH_SRC_DTYPE(DType::Complex128, std::complex<double>)
    }

    #undef DISPATCH_SRC_DTYPE

    // If original tensor was on GPU, move result back to GPU
    if (was_on_gpu) {
        return result.to(impl_->device);
    }

    return result;
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

    // Create new tensor with same shape, dtype, and device
    Tensor result(impl_->shape, impl_->dtype, impl_->device);
    result.impl_->requires_grad = impl_->requires_grad;

    // Copy data
    const size_t size_bytes = numel() * dtype_size();

    if (impl_->device.type == Device::Type::CPU) {
        // CPU copy
        std::memcpy(result.impl_->storage->data(),
                    impl_->storage->data(),
                    size_bytes);
    } else {
        // Device copy
        auto* backend = backend_registry().get_backend(impl_->device.type);
        if (!backend) {
            throw std::runtime_error("Backend not available for device");
        }
        backend->copy(result.impl_->storage->data(),
                      impl_->storage->data(),
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

// Scalar operations - use tensor path for device-agnostic execution
auto Tensor::operator+(float scalar) const -> Tensor {
    if (!impl_) return *this;

    // Create scalar tensor and use element-wise add
    auto scalar_tensor = full(std::vector<int64_t>(impl_->shape.begin(), impl_->shape.end()),
                             scalar, impl_->dtype, impl_->device);
    return *this + scalar_tensor;
}

auto Tensor::operator-(float scalar) const -> Tensor {
    if (!impl_) return *this;

    // Create scalar tensor and use element-wise sub
    auto scalar_tensor = full(std::vector<int64_t>(impl_->shape.begin(), impl_->shape.end()),
                             scalar, impl_->dtype, impl_->device);
    return *this - scalar_tensor;
}

auto Tensor::operator*(float scalar) const -> Tensor {
    if (!impl_) return *this;

    // Create scalar tensor and use element-wise mul
    auto scalar_tensor = full(std::vector<int64_t>(impl_->shape.begin(), impl_->shape.end()),
                             scalar, impl_->dtype, impl_->device);
    return *this * scalar_tensor;
}

auto Tensor::operator/(float scalar) const -> Tensor {
    if (!impl_) return *this;

    // Create scalar tensor and use element-wise div
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

    // Direct implementation - fill tensor data with the value
    const int64_t n = numel();

    // Handle different dtypes
    switch (impl_->dtype) {
        case DType::Float32: {
            float* ptr = data<float>();
            for (int64_t i = 0; i < n; ++i) {
                ptr[i] = value;
            }
            break;
        }
        case DType::Float64: {
            double* ptr = data<double>();
            for (int64_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<double>(value);
            }
            break;
        }
        case DType::Int32: {
            int32_t* ptr = data<int32_t>();
            for (int64_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<int32_t>(value);
            }
            break;
        }
        case DType::Int64: {
            int64_t* ptr = data<int64_t>();
            for (int64_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<int64_t>(value);
            }
            break;
        }
        case DType::UInt8: {
            uint8_t* ptr = data<uint8_t>();
            for (int64_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<uint8_t>(value);
            }
            break;
        }
        default:
            throw std::runtime_error("fill_ not supported for this dtype");
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
        if (total == 0) {
            throw std::invalid_argument("Cannot infer dimension: product of known dimensions is zero");
        }
        if (numel() % total != 0) {
            throw std::invalid_argument("Cannot infer dimension");
        }
        new_shape[infer_dim] = numel() / total;
        total = numel();
    }

    // Validate total elements match
    if (total != numel()) {
        std::string current_shape_str = "[";
        for (size_t i = 0; i < shape().size(); ++i) {
            current_shape_str += std::to_string(shape()[i]);
            if (i < shape().size() - 1) current_shape_str += ", ";
        }
        current_shape_str += "]";

        std::string target_shape_str = "[";
        for (size_t i = 0; i < new_shape.size(); ++i) {
            target_shape_str += std::to_string(new_shape[i]);
            if (i < new_shape.size() - 1) target_shape_str += ", ";
        }
        target_shape_str += "]";

        throw std::invalid_argument(
            "Shape incompatible with number of elements: trying to reshape " +
            current_shape_str + " (numel=" + std::to_string(numel()) + ") to " +
            target_shape_str + " (total=" + std::to_string(total) + ")"
        );
    }

    // Build attributes with new shape
    OpAttributes attrs;
    std::string shape_str;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (i > 0) shape_str += ",";
        shape_str += std::to_string(new_shape[i]);
    }
    attrs["shape"] = shape_str;

    // Dispatch to backend for reshape operation
    std::vector<Tensor> inputs = {*this};
    return Dispatcher::dispatch("reshape", inputs, attrs)[0];
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

    // Build attributes with new shape
    OpAttributes attrs;
    std::string shape_str;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (i > 0) shape_str += ",";
        shape_str += std::to_string(new_shape[i]);
    }
    attrs["shape"] = shape_str;

    // Dispatch to backend (uses reshape since view is just reshape on contiguous tensor)
    std::vector<Tensor> inputs = {*this};
    return Dispatcher::dispatch("reshape", inputs, attrs)[0];
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

auto Tensor::nonzero() const -> Tensor {
    // Forward to the ops function
    return tenzor::nonzero(*this);
}

// Indexing
auto Tensor::operator[](int64_t idx) const -> Tensor {
    // TODO: Implement indexing
    return *this;
}

auto Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("Cannot slice null tensor");
    }

    const int64_t ndims = ndim();

    // Normalize dimension
    if (dim < 0) dim += ndims;
    if (dim < 0 || dim >= ndims) {
        throw std::out_of_range("Dimension out of range for slice");
    }

    const int64_t dim_size = impl_->shape[dim];

    // Normalize start and end indices
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;

    // Clamp to valid range
    start = std::clamp(start, int64_t{0}, dim_size);
    end = std::clamp(end, int64_t{0}, dim_size);

    if (step <= 0) {
        throw std::invalid_argument("Step must be positive");
    }

    // Calculate new dimension size
    int64_t new_dim_size = 0;
    if (end > start) {
        new_dim_size = (end - start + step - 1) / step;  // Ceiling division
    }

    // Create new tensor that shares storage with original
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*impl_);

    // Update shape for sliced dimension
    result.impl_->shape[dim] = new_dim_size;

    // Update offset to start at the correct position
    result.impl_->offset += start * impl_->strides[dim];

    // Update stride if step != 1
    if (step != 1) {
        result.impl_->strides[dim] *= step;
    }

    return result;
}

// Comparison operators - delegate to dedicated comparison functions
auto Tensor::operator==(const Tensor& other) const -> Tensor {
    return tenzor::eq(*this, other);
}

auto Tensor::operator!=(const Tensor& other) const -> Tensor {
    return tenzor::ne(*this, other);
}

auto Tensor::operator<(const Tensor& other) const -> Tensor {
    return tenzor::lt(*this, other);
}

auto Tensor::operator>(const Tensor& other) const -> Tensor {
    return tenzor::gt(*this, other);
}

auto Tensor::operator<=(const Tensor& other) const -> Tensor {
    return tenzor::le(*this, other);
}

auto Tensor::operator>=(const Tensor& other) const -> Tensor {
    return tenzor::ge(*this, other);
}

// ============================================================================
// Phase 8 Utility Methods
// ============================================================================

auto Tensor::dtype_size() const noexcept -> size_t {
    if (!impl_) return 0;
    return tenzor::dtype_size(impl_->dtype);
}

auto Tensor::data_ptr() -> void* {
    if (!impl_ || !impl_->storage) {
        return nullptr;
    }
    return impl_->storage->data();
}

auto Tensor::data_ptr() const -> const void* {
    if (!impl_ || !impl_->storage) {
        return nullptr;
    }
    return impl_->storage->data();
}

auto Tensor::zeros_like(const Tensor& other) -> Tensor {
    if (!other.impl_) {
        return Tensor();
    }

    // Create zero tensor with same shape, dtype, and device
    std::vector<int64_t> shape_vec(other.shape().begin(), other.shape().end());
    return tenzor::zeros(shape_vec, other.dtype(), other.device());
}

} // namespace tenzor
