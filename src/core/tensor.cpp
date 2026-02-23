#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/utils/error.hpp"
#include <numeric>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>
#include <stdexcept>

namespace tenzor {

// TensorImpl implementation
TensorImpl::TensorImpl(std::vector<int64_t> shape_, DType dtype_, Device device_,
                       bool zero_init)
    : shape(std::move(shape_)), dtype(dtype_), device(device_) {
    is_contiguous_cache_.store(1, std::memory_order_relaxed);  // freshly constructed = contiguous

    // Compute strides
    strides = compute_strides(this->shape);

    // Allocate storage
    size_t size_bytes = numel() * dtype_size(dtype);

    // UNIFIED PATH: All devices go through backend allocator
    auto* backend = backend_registry().get_backend(device.type);
    if (!backend) {
        throw std::runtime_error("Backend not available for device: " + device.to_string());
    }

    // Allocate memory via backend (which may use caching allocator)
    void* ptr = backend->allocate(size_bytes, device.index);
    if (!ptr && size_bytes > 0) {
        throw std::runtime_error("Failed to allocate memory for tensor");
    }

    // Use DeviceStorage for ALL devices (including CPU)
    storage = std::make_shared<DeviceStorage>(ptr, size_bytes, device, backend);

    // Zero-initialize if requested (CPU only - device kernels handle their own init)
    if (zero_init && device.type == Device::Type::CPU && size_bytes > 0) {
        std::memset(ptr, 0, size_bytes);
    }
}

auto TensorImpl::numel() const -> int64_t {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                          std::multiplies<int64_t>{});
}

auto TensorImpl::is_contiguous() const -> bool {
    // Return cached result if available (atomic load for thread safety)
    auto cached = is_contiguous_cache_.load(std::memory_order_relaxed);
    if (cached >= 0) {
        return cached == 1;
    }
    // A tensor is contiguous if its strides match the expected row-major strides.
    // Offset does not affect contiguity — a slice can be contiguous at a non-zero offset.
    auto expected_strides = compute_strides(shape);
    bool result = (strides == expected_strides);
    is_contiguous_cache_.store(result ? 1 : 0, std::memory_order_relaxed);
    return result;
}

// Tensor implementation
Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device device)
    : impl_(std::make_shared<TensorImpl>(std::move(shape), dtype, device, true)) {}

auto Tensor::empty_uninitialized(std::vector<int64_t> shape, DType dtype, Device device)
    -> Tensor {
    Tensor t;
    t.impl_ = std::make_shared<TensorImpl>(std::move(shape), dtype, device, false);
    return t;
}

auto Tensor::shape() const -> std::span<const int64_t> {
    if (!impl_) throw std::runtime_error("Operation on uninitialized tensor");
    return impl_->shape;
}

auto Tensor::strides() const -> std::span<const int64_t> {
    if (!impl_) throw std::runtime_error("Operation on uninitialized tensor");
    return impl_->strides;
}

auto Tensor::ndim() const -> int64_t {
    if (!impl_) throw std::runtime_error("Operation on uninitialized tensor");
    return static_cast<int64_t>(impl_->shape.size());
}

auto Tensor::numel() const -> int64_t {
    if (!impl_) throw std::runtime_error("Operation on uninitialized tensor");
    return impl_->numel();
}

auto Tensor::dtype() const -> DType {
    if (!impl_) throw std::runtime_error("Operation on uninitialized tensor");
    return impl_->dtype;
}

auto Tensor::device() const -> const Device& {
    if (!impl_) throw std::runtime_error("Operation on uninitialized tensor");
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
    if (!impl_ || !impl_->storage) {
        throw std::runtime_error("Cannot access data of uninitialized tensor");
    }
    // Allow byte-level access (uint8_t) for any dtype - used for raw memory operations
    // For other types, validate dtype matches requested type
    using CleanT = std::remove_const_t<T>;
    if constexpr (!std::is_same_v<CleanT, uint8_t> && !std::is_same_v<CleanT, char> &&
                  !std::is_same_v<CleanT, unsigned char> && !std::is_same_v<CleanT, signed char>) {
        constexpr DType expected_dtype = type_to_dtype_v<CleanT>;
        if (impl_->dtype != expected_dtype) {
            throw DTypeException("Type mismatch: requested type does not match tensor dtype (expected " +
                std::string(dtype_name(impl_->dtype)) + ")");
        }
    }
    // Bounds check: compute max reachable element using strides (handles non-contiguous)
    {
        size_t storage_elements = impl_->storage->size_bytes() / tenzor::dtype_size(impl_->dtype);
        size_t max_offset = static_cast<size_t>(impl_->offset);
        for (int64_t d = 0; d < ndim(); ++d) {
            if (impl_->shape[d] > 0)
                max_offset += static_cast<size_t>(impl_->shape[d] - 1) * static_cast<size_t>(std::abs(impl_->strides[d]));
        }
        if (max_offset >= storage_elements) {
            throw std::out_of_range("Tensor data access: max reachable offset exceeds storage bounds");
        }
    }
    // For byte-level access types, scale offset by dtype size so pointer arithmetic
    // advances by the correct number of bytes (offset is in elements, not bytes)
    if constexpr (std::is_same_v<CleanT, uint8_t> || std::is_same_v<CleanT, char> ||
                  std::is_same_v<CleanT, unsigned char> || std::is_same_v<CleanT, signed char>) {
        auto* base = static_cast<uint8_t*>(impl_->storage->data());
        return reinterpret_cast<T*>(base + impl_->offset * tenzor::dtype_size(impl_->dtype));
    } else {
        return static_cast<T*>(impl_->storage->data()) + impl_->offset;
    }
}

template<typename T>
auto Tensor::data() const -> const T* {
    if (!impl_ || !impl_->storage) {
        throw std::runtime_error("Cannot access data of uninitialized tensor");
    }
    // Allow byte-level access (uint8_t) for any dtype - used for raw memory operations
    // For other types, validate dtype matches requested type
    using CleanT = std::remove_const_t<T>;
    if constexpr (!std::is_same_v<CleanT, uint8_t> && !std::is_same_v<CleanT, char> &&
                  !std::is_same_v<CleanT, unsigned char> && !std::is_same_v<CleanT, signed char>) {
        constexpr DType expected_dtype = type_to_dtype_v<CleanT>;
        if (impl_->dtype != expected_dtype) {
            throw DTypeException("Type mismatch: requested type does not match tensor dtype (expected " +
                std::string(dtype_name(impl_->dtype)) + ")");
        }
    }
    // Bounds check: compute max reachable element using strides (handles non-contiguous)
    {
        size_t storage_elements = impl_->storage->size_bytes() / tenzor::dtype_size(impl_->dtype);
        size_t max_offset = static_cast<size_t>(impl_->offset);
        for (int64_t d = 0; d < ndim(); ++d) {
            if (impl_->shape[d] > 0)
                max_offset += static_cast<size_t>(impl_->shape[d] - 1) * static_cast<size_t>(std::abs(impl_->strides[d]));
        }
        if (max_offset >= storage_elements) {
            throw std::out_of_range("Tensor data access: max reachable offset exceeds storage bounds");
        }
    }
    // For byte-level access types, scale offset by dtype size so pointer arithmetic
    // advances by the correct number of bytes (offset is in elements, not bytes)
    if constexpr (std::is_same_v<CleanT, uint8_t> || std::is_same_v<CleanT, char> ||
                  std::is_same_v<CleanT, unsigned char> || std::is_same_v<CleanT, signed char>) {
        const auto* base = static_cast<const uint8_t*>(impl_->storage->data());
        return reinterpret_cast<const T*>(base + impl_->offset * tenzor::dtype_size(impl_->dtype));
    } else {
        return static_cast<const T*>(impl_->storage->data()) + impl_->offset;
    }
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
template auto Tensor::data<int16_t>() -> int16_t*;
template auto Tensor::data<int16_t>() const -> const int16_t*;
template auto Tensor::data<uint16_t>() -> uint16_t*;
template auto Tensor::data<uint16_t>() const -> const uint16_t*;
template auto Tensor::data<uint32_t>() -> uint32_t*;
template auto Tensor::data<uint32_t>() const -> const uint32_t*;
template auto Tensor::data<uint64_t>() -> uint64_t*;
template auto Tensor::data<uint64_t>() const -> const uint64_t*;
template auto Tensor::data<bool>() -> bool*;
template auto Tensor::data<bool>() const -> const bool*;

// Additional instantiations for const-qualified template parameters (used by quantization)
template auto Tensor::data<const float>() -> const float*;
template auto Tensor::data<const float>() const -> const float*;
template auto Tensor::data<const unsigned char>() const -> const unsigned char*;
template auto Tensor::data<const signed char>() const -> const signed char*;
template auto Tensor::data<const int>() const -> const int*;

// Generic item<T>() implementation — replaces 14 copy-pasted specializations
template<typename T>
requires ScalarType<T>
auto Tensor::item() const -> T {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for single-element tensors");
    }
    constexpr DType expected = type_to_dtype<T>::value;
    if (dtype() != expected) {
        throw std::runtime_error(
            std::string("Type mismatch: tensor dtype is ") +
            std::string(dtype_name(dtype())) + " but item<" +
            std::string(dtype_name(expected)) + ">() was called");
    }
    if (device().type != Device::Type::CPU) {
        auto cpu_tensor = cpu();
        return *cpu_tensor.data<T>();
    }
    return *data<T>();
}

// Explicit instantiations for all scalar types
template auto Tensor::item<float>() const -> float;
template auto Tensor::item<Float16>() const -> Float16;
template auto Tensor::item<BFloat16>() const -> BFloat16;
template auto Tensor::item<double>() const -> double;
template auto Tensor::item<int32_t>() const -> int32_t;
template auto Tensor::item<int64_t>() const -> int64_t;
template auto Tensor::item<int16_t>() const -> int16_t;
template auto Tensor::item<int8_t>() const -> int8_t;
template auto Tensor::item<uint8_t>() const -> uint8_t;
template auto Tensor::item<uint16_t>() const -> uint16_t;
template auto Tensor::item<uint32_t>() const -> uint32_t;
template auto Tensor::item<uint64_t>() const -> uint64_t;
template auto Tensor::item<bool>() const -> bool;
template auto Tensor::item<std::complex<float>>() const -> std::complex<float>;
template auto Tensor::item<std::complex<double>>() const -> std::complex<double>;

// Core tensor operation implementations
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

    // Non-contiguous GPU tensors: make contiguous on-device first (single GPU kernel),
    // then fall through to the normal contiguous transfer path below.
    if (!is_contiguous() && impl_->device.type != Device::Type::CPU) {
        return contiguous().to(device);
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

    // Synchronize source device before copy if transferring from device
    if (copy_kind == CopyKind::DeviceToHost || copy_kind == CopyKind::DeviceToDevice) {
        auto* src_backend = backend_registry().get_backend(cont.impl_->device.type);
        if (src_backend) {
            src_backend->synchronize(cont.impl_->device.index);
        }
    }

    // Perform the copy
    // IMPORTANT: Calculate source pointer with offset in bytes, not elements
    // cont.impl_->offset is in elements, so we multiply by dtype_size() to get bytes
    const void* src_ptr = static_cast<const uint8_t*>(cont.impl_->storage->data()) +
                          (cont.impl_->offset * tenzor::dtype_size(cont.impl_->dtype));

    backend->copy(result.impl_->storage->data(),
                  src_ptr,
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

    // Try GPU-side Cast kernel to avoid costly CPU round-trip (GPU -> CPU -> cast -> GPU).
    // If a Cast kernel is registered for the current device, dispatch directly on-device.
    if (impl_->device.type != Device::Type::CPU &&
        is_op_supported(OpId::Cast, impl_->device.type)) {
        // Ensure contiguous layout for the cast kernel
        Tensor src = is_contiguous() ? *this : contiguous();
        OpAttributes attrs;
        attrs["target_dtype"] = std::to_string(static_cast<uint8_t>(dtype));
        Tensor result = dispatch_single(OpId::Cast, std::span<const Tensor>(&src, 1), attrs);
        result.impl_->requires_grad = impl_->requires_grad;
        return result;
    }

    // Fallback: convert on CPU (for CPU tensors or backends without Cast kernel)
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

    // Ensure source is contiguous before memcpy — non-contiguous tensors
    // (after transpose, slice with stride) have gaps that memcpy would corrupt
    Tensor src = is_contiguous() ? *this : contiguous();

    // Create new tensor with same shape, dtype, and device
    Tensor result(impl_->shape, impl_->dtype, impl_->device);
    result.impl_->requires_grad = impl_->requires_grad;

    // Copy data from contiguous source
    const size_t size_bytes = src.numel() * src.dtype_size();

    if (impl_->device.type == Device::Type::CPU) {
        // CPU copy
        std::memcpy(result.impl_->storage->data(),
                    src.data_ptr(),
                    size_bytes);
    } else {
        // Device copy
        auto* backend = backend_registry().get_backend(impl_->device.type);
        if (!backend) {
            throw std::runtime_error("Backend not available for device");
        }
        backend->copy(result.impl_->storage->data(),
                      src.data_ptr(),
                      size_bytes,
                      CopyKind::DeviceToDevice);
    }

    return result;
}

auto Tensor::detach() const -> Tensor {
    // Share storage (zero-copy) like view() — no need to copy data
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*impl_);
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
    return dispatch(OpId::Contiguous, inputs)[0];
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

// Scalar operations — dispatch to optimized scalar overloads
auto Tensor::operator+(double scalar) const -> Tensor {
    return tenzor::add(*this, scalar);
}

auto Tensor::operator-(double scalar) const -> Tensor {
    return tenzor::sub(*this, scalar);
}

auto Tensor::operator*(double scalar) const -> Tensor {
    return tenzor::mul(*this, scalar);
}

auto Tensor::operator/(double scalar) const -> Tensor {
    return tenzor::div(*this, scalar);
}

// In-place operations — dispatch through in-place kernels so views/aliases see updates
// If the other operand shares storage with this tensor (e.g. a += a),
// clone it first to avoid undefined behavior from reading and writing
// the same buffer simultaneously.
auto Tensor::operator+=(const Tensor& other) -> Tensor& {
    auto& table = DispatchTableRegistry::get_table(impl_->device.type);
    bool aliased = impl_ && other.impl_ && impl_->storage && other.impl_->storage &&
                   impl_->storage.get() == other.impl_->storage.get();
    std::array<Tensor, 1> others = {aliased ? other.clone() : other};
    table.dispatch_inplace(OpId::AddInplace, *this, others);
    return *this;
}

auto Tensor::operator-=(const Tensor& other) -> Tensor& {
    auto& table = DispatchTableRegistry::get_table(impl_->device.type);
    bool aliased = impl_ && other.impl_ && impl_->storage && other.impl_->storage &&
                   impl_->storage.get() == other.impl_->storage.get();
    std::array<Tensor, 1> others = {aliased ? other.clone() : other};
    table.dispatch_inplace(OpId::SubInplace, *this, others);
    return *this;
}

auto Tensor::operator*=(const Tensor& other) -> Tensor& {
    auto& table = DispatchTableRegistry::get_table(impl_->device.type);
    bool aliased = impl_ && other.impl_ && impl_->storage && other.impl_->storage &&
                   impl_->storage.get() == other.impl_->storage.get();
    std::array<Tensor, 1> others = {aliased ? other.clone() : other};
    table.dispatch_inplace(OpId::MulInplace, *this, others);
    return *this;
}

auto Tensor::operator/=(const Tensor& other) -> Tensor& {
    auto& table = DispatchTableRegistry::get_table(impl_->device.type);
    bool aliased = impl_ && other.impl_ && impl_->storage && other.impl_->storage &&
                   impl_->storage.get() == other.impl_->storage.get();
    std::array<Tensor, 1> others = {aliased ? other.clone() : other};
    table.dispatch_inplace(OpId::DivInplace, *this, others);
    return *this;
}

auto Tensor::fill_(float value) -> Tensor& {
    if (!impl_) {
        return *this;
    }

    // Non-contiguous tensors: iterate using strides to fill each element in-place
    if (!is_contiguous()) {
        if (device().type != Device::Type::CPU) {
            // TODO: Non-CPU non-contiguous fill should use a device kernel.
            // For now, this is a known limitation — callers should make tensors
            // contiguous before calling fill_() on non-CPU devices.
            throw std::runtime_error(
                "fill_() on non-contiguous non-CPU tensors is not supported. "
                "Call .contiguous() first or use a contiguous tensor.");
        }
        auto ndims = this->ndim();
        auto shp = this->shape();
        auto str = this->strides();
        auto elem_size = tenzor::dtype_size(dtype());
        auto* base = static_cast<uint8_t*>(data_ptr());
        std::vector<int64_t> indices(ndims, 0);
        for (int64_t i = 0; i < numel(); ++i) {
            int64_t offset = 0;
            for (int64_t d = 0; d < ndims; ++d)
                offset += indices[d] * str[d] * elem_size;
            // Fill single element at base + offset
            switch (dtype()) {
                case DType::Float32: *reinterpret_cast<float*>(base + offset) = value; break;
                case DType::Float64: *reinterpret_cast<double*>(base + offset) = static_cast<double>(value); break;
                case DType::Int32: *reinterpret_cast<int32_t*>(base + offset) = static_cast<int32_t>(value); break;
                case DType::Int64: *reinterpret_cast<int64_t*>(base + offset) = static_cast<int64_t>(value); break;
                case DType::Int16: *reinterpret_cast<int16_t*>(base + offset) = static_cast<int16_t>(value); break;
                case DType::Int8: *reinterpret_cast<int8_t*>(base + offset) = static_cast<int8_t>(value); break;
                case DType::UInt8: *reinterpret_cast<uint8_t*>(base + offset) = static_cast<uint8_t>(value); break;
                case DType::UInt16: *reinterpret_cast<uint16_t*>(base + offset) = static_cast<uint16_t>(value); break;
                case DType::UInt32: *reinterpret_cast<uint32_t*>(base + offset) = static_cast<uint32_t>(value); break;
                case DType::UInt64: *reinterpret_cast<uint64_t*>(base + offset) = static_cast<uint64_t>(value); break;
                case DType::Float16: *reinterpret_cast<Float16*>(base + offset) = Float16(value); break;
                case DType::BFloat16: *reinterpret_cast<BFloat16*>(base + offset) = BFloat16(value); break;
                case DType::Bool: *reinterpret_cast<bool*>(base + offset) = (value != 0.0f); break;
                default: throw std::runtime_error("fill_ not supported for this dtype");
            }
            // Increment indices (row-major order)
            for (int64_t d = ndims - 1; d >= 0; --d) {
                if (++indices[d] < shp[d]) break;
                indices[d] = 0;
            }
        }
        return *this;
    }

    const int64_t n = numel();

    // For non-CPU devices, create a CPU tensor, fill it, then copy into existing storage
    if (device().type != Device::Type::CPU) {
        // Create CPU tensor with same shape and dtype
        std::vector<int64_t> shape_vec(shape().begin(), shape().end());
        Tensor cpu_tensor(shape_vec, dtype(), Device::cpu());
        cpu_tensor.fill_(value);  // Recursively fill on CPU

        // Copy filled data into this tensor's existing storage (preserves aliased views)
        auto* backend = backend_registry().get_backend(device().type);
        const size_t size_bytes = numel() * dtype_size();
        backend->copy(impl_->storage->data(), cpu_tensor.impl_->storage->data(),
                     size_bytes, CopyKind::HostToDevice);
        return *this;
    }

    // Direct implementation for CPU tensors - fill data with the value
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
        case DType::UInt16: {
            uint16_t* ptr = data<uint16_t>();
            for (int64_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<uint16_t>(value);
            }
            break;
        }
        case DType::UInt32: {
            uint32_t* ptr = data<uint32_t>();
            for (int64_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<uint32_t>(value);
            }
            break;
        }
        case DType::UInt64: {
            uint64_t* ptr = data<uint64_t>();
            for (int64_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<uint64_t>(value);
            }
            break;
        }
        case DType::Float16: {
            Float16* ptr = data<Float16>();
            Float16 f16_value(value);
            for (int64_t i = 0; i < n; ++i) {
                ptr[i] = f16_value;
            }
            break;
        }
        case DType::BFloat16: {
            BFloat16* ptr = data<BFloat16>();
            BFloat16 bf16_value(value);
            for (int64_t i = 0; i < n; ++i) {
                ptr[i] = bf16_value;
            }
            break;
        }
        case DType::Int8: {
            int8_t* ptr = data<int8_t>();
            for (int64_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<int8_t>(value);
            }
            break;
        }
        case DType::Int16: {
            int16_t* ptr = data<int16_t>();
            for (int64_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<int16_t>(value);
            }
            break;
        }
        case DType::Bool: {
            bool* ptr = data<bool>();
            bool bool_value = (value != 0.0f);
            for (int64_t i = 0; i < n; ++i) {
                ptr[i] = bool_value;
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
                throw std::runtime_error("Only one dimension can be inferred");
            }
            infer_dim = static_cast<int64_t>(i);
        } else if (new_shape[i] <= 0) {
            throw std::runtime_error("Invalid shape dimension");
        } else {
            total *= new_shape[i];
        }
    }

    if (infer_dim != -1) {
        if (total == 0) {
            throw std::runtime_error("Cannot infer dimension: product of known dimensions is zero");
        }
        if (numel() % total != 0) {
            throw std::runtime_error("Cannot infer dimension");
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
    return dispatch(OpId::Reshape, inputs, attrs)[0];
}

auto Tensor::view(std::vector<int64_t> new_shape) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("Cannot view null tensor");
    }

    if (!is_contiguous()) {
        throw std::runtime_error("View requires contiguous tensor. Use reshape() or contiguous() first.");
    }

    // Handle -1 inference (one dimension can be inferred)
    int64_t infer_dim = -1;
    int64_t total = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (infer_dim != -1) {
                throw std::runtime_error("Only one dimension can be inferred");
            }
            infer_dim = static_cast<int64_t>(i);
        } else if (new_shape[i] <= 0) {
            throw std::runtime_error("Invalid shape dimension");
        } else {
            total *= new_shape[i];
        }
    }

    if (infer_dim != -1) {
        if (total == 0) {
            throw std::runtime_error("Cannot infer dimension: product of known dimensions is zero");
        }
        if (numel() % total != 0) {
            throw std::runtime_error("Cannot infer dimension");
        }
        new_shape[infer_dim] = numel() / total;
        total = numel();
    }

    // Validate total elements match
    if (total != numel()) {
        throw std::runtime_error("View shape incompatible with number of elements");
    }

    // View is a zero-copy operation - create a new tensor that shares the same storage
    // This is the key difference from reshape: view MUST share storage
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*impl_);

    // Update shape and recompute strides for contiguous layout
    result.impl_->shape = std::move(new_shape);
    result.impl_->strides = compute_strides(result.impl_->shape);

    // CRITICAL: Share the same storage - this is what makes it a view
    // The storage is already shared via the copy constructor of TensorImpl
    // which copies the shared_ptr<Storage>

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
        throw std::out_of_range("Dimension out of range for transpose");
    }

    // For 2D tensors, default is transpose(0, 1)
    if (ndims == 2 && dim0 == 0 && dim1 == 1) {
        // Simple transpose - just swap shape and strides
        Tensor result;
        result.impl_ = std::make_shared<TensorImpl>(*impl_);
        std::swap(result.impl_->shape[0], result.impl_->shape[1]);
        std::swap(result.impl_->strides[0], result.impl_->strides[1]);
        result.impl_->is_contiguous_cache_.store(-1, std::memory_order_relaxed);
        return result;
    }

    // General case - swap specified dimensions
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*impl_);
    std::swap(result.impl_->shape[dim0], result.impl_->shape[dim1]);
    std::swap(result.impl_->strides[dim0], result.impl_->strides[dim1]);
    result.impl_->is_contiguous_cache_.store(-1, std::memory_order_relaxed);

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
            throw std::invalid_argument("Dimension out of range in permutation");
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
    result.impl_->is_contiguous_cache_.store(-1, std::memory_order_relaxed);

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
            throw std::out_of_range("Dimension out of range for squeeze");
        }

        // PyTorch behavior: if dimension is not singleton, return unchanged tensor
        if (impl_->shape[d] != 1) {
            return *this;
        }

        // Remove the dimension
        Tensor result;
        result.impl_ = std::make_shared<TensorImpl>(*impl_);
        result.impl_->shape.erase(result.impl_->shape.begin() + d);
        result.impl_->strides.erase(result.impl_->strides.begin() + d);
        result.impl_->is_contiguous_cache_.store(-1, std::memory_order_relaxed);

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
        result.impl_->is_contiguous_cache_.store(-1, std::memory_order_relaxed);

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
    result.impl_->is_contiguous_cache_.store(-1, std::memory_order_relaxed);

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
        throw std::runtime_error("start_dim out of range");
    }
    if (end_dim < 0 || end_dim >= ndims) {
        throw std::runtime_error("end_dim out of range");
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
    if (!impl_) {
        throw std::runtime_error("Cannot index null tensor");
    }

    const int64_t ndims = ndim();
    if (ndims == 0) {
        throw std::runtime_error("Cannot index 0D scalar tensor");
    }

    const int64_t dim0_size = impl_->shape[0];

    // Handle negative indices (e.g., tensor[-1] is the last element)
    int64_t normalized_idx = idx;
    if (normalized_idx < 0) {
        normalized_idx += dim0_size;
    }

    // Validate index is in bounds
    if (normalized_idx < 0 || normalized_idx >= dim0_size) {
        throw std::runtime_error("Index " + std::to_string(idx) +
                                " is out of bounds for dimension 0 with size " +
                                std::to_string(dim0_size));
    }

    // Use slice to get a single element along dimension 0
    // slice(dim, start, end, step) - end is exclusive
    Tensor result = slice(0, normalized_idx, normalized_idx + 1, 1);

    // For 1D tensors: squeeze to get a 0D scalar tensor
    // For multi-dimensional tensors: squeeze only dimension 0 to get a view
    result = result.squeeze(0);

    return result;
}

auto Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("Cannot slice null tensor");
    }

    const int64_t ndims = ndim();

    // Normalize dimension
    if (dim < 0) dim += ndims;
    if (dim < 0 || dim >= ndims) {
        throw std::runtime_error("Dimension out of range for slice");
    }

    const int64_t dim_size = impl_->shape[dim];

    // Normalize start and end indices
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;

    // Validate bounds (strict mode - no clamping)
    if (start < 0 || start > dim_size) {
        throw std::runtime_error("Slice start index " + std::to_string(start) +
            " out of bounds for dimension with size " + std::to_string(dim_size));
    }
    if (end < 0 || end > dim_size) {
        throw std::runtime_error("Slice end index " + std::to_string(end) +
            " out of bounds for dimension with size " + std::to_string(dim_size));
    }
    if (start > end) {
        throw std::runtime_error("Slice start index " + std::to_string(start) +
            " greater than end index " + std::to_string(end));
    }

    if (step <= 0) {
        throw std::runtime_error("Step must be positive");
    }

    // Calculate new dimension size
    int64_t new_dim_size = (end - start + step - 1) / step;  // Ceiling division

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

    // Validate that the slice doesn't exceed storage bounds
    int64_t max_offset = result.impl_->offset;
    // Calculate the furthest element this slice can access (with overflow checks)
    for (int64_t d = 0; d < static_cast<int64_t>(result.impl_->shape.size()); ++d) {
        if (result.impl_->shape[d] > 0) {
            int64_t extent = result.impl_->shape[d] - 1;
            int64_t stride = result.impl_->strides[d];
            // Check multiplication overflow
            if (stride != 0 && extent > std::numeric_limits<int64_t>::max() / std::abs(stride)) {
                throw std::overflow_error("Slice offset computation overflows int64_t");
            }
            int64_t delta = extent * stride;
            // Check addition overflow
            if (delta > 0 && max_offset > std::numeric_limits<int64_t>::max() - delta) {
                throw std::overflow_error("Slice offset computation overflows int64_t");
            }
            max_offset += delta;
        }
    }
    int64_t storage_elements = static_cast<int64_t>(result.impl_->storage->size_bytes() / tenzor::dtype_size(result.impl_->dtype));
    if (max_offset >= storage_elements) {
        throw std::out_of_range("Slice offset exceeds storage bounds");
    }

    result.impl_->is_contiguous_cache_.store(-1, std::memory_order_relaxed);

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
    // Account for offset when accessing sliced tensors
    auto* base_ptr = static_cast<uint8_t*>(impl_->storage->data());
    return base_ptr + impl_->offset * dtype_size();
}

auto Tensor::data_ptr() const -> const void* {
    if (!impl_ || !impl_->storage) {
        return nullptr;
    }
    // Account for offset when accessing sliced tensors
    const auto* base_ptr = static_cast<const uint8_t*>(impl_->storage->data());
    return base_ptr + impl_->offset * dtype_size();
}

auto Tensor::zeros_like(const Tensor& other) -> Tensor {
    if (!other.impl_) {
        return Tensor();
    }

    // Create zero tensor with same shape, dtype, and device
    std::vector<int64_t> shape_vec(other.shape().begin(), other.shape().end());
    return tenzor::zeros(shape_vec, other.dtype(), other.device());
}

// ============================================================================
// Memory Format Support
// ============================================================================

auto Tensor::memory_format() const noexcept -> MemoryFormat {
    if (!impl_) return MemoryFormat::Contiguous;

    // Only 4D tensors can be ChannelsLast
    if (impl_->shape.size() != 4) {
        return MemoryFormat::Contiguous;
    }

    // Check if strides match NHWC pattern
    auto nhwc_strides = compute_channels_last_strides(impl_->shape);
    if (impl_->strides == nhwc_strides && impl_->offset == 0) {
        return MemoryFormat::ChannelsLast;
    }

    // Check for 5D ChannelsLast3d
    if (impl_->shape.size() == 5) {
        auto ndhwc_strides = compute_channels_last_3d_strides(impl_->shape);
        if (impl_->strides == ndhwc_strides && impl_->offset == 0) {
            return MemoryFormat::ChannelsLast3d;
        }
    }

    return MemoryFormat::Contiguous;
}

auto Tensor::is_contiguous(MemoryFormat format) const noexcept -> bool {
    if (!impl_) return true;

    switch (format) {
        case MemoryFormat::Contiguous: {
            // Standard row-major contiguous check
            auto expected = compute_strides(impl_->shape);
            return impl_->strides == expected && impl_->offset == 0;
        }
        case MemoryFormat::ChannelsLast: {
            if (impl_->shape.size() != 4) return false;
            auto expected = compute_channels_last_strides(impl_->shape);
            return impl_->strides == expected && impl_->offset == 0;
        }
        case MemoryFormat::ChannelsLast3d: {
            if (impl_->shape.size() != 5) return false;
            auto expected = compute_channels_last_3d_strides(impl_->shape);
            return impl_->strides == expected && impl_->offset == 0;
        }
        case MemoryFormat::Preserve:
            // Preserve means "keep current format", so any contiguous layout counts
            return is_contiguous() || is_contiguous(MemoryFormat::ChannelsLast);
    }

    return false;
}

auto Tensor::to(MemoryFormat format) const -> Tensor {
    if (!impl_) return *this;

    // If already in the target format, return as-is
    if (is_contiguous(format)) {
        return *this;
    }

    // Handle Preserve format - no conversion needed
    if (format == MemoryFormat::Preserve) {
        return *this;
    }

    // For non-4D tensors, ChannelsLast doesn't apply
    if (format == MemoryFormat::ChannelsLast && impl_->shape.size() != 4) {
        return *this;
    }

    // For non-5D tensors, ChannelsLast3d doesn't apply
    if (format == MemoryFormat::ChannelsLast3d && impl_->shape.size() != 5) {
        return *this;
    }

    // Dispatch to backend for memory format conversion
    OpAttributes attrs;
    attrs["memory_format"] = std::to_string(static_cast<int>(format));

    std::vector<Tensor> inputs = {*this};
    return dispatch(OpId::ToMemoryFormat, inputs, attrs)[0];
}

} // namespace tenzor
