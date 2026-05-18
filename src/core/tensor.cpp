#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/core/jit_hooks.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/safe_math.hpp"
#include "tenzor/utils/memory_profiler.hpp"
#include "tenzor/core/memory_manager.hpp"
#include "tenzor/core/checked_math.hpp"
#include <numeric>
#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
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
    is_contiguous_cache_.store(1, std::memory_order_release);  // freshly constructed = contiguous

    // Compute strides
    strides = compute_strides(this->shape);

    // Allocate storage (checked multiply to prevent overflow)
    int64_t n = numel();
    size_t elem_sz = dtype_size(dtype);
    if (n > 0 && static_cast<size_t>(n) > std::numeric_limits<size_t>::max() / elem_sz) {
        throw std::overflow_error("Tensor allocation size overflow");
    }
    size_t size_bytes = static_cast<size_t>(n) * elem_sz;

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

    // Record allocation in global memory profiler
    if (size_bytes > 0) {
        MemoryProfiler::instance().on_allocate(size_bytes);
    }

    // Use DeviceStorage for ALL devices (including CPU)
    storage = make_intrusive<DeviceStorage>(ptr, size_bytes, device);

    // Zero-initialize if requested
    if (zero_init && size_bytes > 0) {
        if (device.type == Device::Type::CPU) {
            std::memset(ptr, 0, size_bytes);
        } else {
            backend->memset(ptr, 0, size_bytes, device.index);
        }
    }

}

auto TensorImpl::numel() const -> int64_t {
    int64_t result = 1;
    for (auto dim : shape) {
        // Check for overflow before multiplying
        if (dim != 0 && detail::safe_abs(result) > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / detail::safe_abs(dim)) {
            throw std::overflow_error("Tensor size overflow: shape produces more than INT64_MAX elements");
        }
        result *= dim;
    }
    return result;
}

auto TensorImpl::is_contiguous() const -> bool {
    // Return cached result if available (atomic load for thread safety)
    auto cached = is_contiguous_cache_.load(std::memory_order_acquire);
    if (cached >= 0) {
        return cached == 1;
    }
    // A tensor is contiguous if its strides match the expected row-major strides.
    // Offset does not affect contiguity — a slice can be contiguous at a non-zero offset.
    auto expected_strides = compute_strides(shape);
    bool result = (strides == expected_strides);
    is_contiguous_cache_.store(result ? 1 : 0, std::memory_order_release);
    return result;
}

// Tensor implementation
Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device device)
    : impl_(make_intrusive<TensorImpl>(std::move(shape), dtype, device, true)) {}

auto Tensor::empty_uninitialized(std::vector<int64_t> shape, DType dtype, Device device)
    -> Tensor {
    Tensor t;
    t.impl_ = make_intrusive<TensorImpl>(std::move(shape), dtype, device, false);
    return t;
}

// TensorImpl: construct from pre-existing storage (no allocation)
TensorImpl::TensorImpl(intrusive_ptr<Storage> storage_,
                       std::vector<int64_t> shape_,
                       std::vector<int64_t> strides_,
                       DType dtype_, Device device_)
    : storage(std::move(storage_))
    , shape(std::move(shape_))
    , strides(std::move(strides_))
    , dtype(dtype_)
    , device(device_) {
    is_contiguous_cache_.store(-1, std::memory_order_release);  // lazily computed
}

auto Tensor::from_blob(void* data,
                       std::vector<int64_t> shape,
                       DType dtype,
                       Device device,
                       std::function<void(void*)> deleter) -> Tensor {
    // Validate dims and compute element count with overflow checking
    // (mirrors TensorImpl::numel() at lines 79-89 and the size_bytes guard
    // in the allocating ctor at lines 40-46).
    auto strides = compute_strides(shape);
    int64_t n = 1;
    for (auto dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("from_blob: shape dimensions must be non-negative");
        }
        if (dim != 0 &&
            detail::safe_abs(n) > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                                      detail::safe_abs(dim)) {
            throw std::overflow_error(
                "from_blob: shape produces more than INT64_MAX elements");
        }
        n *= dim;
    }

    if (n > 0 && data == nullptr) {
        throw std::runtime_error("from_blob: data must not be null for non-empty tensor");
    }

    size_t elem_sz = tenzor::dtype_size(dtype);
    if (n > 0 && static_cast<size_t>(n) > std::numeric_limits<size_t>::max() / elem_sz) {
        throw std::overflow_error("from_blob: byte size overflow");
    }
    size_t size_bytes = static_cast<size_t>(n) * elem_sz;

    // Create ExternalStorage (does NOT take ownership unless deleter is provided)
    auto storage = make_intrusive<ExternalStorage>(
        data, size_bytes, device, std::move(deleter));

    // Build tensor from pre-existing storage
    Tensor t;
    t.impl_ = make_intrusive<TensorImpl>(
        std::move(storage), std::move(shape), std::move(strides),
        dtype, device);
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
    // An undefined Tensor has zero elements. This matches PyTorch's
    // `Tensor::numel()` semantics and lets callers use `t.numel() == 0`
    // as a presence check on default-constructed Tensors (e.g. the
    // static empty returned by Node::get_tensor_attr() when the
    // attribute is absent).
    if (!impl_) return 0;
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

auto Tensor::version() const noexcept -> uint64_t {
    if (!impl_) return 0;
    return impl_->version_counter_.load(std::memory_order_acquire);
}

auto Tensor::bump_version() -> void {
    if (impl_) {
        impl_->version_counter_.fetch_add(1, std::memory_order_release);
    }
}

auto Tensor::is_view() const noexcept -> bool {
    if (!impl_) return false;
    return impl_->view_base_ != nullptr;
}

auto Tensor::_view_base() const noexcept -> TensorImpl* {
    if (!impl_) return nullptr;
    return impl_->view_base_.get();
}

auto Tensor::_set_view_base(intrusive_ptr<TensorImpl> base) noexcept -> void {
    if (impl_) {
        impl_->view_base_ = std::move(base);
    }
}

auto Tensor::is_quantized() const noexcept -> bool {
    if (!impl_) return false;
    return tenzor::is_quantized(impl_->dtype);
}

auto Tensor::q_scale() const -> double {
    if (!is_quantized()) {
        throw std::runtime_error("q_scale() called on non-quantized tensor");
    }
    return impl_->q_scale_;
}

auto Tensor::q_zero_point() const -> int64_t {
    if (!is_quantized()) {
        throw std::runtime_error("q_zero_point() called on non-quantized tensor");
    }
    return impl_->q_zero_point_;
}

auto Tensor::int_repr() const -> Tensor {
    if (!is_quantized()) {
        throw std::runtime_error("int_repr() called on non-quantized tensor");
    }
    DType int_dtype;
    if (impl_->dtype == DType::QUInt8) {
        int_dtype = DType::UInt8;
    } else if (impl_->dtype == DType::QInt8) {
        int_dtype = DType::Int8;
    } else {
        // QInt4x2 packs two int4 values per byte; the packed layout is not a
        // drop-in Int8 view. Dequantize first for that case.
        throw std::runtime_error(
            "int_repr(): unsupported quantized dtype — QInt4x2 packed layout "
            "is not exposed via int_repr; use dequantize() then requantize if needed");
    }
    // Zero-copy view: share Storage, reinterpret the dtype. QInt8/QUInt8 and
    // Int8/UInt8 share the same 1-byte element size, so element-unit strides
    // and offset carry over unchanged. view_base_ points at the quantized
    // parent so in-place mutations on either side are observed by autograd's
    // saved-tensor version check.
    Tensor result;
    result.impl_ = make_intrusive<TensorImpl>(
        impl_->storage,
        std::vector<int64_t>(impl_->shape),
        std::vector<int64_t>(impl_->strides),
        int_dtype,
        impl_->device);
    result.impl_->offset = impl_->offset;
    result.impl_->view_base_ = impl_->view_base_ ? impl_->view_base_ : impl_;
    return result;
}

auto Tensor::dequantize() const -> Tensor {
    if (!is_quantized()) {
        throw std::runtime_error("dequantize() called on non-quantized tensor");
    }
    // Dequantize: float_val = (int_val - zero_point) * scale
    auto int_tensor = int_repr().to(Device::cpu());
    auto float_tensor = zeros(std::vector<int64_t>(impl_->shape), DType::Float32, Device::cpu());
    size_t n = numel();
    float* out = float_tensor.data<float>();
    double scale = impl_->q_scale_;
    int64_t zp = impl_->q_zero_point_;

    if (impl_->dtype == DType::QUInt8) {
        const uint8_t* in = int_tensor.data<uint8_t>();
        for (size_t i = 0; i < n; ++i) {
            out[i] = static_cast<float>((static_cast<int64_t>(in[i]) - zp) * scale);
        }
    } else {
        const int8_t* in = int_tensor.data<int8_t>();
        for (size_t i = 0; i < n; ++i) {
            out[i] = static_cast<float>((static_cast<int64_t>(in[i]) - zp) * scale);
        }
    }

    return float_tensor.to(impl_->device);
}

auto Tensor::set_quantization_params(double scale, int64_t zero_point) -> void {
    if (!impl_ || !tenzor::is_quantized(impl_->dtype)) {
        throw std::runtime_error("set_quantization_params: tensor is not quantized");
    }
    impl_->q_scale_ = scale;
    impl_->q_zero_point_ = zero_point;
}

auto Tensor::is_per_channel_quantized() const noexcept -> bool {
    return impl_ && tenzor::is_quantized(impl_->dtype) && impl_->q_scales_.has_value();
}

auto Tensor::q_per_channel_scales() const -> const std::vector<double>& {
    if (!impl_ || !impl_->q_scales_.has_value()) {
        throw std::runtime_error("q_per_channel_scales: tensor is not per-channel quantized");
    }
    return impl_->q_scales_.value();
}

auto Tensor::q_per_channel_zero_points() const -> const std::vector<int64_t>& {
    if (!impl_ || !impl_->q_zero_points_.has_value()) {
        throw std::runtime_error("q_per_channel_zero_points: tensor is not per-channel quantized");
    }
    return impl_->q_zero_points_.value();
}

auto Tensor::q_per_channel_axis() const -> int64_t {
    if (!impl_ || !impl_->q_scales_.has_value()) {
        throw std::runtime_error("q_per_channel_axis: tensor is not per-channel quantized");
    }
    return impl_->q_axis_;
}

auto Tensor::set_per_channel_quantization_params(
    std::vector<double> scales,
    std::vector<int64_t> zero_points,
    int64_t axis) -> void {
    if (!impl_ || !tenzor::is_quantized(impl_->dtype)) {
        throw std::runtime_error("set_per_channel_quantization_params: tensor is not quantized");
    }
    if (scales.size() != zero_points.size()) {
        throw std::runtime_error("set_per_channel_quantization_params: scales and zero_points must have same size");
    }
    if (axis < 0 || axis >= static_cast<int64_t>(impl_->shape.size())) {
        throw std::runtime_error("set_per_channel_quantization_params: axis out of range");
    }
    if (static_cast<int64_t>(scales.size()) != impl_->shape[axis]) {
        throw std::runtime_error("set_per_channel_quantization_params: scales size must match shape[axis]");
    }
    impl_->q_scales_ = std::move(scales);
    impl_->q_zero_points_ = std::move(zero_points);
    impl_->q_axis_ = axis;
}

auto Tensor::is_contiguous() const noexcept -> bool {
    if (!impl_) return true;
    return impl_->is_contiguous();
}

auto Tensor::offset() const -> int64_t {
    if (!impl_) return 0;
    return impl_->offset;
}

auto Tensor::storage() const -> const intrusive_ptr<Storage>& {
    if (!impl_) throw std::runtime_error("Operation on uninitialized tensor");
    return impl_->storage;
}

auto Tensor::is_pinned() const -> bool {
    if (!impl_ || !impl_->storage) return false;
    return impl_->storage->is_pinned();
}

auto Tensor::shape_info_snapshot() const -> std::shared_ptr<const ShapeInfo> {
    if (!impl_) return nullptr;
    // Fast path: return the cached snapshot if a previous call built one and
    // no mutator has invalidated it since. The C++20 std::atomic<shared_ptr>
    // class template does atomic refcount manipulation on the control block;
    // it replaces the libstdc++ atomic_load/store free functions that used a
    // hashed spinlock pool and caused a UAF under pybind11 copy patterns.
    if (auto cached = impl_->shape_info_cache_.load(std::memory_order_acquire)) {
        return cached;
    }
    auto fresh = std::make_shared<const ShapeInfo>(ShapeInfo{
        /* shape   */ impl_->shape,
        /* strides */ impl_->strides,
        /* offset  */ impl_->offset,
    });
    // CAS install: a concurrent racer may have installed a snapshot between
    // our load and this store. If so, prefer theirs so every reader that
    // observes a non-null cache sees the same pointer — the pointer-identity
    // guarantee the test relies on. A mutator that races in between only
    // causes the next reader to rebuild.
    std::shared_ptr<const ShapeInfo> expected{nullptr};
    if (impl_->shape_info_cache_.compare_exchange_strong(
            expected, fresh,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return fresh;
    }
    return expected;
}

auto Tensor::pin_memory() -> Tensor& {
    if (!impl_ || !impl_->storage) return *this;
    // Only CPU tensors can be pinned; Storage::pin() handles the
    // device check internally and reports failure via its bool return.
    // We don't need to branch on the result — on non-CUDA builds it
    // silently no-ops and is_pinned() will continue to return false.
    impl_->storage->pin();
    return *this;
}

auto may_alias(const Tensor& a, const Tensor& b) -> bool {
    // Uninitialized tensors never alias.
    if (!a.impl() || !b.impl()) return false;

    // Same Tensor object (same impl): treat as non-aliasing. Most in-place
    // kernels handle x.op_(x) correctly; this avoids false positives on the
    // common idiom while still catching view/slice aliasing.
    if (a.impl().get() == b.impl().get()) return false;

    // Different storage: definitely no alias.
    const auto& sa = a.storage();
    const auto& sb = b.storage();
    if (!sa || !sb) return false;
    if (sa.get() != sb.get()) return false;

    // Same storage: compute byte spans. This overestimates touched bytes for
    // strided views (safe — the alternative would be to walk the stride
    // pattern, which is both expensive and unnecessary here).
    const auto esize_a = static_cast<int64_t>(a.dtype_size());
    const auto esize_b = static_cast<int64_t>(b.dtype_size());
    const auto start_a = a.offset() * esize_a;
    const auto start_b = b.offset() * esize_b;
    const auto end_a   = start_a + a.numel() * esize_a;
    const auto end_b   = start_b + b.numel() * esize_b;
    return start_a < end_b && start_b < end_a;
}

auto Tensor::set_requires_grad(bool requires_grad) -> void {
    if (!impl_) throw std::runtime_error("Operation on uninitialized tensor");
    impl_->requires_grad = requires_grad;
}

// NOTE: mutable_shape(), mutable_strides(), and set_offset() are internal
// mutators used by backend kernels constructing view-like result tensors.
// Callers must hold external synchronization; these are not thread-safe.
//
// Each mutator invalidates BOTH the contiguity cache and the memory-format
// cache (previously only the contiguity cache), and bumps the version counter
// so that autograd's in-place detection can catch a saved tensor whose
// metadata was changed out from under it.

auto Tensor::mutable_shape() -> std::vector<int64_t>& {
    if (!impl_) throw std::runtime_error("Operation on uninitialized tensor");
    impl_->is_contiguous_cache_.store(-1, std::memory_order_release);
    impl_->memory_format_cache_.store(-1, std::memory_order_release);
    impl_->version_counter_.fetch_add(1, std::memory_order_release);
    impl_->shape_info_cache_.store(nullptr, std::memory_order_release);
    return impl_->shape;
}

auto Tensor::mutable_strides() -> std::vector<int64_t>& {
    if (!impl_) throw std::runtime_error("Operation on uninitialized tensor");
    impl_->is_contiguous_cache_.store(-1, std::memory_order_release);
    impl_->memory_format_cache_.store(-1, std::memory_order_release);
    impl_->version_counter_.fetch_add(1, std::memory_order_release);
    impl_->shape_info_cache_.store(nullptr, std::memory_order_release);
    return impl_->strides;
}

auto Tensor::set_offset(int64_t offset) -> void {
    if (!impl_) throw std::runtime_error("Operation on uninitialized tensor");
    impl_->offset = offset;
    impl_->is_contiguous_cache_.store(-1, std::memory_order_release);
    impl_->memory_format_cache_.store(-1, std::memory_order_release);
    impl_->version_counter_.fetch_add(1, std::memory_order_release);
    impl_->shape_info_cache_.store(nullptr, std::memory_order_release);
}

auto Tensor::invalidate_contiguity_cache() -> void {
    if (impl_) {
        impl_->is_contiguous_cache_.store(-1, std::memory_order_release);
        impl_->memory_format_cache_.store(-1, std::memory_order_release);
    }
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
    // Bounds check: compute min/max reachable element using strides.
    // Negative strides (from flip/slice with step<0) shift the reachable range
    // downward from the base offset, so we track both endpoints.
    // Skip for empty tensors (any dimension is 0) — no elements to access.
    if (impl_->numel() > 0) {
        size_t storage_elements = impl_->storage->size_bytes() / tenzor::dtype_size(impl_->dtype);
        int64_t min_offset = static_cast<int64_t>(impl_->offset);
        int64_t max_offset = static_cast<int64_t>(impl_->offset);
        for (int64_t d = 0; d < ndim(); ++d) {
            if (impl_->shape[d] > 0) {
                int64_t extent = checked_mul(impl_->shape[d] - 1, impl_->strides[d]);
                if (extent >= 0) {
                    max_offset += extent;
                } else {
                    min_offset += extent;
                }
            }
        }
        if (min_offset < 0 || static_cast<size_t>(max_offset) >= storage_elements) {
            throw std::out_of_range("Tensor data access: reachable offset exceeds storage bounds");
        }
    }
    // For byte-level access types, scale offset by dtype size so pointer arithmetic
    // advances by the correct number of bytes (offset is in elements, not bytes)
    if constexpr (std::is_same_v<CleanT, uint8_t> || std::is_same_v<CleanT, char> ||
                  std::is_same_v<CleanT, unsigned char> || std::is_same_v<CleanT, signed char>) {
        auto* base = static_cast<uint8_t*>(impl_->storage->data());
        return reinterpret_cast<T*>(base + checked_mul(static_cast<int64_t>(impl_->offset),
                                                       static_cast<int64_t>(tenzor::dtype_size(impl_->dtype))));
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
    // Bounds check: compute min/max reachable element using strides.
    // Negative strides (from flip/slice with step<0) shift the reachable range
    // downward from the base offset, so we track both endpoints.
    // Skip for empty tensors (any dimension is 0) — no elements to access.
    if (impl_->numel() > 0) {
        size_t storage_elements = impl_->storage->size_bytes() / tenzor::dtype_size(impl_->dtype);
        int64_t min_offset = static_cast<int64_t>(impl_->offset);
        int64_t max_offset = static_cast<int64_t>(impl_->offset);
        for (int64_t d = 0; d < ndim(); ++d) {
            if (impl_->shape[d] > 0) {
                int64_t extent = checked_mul(impl_->shape[d] - 1, impl_->strides[d]);
                if (extent >= 0) {
                    max_offset += extent;
                } else {
                    min_offset += extent;
                }
            }
        }
        if (min_offset < 0 || static_cast<size_t>(max_offset) >= storage_elements) {
            throw std::out_of_range("Tensor data access: reachable offset exceeds storage bounds");
        }
    }
    // For byte-level access types, scale offset by dtype size so pointer arithmetic
    // advances by the correct number of bytes (offset is in elements, not bytes)
    if constexpr (std::is_same_v<CleanT, uint8_t> || std::is_same_v<CleanT, char> ||
                  std::is_same_v<CleanT, unsigned char> || std::is_same_v<CleanT, signed char>) {
        const auto* base = static_cast<const uint8_t*>(impl_->storage->data());
        return reinterpret_cast<const T*>(base + checked_mul(static_cast<int64_t>(impl_->offset),
                                                              static_cast<int64_t>(tenzor::dtype_size(impl_->dtype))));
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
    // If a JIT trace is active, .item() is a graph break: the scalar
    // gets baked into the trace as a constant, and any downstream
    // Python `if` / `while` on this value silently freezes the taken
    // branch. Notify the tracer so it can warn (default) or throw
    // (TENZOR_JIT_STRICT=1). Free when no trace is active.
    tenzor::detail::notify_graph_break("scalar extraction (.item()) on a traced tensor");
    if (device().type != Device::Type::CPU) {
        // Single-element transfer: copy just sizeof(T) bytes instead of entire tensor
        T value;
        auto* backend = backend_registry().get_backend(device().type);
        backend->copy(&value, data_ptr(), sizeof(T), CopyKind::DeviceToHost);
        backend->synchronize(device().index);
        return value;
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

    // If already on the target device and contiguous, return *this (no copy).
    // This is a standard optimization (matches PyTorch behavior). The returned
    // tensor shares storage with the original. Use .clone() if a copy is needed.
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

    // GPU views with non-zero offset: copy to own storage via backend dispatch
    // so the transfer path operates on a simple zero-offset buffer.
    if (impl_->offset != 0 && impl_->device.type != Device::Type::CPU) {
        Tensor owned = contiguous();
        if (owned.data_ptr() != data_ptr()) {
            // contiguous() made a copy — recurse with the clean tensor
            return owned.to(device);
        }
        // If contiguous() returned *this (already contiguous but with offset),
        // fall through to the copy path which handles offset correctly.
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

auto Tensor::to(Device device, DType dtype) const -> Tensor {
    if (!impl_) return *this;

    bool same_device = (impl_->device == device);
    bool same_dtype = (impl_->dtype == dtype);

    if (same_device && same_dtype && is_contiguous()) return *this;

    // Transfer to device first (preserving current dtype), then cast on-device.
    // This avoids: (1) casting on source device then transferring, or
    // (2) transferring then casting on CPU if target is GPU.
    Tensor on_device = same_device ? *this : to(device);
    return same_dtype ? on_device : on_device.to(dtype);
}

auto Tensor::to(DType dtype) const -> Tensor {
    if (!impl_) {
        return *this;
    }

    // If already the target dtype, return as-is
    if (impl_->dtype == dtype) {
        return *this;
    }

    // Casting to a quantized dtype is not meaningful: quantized tensors carry
    // scale/zero_point metadata that plain .to() cannot supply. Force the
    // caller to use the explicit quantize API instead of silently returning a
    // zero-initialized tensor (which is what the old fall-through did).
    if (tenzor::is_quantized(dtype)) {
        throw std::runtime_error(
            ".to(quantized dtype) is not supported — use tenzor::quantize_per_tensor("
            "x, scale, zero_point, dtype) or quantize_per_channel() to build a quantized tensor");
    }
    if (tenzor::is_quantized(impl_->dtype)) {
        throw std::runtime_error(
            ".to(dtype) on a quantized tensor is not supported — call dequantize() first, "
            "then cast the resulting float tensor");
    }

    // GPU-side Cast kernel: dispatch directly on-device when registered.
    if (impl_->device.type != Device::Type::CPU) {
        if (!is_op_supported(OpId::Cast, impl_->device.type)) {
            // Silently routing a GPU dtype-cast through the CPU backend (the
            // historical fallback) violates the project's no-CPU-fallback
            // policy: it makes a multi-GB H2D + serial scalar conversion +
            // D2H round-trip look like a free op. Force the user to either
            // register a native Cast kernel for this device, or move the
            // tensor to CPU explicitly with .cpu() before casting.
            std::string src_dtype_str{tenzor::dtype_name(impl_->dtype)};
            std::string dst_dtype_str{tenzor::dtype_name(dtype)};
            std::string device_str = impl_->device.to_string();
            throw std::runtime_error(
                "Tensor::to(DType): no native Cast kernel registered for device " +
                device_str + " (cast " + src_dtype_str + " -> " + dst_dtype_str +
                "). Either register a Cast kernel for this backend or call "
                ".cpu() explicitly before casting.");
        }
        Tensor src = is_contiguous() ? *this : contiguous();
        OpAttributes attrs;
        attrs.set(AttrKey::TargetDtype, static_cast<int64_t>(static_cast<uint8_t>(dtype)));
        Tensor result = dispatch_single(OpId::Cast, std::span<const Tensor>(&src, 1), attrs);
        result.impl_->requires_grad = impl_->requires_grad;
        return result;
    }

    // CPU dtype conversion (input is already on CPU; this is the genuine path).
    Tensor cpu_tensor = *this;

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
            } else if constexpr (std::is_same_v<SrcT, FP8_E4M3> || std::is_same_v<SrcT, FP8_E5M2>) {
                // Convert FP8 to float, then to target type
                float intermediate = static_cast<float>(src_ptr[i]);
                if constexpr (std::is_same_v<DstT, FP8_E4M3> || std::is_same_v<DstT, FP8_E5M2>) {
                    dst_ptr[i] = DstT(intermediate);
                } else if constexpr (std::is_same_v<DstT, Float16> || std::is_same_v<DstT, BFloat16>) {
                    dst_ptr[i] = DstT(intermediate);
                } else if constexpr (std::is_same_v<DstT, std::complex<float>>) {
                    dst_ptr[i] = std::complex<float>(intermediate, 0.0f);
                } else if constexpr (std::is_same_v<DstT, std::complex<double>>) {
                    dst_ptr[i] = std::complex<double>(static_cast<double>(intermediate), 0.0);
                } else {
                    dst_ptr[i] = static_cast<DstT>(intermediate);
                }
            } else if constexpr (std::is_same_v<DstT, FP8_E4M3> || std::is_same_v<DstT, FP8_E5M2>) {
                // Convert source to float, then to FP8
                float intermediate;
                if constexpr (std::is_same_v<SrcT, std::complex<float>>) {
                    intermediate = src_ptr[i].real();
                } else if constexpr (std::is_same_v<SrcT, std::complex<double>>) {
                    intermediate = static_cast<float>(src_ptr[i].real());
                } else {
                    intermediate = static_cast<float>(src_ptr[i]);
                }
                dst_ptr[i] = DstT(intermediate);
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
                case DType::FP8_E4M3: convert_elements.template operator()<SrcT, FP8_E4M3>(); break; \
                case DType::FP8_E5M2: convert_elements.template operator()<SrcT, FP8_E5M2>(); break; \
                default: break; \
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
        DISPATCH_SRC_DTYPE(DType::FP8_E4M3, FP8_E4M3)
        DISPATCH_SRC_DTYPE(DType::FP8_E5M2, FP8_E5M2)
        // Quantized source/dest pairs are rejected at the top of Tensor::to(DType)
        // so no case for QInt8/QUInt8/QInt4x2 is reachable here.
        default:
            break;
    }

    #undef DISPATCH_SRC_DTYPE

    // The GPU branch above either dispatched or threw, so this CPU cast path
    // produces a CPU result whose device already matches *this.
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

    // Allocate uninitialized — memcpy overwrites all bytes immediately
    Tensor result = Tensor::empty_uninitialized(impl_->shape, impl_->dtype, impl_->device);
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
    if (!impl_) {
        throw std::runtime_error("Cannot detach an uninitialized tensor");
    }
    // Share storage (zero-copy) like view() — no need to copy data
    Tensor result;
    result.impl_ = make_intrusive<TensorImpl>(*impl_);
    result.impl_->requires_grad = false;
    result.impl_->view_base_ = impl_->view_base_ ? impl_->view_base_ : impl_;
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
    std::array<Tensor, 1> inputs = {*this};
    return dispatch(OpId::Contiguous, inputs)[0];
}

auto Tensor::new_zeros(std::vector<int64_t> shape) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("new_zeros called on uninitialized tensor");
    }
    return tenzor::zeros(std::move(shape), impl_->dtype, impl_->device);
}

auto Tensor::new_ones(std::vector<int64_t> shape) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("new_ones called on uninitialized tensor");
    }
    return tenzor::ones(std::move(shape), impl_->dtype, impl_->device);
}

auto Tensor::new_empty(std::vector<int64_t> shape) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("new_empty called on uninitialized tensor");
    }
    return Tensor::empty_uninitialized(std::move(shape), impl_->dtype, impl_->device);
}

auto Tensor::new_full(std::vector<int64_t> shape, double fill_value) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("new_full called on uninitialized tensor");
    }
    return tenzor::full(std::move(shape), fill_value, impl_->dtype, impl_->device);
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

// Reverse scalar operators (scalar op tensor)
auto operator+(double s, const Tensor& t) -> Tensor {
    return t + s;  // addition is commutative
}

auto operator-(double s, const Tensor& t) -> Tensor {
    return tenzor::neg(t) + s;
}

auto operator*(double s, const Tensor& t) -> Tensor {
    return t * s;  // multiplication is commutative
}

auto operator/(double s, const Tensor& t) -> Tensor {
    auto numerator = tenzor::full(
        std::vector<int64_t>(t.shape().begin(), t.shape().end()),
        s, t.dtype(), t.device());
    return tenzor::div(numerator, t);
}

// In-place operations — dispatch through in-place kernels so views/aliases see updates
// If the other operand shares storage with this tensor (e.g. a += a),
// clone it first to avoid undefined behavior from reading and writing
// the same buffer simultaneously.
auto Tensor::operator+=(const Tensor& other) -> Tensor& {
    if (!impl_) throw std::runtime_error("Cannot perform in-place add on uninitialized tensor");
    auto& table = DispatchTableRegistry::get_table(impl_->device.type);
    bool aliased = other.impl_ && impl_->storage && other.impl_->storage &&
                   impl_->storage.get() == other.impl_->storage.get();
    std::array<Tensor, 1> others = {aliased ? other.clone() : other};
    table.dispatch_inplace(OpId::AddInplace, *this, others);
    return *this;
}

auto Tensor::operator-=(const Tensor& other) -> Tensor& {
    if (!impl_) throw std::runtime_error("Cannot perform in-place sub on uninitialized tensor");
    auto& table = DispatchTableRegistry::get_table(impl_->device.type);
    bool aliased = other.impl_ && impl_->storage && other.impl_->storage &&
                   impl_->storage.get() == other.impl_->storage.get();
    std::array<Tensor, 1> others = {aliased ? other.clone() : other};
    table.dispatch_inplace(OpId::SubInplace, *this, others);
    return *this;
}

auto Tensor::operator*=(const Tensor& other) -> Tensor& {
    if (!impl_) throw std::runtime_error("Cannot perform in-place mul on uninitialized tensor");
    auto& table = DispatchTableRegistry::get_table(impl_->device.type);
    bool aliased = other.impl_ && impl_->storage && other.impl_->storage &&
                   impl_->storage.get() == other.impl_->storage.get();
    std::array<Tensor, 1> others = {aliased ? other.clone() : other};
    table.dispatch_inplace(OpId::MulInplace, *this, others);
    return *this;
}

auto Tensor::operator/=(const Tensor& other) -> Tensor& {
    if (!impl_) throw std::runtime_error("Cannot perform in-place div on uninitialized tensor");
    auto& table = DispatchTableRegistry::get_table(impl_->device.type);
    bool aliased = other.impl_ && impl_->storage && other.impl_->storage &&
                   impl_->storage.get() == other.impl_->storage.get();
    std::array<Tensor, 1> others = {aliased ? other.clone() : other};
    table.dispatch_inplace(OpId::DivInplace, *this, others);
    return *this;
}

auto Tensor::fill_(double value) -> Tensor& {
    if (!impl_) {
        return *this;
    }

    // Non-contiguous tensors: iterate using strides to fill each element in-place
    if (!is_contiguous()) {
        if (device().type != Device::Type::CPU) {
            // Dispatch to backend's StridedFill kernel — avoids expensive GPU→CPU→GPU round-trip
            auto& table = DispatchTableRegistry::get_table(impl_->device.type);
            if (table.has_inplace_kernel(OpId::StridedFill)) {
                OpAttributes attrs;
                attrs.set(AttrKey::Value, value);
                table.dispatch_inplace(OpId::StridedFill, *this, {}, attrs);
                return *this;
            }
            // Fallback for backends without StridedFill: contiguous copy + fill
            auto contig = contiguous();
            contig.fill_(value);
            auto shp = shape();
            auto result = contig.reshape({shp.begin(), shp.end()});
            impl_ = result.impl_;
            return *this;
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
                offset = checked_add(offset, checked_mul(checked_mul(indices[d], str[d]),
                                                         static_cast<int64_t>(elem_size)));
            // Fill single element at base + offset
            switch (dtype()) {
                case DType::Float32: *reinterpret_cast<float*>(base + offset) = static_cast<float>(value); break;
                case DType::Float64: *reinterpret_cast<double*>(base + offset) = value; break;
                case DType::Int32: *reinterpret_cast<int32_t*>(base + offset) = checked_narrow<int32_t>(value); break;
                case DType::Int64: *reinterpret_cast<int64_t*>(base + offset) = checked_narrow<int64_t>(value); break;
                case DType::Int16: *reinterpret_cast<int16_t*>(base + offset) = checked_narrow<int16_t>(value); break;
                case DType::Int8: *reinterpret_cast<int8_t*>(base + offset) = checked_narrow<int8_t>(value); break;
                case DType::UInt8: *reinterpret_cast<uint8_t*>(base + offset) = checked_narrow<uint8_t>(value); break;
                case DType::UInt16: *reinterpret_cast<uint16_t*>(base + offset) = checked_narrow<uint16_t>(value); break;
                case DType::UInt32: *reinterpret_cast<uint32_t*>(base + offset) = checked_narrow<uint32_t>(value); break;
                case DType::UInt64: *reinterpret_cast<uint64_t*>(base + offset) = checked_narrow<uint64_t>(value); break;
                case DType::Float16: *reinterpret_cast<Float16*>(base + offset) = Float16(static_cast<float>(value)); break;
                case DType::BFloat16: *reinterpret_cast<BFloat16*>(base + offset) = BFloat16(static_cast<float>(value)); break;
                case DType::Bool: *reinterpret_cast<bool*>(base + offset) = (value != 0.0f); break;
                case DType::Complex64: *reinterpret_cast<std::complex<float>*>(base + offset) = std::complex<float>(value, 0.0f); break;
                case DType::Complex128: *reinterpret_cast<std::complex<double>*>(base + offset) = std::complex<double>(static_cast<double>(value), 0.0); break;
                case DType::FP8_E4M3: *reinterpret_cast<FP8_E4M3*>(base + offset) = FP8_E4M3(static_cast<float>(value)); break;
                case DType::FP8_E5M2: *reinterpret_cast<FP8_E5M2*>(base + offset) = FP8_E5M2(static_cast<float>(value)); break;
                case DType::QInt8:
                    if (q_scale() == 0.0) {
                        throw std::runtime_error(
                            "fill_ on quantized tensor requires quantization params: "
                            "call set_quantization_params(scale, zero_point) first");
                    } else {
                        const int64_t qval = static_cast<int64_t>(std::round(value / q_scale())) + q_zero_point();
                        *reinterpret_cast<int8_t*>(base + offset) = static_cast<int8_t>(
                            std::clamp(qval, static_cast<int64_t>(-128), static_cast<int64_t>(127)));
                    }
                    break;
                case DType::QUInt8:
                    if (q_scale() == 0.0) {
                        throw std::runtime_error(
                            "fill_ on quantized tensor requires quantization params: "
                            "call set_quantization_params(scale, zero_point) first");
                    } else {
                        const int64_t qval = static_cast<int64_t>(std::round(value / q_scale())) + q_zero_point();
                        *reinterpret_cast<uint8_t*>(base + offset) = static_cast<uint8_t>(
                            std::clamp(qval, static_cast<int64_t>(0), static_cast<int64_t>(255)));
                    }
                    break;
                case DType::QInt4x2:
                    if (q_scale() == 0.0) {
                        throw std::runtime_error(
                            "fill_ on quantized tensor requires quantization params: "
                            "call set_quantization_params(scale, zero_point) first");
                    } else {
                        // Pack two 4-bit signed values per byte; for a scalar element
                        // both nibbles are the same clamped qval.
                        const int64_t qval = static_cast<int64_t>(std::round(value / q_scale())) + q_zero_point();
                        const int64_t clamped = std::clamp(qval, static_cast<int64_t>(-8), static_cast<int64_t>(7));
                        *reinterpret_cast<uint8_t*>(base + offset) =
                            static_cast<uint8_t>((clamped & 0xF) | ((clamped & 0xF) << 4));
                    }
                    break;
                default: throw std::runtime_error("fill_ not supported for this dtype");
            }
            // Increment indices (row-major order)
            for (int64_t d = ndims - 1; d >= 0; --d) {
                if (++indices[d] < shp[d]) break;
                indices[d] = 0;
            }
        }
        bump_version();
        return *this;
    }

    const int64_t n = numel();

    // For non-CPU devices, use backend memset for zero-fill, or small host buffer for non-zero
    if (device().type != Device::Type::CPU) {
        auto* backend = backend_registry().get_backend(device().type);
        const size_t size_bytes = static_cast<size_t>(n) * dtype_size();
        if (value == 0.0f && (!is_quantized() || q_zero_point() == 0)) {
            // Fast path: memset to zero directly on device.
            // Guard: quantized tensors with non-zero zero_point encode 0.0 as
            // q_zero_point bytes, not 0x00, so we must NOT memset in that case.
            backend->memset(data_ptr(), 0, size_bytes, device().index);
        } else {
            // Use in-place StridedFill to preserve tensor identity (version tracking, views)
            auto& table = DispatchTableRegistry::get_table(impl_->device.type);
            if (table.has_inplace_kernel(OpId::StridedFill)) {
                OpAttributes attrs;
                attrs.set(AttrKey::Value, value);
                table.dispatch_inplace(OpId::StridedFill, *this, {}, attrs);
            } else {
                // Fallback: dispatch Fill and replace impl
                OpAttributes attrs;
                attrs.set(AttrKey::Value, static_cast<double>(value));
                std::array<Tensor, 1> inputs = {*this};
                auto result = table.dispatch(OpId::Fill, inputs, attrs);
                if (!result.empty()) {
                    impl_ = result[0].impl_;
                }
            }
        }
        bump_version();
        return *this;
    }

    // Fast path: zero fill with memset (all IEEE/integer zero representations are 0x00).
    // Guard: quantized tensors with non-zero zero_point encode 0.0 as the zero_point
    // byte value, NOT 0x00, so the memset would silently produce the wrong result.
    if (value == 0.0f && (!is_quantized() || q_zero_point() == 0)) {
        std::memset(data_ptr(), 0, static_cast<size_t>(n) * dtype_size());
        bump_version();
        return *this;
    }

    // Direct implementation for CPU tensors - use std::fill_n (auto-vectorized by compiler)
    switch (impl_->dtype) {
        case DType::Float32: std::fill_n(data<float>(), n, static_cast<float>(value)); break;
        case DType::Float64: std::fill_n(data<double>(), n, static_cast<double>(value)); break;
        case DType::Int32: std::fill_n(data<int32_t>(), n, checked_narrow<int32_t>(value)); break;
        case DType::Int64: std::fill_n(data<int64_t>(), n, checked_narrow<int64_t>(value)); break;
        case DType::UInt8: std::fill_n(data<uint8_t>(), n, checked_narrow<uint8_t>(value)); break;
        case DType::UInt16: std::fill_n(data<uint16_t>(), n, checked_narrow<uint16_t>(value)); break;
        case DType::UInt32: std::fill_n(data<uint32_t>(), n, checked_narrow<uint32_t>(value)); break;
        case DType::UInt64: std::fill_n(data<uint64_t>(), n, checked_narrow<uint64_t>(value)); break;
        case DType::Float16: std::fill_n(data<Float16>(), n, Float16(static_cast<float>(value))); break;
        case DType::BFloat16: std::fill_n(data<BFloat16>(), n, BFloat16(static_cast<float>(value))); break;
        case DType::Int8: std::fill_n(data<int8_t>(), n, checked_narrow<int8_t>(value)); break;
        case DType::Int16: std::fill_n(data<int16_t>(), n, checked_narrow<int16_t>(value)); break;
        case DType::Bool: std::fill_n(data<bool>(), n, value != 0.0f); break;
        case DType::Complex64: std::fill_n(data<std::complex<float>>(), n, std::complex<float>(value, 0.0f)); break;
        case DType::Complex128: std::fill_n(data<std::complex<double>>(), n, std::complex<double>(static_cast<double>(value), 0.0)); break;
        case DType::FP8_E4M3: std::fill_n(data<FP8_E4M3>(), n, FP8_E4M3(static_cast<float>(value))); break;
        case DType::FP8_E5M2: std::fill_n(data<FP8_E5M2>(), n, FP8_E5M2(static_cast<float>(value))); break;
        case DType::QInt8:
            if (q_scale() == 0.0) {
                throw std::runtime_error(
                    "fill_ on quantized tensor requires quantization params: "
                    "call set_quantization_params(scale, zero_point) first");
            } else {
                // quantized value = round(value / scale) + zero_point, clamped to int8 range
                const int64_t qval = static_cast<int64_t>(std::round(value / q_scale())) + q_zero_point();
                std::fill_n(data<int8_t>(), n, static_cast<int8_t>(
                    std::clamp(qval, static_cast<int64_t>(-128), static_cast<int64_t>(127))));
            }
            break;
        case DType::QUInt8:
            if (q_scale() == 0.0) {
                throw std::runtime_error(
                    "fill_ on quantized tensor requires quantization params: "
                    "call set_quantization_params(scale, zero_point) first");
            } else {
                // quantized value = round(value / scale) + zero_point, clamped to uint8 range [0, 255]
                const int64_t qval = static_cast<int64_t>(std::round(value / q_scale())) + q_zero_point();
                std::fill_n(data<uint8_t>(), n, static_cast<uint8_t>(
                    std::clamp(qval, static_cast<int64_t>(0), static_cast<int64_t>(255))));
            }
            break;
        case DType::QInt4x2:
            if (q_scale() == 0.0) {
                throw std::runtime_error(
                    "fill_ on quantized tensor requires quantization params: "
                    "call set_quantization_params(scale, zero_point) first");
            } else {
                // Two 4-bit signed values per byte; clamp to [-8, 7] and pack nibbles.
                // For a uniform fill both nibbles equal qval:
                //   byte = (qval & 0xF) | ((qval & 0xF) << 4)
                const int64_t qval = static_cast<int64_t>(std::round(value / q_scale())) + q_zero_point();
                const int64_t clamped = std::clamp(qval, static_cast<int64_t>(-8), static_cast<int64_t>(7));
                const uint8_t qbyte = static_cast<uint8_t>((clamped & 0xF) | ((clamped & 0xF) << 4));
                std::fill_n(reinterpret_cast<uint8_t*>(data<int8_t>()), n, qbyte);
            }
            break;
        default:
            throw std::runtime_error(std::string("fill_: unsupported dtype ") +
                                     std::string(dtype_name(impl_->dtype)));
    }

    bump_version();
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

    // Handle -1 inference (one dimension can be inferred).
    // Dimension 0 is allowed for zero-element tensors.
    int64_t infer_dim = -1;
    int64_t total = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (infer_dim != -1) {
                throw std::runtime_error("Only one dimension can be inferred");
            }
            infer_dim = static_cast<int64_t>(i);
        } else if (new_shape[i] < 0) {
            throw std::runtime_error("Invalid shape dimension: " + std::to_string(new_shape[i]));
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

    // Non-contiguous tensors cannot be reshaped as a view — make contiguous first.
    // This is safe for all backends and avoids silent corruption when GPU reshape
    // kernels assume contiguous layout.
    Tensor src = is_contiguous() ? *this : contiguous();

    // Build attributes with new shape
    OpAttributes attrs;
    std::string shape_str;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (i > 0) shape_str += ",";
        shape_str += std::to_string(new_shape[i]);
    }
    attrs.set(AttrKey::Shape, shape_str);

    // Dispatch to backend for reshape operation
    std::vector<Tensor> inputs = {src};
    return dispatch(OpId::Reshape, inputs, attrs)[0];
}

auto Tensor::view(std::vector<int64_t> new_shape) const -> Tensor {
    if (!impl_) {
        throw std::runtime_error("Cannot view null tensor");
    }

    if (!is_contiguous()) {
        throw std::runtime_error("View requires contiguous tensor. Use reshape() or contiguous() first.");
    }

    // Handle -1 inference (one dimension can be inferred).
    // Dimension 0 is allowed for zero-element tensors.
    int64_t infer_dim = -1;
    int64_t total = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (infer_dim != -1) {
                throw std::runtime_error("Only one dimension can be inferred");
            }
            infer_dim = static_cast<int64_t>(i);
        } else if (new_shape[i] < 0) {
            throw std::runtime_error("Invalid shape dimension: " + std::to_string(new_shape[i]));
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
    result.impl_ = make_intrusive<TensorImpl>(*impl_);

    // Update shape and recompute strides for contiguous layout
    result.impl_->shape = std::move(new_shape);
    result.impl_->strides = compute_strides(result.impl_->shape);
    result.impl_->view_base_ = impl_->view_base_ ? impl_->view_base_ : impl_;

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
        result.impl_ = make_intrusive<TensorImpl>(*impl_);
        std::swap(result.impl_->shape[0], result.impl_->shape[1]);
        std::swap(result.impl_->strides[0], result.impl_->strides[1]);
        result.impl_->is_contiguous_cache_.store(-1, std::memory_order_release);
        result.impl_->view_base_ = impl_->view_base_ ? impl_->view_base_ : impl_;
        return result;
    }

    // General case - swap specified dimensions
    Tensor result;
    result.impl_ = make_intrusive<TensorImpl>(*impl_);
    std::swap(result.impl_->shape[dim0], result.impl_->shape[dim1]);
    std::swap(result.impl_->strides[dim0], result.impl_->strides[dim1]);
    result.impl_->is_contiguous_cache_.store(-1, std::memory_order_release);
    result.impl_->view_base_ = impl_->view_base_ ? impl_->view_base_ : impl_;

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
    result.impl_ = make_intrusive<TensorImpl>(*impl_);

    std::vector<int64_t> new_shape(ndims);
    std::vector<int64_t> new_strides(ndims);

    for (int64_t i = 0; i < ndims; ++i) {
        new_shape[i] = impl_->shape[dims[i]];
        new_strides[i] = impl_->strides[dims[i]];
    }

    result.impl_->shape = std::move(new_shape);
    result.impl_->strides = std::move(new_strides);
    result.impl_->is_contiguous_cache_.store(-1, std::memory_order_release);
    result.impl_->view_base_ = impl_->view_base_ ? impl_->view_base_ : impl_;

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
        result.impl_ = make_intrusive<TensorImpl>(*impl_);
        result.impl_->shape.erase(result.impl_->shape.begin() + d);
        result.impl_->strides.erase(result.impl_->strides.begin() + d);
        result.impl_->is_contiguous_cache_.store(-1, std::memory_order_release);
        result.impl_->view_base_ = impl_->view_base_ ? impl_->view_base_ : impl_;

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

        // A 0-D (scalar) tensor is valid when all dimensions are squeezed.
        // numel() == 1 for an empty shape (identity of multiplication).

        Tensor result;
        result.impl_ = make_intrusive<TensorImpl>(*impl_);
        result.impl_->shape = std::move(new_shape);
        result.impl_->strides = std::move(new_strides);
        result.impl_->is_contiguous_cache_.store(-1, std::memory_order_release);
        result.impl_->view_base_ = impl_->view_base_ ? impl_->view_base_ : impl_;

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
    result.impl_ = make_intrusive<TensorImpl>(*impl_);

    result.impl_->shape.insert(result.impl_->shape.begin() + dim, 1);

    // Compute stride for new dimension (should be product of all following dims)
    int64_t new_stride = (dim < ndims) ? impl_->strides[dim] : 1;
    result.impl_->strides.insert(result.impl_->strides.begin() + dim, new_stride);
    result.impl_->is_contiguous_cache_.store(-1, std::memory_order_release);
    result.impl_->view_base_ = impl_->view_base_ ? impl_->view_base_ : impl_;

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

auto Tensor::expand(std::vector<int64_t> target_shape) const -> Tensor {
    return tenzor::expand(*this, std::move(target_shape));
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
    result.impl_ = make_intrusive<TensorImpl>(*impl_);

    // Update shape for sliced dimension
    result.impl_->shape[dim] = new_dim_size;

    // Update offset to start at the correct position
    result.impl_->offset += start * impl_->strides[dim];

    // Update stride if step != 1
    if (step != 1) {
        result.impl_->strides[dim] *= step;
    }

    // Validate that the slice doesn't exceed storage bounds.
    // Track both min and max reachable offsets to handle negative strides correctly.
    int64_t min_offset = result.impl_->offset;
    int64_t max_offset = result.impl_->offset;
    for (int64_t d = 0; d < static_cast<int64_t>(result.impl_->shape.size()); ++d) {
        if (result.impl_->shape[d] > 0) {
            int64_t extent = result.impl_->shape[d] - 1;
            int64_t stride = result.impl_->strides[d];
            // Check multiplication overflow
            if (stride != 0 && static_cast<uint64_t>(extent) > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / detail::safe_abs(stride)) {
                throw std::overflow_error("Slice offset computation overflows int64_t");
            }
            int64_t delta = extent * stride;
            if (delta >= 0) {
                // Check addition overflow for positive delta
                if (max_offset > std::numeric_limits<int64_t>::max() - delta) {
                    throw std::overflow_error("Slice offset computation overflows int64_t");
                }
                max_offset += delta;
            } else {
                // Check subtraction underflow for negative delta
                if (min_offset < std::numeric_limits<int64_t>::min() - delta) {
                    throw std::overflow_error("Slice offset computation overflows int64_t");
                }
                min_offset += delta;
            }
        }
    }
    int64_t storage_elements = static_cast<int64_t>(result.impl_->storage->size_bytes() / tenzor::dtype_size(result.impl_->dtype));
    if (min_offset < 0 || max_offset >= storage_elements) {
        throw std::out_of_range("Slice offset exceeds storage bounds");
    }

    result.impl_->is_contiguous_cache_.store(-1, std::memory_order_release);
    result.impl_->view_base_ = impl_->view_base_ ? impl_->view_base_ : impl_;

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
    return base_ptr + checked_mul(static_cast<int64_t>(impl_->offset),
                                  static_cast<int64_t>(dtype_size()));
}

auto Tensor::data_ptr() const -> const void* {
    if (!impl_ || !impl_->storage) {
        return nullptr;
    }
    // Account for offset when accessing sliced tensors
    const auto* base_ptr = static_cast<const uint8_t*>(impl_->storage->data());
    return base_ptr + checked_mul(static_cast<int64_t>(impl_->offset),
                                  static_cast<int64_t>(dtype_size()));
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

    // Return cached result if available
    auto cached = impl_->memory_format_cache_.load(std::memory_order_acquire);
    if (cached >= 0) return static_cast<MemoryFormat>(cached);

    // Compute and cache
    MemoryFormat fmt = MemoryFormat::Contiguous;

    // Check 4D ChannelsLast (NHWC)
    if (impl_->shape.size() == 4) {
        auto nhwc_strides = compute_channels_last_strides(impl_->shape);
        if (impl_->strides == nhwc_strides) {
            fmt = MemoryFormat::ChannelsLast;
        }
    }
    // Check 5D ChannelsLast3d (NDHWC)
    else if (impl_->shape.size() == 5) {
        auto ndhwc_strides = compute_channels_last_3d_strides(impl_->shape);
        if (impl_->strides == ndhwc_strides) {
            fmt = MemoryFormat::ChannelsLast3d;
        }
    }

    impl_->memory_format_cache_.store(static_cast<int8_t>(fmt), std::memory_order_release);
    return fmt;
}

auto Tensor::is_contiguous(MemoryFormat format) const noexcept -> bool {
    if (!impl_) return true;

    switch (format) {
        case MemoryFormat::Contiguous: {
            // Standard row-major contiguous check.
            // Offset is irrelevant — a slice at offset N with correct strides IS contiguous.
            auto expected = compute_strides(impl_->shape);
            return impl_->strides == expected;
        }
        case MemoryFormat::ChannelsLast: {
            if (impl_->shape.size() != 4) return false;
            auto expected = compute_channels_last_strides(impl_->shape);
            return impl_->strides == expected;
        }
        case MemoryFormat::ChannelsLast3d: {
            if (impl_->shape.size() != 5) return false;
            auto expected = compute_channels_last_3d_strides(impl_->shape);
            return impl_->strides == expected;
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
    attrs.set(AttrKey::MemoryFormat, static_cast<int64_t>(static_cast<int>(format)));

    std::vector<Tensor> inputs = {*this};
    return dispatch(OpId::ToMemoryFormat, inputs, attrs)[0];
}

// Member methods delegating to free functions
auto Tensor::narrow(int64_t dim, int64_t start, int64_t length) const -> Tensor {
    return tenzor::narrow(*this, dim, start, length);
}

auto Tensor::select(int64_t dim, int64_t index) const -> Tensor {
    return tenzor::select(*this, dim, index);
}

auto Tensor::unfold(int64_t dim, int64_t size, int64_t step) const -> Tensor {
    int64_t nd = this->ndim();
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd) {
        throw std::out_of_range("unfold: dim " + std::to_string(dim) +
                                " out of range for " + std::to_string(nd) + "D tensor");
    }
    int64_t dim_size = shape()[dim];
    if (size <= 0 || size > dim_size) {
        throw std::invalid_argument("unfold: size must be in (0, " +
                                    std::to_string(dim_size) + "]");
    }
    if (step <= 0) {
        throw std::invalid_argument("unfold: step must be > 0");
    }

    int64_t num_windows = (dim_size - size) / step + 1;

    // Build new shape: replace shape[dim] with num_windows, append size
    auto old_shape = shape();
    auto old_strides = strides();
    std::vector<int64_t> new_shape(old_shape.begin(), old_shape.end());
    new_shape[dim] = num_windows;
    new_shape.push_back(size);

    // Build new strides: stride[dim] *= step, append old stride[dim]
    std::vector<int64_t> new_strides(old_strides.begin(), old_strides.end());
    int64_t old_stride_dim = new_strides[dim];
    new_strides[dim] = old_stride_dim * step;
    new_strides.push_back(old_stride_dim);

    return tenzor::as_strided(*this, new_shape, new_strides, offset());
}

auto Tensor::chunk(int64_t chunks, int64_t dim) const -> std::vector<Tensor> {
    return tenzor::chunk(*this, chunks, dim);
}

// ============================================================================
// Named Dimensions (experimental)
// ============================================================================

auto Dimname::intern(std::string_view name) -> const std::string* {
    static std::unordered_set<std::string> pool;
    static std::shared_mutex mutex;

    // Fast path: read lock
    {
        std::shared_lock lock(mutex);
        auto it = pool.find(std::string(name));
        if (it != pool.end()) return &(*it);
    }
    // Slow path: write lock
    {
        std::unique_lock lock(mutex);
        auto [it, inserted] = pool.emplace(name);
        return &(*it);
    }
}

auto Tensor::names() const -> std::optional<DimnameList> {
    if (!impl_) return std::nullopt;
    return impl_->names_;
}

auto Tensor::has_names() const noexcept -> bool {
    return impl_ && impl_->names_.has_value();
}

auto Tensor::rename(DimnameList names) const -> Tensor {
    if (!impl_) throw std::runtime_error("rename: tensor is not initialized");
    if (static_cast<int64_t>(names.size()) != ndim()) {
        throw std::invalid_argument(
            "rename: expected " + std::to_string(ndim()) + " names, got " +
            std::to_string(names.size()));
    }
    validate_dimnames(names);

    // Create a new TensorImpl that shares storage but has different names
    auto new_impl = make_intrusive<TensorImpl>(*impl_);
    new_impl->names_ = std::move(names);
    Tensor result;
    result.impl_ = std::move(new_impl);
    return result;
}

auto Tensor::rename(std::optional<DimnameList> names) const -> Tensor {
    if (!names) {
        auto new_impl = make_intrusive<TensorImpl>(*impl_);
        new_impl->names_ = std::nullopt;
        Tensor result;
        result.impl_ = std::move(new_impl);
        return result;
    }
    return rename(std::move(*names));
}

auto Tensor::dim_index(std::string_view name) const -> int64_t {
    if (!impl_ || !impl_->names_) {
        throw std::invalid_argument("dim_index: tensor has no named dimensions");
    }
    return find_dim_by_name(*impl_->names_, name);
}

auto quantize_per_tensor(const Tensor& input, double scale, int64_t zero_point,
                         DType dtype) -> Tensor {
    if (!tenzor::is_quantized(dtype)) {
        throw std::invalid_argument("quantize_per_tensor: target dtype must be quantized (QInt8, QUInt8, QInt4x2)");
    }
    if (scale <= 0.0) {
        throw std::invalid_argument("quantize_per_tensor: scale must be positive");
    }

    // Work on CPU
    auto cpu_input = input.to(Device::cpu());
    auto shape = std::vector<int64_t>(cpu_input.shape().begin(), cpu_input.shape().end());
    size_t n = cpu_input.numel();

    // Create output quantized tensor
    Tensor result(shape, dtype, Device::cpu());

    if (dtype == DType::QUInt8) {
        const float* in = cpu_input.data<float>();
        uint8_t* out = result.data<uint8_t>();
        for (size_t i = 0; i < n; ++i) {
            int64_t q = static_cast<int64_t>(std::round(in[i] / scale)) + zero_point;
            out[i] = static_cast<uint8_t>(std::clamp(q, int64_t(0), int64_t(255)));
        }
    } else if (dtype == DType::QInt8) {
        const float* in = cpu_input.data<float>();
        int8_t* out = result.data<int8_t>();
        for (size_t i = 0; i < n; ++i) {
            int64_t q = static_cast<int64_t>(std::round(in[i] / scale)) + zero_point;
            out[i] = static_cast<int8_t>(std::clamp(q, int64_t(-128), int64_t(127)));
        }
    } else if (dtype == DType::QInt4x2) {
        // Audit J1: real Int4 packing — two 4-bit signed values per byte.
        //
        // Layout: the last dim is "packed" (halved with ceil-rounding for
        // odd lengths). For an input shape of `[..., N]`, the output
        // QInt4x2 tensor has shape `[..., (N + 1) / 2]`. Each output byte
        // stores two 4-bit signed values: bits 0..3 hold the even-indexed
        // value and bits 4..7 hold the odd-indexed value (low/high nibble).
        // Odd N leaves the high nibble of the last byte zero.
        //
        // Quantization: q = round(x / scale) + zero_point, clamped to the
        // signed 4-bit range [-8, 7]. The clamped value is masked to
        // 4 bits via `& 0xF` (two's-complement preserved).
        const int64_t Q_MIN = -8;
        const int64_t Q_MAX =  7;

        const auto& in_shape = cpu_input.shape();
        const int64_t last_in = in_shape.empty() ? 0 : in_shape.back();
        const int64_t last_packed = (last_in + 1) / 2;
        std::vector<int64_t> packed_shape(in_shape.begin(), in_shape.end());
        if (packed_shape.empty()) {
            packed_shape.push_back(0);  // empty input → empty output
        } else {
            packed_shape.back() = last_packed;
        }

        // Re-allocate `result` with the packed shape (the earlier
        // shape-preserving allocation is wrong for QInt4x2).
        result = Tensor(packed_shape, dtype, Device::cpu());

        const float* in = cpu_input.data<float>();
        uint8_t* out = reinterpret_cast<uint8_t*>(result.data<int8_t>());

        const int64_t outer = (last_in == 0) ? 0 :
            (static_cast<int64_t>(n) / last_in);

        for (int64_t row = 0; row < outer; ++row) {
            const float* in_row = in + row * last_in;
            uint8_t* out_row = out + row * last_packed;
            for (int64_t j = 0; j < last_packed; ++j) {
                int64_t i_lo = 2 * j;
                int64_t i_hi = i_lo + 1;
                int64_t q_lo = static_cast<int64_t>(std::round(in_row[i_lo] / scale)) + zero_point;
                q_lo = std::clamp(q_lo, Q_MIN, Q_MAX);
                uint8_t lo_nibble = static_cast<uint8_t>(q_lo & 0xF);

                uint8_t hi_nibble = 0;
                if (i_hi < last_in) {
                    int64_t q_hi = static_cast<int64_t>(std::round(in_row[i_hi] / scale)) + zero_point;
                    q_hi = std::clamp(q_hi, Q_MIN, Q_MAX);
                    hi_nibble = static_cast<uint8_t>(q_hi & 0xF);
                }
                out_row[j] = static_cast<uint8_t>((hi_nibble << 4) | lo_nibble);
            }
        }
    } else {
        throw std::runtime_error("quantize_per_tensor: unsupported dtype");
    }

    // Store quantization parameters
    result.set_quantization_params(scale, zero_point);

    return result.to(input.device());
}

auto int4_is_native(Device::Type device_type) -> bool {
    // See tenzor.hpp for the full matrix. CPU/CUDA/ROCm have int4
    // quantized kernels; Vulkan/OneAPI/MPS do not and would need a
    // software bit-unpack compute shader that hasn't been written yet.
    switch (device_type) {
        case Device::Type::CPU:
        case Device::Type::CUDA:
        case Device::Type::ROCm:
            return true;
        case Device::Type::OneAPI:
        case Device::Type::Vulkan:
        case Device::Type::MPS:
            return false;
        default:
            return false;
    }
}

} // namespace tenzor
