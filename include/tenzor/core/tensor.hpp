#pragma once

#include <memory>
#include <vector>
#include <span>
#include <optional>
#include "dtype.hpp"
#include "device.hpp"
#include "storage.hpp"

namespace tenzor {

// Forward declarations
class TensorImpl;
class Tensor;

// Forward declarations for backend kernel functions
namespace cpu {
    auto clone_kernel(const tenzor::Tensor& input) -> tenzor::Tensor;
    auto reshape_kernel(const tenzor::Tensor& input, const std::vector<int64_t>& new_shape) -> tenzor::Tensor;
    auto transpose_kernel(const tenzor::Tensor& input, int64_t dim0, int64_t dim1) -> tenzor::Tensor;
    auto permute_kernel(const tenzor::Tensor& input, const std::vector<int64_t>& dims) -> tenzor::Tensor;
    auto squeeze_kernel(const tenzor::Tensor& input, int64_t dim) -> tenzor::Tensor;
    auto unsqueeze_kernel(const tenzor::Tensor& input, int64_t dim) -> tenzor::Tensor;
    auto contiguous_kernel(const tenzor::Tensor& input) -> tenzor::Tensor;
}
namespace cuda {
    class CUDAKernelAccess;  // Forward declaration for friend access

    auto clone_kernel(const tenzor::Tensor& input) -> tenzor::Tensor;
    auto reshape_kernel(const tenzor::Tensor& input, const std::vector<int64_t>& new_shape) -> tenzor::Tensor;
    auto transpose_kernel(const tenzor::Tensor& input, int64_t dim0, int64_t dim1) -> tenzor::Tensor;
    auto permute_kernel(const tenzor::Tensor& input, const std::vector<int64_t>& dims) -> tenzor::Tensor;
    auto squeeze_kernel(const tenzor::Tensor& input, int64_t dim) -> tenzor::Tensor;
    auto unsqueeze_kernel(const tenzor::Tensor& input, int64_t dim) -> tenzor::Tensor;
    auto contiguous_kernel(const tenzor::Tensor& input) -> tenzor::Tensor;
}

// Main Tensor class
class Tensor {
public:
    // Construction
    Tensor() = default;
    Tensor(std::vector<int64_t> shape, DType dtype, Device device);
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() = default;

    // Properties
    auto shape() const noexcept -> std::span<const int64_t>;
    auto strides() const noexcept -> std::span<const int64_t>;
    auto ndim() const noexcept -> int64_t;
    auto numel() const noexcept -> int64_t;
    auto dtype() const noexcept -> DType;
    auto device() const noexcept -> const Device&;
    auto requires_grad() const noexcept -> bool;
    auto is_contiguous() const noexcept -> bool;

    // Data access (type-safe)
    template<typename T> requires ScalarType<T>
    auto data() -> T*;

    template<typename T> requires ScalarType<T>
    auto data() const -> const T*;

    // Item extraction (for scalar tensors)
    template<typename T> requires ScalarType<T>
    auto item() const -> T;

    // Device management
    auto to(Device device) const -> Tensor;
    auto to(DType dtype) const -> Tensor;
    auto cuda(int32_t device_id = 0) const -> Tensor;
    auto cpu() const -> Tensor;

    // Shape manipulation
    auto reshape(std::vector<int64_t> new_shape) const -> Tensor;
    auto view(std::vector<int64_t> new_shape) const -> Tensor;  // zero-copy
    auto transpose(int64_t dim0, int64_t dim1) const -> Tensor;
    auto permute(std::vector<int64_t> dims) const -> Tensor;
    auto squeeze(std::optional<int64_t> dim = std::nullopt) const -> Tensor;
    auto unsqueeze(int64_t dim) const -> Tensor;
    auto flatten(int64_t start_dim = 0, int64_t end_dim = -1) const -> Tensor;

    // Indexing
    auto operator[](int64_t idx) const -> Tensor;
    auto slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const -> Tensor;

    // Arithmetic operators
    auto operator+(const Tensor& other) const -> Tensor;
    auto operator-(const Tensor& other) const -> Tensor;
    auto operator*(const Tensor& other) const -> Tensor;
    auto operator/(const Tensor& other) const -> Tensor;

    // Scalar operations
    auto operator+(float scalar) const -> Tensor;
    auto operator-(float scalar) const -> Tensor;
    auto operator*(float scalar) const -> Tensor;
    auto operator/(float scalar) const -> Tensor;

    // In-place operations
    auto operator+=(const Tensor& other) -> Tensor&;
    auto operator-=(const Tensor& other) -> Tensor&;
    auto operator*=(const Tensor& other) -> Tensor&;
    auto operator/=(const Tensor& other) -> Tensor&;
    auto fill_(float value) -> Tensor&;
    auto zero_() -> Tensor&;

    // Comparison
    auto operator==(const Tensor& other) const -> Tensor;
    auto operator!=(const Tensor& other) const -> Tensor;
    auto operator<(const Tensor& other) const -> Tensor;
    auto operator>(const Tensor& other) const -> Tensor;

    // Memory management
    auto clone() const -> Tensor;
    auto detach() const -> Tensor;
    auto contiguous() const -> Tensor;

    // Implementation access
    auto impl() const -> const std::shared_ptr<TensorImpl>& { return impl_; }

private:
    std::shared_ptr<TensorImpl> impl_;

    friend class Variable;
    friend class cuda::CUDAKernelAccess;  // Allow CUDA kernels to access impl_

    // Friend declarations for backend kernels that need direct access to impl_
    friend auto cpu::clone_kernel(const Tensor& input) -> Tensor;
    friend auto cpu::reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape) -> Tensor;
    friend auto cpu::transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor;
    friend auto cpu::permute_kernel(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor;
    friend auto cpu::squeeze_kernel(const Tensor& input, int64_t dim) -> Tensor;
    friend auto cpu::unsqueeze_kernel(const Tensor& input, int64_t dim) -> Tensor;
    friend auto cpu::contiguous_kernel(const Tensor& input) -> Tensor;
    friend auto cuda::clone_kernel(const Tensor& input) -> Tensor;
    friend auto cuda::reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape) -> Tensor;
    friend auto cuda::transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor;
    friend auto cuda::permute_kernel(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor;
    friend auto cuda::squeeze_kernel(const Tensor& input, int64_t dim) -> Tensor;
    friend auto cuda::unsqueeze_kernel(const Tensor& input, int64_t dim) -> Tensor;
    friend auto cuda::contiguous_kernel(const Tensor& input) -> Tensor;
};

// Tensor implementation (PImpl pattern)
class TensorImpl {
public:
    std::shared_ptr<Storage> storage;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t offset{0};
    DType dtype;
    Device device;
    bool requires_grad{false};

    TensorImpl(std::vector<int64_t> shape, DType dtype, Device device);

    auto numel() const -> int64_t;
    auto is_contiguous() const -> bool;
};

} // namespace tenzor
