/**
 * @file tensor.hpp
 * @brief Core tensor class for multi-dimensional array operations
 *
 * Provides the main Tensor class with support for multi-dimensional arrays,
 * automatic differentiation, device management, and various mathematical operations.
 * Uses PImpl pattern for efficient memory management and copy semantics.
 */

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
namespace rocm {
    class HIPKernelAccess;  // Forward declaration for friend access
}

// Forward declaration for Vulkan backend
class VulkanBackend;

/**
 * @brief Multi-dimensional array with automatic differentiation support.
 *
 * The Tensor class is the core data structure for numerical computations.
 * It supports:
 * - Multi-dimensional arrays with arbitrary shape
 * - Multiple data types (float, int, complex, etc.)
 * - Multiple device backends (CPU, CUDA, ROCm, OneAPI)
 * - Automatic differentiation for gradient computation
 * - Efficient memory management with shared storage
 * - Broadcasting and advanced indexing
 *
 * Tensors use copy-on-write semantics for efficient memory usage.
 *
 * @code
 * // Create tensors
 * Tensor a({3, 4}, DType::Float32, Device::cpu());
 * Tensor b = a.cuda();  // Move to GPU
 *
 * // Operations
 * Tensor c = a + b;
 * Tensor d = c.reshape({12});
 * @endcode
 */
class Tensor {
public:
    /**
     * @brief Default constructor creating an empty tensor.
     */
    Tensor() = default;

    /**
     * @brief Construct tensor with specified shape, dtype, and device.
     *
     * @param shape Dimensions of the tensor (e.g., {3, 4} for 3x4 matrix)
     * @param dtype Data type of tensor elements
     * @param device Device where tensor will be allocated
     *
     * @code
     * Tensor t({2, 3, 4}, DType::Float32, Device::cuda(0));
     * @endcode
     */
    Tensor(std::vector<int64_t> shape, DType dtype, Device device);

    /**
     * @brief Copy constructor (shallow copy with shared storage).
     */
    Tensor(const Tensor&) = default;

    /**
     * @brief Move constructor.
     */
    Tensor(Tensor&&) noexcept = default;

    /**
     * @brief Copy assignment operator (shallow copy with shared storage).
     */
    Tensor& operator=(const Tensor&) = default;

    /**
     * @brief Move assignment operator.
     */
    Tensor& operator=(Tensor&&) noexcept = default;

    /**
     * @brief Destructor.
     */
    ~Tensor() = default;

    // ============================================================================
    // Properties
    // ============================================================================

    /**
     * @brief Get the shape of the tensor.
     *
     * @return Span of dimension sizes (e.g., {3, 4} for 3x4 matrix)
     *
     * @code
     * Tensor t({2, 3, 4}, DType::Float32, Device::cpu());
     * auto s = t.shape();  // Returns {2, 3, 4}
     * @endcode
     */
    auto shape() const noexcept -> std::span<const int64_t>;

    /**
     * @brief Get the strides of the tensor.
     *
     * Strides define memory layout - the number of elements to skip
     * to move to the next element along each dimension.
     *
     * @return Span of stride values
     *
     * @code
     * Tensor t({2, 3}, DType::Float32, Device::cpu());
     * auto s = t.strides();  // Returns {3, 1} for row-major layout
     * @endcode
     */
    auto strides() const noexcept -> std::span<const int64_t>;

    /**
     * @brief Get the number of dimensions.
     *
     * @return Number of dimensions (rank) of the tensor
     */
    auto ndim() const noexcept -> int64_t;

    /**
     * @brief Get the total number of elements.
     *
     * @return Product of all dimension sizes
     *
     * @code
     * Tensor t({2, 3, 4}, DType::Float32, Device::cpu());
     * auto n = t.numel();  // Returns 24
     * @endcode
     */
    auto numel() const noexcept -> int64_t;

    /**
     * @brief Get the data type of tensor elements.
     *
     * @return DType enumeration value
     */
    auto dtype() const noexcept -> DType;

    /**
     * @brief Get the device where tensor resides.
     *
     * @return Device reference
     */
    auto device() const noexcept -> const Device&;

    /**
     * @brief Check if tensor requires gradient computation.
     *
     * @return true if gradients will be computed for this tensor
     */
    auto requires_grad() const noexcept -> bool;

    /**
     * @brief Check if tensor is contiguous in memory.
     *
     * A tensor is contiguous if elements are stored in row-major order
     * without gaps. Many operations are faster on contiguous tensors.
     *
     * @return true if tensor is contiguous
     *
     * @see contiguous()
     */
    auto is_contiguous() const noexcept -> bool;

    // ============================================================================
    // Data Access
    // ============================================================================

    /**
     * @brief Get mutable pointer to tensor data.
     *
     * Provides type-safe access to underlying data buffer. Type must match
     * the tensor's dtype.
     *
     * @tparam T Scalar type (must match tensor dtype)
     * @return Pointer to first element
     * @throws std::runtime_error if type doesn't match dtype
     *
     * @code
     * Tensor t({3, 4}, DType::Float32, Device::cpu());
     * float* ptr = t.data<float>();
     * ptr[0] = 1.0f;
     * @endcode
     */
    template<typename T>
    auto data() -> T*;

    /**
     * @brief Get const pointer to tensor data.
     *
     * @tparam T Scalar type (must match tensor dtype)
     * @return Const pointer to first element
     * @throws std::runtime_error if type doesn't match dtype
     */
    template<typename T>
    auto data() const -> const T*;

    /**
     * @brief Extract scalar value from 0-dimensional tensor.
     *
     * @tparam T Scalar type (must match tensor dtype)
     * @return Scalar value
     * @throws std::runtime_error if tensor is not 0-dimensional or type mismatch
     *
     * @code
     * Tensor t = sum_of_tensor;  // 0-dimensional result
     * float value = t.item<float>();
     * @endcode
     */
    template<typename T> requires ScalarType<T>
    auto item() const -> T;

    // ============================================================================
    // Device Management
    // ============================================================================

    /**
     * @brief Move tensor to specified device.
     *
     * Creates a new tensor on the target device with copied data.
     * If tensor is already on the device, returns a copy.
     *
     * @param device Target device
     * @return New tensor on specified device
     *
     * @code
     * Tensor cpu_t({3, 4}, DType::Float32, Device::cpu());
     * Tensor gpu_t = cpu_t.to(Device::cuda(0));
     * @endcode
     */
    auto to(Device device) const -> Tensor;

    /**
     * @brief Convert tensor to specified data type.
     *
     * @param dtype Target data type
     * @return New tensor with converted data type
     *
     * @code
     * Tensor f32_t({3, 4}, DType::Float32, Device::cpu());
     * Tensor f64_t = f32_t.to(DType::Float64);
     * @endcode
     */
    auto to(DType dtype) const -> Tensor;

    /**
     * @brief Move tensor to CUDA device.
     *
     * @param device_id GPU device index (default: 0)
     * @return New tensor on CUDA device
     *
     * @code
     * Tensor t = cpu_tensor.cuda(1);  // Move to GPU 1
     * @endcode
     */
    auto cuda(int32_t device_id = 0) const -> Tensor;

    /**
     * @brief Move tensor to CPU.
     *
     * @return New tensor on CPU
     */
    auto cpu() const -> Tensor;

    // ============================================================================
    // Shape Manipulation
    // ============================================================================

    /**
     * @brief Reshape tensor to new dimensions.
     *
     * Returns a tensor with the same data but different shape.
     * Total number of elements must remain the same.
     *
     * @param new_shape New dimensions (use -1 for auto-inferred dimension)
     * @return Reshaped tensor (may share storage if contiguous)
     * @throws std::runtime_error if new shape is incompatible
     *
     * @code
     * Tensor t({2, 6}, DType::Float32, Device::cpu());
     * Tensor r = t.reshape({3, 4});  // 2x6 -> 3x4
     * Tensor a = t.reshape({-1});    // Flatten to 1D
     * @endcode
     */
    auto reshape(std::vector<int64_t> new_shape) const -> Tensor;

    /**
     * @brief Create view with new shape (zero-copy).
     *
     * Like reshape but guarantees zero-copy operation.
     * Requires tensor to be contiguous.
     *
     * @param new_shape New dimensions
     * @return View with new shape (shares storage)
     * @throws std::runtime_error if tensor is not contiguous
     */
    auto view(std::vector<int64_t> new_shape) const -> Tensor;

    /**
     * @brief Transpose two dimensions.
     *
     * @param dim0 First dimension to swap
     * @param dim1 Second dimension to swap
     * @return Transposed tensor (shares storage)
     *
     * @code
     * Tensor t({2, 3, 4}, DType::Float32, Device::cpu());
     * Tensor t2 = t.transpose(0, 2);  // Shape: {4, 3, 2}
     * @endcode
     */
    auto transpose(int64_t dim0, int64_t dim1) const -> Tensor;

    /**
     * @brief Permute dimensions.
     *
     * @param dims New ordering of dimensions
     * @return Permuted tensor (shares storage)
     * @throws std::runtime_error if dims is invalid
     *
     * @code
     * Tensor t({2, 3, 4}, DType::Float32, Device::cpu());
     * Tensor p = t.permute({2, 0, 1});  // Shape: {4, 2, 3}
     * @endcode
     */
    auto permute(std::vector<int64_t> dims) const -> Tensor;

    /**
     * @brief Remove dimensions of size 1.
     *
     * @param dim Optional specific dimension to squeeze (nullopt = squeeze all)
     * @return Tensor with size-1 dimensions removed
     *
     * @code
     * Tensor t({1, 3, 1, 4}, DType::Float32, Device::cpu());
     * Tensor s = t.squeeze();     // Shape: {3, 4}
     * Tensor s2 = t.squeeze(0);   // Shape: {3, 1, 4}
     * @endcode
     */
    auto squeeze(std::optional<int64_t> dim = std::nullopt) const -> Tensor;

    /**
     * @brief Add dimension of size 1.
     *
     * @param dim Position to insert new dimension
     * @return Tensor with added dimension
     *
     * @code
     * Tensor t({3, 4}, DType::Float32, Device::cpu());
     * Tensor u = t.unsqueeze(0);  // Shape: {1, 3, 4}
     * Tensor u2 = t.unsqueeze(1); // Shape: {3, 1, 4}
     * @endcode
     */
    auto unsqueeze(int64_t dim) const -> Tensor;

    /**
     * @brief Flatten tensor to 1D or partially flatten.
     *
     * @param start_dim First dimension to flatten (default: 0)
     * @param end_dim Last dimension to flatten (default: -1 = last)
     * @return Flattened tensor
     *
     * @code
     * Tensor t({2, 3, 4}, DType::Float32, Device::cpu());
     * Tensor f = t.flatten();         // Shape: {24}
     * Tensor f2 = t.flatten(1, 2);    // Shape: {2, 12}
     * @endcode
     */
    auto flatten(int64_t start_dim = 0, int64_t end_dim = -1) const -> Tensor;

    /**
     * @brief Return indices of non-zero elements.
     *
     * Returns a 2D tensor where each row contains the indices of a non-zero element.
     *
     * @return Tensor of indices (num_nonzero, ndim) in Int64 format
     *
     * @code
     * Tensor t = Tensor::from_data(std::vector<float>{0, 1, 0, 2}, {4});
     * Tensor idx = t.nonzero(); // Shape: {2, 1}, values: [[1], [3]]
     * @endcode
     */
    auto nonzero() const -> Tensor;

    // ============================================================================
    // Indexing
    // ============================================================================

    /**
     * @brief Index tensor along first dimension.
     *
     * @param idx Index value (supports negative indexing)
     * @return Tensor with first dimension removed
     *
     * @code
     * Tensor t({3, 4}, DType::Float32, Device::cpu());
     * Tensor row = t[0];    // Shape: {4}
     * Tensor last = t[-1];  // Last row
     * @endcode
     */
    auto operator[](int64_t idx) const -> Tensor;

    /**
     * @brief Slice tensor along a dimension.
     *
     * @param dim Dimension to slice
     * @param start Start index (inclusive)
     * @param end End index (exclusive)
     * @param step Step size (default: 1)
     * @return Sliced tensor (shares storage)
     *
     * @code
     * Tensor t({10, 5}, DType::Float32, Device::cpu());
     * Tensor s = t.slice(0, 2, 7);     // Rows 2-6
     * Tensor s2 = t.slice(0, 0, 10, 2); // Every other row
     * @endcode
     */
    auto slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const -> Tensor;

    // ============================================================================
    // Arithmetic Operators
    // ============================================================================

    /**
     * @brief Element-wise addition.
     *
     * @param other Tensor to add (must be broadcastable)
     * @return New tensor with sum
     *
     * @code
     * Tensor a({3, 4}, DType::Float32, Device::cpu());
     * Tensor b({3, 4}, DType::Float32, Device::cpu());
     * Tensor c = a + b;
     * @endcode
     */
    auto operator+(const Tensor& other) const -> Tensor;

    /**
     * @brief Element-wise subtraction.
     *
     * @param other Tensor to subtract (must be broadcastable)
     * @return New tensor with difference
     */
    auto operator-(const Tensor& other) const -> Tensor;

    /**
     * @brief Element-wise multiplication.
     *
     * @param other Tensor to multiply (must be broadcastable)
     * @return New tensor with product
     */
    auto operator*(const Tensor& other) const -> Tensor;

    /**
     * @brief Element-wise division.
     *
     * @param other Tensor to divide by (must be broadcastable)
     * @return New tensor with quotient
     */
    auto operator/(const Tensor& other) const -> Tensor;

    // ============================================================================
    // Scalar Operations
    // ============================================================================

    /**
     * @brief Add scalar to all elements.
     *
     * @param scalar Value to add
     * @return New tensor with scalar added
     */
    auto operator+(float scalar) const -> Tensor;

    /**
     * @brief Subtract scalar from all elements.
     *
     * @param scalar Value to subtract
     * @return New tensor with scalar subtracted
     */
    auto operator-(float scalar) const -> Tensor;

    /**
     * @brief Multiply all elements by scalar.
     *
     * @param scalar Value to multiply by
     * @return New tensor with scalar multiplication
     */
    auto operator*(float scalar) const -> Tensor;

    /**
     * @brief Divide all elements by scalar.
     *
     * @param scalar Value to divide by
     * @return New tensor with scalar division
     */
    auto operator/(float scalar) const -> Tensor;

    // ============================================================================
    // In-place Operations
    // ============================================================================

    /**
     * @brief In-place element-wise addition.
     *
     * @param other Tensor to add (must be broadcastable)
     * @return Reference to this tensor
     *
     * @note Modifies this tensor in-place
     */
    auto operator+=(const Tensor& other) -> Tensor&;

    /**
     * @brief In-place element-wise subtraction.
     *
     * @param other Tensor to subtract
     * @return Reference to this tensor
     */
    auto operator-=(const Tensor& other) -> Tensor&;

    /**
     * @brief In-place element-wise multiplication.
     *
     * @param other Tensor to multiply
     * @return Reference to this tensor
     */
    auto operator*=(const Tensor& other) -> Tensor&;

    /**
     * @brief In-place element-wise division.
     *
     * @param other Tensor to divide by
     * @return Reference to this tensor
     */
    auto operator/=(const Tensor& other) -> Tensor&;

    /**
     * @brief Fill tensor with scalar value.
     *
     * @param value Value to fill with
     * @return Reference to this tensor
     *
     * @code
     * Tensor t({3, 4}, DType::Float32, Device::cpu());
     * t.fill_(1.5f);  // All elements = 1.5
     * @endcode
     */
    auto fill_(float value) -> Tensor&;

    /**
     * @brief Fill tensor with zeros.
     *
     * @return Reference to this tensor
     *
     * @code
     * Tensor t({3, 4}, DType::Float32, Device::cpu());
     * t.zero_();  // All elements = 0
     * @endcode
     */
    auto zero_() -> Tensor&;

    // ============================================================================
    // Comparison Operators
    // ============================================================================

    /**
     * @brief Element-wise equality comparison.
     *
     * @param other Tensor to compare
     * @return Boolean tensor with comparison results
     */
    auto operator==(const Tensor& other) const -> Tensor;

    /**
     * @brief Element-wise inequality comparison.
     *
     * @param other Tensor to compare
     * @return Boolean tensor with comparison results
     */
    auto operator!=(const Tensor& other) const -> Tensor;

    /**
     * @brief Element-wise less-than comparison.
     *
     * @param other Tensor to compare
     * @return Boolean tensor with comparison results
     */
    auto operator<(const Tensor& other) const -> Tensor;

    /**
     * @brief Element-wise greater-than comparison.
     *
     * @param other Tensor to compare
     * @return Boolean tensor with comparison results
     */
    auto operator>(const Tensor& other) const -> Tensor;

    /**
     * @brief Element-wise less-than-or-equal comparison.
     *
     * @param other Tensor to compare
     * @return Boolean tensor with comparison results
     */
    auto operator<=(const Tensor& other) const -> Tensor;

    /**
     * @brief Element-wise greater-than-or-equal comparison.
     *
     * @param other Tensor to compare
     * @return Boolean tensor with comparison results
     */
    auto operator>=(const Tensor& other) const -> Tensor;

    // ============================================================================
    // Memory Management
    // ============================================================================

    /**
     * @brief Create deep copy of tensor.
     *
     * Allocates new storage and copies all data.
     *
     * @return New independent tensor with copied data
     *
     * @code
     * Tensor a({3, 4}, DType::Float32, Device::cpu());
     * Tensor b = a.clone();  // Independent copy
     * @endcode
     */
    auto clone() const -> Tensor;

    /**
     * @brief Detach tensor from autograd graph.
     *
     * Creates a new tensor that shares storage but has no gradient history.
     * Use when you want to prevent gradient computation.
     *
     * @return Detached tensor (shares storage, no gradients)
     *
     * @code
     * Tensor a = some_computation();
     * Tensor b = a.detach();  // No gradients through b
     * @endcode
     */
    auto detach() const -> Tensor;

    /**
     * @brief Return contiguous copy if needed.
     *
     * If tensor is already contiguous, returns this tensor.
     * Otherwise, creates a new contiguous copy.
     *
     * @return Contiguous tensor (may be this or new tensor)
     *
     * @code
     * Tensor t = some_tensor.transpose(0, 1);
     * Tensor c = t.contiguous();  // Ensures contiguous layout
     * @endcode
     */
    auto contiguous() const -> Tensor;

    // ============================================================================
    // Utility Methods (Phase 8 additions)
    // ============================================================================

    /**
     * @brief Get size of data type in bytes.
     *
     * @return Number of bytes per element
     *
     * @code
     * Tensor t({3, 4}, DType::Float32, Device::cpu());
     * size_t bytes = t.dtype_size();  // Returns 4
     * @endcode
     */
    auto dtype_size() const noexcept -> size_t;

    /**
     * @brief Get raw pointer to tensor data (type-erased).
     *
     * Returns void* pointer to raw data buffer. Use with caution.
     * Prefer typed data<T>() method when possible.
     *
     * @return Void pointer to first element
     *
     * @code
     * Tensor t({3, 4}, DType::Float32, Device::cpu());
     * void* ptr = t.data_ptr();
     * @endcode
     */
    auto data_ptr() -> void*;

    /**
     * @brief Get const raw pointer to tensor data (type-erased).
     *
     * @return Const void pointer to first element
     */
    auto data_ptr() const -> const void*;

    /**
     * @brief Create zero tensor with same shape and dtype as input.
     *
     * @param other Template tensor
     * @return New zero tensor with same shape/dtype/device
     *
     * @code
     * Tensor a({3, 4}, DType::Float32, Device::cpu());
     * Tensor zeros = Tensor::zeros_like(a);
     * @endcode
     */
    static auto zeros_like(const Tensor& other) -> Tensor;

    // Implementation access
    auto impl() const -> const std::shared_ptr<TensorImpl>& { return impl_; }

private:
    std::shared_ptr<TensorImpl> impl_;

    friend class Variable;
    friend class cuda::CUDAKernelAccess;  // Allow CUDA kernels to access impl_
    friend class rocm::HIPKernelAccess;   // Allow ROCm/HIP kernels to access impl_
    friend class VulkanBackend;           // Allow Vulkan backend to access impl_

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

/**
 * @brief Internal tensor implementation (PImpl pattern).
 *
 * Manages the actual tensor data and metadata. Not intended for direct use.
 * Access tensors through the Tensor class interface.
 *
 * @note This class is an implementation detail and may change between versions.
 */
class TensorImpl {
public:
    std::shared_ptr<Storage> storage;    ///< Shared storage for tensor data
    std::vector<int64_t> shape;          ///< Tensor dimensions
    std::vector<int64_t> strides;        ///< Memory strides per dimension
    int64_t offset{0};                   ///< Offset into storage
    DType dtype;                         ///< Element data type
    Device device;                       ///< Device location
    bool requires_grad{false};           ///< Gradient computation flag

    /**
     * @brief Construct tensor implementation.
     *
     * @param shape Tensor dimensions
     * @param dtype Element data type
     * @param device Device location
     */
    TensorImpl(std::vector<int64_t> shape, DType dtype, Device device);

    /**
     * @brief Get total number of elements.
     *
     * @return Product of all dimensions
     */
    auto numel() const -> int64_t;

    /**
     * @brief Check if tensor is contiguous in memory.
     *
     * @return true if elements are stored without gaps
     */
    auto is_contiguous() const -> bool;
};

} // namespace tenzor
