/**
 * @file tensor.hpp
 * @brief Core tensor class for multi-dimensional array operations
 *
 * Provides the main Tensor class with support for multi-dimensional arrays,
 * automatic differentiation, device management, and various mathematical operations.
 * Uses PImpl pattern for efficient memory management and copy semantics.
 *
 * @par Thread Safety
 * - **Read operations** (shape, dtype, data access, const methods) are thread-safe.
 *   Multiple threads may read the same tensor concurrently.
 * - **Concurrent read + write** to the same tensor requires external synchronization.
 *   This includes in-place operations (+=, fill_), reshape, and any mutation.
 * - **Independent tensors** can be operated on concurrently without synchronization.
 * - **Autograd forward pass** is NOT thread-safe on shared Variables. Each thread
 *   should use its own Variable or synchronize access.
 * - **Autograd backward pass** is thread-safe with per-Variable `make_thread_safe()`.
 */

#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <span>
#include <functional>
#include <optional>
#include "dtype.hpp"
#include "device.hpp"
#include "storage.hpp"
#include "shape.hpp"
#include "dimname.hpp"

namespace tenzor {

/**
 * @brief Memory format for tensor layout optimization.
 *
 * Describes the physical memory layout of tensor data. Memory format is
 * expressed through stride patterns while keeping the logical dimension
 * order unchanged.
 *
 * For a 4D tensor with shape [N, C, H, W]:
 * - Contiguous (NCHW): strides = [C*H*W, H*W, W, 1]
 * - ChannelsLast (NHWC): strides = [H*W*C, 1, W*C, C]
 *
 * Using ChannelsLast format on modern GPUs with Tensor Cores can provide
 * 30-100% speedup for convolution operations.
 */
enum class MemoryFormat {
    Contiguous,      ///< Standard row-major (NCHW for 4D)
    ChannelsLast,    ///< Channels-last layout (NHWC for 4D)
    ChannelsLast3d,  ///< Channels-last for 5D (NDHWC)
    Preserve         ///< Preserve input format in operations
};

// Forward declarations
class TensorImpl;
class Tensor;
class TensorAccessor;

// Forward declarations for backend kernel accessor classes
namespace cuda {
    class CUDAKernelAccess;
}
namespace rocm {
    class HIPKernelAccess;
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
 * @warning **Shallow copy semantics (matches PyTorch):** Tensor copies share
 * underlying storage. `Tensor b = a;` does NOT copy data — both `a` and `b`
 * point to the same memory. Mutations through one (e.g., `fill_()`) are visible
 * through the other. Use `.clone()` for an independent deep copy.
 * Similarly, `.to(device)` returns `*this` (no copy) when already on the target
 * device. `.reshape()`, `.transpose()`, `.slice()` also create views sharing
 * storage — NOT copies.
 *
 * View vs Copy summary:
 * | View (shares storage)         | Copy (new storage)          |
 * |-------------------------------|-----------------------------|
 * | reshape, view                 | clone                       |
 * | transpose, permute            | contiguous (if non-contig)  |
 * | squeeze, unsqueeze            | repeat                      |
 * | flatten (if contiguous)       | flatten (if non-contiguous) |
 * | slice, select, narrow         |                             |
 * | expand, chunk                 |                             |
 * | detach                        |                             |
 *
 * Tensors use shared storage with reference counting for efficient memory usage.
 *
 * Thread Safety:
 * - Tensor is NOT thread-safe for concurrent mutation (write + write, or read + write)
 * - Tensor IS safe for concurrent reads from multiple threads
 * - For concurrent gradient accumulation, use Variable::make_thread_safe()
 * - Forward/backward on shared Variables from different threads requires
 *   external synchronization (e.g., mutex)
 *
 * @threadsafety Read operations (shape(), strides(), ndim(), numel(), dtype(),
 * device(), is_contiguous(), data<T>() const, item<T>()) are thread-safe and
 * may be called concurrently from multiple threads on the same Tensor instance.
 * Write operations (including in-place ops such as fill_(), zero_(), operator+=,
 * and view-creating mutations like reshape() and transpose()) require external
 * synchronization. The version_counter_ and is_contiguous_cache_ members use
 * std::atomic for safe concurrent reads, but concurrent read+write still
 * requires a fence or external lock on the mutable operation side.
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
     * Memory is zero-initialized by default.
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
     * @brief Create an uninitialized tensor.
     *
     * Memory is NOT zero-initialized. Use when you will immediately
     * overwrite all values (e.g., output of arithmetic operations).
     * This is more efficient as it avoids wasteful memset.
     *
     * @param shape Dimensions of the tensor
     * @param dtype Data type of tensor elements
     * @param device Device where tensor will be allocated
     * @return Uninitialized tensor
     *
     * @warning Reading from uninitialized memory is undefined behavior.
     *          Only use when all elements will be written before reading.
     *
     * @code
     * auto output = Tensor::empty_uninitialized({1000, 1000}, DType::Float32, Device::cpu());
     * // Fill output with computed values...
     * @endcode
     */
    static auto empty_uninitialized(std::vector<int64_t> shape,
                                     DType dtype = DType::Float32,
                                     Device device = Device::cpu()) -> Tensor;

    /**
     * @brief Wrap externally-owned memory as a Tensor without copying.
     *
     * Creates a tensor that references the given data pointer directly.
     * No memory allocation or copy occurs. The caller must ensure the data
     * remains valid for the lifetime of the tensor (and any views of it).
     *
     * @param data Pointer to existing memory (must not be null if numel > 0)
     * @param shape Tensor dimensions
     * @param dtype Element data type (default: Float32)
     * @param device Device where the memory resides (default: CPU)
     * @param deleter Optional deleter called when the last tensor sharing this
     *                storage is destroyed. If null, memory is not freed.
     * @return Tensor wrapping the external memory
     *
     * @code
     * // Wrap a raw C array
     * float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
     * auto t = Tensor::from_blob(data, {2, 2});
     *
     * // Wrap with custom deleter
     * float* gpu_data = my_gpu_alloc(1024);
     * auto t = Tensor::from_blob(gpu_data, {1024}, DType::Float32,
     *     Device::cuda(0), [](void* p) { my_gpu_free(p); });
     * @endcode
     *
     * @warning Modifying the tensor modifies the underlying memory. If the
     *          external memory is freed while the tensor is alive, behavior
     *          is undefined.
     */
    static auto from_blob(void* data,
                          std::vector<int64_t> shape,
                          DType dtype = DType::Float32,
                          Device device = Device::cpu(),
                          std::function<void(void*)> deleter = nullptr) -> Tensor;

    /**
     * @brief Shallow copy — shares underlying storage with other.
     * Modifications to data through either handle are visible to both.
     * Use clone() for a deep copy.
     */
    Tensor(const Tensor&) = default;

    /**
     * @brief Move constructor.
     */
    Tensor(Tensor&&) noexcept = default;

    /// Shallow copy assignment — shares underlying storage with other.
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
    auto shape() const -> std::span<const int64_t>;

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
    auto strides() const -> std::span<const int64_t>;

    /**
     * @brief Get the number of dimensions.
     *
     * @return Number of dimensions (rank) of the tensor
     */
    auto ndim() const -> int64_t;

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
    auto numel() const -> int64_t;

    /**
     * @brief Get the data type of tensor elements.
     *
     * @return DType enumeration value
     */
    auto dtype() const -> DType;

    /**
     * @brief Get the device where tensor resides.
     *
     * @return Device reference
     */
    auto device() const -> const Device&;

    /**
     * @brief Check if tensor requires gradient computation.
     *
     * @return true if gradients will be computed for this tensor
     */
    auto requires_grad() const noexcept -> bool;

    /// Get the mutation version counter (incremented on in-place operations)
    auto version() const noexcept -> uint64_t;

    /// Increment the mutation version counter
    auto bump_version() -> void;

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

    /**
     * @brief Check if tensor has been initialized with valid storage.
     *
     * Call this before accessing tensor properties on potentially-uninitialized
     * tensors. Non-noexcept accessors (shape(), dtype(), device(), etc.) throw
     * std::runtime_error on invalid tensors. Noexcept accessors
     * (requires_grad(), version(), is_contiguous()) return safe defaults
     * (false, 0, true) on invalid tensors without throwing.
     *
     * @return true if tensor has a valid implementation, false if default-constructed
     */
    auto is_valid() const noexcept -> bool { return impl_ != nullptr; }

    // ========================================================================
    // Named Dimensions (experimental)
    // ========================================================================

    /**
     * @brief Get the dimension names, if any.
     * @return nullopt if tensor is unnamed, otherwise the name list
     */
    auto names() const -> std::optional<DimnameList>;

    /**
     * @brief Check if tensor has named dimensions.
     */
    auto has_names() const noexcept -> bool;

    /**
     * @brief Return a view of this tensor with the given dimension names.
     *
     * The new tensor shares storage with this tensor.
     *
     * @param names List of dimension names (size must equal ndim())
     * @return Named tensor view
     * @throws std::invalid_argument if names.size() != ndim() or duplicates
     */
    auto rename(DimnameList names) const -> Tensor;

    /**
     * @brief Return a view with names set or cleared.
     * @param names nullopt to clear names, or a new name list
     */
    auto rename(std::optional<DimnameList> names) const -> Tensor;

    /**
     * @brief Find the index of a dimension by name.
     * @throws std::invalid_argument if name not found or tensor is unnamed
     */
    auto dim_index(std::string_view name) const -> int64_t;

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
     * WARNING: For GPU tensors, this implicitly copies data to CPU, which
     * synchronizes the device and blocks the host thread. Avoid calling
     * in performance-critical loops.
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
     * Returns this tensor (no copy) if already on the target device.
     *
     * @param device Target device
     * @return Tensor on specified device (may be this or new tensor)
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
     * @brief Move tensor to specified device and convert dtype in one operation.
     *
     * More efficient than chaining `.to(device).to(dtype)` — performs device
     * transfer first, then on-device cast (single intermediate allocation).
     *
     * @param device Target device
     * @param dtype Target data type
     * @return New tensor on target device with target dtype
     */
    auto to(Device device, DType dtype) const -> Tensor;

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
     * @return View — tensor with new shape (shares storage)
     * @throws std::runtime_error if tensor is not contiguous
     */
    auto view(std::vector<int64_t> new_shape) const -> Tensor;

    /**
     * @brief Transpose two dimensions.
     *
     * @param dim0 First dimension to swap
     * @param dim1 Second dimension to swap
     * @return View — transposed tensor (shares storage)
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
     * @return View — permuted tensor (shares storage)
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
     * @return View — tensor with size-1 dimensions removed
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
     * @return View — tensor with added dimension
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
     * @return View if contiguous, Copy otherwise — flattened tensor
     *
     * @code
     * Tensor t({2, 3, 4}, DType::Float32, Device::cpu());
     * Tensor f = t.flatten();         // Shape: {24}
     * Tensor f2 = t.flatten(1, 2);    // Shape: {2, 12}
     * @endcode
     */
    auto flatten(int64_t start_dim = 0, int64_t end_dim = -1) const -> Tensor;

    /**
     * @brief Expand tensor to a larger size by broadcasting.
     *
     * Uses -1 in shape to keep existing dimension size.
     * No data is copied — returns a view with broadcast strides.
     *
     * @param shape Target shape (must be compatible with broadcasting rules)
     * @return View — expanded tensor with broadcast strides
     */
    auto expand(std::vector<int64_t> shape) const -> Tensor;

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
     * @return View — sliced tensor (shares storage)
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
    auto operator+(double scalar) const -> Tensor;

    /**
     * @brief Subtract scalar from all elements.
     *
     * @param scalar Value to subtract
     * @return New tensor with scalar subtracted
     */
    auto operator-(double scalar) const -> Tensor;

    /**
     * @brief Multiply all elements by scalar.
     *
     * @param scalar Value to multiply by
     * @return New tensor with scalar multiplication
     */
    auto operator*(double scalar) const -> Tensor;

    /**
     * @brief Divide all elements by scalar.
     *
     * @param scalar Value to divide by
     * @return New tensor with scalar division
     */
    auto operator/(double scalar) const -> Tensor;

    // Reverse scalar operators (scalar op tensor)
    friend auto operator+(double s, const Tensor& t) -> Tensor;
    friend auto operator-(double s, const Tensor& t) -> Tensor;
    friend auto operator*(double s, const Tensor& t) -> Tensor;
    friend auto operator/(double s, const Tensor& t) -> Tensor;

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
    auto fill_(double value) -> Tensor&;

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
     * @return Copy — new independent tensor with copied data
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
     * @return View — detached tensor (shares storage, no gradients)
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
     * @return This tensor if already contiguous, Copy otherwise
     *
     * @code
     * Tensor t = some_tensor.transpose(0, 1);
     * Tensor c = t.contiguous();  // Ensures contiguous layout
     * @endcode
     */
    auto contiguous() const -> Tensor;

    // ============================================================================
    // Memory Format Support
    // ============================================================================

    /**
     * @brief Get the memory format of the tensor.
     *
     * Detects memory format from stride pattern. For 4D tensors:
     * - ChannelsLast if strides match NHWC pattern
     * - Contiguous otherwise
     *
     * @return Detected memory format
     *
     * @code
     * Tensor x = randn({1, 3, 224, 224}).to(MemoryFormat::ChannelsLast);
     * auto fmt = x.memory_format();  // MemoryFormat::ChannelsLast
     * @endcode
     */
    auto memory_format() const noexcept -> MemoryFormat;

    /**
     * @brief Check if tensor is contiguous in the specified memory format.
     *
     * @param format Memory format to check against
     * @return true if tensor's strides match the format's expected strides
     *
     * @code
     * Tensor x({1, 64, 32, 32}, DType::Float32, Device::cuda());
     * bool is_nchw = x.is_contiguous(MemoryFormat::Contiguous);    // true
     * bool is_nhwc = x.is_contiguous(MemoryFormat::ChannelsLast);  // false
     * @endcode
     */
    auto is_contiguous(MemoryFormat format) const noexcept -> bool;

    /**
     * @brief Convert tensor to specified memory format.
     *
     * Creates a new tensor with data rearranged to match the target format.
     * If tensor is already in the target format, returns a shallow copy.
     *
     * For 4D tensors:
     * - ChannelsLast: NCHW data → NHWC layout (same logical shape)
     * - Contiguous: NHWC data → NCHW layout
     *
     * @param format Target memory format
     * @return Tensor in the specified format
     *
     * @code
     * // Convert once at the start
     * Tensor x = randn({1, 3, 224, 224}).to(MemoryFormat::ChannelsLast);
     *
     * // Operations preserve format - no conversion overhead
     * Tensor y = conv2d(x, weight);  // Stays in NHWC
     * Tensor z = batchnorm(y);       // Still NHWC
     *
     * // Convert back if needed for interop
     * Tensor output = z.to(MemoryFormat::Contiguous);
     * @endcode
     *
     * @note This is the PyTorch-compatible approach. Using ChannelsLast
     *       on Tensor Core GPUs provides 30-100% Conv2D speedup.
     */
    auto to(MemoryFormat format) const -> Tensor;

    // ============================================================================
    // Convenience Accessors
    // ============================================================================

    /**
     * @brief Get the size of a specific dimension.
     *
     * @param dim Dimension index (supports negative indexing)
     * @return Size of the specified dimension
     * @throws std::out_of_range if dim is out of bounds
     *
     * @code
     * Tensor t({2, 3, 4}, DType::Float32, Device::cpu());
     * auto s = t.size(1);   // Returns 3
     * auto s2 = t.size(-1); // Returns 4
     * @endcode
     */
    auto size(int64_t dim) const -> int64_t {
        if (dim < 0) dim += ndim();
        if (dim < 0 || dim >= ndim())
            throw std::out_of_range("Dimension out of range (got " + std::to_string(dim) +
                                    " for tensor with " + std::to_string(ndim()) + " dimensions)");
        return shape()[dim];
    }

    /**
     * @brief Get the stride of a specific dimension.
     *
     * @param dim Dimension index (supports negative indexing)
     * @return Stride of the specified dimension
     * @throws std::out_of_range if dim is out of bounds
     *
     * @code
     * Tensor t({2, 3, 4}, DType::Float32, Device::cpu());
     * auto s = t.stride(1);   // Returns 4
     * auto s2 = t.stride(-1); // Returns 1
     * @endcode
     */
    auto stride(int64_t dim) const -> int64_t {
        if (dim < 0) dim += ndim();
        if (dim < 0 || dim >= ndim())
            throw std::out_of_range("Dimension out of range (got " + std::to_string(dim) +
                                    " for tensor with " + std::to_string(ndim()) + " dimensions)");
        return strides()[dim];
    }

    /**
     * @brief Get the number of dimensions (alias for ndim()).
     *
     * @return Number of dimensions (rank) of the tensor
     */
    auto dim() const -> int64_t { return ndim(); }

    /**
     * @brief Check if tensor has a floating-point data type.
     *
     * @return true for Float16, BFloat16, Float32, Float64
     */
    auto is_floating_point() const noexcept -> bool {
        auto dt = dtype();
        return dt == DType::Float16 || dt == DType::BFloat16 ||
               dt == DType::Float32 || dt == DType::Float64;
    }

    /**
     * @brief Check if tensor has a complex data type.
     *
     * @return true for Complex64, Complex128
     */
    auto is_complex() const noexcept -> bool {
        auto dt = dtype();
        return dt == DType::Complex64 || dt == DType::Complex128;
    }

    /**
     * @brief Check if tensor has a signed data type.
     *
     * @return true for all floating-point, complex, and signed integer types
     */
    auto is_signed() const noexcept -> bool {
        auto dt = dtype();
        return dt != DType::UInt8 && dt != DType::UInt16 &&
               dt != DType::UInt32 && dt != DType::UInt64 && dt != DType::Bool;
    }

    /**
     * @brief Get the size of each element in bytes (alias for dtype_size()).
     *
     * @return Number of bytes per element
     *
     * @code
     * Tensor t({3, 4}, DType::Float32, Device::cpu());
     * size_t bytes = t.element_size();  // Returns 4
     * @endcode
     */
    auto element_size() const noexcept -> size_t { return dtype_size(); }

    /**
     * @brief Narrow (slice) tensor along a dimension.
     * @param dim Dimension to narrow
     * @param start Start index
     * @param length Number of elements to keep
     * @return View — narrowed tensor (shares storage)
     */
    auto narrow(int64_t dim, int64_t start, int64_t length) const -> Tensor;

    /**
     * @brief Select a single index along a dimension, removing that dimension.
     * @param dim Dimension to select from
     * @param index Index to select
     * @return View — tensor with dim removed (shares storage)
     */
    auto select(int64_t dim, int64_t index) const -> Tensor;

    /**
     * @brief Split tensor into chunks along a dimension.
     * @param chunks Number of chunks
     * @param dim Dimension to split along (default: 0)
     * @return Views — vector of tensor chunk views (share storage)
     */
    auto chunk(int64_t chunks, int64_t dim = 0) const -> std::vector<Tensor>;

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

    // ============================================================================
    // Low-level Accessors (for backend kernel implementations)
    // ============================================================================

    /**
     * @brief Get the storage offset (in elements).
     */
    auto offset() const -> int64_t;

    /**
     * @brief Get the underlying storage.
     */
    auto storage() const -> const intrusive_ptr<Storage>&;

    /**
     * @brief Set the requires_grad flag.
     */
    auto set_requires_grad(bool requires_grad) -> void;

    /// @name Internal Mutation API (backend kernels only)
    /// @warning These methods are for backend kernel implementations creating
    /// view tensors (reshape, transpose, slice). Do NOT use in user-facing code.
    /// Misuse can corrupt tensor metadata and invalidate storage bounds checks.
    /// @{
#ifdef TENZOR_INTERNAL

    /// @brief Get mutable reference to shape vector (for view-creating kernels).
    auto mutable_shape() -> std::vector<int64_t>&;

    /// @brief Get mutable reference to strides vector (for view-creating kernels).
    auto mutable_strides() -> std::vector<int64_t>&;

    /// @brief Set the storage offset (in elements).
    auto set_offset(int64_t offset) -> void;

    /// @brief Invalidate the cached contiguity flag (call after modifying strides).
    auto invalidate_contiguity_cache() -> void;

#endif // TENZOR_INTERNAL
    /// @}

private:
    std::shared_ptr<TensorImpl> impl_;

    friend class Variable;
    friend class TensorAccessor;  // Single accessor for all backend kernel access to impl_
    friend class VulkanBackend;   // Vulkan backend needs direct access for buffer management
};

/**
 * @brief Privileged accessor for Tensor internals.
 *
 * Provides controlled access to Tensor::impl_ for backend kernels that need
 * to create view tensors (reshape, transpose, slice, etc.) by sharing storage.
 * This consolidates friend access into a single class instead of requiring
 * each kernel function to be individually friended.
 *
 * Backend accessor classes (CUDAKernelAccess, HIPKernelAccess) and CPU kernel
 * functions should use this class to access Tensor internals.
 */
class TensorAccessor {
public:
    static auto get_impl(const Tensor& t) -> const std::shared_ptr<TensorImpl>& {
        return t.impl_;
    }

    static auto get_impl_mutable(Tensor& t) -> std::shared_ptr<TensorImpl>& {
        return t.impl_;
    }
};

/**
 * @brief Internal tensor implementation (PImpl pattern).
 *
 * Manages the actual tensor data and metadata. Not intended for direct use.
 * Access tensors through the Tensor class interface.
 *
 * @note This class is an implementation detail and may change between versions.
 */
class TensorImpl : public IntrusiveRefCounted {
public:
    /**
     * @brief Construct tensor implementation.
     *
     * @param shape Tensor dimensions
     * @param dtype Element data type
     * @param device Device location
     * @param zero_init If true, zero-initialize memory (default: true)
     */
    TensorImpl(std::vector<int64_t> shape, DType dtype, Device device,
               bool zero_init = true);

    /**
     * @brief Construct from pre-existing storage (no allocation).
     *
     * Used by Tensor::from_blob() to wrap external memory.
     *
     * @param storage Shared storage to use
     * @param shape Tensor dimensions
     * @param strides Memory strides per dimension
     * @param dtype Element data type
     * @param device Device location
     */
    TensorImpl(intrusive_ptr<Storage> storage,
               std::vector<int64_t> shape,
               std::vector<int64_t> strides,
               DType dtype, Device device);

    /// Copy constructor (needed because std::atomic is not copyable)
    TensorImpl(const TensorImpl& other)
        : IntrusiveRefCounted(other)
        , storage(other.storage)
        , shape(other.shape)
        , strides(other.strides)
        , offset(other.offset)
        , dtype(other.dtype)
        , device(other.device)
        , requires_grad(other.requires_grad)
        , names_(other.names_)
        , is_contiguous_cache_(other.is_contiguous_cache_.load(std::memory_order_relaxed))
        , memory_format_cache_(other.memory_format_cache_.load(std::memory_order_relaxed))
        , version_counter_(other.version_counter_.load(std::memory_order_relaxed)) {}

    TensorImpl& operator=(const TensorImpl&) = delete;

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

private:
    friend class Tensor;

    intrusive_ptr<Storage> storage;    ///< Shared storage for tensor data
    std::vector<int64_t> shape;          ///< Tensor dimensions
    std::vector<int64_t> strides;        ///< Memory strides per dimension
    int64_t offset{0};                   ///< Offset into storage
    DType dtype;                         ///< Element data type
    Device device;                       ///< Device location
    bool requires_grad{false};           ///< Gradient computation flag
    std::optional<DimnameList> names_;   ///< Optional dimension names (experimental)
    // SAFETY: Stale reads are harmless (triggers recomputation). No ordering
    // dependency on other fields — relaxed load/store is sufficient.
    mutable std::atomic<int8_t> is_contiguous_cache_{-1};  ///< Cached contiguity: -1=unset, 0=false, 1=true
    // SAFETY: Same relaxed semantics as is_contiguous_cache_. Encoding matches
    // MemoryFormat enum: -1=unset, 0=Contiguous, 1=ChannelsLast, 2=ChannelsLast3d.
    static_assert(static_cast<int>(MemoryFormat::Contiguous) == 0
               && static_cast<int>(MemoryFormat::ChannelsLast) == 1
               && static_cast<int>(MemoryFormat::ChannelsLast3d) == 2,
                  "MemoryFormat enum values must match cache encoding");
    mutable std::atomic<int8_t> memory_format_cache_{-1};  ///< Cached memory format: -1=unset
    // SAFETY: Relaxed only in copy construction (snapshot of current version).
    // Mutation paths (increment_version()) use release ordering; version checks
    // in autograd use acquire ordering to observe the mutated tensor state.
    std::atomic<uint64_t> version_counter_{0};  ///< Mutation version for autograd in-place detection
};

} // namespace tenzor
