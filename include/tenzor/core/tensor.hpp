/**
 * @file tensor.hpp
 * @brief Core tensor class for multi-dimensional array operations
 *
 * Provides the main Tensor class with support for multi-dimensional arrays,
 * automatic differentiation, device management, and various mathematical operations.
 * Uses PImpl pattern for efficient memory management and copy semantics.
 *
 * @par Thread Safety
 * Tensor objects are **not internally synchronized** for mutation. The following
 * rules apply:
 * - **Concurrent const reads with no writer** are safe. Multiple threads may
 *   call const methods (`shape()`, `strides()`, `dtype()`, `data<T>()`, etc.)
 *   on the same tensor as long as no other thread is mutating it.
 * - **Concurrent read + write to the same tensor is undefined behaviour**
 *   and requires external synchronization. `shape_` and `strides_` are stored
 *   as plain `std::vector<int64_t>`; readers racing against `resize_`, a
 *   view-producing op, or any in-place mutation can observe torn state.
 *   Affected operations include in-place arithmetic (`+=`, `fill_`), reshape,
 *   transpose/permute, and every shape-changing op.
 * - **Independent tensors** can be operated on concurrently without
 *   synchronization even when they share underlying `Storage`, because
 *   `Storage`'s reference count and mutability bits are atomic.
 * - **Cached derived state** — `is_contiguous_cache_`, `memory_format_cache_`,
 *   and `version_counter_` — is stored as `std::atomic`, so cache invalidation
 *   is race-free even while a writer updates shape/strides. This does *not*
 *   make shape/strides reads safe; it only means the cache itself cannot
 *   be left in an inconsistent half-updated state.
 * - **Autograd forward pass** is NOT thread-safe on shared Variables. Each
 *   thread should use its own Variable or synchronize access externally.
 * - **Autograd backward pass** is thread-safe for gradient accumulation when
 *   the Variable is marked with `make_thread_safe()` — that path uses an
 *   internal `grad_mutex_`.
 *
 * @note For async work that needs a stable shape/strides pair across
 * thread boundaries, call :func:`shape_info_snapshot` — it returns an
 * immutable ``std::shared_ptr<const ShapeInfo>`` that owns its own
 * copy of the shape/strides vectors and is unaffected by subsequent
 * mutations to the source tensor.
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
/**
 * @brief Immutable snapshot of a tensor's shape and strides.
 *
 * Produced by :func:`Tensor::shape_info_snapshot`. The snapshot is
 * independent of the Tensor it came from — once constructed, its
 * contents never change, so the caller can share it freely across
 * threads or capture it into an async kernel closure without worrying
 * about the source tensor being reshaped underneath them.
 *
 * Holding a snapshot alive is as cheap as bumping a shared_ptr
 * refcount. The snapshot owns its data; the source Tensor can be
 * destroyed without invalidating the snapshot.
 *
 * @note The snapshot captures only shape/strides/offset. It does NOT
 * reference the storage buffer. For a stable data-pointer view,
 * combine this with :func:`Tensor::storage` which itself is
 * intrusive-refcounted.
 */
struct ShapeInfo {
    std::vector<int64_t> shape;     ///< Tensor shape at capture time
    std::vector<int64_t> strides;   ///< Strides at capture time
    int64_t offset{0};              ///< Offset into storage at capture time
};

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

    // ---- View Tracking API ----

    /// Check if this tensor is a view of another tensor (shares storage via reshape/transpose/slice/etc.)
    auto is_view() const noexcept -> bool;

    /// Get the base tensor that this view was derived from (nullptr if not a view)
    auto _view_base() const noexcept -> TensorImpl*;

    /// Mark this tensor as a view of another tensor (called internally by view-creating ops)
    auto _set_view_base(intrusive_ptr<TensorImpl> base) noexcept -> void;

    // ---- Quantization API ----

    /// Check if this tensor has a quantized dtype
    auto is_quantized() const noexcept -> bool;

    /// Get quantization scale (only valid for quantized tensors)
    auto q_scale() const -> double;

    /// Get quantization zero point (only valid for quantized tensors)
    auto q_zero_point() const -> int64_t;

    /// Get the underlying integer representation of a quantized tensor
    auto int_repr() const -> Tensor;

    /// Dequantize: convert quantized tensor back to Float32
    auto dequantize() const -> Tensor;

    /// Set quantization parameters (only for quantized tensors)
    auto set_quantization_params(double scale, int64_t zero_point) -> void;

    // ---- Per-channel quantization API ----

    /// Check if this tensor uses per-channel quantization
    auto is_per_channel_quantized() const noexcept -> bool;

    /// Get per-channel quantization scales (only valid for per-channel quantized tensors)
    auto q_per_channel_scales() const -> const std::vector<double>&;

    /// Get per-channel quantization zero points
    auto q_per_channel_zero_points() const -> const std::vector<int64_t>&;

    /// Get per-channel quantization axis
    auto q_per_channel_axis() const -> int64_t;

    /// Set per-channel quantization parameters
    auto set_per_channel_quantization_params(
        std::vector<double> scales,
        std::vector<int64_t> zero_points,
        int64_t axis) -> void;

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
    auto impl() const -> const intrusive_ptr<TensorImpl>& { return impl_; }

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
     * @brief Whether this tensor's underlying CPU storage is page-locked.
     *
     * Returns true only for CPU tensors whose storage has been pinned
     * via `pin_memory()` (or allocated pinned from the start). GPU
     * tensors always return false — the pinned-memory concept only
     * applies to host allocations that CUDA DMA can target directly.
     */
    auto is_pinned() const -> bool;

    /**
     * @brief Page-lock this tensor's CPU storage for fast GPU transfers.
     *
     * Delegates to Storage::pin(), which calls `cudaHostRegister` on
     * the underlying buffer when CUDA is available. On GPU tensors,
     * on non-CUDA builds, or if the buffer cannot be registered, this
     * is a silent no-op and the tensor is returned unchanged.
     *
     * Unlike `tensor.to(Device::cpu())`, this does not allocate a new
     * buffer — the pin is in-place, shared across every view that
     * aliases the same storage. The destructor automatically calls
     * `cudaHostUnregister` when the last reference drops.
     *
     * @return Reference to self for chaining.
     */
    auto pin_memory() -> Tensor&;

    /**
     * @brief Return an immutable snapshot of this tensor's shape,
     *        strides, and storage offset.
     *
     * The returned :class:`ShapeInfo` is a standalone shared_ptr that
     * owns its own copies of the shape/strides vectors and is safe to
     * share across threads or capture into async-kernel closures:
     * mutating the source Tensor (reshape, resize, transpose in place,
     * etc.) does not affect snapshots taken before the mutation.
     *
     * Under the hood, TensorImpl keeps a lazy cache: the first call
     * constructs a new ShapeInfo from the current state and stores it
     * atomically; subsequent calls return the cached pointer as long
     * as no mutator has invalidated it. Mutators
     * (`mutable_shape`, `mutable_strides`, `set_offset`) reset the
     * cache so the next snapshot rebuilds.
     *
     * @note The snapshot captures a stable shape/strides pair for the
     * caller's subsequent use, but TensorImpl's internal mutators are
     * still lock-free; if another thread races a mutation against a
     * snapshot-building read, the caller is responsible for external
     * synchronization (same contract as `.shape()` itself).
     *
     * @return shared_ptr to an immutable ShapeInfo, or nullptr if the
     *         tensor is uninitialized.
     */
    auto shape_info_snapshot() const -> std::shared_ptr<const ShapeInfo>;

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
    intrusive_ptr<TensorImpl> impl_;

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
    static auto get_impl(const Tensor& t) -> const intrusive_ptr<TensorImpl>& {
        return t.impl_;
    }

    static auto get_impl_mutable(Tensor& t) -> intrusive_ptr<TensorImpl>& {
        return t.impl_;
    }
};

/**
 * @brief Conservatively detect whether two tensors may alias the same memory.
 *
 * Used by dispatch_inplace() to catch bugs where an in-place op's target
 * overlaps with one of its input tensors (for example `a.copy_(a.view(...))`
 * where reads from the source would stomp on writes to the target).
 *
 * Semantics:
 *   - Two references to the exact same Tensor object (same TensorImpl) are
 *     treated as NON-aliasing. This is the `x.add_(x)` case, which most
 *     element-wise in-place kernels handle correctly.
 *   - Different storage objects are never aliasing.
 *   - Same storage with overlapping byte ranges returns true. This
 *     overestimates aliasing for strided views (the real touched bytes may
 *     be sparser than the [offset, offset+numel*dtype_size) span), which is
 *     safe: false positives can be silenced with
 *     `AttrKey::IgnoreAliasCheck`, but false negatives cause data corruption.
 *
 * @param a First tensor
 * @param b Second tensor
 * @return true if the two tensors may alias each other in a way that an
 *         in-place kernel could corrupt
 */
auto may_alias(const Tensor& a, const Tensor& b) -> bool;

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
        , version_counter_(other.version_counter_.load(std::memory_order_relaxed))
        , shape_info_cache_(std::atomic_load(&other.shape_info_cache_)) {}

    TensorImpl& operator=(const TensorImpl&) = delete;

    /// Get the mutation version counter
    auto version() const noexcept -> uint64_t {
        return version_counter_.load(std::memory_order_acquire);
    }

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
    intrusive_ptr<TensorImpl> view_base_;    ///< Owning pointer to base tensor for views (prevents UAF)

    // Quantization metadata (only valid when dtype is QInt8/QUInt8/QInt4x2)
    double q_scale_{0.0};               ///< Quantization scale factor
    int64_t q_zero_point_{0};           ///< Quantization zero point
    std::optional<std::vector<double>> q_scales_;       ///< Per-channel quantization scales
    std::optional<std::vector<int64_t>> q_zero_points_; ///< Per-channel quantization zero points
    int64_t q_axis_{-1};                                 ///< Quantization axis (-1 = per-tensor)
    // SAFETY: Uses acquire/release ordering to synchronize with version_counter_
    // updates, ensuring autograd sees consistent contiguity state after mutations.
    mutable std::atomic<int8_t> is_contiguous_cache_{-1};  ///< Cached contiguity: -1=unset, 0=false, 1=true
    // SAFETY: Same acquire/release semantics as is_contiguous_cache_. Encoding matches
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

    /// Lazy, invalidation-on-mutation cache of an immutable ShapeInfo
    /// snapshot. Produced on first call to Tensor::shape_info_snapshot()
    /// and reset to nullptr by mutable_shape/strides/set_offset. Uses
    /// plain std::shared_ptr with std::atomic_load/store free functions
    /// for portability — std::atomic<std::shared_ptr<T>> (C++20) works
    /// but the free functions are still supported and avoid gcc
    /// std::atomic<shared_ptr> initializer-list quirks.
    mutable std::shared_ptr<const ShapeInfo> shape_info_cache_;
};

/**
 * @brief Quantize a float tensor to a quantized dtype.
 *
 * Converts float values to quantized integers using:
 *   q_val = clamp(round(float_val / scale) + zero_point, qmin, qmax)
 *
 * @param input Float tensor to quantize
 * @param scale Quantization scale factor
 * @param zero_point Quantization zero point
 * @param dtype Target quantized dtype (QInt8, QUInt8)
 * @return Quantized tensor with scale and zero_point metadata
 */
auto quantize_per_tensor(const Tensor& input, double scale, int64_t zero_point,
                         DType dtype = DType::QInt8) -> Tensor;

} // namespace tenzor
