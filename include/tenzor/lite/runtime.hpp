/**
 * @file runtime.hpp
 * @brief Lightweight inference-only runtime for mobile/embedded deployment
 *
 * The lite runtime is a minimal C++ library for executing optimized
 * models on resource-constrained devices. Features:
 * - No autograd, no Python, minimal dependencies
 * - Static memory planning (zero allocation during inference)
 * - ARM NEON / x86 SSE/AVX kernel dispatch
 * - INT8 quantized execution
 * - TZLITE model format with embedded weights
 *
 * Target binary size: <5MB (stripped, static, ARM64)
 *
 * Usage:
 * @code
 * auto runtime = LiteRuntime::load("model.tzlite");
 * LiteTensor input = runtime->create_input({1, 3, 224, 224});
 * // ... fill input data ...
 * LiteTensor output = runtime->forward(input);
 * @endcode
 */

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "../core/dtype.hpp"

namespace tenzor {
namespace lite {

// ============================================================================
// LiteTensor: minimal tensor with no heap allocation for metadata
// ============================================================================

/**
 * @brief Maximum number of dimensions supported.
 */
constexpr int kMaxDims = 8;

/**
 * @brief Lightweight tensor for inference.
 *
 * Flat struct with no reference counting, no copy-on-write.
 * Shape/strides use fixed arrays to avoid heap allocation.
 * Data pointer typically points into the arena allocator.
 */
struct LiteTensor {
    void* data{nullptr};                       ///< Raw data pointer (into arena or owned)
    std::array<int64_t, kMaxDims> shape{};     ///< Dimensions
    std::array<int64_t, kMaxDims> strides{};   ///< Strides in elements
    int32_t ndim{0};                           ///< Number of dimensions
    DType dtype{DType::Float32};               ///< Data type
    bool owns_data{false};                     ///< True if this tensor owns its data

    ~LiteTensor();

    /// Total number of elements.
    auto numel() const -> int64_t;

    /// Size in bytes.
    auto nbytes() const -> int64_t;

    /// Typed data access.
    template<typename T>
    auto data_as() -> T* { return static_cast<T*>(data); }

    template<typename T>
    auto data_as() const -> const T* { return static_cast<const T*>(data); }
};

// ============================================================================
// LiteAllocator: arena-based static allocator
// ============================================================================

/**
 * @brief Arena allocator for zero-allocation inference.
 *
 * Pre-allocates memory pools based on the model's MemoryPlan.
 * During inference, get_buffer() returns pointers into these
 * pre-allocated pools with zero overhead.
 */
class LiteAllocator {
public:
    /**
     * @brief Initialize with pool sizes from the memory plan.
     *
     * @param pool_sizes Sizes of each memory pool in bytes
     * @param alignment Byte alignment (16 for NEON, 64 for AVX)
     */
    explicit LiteAllocator(const std::vector<size_t>& pool_sizes,
                           size_t alignment = 64);
    ~LiteAllocator();

    LiteAllocator(const LiteAllocator&) = delete;
    auto operator=(const LiteAllocator&) -> LiteAllocator& = delete;

    /**
     * @brief Get a buffer pointer for a value.
     *
     * @param buffer_id Pool index
     * @param offset Byte offset within the pool
     * @return Raw pointer to the buffer location
     */
    auto get_buffer(size_t buffer_id, size_t offset) -> void*;

    /**
     * @brief Total allocated memory across all pools.
     */
    auto total_bytes() const -> size_t { return total_bytes_; }

private:
    std::vector<void*> pools_;
    std::vector<size_t> pool_sizes_;
    size_t total_bytes_{0};
    size_t alignment_;
};

// ============================================================================
// LiteRuntime: main public API
// ============================================================================

/**
 * @brief The single public API class for mobile inference.
 *
 * Loads a TZLITE model, manages memory, and executes the graph.
 * Thread-safe for concurrent calls to forward() (each call
 * uses its own set of intermediate buffers if the allocator
 * supports it, or serializes otherwise).
 */
class LiteRuntime {
public:
    /**
     * @brief Load a model from a TZLITE file.
     *
     * @param path Path to the .tzlite model file
     * @return Runtime instance ready for inference
     */
    static auto load(const std::string& path) -> std::unique_ptr<LiteRuntime>;

    /**
     * @brief Load a model from a memory buffer.
     *
     * Useful for Android assets or embedded resources.
     *
     * @param data Pointer to model data
     * @param size Size of model data in bytes
     * @return Runtime instance ready for inference
     */
    static auto load(const void* data, size_t size) -> std::unique_ptr<LiteRuntime>;

    ~LiteRuntime();

    /**
     * @brief Run inference with a single input.
     *
     * @param input Input tensor
     * @return Output tensor
     */
    auto forward(const LiteTensor& input) -> LiteTensor;

    /**
     * @brief Run inference with multiple inputs.
     *
     * @param inputs Input tensors
     * @return Output tensors
     */
    auto forward(const std::vector<LiteTensor>& inputs) -> std::vector<LiteTensor>;

    /**
     * @brief Get expected input shapes.
     */
    auto input_shapes() const -> std::vector<std::vector<int64_t>>;

    /**
     * @brief Get expected output shapes.
     */
    auto output_shapes() const -> std::vector<std::vector<int64_t>>;

    /**
     * @brief Get model metadata.
     *
     * @param key Metadata key
     * @return Value, or empty string if not found
     */
    auto model_metadata(const std::string& key) const -> std::string;

    /**
     * @brief Create an input tensor with the expected shape.
     *
     * Allocates memory for the input. Caller fills data.
     *
     * @param shape Input shape (must match model's expected input)
     * @param dtype Data type (default: Float32)
     * @return Allocated input tensor
     */
    auto create_input(const std::vector<int64_t>& shape,
                      DType dtype = DType::Float32) -> LiteTensor;

private:
    LiteRuntime() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lite
} // namespace tenzor
