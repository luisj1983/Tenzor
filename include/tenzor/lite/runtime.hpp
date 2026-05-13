/**
 * @file runtime.hpp
 * @brief Inference-only runtime for ahead-of-time-compiled `.tzlite` models.
 *
 * The Lite runtime executes a serialised, optimised execution plan against
 * Tenzor's main per-backend kernel dispatch table. It does not maintain a
 * parallel kernel registry; an OpId-keyed `LiteNode` is dispatched through
 * the same code path as eager execution, so the runtime inherits whatever
 * backends (CPU, CUDA, ROCm, Vulkan, OneAPI) the host build supports.
 *
 * Features (cumulative across phases):
 *  - Phase 1: in-memory graphs of OpId nodes, executed via dispatch<OpId>.
 *  - Phase 2: TLV-sectioned `.tzlite` file with mmap'd SafeTensors weights
 *             and an AOT static memory plan (zero alloc during inference).
 *  - Phase 3: exporter that walks an nn::Module via the JIT tracer, runs
 *             graph-level optimisation passes, and emits `.tzlite`.
 *  - Phase 4: Python bindings (`tz.lite.export`, `tz.lite.Runtime`).
 *  - Phase 5: caller-selectable backend (CPU/CUDA/...) at load time,
 *             quantised int8 execution.
 *
 * Usage (post-Phase 3 / Phase 4):
 * @code
 *   tenzor::lite::export_to_tzlite(module, "model.tzlite", {.example_input_shapes={...}});
 *   auto runtime = LiteRuntime::load("model.tzlite");
 *   auto out = runtime->forward(input);
 * @endcode
 */

#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "../core/dtype.hpp"

namespace tenzor {
namespace lite {

// Forward declarations (defined in lite_graph.hpp). Kept here to avoid a
// header cycle (lite_graph.hpp #includes runtime.hpp).
class LiteGraph;

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

    LiteTensor() = default;
    ~LiteTensor();

    // Copy disabled — a LiteTensor with owns_data=true cannot safely share
    // its buffer with a copy. Callers that need a duplicate must allocate
    // explicitly via the API.
    LiteTensor(const LiteTensor&) = delete;
    auto operator=(const LiteTensor&) -> LiteTensor& = delete;

    // Move transfers ownership: source is zeroed so its destructor is a
    // no-op. Keeps put-into-vector / return-by-value safe when owns_data is
    // true.
    LiteTensor(LiteTensor&& other) noexcept
        : data(other.data),
          shape(other.shape),
          strides(other.strides),
          ndim(other.ndim),
          dtype(other.dtype),
          owns_data(other.owns_data)
    {
        other.data = nullptr;
        other.owns_data = false;
        other.ndim = 0;
    }
    auto operator=(LiteTensor&& other) noexcept -> LiteTensor& {
        if (this != &other) {
            if (owns_data && data) std::free(data);
            data       = other.data;
            shape      = other.shape;
            strides    = other.strides;
            ndim       = other.ndim;
            dtype      = other.dtype;
            owns_data  = other.owns_data;
            other.data = nullptr;
            other.owns_data = false;
            other.ndim = 0;
        }
        return *this;
    }

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

    /**
     * @brief Build a runtime around an in-memory graph (no file I/O).
     *
     * Useful for tests and for callers that produce a graph programmatically.
     * The graph is consumed by the runtime — caller's local variable is left
     * empty.
     */
    static auto from_graph(LiteGraph graph) -> std::unique_ptr<LiteRuntime>;

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
