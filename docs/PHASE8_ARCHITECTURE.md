# Phase 8 Architecture: Advanced Training Features

**Version:** 1.0.0
**Date:** 2025-10-13
**Status:** Design Complete
**Author:** Tenzor Architecture Team

## Executive Summary

Phase 8 introduces production-grade deep learning infrastructure for Tenzor, focusing on four critical systems:

1. **Mixed Precision Training** - Float16/BFloat16 support with automatic loss scaling
2. **Memory Management** - Caching allocator with fragmentation reduction
3. **DataLoader** - Multi-threaded data loading with prefetching
4. **Multi-GPU** - DataParallel for distributed training

This architecture integrates seamlessly with existing systems while maintaining backward compatibility.

---

## Table of Contents

1. [Mixed Precision Training Architecture](#1-mixed-precision-training-architecture)
2. [Memory Management Architecture](#2-memory-management-architecture)
3. [DataLoader Architecture](#3-dataloader-architecture)
4. [Multi-GPU Architecture](#4-multi-gpu-architecture)
5. [Integration Points](#5-integration-points)
6. [Thread Safety Considerations](#6-thread-safety-considerations)
7. [Performance Targets](#7-performance-targets)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Mixed Precision Training Architecture

### 1.1 Overview

Mixed precision training uses Float16/BFloat16 for forward/backward passes and Float32 for optimizer updates, providing:
- **2-3x speedup** on modern GPUs (Tensor Cores)
- **~50% memory reduction** for activations and gradients
- **Maintained numerical stability** via loss scaling

### 1.2 Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Mixed Precision System                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────┐    ┌─────────────────┐                  │
│  │   DType        │───▶│  Float16Traits  │                  │
│  │  Extensions    │    │  BFloat16Traits │                  │
│  └────────────────┘    └─────────────────┘                  │
│          │                      │                            │
│          ▼                      ▼                            │
│  ┌──────────────────────────────────────┐                   │
│  │         Autocast Context             │                   │
│  │  - Thread-local state                │                   │
│  │  - Dtype policy (FP16/BF16)          │                   │
│  │  - Operation whitelist/blacklist     │                   │
│  └──────────────────────────────────────┘                   │
│          │                                                    │
│          ▼                                                    │
│  ┌──────────────────────────────────────┐                   │
│  │         GradScaler                   │                   │
│  │  - Loss scaling factor               │                   │
│  │  - Dynamic scale adjustment          │                   │
│  │  - Overflow/Underflow detection      │                   │
│  │  - Master weight management          │                   │
│  └──────────────────────────────────────┘                   │
│          │                                                    │
│          ▼                                                    │
│  ┌──────────────────────────────────────┐                   │
│  │    Autograd Integration              │                   │
│  │  - Automatic dtype casting           │                   │
│  │  - Gradient scaling in backward      │                   │
│  │  - Unscaling before optimizer        │                   │
│  └──────────────────────────────────────┘                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Class Hierarchy

```cpp
// include/tenzor/core/dtype.hpp

/**
 * @brief Extended DType enumeration with half-precision types.
 *
 * Extends existing DType enum to include:
 * - Float16: IEEE 754 half-precision (1 sign, 5 exp, 10 mantissa)
 * - BFloat16: Google Brain float (1 sign, 8 exp, 7 mantissa)
 */
enum class DType : uint8_t {
    // ... existing types ...
    Float16,    ///< IEEE 754 half-precision (16-bit)
    BFloat16,   ///< Brain floating point (16-bit)
};

/**
 * @brief Traits for Float16 type.
 *
 * Provides conversions between Float16 and Float32.
 * Uses bit manipulation for IEEE 754 compliance.
 */
struct Float16 {
    uint16_t bits;

    // Constructors
    Float16() = default;
    explicit Float16(float f);

    // Conversions
    explicit operator float() const;

    // Comparison operators
    auto operator==(Float16 other) const -> bool;
    auto operator!=(Float16 other) const -> bool;
    auto operator<(Float16 other) const -> bool;
    auto operator>(Float16 other) const -> bool;

    // Arithmetic (promoted to float)
    auto operator+(Float16 other) const -> Float16;
    auto operator-(Float16 other) const -> Float16;
    auto operator*(Float16 other) const -> Float16;
    auto operator/(Float16 other) const -> Float16;

private:
    // IEEE 754 conversion helpers
    static auto float_to_bits(float f) -> uint16_t;
    static auto bits_to_float(uint16_t bits) -> float;
};

/**
 * @brief Traits for BFloat16 type.
 *
 * Brain Float16 uses same exponent range as Float32 (8 bits)
 * but reduced mantissa (7 bits), providing better dynamic range
 * than Float16 at cost of precision.
 */
struct BFloat16 {
    uint16_t bits;

    // Constructors
    BFloat16() = default;
    explicit BFloat16(float f);

    // Conversions
    explicit operator float() const;

    // Comparison operators
    auto operator==(BFloat16 other) const -> bool;
    auto operator!=(BFloat16 other) const -> bool;

    // Arithmetic (promoted to float)
    auto operator+(BFloat16 other) const -> BFloat16;
    auto operator-(BFloat16 other) const -> BFloat16;
    auto operator*(BFloat16 other) const -> BFloat16;
    auto operator/(BFloat16 other) const -> BFloat16;

private:
    // Simple truncation for BFloat16
    static auto float_to_bfloat(float f) -> uint16_t;
    static auto bfloat_to_float(uint16_t bits) -> float;
};

// Type traits extensions
template<> struct dtype_traits<DType::Float16> { using type = Float16; };
template<> struct dtype_traits<DType::BFloat16> { using type = BFloat16; };
```

### 1.4 Autocast Mechanism

```cpp
// include/tenzor/autograd/autocast.hpp

namespace tenzor {
namespace autograd {

/**
 * @brief Autocast policy enumeration.
 *
 * Defines which precision mode to use for automatic casting.
 */
enum class AutocastMode {
    Disabled,   ///< No automatic casting
    Float16,    ///< Cast to Float16
    BFloat16    ///< Cast to BFloat16
};

/**
 * @brief Thread-local autocast context.
 *
 * Manages automatic dtype casting for operations within scope.
 * Uses RAII pattern for scope-based behavior.
 *
 * Design Pattern: RAII, Thread-Local Storage
 *
 * @code
 * // Enable autocast for forward pass
 * {
 *     AutocastContext ctx(AutocastMode::Float16);
 *     Variable output = model.forward(input);  // Uses FP16
 * }
 * // Autocast disabled outside scope
 * @endcode
 */
class AutocastContext {
public:
    /**
     * @brief Construct and enable autocast context.
     *
     * @param mode Precision mode (Float16/BFloat16)
     * @param enabled Whether to enable (default: true)
     */
    explicit AutocastContext(AutocastMode mode, bool enabled = true);

    /**
     * @brief Destructor restores previous context.
     */
    ~AutocastContext();

    // Non-copyable, non-movable
    AutocastContext(const AutocastContext&) = delete;
    AutocastContext& operator=(const AutocastContext&) = delete;

    /**
     * @brief Check if autocast is currently enabled.
     */
    static auto is_enabled() -> bool;

    /**
     * @brief Get current autocast mode.
     */
    static auto get_mode() -> AutocastMode;

    /**
     * @brief Cast tensor to autocast dtype if enabled.
     *
     * @param tensor Input tensor
     * @param op_name Operation name for whitelist/blacklist
     * @return Casted tensor or original if autocast disabled
     */
    static auto cast_if_enabled(const Tensor& tensor,
                                const std::string& op_name) -> Tensor;

private:
    AutocastMode prev_mode_;
    bool prev_enabled_;

    // Thread-local state
    static thread_local AutocastMode current_mode_;
    static thread_local bool enabled_;

    // Operation categorization
    static auto should_cast(const std::string& op_name) -> bool;

    // Whitelist: operations safe in low precision
    static const std::unordered_set<std::string> whitelist_;

    // Blacklist: operations requiring high precision
    static const std::unordered_set<std::string> blacklist_;
};

/**
 * @brief RAII helper for autocast scope.
 *
 * @code
 * auto autocast = make_autocast(AutocastMode::Float16);
 * Variable y = model.forward(x);  // Automatic FP16 casting
 * @endcode
 */
auto make_autocast(AutocastMode mode) -> std::unique_ptr<AutocastContext>;

} // namespace autograd
} // namespace tenzor
```

### 1.5 Gradient Scaler

```cpp
// include/tenzor/autograd/grad_scaler.hpp

namespace tenzor {
namespace autograd {

/**
 * @brief Gradient scaler for mixed precision training.
 *
 * Implements loss scaling to prevent gradient underflow in Float16/BFloat16.
 * Dynamically adjusts scale factor based on overflow detection.
 *
 * Algorithm:
 * 1. Scale loss before backward: loss_scaled = loss * scale_factor
 * 2. Backward pass computes: gradients_scaled = gradients * scale_factor
 * 3. Unscale gradients: gradients = gradients_scaled / scale_factor
 * 4. Check for overflow/underflow
 * 5. Adjust scale_factor dynamically
 * 6. Skip optimizer step if overflow detected
 *
 * Design Pattern: Strategy Pattern for scaling policies
 * Thread Safety: NOT thread-safe, use one per optimizer
 *
 * @code
 * GradScaler scaler(2048.0f);  // Initial scale
 *
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     Variable loss = criterion(output, target);
 *
 *     // Scale loss and backward
 *     scaler.scale(loss).backward();
 *
 *     // Unscale gradients and step
 *     scaler.step(optimizer);
 *     scaler.update();
 * }
 * @endcode
 */
class GradScaler {
public:
    /**
     * @brief Construct gradient scaler.
     *
     * @param init_scale Initial scale factor (default: 65536.0 = 2^16)
     * @param growth_factor Scale growth rate (default: 2.0)
     * @param backoff_factor Scale reduction rate (default: 0.5)
     * @param growth_interval Steps before growing scale (default: 2000)
     * @param enabled Enable scaling (default: true)
     */
    explicit GradScaler(
        float init_scale = 65536.0f,
        float growth_factor = 2.0f,
        float backoff_factor = 0.5f,
        int growth_interval = 2000,
        bool enabled = true
    );

    /**
     * @brief Scale loss tensor before backward.
     *
     * @param loss Unscaled loss
     * @return Scaled loss (loss * scale_factor)
     */
    auto scale(const Variable& loss) -> Variable;

    /**
     * @brief Unscale gradients and perform optimizer step.
     *
     * Steps:
     * 1. Unscale all parameter gradients
     * 2. Check for overflow/inf/nan
     * 3. Call optimizer.step() if no overflow
     * 4. Update internal state
     *
     * @param optimizer Optimizer to step
     * @return true if step was performed, false if skipped
     */
    auto step(Optimizer& optimizer) -> bool;

    /**
     * @brief Update scale factor based on overflow history.
     *
     * Call after optimizer.step() to adjust scale factor:
     * - Decrease scale if overflow detected
     * - Increase scale if N consecutive successful steps
     */
    auto update() -> void;

    /**
     * @brief Unscale gradients manually.
     *
     * Divides all gradients by current scale factor.
     *
     * @param parameters Parameters to unscale
     */
    auto unscale_(std::vector<Variable*> parameters) -> void;

    /**
     * @brief Get current scale factor.
     */
    auto get_scale() const -> float { return scale_; }

    /**
     * @brief Check if overflow was detected in last step.
     */
    auto has_overflow() const -> bool { return found_inf_; }

    /**
     * @brief Get growth counter.
     */
    auto get_growth_tracker() const -> int { return growth_tracker_; }

    /**
     * @brief Reset scaler state.
     */
    auto reset() -> void;

    /**
     * @brief Get state for serialization.
     */
    auto state_dict() const -> std::unordered_map<std::string, float>;

    /**
     * @brief Load state from dictionary.
     */
    auto load_state_dict(const std::unordered_map<std::string, float>& state) -> void;

private:
    float scale_;              ///< Current scale factor
    float growth_factor_;      ///< Multiplicative growth rate
    float backoff_factor_;     ///< Multiplicative reduction rate
    int growth_interval_;      ///< Steps before attempting growth
    int growth_tracker_;       ///< Counter for successful steps
    bool enabled_;             ///< Enable/disable scaling
    bool found_inf_;           ///< Overflow detected in last step

    /**
     * @brief Check tensors for inf/nan.
     *
     * @param tensors Tensors to check
     * @return true if any inf/nan found
     */
    auto check_overflow(const std::vector<Tensor>& tensors) -> bool;

    /**
     * @brief Grow scale factor.
     */
    auto grow_scale() -> void;

    /**
     * @brief Reduce scale factor.
     */
    auto backoff_scale() -> void;
};

} // namespace autograd
} // namespace tenzor
```

### 1.6 Autograd Integration

```cpp
// Modifications to include/tenzor/autograd/function.hpp

namespace tenzor {

/**
 * @brief Extended Function base class with autocast support.
 *
 * Automatic dtype casting is applied in forward pass based on
 * thread-local autocast context.
 */
class Function : public std::enable_shared_from_this<Function> {
public:
    // ... existing interface ...

protected:
    /**
     * @brief Apply autocast to input tensors.
     *
     * Automatically called at the start of forward pass.
     * Casts tensors to FP16/BF16 if autocast is enabled.
     *
     * @param inputs Input variables
     * @return Casted inputs if autocast enabled
     */
    auto apply_autocast(std::vector<Variable> inputs) const
        -> std::vector<Variable>;

    /**
     * @brief Get operation name for autocast whitelist.
     */
    virtual auto operation_name() const -> std::string = 0;
};

} // namespace tenzor
```

### 1.7 Mixed Precision Training Workflow

```cpp
// Example usage pattern

#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/autocast.hpp>
#include <tenzor/autograd/grad_scaler.hpp>

using namespace tenzor;
using namespace tenzor::autograd;

// Training loop with mixed precision
void train_with_amp(
    nn::Module& model,
    optim::Optimizer& optimizer,
    DataLoader& train_loader,
    nn::Loss& criterion,
    int num_epochs
) {
    // Create gradient scaler
    GradScaler scaler(65536.0f);  // Initial scale = 2^16

    model.train();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (auto& batch : train_loader) {
            auto [inputs, targets] = batch;

            optimizer.zero_grad();

            // Forward pass with autocast
            Variable output;
            {
                AutocastContext ctx(AutocastMode::Float16);
                output = model.forward(inputs);
            }

            // Compute loss (in Float32 for numerical stability)
            Variable loss = criterion(output, targets);

            // Scale loss and backward
            scaler.scale(loss).backward();

            // Unscale gradients, check overflow, and step
            bool stepped = scaler.step(optimizer);

            // Update scale factor
            scaler.update();

            if (!stepped) {
                std::cout << "Overflow detected, skipping step\n";
            }
        }
    }
}
```

### 1.8 Backend Kernel Support

```cpp
// Modifications to backend/dispatch.hpp

namespace tenzor {

/**
 * @brief Dispatcher handles autocast-aware operation routing.
 *
 * Automatically casts inputs based on autocast context before
 * dispatching to backend kernels.
 */
class Dispatcher {
public:
    /**
     * @brief Dispatch operation with autocast support.
     *
     * @param op_name Operation identifier
     * @param inputs Input tensors (may be cast)
     * @param attrs Operation attributes
     * @return Output tensors
     */
    static auto dispatch(const std::string& op_name,
                        std::span<const Tensor> inputs,
                        const OpAttributes& attrs = {})
        -> std::vector<Tensor>;

private:
    /**
     * @brief Apply autocast before dispatch.
     */
    static auto apply_autocast(const std::string& op_name,
                               std::vector<Tensor> inputs)
        -> std::vector<Tensor>;
};

} // namespace tenzor
```

### 1.9 CUDA Kernel Integration

```cuda
// src/backends/cuda/kernels/fp16_kernels.cu

namespace tenzor {
namespace cuda {

/**
 * @brief CUDA kernels optimized for Float16/BFloat16.
 *
 * Uses Tensor Cores (compute capability >= 7.0) for maximum performance.
 * Falls back to CUDA Cores on older architectures.
 */

// Matrix multiplication using Tensor Cores
template<typename T>  // T = __half or __nv_bfloat16
__global__ void matmul_fp16_kernel(
    const T* A,
    const T* B,
    T* C,
    int M, int N, int K
) {
    // Use wmma (Warp Matrix Multiply-Accumulate) for Tensor Cores
    // ... implementation using nvcuda::wmma API ...
}

// Element-wise operations (add, mul, etc.)
template<typename T>
__global__ void elementwise_fp16_kernel(
    const T* input,
    T* output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Vector loads for coalesced access
        // ... implementation ...
    }
}

} // namespace cuda
} // namespace tenzor
```

---

## 2. Memory Management Architecture

### 2.1 Overview

Efficient GPU memory management is critical for deep learning:
- **Problem**: cudaMalloc/cudaFree are expensive (~10μs per call)
- **Solution**: Caching allocator pools and reuses memory blocks
- **Benefits**: 10-100x faster allocation, reduced fragmentation

### 2.2 Component Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Memory Management System                    │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              CachingAllocator                            │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │ │
│  │  │ Free Blocks  │  │ Active Blocks│  │ Stream Order │  │ │
│  │  │   (size)     │  │   (ptr->md)  │  │  Tracking    │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
│          │                   │                   │            │
│          ▼                   ▼                   ▼            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Memory Pools │  │ Block Splits │  │ Coalescing   │      │
│  │  Per Device  │  │  & Merging   │  │  Algorithm   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│          │                                                    │
│          ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Memory Statistics                           │ │
│  │  - Allocated bytes                                       │ │
│  │  - Cached bytes                                          │ │
│  │  - Peak memory usage                                     │ │
│  │  - Fragmentation metrics                                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 Caching Allocator Design

```cpp
// include/tenzor/core/allocator.hpp

namespace tenzor {

/**
 * @brief Memory block metadata.
 *
 * Tracks allocation state, size, and stream association.
 */
struct Block {
    void* ptr;              ///< Device pointer
    size_t size;            ///< Block size in bytes
    bool is_free;           ///< Free/allocated state
    StreamHandle stream;    ///< Associated stream (nullptr = default)
    int device_id;          ///< Device index

    // Linked list for adjacent blocks
    Block* prev;
    Block* next;

    Block(void* ptr, size_t size, int device_id)
        : ptr(ptr), size(size), is_free(true),
          stream(nullptr), device_id(device_id),
          prev(nullptr), next(nullptr) {}
};

/**
 * @brief Memory pool for a single device.
 *
 * Manages free and allocated blocks using best-fit allocation strategy.
 * Implements block splitting and coalescing for fragmentation reduction.
 *
 * Design Pattern: Object Pool
 * Thread Safety: Protected by mutex
 *
 * Allocation Strategy:
 * 1. Round up size to power-of-2 or size class
 * 2. Search free blocks for best fit
 * 3. Split block if much larger than needed
 * 4. If no suitable block, allocate from device
 *
 * Deallocation Strategy:
 * 1. Mark block as free
 * 2. Coalesce with adjacent free blocks
 * 3. Return to appropriate size class pool
 */
class DeviceMemoryPool {
public:
    /**
     * @brief Construct memory pool for device.
     *
     * @param device_id Device index
     * @param backend Backend for actual allocations
     */
    DeviceMemoryPool(int device_id, Backend* backend);

    /**
     * @brief Destructor frees all cached memory.
     */
    ~DeviceMemoryPool();

    /**
     * @brief Allocate memory block.
     *
     * @param size Requested size in bytes
     * @param stream Associated stream (nullptr = default)
     * @return Pointer to allocated memory
     * @throws std::bad_alloc if allocation fails
     */
    auto allocate(size_t size, StreamHandle stream = nullptr) -> void*;

    /**
     * @brief Free memory block (return to pool).
     *
     * @param ptr Pointer to free
     */
    auto free(void* ptr) -> void;

    /**
     * @brief Free all cached memory.
     *
     * Returns all cached blocks to device, keeping only active allocations.
     */
    auto empty_cache() -> void;

    /**
     * @brief Synchronize stream and release associated blocks.
     *
     * @param stream Stream to synchronize
     */
    auto synchronize_stream(StreamHandle stream) -> void;

    /**
     * @brief Get memory statistics.
     */
    auto get_stats() const -> MemoryStats;

private:
    int device_id_;
    Backend* backend_;

    // Free blocks organized by size class
    std::map<size_t, std::vector<Block*>> free_blocks_;

    // Active allocations
    std::unordered_map<void*, Block*> active_blocks_;

    // All blocks (for tracking)
    std::vector<std::unique_ptr<Block>> all_blocks_;

    // Statistics
    size_t allocated_bytes_{0};
    size_t cached_bytes_{0};
    size_t peak_allocated_{0};

    // Thread safety
    mutable std::mutex mutex_;

    /**
     * @brief Find best-fit free block.
     *
     * @param size Requested size
     * @return Block pointer or nullptr if not found
     */
    auto find_free_block(size_t size) -> Block*;

    /**
     * @brief Allocate new block from device.
     *
     * @param size Block size
     * @return New block
     */
    auto allocate_block(size_t size) -> Block*;

    /**
     * @brief Split block if much larger than needed.
     *
     * @param block Block to split
     * @param size Desired size
     * @return Split block (may be same as input)
     */
    auto split_block(Block* block, size_t size) -> Block*;

    /**
     * @brief Coalesce adjacent free blocks.
     *
     * @param block Block to coalesce
     */
    auto coalesce_blocks(Block* block) -> void;

    /**
     * @brief Round size to allocation granularity.
     *
     * Uses power-of-2 rounding up to 1MB, then 512KB increments.
     *
     * @param size Requested size
     * @return Rounded size
     */
    auto round_size(size_t size) const -> size_t;
};

/**
 * @brief Memory statistics structure.
 */
struct MemoryStats {
    size_t allocated_bytes;      ///< Currently allocated
    size_t cached_bytes;          ///< Cached (free but not released)
    size_t peak_allocated;        ///< Peak allocation
    size_t num_allocs;            ///< Total allocations
    size_t num_device_allocs;     ///< Actual device allocations
    float fragmentation_ratio;    ///< Fragmentation metric (0-1)
};

/**
 * @brief Global caching allocator.
 *
 * Singleton managing memory pools for all devices.
 * Replaces default allocator in Backend implementations.
 *
 * Design Pattern: Singleton, Strategy
 * Thread Safety: Thread-safe via per-device mutexes
 *
 * @code
 * // Get global allocator
 * auto& allocator = CachingAllocator::get();
 *
 * // Allocate memory
 * void* ptr = allocator.allocate(1024, 0);  // 1KB on device 0
 *
 * // Free memory (returned to cache)
 * allocator.free(ptr);
 *
 * // Clear cache
 * allocator.empty_cache();
 * @endcode
 */
class CachingAllocator {
public:
    /**
     * @brief Get singleton instance.
     */
    static auto get() -> CachingAllocator&;

    /**
     * @brief Allocate memory on device.
     *
     * @param size Size in bytes
     * @param device_id Device index
     * @param stream Associated stream
     * @return Pointer to memory
     */
    auto allocate(size_t size, int device_id,
                 StreamHandle stream = nullptr) -> void*;

    /**
     * @brief Free memory.
     *
     * @param ptr Pointer to free
     */
    auto free(void* ptr) -> void;

    /**
     * @brief Empty all caches.
     */
    auto empty_cache() -> void;

    /**
     * @brief Empty cache for specific device.
     *
     * @param device_id Device index
     */
    auto empty_cache(int device_id) -> void;

    /**
     * @brief Get statistics for device.
     *
     * @param device_id Device index
     * @return Memory statistics
     */
    auto get_stats(int device_id) const -> MemoryStats;

    /**
     * @brief Get total statistics across all devices.
     */
    auto get_total_stats() const -> MemoryStats;

    /**
     * @brief Set memory fraction limit.
     *
     * Limits cache size to fraction of total device memory.
     *
     * @param fraction Fraction in [0, 1]
     */
    auto set_memory_fraction(float fraction) -> void;

private:
    CachingAllocator() = default;

    // Per-device memory pools
    std::unordered_map<int, std::unique_ptr<DeviceMemoryPool>> pools_;

    // Backends for each device type
    std::unordered_map<Device::Type, Backend*> backends_;

    mutable std::mutex mutex_;

    /**
     * @brief Get or create pool for device.
     */
    auto get_pool(int device_id) -> DeviceMemoryPool*;
};

} // namespace tenzor
```

### 2.4 Integration with Storage

```cpp
// Modifications to include/tenzor/core/storage.hpp

namespace tenzor {

/**
 * @brief Cached device storage using CachingAllocator.
 *
 * Drop-in replacement for DeviceStorage that uses caching allocator.
 */
class CachedDeviceStorage : public Storage {
public:
    /**
     * @brief Allocate device memory using caching allocator.
     *
     * @param size_bytes Size to allocate
     * @param device Device specification
     */
    CachedDeviceStorage(size_t size_bytes, Device device);

    /**
     * @brief Destructor returns memory to cache.
     */
    ~CachedDeviceStorage() override;

    // ... same interface as DeviceStorage ...

private:
    void* device_ptr_{nullptr};
    size_t size_{0};
    Device device_;
    mutable std::atomic<int64_t> ref_count_{1};
};

} // namespace tenzor
```

### 2.5 Memory Management API

```cpp
// include/tenzor/core/memory.hpp

namespace tenzor {
namespace memory {

/**
 * @brief Get current memory usage.
 *
 * @param device_id Device index (-1 for all devices)
 * @return Memory statistics
 */
auto get_memory_stats(int device_id = -1) -> MemoryStats;

/**
 * @brief Clear memory cache.
 *
 * @param device_id Device index (-1 for all devices)
 */
auto empty_cache(int device_id = -1) -> void;

/**
 * @brief Set maximum memory fraction.
 *
 * @param fraction Fraction of device memory to use [0, 1]
 */
auto set_memory_fraction(float fraction) -> void;

/**
 * @brief Reset peak memory statistics.
 */
auto reset_peak_stats() -> void;

/**
 * @brief Print memory summary.
 *
 * @param device_id Device index (-1 for all devices)
 */
auto print_memory_summary(int device_id = -1) -> void;

} // namespace memory
} // namespace tenzor
```

---

## 3. DataLoader Architecture

### 3.1 Overview

Efficient data loading is essential for GPU training:
- **Problem**: Data loading becomes bottleneck when GPU is fast
- **Solution**: Multi-threaded prefetching with worker pools
- **Benefits**: Overlap I/O with computation, 2-5x speedup

### 3.2 Component Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      DataLoader System                        │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐│
│  │                    Dataset (Abstract)                     ││
│  │  ┌────────────────┐    ┌──────────────────┐             ││
│  │  │ __len__()      │    │  __getitem__(i)  │             ││
│  │  │ returns: N     │    │  returns: Sample │             ││
│  │  └────────────────┘    └──────────────────┘             ││
│  └──────────────────────────────────────────────────────────┘│
│           │                                                    │
│           ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐│
│  │                  DataLoader                               ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     ││
│  │  │Worker Pool  │  │Batch Queue  │  │ Prefetching │     ││
│  │  │(N threads)  │  │(Bounded)    │  │  (K batches)│     ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘     ││
│  └──────────────────────────────────────────────────────────┘│
│           │                    │                    │          │
│           ▼                    ▼                    ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Sampler   │    │  Collate    │    │ Pin Memory  │      │
│  │ (Shuffling) │    │  Function   │    │  Transfer   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 Dataset Interface

```cpp
// include/tenzor/data/dataset.hpp

namespace tenzor {
namespace data {

/**
 * @brief Sample type returned by dataset.
 *
 * Typically (input, target) pair but can be arbitrary tuple.
 */
using Sample = std::vector<Tensor>;

/**
 * @brief Abstract dataset interface.
 *
 * Base class for all datasets. Derived classes must implement
 * size() and operator[] to provide data access.
 *
 * Design Pattern: Template Method
 * Thread Safety: Must be thread-safe for multi-worker DataLoader
 *
 * @code
 * class MyDataset : public Dataset {
 * public:
 *     auto size() const -> size_t override {
 *         return data_.size();
 *     }
 *
 *     auto operator[](size_t index) -> Sample override {
 *         return {data_[index], labels_[index]};
 *     }
 *
 * private:
 *     std::vector<Tensor> data_;
 *     std::vector<Tensor> labels_;
 * };
 * @endcode
 */
class Dataset {
public:
    virtual ~Dataset() = default;

    /**
     * @brief Get dataset size.
     *
     * @return Number of samples
     */
    virtual auto size() const -> size_t = 0;

    /**
     * @brief Get sample at index.
     *
     * @param index Sample index [0, size)
     * @return Sample (typically {input, target})
     * @throws std::out_of_range if index invalid
     */
    virtual auto operator[](size_t index) -> Sample = 0;

    /**
     * @brief Alternative access method.
     */
    auto get(size_t index) -> Sample {
        return (*this)[index];
    }
};

/**
 * @brief Tensor dataset for in-memory data.
 *
 * Wraps existing tensors into dataset interface.
 *
 * @code
 * Tensor data = randn({1000, 784});
 * Tensor labels = randint(0, 10, {1000});
 *
 * auto dataset = TensorDataset({data, labels});
 * @endcode
 */
class TensorDataset : public Dataset {
public:
    /**
     * @brief Construct from tensor list.
     *
     * @param tensors List of tensors (must have same length in dim 0)
     */
    explicit TensorDataset(std::vector<Tensor> tensors);

    auto size() const -> size_t override;
    auto operator[](size_t index) -> Sample override;

private:
    std::vector<Tensor> tensors_;
    size_t size_;
};

/**
 * @brief Subset of a dataset.
 *
 * Creates a view into existing dataset with specified indices.
 * Useful for train/val splits.
 *
 * @code
 * auto full_dataset = TensorDataset({data, labels});
 *
 * std::vector<size_t> train_indices = {0, 1, 2, 3, 4};
 * auto train_dataset = Subset(full_dataset, train_indices);
 * @endcode
 */
class Subset : public Dataset {
public:
    /**
     * @brief Construct subset from dataset and indices.
     *
     * @param dataset Parent dataset
     * @param indices Indices to include
     */
    Subset(std::shared_ptr<Dataset> dataset,
           std::vector<size_t> indices);

    auto size() const -> size_t override;
    auto operator[](size_t index) -> Sample override;

private:
    std::shared_ptr<Dataset> dataset_;
    std::vector<size_t> indices_;
};

} // namespace data
} // namespace tenzor
```

### 3.4 DataLoader Implementation

```cpp
// include/tenzor/data/dataloader.hpp

namespace tenzor {
namespace data {

/**
 * @brief Batch collation function type.
 *
 * Combines multiple samples into batched tensors.
 */
using CollateFn = std::function<Sample(std::vector<Sample>)>;

/**
 * @brief Default collate function.
 *
 * Stacks samples along batch dimension.
 * Assumes all samples have same structure and shapes.
 *
 * @param samples List of samples
 * @return Batched sample
 */
auto default_collate(std::vector<Sample> samples) -> Sample;

/**
 * @brief Multi-threaded data loader with prefetching.
 *
 * Loads data in background threads while model trains.
 * Implements producer-consumer pattern with bounded queue.
 *
 * Design Pattern: Producer-Consumer, Iterator
 * Thread Safety: Thread-safe iteration
 *
 * Features:
 * - Multi-worker parallelism
 * - Prefetching (load N batches ahead)
 * - Automatic batching and shuffling
 * - Pin memory for fast GPU transfer
 * - Drop last incomplete batch
 *
 * @code
 * auto dataset = TensorDataset({data, labels});
 * DataLoader loader(
 *     dataset,
 *     32,      // batch_size
 *     true,    // shuffle
 *     4,       // num_workers
 *     2        // prefetch_factor (batches per worker)
 * );
 *
 * for (auto& batch : loader) {
 *     auto [inputs, targets] = batch;
 *     // ... training ...
 * }
 * @endcode
 */
class DataLoader {
public:
    /**
     * @brief Construct data loader.
     *
     * @param dataset Dataset to load from
     * @param batch_size Samples per batch
     * @param shuffle Shuffle samples each epoch
     * @param num_workers Number of worker threads (0 = main thread)
     * @param prefetch_factor Batches to prefetch per worker
     * @param pin_memory Pin memory for CUDA transfer
     * @param drop_last Drop last incomplete batch
     * @param collate_fn Batch collation function
     */
    DataLoader(
        std::shared_ptr<Dataset> dataset,
        size_t batch_size = 1,
        bool shuffle = false,
        int num_workers = 0,
        int prefetch_factor = 2,
        bool pin_memory = false,
        bool drop_last = false,
        CollateFn collate_fn = default_collate
    );

    /**
     * @brief Destructor stops workers.
     */
    ~DataLoader();

    // Non-copyable
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    /**
     * @brief Iterator over batches.
     *
     * Thread-safe forward iterator that pulls batches from queue.
     */
    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Sample;
        using difference_type = std::ptrdiff_t;
        using pointer = Sample*;
        using reference = Sample&;

        Iterator(DataLoader* loader, bool is_end = false);

        auto operator++() -> Iterator&;
        auto operator++(int) -> Iterator;
        auto operator*() -> Sample&;
        auto operator->() -> Sample*;
        auto operator==(const Iterator& other) const -> bool;
        auto operator!=(const Iterator& other) const -> bool;

    private:
        DataLoader* loader_;
        Sample current_batch_;
        size_t batch_idx_;
        bool is_end_;

        auto fetch_next() -> void;
    };

    /**
     * @brief Begin iteration (starts workers if needed).
     */
    auto begin() -> Iterator;

    /**
     * @brief End iteration marker.
     */
    auto end() -> Iterator;

    /**
     * @brief Get number of batches per epoch.
     */
    auto size() const -> size_t;

    /**
     * @brief Reset for new epoch (reshuffles if enabled).
     */
    auto reset() -> void;

private:
    std::shared_ptr<Dataset> dataset_;
    size_t batch_size_;
    bool shuffle_;
    int num_workers_;
    int prefetch_factor_;
    bool pin_memory_;
    bool drop_last_;
    CollateFn collate_fn_;

    // Worker state
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_workers_{false};

    // Batch queue (bounded, thread-safe)
    std::queue<Sample> batch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    size_t max_queue_size_;

    // Index queue for workers
    std::queue<std::vector<size_t>> index_queue_;
    std::mutex index_mutex_;
    std::condition_variable index_cv_;

    // Current epoch state
    std::vector<size_t> indices_;
    size_t current_batch_idx_{0};
    size_t num_batches_;

    /**
     * @brief Worker thread function.
     *
     * Pulls batch indices, loads samples, collates, and enqueues.
     */
    auto worker_loop() -> void;

    /**
     * @brief Start worker threads.
     */
    auto start_workers() -> void;

    /**
     * @brief Stop worker threads.
     */
    auto stop_workers() -> void;

    /**
     * @brief Generate batch indices for epoch.
     */
    auto generate_indices() -> void;

    /**
     * @brief Enqueue batch indices for workers.
     */
    auto enqueue_batch_indices() -> void;

    /**
     * @brief Load batch (single-threaded path).
     */
    auto load_batch_sync(const std::vector<size_t>& indices) -> Sample;

    /**
     * @brief Pin memory for CUDA transfer.
     */
    auto pin_sample(Sample& sample) -> void;
};

} // namespace data
} // namespace tenzor
```

### 3.5 Sampler Interface

```cpp
// include/tenzor/data/sampler.hpp

namespace tenzor {
namespace data {

/**
 * @brief Abstract sampler interface.
 *
 * Defines order for accessing dataset samples.
 */
class Sampler {
public:
    virtual ~Sampler() = default;

    /**
     * @brief Generate sequence of indices.
     *
     * @return Vector of indices in access order
     */
    virtual auto sample() -> std::vector<size_t> = 0;

    /**
     * @brief Get number of samples.
     */
    virtual auto size() const -> size_t = 0;
};

/**
 * @brief Sequential sampler (no shuffling).
 */
class SequentialSampler : public Sampler {
public:
    explicit SequentialSampler(size_t size) : size_(size) {}

    auto sample() -> std::vector<size_t> override;
    auto size() const -> size_t override { return size_; }

private:
    size_t size_;
};

/**
 * @brief Random sampler (shuffling).
 */
class RandomSampler : public Sampler {
public:
    explicit RandomSampler(size_t size, uint64_t seed = 0);

    auto sample() -> std::vector<size_t> override;
    auto size() const -> size_t override { return size_; }

private:
    size_t size_;
    std::mt19937_64 rng_;
};

} // namespace data
} // namespace tenzor
```

---

## 4. Multi-GPU Architecture

### 4.1 Overview

Multi-GPU training parallelizes model across devices:
- **Problem**: Single GPU limits model size and batch size
- **Solution**: Replicate model, split batches, synchronize gradients
- **Benefits**: Near-linear speedup for data parallelism

### 4.2 Component Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Multi-GPU System (DataParallel)             │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐│
│  │                 DataParallel Module                       ││
│  │                                                            ││
│  │  Master GPU (device 0)                                    ││
│  │  ┌──────────────────────────────────────────┐            ││
│  │  │         Original Model                    │            ││
│  │  │  - Master parameters                      │            ││
│  │  │  - Optimizer state                        │            ││
│  │  └──────────────────────────────────────────┘            ││
│  │           │                                                ││
│  │           │ Replicate                                     ││
│  │           ▼                                                ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      ││
│  │  │  Replica    │  │  Replica    │  │  Replica    │      ││
│  │  │  GPU 0      │  │  GPU 1      │  │  GPU 2      │      ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘      ││
│  │        │                │                │                ││
│  │        │ Split Batch    │                │                ││
│  │        ▼                ▼                ▼                ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      ││
│  │  │  Forward    │  │  Forward    │  │  Forward    │      ││
│  │  │  (chunk 0)  │  │  (chunk 1)  │  │  (chunk 2)  │      ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘      ││
│  │        │                │                │                ││
│  │        ▼                ▼                ▼                ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      ││
│  │  │  Backward   │  │  Backward   │  │  Backward   │      ││
│  │  │  (local)    │  │  (local)    │  │  (local)    │      ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘      ││
│  │        │                │                │                ││
│  │        └────────────────┴────────────────┘                ││
│  │                         │                                  ││
│  │                  AllReduce Gradients                      ││
│  │                         │                                  ││
│  │                         ▼                                  ││
│  │          ┌──────────────────────────────┐                 ││
│  │          │    Gradient Synchronization  │                 ││
│  │          │  - Gather to master          │                 ││
│  │          │  - Average gradients         │                 ││
│  │          │  - Update master params      │                 ││
│  │          └──────────────────────────────┘                 ││
│  └──────────────────────────────────────────────────────────┘│
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 DataParallel Module

```cpp
// include/tenzor/nn/parallel/data_parallel.hpp

namespace tenzor {
namespace nn {

/**
 * @brief Data parallelism wrapper for multi-GPU training.
 *
 * Replicates module across GPUs, splits input batch, executes forward
 * pass in parallel, synchronizes gradients during backward.
 *
 * Design Pattern: Decorator, SPMD (Single Program Multiple Data)
 * Thread Safety: Forward/backward are thread-safe
 *
 * Algorithm:
 * 1. Replicate model to all GPUs
 * 2. Split input batch into chunks (one per GPU)
 * 3. Scatter chunks to GPUs
 * 4. Forward pass in parallel
 * 5. Gather outputs to master GPU
 * 6. Backward pass in parallel (from gathered loss)
 * 7. AllReduce gradients (average across GPUs)
 * 8. Optimizer updates master model
 *
 * Limitations:
 * - Requires batch_size >= num_gpus
 * - Single-node only (use DistributedDataParallel for multi-node)
 * - Master GPU requires extra memory (gathers outputs)
 *
 * @code
 * // Wrap model for multi-GPU
 * auto model = std::make_shared<MyModel>();
 * auto parallel_model = DataParallel(
 *     model,
 *     {0, 1, 2, 3},  // Use GPUs 0-3
 *     0              // Master GPU
 * );
 *
 * // Training loop unchanged
 * for (auto& batch : dataloader) {
 *     auto [inputs, targets] = batch;
 *
 *     optimizer.zero_grad();
 *     Variable output = parallel_model.forward(inputs);
 *     Variable loss = criterion(output, targets);
 *     loss.backward();
 *     optimizer.step();
 * }
 * @endcode
 */
class DataParallel : public Module {
public:
    /**
     * @brief Construct data parallel module.
     *
     * @param module Model to parallelize
     * @param device_ids GPU devices to use
     * @param output_device Master GPU (default: device_ids[0])
     * @param dim Batch dimension (default: 0)
     */
    DataParallel(
        std::shared_ptr<Module> module,
        std::vector<int> device_ids,
        int output_device = -1,
        int dim = 0
    );

    /**
     * @brief Forward pass with data parallelism.
     *
     * Steps:
     * 1. Replicate module to all devices
     * 2. Split input along batch dimension
     * 3. Scatter inputs to devices
     * 4. Parallel forward on each device
     * 5. Gather outputs to master device
     * 6. Concatenate outputs
     *
     * @param input Input variable (must be on master device)
     * @return Output variable (on master device)
     * @throws std::runtime_error if batch_size < num_devices
     */
    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Get underlying module.
     */
    auto module() -> std::shared_ptr<Module> { return module_; }

    /**
     * @brief Get device IDs.
     */
    auto device_ids() const -> const std::vector<int>& {
        return device_ids_;
    }

    /**
     * @brief Get master device ID.
     */
    auto output_device() const -> int { return output_device_; }

    /**
     * @brief Gather outputs from all devices.
     *
     * @param outputs Outputs from each device
     * @return Concatenated output on master device
     */
    auto gather(const std::vector<Variable>& outputs) -> Variable;

    /**
     * @brief Override parameters to return master module params.
     */
    auto parameters() -> std::vector<Variable*> override;

    /**
     * @brief Override named_parameters for master module.
     */
    auto named_parameters()
        -> std::vector<std::pair<std::string, Variable*>> override;

private:
    std::shared_ptr<Module> module_;     ///< Original module
    std::vector<int> device_ids_;        ///< GPU device IDs
    int output_device_;                  ///< Master GPU
    int dim_;                            ///< Batch dimension

    // Replicated modules (one per device)
    std::vector<std::shared_ptr<Module>> replicas_;

    // Streams for parallel execution
    std::vector<StreamHandle> streams_;

    /**
     * @brief Replicate module to all devices.
     *
     * Creates shallow copies with parameter references to master.
     *
     * @return Vector of module replicas
     */
    auto replicate() -> std::vector<std::shared_ptr<Module>>;

    /**
     * @brief Split tensor along dimension.
     *
     * @param tensor Tensor to split
     * @param chunks Number of chunks
     * @param dim Dimension to split along
     * @return Vector of tensor chunks
     */
    auto scatter(const Variable& tensor, int chunks, int dim)
        -> std::vector<Variable>;

    /**
     * @brief Execute forward in parallel.
     *
     * @param replicas Module replicas
     * @param inputs Input chunks
     * @return Output chunks
     */
    auto parallel_forward(
        const std::vector<std::shared_ptr<Module>>& replicas,
        const std::vector<Variable>& inputs
    ) -> std::vector<Variable>;

    /**
     * @brief Synchronize gradients across devices.
     *
     * Averages gradients from all replicas into master module.
     */
    auto synchronize_gradients() -> void;
};

/**
 * @brief Helper to create DataParallel module.
 *
 * @param module Module to parallelize
 * @param device_ids GPU device IDs (default: all available)
 * @return DataParallel wrapper
 */
auto make_data_parallel(
    std::shared_ptr<Module> module,
    std::vector<int> device_ids = {}
) -> std::shared_ptr<DataParallel>;

} // namespace nn
} // namespace tenzor
```

### 4.4 Gradient Synchronization

```cpp
// include/tenzor/nn/parallel/comm.hpp

namespace tenzor {
namespace nn {
namespace parallel {

/**
 * @brief Communication primitives for multi-GPU.
 *
 * Implements collective operations for gradient synchronization.
 */

/**
 * @brief AllReduce operation on tensors.
 *
 * Reduces tensors from all devices and broadcasts result.
 * Uses ring-AllReduce algorithm for efficiency.
 *
 * @param tensors Tensors to reduce (one per device)
 * @param op Reduction operation (Sum, Mean)
 * @return Reduced tensor on each device
 */
auto allreduce(
    const std::vector<Tensor>& tensors,
    ReduceOp op = ReduceOp::Mean
) -> std::vector<Tensor>;

/**
 * @brief Broadcast tensor from source to all devices.
 *
 * @param tensor Source tensor
 * @param src_device Source device
 * @param dst_devices Destination devices
 * @return Broadcasted tensors
 */
auto broadcast(
    const Tensor& tensor,
    int src_device,
    const std::vector<int>& dst_devices
) -> std::vector<Tensor>;

/**
 * @brief Gather tensors from all devices to master.
 *
 * @param tensors Tensors to gather
 * @param dst_device Destination device
 * @param dim Dimension to concatenate along
 * @return Gathered tensor
 */
auto gather(
    const std::vector<Tensor>& tensors,
    int dst_device,
    int dim = 0
) -> Tensor;

/**
 * @brief Scatter tensor from master to all devices.
 *
 * @param tensor Tensor to scatter
 * @param dst_devices Destination devices
 * @param dim Dimension to split along
 * @return Scattered tensor chunks
 */
auto scatter(
    const Tensor& tensor,
    const std::vector<int>& dst_devices,
    int dim = 0
) -> std::vector<Tensor>;

/**
 * @brief Reduction operation type.
 */
enum class ReduceOp {
    Sum,    ///< Sum across devices
    Mean,   ///< Average across devices
    Max,    ///< Maximum across devices
    Min     ///< Minimum across devices
};

} // namespace parallel
} // namespace nn
} // namespace tenzor
```

### 4.5 Multi-GPU Training Example

```cpp
// Example: Training with DataParallel

#include <tenzor/tenzor.hpp>
#include <tenzor/nn/parallel/data_parallel.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    // Initialize library
    initialize();

    // Check GPU availability
    int num_gpus = Device::cuda_device_count();
    std::cout << "Found " << num_gpus << " GPUs\n";

    // Create model
    auto model = std::make_shared<MyModel>();

    // Wrap in DataParallel
    std::vector<int> device_ids;
    for (int i = 0; i < num_gpus; ++i) {
        device_ids.push_back(i);
    }
    auto parallel_model = make_data_parallel(model, device_ids);

    // Create optimizer (operates on master parameters)
    optim::Adam optimizer(parallel_model->parameters(), 1e-3);

    // Create loss
    auto criterion = CrossEntropyLoss();

    // Create dataloader
    auto dataset = load_dataset();
    data::DataLoader loader(
        dataset,
        128,    // batch_size (must be >= num_gpus)
        true,   // shuffle
        4       // num_workers
    );

    // Training loop
    for (int epoch = 0; epoch < 10; ++epoch) {
        parallel_model->train();

        for (auto& batch : loader) {
            auto [inputs, targets] = batch;

            // Move batch to master GPU
            inputs = inputs.cuda(0);
            targets = targets.cuda(0);

            // Forward (automatically parallelized)
            optimizer.zero_grad();
            Variable output = parallel_model->forward(inputs);

            // Loss and backward
            Variable loss = criterion(output, targets);
            loss.backward();

            // Optimizer step (on master parameters)
            optimizer.step();
        }

        std::cout << "Epoch " << epoch << " complete\n";
    }

    finalize();
    return 0;
}
```

---

## 5. Integration Points

### 5.1 Integration Matrix

| Component | Integrates With | Integration Type | Modification Required |
|-----------|----------------|------------------|----------------------|
| Mixed Precision | Autograd | Extend Function::forward | Moderate |
| Mixed Precision | Backend | Add FP16/BF16 kernels | Major |
| Mixed Precision | Optimizer | GradScaler integration | Minor |
| Memory Manager | Backend | Replace allocate/free | Moderate |
| Memory Manager | Storage | Use CachedDeviceStorage | Minor |
| DataLoader | Tensor | Batch collation | Minor |
| DataLoader | Thread Pool | Worker management | Minor |
| Multi-GPU | Module | Wrap existing modules | Minor |
| Multi-GPU | Autograd | Gradient synchronization | Moderate |
| Multi-GPU | Backend | Multi-device support | Minor |

### 5.2 Backward Compatibility

**Guaranteed Compatible:**
- All existing code continues to work without changes
- Mixed precision is opt-in via AutocastContext
- Memory manager is transparent (drop-in replacement)
- DataLoader provides familiar PyTorch-like API
- DataParallel is a wrapper (non-intrusive)

**Migration Path:**
```cpp
// Old code (still works)
Variable output = model.forward(input);

// New code (mixed precision)
{
    AutocastContext ctx(AutocastMode::Float16);
    Variable output = model.forward(input);
}

// Old code (still works)
auto dataset = TensorDataset({data, labels});

// New code (multi-threading)
DataLoader loader(dataset, 32, true, 4);
```

### 5.3 API Stability

**Stable APIs (Phase 8+):**
- `tenzor::DType` enum extensions
- `tenzor::autograd::AutocastContext`
- `tenzor::autograd::GradScaler`
- `tenzor::data::Dataset`
- `tenzor::data::DataLoader`
- `tenzor::nn::DataParallel`
- `tenzor::memory::*` utilities

**Internal APIs (subject to change):**
- `CachingAllocator` implementation details
- Worker thread scheduling
- Gradient synchronization internals

---

## 6. Thread Safety Considerations

### 6.1 Thread Safety Matrix

| Component | Thread Safety | Synchronization | Notes |
|-----------|--------------|-----------------|-------|
| AutocastContext | Thread-local | None | Each thread has own state |
| GradScaler | Not thread-safe | External | Use one per optimizer |
| CachingAllocator | Thread-safe | Per-device mutex | Fine-grained locking |
| DeviceMemoryPool | Thread-safe | Mutex | Coarse-grained locking |
| Dataset | Must be thread-safe | User responsibility | Called from workers |
| DataLoader | Thread-safe | Queue mutex + CV | Producer-consumer |
| DataParallel | Thread-safe | CUDA streams | Implicit synchronization |

### 6.2 Locking Hierarchy

To prevent deadlocks, locks must be acquired in this order:

1. **Global Registry Lock** (backend_registry)
2. **Allocator Lock** (CachingAllocator::mutex_)
3. **Pool Lock** (DeviceMemoryPool::mutex_)
4. **Queue Lock** (DataLoader::queue_mutex_)
5. **Stream Lock** (CUDA stream synchronization)

**Rule:** Never acquire a lock while holding a lock at a higher level.

### 6.3 Race Condition Prevention

**AutocastContext:**
- Thread-local storage eliminates races
- No synchronization needed

**CachingAllocator:**
- Mutex protects free_blocks_ and active_blocks_
- Per-device pools reduce contention
- Read-only operations use shared_lock (C++17)

**DataLoader:**
- Bounded queue with condition variables
- Worker threads coordinate via index_queue_
- Batch queue protected by queue_mutex_

**DataParallel:**
- CUDA streams provide implicit synchronization
- Gradient synchronization uses allreduce barriers
- No explicit locks needed (rely on CUDA synchronization)

### 6.4 Memory Ordering

**Atomic Operations:**
```cpp
// Storage reference counting (relaxed ordering)
ref_count_.fetch_add(1, std::memory_order_relaxed);
ref_count_.fetch_sub(1, std::memory_order_release);

// Allocator statistics (relaxed ordering)
allocated_bytes_.fetch_add(size, std::memory_order_relaxed);

// Worker stop flag (acquire-release ordering)
stop_workers_.store(true, std::memory_order_release);
if (stop_workers_.load(std::memory_order_acquire)) { ... }
```

---

## 7. Performance Targets

### 7.1 Mixed Precision Training

**Target Speedup:**
- Forward pass: **2.5x faster** (GPU with Tensor Cores)
- Backward pass: **2.2x faster**
- End-to-end: **2.0x faster** (including data loading)
- Memory usage: **45% reduction** (activations + gradients)

**Validation:**
- ResNet-50 training: 320 images/sec → 640 images/sec (V100)
- Transformer training: 25k tokens/sec → 50k tokens/sec
- Memory: 16GB → 9GB (batch_size=64)

### 7.2 Memory Management

**Target Improvements:**
- Allocation time: **100x faster** (10μs → 100ns)
- Fragmentation: **< 5%** (vs 20-30% without caching)
- OOM prevention: **Effective** (dynamic cache eviction)

**Validation:**
- Microbenchmark: 1000 allocs in 0.1ms (vs 10ms without cache)
- Peak memory: Within 5% of theoretical minimum
- Long training runs: No memory leaks

### 7.3 Data Loading

**Target Speedup:**
- Single-threaded: Baseline
- 4 workers: **3.5x faster**
- 8 workers: **6.0x faster** (with prefetching)
- Overhead: **< 5%** (worker management)

**Validation:**
- ImageNet loading: 5000 images/sec (8 workers)
- CPU utilization: 80% (with 8 cores)
- GPU utilization: 95% (vs 60% without prefetching)

### 7.4 Multi-GPU Training

**Target Speedup (4 GPUs):**
- Linear scaling: **4.0x** (ideal)
- Achieved: **3.6x** (90% efficiency)
- Overhead: **10%** (gradient synchronization)

**Validation:**
- ResNet-50: 1280 images/sec (4x V100)
- Gradient sync: < 5ms per step
- Memory per GPU: Constant (no duplication)

### 7.5 Performance Profiling

**Metrics to Track:**
```cpp
struct PerformanceMetrics {
    // Mixed precision
    float fp16_speedup;
    float memory_savings;
    int overflow_count;

    // Memory management
    float allocation_time_avg;
    float fragmentation_ratio;
    size_t peak_memory;

    // Data loading
    float throughput_images_per_sec;
    float cpu_utilization;
    int queue_stalls;

    // Multi-GPU
    float scaling_efficiency;
    float gradient_sync_time;
    float imbalance_factor;
};
```

---

## 8. Implementation Roadmap

### 8.1 Phase 8.1: Mixed Precision (4 weeks)

**Week 1-2: Core Types**
- [ ] Implement Float16 and BFloat16 structs
- [ ] Add DType enum extensions
- [ ] Implement conversion operators
- [ ] Unit tests for precision and range

**Week 3: Autocast Mechanism**
- [ ] Implement AutocastContext
- [ ] Thread-local state management
- [ ] Operation whitelist/blacklist
- [ ] Integration tests with existing ops

**Week 4: Gradient Scaler**
- [ ] Implement GradScaler class
- [ ] Loss scaling and unscaling
- [ ] Overflow detection
- [ ] Dynamic scale adjustment
- [ ] Integration with optimizers

**Deliverables:**
- `include/tenzor/core/dtype.hpp` (extended)
- `include/tenzor/autograd/autocast.hpp`
- `include/tenzor/autograd/grad_scaler.hpp`
- `tests/unit/test_mixed_precision.cpp`
- Documentation and examples

### 8.2 Phase 8.2: Memory Management (3 weeks)

**Week 1: Allocator Design**
- [ ] Implement DeviceMemoryPool
- [ ] Block splitting and coalescing
- [ ] Size class organization
- [ ] Unit tests for correctness

**Week 2: Caching Allocator**
- [ ] Implement CachingAllocator singleton
- [ ] Multi-device support
- [ ] Statistics tracking
- [ ] Integration with Backend

**Week 3: Storage Integration**
- [ ] Implement CachedDeviceStorage
- [ ] Replace default allocator
- [ ] Performance benchmarks
- [ ] Fragmentation analysis

**Deliverables:**
- `include/tenzor/core/allocator.hpp`
- `src/core/allocator.cpp`
- `tests/unit/test_caching_allocator.cpp`
- `benchmarks/bench_allocation.cpp`
- Memory profiling tools

### 8.3 Phase 8.3: DataLoader (3 weeks)

**Week 1: Dataset Interface**
- [ ] Implement Dataset base class
- [ ] TensorDataset implementation
- [ ] Subset and transforms
- [ ] Unit tests

**Week 2: DataLoader Core**
- [ ] Implement single-threaded DataLoader
- [ ] Batch collation
- [ ] Shuffling and sampling
- [ ] Iterator interface

**Week 3: Multi-threading**
- [ ] Worker thread pool
- [ ] Prefetching with bounded queue
- [ ] Pin memory support
- [ ] Performance tests

**Deliverables:**
- `include/tenzor/data/dataset.hpp`
- `include/tenzor/data/dataloader.hpp`
- `src/data/dataloader.cpp`
- `tests/unit/test_dataloader.cpp`
- `examples/dataloader_example.cpp`

### 8.4 Phase 8.4: Multi-GPU (4 weeks)

**Week 1: Communication Primitives**
- [ ] Implement allreduce, broadcast
- [ ] Implement gather, scatter
- [ ] CUDA stream management
- [ ] Unit tests

**Week 2: DataParallel Module**
- [ ] Implement DataParallel wrapper
- [ ] Model replication
- [ ] Batch splitting
- [ ] Forward pass parallelization

**Week 3: Gradient Synchronization**
- [ ] Implement gradient averaging
- [ ] Backward hook integration
- [ ] Synchronization optimization
- [ ] Correctness tests

**Week 4: Integration & Testing**
- [ ] End-to-end multi-GPU training
- [ ] Scaling efficiency benchmarks
- [ ] Documentation
- [ ] Examples (ResNet, Transformer)

**Deliverables:**
- `include/tenzor/nn/parallel/data_parallel.hpp`
- `include/tenzor/nn/parallel/comm.hpp`
- `src/nn/parallel/data_parallel.cpp`
- `tests/integration/test_multi_gpu.cpp`
- `examples/multi_gpu_training.cpp`

### 8.5 Testing & Validation (2 weeks)

**Week 1: Integration Testing**
- [ ] Mixed precision + DataLoader
- [ ] Multi-GPU + Memory Management
- [ ] Full training pipelines
- [ ] Numerical accuracy validation

**Week 2: Performance Validation**
- [ ] Benchmark suite
- [ ] Profiling and optimization
- [ ] Scaling studies
- [ ] Documentation updates

**Deliverables:**
- Comprehensive test suite
- Performance report
- User guide and tutorials
- Migration guide from Phase 7

### 8.6 Total Timeline

**Total Duration:** 16 weeks (4 months)
**Team Size:** 2-3 engineers
**Effort:** ~1000 engineering hours

**Milestones:**
- M1 (Week 4): Mixed precision training functional
- M2 (Week 7): Memory management integrated
- M3 (Week 10): DataLoader with multi-threading
- M4 (Week 14): Multi-GPU training working
- M5 (Week 16): Production ready

---

## 9. Risk Assessment

### 9.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Float16 numerical instability | High | High | Extensive testing, loss scaling |
| Memory fragmentation | Medium | High | Advanced coalescing algorithms |
| DataLoader thread contention | Medium | Medium | Lock-free queues, profiling |
| Multi-GPU scaling efficiency | Medium | High | Optimize gradient sync, overlap |
| CUDA compatibility issues | Low | High | Test on multiple architectures |

### 9.2 Integration Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking existing API | Low | High | Extensive backward compat tests |
| Performance regressions | Medium | Medium | Continuous benchmarking |
| Memory leaks in threads | Medium | High | Valgrind, sanitizers |
| Race conditions | Medium | High | ThreadSanitizer, stress tests |

---

## 10. Future Extensions

### 10.1 Phase 8+

**Distributed Training (Phase 9):**
- DistributedDataParallel (multi-node)
- NCCL backend integration
- Gradient compression
- Ring-AllReduce optimization

**Advanced Memory (Phase 9):**
- Memory defragmentation
- Persistent memory pools
- Unified memory support
- Memory-aware scheduling

**Data Pipeline (Phase 9):**
- GPU data augmentation
- DALI integration
- Asynchronous prefetching
- Zero-copy transfers

**Model Parallelism (Phase 10):**
- Pipeline parallelism
- Tensor parallelism
- Model sharding
- Megatron-style parallelism

---

## Appendix A: File Structure

```
include/tenzor/
├── core/
│   ├── dtype.hpp              # Extended with Float16/BFloat16
│   ├── allocator.hpp          # NEW: Caching allocator
│   └── memory.hpp             # NEW: Memory utilities
├── autograd/
│   ├── autocast.hpp           # NEW: Autocast context
│   └── grad_scaler.hpp        # NEW: Gradient scaler
├── data/
│   ├── dataset.hpp            # NEW: Dataset interface
│   ├── dataloader.hpp         # NEW: DataLoader
│   └── sampler.hpp            # NEW: Samplers
└── nn/
    └── parallel/
        ├── data_parallel.hpp  # NEW: DataParallel
        └── comm.hpp           # NEW: Communication primitives

src/
├── core/
│   ├── allocator.cpp          # NEW: Allocator implementation
│   └── memory.cpp             # NEW: Memory management
├── autograd/
│   ├── autocast.cpp           # NEW: Autocast implementation
│   └── grad_scaler.cpp        # NEW: Gradient scaler
├── data/
│   └── dataloader.cpp         # NEW: DataLoader implementation
├── nn/
│   └── parallel/
│       ├── data_parallel.cpp  # NEW: DataParallel
│       └── comm.cpp           # NEW: Communication
└── backends/
    └── cuda/
        └── kernels/
            └── fp16_kernels.cu # NEW: FP16 CUDA kernels

tests/
├── unit/
│   ├── test_mixed_precision.cpp
│   ├── test_caching_allocator.cpp
│   ├── test_dataloader.cpp
│   └── test_data_parallel.cpp
└── integration/
    ├── test_amp_training.cpp
    └── test_multi_gpu_training.cpp

examples/
├── mixed_precision_training.cpp
├── memory_profiling.cpp
├── dataloader_example.cpp
└── multi_gpu_training.cpp
```

---

## Appendix B: References

**Papers:**
1. "Mixed Precision Training" (Micikevicius et al., 2018)
2. "CUDA Best Practices Guide" (NVIDIA)
3. "PyTorch DataLoader Design" (PyTorch Documentation)
4. "Horovod: fast and easy distributed deep learning" (Sergeev & Del Balso, 2018)

**Libraries:**
- PyTorch mixed precision (torch.cuda.amp)
- TensorFlow AutoGraph
- NVIDIA Apex
- PyTorch DataLoader
- Horovod

**Standards:**
- IEEE 754 (Floating-point arithmetic)
- C++17/20 (Thread safety, memory model)
- CUDA Programming Guide
- cuBLAS/cuDNN documentation

---

**Document Status:** Design Complete
**Next Steps:** Begin Phase 8.1 implementation
**Review Date:** 2025-10-20
