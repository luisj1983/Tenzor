# Runtime Operation Dispatcher Architecture Design

## Executive Summary

This document presents a comprehensive architecture for a runtime operation dispatcher that eliminates all compile-time `#ifdef TENZOR_HAS_CUDA` checks from the Tenzor tensor library frontend. The design provides clean separation between frontend operations and backend implementations while maintaining performance and extensibility.

## Current Architecture Analysis

### Existing Components

The Tenzor library already has a solid foundation for runtime dispatch:

1. **Backend Interface** (`tenzor/backend/backend.hpp`)
   - Abstract `Backend` class defining the interface for all backends
   - Provides memory management, synchronization, and kernel dispatch
   - Each backend (CPU, CUDA, ROCm, OneAPI, Vulkan, Metal, WebGPU) implements this interface

2. **Operation Registry** (`tenzor/backend/registry.hpp`)
   - Thread-safe registry mapping (operation_name, device_type) → kernel function
   - Two-level map structure for efficient lookup
   - Global singleton access via `operation_registry()`

3. **Dispatcher** (`tenzor/backend/dispatch.hpp`)
   - Central routing logic for operations
   - Device compatibility checking
   - Backend selection based on tensor device

4. **Frontend Operations** (`tenzor/ops/math.hpp`, etc.)
   - Already using `Dispatcher::dispatch()` for all operations
   - Backend-agnostic implementation

### Current Flow

```
Frontend Operation (e.g., add)
    ↓
Dispatcher::dispatch("add", inputs)
    ↓
Backend::dispatch("add", inputs, attrs)
    ↓
Backend-specific kernel execution
```

### Issues with Current Architecture

1. **Hardcoded Backend Kernels**: CPU backend has hardcoded dispatch logic in `cpu_backend.cpp`
2. **No Fallback Mechanism**: Missing graceful fallback when backend doesn't implement operation
3. **No Runtime Registration**: Backends cannot dynamically register capabilities
4. **Limited Extensibility**: Adding new backends requires modifying core files

## Proposed Architecture

### Design Goals

1. **Zero Compile-Time Conditionals**: Frontend code contains no `#ifdef` checks
2. **Runtime Registration**: Backends register operations during library initialization
3. **Graceful Fallback**: Automatic CPU fallback for missing backend implementations
4. **Type Safety**: Strong typing for operation signatures
5. **Performance**: Zero overhead for hot-path dispatch
6. **Extensibility**: Easy addition of new backends and operations

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Layer                          │
│  (ops/math.hpp, nn/layers/*, autograd/*)                   │
│                                                             │
│  - Backend-agnostic operation calls                        │
│  - Uses only public Tensor API                             │
│  - No device-specific code                                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ Dispatcher::dispatch("op_name", inputs, attrs)
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                  Dispatcher Layer                           │
│  (backend/dispatch.hpp)                                     │
│                                                             │
│  1. Validate device compatibility                          │
│  2. Select backend based on tensor device                  │
│  3. Delegate to backend dispatcher                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ Backend::dispatch("op_name", inputs, attrs)
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                  Backend Layer                              │
│  (backends/cpu/*, backends/cuda/*, etc.)                   │
│                                                             │
│  1. Lookup operation in OperationRegistry                  │
│  2. Execute registered kernel                              │
│  3. Return result tensors                                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ KernelFunction(inputs, attrs)
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                  Kernel Layer                               │
│  (backends/*/kernels/*)                                     │
│                                                             │
│  - Device-specific implementations                         │
│  - CUDA, ROCm, OneAPI, Vulkan kernels                     │
│  - SIMD optimizations                                      │
└─────────────────────────────────────────────────────────────┘
```

### Component Design

#### 1. Operation Registry (Enhanced)

**File**: `include/tenzor/backend/registry.hpp`

The existing `OperationRegistry` already provides the core functionality. We enhance it with:

- **Fallback Support**: Automatic CPU fallback registration
- **Capability Queries**: Check if backend supports operation
- **Batch Registration**: Register multiple operations efficiently

```cpp
class OperationRegistry {
public:
    // Existing methods (already implemented)
    auto register_kernel(std::string_view op_name,
                        Device::Type device_type,
                        KernelFunction kernel) -> void;

    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor>;

    auto has_kernel(std::string_view op_name,
                   Device::Type device_type) const -> bool;

    // NEW: Fallback support
    auto register_fallback(std::string_view op_name,
                          Device::Type device_type,
                          Device::Type fallback_type) -> void;

    // NEW: Get kernel with fallback resolution
    auto get_kernel_with_fallback(std::string_view op_name,
                                  Device::Type device_type) const
        -> std::pair<KernelFunction, Device::Type>;

    // NEW: Batch registration helper
    template<typename BackendType>
    auto register_backend_operations(BackendType& backend) -> void;

private:
    mutable std::shared_mutex mutex_;

    // Operation -> Device -> Kernel
    std::unordered_map<
        std::string,
        std::unordered_map<Device::Type, KernelFunction>
    > kernels_;

    // Operation -> Device -> Fallback Device
    std::unordered_map<
        std::string,
        std::unordered_map<Device::Type, Device::Type>
    > fallbacks_;
};
```

#### 2. Backend Interface (Enhanced)

**File**: `include/tenzor/backend/backend.hpp`

The existing `Backend` interface is good but we add registration support:

```cpp
class Backend {
public:
    virtual ~Backend() = default;

    // Existing methods
    virtual auto name() const -> std::string_view = 0;
    virtual auto device_count() const -> int32_t = 0;
    virtual auto is_available() const -> bool = 0;

    // Memory management
    virtual auto allocate(size_t bytes, int32_t device_id) -> void* = 0;
    virtual auto deallocate(void* ptr) -> void = 0;
    virtual auto copy(void* dst, const void* src, size_t bytes,
                     CopyKind kind) -> void = 0;

    // Synchronization
    virtual auto synchronize(int32_t device_id) -> void = 0;
    virtual auto create_stream(int32_t device_id) -> StreamHandle = 0;
    virtual auto destroy_stream(StreamHandle stream) -> void = 0;
    virtual auto synchronize_stream(StreamHandle stream) -> void = 0;

    // Dispatch (delegated to OperationRegistry)
    virtual auto dispatch(const std::string& op_name,
                         std::span<const Tensor> inputs,
                         const OpAttributes& attrs) -> std::vector<Tensor>;

    // NEW: Registration hook called during backend initialization
    virtual auto register_operations(OperationRegistry& registry) -> void = 0;

    // NEW: Get device type this backend handles
    virtual auto device_type() const -> Device::Type = 0;

    // NEW: Get list of supported operations
    virtual auto supported_operations() const -> std::vector<std::string> = 0;
};
```

#### 3. Backend Registry

**File**: `include/tenzor/backend/backend_registry.hpp` (new)

Manages available backends and their lifecycle:

```cpp
class BackendRegistry {
public:
    static auto instance() -> BackendRegistry&;

    // Register a backend (called during library initialization)
    auto register_backend(Device::Type type,
                         std::unique_ptr<Backend> backend) -> void;

    // Get backend for device type
    auto get_backend(Device::Type type) -> Backend*;

    // Check if backend is available
    auto has_backend(Device::Type type) const -> bool;

    // Get all available backends
    auto available_backends() const -> std::vector<Device::Type>;

    // Initialize all backends (calls register_operations on each)
    auto initialize_all() -> void;

private:
    BackendRegistry() = default;

    std::shared_mutex mutex_;
    std::unordered_map<Device::Type, std::unique_ptr<Backend>> backends_;
    bool initialized_{false};
};

// Global accessor
inline auto backend_registry() -> BackendRegistry& {
    return BackendRegistry::instance();
}
```

#### 4. Operation Registration Macros

**File**: `include/tenzor/backend/registration.hpp` (new)

Convenience macros for backend developers:

```cpp
// Helper for kernel function wrapping
namespace detail {
    template<typename Fn>
    auto wrap_kernel(Fn&& fn) -> KernelFunction {
        return [fn = std::forward<Fn>(fn)](
            std::span<const Tensor> inputs,
            const OpAttributes& attrs) -> std::vector<Tensor> {
            return {fn(inputs, attrs)};
        };
    }
}

// Registration macro for single-output operations
#define TENZOR_REGISTER_KERNEL(backend_type, op_name, kernel_fn) \
    registry.register_kernel(                                     \
        op_name,                                                  \
        backend_type,                                            \
        [](std::span<const Tensor> inputs,                       \
           const OpAttributes& attrs) -> std::vector<Tensor> {   \
            return {kernel_fn(inputs, attrs)};                   \
        }                                                         \
    )

// Registration macro for multi-output operations
#define TENZOR_REGISTER_KERNEL_MULTI(backend_type, op_name, kernel_fn) \
    registry.register_kernel(                                           \
        op_name,                                                        \
        backend_type,                                                   \
        kernel_fn                                                       \
    )

// Fallback registration
#define TENZOR_REGISTER_FALLBACK(op_name, device_type, fallback_type) \
    registry.register_fallback(op_name, device_type, fallback_type)
```

### Backend Implementation Pattern

#### CPU Backend Example

**File**: `src/backends/cpu/cpu_backend.cpp`

```cpp
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/registry.hpp"
#include "tenzor/backend/registration.hpp"
#include <cstring>

namespace tenzor {
namespace cpu {

// Forward declarations for CPU kernels
auto add_kernel(std::span<const Tensor> inputs,
                const OpAttributes& attrs) -> Tensor;
auto matmul_kernel(std::span<const Tensor> inputs,
                   const OpAttributes& attrs) -> Tensor;
auto relu_kernel(std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> Tensor;
// ... more kernels

} // namespace cpu

class CPUBackend : public Backend {
public:
    auto name() const -> std::string_view override {
        return "cpu";
    }

    auto device_type() const -> Device::Type override {
        return Device::Type::CPU;
    }

    auto device_count() const -> int32_t override {
        return 1;
    }

    auto is_available() const -> bool override {
        return true;  // CPU always available
    }

    auto supported_operations() const -> std::vector<std::string> override {
        return {
            "add", "sub", "mul", "div",
            "matmul", "bmm", "dot",
            "relu", "sigmoid", "tanh",
            "conv2d", "pool2d",
            // ... full list
        };
    }

    auto register_operations(OperationRegistry& registry) -> void override {
        // Math operations
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "add", cpu::add_kernel);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "sub", cpu::sub_kernel);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "mul", cpu::mul_kernel);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "div", cpu::div_kernel);

        // Matrix operations
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "matmul", cpu::matmul_kernel);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "bmm", cpu::bmm_kernel);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "dot", cpu::dot_kernel);

        // Activations
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "relu", cpu::relu_kernel);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "sigmoid", cpu::sigmoid_kernel);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "tanh", cpu::tanh_kernel);

        // Convolutions
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "conv2d", cpu::conv2d_kernel);

        // ... register all operations
    }

    // Memory management implementations
    auto allocate(size_t bytes, int32_t device_id) -> void* override {
        return std::malloc(bytes);
    }

    auto deallocate(void* ptr) -> void override {
        std::free(ptr);
    }

    auto copy(void* dst, const void* src, size_t bytes,
             CopyKind kind) -> void override {
        std::memcpy(dst, src, bytes);
    }

    auto synchronize(int32_t device_id) -> void override {
        // No-op for CPU
    }

    auto create_stream(int32_t device_id) -> StreamHandle override {
        return nullptr;  // No streams for CPU
    }

    auto destroy_stream(StreamHandle stream) -> void override {
        // No-op for CPU
    }

    auto synchronize_stream(StreamHandle stream) -> void override {
        // No-op for CPU
    }
};

} // namespace tenzor

// Backend factory function
extern "C" auto create_cpu_backend() -> std::unique_ptr<tenzor::Backend> {
    return std::make_unique<tenzor::CPUBackend>();
}
```

#### CUDA Backend Example

**File**: `src/backends/cuda/cuda_backend.cpp`

```cpp
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/registry.hpp"
#include "tenzor/backend/registration.hpp"

// ONLY include CUDA headers in backend implementation
#ifdef TENZOR_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace tenzor {
namespace cuda {

#ifdef TENZOR_HAS_CUDA

// Forward declarations for CUDA kernels
auto add_kernel(std::span<const Tensor> inputs,
                const OpAttributes& attrs) -> Tensor;
auto matmul_kernel(std::span<const Tensor> inputs,
                   const OpAttributes& attrs) -> Tensor;
// ... more kernels

class CUDABackend : public Backend {
public:
    auto name() const -> std::string_view override {
        return "cuda";
    }

    auto device_type() const -> Device::Type override {
        return Device::Type::CUDA;
    }

    auto device_count() const -> int32_t override {
        int count = 0;
        cudaGetDeviceCount(&count);
        return count;
    }

    auto is_available() const -> bool override {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return err == cudaSuccess && count > 0;
    }

    auto supported_operations() const -> std::vector<std::string> override {
        return {
            "add", "sub", "mul", "div",
            "matmul", "bmm",
            "relu", "sigmoid",
            "conv2d",
            // ... operations implemented in CUDA
        };
    }

    auto register_operations(OperationRegistry& registry) -> void override {
        // Register CUDA implementations
        TENZOR_REGISTER_KERNEL(Device::Type::CUDA, "add", cuda::add_kernel);
        TENZOR_REGISTER_KERNEL(Device::Type::CUDA, "sub", cuda::sub_kernel);
        TENZOR_REGISTER_KERNEL(Device::Type::CUDA, "mul", cuda::mul_kernel);
        TENZOR_REGISTER_KERNEL(Device::Type::CUDA, "matmul", cuda::matmul_kernel);
        TENZOR_REGISTER_KERNEL(Device::Type::CUDA, "conv2d", cuda::conv2d_kernel);

        // Register CPU fallback for unimplemented operations
        // If CUDA doesn't implement an operation, fall back to CPU
        TENZOR_REGISTER_FALLBACK("advanced_op",
                                 Device::Type::CUDA,
                                 Device::Type::CPU);
    }

    auto allocate(size_t bytes, int32_t device_id) -> void* override {
        void* ptr = nullptr;
        cudaSetDevice(device_id);
        cudaMalloc(&ptr, bytes);
        return ptr;
    }

    auto deallocate(void* ptr) -> void override {
        cudaFree(ptr);
    }

    auto copy(void* dst, const void* src, size_t bytes,
             CopyKind kind) -> void override {
        cudaMemcpyKind cuda_kind;
        switch (kind) {
            case CopyKind::HostToHost:
                cuda_kind = cudaMemcpyHostToHost; break;
            case CopyKind::HostToDevice:
                cuda_kind = cudaMemcpyHostToDevice; break;
            case CopyKind::DeviceToHost:
                cuda_kind = cudaMemcpyDeviceToHost; break;
            case CopyKind::DeviceToDevice:
                cuda_kind = cudaMemcpyDeviceToDevice; break;
        }
        cudaMemcpy(dst, src, bytes, cuda_kind);
    }

    auto synchronize(int32_t device_id) -> void override {
        cudaSetDevice(device_id);
        cudaDeviceSynchronize();
    }

    auto create_stream(int32_t device_id) -> StreamHandle override {
        cudaSetDevice(device_id);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return static_cast<StreamHandle>(stream);
    }

    auto destroy_stream(StreamHandle stream) -> void override {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    }

    auto synchronize_stream(StreamHandle stream) -> void override {
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    }
};

#endif // TENZOR_HAS_CUDA

} // namespace cuda
} // namespace tenzor

// Backend factory function
extern "C" auto create_cuda_backend() -> std::unique_ptr<tenzor::Backend> {
#ifdef TENZOR_HAS_CUDA
    return std::make_unique<tenzor::cuda::CUDABackend>();
#else
    return nullptr;  // CUDA not available at compile time
#endif
}
```

### Library Initialization

**File**: `src/core/init.cpp` (new)

```cpp
#include "tenzor/backend/backend_registry.hpp"
#include "tenzor/backend/registry.hpp"

namespace tenzor {

// Forward declarations for backend factory functions
extern "C" {
    auto create_cpu_backend() -> std::unique_ptr<Backend>;

    #ifdef TENZOR_HAS_CUDA
    auto create_cuda_backend() -> std::unique_ptr<Backend>;
    #endif

    #ifdef TENZOR_HAS_ROCM
    auto create_rocm_backend() -> std::unique_ptr<Backend>;
    #endif

    #ifdef TENZOR_HAS_ONEAPI
    auto create_oneapi_backend() -> std::unique_ptr<Backend>;
    #endif

    #ifdef TENZOR_HAS_VULKAN
    auto create_vulkan_backend() -> std::unique_ptr<Backend>;
    #endif
}

// Initialize all available backends
auto initialize_backends() -> void {
    auto& registry = backend_registry();
    auto& op_registry = operation_registry();

    // CPU backend (always available)
    {
        auto backend = create_cpu_backend();
        if (backend && backend->is_available()) {
            backend->register_operations(op_registry);
            registry.register_backend(Device::Type::CPU, std::move(backend));
        }
    }

    // CUDA backend (if compiled with CUDA support)
    #ifdef TENZOR_HAS_CUDA
    {
        auto backend = create_cuda_backend();
        if (backend && backend->is_available()) {
            backend->register_operations(op_registry);
            registry.register_backend(Device::Type::CUDA, std::move(backend));
        }
    }
    #endif

    // ROCm backend (if compiled with ROCm support)
    #ifdef TENZOR_HAS_ROCM
    {
        auto backend = create_rocm_backend();
        if (backend && backend->is_available()) {
            backend->register_operations(op_registry);
            registry.register_backend(Device::Type::ROCm, std::move(backend));
        }
    }
    #endif

    // OneAPI backend (if compiled with OneAPI support)
    #ifdef TENZOR_HAS_ONEAPI
    {
        auto backend = create_oneapi_backend();
        if (backend && backend->is_available()) {
            backend->register_operations(op_registry);
            registry.register_backend(Device::Type::OneAPI, std::move(backend));
        }
    }
    #endif

    // Vulkan backend (if compiled with Vulkan support)
    #ifdef TENZOR_HAS_VULKAN
    {
        auto backend = create_vulkan_backend();
        if (backend && backend->is_available()) {
            backend->register_operations(op_registry);
            registry.register_backend(Device::Type::Vulkan, std::move(backend));
        }
    }
    #endif
}

// Automatic initialization using constructor function
struct BackendInitializer {
    BackendInitializer() {
        initialize_backends();
    }
};

// Static initializer runs before main()
static BackendInitializer g_backend_initializer;

} // namespace tenzor
```

### Fallback Mechanism

**File**: `src/backend/registry.cpp` (enhanced)

```cpp
auto OperationRegistry::get_kernel_with_fallback(
    std::string_view op_name,
    Device::Type device_type) const -> std::pair<KernelFunction, Device::Type> {

    std::shared_lock lock(mutex_);

    // Try to find kernel for requested device
    auto op_it = kernels_.find(std::string(op_name));
    if (op_it != kernels_.end()) {
        auto kernel_it = op_it->second.find(device_type);
        if (kernel_it != op_it->second.end()) {
            return {kernel_it->second, device_type};
        }
    }

    // Not found - check for registered fallback
    auto fallback_it = fallbacks_.find(std::string(op_name));
    if (fallback_it != fallbacks_.end()) {
        auto device_fallback_it = fallback_it->second.find(device_type);
        if (device_fallback_it != fallback_it->second.end()) {
            Device::Type fallback_type = device_fallback_it->second;

            // Try to get kernel for fallback device
            if (op_it != kernels_.end()) {
                auto kernel_it = op_it->second.find(fallback_type);
                if (kernel_it != op_it->second.end()) {
                    // Log fallback
                    std::cerr << "Warning: Operation '" << op_name
                              << "' falling back from "
                              << device_type_to_string(device_type)
                              << " to "
                              << device_type_to_string(fallback_type)
                              << std::endl;

                    return {kernel_it->second, fallback_type};
                }
            }
        }
    }

    throw std::runtime_error(
        "Operation '" + std::string(op_name) +
        "' not implemented for device type " +
        device_type_to_string(device_type) +
        " and no fallback available"
    );
}

auto OperationRegistry::dispatch(
    const std::string& op_name,
    std::span<const Tensor> inputs,
    const OpAttributes& attrs) -> std::vector<Tensor> {

    if (inputs.empty()) {
        throw std::runtime_error("Cannot dispatch operation with no inputs");
    }

    Device::Type device_type = inputs[0].device().type;

    // Get kernel with fallback support
    auto [kernel, actual_device] = get_kernel_with_fallback(op_name, device_type);

    // If fallback to different device, need to move tensors
    if (actual_device != device_type) {
        std::vector<Tensor> moved_inputs;
        moved_inputs.reserve(inputs.size());

        Device fallback_device;
        fallback_device.type = actual_device;
        fallback_device.index = 0;  // Use device 0 for fallback

        for (const auto& tensor : inputs) {
            moved_inputs.push_back(tensor.to(fallback_device));
        }

        // Execute on fallback device
        auto results = kernel(moved_inputs, attrs);

        // Move results back to original device
        std::vector<Tensor> final_results;
        final_results.reserve(results.size());

        Device original_device;
        original_device.type = device_type;
        original_device.index = inputs[0].device().index;

        for (auto& result : results) {
            final_results.push_back(result.to(original_device));
        }

        return final_results;
    }

    // Execute on requested device
    return kernel(inputs, attrs);
}
```

## Frontend Usage Examples

### Example 1: Math Operations (Already Backend-Agnostic)

**File**: `include/tenzor/ops/math.hpp`

```cpp
namespace tenzor {

// These operations are ALREADY backend-agnostic
// No changes needed!

auto add(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<Tensor> inputs = {a, b};
    return Dispatcher::dispatch("add", inputs)[0];
}

auto matmul(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<Tensor> inputs = {a, b};
    return Dispatcher::dispatch("matmul", inputs)[0];
}

auto relu(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("relu", inputs)[0];
}

} // namespace tenzor
```

### Example 2: Neural Network Layers

**File**: `include/tenzor/nn/layers/linear.hpp`

```cpp
namespace tenzor {
namespace nn {

class Linear {
public:
    Linear(int64_t in_features, int64_t out_features, bool bias = true)
        : in_features_(in_features), out_features_(out_features) {

        // Initialize weights and bias
        // These are device-agnostic
        weight_ = randn({out_features, in_features});
        if (bias) {
            bias_ = zeros({out_features});
        }
    }

    auto forward(const Tensor& input) -> Tensor {
        // Pure frontend code - no backend checks
        Tensor output = matmul(input, weight_.transpose(0, 1));

        if (bias_.has_value()) {
            output = output + bias_.value();
        }

        return output;
    }

    auto to(Device device) -> Linear& {
        weight_ = weight_.to(device);
        if (bias_.has_value()) {
            bias_ = bias_.value().to(device);
        }
        return *this;
    }

private:
    int64_t in_features_;
    int64_t out_features_;
    Tensor weight_;
    std::optional<Tensor> bias_;
};

} // namespace nn
} // namespace tenzor
```

### Example 3: User Application

```cpp
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/nn/layers/linear.hpp>

int main() {
    using namespace tenzor;

    // Create tensors on different devices
    Tensor cpu_a = randn({128, 64}, DType::Float32, Device::cpu());
    Tensor cpu_b = randn({128, 64}, DType::Float32, Device::cpu());

    // CPU computation (uses CPU backend)
    Tensor cpu_result = add(cpu_a, cpu_b);

    // Move to CUDA if available
    if (backend_registry().has_backend(Device::Type::CUDA)) {
        Tensor cuda_a = cpu_a.cuda(0);
        Tensor cuda_b = cpu_b.cuda(0);

        // CUDA computation (uses CUDA backend)
        Tensor cuda_result = add(cuda_a, cuda_b);

        // If operation not implemented in CUDA, automatically falls back to CPU
        Tensor complex_result = some_advanced_operation(cuda_a);
    }

    // Neural network layer works on any device
    nn::Linear layer(64, 32);

    // CPU execution
    Tensor cpu_output = layer.forward(cpu_a);

    // CUDA execution (if available)
    layer.to(Device::cuda(0));
    Tensor cuda_output = layer.forward(cuda_a);

    return 0;
}
```

## Performance Considerations

### 1. Dispatch Overhead

**Hot-Path Optimization**:
```cpp
// Cache kernel lookup for repeated operations
class CachedDispatcher {
    struct CacheKey {
        std::string op_name;
        Device::Type device_type;

        auto operator==(const CacheKey& other) const -> bool {
            return op_name == other.op_name &&
                   device_type == other.device_type;
        }
    };

    struct CacheKeyHash {
        auto operator()(const CacheKey& key) const -> size_t {
            return std::hash<std::string>{}(key.op_name) ^
                   (std::hash<int>{}(static_cast<int>(key.device_type)) << 1);
        }
    };

    mutable std::unordered_map<CacheKey, KernelFunction, CacheKeyHash> cache_;

    auto dispatch_cached(const std::string& op_name,
                        std::span<const Tensor> inputs,
                        const OpAttributes& attrs) -> std::vector<Tensor> {
        CacheKey key{op_name, inputs[0].device().type};

        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second(inputs, attrs);  // Fast path
        }

        // Slow path - lookup and cache
        auto [kernel, _] = operation_registry()
            .get_kernel_with_fallback(op_name, inputs[0].device().type);
        cache_[key] = kernel;
        return kernel(inputs, attrs);
    }
};
```

### 2. Memory Transfer Overhead

When falling back to CPU, minimize transfers:

```cpp
// Smart fallback - batch operations before transfer
class BatchedFallback {
    auto execute_with_fallback(
        const std::vector<std::string>& op_names,
        const std::vector<std::span<const Tensor>>& inputs) -> std::vector<Tensor> {

        // Move all tensors once
        Device cpu_device = Device::cpu();
        std::vector<Tensor> cpu_tensors;

        for (const auto& input_span : inputs) {
            for (const auto& tensor : input_span) {
                cpu_tensors.push_back(tensor.to(cpu_device));
            }
        }

        // Execute all operations on CPU
        std::vector<Tensor> results;
        // ... execute operations ...

        // Move results back once
        Device original_device = inputs[0][0].device();
        for (auto& result : results) {
            result = result.to(original_device);
        }

        return results;
    }
};
```

### 3. Inline Small Operations

For trivial operations, consider inlining:

```cpp
template<typename T>
inline auto add_inline(Tensor& self, const Tensor& other) -> Tensor& {
    // For small tensors on same device, inline the operation
    if (self.numel() < 1000 && self.device() == other.device()) {
        T* self_data = self.data<T>();
        const T* other_data = other.data<T>();

        for (int64_t i = 0; i < self.numel(); ++i) {
            self_data[i] += other_data[i];
        }

        return self;
    }

    // Large tensors - dispatch to backend
    return Dispatcher::dispatch("add_", {self, other})[0];
}
```

## Testing Strategy

### 1. Backend Registration Tests

```cpp
TEST(BackendRegistryTest, RegisterAndRetrieve) {
    auto& registry = backend_registry();

    // CPU should always be registered
    ASSERT_TRUE(registry.has_backend(Device::Type::CPU));

    auto* cpu_backend = registry.get_backend(Device::Type::CPU);
    ASSERT_NE(cpu_backend, nullptr);
    EXPECT_EQ(cpu_backend->name(), "cpu");
}

TEST(BackendRegistryTest, UnavailableBackend) {
    auto& registry = backend_registry();

    // Should return nullptr for unregistered backends
    auto* invalid = registry.get_backend(static_cast<Device::Type>(999));
    EXPECT_EQ(invalid, nullptr);
}
```

### 2. Operation Dispatch Tests

```cpp
TEST(OperationRegistryTest, DispatchToCorrectBackend) {
    Tensor cpu_a = randn({10, 10}, DType::Float32, Device::cpu());
    Tensor cpu_b = randn({10, 10}, DType::Float32, Device::cpu());

    // Should dispatch to CPU backend
    Tensor result = add(cpu_a, cpu_b);

    EXPECT_EQ(result.device().type, Device::Type::CPU);
    EXPECT_EQ(result.shape(), cpu_a.shape());
}

TEST(OperationRegistryTest, FallbackToCPU) {
    if (!backend_registry().has_backend(Device::Type::CUDA)) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor cuda_a = randn({10, 10}, DType::Float32, Device::cuda(0));

    // Operation not implemented in CUDA - should fall back to CPU
    Tensor result = unimplemented_cuda_op(cuda_a);

    // Result should be back on CUDA
    EXPECT_EQ(result.device().type, Device::Type::CUDA);
}
```

### 3. Performance Tests

```cpp
BENCHMARK(BM_DispatchOverhead) {
    Tensor a = randn({1000, 1000}, DType::Float32, Device::cpu());
    Tensor b = randn({1000, 1000}, DType::Float32, Device::cpu());

    for (auto _ : state) {
        Tensor result = add(a, b);
        benchmark::DoNotOptimize(result);
    }
}
```

## Migration Guide

### Step 1: Update Backend Implementations

For each backend (CPU, CUDA, ROCm, etc.):

1. Add `register_operations()` method
2. Add `device_type()` and `supported_operations()` methods
3. Remove hardcoded dispatch logic
4. Use registration macros

### Step 2: Update Library Initialization

1. Create `init.cpp` with backend registration
2. Add static initializer
3. Test backend availability

### Step 3: Clean Frontend Code

1. Remove all `#ifdef` checks from frontend
2. Ensure all operations use `Dispatcher::dispatch()`
3. Add device compatibility checks where needed

### Step 4: Add Fallback Policies

1. Identify operations not implemented in all backends
2. Register CPU fallbacks
3. Add logging for fallback usage

## Benefits Summary

1. **Clean Separation**: Frontend code has zero backend dependencies
2. **Extensibility**: New backends can be added without modifying core code
3. **Type Safety**: Compile-time type checking for operation signatures
4. **Performance**: Minimal dispatch overhead with caching
5. **Robustness**: Graceful fallback for missing implementations
6. **Testability**: Easy to mock backends for testing
7. **Maintainability**: Clear ownership boundaries

## Future Enhancements

1. **Dynamic Backend Loading**: Load backends as plugins at runtime
2. **Operation Fusion**: Automatic fusion of operations at dispatch layer
3. **Multi-Device Execution**: Automatic work distribution across devices
4. **JIT Compilation**: Runtime kernel generation for custom operations
5. **Profiling Integration**: Built-in performance monitoring at dispatch layer

## Conclusion

The proposed runtime dispatcher architecture provides a robust, extensible, and performant solution for eliminating compile-time backend checks while maintaining clean separation of concerns. The design leverages existing infrastructure in Tenzor and requires minimal changes to achieve full backend abstraction.
