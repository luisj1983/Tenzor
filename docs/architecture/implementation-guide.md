# Runtime Dispatcher Implementation Guide

This guide provides step-by-step instructions for implementing the runtime operation dispatcher architecture in Tenzor.

## Phase 1: Core Infrastructure

### Step 1.1: Enhance OperationRegistry

**File**: `src/backend/registry.cpp`

Add fallback support to the existing `OperationRegistry`:

```cpp
#include "tenzor/backend/registry.hpp"
#include <iostream>
#include <sstream>

namespace tenzor {

auto OperationRegistry::register_fallback(
    std::string_view op_name,
    Device::Type device_type,
    Device::Type fallback_type) -> void {

    std::unique_lock lock(mutex_);
    fallbacks_[std::string(op_name)][device_type] = fallback_type;
}

auto OperationRegistry::get_kernel_with_fallback(
    std::string_view op_name,
    Device::Type device_type) const
    -> std::pair<KernelFunction, Device::Type> {

    std::shared_lock lock(mutex_);

    // First, try direct lookup
    auto op_it = kernels_.find(std::string(op_name));
    if (op_it != kernels_.end()) {
        auto kernel_it = op_it->second.find(device_type);
        if (kernel_it != op_it->second.end()) {
            return {kernel_it->second, device_type};
        }
    }

    // Not found - check fallback
    auto fallback_op_it = fallbacks_.find(std::string(op_name));
    if (fallback_op_it != fallbacks_.end()) {
        auto fallback_device_it = fallback_op_it->second.find(device_type);
        if (fallback_device_it != fallback_op_it->second.end()) {
            Device::Type fallback_type = fallback_device_it->second;

            // Lookup kernel for fallback device
            if (op_it != kernels_.end()) {
                auto kernel_it = op_it->second.find(fallback_type);
                if (kernel_it != op_it->second.end()) {
                    // Log fallback usage
                    std::cerr << "Warning: Operation '" << op_name
                              << "' falling back from "
                              << static_cast<int>(device_type)
                              << " to "
                              << static_cast<int>(fallback_type)
                              << std::endl;

                    return {kernel_it->second, fallback_type};
                }
            }
        }
    }

    // No kernel found
    std::ostringstream error;
    error << "Operation '" << op_name
          << "' not implemented for device type "
          << static_cast<int>(device_type);
    throw std::runtime_error(error.str());
}

} // namespace tenzor
```

**File**: `include/tenzor/backend/registry.hpp`

Add declarations to header:

```cpp
class OperationRegistry {
public:
    // ... existing methods ...

    // NEW: Fallback support
    auto register_fallback(std::string_view op_name,
                          Device::Type device_type,
                          Device::Type fallback_type) -> void;

    auto get_kernel_with_fallback(std::string_view op_name,
                                  Device::Type device_type) const
        -> std::pair<KernelFunction, Device::Type>;

private:
    mutable std::shared_mutex mutex_;

    std::unordered_map<
        std::string,
        std::unordered_map<Device::Type, KernelFunction>
    > kernels_;

    // NEW: Fallback mappings
    std::unordered_map<
        std::string,
        std::unordered_map<Device::Type, Device::Type>
    > fallbacks_;
};
```

### Step 1.2: Create BackendRegistry

**File**: `include/tenzor/backend/backend_registry.hpp` (new)

```cpp
#pragma once

#include <memory>
#include <unordered_map>
#include <shared_mutex>
#include <vector>
#include "backend.hpp"
#include "../core/device.hpp"

namespace tenzor {

class BackendRegistry {
public:
    static auto instance() -> BackendRegistry&;

    auto register_backend(Device::Type type,
                         std::unique_ptr<Backend> backend) -> void;

    auto get_backend(Device::Type type) -> Backend*;

    auto has_backend(Device::Type type) const -> bool;

    auto available_backends() const -> std::vector<Device::Type>;

private:
    BackendRegistry() = default;

    mutable std::shared_mutex mutex_;
    std::unordered_map<Device::Type, std::unique_ptr<Backend>> backends_;
};

inline auto backend_registry() -> BackendRegistry& {
    return BackendRegistry::instance();
}

} // namespace tenzor
```

**File**: `src/backend/backend_registry.cpp` (new)

```cpp
#include "tenzor/backend/backend_registry.hpp"
#include <iostream>

namespace tenzor {

auto BackendRegistry::instance() -> BackendRegistry& {
    static BackendRegistry instance;
    return instance;
}

auto BackendRegistry::register_backend(
    Device::Type type,
    std::unique_ptr<Backend> backend) -> void {

    if (!backend) {
        std::cerr << "Warning: Attempted to register null backend for type "
                  << static_cast<int>(type) << std::endl;
        return;
    }

    std::unique_lock lock(mutex_);

    if (backends_.find(type) != backends_.end()) {
        std::cerr << "Warning: Backend for type "
                  << static_cast<int>(type)
                  << " already registered, replacing" << std::endl;
    }

    std::cout << "Registering backend: " << backend->name()
              << " for device type " << static_cast<int>(type)
              << std::endl;

    backends_[type] = std::move(backend);
}

auto BackendRegistry::get_backend(Device::Type type) -> Backend* {
    std::shared_lock lock(mutex_);

    auto it = backends_.find(type);
    if (it != backends_.end()) {
        return it->second.get();
    }

    return nullptr;
}

auto BackendRegistry::has_backend(Device::Type type) const -> bool {
    std::shared_lock lock(mutex_);
    return backends_.find(type) != backends_.end();
}

auto BackendRegistry::available_backends() const -> std::vector<Device::Type> {
    std::shared_lock lock(mutex_);

    std::vector<Device::Type> types;
    types.reserve(backends_.size());

    for (const auto& [type, backend] : backends_) {
        types.push_back(type);
    }

    return types;
}

} // namespace tenzor
```

### Step 1.3: Create Registration Helpers

**File**: `include/tenzor/backend/registration.hpp` (new)

```cpp
#pragma once

#include "registry.hpp"
#include "../core/device.hpp"
#include <functional>

namespace tenzor {

// Kernel wrapper for single-output operations
template<typename KernelFn>
auto wrap_single_output(KernelFn&& kernel) -> KernelFunction {
    return [kernel = std::forward<KernelFn>(kernel)](
        std::span<const Tensor> inputs,
        const OpAttributes& attrs) -> std::vector<Tensor> {
        return {kernel(inputs, attrs)};
    };
}

// Registration macro for single-output operations
#define TENZOR_REGISTER_KERNEL(backend_type, op_name, kernel_fn)         \
    do {                                                                  \
        registry.register_kernel(                                         \
            op_name,                                                      \
            backend_type,                                                 \
            [](std::span<const Tensor> inputs,                           \
               const OpAttributes& attrs) -> std::vector<Tensor> {       \
                return {kernel_fn(inputs, attrs)};                       \
            }                                                             \
        );                                                                \
    } while(0)

// Registration macro for multi-output operations
#define TENZOR_REGISTER_KERNEL_MULTI(backend_type, op_name, kernel_fn)   \
    do {                                                                  \
        registry.register_kernel(op_name, backend_type, kernel_fn);      \
    } while(0)

// Fallback registration macro
#define TENZOR_REGISTER_FALLBACK(op_name, device_type, fallback_type)    \
    do {                                                                  \
        registry.register_fallback(op_name, device_type, fallback_type); \
    } while(0)

} // namespace tenzor
```

## Phase 2: Update Backend Interface

### Step 2.1: Enhance Backend Base Class

**File**: `include/tenzor/backend/backend.hpp`

Add new virtual methods:

```cpp
class Backend {
public:
    virtual ~Backend() = default;

    // Existing methods
    virtual auto name() const -> std::string_view = 0;
    virtual auto device_count() const -> int32_t = 0;
    virtual auto is_available() const -> bool = 0;
    virtual auto allocate(size_t bytes, int32_t device_id) -> void* = 0;
    virtual auto deallocate(void* ptr) -> void = 0;
    virtual auto copy(void* dst, const void* src, size_t bytes,
                     CopyKind kind) -> void = 0;
    virtual auto synchronize(int32_t device_id) -> void = 0;
    virtual auto create_stream(int32_t device_id) -> StreamHandle = 0;
    virtual auto destroy_stream(StreamHandle stream) -> void = 0;
    virtual auto synchronize_stream(StreamHandle stream) -> void = 0;

    // NEW: Device type this backend handles
    virtual auto device_type() const -> Device::Type = 0;

    // NEW: Registration hook
    virtual auto register_operations(OperationRegistry& registry) -> void = 0;

    // NEW: Supported operations list
    virtual auto supported_operations() const -> std::vector<std::string> = 0;

    // Modified: dispatch now uses OperationRegistry internally
    virtual auto dispatch(const std::string& op_name,
                         std::span<const Tensor> inputs,
                         const OpAttributes& attrs) -> std::vector<Tensor>;
};
```

**File**: `src/backend/backend.cpp` (new)

Implement default dispatch behavior:

```cpp
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/registry.hpp"

namespace tenzor {

auto Backend::dispatch(
    const std::string& op_name,
    std::span<const Tensor> inputs,
    const OpAttributes& attrs) -> std::vector<Tensor> {

    auto& registry = operation_registry();

    // Get kernel with fallback support
    auto [kernel, actual_device] = registry.get_kernel_with_fallback(
        op_name,
        this->device_type()
    );

    // Check if we need to use fallback
    if (actual_device != this->device_type()) {
        // Fallback to different device
        std::vector<Tensor> moved_inputs;
        moved_inputs.reserve(inputs.size());

        Device fallback_device;
        fallback_device.type = actual_device;
        fallback_device.index = 0;

        // Move tensors to fallback device
        for (const auto& tensor : inputs) {
            moved_inputs.push_back(tensor.to(fallback_device));
        }

        // Execute on fallback device
        auto results = kernel(moved_inputs, attrs);

        // Move results back to original device
        std::vector<Tensor> final_results;
        final_results.reserve(results.size());

        Device original_device;
        original_device.type = this->device_type();
        original_device.index = inputs[0].device().index;

        for (auto& result : results) {
            final_results.push_back(result.to(original_device));
        }

        return final_results;
    }

    // Execute on requested device
    return kernel(inputs, attrs);
}

} // namespace tenzor
```

## Phase 3: Update CPU Backend

### Step 3.1: Refactor CPUBackend

**File**: `src/backends/cpu/cpu_backend.cpp`

Update to use registration system:

```cpp
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/registry.hpp"
#include "tenzor/backend/registration.hpp"
#include <cstring>

namespace tenzor {
namespace cpu {

// Kernel function adapters (convert old signatures to new)
auto add_kernel_wrapper(std::span<const Tensor> inputs,
                       const OpAttributes& attrs) -> Tensor {
    if (inputs.size() != 2) {
        throw std::runtime_error("add requires 2 inputs");
    }
    return add_kernel(inputs[0], inputs[1]);
}

auto matmul_kernel_wrapper(std::span<const Tensor> inputs,
                          const OpAttributes& attrs) -> Tensor {
    if (inputs.size() != 2) {
        throw std::runtime_error("matmul requires 2 inputs");
    }
    return matmul_kernel(inputs[0], inputs[1]);
}

auto relu_kernel_wrapper(std::span<const Tensor> inputs,
                        const OpAttributes& attrs) -> Tensor {
    if (inputs.size() != 1) {
        throw std::runtime_error("relu requires 1 input");
    }
    return relu_kernel(inputs[0]);
}

auto conv2d_kernel_wrapper(std::span<const Tensor> inputs,
                          const OpAttributes& attrs) -> Tensor {
    if (inputs.size() < 2 || inputs.size() > 3) {
        throw std::runtime_error("conv2d requires 2 or 3 inputs");
    }

    const Tensor* bias = inputs.size() == 3 ? &inputs[2] : nullptr;

    // Parse attributes
    auto get_attr = [&](const std::string& key, int64_t default_val) {
        auto it = attrs.find(key);
        return it != attrs.end() ? std::stoll(it->second) : default_val;
    };

    int64_t stride = get_attr("stride", 1);
    int64_t padding = get_attr("padding", 0);
    int64_t dilation = get_attr("dilation", 1);
    int64_t groups = get_attr("groups", 1);

    return conv2d_forward_kernel(
        inputs[0], inputs[1], bias,
        stride, padding, dilation, groups
    );
}

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
        return true;
    }

    auto supported_operations() const -> std::vector<std::string> override {
        return {
            // Math operations
            "add", "sub", "mul", "div",
            "matmul", "bmm", "dot",
            "pow", "exp", "log", "sqrt",
            "sin", "cos", "tan", "tanh",
            "abs", "neg", "sign",

            // Activations
            "relu", "sigmoid", "gelu", "swish",
            "leaky_relu", "softmax", "log_softmax",

            // Convolutions
            "conv2d",

            // Reductions
            "sum", "mean", "max", "min",
            "argmax", "argsort",

            // Transforms
            "contiguous", "clone", "fill",
            "reshape", "transpose", "permute",
            "squeeze", "unsqueeze",

            // Comparisons
            "eq", "ne", "lt", "le", "gt", "ge",

            // Creation
            "zeros", "ones", "rand", "randn",
        };
    }

    auto register_operations(OperationRegistry& registry) -> void override {
        // Math operations
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "add", cpu::add_kernel_wrapper);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "sub", cpu::sub_kernel_wrapper);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "mul", cpu::mul_kernel_wrapper);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "div", cpu::div_kernel_wrapper);

        // Matrix operations
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "matmul", cpu::matmul_kernel_wrapper);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "bmm", cpu::bmm_kernel_wrapper);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "dot", cpu::dot_kernel_wrapper);

        // Activations
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "relu", cpu::relu_kernel_wrapper);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "sigmoid", cpu::sigmoid_kernel_wrapper);
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "tanh", cpu::tanh_kernel_wrapper);

        // Convolutions
        TENZOR_REGISTER_KERNEL(Device::Type::CPU, "conv2d", cpu::conv2d_kernel_wrapper);

        // ... register all other operations ...
    }

    // Memory management (unchanged)
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
        return nullptr;
    }

    auto destroy_stream(StreamHandle stream) -> void override {
        // No-op for CPU
    }

    auto synchronize_stream(StreamHandle stream) -> void override {
        // No-op for CPU
    }
};

} // namespace tenzor

// Backend factory
extern "C" auto create_cpu_backend() -> std::unique_ptr<tenzor::Backend> {
    return std::make_unique<tenzor::CPUBackend>();
}
```

## Phase 4: Update CUDA Backend

### Step 4.1: Refactor CUDABackend

**File**: `src/backends/cuda/cuda_backend.cpp`

```cpp
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/registry.hpp"
#include "tenzor/backend/registration.hpp"

#ifdef TENZOR_HAS_CUDA
#include <cuda_runtime.h>

namespace tenzor {
namespace cuda {

// Kernel wrappers (similar to CPU)
auto add_kernel_wrapper(std::span<const Tensor> inputs,
                       const OpAttributes& attrs) -> Tensor;

auto matmul_kernel_wrapper(std::span<const Tensor> inputs,
                          const OpAttributes& attrs) -> Tensor;

// ... more wrappers ...

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
            "relu", "sigmoid", "tanh",
            "conv2d",
            // ... operations with CUDA kernels
        };
    }

    auto register_operations(OperationRegistry& registry) -> void override {
        // Register CUDA kernels
        TENZOR_REGISTER_KERNEL(Device::Type::CUDA, "add", cuda::add_kernel_wrapper);
        TENZOR_REGISTER_KERNEL(Device::Type::CUDA, "matmul", cuda::matmul_kernel_wrapper);
        TENZOR_REGISTER_KERNEL(Device::Type::CUDA, "conv2d", cuda::conv2d_kernel_wrapper);

        // Register CPU fallback for operations not yet implemented
        TENZOR_REGISTER_FALLBACK("advanced_transform", Device::Type::CUDA, Device::Type::CPU);
        TENZOR_REGISTER_FALLBACK("custom_op", Device::Type::CUDA, Device::Type::CPU);
    }

    // Memory management
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

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_HAS_CUDA

// Factory function
extern "C" auto create_cuda_backend() -> std::unique_ptr<tenzor::Backend> {
#ifdef TENZOR_HAS_CUDA
    return std::make_unique<tenzor::cuda::CUDABackend>();
#else
    return nullptr;
#endif
}
```

## Phase 5: Library Initialization

### Step 5.1: Create Initialization System

**File**: `include/tenzor/core/init.hpp` (new)

```cpp
#pragma once

namespace tenzor {

// Initialize all available backends
auto initialize_backends() -> void;

// Check if library is initialized
auto is_initialized() -> bool;

} // namespace tenzor
```

**File**: `src/core/init.cpp` (new)

```cpp
#include "tenzor/core/init.hpp"
#include "tenzor/backend/backend_registry.hpp"
#include "tenzor/backend/registry.hpp"
#include <iostream>

namespace tenzor {

// Backend factory declarations
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

static bool g_initialized = false;

auto initialize_backends() -> void {
    if (g_initialized) {
        return;
    }

    std::cout << "Initializing Tenzor backends..." << std::endl;

    auto& backend_reg = backend_registry();
    auto& op_reg = operation_registry();

    // CPU (always available)
    {
        auto backend = create_cpu_backend();
        if (backend && backend->is_available()) {
            std::cout << "  Initializing CPU backend" << std::endl;
            backend->register_operations(op_reg);
            backend_reg.register_backend(Device::Type::CPU, std::move(backend));
        }
    }

    // CUDA
    #ifdef TENZOR_HAS_CUDA
    {
        auto backend = create_cuda_backend();
        if (backend && backend->is_available()) {
            std::cout << "  Initializing CUDA backend ("
                      << backend->device_count() << " devices)" << std::endl;
            backend->register_operations(op_reg);
            backend_reg.register_backend(Device::Type::CUDA, std::move(backend));
        } else {
            std::cout << "  CUDA backend unavailable" << std::endl;
        }
    }
    #endif

    // ROCm
    #ifdef TENZOR_HAS_ROCM
    {
        auto backend = create_rocm_backend();
        if (backend && backend->is_available()) {
            std::cout << "  Initializing ROCm backend" << std::endl;
            backend->register_operations(op_reg);
            backend_reg.register_backend(Device::Type::ROCm, std::move(backend));
        }
    }
    #endif

    // OneAPI
    #ifdef TENZOR_HAS_ONEAPI
    {
        auto backend = create_oneapi_backend();
        if (backend && backend->is_available()) {
            std::cout << "  Initializing OneAPI backend" << std::endl;
            backend->register_operations(op_reg);
            backend_reg.register_backend(Device::Type::OneAPI, std::move(backend));
        }
    }
    #endif

    // Vulkan
    #ifdef TENZOR_HAS_VULKAN
    {
        auto backend = create_vulkan_backend();
        if (backend && backend->is_available()) {
            std::cout << "  Initializing Vulkan backend" << std::endl;
            backend->register_operations(op_reg);
            backend_reg.register_backend(Device::Type::Vulkan, std::move(backend));
        }
    }
    #endif

    g_initialized = true;
    std::cout << "Tenzor initialization complete" << std::endl;
}

auto is_initialized() -> bool {
    return g_initialized;
}

// Automatic initialization
struct BackendInitializer {
    BackendInitializer() {
        initialize_backends();
    }
};

static BackendInitializer g_backend_initializer;

} // namespace tenzor
```

## Phase 6: Testing

### Step 6.1: Backend Registration Tests

**File**: `tests/backend/test_registry.cpp`

```cpp
#include <gtest/gtest.h>
#include "tenzor/backend/backend_registry.hpp"
#include "tenzor/backend/registry.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;

TEST(BackendRegistryTest, CPUBackendAvailable) {
    auto& registry = backend_registry();
    ASSERT_TRUE(registry.has_backend(Device::Type::CPU));

    auto* backend = registry.get_backend(Device::Type::CPU);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->name(), "cpu");
    EXPECT_TRUE(backend->is_available());
}

TEST(BackendRegistryTest, ListAvailableBackends) {
    auto& registry = backend_registry();
    auto backends = registry.available_backends();

    EXPECT_GE(backends.size(), 1);  // At least CPU

    for (auto type : backends) {
        auto* backend = registry.get_backend(type);
        ASSERT_NE(backend, nullptr);
        std::cout << "Available backend: " << backend->name() << std::endl;
    }
}

TEST(OperationRegistryTest, OperationsRegistered) {
    auto& registry = operation_registry();

    // CPU operations should be registered
    EXPECT_TRUE(registry.has_kernel("add", Device::Type::CPU));
    EXPECT_TRUE(registry.has_kernel("matmul", Device::Type::CPU));
    EXPECT_TRUE(registry.has_kernel("relu", Device::Type::CPU));
}

TEST(OperationDispatchTest, CPUAdd) {
    Tensor a = tenzor::zeros({10, 10}, DType::Float32, Device::cpu());
    Tensor b = tenzor::ones({10, 10}, DType::Float32, Device::cpu());

    Tensor result = add(a, b);

    EXPECT_EQ(result.device().type, Device::Type::CPU);
    EXPECT_EQ(result.shape()[0], 10);
    EXPECT_EQ(result.shape()[1], 10);

    // Check values
    auto* data = result.data<float>();
    for (int64_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}
```

### Step 6.2: Fallback Tests

**File**: `tests/backend/test_fallback.cpp`

```cpp
#include <gtest/gtest.h>
#include "tenzor/backend/registry.hpp"
#include "tenzor/core/tensor.hpp"

using namespace tenzor;

TEST(FallbackTest, RegisterFallback) {
    auto& registry = operation_registry();

    // Register a fake operation for CPU only
    registry.register_kernel(
        "test_op",
        Device::Type::CPU,
        [](std::span<const Tensor> inputs, const OpAttributes&) {
            return std::vector<Tensor>{inputs[0]};
        }
    );

    // Register fallback from CUDA to CPU
    registry.register_fallback("test_op", Device::Type::CUDA, Device::Type::CPU);

    // Should be able to get kernel with fallback
    auto [kernel, device] = registry.get_kernel_with_fallback(
        "test_op",
        Device::Type::CUDA
    );

    EXPECT_EQ(device, Device::Type::CPU);
}

TEST(FallbackTest, MissingOperationThrows) {
    auto& registry = operation_registry();

    EXPECT_THROW(
        registry.get_kernel_with_fallback("nonexistent_op", Device::Type::CPU),
        std::runtime_error
    );
}
```

## Phase 7: CMake Integration

### Step 7.1: Update CMakeLists.txt

```cmake
# Add new source files
set(TENZOR_CORE_SOURCES
    # ... existing sources ...
    src/core/init.cpp
    src/backend/backend_registry.cpp
    src/backend/backend.cpp
)

# Add new headers
set(TENZOR_CORE_HEADERS
    # ... existing headers ...
    include/tenzor/core/init.hpp
    include/tenzor/backend/backend_registry.hpp
    include/tenzor/backend/registration.hpp
)

# Backends are compiled conditionally
if(TENZOR_USE_CUDA)
    add_library(tenzor_cuda
        src/backends/cuda/cuda_backend.cpp
        # ... CUDA sources ...
    )
    target_link_libraries(tenzor_cuda CUDA::cudart)
endif()

if(TENZOR_USE_ROCM)
    add_library(tenzor_rocm
        src/backends/rocm/rocm_backend.cpp
        # ... ROCm sources ...
    )
endif()

# Link backends to main library
target_link_libraries(tenzor_core
    PRIVATE
    $<TARGET_NAME_IF_EXISTS:tenzor_cuda>
    $<TARGET_NAME_IF_EXISTS:tenzor_rocm>
)
```

## Summary

This implementation guide provides:

1. **Core infrastructure** for operation and backend registration
2. **Backend refactoring** to use the registration system
3. **Automatic initialization** at library load time
4. **Fallback support** for graceful degradation
5. **Comprehensive tests** for validation
6. **CMake integration** for build system

The architecture maintains backward compatibility while providing a clean path forward for eliminating compile-time backend checks from frontend code.
