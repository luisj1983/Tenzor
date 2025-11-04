# Backend Registration and Operation Dispatch Architecture Analysis

**Project**: Tenzor Deep Learning Framework
**Date**: 2025-11-01
**Analyzed Components**: Backend system, operation registry, dispatch mechanism

---

## Executive Summary

Tenzor implements a **dual-layer dispatch architecture** combining:
1. **Backend-level dispatch** - Each backend (CPU, CUDA, ROCm, etc.) has its own internal dispatch mechanism
2. **Global operation registry** - A centralized kernel registry that routes operations based on device type

This analysis reveals the current architecture patterns and identifies required changes for implementing runtime dispatch functionality.

---

## 1. Current Architecture Overview

### 1.1 Component Hierarchy

```
User Code
    ↓
Dispatcher::dispatch()                    [Global entry point]
    ↓
Backend* backend = get_backend()          [Backend lookup by device type]
    ↓
backend->dispatch(op_name, inputs, attrs) [Backend-specific dispatch]
    ↓
Kernel Implementation                     [CPU/CUDA/ROCm kernels]
```

### 1.2 Key Components

| Component | File Location | Purpose |
|-----------|--------------|---------|
| `Backend` (abstract) | `/home/lee/Projects/Tenzor/include/tenzor/backend/backend.hpp` | Base interface for all backends |
| `BackendLoader` | `/home/lee/Projects/Tenzor/include/tenzor/backend/loader.hpp` | Dynamic backend loading and registration |
| `OperationRegistry` | `/home/lee/Projects/Tenzor/include/tenzor/backend/registry.hpp` | Global kernel registry (op_name → device → kernel) |
| `Dispatcher` | `/home/lee/Projects/Tenzor/include/tenzor/backend/dispatch.hpp` | Central operation routing |
| `CPUBackend` | `/home/lee/Projects/Tenzor/src/backends/cpu/cpu_backend.cpp` | CPU backend implementation |
| `CUDABackend` | `/home/lee/Projects/Tenzor/src/backends/cuda/cuda_backend.cpp` | CUDA backend implementation |

---

## 2. Backend Registration Mechanism

### 2.1 Registration Flow

The initialization process in `/home/lee/Projects/Tenzor/src/core/init.cpp` follows this pattern:

```cpp
// Step 1: Load backend from shared library (.so file)
auto result = loader.load_backend("tenzor_backend_cpu.so");

// Step 2: Register backend instance
loader.register_backend(backend->name(), std::move(backend_unique));

// Step 3: Register individual operations to OperationRegistry
registry.register_kernel("add", Device::Type::CPU,
    [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return cpu_backend->dispatch("add", inputs, attrs);
    });
```

### 2.2 Backend Factory Pattern

Each backend exports a C-style factory function:

**File**: `/home/lee/Projects/Tenzor/src/backends/cpu/cpu_backend.cpp` (lines 953-957)
```cpp
extern "C" {
    auto create_backend() -> std::unique_ptr<Backend> {
        return std::make_unique<CPUBackend>();
    }
}
```

### 2.3 Backend Storage

**BackendLoader** maintains two registries:

**File**: `/home/lee/Projects/Tenzor/include/tenzor/backend/loader.hpp` (lines 163-164)
```cpp
std::unordered_map<std::string, std::unique_ptr<Backend>> backends_;     // By name
std::unordered_map<Device::Type, Backend*> device_to_backend_;           // By device type
```

**Access Methods**:
```cpp
Backend* get_backend(std::string_view name);    // Lookup by name ("cpu", "cuda")
Backend* get_backend(Device::Type type);        // Lookup by device type
```

---

## 3. Operation Dispatch Mechanism

### 3.1 Two-Level Dispatch Architecture

#### Level 1: Global Dispatcher (Centralized Routing)

**File**: `/home/lee/Projects/Tenzor/src/backend/dispatch.cpp` (lines 8-23)
```cpp
auto Dispatcher::dispatch(const std::string& op_name,
                         std::span<const Tensor> inputs,
                         const OpAttributes& attrs) -> std::vector<Tensor> {
    // Check device compatibility
    if (!check_device_compatibility(inputs)) {
        throw DeviceException("All input tensors must be on the same device");
    }

    // Get the appropriate backend and dispatch to it
    Backend* backend = get_backend(inputs);
    if (!backend) {
        throw TenzorException("No backend available for tensors");
    }

    return backend->dispatch(op_name, inputs, attrs);
}
```

**Key Points**:
- Entry point for all tensor operations
- Validates device compatibility
- Routes to appropriate backend based on tensor device type
- **Currently goes directly to backend, bypassing OperationRegistry**

#### Level 2: Backend Internal Dispatch

Each backend implements its own `dispatch()` method with a massive if-else chain:

**Example from CPUBackend** (`/home/lee/Projects/Tenzor/src/backends/cpu/cpu_backend.cpp`, lines 150-949):

```cpp
auto dispatch(const std::string& op_name,
             std::span<const Tensor> inputs,
             const OpAttributes& attrs) -> std::vector<Tensor> override {

    // Massive if-else chain
    if (op_name == "add") {
        return {cpu::add_kernel(inputs[0], inputs[1])};
    }
    else if (op_name == "sub") {
        return {cpu::sub_kernel(inputs[0], inputs[1])};
    }
    else if (op_name == "mul") {
        return {cpu::mul_kernel(inputs[0], inputs[1])};
    }
    // ... 100+ more operations ...
    else {
        throw std::runtime_error("Unknown operation");
    }
}
```

**Similar pattern in CUDABackend** (`/home/lee/Projects/Tenzor/src/backends/cuda/cuda_backend.cpp`, lines 217-1006):
- 800+ lines of if-else dispatch logic
- Parses string attributes
- Calls CUDA kernel implementations

### 3.2 OperationRegistry (Currently Unused for Dispatch)

**File**: `/home/lee/Projects/Tenzor/include/tenzor/backend/registry.hpp` (lines 75-163)

```cpp
class OperationRegistry {
public:
    // Register kernel: op_name + device_type → kernel_function
    auto register_kernel(std::string_view op_name,
                        Device::Type device_type,
                        KernelFunction kernel) -> void;

    // Dispatch to registered kernel
    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor>;

private:
    // Two-level map: operation -> device -> kernel
    std::unordered_map<
        std::string,
        std::unordered_map<Device::Type, KernelFunction>
    > kernels_;
};
```

**Current State**:
- Operations ARE registered to OperationRegistry during initialization
- Registration creates lambda wrappers that call `backend->dispatch()`
- **But OperationRegistry::dispatch() is NEVER called** - operations go directly to backends

**Evidence from init.cpp** (line 63-66):
```cpp
registry.register_kernel("add", Device::Type::CPU,
    [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return cpu_backend->dispatch("add", inputs, attrs);  // Still calls backend!
    });
```

---

## 4. Architecture Pattern Analysis

### 4.1 Current Pattern: Backend-Centric Dispatch

```
Dispatcher::dispatch()
    ↓
get_backend(tensors[0].device())
    ↓
Backend::dispatch()  ← Large if-else chain (800+ lines)
    ↓
cpu::kernel() or cuda::kernel()
```

**Characteristics**:
- Each backend duplicates dispatch logic
- If-else chains become massive (800+ lines per backend)
- OperationRegistry exists but is bypassed
- Hard to add new operations (modify each backend's dispatch method)

### 4.2 Problems with Current Architecture

1. **Code Duplication**: Same dispatch logic repeated in every backend
2. **Maintainability**: Adding operations requires modifying multiple files
3. **String Parsing**: Attribute parsing duplicated across backends
4. **No Runtime Lookup**: Can't dynamically check operation availability
5. **Unused Infrastructure**: OperationRegistry is populated but never used

---

## 5. Runtime Dispatch Requirements

### 5.1 What "Runtime Dispatch" Means

Currently, operations are dispatched through:
1. Hardcoded if-else chains in each backend
2. Compile-time function bindings

**Runtime dispatch** would enable:
- Dynamic operation discovery at runtime
- Query which operations are available for a device
- Add/remove operations without recompilation
- Plugin-based operation extensions
- Fallback strategies when operations are unavailable

### 5.2 Required Changes for Runtime Dispatch

#### Change 1: Use OperationRegistry as Primary Dispatcher

**Current Flow**:
```cpp
Dispatcher → Backend::dispatch() → if-else chain → kernel
```

**Target Flow**:
```cpp
Dispatcher → OperationRegistry::dispatch() → registered kernel function → backend kernel
```

**Implementation**:
```cpp
// In Dispatcher::dispatch()
auto Dispatcher::dispatch(const std::string& op_name,
                         std::span<const Tensor> inputs,
                         const OpAttributes& attrs) -> std::vector<Tensor> {
    // Use OperationRegistry instead of Backend
    return operation_registry().dispatch(op_name, inputs, attrs);
}
```

#### Change 2: Register Direct Kernel Functions

**Current Registration** (calls backend dispatch):
```cpp
registry.register_kernel("add", Device::Type::CPU,
    [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return cpu_backend->dispatch("add", inputs, attrs);  // Indirect
    });
```

**Target Registration** (direct kernel binding):
```cpp
registry.register_kernel("add", Device::Type::CPU,
    [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return {cpu::add_kernel(inputs[0], inputs[1])};  // Direct
    });
```

#### Change 3: Eliminate Backend::dispatch()

Backends should provide:
- Memory management (allocate, deallocate, copy)
- Stream management
- **NOT** operation dispatch

**Target Backend Interface**:
```cpp
class Backend {
public:
    // Keep these
    virtual auto allocate(size_t bytes, int32_t device_id) -> void* = 0;
    virtual auto deallocate(void* ptr) -> void = 0;
    virtual auto synchronize(int32_t device_id) -> void = 0;

    // REMOVE this
    // virtual auto dispatch(...) -> std::vector<Tensor> = 0;
};
```

#### Change 4: Add Runtime Query API

Enable runtime operation introspection:

```cpp
// Check if operation is available for device
bool has_operation(std::string_view op_name, Device::Type device_type);

// List all available operations for device
std::vector<std::string> available_operations(Device::Type device_type);

// Get operation metadata
struct OperationInfo {
    std::string name;
    std::vector<Device::Type> supported_devices;
    std::vector<std::string> required_attrs;
};
OperationInfo get_operation_info(std::string_view op_name);
```

#### Change 5: Implement Fallback Mechanism

When operation unavailable on device:

```cpp
auto OperationRegistry::dispatch_with_fallback(
    const std::string& op_name,
    std::span<const Tensor> inputs,
    const OpAttributes& attrs) -> std::vector<Tensor> {

    auto device_type = inputs[0].device().type;

    // Try primary device
    if (has_kernel(op_name, device_type)) {
        return dispatch(op_name, inputs, attrs);
    }

    // Fallback to CPU
    if (device_type != Device::Type::CPU && has_kernel(op_name, Device::Type::CPU)) {
        auto cpu_inputs = transfer_to_cpu(inputs);
        auto cpu_result = dispatch(op_name, cpu_inputs, attrs);
        return transfer_to_device(cpu_result, device_type);
    }

    throw OperationNotSupportedException(op_name, device_type);
}
```

---

## 6. Code Snippets Showing Current Patterns

### 6.1 Backend Dispatch Example (CPU)

**File**: `/home/lee/Projects/Tenzor/src/backends/cpu/cpu_backend.cpp` (lines 169-198)

```cpp
if (op_name == "add") {
    if (inputs.size() != 2) {
        throw std::invalid_argument("add operation requires exactly 2 inputs");
    }
    return {cpu::add_kernel(inputs[0], inputs[1])};
}
else if (op_name == "sub") {
    if (inputs.size() != 2) {
        throw std::invalid_argument("sub operation requires exactly 2 inputs");
    }
    return {cpu::sub_kernel(inputs[0], inputs[1])};
}
else if (op_name == "mul") {
    if (inputs.size() != 2) {
        throw std::invalid_argument("mul operation requires exactly 2 inputs");
    }
    return {cpu::mul_kernel(inputs[0], inputs[1])};
}
// ... continues for 100+ operations ...
```

### 6.2 Backend Dispatch Example (CUDA)

**File**: `/home/lee/Projects/Tenzor/src/backends/cuda/cuda_backend.cpp` (lines 258-287)

```cpp
if (op_name == "add") {
    if (inputs.size() != 2) {
        throw std::invalid_argument("add operation requires exactly 2 inputs");
    }
    return {cuda::add_kernel(inputs[0], inputs[1], stream)};
}
else if (op_name == "sub") {
    if (inputs.size() != 2) {
        throw std::invalid_argument("sub operation requires exactly 2 inputs");
    }
    return {cuda::sub_kernel(inputs[0], inputs[1], stream)};
}
else if (op_name == "mul") {
    if (inputs.size() != 2) {
        throw std::invalid_argument("mul operation requires exactly 2 inputs");
    }
    return {cuda::mul_kernel(inputs[0], inputs[1], stream)};
}
// ... similar pattern with stream parameter ...
```

### 6.3 Current Registration (Indirect)

**File**: `/home/lee/Projects/Tenzor/src/core/init.cpp` (lines 63-86)

```cpp
auto* cpu_backend = cpu_backend_ptr;

// Register operations by forwarding to backend
registry.register_kernel("add", Device::Type::CPU,
    [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return cpu_backend->dispatch("add", inputs, attrs);  // ← Goes back to backend
    });

registry.register_kernel("sub", Device::Type::CPU,
    [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return cpu_backend->dispatch("sub", inputs, attrs);
    });

registry.register_kernel("mul", Device::Type::CPU,
    [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return cpu_backend->dispatch("mul", inputs, attrs);
    });

registry.register_kernel("div", Device::Type::CPU,
    [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return cpu_backend->dispatch("div", inputs, attrs);
    });
```

**Problem**: This creates circular dispatch:
```
registry.dispatch("add") → lambda → backend->dispatch("add") → if-else → kernel
```

Should be:
```
registry.dispatch("add") → kernel directly
```

---

## 7. Architecture Changes Summary

### 7.1 Current State

| Component | Current Behavior | Issues |
|-----------|-----------------|--------|
| `Dispatcher` | Calls `Backend::dispatch()` directly | Bypasses OperationRegistry |
| `Backend::dispatch()` | 800+ line if-else chain | Unmaintainable, duplicated |
| `OperationRegistry` | Populated but unused | Wasted infrastructure |
| Operation Lookup | Compile-time only | No runtime introspection |
| Fallback | None | No graceful degradation |

### 7.2 Target State

| Component | Target Behavior | Benefits |
|-----------|----------------|----------|
| `Dispatcher` | Calls `OperationRegistry::dispatch()` | Centralized dispatch |
| `Backend` | Only memory/stream management | Simplified backends |
| `OperationRegistry` | Primary dispatch mechanism | Single source of truth |
| Operation Lookup | Runtime query API | Dynamic discovery |
| Fallback | Automatic CPU fallback | Graceful degradation |

### 7.3 Migration Path

1. **Phase 1**: Add runtime query API to OperationRegistry
   - `has_kernel()`, `available_operations()`, `get_operation_info()`

2. **Phase 2**: Modify Dispatcher to use OperationRegistry
   - Change `Dispatcher::dispatch()` to call `operation_registry().dispatch()`

3. **Phase 3**: Update kernel registration to be direct
   - Change lambda wrappers to call kernels directly, not `backend->dispatch()`

4. **Phase 4**: Remove Backend::dispatch() method
   - Simplify Backend interface to only resource management

5. **Phase 5**: Add fallback mechanisms
   - Implement automatic CPU fallback for unsupported operations

---

## 8. Recommendations

### Immediate Actions

1. **Refactor Dispatcher** to use OperationRegistry as primary dispatch mechanism
2. **Update kernel registration** to call kernels directly instead of going through backend
3. **Add query API** for runtime operation introspection
4. **Implement fallback logic** for missing operations

### Long-term Improvements

1. **Plugin Architecture**: Enable dynamic loading of operation plugins
2. **JIT Compilation**: Generate specialized kernels at runtime
3. **Multi-Device Dispatch**: Support operations spanning multiple devices
4. **Operation Caching**: Cache compiled/optimized operations for repeated use

---

## 9. Conclusion

Tenzor has a well-designed **OperationRegistry infrastructure** that is currently **underutilized**. The existing architecture routes all operations through backend-specific dispatch methods with large if-else chains, bypassing the registry.

To achieve **true runtime dispatch**, the following changes are required:

1. Make `OperationRegistry` the **primary dispatch mechanism**
2. Register kernels **directly** instead of through backend wrappers
3. Remove or simplify `Backend::dispatch()` to only handle resource management
4. Add **runtime query APIs** for operation discovery
5. Implement **fallback mechanisms** for graceful degradation

This will enable:
- Dynamic operation availability checking
- Automatic fallback to CPU
- Plugin-based operation extensions
- Simplified backend implementations
- Better maintainability and extensibility

---

**Analyzed Files**:
- `/home/lee/Projects/Tenzor/include/tenzor/backend/backend.hpp`
- `/home/lee/Projects/Tenzor/include/tenzor/backend/registry.hpp`
- `/home/lee/Projects/Tenzor/include/tenzor/backend/dispatch.hpp`
- `/home/lee/Projects/Tenzor/include/tenzor/backend/loader.hpp`
- `/home/lee/Projects/Tenzor/src/backend/backend.cpp`
- `/home/lee/Projects/Tenzor/src/backend/registry.cpp`
- `/home/lee/Projects/Tenzor/src/backend/dispatch.cpp`
- `/home/lee/Projects/Tenzor/src/backend/loader.cpp`
- `/home/lee/Projects/Tenzor/src/backends/cpu/cpu_backend.cpp`
- `/home/lee/Projects/Tenzor/src/backends/cuda/cuda_backend.cpp`
- `/home/lee/Projects/Tenzor/src/core/init.cpp`
