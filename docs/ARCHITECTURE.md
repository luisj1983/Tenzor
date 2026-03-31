# Tenzor Architecture

This document describes the high-level architecture and design principles of the Tenzor library.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Backend System](#backend-system)
- [Autograd Engine](#autograd-engine)
- [Neural Network Module](#neural-network-module)
- [Memory Management](#memory-management)
- [Threading Model](#threading-model)
- [Design Principles](#design-principles)

## Overview

Tenzor is designed as a modular, high-performance tensor computation library with the following goals:

1. **Performance**: Match or exceed PyTorch/TensorFlow performance
2. **Portability**: Run on CPU, NVIDIA, AMD, Intel, and Apple hardware
3. **Usability**: Provide an intuitive, PyTorch-like API
4. **Extensibility**: Easy to add new operations, backends, and features

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        User Applications                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                     Python Bindings (pybind11)                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   Models     │  │     JIT      │  │    ONNX      │  │Quantization │ │
│  │   (ResNet,   │  │  (Tracing,   │  │  (Import/    │  │ (INT8/FP16) │ │
│  │    BERT...)  │  │   Fusion)    │  │   Export)    │  │             │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│                    Neural Network Module (nn::)                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐           │
│  │   Layers   │ │   Loss     │ │ Optimizers │ │ Schedulers │           │
│  │ (Linear,   │ │ Functions  │ │ (Adam,SGD) │ │ (StepLR,   │           │
│  │  Conv2d)   │ │ (CE, MSE)  │ │            │ │  Cosine)   │           │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘           │
├─────────────────────────────────────────────────────────────────────────┤
│                       Autograd Engine                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Variable ─── Computational Graph ─── Gradient Computation       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                     Core Tensor Operations                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐           │
│  │    Math    │ │  Reduction │ │  Transform │ │  Indexing  │           │
│  │ (matmul,   │ │ (sum,mean, │ │ (reshape,  │ │ (slice,    │           │
│  │  add,mul)  │ │  max,min)  │ │  permute)  │ │  gather)   │           │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘           │
├─────────────────────────────────────────────────────────────────────────┤
│                    Backend Abstraction Layer                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Backend Interface ─── Operation Registry ─── Kernel Dispatch    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │   CPU   │ │  CUDA   │ │  ROCm   │ │ OneAPI  │ │ Vulkan  │           │
│  │  (SIMD) │ │(cuBLAS) │ │(hipBLAS)│ │ (oneMKL)│ │(Compute)│           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
├─────────────────────────────────────────────────────────────────────────┤
│                      Hardware Layer                                     │
│     CPU (x86/ARM)  │  NVIDIA GPU  │  AMD GPU  │  Intel GPU            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### Tensor (`core/tensor.hpp`)

The `Tensor` class is the fundamental data structure:

```cpp
class Tensor {
    std::shared_ptr<TensorStorage> storage_;  // Data storage
    std::vector<int64_t> shape_;              // Dimensions
    std::vector<int64_t> strides_;            // Memory strides
    int64_t offset_;                          // Data offset
    DType dtype_;                             // Data type
    Device device_;                           // Execution device
};
```

Key features:
- **Copy-on-Write**: Tensors share storage until modified
- **View Semantics**: Reshape/transpose create views, not copies
- **Strided Layout**: Supports non-contiguous memory access
- **Device Agnostic**: Same API regardless of hardware

### TensorStorage (`core/storage.hpp`)

Manages raw memory allocation:

```cpp
class TensorStorage {
    void* data_;                    // Raw data pointer
    size_t size_;                   // Size in bytes
    Device device_;                 // Where data resides
    std::shared_ptr<Allocator> allocator_;  // Memory allocator
};
```

### DType (`core/dtype.hpp`)

Type-safe enumeration of supported data types:

```cpp
enum class DType : uint8_t {
    Float32, Float64, Float16, BFloat16,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Bool, Complex64, Complex128
};
```

### Device (`core/device.hpp`)

Represents execution targets:

```cpp
struct Device {
    enum class Type : uint8_t {
        CPU, CUDA, ROCm, OneAPI, Vulkan
    };
    Type type;
    int32_t index;  // Device index for multi-device systems
};
```

## Backend System

### Backend Interface (`backend/backend.hpp`)

All backends implement this interface:

```cpp
class Backend {
public:
    virtual ~Backend() = default;

    // Lifecycle
    virtual void initialize() = 0;
    virtual void shutdown() = 0;
    virtual bool is_available() const = 0;

    // Memory management
    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void copy(void* dst, const void* src, size_t size) = 0;

    // Device info
    virtual int device_count() const = 0;
    virtual std::string device_name(int index) const = 0;
    virtual size_t total_memory(int index) const = 0;
};
```

### Operation Registry (`backend/operation_registry.hpp`)

Operations are registered and dispatched dynamically:

```cpp
class OperationRegistry {
public:
    template<typename Op>
    void register_operation(Device::Type device);

    template<typename Op>
    Op* get_operation(Device::Type device);
};
```

### Backend Implementations

| Backend | Location | Key Technologies |
|---------|----------|------------------|
| CPU | `backends/cpu/` | SIMD (AVX2/AVX-512), OpenMP |
| CUDA | `backends/cuda/` | cuBLAS, cuDNN, CUTLASS |
| ROCm | `backends/rocm/` | hipBLAS, MIOpen |
| OneAPI | `backends/oneapi/` | oneMKL, oneDNN |
| Vulkan | `backends/vulkan/` | Compute shaders |
| Metal | `backends/metal/` | Metal Performance Shaders |

### Kernel Dispatch Flow

```
User Code: z = matmul(x, y)
    │
    ▼
Operation Dispatcher
    │
    ├── Check device of x, y
    ├── Validate shapes
    ├── Select backend
    │
    ▼
Backend-Specific Implementation
    │
    ├── CPU: SIMD-optimized BLAS
    ├── CUDA: cuBLAS gemm
    ├── ROCm: hipBLAS gemm
    └── ...
    │
    ▼
Return Tensor z
```

## Autograd Engine

### Variable (`autograd/variable.hpp`)

Wraps tensors for gradient tracking:

```cpp
class Variable {
    Tensor data_;                              // Tensor data
    Tensor grad_;                              // Gradient tensor
    std::shared_ptr<GradFn> grad_fn_;         // Gradient function
    bool requires_grad_;                       // Track gradients?
    std::vector<Variable> saved_tensors_;     // Saved for backward
};
```

### Computational Graph

The autograd engine builds a DAG during forward pass:

```
Forward Pass:
    x ──┬── Linear(w1) ── ReLU ── Linear(w2) ── Loss
        │       │                     │
        └───────┴─────────────────────┴── saved_tensors

Backward Pass:
    Loss.backward()
        │
        ▼
    ∂L/∂w2 ← Linear.backward()
        │
        ▼
    ∂L/∂h ← ReLU.backward()
        │
        ▼
    ∂L/∂w1, ∂L/∂x ← Linear.backward()
```

### Gradient Functions (`autograd/functions.hpp`)

Each operation defines its backward:

```cpp
class MatMulBackward : public GradFn {
public:
    Tensor backward(const Tensor& grad_output) override {
        // grad_a = grad_output @ b.T
        // grad_b = a.T @ grad_output
        return {grad_a, grad_b};
    }

private:
    Tensor a_, b_;  // Saved from forward
};
```

### Gradient Accumulation

```cpp
void Variable::backward() {
    // Topological sort of computation graph
    auto sorted = topological_sort(this);

    // Initialize gradient
    this->grad_ = ones_like(this->data_);

    // Backward pass
    for (auto& node : reverse(sorted)) {
        auto grads = node.grad_fn->backward(node.grad_);
        for (auto& [input, grad] : zip(node.inputs, grads)) {
            input.grad_ += grad;  // Accumulate gradients
        }
    }
}
```

## Neural Network Module

### Module Base Class (`nn/module.hpp`)

```cpp
class Module {
public:
    virtual Variable forward(const Variable& input) = 0;

    // Parameter management
    std::vector<std::shared_ptr<Variable>> parameters();
    void register_parameter(const std::string& name, Variable param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);

    // Device transfer
    void to(Device device);
    void cuda(int index = 0);
    void cpu();

    // Training mode
    void train(bool mode = true);
    void eval();
};
```

### Layer Implementation Example

```cpp
class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, bool bias = true)
        : weight_(register_parameter("weight",
            Variable(randn({out_features, in_features}) * std::sqrt(2.0 / in_features), true)))
        , bias_(bias ? register_parameter("bias", Variable(zeros({out_features}), true)) : nullptr)
    {}

    Variable forward(const Variable& input) override {
        auto output = matmul(input, weight_.t());
        if (bias_) {
            output = output + *bias_;
        }
        return output;
    }

private:
    Variable weight_;
    std::shared_ptr<Variable> bias_;
};
```

### Module Hierarchy

```
Module
├── Linear
├── Conv1d/Conv2d/Conv3d
├── BatchNorm1d/BatchNorm2d
├── LayerNorm
├── Dropout
├── ReLU/GELU/SiLU/...
├── LSTM/GRU
├── MultiheadAttention
├── Embedding
├── Sequential
└── ModuleList/ModuleDict
```

## Memory Management

### Caching Allocator (`backend/allocator.hpp`)

Reduces allocation overhead through caching:

```cpp
class CachingAllocator {
public:
    void* allocate(size_t size) {
        // Check cache for suitable block
        if (auto block = find_cached_block(size)) {
            return block;
        }
        // Allocate new block
        return backend_allocate(round_size(size));
    }

    void deallocate(void* ptr) {
        // Return to cache instead of freeing
        cache_.insert(ptr);
    }

private:
    std::set<Block, SizeComparator> cache_;
};
```

### Memory Pool Strategy

```
┌──────────────────────────────────────────────────────┐
│                  Memory Pool                          │
├──────────────────────────────────────────────────────┤
│  Small Blocks    │  Medium Blocks  │  Large Blocks   │
│  (< 1MB)         │  (1MB - 10MB)   │  (> 10MB)       │
├──────────────────────────────────────────────────────┤
│  Free List       │  Free List      │  Direct Alloc   │
│  (Round to 256B) │  (Round to 2MB) │  (Exact size)   │
└──────────────────────────────────────────────────────┘
```

### Memory Transfer

```cpp
// Cross-device transfer
Tensor Tensor::to(Device target) {
    if (device_ == target) return *this;

    // Allocate on target
    auto result = empty(shape_, dtype_, target);

    // Copy data
    if (device_.type == Device::Type::CPU) {
        // CPU → GPU
        backend->copy_host_to_device(result.data_ptr(), data_ptr(), nbytes());
    } else if (target.type == Device::Type::CPU) {
        // GPU → CPU
        backend->copy_device_to_host(result.data_ptr(), data_ptr(), nbytes());
    } else {
        // GPU → GPU (peer transfer or via CPU)
        backend->copy_device_to_device(result.data_ptr(), data_ptr(), nbytes());
    }

    return result;
}
```

## Threading Model

### CPU Parallelization

```cpp
// OpenMP parallel loops for CPU operations
void add_kernel_cpu(float* out, const float* a, const float* b, size_t n) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
```

### Thread Safety

- **Tensor Operations**: Thread-safe for independent tensors
- **Autograd**: Single-threaded backward pass (Python GIL in bindings)
- **Memory Allocator**: Thread-local caches with global lock for large allocations
- **Model Parameters**: Copy-on-write for safe multi-threaded inference

### Stream Management (CUDA)

```cpp
class CUDAStream {
public:
    void synchronize();
    void wait_event(CUDAEvent& event);

    // Operations on this stream
    Tensor matmul_async(const Tensor& a, const Tensor& b);
};

// Multiple streams for overlap
auto stream1 = CUDAStream();
auto stream2 = CUDAStream();

// Parallel execution
auto a = stream1.matmul_async(x, y);
auto b = stream2.matmul_async(z, w);

// Synchronize
stream1.synchronize();
stream2.synchronize();
```

## Design Principles

### 1. Zero-Cost Abstractions

C++ templates eliminate runtime overhead:

```cpp
template<typename T>
void elementwise_add(T* out, const T* a, const T* b, size_t n) {
    // Compiled to optimal code for each type
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
```

### 2. Value Semantics with Shared Storage

```cpp
Tensor a = randn({1000, 1000});
Tensor b = a;  // Shallow copy, shared storage

b = b + 1;     // Copy-on-write: b gets new storage
```

### 3. RAII Resource Management

```cpp
class CUDAGuard {
public:
    CUDAGuard(int device) : prev_(current_device()) {
        set_device(device);
    }
    ~CUDAGuard() {
        set_device(prev_);
    }
private:
    int prev_;
};

// Automatic device restore
{
    CUDAGuard guard(1);  // Switch to device 1
    // Operations on device 1
}  // Automatically restored to previous device
```

### 4. Lazy Evaluation (JIT)

```cpp
// Operations can be fused at JIT compile time
auto traced = jit::trace(model, example_input);
traced.optimize();  // Fuses: Linear + ReLU + Linear
traced.save("model.pt");
```

### 5. Extensibility

Adding a new operation:

```cpp
// 1. Define operation
Tensor my_custom_op(const Tensor& x, const Tensor& y);

// 2. Register for each backend
REGISTER_OPERATION(my_custom_op, CPU, my_custom_op_cpu);
REGISTER_OPERATION(my_custom_op, CUDA, my_custom_op_cuda);

// 3. Define backward for autograd
class MyCustomOpBackward : public GradFn {
    Tensor backward(const Tensor& grad) override { ... }
};
```

## Directory Structure

```
tenzor/
├── include/tenzor/           # Public headers
│   ├── core/                 # Tensor, DType, Device
│   ├── autograd/             # Variable, GradFn
│   ├── nn/                   # Neural network layers
│   ├── optim/                # Optimizers
│   ├── ops/                  # Operation declarations
│   ├── backend/              # Backend interface
│   ├── backends/             # Backend implementations
│   ├── data/                 # Data loading
│   ├── jit/                  # JIT compilation
│   ├── onnx/                 # ONNX import/export
│   └── utils/                # Utilities
├── src/                      # Implementation files
├── python/                   # Python bindings
├── tests/                    # Test suite
├── benchmarks/               # Performance benchmarks
└── examples/                 # Example code
```

## Performance Considerations

### Memory Access Patterns

- Prefer contiguous memory layouts
- Use views instead of copies when possible
- Batch small operations to reduce kernel launch overhead

### Kernel Optimization

- SIMD vectorization for CPU
- Warp-level primitives for CUDA
- Kernel fusion for reduced memory bandwidth

### Profiling

```cpp
// Built-in profiling
{
    auto profiler = utils::Profiler("forward_pass");
    auto output = model->forward(input);
}  // Prints timing on destruction
```

---

For more details, see the [API Documentation](api/html/index.html) and source code comments.
