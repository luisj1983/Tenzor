# Runtime Dispatcher Sequence Diagrams

## Operation Dispatch Flow

### 1. Standard Operation Dispatch (No Fallback)

```
User Code                 Frontend (ops/math.hpp)     Dispatcher              Backend              Kernel
    |                              |                      |                      |                     |
    |  add(cuda_a, cuda_b)         |                      |                      |                     |
    |---------------------------->|                      |                      |                     |
    |                              |                      |                      |                     |
    |                              | dispatch("add", {a,b})                     |                     |
    |                              |--------------------->|                      |                     |
    |                              |                      |                      |                     |
    |                              |                      | check device compat  |                     |
    |                              |                      |---                   |                     |
    |                              |                      |  |                   |                     |
    |                              |                      |<--                   |                     |
    |                              |                      |                      |                     |
    |                              |                      | get_backend(CUDA)    |                     |
    |                              |                      |--------------------->|                     |
    |                              |                      |                      |                     |
    |                              |                      |     backend ptr      |                     |
    |                              |                      |<---------------------|                     |
    |                              |                      |                      |                     |
    |                              |                      | backend->dispatch("add", inputs, attrs)    |
    |                              |                      |--------------------->|                     |
    |                              |                      |                      |                     |
    |                              |                      |                      | lookup kernel       |
    |                              |                      |                      | in registry         |
    |                              |                      |                      |---                  |
    |                              |                      |                      |  |                  |
    |                              |                      |                      |<--                  |
    |                              |                      |                      |                     |
    |                              |                      |                      | execute kernel      |
    |                              |                      |                      |-------------------->|
    |                              |                      |                      |                     |
    |                              |                      |                      |     result          |
    |                              |                      |                      |<--------------------|
    |                              |                      |                      |                     |
    |                              |                      |      result          |                     |
    |                              |                      |<---------------------|                     |
    |                              |                      |                      |                     |
    |                              |      result          |                      |                     |
    |                              |<---------------------|                      |                     |
    |                              |                      |                      |                     |
    |         result               |                      |                      |                     |
    |<----------------------------|                      |                      |                     |
    |                              |                      |                      |                     |
```

### 2. Operation Dispatch with CPU Fallback

```
User Code              Frontend           Dispatcher        OperationRegistry     CPUBackend      CUDABackend
    |                      |                   |                    |                  |               |
    |  custom_op(cuda_a)   |                   |                    |                  |               |
    |--------------------->|                   |                    |                  |               |
    |                      |                   |                    |                  |               |
    |                      | dispatch("custom_op", {a})              |                  |               |
    |                      |------------------>|                    |                  |               |
    |                      |                   |                    |                  |               |
    |                      |                   | get_backend(CUDA)  |                  |               |
    |                      |                   |------------------->|                  |               |
    |                      |                   |                    |                  |               |
    |                      |                   |   cuda_backend     |                  |               |
    |                      |                   |<-------------------|                  |               |
    |                      |                   |                    |                  |               |
    |                      |                   | backend->dispatch("custom_op", inputs, attrs)          |
    |                      |                   |-------------------------------------------------->|   |
    |                      |                   |                    |                  |               |
    |                      |                   |                    | lookup "custom_op" for CUDA      |
    |                      |                   |                    |<---------------------------------|
    |                      |                   |                    |                  |               |
    |                      |                   |                    | NOT FOUND - check fallback       |
    |                      |                   |                    |---               |               |
    |                      |                   |                    |  |               |               |
    |                      |                   |                    |<--               |               |
    |                      |                   |                    |                  |               |
    |                      |                   |                    | fallback: CPU    |               |
    |                      |                   |                    |                  |               |
    |                      |                   |                    | transfer tensors to CPU          |
    |                      |                   |                    |----------------->|               |
    |                      |                   |                    |                  |               |
    |                      |                   |                    | execute on CPU   |               |
    |                      |                   |                    |----------------->|               |
    |                      |                   |                    |                  |               |
    |                      |                   |                    |  cpu_result      |               |
    |                      |                   |                    |<-----------------|               |
    |                      |                   |                    |                  |               |
    |                      |                   |                    | transfer result back to CUDA     |
    |                      |                   |                    |--------------------------------->|
    |                      |                   |                    |                  |               |
    |                      |                   |      result        |                  |               |
    |                      |                   |<-------------------|                  |               |
    |                      |                   |                    |                  |               |
    |                      |      result       |                    |                  |               |
    |                      |<------------------|                    |                  |               |
    |                      |                   |                    |                  |               |
    |      result          |                   |                    |                  |               |
    |<---------------------|                   |                    |                  |               |
    |                      |                   |                    |                  |               |
```

### 3. Library Initialization Sequence

```
Program Start      init.cpp           BackendRegistry    OperationRegistry    CPU Backend    CUDA Backend
    |                  |                      |                    |                |                |
    | static init      |                      |                    |                |                |
    |----------------->|                      |                    |                |                |
    |                  |                      |                    |                |                |
    |                  | initialize_backends()|                    |                |                |
    |                  |---                   |                    |                |                |
    |                  |  |                   |                    |                |                |
    |                  |<--                   |                    |                |                |
    |                  |                      |                    |                |                |
    |                  | create_cpu_backend() |                    |                |                |
    |                  |------------------------------------------------->|           |                |
    |                  |                      |                    |                |                |
    |                  |  cpu_backend         |                    |                |                |
    |                  |<--------------------------------------------------|           |                |
    |                  |                      |                    |                |                |
    |                  | backend->is_available()                   |                |                |
    |                  |------------------------------------------------->|           |                |
    |                  |                      |                    |                |                |
    |                  |      true            |                    |                |                |
    |                  |<--------------------------------------------------|           |                |
    |                  |                      |                    |                |                |
    |                  | backend->register_operations(op_registry) |                |                |
    |                  |------------------------------------------------->|           |                |
    |                  |                      |                    |                |                |
    |                  |                      |                    | register_kernel("add", CPU, fn) |
    |                  |                      |                    |<---------------|                |
    |                  |                      |                    |                |                |
    |                  |                      |                    | register_kernel("matmul", CPU, fn)
    |                  |                      |                    |<---------------|                |
    |                  |                      |                    |                |                |
    |                  |                      |                    | ... (all ops)  |                |
    |                  |                      |                    |                |                |
    |                  | registry.register_backend(CPU, backend)   |                |                |
    |                  |--------------------->|                    |                |                |
    |                  |                      |                    |                |                |
    |                  | #ifdef CUDA          |                    |                |                |
    |                  | create_cuda_backend()|                    |                |                |
    |                  |---------------------------------------------------------------->|            |
    |                  |                      |                    |                |                |
    |                  |  cuda_backend        |                    |                |                |
    |                  |<----------------------------------------------------------------|            |
    |                  |                      |                    |                |                |
    |                  | backend->is_available()                   |                |                |
    |                  |---------------------------------------------------------------->|            |
    |                  |                      |                    |                |                |
    |                  |   true (if CUDA HW)  |                    |                |                |
    |                  |<----------------------------------------------------------------|            |
    |                  |                      |                    |                |                |
    |                  | backend->register_operations(op_registry) |                |                |
    |                  |---------------------------------------------------------------->|            |
    |                  |                      |                    |                |                |
    |                  |                      |                    | register_kernel("add", CUDA, fn)|
    |                  |                      |                    |<-------------------------------|
    |                  |                      |                    |                |                |
    |                  |                      |                    | register_fallback("custom", CUDA, CPU)
    |                  |                      |                    |<-------------------------------|
    |                  |                      |                    |                |                |
    |                  | registry.register_backend(CUDA, backend)  |                |                |
    |                  |--------------------->|                    |                |                |
    |                  |                      |                    |                |                |
    | initialization   |                      |                    |                |                |
    | complete         |                      |                    |                |                |
    |<-----------------|                      |                    |                |                |
    |                  |                      |                    |                |                |
```

## Data Structures

### OperationRegistry Internal Structure

```
OperationRegistry
├── kernels_: unordered_map
│   ├── "add" → unordered_map
│   │   ├── Device::Type::CPU → KernelFunction (cpu_add_kernel)
│   │   ├── Device::Type::CUDA → KernelFunction (cuda_add_kernel)
│   │   └── Device::Type::ROCm → KernelFunction (rocm_add_kernel)
│   │
│   ├── "matmul" → unordered_map
│   │   ├── Device::Type::CPU → KernelFunction (cpu_matmul_kernel)
│   │   ├── Device::Type::CUDA → KernelFunction (cuda_matmul_kernel)
│   │   └── Device::Type::ROCm → KernelFunction (rocm_matmul_kernel)
│   │
│   └── "conv2d" → unordered_map
│       ├── Device::Type::CPU → KernelFunction (cpu_conv2d_kernel)
│       └── Device::Type::CUDA → KernelFunction (cuda_conv2d_kernel)
│
└── fallbacks_: unordered_map
    ├── "advanced_op" → unordered_map
    │   ├── Device::Type::CUDA → Device::Type::CPU
    │   └── Device::Type::ROCm → Device::Type::CPU
    │
    └── "custom_transform" → unordered_map
        └── Device::Type::CUDA → Device::Type::CPU
```

### BackendRegistry Internal Structure

```
BackendRegistry
├── backends_: unordered_map
│   ├── Device::Type::CPU → unique_ptr<CPUBackend>
│   ├── Device::Type::CUDA → unique_ptr<CUDABackend>
│   ├── Device::Type::ROCm → unique_ptr<ROCmBackend>
│   └── Device::Type::OneAPI → unique_ptr<OneAPIBackend>
│
└── initialized_: bool
```

## Decision Tree for Operation Dispatch

```
┌─────────────────────────────────────┐
│  Dispatcher::dispatch(op, inputs)   │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Check device compatibility          │
│  (all inputs on same device?)        │
└──────────────┬───────────────────────┘
               │
               ├─ No ──> Throw DeviceException
               │
               ▼ Yes
┌──────────────────────────────────────┐
│  Get device type from first input    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Backend* backend =                  │
│    BackendRegistry::get_backend(type)│
└──────────────┬───────────────────────┘
               │
               ├─ nullptr ──> Throw TenzorException
               │
               ▼ valid ptr
┌──────────────────────────────────────┐
│  backend->dispatch(op, inputs, attrs)│
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  OperationRegistry::get_kernel_with_ │
│    fallback(op, device_type)         │
└──────────────┬───────────────────────┘
               │
               ├─ Found kernel for device ──> Execute kernel
               │
               ▼ Not found
┌──────────────────────────────────────┐
│  Check fallback registry             │
└──────────────┬───────────────────────┘
               │
               ├─ No fallback ──> Throw runtime_error
               │
               ▼ Fallback found (e.g., CPU)
┌──────────────────────────────────────┐
│  1. Transfer tensors to CPU          │
│  2. Execute kernel on CPU            │
│  3. Transfer results back            │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Return result tensors               │
└──────────────────────────────────────┘
```

## Timeline: Typical Operation Execution

```
Time  │  Activity                                      CPU Usage  GPU Usage
──────┼─────────────────────────────────────────────  ─────────  ─────────
      │
  0ms │  User calls add(cuda_a, cuda_b)               ████
      │
  1ms │  Frontend dispatch("add", {a, b})             ███
      │
  2ms │  Device compatibility check                   ██
      │
  3ms │  Get CUDA backend from registry               █
      │
  4ms │  Lookup "add" kernel in registry              █
      │
  5ms │  Launch CUDA kernel                           █
      │
  6ms │  Kernel executes on GPU                                  ████████
      │
  7ms │                                                           ████████
      │
  8ms │                                                           ████████
      │
  9ms │  Kernel completes                                        █
      │
 10ms │  Return result to user                        █
      │
```

## Comparison: With vs Without Fallback

### Scenario: CUDA backend missing "custom_transform" operation

#### Without Fallback (Error)

```
User Code            Dispatcher          CUDA Backend
    |                     |                    |
    | custom_transform()  |                    |
    |-------------------->|                    |
    |                     |                    |
    |                     | dispatch()         |
    |                     |------------------->|
    |                     |                    |
    |                     |                    | lookup fails
    |                     |                    |---
    |                     |                    |  |
    |                     |                    |<--
    |                     |                    |
    |                     | runtime_error      |
    |                     |<-------------------|
    |                     |                    |
    | exception           |                    |
    |<--------------------|                    |
    |                     |                    |
  [CRASH]
```

#### With Fallback (Success)

```
User Code            Dispatcher          CUDA Backend      CPU Backend
    |                     |                    |                |
    | custom_transform()  |                    |                |
    |-------------------->|                    |                |
    |                     |                    |                |
    |                     | dispatch()         |                |
    |                     |------------------->|                |
    |                     |                    |                |
    |                     |                    | lookup fails   |
    |                     |                    | check fallback |
    |                     |                    |---             |
    |                     |                    |  |             |
    |                     |                    |<--             |
    |                     |                    |                |
    |                     |                    | transfer to CPU|
    |                     |                    |--------------->|
    |                     |                    |                |
    |                     |                    |                | execute
    |                     |                    |                |---
    |                     |                    |                |  |
    |                     |                    |                |<--
    |                     |                    |                |
    |                     |                    | result         |
    |                     |                    |<---------------|
    |                     |                    |                |
    |                     |                    | transfer to GPU|
    |                     |                    |---             |
    |                     |                    |  |             |
    |                     |                    |<--             |
    |                     |                    |                |
    |                     | result             |                |
    |                     |<-------------------|                |
    |                     |                    |                |
    | result              |                    |                |
    |<--------------------|                    |                |
    |                     |                    |                |
  [SUCCESS]
```

## Performance Metrics

### Dispatch Overhead Breakdown

```
Component                    Time (ns)    % of Total
─────────────────────────────────────────────────────
Frontend function call           10          2%
Device compatibility check       30          6%
Backend registry lookup          20          4%
Operation registry lookup        50         10%
Kernel function call             40          8%
─────────────────────────────────────────────────────
Total Dispatch Overhead         150         30%
Actual Kernel Execution         350         70%
─────────────────────────────────────────────────────
Total Operation Time            500        100%
```

### Fallback Performance Impact

```
Scenario                      Time (μs)    Overhead
──────────────────────────────────────────────────────
Direct GPU execution              100          0%
GPU → CPU → GPU (small tensor)    450        350%
GPU → CPU → GPU (large tensor)   5000       4900%
```

**Recommendation**: Use fallback only for operations that are:
1. Infrequently called
2. Operating on small tensors
3. Not performance-critical

For frequently-used operations, implement native backend support instead of relying on fallback.
