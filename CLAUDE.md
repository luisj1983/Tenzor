# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# From build/ directory (already configured with Ninja)
ninja                           # Build all targets
ninja tenzor                    # Build core library only
ninja tenzor_backend_cpu        # Build CPU backend
ninja tenzor_backend_cuda       # Build CUDA backend
ninja tenzor_core               # Build Python module

# Fresh configuration (from project root)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

## Testing

```bash
# From build/ directory
ctest --output-on-failure                    # Run all tests
ctest -R "pattern"                           # Run tests matching pattern
ctest -R "test_name" -V                      # Run single test with verbose output
./bin/test_executable_name                   # Run test binary directly

# Examples:
ctest -R "BatchNorm"                         # All BatchNorm tests
ctest -R "cpu_float32"                       # CPU Float32 tests only
ctest -R "Linear.*cpu"                       # Linear layer CPU tests
```

## Python Development

```bash
# From project root (build/ must exist with compiled module)
cd build && python -c "import sys; sys.path.insert(0, 'python'); import tenzor as tz; tz.initialize()"

# Run Python benchmarks
cd benchmarks/python && python run_benchmarks.py --device cpu --quick
```

## Architecture Overview

**Tenzor** is a high-performance tensor computation library with automatic differentiation, similar to PyTorch but written in modern C++23.

### Layered Design

```
Python Bindings (pybind11)  →  python/tenzor/
         ↓
Neural Network API          →  include/tenzor/nn/, src/nn/
         ↓
Autograd Engine             →  include/tenzor/autograd/, src/autograd/
         ↓
Tensor Operations           →  include/tenzor/ops/, src/ops/
         ↓
Backend Abstraction         →  include/tenzor/backend/
         ↓
Backend Implementations     →  src/backends/{cpu,cuda,rocm,vulkan,oneapi,metal}/
```

### Key Directories

- `include/tenzor/core/` - Tensor, Device, DType, Storage (public API)
- `include/tenzor/ops/` - Operation declarations (math, creation, reduction, transform, indexing)
- `src/backends/cpu/kernels/` - CPU kernel implementations (SIMD, OpenMP, MKL)
- `src/nn/layers/` - Neural network layer implementations
- `tests/backend_parity/` - Cross-backend correctness tests

### Backend Dispatch System

Operations use O(1) dispatch via `OpId` enum:
```cpp
// src/ops/math.cpp
return dispatch<OpId::MatMul>(inputs)[0];  // Dispatches to registered backend kernel
```

Backend kernels register in `src/backends/*/kernel_registry.cpp`:
```cpp
TENZOR_REGISTER_BINARY_KERNEL(table, MatMul, cpu::matmul_kernel);
```

### Sparse Dispatch

Sparse tensor operations (OpIds 460-464: SparseSpMM, SparseSpMV, SparseToDense, DenseToSparse,
SparseAdd) are registered in all backend dispatch tables (CPU, CUDA, ROCm, OneAPI). The wrapper
kernels reconstruct a `SparseTensor` from CSR components (crow_indices, col_indices, values) passed
as plain Tensors, then delegate to `sparse::spmm()` / `sparse::spmv()` / `sparse::add()` in
`src/sparse/sparse_ops.cpp`. Those functions internally select MKL, cuSPARSE, rocSPARSE, or oneMKL
based on device type. Vulkan sparse ops use native Vulkan dispatch (registered separately).

### Autograd Pattern

Variables wrap Tensors with gradient tracking:
```cpp
// Forward creates computation graph
auto result = autograd::matmul(x, w);  // Returns Variable with grad_fn

// Backward traverses graph
result.backward();  // Computes gradients via chain rule
```

### Multi-DType Support

Operations must handle multiple dtypes (Float32, Float64, Float16, BFloat16, Int8, etc.):
```cpp
if (a.dtype() == DType::Float32) {
    // Float32 path (often MKL/BLAS optimized)
} else if (a.dtype() == DType::Float16) {
    // Float16: typically convert to Float32, compute, convert back
}
```

### CPU Optimization Patterns

- **MKL BLAS**: `cblas_sgemm`, `cblas_sgemm_batch_strided` for matrix ops
- **SIMD**: AVX512/AVX2/SSE2 intrinsics with compile-time detection
- **OpenMP**: Parallel loops for large tensors (threshold ~65536 elements)
- **oneDNN**: For complex ops like convolution and normalization

### Python Bindings

`python/bindings.cpp` exposes C++ API via pybind11. The module is `tenzor_core`, wrapped by `python/tenzor/__init__.py` which adds Python-friendly APIs.

## Code Conventions

- C++23 features (concepts, ranges, structured bindings)
- snake_case for functions/variables, PascalCase for classes
- RAII and smart pointers (no raw new/delete)
- Doxygen comments for public APIs
- Tests required for new features
