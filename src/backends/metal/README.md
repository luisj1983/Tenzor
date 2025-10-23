# Tenzor Metal Backend

Complete Metal compute backend for macOS and iOS devices with Apple Silicon optimization.

## Overview

This Metal backend provides GPU acceleration for Tenzor on Apple platforms, leveraging:
- **Metal Performance Shaders (MPS)** for optimized BLAS operations
- **Unified Memory Architecture** for efficient CPU-GPU data transfer
- **Apple Silicon optimization** (M1/M2/M3/M4)
- **Native Metal compute shaders** for maximum performance

## Architecture

```
metal/
├── metal_backend.hpp       # Main backend interface
├── metal_backend.mm        # Objective-C++ implementation
├── metal_utils.hpp         # Helper utilities and error handling
├── CMakeLists.txt          # Build configuration
└── kernels/                # Metal shader files
    ├── matmul.metal        # Matrix multiplication (with MPS)
    ├── conv2d.metal        # 2D convolution operations
    ├── pooling.metal       # Pooling operations
    ├── batchnorm.metal     # Batch normalization
    ├── activations.metal   # Activation functions
    ├── reduction.metal     # Reduction operations
    ├── transform.metal     # Transform operations
    ├── math.metal          # Element-wise math
    └── indexing.metal      # Indexing operations
```

## Features

### Implemented Operations

#### Matrix Operations
- Matrix multiplication (FP32, FP16) with MPS acceleration
- Batch matrix multiplication
- Matrix-vector multiplication
- Transposed matrix operations
- Tiled GEMM for optimal performance

#### Convolution Operations
- 2D convolution (direct, im2col)
- Depthwise convolution
- Pointwise (1x1) convolution
- Transposed convolution (deconvolution)
- Dilated convolution

#### Pooling Operations
- Max pooling 2D with indices
- Average pooling 2D
- Adaptive pooling (max/avg)
- Global pooling (max/avg)
- LP pooling

#### Normalization
- Batch normalization (training/inference)
- Layer normalization
- Group normalization
- Instance normalization

#### Activation Functions
- ReLU, Leaky ReLU, PReLU
- GELU (Gaussian Error Linear Unit)
- Sigmoid, Tanh
- Softmax, Log Softmax
- Swish (SiLU), Mish
- ELU, Hardswish

#### Element-wise Operations
- Arithmetic: add, sub, mul, div
- Math functions: exp, log, sqrt, pow
- Trigonometric: sin, cos
- Utilities: abs, neg, reciprocal, clamp, sign
- Advanced: fma, lerp, min, max

#### Reduction Operations
- Sum, mean, product
- Max, min with indices (argmax, argmin)
- L2 norm, variance, standard deviation
- Cumulative sum (scan)
- Axis-specific reductions

#### Transform Operations
- Transpose (2D, N-D with tiling)
- Reshape, flatten
- Permute (general transpose)
- Expand (broadcasting)
- Repeat (tiling)
- Concatenate, split
- Slice, flip, roll

#### Indexing Operations
- Gather, scatter
- Index select
- Masked select, masked fill
- Advanced indexing
- Take along axis
- Embedding lookup
- One-hot encoding
- Diagonal operations

## Performance Optimizations

### Apple Silicon Specific
- **Thread Group Sizing**: Optimized for M-series GPU architecture
- **Tile Sizes**: 32x32 for matmul, 16x16 for convolution
- **Shared Memory**: Threadgroup memory for data reuse
- **Unified Memory**: MTLResourceStorageModeShared for zero-copy access

### Metal Best Practices
- Pipeline state caching for reduced overhead
- Command buffer batching
- Async execution support
- Proper synchronization primitives
- Memory tracking and management

## Usage

```cpp
#include <tenzor/backends/metal/metal_backend.hpp>

using namespace tenzor::backend::metal;

// Initialize backend
MetalBackend backend;
if (!backend.initialize()) {
    // Handle error
}

// Allocate memory
void* a = backend.allocate(1024 * sizeof(float));
void* b = backend.allocate(1024 * sizeof(float));
void* c = backend.allocate(1024 * sizeof(float));

// Copy data to device
backend.memcpy_h2d(a, host_data_a, 1024 * sizeof(float));
backend.memcpy_h2d(b, host_data_b, 1024 * sizeof(float));

// Perform matrix multiplication (32x32 x 32x32)
backend.matmul(a, b, c, 32, 32, 32,
              false, false, 1.0f, 0.0f,
              DataType::Float32);

// Synchronize
backend.synchronize();

// Copy result back
backend.memcpy_d2h(host_data_c, c, 1024 * sizeof(float));

// Cleanup
backend.deallocate(a);
backend.deallocate(b);
backend.deallocate(c);
```

## Building

The Metal backend is automatically built on macOS:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make tenzor_metal
```

### Requirements
- macOS 10.15+ or iOS 13+
- Xcode Command Line Tools
- Apple Silicon (M1/M2/M3) recommended
- Metal 2.0+

### Compilation Process
1. Metal shaders (.metal) → .air (intermediate)
2. .air files → .metallib (shader library)
3. C++ backend links with .metallib at runtime

## Metal Performance Shaders (MPS) Integration

For maximum performance, critical operations use MPS:

- **Matrix Multiplication**: `MPSMatrixMultiplication`
- **Convolution**: `MPSCNNConvolution` (with proper weight formatting)
- **Image Operations**: MPS image descriptors for efficient data layout

## Data Types

Supported data types:
- `Float32` (32-bit floating point)
- `Float16` (16-bit half precision)
- `Int32` (32-bit integer)
- `Int8` (8-bit integer)

## Memory Management

- **Unified Memory**: Shared between CPU and GPU (zero-copy)
- **Tracking**: Automatic memory usage tracking
- **Thread Safety**: Command buffer synchronization
- **Resource Management**: RAII-style cleanup

## Thread Configuration

Optimal thread group sizes:
- 1D operations: 256 threads
- 2D operations: 16x16 threads
- 3D operations: 8x8x4 threads
- Matrix operations: 32x32 tiles

## Limitations

- Requires macOS/iOS platform
- Some advanced MPS features need proper data formatting
- Memory is limited by system RAM (unified memory)
- No multi-GPU support (Metal typically uses single device)

## Future Enhancements

- [ ] MLX integration for ML-specific optimizations
- [ ] Metal 3 features (mesh shaders, ray tracing)
- [ ] Neural Engine integration
- [ ] More MPS accelerated operations
- [ ] Metal FX for upscaling/denoising
- [ ] iOS/iPadOS optimization
- [ ] Metal shader debugging tools

## Performance Benchmarks

Expected performance on M1 Pro:
- Matrix Multiplication (4096x4096): ~800 GFLOPS
- Conv2D (ResNet-50 layers): ~40-60 ms/layer
- Element-wise ops: ~200 GB/s bandwidth

## Troubleshooting

### Common Issues

1. **Shader compilation fails**
   - Ensure Xcode Command Line Tools installed
   - Check Metal shader syntax

2. **Runtime crashes**
   - Verify buffer sizes match expectations
   - Check for out-of-bounds access
   - Enable Metal API validation

3. **Poor performance**
   - Verify running on GPU (not integrated graphics)
   - Check thread group sizes
   - Profile with Metal System Trace

### Debugging

Enable Metal API validation:
```bash
export MTL_DEBUG_LAYER=1
export MTL_SHADER_VALIDATION=1
```

Use Xcode Instruments for profiling:
- Metal System Trace
- GPU counters
- Memory analysis

## References

- [Metal Programming Guide](https://developer.apple.com/metal/)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Metal Best Practices](https://developer.apple.com/metal/best-practices/)
- [Apple Silicon Optimization](https://developer.apple.com/documentation/metal/gpu_features)

## License

Same as Tenzor main project.
