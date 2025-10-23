# Tenzor WebGPU Backend

Complete WebGPU backend implementation for Tenzor, enabling high-performance tensor operations in web browsers and native applications using the WebGPU API.

## Features

### Core Backend (`webgpu_backend.hpp/cpp`)

- **Device Management**: Automatic adapter selection and device initialization
- **Buffer Management**: Efficient GPU memory allocation and management
- **Memory Operations**: Async read/write, buffer copying, staging buffers
- **Queue Management**: Command submission and synchronization
- **Shader Compilation**: WGSL shader loading and pipeline creation
- **Error Handling**: Comprehensive error callbacks and reporting

### WGSL Compute Shaders

All shaders are optimized for browser compatibility with proper workgroup sizes:

#### 1. Matrix Operations (`matmul.wgsl`)
- **Tiled Matrix Multiplication**: 16x16 tiles with shared memory
- **Matrix-Vector Multiplication**: Optimized for large matrices
- **Batch Matrix Multiplication**: Support for batched operations
- **Transpose Support**: Both transA and transB options
- **Alpha/Beta Scaling**: GEMM-style operations

#### 2. Convolution (`conv2d.wgsl`)
- **Direct Convolution**: Standard 2D convolution
- **Tiled Convolution**: Shared memory optimization
- **Depthwise Convolution**: Efficient depthwise separable convolutions
- **Grouped Convolution**: Support for group parameter
- **Padding/Stride/Dilation**: Full parameter support

#### 3. Pooling (`pooling.wgsl`)
- **Max Pooling**: 2D max pooling with arbitrary kernel sizes
- **Average Pooling**: 2D average pooling
- **Global Pooling**: Global max/average pooling
- **Adaptive Pooling**: Automatic stride/kernel calculation

#### 4. Normalization (`batchnorm.wgsl`)
- **Batch Normalization**: Multi-stage implementation (mean, variance, normalize)
- **Layer Normalization**: For transformer models
- **Group Normalization**: Channel grouping support
- **Running Statistics**: Training/inference mode support

#### 5. Activations (`activations.wgsl`)
- **Basic**: ReLU, LeakyReLU, ELU
- **Smooth**: GELU, Swish/SiLU, Mish
- **Sigmoid Family**: Sigmoid, Tanh, Hardsigmoid, Hardswish
- **Softmax**: Three-stage implementation with numerical stability
- **PReLU**: Parametric activation with per-channel weights

#### 6. Reductions (`reduction.wgsl`)
- **Statistical**: Sum, Mean, Variance, StdDev
- **Comparison**: Max, Min, ArgMax, ArgMin
- **Other**: Product, L2 Norm
- **Shared Memory**: Efficient workgroup-level reductions

#### 7. Transforms (`transform.wgsl`)
- **Transpose**: General N-D transpose with 2D optimization
- **Reshape/Flatten**: Memory layout transformations
- **Slice**: Multi-dimensional slicing with strides
- **Concatenation**: Tensor concatenation along dimensions
- **Stack/Split**: Stack tensors or split along dimensions
- **Tile/Repeat**: Tensor replication
- **Flip**: Reverse along dimensions

#### 8. Math Operations (`math.wgsl`)
- **Binary**: Add, Sub, Mul, Div, Pow
- **Unary**: Sqrt, Exp, Log, Abs, Neg, Sign, Reciprocal
- **Trigonometric**: Sin, Cos, Tan, Asin, Acos, Atan
- **Comparison**: Eq, Ne, Lt, Le, Gt, Ge
- **Broadcasting**: Multi-dimensional broadcasting support
- **Utility**: Clamp, Where, Fill, Copy

#### 9. Indexing (`indexing.wgsl`)
- **Gather/Scatter**: Advanced indexing operations
- **Index Select**: Fast indexing along dimension 0
- **Masked Operations**: Fill, select with boolean masks
- **Take/Put**: Direct index-based access
- **Embedding**: Efficient embedding table lookup
- **One-Hot**: One-hot encoding

### Utilities (`webgpu_utils.hpp`)

- **PipelineBuilder**: Fluent API for pipeline creation
- **BindGroupBuilder**: Easy bind group management
- **ComputePassHelper**: RAII compute pass wrapper
- **WorkgroupCalculation**: Automatic workgroup size optimization
- **UniformBuffer/StorageBuffer**: Templated buffer wrappers
- **DispatchHelper**: Automatic workgroup dispatch calculation
- **ShaderCache**: Compiled shader caching
- **BufferPool**: Buffer reuse for memory efficiency
- **TimingHelper**: GPU profiling support

## Building

### Requirements

One of the following WebGPU implementations:

- **Dawn** (Google's implementation): https://dawn.googlesource.com/dawn
- **wgpu-native** (Rust-based): https://github.com/gfx-rs/wgpu-native

### CMake Configuration

```bash
# Enable WebGPU backend
cmake -DTENZOR_ENABLE_WEBGPU=ON ..

# For Emscripten/WASM builds
emcmake cmake -DTENZOR_ENABLE_WEBGPU=ON ..
```

### Dependencies

```cmake
# Option 1: Dawn
find_package(Dawn REQUIRED)

# Option 2: wgpu-native
find_package(wgpu-native REQUIRED)

# Option 3: pkg-config
pkg_check_modules(WEBGPU REQUIRED webgpu)
```

## Usage

### Basic Example

```cpp
#include "webgpu_backend.hpp"

using namespace tenzor::webgpu;

// Initialize backend
WebGPUConfig config;
config.powerPreference = WGPUPowerPreference_HighPerformance;

WebGPUBackend backend(config);
if (!backend.initialize()) {
    std::cerr << "Failed to initialize WebGPU: " << backend.getLastError() << std::endl;
    return 1;
}

// Create buffers
auto inputA = backend.createStorageBuffer(1024 * sizeof(float));
auto inputB = backend.createStorageBuffer(1024 * sizeof(float));
auto output = backend.createStorageBuffer(1024 * sizeof(float));

// Write data
std::vector<float> dataA(1024, 1.0f);
std::vector<float> dataB(1024, 2.0f);
backend.writeBuffer(*inputA, dataA.data(), dataA.size() * sizeof(float));
backend.writeBuffer(*inputB, dataB.data(), dataB.size() * sizeof(float));

// Load shader
auto pipeline = backend.loadShaderFromFile("kernels/math.wgsl", "add");

// Create bind group
std::vector<std::shared_ptr<WebGPUBuffer>> buffers = {inputA, inputB, output};
auto bindGroup = backend.createBindGroup(*pipeline, buffers);

// Execute
backend.compute(*pipeline, bindGroup, (1024 + 255) / 256);

// Read results
auto resultFuture = backend.readBuffer(*output);
auto result = resultFuture.get();
```

### Matrix Multiplication

```cpp
#include "webgpu_utils.hpp"

using namespace tenzor::webgpu;
using namespace tenzor::webgpu::utils;

// Matrix dimensions: C = A * B
// A: M x K, B: K x N, C: M x N
uint32_t M = 1024, N = 1024, K = 1024;

// Create buffers
StorageBuffer<float> matrixA(backend, M * K);
StorageBuffer<float> matrixB(backend, K * N);
StorageBuffer<float> matrixC(backend, M * N);

// Prepare parameters
struct MatmulParams {
    uint32_t M, N, K;
    uint32_t transA, transB;
    float alpha, beta;
} params = {M, N, K, 0, 0, 1.0f, 0.0f};

UniformBuffer<MatmulParams> uniformBuf(backend, params);

// Load shader
auto pipeline = backend.loadShaderFromFile("kernels/matmul.wgsl", "main");

// Execute with automatic workgroup calculation
DispatchHelper::dispatch2D(backend, *pipeline, bindGroup,
                          (N + 15) / 16, (M + 15) / 16, 16, 16);

// Read results
auto result = matrixC.read().get();
```

### Convolution

```cpp
struct Conv2DParams {
    uint32_t batchSize, inChannels, outChannels;
    uint32_t inHeight, inWidth;
    uint32_t outHeight, outWidth;
    uint32_t kernelHeight, kernelWidth;
    uint32_t strideHeight, strideWidth;
    uint32_t padHeight, padWidth;
    uint32_t dilationHeight, dilationWidth;
    uint32_t groups;
};

Conv2DParams params = {
    .batchSize = 1,
    .inChannels = 3,
    .outChannels = 64,
    .inHeight = 224,
    .inWidth = 224,
    .outHeight = 224,
    .outWidth = 224,
    .kernelHeight = 3,
    .kernelWidth = 3,
    .strideHeight = 1,
    .strideWidth = 1,
    .padHeight = 1,
    .padWidth = 1,
    .dilationHeight = 1,
    .dilationWidth = 1,
    .groups = 1
};

auto pipeline = backend.loadShaderFromFile("kernels/conv2d.wgsl", "conv2d_direct");

// Execute convolution
DispatchHelper::dispatch2D(backend, *pipeline, bindGroup,
                          params.outWidth, params.outHeight, 16, 16);
```

## Browser Compatibility

All shaders are designed for maximum browser compatibility:

- **Workgroup Sizes**: Limited to 256 invocations (16x16 for 2D)
- **Shared Memory**: Conservative usage (<16KB per workgroup)
- **WGSL Version**: Compatible with WebGPU 1.0 specification
- **Features**: No optional features required
- **Async Operations**: Proper async/await support

## Performance Optimizations

1. **Tiling**: Matrix operations use shared memory tiling
2. **Coalesced Access**: Memory access patterns optimized for GPU
3. **Workgroup Reductions**: Efficient parallel reductions
4. **Buffer Pooling**: Reuse buffers to reduce allocations
5. **Pipeline Caching**: Compiled shaders cached for reuse
6. **Async Operations**: Non-blocking GPU operations

## Architecture

```
webgpu_backend/
├── webgpu_backend.hpp      # Main backend interface
├── webgpu_backend.cpp      # Backend implementation
├── webgpu_utils.hpp        # Utility classes and helpers
├── CMakeLists.txt          # Build configuration
└── kernels/                # WGSL compute shaders
    ├── matmul.wgsl         # Matrix multiplication
    ├── conv2d.wgsl         # 2D convolution
    ├── pooling.wgsl        # Pooling operations
    ├── batchnorm.wgsl      # Normalization layers
    ├── activations.wgsl    # Activation functions
    ├── reduction.wgsl      # Reduction operations
    ├── transform.wgsl      # Tensor transformations
    ├── math.wgsl           # Math operations
    └── indexing.wgsl       # Indexing operations
```

## Error Handling

The backend provides comprehensive error handling:

```cpp
backend.setErrorCallback([](WGPUErrorType type, const char* message) {
    std::cerr << "WebGPU Error: " << message << std::endl;
});

// Check for errors
if (!backend.initialize()) {
    std::cerr << "Init failed: " << backend.getLastError() << std::endl;
}
```

## Cross-Platform Support

- **Native**: Works with Dawn or wgpu-native on Windows, Linux, macOS
- **Web**: Compiles to WASM with Emscripten
- **Metal**: Via Dawn on macOS/iOS
- **Vulkan**: Via Dawn on Windows/Linux/Android
- **D3D12**: Via Dawn on Windows

## Testing

```bash
# Run WebGPU backend tests
ctest -L webgpu

# Run specific test
./test_webgpu_backend
```

## Limitations

- Maximum buffer size: 1GB (configurable)
- Maximum workgroup size: 256 (browser compatibility)
- Shared memory per workgroup: 16KB
- No support for float64 (not in WebGPU 1.0 spec)

## Future Enhancements

- [ ] Subgroup operations (when available in browsers)
- [ ] Tensor cores / cooperative matrix (when standardized)
- [ ] Multi-device support
- [ ] Automatic shader optimization
- [ ] JIT shader compilation with specialization constants
- [ ] Memory pool with defragmentation
- [ ] Advanced profiling and debugging tools

## References

- WebGPU Specification: https://www.w3.org/TR/webgpu/
- WGSL Specification: https://www.w3.org/TR/WGSL/
- Dawn Project: https://dawn.googlesource.com/dawn
- wgpu-native: https://github.com/gfx-rs/wgpu-native

## License

Part of the Tenzor project. See main LICENSE file.
