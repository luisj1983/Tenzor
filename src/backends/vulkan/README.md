# Vulkan Compute Backend for Tenzor

A cross-platform GPU compute backend using Vulkan API, providing high-performance tensor operations on Windows, Linux, and macOS.

## Features

- **Cross-Platform**: Works on Windows, Linux, and macOS with MoltenVK
- **Compute Shaders**: All operations implemented in GLSL compute shaders
- **Optimized Memory Management**: Staging buffers for efficient host-device transfers
- **Pipeline Caching**: Compiled shaders cached for performance
- **Buffer Management**: Device-local memory with automatic staging

## Implemented Operations

### Math Operations (math.comp)
- Element-wise: add, sub, mul, div
- Unary: sqrt, exp, log, neg, abs, sign

### Matrix Operations (matmul.comp)
- Tiled matrix multiplication
- Optimized for GPU cache locality

### Activations (activations.comp)
- ReLU, Sigmoid, Tanh
- GELU (Gaussian Error Linear Unit)
- LeakyReLU with configurable alpha

### Reductions (reduction.comp)
- Sum, mean, max, min
- Parallel reduction using shared memory
- Support for dimension-specific reductions

### Convolution (conv2d.comp)
- 2D convolution with bias
- Configurable stride, padding, dilation
- Support for grouped convolutions

### Pooling (pooling.comp)
- MaxPool2d
- AvgPool2d
- Configurable kernel size and stride

### Batch Normalization (batchnorm.comp)
- Forward pass with optional affine transformation
- Epsilon parameter for numerical stability

### Transforms (transform.comp)
- Reshape, transpose, permute
- Memory-efficient view operations

### Indexing (indexing.comp)
- Gather and scatter operations
- Slicing along dimensions

## Requirements

### Vulkan SDK
Install the Vulkan SDK from [LunarG](https://vulkan.lunarg.com/):

**Linux:**
```bash
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list https://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list
sudo apt update
sudo apt install vulkan-sdk
```

**macOS:**
```bash
brew install molten-vk vulkan-headers vulkan-loader
```

**Windows:**
Download and install from [LunarG SDK Downloads](https://vulkan.lunarg.com/sdk/home)

### GPU Drivers
Ensure you have up-to-date GPU drivers with Vulkan support:
- **NVIDIA**: Driver 450+ (Vulkan 1.2+)
- **AMD**: Adrenalin 20.11.2+ (Vulkan 1.2+)
- **Intel**: Mesa 20.2+ (Linux) or latest Windows drivers

## Building

The Vulkan backend is automatically built if the Vulkan SDK is detected:

```bash
mkdir build
cd build
cmake .. -DTENZOR_BUILD_VULKAN=ON
make -j$(nproc)
```

### Environment Variables

**VULKAN_SDK**: Path to Vulkan SDK (auto-detected on most systems)
```bash
export VULKAN_SDK=/path/to/vulkan/sdk
```

**TENZOR_VULKAN_SHADER_PATH**: Custom shader path (optional)
```bash
export TENZOR_VULKAN_SHADER_PATH=/path/to/shaders/
```

## Usage

```cpp
#include <tenzor/tensor.hpp>

// Create tensors on Vulkan device
auto device = Device::vulkan(0);
auto a = tenzor::randn({1024, 1024}, DType::Float32, device);
auto b = tenzor::randn({1024, 1024}, DType::Float32, device);

// Perform operations
auto c = a.matmul(b);
auto d = c.relu();

// Synchronize device
device.synchronize();
```

## Architecture

### Backend Structure
```
vulkan_backend.cpp      - Main backend implementation
vulkan_backend.hpp      - Backend interface
vulkan_utils.hpp        - Utility classes and helpers
kernels/                - GLSL compute shaders
  ├── math.comp         - Arithmetic operations
  ├── matmul.comp       - Matrix multiplication
  ├── activations.comp  - Activation functions
  ├── reduction.comp    - Reduction operations
  ├── conv2d.comp       - 2D convolution
  ├── pooling.comp      - Pooling operations
  ├── batchnorm.comp    - Batch normalization
  ├── transform.comp    - Tensor transformations
  └── indexing.comp     - Indexing operations
```

### Memory Management

1. **Device Memory**: VkBuffer with `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`
2. **Staging Buffers**: Host-visible memory for transfers
3. **Buffer Caching**: Reuse staging buffers to avoid allocation overhead

### Pipeline Management

1. **Shader Compilation**: GLSL → SPIR-V at build time
2. **Pipeline Creation**: Lazy creation on first use
3. **Pipeline Caching**: Compiled pipelines cached per device

### Execution Model

1. **Command Buffers**: Single-use command buffers for operations
2. **Synchronization**: Fence-based synchronization for host
3. **Compute Queues**: Dedicated compute queue per device

## Performance Considerations

### Optimizations
- **Tiled Matrix Multiplication**: 16x16 tiles with shared memory
- **Parallel Reductions**: Two-stage reduction for large tensors
- **Memory Coalescing**: Aligned memory accesses in shaders
- **Pipeline Reuse**: Cached pipelines avoid recompilation

### Workgroup Sizes
- Math operations: 256 threads (1D)
- Matrix multiplication: 16x16 threads (2D)
- Convolution: 16x16 threads (2D)
- Reductions: 256 threads with shared memory

## Debugging

Enable Vulkan validation layers:

```bash
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
```

View shader compilation:
```bash
glslc -fshader-stage=compute kernels/matmul.comp -o matmul.spv
spirv-dis matmul.spv  # Disassemble SPIR-V
```

## Known Limitations

1. **Descriptor Sets**: Currently uses fixed descriptor layout
2. **Async Operations**: Limited async execution support
3. **Multi-Device**: Single device operations only
4. **FP16 Support**: Requires device extension check
5. **Conv2d**: Basic implementation, not yet optimized

## Future Improvements

- [ ] Subgroup operations for better performance
- [ ] Push descriptors for reduced overhead
- [ ] Timeline semaphores for async execution
- [ ] Sparse tensor support
- [ ] FP16/BF16 compute shaders
- [ ] Optimized convolution (Winograd, FFT-based)
- [ ] Multi-device tensor distribution

## References

- [Vulkan Specification](https://www.khronos.org/vulkan/)
- [Vulkan Compute Shader Tutorial](https://www.khronos.org/blog/vulkan-compute-shaders)
- [GLSL Language Specification](https://www.khronos.org/opengl/wiki/Core_Language_(GLSL))
- [SPIR-V Specification](https://www.khronos.org/spir/)

## License

Same as Tenzor project license.
