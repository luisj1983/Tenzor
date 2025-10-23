# Vulkan Backend Quick Start Guide

## Installation

### 1. Install Vulkan SDK

**Ubuntu/Debian:**
```bash
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list \
    https://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list
sudo apt update
sudo apt install vulkan-sdk
```

**Arch Linux:**
```bash
sudo pacman -S vulkan-devel
```

**macOS:**
```bash
brew install molten-vk vulkan-headers vulkan-loader
```

**Windows:**
Download from: https://vulkan.lunarg.com/sdk/home

### 2. Verify Installation

```bash
vulkaninfo | head -n 20
```

You should see your GPU(s) listed with Vulkan support.

## Building the Backend

### From Tenzor root directory:

```bash
mkdir -p build
cd build

# Configure with Vulkan enabled
cmake .. -DTENZOR_BUILD_VULKAN=ON

# Build
make -j$(nproc)

# Verify shaders were compiled
ls build/shaders/vulkan/*.spv
```

Expected output:
```
activations.spv
batchnorm.spv
conv2d.spv
indexing.spv
math.spv
matmul.spv
pooling.spv
reduction.spv
transform.spv
```

## Quick Test

### C++ Example

```cpp
#include <tenzor/tensor.hpp>
#include <iostream>

int main() {
    // Check Vulkan availability
    auto backend = tenzor::get_backend("vulkan");
    if (!backend->is_available()) {
        std::cerr << "Vulkan not available!\n";
        return 1;
    }

    std::cout << "Vulkan devices: " << backend->device_count() << "\n";

    // Create tensors on Vulkan device
    auto device = tenzor::Device::vulkan(0);

    // Simple math operations
    auto a = tenzor::randn({256, 256}, tenzor::DType::Float32, device);
    auto b = tenzor::randn({256, 256}, tenzor::DType::Float32, device);
    auto c = a + b;

    // Matrix multiplication
    auto d = a.matmul(b);

    // Activations
    auto e = d.relu();

    // Synchronize and copy back
    device.synchronize();
    auto result = e.cpu();

    std::cout << "Result shape: ";
    for (auto dim : result.shape()) {
        std::cout << dim << " ";
    }
    std::cout << "\n";

    return 0;
}
```

### Python Example (with bindings)

```python
import tenzor

# Check Vulkan
backend = tenzor.get_backend("vulkan")
print(f"Vulkan available: {backend.is_available()}")
print(f"Devices: {backend.device_count()}")

# Create tensors
device = tenzor.Device.vulkan(0)
a = tenzor.randn([1024, 1024], device=device)
b = tenzor.randn([1024, 1024], device=device)

# Compute
c = a @ b  # matmul
c = c.relu()

# Synchronize
device.synchronize()

# Copy to CPU
result = c.cpu()
print(f"Result: {result.shape}")
```

## Performance Testing

### Benchmark Script

```bash
# From build directory
./benchmark_vulkan --operations matmul,conv2d,relu \
                   --sizes 512,1024,2048 \
                   --iterations 100
```

### Expected Performance

On NVIDIA RTX 3080 (example):
- **MatMul (1024x1024)**: ~0.15ms
- **Conv2d (3x3, 64 channels)**: ~0.8ms
- **ReLU (1M elements)**: ~0.02ms

## Troubleshooting

### Problem: "Vulkan not available"

**Solution:**
```bash
# Check Vulkan devices
vulkaninfo --summary

# Check GPU support
lspci | grep -i vga

# Update drivers
sudo ubuntu-drivers autoinstall  # Ubuntu
```

### Problem: "glslc not found"

**Solution:**
```bash
# Find glslc
which glslc

# If not found, set VULKAN_SDK
export VULKAN_SDK=/usr/local/VulkanSDK/1.3.xxx
export PATH=$VULKAN_SDK/bin:$PATH
```

### Problem: Shader compilation errors

**Solution:**
```bash
# Manually compile shaders
cd src/backends/vulkan/kernels
glslc -fshader-stage=compute math.comp -o math.spv

# Check for errors
echo $?  # Should be 0
```

### Problem: Validation errors

**Enable validation layers for debugging:**
```bash
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d

# Run your program
./your_program
```

### Problem: Memory leaks

**Use Vulkan validation layers:**
```bash
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT

# Should report any leaks on exit
./your_program
```

## Environment Variables

### Required
- `VULKAN_SDK`: Path to Vulkan SDK (usually auto-detected)

### Optional
- `TENZOR_VULKAN_SHADER_PATH`: Custom shader directory
- `VK_INSTANCE_LAYERS`: Enable validation layers
- `VK_LOADER_DEBUG`: Vulkan loader debug output

### Example Configuration

```bash
# ~/.bashrc or ~/.zshrc
export VULKAN_SDK=/usr/local/VulkanSDK/1.3.xxx
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
export VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d

# For debugging (optional)
# export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
```

## Profiling

### RenderDoc

1. Install RenderDoc: https://renderdoc.org/
2. Launch your application through RenderDoc
3. Capture frames to see GPU operations
4. Analyze shader performance

### Vulkan Profiler

```bash
# NVIDIA Nsight Systems
nsys profile --trace vulkan ./your_program

# AMD Radeon GPU Profiler
RadeonGPUProfiler --trace vulkan ./your_program
```

## Next Steps

1. **Read the README**: `src/backends/vulkan/README.md`
2. **Review examples**: `examples/vulkan/`
3. **Run tests**: `build/tests/test_vulkan_backend`
4. **Benchmark**: Compare with CUDA/ROCm backends
5. **Optimize**: Profile and tune your workloads

## Common Operations

### Matrix Multiplication
```cpp
auto c = a.matmul(b);  // Uses matmul.comp shader
```

### Convolution
```cpp
auto output = tenzor::conv2d(input, weight, bias,
    /*stride=*/1, /*padding=*/1);  // Uses conv2d.comp
```

### Activations
```cpp
auto relu_out = input.relu();      // activations.comp (op=0)
auto sigmoid_out = input.sigmoid(); // activations.comp (op=1)
auto tanh_out = input.tanh();      // activations.comp (op=2)
```

### Reductions
```cpp
auto sum = tensor.sum();           // reduction.comp (op=0)
auto mean = tensor.mean();         // reduction.comp (op=1)
auto max = tensor.max();           // reduction.comp (op=2)
```

### Pooling
```cpp
auto max_pool = tenzor::max_pool2d(input,
    /*kernel_size=*/2, /*stride=*/2);  // pooling.comp (type=0)

auto avg_pool = tenzor::avg_pool2d(input,
    /*kernel_size=*/2, /*stride=*/2);  // pooling.comp (type=1)
```

## Support

- **Documentation**: `src/backends/vulkan/README.md`
- **Examples**: `examples/vulkan/`
- **Tests**: `tests/vulkan/`
- **Issues**: GitHub Issues with `backend:vulkan` label

## Additional Resources

- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [Vulkan Compute Examples](https://github.com/SaschaWillems/Vulkan)
- [GLSL Compute Shader Tutorial](https://www.khronos.org/opengl/wiki/Compute_Shader)
- [Tenzor Documentation](https://docs.tenzor.io)

---

**Last Updated**: 2025-10-23
**Vulkan Backend Version**: 1.0.0
