# Frequently Asked Questions (FAQ)

## Table of Contents

- [General Questions](#general-questions)
- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [GPU & Backend Issues](#gpu--backend-issues)
- [Python Bindings](#python-bindings)
- [Training & Models](#training--models)
- [Troubleshooting](#troubleshooting)

---

## General Questions

### What is Tenzor?

Tenzor is a high-performance tensor computation and deep learning library written in modern C++23. It provides a PyTorch-like API with support for multiple hardware backends (CPU, NVIDIA CUDA, AMD ROCm, Intel OneAPI, Vulkan, Apple Metal).

### How does Tenzor compare to PyTorch/TensorFlow?

| Feature | Tenzor | PyTorch | TensorFlow |
|---------|--------|---------|------------|
| Language | C++23 | Python/C++ | Python/C++ |
| Performance | Comparable/faster | Baseline | Comparable |
| Multi-backend | 6 backends | 2 backends | 2 backends |
| Binary size | ~50 MB | ~500 MB | ~1 GB |
| Startup time | <100ms | ~2s | ~3s |
| Memory footprint | Lower | Baseline | Higher |

Tenzor is ideal when you need:
- Native C++ integration
- Smaller deployment size
- Support for diverse hardware (Vulkan, Metal, OneAPI)
- Fine-grained control over execution

### What license is Tenzor released under?

Tenzor is released under the MIT License, allowing free use in both commercial and open-source projects.

### What platforms are supported?

- **Linux**: Ubuntu 20.04+, Fedora 35+, Arch Linux (x86_64, ARM64)
- **macOS**: 12.0+ (Intel and Apple Silicon)
- **Windows**: Windows 10+ (x86_64)

---

## Installation

### What are the minimum system requirements?

- **CPU**: x86_64 with SSE4.2 (AVX2 recommended)
- **RAM**: 4 GB minimum, 16 GB recommended
- **Disk**: 500 MB for base install, 2 GB with all backends
- **Compiler**: GCC 12+, Clang 15+, or MSVC 2022

### How do I install Tenzor?

See the [Installation Guide](../INSTALL.md) for detailed instructions. Quick start:

```bash
git clone https://github.com/leelee222/tenzor.git
cd tenzor && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
sudo cmake --install .
```

### Why does CMake fail to find CUDA?

Ensure CUDA is properly installed and in your PATH:

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Specify CUDA path explicitly
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### How do I build without GPU support?

```bash
cmake .. -DTENZOR_BUILD_CUDA=OFF -DTENZOR_BUILD_ROCM=OFF -DTENZOR_BUILD_ONEAPI=OFF
```

### Why am I getting C++23 compilation errors?

Ensure you're using a compatible compiler:

```bash
# Check GCC version (need 12+)
g++ --version

# Specify compiler explicitly
cmake .. -DCMAKE_CXX_COMPILER=g++-13
```

---

## Usage

### How do I create a tensor?

**C++:**
```cpp
#include <tenzor/tenzor.hpp>

auto a = tenzor::zeros({3, 3});              // 3x3 zeros
auto b = tenzor::randn({3, 3});              // Random normal
auto c = tenzor::tensor({1.0f, 2.0f, 3.0f}); // From data
```

**Python:**
```python
import tenzor as tz

a = tz.zeros([3, 3])
b = tz.randn([3, 3])
c = tz.tensor([1.0, 2.0, 3.0])
```

### How do I move tensors between CPU and GPU?

```cpp
// C++
auto x = tenzor::randn({1000, 1000});
auto x_gpu = x.cuda();     // Move to GPU
auto x_cpu = x_gpu.cpu();  // Move back to CPU
```

```python
# Python
x = tz.randn([1000, 1000])
x_gpu = x.cuda()     # Move to GPU
x_cpu = x_gpu.cpu()  # Move back to CPU
```

### How do I check if GPU is available?

```cpp
// C++
if (tenzor::cuda_available()) {
    std::cout << "CUDA available with " << tenzor::cuda_device_count() << " devices\n";
}
```

```python
# Python
if tz.cuda_available():
    print(f"CUDA available with {tz.cuda_device_count()} devices")
```

### How do I define a custom model?

**C++:**
```cpp
class MyModel : public tenzor::nn::Module {
public:
    MyModel() {
        fc1 = register_module("fc1", std::make_shared<nn::Linear>(784, 128));
        fc2 = register_module("fc2", std::make_shared<nn::Linear>(128, 10));
    }

    Variable forward(const Variable& x) override {
        auto h = nn::relu(fc1->forward(x));
        return fc2->forward(h);
    }

private:
    std::shared_ptr<nn::Linear> fc1, fc2;
};
```

**Python:**
```python
class MyModel(tz.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = tz.nn.Linear(784, 128)
        self.fc2 = tz.nn.Linear(128, 10)

    def forward(self, x):
        h = tz.nn.relu(self.fc1(x))
        return self.fc2(h)
```

### How do I save and load models?

```cpp
// C++
// Save
model->save("model.pt");

// Load
auto loaded_model = std::make_shared<MyModel>();
loaded_model->load("model.pt");
```

```python
# Python
# Save
model.save("model.pt")

# Load
loaded_model = MyModel()
loaded_model.load("model.pt")
```

---

## Performance

### How can I improve training speed?

1. **Use GPU acceleration**: Move model and data to GPU
2. **Enable mixed precision**: Use FP16/BF16 for faster compute
3. **Increase batch size**: Better GPU utilization
4. **Use DataLoader with workers**: Parallel data loading
5. **Enable kernel fusion**: JIT compile your model

```python
# Enable mixed precision
with tz.amp.autocast():
    output = model(input)
    loss = criterion(output, target)
```

### Why is my model slower than PyTorch?

Common causes:
1. **Small batch sizes**: GPU underutilization
2. **Frequent CPU-GPU transfers**: Keep data on GPU
3. **Debug build**: Use Release build
4. **Unoptimized operations**: Check for custom ops

Profile your code:
```cpp
{
    auto profiler = tenzor::utils::Profiler("forward_pass");
    auto output = model->forward(input);
}  // Prints timing
```

### How much VRAM does Tenzor use?

Tenzor uses a caching allocator to minimize allocation overhead. Memory usage depends on:
- Model size
- Batch size
- Activation storage (for backward pass)
- Gradient storage

Check memory usage:
```python
print(f"Allocated: {tz.cuda_memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {tz.cuda_memory_cached() / 1e9:.2f} GB")
```

### How do I reduce memory usage?

1. **Gradient checkpointing**: Trade compute for memory
2. **Mixed precision**: FP16 uses half the memory
3. **Smaller batch sizes**: Reduce activation memory
4. **Clear cache**: `tz.cuda_empty_cache()`

```python
# Enable gradient checkpointing
model.enable_gradient_checkpointing()
```

---

## GPU & Backend Issues

### CUDA out of memory error

**Solutions:**
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision training
4. Clear CUDA cache: `tenzor::cuda_empty_cache()`

```bash
# Set memory limit
export TENZOR_MEMORY_POOL_SIZE=4096  # 4GB limit
```

### Vulkan backend not working

1. **Check Vulkan installation**:
```bash
vulkaninfo | grep "Vulkan Instance"
```

2. **Install Vulkan SDK**:
```bash
# Ubuntu
sudo apt install vulkan-sdk

# macOS
brew install molten-vk
```

3. **Verify GPU support**:
```bash
vulkaninfo | grep "deviceName"
```

### ROCm not detecting AMD GPU

1. **Check ROCm installation**:
```bash
rocm-smi
```

2. **Add user to video group**:
```bash
sudo usermod -aG video $USER
```

3. **Set environment**:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For newer GPUs
```

### OneAPI backend issues

1. **Source environment**:
```bash
source /opt/intel/oneapi/setvars.sh
```

2. **Check device**:
```bash
sycl-ls
```

---

## Python Bindings

### ImportError: No module named 'tenzor'

**Solution 1**: Install the package from source (recommended)
```bash
cd tenzor
pip install .
```

**Solution 2**: Add to PYTHONPATH (after building with CMake)
```bash
export PYTHONPATH=/path/to/tenzor/python:$PYTHONPATH
```

### NumPy interoperability

```python
import numpy as np
import tenzor as tz

# NumPy → Tenzor
np_arr = np.random.randn(3, 3).astype(np.float32)
tz_tensor = tz.from_numpy(np_arr)

# Tenzor → NumPy (creates copy)
np_arr = tz_tensor.numpy()
```

### Python version compatibility

Tenzor supports Python 3.8 - 3.13. Check your version:
```bash
python --version
```

---

## Training & Models

### Loss is NaN or Inf

**Causes and solutions:**

1. **Learning rate too high**: Reduce by 10x
2. **Gradient explosion**: Enable gradient clipping
3. **Numerical instability**: Use mixed precision with loss scaling
4. **Data issues**: Check for NaN/Inf in input

```python
# Enable gradient clipping
tz.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check for NaN
if tz.isnan(loss).any():
    print("NaN detected!")
```

### Model not learning

**Checklist:**
1. Verify data is correct (visualize samples)
2. Check that loss decreases initially
3. Ensure model is in training mode: `model.train()`
4. Verify gradients are non-zero
5. Try a simpler model first

```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean = {param.grad.mean():.6f}")
```

### How do I use pretrained models?

```python
# Load pretrained ResNet
model = tz.models.resnet50(pretrained=True)

# For inference
model.eval()
with tz.no_grad():
    output = model(input)
```

### How do I export to ONNX?

```python
# Export model
tz.onnx.export(model, example_input, "model.onnx")

# Import ONNX model
imported = tz.onnx.load("model.onnx")
```

---

## Troubleshooting

### Segmentation fault

**Common causes:**
1. Out-of-bounds tensor access
2. Use-after-free (dangling tensor reference)
3. Stack overflow (deep recursion)

**Debug steps:**
```bash
# Build with debug symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Run with AddressSanitizer
cmake .. -DTENZOR_ENABLE_ASAN=ON

# Use GDB
gdb ./your_program
```

### Compilation taking too long

**Solutions:**
1. Use Ninja instead of Make
2. Enable parallel compilation: `-j$(nproc)`
3. Use ccache: `cmake .. -DCMAKE_CXX_COMPILER_LAUNCHER=ccache`
4. Disable unused backends

### Tests failing

```bash
# Run specific test
./bin/test_tensor

# Run with verbose output
ctest --output-on-failure

# Run single test case
./bin/test_tensor --gtest_filter="TensorTest.BasicOps"
```

### Where can I get help?

- **GitHub Issues**: [Report bugs](https://github.com/leelee222/tenzor/issues)
- **Discussions**: [Ask questions](https://github.com/leelee222/tenzor/discussions)
- **Documentation**: [API Reference](api/html/index.html)

---

## Still have questions?

If your question isn't answered here, please:
1. Search [existing issues](https://github.com/leelee222/tenzor/issues)
2. Check the [API documentation](api/html/index.html)
3. Open a [new discussion](https://github.com/leelee222/tenzor/discussions/new)
