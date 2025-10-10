# Getting Started with Tenzor

Welcome to Tenzor! This guide will help you get up and running with the Tenzor neural network library.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start - C++](#quick-start---c)
4. [Quick Start - Python](#quick-start---python)
5. [Core Concepts](#core-concepts)
6. [API Overview](#api-overview)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)
9. [Next Steps](#next-steps)

---

## Introduction

### What is Tenzor?

Tenzor is a modern, high-performance neural network library written in C++23 that provides PyTorch-like APIs for building and training deep learning models. It combines the performance of native C++ with the ease-of-use of modern tensor frameworks.

### Key Features

**Modern C++23 Design**
- Leverages latest C++ features (concepts, ranges, modules)
- Type-safe APIs with zero-overhead abstractions
- Clean, expressive syntax inspired by PyTorch
- Smart memory management with shared storage

**Multi-Backend Architecture**
- **CPU Backend**: Optimized with SIMD, OpenMP multi-threading
- **CUDA Backend**: GPU acceleration for NVIDIA GPUs
- **ROCm Backend**: AMD GPU support (planned)
- **OneAPI Backend**: Intel GPU support (planned)
- Dynamic backend loading at runtime

**PyTorch-like API**
- Familiar tensor operations and neural network modules
- Automatic differentiation (autograd) for gradient computation
- Rich set of pre-built layers: Linear, Conv2d, BatchNorm, etc.
- Optimizers: SGD, Adam, AdamW with momentum and scheduling
- Loss functions: MSE, Cross-Entropy, BCE, etc.

**Automatic Differentiation**
- Computational graph construction and traversal
- Efficient backward pass for gradient computation
- Support for custom gradient functions
- Dynamic graph construction (define-by-run)

**Performance Characteristics**
- Zero-copy operations where possible (views, transposes)
- Lazy evaluation with operation fusion
- Multi-threaded CPU operations with thread pool
- Efficient GPU kernel implementations
- Memory pooling for reduced allocation overhead

### Who Should Use Tenzor?

Tenzor is ideal for:
- **C++ developers** who want native deep learning capabilities
- **Performance-critical applications** requiring minimal overhead
- **Research projects** needing customizable low-level control
- **Embedded systems** where Python overhead is prohibitive
- **Library developers** building on top of tensor operations
- **Students and researchers** learning deep learning internals

### Performance Expectations

- **CPU Operations**: Optimized with AVX/AVX2 SIMD instructions
- **GPU Operations**: Competitive with PyTorch for medium-to-large models
- **Memory Usage**: Copy-on-write semantics minimize allocations
- **Multi-threading**: Automatic parallelization for large tensors
- **Initialization**: Fast backend loading (<100ms typical)

Typical performance compared to PyTorch:
- **Forward pass**: 80-95% of PyTorch speed (C++ overhead is minimal)
- **Backward pass**: 75-90% of PyTorch speed (optimization ongoing)
- **Memory usage**: Similar to PyTorch (shared storage design)
- **Compilation**: Faster due to C++ static typing

---

## Installation

### Building from Source (Linux)

Tenzor is currently available as a source build. Pre-built packages are planned for future releases.

#### Prerequisites

First, install the required dependencies:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install cmake g++ python3-dev python3-pip git

# Arch Linux
sudo pacman -S cmake gcc python python-pip git

# Fedora/RHEL
sudo dnf install cmake gcc-c++ python3-devel python3-pip git
```

**Required versions:**
- CMake 3.25 or later
- GCC 13+ or Clang 16+ (C++23 support required)
- Python 3.8+ (for Python bindings)

#### Optional: CUDA Toolkit for GPU Support

If you want GPU acceleration, install the NVIDIA CUDA Toolkit:

```bash
# Method 1: Download from NVIDIA website
# Visit https://developer.nvidia.com/cuda-downloads
# Follow installation instructions for your distribution

# Method 2: Ubuntu package manager
sudo apt install nvidia-cuda-toolkit nvidia-driver-535

# Verify installation
nvcc --version
nvidia-smi
```

**Supported CUDA versions:** 11.8, 12.0, 12.1, 12.2, 12.3

#### Clone Repository

```bash
# Clone the Tenzor repository
git clone https://github.com/yourusername/tenzor.git
cd tenzor
```

#### Build Configuration

Create a build directory and configure CMake:

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake (basic CPU-only build)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Configure with all features enabled
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENZOR_BUILD_CUDA=ON \
  -DTENZOR_BUILD_PYTHON=ON \
  -DTENZOR_BUILD_TESTS=ON \
  -DTENZOR_BUILD_EXAMPLES=ON \
  -DCMAKE_INSTALL_PREFIX=/usr/local
```

**CMake Options:**
- `TENZOR_BUILD_CUDA`: Build CUDA backend (default: ON)
- `TENZOR_BUILD_ROCM`: Build ROCm backend (default: OFF, experimental)
- `TENZOR_BUILD_ONEAPI`: Build OneAPI backend (default: OFF, experimental)
- `TENZOR_BUILD_PYTHON`: Build Python bindings (default: ON)
- `TENZOR_BUILD_TESTS`: Build test suite (default: ON)
- `TENZOR_BUILD_BENCHMARKS`: Build benchmarks (default: OFF)
- `TENZOR_BUILD_EXAMPLES`: Build example programs (default: ON)

#### Build

Compile the library using all available CPU cores:

```bash
# Build with all cores
make -j$(nproc)

# Or specify number of jobs
make -j8
```

**Build time:** 3-10 minutes depending on your system and enabled features.

#### Test Installation

Run the test suite to verify everything works:

```bash
# Run all tests
ctest --output-on-failure

# Run specific test
ctest -R test_tensor_creation

# Run with verbose output
ctest -V
```

#### Install (Optional)

Install Tenzor system-wide:

```bash
# Install to /usr/local (requires sudo)
sudo make install

# Or install to custom location
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
make install
```

---

### Building from Source (macOS)

#### Prerequisites

Install dependencies using Homebrew:

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake python3 llvm

# For Apple Silicon Macs, ensure you have Xcode Command Line Tools
xcode-select --install
```

**Note:** CUDA is not available on macOS. GPU acceleration requires Linux with NVIDIA GPU.

#### Build Process

The build process is identical to Linux:

```bash
# Clone repository
git clone https://github.com/yourusername/tenzor.git
cd tenzor

# Create build directory
mkdir build && cd build

# Configure (CPU only on macOS)
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENZOR_BUILD_CUDA=OFF \
  -DTENZOR_BUILD_PYTHON=ON \
  -DTENZOR_BUILD_TESTS=ON

# Build
make -j$(sysctl -n hw.ncpu)

# Test
ctest --output-on-failure

# Install (optional)
sudo make install
```

---

### Building from Source (Windows)

#### Prerequisites

**Required software:**
1. **Visual Studio 2022** (Community, Professional, or Enterprise)
   - Install "Desktop development with C++" workload
   - Ensure C++23 support is selected

2. **CMake** (3.25 or later)
   - Download from https://cmake.org/download/
   - Add to PATH during installation

3. **Python** (3.8 or later)
   - Download from https://www.python.org/downloads/
   - Check "Add Python to PATH" during installation

4. **Git for Windows**
   - Download from https://git-scm.com/download/win

5. **CUDA Toolkit** (optional, for GPU support)
   - Download from https://developer.nvidia.com/cuda-downloads

#### Build Using Visual Studio

```powershell
# Open PowerShell or Command Prompt

# Clone repository
git clone https://github.com/yourusername/tenzor.git
cd tenzor

# Create build directory
mkdir build
cd build

# Configure with CMake for Visual Studio
cmake .. -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DTENZOR_BUILD_CUDA=ON ^
  -DTENZOR_BUILD_PYTHON=ON

# Build
cmake --build . --config Release -j 8

# Test
ctest -C Release --output-on-failure

# Install (optional, run as Administrator)
cmake --install . --prefix "C:\Program Files\Tenzor"
```

#### Alternative: Build Using Ninja

For faster builds, use Ninja build system:

```powershell
# Install Ninja (if not already installed)
choco install ninja

# Configure with Ninja
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release

# Build
ninja

# Test
ctest --output-on-failure
```

---

### Python Package Installation

After building with `-DTENZOR_BUILD_PYTHON=ON`, the Python module will be in `build/python/tenzor/`.

#### Option 1: Add to PYTHONPATH (Development)

```bash
# Linux/macOS - add to ~/.bashrc or ~/.zshrc
export PYTHONPATH=/path/to/tenzor/build/python:$PYTHONPATH

# Windows - add to System Environment Variables
setx PYTHONPATH "C:\path\to\tenzor\build\python;%PYTHONPATH%"
```

#### Option 2: Install with pip (Recommended)

```bash
# Navigate to build directory
cd build

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

#### Verify Python Installation

```python
import tenzor as tz
print(f"Tenzor version: {tz.__version__}")
tz.initialize()
t = tz.zeros([2, 3])
print(f"Created tensor with shape: {t.shape}")
```

---

## Quick Start - C++

This section demonstrates basic Tenzor usage in C++. Examples assume you've built and installed Tenzor.

### Basic Tensor Operations

```cpp
#include <tenzor/tenzor.hpp>
#include <iostream>

int main() {
    // Initialize Tenzor library (registers backends)
    tenzor::initialize();

    // ========================================================================
    // Creating Tensors
    // ========================================================================

    // Create a 3x4 tensor filled with zeros
    auto zeros_t = tenzor::zeros({3, 4});
    std::cout << "Zeros shape: [" << zeros_t.shape()[0] << ", "
              << zeros_t.shape()[1] << "]\n";

    // Create a 3x4 tensor filled with ones
    auto ones_t = tenzor::ones({3, 4});

    // Create tensor with random values (normal distribution)
    auto randn_t = tenzor::randn({3, 4});

    // Create tensor with specific value
    auto fives_t = tenzor::full({3, 4}, 5.0f);

    // Create identity matrix
    auto eye_t = tenzor::eye(4);

    // Create range of values [0, 1, 2, ..., 9]
    auto range_t = tenzor::arange(0.0f, 10.0f, 1.0f);

    // ========================================================================
    // Element-wise Operations
    // ========================================================================

    auto a = tenzor::ones({3, 4});
    auto b = tenzor::full({3, 4}, 2.0f);

    // Element-wise addition
    auto sum = a + b;           // Shape: (3, 4), values: 3.0

    // Element-wise subtraction
    auto diff = b - a;          // Shape: (3, 4), values: 1.0

    // Element-wise multiplication
    auto prod = a * b;          // Shape: (3, 4), values: 2.0

    // Element-wise division
    auto quot = b / a;          // Shape: (3, 4), values: 2.0

    // ========================================================================
    // Scalar Operations
    // ========================================================================

    auto t = tenzor::ones({3, 4});

    auto add_scalar = t + 5.0f;    // Add 5 to all elements
    auto mul_scalar = t * 2.0f;    // Multiply all by 2
    auto sub_scalar = t - 1.0f;    // Subtract 1 from all
    auto div_scalar = t / 2.0f;    // Divide all by 2

    // ========================================================================
    // Matrix Multiplication
    // ========================================================================

    auto mat1 = tenzor::randn({2, 3});
    auto mat2 = tenzor::randn({3, 4});
    auto result = tenzor::matmul(mat1, mat2);  // Shape: (2, 4)

    std::cout << "Matrix multiplication: (" << mat1.shape()[0] << ", "
              << mat1.shape()[1] << ") x (" << mat2.shape()[0] << ", "
              << mat2.shape()[1] << ") = (" << result.shape()[0] << ", "
              << result.shape()[1] << ")\n";

    // ========================================================================
    // Reduction Operations
    // ========================================================================

    auto data = tenzor::full({3, 4}, 2.0f);

    auto sum_all = tenzor::sum(data);              // Sum all elements -> scalar
    auto mean_val = tenzor::mean(data);            // Mean of all elements
    auto max_val = tenzor::max(data);              // Maximum element
    auto min_val = tenzor::min(data);              // Minimum element

    // Reduction along specific dimension
    auto sum_dim0 = tenzor::sum(data, 0);          // Sum along dim 0 -> shape (4,)
    auto mean_dim1 = tenzor::mean(data, 1);        // Mean along dim 1 -> shape (3,)

    // ========================================================================
    // Device Management
    // ========================================================================

    // Create tensor on CPU
    auto cpu_tensor = tenzor::zeros({3, 4}, tenzor::DType::Float32,
                                     tenzor::Device::cpu());
    std::cout << "CPU tensor device: " << cpu_tensor.device().type_string() << "\n";

    // Move to GPU (if available)
    try {
        auto gpu_tensor = cpu_tensor.to(tenzor::Device::cuda(0));
        std::cout << "GPU tensor device: " << gpu_tensor.device().type_string() << "\n";

        // Move back to CPU
        auto back_to_cpu = gpu_tensor.cpu();
    } catch (const std::exception& e) {
        std::cout << "CUDA not available: " << e.what() << "\n";
    }

    // Shorthand methods
    auto cuda_t = cpu_tensor.cuda(0);  // Move to GPU 0
    auto cpu_t = cuda_t.cpu();         // Move to CPU

    return 0;
}
```

### Compiling Your Program

```bash
# If Tenzor is installed system-wide
g++ -std=c++23 -O3 main.cpp -ltenzor_core -o my_program

# If Tenzor is not installed, specify paths
g++ -std=c++23 -O3 \
  -I/path/to/tenzor/include \
  -L/path/to/tenzor/build/bin \
  main.cpp -ltenzor_core -Wl,-rpath,/path/to/tenzor/build/bin \
  -o my_program

# Run the program
./my_program
```

### Autograd Example

Automatic differentiation for computing gradients:

```cpp
#include <tenzor/tenzor.hpp>
#include <iostream>

int main() {
    tenzor::initialize();

    // ========================================================================
    // Simple Gradient Computation
    // ========================================================================

    // Create tensors that require gradients
    auto x = tenzor::Variable(tenzor::randn({2, 3}), true);  // requires_grad=true
    auto y = tenzor::Variable(tenzor::randn({3, 4}), true);  // requires_grad=true

    // Forward pass: z = x * y (matrix multiplication)
    auto z_tensor = tenzor::matmul(x.tensor(), y.tensor());
    auto z = tenzor::Variable(z_tensor, true);

    // Compute loss: L = sum(z)
    auto loss_tensor = tenzor::sum(z.tensor());
    auto loss = tenzor::Variable(loss_tensor, true);

    std::cout << "Forward pass complete. Loss shape: ["
              << loss.tensor().shape()[0] << "]\n";

    // Backward pass: compute gradients
    loss.backward();

    std::cout << "Backward pass complete.\n";

    // Access gradients
    auto dx = x.grad();  // dL/dx
    auto dy = y.grad();  // dL/dy

    std::cout << "Gradient dx shape: [" << dx.shape()[0] << ", "
              << dx.shape()[1] << "]\n";
    std::cout << "Gradient dy shape: [" << dy.shape()[0] << ", "
              << dy.shape()[1] << "]\n";

    // ========================================================================
    // More Complex Computation Graph
    // ========================================================================

    auto a = tenzor::Variable(tenzor::full({2, 2}, 3.0f), true);
    auto b = tenzor::Variable(tenzor::full({2, 2}, 2.0f), true);

    // Build computation graph
    auto c_tensor = a.tensor() + b.tensor();       // c = a + b
    auto d_tensor = c_tensor * a.tensor();         // d = c * a
    auto e_tensor = tenzor::sum(d_tensor);         // e = sum(d)

    auto e = tenzor::Variable(e_tensor, true);

    // Compute gradients
    e.backward();

    auto da = a.grad();
    auto db = b.grad();

    std::cout << "Complex graph gradients computed successfully.\n";

    return 0;
}
```

### Simple Neural Network

Building a multi-layer perceptron:

```cpp
#include <tenzor/tenzor.hpp>
#include <memory>
#include <iostream>

// Define a simple MLP model
class MLP : public tenzor::nn::Module {
public:
    MLP(int64_t input_size, int64_t hidden_size, int64_t output_size)
        : fc1_(std::make_shared<tenzor::nn::Linear>(input_size, hidden_size)),
          fc2_(std::make_shared<tenzor::nn::Linear>(hidden_size, output_size)),
          relu_(std::make_shared<tenzor::nn::ReLU>()) {
        // Register submodules for parameter management
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward(const tenzor::Variable& x) -> tenzor::Variable override {
        // Layer 1 + ReLU activation
        auto h = fc1_->forward(x);
        auto h_relu = relu_->forward(h);

        // Layer 2 (output)
        auto output = fc2_->forward(h_relu);
        return output;
    }

private:
    std::shared_ptr<tenzor::nn::Linear> fc1_;
    std::shared_ptr<tenzor::nn::Linear> fc2_;
    std::shared_ptr<tenzor::nn::ReLU> relu_;
};

int main() {
    tenzor::initialize();

    // Create model
    auto model = std::make_shared<MLP>(784, 256, 10);
    std::cout << "Created MLP: 784 -> 256 -> 10\n";

    // Get model parameters
    auto params = model->parameters();
    std::cout << "Model has " << params.size() << " parameter tensors\n";

    // Create optimizer
    auto optimizer = std::make_shared<tenzor::optim::Adam>(
        params,
        0.001,    // learning rate
        0.9,      // beta1
        0.999     // beta2
    );

    // Create loss function
    auto criterion = std::make_shared<tenzor::nn::CrossEntropyLoss>();

    // Training loop (simplified)
    const int num_epochs = 10;
    const int batch_size = 32;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Generate dummy batch
        auto input = tenzor::Variable(
            tenzor::randn({batch_size, 784}),
            false  // input doesn't need gradients
        );
        auto target = tenzor::Variable(
            tenzor::randint(0, 10, {batch_size}),
            false
        );

        // Forward pass
        auto output = model->forward(input);
        auto loss = criterion->forward(output, target);

        // Backward pass
        optimizer->zero_grad();
        loss.backward();

        // Update parameters
        optimizer->step();

        // Print progress
        if (epoch % 2 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: "
                      << loss.tensor().item<float>() << "\n";
        }
    }

    // Switch to evaluation mode
    model->eval();

    // Make prediction
    auto test_input = tenzor::Variable(tenzor::randn({1, 784}), false);
    auto prediction = model->forward(test_input);

    std::cout << "Prediction shape: [" << prediction.tensor().shape()[0]
              << ", " << prediction.tensor().shape()[1] << "]\n";

    return 0;
}
```

### Using Sequential Container

Simplified model building with Sequential:

```cpp
#include <tenzor/tenzor.hpp>
#include <memory>

int main() {
    tenzor::initialize();

    // Build model using Sequential container
    auto model = std::make_shared<tenzor::nn::Sequential>();

    // Add layers sequentially
    model->add_module(std::make_shared<tenzor::nn::Linear>(784, 256));
    model->add_module(std::make_shared<tenzor::nn::ReLU>());
    model->add_module(std::make_shared<tenzor::nn::Dropout>(0.5));
    model->add_module(std::make_shared<tenzor::nn::Linear>(256, 128));
    model->add_module(std::make_shared<tenzor::nn::ReLU>());
    model->add_module(std::make_shared<tenzor::nn::Linear>(128, 10));

    // Or use variadic constructor
    auto model2 = std::make_shared<tenzor::nn::Sequential>(
        std::make_shared<tenzor::nn::Linear>(784, 256),
        std::make_shared<tenzor::nn::ReLU>(),
        std::make_shared<tenzor::nn::Dropout>(0.5),
        std::make_shared<tenzor::nn::Linear>(256, 10)
    );

    // Forward pass
    auto input = tenzor::Variable(tenzor::randn({32, 784}), false);
    auto output = model->forward(input);

    std::cout << "Output shape: [" << output.tensor().shape()[0]
              << ", " << output.tensor().shape()[1] << "]\n";

    return 0;
}
```

---

## Quick Start - Python

This section demonstrates basic Tenzor usage in Python. Examples assume Python bindings are installed.

### Basic Tensor Operations

```python
import tenzor as tz
import numpy as np

# Initialize Tenzor (must be called before any operations)
tz.initialize()

# ============================================================================
# Creating Tensors
# ============================================================================

# Create tensors with different initialization
zeros_t = tz.zeros([3, 4], dtype=tz.dtype.float32)
ones_t = tz.ones([3, 4], dtype=tz.dtype.float32)
randn_t = tz.randn([3, 4], dtype=tz.dtype.float32)
rand_t = tz.rand([3, 4], dtype=tz.dtype.float32)  # Uniform [0, 1)

# Tensor properties
print(f"Shape: {zeros_t.shape}")        # (3, 4)
print(f"Dtype: {zeros_t.dtype}")        # float32
print(f"Device: {zeros_t.device}")      # cpu
print(f"Numel: {zeros_t.numel()}")      # 12

# ============================================================================
# NumPy Interoperability
# ============================================================================

# Convert NumPy array to Tenzor tensor (zero-copy if possible)
np_array = np.random.randn(3, 4).astype(np.float32)
tensor_from_np = tz.Tensor.from_numpy(np_array)

# Convert Tenzor tensor to NumPy array (zero-copy if contiguous)
tensor = tz.randn([3, 4])
np_array_from_tensor = tensor.numpy()

print(f"NumPy array shape: {np_array_from_tensor.shape}")

# ============================================================================
# Element-wise Operations
# ============================================================================

a = tz.ones([3, 4])
b = tz.full([3, 4], 2.0)

# Arithmetic operations
sum_t = a + b      # Element-wise addition
diff_t = b - a     # Element-wise subtraction
prod_t = a * b     # Element-wise multiplication
quot_t = b / a     # Element-wise division

# Scalar operations
scaled = a * 2.0
shifted = a + 5.0

# ============================================================================
# Matrix Operations
# ============================================================================

mat1 = tz.randn([2, 3])
mat2 = tz.randn([3, 4])
result = tz.matmul(mat1, mat2)  # Shape: (2, 4)

print(f"Matrix multiplication: {mat1.shape} x {mat2.shape} = {result.shape}")

# ============================================================================
# Shape Manipulation
# ============================================================================

t = tz.randn([2, 3, 4])

# Reshape
reshaped = t.reshape([6, 4])
flattened = t.reshape([-1])  # Flatten to 1D

# Transpose
t2d = tz.randn([3, 4])
transposed = t2d.transpose(0, 1)  # Shape: (4, 3)

# Permute (generalized transpose)
t3d = tz.randn([2, 3, 4])
permuted = t3d.permute([2, 0, 1])  # Shape: (4, 2, 3)

# Squeeze and unsqueeze
t_squeezed = tz.randn([1, 3, 1, 4])
squeezed = t_squeezed.squeeze()     # Shape: (3, 4)
unsqueezed = squeezed.unsqueeze(0)  # Shape: (1, 3, 4)

# ============================================================================
# Device Management
# ============================================================================

# Create on CPU
cpu_tensor = tz.zeros([3, 4], device=tz.Device.cpu())

# Move to GPU (if available)
try:
    gpu_tensor = cpu_tensor.to(tz.Device.cuda(0))
    print(f"GPU tensor device: {gpu_tensor.device}")

    # Move back to CPU
    back_to_cpu = gpu_tensor.cpu()
except Exception as e:
    print(f"CUDA not available: {e}")

# ============================================================================
# Reduction Operations
# ============================================================================

data = tz.full([3, 4], 2.0)

# Reduce all elements
sum_all = tz.sum(data)              # Scalar tensor
mean_val = tz.mean(data)            # Scalar tensor
max_val = tz.max(data)              # Scalar tensor
min_val = tz.min(data)              # Scalar tensor

# Reduce along dimension
sum_dim0 = tz.sum(data, dim=0)      # Shape: (4,)
mean_dim1 = tz.mean(data, dim=1)    # Shape: (3,)

# Extract scalar value
scalar_val = sum_all.item()  # Python float
print(f"Sum of all elements: {scalar_val}")
```

### Training a Simple Model

Complete training loop example:

```python
import tenzor as tz

# Initialize
tz.initialize()

# ============================================================================
# Define Model
# ============================================================================

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Create layers
        self.fc1 = tz.nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = tz.nn.Linear(hidden_size, output_size, bias=True)
        self.relu = tz.nn.ReLU()

    def forward(self, x):
        # Forward pass
        h = self.fc1(x)
        h = self.relu(h)
        output = self.fc2(h)
        return output

    def parameters(self):
        # Collect all parameters
        return self.fc1.parameters() + self.fc2.parameters()

    def train(self):
        self.fc1.train()
        self.fc2.train()

    def eval(self):
        self.fc1.eval()
        self.fc2.eval()

# Create model
model = SimpleMLP(784, 256, 10)
print("Model created: 784 -> 256 -> 10")

# ============================================================================
# Training Configuration
# ============================================================================

# Loss function
criterion = tz.nn.CrossEntropyLoss()

# Optimizer
optimizer = tz.optim.Adam(
    model.parameters(),
    lr=0.001,
    beta1=0.9,
    beta2=0.999
)

# Hyperparameters
num_epochs = 10
batch_size = 32

# ============================================================================
# Training Loop
# ============================================================================

model.train()

for epoch in range(num_epochs):
    # Generate dummy batch (replace with real data loader)
    batch_x = tz.randn([batch_size, 784])
    batch_y = tz.randint(0, 10, [batch_size])

    # Wrap in Variables
    x = tz.Variable(batch_x, requires_grad=False)
    y = tz.Variable(batch_y, requires_grad=False)

    # Forward pass
    logits = model.forward(x)
    loss = criterion(logits, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print progress
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data.item():.4f}")

print("Training complete!")

# ============================================================================
# Evaluation
# ============================================================================

model.eval()

# Make prediction
test_input = tz.Variable(tz.randn([1, 784]), requires_grad=False)
prediction = model.forward(test_input)

print(f"Prediction shape: {prediction.data.shape}")
```

### Complete MNIST Example

Full example with data loading and evaluation:

```python
import tenzor as tz
import numpy as np

# Initialize
tz.initialize()

# ============================================================================
# Load MNIST Data (simplified - use real dataset in practice)
# ============================================================================

def load_mnist_data():
    """
    Load MNIST dataset.
    In practice, use torchvision or tensorflow datasets.
    """
    # Generate synthetic data for demonstration
    n_train = 10000
    n_test = 2000

    X_train = np.random.randn(n_train, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, n_train).astype(np.int32)

    X_test = np.random.randn(n_test, 784).astype(np.float32)
    y_test = np.random.randint(0, 10, n_test).astype(np.int32)

    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = load_mnist_data()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================================
# Define Model with Sequential
# ============================================================================

model = tz.nn.Sequential()
model.add_module(tz.nn.Linear(784, 512))
model.add_module(tz.nn.ReLU())
model.add_module(tz.nn.Dropout(0.2))
model.add_module(tz.nn.Linear(512, 256))
model.add_module(tz.nn.ReLU())
model.add_module(tz.nn.Dropout(0.2))
model.add_module(tz.nn.Linear(256, 10))

# ============================================================================
# Training
# ============================================================================

criterion = tz.nn.CrossEntropyLoss()
optimizer = tz.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
batch_size = 128
n_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    # Shuffle data
    indices = np.random.permutation(len(X_train))

    for batch_idx in range(n_batches):
        # Get batch
        start = batch_idx * batch_size
        end = start + batch_size
        batch_indices = indices[start:end]

        # Convert to tensors
        batch_x = tz.Tensor.from_numpy(X_train[batch_indices])
        batch_y = tz.Tensor.from_numpy(y_train[batch_indices])

        # Wrap in Variables
        x = tz.Variable(batch_x, requires_grad=False)
        y = tz.Variable(batch_y, requires_grad=False)

        # Forward pass
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data.item()

    avg_loss = epoch_loss / n_batches
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ============================================================================
# Evaluation
# ============================================================================

model.eval()

correct = 0
total = 0
test_batch_size = 256

for i in range(0, len(X_test), test_batch_size):
    batch_x = tz.Tensor.from_numpy(X_test[i:i+test_batch_size])
    batch_y = y_test[i:i+test_batch_size]

    x = tz.Variable(batch_x, requires_grad=False)

    # Forward pass
    output = model(x)

    # Get predictions
    predictions = output.data.numpy().argmax(axis=1)

    # Count correct
    correct += (predictions == batch_y).sum()
    total += len(batch_y)

accuracy = 100.0 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

# ============================================================================
# Save Model
# ============================================================================

# Save model weights
model.save("mnist_model.bin")
print("Model saved to mnist_model.bin")

# Load model weights
model.load("mnist_model.bin")
print("Model loaded successfully")
```

### Using Built-in Layers

Examples of various neural network layers:

```python
import tenzor as tz

tz.initialize()

# ============================================================================
# Linear Layers
# ============================================================================

linear = tz.nn.Linear(128, 64, bias=True)
x = tz.Variable(tz.randn([32, 128]), requires_grad=False)
output = linear(x)  # Shape: (32, 64)

# ============================================================================
# Convolutional Layers
# ============================================================================

# Conv2d: in_channels, out_channels, kernel_size, stride, padding
conv = tz.nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)

# Input: (batch, channels, height, width)
x = tz.Variable(tz.randn([8, 3, 32, 32]), requires_grad=False)
output = conv(x)  # Shape: (8, 64, 32, 32)

# ============================================================================
# Normalization Layers
# ============================================================================

# Batch Normalization (2D)
batch_norm = tz.nn.BatchNorm2d(64)
x = tz.Variable(tz.randn([8, 64, 32, 32]), requires_grad=False)
output = batch_norm(x)  # Shape: (8, 64, 32, 32)

# Layer Normalization
layer_norm = tz.nn.LayerNorm([128])
x = tz.Variable(tz.randn([32, 128]), requires_grad=False)
output = layer_norm(x)  # Shape: (32, 128)

# ============================================================================
# Activation Functions
# ============================================================================

relu = tz.nn.ReLU()
gelu = tz.nn.GELU()
sigmoid = tz.nn.Sigmoid()
tanh = tz.nn.Tanh()
softmax = tz.nn.Softmax(dim=1)

x = tz.Variable(tz.randn([32, 10]), requires_grad=False)
relu_out = relu(x)
gelu_out = gelu(x)
sigmoid_out = sigmoid(x)
tanh_out = tanh(x)
softmax_out = softmax(x)

# ============================================================================
# Pooling Layers
# ============================================================================

# Max pooling
maxpool = tz.nn.MaxPool2d(kernel_size=2, stride=2)
x = tz.Variable(tz.randn([8, 64, 32, 32]), requires_grad=False)
output = maxpool(x)  # Shape: (8, 64, 16, 16)

# Average pooling
avgpool = tz.nn.AvgPool2d(kernel_size=2, stride=2)
output = avgpool(x)  # Shape: (8, 64, 16, 16)

# ============================================================================
# Dropout
# ============================================================================

dropout = tz.nn.Dropout(p=0.5)
x = tz.Variable(tz.randn([32, 128]), requires_grad=False)
output = dropout(x)  # Randomly zeros 50% of elements during training

# ============================================================================
# Loss Functions
# ============================================================================

# Mean Squared Error
mse_loss = tz.nn.MSELoss()
pred = tz.Variable(tz.randn([32, 10]), requires_grad=False)
target = tz.Variable(tz.randn([32, 10]), requires_grad=False)
loss = mse_loss(pred, target)

# Cross Entropy
ce_loss = tz.nn.CrossEntropyLoss()
logits = tz.Variable(tz.randn([32, 10]), requires_grad=False)
labels = tz.Variable(tz.randint(0, 10, [32]), requires_grad=False)
loss = ce_loss(logits, labels)

# Binary Cross Entropy
bce_loss = tz.nn.BCELoss()
pred = tz.Variable(tz.rand([32, 1]), requires_grad=False)
target = tz.Variable(tz.rand([32, 1]), requires_grad=False)
loss = bce_loss(pred, target)
```

---

## Core Concepts

Understanding Tenzor's fundamental concepts will help you use the library effectively.

### Tensors

**What are Tensors?**

Tensors are multi-dimensional arrays that form the foundation of neural network computations. Think of them as generalizations of vectors and matrices:
- 0-D tensor: scalar (single value)
- 1-D tensor: vector (list of values)
- 2-D tensor: matrix (table of values)
- 3-D+ tensor: higher-dimensional arrays

**Tensor Properties:**

```cpp
auto t = tenzor::randn({2, 3, 4});

// Shape: dimensions of the tensor
auto shape = t.shape();  // [2, 3, 4]

// Strides: memory layout (elements to skip per dimension)
auto strides = t.strides();  // [12, 4, 1] for row-major layout

// Number of dimensions
auto ndim = t.ndim();  // 3

// Total number of elements
auto numel = t.numel();  // 24

// Data type
auto dtype = t.dtype();  // Float32, Float64, Int32, etc.

// Device location
auto device = t.device();  // CPU, CUDA, etc.
```

**Memory Layout:**

Tenzor uses row-major (C-style) memory layout by default:
- Elements are stored contiguously in memory
- Last dimension varies fastest
- Enables efficient cache access patterns

```cpp
// Contiguous tensor: elements stored without gaps
auto t = tenzor::ones({2, 3});  // Contiguous

// Non-contiguous: transposed or sliced tensors
auto t_T = t.transpose(0, 1);  // Not contiguous

// Make contiguous if needed
auto t_cont = t_T.contiguous();  // Allocates new contiguous storage
```

**Copy-on-Write Semantics:**

Tenzor uses shared storage for efficiency:
- Copying a tensor shares the underlying storage
- Modifications trigger copy only when necessary
- Views (transpose, slice) share storage with original

```cpp
auto a = tenzor::randn({3, 4});
auto b = a;  // Shares storage with 'a'

// Operations create new tensors
auto c = a + b;  // New storage

// Clone creates independent copy
auto d = a.clone();  // New storage, data copied
```

### Autograd

**Automatic Differentiation:**

Autograd automatically computes gradients by tracking operations in a computational graph.

**Key Components:**

1. **Variable**: Wraps a tensor and tracks computation history
2. **Computational Graph**: Records operations for backpropagation
3. **Gradient Functions**: Define how to compute derivatives
4. **Backward Pass**: Traverses graph to compute gradients

**Basic Usage:**

```cpp
// Create variables that require gradients
auto x = tenzor::Variable(tenzor::randn({2, 3}), true);  // requires_grad=true

// Operations are tracked
auto y = x * 2.0f;
auto z = y + 1.0f;
auto loss = tenzor::sum(z.tensor());

// Compute gradients
loss.backward();  // Backpropagate through graph

// Access gradient
auto dx = x.grad();  // dL/dx
```

**Gradient Accumulation:**

Gradients accumulate by default (useful for batch processing):

```cpp
auto x = tenzor::Variable(tenzor::randn({2, 3}), true);

// First forward/backward
auto loss1 = tenzor::sum(x.tensor() * 2.0f);
loss1.backward();  // dx += gradient from loss1

// Second forward/backward
auto loss2 = tenzor::sum(x.tensor() * 3.0f);
loss2.backward();  // dx += gradient from loss2 (accumulated!)

// Clear gradients before next iteration
x.zero_grad();
```

**Detaching from Graph:**

Stop gradient tracking for specific tensors:

```cpp
auto x = tenzor::Variable(tenzor::randn({2, 3}), true);
auto y = x * 2.0f;

// Detach y from graph
auto y_detached = y.detach();

// Operations on y_detached don't affect gradients
auto z = y_detached + 1.0f;  // No gradient flow through this path
```

### Modules

**Neural Network Building Blocks:**

Modules are reusable components for building neural networks:
- Encapsulate parameters (weights, biases)
- Define forward pass computation
- Support hierarchical composition
- Enable training/evaluation modes

**Module Hierarchy:**

```cpp
class MyNetwork : public tenzor::nn::Module {
public:
    MyNetwork(int input_size, int hidden_size, int output_size) {
        // Create submodules
        fc1_ = std::make_shared<tenzor::nn::Linear>(input_size, hidden_size);
        fc2_ = std::make_shared<tenzor::nn::Linear>(hidden_size, output_size);

        // Register for parameter management
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward(const tenzor::Variable& x) -> tenzor::Variable override {
        auto h = fc1_->forward(x);
        return fc2_->forward(h);
    }

private:
    std::shared_ptr<tenzor::nn::Linear> fc1_;
    std::shared_ptr<tenzor::nn::Linear> fc2_;
};
```

**Parameter Management:**

Modules automatically track all parameters:

```cpp
auto model = std::make_shared<MyNetwork>(784, 128, 10);

// Get all parameters (recursively from submodules)
auto params = model->parameters();

// Get named parameters
auto named_params = model->named_parameters();
for (auto& [name, param] : named_params) {
    std::cout << name << ": " << param->shape() << "\n";
}
```

**Training vs. Evaluation:**

Switch behavior for layers like Dropout and BatchNorm:

```cpp
// Training mode: dropout active, batch norm updates statistics
model->train();

// Evaluation mode: dropout inactive, batch norm uses running stats
model->eval();

// Check current mode
bool is_training = model->is_training();
```

### Optimizers

**Parameter Update Algorithms:**

Optimizers update model parameters using computed gradients:

**SGD (Stochastic Gradient Descent):**
```cpp
auto optimizer = std::make_shared<tenzor::optim::SGD>(
    model->parameters(),
    0.01,      // learning rate
    0.9,       // momentum
    0.0,       // dampening
    0.0001     // weight decay (L2 regularization)
);
```

**Adam (Adaptive Moment Estimation):**
```cpp
auto optimizer = std::make_shared<tenzor::optim::Adam>(
    model->parameters(),
    0.001,     // learning rate
    0.9,       // beta1 (momentum)
    0.999,     // beta2 (RMSprop)
    1e-8       // epsilon (numerical stability)
);
```

**Training Loop Pattern:**

```cpp
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& [input, target] : data_loader) {
        // 1. Forward pass
        auto output = model->forward(input);
        auto loss = criterion->forward(output, target);

        // 2. Zero gradients
        optimizer->zero_grad();

        // 3. Backward pass
        loss.backward();

        // 4. Update parameters
        optimizer->step();
    }
}
```

**Learning Rate Scheduling:**

Adjust learning rate during training:

```cpp
auto scheduler = std::make_shared<tenzor::optim::StepLR>(
    optimizer,
    10,        // step_size: decay every 10 epochs
    0.1        // gamma: multiply lr by 0.1
);

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    // Training loop...

    // Update learning rate
    scheduler->step();
}
```

---

## API Overview

Quick reference for commonly used operations.

### Tensor Creation

```cpp
// Fill with constants
zeros({3, 4})              // All zeros
ones({3, 4})               // All ones
full({3, 4}, 5.0f)         // All 5.0
empty({3, 4})              // Uninitialized

// Random initialization
rand({3, 4})               // Uniform [0, 1)
randn({3, 4})              // Normal N(0, 1)

// Ranges
arange(0, 10, 1)           // [0, 1, 2, ..., 9]
linspace(0, 10, 5)         // [0, 2.5, 5, 7.5, 10]

// Special matrices
eye(4)                     // 4x4 identity

// Like another tensor
zeros_like(other)
ones_like(other)
rand_like(other)
randn_like(other)
```

### Mathematical Operations

```cpp
// Element-wise unary
exp(x)                     // e^x
log(x)                     // Natural logarithm
sqrt(x)                    // Square root
abs(x)                     // Absolute value
neg(x)                     // Negation
pow(x, n)                  // x^n

// Trigonometric
sin(x), cos(x), tan(x)
asin(x), acos(x), atan(x)
sinh(x), cosh(x), tanh(x)

// Element-wise binary
add(a, b)                  // a + b
sub(a, b)                  // a - b
mul(a, b)                  // a * b (element-wise)
div(a, b)                  // a / b

// Matrix operations
matmul(a, b)               // Matrix multiplication
dot(a, b)                  // Dot product (1D)
```

### Reduction Operations

```cpp
// Reduce all elements
sum(x)                     // Sum all
mean(x)                    // Average
std(x)                     // Standard deviation
var(x)                     // Variance
max(x)                     // Maximum
min(x)                     // Minimum
prod(x)                    // Product

// Reduce along dimension
sum(x, dim)                // Sum along dimension
mean(x, dim)
max(x, dim)
min(x, dim)
```

### Shape Manipulation

```cpp
// Reshaping
reshape({new_shape})       // Change shape
view({new_shape})          // View with new shape (zero-copy)
flatten()                  // Flatten to 1D
flatten(start, end)        // Partial flatten

// Dimension operations
transpose(dim0, dim1)      // Swap two dimensions
permute({dims})            // Reorder dimensions
squeeze(dim)               // Remove size-1 dims
unsqueeze(dim)             // Add size-1 dim

// Memory layout
contiguous()               // Ensure contiguous memory
clone()                    // Deep copy
```

### Neural Network Layers

```cpp
// Fully connected
Linear(in, out, bias)

// Convolutional
Conv2d(in_ch, out_ch, kernel, stride, padding)
ConvTranspose2d(...)

// Normalization
BatchNorm2d(num_features)
LayerNorm(normalized_shape)

// Regularization
Dropout(p)

// Pooling
MaxPool2d(kernel, stride)
AvgPool2d(kernel, stride)

// Reshaping
Flatten(start_dim, end_dim)
```

### Activation Functions

```cpp
ReLU()                     // max(0, x)
GELU()                     // Gaussian Error Linear Unit
Sigmoid()                  // 1 / (1 + e^-x)
Tanh()                     // Hyperbolic tangent
Softmax(dim)               // Softmax along dimension
LogSoftmax(dim)            // Log-softmax
LeakyReLU(alpha)           // max(alpha*x, x)
```

### Loss Functions

```cpp
MSELoss()                  // Mean squared error
L1Loss()                   // Mean absolute error
CrossEntropyLoss()         // Cross-entropy (includes softmax)
NLLLoss()                  // Negative log likelihood
BCELoss()                  // Binary cross-entropy
BCEWithLogitsLoss()        // BCE with logits
```

### Optimizers

```cpp
SGD(params, lr, momentum, weight_decay)
Adam(params, lr, beta1, beta2, eps)
AdamW(params, lr, beta1, beta2, eps, weight_decay)
```

### Learning Rate Schedulers

```cpp
StepLR(optimizer, step_size, gamma)
ExponentialLR(optimizer, gamma)
CosineAnnealingLR(optimizer, T_max)
```

---

## Examples

Tenzor includes a comprehensive set of examples demonstrating various features.

### Example Directory Structure

```
examples/
├── cpp/
│   ├── simple_example.cpp          # Basic tensor operations
│   ├── mnist_example.cpp           # MNIST digit classification
│   ├── backend_example.cpp         # Multi-backend usage
│   ├── serialization_example.cpp   # Model saving/loading
│   └── custom_op_example.cpp       # Custom operations
└── python/
    ├── 01_tensor_basics.py         # Tensor creation and operations
    ├── 02_autograd_basics.py       # Automatic differentiation
    ├── 03_linear_regression.py     # Simple regression example
    ├── 04_mnist_mlp.py             # MNIST with MLP
    ├── 05_cnn_classification.py    # CNN for image classification
    └── 06_custom_layer.py          # Custom neural network layer
```

### Running Examples

**C++ Examples:**

```bash
# Build examples (if TENZOR_BUILD_EXAMPLES=ON)
cd build
make

# Run simple example
./bin/simple_example

# Run MNIST example
./bin/mnist_example
```

**Python Examples:**

```bash
# Navigate to examples directory
cd examples/python

# Run tensor basics
python 01_tensor_basics.py

# Run MNIST MLP
python 04_mnist_mlp.py
```

### Recommended Learning Path

1. **Start with Basics:**
   - `01_tensor_basics.py` - Learn tensor creation and manipulation
   - `simple_example.cpp` - C++ tensor operations

2. **Learn Autograd:**
   - `02_autograd_basics.py` - Understand automatic differentiation
   - Experiment with gradient computation

3. **Simple Models:**
   - `03_linear_regression.py` - Build your first model
   - Understand training loops

4. **Deep Learning:**
   - `04_mnist_mlp.py` - Multi-layer perceptron
   - `05_cnn_classification.py` - Convolutional networks

5. **Advanced Topics:**
   - `06_custom_layer.py` - Custom layers and operations
   - `backend_example.cpp` - Multi-backend programming
   - `serialization_example.cpp` - Model persistence

### Example: Complete CNN for Image Classification

See `examples/python/05_cnn_classification.py` for a complete convolutional neural network implementation including:
- Data loading and preprocessing
- CNN architecture with Conv2d, MaxPool, BatchNorm
- Training loop with learning rate scheduling
- Evaluation and accuracy metrics
- Model saving and loading

---

## Troubleshooting

Common issues and solutions when using Tenzor.

### CUDA Not Found

**Error:** CMake cannot find CUDA Toolkit

**Symptoms:**
```
Could NOT find CUDA (missing: CUDA_TOOLKIT)
CUDA backend will not be built
```

**Solutions:**

1. **Install CUDA Toolkit:**
   ```bash
   # Ubuntu
   sudo apt install nvidia-cuda-toolkit

   # Or download from NVIDIA
   wget https://developer.nvidia.com/cuda-downloads
   ```

2. **Set CUDA_HOME environment variable:**
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

3. **Specify CUDA location in CMake:**
   ```bash
   cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.0
   ```

4. **Build without CUDA:**
   ```bash
   cmake .. -DTENZOR_BUILD_CUDA=OFF
   ```

### Python Import Error

**Error:** Cannot import tenzor module

**Symptoms:**
```python
>>> import tenzor
ModuleNotFoundError: No module named 'tenzor'
```

**Solutions:**

1. **Check PYTHONPATH:**
   ```bash
   # Add build directory to PYTHONPATH
   export PYTHONPATH=/path/to/tenzor/build/python:$PYTHONPATH
   python -c "import tenzor; print('Success')"
   ```

2. **Install with pip:**
   ```bash
   cd /path/to/tenzor/build
   pip install -e .
   ```

3. **Verify build succeeded:**
   ```bash
   ls build/python/tenzor/tenzor_core*.so
   # Should show the compiled Python module
   ```

4. **Check Python version:**
   ```bash
   # Ensure Python used for build matches Python used to run
   python --version
   cmake .. -DPython_EXECUTABLE=$(which python)
   ```

### C++ Compilation Errors

**Error:** Compiler doesn't support C++23

**Symptoms:**
```
error: expected ';' at end of declaration
error: 'concept' does not name a type
```

**Solutions:**

1. **Update compiler:**
   ```bash
   # GCC 13+
   sudo apt install g++-13
   export CXX=g++-13

   # Clang 16+
   sudo apt install clang-16
   export CXX=clang++-16
   ```

2. **Specify compiler in CMake:**
   ```bash
   cmake .. -DCMAKE_CXX_COMPILER=g++-13
   ```

3. **Check compiler version:**
   ```bash
   g++ --version  # Should be 13.0 or later
   ```

### Out of Memory Errors

**Error:** CUDA out of memory or system OOM

**Symptoms:**
```
RuntimeError: CUDA out of memory
Killed (OOM killer)
```

**Solutions:**

1. **Reduce batch size:**
   ```cpp
   const int batch_size = 16;  // Smaller batch
   ```

2. **Use CPU instead of GPU:**
   ```cpp
   auto device = tenzor::Device::cpu();
   ```

3. **Clear CUDA cache:**
   ```cpp
   // After training iteration
   tenzor::cuda::synchronize();
   ```

4. **Monitor memory usage:**
   ```bash
   # GPU memory
   nvidia-smi

   # System memory
   htop
   ```

5. **Use gradient checkpointing (advanced):**
   - Trade computation for memory
   - Recompute activations during backward pass

### Linking Errors

**Error:** Cannot find libtenzor_core.so

**Symptoms:**
```
error while loading shared libraries: libtenzor_core.so: cannot open shared object file
```

**Solutions:**

1. **Set LD_LIBRARY_PATH:**
   ```bash
   export LD_LIBRARY_PATH=/path/to/tenzor/build/bin:$LD_LIBRARY_PATH
   ```

2. **Use RPATH when compiling:**
   ```bash
   g++ main.cpp -ltenzor_core -Wl,-rpath,/path/to/tenzor/build/bin
   ```

3. **Install system-wide:**
   ```bash
   sudo make install
   sudo ldconfig
   ```

### Performance Issues

**Issue:** Operations are slower than expected

**Solutions:**

1. **Ensure Release build:**
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

2. **Enable optimizations:**
   ```bash
   cmake .. -DCMAKE_CXX_FLAGS="-O3 -march=native"
   ```

3. **Use GPU for large tensors:**
   ```cpp
   auto tensor = tenzor::randn({1000, 1000}).cuda();
   ```

4. **Check for unnecessary copies:**
   ```cpp
   // Bad: copies data
   auto t = tensor.clone();

   // Good: view (zero-copy)
   auto t = tensor.view({-1});
   ```

5. **Profile your code:**
   ```bash
   # CPU profiling
   perf record ./program
   perf report

   # GPU profiling
   nsys profile ./program
   ```

### Build Failures

**Issue:** Build fails with cryptic errors

**Solutions:**

1. **Clean build:**
   ```bash
   rm -rf build
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

2. **Update submodules:**
   ```bash
   git submodule update --init --recursive
   ```

3. **Check dependencies:**
   ```bash
   # Ensure all required packages installed
   sudo apt install cmake g++ python3-dev
   ```

4. **Verbose build output:**
   ```bash
   make VERBOSE=1
   ```

5. **Report issue:**
   - Include CMake output
   - Include compiler version
   - Include error messages
   - Post to GitHub Issues

---

## Next Steps

Congratulations on completing the Getting Started guide! Here are suggested next steps to deepen your understanding.

### Read the Documentation

**API Documentation:**
- Browse the [API Reference](api/index.html) (Doxygen-generated)
- Detailed documentation for every class and function
- Code examples and usage patterns

**Design Documentation:**
- [DESIGN.md](DESIGN.md) - Architecture overview
- Backend system design
- Memory management strategy
- Autograd implementation details

**Phase Reports:**
- [Phase 5 Completion](PHASE5_COMPLETION_SUMMARY.md) - Latest features
- [Phase 6 Completion](PHASE6_COMPLETION_REPORT.md) - Python bindings
- Implementation status and benchmarks

### Try the Examples

Work through examples in order of complexity:

1. **Tensor operations** (`01_tensor_basics.py`)
2. **Autograd** (`02_autograd_basics.py`)
3. **Linear regression** (`03_linear_regression.py`)
4. **MNIST MLP** (`04_mnist_mlp.py`)
5. **CNN classification** (`05_cnn_classification.py`)
6. **Custom layers** (`06_custom_layer.py`)

### Build Your Own Models

Practice by implementing classic models:

**Image Classification:**
- LeNet-5 for MNIST
- AlexNet for ImageNet
- ResNet for deep networks
- VGG for feature extraction

**Natural Language Processing:**
- RNN for sequence modeling
- LSTM for text generation
- Attention mechanisms
- Transformer architecture (advanced)

**Generative Models:**
- Autoencoder for dimensionality reduction
- Variational Autoencoder (VAE)
- Generative Adversarial Network (GAN)

### Optimize Performance

Learn optimization techniques:

**CPU Optimization:**
- Enable OpenMP multi-threading
- Use SIMD instructions
- Profile hotspots with perf
- Optimize memory access patterns

**GPU Optimization:**
- Batch operations for parallelism
- Minimize host-device transfers
- Use mixed precision training
- Profile with NVIDIA Nsight

**Memory Optimization:**
- Use views instead of copies
- Enable gradient checkpointing
- Clear caches regularly
- Monitor memory usage

### Contribute to Tenzor

Join the development community:

**Report Issues:**
- Bug reports on GitHub Issues
- Feature requests
- Documentation improvements

**Contribute Code:**
- Read [CONTRIBUTING.md](CONTRIBUTING.md)
- Start with "good first issue" labels
- Submit pull requests
- Add tests for new features

**Improve Documentation:**
- Fix typos and errors
- Add examples
- Write tutorials
- Create guides

### Join the Community

Connect with other Tenzor users:

**GitHub Discussions:**
- Ask questions
- Share projects
- Discuss features
- Help others

**Social Media:**
- Follow @TenzorML on Twitter
- Share your work
- Stay updated on releases

### Learn Deep Learning

Recommended resources for learning:

**Books:**
- *Deep Learning* by Goodfellow, Bengio, Courville
- *Neural Networks and Deep Learning* by Michael Nielsen
- *Dive into Deep Learning* (d2l.ai)

**Courses:**
- Stanford CS231n (Convolutional Networks)
- Stanford CS224n (NLP)
- fast.ai Practical Deep Learning
- Coursera Deep Learning Specialization

**Papers:**
- *ImageNet Classification* (AlexNet)
- *Very Deep Networks* (VGG)
- *Deep Residual Learning* (ResNet)
- *Attention Is All You Need* (Transformer)

### Explore Advanced Topics

Push the boundaries:

**Distributed Training:**
- Multi-GPU training
- Data parallelism
- Model parallelism
- Gradient accumulation

**Mixed Precision:**
- FP16/FP32 training
- Automatic mixed precision
- Loss scaling
- Gradient clipping

**Quantization:**
- INT8 inference
- Post-training quantization
- Quantization-aware training
- Model compression

**Deployment:**
- Model export (ONNX)
- Inference optimization
- Edge deployment
- Production serving

---

## Summary

You've learned the fundamentals of Tenzor:

✅ **Installation:** Build from source with CMake
✅ **C++ API:** Tensor operations, autograd, neural networks
✅ **Python API:** PyTorch-like interface with NumPy interop
✅ **Core Concepts:** Tensors, autograd, modules, optimizers
✅ **Examples:** Complete training pipelines
✅ **Troubleshooting:** Common issues and solutions

**Key Takeaways:**

1. **Always initialize:** Call `tenzor::initialize()` or `tz.initialize()`
2. **Use autograd:** Enable `requires_grad` for automatic differentiation
3. **Manage devices:** Move tensors between CPU and GPU efficiently
4. **Build with modules:** Compose neural networks from reusable components
5. **Train systematically:** Forward → backward → optimizer step
6. **Profile and optimize:** Measure performance before optimizing

**Happy Coding with Tenzor!** 🚀

For questions and support:
- 📚 Documentation: [docs/api/index.html](docs/api/index.html)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/tenzor/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/tenzor/issues)
- 📧 Email: support@tenzor.ml
