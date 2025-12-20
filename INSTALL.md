# Installation Guide

This guide covers all methods to install Tenzor on your system.

## Table of Contents

- [Quick Install](#quick-install)
- [System Requirements](#system-requirements)
- [Building from Source](#building-from-source)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Backend Configuration](#backend-configuration)
- [Python Bindings](#python-bindings)
- [Verifying Installation](#verifying-installation)
- [Troubleshooting](#troubleshooting)

## Quick Install

### From Source (Recommended)

```bash
git clone https://github.com/leelee222/tenzor.git
cd tenzor
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
sudo cmake --install .
```

### Python Package

```bash
# From the project root directory
cd tenzor
pip install .

# Or for development (editable install)
pip install -e .
```

## System Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | x86_64 with SSE4.2 | x86_64 with AVX2/AVX-512 |
| RAM | 4 GB | 16 GB+ |
| Disk Space | 500 MB | 2 GB (with all backends) |
| OS | Linux (kernel 4.15+), macOS 12+, Windows 10+ | Ubuntu 22.04+, macOS 13+ |

### Compiler Requirements

| Compiler | Minimum Version | Notes |
|----------|-----------------|-------|
| GCC | 12.0 | Recommended for Linux |
| Clang | 15.0 | Recommended for macOS |
| MSVC | 2022 (17.0) | Windows only |
| Intel ICX | 2023.0 | For OneAPI backend |

### Build Tools

- **CMake**: 3.25 or higher
- **Ninja** (optional): 1.10+ for faster builds
- **Python**: 3.8 - 3.13 (for Python bindings)

## Building from Source

### Step 1: Install Dependencies

#### Ubuntu/Debian

```bash
# Essential build tools
sudo apt update
sudo apt install -y build-essential cmake ninja-build git

# Optional: Python development
sudo apt install -y python3-dev python3-pip python3-venv

# Optional: Documentation
sudo apt install -y doxygen graphviz
```

#### Fedora/RHEL

```bash
sudo dnf install -y gcc-c++ cmake ninja-build git
sudo dnf install -y python3-devel python3-pip
```

#### Arch Linux

```bash
sudo pacman -S base-devel cmake ninja git python python-pip
```

#### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install dependencies via Homebrew
brew install cmake ninja python
```

#### Windows

1. Install [Visual Studio 2022](https://visualstudio.microsoft.com/) with C++ workload
2. Install [CMake](https://cmake.org/download/)
3. Install [Git](https://git-scm.com/download/win)
4. Install [Python](https://www.python.org/downloads/) (optional)

### Step 2: Clone Repository

```bash
git clone https://github.com/leelee222/tenzor.git
cd tenzor
```

### Step 3: Configure Build

#### Basic Configuration

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

#### Full-Featured Configuration

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENZOR_BUILD_CUDA=ON \
    -DTENZOR_BUILD_PYTHON=ON \
    -DTENZOR_BUILD_TESTS=ON \
    -DTENZOR_BUILD_BENCHMARKS=ON \
    -DTENZOR_BUILD_EXAMPLES=ON \
    -GNinja
```

#### Using Ninja (Faster Builds)

```bash
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
ninja -j$(nproc)
```

### Step 4: Build

```bash
# With Make
cmake --build . -j$(nproc)

# With Ninja
ninja

# Windows (Visual Studio)
cmake --build . --config Release --parallel
```

### Step 5: Install

```bash
# System-wide installation (requires sudo)
sudo cmake --install .

# Custom installation prefix
cmake --install . --prefix /opt/tenzor
```

### Build Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `CMAKE_BUILD_TYPE` | Build type (Debug/Release/RelWithDebInfo) | Release |
| `CMAKE_INSTALL_PREFIX` | Installation directory | /usr/local |
| `TENZOR_BUILD_CUDA` | Enable NVIDIA CUDA backend | ON (if found) |
| `TENZOR_BUILD_ROCM` | Enable AMD ROCm backend | OFF |
| `TENZOR_BUILD_ONEAPI` | Enable Intel OneAPI backend | OFF |
| `TENZOR_BUILD_VULKAN` | Enable Vulkan compute backend | OFF |
| `TENZOR_BUILD_METAL` | Enable Apple Metal backend | OFF (macOS auto) |
| `TENZOR_BUILD_WEBGPU` | Enable WebGPU backend | OFF |
| `TENZOR_BUILD_PYTHON` | Build Python bindings | ON |
| `TENZOR_BUILD_TESTS` | Build test suite | ON |
| `TENZOR_BUILD_BENCHMARKS` | Build performance benchmarks | OFF |
| `TENZOR_BUILD_EXAMPLES` | Build example programs | ON |
| `TENZOR_BUILD_DOCS` | Build documentation | OFF |
| `TENZOR_ENABLE_OPENMP` | Enable OpenMP parallelization | ON |
| `TENZOR_ENABLE_SIMD` | Enable SIMD optimizations | ON |

## Platform-Specific Instructions

### Linux with NVIDIA GPU (CUDA)

```bash
# Install CUDA Toolkit (12.0+)
# Download from: https://developer.nvidia.com/cuda-downloads

# Verify installation
nvcc --version
nvidia-smi

# Build with CUDA
cmake .. -DTENZOR_BUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"
```

### Linux with AMD GPU (ROCm)

```bash
# Install ROCm (5.0+)
# Follow: https://rocm.docs.amd.com/

# Verify installation
rocm-smi

# Build with ROCm
cmake .. -DTENZOR_BUILD_ROCM=ON
```

### Linux with Intel GPU (OneAPI)

```bash
# Install Intel OneAPI Base Toolkit
# Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

# Source environment
source /opt/intel/oneapi/setvars.sh

# Build with OneAPI
cmake .. -DTENZOR_BUILD_ONEAPI=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
```

### macOS with Apple Silicon

```bash
# Metal backend is enabled by default on macOS
cmake .. -DTENZOR_BUILD_METAL=ON -DCMAKE_BUILD_TYPE=Release

# For x86_64 compatibility
cmake .. -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"
```

### Windows with Visual Studio

```powershell
# Open Developer PowerShell for VS 2022
mkdir build
cd build

# Configure
cmake .. -G "Visual Studio 17 2022" -A x64 -DTENZOR_BUILD_CUDA=ON

# Build
cmake --build . --config Release --parallel

# Install
cmake --install . --config Release
```

### Cross-Platform Vulkan Backend

```bash
# Install Vulkan SDK
# Download from: https://vulkan.lunarg.com/sdk/home

# Linux
sudo apt install vulkan-sdk

# macOS (via MoltenVK)
brew install molten-vk

# Build with Vulkan
cmake .. -DTENZOR_BUILD_VULKAN=ON
```

## Backend Configuration

### Runtime Backend Selection

Tenzor automatically detects available backends at runtime. You can also explicitly select:

```cpp
#include <tenzor/tenzor.hpp>

// Use specific backend
auto tensor = tenzor::zeros({3, 3}, tenzor::DType::Float32, tenzor::Device::cuda(0));

// Check available backends
if (tenzor::cuda_available()) {
    std::cout << "CUDA available with " << tenzor::cuda_device_count() << " devices\n";
}
```

```python
import tenzor as tz

# Check backends
print(f"CUDA available: {tz.cuda_available()}")
print(f"Device count: {tz.cuda_device_count()}")

# Use specific device
x = tz.zeros([3, 3], device=tz.Device.cuda(0))
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `TENZOR_DEVICE` | Default device | `cuda:0`, `cpu` |
| `TENZOR_NUM_THREADS` | CPU thread count | `8` |
| `TENZOR_MEMORY_POOL_SIZE` | Memory pool size (MB) | `1024` |
| `TENZOR_LOG_LEVEL` | Logging verbosity | `debug`, `info`, `warn` |
| `CUDA_VISIBLE_DEVICES` | Visible CUDA devices | `0,1` |

## Python Bindings

### Install from Source (Recommended)

This will automatically build the C++ library and install Python bindings:

```bash
# From the project root directory
cd tenzor
pip install .
```

### Development Install

For development with live code updates:

```bash
cd tenzor
pip install -e .
```

### Alternative: Using PYTHONPATH

If you've already built the library with CMake and want to use the Python bindings without pip:

```bash
# Build first
mkdir build && cd build
cmake .. -DTENZOR_BUILD_PYTHON=ON
cmake --build . -j$(nproc)

# Add to PYTHONPATH
export PYTHONPATH=/path/to/tenzor/python:$PYTHONPATH
```

### Verify Python Installation

```python
import tenzor as tz

# Check version
print(f"Tenzor version: {tz.__version__}")

# Create tensor
x = tz.randn([3, 3])
print(x)

# GPU test (if available)
if tz.cuda_available():
    y = x.cuda()
    print(f"GPU tensor: {y.device}")
```

### NumPy Interoperability

```python
import numpy as np
import tenzor as tz

# NumPy to Tenzor
np_array = np.random.randn(3, 3).astype(np.float32)
tz_tensor = tz.from_numpy(np_array)

# Tenzor to NumPy
back_to_numpy = tz_tensor.numpy()
```

## Verifying Installation

### C++ Verification

```bash
# Run tests
cd build
ctest --output-on-failure

# Or run specific test
./bin/test_tensor
./bin/test_autograd
```

### Quick Test Program

```cpp
// test_install.cpp
#include <tenzor/tenzor.hpp>
#include <iostream>

int main() {
    tenzor::initialize();

    auto x = tenzor::randn({3, 3});
    auto y = tenzor::randn({3, 3});
    auto z = tenzor::matmul(x, y);

    std::cout << "Installation successful!\n";
    std::cout << "Result shape: [" << z.shape()[0] << ", " << z.shape()[1] << "]\n";

    return 0;
}
```

Compile and run:

```bash
g++ -std=c++23 test_install.cpp -ltenzor_core -o test_install
./test_install
```

### Python Verification

```bash
python -c "import tenzor; print('Tenzor', tenzor.__version__, 'installed successfully!')"
```

## Troubleshooting

### Common Issues

#### CMake Cannot Find CUDA

```bash
# Specify CUDA path explicitly
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
         -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

#### Linker Errors with C++23 Features

Ensure your compiler supports C++23:

```bash
# Check GCC version
g++ --version  # Should be 12.0+

# Use specific compiler
cmake .. -DCMAKE_CXX_COMPILER=g++-13
```

#### Python Module Not Found

```bash
# Add to PYTHONPATH
export PYTHONPATH=/path/to/tenzor/build/python:$PYTHONPATH

# Or install properly
cd build && pip install python/
```

#### CUDA Out of Memory

```bash
# Reduce memory pool size
export TENZOR_MEMORY_POOL_SIZE=512

# Or in code
tenzor::set_memory_pool_size(512 * 1024 * 1024);  // 512 MB
```

#### OpenMP Not Found

```bash
# Ubuntu/Debian
sudo apt install libomp-dev

# macOS
brew install libomp
cmake .. -DOpenMP_ROOT=$(brew --prefix)/opt/libomp
```

### Getting Help

- **GitHub Issues**: [Report bugs and request features](https://github.com/leelee222/tenzor/issues)
- **Discussions**: [Ask questions](https://github.com/leelee222/tenzor/discussions)
- **Documentation**: [Full API reference](docs/api/html/index.html)

### Debug Build

For debugging issues, build with debug symbols:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DTENZOR_ENABLE_ASAN=ON
cmake --build .
```

This enables AddressSanitizer for detecting memory issues.
