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
git clone https://github.com/skreamz/Tenzor.git
cd Tenzor
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
sudo cmake --install .
```

### Python Package

```bash
# From the project root directory
cd Tenzor
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
| GCC | 13.0 | Recommended for Linux |
| Clang | 15.0 | Recommended for macOS |
| MSVC | 2022 (17.0) | Windows only |
| Intel ICX | 2023.0 | For OneAPI backend |

### Build Tools

- **CMake**: 3.25 or higher
- **Ninja** (optional): 1.10+ for faster builds
- **Python**: 3.9 - 3.13 (for Python bindings; matches `requires-python = ">=3.9"` in `pyproject.toml`)

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
git clone https://github.com/skreamz/Tenzor.git
cd Tenzor
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
| `TENZOR_BUILD_CUDA` | Enable NVIDIA CUDA backend | ON |
| `TENZOR_BUILD_ROCM` | Enable AMD ROCm backend | ON |
| `TENZOR_BUILD_ONEAPI` | Enable Intel OneAPI backend | ON |
| `TENZOR_BUILD_VULKAN` | Enable Vulkan compute backend | ON |
| `TENZOR_BUILD_MPS` | Enable Apple Metal/MPS backend | ON (macOS only) |
| `TENZOR_BUILD_PYTHON` | Build Python bindings | ON |
| `TENZOR_BUILD_TESTS` | Build test suite | ON |
| `TENZOR_BUILD_BENCHMARKS` | Build performance benchmarks | ON |
| `TENZOR_BUILD_EXAMPLES` | Build example programs | ON |
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

### Windows with MSVC

Use the **Ninja** generator, not the "Visual Studio 17 2022" generator. CMake
hard-blocks mixing MSVC with Clang-family languages on Windows
(`Windows-Clang.cmake`), and the ROCm and OneAPI backends are Clang-family
toolchains (`hipcc`/`icpx`) — the VS generator can't build them at all. Every
backend builds fine under Ninja with `cl.exe` as the host compiler; ROCm and
OneAPI sources are compiled via direct custom `add_custom_command` calls to
`hipcc.exe`/`icpx.exe` rather than CMake's native HIP/SYCL language support.

**Prerequisites:**

1. [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) (or full VS) with the "Desktop development with C++" workload — provides `cl.exe`, `link.exe`, `dumpbin.exe`.
2. [CMake](https://cmake.org/download/) 3.25+ and [Ninja](https://ninja-build.org/).
3. [vcpkg](https://github.com/microsoft/vcpkg) (for zlib/curl/OpenSSL/spdlog etc.) — clone it and note the path to its `scripts/buildsystems/vcpkg.cmake` toolchain file.
4. Optional, per backend you want: [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) (OneAPI backend, and MKL/oneDNN which the CPU backend also uses), [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (CUDA backend), [AMD ROCm/HIP SDK for Windows](https://rocm.docs.amd.com/) (ROCm backend), [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) (Vulkan backend).
5. [Python](https://www.python.org/downloads/) 3.9–3.13 + `pip install pybind11` if building Python bindings.

**Configure and build**, from a `vcvars64.bat`-initialized shell (Developer PowerShell/Command Prompt for VS 2022) with `cmake`/`ninja` and any vendor compiler `bin/` directories (e.g. oneAPI's `compiler\<ver>\bin`, ROCm's `bin`) on `PATH`:

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CXX_COMPILER=cl `
    -DTENZOR_BUILD_CUDA=ON -DTENZOR_BUILD_ROCM=ON -DTENZOR_BUILD_ONEAPI=ON -DTENZOR_BUILD_VULKAN=ON `
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" `
    -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake `
    -DVCPKG_TARGET_TRIPLET=x64-windows `
    -DCMAKE_PREFIX_PATH="C:/Program Files (x86)/Intel/oneAPI/tbb/<version>" `
    -DPython_EXECUTABLE=C:/path/to/python.exe `
    -Dpybind11_DIR=C:/path/to/site-packages/pybind11/share/cmake/pybind11

cmake --build build --config Release
```

Turn off `-DTENZOR_BUILD_*` for any backend whose SDK isn't installed — CPU-only (`-DTENZOR_BUILD_CUDA=OFF -DTENZOR_BUILD_ROCM=OFF -DTENZOR_BUILD_ONEAPI=OFF -DTENZOR_BUILD_VULKAN=OFF`) needs only MSVC + vcpkg + oneAPI's MKL/oneDNN (the CPU backend itself depends on those).

**Running the result:** `tenzor_core.dll`, on Windows, statically links a handful of vendor runtime libraries directly (MKL threading, and — when built — CUDA's NVRTC and ROCm's HIP runtime compiler). Those resolve via the OS's normal DLL search order, which includes `PATH` — so add each enabled vendor SDK's `bin` directory to `PATH` before running a C++ executable that links `tenzor_core`. Backend `.dll`s themselves (`tenzor_backend_*.dll`) are loaded dynamically at runtime by `BackendLoader`, which registers the same vendor directories itself (`src/backend/loader.cpp`, from a list CMake generates at configure time into `build/generated/tenzor_win_dll_dirs.generated.hpp`) — no extra setup needed for those. The Python package does the equivalent automatically (`python/tenzor/__init__.py`, from `build/python/tenzor/_win_dll_dirs.py`), so `import tenzor` works with no manual `PATH`/`os.add_dll_directory` setup at all.

```powershell
cmake --install build --config Release
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
auto* cuda_backend = tenzor::backend_registry().get_backend("cuda");
if (cuda_backend != nullptr && cuda_backend->is_available()) {
    std::cout << "CUDA available with " << cuda_backend->device_count() << " devices\n";
}
```

```python
import tenzor as tz

# Check backends
print(f"CUDA available: {tz.cuda_is_available()}")
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
cd Tenzor
pip install .
```

### Development Install

For development with live code updates:

```bash
cd Tenzor
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
if tz.cuda_is_available():
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
g++ --version  # Should be 13.0+

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

- **GitHub Issues**: [Report bugs and request features](https://github.com/skreamz/Tenzor/issues)
- **Discussions**: [Ask questions](https://github.com/skreamz/Tenzor/discussions)
- **Documentation**: Full API reference — generate locally with `doxygen Doxyfile` (requires the `doxygen`/`graphviz` packages above), then open `docs/api/html/index.html`. Not pre-built or committed in this repo.

### Debug Build

For debugging issues, build with debug symbols:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DTENZOR_ENABLE_ASAN=ON
cmake --build .
```

This enables AddressSanitizer for detecting memory issues.
