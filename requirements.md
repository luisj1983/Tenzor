# Tenzor Build Requirements

## Required (core build)

| Dependency | Min Version | Manjaro Package | Notes |
|---|---|---|---|
| CMake | 3.25 | `cmake` | Build system |
| Ninja | - | `ninja` | Build generator |
| GCC | 13+ | `gcc` | C++23 support required |
| pthreads | - | (included with glibc) | Threading |
| libcurl | - | `curl` | ModelHub downloads |
| OpenSSL | - | `openssl` | ModelHub HTTPS |

```bash
sudo pacman -S cmake ninja gcc curl openssl
```

## Strongly Recommended

| Dependency | Manjaro Package | Notes |
|---|---|---|
| OpenMP | (included with `gcc`) | Parallel CPU kernels |
| Google Test | `gtest` | Auto-downloaded via FetchContent if not found |

```bash
sudo pacman -S gtest
```

## Optional - CPU Performance

| Dependency | Manjaro Package | Notes |
|---|---|---|
| Intel MKL | `intel-oneapi-mkl` (AUR) | Optimized BLAS/GEMM (5-10x faster matmul) |
| OpenBLAS | `openblas` | Fallback BLAS if MKL not available |
| Intel oneDNN | `onednn` (AUR) | Optimized conv, batchnorm, etc. |

```bash
# Either MKL (preferred):
yay -S intel-oneapi-mkl
# Or OpenBLAS (fallback):
sudo pacman -S openblas

# Optional oneDNN:
yay -S onednn
```

## Optional - CUDA Backend

Disabled with `-DTENZOR_BUILD_CUDA=OFF`

| Dependency | Min Version | Notes |
|---|---|---|
| CUDA Toolkit | 12.0 | Includes cuBLAS, cuRAND |
| cuDNN | - | Conv2d, pooling, softmax acceleration |
| cuDNN Frontend | 9.0+ (cuDNN) | SDPA support (auto-fetched if cuDNN >= 9) |
| NCCL | - | Distributed GPU training |

```bash
sudo pacman -S cuda cudnn
# NCCL from AUR if needed:
yay -S nccl
```

## Optional - Vulkan Backend

Disabled with `-DTENZOR_BUILD_VULKAN=OFF`

| Dependency | Manjaro Package | Notes |
|---|---|---|
| Vulkan SDK | `vulkan-devel` | Headers + loader |
| glslc | `shaderc` | GLSL to SPIR-V shader compiler |
| Python 3 | `python` | Shader embedding script |

```bash
sudo pacman -S vulkan-devel shaderc python
```

## Optional - OneAPI Backend

Disabled with `-DTENZOR_BUILD_ONEAPI=OFF`

| Dependency | Source | Notes |
|---|---|---|
| Intel oneAPI Base Toolkit | [intel.com](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) | Provides `icpx` compiler + SYCL runtime |
| oneMKL SYCL interface | (included in oneAPI) | Optional, for accelerated math |

## Optional - ROCm Backend

Disabled by default (`-DTENZOR_BUILD_ROCM=ON` to enable)

| Dependency | Manjaro Package | Notes |
|---|---|---|
| HIP Runtime | `hip-runtime-amd` | AMD HIP compiler + runtime |
| rocBLAS | `rocblas` | BLAS for AMD GPUs (like cuBLAS) |
| MIOpen | `miopen-hip` | DNN library (like cuDNN) |
| hipRAND | `hiprand` | Random number generation |

```bash
sudo pacman -S hip-runtime-amd rocblas miopen-hip hiprand

# Configure (must specify HIP compiler path)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DTENZOR_BUILD_ROCM=ON \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
```

## Optional - Python Bindings

Disabled with `-DTENZOR_BUILD_PYTHON=OFF`

| Dependency | Manjaro Package | Notes |
|---|---|---|
| Python dev | `python` | Headers + interpreter |
| pybind11 | `pybind11` | C++/Python bridge |

```bash
sudo pacman -S python pybind11
```

## Optional - Documentation

| Dependency | Manjaro Package | Notes |
|---|---|---|
| Doxygen | `doxygen` | API docs generation |

## Quick Start (minimal CPU-only build)

```bash
# Install minimum requirements
sudo pacman -S cmake ninja gcc curl openssl openblas gtest

# Configure with only CPU backend
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DTENZOR_BUILD_CUDA=OFF \
    -DTENZOR_BUILD_ONEAPI=OFF \
    -DTENZOR_BUILD_VULKAN=OFF \
    -DTENZOR_BUILD_ROCM=OFF

# Build
ninja -C build
```

## Full Build (all available backends)

```bash
# Install everything available from pacman
sudo pacman -S cmake ninja gcc curl openssl openblas gtest \
    vulkan-devel shaderc python pybind11 cuda cudnn nccl \
    hip-runtime-amd rocblas miopen-hip hiprand doxygen

# Configure with all backends
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DTENZOR_BUILD_ROCM=ON \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++

# Build
ninja -C build
```
