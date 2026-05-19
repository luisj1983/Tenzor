# MLIR JIT setup (Phase 13)

Phase 13's `TENZOR_USE_MLIR_JIT=ON` requires three system dependencies:
MLIR (build-time C++ API), StableHLO (dialect library), and the IREE runtime
(for in-process execution) plus the `iree-compile` binary (invoked as a
subprocess at runtime).

## Tested platform: Arch Linux x86_64

### IREE 3.x (runtime + iree-compile binary)

The IREE distribution tarball from GitHub releases is the recommended path on
Arch. It bundles `iree-compile`, the IREE C runtime headers, and static libs:

```bash
mkdir -p ~/iree-dist
curl -L https://github.com/iree-org/iree/releases/download/iree-3.12.0rc20260518/iree-dist-3.12.0rc20260518-linux-x86_64.tar.xz \
     -o ~/iree-dist/iree-dist.tar.xz
tar -xf ~/iree-dist/iree-dist.tar.xz -C ~/iree-dist
# iree-compile is at ~/iree-dist/bin/iree-compile
# cmake config is at ~/iree-dist/lib/cmake/IREE/IREERuntimeConfig.cmake
```

Set environment variables:
```bash
export IREE_RUNTIME_DIR=~/iree-dist/lib/cmake/IREE
export TENZOR_IREE_COMPILE=~/iree-dist/bin/iree-compile
```

### MLIR 18 (C++ headers + mlir-tblgen)

The LLVM 18 pre-built release from llvm.org requires `libtinfo.so.5`
(Ubuntu ABI) which conflicts with Arch's `libtinfo.so.6`. Build mlir-tblgen
and MLIR libraries from the LLVM 22.x source instead:

```bash
# Sparse-checkout LLVM 22 source (mlir + llvm subdirs only)
mkdir -p ~/llvm22-src && cd ~/llvm22-src
git init && git remote add origin https://github.com/llvm/llvm-project.git
git sparse-checkout init
git sparse-checkout set llvm mlir cmake third-party
git fetch --depth=1 origin llvmorg-22.1.5
git checkout FETCH_HEAD -- llvm mlir cmake third-party

# Build mlir-tblgen + required MLIR libraries
cmake -B build -S llvm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=~/mlir-install

cd build && ninja mlir-tblgen MLIRIR MLIRSupport MLIRPass MLIRTransforms \
    MLIRFuncDialect MLIRBuiltinToLLVMIRTranslation
ninja install
```

Set environment variables:
```bash
export MLIR_DIR=~/mlir-install/lib/cmake/mlir
```

### StableHLO 1.5.0

Build from source against the MLIR installation above. Patch the CMakeLists to
remove test and integration subdirs that require `FileCheck` as a CMake target:

```bash
git clone --branch v1.5.0 --depth 1 https://github.com/openxla/stablehlo ~/stablehlo-src

# Minimal build (dialect only)
cmake -B ~/stablehlo-src/build -S ~/stablehlo-src -G Ninja \
  -DMLIR_DIR=~/mlir-install/lib/cmake/mlir \
  -DLLVM_DIR=~/mlir-install/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/stablehlo-install \
  -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
  -DSTABLEHLO_BUILD_EMBEDDED=OFF

cmake --build ~/stablehlo-src/build --target StablehloOps
cmake --install ~/stablehlo-src/build
```

Set environment variables:
```bash
export STABLEHLO_DIR=~/stablehlo-install/lib/cmake/stablehlo
```

### Ubuntu 24.04 (alternative)

```bash
# MLIR + LLVM 18 (includes mlir-tblgen and CMake configs)
sudo apt install llvm-18-dev libmlir-18-dev mlir-18-tools

# StableHLO 1.5.0 (same as above, but using apt-installed MLIR)
git clone https://github.com/openxla/stablehlo --branch v1.5.0 ~/stablehlo-src
cmake -B ~/stablehlo-src/build -S ~/stablehlo-src \
  -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir
cmake --build ~/stablehlo-src/build --target install

# IREE 3.0 (runtime + iree-compile binary)
pip install iree-base-compiler iree-base-runtime
# OR use the iree-dist tarball (see Arch section above)
```

## Configuring Tenzor with MLIR JIT enabled

```bash
cmake -B build -G Ninja \
  -DTENZOR_USE_MLIR_JIT=ON \
  -DMLIR_DIR=$MLIR_DIR \
  -DSTABLEHLO_DIR=$STABLEHLO_DIR \
  -DIREE_RUNTIME_DIR=$IREE_RUNTIME_DIR \
  -DTENZOR_IREE_COMPILE=$TENZOR_IREE_COMPILE
```

## Environment variable overrides

| Variable | Purpose |
|---|---|
| `MLIR_DIR` | Path to MLIR's `cmake/mlir/` directory |
| `STABLEHLO_DIR` | Path to StableHLO's CMake config |
| `IREE_RUNTIME_DIR` | Path to IREE runtime's CMake config |
| `TENZOR_IREE_COMPILE` | Full path to `iree-compile` executable |
