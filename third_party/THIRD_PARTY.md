# Vendored third-party dependencies

This file pins each vendored dependency under `third_party/` and documents
its upgrade gate. Anyone bumping a vendored version must update both the
in-tree files and the corresponding row here, and run the listed
re-validation step.

Background:
- audit-2 O.3 introduced the in-file pin comment in `nlohmann/json.hpp`.
- audit-3 T.17 extends the pin convention to every vendored directory so
  the upgrade gate is centralised and discoverable.

| Path | Version | Source | Upgrade gate |
|------|---------|--------|--------------|
| `third_party/nlohmann/json.hpp` | **v3.11.3** | github.com/nlohmann/json (MIT) | See in-file comment at the top of `json.hpp` (audit-2 O.3). The macro `NLOHMANN_JSON_VERSION_MAJOR/MINOR/PATCH` must stay in sync with the literal version banner; do not hand-edit either independently. |
| `third_party/iree_dist/` | Vendored 2026-05-19; no upstream version metadata in the dist tarball (`iree-dist.tar.xz`). The dist contains the IREE compiler + runtime binaries and CMake config files under `lib/cmake/IREE/`. | github.com/iree-org/iree (Apache-2.0 WITH LLVM-exception) | **Do not bump without re-validating MLIR/LLVM compatibility against `src/jit/codegen.cpp` and the IREE compiler invocation in `src/jit/iree_backend.cpp`.** IREE pins its own LLVM revision; mismatch with the MLIR dialects emitted by Tenzor's JIT codegen surfaces as `iree-compile` failing on `linalg.*` / `tensor.*` ops that changed signatures. After any bump, run `ctest -R "jit\|iree" -j1` and re-build at least one model end-to-end through the JIT path before landing. |
| `third_party/iree_compat/` | Vendored 2026-05-19; same upstream as `iree_dist`. Contains the IREE runtime headers (`iree/base/`, `iree/hal/`, `iree/vm/`, etc.) needed for in-process Tenzor → IREE runtime calls. | github.com/iree-org/iree (Apache-2.0 WITH LLVM-exception) | Must be upgraded in lockstep with `iree_dist` — the runtime headers here MUST match the runtime ABI shipped in `iree_dist/lib/libiree_runtime*`. After any bump, run the `jit/` test subtree (`ctest -L jit -j1`) and at least one end-to-end inference through the IREE runtime path. |

## Dependencies sourced via CMake (not vendored in-tree)

These are listed here so the audit surface is complete; they are NOT
under `third_party/` but are tracked by `find_package`/`FetchContent` in
the top-level CMakeLists.txt.

| Dep | Version source | Notes |
|-----|----------------|-------|
| `spdlog` | system `find_package(spdlog)` else FetchContent `v1.14.1` (top-level `CMakeLists.txt::417`) | audit-2 D.1. The FetchContent tag is the upgrade gate. |
| `fmt` | transitively from spdlog (system or FetchContent) | No direct pin; tracks spdlog. |
| `pybind11` | `find_package(pybind11 CONFIG)` (top-level `CMakeLists.txt::451`) | System-provided; if a specific version is required for Python ABI parity, add it to `CMakePresets`. |
| `MKL` | `find_package(MKL CONFIG QUIET)` via `cmake/FindMKL.cmake` | System-provided (e.g. `intel-oneapi-mkl`). Optional unless `TENZOR_REQUIRE_MKL=ON`, in which case its absence is a hard configure error. |
| `oneDNN` | `cmake/FindOneDNN.cmake` | System-provided. Used for CPU conv/batchnorm/etc. kernels when available. |
| `GTest` (googletest) | FetchContent, only when `TENZOR_BUILD_TESTS=ON` (top-level `CMakeLists.txt::568`) | Auto-downloaded if no system package is found. |
| `Threads` / `OpenMP` | `find_package(Threads REQUIRED)` / `find_package(OpenMP)` (`CMakeLists.txt::558-559`) | System toolchain-provided; OpenMP powers CPU kernel parallelism. |
| `Torch` (LibTorch) | `find_package(Torch REQUIRED)` (`CMakeLists.txt::643`), only for the PyTorch-comparison Python bindings/benchmarks | Optional — only needed when building the PyTorch-comparison path. |
| `IREECompiler` / `IREERuntime` | `find_package(IREECompiler/IREERuntime CONFIG)` (`CMakeLists.txt::218-244`) | Resolved from the vendored `third_party/iree_dist/` CMake config, not a separate system install — see the vendored-dependency table above for the upgrade gate. |
| `NCCL` | `cmake/FindNCCL.cmake` | Optional, for distributed multi-GPU CUDA training. |

## How to bump a vendored dep

1. Update the in-tree files under `third_party/<dep>/`.
2. Update the row in this file (version, vendor date if no upstream tag).
3. Update any in-file pin comment (e.g. `json.hpp` banner) so the macro
   and the comment agree.
4. Run the upgrade gate listed in the table (re-validation tests).
5. Commit with a `BREAKING:` prefix if the bump changes any user-visible
   API surface.
