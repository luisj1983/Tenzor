# MLIR JIT setup (Phase 13)

Phase 13's `@tz.jit` decorator goes through an MLIR/StableHLO/IREE pipeline.
Building Tenzor with `-DTENZOR_USE_MLIR_JIT=ON` requires IREE 3.x.

The default build (`TENZOR_USE_MLIR_JIT=OFF`) does **not** depend on any
of this — only the JIT pipeline does.

## Architecture

| Component | Source | License |
|---|---|---|
| IREE public C-API headers (`iree/{base,hal,vm,runtime,io}`) | Vendored in-repo at `third_party/iree_compat/` (363 files from authentic IREE 3.11.0 upstream) | Apache 2.0 WITH LLVM-exception |
| IREE Compiler + Runtime static libs + CMake config | External; resolved via the discovery chain below | Apache 2.0 WITH LLVM-exception |
| `iree-compile` binary (for subprocess compile path and debug dumping) | External; resolved via the discovery chain below | Apache 2.0 WITH LLVM-exception |
| Tenzor's text-MLIR emitter + custom_call plugin glue | `src/jit/mlir/` | MIT (Tenzor's license) |

MLIR and StableHLO are bundled **inside** IREE — Tenzor consumes IREE's
C APIs only, so there's no separate `find_package(MLIR)` /
`find_package(StableHLO)` step.

## Discovery chain for IREE binaries

When you run `cmake -DTENZOR_USE_MLIR_JIT=ON ..`, CMake looks for IREE
in this order. The first match wins:

1. **`$IREE_DIR` / `$TENZOR_IREE_DIR` environment variable** — explicit
   override. Set to the dir containing
   `lib/cmake/IREE/IREE{Compiler,Runtime}Config.cmake`.
2. **`third_party/iree_dist/` inside the project tree** — gitignored;
   not committed. See "Installing the IREE distribution" below.
3. **System-installed IREE** via `find_package`'s default search paths.
4. **`FetchContent` source build** — last resort. Builds IREE (with
   its bundled LLVM + StableHLO) from pinned git tag `iree-3.11.0`.
   **First build is ~30–60 minutes and produces ~10 GB of intermediate
   artifacts.** Subsequent builds are incremental.

Skip step 4 by satisfying any of 1–3.

## Installing the IREE distribution

### Option A: project-local `third_party/iree_dist/` (recommended)

Drop a complete IREE C++ distribution into `third_party/iree_dist/`.
The path is gitignored — no risk of committing 446 MB of binaries.
CMake auto-discovers it (step 2 above).

```bash
# After obtaining a complete IREE 3.x dist (built from source or from
# an official binary release once available), extract such that:
ls third_party/iree_dist/
# bin/  include/  lib/  share/
ls third_party/iree_dist/lib/cmake/IREE/
# IREECompilerConfig.cmake  IREERuntimeConfig.cmake  ...

# CMake will print on configure:
#   Using project-local IREE at: <path>/third_party/iree_dist
```

### Option B: pip `iree-base-compiler` wheel (compile path only)

The PyPI wheel ships `iree-compile` and Python runtime modules but **not**
the C++ static libs or headers — sufficient for the subprocess compile
path. Insufficient for the in-process custom_call plugin (Group D of the
Phase 1A plan). Use this when subprocess invocation is acceptable.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install iree-base-compiler==3.11.0 iree-base-runtime==3.11.0
export TENZOR_IREE_COMPILE=$(which iree-compile)
```

CMake `find_program` also searches `.venvs/tenzor-jit/bin/` (a common
project-local venv path) automatically.

### Option C: FetchContent source build (no install)

Don't install anything. CMake's `FetchContent` (step 4 above) will
clone IREE 3.11.0, build it from source against its bundled LLVM and
StableHLO, and consume the result.

- **Cold-build cost**: ~30–60 min, ~10 GB intermediate artifacts.
- **Subsequent builds**: incremental, fast.
- **Network required** on first configure (one-time clone).

Recommended for new contributors who don't already have IREE installed.

## Verifying the install

```bash
cmake -B build -G Ninja -DTENZOR_USE_MLIR_JIT=ON
```

Expected output:
```
Tenzor: MLIR JIT enabled (text-based, IREE-driven per 2026-05-19 amendment)
  IREE 3.0
  Using project-local IREE at: <path>/third_party/iree_dist     # if option A
  iree-compile: <path>/iree-compile
  iree-compile HAL targets:
    - cuda
    - llvm-cpu
    - metal-spirv
    - rocm
    - vmvx
    - vmvx-inline
    - vulkan-spirv
```

If your HAL target list is shorter than expected, your IREE build
disabled those backends. The official `iree-base-compiler` PyPI wheel
ships with all 7. Custom builds often drop CUDA/ROCm to reduce size.

## ROCm runtime libraries (for IREE HIP HAL)

The IREE runtime dlopens `libamdhip64.so` and several siblings when an
`hip` device is requested. CMake auto-detects `/opt/rocm/lib`; override
with:

```bash
export TENZOR_ROCM_RUNTIME_LIB_DIR=/opt/rocm/lib
```

CMake's discovery rejects zero-byte ROCm libs (symptom of a broken
ROCm package install) and falls back to skipping the HIP HAL.

## Licensing notes

- **Vendored headers** at `third_party/iree_compat/` are authentic IREE
  3.11.0 source: Apache 2.0 WITH LLVM-exception. Each file preserves
  its original SPDX header. See `third_party/iree_compat/LICENSE` and
  `third_party/iree_compat/NOTICE`.
- **IREE binaries** (when installed at `third_party/iree_dist/`) are
  not part of this repository (the path is gitignored). If you
  redistribute a built Tenzor binary that links statically against
  IREE, include IREE's LICENSE and NOTICE files alongside.
- **Tenzor itself** is MIT (see `LICENSE` at the repo root).

## Environment variable overrides

| Variable | Purpose |
|---|---|
| `IREE_DIR` / `TENZOR_IREE_DIR` | Override discovery for the IREE CMake config dir |
| `TENZOR_IREE_COMPILE` | Full path to a specific `iree-compile` executable |
| `TENZOR_IREE_RUN_MODULE` | Full path to a specific `iree-run-module` executable |
| `TENZOR_ROCM_RUNTIME_LIB_DIR` | Override `/opt/rocm/lib` autodetect |
