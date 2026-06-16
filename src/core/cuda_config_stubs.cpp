/**
 * @file cuda_config_stubs.cpp
 * @brief No-op stubs for tenzor::cuda::matmul knobs when CUDA isn't built.
 *
 * The Python binding (python/bindings/bindings_core.cpp) exposes
 * `tenzor.cuda.matmul.{allow_tf32,set_allow_tf32,warn_fp16_saturation,
 * set_warn_fp16_saturation}` unconditionally so the Python API surface
 * is the same on every build. The real definitions live in
 * src/backends/cuda/kernels/matmul.cu, which is only compiled when
 * TENZOR_BUILD_CUDA=ON. Without these stubs, importing the Python
 * module on a CUDA-less build fails with:
 *   ImportError: undefined symbol: _ZN6tenzor4cuda6matmul24set_warn_fp16_saturationEb
 *
 * The stubs preserve the thread-local toggle semantics so callers that
 * read back what they set still observe their value, even though no
 * CUDA kernel will ever consult it.
 */

#include "tenzor/backend/cuda_config.hpp"

#ifndef TENZOR_USE_CUDA

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace tenzor::cuda::matmul {

// NOTE: even on a core library built with CUDA off, the real CUDA backend is
// loaded at runtime as a separate .so that ALSO exports these symbols. Because
// the core library is loaded first, the dynamic linker binds the CUDA backend's
// internal allow_tf32() calls to THESE definitions (symbol interposition). So
// these "stubs" must be behaviorally identical to the real implementation in
// src/backends/cuda/kernels/matmul.cu — in particular they MUST honor the
// documented TENZOR_DISABLE_TF32 env var, otherwise cuBLAS silently keeps TF32
// on and CPU<->CUDA Float32 parity is broken. Tri-state: -1 = resolve from env
// on first read (lazily, so a setenv before the first gemm is honored).
namespace {
std::atomic<int> g_allow_tf32{-1};
std::atomic<bool> g_warn_fp16_saturation{false};

auto tf32_default_from_env() -> bool {
    const char* v = std::getenv("TENZOR_DISABLE_TF32");
    if (v && (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0)) {
        return false;  // TF32 disabled
    }
    return true;  // TF32 allowed (default)
}
}  // namespace

auto allow_tf32() -> bool {
    int v = g_allow_tf32.load(std::memory_order_relaxed);
    if (v < 0) {
        v = tf32_default_from_env() ? 1 : 0;
        int expected = -1;
        g_allow_tf32.compare_exchange_strong(expected, v, std::memory_order_relaxed);
        v = g_allow_tf32.load(std::memory_order_relaxed);
    }
    return v != 0;
}
auto set_allow_tf32(bool value) -> void {
    g_allow_tf32.store(value ? 1 : 0, std::memory_order_relaxed);
}
auto warn_fp16_saturation() -> bool {
    return g_warn_fp16_saturation.load(std::memory_order_relaxed);
}
auto set_warn_fp16_saturation(bool value) -> void {
    g_warn_fp16_saturation.store(value, std::memory_order_relaxed);
}

}  // namespace tenzor::cuda::matmul

#endif  // !TENZOR_USE_CUDA
