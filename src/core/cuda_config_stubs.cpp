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

namespace tenzor::cuda::matmul {

namespace {
thread_local bool g_allow_tf32 = true;
thread_local bool g_warn_fp16_saturation = false;
}

auto allow_tf32() -> bool { return g_allow_tf32; }
auto set_allow_tf32(bool value) -> void { g_allow_tf32 = value; }
auto warn_fp16_saturation() -> bool { return g_warn_fp16_saturation; }
auto set_warn_fp16_saturation(bool value) -> void { g_warn_fp16_saturation = value; }

}  // namespace tenzor::cuda::matmul

#endif  // !TENZOR_USE_CUDA
