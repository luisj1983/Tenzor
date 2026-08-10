/**
 * @file cuda_config.cpp
 * @brief Definitions for the process-global CUDA backend runtime toggles
 *        declared in include/tenzor/backend/cuda_config.hpp.
 *
 * The matmul-specific toggles (allow_tf32, warn_fp16_saturation) are still
 * defined in kernels/matmul.cu next to the gemm dispatch. The
 * force_custom_kernels flag lives here because it is consumed by both
 * kernels/matmul.cu (cuBLAS bypass) and cuda_kernel_registry.cpp (cuDNN
 * bypass for conv / batchnorm), and it carries no CUDA API dependency, so a
 * plain translation unit linked into tenzor_backend_cuda is the cleanest
 * shared home.
 */

#include "tenzor/backend/cuda_config.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace tenzor::cuda {

namespace {

// Tri-state: -1 = unresolved (resolve from env on first read), 0 = vendor
// dispatch (default), 1 = force custom kernels. Resolution is LAZY (on first
// force_custom_kernels() call) rather than at static-init: the CUDA backend
// .so is dlopen'd by tenzor::initialize(), and a process that sets
// TENZOR_FORCE_CUSTOM_KERNELS=1 just before init would otherwise have the env
// read at .so load time — which can precede the setenv — leaving custom
// kernels off when the caller asked for them on. Mirrors the g_allow_tf32
// pattern in kernels/matmul.cu.
auto force_custom_default_from_env() -> bool {
    const char* v = std::getenv("TENZOR_FORCE_CUSTOM_KERNELS");
    return v != nullptr &&
           (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0);
}

std::atomic<int> g_force_custom{-1};

} // namespace

auto force_custom_kernels() -> bool {
    int v = g_force_custom.load(std::memory_order_relaxed);
    if (v < 0) {
        v = force_custom_default_from_env() ? 1 : 0;
        int expected = -1;
        // If another thread (or an explicit set_force_custom_kernels)
        // resolved it first, keep that value rather than overwriting.
        g_force_custom.compare_exchange_strong(expected, v,
                                               std::memory_order_relaxed);
        v = g_force_custom.load(std::memory_order_relaxed);
    }
    return v != 0;
}

auto set_force_custom_kernels(bool value) -> void {
    g_force_custom.store(value ? 1 : 0, std::memory_order_relaxed);
}

} // namespace tenzor::cuda