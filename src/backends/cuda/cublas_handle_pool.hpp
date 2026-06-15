#pragma once

/**
 * @file cublas_handle_pool.hpp
 * @brief Centralized cuBLAS handle management for the CUDA backend.
 *
 * Provides a single cuBLAS handle shared across all CUDA kernel files
 * (matmul, conv2d, fused_ops, cublas_ops) to avoid redundant handle
 * creation and inconsistent lifetime management.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "tenzor/backend/cuda_config.hpp"

namespace tenzor {
namespace cuda {

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                        \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            throw std::runtime_error(                                          \
                std::string("cuBLAS error at ") + __FILE__ + ":" +            \
                std::to_string(__LINE__) + " - status code " +                \
                std::to_string(static_cast<int>(status)));                     \
        }                                                                      \
    } while (0)
#endif

class CuBLASHandlePool {
public:
    /// Returns a per-thread cuBLAS handle, optionally bound to the given stream.
    /// Each thread owns its own handle, so no data race on cublasSetStream.
    static cublasHandle_t get(cudaStream_t stream = nullptr) {
        ensure_initialized();
        if (stream != last_stream()) {
            CUBLAS_CHECK(cublasSetStream(handle(), stream));
            last_stream() = stream;
        }
        apply_math_mode();
        return handle();
    }

    static void shutdown() {
        // No-op: thread_local instances destroyed when each thread exits.
    }


private:
    // RAII guard owned per-thread; destroys the handle when the thread exits.
    struct HandleGuard {
        cublasHandle_t handle = nullptr;
        cudaStream_t last_stream = nullptr;
        // Tracks the math mode currently applied to the handle. Initialized to
        // an invalid sentinel so the first apply_math_mode() always sets it.
        int applied_tf32 = -1;  // -1 = unset, 0 = default math, 1 = TF32 op math
        bool sm_ge_80 = false;
        bool sm_checked = false;
        ~HandleGuard() {
            if (handle) {
                cublasDestroy(handle);
                handle = nullptr;
            }
        }
    };
    static HandleGuard& guard() {
        static thread_local HandleGuard g;
        return g;
    }
    static cublasHandle_t& handle() { return guard().handle; }
    static cudaStream_t& last_stream() { return guard().last_stream; }
    static void ensure_initialized() {
        if (handle() == nullptr) {
            CUBLAS_CHECK(cublasCreate(&handle()));
            // Math mode is set lazily by apply_math_mode() from allow_tf32(),
            // so plain (non-Ex) cublasSgemm calls honor the global TF32 toggle
            // instead of being silently forced into TF32 here.
        }
    }
    // Re-apply the handle math mode from the (runtime-toggleable, thread-local)
    // allow_tf32() setting. Only issues a cublasSetMathMode when the desired
    // mode actually changes, and only on SM>=8 where TF32 op math is available.
    static void apply_math_mode() {
#if CUDA_VERSION >= 11000
        auto& g = guard();
        if (!g.sm_checked) {
            int dev = 0;
            cudaGetDevice(&dev);
            int major = 0;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            g.sm_ge_80 = (major >= 8);
            g.sm_checked = true;
        }
        if (!g.sm_ge_80) {
            return;  // pre-Ampere: no TF32 path, default math only
        }
        const int want = ::tenzor::cuda::matmul::allow_tf32() ? 1 : 0;
        if (g.applied_tf32 != want) {
            CUBLAS_CHECK(cublasSetMathMode(
                g.handle,
                want ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
            g.applied_tf32 = want;
        }
#endif
    }

    CuBLASHandlePool() = delete;
    CuBLASHandlePool(const CuBLASHandlePool&) = delete;
    CuBLASHandlePool& operator=(const CuBLASHandlePool&) = delete;
};

} // namespace cuda
} // namespace tenzor
