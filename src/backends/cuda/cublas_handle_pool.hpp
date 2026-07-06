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
#include <unordered_map>

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
    ///
    /// When @p force_precise_f32 is true, the handle's math mode is forced to
    /// CUBLAS_DEFAULT_MATH (exact IEEE FP32) regardless of the global
    /// allow_tf32() toggle. This mirrors the cuDNN backward path's
    /// prefer_precise_f32 gate and is used by the native conv2d/ConvTranspose
    /// backward GEMMs so their gradients match CPU/other-backend exact FP32
    /// instead of dropping mantissa bits to TF32 on Ampere+. The math-mode
    /// change is tracked, so the next unforced get() re-applies the TF32 mode.
    static cublasHandle_t get(cudaStream_t stream = nullptr,
                              bool force_precise_f32 = false) {
        ensure_initialized();
        if (stream != last_stream()) {
            CUBLAS_CHECK(cublasSetStream(handle(), stream));
            last_stream() = stream;
        }
        apply_math_mode(force_precise_f32);
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
    // Per-thread, per-device handle cache. A single thread_local handle is
    // wrong on multi-GPU: a cuBLAS handle is bound to the device that was
    // current at cublasCreate() time, so reusing it after the thread switches
    // devices runs the GEMM on the wrong GPU (or fails). Key by the current
    // device, mirroring the device-keyed cublasLt/compute-capability caches in
    // cublas_ops.cu.
    static HandleGuard& guard() {
        static thread_local std::unordered_map<int, HandleGuard> guards;
        int device = 0;
        cudaGetDevice(&device);
        return guards[device];
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
    static void apply_math_mode(bool force_precise_f32 = false) {
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
        const int want =
            (!force_precise_f32 && ::tenzor::cuda::matmul::allow_tf32()) ? 1 : 0;
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
