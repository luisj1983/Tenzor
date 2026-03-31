#pragma once
/**
 * @file rocm_error.hpp
 * @brief Shared error-checking macros for the ROCm/HIP backend.
 *
 * Provides HIP_CHECK, ROCBLAS_CHECK, and HIP_POST_LAUNCH_CHECK so that
 * every kernel file uses the same definitions instead of duplicating them.
 */

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// HIP runtime error checking
// ---------------------------------------------------------------------------
#ifndef HIP_CHECK
#define HIP_CHECK(call)                                                        \
    do {                                                                        \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            throw std::runtime_error(                                          \
                std::string("HIP error at ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + " - " + hipGetErrorString(err));    \
        }                                                                      \
    } while (0)
#endif

// ---------------------------------------------------------------------------
// Post-kernel-launch error check (shorthand for hipGetLastError)
// ---------------------------------------------------------------------------
#ifndef HIP_POST_LAUNCH_CHECK
#define HIP_POST_LAUNCH_CHECK() HIP_CHECK(hipGetLastError())
#endif

// ---------------------------------------------------------------------------
// rocBLAS status checking
// ---------------------------------------------------------------------------
#ifdef __has_include
#if __has_include(<rocblas/rocblas.h>)
#include <rocblas/rocblas.h>
#ifndef ROCBLAS_CHECK
#define ROCBLAS_CHECK(call)                                                    \
    do {                                                                        \
        rocblas_status status = (call);                                        \
        if (status != rocblas_status_success) {                                \
            throw std::runtime_error(                                          \
                std::string("rocBLAS error at ") + __FILE__ + ":" +           \
                std::to_string(__LINE__) + " - status " +                     \
                std::to_string(static_cast<int>(status)));                    \
        }                                                                      \
    } while (0)
#endif
#endif
#endif
