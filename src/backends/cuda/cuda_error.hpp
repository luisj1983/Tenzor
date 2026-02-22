#pragma once
// Centralized CUDA error handling macros for the Tenzor CUDA backend.
// Include this instead of defining per-file macros.

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA error at ") + __FILE__ + ":" +              \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(err));   \
        }                                                                      \
    } while (0)

// Check for errors from the most recent kernel launch.
// Must be called after every <<<>>> launch to catch async errors eagerly.
#define CUDA_POST_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

#ifdef TENZOR_HAS_CUBLAS
#include <cublas_v2.h>
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

#ifdef TENZOR_HAS_CUDNN
#include <cudnn.h>
#define CUDNN_CHECK(call)                                                      \
    do {                                                                        \
        cudnnStatus_t status = (call);                                         \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            throw std::runtime_error(                                          \
                std::string("cuDNN error at ") + __FILE__ + ":" +             \
                std::to_string(__LINE__) + " - " + cudnnGetErrorString(status));\
        }                                                                      \
    } while (0)
#endif
