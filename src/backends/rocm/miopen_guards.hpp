#pragma once

/// @file miopen_guards.hpp
/// @brief RAII guards for MIOpen resources to prevent leaks on exception paths.
///
/// MIOpen is unavailable on Windows (not shipped as part of the Windows HIP
/// SDK), so every actual use of these guards is already gated behind #ifdef
/// USE_MIOPEN at the call site (see conv2d.hip.cpp etc.) -- but this header
/// was itself included unconditionally there, and its own unconditional
/// #include <miopen/miopen.h> below fails to find the header on Windows.
/// Guarding the whole file keeps it a no-op when MIOpen isn't being used,
/// matching the call sites' existing convention.
#ifdef USE_MIOPEN

#include <miopen/miopen.h>
#include <stdexcept>
#include <string>

namespace tenzor::rocm {

/// RAII guard for miopenHandle_t.
struct MiopenHandleGuard {
    miopenHandle_t handle = nullptr;

    MiopenHandleGuard() {
        auto status = miopenCreate(&handle);
        if (status != miopenStatusSuccess) {
            throw std::runtime_error(
                std::string("miopenCreate failed with status ") +
                std::to_string(static_cast<int>(status)));
        }
    }

    ~MiopenHandleGuard() noexcept {
        if (handle) miopenDestroy(handle);
    }

    MiopenHandleGuard(const MiopenHandleGuard&) = delete;
    MiopenHandleGuard& operator=(const MiopenHandleGuard&) = delete;
};

/// RAII guard for miopenTensorDescriptor_t.
struct MiopenTensorDescGuard {
    miopenTensorDescriptor_t desc = nullptr;

    MiopenTensorDescGuard() {
        auto status = miopenCreateTensorDescriptor(&desc);
        if (status != miopenStatusSuccess) {
            throw std::runtime_error(
                std::string("miopenCreateTensorDescriptor failed with status ") +
                std::to_string(static_cast<int>(status)));
        }
    }

    ~MiopenTensorDescGuard() noexcept {
        if (desc) miopenDestroyTensorDescriptor(desc);
    }

    MiopenTensorDescGuard(const MiopenTensorDescGuard&) = delete;
    MiopenTensorDescGuard& operator=(const MiopenTensorDescGuard&) = delete;
};

/// RAII guard for miopenConvolutionDescriptor_t.
struct MiopenConvDescGuard {
    miopenConvolutionDescriptor_t desc = nullptr;

    MiopenConvDescGuard() {
        auto status = miopenCreateConvolutionDescriptor(&desc);
        if (status != miopenStatusSuccess) {
            throw std::runtime_error(
                std::string("miopenCreateConvolutionDescriptor failed with status ") +
                std::to_string(static_cast<int>(status)));
        }
    }

    ~MiopenConvDescGuard() noexcept {
        if (desc) miopenDestroyConvolutionDescriptor(desc);
    }

    MiopenConvDescGuard(const MiopenConvDescGuard&) = delete;
    MiopenConvDescGuard& operator=(const MiopenConvDescGuard&) = delete;
};

/// RAII guard for miopenTensorDescriptor_t used as a filter descriptor.
/// (MIOpen uses tensor descriptors for filters, not a separate type.)
using MiopenFilterDescGuard = MiopenTensorDescGuard;

/// RAII guard for miopenPoolingDescriptor_t.
struct MiopenPoolingDescGuard {
    miopenPoolingDescriptor_t desc = nullptr;

    MiopenPoolingDescGuard() {
        auto status = miopenCreatePoolingDescriptor(&desc);
        if (status != miopenStatusSuccess) {
            throw std::runtime_error(
                std::string("miopenCreatePoolingDescriptor failed with status ") +
                std::to_string(static_cast<int>(status)));
        }
    }

    ~MiopenPoolingDescGuard() noexcept {
        if (desc) miopenDestroyPoolingDescriptor(desc);
    }

    MiopenPoolingDescGuard(const MiopenPoolingDescGuard&) = delete;
    MiopenPoolingDescGuard& operator=(const MiopenPoolingDescGuard&) = delete;
};

/// RAII guard for miopenRNNDescriptor_t.
struct MiopenRNNDescGuard {
    miopenRNNDescriptor_t desc = nullptr;

    MiopenRNNDescGuard() {
        auto status = miopenCreateRNNDescriptor(&desc);
        if (status != miopenStatusSuccess) {
            throw std::runtime_error(
                std::string("miopenCreateRNNDescriptor failed with status ") +
                std::to_string(static_cast<int>(status)));
        }
    }

    ~MiopenRNNDescGuard() noexcept {
        if (desc) miopenDestroyRNNDescriptor(desc);
    }

    MiopenRNNDescGuard(const MiopenRNNDescGuard&) = delete;
    MiopenRNNDescGuard& operator=(const MiopenRNNDescGuard&) = delete;
};

/// Note: MIOpen batch normalization uses miopenTensorDescriptor_t for the
/// per-channel parameters (mean, variance, scale, bias), not a dedicated
/// batchnorm descriptor. Use MiopenTensorDescGuard for those descriptors.

}  // namespace tenzor::rocm

#endif  // USE_MIOPEN
