#pragma once

/// @file miopen_guards.hpp
/// @brief RAII guards for MIOpen resources to prevent leaks on exception paths.

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

}  // namespace tenzor::rocm
