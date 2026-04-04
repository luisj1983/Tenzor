/**
 * @file fp16_saturate.hpp
 * @brief FP16 saturation utility for OneAPI/SYCL kernels.
 *
 * Clamps ±Inf values to ±65504 (max finite Float16) to prevent NaN propagation
 * in deep networks. This matches the CUDA and ROCm backends' fp16_saturate
 * behavior.
 *
 * IEEE 754 FP32→FP16 conversion produces ±Inf for values outside [-65504, 65504].
 * When Inf interacts with other values (e.g., Inf - Inf, 0 * Inf), NaN results.
 * Saturating to finite bounds prevents this cascade.
 */
#pragma once

#include <sycl/sycl.hpp>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"

namespace tenzor {
namespace oneapi {

/**
 * @brief In-place FP16 saturation: clamp ±Inf to ±65504.
 *
 * @param data Pointer to Float16 data on device
 * @param n Number of elements
 * @param queue SYCL queue for kernel submission
 */
inline void fp16_saturate(void* data, int64_t n, sycl::queue& queue) {
    if (n <= 0 || data == nullptr) return;

    constexpr float kHalfMax = 65504.0f;
    auto* ptr = static_cast<sycl::half*>(data);

    queue.parallel_for(sycl::range<1>(static_cast<size_t>(n)),
        [=](sycl::id<1> idx) {
            float val = static_cast<float>(ptr[idx]);
            if (val > kHalfMax || val < -kHalfMax) {
                ptr[idx] = static_cast<sycl::half>(
                    sycl::fmin(sycl::fmax(val, -kHalfMax), kHalfMax));
            }
        }).wait();
}

/**
 * @brief Saturate a tensor's FP16 values if the tensor is Float16.
 *
 * No-op for non-Float16 tensors.
 *
 * @param tensor The tensor to saturate (modified in-place)
 * @param queue SYCL queue for kernel submission
 */
inline void fp16_saturate_if_needed(Tensor& tensor, sycl::queue& queue) {
    if (tensor.dtype() == DType::Float16 && tensor.numel() > 0) {
        fp16_saturate(tensor.data_ptr(), tensor.numel(), queue);
    }
}

} // namespace oneapi
} // namespace tenzor
