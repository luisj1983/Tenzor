/**
 * @file vision_kernels.hpp
 * @brief CUDA kernel declarations for vision operations
 */

#pragma once

#include "tenzor/core/tensor.hpp"
#include <vector>
#include <string>

namespace tenzor {
namespace cuda {

/**
 * @brief Unfold operation (im2col for sliding windows) - CUDA implementation
 */
auto unfold_cuda(const Tensor& input,
                 int64_t kernel_size,
                 int64_t stride,
                 int64_t padding,
                 int64_t dilation) -> Tensor;

/**
 * @brief Fold operation (col2im for sliding windows) - CUDA implementation
 */
auto fold_cuda(const Tensor& input,
               const std::vector<int64_t>& output_size,
               int64_t kernel_size,
               int64_t stride,
               int64_t padding,
               int64_t dilation) -> Tensor;

/**
 * @brief Interpolate (resize) operation - CUDA implementation
 */
auto interpolate_cuda(const Tensor& input,
                      const std::vector<int64_t>& size,
                      const std::string& mode,
                      bool align_corners) -> Tensor;

} // namespace cuda
} // namespace tenzor
