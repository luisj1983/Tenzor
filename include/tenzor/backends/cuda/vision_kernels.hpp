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
 *        LL.3: per-axis kernel/stride/padding/dilation.
 */
auto unfold_cuda(const Tensor& input,
                 int64_t kernel_h, int64_t kernel_w,
                 int64_t stride_h, int64_t stride_w,
                 int64_t padding_h, int64_t padding_w,
                 int64_t dilation_h, int64_t dilation_w) -> Tensor;

/**
 * @brief Fold operation (col2im for sliding windows) - CUDA implementation
 *        LL.3: per-axis kernel/stride/padding/dilation.
 */
auto fold_cuda(const Tensor& input,
               const std::vector<int64_t>& output_size,
               int64_t kernel_h, int64_t kernel_w,
               int64_t stride_h, int64_t stride_w,
               int64_t padding_h, int64_t padding_w,
               int64_t dilation_h, int64_t dilation_w) -> Tensor;

/**
 * @brief Interpolate (resize) operation - CUDA implementation
 */
auto interpolate_cuda(const Tensor& input,
                      const std::vector<int64_t>& size,
                      const std::string& mode,
                      bool align_corners) -> Tensor;

/**
 * @brief Adaptive average pooling 2D forward - CUDA implementation
 */
auto adaptive_avg_pool2d_forward(const Tensor& input,
                                  int64_t output_h,
                                  int64_t output_w) -> Tensor;

/**
 * @brief Adaptive average pooling 2D backward - CUDA implementation
 */
auto adaptive_avg_pool2d_backward(const Tensor& grad_output,
                                   int64_t H_in,
                                   int64_t W_in) -> Tensor;

/**
 * @brief Gather operation for relative position bias - CUDA implementation
 */
auto gather_relative_position_bias(const Tensor& table,
                                    const Tensor& indices,
                                    int64_t num_positions,
                                    int64_t num_heads) -> Tensor;

/**
 * @brief Create shifted window attention mask - CUDA implementation
 */
auto create_shifted_window_mask_cuda(int64_t H, int64_t W,
                                      int64_t window_size,
                                      int64_t shift_size,
                                      DType dtype) -> Tensor;

} // namespace cuda
} // namespace tenzor
