/**
 * @file conv_kernels.hpp
 * @brief CUDA convolution kernel declarations
 */

#pragma once

#include "tenzor/core/tensor.hpp"
#include <cuda_runtime.h>
#include <tuple>

namespace tenzor {
namespace cuda {

/**
 * @brief Forward pass for 2D convolution using CUDA kernels
 *
 * Implements Conv2d forward using im2col + cuBLAS GEMM approach.
 * Supports FP32, FP64, and FP16 (with Tensor Cores).
 *
 * @param input Input tensor (batch, in_channels, height, width)
 * @param weight Weight tensor (out_channels, in_channels_per_group, kernel_h, kernel_w)
 * @param bias Optional bias tensor (out_channels) or nullptr
 * @param stride Stride for convolution
 * @param padding Padding amount
 * @param dilation Dilation factor
 * @param groups Number of groups for grouped convolution
 * @param stream CUDA stream for execution
 * @return Output tensor (batch, out_channels, out_h, out_w)
 */
auto conv2d_forward_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream = nullptr
) -> Tensor;

/**
 * @brief Backward pass for 2D convolution using CUDA kernels
 *
 * Computes gradients w.r.t. input, weight, and bias using im2col/col2im
 * approach with cuBLAS GEMM. Supports FP32, FP64, and FP16.
 *
 * @param grad_output Gradient w.r.t. output (batch, out_channels, out_h, out_w)
 * @param input Forward pass input (batch, in_channels, height, width)
 * @param weight Forward pass weight (out_channels, in_channels_per_group, kernel_h, kernel_w)
 * @param stride Stride used in forward pass
 * @param padding Padding used in forward pass
 * @param dilation Dilation used in forward pass
 * @param groups Number of groups used in forward pass
 * @param compute_grad_input Whether to compute gradient w.r.t. input
 * @param compute_grad_weight Whether to compute gradient w.r.t. weight
 * @param compute_grad_bias Whether to compute gradient w.r.t. bias
 * @param stream CUDA stream for execution
 * @return Tuple of (grad_input, grad_weight, grad_bias)
 */
auto conv2d_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool compute_grad_input,
    bool compute_grad_weight,
    bool compute_grad_bias,
    cudaStream_t stream = nullptr
) -> std::tuple<Tensor, Tensor, Tensor>;

} // namespace cuda
} // namespace tenzor
