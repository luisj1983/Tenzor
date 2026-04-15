/**
 * @file vision.hpp
 * @brief Vision-specific tensor operations
 *
 * Implements utility operations for computer vision models:
 * - Unfold/Fold for patch extraction (im2col/col2im)
 * - Window partition/reverse for Swin Transformer
 */

#pragma once

#include "../core/tensor.hpp"
#include "../core/device.hpp"
#include <vector>

namespace tenzor {
namespace ops {

/**
 * @brief Extract sliding local blocks (unfold/im2col operation).
 *
 * Extracts sliding local blocks from a batched input tensor.
 * Useful for patch extraction in Vision Transformers and efficient
 * convolution implementations.
 *
 * This is a more general version of im2col that works for any kernel size,
 * stride, padding, and dilation.
 *
 * @param input Input tensor of shape (N, C, H, W)
 * @param kernel_size Size of sliding blocks (square)
 * @param stride Stride of sliding blocks (default: 1)
 * @param padding Padding applied to input (default: 0)
 * @param dilation Dilation of kernel elements (default: 1)
 * @return Unfolded tensor of shape (N, C*K*K, L)
 *         where L = number of blocks = output_h * output_w
 *
 * Output dimensions:
 * - output_h = floor((H + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
 * - output_w = floor((W + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
 * - L = output_h * output_w
 *
 * @code
 * // Extract 16x16 patches with stride 16 (non-overlapping)
 * Tensor img({1, 3, 224, 224}, DType::Float32, Device::cpu());
 * Tensor patches = unfold(img, 16, 16, 0, 1);
 * // Shape: (1, 3*16*16, 196) where 196 = (224/16)^2
 *
 * // Extract 7x7 patches with stride 1 (overlapping)
 * Tensor feat({1, 64, 32, 32}, DType::Float32, Device::cpu());
 * Tensor blocks = unfold(feat, 7, 1, 3, 1);
 * // Shape: (1, 64*7*7, 1024) where 1024 = 32*32
 * @endcode
 */
auto unfold(const Tensor& input,
            int64_t kernel_size,
            int64_t stride = 1,
            int64_t padding = 0,
            int64_t dilation = 1) -> Tensor;

/**
 * @brief Fold tensor back to spatial dimensions (col2im).
 *
 * Reverse operation of unfold. Accumulates overlapping blocks into
 * output tensor. When blocks overlap, values are summed.
 *
 * @param input Unfolded tensor of shape (N, C*K*K, L)
 * @param output_size Output spatial size as (H, W)
 * @param kernel_size Size of blocks
 * @param stride Stride used in unfold
 * @param padding Padding used in unfold
 * @param dilation Dilation used in unfold
 * @return Folded tensor of shape (N, C, H, W)
 *
 * @code
 * Tensor unfolded({1, 768, 196}, DType::Float32, Device::cpu());
 * Tensor img = fold(unfolded, {224, 224}, 16, 16, 0, 1);
 * // Shape: (1, 3, 224, 224) where C = 768/(16*16) = 3
 * @endcode
 */
auto fold(const Tensor& input,
          const std::vector<int64_t>& output_size,
          int64_t kernel_size,
          int64_t stride = 1,
          int64_t padding = 0,
          int64_t dilation = 1) -> Tensor;

/**
 * @brief Interpolate tensor to specified size or scale factor.
 *
 * Resizes input tensor to target size using specified interpolation mode.
 * Commonly used for upsampling in segmentation networks and feature pyramid networks.
 *
 * @param input Input tensor of shape (N, C, H, W) or (N, C, D, H, W)
 * @param size Target output size as {H_out, W_out} or {D_out, H_out, W_out}
 * @param mode Interpolation mode: "nearest", "bilinear", "bicubic", "trilinear" (default: "bilinear")
 * @param align_corners If true, align corner pixels (default: false)
 * @return Interpolated tensor of same dimensionality as input
 *
 * **Modes:**
 * - "nearest": Nearest neighbor interpolation (fastest, blocky) — 4D or 5D
 * - "bilinear": Bilinear interpolation (smooth, commonly used) — 4D only
 * - "bicubic": Bicubic interpolation (smoother, slower) — 4D only
 * - "trilinear": Trilinear interpolation for volumetric data — 5D only
 *
 * **align_corners:**
 * - false (default): Pixels are treated as unit squares (PyTorch default)
 * - true: Pixels are treated as points at corners
 *
 * @code
 * // Upsample feature map 2x using bilinear
 * Tensor feat({1, 256, 32, 32}, DType::Float32, Device::cpu());
 * Tensor upsampled = interpolate(feat, {64, 64}, "bilinear");
 * // Shape: (1, 256, 64, 64)
 *
 * // Downsample image using nearest neighbor
 * Tensor img({1, 3, 256, 256}, DType::Float32, Device::cpu());
 * Tensor small = interpolate(img, {128, 128}, "nearest");
 * // Shape: (1, 3, 128, 128)
 * @endcode
 */
auto interpolate(const Tensor& input,
                const std::vector<int64_t>& size,
                const std::string& mode = "bilinear",
                bool align_corners = false) -> Tensor;

/**
 * @brief Spatial transformer network grid sampling.
 *
 * Samples from input using grid coordinates, supporting bilinear, nearest,
 * and bicubic interpolation with configurable padding modes.
 *
 * @param input Input tensor of shape (N, C, H_in, W_in)
 * @param grid Grid tensor of shape (N, H_out, W_out, 2) with values in [-1, 1]
 * @param mode Interpolation mode: "bilinear", "nearest", or "bicubic" (default: "bilinear")
 * @param padding_mode How to handle out-of-bound grid values: "zeros", "border", or "reflection"
 * @param align_corners If true, grid extremes (-1, 1) map to pixel centers at corners
 * @return Output tensor of shape (N, C, H_out, W_out)
 */
auto grid_sample(const Tensor& input,
                 const Tensor& grid,
                 const std::string& mode = "bilinear",
                 const std::string& padding_mode = "zeros",
                 bool align_corners = false) -> Tensor;

/**
 * @brief Generate a 2D affine grid for grid_sample.
 *
 * Given a batch of 2x3 affine matrices, generates a grid of (x, y) coordinates
 * that can be used with grid_sample() to perform spatial transformations.
 *
 * @param theta Affine transformation matrices of shape (N, 2, 3)
 * @param size Output spatial size as {N, C, H, W}
 * @param align_corners If true, grid extremes correspond to pixel corners
 * @return Grid tensor of shape (N, H, W, 2) with values in [-1, 1]
 */
auto affine_grid(const Tensor& theta,
                 const std::vector<int64_t>& size,
                 bool align_corners = false) -> Tensor;

/**
 * @brief Deformable 2D convolution (DCNv2).
 *
 * Applies a deformable convolution where sampling positions are augmented
 * with learnable 2D offsets and optional modulation masks, enabling the
 * receptive field to adapt to object geometry.
 *
 * Reference: Zhu et al., "Deformable ConvNets v2" (CVPR 2019)
 *
 * @param input Input tensor of shape (N, C_in, H, W)
 * @param offset Offset tensor of shape (N, offset_groups * 2 * kH * kW, H_out, W_out)
 * @param weight Weight tensor of shape (C_out, C_in / groups, kH, kW)
 * @param bias Bias tensor of shape (C_out), or empty tensor for no bias
 * @param mask Modulation mask of shape (N, offset_groups * kH * kW, H_out, W_out),
 *             or empty tensor for no mask (DCNv1 behavior, mask = 1)
 * @param stride_h Vertical stride
 * @param stride_w Horizontal stride
 * @param padding_h Vertical padding
 * @param padding_w Horizontal padding
 * @param dilation_h Vertical dilation
 * @param dilation_w Horizontal dilation
 * @param groups Number of blocked connections from input to output channels
 * @param offset_groups Number of groups for the offset/mask (typically 1)
 * @return Output tensor of shape (N, C_out, H_out, W_out)
 */
auto deformable_conv2d(const Tensor& input, const Tensor& offset,
                       const Tensor& weight, const Tensor& bias,
                       const Tensor& mask,
                       int64_t stride_h, int64_t stride_w,
                       int64_t padding_h, int64_t padding_w,
                       int64_t dilation_h, int64_t dilation_w,
                       int64_t groups, int64_t offset_groups) -> Tensor;

} // namespace ops
} // namespace tenzor
