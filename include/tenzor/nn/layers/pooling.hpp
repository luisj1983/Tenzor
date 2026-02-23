/**
 * @file pooling.hpp
 * @brief Pooling layers for spatial downsampling
 *
 * Implements max pooling, average pooling, and adaptive pooling operations
 * for reducing spatial dimensions while retaining important features.
 */

#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief 2D max pooling layer.
 *
 * Applies 2D max pooling over an input signal by taking the maximum value
 * within each pooling window. Commonly used for spatial downsampling in
 * convolutional networks while preserving salient features.
 *
 * Shape:
 * - Input: (N, C, H_in, W_in)
 * - Output: (N, C, H_out, W_out)
 *
 * Output dimensions:
 * - H_out = floor((H_in + 2*padding - kernel_size) / stride + 1)
 * - W_out = floor((W_in + 2*padding - kernel_size) / stride + 1)
 *
 * @code
 * MaxPool2d pool(2);  // 2x2 pooling, stride=2, halves spatial dims
 *
 * Variable x(Tensor({batch, 64, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable pooled = pool.forward(x);  // Shape: {batch, 64, 16, 16}
 * @endcode
 *
 * @see AvgPool2d for average pooling
 * @see AdaptiveAvgPool2d for adaptive pooling
 */
class MaxPool2d : public Module {
public:
    /**
     * @brief Construct 2D max pooling layer.
     *
     * @param kernel_size Size of pooling window (square)
     * @param stride Stride of pooling window (default: same as kernel_size)
     * @param padding Zero-padding added to input (default: 0)
     *
     * @code
     * MaxPool2d pool1(2);         // 2x2, stride=2
     * MaxPool2d pool2(3, 2);      // 3x3, stride=2
     * MaxPool2d pool3(2, 1, 1);   // 2x2, stride=1, padding=1
     * @endcode
     */
    MaxPool2d(int64_t kernel_size,
             int64_t stride = -1,  // Default: same as kernel_size
             int64_t padding = 0);

    /**
     * @brief Forward pass through max pooling.
     *
     * @param input Input variable of shape (N, C, H, W)
     * @return Pooled output of shape (N, C, H_out, W_out)
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t kernel_size_;  ///< Pooling window size
    int64_t stride_;       ///< Stride
    int64_t padding_;      ///< Padding
};

/**
 * @brief 2D average pooling layer.
 *
 * Applies 2D average pooling by computing the average value within each
 * pooling window. Provides smooth downsampling and can help reduce noise.
 *
 * Shape:
 * - Input: (N, C, H_in, W_in)
 * - Output: (N, C, H_out, W_out)
 *
 * @code
 * AvgPool2d pool(2);  // 2x2 average pooling
 *
 * Variable x(Tensor({batch, 64, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable pooled = pool.forward(x);  // Shape: {batch, 64, 16, 16}
 * @endcode
 *
 * @see MaxPool2d for max pooling
 */
class AvgPool2d : public Module {
public:
    /**
     * @brief Construct 2D average pooling layer.
     *
     * @param kernel_size Size of pooling window (square)
     * @param stride Stride of pooling window (default: same as kernel_size)
     * @param padding Zero-padding added to input (default: 0)
     */
    AvgPool2d(int64_t kernel_size,
             int64_t stride = -1,  // Default: same as kernel_size
             int64_t padding = 0);

    /**
     * @brief Forward pass through average pooling.
     *
     * @param input Input variable of shape (N, C, H, W)
     * @return Pooled output of shape (N, C, H_out, W_out)
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t kernel_size_;  ///< Pooling window size
    int64_t stride_;       ///< Stride
    int64_t padding_;      ///< Padding
};

/**
 * @brief 2D adaptive average pooling layer.
 *
 * Performs average pooling with adaptive kernel and stride sizes to produce
 * a fixed output size regardless of input dimensions. Useful for handling
 * variable-sized inputs in networks like ResNet and classification heads.
 *
 * The pooling parameters are automatically computed based on input and output sizes.
 *
 * Shape:
 * - Input: (N, C, H_in, W_in) - any spatial dimensions
 * - Output: (N, C, output_h, output_w) - fixed size
 *
 * @code
 * AdaptiveAvgPool2d pool(7, 7);  // Always output 7x7
 *
 * Variable x1(Tensor({batch, 512, 14, 14}, DType::Float32, Device::cpu()), true);
 * Variable out1 = pool.forward(x1);  // Shape: {batch, 512, 7, 7}
 *
 * Variable x2(Tensor({batch, 512, 28, 28}, DType::Float32, Device::cpu()), true);
 * Variable out2 = pool.forward(x2);  // Shape: {batch, 512, 7, 7}
 * @endcode
 *
 * @note Common use: AdaptiveAvgPool2d(1, 1) for global average pooling
 *
 * @see AvgPool2d for fixed-size average pooling
 */
class AdaptiveAvgPool2d : public Module {
public:
    /**
     * @brief Construct adaptive average pooling layer.
     *
     * @param output_h Output height
     * @param output_w Output width
     *
     * @code
     * AdaptiveAvgPool2d pool1(7, 7);    // Fixed 7x7 output
     * AdaptiveAvgPool2d pool2(1, 1);    // Global average pooling
     * @endcode
     */
    AdaptiveAvgPool2d(int64_t output_h, int64_t output_w);

    /**
     * @brief Convenience constructor for square output.
     *
     * @param output_size Both height and width of output
     *
     * @code
     * AdaptiveAvgPool2d pool(7);  // 7x7 output
     * @endcode
     */
    explicit AdaptiveAvgPool2d(int64_t output_size)
        : AdaptiveAvgPool2d(output_size, output_size) {}

    /**
     * @brief Forward pass through adaptive average pooling.
     *
     * Automatically computes pooling parameters to achieve desired output size.
     *
     * @param input Input variable of shape (N, C, H_in, W_in)
     * @return Pooled output of shape (N, C, output_h, output_w)
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t output_h_;  ///< Target output height
    int64_t output_w_;  ///< Target output width
};

/**
 * @brief 3D max pooling layer.
 *
 * Shape: Input (N, C, D, H, W) -> Output (N, C, D_out, H_out, W_out)
 */
class MaxPool3d : public Module {
public:
    MaxPool3d(int64_t kernel_size,
             int64_t stride = -1,
             int64_t padding = 0);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};

/**
 * @brief 3D average pooling layer.
 *
 * Shape: Input (N, C, D, H, W) -> Output (N, C, D_out, H_out, W_out)
 */
class AvgPool3d : public Module {
public:
    AvgPool3d(int64_t kernel_size,
             int64_t stride = -1,
             int64_t padding = 0);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};

/**
 * @brief 1D max pooling layer.
 *
 * Shape: Input (N, C, L_in) -> Output (N, C, L_out)
 * L_out = floor((L_in + 2*padding - kernel_size) / stride + 1)
 */
class MaxPool1d : public Module {
public:
    MaxPool1d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0);
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};

/**
 * @brief 1D average pooling layer.
 *
 * Shape: Input (N, C, L_in) -> Output (N, C, L_out)
 */
class AvgPool1d : public Module {
public:
    AvgPool1d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0);
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};

/**
 * @brief 1D adaptive average pooling layer.
 *
 * Shape: Input (N, C, L_in) -> Output (N, C, output_size)
 */
class AdaptiveAvgPool1d : public Module {
public:
    explicit AdaptiveAvgPool1d(int64_t output_size);
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t output_size_;
};

/**
 * @brief 2D adaptive max pooling layer.
 *
 * Shape: Input (N, C, H_in, W_in) -> Output (N, C, output_h, output_w)
 */
class AdaptiveMaxPool2d : public Module {
public:
    AdaptiveMaxPool2d(int64_t output_h, int64_t output_w);
    explicit AdaptiveMaxPool2d(int64_t output_size)
        : AdaptiveMaxPool2d(output_size, output_size) {}
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t output_h_;
    int64_t output_w_;
};

} // namespace nn
} // namespace tenzor
