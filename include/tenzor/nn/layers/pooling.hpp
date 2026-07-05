/**
 * @file pooling.hpp
 * @brief Pooling layers for spatial downsampling
 *
 * Implements max pooling, average pooling, and adaptive pooling operations
 * for reducing spatial dimensions while retaining important features.
 */

#pragma once

#include "../module.hpp"
#include <array>

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
    /// Scalar (symmetric) ctor — delegates to the per-axis form with
    /// kernel/stride/padding replicated on both axes.
    MaxPool2d(int64_t kernel_size,
             int64_t stride = -1,  // Default: same as kernel_size
             int64_t padding = 0,
             bool ceil_mode = false,
             bool return_indices = false);

    /// Per-axis ctor accepting (kernel_h, kernel_w) tuples for asymmetric
    /// pooling — matches PyTorch's `MaxPool2d(kernel_size=(2, 3), stride=(2, 3))`.
    MaxPool2d(std::array<int64_t, 2> kernel_size,
             std::array<int64_t, 2> stride = {-1, -1},
             std::array<int64_t, 2> padding = {0, 0},
             bool ceil_mode = false,
             bool return_indices = false);

    /**
     * @brief Forward pass through max pooling.
     *
     * @param input Input variable of shape (N, C, H, W)
     * @return Pooled output of shape (N, C, H_out, W_out)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /// Whether ceil_mode is enabled for output size calculation.
    auto get_ceil_mode() const -> bool { return ceil_mode_; }

    /// Whether return_indices is enabled.
    auto get_return_indices() const -> bool { return return_indices_; }

    auto extra_repr() const -> std::string override {
        auto pair = [](int64_t a, int64_t b) -> std::string {
            if (a == b) return std::to_string(a);
            return "(" + std::to_string(a) + ", " + std::to_string(b) + ")";
        };
        return "kernel_size=" + pair(kernel_size_h_, kernel_size_w_) +
               ", stride="    + pair(stride_h_,      stride_w_) +
               ", padding="   + pair(padding_h_,     padding_w_);
    }

    auto get_kernel_size() const -> std::array<int64_t, 2> { return {kernel_size_h_, kernel_size_w_}; }
    auto get_stride()      const -> std::array<int64_t, 2> { return {stride_h_,      stride_w_}; }
    auto get_padding()     const -> std::array<int64_t, 2> { return {padding_h_,     padding_w_}; }

private:
    // Per-axis storage (H, W). Scalar ctor replicates the value across both axes.
    int64_t kernel_size_h_;
    int64_t kernel_size_w_;
    int64_t stride_h_;
    int64_t stride_w_;
    int64_t padding_h_;
    int64_t padding_w_;
    // Symmetric-only legacy fields, kept for backward-source-compatibility with
    // any external code accessing kernel_size_ etc. as int64_t. Initialised to
    // the H-axis value; check the get_kernel_size() / per-axis getters when
    // asymmetric values matter.
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    bool ceil_mode_;       ///< Use ceil instead of floor for output size
    bool return_indices_;  ///< Store indices of max values
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
    /// Scalar (symmetric) ctor.
    AvgPool2d(int64_t kernel_size,
             int64_t stride = -1,  // Default: same as kernel_size
             int64_t padding = 0,
             bool count_include_pad = true,
             bool ceil_mode = false);

    /// Per-axis ctor accepting (kernel_h, kernel_w) tuples.
    AvgPool2d(std::array<int64_t, 2> kernel_size,
             std::array<int64_t, 2> stride = {-1, -1},
             std::array<int64_t, 2> padding = {0, 0},
             bool count_include_pad = true,
             bool ceil_mode = false);

    /**
     * @brief Forward pass through average pooling.
     *
     * @param input Input variable of shape (N, C, H, W)
     * @return Pooled output of shape (N, C, H_out, W_out)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        auto pair = [](int64_t a, int64_t b) -> std::string {
            if (a == b) return std::to_string(a);
            return "(" + std::to_string(a) + ", " + std::to_string(b) + ")";
        };
        return "kernel_size=" + pair(kernel_size_h_, kernel_size_w_) +
               ", stride="    + pair(stride_h_,      stride_w_) +
               ", padding="   + pair(padding_h_,     padding_w_) +
               ", count_include_pad=" + (count_include_pad_ ? "true" : "false");
    }

    auto get_kernel_size() const -> std::array<int64_t, 2> { return {kernel_size_h_, kernel_size_w_}; }
    auto get_stride()      const -> std::array<int64_t, 2> { return {stride_h_,      stride_w_}; }
    auto get_padding()     const -> std::array<int64_t, 2> { return {padding_h_,     padding_w_}; }
    auto get_count_include_pad() const -> bool { return count_include_pad_; }
    auto get_ceil_mode() const -> bool { return ceil_mode_; }

private:
    int64_t kernel_size_h_, kernel_size_w_;
    int64_t stride_h_,      stride_w_;
    int64_t padding_h_,     padding_w_;
    int64_t kernel_size_, stride_, padding_;
    bool count_include_pad_;
    bool ceil_mode_;
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

    auto get_output_size() const -> std::array<int64_t, 2> { return {output_h_, output_w_}; }

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
    /// Scalar (symmetric) ctor.
    MaxPool3d(int64_t kernel_size,
             int64_t stride = -1,
             int64_t padding = 0,
             bool ceil_mode = false,
             bool return_indices = false);

    /// Per-axis ctor accepting (kernel_d, kernel_h, kernel_w) tuples.
    MaxPool3d(std::array<int64_t, 3> kernel_size,
             std::array<int64_t, 3> stride = {-1, -1, -1},
             std::array<int64_t, 3> padding = {0, 0, 0},
             bool ceil_mode = false,
             bool return_indices = false);

    auto forward_impl(const Variable& input) -> Variable override;

    auto get_ceil_mode() const -> bool { return ceil_mode_; }
    auto get_return_indices() const -> bool { return return_indices_; }

    auto get_kernel_size() const -> std::array<int64_t, 3> { return {kernel_size_d_, kernel_size_h_, kernel_size_w_}; }
    auto get_stride()      const -> std::array<int64_t, 3> { return {stride_d_,      stride_h_,      stride_w_}; }
    auto get_padding()     const -> std::array<int64_t, 3> { return {padding_d_,     padding_h_,     padding_w_}; }

private:
    int64_t kernel_size_d_, kernel_size_h_, kernel_size_w_;
    int64_t stride_d_,      stride_h_,      stride_w_;
    int64_t padding_d_,     padding_h_,     padding_w_;
    int64_t kernel_size_, stride_, padding_;
    bool ceil_mode_;
    bool return_indices_;
};

/**
 * @brief 3D average pooling layer.
 *
 * Shape: Input (N, C, D, H, W) -> Output (N, C, D_out, H_out, W_out)
 */
class AvgPool3d : public Module {
public:
    /// Scalar (symmetric) ctor.
    AvgPool3d(int64_t kernel_size,
             int64_t stride = -1,
             int64_t padding = 0,
             bool count_include_pad = true,
             bool ceil_mode = false);

    /// Per-axis ctor accepting (kernel_d, kernel_h, kernel_w) tuples.
    AvgPool3d(std::array<int64_t, 3> kernel_size,
             std::array<int64_t, 3> stride = {-1, -1, -1},
             std::array<int64_t, 3> padding = {0, 0, 0},
             bool count_include_pad = true,
             bool ceil_mode = false);

    auto forward_impl(const Variable& input) -> Variable override;

    auto get_kernel_size() const -> std::array<int64_t, 3> { return {kernel_size_d_, kernel_size_h_, kernel_size_w_}; }
    auto get_stride()      const -> std::array<int64_t, 3> { return {stride_d_,      stride_h_,      stride_w_}; }
    auto get_padding()     const -> std::array<int64_t, 3> { return {padding_d_,     padding_h_,     padding_w_}; }
    auto get_count_include_pad() const -> bool { return count_include_pad_; }
    auto get_ceil_mode() const -> bool { return ceil_mode_; }

private:
    int64_t kernel_size_d_, kernel_size_h_, kernel_size_w_;
    int64_t stride_d_,      stride_h_,      stride_w_;
    int64_t padding_d_,     padding_h_,     padding_w_;
    int64_t kernel_size_, stride_, padding_;
    bool count_include_pad_;
    bool ceil_mode_;
};

/**
 * @brief 1D max pooling layer.
 *
 * Shape: Input (N, C, L_in) -> Output (N, C, L_out)
 * L_out = floor((L_in + 2*padding - kernel_size) / stride + 1)
 */
class MaxPool1d : public Module {
public:
    MaxPool1d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0,
              bool ceil_mode = false, bool return_indices = false);
    auto forward_impl(const Variable& input) -> Variable override;

    auto get_ceil_mode() const -> bool { return ceil_mode_; }
    auto get_return_indices() const -> bool { return return_indices_; }

    auto get_kernel_size() const -> int64_t { return kernel_size_; }
    auto get_stride()      const -> int64_t { return stride_; }
    auto get_padding()     const -> int64_t { return padding_; }

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    bool ceil_mode_;
    bool return_indices_;
};

/**
 * @brief 1D average pooling layer.
 *
 * Shape: Input (N, C, L_in) -> Output (N, C, L_out)
 */
class AvgPool1d : public Module {
public:
    AvgPool1d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0,
              bool count_include_pad = true, bool ceil_mode = false);
    auto forward_impl(const Variable& input) -> Variable override;

    auto get_kernel_size() const -> int64_t { return kernel_size_; }
    auto get_stride()      const -> int64_t { return stride_; }
    auto get_padding()     const -> int64_t { return padding_; }
    auto get_count_include_pad() const -> bool { return count_include_pad_; }
    auto get_ceil_mode() const -> bool { return ceil_mode_; }

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    bool count_include_pad_;
    bool ceil_mode_;
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

    auto get_output_size() const -> int64_t { return output_size_; }

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

    auto get_output_size() const -> std::array<int64_t, 2> { return {output_h_, output_w_}; }

private:
    int64_t output_h_;
    int64_t output_w_;
};

/**
 * @brief 1D adaptive max pooling layer.
 *
 * Shape: Input (N, C, L_in) -> Output (N, C, output_size)
 */
class AdaptiveMaxPool1d : public Module {
public:
    explicit AdaptiveMaxPool1d(int64_t output_size);
    auto forward_impl(const Variable& input) -> Variable override;

    auto get_output_size() const -> int64_t { return output_size_; }

private:
    int64_t output_size_;
};

/**
 * @brief 3D adaptive max pooling layer.
 *
 * Shape: Input (N, C, D_in, H_in, W_in) -> Output (N, C, output_d, output_h, output_w)
 */
class AdaptiveMaxPool3d : public Module {
public:
    AdaptiveMaxPool3d(int64_t output_d, int64_t output_h, int64_t output_w);
    explicit AdaptiveMaxPool3d(int64_t output_size)
        : AdaptiveMaxPool3d(output_size, output_size, output_size) {}
    auto forward_impl(const Variable& input) -> Variable override;

    auto get_output_size() const -> std::array<int64_t, 3> { return {output_d_, output_h_, output_w_}; }

private:
    int64_t output_d_;
    int64_t output_h_;
    int64_t output_w_;
};

/**
 * @brief 3D adaptive average pooling layer.
 *
 * Shape: Input (N, C, D_in, H_in, W_in) -> Output (N, C, output_d, output_h, output_w)
 */
class AdaptiveAvgPool3d : public Module {
public:
    AdaptiveAvgPool3d(int64_t output_d, int64_t output_h, int64_t output_w);
    explicit AdaptiveAvgPool3d(int64_t output_size)
        : AdaptiveAvgPool3d(output_size, output_size, output_size) {}
    auto forward_impl(const Variable& input) -> Variable override;

    auto get_output_size() const -> std::array<int64_t, 3> { return {output_d_, output_h_, output_w_}; }

private:
    int64_t output_d_;
    int64_t output_h_;
    int64_t output_w_;
};

/**
 * @brief 1D power-average pooling layer.
 *
 * Computes: (sum(|x|^p, kernel) / kernel_size)^(1/p)
 * Equivalent to applying avg_pool on the p-th power of absolute values,
 * then taking the p-th root.
 *
 * Shape: Input (N, C, L_in) -> Output (N, C, L_out)
 * L_out = floor((L_in - kernel_size) / stride + 1)
 *
 * @code
 * LPPool1d pool(2, 3, 2);  // L2 pooling, kernel=3, stride=2
 * Variable x(Tensor({batch, 64, 100}, DType::Float32, Device::cpu()), true);
 * Variable pooled = pool.forward(x);
 * @endcode
 */
class LPPool1d : public Module {
public:
    /**
     * @brief Construct 1D LP pooling layer.
     *
     * @param norm_type The exponent p for Lp norm (must be >= 1)
     * @param kernel_size Size of the pooling window
     * @param stride Stride of the pooling window (default: same as kernel_size)
     */
    LPPool1d(int64_t norm_type, int64_t kernel_size, int64_t stride = -1);

    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        return "norm_type=" + std::to_string(norm_type_) +
               ", kernel_size=" + std::to_string(kernel_size_) +
               ", stride=" + std::to_string(stride_);
    }

private:
    int64_t norm_type_;    ///< Exponent p for the Lp norm
    int64_t kernel_size_;  ///< Pooling window size
    int64_t stride_;       ///< Stride
};

/**
 * @brief 2D power-average pooling layer.
 *
 * Computes: (sum(|x|^p, kernel) / kernel_size)^(1/p)
 * Applies avg_pool2d on the p-th power of absolute values, then takes
 * the p-th root.
 *
 * Shape: Input (N, C, H_in, W_in) -> Output (N, C, H_out, W_out)
 *
 * @code
 * LPPool2d pool(2, {3, 3}, {2, 2});  // L2 pooling, 3x3 kernel, stride 2
 * Variable x(Tensor({batch, 64, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable pooled = pool.forward(x);
 * @endcode
 */
class LPPool2d : public Module {
public:
    /**
     * @brief Construct 2D LP pooling layer.
     *
     * @param norm_type The exponent p for Lp norm (must be >= 1)
     * @param kernel_size Size of the pooling window (height, width)
     * @param stride Stride of the pooling window (default: same as kernel_size)
     */
    LPPool2d(int64_t norm_type,
             std::pair<int64_t, int64_t> kernel_size,
             std::pair<int64_t, int64_t> stride = {-1, -1});

    /**
     * @brief Convenience constructor for square kernel/stride.
     */
    LPPool2d(int64_t norm_type, int64_t kernel_size, int64_t stride = -1)
        : LPPool2d(norm_type, {kernel_size, kernel_size},
                   stride == -1 ? std::pair<int64_t,int64_t>{-1, -1} : std::pair<int64_t,int64_t>{stride, stride}) {}

    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        return "norm_type=" + std::to_string(norm_type_) +
               ", kernel_size=(" + std::to_string(kernel_size_.first) +
               ", " + std::to_string(kernel_size_.second) + ")" +
               ", stride=(" + std::to_string(stride_.first) +
               ", " + std::to_string(stride_.second) + ")";
    }

private:
    int64_t norm_type_;                       ///< Exponent p for the Lp norm
    std::pair<int64_t, int64_t> kernel_size_; ///< Pooling window (H, W)
    std::pair<int64_t, int64_t> stride_;      ///< Stride (H, W)
};

} // namespace nn
} // namespace tenzor
