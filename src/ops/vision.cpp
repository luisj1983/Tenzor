/**
 * @file vision.cpp
 * @brief Implementation of vision-specific tensor operations
 */

#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <stdexcept>
#include <algorithm>

namespace tenzor::ops {

namespace {

// Calculate output size for unfold operation
auto calculate_unfold_output_size(int64_t input_size, int64_t kernel_size,
                                   int64_t stride, int64_t padding, int64_t dilation) -> int64_t {
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

} // anonymous namespace

auto unfold(const Tensor& input,
            int64_t kernel_size,
            int64_t stride,
            int64_t padding,
            int64_t dilation) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument(
            "unfold expects 4D input (N, C, H, W), got " +
            std::to_string(shape.size()) + "D");
    }

    // Validate positivity of structural parameters before any division/modulo
    // (division by stride happens in the backend; guard here for a clean error).
    if (kernel_size <= 0) {
        throw std::invalid_argument(
            "unfold: kernel_size must be positive, got " + std::to_string(kernel_size));
    }
    if (stride <= 0) {
        throw std::invalid_argument(
            "unfold: stride must be positive, got " + std::to_string(stride));
    }
    if (dilation <= 0) {
        throw std::invalid_argument(
            "unfold: dilation must be positive, got " + std::to_string(dilation));
    }
    if (padding < 0) {
        throw std::invalid_argument(
            "unfold: padding must be non-negative, got " + std::to_string(padding));
    }

    // Set up operation attributes
    NewOpAttributes attrs;
    attrs.set(AttrKey::KernelSize, kernel_size);
    attrs.set(AttrKey::Stride, stride);
    attrs.set(AttrKey::Padding, padding);
    attrs.set(AttrKey::Dilation, dilation);

    // Dispatch to appropriate backend
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Unfold, inputs, attrs)[0];
}

auto fold(const Tensor& input,
          const std::vector<int64_t>& output_size,
          int64_t kernel_size,
          int64_t stride,
          int64_t padding,
          int64_t dilation) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument(
            "fold expects 3D input (N, C*K*K, L), got " +
            std::to_string(shape.size()) + "D");
    }

    if (output_size.size() != 2) {
        throw std::invalid_argument(
            "output_size must have 2 elements (H, W), got " +
            std::to_string(output_size.size()));
    }

    // Validate positivity of structural parameters before any division/modulo.
    // kernel_size guards the modulo on col_channels and the output-size formula;
    // stride/dilation guard the integer division in calculate_unfold_output_size().
    if (kernel_size <= 0) {
        throw std::invalid_argument(
            "fold: kernel_size must be positive, got " + std::to_string(kernel_size));
    }
    if (stride <= 0) {
        throw std::invalid_argument(
            "fold: stride must be positive, got " + std::to_string(stride));
    }
    if (dilation <= 0) {
        throw std::invalid_argument(
            "fold: dilation must be positive, got " + std::to_string(dilation));
    }
    if (padding < 0) {
        throw std::invalid_argument(
            "fold: padding must be non-negative, got " + std::to_string(padding));
    }

    int64_t col_channels = shape[1];

    // Validate channel dimension
    if (col_channels % (kernel_size * kernel_size) != 0) {
        throw std::invalid_argument(
            "Second dimension (" + std::to_string(col_channels) +
            ") must be divisible by kernel_size^2 (" +
            std::to_string(kernel_size * kernel_size) + ")");
    }

    // Validate num_blocks
    int64_t out_h = calculate_unfold_output_size(output_size[0], kernel_size, stride, padding, dilation);
    int64_t out_w = calculate_unfold_output_size(output_size[1], kernel_size, stride, padding, dilation);
    if (shape[2] != out_h * out_w) {
        throw std::invalid_argument(
            "Number of blocks (" + std::to_string(shape[2]) +
            ") doesn't match output size");
    }

    // Serialize output_size as comma-separated string
    std::string size_str;
    for (size_t i = 0; i < output_size.size(); ++i) {
        if (i > 0) size_str += ",";
        size_str += std::to_string(output_size[i]);
    }

    // Set up operation attributes
    NewOpAttributes attrs;
    attrs.set(AttrKey::OutputSize, size_str);
    attrs.set(AttrKey::KernelSize, kernel_size);
    attrs.set(AttrKey::Stride, stride);
    attrs.set(AttrKey::Padding, padding);
    attrs.set(AttrKey::Dilation, dilation);

    // Dispatch to appropriate backend
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Fold, inputs, attrs)[0];
}

auto interpolate(const Tensor& input,
                const std::vector<int64_t>& size,
                const std::string& mode,
                bool align_corners) -> Tensor {
    auto shape = input.shape();
    auto ndim = shape.size();

    // Validate input dimensionality
    if (ndim != 4 && ndim != 5) {
        throw std::invalid_argument(
            "interpolate expects 4D (N,C,H,W) or 5D (N,C,D,H,W) input, got " +
            std::to_string(ndim) + "D");
    }

    // Validate mode vs dimensionality
    if (mode == "trilinear") {
        if (ndim != 5) {
            throw std::invalid_argument(
                "trilinear interpolation requires 5D input (N,C,D,H,W), got " +
                std::to_string(ndim) + "D");
        }
        if (size.size() != 3) {
            throw std::invalid_argument(
                "trilinear size must have 3 elements (D,H,W), got " +
                std::to_string(size.size()));
        }
    } else if (mode == "nearest" || mode == "bilinear" || mode == "bicubic") {
        if (ndim == 5 && mode != "nearest") {
            throw std::invalid_argument(
                "5D input only supports 'nearest' or 'trilinear' mode, got '" + mode + "'");
        }
        size_t expected_size_len = (ndim == 4) ? 2 : 3;
        if (size.size() != expected_size_len) {
            throw std::invalid_argument(
                "size must have " + std::to_string(expected_size_len) +
                " elements, got " + std::to_string(size.size()));
        }
    } else {
        throw std::invalid_argument(
            "Unsupported interpolation mode: " + mode +
            ". Supported modes: 'nearest', 'bilinear', 'bicubic', 'trilinear'");
    }

    // Check for no-op (output size matches input spatial dims)
    bool same_size = true;
    for (size_t i = 0; i < size.size(); ++i) {
        if (shape[2 + i] != size[i]) { same_size = false; break; }
    }
    if (same_size) return input;

    // Serialize size as comma-separated string
    std::string size_str;
    for (size_t i = 0; i < size.size(); ++i) {
        if (i > 0) size_str += ",";
        size_str += std::to_string(size[i]);
    }

    // Set up operation attributes
    OpAttributes attrs;
    attrs.set(AttrKey::OutputSize, size_str);
    attrs.set(AttrKey::Mode, mode);
    attrs.set(AttrKey::AlignCorners, align_corners);

    // Dispatch to appropriate backend
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Interpolate, inputs, attrs)[0];
}

auto grid_sample(const Tensor& input,
                 const Tensor& grid,
                 const std::string& mode,
                 const std::string& padding_mode,
                 bool align_corners) -> Tensor {
    auto in_shape = input.shape();
    auto grid_shape = grid.shape();

    if (in_shape.size() != 4) {
        throw std::invalid_argument(
            "grid_sample expects 4D input (N, C, H, W), got " +
            std::to_string(in_shape.size()) + "D");
    }
    if (grid_shape.size() != 4 || grid_shape[3] != 2) {
        throw std::invalid_argument(
            "grid_sample expects grid of shape (N, H_out, W_out, 2)");
    }
    if (in_shape[0] != grid_shape[0]) {
        throw std::invalid_argument(
            "grid_sample: batch size mismatch between input and grid");
    }

    if (mode != "bilinear" && mode != "nearest" && mode != "bicubic") {
        throw std::invalid_argument(
            "grid_sample: mode must be 'bilinear', 'nearest', or 'bicubic', got '" + mode + "'");
    }
    if (padding_mode != "zeros" && padding_mode != "border" && padding_mode != "reflection") {
        throw std::invalid_argument(
            "grid_sample: padding_mode must be 'zeros', 'border', or 'reflection', got '" +
            padding_mode + "'");
    }

    OpAttributes attrs;
    attrs.set(AttrKey::Mode, mode);
    attrs.set(AttrKey::PaddingMode, padding_mode);
    attrs.set(AttrKey::AlignCorners, align_corners);

    std::vector<Tensor> inputs = {input, grid};
    return dispatch(OpId::GridSample, inputs, attrs)[0];
}

auto affine_grid(const Tensor& theta,
                 const std::vector<int64_t>& size,
                 bool align_corners) -> Tensor {
    auto theta_shape = theta.shape();

    if (theta_shape.size() != 3 || theta_shape[1] != 2 || theta_shape[2] != 3) {
        throw std::invalid_argument(
            "affine_grid expects theta of shape (N, 2, 3)");
    }
    if (size.size() != 4) {
        throw std::invalid_argument(
            "affine_grid expects size as {N, C, H, W}");
    }
    if (theta_shape[0] != size[0]) {
        throw std::invalid_argument(
            "affine_grid: batch size of theta must match size[0]");
    }

    // Serialize size as comma-separated string
    std::string size_str;
    for (size_t i = 0; i < size.size(); ++i) {
        if (i > 0) size_str += ",";
        size_str += std::to_string(size[i]);
    }

    OpAttributes attrs;
    attrs.set(AttrKey::OutputSize, size_str);
    attrs.set(AttrKey::AlignCorners, align_corners);

    std::vector<Tensor> inputs = {theta};
    return dispatch(OpId::AffineGrid, inputs, attrs)[0];
}

auto deformable_conv2d(const Tensor& input, const Tensor& offset,
                       const Tensor& weight, const Tensor& bias,
                       const Tensor& mask,
                       int64_t stride_h, int64_t stride_w,
                       int64_t padding_h, int64_t padding_w,
                       int64_t dilation_h, int64_t dilation_w,
                       int64_t groups, int64_t offset_groups) -> Tensor {
    auto ishape = input.shape();
    if (ishape.size() != 4) {
        throw std::invalid_argument(
            "deformable_conv2d expects 4D input (N, C, H, W), got " +
            std::to_string(ishape.size()) + "D");
    }
    auto wshape = weight.shape();
    if (wshape.size() != 4) {
        throw std::invalid_argument(
            "deformable_conv2d expects 4D weight (C_out, C_in/groups, kH, kW), got " +
            std::to_string(wshape.size()) + "D");
    }

    OpAttributes attrs;
    attrs.set(AttrKey::StrideH, stride_h);
    attrs.set(AttrKey::StrideW, stride_w);
    attrs.set(AttrKey::PaddingH, padding_h);
    attrs.set(AttrKey::PaddingW, padding_w);
    attrs.set(AttrKey::DilationH, dilation_h);
    attrs.set(AttrKey::DilationW, dilation_w);
    attrs.set(AttrKey::Groups, groups);
    attrs.set(AttrKey::OffsetGroups, offset_groups);
    // Create zero-element tensors for uninitialized bias/mask so dispatch doesn't crash
    bool has_bias = bias.is_valid() && bias.numel() > 0;
    bool has_mask = mask.is_valid() && mask.numel() > 0;
    attrs.set(AttrKey::UseMask, has_mask ? 1 : 0);

    Tensor safe_bias = has_bias ? bias : zeros({0}, input.dtype(), input.device());
    Tensor safe_mask = has_mask ? mask : zeros({0}, input.dtype(), input.device());

    std::vector<Tensor> inputs = {input, offset, weight, safe_bias, safe_mask};
    return dispatch(OpId::DeformableConv2dForward, inputs, attrs)[0];
}

} // namespace tenzor::ops
