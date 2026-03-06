/**
 * @file vision.cpp
 * @brief Implementation of vision-specific tensor operations
 */

#include "tenzor/ops/vision.hpp"
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

} // namespace tenzor::ops
