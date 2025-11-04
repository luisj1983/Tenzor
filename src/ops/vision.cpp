/**
 * @file vision.cpp
 * @brief Implementation of vision-specific tensor operations
 */

#include "tenzor/ops/vision.hpp"
#include "tenzor/backend/dispatch.hpp"
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
    OpAttributes attrs;
    attrs["kernel_size"] = std::to_string(kernel_size);
    attrs["stride"] = std::to_string(stride);
    attrs["padding"] = std::to_string(padding);
    attrs["dilation"] = std::to_string(dilation);

    // Dispatch to appropriate backend
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("unfold", inputs, attrs)[0];
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
    OpAttributes attrs;
    attrs["output_size"] = size_str;
    attrs["kernel_size"] = std::to_string(kernel_size);
    attrs["stride"] = std::to_string(stride);
    attrs["padding"] = std::to_string(padding);
    attrs["dilation"] = std::to_string(dilation);

    // Dispatch to appropriate backend
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("fold", inputs, attrs)[0];
}

auto interpolate(const Tensor& input,
                const std::vector<int64_t>& size,
                const std::string& mode,
                bool align_corners) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument(
            "interpolate expects 4D input (N, C, H, W), got " +
            std::to_string(shape.size()) + "D");
    }

    if (size.size() != 2) {
        throw std::invalid_argument(
            "size must have 2 elements (H, W), got " +
            std::to_string(size.size()));
    }

    // If output size matches input size, return copy
    if (shape[2] == size[0] && shape[3] == size[1]) {
        return input;
    }

    // Validate mode
    if (mode != "nearest" && mode != "bilinear" && mode != "bicubic") {
        throw std::invalid_argument(
            "Unsupported interpolation mode: " + mode +
            ". Supported modes: 'nearest', 'bilinear', 'bicubic'");
    }

    // Serialize size as comma-separated string
    std::string size_str;
    for (size_t i = 0; i < size.size(); ++i) {
        if (i > 0) size_str += ",";
        size_str += std::to_string(size[i]);
    }

    // Set up operation attributes
    OpAttributes attrs;
    attrs["size"] = size_str;
    attrs["mode"] = mode;
    attrs["align_corners"] = align_corners ? "1" : "0";

    // Dispatch to appropriate backend
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("interpolate", inputs, attrs)[0];
}

} // namespace tenzor::ops
