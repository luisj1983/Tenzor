/**
 * @file vision.cpp
 * @brief Implementation of vision-specific tensor operations
 */

#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/creation.hpp"
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

    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t height = shape[2];
    int64_t width = shape[3];

    // Calculate output dimensions
    int64_t out_h = calculate_unfold_output_size(height, kernel_size, stride, padding, dilation);
    int64_t out_w = calculate_unfold_output_size(width, kernel_size, stride, padding, dilation);
    int64_t num_blocks = out_h * out_w;

    // Output shape: (N, C*K*K, L) where L = out_h * out_w
    auto output = zeros({batch, channels * kernel_size * kernel_size, num_blocks},
                       input.dtype(), input.device());

    // Transfer to CPU for processing if needed
    // TODO: Implement CUDA kernel for unfold operation
    Tensor input_cpu = (input.device().type == Device::Type::CUDA) ?
                       input.to(Device::cpu()) : input;
    Tensor output_cpu = (output.device().type == Device::Type::CUDA) ?
                        output.to(Device::cpu()) : output;

    const float* input_data = input_cpu.data<float>();
    float* output_data = output_cpu.data<float>();

    // Extract sliding blocks
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    // Column index in output
                    int64_t col_c = c * kernel_size * kernel_size + kh * kernel_size + kw;

                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        for (int64_t ow = 0; ow < out_w; ++ow) {
                            // Calculate input position with padding and dilation
                            int64_t ih = oh * stride - padding + kh * dilation;
                            int64_t iw = ow * stride - padding + kw * dilation;

                            int64_t output_idx = b * (channels * kernel_size * kernel_size * num_blocks) +
                                               col_c * num_blocks +
                                               oh * out_w + ow;

                            // Check bounds and apply padding
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int64_t input_idx = b * (channels * height * width) +
                                                   c * (height * width) +
                                                   ih * width + iw;
                                output_data[output_idx] = input_data[input_idx];
                            } else {
                                output_data[output_idx] = 0.0f;  // Padding with zeros
                            }
                        }
                    }
                }
            }
        }
    }

    // Transfer back to original device if needed
    return (input.device().type == Device::Type::CUDA) ?
           output_cpu.to(input.device()) : output_cpu;
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

    int64_t batch = shape[0];
    int64_t col_channels = shape[1];
    int64_t num_blocks = shape[2];

    int64_t height = output_size[0];
    int64_t width = output_size[1];

    // Validate channel dimension
    if (col_channels % (kernel_size * kernel_size) != 0) {
        throw std::invalid_argument(
            "Second dimension (" + std::to_string(col_channels) +
            ") must be divisible by kernel_size^2 (" +
            std::to_string(kernel_size * kernel_size) + ")");
    }

    int64_t channels = col_channels / (kernel_size * kernel_size);

    // Validate num_blocks
    int64_t out_h = calculate_unfold_output_size(height, kernel_size, stride, padding, dilation);
    int64_t out_w = calculate_unfold_output_size(width, kernel_size, stride, padding, dilation);
    if (num_blocks != out_h * out_w) {
        throw std::invalid_argument(
            "Number of blocks (" + std::to_string(num_blocks) +
            ") doesn't match output size");
    }

    // Output shape: (N, C, H, W)
    auto output = zeros({batch, channels, height, width}, input.dtype(), input.device());

    // Transfer to CPU for processing if needed
    // TODO: Implement CUDA kernel for fold operation
    Tensor input_cpu = (input.device().type == Device::Type::CUDA) ?
                       input.to(Device::cpu()) : input;
    Tensor output_cpu = (output.device().type == Device::Type::CUDA) ?
                        output.to(Device::cpu()) : output;

    const float* input_data = input_cpu.data<float>();
    float* output_data = output_cpu.data<float>();

    // Accumulate overlapping blocks
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    // Column index in input
                    int64_t col_c = c * kernel_size * kernel_size + kh * kernel_size + kw;

                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        for (int64_t ow = 0; ow < out_w; ++ow) {
                            // Calculate output position
                            int64_t ih = oh * stride - padding + kh * dilation;
                            int64_t iw = ow * stride - padding + kw * dilation;

                            // Check bounds
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int64_t input_idx = b * (col_channels * num_blocks) +
                                                   col_c * num_blocks +
                                                   oh * out_w + ow;

                                int64_t output_idx = b * (channels * height * width) +
                                                    c * (height * width) +
                                                    ih * width + iw;

                                // Accumulate (sum overlapping values)
                                output_data[output_idx] += input_data[input_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Transfer back to original device if needed
    return (input.device().type == Device::Type::CUDA) ?
           output_cpu.to(input.device()) : output_cpu;
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

    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t in_h = shape[2];
    int64_t in_w = shape[3];
    int64_t out_h = size[0];
    int64_t out_w = size[1];

    // If output size matches input size, return copy
    if (in_h == out_h && in_w == out_w) {
        return input;
    }

    // Create output tensor
    auto output = zeros({batch, channels, out_h, out_w}, input.dtype(), input.device());

    // Transfer to CPU for processing
    // TODO: Implement CUDA kernels for interpolation
    Tensor input_cpu = (input.device().type == Device::Type::CUDA) ?
                       input.to(Device::cpu()) : input;
    Tensor output_cpu = (output.device().type == Device::Type::CUDA) ?
                        output.to(Device::cpu()) : output;

    const float* input_data = input_cpu.data<float>();
    float* output_data = output_cpu.data<float>();

    if (mode == "nearest") {
        // Nearest neighbor interpolation
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        // Calculate source position
                        float scale_h = static_cast<float>(in_h) / out_h;
                        float scale_w = static_cast<float>(in_w) / out_w;

                        int64_t ih = static_cast<int64_t>(oh * scale_h);
                        int64_t iw = static_cast<int64_t>(ow * scale_w);

                        // Clamp to valid range
                        ih = std::clamp(ih, int64_t(0), in_h - 1);
                        iw = std::clamp(iw, int64_t(0), in_w - 1);

                        int64_t in_idx = b * (channels * in_h * in_w) +
                                        c * (in_h * in_w) +
                                        ih * in_w + iw;
                        int64_t out_idx = b * (channels * out_h * out_w) +
                                         c * (out_h * out_w) +
                                         oh * out_w + ow;

                        output_data[out_idx] = input_data[in_idx];
                    }
                }
            }
        }
    } else if (mode == "bilinear") {
        // Bilinear interpolation
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        // Calculate source position (floating point)
                        float y, x;
                        if (align_corners) {
                            // Align corners: map [0, out-1] to [0, in-1]
                            y = (out_h > 1) ? oh * static_cast<float>(in_h - 1) / (out_h - 1) : 0.0f;
                            x = (out_w > 1) ? ow * static_cast<float>(in_w - 1) / (out_w - 1) : 0.0f;
                        } else {
                            // Half-pixel centers: pixels are unit squares
                            float scale_h = static_cast<float>(in_h) / out_h;
                            float scale_w = static_cast<float>(in_w) / out_w;
                            y = (oh + 0.5f) * scale_h - 0.5f;
                            x = (ow + 0.5f) * scale_w - 0.5f;
                        }

                        // Clamp to valid range
                        y = std::clamp(y, 0.0f, static_cast<float>(in_h - 1));
                        x = std::clamp(x, 0.0f, static_cast<float>(in_w - 1));

                        // Get integer and fractional parts
                        int64_t y0 = static_cast<int64_t>(y);
                        int64_t x0 = static_cast<int64_t>(x);
                        int64_t y1 = std::min(y0 + 1, in_h - 1);
                        int64_t x1 = std::min(x0 + 1, in_w - 1);

                        float fy = y - y0;
                        float fx = x - x0;

                        // Bilinear interpolation weights
                        float w00 = (1.0f - fy) * (1.0f - fx);
                        float w01 = (1.0f - fy) * fx;
                        float w10 = fy * (1.0f - fx);
                        float w11 = fy * fx;

                        // Get pixel values
                        auto get_pixel = [&](int64_t h, int64_t w) -> float {
                            int64_t idx = b * (channels * in_h * in_w) +
                                         c * (in_h * in_w) +
                                         h * in_w + w;
                            return input_data[idx];
                        };

                        float v00 = get_pixel(y0, x0);
                        float v01 = get_pixel(y0, x1);
                        float v10 = get_pixel(y1, x0);
                        float v11 = get_pixel(y1, x1);

                        // Interpolate
                        float value = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;

                        int64_t out_idx = b * (channels * out_h * out_w) +
                                         c * (out_h * out_w) +
                                         oh * out_w + ow;
                        output_data[out_idx] = value;
                    }
                }
            }
        }
    } else {
        throw std::invalid_argument(
            "Unsupported interpolation mode: " + mode +
            ". Supported modes: 'nearest', 'bilinear'");
    }

    // Transfer back to original device if needed
    return (input.device().type == Device::Type::CUDA) ?
           output_cpu.to(input.device()) : output_cpu;
}

} // namespace tenzor::ops
