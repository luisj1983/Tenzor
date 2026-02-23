#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace tenzor::nn {

// Helper namespace for convolution operations
namespace {

// Calculate output size for convolution
auto calculate_output_size(int64_t input_size, int64_t kernel_size,
                           int64_t stride, int64_t padding, int64_t dilation) -> int64_t {
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

// Pad a 3D tensor [N, C, L] in the last dimension
auto pad_1d(const Tensor& input, int64_t padding) -> Tensor {
    if (padding <= 0) return input;

    auto shape = input.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];

    // Create zero padding tensors
    auto left_pad = zeros({batch, channels, padding}, input.dtype(), input.device());
    auto right_pad = zeros({batch, channels, padding}, input.dtype(), input.device());

    // Concatenate: [left_pad, input, right_pad]
    return cat({left_pad, input, right_pad}, 2);
}

// CPU fallback for im2col - only used when CUDA is not available
auto im2col_cpu(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
                int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w,
                int64_t dilation) -> Tensor {
    auto shape = input.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t height = shape[2];
    int64_t width = shape[3];

    int64_t out_h = calculate_output_size(height, kernel_h, stride_h, padding_h, dilation);
    int64_t out_w = calculate_output_size(width, kernel_w, stride_w, padding_w, dilation);

    // Convert to Float32 for im2col computation (matches PyTorch CPU conv fallback behavior)
    Tensor input_f32 = (input.dtype() != DType::Float32) ? input.to(DType::Float32) : input;
    auto col_f32 = zeros({batch, channels * kernel_h * kernel_w, out_h * out_w}, DType::Float32, input.device());
    const float* input_data = input_f32.data<float>();
    float* col_data = col_f32.data<float>();

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t col_c = c * kernel_h * kernel_w + kh * kernel_w + kw;

                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        for (int64_t ow = 0; ow < out_w; ++ow) {
                            int64_t ih = oh * stride_h - padding_h + kh * dilation;
                            int64_t iw = ow * stride_w - padding_w + kw * dilation;

                            int64_t col_idx = b * (channels * kernel_h * kernel_w * out_h * out_w) +
                                            col_c * (out_h * out_w) +
                                            oh * out_w + ow;

                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int64_t input_idx = b * (channels * height * width) +
                                                   c * (height * width) +
                                                   ih * width + iw;
                                col_data[col_idx] = input_data[input_idx];
                            } else {
                                col_data[col_idx] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }

    // Convert back to original dtype if needed
    return (input.dtype() != DType::Float32) ? col_f32.to(input.dtype()) : col_f32;
}

auto im2col_cpu(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
                int64_t stride, int64_t padding, int64_t dilation) -> Tensor {
    return im2col_cpu(input, kernel_h, kernel_w, stride, stride, padding, padding, dilation);
}

// CPU fallback for col2im - only used when CUDA is not available
auto col2im_cpu(const Tensor& col, int64_t channels, int64_t height, int64_t width,
                int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                int64_t padding_h, int64_t padding_w, int64_t dilation) -> Tensor {
    auto col_shape = col.shape();
    int64_t batch = col_shape[0];
    int64_t out_h = calculate_output_size(height, kernel_h, stride_h, padding_h, dilation);
    int64_t out_w = calculate_output_size(width, kernel_w, stride_w, padding_w, dilation);

    // Convert to Float32 for col2im computation (matches PyTorch CPU conv fallback behavior)
    Tensor col_f32 = (col.dtype() != DType::Float32) ? col.to(DType::Float32) : col;
    auto output_f32 = zeros({batch, channels, height, width}, DType::Float32, col.device());
    const float* col_data = col_f32.data<float>();
    float* output_data = output_f32.data<float>();

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t col_c = c * kernel_h * kernel_w + kh * kernel_w + kw;

                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        for (int64_t ow = 0; ow < out_w; ++ow) {
                            int64_t ih = oh * stride_h - padding_h + kh * dilation;
                            int64_t iw = ow * stride_w - padding_w + kw * dilation;

                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int64_t col_idx = b * (channels * kernel_h * kernel_w * out_h * out_w) +
                                                col_c * (out_h * out_w) +
                                                oh * out_w + ow;
                                int64_t output_idx = b * (channels * height * width) +
                                                    c * (height * width) +
                                                    ih * width + iw;
                                output_data[output_idx] += col_data[col_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Convert back to original dtype if needed
    return (col.dtype() != DType::Float32) ? output_f32.to(col.dtype()) : output_f32;
}

auto col2im_cpu(const Tensor& col, int64_t channels, int64_t height, int64_t width,
                int64_t kernel_h, int64_t kernel_w, int64_t stride,
                int64_t padding, int64_t dilation) -> Tensor {
    return col2im_cpu(col, channels, height, width, kernel_h, kernel_w,
                      stride, stride, padding, padding, dilation);
}

} // anonymous namespace

// ============================================================================
// Conv2dBackward - Autograd Function
// ============================================================================

class Conv2dBackward : public Function {
public:
    Conv2dBackward(int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                   std::vector<Tensor> tensors_to_save)
        : stride_(stride), padding_(padding), dilation_(dilation), groups_(groups) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("Conv2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];

        // Use backend dispatcher for gradient computation (routes to CUDA/cuDNN automatically)
        // Use single unified backward dispatch for better performance (one cuDNN call vs 3)
        std::vector<Tensor> tensors_for_dispatch = {grad_output};
        auto* backend = Dispatcher::get_backend(tensors_for_dispatch);

        bool has_bias = saved_tensors_.size() > 2;

        // Prepare attributes for unified backward
        OpAttributes backward_attrs = {
            {"stride", std::to_string(stride_)},
            {"padding", std::to_string(padding_)},
            {"dilation", std::to_string(dilation_)},
            {"groups", std::to_string(groups_)},
            {"compute_grad_input", "1"},
            {"compute_grad_weight", "1"},
            {"compute_grad_bias", has_bias ? "1" : "0"}
        };

        // Single dispatch call: conv2d_backward(grad_output, input, weight) -> [grad_input, grad_weight, grad_bias]
        std::vector<Tensor> backward_inputs = {grad_output, input, weight};
        auto backward_result = backend->dispatch(
            "conv2d_backward",
            backward_inputs,
            backward_attrs
        );

        if (has_bias) {
            return {backward_result[0], backward_result[1], backward_result[2]};
        }
        return {backward_result[0], backward_result[1]};
    }

private:
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t groups_;
};

// ============================================================================
// Conv2d Implementation
// ============================================================================

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
              int64_t stride, int64_t padding, int64_t dilation,
              int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride),
      padding_(padding), dilation_(dilation), groups_(groups <= 0 ? 1 : groups) {

    // Ensure groups is valid (must be >= 1)
    if (groups_ <= 0) {
        groups_ = 1;
    }
    if (in_channels % groups_ != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups_ != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }

    std::vector<int64_t> weight_shape = {out_channels, in_channels / groups_, kernel_size, kernel_size};
    int64_t fan_in = (in_channels / groups_) * kernel_size * kernel_size;
    float std_init = std::sqrt(2.0f / fan_in);
    auto weight_tensor = randn(weight_shape) * std_init;
    auto weight_init = Variable(weight_tensor, true);
    register_parameter("weight", weight_init);

    if (bias) {
        std::vector<int64_t> bias_shape = {out_channels};
        auto bias_init = Variable(zeros(bias_shape), true);
        register_parameter("bias", bias_init);
    }
}

auto Conv2d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("Conv2d expects 4D input [batch, channels, height, width]");
    }

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }

    int64_t out_h = calculate_output_size(height, kernel_size_, stride_, padding_, dilation_);
    int64_t out_w = calculate_output_size(width, kernel_size_, stride_, padding_, dilation_);

    if (out_h <= 0 || out_w <= 0) {
        throw std::runtime_error(
            "Invalid Conv2d configuration: output dimensions are non-positive (out_h=" +
            std::to_string(out_h) + ", out_w=" + std::to_string(out_w) + ")"
        );
    }

    auto& weight = *parameters_["weight"];
    auto bias_it = parameters_.find("bias");
    Device original_device = input.tensor().device();

    // Handle dtype and device mismatch: convert weight and bias to input's dtype and device if needed
    Variable weight_matched = weight;
    bool weight_needs_conversion = (input.dtype() != weight.dtype()) ||
                                   (input.tensor().device().type != weight.tensor().device().type);
    if (weight_needs_conversion) {
        auto weight_converted = weight.tensor();
        // Convert device first if needed
        if (input.tensor().device().type != weight.tensor().device().type) {
            weight_converted = weight_converted.to(original_device);
        }
        // Convert dtype if needed
        if (input.dtype() != weight_converted.dtype()) {
            weight_converted = weight_converted.to(input.dtype());
        }
        weight_matched = Variable(weight_converted, weight.requires_grad());
        weight_matched.set_grad_fn(weight.grad_fn());
    }

    const Tensor* bias_ptr = nullptr;
    Variable bias_matched;
    if (bias_it != parameters_.end()) {
        auto& bias = *bias_it->second;
        bool bias_needs_conversion = (input.dtype() != bias.dtype()) ||
                                     (input.tensor().device().type != bias.tensor().device().type);
        if (bias_needs_conversion) {
            auto bias_converted = bias.tensor();
            // Convert device first if needed
            if (input.tensor().device().type != bias.tensor().device().type) {
                bias_converted = bias_converted.to(original_device);
            }
            // Convert dtype if needed
            if (input.dtype() != bias_converted.dtype()) {
                bias_converted = bias_converted.to(input.dtype());
            }
            bias_matched = Variable(bias_converted, bias.requires_grad());
            bias_matched.set_grad_fn(bias.grad_fn());
            bias_ptr = &bias_matched.tensor();
        } else {
            bias_ptr = &bias.tensor();
        }
    }

    Tensor output;

    // Use backend dispatcher (routes to CUDA/cuDNN/CPU automatically based on tensor device)
    std::vector<Tensor> tensors_for_dispatch = {input.tensor()};
    auto* backend = Dispatcher::get_backend(tensors_for_dispatch);

    // Compute output using backend operation
    std::vector<Tensor> inputs_vec = {input.tensor(), weight_matched.tensor()};
    if (bias_ptr != nullptr) {
        inputs_vec.push_back(*bias_ptr);
    }

    OpAttributes forward_attrs = {
        {"stride", std::to_string(stride_)},
        {"padding", std::to_string(padding_)},
        {"dilation", std::to_string(dilation_)},
        {"groups", std::to_string(groups_)}
    };
    auto output_result = backend->dispatch(
        "conv2d_forward",
        std::span<const Tensor>(inputs_vec),
        forward_attrs
    );
    output = output_result[0];

    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    if (input.requires_grad() || weight.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_ptr != nullptr) {
            tensors_to_save = {input.tensor(), weight_matched.tensor(), *bias_ptr};
        } else {
            tensors_to_save = {input.tensor(), weight_matched.tensor()};
        }

        auto backward_fn = std::make_shared<Conv2dBackward>(
            stride_, padding_, dilation_, groups_, std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        // MUST include all inputs to maintain 1:1 index correspondence with gradients
        // The engine correctly skips variables that don't require grad
        std::vector<Variable> input_vars = {input, *parameters_["weight"]};
        if (bias_it != parameters_.end()) {
            input_vars.push_back(*bias_it->second);
        }
        backward_fn->set_input_variables(input_vars);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

auto Conv2d::reset_parameters() -> void {
    int64_t fan_in = in_channels_ / groups_ * kernel_size_ * kernel_size_;
    float std = std::sqrt(2.0f / fan_in);

    std::vector<int64_t> weight_shape = {out_channels_, in_channels_ / groups_, kernel_size_, kernel_size_};
    auto new_weight_tensor = randn(weight_shape) * std;
    parameters_["weight"] = std::make_shared<Variable>(new_weight_tensor, true);

    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        std::vector<int64_t> bias_shape = {out_channels_};
        bias_it->second = std::make_shared<Variable>(zeros(bias_shape), true);
    }
}

// ============================================================================
// Conv1d Implementation (similar structure with 1D operations)
// ============================================================================

class Conv1dBackward : public Function {
public:
    Conv1dBackward(int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                   std::vector<Tensor> tensors_to_save)
        : stride_(stride), padding_(padding), dilation_(dilation), groups_(groups) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("Conv1dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];

        // Manually pad the input in the length dimension (same as forward)
        Tensor input_padded = input;
        if (padding_ > 0) {
            input_padded = pad_1d(input, padding_);
        }

        // Add height dimension of 1 to convert to 4D for Conv2d operations
        auto grad_4d = grad_output.unsqueeze(2);
        auto input_4d = input_padded.unsqueeze(2);
        auto weight_4d = weight.unsqueeze(2);

        // Use backend dispatcher for Conv2d backward with padding=0
        std::vector<Tensor> tensors_for_dispatch = {grad_4d};
        auto* backend = Dispatcher::get_backend(tensors_for_dispatch);

        bool has_bias = saved_tensors_.size() > 2;

        OpAttributes backward_attrs = {
            {"stride", std::to_string(stride_)},
            {"padding", "0"},
            {"dilation", std::to_string(dilation_)},
            {"groups", std::to_string(groups_)},
            {"compute_grad_input", "1"},
            {"compute_grad_weight", "1"},
            {"compute_grad_bias", has_bias ? "1" : "0"}
        };

        std::vector<Tensor> backward_inputs = {grad_4d, input_4d, weight_4d};
        auto backward_result = backend->dispatch(
            "conv2d_backward",
            backward_inputs,
            backward_attrs
        );

        // Squeeze height dimension: [N,C,1,L] -> [N,C,L]
        Tensor grad_input_padded = backward_result[0].squeeze(2);
        Tensor grad_weight = backward_result[1].squeeze(2);

        // Remove padding from grad_input to match original input shape
        Tensor grad_input = grad_input_padded;
        if (padding_ > 0) {
            int64_t length = input.shape()[2];
            // Slice to remove padding: [N, C, L + 2*padding] -> [N, C, L]
            grad_input = grad_input_padded.slice(2, padding_, padding_ + length);
        }

        if (has_bias) {
            return {grad_input, grad_weight, backward_result[2]};
        }
        return {grad_input, grad_weight};
    }

private:
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t groups_;
};

Conv1d::Conv1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
               int64_t stride, int64_t padding, int64_t dilation,
               int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride),
      padding_(padding), dilation_(dilation), groups_(groups) {

    if (in_channels % groups != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }

    std::vector<int64_t> weight_shape = {out_channels, in_channels / groups, kernel_size};
    int64_t fan_in = (in_channels / groups) * kernel_size;
    float std_init = std::sqrt(2.0f / fan_in);
    auto weight_tensor = randn(weight_shape) * std_init;
    auto weight_var = Variable(weight_tensor, true);
    register_parameter("weight", weight_var);

    if (bias) {
        std::vector<int64_t> bias_shape = {out_channels};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto bias_tensor = (rand(bias_shape) * 2.0f * bound) - bound;
        auto bias_var = Variable(bias_tensor, true);
        register_parameter("bias", bias_var);
    }
}

auto Conv1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("Conv1d expects 3D input [batch, channels, length]");
    }

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t length = input_shape[2];

    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }

    int64_t out_len = calculate_output_size(length, kernel_size_, stride_, padding_, dilation_);
    if (out_len <= 0) {
        throw std::runtime_error(
            "Invalid Conv1d configuration: output length is non-positive (out_len=" +
            std::to_string(out_len) + ")"
        );
    }

    auto& weight = *parameters_["weight"];
    auto bias_it = parameters_.find("bias");
    Device original_device = input.tensor().device();

    // Handle dtype and device mismatch for weight
    Tensor weight_matched = weight.tensor();
    bool weight_needs_conversion = (input.dtype() != weight.dtype()) ||
                                   (input.tensor().device().type != weight.tensor().device().type);
    if (weight_needs_conversion) {
        if (input.tensor().device().type != weight.tensor().device().type) {
            weight_matched = weight_matched.to(original_device);
        }
        if (input.dtype() != weight_matched.dtype()) {
            weight_matched = weight_matched.to(input.dtype());
        }
    }

    const Tensor* bias_ptr = nullptr;
    Tensor bias_matched;
    if (bias_it != parameters_.end()) {
        auto& bias = *bias_it->second;
        bool bias_needs_conversion = (input.dtype() != bias.dtype()) ||
                                     (input.tensor().device().type != bias.tensor().device().type);
        if (bias_needs_conversion) {
            bias_matched = bias.tensor();
            if (input.tensor().device().type != bias.tensor().device().type) {
                bias_matched = bias_matched.to(original_device);
            }
            if (input.dtype() != bias_matched.dtype()) {
                bias_matched = bias_matched.to(input.dtype());
            }
            bias_ptr = &bias_matched;
        } else {
            bias_ptr = &bias.tensor();
        }
    }

    // Manually pad in the length dimension, then use padding=0 for conv2d
    // This avoids the symmetric padding issue where padding also affects height
    Tensor input_tensor = input.tensor();
    if (padding_ > 0) {
        input_tensor = pad_1d(input_tensor, padding_);
    }

    // Add height dimension of 1: [N, C, L] -> [N, C, 1, L]
    auto input_4d = input_tensor.unsqueeze(2);
    auto weight_4d = weight_matched.unsqueeze(2);

    // Use backend dispatcher for Conv2d with padding=0 (we already padded manually)
    std::vector<Tensor> tensors_for_dispatch = {input_4d};
    auto* backend = Dispatcher::get_backend(tensors_for_dispatch);

    std::vector<Tensor> inputs_vec = {input_4d, weight_4d};
    if (bias_ptr != nullptr) {
        inputs_vec.push_back(*bias_ptr);
    }

    // Use padding=0 since we already padded the input manually
    OpAttributes forward_attrs = {
        {"stride", std::to_string(stride_)},
        {"padding", "0"},
        {"dilation", std::to_string(dilation_)},
        {"groups", std::to_string(groups_)}
    };

    auto output_result = backend->dispatch(
        "conv2d_forward",
        std::span<const Tensor>(inputs_vec),
        forward_attrs
    );
    Tensor output_4d = output_result[0];

    // Remove height dimension: [N, C_out, 1, L_out] -> [N, C_out, L_out]
    Tensor output = output_4d.squeeze(2);

    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    if (input.requires_grad() || weight.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_ptr != nullptr) {
            tensors_to_save = {input.tensor(), weight.tensor(), *bias_ptr};
        } else {
            tensors_to_save = {input.tensor(), weight.tensor()};
        }

        auto backward_fn = std::make_shared<Conv1dBackward>(
            stride_, padding_, dilation_, groups_, std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        std::vector<Variable> input_vars = {input, *parameters_["weight"]};
        if (bias_it != parameters_.end()) {
            input_vars.push_back(*bias_it->second);
        }
        backward_fn->set_input_variables(input_vars);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

// ============================================================================
// ConvTranspose2d Implementation
// ============================================================================

class ConvTranspose2dBackward : public Function {
public:
    ConvTranspose2dBackward(int64_t stride, int64_t padding, int64_t output_padding,
                            int64_t dilation, int64_t groups,
                            std::vector<Tensor> tensors_to_save)
        : stride_(stride), padding_(padding), output_padding_(output_padding),
          dilation_(dilation), groups_(groups) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("ConvTranspose2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // ConvTranspose2d backward:
        // - grad_input = conv2d(grad_output, weight) with adjusted padding
        // - grad_weight computed from input and grad_output
        // - grad_bias = sum(grad_output, dims=[0,2,3])

        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];

        bool has_bias = saved_tensors_.size() > 2;

        auto grad_output_shape = grad_output.shape();
        auto input_shape = input.shape();
        auto weight_shape = weight.shape();

        int64_t batch = input_shape[0];
        int64_t in_channels = input_shape[1];
        int64_t out_channels = weight_shape[1] * groups_;
        int64_t kernel_h = weight_shape[2];
        int64_t kernel_w = weight_shape[3];

        // grad_input: Use regular conv2d forward with grad_output as input
        // The backward of ConvTranspose2d w.r.t. input is a regular Conv2d:
        //   grad_input = Conv2d(grad_output, weight, stride=stride, padding=padding)
        // The weight layout [in_ch, out_ch, kH, kW] matches Conv2d weight [C_out, C_in, kH, kW]
        // where C_out=in_ch (channels of grad_input) and C_in=out_ch (channels of grad_output).
        OpAttributes conv_attrs = {
            {"stride", std::to_string(stride_)},
            {"padding", std::to_string(padding_)},
            {"dilation", std::to_string(dilation_)},
            {"groups", std::to_string(groups_)}
        };

        std::vector<Tensor> conv_inputs = {grad_output, weight};
        auto conv_result = dispatch(OpId::Conv2dForward, std::span<const Tensor>(conv_inputs), conv_attrs);
        Tensor grad_input = conv_result[0];

        // Handle potential shape mismatch due to output_padding in the forward pass.
        // Slice grad_input to match the original input shape if dimensions differ.
        if (!std::equal(grad_input.shape().begin(), grad_input.shape().end(),
                        input_shape.begin(), input_shape.end())) {
            // Slice spatial dimensions to match input_shape
            auto gi_shape = grad_input.shape();
            if (gi_shape.size() == 4 && input_shape.size() == 4 &&
                gi_shape[0] == input_shape[0] && gi_shape[1] == input_shape[1]) {
                // Spatial dimensions may differ by output_padding amount - slice to match
                grad_input = tenzor::slice(grad_input, 2, 0, input_shape[2]);
                grad_input = tenzor::slice(grad_input, 3, 0, input_shape[3]);
            }
        }

        // grad_weight: For ConvTranspose2d, the weight gradient involves correlating
        // the input with grad_output. We dispatch to conv2d_backward_weight with swapped roles:
        // input (to ConvTranspose2d) acts as grad_output, and grad_output acts as input.
        OpAttributes weight_grad_attrs = {
            {"stride", std::to_string(stride_)},
            {"padding", std::to_string(padding_)},
            {"dilation", std::to_string(dilation_)},
            {"groups", std::to_string(groups_)},
            {"weight_shape", std::to_string(weight_shape[0]) + "," +
                             std::to_string(weight_shape[1]) + "," +
                             std::to_string(weight_shape[2]) + "," +
                             std::to_string(weight_shape[3])}
        };

        // Swap roles: input as grad_output, grad_output as input for weight gradient
        std::vector<Tensor> weight_grad_inputs = {input, grad_output};
        auto weight_grad_result = dispatch(OpId::Conv2dBackwardWeight, std::span<const Tensor>(weight_grad_inputs), weight_grad_attrs);
        Tensor grad_weight = weight_grad_result[0];

        if (has_bias) {
            // grad_bias = sum(grad_output, dims=[0,2,3])
            Tensor grad_bias = tenzor::sum(tenzor::sum(tenzor::sum(grad_output, 0, false), 1, false), 1, false);
            return {grad_input, grad_weight, grad_bias};
        }
        return {grad_input, grad_weight};
    }

private:
    int64_t stride_;
    int64_t padding_;
    int64_t output_padding_;
    int64_t dilation_;
    int64_t groups_;
};

ConvTranspose2d::ConvTranspose2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                                 int64_t stride, int64_t padding, int64_t output_padding,
                                 int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride),
      padding_(padding), output_padding_(output_padding), groups_(groups) {

    if (in_channels % groups != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }
    if (output_padding >= stride) {
        throw std::invalid_argument("output_padding must be smaller than stride");
    }

    std::vector<int64_t> weight_shape = {in_channels, out_channels / groups, kernel_size, kernel_size};
    int64_t fan_in = in_channels * kernel_size * kernel_size;
    float std_init = std::sqrt(2.0f / fan_in);
    auto weight_tensor = randn(weight_shape) * std_init;
    register_parameter("weight", Variable(weight_tensor, true));

    if (bias) {
        std::vector<int64_t> bias_shape = {out_channels};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto bias_tensor = (rand(bias_shape) * 2.0f * bound) - bound;
        register_parameter("bias", Variable(bias_tensor, true));
    }
}

auto ConvTranspose2d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("ConvTranspose2d expects 4D input [batch, channels, height, width]");
    }

    int64_t in_channels = input_shape[1];
    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }

    auto& weight = *parameters_["weight"];
    auto bias_it = parameters_.find("bias");
    Device original_device = input.tensor().device();

    // Handle dtype and device mismatch
    Variable weight_matched = weight;
    bool weight_needs_conversion = (input.dtype() != weight.dtype()) ||
                                   (input.tensor().device().type != weight.tensor().device().type);
    if (weight_needs_conversion) {
        auto weight_converted = weight.tensor();
        if (input.tensor().device().type != weight.tensor().device().type) {
            weight_converted = weight_converted.to(original_device);
        }
        if (input.dtype() != weight_converted.dtype()) {
            weight_converted = weight_converted.to(input.dtype());
        }
        weight_matched = Variable(weight_converted, weight.requires_grad());
        weight_matched.set_grad_fn(weight.grad_fn());
    }

    const Tensor* bias_ptr = nullptr;
    Variable bias_matched;
    if (bias_it != parameters_.end()) {
        auto& bias = *bias_it->second;
        bool bias_needs_conversion = (input.dtype() != bias.dtype()) ||
                                     (input.tensor().device().type != bias.tensor().device().type);
        if (bias_needs_conversion) {
            auto bias_converted = bias.tensor();
            if (input.tensor().device().type != bias.tensor().device().type) {
                bias_converted = bias_converted.to(original_device);
            }
            if (input.dtype() != bias_converted.dtype()) {
                bias_converted = bias_converted.to(input.dtype());
            }
            bias_matched = Variable(bias_converted, bias.requires_grad());
            bias_matched.set_grad_fn(bias.grad_fn());
            bias_ptr = &bias_matched.tensor();
        } else {
            bias_ptr = &bias.tensor();
        }
    }

    Tensor output;

    // Use CPU backend through operation registry
    std::vector<Tensor> tensors_for_dispatch = {input.tensor()};
    auto* backend = Dispatcher::get_backend(tensors_for_dispatch);

    std::vector<Tensor> inputs_vec = {input.tensor(), weight_matched.tensor()};
    if (bias_ptr != nullptr) {
        inputs_vec.push_back(*bias_ptr);
    }

    OpAttributes forward_attrs = {
        {"stride", std::to_string(stride_)},
        {"padding", std::to_string(padding_)},
        {"output_padding", std::to_string(output_padding_)},
        {"dilation", "1"},  // ConvTranspose2d uses dilation=1
        {"groups", std::to_string(groups_)}
    };
    auto output_result = backend->dispatch(
        "conv_transpose2d_forward",
        std::span<const Tensor>(inputs_vec),
        forward_attrs
    );
    output = output_result[0];

    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    // Set up autograd if needed
    if (input.requires_grad() || weight.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_ptr != nullptr) {
            tensors_to_save = {input.tensor(), weight_matched.tensor(), *bias_ptr};
        } else {
            tensors_to_save = {input.tensor(), weight_matched.tensor()};
        }

        auto backward_fn = std::make_shared<ConvTranspose2dBackward>(
            stride_, padding_, output_padding_, 1 /* dilation */, groups_,
            std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        // MUST include all inputs to maintain 1:1 index correspondence with gradients
        // The engine correctly skips variables that don't require grad
        std::vector<Variable> input_vars = {input, *parameters_["weight"]};
        if (bias_it != parameters_.end()) {
            input_vars.push_back(*bias_it->second);
        }
        backward_fn->set_input_variables(input_vars);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

auto ConvTranspose2d::reset_parameters() -> void {
    int64_t fan_in = in_channels_ * kernel_size_ * kernel_size_;
    float std = std::sqrt(2.0f / fan_in);

    std::vector<int64_t> weight_shape = {in_channels_, out_channels_ / groups_, kernel_size_, kernel_size_};
    auto new_weight_tensor = randn(weight_shape) * std;
    parameters_["weight"] = std::make_shared<Variable>(new_weight_tensor, true);

    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        std::vector<int64_t> bias_shape = {out_channels_};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto new_bias_tensor = (rand(bias_shape) * 2.0f * bound) - bound;
        bias_it->second = std::make_shared<Variable>(new_bias_tensor, true);
    }
}

// ============================================================================
// Conv3d Implementation
// ============================================================================

class Conv3dBackward : public Function {
public:
    Conv3dBackward(int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                   std::vector<Tensor> tensors_to_save)
        : stride_(stride), padding_(padding), dilation_(dilation), groups_(groups) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("Conv3dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];

        std::vector<Tensor> tensors_for_dispatch = {grad_output};
        auto* backend = Dispatcher::get_backend(tensors_for_dispatch);

        bool has_bias = saved_tensors_.size() > 2;

        OpAttributes backward_attrs = {
            {"stride", std::to_string(stride_)},
            {"padding", std::to_string(padding_)},
            {"dilation", std::to_string(dilation_)},
            {"groups", std::to_string(groups_)},
            {"compute_grad_input", "1"},
            {"compute_grad_weight", "1"},
            {"compute_grad_bias", has_bias ? "1" : "0"}
        };

        std::vector<Tensor> backward_inputs = {grad_output, input, weight};
        auto backward_result = backend->dispatch(
            "conv3d_backward",
            backward_inputs,
            backward_attrs
        );

        if (has_bias) {
            return {backward_result[0], backward_result[1], backward_result[2]};
        }
        return {backward_result[0], backward_result[1]};
    }

private:
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t groups_;
};

Conv3d::Conv3d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
              int64_t stride, int64_t padding, int64_t dilation,
              int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride),
      padding_(padding), dilation_(dilation), groups_(groups <= 0 ? 1 : groups) {

    if (groups_ <= 0) {
        groups_ = 1;
    }
    if (in_channels % groups_ != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups_ != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }

    // Weight shape: (C_out, C_in/groups, K, K, K)
    std::vector<int64_t> weight_shape = {out_channels, in_channels / groups_,
                                         kernel_size, kernel_size, kernel_size};
    int64_t fan_in = (in_channels / groups_) * kernel_size * kernel_size * kernel_size;
    float std_init = std::sqrt(2.0f / fan_in);
    auto weight_tensor = randn(weight_shape) * std_init;
    auto weight_init = Variable(weight_tensor, true);
    register_parameter("weight", weight_init);

    if (bias) {
        std::vector<int64_t> bias_shape = {out_channels};
        auto bias_init = Variable(zeros(bias_shape), true);
        register_parameter("bias", bias_init);
    }
}

auto Conv3d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("Conv3d expects 5D input [batch, channels, depth, height, width]");
    }

    int64_t in_channels = input_shape[1];
    int64_t depth = input_shape[2];
    int64_t height = input_shape[3];
    int64_t width = input_shape[4];

    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }

    int64_t out_d = calculate_output_size(depth, kernel_size_, stride_, padding_, dilation_);
    int64_t out_h = calculate_output_size(height, kernel_size_, stride_, padding_, dilation_);
    int64_t out_w = calculate_output_size(width, kernel_size_, stride_, padding_, dilation_);

    if (out_d <= 0 || out_h <= 0 || out_w <= 0) {
        throw std::runtime_error(
            "Invalid Conv3d configuration: output dimensions are non-positive (out_d=" +
            std::to_string(out_d) + ", out_h=" + std::to_string(out_h) +
            ", out_w=" + std::to_string(out_w) + ")");
    }

    auto& weight = *parameters_["weight"];
    auto bias_it = parameters_.find("bias");
    Device original_device = input.tensor().device();

    // Match weight dtype/device to input
    Variable weight_matched = weight;
    bool weight_needs_conversion = (input.dtype() != weight.dtype()) ||
                                   (input.tensor().device().type != weight.tensor().device().type);
    if (weight_needs_conversion) {
        auto weight_converted = weight.tensor();
        if (input.tensor().device().type != weight.tensor().device().type) {
            weight_converted = weight_converted.to(original_device);
        }
        if (input.dtype() != weight_converted.dtype()) {
            weight_converted = weight_converted.to(input.dtype());
        }
        weight_matched = Variable(weight_converted, weight.requires_grad());
        weight_matched.set_grad_fn(weight.grad_fn());
    }

    const Tensor* bias_ptr = nullptr;
    Variable bias_matched;
    if (bias_it != parameters_.end()) {
        auto& bias = *bias_it->second;
        bool bias_needs_conversion = (input.dtype() != bias.dtype()) ||
                                     (input.tensor().device().type != bias.tensor().device().type);
        if (bias_needs_conversion) {
            auto bias_converted = bias.tensor();
            if (input.tensor().device().type != bias.tensor().device().type) {
                bias_converted = bias_converted.to(original_device);
            }
            if (input.dtype() != bias_converted.dtype()) {
                bias_converted = bias_converted.to(input.dtype());
            }
            bias_matched = Variable(bias_converted, bias.requires_grad());
            bias_matched.set_grad_fn(bias.grad_fn());
            bias_ptr = &bias_matched.tensor();
        } else {
            bias_ptr = &bias.tensor();
        }
    }

    // Backend dispatch
    std::vector<Tensor> tensors_for_dispatch = {input.tensor()};
    auto* backend = Dispatcher::get_backend(tensors_for_dispatch);

    std::vector<Tensor> inputs_vec = {input.tensor(), weight_matched.tensor()};
    if (bias_ptr != nullptr) {
        inputs_vec.push_back(*bias_ptr);
    }

    OpAttributes forward_attrs = {
        {"stride", std::to_string(stride_)},
        {"padding", std::to_string(padding_)},
        {"dilation", std::to_string(dilation_)},
        {"groups", std::to_string(groups_)}
    };
    auto output_result = backend->dispatch(
        "conv3d_forward",
        std::span<const Tensor>(inputs_vec),
        forward_attrs
    );
    auto output = output_result[0];

    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    if (input.requires_grad() || weight.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_ptr != nullptr) {
            tensors_to_save = {input.tensor(), weight_matched.tensor(), *bias_ptr};
        } else {
            tensors_to_save = {input.tensor(), weight_matched.tensor()};
        }

        auto backward_fn = std::make_shared<Conv3dBackward>(
            stride_, padding_, dilation_, groups_, std::move(tensors_to_save));

        result.set_grad_fn(backward_fn);

        std::vector<Variable> input_vars = {input, *parameters_["weight"]};
        if (bias_it != parameters_.end()) {
            input_vars.push_back(*bias_it->second);
        }
        backward_fn->set_input_variables(input_vars);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

auto Conv3d::reset_parameters() -> void {
    int64_t fan_in = in_channels_ / groups_ * kernel_size_ * kernel_size_ * kernel_size_;
    float std = std::sqrt(2.0f / fan_in);
    auto new_weight = randn({out_channels_, in_channels_ / groups_,
                            kernel_size_, kernel_size_, kernel_size_}) * std;
    auto weight_ = Variable(new_weight, true);
    parameters_["weight"] = std::make_shared<Variable>(weight_);
}

} // namespace tenzor::nn
