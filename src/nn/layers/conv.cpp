#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <omp.h>

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

    // Only upcast Float16/BFloat16 to Float32; preserve Float32/Float64 natively
    bool needs_upcast = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
    DType work_dtype = needs_upcast ? DType::Float32 : input.dtype();
    Tensor input_work = needs_upcast ? input.to(DType::Float32) : input;
    auto col = zeros({batch, channels * kernel_h * kernel_w, out_h * out_w}, work_dtype, input.device());

    // Use typed lambda to handle both float and double paths
    auto do_im2col = [&](auto* input_data, auto* col_data) {
        using T = std::remove_pointer_t<decltype(input_data)>;
        const bool parallelize = batch * out_h * out_w > 4096;

        // Restructured loop order: batch x out_h are the two outermost loops
        // so they can be parallelized with collapse(2).  The inner loops over
        // channels/kernel elements write to non-overlapping col_idx positions
        // for each (b, oh) pair, so no synchronization is needed.
        #pragma omp parallel for collapse(2) if(parallelize) schedule(static)
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t c = 0; c < channels; ++c) {
                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t col_c = c * kernel_h * kernel_w + kh * kernel_w + kw;

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
                                    col_data[col_idx] = T(0);
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    if (work_dtype == DType::Float64) {
        do_im2col(input_work.data<double>(), col.data<double>());
    } else {
        do_im2col(input_work.data<float>(), col.data<float>());
    }

    // Convert back only if we upcasted
    return needs_upcast ? col.to(input.dtype()) : col;
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

    // Only upcast Float16/BFloat16 to Float32; preserve Float32/Float64 natively
    bool needs_upcast = (col.dtype() == DType::Float16 || col.dtype() == DType::BFloat16);
    DType work_dtype = needs_upcast ? DType::Float32 : col.dtype();
    Tensor col_work = needs_upcast ? col.to(DType::Float32) : col;
    auto output = zeros({batch, channels, height, width}, work_dtype, col.device());

    auto do_col2im = [&](const auto* col_data, auto* output_data) {
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
    };

    if (work_dtype == DType::Float64) {
        do_col2im(col_work.data<double>(), output.data<double>());
    } else {
        do_col2im(col_work.data<float>(), output.data<float>());
    }

    return needs_upcast ? output.to(col.dtype()) : output;
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
    Conv2dBackward(int64_t stride_h, int64_t stride_w,
                   int64_t padding_h, int64_t padding_w,
                   int64_t dilation_h, int64_t dilation_w,
                   int64_t groups,
                   std::vector<Tensor> tensors_to_save)
        : stride_h_(stride_h), stride_w_(stride_w),
          padding_h_(padding_h), padding_w_(padding_w),
          dilation_h_(dilation_h), dilation_w_(dilation_w),
          groups_(groups) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("Conv2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];

        bool has_bias = saved_tensors_.size() > 2;

        // Use OpId-based dispatch for each gradient component
        OpAttributes backward_attrs;
        backward_attrs.set(AttrKey::Stride, stride_h_);
        backward_attrs.set(AttrKey::Padding, padding_h_);
        backward_attrs.set(AttrKey::Dilation, dilation_h_);
        backward_attrs.set(AttrKey::StrideH, stride_h_);
        backward_attrs.set(AttrKey::StrideW, stride_w_);
        backward_attrs.set(AttrKey::PaddingH, padding_h_);
        backward_attrs.set(AttrKey::PaddingW, padding_w_);
        backward_attrs.set(AttrKey::DilationH, dilation_h_);
        backward_attrs.set(AttrKey::DilationW, dilation_w_);
        backward_attrs.set(AttrKey::Groups, groups_);

        std::vector<Tensor> backward_inputs = {grad_output, input, weight};

        // Dispatch individual backward ops via OpId
        auto grad_input_result = dispatch(OpId::Conv2dBackwardInput, backward_inputs, backward_attrs);
        auto grad_weight_result = dispatch(OpId::Conv2dBackwardWeight, backward_inputs, backward_attrs);

        if (has_bias) {
            auto grad_bias_result = dispatch(OpId::Conv2dBackwardBias, backward_inputs, backward_attrs);
            return {grad_input_result[0], grad_weight_result[0], grad_bias_result[0]};
        }
        return {grad_input_result[0], grad_weight_result[0]};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Delegate to tensor backward and wrap results
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

private:
    int64_t stride_h_, stride_w_;
    int64_t padding_h_, padding_w_;
    int64_t dilation_h_, dilation_w_;
    int64_t groups_;
};

// ============================================================================
// Conv2d Implementation
// ============================================================================

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
              int64_t stride, int64_t padding, int64_t dilation,
              int64_t groups, bool bias)
    : Conv2d(in_channels, out_channels,
             {kernel_size, kernel_size}, {stride, stride},
             {padding, padding}, {dilation, dilation},
             groups, bias) {}

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels,
              std::pair<int64_t, int64_t> kernel_size,
              std::pair<int64_t, int64_t> stride,
              std::pair<int64_t, int64_t> padding,
              std::pair<int64_t, int64_t> dilation,
              int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_h_(kernel_size.first), kernel_w_(kernel_size.second),
      stride_h_(stride.first), stride_w_(stride.second),
      padding_h_(padding.first), padding_w_(padding.second),
      dilation_h_(dilation.first), dilation_w_(dilation.second),
      groups_(groups <= 0 ? 1 : groups) {

    if (groups_ <= 0) {
        groups_ = 1;
    }
    if (in_channels % groups_ != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups_ != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }

    std::vector<int64_t> weight_shape = {out_channels, in_channels / groups_, kernel_h_, kernel_w_};
    int64_t fan_in = (in_channels / groups_) * kernel_h_ * kernel_w_;
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

    int64_t out_h = calculate_output_size(height, kernel_h_, stride_h_, padding_h_, dilation_h_);
    int64_t out_w = calculate_output_size(width, kernel_w_, stride_w_, padding_w_, dilation_w_);

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

    // Use backend dispatcher (routes to CUDA/cuDNN/CPU automatically based on tensor device)
    std::vector<Tensor> inputs_vec = {input.tensor(), weight_matched.tensor()};
    if (bias_ptr != nullptr) {
        inputs_vec.push_back(*bias_ptr);
    }

    // Pass both paired and single-value keys for backward compat with backends
    NewOpAttributes forward_attrs;
    forward_attrs.set(AttrKey::Stride, stride_h_);
    forward_attrs.set(AttrKey::Padding, padding_h_);
    forward_attrs.set(AttrKey::Dilation, dilation_h_);
    forward_attrs.set(AttrKey::StrideH, stride_h_);
    forward_attrs.set(AttrKey::StrideW, stride_w_);
    forward_attrs.set(AttrKey::PaddingH, padding_h_);
    forward_attrs.set(AttrKey::PaddingW, padding_w_);
    forward_attrs.set(AttrKey::DilationH, dilation_h_);
    forward_attrs.set(AttrKey::DilationW, dilation_w_);
    forward_attrs.set(AttrKey::Groups, groups_);
    auto output_result = dispatch_to_device(OpId::Conv2dForward, input.tensor().device().type,
        inputs_vec, forward_attrs);
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
            stride_h_, stride_w_, padding_h_, padding_w_,
            dilation_h_, dilation_w_, groups_, std::move(tensors_to_save)
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

auto Conv2d::reset_parameters() -> void {
    int64_t fan_in = in_channels_ / groups_ * kernel_h_ * kernel_w_;
    float std = std::sqrt(2.0f / fan_in);

    std::vector<int64_t> weight_shape = {out_channels_, in_channels_ / groups_, kernel_h_, kernel_w_};
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

        bool has_bias = saved_tensors_.size() > 2;

        OpAttributes backward_attrs;
        backward_attrs.set(AttrKey::Stride, stride_);
        backward_attrs.set(AttrKey::Padding, 0);
        backward_attrs.set(AttrKey::Dilation, dilation_);
        backward_attrs.set(AttrKey::Groups, groups_);

        std::vector<Tensor> backward_inputs = {grad_4d, input_4d, weight_4d};

        // Use OpId-based dispatch for each gradient component
        auto grad_input_result = dispatch(OpId::Conv2dBackwardInput, backward_inputs, backward_attrs);
        auto grad_weight_result = dispatch(OpId::Conv2dBackwardWeight, backward_inputs, backward_attrs);

        // Squeeze height dimension: [N,C,1,L] -> [N,C,L]
        Tensor grad_input_padded = grad_input_result[0].squeeze(2);
        Tensor grad_weight = grad_weight_result[0].squeeze(2);

        // Remove padding from grad_input to match original input shape
        Tensor grad_input = grad_input_padded;
        if (padding_ > 0) {
            int64_t length = input.shape()[2];
            // Slice to remove padding: [N, C, L + 2*padding] -> [N, C, L]
            grad_input = grad_input_padded.slice(2, padding_, padding_ + length);
        }

        if (has_bias) {
            auto grad_bias_result = dispatch(OpId::Conv2dBackwardBias, backward_inputs, backward_attrs);
            return {grad_input, grad_weight, grad_bias_result[0]};
        }
        return {grad_input, grad_weight};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Delegate to tensor backward and wrap results
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
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

    std::vector<Tensor> inputs_vec = {input_4d, weight_4d};
    if (bias_ptr != nullptr) {
        inputs_vec.push_back(*bias_ptr);
    }

    // Use padding=0 since we already padded the input manually
    NewOpAttributes forward_attrs;
    forward_attrs.set(AttrKey::Stride, stride_);
    forward_attrs.set(AttrKey::Padding, static_cast<int64_t>(0));
    forward_attrs.set(AttrKey::Dilation, dilation_);
    forward_attrs.set(AttrKey::Groups, groups_);

    auto output_result = dispatch(OpId::Conv2dForward,
        std::span<const Tensor>(inputs_vec),
        forward_attrs);
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
        NewOpAttributes conv_attrs;
        conv_attrs.set(AttrKey::Stride, stride_);
        conv_attrs.set(AttrKey::Padding, padding_);
        conv_attrs.set(AttrKey::Dilation, dilation_);
        conv_attrs.set(AttrKey::Groups, groups_);

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
        std::string ws_str = std::to_string(weight_shape[0]) + "," +
                             std::to_string(weight_shape[1]) + "," +
                             std::to_string(weight_shape[2]) + "," +
                             std::to_string(weight_shape[3]);
        NewOpAttributes weight_grad_attrs;
        weight_grad_attrs.set(AttrKey::Stride, stride_);
        weight_grad_attrs.set(AttrKey::Padding, padding_);
        weight_grad_attrs.set(AttrKey::Dilation, dilation_);
        weight_grad_attrs.set(AttrKey::Groups, groups_);
        weight_grad_attrs.set(AttrKey::WeightShape, std::string_view(ws_str));

        // Conv2dBackwardWeight expects [grad_output, input, weight]
        // For ConvTranspose2d weight grad, we swap roles: input acts as grad_output, grad_output acts as input
        std::vector<Tensor> weight_grad_inputs = {input, grad_output, weight};
        auto weight_grad_result = dispatch(OpId::Conv2dBackwardWeight, std::span<const Tensor>(weight_grad_inputs), weight_grad_attrs);
        Tensor grad_weight = weight_grad_result[0];

        if (has_bias) {
            // grad_bias = sum(grad_output, dims=[0,2,3])
            Tensor grad_bias = tenzor::sum(tenzor::sum(tenzor::sum(grad_output, 0, false), 1, false), 1, false);
            return {grad_input, grad_weight, grad_bias};
        }
        return {grad_input, grad_weight};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Delegate to tensor backward and wrap results
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
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

    std::vector<Tensor> inputs_vec = {input.tensor(), weight_matched.tensor()};
    if (bias_ptr != nullptr) {
        inputs_vec.push_back(*bias_ptr);
    }

    NewOpAttributes forward_attrs;
    forward_attrs.set(AttrKey::Stride, stride_);
    forward_attrs.set(AttrKey::Padding, padding_);
    forward_attrs.set(AttrKey::OutputPadding, output_padding_);
    forward_attrs.set(AttrKey::Dilation, static_cast<int64_t>(1));  // ConvTranspose2d uses dilation=1
    forward_attrs.set(AttrKey::Groups, groups_);
    auto output_result = dispatch(OpId::ConvTranspose2dForward,
        std::span<const Tensor>(inputs_vec),
        forward_attrs);
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
        bool has_bias = saved_tensors_.size() > 2;

        // Build shape strings for dispatch
        auto shape_to_str = [](std::span<const int64_t> s) {
            std::string r;
            for (size_t i = 0; i < s.size(); ++i) {
                if (i > 0) r += ",";
                r += std::to_string(s[i]);
            }
            return r;
        };

        NewOpAttributes common_attrs;
        common_attrs.set(AttrKey::Stride, stride_);
        common_attrs.set(AttrKey::Padding, padding_);
        common_attrs.set(AttrKey::Dilation, dilation_);
        common_attrs.set(AttrKey::Groups, groups_);

        // Backward input
        NewOpAttributes bi_attrs = common_attrs;
        bi_attrs.set(AttrKey::InputShape, std::string_view(shape_to_str(input.shape())));
        std::vector<Tensor> bi_inputs = {grad_output, weight};
        auto grad_input = dispatch<OpId::Conv3dBackwardInput>(bi_inputs, bi_attrs)[0];

        // Backward weight
        NewOpAttributes bw_attrs = common_attrs;
        bw_attrs.set(AttrKey::WeightShape, std::string_view(shape_to_str(weight.shape())));
        std::vector<Tensor> bw_inputs = {grad_output, input};
        auto grad_weight = dispatch<OpId::Conv3dBackwardWeight>(bw_inputs, bw_attrs)[0];

        if (has_bias) {
            std::vector<Tensor> bb_inputs = {grad_output};
            auto grad_bias = dispatch<OpId::Conv3dBackwardBias>(bb_inputs)[0];
            return {grad_input, grad_weight, grad_bias};
        }
        return {grad_input, grad_weight};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Delegate to tensor backward and wrap results
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
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
    std::vector<Tensor> inputs_vec = {input.tensor(), weight_matched.tensor()};
    if (bias_ptr != nullptr) {
        inputs_vec.push_back(*bias_ptr);
    }

    NewOpAttributes forward_attrs;
    forward_attrs.set(AttrKey::Stride, stride_);
    forward_attrs.set(AttrKey::Padding, padding_);
    forward_attrs.set(AttrKey::Dilation, dilation_);
    forward_attrs.set(AttrKey::Groups, groups_);
    auto output_result = dispatch<OpId::Conv3dForward>(inputs_vec, forward_attrs);
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

// ============================================================================
// ConvTranspose3d Implementation
// ============================================================================

class ConvTranspose3dBackward : public Function {
public:
    ConvTranspose3dBackward(int64_t stride, int64_t padding, int64_t output_padding,
                            int64_t dilation, int64_t groups,
                            std::vector<Tensor> tensors_to_save)
        : stride_(stride), padding_(padding), output_padding_(output_padding),
          dilation_(dilation), groups_(groups) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("ConvTranspose3dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];
        bool has_bias = saved_tensors_.size() > 2;

        auto shape_to_str = [](std::span<const int64_t> s) {
            std::string r;
            for (size_t i = 0; i < s.size(); ++i) {
                if (i > 0) r += ",";
                r += std::to_string(s[i]);
            }
            return r;
        };

        NewOpAttributes common_attrs;
        common_attrs.set(AttrKey::Stride, stride_);
        common_attrs.set(AttrKey::Padding, padding_);
        common_attrs.set(AttrKey::OutputPadding, output_padding_);
        common_attrs.set(AttrKey::Dilation, dilation_);
        common_attrs.set(AttrKey::Groups, groups_);

        // Backward input
        NewOpAttributes bi_attrs = common_attrs;
        bi_attrs.set(AttrKey::InputShape, std::string_view(shape_to_str(input.shape())));
        std::vector<Tensor> bi_inputs = {grad_output, weight};
        auto grad_input = dispatch<OpId::ConvTranspose3dBackwardInput>(bi_inputs, bi_attrs)[0];

        // Backward weight
        NewOpAttributes bw_attrs = common_attrs;
        bw_attrs.set(AttrKey::WeightShape, std::string_view(shape_to_str(weight.shape())));
        std::vector<Tensor> bw_inputs = {grad_output, input};
        auto grad_weight = dispatch<OpId::ConvTranspose3dBackwardWeight>(bw_inputs, bw_attrs)[0];

        if (has_bias) {
            // grad_bias = sum(grad_output, dims=[0,2,3,4])
            std::vector<Tensor> bb_inputs = {grad_output};
            auto grad_bias = dispatch<OpId::ConvTranspose3dBackwardBias>(bb_inputs)[0];
            return {grad_input, grad_weight, grad_bias};
        }
        return {grad_input, grad_weight};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

private:
    int64_t stride_;
    int64_t padding_;
    int64_t output_padding_;
    int64_t dilation_;
    int64_t groups_;
};

ConvTranspose3d::ConvTranspose3d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                                  int64_t stride, int64_t padding, int64_t output_padding,
                                  int64_t dilation, int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride),
      padding_(padding), output_padding_(output_padding),
      dilation_(dilation), groups_(groups) {

    if (in_channels % groups != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }
    if (output_padding >= stride) {
        throw std::invalid_argument("output_padding must be smaller than stride");
    }

    // Weight shape: (C_in, C_out/groups, K, K, K)
    std::vector<int64_t> weight_shape = {in_channels, out_channels / groups,
                                         kernel_size, kernel_size, kernel_size};
    int64_t fan_in = in_channels * kernel_size * kernel_size * kernel_size;
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

auto ConvTranspose3d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("ConvTranspose3d expects 5D input [batch, channels, depth, height, width]");
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

    // Dispatch forward via OpId
    std::vector<Tensor> inputs_vec = {input.tensor(), weight_matched.tensor()};
    if (bias_ptr != nullptr) {
        inputs_vec.push_back(*bias_ptr);
    }

    NewOpAttributes forward_attrs;
    forward_attrs.set(AttrKey::Stride, stride_);
    forward_attrs.set(AttrKey::Padding, padding_);
    forward_attrs.set(AttrKey::OutputPadding, output_padding_);
    forward_attrs.set(AttrKey::Dilation, dilation_);
    forward_attrs.set(AttrKey::Groups, groups_);

    auto output_result = dispatch<OpId::ConvTranspose3dForward>(inputs_vec, forward_attrs);
    auto output = output_result[0];

    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    // Set up autograd
    if (input.requires_grad() || weight.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_ptr != nullptr) {
            tensors_to_save = {input.tensor(), weight_matched.tensor(), *bias_ptr};
        } else {
            tensors_to_save = {input.tensor(), weight_matched.tensor()};
        }

        auto backward_fn = std::make_shared<ConvTranspose3dBackward>(
            stride_, padding_, output_padding_, dilation_, groups_,
            std::move(tensors_to_save)
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

auto ConvTranspose3d::reset_parameters() -> void {
    int64_t fan_in = in_channels_ * kernel_size_ * kernel_size_ * kernel_size_;
    float std = std::sqrt(2.0f / fan_in);

    std::vector<int64_t> weight_shape = {in_channels_, out_channels_ / groups_,
                                         kernel_size_, kernel_size_, kernel_size_};
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
// ConvTranspose1d Implementation
// ============================================================================

class ConvTranspose1dBackward : public Function {
public:
    ConvTranspose1dBackward(int64_t stride, int64_t padding, int64_t output_padding,
                            int64_t groups, std::vector<Tensor> tensors_to_save)
        : stride_(stride), padding_(padding), output_padding_(output_padding),
          groups_(groups) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("ConvTranspose1dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];
        bool has_bias = saved_tensors_.size() > 2;

        auto input_shape = input.shape();

        // Convert to 4D for ConvTranspose2d backward dispatch
        auto grad_4d = grad_output.unsqueeze(2);     // [N, C_out, L] -> [N, C_out, 1, L]
        auto input_4d = input.unsqueeze(2);           // [N, C_in, L] -> [N, C_in, 1, L]
        auto weight_4d = weight.unsqueeze(2);         // [C_in, C_out/g, K] -> [C_in, C_out/g, 1, K]

        // grad_input: backward of ConvTranspose w.r.t. input is regular Conv2d
        NewOpAttributes conv_attrs;
        conv_attrs.set(AttrKey::Stride, stride_);
        conv_attrs.set(AttrKey::Padding, padding_);
        conv_attrs.set(AttrKey::Dilation, static_cast<int64_t>(1));
        conv_attrs.set(AttrKey::Groups, groups_);

        std::vector<Tensor> conv_inputs = {grad_4d, weight_4d};
        auto conv_result = dispatch(OpId::Conv2dForward, std::span<const Tensor>(conv_inputs), conv_attrs);
        Tensor grad_input_4d = conv_result[0];

        // Handle potential shape mismatch from output_padding
        auto gi_shape = grad_input_4d.shape();
        if (gi_shape.size() == 4 && gi_shape[2] != 1) {
            grad_input_4d = tenzor::slice(grad_input_4d, 2, 0, 1);
        }
        if (gi_shape[3] != input_shape[2]) {
            grad_input_4d = tenzor::slice(grad_input_4d, 3, 0, input_shape[2]);
        }

        Tensor grad_input = grad_input_4d.squeeze(2);

        // grad_weight: swap roles of input and grad_output
        auto weight_4d_shape = weight_4d.shape();
        std::string ws_str = std::to_string(weight_4d_shape[0]) + "," +
                             std::to_string(weight_4d_shape[1]) + "," +
                             std::to_string(weight_4d_shape[2]) + "," +
                             std::to_string(weight_4d_shape[3]);
        NewOpAttributes weight_grad_attrs;
        weight_grad_attrs.set(AttrKey::Stride, stride_);
        weight_grad_attrs.set(AttrKey::Padding, padding_);
        weight_grad_attrs.set(AttrKey::Dilation, static_cast<int64_t>(1));
        weight_grad_attrs.set(AttrKey::Groups, groups_);
        weight_grad_attrs.set(AttrKey::WeightShape, std::string_view(ws_str));

        // Conv2dBackwardWeight expects [grad_output, input, weight]
        std::vector<Tensor> weight_grad_inputs = {input_4d, grad_4d, weight_4d};
        auto weight_grad_result = dispatch(OpId::Conv2dBackwardWeight,
            std::span<const Tensor>(weight_grad_inputs), weight_grad_attrs);
        Tensor grad_weight = weight_grad_result[0].squeeze(2);

        if (has_bias) {
            // grad_bias = sum(grad_output, dims=[0,2])
            Tensor grad_bias = tenzor::sum(tenzor::sum(grad_output, 0, false), 1, false);
            return {grad_input, grad_weight, grad_bias};
        }
        return {grad_input, grad_weight};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

private:
    int64_t stride_;
    int64_t padding_;
    int64_t output_padding_;
    int64_t groups_;
};

ConvTranspose1d::ConvTranspose1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
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

    // Weight shape: [in_channels, out_channels/groups, kernel_size]
    std::vector<int64_t> weight_shape = {in_channels, out_channels / groups, kernel_size};
    int64_t fan_in = in_channels * kernel_size;
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

auto ConvTranspose1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("ConvTranspose1d expects 3D input [batch, channels, length]");
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

    // Unsqueeze to 4D: [N, C, L] -> [N, C, 1, L]
    auto input_4d = input.tensor().unsqueeze(2);
    auto weight_4d = weight_matched.tensor().unsqueeze(2);

    // Dispatch to ConvTranspose2d forward
    std::vector<Tensor> inputs_vec = {input_4d, weight_4d};
    if (bias_ptr != nullptr) {
        inputs_vec.push_back(*bias_ptr);
    }

    NewOpAttributes forward_attrs;
    forward_attrs.set(AttrKey::Stride, stride_);
    forward_attrs.set(AttrKey::Padding, padding_);
    forward_attrs.set(AttrKey::OutputPadding, output_padding_);
    forward_attrs.set(AttrKey::Dilation, static_cast<int64_t>(1));
    forward_attrs.set(AttrKey::Groups, groups_);
    auto output_result = dispatch(OpId::ConvTranspose2dForward,
        std::span<const Tensor>(inputs_vec),
        forward_attrs);
    Tensor output_4d = output_result[0];

    // Squeeze back: [N, C_out, 1, L_out] -> [N, C_out, L_out]
    Tensor output = output_4d.squeeze(2);

    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    if (input.requires_grad() || weight.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_ptr != nullptr) {
            tensors_to_save = {input.tensor(), weight_matched.tensor(), *bias_ptr};
        } else {
            tensors_to_save = {input.tensor(), weight_matched.tensor()};
        }

        auto backward_fn = std::make_shared<ConvTranspose1dBackward>(
            stride_, padding_, output_padding_, groups_, std::move(tensors_to_save)
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

auto ConvTranspose1d::reset_parameters() -> void {
    int64_t fan_in = in_channels_ * kernel_size_;
    float std = std::sqrt(2.0f / fan_in);

    std::vector<int64_t> weight_shape = {in_channels_, out_channels_ / groups_, kernel_size_};
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

} // namespace tenzor::nn
