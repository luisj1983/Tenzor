#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/backend/dispatch.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

// Include CUDA kernel headers when available
#ifdef TENZOR_HAS_CUDA
#include "tenzor/backends/cuda/conv_kernels.hpp"
#endif

#ifdef TENZOR_HAS_CUDNN
#include "tenzor/backend/cudnn_wrapper.hpp"
#endif

namespace tenzor::nn {

// Helper namespace for convolution operations
namespace {

// Calculate output size for convolution
auto calculate_output_size(int64_t input_size, int64_t kernel_size,
                           int64_t stride, int64_t padding, int64_t dilation) -> int64_t {
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
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

    auto col = zeros({batch, channels * kernel_h * kernel_w, out_h * out_w}, input.dtype(), input.device());

    const float* input_data = input.data<float>();
    float* col_data = col.data<float>();

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

    return col;
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

    auto output = zeros({batch, channels, height, width}, col.dtype(), col.device());

    const float* col_data = col.data<float>();
    float* output_data = output.data<float>();

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

    return output;
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
        Device original_device = grad_outputs[0].device();

        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];

        auto weight_shape = weight.shape();
        int64_t out_channels = weight_shape[0];
        int64_t in_channels_per_group = weight_shape[1];
        int64_t kernel_h = weight_shape[2];
        int64_t kernel_w = weight_shape[3];

        auto input_shape = input.shape();
        int64_t batch = input_shape[0];
        int64_t in_channels = in_channels_per_group * groups_;
        int64_t height = input_shape[2];
        int64_t width = input_shape[3];

        // Use backend dispatcher for gradient computation
        #ifdef TENZOR_HAS_CUDA
        if (original_device.type == Device::Type::CUDA) {
            // Use CUDA kernels directly
            #ifdef TENZOR_HAS_CUDNN
            // Try cuDNN first for better performance
            try {
                auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv2d_backward(
                    grad_output, input, weight,
                    stride_, padding_, dilation_, groups_,
                    true, true, saved_tensors_.size() > 2,
                    nullptr
                );

                if (saved_tensors_.size() > 2) {
                    return {grad_input, grad_weight, grad_bias};
                } else {
                    return {grad_input, grad_weight};
                }
            } catch (const std::exception& e) {
                // Fall back to custom CUDA kernels
                auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
                    grad_output, input, weight,
                    stride_, padding_, dilation_, groups_,
                    true, true, saved_tensors_.size() > 2,
                    nullptr
                );

                if (saved_tensors_.size() > 2) {
                    return {grad_input, grad_weight, grad_bias};
                } else {
                    return {grad_input, grad_weight};
                }
            }
            #else
            // Use custom CUDA kernels
            auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
                grad_output, input, weight,
                stride_, padding_, dilation_, groups_,
                true, true, saved_tensors_.size() > 2,
                nullptr
            );

            if (saved_tensors_.size() > 2) {
                return {grad_input, grad_weight, grad_bias};
            } else {
                return {grad_input, grad_weight};
            }
            #endif
        }
        #endif

        // Use CPU backend through operation registry for proper dtype support
        std::vector<Tensor> tensors_for_dispatch = {grad_output};
        auto* backend = Dispatcher::get_backend(tensors_for_dispatch);

        // Prepare input shape for backward computation
        std::string input_shape_str = std::to_string(batch) + "," + std::to_string(in_channels) + "," +
                                      std::to_string(height) + "," + std::to_string(width);
        std::string weight_shape_str = std::to_string(out_channels) + "," + std::to_string(in_channels_per_group) + "," +
                                       std::to_string(kernel_h) + "," + std::to_string(kernel_w);

        // Compute gradients using backend operations
        std::vector<Tensor> grad_input_inputs = {grad_output, weight};
        OpAttributes grad_input_attrs = {
            {"input_shape", input_shape_str},
            {"stride", std::to_string(stride_)},
            {"padding", std::to_string(padding_)},
            {"dilation", std::to_string(dilation_)},
            {"groups", std::to_string(groups_)}
        };
        auto grad_input_result = backend->dispatch(
            "conv2d_backward_input",
            grad_input_inputs,
            grad_input_attrs
        );
        Tensor grad_input = grad_input_result[0];

        std::vector<Tensor> grad_weight_inputs = {grad_output, input};
        OpAttributes grad_weight_attrs = {
            {"weight_shape", weight_shape_str},
            {"stride", std::to_string(stride_)},
            {"padding", std::to_string(padding_)},
            {"dilation", std::to_string(dilation_)},
            {"groups", std::to_string(groups_)}
        };
        auto grad_weight_result = backend->dispatch(
            "conv2d_backward_weight",
            grad_weight_inputs,
            grad_weight_attrs
        );
        Tensor grad_weight = grad_weight_result[0];

        // Gradient w.r.t bias
        if (saved_tensors_.size() > 2) {
            std::vector<Tensor> grad_bias_inputs = {grad_output};
            OpAttributes empty_attrs{};
            auto grad_bias_result = backend->dispatch(
                "conv2d_backward_bias",
                grad_bias_inputs,
                empty_attrs
            );
            Tensor grad_bias = grad_bias_result[0];
            return {grad_input, grad_weight, grad_bias};
        }

        return {grad_input, grad_weight};

        /*
        // Old CPU fallback implementation - replaced with backend call above
        Tensor grad_input = zeros({batch, in_channels, height, width}, input.dtype());
        int64_t out_channels_per_group = out_channels / groups_;

        auto grad_shape = grad_output.shape();
        int64_t out_h = grad_shape[2];
        int64_t out_w = grad_shape[3];

        for (int64_t g = 0; g < groups_; ++g) {
            int64_t in_start = g * in_channels_per_group;
            int64_t out_start = g * out_channels_per_group;

            auto grad_slice = zeros({batch, out_channels_per_group, out_h, out_w});
            const float* grad_data = grad_output.data<float>();
            float* grad_slice_data = grad_slice.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                    for (int64_t h = 0; h < out_h; ++h) {
                        for (int64_t w = 0; w < out_w; ++w) {
                            int64_t src_idx = b * (out_channels * out_h * out_w) +
                                            (out_start + oc) * (out_h * out_w) +
                                            h * out_w + w;
                            int64_t dst_idx = b * (out_channels_per_group * out_h * out_w) +
                                            oc * (out_h * out_w) +
                                            h * out_w + w;
                            grad_slice_data[dst_idx] = grad_data[src_idx];
                        }
                    }
                }
            }

            auto weight_slice = zeros({out_channels_per_group, in_channels_per_group, kernel_h, kernel_w});
            const float* weight_data = weight.data<float>();
            float* weight_slice_data = weight_slice.data<float>();

            for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t src_idx = (out_start + oc) * (in_channels_per_group * kernel_h * kernel_w) +
                                            ic * (kernel_h * kernel_w) +
                                            kh * kernel_w + kw;
                            int64_t dst_idx = oc * (in_channels_per_group * kernel_h * kernel_w) +
                                            ic * (kernel_h * kernel_w) +
                                            kh * kernel_w + kw;
                            weight_slice_data[dst_idx] = weight_data[src_idx];
                        }
                    }
                }
            }

            auto grad_reshaped = grad_slice.reshape({batch, out_channels_per_group, out_h * out_w});
            auto weight_reshaped = weight_slice.reshape({out_channels_per_group, in_channels_per_group * kernel_h * kernel_w});

            auto grad_col = zeros({batch, in_channels_per_group * kernel_h * kernel_w, out_h * out_w});
            float* grad_col_data = grad_col.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                auto grad_b = zeros({out_channels_per_group, out_h * out_w});
                const float* grad_reshaped_data = grad_reshaped.data<float>();
                float* grad_b_data = grad_b.data<float>();

                int64_t slice_size = out_channels_per_group * out_h * out_w;
                for (int64_t i = 0; i < slice_size; ++i) {
                    grad_b_data[i] = grad_reshaped_data[b * slice_size + i];
                }

                auto weight_t = weight_reshaped.transpose(0, 1).contiguous();
                auto grad_col_b = matmul(weight_t, grad_b);

                const float* src = grad_col_b.data<float>();
                float* dst = grad_col_data + b * in_channels_per_group * kernel_h * kernel_w * out_h * out_w;
                std::copy_n(src, in_channels_per_group * kernel_h * kernel_w * out_h * out_w, dst);
            }

            auto grad_input_slice = col2im_cpu(grad_col, in_channels_per_group, height, width,
                                               kernel_h, kernel_w, stride_, padding_, dilation_);

            const float* grad_input_slice_data = grad_input_slice.data<float>();
            float* grad_input_data = grad_input.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t h = 0; h < height; ++h) {
                        for (int64_t w = 0; w < width; ++w) {
                            int64_t src_idx = b * (in_channels_per_group * height * width) +
                                            ic * (height * width) +
                                            h * width + w;
                            int64_t dst_idx = b * (in_channels * height * width) +
                                            (in_start + ic) * (height * width) +
                                            h * width + w;
                            grad_input_data[dst_idx] += grad_input_slice_data[src_idx];
                        }
                    }
                }
            }
        }
        */
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
        throw std::invalid_argument(
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

    #ifdef TENZOR_HAS_CUDA
    if (original_device.type == Device::Type::CUDA) {
        #ifdef TENZOR_HAS_CUDNN
        // Try cuDNN first for optimal performance
        try {
            output = cuda::cudnn_conv2d_forward(
                input.tensor(), weight_matched.tensor(), bias_ptr,
                stride_, padding_, dilation_, groups_,
                nullptr
            );
        } catch (const std::exception& e) {
            // Fall back to custom CUDA kernels
            output = cuda::conv2d_forward_kernel(
                input.tensor(), weight_matched.tensor(), bias_ptr,
                stride_, padding_, dilation_, groups_,
                nullptr
            );
        }
        #else
        // Use custom CUDA kernels
        output = cuda::conv2d_forward_kernel(
            input.tensor(), weight_matched.tensor(), bias_ptr,
            stride_, padding_, dilation_, groups_,
            nullptr
        );
        #endif
    } else
    #endif
    {
        // Use CPU backend through operation registry for proper dtype support
        std::vector<Tensor> tensors_for_dispatch = {input.tensor()};
        auto* backend = Dispatcher::get_backend(tensors_for_dispatch);

        // Compute output using backend operation
        // Pass bias as third input if present
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
    }

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
        // Implement Conv1d backward (similar to Conv2d but for 1D)
        // This would involve adding a dimension to use Conv2d operations
        // or implementing specific 1D im2col/col2im operations

        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];

        // Add height dimension of 1 to convert to 2D
        auto grad_2d = grad_output.unsqueeze(2);
        auto input_2d = input.unsqueeze(2);
        auto weight_2d = weight.unsqueeze(2);

        // Use Conv2d backward
        // (Implementation continues with similar pattern to Conv2d)

        throw std::runtime_error("Conv1d backward not yet fully implemented");
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
        auto bias_tensor = (randn(bias_shape) * 2.0f * bound) - bound;
        auto bias_var = Variable(bias_tensor, true);
        register_parameter("bias", bias_var);
    }
}

auto Conv1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("Conv1d expects 3D input [batch, channels, length]");
    }

    // Add height dimension and use Conv2d
    auto input_4d = input.tensor().unsqueeze(2);
    auto& weight = *parameters_["weight"];
    auto weight_4d = weight.tensor().unsqueeze(2);

    // Use Conv2d forward with adapted parameters
    // (Implementation continues with Conv2d-based approach)

    throw std::runtime_error("Conv1d forward not yet fully implemented");
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
        // ConvTranspose backward is essentially a regular convolution
        // (Implementation would mirror Conv2d forward with role reversal)

        throw std::runtime_error("ConvTranspose2d backward not yet fully implemented");
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
    weight_ = Variable(weight_tensor, true);
    register_parameter("weight", weight_);

    if (bias) {
        std::vector<int64_t> bias_shape = {out_channels};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto bias_tensor = (randn(bias_shape) * 2.0f * bound) - bound;
        bias_ = Variable(bias_tensor, true);
        register_parameter("bias", *bias_);
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

    // TODO: Implement backward autograd function for ConvTranspose2d
    // For now, we support forward pass only

    return result;
}

auto ConvTranspose2d::reset_parameters() -> void {
    int64_t fan_in = in_channels_ * kernel_size_ * kernel_size_;
    float std = std::sqrt(2.0f / fan_in);

    std::vector<int64_t> weight_shape = {in_channels_, out_channels_ / groups_, kernel_size_, kernel_size_};
    auto new_weight_tensor = randn(weight_shape) * std;
    weight_ = Variable(new_weight_tensor, true);
    parameters_["weight"] = std::make_shared<Variable>(weight_);

    if (bias_.has_value()) {
        std::vector<int64_t> bias_shape = {out_channels_};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto new_bias_tensor = (randn(bias_shape) * 2.0f * bound) - bound;
        bias_ = Variable(new_bias_tensor, true);
        parameters_["bias"] = std::make_shared<Variable>(*bias_);
    }
}

} // namespace tenzor::nn
