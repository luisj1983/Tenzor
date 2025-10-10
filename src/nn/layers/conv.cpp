#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/function.hpp"
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

// im2col transformation: unfold input tensor for convolution
// Input: [batch, in_channels, height, width]
// Output: [batch, in_channels * kernel_h * kernel_w, out_height * out_width]
auto im2col(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
            int64_t stride, int64_t padding, int64_t dilation) -> Tensor {
    auto shape = input.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t height = shape[2];
    int64_t width = shape[3];

    int64_t out_h = calculate_output_size(height, kernel_h, stride, padding, dilation);
    int64_t out_w = calculate_output_size(width, kernel_w, stride, padding, dilation);

    // Create output tensor on same device as input
    auto col = zeros({batch, channels * kernel_h * kernel_w, out_h * out_w}, input.dtype(), input.device());

    // For GPU tensors, transfer to CPU, process, then transfer back
    // TODO: Implement native GPU kernels for im2col operation
    Tensor input_cpu = (input.device().type == Device::Type::CUDA) ? input.to(Device::cpu()) : input;
    Tensor col_cpu = (col.device().type == Device::Type::CUDA) ? col.to(Device::cpu()) : col;

    // Access input data (now guaranteed to be on CPU)
    const float* input_data = input_cpu.data<float>();
    float* col_data = col_cpu.data<float>();

    // Perform im2col transformation
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t col_c = c * kernel_h * kernel_w + kh * kernel_w + kw;

                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        for (int64_t ow = 0; ow < out_w; ++ow) {
                            // Calculate input position with padding and dilation
                            int64_t ih = oh * stride - padding + kh * dilation;
                            int64_t iw = ow * stride - padding + kw * dilation;

                            int64_t col_idx = b * (channels * kernel_h * kernel_w * out_h * out_w) +
                                            col_c * (out_h * out_w) +
                                            oh * out_w + ow;

                            // Check bounds and apply padding
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int64_t input_idx = b * (channels * height * width) +
                                                   c * (height * width) +
                                                   ih * width + iw;
                                col_data[col_idx] = input_data[input_idx];
                            } else {
                                col_data[col_idx] = 0.0f;  // Padding with zeros
                            }
                        }
                    }
                }
            }
        }
    }

    // Transfer back to original device if needed
    return (input.device().type == Device::Type::CUDA) ? col_cpu.to(input.device()) : col_cpu;
}

// col2im transformation: reverse of im2col for gradient computation
// Input: [batch, in_channels * kernel_h * kernel_w, out_height * out_width]
// Output: [batch, in_channels, height, width]
auto col2im(const Tensor& col, int64_t channels, int64_t height, int64_t width,
            int64_t kernel_h, int64_t kernel_w, int64_t stride,
            int64_t padding, int64_t dilation) -> Tensor {
    auto col_shape = col.shape();
    int64_t batch = col_shape[0];
    int64_t out_h = calculate_output_size(height, kernel_h, stride, padding, dilation);
    int64_t out_w = calculate_output_size(width, kernel_w, stride, padding, dilation);

    // Create output tensor on same device as input
    auto output = zeros({batch, channels, height, width}, col.dtype(), col.device());

    // For GPU tensors, transfer to CPU, process, then transfer back
    // TODO: Implement native GPU kernels for col2im operation
    Tensor col_cpu = (col.device().type == Device::Type::CUDA) ? col.to(Device::cpu()) : col;
    Tensor output_cpu = (output.device().type == Device::Type::CUDA) ? output.to(Device::cpu()) : output;

    const float* col_data = col_cpu.data<float>();
    float* output_data = output_cpu.data<float>();

    // Perform col2im transformation (accumulate gradients)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t col_c = c * kernel_h * kernel_w + kh * kernel_w + kw;

                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        for (int64_t ow = 0; ow < out_w; ++ow) {
                            int64_t ih = oh * stride - padding + kh * dilation;
                            int64_t iw = ow * stride - padding + kw * dilation;

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

    // Transfer back to original device if needed
    return (col.device().type == Device::Type::CUDA) ? output_cpu.to(col.device()) : output_cpu;
}

} // anonymous namespace

// Conv2dBackward autograd function
class Conv2dBackward : public Function {
public:
    Conv2dBackward(int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                   std::vector<Tensor> tensors_to_save)
        : stride_(stride), padding_(padding), dilation_(dilation), groups_(groups) {
        // Save tensors in constructor (protected member access is allowed here)
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        // Not used - forward is handled by Conv2d::forward
        throw std::runtime_error("Conv2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // grad_outputs[0]: gradient w.r.t output [batch, out_channels, out_h, out_w]
        // saved_tensors_[0]: input [batch, in_channels, in_h, in_w]
        // saved_tensors_[1]: weight [out_channels, in_channels/groups, kernel_h, kernel_w]

        // Detect if we're on CUDA and transfer to CPU for computation
        Device original_device = grad_outputs[0].device();
        bool use_gpu = (original_device.type == Device::Type::CUDA);

        // Transfer all tensors to CPU for backward computation
        const Tensor grad_output = use_gpu ? grad_outputs[0].to(Device::cpu()) : grad_outputs[0];
        const Tensor input = use_gpu ? saved_tensors_[0].to(Device::cpu()) : saved_tensors_[0];
        const Tensor weight = use_gpu ? saved_tensors_[1].to(Device::cpu()) : saved_tensors_[1];

        auto weight_shape = weight.shape();
        int64_t out_channels = weight_shape[0];
        int64_t in_channels_per_group = weight_shape[1];
        int64_t kernel_h = weight_shape[2];
        int64_t kernel_w = weight_shape[3];
        int64_t in_channels = in_channels_per_group * groups_;

        auto input_shape = input.shape();
        int64_t batch = input_shape[0];
        int64_t height = input_shape[2];
        int64_t width = input_shape[3];

        auto grad_shape = grad_output.shape();
        int64_t out_h = grad_shape[2];
        int64_t out_w = grad_shape[3];

        // Gradient w.r.t input
        Tensor grad_input = zeros({batch, in_channels, height, width});

        int64_t out_channels_per_group = out_channels / groups_;

        // Process each group separately
        for (int64_t g = 0; g < groups_; ++g) {
            int64_t in_start = g * in_channels_per_group;
            int64_t out_start = g * out_channels_per_group;

            // Extract grad_output slice for this group [batch, out_channels_per_group, out_h, out_w]
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

            // Extract weight slice for this group [out_channels_per_group, in_channels_per_group, k_h, k_w]
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

            // Reshape for matmul
            auto grad_reshaped = grad_slice.reshape({batch, out_channels_per_group, out_h * out_w});
            auto weight_reshaped = weight_slice.reshape({out_channels_per_group, in_channels_per_group * kernel_h * kernel_w});

            // Compute gradient in col format for this group
            auto grad_col = zeros({batch, in_channels_per_group * kernel_h * kernel_w, out_h * out_w});
            float* grad_col_data = grad_col.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                // Manually extract 2D slice [out_channels_per_group, out_h * out_w] for batch b
                // since operator[] is not implemented
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

            // Convert col back to image format for this group
            auto grad_input_slice = col2im(grad_col, in_channels_per_group, height, width,
                                          kernel_h, kernel_w, stride_, padding_, dilation_);

            // Copy to appropriate position in grad_input
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

        // Gradient w.r.t weight
        Tensor grad_weight = zeros({out_channels, in_channels_per_group, kernel_h, kernel_w});
        float* grad_weight_data = grad_weight.data<float>();

        // Process each group separately
        for (int64_t g = 0; g < groups_; ++g) {
            int64_t in_start = g * in_channels_per_group;
            int64_t out_start = g * out_channels_per_group;

            // Extract input slice for this group [batch, in_channels_per_group, height, width]
            auto input_slice = zeros({batch, in_channels_per_group, height, width});
            const float* input_data = input.data<float>();
            float* input_slice_data = input_slice.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t h = 0; h < height; ++h) {
                        for (int64_t w = 0; w < width; ++w) {
                            int64_t src_idx = b * (in_channels * height * width) +
                                            (in_start + ic) * (height * width) +
                                            h * width + w;
                            int64_t dst_idx = b * (in_channels_per_group * height * width) +
                                            ic * (height * width) +
                                            h * width + w;
                            input_slice_data[dst_idx] = input_data[src_idx];
                        }
                    }
                }
            }

            // Apply im2col to input slice
            auto input_col = im2col(input_slice, kernel_h, kernel_w, stride_, padding_, dilation_);

            // Extract grad_output slice for this group [batch, out_channels_per_group, out_h, out_w]
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

            // Reshape for matmul
            auto grad_reshaped = grad_slice.reshape({batch, out_channels_per_group, out_h * out_w});
            auto input_col_reshaped = input_col.reshape({batch, in_channels_per_group * kernel_h * kernel_w, out_h * out_w});

            // Initialize gradient weight for this group
            auto grad_weight_group = zeros({out_channels_per_group, in_channels_per_group * kernel_h * kernel_w});
            float* grad_weight_group_data = grad_weight_group.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                // Manually extract 2D slices since operator[] is not implemented
                // grad_b: [out_channels_per_group, out_h * out_w]
                auto grad_b = zeros({out_channels_per_group, out_h * out_w});
                const float* grad_reshaped_data = grad_reshaped.data<float>();
                float* grad_b_data = grad_b.data<float>();

                int64_t grad_slice_size = out_channels_per_group * out_h * out_w;
                for (int64_t i = 0; i < grad_slice_size; ++i) {
                    grad_b_data[i] = grad_reshaped_data[b * grad_slice_size + i];
                }

                // input_col_b: [in_channels_per_group * kernel_h * kernel_w, out_h * out_w]
                auto input_col_b = zeros({in_channels_per_group * kernel_h * kernel_w, out_h * out_w});
                const float* input_col_reshaped_data = input_col_reshaped.data<float>();
                float* input_col_b_data = input_col_b.data<float>();

                int64_t input_slice_size = in_channels_per_group * kernel_h * kernel_w * out_h * out_w;
                for (int64_t i = 0; i < input_slice_size; ++i) {
                    input_col_b_data[i] = input_col_reshaped_data[b * input_slice_size + i];
                }

                auto input_col_b_t = input_col_b.transpose(0, 1).contiguous();
                auto grad_weight_b = matmul(grad_b, input_col_b_t);

                const float* src = grad_weight_b.data<float>();
                for (int64_t i = 0; i < out_channels_per_group * in_channels_per_group * kernel_h * kernel_w; ++i) {
                    grad_weight_group_data[i] += src[i];
                }
            }

            // Reshape and copy to appropriate position in grad_weight
            grad_weight_group = grad_weight_group.reshape({out_channels_per_group, in_channels_per_group, kernel_h, kernel_w});
            const float* grad_weight_group_data_src = grad_weight_group.data<float>();

            for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t src_idx = oc * (in_channels_per_group * kernel_h * kernel_w) +
                                            ic * (kernel_h * kernel_w) +
                                            kh * kernel_w + kw;
                            int64_t dst_idx = (out_start + oc) * (in_channels_per_group * kernel_h * kernel_w) +
                                            ic * (kernel_h * kernel_w) +
                                            kh * kernel_w + kw;
                            grad_weight_data[dst_idx] = grad_weight_group_data_src[src_idx];
                        }
                    }
                }
            }
        }

        // Gradient w.r.t bias (if exists)
        Tensor grad_bias;
        if (saved_tensors_.size() > 2) {  // bias exists
            // Sum over batch, height, width dimensions
            auto grad_reshaped = grad_output.reshape({batch, out_channels, out_h * out_w});
            grad_bias = zeros({out_channels});
            float* grad_bias_data = grad_bias.data<float>();
            const float* grad_data = grad_reshaped.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t c = 0; c < out_channels; ++c) {
                    for (int64_t i = 0; i < out_h * out_w; ++i) {
                        int64_t idx = b * (out_channels * out_h * out_w) + c * (out_h * out_w) + i;
                        grad_bias_data[c] += grad_data[idx];
                    }
                }
            }
        }

        // Transfer gradients back to original device if needed
        if (use_gpu) {
            grad_input = grad_input.to(original_device);
            grad_weight = grad_weight.to(original_device);
            if (saved_tensors_.size() > 2) {
                grad_bias = grad_bias.to(original_device);
            }
        }

        return saved_tensors_.size() > 2 ?
            std::vector<Tensor>{grad_input, grad_weight, grad_bias} :
            std::vector<Tensor>{grad_input, grad_weight};
    }

private:
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t groups_;
};

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
              int64_t stride, int64_t padding, int64_t dilation,
              int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride),
      padding_(padding), dilation_(dilation), groups_(groups) {

    // Validate parameters
    if (in_channels % groups != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }

    // Initialize weight with He/Kaiming initialization
    std::vector<int64_t> weight_shape = {out_channels, in_channels / groups, kernel_size, kernel_size};
    int64_t fan_in = (in_channels / groups) * kernel_size * kernel_size;
    float std_init = std::sqrt(2.0f / fan_in);
    auto weight_tensor = randn(weight_shape) * std_init;
    auto weight_init = Variable(weight_tensor, true);
    register_parameter("weight", weight_init);

    // Initialize bias with zeros
    if (bias) {
        std::vector<int64_t> bias_shape = {out_channels};
        auto bias_init = Variable(zeros(bias_shape), true);
        register_parameter("bias", bias_init);
    }
}

auto Conv2d::forward(const Variable& input) -> Variable {
    // Input shape: [batch, in_channels, height, width]
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

    // Calculate output dimensions
    int64_t out_h = calculate_output_size(height, kernel_size_, stride_, padding_, dilation_);
    int64_t out_w = calculate_output_size(width, kernel_size_, stride_, padding_, dilation_);

    // Get weight from parameters
    Variable& weight = parameters_["weight"];

    // Get weight shape information
    auto weight_shape = weight.tensor().shape();
    int64_t in_channels_per_group = weight_shape[1];
    int64_t out_channels_per_group = out_channels_ / groups_;

    // Create output tensor on same device as input
    auto output = zeros({batch, out_channels_, out_h, out_w}, input.tensor().dtype(), input.tensor().device());

    // For GPU execution, we need to work on CPU for now
    // TODO: Implement native GPU convolution kernels
    Device original_device = input.tensor().device();
    bool use_gpu = (original_device.type == Device::Type::CUDA);

    Tensor input_work = use_gpu ? input.tensor().to(Device::cpu()) : input.tensor();
    Tensor weight_work = use_gpu ? weight.tensor().to(Device::cpu()) : weight.tensor();
    Tensor output_work = use_gpu ? output.to(Device::cpu()) : output;

    // Process each group separately
    for (int64_t g = 0; g < groups_; ++g) {
        // Calculate channel ranges for this group
        int64_t in_start = g * in_channels_per_group;
        int64_t in_end = (g + 1) * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;
        int64_t out_end = (g + 1) * out_channels_per_group;

        // Extract input slice for this group [batch, in_channels_per_group, height, width]
        std::vector<int64_t> input_slice_shape = {batch, in_channels_per_group, height, width};
        auto input_slice = zeros(input_slice_shape);

        const float* input_data = input_work.data<float>();
        float* input_slice_data = input_slice.data<float>();

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < in_channels_per_group; ++c) {
                for (int64_t h = 0; h < height; ++h) {
                    for (int64_t w = 0; w < width; ++w) {
                        int64_t src_idx = b * (in_channels_ * height * width) +
                                         (in_start + c) * (height * width) +
                                         h * width + w;
                        int64_t dst_idx = b * (in_channels_per_group * height * width) +
                                         c * (height * width) +
                                         h * width + w;
                        input_slice_data[dst_idx] = input_data[src_idx];
                    }
                }
            }
        }

        // Apply im2col to this group's input
        auto input_col = im2col(input_slice, kernel_size_, kernel_size_,
                               stride_, padding_, dilation_);

        // Extract weight slice for this group [out_channels_per_group, in_channels_per_group, k_h, k_w]
        std::vector<int64_t> weight_slice_shape = {out_channels_per_group, in_channels_per_group,
                                                    kernel_size_, kernel_size_};
        auto weight_slice = zeros(weight_slice_shape);

        const float* weight_data = weight_work.data<float>();
        float* weight_slice_data = weight_slice.data<float>();

        for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
            for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                for (int64_t kh = 0; kh < kernel_size_; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size_; ++kw) {
                        int64_t src_idx = (out_start + oc) * (in_channels_per_group * kernel_size_ * kernel_size_) +
                                         ic * (kernel_size_ * kernel_size_) +
                                         kh * kernel_size_ + kw;
                        int64_t dst_idx = oc * (in_channels_per_group * kernel_size_ * kernel_size_) +
                                         ic * (kernel_size_ * kernel_size_) +
                                         kh * kernel_size_ + kw;
                        weight_slice_data[dst_idx] = weight_data[src_idx];
                    }
                }
            }
        }

        // Reshape weight for matmul: [out_channels_per_group, in_channels_per_group * k_h * k_w]
        int64_t kernel_size_flat = in_channels_per_group * kernel_size_ * kernel_size_;
        auto weight_reshaped = weight_slice.reshape({out_channels_per_group, kernel_size_flat});

        // Perform matrix multiplication for this group
        // input_col: [batch, kernel_size_flat, out_h * out_w]
        // weight: [out_channels_per_group, kernel_size_flat]

        int64_t spatial_size = out_h * out_w;

        // Process each batch
        for (int64_t b = 0; b < batch; ++b) {
            // Extract input_col for this batch: [kernel_size_flat, spatial_size]
            auto input_col_b = zeros({kernel_size_flat, spatial_size});
            const float* input_col_data = input_col.data<float>();
            float* input_col_b_data = input_col_b.data<float>();

            for (int64_t i = 0; i < kernel_size_flat * spatial_size; ++i) {
                input_col_b_data[i] = input_col_data[b * kernel_size_flat * spatial_size + i];
            }

            // Matmul: weight @ input_col_b = [out_channels_per_group, kernel_size_flat] @ [kernel_size_flat, spatial_size]
            // -> [out_channels_per_group, spatial_size]
            auto output_group = matmul(weight_reshaped, input_col_b);

            // Copy to output tensor at appropriate position
            const float* output_group_data = output_group.data<float>();
            float* output_data = output_work.data<float>();

            for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    int64_t h_idx = s / out_w;
                    int64_t w_idx = s % out_w;
                    int64_t dst_idx = b * (out_channels_ * out_h * out_w) +
                                     (out_start + oc) * (out_h * out_w) +
                                     h_idx * out_w + w_idx;
                    int64_t src_idx = oc * spatial_size + s;
                    output_data[dst_idx] = output_group_data[src_idx];
                }
            }
        }
    }

    // Add bias if present
    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        Variable& bias = bias_it->second;

        // Get bias on CPU if we're using GPU
        Tensor bias_work = use_gpu ? bias.tensor().to(Device::cpu()) : bias.tensor();

        // Manual broadcasting for bias
        float* out_data = output_work.data<float>();
        const float* bias_data = bias_work.data<float>();

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < out_channels_; ++c) {
                for (int64_t h = 0; h < out_h; ++h) {
                    for (int64_t w = 0; w < out_w; ++w) {
                        int64_t idx = b * (out_channels_ * out_h * out_w) +
                                     c * (out_h * out_w) +
                                     h * out_w + w;
                        out_data[idx] += bias_data[c];
                    }
                }
            }
        }
    }

    // Transfer back to GPU if needed
    if (use_gpu) {
        output = output_work.to(original_device);
    } else {
        output = output_work;
    }

    // Create output variable with autograd support
    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    // Setup backward function if gradient is required
    if (input.requires_grad() || weight.requires_grad()) {
        // Prepare tensors to save
        std::vector<Tensor> tensors_to_save;
        if (bias_it != parameters_.end()) {
            Variable& bias = bias_it->second;
            tensors_to_save = {input.tensor(), weight.tensor(), bias.tensor()};
        } else {
            tensors_to_save = {input.tensor(), weight.tensor()};
        }

        // Create backward function with saved tensors
        auto backward_fn = std::make_shared<Conv2dBackward>(
            stride_, padding_, dilation_, groups_, std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        // Track input variables for gradient accumulation
        std::vector<Variable*> input_vars;
        if (input.requires_grad()) {
            input_vars.push_back(const_cast<Variable*>(&input));
        }
        if (weight.requires_grad()) {
            input_vars.push_back(&weight);
        }
        if (bias_it != parameters_.end() && bias_it->second.requires_grad()) {
            input_vars.push_back(&(bias_it->second));
        }
        backward_fn->set_input_variables(input_vars);

        // CRITICAL FIX: Connect to input's grad_fn to continue the backward chain
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

auto Conv2d::reset_parameters() -> void {
    // Kaiming/He initialization for ReLU activations
    // std = sqrt(2 / (in_channels * kernel_h * kernel_w))
    int64_t fan_in = in_channels_ / groups_ * kernel_size_ * kernel_size_;
    float std = std::sqrt(2.0f / fan_in);

    // Reinitialize weight by creating a new Variable
    std::vector<int64_t> weight_shape = {out_channels_, in_channels_ / groups_, kernel_size_, kernel_size_};
    auto new_weight_tensor = randn(weight_shape) * std;
    parameters_["weight"] = Variable(new_weight_tensor, true);

    // Initialize bias with zeros
    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        std::vector<int64_t> bias_shape = {out_channels_};
        bias_it->second = Variable(zeros(bias_shape), true);
    }
}

} // namespace tenzor::nn
