#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/backend/dispatch.hpp"
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

// im2col transformation: unfold input tensor for convolution (with separate padding for height/width)
// Input: [batch, in_channels, height, width]
// Output: [batch, in_channels * kernel_h * kernel_w, out_height * out_width]
auto im2col(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
            int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w,
            int64_t dilation) -> Tensor {
    auto shape = input.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t height = shape[2];
    int64_t width = shape[3];

    int64_t out_h = calculate_output_size(height, kernel_h, stride_h, padding_h, dilation);
    int64_t out_w = calculate_output_size(width, kernel_w, stride_w, padding_w, dilation);

    // Create output tensor on same device as input
    auto col = zeros({batch, channels * kernel_h * kernel_w, out_h * out_w}, input.dtype(), input.device());

    // GPU kernels are implemented in conv2d.cu and accessed via backend dispatcher
    // For CPU execution, process directly
    // Backend-agnostic: check for CPU, not against CUDA (supports OneAPI/Vulkan/ROCm)
    const bool is_cpu = (input.device().type == Device::Type::CPU);
    Tensor input_cpu = is_cpu ? input : input.to(Device::cpu());
    Tensor col_cpu = is_cpu ? col : col.to(Device::cpu());

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
                            int64_t ih = oh * stride_h - padding_h + kh * dilation;
                            int64_t iw = ow * stride_w - padding_w + kw * dilation;

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

    // Transfer back to original device if needed (GPU kernels handle this automatically via dispatcher)
    return is_cpu ? col_cpu : col_cpu.to(input.device());
}

// im2col transformation: unfold input tensor for convolution (same padding for both dimensions)
// Input: [batch, in_channels, height, width]
// Output: [batch, in_channels * kernel_h * kernel_w, out_height * out_width]
auto im2col(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
            int64_t stride, int64_t padding, int64_t dilation) -> Tensor {
    return im2col(input, kernel_h, kernel_w, stride, stride, padding, padding, dilation);
}

// col2im transformation: reverse of im2col for gradient computation (with separate padding/stride for height/width)
// Input: [batch, in_channels * kernel_h * kernel_w, out_height * out_width]
// Output: [batch, in_channels, height, width]
auto col2im(const Tensor& col, int64_t channels, int64_t height, int64_t width,
            int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
            int64_t padding_h, int64_t padding_w, int64_t dilation) -> Tensor {
    auto col_shape = col.shape();
    int64_t batch = col_shape[0];
    int64_t out_h = calculate_output_size(height, kernel_h, stride_h, padding_h, dilation);
    int64_t out_w = calculate_output_size(width, kernel_w, stride_w, padding_w, dilation);

    // Create output tensor on same device as input
    auto output = zeros({batch, channels, height, width}, col.dtype(), col.device());

    // GPU kernels are implemented in conv2d.cu and accessed via backend dispatcher
    // For CPU execution, process directly
    // Backend-agnostic: check for CPU, not against CUDA (supports OneAPI/Vulkan/ROCm)
    const bool is_cpu = (col.device().type == Device::Type::CPU);
    Tensor col_cpu = is_cpu ? col : col.to(Device::cpu());
    Tensor output_cpu = is_cpu ? output : output.to(Device::cpu());

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

    // Transfer back to original device if needed (GPU kernels handle this automatically via dispatcher)
    return is_cpu ? output_cpu : output_cpu.to(col.device());
}

// col2im transformation: reverse of im2col for gradient computation (same padding/stride for both dimensions)
// Input: [batch, in_channels * kernel_h * kernel_w, out_height * out_width]
// Output: [batch, in_channels, height, width]
auto col2im(const Tensor& col, int64_t channels, int64_t height, int64_t width,
            int64_t kernel_h, int64_t kernel_w, int64_t stride,
            int64_t padding, int64_t dilation) -> Tensor {
    return col2im(col, channels, height, width, kernel_h, kernel_w,
                  stride, stride, padding, padding, dilation);
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

        // Backend-agnostic: detect non-CPU devices and transfer to CPU for computation
        Device original_device = grad_outputs[0].device();
        bool need_cpu_transfer = (original_device.type != Device::Type::CPU);

        // Transfer all tensors to CPU for backward computation
        const Tensor grad_output = need_cpu_transfer ? grad_outputs[0].to(Device::cpu()) : grad_outputs[0];
        const Tensor input = need_cpu_transfer ? saved_tensors_[0].to(Device::cpu()) : saved_tensors_[0];
        const Tensor weight = need_cpu_transfer ? saved_tensors_[1].to(Device::cpu()) : saved_tensors_[1];

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
        Tensor grad_input = zeros({batch, in_channels, height, width}, DType::Float32, Device::cpu());

        int64_t out_channels_per_group = out_channels / groups_;

        // Process each group separately
        for (int64_t g = 0; g < groups_; ++g) {
            int64_t in_start = g * in_channels_per_group;
            int64_t out_start = g * out_channels_per_group;

            // Extract grad_output slice for this group [batch, out_channels_per_group, out_h, out_w]
            auto grad_slice = zeros({batch, out_channels_per_group, out_h, out_w}, DType::Float32, Device::cpu());
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
            auto weight_slice = zeros({out_channels_per_group, in_channels_per_group, kernel_h, kernel_w}, DType::Float32, Device::cpu());
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
            auto grad_col = zeros({batch, in_channels_per_group * kernel_h * kernel_w, out_h * out_w}, DType::Float32, Device::cpu());
            float* grad_col_data = grad_col.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                // Manually extract 2D slice [out_channels_per_group, out_h * out_w] for batch b
                // since operator[] is not implemented
                auto grad_b = zeros({out_channels_per_group, out_h * out_w}, DType::Float32, Device::cpu());
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
        Tensor grad_weight = zeros({out_channels, in_channels_per_group, kernel_h, kernel_w}, DType::Float32, Device::cpu());
        float* grad_weight_data = grad_weight.data<float>();

        // Process each group separately
        for (int64_t g = 0; g < groups_; ++g) {
            int64_t in_start = g * in_channels_per_group;
            int64_t out_start = g * out_channels_per_group;

            // Extract input slice for this group [batch, in_channels_per_group, height, width]
            auto input_slice = zeros({batch, in_channels_per_group, height, width}, DType::Float32, Device::cpu());
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
            auto grad_slice = zeros({batch, out_channels_per_group, out_h, out_w}, DType::Float32, Device::cpu());
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
            auto grad_weight_group = zeros({out_channels_per_group, in_channels_per_group * kernel_h * kernel_w}, DType::Float32, Device::cpu());
            float* grad_weight_group_data = grad_weight_group.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                // Manually extract 2D slices since operator[] is not implemented
                // grad_b: [out_channels_per_group, out_h * out_w]
                auto grad_b = zeros({out_channels_per_group, out_h * out_w}, DType::Float32, Device::cpu());
                const float* grad_reshaped_data = grad_reshaped.data<float>();
                float* grad_b_data = grad_b.data<float>();

                int64_t grad_slice_size = out_channels_per_group * out_h * out_w;
                for (int64_t i = 0; i < grad_slice_size; ++i) {
                    grad_b_data[i] = grad_reshaped_data[b * grad_slice_size + i];
                }

                // input_col_b: [in_channels_per_group * kernel_h * kernel_w, out_h * out_w]
                auto input_col_b = zeros({in_channels_per_group * kernel_h * kernel_w, out_h * out_w}, DType::Float32, Device::cpu());
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
            grad_bias = zeros({out_channels}, DType::Float32, Device::cpu());
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
        if (need_cpu_transfer) {
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
    // std::cerr << "[DEBUG Conv2d::forward] Entry!" << std::endl;
    // Input shape: [batch, in_channels, height, width]
    // std::cerr << "[DEBUG Conv2d::forward] Getting input shape..." << std::endl;
    auto input_shape = input.shape();
    // std::cerr << "[DEBUG Conv2d::forward] Got input shape: [" << input_shape.size() << " dims]" << std::endl;

    // std::cerr << "[DEBUG] Checking if size == 4..." << std::endl;
    if (input_shape.size() != 4) {
        throw std::invalid_argument("Conv2d expects 4D input [batch, channels, height, width]");
    }
    // std::cerr << "[DEBUG] Size check passed" << std::endl;

    // std::cerr << "[DEBUG] Getting batch from input_shape[0]..." << std::endl;
    int64_t batch = input_shape[0];
    // std::cerr << "[DEBUG] batch = " << batch << std::endl;

    // std::cerr << "[DEBUG] Getting in_channels from input_shape[1]..." << std::endl;
    int64_t in_channels = input_shape[1];
    // std::cerr << "[DEBUG] in_channels = " << in_channels << std::endl;

    // std::cerr << "[DEBUG] Getting height from input_shape[2]..." << std::endl;
    int64_t height = input_shape[2];
    // std::cerr << "[DEBUG] height = " << height << std::endl;

    // std::cerr << "[DEBUG] Getting width from input_shape[3]..." << std::endl;
    int64_t width = input_shape[3];
    // std::cerr << "[DEBUG] width = " << width << std::endl;

    // std::cerr << "[DEBUG] Checking channel mismatch..." << std::endl;
    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }
    // std::cerr << "[DEBUG] Channel check passed" << std::endl;

    // Calculate output dimensions
    // std::cerr << "[DEBUG] Calculating output dimensions..." << std::endl;
    int64_t out_h = calculate_output_size(height, kernel_size_, stride_, padding_, dilation_);
    // std::cerr << "[DEBUG] out_h = " << out_h << std::endl;
    int64_t out_w = calculate_output_size(width, kernel_size_, stride_, padding_, dilation_);
    // std::cerr << "[DEBUG] out_w = " << out_w << std::endl;

    // Validate output dimensions to prevent memory allocation errors
    // std::cerr << "[DEBUG] Validating output dimensions..." << std::endl;
    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument(
            "Invalid Conv2d configuration: output dimensions are non-positive (out_h=" +
            std::to_string(out_h) + ", out_w=" + std::to_string(out_w) + "). " +
            "Input size=" + std::to_string(height) + "x" + std::to_string(width) +
            ", kernel_size=" + std::to_string(kernel_size_) +
            ", stride=" + std::to_string(stride_) +
            ", padding=" + std::to_string(padding_) +
            ", dilation=" + std::to_string(dilation_) +
            ". Try reducing kernel_size, dilation, or increasing input size/padding."
        );
    }
    // std::cerr << "[DEBUG] Output dimensions valid" << std::endl;

    // Get weight from parameters
    // std::cerr << "[DEBUG] Getting weight from parameters..." << std::endl;
    auto& weight = *parameters_["weight"];
    // std::cerr << "[DEBUG] Weight retrieved successfully" << std::endl;

    // Get weight shape information
    // std::cerr << "[DEBUG] Getting weight shape..." << std::endl;
    auto weight_shape = weight.tensor().shape();
    // std::cerr << "[DEBUG] weight_shape size: " << weight_shape.size() << std::endl;
    int64_t in_channels_per_group = weight_shape[1];
    int64_t out_channels_per_group = out_channels_ / groups_;
    // std::cerr << "[DEBUG] in_channels_per_group = " << in_channels_per_group << ", out_channels_per_group = " << out_channels_per_group << std::endl;

    // Dispatch convolution via OperationRegistry
    // This automatically routes to the correct backend (CPU, CUDA, ROCm, OneAPI, Vulkan)
    OpAttributes attrs;
    attrs["stride"] = std::to_string(stride_);
    attrs["padding"] = std::to_string(padding_);
    attrs["dilation"] = std::to_string(dilation_);
    attrs["groups"] = std::to_string(groups_);

    // Prepare input tensors for dispatcher
    // Ensure weight and bias are on same device as input
    Device input_device = input.device();
    Tensor weight_tensor = weight.tensor();
    if (weight_tensor.device() != input_device) {
        weight_tensor = weight_tensor.to(input_device);
    }

    std::vector<Tensor> conv_inputs = {input.tensor(), weight_tensor};
    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        Tensor bias_tensor = bias_it->second->tensor();
        if (bias_tensor.device() != input_device) {
            bias_tensor = bias_tensor.to(input_device);
        }
        conv_inputs.push_back(bias_tensor);
    }

    // Dispatch to appropriate backend via OperationRegistry
    auto outputs = Dispatcher::dispatch("conv2d_forward", conv_inputs, attrs);
    auto output = outputs[0];

    // Create output variable with autograd support
    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    // Setup backward function if gradient is required
    if (input.requires_grad() || weight.requires_grad()) {
        // Prepare tensors to save
        auto bias_it = parameters_.find("bias");
        std::vector<Tensor> tensors_to_save;
        if (bias_it != parameters_.end()) {
            auto& bias = *bias_it->second;
            tensors_to_save = {input.tensor(), weight.tensor(), bias.tensor()};
        } else {
            tensors_to_save = {input.tensor(), weight.tensor()};
        }

        // Create backward function with saved tensors
        auto backward_fn = std::make_shared<Conv2dBackward>(
            stride_, padding_, dilation_, groups_, std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        // Track input variables for gradient accumulation using pointers
        std::vector<Variable> input_vars;
        if (input.requires_grad()) {
            input_vars.push_back(input);
        }
        if (weight.requires_grad()) {
            input_vars.push_back(*parameters_["weight"]);
        }
        if (bias_it != parameters_.end() && bias_it->second->requires_grad()) {
            input_vars.push_back(*bias_it->second);
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
    parameters_["weight"] = std::make_shared<Variable>(new_weight_tensor, true);

    // Initialize bias with zeros
    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        std::vector<int64_t> bias_shape = {out_channels_};
        bias_it->second = std::make_shared<Variable>(zeros(bias_shape), true);
    }
}

// ============================================================================
// Conv1d Implementation
// ============================================================================

// Conv1dBackward autograd function
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
        // grad_outputs[0]: gradient w.r.t output [batch, out_channels, length_out]
        // saved_tensors_[0]: input [batch, in_channels, length]
        // saved_tensors_[1]: weight [out_channels, in_channels/groups, kernel_size]

        Device original_device = grad_outputs[0].device();
        bool need_cpu_transfer = (original_device.type != Device::Type::CPU);

        const Tensor grad_output = need_cpu_transfer ? grad_outputs[0].to(Device::cpu()) : grad_outputs[0];
        const Tensor input = need_cpu_transfer ? saved_tensors_[0].to(Device::cpu()) : saved_tensors_[0];
        const Tensor weight = need_cpu_transfer ? saved_tensors_[1].to(Device::cpu()) : saved_tensors_[1];

        auto weight_shape = weight.shape();
        int64_t out_channels = weight_shape[0];
        int64_t in_channels_per_group = weight_shape[1];
        int64_t kernel_size = weight_shape[2];
        int64_t in_channels = in_channels_per_group * groups_;

        auto input_shape = input.shape();
        int64_t batch = input_shape[0];
        int64_t length = input_shape[2];

        auto grad_shape = grad_output.shape();
        int64_t length_out = grad_shape[2];

        // Gradient w.r.t input
        Tensor grad_input = zeros({batch, in_channels, length}, DType::Float32, Device::cpu());
        int64_t out_channels_per_group = out_channels / groups_;

        // Process each group
        for (int64_t g = 0; g < groups_; ++g) {
            int64_t in_start = g * in_channels_per_group;
            int64_t out_start = g * out_channels_per_group;

            // Extract grad_output slice for this group
            auto grad_slice = zeros({batch, out_channels_per_group, length_out}, DType::Float32, Device::cpu());
            const float* grad_data = grad_output.data<float>();
            float* grad_slice_data = grad_slice.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                    for (int64_t l = 0; l < length_out; ++l) {
                        int64_t src_idx = b * (out_channels * length_out) +
                                        (out_start + oc) * length_out + l;
                        int64_t dst_idx = b * (out_channels_per_group * length_out) +
                                        oc * length_out + l;
                        grad_slice_data[dst_idx] = grad_data[src_idx];
                    }
                }
            }

            // Extract weight slice for this group
            auto weight_slice = zeros({out_channels_per_group, in_channels_per_group, kernel_size}, DType::Float32, Device::cpu());
            const float* weight_data = weight.data<float>();
            float* weight_slice_data = weight_slice.data<float>();

            for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t k = 0; k < kernel_size; ++k) {
                        int64_t src_idx = (out_start + oc) * (in_channels_per_group * kernel_size) +
                                        ic * kernel_size + k;
                        int64_t dst_idx = oc * (in_channels_per_group * kernel_size) +
                                        ic * kernel_size + k;
                        weight_slice_data[dst_idx] = weight_data[src_idx];
                    }
                }
            }

            // Reshape for matmul
            auto grad_reshaped = grad_slice.reshape({batch, out_channels_per_group, length_out});
            auto weight_reshaped = weight_slice.reshape({out_channels_per_group, in_channels_per_group * kernel_size});

            // Compute gradient in col format
            auto grad_col = zeros({batch, in_channels_per_group * kernel_size, length_out}, DType::Float32, Device::cpu());
            float* grad_col_data = grad_col.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                auto grad_b = zeros({out_channels_per_group, length_out}, DType::Float32, Device::cpu());
                const float* grad_reshaped_data = grad_reshaped.data<float>();
                float* grad_b_data = grad_b.data<float>();

                int64_t slice_size = out_channels_per_group * length_out;
                for (int64_t i = 0; i < slice_size; ++i) {
                    grad_b_data[i] = grad_reshaped_data[b * slice_size + i];
                }

                auto weight_t = weight_reshaped.transpose(0, 1).contiguous();
                auto grad_col_b = matmul(weight_t, grad_b);

                const float* src = grad_col_b.data<float>();
                float* dst = grad_col_data + b * in_channels_per_group * kernel_size * length_out;
                std::copy_n(src, in_channels_per_group * kernel_size * length_out, dst);
            }

            // Convert col back to 1D format (kernel_h=1, kernel_w=kernel_size for 1D convolution)
            // Use stride_h=1, stride_w=stride_, padding_h=0, padding_w=padding_
            auto grad_input_slice = col2im(grad_col, in_channels_per_group, 1, length,
                                          1, kernel_size, 1, stride_, 0, padding_, dilation_);

            // Squeeze out the height dimension (which is 1)
            auto grad_input_slice_squeezed = grad_input_slice.reshape({batch, in_channels_per_group, length});

            // Copy to appropriate position in grad_input
            const float* grad_input_slice_data = grad_input_slice_squeezed.data<float>();
            float* grad_input_data = grad_input.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t l = 0; l < length; ++l) {
                        int64_t src_idx = b * (in_channels_per_group * length) +
                                        ic * length + l;
                        int64_t dst_idx = b * (in_channels * length) +
                                        (in_start + ic) * length + l;
                        grad_input_data[dst_idx] += grad_input_slice_data[src_idx];
                    }
                }
            }
        }

        // Gradient w.r.t weight
        Tensor grad_weight = zeros({out_channels, in_channels_per_group, kernel_size}, DType::Float32, Device::cpu());
        float* grad_weight_data = grad_weight.data<float>();

        for (int64_t g = 0; g < groups_; ++g) {
            int64_t in_start = g * in_channels_per_group;
            int64_t out_start = g * out_channels_per_group;

            // Extract input slice
            auto input_slice = zeros({batch, in_channels_per_group, length}, DType::Float32, Device::cpu());
            const float* input_data = input.data<float>();
            float* input_slice_data = input_slice.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t l = 0; l < length; ++l) {
                        int64_t src_idx = b * (in_channels * length) +
                                        (in_start + ic) * length + l;
                        int64_t dst_idx = b * (in_channels_per_group * length) +
                                        ic * length + l;
                        input_slice_data[dst_idx] = input_data[src_idx];
                    }
                }
            }

            // Reshape to 4D for im2col: [batch, in_channels_per_group, 1, length]
            auto input_slice_4d = input_slice.reshape({batch, in_channels_per_group, 1, length});
            // Use kernel_h=1, kernel_w=kernel_size, stride_h=1, stride_w=stride_, padding_h=0, padding_w=padding_
            auto input_col = im2col(input_slice_4d, 1, kernel_size, 1, stride_, 0, padding_, dilation_);

            // Extract grad_output slice
            auto grad_slice = zeros({batch, out_channels_per_group, length_out}, DType::Float32, Device::cpu());
            const float* grad_data = grad_output.data<float>();
            float* grad_slice_data = grad_slice.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                    for (int64_t l = 0; l < length_out; ++l) {
                        int64_t src_idx = b * (out_channels * length_out) +
                                        (out_start + oc) * length_out + l;
                        int64_t dst_idx = b * (out_channels_per_group * length_out) +
                                        oc * length_out + l;
                        grad_slice_data[dst_idx] = grad_data[src_idx];
                    }
                }
            }

            // Reshape for matmul
            auto grad_reshaped = grad_slice.reshape({batch, out_channels_per_group, length_out});
            auto input_col_reshaped = input_col.reshape({batch, in_channels_per_group * kernel_size, length_out});

            // Initialize gradient weight for this group
            auto grad_weight_group = zeros({out_channels_per_group, in_channels_per_group * kernel_size}, DType::Float32, Device::cpu());
            float* grad_weight_group_data = grad_weight_group.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                auto grad_b = zeros({out_channels_per_group, length_out}, DType::Float32, Device::cpu());
                const float* grad_reshaped_data = grad_reshaped.data<float>();
                float* grad_b_data = grad_b.data<float>();

                int64_t grad_slice_size = out_channels_per_group * length_out;
                for (int64_t i = 0; i < grad_slice_size; ++i) {
                    grad_b_data[i] = grad_reshaped_data[b * grad_slice_size + i];
                }

                auto input_col_b = zeros({in_channels_per_group * kernel_size, length_out}, DType::Float32, Device::cpu());
                const float* input_col_reshaped_data = input_col_reshaped.data<float>();
                float* input_col_b_data = input_col_b.data<float>();

                int64_t input_slice_size = in_channels_per_group * kernel_size * length_out;
                for (int64_t i = 0; i < input_slice_size; ++i) {
                    input_col_b_data[i] = input_col_reshaped_data[b * input_slice_size + i];
                }

                auto input_col_b_t = input_col_b.transpose(0, 1).contiguous();
                auto grad_weight_b = matmul(grad_b, input_col_b_t);

                const float* src = grad_weight_b.data<float>();
                for (int64_t i = 0; i < out_channels_per_group * in_channels_per_group * kernel_size; ++i) {
                    grad_weight_group_data[i] += src[i];
                }
            }

            // Reshape and copy to appropriate position
            grad_weight_group = grad_weight_group.reshape({out_channels_per_group, in_channels_per_group, kernel_size});
            const float* grad_weight_group_data_src = grad_weight_group.data<float>();

            for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t k = 0; k < kernel_size; ++k) {
                        int64_t src_idx = oc * (in_channels_per_group * kernel_size) +
                                        ic * kernel_size + k;
                        int64_t dst_idx = (out_start + oc) * (in_channels_per_group * kernel_size) +
                                        ic * kernel_size + k;
                        grad_weight_data[dst_idx] = grad_weight_group_data_src[src_idx];
                    }
                }
            }
        }

        // Gradient w.r.t bias
        Tensor grad_bias;
        if (saved_tensors_.size() > 2) {
            grad_bias = zeros({out_channels}, DType::Float32, Device::cpu());
            float* grad_bias_data = grad_bias.data<float>();
            const float* grad_data = grad_output.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t c = 0; c < out_channels; ++c) {
                    for (int64_t l = 0; l < length_out; ++l) {
                        int64_t idx = b * (out_channels * length_out) + c * length_out + l;
                        grad_bias_data[c] += grad_data[idx];
                    }
                }
            }
        }

        // Transfer back to original device
        if (need_cpu_transfer) {
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

// Example: Conv1d(3, 16, 5) creates 1D convolution with:
// - 3 input channels, 16 output channels, kernel size 5
// Input shape: (N, 3, L) -> Output shape: (N, 16, L_out)
Conv1d::Conv1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
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
    std::vector<int64_t> weight_shape = {out_channels, in_channels / groups, kernel_size};
    int64_t fan_in = (in_channels / groups) * kernel_size;
    float std_init = std::sqrt(2.0f / fan_in);
    auto weight_tensor = randn(weight_shape) * std_init;
    auto weight_var = Variable(weight_tensor, true);
    register_parameter("weight", weight_var);

    // Initialize bias
    if (bias) {
        std::vector<int64_t> bias_shape = {out_channels};
        // Uniform initialization: U(-sqrt(k), sqrt(k)) where k = 1/fan_in
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto bias_tensor = (randn(bias_shape) * 2.0f * bound) - bound;
        auto bias_var = Variable(bias_tensor, true);
        register_parameter("bias", bias_var);
    }
}

auto Conv1d::forward(const Variable& input) -> Variable {
    // Input shape: [batch, in_channels, length]
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

    // Calculate output length: L_out = floor((L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    int64_t length_out = calculate_output_size(length, kernel_size_, stride_, padding_, dilation_);

    // Validate output size to prevent memory allocation errors
    if (length_out <= 0) {
        throw std::invalid_argument(
            "Invalid Conv1d configuration: output length is non-positive (" +
            std::to_string(length_out) + "). Input length=" + std::to_string(length) +
            ", kernel_size=" + std::to_string(kernel_size_) +
            ", stride=" + std::to_string(stride_) +
            ", padding=" + std::to_string(padding_) +
            ", dilation=" + std::to_string(dilation_) +
            ". Try reducing kernel_size, dilation, or increasing input length/padding."
        );
    }

    // Strategy: Use Conv2d internally by treating 1D as 2D with height=1
    // Reshape input from [batch, channels, length] to [batch, channels, 1, length]
    auto input_4d = input.tensor().reshape({batch, in_channels, 1, length});
    auto input_4d_var = Variable(input_4d, input.requires_grad());

    // Copy grad_fn to maintain computation graph
    if (input.grad_fn()) {
        input_4d_var.set_grad_fn(input.grad_fn());
    }

    Device original_device = input.tensor().device();
    bool need_cpu_transfer = (original_device.type != Device::Type::CPU);

    Tensor input_work = need_cpu_transfer ? input_4d.to(Device::cpu()) : input_4d;
    auto& weight = *parameters_["weight"];
    Tensor weight_work = need_cpu_transfer ? weight.tensor().to(Device::cpu()) : weight.tensor();

    auto weight_shape = weight.tensor().shape();
    int64_t in_channels_per_group = weight_shape[1];
    int64_t out_channels_per_group = out_channels_ / groups_;

    auto output = zeros({batch, out_channels_, length_out}, input.tensor().dtype(), input.tensor().device());
    Tensor output_work = need_cpu_transfer ? output.to(Device::cpu()) : output;

    // Process each group
    for (int64_t g = 0; g < groups_; ++g) {
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Extract input slice [batch, in_channels_per_group, 1, length]
        auto input_slice = zeros({batch, in_channels_per_group, 1, length}, DType::Float32, Device::cpu());
        const float* input_data = input_work.data<float>();
        float* input_slice_data = input_slice.data<float>();

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < in_channels_per_group; ++c) {
                for (int64_t l = 0; l < length; ++l) {
                    int64_t src_idx = b * (in_channels_ * 1 * length) +
                                     (in_start + c) * (1 * length) + l;
                    int64_t dst_idx = b * (in_channels_per_group * 1 * length) +
                                     c * (1 * length) + l;
                    input_slice_data[dst_idx] = input_data[src_idx];
                }
            }
        }

        // Apply im2col (kernel_h=1, kernel_w=kernel_size for 1D convolution)
        // Use padding_h=0, padding_w=padding_ to only pad the width (length) dimension
        auto input_col = im2col(input_slice, 1, kernel_size_, 1, stride_, 0, padding_, dilation_);

        // Extract weight slice [out_channels_per_group, in_channels_per_group, kernel_size]
        auto weight_slice = zeros({out_channels_per_group, in_channels_per_group, kernel_size_, 1}, DType::Float32, Device::cpu());
        const float* weight_data = weight_work.data<float>();
        float* weight_slice_data = weight_slice.data<float>();

        for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
            for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                for (int64_t k = 0; k < kernel_size_; ++k) {
                    int64_t src_idx = (out_start + oc) * (in_channels_per_group * kernel_size_) +
                                     ic * kernel_size_ + k;
                    int64_t dst_idx = oc * (in_channels_per_group * kernel_size_ * 1) +
                                     ic * (kernel_size_ * 1) + k * 1;
                    weight_slice_data[dst_idx] = weight_data[src_idx];
                }
            }
        }

        // Reshape for matmul
        int64_t kernel_size_flat = in_channels_per_group * kernel_size_;
        auto weight_reshaped = weight_slice.reshape({out_channels_per_group, kernel_size_flat});

        // Process each batch
        for (int64_t b = 0; b < batch; ++b) {
            auto input_col_b = zeros({kernel_size_flat, length_out}, DType::Float32, Device::cpu());
            const float* input_col_data = input_col.data<float>();
            float* input_col_b_data = input_col_b.data<float>();

            for (int64_t i = 0; i < kernel_size_flat * length_out; ++i) {
                input_col_b_data[i] = input_col_data[b * kernel_size_flat * length_out + i];
            }

            auto output_group = matmul(weight_reshaped, input_col_b);

            // Copy to output
            const float* output_group_data = output_group.data<float>();
            float* output_data = output_work.data<float>();

            for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                for (int64_t l = 0; l < length_out; ++l) {
                    int64_t dst_idx = b * (out_channels_ * length_out) +
                                     (out_start + oc) * length_out + l;
                    int64_t src_idx = oc * length_out + l;
                    output_data[dst_idx] = output_group_data[src_idx];
                }
            }
        }
    }

    // Add bias if present
    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        auto& bias = *bias_it->second;
        Tensor bias_work = need_cpu_transfer ? bias.tensor().to(Device::cpu()) : bias.tensor();
        float* out_data = output_work.data<float>();
        const float* bias_data = bias_work.data<float>();

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < out_channels_; ++c) {
                for (int64_t l = 0; l < length_out; ++l) {
                    int64_t idx = b * (out_channels_ * length_out) + c * length_out + l;
                    out_data[idx] += bias_data[c];
                }
            }
        }
    }

    // Transfer back to GPU if needed
    if (need_cpu_transfer) {
        output = output_work.to(original_device);
    } else {
        output = output_work;
    }

    // Create output variable with autograd
    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    if (input.requires_grad() || weight.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_it != parameters_.end()) {
            auto& bias = *bias_it->second;
            tensors_to_save = {input.tensor(), weight.tensor(), bias.tensor()};
        } else {
            tensors_to_save = {input.tensor(), weight.tensor()};
        }

        auto backward_fn = std::make_shared<Conv1dBackward>(
            stride_, padding_, dilation_, groups_, std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        // Track input variables for gradient accumulation using pointers
        std::vector<Variable> input_vars;
        if (input.requires_grad()) {
            input_vars.push_back(input);
        }
        if (weight.requires_grad()) {
            input_vars.push_back(*parameters_["weight"]);
        }
        if (bias_it != parameters_.end() && bias_it->second->requires_grad()) {
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

auto Conv1d::reset_parameters() -> void {
    // Kaiming/He initialization
    int64_t fan_in = in_channels_ / groups_ * kernel_size_;
    float std = std::sqrt(2.0f / fan_in);

    std::vector<int64_t> weight_shape = {out_channels_, in_channels_ / groups_, kernel_size_};
    auto new_weight_tensor = randn(weight_shape) * std;
    parameters_["weight"] = std::make_shared<Variable>(new_weight_tensor, true);

    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        std::vector<int64_t> bias_shape = {out_channels_};
        auto new_bias_tensor = (randn(bias_shape) * 2.0f * bound) - bound;
        bias_it->second = std::make_shared<Variable>(new_bias_tensor, true);
    }
}

// ============================================================================
// ConvTranspose2d Implementation
// ============================================================================

// ConvTranspose2dBackward autograd function
class ConvTranspose2dBackward : public Function {
public:
    ConvTranspose2dBackward(int64_t stride, int64_t padding, int64_t output_padding,
                            int64_t dilation, int64_t groups, std::vector<Tensor> tensors_to_save)
        : stride_(stride), padding_(padding), output_padding_(output_padding),
          dilation_(dilation), groups_(groups) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("ConvTranspose2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // Transposed convolution backward = regular convolution forward
        // grad_outputs[0]: gradient w.r.t output [batch, out_channels, H_out, W_out]
        // saved_tensors_[0]: input [batch, in_channels, H_in, W_in]
        // saved_tensors_[1]: weight [in_channels, out_channels/groups, kernel_h, kernel_w]

        Device original_device = grad_outputs[0].device();
        bool need_cpu_transfer = (original_device.type != Device::Type::CPU);

        const Tensor grad_output = need_cpu_transfer ? grad_outputs[0].to(Device::cpu()) : grad_outputs[0];
        const Tensor input = need_cpu_transfer ? saved_tensors_[0].to(Device::cpu()) : saved_tensors_[0];
        const Tensor weight = need_cpu_transfer ? saved_tensors_[1].to(Device::cpu()) : saved_tensors_[1];

        auto weight_shape = weight.shape();
        int64_t in_channels = weight_shape[0];
        int64_t out_channels_per_group = weight_shape[1];
        int64_t kernel_h = weight_shape[2];
        int64_t kernel_w = weight_shape[3];
        int64_t out_channels = out_channels_per_group * groups_;

        auto input_shape = input.shape();
        int64_t batch = input_shape[0];
        int64_t height_in = input_shape[2];
        int64_t width_in = input_shape[3];

        auto grad_shape = grad_output.shape();
        int64_t height_out = grad_shape[2];
        int64_t width_out = grad_shape[3];

        // Debug: Print tensor shapes at the start
        std::cout << "ConvTranspose2d backward START: input shape=[" << batch << ", " << in_channels
                  << ", " << height_in << ", " << width_in << "], grad_output shape=["
                  << batch << ", " << out_channels << ", " << height_out << ", " << width_out << "]" << std::endl;

        // Gradient w.r.t input: Apply regular convolution with grad_output and flipped weight
        Tensor grad_input = zeros({batch, in_channels, height_in, width_in}, DType::Float32, Device::cpu());
        int64_t in_channels_per_group = in_channels / groups_;

        // For transposed conv backward (= regular conv forward):
        // We need to convolve grad_output with weight to get grad_input
        for (int64_t g = 0; g < groups_; ++g) {
            int64_t in_start = g * in_channels_per_group;
            int64_t out_start = g * out_channels_per_group;

            // Extract grad_output slice
            auto grad_slice = zeros({batch, out_channels_per_group, height_out, width_out}, DType::Float32, Device::cpu());
            const float* grad_data = grad_output.data<float>();
            float* grad_slice_data = grad_slice.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                    for (int64_t h = 0; h < height_out; ++h) {
                        for (int64_t w = 0; w < width_out; ++w) {
                            int64_t src_idx = b * (out_channels * height_out * width_out) +
                                            (out_start + oc) * (height_out * width_out) +
                                            h * width_out + w;
                            int64_t dst_idx = b * (out_channels_per_group * height_out * width_out) +
                                            oc * (height_out * width_out) +
                                            h * width_out + w;
                            grad_slice_data[dst_idx] = grad_data[src_idx];
                        }
                    }
                }
            }

            // Extract weight slice and flip it
            auto weight_slice = zeros({out_channels_per_group, in_channels_per_group, kernel_h, kernel_w}, DType::Float32, Device::cpu());
            const float* weight_data = weight.data<float>();
            float* weight_slice_data = weight_slice.data<float>();

            for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t src_idx = (in_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                            oc * (kernel_h * kernel_w) +
                                            kh * kernel_w + kw;
                            // Flip the kernel
                            int64_t dst_idx = oc * (in_channels_per_group * kernel_h * kernel_w) +
                                            ic * (kernel_h * kernel_w) +
                                            (kernel_h - 1 - kh) * kernel_w + (kernel_w - 1 - kw);
                            weight_slice_data[dst_idx] = weight_data[src_idx];
                        }
                    }
                }
            }

            // For ConvTranspose2d backward w.r.t. input, we need to reverse the forward operation:
            // Forward: input_flat → matmul(weight^T, input_flat) → output_col → col2im → output
            // Backward: grad_output → im2col_backward → matmul(weight, grad_col) → grad_input_flat → reshape

            // The spatial dimension in forward's output_col is H_in * W_in
            int64_t spatial_in = height_in * width_in;

            // Reshape grad_output to flat format: [batch, out_channels_per_group, H_out * W_out]
            auto grad_output_flat = grad_slice.reshape({batch, out_channels_per_group, height_out * width_out});

            // We need to convert grad_output from spatial format back to column format
            // This is the gradient through col2im, which requires accumulating based on the col2im logic
            // Instead of implementing a complex col2im gradient, we use im2col which effectively reverses it
            auto grad_col = im2col(grad_slice, kernel_h, kernel_w, stride_, padding_, dilation_);

            // Get the actual spatial size from im2col output
            auto grad_col_shape = grad_col.shape();
            int64_t actual_spatial_size = grad_col_shape[2];

            // Reshape weight to match forward: [in_channels_per_group, out_channels_per_group * K * K]
            int64_t kernel_flat = out_channels_per_group * kernel_h * kernel_w;
            auto weight_reshaped = weight_slice.reshape({in_channels_per_group, kernel_flat});

            // Now compute grad_input by reversing the forward matmul
            // Forward was: output_col = weight_reshaped^T @ input_flat
            // So backward is: grad_input_flat = weight_reshaped @ grad_col
            auto grad_input_flat = zeros({batch, in_channels_per_group, actual_spatial_size}, DType::Float32, Device::cpu());
            float* grad_input_flat_data = grad_input_flat.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                // Extract grad_col for this batch: [out_channels_per_group * K * K, spatial]
                auto grad_col_b = zeros({kernel_flat, actual_spatial_size}, DType::Float32, Device::cpu());
                const float* grad_col_ptr = grad_col.data<float>();
                float* grad_col_b_data = grad_col_b.data<float>();

                int64_t col_size = kernel_flat * actual_spatial_size;
                for (int64_t i = 0; i < col_size; ++i) {
                    grad_col_b_data[i] = grad_col_ptr[b * col_size + i];
                }

                // Matmul: weight_reshaped @ grad_col_b
                // = [in_channels_per_group, out_channels_per_group * K * K] @ [out_channels_per_group * K * K, spatial]
                // = [in_channels_per_group, spatial]
                auto grad_input_b = matmul(weight_reshaped, grad_col_b);

                // Copy to output
                const float* src = grad_input_b.data<float>();
                float* dst = grad_input_flat_data + b * in_channels_per_group * actual_spatial_size;
                std::copy_n(src, in_channels_per_group * actual_spatial_size, dst);
            }

            // Reshape grad_input_flat to spatial format: [batch, in_channels_per_group, H_in, W_in]
            auto grad_input_slice = grad_input_flat.reshape({batch, in_channels_per_group, height_in, width_in});

            // Accumulate into grad_input
            const float* grad_input_slice_data_src = grad_input_slice.data<float>();
            float* grad_input_data = grad_input.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t h = 0; h < height_in; ++h) {
                        for (int64_t w = 0; w < width_in; ++w) {
                            int64_t src_idx = b * (in_channels_per_group * height_in * width_in) +
                                            ic * (height_in * width_in) +
                                            h * width_in + w;
                            int64_t dst_idx = b * (in_channels * height_in * width_in) +
                                            (in_start + ic) * (height_in * width_in) +
                                            h * width_in + w;
                            grad_input_data[dst_idx] += grad_input_slice_data_src[src_idx];
                        }
                    }
                }
            }
        }

        // Gradient w.r.t weight
        Tensor grad_weight = zeros({in_channels, out_channels_per_group, kernel_h, kernel_w}, DType::Float32, Device::cpu());
        float* grad_weight_data = grad_weight.data<float>();

        for (int64_t g = 0; g < groups_; ++g) {
            int64_t in_start = g * in_channels_per_group;
            int64_t out_start = g * out_channels_per_group;

            // Extract input slice
            auto input_slice = zeros({batch, in_channels_per_group, height_in, width_in}, DType::Float32, Device::cpu());
            const float* input_data = input.data<float>();
            float* input_slice_data = input_slice.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t h = 0; h < height_in; ++h) {
                        for (int64_t w = 0; w < width_in; ++w) {
                            int64_t src_idx = b * (in_channels * height_in * width_in) +
                                            (in_start + ic) * (height_in * width_in) +
                                            h * width_in + w;
                            int64_t dst_idx = b * (in_channels_per_group * height_in * width_in) +
                                            ic * (height_in * width_in) +
                                            h * width_in + w;
                            input_slice_data[dst_idx] = input_data[src_idx];
                        }
                    }
                }
            }

            // For weight gradient, we need:
            // - input in flat format: [batch, in_channels_per_group, H_in * W_in]
            // - grad_output in column format: [batch, out_channels_per_group * K * K, spatial]
            // The spatial dimension should match H_in * W_in after im2col with the correct parameters

            // Reshape input to flat format: [batch, in_channels_per_group, H_in * W_in]
            auto input_flat = input_slice.reshape({batch, in_channels_per_group, height_in * width_in});

            // Extract grad_output slice
            auto grad_slice = zeros({batch, out_channels_per_group, height_out, width_out}, DType::Float32, Device::cpu());
            const float* grad_data = grad_output.data<float>();
            float* grad_slice_data = grad_slice.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                    for (int64_t h = 0; h < height_out; ++h) {
                        for (int64_t w = 0; w < width_out; ++w) {
                            int64_t src_idx = b * (out_channels * height_out * width_out) +
                                            (out_start + oc) * (height_out * width_out) +
                                            h * width_out + w;
                            int64_t dst_idx = b * (out_channels_per_group * height_out * width_out) +
                                            oc * (height_out * width_out) +
                                            h * width_out + w;
                            grad_slice_data[dst_idx] = grad_data[src_idx];
                        }
                    }
                }
            }

            // Apply im2col to grad_output to get grad_col
            // With correct parameters, this should give spatial dimension = H_in * W_in
            auto grad_output_col = im2col(grad_slice, kernel_h, kernel_w, stride_, padding_, dilation_);
            auto grad_col_shape = grad_output_col.shape();
            int64_t spatial_weight = grad_col_shape[2];

            // Verify that spatial dimensions match
            if (spatial_weight != height_in * width_in) {
                throw std::runtime_error(
                    "ConvTranspose2d backward weight gradient: spatial dimension mismatch. "
                    "im2col(grad_output) gave spatial=" + std::to_string(spatial_weight) +
                    " but expected H_in*W_in=" + std::to_string(height_in * width_in)
                );
            }

            // Both tensors now have spatial dimension height_in * width_in
            auto grad_col_reshaped = grad_output_col.reshape({batch, out_channels_per_group * kernel_h * kernel_w, spatial_weight});

            // Compute grad_weight: input_flat @ grad_col^T
            // input_flat: [batch, in_channels, spatial]
            // grad_col: [batch, out_channels * K * K, spatial]
            // Result: [in_channels, out_channels * K * K]
            auto grad_weight_group = zeros({in_channels_per_group, out_channels_per_group * kernel_h * kernel_w}, DType::Float32, Device::cpu());
            float* grad_weight_group_data = grad_weight_group.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                // Extract input for this batch: [in_channels, spatial]
                auto input_b = zeros({in_channels_per_group, spatial_weight}, DType::Float32, Device::cpu());
                const float* input_data = input_flat.data<float>();
                float* input_b_data = input_b.data<float>();

                int64_t input_size = in_channels_per_group * spatial_weight;
                for (int64_t i = 0; i < input_size; ++i) {
                    input_b_data[i] = input_data[b * input_size + i];
                }

                // Extract grad_col for this batch: [out_channels * K * K, spatial]
                auto grad_col_b = zeros({out_channels_per_group * kernel_h * kernel_w, spatial_weight}, DType::Float32, Device::cpu());
                const float* grad_col_data = grad_col_reshaped.data<float>();
                float* grad_col_b_data = grad_col_b.data<float>();

                int64_t grad_size = out_channels_per_group * kernel_h * kernel_w * spatial_weight;
                for (int64_t i = 0; i < grad_size; ++i) {
                    grad_col_b_data[i] = grad_col_data[b * grad_size + i];
                }

                // Matmul: input_b @ grad_col_b^T
                // = [in_channels, spatial] @ [spatial, out_channels * K * K]
                // = [in_channels, out_channels * K * K]
                auto grad_col_b_t = grad_col_b.transpose(0, 1).contiguous();
                auto grad_weight_b = matmul(input_b, grad_col_b_t);

                const float* src = grad_weight_b.data<float>();
                for (int64_t i = 0; i < in_channels_per_group * out_channels_per_group * kernel_h * kernel_w; ++i) {
                    grad_weight_group_data[i] += src[i];
                }
            }

            // Reshape and copy to grad_weight
            grad_weight_group = grad_weight_group.reshape({in_channels_per_group, out_channels_per_group, kernel_h, kernel_w});
            const float* grad_weight_group_data_src = grad_weight_group.data<float>();

            for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t src_idx = ic * (out_channels_per_group * kernel_h * kernel_w) +
                                            oc * (kernel_h * kernel_w) +
                                            kh * kernel_w + kw;
                            int64_t dst_idx = (in_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                            oc * (kernel_h * kernel_w) +
                                            kh * kernel_w + kw;
                            grad_weight_data[dst_idx] = grad_weight_group_data_src[src_idx];
                        }
                    }
                }
            }
        }

        // Gradient w.r.t bias
        Tensor grad_bias;
        if (saved_tensors_.size() > 2) {
            grad_bias = zeros({out_channels}, DType::Float32, Device::cpu());
            float* grad_bias_data = grad_bias.data<float>();
            const float* grad_data = grad_output.data<float>();

            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t c = 0; c < out_channels; ++c) {
                    for (int64_t h = 0; h < height_out; ++h) {
                        for (int64_t w = 0; w < width_out; ++w) {
                            int64_t idx = b * (out_channels * height_out * width_out) +
                                        c * (height_out * width_out) +
                                        h * width_out + w;
                            grad_bias_data[c] += grad_data[idx];
                        }
                    }
                }
            }
        }

        // Transfer back to original device
        if (need_cpu_transfer) {
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
    int64_t output_padding_;
    int64_t dilation_;
    int64_t groups_;
};

// Example: ConvTranspose2d(16, 3, 4, 2) creates transposed 2D convolution for upsampling:
// - 16 input channels, 3 output channels, kernel size 4x4, stride 2 (2x upsampling)
// Input shape: (N, 16, H, W) -> Output shape: (N, 3, H*2, W*2) approximately
// Commonly used in decoder networks, GANs, and autoencoders for upsampling feature maps
ConvTranspose2d::ConvTranspose2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                                 int64_t stride, int64_t padding, int64_t output_padding,
                                 int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride),
      padding_(padding), output_padding_(output_padding), groups_(groups) {

    // Validate parameters
    if (in_channels % groups != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }
    if (output_padding >= stride) {
        throw std::invalid_argument("output_padding must be smaller than stride");
    }

    // Weight shape for transposed conv: [in_channels, out_channels/groups, kernel_h, kernel_w]
    std::vector<int64_t> weight_shape = {in_channels, out_channels / groups, kernel_size, kernel_size};
    int64_t fan_in = in_channels * kernel_size * kernel_size;
    float std_init = std::sqrt(2.0f / fan_in);
    auto weight_tensor = randn(weight_shape) * std_init;
    weight_ = Variable(weight_tensor, true);
    register_parameter("weight", weight_);

    // Initialize bias
    if (bias) {
        std::vector<int64_t> bias_shape = {out_channels};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto bias_tensor = (randn(bias_shape) * 2.0f * bound) - bound;
        bias_ = Variable(bias_tensor, true);
        register_parameter("bias", *bias_);
    }
}

auto ConvTranspose2d::forward(const Variable& input) -> Variable {
    // Input shape: [batch, in_channels, height_in, width_in]
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("ConvTranspose2d expects 4D input [batch, channels, height, width]");
    }

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height_in = input_shape[2];
    int64_t width_in = input_shape[3];

    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }

    // Calculate output dimensions
    // H_out = (H_in - 1) * stride - 2*padding + kernel_size + output_padding
    int64_t height_out = (height_in - 1) * stride_ - 2 * padding_ + kernel_size_ + output_padding_;
    int64_t width_out = (width_in - 1) * stride_ - 2 * padding_ + kernel_size_ + output_padding_;

    // Validate output dimensions to prevent memory allocation errors
    if (height_out <= 0 || width_out <= 0) {
        throw std::invalid_argument(
            "Invalid ConvTranspose2d configuration: output dimensions are non-positive (out_h=" +
            std::to_string(height_out) + ", out_w=" + std::to_string(width_out) + "). " +
            "Input size=" + std::to_string(height_in) + "x" + std::to_string(width_in) +
            ", kernel_size=" + std::to_string(kernel_size_) +
            ", stride=" + std::to_string(stride_) +
            ", padding=" + std::to_string(padding_) +
            ", output_padding=" + std::to_string(output_padding_) +
            ". Check your layer configuration."
        );
    }

    Device original_device = input.tensor().device();
    bool need_cpu_transfer = (original_device.type != Device::Type::CPU);

    Tensor input_work = need_cpu_transfer ? input.tensor().to(Device::cpu()) : input.tensor();
    Tensor weight_work = need_cpu_transfer ? weight_.tensor().to(Device::cpu()) : weight_.tensor();

    int64_t in_channels_per_group = in_channels_ / groups_;
    int64_t out_channels_per_group = out_channels_ / groups_;

    auto output = zeros({batch, out_channels_, height_out, width_out}, input.tensor().dtype(), input.tensor().device());
    Tensor output_work = need_cpu_transfer ? output.to(Device::cpu()) : output;

    // Transposed convolution = GEMM → col2im (reverse of regular convolution)
    for (int64_t g = 0; g < groups_; ++g) {
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Extract input slice [batch, in_channels_per_group, height_in, width_in]
        auto input_slice = zeros({batch, in_channels_per_group, height_in, width_in}, DType::Float32, Device::cpu());
        const float* input_data = input_work.data<float>();
        float* input_slice_data = input_slice.data<float>();

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < in_channels_per_group; ++c) {
                for (int64_t h = 0; h < height_in; ++h) {
                    for (int64_t w = 0; w < width_in; ++w) {
                        int64_t src_idx = b * (in_channels_ * height_in * width_in) +
                                         (in_start + c) * (height_in * width_in) +
                                         h * width_in + w;
                        int64_t dst_idx = b * (in_channels_per_group * height_in * width_in) +
                                         c * (height_in * width_in) +
                                         h * width_in + w;
                        input_slice_data[dst_idx] = input_data[src_idx];
                    }
                }
            }
        }

        // Extract weight slice [in_channels_per_group, out_channels_per_group, kernel, kernel]
        auto weight_slice = zeros({in_channels_per_group, out_channels_per_group, kernel_size_, kernel_size_}, DType::Float32, Device::cpu());
        const float* weight_data = weight_work.data<float>();
        float* weight_slice_data = weight_slice.data<float>();

        for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
            for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                for (int64_t kh = 0; kh < kernel_size_; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size_; ++kw) {
                        int64_t src_idx = (in_start + ic) * (out_channels_per_group * kernel_size_ * kernel_size_) +
                                         oc * (kernel_size_ * kernel_size_) +
                                         kh * kernel_size_ + kw;
                        int64_t dst_idx = ic * (out_channels_per_group * kernel_size_ * kernel_size_) +
                                         oc * (kernel_size_ * kernel_size_) +
                                         kh * kernel_size_ + kw;
                        weight_slice_data[dst_idx] = weight_data[src_idx];
                    }
                }
            }
        }

        // Reshape input for matmul: [batch, in_channels_per_group, height_in * width_in]
        auto input_reshaped = input_slice.reshape({batch, in_channels_per_group, height_in * width_in});

        // Reshape weight: [in_channels_per_group, out_channels_per_group * kernel * kernel]
        int64_t kernel_flat = out_channels_per_group * kernel_size_ * kernel_size_;
        auto weight_reshaped = weight_slice.reshape({in_channels_per_group, kernel_flat});

        // Process each batch: matmul to get col representation
        auto output_col = zeros({batch, out_channels_per_group * kernel_size_ * kernel_size_, height_in * width_in}, DType::Float32, Device::cpu());
        float* output_col_data = output_col.data<float>();

        for (int64_t b = 0; b < batch; ++b) {
            // Extract input for this batch
            auto input_b = zeros({in_channels_per_group, height_in * width_in}, DType::Float32, Device::cpu());
            const float* input_reshaped_data = input_reshaped.data<float>();
            float* input_b_data = input_b.data<float>();

            int64_t slice_size = in_channels_per_group * height_in * width_in;
            for (int64_t i = 0; i < slice_size; ++i) {
                input_b_data[i] = input_reshaped_data[b * slice_size + i];
            }

            // GEMM: weight^T @ input_b = [kernel_flat, in_channels_per_group] @ [in_channels_per_group, spatial]
            auto weight_t = weight_reshaped.transpose(0, 1).contiguous();
            auto output_col_b = matmul(weight_t, input_b);

            // Copy to output_col
            const float* src = output_col_b.data<float>();
            float* dst = output_col_data + b * kernel_flat * height_in * width_in;
            std::copy_n(src, kernel_flat * height_in * width_in, dst);
        }

        // Apply col2im to get the actual output
        auto output_slice = col2im(output_col, out_channels_per_group, height_out, width_out,
                                  kernel_size_, kernel_size_, stride_, padding_, 1);

        // Copy to output at appropriate position
        const float* output_slice_data = output_slice.data<float>();
        float* output_data = output_work.data<float>();

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < out_channels_per_group; ++c) {
                for (int64_t h = 0; h < height_out; ++h) {
                    for (int64_t w = 0; w < width_out; ++w) {
                        int64_t src_idx = b * (out_channels_per_group * height_out * width_out) +
                                        c * (height_out * width_out) +
                                        h * width_out + w;
                        int64_t dst_idx = b * (out_channels_ * height_out * width_out) +
                                        (out_start + c) * (height_out * width_out) +
                                        h * width_out + w;
                        output_data[dst_idx] = output_slice_data[src_idx];
                    }
                }
            }
        }
    }

    // Add bias if present
    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        auto& bias = *bias_it->second;
        Tensor bias_work = need_cpu_transfer ? bias.tensor().to(Device::cpu()) : bias.tensor();
        float* out_data = output_work.data<float>();
        const float* bias_data = bias_work.data<float>();

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < out_channels_; ++c) {
                for (int64_t h = 0; h < height_out; ++h) {
                    for (int64_t w = 0; w < width_out; ++w) {
                        int64_t idx = b * (out_channels_ * height_out * width_out) +
                                     c * (height_out * width_out) +
                                     h * width_out + w;
                        out_data[idx] += bias_data[c];
                    }
                }
            }
        }
    }

    // Transfer back to GPU if needed
    if (need_cpu_transfer) {
        output = output_work.to(original_device);
    } else {
        output = output_work;
    }

    // Create output variable with autograd
    auto result = Variable(output, input.requires_grad() || weight_.requires_grad());

    if (input.requires_grad() || weight_.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_) {
            tensors_to_save = {input.tensor(), weight_.tensor(), bias_->tensor()};
        } else {
            tensors_to_save = {input.tensor(), weight_.tensor()};
        }

        auto backward_fn = std::make_shared<ConvTranspose2dBackward>(
            stride_, padding_, output_padding_, 1, groups_, std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        // Track input variables for gradient accumulation
        std::vector<Variable> input_vars;
        if (input.requires_grad()) {
            input_vars.push_back(input);
        }
        if (weight_.requires_grad()) {
            input_vars.push_back(*parameters_["weight"]);
        }
        if (bias_) {
            auto bias_it = parameters_.find("bias");
            if (bias_it != parameters_.end() && bias_it->second->requires_grad()) {
                input_vars.push_back(*bias_it->second);
            }
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
    // Kaiming/He initialization
    int64_t fan_in = in_channels_ * kernel_size_ * kernel_size_;
    float std = std::sqrt(2.0f / fan_in);

    std::vector<int64_t> weight_shape = {in_channels_, out_channels_ / groups_, kernel_size_, kernel_size_};
    auto new_weight_tensor = randn(weight_shape) * std;
    weight_ = Variable(new_weight_tensor, true);
    parameters_["weight"] = std::make_shared<Variable>(weight_);

    if (bias_) {
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        std::vector<int64_t> bias_shape = {out_channels_};
        auto new_bias_tensor = (randn(bias_shape) * 2.0f * bound) - bound;
        *bias_ = Variable(new_bias_tensor, true);
        parameters_["bias"] = std::make_shared<Variable>(*bias_);
    }
}

} // namespace tenzor::nn
