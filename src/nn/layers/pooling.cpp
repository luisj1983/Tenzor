#include "tenzor/nn/layers/pooling.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/autograd/function.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>
#include <vector>

namespace tenzor::nn {

// Helper namespace for pooling operations
namespace {

// Calculate output size for pooling
auto calculate_pool_output_size(int64_t input_size, int64_t kernel_size,
                                int64_t stride, int64_t padding) -> int64_t {
    if (stride == 0) {
        throw std::invalid_argument("Pooling: stride cannot be zero");
    }
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

} // anonymous namespace

// =======================
// MaxPool2d Implementation
// =======================

// MaxPool2d autograd function
class MaxPool2dBackward : public Function {
public:
    MaxPool2dBackward(int64_t kernel_size, int64_t stride, int64_t padding,
                     std::vector<Tensor> tensors_to_save)
        : kernel_size_(kernel_size), stride_(stride), padding_(padding) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("MaxPool2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // grad_outputs[0]: gradient w.r.t output [N, C, H_out, W_out]
        // saved_tensors_[0]: input [N, C, H_in, W_in]
        // saved_tensors_[1]: indices (flattened max positions) [N, C, H_out, W_out]

        const auto& grad_output = grad_outputs[0];
        const auto& input = saved_tensors_[0];
        const auto& indices = saved_tensors_[1];

        auto input_shape = input.shape();
        int64_t N = input_shape[0];
        int64_t C = input_shape[1];
        int64_t H_in = input_shape[2];
        int64_t W_in = input_shape[3];

        auto grad_shape = grad_output.shape();
        int64_t H_out = grad_shape[2];
        int64_t W_out = grad_shape[3];

        // Initialize gradient w.r.t input with zeros on same device
        auto grad_input = zeros({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());
        float* grad_input_data = grad_input.data<float>();
        const float* grad_output_data = grad_output.data<float>();
        const float* indices_data = indices.data<float>();

        // Distribute gradients to max element positions
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                    for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                        int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;

                        // Get the index of max element in input
                        int64_t max_idx = static_cast<int64_t>(indices_data[out_idx]);

                        // Add gradient to the position of max element
                        grad_input_data[max_idx] += grad_output_data[out_idx];
                    }
                }
            }
        }

        return {grad_input};
    }

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};

MaxPool2d::MaxPool2d(int64_t kernel_size, int64_t stride, int64_t padding)
    : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride),
      padding_(padding) {}

auto MaxPool2d::forward(const Variable& input) -> Variable {
    // Input shape: [N, C, H_in, W_in]
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("MaxPool2d expects 4D input [batch, channels, height, width]");
    }

    // Store original device
    Device original_device = input.tensor().device();

    // Transfer input to CPU if needed (current implementation uses CPU loops)
    Variable cpu_input = (original_device.type != Device::Type::CPU)
                        ? Variable(input.tensor().to(Device::cpu()), input.requires_grad())
                        : input;

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];

    // Calculate output dimensions
    int64_t H_out = calculate_pool_output_size(H_in, kernel_size_, stride_, padding_);
    int64_t W_out = calculate_pool_output_size(W_in, kernel_size_, stride_, padding_);

    // Create output tensor and indices tensor on CPU
    auto output = zeros({N, C, H_out, W_out}, cpu_input.tensor().dtype(), Device::cpu());
    auto indices = zeros({N, C, H_out, W_out}, cpu_input.tensor().dtype(), Device::cpu());

    const float* input_data = cpu_input.tensor().data<float>();
    float* output_data = output.data<float>();
    float* indices_data = indices.data<float>();

    // Perform max pooling
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                    // Calculate input window boundaries
                    int64_t h_start = h_out * stride_ - padding_;
                    int64_t w_start = w_out * stride_ - padding_;
                    int64_t h_end = h_start + kernel_size_;
                    int64_t w_end = w_start + kernel_size_;

                    // Find max value in window
                    float max_val = -std::numeric_limits<float>::infinity();
                    int64_t max_idx = 0;

                    for (int64_t h = h_start; h < h_end; ++h) {
                        for (int64_t w = w_start; w < w_end; ++w) {
                            // Check bounds and apply padding (padding is treated as -inf)
                            if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                                int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                                if (input_data[input_idx] > max_val) {
                                    max_val = input_data[input_idx];
                                    max_idx = input_idx;
                                }
                            }
                        }
                    }

                    int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                    output_data[out_idx] = max_val;
                    indices_data[out_idx] = static_cast<float>(max_idx);
                }
            }
        }
    }

    // Create output variable with autograd support
    auto result = Variable(output, input.requires_grad());

    // Setup backward function if gradient is required
    if (input.requires_grad()) {
        std::vector<Tensor> tensors_to_save = {input.tensor(), indices};

        auto backward_fn = std::make_shared<MaxPool2dBackward>(
            kernel_size_, stride_, padding_, std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        backward_fn->set_input_variables(input_vars);

        // CRITICAL: Connect to input's grad_fn to continue the backward chain
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    // Transfer result back to original device if needed
    if (original_device.type != Device::Type::CPU) {
        result = Variable(result.tensor().to(original_device), result.requires_grad());
        // Note: saved_tensors remain on CPU for backward pass compatibility
    }

    return result;
}

// =======================
// AvgPool2d Implementation
// =======================

// AvgPool2d autograd function
class AvgPool2dBackward : public Function {
public:
    AvgPool2dBackward(int64_t kernel_size, int64_t stride, int64_t padding,
                     int64_t H_in, int64_t W_in,
                     std::vector<Tensor> tensors_to_save)
        : kernel_size_(kernel_size), stride_(stride), padding_(padding),
          H_in_(H_in), W_in_(W_in) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("AvgPool2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // grad_outputs[0]: gradient w.r.t output [N, C, H_out, W_out]

        const auto& grad_output = grad_outputs[0];
        auto grad_shape = grad_output.shape();
        int64_t N = grad_shape[0];
        int64_t C = grad_shape[1];
        int64_t H_out = grad_shape[2];
        int64_t W_out = grad_shape[3];

        // Initialize gradient w.r.t input with zeros on same device
        auto grad_input = zeros({N, C, H_in_, W_in_}, grad_output.dtype(), grad_output.device());
        float* grad_input_data = grad_input.data<float>();
        const float* grad_output_data = grad_output.data<float>();

        // Distribute gradients evenly across pooling windows
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                    for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                        // Calculate input window boundaries
                        int64_t h_start = h_out * stride_ - padding_;
                        int64_t w_start = w_out * stride_ - padding_;
                        int64_t h_end = h_start + kernel_size_;
                        int64_t w_end = w_start + kernel_size_;

                        // Count valid elements in window (for proper averaging)
                        int64_t count = 0;
                        for (int64_t h = h_start; h < h_end; ++h) {
                            for (int64_t w = w_start; w < w_end; ++w) {
                                if (h >= 0 && h < H_in_ && w >= 0 && w < W_in_) {
                                    count++;
                                }
                            }
                        }

                        // Get gradient for this output position
                        int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                        float grad_val = grad_output_data[out_idx] / static_cast<float>(count);

                        // Distribute gradient evenly to all elements in window
                        for (int64_t h = h_start; h < h_end; ++h) {
                            for (int64_t w = w_start; w < w_end; ++w) {
                                if (h >= 0 && h < H_in_ && w >= 0 && w < W_in_) {
                                    int64_t input_idx = ((n * C + c) * H_in_ + h) * W_in_ + w;
                                    grad_input_data[input_idx] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        return {grad_input};
    }

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t H_in_;
    int64_t W_in_;
};

AvgPool2d::AvgPool2d(int64_t kernel_size, int64_t stride, int64_t padding)
    : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride),
      padding_(padding) {}

auto AvgPool2d::forward(const Variable& input) -> Variable {
    // Input shape: [N, C, H_in, W_in]
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("AvgPool2d expects 4D input [batch, channels, height, width]");
    }

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];

    // Calculate output dimensions
    int64_t H_out = calculate_pool_output_size(H_in, kernel_size_, stride_, padding_);
    int64_t W_out = calculate_pool_output_size(W_in, kernel_size_, stride_, padding_);

    // Create output tensor on same device as input
    auto output = zeros({N, C, H_out, W_out}, input.tensor().dtype(), input.tensor().device());

    const float* input_data = input.tensor().data<float>();
    float* output_data = output.data<float>();

    // Perform average pooling
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                    // Calculate input window boundaries
                    int64_t h_start = h_out * stride_ - padding_;
                    int64_t w_start = w_out * stride_ - padding_;
                    int64_t h_end = h_start + kernel_size_;
                    int64_t w_end = w_start + kernel_size_;

                    // Compute average in window
                    float sum = 0.0f;
                    int64_t count = 0;

                    for (int64_t h = h_start; h < h_end; ++h) {
                        for (int64_t w = w_start; w < w_end; ++w) {
                            // Check bounds and apply padding (padding is treated as 0)
                            if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                                int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                                sum += input_data[input_idx];
                                count++;
                            }
                        }
                    }

                    int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                    output_data[out_idx] = sum / static_cast<float>(count);
                }
            }
        }
    }

    // Create output variable with autograd support
    auto result = Variable(output, input.requires_grad());

    // Setup backward function if gradient is required
    if (input.requires_grad()) {
        std::vector<Tensor> tensors_to_save = {};  // No tensors needed for avg pool backward

        auto backward_fn = std::make_shared<AvgPool2dBackward>(
            kernel_size_, stride_, padding_, H_in, W_in, std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        backward_fn->set_input_variables(input_vars);

        // CRITICAL: Connect to input's grad_fn to continue the backward chain
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

// =======================
// AdaptiveAvgPool2d Implementation
// =======================

// AdaptiveAvgPool2d autograd function
class AdaptiveAvgPool2dBackward : public Function {
public:
    AdaptiveAvgPool2dBackward(int64_t H_in, int64_t W_in,
                             int64_t H_out, int64_t W_out,
                             std::vector<Tensor> tensors_to_save)
        : H_in_(H_in), W_in_(W_in), H_out_(H_out), W_out_(W_out) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("AdaptiveAvgPool2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // grad_outputs[0]: gradient w.r.t output [N, C, H_out, W_out]

        const auto& grad_output = grad_outputs[0];
        auto grad_shape = grad_output.shape();
        int64_t N = grad_shape[0];
        int64_t C = grad_shape[1];

        // Initialize gradient w.r.t input with zeros on same device
        auto grad_input = zeros({N, C, H_in_, W_in_}, grad_output.dtype(), grad_output.device());
        float* grad_input_data = grad_input.data<float>();
        const float* grad_output_data = grad_output.data<float>();

        // Distribute gradients based on adaptive pooling windows
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h_out = 0; h_out < H_out_; ++h_out) {
                    for (int64_t w_out = 0; w_out < W_out_; ++w_out) {
                        // Calculate adaptive pooling window
                        int64_t h_start = (h_out * H_in_) / H_out_;
                        int64_t h_end = ((h_out + 1) * H_in_) / H_out_;
                        int64_t w_start = (w_out * W_in_) / W_out_;
                        int64_t w_end = ((w_out + 1) * W_in_) / W_out_;

                        // Count elements in window
                        int64_t count = (h_end - h_start) * (w_end - w_start);

                        // Get gradient for this output position
                        int64_t out_idx = ((n * C + c) * H_out_ + h_out) * W_out_ + w_out;
                        float grad_val = grad_output_data[out_idx] / static_cast<float>(count);

                        // Distribute gradient evenly to all elements in window
                        for (int64_t h = h_start; h < h_end; ++h) {
                            for (int64_t w = w_start; w < w_end; ++w) {
                                int64_t input_idx = ((n * C + c) * H_in_ + h) * W_in_ + w;
                                grad_input_data[input_idx] += grad_val;
                            }
                        }
                    }
                }
            }
        }

        return {grad_input};
    }

private:
    int64_t H_in_;
    int64_t W_in_;
    int64_t H_out_;
    int64_t W_out_;
};

AdaptiveAvgPool2d::AdaptiveAvgPool2d(int64_t output_h, int64_t output_w)
    : output_h_(output_h), output_w_(output_w) {}

auto AdaptiveAvgPool2d::forward(const Variable& input) -> Variable {
    // Input shape: [N, C, H_in, W_in]
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("AdaptiveAvgPool2d expects 4D input [batch, channels, height, width]");
    }

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];

    // Output dimensions are fixed
    int64_t H_out = output_h_;
    int64_t W_out = output_w_;

    // Create output tensor on same device as input
    auto output = zeros({N, C, H_out, W_out}, input.tensor().dtype(), input.tensor().device());

    const float* input_data = input.tensor().data<float>();
    float* output_data = output.data<float>();

    // Perform adaptive average pooling
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                    // Calculate adaptive pooling window
                    // This ensures uniform coverage of input
                    int64_t h_start = (h_out * H_in) / H_out;
                    int64_t h_end = ((h_out + 1) * H_in) / H_out;
                    int64_t w_start = (w_out * W_in) / W_out;
                    int64_t w_end = ((w_out + 1) * W_in) / W_out;

                    // Compute average in adaptive window
                    float sum = 0.0f;
                    int64_t count = 0;

                    for (int64_t h = h_start; h < h_end; ++h) {
                        for (int64_t w = w_start; w < w_end; ++w) {
                            int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                            sum += input_data[input_idx];
                            count++;
                        }
                    }

                    int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                    output_data[out_idx] = sum / static_cast<float>(count);
                }
            }
        }
    }

    // Create output variable with autograd support
    auto result = Variable(output, input.requires_grad());

    // Setup backward function if gradient is required
    if (input.requires_grad()) {
        std::vector<Tensor> tensors_to_save = {};  // No tensors needed

        auto backward_fn = std::make_shared<AdaptiveAvgPool2dBackward>(
            H_in, W_in, H_out, W_out, std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        backward_fn->set_input_variables(input_vars);

        // CRITICAL: Connect to input's grad_fn to continue the backward chain
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

} // namespace tenzor::nn
