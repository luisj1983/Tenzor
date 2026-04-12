#include "tenzor/nn/layers/pooling.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>
#include <vector>

namespace tenzor::nn {

// Helper namespace for pooling operations
namespace {

// Calculate output size for pooling
auto calculate_pool_output_size(int64_t input_size, int64_t kernel_size,
                                int64_t stride, int64_t padding,
                                bool ceil_mode = false) -> int64_t {
    if (stride == 0) {
        throw std::runtime_error("Pooling: stride cannot be zero");
    }
    if (ceil_mode) {
        return (input_size + 2 * padding - kernel_size + stride - 1) / stride + 1;
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
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("MaxPool2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // grad_outputs[0]: gradient w.r.t output [N, C, H_out, W_out]
        // saved_tensors_[0]: input [N, C, H_in, W_in]
        // saved_tensors_[1]: indices [N, C, H_out, W_out]
        // saved_tensors_[2]: output [N, C, H_out, W_out] (for cuDNN backward)

        const auto& grad_output = grad_outputs[0];
        const auto& input = saved_tensors_[0];
        const auto& indices = saved_tensors_[1];
        const auto& output = saved_tensors_[2];

        auto input_shape = input.shape();
        int64_t N = input_shape[0];
        int64_t C = input_shape[1];
        int64_t H_in = input_shape[2];
        int64_t W_in = input_shape[3];

        // Use OpId dispatch for non-CPU devices (CUDA, Vulkan, etc.)
        if (grad_output.device().type != Device::Type::CPU) {
            std::vector<Tensor> inputs = {grad_output, indices, input, output};
            OpAttributes bwd_attrs;
            bwd_attrs.set(AttrKey::KernelSize, kernel_size_);
            bwd_attrs.set(AttrKey::Stride, stride_);
            bwd_attrs.set(AttrKey::Padding, padding_);
            bwd_attrs.set(AttrKey::InputShape, std::to_string(N) + "," + std::to_string(C) + "," + std::to_string(H_in) + "," + std::to_string(W_in));
            auto result = dispatch_to_device(OpId::MaxPool2dBackward, grad_output.device().type,
                inputs, bwd_attrs);
            return {result[0]};
        }

        // CPU path
        auto grad_shape = grad_output.shape();
        int64_t H_out = grad_shape[2];
        int64_t W_out = grad_shape[3];

        auto dtype = grad_output.dtype();
        auto grad_input = zeros({N, C, H_in, W_in}, dtype, Device::cpu());

        // Distribute gradients to max element positions with proper dtype handling
        // Indices are always stored as Int64
        const int64_t* indices_data = indices.data<int64_t>();

        if (dtype == DType::Float32) {
            float* grad_input_data = grad_input.data<float>();
            const float* grad_output_data = grad_output.data<float>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            int64_t max_idx = indices_data[out_idx];
                            grad_input_data[max_idx] += grad_output_data[out_idx];
                        }
                    }
                }
            }
        } else if (dtype == DType::Float64) {
            double* grad_input_data = grad_input.data<double>();
            const double* grad_output_data = grad_output.data<double>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            int64_t max_idx = indices_data[out_idx];
                            grad_input_data[max_idx] += grad_output_data[out_idx];
                        }
                    }
                }
            }
        } else if (dtype == DType::Float16) {
            // Accumulate in Float32 buffer for precision, convert back at end
            std::vector<int64_t> gi_shape(grad_input.shape().begin(), grad_input.shape().end());
            Tensor grad_input_f32(gi_shape, DType::Float32, grad_input.device());
            grad_input_f32.zero_();
            float* gi_f32_data = grad_input_f32.data<float>();
            const Float16* grad_output_data = grad_output.data<Float16>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            int64_t max_idx = indices_data[out_idx];
                            gi_f32_data[max_idx] += static_cast<float>(grad_output_data[out_idx]);
                        }
                    }
                }
            }
            // Convert accumulated Float32 gradients back to Float16
            grad_input = grad_input_f32.to(DType::Float16);
        } else {
            throw std::runtime_error("MaxPool2dBackward: Unsupported dtype");
        }

        return {grad_input};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Pooling backward uses index-based scatter -- delegate to tensor backward and wrap results
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

    // P4.2d: the passthrough above returns Variables without grad_fn,
    // so higher-order gradients through this op are structurally zero.
    // Declare that honestly so the engine's disconnection counter
    // reports accurately in Warn mode and throws in Error mode.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};

MaxPool2d::MaxPool2d(int64_t kernel_size, int64_t stride, int64_t padding,
                     bool ceil_mode, bool return_indices)
    : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride),
      padding_(padding), ceil_mode_(ceil_mode), return_indices_(return_indices) {
    if (kernel_size <= 0) {
        throw std::runtime_error("MaxPool2d: kernel_size must be positive, got " +
            std::to_string(kernel_size));
    }
    if (padding < 0) {
        throw std::runtime_error("MaxPool2d: padding must be non-negative, got " +
            std::to_string(padding));
    }
}

auto MaxPool2d::forward_impl(const Variable& input) -> Variable {
    // Input shape: [N, C, H_in, W_in]
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("MaxPool2d expects 4D input [batch, channels, height, width]");
    }

    Device original_device = input.tensor().device();
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];

    Tensor output, indices;

    // Use OpId dispatch for non-CPU devices (CUDA, Vulkan, etc.)
    if (original_device.type != Device::Type::CPU) {
        std::vector<Tensor> inputs = {input.tensor()};
        OpAttributes fwd_attrs;
        fwd_attrs.set(AttrKey::KernelSize, kernel_size_);
        fwd_attrs.set(AttrKey::Stride, stride_);
        fwd_attrs.set(AttrKey::Padding, padding_);
        auto result = dispatch_to_device(OpId::MaxPool2dForward, original_device.type,
            inputs, fwd_attrs);
        output = result[0];
        indices = result[1];
    } else {
        // CPU path
        // Calculate output dimensions
        int64_t H_out = calculate_pool_output_size(H_in, kernel_size_, stride_, padding_, ceil_mode_);
        int64_t W_out = calculate_pool_output_size(W_in, kernel_size_, stride_, padding_, ceil_mode_);

        // Create output tensor and indices tensor on CPU
        auto dtype = input.tensor().dtype();
        output = zeros({N, C, H_out, W_out}, dtype, Device::cpu());
        // Always store indices as Int64 for precision (Float16 can't represent large indices)
        indices = zeros({N, C, H_out, W_out}, DType::Int64, Device::cpu());

        // Perform max pooling with proper dtype handling
        if (dtype == DType::Float32) {
            const float* input_data = input.tensor().data<float>();
            float* output_data = output.data<float>();
            int64_t* indices_data = indices.data<int64_t>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = h_out * stride_ - padding_;
                            int64_t w_start = w_out * stride_ - padding_;
                            int64_t h_end = h_start + kernel_size_;
                            int64_t w_end = w_start + kernel_size_;

                            float max_val = -std::numeric_limits<float>::infinity();
                            int64_t max_idx = 0;

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
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
                            indices_data[out_idx] = max_idx;
                        }
                    }
                }
            }
        } else if (dtype == DType::Float64) {
            const double* input_data = input.tensor().data<double>();
            double* output_data = output.data<double>();
            int64_t* indices_data = indices.data<int64_t>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = h_out * stride_ - padding_;
                            int64_t w_start = w_out * stride_ - padding_;
                            int64_t h_end = h_start + kernel_size_;
                            int64_t w_end = w_start + kernel_size_;

                            double max_val = -std::numeric_limits<double>::infinity();
                            int64_t max_idx = 0;

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
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
                            indices_data[out_idx] = max_idx;
                        }
                    }
                }
            }
        } else if (dtype == DType::Float16) {
            const Float16* input_data = input.tensor().data<Float16>();
            Float16* output_data = output.data<Float16>();
            int64_t* indices_data = indices.data<int64_t>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = h_out * stride_ - padding_;
                            int64_t w_start = w_out * stride_ - padding_;
                            int64_t h_end = h_start + kernel_size_;
                            int64_t w_end = w_start + kernel_size_;

                            float max_val = -std::numeric_limits<float>::infinity();
                            int64_t max_idx = 0;

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                                        int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                                        float val = static_cast<float>(input_data[input_idx]);
                                        if (val > max_val) {
                                            max_val = val;
                                            max_idx = input_idx;
                                        }
                                    }
                                }
                            }

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            output_data[out_idx] = Float16(max_val);
                            indices_data[out_idx] = max_idx;
                        }
                    }
                }
            }
        } else if (dtype == DType::BFloat16) {
            const BFloat16* input_data = input.tensor().data<BFloat16>();
            BFloat16* output_data = output.data<BFloat16>();
            int64_t* indices_data = indices.data<int64_t>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = h_out * stride_ - padding_;
                            int64_t w_start = w_out * stride_ - padding_;
                            int64_t h_end = h_start + kernel_size_;
                            int64_t w_end = w_start + kernel_size_;

                            float max_val = -std::numeric_limits<float>::infinity();
                            int64_t max_idx = 0;

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                                        int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                                        float val = static_cast<float>(input_data[input_idx]);
                                        if (val > max_val) {
                                            max_val = val;
                                            max_idx = input_idx;
                                        }
                                    }
                                }
                            }

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            output_data[out_idx] = BFloat16(max_val);
                            indices_data[out_idx] = max_idx;
                        }
                    }
                }
            }
        } else {
            throw std::runtime_error("MaxPool2d: Unsupported dtype");
        }
    } // end CPU path

    // Create output variable with autograd support
    auto result = Variable(output, input.requires_grad());

    // Setup backward function if gradient is required
    if (input.requires_grad()) {
        std::vector<Tensor> tensors_to_save = {input.tensor(), indices, output};

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
        save_for_backward(std::move(tensors_to_save));
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

        // Use OpId dispatch for non-CPU devices (CUDA, Vulkan, etc.)
        if (grad_output.device().type != Device::Type::CPU) {
            // cuDNN path needs [grad_output, input]; non-cuDNN needs InputShape attr
            // Save input in saved_tensors_[0] for this purpose
            std::vector<Tensor> inputs = {grad_output, saved_tensors_[0]};
            OpAttributes bwd_attrs;
            bwd_attrs.set(AttrKey::InputShape, std::to_string(N) + "," + std::to_string(C) + "," + std::to_string(H_in_) + "," + std::to_string(W_in_));
            bwd_attrs.set(AttrKey::KernelSize, kernel_size_);
            bwd_attrs.set(AttrKey::Stride, stride_);
            bwd_attrs.set(AttrKey::Padding, padding_);
            auto result = dispatch_to_device(OpId::AvgPool2dBackward, grad_output.device().type,
                inputs, bwd_attrs);
            return {result[0]};
        }

        // CPU path - Initialize gradient w.r.t input with zeros on same device
        auto grad_input = zeros({N, C, H_in_, W_in_}, grad_output.dtype(), grad_output.device());

        auto dtype = grad_output.dtype();
        if (dtype == DType::Float32) {
            float* grad_input_data = grad_input.data<float>();
            const float* grad_output_data = grad_output.data<float>();

            // Distribute gradients evenly across pooling windows
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = h_out * stride_ - padding_;
                            int64_t w_start = w_out * stride_ - padding_;
                            int64_t h_end = h_start + kernel_size_;
                            int64_t w_end = w_start + kernel_size_;

                            int64_t count = 0;
                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in_ && w >= 0 && w < W_in_) {
                                        count++;
                                    }
                                }
                            }

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            float grad_val = grad_output_data[out_idx] / static_cast<float>(count);

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
        } else if (dtype == DType::Float64) {
            double* grad_input_data = grad_input.data<double>();
            const double* grad_output_data = grad_output.data<double>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = h_out * stride_ - padding_;
                            int64_t w_start = w_out * stride_ - padding_;
                            int64_t h_end = h_start + kernel_size_;
                            int64_t w_end = w_start + kernel_size_;

                            int64_t count = 0;
                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in_ && w >= 0 && w < W_in_) {
                                        count++;
                                    }
                                }
                            }

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            double grad_val = grad_output_data[out_idx] / static_cast<double>(count);

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
        } else if (dtype == DType::Float16) {
            Float16* grad_input_data = grad_input.data<Float16>();
            const Float16* grad_output_data = grad_output.data<Float16>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = h_out * stride_ - padding_;
                            int64_t w_start = w_out * stride_ - padding_;
                            int64_t h_end = h_start + kernel_size_;
                            int64_t w_end = w_start + kernel_size_;

                            int64_t count = 0;
                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in_ && w >= 0 && w < W_in_) {
                                        count++;
                                    }
                                }
                            }

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            float grad_val = static_cast<float>(grad_output_data[out_idx]) / static_cast<float>(count);

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in_ && w >= 0 && w < W_in_) {
                                        int64_t input_idx = ((n * C + c) * H_in_ + h) * W_in_ + w;
                                        grad_input_data[input_idx] = Float16(static_cast<float>(grad_input_data[input_idx]) + grad_val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return {grad_input};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Pooling backward uses scatter operations -- delegate to tensor backward and wrap results
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

    // P4.2d: the passthrough above returns Variables without grad_fn,
    // so higher-order gradients through this op are structurally zero.
    // Declare that honestly so the engine's disconnection counter
    // reports accurately in Warn mode and throws in Error mode.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

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

auto AvgPool2d::forward_impl(const Variable& input) -> Variable {
    // Input shape: [N, C, H_in, W_in]
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::runtime_error("AvgPool2d expects 4D input [batch, channels, height, width]");
    }

    Device original_device = input.tensor().device();
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];

    Tensor output;

    // Use OpId dispatch for non-CPU devices (CUDA, Vulkan, etc.)
    if (original_device.type != Device::Type::CPU) {
        std::vector<Tensor> inputs = {input.tensor()};
        OpAttributes fwd_attrs;
        fwd_attrs.set(AttrKey::KernelSize, kernel_size_);
        fwd_attrs.set(AttrKey::Stride, stride_);
        fwd_attrs.set(AttrKey::Padding, padding_);
        auto result = dispatch_to_device(OpId::AvgPool2dForward, original_device.type,
            inputs, fwd_attrs);
        output = result[0];
    } else {
        // CPU path
        // Calculate output dimensions
        int64_t H_out = calculate_pool_output_size(H_in, kernel_size_, stride_, padding_);
        int64_t W_out = calculate_pool_output_size(W_in, kernel_size_, stride_, padding_);

        // Create output tensor on CPU
        auto dtype = input.tensor().dtype();
        output = zeros({N, C, H_out, W_out}, dtype, Device::cpu());

        if (dtype == DType::Float32) {
            const float* input_data = input.tensor().data<float>();
            float* output_data = output.data<float>();

            // Perform average pooling
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = h_out * stride_ - padding_;
                            int64_t w_start = w_out * stride_ - padding_;
                            int64_t h_end = h_start + kernel_size_;
                            int64_t w_end = w_start + kernel_size_;

                            float sum = 0.0f;
                            int64_t count = 0;

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
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
        } else if (dtype == DType::Float64) {
            const double* input_data = input.tensor().data<double>();
            double* output_data = output.data<double>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = h_out * stride_ - padding_;
                            int64_t w_start = w_out * stride_ - padding_;
                            int64_t h_end = h_start + kernel_size_;
                            int64_t w_end = w_start + kernel_size_;

                            double sum = 0.0;
                            int64_t count = 0;

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                                        int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                                        sum += input_data[input_idx];
                                        count++;
                                    }
                                }
                            }

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            output_data[out_idx] = sum / static_cast<double>(count);
                        }
                    }
                }
            }
        } else if (dtype == DType::Float16) {
            const Float16* input_data = input.tensor().data<Float16>();
            Float16* output_data = output.data<Float16>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = h_out * stride_ - padding_;
                            int64_t w_start = w_out * stride_ - padding_;
                            int64_t h_end = h_start + kernel_size_;
                            int64_t w_end = w_start + kernel_size_;

                            float sum = 0.0f;
                            int64_t count = 0;

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                                        int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                                        sum += static_cast<float>(input_data[input_idx]);
                                        count++;
                                    }
                                }
                            }

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            output_data[out_idx] = Float16(sum / static_cast<float>(count));
                        }
                    }
                }
            }
        }
    }

    // Create output variable with autograd support
    auto result = Variable(output, input.requires_grad());

    // Setup backward function if gradient is required
    if (input.requires_grad()) {
        std::vector<Tensor> tensors_to_save = {input.tensor()};  // Save input for cuDNN backward

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
        save_for_backward(std::move(tensors_to_save));
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

        // Save target device for later
        auto target_device = grad_output.device();
        auto target_dtype = grad_output.dtype();

        // Use operation registry to dispatch to appropriate backend
        if (target_device.type != Device::Type::CPU) {
            OpAttributes attrs;
            attrs.set(AttrKey::InputH, H_in_);
            attrs.set(AttrKey::InputW, W_in_);
            std::vector<Tensor> inputs = {grad_output, saved_tensors_[0]};
            return dispatch<OpId::AdaptiveAvgPool2dBackward>(inputs, attrs);
        }

        // CPU fallback
        auto grad_output_cpu = grad_output.to(Device::cpu());
        auto grad_input = zeros({N, C, H_in_, W_in_}, target_dtype, Device::cpu());

        auto dtype = target_dtype;
        if (dtype == DType::Float32) {
            float* grad_input_data = grad_input.data<float>();
            const float* grad_output_data = grad_output_cpu.data<float>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out_; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out_; ++w_out) {
                            int64_t h_start = (h_out * H_in_) / H_out_;
                            int64_t h_end = ((h_out + 1) * H_in_) / H_out_;
                            int64_t w_start = (w_out * W_in_) / W_out_;
                            int64_t w_end = ((w_out + 1) * W_in_) / W_out_;

                            int64_t count = (h_end - h_start) * (w_end - w_start);
                            int64_t out_idx = ((n * C + c) * H_out_ + h_out) * W_out_ + w_out;
                            float grad_val = grad_output_data[out_idx] / static_cast<float>(count);

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
        } else if (dtype == DType::Float64) {
            double* grad_input_data = grad_input.data<double>();
            const double* grad_output_data = grad_output_cpu.data<double>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out_; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out_; ++w_out) {
                            int64_t h_start = (h_out * H_in_) / H_out_;
                            int64_t h_end = ((h_out + 1) * H_in_) / H_out_;
                            int64_t w_start = (w_out * W_in_) / W_out_;
                            int64_t w_end = ((w_out + 1) * W_in_) / W_out_;

                            int64_t count = (h_end - h_start) * (w_end - w_start);
                            int64_t out_idx = ((n * C + c) * H_out_ + h_out) * W_out_ + w_out;
                            double grad_val = grad_output_data[out_idx] / static_cast<double>(count);

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
        } else if (dtype == DType::Float16) {
            auto grad_output_f32 = grad_output_cpu.to(DType::Float32);
            auto grad_input_f32 = zeros({N, C, H_in_, W_in_}, DType::Float32, Device::cpu());

            float* grad_input_data = grad_input_f32.data<float>();
            const float* grad_output_data = grad_output_f32.data<float>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out_; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out_; ++w_out) {
                            int64_t h_start = (h_out * H_in_) / H_out_;
                            int64_t h_end = ((h_out + 1) * H_in_) / H_out_;
                            int64_t w_start = (w_out * W_in_) / W_out_;
                            int64_t w_end = ((w_out + 1) * W_in_) / W_out_;

                            int64_t count = (h_end - h_start) * (w_end - w_start);
                            int64_t out_idx = ((n * C + c) * H_out_ + h_out) * W_out_ + w_out;
                            float grad_val = grad_output_data[out_idx] / static_cast<float>(count);

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

            grad_input = grad_input_f32.to(DType::Float16);
        } else {
            throw std::runtime_error("AdaptiveAvgPool2dBackward: Unsupported dtype");
        }

        return {grad_input};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Adaptive pooling backward uses scatter operations -- delegate to tensor backward and wrap results
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

    // P4.2d: the passthrough above returns Variables without grad_fn,
    // so higher-order gradients through this op are structurally zero.
    // Declare that honestly so the engine's disconnection counter
    // reports accurately in Warn mode and throws in Error mode.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

private:
    int64_t H_in_;
    int64_t W_in_;
    int64_t H_out_;
    int64_t W_out_;
};

AdaptiveAvgPool2d::AdaptiveAvgPool2d(int64_t output_h, int64_t output_w)
    : output_h_(output_h), output_w_(output_w) {}

auto AdaptiveAvgPool2d::forward_impl(const Variable& input) -> Variable {
    // Input shape: [N, C, H_in, W_in]
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::runtime_error("AdaptiveAvgPool2d expects 4D input [batch, channels, height, width]");
    }

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];

    // Output dimensions are fixed
    int64_t H_out = output_h_;
    int64_t W_out = output_w_;

    // Save original device and dtype for later
    auto target_device = input.tensor().device();
    auto target_dtype = input.tensor().dtype();

    Tensor output;

    // Use operation registry to dispatch to appropriate backend
    if (target_device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::OutputSizeH, H_out);
        attrs.set(AttrKey::OutputSizeW, W_out);
        std::vector<Tensor> inputs = {input.tensor()};
        auto results = dispatch<OpId::AdaptiveAvgPool2d>(inputs, attrs);
        output = results[0];
    } else {
        // CPU fallback
        auto input_cpu = input.tensor().to(Device::cpu());
        output = zeros({N, C, H_out, W_out}, target_dtype, Device::cpu());

        auto dtype = target_dtype;
        if (dtype == DType::Float32) {
            const float* input_data = input_cpu.data<float>();
            float* output_data = output.data<float>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = (h_out * H_in) / H_out;
                            int64_t h_end = ((h_out + 1) * H_in) / H_out;
                            int64_t w_start = (w_out * W_in) / W_out;
                            int64_t w_end = ((w_out + 1) * W_in) / W_out;

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
        } else if (dtype == DType::Float64) {
            const double* input_data = input_cpu.data<double>();
            double* output_data = output.data<double>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = (h_out * H_in) / H_out;
                            int64_t h_end = ((h_out + 1) * H_in) / H_out;
                            int64_t w_start = (w_out * W_in) / W_out;
                            int64_t w_end = ((w_out + 1) * W_in) / W_out;

                            double sum = 0.0;
                            int64_t count = 0;

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                                    sum += input_data[input_idx];
                                    count++;
                                }
                            }

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            output_data[out_idx] = sum / static_cast<double>(count);
                        }
                    }
                }
            }
        } else if (dtype == DType::Float16) {
            auto input_f32 = input_cpu.to(DType::Float32);
            auto output_f32 = zeros({N, C, H_out, W_out}, DType::Float32, Device::cpu());

            const float* input_data = input_f32.data<float>();
            float* output_data = output_f32.data<float>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = (h_out * H_in) / H_out;
                            int64_t h_end = ((h_out + 1) * H_in) / H_out;
                            int64_t w_start = (w_out * W_in) / W_out;
                            int64_t w_end = ((w_out + 1) * W_in) / W_out;

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

            output = output_f32.to(DType::Float16);
        } else if (dtype == DType::BFloat16) {
            auto input_f32 = input_cpu.to(DType::Float32);
            auto output_f32 = zeros({N, C, H_out, W_out}, DType::Float32, Device::cpu());

            const float* input_data = input_f32.data<float>();
            float* output_data = output_f32.data<float>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = (h_out * H_in) / H_out;
                            int64_t h_end = ((h_out + 1) * H_in) / H_out;
                            int64_t w_start = (w_out * W_in) / W_out;
                            int64_t w_end = ((w_out + 1) * W_in) / W_out;

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

            output = output_f32.to(DType::BFloat16);
        } else {
            throw std::runtime_error("AdaptiveAvgPool2d: Unsupported dtype");
        }
    }

    // Create output variable with autograd support
    auto result = Variable(output, input.requires_grad());

    // Setup backward function if gradient is required
    if (input.requires_grad()) {
        std::vector<Tensor> tensors_to_save = {input.tensor()};

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

// ============================================================================
// Backward Function classes for 1D and 3D pooling
// ============================================================================
// These wrap the registered OpId backward kernels to wire pool ops into the
// autograd graph. Each Function class:
//   - Saves the input shape (and indices for max-pool variants)
//   - On backward(), dispatches to the registered backend kernel via OpId
//   - Returns the resulting input gradient
//
// The backward kernels are registered in src/backends/*/cpu_kernel_registry.cpp
// (and equivalent for other backends), so this dispatcher pattern works on all
// backends without needing CPU-specific code paths in this file.

namespace {

inline std::string pool_shape_to_string(const std::vector<int64_t>& shape) {
    std::string s;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ',';
        s += std::to_string(shape[i]);
    }
    return s;
}

// Generic backward Function for pool ops that need an input shape attribute.
// Used for AvgPool variants (no indices needed).
template <OpId BackwardOp>
class AvgPoolNdBackward : public Function {
public:
    AvgPoolNdBackward(std::vector<int64_t> input_shape,
                      int64_t kernel_size, int64_t stride, int64_t padding)
        : input_shape_(std::move(input_shape)),
          kernel_size_(kernel_size), stride_(stride), padding_(padding) {}

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error("AvgPoolNdBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        OpAttributes attrs;
        attrs.set(AttrKey::InputShape, pool_shape_to_string(input_shape_));
        attrs.set(AttrKey::KernelSize, kernel_size_);
        attrs.set(AttrKey::Stride, stride_);
        attrs.set(AttrKey::Padding, padding_);
        std::vector<Tensor> inputs = {grad_outputs[0]};
        auto result = dispatch_to_device(BackwardOp, grad_outputs[0].device().type, inputs, attrs);
        return {result[0]};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

    // P4.2d: the passthrough above returns Variables without grad_fn,
    // so higher-order gradients through this op are structurally zero.
    // Declare that honestly so the engine's disconnection counter
    // reports accurately in Warn mode and throws in Error mode.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

private:
    std::vector<int64_t> input_shape_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};

// Generic backward Function for max-pool variants (needs saved indices).
template <OpId BackwardOp>
class MaxPoolNdBackward : public Function {
public:
    MaxPoolNdBackward(std::vector<int64_t> input_shape, Tensor indices)
        : input_shape_(std::move(input_shape)) {
        save_for_backward({std::move(indices)});
    }

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error("MaxPoolNdBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        OpAttributes attrs;
        attrs.set(AttrKey::InputShape, pool_shape_to_string(input_shape_));
        std::vector<Tensor> inputs = {grad_outputs[0], saved_tensors_[0]};
        auto result = dispatch_to_device(BackwardOp, grad_outputs[0].device().type, inputs, attrs);
        return {result[0]};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

    // P4.2d: the passthrough above returns Variables without grad_fn,
    // so higher-order gradients through this op are structurally zero.
    // Declare that honestly so the engine's disconnection counter
    // reports accurately in Warn mode and throws in Error mode.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

private:
    std::vector<int64_t> input_shape_;
};

// Helper to wire a backward Function as the grad_fn of a result Variable.
inline void wire_grad_fn(Variable& result, const Variable& input,
                         std::shared_ptr<Function> backward_fn) {
    result.set_grad_fn(backward_fn);
    std::vector<Variable> input_vars{input};
    backward_fn->set_input_variables(input_vars);
    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) next_funcs.push_back(input.grad_fn());
    backward_fn->set_next_functions(next_funcs);
}

} // anonymous namespace

// ============================================================================
// MaxPool3d Implementation
// ============================================================================

MaxPool3d::MaxPool3d(int64_t kernel_size, int64_t stride, int64_t padding,
                     bool ceil_mode, bool return_indices)
    : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride),
      padding_(padding), ceil_mode_(ceil_mode), return_indices_(return_indices) {
    if (kernel_size <= 0) {
        throw std::runtime_error("MaxPool3d: kernel_size must be positive");
    }
    if (padding < 0) {
        throw std::runtime_error("MaxPool3d: padding must be non-negative");
    }
}

auto MaxPool3d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("MaxPool3d expects 5D input [batch, channels, depth, height, width]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::KernelSize, kernel_size_);
    fwd_attrs.set(AttrKey::Stride, stride_);
    fwd_attrs.set(AttrKey::Padding, padding_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::MaxPool3dForward, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];
    Tensor indices = fwd_result[1];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<MaxPoolNdBackward<OpId::MaxPool3dBackward>>(
            in_shape_vec, indices);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AvgPool3d Implementation
// ============================================================================

AvgPool3d::AvgPool3d(int64_t kernel_size, int64_t stride, int64_t padding)
    : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride),
      padding_(padding) {
    if (kernel_size <= 0) {
        throw std::runtime_error("AvgPool3d: kernel_size must be positive");
    }
    if (padding < 0) {
        throw std::runtime_error("AvgPool3d: padding must be non-negative");
    }
}

auto AvgPool3d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("AvgPool3d expects 5D input [batch, channels, depth, height, width]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::KernelSize, kernel_size_);
    fwd_attrs.set(AttrKey::Stride, stride_);
    fwd_attrs.set(AttrKey::Padding, padding_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AvgPool3dForward, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<AvgPoolNdBackward<OpId::AvgPool3dBackward>>(
            in_shape_vec, kernel_size_, stride_, padding_);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// MaxPool1d implementation
// ============================================================================

MaxPool1d::MaxPool1d(int64_t kernel_size, int64_t stride, int64_t padding,
                     bool ceil_mode, bool return_indices)
    : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride),
      padding_(padding), ceil_mode_(ceil_mode), return_indices_(return_indices) {
    if (kernel_size <= 0) throw std::runtime_error("MaxPool1d: kernel_size must be positive");
    if (padding < 0) throw std::runtime_error("MaxPool1d: padding must be non-negative");
}

auto MaxPool1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("MaxPool1d expects 3D input [batch, channels, length]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::KernelSize, kernel_size_);
    fwd_attrs.set(AttrKey::Stride, stride_);
    fwd_attrs.set(AttrKey::Padding, padding_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::MaxPool1dForward, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];
    Tensor indices = fwd_result[1];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<MaxPoolNdBackward<OpId::MaxPool1dBackward>>(
            in_shape_vec, indices);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AvgPool1d implementation
// ============================================================================

AvgPool1d::AvgPool1d(int64_t kernel_size, int64_t stride, int64_t padding)
    : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride),
      padding_(padding) {
    if (kernel_size <= 0) throw std::runtime_error("AvgPool1d: kernel_size must be positive");
    if (padding < 0) throw std::runtime_error("AvgPool1d: padding must be non-negative");
}

auto AvgPool1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("AvgPool1d expects 3D input [batch, channels, length]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::KernelSize, kernel_size_);
    fwd_attrs.set(AttrKey::Stride, stride_);
    fwd_attrs.set(AttrKey::Padding, padding_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AvgPool1dForward, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<AvgPoolNdBackward<OpId::AvgPool1dBackward>>(
            in_shape_vec, kernel_size_, stride_, padding_);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AdaptiveAvgPool1d implementation
// ============================================================================

AdaptiveAvgPool1d::AdaptiveAvgPool1d(int64_t output_size)
    : output_size_(output_size) {
    if (output_size <= 0) throw std::runtime_error("AdaptiveAvgPool1d: output_size must be positive");
}

auto AdaptiveAvgPool1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("AdaptiveAvgPool1d expects 3D input [batch, channels, length]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::OutputSize, output_size_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AdaptiveAvgPool1d, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        // AdaptiveAvgPool backward only needs the input shape (no kernel/stride/padding).
        auto backward_fn = std::make_shared<AvgPoolNdBackward<OpId::AdaptiveAvgPool1dBackward>>(
            in_shape_vec, /*kernel_size=*/0, /*stride=*/0, /*padding=*/0);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AdaptiveMaxPool2d implementation
// ============================================================================

AdaptiveMaxPool2d::AdaptiveMaxPool2d(int64_t output_h, int64_t output_w)
    : output_h_(output_h), output_w_(output_w) {
    if (output_h <= 0 || output_w <= 0) {
        throw std::runtime_error("AdaptiveMaxPool2d: output dimensions must be positive");
    }
}

auto AdaptiveMaxPool2d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("AdaptiveMaxPool2d expects 4D input [batch, channels, height, width]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::OutputSizeH, output_h_);
    fwd_attrs.set(AttrKey::OutputSizeW, output_w_);

    // Dispatch to backend kernel (CPU kernel supports Float32/Float64/Float16/BFloat16,
    // matching the other adaptive-pool layers).  The inlined CPU path that only
    // handled Float32/Float64 has been removed.
    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AdaptiveMaxPool2d, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];
    Tensor indices = fwd_result[1];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<MaxPoolNdBackward<OpId::AdaptiveMaxPool2dBackward>>(
            in_shape_vec, indices);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AdaptiveMaxPool1d implementation
// ============================================================================

AdaptiveMaxPool1d::AdaptiveMaxPool1d(int64_t output_size)
    : output_size_(output_size) {
    if (output_size <= 0) throw std::runtime_error("AdaptiveMaxPool1d: output_size must be positive");
}

auto AdaptiveMaxPool1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("AdaptiveMaxPool1d expects 3D input [batch, channels, length]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::OutputSize, output_size_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AdaptiveMaxPool1d, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];
    Tensor indices = fwd_result[1];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<MaxPoolNdBackward<OpId::AdaptiveMaxPool1dBackward>>(
            in_shape_vec, indices);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AdaptiveMaxPool3d implementation
// ============================================================================

AdaptiveMaxPool3d::AdaptiveMaxPool3d(int64_t output_d, int64_t output_h, int64_t output_w)
    : output_d_(output_d), output_h_(output_h), output_w_(output_w) {
    if (output_d <= 0 || output_h <= 0 || output_w <= 0) {
        throw std::runtime_error("AdaptiveMaxPool3d: output dimensions must be positive");
    }
}

auto AdaptiveMaxPool3d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("AdaptiveMaxPool3d expects 5D input [batch, channels, depth, height, width]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::OutputSizeD, output_d_);
    fwd_attrs.set(AttrKey::OutputSizeH, output_h_);
    fwd_attrs.set(AttrKey::OutputSizeW, output_w_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AdaptiveMaxPool3d, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];
    Tensor indices = fwd_result[1];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<MaxPoolNdBackward<OpId::AdaptiveMaxPool3dBackward>>(
            in_shape_vec, indices);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AdaptiveAvgPool3d implementation
// ============================================================================

AdaptiveAvgPool3d::AdaptiveAvgPool3d(int64_t output_d, int64_t output_h, int64_t output_w)
    : output_d_(output_d), output_h_(output_h), output_w_(output_w) {
    if (output_d <= 0 || output_h <= 0 || output_w <= 0) {
        throw std::runtime_error("AdaptiveAvgPool3d: output dimensions must be positive");
    }
}

auto AdaptiveAvgPool3d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("AdaptiveAvgPool3d expects 5D input [batch, channels, depth, height, width]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::OutputSizeD, output_d_);
    fwd_attrs.set(AttrKey::OutputSizeH, output_h_);
    fwd_attrs.set(AttrKey::OutputSizeW, output_w_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AdaptiveAvgPool3d, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        // Adaptive pool backward only needs the input shape.
        auto backward_fn = std::make_shared<AvgPoolNdBackward<OpId::AdaptiveAvgPool3dBackward>>(
            in_shape_vec, /*kernel_size=*/0, /*stride=*/0, /*padding=*/0);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

} // namespace tenzor::nn
