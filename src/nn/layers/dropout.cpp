#include "tenzor/nn/layers/dropout.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/autograd/function.hpp"
#include <stdexcept>

namespace tenzor::nn {

// Dropout autograd function
class DropoutBackward : public Function {
public:
    DropoutBackward(Tensor mask, double scale) : mask_(std::move(mask)), scale_(scale) {}

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("DropoutBackward expects 1 input");
        }

        // Apply mask and scale: output = input * mask * scale
        auto shape_span = inputs[0].tensor().shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        auto output = mul(mul(inputs[0].tensor(), mask_),
                         full(shape_vec, static_cast<float>(scale_),
                              inputs[0].tensor().dtype(), inputs[0].tensor().device()));

        std::vector<Variable> result;
        result.push_back(Variable(output, inputs[0].requires_grad()));
        return result;
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        if (grad_outputs.size() != 1) {
            throw std::invalid_argument("DropoutBackward expects 1 gradient output");
        }

        // Gradient: grad_input = grad_output * mask * scale
        auto shape_span = grad_outputs[0].shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        auto grad_input = mul(mul(grad_outputs[0], mask_),
                             full(shape_vec, static_cast<float>(scale_),
                                  grad_outputs[0].dtype(), grad_outputs[0].device()));

        std::vector<Tensor> result;
        result.push_back(grad_input);
        return result;
    }

private:
    Tensor mask_;
    double scale_;
};

// Element-wise Dropout
Dropout::Dropout(double p) : p_(p) {
    if (p < 0.0 || p >= 1.0) {
        throw std::invalid_argument("Dropout probability must be in [0, 1)");
    }
}

auto Dropout::forward_impl(const Variable& input) -> Variable {
    // During inference, return input unchanged
    if (!is_training()) {
        return input;
    }

    // During training, apply inverted dropout
    // 1. Generate random mask using Bernoulli distribution
    // 2. Scale by 1/(1-p) to maintain expected value

    // Generate uniform random values [0, 1) or all ones if p==0
    auto shape_span = input.tensor().shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());

    // Special case: p=1.0 means drop everything, return zeros
    if (p_ == 1.0) {
        auto output_tensor = zeros(shape_vec, input.tensor().dtype(), input.tensor().device());
        return Variable(output_tensor, input.requires_grad());
    }

    Tensor mask_data;
    if (p_ == 0.0) {
        // No dropout - mask is all ones
        mask_data = ones(shape_vec, input.tensor().dtype(), input.tensor().device());
    } else {
        // Create random tensor on CPU to access its data
        auto random_tensor = rand(shape_vec, input.tensor().dtype(), Device::cpu());

        // Create binary mask: mask = (random > p)
        // Elements with random value > p are kept (set to 1), others are dropped (set to 0)
        auto mask_data_cpu = zeros(shape_vec, input.tensor().dtype(), Device::cpu());

        // Create mask on CPU by comparing random values with p
        // mask[i] = random[i] > p ? 1 : 0
        size_t numel = random_tensor.numel();
        void* random_ptr = random_tensor.impl()->storage->data();
        void* mask_ptr = mask_data_cpu.impl()->storage->data();

        if (random_tensor.dtype() == DType::Float16) {
            Float16* rand_data = static_cast<Float16*>(random_ptr);
            Float16* mask_out = static_cast<Float16*>(mask_ptr);
            for (size_t i = 0; i < numel; ++i) {
                float rand_val = static_cast<float>(rand_data[i]);
                mask_out[i] = Float16(rand_val > static_cast<float>(p_) ? 1.0f : 0.0f);
            }
        } else if (random_tensor.dtype() == DType::Float32) {
            float* rand_data = static_cast<float*>(random_ptr);
            float* mask_out = static_cast<float*>(mask_ptr);
            for (size_t i = 0; i < numel; ++i) {
                mask_out[i] = rand_data[i] > static_cast<float>(p_) ? 1.0f : 0.0f;
            }
        } else if (random_tensor.dtype() == DType::Float64) {
            double* rand_data = static_cast<double*>(random_ptr);
            double* mask_out = static_cast<double*>(mask_ptr);
            for (size_t i = 0; i < numel; ++i) {
                mask_out[i] = rand_data[i] > p_ ? 1.0 : 0.0;
            }
        } else {
            throw std::runtime_error("Dropout only supports Float16, Float32 and Float64 dtypes");
        }

        // Transfer mask to target device if needed
        mask_data = (input.tensor().device().type == Device::Type::CPU) ?
                    mask_data_cpu : mask_data_cpu.to(input.tensor().device());
    }

    // Apply inverted dropout: output = input * mask / (1 - p)
    double scale = 1.0 / (1.0 - p_);

    // Compute forward: output = input * mask * scale
    auto scale_tensor = full(shape_vec, static_cast<float>(scale),
                            input.tensor().dtype(), input.tensor().device());
    auto output_tensor = mul(mul(input.tensor(), mask_data), scale_tensor);

    // Create output variable
    Variable output(output_tensor, input.requires_grad());

    // Set up autograd if input requires grad
    if (input.requires_grad()) {
        // Create autograd function
        auto dropout_fn = std::make_shared<DropoutBackward>(mask_data, scale);

        // Track input variable for gradient accumulation
        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        dropout_fn->set_input_variables(input_vars);

        // Set up backward graph - link to input's grad_fn if it exists
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        dropout_fn->set_next_functions(next_funcs);

        // Set gradient function on output
        output.set_grad_fn(dropout_fn);
    }

    return output;
}

// Channel-wise Dropout (Dropout2d)
Dropout2d::Dropout2d(double p) : p_(p) {
    if (p < 0.0 || p >= 1.0) {
        throw std::invalid_argument("Dropout2d probability must be in [0, 1)");
    }
}

auto Dropout2d::forward_impl(const Variable& input) -> Variable {
    // During inference, return input unchanged
    if (!is_training()) {
        return input;
    }

    if (p_ == 0.0) {
        return input;  // No dropout
    }

    // Special case: p=1.0 means drop all channels, return zeros
    if (p_ == 1.0) {
        auto shape = input.tensor().shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto output_tensor = zeros(shape_vec, input.tensor().dtype(), input.tensor().device());
        return Variable(output_tensor, input.requires_grad());
    }

    // Input shape: [N, C, H, W] or [C, H, W]
    auto shape = input.tensor().shape();

    if (shape.size() < 2) {
        throw std::invalid_argument("Dropout2d requires at least 2D input (C, H) or (N, C, H, W)");
    }

    // For 2D dropout, we drop entire channels
    // Mask shape: [N, C, 1, 1] or [C, 1, 1] - one mask value per channel
    std::vector<int64_t> mask_shape;

    if (shape.size() == 4) {
        // [N, C, H, W] -> mask [N, C, 1, 1]
        mask_shape = {shape[0], shape[1], 1, 1};
    } else if (shape.size() == 3) {
        // [C, H, W] -> mask [C, 1, 1]
        mask_shape = {shape[0], 1, 1};
    } else if (shape.size() == 2) {
        // [C, H] -> mask [C, 1]
        mask_shape = {shape[0], 1};
    } else {
        throw std::invalid_argument("Dropout2d input must be 2D, 3D or 4D");
    }

    // Generate random values for each channel on CPU
    auto random_tensor = rand(mask_shape, input.tensor().dtype(), Device::cpu());

    // Create binary mask on CPU
    auto mask_data = zeros_like(random_tensor);

    size_t numel = random_tensor.numel();
    void* random_ptr = random_tensor.impl()->storage->data();
    void* mask_ptr = mask_data.impl()->storage->data();

    if (random_tensor.dtype() == DType::Float16) {
        Float16* rand_data = static_cast<Float16*>(random_ptr);
        Float16* mask_out = static_cast<Float16*>(mask_ptr);
        for (size_t i = 0; i < numel; ++i) {
            float rand_val = static_cast<float>(rand_data[i]);
            mask_out[i] = Float16(rand_val > static_cast<float>(p_) ? 1.0f : 0.0f);
        }
    } else if (random_tensor.dtype() == DType::Float32) {
        float* rand_data = static_cast<float*>(random_ptr);
        float* mask_out = static_cast<float*>(mask_ptr);
        for (size_t i = 0; i < numel; ++i) {
            mask_out[i] = rand_data[i] > static_cast<float>(p_) ? 1.0f : 0.0f;
        }
    } else if (random_tensor.dtype() == DType::Float64) {
        double* rand_data = static_cast<double*>(random_ptr);
        double* mask_out = static_cast<double*>(mask_ptr);
        for (size_t i = 0; i < numel; ++i) {
            mask_out[i] = rand_data[i] > p_ ? 1.0 : 0.0;
        }
    } else {
        throw std::runtime_error("Dropout2d only supports Float16, Float32 and Float64 dtypes");
    }

    // Manually expand mask to input shape for proper channel-wise dropout
    double scale = 1.0 / (1.0 - p_);

    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    // Create expanded mask on CPU first
    auto expanded_mask = zeros(shape_vec, input.tensor().dtype(), Device::cpu());

    // Copy mask values to all spatial positions within each channel
    // Use template lambda to support multiple dtypes
    auto expand_mask = [&]<typename T>(T*) {
        const T* mask_ptr_data = static_cast<const T*>(mask_data.impl()->storage->data());
        T* expanded_ptr = static_cast<T*>(expanded_mask.impl()->storage->data());

        if (shape.size() == 4) {
            // [N, C, H, W]
            int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    T mask_val = mask_ptr_data[n * C + c];
                    for (int64_t h = 0; h < H; ++h) {
                        for (int64_t w = 0; w < W; ++w) {
                            expanded_ptr[n * (C * H * W) + c * (H * W) + h * W + w] = mask_val;
                        }
                    }
                }
            }
        } else if (shape.size() == 3) {
            // [C, H, W]
            int64_t C = shape[0], H = shape[1], W = shape[2];
            for (int64_t c = 0; c < C; ++c) {
                T mask_val = mask_ptr_data[c];
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        expanded_ptr[c * (H * W) + h * W + w] = mask_val;
                    }
                }
            }
        } else if (shape.size() == 2) {
            // [C, H]
            int64_t C = shape[0], H = shape[1];
            for (int64_t c = 0; c < C; ++c) {
                T mask_val = mask_ptr_data[c];
                for (int64_t h = 0; h < H; ++h) {
                    expanded_ptr[c * H + h] = mask_val;
                }
            }
        }
    };

    switch (input.tensor().dtype()) {
        case DType::Float32:
            expand_mask(static_cast<float*>(nullptr));
            break;
        case DType::Float64:
            expand_mask(static_cast<double*>(nullptr));
            break;
        case DType::Float16:
            expand_mask(static_cast<Float16*>(nullptr));
            break;
        default:
            throw std::runtime_error("Dropout2d only supports Float16, Float32 and Float64 dtypes");
    }

    // Transfer expanded mask to target device if needed
    auto expanded_mask_final = (input.tensor().device().type == Device::Type::CPU) ?
                               expanded_mask : expanded_mask.to(input.tensor().device());

    // Compute forward: output = input * expanded_mask * scale
    auto scale_tensor = full(shape_vec, static_cast<float>(scale),
                            input.tensor().dtype(), input.tensor().device());
    auto output_tensor = mul(mul(input.tensor(), expanded_mask_final), scale_tensor);

    // Create output variable
    Variable output(output_tensor, input.requires_grad());

    // Set up autograd if input requires grad
    if (input.requires_grad()) {
        // Create autograd function with expanded mask
        auto dropout_fn = std::make_shared<DropoutBackward>(expanded_mask_final, scale);

        // Track input variable for gradient accumulation
        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        dropout_fn->set_input_variables(input_vars);

        // Set up backward graph - link to input's grad_fn if it exists
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        dropout_fn->set_next_functions(next_funcs);

        // Set gradient function on output
        output.set_grad_fn(dropout_fn);
    }

    return output;
}

// AlphaDropout autograd function
class AlphaDropoutBackward : public Function {
public:
    AlphaDropoutBackward(Tensor mask, double a, double b)
        : mask_(std::move(mask)), a_(a), b_(b) {}

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("AlphaDropoutBackward expects 1 input");
        }

        // Apply affine transformation: output = a * input + b
        auto shape_span = inputs[0].tensor().shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        auto a_tensor = full(shape_vec, static_cast<float>(a_),
                            inputs[0].tensor().dtype(), inputs[0].tensor().device());
        auto b_tensor = full(shape_vec, static_cast<float>(b_),
                            inputs[0].tensor().dtype(), inputs[0].tensor().device());
        auto output = add(mul(inputs[0].tensor(), a_tensor), b_tensor);

        std::vector<Variable> result;
        result.push_back(Variable(output, inputs[0].requires_grad()));
        return result;
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        if (grad_outputs.size() != 1) {
            throw std::invalid_argument("AlphaDropoutBackward expects 1 gradient output");
        }

        // Gradient: grad_input = grad_output * a * mask
        // Only kept elements (mask=1) receive gradients, scaled by a
        auto shape_span = grad_outputs[0].shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        auto a_tensor = full(shape_vec, static_cast<float>(a_),
                            grad_outputs[0].dtype(), grad_outputs[0].device());
        auto grad_input = mul(mul(grad_outputs[0], a_tensor), mask_);

        std::vector<Tensor> result;
        result.push_back(grad_input);
        return result;
    }

private:
    Tensor mask_;
    double a_;
    double b_;
};

// Alpha Dropout (for SELU networks)
AlphaDropout::AlphaDropout(double p) : p_(p) {
    if (p < 0.0 || p >= 1.0) {
        throw std::invalid_argument("AlphaDropout probability must be in [0, 1)");
    }
}

auto AlphaDropout::forward_impl(const Variable& input) -> Variable {
    // During inference, return input unchanged
    if (!is_training()) {
        return input;
    }

    if (p_ == 0.0) {
        return input;  // No dropout
    }

    // SELU constants
    const double alpha = 1.6732632423543772848170429916717;
    const double scale = 1.0507009873554804934193349852946;
    const double alpha_p = -alpha * scale;  // approximately -1.7581

    // Affine transformation parameters to maintain mean=0, var=1
    const double a = std::sqrt((1.0 - p_) * (1.0 + p_ * alpha_p * alpha_p));
    const double b = -a * alpha_p * p_ / (1.0 - p_);

    // Generate random mask
    auto shape_span = input.tensor().shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());

    // Create random tensor on CPU
    auto random_tensor = rand(shape_vec, input.tensor().dtype(), Device::cpu());

    // Create binary mask: 1 for kept, 0 for dropped
    auto mask_data_cpu = zeros(shape_vec, input.tensor().dtype(), Device::cpu());

    size_t numel = random_tensor.numel();
    void* random_ptr = random_tensor.impl()->storage->data();
    void* mask_ptr = mask_data_cpu.impl()->storage->data();

    if (random_tensor.dtype() == DType::Float16) {
        Float16* rand_data = static_cast<Float16*>(random_ptr);
        Float16* mask_out = static_cast<Float16*>(mask_ptr);
        for (size_t i = 0; i < numel; ++i) {
            float rand_val = static_cast<float>(rand_data[i]);
            mask_out[i] = Float16(rand_val > static_cast<float>(p_) ? 1.0f : 0.0f);
        }
    } else if (random_tensor.dtype() == DType::Float32) {
        float* rand_data = static_cast<float*>(random_ptr);
        float* mask_out = static_cast<float*>(mask_ptr);
        for (size_t i = 0; i < numel; ++i) {
            mask_out[i] = rand_data[i] > static_cast<float>(p_) ? 1.0f : 0.0f;
        }
    } else if (random_tensor.dtype() == DType::Float64) {
        double* rand_data = static_cast<double*>(random_ptr);
        double* mask_out = static_cast<double*>(mask_ptr);
        for (size_t i = 0; i < numel; ++i) {
            mask_out[i] = rand_data[i] > p_ ? 1.0 : 0.0;
        }
    } else {
        throw std::runtime_error("AlphaDropout only supports Float16, Float32 and Float64 dtypes");
    }

    // Transfer mask to target device if needed
    auto mask_data = (input.tensor().device().type == Device::Type::CPU) ?
                     mask_data_cpu : mask_data_cpu.to(input.tensor().device());

    // Create masked input: set dropped elements to alpha_p
    // masked_input = input * mask + alpha_p * (1 - mask)
    auto ones_tensor = ones(shape_vec, input.tensor().dtype(), input.tensor().device());
    auto inverted_mask = sub(ones_tensor, mask_data);
    auto alpha_p_tensor = full(shape_vec, static_cast<float>(alpha_p),
                               input.tensor().dtype(), input.tensor().device());

    auto kept_part = mul(input.tensor(), mask_data);
    auto dropped_part = mul(alpha_p_tensor, inverted_mask);
    auto masked_input = add(kept_part, dropped_part);

    // Apply affine transformation: output = a * masked_input + b
    auto a_tensor = full(shape_vec, static_cast<float>(a),
                        input.tensor().dtype(), input.tensor().device());
    auto b_tensor = full(shape_vec, static_cast<float>(b),
                        input.tensor().dtype(), input.tensor().device());
    auto output_tensor = add(mul(masked_input, a_tensor), b_tensor);

    // Create output variable
    Variable output(output_tensor, input.requires_grad());

    // Set up autograd if input requires grad
    if (input.requires_grad()) {
        // Create autograd function with mask and affine parameters
        auto alpha_dropout_fn = std::make_shared<AlphaDropoutBackward>(mask_data, a, b);

        // Track input variable for gradient accumulation
        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        alpha_dropout_fn->set_input_variables(input_vars);

        // Set up backward graph - link to input's grad_fn if it exists
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        alpha_dropout_fn->set_next_functions(next_funcs);

        // Set gradient function on output
        output.set_grad_fn(alpha_dropout_fn);
    }

    return output;
}

} // namespace tenzor::nn
