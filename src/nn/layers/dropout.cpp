#include "tenzor/nn/layers/dropout.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/function.hpp"
#include <algorithm>
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

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        if (grad_outputs.size() != 1) {
            throw std::invalid_argument("DropoutBackward expects 1 gradient output");
        }

        // Gradient: grad_input = grad_output * mask * scale
        // mask is a constant (no grad tracking needed)
        auto& grad = grad_outputs[0];
        Variable mask_var(mask_, false);
        auto grad_masked = grad * mask_var;
        auto grad_input = grad_masked * scale_;
        return {grad_input};
    }

private:
    Tensor mask_;
    double scale_;
};

// Element-wise Dropout
Dropout::Dropout(double p) : p_(p) {
    // p == 1.0 is rejected because the inverted-dropout scaling factor
    // 1/(1-p) would divide by zero. PyTorch silently allows this and the
    // forward returns NaN; we reject up front to surface the bug at
    // construction time.
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
        // Generate the random tensor and run the threshold compare in Float32,
        // then cast the resulting 0/1 mask to the input dtype. Generating the
        // randoms directly in a half dtype quantizes both the uniform draw and
        // the threshold compare, biasing the realised keep probability away from
        // (1 - p). Float32 keeps the Bernoulli draw faithful (matches the
        // VariationalDropout pattern below).
        const auto in_dtype = input.tensor().dtype();
        const auto in_device = input.tensor().device();
        auto random_tensor = rand(shape_vec, DType::Float32, in_device);

        // Create binary mask using device-native comparison: mask = (random > p)
        auto threshold = full(shape_vec, static_cast<float>(p_),
                             DType::Float32, in_device);
        auto mask_bool = gt(random_tensor, threshold);
        auto ones_tensor = ones(shape_vec, DType::Float32, in_device);
        auto zeros_tensor = zeros(shape_vec, DType::Float32, in_device);
        auto mask_f32 = where(mask_bool, ones_tensor, zeros_tensor);
        mask_data = (in_dtype == DType::Float32) ? mask_f32 : mask_f32.to(in_dtype);
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
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("Dropout2d probability must be in [0, 1]");
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

    // Generate randoms + threshold compare in Float32, then cast the 0/1 mask
    // to the input dtype. A half-precision draw/compare biases the keep
    // probability; Float32 keeps the Bernoulli draw faithful. (Mirrors Dropout.)
    const auto in_dtype = input.tensor().dtype();
    const auto in_device = input.tensor().device();
    auto random_tensor = rand(mask_shape, DType::Float32, in_device);

    // Create binary mask using device-native comparison: mask = (random > p)
    auto threshold = full(mask_shape, static_cast<float>(p_),
                         DType::Float32, in_device);
    auto mask_bool = gt(random_tensor, threshold);
    auto mask_ones = ones(mask_shape, DType::Float32, in_device);
    auto mask_zeros = zeros(mask_shape, DType::Float32, in_device);
    auto channel_mask_f32 = where(mask_bool, mask_ones, mask_zeros);
    auto channel_mask = (in_dtype == DType::Float32)
                            ? channel_mask_f32 : channel_mask_f32.to(in_dtype);

    // Expand channel mask to full input shape via broadcasting
    // mask_shape has trailing 1s (e.g., [N,C,1,1]) so expand() broadcasts to full [N,C,H,W]
    double scale = 1.0 / (1.0 - p_);
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    auto expanded_mask_final = expand(channel_mask, shape_vec);

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

// Channel-wise Dropout3d (volumetric analogue of Dropout2d)
Dropout3d::Dropout3d(double p) : p_(p) {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("Dropout3d probability must be in [0, 1]");
    }
}

auto Dropout3d::forward_impl(const Variable& input) -> Variable {
    if (!is_training()) {
        return input;
    }
    if (p_ == 0.0) {
        return input;
    }

    auto shape = input.tensor().shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());

    if (p_ == 1.0) {
        auto output_tensor = zeros(shape_vec, input.tensor().dtype(), input.tensor().device());
        return Variable(output_tensor, input.requires_grad());
    }

    if (shape.size() < 3) {
        throw std::invalid_argument(
            "Dropout3d requires at least 3D input (C, D, H) or (N, C, D, H, W)");
    }

    // Mask shape: broadcast across spatial (D, H, W) but vary per (N, C).
    std::vector<int64_t> mask_shape;
    if (shape.size() == 5) {
        mask_shape = {shape[0], shape[1], 1, 1, 1};     // [N, C, 1, 1, 1]
    } else if (shape.size() == 4) {
        mask_shape = {shape[0], 1, 1, 1};               // [C, 1, 1, 1]
    } else if (shape.size() == 3) {
        mask_shape = {shape[0], 1, 1};                  // [C, 1, 1]
    } else {
        throw std::invalid_argument("Dropout3d input must be 3D, 4D or 5D");
    }

    // Generate randoms + threshold compare in Float32, then cast the 0/1 mask
    // to the input dtype. A half-precision draw/compare biases the keep
    // probability; Float32 keeps the Bernoulli draw faithful. (Mirrors Dropout.)
    const auto in_dtype = input.tensor().dtype();
    const auto in_device = input.tensor().device();
    auto random_tensor = rand(mask_shape, DType::Float32, in_device);
    auto threshold = full(mask_shape, static_cast<float>(p_),
                         DType::Float32, in_device);
    auto mask_bool = gt(random_tensor, threshold);
    auto mask_ones = ones(mask_shape, DType::Float32, in_device);
    auto mask_zeros = zeros(mask_shape, DType::Float32, in_device);
    auto channel_mask_f32 = where(mask_bool, mask_ones, mask_zeros);
    auto channel_mask = (in_dtype == DType::Float32)
                            ? channel_mask_f32 : channel_mask_f32.to(in_dtype);

    double scale = 1.0 / (1.0 - p_);
    auto expanded_mask_final = expand(channel_mask, shape_vec);

    auto scale_tensor = full(shape_vec, static_cast<float>(scale),
                            input.tensor().dtype(), input.tensor().device());
    auto output_tensor = mul(mul(input.tensor(), expanded_mask_final), scale_tensor);

    Variable output(output_tensor, input.requires_grad());

    if (input.requires_grad()) {
        auto dropout_fn = std::make_shared<DropoutBackward>(expanded_mask_final, scale);
        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        dropout_fn->set_input_variables(input_vars);
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        dropout_fn->set_next_functions(next_funcs);
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

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        if (grad_outputs.size() != 1) {
            throw std::invalid_argument("AlphaDropoutBackward expects 1 gradient output");
        }

        // Gradient: grad_input = grad_output * a * mask
        // mask and a are constants (no grad tracking needed)
        auto& grad = grad_outputs[0];
        Variable mask_var(mask_, false);
        auto grad_scaled = grad * a_;
        auto grad_input = grad_scaled * mask_var;
        return {grad_input};
    }

private:
    Tensor mask_;
    double a_;
    double b_;
};

// Alpha Dropout (for SELU networks)
AlphaDropout::AlphaDropout(double p) : p_(p) {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("AlphaDropout probability must be in [0, 1]");
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

    // Affine transformation parameters to maintain mean=0, var=1.
    // At p == 1.0 every unit is dropped: a = sqrt(0) = 0 and the b expression
    // would divide by (1 - p) == 0. Guard it so the degenerate all-dropped
    // output is the finite constant a*alpha_p + b = 0 (zeros) instead of NaN.
    const double keep = 1.0 - p_;
    const double a = std::sqrt(keep * (1.0 + p_ * alpha_p * alpha_p));
    const double b = (keep > 0.0) ? (-a * alpha_p * p_ / keep) : 0.0;

    // Generate random mask directly on target device
    auto shape_span = input.tensor().shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());

    // Generate randoms + threshold compare in Float32, then cast the 0/1 mask
    // to the input dtype. A half-precision draw/compare biases the keep
    // probability; Float32 keeps the Bernoulli draw faithful. (Mirrors Dropout.)
    const auto in_dtype = input.tensor().dtype();
    const auto in_device = input.tensor().device();
    auto random_tensor = rand(shape_vec, DType::Float32, in_device);

    // Create binary mask using device-native comparison: mask = (random > p)
    auto threshold = full(shape_vec, static_cast<float>(p_),
                         DType::Float32, in_device);
    auto mask_bool = gt(random_tensor, threshold);
    auto mask_ones = ones(shape_vec, DType::Float32, in_device);
    auto mask_zeros = zeros(shape_vec, DType::Float32, in_device);
    auto mask_f32 = where(mask_bool, mask_ones, mask_zeros);
    auto mask_data = (in_dtype == DType::Float32) ? mask_f32 : mask_f32.to(in_dtype);

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

// Variational Dropout (Gal & Ghahramani 2016)
VariationalDropout::VariationalDropout(double p) : p_(p) {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("VariationalDropout probability must be in [0, 1]");
    }
}

auto VariationalDropout::reset_mask() -> void {
    mask_valid_ = false;
}

auto VariationalDropout::forward_impl(const Variable& input) -> Variable {
    if (!is_training()) {
        return input;
    }

    auto shape_span = input.tensor().shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());

    if (p_ == 1.0) {
        auto output_tensor = zeros(shape_vec, input.tensor().dtype(), input.tensor().device());
        return Variable(output_tensor, input.requires_grad());
    }

    if (p_ == 0.0) {
        return input;
    }

    double scale = 1.0 / (1.0 - p_);

    // A cached mask is reused across calls (same mask per sequence, by design).
    // But if the current input is no longer broadcast-compatible with the cached
    // mask (e.g. a different batch size or feature width on a later call), the
    // mul() below would either throw or silently misbroadcast. Detect that and
    // regenerate the mask for the new shape rather than reusing a stale one.
    if (mask_valid_) {
        const auto& mask_shape = mask_.shape();
        // NumPy/Tenzor broadcasting: align shapes from the trailing dimension;
        // each pair of dims must be equal or one of them must be 1.
        bool compatible = true;
        const int64_t mn = static_cast<int64_t>(mask_shape.size());
        const int64_t in = static_cast<int64_t>(shape_vec.size());
        const int64_t n = std::max(mn, in);
        for (int64_t i = 1; i <= n; ++i) {
            int64_t md = (i <= mn) ? mask_shape[mn - i] : 1;
            int64_t id = (i <= in) ? shape_vec[in - i] : 1;
            if (md != id && md != 1 && id != 1) {
                compatible = false;
                break;
            }
        }
        if (!compatible) {
            mask_valid_ = false;
        }
    }

    // Generate mask on first call after reset (or first call ever), or when the
    // cached mask was invalidated above due to a shape change.
    if (!mask_valid_) {
        // For 3D input (T, B, F): mask shape is (1, B, F) to broadcast over time
        // For 2D input (B, F): mask shape is (B, F)
        std::vector<int64_t> mask_shape;
        if (shape_vec.size() >= 3) {
            mask_shape.push_back(1);
            for (size_t i = 1; i < shape_vec.size(); ++i) {
                mask_shape.push_back(shape_vec[i]);
            }
        } else {
            mask_shape = shape_vec;
        }

        // Generate Bernoulli mask in Float32 regardless of the parameter dtype,
        // then cast to the target dtype. Use tensor ops so this works on any
        // device (CPU/CUDA/etc.) — raw data_ptr access on GPU tensors segfaults.
        const auto target_dtype = input.tensor().dtype();
        const auto target_device = input.tensor().device();
        auto rand_vals = rand(mask_shape, DType::Float32, target_device);
        const float thresh = static_cast<float>(1.0 - p_);
        auto thresh_tensor = full(mask_shape, thresh, DType::Float32, target_device);
        auto mask_f32 = tenzor::lt(rand_vals, thresh_tensor).to(DType::Float32);
        mask_ = (target_dtype == DType::Float32) ? mask_f32 : mask_f32.to(target_dtype);

        mask_valid_ = true;
    }

    // Apply mask with scaling: output = input * mask * scale
    auto masked = mul(input.tensor(), mask_);
    auto scale_shape = shape_vec;
    auto output_tensor = mul(masked,
        full(scale_shape, static_cast<float>(scale),
             input.tensor().dtype(), input.tensor().device()));

    auto output = Variable(output_tensor, input.requires_grad());

    // Set up autograd if needed
    if (input.requires_grad()) {
        auto dropout_fn = std::make_shared<DropoutBackward>(mask_, scale);

        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        dropout_fn->set_input_variables(input_vars);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        dropout_fn->set_next_functions(next_funcs);

        output.set_grad_fn(dropout_fn);
    }

    return output;
}

} // namespace tenzor::nn
