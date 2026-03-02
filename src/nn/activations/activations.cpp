#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>
#include <iostream>

namespace tenzor::nn {

// Backward function for ReLU
class ReLUBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("ReLUBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output = grad_outputs[0];
        const auto& input = saved_tensors()[0];

        // Use the backend's native relu_backward kernel for efficiency
        // This avoids CPU transfers that happen when using comparison + dtype conversion
        std::vector<Tensor> backward_inputs = {grad_output, input};
        auto grad_input = dispatch(OpId::ReLUBackward, backward_inputs)[0];

        return {grad_input};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // ReLU backward: grad_input = grad_output * (input > 0)
        // The mask is non-differentiable, compute at Tensor level
        const auto& input = saved_tensors()[0];
        std::vector<Tensor> mask_inputs = {ones(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                                                 input.dtype(), input.device()), input};
        auto mask = dispatch(OpId::ReLUBackward, mask_inputs)[0];
        Variable mask_var(mask, false);
        return {grad_outputs[0] * mask_var};
    }
};

// Backward function for Sigmoid
class SigmoidBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("SigmoidBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output = grad_outputs[0];
        auto& output = saved_tensors()[0];  // sigmoid(x)

        // d_sigmoid/dx = sigmoid(x) * (1 - sigmoid(x))
        auto shape_vec = std::vector<int64_t>(output.shape().begin(), output.shape().end());
        auto one_tensor = ones(shape_vec, output.dtype(), output.device());
        auto one_minus_output = one_tensor - output;
        std::vector<Tensor> result;
        result.push_back(grad_output * output * one_minus_output);
        return result;
    }
};

// Backward function for Tanh
class TanhBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("TanhBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output = grad_outputs[0];
        auto& output = saved_tensors()[0];  // tanh(x)

        // d_tanh/dx = 1 - tanh^2(x)
        auto output_squared = output * output;
        auto shape_vec = std::vector<int64_t>(output.shape().begin(), output.shape().end());
        auto one_tensor = ones(shape_vec, output.dtype(), output.device());
        auto one_minus_sq = one_tensor - output_squared;
        std::vector<Tensor> result;
        result.push_back(grad_output * one_minus_sq);
        return result;
    }
};

// Backward function for GELU
class GeLUBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("GeLUBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output = grad_outputs[0];
        const auto& input = saved_tensors()[0];

        // Use the fused gelu_backward kernel which computes the entire derivative
        // in Float32 internally. This avoids Float16 intermediate overflow that occurs
        // when computing the derivative with individual tensor operations (e.g., x*x
        // overflows Float16 for |x| > 256, leading to inf and then 0*inf = NaN).
        std::vector<Tensor> backward_inputs = {grad_output, input};
        auto result_tensor = dispatch(OpId::GeluBackward, backward_inputs)[0];

        std::vector<Tensor> result;
        result.push_back(result_tensor);
        return result;
    }
};

// Backward function for Swish
class SwishBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("SwishBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output = grad_outputs[0];
        auto& input = saved_tensors()[0];      // original input x

        // Swish(x) = x * sigmoid(x)
        // d(Swish)/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //             = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

        auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

        // Compute sigmoid(x)
        std::vector<Tensor> sig_vec = {input};
        auto sigmoid_x = dispatch(OpId::Sigmoid, sig_vec)[0];

        auto one_tensor = ones(shape_vec, input.dtype(), input.device());
        auto one_minus_sigmoid = one_tensor - sigmoid_x;

        // d(Swish)/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        auto grad_swish = sigmoid_x * (one_tensor + input * one_minus_sigmoid);

        std::vector<Tensor> result;
        result.push_back(grad_output * grad_swish);
        return result;
    }
};

// Backward function for ELU
class ELUBackward : public Function {
public:
    ELUBackward(double alpha) : alpha_(alpha) {}

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("ELUBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output = grad_outputs[0];
        auto& output = saved_tensors()[0];  // ELU output

        // ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
        // d(ELU)/dx = 1 if x > 0, alpha * exp(x) = output + alpha if x <= 0

        auto shape_vec = std::vector<int64_t>(output.shape().begin(), output.shape().end());
        auto zero = zeros(shape_vec, output.dtype(), output.device());
        auto one_tensor = ones(shape_vec, output.dtype(), output.device());
        auto alpha_tensor = one_tensor * alpha_;

        // Piecewise gradient: 1 for x>0, output+alpha for x<=0
        auto condition = gt(output, zero);
        auto grad_elu = where(condition, one_tensor, output + alpha_tensor);

        std::vector<Tensor> result;
        result.push_back(grad_output * grad_elu);
        return result;
    }

private:
    double alpha_;
};

// Backward function for SELU
class SELUBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("SELUBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output = grad_outputs[0];
        auto& output = saved_tensors()[0];

        // SELU constants
        const double scale = 1.0507009873554804934193349852946;
        const double alpha = 1.6732632423543772848170429916717;

        // SELU(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
        // d(SELU)/dx = scale if x > 0, scale * alpha * exp(x) if x <= 0

        auto shape_vec = std::vector<int64_t>(output.shape().begin(), output.shape().end());
        auto zero = zeros(shape_vec, output.dtype(), output.device());
        auto scale_tensor = ones(shape_vec, output.dtype(), output.device()) * scale;
        auto scale_alpha_tensor = ones(shape_vec, output.dtype(), output.device()) * (scale * alpha);

        // Piecewise gradient: scale for x>0, output+scale*alpha for x<=0
        auto condition = gt(output, zero);
        auto grad_selu = where(condition, scale_tensor, output + scale_alpha_tensor);

        std::vector<Tensor> result;
        result.push_back(grad_output * grad_selu);
        return result;
    }
};

// Backward function for Mish
class MishBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("MishBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output = grad_outputs[0];
        auto& input = saved_tensors()[0];

        // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

        // Compute softplus(x) = ln(1 + exp(x))
        std::vector<Tensor> sp_vec = {input};
        auto softplus_x = dispatch(OpId::Softplus, sp_vec)[0];

        // Compute tanh(softplus(x))
        std::vector<Tensor> tanh_vec = {softplus_x};
        auto tanh_sp = dispatch(OpId::Tanh, tanh_vec)[0];

        // Compute sigmoid(x)
        std::vector<Tensor> sig_vec = {input};
        auto sigmoid_x = dispatch(OpId::Sigmoid, sig_vec)[0];

        auto one_tensor = ones(shape_vec, input.dtype(), input.device());

        // d(Mish)/dx = tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x)
        //            = tanh(softplus(x)) + x * (1 - tanh²(softplus(x))) * sigmoid(x)
        auto tanh_sp_sq = tanh_sp * tanh_sp;
        auto sech_sq = one_tensor - tanh_sp_sq;
        auto grad_mish = tanh_sp + input * sech_sq * sigmoid_x;

        std::vector<Tensor> result;
        result.push_back(grad_output * grad_mish);
        return result;
    }
};

// Module implementations
auto ReLU::forward_impl(const Variable& input) -> Variable {
    return relu(input);
}

auto ReLU6::forward_impl(const Variable& input) -> Variable {
    // ReLU6(x) = min(max(0, x), 6)
    return clamp(relu(input), 0.0f, 6.0f);
}

LeakyReLU::LeakyReLU(double negative_slope) : negative_slope_(negative_slope) {}

auto LeakyReLU::forward_impl(const Variable& input) -> Variable {
    return nn::leaky_relu(input, negative_slope_);
}

auto Sigmoid::forward_impl(const Variable& input) -> Variable {
    return nn::sigmoid(input);
}

auto Tanh::forward_impl(const Variable& input) -> Variable {
    return nn::tanh(input);
}

Softmax::Softmax(int64_t dim) : dim_(dim) {}

auto Softmax::forward_impl(const Variable& input) -> Variable {
    return tenzor::softmax(input, dim_);
}

// Functional implementations with autograd support
auto relu(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = dispatch(OpId::ReLU, inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = dispatch(OpId::ReLU, inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<ReLUBackward>();
    // Save only input - relu_backward kernel only needs input
    grad_fn->save_for_backward({input.tensor()});

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto sigmoid(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = dispatch(OpId::Sigmoid, inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = dispatch(OpId::Sigmoid, inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<SigmoidBackward>();
    grad_fn->save_for_backward({result_tensor});  // Save output for backward

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto tanh(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = dispatch(OpId::Tanh, inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = dispatch(OpId::Tanh, inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<TanhBackward>();
    grad_fn->save_for_backward({result_tensor});  // Save output for backward

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto leaky_relu(const Variable& input, double negative_slope) -> Variable {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Alpha, static_cast<double>(negative_slope));
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = dispatch(OpId::LeakyReLU, inputs, attrs)[0];
    return Variable(result, input.requires_grad());
}

auto gelu(const Variable& input, const std::string& approximate) -> Variable {
    // Tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    if (approximate == "tanh") {
        // Compute using existing ops for autograd support
        auto x = input;
        auto x_cubed = x * x * x;
        // Scale factors as scalar tensors
        Tensor coef_t = full({1}, 0.044715f, input.tensor().dtype(), input.tensor().device());
        Tensor sqrt_2_pi_t = full({1}, static_cast<float>(std::sqrt(2.0 / M_PI)),
                                  input.tensor().dtype(), input.tensor().device());
        Tensor half_t = full({1}, 0.5f, input.tensor().dtype(), input.tensor().device());
        Tensor one_t = full({1}, 1.0f, input.tensor().dtype(), input.tensor().device());

        Variable coef(coef_t, false);
        Variable sqrt_2_pi(sqrt_2_pi_t, false);
        Variable half(half_t, false);
        Variable one(one_t, false);

        auto inner = sqrt_2_pi * (x + coef * x_cubed);
        auto tanh_val = nn::tanh(inner);
        return half * x * (one + tanh_val);
    }

    // Exact GELU via erf dispatch
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = dispatch(OpId::Gelu, inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = dispatch(OpId::Gelu, inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<GeLUBackward>();
    // Save BOTH input and output
    grad_fn->save_for_backward({input.tensor(), result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto softmax(const Variable& input, int64_t dim) -> Variable {
    // Use autograd-aware version from tenzor namespace
    return tenzor::softmax(input, dim);
}

auto log_softmax(const Variable& input, int64_t dim) -> Variable {
    // Use autograd-aware version from tenzor namespace
    return tenzor::log_softmax(input, dim);
}

LogSoftmax::LogSoftmax(int64_t dim) : dim_(dim) {}

auto LogSoftmax::forward_impl(const Variable& input) -> Variable {
    return tenzor::log_softmax(input, dim_);
}

GELU::GELU(const std::string& approximate) : approximate_(approximate) {
    if (approximate != "none" && approximate != "tanh") {
        throw std::invalid_argument("GELU approximate must be 'none' or 'tanh', got '" + approximate + "'");
    }
}

auto GELU::forward_impl(const Variable& input) -> Variable {
    return gelu(input, approximate_);
}

auto ELU::forward_impl(const Variable& input) -> Variable {
    return elu(input, alpha_);
}

ELU::ELU(double alpha) : alpha_(alpha) {}

auto elu(const Variable& input, double alpha) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        NewOpAttributes attrs;
        attrs.set(AttrKey::Alpha, static_cast<double>(alpha));
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = dispatch(OpId::Elu, inputs, attrs)[0];
        return Variable(result, false);
    }

    // Compute forward
    NewOpAttributes attrs;
    attrs.set(AttrKey::Alpha, static_cast<double>(alpha));
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = dispatch(OpId::Elu, inputs_vec, attrs)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<ELUBackward>(alpha);
    grad_fn->save_for_backward({result_tensor});  // Save output for backward

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto SELU::forward_impl(const Variable& input) -> Variable {
    return nn::selu(input);
}

auto selu(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = dispatch(OpId::Selu, inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = dispatch(OpId::Selu, inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<SELUBackward>();
    grad_fn->save_for_backward({result_tensor});  // Save output for backward

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto Swish::forward_impl(const Variable& input) -> Variable {
    return swish(input);
}

auto swish(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = dispatch(OpId::Swish, inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = dispatch(OpId::Swish, inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<SwishBackward>();
    grad_fn->save_for_backward({input.tensor()});  // Save input for backward

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto Mish::forward_impl(const Variable& input) -> Variable {
    return nn::mish(input);
}

auto mish(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = dispatch(OpId::Mish, inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = dispatch(OpId::Mish, inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<MishBackward>();
    grad_fn->save_for_backward({input.tensor()});  // Save input for backward

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// In-place activation functions
auto relu_(Tensor& input) -> Tensor& {
    if (!input.is_contiguous()) {
        throw std::runtime_error("In-place relu requires contiguous tensor");
    }

    dispatch_inplace(OpId::ReLUInplace, input, std::span<const Tensor>{});

    return input;
}

auto sigmoid_(Tensor& input) -> Tensor& {
    if (!input.is_contiguous()) {
        throw std::runtime_error("In-place sigmoid requires contiguous tensor");
    }

    dispatch_inplace(OpId::SigmoidInplace, input, std::span<const Tensor>{});

    return input;
}

auto tanh_(Tensor& input) -> Tensor& {
    if (!input.is_contiguous()) {
        throw std::runtime_error("In-place tanh requires contiguous tensor");
    }

    dispatch_inplace(OpId::TanhInplace, input, std::span<const Tensor>{});

    return input;
}

auto leaky_relu_(Tensor& input, double negative_slope) -> Tensor& {
    if (!input.is_contiguous()) {
        throw std::runtime_error("In-place leaky_relu requires contiguous tensor");
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::Negative_slope, negative_slope);
    dispatch_inplace(OpId::LeakyReLUInplace, input, std::span<const Tensor>{}, attrs);

    return input;
}

auto gelu_(Tensor& input) -> Tensor& {
    if (!input.is_contiguous()) {
        throw std::runtime_error("In-place gelu requires contiguous tensor");
    }

    dispatch_inplace(OpId::GeluInplace, input, std::span<const Tensor>{});

    return input;
}

// PReLU implementation
PReLU::PReLU(int64_t num_parameters, double init)
    : num_parameters_(num_parameters) {
    auto weight_tensor = tenzor::full({num_parameters}, init, DType::Float32, Device::cpu());
    Variable weight_var(weight_tensor, true);
    register_parameter("weight", std::move(weight_var));
}

auto PReLU::forward_impl(const Variable& input) -> Variable {
    // PReLU(x) = max(0, x) + weight * min(0, x)
    // = relu(x) + weight * (x - relu(x))  [avoiding min for autograd]
    auto& weight = *parameters_["weight"];
    auto relu_x = relu(input);
    auto neg_part = input - relu_x;  // min(0, x)

    // weight has shape [num_parameters], broadcasts with input
    return relu_x + weight * neg_part;
}

auto Hardswish::forward_impl(const Variable& input) -> Variable {
    // Hardswish(x) = x * clamp(x + 3, 0, 6) / 6
    // Using tensor-level clamp since Variable doesn't have one
    auto x_plus_3 = input + 3.0f;
    auto clamped = Variable(
        tenzor::clamp(x_plus_3.tensor(), 0.0f, 6.0f),
        input.requires_grad());
    return input * clamped / 6.0f;
}

auto Hardsigmoid::forward_impl(const Variable& input) -> Variable {
    // Hardsigmoid(x) = clamp(x + 3, 0, 6) / 6
    auto x_plus_3 = input + 3.0f;
    auto clamped = Variable(
        tenzor::clamp(x_plus_3.tensor(), 0.0f, 6.0f),
        input.requires_grad());
    return clamped / 6.0f;
}

// Functional Hardswish
auto hardswish(const Variable& input) -> Variable {
    auto x_plus_3 = input + 3.0f;
    auto clamped = Variable(
        tenzor::clamp(x_plus_3.tensor(), 0.0f, 6.0f),
        input.requires_grad());
    return input * clamped / 6.0f;
}

// Functional Hardsigmoid
auto hardsigmoid(const Variable& input) -> Variable {
    auto x_plus_3 = input + 3.0f;
    auto clamped = Variable(
        tenzor::clamp(x_plus_3.tensor(), 0.0f, 6.0f),
        input.requires_grad());
    return clamped / 6.0f;
}

// GLU module
auto GLU::forward_impl(const Variable& input) -> Variable {
    return glu(input, dim_);
}

// Functional GLU
auto glu(const Variable& input, int64_t dim) -> Variable {
    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("GLU: dim " + std::to_string(dim) + " out of range for " + std::to_string(ndim) + "D input");
    }
    if (shape[dim] % 2 != 0) {
        throw std::invalid_argument("GLU: input size along dim " + std::to_string(dim) + " must be even, got " + std::to_string(shape[dim]));
    }
    int64_t half = shape[dim] / 2;
    // Split input into two halves along dim
    auto a = Variable(input.tensor().slice(dim, 0, half), input.requires_grad());
    auto b = Variable(input.tensor().slice(dim, half, shape[dim]), input.requires_grad());
    // GLU(a, b) = a * sigmoid(b)
    return a * nn::sigmoid(b);
}

} // namespace tenzor::nn
