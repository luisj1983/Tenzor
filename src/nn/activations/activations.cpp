#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/creation.hpp"
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
        const auto& output = saved_tensors()[1];  // Use output tensor for dtype reference

        // d_relu/dx = 1 if x > 0, else 0
        // grad_input = grad_output * (input > 0)

        // Create zero tensor for comparison
        auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
        auto zero_tensor = zeros(shape_vec, input.dtype(), input.device());

        // Perform comparison to get boolean mask
        auto mask = input > zero_tensor;  // Returns Bool tensor

        // Convert mask to same dtype as grad_output for multiplication
        auto mask_float = mask.to(grad_output.dtype());

        // Apply mask to gradient
        std::vector<Tensor> result;
        result.push_back(grad_output * mask_float);
        return result;
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

// Module implementations
auto ReLU::forward(const Variable& input) -> Variable {
    return relu(input);
}

LeakyReLU::LeakyReLU(double negative_slope) : negative_slope_(negative_slope) {}

auto LeakyReLU::forward(const Variable& input) -> Variable {
    return leaky_relu(input, negative_slope_);
}

auto Sigmoid::forward(const Variable& input) -> Variable {
    return sigmoid(input);
}

auto Tanh::forward(const Variable& input) -> Variable {
    return tanh(input);
}

Softmax::Softmax(int64_t dim) : dim_(dim) {}

auto Softmax::forward(const Variable& input) -> Variable {
    return tenzor::softmax(input, dim_);
}

// Functional implementations with autograd support
auto relu(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = Dispatcher::dispatch("relu", inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = Dispatcher::dispatch("relu", inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<ReLUBackward>();
    // Save BOTH input and output to avoid dtype issues during checkpoint recomputation
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

auto sigmoid(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = Dispatcher::dispatch("sigmoid", inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = Dispatcher::dispatch("sigmoid", inputs_vec)[0];

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
        auto result = Dispatcher::dispatch("tanh", inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = Dispatcher::dispatch("tanh", inputs_vec)[0];

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
    OpAttributes attrs;
    attrs["alpha"] = std::to_string(static_cast<float>(negative_slope));
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("leaky_relu", inputs, attrs)[0];
    return Variable(result, input.requires_grad());
}

auto gelu(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("gelu", inputs)[0];
    return Variable(result, input.requires_grad());
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

auto LogSoftmax::forward(const Variable& input) -> Variable {
    return tenzor::log_softmax(input, dim_);
}

auto GELU::forward(const Variable& input) -> Variable {
    return gelu(input);
}

auto ELU::forward(const Variable& input) -> Variable {
    return elu(input, alpha_);
}

ELU::ELU(double alpha) : alpha_(alpha) {}

auto elu(const Variable& input, double alpha) -> Variable {
    OpAttributes attrs;
    attrs["alpha"] = std::to_string(static_cast<float>(alpha));
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("elu", inputs, attrs)[0];
    return Variable(result, input.requires_grad());
}

auto SELU::forward(const Variable& input) -> Variable {
    return selu(input);
}

auto selu(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("selu", inputs)[0];
    return Variable(result, input.requires_grad());
}

auto Swish::forward(const Variable& input) -> Variable {
    return swish(input);
}

auto swish(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("swish", inputs)[0];
    return Variable(result, input.requires_grad());
}

auto Mish::forward(const Variable& input) -> Variable {
    return mish(input);
}

auto mish(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("mish", inputs)[0];
    return Variable(result, input.requires_grad());
}

} // namespace tenzor::nn
