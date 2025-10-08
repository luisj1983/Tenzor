#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/backend/dispatch.hpp"

namespace tenzor::nn {

// Module implementations
auto ReLU::forward(const Variable& input) -> Variable {
    return relu(input);
}

auto Sigmoid::forward(const Variable& input) -> Variable {
    return sigmoid(input);
}

auto Tanh::forward(const Variable& input) -> Variable {
    return tanh(input);
}

auto Softmax::forward(const Variable& input) -> Variable {
    return softmax(input, dim_);
}

// Functional implementations (stubs - will be dispatched to backends)
auto relu(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("relu", inputs)[0];
    return Variable(result, input.requires_grad());
}

auto sigmoid(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("sigmoid", inputs)[0];
    return Variable(result, input.requires_grad());
}

auto tanh(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("tanh", inputs)[0];
    return Variable(result, input.requires_grad());
}

auto gelu(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("gelu", inputs)[0];
    return Variable(result, input.requires_grad());
}

auto softmax(const Variable& input, int64_t dim) -> Variable {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("softmax", inputs, attrs)[0];
    return Variable(result, input.requires_grad());
}

} // namespace tenzor::nn
