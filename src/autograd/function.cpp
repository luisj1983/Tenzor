#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/math.hpp"

namespace tenzor {

auto Function::set_next_functions(std::vector<std::shared_ptr<Function>> funcs) -> void {
    next_functions_ = std::move(funcs);
}

auto Function::next_functions() const -> const std::vector<std::shared_ptr<Function>>& {
    return next_functions_;
}

auto Function::save_for_backward(std::vector<Tensor> tensors) -> void {
    saved_tensors_ = std::move(tensors);
}

auto Function::saved_tensors() const -> const std::vector<Tensor>& {
    return saved_tensors_;
}

// AddBackward implementation
auto AddBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = add(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto AddBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {grad_outputs[0], grad_outputs[0]};
}

// SubBackward implementation
auto SubBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = sub(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto SubBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(a-b)/da = 1, d(a-b)/db = -1
    return {grad_outputs[0], grad_outputs[0] * (-1.0f)};
}

// MulBackward implementation
auto MulBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    saved_tensors_ = {inputs[0].tensor(), inputs[1].tensor()};
    auto result = mul(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto MulBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(a*b)/da = b, d(a*b)/db = a
    return {
        mul(grad_outputs[0], saved_tensors_[1]),
        mul(grad_outputs[0], saved_tensors_[0])
    };
}

// MatMulBackward implementation
auto MatMulBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    saved_tensors_ = {inputs[0].tensor(), inputs[1].tensor()};
    auto result = matmul(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto MatMulBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // TODO: Implement proper matmul backward
    return {saved_tensors_[0], saved_tensors_[1]};
}

} // namespace tenzor
