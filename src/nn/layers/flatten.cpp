#include "tenzor/nn/layers/flatten.hpp"
#include "tenzor/autograd/function.hpp"
#include <stdexcept>

namespace tenzor::nn {

// Flatten backward function
class FlattenBackward : public Function {
public:
    FlattenBackward(std::vector<int64_t> input_shape)
        : input_shape_(std::move(input_shape)) {}

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("FlattenBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        if (grad_outputs.size() != 1) {
            throw std::invalid_argument("FlattenBackward expects 1 gradient output");
        }

        // Reshape gradient back to original input shape
        auto grad_input = grad_outputs[0].reshape(input_shape_);

        std::vector<Tensor> result;
        result.push_back(grad_input);
        return result;
    }

private:
    std::vector<int64_t> input_shape_;
};

Flatten::Flatten(int64_t start_dim, int64_t end_dim)
    : start_dim_(start_dim), end_dim_(end_dim) {}

auto Flatten::forward_impl(const Variable& input) -> Variable {
    auto shape = input.tensor().shape();
    auto ndim = static_cast<int64_t>(shape.size());

    // Normalize negative dimensions
    int64_t start = start_dim_ < 0 ? ndim + start_dim_ : start_dim_;
    int64_t end = end_dim_ < 0 ? ndim + end_dim_ : end_dim_;

    if (start < 0 || start >= ndim) {
        throw std::invalid_argument("start_dim out of range");
    }
    if (end < 0 || end >= ndim) {
        throw std::invalid_argument("end_dim out of range");
    }
    if (start > end) {
        throw std::invalid_argument("start_dim must be <= end_dim");
    }

    // Compute new shape
    std::vector<int64_t> new_shape;

    // Keep dimensions before start_dim
    for (int64_t i = 0; i < start; ++i) {
        new_shape.push_back(shape[i]);
    }

    // Flatten dimensions from start_dim to end_dim (inclusive)
    int64_t flattened_size = 1;
    for (int64_t i = start; i <= end; ++i) {
        flattened_size *= shape[i];
    }
    new_shape.push_back(flattened_size);

    // Keep dimensions after end_dim
    for (int64_t i = end + 1; i < ndim; ++i) {
        new_shape.push_back(shape[i]);
    }

    // Reshape tensor
    auto output_tensor = input.tensor().reshape(new_shape);

    // Create output variable
    Variable output(output_tensor, input.requires_grad());

    // Set up autograd if input requires grad
    if (input.requires_grad()) {
        // Save original input shape for backward
        std::vector<int64_t> input_shape_vec(shape.begin(), shape.end());

        // Create backward function
        auto flatten_fn = std::make_shared<FlattenBackward>(input_shape_vec);

        // Track input variable for gradient accumulation
        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        flatten_fn->set_input_variables(input_vars);

        // Set up backward graph - link to input's grad_fn if it exists
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        flatten_fn->set_next_functions(next_funcs);

        // Set gradient function on output
        output.set_grad_fn(flatten_fn);
    }

    return output;
}

} // namespace tenzor::nn
