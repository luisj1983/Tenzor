#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/backend/dispatch.hpp"

namespace tenzor {

auto sum(const Variable& input, std::optional<int64_t> dim, bool keepdim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        // No gradient needed, just compute
        return Variable(tenzor::sum(input.tensor(), dim, keepdim), false);
    }

    auto grad_fn = std::make_shared<SumBackward>(dim, keepdim);

    // Save input tensor for backward pass
    grad_fn->save_for_backward({input.tensor()});

    // Set up backward graph
    std::vector<std::shared_ptr<Function>> next_funcs;

    next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf

    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }

    grad_fn->set_input_variables(input_vars);

    // Compute result
    auto result_tensor = tenzor::sum(input.tensor(), dim, keepdim);
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto mean(const Variable& input, std::optional<int64_t> dim, bool keepdim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::mean(input.tensor(), dim, keepdim), false);
    }

    auto grad_fn = std::make_shared<MeanBackward>(dim, keepdim);

    // Save input tensor for backward pass
    grad_fn->save_for_backward({input.tensor()});

    std::vector<std::shared_ptr<Function>> next_funcs;


    next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf


    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }

    grad_fn->set_input_variables(input_vars);

    auto result_tensor = tenzor::mean(input.tensor(), dim, keepdim);
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto log(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::log(input.tensor()), false);
    }

    auto grad_fn = std::make_shared<LogBackward>();

    // Save input tensor for backward pass
    grad_fn->save_for_backward({input.tensor()});

    std::vector<std::shared_ptr<Function>> next_funcs;


    next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf


    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }

    grad_fn->set_input_variables(input_vars);

    auto result_tensor = tenzor::log(input.tensor());
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto exp(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::exp(input.tensor()), false);
    }

    auto grad_fn = std::make_shared<ExpBackward>();

    // Compute result first
    auto result_tensor = tenzor::exp(input.tensor());

    // Save output for backward pass (d/dx exp(x) = exp(x))
    grad_fn->save_for_backward({result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs;


    next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf


    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }

    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto neg(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::neg(input.tensor()), false);
    }

    auto grad_fn = std::make_shared<NegBackward>();

    std::vector<std::shared_ptr<Function>> next_funcs;


    next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf


    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }

    grad_fn->set_input_variables(input_vars);

    auto result_tensor = tenzor::neg(input.tensor());
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto softmax(const Variable& input, int64_t dim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        OpAttributes attrs;
        attrs["dim"] = std::to_string(dim);
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = Dispatcher::dispatch("softmax", inputs, attrs)[0];
        return Variable(result, false);
    }

    auto grad_fn = std::make_shared<SoftmaxBackward>(dim);

    // Compute forward and save output for backward
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);
    std::vector<Tensor> input_tensors = {input.tensor()};
    auto result_tensor = Dispatcher::dispatch("softmax", input_tensors, attrs)[0];

    grad_fn->save_for_backward({result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs;


    next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf


    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }

    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto log_softmax(const Variable& input, int64_t dim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        OpAttributes attrs;
        attrs["dim"] = std::to_string(dim);
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = Dispatcher::dispatch("log_softmax", inputs, attrs)[0];
        return Variable(result, false);
    }

    auto grad_fn = std::make_shared<LogSoftmaxBackward>(dim);

    // Compute forward and save output for backward
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);
    std::vector<Tensor> input_tensors = {input.tensor()};
    auto result_tensor = Dispatcher::dispatch("log_softmax", input_tensors, attrs)[0];

    grad_fn->save_for_backward({result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs;


    next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf


    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }

    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto abs(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::abs(input.tensor()), false);
    }

    auto grad_fn = std::make_shared<AbsBackward>();

    // Save input tensor for backward pass
    grad_fn->save_for_backward({input.tensor()});

    std::vector<std::shared_ptr<Function>> next_funcs;


    next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf


    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }

    grad_fn->set_input_variables(input_vars);

    auto result_tensor = tenzor::abs(input.tensor());
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto clamp(const Variable& input, float min, float max) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::clamp(input.tensor(), min, max), false);
    }

    auto grad_fn = std::make_shared<ClampBackward>(min, max);

    // Save input tensor for backward pass
    grad_fn->save_for_backward({input.tensor()});

    std::vector<std::shared_ptr<Function>> next_funcs;


    next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf


    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }

    grad_fn->set_input_variables(input_vars);

    auto result_tensor = tenzor::clamp(input.tensor(), min, max);
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto max(const Variable& input, std::optional<int64_t> dim, bool keepdim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::max(input.tensor(), dim, keepdim), false);
    }

    auto grad_fn = std::make_shared<MaxBackward>(dim, keepdim);

    // Compute result first so we can save it
    auto result_tensor = tenzor::max(input.tensor(), dim, keepdim);

    // Save input and output for backward pass
    grad_fn->save_for_backward({input.tensor(), result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs;


    next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf


    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }

    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto reshape(const Variable& input, const std::vector<int64_t>& shape) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        // No gradient needed, just compute
        return Variable(tenzor::reshape(input.tensor(), shape), false);
    }

    // Save original input shape for backward pass
    std::vector<int64_t> input_shape(input.shape().begin(), input.shape().end());
    auto grad_fn = std::make_shared<ReshapeBackward>(input_shape);

    // Set up backward graph
    std::vector<std::shared_ptr<Function>> next_funcs;

    next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf

    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }

    grad_fn->set_input_variables(input_vars);

    // Compute result
    auto result_tensor = tenzor::reshape(input.tensor(), shape);
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto permute(const Variable& input, const std::vector<int64_t>& dims) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        // No gradient needed, just compute
        return Variable(tenzor::permute(input.tensor(), dims), false);
    }

    auto grad_fn = std::make_shared<PermuteBackward>(dims);

    // Set up backward graph
    std::vector<std::shared_ptr<Function>> next_funcs;

    next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf

    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }

    grad_fn->set_input_variables(input_vars);

    // Compute result
    auto result_tensor = tenzor::permute(input.tensor(), dims);
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto transpose(const Variable& input, int64_t dim0, int64_t dim1) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::transpose(input.tensor(), dim0, dim1), false);
    }

    auto grad_fn = std::make_shared<TransposeBackward>(dim0, dim1);
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(input.grad_fn());
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }
    grad_fn->set_input_variables(input_vars);

    auto result_tensor = tenzor::transpose(input.tensor(), dim0, dim1);
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto squeeze(const Variable& input, int64_t dim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::squeeze(input.tensor(), dim), false);
    }

    auto grad_fn = std::make_shared<SqueezeBackward>(dim);
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(input.grad_fn());
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }
    grad_fn->set_input_variables(input_vars);

    auto result_tensor = tenzor::squeeze(input.tensor(), dim);
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto roll(const Variable& input, int64_t shifts, int64_t dim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::roll(input.tensor(), shifts, dim), false);
    }

    auto grad_fn = std::make_shared<RollBackward>(shifts, dim);
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(input.grad_fn());
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }
    grad_fn->set_input_variables(input_vars);

    auto result_tensor = tenzor::roll(input.tensor(), shifts, dim);
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto cat(const std::vector<Variable>& inputs, int64_t dim) -> Variable {
    // Check if any input requires grad
    bool any_requires_grad = false;
    for (const auto& input : inputs) {
        if (input.requires_grad()) {
            any_requires_grad = true;
            break;
        }
    }

    if (!any_requires_grad || !is_grad_enabled()) {
        // No gradient needed, just compute
        std::vector<Tensor> tensors;
        tensors.reserve(inputs.size());
        for (const auto& var : inputs) {
            tensors.push_back(var.tensor());
        }
        return Variable(tenzor::cat(tensors, dim), false);
    }

    // Normalize negative dimension index
    if (!inputs.empty()) {
        auto ndim = static_cast<int64_t>(inputs[0].shape().size());
        if (dim < 0) {
            dim += ndim;
        }
    }

    // Collect split sizes (size of each input along concatenation dimension)
    std::vector<int64_t> split_sizes;
    split_sizes.reserve(inputs.size());
    for (const auto& input : inputs) {
        split_sizes.push_back(input.shape()[dim]);
    }

    auto grad_fn = std::make_shared<CatBackward>(split_sizes, dim);

    // Set up backward graph - one next_func per input
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.reserve(inputs.size());
    for (const auto& input : inputs) {
        next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf
    }
    grad_fn->set_next_functions(next_funcs);

    // Track input variables that require grad for gradient accumulation
    std::vector<Variable> input_vars;
    for (const auto& input : inputs) {
        if (input.requires_grad()) {
            input_vars.push_back(input);
        }
    }
    grad_fn->set_input_variables(input_vars);

    // Compute result using tensor-level cat
    std::vector<Tensor> tensors;
    tensors.reserve(inputs.size());
    for (const auto& var : inputs) {
        tensors.push_back(var.tensor());
    }
    auto result_tensor = tenzor::cat(tensors, dim);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto slice(const Variable& input, int64_t dim, int64_t start, int64_t end, int64_t step) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        // No gradient needed, just compute using Tensor::slice() method
        auto result = input.tensor().slice(dim, start, end, step);
        return Variable(result, false);
    }

    // Save original input shape for backward pass
    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    auto grad_fn = std::make_shared<SliceBackward>(input_shape, dim, start, end, step);

    // Set up backward graph
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf
    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    if (input.requires_grad()) {
        input_vars.push_back(input);
    }
    grad_fn->set_input_variables(input_vars);

    // Compute result using Tensor::slice() method directly
    auto result_tensor = input.tensor().slice(dim, start, end, step);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto bmm(const Variable& a, const Variable& b) -> Variable {
    if ((!a.requires_grad() && !b.requires_grad()) || !is_grad_enabled()) {
        // No gradient needed, just compute
        return Variable(tenzor::bmm(a.tensor(), b.tensor()), false);
    }

    auto grad_fn = std::make_shared<BmmBackward>();

    // Save input tensors for backward pass
    grad_fn->save_for_backward({a.tensor(), b.tensor()});

    // Set up backward graph - MUST maintain index correspondence with input_grads!
    // Use nullptr for leaf variables to preserve indices
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(a.grad_fn());  // nullptr if a is leaf
    next_funcs.push_back(b.grad_fn());  // nullptr if b is leaf
    grad_fn->set_next_functions(next_funcs);

    // Track input variables for gradient accumulation
    // MUST include both variables to maintain 1:1 index correspondence with input_grads
    // The engine correctly skips variables that don't require grad
    std::vector<Variable> input_vars = {a, b};
    grad_fn->set_input_variables(input_vars);

    // Compute result
    auto result_tensor = tenzor::bmm(a.tensor(), b.tensor());
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto matmul(const Variable& a, const Variable& b) -> Variable {
    if ((!a.requires_grad() && !b.requires_grad()) || !is_grad_enabled()) {
        // No gradient needed, just compute
        return Variable(tenzor::matmul(a.tensor(), b.tensor()), false);
    }

    auto grad_fn = std::make_shared<MatMulBackward>();

    // Save input tensors for backward pass
    auto a_tensor = a.tensor();
    auto b_tensor = b.tensor();

    grad_fn->save_for_backward({a_tensor, b_tensor});

    // Set up backward graph - MUST maintain index correspondence with input_grads!
    // Use nullptr for leaf variables to preserve indices
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(a.grad_fn());  // nullptr if a is leaf
    next_funcs.push_back(b.grad_fn());  // nullptr if b is leaf
    grad_fn->set_next_functions(next_funcs);

    // Track input variables for gradient accumulation
    // MUST include both variables to maintain 1:1 index correspondence with input_grads
    // The engine correctly skips variables that don't require grad
    std::vector<Variable> input_vars = {a, b};
    grad_fn->set_input_variables(input_vars);

    // Compute result
    auto result_tensor = tenzor::matmul(a_tensor, b_tensor);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

} // namespace tenzor
