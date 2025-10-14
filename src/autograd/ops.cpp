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

    // Track input variable
    grad_fn->set_input_variables({const_cast<Variable*>(&input)});

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

    grad_fn->set_input_variables({const_cast<Variable*>(&input)});

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

    grad_fn->set_input_variables({const_cast<Variable*>(&input)});

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

    grad_fn->set_input_variables({const_cast<Variable*>(&input)});

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

    grad_fn->set_input_variables({const_cast<Variable*>(&input)});

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

    grad_fn->set_input_variables({const_cast<Variable*>(&input)});

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

    grad_fn->set_input_variables({const_cast<Variable*>(&input)});

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

    grad_fn->set_input_variables({const_cast<Variable*>(&input)});

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

    grad_fn->set_input_variables({const_cast<Variable*>(&input)});

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

    grad_fn->set_input_variables({const_cast<Variable*>(&input)});

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

    // Track input variable
    grad_fn->set_input_variables({const_cast<Variable*>(&input)});

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

    // Track input variable
    grad_fn->set_input_variables({const_cast<Variable*>(&input)});

    // Compute result
    auto result_tensor = tenzor::permute(input.tensor(), dims);
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

    // Track input variables
    grad_fn->set_input_variables({const_cast<Variable*>(&a), const_cast<Variable*>(&b)});

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
    grad_fn->save_for_backward({a.tensor(), b.tensor()});

    // Set up backward graph - MUST maintain index correspondence with input_grads!
    // Use nullptr for leaf variables to preserve indices
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(a.grad_fn());  // nullptr if a is leaf
    next_funcs.push_back(b.grad_fn());  // nullptr if b is leaf
    grad_fn->set_next_functions(next_funcs);

    // Track input variables
    grad_fn->set_input_variables({const_cast<Variable*>(&a), const_cast<Variable*>(&b)});

    // Compute result
    auto result_tensor = tenzor::matmul(a.tensor(), b.tensor());
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

} // namespace tenzor
