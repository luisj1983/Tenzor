#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <optional>
#include <tuple>

namespace tenzor {

// Optimized forward linear computation via backend dispatch
// Uses dispatch_single to avoid output vector allocation
static Tensor linear_forward_dispatch(const Tensor& x, const Tensor& w, const Tensor& b) {
    // Tensor copies are cheap (shared_ptr storage), but we minimize them
    std::vector<Tensor> inputs = {x, w, b};
    return dispatch_single<OpId::Linear>(inputs);
}

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
        attrs.set(AttrKey::Dim, dim);
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = dispatch(OpId::Softmax, inputs, attrs)[0];
        return Variable(result, false);
    }

    auto grad_fn = std::make_shared<SoftmaxBackward>(dim);

    // Compute forward and save output for backward
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> input_tensors = {input.tensor()};
    auto result_tensor = dispatch(OpId::Softmax, input_tensors, attrs)[0];

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
        attrs.set(AttrKey::Dim, dim);
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = dispatch(OpId::LogSoftmax, inputs, attrs)[0];
        return Variable(result, false);
    }

    auto grad_fn = std::make_shared<LogSoftmaxBackward>(dim);

    // Compute forward and save output for backward
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> input_tensors = {input.tensor()};
    auto result_tensor = dispatch(OpId::LogSoftmax, input_tensors, attrs)[0];

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

    // Track input variables for gradient accumulation
    // MUST include all inputs to maintain 1:1 index correspondence with gradients
    // The engine correctly skips variables that don't require grad
    std::vector<Variable> input_vars(inputs.begin(), inputs.end());
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

    // When create_graph is active, also save Variables to preserve graph connections
    if (is_creating_graph()) {
        grad_fn->save_variables_for_backward({a, b});
    }

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

auto linear(const Variable& x, const Variable& w, const Variable& b) -> Variable {
    // Fast path: no gradients needed
    if ((!x.requires_grad() && !w.requires_grad() && !b.requires_grad()) || !is_grad_enabled()) {
        // Dispatch to backend (CPU uses MKL, CUDA uses cuBLAS, etc.)
        auto result = linear_forward_dispatch(x.tensor(), w.tensor(), b.tensor());
        return Variable(result, false);
    }

    // Create backward function
    auto grad_fn = std::make_shared<LinearBackward>();

    // Save all input tensors for backward pass
    grad_fn->save_for_backward({x.tensor(), w.tensor(), b.tensor()});

    // Set up backward graph - maintain index correspondence
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(x.grad_fn());  // nullptr if x is leaf
    next_funcs.push_back(w.grad_fn());  // nullptr if w is leaf
    next_funcs.push_back(b.grad_fn());  // nullptr if b is leaf
    grad_fn->set_next_functions(next_funcs);

    // Track input variables for gradient accumulation
    // MUST include all three to maintain 1:1 index correspondence with gradients
    std::vector<Variable> input_vars = {x, w, b};
    grad_fn->set_input_variables(input_vars);

    // Dispatch forward to backend (CPU uses MKL, CUDA uses cuBLAS, etc.)
    auto result_tensor = linear_forward_dispatch(x.tensor(), w.tensor(), b.tensor());

    // Create output with grad_fn
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

// ============================================================================
// Helper: Unary op wrapper with autograd (saves input for backward)
// ============================================================================
namespace {
template<typename BackwardT, typename TensorOp>
auto unary_autograd(const Variable& input, TensorOp&& tensor_op) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tensor_op(input.tensor()), false);
    }
    auto grad_fn = std::make_shared<BackwardT>();
    grad_fn->save_for_backward({input.tensor()});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    auto result = tensor_op(input.tensor());
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// Variant that saves output instead of input (for sigmoid, tanh, sqrt, etc.)
template<typename BackwardT, typename TensorOp>
auto unary_autograd_save_output(const Variable& input, TensorOp&& tensor_op) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tensor_op(input.tensor()), false);
    }
    auto result = tensor_op(input.tensor());
    auto grad_fn = std::make_shared<BackwardT>();
    grad_fn->save_for_backward({result});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// Variant that saves input + a scalar parameter as a second tensor
template<typename BackwardT, typename TensorOp>
auto unary_autograd_with_param(const Variable& input, float param, TensorOp&& tensor_op) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tensor_op(input.tensor()), false);
    }
    auto grad_fn = std::make_shared<BackwardT>();
    auto param_tensor = full({1}, param, input.tensor().dtype(), input.tensor().device());
    grad_fn->save_for_backward({input.tensor(), param_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    auto result = tensor_op(input.tensor());
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}
} // anonymous namespace

// ============================================================================
// Activation Functions (Variable wrappers)
// ============================================================================

auto sigmoid(const Variable& input) -> Variable {
    return unary_autograd_save_output<SigmoidBackward_AG>(input,
        [](const Tensor& t) { return tenzor::sigmoid(t); });
}

auto tanh(const Variable& input) -> Variable {
    return unary_autograd_save_output<TanhBackward_AG>(input,
        [](const Tensor& t) { return tenzor::tanh(t); });
}

auto gelu(const Variable& input) -> Variable {
    return unary_autograd<GeluBackward>(input,
        [](const Tensor& t) {
            std::vector<Tensor> inputs = {t};
            return dispatch(OpId::Gelu, inputs)[0];
        });
}

auto elu(const Variable& input, float alpha) -> Variable {
    return unary_autograd_with_param<EluBackward>(input, alpha,
        [](const Tensor& t) {
            std::vector<Tensor> inputs = {t};
            return dispatch(OpId::Elu, inputs)[0];
        });
}

auto selu(const Variable& input) -> Variable {
    return unary_autograd<SeluBackward>(input,
        [](const Tensor& t) {
            std::vector<Tensor> inputs = {t};
            return dispatch(OpId::Selu, inputs)[0];
        });
}

auto mish(const Variable& input) -> Variable {
    return unary_autograd<MishBackward>(input,
        [](const Tensor& t) {
            std::vector<Tensor> inputs = {t};
            return dispatch(OpId::Mish, inputs)[0];
        });
}

auto leaky_relu(const Variable& input, float negative_slope) -> Variable {
    return unary_autograd_with_param<LeakyReluBackward>(input, negative_slope,
        [negative_slope](const Tensor& t) {
            std::vector<Tensor> inputs = {t};
            OpAttributes attrs;
            attrs.set(AttrKey::Negative_slope, static_cast<double>(negative_slope));
            return dispatch(OpId::LeakyReLU, inputs, attrs)[0];
        });
}

auto softplus(const Variable& input, float beta) -> Variable {
    return unary_autograd_with_param<SoftplusBackward>(input, beta,
        [](const Tensor& t) {
            std::vector<Tensor> inputs = {t};
            return dispatch(OpId::Softplus, inputs)[0];
        });
}

// ============================================================================
// Element-wise Math Operations (Variable wrappers)
// ============================================================================

auto sqrt(const Variable& input) -> Variable {
    return unary_autograd_save_output<SqrtBackward>(input,
        [](const Tensor& t) { return tenzor::sqrt(t); });
}

auto pow(const Variable& input, float exponent) -> Variable {
    return unary_autograd_with_param<PowBackward>(input, exponent,
        [exponent](const Tensor& t) { return tenzor::pow(t, exponent); });
}

auto reciprocal(const Variable& input) -> Variable {
    return unary_autograd_save_output<ReciprocalBackward>(input,
        [](const Tensor& t) { return tenzor::reciprocal(t); });
}

auto sin(const Variable& input) -> Variable {
    return unary_autograd<SinBackward>(input,
        [](const Tensor& t) { return tenzor::sin(t); });
}

auto cos(const Variable& input) -> Variable {
    return unary_autograd<CosBackward>(input,
        [](const Tensor& t) { return tenzor::cos(t); });
}

auto tan(const Variable& input) -> Variable {
    return unary_autograd_save_output<TanBackward>(input,
        [](const Tensor& t) { return tenzor::tan(t); });
}

auto asin(const Variable& input) -> Variable {
    return unary_autograd<AsinBackward>(input,
        [](const Tensor& t) { return tenzor::asin(t); });
}

auto acos(const Variable& input) -> Variable {
    return unary_autograd<AcosBackward>(input,
        [](const Tensor& t) { return tenzor::acos(t); });
}

auto atan(const Variable& input) -> Variable {
    return unary_autograd<AtanBackward>(input,
        [](const Tensor& t) { return tenzor::atan(t); });
}

auto sinh(const Variable& input) -> Variable {
    return unary_autograd<SinhBackward>(input,
        [](const Tensor& t) { return tenzor::sinh(t); });
}

auto cosh(const Variable& input) -> Variable {
    return unary_autograd<CoshBackward>(input,
        [](const Tensor& t) { return tenzor::cosh(t); });
}

// ============================================================================
// Extended Math Operations (Variable wrappers)
// ============================================================================

auto erf(const Variable& input) -> Variable {
    return unary_autograd<ErfBackward>(input,
        [](const Tensor& t) { return tenzor::erf(t); });
}

auto erfc(const Variable& input) -> Variable {
    return unary_autograd<ErfcBackward>(input,
        [](const Tensor& t) { return tenzor::erfc(t); });
}

auto log2(const Variable& input) -> Variable {
    return unary_autograd<Log2Backward>(input,
        [](const Tensor& t) { return tenzor::log2(t); });
}

auto log10(const Variable& input) -> Variable {
    return unary_autograd<Log10Backward>(input,
        [](const Tensor& t) { return tenzor::log10(t); });
}

auto log1p(const Variable& input) -> Variable {
    return unary_autograd<Log1pBackward>(input,
        [](const Tensor& t) { return tenzor::log1p(t); });
}

auto exp2(const Variable& input) -> Variable {
    return unary_autograd_save_output<Exp2Backward>(input,
        [](const Tensor& t) { return tenzor::exp2(t); });
}

auto expm1(const Variable& input) -> Variable {
    return unary_autograd<Expm1Backward>(input,
        [](const Tensor& t) { return tenzor::expm1(t); });
}

auto atan2(const Variable& y, const Variable& x) -> Variable {
    bool needs_grad = (y.requires_grad() || x.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(tenzor::atan2(y.tensor(), x.tensor()), false);
    }
    auto grad_fn = std::make_shared<Atan2Backward>();
    grad_fn->save_for_backward({y.tensor(), x.tensor()});
    grad_fn->set_next_functions({y.grad_fn(), x.grad_fn()});
    grad_fn->set_input_variables({y, x});
    auto result = tenzor::atan2(y.tensor(), x.tensor());
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ============================================================================
// Additional Reduction Operations (Variable wrappers)
// ============================================================================

auto min(const Variable& input, std::optional<int64_t> dim, bool keepdim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::min(input.tensor(), dim, keepdim), false);
    }
    auto result = tenzor::min(input.tensor(), dim, keepdim);
    auto grad_fn = std::make_shared<MinBackward>(dim, keepdim);
    grad_fn->save_for_backward({input.tensor(), result});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto std(const Variable& input, std::optional<int64_t> dim, bool keepdim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::std(input.tensor(), dim, keepdim), false);
    }
    auto result = tenzor::std(input.tensor(), dim, keepdim);
    auto grad_fn = std::make_shared<StdBackward>(dim, keepdim);
    grad_fn->save_for_backward({input.tensor(), result});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto var(const Variable& input, std::optional<int64_t> dim, bool keepdim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::var(input.tensor(), dim, keepdim), false);
    }
    auto result = tenzor::var(input.tensor(), dim, keepdim);
    auto grad_fn = std::make_shared<VarBackward>(dim, keepdim);
    grad_fn->save_for_backward({input.tensor()});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto prod(const Variable& input, std::optional<int64_t> dim, bool keepdim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::prod(input.tensor(), dim, keepdim), false);
    }
    auto result = tenzor::prod(input.tensor(), dim, keepdim);
    auto grad_fn = std::make_shared<ProdBackward>(dim, keepdim);
    grad_fn->save_for_backward({input.tensor(), result});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto logsumexp(const Variable& input, int64_t dim, bool keepdim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::logsumexp(input.tensor(), dim, keepdim), false);
    }
    auto result = tenzor::logsumexp(input.tensor(), dim, keepdim);
    auto grad_fn = std::make_shared<LogSumExpBackward>(dim, keepdim);
    grad_fn->save_for_backward({input.tensor(), result});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ============================================================================
// Shape/Indexing Operations (Variable wrappers)
// ============================================================================

auto unsqueeze(const Variable& input, int64_t dim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::unsqueeze(input.tensor(), dim), false);
    }
    auto result = tenzor::unsqueeze(input.tensor(), dim);
    auto grad_fn = std::make_shared<UnsqueezeBackward>();
    auto dim_tensor = full({1}, static_cast<float>(dim), DType::Float32, Device::cpu());
    grad_fn->save_for_backward({dim_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto expand(const Variable& input, const std::vector<int64_t>& shape) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::expand(input.tensor(), shape), false);
    }
    auto result = tenzor::expand(input.tensor(), shape);
    auto grad_fn = std::make_shared<ExpandBackward>();
    // Save original shape as tensor for backward
    std::vector<float> shape_data(input.tensor().shape().begin(), input.tensor().shape().end());
    auto shape_tensor = zeros({static_cast<int64_t>(shape_data.size())}, DType::Float32, Device::cpu());
    std::memcpy(shape_tensor.data_ptr(), shape_data.data(), shape_data.size() * sizeof(float));
    grad_fn->save_for_backward({shape_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto flatten(const Variable& input, int64_t start_dim, int64_t end_dim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::flatten(input.tensor(), start_dim, end_dim), false);
    }
    auto result = tenzor::flatten(input.tensor(), start_dim, end_dim);
    auto grad_fn = std::make_shared<FlattenBackward>();
    // Save original shape as tensor for backward
    std::vector<float> shape_data(input.tensor().shape().begin(), input.tensor().shape().end());
    auto shape_tensor = zeros({static_cast<int64_t>(shape_data.size())}, DType::Float32, Device::cpu());
    std::memcpy(shape_tensor.data_ptr(), shape_data.data(), shape_data.size() * sizeof(float));
    grad_fn->save_for_backward({shape_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto where(const Variable& condition, const Variable& x, const Variable& y) -> Variable {
    bool needs_grad = (x.requires_grad() || y.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(tenzor::where(condition.tensor(), x.tensor(), y.tensor()), false);
    }
    auto result = tenzor::where(condition.tensor(), x.tensor(), y.tensor());
    auto grad_fn = std::make_shared<WhereBackward>();
    grad_fn->save_for_backward({condition.tensor()});
    grad_fn->set_next_functions({nullptr, x.grad_fn(), y.grad_fn()});
    grad_fn->set_input_variables({condition, x, y});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto gather(const Variable& input, int64_t dim, const Tensor& index) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::gather(input.tensor(), dim, index), false);
    }
    auto result = tenzor::gather(input.tensor(), dim, index);
    auto grad_fn = std::make_shared<GatherBackward>();
    auto dim_tensor = full({1}, static_cast<float>(dim), DType::Float32, Device::cpu());
    // Save input shape for backward scatter_add
    std::vector<float> shape_data(input.tensor().shape().begin(), input.tensor().shape().end());
    auto shape_tensor = zeros({static_cast<int64_t>(shape_data.size())}, DType::Float32, Device::cpu());
    std::memcpy(shape_tensor.data_ptr(), shape_data.data(), shape_data.size() * sizeof(float));
    grad_fn->save_for_backward({dim_tensor, index, shape_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto scatter(const Variable& input, int64_t dim, const Tensor& index, const Variable& src) -> Variable {
    bool needs_grad = (input.requires_grad() || src.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(tenzor::scatter(input.tensor(), dim, index, src.tensor()), false);
    }
    auto result = tenzor::scatter(input.tensor(), dim, index, src.tensor());
    auto grad_fn = std::make_shared<ScatterBackward>();
    auto dim_tensor = full({1}, static_cast<float>(dim), DType::Float32, Device::cpu());
    grad_fn->save_for_backward({dim_tensor, index});
    grad_fn->set_next_functions({input.grad_fn(), src.grad_fn()});
    grad_fn->set_input_variables({input, src});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto index_select(const Variable& input, int64_t dim, const Tensor& index) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::index_select(input.tensor(), dim, index), false);
    }
    auto result = tenzor::index_select(input.tensor(), dim, index);
    auto grad_fn = std::make_shared<IndexSelectBackward>();
    auto dim_tensor = full({1}, static_cast<float>(dim), DType::Float32, Device::cpu());
    std::vector<float> shape_data(input.tensor().shape().begin(), input.tensor().shape().end());
    auto shape_tensor = zeros({static_cast<int64_t>(shape_data.size())}, DType::Float32, Device::cpu());
    std::memcpy(shape_tensor.data_ptr(), shape_data.data(), shape_data.size() * sizeof(float));
    grad_fn->save_for_backward({dim_tensor, index, shape_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto narrow(const Variable& input, int64_t dim, int64_t start, int64_t length) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(input.tensor().slice(dim, start, start + length), false);
    }
    auto result = input.tensor().slice(dim, start, start + length);
    auto grad_fn = std::make_shared<NarrowBackward>();
    auto dim_tensor = full({1}, static_cast<float>(dim), DType::Float32, Device::cpu());
    auto start_tensor = full({1}, static_cast<float>(start), DType::Float32, Device::cpu());
    std::vector<float> shape_data(input.tensor().shape().begin(), input.tensor().shape().end());
    auto shape_tensor = zeros({static_cast<int64_t>(shape_data.size())}, DType::Float32, Device::cpu());
    std::memcpy(shape_tensor.data_ptr(), shape_data.data(), shape_data.size() * sizeof(float));
    grad_fn->save_for_backward({dim_tensor, start_tensor, shape_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto flip(const Variable& input, const std::vector<int64_t>& dims) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::flip(input.tensor(), dims), false);
    }
    auto result = tenzor::flip(input.tensor(), dims);
    auto grad_fn = std::make_shared<FlipBackward>();
    // Save dims as tensor
    auto dims_tensor = zeros({static_cast<int64_t>(dims.size())}, DType::Float32, Device::cpu());
    auto* ptr = dims_tensor.data<float>();
    for (size_t i = 0; i < dims.size(); ++i) ptr[i] = static_cast<float>(dims[i]);
    grad_fn->save_for_backward({dims_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto repeat(const Variable& input, const std::vector<int64_t>& repeats) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::repeat(input.tensor(), repeats), false);
    }
    auto result = tenzor::repeat(input.tensor(), repeats);
    auto grad_fn = std::make_shared<RepeatBackward>();
    // Save original shape and repeats
    std::vector<float> shape_data(input.tensor().shape().begin(), input.tensor().shape().end());
    auto shape_tensor = zeros({static_cast<int64_t>(shape_data.size())}, DType::Float32, Device::cpu());
    std::memcpy(shape_tensor.data_ptr(), shape_data.data(), shape_data.size() * sizeof(float));
    auto repeats_tensor = zeros({static_cast<int64_t>(repeats.size())}, DType::Float32, Device::cpu());
    auto* rptr = repeats_tensor.data<float>();
    for (size_t i = 0; i < repeats.size(); ++i) rptr[i] = static_cast<float>(repeats[i]);
    grad_fn->save_for_backward({shape_tensor, repeats_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ============================================================================
// Linear Algebra Operations
// ============================================================================

auto det(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::linalg::det(input.tensor()), false);
    }

    auto result_tensor = tenzor::linalg::det(input.tensor());
    auto inv_tensor = tenzor::linalg::inv(input.tensor());

    auto grad_fn = std::make_shared<DetBackward>();
    grad_fn->save_for_backward({result_tensor, inv_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto inv(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::linalg::inv(input.tensor()), false);
    }

    auto result_tensor = tenzor::linalg::inv(input.tensor());

    auto grad_fn = std::make_shared<InvBackward>();
    grad_fn->save_for_backward({result_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto solve(const Variable& A, const Variable& B) -> Variable {
    if ((!A.requires_grad() && !B.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::linalg::solve(A.tensor(), B.tensor()), false);
    }

    auto result_tensor = tenzor::linalg::solve(A.tensor(), B.tensor());

    auto grad_fn = std::make_shared<SolveBackward>();
    grad_fn->save_for_backward({A.tensor(), result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(A.grad_fn());
    next_funcs.push_back(B.grad_fn());
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (A.requires_grad()) input_vars.push_back(A);
    if (B.requires_grad()) input_vars.push_back(B);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto cholesky(const Variable& input, bool upper) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::linalg::cholesky(input.tensor(), upper), false);
    }

    auto result_tensor = tenzor::linalg::cholesky(input.tensor(), upper);

    auto grad_fn = std::make_shared<CholeskyBackward>(upper);
    grad_fn->save_for_backward({result_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto svd(const Variable& input, bool full_matrices) -> std::tuple<Variable, Variable, Variable> {
    if (!input.requires_grad() || !is_grad_enabled()) {
        auto [U, S, Vh] = tenzor::linalg::svd(input.tensor(), full_matrices);
        return {Variable(U, false), Variable(S, false), Variable(Vh, false)};
    }

    auto [U_tensor, S_tensor, Vh_tensor] = tenzor::linalg::svd(input.tensor(), full_matrices);

    auto grad_fn = std::make_shared<SvdBackward>(full_matrices);
    grad_fn->save_for_backward({U_tensor, S_tensor, Vh_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable U_var(U_tensor, true);
    Variable S_var(S_tensor, true);
    Variable Vh_var(Vh_tensor, true);

    U_var.set_grad_fn(grad_fn);
    S_var.set_grad_fn(grad_fn);
    Vh_var.set_grad_fn(grad_fn);

    return {U_var, S_var, Vh_var};
}

auto qr(const Variable& input) -> std::tuple<Variable, Variable> {
    if (!input.requires_grad() || !is_grad_enabled()) {
        auto [Q, R] = tenzor::linalg::qr(input.tensor());
        return {Variable(Q, false), Variable(R, false)};
    }

    auto [Q_tensor, R_tensor] = tenzor::linalg::qr(input.tensor());

    auto grad_fn = std::make_shared<QrBackward>();
    grad_fn->save_for_backward({Q_tensor, R_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable Q_var(Q_tensor, true);
    Variable R_var(R_tensor, true);

    Q_var.set_grad_fn(grad_fn);
    R_var.set_grad_fn(grad_fn);

    return {Q_var, R_var};
}

auto eigh(const Variable& input) -> std::tuple<Variable, Variable> {
    if (!input.requires_grad() || !is_grad_enabled()) {
        auto [W, V] = tenzor::linalg::eigh(input.tensor());
        return {Variable(W, false), Variable(V, false)};
    }

    auto [W_tensor, V_tensor] = tenzor::linalg::eigh(input.tensor());

    auto grad_fn = std::make_shared<EighBackward>();
    grad_fn->save_for_backward({W_tensor, V_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable W_var(W_tensor, true);
    Variable V_var(V_tensor, true);

    W_var.set_grad_fn(grad_fn);
    V_var.set_grad_fn(grad_fn);

    return {W_var, V_var};
}

auto eigvalsh(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::linalg::eigvalsh(input.tensor()), false);
    }

    // We need eigenvectors for backward, so compute full eigh
    auto [W_tensor, V_tensor] = tenzor::linalg::eigh(input.tensor());

    auto grad_fn = std::make_shared<EigvalshBackward>();
    grad_fn->save_for_backward({V_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable output(W_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto linalg_norm(const Variable& input, const std::string& ord) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::linalg::norm(input.tensor(), ord), false);
    }

    auto result_tensor = tenzor::linalg::norm(input.tensor(), ord);

    auto grad_fn = std::make_shared<NormBackward_Linalg>(ord);
    grad_fn->save_for_backward({input.tensor(), result_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto slogdet(const Variable& input) -> std::tuple<Variable, Variable> {
    if (!input.requires_grad() || !is_grad_enabled()) {
        auto [sign, logabsdet] = tenzor::linalg::slogdet(input.tensor());
        return {Variable(sign, false), Variable(logabsdet, false)};
    }

    auto [sign_tensor, logabsdet_tensor] = tenzor::linalg::slogdet(input.tensor());
    auto inv_tensor = tenzor::linalg::inv(input.tensor());

    auto grad_fn = std::make_shared<SlogdetBackward>();
    grad_fn->save_for_backward({inv_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    // sign has no gradient (discrete), but we still wrap it as Variable
    Variable sign_var(sign_tensor, false);
    Variable logabsdet_var(logabsdet_tensor, true);
    logabsdet_var.set_grad_fn(grad_fn);

    return {sign_var, logabsdet_var};
}

// ============================================================================
// Sparse Autograd Operations
// ============================================================================

auto spmm(const SparseTensor& sparse, const Variable& dense) -> Variable {
    if (!dense.requires_grad() || !is_grad_enabled()) {
        // No gradient needed, just compute the forward pass
        return Variable(sparse::spmm(sparse, dense.tensor()), false);
    }

    // Compute forward: Y = S @ D
    auto result_tensor = sparse::spmm(sparse, dense.tensor());

    // For backward: grad_D = S^T @ grad_Y
    // Store S^T as a SparseTensor (avoids converting to dense).
    auto sparse_transposed = sparse.transpose();  // shape (K, M)

    auto grad_fn = std::make_shared<SpMMBackward>();
    grad_fn->set_sparse_transposed(std::move(sparse_transposed));
    grad_fn->set_next_functions({dense.grad_fn()});
    grad_fn->set_input_variables({dense});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto spmv(const SparseTensor& sparse, const Variable& vec) -> Variable {
    if (!vec.requires_grad() || !is_grad_enabled()) {
        return Variable(sparse::spmv(sparse, vec.tensor()), false);
    }

    // Compute forward: y = S @ v
    auto result_tensor = sparse::spmv(sparse, vec.tensor());

    // For backward: grad_v = S^T @ grad_y
    // Store S^T as a SparseTensor (avoids converting to dense).
    auto sparse_transposed = sparse.transpose();  // shape (K, M)

    auto grad_fn = std::make_shared<SpMVBackward>();
    grad_fn->set_sparse_transposed(std::move(sparse_transposed));
    grad_fn->set_next_functions({vec.grad_fn()});
    grad_fn->set_input_variables({vec});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ============================================================================
// Cumulative, Sorting, and Triangular Operations
// ============================================================================

auto cumsum(const Variable& input, int64_t dim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::cumsum(input.tensor(), dim), false);
    }
    auto result = tenzor::cumsum(input.tensor(), dim);
    auto grad_fn = std::make_shared<CumSumBackward>(dim);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto cumprod(const Variable& input, int64_t dim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::cumprod(input.tensor(), dim), false);
    }
    auto result = tenzor::cumprod(input.tensor(), dim);
    auto grad_fn = std::make_shared<CumProdBackward>(dim);
    grad_fn->save_for_backward({input.tensor(), result});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto topk(const Variable& input, int64_t k, int64_t dim,
          bool largest, bool sorted) -> std::pair<Variable, Tensor> {
    auto [values, indices] = tenzor::topk(input.tensor(), k, dim, largest, sorted);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return {Variable(values, false), indices};
    }
    auto grad_fn = std::make_shared<TopKBackward>(k, dim);
    // Save original shape and indices
    auto shape = input.tensor().shape();
    auto shape_tensor = zeros({static_cast<int64_t>(shape.size())}, DType::Int64, Device::cpu());
    auto* sptr = shape_tensor.data<int64_t>();
    for (size_t i = 0; i < shape.size(); ++i) sptr[i] = shape[i];
    grad_fn->save_for_backward({shape_tensor, indices});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(values, true);
    output.set_grad_fn(grad_fn);
    return {output, indices};
}

auto sort(const Variable& input, int64_t dim,
          bool descending) -> std::pair<Variable, Tensor> {
    auto [sorted_vals, indices] = tenzor::sort(input.tensor(), dim, descending);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return {Variable(sorted_vals, false), indices};
    }
    auto grad_fn = std::make_shared<SortBackward>(dim);
    // Save original shape and indices
    auto shape = input.tensor().shape();
    auto shape_tensor = zeros({static_cast<int64_t>(shape.size())}, DType::Int64, Device::cpu());
    auto* sptr = shape_tensor.data<int64_t>();
    for (size_t i = 0; i < shape.size(); ++i) sptr[i] = shape[i];
    grad_fn->save_for_backward({shape_tensor, indices});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(sorted_vals, true);
    output.set_grad_fn(grad_fn);
    return {output, indices};
}

auto diag(const Variable& input, int64_t diagonal) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::diag(input.tensor(), diagonal), false);
    }
    auto result = tenzor::diag(input.tensor(), diagonal);
    auto grad_fn = std::make_shared<DiagBackward>(input.tensor().ndim(), diagonal);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto trace(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::trace(input.tensor()), false);
    }
    auto result = tenzor::trace(input.tensor());
    int64_t n = input.tensor().shape()[0];
    auto grad_fn = std::make_shared<TraceBackward>(n);
    grad_fn->save_for_backward({input.tensor()});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto triu(const Variable& input, int64_t diagonal) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::triu(input.tensor(), diagonal), false);
    }
    auto result = tenzor::triu(input.tensor(), diagonal);
    auto grad_fn = std::make_shared<TriuBackward>(diagonal);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto tril(const Variable& input, int64_t diagonal) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::tril(input.tensor(), diagonal), false);
    }
    auto result = tenzor::tril(input.tensor(), diagonal);
    auto grad_fn = std::make_shared<TrilBackward>(diagonal);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ============================================================================
// FFT Autograd Operations
// ============================================================================

namespace fft_autograd {

auto fft(const Variable& input,
         std::optional<int64_t> n, int64_t dim,
         const std::string& norm) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::fft::fft(input.tensor(), n, dim, norm), false);
    }
    auto result = tenzor::fft::fft(input.tensor(), n, dim, norm);
    auto grad_fn = std::make_shared<FFTBackward>(n, dim, norm);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto ifft(const Variable& input,
          std::optional<int64_t> n, int64_t dim,
          const std::string& norm) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::fft::ifft(input.tensor(), n, dim, norm), false);
    }
    auto result = tenzor::fft::ifft(input.tensor(), n, dim, norm);
    auto grad_fn = std::make_shared<IFFTBackward>(n, dim, norm);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto rfft(const Variable& input,
          std::optional<int64_t> n, int64_t dim,
          const std::string& norm) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::fft::rfft(input.tensor(), n, dim, norm), false);
    }
    // Save the original signal length for irfft in backward
    int64_t actual_dim = dim < 0 ? dim + input.tensor().ndim() : dim;
    int64_t signal_length = input.tensor().shape()[actual_dim];
    auto result = tenzor::fft::rfft(input.tensor(), n, dim, norm);
    auto grad_fn = std::make_shared<RFFTBackward>(signal_length, dim, norm);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto irfft(const Variable& input,
           std::optional<int64_t> n, int64_t dim,
           const std::string& norm) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::fft::irfft(input.tensor(), n, dim, norm), false);
    }
    auto result = tenzor::fft::irfft(input.tensor(), n, dim, norm);
    auto grad_fn = std::make_shared<IRFFTBackward>(dim, norm);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

} // namespace fft_autograd

} // namespace tenzor
