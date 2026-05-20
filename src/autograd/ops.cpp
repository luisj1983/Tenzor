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
#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/custom_op.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/jit/tracer.hpp"
#include <optional>
#include <tuple>

namespace tenzor {

// ─────────────────────────────────────────────────────────────────────
// JIT tracing helpers for autograd-level shape ops.
//
// Most shape-only operations (reshape, transpose, permute, slice, cat)
// do NOT dispatch through the backend kernel registry on CPU — they
// rewrite strides directly in `Tensor::reshape` / `Tensor::permute` /
// etc. As a result the dispatch-level tracing interceptor never sees
// them and the resulting graph is missing the shape transitions
// (causing later ops to see wrong rank/shape and the lowering to fail
// with "use of value expects different type than prior uses").
//
// To close that gap, the Variable-level autograd wrappers below
// explicitly post a `TracedOp` to `Tracer::get_instance()` when
// tracing is active. We use `register_new_tensor` for the output so
// view-style aliasing (output shares `data_ptr()` with input) doesn't
// collapse them into the same graph value.
// ─────────────────────────────────────────────────────────────────────
namespace {

inline auto jit_tracing_active() -> bool {
    return ::tenzor::jit::Tracer::get_instance().is_tracing();
}

inline auto jit_record_shape_op(
    ::tenzor::jit::OpType op_type,
    const std::vector<const Variable*>& inputs,
    const Tensor& output,
    const std::vector<std::pair<const char*, int64_t>>& int_attrs = {},
    const std::vector<std::pair<const char*,
                               std::vector<int64_t>>>& vec_attrs = {})
    -> std::string {
    auto& tracer = ::tenzor::jit::Tracer::get_instance();
    std::vector<std::string> input_ids;
    input_ids.reserve(inputs.size());
    for (auto* v : inputs) {
        input_ids.push_back(tracer.register_tensor(v->tensor()));
    }
    auto out_id = tracer.register_new_tensor(output);
    ::tenzor::jit::TracedOp op(op_type, std::move(input_ids), {out_id});
    for (const auto& [name, val] : int_attrs) {
        op.int_attrs[name] = val;
    }
    for (const auto& [name, val] : vec_attrs) {
        op.vec_attrs[name] = val;
    }
    tracer.record_op(std::move(op));
    return out_id;
}

}  // namespace

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

    // Save input, output, and optionally dim for backward. MaxBackward reads
    // saved_tensors_[2] when dim was provided.
    std::vector<Tensor> max_saved = {input.tensor(), result_tensor};
    if (dim.has_value()) {
        Tensor dim_t({1}, DType::Int64, Device::cpu());
        dim_t.data<int64_t>()[0] = *dim;
        max_saved.push_back(dim_t);
    }
    grad_fn->save_for_backward(std::move(max_saved));

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

auto median(const Variable& input, std::optional<int64_t> dim, bool keepdim) -> Variable {
    int64_t d = dim.value_or(-1);
    if (!input.requires_grad() || !is_grad_enabled()) {
        auto [values, indices] = ::tenzor::median(input.tensor(), d, keepdim);
        return Variable(values, false);
    }

    auto grad_fn = std::make_shared<MedianBackward>(dim, keepdim);

    auto [result_tensor, result_indices] = ::tenzor::median(input.tensor(), d, keepdim);

    grad_fn->save_for_backward({input.tensor(), result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(input.grad_fn());

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

auto mode(const Variable& input, std::optional<int64_t> dim, bool keepdim) -> Variable {
    int64_t d = dim.value_or(-1);
    if (!input.requires_grad() || !is_grad_enabled()) {
        auto [values, indices] = ::tenzor::mode(input.tensor(), d, keepdim);
        return Variable(values, false);
    }

    auto grad_fn = std::make_shared<ModeBackward>(dim, keepdim);

    auto [result_tensor, result_indices] = ::tenzor::mode(input.tensor(), d, keepdim);

    grad_fn->save_for_backward({input.tensor(), result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(input.grad_fn());

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
    auto compute = [&]() {
        return tenzor::reshape(input.tensor(), shape);
    };
    // Always record the *resolved* shape (using the eager output's
    // concrete dims) so the JIT graph never carries a `-1` placeholder
    // that the MLIR lowering can't materialise.
    auto record = [&](const Tensor& t) {
        if (!jit_tracing_active()) return;
        std::vector<int64_t> resolved(t.shape().begin(), t.shape().end());
        jit_record_shape_op(::tenzor::jit::OpType::Reshape, {&input}, t,
                            {},
                            {{"shape", resolved}});
    };

    if (!input.requires_grad() || !is_grad_enabled()) {
        // No gradient needed, just compute (and possibly record a trace).
        auto out_t = compute();
        record(out_t);
        return Variable(out_t, false);
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
    auto result_tensor = compute();
    record(result_tensor);
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto permute(const Variable& input, const std::vector<int64_t>& dims) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        // No gradient needed, just compute
        auto out_t = tenzor::permute(input.tensor(), dims);
        if (jit_tracing_active()) {
            jit_record_shape_op(::tenzor::jit::OpType::Permute, {&input},
                                out_t,
                                {},
                                {{"dims", dims}});
        }
        return Variable(out_t, false);
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
    if (jit_tracing_active()) {
        jit_record_shape_op(::tenzor::jit::OpType::Permute, {&input},
                            result_tensor,
                            {},
                            {{"dims", dims}});
    }
    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto transpose(const Variable& input, int64_t dim0, int64_t dim1) -> Variable {
    // Normalise negative dims for the trace; the lowering expects [0, rank).
    int64_t recorded_dim0 = dim0;
    int64_t recorded_dim1 = dim1;
    {
        auto rank = static_cast<int64_t>(input.shape().size());
        if (recorded_dim0 < 0) recorded_dim0 += rank;
        if (recorded_dim1 < 0) recorded_dim1 += rank;
    }

    auto compute = [&]() {
        return tenzor::transpose(input.tensor(), dim0, dim1);
    };
    auto record = [&](const Tensor& t) {
        if (jit_tracing_active()) {
            jit_record_shape_op(::tenzor::jit::OpType::Transpose, {&input}, t,
                                {{"dim0", recorded_dim0},
                                 {"dim1", recorded_dim1}});
        }
    };

    if (!input.requires_grad() || !is_grad_enabled()) {
        auto out_t = compute();
        record(out_t);
        return Variable(out_t, false);
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

    auto result_tensor = compute();
    record(result_tensor);
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
    // Normalise negative dim for the trace.
    int64_t recorded_dim = dim;
    if (!inputs.empty()) {
        auto ndim = static_cast<int64_t>(inputs[0].shape().size());
        if (recorded_dim < 0) recorded_dim += ndim;
    }

    auto record = [&](const Tensor& t) {
        if (!jit_tracing_active()) return;
        auto& tracer = ::tenzor::jit::Tracer::get_instance();
        std::vector<std::string> input_ids;
        input_ids.reserve(inputs.size());
        for (const auto& v : inputs) {
            input_ids.push_back(tracer.register_tensor(v.tensor()));
        }
        auto out_id = tracer.register_new_tensor(t);
        ::tenzor::jit::TracedOp op(::tenzor::jit::OpType::Cat,
                                   std::move(input_ids), {out_id});
        op.int_attrs["dim"] = recorded_dim;
        tracer.record_op(std::move(op));
    };

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
        auto result = tenzor::cat(tensors, dim);
        record(result);
        return Variable(result, false);
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
    record(result_tensor);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);

    return output;
}

auto slice(const Variable& input, int64_t dim, int64_t start, int64_t end, int64_t step) -> Variable {
    // Normalise dim for the trace so the lowering doesn't have to.
    int64_t recorded_dim = dim;
    if (recorded_dim < 0) {
        recorded_dim += static_cast<int64_t>(input.shape().size());
    }

    auto compute = [&]() {
        return input.tensor().slice(dim, start, end, step);
    };
    auto record = [&](const Tensor& t) {
        if (jit_tracing_active()) {
            jit_record_shape_op(::tenzor::jit::OpType::Slice, {&input}, t,
                                {{"dim", recorded_dim},
                                 {"start", start},
                                 {"end", end},
                                 {"step", step}});
        }
    };

    if (!input.requires_grad() || !is_grad_enabled()) {
        auto result = compute();
        record(result);
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
    auto result_tensor = compute();
    record(result_tensor);

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
    // Forward must pass alpha as an attribute so backend kernels use the
    // requested scale — previously the dispatch was called with no attrs
    // and Elu fell back to its default alpha=1.0, producing silently
    // wrong outputs when callers asked for a non-unit alpha.
    return unary_autograd_with_param<EluBackward>(input, alpha,
        [alpha](const Tensor& t) {
            std::vector<Tensor> inputs = {t};
            OpAttributes attrs;
            attrs.set(AttrKey::Alpha, static_cast<double>(alpha));
            return dispatch(OpId::Elu, inputs, attrs)[0];
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

auto leaky_relu(const Variable& input, double negative_slope) -> Variable {
    return unary_autograd_with_param<LeakyReluBackward>(input, negative_slope,
        [negative_slope](const Tensor& t) {
            std::vector<Tensor> inputs = {t};
            OpAttributes attrs;
            // Every backend kernel reads AttrKey::Alpha; MPS and the JIT
            // tracer read AttrKey::Negative_slope. Set both so the value
            // isn't silently dropped on either side. Stored as double for
            // Float64-input parity (passing float here loses ~7 mantissa
            // bits and breaks gradcheck / forward exact-equality).
            attrs.set(AttrKey::Alpha, negative_slope);
            attrs.set(AttrKey::Negative_slope, negative_slope);
            return dispatch(OpId::LeakyReLU, inputs, attrs)[0];
        });
}

auto softplus(const Variable& input, float beta) -> Variable {
    // Same story as elu above: forward must pass beta through OpAttributes.
    return unary_autograd_with_param<SoftplusBackward>(input, beta,
        [beta](const Tensor& t) {
            std::vector<Tensor> inputs = {t};
            OpAttributes attrs;
            attrs.set(AttrKey::Beta, static_cast<double>(beta));
            return dispatch(OpId::Softplus, inputs, attrs)[0];
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

auto erfinv(const Variable& input) -> Variable {
    // ErfInvBackward saves the output (not input) for efficient backward
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::erfinv(input.tensor()), false);
    }
    auto grad_fn = std::make_shared<ErfInvBackward>();
    auto result = tenzor::erfinv(input.tensor());
    grad_fn->save_for_backward({result});  // Save output
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto gamma(const Variable& input) -> Variable {
    return unary_autograd<GammaBackward>(input,
        [](const Tensor& t) { return tenzor::gamma(t); });
}

auto lgamma(const Variable& input) -> Variable {
    return unary_autograd<LgammaBackward>(input,
        [](const Tensor& t) { return tenzor::lgamma(t); });
}

auto digamma(const Variable& input) -> Variable {
    return unary_autograd<DigammaBackward>(input,
        [](const Tensor& t) { return tenzor::digamma(t); });
}

auto polygamma(int64_t n, const Variable& input) -> Variable {
    // Polygamma: d/dx ψ^(n)(x) = ψ^(n+1)(x).
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::polygamma(n, input.tensor()), false);
    }
    struct Grad : public Function {
        int64_t n;
        explicit Grad(int64_t nn) : n(nn) {}
        auto forward(std::vector<Variable>) -> std::vector<Variable> override {
            throw std::runtime_error("PolygammaBackward::forward should not be called");
        }
        auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
            const auto& grad = grad_outputs[0];
            const auto& x = saved_tensors_[0];
            auto next = tenzor::polygamma(n + 1, x);
            return {mul(grad, next)};
        }
    };
    auto result = tenzor::polygamma(n, input.tensor());
    auto grad_fn = std::make_shared<Grad>(n);
    grad_fn->save_for_backward({input.tensor()});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto bessel_i0(const Variable& input) -> Variable {
    return unary_autograd<BesselI0Backward>(input,
        [](const Tensor& t) { return tenzor::bessel_i0(t); });
}

auto bessel_i1(const Variable& input) -> Variable {
    return unary_autograd<BesselI1Backward>(input,
        [](const Tensor& t) { return tenzor::bessel_i1(t); });
}

auto sinc(const Variable& input) -> Variable {
    return unary_autograd<SincBackward>(input,
        [](const Tensor& t) { return tenzor::sinc(t); });
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
    grad_fn->input_shape_y_ = std::vector<int64_t>(y.shape().begin(), y.shape().end());
    grad_fn->input_shape_x_ = std::vector<int64_t>(x.shape().begin(), x.shape().end());
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
    std::vector<Tensor> saved = {input.tensor(), result};
    if (dim.has_value()) {
        Tensor dim_t({1}, DType::Int64, Device::cpu());
        dim_t.data<int64_t>()[0] = *dim;
        saved.push_back(dim_t);
    }
    grad_fn->save_for_backward(std::move(saved));
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
    // StdBackward reads dim from saved_tensors_[2] when present; saving it
    // keeps the reduction axis correct for dim-specific gradients.
    std::vector<Tensor> saved = {input.tensor(), result};
    if (dim.has_value()) {
        Tensor dim_t({1}, DType::Int64, Device::cpu());
        dim_t.data<int64_t>()[0] = *dim;
        saved.push_back(dim_t);
    }
    grad_fn->save_for_backward(std::move(saved));
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
    // VarBackward::backward reads saved_tensors_[0]=input and [1]=result; when a
    // dim is supplied, it optionally reads [2]=dim as an Int64 scalar tensor.
    std::vector<Tensor> saved = {input.tensor(), result};
    if (dim.has_value()) {
        Tensor dim_t({1}, DType::Int64, Device::cpu());
        dim_t.data<int64_t>()[0] = *dim;
        saved.push_back(dim_t);
    }
    grad_fn->save_for_backward(std::move(saved));
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
    std::vector<Tensor> saved = {input.tensor(), result};
    if (dim.has_value()) {
        Tensor dim_t({1}, DType::Int64, Device::cpu());
        dim_t.data<int64_t>()[0] = *dim;
        saved.push_back(dim_t);
    }
    grad_fn->save_for_backward(std::move(saved));
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
    // LogSumExpBackward unconditionally reads saved_tensors_[2] as the dim;
    // the dim is always present for this op (required parameter).
    Tensor dim_t({1}, DType::Int64, Device::cpu());
    dim_t.data<int64_t>()[0] = dim;
    grad_fn->save_for_backward({input.tensor(), result, dim_t});
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
    auto dim_tensor = zeros({1}, DType::Int64, Device::cpu());
    dim_tensor.data<int64_t>()[0] = dim;
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
    const auto& sh = input.tensor().shape();
    auto shape_tensor = zeros({static_cast<int64_t>(sh.size())}, DType::Int64, Device::cpu());
    std::memcpy(shape_tensor.data_ptr(), sh.data(), sh.size() * sizeof(int64_t));
    grad_fn->save_for_backward({shape_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto to_device(const Variable& input, Device target) -> Variable {
    if (input.tensor().device() == target) {
        return input;
    }
    auto result = input.tensor().to(target);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<DeviceTransferBackward>();
    grad_fn->source_device = input.tensor().device();
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
    const auto& sh = input.tensor().shape();
    auto shape_tensor = zeros({static_cast<int64_t>(sh.size())}, DType::Int64, Device::cpu());
    std::memcpy(shape_tensor.data_ptr(), sh.data(), sh.size() * sizeof(int64_t));
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
    // Condition is non-differentiable; only x and y carry gradients.
    // WhereBackward::backward returns {grad_x, grad_y} in that order, so
    // input_variables / next_functions must match — previously this
    // passed `{condition, x, y}` which misaligned the engine's index-
    // based gradient accumulation and pushed grad_y into x (and dropped
    // y's gradient entirely). The condition tensor is still kept in
    // saved_tensors_ so the backward can read it.
    grad_fn->save_for_backward({condition.tensor()});
    grad_fn->set_next_functions({x.grad_fn(), y.grad_fn()});
    grad_fn->set_input_variables({x, y});
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
    auto dim_tensor = zeros({1}, DType::Int64, Device::cpu());
    dim_tensor.data<int64_t>()[0] = dim;
    // Save input shape for backward scatter_add
    const auto& sh = input.tensor().shape();
    auto shape_tensor = zeros({static_cast<int64_t>(sh.size())}, DType::Int64, Device::cpu());
    std::memcpy(shape_tensor.data_ptr(), sh.data(), sh.size() * sizeof(int64_t));
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
    auto dim_tensor = zeros({1}, DType::Int64, Device::cpu());
    dim_tensor.data<int64_t>()[0] = dim;
    grad_fn->save_for_backward({dim_tensor, index});
    grad_fn->set_next_functions({input.grad_fn(), src.grad_fn()});
    grad_fn->set_input_variables({input, src});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto scatter_add(const Variable& input, int64_t dim, const Tensor& index, const Variable& src) -> Variable {
    bool needs_grad = (input.requires_grad() || src.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(tenzor::scatter_add(input.tensor(), dim, index, src.tensor()), false);
    }
    auto result = tenzor::scatter_add(input.tensor(), dim, index, src.tensor());
    auto grad_fn = std::make_shared<ScatterAddBackward>();
    auto dim_tensor = zeros({1}, DType::Int64, Device::cpu());
    dim_tensor.data<int64_t>()[0] = dim;
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
    auto dim_tensor = zeros({1}, DType::Int64, Device::cpu());
    dim_tensor.data<int64_t>()[0] = dim;
    const auto& sh = input.tensor().shape();
    auto shape_tensor = zeros({static_cast<int64_t>(sh.size())}, DType::Int64, Device::cpu());
    std::memcpy(shape_tensor.data_ptr(), sh.data(), sh.size() * sizeof(int64_t));
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
    auto dim_tensor = zeros({1}, DType::Int64, Device::cpu());
    dim_tensor.data<int64_t>()[0] = dim;
    auto start_tensor = zeros({1}, DType::Int64, Device::cpu());
    start_tensor.data<int64_t>()[0] = start;
    const auto& sh = input.tensor().shape();
    auto shape_tensor = zeros({static_cast<int64_t>(sh.size())}, DType::Int64, Device::cpu());
    std::memcpy(shape_tensor.data_ptr(), sh.data(), sh.size() * sizeof(int64_t));
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
    auto dims_tensor = zeros({static_cast<int64_t>(dims.size())}, DType::Int64, Device::cpu());
    auto* ptr = dims_tensor.data<int64_t>();
    for (size_t i = 0; i < dims.size(); ++i) ptr[i] = dims[i];
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
    const auto& sh = input.tensor().shape();
    auto shape_tensor = zeros({static_cast<int64_t>(sh.size())}, DType::Int64, Device::cpu());
    std::memcpy(shape_tensor.data_ptr(), sh.data(), sh.size() * sizeof(int64_t));
    auto repeats_tensor = zeros({static_cast<int64_t>(repeats.size())}, DType::Int64, Device::cpu());
    auto* rptr = repeats_tensor.data<int64_t>();
    for (size_t i = 0; i < repeats.size(); ++i) rptr[i] = repeats[i];
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

    // audit-2026-05-03 bug #1: backward returns {grad_A, grad_B} in input
    // order. The autograd engine zips these against `input_vars` by index,
    // so we must always push BOTH A and B — even when only one
    // requires_grad — otherwise grad_A would be assigned to B (and vice
    // versa), and finite-diff fails for whichever side is "off-by-one".
    grad_fn->set_input_variables({A, B});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ============================================================================
// audit-2026-05-03 Phase 8 — new linalg Variable wrappers.
// ============================================================================

auto lu_solve(const Tensor& LU_data, const Tensor& pivots,
              const Variable& B) -> Variable {
    if (!B.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::linalg::lu_solve(LU_data, pivots, B.tensor()), false);
    }

    // Compute X = lu_solve(LU, pivots, B) AND reconstruct A so backward can
    // call solve(A^T, dL/dX). Reconstruction: A = P^T @ L @ U.
    auto X_tensor = tenzor::linalg::lu_solve(LU_data, pivots, B.tensor());
    // Reconstruct A by applying lu_solve to identity then inverting:
    // simpler — use solve(I, B_via_lu) trick. Here we just save LU and
    // reconstruct on demand. For simplicity compute A via factor extraction.
    // Pragmatic shortcut: save (LU_data, pivots) directly; backward does
    // a lu_solve on the transposed system instead of needing A explicitly.
    // PyTorch uses lu_solve_(transpose=True) for this; we approximate by
    // reconstructing via lu_solve(LU, pivots, eye()) which gives A^{-1},
    // then inverting again. Cheaper: derive A by composing tril(LU,-1) +
    // I + triu(LU, 0) = L + U... but the packed factor layout may differ.
    //
    // For correctness, just reconstruct A symbolically via a separate
    // lu_solve call: A = solve_for_A_st_(LU, pivots).
    // Since reconstructing is a one-time cost, do it via lu_solve(LU, pivots, eye)
    // to get A^{-1}, then inv() to get A.
    auto N = LU_data.shape()[LU_data.ndim() - 1];
    auto eye_t = tenzor::eye(N, std::nullopt, LU_data.dtype(), LU_data.device());
    auto A_inv = tenzor::linalg::lu_solve(LU_data, pivots, eye_t);
    auto A = tenzor::linalg::inv(A_inv);

    auto grad_fn = std::make_shared<LUSolveBackward>();
    grad_fn->save_for_backward({A, X_tensor});
    grad_fn->set_next_functions({B.grad_fn()});
    grad_fn->set_input_variables({B});

    Variable output(X_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto lu(const Variable& A) -> std::tuple<Variable, Variable, Variable> {
    if (!A.requires_grad() || !is_grad_enabled()) {
        auto [L, U, pivots] = tenzor::linalg::lu(A.tensor());
        return {Variable(L, false), Variable(U, false), Variable(pivots, false)};
    }

    auto [L_t, U_t, pivots_t] = tenzor::linalg::lu(A.tensor());

    // audit-2026-05-03 — separate backward instances for L and U so the
    // per-output gradients accumulate at distinct engine slots. Pivots are
    // non-differentiable; pivots_var has requires_grad=false.
    auto grad_fn_L = std::make_shared<LUBackward>(/*output_slot=*/0);
    grad_fn_L->save_for_backward({L_t, U_t, pivots_t});
    grad_fn_L->set_next_functions({A.grad_fn()});
    grad_fn_L->set_input_variables({A});

    auto grad_fn_U = std::make_shared<LUBackward>(/*output_slot=*/1);
    grad_fn_U->save_for_backward({L_t, U_t, pivots_t});
    grad_fn_U->set_next_functions({A.grad_fn()});
    grad_fn_U->set_input_variables({A});

    Variable L_var(L_t, true);
    Variable U_var(U_t, true);
    Variable pivots_var(pivots_t, false);

    L_var.set_grad_fn(grad_fn_L);
    U_var.set_grad_fn(grad_fn_U);

    return {L_var, U_var, pivots_var};
}

auto eig(const Variable& A) -> std::tuple<Variable, Variable, Variable> {
    if (!A.requires_grad() || !is_grad_enabled()) {
        auto [W_re, W_im, V] = tenzor::linalg::eig(A.tensor());
        return {Variable(W_re, false), Variable(W_im, false), Variable(V, false)};
    }

    auto [W_re_t, W_im_t, V_t] = tenzor::linalg::eig(A.tensor());

    auto grad_fn = std::make_shared<EigBackward>();
    // Save W_real, W_imag, V so the backward can use all of them.
    // Previously only W_real and V were saved, which prevented
    // EigBackward from detecting complex eigenvalues (audit item A.10).
    grad_fn->save_for_backward({W_re_t, W_im_t, V_t});
    grad_fn->set_next_functions({A.grad_fn()});
    grad_fn->set_input_variables({A});

    Variable W_re_var(W_re_t, true);
    Variable W_im_var(W_im_t, true);
    Variable V_var(V_t, true);

    W_re_var.set_grad_fn(grad_fn);
    W_im_var.set_grad_fn(grad_fn);
    V_var.set_grad_fn(grad_fn);

    return {W_re_var, W_im_var, V_var};
}

// ============================================================================
// audit-2026-05-03 Phase 11 — N-D FFT Variable wrappers via composition.
// ============================================================================

auto fft2(const Variable& input,
          std::optional<std::vector<int64_t>> s,
          std::vector<int64_t> dim,
          const std::string& norm) -> Variable {
    auto out = input;
    for (size_t i = 0; i < dim.size(); ++i) {
        std::optional<int64_t> n_i;
        if (s.has_value()) n_i = s.value()[i];
        out = ::tenzor::fft_autograd::fft(out, n_i, dim[i], norm);
    }
    return out;
}

auto ifft2(const Variable& input,
           std::optional<std::vector<int64_t>> s,
           std::vector<int64_t> dim,
           const std::string& norm) -> Variable {
    auto out = input;
    for (size_t i = 0; i < dim.size(); ++i) {
        std::optional<int64_t> n_i;
        if (s.has_value()) n_i = s.value()[i];
        out = ::tenzor::fft_autograd::ifft(out, n_i, dim[i], norm);
    }
    return out;
}

auto fftn(const Variable& input,
          std::optional<std::vector<int64_t>> s,
          std::optional<std::vector<int64_t>> dim,
          const std::string& norm) -> Variable {
    std::vector<int64_t> dims;
    if (dim.has_value()) {
        dims = dim.value();
    } else {
        // Default: all dimensions.
        const auto ndim = static_cast<int64_t>(input.shape().size());
        for (int64_t d = 0; d < ndim; ++d) dims.push_back(d);
    }
    auto out = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        std::optional<int64_t> n_i;
        if (s.has_value()) n_i = s.value()[i];
        out = ::tenzor::fft_autograd::fft(out, n_i, dims[i], norm);
    }
    return out;
}

auto ifftn(const Variable& input,
           std::optional<std::vector<int64_t>> s,
           std::optional<std::vector<int64_t>> dim,
           const std::string& norm) -> Variable {
    std::vector<int64_t> dims;
    if (dim.has_value()) {
        dims = dim.value();
    } else {
        const auto ndim = static_cast<int64_t>(input.shape().size());
        for (int64_t d = 0; d < ndim; ++d) dims.push_back(d);
    }
    auto out = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        std::optional<int64_t> n_i;
        if (s.has_value()) n_i = s.value()[i];
        out = ::tenzor::fft_autograd::ifft(out, n_i, dims[i], norm);
    }
    return out;
}

auto cholesky_solve(const Variable& B, const Variable& L, bool upper) -> Variable {
    if ((!B.requires_grad() && !L.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::linalg::cholesky_solve(B.tensor(), L.tensor(), upper), false);
    }

    auto result_tensor = tenzor::linalg::cholesky_solve(B.tensor(), L.tensor(), upper);

    auto grad_fn = std::make_shared<CholeskySolveBackward>(upper);
    grad_fn->save_for_backward({result_tensor, L.tensor()});

    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(B.grad_fn());
    next_funcs.push_back(L.grad_fn());
    grad_fn->set_next_functions(next_funcs);

    // Same fix as solve(): always push both inputs in declared order so the
    // engine's grad-zip lines up with backward's {grad_B, grad_L} return.
    grad_fn->set_input_variables({B, L});

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

    // audit-2026-05-03 — per-output Function instances so the engine doesn't
    // collapse the U / S / Vh gradients into a single accumulator entry.
    auto make_grad_fn = [&](int slot) {
        auto fn = std::make_shared<SvdBackward>(full_matrices, slot);
        fn->save_for_backward({U_tensor, S_tensor, Vh_tensor});
        fn->set_next_functions({input.grad_fn()});
        fn->set_input_variables({input});
        return fn;
    };

    Variable U_var(U_tensor, true);
    Variable S_var(S_tensor, true);
    Variable Vh_var(Vh_tensor, true);

    U_var.set_grad_fn(make_grad_fn(0));
    S_var.set_grad_fn(make_grad_fn(1));
    Vh_var.set_grad_fn(make_grad_fn(2));

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

auto sparse_add(const SparseTensor& sparse, const Variable& dense) -> Variable {
    if (!dense.requires_grad() || !is_grad_enabled()) {
        return Variable(sparse::add(sparse, dense.tensor()), false);
    }

    // Compute forward: Y = S + D
    auto result_tensor = sparse::add(sparse, dense.tensor());

    auto grad_fn = std::make_shared<SparseAddBackward>();
    grad_fn->set_next_functions({dense.grad_fn()});
    grad_fn->set_input_variables({dense});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto sparse_triangular_solve(const SparseTensor& L,
                              const Variable& b,
                              bool upper) -> Variable {
    auto result_tensor = sparse::sparse_triangular_solve(L, b.tensor(), upper);
    if (!b.requires_grad() || !is_grad_enabled()) {
        return Variable(result_tensor, false);
    }

    // SparseTriSolveBackward needs L^T as a sparse matrix to compute
    // grad_b = L^{-T} @ grad_x. Pre-compute the transpose once.
    auto Lt = L.transpose();

    auto grad_fn = std::make_shared<SparseTriSolveBackward>();
    grad_fn->set_sparse_l_transposed(std::move(Lt));
    grad_fn->set_upper(upper);
    grad_fn->save_for_backward({result_tensor});
    grad_fn->set_next_functions({b.grad_fn()});
    grad_fn->set_input_variables({b});

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

// ============================================================================
// Phase A.3 — STFT / ISTFT Variable wrappers.
// ============================================================================

auto stft(const Variable& input,
          int64_t n_fft,
          int64_t hop_length,
          int64_t win_length,
          const Tensor& window,
          bool center,
          bool normalized,
          bool onesided) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::fft::stft(input.tensor(), n_fft, hop_length, win_length,
                                           window, center, normalized, onesided),
                        false);
    }
    int64_t signal_length = input.tensor().shape().back();
    auto result = tenzor::fft::stft(input.tensor(), n_fft, hop_length, win_length,
                                     window, center, normalized, onesided);
    auto grad_fn = std::make_shared<STFTBackward>(
        n_fft, hop_length, win_length, window, center, normalized, onesided,
        signal_length);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto istft(const Variable& input,
           int64_t n_fft,
           int64_t hop_length,
           int64_t win_length,
           const Tensor& window,
           bool center,
           bool normalized,
           bool onesided,
           std::optional<int64_t> length) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::fft::istft(input.tensor(), n_fft, hop_length, win_length,
                                            window, center, normalized, onesided, length),
                        false);
    }
    auto result = tenzor::fft::istft(input.tensor(), n_fft, hop_length, win_length,
                                      window, center, normalized, onesided, length);
    auto grad_fn = std::make_shared<ISTFTBackward>(
        n_fft, hop_length, win_length, window, center, normalized, onesided);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

} // namespace fft_autograd

// ============================================================================
// Custom Op Autograd Dispatch
// ============================================================================

auto dispatch_custom_op(CustomOpId id,
                        std::vector<Variable> inputs,
                        const OpAttributes& attrs) -> Variable {
    // Check if any input requires grad
    bool any_requires_grad = false;
    for (const auto& v : inputs) {
        if (v.requires_grad()) {
            any_requires_grad = true;
            break;
        }
    }

    // Extract raw tensors for forward dispatch
    std::vector<Tensor> raw_inputs;
    raw_inputs.reserve(inputs.size());
    for (const auto& v : inputs) {
        raw_inputs.push_back(v.tensor());
    }

    // Forward compute via raw dispatch
    auto result = tenzor::dispatch_custom_op(id, raw_inputs, attrs);

    if (!any_requires_grad || !is_grad_enabled()) {
        return Variable(result, false);
    }

    // Check if this custom op has a registered backward
    auto& registry = CustomOpRegistry::instance();
    auto backward_info = registry.get_backward(id);
    if (!backward_info) {
        // No backward registered — return result without grad tracking
        return Variable(result, false);
    }

    auto& [backward_fn, save_fn] = *backward_info;

    // Audit D2: also fetch the optional Variable-level backward. When
    // present, `CustomOpBackward::backward_with_variables` will preserve
    // the autograd graph for higher-order grads; when absent, the op
    // honestly reports `is_higher_order_stub() = true`.
    auto var_backward_opt = registry.get_var_backward(id);
    auto grad_fn = var_backward_opt
        ? std::make_shared<CustomOpBackward>(backward_fn, *var_backward_opt)
        : std::make_shared<CustomOpBackward>(backward_fn);

    // Determine what to save for backward
    if (save_fn) {
        grad_fn->save_for_backward(save_fn(raw_inputs, result));
    } else {
        // Default: save all inputs
        grad_fn->save_for_backward(std::vector<Tensor>(raw_inputs.begin(), raw_inputs.end()));
    }

    // Set up backward graph connections
    std::vector<std::shared_ptr<Function>> next_funcs;
    std::vector<Variable> input_vars;
    next_funcs.reserve(inputs.size());
    for (const auto& v : inputs) {
        next_funcs.push_back(v.grad_fn());
        if (v.requires_grad()) {
            input_vars.push_back(v);
        }
    }
    grad_fn->set_next_functions(std::move(next_funcs));
    grad_fn->set_input_variables(std::move(input_vars));

    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ============================================================================
// New Op Autograd Wrappers (Phase 7)
// ============================================================================

auto logaddexp(const Variable& a, const Variable& b) -> Variable {
    if ((!a.requires_grad() && !b.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::logaddexp(a.tensor(), b.tensor()), false);
    }

    auto result_tensor = tenzor::logaddexp(a.tensor(), b.tensor());

    auto grad_fn = std::make_shared<LogAddExpBackward>();
    grad_fn->save_for_backward({a.tensor(), b.tensor()});
    grad_fn->input_shape_a_ = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    grad_fn->input_shape_b_ = std::vector<int64_t>(b.shape().begin(), b.shape().end());

    std::vector<std::shared_ptr<Function>> next_funcs = {a.grad_fn(), b.grad_fn()};
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (a.requires_grad()) input_vars.push_back(a);
    if (b.requires_grad()) input_vars.push_back(b);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto logaddexp2(const Variable& a, const Variable& b) -> Variable {
    if ((!a.requires_grad() && !b.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::logaddexp2(a.tensor(), b.tensor()), false);
    }

    auto result_tensor = tenzor::logaddexp2(a.tensor(), b.tensor());

    auto grad_fn = std::make_shared<LogAddExp2Backward>();
    grad_fn->save_for_backward({a.tensor(), b.tensor(), result_tensor});
    grad_fn->input_shape_a_ = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    grad_fn->input_shape_b_ = std::vector<int64_t>(b.shape().begin(), b.shape().end());

    std::vector<std::shared_ptr<Function>> next_funcs = {a.grad_fn(), b.grad_fn()};
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (a.requires_grad()) input_vars.push_back(a);
    if (b.requires_grad()) input_vars.push_back(b);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto xlogy(const Variable& x, const Variable& y) -> Variable {
    if ((!x.requires_grad() && !y.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::xlogy(x.tensor(), y.tensor()), false);
    }

    auto result_tensor = tenzor::xlogy(x.tensor(), y.tensor());

    auto grad_fn = std::make_shared<XLogYBackward>();
    grad_fn->save_for_backward({x.tensor(), y.tensor()});
    grad_fn->input_shape_x_ = std::vector<int64_t>(x.shape().begin(), x.shape().end());
    grad_fn->input_shape_y_ = std::vector<int64_t>(y.shape().begin(), y.shape().end());

    std::vector<std::shared_ptr<Function>> next_funcs = {x.grad_fn(), y.grad_fn()};
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (x.requires_grad()) input_vars.push_back(x);
    if (y.requires_grad()) input_vars.push_back(y);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto i0e(const Variable& input) -> Variable {
    return unary_autograd<I0eBackward>(input,
        [](const Tensor& t) { return tenzor::i0e(t); });
}

auto i1e(const Variable& input) -> Variable {
    return unary_autograd<I1eBackward>(input,
        [](const Tensor& t) { return tenzor::i1e(t); });
}

auto entr(const Variable& input) -> Variable {
    return unary_autograd<EntrBackward>(input,
        [](const Tensor& t) { return tenzor::entr(t); });
}

auto spherical_bessel_j0(const Variable& input) -> Variable {
    return unary_autograd<SphericalBesselJ0Backward>(input,
        [](const Tensor& t) { return tenzor::spherical_bessel_j0(t); });
}

// Phase 12 — Bessel J0/J1/Y0/Y1 and Zeta autograd wrappers.
auto bessel_j0(const Variable& input) -> Variable {
    return unary_autograd<BesselJ0Backward>(input,
        [](const Tensor& t) { return tenzor::bessel_j0(t); });
}
auto bessel_j1(const Variable& input) -> Variable {
    return unary_autograd<BesselJ1Backward>(input,
        [](const Tensor& t) { return tenzor::bessel_j1(t); });
}
auto bessel_y0(const Variable& input) -> Variable {
    return unary_autograd<BesselY0Backward>(input,
        [](const Tensor& t) { return tenzor::bessel_y0(t); });
}
auto bessel_y1(const Variable& input) -> Variable {
    return unary_autograd<BesselY1Backward>(input,
        [](const Tensor& t) { return tenzor::bessel_y1(t); });
}

// zeta(s, q) — only q gets a non-trivial gradient (-s * zeta(s+1, q)).
auto zeta(const Variable& s, const Variable& q) -> Variable {
    if ((!s.requires_grad() && !q.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::zeta(s.tensor(), q.tensor()), false);
    }
    auto result = tenzor::zeta(s.tensor(), q.tensor());
    auto grad_fn = std::make_shared<ZetaBackward>();
    grad_fn->save_for_backward({s.tensor(), q.tensor()});
    grad_fn->set_next_functions({s.grad_fn(), q.grad_fn()});
    grad_fn->set_input_variables({s, q});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// betainc(a, b, x) — only x gets a non-trivial gradient.
auto betainc(const Variable& a, const Variable& b, const Variable& x) -> Variable {
    if ((!a.requires_grad() && !b.requires_grad() && !x.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::betainc(a.tensor(), b.tensor(), x.tensor()), false);
    }
    auto result = tenzor::betainc(a.tensor(), b.tensor(), x.tensor());
    auto grad_fn = std::make_shared<BetaIncBackward>();
    grad_fn->save_for_backward({a.tensor(), b.tensor(), x.tensor()});
    grad_fn->set_next_functions({a.grad_fn(), b.grad_fn(), x.grad_fn()});
    grad_fn->set_input_variables({a, b, x});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto ndtr(const Variable& input) -> Variable {
    return unary_autograd<NdtrBackward>(input,
        [](const Tensor& t) { return tenzor::ndtr(t); });
}

auto log_ndtr(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::log_ndtr(input.tensor()), false);
    }
    auto result = tenzor::log_ndtr(input.tensor());
    auto grad_fn = std::make_shared<LogNdtrBackward>();
    grad_fn->save_for_backward({input.tensor(), result});  // Save both input and output
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto multigammaln(const Variable& input, int64_t p) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::multigammaln(input.tensor(), p), false);
    }
    auto grad_fn = std::make_shared<MultigammalnBackward>(p);
    grad_fn->save_for_backward({input.tensor()});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    auto result = tenzor::multigammaln(input.tensor(), p);
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto cosine_similarity(const Variable& x1, const Variable& x2,
                       int64_t dim, double eps) -> Variable {
    if ((!x1.requires_grad() && !x2.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::cosine_similarity(x1.tensor(), x2.tensor(), dim, eps), false);
    }

    auto result_tensor = tenzor::cosine_similarity(x1.tensor(), x2.tensor(), dim, eps);

    auto grad_fn = std::make_shared<CosineSimilarityBackward>(dim, eps);
    grad_fn->save_for_backward({x1.tensor(), x2.tensor(), result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs = {x1.grad_fn(), x2.grad_fn()};
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (x1.requires_grad()) input_vars.push_back(x1);
    if (x2.requires_grad()) input_vars.push_back(x2);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto renorm(const Variable& input, double p, int64_t dim, double maxnorm) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::renorm(input.tensor(), p, dim, maxnorm), false);
    }

    auto result_tensor = tenzor::renorm(input.tensor(), p, dim, maxnorm);

    auto grad_fn = std::make_shared<RenormBackward>(p, dim, maxnorm);
    grad_fn->save_for_backward({input.tensor(), result_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto cholesky_inverse(const Variable& input, bool upper) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::linalg::cholesky_inverse(input.tensor(), upper), false);
    }

    auto result_tensor = tenzor::linalg::cholesky_inverse(input.tensor(), upper);

    auto grad_fn = std::make_shared<CholeskyInverseBackward>(upper);
    grad_fn->save_for_backward({input.tensor(), result_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto ldl_factor(const Variable& input) -> std::tuple<Variable, Variable> {
    auto [LD_tensor, pivots_tensor] = tenzor::linalg::ldl_factor(input.tensor());

    if (!input.requires_grad() || !is_grad_enabled()) {
        return {Variable(LD_tensor, false), Variable(pivots_tensor, false)};
    }

    auto grad_fn = std::make_shared<LinalgLDLFactorBackward>();
    // audit-2026-05-03 Phase 12 — save A and LD so backward can run the
    // structural-symmetric backprop derived from A = L D L^T.
    grad_fn->save_for_backward({input.tensor(), LD_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable LD_out(LD_tensor, true);
    LD_out.set_grad_fn(grad_fn);
    // Pivots are integer indices, not differentiable
    Variable pivots_out(pivots_tensor, false);
    return {LD_out, pivots_out};
}

auto ldl_solve(const Variable& LD, const Tensor& pivots, const Variable& B) -> Variable {
    if ((!LD.requires_grad() && !B.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::linalg::ldl_solve(LD.tensor(), pivots, B.tensor()), false);
    }

    auto result_tensor = tenzor::linalg::ldl_solve(LD.tensor(), pivots, B.tensor());

    auto grad_fn = std::make_shared<LinalgLDLSolveBackward>();
    grad_fn->save_for_backward({LD.tensor(), pivots});

    // Only B gradient is computed (LD gradient would require complex structured backprop)
    std::vector<std::shared_ptr<Function>> next_funcs = {B.grad_fn()};
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (B.requires_grad()) input_vars.push_back(B);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto householder_product(const Variable& input, const Variable& tau) -> Variable {
    if ((!input.requires_grad() && !tau.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::linalg::householder_product(input.tensor(), tau.tensor()), false);
    }

    auto result_tensor = tenzor::linalg::householder_product(input.tensor(), tau.tensor());

    auto grad_fn = std::make_shared<LinalgHouseholderBackward>();
    grad_fn->save_for_backward({input.tensor(), tau.tensor()});

    std::vector<std::shared_ptr<Function>> next_funcs = {input.grad_fn(), tau.grad_fn()};
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (input.requires_grad()) input_vars.push_back(input);
    if (tau.requires_grad()) input_vars.push_back(tau);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto tensorinv(const Variable& input, int64_t ind) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::linalg::tensorinv(input.tensor(), ind), false);
    }

    auto result_tensor = tenzor::linalg::tensorinv(input.tensor(), ind);

    auto grad_fn = std::make_shared<TensorInvBackward>(ind);
    grad_fn->save_for_backward({result_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto tensorsolve(const Variable& A, const Variable& B) -> Variable {
    if ((!A.requires_grad() && !B.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::linalg::tensorsolve(A.tensor(), B.tensor()), false);
    }

    auto result_tensor = tenzor::linalg::tensorsolve(A.tensor(), B.tensor());

    auto grad_fn = std::make_shared<TensorSolveBackward>();
    grad_fn->save_for_backward({A.tensor(), B.tensor(), result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs = {A.grad_fn(), B.grad_fn()};
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (A.requires_grad()) input_vars.push_back(A);
    if (B.requires_grad()) input_vars.push_back(B);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto vector_norm(const Variable& input, double ord,
                 std::vector<int64_t> dim, bool keepdim) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::linalg::vector_norm(input.tensor(), ord, dim, keepdim), false);
    }

    auto result_tensor = tenzor::linalg::vector_norm(input.tensor(), ord, dim, keepdim);

    auto grad_fn = std::make_shared<LinalgVectorNormBackward>(ord, dim, keepdim);
    grad_fn->save_for_backward({input.tensor(), result_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto matrix_norm(const Variable& input, double ord) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::linalg::matrix_norm(input.tensor(), ord), false);
    }

    auto result_tensor = tenzor::linalg::matrix_norm(input.tensor(), ord);

    auto grad_fn = std::make_shared<LinalgMatrixNormBackward>(ord);
    grad_fn->save_for_backward({input.tensor(), result_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto vecdot(const Variable& a, const Variable& b, int64_t dim) -> Variable {
    if ((!a.requires_grad() && !b.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::linalg::vecdot(a.tensor(), b.tensor(), dim), false);
    }

    auto result_tensor = tenzor::linalg::vecdot(a.tensor(), b.tensor(), dim);

    auto grad_fn = std::make_shared<LinalgVecdotBackward>(dim);
    grad_fn->save_for_backward({a.tensor(), b.tensor()});

    std::vector<std::shared_ptr<Function>> next_funcs = {a.grad_fn(), b.grad_fn()};
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    if (a.requires_grad()) input_vars.push_back(a);
    if (b.requires_grad()) input_vars.push_back(b);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto as_strided(const Variable& input, std::span<const int64_t> size,
                std::span<const int64_t> stride,
                std::optional<int64_t> storage_offset) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::as_strided(input.tensor(), size, stride, storage_offset), false);
    }

    auto result_tensor = tenzor::as_strided(input.tensor(), size, stride, storage_offset);

    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto size_vec = std::vector<int64_t>(size.begin(), size.end());
    auto stride_vec = std::vector<int64_t>(stride.begin(), stride.end());

    auto grad_fn = std::make_shared<AsStridedBackward>(
        input_shape, size_vec, stride_vec, storage_offset);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ============================================================================
// View-as-real / view-as-complex Variable wrappers.
// Useful for reducing complex outputs to real (e.g., wrapping FFT for
// gradcheck). The *Backward classes are defined in function_shape.cpp.
// ============================================================================

auto view_as_real(const Variable& input) -> Variable {
    auto out_t = ::tenzor::view_as_real(input.tensor());
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(out_t, false);
    }
    auto grad_fn = std::make_shared<ViewAsRealBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(out_t, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto view_as_complex(const Variable& input) -> Variable {
    auto out_t = ::tenzor::view_as_complex(input.tensor());
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(out_t, false);
    }
    auto grad_fn = std::make_shared<ViewAsComplexBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(out_t, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ============================================================================
// Vision Operations — Variable-level wrappers for grid_sample / affine_grid.
// The *Backward classes already exist (see include/tenzor/autograd/function.hpp
// and src/autograd/function_vision.cpp); the wrappers here expose them through
// the public Variable API for use by gradcheck and end-user code.
// ============================================================================

auto grid_sample(const Variable& input,
                 const Variable& grid,
                 const std::string& mode,
                 const std::string& padding_mode,
                 bool align_corners) -> Variable {
    auto out_t = ::tenzor::ops::grid_sample(input.tensor(), grid.tensor(),
                                            mode, padding_mode, align_corners);
    bool needs_grad = (input.requires_grad() || grid.requires_grad())
                      && is_grad_enabled();
    if (!needs_grad) {
        return Variable(out_t, false);
    }
    auto grad_fn = std::make_shared<GridSampleBackward>();
    grad_fn->mode_ = mode;
    grad_fn->padding_mode_ = padding_mode;
    grad_fn->align_corners_ = align_corners;
    grad_fn->save_for_backward({input.tensor(), grid.tensor()});

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (auto fn = input.grad_fn()) next_funcs.push_back(fn);
    if (auto fn = grid.grad_fn()) next_funcs.push_back(fn);
    grad_fn->set_next_functions(std::move(next_funcs));

    std::vector<Variable> input_vars{input};
    if (grid.requires_grad()) input_vars.push_back(grid);
    grad_fn->set_input_variables(std::move(input_vars));

    Variable output(out_t, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto affine_grid(const Variable& theta,
                 const std::vector<int64_t>& size,
                 bool align_corners) -> Variable {
    auto out_t = ::tenzor::ops::affine_grid(theta.tensor(), size, align_corners);
    if (!theta.requires_grad() || !is_grad_enabled()) {
        return Variable(out_t, false);
    }
    auto grad_fn = std::make_shared<AffineGridBackward>();
    grad_fn->size_ = size;
    grad_fn->align_corners_ = align_corners;
    grad_fn->save_for_backward({theta.tensor()});

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (auto fn = theta.grad_fn()) next_funcs.push_back(fn);
    grad_fn->set_next_functions(std::move(next_funcs));
    grad_fn->set_input_variables({theta});

    Variable output(out_t, true);
    output.set_grad_fn(grad_fn);
    return output;
}

} // namespace tenzor
