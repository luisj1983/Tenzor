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
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/logging.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <optional>
#include <stdexcept>
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
    // GG.1: keep the input Variable for higher-order autograd.
    if (is_creating_graph()) {
        grad_fn->save_variables_for_backward({input});
    }

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
    // GG.1: save the input Variable so backward_with_variables can
    // recompute the output on the live graph (the saved output Tensor
    // can't carry grad_fn back through this same Function — cycle).
    if (is_creating_graph()) {
        grad_fn->save_variables_for_backward({input});
    }

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
    // GG.1: save input as Variable; backward_with_variables recomputes the
    // softmax on the live graph (the saved output Tensor would create a
    // grad_fn cycle if wrapped with requires_grad=true).
    if (is_creating_graph()) {
        grad_fn->save_variables_for_backward({input});
    }

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
    // GG.1: save input as Variable; backward_with_variables recomputes the
    // log_softmax on the live graph.
    if (is_creating_graph()) {
        grad_fn->save_variables_for_backward({input});
    }

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
    // GG.1: keep input Variable for higher-order autograd.
    if (is_creating_graph()) {
        grad_fn->save_variables_for_backward({input});
    }

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

auto clamp(const Variable& input, double min, double max) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::clamp(input.tensor(), min, max), false);
    }

    auto grad_fn = std::make_shared<ClampBackward>(min, max);

    // Save input tensor for backward pass
    grad_fn->save_for_backward({input.tensor()});
    // GG.1: keep input Variable for higher-order autograd.
    if (is_creating_graph()) {
        grad_fn->save_variables_for_backward({input});
    }

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

    // Compute result first so we can record the resolved output shape on the
    // grad_fn — A.4 multi-op JVP traversal needs the post-reshape shape to
    // dispatch the forward-mode rule.
    auto result_tensor = compute();
    record(result_tensor);
    std::vector<int64_t> output_shape(result_tensor.shape().begin(),
                                      result_tensor.shape().end());

    auto grad_fn = std::make_shared<ReshapeBackward>(input_shape, output_shape);

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

    // FF.1: normalise negative dim against input rank before saving so RollBackward
    // sees a canonical axis (mirrors Slice/Chunk/SelectScatter dim handling).
    int64_t normalised_dim = dim;
    auto ndim = static_cast<int64_t>(input.shape().size());
    if (normalised_dim < 0) normalised_dim += ndim;

    auto grad_fn = std::make_shared<RollBackward>(shifts, normalised_dim);
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
    // Normalise dim against input rank up front so SliceBackward never receives
    // a negative axis (UB inside the backward's index-builder which indexes
    // `grad_output.shape()[dim_]` raw). Throw if still out of range — the slice
    // call below would also fail, but catching it here gives a clearer error.
    const int64_t ndim = static_cast<int64_t>(input.shape().size());
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("slice: dim out of range");
    }
    int64_t recorded_dim = dim;

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

// V.10: Variable-level chunk/split as decomposition over autograd::slice.
//
// The raw `tenzor::chunk` / `tenzor::split_with_sizes` operate on `Tensor`
// and sever the autograd graph when called from user code (a frequent
// pattern in MoE / transformer projections that split QKV after a linear).
// By implementing chunk/split as repeated calls into `autograd::slice`,
// each output Variable gets its own SliceBackward, scatters its incoming
// gradient back to the right window of the input, and naturally supports
// `create_graph` (SliceBackward already does). The JVP path lifts for
// free because SliceBackward has a registered JVP rule — no separate
// chunk/split JVP rule is needed (the jvp_rules.cpp stubs at OpId::Chunk /
// OpId::Split remain registered for any caller that hits the raw OpId
// dispatch directly, but the Variable-level entry now decomposes upstream).
auto chunk(const Variable& input, int64_t chunks, int64_t dim) -> std::vector<Variable> {
    if (chunks <= 0) {
        throw std::invalid_argument("autograd::chunk: number of chunks must be positive");
    }
    const int64_t ndim = static_cast<int64_t>(input.shape().size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("autograd::chunk: dim out of range");
    }
    const int64_t dim_size = input.shape()[dim];

    std::vector<Variable> result;
    result.reserve(static_cast<size_t>(chunks));

    // audit-5 Y.5: when dim_size == 0 we still owe the caller `chunks`
    // zero-sized slices (PyTorch parity).  The general loop below exits at
    // the first iteration (start=0 >= dim_size=0) and returns an empty
    // vector — callers doing `auto [a, b, c] = chunk(x, 3)` would then
    // perform an out-of-range vector access on a zero-length dim.
    if (dim_size == 0) {
        for (int64_t i = 0; i < chunks; ++i) {
            result.push_back(slice(input, dim, /*start=*/0, /*end=*/0, /*step=*/1));
        }
        return result;
    }

    // Mirror tenzor::chunk's ceiling-division partition semantics so the
    // Variable form is shape-identical to the raw form.
    const int64_t chunk_size = (dim_size + chunks - 1) / chunks;

    for (int64_t i = 0; i < chunks; ++i) {
        int64_t start = i * chunk_size;
        if (start >= dim_size) {
            break;  // Fewer than `chunks` outputs when dim_size < chunks.
        }
        int64_t end = std::min(start + chunk_size, dim_size);
        result.push_back(slice(input, dim, start, end, /*step=*/1));
    }
    return result;
}

auto split(const Variable& input, int64_t split_size, int64_t dim) -> std::vector<Variable> {
    if (split_size <= 0) {
        throw std::invalid_argument("autograd::split: split_size must be positive");
    }
    const int64_t ndim = static_cast<int64_t>(input.shape().size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("autograd::split: dim out of range");
    }
    const int64_t dim_size = input.shape()[dim];

    std::vector<Variable> result;
    // audit-5 Z.27: PyTorch parity — `torch.split(zeros(0, 5), 2, dim=0)`
    // returns a 1-tuple with a single empty `[0, 5]` output, not an empty
    // tuple.  The general loop below exits immediately when dim_size == 0
    // (start=0 is not < 0), which would silently yield an empty vector and
    // surprise structured-binding callers.
    if (dim_size == 0) {
        result.push_back(slice(input, dim, /*start=*/0, /*end=*/0, /*step=*/1));
        return result;
    }
    for (int64_t start = 0; start < dim_size; start += split_size) {
        int64_t end = std::min(start + split_size, dim_size);
        result.push_back(slice(input, dim, start, end, /*step=*/1));
    }
    return result;
}

// Audit-7 EE.3: Variable-level split_with_sizes — heterogeneous chunk widths.
// Same decomposition strategy as chunk/split: each output is `autograd::slice`
// so its own SliceBackward routes the gradient back to the correct window.
auto split_with_sizes(const Variable& input,
                      const std::vector<int64_t>& split_sizes,
                      int64_t dim) -> std::vector<Variable> {
    const int64_t ndim = static_cast<int64_t>(input.shape().size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("autograd::split_with_sizes: dim out of range");
    }
    const int64_t dim_size = input.shape()[dim];

    int64_t total = 0;
    for (int64_t s : split_sizes) {
        if (s < 0) {
            throw std::invalid_argument(
                "autograd::split_with_sizes: split sizes must be non-negative");
        }
        total += s;
    }
    if (total != dim_size) {
        throw std::invalid_argument(
            "autograd::split_with_sizes: sum of split_sizes (" +
            std::to_string(total) + ") does not match dim size (" +
            std::to_string(dim_size) + ")");
    }

    std::vector<Variable> result;
    result.reserve(split_sizes.size());
    int64_t start = 0;
    for (int64_t s : split_sizes) {
        result.push_back(slice(input, dim, start, start + s, /*step=*/1));
        start += s;
    }
    return result;
}

// Audit-7 EE.3: Variable-level clone — fresh storage with identity grad.
// Implemented as `input * 1.0` so existing MulBackward already does the right
// thing (∂(x*1)/∂x = 1) without adding a new Function class. The output
// tensor is freshly allocated by the multiplication kernel, satisfying the
// narrow_copy "detached storage" contract.
auto clone(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(input.tensor().clone(), false);
    }
    return input * 1.0;
}

// Audit-7 EE.3: Variable-level unbind — split along `dim` then squeeze.
// Pattern mirrors raw `tenzor::unbind` (slice(dim, i, i+1) then squeeze(dim))
// but both stages are autograd-aware so the resulting Variables carry
// SliceBackward + SqueezeBackward chains.
auto unbind(const Variable& input, int64_t dim) -> std::vector<Variable> {
    const int64_t ndim = static_cast<int64_t>(input.shape().size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("autograd::unbind: dim out of range");
    }
    const int64_t n = input.shape()[dim];

    std::vector<Variable> result;
    result.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        auto s = slice(input, dim, i, i + 1, /*step=*/1);
        result.push_back(squeeze(s, dim));
    }
    return result;
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
    // GG.1: also save the input Variable so backward_with_variables can
    // walk the upstream graph for higher-order autograd. Without this, the
    // Backward's `backward_with_variables` rewraps `saved_tensors_[0]` as
    // a fresh `Variable(t, false)` and severs the chain.
    if (is_creating_graph()) {
        grad_fn->save_variables_for_backward({input});
    }
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    auto result = tensor_op(input.tensor());
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// Variant that saves output instead of input (for sigmoid, tanh, sqrt, etc.)
// GG.1: ALSO saves the input Variable when create_graph is on, so that
// backward_with_variables can recompute the output from the saved input
// Variable (preserving the graph chain). The saved output Tensor remains
// the cheap path used by Tensor-only backward(). We can't directly save
// the output as a graph-connected Variable because the output's grad_fn
// is the very Function we're constructing — that would be a cycle.
template<typename BackwardT, typename TensorOp>
auto unary_autograd_save_output(const Variable& input, TensorOp&& tensor_op) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tensor_op(input.tensor()), false);
    }
    auto result = tensor_op(input.tensor());
    auto grad_fn = std::make_shared<BackwardT>();
    grad_fn->save_for_backward({result});
    if (is_creating_graph()) {
        grad_fn->save_variables_for_backward({input});
    }
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// Variant that saves input + a scalar parameter as a second tensor.
// `Scalar` is a template parameter (typically float for activations whose
// public API still takes float — elu/leaky_relu/softplus — and double for
// the Float64-precision-preserving pow path; see audit-3 R.1).
template<typename BackwardT, typename TensorOp, typename Scalar>
auto unary_autograd_with_param(const Variable& input, Scalar param, TensorOp&& tensor_op) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tensor_op(input.tensor()), false);
    }
    auto grad_fn = std::make_shared<BackwardT>();
    auto param_tensor = full({1}, static_cast<double>(param), input.tensor().dtype(), input.tensor().device());
    grad_fn->save_for_backward({input.tensor(), param_tensor});
    // GG.1: save only the input as a Variable; the scalar parameter is a
    // constant (does not participate in the adjoint's autograd graph).
    if (is_creating_graph()) {
        grad_fn->save_variables_for_backward({input});
    }
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

auto pow(const Variable& input, double exponent) -> Variable {
    // Audit-6 AA.5: PyTorch's pow contract requires `0^0 == 1` and
    // `0^b == 0` for `b > 0`. The underlying tensor `pow(t, double)` does
    // honour C99 pow semantics on most backends, but exponent == 0 also
    // needs ones_like even when the input contains NaN/Inf entries (per
    // IEEE 754 and PyTorch). Short-circuit those two scalar cases here so
    // the result is unambiguous across every backend.
    if (exponent == 0.0) {
        // 0-d exponent: result is identically 1 for every input element.
        auto ones = tenzor::ones_like(input.tensor());
        return Variable(ones, false);
    }
    return unary_autograd_with_param<PowBackward>(input, exponent,
        [exponent](const Tensor& t) { return tenzor::pow(t, exponent); });
}

// Audit-5 Z.20 / audit-6 AA.5: Variable ** Variable and scalar ** Variable.
// The naive `exp(b * log(a))` form is NaN at `a == 0` because `log(0) = -inf`
// and `0 * -inf = NaN`. PyTorch's contract is:
//   - `0 ^ b == 1`  if b == 0
//   - `0 ^ b == 0`  if b > 0
//   - `0 ^ b == inf` if b < 0  (mathematically a singularity; we fold this
//                                into the exp(b*log(0)) = exp(-inf*b<0) = inf
//                                produced by the standard branch, which is
//                                what PyTorch does)
//   - `(-a) ^ b` for a > 0 and b non-integer is NaN (unchanged).
//
// We branch on `base == 0` and route the zero-base case through a
// `where(b == 0, 1, 0)` to avoid the `0 * -inf = NaN` hazard. We use a
// `where(base == 0, 1, base)` "safe base" inside the `exp(b*log(.))` branch
// so the log is never applied to literal zero (its result is discarded by
// the outer where, but it must not be NaN in case of downstream chain-rule).
auto pow(const Variable& base, const Variable& exponent) -> Variable {
    // Build a "safe base" tensor that replaces zeros with ones so `log(.)`
    // produces finite values; the outer `where` discards those entries.
    auto zero_t = tenzor::zeros_like(base.tensor());
    auto one_t = tenzor::ones_like(base.tensor());
    Variable zero_var(zero_t, false);
    Variable one_var(one_t, false);

    auto base_is_zero = ::tenzor::eq(base, zero_var);

    // Zero-base branch: where(exp == 0, 1, 0). Build via Variable ops so
    // the gradient routing is consistent (this branch has no input-grad
    // contribution because both 1 and 0 are constants here, which matches
    // PyTorch — see Note in torch/csrc/autograd/FunctionsManual.cpp).
    auto exp_is_zero = ::tenzor::eq(exponent, zero_var);
    auto zero_base_result = ::tenzor::where(exp_is_zero, one_var, zero_var);

    // audit-7 DD.4: negative-base handling. PyTorch defines
    // (-2)^3 == -8 for integer exponents but returns NaN for fractional
    // exponents. The previous implementation reduced safe_base only on the
    // base==0 path, leaving negative bases unchanged — `log(negative)`
    // then poisoned the standard branch with NaN even for cases like
    // (-2)^3 that have a well-defined real value.
    //
    // Strategy: route negative bases through `|base|^e` and multiply by
    // the sign factor (-1)^e. The sign factor is computed as
    // `cos(pi * round(e))` which is exactly ±1 whenever e is integer-valued.
    // For non-integer exponents we emit a NaN result (matching PyTorch /
    // IEEE) and warn once that the gradient through such a point is
    // ill-defined.
    auto base_is_neg = ::tenzor::lt(base, zero_var);
    auto exp_is_int = ::tenzor::eq(::tenzor::round(exponent), exponent);

    // safe_base used by the standard branch: replace base==0 with 1 (so
    // log is finite) and replace negative bases with |base| (so log is
    // finite for the integer-exp branch). The outer `where` cascade
    // selects the right result per element.
    auto abs_base = ::tenzor::abs(base);
    auto safe_base = ::tenzor::where(base_is_zero, one_var, abs_base);
    auto pos_branch = ::tenzor::exp(exponent * ::tenzor::log(safe_base));

    // Sign factor for negative-base + integer-exponent case.
    // cos(pi * k) == (-1)^k for any integer k; rounding the exponent
    // first guarantees we never feed a non-integer multiple of pi to cos,
    // so the result is always exactly ±1 on the integer-exp path.
    const double pi = 3.14159265358979323846;
    auto rounded_exp = ::tenzor::round(exponent);
    auto sign_factor = ::tenzor::cos(rounded_exp * pi);
    auto neg_base_int_result = sign_factor * pos_branch;

    // Non-integer exponent on a negative base: undefined real gradient.
    // Materialise NaN explicitly so the failure is loud and matches the
    // IEEE convention.
    auto nan_t = tenzor::full(
        std::vector<int64_t>(base.shape().begin(), base.shape().end()),
        std::numeric_limits<double>::quiet_NaN(),
        base.tensor().dtype(), base.tensor().device());
    Variable nan_var(nan_t, false);
    TENZOR_WARN_ONCE("pow: negative base with non-integer exponent has no real "
                     "gradient; emitting NaN for those elements");

    auto neg_base_result = ::tenzor::where(exp_is_int, neg_base_int_result, nan_var);

    auto standard_result = ::tenzor::where(base_is_neg, neg_base_result, pos_branch);

    return ::tenzor::where(base_is_zero, zero_base_result, standard_result);
}

auto pow(double base, const Variable& exponent) -> Variable {
    // Scalar base case. Folds into the constant-base path:
    //   - base > 0: log(base) is a constant; result is exp(b * log(base)).
    //   - base == 0: result is `where(b == 0, 1, 0)`. (b < 0 gives `inf`
    //                in the standard branch, which is the IEEE result we
    //                want.) We don't emit `log(0) = -inf` into the graph.
    //   - base < 0: NaN unless b is an integer; PyTorch returns NaN here
    //                in the autograd path, which is what `log(base) = nan`
    //                yields naturally.
    if (base == 0.0) {
        // FF.2: Route the zero-base branch through Variable-level where so the
        // graph carries a dependency on exp_var. The previous early-return
        // built a fresh Variable with no grad_fn → JVP and reverse-mode grad
        // disagreed (JVP saw the constant, grad saw no input dependency at
        // all). The data-branch tensors are ones_like / zeros_like of exp_var
        // and remain constants (PyTorch convention is zero gradient through
        // both branches), but `exp_is_zero = (exp_var == 0)` is now an
        // exp_var-derived Variable so the chain rule has a path to walk.
        auto ones_t = tenzor::ones_like(exponent.tensor());
        auto zeros_t = tenzor::zeros_like(exponent.tensor());
        Variable ones_v(ones_t, exponent.requires_grad());
        Variable zeros_v(zeros_t, exponent.requires_grad());
        Variable zero_scalar(tenzor::zeros_like(exponent.tensor()), false);
        auto exp_is_zero = ::tenzor::eq(exponent, zero_scalar);
        return ::tenzor::where(exp_is_zero, ones_v, zeros_v);
    }
    double log_base = std::log(base);
    return ::tenzor::exp(exponent * log_base);
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
    // GG.1: keep input Variables for higher-order autograd through atan2.
    if (is_creating_graph()) {
        grad_fn->save_variables_for_backward({y, x});
    }
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

auto std(const Variable& input, std::optional<int64_t> dim, bool keepdim, bool unbiased) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::std(input.tensor(), dim, keepdim, unbiased), false);
    }
    auto result = tenzor::std(input.tensor(), dim, keepdim, unbiased);
    // R.3: thread unbiased through to the backward so the denominator
    // matches the forward (N-1 when unbiased=true; N when false). The
    // flag is also surfaced via saved_attributes() (R.8).
    auto grad_fn = std::make_shared<StdBackward>(dim, keepdim, unbiased);
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

auto var(const Variable& input, std::optional<int64_t> dim, bool keepdim, bool unbiased) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::var(input.tensor(), dim, keepdim, unbiased), false);
    }
    auto result = tenzor::var(input.tensor(), dim, keepdim, unbiased);
    // R.3: thread unbiased through to the backward so the denominator
    // matches the forward (N-1 when unbiased=true; N when false). The
    // flag is also surfaced via saved_attributes() (R.8).
    auto grad_fn = std::make_shared<VarBackward>(dim, keepdim, unbiased);
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
    // GG.1: keep input Variable; backward_with_variables recomputes lse on
    // the live graph (saved output Tensor cannot carry grad_fn back through
    // this Function without forming a cycle).
    if (is_creating_graph()) {
        grad_fn->save_variables_for_backward({input});
    }
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
    // audit-7 DD.3: backend `where` kernels (CUDA / ROCm / Vulkan / OneAPI)
    // require contiguous bool input. CC.4 only forced contiguity on the
    // SAVED condition for backward, but the forward dispatch still ran on
    // the raw (possibly transposed/permuted) tensor and crashed. Materialise
    // once at the top so both forward and backward see the same clean
    // contiguous buffer.
    auto cond_c = condition.tensor().contiguous();
    bool needs_grad = (x.requires_grad() || y.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(tenzor::where(cond_c, x.tensor(), y.tensor()), false);
    }
    auto result = tenzor::where(cond_c, x.tensor(), y.tensor());
    auto grad_fn = std::make_shared<WhereBackward>();
    // Condition is non-differentiable; only x and y carry gradients.
    // WhereBackward::backward returns {grad_x, grad_y} in that order, so
    // input_variables / next_functions must match — previously this
    // passed `{condition, x, y}` which misaligned the engine's index-
    // based gradient accumulation and pushed grad_y into x (and dropped
    // y's gradient entirely). The condition tensor is still kept in
    // saved_tensors_ so the backward can read it.
    //
    // CC.4: save condition.contiguous(). The backward dispatches
    // `tenzor::where` on the saved condition; backend `where` kernels (CUDA
    // / ROCm / Vulkan / OneAPI) require contiguous bool input and throw
    // when they encounter a non-contiguous saved view (e.g. condition was
    // produced by a `transpose`/`permute` and never materialised). Forcing
    // contiguity at save-time gives the backward a clean buffer with no
    // hidden stride dependency.
    grad_fn->save_for_backward({cond_c});
    // audit-5 Y.4: save the un-broadcasted x/y shapes so the backward can
    // reduce the masked grad back to each input's original shape (mirroring
    // BinaryOp backwards). `tenzor::where` materialises the broadcast shape,
    // so without this the engine would receive grad_x at the broadcast shape
    // — accumulation fails or silently scatters.
    grad_fn->input_shape_x_ = std::vector<int64_t>(x.shape().begin(), x.shape().end());
    grad_fn->input_shape_y_ = std::vector<int64_t>(y.shape().begin(), y.shape().end());
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
    // GG.1: keep input Variable; backward_with_variables recomputes cumprod
    // on the live graph.
    if (is_creating_graph()) {
        grad_fn->save_variables_for_backward({input});
    }
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
    // R.7: Save the frequency-bin count of the input *before* the irfft so
    // the backward can pass it back to rfft. Without this, the backward
    // would let rfft infer n from the time-domain output length, which is
    // wrong for any padded/truncated irfft (n != 2*(input_bins-1)).
    int64_t actual_dim = dim < 0 ? dim + input.tensor().ndim() : dim;
    int64_t n_orig = input.tensor().shape()[actual_dim];
    auto result = tenzor::fft::irfft(input.tensor(), n, dim, norm);
    auto grad_fn = std::make_shared<IRFFTBackward>(dim, norm, n_orig);
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
    // Audit-7 EE.4: capture input dtype so the adjoint can return the
    // time-domain gradient in the original (possibly non-F32) dtype family.
    auto grad_fn = std::make_shared<STFTBackward>(
        n_fft, hop_length, win_length, window, center, normalized, onesided,
        signal_length, input.tensor().dtype());
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
    // Audit-7 EE.4: capture complex input dtype so the adjoint stays in the
    // forward family (Complex64 vs Complex128).
    auto grad_fn = std::make_shared<ISTFTBackward>(
        n_fft, hop_length, win_length, window, center, normalized, onesided,
        input.tensor().dtype());
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

// Variable-level conj routed through ConjBackward so the Wirtinger paths in
// MulBackward / DivBackward / MatMulBackward (audit-5 X.5) preserve the
// grad_fn carried on saved_variables_. ConjBackward::backward is `conj(grad)`
// — linear and stateless, so the wrapper mirrors view_as_real.
auto conj(const Variable& input) -> Variable {
    auto out_t = ::tenzor::conj(input.tensor());
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(out_t, false);
    }
    auto grad_fn = std::make_shared<ConjBackward>();
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

// ===========================================================================
// Audit E.7 continuation: wrappers for OpIds whose Function classes are
// declared in function.hpp and implemented in function_new_ops.cpp.
//
// Differentiable wrappers use either unary_autograd<>{} (save input) or
// unary_autograd_save_output<>{} (save output). Non-differentiable wrappers
// still attach a Function so that calling .backward() through them raises
// tenzor::NonDifferentiable instead of silently dropping the graph edge.
// ===========================================================================

// square(x) = x * x  — save input
auto square(const Variable& input) -> Variable {
    return unary_autograd<SquareBackward>(input,
        [](const Tensor& t) { return tenzor::square(t); });
}

// rsqrt(x) = 1/sqrt(x)  — save output
auto rsqrt(const Variable& input) -> Variable {
    return unary_autograd_save_output<RsqrtBackward>(input,
        [](const Tensor& t) { return tenzor::rsqrt(t); });
}

// deg2rad — no saved tensor, slope is a constant; reuse unary_autograd which
// saves the input but the backward simply ignores it.
auto deg2rad(const Variable& input) -> Variable {
    return unary_autograd<Deg2RadBackward>(input,
        [](const Tensor& t) { return tenzor::deg2rad(t); });
}

// rad2deg — same shape as deg2rad
auto rad2deg(const Variable& input) -> Variable {
    return unary_autograd<Rad2DegBackward>(input,
        [](const Tensor& t) { return tenzor::rad2deg(t); });
}

// logit(x) = log(x/(1-x)) — save input
auto logit(const Variable& input, double eps) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::logit(input.tensor(), eps), false);
    }
    auto grad_fn = std::make_shared<LogitBackward>();
    grad_fn->save_for_backward({input.tensor()});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    auto result = tenzor::logit(input.tensor(), eps);
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// nan_to_num — save input (we need isfinite(x) for the mask in backward)
auto nan_to_num(const Variable& input,
                double nan, double posinf, double neginf) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::nan_to_num(input.tensor(), nan, posinf, neginf),
                        false);
    }
    auto grad_fn = std::make_shared<NanToNumBackward>();
    grad_fn->save_for_backward({input.tensor()});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    auto result = tenzor::nan_to_num(input.tensor(), nan, posinf, neginf);
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// --- non-differentiable wrappers ----------------------------------------
//
// We still attach a Function with set_next_functions/set_input_variables so
// the graph topology is preserved. Calling backward() through any of these
// raises tenzor::NonDifferentiable with the op's documented reason.

auto heaviside(const Variable& input, const Variable& values) -> Variable {
    auto result = tenzor::heaviside(input.tensor(), values.tensor());
    if ((!input.requires_grad() && !values.requires_grad()) ||
        !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<HeavisideBackward>();
    std::vector<std::shared_ptr<Function>> next_funcs;
    if (auto fn = input.grad_fn()) next_funcs.push_back(fn);
    if (auto fn = values.grad_fn()) next_funcs.push_back(fn);
    grad_fn->set_next_functions(std::move(next_funcs));
    grad_fn->set_input_variables({input, values});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto signbit(const Variable& input) -> Variable {
    auto result = tenzor::signbit(input.tensor());
    // Output is Bool — gradient flow through it is meaningless, but we still
    // wire a backward that throws NonDifferentiable if anyone tries to .backward().
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<SignbitBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto frexp(const Variable& input) -> std::pair<Variable, Variable> {
    auto [mantissa_t, exponent_t] = tenzor::frexp(input.tensor());
    if (!input.requires_grad() || !is_grad_enabled()) {
        return {Variable(mantissa_t, false), Variable(exponent_t, false)};
    }
    auto grad_fn = std::make_shared<FrexpBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    // Only mantissa carries the (non-differentiable) grad_fn — exponent is
    // integer-typed by construction and never participates in a grad graph.
    Variable mantissa(mantissa_t, true);
    mantissa.set_grad_fn(grad_fn);
    Variable exponent(exponent_t, false);
    return {mantissa, exponent};
}

auto histogram(const Variable& input, int64_t bins, double min, double max)
    -> std::pair<Variable, Variable> {
    auto [counts_t, edges_t] = tenzor::histogram(input.tensor(), bins, min, max);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return {Variable(counts_t, false), Variable(edges_t, false)};
    }
    auto grad_fn = std::make_shared<HistogramBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable counts(counts_t, true);
    counts.set_grad_fn(grad_fn);
    Variable edges(edges_t, false);
    return {counts, edges};
}

// ===========================================================================
// Audit E.7 continuation (batch 2): autograd wrappers for the second set of
// 10 OpIds. Differentiable wrappers save the inputs they need for backward;
// non-differentiable wrappers still attach a Function with the correct
// next_functions / input_variables wiring so calling backward() through them
// raises tenzor::NonDifferentiable instead of silently dropping the edge.
// ===========================================================================

// sign(x) — gradient is zero almost everywhere. Save the input so backward
// can produce a zero buffer of the right shape/dtype; reuse the unary
// helper which already handles the no-grad fast path.
auto sign(const Variable& input) -> Variable {
    return unary_autograd<SignBackward>(input,
        [](const Tensor& t) { return tenzor::sign(t); });
}

// hypot(x, y) — binary; saves both inputs and the broadcast input shapes.
auto hypot(const Variable& x, const Variable& y) -> Variable {
    bool needs_grad = (x.requires_grad() || y.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(tenzor::hypot(x.tensor(), y.tensor()), false);
    }
    auto grad_fn = std::make_shared<HypotBackward>();
    grad_fn->input_shape_x_ = std::vector<int64_t>(x.shape().begin(), x.shape().end());
    grad_fn->input_shape_y_ = std::vector<int64_t>(y.shape().begin(), y.shape().end());
    grad_fn->save_for_backward({x.tensor(), y.tensor()});
    grad_fn->set_next_functions({x.grad_fn(), y.grad_fn()});
    grad_fn->set_input_variables({x, y});
    auto result = tenzor::hypot(x.tensor(), y.tensor());
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// copysign(magnitude, sign_src) — saves only sign_src (we need its sign);
// the magnitude's gradient is sign(sign_src) which is purely derived from
// the saved tensor.
auto copysign(const Variable& magnitude, const Variable& sign_src) -> Variable {
    bool needs_grad = (magnitude.requires_grad() || sign_src.requires_grad())
                      && is_grad_enabled();
    if (!needs_grad) {
        return Variable(tenzor::copysign(magnitude.tensor(), sign_src.tensor()),
                        false);
    }
    auto grad_fn = std::make_shared<CopysignBackward>();
    grad_fn->input_shape_mag_ =
        std::vector<int64_t>(magnitude.shape().begin(), magnitude.shape().end());
    grad_fn->input_shape_sign_ =
        std::vector<int64_t>(sign_src.shape().begin(), sign_src.shape().end());
    grad_fn->save_for_backward({sign_src.tensor()});
    grad_fn->set_next_functions({magnitude.grad_fn(), sign_src.grad_fn()});
    grad_fn->set_input_variables({magnitude, sign_src});
    auto result = tenzor::copysign(magnitude.tensor(), sign_src.tensor());
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// xlog1py(x, y) — binary; saves both inputs and shapes for broadcast.
auto xlog1py(const Variable& x, const Variable& y) -> Variable {
    bool needs_grad = (x.requires_grad() || y.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(tenzor::xlog1py(x.tensor(), y.tensor()), false);
    }
    auto grad_fn = std::make_shared<Xlog1pyBackward>();
    grad_fn->input_shape_x_ = std::vector<int64_t>(x.shape().begin(), x.shape().end());
    grad_fn->input_shape_y_ = std::vector<int64_t>(y.shape().begin(), y.shape().end());
    grad_fn->save_for_backward({x.tensor(), y.tensor()});
    grad_fn->set_next_functions({x.grad_fn(), y.grad_fn()});
    grad_fn->set_input_variables({x, y});
    auto result = tenzor::xlog1py(x.tensor(), y.tensor());
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// addcmul(a, b, c, value) — ternary; saves b, c and the scalar value.
auto addcmul(const Variable& a, const Variable& b, const Variable& c,
             double value) -> Variable {
    bool needs_grad = (a.requires_grad() || b.requires_grad() || c.requires_grad())
                      && is_grad_enabled();
    if (!needs_grad) {
        return Variable(
            tenzor::addcmul(a.tensor(), b.tensor(), c.tensor(), value),
            false);
    }
    auto grad_fn = std::make_shared<AddcmulBackward>();
    grad_fn->value_ = value;
    grad_fn->input_shape_a_ = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    grad_fn->input_shape_b_ = std::vector<int64_t>(b.shape().begin(), b.shape().end());
    grad_fn->input_shape_c_ = std::vector<int64_t>(c.shape().begin(), c.shape().end());
    grad_fn->save_for_backward({b.tensor(), c.tensor()});
    grad_fn->set_next_functions({a.grad_fn(), b.grad_fn(), c.grad_fn()});
    grad_fn->set_input_variables({a, b, c});
    auto result = tenzor::addcmul(a.tensor(), b.tensor(), c.tensor(), value);
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// addcdiv(a, b, c, value) — ternary; saves b, c and the scalar value.
auto addcdiv(const Variable& a, const Variable& b, const Variable& c,
             double value) -> Variable {
    bool needs_grad = (a.requires_grad() || b.requires_grad() || c.requires_grad())
                      && is_grad_enabled();
    if (!needs_grad) {
        return Variable(
            tenzor::addcdiv(a.tensor(), b.tensor(), c.tensor(), value),
            false);
    }
    auto grad_fn = std::make_shared<AddcdivBackward>();
    grad_fn->value_ = value;
    grad_fn->input_shape_a_ = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    grad_fn->input_shape_b_ = std::vector<int64_t>(b.shape().begin(), b.shape().end());
    grad_fn->input_shape_c_ = std::vector<int64_t>(c.shape().begin(), c.shape().end());
    grad_fn->save_for_backward({b.tensor(), c.tensor()});
    grad_fn->set_next_functions({a.grad_fn(), b.grad_fn(), c.grad_fn()});
    grad_fn->set_input_variables({a, b, c});
    auto result = tenzor::addcdiv(a.tensor(), b.tensor(), c.tensor(), value);
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// --- non-differentiable wrappers ----------------------------------------

auto floor(const Variable& input) -> Variable {
    auto result = tenzor::floor(input.tensor());
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<FloorBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto ceil(const Variable& input) -> Variable {
    auto result = tenzor::ceil(input.tensor());
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<CeilBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto isnan(const Variable& input) -> Variable {
    auto result = tenzor::isnan(input.tensor());
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<IsNanBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto logical_and(const Variable& a, const Variable& b) -> Variable {
    auto result = tenzor::logical_and(a.tensor(), b.tensor());
    if ((!a.requires_grad() && !b.requires_grad()) || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<LogicalAndBackward>();
    std::vector<std::shared_ptr<Function>> next_funcs;
    if (auto fn = a.grad_fn()) next_funcs.push_back(fn);
    if (auto fn = b.grad_fn()) next_funcs.push_back(fn);
    grad_fn->set_next_functions(std::move(next_funcs));
    grad_fn->set_input_variables({a, b});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ===========================================================================
// Audit E.7 continuation (batch 3): autograd wrappers for the third set of
// 10 OpIds. Differentiable wrappers save the inputs they need for backward;
// non-differentiable wrappers still attach a Function with the correct
// next_functions / input_variables wiring so calling backward() through them
// raises tenzor::NonDifferentiable instead of silently dropping the edge.
// ===========================================================================

// addmm(input, mat1, mat2, beta, alpha): saves mat1 + mat2 plus the two
// scalar coefficients. The "input" bias slot is only used for shape /
// broadcast-reduction (its values are not needed because the local Jacobian
// w.r.t. input is just the scalar beta).
auto addmm(const Variable& input, const Variable& mat1, const Variable& mat2,
           double beta, double alpha) -> Variable {
    bool needs_grad = (input.requires_grad() || mat1.requires_grad()
                       || mat2.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(
            tenzor::addmm(input.tensor(), mat1.tensor(), mat2.tensor(),
                          beta, alpha),
            false);
    }
    auto grad_fn = std::make_shared<AddmmBackward>();
    grad_fn->beta_ = beta;
    grad_fn->alpha_ = alpha;
    grad_fn->input_shape_input_ =
        std::vector<int64_t>(input.shape().begin(), input.shape().end());
    grad_fn->input_shape_mat1_ =
        std::vector<int64_t>(mat1.shape().begin(), mat1.shape().end());
    grad_fn->input_shape_mat2_ =
        std::vector<int64_t>(mat2.shape().begin(), mat2.shape().end());
    grad_fn->save_for_backward({mat1.tensor(), mat2.tensor()});
    grad_fn->set_next_functions({input.grad_fn(), mat1.grad_fn(), mat2.grad_fn()});
    grad_fn->set_input_variables({input, mat1, mat2});
    auto result = tenzor::addmm(input.tensor(), mat1.tensor(), mat2.tensor(),
                                beta, alpha);
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// addmv(input, mat, vec, beta, alpha): matrix-vector version of addmm.
auto addmv(const Variable& input, const Variable& mat, const Variable& vec,
           double beta, double alpha) -> Variable {
    bool needs_grad = (input.requires_grad() || mat.requires_grad()
                       || vec.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(
            tenzor::addmv(input.tensor(), mat.tensor(), vec.tensor(),
                          beta, alpha),
            false);
    }
    auto grad_fn = std::make_shared<AddmvBackward>();
    grad_fn->beta_ = beta;
    grad_fn->alpha_ = alpha;
    grad_fn->input_shape_input_ =
        std::vector<int64_t>(input.shape().begin(), input.shape().end());
    grad_fn->input_shape_mat_ =
        std::vector<int64_t>(mat.shape().begin(), mat.shape().end());
    grad_fn->input_shape_vec_ =
        std::vector<int64_t>(vec.shape().begin(), vec.shape().end());
    grad_fn->save_for_backward({mat.tensor(), vec.tensor()});
    grad_fn->set_next_functions({input.grad_fn(), mat.grad_fn(), vec.grad_fn()});
    grad_fn->set_input_variables({input, mat, vec});
    auto result = tenzor::addmv(input.tensor(), mat.tensor(), vec.tensor(),
                                beta, alpha);
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// baddbmm(input, batch1, batch2, beta, alpha): batched addmm.
auto baddbmm(const Variable& input, const Variable& batch1, const Variable& batch2,
             double beta, double alpha) -> Variable {
    bool needs_grad = (input.requires_grad() || batch1.requires_grad()
                       || batch2.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(
            tenzor::baddbmm(input.tensor(), batch1.tensor(), batch2.tensor(),
                            beta, alpha),
            false);
    }
    auto grad_fn = std::make_shared<BaddbmmBackward>();
    grad_fn->beta_ = beta;
    grad_fn->alpha_ = alpha;
    grad_fn->input_shape_input_ =
        std::vector<int64_t>(input.shape().begin(), input.shape().end());
    grad_fn->input_shape_batch1_ =
        std::vector<int64_t>(batch1.shape().begin(), batch1.shape().end());
    grad_fn->input_shape_batch2_ =
        std::vector<int64_t>(batch2.shape().begin(), batch2.shape().end());
    grad_fn->save_for_backward({batch1.tensor(), batch2.tensor()});
    grad_fn->set_next_functions(
        {input.grad_fn(), batch1.grad_fn(), batch2.grad_fn()});
    grad_fn->set_input_variables({input, batch1, batch2});
    auto result = tenzor::baddbmm(input.tensor(), batch1.tensor(),
                                  batch2.tensor(), beta, alpha);
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// nansum(x, dim, keepdim): saves the input so backward can recompute the
// NaN mask (the synthetic-zero positions contribute no gradient).
auto nansum(const Variable& input, std::optional<int64_t> dim, bool keepdim)
    -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::nansum(input.tensor(), dim, keepdim), false);
    }
    auto grad_fn = std::make_shared<NansumBackward>(dim, keepdim);
    grad_fn->save_for_backward({input.tensor()});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    auto result = tenzor::nansum(input.tensor(), dim, keepdim);
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// tile(input, reps): saves the original shape and the reps for backward.
auto tile(const Variable& input, std::vector<int64_t> reps) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::tile(input.tensor(), reps), false);
    }
    auto grad_fn = std::make_shared<TileBackward>();
    grad_fn->original_shape_ =
        std::vector<int64_t>(input.shape().begin(), input.shape().end());
    grad_fn->reps_ = reps;
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    auto result = tenzor::tile(input.tensor(), reps);
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// --- non-differentiable wrappers ----------------------------------------

auto count_nonzero(const Variable& input, std::optional<int64_t> dim) -> Variable {
    auto result = tenzor::count_nonzero(input.tensor(), dim);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<CountNonzeroBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto isinf(const Variable& input) -> Variable {
    auto result = tenzor::isinf(input.tensor());
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<IsInfBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto bitwise_and(const Variable& a, const Variable& b) -> Variable {
    auto result = tenzor::bitwise_and(a.tensor(), b.tensor());
    if ((!a.requires_grad() && !b.requires_grad()) || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<BitwiseAndBackward>();
    std::vector<std::shared_ptr<Function>> next_funcs;
    if (auto fn = a.grad_fn()) next_funcs.push_back(fn);
    if (auto fn = b.grad_fn()) next_funcs.push_back(fn);
    grad_fn->set_next_functions(std::move(next_funcs));
    grad_fn->set_input_variables({a, b});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto round(const Variable& input) -> Variable {
    auto result = tenzor::round(input.tensor());
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<RoundBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto eq(const Variable& a, const Variable& b) -> Variable {
    auto result = tenzor::eq(a.tensor(), b.tensor());
    if ((!a.requires_grad() && !b.requires_grad()) || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<EqBackward>();
    std::vector<std::shared_ptr<Function>> next_funcs;
    if (auto fn = a.grad_fn()) next_funcs.push_back(fn);
    if (auto fn = b.grad_fn()) next_funcs.push_back(fn);
    grad_fn->set_next_functions(std::move(next_funcs));
    grad_fn->set_input_variables({a, b});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ===========================================================================
// Audit E.7 continuation (batch 4): autograd wrappers for another 10 OpIds.
// Non-differentiable wrappers still attach a Function with full
// next_functions / input_variables wiring so calling backward() through them
// raises tenzor::NonDifferentiable instead of silently dropping the edge.
// ===========================================================================

// Local helper: build a non-diff binary wrapper. Cuts copy-paste across the
// five comparisons + two bitwise binaries. FactoryFn returns a fresh Function
// instance for the OpId (any of {Ne,Lt,Le,Gt,Ge,BitwiseOr,BitwiseXor}Backward).
namespace {
template <typename BackwardCls, typename EagerFn>
auto make_nondiff_binary(const Variable& a, const Variable& b, EagerFn&& eager)
    -> Variable {
    auto result = eager(a.tensor(), b.tensor());
    if ((!a.requires_grad() && !b.requires_grad()) || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<BackwardCls>();
    std::vector<std::shared_ptr<Function>> next_funcs;
    if (auto fn = a.grad_fn()) next_funcs.push_back(fn);
    if (auto fn = b.grad_fn()) next_funcs.push_back(fn);
    grad_fn->set_next_functions(std::move(next_funcs));
    grad_fn->set_input_variables({a, b});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

template <typename BackwardCls, typename EagerFn>
auto make_nondiff_unary(const Variable& input, EagerFn&& eager) -> Variable {
    auto result = eager(input.tensor());
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<BackwardCls>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}
} // namespace

auto ne(const Variable& a, const Variable& b) -> Variable {
    return make_nondiff_binary<NeBackward>(
        a, b, [](const Tensor& x, const Tensor& y) { return tenzor::ne(x, y); });
}

auto lt(const Variable& a, const Variable& b) -> Variable {
    return make_nondiff_binary<LtBackward>(
        a, b, [](const Tensor& x, const Tensor& y) { return tenzor::lt(x, y); });
}

auto le(const Variable& a, const Variable& b) -> Variable {
    return make_nondiff_binary<LeBackward>(
        a, b, [](const Tensor& x, const Tensor& y) { return tenzor::le(x, y); });
}

auto gt(const Variable& a, const Variable& b) -> Variable {
    return make_nondiff_binary<GtBackward>(
        a, b, [](const Tensor& x, const Tensor& y) { return tenzor::gt(x, y); });
}

auto ge(const Variable& a, const Variable& b) -> Variable {
    return make_nondiff_binary<GeBackward>(
        a, b, [](const Tensor& x, const Tensor& y) { return tenzor::ge(x, y); });
}

auto bitwise_or(const Variable& a, const Variable& b) -> Variable {
    return make_nondiff_binary<BitwiseOrBackward>(
        a, b,
        [](const Tensor& x, const Tensor& y) { return tenzor::bitwise_or(x, y); });
}

auto bitwise_xor(const Variable& a, const Variable& b) -> Variable {
    return make_nondiff_binary<BitwiseXorBackward>(
        a, b,
        [](const Tensor& x, const Tensor& y) { return tenzor::bitwise_xor(x, y); });
}

auto bitwise_not(const Variable& input) -> Variable {
    return make_nondiff_unary<BitwiseNotBackward>(
        input, [](const Tensor& x) { return tenzor::bitwise_not(x); });
}

auto isfinite(const Variable& input) -> Variable {
    return make_nondiff_unary<IsFiniteBackward>(
        input, [](const Tensor& x) { return tenzor::isfinite(x); });
}

// logcumsumexp(x, dim): saves x and y because the backward needs both
//   grad_x = exp(x) * flip(cumsum(flip(grad_y * exp(-y), dim), dim), dim).
auto logcumsumexp(const Variable& input, int64_t dim) -> Variable {
    auto result = tenzor::logcumsumexp(input.tensor(), dim);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<LogcumsumexpBackward>(dim);
    grad_fn->save_for_backward({input.tensor(), result});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ============================================================================
// Audit E.7 continuation (batch 5): 10 more wrappers.
// ============================================================================

// --- non-differentiable ------------------------------------------------

auto isposinf(const Variable& input) -> Variable {
    return make_nondiff_unary<IsPosInfBackward>(
        input, [](const Tensor& x) { return tenzor::isposinf(x); });
}

auto isneginf(const Variable& input) -> Variable {
    return make_nondiff_unary<IsNegInfBackward>(
        input, [](const Tensor& x) { return tenzor::isneginf(x); });
}

auto trunc(const Variable& input) -> Variable {
    return make_nondiff_unary<TruncBackward>(
        input, [](const Tensor& x) { return tenzor::trunc(x); });
}

namespace {
template <typename BackwardCls, typename EagerFn>
auto make_nondiff_reduction(const Variable& input, EagerFn&& eager) -> Variable {
    auto result = eager(input.tensor());
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<BackwardCls>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}
} // namespace

auto any(const Variable& input, std::optional<int64_t> dim, bool keepdim) -> Variable {
    return make_nondiff_reduction<AnyBackward>(
        input, [&](const Tensor& x) { return tenzor::any(x, dim, keepdim); });
}

auto all(const Variable& input, std::optional<int64_t> dim, bool keepdim) -> Variable {
    return make_nondiff_reduction<AllBackward>(
        input, [&](const Tensor& x) { return tenzor::all(x, dim, keepdim); });
}

auto has_inf_nan(const Variable& input) -> Variable {
    return make_nondiff_unary<HasInfNanBackward>(
        input, [](const Tensor& x) { return tenzor::has_inf_nan(x); });
}

// --- differentiable ----------------------------------------------------

// nanmean(x, dim, keepdim): saves the input so backward can recompute the
// non-NaN count and the NaN mask.
auto nanmean(const Variable& input, std::optional<int64_t> dim, bool keepdim)
    -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(tenzor::nanmean(input.tensor(), dim, keepdim), false);
    }
    auto grad_fn = std::make_shared<NanmeanBackward>(dim, keepdim);
    grad_fn->save_for_backward({input.tensor()});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    auto result = tenzor::nanmean(input.tensor(), dim, keepdim);
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// masked_fill(x, mask, value): mask is a plain Tensor (not a Variable), value
// is a constant scalar. Saves the mask so backward can zero grad at masked
// positions.
auto masked_fill(const Variable& input, const Tensor& mask, float value)
    -> Variable {
    auto result = tenzor::masked_fill(input.tensor(), mask, value);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<MaskedFillBackward>();
    grad_fn->save_for_backward({mask});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// masked_select(x, mask): saves the mask and a CPU Int64 tensor encoding the
// input shape so backward can scatter grad_y back into a zeros_like(x).
//
// CC.1: assert mask.shape() == input.shape() at the autograd wrapper. The
// scatter-back path in MaskedSelectBackward assumes the saved mask has the
// same shape as the input so a single zeros_like(input) + scatter-add over
// the mask positions reconstructs grad_x. If mask broadcasts (smaller rank
// or unequal extents), the saved mask aliases a view and the scatter would
// either fault or — worse — silently scatter into the wrong cells. Take the
// simpler assert path here; relax to broadcast+reduce-after-scatter only
// when a real use-case justifies the extra forward bookkeeping.
auto masked_select(const Variable& input, const Tensor& mask) -> Variable {
    // `shape()` returns a span, so use ranges::equal for the comparison.
    TENZOR_CHECK_SHAPE(
        std::ranges::equal(mask.shape(), input.tensor().shape()),
        "masked_select (autograd): mask shape must equal input shape. "
        "Broadcasting masks would corrupt the backward scatter-add — see "
        "MaskedSelectBackward in src/autograd/function_shape_ext.cpp. "
        "Expand the mask to input.shape() before calling masked_select if "
        "you need autograd through it.");
    auto result = tenzor::masked_select(input.tensor(), mask);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<MaskedSelectBackward>();
    const auto& sh = input.tensor().shape();
    auto shape_tensor = zeros({static_cast<int64_t>(sh.size())},
                              DType::Int64, Device::cpu());
    if (!sh.empty()) {
        std::memcpy(shape_tensor.data_ptr(), sh.data(),
                    sh.size() * sizeof(int64_t));
    }
    grad_fn->save_for_backward({mask, shape_tensor});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// masked_scatter(x, mask, source): mask is a plain Tensor (non-diff); source
// is a Variable that participates in the autograd graph. Saves the mask and
// source.shape so backward can pad the source grad to match source's shape.
auto masked_scatter(const Variable& input, const Tensor& mask,
                    const Variable& source) -> Variable {
    auto result = tenzor::masked_scatter(input.tensor(), mask, source.tensor());
    bool any_requires =
        (input.requires_grad() || source.requires_grad()) && is_grad_enabled();
    if (!any_requires) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<MaskedScatterBackward>();
    grad_fn->save_for_backward({mask});
    grad_fn->source_shape_ =
        std::vector<int64_t>(source.tensor().shape().begin(),
                             source.tensor().shape().end());
    grad_fn->set_next_functions({input.grad_fn(), source.grad_fn()});
    grad_fn->set_input_variables({input, source});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// =========================================================================
// Audit E.7 batch 6 — special-math closed forms + view/index ops
// =========================================================================

// igamma(a, x) — only x gets a non-trivial gradient.
auto igamma(const Variable& a, const Variable& x) -> Variable {
    if ((!a.requires_grad() && !x.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::igamma(a.tensor(), x.tensor()), false);
    }
    auto result = tenzor::igamma(a.tensor(), x.tensor());
    auto grad_fn = std::make_shared<IgammaBackward>();
    grad_fn->save_for_backward({a.tensor(), x.tensor()});
    grad_fn->set_next_functions({a.grad_fn(), x.grad_fn()});
    grad_fn->set_input_variables({a, x});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// igammac(a, x) — only x gets a non-trivial gradient.
auto igammac(const Variable& a, const Variable& x) -> Variable {
    if ((!a.requires_grad() && !x.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::igammac(a.tensor(), x.tensor()), false);
    }
    auto result = tenzor::igammac(a.tensor(), x.tensor());
    auto grad_fn = std::make_shared<IgammacBackward>();
    grad_fn->save_for_backward({a.tensor(), x.tensor()});
    grad_fn->set_next_functions({a.grad_fn(), x.grad_fn()});
    grad_fn->set_input_variables({a, x});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// beta(a, b) — both inputs differentiable via digamma.
auto beta(const Variable& a, const Variable& b) -> Variable {
    if ((!a.requires_grad() && !b.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::beta(a.tensor(), b.tensor()), false);
    }
    auto result = tenzor::beta(a.tensor(), b.tensor());
    auto grad_fn = std::make_shared<BetaBackward>();
    // Save a, b, and the realised output y (used in dB/da = y*(psi(a) - psi(a+b))).
    grad_fn->save_for_backward({a.tensor(), b.tensor(), result});
    grad_fn->input_shape_a_ = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    grad_fn->input_shape_b_ = std::vector<int64_t>(b.shape().begin(), b.shape().end());
    grad_fn->set_next_functions({a.grad_fn(), b.grad_fn()});
    grad_fn->set_input_variables({a, b});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// pairwise_distance(x1, x2, p) — per-row L_p distance.
auto pairwise_distance(const Variable& x1, const Variable& x2, double p) -> Variable {
    if ((!x1.requires_grad() && !x2.requires_grad()) || !is_grad_enabled()) {
        return Variable(tenzor::pairwise_distance(x1.tensor(), x2.tensor(), p), false);
    }
    auto result = tenzor::pairwise_distance(x1.tensor(), x2.tensor(), p);
    auto grad_fn = std::make_shared<PairwiseDistanceBackward>(p);
    grad_fn->save_for_backward({x1.tensor(), x2.tensor(), result});
    grad_fn->set_next_functions({x1.grad_fn(), x2.grad_fn()});
    grad_fn->set_input_variables({x1, x2});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// pdist(input, p) — non-differentiable (kernel gap; see PdistBackward).
auto pdist(const Variable& input, double p) -> Variable {
    auto result = tenzor::pdist(input.tensor(), p);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<PdistBackward>(p);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// cdist(x1, x2, p) — non-differentiable (kernel gap; see CDistBackward).
auto cdist(const Variable& x1, const Variable& x2, double p) -> Variable {
    auto result = tenzor::cdist(x1.tensor(), x2.tensor(), p);
    bool needs_grad = (x1.requires_grad() || x2.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<CDistBackward>(p);
    grad_fn->set_next_functions({x1.grad_fn(), x2.grad_fn()});
    grad_fn->set_input_variables({x1, x2});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// index(input, indices) — advanced indexing.
// Marked non-differentiable until an accumulating multi-dim scatter
// kernel lands (see AdvancedIndexBackward docs).
auto index(const Variable& input,
           const std::vector<std::optional<Tensor>>& indices) -> Variable {
    auto result = tenzor::index(input.tensor(), indices);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<AdvancedIndexBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// one_hot — non-differentiable (integer index input).
auto one_hot(const Variable& input, int64_t num_classes) -> Variable {
    auto result = tenzor::one_hot(input.tensor(), num_classes);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<OneHotBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// lerp(start, end, weight) — tensor-weight overload.
auto lerp(const Variable& start, const Variable& end, const Variable& weight) -> Variable {
    bool needs_grad =
        (start.requires_grad() || end.requires_grad() || weight.requires_grad()) &&
        is_grad_enabled();
    if (!needs_grad) {
        return Variable(tenzor::lerp(start.tensor(), end.tensor(), weight.tensor()), false);
    }
    auto result = tenzor::lerp(start.tensor(), end.tensor(), weight.tensor());
    auto grad_fn = std::make_shared<LerpBackward>();
    grad_fn->save_for_backward({start.tensor(), end.tensor(), weight.tensor()});
    grad_fn->input_shape_start_  = std::vector<int64_t>(start.shape().begin(),  start.shape().end());
    grad_fn->input_shape_end_    = std::vector<int64_t>(end.shape().begin(),    end.shape().end());
    grad_fn->input_shape_weight_ = std::vector<int64_t>(weight.shape().begin(), weight.shape().end());
    grad_fn->set_next_functions({start.grad_fn(), end.grad_fn(), weight.grad_fn()});
    grad_fn->set_input_variables({start, end, weight});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// lerp(start, end, weight) — scalar-weight overload.
auto lerp(const Variable& start, const Variable& end, double weight) -> Variable {
    bool needs_grad =
        (start.requires_grad() || end.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(tenzor::lerp(start.tensor(), end.tensor(), weight), false);
    }
    auto result = tenzor::lerp(start.tensor(), end.tensor(), weight);
    auto grad_fn = std::make_shared<LerpBackward>(weight);
    grad_fn->save_for_backward({start.tensor(), end.tensor()});
    grad_fn->input_shape_start_ = std::vector<int64_t>(start.shape().begin(), start.shape().end());
    grad_fn->input_shape_end_   = std::vector<int64_t>(end.shape().begin(),   end.shape().end());
    grad_fn->set_next_functions({start.grad_fn(), end.grad_fn()});
    grad_fn->set_input_variables({start, end});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// cross(a, b, dim) — 3-vector cross product.
auto cross(const Variable& a, const Variable& b, int64_t dim) -> Variable {
    bool needs_grad = (a.requires_grad() || b.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(tenzor::cross(a.tensor(), b.tensor(), dim), false);
    }
    auto result = tenzor::cross(a.tensor(), b.tensor(), dim);
    auto grad_fn = std::make_shared<CrossBackward>(dim);
    grad_fn->save_for_backward({a.tensor(), b.tensor()});
    grad_fn->set_next_functions({a.grad_fn(), b.grad_fn()});
    grad_fn->set_input_variables({a, b});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ---- Audit E.7 batch 7 — index/scatter/view ops --------------------------

// index_add(input, dim, index, source). `index` is integer tensor (non-diff).
auto index_add(const Variable& input, int64_t dim, const Tensor& index,
               const Variable& source) -> Variable {
    auto result = tenzor::index_add(input.tensor(), dim, index, source.tensor());
    bool needs_grad =
        (input.requires_grad() || source.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<IndexAddBackward>(dim);
    grad_fn->save_for_backward({index});
    grad_fn->set_next_functions({input.grad_fn(), source.grad_fn()});
    grad_fn->set_input_variables({input, source});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// index_copy(input, dim, index, source).
auto index_copy(const Variable& input, int64_t dim, const Tensor& index,
                const Variable& source) -> Variable {
    auto result = tenzor::index_copy(input.tensor(), dim, index, source.tensor());
    bool needs_grad =
        (input.requires_grad() || source.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<IndexCopyBackward>(dim);
    grad_fn->save_for_backward({index});
    grad_fn->set_next_functions({input.grad_fn(), source.grad_fn()});
    grad_fn->set_input_variables({input, source});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// index_fill(input, dim, index, value). value scalar non-diff.
auto index_fill(const Variable& input, int64_t dim, const Tensor& index,
                float value) -> Variable {
    auto result = tenzor::index_fill(input.tensor(), dim, index, value);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<IndexFillBackward>(dim);
    grad_fn->save_for_backward({index});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// select_scatter(input, src, dim, index).
auto select_scatter(const Variable& input, const Variable& src,
                    int64_t dim, int64_t index) -> Variable {
    auto result = tenzor::select_scatter(input.tensor(), src.tensor(), dim, index);
    bool needs_grad =
        (input.requires_grad() || src.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<SelectScatterBackward>(dim, index);
    grad_fn->set_next_functions({input.grad_fn(), src.grad_fn()});
    grad_fn->set_input_variables({input, src});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// slice_scatter(input, src, dim, start, end, step).
auto slice_scatter(const Variable& input, const Variable& src, int64_t dim,
                   int64_t start, int64_t end, int64_t step) -> Variable {
    auto result = tenzor::slice_scatter(input.tensor(), src.tensor(), dim,
                                        start, end, step);
    bool needs_grad =
        (input.requires_grad() || src.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<SliceScatterBackward>(dim, start, end, step);
    grad_fn->src_shape_ =
        std::vector<int64_t>(src.shape().begin(), src.shape().end());
    grad_fn->set_next_functions({input.grad_fn(), src.grad_fn()});
    grad_fn->set_input_variables({input, src});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// diagonal_scatter(input, src, offset, dim1, dim2). Non-differentiable
// (see DiagonalScatterBackward docs for the missing N-D diagonal extractor).
auto diagonal_scatter(const Variable& input, const Variable& src,
                      int64_t offset, int64_t dim1, int64_t dim2) -> Variable {
    auto result = tenzor::diagonal_scatter(input.tensor(), src.tensor(),
                                           offset, dim1, dim2);
    bool needs_grad =
        (input.requires_grad() || src.requires_grad()) && is_grad_enabled();
    if (!needs_grad) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<DiagonalScatterBackward>(offset, dim1, dim2);
    grad_fn->set_next_functions({input.grad_fn(), src.grad_fn()});
    grad_fn->set_input_variables({input, src});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// repeat_interleave(input, repeats: int, dim).
auto repeat_interleave(const Variable& input, int64_t repeats,
                       std::optional<int64_t> dim) -> Variable {
    auto result = tenzor::repeat_interleave(input.tensor(), repeats, dim);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    // CC.2: `was_flattened` records whether forward took the nullopt-dim
    // flatten path. Backward needs this to reshape grad_y back to the
    // original rank-N input shape; without it the backward can't tell a
    // genuine 1D input from an N-D input that was flattened internally.
    bool was_flattened = !dim.has_value();
    auto grad_fn = std::make_shared<RepeatInterleaveBackward>(
        repeats, dim, std::move(input_shape), was_flattened);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// repeat_interleave(input, repeats: Tensor, dim) — non-differentiable
// (per-element variable-length expansion; see RepeatInterleaveBackward docs).
// We still attach a Function so the graph fails *loudly* at backward time
// rather than silently producing wrong gradients. Reuses the
// `RepeatInterleaveBackward` op_id via a typed wrapper that throws.
auto repeat_interleave(const Variable& input, const Tensor& repeats,
                       std::optional<int64_t> dim) -> Variable {
    auto result = tenzor::repeat_interleave(input.tensor(), repeats, dim);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    // Attach a uniform-repeats backward configured to throw NonDifferentiable
    // at use time — by using repeats=-1 we sentinel that the per-element
    // overload was used; the class detects this and throws.
    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    bool was_flattened = !dim.has_value();
    auto grad_fn = std::make_shared<RepeatInterleaveBackward>(
        /*repeats=*/-1, dim, std::move(input_shape), was_flattened);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// unfold(input, kernel_size, stride, padding, dilation) — im2col.
auto unfold(const Variable& input, int64_t kernel_size, int64_t stride,
            int64_t padding, int64_t dilation) -> Variable {
    auto result = tenzor::ops::unfold(input.tensor(), kernel_size, stride,
                                      padding, dilation);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    // Pull H, W from the input shape (N, C, H, W).
    auto in_shape = input.shape();
    if (in_shape.size() != 4) {
        throw std::runtime_error(
            "unfold (autograd wrapper) expects 4D input (N, C, H, W); got rank "
            + std::to_string(in_shape.size()));
    }
    int64_t H = in_shape[2];
    int64_t W = in_shape[3];
    auto grad_fn = std::make_shared<UnfoldBackward>(
        kernel_size, stride, padding, dilation, H, W);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// nonzero — non-differentiable.
auto nonzero(const Variable& input) -> Variable {
    auto result = tenzor::nonzero(input.tensor());
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<NonzeroBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// unique — non-differentiable. Returns only the unique-values tensor; the
// optional inverse/counts outputs are integer-typed and don't participate in
// the autograd surface.
auto unique(const Variable& input, bool sorted, bool return_inverse,
            bool return_counts) -> Variable {
    auto [values, inverse, counts] =
        tenzor::unique(input.tensor(), sorted, return_inverse, return_counts);
    (void)inverse;
    (void)counts;
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(values, false);
    }
    auto grad_fn = std::make_shared<UniqueBackward>();
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(values, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// =========================================================================
// Audit E.7 batch 8 — order stats, integration, segment ops
// =========================================================================

// aminmax — returns (min, max). Each output Variable owns its own
// AminmaxBackward, configured to scatter its incoming grad onto positions
// equal to its saved value (tie-normalised). The autograd engine sums when
// both branches share an upstream input.
auto aminmax(const Variable& input, std::optional<int64_t> dim, bool keepdim)
    -> std::pair<Variable, Variable> {
    auto [min_vals, max_vals] =
        tenzor::aminmax(input.tensor(), dim, keepdim);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return {Variable(min_vals, false), Variable(max_vals, false)};
    }

    auto make_branch = [&](const Tensor& vals, bool is_max) {
        auto grad_fn = std::make_shared<AminmaxBackward>(dim, keepdim, is_max);
        grad_fn->save_for_backward({input.tensor(), vals});
        grad_fn->set_next_functions({input.grad_fn()});
        grad_fn->set_input_variables({input});
        Variable out(vals, true);
        out.set_grad_fn(grad_fn);
        return out;
    };
    return {make_branch(min_vals, /*is_max=*/false),
            make_branch(max_vals, /*is_max=*/true)};
}

// kthvalue — returns value only at the autograd surface (index is non-diff).
auto kthvalue(const Variable& input, int64_t k, int64_t dim, bool keepdim)
    -> Variable {
    auto [values, indices] =
        tenzor::kthvalue(input.tensor(), k, dim, keepdim);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(values, false);
    }
    auto grad_fn = std::make_shared<KthvalueBackward>(k, dim, keepdim);
    grad_fn->save_for_backward({input.tensor(), values});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(values, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// quantile — non-differentiable.
auto quantile(const Variable& input, double q, std::optional<int64_t> dim,
              bool keepdim) -> Variable {
    auto values = tenzor::quantile(input.tensor(), q, dim, keepdim);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(values, false);
    }
    auto grad_fn = std::make_shared<QuantileBackward>(q, dim, keepdim);
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(values, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// nanmedian — differentiable (median backward with NaN positions excluded).
auto nanmedian(const Variable& input, std::optional<int64_t> dim) -> Variable {
    auto values = tenzor::nanmedian(input.tensor(), dim);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(values, false);
    }
    auto grad_fn = std::make_shared<NanmedianBackward>(dim);
    grad_fn->save_for_backward({input.tensor(), values});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable output(values, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// trapezoid — uniform-dx overload (only y is differentiable).
auto trapezoid(const Variable& y, double dx, int64_t dim) -> Variable {
    auto result = tenzor::trapezoid(y.tensor(), dx, dim);
    if (!y.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<TrapezoidBackward>(dim, dx, /*has_x=*/false);
    grad_fn->save_for_backward({y.tensor()});
    grad_fn->set_next_functions({y.grad_fn()});
    grad_fn->set_input_variables({y});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// trapezoid — non-uniform x overload. x is treated as non-diff (matches
// PyTorch). We still attach it as a saved tensor so the backward can read
// dx_i = x_{i+1} - x_i. The next_functions slot for x is nullptr (no grad
// routed back); the backward returns a zero of x.shape for that slot.
auto trapezoid(const Variable& y, const Variable& x, int64_t dim) -> Variable {
    auto result = tenzor::trapezoid(y.tensor(), x.tensor(), dim);
    const bool y_needs = y.requires_grad() && is_grad_enabled();
    if (!y_needs) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<TrapezoidBackward>(dim, /*dx=*/0.0, /*has_x=*/true);
    grad_fn->save_for_backward({y.tensor(), x.tensor()});
    // We don't route a gradient through x, but the next_functions vector
    // must align with the input_variables vector for the engine. Provide
    // nullptr for x's slot.
    grad_fn->set_next_functions({y.grad_fn(), nullptr});
    grad_fn->set_input_variables({y, x});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// cumulative_trapezoid — uniform-dx overload.
auto cumulative_trapezoid(const Variable& y, double dx, int64_t dim) -> Variable {
    auto result = tenzor::cumulative_trapezoid(y.tensor(), dx, dim);
    if (!y.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<CumulativeTrapezoidBackward>(
        dim, dx, /*has_x=*/false);
    grad_fn->save_for_backward({y.tensor()});
    grad_fn->set_next_functions({y.grad_fn()});
    grad_fn->set_input_variables({y});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// cumulative_trapezoid — non-uniform x overload (only y is diff).
auto cumulative_trapezoid(const Variable& y, const Variable& x, int64_t dim)
    -> Variable {
    auto result = tenzor::cumulative_trapezoid(y.tensor(), x.tensor(), dim);
    const bool y_needs = y.requires_grad() && is_grad_enabled();
    if (!y_needs) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<CumulativeTrapezoidBackward>(
        dim, /*dx=*/0.0, /*has_x=*/true);
    grad_fn->save_for_backward({y.tensor(), x.tensor()});
    grad_fn->set_next_functions({y.grad_fn(), nullptr});
    grad_fn->set_input_variables({y, x});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// segment_reduce — non-differentiable (per project policy; see
// SegmentReduceBackward).
auto segment_reduce(const Variable& data, const Tensor& offsets,
                    const std::string& reduce, int64_t axis) -> Variable {
    auto result = tenzor::segment_reduce(data.tensor(), offsets, reduce, axis);
    if (!data.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<SegmentReduceBackward>();
    grad_fn->set_next_functions({data.grad_fn()});
    grad_fn->set_input_variables({data});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// gumbel_softmax — non-differentiable (forward doesn't save the Gumbel
// noise; see GumbelSoftmaxBackward).
auto gumbel_softmax(const Variable& logits, double tau, bool hard, int64_t dim)
    -> Variable {
    auto result = tenzor::gumbel_softmax(logits.tensor(), tau, hard, dim);
    if (!logits.requires_grad() || !is_grad_enabled()) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<GumbelSoftmaxBackward>(tau, hard, dim);
    grad_fn->set_next_functions({logits.grad_fn()});
    grad_fn->set_input_variables({logits});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// cummax — returns (values_var, indices_var). values is differentiable
// (scatter_add of grad onto saved indices); indices is non-diff (Int64).
// Saves an input-shape placeholder + indices so the backward can rebuild
// a zeros tensor of the right shape.
namespace {
auto make_shape_placeholder(const std::vector<int64_t>& shape,
                            DType dtype, Device device) -> Tensor {
    // Zero tensor of the requested shape; the backward only reads
    // `.shape()`. Use the input's dtype/device for parity in saved storage.
    return zeros(shape, dtype, device);
}
}  // namespace

auto cummax(const Variable& input, int64_t dim)
    -> std::pair<Variable, Variable> {
    auto [values, indices] = tenzor::cummax(input.tensor(), dim);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return {Variable(values, false), Variable(indices, false)};
    }
    auto grad_fn = std::make_shared<CumMaxBackward>(dim);
    auto input_shape =
        std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto shape_holder =
        make_shape_placeholder(input_shape, values.dtype(), values.device());
    grad_fn->save_for_backward({shape_holder, indices});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable values_var(values, true);
    values_var.set_grad_fn(grad_fn);
    // Indices: not differentiable. Return as a no-grad Variable.
    return {values_var, Variable(indices, false)};
}

auto cummin(const Variable& input, int64_t dim)
    -> std::pair<Variable, Variable> {
    auto [values, indices] = tenzor::cummin(input.tensor(), dim);
    if (!input.requires_grad() || !is_grad_enabled()) {
        return {Variable(values, false), Variable(indices, false)};
    }
    auto grad_fn = std::make_shared<CumMinBackward>(dim);
    auto input_shape =
        std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto shape_holder =
        make_shape_placeholder(input_shape, values.dtype(), values.device());
    grad_fn->save_for_backward({shape_holder, indices});
    grad_fn->set_next_functions({input.grad_fn()});
    grad_fn->set_input_variables({input});
    Variable values_var(values, true);
    values_var.set_grad_fn(grad_fn);
    return {values_var, Variable(indices, false)};
}

// ---- Audit E.7 batch 9 — indexing reductions ---------------------------

// scatter_reduce(input, dim, index, src, reduce, include_self)
// Differentiable for "sum"/"mean" — see ScatterReduceBackward for the
// closed form (gather / scatter-add of grad_out divided by per-position
// count for mean). For other reductions the wrapper still attaches the
// grad_fn so backward() throws a typed NonDifferentiable.
auto scatter_reduce(const Variable& input, int64_t dim, const Tensor& index,
                    const Variable& src, const std::string& reduce,
                    bool include_self) -> Variable {
    auto result = tenzor::scatter_reduce(input.tensor(), dim, index,
                                          src.tensor(), reduce, include_self);
    bool needs_grad = (input.requires_grad() || src.requires_grad())
                      && is_grad_enabled();
    if (!needs_grad) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<ScatterReduceBackward>(dim, reduce,
                                                            include_self);
    grad_fn->save_for_backward({index});
    grad_fn->set_next_functions({input.grad_fn(), src.grad_fn()});
    grad_fn->set_input_variables({input, src});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// index_reduce(input, dim, index, src, reduce, include_self)
auto index_reduce(const Variable& input, int64_t dim, const Tensor& index,
                  const Variable& src, const std::string& reduce,
                  bool include_self) -> Variable {
    auto result = tenzor::index_reduce(input.tensor(), dim, index,
                                        src.tensor(), reduce, include_self);
    bool needs_grad = (input.requires_grad() || src.requires_grad())
                      && is_grad_enabled();
    if (!needs_grad) {
        return Variable(result, false);
    }
    auto grad_fn = std::make_shared<IndexReduceBackward>(dim, reduce,
                                                          include_self);
    grad_fn->save_for_backward({index});
    grad_fn->set_next_functions({input.grad_fn(), src.grad_fn()});
    grad_fn->set_input_variables({input, src});
    Variable output(result, true);
    output.set_grad_fn(grad_fn);
    return output;
}

} // namespace tenzor
