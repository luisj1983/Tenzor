/**
 * @file function_distributed.cpp
 * @brief Autograd Function for distributed collective ops (A5).
 *
 * Implements `tenzor::DistributedAllReduceBackward` and the public wrapper
 * `tenzor::distributed_all_reduce`. See `function_distributed.hpp` for the
 * math derivation and design notes.
 */

#include "tenzor/autograd/function_distributed.hpp"
#include "tenzor/autograd/variable.hpp"   // is_grad_enabled()
#include "tenzor/ops/creation.hpp"

#include <stdexcept>
#include <string>

namespace tenzor {

namespace {

auto is_differentiable_reduce(distributed::ReduceOp op) -> bool {
    using R = distributed::ReduceOp;
    return op == R::SUM || op == R::AVG;
}

auto reduce_op_name(distributed::ReduceOp op) -> const char* {
    using R = distributed::ReduceOp;
    switch (op) {
        case R::SUM:     return "SUM";
        case R::PRODUCT: return "PRODUCT";
        case R::MIN:     return "MIN";
        case R::MAX:     return "MAX";
        case R::BAND:    return "BAND";
        case R::BOR:     return "BOR";
        case R::BXOR:    return "BXOR";
        case R::AVG:     return "AVG";
    }
    return "?";
}

} // namespace

auto DistributedAllReduceBackward::forward(std::vector<Variable> inputs)
    -> std::vector<Variable>
{
    // Forward should not be called directly; the public wrapper
    // `distributed_all_reduce` does the actual work and wires `grad_fn`.
    // This mirrors the convention used by the other Function subclasses
    // (e.g. UpsampleBilinearBackward).
    (void)inputs;
    throw std::runtime_error(
        "DistributedAllReduceBackward::forward should not be called directly; "
        "use tenzor::distributed_all_reduce(input, pg, op).");
}

auto DistributedAllReduceBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor>
{
    if (grad_outputs.size() != 1) {
        throw std::runtime_error(
            "DistributedAllReduceBackward::backward expects exactly 1 grad output");
    }
    if (!is_differentiable_reduce(op_)) {
        throw std::runtime_error(
            std::string("DistributedAllReduceBackward: ReduceOp::") +
            reduce_op_name(op_) +
            " is not differentiable through all-reduce. Only SUM and AVG are.");
    }
    if (pg_ == nullptr) {
        throw std::runtime_error(
            "DistributedAllReduceBackward: process_group is null");
    }

    // Clone so the caller's grad tensor isn't mutated by the in-place reduce.
    Tensor grad = grad_outputs[0].clone();
    pg_->all_reduce(grad, op_);
    return {grad};
}

auto DistributedAllReduceBackward::backward_with_variables(
    std::vector<Variable> grad_outputs) -> std::vector<Variable>
{
    if (grad_outputs.size() != 1) {
        throw std::runtime_error(
            "DistributedAllReduceBackward::backward_with_variables expects 1 grad");
    }
    if (!is_differentiable_reduce(op_)) {
        throw std::runtime_error(
            std::string("DistributedAllReduceBackward: ReduceOp::") +
            reduce_op_name(op_) +
            " is not differentiable through all-reduce. Only SUM and AVG are.");
    }
    // Re-enter the public wrapper so higher-order gradients build their own
    // DistributedAllReduceBackward node. The graph remains intact.
    return {distributed_all_reduce(grad_outputs[0], pg_, op_)};
}

auto distributed_all_reduce(
    const Variable& input,
    std::shared_ptr<distributed::ProcessGroupBase> pg,
    distributed::ReduceOp op) -> Variable
{
    // audit-10 OO.2: use std::runtime_error to match the Tensor-level
    // DistributedAllReduceBackward::backward() validation above (which also
    // throws runtime_error on null pg / non-differentiable op).  Previously
    // the public wrapper threw std::invalid_argument, so callers writing
    // try/catch around distributed_all_reduce vs. backward() needed two
    // different handlers for the same logical precondition.
    if (pg == nullptr) {
        throw std::runtime_error(
            "distributed_all_reduce: process_group must not be null");
    }
    if (!is_differentiable_reduce(op)) {
        throw std::runtime_error(
            std::string("distributed_all_reduce: ReduceOp::") +
            reduce_op_name(op) +
            " is not differentiable. Only SUM and AVG are supported here.");
    }

    // Clone the input tensor so the in-place reduce does not mutate the
    // caller's Variable.
    Tensor reduced = input.tensor().clone();
    pg->all_reduce(reduced, op);

    Variable result(reduced, input.requires_grad());

    if (input.requires_grad() && is_grad_enabled()) {
        auto grad_fn = std::make_shared<DistributedAllReduceBackward>(pg, op);

        // No saved tensors are needed for all-reduce backward (it depends
        // only on the incoming gradient and on pg/op).
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.push_back(input.grad_fn());  // nullptr if leaf
        grad_fn->set_next_functions(std::move(next_funcs));

        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        grad_fn->set_input_variables(std::move(input_vars));

        result.set_grad_fn(grad_fn);
    }

    return result;
}

} // namespace tenzor
