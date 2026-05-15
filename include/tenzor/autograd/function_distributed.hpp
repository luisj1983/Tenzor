/**
 * @file function_distributed.hpp
 * @brief Autograd Function for distributed collective operations.
 *
 * Provides a Variable-level all-reduce primitive (`distributed_all_reduce`)
 * whose forward dispatches `pg->all_reduce(tensor, op)` and whose backward
 * applies the same all-reduce to the incoming gradient.
 *
 * Math justification: with L = sum_r L_r computed globally across ranks,
 * for `y_local = all_reduce(x_local, SUM)` we have y identical on every rank,
 * and dL/dx_local = sum_r dL_r/dy_local = all_reduce(grad_y_local, SUM).
 * For AVG the same identity holds with a 1/N factor that is already part of
 * the AVG reduction. Higher-order gradients flow through because the
 * backward simply re-enters `distributed_all_reduce` at the Variable level.
 *
 * Audit remediation: A5. Unblocks C1 (SyncBatchNorm autograd-aware all-reduce).
 */

#pragma once

#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/distributed/distributed.hpp"     // for ReduceOp
#include "tenzor/distributed/process_group.hpp"   // for ProcessGroupBase
#include <memory>

namespace tenzor {

/**
 * @brief Autograd Function for the collective all-reduce.
 *
 * Forward: calls `pg->all_reduce(local_clone, op)` and returns the reduced
 * tensor as a Variable. The input's underlying buffer is NOT mutated — we
 * clone first.
 *
 * Backward: for SUM/AVG, the gradient flow is itself an all-reduce of the
 * incoming gradient with the same op (see file header). For other ops
 * (PRODUCT/MIN/MAX/B*) gradients are not generally well-defined in the
 * pure-collective sense and the backward throws.
 */
class DistributedAllReduceBackward : public Function {
public:
    DistributedAllReduceBackward(
        std::shared_ptr<distributed::ProcessGroupBase> pg,
        distributed::ReduceOp op)
        : pg_(std::move(pg)), op_(op) {}

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs)
        -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }

private:
    std::shared_ptr<distributed::ProcessGroupBase> pg_;
    distributed::ReduceOp op_;
};

/**
 * @brief Differentiable all-reduce on a Variable.
 *
 * @param input Local Variable to reduce across the process group.
 * @param pg    Process group spanning the participating ranks.
 * @param op    Reduction op. Only `SUM` and `AVG` are differentiable.
 *
 * @return A new Variable whose tensor holds the reduced value; on every rank
 *         the values are identical. Gradient flow back through this op is
 *         itself an all-reduce of the incoming gradient.
 *
 * @note Single-process semantics (`pg->world_size() == 1`) is a clone — the
 *       output Variable shares no buffer with `input`. This preserves
 *       autograd invariants (no aliasing through the op).
 */
auto distributed_all_reduce(
    const Variable& input,
    std::shared_ptr<distributed::ProcessGroupBase> pg,
    distributed::ReduceOp op = distributed::ReduceOp::SUM) -> Variable;

} // namespace tenzor
