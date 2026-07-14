/**
 * @file control_flow.cpp
 * @brief Implementation of JIT-compatible control flow primitives
 */

#include "tenzor/jit/control_flow.hpp"

namespace tenzor {
namespace jit {

// ============================================================================
// tensor_condition_to_bool
// ============================================================================

auto tensor_condition_to_bool(const Tensor& condition) -> bool {
    return condition.to(Device::cpu()).to(DType::Float64).item<double>() != 0.0;
}

// ============================================================================
// cond (multi-output)
// ============================================================================

auto cond(const Tensor& condition,
          std::function<std::vector<Variable>(const std::vector<Variable>&)> then_fn,
          std::function<std::vector<Variable>(const std::vector<Variable>&)> else_fn,
          const std::vector<Variable>& args) -> std::vector<Variable> {

    auto& tracer = Tracer::get_instance();

    if (tracer.is_tracing()) {
        // Tracing mode: record both branches as subgraphs
        return tracer.trace_if(condition,
                               std::move(then_fn),
                               std::move(else_fn),
                               args);
    }

    // Eager mode: evaluate condition and call appropriate branch.
    // tensor_condition_to_bool's doc comment explains the CPU-first,
    // Float64-widen cast order this shares with while_loop(), the scripted
    // `if` (script.cpp), and the compiled-graph interpreter (graph.cpp).
    bool cond_val = tensor_condition_to_bool(condition);

    if (cond_val) {
        return then_fn(args);
    } else {
        return else_fn(args);
    }
}

// ============================================================================
// cond (single-output convenience)
// ============================================================================

auto cond(const Tensor& condition,
          std::function<Variable(const Variable&)> then_fn,
          std::function<Variable(const Variable&)> else_fn,
          const Variable& input) -> Variable {

    auto results = cond(
        condition,
        [&then_fn](const std::vector<Variable>& args) -> std::vector<Variable> {
            return {then_fn(args[0])};
        },
        [&else_fn](const std::vector<Variable>& args) -> std::vector<Variable> {
            return {else_fn(args[0])};
        },
        {input});

    // A branch that returned nothing yields an empty scalar; inherit the input's
    // device and dtype so a CUDA/ROCm/Float64 trace doesn't silently degrade to
    // a CPU Float32 scalar (which would then device-mismatch downstream ops).
    return results.empty()
               ? Variable(Tensor({}, input.tensor().dtype(),
                                  input.tensor().device()))
               : results[0];
}

// ============================================================================
// while_loop
// ============================================================================

auto while_loop(int64_t max_iter,
                std::function<Tensor(const std::vector<Variable>&)> cond_fn,
                std::function<std::vector<Variable>(const std::vector<Variable>&)> body_fn,
                const std::vector<Variable>& carried) -> std::vector<Variable> {

    auto& tracer = Tracer::get_instance();

    if (tracer.is_tracing()) {
        // Tracing mode: record loop body as subgraph
        return tracer.trace_loop(max_iter,
                                 std::move(cond_fn),
                                 std::move(body_fn),
                                 carried);
    }

    // Eager mode: execute loop directly
    std::vector<Variable> state = carried;

    for (int64_t i = 0; i < max_iter; ++i) {
        // Check condition. tensor_condition_to_bool (see cond()) applies the
        // same host-side, Float64-widening cast so a NaN or denormal-small
        // predicate can't flip this loop's exit decision vs CPU.
        Tensor cond_result = cond_fn(state);
        if (!tensor_condition_to_bool(cond_result)) {
            break;
        }

        // Execute body
        state = body_fn(state);
    }

    return state;
}

} // namespace jit
} // namespace tenzor
