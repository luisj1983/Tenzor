/**
 * @file control_flow.cpp
 * @brief Implementation of JIT-compatible control flow primitives
 */

#include "tenzor/jit/control_flow.hpp"

namespace tenzor {
namespace jit {

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

    // Eager mode: evaluate condition and call appropriate branch. Move to CPU
    // BEFORE the Float32 cast: casting on the device lets some backends (e.g.
    // ROCm) canonicalize a NaN condition to 0, flipping the taken branch vs CPU
    // (NaN != 0 is true). A host-side cast keeps NaN as NaN so the branch choice
    // is identical across all backends.
    bool cond_val =
        condition.to(Device::cpu()).to(DType::Float32).item<float>() != 0.0f;

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
        // Check condition
        Tensor cond_result = cond_fn(state);
        // Host-side cast (see cond()): a device-side Float32 cast can canonicalize
        // a NaN loop predicate to 0 on some backends, exiting the loop early vs
        // CPU (NaN == 0 is false → continue). Keep the check deterministic.
        if (cond_result.to(Device::cpu()).to(DType::Float32).item<float>() == 0.0f) {
            break;
        }

        // Execute body
        state = body_fn(state);
    }

    return state;
}

} // namespace jit
} // namespace tenzor
