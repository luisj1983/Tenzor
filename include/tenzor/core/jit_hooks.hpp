/**
 * @file jit_hooks.hpp
 * @brief Thread-local hook points that let `tenzor_core` notify higher
 *        layers (the JIT tracer in particular) about operations that
 *        cannot be cleanly represented in a traced graph.
 *
 * The tracer lives in `tenzor::jit`, which depends on `tenzor_core`. We
 * cannot include `jit/tracer.hpp` from `core/tensor.cpp` without a
 * cyclic dependency, so the JIT side installs a callback at start_trace()
 * and `Tensor::item()` / other leaf operations notify that callback via
 * the free function defined here.
 *
 * If no hook is installed (the common non-traced case), notify_*() is a
 * single atomic-load null check — essentially free.
 */

#pragma once

#include <functional>
#include <string>

namespace tenzor::detail {

/**
 * @brief Hook type invoked when an op cannot be traced cleanly.
 *
 * `reason` is a short human-readable string describing the violation,
 * e.g. "scalar extraction (.item())". The tracer uses it for error
 * messages and telemetry.
 */
using GraphBreakHook = std::function<void(const std::string& reason)>;

/**
 * @brief Install a graph-break hook for the current thread.
 *
 * Called by `TracingGuard` at `start_trace()`. Passing a default-
 * constructed `GraphBreakHook` clears the hook.
 */
void set_graph_break_hook(GraphBreakHook hook);

/**
 * @brief Report that an un-traceable operation just executed.
 *
 * No-op if no hook is installed on the current thread. Otherwise the
 * hook decides whether to warn, record, or throw. In `TENZOR_JIT_STRICT`
 * mode the Tracer's hook throws `std::runtime_error` with a message
 * that names the offending op and suggests `tenzor::jit::cond` /
 * `jit::while_loop` as the replacement.
 */
void notify_graph_break(const std::string& reason);

} // namespace tenzor::detail
