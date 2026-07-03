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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>

namespace tenzor {
// Forward declarations — keep this bridge header light (no backend/op headers).
class Tensor;
enum class OpId : std::uint16_t;
class NewOpAttributes;
using OpAttributes = NewOpAttributes;
} // namespace tenzor

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

/**
 * @brief Hook type invoked when an in-place op mutates a tensor.
 *
 * In-place kernels dispatch through `dispatch_inplace`, which deliberately
 * bypasses the DispatchInterceptorStack (autocast on in-place ops is
 * semantically unsafe). The tracer therefore never sees add_/sub_/mul_/div_/
 * relu_-style mutations via the normal interceptor, and — because the mutated
 * tensor keeps the same storage/shape/strides — subsequent reads of it would
 * resolve to the PRE-mutation graph value. This hook lets `dispatch_inplace`
 * notify the tracer AFTER the mutation so it can record a real node that
 * produces a fresh SSA value for the mutated tensor (value versioning).
 *
 * @param op     The in-place OpId (e.g. OpId::AddInplace).
 * @param target The just-mutated tensor (post-mutation contents).
 * @param others Pointer to the additional input tensors (may be null if none).
 * @param num_others Count of additional inputs.
 * @param attrs  Op attributes (e.g. clamp bounds, leaky-relu slope).
 */
using InplaceOpHook = std::function<void(
    OpId op, Tensor& target, const Tensor* others, std::size_t num_others,
    const OpAttributes& attrs)>;

/**
 * @brief Install an in-place op hook for the current thread.
 *
 * Called by `TracingGuard`. Passing a default-constructed hook clears it.
 */
void set_inplace_op_hook(InplaceOpHook hook);

/**
 * @brief Report that an in-place op just mutated `target`.
 *
 * No-op (single atomic-free thread-local null check) when no hook is
 * installed, i.e. outside a trace.
 */
void notify_inplace_op(OpId op, Tensor& target, const Tensor* others,
                       std::size_t num_others, const OpAttributes& attrs);

/**
 * @brief True when an in-place hook is installed on this thread.
 *
 * Lets `dispatch_inplace` skip building the notification arguments entirely
 * on the common (non-traced) path.
 */
bool inplace_op_hook_active() noexcept;

} // namespace tenzor::detail
