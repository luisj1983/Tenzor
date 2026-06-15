/**
 * @file dispatch_interceptor.hpp
 * @brief Composable dispatch interceptor stack for layering cross-cutting
 *        concerns (autocast, profiling, tracing) onto the dispatch path.
 *
 * Design goals:
 * - Zero overhead when no interceptors are active (single branch on empty check)
 * - Thread-local stack for per-thread composition
 * - RAII guard for exception-safe push/pop
 *
 * Usage:
 * @code
 * // Push an interceptor that logs every dispatch
 * InterceptorGuard guard([](OpId op, std::span<const Tensor> inputs,
 *                           const OpAttributes& attrs, DispatchNext next) {
 *     std::cout << "dispatching " << static_cast<int>(op) << "\n";
 *     return next(op, inputs, attrs);
 * });
 * @endcode
 */

#pragma once

#include <functional>
#include <span>
#include <stdexcept>
#include <vector>
#include "../core/tensor.hpp"
#include "../ops/op_id.hpp"
#include "backend.hpp"  // For OpAttributes (alias for NewOpAttributes)

namespace tenzor {

/// Callable that invokes the next interceptor (or the terminal kernel).
using DispatchNext = std::function<std::vector<Tensor>(
    OpId, std::span<const Tensor>, const OpAttributes&)>;

/// Single-output variant.
using DispatchNextSingle = std::function<Tensor(
    OpId, std::span<const Tensor>, const OpAttributes&)>;

/// An interceptor receives (op, inputs, attrs, next) and must call next to
/// continue the chain (or short-circuit by returning its own result).
using DispatchInterceptor = std::function<std::vector<Tensor>(
    OpId, std::span<const Tensor>, const OpAttributes&, DispatchNext)>;

/**
 * @brief Thread-local interceptor stack for composable dispatch.
 *
 * When the stack is empty, `run()` calls the terminal directly with
 * no overhead beyond a single `empty()` check (~1ns).
 *
 * With N interceptors, the chain is:
 *   interceptor[0] → interceptor[1] → ... → interceptor[N-1] → terminal
 */
class DispatchInterceptorStack {
public:
    /// Push an interceptor onto the current thread's stack.
    static void push(DispatchInterceptor interceptor) {
        stack_().push_back(std::move(interceptor));
    }

    /// Pop the most recently pushed interceptor.
    static void pop() {
        auto& s = stack_();
        if (!s.empty()) s.pop_back();
    }

    /// Number of active interceptors on this thread.
    static size_t depth() noexcept {
        return stack_().size();
    }

    /**
     * @brief Execute the interceptor chain, ending with terminal.
     *
     * Fast path: if no interceptors, calls terminal directly.
     */
    static std::vector<Tensor> run(
        OpId op,
        std::span<const Tensor> inputs,
        const OpAttributes& attrs,
        DispatchNext terminal)
    {
        auto& s = stack_();
        if (s.empty()) [[likely]] {
            return terminal(op, inputs, attrs);
        }
        return run_chain_(s, 0, op, inputs, attrs, std::move(terminal));
    }

    /**
     * @brief Single-output variant that avoids vector allocation on the fast path.
     */
    static Tensor run_single(
        OpId op,
        std::span<const Tensor> inputs,
        const OpAttributes& attrs,
        DispatchNextSingle terminal)
    {
        auto& s = stack_();
        if (s.empty()) [[likely]] {
            return terminal(op, inputs, attrs);
        }
        // Wrap terminal as multi-output, run chain, extract element 0
        DispatchNext wrapped = [t = std::move(terminal)](
            OpId o, std::span<const Tensor> i, const OpAttributes& a) {
            return std::vector<Tensor>{t(o, i, a)};
        };
        auto result = run_chain_(s, 0, op, inputs, attrs, std::move(wrapped));
        // An interceptor short-circuiting with no output is a programming
        // error. Throw a clear message rather than returning a null-storage
        // Tensor that surfaces as a confusing downstream failure — mirroring
        // BackendDispatchTable::dispatch_single.
        if (result.empty()) {
            throw std::runtime_error(
                "DispatchInterceptorStack::run_single: interceptor chain returned empty result");
        }
        return std::move(result[0]);
    }

private:
    /// Thread-local stack accessor. Defined out-of-line in a single
    /// translation unit (dispatch_interceptor.cpp) so there is exactly ONE
    /// thread_local instance process-wide. A header-inline `static thread_local`
    /// could be duplicated across translation units / inline instantiations
    /// (so push()/depth() in one TU and run() in another saw different stacks),
    /// which silently dropped dispatch interceptors — e.g. the JIT tracer
    /// recorded ops dispatched from some TUs but not others.
    static std::vector<DispatchInterceptor>& stack_();

    /// Recursively build the chain: interceptor[idx] calls run_chain_(idx+1).
    static std::vector<Tensor> run_chain_(
        std::vector<DispatchInterceptor>& s,
        size_t idx,
        OpId op,
        std::span<const Tensor> inputs,
        const OpAttributes& attrs,
        DispatchNext terminal)
    {
        if (idx >= s.size()) {
            return terminal(op, inputs, attrs);
        }
        // Build the "next" callable that invokes the rest of the chain.
        // Capture `terminal` BY VALUE and forward a copy on each call so an
        // interceptor that invokes next() more than once (retry/fan-out) does
        // not move from an already-moved-from std::function (which would call an
        // empty function -> std::bad_function_call).
        DispatchNext next = [&s, idx, terminal](
            OpId o, std::span<const Tensor> i, const OpAttributes& a) {
            return run_chain_(s, idx + 1, o, i, a, terminal);
        };
        return s[idx](op, inputs, attrs, std::move(next));
    }
};

/**
 * @brief RAII guard for push/pop of a dispatch interceptor.
 *
 * @code
 * {
 *     InterceptorGuard guard(my_interceptor);
 *     // ... dispatches go through my_interceptor ...
 * } // guard destroyed, interceptor popped
 * @endcode
 */
class InterceptorGuard {
public:
    explicit InterceptorGuard(DispatchInterceptor interceptor) {
        DispatchInterceptorStack::push(std::move(interceptor));
    }

    ~InterceptorGuard() {
        DispatchInterceptorStack::pop();
    }

    // Non-copyable, non-movable
    InterceptorGuard(const InterceptorGuard&) = delete;
    InterceptorGuard& operator=(const InterceptorGuard&) = delete;
};

} // namespace tenzor
