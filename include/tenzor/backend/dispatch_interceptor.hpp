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

#include <atomic>
#include <functional>
#include <mutex>
#include <shared_mutex>
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
     * @brief Install an interceptor into the process-global registry.
     *
     * Unlike push()/pop() (which affect only the calling thread's stack),
     * a globally-installed interceptor is consulted by run()/run_single()
     * on EVERY thread. This is required for cross-thread profiling/tracing:
     * a guard created on the main thread must observe dispatches issued by
     * DataLoader / autograd-engine worker threads too.
     *
     * Global interceptors run OUTSIDE the calling thread's thread-local
     * interceptors (i.e. they wrap them), so the existing per-thread chain
     * order is preserved. Thread-safe: writers take a unique lock, readers
     * (every dispatch) take a shared lock only when at least one global
     * interceptor is installed (atomic fast-path otherwise).
     */
    static void push_global(DispatchInterceptor interceptor) {
        std::unique_lock lock(global_mutex_());
        auto& g = global_();
        g.push_back(std::move(interceptor));
        global_count_().store(g.size(), std::memory_order_release);
    }

    /// Remove the most recently installed global interceptor (LIFO).
    static void pop_global() {
        std::unique_lock lock(global_mutex_());
        auto& g = global_();
        if (!g.empty()) g.pop_back();
        global_count_().store(g.size(), std::memory_order_release);
    }

    /// Number of process-global interceptors currently installed.
    static size_t global_depth() noexcept {
        return global_count_().load(std::memory_order_acquire);
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
        // Fast path: nothing installed on this thread AND no global
        // interceptor — single atomic load + empty() check, ~1ns.
        if (stack_().empty() &&
            global_count_().load(std::memory_order_acquire) == 0) [[likely]] {
            return terminal(op, inputs, attrs);
        }
        // Combine global (outer) + thread-local (inner) interceptors. The
        // combined vector outlives the fully-synchronous run_chain_ recursion.
        auto combined = active_interceptors_();
        if (combined.empty()) [[unlikely]] {
            // Race: the only global interceptor was removed after the check.
            return terminal(op, inputs, attrs);
        }
        return run_chain_(combined, 0, op, inputs, attrs, std::move(terminal));
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
        // Fast path: nothing installed on this thread AND no global interceptor.
        if (stack_().empty() &&
            global_count_().load(std::memory_order_acquire) == 0) [[likely]] {
            return terminal(op, inputs, attrs);
        }
        auto combined = active_interceptors_();
        if (combined.empty()) [[unlikely]] {
            // Race: the only global interceptor was removed after the check.
            return terminal(op, inputs, attrs);
        }
        // Wrap terminal as multi-output, run chain, extract element 0
        DispatchNext wrapped = [t = std::move(terminal)](
            OpId o, std::span<const Tensor> i, const OpAttributes& a) {
            return std::vector<Tensor>{t(o, i, a)};
        };
        auto result = run_chain_(combined, 0, op, inputs, attrs, std::move(wrapped));
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

    /// Process-global interceptor registry, consulted by every thread's
    /// dispatch. Like stack_(), these are defined out-of-line in a single
    /// translation unit (dispatch.cpp) so there is exactly ONE instance of
    /// each process-wide. global_mutex_ guards writes to global_(); readers
    /// take a shared lock (only when global_count_ > 0). global_count_ is an
    /// atomic mirror of global_().size() enabling a lock-free fast-path check.
    static std::shared_mutex& global_mutex_();
    static std::vector<DispatchInterceptor>& global_();
    static std::atomic<std::size_t>& global_count_();

    /// Snapshot of the active interceptor chain for this dispatch: a copy of
    /// the global interceptors (outer) followed by this thread's thread-local
    /// interceptors (inner). Copying isolates the chain from concurrent
    /// install/uninstall and keeps it valid for the whole run_chain_ recursion.
    static std::vector<DispatchInterceptor> active_interceptors_() {
        std::vector<DispatchInterceptor> combined;
        {
            std::shared_lock lock(global_mutex_());
            combined = global_();  // copy global interceptors (outermost)
        }
        auto& s = stack_();
        combined.insert(combined.end(), s.begin(), s.end());  // thread-local (inner)
        return combined;
    }

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

/**
 * @brief RAII guard that installs an interceptor into the process-global
 *        registry, so it fires on dispatches from ALL threads for its lifetime.
 *
 * Use this (instead of InterceptorGuard) when the interceptor must observe
 * work scheduled onto worker threads — e.g. profiling/tracing a forward pass
 * that fans out across DataLoader or autograd-engine workers. Installs are
 * LIFO, matching the per-thread guard's stack discipline.
 */
class GlobalInterceptorGuard {
public:
    explicit GlobalInterceptorGuard(DispatchInterceptor interceptor) {
        DispatchInterceptorStack::push_global(std::move(interceptor));
    }

    ~GlobalInterceptorGuard() {
        DispatchInterceptorStack::pop_global();
    }

    // Non-copyable, non-movable
    GlobalInterceptorGuard(const GlobalInterceptorGuard&) = delete;
    GlobalInterceptorGuard& operator=(const GlobalInterceptorGuard&) = delete;
};

} // namespace tenzor
