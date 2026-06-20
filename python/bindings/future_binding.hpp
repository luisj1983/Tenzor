// Future<T> pybind11 wrappers for async ops and RPC.
//
// Audit item C.6: previously `async_matmul` / `async_add` / `rpc_async`
// Python bindings called `.wait()` / `.get()` and returned the resolved
// Tensor, defeating the purpose of an async API. This header exposes two
// thin wrappers around the underlying future types so Python callers can
// overlap compute:
//
//   - TensorFuture     wraps  tenzor::Future<Tensor>
//   - TensorListFuture wraps  std::future<std::vector<Tensor>>
//
// Both expose torch.futures.Future-like semantics:
//   .done()   non-blocking poll
//   .wait()   block until resolved (returns None)
//   .result() block and return the Tensor / Tensor list, re-throwing any
//             exception captured by the underlying future.
//
// Registration is centralised in `register_future_types()`, which is
// invoked exactly once from `bindings.cpp::PYBIND11_MODULE` *before* any
// submodule that produces futures (async_ops, distributed.rpc) is
// registered. Submodule TUs include this header purely to use the
// wrapper structs as return values.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <tenzor/core/tensor.hpp>
#include <tenzor/utils/threading/future.hpp>

namespace tenzor::python {

/**
 * @brief Python wrapper around `tenzor::Future<Tensor>`.
 *
 * The underlying `tenzor::Future<T>::wait()` is non-destructive: its shared
 * state's `get()` returns a *copy* of the stored value (see
 * `SharedState::get()` in future.hpp) and never invalidates `ready_`. As a
 * result `.result()` and `.wait()` are repeatable, and `.done()` stays True
 * once the operation has completed.
 */
struct TensorFuture {
    // shared_ptr because pybind11 needs to copy/hold the handle while
    // the Python object is alive; `tenzor::Future` itself is movable but
    // the shared state inside it is reference-counted, so wrapping in
    // shared_ptr is safe and cheap.
    std::shared_ptr<tenzor::Future<tenzor::Tensor>> fut;

    explicit TensorFuture(tenzor::Future<tenzor::Tensor> f)
        : fut(std::make_shared<tenzor::Future<tenzor::Tensor>>(std::move(f))) {}

    bool done() const {
        return fut && fut->is_ready();
    }

    void wait_only() {
        if (!fut) {
            throw std::runtime_error("TensorFuture: invalid (no shared state)");
        }
        // Block until ready, then discard the value (Python sees None).
        // `Future::wait()` is non-destructive — it returns a copy of the
        // stored value and leaves the shared state intact — so a later
        // .result() / .wait() still works and .done() stays True.
        (void)fut->wait();
    }

    tenzor::Tensor result() {
        if (!fut) {
            throw std::runtime_error("TensorFuture: invalid (no shared state)");
        }
        return fut->wait();
    }
};

/**
 * @brief Python wrapper around `std::future<std::vector<Tensor>>`
 *        (used by `tenzor::distributed::rpc::rpc_async`).
 *
 * `std::future::get()` is one-shot and destructive: it consumes the shared
 * state, after which `valid()` is false. That would make `.done()` flip back
 * to False after a `.result()` and a second `.result()` throw, diverging from
 * TensorFuture (whose `tenzor::Future<T>::wait()` is repeatable). To honour
 * the documented "same contract as TensorFuture", we resolve the std::future
 * exactly once and cache the value (or the captured exception). All subsequent
 * `.done()` / `.wait()` / `.result()` calls are served from the cache, so:
 *   - `.done()` stays True once the RPC has completed,
 *   - `.result()` is repeatable and returns a copy each time,
 *   - any exception raised by the RPC is re-raised on every `.result()`.
 *
 * A mutex serialises access to the underlying std::future and the cache.
 * `std::future` provides no thread-safety for concurrent member access on a
 * single object, and `.result()`/`.wait()` run with the GIL released, so the
 * mutex is required to make concurrent `.done()` / `.result()` / `.wait()`
 * from multiple Python threads well-defined.
 */
struct TensorListFuture {
    struct State {
        std::mutex mutex;
        std::future<std::vector<tenzor::Tensor>> fut;
        std::optional<std::vector<tenzor::Tensor>> value;
        std::exception_ptr exception;
        std::atomic<bool> resolved{false};

        explicit State(std::future<std::vector<tenzor::Tensor>> f)
            : fut(std::move(f)) {}

        // Resolve the std::future exactly once, caching the value or the
        // captured exception. Must be called with `mutex` held. Returns once
        // `resolved` is true.
        void resolve_locked() {
            if (resolved.load(std::memory_order_acquire)) {
                return;
            }
            if (!fut.valid()) {
                // Should not happen: we only ever consume the future here, and
                // only when not yet resolved.
                exception = std::make_exception_ptr(std::runtime_error(
                    "TensorListFuture: invalid (no shared state)"));
                resolved.store(true, std::memory_order_release);
                return;
            }
            try {
                value = fut.get();
            } catch (...) {
                exception = std::current_exception();
            }
            resolved.store(true, std::memory_order_release);
        }
    };

    std::shared_ptr<State> state;

    explicit TensorListFuture(std::future<std::vector<tenzor::Tensor>> f)
        : state(std::make_shared<State>(std::move(f))) {}

    bool done() const {
        if (!state) {
            return false;
        }
        // Fast path: already resolved (and cached) -> always True.
        if (state->resolved.load(std::memory_order_acquire)) {
            return true;
        }
        // Non-blocking poll of the still-pending std::future. Guard with the
        // mutex because a concurrent .result()/.wait() may be calling get()
        // on the same object with the GIL released.
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->resolved.load(std::memory_order_relaxed)) {
            return true;
        }
        if (!state->fut.valid()) {
            return false;
        }
        return state->fut.wait_for(std::chrono::seconds(0)) ==
               std::future_status::ready;
    }

    void wait_only() const {
        if (!state) {
            throw std::runtime_error("TensorListFuture: invalid (no shared state)");
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        state->resolve_locked();
        // Mirror .result() error semantics: re-raise any captured exception
        // so wait() surfaces RPC failures rather than silently returning.
        if (state->exception) {
            std::rethrow_exception(state->exception);
        }
    }

    std::vector<tenzor::Tensor> result() {
        if (!state) {
            throw std::runtime_error("TensorListFuture: invalid (no shared state)");
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        state->resolve_locked();
        if (state->exception) {
            std::rethrow_exception(state->exception);
        }
        // Return a copy so .result() stays repeatable, matching TensorFuture.
        return *state->value;
    }
};

/**
 * @brief Register Future wrappers on the root module.
 *
 * Called exactly once from `PYBIND11_MODULE(tenzor_core, m)` before any
 * submodule that returns a future. Idempotent in the sense that pybind11
 * itself will diagnose double-registration, so callers must ensure single
 * invocation.
 */
inline void register_future_types(pybind11::module_& m) {
    namespace py = pybind11;

    py::class_<TensorFuture>(m, "TensorFuture",
        "Future-like handle to an asynchronous Tensor result.\n\n"
        "Returned by tenzor.async_ops.async_* operations. Mirrors\n"
        "`torch.futures.Future` semantics: .done() to poll, .wait() to\n"
        "block without returning the value, .result() to block and\n"
        "retrieve the Tensor (re-raises any captured exception).")
        .def("done", &TensorFuture::done,
             "Return True if the asynchronous operation has completed.")
        .def("wait", &TensorFuture::wait_only,
             py::call_guard<py::gil_scoped_release>(),
             "Block until the asynchronous operation completes. Returns None.")
        .def("result", &TensorFuture::result,
             py::call_guard<py::gil_scoped_release>(),
             "Block until the operation completes and return the Tensor.\n"
             "Re-raises any exception thrown by the underlying op.");

    py::class_<TensorListFuture>(m, "TensorListFuture",
        "Future-like handle to an asynchronous List[Tensor] result.\n\n"
        "Returned by tenzor.distributed.rpc.rpc_async. See TensorFuture\n"
        "for the .done()/.wait()/.result() contract.")
        .def("done", &TensorListFuture::done,
             "Return True if the asynchronous RPC has completed.")
        .def("wait", &TensorListFuture::wait_only,
             py::call_guard<py::gil_scoped_release>(),
             "Block until the asynchronous RPC completes. Returns None.")
        .def("result", &TensorListFuture::result,
             py::call_guard<py::gil_scoped_release>(),
             "Block until the RPC completes and return the result tensors.");
}

} // namespace tenzor::python
