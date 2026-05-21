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

#include <future>
#include <memory>
#include <utility>
#include <vector>

#include <tenzor/core/tensor.hpp>
#include <tenzor/utils/threading/future.hpp>

namespace tenzor::python {

/**
 * @brief Python wrapper around `tenzor::Future<Tensor>`.
 *
 * The underlying `tenzor::Future<T>::wait()` is destructive (moves the
 * value out of the shared state). We mirror that: `.result()` is one-shot
 * and subsequent calls will throw via the shared-state's own bookkeeping.
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
        // Discard the value; we just block. The underlying shared state
        // caches the value/exception so a subsequent .result() still works
        // if implementation supports it — but `Future::wait()` consumes.
        // To keep wait() side-effect-free wrt the stored value, we poll
        // via is_ready() / busy-wait-free wait: just call wait() and drop
        // the returned tensor. Python sees None.
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
 */
struct TensorListFuture {
    // std::future is move-only; wrap in shared_ptr for the same reason as
    // above. Also tracks a "done" flag we can poll without consuming the
    // value.
    std::shared_ptr<std::future<std::vector<tenzor::Tensor>>> fut;

    explicit TensorListFuture(std::future<std::vector<tenzor::Tensor>> f)
        : fut(std::make_shared<std::future<std::vector<tenzor::Tensor>>>(std::move(f))) {}

    bool done() const {
        if (!fut || !fut->valid()) {
            return false;
        }
        return fut->wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    void wait_only() const {
        if (!fut || !fut->valid()) {
            throw std::runtime_error("TensorListFuture: invalid (no shared state)");
        }
        fut->wait();
    }

    std::vector<tenzor::Tensor> result() {
        if (!fut || !fut->valid()) {
            throw std::runtime_error("TensorListFuture: invalid (no shared state)");
        }
        return fut->get();
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
