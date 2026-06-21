// GIL-safe wrapper for a Python object captured into a C++ std::function.
//
// pybind11's `py::object` performs a non-atomic CPython Py_INCREF on copy and
// Py_DECREF on destruction. When such an object is captured into a C++
// callable (`std::function`) that the autograd engine or the RPC agent copies,
// moves, or destroys on a worker thread that does NOT hold the GIL, those
// refcount operations corrupt the interpreter (the documented
// "inc_ref()/dec_ref() called while the GIL is not held" abort).
//
// `GilSafePyObject` owns the underlying CPython object via a raw `PyObject*`
// guarded refcount: every operation that touches the refcount (copy-ctor,
// copy-assign, move-assign-with-drop, destructor) acquires the GIL first via
// `py::gil_scoped_acquire`. Move construction merely transfers the raw pointer
// (no refcount change) so it is safe without the GIL. Calling the object
// acquires the GIL and invokes it.
//
// This is the holder both `register_function` (RPC) and `Variable.register_hook`
// (autograd) use so the captured Python callable's lifetime is always
// manipulated under the GIL regardless of which C++ thread owns the closure.

#pragma once

#include <pybind11/pybind11.h>

#include <Python.h>

#include <utility>

namespace tenzor::python {

class GilSafePyObject {
public:
    GilSafePyObject() noexcept = default;

    // Takes a new reference to `obj`. Caller must hold the GIL (true at the
    // binding boundary where the object is first captured).
    explicit GilSafePyObject(pybind11::object obj) noexcept {
        // `release()` hands us the owned reference without touching refcount.
        ptr_ = obj.release().ptr();
    }

    GilSafePyObject(const GilSafePyObject& other) {
        if (other.ptr_ != nullptr) {
            pybind11::gil_scoped_acquire gil;
            Py_INCREF(other.ptr_);
            ptr_ = other.ptr_;
        }
    }

    GilSafePyObject(GilSafePyObject&& other) noexcept
        : ptr_(other.ptr_) {
        other.ptr_ = nullptr;  // pure pointer transfer, no refcount change
    }

    GilSafePyObject& operator=(const GilSafePyObject& other) {
        if (this == &other) {
            return *this;
        }
        // Compute the new reference (incref under GIL), then drop the old one
        // (decref under GIL). Do both under a single acquire.
        PyObject* old = nullptr;
        {
            pybind11::gil_scoped_acquire gil;
            if (other.ptr_ != nullptr) {
                Py_INCREF(other.ptr_);
            }
            old = ptr_;
            ptr_ = other.ptr_;
            if (old != nullptr) {
                Py_DECREF(old);
            }
        }
        return *this;
    }

    GilSafePyObject& operator=(GilSafePyObject&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        PyObject* old = ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = nullptr;
        if (old != nullptr) {
            // We held a reference that must be released under the GIL.
            pybind11::gil_scoped_acquire gil;
            Py_DECREF(old);
        }
        return *this;
    }

    ~GilSafePyObject() {
        if (ptr_ != nullptr) {
            // The interpreter may already be finalized at process teardown; in
            // that case touching the GIL is undefined. Guard with the
            // initialized check (matches the numpy keepalive deleter pattern).
            if (Py_IsInitialized()) {
                pybind11::gil_scoped_acquire gil;
                Py_DECREF(ptr_);
            }
            ptr_ = nullptr;
        }
    }

    // Borrow the underlying object. Caller MUST hold the GIL.
    pybind11::handle handle() const noexcept {
        return pybind11::handle(ptr_);
    }

    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    PyObject* ptr_ = nullptr;
};

} // namespace tenzor::python
