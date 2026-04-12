// tenzor.linalg Python bindings. Extracted from python/bindings.cpp as
// part of P3.4 (incremental split of the 10k-line binding monolith).
//
// Besides the existing bindings, this file adds the P2.1 routines:
// lstsq, pinv, matrix_exp. Those existed in C++ but were never exposed
// to Python before the split.

#include "register.hpp"

#include <pybind11/stl.h>

#include <tenzor/ops/linalg.hpp>
#include <tenzor/core/tensor.hpp>

namespace py = pybind11;

namespace tenzor::python {

void register_linalg(py::module_& m) {
    auto linalg_mod = m.def_submodule("linalg", "Linear algebra operations");

    linalg_mod.def("det", &tenzor::linalg::det, "Compute matrix determinant",
                   py::arg("A"),
                   py::call_guard<py::gil_scoped_release>());
    linalg_mod.def("inv", &tenzor::linalg::inv, "Compute matrix inverse",
                   py::arg("A"),
                   py::call_guard<py::gil_scoped_release>());
    linalg_mod.def("solve", &tenzor::linalg::solve, "Solve linear system AX = B",
                   py::arg("A"), py::arg("B"),
                   py::call_guard<py::gil_scoped_release>());
    linalg_mod.def("cholesky", &tenzor::linalg::cholesky, "Cholesky decomposition",
                   py::arg("A"), py::arg("upper") = false,
                   py::call_guard<py::gil_scoped_release>());
    linalg_mod.def("norm", &tenzor::linalg::norm, "Matrix norm",
                   py::arg("A"), py::arg("ord") = "fro",
                   py::call_guard<py::gil_scoped_release>());

    // NOTE on GIL handling for multi-output linalg ops: pybind11's
    // `py::call_guard<py::gil_scoped_release>()` wraps the ENTIRE lambda,
    // so `py::make_tuple(...)` inside the body runs without the GIL and
    // crashes on any reference count manipulation. We therefore release
    // the GIL manually only around the C++ computation, then re-acquire
    // it implicitly when the scoped_release destructor runs before
    // `py::make_tuple`. The single-output bindings above are fine with
    // call_guard because their return value is a plain Tensor that
    // pybind11 converts after the lambda returns.
    linalg_mod.def("slogdet", [](const tenzor::Tensor& A) {
        std::tuple<tenzor::Tensor, tenzor::Tensor> result;
        {
            py::gil_scoped_release release;
            result = tenzor::linalg::slogdet(A);
        }
        auto [sign, logabsdet] = std::move(result);
        return py::make_tuple(sign, logabsdet);
    }, "Sign and log of absolute determinant", py::arg("A"));

    linalg_mod.def("svd", [](const tenzor::Tensor& A, bool full_matrices) {
        std::tuple<tenzor::Tensor, tenzor::Tensor, tenzor::Tensor> result;
        {
            py::gil_scoped_release release;
            result = tenzor::linalg::svd(A, full_matrices);
        }
        auto [U, S, Vh] = std::move(result);
        return py::make_tuple(U, S, Vh);
    }, "Singular Value Decomposition",
       py::arg("A"), py::arg("full_matrices") = true);

    linalg_mod.def("qr", [](const tenzor::Tensor& A) {
        std::tuple<tenzor::Tensor, tenzor::Tensor> result;
        {
            py::gil_scoped_release release;
            result = tenzor::linalg::qr(A);
        }
        auto [Q, R] = std::move(result);
        return py::make_tuple(Q, R);
    }, "QR decomposition", py::arg("A"));

    linalg_mod.def("eigh", [](const tenzor::Tensor& A) {
        std::tuple<tenzor::Tensor, tenzor::Tensor> result;
        {
            py::gil_scoped_release release;
            result = tenzor::linalg::eigh(A);
        }
        auto [eigenvalues, eigenvectors] = std::move(result);
        return py::make_tuple(eigenvalues, eigenvectors);
    }, "Eigendecomposition of symmetric matrix", py::arg("A"));

    linalg_mod.def("eigvalsh", &tenzor::linalg::eigvalsh,
                   "Eigenvalues of symmetric matrix", py::arg("A"),
                   py::call_guard<py::gil_scoped_release>());

    linalg_mod.def("matrix_power", &tenzor::linalg::matrix_power,
                   "Matrix power via binary exponentiation",
                   py::arg("A"), py::arg("n"),
                   py::call_guard<py::gil_scoped_release>());

    // P2.1 additions — previously C++-only.
    linalg_mod.def("lstsq", [](const tenzor::Tensor& A, const tenzor::Tensor& B) {
        std::tuple<tenzor::Tensor, tenzor::Tensor> result;
        {
            py::gil_scoped_release release;
            result = tenzor::linalg::lstsq(A, B);
        }
        auto [solution, residuals] = std::move(result);
        return py::make_tuple(solution, residuals);
    }, "Least-squares solution of A @ X = B via LAPACK ?gels",
       py::arg("A"), py::arg("B"));

    linalg_mod.def("pinv", &tenzor::linalg::pinv,
                   "Moore-Penrose pseudoinverse of A via SVD",
                   py::arg("A"), py::arg("rcond") = 1e-15,
                   py::call_guard<py::gil_scoped_release>());

    linalg_mod.def("matrix_exp", &tenzor::linalg::matrix_exp,
                   "Matrix exponential via Padé-13 scaling-and-squaring",
                   py::arg("A"),
                   py::call_guard<py::gil_scoped_release>());
}

} // namespace tenzor::python
