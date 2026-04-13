// tenzor.fft Python bindings. Extracted from python/bindings.cpp as
// part of P3.4.
//
// Besides the existing fft/ifft/rfft/irfft/fft2/ifft2/fftn/ifftn, this
// file adds the P2.7 routines: fftshift, ifftshift, hfft, ihfft. Those
// existed in C++ but were never exposed to Python before.

#include "register.hpp"

#include <pybind11/stl.h>

#include <tenzor/ops/fft.hpp>
#include <tenzor/core/tensor.hpp>

namespace py = pybind11;

namespace tenzor::python {

void register_fft(py::module_& m) {
    auto fft_mod = m.def_submodule("fft", "Fast Fourier Transform operations");

    fft_mod.def("fft", [](const tenzor::Tensor& input, std::optional<int64_t> n,
                           int64_t dim, const std::string& norm) {
        return tenzor::fft::fft(input, n, dim, norm);
    }, "1-D complex-to-complex FFT",
       py::arg("input"), py::arg("n") = py::none(), py::arg("dim") = -1,
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("ifft", [](const tenzor::Tensor& input, std::optional<int64_t> n,
                            int64_t dim, const std::string& norm) {
        return tenzor::fft::ifft(input, n, dim, norm);
    }, "1-D inverse complex-to-complex FFT",
       py::arg("input"), py::arg("n") = py::none(), py::arg("dim") = -1,
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("rfft", [](const tenzor::Tensor& input, std::optional<int64_t> n,
                            int64_t dim, const std::string& norm) {
        return tenzor::fft::rfft(input, n, dim, norm);
    }, "1-D real-to-complex FFT",
       py::arg("input"), py::arg("n") = py::none(), py::arg("dim") = -1,
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("irfft", [](const tenzor::Tensor& input, std::optional<int64_t> n,
                             int64_t dim, const std::string& norm) {
        return tenzor::fft::irfft(input, n, dim, norm);
    }, "1-D complex-to-real inverse FFT",
       py::arg("input"), py::arg("n") = py::none(), py::arg("dim") = -1,
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("fft2", [](const tenzor::Tensor& input,
                            std::optional<std::vector<int64_t>> s,
                            std::vector<int64_t> dim, const std::string& norm) {
        return tenzor::fft::fft2(input, s, dim, norm);
    }, "2-D complex-to-complex FFT",
       py::arg("input"), py::arg("s") = py::none(),
       py::arg("dim") = std::vector<int64_t>{-2, -1},
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("ifft2", [](const tenzor::Tensor& input,
                             std::optional<std::vector<int64_t>> s,
                             std::vector<int64_t> dim, const std::string& norm) {
        return tenzor::fft::ifft2(input, s, dim, norm);
    }, "2-D inverse complex-to-complex FFT",
       py::arg("input"), py::arg("s") = py::none(),
       py::arg("dim") = std::vector<int64_t>{-2, -1},
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("fftn", [](const tenzor::Tensor& input,
                            std::optional<std::vector<int64_t>> s,
                            std::optional<std::vector<int64_t>> dim, const std::string& norm) {
        return tenzor::fft::fftn(input, s, dim, norm);
    }, "N-D complex-to-complex FFT",
       py::arg("input"), py::arg("s") = py::none(), py::arg("dim") = py::none(),
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("ifftn", [](const tenzor::Tensor& input,
                             std::optional<std::vector<int64_t>> s,
                             std::optional<std::vector<int64_t>> dim, const std::string& norm) {
        return tenzor::fft::ifftn(input, s, dim, norm);
    }, "N-D inverse complex-to-complex FFT",
       py::arg("input"), py::arg("s") = py::none(), py::arg("dim") = py::none(),
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    // P2.7 additions — previously C++-only.
    fft_mod.def("fftshift", &tenzor::fft::fftshift,
                "Shift the zero-frequency component to the center of the spectrum",
                py::arg("input"), py::arg("dims") = std::vector<int64_t>{},
                py::call_guard<py::gil_scoped_release>());

    fft_mod.def("ifftshift", &tenzor::fft::ifftshift,
                "Inverse of fftshift — undoes the circular shift",
                py::arg("input"), py::arg("dims") = std::vector<int64_t>{},
                py::call_guard<py::gil_scoped_release>());

    fft_mod.def("hfft", [](const tenzor::Tensor& input, std::optional<int64_t> n,
                            int64_t dim, const std::string& norm) {
        return tenzor::fft::hfft(input, n, dim, norm);
    }, "Hermitian FFT: real output from Hermitian-symmetric complex input",
       py::arg("input"), py::arg("n") = py::none(), py::arg("dim") = -1,
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("ihfft", [](const tenzor::Tensor& input, std::optional<int64_t> n,
                             int64_t dim, const std::string& norm) {
        return tenzor::fft::ihfft(input, n, dim, norm);
    }, "Inverse Hermitian FFT: Hermitian-symmetric output from real input",
       py::arg("input"), py::arg("n") = py::none(), py::arg("dim") = -1,
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("rfft2", [](const tenzor::Tensor& input,
                             std::optional<std::vector<int64_t>> s,
                             std::vector<int64_t> dim, const std::string& norm) {
        return tenzor::fft::rfft2(input, s, dim, norm);
    }, "2-D real-to-complex FFT",
       py::arg("input"), py::arg("s") = py::none(),
       py::arg("dim") = std::vector<int64_t>{-2, -1}, py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("irfft2", [](const tenzor::Tensor& input,
                              std::optional<std::vector<int64_t>> s,
                              std::vector<int64_t> dim, const std::string& norm) {
        return tenzor::fft::irfft2(input, s, dim, norm);
    }, "2-D complex-to-real inverse FFT",
       py::arg("input"), py::arg("s") = py::none(),
       py::arg("dim") = std::vector<int64_t>{-2, -1}, py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("rfftn", [](const tenzor::Tensor& input,
                             std::optional<std::vector<int64_t>> s,
                             std::optional<std::vector<int64_t>> dim, const std::string& norm) {
        return tenzor::fft::rfftn(input, s, dim, norm);
    }, "N-D real-to-complex FFT",
       py::arg("input"), py::arg("s") = py::none(), py::arg("dim") = py::none(),
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("irfftn", [](const tenzor::Tensor& input,
                              std::optional<std::vector<int64_t>> s,
                              std::optional<std::vector<int64_t>> dim, const std::string& norm) {
        return tenzor::fft::irfftn(input, s, dim, norm);
    }, "N-D complex-to-real inverse FFT",
       py::arg("input"), py::arg("s") = py::none(), py::arg("dim") = py::none(),
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());
}

} // namespace tenzor::python
