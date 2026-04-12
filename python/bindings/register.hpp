// Shared declarations for the python/bindings.cpp submodule split.
//
// The main `python/bindings.cpp` is the PYBIND11_MODULE entry point and
// remains responsible for the root-level tensor/device/autograd/nn
// bindings. Self-contained submodules (linalg, fft, vision/detection,
// etc.) are progressively extracted into separate TUs in this
// subdirectory; each new file declares a `void register_<name>(py::module_&)`
// function that is forward-declared below and called from bindings.cpp.
//
// This file should be included by bindings.cpp and by each extracted TU.
// Do not add implementation here — keep it a pure header.

#pragma once

#include <pybind11/pybind11.h>

namespace tenzor::python {

// Self-contained linalg submodule. Creates m.linalg and binds det/inv/
// solve/cholesky/norm/slogdet/svd/qr/eigh/eigvalsh/matrix_power plus the
// P2.1 additions (lstsq/pinv/matrix_exp).
void register_linalg(pybind11::module_& m);

// Self-contained fft submodule. Creates m.fft and binds fft/ifft/rfft/
// irfft/fft2/ifft2/fftn/ifftn plus the P2.7 additions
// (fftshift/ifftshift/hfft/ihfft).
void register_fft(pybind11::module_& m);

// Self-contained vision / detection / async_ops / fused submodules.
// Creates m.vision, m.detection, m.async_ops, m.fused with their ops.
void register_vision_detection(pybind11::module_& m);

} // namespace tenzor::python
