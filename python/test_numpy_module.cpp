#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <tenzor/tenzor.hpp>
#include "numpy_interop.hpp"

namespace py = pybind11;

PYBIND11_MODULE(test_numpy_core, m) {
    m.doc() = "Test module for NumPy interop";

    // Initialize Tenzor
    m.def("initialize", &tenzor::initialize, "Initialize the Tenzor library");

    // Device
    py::class_<tenzor::Device>(m, "Device")
        .def_static("cpu", &tenzor::Device::cpu)
        .def_static("cuda", &tenzor::Device::cuda, py::arg("index") = 0)
        .def_readonly("type", &tenzor::Device::type);

    // DType enum
    py::enum_<tenzor::DType>(m, "dtype")
        .value("float32", tenzor::DType::Float32)
        .value("float64", tenzor::DType::Float64)
        .value("int32", tenzor::DType::Int32)
        .value("int64", tenzor::DType::Int64)
        .value("uint8", tenzor::DType::UInt8)
        .value("bool", tenzor::DType::Bool);

    // Tensor class (minimal)
    py::class_<tenzor::Tensor>(m, "Tensor")
        .def(py::init<std::vector<int64_t>, tenzor::DType, tenzor::Device>(),
             py::arg("shape"),
             py::arg("dtype") = tenzor::DType::Float32,
             py::arg("device") = tenzor::Device::cpu())
        .def_property_readonly("shape",
            [](const tenzor::Tensor& t) {
                auto s = t.shape();
                return std::vector<int64_t>(s.begin(), s.end());
            })
        .def_property_readonly("dtype", &tenzor::Tensor::dtype)
        .def_property_readonly("device", &tenzor::Tensor::device)
        .def_property_readonly("numel", &tenzor::Tensor::numel)
        .def_property_readonly("is_contiguous", &tenzor::Tensor::is_contiguous)
        // NumPy interoperability
        .def("numpy", &tenzor::numpy::tensor_to_numpy,
             "Convert tensor to NumPy array")
        .def_static("from_numpy", &tenzor::numpy::numpy_to_tensor,
             py::arg("array"), py::arg("device") = tenzor::Device::cpu(),
             "Create tensor from NumPy array")
        .def("cpu", &tenzor::Tensor::cpu)
        .def("cuda", &tenzor::Tensor::cuda, py::arg("device_id") = 0);

    // Creation operations
    m.def("zeros", &tenzor::zeros, "Create tensor filled with zeros",
         py::arg("shape"),
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("ones", &tenzor::ones, "Create tensor filled with ones",
         py::arg("shape"),
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());
}
