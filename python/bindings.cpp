#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <tenzor/tenzor.hpp>

namespace py = pybind11;

PYBIND11_MODULE(tenzor_core, m) {
    m.doc() = "Tenzor: High-performance tensor library";

    // Device
    py::class_<tenzor::Device>(m, "Device")
        .def(py::init<tenzor::Device::Type, int32_t>())
        .def_static("cpu", &tenzor::Device::cpu)
        .def_static("cuda", &tenzor::Device::cuda, py::arg("index") = 0)
        .def_readonly("type", &tenzor::Device::type)
        .def_readonly("index", &tenzor::Device::index)
        .def("__repr__", [](const tenzor::Device& d) {
            return d.to_string();
        });

    // DType enum
    py::enum_<tenzor::DType>(m, "dtype")
        .value("float32", tenzor::DType::Float32)
        .value("float64", tenzor::DType::Float64)
        .value("float16", tenzor::DType::Float16)
        .value("int32", tenzor::DType::Int32)
        .value("int64", tenzor::DType::Int64)
        .value("uint8", tenzor::DType::UInt8)
        .value("bool", tenzor::DType::Bool);

    // Tensor class
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
        .def_property_readonly("ndim", &tenzor::Tensor::ndim)
        .def_property_readonly("dtype", &tenzor::Tensor::dtype)
        .def_property_readonly("device", &tenzor::Tensor::device)
        .def("to", py::overload_cast<tenzor::Device>(&tenzor::Tensor::to, py::const_))
        .def("reshape", &tenzor::Tensor::reshape)
        .def("__add__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return a + b; })
        .def("__sub__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return a - b; })
        .def("__mul__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return a * b; })
        .def("__repr__", [](const tenzor::Tensor& t) {
            return "Tensor(shape=[...])";
        });

    // Operations
    m.def("zeros", &tenzor::zeros, "Create tensor filled with zeros",
         py::arg("shape"),
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("ones", &tenzor::ones, "Create tensor filled with ones",
         py::arg("shape"),
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("randn", &tenzor::randn, "Create tensor with random normal values",
         py::arg("shape"),
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("matmul", &tenzor::matmul, "Matrix multiplication");

    // Autograd
    py::class_<tenzor::Variable>(m, "Variable")
        .def(py::init<tenzor::Tensor, bool>(),
             py::arg("data"), py::arg("requires_grad") = false)
        .def("backward", &tenzor::Variable::backward, py::arg("gradient") = py::none())
        .def_property_readonly("data", py::overload_cast<>(&tenzor::Variable::tensor, py::const_))
        .def_property_readonly("grad", py::overload_cast<>(&tenzor::Variable::grad, py::const_));

    // Neural network
    auto nn = m.def_submodule("nn", "Neural network components");

    py::class_<tenzor::nn::Module, std::shared_ptr<tenzor::nn::Module>>(nn, "Module")
        .def("forward", &tenzor::nn::Module::forward)
        .def("__call__", &tenzor::nn::Module::operator())
        .def("parameters", &tenzor::nn::Module::parameters)
        .def("train", &tenzor::nn::Module::train, py::arg("mode") = true)
        .def("eval", &tenzor::nn::Module::eval)
        .def("cuda", &tenzor::nn::Module::cuda, py::arg("device_id") = 0)
        .def("cpu", &tenzor::nn::Module::cpu);

    py::class_<tenzor::nn::Linear, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Linear>>(nn, "Linear")
        .def(py::init<int64_t, int64_t, bool>(),
             py::arg("in_features"), py::arg("out_features"),
             py::arg("bias") = true);

    // Optimizers
    auto optim = m.def_submodule("optim", "Optimization algorithms");

    py::class_<tenzor::optim::SGD>(optim, "SGD")
        .def(py::init<std::vector<tenzor::Variable*>, double, double, double, double, bool>(),
             py::arg("params"), py::arg("lr"),
             py::arg("momentum") = 0.0, py::arg("dampening") = 0.0,
             py::arg("weight_decay") = 0.0, py::arg("nesterov") = false)
        .def("step", &tenzor::optim::SGD::step)
        .def("zero_grad", &tenzor::optim::SGD::zero_grad);

    py::class_<tenzor::optim::Adam>(optim, "Adam")
        .def(py::init<std::vector<tenzor::Variable*>, double, double, double, double, double, bool>(),
             py::arg("params"), py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             py::arg("amsgrad") = false)
        .def("step", &tenzor::optim::Adam::step)
        .def("zero_grad", &tenzor::optim::Adam::zero_grad);
}
