#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/indexing.hpp>
#include <tenzor/nn/optim/scheduler.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include <tenzor/nn/layers/attention.hpp>
#include <tenzor/nn/layers/transformer.hpp>
#include <tenzor/nn/layers/embedding.hpp>
#include <tenzor/nn/optim/rmsprop.hpp>
#include <tenzor/nn/optim/adagrad.hpp>
#include <tenzor/nn/optim/adadelta.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include "numpy_interop.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tenzor_core, m) {
    m.doc() = "Tenzor: High-performance tensor library";

    // Library initialization
    m.def("initialize", &tenzor::initialize,
          "Initialize the Tenzor library (registers backends and operations)");

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
        .value("bfloat16", tenzor::DType::BFloat16)
        .value("int8", tenzor::DType::Int8)
        .value("int16", tenzor::DType::Int16)
        .value("int32", tenzor::DType::Int32)
        .value("int64", tenzor::DType::Int64)
        .value("uint8", tenzor::DType::UInt8)
        .value("uint16", tenzor::DType::UInt16)
        .value("uint32", tenzor::DType::UInt32)
        .value("uint64", tenzor::DType::UInt64)
        .value("bool", tenzor::DType::Bool)
        .value("complex64", tenzor::DType::Complex64)
        .value("complex128", tenzor::DType::Complex128);

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
        .def_property_readonly("numel", &tenzor::Tensor::numel)
        .def_property_readonly("is_contiguous", &tenzor::Tensor::is_contiguous)
        .def("to", py::overload_cast<tenzor::Device>(&tenzor::Tensor::to, py::const_))
        .def("reshape", &tenzor::Tensor::reshape)
        // Shape manipulation
        .def("transpose", &tenzor::Tensor::transpose,
             py::arg("dim0"), py::arg("dim1"))
        .def("permute", &tenzor::Tensor::permute,
             py::arg("dims"))
        .def("squeeze", &tenzor::Tensor::squeeze,
             py::arg("dim") = py::none())
        .def("unsqueeze", &tenzor::Tensor::unsqueeze,
             py::arg("dim"))
        .def("flatten", &tenzor::Tensor::flatten,
             py::arg("start_dim") = 0, py::arg("end_dim") = -1)
        .def("view", &tenzor::Tensor::view,
             py::arg("shape"))
        // Memory operations
        .def("clone", &tenzor::Tensor::clone)
        .def("detach", &tenzor::Tensor::detach)
        .def("contiguous", &tenzor::Tensor::contiguous)
        // NumPy interoperability
        .def("numpy", &tenzor::numpy::tensor_to_numpy,
             "Convert tensor to NumPy array (zero-copy when possible)")
        .def_static("from_numpy", &tenzor::numpy::numpy_to_tensor,
             py::arg("array"), py::arg("device") = tenzor::Device::cpu(),
             "Create tensor from NumPy array (zero-copy when possible)")
        // Scalar extraction
        .def("item", [](const tenzor::Tensor& t) -> py::object {
            if (t.numel() != 1) {
                throw std::runtime_error("item() only works for scalar tensors");
            }
            switch (t.dtype()) {
                case tenzor::DType::Float32:
                    return py::cast(t.item<float>());
                case tenzor::DType::Float64:
                    return py::cast(t.item<double>());
                case tenzor::DType::Int8:
                    return py::cast(t.item<int8_t>());
                case tenzor::DType::Int16:
                    return py::cast(t.item<int16_t>());
                case tenzor::DType::Int32:
                    return py::cast(t.item<int32_t>());
                case tenzor::DType::Int64:
                    return py::cast(t.item<int64_t>());
                case tenzor::DType::UInt8:
                    return py::cast(t.item<uint8_t>());
                case tenzor::DType::UInt16:
                    return py::cast(t.item<uint16_t>());
                case tenzor::DType::UInt32:
                    return py::cast(t.item<uint32_t>());
                case tenzor::DType::UInt64:
                    return py::cast(t.item<uint64_t>());
                case tenzor::DType::Bool:
                    return py::cast(t.item<bool>());
                case tenzor::DType::Complex64:
                    return py::cast(t.item<std::complex<float>>());
                case tenzor::DType::Complex128:
                    return py::cast(t.item<std::complex<double>>());
                case tenzor::DType::Float16:
                case tenzor::DType::BFloat16:
                    throw std::runtime_error("Float16 and BFloat16 dtypes not yet supported for item()");
                default:
                    throw std::runtime_error("Unsupported dtype for item()");
            }
        }, "Extract scalar value from single-element tensor")
        // Arithmetic operators
        .def("__add__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return a + b; })
        .def("__sub__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return a - b; })
        .def("__mul__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return a * b; })
        .def("__repr__", [](const tenzor::Tensor& t) {
            return "Tensor(shape=[...])";
        })
        // Python-style indexing
        .def("__getitem__", [](const tenzor::Tensor& self, py::object key) -> tenzor::Tensor {
            // Handle integer indexing
            if (py::isinstance<py::int_>(key)) {
                int64_t idx = py::cast<int64_t>(key);
                auto shape = self.shape();
                if (shape.empty()) {
                    throw std::runtime_error("Cannot index scalar tensor");
                }
                // Handle negative indexing
                if (idx < 0) {
                    idx += shape[0];
                }
                if (idx < 0 || idx >= shape[0]) {
                    throw std::out_of_range("Index out of range");
                }
                // Return slice along first dimension (squeeze will remove dim if size is 1)
                auto sliced = self.slice(0, idx, idx + 1);
                // Only squeeze if the dimension actually has size 1
                auto sliced_shape = sliced.shape();
                if (!sliced_shape.empty() && sliced_shape[0] == 1) {
                    return sliced.squeeze(0);
                }
                return sliced;
            }
            // Handle slice objects (basic implementation)
            else if (py::isinstance<py::slice>(key)) {
                py::slice slice_obj = py::cast<py::slice>(key);
                py::ssize_t start, stop, step, length;
                auto shape = self.shape();
                if (shape.empty()) {
                    throw std::runtime_error("Cannot slice scalar tensor");
                }
                if (!slice_obj.compute(shape[0], &start, &stop, &step, &length)) {
                    throw std::runtime_error("Invalid slice");
                }
                if (step != 1) {
                    throw std::runtime_error("Slice step not supported yet");
                }
                return self.slice(0, start, stop);
            }
            // Handle tuple of indices/slices (basic implementation)
            else if (py::isinstance<py::tuple>(key)) {
                py::tuple indices = py::cast<py::tuple>(key);
                tenzor::Tensor result = self;
                int squeeze_count = 0;  // Track dimensions that need squeezing
                for (size_t i = 0; i < indices.size(); ++i) {
                    if (py::isinstance<py::int_>(indices[i])) {
                        int64_t idx = py::cast<int64_t>(indices[i]);
                        auto shape = result.shape();
                        size_t dim = i - squeeze_count;  // Adjust for squeezed dimensions
                        if (dim >= shape.size()) {
                            throw std::out_of_range("Too many indices");
                        }
                        if (idx < 0) {
                            idx += shape[dim];
                        }
                        result = result.slice(dim, idx, idx + 1);
                        // Check if we can squeeze this dimension
                        auto new_shape = result.shape();
                        if (dim < new_shape.size() && new_shape[dim] == 1) {
                            result = result.squeeze(dim);
                            squeeze_count++;
                        }
                    } else if (py::isinstance<py::slice>(indices[i])) {
                        py::slice slice_obj = py::cast<py::slice>(indices[i]);
                        py::ssize_t start, stop, step, length;
                        auto shape = result.shape();
                        size_t dim = i - squeeze_count;  // Adjust for squeezed dimensions
                        if (dim >= shape.size()) {
                            throw std::out_of_range("Too many indices");
                        }
                        if (!slice_obj.compute(shape[dim], &start, &stop, &step, &length)) {
                            throw std::runtime_error("Invalid slice");
                        }
                        if (step != 1) {
                            throw std::runtime_error("Slice step not supported yet");
                        }
                        result = result.slice(dim, start, stop);
                    }
                }
                return result;
            }
            throw std::runtime_error("Unsupported index type");
        }, py::arg("key"), "Get tensor slice or element")
        .def("__setitem__", [](tenzor::Tensor& self, py::object key, py::object value) {
            // Basic implementation - can be extended for more complex indexing
            throw std::runtime_error("__setitem__ not fully implemented yet. Use tensor operations instead.");
        }, py::arg("key"), py::arg("value"), "Set tensor slice or element");

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

    m.def("matmul", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
         return tenzor::matmul(a, b);
         }, "Matrix multiplication",
         py::arg("a"), py::arg("b"));

    // Math operations - using lambda wrappers for overloaded functions
    m.def("exp", [](const tenzor::Tensor& t) { return tenzor::exp(t); },
         "Element-wise exponential");
    m.def("log", [](const tenzor::Tensor& t) { return tenzor::log(t); },
         "Element-wise natural logarithm");
    m.def("sqrt", [](const tenzor::Tensor& t) { return tenzor::sqrt(t); },
         "Element-wise square root");
    m.def("abs", [](const tenzor::Tensor& t) { return tenzor::abs(t); },
         "Element-wise absolute value");
    m.def("pow", [](const tenzor::Tensor& input, double exponent) {
         return tenzor::pow(input, exponent);
         }, "Element-wise power",
         py::arg("input"), py::arg("exponent"));
    m.def("sin", [](const tenzor::Tensor& t) { return tenzor::sin(t); },
         "Element-wise sine");
    m.def("cos", [](const tenzor::Tensor& t) { return tenzor::cos(t); },
         "Element-wise cosine");
    m.def("tanh", [](const tenzor::Tensor& t) { return tenzor::tanh(t); },
         "Element-wise hyperbolic tangent");

    // Reduction operations - using lambda wrappers for overloaded functions
    m.def("sum", [](const tenzor::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::sum(input, dim, keepdim);
         }, "Sum reduction",
         py::arg("input"),
         py::arg("dim") = py::none(),
         py::arg("keepdim") = false);
    m.def("mean", [](const tenzor::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::mean(input, dim, keepdim);
         }, "Mean reduction",
         py::arg("input"),
         py::arg("dim") = py::none(),
         py::arg("keepdim") = false);
    m.def("max", [](const tenzor::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::max(input, dim, keepdim);
         }, "Max reduction",
         py::arg("input"),
         py::arg("dim") = py::none(),
         py::arg("keepdim") = false);
    m.def("min", [](const tenzor::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::min(input, dim, keepdim);
         }, "Min reduction",
         py::arg("input"),
         py::arg("dim") = py::none(),
         py::arg("keepdim") = false);

    // Transform operations
    m.def("transpose", &tenzor::transpose, "Transpose two dimensions",
         py::arg("input"), py::arg("dim0"), py::arg("dim1"));
    m.def("permute", [](const tenzor::Tensor& input, std::vector<int64_t> dims) {
         return tenzor::permute(input, dims);
         }, "Permute dimensions",
         py::arg("input"), py::arg("dims"));
    m.def("squeeze", &tenzor::squeeze, "Remove dimensions of size 1",
         py::arg("input"), py::arg("dim") = py::none());
    m.def("unsqueeze", &tenzor::unsqueeze, "Add dimension of size 1",
         py::arg("input"), py::arg("dim"));
    m.def("flatten", &tenzor::flatten, "Flatten tensor",
         py::arg("input"),
         py::arg("start_dim") = 0,
         py::arg("end_dim") = -1);
    m.def("contiguous", &tenzor::contiguous, "Make tensor contiguous");

    // Indexing operations (only bind implemented functions)
    m.def("slice", &tenzor::slice, "Slice tensor along dimension",
         py::arg("input"), py::arg("dim"), py::arg("start"), py::arg("end"),
         py::arg("step") = 1);
    m.def("index_select", &tenzor::index_select, "Select indices along dimension",
         py::arg("input"), py::arg("dim"), py::arg("index"));

    // Note: The following operations are declared in headers but not yet implemented:
    // gather, scatter, masked_select, masked_fill, where, take, put
    // They can be added once the implementations are complete

    // Autograd
    py::class_<tenzor::Variable>(m, "Variable")
        .def(py::init<tenzor::Tensor, bool>(),
             py::arg("data"), py::arg("requires_grad") = false)
        .def("backward", &tenzor::Variable::backward, py::arg("gradient") = py::none())
        .def_property_readonly("data", py::overload_cast<>(&tenzor::Variable::tensor, py::const_))
        .def_property_readonly("grad", py::overload_cast<>(&tenzor::Variable::grad, py::const_));

    // NoGradGuard for RAII-style gradient control
    py::class_<tenzor::NoGradGuard>(m, "NoGradGuard")
        .def(py::init<>(),
             "Context manager for disabling gradient computation");

    // Gradient control functions
    m.def("is_grad_enabled", &tenzor::is_grad_enabled,
          "Check if gradient computation is globally enabled");

    m.def("set_grad_enabled", &tenzor::set_grad_enabled,
          py::arg("enabled"),
          "Set global gradient computation state");

    // Python context manager for no_grad
    m.def("no_grad", []() {
        return tenzor::NoGradGuard();
    }, "Context manager for disabling gradient computation");

    // Python context manager for enable_grad
    m.def("enable_grad", []() {
        struct EnableGradGuard {
            EnableGradGuard() : prev_state_(tenzor::is_grad_enabled()) {
                tenzor::set_grad_enabled(true);
            }
            ~EnableGradGuard() {
                tenzor::set_grad_enabled(prev_state_);
            }
            bool prev_state_;
        };
        return EnableGradGuard();
    }, "Context manager for enabling gradient computation");

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

    // Convolution layers
    py::class_<tenzor::nn::Conv2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Conv2d>>(nn, "Conv2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size"),
             py::arg("stride") = 1,
             py::arg("padding") = 0,
             py::arg("dilation") = 1,
             py::arg("groups") = 1,
             py::arg("bias") = true);

    // Conv1d - verified implemented in conv.cpp (lines 989-1216)
    py::class_<tenzor::nn::Conv1d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Conv1d>>(nn, "Conv1d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size"),
             py::arg("stride") = 1,
             py::arg("padding") = 0,
             py::arg("dilation") = 1,
             py::arg("groups") = 1,
             py::arg("bias") = true);

    // ConvTranspose2d - verified implemented in conv.cpp (lines 1219-1787)
    py::class_<tenzor::nn::ConvTranspose2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ConvTranspose2d>>(nn, "ConvTranspose2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size"),
             py::arg("stride") = 1,
             py::arg("padding") = 0,
             py::arg("output_padding") = 0,
             py::arg("groups") = 1,
             py::arg("bias") = true);

    // Normalization layers
    py::class_<tenzor::nn::BatchNorm2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::BatchNorm2d>>(nn, "BatchNorm2d")
        .def(py::init<int64_t, double, double, bool, bool>(),
             py::arg("num_features"),
             py::arg("eps") = 1e-5,
             py::arg("momentum") = 0.1,
             py::arg("affine") = true,
             py::arg("track_running_stats") = true);

    py::class_<tenzor::nn::BatchNorm1d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::BatchNorm1d>>(nn, "BatchNorm1d")
        .def(py::init<int64_t, double, double, bool, bool>(),
             py::arg("num_features"),
             py::arg("eps") = 1e-5,
             py::arg("momentum") = 0.1,
             py::arg("affine") = true,
             py::arg("track_running_stats") = true);

    py::class_<tenzor::nn::LayerNorm, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::LayerNorm>>(nn, "LayerNorm")
        .def(py::init<std::vector<int64_t>, double, bool>(),
             py::arg("normalized_shape"),
             py::arg("eps") = 1e-5,
             py::arg("elementwise_affine") = true);

    py::class_<tenzor::nn::GroupNorm, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::GroupNorm>>(nn, "GroupNorm")
        .def(py::init<int64_t, int64_t, double, bool>(),
             py::arg("num_groups"),
             py::arg("num_channels"),
             py::arg("eps") = 1e-5,
             py::arg("affine") = true);

    // Regularization layers
    py::class_<tenzor::nn::Dropout, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Dropout>>(nn, "Dropout")
        .def(py::init<double>(),
             py::arg("p") = 0.5);

    py::class_<tenzor::nn::Dropout2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Dropout2d>>(nn, "Dropout2d")
        .def(py::init<double>(),
             py::arg("p") = 0.5);

    // AlphaDropout - verified implemented in dropout.cpp (lines 293-443)
    py::class_<tenzor::nn::AlphaDropout, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::AlphaDropout>>(nn, "AlphaDropout")
        .def(py::init<double>(),
             py::arg("p") = 0.5);

    // Pooling layers
    py::class_<tenzor::nn::MaxPool2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::MaxPool2d>>(nn, "MaxPool2d")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("kernel_size"),
             py::arg("stride") = -1,
             py::arg("padding") = 0);

    py::class_<tenzor::nn::AvgPool2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::AvgPool2d>>(nn, "AvgPool2d")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("kernel_size"),
             py::arg("stride") = -1,
             py::arg("padding") = 0);

    py::class_<tenzor::nn::AdaptiveAvgPool2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::AdaptiveAvgPool2d>>(nn, "AdaptiveAvgPool2d")
        .def(py::init<int64_t, int64_t>(),
             py::arg("output_h"), py::arg("output_w"))
        .def(py::init<int64_t>(),
             py::arg("output_size"));

    // Utility layers
    py::class_<tenzor::nn::Flatten, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Flatten>>(nn, "Flatten")
        .def(py::init<int64_t, int64_t>(),
             py::arg("start_dim") = 1,
             py::arg("end_dim") = -1);

    // Sequential container
    py::class_<tenzor::nn::Sequential, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Sequential>>(nn, "Sequential")
        .def(py::init<>())
        .def("add_module", &tenzor::nn::Sequential::add_module,
             py::return_value_policy::reference_internal,
             py::arg("module"));

    // Activation function classes
    py::class_<tenzor::nn::ReLU, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ReLU>>(nn, "ReLU")
        .def(py::init<>());

    py::class_<tenzor::nn::LeakyReLU, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::LeakyReLU>>(nn, "LeakyReLU")
        .def(py::init<double>(),
             py::arg("negative_slope") = 0.01);

    py::class_<tenzor::nn::ELU, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ELU>>(nn, "ELU")
        .def(py::init<double>(),
             py::arg("alpha") = 1.0);

    py::class_<tenzor::nn::GELU, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::GELU>>(nn, "GELU")
        .def(py::init<>());

    py::class_<tenzor::nn::Sigmoid, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Sigmoid>>(nn, "Sigmoid")
        .def(py::init<>());

    py::class_<tenzor::nn::Tanh, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Tanh>>(nn, "Tanh")
        .def(py::init<>());

    py::class_<tenzor::nn::Softmax, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Softmax>>(nn, "Softmax")
        .def(py::init<int64_t>(),
             py::arg("dim") = -1);

    py::class_<tenzor::nn::LogSoftmax, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::LogSoftmax>>(nn, "LogSoftmax")
        .def(py::init<int64_t>(),
             py::arg("dim") = -1);

    py::class_<tenzor::nn::SELU, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::SELU>>(nn, "SELU")
        .def(py::init<>());

    py::class_<tenzor::nn::Swish, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Swish>>(nn, "Swish")
        .def(py::init<>());

    py::class_<tenzor::nn::Mish, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Mish>>(nn, "Mish")
        .def(py::init<>());

    // RNN layers
    py::class_<tenzor::nn::RNNCell, tenzor::nn::Module, std::shared_ptr<tenzor::nn::RNNCell>>(nn, "RNNCell")
        .def(py::init<int64_t, int64_t, const std::string&, bool>(),
             py::arg("input_size"), py::arg("hidden_size"),
             py::arg("nonlinearity") = "tanh", py::arg("bias") = true)
        .def("forward", [](tenzor::nn::RNNCell& self, const tenzor::Variable& input, const tenzor::Variable& hx) {
            return self.forward(input, hx);
        }, py::arg("input"), py::arg("hx") = tenzor::Variable{});

    py::class_<tenzor::nn::RNN, tenzor::nn::Module, std::shared_ptr<tenzor::nn::RNN>>(nn, "RNN")
        .def(py::init<int64_t, int64_t, int64_t, const std::string&, bool, bool, double, bool>(),
             py::arg("input_size"), py::arg("hidden_size"), py::arg("num_layers") = 1,
             py::arg("nonlinearity") = "tanh", py::arg("bias") = true,
             py::arg("batch_first") = false, py::arg("dropout") = 0.0,
             py::arg("bidirectional") = false)
        .def("forward", [](tenzor::nn::RNN& self, const tenzor::Variable& input, const tenzor::Variable& hx) {
            return self.forward(input, hx);
        }, py::arg("input"), py::arg("hx") = tenzor::Variable{});

    py::class_<tenzor::nn::LSTMCell, tenzor::nn::Module, std::shared_ptr<tenzor::nn::LSTMCell>>(nn, "LSTMCell")
        .def(py::init<int64_t, int64_t, bool>(),
             py::arg("input_size"), py::arg("hidden_size"), py::arg("bias") = true)
        .def("forward", [](tenzor::nn::LSTMCell& self, const tenzor::Variable& input,
                           const tenzor::Variable& hx, const tenzor::Variable& cx) {
            return self.forward(input, hx, cx);
        }, py::arg("input"), py::arg("hx") = tenzor::Variable{},
           py::arg("cx") = tenzor::Variable{});

    py::class_<tenzor::nn::LSTM, tenzor::nn::Module, std::shared_ptr<tenzor::nn::LSTM>>(nn, "LSTM")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool, double, bool, int64_t>(),
             py::arg("input_size"), py::arg("hidden_size"), py::arg("num_layers") = 1,
             py::arg("bias") = true, py::arg("batch_first") = false,
             py::arg("dropout") = 0.0, py::arg("bidirectional") = false,
             py::arg("proj_size") = 0)
        .def("forward", [](tenzor::nn::LSTM& self, const tenzor::Variable& input,
                           const std::pair<tenzor::Variable, tenzor::Variable>& hx) {
            return self.forward(input, hx);
        }, py::arg("input"), py::arg("hx") = std::pair<tenzor::Variable, tenzor::Variable>{});

    py::class_<tenzor::nn::GRUCell, tenzor::nn::Module, std::shared_ptr<tenzor::nn::GRUCell>>(nn, "GRUCell")
        .def(py::init<int64_t, int64_t, bool>(),
             py::arg("input_size"), py::arg("hidden_size"), py::arg("bias") = true)
        .def("forward", [](tenzor::nn::GRUCell& self, const tenzor::Variable& input, const tenzor::Variable& hx) {
            return self.forward(input, hx);
        }, py::arg("input"), py::arg("hx") = tenzor::Variable{});

    py::class_<tenzor::nn::GRU, tenzor::nn::Module, std::shared_ptr<tenzor::nn::GRU>>(nn, "GRU")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool, double, bool>(),
             py::arg("input_size"), py::arg("hidden_size"), py::arg("num_layers") = 1,
             py::arg("bias") = true, py::arg("batch_first") = false,
             py::arg("dropout") = 0.0, py::arg("bidirectional") = false)
        .def("forward", [](tenzor::nn::GRU& self, const tenzor::Variable& input, const tenzor::Variable& hx) {
            return self.forward(input, hx);
        }, py::arg("input"), py::arg("hx") = tenzor::Variable{});

    // Attention and Transformer
    py::class_<tenzor::nn::MultiheadAttention, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::MultiheadAttention>>(nn, "MultiheadAttention")
        .def(py::init<int64_t, int64_t, double, bool, bool, bool, int64_t, int64_t, bool>(),
             py::arg("embed_dim"), py::arg("num_heads"), py::arg("dropout") = 0.0,
             py::arg("bias") = true, py::arg("add_bias_kv") = false,
             py::arg("add_zero_attn") = false, py::arg("kdim") = 0,
             py::arg("vdim") = 0, py::arg("batch_first") = false)
        .def("forward", [](tenzor::nn::MultiheadAttention& self, const tenzor::Variable& query,
                           const tenzor::Variable& key, const tenzor::Variable& value,
                           const tenzor::Tensor& key_padding_mask, const tenzor::Tensor& attn_mask,
                           bool need_weights) {
            return self.forward(query, key, value, key_padding_mask, attn_mask, need_weights);
        }, py::arg("query"), py::arg("key"), py::arg("value"),
           py::arg("key_padding_mask") = tenzor::Tensor{},
           py::arg("attn_mask") = tenzor::Tensor{},
           py::arg("need_weights") = true);

    py::class_<tenzor::nn::PositionalEncoding, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::PositionalEncoding>>(nn, "PositionalEncoding")
        .def(py::init<int64_t, int64_t, double>(),
             py::arg("d_model"), py::arg("max_len") = 5000, py::arg("dropout") = 0.0)
        .def("forward", &tenzor::nn::PositionalEncoding::forward);

    py::class_<tenzor::nn::TransformerEncoderLayer, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::TransformerEncoderLayer>>(nn, "TransformerEncoderLayer")
        .def(py::init<int64_t, int64_t, int64_t, double, const std::string&, bool>(),
             py::arg("d_model"), py::arg("nhead"), py::arg("dim_feedforward") = 2048,
             py::arg("dropout") = 0.1, py::arg("activation") = "relu",
             py::arg("batch_first") = false)
        .def("forward", [](tenzor::nn::TransformerEncoderLayer& self, const tenzor::Variable& src,
                           const tenzor::Tensor& src_mask, const tenzor::Tensor& src_key_padding_mask) {
            return self.forward(src, src_mask, src_key_padding_mask);
        }, py::arg("src"), py::arg("src_mask") = tenzor::Tensor{},
           py::arg("src_key_padding_mask") = tenzor::Tensor{});

    py::class_<tenzor::nn::TransformerEncoder, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::TransformerEncoder>>(nn, "TransformerEncoder")
        .def(py::init<std::shared_ptr<tenzor::nn::TransformerEncoderLayer>, int64_t,
                     std::shared_ptr<tenzor::nn::LayerNorm>>(),
             py::arg("encoder_layer"), py::arg("num_layers"), py::arg("norm") = nullptr)
        .def("forward", [](tenzor::nn::TransformerEncoder& self, const tenzor::Variable& src,
                           const tenzor::Tensor& mask, const tenzor::Tensor& src_key_padding_mask) {
            return self.forward(src, mask, src_key_padding_mask);
        }, py::arg("src"), py::arg("mask") = tenzor::Tensor{},
           py::arg("src_key_padding_mask") = tenzor::Tensor{});

    py::class_<tenzor::nn::TransformerDecoderLayer, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::TransformerDecoderLayer>>(nn, "TransformerDecoderLayer")
        .def(py::init<int64_t, int64_t, int64_t, double, const std::string&, bool>(),
             py::arg("d_model"), py::arg("nhead"), py::arg("dim_feedforward") = 2048,
             py::arg("dropout") = 0.1, py::arg("activation") = "relu",
             py::arg("batch_first") = false)
        .def("forward", [](tenzor::nn::TransformerDecoderLayer& self, const tenzor::Variable& tgt,
                           const tenzor::Variable& memory, const tenzor::Tensor& tgt_mask,
                           const tenzor::Tensor& memory_mask, const tenzor::Tensor& tgt_key_padding_mask,
                           const tenzor::Tensor& memory_key_padding_mask) {
            return self.forward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask);
        }, py::arg("tgt"), py::arg("memory"),
           py::arg("tgt_mask") = tenzor::Tensor{},
           py::arg("memory_mask") = tenzor::Tensor{},
           py::arg("tgt_key_padding_mask") = tenzor::Tensor{},
           py::arg("memory_key_padding_mask") = tenzor::Tensor{});

    py::class_<tenzor::nn::TransformerDecoder, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::TransformerDecoder>>(nn, "TransformerDecoder")
        .def(py::init<std::shared_ptr<tenzor::nn::TransformerDecoderLayer>, int64_t,
                     std::shared_ptr<tenzor::nn::LayerNorm>>(),
             py::arg("decoder_layer"), py::arg("num_layers"), py::arg("norm") = nullptr)
        .def("forward", [](tenzor::nn::TransformerDecoder& self, const tenzor::Variable& tgt,
                           const tenzor::Variable& memory, const tenzor::Tensor& tgt_mask,
                           const tenzor::Tensor& memory_mask, const tenzor::Tensor& tgt_key_padding_mask,
                           const tenzor::Tensor& memory_key_padding_mask) {
            return self.forward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask);
        }, py::arg("tgt"), py::arg("memory"),
           py::arg("tgt_mask") = tenzor::Tensor{},
           py::arg("memory_mask") = tenzor::Tensor{},
           py::arg("tgt_key_padding_mask") = tenzor::Tensor{},
           py::arg("memory_key_padding_mask") = tenzor::Tensor{});

    py::class_<tenzor::nn::Transformer, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Transformer>>(nn, "Transformer")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, double, const std::string&, bool>(),
             py::arg("d_model") = 512, py::arg("nhead") = 8,
             py::arg("num_encoder_layers") = 6, py::arg("num_decoder_layers") = 6,
             py::arg("dim_feedforward") = 2048, py::arg("dropout") = 0.1,
             py::arg("activation") = "relu", py::arg("batch_first") = false)
        .def("forward", [](tenzor::nn::Transformer& self, const tenzor::Variable& src,
                           const tenzor::Variable& tgt, const tenzor::Tensor& src_mask,
                           const tenzor::Tensor& tgt_mask, const tenzor::Tensor& memory_mask,
                           const tenzor::Tensor& src_key_padding_mask, const tenzor::Tensor& tgt_key_padding_mask,
                           const tenzor::Tensor& memory_key_padding_mask) {
            return self.forward(src, tgt, src_mask, tgt_mask, memory_mask,
                              src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask);
        }, py::arg("src"), py::arg("tgt"),
           py::arg("src_mask") = tenzor::Tensor{},
           py::arg("tgt_mask") = tenzor::Tensor{},
           py::arg("memory_mask") = tenzor::Tensor{},
           py::arg("src_key_padding_mask") = tenzor::Tensor{},
           py::arg("tgt_key_padding_mask") = tenzor::Tensor{},
           py::arg("memory_key_padding_mask") = tenzor::Tensor{});

    // Embedding layers
    py::class_<tenzor::nn::Embedding, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Embedding>>(nn, "Embedding")
        .def(py::init<int64_t, int64_t, int64_t, double, double, bool, bool>(),
             py::arg("num_embeddings"), py::arg("embedding_dim"),
             py::arg("padding_idx") = -1, py::arg("max_norm") = 0.0,
             py::arg("norm_type") = 2.0, py::arg("scale_grad_by_freq") = false,
             py::arg("sparse") = false)
        .def("forward", &tenzor::nn::Embedding::forward)
        .def("weight", py::overload_cast<>(&tenzor::nn::Embedding::weight));

    py::class_<tenzor::nn::EmbeddingBag, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::EmbeddingBag>>(nn, "EmbeddingBag")
        .def(py::init<int64_t, int64_t, double, double, bool, const std::string&, bool, bool>(),
             py::arg("num_embeddings"), py::arg("embedding_dim"),
             py::arg("max_norm") = 0.0, py::arg("norm_type") = 2.0,
             py::arg("scale_grad_by_freq") = false, py::arg("mode") = "mean",
             py::arg("sparse") = false, py::arg("include_last_offset") = false)
        .def("forward", [](tenzor::nn::EmbeddingBag& self, const tenzor::Variable& input, const tenzor::Variable& offsets) {
            return self.forward(input, offsets);
        }, py::arg("input"), py::arg("offsets") = tenzor::Variable{});

    // Functional activation functions
    nn.def("relu", &tenzor::nn::relu, "ReLU activation function");
    nn.def("leaky_relu", &tenzor::nn::leaky_relu, "Leaky ReLU activation function",
          py::arg("input"), py::arg("negative_slope") = 0.01);
    nn.def("elu", &tenzor::nn::elu, "ELU activation function",
          py::arg("input"), py::arg("alpha") = 1.0);
    nn.def("gelu", &tenzor::nn::gelu, "GELU activation function");
    nn.def("sigmoid", &tenzor::nn::sigmoid, "Sigmoid activation function");
    nn.def("tanh", &tenzor::nn::tanh, "Tanh activation function");
    nn.def("softmax", &tenzor::nn::softmax, "Softmax activation function",
          py::arg("input"), py::arg("dim") = -1);
    nn.def("log_softmax", &tenzor::nn::log_softmax, "Log-Softmax activation function",
          py::arg("input"), py::arg("dim") = -1);
    nn.def("selu", &tenzor::nn::selu, "SELU activation function");
    nn.def("swish", &tenzor::nn::swish, "Swish activation function");
    nn.def("mish", &tenzor::nn::mish, "Mish activation function");

    // Reduction enum for loss functions
    py::enum_<tenzor::nn::Reduction>(nn, "Reduction")
        .value("none", tenzor::nn::Reduction::None)
        .value("mean", tenzor::nn::Reduction::Mean)
        .value("sum", tenzor::nn::Reduction::Sum);

    // Loss function classes
    py::class_<tenzor::nn::MSELoss>(nn, "MSELoss")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::MSELoss::forward)
        .def("__call__", &tenzor::nn::MSELoss::operator());

    py::class_<tenzor::nn::L1Loss>(nn, "L1Loss")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::L1Loss::forward)
        .def("__call__", &tenzor::nn::L1Loss::operator());

    py::class_<tenzor::nn::SmoothL1Loss>(nn, "SmoothL1Loss")
        .def(py::init<tenzor::nn::Reduction, double>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean,
             py::arg("beta") = 1.0)
        .def("forward", &tenzor::nn::SmoothL1Loss::forward)
        .def("__call__", &tenzor::nn::SmoothL1Loss::operator());

    py::class_<tenzor::nn::CrossEntropyLoss>(nn, "CrossEntropyLoss")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::CrossEntropyLoss::forward)
        .def("__call__", &tenzor::nn::CrossEntropyLoss::operator());

    py::class_<tenzor::nn::NLLLoss>(nn, "NLLLoss")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::NLLLoss::forward)
        .def("__call__", &tenzor::nn::NLLLoss::operator());

    py::class_<tenzor::nn::BCELoss>(nn, "BCELoss")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::BCELoss::forward)
        .def("__call__", &tenzor::nn::BCELoss::operator());

    py::class_<tenzor::nn::BCEWithLogitsLoss>(nn, "BCEWithLogitsLoss")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::BCEWithLogitsLoss::forward)
        .def("__call__", &tenzor::nn::BCEWithLogitsLoss::operator());

    // Advanced loss functions
    py::class_<tenzor::nn::KLDivLoss>(nn, "KLDivLoss")
        .def(py::init<const std::string&, bool>(),
             py::arg("reduction") = "mean", py::arg("log_target") = false)
        .def("forward", &tenzor::nn::KLDivLoss::forward)
        .def("__call__", &tenzor::nn::KLDivLoss::operator());

    py::class_<tenzor::nn::FocalLoss>(nn, "FocalLoss")
        .def(py::init<double, double, const std::string&>(),
             py::arg("alpha") = 1.0, py::arg("gamma") = 2.0,
             py::arg("reduction") = "mean")
        .def("forward", &tenzor::nn::FocalLoss::forward)
        .def("__call__", &tenzor::nn::FocalLoss::operator());

    py::class_<tenzor::nn::DiceLoss>(nn, "DiceLoss")
        .def(py::init<double, const std::string&>(),
             py::arg("smooth") = 1.0, py::arg("reduction") = "mean")
        .def("forward", &tenzor::nn::DiceLoss::forward)
        .def("__call__", &tenzor::nn::DiceLoss::operator());

    py::class_<tenzor::nn::HuberLoss>(nn, "HuberLoss")
        .def(py::init<double, const std::string&>(),
             py::arg("delta") = 1.0, py::arg("reduction") = "mean")
        .def("forward", &tenzor::nn::HuberLoss::forward)
        .def("__call__", &tenzor::nn::HuberLoss::operator());

    // Functional loss functions
    nn.def("mse_loss", &tenzor::nn::mse_loss, "MSE loss function",
          py::arg("input"), py::arg("target"),
          py::arg("reduction") = tenzor::nn::Reduction::Mean);
    nn.def("l1_loss", &tenzor::nn::l1_loss, "L1 loss function",
          py::arg("input"), py::arg("target"),
          py::arg("reduction") = tenzor::nn::Reduction::Mean);
    nn.def("cross_entropy", &tenzor::nn::cross_entropy, "Cross entropy loss function",
          py::arg("input"), py::arg("target"),
          py::arg("reduction") = tenzor::nn::Reduction::Mean);
    nn.def("nll_loss", &tenzor::nn::nll_loss, "NLL loss function",
          py::arg("input"), py::arg("target"),
          py::arg("reduction") = tenzor::nn::Reduction::Mean);
    nn.def("bce_loss", &tenzor::nn::bce_loss, "BCE loss function",
          py::arg("input"), py::arg("target"),
          py::arg("reduction") = tenzor::nn::Reduction::Mean);

    // Optimizers
    auto optim = m.def_submodule("optim", "Optimization algorithms");

    py::class_<tenzor::optim::SGD>(optim, "SGD")
        .def(py::init<std::vector<tenzor::Variable*>, double, double, double, double, bool>(),
             py::arg("params"), py::arg("lr"),
             py::arg("momentum") = 0.0, py::arg("dampening") = 0.0,
             py::arg("weight_decay") = 0.0, py::arg("nesterov") = false)
        .def("step", &tenzor::optim::SGD::step)
        .def("zero_grad", &tenzor::optim::SGD::zero_grad)
        .def("set_lr", &tenzor::optim::SGD::set_lr,
             py::arg("lr"), "Set learning rate")
        .def("get_lr", &tenzor::optim::SGD::get_lr,
             "Get current learning rate")
        .def("state_dict", &tenzor::optim::SGD::state_dict,
             "Get optimizer state dictionary")
        .def("load_state_dict", &tenzor::optim::SGD::load_state_dict,
             py::arg("state"), "Load optimizer state dictionary");

    py::class_<tenzor::optim::Adam>(optim, "Adam")
        .def(py::init<std::vector<tenzor::Variable*>, double, double, double, double, double, bool>(),
             py::arg("params"), py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             py::arg("amsgrad") = false)
        .def("step", &tenzor::optim::Adam::step)
        .def("zero_grad", &tenzor::optim::Adam::zero_grad)
        .def("set_lr", &tenzor::optim::Adam::set_lr,
             py::arg("lr"), "Set learning rate")
        .def("get_lr", &tenzor::optim::Adam::get_lr,
             "Get current learning rate")
        .def("state_dict", &tenzor::optim::Adam::state_dict,
             "Get optimizer state dictionary")
        .def("load_state_dict", &tenzor::optim::Adam::load_state_dict,
             py::arg("state"), "Load optimizer state dictionary");

    py::class_<tenzor::optim::AdamW>(optim, "AdamW")
        .def(py::init<std::vector<tenzor::Variable*>, double, double, double, double, double, bool>(),
             py::arg("params"), py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.01,
             py::arg("amsgrad") = false)
        .def("step", &tenzor::optim::AdamW::step)
        .def("zero_grad", &tenzor::optim::AdamW::zero_grad)
        .def("set_lr", &tenzor::optim::AdamW::set_lr,
             py::arg("lr"), "Set learning rate")
        .def("get_lr", &tenzor::optim::AdamW::get_lr,
             "Get current learning rate")
        .def("state_dict", &tenzor::optim::AdamW::state_dict,
             "Get optimizer state dictionary")
        .def("load_state_dict", &tenzor::optim::AdamW::load_state_dict,
             py::arg("state"), "Load optimizer state dictionary");

    // Additional optimizers
    py::class_<tenzor::optim::RMSprop>(optim, "RMSprop")
        .def(py::init<std::vector<tenzor::Variable*>, double, double, double, double, double, bool>(),
             py::arg("params"), py::arg("lr") = 0.01, py::arg("alpha") = 0.99,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             py::arg("momentum") = 0.0, py::arg("centered") = false)
        .def("step", &tenzor::optim::RMSprop::step)
        .def("zero_grad", &tenzor::optim::RMSprop::zero_grad)
        .def("state_dict", &tenzor::optim::RMSprop::state_dict)
        .def("load_state_dict", &tenzor::optim::RMSprop::load_state_dict);

    py::class_<tenzor::optim::Adagrad>(optim, "Adagrad")
        .def(py::init<std::vector<tenzor::Variable*>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 0.01, py::arg("lr_decay") = 0.0,
             py::arg("weight_decay") = 0.0, py::arg("initial_accumulator_value") = 0.0,
             py::arg("eps") = 1e-10)
        .def("step", &tenzor::optim::Adagrad::step)
        .def("zero_grad", &tenzor::optim::Adagrad::zero_grad);

    py::class_<tenzor::optim::Adadelta>(optim, "Adadelta")
        .def(py::init<std::vector<tenzor::Variable*>, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 1.0, py::arg("rho") = 0.9,
             py::arg("eps") = 1e-6, py::arg("weight_decay") = 0.0)
        .def("step", &tenzor::optim::Adadelta::step)
        .def("zero_grad", &tenzor::optim::Adadelta::zero_grad);

    // Learning rate schedulers
    auto lr_scheduler = optim.def_submodule("lr_scheduler", "Learning rate scheduling");

    // Base scheduler class
    py::class_<tenzor::optim::LRScheduler>(lr_scheduler, "LRScheduler")
        .def("step", &tenzor::optim::LRScheduler::step,
             "Step the scheduler (typically called once per epoch)")
        .def("get_last_lr", &tenzor::optim::LRScheduler::get_last_lr,
             "Get the last computed learning rate")
        .def("get_lr", &tenzor::optim::LRScheduler::get_lr,
             "Get the current learning rate");

    // StepLR scheduler
    py::class_<tenzor::optim::StepLR, tenzor::optim::LRScheduler>(lr_scheduler, "StepLR")
        .def(py::init<tenzor::optim::SGD&, int, double>(),
             py::arg("optimizer"),
             py::arg("step_size"),
             py::arg("gamma") = 0.1,
             "Decays learning rate by gamma every step_size epochs")
        .def(py::init<tenzor::optim::Adam&, int, double>(),
             py::arg("optimizer"),
             py::arg("step_size"),
             py::arg("gamma") = 0.1)
        .def(py::init<tenzor::optim::AdamW&, int, double>(),
             py::arg("optimizer"),
             py::arg("step_size"),
             py::arg("gamma") = 0.1)
        .def("get_epoch", &tenzor::optim::StepLR::get_epoch,
             "Get current epoch number");

    // ExponentialLR scheduler
    py::class_<tenzor::optim::ExponentialLR, tenzor::optim::LRScheduler>(lr_scheduler, "ExponentialLR")
        .def(py::init<tenzor::optim::SGD&, double>(),
             py::arg("optimizer"),
             py::arg("gamma"),
             "Decays learning rate exponentially by gamma every epoch")
        .def(py::init<tenzor::optim::Adam&, double>(),
             py::arg("optimizer"),
             py::arg("gamma"))
        .def(py::init<tenzor::optim::AdamW&, double>(),
             py::arg("optimizer"),
             py::arg("gamma"))
        .def("get_epoch", &tenzor::optim::ExponentialLR::get_epoch,
             "Get current epoch number");

    // CosineAnnealingLR scheduler
    py::class_<tenzor::optim::CosineAnnealingLR, tenzor::optim::LRScheduler>(lr_scheduler, "CosineAnnealingLR")
        .def(py::init<tenzor::optim::SGD&, int, double>(),
             py::arg("optimizer"),
             py::arg("T_max"),
             py::arg("eta_min") = 0.0,
             "Cosine annealing learning rate schedule")
        .def(py::init<tenzor::optim::Adam&, int, double>(),
             py::arg("optimizer"),
             py::arg("T_max"),
             py::arg("eta_min") = 0.0)
        .def(py::init<tenzor::optim::AdamW&, int, double>(),
             py::arg("optimizer"),
             py::arg("T_max"),
             py::arg("eta_min") = 0.0)
        .def("get_epoch", &tenzor::optim::CosineAnnealingLR::get_epoch,
             "Get current epoch number");

    // Advanced schedulers
    py::class_<tenzor::optim::ReduceLROnPlateau, tenzor::optim::LRScheduler>(lr_scheduler, "ReduceLROnPlateau")
        .def(py::init<tenzor::optim::SGD&, const std::string&, double, int64_t, double,
                     const std::string&, int64_t, double, double>(),
             py::arg("optimizer"), py::arg("mode") = "min", py::arg("factor") = 0.1,
             py::arg("patience") = 10, py::arg("threshold") = 1e-4,
             py::arg("threshold_mode") = "rel", py::arg("cooldown") = 0,
             py::arg("min_lr") = 0.0, py::arg("eps") = 1e-8,
             "Reduce learning rate when metric plateaus")
        .def(py::init<tenzor::optim::Adam&, const std::string&, double, int64_t, double,
                     const std::string&, int64_t, double, double>(),
             py::arg("optimizer"), py::arg("mode") = "min", py::arg("factor") = 0.1,
             py::arg("patience") = 10, py::arg("threshold") = 1e-4,
             py::arg("threshold_mode") = "rel", py::arg("cooldown") = 0,
             py::arg("min_lr") = 0.0, py::arg("eps") = 1e-8)
        .def(py::init<tenzor::optim::AdamW&, const std::string&, double, int64_t, double,
                     const std::string&, int64_t, double, double>(),
             py::arg("optimizer"), py::arg("mode") = "min", py::arg("factor") = 0.1,
             py::arg("patience") = 10, py::arg("threshold") = 1e-4,
             py::arg("threshold_mode") = "rel", py::arg("cooldown") = 0,
             py::arg("min_lr") = 0.0, py::arg("eps") = 1e-8)
        .def("step", py::overload_cast<double>(&tenzor::optim::ReduceLROnPlateau::step));

    py::class_<tenzor::optim::CyclicLR, tenzor::optim::LRScheduler>(lr_scheduler, "CyclicLR")
        .def(py::init<tenzor::optim::SGD&, double, double, int64_t, int64_t,
                     const std::string&, double, double, const std::string&>(),
             py::arg("optimizer"), py::arg("base_lr"), py::arg("max_lr"),
             py::arg("step_size_up") = 2000, py::arg("step_size_down") = -1,
             py::arg("mode") = "triangular", py::arg("gamma") = 1.0,
             py::arg("scale_fn") = 1.0, py::arg("scale_mode") = "cycle",
             "Cyclic learning rate schedule")
        .def(py::init<tenzor::optim::Adam&, double, double, int64_t, int64_t,
                     const std::string&, double, double, const std::string&>(),
             py::arg("optimizer"), py::arg("base_lr"), py::arg("max_lr"),
             py::arg("step_size_up") = 2000, py::arg("step_size_down") = -1,
             py::arg("mode") = "triangular", py::arg("gamma") = 1.0,
             py::arg("scale_fn") = 1.0, py::arg("scale_mode") = "cycle")
        .def(py::init<tenzor::optim::AdamW&, double, double, int64_t, int64_t,
                     const std::string&, double, double, const std::string&>(),
             py::arg("optimizer"), py::arg("base_lr"), py::arg("max_lr"),
             py::arg("step_size_up") = 2000, py::arg("step_size_down") = -1,
             py::arg("mode") = "triangular", py::arg("gamma") = 1.0,
             py::arg("scale_fn") = 1.0, py::arg("scale_mode") = "cycle")
        .def("step", &tenzor::optim::CyclicLR::step);

    py::class_<tenzor::optim::OneCycleLR, tenzor::optim::LRScheduler>(lr_scheduler, "OneCycleLR")
        .def(py::init<tenzor::optim::SGD&, double, int64_t, int64_t, int64_t, double,
                     const std::string&, double, double>(),
             py::arg("optimizer"), py::arg("max_lr"), py::arg("total_steps"),
             py::arg("epochs") = -1, py::arg("steps_per_epoch") = -1,
             py::arg("pct_start") = 0.3, py::arg("anneal_strategy") = "cos",
             py::arg("div_factor") = 25.0, py::arg("final_div_factor") = 1e4,
             "One cycle learning rate schedule")
        .def(py::init<tenzor::optim::Adam&, double, int64_t, int64_t, int64_t, double,
                     const std::string&, double, double>(),
             py::arg("optimizer"), py::arg("max_lr"), py::arg("total_steps"),
             py::arg("epochs") = -1, py::arg("steps_per_epoch") = -1,
             py::arg("pct_start") = 0.3, py::arg("anneal_strategy") = "cos",
             py::arg("div_factor") = 25.0, py::arg("final_div_factor") = 1e4)
        .def(py::init<tenzor::optim::AdamW&, double, int64_t, int64_t, int64_t, double,
                     const std::string&, double, double>(),
             py::arg("optimizer"), py::arg("max_lr"), py::arg("total_steps"),
             py::arg("epochs") = -1, py::arg("steps_per_epoch") = -1,
             py::arg("pct_start") = 0.3, py::arg("anneal_strategy") = "cos",
             py::arg("div_factor") = 25.0, py::arg("final_div_factor") = 1e4)
        .def("step", &tenzor::optim::OneCycleLR::step);

    py::class_<tenzor::optim::CosineAnnealingWarmRestarts, tenzor::optim::LRScheduler>(lr_scheduler, "CosineAnnealingWarmRestarts")
        .def(py::init<tenzor::optim::SGD&, int64_t, int64_t, double>(),
             py::arg("optimizer"), py::arg("T_0"), py::arg("T_mult") = 1,
             py::arg("eta_min") = 0.0,
             "Cosine annealing with warm restarts")
        .def(py::init<tenzor::optim::Adam&, int64_t, int64_t, double>(),
             py::arg("optimizer"), py::arg("T_0"), py::arg("T_mult") = 1,
             py::arg("eta_min") = 0.0)
        .def(py::init<tenzor::optim::AdamW&, int64_t, int64_t, double>(),
             py::arg("optimizer"), py::arg("T_0"), py::arg("T_mult") = 1,
             py::arg("eta_min") = 0.0)
        .def("step", &tenzor::optim::CosineAnnealingWarmRestarts::step);
}
