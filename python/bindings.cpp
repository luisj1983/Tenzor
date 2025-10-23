#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/indexing.hpp>
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/backend/loader.hpp>
#include <tenzor/backend/backend.hpp>
#include <tenzor/nn/optim/scheduler.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include <tenzor/nn/layers/attention.hpp>
#include <tenzor/nn/layers/transformer.hpp>
#include <tenzor/nn/layers/embedding.hpp>
#include <tenzor/nn/optim/rmsprop.hpp>
#include <tenzor/nn/optim/adagrad.hpp>
#include <tenzor/nn/optim/adadelta.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <tenzor/nn/parallel/distributed_data_parallel.hpp>
#include <tenzor/models/hub.hpp>
#include <tenzor/onnx/exporter.hpp>
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
        .def("fill_", &tenzor::Tensor::fill_, py::arg("value"),
             "Fill tensor with scalar value in-place")
        .def("zero_", &tenzor::Tensor::zero_,
             "Fill tensor with zeros in-place")
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
            // Helper function to convert Python value to tensor
            auto value_to_tensor = [&](py::object val) -> tenzor::Tensor {
                if (py::isinstance<tenzor::Tensor>(val)) {
                    return py::cast<tenzor::Tensor>(val);
                } else if (py::isinstance<py::float_>(val) || py::isinstance<py::int_>(val)) {
                    // Scalar value - create single-element tensor
                    float scalar = py::cast<float>(val);
                    auto scalar_tensor = tenzor::empty({1}, self.dtype(), self.device());

                    // Fill with scalar value based on dtype
                    switch (self.dtype()) {
                        case tenzor::DType::Float32:
                            *scalar_tensor.data<float>() = scalar;
                            break;
                        case tenzor::DType::Float64:
                            *scalar_tensor.data<double>() = static_cast<double>(scalar);
                            break;
                        case tenzor::DType::Int32:
                            *scalar_tensor.data<int32_t>() = static_cast<int32_t>(scalar);
                            break;
                        case tenzor::DType::Int64:
                            *scalar_tensor.data<int64_t>() = static_cast<int64_t>(scalar);
                            break;
                        case tenzor::DType::UInt8:
                            *scalar_tensor.data<uint8_t>() = static_cast<uint8_t>(scalar);
                            break;
                        case tenzor::DType::Bool:
                            *scalar_tensor.data<bool>() = static_cast<bool>(scalar);
                            break;
                        default:
                            throw std::runtime_error("Unsupported dtype for scalar assignment");
                    }
                    return scalar_tensor;
                } else {
                    throw std::runtime_error("Value must be a Tensor or scalar");
                }
            };

            // Helper function to copy data from source to destination with broadcasting
            auto copy_with_broadcast = [](tenzor::Tensor& dst, const tenzor::Tensor& src) {
                // Check device compatibility
                if (dst.device().type != src.device().type) {
                    throw std::runtime_error("Source and destination tensors must be on the same device");
                }

                auto dst_shape = dst.shape();
                auto src_shape = src.shape();

                // If source is scalar, broadcast to fill destination
                if (src.numel() == 1) {
                    // Get scalar value from source
                    auto src_cpu = (src.device().type == tenzor::Device::Type::CPU) ? src : src.cpu();
                    float scalar_value;
                    switch (src.dtype()) {
                        case tenzor::DType::Float32:
                            scalar_value = *src_cpu.data<float>();
                            break;
                        case tenzor::DType::Float64:
                            scalar_value = static_cast<float>(*src_cpu.data<double>());
                            break;
                        case tenzor::DType::Int32:
                            scalar_value = static_cast<float>(*src_cpu.data<int32_t>());
                            break;
                        case tenzor::DType::Int64:
                            scalar_value = static_cast<float>(*src_cpu.data<int64_t>());
                            break;
                        default:
                            scalar_value = 0.0f;
                    }
                    dst.fill_(scalar_value);
                    return;
                }

                // Check if shapes match exactly
                if (dst_shape.size() == src_shape.size()) {
                    bool shapes_match = true;
                    for (size_t i = 0; i < dst_shape.size(); ++i) {
                        if (dst_shape[i] != src_shape[i]) {
                            shapes_match = false;
                            break;
                        }
                    }
                    if (shapes_match) {
                        // Direct copy - same shape
                        if (dst.is_contiguous() && src.is_contiguous()) {
                            // Fast path: both contiguous
                            size_t bytes = dst.numel() * dst.dtype_size();
                            if (dst.device().type == tenzor::Device::Type::CPU) {
                                std::memcpy(dst.data_ptr(), src.data_ptr(), bytes);
                            } else {
                                // Use backend copy for device tensors
                                auto* backend = tenzor::backend_registry().get_backend(dst.device().type);
                                if (backend) {
                                    backend->copy(dst.data_ptr(), src.data_ptr(), bytes,
                                                tenzor::CopyKind::DeviceToDevice);
                                }
                            }
                        } else {
                            // Slow path: handle non-contiguous tensors
                            throw std::runtime_error("Non-contiguous tensor assignment not yet implemented");
                        }
                        return;
                    }
                }

                // Check if broadcasting is possible
                bool can_broadcast = true;
                int64_t dst_ndim = static_cast<int64_t>(dst_shape.size());
                int64_t src_ndim = static_cast<int64_t>(src_shape.size());

                if (src_ndim > dst_ndim) {
                    can_broadcast = false;
                } else {
                    // Check broadcasting rules
                    for (int64_t i = 0; i < src_ndim; ++i) {
                        int64_t dst_dim = dst_shape[dst_ndim - 1 - i];
                        int64_t src_dim = src_shape[src_ndim - 1 - i];
                        if (src_dim != 1 && src_dim != dst_dim) {
                            can_broadcast = false;
                            break;
                        }
                    }
                }

                if (!can_broadcast) {
                    throw std::runtime_error("Shape mismatch: cannot broadcast source shape to destination shape");
                }

                // TODO: Implement proper broadcasting copy
                throw std::runtime_error("Broadcasting assignment not yet fully implemented");
            };

            // Handle integer indexing: tensor[0] = value
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

                // Get slice along first dimension
                auto sliced = self.slice(0, idx, idx + 1);
                // Squeeze to remove the indexed dimension
                auto target = sliced.squeeze(0);

                // Convert value and copy
                auto value_tensor = value_to_tensor(value);
                copy_with_broadcast(target, value_tensor);
            }
            // Handle slice objects: tensor[1:5] = value
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
                    throw std::runtime_error("Slice step not supported yet for assignment");
                }

                // Get sliced view
                auto target = self.slice(0, start, stop);

                // Convert value and copy
                auto value_tensor = value_to_tensor(value);
                copy_with_broadcast(target, value_tensor);
            }
            // Handle tuple of indices/slices: tensor[0, :, 1:3] = value
            else if (py::isinstance<py::tuple>(key)) {
                py::tuple indices = py::cast<py::tuple>(key);
                tenzor::Tensor target = self;
                int squeeze_count = 0;  // Track dimensions that need squeezing
                std::vector<int64_t> squeeze_dims;  // Dimensions to squeeze at the end

                for (size_t i = 0; i < indices.size(); ++i) {
                    size_t adjusted_dim = i - squeeze_count;
                    auto target_shape = target.shape();

                    if (adjusted_dim >= target_shape.size()) {
                        throw std::out_of_range("Too many indices");
                    }

                    if (py::isinstance<py::int_>(indices[i])) {
                        int64_t idx = py::cast<int64_t>(indices[i]);

                        // Handle negative indexing
                        if (idx < 0) {
                            idx += target_shape[adjusted_dim];
                        }
                        if (idx < 0 || idx >= target_shape[adjusted_dim]) {
                            throw std::out_of_range("Index out of range");
                        }

                        // Slice along this dimension
                        target = target.slice(adjusted_dim, idx, idx + 1);
                        squeeze_dims.push_back(adjusted_dim);
                        squeeze_count++;

                    } else if (py::isinstance<py::slice>(indices[i])) {
                        py::slice slice_obj = py::cast<py::slice>(indices[i]);
                        py::ssize_t start, stop, step, length;

                        if (!slice_obj.compute(target_shape[adjusted_dim], &start, &stop, &step, &length)) {
                            throw std::runtime_error("Invalid slice");
                        }
                        if (step != 1) {
                            throw std::runtime_error("Slice step not supported yet for assignment");
                        }

                        target = target.slice(adjusted_dim, start, stop);
                    } else if (py::isinstance<py::ellipsis>(indices[i])) {
                        // Ellipsis: skip remaining dimensions until we have room for rest of indices
                        int64_t remaining_indices = static_cast<int64_t>(indices.size()) - static_cast<int64_t>(i) - 1;
                        int64_t remaining_dims = static_cast<int64_t>(target_shape.size()) - static_cast<int64_t>(adjusted_dim);
                        int64_t dims_to_skip = remaining_dims - remaining_indices;

                        if (dims_to_skip < 0) {
                            throw std::runtime_error("Invalid ellipsis: too many indices");
                        }

                        // Skip these dimensions (no slicing needed)
                        squeeze_count += dims_to_skip;
                    } else {
                        throw std::runtime_error("Unsupported index type in tuple");
                    }
                }

                // Squeeze indexed dimensions (from back to front to maintain indices)
                for (auto it = squeeze_dims.rbegin(); it != squeeze_dims.rend(); ++it) {
                    int64_t dim = *it;
                    // Adjust for previously squeezed dimensions
                    for (auto prev_it = it + 1; prev_it != squeeze_dims.rend(); ++prev_it) {
                        if (*prev_it < dim) {
                            dim--;
                        }
                    }
                    if (dim >= 0 && dim < target.ndim()) {
                        auto target_shape = target.shape();
                        if (target_shape[dim] == 1) {
                            target = target.squeeze(dim);
                        }
                    }
                }

                // Convert value and copy
                auto value_tensor = value_to_tensor(value);
                copy_with_broadcast(target, value_tensor);
            }
            else {
                throw std::runtime_error("Unsupported index type for assignment");
            }
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
    m.def("transpose", static_cast<tenzor::Tensor(*)(const tenzor::Tensor&, int64_t, int64_t)>(&tenzor::transpose),
         "Transpose two dimensions",
         py::arg("input"), py::arg("dim0"), py::arg("dim1"));
    m.def("permute", [](const tenzor::Tensor& input, std::vector<int64_t> dims) {
         return tenzor::permute(input, dims);
         }, "Permute dimensions",
         py::arg("input"), py::arg("dims"));
    m.def("squeeze", static_cast<tenzor::Tensor(*)(const tenzor::Tensor&, std::optional<int64_t>)>(&tenzor::squeeze),
         "Remove dimensions of size 1",
         py::arg("input"), py::arg("dim") = py::none());
    m.def("unsqueeze", &tenzor::unsqueeze, "Add dimension of size 1",
         py::arg("input"), py::arg("dim"));
    m.def("flatten", &tenzor::flatten, "Flatten tensor",
         py::arg("input"),
         py::arg("start_dim") = 0,
         py::arg("end_dim") = -1);
    m.def("contiguous", &tenzor::contiguous, "Make tensor contiguous");

    // Indexing operations
    // Cast to the tensor-level slice function to avoid ambiguity with autograd::slice
    m.def("slice", static_cast<tenzor::Tensor(*)(const tenzor::Tensor&, int64_t, int64_t, int64_t, int64_t)>(&tenzor::slice),
         "Slice tensor along dimension",
         py::arg("input"), py::arg("dim"), py::arg("start"), py::arg("end"),
         py::arg("step") = 1);
    m.def("index_select", &tenzor::index_select, "Select indices along dimension",
         py::arg("input"), py::arg("dim"), py::arg("index"));
    m.def("gather", &tenzor::gather, "Gather elements along dimension",
         py::arg("input"), py::arg("dim"), py::arg("index"));
    m.def("scatter", &tenzor::scatter, "Scatter elements along dimension",
         py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("src"));
    m.def("masked_select", &tenzor::masked_select, "Select elements where mask is true",
         py::arg("input"), py::arg("mask"));
    m.def("masked_fill", &tenzor::masked_fill, "Fill elements with value where mask is true",
         py::arg("input"), py::arg("mask"), py::arg("value"));
    m.def("where", &tenzor::where, "Conditional element selection",
         py::arg("condition"), py::arg("x"), py::arg("y"));
    m.def("take", &tenzor::take, "Take elements from flattened tensor",
         py::arg("input"), py::arg("index"));
    m.def("put", &tenzor::put, "Put elements into flattened tensor",
         py::arg("input"), py::arg("index"), py::arg("source"));

    // Advanced operations (Phase 6)
    // expand is already in transform operations
    m.def("topk", [](const tenzor::Tensor& input, int64_t k, int64_t dim, bool largest, bool sorted) {
        return tenzor::topk(input, k, dim, largest, sorted);
    }, "Find top k elements",
         py::arg("input"), py::arg("k"), py::arg("dim") = -1,
         py::arg("largest") = true, py::arg("sorted") = true);
    m.def("sort", [](const tenzor::Tensor& input, int64_t dim, bool descending) {
        return tenzor::sort(input, dim, descending);
    }, "Sort tensor along dimension",
         py::arg("input"), py::arg("dim") = -1, py::arg("descending") = false);
    m.def("unique", [](const tenzor::Tensor& input, bool sorted, bool return_inverse, bool return_counts) {
        return tenzor::unique(input, sorted, return_inverse, return_counts);
    }, "Find unique elements",
         py::arg("input"), py::arg("sorted") = true,
         py::arg("return_inverse") = false, py::arg("return_counts") = false);
    m.def("cumsum", [](const tenzor::Tensor& input, int64_t dim) {
        return tenzor::cumsum(input, dim);
    }, "Cumulative sum",
         py::arg("input"), py::arg("dim"));
    m.def("cumprod", [](const tenzor::Tensor& input, int64_t dim) {
        return tenzor::cumprod(input, dim);
    }, "Cumulative product",
         py::arg("input"), py::arg("dim"));

    // argmax and argmin (already declared in reduction.hpp)
    m.def("argmax", &tenzor::argmax, "Indices of maximum values",
         py::arg("input"), py::arg("dim") = py::none(), py::arg("keepdim") = false);
    m.def("argmin", &tenzor::argmin, "Indices of minimum values",
         py::arg("input"), py::arg("dim") = py::none(), py::arg("keepdim") = false);

    // Autograd
    py::class_<tenzor::Variable, std::shared_ptr<tenzor::Variable>>(m, "Variable")
        .def(py::init([](tenzor::Tensor data, bool requires_grad) {
            return std::make_shared<tenzor::Variable>(data, requires_grad);
        }), py::arg("data"), py::arg("requires_grad") = false)
        .def("backward", &tenzor::Variable::backward,
             py::arg("gradient") = py::none(),
             py::arg("retain_graph") = false,
             "Compute gradients via backpropagation")
        .def_property_readonly("data", py::overload_cast<>(&tenzor::Variable::tensor, py::const_))
        .def_property_readonly("grad", py::overload_cast<>(&tenzor::Variable::grad, py::const_))
        .def_property_readonly("grad_fn", &tenzor::Variable::grad_fn,
             "Get gradient function that created this variable")
        .def_property_readonly("is_leaf", &tenzor::Variable::is_leaf,
             "Check if variable is a leaf node")
        .def("register_hook", &tenzor::Variable::register_hook,
             py::arg("hook"),
             "Register a backward hook function")
        .def("retain_grad", &tenzor::Variable::retain_grad,
             "Enable gradient retention for non-leaf variables")
        .def_property_readonly("retains_grad", &tenzor::Variable::retains_grad,
             "Check if variable retains gradient");

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
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, bool>(),
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
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double, bool>(),
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
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double, bool>(),
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
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double, bool>(),
             py::arg("params"), py::arg("lr") = 0.01, py::arg("alpha") = 0.99,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             py::arg("momentum") = 0.0, py::arg("centered") = false)
        .def("step", &tenzor::optim::RMSprop::step)
        .def("zero_grad", &tenzor::optim::RMSprop::zero_grad)
        .def("state_dict", &tenzor::optim::RMSprop::state_dict)
        .def("load_state_dict", &tenzor::optim::RMSprop::load_state_dict);

    py::class_<tenzor::optim::Adagrad>(optim, "Adagrad")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 0.01, py::arg("lr_decay") = 0.0,
             py::arg("weight_decay") = 0.0, py::arg("initial_accumulator_value") = 0.0,
             py::arg("eps") = 1e-10)
        .def("step", &tenzor::optim::Adagrad::step)
        .def("zero_grad", &tenzor::optim::Adagrad::zero_grad);

    py::class_<tenzor::optim::Adadelta>(optim, "Adadelta")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double>(),
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

    // Distributed training
    auto distributed = nn.def_submodule("parallel", "Distributed and parallel training");

    // ProcessGroup
    py::class_<tenzor::nn::ProcessGroup, std::shared_ptr<tenzor::nn::ProcessGroup>>(distributed, "ProcessGroup")
        .def(py::init<int, int, const std::string&>(),
             py::arg("rank"), py::arg("world_size"), py::arg("backend") = "nccl",
             "Create a process group for distributed training")
        .def_property_readonly("rank", &tenzor::nn::ProcessGroup::rank,
             "Get process rank")
        .def_property_readonly("world_size", &tenzor::nn::ProcessGroup::world_size,
             "Get world size (total number of processes)")
        .def_property_readonly("backend", &tenzor::nn::ProcessGroup::backend,
             "Get backend name")
        .def("barrier", &tenzor::nn::ProcessGroup::barrier,
             "Synchronize all processes");

    // DistributedDataParallel
    py::class_<tenzor::nn::DistributedDataParallel, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::DistributedDataParallel>>(distributed, "DistributedDataParallel")
        .def(py::init<std::shared_ptr<tenzor::nn::Module>,
                     std::shared_ptr<tenzor::nn::ProcessGroup>,
                     std::vector<int>, int, bool, bool, bool, size_t>(),
             py::arg("module"),
             py::arg("process_group"),
             py::arg("device_ids") = std::vector<int>{},
             py::arg("output_device") = -1,
             py::arg("broadcast_buffers") = true,
             py::arg("find_unused_parameters") = false,
             py::arg("gradient_as_bucket_view") = false,
             py::arg("bucket_size_mb") = 25,
             "Wrap module for distributed data parallel training")
        .def("forward", &tenzor::nn::DistributedDataParallel::forward,
             py::arg("input"),
             "Forward pass with automatic gradient synchronization")
        .def_property_readonly("module", &tenzor::nn::DistributedDataParallel::module,
             "Get underlying module")
        .def_property_readonly("process_group", &tenzor::nn::DistributedDataParallel::process_group,
             "Get process group")
        .def_property_readonly("device_ids", &tenzor::nn::DistributedDataParallel::device_ids,
             "Get local device IDs")
        .def_property_readonly("output_device", &tenzor::nn::DistributedDataParallel::output_device,
             "Get master device ID")
        .def("join", &tenzor::nn::DistributedDataParallel::join,
             "Wait for all processes to finish current iteration");

    // Helper functions
    distributed.def("init_process_group", &tenzor::nn::init_process_group,
         py::arg("backend") = "nccl",
         "Initialize distributed training environment from environment variables");

    distributed.def("destroy_process_group", &tenzor::nn::destroy_process_group,
         py::arg("process_group"),
         "Destroy process group and cleanup resources");

    // ModelHub for pretrained weight management
    auto models = m.def_submodule("models", "Pretrained model hub");

    // HubConfig
    py::class_<tenzor::models::HubConfig>(models, "HubConfig")
        .def(py::init<>())
        .def_readwrite("cache_dir", &tenzor::models::HubConfig::cache_dir,
             "Cache directory path")
        .def_readwrite("max_cache_size", &tenzor::models::HubConfig::max_cache_size,
             "Maximum cache size in bytes (0 = unlimited)")
        .def_readwrite("verify_checksums", &tenzor::models::HubConfig::verify_checksums,
             "Whether to verify SHA256 checksums")
        .def_readwrite("resume_downloads", &tenzor::models::HubConfig::resume_downloads,
             "Whether to resume interrupted downloads")
        .def_readwrite("connection_timeout", &tenzor::models::HubConfig::connection_timeout,
             "Connection timeout in seconds")
        .def_readwrite("max_retries", &tenzor::models::HubConfig::max_retries,
             "Maximum number of download retries")
        .def_readwrite("show_progress", &tenzor::models::HubConfig::show_progress,
             "Whether to show progress by default");

    // ModelWeightInfo
    py::class_<tenzor::models::ModelWeightInfo>(models, "ModelWeightInfo")
        .def(py::init<>())
        .def_readwrite("name", &tenzor::models::ModelWeightInfo::name,
             "Model name")
        .def_readwrite("url", &tenzor::models::ModelWeightInfo::url,
             "Download URL")
        .def_readwrite("sha256", &tenzor::models::ModelWeightInfo::sha256,
             "Expected SHA256 checksum")
        .def_readwrite("size", &tenzor::models::ModelWeightInfo::size,
             "File size in bytes")
        .def_readwrite("description", &tenzor::models::ModelWeightInfo::description,
             "Model description");

    // DownloadStats
    py::class_<tenzor::models::DownloadStats>(models, "DownloadStats")
        .def_readonly("total_bytes", &tenzor::models::DownloadStats::total_bytes,
             "Total bytes downloaded")
        .def_readonly("bytes_downloaded", &tenzor::models::DownloadStats::bytes_downloaded,
             "Bytes downloaded in this session")
        .def_readonly("download_time", &tenzor::models::DownloadStats::download_time,
             "Time taken in seconds")
        .def_readonly("average_speed", &tenzor::models::DownloadStats::average_speed,
             "Average speed in bytes/sec")
        .def_readonly("resumed", &tenzor::models::DownloadStats::resumed,
             "Whether download was resumed")
        .def_readonly("verified", &tenzor::models::DownloadStats::verified,
             "Whether checksum was verified");

    // ModelHub
    py::class_<tenzor::models::ModelHub>(models, "Hub")
        .def_static("download_weights",
             [](const std::string& model_name,
                const std::string& url,
                const std::string& expected_sha256,
                bool show_progress,
                py::object progress_callback) {
                 tenzor::models::ProgressCallback callback = nullptr;
                 if (!progress_callback.is_none()) {
                     callback = [progress_callback](size_t downloaded, size_t total,
                                                    double speed, double eta) {
                         try {
                             progress_callback(downloaded, total, speed, eta);
                         } catch (const py::error_already_set& e) {
                             std::cerr << "Error in progress callback: " << e.what() << std::endl;
                         }
                     };
                 }
                 return tenzor::models::ModelHub::download_weights(
                     model_name, url, expected_sha256, show_progress, callback);
             },
             py::arg("model_name"),
             py::arg("url"),
             py::arg("expected_sha256") = "",
             py::arg("show_progress") = true,
             py::arg("progress_callback") = py::none(),
             "Download pretrained weights from URL")
        .def_static("download_pretrained",
             [](const std::string& model_name,
                bool show_progress,
                py::object progress_callback) {
                 tenzor::models::ProgressCallback callback = nullptr;
                 if (!progress_callback.is_none()) {
                     callback = [progress_callback](size_t downloaded, size_t total,
                                                    double speed, double eta) {
                         try {
                             progress_callback(downloaded, total, speed, eta);
                         } catch (const py::error_already_set& e) {
                             std::cerr << "Error in progress callback: " << e.what() << std::endl;
                         }
                     };
                 }
                 return tenzor::models::ModelHub::download_pretrained(
                     model_name, show_progress, callback);
             },
             py::arg("model_name"),
             py::arg("show_progress") = true,
             py::arg("progress_callback") = py::none(),
             "Download registered pretrained model")
        .def_static("load_pretrained_weights",
             &tenzor::models::ModelHub::load_pretrained_weights,
             py::arg("model"),
             py::arg("weights_path"),
             py::arg("strict") = true,
             "Load pretrained weights into model")
        .def_static("set_cache_dir",
             &tenzor::models::ModelHub::set_cache_dir,
             py::arg("path"),
             "Set cache directory")
        .def_static("get_cache_dir",
             &tenzor::models::ModelHub::get_cache_dir,
             "Get cache directory")
        .def_static("set_config",
             &tenzor::models::ModelHub::set_config,
             py::arg("config"),
             "Set ModelHub configuration")
        .def_static("get_config",
             &tenzor::models::ModelHub::get_config,
             "Get ModelHub configuration")
        .def_static("clear_cache",
             &tenzor::models::ModelHub::clear_cache,
             "Clear all cached weights")
        .def_static("cache_size",
             &tenzor::models::ModelHub::cache_size,
             "Get total cache size in bytes")
        .def_static("list_cached_models",
             &tenzor::models::ModelHub::list_cached_models,
             "List cached model names")
        .def_static("is_cached",
             &tenzor::models::ModelHub::is_cached,
             py::arg("model_name"),
             "Check if model is cached")
        .def_static("get_cached_path",
             &tenzor::models::ModelHub::get_cached_path,
             py::arg("model_name"),
             "Get cached weights path")
        .def_static("register_model",
             &tenzor::models::ModelHub::register_model,
             py::arg("info"),
             "Register a model in the hub")
        .def_static("register_models",
             &tenzor::models::ModelHub::register_models,
             py::arg("models"),
             "Register multiple models")
        .def_static("get_model_info",
             &tenzor::models::ModelHub::get_model_info,
             py::arg("model_name"),
             "Get registered model info")
        .def_static("list_registered_models",
             &tenzor::models::ModelHub::list_registered_models,
             "List all registered models")
        .def_static("is_registered",
             &tenzor::models::ModelHub::is_registered,
             py::arg("model_name"),
             "Check if model is registered")
        .def_static("remove_from_cache",
             &tenzor::models::ModelHub::remove_from_cache,
             py::arg("model_name"),
             "Remove model from cache")
        .def_static("get_last_download_stats",
             &tenzor::models::ModelHub::get_last_download_stats,
             "Get statistics for last download")
        .def_static("verify_checksum",
             &tenzor::models::ModelHub::verify_checksum,
             py::arg("file_path"),
             py::arg("expected_sha256"),
             "Verify file checksum")
        .def_static("compute_checksum",
             &tenzor::models::ModelHub::compute_checksum,
             py::arg("file_path"),
             "Compute SHA256 checksum of file")
        .def_static("clean_cache",
             &tenzor::models::ModelHub::clean_cache,
             py::arg("max_size"),
             "Clean cache to fit within size limit");

    // Helper function for loading pretrained models
    models.def("load_pretrained",
        [](tenzor::nn::Module& model, const std::string& model_name,
           bool show_progress, bool strict) {
            std::string weights_path = tenzor::models::ModelHub::download_pretrained(
                model_name, show_progress);
            tenzor::models::ModelHub::load_pretrained_weights(model, weights_path, strict);
        },
        py::arg("model"),
        py::arg("model_name"),
        py::arg("show_progress") = true,
        py::arg("strict") = true,
        "Download and load pretrained weights into model");

    // PyTorch interoperability (optional - requires torch headers)
    #ifdef TENZOR_HAS_TORCH
    auto torch_mod = m.def_submodule("torch_interop", "PyTorch tensor interoperability");

    torch_mod.def("can_zero_copy_to_torch", &tenzor::torch_interop::can_zero_copy_to_torch,
                  "Check if zero-copy conversion to PyTorch is possible");
    torch_mod.def("tensor_to_torch", &tenzor::torch_interop::tensor_to_torch,
                  py::arg("tensor"), py::arg("requires_grad") = false,
                  "Convert Tenzor tensor to PyTorch tensor");
    torch_mod.def("tensor_from_torch", &tenzor::torch_interop::tensor_from_torch,
                  py::arg("torch_tensor"), py::arg("device") = py::none(),
                  "Convert PyTorch tensor to Tenzor tensor");
    #endif

    // ONNX export/import functionality
    auto onnx_mod = m.def_submodule("onnx", "ONNX model export and import");

    // ONNX data type enum
    py::enum_<tenzor::onnx::ONNXDataType>(onnx_mod, "DataType")
        .value("UNDEFINED", tenzor::onnx::ONNXDataType::UNDEFINED)
        .value("FLOAT", tenzor::onnx::ONNXDataType::FLOAT)
        .value("UINT8", tenzor::onnx::ONNXDataType::UINT8)
        .value("INT8", tenzor::onnx::ONNXDataType::INT8)
        .value("UINT16", tenzor::onnx::ONNXDataType::UINT16)
        .value("INT16", tenzor::onnx::ONNXDataType::INT16)
        .value("INT32", tenzor::onnx::ONNXDataType::INT32)
        .value("INT64", tenzor::onnx::ONNXDataType::INT64)
        .value("STRING", tenzor::onnx::ONNXDataType::STRING)
        .value("BOOL", tenzor::onnx::ONNXDataType::BOOL)
        .value("FLOAT16", tenzor::onnx::ONNXDataType::FLOAT16)
        .value("DOUBLE", tenzor::onnx::ONNXDataType::DOUBLE)
        .value("UINT32", tenzor::onnx::ONNXDataType::UINT32)
        .value("UINT64", tenzor::onnx::ONNXDataType::UINT64)
        .value("COMPLEX64", tenzor::onnx::ONNXDataType::COMPLEX64)
        .value("COMPLEX128", tenzor::onnx::ONNXDataType::COMPLEX128)
        .value("BFLOAT16", tenzor::onnx::ONNXDataType::BFLOAT16);

    // ONNXTensor class
    py::class_<tenzor::onnx::ONNXTensor>(onnx_mod, "Tensor")
        .def(py::init<const tenzor::Tensor&, const std::string&>(),
             py::arg("tensor"), py::arg("name"))
        .def_readwrite("name", &tenzor::onnx::ONNXTensor::name)
        .def_readwrite("dtype", &tenzor::onnx::ONNXTensor::dtype)
        .def_readwrite("dims", &tenzor::onnx::ONNXTensor::dims)
        .def_readwrite("raw_data", &tenzor::onnx::ONNXTensor::raw_data)
        .def("numel", &tenzor::onnx::ONNXTensor::numel,
             "Get total number of elements")
        .def("size_bytes", &tenzor::onnx::ONNXTensor::size_bytes,
             "Get size in bytes");

    // ONNXValueInfo class
    py::class_<tenzor::onnx::ONNXValueInfo>(onnx_mod, "ValueInfo")
        .def(py::init<const std::string&, tenzor::onnx::ONNXDataType, const std::vector<int64_t>&>(),
             py::arg("name"), py::arg("dtype"), py::arg("shape"))
        .def_readwrite("name", &tenzor::onnx::ONNXValueInfo::name)
        .def_readwrite("dtype", &tenzor::onnx::ONNXValueInfo::dtype)
        .def_readwrite("shape", &tenzor::onnx::ONNXValueInfo::shape);

    // ONNXNode class
    py::class_<tenzor::onnx::ONNXNode>(onnx_mod, "Node")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("op_type"), py::arg("name"))
        .def_readwrite("op_type", &tenzor::onnx::ONNXNode::op_type)
        .def_readwrite("name", &tenzor::onnx::ONNXNode::name)
        .def_readwrite("inputs", &tenzor::onnx::ONNXNode::inputs)
        .def_readwrite("outputs", &tenzor::onnx::ONNXNode::outputs)
        .def("add_input", &tenzor::onnx::ONNXNode::add_input, py::arg("input"))
        .def("add_output", &tenzor::onnx::ONNXNode::add_output, py::arg("output"));

    // ONNXGraph class
    py::class_<tenzor::onnx::ONNXGraph>(onnx_mod, "Graph")
        .def(py::init<const std::string&>(), py::arg("name") = "graph")
        .def_readwrite("name", &tenzor::onnx::ONNXGraph::name)
        .def_readwrite("nodes", &tenzor::onnx::ONNXGraph::nodes)
        .def_readwrite("inputs", &tenzor::onnx::ONNXGraph::inputs)
        .def_readwrite("outputs", &tenzor::onnx::ONNXGraph::outputs)
        .def_readwrite("initializers", &tenzor::onnx::ONNXGraph::initializers)
        .def("add_node", &tenzor::onnx::ONNXGraph::add_node, py::arg("node"))
        .def("add_input", &tenzor::onnx::ONNXGraph::add_input, py::arg("input"))
        .def("add_output", &tenzor::onnx::ONNXGraph::add_output, py::arg("output"))
        .def("add_initializer", &tenzor::onnx::ONNXGraph::add_initializer, py::arg("tensor"))
        .def("add_value_info", &tenzor::onnx::ONNXGraph::add_value_info, py::arg("info"))
        .def("get_unique_name", &tenzor::onnx::ONNXGraph::get_unique_name, py::arg("prefix"));

    // ONNXExporter class
    py::class_<tenzor::onnx::ONNXExporter>(onnx_mod, "Exporter")
        .def(py::init<int64_t>(), py::arg("opset_version") = 13,
             "Create ONNX exporter with specified opset version")
        .def("set_model_name", &tenzor::onnx::ONNXExporter::set_model_name,
             py::arg("name"), "Set model name")
        .def("set_opset_version", &tenzor::onnx::ONNXExporter::set_opset_version,
             py::arg("version"), "Set ONNX opset version")
        .def("set_description", &tenzor::onnx::ONNXExporter::set_description,
             py::arg("desc"), "Set model description")
        .def("set_producer_name", &tenzor::onnx::ONNXExporter::set_producer_name,
             py::arg("name"), "Set producer name")
        .def("set_model_version", &tenzor::onnx::ONNXExporter::set_model_version,
             py::arg("version"), "Set model version")
        .def("add_input", &tenzor::onnx::ONNXExporter::add_input,
             py::arg("tensor"), py::arg("name"),
             py::arg("dynamic_axes") = std::unordered_map<int64_t, std::string>(),
             "Add model input")
        .def("add_output", &tenzor::onnx::ONNXExporter::add_output,
             py::arg("tensor"), py::arg("name"),
             "Add model output")
        .def("export_to_file", &tenzor::onnx::ONNXExporter::export_to_file,
             py::arg("filepath"),
             "Export model to ONNX file")
        .def("export_to_bytes", &tenzor::onnx::ONNXExporter::export_to_bytes,
             "Export model to ONNX bytes")
        .def("get_graph", &tenzor::onnx::ONNXExporter::get_graph,
             py::return_value_policy::reference_internal,
             "Get the ONNX graph")
        .def("clear", &tenzor::onnx::ONNXExporter::clear,
             "Clear the exporter state");

    // High-level export function
    onnx_mod.def("export",
        [](std::shared_ptr<tenzor::nn::Module> module,
           const tenzor::Tensor& dummy_input,
           const std::string& filepath,
           const std::vector<std::string>& input_names,
           const std::vector<std::string>& output_names,
           int64_t opset_version,
           bool verbose) {
            if (verbose) {
                std::cout << "Exporting model to ONNX format..." << std::endl;
                std::cout << "  Output file: " << filepath << std::endl;
                std::cout << "  Opset version: " << opset_version << std::endl;
                std::cout << "  Input names: ";
                for (const auto& name : input_names) std::cout << name << " ";
                std::cout << std::endl;
                std::cout << "  Output names: ";
                for (const auto& name : output_names) std::cout << name << " ";
                std::cout << std::endl;
            }

            tenzor::onnx::export_to_onnx(module, dummy_input, filepath,
                                        input_names, output_names, opset_version);

            if (verbose) {
                std::cout << "Model exported successfully!" << std::endl;
            }
        },
        py::arg("module"),
        py::arg("dummy_input"),
        py::arg("filepath"),
        py::arg("input_names") = std::vector<std::string>{"input"},
        py::arg("output_names") = std::vector<std::string>{"output"},
        py::arg("opset_version") = 13,
        py::arg("verbose") = false,
        R"pbdoc(
            Export a Tenzor module to ONNX format.

            Args:
                module: The neural network module to export
                dummy_input: Example input tensor for shape inference
                filepath: Output ONNX file path
                input_names: List of input names (default: ["input"])
                output_names: List of output names (default: ["output"])
                opset_version: ONNX opset version (default: 13)
                verbose: Print export progress (default: False)

            Example:
                >>> model = tenzor.nn.Linear(10, 5)
                >>> dummy = tenzor.Tensor([1, 10], dtype=tenzor.dtype.float32)
                >>> tenzor.onnx.export(model, dummy, "model.onnx", verbose=True)
        )pbdoc");

    // Utility function to convert DType to ONNX DataType
    onnx_mod.def("dtype_to_onnx", &tenzor::onnx::dtype_to_onnx,
                 py::arg("dtype"),
                 "Convert Tenzor DType to ONNX DataType");
}

