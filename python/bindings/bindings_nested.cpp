/**
 * @file bindings_nested.cpp
 * @brief Python bindings for NestedTensor and nested operations
 *
 * Exposes NestedTensor class, factory functions, element-wise and
 * offset-aware operations to Python via pybind11.
 */

#include "register.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

#include <tenzor/nested/nested_tensor.hpp>
#include <tenzor/nested/nested_ops.hpp>
#include <tenzor/ops/creation.hpp>

namespace py = pybind11;

namespace tenzor::python {

void register_nested(py::module_& m) {
    auto nested_mod = m.def_submodule("nested", "Nested (jagged) tensor support");

    // =========================================================================
    // NestedTensor class
    // =========================================================================
    py::class_<NestedTensor>(nested_mod, "NestedTensor",
        "Nested tensor with jagged (variable-length) layout.\n\n"
        "Stores a batch of tensors with variable sizes along one ragged\n"
        "dimension using a contiguous values buffer and cumulative offsets.")

        // Factory methods
        .def_static("from_tensor_list",
            [](std::vector<Tensor> tensors) {
                return NestedTensor::from_tensor_list(tensors);
            },
            py::arg("tensors"),
            "Create NestedTensor from a list of tensors")

        .def_static("from_padded",
            &NestedTensor::from_padded,
            py::arg("padded"), py::arg("lengths"),
            "Create NestedTensor from a padded tensor and lengths")

        .def_static("from_jagged",
            &NestedTensor::from_jagged,
            py::arg("values"), py::arg("offsets"),
            py::arg("ragged_dim") = 1,
            "Create NestedTensor from pre-existing values and offsets")

        // Properties
        .def_property_readonly("is_nested", &NestedTensor::is_nested)
        .def_property_readonly("batch_size", &NestedTensor::batch_size)
        .def_property_readonly("ragged_dim", &NestedTensor::ragged_dim)
        .def_property_readonly("values", &NestedTensor::values)
        .def_property_readonly("offsets", &NestedTensor::offsets)
        .def_property_readonly("dtype",
            [](const NestedTensor& self) { return self.dtype(); })
        .def_property_readonly("device",
            [](const NestedTensor& self) { return self.device(); })
        .def_property_readonly("ndim", &NestedTensor::ndim)
        .def_property_readonly("numel", &NestedTensor::numel)

        .def("lengths", &NestedTensor::lengths,
             "Per-element lengths along the ragged dimension")
        .def("nested_sizes", &NestedTensor::nested_sizes,
             "Per-element shapes as list of lists")
        .def("max_length", &NestedTensor::max_length,
             "Maximum length along the ragged dimension")

        // Conversion
        .def("to_padded_tensor", &NestedTensor::to_padded_tensor,
             py::arg("padding_value") = 0.0,
             "Convert to padded dense tensor")
        .def("unbind", &NestedTensor::unbind,
             "Unbind into individual tensors")
        .def("select", &NestedTensor::select,
             py::arg("index"),
             "Select a single element from the batch")

        // Device / dtype
        .def("to",
            [](const NestedTensor& self, Device device) {
                return self.to(device);
            },
            py::arg("device"), "Transfer to device")
        .def("to_dtype",
            [](const NestedTensor& self, DType dtype) {
                return self.to(dtype);
            },
            py::arg("dtype"), "Convert to dtype")
        .def("contiguous", &NestedTensor::contiguous)
        .def("clone", &NestedTensor::clone)

        // Gradient
        .def_property_readonly("requires_grad", &NestedTensor::requires_grad)
        .def("requires_grad_", &NestedTensor::requires_grad_,
             py::arg("requires_grad") = true)

        // Operator overloads
        .def("__add__", [](const NestedTensor& a, const NestedTensor& b) {
            return nested_add(a, b);
        })
        .def("__sub__", [](const NestedTensor& a, const NestedTensor& b) {
            return nested_sub(a, b);
        })
        .def("__mul__", [](const NestedTensor& a, const NestedTensor& b) {
            return nested_mul(a, b);
        })
        .def("__truediv__", [](const NestedTensor& a, const NestedTensor& b) {
            return nested_div(a, b);
        })
        .def("__neg__", [](const NestedTensor& a) {
            return nested_neg(a);
        })
        .def("__add__", [](const NestedTensor& a, double scalar) {
            return nested_add_scalar(a, scalar);
        })
        .def("__radd__", [](const NestedTensor& a, double scalar) {
            return nested_add_scalar(a, scalar);
        })
        .def("__mul__", [](const NestedTensor& a, double scalar) {
            return nested_mul_scalar(a, scalar);
        })
        .def("__rmul__", [](const NestedTensor& a, double scalar) {
            return nested_mul_scalar(a, scalar);
        })

        // Indexing
        .def("__getitem__", &NestedTensor::select)
        .def("__len__", &NestedTensor::batch_size)

        // String representation
        .def("__repr__", [](const NestedTensor& nt) {
            std::ostringstream os;
            os << "NestedTensor(batch_size=" << nt.batch_size()
               << ", numel=" << nt.numel()
               << ", dtype=" << tenzor::dtype_name(nt.dtype())
               << ", device=" << nt.device().to_string() << ")";
            return os.str();
        })
    ;

    // =========================================================================
    // Module-level factory function
    // =========================================================================
    nested_mod.def("nested_tensor",
        [](std::vector<Tensor> tensors) {
            return tenzor::nested_tensor(std::move(tensors));
        },
        py::arg("tensors"),
        "Create a NestedTensor from a list of tensors");

    // =========================================================================
    // Module-level operations
    // =========================================================================
    nested_mod.def("add", &nested_add, py::arg("a"), py::arg("b"));
    nested_mod.def("sub", &nested_sub, py::arg("a"), py::arg("b"));
    nested_mod.def("mul", &nested_mul, py::arg("a"), py::arg("b"));
    nested_mod.def("div", &nested_div, py::arg("a"), py::arg("b"));
    nested_mod.def("neg", &nested_neg, py::arg("a"));
    nested_mod.def("relu", &nested_relu, py::arg("a"));
    nested_mod.def("gelu", &nested_gelu, py::arg("a"));
    nested_mod.def("sigmoid", &nested_sigmoid, py::arg("a"));
    nested_mod.def("tanh", &nested_tanh, py::arg("a"));
    nested_mod.def("abs", &nested_abs, py::arg("a"));
    nested_mod.def("add_scalar", &nested_add_scalar,
                   py::arg("a"), py::arg("scalar"));
    nested_mod.def("mul_scalar", &nested_mul_scalar,
                   py::arg("a"), py::arg("scalar"));

    nested_mod.def("softmax", &nested_softmax,
                   py::arg("input"), py::arg("dim"));
    nested_mod.def("log_softmax", &nested_log_softmax,
                   py::arg("input"), py::arg("dim"));
    nested_mod.def("layer_norm", &nested_layer_norm,
                   py::arg("input"), py::arg("weight"), py::arg("bias"),
                   py::arg("eps") = 1e-5);
    nested_mod.def("sum", &nested_sum,
                   py::arg("input"), py::arg("dim"),
                   py::arg("keepdim") = false);
    nested_mod.def("mean", &nested_mean,
                   py::arg("input"), py::arg("dim"),
                   py::arg("keepdim") = false);

    nested_mod.def("linear", [](const NestedTensor& input, const Tensor& weight,
                                std::optional<Tensor> bias) {
        return nested_linear(input, weight,
                             bias.has_value() ? &bias.value() : nullptr);
    }, py::arg("input"), py::arg("weight"), py::arg("bias") = py::none());

    nested_mod.def("matmul", &nested_matmul,
                   py::arg("a"), py::arg("b"));
    nested_mod.def("attention", &nested_attention,
                   py::arg("query"), py::arg("key"), py::arg("value"),
                   py::arg("scale") = -1.0, py::arg("causal") = false);

    nested_mod.def("cat", [](std::vector<NestedTensor> tensors, int64_t dim) {
        return nested_cat(tensors, dim);
    }, py::arg("tensors"), py::arg("dim") = 0);

    nested_mod.def("dropout", &nested_dropout,
                   py::arg("input"), py::arg("p"), py::arg("training"));
}

} // namespace tenzor::python
