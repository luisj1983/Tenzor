// tenzor.autograd + tenzor.func Python bindings. Extracted from
// python/bindings.cpp as part of P3.4.
//
// Covers: autograd submodule (custom Function, grad, make_dot,
// optimize_graph) and func submodule (grad, vmap, jacrev, jacfwd,
// hessian, jvp — the torch.func composable-transforms equivalent).

#include "register.hpp"

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <tenzor/autograd/function.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/functional.hpp>
#include <tenzor/autograd/graph_optimizer.hpp>
#include <tenzor/autograd/graph_viz.hpp>
#include <tenzor/autograd/vmap.hpp>
#include <tenzor/core/tensor.hpp>

namespace py = pybind11;

// Forward-declare the trampoline that lives in bindings.cpp (it needs
// the Python module scope, so it stays there; we just reference its type).
// PyCustomFunction is defined in bindings.cpp — we forward-declare here
// because the autograd_mod registers it.  If it moves to a shared header
// later, drop this.
class PyCustomFunction;

namespace tenzor::python {

void register_autograd(py::module_& m) {
    auto autograd_mod = m.def_submodule("autograd", "Autograd components");

    // autograd.grad() — functional gradient computation
    autograd_mod.def("grad", [](
        py::object outputs_obj,
        py::object inputs_obj,
        py::object grad_outputs_obj,
        bool retain_graph,
        bool create_graph
    ) -> py::tuple {
        std::vector<tenzor::Variable> outputs;
        if (py::isinstance<tenzor::Variable>(outputs_obj)) {
            outputs.push_back(outputs_obj.cast<tenzor::Variable>());
        } else {
            for (auto o : outputs_obj.cast<py::sequence>()) {
                outputs.push_back(o.cast<tenzor::Variable>());
            }
        }

        std::vector<tenzor::Variable> inputs;
        if (py::isinstance<tenzor::Variable>(inputs_obj)) {
            inputs.push_back(inputs_obj.cast<tenzor::Variable>());
        } else {
            for (auto i : inputs_obj.cast<py::sequence>()) {
                inputs.push_back(i.cast<tenzor::Variable>());
            }
        }

        for (auto& inp : inputs) {
            inp.zero_grad();
            inp.retain_grad();
        }

        for (size_t i = 0; i < outputs.size(); ++i) {
            std::optional<tenzor::Tensor> grad_out;
            if (!grad_outputs_obj.is_none()) {
                auto grad_outputs = grad_outputs_obj.cast<py::sequence>();
                if (i < static_cast<size_t>(py::len(grad_outputs))) {
                    auto g = grad_outputs[i];
                    if (!g.is_none()) {
                        grad_out = g.cast<tenzor::Tensor>();
                    }
                }
            }
            bool retain = retain_graph || create_graph || (i < outputs.size() - 1);
            outputs[i].backward(grad_out, retain, create_graph);
        }

        py::tuple result(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i].has_grad()) {
                result[i] = py::cast(inputs[i].grad());
            } else {
                result[i] = py::none();
            }
        }
        return result;
    },
    "Compute gradients of outputs w.r.t. inputs",
    py::arg("outputs"),
    py::arg("inputs"),
    py::arg("grad_outputs") = py::none(),
    py::arg("retain_graph") = false,
    py::arg("create_graph") = false);

    autograd_mod.def("make_dot", [](const tenzor::Variable& root,
                                     py::dict params_dict) -> std::string {
        std::unordered_map<std::string, tenzor::Variable> params;
        for (auto& [key, val] : params_dict) {
            params[key.cast<std::string>()] = val.cast<tenzor::Variable>();
        }
        return tenzor::make_dot(root, params);
    },
    "Generate Graphviz DOT string for the computation graph",
    py::arg("root"),
    py::arg("params") = py::dict());

    autograd_mod.def("optimize_graph", [](tenzor::Variable& root) {
        tenzor::GraphOptimizer optimizer;
        auto stats = optimizer.optimize_variable(root);
        py::dict result;
        result["linear_relu_fused"] = stats.linear_relu_fused;
        result["conv_batchnorm_fused"] = stats.conv_batchnorm_fused;
        result["dead_nodes_removed"] = stats.dead_nodes_removed;
        result["total"] = stats.total();
        return result;
    },
    py::arg("root"),
    "Optimize the computation graph of a Variable. Returns optimization stats dict.");

    // Composable function transforms (torch.func equivalent)
    auto func_mod = m.def_submodule("func", "Composable function transforms");

    func_mod.def("grad", [](py::function f) {
        return py::cpp_function([f](const tenzor::Variable& x) -> tenzor::Variable {
            py::gil_scoped_acquire gil;
            tenzor::Variable x_copy(x.tensor().clone(), true);
            py::object result = f(x_copy);
            tenzor::Variable output = result.cast<tenzor::Variable>();
            output.backward();
            auto g = x_copy.grad();
            if (!g.has_value()) {
                throw std::runtime_error("grad: no gradient computed");
            }
            return tenzor::Variable(g.value(), false);
        });
    }, py::arg("f"),
    "Return a function that computes the gradient of f.");

    // 5th-audit B'5: every inner `cpp_fn` lambda below now acquires the GIL
    // before touching `f(input)` or `result.cast<...>()`. The outer cpp_function
    // already holds it on entry, but the underlying C++ transforms
    // (tenzor::jacobian / hessian / vmap / jvp / hvp / vhp / vjp) may release
    // the GIL across worker threads for performance; an inner re-entry into
    // pybind11 without GIL would crash. The `gil_scoped_acquire` inside the
    // inner lambda is a no-op when the GIL is already held (re-entry-safe)
    // and a real acquire on a child thread.

    func_mod.def("vmap", [](py::function f, int64_t in_dim, [[maybe_unused]] int64_t out_dim) {
        return py::cpp_function([f, in_dim](const tenzor::Variable& batched_input) -> tenzor::Variable {
            py::gil_scoped_acquire gil;
            auto cpp_fn = [&f](const tenzor::Variable& x) -> tenzor::Variable {
                py::gil_scoped_acquire inner_gil;
                py::object result = f(x);
                return result.cast<tenzor::Variable>();
            };
            return tenzor::vmap(cpp_fn, batched_input, in_dim);
        });
    }, py::arg("f"), py::arg("in_dim") = 0, py::arg("out_dim") = 0,
    "Return a vectorized version of f that maps over a batch dimension.");

    func_mod.def("jacrev", [](py::function f) {
        return py::cpp_function([f](const tenzor::Variable& x) -> tenzor::Variable {
            py::gil_scoped_acquire gil;
            auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
                py::gil_scoped_acquire inner_gil;
                py::object result = f(input);
                return result.cast<tenzor::Variable>();
            };
            tenzor::Tensor J = tenzor::jacobian(cpp_fn, x);
            return tenzor::Variable(J, false);
        });
    }, py::arg("f"),
    "Return a function that computes the reverse-mode Jacobian of f.");

    func_mod.def("jacfwd", [](py::function f) {
        return py::cpp_function([f](const tenzor::Variable& x) -> tenzor::Variable {
            py::gil_scoped_acquire gil;
            auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
                py::gil_scoped_acquire inner_gil;
                py::object result = f(input);
                return result.cast<tenzor::Variable>();
            };
            tenzor::Tensor J = tenzor::jacobian(cpp_fn, x);
            return tenzor::Variable(J, false);
        });
    }, py::arg("f"),
    "Return a function that computes the forward-mode Jacobian of f.");

    func_mod.def("hessian", [](py::function f) {
        return py::cpp_function([f](const tenzor::Variable& x) -> tenzor::Variable {
            py::gil_scoped_acquire gil;
            auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
                py::gil_scoped_acquire inner_gil;
                py::object result = f(input);
                return result.cast<tenzor::Variable>();
            };
            tenzor::Tensor H = tenzor::hessian(cpp_fn, x);
            return tenzor::Variable(H, false);
        });
    }, py::arg("f"),
    "Return a function that computes the Hessian of a scalar-valued f.");

    func_mod.def("jvp", [](py::function f, const tenzor::Variable& x,
                           const tenzor::Tensor& tangent) {
        py::gil_scoped_acquire gil;
        auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
            py::gil_scoped_acquire inner_gil;
            py::object result = f(input);
            return result.cast<tenzor::Variable>();
        };
        auto [out, tangent_out] = tenzor::jvp(cpp_fn, x, tangent);
        return py::make_tuple(out, tangent_out);
    }, py::arg("f"), py::arg("x"), py::arg("tangent"),
    "Forward-mode Jacobian-vector product. Returns (output, J_f(x) @ tangent).");

    func_mod.def("hvp", [](py::function f, const tenzor::Variable& x,
                           const tenzor::Tensor& v) {
        py::gil_scoped_acquire gil;
        auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
            py::gil_scoped_acquire inner_gil;
            py::object result = f(input);
            return result.cast<tenzor::Variable>();
        };
        auto [out, hvp_result] = tenzor::hvp(cpp_fn, x, v);
        return py::make_tuple(out, hvp_result);
    }, py::arg("f"), py::arg("x"), py::arg("v"),
    "Hessian-vector product. Returns (output, H_f(x) @ v).");

    func_mod.def("vhp", [](py::function f, const tenzor::Variable& x,
                           const tenzor::Tensor& v) {
        py::gil_scoped_acquire gil;
        auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
            py::gil_scoped_acquire inner_gil;
            py::object result = f(input);
            return result.cast<tenzor::Variable>();
        };
        auto [out, vhp_result] = tenzor::vhp(cpp_fn, x, v);
        return py::make_tuple(out, vhp_result);
    }, py::arg("f"), py::arg("x"), py::arg("v"),
    "Vector-Hessian product. Returns (output, v^T @ H_f(x)).");

    func_mod.def("vjp", [](py::function f, const tenzor::Variable& x,
                           const tenzor::Tensor& cotangent) {
        py::gil_scoped_acquire gil;
        auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
            py::gil_scoped_acquire inner_gil;
            py::object result = f(input);
            return result.cast<tenzor::Variable>();
        };
        auto [out, vjp_result] = tenzor::vjp(cpp_fn, x, cotangent);
        return py::make_tuple(out, vjp_result);
    }, py::arg("f"), py::arg("x"), py::arg("cotangent"),
    "Vector-Jacobian product (reverse-mode). Returns (output, v^T @ J_f(x)).");
}

} // namespace tenzor::python
