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
#include <tenzor/autograd/gradcheck.hpp>
#include <tenzor/autograd/graph_optimizer.hpp>
#include <tenzor/autograd/graph_viz.hpp>
#include <tenzor/autograd/vmap.hpp>
#include <tenzor/autograd/ops.hpp>  // tenzor::permute(Variable, ...) for vmap out_dim
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

        // R.22: pre-extract grad_outputs under the GIL, then release it across
        // the long-running C++ backward traversal. The engine reacquires the
        // GIL internally for any Python Function.backward callback.
        std::vector<std::optional<tenzor::Tensor>> grad_outs_vec(outputs.size());
        if (!grad_outputs_obj.is_none()) {
            auto grad_outputs = grad_outputs_obj.cast<py::sequence>();
            for (size_t i = 0; i < outputs.size(); ++i) {
                if (i < static_cast<size_t>(py::len(grad_outputs))) {
                    auto g = grad_outputs[i];
                    if (!g.is_none()) {
                        grad_outs_vec[i] = g.cast<tenzor::Tensor>();
                    }
                }
            }
        }

        // V.32: PyTorch's torch.autograd.grad() never touches caller .grad
        // or .retain_grad. Capture pre-call state so we can restore it after
        // the scratch backward.
        std::vector<std::optional<tenzor::Tensor>> saved_grads(inputs.size());
        std::vector<bool> saved_retains(inputs.size(), false);
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i].has_grad()) {
                saved_grads[i] = inputs[i].grad();
            }
            saved_retains[i] = inputs[i].retains_grad();
        }

        // Restore the caller's pre-call .grad and retain_grad flags. Must run
        // on every exit path, including when backward() throws — otherwise the
        // caller's grads are left zeroed/overwritten by the scratch backward.
        auto restore = [&]() {
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (saved_grads[i].has_value()) {
                    inputs[i].set_grad(saved_grads[i].value());
                } else {
                    inputs[i].zero_grad();
                }
                inputs[i].set_retain_grad(saved_retains[i]);
            }
        };

        py::tuple result(inputs.size());
        try {
            {
                py::gil_scoped_release release;
                for (auto& inp : inputs) {
                    inp.zero_grad();
                    inp.retain_grad();
                }

                for (size_t i = 0; i < outputs.size(); ++i) {
                    bool retain = retain_graph || create_graph || (i < outputs.size() - 1);
                    outputs[i].backward(grad_outs_vec[i], retain, create_graph);
                }
            }

            // Collect scratch grads into the result tuple (before restoring).
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (inputs[i].has_grad()) {
                    result[i] = py::cast(inputs[i].grad());
                } else {
                    result[i] = py::none();
                }
            }
        } catch (...) {
            restore();
            throw;
        }
        restore();
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
            // S.21: clone preserves the input's requires_grad flag instead
            // of unconditionally overriding it to true. Differentiation
            // through f only works when the input was created with
            // requires_grad=True; silently flipping it for the user
            // masked the common mistake of feeding a detached/leaf-only
            // tensor and getting back a graph that nominally tracked
            // grads but referenced no upstream variables. Callers who
            // want a non-grad probe should pass a non-grad input — the
            // resulting "no gradient computed" diagnostic below makes the
            // mismatch obvious.
            tenzor::Variable x_copy(x.tensor().clone(), x.requires_grad());
            py::object result = f(x_copy);
            tenzor::Variable output = result.cast<tenzor::Variable>();
            // R.22: release the GIL across the C++ backward traversal.
            {
                py::gil_scoped_release release;
                output.backward();
            }
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

    func_mod.def("vmap", [](py::function f, int64_t in_dim, int64_t out_dim) {
        return py::cpp_function([f, in_dim, out_dim](const tenzor::Variable& batched_input) -> tenzor::Variable {
            auto cpp_fn = [&f](const tenzor::Variable& x) -> tenzor::Variable {
                py::gil_scoped_acquire inner_gil;
                py::object result = f(x);
                return result.cast<tenzor::Variable>();
            };
            // R.22: outer GIL release; inner cpp_fn reacquires for the Python callback.
            py::gil_scoped_release release;
            // tenzor::vmap stacks the per-slice results back along `in_dim`, so
            // the batch axis of the output sits at position in_dim. Honour
            // out_dim by moving that axis from in_dim to out_dim — previously
            // out_dim was accepted but ignored ([[maybe_unused]]), silently
            // placing the batch axis on the wrong dimension for out_dim !=
            // in_dim vs. the documented torch.func.vmap semantics. The move is
            // done at the Variable level (autograd::permute) so the grad_fn
            // chain is preserved.
            tenzor::Variable result = tenzor::vmap(cpp_fn, batched_input, in_dim);
            const int64_t ndim =
                static_cast<int64_t>(result.tensor().shape().size());
            int64_t src = in_dim < 0 ? in_dim + ndim : in_dim;
            int64_t dst = out_dim < 0 ? out_dim + ndim : out_dim;
            if (src == dst || ndim == 0) {
                return result;
            }
            if (src < 0 || src >= ndim || dst < 0 || dst >= ndim) {
                throw std::runtime_error(
                    "func.vmap: out_dim is out of range for the vmapped output");
            }
            std::vector<int64_t> order;
            order.reserve(static_cast<size_t>(ndim));
            for (int64_t d = 0; d < ndim; ++d) {
                if (d != src) {
                    order.push_back(d);
                }
            }
            order.insert(order.begin() + dst, src);
            return tenzor::permute(result, order);
        });
    }, py::arg("f"), py::arg("in_dim") = 0, py::arg("out_dim") = 0,
    "Return a vectorized version of f that maps over a batch dimension.");

    func_mod.def("jacrev", [](py::function f) {
        return py::cpp_function([f](const tenzor::Variable& x) -> tenzor::Variable {
            auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
                py::gil_scoped_acquire inner_gil;
                py::object result = f(input);
                return result.cast<tenzor::Variable>();
            };
            // R.22: outer GIL release across the C++ jacobian probe loop.
            tenzor::Tensor J;
            {
                py::gil_scoped_release release;
                J = tenzor::jacobian(cpp_fn, x);
            }
            return tenzor::Variable(J, false);
        });
    }, py::arg("f"),
    "Return a function that computes the Jacobian of f. NOTE: mode is selected "
    "automatically by tenzor::jacobian (forward-mode when outputs <= inputs, "
    "reverse otherwise); this does not force reverse-mode.");

    func_mod.def("jacfwd", [](py::function f) {
        return py::cpp_function([f](const tenzor::Variable& x) -> tenzor::Variable {
            auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
                py::gil_scoped_acquire inner_gil;
                py::object result = f(input);
                return result.cast<tenzor::Variable>();
            };
            // R.22: outer GIL release.
            tenzor::Tensor J;
            {
                py::gil_scoped_release release;
                J = tenzor::jacobian(cpp_fn, x);
            }
            return tenzor::Variable(J, false);
        });
    }, py::arg("f"),
    "Return a function that computes the Jacobian of f. NOTE: mode is selected "
    "automatically by tenzor::jacobian (forward-mode when outputs <= inputs, "
    "reverse otherwise); this does not force forward-mode.");

    func_mod.def("hessian", [](py::function f) {
        return py::cpp_function([f](const tenzor::Variable& x) -> tenzor::Variable {
            auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
                py::gil_scoped_acquire inner_gil;
                py::object result = f(input);
                return result.cast<tenzor::Variable>();
            };
            // R.22: outer GIL release.
            tenzor::Tensor H;
            {
                py::gil_scoped_release release;
                H = tenzor::hessian(cpp_fn, x);
            }
            return tenzor::Variable(H, false);
        });
    }, py::arg("f"),
    "Return a function that computes the Hessian of a scalar-valued f.");

    func_mod.def("jvp", [](py::function f, const tenzor::Variable& x,
                           const tenzor::Tensor& tangent,
                           const std::string& mode) {
        auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
            py::gil_scoped_acquire inner_gil;
            py::object result = f(input);
            return result.cast<tenzor::Variable>();
        };
        tenzor::JvpMode jvp_mode;
        if (mode == "walker") {
            jvp_mode = tenzor::JvpMode::Walker;
        } else if (mode == "dual") {
            jvp_mode = tenzor::JvpMode::Dual;
        } else {
            throw std::invalid_argument(
                "jvp: mode must be 'walker' or 'dual', got '" + mode + "'");
        }
        // R.22: outer GIL release; inner cpp_fn reacquires for the Python callback.
        tenzor::Variable out;
        tenzor::Tensor tangent_out;
        {
            py::gil_scoped_release release;
            auto pair = tenzor::jvp(cpp_fn, x, tangent, jvp_mode);
            out = std::move(pair.first);
            tangent_out = std::move(pair.second);
        }
        return py::make_tuple(out, tangent_out);
    }, py::arg("f"), py::arg("x"), py::arg("tangent"),
    py::arg("mode") = std::string("walker"),
    "Forward-mode Jacobian-vector product. Returns (output, J_f(x) @ tangent).\n"
    "mode='walker' (default): build the autograd graph then walk it.\n"
    "mode='dual': raise the is_dual_mode() TLS flag while invoking f, enabling\n"
    "  per-op dual_apply<> interceptors when present.");

    func_mod.def("hvp", [](py::function f, const tenzor::Variable& x,
                           const tenzor::Tensor& v) {
        auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
            py::gil_scoped_acquire inner_gil;
            py::object result = f(input);
            return result.cast<tenzor::Variable>();
        };
        // R.22: outer GIL release.
        tenzor::Variable out;
        tenzor::Tensor hvp_result;
        {
            py::gil_scoped_release release;
            auto pair = tenzor::hvp(cpp_fn, x, v);
            out = std::move(pair.first);
            hvp_result = std::move(pair.second);
        }
        return py::make_tuple(out, hvp_result);
    }, py::arg("f"), py::arg("x"), py::arg("v"),
    "Hessian-vector product. Returns (output, H_f(x) @ v).");

    func_mod.def("vhp", [](py::function f, const tenzor::Variable& x,
                           const tenzor::Tensor& v) {
        auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
            py::gil_scoped_acquire inner_gil;
            py::object result = f(input);
            return result.cast<tenzor::Variable>();
        };
        // R.22: outer GIL release.
        tenzor::Variable out;
        tenzor::Tensor vhp_result;
        {
            py::gil_scoped_release release;
            auto pair = tenzor::vhp(cpp_fn, x, v);
            out = std::move(pair.first);
            vhp_result = std::move(pair.second);
        }
        return py::make_tuple(out, vhp_result);
    }, py::arg("f"), py::arg("x"), py::arg("v"),
    "Vector-Hessian product. Returns (output, v^T @ H_f(x)).");

    func_mod.def("vjp", [](py::function f, const tenzor::Variable& x,
                           const tenzor::Tensor& cotangent) {
        auto cpp_fn = [&f](const tenzor::Variable& input) -> tenzor::Variable {
            py::gil_scoped_acquire inner_gil;
            py::object result = f(input);
            return result.cast<tenzor::Variable>();
        };
        // R.22: outer GIL release.
        tenzor::Variable out;
        tenzor::Tensor vjp_result;
        {
            py::gil_scoped_release release;
            auto pair = tenzor::vjp(cpp_fn, x, cotangent);
            out = std::move(pair.first);
            vjp_result = std::move(pair.second);
        }
        return py::make_tuple(out, vjp_result);
    }, py::arg("f"), py::arg("x"), py::arg("cotangent"),
    "Vector-Jacobian product (reverse-mode). Returns (output, v^T @ J_f(x)).");

    // Audit E.11: expose gradcheck / gradgradcheck. The C++ helpers exist
    // and are heavily used internally, but had no Python bindings — the
    // autograd.pyi declarations were a documentation lie.
    autograd_mod.def("gradcheck",
        [](py::function f, const tenzor::Variable& input,
           double eps, double atol, double rtol,
           bool raise_exception) -> bool {
            auto cpp_fn = [&f](const tenzor::Variable& x) -> tenzor::Variable {
                py::gil_scoped_acquire inner_gil;
                py::object result = f(x);
                return result.cast<tenzor::Variable>();
            };
            // R.22: outer GIL release across the FD probe loop.
            py::gil_scoped_release release;
            return tenzor::gradcheck(cpp_fn, input, eps, atol, rtol, raise_exception);
        },
        py::arg("func"),
        py::arg("input"),
        py::arg("eps") = 1e-6,
        py::arg("atol") = 1e-5,
        py::arg("rtol") = 1e-3,
        py::arg("raise_exception") = false,
        "First-order gradient check via finite differences. Returns true if "
        "analytical gradients match the numerical estimate within `atol+rtol`.");

    autograd_mod.def("gradgradcheck",
        [](py::function f, const tenzor::Variable& input,
           double eps, double atol, double rtol,
           bool raise_exception) -> bool {
            auto cpp_fn = [&f](const tenzor::Variable& x) -> tenzor::Variable {
                py::gil_scoped_acquire inner_gil;
                py::object result = f(x);
                return result.cast<tenzor::Variable>();
            };
            // R.22: outer GIL release across the FD probe loop.
            py::gil_scoped_release release;
            return tenzor::gradgradcheck(cpp_fn, input, eps, atol, rtol, raise_exception);
        },
        py::arg("func"),
        py::arg("input"),
        py::arg("eps") = 1e-6,
        py::arg("atol") = 1e-5,
        py::arg("rtol") = 1e-3,
        py::arg("raise_exception") = false,
        "Second-order gradient check: gradcheck applied to the analytical "
        "gradient function. Returns true if Hessian-vector products match the "
        "numerical estimate.");
}

} // namespace tenzor::python
