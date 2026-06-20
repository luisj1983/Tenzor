// tenzor.jit Python bindings. Extracted from python/bindings.cpp as
// part of P3.4 (incremental split of the ~10k-line monolith).
//
// Covers: IR types (OpType/Value/Node/Graph), Tracer/TracingGuard,
// Compiler, free-function trace/optimize/save/load helpers, CompiledModule,
// and the tenzor.jit.compile + tenzor.compile torch.compile-equivalent.

#include "register.hpp"

#include <algorithm>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <tenzor/autograd/variable.hpp>
#include <tenzor/jit/compile.hpp>
#include <tenzor/jit/compiler.hpp>
#include <tenzor/jit/graph.hpp>
#include <tenzor/jit/script.hpp>
#include <tenzor/jit/serialization.hpp>
#include <tenzor/jit/tracer.hpp>
#include <tenzor/jit/tracing_interceptor.hpp>
#include <tenzor/backend/dispatch_interceptor.hpp>
#include <tenzor/lazy/lazy_tensor.hpp>
#include <tenzor/nn/module.hpp>

#ifdef TENZOR_HAS_MLIR_JIT
#include <tenzor/jit/mlir/iree_compile.hpp>
#endif

namespace py = pybind11;

namespace tenzor::python {

// Per-thread, *per-Tracer-instance* bookkeeping for the tracing interceptor
// that the Python Tracer API pushes onto the (thread-local)
// DispatchInterceptorStack. A single thread-wide flag was wrong: the Python
// `Tracer` class exposes `py::init<>()`, so a user can create several distinct
// Tracer objects on one thread and interleave their start/end calls. A shared
// flag let a second `start_trace()` skip installing its own interceptor (it
// would then record an empty graph) and let one tracer's `end_trace()` pop an
// interceptor owned by another tracer, corrupting the latter's recording.
//
// We therefore track installation state keyed by `Tracer*`. The underlying
// DispatchInterceptorStack is strictly LIFO (only the top can be popped), so we
// additionally keep a per-thread stack of the Tracer instances that currently
// own an installed interceptor. This lets us:
//   * install at most one interceptor per Tracer instance (idempotent
//     start_trace per instance), while still allowing distinct instances to
//     each install their own;
//   * pop the correct interceptor in `end_trace()` / `clear()` even when calls
//     are properly nested, and detect+repair improper (non-LIFO) teardown by
//     unwinding everything above the target so the stack is never left holding
//     a stale interceptor for a finished tracer.
//
// Tracing is inherently thread-local (the dispatch stack is thread-local), so
// thread_local storage is the right scope.
namespace {

// Owner stack: front == bottom, back == top of the matching slice of the
// DispatchInterceptorStack that the Python Tracer API installed on this thread.
std::vector<tenzor::jit::Tracer*>& py_tracer_owner_stack() {
    static thread_local std::vector<tenzor::jit::Tracer*> owners;
    return owners;
}

bool py_tracer_is_installed(tenzor::jit::Tracer* self) {
    auto& owners = py_tracer_owner_stack();
    return std::find(owners.begin(), owners.end(), self) != owners.end();
}

// Push a tracing interceptor for `self` if it does not already own one.
void py_tracer_install(tenzor::jit::Tracer& self) {
    if (py_tracer_is_installed(&self)) return;
    DispatchInterceptorStack::push(
        tenzor::jit::make_tracing_interceptor(self, nullptr));
    py_tracer_owner_stack().push_back(&self);
}

// Remove the tracing interceptor owned by `self`, if any. To keep the LIFO
// DispatchInterceptorStack consistent, pop every interceptor sitting above
// `self`'s entry as well (those belong to tracers that were never properly
// ended). Returns true if `self` owned an interceptor that was removed.
bool py_tracer_uninstall(tenzor::jit::Tracer* self) {
    auto& owners = py_tracer_owner_stack();
    auto it = std::find(owners.rbegin(), owners.rend(), self);
    if (it == owners.rend()) return false;
    // Number of owners from the top down to and including `self`.
    const std::size_t pops =
        static_cast<std::size_t>(std::distance(owners.rbegin(), it)) + 1;
    for (std::size_t i = 0; i < pops; ++i) {
        DispatchInterceptorStack::pop();
        owners.pop_back();
    }
    return true;
}

} // namespace

void register_jit(py::module_& m) {
    auto jit = m.def_submodule("jit", "JIT compilation and tracing");

    py::enum_<tenzor::jit::OpType>(jit, "OpType")
        .value("Add", tenzor::jit::OpType::Add)
        .value("Sub", tenzor::jit::OpType::Sub)
        .value("Mul", tenzor::jit::OpType::Mul)
        .value("Div", tenzor::jit::OpType::Div)
        .value("MatMul", tenzor::jit::OpType::MatMul)
        .value("ReLU", tenzor::jit::OpType::ReLU)
        .value("Sigmoid", tenzor::jit::OpType::Sigmoid)
        .value("Tanh", tenzor::jit::OpType::Tanh)
        // W.31: surface GELU on the Python enum so trace regression tests can
        // assert ``OpType.GELU`` directly instead of scanning string dumps.
        // The C++ enum value has existed since the audit-3 trace expansion.
        .value("GELU", tenzor::jit::OpType::GELU)
        .value("Softmax", tenzor::jit::OpType::Softmax)
        .value("Conv2d", tenzor::jit::OpType::Conv2d)
        .value("BatchNorm2d", tenzor::jit::OpType::BatchNorm2d)
        .value("LayerNorm", tenzor::jit::OpType::LayerNorm)
        .value("MaxPool2d", tenzor::jit::OpType::MaxPool2d)
        .value("AvgPool2d", tenzor::jit::OpType::AvgPool2d)
        .value("Reshape", tenzor::jit::OpType::Reshape)
        .value("Transpose", tenzor::jit::OpType::Transpose)
        .value("Flatten", tenzor::jit::OpType::Flatten)
        .value("Linear", tenzor::jit::OpType::Linear)
        .value("Constant", tenzor::jit::OpType::Constant)
        .value("Input", tenzor::jit::OpType::Input)
        .value("Output", tenzor::jit::OpType::Output)
        // Remaining OpType members — the enum is the public IR vocabulary, so
        // expose every value (previously only a subset was bound, which made
        // e.g. ``OpType.Pow`` raise AttributeError in trace-inspection tests
        // even though the C++ tracer emits it).
        .value("Bmm", tenzor::jit::OpType::Bmm)
        .value("LogSoftmax", tenzor::jit::OpType::LogSoftmax)
        .value("AdaptiveAvgPool2d", tenzor::jit::OpType::AdaptiveAvgPool2d)
        .value("ConvTranspose", tenzor::jit::OpType::ConvTranspose)
        .value("Permute", tenzor::jit::OpType::Permute)
        .value("Squeeze", tenzor::jit::OpType::Squeeze)
        .value("Unsqueeze", tenzor::jit::OpType::Unsqueeze)
        .value("Sum", tenzor::jit::OpType::Sum)
        .value("Mean", tenzor::jit::OpType::Mean)
        .value("Max", tenzor::jit::OpType::Max)
        .value("Min", tenzor::jit::OpType::Min)
        .value("Exp", tenzor::jit::OpType::Exp)
        .value("Log", tenzor::jit::OpType::Log)
        .value("Sqrt", tenzor::jit::OpType::Sqrt)
        .value("Pow", tenzor::jit::OpType::Pow)
        .value("Abs", tenzor::jit::OpType::Abs)
        .value("Neg", tenzor::jit::OpType::Neg)
        .value("Clamp", tenzor::jit::OpType::Clamp)
        .value("Slice", tenzor::jit::OpType::Slice)
        .value("Cat", tenzor::jit::OpType::Cat)
        .value("Dropout", tenzor::jit::OpType::Dropout)
        .value("Embedding", tenzor::jit::OpType::Embedding)
        .value("Det", tenzor::jit::OpType::Det)
        .value("Inv", tenzor::jit::OpType::Inv)
        .value("Solve", tenzor::jit::OpType::Solve)
        .value("Cholesky", tenzor::jit::OpType::Cholesky)
        .value("Svd", tenzor::jit::OpType::Svd)
        .value("Qr", tenzor::jit::OpType::Qr)
        .value("Eigh", tenzor::jit::OpType::Eigh)
        .value("Eigvalsh", tenzor::jit::OpType::Eigvalsh)
        .value("Norm", tenzor::jit::OpType::Norm)
        .value("Slogdet", tenzor::jit::OpType::Slogdet)
        .value("FlashAttention", tenzor::jit::OpType::FlashAttention)
        .value("FusedFFN", tenzor::jit::OpType::FusedFFN)
        .value("ResidualAdd", tenzor::jit::OpType::ResidualAdd)
        .value("ShapeGuard", tenzor::jit::OpType::ShapeGuard)
        .value("GuardNode", tenzor::jit::OpType::GuardNode)
        .value("SwapOut", tenzor::jit::OpType::SwapOut)
        .value("SwapIn", tenzor::jit::OpType::SwapIn)
        .value("QuantizedLinear", tenzor::jit::OpType::QuantizedLinear)
        .value("QuantizedConv2d", tenzor::jit::OpType::QuantizedConv2d)
        .value("Dequantize", tenzor::jit::OpType::Dequantize)
        .value("Quantize", tenzor::jit::OpType::Quantize)
        .value("SparseMatMul", tenzor::jit::OpType::SparseMatMul)
        .value("DenseToSparse", tenzor::jit::OpType::DenseToSparse)
        .value("If", tenzor::jit::OpType::If)
        .value("Loop", tenzor::jit::OpType::Loop)
        .value("LayoutConvert", tenzor::jit::OpType::LayoutConvert)
        .value("Cast", tenzor::jit::OpType::Cast)
        .value("SiLU", tenzor::jit::OpType::SiLU)
        .value("Where", tenzor::jit::OpType::Where)
        .value("Stack", tenzor::jit::OpType::Stack)
        .value("Broadcast", tenzor::jit::OpType::Broadcast)
        .value("IndexSelect", tenzor::jit::OpType::IndexSelect)
        .value("RMSNorm", tenzor::jit::OpType::RMSNorm)
        .value("GQA", tenzor::jit::OpType::GQA)
        .value("RoPE", tenzor::jit::OpType::RoPE)
        .value("Padding", tenzor::jit::OpType::Padding)
        .value("Interpolate", tenzor::jit::OpType::Interpolate);

    py::class_<tenzor::jit::Value, std::shared_ptr<tenzor::jit::Value>>(jit, "Value",
        "Represents a tensor value in the IR graph")
        .def_property_readonly("id", &tenzor::jit::Value::id)
        .def_property_readonly("shape", &tenzor::jit::Value::shape)
        .def_property_readonly("dtype", &tenzor::jit::Value::dtype)
        .def_property_readonly("device", &tenzor::jit::Value::device);

    py::class_<tenzor::jit::Node, std::shared_ptr<tenzor::jit::Node>>(jit, "Node",
        "Represents an operation node in the IR graph")
        .def_property_readonly("op_type", &tenzor::jit::Node::op_type)
        .def_property_readonly("name", &tenzor::jit::Node::name)
        .def("set_name", &tenzor::jit::Node::set_name)
        .def("get_attr", &tenzor::jit::Node::get_attr)
        .def("get_int_attr", &tenzor::jit::Node::get_int_attr)
        .def("get_vec_attr", &tenzor::jit::Node::get_vec_attr)
        .def("get_bool_attr", &tenzor::jit::Node::get_bool_attr)
        .def("has_attr", &tenzor::jit::Node::has_attr);

    py::class_<tenzor::jit::Graph, std::shared_ptr<tenzor::jit::Graph>>(jit, "Graph",
        "IR graph representing a complete computation")
        .def(py::init<>())
        .def("num_nodes", &tenzor::jit::Graph::num_nodes)
        .def("num_values", &tenzor::jit::Graph::num_values)
        // W.31: expose the node list so trace regression tests can scan for
        // particular OpTypes (e.g. confirming gelu traces as a single GELU
        // node, not its underlying decomposition into Mul/Add/Tanh).
        .def("nodes", &tenzor::jit::Graph::nodes,
             py::return_value_policy::reference_internal,
             "Topologically sorted list of all nodes in the graph")
        .def("forward", &tenzor::jit::Graph::forward,
             py::arg("inputs"),
             "Execute graph with runtime inputs",
             py::call_guard<py::gil_scoped_release>())
        // Audit-8 II.11: Graph::save / Graph::load are pure disk I/O on the
        // already-built C++ IR — no Python objects touched. Drop the GIL so
        // other Python threads (DataLoader workers, DDP comm) keep running
        // while a fused graph is written or read from disk.
        .def("save", &tenzor::jit::Graph::save,
             py::arg("path"),
             "Save graph to file",
             py::call_guard<py::gil_scoped_release>())
        .def_static("load", &tenzor::jit::Graph::load,
             py::arg("path"),
             "Load graph from file",
             py::call_guard<py::gil_scoped_release>())
        .def("to_string", &tenzor::jit::Graph::to_string,
             "Get string representation of graph")
        .def("topological_sort", &tenzor::jit::Graph::topological_sort)
        .def("infer_types", &tenzor::jit::Graph::infer_types,
             py::call_guard<py::gil_scoped_release>())
        .def("__repr__", &tenzor::jit::Graph::to_string);

    py::class_<tenzor::jit::Tracer>(jit, "Tracer",
        "Tracing context for recording operations")
        .def(py::init<>())
        .def("start_trace", [](tenzor::jit::Tracer& self) {
                // Recording requires the dispatch interceptor that turns each
                // dispatched op into a Tracer::record_op call — the raw
                // Tracer::start_trace only flips the tracing flag (the C++
                // TracingGuard installs the interceptor separately). Without
                // this, a Python `Tracer().start_trace()` produced an empty
                // graph (op_types == []). Install it here so the direct Tracer
                // API records ops the same way TracingGuard does.
                //
                // Installation is keyed per Tracer instance, so two distinct
                // Tracer objects on the same thread each get their own
                // interceptor instead of the second silently reusing (or
                // skipping) the first's.
                self.start_trace();
                py_tracer_install(self);
             },
             "Start recording operations")
        .def("end_trace", [](tenzor::jit::Tracer& self,
                             const std::vector<tenzor::Variable>& inputs,
                             const std::vector<tenzor::Variable>& outputs) {
                py_tracer_uninstall(&self);
                return self.end_trace(inputs, outputs);
             },
             py::arg("inputs"), py::arg("outputs"),
             "Stop recording and build IR graph")
        .def("is_tracing", &tenzor::jit::Tracer::is_tracing,
             "Check if tracing is active")
        .def("clear", [](tenzor::jit::Tracer& self) {
                // Recovery path: if a trace was started but never properly
                // ended (e.g. an exception fired in the traced region before
                // end_trace ran), the interceptor would otherwise stay
                // installed forever and keep recording every dispatched op.
                // Remove this tracer's interceptor before clearing its state so
                // clear() is a complete reset reachable from Python.
                py_tracer_uninstall(&self);
                self.clear();
             },
             "Clear all recorded operations")
        .def("graph_break_count", &tenzor::jit::Tracer::graph_break_count,
             "Number of graph breaks recorded during the current (or last)\n"
             "trace session. Non-zero indicates operations that couldn't be\n"
             "captured in the traced graph — use to diagnose why a model\n"
             "falls back to eager execution.")
        .def("record_graph_break", &tenzor::jit::Tracer::record_graph_break,
             py::arg("reason"),
             "Record a graph break with a human-readable reason. Used by\n"
             "the tracer itself; exposed for tests and advanced debugging.")
        .def_static("get_instance", &tenzor::jit::Tracer::get_instance,
             py::return_value_policy::reference,
             "Get thread-local tracer instance");

    py::class_<tenzor::jit::TracingGuard>(jit, "TracingGuard",
        "RAII guard for tracing scope")
        .def(py::init<>())
        .def("get_graph", &tenzor::jit::TracingGuard::get_graph,
             py::arg("inputs"), py::arg("outputs"),
             "Get traced graph");

    py::class_<tenzor::jit::Compiler>(jit, "Compiler",
        "Graph optimization compiler")
        .def(py::init<bool>(),
             py::arg("enable_default_passes") = true,
             "Create compiler with optional default passes")
        .def("optimize", &tenzor::jit::Compiler::optimize,
             py::arg("graph"), py::arg("max_iterations") = 10,
             "Optimize graph with all passes",
             py::call_guard<py::gil_scoped_release>())
        .def("set_verbose", &tenzor::jit::Compiler::set_verbose,
             py::arg("enable"),
             "Enable verbose logging")
        .def("clear_stats", &tenzor::jit::Compiler::clear_stats);

    jit.def("trace", py::overload_cast<std::shared_ptr<tenzor::nn::Module>,
            const tenzor::Variable&>(&tenzor::jit::trace),
            py::arg("module"), py::arg("dummy_input"),
            "Trace a module's forward pass",
            // Y.26: release the GIL during trace — runs the module's
            // forward pass through native ops, which can be very long for
            // realistic models; holding the GIL would block DataLoader /
            // DDP comm threads behind it.
            py::call_guard<py::gil_scoped_release>());

    jit.def("optimize_graph", &tenzor::jit::optimize_graph,
            py::arg("graph"),
            "Apply standard optimizations to graph",
            py::call_guard<py::gil_scoped_release>());

    // Audit-8 II.11: serialise/deserialise are pure C++ disk I/O — no Python
    // objects touched. Drop the GIL across the call so other threads run.
    jit.def("save_graph", &tenzor::jit::save_graph,
            py::arg("graph"), py::arg("path"),
            "Save graph to file",
            py::call_guard<py::gil_scoped_release>());

    jit.def("load_graph", &tenzor::jit::load_graph,
            py::arg("path"),
            "Load graph from file",
            py::call_guard<py::gil_scoped_release>());

    jit.def("export_graph_text", &tenzor::jit::export_graph_text,
            py::arg("graph"), py::arg("path"),
            "Export graph as text for debugging");

    jit.def("export_graph_dot", &tenzor::jit::export_graph_dot,
            py::arg("graph"), py::arg("path"),
            "Export graph as DOT file for visualization");

    jit.def("get_graph_stats", &tenzor::jit::get_graph_stats,
            py::arg("graph"),
            "Get graph statistics");

    jit.def("verify_graph", &tenzor::jit::verify_graph,
            py::arg("graph"),
            "Verify graph integrity, returns list of errors",
            py::call_guard<py::gil_scoped_release>());

    py::class_<tenzor::jit::CompiledModule,
               std::shared_ptr<tenzor::jit::CompiledModule>>(jit, "CompiledModule",
        "Callable wrapper around a traced+compiled graph (torch.jit.ScriptModule analog)")
        // Y.26: ``Graph.forward`` (above) already releases the GIL; mirror
        // that contract on the user-facing CompiledModule entry points
        // (forward / __call__ / trace_module) so a Python inference loop
        // doesn't block DataLoader workers or DDP comm threads on the
        // duration of a fused-graph execution.
        .def("forward",
             py::overload_cast<const tenzor::Variable&>(&tenzor::jit::CompiledModule::forward),
             py::arg("input"),
             "Execute the compiled graph on a Variable input",
             py::call_guard<py::gil_scoped_release>())
        .def("__call__",
             py::overload_cast<const tenzor::Variable&>(&tenzor::jit::CompiledModule::forward),
             py::arg("input"),
             "Callable alias for forward()",
             py::call_guard<py::gil_scoped_release>())
        .def("optimize_for_inference",
             &tenzor::jit::CompiledModule::optimize_for_inference,
             "Apply inference-only optimization passes",
             py::call_guard<py::gil_scoped_release>());

    jit.def("trace_module",
            py::overload_cast<std::shared_ptr<tenzor::nn::Module>,
                              const tenzor::Variable&>(&tenzor::jit::CompiledModule::trace),
            py::arg("module"), py::arg("example_input"),
            "Trace a module's forward pass, returning a callable CompiledModule.",
            py::call_guard<py::gil_scoped_release>());

    // Compile API (torch.compile equivalent)
    jit.def("compile", [](py::function fn, bool fullgraph, std::string mode) {
        // The C++ side stores the callable in N-input form; the Python
        // wrapper unpacks the span as positional arguments so user code
        // can keep writing `def f(x):` or `def f(x, y, z):` naturally.
        auto cpp_fn =
            [fn](std::span<const tenzor::Variable> inputs) -> tenzor::Variable {
            py::gil_scoped_acquire acquire;
            if (inputs.size() == 1) {
                auto result = fn(inputs[0]);
                return result.cast<tenzor::Variable>();
            }
            py::tuple args(inputs.size());
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                args[i] = py::cast(inputs[i]);
            }
            auto result = fn(*args);
            return result.cast<tenzor::Variable>();
        };

        tenzor::jit::CompileConfig config;
        config.fullgraph = fullgraph;
        config.mode = std::move(mode);

        auto compiled = std::make_shared<tenzor::jit::CompiledFunction>(
            tenzor::jit::CompiledFunction::FnTypeN(std::move(cpp_fn)),
            std::move(config));

        // Returned py::cpp_function accepts any positional Variables. The
        // pybind11 *args dispatch happens via py::args; we then forward to
        // the span-based operator().
        return py::cpp_function([compiled](py::args args) {
            std::vector<tenzor::Variable> inputs;
            inputs.reserve(args.size());
            for (auto& a : args) {
                inputs.push_back(a.cast<tenzor::Variable>());
            }
            py::gil_scoped_release release;
            return (*compiled)(
                std::span<const tenzor::Variable>(inputs.data(), inputs.size()));
        });
    },
    py::arg("fn"),
    py::arg("fullgraph") = false,
    py::arg("mode") = "default",
    "Compile a function for automatic graph capture and optimization.\n"
    "First call traces and compiles; subsequent calls use cached compiled graph.\n"
    "Shape mismatches trigger recompilation (up to 8 shapes cached).\n"
    "Accepts any number of positional Variable arguments.");

    // tenzor.compile alias
    m.def("compile", [jit](py::function fn, bool fullgraph, std::string mode) {
        return jit.attr("compile")(fn, fullgraph, mode);
    },
    py::arg("fn"),
    py::arg("fullgraph") = false,
    py::arg("mode") = "default",
    "Compile a function for automatic graph capture (alias for jit.compile).");

    // compile_script — Python-subset scripting frontend
    jit.def("compile_script", [](const std::string& source) {
        return tenzor::jit::compile_script(source.c_str());
    },
    py::arg("source"),
    "Compile a Python-subset script into a CompiledModule.\n"
    "MVP grammar: single `def forward(x): return EXPR` with arithmetic on x,\n"
    "float/int literals, +, -, *, /, and parentheses. Throws on parse error.\n"
    "Uses a default CPU+Float32 {1}-element dummy for tracing.");

    jit.def("compile_script", [](const std::string& source, const tenzor::Tensor& dummy) {
        return tenzor::jit::compile_script(source.c_str(), dummy);
    },
    py::arg("source"), py::arg("dummy"),
    "Overload that specialises the compiled module for the supplied dummy\n"
    "input's dtype, device, and shape. Use this when compiling for non-CPU\n"
    "or non-Float32 inputs.");

    // =========================================================================
    // Group F — Debug UX: show_graph / show_mlir / show_stablehlo / show_iree
    // and cache_stats. F.1 lands show_graph plus the shared
    // CompiledFunctionHandle binding that the python/tenzor/jit.py decorator
    // routes through (`_core.jit.compile_function`).
    // =========================================================================

    py::class_<tenzor::jit::CompiledFunction,
               std::shared_ptr<tenzor::jit::CompiledFunction>>(
        jit, "CompiledFunctionHandle",
        "Underlying object held by the @tz.jit decorator. Exposes __call__\n"
        "for invocation and is the argument accepted by the show_* free\n"
        "functions on tenzor_core.jit.")
        .def("__call__",
             [](tenzor::jit::CompiledFunction& self, py::args args) {
                 std::vector<tenzor::Variable> inputs;
                 inputs.reserve(args.size());
                 for (auto& a : args) {
                     inputs.push_back(a.cast<tenzor::Variable>());
                 }
                 if (inputs.empty()) {
                     throw std::invalid_argument(
                         "CompiledFunctionHandle.__call__: expected at "
                         "least one positional Variable input");
                 }
                 py::gil_scoped_release release;
                 return self(std::span<const tenzor::Variable>(
                     inputs.data(), inputs.size()));
             });

    jit.def("compile_function",
        [](py::function fn, std::string backend, std::string target,
           bool fallback_to_eager) {
            auto cpp_fn = [fn](std::span<const tenzor::Variable> inputs)
                -> tenzor::Variable {
                py::gil_scoped_acquire acquire;
                // Forward inputs as positional arguments. Use pybind11's
                // automatic conversion (same path the previous single-input
                // form took when calling fn(input)) by building the args
                // tuple via py::make_tuple's variadic spread — but since
                // arity is dynamic, fall back to explicit cast which uses
                // the same default return-value policy as the implicit
                // single-arg path.
                if (inputs.size() == 1) {
                    auto result = fn(inputs[0]);
                    return result.cast<tenzor::Variable>();
                }
                py::tuple args(inputs.size());
                for (std::size_t i = 0; i < inputs.size(); ++i) {
                    args[i] = py::cast(inputs[i]);
                }
                auto result = fn(*args);
                return result.cast<tenzor::Variable>();
            };

            tenzor::jit::CompileConfig config;
            config.backend = std::move(backend);
            config.target  = std::move(target);
            // `fallback_to_eager` is the inverse of CompileConfig::strict.
            // - fallback_to_eager=false  → strict=true  → throw on coverage gap
            // - fallback_to_eager=true   → strict=false → degrade to eager
            // This matches the @tz.jit Python decorator docstring.
            config.strict = !fallback_to_eager;

            // Audit-4 W.15: under config.strict the CompiledFunction ctor
            // may eagerly run IREE compilation (long-running C++/MLIR work
            // that never touches Python objects after the trace lambda has
            // released the GIL itself). Drop the GIL here so other Python
            // threads can make progress during compilation. The trace
            // lambda above re-acquires the GIL via py::gil_scoped_acquire
            // before calling the user-supplied Python @fn.
            py::gil_scoped_release release;
            return std::make_shared<tenzor::jit::CompiledFunction>(
                tenzor::jit::CompiledFunction::FnTypeN(std::move(cpp_fn)),
                std::move(config));
        },
        py::arg("fn"),
        py::arg("backend") = "mlir",
        py::arg("target")  = "auto",
        py::arg("fallback_to_eager") = false,
        "Build a CompiledFunctionHandle for @tz.jit. Use the returned\n"
        "object's __call__ to invoke and the show_* free functions on\n"
        "this module to introspect the pipeline.\n"
        "Accepts any number of positional Variable arguments.");

    // Audit-4 W.15: every show_* dump runs tracing + MLIR lowering +
    // (for show_iree) the iree-compile pipeline. None of those touch
    // Python objects after the trace lambda's inner py::gil_scoped_acquire
    // re-enters for user callbacks, so we drop the GIL across the whole
    // call to let other Python threads make progress.
    jit.def("show_graph",
        [](std::shared_ptr<tenzor::jit::CompiledFunction> cf,
           const tenzor::Variable& example) {
            if (!cf) {
                throw std::runtime_error(
                    "show_graph: passed object is not a tz.jit-compiled "
                    "function (no _tz_compiled attribute)");
            }
            py::gil_scoped_release release;
            return cf->dump_graph(example);
        },
        py::arg("compiled"), py::arg("example"),
        "Trace and dump the optimized tenzor::jit::Graph as text.");

    jit.def("show_mlir",
        [](std::shared_ptr<tenzor::jit::CompiledFunction> cf,
           const tenzor::Variable& example) {
            if (!cf) {
                throw std::runtime_error(
                    "show_mlir: passed object is not a tz.jit-compiled "
                    "function");
            }
            py::gil_scoped_release release;
            return cf->dump_mlir(example);
        },
        py::arg("compiled"), py::arg("example"),
        "Lower the traced graph and return the StableHLO text "
        "(plugin-enabled).");

    jit.def("show_stablehlo",
        [](std::shared_ptr<tenzor::jit::CompiledFunction> cf,
           const tenzor::Variable& example) {
            if (!cf) {
                throw std::runtime_error(
                    "show_stablehlo: passed object is not a tz.jit-compiled "
                    "function");
            }
            py::gil_scoped_release release;
            return cf->dump_stablehlo(example);
        },
        py::arg("compiled"), py::arg("example"),
        "Lower the traced graph with plugin_enabled=false (custom_call ops "
        "decomposed).");

    jit.def("show_iree",
        [](std::shared_ptr<tenzor::jit::CompiledFunction> cf,
           const tenzor::Variable& example) {
            if (!cf) {
                throw std::runtime_error(
                    "show_iree: passed object is not a tz.jit-compiled "
                    "function");
            }
            py::gil_scoped_release release;
            return cf->dump_iree(example);
        },
        py::arg("compiled"), py::arg("example"),
        "Run iree-compile --mlir-print-ir-after-all on the lowered MLIR "
        "and return the captured pipeline trace.");

#ifdef TENZOR_HAS_MLIR_JIT
    jit.def("cache_stats", []() {
        const auto s = tenzor::jit::mlir_jit::cache_stats();
        py::dict d;
        d["hits"]             = s.hits;
        d["misses"]           = s.misses;
        d["retraces"]         = s.retraces;
        d["evictions"]        = s.evictions;
        d["total_compile_ms"] = s.total_compile_ms;
        return d;
    },
    "Return a dict of JIT cache counters: hits, misses, retraces, "
    "evictions, total_compile_ms.");

    jit.def("reset_cache_stats", []() {
        tenzor::jit::mlir_jit::reset_cache_stats();
    },
    "Reset all JIT cache counters back to zero. Mainly for tests.");
#endif

    // =========================================================================
    // Lazy Tensor API
    // =========================================================================
    auto lazy = m.def_submodule("lazy", "Lazy/deferred tensor execution");

    py::class_<tenzor::lazy::LazyTensor>(lazy, "LazyTensor",
        "A deferred tensor that records operations as a graph.")
        .def_static("from_tensor", &tenzor::lazy::LazyTensor::from_tensor,
             "Create a LazyTensor from an existing Tensor", py::arg("tensor"))
        .def_static("placeholder", &tenzor::lazy::LazyTensor::placeholder,
             "Create a placeholder LazyTensor with given shape/dtype/device",
             py::arg("shape"), py::arg("dtype") = tenzor::DType::Float32,
             py::arg("device") = tenzor::Device::cpu(), py::arg("name") = "")
        .def("materialize", &tenzor::lazy::LazyTensor::materialize,
             "Execute the computation graph and return the concrete Tensor",
             py::call_guard<py::gil_scoped_release>())
        .def("is_materialized", &tenzor::lazy::LazyTensor::is_materialized)
        .def_property_readonly("shape", &tenzor::lazy::LazyTensor::shape)
        .def_property_readonly("ndim", &tenzor::lazy::LazyTensor::ndim)
        .def_property_readonly("dtype", &tenzor::lazy::LazyTensor::dtype)
        .def_property_readonly("device", &tenzor::lazy::LazyTensor::device);

    // Lazy operations
    lazy.def("add", &tenzor::lazy::add, py::arg("a"), py::arg("b"));
    lazy.def("sub", &tenzor::lazy::sub, py::arg("a"), py::arg("b"));
    lazy.def("mul", &tenzor::lazy::mul, py::arg("a"), py::arg("b"));
    lazy.def("div", &tenzor::lazy::div, py::arg("a"), py::arg("b"));
    lazy.def("matmul", &tenzor::lazy::matmul, py::arg("a"), py::arg("b"));
    lazy.def("neg", &tenzor::lazy::neg, py::arg("a"));
    lazy.def("relu", &tenzor::lazy::relu, py::arg("a"));
    lazy.def("sigmoid", &tenzor::lazy::sigmoid, py::arg("a"));
    lazy.def("tanh", &tenzor::lazy::tanh, py::arg("a"));
    lazy.def("exp", &tenzor::lazy::exp, py::arg("a"));
    lazy.def("log", &tenzor::lazy::log, py::arg("a"));
    lazy.def("sqrt", &tenzor::lazy::sqrt, py::arg("a"));
    lazy.def("transpose", &tenzor::lazy::transpose, py::arg("a"), py::arg("dim0"), py::arg("dim1"));
    lazy.def("reshape", &tenzor::lazy::reshape, py::arg("a"), py::arg("shape"));
    lazy.def("sum", &tenzor::lazy::sum, py::arg("a"), py::arg("dim") = py::none());
    lazy.def("mean", &tenzor::lazy::mean, py::arg("a"), py::arg("dim") = py::none());
}

} // namespace tenzor::python
