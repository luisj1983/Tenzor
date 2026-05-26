#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <optional>
#include "bindings/register.hpp"  // split-out submodule registrars
#include "bindings/future_binding.hpp"  // TensorFuture / TensorListFuture (audit C.6)
#include <cassert>                 // 5th-audit B6 PyGILState_Check assertions
#include <iostream>
#include <sstream>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/graph_optimizer.hpp>
#include <tenzor/onnx/graph_module.hpp>
#include <tenzor/core/device_guard.hpp>
#include <tenzor/core/dlpack.hpp>
#include <tenzor/ops/custom_op.hpp>
#include <tenzor/ops/indexing.hpp>
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/linalg.hpp>
#include <tenzor/ops/transform.hpp>
#include <tenzor/ops/fp8_scaling.hpp>
#include <tenzor/backend/loader.hpp>
#include <tenzor/backend/backend.hpp>
#include <tenzor/backend/cuda_config.hpp>
#include <tenzor/jit/compile.hpp>
#include <tenzor/distributed/rpc/rpc.hpp>
#include <tenzor/backend/dispatch_table.hpp>
#include <tenzor/nn/optim/scheduler.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include <tenzor/nn/layers/attention.hpp>
#include <tenzor/nn/layers/transformer.hpp>
#include <tenzor/nn/layers/embedding.hpp>
#include <tenzor/nn/layers/lazy_linear.hpp>
#include <tenzor/nn/layers/sync_batchnorm.hpp>
#include <tenzor/nn/optim/rmsprop.hpp>
#include <tenzor/nn/optim/adagrad.hpp>
#include <tenzor/nn/optim/adadelta.hpp>
#include <tenzor/nn/optim/radam.hpp>
#include <tenzor/nn/optim/nadam.hpp>
#include <tenzor/nn/optim/adamax.hpp>
#include <tenzor/nn/optim/lamb.hpp>
#include <tenzor/nn/optim/sparse_adam.hpp>
#include <tenzor/nn/init.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <tenzor/nn/loss/contrastive.hpp>
#include <tenzor/nn/callbacks.hpp>
#include <tenzor/nn/training.hpp>
#include <tenzor/nn/checkpoint.hpp>
#include <tenzor/backend/cuda_graph.hpp>
#include <tenzor/nn/mixed_precision.hpp>
#include <tenzor/nn/amp/grad_scaler.hpp>
#include <tenzor/nn/amp/autocast.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/distributed/ddp.hpp>
#include <tenzor/distributed/fsdp.hpp>
#include <tenzor/distributed/gradient_compression.hpp>
#include <tenzor/serving/server.hpp>
#include <tenzor/models/hub.hpp>
#include <tenzor/onnx/exporter.hpp>
#include <tenzor/export/export.hpp>
#include <tenzor/data/dataset.hpp>
#include <tenzor/data/datasets/mnist.hpp>
#include <tenzor/data/datasets/cifar10.hpp>
#include <tenzor/data/datasets/imagenet.hpp>
#include <tenzor/data/dataloader.hpp>
#include <tenzor/nn/compression/pruning.hpp>
#include <tenzor/nn/quantization.hpp>
#include <tenzor/jit/tracer.hpp>
#include <tenzor/jit/compiler.hpp>
#include <tenzor/jit/graph.hpp>
#include <tenzor/jit/serialization.hpp>
#include <tenzor/ops/vision.hpp>
#include <tenzor/ops/detection.hpp>
#include <tenzor/ops/async_ops.hpp>
#include <tenzor/ops/fused_ops.hpp>
#include <tenzor/data/transforms.hpp>
#include <tenzor/autograd/graph_viz.hpp>
#include <tenzor/utils/tensorboard.hpp>
#include <tenzor/utils/benchmark.hpp>
#include <tenzor/utils/monitor.hpp>
#include <tenzor/nn/optim/adam_atan2.hpp>
#include <tenzor/nn/layers/hrm.hpp>
#include <tenzor/nn/layers/moe.hpp>
#include <tenzor/nn/layers/alibi.hpp>
#include <tenzor/nn/layers/gqa_attention.hpp>
#include <tenzor/nn/layers/drop_path.hpp>
#include <tenzor/nn/layers/vision.hpp>
#include <tenzor/nn/layers/mobilenet.hpp>
#include <tenzor/nn/layers/segmentation.hpp>
#include <tenzor/nn/layers/sparse_linear.hpp>
#include <tenzor/nn/layers/sparse_embedding.hpp>
#include <tenzor/nn/serialize.hpp>
#include <tenzor/models/resnet.hpp>
#include <tenzor/models/vgg.hpp>
#include <tenzor/models/alexnet.hpp>
#include <tenzor/models/mobilenet.hpp>
#include <tenzor/models/efficientnet.hpp>
#include <tenzor/models/googlenet.hpp>
#include <tenzor/models/convnext.hpp>
#include <tenzor/models/vit.hpp>
#include <tenzor/models/swin_transformer.hpp>
#include <tenzor/models/bert.hpp>
#include <tenzor/models/roberta.hpp>
#include <tenzor/models/albert.hpp>
#include <tenzor/models/gpt.hpp>
#include <tenzor/models/t5.hpp>
#include <tenzor/models/electra.hpp>
#include <tenzor/models/unet.hpp>
#include <tenzor/models/deeplabv3plus.hpp>
#include <tenzor/models/yolo.hpp>
#include <tenzor/models/faster_rcnn.hpp>
#include <tenzor/models/mask_rcnn.hpp>
#include <tenzor/onnx/importer.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/fft.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>
#include <tenzor/sparse/sparse_ops.hpp>
#include <tenzor/autograd/anomaly_mode.hpp>
#include <tenzor/autograd/checkpoint.hpp>
#include <tenzor/autograd/functional.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/autograd/vmap.hpp>
#include <tenzor/autograd/dual.hpp>
#include <tenzor/backend/cpu_caching_allocator.hpp>
#include <tenzor/nn/utils/clip_grad.hpp>
#include <tenzor/nn/utils/rnn_utils.hpp>
#include <tenzor/utils/error.hpp>
#include <tenzor/utils/config.hpp>
#include <tenzor/utils/memory_profiler.hpp>
#include <tenzor/autograd/profiler.hpp>
#include <tenzor/backend/profiling_interceptor.hpp>
#include "numpy_interop.hpp"
#include <thread>
#include <cstdlib>

// ============================================================================
// Early OpenMP Configuration
// ============================================================================
// This constructor runs when the shared library is loaded, BEFORE main() or
// any module initialization. It sets OMP_NUM_THREADS environment variable
// to use all available hardware threads if not already set.
// This is necessary because once the OpenMP runtime is initialized (e.g., by
// MKL via NumPy/PyTorch), omp_set_num_threads() may not affect all runtimes.
__attribute__((constructor(101)))  // Priority 101 = very early
static void configure_openmp_threads() {
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
        std::string env_value = std::to_string(num_threads);
        setenv("OMP_NUM_THREADS", env_value.c_str(), 0);  // 0 = don't overwrite
    }
}

namespace py = pybind11;

// Forward declaration for compression bindings
void bind_compression(py::module& m);

// PyModule trampoline — moved to shared header for bindings_nn.cpp
#include "bindings/py_module_trampoline.hpp"

// ModuleList and ModuleDict are now implemented in C++ at tenzor::nn::ModuleList / ModuleDict

PYBIND11_MODULE(tenzor_core, m) {
    m.doc() = "Tenzor: High-performance tensor library";

    // ========================================================================
    // Custom Exception Hierarchy
    // ========================================================================
    // Register C++ exceptions as Python exceptions so they can be caught with
    // except tz.ShapeError, etc. Each derives from the base TenzorError which
    // itself derives from RuntimeError.
    static auto py_tenzor_error = py::register_exception<tenzor::TenzorException>(
        m, "TenzorError");
    py::register_exception<tenzor::ShapeException>(
        m, "ShapeError", py_tenzor_error.ptr());
    py::register_exception<tenzor::DTypeException>(
        m, "DTypeError", py_tenzor_error.ptr());
    py::register_exception<tenzor::DeviceException>(
        m, "DeviceError", py_tenzor_error.ptr());
    py::register_exception<tenzor::AutogradException>(
        m, "AutogradError", py_tenzor_error.ptr());
    py::register_exception<tenzor::BackendException>(
        m, "BackendError", py_tenzor_error.ptr());
    py::register_exception<tenzor::MemoryException>(
        m, "MemoryError", py_tenzor_error.ptr());
    // 5th-audit B'7: register every TenzorException-derived class that ships
    // in the public API. Pre-fix `TensorBoardException` was missing and would
    // surface only through the catch-all translator below, losing its
    // distinctive Python type (users couldn't write
    // `except tz.TensorBoardError`). The catch-all is kept as a safety net.
    py::register_exception<tenzor::TensorBoardException>(
        m, "TensorBoardError", py_tenzor_error.ptr());
    // Python-parity exception types — map to the corresponding Python builtins
    // so `except IndexError`, `except TypeError`, `except ValueError`,
    // `except NotImplementedError`, `except RuntimeError` all work as
    // Python users expect when calling Tenzor ops.
    py::register_exception<tenzor::IndexError>(
        m, "IndexError", PyExc_IndexError);
    py::register_exception<tenzor::ValueError>(
        m, "ValueError", PyExc_ValueError);
    py::register_exception<tenzor::TypeError>(
        m, "TypeError", PyExc_TypeError);
    py::register_exception<tenzor::NotImplementedError>(
        m, "NotImplementedError", PyExc_NotImplementedError);
    py::register_exception<tenzor::RuntimeError>(
        m, "RuntimeError", PyExc_RuntimeError);

    // Catch-all translator: any future TenzorException-derived types not
    // explicitly registered above will still map to TenzorError in Python.
    // This also covers the case where pybind11's per-class translator misses
    // a cross-DSO-boundary exception (RTTI mismatch); the catch matches by
    // base type.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const tenzor::TenzorException& e) {
            PyErr_SetString(py_tenzor_error.ptr(), e.what());
        }
    });

    // Library initialization
    m.def("initialize", &tenzor::initialize,
          py::call_guard<py::gil_scoped_release>(),
          "Initialize the Tenzor library (registers backends and operations)");

    m.def("mkl_cleanup", &tenzor::mkl_cleanup,
          "Free MKL internal buffers (call before using PyTorch/NumPy with MKL)");

    m.def("set_deterministic", &tenzor::set_deterministic, py::arg("mode"),
          "Enable or disable deterministic operations");
    m.def("is_deterministic", &tenzor::is_deterministic,
          "Check if deterministic mode is enabled");


    // Core bindings: DType, Device, Tensor, Variable, context managers
    // See python/bindings/bindings_core.cpp
    tenzor::python::register_core(m);

    // Future<Tensor> wrappers (audit C.6). Must precede any submodule that
    // returns a TensorFuture / TensorListFuture (async_ops, distributed.rpc).
    tenzor::python::register_future_types(m);

    // ========================================================================
    // Custom autograd Function base class
    // ========================================================================
    // Enables Python users to define custom differentiable operations:
    //   class MyReLU(tenzor.autograd.Function):
    //       @staticmethod
    //       def forward(ctx, input):
    //           ctx.save_for_backward(input)
    //           return tenzor.relu(input)
    //       @staticmethod
    //       def backward(ctx, grad_output):
    //           input, = ctx.saved_tensors
    //           grad = grad_output * (input > 0).to(tenzor.float32)
    //           return (grad,)

    // FunctionContext holds saved tensors and metadata for backward
    struct PyFunctionCtx {
        std::vector<tenzor::Tensor> saved_tensors_;

        void save_for_backward(py::args tensors) {
            saved_tensors_.clear();
            for (auto& t : tensors) {
                saved_tensors_.push_back(t.cast<tenzor::Tensor>());
            }
        }

        py::tuple saved_tensors() const {
            py::tuple result(saved_tensors_.size());
            for (size_t i = 0; i < saved_tensors_.size(); ++i) {
                result[i] = py::cast(saved_tensors_[i]);
            }
            return result;
        }
    };

    py::class_<PyFunctionCtx, std::shared_ptr<PyFunctionCtx>>(m, "FunctionCtx",
        "Context object for custom autograd Functions")
        .def("save_for_backward", &PyFunctionCtx::save_for_backward,
             "Save tensors for backward pass")
        .def_property_readonly("saved_tensors", &PyFunctionCtx::saved_tensors,
             "Get saved tensors");

    // PyCustomFunction bridges Python custom Functions to C++ autograd graph.
    //
    // 5th-audit B6 / B'9 (GIL safety + reference-cycle hygiene):
    //
    //   * Every method that touches `py_forward_fn_`, `py_backward_fn_`, or
    //     `ctx_` MUST hold the Python GIL. The C++ autograd engine releases
    //     the GIL around `Variable::backward()`, so calls into this object
    //     can arrive from arbitrary C++ worker threads. All such methods
    //     below begin with `py::gil_scoped_acquire`.
    //   * The destructor likewise acquires the GIL before dropping the
    //     `py::object` members — destroying a `py::object` without the GIL
    //     would race the CPython refcount.
    //   * Reference-cycle note: a Python user closure for `forward`/`backward`
    //     captured by a `staticmethod` is owned at the class level, not by
    //     a `PyCustomFunction` instance; the lifetime is bounded by the
    //     class object, which Python cleans up at interpreter shutdown.
    //     We never store `self` inside the closure ourselves, so no
    //     class -> instance -> closure -> instance cycle is introduced
    //     here. Future modifiers: keep `py_forward_fn_` and `py_backward_fn_`
    //     as Python staticmethod-compatible objects; do not bind `self`
    //     into either.
    struct PyCustomFunction : public tenzor::Function {
        py::object py_forward_fn_;  // Python static forward function
        py::object py_backward_fn_; // Python static backward function
        std::shared_ptr<PyFunctionCtx> ctx_;

        PyCustomFunction(py::object forward_fn, py::object backward_fn)
            : py_forward_fn_(std::move(forward_fn)),
              py_backward_fn_(std::move(backward_fn)),
              ctx_(std::make_shared<PyFunctionCtx>()) {}

        // Destructor must acquire the GIL before releasing py::object members
        // — Variable::backward() runs under py::gil_scoped_release, so any
        // cleanup path that drops the last shared_ptr to a PyCustomFunction
        // (e.g. BackwardEngine::cleanup_graph clearing its sorted vector)
        // would otherwise destroy py::object without the GIL, corrupting
        // CPython refcounts and terminating the process.
        ~PyCustomFunction() override {
            py::gil_scoped_acquire acquire;
            py_forward_fn_.release().dec_ref();
            py_backward_fn_.release().dec_ref();
            ctx_.reset();
        }

        auto forward(std::vector<tenzor::Variable> inputs) -> std::vector<tenzor::Variable> override {
            py::gil_scoped_acquire acquire;
            // 5th-audit B6 defensive assert: catch any future GIL-release
            // bypass at the soonest possible point. In Release builds this
            // is compiled out; in Debug it fires before any Python touch.
            assert(PyGILState_Check() && "PyCustomFunction::forward called without the GIL");
            try {
                // Build args: (ctx, *inputs_as_tensors)
                py::list args;
                args.append(py::cast(ctx_));
                for (auto& v : inputs) {
                    args.append(py::cast(v.tensor()));
                }
                auto result = py_forward_fn_(*py::tuple(args));

                // Result can be a single Tensor or tuple of Tensors
                std::vector<tenzor::Variable> outputs;
                if (py::isinstance<tenzor::Tensor>(result)) {
                    outputs.emplace_back(result.cast<tenzor::Tensor>(), false);
                } else {
                    auto result_tuple = result.cast<py::tuple>();
                    for (auto& item : result_tuple) {
                        outputs.emplace_back(item.cast<tenzor::Tensor>(), false);
                    }
                }
                // Save tensors from ctx into C++ Function's saved_tensors_
                // and record their version counters for in-place detection
                saved_tensors_ = ctx_->saved_tensors_;
                saved_versions_.clear();
                saved_versions_.reserve(saved_tensors_.size());
                for (auto& t : saved_tensors_) {
                    saved_versions_.push_back(t.version());
                }
                return outputs;
            } catch (py::error_already_set& e) {
                // 5th-audit B'6: explicit restore-and-rethrow preserves the
                // Python traceback. Pre-fix `throw;` propagated `e` by
                // reference; if pybind11 hadn't yet snapshotted the
                // PyErr_* state into `e`, the traceback could be lost
                // when the destination Python frame restored it.
                e.restore();
                throw py::error_already_set();
            }
        }

        auto backward(std::vector<tenzor::Tensor> grad_outputs) -> std::vector<tenzor::Tensor> override {
            py::gil_scoped_acquire acquire;
            assert(PyGILState_Check() && "PyCustomFunction::backward called without the GIL");
            try {
                // Restore ctx saved tensors
                ctx_->saved_tensors_ = saved_tensors_;

                py::list args;
                args.append(py::cast(ctx_));
                for (auto& g : grad_outputs) {
                    args.append(py::cast(g));
                }
                auto result = py_backward_fn_(*py::tuple(args));

                std::vector<tenzor::Tensor> grads;
                if (py::isinstance<tenzor::Tensor>(result)) {
                    grads.push_back(result.cast<tenzor::Tensor>());
                } else {
                    auto result_tuple = result.cast<py::tuple>();
                    for (auto& item : result_tuple) {
                        if (item.is_none()) {
                            grads.push_back(tenzor::Tensor{});
                        } else {
                            grads.push_back(item.cast<tenzor::Tensor>());
                        }
                    }
                }
                return grads;
            } catch (py::error_already_set& e) {
                // 5th-audit B'6: see forward() — restore-and-rethrow to
                // keep the Python traceback intact.
                e.restore();
                throw py::error_already_set();
            }
        }

        auto backward_with_variables(std::vector<tenzor::Variable> grad_outputs) -> std::vector<tenzor::Variable> override {
            // Extract tensors, call the Python backward, wrap results with requires_grad=true
            std::vector<tenzor::Tensor> tensor_grads;
            tensor_grads.reserve(grad_outputs.size());
            for (auto& var : grad_outputs) {
                tensor_grads.push_back(var.tensor());
            }

            auto result_tensors = backward(tensor_grads);

            std::vector<tenzor::Variable> result_vars;
            result_vars.reserve(result_tensors.size());
            for (auto& t : result_tensors) {
                result_vars.emplace_back(t, true);
            }
            return result_vars;
        }

        auto name() const -> std::string override { return "PyCustomFunction"; }
    };

    // Autograd + func transforms — see python/bindings/bindings_autograd.cpp
    tenzor::python::register_autograd(m);

    // PyCustomFunction trampoline must be registered after register_autograd
    // creates the autograd submodule, because its definition lives in this TU.
    {
        auto autograd_mod = m.attr("autograd").cast<py::module_>();
        auto custom_func_cls = py::class_<PyCustomFunction, std::shared_ptr<PyCustomFunction>>(
            autograd_mod, "_CustomFunctionImpl");
        (void)custom_func_cls;

        // Helper: apply() creates a PyCustomFunction, runs forward, wires into autograd
    autograd_mod.def("apply_custom_function", [](py::object py_cls, py::args inputs) {
        // Get forward/backward static methods from the class
        auto forward_fn = py_cls.attr("forward");
        auto backward_fn = py_cls.attr("backward");

        auto func = std::make_shared<PyCustomFunction>(forward_fn, backward_fn);

        // Convert inputs to Variables
        std::vector<tenzor::Variable> var_inputs;
        for (auto& inp : inputs) {
            if (py::isinstance<tenzor::Variable>(inp)) {
                var_inputs.push_back(inp.cast<tenzor::Variable>());
            } else if (py::isinstance<tenzor::Tensor>(inp)) {
                var_inputs.emplace_back(inp.cast<tenzor::Tensor>(), false);
            } else {
                throw std::runtime_error("apply_custom_function: inputs must be Variable or Tensor");
            }
        }

        // Run forward
        auto outputs = func->forward(var_inputs);

        // Wire outputs into autograd graph
        bool any_requires_grad = false;
        for (auto& v : var_inputs) {
            if (v.requires_grad()) {
                any_requires_grad = true;
                break;
            }
        }

        if (any_requires_grad) {
            func->set_input_variables(var_inputs);
            std::vector<std::shared_ptr<tenzor::Function>> next_fns;
            for (auto& v : var_inputs) {
                next_fns.push_back(v.grad_fn());
            }
            func->set_next_functions(next_fns);

            for (auto& out : outputs) {
                out.set_requires_grad(true);
                out.set_grad_fn(func);
            }
        }

        if (outputs.size() == 1) {
            return py::cast(outputs[0]);
        }
        py::tuple result_tuple(outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            result_tuple[i] = py::cast(outputs[i]);
        }
        return static_cast<py::object>(result_tuple);
    }, "Apply a custom autograd Function class");
    } // PyCustomFunction registration scope

    // Neural network — see python/bindings/bindings_nn.cpp
    tenzor::python::register_nn(m);

    // Optimizers — see python/bindings/bindings_optim.cpp
    tenzor::python::register_optim(m);

    // Distributed training — see python/bindings/bindings_distributed.cpp
    tenzor::python::register_distributed(m);

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
                             py::gil_scoped_acquire acquire;
                             progress_callback(downloaded, total, speed, eta);
                         } catch (const py::error_already_set& e) {
                             PyErr_WarnEx(PyExc_RuntimeWarning,
                                 (std::string("Callback error: ") + e.what()).c_str(), 1);
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
                             py::gil_scoped_acquire acquire;
                             progress_callback(downloaded, total, speed, eta);
                         } catch (const py::error_already_set& e) {
                             PyErr_WarnEx(PyExc_RuntimeWarning,
                                 (std::string("Callback error: ") + e.what()).c_str(), 1);
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
    torch_mod.def("can_zero_copy_from_torch", &tenzor::torch_interop::can_zero_copy_from_torch,
                  py::arg("torch_tensor"),
                  "Check if zero-copy conversion from PyTorch is possible");
    torch_mod.def("variable_to_torch", &tenzor::torch_interop::variable_to_torch,
                  py::arg("variable"),
                  "Convert Tenzor Variable to PyTorch Variable with autograd");
    torch_mod.def("variable_from_torch", &tenzor::torch_interop::variable_from_torch,
                  py::arg("torch_variable"),
                  "Convert PyTorch Variable to Tenzor Variable");
    torch_mod.def("dtype_to_torch", &tenzor::torch_interop::dtype_to_torch,
                  py::arg("dtype"),
                  "Map Tenzor DType to PyTorch ScalarType (as int)");
    torch_mod.def("dtype_from_torch", &tenzor::torch_interop::dtype_from_torch,
                  py::arg("torch_dtype"),
                  "Map PyTorch ScalarType (as int) to Tenzor DType");
    torch_mod.def("device_to_torch_string", &tenzor::torch_interop::device_to_torch_string,
                  py::arg("device"),
                  "Map Tenzor Device to PyTorch device string");
    torch_mod.def("device_from_torch_string", &tenzor::torch_interop::device_from_torch_string,
                  py::arg("device_str"),
                  "Map PyTorch device string to Tenzor Device");
    torch_mod.def("sync_gradients", &tenzor::torch_interop::sync_gradients,
                  py::arg("tenzor_var"), py::arg("torch_var"),
                  py::arg("tenzor_to_torch") = true,
                  "Synchronize gradient storage between Tenzor and PyTorch");
    #endif

    // =============================================================================
    // nn.init - Weight Initialization
    // =============================================================================
    auto nn_mod = m.attr("nn").cast<py::module_>();
    auto init = nn_mod.def_submodule("init", "Weight initialization utilities");

    py::enum_<tenzor::nn::init::FanMode>(init, "FanMode", "Fan mode for Kaiming initialization")
        .value("fan_in", tenzor::nn::init::FanMode::FanIn, "Use fan_in")
        .value("fan_out", tenzor::nn::init::FanMode::FanOut, "Use fan_out")
        .export_values();

    init.def("calculate_fan_in_and_fan_out", &tenzor::nn::init::calculate_fan_in_and_fan_out,
             "Calculate fan_in and fan_out for a tensor", py::arg("tensor"));
    init.def("calculate_gain", &tenzor::nn::init::calculate_gain,
             "Calculate recommended gain for a nonlinearity",
             py::arg("nonlinearity"), py::arg("param") = 0.01);
    init.def("xavier_uniform_", &tenzor::nn::init::xavier_uniform_,
             py::arg("tensor"), py::arg("gain") = 1.0, py::return_value_policy::reference);
    init.def("xavier_normal_", &tenzor::nn::init::xavier_normal_,
             py::arg("tensor"), py::arg("gain") = 1.0, py::return_value_policy::reference);
    init.def("kaiming_uniform_", &tenzor::nn::init::kaiming_uniform_,
             py::arg("tensor"), py::arg("a") = 0.0,
             py::arg("mode") = tenzor::nn::init::FanMode::FanIn,
             py::arg("nonlinearity") = "leaky_relu",
             py::return_value_policy::reference);
    init.def("kaiming_normal_", &tenzor::nn::init::kaiming_normal_,
             py::arg("tensor"), py::arg("a") = 0.0,
             py::arg("mode") = tenzor::nn::init::FanMode::FanIn,
             py::arg("nonlinearity") = "leaky_relu",
             py::return_value_policy::reference);
    init.def("lecun_uniform_", &tenzor::nn::init::lecun_uniform_,
             py::arg("tensor"), py::return_value_policy::reference);
    init.def("lecun_normal_", &tenzor::nn::init::lecun_normal_,
             py::arg("tensor"), py::return_value_policy::reference);
    init.def("orthogonal_", &tenzor::nn::init::orthogonal_,
             py::arg("tensor"), py::arg("gain") = 1.0, py::return_value_policy::reference);
    init.def("uniform_", &tenzor::nn::init::uniform_,
             py::arg("tensor"), py::arg("low") = 0.0, py::arg("high") = 1.0,
             py::return_value_policy::reference);
    init.def("normal_", &tenzor::nn::init::normal_,
             py::arg("tensor"), py::arg("mean") = 0.0, py::arg("std") = 1.0,
             py::return_value_policy::reference);
    init.def("constant_", &tenzor::nn::init::constant_,
             py::arg("tensor"), py::arg("value"), py::return_value_policy::reference);
    init.def("zeros_", &tenzor::nn::init::zeros_,
             py::arg("tensor"), py::return_value_policy::reference);
    init.def("ones_", &tenzor::nn::init::ones_,
             py::arg("tensor"), py::return_value_policy::reference);
    init.def("trunc_normal_", &tenzor::nn::init::trunc_normal_,
             py::arg("tensor"), py::arg("mean") = 0.0, py::arg("std") = 1.0,
             py::arg("a") = -2.0, py::arg("b") = 2.0, py::return_value_policy::reference);
    init.def("dirac_", &tenzor::nn::init::dirac_,
             py::arg("tensor"), py::arg("groups") = 1, py::return_value_policy::reference);
    init.def("sparse_", &tenzor::nn::init::sparse_,
             py::arg("tensor"), py::arg("sparsity"), py::arg("std") = 0.01,
             py::return_value_policy::reference);

    // Data loading utilities
    auto data_mod = m.def_submodule("data", "Data loading and dataset utilities");

    // WorkerInfo struct for distributed sharding of IterableDatasets
    py::class_<tenzor::data::WorkerInfo>(data_mod, "WorkerInfo",
             "Metadata describing the current DataLoader worker context")
        .def(py::init<>())
        .def(py::init([](int worker_id, int num_workers, int64_t seed, int rank, int world_size) {
            tenzor::data::WorkerInfo info;
            info.worker_id = worker_id;
            info.num_workers = num_workers;
            info.seed = seed;
            info.rank = rank;
            info.world_size = world_size;
            return info;
        }), py::arg("worker_id") = 0, py::arg("num_workers") = 1,
            py::arg("seed") = 0, py::arg("rank") = 0, py::arg("world_size") = 1)
        .def_readwrite("worker_id", &tenzor::data::WorkerInfo::worker_id,
                       "Index of this worker within the DataLoader (0-based)")
        .def_readwrite("num_workers", &tenzor::data::WorkerInfo::num_workers,
                       "Total number of workers in this DataLoader")
        .def_readwrite("seed", &tenzor::data::WorkerInfo::seed,
                       "Per-worker random seed for reproducibility")
        .def_readwrite("rank", &tenzor::data::WorkerInfo::rank,
                       "Distributed rank of this process")
        .def_readwrite("world_size", &tenzor::data::WorkerInfo::world_size,
                       "Total number of distributed processes")
        .def("__repr__", [](const tenzor::data::WorkerInfo& info) {
            return "WorkerInfo(worker_id=" + std::to_string(info.worker_id) +
                   ", num_workers=" + std::to_string(info.num_workers) +
                   ", seed=" + std::to_string(info.seed) +
                   ", rank=" + std::to_string(info.rank) +
                   ", world_size=" + std::to_string(info.world_size) + ")";
        });

    data_mod.def("get_worker_info", &tenzor::data::get_worker_info,
                 R"pbdoc(
                     Get the WorkerInfo for the current DataLoader worker thread.

                     Returns None if called outside a DataLoader worker thread.

                     Returns:
                         WorkerInfo or None
                 )pbdoc");
    data_mod.def("set_worker_info", &tenzor::data::set_worker_info,
                 py::arg("info"),
                 "Set the WorkerInfo for the current thread");
    data_mod.def("clear_worker_info", &tenzor::data::clear_worker_info,
                 "Clear the WorkerInfo for the current thread");

    // Dataset abstract base class (must be registered before derived classes)
    py::class_<tenzor::data::Dataset, std::shared_ptr<tenzor::data::Dataset>>(data_mod, "Dataset")
        .def("size", &tenzor::data::Dataset::size,
             "Get the number of samples in the dataset")
        .def("__len__", &tenzor::data::Dataset::size,
             "Get the number of samples in the dataset")
        .def("get", &tenzor::data::Dataset::get,
             py::arg("index"),
             "Get a sample at the specified index")
        .def("__getitem__", &tenzor::data::Dataset::get,
             py::arg("index"),
             "Get a sample at the specified index")
        .def("empty", &tenzor::data::Dataset::empty,
             "Check if the dataset is empty");

    // IterableDataset (derives from Dataset, so registered after)
    py::class_<tenzor::data::IterableDataset, tenzor::data::Dataset,
               std::shared_ptr<tenzor::data::IterableDataset>>(data_mod, "IterableDataset",
             "Abstract base class for iterable-style streaming datasets");

    // TensorDataset
    py::class_<tenzor::data::TensorDataset, tenzor::data::Dataset,
               std::shared_ptr<tenzor::data::TensorDataset>>(data_mod, "TensorDataset")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("inputs"), py::arg("targets"),
             R"pbdoc(
                Create a dataset from input and target tensors.

                Args:
                    inputs: Input tensor with shape [N, ...]
                    targets: Target tensor with shape [N, ...]

                Example:
                    >>> inputs = tenzor.randn([100, 10])
                    >>> targets = tenzor.randint(0, 2, [100])
                    >>> dataset = tenzor.data.TensorDataset(inputs, targets)
             )pbdoc")
        .def("size", &tenzor::data::TensorDataset::size)
        .def("__len__", &tenzor::data::TensorDataset::size)
        .def("get", &tenzor::data::TensorDataset::get, py::arg("index"))
        .def("__getitem__", &tenzor::data::TensorDataset::get, py::arg("index"));

    // DataLoaderConfig
    py::class_<tenzor::data::DataLoaderConfig>(data_mod, "DataLoaderConfig")
        .def(py::init<>())
        .def_readwrite("batch_size", &tenzor::data::DataLoaderConfig::batch_size,
                       "Number of samples per batch")
        .def_readwrite("shuffle", &tenzor::data::DataLoaderConfig::shuffle,
                       "Whether to shuffle data each epoch")
        .def_readwrite("num_workers", &tenzor::data::DataLoaderConfig::num_workers,
                       "Number of worker threads for parallel loading")
        .def_readwrite("pin_memory", &tenzor::data::DataLoaderConfig::pin_memory,
                       "Pin memory for faster CUDA transfer")
        .def_readwrite("drop_last", &tenzor::data::DataLoaderConfig::drop_last,
                       "Drop last incomplete batch")
        .def_readwrite("prefetch_factor", &tenzor::data::DataLoaderConfig::prefetch_factor,
                       "Number of batches to prefetch per worker");

    // Batch struct
    py::class_<tenzor::data::Batch>(data_mod, "Batch")
        .def(py::init<>())
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("inputs"), py::arg("targets"))
        .def_readwrite("inputs", &tenzor::data::Batch::inputs,
                       "Batched input tensor")
        .def_readwrite("targets", &tenzor::data::Batch::targets,
                       "Batched target tensor");

    // DataLoader
    py::class_<tenzor::data::DataLoader>(data_mod, "DataLoader")
        .def(py::init<std::shared_ptr<tenzor::data::Dataset>, const tenzor::data::DataLoaderConfig&>(),
             py::arg("dataset"), py::arg("config"),
             "Create DataLoader with configuration object")
        .def(py::init<std::shared_ptr<tenzor::data::Dataset>, size_t, bool, size_t, bool, bool>(),
             py::arg("dataset"),
             py::arg("batch_size"),
             py::arg("shuffle") = false,
             py::arg("num_workers") = 0,
             py::arg("pin_memory") = false,
             py::arg("drop_last") = false,
             R"pbdoc(
                Create a DataLoader for efficient batch loading.

                Args:
                    dataset: Dataset to load from
                    batch_size: Number of samples per batch
                    shuffle: Whether to shuffle data at the start of each epoch
                    num_workers: Number of worker threads (0 = single-threaded)
                    pin_memory: Pin memory for faster CUDA transfer
                    drop_last: Drop the last incomplete batch

                Example:
                    >>> dataset = tenzor.data.TensorDataset(inputs, targets)
                    >>> loader = tenzor.data.DataLoader(dataset, batch_size=32,
                    ...                                  shuffle=True, num_workers=4)
                    >>> for batch in loader:
                    ...     print(batch.inputs.shape, batch.targets.shape)
             )pbdoc")
        .def("__iter__", [](tenzor::data::DataLoader& self) {
            return py::make_iterator(self.begin(), self.end());
        }, py::keep_alive<0, 1>(),
        "Iterate over batches")
        .def("__len__", &tenzor::data::DataLoader::size,
             "Get the number of batches per epoch")
        .def("size", &tenzor::data::DataLoader::size,
             "Get the number of batches per epoch")
        .def("reset", &tenzor::data::DataLoader::reset,
             "Reset loader for new epoch (reshuffles if shuffle is enabled)");

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

    // ONNXExportValueInfo class (for exporting)
    py::class_<tenzor::onnx::ONNXExportValueInfo>(onnx_mod, "ValueInfo")
        .def(py::init<const std::string&, tenzor::onnx::ONNXDataType, const std::vector<int64_t>&>(),
             py::arg("name"), py::arg("dtype"), py::arg("shape"))
        .def_readwrite("name", &tenzor::onnx::ONNXExportValueInfo::name)
        .def_readwrite("dtype", &tenzor::onnx::ONNXExportValueInfo::dtype)
        .def_readwrite("shape", &tenzor::onnx::ONNXExportValueInfo::shape);

    // ONNXExportNode class (for exporting)
    py::class_<tenzor::onnx::ONNXExportNode>(onnx_mod, "Node")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("op_type"), py::arg("name"))
        .def_readwrite("op_type", &tenzor::onnx::ONNXExportNode::op_type)
        .def_readwrite("name", &tenzor::onnx::ONNXExportNode::name)
        .def_readwrite("inputs", &tenzor::onnx::ONNXExportNode::inputs)
        .def_readwrite("outputs", &tenzor::onnx::ONNXExportNode::outputs)
        .def("add_input", &tenzor::onnx::ONNXExportNode::add_input, py::arg("input"))
        .def("add_output", &tenzor::onnx::ONNXExportNode::add_output, py::arg("output"));

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
        .def("add_output", [](tenzor::onnx::ONNXExporter& self,
                               const tenzor::Tensor& tensor,
                               const std::string& name,
                               const std::unordered_map<int64_t, std::string>& dynamic_axes) {
                 self.add_output(tensor, name, dynamic_axes);
             },
             py::arg("tensor"), py::arg("name"),
             py::arg("dynamic_axes") = std::unordered_map<int64_t, std::string>(),
             "Add model output")
        // 6th-audit Fix #2: `export_to_file` is overloaded since 5th-audit C6
        // (legacy 1-arg form + 3-arg external_data form). Disambiguate via
        // explicit member-function-pointer cast — pybind11 cannot deduce the
        // overload set from a bare `&Class::method`.
        .def("export_to_file",
             static_cast<void (tenzor::onnx::ONNXExporter::*)(const std::string&)>(
                 &tenzor::onnx::ONNXExporter::export_to_file),
             py::arg("filepath"),
             "Export model to a single ONNX file (legacy single-file form).")
        // 6th-audit Fix #2: expose the external_data overload so Python users
        // can actually export >2 GB models. Defaults match the C++ API:
        // `use_external_data=None` → auto-enable at >1.5 GB initializer total;
        // `external_data_threshold_bytes=1 MiB` per-tensor.
        .def("export_to_file",
             static_cast<void (tenzor::onnx::ONNXExporter::*)(
                 const std::string&, std::optional<bool>, size_t)>(
                 &tenzor::onnx::ONNXExporter::export_to_file),
             py::arg("filepath"),
             py::arg("use_external_data") = std::optional<bool>{},
             py::arg("external_data_threshold_bytes") = size_t{1ULL << 20},
             "Export model to ONNX, optionally splitting large initializers "
             "into a sidecar .data file. Set `use_external_data=True` to "
             "force external data, `False` for single-file, or `None` (default) "
             "to auto-enable when total initializer bytes exceed 1.5 GB.")
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

    // Utility function to convert ONNX DataType to DType
    onnx_mod.def("onnx_to_dtype", &tenzor::onnx::onnx_to_dtype,
                 py::arg("onnx_dtype"),
                 "Convert ONNX DataType to Tenzor DType");

    // =========================================================================
    // ONNX Importer
    // =========================================================================

    // ONNXImporter class
    py::class_<tenzor::onnx::ONNXImporter>(onnx_mod, "Importer",
        "Import ONNX models into Tenzor")
        .def(py::init<bool>(),
             py::arg("verbose") = false,
             "Create ONNX importer with optional verbose output")
        .def("import_from_file", &tenzor::onnx::ONNXImporter::import_from_file,
             py::arg("filepath"),
             "Import model from ONNX file")
        .def("import_from_bytes", &tenzor::onnx::ONNXImporter::import_from_bytes,
             py::arg("bytes"),
             "Import model from ONNX bytes")
        .def("get_model_data", &tenzor::onnx::ONNXImporter::get_model_data,
             py::return_value_policy::reference_internal,
             "Get the parsed ONNX model data")
        .def("set_verbose", &tenzor::onnx::ONNXImporter::set_verbose,
             py::arg("verbose"),
             "Enable or disable verbose output")
        .def("set_device", &tenzor::onnx::ONNXImporter::set_device,
             py::arg("device"),
             "Set target device for imported model");

    // GraphModule for DAG-structured ONNX models
    py::class_<tenzor::onnx::GraphModule, tenzor::nn::Module,
               std::shared_ptr<tenzor::onnx::GraphModule>>(onnx_mod, "GraphModule",
        "DAG-aware module for executing non-sequential ONNX graphs")
        .def(py::init<>())
        .def("add_constant", &tenzor::onnx::GraphModule::add_constant,
             py::arg("name"), py::arg("value"))
        .def("set_input_names", &tenzor::onnx::GraphModule::set_input_names,
             py::arg("names"))
        .def("set_output_names", &tenzor::onnx::GraphModule::set_output_names,
             py::arg("names"))
        .def("num_ops", &tenzor::onnx::GraphModule::num_ops);

    // ONNXModelData struct (read-only access)
    py::class_<tenzor::onnx::ONNXModelData>(onnx_mod, "ModelData",
        "ONNX model metadata")
        .def_readonly("ir_version", &tenzor::onnx::ONNXModelData::ir_version)
        .def_readonly("opset_version", &tenzor::onnx::ONNXModelData::opset_version)
        .def_readonly("model_version", &tenzor::onnx::ONNXModelData::model_version)
        .def_readonly("producer_name", &tenzor::onnx::ONNXModelData::producer_name)
        .def_readonly("doc_string", &tenzor::onnx::ONNXModelData::doc_string);

    // High-level import function
    onnx_mod.def("load", &tenzor::onnx::import_onnx,
                 py::arg("filepath"),
                 py::arg("verbose") = false,
                 R"pbdoc(
            Load an ONNX model and convert it to a Tenzor module.

            Args:
                filepath: Path to the ONNX model file
                verbose: Print import progress (default: False)

            Returns:
                A Tenzor nn.Module representing the imported model

            Example:
                >>> model = tenzor.onnx.load("model.onnx", verbose=True)
                >>> output = model(input_tensor)
        )pbdoc");

    // ========== Automatic Mixed Precision (AMP) ==========
    auto amp = m.def_submodule("amp", "Automatic Mixed Precision utilities");

    // GradScaler class
    py::class_<tenzor::nn::amp::GradScaler>(amp, "GradScaler",
        R"pbdoc(
            Gradient scaler for automatic mixed precision training.

            Helps prevent gradient underflow when training with FP16/mixed precision
            by scaling the loss before backward() and unscaling gradients before optimizer.step().
        )pbdoc")
        .def(py::init<float, float, float, int>(),
             py::arg("init_scale") = 65536.0f,
             py::arg("growth_factor") = 2.0f,
             py::arg("backoff_factor") = 0.5f,
             py::arg("growth_interval") = 2000,
             "Create gradient scaler with specified parameters")
        // audit-10 OO.8: release the GIL across the C++ kernel work in
        // scale/unscale_/step/update/found_inf_nan. These touch parameter
        // gradient buffers and dispatch backend kernels; holding the GIL
        // serialised AMP training against DataLoader workers and any other
        // Python thread.
        .def("scale", &tenzor::nn::amp::GradScaler::scale,
             py::arg("loss"),
             "Scale loss by current scale factor",
             py::call_guard<py::gil_scoped_release>())
        .def("unscale_", &tenzor::nn::amp::GradScaler::unscale_,
             py::arg("optimizer"),
             "Unscale gradients in optimizer parameters",
             py::call_guard<py::gil_scoped_release>())
        .def("step", &tenzor::nn::amp::GradScaler::step,
             py::arg("optimizer"),
             "Execute optimizer step with overflow detection",
             py::call_guard<py::gil_scoped_release>())
        .def("update", &tenzor::nn::amp::GradScaler::update,
             "Update scale factor based on overflow history",
             py::call_guard<py::gil_scoped_release>())
        .def("get_scale", &tenzor::nn::amp::GradScaler::get_scale,
             "Get current scale factor")
        .def("get_growth_tracker", &tenzor::nn::amp::GradScaler::get_growth_tracker,
             "Get number of consecutive successful iterations")
        .def("found_inf_nan", &tenzor::nn::amp::GradScaler::found_inf_nan,
             "Check if overflow was detected in last step",
             py::call_guard<py::gil_scoped_release>())
        .def("reset", &tenzor::nn::amp::GradScaler::reset,
             "Reset scaler to initial state")
        .def("state_dict", &tenzor::nn::amp::GradScaler::state_dict,
             "Get scaler state for serialization")
        .def("load_state_dict", &tenzor::nn::amp::GradScaler::load_state_dict,
             py::arg("state"),
             "Load scaler state from dictionary");

    // Autocast context manager — uses a wrapper to defer activation to __enter__.
    // HH.23: previously held a ``std::vector<unique_ptr<Autocast>>`` per
    // instance, mutated by __enter__ / __exit__ without a lock. If the same
    // PyAutocastContext Python object was shared across threads (e.g. a
    // module-level ``ac = autocast(...)`` used by ``with ac: ...`` from a
    // DataLoader worker and the main thread) the unsynchronised vector
    // push/pop is a data race and the LIFO ordering breaks. Simpler fix:
    // store a single ``unique_ptr<Autocast>`` slot per instance. __enter__
    // populates it, __exit__ resets it. Re-using the same PyAutocastContext
    // for nested ``with`` blocks within a single thread (the original
    // stacking motivation) was never a well-defined pattern; document that
    // each ``with`` should construct a fresh ``autocast(...)`` call. The
    // copy ctor stays explicitly deleted to keep pybind11 from instantiating
    // a copy path for ``__enter__ -> Self&``.
    struct PyAutocastContext {
        bool enabled_;
        tenzor::DType dtype_;
        tenzor::Device::Type device_type_;
        std::unique_ptr<tenzor::nn::amp::Autocast> guard_;

        PyAutocastContext(bool enabled, tenzor::DType dtype, tenzor::Device::Type device_type)
            : enabled_(enabled), dtype_(dtype), device_type_(device_type) {}

        PyAutocastContext(const PyAutocastContext&) = delete;
        PyAutocastContext& operator=(const PyAutocastContext&) = delete;

        void enter() {
            // Reuse across threads is undefined; the second thread's
            // ``__enter__`` would overwrite the first guard. Single-slot
            // semantics: a fresh ``autocast(...)`` per ``with`` block is
            // the supported pattern.
            //
            // LL.10: also reject re-entry on the same instance within a
            // single thread (``with ac: with ac: ...``). The inner
            // ``__enter__`` would otherwise overwrite ``guard_`` and
            // destroy the outer guard early, silently corrupting the
            // thread-local autocast stack. Matches PyTorch's documented
            // behaviour: each ``with`` needs a fresh ``autocast(...)``.
            if (guard_) {
                throw std::runtime_error(
                    "autocast context is not reentrant — use a fresh "
                    "tz.amp.Autocast() per nested scope");
            }
            guard_ = std::make_unique<tenzor::nn::amp::Autocast>(enabled_, dtype_, device_type_);
        }

        void exit() {
            guard_.reset();  // destructor restores prior thread-local state
        }
    };

    py::class_<PyAutocastContext>(amp, "Autocast",
        R"pbdoc(
            Automatic mixed precision context manager.

            Automatically casts operations to lower precision (Float16 or BFloat16)
            for performance while maintaining numerical stability.

            Usage:
                with tz.amp.Autocast():
                    output = model(input)  # Ops auto-cast to FP16
        )pbdoc")
        .def(py::init<bool, tenzor::DType, tenzor::Device::Type>(),
             py::arg("enabled") = true,
             py::arg("dtype") = tenzor::DType::Float16,
             py::arg("device_type") = tenzor::Device::Type::CUDA,
             "Create autocast context")
        .def("__enter__", [](PyAutocastContext& self) -> PyAutocastContext& {
            self.enter();
            return self;
        })
        .def("__exit__", [](PyAutocastContext& self, py::object, py::object, py::object) {
            self.exit();
            return false;
        })
        .def_static("is_enabled", &tenzor::nn::amp::Autocast::is_enabled,
                    "Check if autocast is currently enabled")
        .def_static("get_dtype", &tenzor::nn::amp::Autocast::get_dtype,
                    "Get the current autocast dtype")
        .def_static("get_device_type", &tenzor::nn::amp::Autocast::get_device_type,
                    "Get the current autocast device type")
        .def_static("should_autocast", &tenzor::nn::amp::Autocast::should_autocast,
                    py::arg("op_name"), py::arg("device"),
                    "Determine if a given operation should be autocast");

    // MixedPrecisionConfig struct
    py::class_<tenzor::nn::MixedPrecisionConfig>(m, "MixedPrecisionConfig",
        "Configuration for mixed precision training")
        .def(py::init<>())
        .def_readwrite("dtype", &tenzor::nn::MixedPrecisionConfig::dtype,
                      "Target dtype for mixed precision (Float16 or BFloat16)")
        .def_readwrite("device_type", &tenzor::nn::MixedPrecisionConfig::device_type,
                      "Device type to apply mixed precision")
        .def_readwrite("enabled", &tenzor::nn::MixedPrecisionConfig::enabled,
                      "Enable automatic mixed precision")
        .def_readwrite("init_scale", &tenzor::nn::MixedPrecisionConfig::init_scale,
                      "Initial loss scale for gradient scaler")
        .def_readwrite("growth_factor", &tenzor::nn::MixedPrecisionConfig::growth_factor,
                      "Scale growth factor on successful iterations")
        .def_readwrite("backoff_factor", &tenzor::nn::MixedPrecisionConfig::backoff_factor,
                      "Scale backoff factor on overflow")
        .def_readwrite("growth_interval", &tenzor::nn::MixedPrecisionConfig::growth_interval,
                      "Iterations before attempting scale growth")
        .def_static("fp16_cuda", &tenzor::nn::MixedPrecisionConfig::fp16_cuda,
                   "Create default FP16 configuration for CUDA")
        .def_static("bfloat16_cuda", &tenzor::nn::MixedPrecisionConfig::bfloat16_cuda,
                   "Create BFloat16 configuration for CUDA")
        .def_static("conservative", &tenzor::nn::MixedPrecisionConfig::conservative,
                   "Create conservative configuration (slower scale growth)");

    // MixedPrecisionTrainer class
    py::class_<tenzor::nn::MixedPrecisionTrainer>(m, "MixedPrecisionTrainer",
        R"pbdoc(
            High-level mixed precision training wrapper.

            Provides a complete training API with automatic mixed precision (AMP)
            and gradient scaling. Handles:
            - Automatic casting of operations to FP16/BF16
            - Loss scaling to prevent gradient underflow
            - Gradient unscaling and overflow detection
            - Dynamic loss scale adjustment

            Example:
                >>> model = tenzor.nn.Linear(10, 5)
                >>> optimizer = tenzor.optim.Adam(model.parameters(), 0.001)
                >>> loss_fn = lambda pred, target: (pred - target).pow(2).mean()
                >>> config = tenzor.MixedPrecisionConfig.fp16_cuda()
                >>> trainer = tenzor.MixedPrecisionTrainer(model, optimizer, loss_fn, config)
                >>> # Train with mixed precision
                >>> for inputs, targets in dataloader:
                >>>     loss = trainer.train_step(inputs, targets)
        )pbdoc")
        .def(py::init<std::shared_ptr<tenzor::nn::Module>,
                     std::shared_ptr<tenzor::optim::Optimizer>,
                     std::function<tenzor::Variable(const tenzor::Variable&, const tenzor::Variable&)>,
                     const tenzor::nn::MixedPrecisionConfig&>(),
             py::arg("model"),
             py::arg("optimizer"),
             py::arg("loss_fn"),
             py::arg("config") = tenzor::nn::MixedPrecisionConfig::fp16_cuda(),
             "Create mixed precision trainer")
        .def("train_step", &tenzor::nn::MixedPrecisionTrainer::train_step,
             py::arg("input"), py::arg("target"),
             "Perform single training step with mixed precision")
        .def("eval_step", &tenzor::nn::MixedPrecisionTrainer::eval_step,
             py::arg("input"), py::arg("target"),
             "Perform evaluation step (no mixed precision)")
        .def("fit", &tenzor::nn::MixedPrecisionTrainer::fit,
             py::arg("train_loader"),
             py::arg("epochs"),
             py::arg("val_loader") = nullptr,
             py::arg("callbacks") = std::vector<std::shared_ptr<tenzor::nn::Callback>>(),
             "Train model for multiple epochs with mixed precision")
        .def("train", &tenzor::nn::MixedPrecisionTrainer::train,
             "Set model to training mode")
        .def("eval", &tenzor::nn::MixedPrecisionTrainer::eval,
             "Set model to evaluation mode")
        .def("is_training", &tenzor::nn::MixedPrecisionTrainer::is_training,
             "Check if model is in training mode")
        .def("model", &tenzor::nn::MixedPrecisionTrainer::model,
             "Get underlying model")
        .def("optimizer", &tenzor::nn::MixedPrecisionTrainer::optimizer,
             "Get optimizer")
        .def("scaler", &tenzor::nn::MixedPrecisionTrainer::scaler,
             py::return_value_policy::reference_internal,
             "Get gradient scaler")
        .def("get_scale", &tenzor::nn::MixedPrecisionTrainer::get_scale,
             "Get current loss scale")
        .def("get_skipped_steps", &tenzor::nn::MixedPrecisionTrainer::get_skipped_steps,
             "Get number of skipped steps due to overflow")
        .def("get_total_steps", &tenzor::nn::MixedPrecisionTrainer::get_total_steps,
             "Get total number of training steps")
        .def("get_config", &tenzor::nn::MixedPrecisionTrainer::get_config,
             py::return_value_policy::reference_internal,
             "Get mixed precision configuration")
        .def("reset_stats", &tenzor::nn::MixedPrecisionTrainer::reset_stats,
             "Reset training statistics");

    // Helper functions
    m.def("create_fp16_trainer", &tenzor::nn::create_fp16_trainer,
          py::arg("model"), py::arg("optimizer"), py::arg("loss_fn"),
          "Create FP16 mixed precision trainer");
    m.def("create_bfloat16_trainer", &tenzor::nn::create_bfloat16_trainer,
          py::arg("model"), py::arg("optimizer"), py::arg("loss_fn"),
          "Create BFloat16 mixed precision trainer");

    // =========================================================================
    // Linear Algebra (linalg) submodule
    // =========================================================================
    // tenzor.linalg submodule (extracted to python/bindings/bindings_linalg.cpp
    // as part of P3.4). The new TU also binds the P2.1 additions
    // lstsq / pinv / matrix_exp which were previously C++-only.
    tenzor::python::register_linalg(m);

    // =========================================================================
    // Advanced Operations
    // =========================================================================
    m.def("einsum", [](const std::string& equation, const std::vector<tenzor::Tensor>& tensors) {
        return tenzor::einsum(equation, tensors);
    }, "Einstein summation",
       py::arg("equation"), py::arg("tensors"),
       py::call_guard<py::gil_scoped_release>());

    m.def("topk", [](const tenzor::Tensor& input, int64_t k, int64_t dim, bool largest, bool sorted) {
        return tenzor::topk(input, k, dim, largest, sorted);
    }, "Find top-k elements",
       py::arg("input"), py::arg("k"), py::arg("dim") = -1,
       py::arg("largest") = true, py::arg("sorted") = true,
       py::call_guard<py::gil_scoped_release>());

    m.def("sort", [](const tenzor::Tensor& input, int64_t dim, bool descending) {
        return tenzor::sort(input, dim, descending);
    }, "Sort tensor along dimension",
       py::arg("input"), py::arg("dim") = -1, py::arg("descending") = false,
       py::call_guard<py::gil_scoped_release>());

    m.def("unique", [](const tenzor::Tensor& input, bool sorted, bool return_inverse, bool return_counts) {
        return tenzor::unique(input, sorted, return_inverse, return_counts);
    }, "Find unique elements",
       py::arg("input"), py::arg("sorted") = true,
       py::arg("return_inverse") = false, py::arg("return_counts") = false,
       py::call_guard<py::gil_scoped_release>());

    m.def("cumsum", [](const tenzor::Tensor& input, int64_t dim) {
        return tenzor::cumsum(input, dim);
    }, "Cumulative sum along dimension",
       py::arg("input"), py::arg("dim"),
       py::call_guard<py::gil_scoped_release>());

    m.def("cumprod", [](const tenzor::Tensor& input, int64_t dim) {
        return tenzor::cumprod(input, dim);
    }, "Cumulative product along dimension",
       py::arg("input"), py::arg("dim"),
       py::call_guard<py::gil_scoped_release>());

    m.def("logcumsumexp", [](const tenzor::Tensor& input, int64_t dim) {
        return tenzor::logcumsumexp(input, dim);
    }, "Log-cumulative-sum-exp along dimension (numerically stable)",
       py::arg("input"), py::arg("dim"),
       py::call_guard<py::gil_scoped_release>());

    m.def("bincount", [](const tenzor::Tensor& input,
                         std::optional<tenzor::Tensor> weights,
                         int64_t minlength) {
        return tenzor::bincount(input, weights, minlength);
    }, "Count occurrences of each value in an integer tensor",
       py::arg("input"), py::arg("weights") = py::none(),
       py::arg("minlength") = 0,
       py::call_guard<py::gil_scoped_release>());

    m.def("index_reduce", [](const tenzor::Tensor& input, int64_t dim,
                              const tenzor::Tensor& index,
                              const tenzor::Tensor& source,
                              const std::string& reduce,
                              bool include_self) {
        return tenzor::index_reduce(input, dim, index, source, reduce, include_self);
    }, "Reduce source into input at specified indices along a dimension",
       py::arg("input"), py::arg("dim"), py::arg("index"),
       py::arg("source"), py::arg("reduce"),
       py::arg("include_self") = true,
       py::call_guard<py::gil_scoped_release>());

    m.def("median", [](const tenzor::Tensor& input, int64_t dim, bool keepdim) {
        return tenzor::median(input, dim, keepdim);
    }, "Median along dimension",
       py::arg("input"), py::arg("dim") = -1, py::arg("keepdim") = false,
       py::call_guard<py::gil_scoped_release>());

    m.def("mode", [](const tenzor::Tensor& input, int64_t dim, bool keepdim) {
        return tenzor::mode(input, dim, keepdim);
    }, "Mode (most frequent value) along dimension",
       py::arg("input"), py::arg("dim") = -1, py::arg("keepdim") = false,
       py::call_guard<py::gil_scoped_release>());

    // =========================================================================
    // FFT submodule (extracted to python/bindings/bindings_fft.cpp as part
    // of P3.4). The new TU also binds the P2.7 additions fftshift /
    // ifftshift / hfft / ihfft which were previously C++-only.
    // =========================================================================
    tenzor::python::register_fft(m);

    // =========================================================================
    // JIT Module - Tracing, Compilation, and Graph Optimization
    // See python/bindings/bindings_jit.cpp
    // =========================================================================
    tenzor::python::register_jit(m);


    // =========================================================================
    // Vision Operations
    // =========================================================================
    // tenzor.vision / detection / async_ops / fused submodules, extracted
    // to python/bindings/bindings_vision_detection.cpp as part of P3.4.
    tenzor::python::register_vision_detection(m);

    // =========================================================================
    // Data Transforms
    // =========================================================================
    auto transforms = data_mod.def_submodule("transforms", "Data augmentation transforms");

    py::class_<tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::Transform>>(transforms, "Transform",
        "Base class for data transforms")
        .def("__call__", &tenzor::data::transforms::Transform::operator());

    py::class_<tenzor::data::transforms::Normalize,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::Normalize>>(transforms, "Normalize",
        "Normalize tensor with mean and std")
        .def(py::init<std::vector<float>, std::vector<float>>(),
             py::arg("mean"), py::arg("std"));

    py::class_<tenzor::data::transforms::ToTensor,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::ToTensor>>(transforms, "ToTensor",
        "Convert to tensor (identity for tensors)")
        .def(py::init<>());

    py::class_<tenzor::data::transforms::Compose,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::Compose>>(transforms, "Compose",
        "Compose multiple transforms")
        .def(py::init<std::vector<std::shared_ptr<tenzor::data::transforms::Transform>>>(),
             py::arg("transforms"));

    py::class_<tenzor::data::transforms::RandomHorizontalFlip,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::RandomHorizontalFlip>>(transforms, "RandomHorizontalFlip",
        "Random horizontal flip")
        .def(py::init<float>(),
             py::arg("p") = 0.5f);

    py::class_<tenzor::data::transforms::RandomVerticalFlip,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::RandomVerticalFlip>>(transforms, "RandomVerticalFlip",
        "Random vertical flip")
        .def(py::init<float>(),
             py::arg("p") = 0.5f);

    py::class_<tenzor::data::transforms::CenterCrop,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::CenterCrop>>(transforms, "CenterCrop",
        "Center crop to target size")
        .def(py::init<int64_t, int64_t>(),
             py::arg("height"), py::arg("width"));

    py::class_<tenzor::data::transforms::RandomCrop,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::RandomCrop>>(transforms, "RandomCrop",
        "Random crop with optional padding")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("height"), py::arg("width"), py::arg("padding") = 0);

    py::class_<tenzor::data::transforms::Resize,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::Resize>>(transforms, "Resize",
        "Resize using nearest-neighbor interpolation")
        .def(py::init<int64_t, int64_t>(),
             py::arg("height"), py::arg("width"));

    py::class_<tenzor::data::transforms::RandomResizedCrop,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::RandomResizedCrop>>(transforms, "RandomResizedCrop",
        "Random crop then resize")
        .def(py::init<int64_t, int64_t, float, float, float, float>(),
             py::arg("height"), py::arg("width"),
             py::arg("scale_min") = 0.08f, py::arg("scale_max") = 1.0f,
             py::arg("ratio_min") = 0.75f, py::arg("ratio_max") = 1.333f);

    py::class_<tenzor::data::transforms::GaussianBlur,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::GaussianBlur>>(transforms, "GaussianBlur",
        "Gaussian blur with random sigma")
        .def(py::init<int, float, float>(),
             py::arg("kernel_size"), py::arg("sigma_min") = 0.1f, py::arg("sigma_max") = 2.0f);

    py::class_<tenzor::data::transforms::RandomAffine,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::RandomAffine>>(transforms, "RandomAffine",
        "Random affine transformation")
        .def(py::init<float, float, float, float, float, float>(),
             py::arg("degrees"),
             py::arg("translate_x") = 0.0f, py::arg("translate_y") = 0.0f,
             py::arg("scale_min") = 1.0f, py::arg("scale_max") = 1.0f,
             py::arg("shear") = 0.0f);

    py::class_<tenzor::data::transforms::RandomRotation,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::RandomRotation>>(transforms, "RandomRotation",
        "Random rotation by angle in [min_degrees, max_degrees]")
        .def(py::init<float, float>(),
             py::arg("min_degrees"), py::arg("max_degrees"));

    py::class_<tenzor::data::transforms::ColorJitter,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::ColorJitter>>(transforms, "ColorJitter",
        "Random brightness, contrast, saturation, and hue adjustments")
        .def(py::init<float, float, float, float>(),
             py::arg("brightness") = 0.0f, py::arg("contrast") = 0.0f,
             py::arg("saturation") = 0.0f, py::arg("hue") = 0.0f);

    py::class_<tenzor::data::transforms::Cutout,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::Cutout>>(transforms, "Cutout",
        "Randomly mask rectangular regions with zeros")
        .def(py::init<int, int>(),
             py::arg("num_holes"), py::arg("hole_size"));

    py::class_<tenzor::data::transforms::Lambda,
               tenzor::data::transforms::Transform,
               std::shared_ptr<tenzor::data::transforms::Lambda>>(transforms, "Lambda",
        "Custom transform from a callable")
        .def(py::init([](py::function func) {
            return std::make_unique<tenzor::data::transforms::Lambda>(
                [func](const tenzor::Tensor& input, const tenzor::Tensor& target)
                    -> std::pair<tenzor::Tensor, tenzor::Tensor> {
                    py::gil_scoped_acquire acquire;
                    auto result = func(input, target);
                    return result.cast<std::pair<tenzor::Tensor, tenzor::Tensor>>();
                });
        }), py::arg("func"));

    // =========================================================================
    // Standard Datasets
    // =========================================================================
    auto datasets = data_mod.def_submodule("datasets", "Standard dataset loaders");

    // MapDataset is the C++ base class of the concrete datasets below
    // (MNIST / CIFAR / ImageFolder / ...). pybind11 requires every base
    // class referenced by a derived py::class_ to have been registered
    // already, so we declare it here as a bare holder before the
    // derived bindings. Previously this was omitted and importing
    // tenzor threw `type "MNIST" referenced unknown base type
    // "tenzor::data::MapDataset"` at module load time.
    py::class_<tenzor::data::MapDataset, tenzor::data::Dataset,
               std::shared_ptr<tenzor::data::MapDataset>>(data_mod, "MapDataset");

    py::class_<tenzor::data::datasets::MNIST, tenzor::data::MapDataset,
               std::shared_ptr<tenzor::data::datasets::MNIST>>(datasets, "MNIST",
        "MNIST handwritten digit dataset (28x28 grayscale, 10 classes)")
        .def(py::init<std::string, bool, bool>(),
             py::arg("root"), py::arg("train") = true, py::arg("normalize") = true);

    py::class_<tenzor::data::datasets::FashionMNIST, tenzor::data::datasets::MNIST,
               std::shared_ptr<tenzor::data::datasets::FashionMNIST>>(datasets, "FashionMNIST",
        "Fashion-MNIST dataset (same format as MNIST)")
        .def(py::init<std::string, bool, bool>(),
             py::arg("root"), py::arg("train") = true, py::arg("normalize") = true);

    py::class_<tenzor::data::datasets::CIFAR10, tenzor::data::MapDataset,
               std::shared_ptr<tenzor::data::datasets::CIFAR10>>(datasets, "CIFAR10",
        "CIFAR-10 image classification (32x32 RGB, 10 classes)")
        .def(py::init<std::string, bool, bool>(),
             py::arg("root"), py::arg("train") = true, py::arg("normalize") = true);

    py::class_<tenzor::data::datasets::CIFAR100, tenzor::data::MapDataset,
               std::shared_ptr<tenzor::data::datasets::CIFAR100>>(datasets, "CIFAR100",
        "CIFAR-100 image classification (32x32 RGB, 100 classes)")
        .def(py::init<std::string, bool, bool>(),
             py::arg("root"), py::arg("train") = true, py::arg("normalize") = true);

    py::class_<tenzor::data::datasets::ImageFolder, tenzor::data::MapDataset,
               std::shared_ptr<tenzor::data::datasets::ImageFolder>>(datasets, "ImageFolder",
        "ImageNet-style folder dataset (root/class_name/image.jpg)")
        .def(py::init<std::string, int64_t, std::vector<std::string>>(),
             py::arg("root"), py::arg("image_size") = 224,
             py::arg("extensions") = std::vector<std::string>{".jpg", ".jpeg", ".png", ".bmp"})
        .def("class_names", &tenzor::data::datasets::ImageFolder::class_names)
        .def("num_classes", &tenzor::data::datasets::ImageFolder::num_classes);

    // =========================================================================
    // TensorBoard Integration
    // =========================================================================
    auto tensorboard = m.def_submodule("tensorboard", "TensorBoard logging");

    py::class_<tenzor::SummaryWriter>(tensorboard, "SummaryWriter",
        "Writer for TensorBoard event files")
        .def(py::init<std::string_view, int, int>(),
             py::arg("log_dir"),
             py::arg("max_queue") = 10,
             py::arg("flush_secs") = 120,
             "Create SummaryWriter for specified log directory")
        .def("add_scalar", &tenzor::SummaryWriter::add_scalar,
             py::arg("tag"), py::arg("value"), py::arg("step"),
             "Log scalar value")
        .def("add_histogram", &tenzor::SummaryWriter::add_histogram,
             py::arg("tag"), py::arg("tensor"), py::arg("step"),
             py::arg("bins") = 30,
             "Log histogram of tensor values")
        .def("add_image", &tenzor::SummaryWriter::add_image,
             py::arg("tag"), py::arg("tensor"), py::arg("step"),
             py::arg("dataformats") = "CHW",
             "Log image tensor")
        .def("add_graph", &tenzor::SummaryWriter::add_graph,
             py::arg("model_name"), py::arg("output"),
             "Log computation graph by walking the autograd grad_fn chain "
             "of the given output Variable")
        .def("flush", &tenzor::SummaryWriter::flush,
             "Flush all pending events to disk")
        .def("close", &tenzor::SummaryWriter::close,
             "Close writer and release resources")
        .def("is_open", &tenzor::SummaryWriter::is_open,
             "Check if writer is open");

    // =========================================================================
    // Benchmark Utilities
    // =========================================================================
    auto benchmark_m = m.def_submodule("benchmark", "Performance benchmarking");

    py::class_<tenzor::benchmark::Timer>(benchmark_m, "Timer",
        "High-resolution timer")
        .def(py::init<>())
        .def("start", &tenzor::benchmark::Timer::start)
        .def("stop", &tenzor::benchmark::Timer::stop)
        .def("elapsed", &tenzor::benchmark::Timer::elapsed);

    py::class_<tenzor::benchmark::BenchmarkStats>(benchmark_m, "BenchmarkStats",
        "Benchmark statistics")
        .def_readonly("mean", &tenzor::benchmark::BenchmarkStats::mean)
        .def_readonly("std_dev", &tenzor::benchmark::BenchmarkStats::std_dev)
        .def_readonly("min", &tenzor::benchmark::BenchmarkStats::min)
        .def_readonly("max", &tenzor::benchmark::BenchmarkStats::max)
        .def_readonly("median", &tenzor::benchmark::BenchmarkStats::median)
        .def_readonly("p95", &tenzor::benchmark::BenchmarkStats::p95)
        .def_readonly("p99", &tenzor::benchmark::BenchmarkStats::p99)
        .def_readonly("num_runs", &tenzor::benchmark::BenchmarkStats::num_runs)
        .def("ops_per_sec", &tenzor::benchmark::BenchmarkStats::ops_per_sec)
        .def("tflops", &tenzor::benchmark::BenchmarkStats::tflops)
        .def("gflops", &tenzor::benchmark::BenchmarkStats::gflops)
        .def("bandwidth_gbs", &tenzor::benchmark::BenchmarkStats::bandwidth_gbs);

    py::class_<tenzor::benchmark::BenchmarkResult>(benchmark_m, "BenchmarkResult",
        "Benchmark result")
        .def_readonly("name", &tenzor::benchmark::BenchmarkResult::name)
        .def_readonly("stats", &tenzor::benchmark::BenchmarkResult::stats)
        .def_readonly("num_flops", &tenzor::benchmark::BenchmarkResult::num_flops)
        .def_readonly("num_bytes", &tenzor::benchmark::BenchmarkResult::num_bytes)
        .def_readonly("tflops", &tenzor::benchmark::BenchmarkResult::tflops)
        .def_readonly("bandwidth_gbs", &tenzor::benchmark::BenchmarkResult::bandwidth_gbs)
        .def("print", &tenzor::benchmark::BenchmarkResult::print)
        .def("to_json", &tenzor::benchmark::BenchmarkResult::to_json);

    py::class_<tenzor::benchmark::Benchmark>(benchmark_m, "Benchmark",
        "Benchmark runner")
        .def(py::init<const std::string&, size_t, size_t>(),
             py::arg("name"),
             py::arg("num_warmup") = 5,
             py::arg("num_runs") = 100)
        .def("run", py::overload_cast<const std::function<void()>&>(
             &tenzor::benchmark::Benchmark::run),
             py::arg("fn"),
             "Run benchmark")
        .def("set_flops", &tenzor::benchmark::Benchmark::set_flops)
        .def("set_bytes", &tenzor::benchmark::Benchmark::set_bytes);

    py::class_<tenzor::benchmark::BenchmarkSuite>(benchmark_m, "BenchmarkSuite",
        "Benchmark suite for running multiple benchmarks")
        .def(py::init<const std::string&>(),
             py::arg("name"))
        .def("add", &tenzor::benchmark::BenchmarkSuite::add)
        .def("run_all", &tenzor::benchmark::BenchmarkSuite::run_all)
        .def("print_summary", &tenzor::benchmark::BenchmarkSuite::print_summary)
        .def("export_json", &tenzor::benchmark::BenchmarkSuite::export_json)
        .def("save_json", &tenzor::benchmark::BenchmarkSuite::save_json);

    // FLOPS calculation helpers
    auto flops_m = benchmark_m.def_submodule("flops", "FLOPS calculation utilities");
    flops_m.def("matmul", &tenzor::benchmark::flops::matmul,
                  py::arg("M"), py::arg("N"), py::arg("K"));
    flops_m.def("conv2d", &tenzor::benchmark::flops::conv2d,
                  py::arg("batch"), py::arg("out_h"), py::arg("out_w"),
                  py::arg("in_channels"), py::arg("out_channels"),
                  py::arg("kernel_h"), py::arg("kernel_w"));
    flops_m.def("elementwise", &tenzor::benchmark::flops::elementwise,
                  py::arg("num_elements"), py::arg("ops_per_element") = 1);

    // Memory calculation helpers
    auto memory_m = benchmark_m.def_submodule("memory", "Memory calculation utilities");
    memory_m.def("matmul", &tenzor::benchmark::memory::matmul,
                   py::arg("M"), py::arg("N"), py::arg("K"),
                   py::arg("element_size") = 4);
    memory_m.def("elementwise", &tenzor::benchmark::memory::elementwise,
                   py::arg("num_elements"),
                   py::arg("num_inputs") = 2,
                   py::arg("num_outputs") = 1,
                   py::arg("element_size") = 4);

    // =========================================================================
    // HRM (Hierarchical Reasoning Model) Layers
    // =========================================================================
    auto hrm = nn_mod.def_submodule("hrm", "Hierarchical Reasoning Model components");

    // HRM Configuration
    py::class_<tenzor::nn::HRMConfig>(hrm, "HRMConfig",
        "Configuration for Hierarchical Reasoning Model")
        .def(py::init<>())
        .def_readwrite("d_model", &tenzor::nn::HRMConfig::d_model)
        .def_readwrite("n_heads", &tenzor::nn::HRMConfig::n_heads)
        .def_readwrite("d_feedforward", &tenzor::nn::HRMConfig::d_feedforward)
        .def_readwrite("n_high_cycles", &tenzor::nn::HRMConfig::n_high_cycles)
        .def_readwrite("t_low_steps", &tenzor::nn::HRMConfig::t_low_steps)
        .def_readwrite("dropout", &tenzor::nn::HRMConfig::dropout)
        .def_readwrite("use_post_norm", &tenzor::nn::HRMConfig::use_post_norm)
        .def_readwrite("deep_supervision", &tenzor::nn::HRMConfig::deep_supervision)
        .def_readwrite("max_seq_len", &tenzor::nn::HRMConfig::max_seq_len)
        .def_readwrite("vocab_size", &tenzor::nn::HRMConfig::vocab_size)
        .def_readwrite("num_classes", &tenzor::nn::HRMConfig::num_classes)
        .def_readwrite("use_stablemax", &tenzor::nn::HRMConfig::use_stablemax)
        .def_readwrite("stablemax_eps", &tenzor::nn::HRMConfig::stablemax_eps)
        .def_readwrite("use_act", &tenzor::nn::HRMConfig::use_act)
        .def_readwrite("use_qlearning_act", &tenzor::nn::HRMConfig::use_qlearning_act)
        .def_readwrite("act_threshold", &tenzor::nn::HRMConfig::act_threshold)
        .def_readwrite("act_epsilon", &tenzor::nn::HRMConfig::act_epsilon)
        .def_readwrite("act_gamma", &tenzor::nn::HRMConfig::act_gamma)
        .def_readwrite("act_lr", &tenzor::nn::HRMConfig::act_lr)
        .def_readwrite("max_segments", &tenzor::nn::HRMConfig::max_segments)
        .def_readwrite("use_lecun_init", &tenzor::nn::HRMConfig::use_lecun_init)
        .def_readwrite("use_truncated_normal", &tenzor::nn::HRMConfig::use_truncated_normal)
        .def_readwrite("init_std", &tenzor::nn::HRMConfig::init_std);

    // RMSNorm is registered in nn module, no need to register again here

    // GateType enum
    py::enum_<tenzor::nn::GateType>(hrm, "GateType")
        .value("Sigmoid", tenzor::nn::GateType::Sigmoid)
        .value("SiLU", tenzor::nn::GateType::SiLU)
        .value("GELU", tenzor::nn::GateType::GELU)
        .value("ReLU", tenzor::nn::GateType::ReLU);

    // GatedLinearUnit (SwiGLU / GeGLU / ReGLU)
    py::class_<tenzor::nn::GatedLinearUnit, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::GatedLinearUnit>>(hrm, "GatedLinearUnit",
        "Gated Linear Unit with configurable gate activation (SwiGLU, GeGLU, ReGLU)")
        .def(py::init<int64_t, int64_t, tenzor::nn::GateType, bool>(),
             py::arg("in_features"), py::arg("hidden_features"),
             py::arg("gate_type") = tenzor::nn::GateType::SiLU,
             py::arg("bias") = false)
        .def_property_readonly("gate_type", &tenzor::nn::GatedLinearUnit::gate_type);

    // GeGLU convenience class
    py::class_<tenzor::nn::GeGLU, tenzor::nn::GatedLinearUnit,
               std::shared_ptr<tenzor::nn::GeGLU>>(hrm, "GeGLU",
        "GELU-gated linear unit (Gemma, etc.)")
        .def(py::init<int64_t, int64_t, bool>(),
             py::arg("in_features"), py::arg("hidden_features"),
             py::arg("bias") = false);

    // ReGLU convenience class
    py::class_<tenzor::nn::ReGLU, tenzor::nn::GatedLinearUnit,
               std::shared_ptr<tenzor::nn::ReGLU>>(hrm, "ReGLU",
        "ReLU-gated linear unit")
        .def(py::init<int64_t, int64_t, bool>(),
             py::arg("in_features"), py::arg("hidden_features"),
             py::arg("bias") = false);

    // MixtureOfExperts
    py::class_<tenzor::nn::MixtureOfExperts, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::MixtureOfExperts>>(hrm, "MixtureOfExperts",
        "Mixture of Experts with top-k routing and load balancing loss")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, double, double, double>(),
             py::arg("input_dim"), py::arg("hidden_dim"),
             py::arg("num_experts"), py::arg("top_k") = 2,
             py::arg("capacity_factor") = 1.25,
             py::arg("aux_loss_weight") = 0.01,
             py::arg("dropout") = 0.0)
        .def("forward_with_loss", &tenzor::nn::MixtureOfExperts::forward_with_loss,
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Forward pass returning (output, aux_loss)")
        .def_property_readonly("num_experts", &tenzor::nn::MixtureOfExperts::num_experts)
        .def_property_readonly("top_k", &tenzor::nn::MixtureOfExperts::top_k);

    // RotaryPositionEmbedding
    py::class_<tenzor::nn::RotaryPositionEmbedding, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::RotaryPositionEmbedding>>(hrm, "RotaryPositionEmbedding",
        "Rotary Position Embedding (RoPE)")
        .def(py::init<int64_t, int64_t, double>(),
             py::arg("dim"), py::arg("max_seq_len") = 2048,
             py::arg("base") = 10000.0);

    // HRMBlock
    py::class_<tenzor::nn::HRMBlock, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::HRMBlock>>(hrm, "HRMBlock",
        "HRM Transformer block with RoPE and optional cross-attention")
        .def(py::init<int64_t, int64_t, int64_t, double, bool, int64_t>(),
             py::arg("d_model"), py::arg("n_heads"), py::arg("d_feedforward"),
             py::arg("dropout") = 0.1, py::arg("use_post_norm") = true,
             py::arg("max_seq_len") = 512);

    // QLearningACT
    py::class_<tenzor::nn::QLearningACT, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::QLearningACT>>(hrm, "QLearningACT",
        "Q-Learning based Adaptive Computational Time")
        .def(py::init<int64_t, int64_t, double, double, double>(),
             py::arg("d_model"), py::arg("max_segments"),
             py::arg("epsilon") = 0.1, py::arg("gamma") = 0.99, py::arg("lr") = 0.01)
        .def("decay_epsilon", &tenzor::nn::QLearningACT::decay_epsilon,
             py::arg("decay_rate") = 0.995, py::arg("min_epsilon") = 0.01);

    // HRM Model
    py::class_<tenzor::nn::HRM, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::HRM>>(hrm, "HRM",
        "Hierarchical Reasoning Model - brain-inspired recurrent architecture")
        .def(py::init<const tenzor::nn::HRMConfig&>(),
             py::arg("config"))
        .def("forward_with_aux", &tenzor::nn::HRM::forward_with_aux,
             py::arg("input"), py::arg("mask") = tenzor::Tensor{},
             "Forward with auxiliary outputs for deep supervision")
        .def("forward_with_segments", &tenzor::nn::HRM::forward_with_segments,
             py::arg("input"), py::arg("targets") = tenzor::Variable{},
             py::arg("mask") = tenzor::Tensor{},
             "Forward with segment-based training")
        .def("config", &tenzor::nn::HRM::config,
             py::return_value_policy::reference)
        .def("num_parameters", &tenzor::nn::HRM::num_parameters)
        .def("apply_hrm_initialization", &tenzor::nn::HRM::apply_hrm_initialization)
        .def("get_qlearning_act", &tenzor::nn::HRM::get_qlearning_act);

    // Stablemax functions
    hrm.def("stablemax", &tenzor::nn::stablemax,
            py::arg("input"), py::arg("dim") = -1, py::arg("eps") = 1e-12,
            "Numerically stable softmax variant");

    hrm.def("stablemax_cross_entropy", &tenzor::nn::stablemax_cross_entropy,
            py::arg("input"), py::arg("target"), py::arg("eps") = 1e-12,
            "Cross-entropy loss with stablemax");

    hrm.def("hrm_deep_supervision_loss", &tenzor::nn::hrm_deep_supervision_loss,
            py::arg("outputs"), py::arg("targets"), py::arg("loss_fn"),
            py::arg("weight_decay") = 0.5,
            "Deep supervision loss for HRM training");

    // Initialization utilities
    hrm.def("lecun_normal_init", &tenzor::nn::lecun_normal_init,
            py::arg("tensor"), py::arg("fan_in"),
            "LeCun normal initialization");

    hrm.def("lecun_uniform_init", &tenzor::nn::lecun_uniform_init,
            py::arg("tensor"), py::arg("fan_in"),
            "LeCun uniform initialization");

    hrm.def("truncated_normal_init", &tenzor::nn::truncated_normal_init,
            py::arg("tensor"), py::arg("mean") = 0.0, py::arg("std") = 1.0,
            py::arg("a") = -2.0, py::arg("b") = 2.0,
            "Truncated normal initialization");

    // =========================================================================
    // Model Architectures
    // =========================================================================

    // ResNet
    py::class_<tenzor::models::BasicBlock, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::BasicBlock>>(models, "BasicBlock",
        "ResNet BasicBlock (for ResNet-18/34)")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, std::shared_ptr<tenzor::nn::Module>>(),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("stride") = 1, py::arg("groups") = 1, py::arg("base_width") = 64,
             py::arg("downsample") = nullptr);

    py::class_<tenzor::models::Bottleneck, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::Bottleneck>>(models, "Bottleneck",
        "ResNet Bottleneck (for ResNet-50/101/152)")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, std::shared_ptr<tenzor::nn::Module>>(),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("stride") = 1, py::arg("groups") = 1, py::arg("base_width") = 64,
             py::arg("downsample") = nullptr);

    py::class_<tenzor::models::ResNet, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::ResNet>>(models, "ResNet",
        "ResNet architecture")
        .def(py::init<std::vector<int64_t>, int64_t, bool, int64_t, int64_t>(),
             py::arg("layers"), py::arg("num_classes") = 1000,
             py::arg("use_basic_block") = true,
             py::arg("groups") = 1, py::arg("base_width") = 64);

    // Convenience functions for standard ResNet variants. pretrained=True
    // downloads ImageNet weights via ModelHub.
    models.def("resnet18",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::resnet18(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "ResNet-18. pretrained=True downloads ImageNet weights via tz.models.hub.");
    models.def("resnet34",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::resnet34(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "ResNet-34.");
    models.def("resnet50",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::resnet50(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "ResNet-50.");
    models.def("resnet101",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::resnet101(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "ResNet-101.");
    models.def("resnet152",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::resnet152(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "ResNet-152.");
    models.def("resnext50_32x4d",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::resnext50_32x4d(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "ResNeXt-50 32x4d.");
    models.def("resnext101_32x8d",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::resnext101_32x8d(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "ResNeXt-101 32x8d.");

    // VGG
    py::class_<tenzor::models::VGG, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::VGG>>(models, "VGG",
        "VGG architecture")
        .def(py::init<tenzor::models::VGGConfig, int64_t, bool, double, bool>(),
             py::arg("config"), py::arg("num_classes") = 1000,
             py::arg("batch_norm") = true, py::arg("dropout") = 0.5,
             py::arg("init_weights") = true);

    py::class_<tenzor::models::VGGConfig>(models, "VGGConfig", "VGG configuration")
        .def(py::init<>())
        .def_readwrite("layers", &tenzor::models::VGGConfig::layers);

    models.def("vgg11",
        [](int64_t num_classes, bool bn, bool pretrained) {
            return tenzor::models::vgg11(num_classes, bn, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("batch_norm") = true, py::arg("pretrained") = false,
        "VGG-11. pretrained=True downloads ImageNet weights via tz.models.hub.");
    models.def("vgg13",
        [](int64_t num_classes, bool bn, bool pretrained) {
            return tenzor::models::vgg13(num_classes, bn, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("batch_norm") = true, py::arg("pretrained") = false,
        "VGG-13.");
    models.def("vgg16",
        [](int64_t num_classes, bool bn, bool pretrained) {
            return tenzor::models::vgg16(num_classes, bn, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("batch_norm") = true, py::arg("pretrained") = false,
        "VGG-16.");
    models.def("vgg19",
        [](int64_t num_classes, bool bn, bool pretrained) {
            return tenzor::models::vgg19(num_classes, bn, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("batch_norm") = true, py::arg("pretrained") = false,
        "VGG-19.");

    // AlexNet
    py::class_<tenzor::models::AlexNet, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::AlexNet>>(models, "AlexNet",
        "AlexNet architecture")
        .def(py::init<int64_t, double>(),
             py::arg("num_classes") = 1000, py::arg("dropout") = 0.5);
    models.def("alexnet",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::alexnet(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "AlexNet. pretrained=True downloads ImageNet weights via tz.models.hub.");

    // MobileNet
    py::class_<tenzor::models::MobileNetV2, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::MobileNetV2>>(models, "MobileNetV2",
        "MobileNetV2 architecture")
        .def(py::init<int64_t, double, double>(),
             py::arg("num_classes") = 1000, py::arg("width_mult") = 1.0,
             py::arg("dropout") = 0.2);
    models.def("mobilenet_v2",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::mobilenet_v2(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "MobileNetV2. pretrained=True downloads ImageNet weights via tz.models.hub.");

    py::class_<tenzor::models::MobileNetV3, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::MobileNetV3>>(models, "MobileNetV3",
        "MobileNetV3 architecture")
        .def(py::init<int64_t, std::string, double, double>(),
             py::arg("num_classes") = 1000, py::arg("mode") = "large",
             py::arg("width_mult") = 1.0, py::arg("dropout") = 0.2);
    models.def("mobilenet_v3_large",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::mobilenet_v3_large(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "MobileNetV3-Large.");
    models.def("mobilenet_v3_small",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::mobilenet_v3_small(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "MobileNetV3-Small.");

    // EfficientNet
    py::class_<tenzor::models::EfficientNet, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::EfficientNet>>(models, "EfficientNet",
        "EfficientNet architecture")
        .def(py::init<tenzor::models::EfficientNetConfig>(),
             py::arg("config"));

    py::class_<tenzor::models::EfficientNetConfig>(models, "EfficientNetConfig",
        "EfficientNet configuration")
        .def(py::init<>())
        .def_readwrite("width_mult", &tenzor::models::EfficientNetConfig::width_mult)
        .def_readwrite("depth_mult", &tenzor::models::EfficientNetConfig::depth_mult)
        .def_readwrite("num_classes", &tenzor::models::EfficientNetConfig::num_classes)
        .def_readwrite("dropout_rate", &tenzor::models::EfficientNetConfig::dropout_rate);

    models.def("efficientnet_b0",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::efficientnet_b0(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "EfficientNet. pretrained=True downloads ImageNet weights via tz.models.hub.");
    models.def("efficientnet_b1",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::efficientnet_b1(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "EfficientNet. pretrained=True downloads ImageNet weights via tz.models.hub.");
    models.def("efficientnet_b2",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::efficientnet_b2(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "EfficientNet. pretrained=True downloads ImageNet weights via tz.models.hub.");
    models.def("efficientnet_b3",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::efficientnet_b3(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "EfficientNet. pretrained=True downloads ImageNet weights via tz.models.hub.");
    models.def("efficientnet_b4",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::efficientnet_b4(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "EfficientNet. pretrained=True downloads ImageNet weights via tz.models.hub.");
    models.def("efficientnet_b5",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::efficientnet_b5(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "EfficientNet. pretrained=True downloads ImageNet weights via tz.models.hub.");
    models.def("efficientnet_b6",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::efficientnet_b6(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "EfficientNet. pretrained=True downloads ImageNet weights via tz.models.hub.");
    models.def("efficientnet_b7",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::efficientnet_b7(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "EfficientNet. pretrained=True downloads ImageNet weights via tz.models.hub.");

    // GoogLeNet
    py::class_<tenzor::models::GoogLeNet, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::GoogLeNet>>(models, "GoogLeNet",
        "GoogLeNet/Inception architecture")
        .def(py::init<int64_t, bool, double, bool>(),
             py::arg("num_classes") = 1000, py::arg("aux_logits") = true,
             py::arg("dropout") = 0.4, py::arg("init_weights") = true);
    models.def("googlenet",
        [](int64_t num_classes, bool pretrained, bool aux_logits) {
            return tenzor::models::googlenet(num_classes, pretrained, aux_logits);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false, py::arg("aux_logits") = true,
        "GoogLeNet. pretrained=True downloads ImageNet weights via tz.models.hub.");

    // ConvNeXt
    py::class_<tenzor::models::ConvNeXt, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::ConvNeXt>>(models, "ConvNeXt",
        "ConvNeXt architecture")
        .def(py::init<int64_t, int64_t, std::vector<int64_t>, std::vector<int64_t>, double, double>(),
             py::arg("in_channels") = 3, py::arg("num_classes") = 1000,
             py::arg("depths") = std::vector<int64_t>{3,3,9,3},
             py::arg("dims") = std::vector<int64_t>{96,192,384,768},
             py::arg("drop_path_rate") = 0.0, py::arg("layer_scale_init_value") = 1e-6);
    models.def("convnext_tiny",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::convnext_tiny(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "ConvNeXt. pretrained=True downloads ImageNet weights via tz.models.hub.");
    models.def("convnext_small",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::convnext_small(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "ConvNeXt. pretrained=True downloads ImageNet weights via tz.models.hub.");
    models.def("convnext_base",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::convnext_base(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "ConvNeXt. pretrained=True downloads ImageNet weights via tz.models.hub.");
    models.def("convnext_large",
        [](int64_t num_classes, bool pretrained) {
            return tenzor::models::convnext_large(num_classes, pretrained);
        },
        py::arg("num_classes") = 1000, py::arg("pretrained") = false,
        "ConvNeXt. pretrained=True downloads ImageNet weights via tz.models.hub.");

    // Vision Transformer (ViT)
    py::class_<tenzor::models::ViTConfig>(models, "ViTConfig", "ViT configuration")
        .def(py::init<>())
        .def_readwrite("hidden_size", &tenzor::models::ViTConfig::hidden_size)
        .def_readwrite("num_hidden_layers", &tenzor::models::ViTConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &tenzor::models::ViTConfig::num_attention_heads)
        .def_readwrite("image_size", &tenzor::models::ViTConfig::image_size)
        .def_readwrite("patch_size", &tenzor::models::ViTConfig::patch_size)
        .def_readwrite("num_channels", &tenzor::models::ViTConfig::num_channels);

    py::class_<tenzor::models::ViT, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::ViT>>(models, "ViT",
        "Vision Transformer")
        .def(py::init<tenzor::models::ViTConfig, bool>(),
             py::arg("config"), py::arg("add_pooling_layer") = true);

    py::class_<tenzor::models::ViTForImageClassification, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::ViTForImageClassification>>(models, "ViTForImageClassification",
        "ViT for image classification")
        .def(py::init<tenzor::models::ViTConfig, int64_t>(),
             py::arg("config"), py::arg("num_labels"));

    models.def("vit_base_patch16", []() {
        return tenzor::models::ViT_Base_Patch16();
    }, "Create ViT-Base/16");
    models.def("vit_large_patch16", []() {
        return tenzor::models::ViT_Large_Patch16();
    }, "Create ViT-Large/16");

    // Swin Transformer
    py::class_<tenzor::models::SwinTransformer, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::SwinTransformer>>(models, "SwinTransformer",
        "Swin Transformer architecture")
        .def(py::init([](int64_t img_size, int64_t patch_size, int64_t in_chans,
                         int64_t num_classes, int64_t embed_dim,
                         std::vector<int64_t> depths, std::vector<int64_t> num_heads,
                         int64_t window_size, double mlp_ratio, bool qkv_bias,
                         double qk_scale, double drop_rate, double attn_drop_rate,
                         double drop_path_rate, bool norm_layer, bool use_checkpoint) {
            return std::make_unique<tenzor::models::SwinTransformer>(
                img_size, patch_size, in_chans, num_classes, embed_dim,
                depths, num_heads, window_size, mlp_ratio, qkv_bias,
                qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                norm_layer, use_checkpoint);
        }),
             py::arg("img_size") = 224, py::arg("patch_size") = 4,
             py::arg("in_chans") = 3, py::arg("num_classes") = 1000,
             py::arg("embed_dim") = 96,
             py::arg("depths") = std::vector<int64_t>{2,2,6,2},
             py::arg("num_heads") = std::vector<int64_t>{3,6,12,24},
             py::arg("window_size") = 7, py::arg("mlp_ratio") = 4.0,
             py::arg("qkv_bias") = true, py::arg("qk_scale") = 0.0,
             py::arg("drop_rate") = 0.0, py::arg("attn_drop_rate") = 0.0,
             py::arg("drop_path_rate") = 0.1, py::arg("norm_layer") = true,
             py::arg("use_checkpoint") = false);
    models.def("swin_tiny",
        [](int64_t num_classes, int64_t img_size, bool pretrained, bool use_checkpoint) {
            return tenzor::models::swin_tiny(num_classes, img_size, pretrained, use_checkpoint);
        },
        py::arg("num_classes") = 1000, py::arg("img_size") = 224,
        py::arg("pretrained") = false, py::arg("use_checkpoint") = false,
        "Swin Transformer Tiny.");
    models.def("swin_small",
        [](int64_t num_classes, int64_t img_size, bool pretrained, bool use_checkpoint) {
            return tenzor::models::swin_small(num_classes, img_size, pretrained, use_checkpoint);
        },
        py::arg("num_classes") = 1000, py::arg("img_size") = 224,
        py::arg("pretrained") = false, py::arg("use_checkpoint") = false,
        "Swin Transformer Small.");
    models.def("swin_base",
        [](int64_t num_classes, int64_t img_size, bool pretrained, bool use_checkpoint) {
            return tenzor::models::swin_base(num_classes, img_size, pretrained, use_checkpoint);
        },
        py::arg("num_classes") = 1000, py::arg("img_size") = 224,
        py::arg("pretrained") = false, py::arg("use_checkpoint") = false,
        "Swin Transformer Base.");

    // BERT
    py::class_<tenzor::models::BertConfig>(models, "BertConfig", "BERT configuration")
        .def(py::init<>())
        .def_readwrite("hidden_size", &tenzor::models::BertConfig::hidden_size)
        .def_readwrite("num_hidden_layers", &tenzor::models::BertConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &tenzor::models::BertConfig::num_attention_heads)
        .def_readwrite("vocab_size", &tenzor::models::BertConfig::vocab_size)
        .def_readwrite("intermediate_size", &tenzor::models::BertConfig::intermediate_size)
        .def_readwrite("hidden_dropout_prob", &tenzor::models::BertConfig::hidden_dropout_prob)
        .def_readwrite("attention_probs_dropout_prob", &tenzor::models::BertConfig::attention_probs_dropout_prob)
        .def_readwrite("max_position_embeddings", &tenzor::models::BertConfig::max_position_embeddings)
        .def_readwrite("type_vocab_size", &tenzor::models::BertConfig::type_vocab_size)
        .def_readwrite("layer_norm_eps", &tenzor::models::BertConfig::layer_norm_eps)
        .def_readwrite("hidden_act", &tenzor::models::BertConfig::hidden_act)
        .def_static("base", &tenzor::models::BertConfig::base, "BERT-Base config")
        .def_static("large", &tenzor::models::BertConfig::large, "BERT-Large config");

    py::class_<tenzor::models::BertModel, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::BertModel>>(models, "BertModel",
        "BERT base model")
        .def(py::init<tenzor::models::BertConfig, bool>(),
             py::arg("config"), py::arg("add_pooling_layer") = true);

    py::class_<tenzor::models::BertForSequenceClassification, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::BertForSequenceClassification>>(models, "BertForSequenceClassification",
        "BERT for sequence classification")
        .def(py::init<tenzor::models::BertConfig, int64_t>(),
             py::arg("config"), py::arg("num_labels"));

    // GPT
    py::class_<tenzor::models::GPT2Config>(models, "GPT2Config", "GPT-2 configuration")
        .def(py::init<>())
        .def_readwrite("vocab_size", &tenzor::models::GPT2Config::vocab_size)
        .def_readwrite("n_positions", &tenzor::models::GPT2Config::n_positions)
        .def_readwrite("n_embd", &tenzor::models::GPT2Config::n_embd)
        .def_readwrite("n_layer", &tenzor::models::GPT2Config::n_layer)
        .def_readwrite("n_head", &tenzor::models::GPT2Config::n_head)
        .def_static("gpt2_small", &tenzor::models::GPT2Config::gpt2_small, "GPT-2 Small config")
        .def_static("gpt2_medium", &tenzor::models::GPT2Config::gpt2_medium, "GPT-2 Medium config")
        .def_static("gpt2_large", &tenzor::models::GPT2Config::gpt2_large, "GPT-2 Large config")
        .def_static("gpt2_xl", &tenzor::models::GPT2Config::gpt2_xl, "GPT-2 XL config");

    py::class_<tenzor::models::GPT2Model, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::GPT2Model>>(models, "GPT2Model",
        "GPT-2 base model")
        .def(py::init<tenzor::models::GPT2Config>(),
             py::arg("config"));

    py::class_<tenzor::models::GPT2LMHeadModel, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::GPT2LMHeadModel>>(models, "GPT2LMHeadModel",
        "GPT-2 language model with LM head")
        .def(py::init<tenzor::models::GPT2Config>(),
             py::arg("config"));

    // T5
    py::class_<tenzor::models::T5Config>(models, "T5Config", "T5 configuration")
        .def(py::init<>())
        .def_readwrite("d_model", &tenzor::models::T5Config::d_model)
        .def_readwrite("d_ff", &tenzor::models::T5Config::d_ff)
        .def_readwrite("num_layers", &tenzor::models::T5Config::num_layers)
        .def_readwrite("num_heads", &tenzor::models::T5Config::num_heads)
        .def_readwrite("vocab_size", &tenzor::models::T5Config::vocab_size)
        .def_static("small", &tenzor::models::T5Config::small, "T5-Small config")
        .def_static("base", &tenzor::models::T5Config::base, "T5-Base config")
        .def_static("large", &tenzor::models::T5Config::large, "T5-Large config");

    py::class_<tenzor::models::T5Model, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::T5Model>>(models, "T5Model",
        "T5 base model")
        .def(py::init<tenzor::models::T5Config>(),
             py::arg("config"));

    py::class_<tenzor::models::T5ForConditionalGeneration, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::T5ForConditionalGeneration>>(models, "T5ForConditionalGeneration",
        "T5 for conditional generation")
        .def(py::init<tenzor::models::T5Config>(),
             py::arg("config"));

    // ALBERT
    py::class_<tenzor::models::AlbertConfig>(models, "AlbertConfig", "ALBERT configuration")
        .def(py::init<>())
        .def_readwrite("hidden_size", &tenzor::models::AlbertConfig::hidden_size)
        .def_readwrite("embedding_size", &tenzor::models::AlbertConfig::embedding_size)
        .def_readwrite("num_hidden_layers", &tenzor::models::AlbertConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &tenzor::models::AlbertConfig::num_attention_heads)
        .def_readwrite("vocab_size", &tenzor::models::AlbertConfig::vocab_size)
        .def_static("base", &tenzor::models::AlbertConfig::base, "ALBERT-Base config")
        .def_static("large", &tenzor::models::AlbertConfig::large, "ALBERT-Large config");

    py::class_<tenzor::models::AlbertModel, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::AlbertModel>>(models, "AlbertModel",
        "ALBERT base model")
        .def(py::init<tenzor::models::AlbertConfig>(),
             py::arg("config"));

    py::class_<tenzor::models::AlbertForSequenceClassification, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::AlbertForSequenceClassification>>(models, "AlbertForSequenceClassification",
        "ALBERT for sequence classification")
        .def(py::init<tenzor::models::AlbertConfig, int64_t>(),
             py::arg("config"), py::arg("num_labels"));

    // RoBERTa
    py::class_<tenzor::models::RobertaConfig>(models, "RobertaConfig", "RoBERTa configuration")
        .def(py::init<>())
        .def_readwrite("hidden_size", &tenzor::models::RobertaConfig::hidden_size)
        .def_readwrite("num_hidden_layers", &tenzor::models::RobertaConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &tenzor::models::RobertaConfig::num_attention_heads)
        .def_readwrite("vocab_size", &tenzor::models::RobertaConfig::vocab_size)
        .def_static("base", &tenzor::models::RobertaConfig::base, "RoBERTa-Base config")
        .def_static("large", &tenzor::models::RobertaConfig::large, "RoBERTa-Large config");

    // ELECTRA
    py::class_<tenzor::models::ElectraConfig>(models, "ElectraConfig", "ELECTRA configuration")
        .def(py::init<>())
        .def_readwrite("hidden_size", &tenzor::models::ElectraConfig::hidden_size)
        .def_readwrite("num_hidden_layers", &tenzor::models::ElectraConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &tenzor::models::ElectraConfig::num_attention_heads)
        .def_readwrite("vocab_size", &tenzor::models::ElectraConfig::vocab_size)
        .def_static("small", &tenzor::models::ElectraConfig::small, "ELECTRA-Small config")
        .def_static("base", &tenzor::models::ElectraConfig::base, "ELECTRA-Base config")
        .def_static("large", &tenzor::models::ElectraConfig::large, "ELECTRA-Large config");

    // U-Net
    py::class_<tenzor::models::UNet, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::UNet>>(models, "UNet",
        "U-Net segmentation architecture")
        .def(py::init<int64_t, int64_t, bool>(),
             py::arg("in_channels"), py::arg("num_classes"),
             py::arg("bilinear") = false);

    // DeepLab v3+
    py::class_<tenzor::models::DeepLabV3PlusEncoder, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::DeepLabV3PlusEncoder>>(models, "DeepLabV3PlusEncoder",
        "DeepLab v3+ encoder")
        .def(py::init<std::string, int64_t, bool>(),
             py::arg("backbone_name") = "resnet50",
             py::arg("output_stride") = 16,
             py::arg("pretrained") = false);

    // YOLO
    py::class_<tenzor::models::Darknet53, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::Darknet53>>(models, "Darknet53",
        "Darknet53 backbone for YOLO")
        .def(py::init<int64_t>(),
             py::arg("in_channels") = 3);

    py::class_<tenzor::models::YOLOv3Head, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::YOLOv3Head>>(models, "YOLOv3Head",
        "YOLOv3 detection head")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("in_channels"), py::arg("num_classes"),
             py::arg("num_anchors") = 3);

    // Faster R-CNN
    py::class_<tenzor::models::FasterRCNN, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::FasterRCNN>>(models, "FasterRCNN",
        "Faster R-CNN object detection")
        .def(py::init<std::shared_ptr<tenzor::nn::Module>, int64_t>(),
             py::arg("backbone"), py::arg("num_classes"));

    // Mask R-CNN
    py::class_<tenzor::models::RPN, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::RPN>>(models, "RPN",
        "Region Proposal Network")
        .def(py::init<int64_t, int64_t>(),
             py::arg("in_channels"), py::arg("num_anchors"));

    py::class_<tenzor::models::ROIHead, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::ROIHead>>(models, "ROIHead",
        "ROI Head for instance segmentation")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("in_channels"), py::arg("num_classes"),
             py::arg("roi_size") = 7);

    py::class_<tenzor::models::MaskRCNN, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::MaskRCNN>>(models, "MaskRCNN",
        "Mask R-CNN instance segmentation")
        .def(py::init<std::shared_ptr<tenzor::nn::Module>, int64_t>(),
             py::arg("backbone"), py::arg("num_classes"));
    models.def("mask_rcnn_resnet50_fpn", &tenzor::models::mask_rcnn_resnet50_fpn,
        py::arg("num_classes") = 80, py::arg("pretrained") = false,
        "Create Mask R-CNN with ResNet-50 FPN backbone",
        py::call_guard<py::gil_scoped_release>());
    models.def("mask_rcnn_resnet101_fpn", &tenzor::models::mask_rcnn_resnet101_fpn,
        py::arg("num_classes") = 80, py::arg("pretrained") = false,
        "Create Mask R-CNN with ResNet-101 FPN backbone",
        py::call_guard<py::gil_scoped_release>());

    // ========== Model Compression (Pruning + Quantization) ==========
    bind_compression(m);
}

/**
 * @file compression_bindings.cpp
 * @brief Python bindings for model compression (pruning + quantization)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <tenzor/nn/compression/pruning.hpp>
#include <tenzor/nn/quantization.hpp>

namespace py = pybind11;
using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::compression;
using namespace tenzor::nn::quantization;

void bind_compression(py::module& m) {
    // Create compression submodule
    auto compression = m.def_submodule("compression", "Model compression utilities");

    // =============================================================================
    // Pruning Enums and Structures
    // =============================================================================

    py::enum_<ImportanceCriterion>(compression, "ImportanceCriterion")
        .value("L1", ImportanceCriterion::L1, "L1 norm (sum of absolute values)")
        .value("L2", ImportanceCriterion::L2, "L2 norm (Euclidean distance)")
        .value("L1Norm", ImportanceCriterion::L1Norm, "L1 norm normalized by parameter count")
        .value("L2Norm", ImportanceCriterion::L2Norm, "L2 norm normalized by parameter count");

    py::enum_<PruningSchedule>(compression, "PruningSchedule")
        .value("OneShot", PruningSchedule::OneShot, "Single pruning step")
        .value("Iterative", PruningSchedule::Iterative, "Linear sparsity increase")
        .value("Polynomial", PruningSchedule::Polynomial, "Polynomial sparsity schedule");

    py::class_<PruningMask>(compression, "PruningMask")
        .def(py::init<>())
        .def_readwrite("mask", &PruningMask::mask)
        .def_readwrite("layer_name", &PruningMask::layer_name)
        .def_readwrite("current_sparsity", &PruningMask::current_sparsity)
        .def("apply", &PruningMask::apply, "Apply mask to weights")
        .def("compute_sparsity", &PruningMask::compute_sparsity,
             "Compute actual sparsity of mask");

    py::class_<PruningConfig>(compression, "PruningConfig")
        .def(py::init<>())
        .def_readwrite("target_sparsity", &PruningConfig::target_sparsity)
        .def_readwrite("current_sparsity", &PruningConfig::current_sparsity)
        .def_readwrite("criterion", &PruningConfig::criterion)
        .def_readwrite("schedule", &PruningConfig::schedule)
        .def_readwrite("num_iterations", &PruningConfig::num_iterations)
        .def_readwrite("current_iteration", &PruningConfig::current_iteration)
        .def_readwrite("masks", &PruningConfig::masks)
        .def("get_current_sparsity", &PruningConfig::get_current_sparsity,
             "Get current sparsity based on schedule");

    // =============================================================================
    // Pruning Functions
    // =============================================================================

    compression.def("compute_importance", &compute_importance,
        py::arg("weights"), py::arg("criterion"),
        "Compute importance scores for tensor weights");

    compression.def("create_mask_from_importance", &create_mask_from_importance,
        py::arg("importance"), py::arg("sparsity"),
        "Create binary mask from importance scores");

    compression.def("prune_unstructured", &prune_unstructured,
        py::arg("module"), py::arg("sparsity"),
        py::arg("criterion") = ImportanceCriterion::L1,
        py::arg("global_pruning") = false,
        R"doc(
        Apply unstructured (fine-grained) pruning to module.

        Prunes individual weights based on magnitude, regardless of structure.

        Args:
            module: Neural network module to prune
            sparsity: Target sparsity level [0, 1] (0.5 = 50% zeros)
            criterion: Importance metric for weight selection
            global_pruning: If True, compute threshold globally across all layers

        Returns:
            PruningConfig with masks for all pruned layers

        Example:
            >>> model = MyModel()
            >>> config = prune_unstructured(model, 0.5, ImportanceCriterion.L1)
            >>> apply_pruning_masks(model, config)
        )doc");

    compression.def("prune_iterative", &prune_iterative,
        py::arg("module"), py::arg("target_sparsity"), py::arg("num_iterations"),
        py::arg("schedule") = PruningSchedule::Iterative,
        py::arg("criterion") = ImportanceCriterion::L1,
        R"doc(
        Iterative magnitude pruning with gradual sparsity increase.

        Args:
            module: Module to prune
            target_sparsity: Final target sparsity [0, 1]
            num_iterations: Number of pruning steps
            schedule: Sparsity increase schedule
            criterion: Importance metric

        Returns:
            Final pruning configuration

        Example:
            >>> config = prune_iterative(model, 0.9, 10, PruningSchedule.Polynomial)
        )doc");

    compression.def("prune_channels", &prune_channels,
        py::arg("module"), py::arg("sparsity"),
        py::arg("criterion") = ImportanceCriterion::L1,
        R"doc(
        Apply structured channel pruning to convolutional layers.

        Removes entire output channels from Conv2d layers based on importance.
        Creates regular sparsity that directly reduces computation.

        Args:
            module: Module containing Conv2d layers
            sparsity: Fraction of channels to prune [0, 1]
            criterion: Channel importance metric

        Returns:
            New module with channels physically removed
        )doc");

    compression.def("prune_filters", &prune_filters,
        py::arg("module"), py::arg("sparsity"),
        py::arg("criterion") = ImportanceCriterion::L1,
        "Prune entire filters from Conv2d layers");

    compression.def("prune_layers", &prune_layers,
        py::arg("module"), py::arg("num_layers"),
        py::arg("criterion") = ImportanceCriterion::L1,
        "Remove entire layers from sequential models");

    compression.def("apply_pruning_masks", &apply_pruning_masks,
        py::arg("module"), py::arg("config"),
        "Apply pruning masks to module parameters");

    compression.def("finalize_pruning", &finalize_pruning,
        py::arg("module"), py::arg("config"),
        "Make pruning permanent by removing zero weights");

    compression.def("remove_pruning", &remove_pruning,
        py::arg("module"), py::arg("config"),
        "Remove all pruning masks and restore dense weights");

    compression.def("compute_sparsity", &compute_sparsity,
        py::arg("module"),
        "Compute actual sparsity of module parameters");

    compression.def("analyze_layer_sparsity", &analyze_layer_sparsity,
        py::arg("module"),
        "Analyze per-layer sparsity");

    compression.def("compute_compression_ratio", &compute_compression_ratio,
        py::arg("original_module"), py::arg("pruned_module"),
        "Estimate compression ratio achieved by pruning");

    compression.def("estimate_flops_reduction", &estimate_flops_reduction,
        py::arg("module"), py::arg("input_shape"),
        "Estimate FLOPs reduction from structured pruning");

    compression.def("sensitivity_analysis", &sensitivity_analysis,
        py::arg("module"), py::arg("validation_fn"),
        py::arg("sparsity_levels") = std::vector<float>{0.1f, 0.3f, 0.5f, 0.7f, 0.9f},
        "Sensitivity analysis for layer-wise pruning");

    compression.def("find_lottery_ticket", &find_lottery_ticket,
        py::arg("module"), py::arg("initial_weights"),
        py::arg("target_sparsity"), py::arg("num_rounds"),
        "Find winning lottery ticket (iterative magnitude pruning)");

    // Quantization — see python/bindings/bindings_quantization.cpp
    tenzor::python::register_quantization(m);

    // Nested Tensor — see python/bindings/bindings_nested.cpp
    tenzor::python::register_nested(m);

    // Lite inference runtime — see python/bindings/bindings_lite.cpp
    tenzor::python::register_lite(m);

    // =============================================================================
    // Memory Management
    // =============================================================================

    m.def("empty_cache", []() {
        // Release CPU cache
        tenzor::cpu::CPUCachingAllocator::instance().release_cached_memory();
        // Note: Could also release GPU caches here if desired
    }, "Release all cached memory back to the system");

    m.def("memory_stats", [](const std::string& device) -> py::dict {
        py::dict d;
        if (device == "cpu" || device.empty()) {
            auto stats = tenzor::cpu::CPUCachingAllocator::instance().get_stats();
            d["allocated_bytes"] = stats.allocated_bytes;
            d["cached_bytes"] = stats.cached_bytes;
            d["peak_allocated_bytes"] = stats.peak_allocated_bytes;
            d["total_allocations"] = stats.total_allocations;
            d["cache_hits"] = stats.cache_hits;
            d["cache_hit_rate"] = stats.total_allocations > 0
                ? static_cast<double>(stats.cache_hits) / static_cast<double>(stats.total_allocations) : 0.0;
            d["num_splits"] = stats.num_splits;
            d["num_backend_allocs"] = stats.num_backend_allocs;
            d["num_backend_frees"] = stats.num_backend_frees;
        }
        return d;
    }, py::arg("device") = "cpu",
    "Get memory allocation statistics for the specified device");

    m.def("reset_memory_stats", [](const std::string& device) {
        if (device == "cpu" || device.empty()) {
            tenzor::cpu::CPUCachingAllocator::instance().reset_stats();
        }
    }, py::arg("device") = "cpu",
    "Reset memory statistics counters for the specified device");

    // -------------------------------------------------------------------------
    // Global Memory Profiler
    // -------------------------------------------------------------------------

    m.def("memory_profiler_stats", []() -> py::dict {
        auto s = tenzor::MemoryProfiler::instance().memory_stats();
        py::dict d;
        d["current_allocated_bytes"] = s.current_allocated_bytes;
        d["peak_allocated_bytes"]    = s.peak_allocated_bytes;
        d["total_allocated_bytes"]   = s.total_allocated_bytes;
        d["allocation_count"]        = s.allocation_count;
        d["deallocation_count"]      = s.deallocation_count;
        return d;
    }, "Get global memory profiler statistics (all devices)");

    m.def("reset_peak_memory_stats", []() {
        tenzor::MemoryProfiler::instance().reset_peak_memory_stats();
    }, "Reset peak memory counter to current allocated bytes");

    m.def("memory_profiler_summary", []() -> std::string {
        return tenzor::MemoryProfiler::instance().memory_summary();
    }, "Get human-readable global memory profiler summary");

    // =========================================================================
    // Operation Profiler
    // =========================================================================
    auto profiler = m.def_submodule("profiler", "Operation profiling for forward and backward passes");

    py::enum_<tenzor::ProfilePhase>(profiler, "Phase")
        .value("Forward", tenzor::ProfilePhase::Forward)
        .value("Backward", tenzor::ProfilePhase::Backward);

    // Thread-local guard storage for the forward profiling interceptor
    static thread_local std::unique_ptr<tenzor::ProfilingInterceptorGuard> s_fwd_guard;

    profiler.def("enable", []() {
        // Enable backward profiling (autograd)
        tenzor::AutogradProfiler::instance().enable();
        // Enable forward profiling (dispatch interceptor)
        tenzor::OpProfiler::instance().enable();
        if (!s_fwd_guard) {
            s_fwd_guard = std::make_unique<tenzor::ProfilingInterceptorGuard>();
        }
    }, "Enable profiling for both forward and backward passes");

    profiler.def("disable", []() {
        tenzor::AutogradProfiler::instance().disable();
        tenzor::OpProfiler::instance().disable();
        s_fwd_guard.reset();  // pop the interceptor
    }, "Disable profiling");

    profiler.def("is_enabled", []() {
        return tenzor::AutogradProfiler::instance().is_enabled()
            || tenzor::OpProfiler::instance().is_enabled();
    }, "Check if profiling is enabled");

    profiler.def("reset", []() {
        tenzor::AutogradProfiler::instance().reset();
        tenzor::OpProfiler::instance().reset();
    }, "Clear all recorded profiles");

    profiler.def("summary", []() {
        std::string out;
        // Forward pass summary from OpProfiler
        out += tenzor::OpProfiler::instance().summary();
        out += "\n";
        // Backward pass summary from AutogradProfiler
        out += tenzor::AutogradProfiler::instance().summary();
        return out;
    }, "Get human-readable profiling summary (forward + backward)");

    profiler.def("profiles", [](std::optional<tenzor::ProfilePhase> phase) -> py::list {
        py::list result;

        // Forward profiles from OpProfiler (dispatch-level)
        if (!phase.has_value() || phase.value() == tenzor::ProfilePhase::Forward) {
            auto fwd_profs = tenzor::OpProfiler::instance().profiles();
            for (const auto& p : fwd_profs) {
                py::dict d;
                d["name"] = "OpId:" + std::to_string(static_cast<int>(p.op));
                d["phase"] = "forward";
                d["total_ms"] = std::chrono::duration<double, std::milli>(p.total_time).count();
                d["call_count"] = p.call_count;
                d["avg_us"] = p.call_count > 0
                    ? std::chrono::duration<double, std::micro>(p.total_time).count() / p.call_count
                    : 0.0;
                result.append(d);
            }
        }

        // Backward profiles from AutogradProfiler
        if (!phase.has_value() || phase.value() == tenzor::ProfilePhase::Backward) {
            auto bwd_profs = tenzor::AutogradProfiler::instance().profiles();
            for (const auto& p : bwd_profs) {
                py::dict d;
                d["name"] = p.name;
                d["phase"] = p.phase == tenzor::ProfilePhase::Forward ? "forward" : "backward";
                d["total_ms"] = std::chrono::duration<double, std::milli>(p.total_time).count();
                d["call_count"] = p.call_count;
                d["avg_us"] = p.call_count > 0
                    ? std::chrono::duration<double, std::micro>(p.total_time).count() / p.call_count
                    : 0.0;
                result.append(d);
            }
        }

        return result;
    }, py::arg("phase") = py::none(),
    "Get profiling data as list of dicts. Optional phase filter.");

    profiler.def("enable_trace", []() {
        tenzor::AutogradProfiler::instance().enable_trace();
        // Also enable OpProfiler + interceptor so forward ops are captured
        tenzor::OpProfiler::instance().enable();
        if (!s_fwd_guard) {
            s_fwd_guard = std::make_unique<tenzor::ProfilingInterceptorGuard>();
        }
    }, "Enable trace mode (records per-invocation events for Chrome trace export)");

    profiler.def("export_chrome_trace", [](const std::string& path) {
        tenzor::AutogradProfiler::instance().export_chrome_trace(path);
    }, py::arg("path"),
    py::call_guard<py::gil_scoped_release>(),
    "Export recorded trace events to Chrome Trace Event Format JSON file");

    // =========================================================================
    // Custom Op Registration API
    // =========================================================================
    m.def("register_custom_op", [](const std::string& name,
                                    const std::string& device_str,
                                    py::function py_kernel) -> uint32_t {
        auto device_type = tenzor::Device::from_string(device_str).type;
        // Wrap Python callable in a CustomKernelFn
        auto kernel = [py_kernel](std::span<const tenzor::Tensor> inputs,
                                   const tenzor::OpAttributes& /*attrs*/) -> tenzor::Tensor {
            py::gil_scoped_acquire gil;
            py::list py_inputs;
            for (const auto& t : inputs) py_inputs.append(t);
            py::object result = py_kernel(py_inputs);
            return result.cast<tenzor::Tensor>();
        };
        auto id = tenzor::register_custom_op(name, device_type, std::move(kernel));
        return id.value;
    }, py::arg("name"), py::arg("device"), py::arg("kernel"),
    "Register a custom operation kernel.\n"
    "Returns an integer op ID for use with dispatch_custom_op().");

    m.def("dispatch_custom_op", [](uint32_t op_id,
                                    py::list py_inputs) -> tenzor::Tensor {
        std::vector<tenzor::Tensor> inputs;
        for (auto& item : py_inputs) {
            inputs.push_back(item.cast<tenzor::Tensor>());
        }
        return tenzor::dispatch_custom_op(tenzor::CustomOpId(op_id), inputs);
    }, py::arg("op_id"), py::arg("inputs"),
    "Dispatch a custom operation by its ID.");

    m.def("set_max_cached_bytes", [](size_t bytes, const std::string& device) {
        if (device == "cpu" || device.empty()) {
            tenzor::cpu::CPUCachingAllocator::instance().set_max_cached_bytes(bytes);
        }
    }, py::arg("bytes"), py::arg("device") = "cpu",
    "Set maximum cached memory for the specified device");

    // =========================================================================
    // Sparse submodule
    // =========================================================================
    auto sparse_mod = m.def_submodule("sparse", "Sparse tensor operations");

    py::enum_<tenzor::SparseLayout>(sparse_mod, "SparseLayout")
        .value("COO", tenzor::SparseLayout::COO)
        .value("CSR", tenzor::SparseLayout::CSR)
        .value("CSC", tenzor::SparseLayout::CSC)
        .value("BSR", tenzor::SparseLayout::BSR);

    py::class_<tenzor::SparseTensor>(sparse_mod, "SparseTensor")
        .def_property_readonly("layout", &tenzor::SparseTensor::layout)
        .def_property_readonly("shape", [](const tenzor::SparseTensor& s) {
            return s.shape();
        })
        .def_property_readonly("dtype", &tenzor::SparseTensor::dtype)
        .def_property_readonly("device", &tenzor::SparseTensor::device)
        .def_property_readonly("nnz", &tenzor::SparseTensor::nnz)
        .def_property_readonly("sparse_dim", &tenzor::SparseTensor::sparse_dim)
        .def_property_readonly("dense_dim", &tenzor::SparseTensor::dense_dim)
        .def_property_readonly("is_coalesced", &tenzor::SparseTensor::is_coalesced)
        .def("indices", &tenzor::SparseTensor::indices, "Get COO indices tensor")
        .def("values", &tenzor::SparseTensor::values, "Get values tensor")
        .def("crow_indices", &tenzor::SparseTensor::crow_indices, "Get CSR compressed row indices")
        .def("col_indices", &tenzor::SparseTensor::col_indices, "Get CSR column indices")
        .def("ccol_indices", &tenzor::SparseTensor::ccol_indices, "Get CSC compressed column indices")
        .def("row_indices", &tenzor::SparseTensor::row_indices, "Get CSC row indices")
        .def("bsr_row_ptr", &tenzor::SparseTensor::bsr_row_ptr, "Get BSR block row pointers")
        .def("bsr_col_ind", &tenzor::SparseTensor::bsr_col_ind, "Get BSR block column indices")
        .def("block_size", [](const tenzor::SparseTensor& s) {
            auto bs = s.block_size();
            return py::make_tuple(bs.first, bs.second);
        }, "Get BSR block dimensions as (block_h, block_w)")
        .def("to_dense", &tenzor::SparseTensor::to_dense, "Convert to dense tensor",
             py::call_guard<py::gil_scoped_release>())
        .def("to_coo", &tenzor::SparseTensor::to_coo, "Convert to COO format",
             py::call_guard<py::gil_scoped_release>())
        .def("to_csr", &tenzor::SparseTensor::to_csr, "Convert to CSR format",
             py::call_guard<py::gil_scoped_release>())
        .def("to_csc", &tenzor::SparseTensor::to_csc, "Convert to CSC format",
             py::call_guard<py::gil_scoped_release>())
        .def("to_bsr", &tenzor::SparseTensor::to_bsr, "Convert to BSR format",
             py::arg("block_size"),
             py::call_guard<py::gil_scoped_release>())
        .def("transpose", &tenzor::SparseTensor::transpose, "Transpose 2D sparse tensor",
             py::call_guard<py::gil_scoped_release>())
        .def("coalesce", &tenzor::SparseTensor::coalesce, "Coalesce COO tensor",
             py::call_guard<py::gil_scoped_release>())
        .def("to", &tenzor::SparseTensor::to, "Transfer to device",
             py::arg("device"),
             py::call_guard<py::gil_scoped_release>())
        .def("__repr__", [](const tenzor::SparseTensor& s) {
            std::ostringstream ss;
            const char* layout_name = "COO";
            switch (s.layout()) {
                case tenzor::SparseLayout::COO: layout_name = "COO"; break;
                case tenzor::SparseLayout::CSR: layout_name = "CSR"; break;
                case tenzor::SparseLayout::CSC: layout_name = "CSC"; break;
                case tenzor::SparseLayout::BSR: layout_name = "BSR"; break;
            }
            ss << "SparseTensor(layout=" << layout_name << ", shape=[";
            auto& shape = s.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << shape[i];
            }
            ss << "], nnz=" << s.nnz() << ", dtype=" << tenzor::dtype_name(s.dtype()) << ")";
            return ss.str();
        });

    sparse_mod.def("sparse_coo", &tenzor::SparseTensor::sparse_coo,
        "Create a COO sparse tensor",
        py::arg("indices"), py::arg("values"), py::arg("shape"));

    sparse_mod.def("sparse_csr", &tenzor::SparseTensor::sparse_csr,
        "Create a CSR sparse tensor",
        py::arg("crow_indices"), py::arg("col_indices"), py::arg("values"), py::arg("shape"));

    sparse_mod.def("to_sparse", &tenzor::to_sparse,
        "Convert dense tensor to sparse COO",
        py::arg("dense"),
        py::call_guard<py::gil_scoped_release>());

    sparse_mod.def("to_sparse_csr", &tenzor::to_sparse_csr,
        "Convert dense tensor to sparse CSR",
        py::arg("dense"),
        py::call_guard<py::gil_scoped_release>());

    sparse_mod.def("sparse_csc", &tenzor::SparseTensor::sparse_csc,
        "Create a CSC sparse tensor",
        py::arg("ccol_indices"), py::arg("row_indices"), py::arg("values"), py::arg("shape"));

    sparse_mod.def("sparse_bsr", &tenzor::SparseTensor::sparse_bsr,
        "Create a BSR sparse tensor",
        py::arg("bsr_row_ptr"), py::arg("bsr_col_ind"), py::arg("values"),
        py::arg("shape"), py::arg("block_size"));

    sparse_mod.def("to_sparse_csc", &tenzor::to_sparse_csc,
        "Convert dense tensor to sparse CSC",
        py::arg("dense"),
        py::call_guard<py::gil_scoped_release>());

    sparse_mod.def("spmm", &tenzor::sparse::spmm,
        "Sparse-dense matrix multiplication",
        py::arg("sparse"), py::arg("dense"),
        py::call_guard<py::gil_scoped_release>());

    sparse_mod.def("spmv", &tenzor::sparse::spmv,
        "Sparse-dense matrix-vector multiplication",
        py::arg("sparse"), py::arg("vec"),
        py::call_guard<py::gil_scoped_release>());

    sparse_mod.def("add", py::overload_cast<const tenzor::SparseTensor&, const tenzor::Tensor&>(
        &tenzor::sparse::add),
        "Sparse-dense addition",
        py::arg("sparse"), py::arg("dense"),
        py::call_guard<py::gil_scoped_release>());

    sparse_mod.def("sparse_add", py::overload_cast<const tenzor::SparseTensor&, const tenzor::SparseTensor&>(
        &tenzor::sparse::add),
        "Sparse-sparse addition",
        py::arg("a"), py::arg("b"),
        py::call_guard<py::gil_scoped_release>());

    sparse_mod.def("mul", &tenzor::sparse::mul,
        "Scalar multiplication of sparse tensor",
        py::arg("sparse"), py::arg("scalar"),
        py::call_guard<py::gil_scoped_release>());

    sparse_mod.def("spgemm", &tenzor::sparse::spgemm,
        "Sparse-sparse matrix multiplication (CSR x CSR -> CSR)",
        py::arg("a"), py::arg("b"),
        py::call_guard<py::gil_scoped_release>());

    sparse_mod.def("triangular_solve", &tenzor::sparse::sparse_triangular_solve,
        "Sparse triangular solve: L @ x = b (lower) or U @ x = b (upper)",
        py::arg("L"), py::arg("b"), py::arg("upper") = false,
        py::call_guard<py::gil_scoped_release>());

    // =========================================================================
    // Forward-mode AD and composable transforms
    // =========================================================================

    m.def("jvp", [](py::function py_func,
                     const tenzor::Variable& input,
                     const tenzor::Tensor& tangent,
                     const std::string& mode) {
        auto func = [&py_func](const tenzor::Variable& x) -> tenzor::Variable {
            py::gil_scoped_acquire gil;
            py::object result = py_func(x);
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
        tenzor::Variable output;
        tenzor::Tensor tangent_out;
        {
            py::gil_scoped_release release;
            auto pair = tenzor::jvp(func, input, tangent, jvp_mode);
            output = std::move(pair.first);
            tangent_out = std::move(pair.second);
        }
        return py::make_tuple(output, tangent_out);
    }, py::arg("func"), py::arg("input"), py::arg("tangent"),
    py::arg("mode") = std::string("walker"),
    "Compute Jacobian-Vector Product (forward-mode AD).\n"
    "Returns (output, tangent_output) where tangent_output = J @ tangent.\n"
    "mode='walker' (default) builds the autograd graph then walks it;\n"
    "mode='dual' additionally raises the is_dual_mode() TLS flag.");

    m.def("jacobian", [](py::function py_func,
                          const tenzor::Variable& input) {
        auto func = [&py_func](const tenzor::Variable& x) -> tenzor::Variable {
            py::gil_scoped_acquire gil;
            py::object result = py_func(x);
            return result.cast<tenzor::Variable>();
        };
        // R.22: outer GIL release; inner cpp_fn reacquires for the Python callback.
        tenzor::Tensor J;
        {
            py::gil_scoped_release release;
            J = tenzor::jacobian(func, input);
        }
        return J;
    }, py::arg("func"), py::arg("input"),
    "Compute full Jacobian matrix of func at input.\n"
    "Returns tensor of shape (output_size, input_size).");

    m.def("hessian", [](py::function py_func,
                         const tenzor::Variable& input) {
        auto func = [&py_func](const tenzor::Variable& x) -> tenzor::Variable {
            py::gil_scoped_acquire gil;
            py::object result = py_func(x);
            return result.cast<tenzor::Variable>();
        };
        return tenzor::hessian(func, input);
    }, py::arg("func"), py::arg("input"),
    "Compute Hessian matrix of scalar function at input.\n"
    "Returns tensor of shape (input_size, input_size).");

    m.def("vmap", [](py::function py_func,
                      const tenzor::Variable& batched_input,
                      int64_t batch_dim) {
        auto func = [&py_func](const tenzor::Variable& x) -> tenzor::Variable {
            py::gil_scoped_acquire gil;
            py::object result = py_func(x);
            return result.cast<tenzor::Variable>();
        };
        return tenzor::vmap(func, batched_input, batch_dim);
    }, py::arg("func"), py::arg("batched_input"), py::arg("batch_dim") = 0,
    "Vectorized map: apply func independently to each element along batch dim.\n"
    "Equivalent to torch.vmap.");

    // =========================================================================
    // Serving submodule
    // =========================================================================
    auto serving = m.def_submodule("serving", "Inference serving infrastructure");

    py::class_<tenzor::serving::ServerConfig>(serving, "ServerConfig")
        .def(py::init<>())
        .def_readwrite("http_port", &tenzor::serving::ServerConfig::http_port)
        .def_readwrite("grpc_port", &tenzor::serving::ServerConfig::grpc_port)
        .def_readwrite("num_workers", &tenzor::serving::ServerConfig::num_workers)
        .def_readwrite("model_repository_path", &tenzor::serving::ServerConfig::model_repository_path);

    py::class_<tenzor::serving::BatchConfig>(serving, "BatchConfig")
        .def(py::init<>())
        .def_readwrite("max_batch_size", &tenzor::serving::BatchConfig::max_batch_size)
        .def_readwrite("max_latency_us", &tenzor::serving::BatchConfig::max_latency_us);

    py::class_<tenzor::serving::InferenceServer>(serving, "InferenceServer")
        .def(py::init<tenzor::serving::ServerConfig>(), py::arg("config"))
        .def("start", &tenzor::serving::InferenceServer::start)
        .def("stop", &tenzor::serving::InferenceServer::stop)
        .def("wait", &tenzor::serving::InferenceServer::wait)
        .def("repository", &tenzor::serving::InferenceServer::repository,
             py::return_value_policy::reference_internal);

    py::class_<tenzor::serving::ModelRepository>(serving, "ModelRepository")
        .def("load_model", &tenzor::serving::ModelRepository::load_model,
             py::arg("name"), py::arg("path"), py::arg("device"),
             py::arg("batch_config") = tenzor::serving::BatchConfig{})
        .def("unload_model", &tenzor::serving::ModelRepository::unload_model,
             py::arg("name"))
        .def("list_models", &tenzor::serving::ModelRepository::list_models);

    // =========================================================================
    // Monitor submodule
    // =========================================================================
    auto monitor_mod = m.def_submodule("monitor", "Event-based monitoring system");

    py::enum_<tenzor::monitor::Aggregation>(monitor_mod, "Aggregation")
        .value("Sum", tenzor::monitor::Aggregation::Sum)
        .value("Mean", tenzor::monitor::Aggregation::Mean)
        .value("Count", tenzor::monitor::Aggregation::Count)
        .value("MinMax", tenzor::monitor::Aggregation::MinMax)
        .value("Value", tenzor::monitor::Aggregation::Value);

    py::class_<tenzor::monitor::Stat>(monitor_mod, "Stat",
        "A named statistic with thread-safe accumulation")
        .def("add", &tenzor::monitor::Stat::add, py::arg("value"),
             "Accumulate a value into this statistic")
        .def("get", &tenzor::monitor::Stat::get,
             "Retrieve the current aggregate value")
        .def("count", &tenzor::monitor::Stat::count,
             "Number of add() calls since last reset")
        .def("name", &tenzor::monitor::Stat::name,
             "The name this stat was registered under")
        .def("reset", &tenzor::monitor::Stat::reset,
             "Reset this statistic to its initial state");

    // Trampoline class for Python-side EventHandler subclasses
    class PyEventHandler : public tenzor::monitor::EventHandler {
    public:
        using tenzor::monitor::EventHandler::EventHandler;

        auto handle(const std::string& event_name,
                    const std::unordered_map<std::string, double>& data) -> void override {
            py::gil_scoped_acquire gil;
            PYBIND11_OVERRIDE_PURE(void, tenzor::monitor::EventHandler, handle, event_name, data);
        }
    };

    py::class_<tenzor::monitor::EventHandler, PyEventHandler,
               std::shared_ptr<tenzor::monitor::EventHandler>>(monitor_mod, "EventHandler",
        "Base class for event handlers")
        .def(py::init<>())
        .def("handle", &tenzor::monitor::EventHandler::handle,
             py::arg("event_name"), py::arg("data"),
             "Handle a named event with key-value data payload");

    py::class_<tenzor::monitor::Monitor>(monitor_mod, "Monitor",
        "Singleton event-based monitoring system")
        .def_static("instance", &tenzor::monitor::Monitor::instance,
                    py::return_value_policy::reference,
                    "Get the process-wide Monitor instance")
        .def("register_stat", &tenzor::monitor::Monitor::register_stat,
             py::arg("name"), py::arg("agg"),
             py::return_value_policy::reference_internal,
             "Register a new statistic with the given aggregation type")
        .def("get_stat", &tenzor::monitor::Monitor::get_stat,
             py::arg("name"),
             py::return_value_policy::reference_internal,
             "Look up a statistic by name (returns None if not found)")
        .def("log_event",
             &tenzor::monitor::Monitor::log_event,
             py::arg("name"),
             py::arg("data") = std::unordered_map<std::string, double>{},
             "Dispatch a named event to all registered handlers")
        .def("add_handler", &tenzor::monitor::Monitor::add_handler,
             py::arg("handler"),
             "Add an event handler to the dispatch chain")
        .def("remove_all_handlers", &tenzor::monitor::Monitor::remove_all_handlers,
             "Remove all event handlers")
        .def("reset_all_stats", &tenzor::monitor::Monitor::reset_all_stats,
             "Reset every registered statistic")
        .def("stat_names", &tenzor::monitor::Monitor::stat_names,
             "Return the names of all registered statistics");

    // =========================================================================
    // Export submodule (AOT compilation)
    // =========================================================================
    auto export_mod = m.def_submodule("export_program", "AOT model export (torch.export-style)");

    py::class_<tenzor::export_::ExportOptions>(export_mod, "ExportOptions",
        "Options controlling the export process")
        .def(py::init<>())
        .def_readwrite("strict", &tenzor::export_::ExportOptions::strict,
                       "Strict mode: disallow dynamic control flow (default: True)")
        .def_readwrite("preserve_module_call_signature",
                       &tenzor::export_::ExportOptions::preserve_module_call_signature,
                       "Preserve module call signature metadata (default: False)");

    py::class_<tenzor::export_::ExportedProgram>(export_mod, "ExportedProgram",
        "A self-contained, serializable representation of a traced model")
        .def("save", &tenzor::export_::ExportedProgram::save,
             py::arg("path"),
             "Serialize the exported program to a binary file")
        .def_static("load", &tenzor::export_::ExportedProgram::load,
             py::arg("path"),
             py::arg("map_location") = std::nullopt,
             "Load an exported program from a binary file.  Pass "
             "map_location=Device.cpu() (or another device) to move every "
             "state tensor to that device on load; defaults to the device "
             "the program was originally saved on.")
        .def("run", &tenzor::export_::ExportedProgram::run,
             py::arg("inputs"),
             "Execute the exported program with runtime inputs")
        .def("state_dict", &tenzor::export_::ExportedProgram::state_dict,
             "Get the captured state dict")
        .def_property_readonly("num_inputs", &tenzor::export_::ExportedProgram::num_inputs,
             "Number of expected inputs")
        .def_property_readonly("num_outputs", &tenzor::export_::ExportedProgram::num_outputs,
             "Number of produced outputs");

    export_mod.def("export_model",
        [](tenzor::nn::Module& module,
           const std::vector<tenzor::Tensor>& example_inputs,
           bool strict,
           bool preserve_signature) {
            tenzor::export_::ExportOptions opts;
            opts.strict = strict;
            opts.preserve_module_call_signature = preserve_signature;
            return tenzor::export_::export_model(module, example_inputs, opts);
        },
        py::arg("module"),
        py::arg("example_inputs"),
        py::arg("strict") = true,
        py::arg("preserve_module_call_signature") = false,
        R"doc(
        Export a module via JIT tracing with example inputs.

        Traces the module's forward pass, captures the computation graph
        and state dict, and returns an ExportedProgram that can be saved,
        loaded, and executed independently of the original module.

        Args:
            module: The neural network module to export
            example_inputs: Example input tensors for tracing
            strict: If True, raise on dynamic control flow (default: True)
            preserve_module_call_signature: Preserve call signature metadata

        Returns:
            ExportedProgram containing the traced graph and state dict

        Example:
            >>> model = MyNetwork()
            >>> dummy = tenzor.Tensor([1, 784], tenzor.float32)
            >>> ep = tenzor.export_program.export_model(model, [dummy])
            >>> ep.save("model.tzep")
        )doc");
}
