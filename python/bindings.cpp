#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <iostream>
#include <sstream>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/graph_optimizer.hpp>
#include <tenzor/onnx/graph_module.hpp>
#include <tenzor/core/device_guard.hpp>
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
#include <tenzor/nn/optim/adam_atan2.hpp>
#include <tenzor/nn/layers/hrm.hpp>
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

// ============================================================================
// PyModule: Trampoline class for Python subclassing of nn::Module
// ============================================================================
// This enables Python users to create custom modules by subclassing tz.nn.Module
// and overriding forward_impl() method. The trampoline intercepts virtual calls
// and routes them to Python implementations.

// Type aliases to help with PYBIND11_OVERRIDE macros (avoid template comma issues)
using VariablePtr = std::shared_ptr<tenzor::Variable>;
using VariablePtrVec = std::vector<VariablePtr>;
using NamedParamPair = std::pair<std::string, VariablePtr>;
using NamedParamVec = std::vector<NamedParamPair>;
using StateDict = std::unordered_map<std::string, tenzor::Tensor>;

class PyModule : public tenzor::nn::Module {
public:
    // Inherit constructors
    using tenzor::nn::Module::Module;

    // Override forward_impl to call Python's forward() or forward_impl() method
    // Python users naturally define forward(), not forward_impl(), so we check both
    auto forward_impl(const tenzor::Variable& input) -> tenzor::Variable override {
        py::gil_scoped_acquire gil;

        // First, try Python's 'forward' method (the natural way users define it)
        py::function forward_override = py::get_override(this, "forward");
        if (forward_override) {
            py::object result = forward_override(input);
            if (!py::isinstance<tenzor::Variable>(result)) {
                throw std::runtime_error(
                    std::string("forward() must return a tenzor.Variable, got ") +
                    std::string(py::str(py::type::handle_of(result).attr("__name__"))));
            }
            return result.cast<tenzor::Variable>();
        }

        // Fall back to forward_impl if no forward override found.
        // Emit a one-time deprecation warning: users should override forward().
        {
            static bool warned = false;
            if (!warned) {
                warned = true;
                if (PyErr_WarnEx(PyExc_DeprecationWarning,
                        "Overriding forward_impl() is deprecated. "
                        "Override forward() instead (same as PyTorch).", 1) < 0) {
                    throw py::error_already_set();
                }
            }
        }
        PYBIND11_OVERRIDE_PURE(
            tenzor::Variable,           // Return type
            tenzor::nn::Module,         // Parent class
            forward_impl,               // Name of function in C++
            input                       // Arguments
        );
    }

    // Override parameters() to allow Python customization
    auto parameters() -> VariablePtrVec override {
        PYBIND11_OVERRIDE(
            VariablePtrVec,
            tenzor::nn::Module,
            parameters
        );
    }

    // Override own_parameters() to allow Python customization
    auto own_parameters() -> VariablePtrVec override {
        PYBIND11_OVERRIDE(
            VariablePtrVec,
            tenzor::nn::Module,
            own_parameters
        );
    }

    // Override named_parameters() to allow Python customization
    auto named_parameters() -> NamedParamVec override {
        PYBIND11_OVERRIDE(
            NamedParamVec,
            tenzor::nn::Module,
            named_parameters
        );
    }

    // Override state_dict() for custom serialization
    auto state_dict() const -> StateDict override {
        PYBIND11_OVERRIDE(
            StateDict,
            tenzor::nn::Module,
            state_dict
        );
    }

    // Override load_state_dict() for custom deserialization
    auto load_state_dict(const StateDict& state) -> void override {
        PYBIND11_OVERRIDE(
            void,
            tenzor::nn::Module,
            load_state_dict,
            state
        );
    }

    // Expose protected methods to Python
    void py_register_parameter(const std::string& name, tenzor::Variable param) {
        register_parameter(name, std::move(param));
    }

    void py_register_buffer(const std::string& name, tenzor::Variable buffer) {
        register_buffer(name, std::move(buffer));
    }

    void py_register_module(const std::string& name, std::shared_ptr<tenzor::nn::Module> module) {
        register_module(name, std::move(module));
    }
};

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

    // Catch-all translator: any future TenzorException-derived types not
    // explicitly registered above will still map to TenzorError in Python.
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

    // Device availability checks
    m.def("cuda_is_available", []() {
        auto& loader = tenzor::backend_registry();
        auto* cuda_backend = loader.get_backend("cuda");
        return cuda_backend != nullptr && cuda_backend->is_available();
    }, "Check if CUDA is available");

    m.def("cuda_device_count", []() {
        auto& loader = tenzor::backend_registry();
        auto* cuda_backend = loader.get_backend("cuda");
        return cuda_backend ? cuda_backend->device_count() : 0;
    }, "Get number of available CUDA devices");

    // CUDA Graph capture and replay
    py::class_<tenzor::CUDAGraph>(m, "CUDAGraph",
        "Captures a sequence of CUDA operations into a graph for fast replay.\n"
        "Shapes must be fixed during capture. No host-device sync during capture.")
        .def(py::init([](int32_t device_id) {
            auto graph = tenzor::CUDAGraph::create(device_id);
            if (!graph) {
                throw std::runtime_error("CUDA not available or invalid device_id");
            }
            return graph;
        }), py::arg("device_id") = 0)
        .def("begin_capture", &tenzor::CUDAGraph::begin_capture,
             "Begin capturing CUDA operations")
        .def("end_capture", &tenzor::CUDAGraph::end_capture,
             "End capture and compile the graph")
        .def("replay", &tenzor::CUDAGraph::replay,
             "Replay the captured graph")
        .def("is_ready", &tenzor::CUDAGraph::is_ready,
             "Check if graph has been captured and is ready for replay")
        .def("__enter__", [](tenzor::CUDAGraph& self) -> tenzor::CUDAGraph& {
            self.begin_capture();
            return self;
        })
        .def("__exit__", [](tenzor::CUDAGraph& self, py::object, py::object, py::object) {
            self.end_capture();
        });

    // Event wrapper for Python bindings
    struct PyEvent {
        tenzor::EventHandle handle{nullptr};
        tenzor::Backend* backend{nullptr};
    };

    py::class_<PyEvent>(m, "Event",
        "Synchronization event for inter-stream coordination and timing.\n"
        "Works with CUDA, ROCm, and OneAPI backends.")
        .def(py::init([](const std::string& device, int32_t device_id, bool enable_timing) {
            auto& loader = tenzor::backend_registry();
            auto* backend = loader.get_backend(device);
            if (!backend || !backend->is_available()) {
                throw std::runtime_error("Backend '" + device + "' is not available");
            }
            PyEvent ev;
            ev.handle = backend->create_event(device_id, enable_timing);
            ev.backend = backend;
            return ev;
        }), py::arg("device") = "cuda", py::arg("device_id") = 0, py::arg("enable_timing") = true)
        .def("record", [](PyEvent& self, tenzor::StreamHandle stream) {
            self.backend->record_event(self.handle, stream);
        }, py::arg("stream") = nullptr, "Record event on a stream")
        .def("wait", [](PyEvent& self, tenzor::StreamHandle stream) {
            self.backend->wait_event(self.handle, stream);
        }, py::arg("stream") = nullptr, "Make a stream wait for this event")
        .def("elapsed_time", [](PyEvent& self, PyEvent& end_event) {
            return self.backend->event_elapsed_ms(self.handle, end_event.handle);
        }, py::arg("end_event"), "Elapsed time in ms between this (start) and end_event")
        .def("__del__", [](PyEvent& self) {
            if (self.handle && self.backend) {
                self.backend->destroy_event(self.handle);
                self.handle = nullptr;
            }
        });

    // Vulkan device availability
    m.def("vulkan_is_available", []() {
        auto& loader = tenzor::backend_registry();
        auto* vulkan_backend = loader.get_backend("vulkan");
        return vulkan_backend != nullptr && vulkan_backend->is_available();
    }, "Check if Vulkan is available");

    m.def("vulkan_device_count", []() {
        auto& loader = tenzor::backend_registry();
        auto* vulkan_backend = loader.get_backend("vulkan");
        return vulkan_backend ? vulkan_backend->device_count() : 0;
    }, "Get number of available Vulkan devices");

    // OneAPI device availability
    m.def("oneapi_is_available", []() {
        auto& loader = tenzor::backend_registry();
        auto* oneapi_backend = loader.get_backend("oneapi");
        return oneapi_backend != nullptr && oneapi_backend->is_available();
    }, "Check if OneAPI (SYCL) is available");

    m.def("oneapi_device_count", []() {
        auto& loader = tenzor::backend_registry();
        auto* oneapi_backend = loader.get_backend("oneapi");
        return oneapi_backend ? oneapi_backend->device_count() : 0;
    }, "Get number of available OneAPI devices");

    // Op coverage introspection API
    m.def("get_supported_ops", [](const std::string& device_str) {
        auto device = tenzor::Device::from_string(device_str);
        auto ops = tenzor::get_supported_ops(device.type);
        std::vector<std::string> names;
        names.reserve(ops.size());
        for (auto op : ops) {
            names.push_back(std::string(tenzor::op_id_to_name(op)));
        }
        return names;
    }, py::arg("device"), "Get list of supported operation names for a device");

    m.def("supports_op", [](const std::string& device_str, const std::string& op_name) {
        auto device = tenzor::Device::from_string(device_str);
        auto& table = tenzor::DispatchTableRegistry::get_table_const(device.type);
        auto ops = table.supported_ops();
        for (auto op : ops) {
            if (std::string(tenzor::op_id_to_name(op)) == op_name) return true;
        }
        return false;
    }, py::arg("device"), py::arg("op_name"), "Check if a device supports a specific operation");

    m.def("op_count", [](const std::string& device_str) {
        auto device = tenzor::Device::from_string(device_str);
        return tenzor::DispatchTableRegistry::get_table_const(device.type).op_count();
    }, py::arg("device"), "Get count of registered operations for a device");

    // ROCm device availability
    m.def("rocm_is_available", []() {
        auto& loader = tenzor::backend_registry();
        auto* rocm_backend = loader.get_backend("rocm");
        return rocm_backend != nullptr && rocm_backend->is_available();
    }, "Check if ROCm (AMD GPU) is available");

    m.def("rocm_device_count", []() {
        auto& loader = tenzor::backend_registry();
        auto* rocm_backend = loader.get_backend("rocm");
        return rocm_backend ? rocm_backend->device_count() : 0;
    }, "Get number of available ROCm devices");

    // List all available backends
    m.def("list_backends", []() {
        auto& loader = tenzor::backend_registry();
        return loader.available_backends();
    }, "Get list of all registered backend names");

    // Generic is_available for any backend
    m.def("is_backend_available", [](const std::string& backend_name) {
        auto& loader = tenzor::backend_registry();
        auto* backend = loader.get_backend(backend_name);
        return backend != nullptr && backend->is_available();
    }, py::arg("backend_name"),
    "Check if a specific backend is available");

    // Generic device_count for any backend
    m.def("backend_device_count", [](const std::string& backend_name) {
        auto& loader = tenzor::backend_registry();
        auto* backend = loader.get_backend(backend_name);
        return backend ? backend->device_count() : 0;
    }, py::arg("backend_name"),
    "Get number of devices for a specific backend");

    // Get device info for a specific device
    m.def("get_device_info", [](const std::string& backend_name, int32_t device_id) {
        auto& loader = tenzor::backend_registry();
        auto* backend = loader.get_backend(backend_name);
        if (!backend) {
            throw std::runtime_error("Backend not found: " + backend_name);
        }
        if (device_id < 0 || device_id >= backend->device_count()) {
            throw std::out_of_range("Invalid device ID: " + std::to_string(device_id));
        }

        auto info = backend->get_device_info(device_id);

        py::dict result;
        result["name"] = info.name;
        result["vendor"] = info.vendor;
        result["driver_version"] = info.driver_version;
        result["total_memory"] = info.total_memory;
        result["available_memory"] = info.available_memory;
        result["compute_units"] = info.compute_units;
        result["max_threads_per_block"] = info.max_threads_per_block;
        result["max_shared_memory"] = info.max_shared_memory;
        result["warp_size"] = info.warp_size;
        result["major_version"] = info.major_version;
        result["minor_version"] = info.minor_version;
        result["supports_fp16"] = info.supports_fp16;
        result["supports_fp64"] = info.supports_fp64;
        result["supports_int8"] = info.supports_int8;
        result["is_integrated"] = info.is_integrated;
        result["is_discrete"] = info.is_discrete;
        result["pci_bus_id"] = info.pci_bus_id;
        result["pci_device_id"] = info.pci_device_id;
        return result;
    }, py::arg("backend_name"), py::arg("device_id") = 0,
    "Get detailed information about a specific device");

    // DeviceInfo as a typed Python class
    py::class_<tenzor::DeviceInfo>(m, "DeviceInfo",
        "Hardware properties for a compute device")
        .def_readonly("name", &tenzor::DeviceInfo::name, "Device name")
        .def_readonly("vendor", &tenzor::DeviceInfo::vendor, "Vendor name")
        .def_readonly("driver_version", &tenzor::DeviceInfo::driver_version, "Driver version")
        .def_readonly("total_memory", &tenzor::DeviceInfo::total_memory, "Total memory (bytes)")
        .def_readonly("available_memory", &tenzor::DeviceInfo::available_memory, "Available memory (bytes)")
        .def_readonly("compute_units", &tenzor::DeviceInfo::compute_units, "Number of compute units/SMs")
        .def_readonly("max_threads_per_block", &tenzor::DeviceInfo::max_threads_per_block, "Max threads per block")
        .def_readonly("max_shared_memory", &tenzor::DeviceInfo::max_shared_memory, "Max shared memory (bytes)")
        .def_readonly("warp_size", &tenzor::DeviceInfo::warp_size, "Warp/wavefront size")
        .def_readonly("major_version", &tenzor::DeviceInfo::major_version, "Compute capability major")
        .def_readonly("minor_version", &tenzor::DeviceInfo::minor_version, "Compute capability minor")
        .def_readonly("supports_fp16", &tenzor::DeviceInfo::supports_fp16, "FP16 support")
        .def_readonly("supports_fp64", &tenzor::DeviceInfo::supports_fp64, "FP64 support")
        .def_readonly("supports_int8", &tenzor::DeviceInfo::supports_int8, "INT8 support")
        .def_readonly("is_integrated", &tenzor::DeviceInfo::is_integrated, "Integrated GPU")
        .def_readonly("is_discrete", &tenzor::DeviceInfo::is_discrete, "Discrete GPU")
        .def_readonly("pci_bus_id", &tenzor::DeviceInfo::pci_bus_id, "PCI bus ID")
        .def_readonly("pci_device_id", &tenzor::DeviceInfo::pci_device_id, "PCI device ID")
        .def("__repr__", [](const tenzor::DeviceInfo& info) {
            return "DeviceInfo(name='" + info.name + "', vendor='" + info.vendor +
                   "', memory=" + std::to_string(info.total_memory / (1024*1024)) + "MB)";
        });

    // Get device properties using a Device object (returns typed DeviceInfo)
    m.def("get_device_properties", [](const tenzor::Device& device) {
        return tenzor::get_device_properties(device);
    }, py::arg("device"),
    "Get detailed hardware properties for a Device object");

    // Get all devices across all backends
    m.def("get_all_devices", []() {
        auto& loader = tenzor::backend_registry();
        py::list devices;

        for (const auto& backend_name : loader.available_backends()) {
            auto* backend = loader.get_backend(backend_name);
            if (backend && backend->is_available()) {
                for (int32_t i = 0; i < backend->device_count(); ++i) {
                    auto info = backend->get_device_info(i);

                    py::dict device;
                    device["backend"] = backend_name;
                    device["device_id"] = i;
                    device["name"] = info.name;
                    device["vendor"] = info.vendor;
                    device["driver_version"] = info.driver_version;
                    device["total_memory"] = info.total_memory;
                    device["available_memory"] = info.available_memory;
                    device["compute_units"] = info.compute_units;
                    device["max_threads_per_block"] = info.max_threads_per_block;
                    device["max_shared_memory"] = info.max_shared_memory;
                    device["warp_size"] = info.warp_size;
                    device["major_version"] = info.major_version;
                    device["minor_version"] = info.minor_version;
                    device["supports_fp16"] = info.supports_fp16;
                    device["supports_fp64"] = info.supports_fp64;
                    device["supports_int8"] = info.supports_int8;
                    device["is_integrated"] = info.is_integrated;
                    device["is_discrete"] = info.is_discrete;
                    device["pci_bus_id"] = info.pci_bus_id;
                    device["pci_device_id"] = info.pci_device_id;
                    devices.append(device);
                }
            }
        }
        return devices;
    }, "Get list of all available devices across all backends with their properties");

    // CUDA backend configuration
    auto cuda_mod = m.def_submodule("cuda", "CUDA backend configuration");
    auto matmul_mod = cuda_mod.def_submodule("matmul", "CUDA matmul configuration");
    matmul_mod.def("allow_tf32", &tenzor::cuda::matmul::allow_tf32,
        "Check if TF32 Tensor Cores are allowed for Float32 matmul");
    matmul_mod.def("set_allow_tf32", &tenzor::cuda::matmul::set_allow_tf32,
        py::arg("value"),
        "Set whether TF32 Tensor Cores are allowed for Float32 matmul");
    matmul_mod.def("warn_fp16_saturation", &tenzor::cuda::matmul::warn_fp16_saturation,
        "Check if FP16 saturation warnings are enabled");
    matmul_mod.def("set_warn_fp16_saturation", &tenzor::cuda::matmul::set_warn_fp16_saturation,
        py::arg("value"),
        "Set whether FP16 saturation warnings are enabled");

    // Device::Type enum (must be defined before Device class for default args)
    py::enum_<tenzor::Device::Type>(m, "DeviceType")
        .value("CPU", tenzor::Device::Type::CPU)
        .value("CUDA", tenzor::Device::Type::CUDA)
        .value("ROCm", tenzor::Device::Type::ROCm)
        .value("OneAPI", tenzor::Device::Type::OneAPI)
        .value("Vulkan", tenzor::Device::Type::Vulkan);

    // Device
    py::class_<tenzor::Device>(m, "Device")
        .def(py::init<tenzor::Device::Type, int32_t>())
        .def_static("cpu", &tenzor::Device::cpu)
        .def_static("cuda", &tenzor::Device::cuda, py::arg("index") = 0)
        .def_static("rocm", &tenzor::Device::rocm, py::arg("index") = 0)
        .def_static("oneapi", &tenzor::Device::oneapi, py::arg("index") = 0)
        .def_static("vulkan", &tenzor::Device::vulkan, py::arg("index") = 0)
        .def_readonly("type", &tenzor::Device::type)
        .def_readonly("index", &tenzor::Device::index)
        .def("__repr__", [](const tenzor::Device& d) {
            return d.to_string();
        });

    // DeviceGuard — RAII device context manager
    py::class_<tenzor::DeviceGuard>(m, "DeviceGuard",
        "RAII guard that sets a device on construction and restores on destruction.\n"
        "Use as a context manager: with tz.DeviceGuard(tz.Device.cuda(1)): ...")
        .def(py::init<tenzor::Device>(), py::arg("device"))
        .def("__enter__", [](tenzor::DeviceGuard& self) -> tenzor::DeviceGuard& {
            return self;
        })
        .def("__exit__", [](tenzor::DeviceGuard& self,
                            py::object /*exc_type*/, py::object /*exc_val*/, py::object /*exc_tb*/) {
            // Destructor handles restore; force it now by calling detail::switch_device
            tenzor::detail::switch_device(self.device().type, self.previous_device_index());
        });

    // current_device utility
    m.def("current_device", &tenzor::current_device, py::arg("type"),
        "Get the current device for a backend type in this thread.");

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
        .value("complex128", tenzor::DType::Complex128)
        .value("qint8", tenzor::DType::QInt8)
        .value("quint8", tenzor::DType::QUInt8)
        .value("qint4x2", tenzor::DType::QInt4x2);

    // Quantization functions
    m.def("quantize_per_tensor", &tenzor::quantize_per_tensor,
          py::arg("input"), py::arg("scale"), py::arg("zero_point"),
          py::arg("dtype") = tenzor::DType::QInt8,
          "Quantize a float tensor to int8/uint8 with scale and zero_point");

    // FP8 scaling utilities
    m.def("fp8_max_value", &tenzor::fp8_max_value,
          py::arg("fp8_dtype"),
          R"doc(Get maximum representable value for an FP8 data type.

Args:
    fp8_dtype: FP8_E4M3 (max=448) or FP8_E5M2 (max=57344)

Returns:
    Maximum finite value as float.
)doc");

    m.def("compute_amax", &tenzor::compute_amax,
          py::arg("tensor"),
          R"doc(Compute the absolute maximum value of a tensor.

Args:
    tensor: Input tensor (any floating-point dtype)

Returns:
    Absolute maximum value as float.
)doc");

    m.def("compute_fp8_scale", &tenzor::compute_fp8_scale,
          py::arg("amax"), py::arg("fp8_dtype"),
          R"doc(Compute the FP8 quantization scale from an amax value.

The scale maps the tensor's dynamic range to the FP8 representable range:
scale = amax / fp8_max.

Args:
    amax: Absolute maximum value of the tensor
    fp8_dtype: Target FP8 dtype

Returns:
    Scale factor as float.
)doc");

    m.def("quantize_to_fp8", &tenzor::quantize_to_fp8,
          py::arg("input"), py::arg("fp8_dtype"),
          py::arg("scale") = std::nullopt,
          py::arg("stochastic_rounding") = false,
          R"doc(Quantize a floating-point tensor to FP8 with per-tensor scaling.

Args:
    input: Input tensor (Float32, Float16, or BFloat16)
    fp8_dtype: Target FP8 dtype (FP8_E4M3 or FP8_E5M2)
    scale: Optional pre-computed scale (auto-computed if None)
    stochastic_rounding: Use stochastic rounding for better training accuracy

Returns:
    Tuple of (FP8 tensor, FP8ScalingParams).

Example::

    fp8_tensor, params = tz.quantize_to_fp8(x, tz.fp8_e4m3)
    x_restored = tz.dequantize_from_fp8(fp8_tensor, params.scale)
)doc");

    m.def("dequantize_from_fp8", &tenzor::dequantize_from_fp8,
          py::arg("fp8_tensor"), py::arg("scale"),
          R"doc(Dequantize an FP8 tensor back to Float32 using a scale factor.

Args:
    fp8_tensor: FP8 tensor to dequantize
    scale: Scale factor from quantization

Returns:
    Float32 tensor.
)doc");

    // Memory format enum (PyTorch-compatible channels_last support)
    py::enum_<tenzor::MemoryFormat>(m, "memory_format")
        .value("contiguous_format", tenzor::MemoryFormat::Contiguous,
               "Standard row-major (NCHW for 4D tensors)")
        .value("channels_last", tenzor::MemoryFormat::ChannelsLast,
               "Channels-last layout (NHWC for 4D tensors) - optimized for Tensor Cores")
        .value("channels_last_3d", tenzor::MemoryFormat::ChannelsLast3d,
               "Channels-last for 5D tensors (NDHWC)")
        .value("preserve_format", tenzor::MemoryFormat::Preserve,
               "Preserve input format in operations");

    // Convenient module-level attributes for memory formats (PyTorch-style)
    m.attr("contiguous_format") = tenzor::MemoryFormat::Contiguous;
    m.attr("channels_last") = tenzor::MemoryFormat::ChannelsLast;
    m.attr("channels_last_3d") = tenzor::MemoryFormat::ChannelsLast3d;
    m.attr("preserve_format") = tenzor::MemoryFormat::Preserve;

    // Tensor class
    py::class_<tenzor::Tensor>(m, "Tensor", py::buffer_protocol())
        .def(py::init<std::vector<int64_t>, tenzor::DType, tenzor::Device>(),
             py::arg("shape"),
             py::arg("dtype") = tenzor::DType::Float32,
             py::arg("device") = tenzor::Device::cpu())
        .def_static("from_blob", [](py::object obj,
                                     std::vector<int64_t> shape,
                                     tenzor::DType dtype,
                                     tenzor::Device device) {
            // Get raw pointer from Python buffer protocol
            py::buffer buf = py::buffer(obj);
            py::buffer_info info = buf.request();
            // Keep the Python object alive by capturing it in the deleter closure.
            // When the last Tensor sharing this storage is destroyed, the shared_ptr
            // to ExternalStorage dies, which destroys the std::function deleter,
            // which drops the py::object reference, allowing Python GC.
            auto obj_ref = std::make_shared<py::object>(std::move(obj));
            return tenzor::Tensor::from_blob(
                info.ptr, std::move(shape), dtype, device,
                [obj_ref](void*) { /* prevent GC until tensor dies */ });
        },
        py::arg("buffer"), py::arg("shape"),
        py::arg("dtype") = tenzor::DType::Float32,
        py::arg("device") = tenzor::Device::cpu(),
        "Wrap a buffer (e.g. numpy array) as a Tensor without copying data.\n"
        "The buffer must remain alive while the tensor exists.")
        .def_property_readonly("shape",
            [](const tenzor::Tensor& t) {
                auto s = t.shape();
                return std::vector<int64_t>(s.begin(), s.end());
            })
        .def_property_readonly("ndim", &tenzor::Tensor::ndim)
        .def_property_readonly("names", [](const tenzor::Tensor& t) -> py::object {
            auto names = t.names();
            if (!names) return py::none();
            py::tuple result(names->size());
            for (size_t i = 0; i < names->size(); ++i) {
                if ((*names)[i].is_wildcard()) {
                    result[i] = py::none();
                } else {
                    result[i] = py::str(std::string((*names)[i].name()));
                }
            }
            return result;
        }, "Get dimension names (None if unnamed)")
        .def("has_names", &tenzor::Tensor::has_names, "Check if tensor has named dimensions")
        .def("rename", [](const tenzor::Tensor& t, py::args py_names) -> tenzor::Tensor {
            tenzor::DimnameList names;
            for (auto& n : py_names) {
                if (n.is_none()) {
                    names.push_back(tenzor::Dimname::wildcard());
                } else {
                    names.push_back(tenzor::Dimname(n.cast<std::string>()));
                }
            }
            return t.rename(std::move(names));
        }, "Return a view with named dimensions")
        .def("dim_index", &tenzor::Tensor::dim_index,
             py::arg("name"), "Find dimension index by name")
        .def_property_readonly("dtype", &tenzor::Tensor::dtype)
        .def_property_readonly("device", &tenzor::Tensor::device)
        .def_property_readonly("is_cuda", [](const tenzor::Tensor& self) {
            return self.device().type == tenzor::Device::Type::CUDA;
        })
        .def_property_readonly("is_cpu", [](const tenzor::Tensor& self) {
            return self.device().type == tenzor::Device::Type::CPU;
        })
        .def_property_readonly("numel", &tenzor::Tensor::numel)
        .def("is_quantized", &tenzor::Tensor::is_quantized, "Check if tensor is quantized")
        .def("q_scale", &tenzor::Tensor::q_scale, "Get quantization scale")
        .def("q_zero_point", &tenzor::Tensor::q_zero_point, "Get quantization zero point")
        .def("int_repr", &tenzor::Tensor::int_repr, "Get integer representation of quantized tensor")
        .def("dequantize", &tenzor::Tensor::dequantize, "Dequantize to float32")
        .def("is_per_channel_quantized", &tenzor::Tensor::is_per_channel_quantized,
            "Check if tensor uses per-channel quantization")
        .def("q_per_channel_scales", &tenzor::Tensor::q_per_channel_scales,
            "Get per-channel quantization scales")
        .def("q_per_channel_zero_points", &tenzor::Tensor::q_per_channel_zero_points,
            "Get per-channel quantization zero points")
        .def("q_per_channel_axis", &tenzor::Tensor::q_per_channel_axis,
            "Get per-channel quantization axis")
        .def("set_per_channel_quantization_params",
            &tenzor::Tensor::set_per_channel_quantization_params,
            py::arg("scales"), py::arg("zero_points"), py::arg("axis"),
            "Set per-channel quantization parameters")
        .def("is_contiguous",
            [](const tenzor::Tensor& t, std::optional<tenzor::MemoryFormat> fmt) {
                if (fmt.has_value()) {
                    return t.is_contiguous(fmt.value());
                }
                return t.is_contiguous();
            },
            py::arg("memory_format") = py::none(),
            "Check if tensor is contiguous. Optionally specify memory_format (channels_last, etc)")
        .def("memory_format", &tenzor::Tensor::memory_format,
             "Get the memory format of the tensor (contiguous_format or channels_last)")
        .def("version", &tenzor::Tensor::version,
             "Get the version counter (incremented by in-place operations)")
        .def("to", py::overload_cast<tenzor::Device>(&tenzor::Tensor::to, py::const_),
             py::arg("device"),
             "Move tensor to specified device",
             py::call_guard<py::gil_scoped_release>())
        .def("to", py::overload_cast<tenzor::Device, tenzor::DType>(&tenzor::Tensor::to, py::const_),
             py::arg("device"), py::arg("dtype"),
             "Move tensor to specified device and convert dtype in one operation",
             py::call_guard<py::gil_scoped_release>())
        .def("to", py::overload_cast<tenzor::MemoryFormat>(&tenzor::Tensor::to, py::const_),
             py::arg("memory_format"),
             "Convert tensor to specified memory format (e.g., channels_last for NHWC)",
             py::call_guard<py::gil_scoped_release>())
        .def("reshape", &tenzor::Tensor::reshape,
             py::arg("shape"),
             "Return a tensor with the same data but a new shape")
        // Shape manipulation
        .def("transpose", &tenzor::Tensor::transpose,
             py::arg("dim0"), py::arg("dim1"),
             "Swap two dimensions of the tensor")
        .def("permute", &tenzor::Tensor::permute,
             py::arg("dims"),
             "Permute the dimensions of the tensor")
        .def("squeeze", &tenzor::Tensor::squeeze,
             py::arg("dim") = py::none(),
             "Remove size-1 dimensions")
        .def("unsqueeze", &tenzor::Tensor::unsqueeze,
             py::arg("dim"),
             "Insert a size-1 dimension at the given position")
        .def("flatten", &tenzor::Tensor::flatten,
             py::arg("start_dim") = 0, py::arg("end_dim") = -1,
             "Flatten consecutive dimensions into one")
        .def("view", &tenzor::Tensor::view,
             py::arg("shape"),
             "Return a view with a new shape (tensor must be contiguous)")
        .def("expand", [](const tenzor::Tensor& self, std::vector<int64_t> shape) {
             return tenzor::expand(self, std::move(shape));
             }, py::arg("shape"),
             "Expand tensor to a larger size (broadcast without copying)",
             py::call_guard<py::gil_scoped_release>())
        .def("repeat", [](const tenzor::Tensor& self, std::vector<int64_t> repeats) {
             return tenzor::repeat(self, std::move(repeats));
             }, py::arg("repeats"),
             "Repeat tensor along each dimension",
             py::call_guard<py::gil_scoped_release>())
        .def("narrow", &tenzor::Tensor::narrow,
             py::arg("dim"), py::arg("start"), py::arg("length"),
             "Narrow (slice) tensor along a dimension",
             py::call_guard<py::gil_scoped_release>())
        .def("select", &tenzor::Tensor::select,
             py::arg("dim"), py::arg("index"),
             "Select a single index along a dimension",
             py::call_guard<py::gil_scoped_release>())
        .def("chunk", &tenzor::Tensor::chunk,
             py::arg("chunks"), py::arg("dim") = 0,
             "Split tensor into chunks along a dimension",
             py::call_guard<py::gil_scoped_release>())
        // Memory operations
        .def("clone", &tenzor::Tensor::clone,
             "Return a copy of the tensor with new storage",
             py::call_guard<py::gil_scoped_release>())
        .def("detach", &tenzor::Tensor::detach,
             "Return a tensor detached from the computation graph")
        .def("contiguous", &tenzor::Tensor::contiguous,
             "Return a contiguous tensor (copy if needed)",
             py::call_guard<py::gil_scoped_release>())
        .def("fill_", &tenzor::Tensor::fill_, py::arg("value"),
             "Fill tensor with scalar value in-place",
             py::call_guard<py::gil_scoped_release>())
        .def("zero_", &tenzor::Tensor::zero_,
             "Fill tensor with zeros in-place",
             py::call_guard<py::gil_scoped_release>())
        // Buffer protocol support (enables memoryview, numpy.asarray, etc.)
        .def_buffer([](tenzor::Tensor& t) -> py::buffer_info {
            if (t.device().type != tenzor::Device::Type::CPU) {
                throw std::runtime_error("Buffer protocol only supported for CPU tensors");
            }
            if (!t.is_contiguous()) {
                throw std::runtime_error("Buffer protocol requires a contiguous tensor. Call .contiguous() first.");
            }

            // Map DType to Python struct format string
            std::string format;
            switch (t.dtype()) {
                case tenzor::DType::Float32: format = py::format_descriptor<float>::format(); break;
                case tenzor::DType::Float64: format = py::format_descriptor<double>::format(); break;
                case tenzor::DType::Int32: format = py::format_descriptor<int32_t>::format(); break;
                case tenzor::DType::Int64: format = py::format_descriptor<int64_t>::format(); break;
                case tenzor::DType::Int16: format = py::format_descriptor<int16_t>::format(); break;
                case tenzor::DType::Int8: format = py::format_descriptor<int8_t>::format(); break;
                case tenzor::DType::UInt8: format = py::format_descriptor<uint8_t>::format(); break;
                case tenzor::DType::Bool: format = py::format_descriptor<bool>::format(); break;
                default: throw std::runtime_error("Buffer protocol not supported for dtype");
            }

            auto shape = t.shape();
            auto strides = t.strides();
            auto elem_size = static_cast<ssize_t>(t.dtype_size());

            std::vector<ssize_t> shape_vec(shape.begin(), shape.end());
            std::vector<ssize_t> stride_bytes;
            for (auto s : strides) {
                stride_bytes.push_back(static_cast<ssize_t>(s) * elem_size);
            }

            return py::buffer_info(
                t.data_ptr(),
                elem_size,
                format,
                static_cast<ssize_t>(t.ndim()),
                shape_vec,
                stride_bytes
            );
        })
        // NumPy interoperability
        .def("numpy", [](const tenzor::Tensor& t) {
             // Release GIL during CPU transfer (pure C++ work)
             tenzor::Tensor cpu_tensor;
             {
                 py::gil_scoped_release release;
                 cpu_tensor = tenzor::numpy::prepare_tensor_for_numpy(t);
             }
             // GIL reacquired for NumPy array creation (Python objects)
             return tenzor::numpy::create_numpy_array(cpu_tensor, t.dtype());
             },
             "Convert tensor to NumPy array (zero-copy when possible)")
        .def_static("from_numpy", &tenzor::numpy::numpy_to_tensor,
             py::arg("array"), py::arg("device") = tenzor::Device::cpu(),
             "Create tensor from NumPy array (zero-copy when possible)")
        // Scalar extraction
        .def("item", [](const tenzor::Tensor& t) -> py::object {
            if (t.numel() != 1) {
                throw std::runtime_error("item() only works for scalar tensors");
            }
            // Extract scalar value with GIL released (may involve device sync)
            auto dtype = t.dtype();
            double dval = 0;
            int64_t ival = 0;
            uint64_t uval = 0;
            bool bval = false;
            std::complex<double> cval{};
            bool is_float = false, is_int = false, is_uint = false, is_bool = false, is_complex = false;
            {
                py::gil_scoped_release release;
                switch (dtype) {
                    case tenzor::DType::Float32:  dval = t.item<float>(); is_float = true; break;
                    case tenzor::DType::Float64:  dval = t.item<double>(); is_float = true; break;
                    case tenzor::DType::Float16:  dval = static_cast<float>(t.data<tenzor::Float16>()[0]); is_float = true; break;
                    case tenzor::DType::BFloat16: dval = static_cast<float>(t.data<tenzor::BFloat16>()[0]); is_float = true; break;
                    case tenzor::DType::Int8:     ival = t.item<int8_t>(); is_int = true; break;
                    case tenzor::DType::Int16:    ival = t.item<int16_t>(); is_int = true; break;
                    case tenzor::DType::Int32:    ival = t.item<int32_t>(); is_int = true; break;
                    case tenzor::DType::Int64:    ival = t.item<int64_t>(); is_int = true; break;
                    case tenzor::DType::UInt8:    uval = t.item<uint8_t>(); is_uint = true; break;
                    case tenzor::DType::UInt16:   uval = t.item<uint16_t>(); is_uint = true; break;
                    case tenzor::DType::UInt32:   uval = t.item<uint32_t>(); is_uint = true; break;
                    case tenzor::DType::UInt64:   uval = t.item<uint64_t>(); is_uint = true; break;
                    case tenzor::DType::Bool:     bval = t.item<bool>(); is_bool = true; break;
                    case tenzor::DType::Complex64: {
                        auto c = t.item<std::complex<float>>();
                        cval = {c.real(), c.imag()};
                        is_complex = true; break;
                    }
                    case tenzor::DType::Complex128: cval = t.item<std::complex<double>>(); is_complex = true; break;
                    default:
                        throw std::runtime_error("Unsupported dtype for item()");
                }
            }
            // Create Python object with GIL held
            if (is_float)   return py::cast(dval);
            if (is_int)     return py::cast(ival);
            if (is_uint)    return py::cast(uval);
            if (is_bool)    return py::cast(bval);
            if (is_complex) return py::cast(cval);
            throw std::runtime_error("Unsupported dtype for item()");
        }, "Extract scalar value from single-element tensor")
        // Arithmetic operators - Tensor-Tensor (GIL released for compute)
        .def("__add__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return a + b; },
             py::call_guard<py::gil_scoped_release>())
        .def("__sub__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return a - b; },
             py::call_guard<py::gil_scoped_release>())
        .def("__mul__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return a * b; },
             py::call_guard<py::gil_scoped_release>())
        .def("__truediv__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return a / b; },
             py::is_operator(), "Element-wise division",
             py::call_guard<py::gil_scoped_release>())
        // Arithmetic operators - Tensor-Scalar (double to match Python float precision)
        .def("__add__", [](const tenzor::Tensor& a, double b) -> tenzor::Tensor {
             return a + b;
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__radd__", [](const tenzor::Tensor& a, double b) -> tenzor::Tensor {
             return a + b;  // addition is commutative
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__sub__", [](const tenzor::Tensor& a, double b) -> tenzor::Tensor {
             return a - b;
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__rsub__", [](const tenzor::Tensor& a, double b) -> tenzor::Tensor {
             return tenzor::neg(a - b);  // b - a = -(a - b)
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__mul__", [](const tenzor::Tensor& a, double b) -> tenzor::Tensor {
             return a * b;
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__rmul__", [](const tenzor::Tensor& a, double b) -> tenzor::Tensor {
             return a * b;  // multiplication is commutative
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__truediv__", [](const tenzor::Tensor& a, double b) -> tenzor::Tensor {
             return a / b;
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__rtruediv__", [](const tenzor::Tensor& a, double b) -> tenzor::Tensor {
             // b / a = b * reciprocal(a)
             return tenzor::reciprocal(a) * b;
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__pow__", [](const tenzor::Tensor& a, double exponent) -> tenzor::Tensor {
             return tenzor::pow(a, exponent);
             }, py::is_operator(), "Element-wise power",
             py::call_guard<py::gil_scoped_release>())
        .def("__neg__", [](const tenzor::Tensor& a) -> tenzor::Tensor { return tenzor::neg(a); },
             py::is_operator(), "Unary negation",
             py::call_guard<py::gil_scoped_release>())
        // Matrix multiplication
        .def("__matmul__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
             return tenzor::matmul(a, b);
             }, py::is_operator(), "Matrix multiplication (@ operator)",
             py::call_guard<py::gil_scoped_release>())
        .def("__rmatmul__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
             return tenzor::matmul(b, a);
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        // Modulo and floor division
        .def("__mod__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
             return tenzor::fmod(a, b);
             }, py::is_operator(), "Element-wise modulo",
             py::call_guard<py::gil_scoped_release>())
        .def("__mod__", [](const tenzor::Tensor& a, double b) -> tenzor::Tensor {
             auto b_tensor = tenzor::full(std::vector<int64_t>{}, b,
                                          a.dtype(), a.device());
             return tenzor::fmod(a, b_tensor);
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__rmod__", [](const tenzor::Tensor& a, double b) -> tenzor::Tensor {
             auto b_tensor = tenzor::full(std::vector<int64_t>{}, b,
                                          a.dtype(), a.device());
             return tenzor::fmod(b_tensor, a);
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__floordiv__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
             return tenzor::floor(a / b);
             }, py::is_operator(), "Element-wise floor division",
             py::call_guard<py::gil_scoped_release>())
        .def("__floordiv__", [](const tenzor::Tensor& a, double b) -> tenzor::Tensor {
             return tenzor::floor(a / b);
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__rfloordiv__", [](const tenzor::Tensor& a, float b) -> tenzor::Tensor {
             auto b_tensor = tenzor::full(std::vector<int64_t>{}, static_cast<double>(b),
                                          a.dtype(), a.device());
             return tenzor::floor(b_tensor / a);
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        // In-place operators
        .def("__iadd__", [](tenzor::Tensor& a, const tenzor::Tensor& b) -> tenzor::Tensor& {
             tenzor::add_(a, b); return a;
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__isub__", [](tenzor::Tensor& a, const tenzor::Tensor& b) -> tenzor::Tensor& {
             tenzor::sub_(a, b); return a;
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__imul__", [](tenzor::Tensor& a, const tenzor::Tensor& b) -> tenzor::Tensor& {
             tenzor::mul_(a, b); return a;
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__itruediv__", [](tenzor::Tensor& a, const tenzor::Tensor& b) -> tenzor::Tensor& {
             tenzor::div_(a, b); return a;
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        // Numeric protocol
        .def("__float__", [](const tenzor::Tensor& t) -> double {
             if (t.numel() != 1) throw py::value_error("only one element tensors can be converted to Python scalars");
             switch (t.dtype()) {
                 case tenzor::DType::Float64: return t.item<double>();
                 case tenzor::DType::Float32: return static_cast<double>(t.item<float>());
                 case tenzor::DType::Float16: return static_cast<double>(static_cast<float>(t.item<tenzor::Float16>()));
                 case tenzor::DType::BFloat16: return static_cast<double>(static_cast<float>(t.item<tenzor::BFloat16>()));
                 case tenzor::DType::Int64: return static_cast<double>(t.item<int64_t>());
                 case tenzor::DType::Int32: return static_cast<double>(t.item<int32_t>());
                 case tenzor::DType::Int16: return static_cast<double>(t.item<int16_t>());
                 case tenzor::DType::Int8: return static_cast<double>(t.item<int8_t>());
                 case tenzor::DType::UInt8: return static_cast<double>(t.item<uint8_t>());
                 case tenzor::DType::Bool: return static_cast<double>(t.item<bool>());
                 default: throw py::type_error("Cannot convert " + std::string(tenzor::dtype_name(t.dtype())) + " to float");
             }
             })
        .def("__int__", [](const tenzor::Tensor& t) -> int64_t {
             if (t.numel() != 1) throw py::value_error("only one element tensors can be converted to Python scalars");
             switch (t.dtype()) {
                 case tenzor::DType::Int64: return t.item<int64_t>();
                 case tenzor::DType::Int32: return static_cast<int64_t>(t.item<int32_t>());
                 case tenzor::DType::Int16: return static_cast<int64_t>(t.item<int16_t>());
                 case tenzor::DType::Int8: return static_cast<int64_t>(t.item<int8_t>());
                 case tenzor::DType::UInt8: return static_cast<int64_t>(t.item<uint8_t>());
                 case tenzor::DType::Bool: return static_cast<int64_t>(t.item<bool>());
                 case tenzor::DType::Float64: return static_cast<int64_t>(t.item<double>());
                 case tenzor::DType::Float32: return static_cast<int64_t>(t.item<float>());
                 case tenzor::DType::Float16: return static_cast<int64_t>(static_cast<float>(t.item<tenzor::Float16>()));
                 case tenzor::DType::BFloat16: return static_cast<int64_t>(static_cast<float>(t.item<tenzor::BFloat16>()));
                 default: throw py::type_error("Cannot convert " + std::string(tenzor::dtype_name(t.dtype())) + " to int");
             }
             })
        .def("__index__", [](const tenzor::Tensor& t) -> int64_t {
             if (t.numel() != 1) {
                 throw py::value_error("only single-element tensors can be converted to Python scalars");
             }
             switch (t.dtype()) {
                 case tenzor::DType::Float64: return static_cast<int64_t>(t.item<double>());
                 case tenzor::DType::Float32: return static_cast<int64_t>(t.item<float>());
                 case tenzor::DType::Int64: return t.item<int64_t>();
                 case tenzor::DType::Int32: return static_cast<int64_t>(t.item<int32_t>());
                 case tenzor::DType::Int16: return static_cast<int64_t>(t.item<int16_t>());
                 case tenzor::DType::Int8: return static_cast<int64_t>(t.item<int8_t>());
                 case tenzor::DType::UInt8: return static_cast<int64_t>(t.item<uint8_t>());
                 case tenzor::DType::Bool: return static_cast<int64_t>(t.item<bool>());
                 default: return static_cast<int64_t>(t.item<float>());
             }
             })
        // Lazy iteration: __len__ + __getitem__ protocol.
        // Python calls __getitem__(0), (1), ... until IndexError — O(1) memory.
        .def("__len__", [](const tenzor::Tensor& t) -> int64_t {
             if (t.ndim() == 0) throw py::type_error("len() of a 0-d tensor");
             return t.shape()[0];
             })
        // Comparison operators
        .def("__eq__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::eq(a, b); },
             py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__ne__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::ne(a, b); },
             py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__lt__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::lt(a, b); },
             py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__le__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::le(a, b); },
             py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__gt__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::gt(a, b); },
             py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__ge__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::ge(a, b); },
             py::is_operator(), py::call_guard<py::gil_scoped_release>())
        // Scalar comparison operators
        .def("__eq__", [](const tenzor::Tensor& a, double b) {
            return tenzor::eq(a, tenzor::full(std::vector<int64_t>{}, b, a.dtype(), a.device()));
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__ne__", [](const tenzor::Tensor& a, double b) {
            return tenzor::ne(a, tenzor::full(std::vector<int64_t>{}, b, a.dtype(), a.device()));
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__lt__", [](const tenzor::Tensor& a, double b) {
            return tenzor::lt(a, tenzor::full(std::vector<int64_t>{}, b, a.dtype(), a.device()));
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__le__", [](const tenzor::Tensor& a, double b) {
            return tenzor::le(a, tenzor::full(std::vector<int64_t>{}, b, a.dtype(), a.device()));
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__gt__", [](const tenzor::Tensor& a, double b) {
            return tenzor::gt(a, tenzor::full(std::vector<int64_t>{}, b, a.dtype(), a.device()));
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__ge__", [](const tenzor::Tensor& a, double b) {
            return tenzor::ge(a, tenzor::full(std::vector<int64_t>{}, b, a.dtype(), a.device()));
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        // Container protocol
        .def("__len__", [](const tenzor::Tensor& t) -> int64_t {
             if (t.ndim() == 0) throw py::value_error("len() of a 0-d tensor");
             return t.shape()[0];
             })
        .def("__bool__", [](const tenzor::Tensor& t) -> bool {
             if (t.numel() != 1) throw py::value_error(
                 "The truth value of a Tensor with more than one element is ambiguous");
             switch (t.dtype()) {
                 case tenzor::DType::Float32: return t.item<float>() != 0.0f;
                 case tenzor::DType::Float64: return t.item<double>() != 0.0;
                 case tenzor::DType::Int32: return t.item<int32_t>() != 0;
                 case tenzor::DType::Int64: return t.item<int64_t>() != 0;
                 case tenzor::DType::Bool: return t.item<bool>();
                 default: return t.item<float>() != 0.0f;
             }
             })
        .def("__str__", [](const tenzor::Tensor& t) {
             // Delegate to __repr__
             return py::str(py::cast(t).attr("__repr__")());
             })
        // Math methods (GIL released for compute)
        .def("exp", [](const tenzor::Tensor& t) { return tenzor::exp(t); },
             "Element-wise exponential",
             py::call_guard<py::gil_scoped_release>())
        .def("log", [](const tenzor::Tensor& t) { return tenzor::log(t); },
             "Element-wise natural logarithm",
             py::call_guard<py::gil_scoped_release>())
        .def("sqrt", [](const tenzor::Tensor& t) { return tenzor::sqrt(t); },
             "Element-wise square root",
             py::call_guard<py::gil_scoped_release>())
        .def("sin", [](const tenzor::Tensor& t) { return tenzor::sin(t); },
             "Element-wise sine",
             py::call_guard<py::gil_scoped_release>())
        .def("cos", [](const tenzor::Tensor& t) { return tenzor::cos(t); },
             "Element-wise cosine",
             py::call_guard<py::gil_scoped_release>())
        .def("tan", [](const tenzor::Tensor& t) { return tenzor::tan(t); },
             "Element-wise tangent",
             py::call_guard<py::gil_scoped_release>())
        .def("abs", [](const tenzor::Tensor& t) { return tenzor::abs(t); },
             "Element-wise absolute value",
             py::call_guard<py::gil_scoped_release>())
        .def("pow", [](const tenzor::Tensor& t, float exponent) {
             return tenzor::pow(t, exponent);
             }, py::arg("exponent"), "Element-wise power",
             py::call_guard<py::gil_scoped_release>())
        // Reduction operations (as member methods calling free functions)
        .def("sum", [](const tenzor::Tensor& t) {
             return tenzor::sum(t, std::nullopt, false);
             }, "Sum all elements",
             py::call_guard<py::gil_scoped_release>())
        .def("sum", [](const tenzor::Tensor& t, int64_t dim, bool keepdim) {
             return tenzor::sum(t, std::make_optional(dim), keepdim);
             }, py::arg("dim"), py::arg("keepdim")=false,
             "Sum along dimension",
             py::call_guard<py::gil_scoped_release>())
        .def("mean", [](const tenzor::Tensor& t) {
             return tenzor::mean(t, std::nullopt, false);
             }, "Mean of all elements",
             py::call_guard<py::gil_scoped_release>())
        .def("mean", [](const tenzor::Tensor& t, int64_t dim, bool keepdim) {
             return tenzor::mean(t, std::make_optional(dim), keepdim);
             }, py::arg("dim"), py::arg("keepdim")=false,
             "Mean along dimension",
             py::call_guard<py::gil_scoped_release>())
        .def("max", [](const tenzor::Tensor& t) {
             return tenzor::max(t, std::nullopt, false);
             }, "Maximum of all elements",
             py::call_guard<py::gil_scoped_release>())
        .def("max", [](const tenzor::Tensor& t, int64_t dim, bool keepdim) {
             return tenzor::max(t, std::make_optional(dim), keepdim);
             }, py::arg("dim"), py::arg("keepdim")=false,
             "Maximum along dimension",
             py::call_guard<py::gil_scoped_release>())
        .def("min", [](const tenzor::Tensor& t) {
             return tenzor::min(t, std::nullopt, false);
             }, "Minimum of all elements",
             py::call_guard<py::gil_scoped_release>())
        .def("min", [](const tenzor::Tensor& t, int64_t dim, bool keepdim) {
             return tenzor::min(t, std::make_optional(dim), keepdim);
             }, py::arg("dim"), py::arg("keepdim")=false,
             "Minimum along dimension",
             py::call_guard<py::gil_scoped_release>())
        .def("any", [](const tenzor::Tensor& t) {
             return tenzor::any(t, std::nullopt, false);
             }, "True if any element is nonzero",
             py::call_guard<py::gil_scoped_release>())
        .def("any", [](const tenzor::Tensor& t, int64_t dim, bool keepdim) {
             return tenzor::any(t, std::make_optional(dim), keepdim);
             }, py::arg("dim"), py::arg("keepdim")=false,
             "Any nonzero along dimension",
             py::call_guard<py::gil_scoped_release>())
        .def("all", [](const tenzor::Tensor& t) {
             return tenzor::all(t, std::nullopt, false);
             }, "True if all elements are nonzero",
             py::call_guard<py::gil_scoped_release>())
        .def("all", [](const tenzor::Tensor& t, int64_t dim, bool keepdim) {
             return tenzor::all(t, std::make_optional(dim), keepdim);
             }, py::arg("dim"), py::arg("keepdim")=false,
             "All nonzero along dimension",
             py::call_guard<py::gil_scoped_release>())
        // Device transfer with overloads
        .def("cuda", [](const tenzor::Tensor& t, int32_t device_id) {
             return t.cuda(device_id);
             }, py::arg("device_id")=0, "Move tensor to CUDA device",
             py::call_guard<py::gil_scoped_release>())
        .def("cpu", [](const tenzor::Tensor& t) {
             return t.cpu();
             }, "Move tensor to CPU",
             py::call_guard<py::gil_scoped_release>())
        // DType conversion
        .def("to", py::overload_cast<tenzor::DType>(&tenzor::Tensor::to, py::const_),
             py::arg("dtype"), "Convert to different dtype",
             py::call_guard<py::gil_scoped_release>())
        // PyTorch-style dtype casting methods
        .def("float", [](const tenzor::Tensor& t) { return t.to(tenzor::DType::Float32); }, "Cast to float32",
             py::call_guard<py::gil_scoped_release>())
        .def("double", [](const tenzor::Tensor& t) { return t.to(tenzor::DType::Float64); }, "Cast to float64",
             py::call_guard<py::gil_scoped_release>())
        .def("half", [](const tenzor::Tensor& t) { return t.to(tenzor::DType::Float16); }, "Cast to float16",
             py::call_guard<py::gil_scoped_release>())
        .def("long", [](const tenzor::Tensor& t) { return t.to(tenzor::DType::Int64); }, "Cast to int64",
             py::call_guard<py::gil_scoped_release>())
        .def("int", [](const tenzor::Tensor& t) { return t.to(tenzor::DType::Int32); }, "Cast to int32",
             py::call_guard<py::gil_scoped_release>())
        // Properties
        .def("dim", &tenzor::Tensor::ndim, "Number of dimensions")
        .def("size", [](const tenzor::Tensor& t) -> py::tuple {
             py::tuple result(t.ndim());
             for (int64_t i = 0; i < t.ndim(); ++i) result[i] = t.shape()[i];
             return result;
             }, "Return shape as tuple")
        .def("size", [](const tenzor::Tensor& t, int64_t dim) -> int64_t {
             if (dim < 0) dim += t.ndim();
             return t.shape()[dim];
             }, py::arg("dim"), "Return size of dimension")
        .def_property_readonly("strides", [](const tenzor::Tensor& t) -> py::tuple {
             py::tuple result(t.ndim());
             auto strides = t.strides();
             for (int64_t i = 0; i < t.ndim(); ++i) result[i] = strides[i];
             return result;
             }, "Tensor strides")
        .def("__repr__", [](const tenzor::Tensor& t) {
            if (!t.is_valid()) {
                return std::string("tensor(<uninitialized>)");
            }
            std::ostringstream ss;
            ss << std::setprecision(4);
            tenzor::Tensor cpu_t = (t.device().type != tenzor::Device::Type::CPU) ? t.cpu() : t;
            tenzor::Tensor cont = cpu_t.is_contiguous() ? cpu_t : cpu_t.contiguous();
            int64_t numel = cont.numel();
            auto shape = cont.shape();
            int64_t ndim = cont.ndim();

            // Format a single element value
            auto fmt_elem = [&](std::ostringstream& os, int64_t flat_idx) {
                if (cont.dtype() == tenzor::DType::Float32) {
                    os << cont.data<float>()[flat_idx];
                } else if (cont.dtype() == tenzor::DType::Float64) {
                    os << cont.data<double>()[flat_idx];
                } else if (cont.dtype() == tenzor::DType::Int32) {
                    os << cont.data<int32_t>()[flat_idx];
                } else if (cont.dtype() == tenzor::DType::Int64) {
                    os << cont.data<int64_t>()[flat_idx];
                } else if (cont.dtype() == tenzor::DType::Bool) {
                    os << (cont.data<bool>()[flat_idx] ? "True" : "False");
                } else {
                    os << "?";
                }
            };

            // Recursive printer for nested tensor structure
            // max_elems_per_dim: show at most this many elements per dimension (3 at start + 3 at end for large)
            constexpr int64_t EDGE_ITEMS = 3;
            constexpr int64_t SUMMARIZE_THRESHOLD = 1000;
            bool summarize = numel > SUMMARIZE_THRESHOLD;

            // Compute strides for flat index calculation in contiguous layout
            std::vector<int64_t> dim_strides(ndim, 1);
            for (int64_t d = ndim - 2; d >= 0; --d) {
                dim_strides[d] = dim_strides[d + 1] * shape[d + 1];
            }

            std::function<void(int64_t /*dim*/, int64_t /*offset*/, int /*indent*/)> print_dim;
            print_dim = [&](int64_t dim, int64_t offset, int indent) {
                int64_t size = shape[dim];
                ss << "[";
                if (dim == ndim - 1) {
                    // Innermost dimension: print values
                    if (!summarize || size <= 2 * EDGE_ITEMS) {
                        for (int64_t i = 0; i < size; ++i) {
                            if (i > 0) ss << ", ";
                            fmt_elem(ss, offset + i);
                        }
                    } else {
                        for (int64_t i = 0; i < EDGE_ITEMS; ++i) {
                            if (i > 0) ss << ", ";
                            fmt_elem(ss, offset + i);
                        }
                        ss << ", ..., ";
                        for (int64_t i = size - EDGE_ITEMS; i < size; ++i) {
                            if (i > size - EDGE_ITEMS) ss << ", ";
                            fmt_elem(ss, offset + i);
                        }
                    }
                } else {
                    // Non-innermost: recurse with nested brackets
                    std::string pad(indent + 1, ' ');
                    bool elide = summarize && size > 2 * EDGE_ITEMS;
                    int64_t stride = dim_strides[dim];
                    auto print_sub = [&](int64_t i) {
                        if (i > 0) {
                            ss << ",";
                            // Add newlines between rows for 2D+
                            int newlines = ndim - dim - 1;
                            for (int nl = 0; nl < newlines; ++nl) ss << "\n";
                            ss << pad;
                        }
                        print_dim(dim + 1, offset + i * stride, indent + 1);
                    };
                    if (!elide) {
                        for (int64_t i = 0; i < size; ++i) print_sub(i);
                    } else {
                        for (int64_t i = 0; i < EDGE_ITEMS; ++i) print_sub(i);
                        ss << ",\n" << pad << "...";
                        for (int64_t i = size - EDGE_ITEMS; i < size; ++i) print_sub(i);
                    }
                }
                ss << "]";
            };

            ss << "tensor(";
            if (ndim == 0) {
                // Scalar tensor
                fmt_elem(ss, 0);
            } else if (numel == 0) {
                // Empty tensor
                ss << "[]";
            } else {
                print_dim(0, 0, 7); // 7 = len("tensor(")
            }

            // dtype annotation (skip float32 like PyTorch)
            if (t.dtype() != tenzor::DType::Float32) {
                ss << ", dtype=" << tenzor::dtype_name(t.dtype());
            }
            if (t.device().type != tenzor::Device::Type::CPU) {
                ss << ", device=" << t.device().to_string();
            }
            ss << ")";
            return ss.str();
        })
        // Pickle support for model saving/loading
        .def(py::pickle(
            // __getstate__: serialize to (shape, dtype_int, device_str, bytes)
            [](const tenzor::Tensor& t) {
                // Move to CPU and make contiguous for serialization
                tenzor::Tensor cpu_t = (t.device().type != tenzor::Device::Type::CPU)
                    ? t.to(tenzor::Device::cpu()) : t;
                if (!cpu_t.is_contiguous()) cpu_t = cpu_t.contiguous();

                auto shape = cpu_t.shape();
                std::vector<int64_t> shape_vec(shape.begin(), shape.end());

                // Serialize raw bytes
                size_t nbytes = static_cast<size_t>(cpu_t.numel()) * tenzor::dtype_size(cpu_t.dtype());
                py::bytes data(reinterpret_cast<const char*>(cpu_t.data_ptr()), nbytes);

                return py::make_tuple(
                    shape_vec,
                    static_cast<int>(cpu_t.dtype()),
                    t.device().to_string(),
                    data
                );
            },
            // __setstate__: deserialize from (shape, dtype_int, device_str, bytes)
            [](py::tuple state) {
                if (state.size() != 4) throw py::value_error("Invalid pickle state");

                auto shape = state[0].cast<std::vector<int64_t>>();
                auto dtype_int = state[1].cast<int>();
                if (dtype_int < 0 || dtype_int > static_cast<int>(tenzor::DType::Complex128))
                    throw py::value_error("Invalid dtype in pickle state: " + std::to_string(dtype_int));
                for (auto d : shape)
                    if (d < 0) throw py::value_error("Negative dimension in pickle state");
                auto dtype = static_cast<tenzor::DType>(dtype_int);
                auto device_str = state[2].cast<std::string>();
                auto data = state[3].cast<std::string>();

                // Create tensor on CPU
                tenzor::Tensor t(shape, dtype, tenzor::Device::cpu());
                size_t nbytes = static_cast<size_t>(t.numel()) * tenzor::dtype_size(dtype);
                if (data.size() != nbytes) {
                    throw py::value_error("Pickle data size mismatch");
                }
                std::memcpy(t.data_ptr(), data.data(), nbytes);

                // Move to original device if needed
                auto target_device = tenzor::Device::from_string(device_str);
                if (target_device.type != tenzor::Device::Type::CPU) {
                    t = t.to(target_device);
                }
                return t;
            }
        ))
        // Python-style indexing
        .def("__getitem__", [](const tenzor::Tensor& self, py::object key) -> tenzor::Tensor {
            // Phase A (GIL held): Parse Python key into C++ types
            enum class IndexKind { Int, Slice, Tuple, TensorMask, FancyIndex };
            IndexKind kind;
            int64_t int_idx = 0;
            int64_t slice_start = 0, slice_stop = 0, slice_step = 1;

            struct TupleEntry {
                bool is_int;
                int64_t int_val;
                int64_t start, stop, step;
            };
            std::vector<TupleEntry> tuple_entries;
            tenzor::Tensor mask_tensor;

            // For fancy indexing: vector of optional<Tensor> index tensors
            std::vector<std::optional<tenzor::Tensor>> fancy_indices;

            // Helper: convert a py::list of ints to an Int64 Tensor on self's device
            auto list_to_index_tensor = [&](py::list lst) -> tenzor::Tensor {
                std::vector<int64_t> vals;
                vals.reserve(lst.size());
                for (auto& item : lst) {
                    vals.push_back(py::cast<int64_t>(item));
                }
                auto t = tenzor::from_data(vals.data(),
                                           {static_cast<int64_t>(vals.size())},
                                           tenzor::Device::cpu());
                if (self.device() != tenzor::Device::cpu()) {
                    t = t.to(self.device());
                }
                return t;
            };

            // Helper: check if a py::object is a fancy-index element (list or int Tensor)
            auto is_fancy_element = [](py::object obj) -> bool {
                if (py::isinstance<py::list>(obj)) return true;
                if (py::isinstance<tenzor::Tensor>(obj)) {
                    auto t = obj.cast<tenzor::Tensor>();
                    return t.dtype() == tenzor::DType::Int32 ||
                           t.dtype() == tenzor::DType::Int64;
                }
                return false;
            };

            if (py::isinstance<py::int_>(key)) {
                kind = IndexKind::Int;
                int_idx = py::cast<int64_t>(key);
            } else if (py::isinstance<py::slice>(key)) {
                kind = IndexKind::Slice;
                py::slice slice_obj = py::cast<py::slice>(key);
                py::ssize_t start, stop, step, length;
                auto shape = self.shape();
                if (shape.empty()) {
                    throw std::runtime_error("Cannot slice scalar tensor");
                }
                if (!slice_obj.compute(shape[0], &start, &stop, &step, &length)) {
                    throw std::runtime_error(
                        "Invalid slice for dimension 0 with size " + std::to_string(shape[0]));
                }
                slice_start = start; slice_stop = stop; slice_step = step;
            } else if (py::isinstance<py::list>(key)) {
                // Single list of ints -> fancy index on dim 0
                kind = IndexKind::FancyIndex;
                fancy_indices.push_back(list_to_index_tensor(py::cast<py::list>(key)));
            } else if (py::isinstance<py::tuple>(key)) {
                py::tuple indices = py::cast<py::tuple>(key);

                // Check if any tuple element triggers fancy indexing
                bool has_fancy = false;
                for (size_t i = 0; i < indices.size(); ++i) {
                    if (is_fancy_element(indices[i])) {
                        has_fancy = true;
                        break;
                    }
                }

                if (has_fancy) {
                    kind = IndexKind::FancyIndex;
                    fancy_indices.reserve(indices.size());
                    for (size_t i = 0; i < indices.size(); ++i) {
                        if (py::isinstance<py::list>(indices[i])) {
                            fancy_indices.push_back(list_to_index_tensor(py::cast<py::list>(indices[i])));
                        } else if (py::isinstance<tenzor::Tensor>(indices[i])) {
                            auto t = indices[i].cast<tenzor::Tensor>();
                            if (t.dtype() == tenzor::DType::Int32 || t.dtype() == tenzor::DType::Int64) {
                                fancy_indices.push_back(t);
                            } else {
                                throw std::runtime_error(
                                    "Tensor index must be integer dtype (Int32/Int64), not boolean or float");
                            }
                        } else if (py::isinstance<py::slice>(indices[i])) {
                            // Slice in a fancy-index context => nullopt (full dim passthrough)
                            fancy_indices.push_back(std::nullopt);
                        } else if (py::isinstance<py::int_>(indices[i])) {
                            // Scalar int in fancy context -> 0-d index tensor
                            int64_t val = py::cast<int64_t>(indices[i]);
                            auto t = tenzor::from_data(&val, {}, tenzor::Device::cpu());
                            if (self.device() != tenzor::Device::cpu()) {
                                t = t.to(self.device());
                            }
                            fancy_indices.push_back(t);
                        } else {
                            throw std::runtime_error("Unsupported index type in tuple");
                        }
                    }
                } else {
                    kind = IndexKind::Tuple;
                    // Pre-parse all tuple entries: need current shape to resolve slices
                    // We parse slices lazily during Phase B since shape changes with each op
                    tuple_entries.reserve(indices.size());
                    for (size_t i = 0; i < indices.size(); ++i) {
                        TupleEntry entry{};
                        if (py::isinstance<py::int_>(indices[i])) {
                            entry.is_int = true;
                            entry.int_val = py::cast<int64_t>(indices[i]);
                        } else if (py::isinstance<py::slice>(indices[i])) {
                            entry.is_int = false;
                            // Extract raw start/stop/step from Python slice object.
                            // None values get sentinel min/max — resolved against dim size in Phase B.
                            py::slice slice_obj = py::cast<py::slice>(indices[i]);
                            auto s_start = slice_obj.attr("start");
                            auto s_stop = slice_obj.attr("stop");
                            auto s_step = slice_obj.attr("step");
                            entry.start = s_start.is_none() ? std::numeric_limits<int64_t>::min() : py::cast<int64_t>(s_start);
                            entry.stop = s_stop.is_none() ? std::numeric_limits<int64_t>::max() : py::cast<int64_t>(s_stop);
                            entry.step = s_step.is_none() ? 1 : py::cast<int64_t>(s_step);
                        } else {
                            throw std::runtime_error("Unsupported index type in tuple");
                        }
                        tuple_entries.push_back(entry);
                    }
                }
            } else if (py::isinstance<tenzor::Tensor>(key)) {
                auto t = key.cast<tenzor::Tensor>();
                if (t.dtype() == tenzor::DType::Int32 || t.dtype() == tenzor::DType::Int64) {
                    // Integer tensor -> fancy index on dim 0
                    kind = IndexKind::FancyIndex;
                    fancy_indices.push_back(t);
                } else {
                    kind = IndexKind::TensorMask;
                    mask_tensor = t;
                }
            } else {
                throw std::runtime_error("Unsupported index type: expected int, slice, list, or Tensor");
            }

            // Phase B (GIL released): Perform tensor operations
            py::gil_scoped_release release;

            switch (kind) {
                case IndexKind::Int: {
                    auto shape = self.shape();
                    if (shape.empty()) {
                        throw std::runtime_error("Cannot index scalar tensor");
                    }
                    int64_t idx = int_idx;
                    if (idx < 0) idx += shape[0];
                    if (idx < 0 || idx >= shape[0]) {
                        throw std::out_of_range(
                            "Index " + std::to_string(int_idx) + " out of range for dimension 0 with size " + std::to_string(shape[0]));
                    }
                    auto sliced = self.slice(0, idx, idx + 1);
                    auto sliced_shape = sliced.shape();
                    if (!sliced_shape.empty() && sliced_shape[0] == 1) {
                        return sliced.squeeze(0);
                    }
                    return sliced;
                }
                case IndexKind::Slice: {
                    return self.slice(0, slice_start, slice_stop, slice_step);
                }
                case IndexKind::Tuple: {
                    tenzor::Tensor result = self;
                    int squeeze_count = 0;
                    for (size_t i = 0; i < tuple_entries.size(); ++i) {
                        auto& entry = tuple_entries[i];
                        if (entry.is_int) {
                            int64_t idx = entry.int_val;
                            auto shape = result.shape();
                            size_t dim = i - squeeze_count;
                            if (dim >= shape.size()) {
                                throw std::out_of_range("Too many indices");
                            }
                            if (idx < 0) idx += shape[dim];
                            result = result.slice(dim, idx, idx + 1);
                            auto new_shape = result.shape();
                            if (dim < new_shape.size() && new_shape[dim] == 1) {
                                result = result.squeeze(dim);
                                squeeze_count++;
                            }
                        } else {
                            auto shape = result.shape();
                            size_t dim = i - squeeze_count;
                            if (dim >= shape.size()) {
                                throw std::out_of_range("Too many indices");
                            }
                            // Resolve slice start/stop against actual dim size
                            int64_t dim_size = shape[dim];
                            int64_t start = entry.start, stop = entry.stop, step = entry.step;
                            if (start == std::numeric_limits<int64_t>::min()) start = (step > 0) ? 0 : dim_size - 1;
                            else if (start < 0) start += dim_size;
                            if (stop == std::numeric_limits<int64_t>::max()) stop = (step > 0) ? dim_size : -1;
                            else if (stop < 0) stop += dim_size;
                            start = std::clamp(start, int64_t(0), dim_size);
                            stop = std::clamp(stop, int64_t(0), dim_size);
                            result = result.slice(dim, start, stop, step);
                        }
                    }
                    return result;
                }
                case IndexKind::TensorMask: {
                    if (mask_tensor.dtype() == tenzor::DType::Bool) {
                        return tenzor::masked_select(self, mask_tensor);
                    }
                    throw std::runtime_error("Unsupported index type: expected bool Tensor");
                }
                case IndexKind::FancyIndex: {
                    return tenzor::index(self, fancy_indices);
                }
            }
            throw std::runtime_error("Unreachable");  // silence compiler warning
        }, py::arg("key"), "Get tensor slice or element")
        .def("__setitem__", [](tenzor::Tensor& self, py::object key, py::object value) {
            // Phase A (GIL held): Parse Python key and value into C++ types
            enum class SetIndexKind { Int, Slice, Tuple };
            SetIndexKind kind;
            int64_t int_idx = 0;
            int64_t slice_start = 0, slice_stop = 0;

            struct SetTupleEntry {
                bool is_int;
                bool is_ellipsis;
                int64_t int_val;
                int64_t start, stop, step;
            };
            std::vector<SetTupleEntry> tuple_entries;

            // Parse key
            if (py::isinstance<py::int_>(key)) {
                kind = SetIndexKind::Int;
                int_idx = py::cast<int64_t>(key);
            } else if (py::isinstance<py::slice>(key)) {
                kind = SetIndexKind::Slice;
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
                slice_start = start; slice_stop = stop;
            } else if (py::isinstance<py::tuple>(key)) {
                kind = SetIndexKind::Tuple;
                py::tuple indices = py::cast<py::tuple>(key);
                tuple_entries.reserve(indices.size());
                for (size_t i = 0; i < indices.size(); ++i) {
                    SetTupleEntry entry{};
                    entry.is_ellipsis = false;
                    if (py::isinstance<py::int_>(indices[i])) {
                        entry.is_int = true;
                        entry.int_val = py::cast<int64_t>(indices[i]);
                    } else if (py::isinstance<py::slice>(indices[i])) {
                        entry.is_int = false;
                        py::slice slice_obj = py::cast<py::slice>(indices[i]);
                        auto s_start = slice_obj.attr("start");
                        auto s_stop = slice_obj.attr("stop");
                        auto s_step = slice_obj.attr("step");
                        entry.start = s_start.is_none() ? std::numeric_limits<int64_t>::min() : py::cast<int64_t>(s_start);
                        entry.stop = s_stop.is_none() ? std::numeric_limits<int64_t>::max() : py::cast<int64_t>(s_stop);
                        entry.step = s_step.is_none() ? 1 : py::cast<int64_t>(s_step);
                    } else if (py::isinstance<py::ellipsis>(indices[i])) {
                        entry.is_ellipsis = true;
                    } else {
                        throw std::runtime_error("Unsupported index type in tuple");
                    }
                    tuple_entries.push_back(entry);
                }
            } else {
                throw std::runtime_error("Unsupported index type for assignment");
            }

            // Parse value under GIL — use int64 for integer dtypes to avoid double truncation
            tenzor::Tensor value_tensor;
            bool is_scalar_value = false;
            double scalar_value = 0.0;
            int64_t int_scalar_value = 0;
            bool is_integer_scalar = false;
            if (py::isinstance<tenzor::Tensor>(value)) {
                value_tensor = py::cast<tenzor::Tensor>(value);
            } else if (py::isinstance<py::bool_>(value)) {
                is_scalar_value = true;
                is_integer_scalar = true;
                int_scalar_value = py::cast<bool>(value) ? 1 : 0;
                scalar_value = static_cast<double>(int_scalar_value);
            } else if (py::isinstance<py::int_>(value)) {
                is_scalar_value = true;
                is_integer_scalar = true;
                int_scalar_value = py::cast<int64_t>(value);
                scalar_value = static_cast<double>(int_scalar_value);
            } else if (py::isinstance<py::float_>(value)) {
                is_scalar_value = true;
                scalar_value = py::cast<double>(value);
            } else {
                throw std::runtime_error("Value must be a Tensor or scalar");
            }

            // Phase B (GIL released): Perform tensor operations
            py::gil_scoped_release release;

            // Helper to create scalar tensor
            auto make_scalar_tensor = [&]() -> tenzor::Tensor {
                auto t = tenzor::empty({1}, self.dtype(), self.device());
                switch (self.dtype()) {
                    case tenzor::DType::Float32:
                        *t.data<float>() = static_cast<float>(scalar_value);
                        break;
                    case tenzor::DType::Float64:
                        *t.data<double>() = scalar_value;
                        break;
                    case tenzor::DType::Int32:
                        *t.data<int32_t>() = is_integer_scalar
                            ? static_cast<int32_t>(int_scalar_value)
                            : static_cast<int32_t>(scalar_value);
                        break;
                    case tenzor::DType::Int64:
                        *t.data<int64_t>() = is_integer_scalar
                            ? int_scalar_value
                            : static_cast<int64_t>(scalar_value);
                        break;
                    case tenzor::DType::UInt8:
                        *t.data<uint8_t>() = is_integer_scalar
                            ? static_cast<uint8_t>(int_scalar_value)
                            : static_cast<uint8_t>(scalar_value);
                        break;
                    case tenzor::DType::Bool:
                        *t.data<bool>() = is_integer_scalar
                            ? (int_scalar_value != 0)
                            : (scalar_value != 0.0);
                        break;
                    default:
                        throw std::runtime_error("Unsupported dtype for scalar assignment");
                }
                return t;
            };

            auto val = is_scalar_value ? make_scalar_tensor() : value_tensor;

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
                    double scalar_value;
                    switch (src.dtype()) {
                        case tenzor::DType::Float32:
                            scalar_value = static_cast<double>(*src_cpu.data<float>());
                            break;
                        case tenzor::DType::Float64:
                            scalar_value = *src_cpu.data<double>();
                            break;
                        case tenzor::DType::Int32:
                            scalar_value = static_cast<double>(*src_cpu.data<int32_t>());
                            break;
                        case tenzor::DType::Int64:
                            scalar_value = static_cast<double>(*src_cpu.data<int64_t>());
                            break;
                        default:
                            throw std::runtime_error("Unsupported dtype for scalar broadcast in __setitem__");
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
                            // Make both tensors contiguous, then copy
                            auto dst_cont = dst.contiguous();
                            auto src_cont = src.contiguous();

                            size_t bytes = dst_cont.numel() * dst_cont.dtype_size();
                            if (dst_cont.device().type == tenzor::Device::Type::CPU &&
                                src_cont.device().type == tenzor::Device::Type::CPU) {
                                std::memcpy(dst_cont.data_ptr(), src_cont.data_ptr(), bytes);
                            } else {
                                // Use backend copy for device tensors
                                auto* backend = tenzor::backend_registry().get_backend(dst_cont.device().type);
                                if (backend) {
                                    backend->copy(dst_cont.data_ptr(), src_cont.data_ptr(), bytes,
                                                tenzor::CopyKind::DeviceToDevice);
                                }
                            }

                            // Copy back to original destination if it was non-contiguous
                            if (!dst.is_contiguous()) {
                                // Element-wise copy from contiguous to non-contiguous
                                auto dst_shape_vec = dst.shape();
                                std::vector<int64_t> indices(dst_shape_vec.size(), 0);
                                size_t total_elements = dst.numel();

                                for (size_t i = 0; i < total_elements; ++i) {
                                    // Calculate linear index in contiguous tensor
                                    size_t linear_idx = i;

                                    // Calculate multi-dimensional index
                                    size_t temp = linear_idx;
                                    for (int64_t dim = static_cast<int64_t>(dst_shape_vec.size()) - 1; dim >= 0; --dim) {
                                        indices[dim] = temp % dst_shape_vec[dim];
                                        temp /= dst_shape_vec[dim];
                                    }

                                    // Calculate offset in non-contiguous tensor using strides
                                    auto strides = dst.strides();
                                    size_t offset = 0;
                                    for (size_t dim = 0; dim < indices.size(); ++dim) {
                                        offset += indices[dim] * strides[dim];
                                    }

                                    // Copy single element based on dtype
                                    void* dst_ptr = static_cast<char*>(dst.data_ptr()) + offset * dst.dtype_size();
                                    void* src_ptr = static_cast<char*>(dst_cont.data_ptr()) + i * dst_cont.dtype_size();
                                    std::memcpy(dst_ptr, src_ptr, dst.dtype_size());
                                }
                            }
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
                    // Include actual shapes in error for easier debugging
                    auto fmt_shape = [](std::span<const int64_t> s) {
                        std::string r = "[";
                        for (size_t i = 0; i < s.size(); ++i) {
                            if (i > 0) r += ", ";
                            r += std::to_string(s[i]);
                        }
                        return r + "]";
                    };
                    throw std::runtime_error(
                        "Shape mismatch: cannot broadcast source shape " +
                        fmt_shape(src_shape) + " to destination shape " +
                        fmt_shape(dst_shape));
                }

                // Use expand() + contiguous copy instead of element-by-element loop
                // expand() creates a view with broadcast strides (no data copy),
                // then .contiguous() materializes the expanded data in one backend call.
                auto dst_shape_vec = std::vector<int64_t>(dst_shape.begin(), dst_shape.end());
                auto expanded = tenzor::expand(src, dst_shape_vec).contiguous();

                // Copy expanded data into destination
                if (dst.is_contiguous()) {
                    size_t bytes = dst.numel() * dst.dtype_size();
                    if (dst.device().type == tenzor::Device::Type::CPU) {
                        std::memcpy(dst.data_ptr(), expanded.data_ptr(), bytes);
                    } else {
                        auto* backend = tenzor::backend_registry().get_backend(dst.device().type);
                        if (backend) {
                            backend->copy(dst.data_ptr(), expanded.data_ptr(), bytes,
                                        tenzor::CopyKind::DeviceToDevice);
                        }
                    }
                } else {
                    // Non-contiguous destination: element-wise copy using strides
                    auto dst_strides = dst.strides();
                    size_t total_elements = dst.numel();
                    std::vector<int64_t> indices(dst_shape_vec.size(), 0);
                    for (size_t i = 0; i < total_elements; ++i) {
                        size_t temp = i;
                        for (int64_t dim = static_cast<int64_t>(dst_shape_vec.size()) - 1; dim >= 0; --dim) {
                            indices[dim] = temp % dst_shape_vec[dim];
                            temp /= dst_shape_vec[dim];
                        }
                        size_t offset = 0;
                        for (size_t dim = 0; dim < indices.size(); ++dim) {
                            offset += indices[dim] * dst_strides[dim];
                        }
                        void* dst_ptr = static_cast<char*>(dst.data_ptr()) + offset * dst.dtype_size();
                        void* src_ptr = static_cast<char*>(expanded.data_ptr()) + i * expanded.dtype_size();
                        std::memcpy(dst_ptr, src_ptr, dst.dtype_size());
                    }
                }
            };

            // Compute target tensor view based on pre-parsed key
            tenzor::Tensor target;
            switch (kind) {
                case SetIndexKind::Int: {
                    auto shape = self.shape();
                    if (shape.empty()) {
                        throw std::runtime_error("Cannot index scalar tensor");
                    }
                    int64_t idx = int_idx;
                    if (idx < 0) idx += shape[0];
                    if (idx < 0 || idx >= shape[0]) {
                        throw std::out_of_range("Index out of range");
                    }
                    auto sliced = self.slice(0, idx, idx + 1);
                    target = sliced.squeeze(0);
                    break;
                }
                case SetIndexKind::Slice: {
                    target = self.slice(0, slice_start, slice_stop);
                    break;
                }
                case SetIndexKind::Tuple: {
                    target = self;
                    int squeeze_count = 0;
                    std::vector<int64_t> squeeze_dims;
                    for (size_t i = 0; i < tuple_entries.size(); ++i) {
                        size_t adjusted_dim = i - squeeze_count;
                        auto target_shape = target.shape();
                        if (adjusted_dim >= target_shape.size()) {
                            throw std::out_of_range("Too many indices");
                        }
                        auto& entry = tuple_entries[i];
                        if (entry.is_ellipsis) {
                            int64_t remaining_indices = static_cast<int64_t>(tuple_entries.size()) - static_cast<int64_t>(i) - 1;
                            int64_t remaining_dims = static_cast<int64_t>(target_shape.size()) - static_cast<int64_t>(adjusted_dim);
                            int64_t dims_to_skip = remaining_dims - remaining_indices;
                            if (dims_to_skip < 0) {
                                throw std::runtime_error("Invalid ellipsis: too many indices");
                            }
                            squeeze_count += dims_to_skip;
                        } else if (entry.is_int) {
                            int64_t idx = entry.int_val;
                            if (idx < 0) idx += target_shape[adjusted_dim];
                            if (idx < 0 || idx >= target_shape[adjusted_dim]) {
                                throw std::out_of_range("Index out of range");
                            }
                            target = target.slice(adjusted_dim, idx, idx + 1);
                            squeeze_dims.push_back(adjusted_dim);
                            squeeze_count++;
                        } else {
                            // Slice entry — resolve against current dim size
                            int64_t dim_size = target_shape[adjusted_dim];
                            int64_t start = entry.start, stop = entry.stop;
                            if (start == std::numeric_limits<int64_t>::min()) start = 0;
                            else if (start < 0) start += dim_size;
                            if (stop == std::numeric_limits<int64_t>::max()) stop = dim_size;
                            else if (stop < 0) stop += dim_size;
                            start = std::clamp(start, int64_t(0), dim_size);
                            stop = std::clamp(stop, int64_t(0), dim_size);
                            if (entry.step != 1) {
                                throw std::runtime_error("Slice step not supported yet for assignment");
                            }
                            target = target.slice(adjusted_dim, start, stop);
                        }
                    }
                    // Squeeze indexed dimensions (from back to front)
                    for (auto it = squeeze_dims.rbegin(); it != squeeze_dims.rend(); ++it) {
                        int64_t dim = *it;
                        for (auto prev_it = it + 1; prev_it != squeeze_dims.rend(); ++prev_it) {
                            if (*prev_it < dim) dim--;
                        }
                        if (dim >= 0 && dim < target.ndim()) {
                            auto ts = target.shape();
                            if (ts[dim] == 1) target = target.squeeze(dim);
                        }
                    }
                    break;
                }
            }

            // Copy value to target with broadcasting
            copy_with_broadcast(target, val);
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

    m.def("randint", &tenzor::randint, "Create tensor with random integers",
         py::arg("low"),
         py::arg("high"),
         py::arg("shape"),
         py::arg("dtype") = tenzor::DType::Int64,
         py::arg("device") = tenzor::Device::cpu());

    m.def("arange", &tenzor::arange, "Create 1D tensor with evenly spaced values",
         py::arg("start"),
         py::arg("end"),
         py::arg("step") = 1.0f,
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    // Missing creation ops
    m.def("rand", [](std::vector<int64_t> shape, tenzor::DType dtype, tenzor::Device device) {
         return tenzor::rand(std::move(shape), dtype, device);
         }, "Create tensor with uniform random values in [0, 1)",
         py::arg("shape"),
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("linspace", [](float start, float end, int64_t steps, tenzor::DType dtype, tenzor::Device device) {
         return tenzor::linspace(start, end, steps, dtype, device);
         }, "Create 1D tensor with linearly spaced values",
         py::arg("start"), py::arg("end"), py::arg("steps"),
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("full", [](std::vector<int64_t> shape, float value, tenzor::DType dtype, tenzor::Device device) {
         return tenzor::full(std::move(shape), value, dtype, device);
         }, "Create tensor filled with a scalar value",
         py::arg("shape"), py::arg("value"),
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("empty", [](std::vector<int64_t> shape, tenzor::DType dtype, tenzor::Device device) {
         return tenzor::empty(std::move(shape), dtype, device);
         }, "Create uninitialized tensor",
         py::arg("shape"),
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("eye", [](int64_t n, std::optional<int64_t> m, tenzor::DType dtype, tenzor::Device device) {
         return tenzor::eye(n, m, dtype, device);
         }, "Create identity matrix",
         py::arg("n"),
         py::arg("m") = py::none(),
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("randperm", &tenzor::randperm, "Create random permutation of integers [0, n)",
          py::arg("n"), py::arg("device") = tenzor::Device::cpu());

    // tensor() - create tensor from Python data (lists, nested lists, scalars)
    m.def("tensor", [](py::object data, std::optional<tenzor::DType> dtype, tenzor::Device device) -> tenzor::Tensor {
        // Helper: recursively determine shape and flatten data
        std::function<void(py::handle, std::vector<int64_t>&, int)> get_shape;
        get_shape = [&](py::handle obj, std::vector<int64_t>& shape, int depth) {
            if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
                auto seq = py::cast<py::sequence>(obj);
                if (static_cast<int>(shape.size()) <= depth)
                    shape.push_back(static_cast<int64_t>(py::len(seq)));
                if (py::len(seq) > 0)
                    get_shape(seq[0], shape, depth + 1);
            }
        };

        std::vector<int64_t> shape;
        get_shape(data, shape, 0);

        // Flatten all values
        std::vector<double> values;
        std::function<void(py::handle)> flatten;
        flatten = [&](py::handle obj) {
            if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
                for (auto item : py::cast<py::sequence>(obj))
                    flatten(item);
            } else {
                values.push_back(py::cast<double>(obj));
            }
        };

        if (py::isinstance<py::list>(data) || py::isinstance<py::tuple>(data)) {
            flatten(data);
        } else {
            // Scalar
            values.push_back(py::cast<double>(data));
        }

        auto actual_dtype = dtype.value_or(tenzor::DType::Float32);

        if (shape.empty()) {
            // Scalar tensor
            auto t = tenzor::full({}, values[0], actual_dtype, device);
            return t;
        }

        // Create tensor and fill
        auto t = tenzor::empty(shape, actual_dtype, device);
        if (actual_dtype == tenzor::DType::Float64) {
            auto* ptr = t.data<double>();
            for (size_t i = 0; i < values.size(); ++i)
                ptr[i] = values[i];
        } else if (actual_dtype == tenzor::DType::Float32) {
            auto* ptr = t.data<float>();
            for (size_t i = 0; i < values.size(); ++i)
                ptr[i] = static_cast<float>(values[i]);
        } else if (actual_dtype == tenzor::DType::Int64) {
            auto* ptr = t.data<int64_t>();
            for (size_t i = 0; i < values.size(); ++i)
                ptr[i] = static_cast<int64_t>(values[i]);
        } else if (actual_dtype == tenzor::DType::Int32) {
            auto* ptr = t.data<int32_t>();
            for (size_t i = 0; i < values.size(); ++i)
                ptr[i] = static_cast<int32_t>(values[i]);
        } else {
            // Fallback: fill as float, then cast
            auto ft = tenzor::empty(shape, tenzor::DType::Float32, device);
            auto* ptr = ft.data<float>();
            for (size_t i = 0; i < values.size(); ++i)
                ptr[i] = static_cast<float>(values[i]);
            t = ft.to(actual_dtype);
        }
        return t;
    }, "Create a tensor from Python data (lists, nested lists, or scalar)",
    py::arg("data"),
    py::arg("dtype") = py::none(),
    py::arg("device") = tenzor::Device::cpu());

    // manual_seed
    m.def("manual_seed", &tenzor::manual_seed,
          "Set the random seed for reproducibility",
          py::arg("seed"));

    // save/load top-level functions (like torch.save / torch.load)
    m.def("save", [](py::object obj, const std::string& path) {
         // Accept either a state_dict (dict) or a Module
         if (py::isinstance<py::dict>(obj)) {
             auto dict = py::cast<py::dict>(obj);
             std::unordered_map<std::string, tenzor::Tensor> state;
             for (auto& [key, val] : dict) {
                 state[py::cast<std::string>(key)] = py::cast<tenzor::Tensor>(val);
             }
             tenzor::nn::Serializer::save(state, path);
         } else if (py::isinstance<tenzor::nn::Module>(obj)) {
             auto& module = py::cast<tenzor::nn::Module&>(obj);
             module.save(path);
         } else {
             throw py::type_error("save() expects a state_dict (dict) or nn.Module");
         }
         }, "Save a state_dict or module to file",
         py::arg("obj"), py::arg("path"));

    m.def("load", [](const std::string& path) {
         return tenzor::nn::Serializer::load(path);
         }, "Load a state_dict from file",
         py::arg("path"));

    // Missing math ops (free functions) - GIL released for compute-heavy ops
    m.def("add", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
         return tenzor::add(a, b);
         }, "Element-wise addition", py::arg("a"), py::arg("b"),
         py::call_guard<py::gil_scoped_release>());
    m.def("sub", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
         return tenzor::sub(a, b);
         }, "Element-wise subtraction", py::arg("a"), py::arg("b"),
         py::call_guard<py::gil_scoped_release>());
    m.def("mul", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
         return tenzor::mul(a, b);
         }, "Element-wise multiplication", py::arg("a"), py::arg("b"),
         py::call_guard<py::gil_scoped_release>());
    m.def("div", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
         return tenzor::div(a, b);
         }, "Element-wise division", py::arg("a"), py::arg("b"),
         py::call_guard<py::gil_scoped_release>());
    m.def("clamp", [](const tenzor::Tensor& input, float min_val, float max_val) {
         return tenzor::clamp(input, min_val, max_val);
         }, "Clamp values to [min, max]",
         py::arg("input"), py::arg("min"), py::arg("max"),
         py::call_guard<py::gil_scoped_release>());

    // Missing transform ops
    m.def("reshape", [](const tenzor::Tensor& input, std::vector<int64_t> shape) {
         return tenzor::reshape(input, std::move(shape));
         }, "Reshape tensor", py::arg("input"), py::arg("shape"));
    m.def("chunk", [](const tenzor::Tensor& input, int64_t chunks, int64_t dim) {
         return tenzor::chunk(input, chunks, dim);
         }, "Split tensor into chunks",
         py::arg("input"), py::arg("chunks"), py::arg("dim") = 0);

    m.def("matmul", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
         return tenzor::matmul(a, b);
         },
         R"doc(Matrix product of two tensors.

Supports 1D-1D (dot product), 2D-2D (matrix multiply), and batched
matrix multiplication with broadcasting. Uses MKL/cuBLAS when available.

Args:
    a: First tensor
    b: Second tensor

Returns:
    Result tensor. Shape depends on input dimensions.

Example::

    C = tz.matmul(A, B)  # (M, K) @ (K, N) -> (M, N)
)doc",
         py::arg("a"), py::arg("b"),
         py::call_guard<py::gil_scoped_release>());

    m.def("bmm", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
         return tenzor::bmm(a, b);
         }, "Batched matrix multiplication",
         py::arg("a"), py::arg("b"),
         py::call_guard<py::gil_scoped_release>());

    // Math operations - using lambda wrappers for overloaded functions
    // GIL released for compute-heavy operations
    m.def("exp", [](const tenzor::Tensor& t) { return tenzor::exp(t); },
         "Element-wise exponential", py::call_guard<py::gil_scoped_release>());
    m.def("log", [](const tenzor::Tensor& t) { return tenzor::log(t); },
         "Element-wise natural logarithm", py::call_guard<py::gil_scoped_release>());
    m.def("sqrt", [](const tenzor::Tensor& t) { return tenzor::sqrt(t); },
         "Element-wise square root", py::call_guard<py::gil_scoped_release>());
    m.def("abs", [](const tenzor::Tensor& t) { return tenzor::abs(t); },
         "Element-wise absolute value", py::call_guard<py::gil_scoped_release>());
    m.def("pow", [](const tenzor::Tensor& input, double exponent) {
         return tenzor::pow(input, exponent);
         }, "Element-wise power",
         py::arg("input"), py::arg("exponent"), py::call_guard<py::gil_scoped_release>());
    m.def("sin", [](const tenzor::Tensor& t) { return tenzor::sin(t); },
         "Element-wise sine", py::call_guard<py::gil_scoped_release>());
    m.def("cos", [](const tenzor::Tensor& t) { return tenzor::cos(t); },
         "Element-wise cosine", py::call_guard<py::gil_scoped_release>());
    m.def("tanh", [](const tenzor::Tensor& t) { return tenzor::tanh(t); },
         "Element-wise hyperbolic tangent", py::call_guard<py::gil_scoped_release>());
    // Extended math operations
    m.def("log2", [](const tenzor::Tensor& t) { return tenzor::log2(t); },
         "Element-wise base-2 logarithm", py::call_guard<py::gil_scoped_release>());
    m.def("log10", [](const tenzor::Tensor& t) { return tenzor::log10(t); },
         "Element-wise base-10 logarithm", py::call_guard<py::gil_scoped_release>());
    m.def("log1p", [](const tenzor::Tensor& t) { return tenzor::log1p(t); },
         "Element-wise log(1 + x)", py::call_guard<py::gil_scoped_release>());
    m.def("exp2", [](const tenzor::Tensor& t) { return tenzor::exp2(t); },
         "Element-wise 2^x", py::call_guard<py::gil_scoped_release>());
    m.def("expm1", [](const tenzor::Tensor& t) { return tenzor::expm1(t); },
         "Element-wise exp(x) - 1", py::call_guard<py::gil_scoped_release>());
    m.def("erf", [](const tenzor::Tensor& t) { return tenzor::erf(t); },
         "Element-wise error function", py::call_guard<py::gil_scoped_release>());
    m.def("erfc", [](const tenzor::Tensor& t) { return tenzor::erfc(t); },
         "Element-wise complementary error function", py::call_guard<py::gil_scoped_release>());
    m.def("erfinv", [](const tenzor::Tensor& t) { return tenzor::erfinv(t); },
         "Element-wise inverse error function", py::call_guard<py::gil_scoped_release>());

    // Special math functions
    m.def("gamma", [](const tenzor::Tensor& t) { return tenzor::gamma(t); },
         "Element-wise gamma function", py::call_guard<py::gil_scoped_release>());
    m.def("lgamma", [](const tenzor::Tensor& t) { return tenzor::lgamma(t); },
         "Element-wise log-gamma function", py::call_guard<py::gil_scoped_release>());
    m.def("digamma", [](const tenzor::Tensor& t) { return tenzor::digamma(t); },
         "Element-wise digamma (psi) function", py::call_guard<py::gil_scoped_release>());
    m.def("polygamma", [](int64_t n, const tenzor::Tensor& t) { return tenzor::polygamma(n, t); },
         "Element-wise polygamma function", py::arg("n"), py::arg("input"),
         py::call_guard<py::gil_scoped_release>());
    m.def("beta", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::beta(a, b); },
         "Element-wise beta function B(a,b)", py::call_guard<py::gil_scoped_release>());
    m.def("betainc", [](const tenzor::Tensor& a, const tenzor::Tensor& b, const tenzor::Tensor& x) {
         return tenzor::betainc(a, b, x); },
         "Regularized incomplete beta function I_x(a,b)", py::call_guard<py::gil_scoped_release>());
    m.def("bessel_j0", [](const tenzor::Tensor& t) { return tenzor::bessel_j0(t); },
         "Bessel function of first kind, order 0", py::call_guard<py::gil_scoped_release>());
    m.def("bessel_j1", [](const tenzor::Tensor& t) { return tenzor::bessel_j1(t); },
         "Bessel function of first kind, order 1", py::call_guard<py::gil_scoped_release>());
    m.def("bessel_y0", [](const tenzor::Tensor& t) { return tenzor::bessel_y0(t); },
         "Bessel function of second kind, order 0", py::call_guard<py::gil_scoped_release>());
    m.def("bessel_y1", [](const tenzor::Tensor& t) { return tenzor::bessel_y1(t); },
         "Bessel function of second kind, order 1", py::call_guard<py::gil_scoped_release>());
    m.def("bessel_i0", [](const tenzor::Tensor& t) { return tenzor::bessel_i0(t); },
         "Modified Bessel function of first kind, order 0", py::call_guard<py::gil_scoped_release>());
    m.def("bessel_i1", [](const tenzor::Tensor& t) { return tenzor::bessel_i1(t); },
         "Modified Bessel function of first kind, order 1", py::call_guard<py::gil_scoped_release>());
    m.def("sinc", [](const tenzor::Tensor& t) { return tenzor::sinc(t); },
         "Normalized sinc function: sin(pi*x)/(pi*x)", py::call_guard<py::gil_scoped_release>());
    m.def("zeta", [](const tenzor::Tensor& x, const tenzor::Tensor& q) { return tenzor::zeta(x, q); },
         "Hurwitz zeta function", py::call_guard<py::gil_scoped_release>());

    // Matrix construction operations
    m.def("kron", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::kron(a, b); },
         "Kronecker product of two 2-D tensors", py::call_guard<py::gil_scoped_release>());
    m.def("block_diag", [](std::vector<tenzor::Tensor> tensors) {
         return tenzor::block_diag(tensors); },
         "Create block diagonal matrix from tensors", py::call_guard<py::gil_scoped_release>());
    m.def("vander", [](const tenzor::Tensor& x, int64_t N, bool increasing) {
         return tenzor::vander(x, N, increasing); },
         "Generate Vandermonde matrix", py::arg("x"), py::arg("N") = -1, py::arg("increasing") = false,
         py::call_guard<py::gil_scoped_release>());
    m.def("cartesian_prod", [](std::vector<tenzor::Tensor> tensors) {
         return tenzor::cartesian_prod(tensors); },
         "Cartesian product of 1-D tensors", py::call_guard<py::gil_scoped_release>());
    m.def("combinations", [](const tenzor::Tensor& input, int64_t r, bool with_replacement) {
         return tenzor::combinations(input, r, with_replacement); },
         "All r-length combinations from input", py::arg("input"), py::arg("r"),
         py::arg("with_replacement") = false, py::call_guard<py::gil_scoped_release>());

    m.def("isnan", [](const tenzor::Tensor& t) { return tenzor::isnan(t); },
         "Element-wise NaN test", py::call_guard<py::gil_scoped_release>());
    m.def("isinf", [](const tenzor::Tensor& t) { return tenzor::isinf(t); },
         "Element-wise infinity test", py::call_guard<py::gil_scoped_release>());
    m.def("isfinite", [](const tenzor::Tensor& t) { return tenzor::isfinite(t); },
         "Element-wise finiteness test", py::call_guard<py::gil_scoped_release>());
    m.def("atan2", [](const tenzor::Tensor& y, const tenzor::Tensor& x) { return tenzor::atan2(y, x); },
         "Element-wise atan2", py::arg("y"), py::arg("x"),
         py::call_guard<py::gil_scoped_release>());
    m.def("fmod", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::fmod(a, b); },
         "Element-wise float modulo", py::arg("a"), py::arg("b"),
         py::call_guard<py::gil_scoped_release>());
    m.def("remainder", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::remainder(a, b); },
         "Element-wise remainder", py::arg("a"), py::arg("b"),
         py::call_guard<py::gil_scoped_release>());
    m.def("lerp", [](const tenzor::Tensor& start, const tenzor::Tensor& end, double weight) {
         return tenzor::lerp(start, end, weight);
         }, "Linear interpolation",
         py::arg("start"), py::arg("end"), py::arg("weight"),
         py::call_guard<py::gil_scoped_release>());
    m.def("logical_and", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
         return tenzor::logical_and(a, b);
         }, "Element-wise logical AND", py::arg("a"), py::arg("b"),
         py::call_guard<py::gil_scoped_release>());
    m.def("logical_or", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
         return tenzor::logical_or(a, b);
         }, "Element-wise logical OR", py::arg("a"), py::arg("b"),
         py::call_guard<py::gil_scoped_release>());
    m.def("logical_not", [](const tenzor::Tensor& t) { return tenzor::logical_not(t); },
         "Element-wise logical NOT", py::call_guard<py::gil_scoped_release>());
    m.def("logical_xor", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
         return tenzor::logical_xor(a, b);
         }, "Element-wise logical XOR", py::arg("a"), py::arg("b"),
         py::call_guard<py::gil_scoped_release>());
    m.def("meshgrid", [](const std::vector<tenzor::Tensor>& tensors, const std::string& indexing) {
         return tenzor::meshgrid(tensors, indexing);
         }, "Generate coordinate grids from 1-D tensors",
         py::arg("tensors"), py::arg("indexing") = "ij");
    m.def("cross", [](const tenzor::Tensor& a, const tenzor::Tensor& b, int64_t dim) {
         return tenzor::cross(a, b, dim);
         }, "Cross product of two tensors along dimension",
         py::arg("input"), py::arg("other"), py::arg("dim") = -1);
    // Search operations
    m.def("searchsorted", [](const tenzor::Tensor& sorted_sequence, const tenzor::Tensor& values, bool right) {
         return tenzor::searchsorted(sorted_sequence, values, right);
         }, "Find insertion indices in sorted sequence",
         py::arg("sorted_sequence"), py::arg("values"), py::arg("right") = false,
         py::call_guard<py::gil_scoped_release>());
    // Sampling operations
    m.def("gumbel_softmax", [](const tenzor::Tensor& logits, double tau, bool hard, int64_t dim) {
         return tenzor::gumbel_softmax(logits, tau, hard, dim);
         }, "Sample from categorical distribution using Gumbel-Softmax trick",
         py::arg("logits"), py::arg("tau") = 1.0, py::arg("hard") = false, py::arg("dim") = -1,
         py::call_guard<py::gil_scoped_release>());
    // More trig/hyperbolic
    m.def("tan", [](const tenzor::Tensor& t) { return tenzor::tan(t); },
         "Element-wise tangent", py::call_guard<py::gil_scoped_release>());
    m.def("asin", [](const tenzor::Tensor& t) { return tenzor::asin(t); },
         "Element-wise arcsine", py::call_guard<py::gil_scoped_release>());
    m.def("acos", [](const tenzor::Tensor& t) { return tenzor::acos(t); },
         "Element-wise arccosine", py::call_guard<py::gil_scoped_release>());
    m.def("atan", [](const tenzor::Tensor& t) { return tenzor::atan(t); },
         "Element-wise arctangent", py::call_guard<py::gil_scoped_release>());
    m.def("sinh", [](const tenzor::Tensor& t) { return tenzor::sinh(t); },
         "Element-wise hyperbolic sine", py::call_guard<py::gil_scoped_release>());
    m.def("cosh", [](const tenzor::Tensor& t) { return tenzor::cosh(t); },
         "Element-wise hyperbolic cosine", py::call_guard<py::gil_scoped_release>());
    // Element-wise ops
    m.def("neg", [](const tenzor::Tensor& t) { return tenzor::neg(t); },
         "Element-wise negation", py::call_guard<py::gil_scoped_release>());
    m.def("sign", [](const tenzor::Tensor& t) { return tenzor::sign(t); },
         "Element-wise sign function", py::call_guard<py::gil_scoped_release>());
    m.def("sigmoid", [](const tenzor::Tensor& t) { return tenzor::sigmoid(t); },
         "Element-wise sigmoid");
    m.def("reciprocal", [](const tenzor::Tensor& t) { return tenzor::reciprocal(t); },
         "Element-wise reciprocal");
    m.def("floor", [](const tenzor::Tensor& t) { return tenzor::floor(t); },
         "Element-wise floor");
    m.def("ceil", [](const tenzor::Tensor& t) { return tenzor::ceil(t); },
         "Element-wise ceil");
    m.def("round", [](const tenzor::Tensor& t) { return tenzor::round(t); },
         "Element-wise round");
    m.def("clamp_min", [](const tenzor::Tensor& t, float min_val) { return tenzor::clamp_min(t, min_val); },
         "Clamp values to minimum", py::arg("input"), py::arg("min"));
    m.def("clamp_max", [](const tenzor::Tensor& t, float max_val) { return tenzor::clamp_max(t, max_val); },
         "Clamp values to maximum", py::arg("input"), py::arg("max"));
    m.def("minimum", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::minimum(a, b); },
         "Element-wise minimum", py::arg("a"), py::arg("b"));
    m.def("maximum", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::maximum(a, b); },
         "Element-wise maximum", py::arg("a"), py::arg("b"));
    // Comparison ops
    m.def("eq", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::eq(a, b); },
         "Element-wise equality", py::arg("a"), py::arg("b"));
    m.def("ne", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::ne(a, b); },
         "Element-wise inequality", py::arg("a"), py::arg("b"));
    m.def("lt", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::lt(a, b); },
         "Element-wise less than", py::arg("a"), py::arg("b"));
    m.def("le", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::le(a, b); },
         "Element-wise less or equal", py::arg("a"), py::arg("b"));
    m.def("gt", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::gt(a, b); },
         "Element-wise greater than", py::arg("a"), py::arg("b"));
    m.def("ge", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::ge(a, b); },
         "Element-wise greater or equal", py::arg("a"), py::arg("b"));
    // Additional reduction ops
    m.def("argmax", [](const tenzor::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::argmax(input, dim, keepdim);
         }, "ArgMax reduction",
         py::arg("input"), py::arg("dim") = py::none(), py::arg("keepdim") = false);
    m.def("argmin", [](const tenzor::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::argmin(input, dim, keepdim);
         }, "ArgMin reduction",
         py::arg("input"), py::arg("dim") = py::none(), py::arg("keepdim") = false);

    // Reduction operations - using lambda wrappers for overloaded functions
    m.def("sum", [](const tenzor::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::sum(input, dim, keepdim);
         }, "Sum reduction (Tensor)",
         py::arg("input"),
         py::arg("dim") = py::none(),
         py::arg("keepdim") = false,
         py::call_guard<py::gil_scoped_release>());
    m.def("sum", [](const tenzor::Variable& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::sum(input, dim, keepdim);
         }, "Sum reduction with autograd (Variable)",
         py::arg("input"),
         py::arg("dim") = py::none(),
         py::arg("keepdim") = false);
    m.def("mean", [](const tenzor::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::mean(input, dim, keepdim);
         }, "Mean reduction (Tensor)",
         py::arg("input"),
         py::arg("dim") = py::none(),
         py::arg("keepdim") = false,
         py::call_guard<py::gil_scoped_release>());
    m.def("mean", [](const tenzor::Variable& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::mean(input, dim, keepdim);
         }, "Mean reduction with autograd (Variable)",
         py::arg("input"),
         py::arg("dim") = py::none(),
         py::arg("keepdim") = false);
    m.def("max", [](const tenzor::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::max(input, dim, keepdim);
         }, "Max reduction",
         py::arg("input"),
         py::arg("dim") = py::none(),
         py::arg("keepdim") = false,
         py::call_guard<py::gil_scoped_release>());
    m.def("min", [](const tenzor::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::min(input, dim, keepdim);
         }, "Min reduction",
         py::arg("input"),
         py::arg("dim") = py::none(),
         py::arg("keepdim") = false,
         py::call_guard<py::gil_scoped_release>());

    m.def("any", [](const tenzor::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::any(input, dim, keepdim);
         }, "Any reduction",
         py::arg("input"),
         py::arg("dim") = py::none(),
         py::arg("keepdim") = false,
         py::call_guard<py::gil_scoped_release>());
    m.def("all", [](const tenzor::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::all(input, dim, keepdim);
         }, "All reduction",
         py::arg("input"),
         py::arg("dim") = py::none(),
         py::arg("keepdim") = false,
         py::call_guard<py::gil_scoped_release>());

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
    m.def("unsqueeze", static_cast<tenzor::Tensor(*)(const tenzor::Tensor&, int64_t)>(&tenzor::unsqueeze),
         "Add dimension of size 1",
         py::arg("input"), py::arg("dim"));
    m.def("flatten", static_cast<tenzor::Tensor(*)(const tenzor::Tensor&, int64_t, int64_t)>(&tenzor::flatten),
         "Flatten tensor",
         py::arg("input"),
         py::arg("start_dim") = 0,
         py::arg("end_dim") = -1);
    m.def("contiguous", &tenzor::contiguous, "Make tensor contiguous");

    // Triangular and diagonal operations
    m.def("triu", [](const tenzor::Tensor& input, int64_t diagonal) {
         return tenzor::triu(input, diagonal);
         }, "Extract upper triangular part",
         py::arg("input"), py::arg("diagonal") = 0);
    m.def("tril", [](const tenzor::Tensor& input, int64_t diagonal) {
         return tenzor::tril(input, diagonal);
         }, "Extract lower triangular part",
         py::arg("input"), py::arg("diagonal") = 0);
    m.def("diag", [](const tenzor::Tensor& input, int64_t diagonal) {
         return tenzor::diag(input, diagonal);
         }, "Extract diagonal or construct diagonal matrix",
         py::arg("input"), py::arg("diagonal") = 0);
    m.def("trace", [](const tenzor::Tensor& input) {
         return tenzor::trace(input);
         }, "Sum of diagonal elements",
         py::arg("input"));
    m.def("flip", [](const tenzor::Tensor& input, std::vector<int64_t> dims) {
         return tenzor::flip(input, std::move(dims));
         }, "Reverse tensor along dimensions",
         py::arg("input"), py::arg("dims"));

    // Concatenation and stacking operations
    m.def("cat", [](const std::vector<tenzor::Tensor>& tensors, int64_t dim) {
         return tenzor::cat(tensors, dim);
         }, "Concatenate tensors along dimension",
         py::arg("tensors"), py::arg("dim")=0);
    m.def("stack", [](const std::vector<tenzor::Tensor>& tensors, int64_t dim) {
         return tenzor::stack(tensors, dim);
         }, "Stack tensors along new dimension",
         py::arg("tensors"), py::arg("dim")=0);
    m.def("split", [](const tenzor::Tensor& tensor, int64_t split_size, int64_t dim) {
         return tenzor::split(tensor, split_size, dim);
         }, "Split tensor into chunks",
         py::arg("tensor"), py::arg("split_size"), py::arg("dim")=0);
    m.def("roll", [](const tenzor::Tensor& input, int64_t shifts, int64_t dim) {
         return tenzor::roll(input, shifts, dim);
         }, "Roll tensor elements along dimension",
         py::arg("input"), py::arg("shifts"), py::arg("dim") = 0,
         py::call_guard<py::gil_scoped_release>());
    m.def("split_with_sizes", [](const tenzor::Tensor& input, const std::vector<int64_t>& split_sizes, int64_t dim) {
         return tenzor::split_with_sizes(input, split_sizes, dim);
         }, "Split tensor into chunks with specified sizes",
         py::arg("input"), py::arg("split_sizes"), py::arg("dim") = 0,
         py::call_guard<py::gil_scoped_release>());

    // Indexing operations
    // Cast to the tensor-level slice function to avoid ambiguity with autograd::slice
    m.def("slice", static_cast<tenzor::Tensor(*)(const tenzor::Tensor&, int64_t, int64_t, int64_t, int64_t)>(&tenzor::slice),
         "Slice tensor along dimension",
         py::arg("input"), py::arg("dim"), py::arg("start"), py::arg("end"),
         py::arg("step") = 1);
    m.def("index_select", static_cast<tenzor::Tensor(*)(const tenzor::Tensor&, int64_t, const tenzor::Tensor&)>(&tenzor::index_select),
         "Select indices along dimension",
         py::arg("input"), py::arg("dim"), py::arg("index"));
    m.def("gather", static_cast<tenzor::Tensor(*)(const tenzor::Tensor&, int64_t, const tenzor::Tensor&)>(&tenzor::gather),
         "Gather elements along dimension",
         py::arg("input"), py::arg("dim"), py::arg("index"));
    m.def("scatter", [](const tenzor::Tensor& input, int64_t dim, const tenzor::Tensor& index, const tenzor::Tensor& src) {
         return tenzor::scatter(input, dim, index, src);
         }, "Scatter elements along dimension",
         py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("src"));
    m.def("scatter_add", [](const tenzor::Tensor& input, int64_t dim, const tenzor::Tensor& index, const tenzor::Tensor& src) {
         return tenzor::scatter_add(input, dim, index, src);
         }, "Scatter-add elements along dimension",
         py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("src"));
    m.def("masked_select", &tenzor::masked_select, "Select elements where mask is true",
         py::arg("input"), py::arg("mask"));
    m.def("masked_fill", &tenzor::masked_fill, "Fill elements with value where mask is true",
         py::arg("input"), py::arg("mask"), py::arg("value"));
    m.def("where", [](const tenzor::Tensor& condition, const tenzor::Tensor& x, const tenzor::Tensor& y) {
         return tenzor::where(condition, x, y);
         }, "Conditional element selection",
         py::arg("condition"), py::arg("x"), py::arg("y"));
    m.def("take", &tenzor::take, "Take elements from flattened tensor",
         py::arg("input"), py::arg("index"));
    m.def("put", &tenzor::put, "Put elements into flattened tensor",
         py::arg("input"), py::arg("index"), py::arg("source"));

    // Broadcast/expansion operations
    m.def("expand", [](const tenzor::Tensor& input, std::vector<int64_t> shape) {
        return tenzor::expand(input, std::move(shape));
    }, "Expand tensor to a larger size (broadcast without copying)",
         py::arg("input"), py::arg("shape"),
         py::call_guard<py::gil_scoped_release>());
    m.def("repeat", [](const tenzor::Tensor& input, std::vector<int64_t> repeats) {
        return tenzor::repeat(input, std::move(repeats));
    }, "Repeat tensor along each dimension",
         py::arg("input"), py::arg("repeats"),
         py::call_guard<py::gil_scoped_release>());
    m.def("tile", [](const tenzor::Tensor& input, std::vector<int64_t> reps) {
        return tenzor::tile(input, std::move(reps));
    }, "Tile tensor by repeating along each dimension",
         py::arg("input"), py::arg("reps"),
         py::call_guard<py::gil_scoped_release>());
    m.def("broadcast_to", [](const tenzor::Tensor& input, std::vector<int64_t> shape) {
        return tenzor::expand(input, std::move(shape));
    }, "Broadcast tensor to a target shape",
         py::arg("input"), py::arg("shape"),
         py::call_guard<py::gil_scoped_release>());

    // Advanced operations
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


    // Autograd
    py::class_<tenzor::Variable, std::shared_ptr<tenzor::Variable>>(m, "Variable")
        .def(py::init([](tenzor::Tensor data, bool requires_grad) {
            return std::make_shared<tenzor::Variable>(data, requires_grad);
        }), py::arg("data"), py::arg("requires_grad") = false)
        .def("backward", &tenzor::Variable::backward,
             py::arg("gradient") = py::none(),
             py::arg("retain_graph") = false,
             py::arg("create_graph") = false,
             py::call_guard<py::gil_scoped_release>(),
             "Compute gradients via backpropagation")
        // Tensor access - both as property and method for compatibility
        .def_property_readonly("data", py::overload_cast<>(&tenzor::Variable::tensor, py::const_))
        .def("tensor", py::overload_cast<>(&tenzor::Variable::tensor),
             "Get the underlying tensor (mutable)")
        .def("tensor", py::overload_cast<>(&tenzor::Variable::tensor, py::const_),
             "Get the underlying tensor (const)")
        // Gradient access
        .def_property_readonly("grad", py::overload_cast<>(&tenzor::Variable::grad, py::const_))
        .def_property_readonly("grad_fn", &tenzor::Variable::grad_fn,
             "Get gradient function that created this variable")
        .def_property_readonly("is_leaf", &tenzor::Variable::is_leaf,
             "Check if variable is a leaf node")
        .def_property_readonly("is_cuda", [](const tenzor::Variable& self) {
            return self.device().type == tenzor::Device::Type::CUDA;
        })
        .def_property_readonly("is_cpu", [](const tenzor::Variable& self) {
            return self.device().type == tenzor::Device::Type::CPU;
        })
        // Gradient control
        .def_property_readonly("requires_grad", &tenzor::Variable::requires_grad,
             "Check if variable requires gradient")
        .def("requires_grad_", &tenzor::Variable::set_requires_grad,
             py::arg("requires_grad") = true,
             "Set requires_grad in-place")
        .def("zero_grad", &tenzor::Variable::zero_grad,
             "Zero the gradient")
        // Hook registration — wrap Python callable to acquire GIL during backward
        .def("register_hook", [](tenzor::Variable& self, py::object hook) {
            py::object hook_ref = hook;  // explicit refcount increment
            auto cpp_hook = [hook_ref](const tenzor::Tensor& grad) -> tenzor::Tensor {
                py::gil_scoped_acquire acquire;
                try {
                    py::object result = hook_ref(grad);
                    return result.cast<tenzor::Tensor>();
                } catch (py::error_already_set& e) {
                    throw;  // Re-raise Python exceptions properly
                }
            };
            return self.register_hook(cpp_hook);
        }, py::arg("hook"), "Register a backward hook function")
        .def("unregister_hook", &tenzor::Variable::unregister_hook,
             py::arg("hook_id"), "Remove a previously registered backward hook")
        .def("retain_grad", &tenzor::Variable::retain_grad,
             "Enable gradient retention for non-leaf variables")
        .def_property_readonly("retains_grad", &tenzor::Variable::retains_grad,
             "Check if variable retains gradient")
        // Shape convenience
        .def("shape", [](const tenzor::Variable& self) {
            auto s = self.tensor().shape();
            return std::vector<int64_t>(s.begin(), s.end());
        }, "Get the shape of the underlying tensor")
        .def("numel", [](const tenzor::Variable& self) {
            return self.tensor().numel();
        }, "Get the number of elements")
        .def("item", [](const tenzor::Variable& self) -> py::object {
            const auto& t = self.tensor();
            if (t.numel() != 1) {
                throw std::runtime_error("item() only works for scalar tensors");
            }
            auto dtype = t.dtype();
            double dval = 0;
            int64_t ival = 0;
            uint64_t uval = 0;
            bool bval = false;
            std::complex<double> cval{};
            bool is_float = false, is_int = false, is_uint = false, is_bool = false, is_complex = false;
            {
                py::gil_scoped_release release;
                switch (dtype) {
                    case tenzor::DType::Float32:  dval = t.item<float>(); is_float = true; break;
                    case tenzor::DType::Float64:  dval = t.item<double>(); is_float = true; break;
                    case tenzor::DType::Float16:  dval = static_cast<float>(t.data<tenzor::Float16>()[0]); is_float = true; break;
                    case tenzor::DType::BFloat16: dval = static_cast<float>(t.data<tenzor::BFloat16>()[0]); is_float = true; break;
                    case tenzor::DType::Int8:     ival = t.item<int8_t>(); is_int = true; break;
                    case tenzor::DType::Int16:    ival = t.item<int16_t>(); is_int = true; break;
                    case tenzor::DType::Int32:    ival = t.item<int32_t>(); is_int = true; break;
                    case tenzor::DType::Int64:    ival = t.item<int64_t>(); is_int = true; break;
                    case tenzor::DType::UInt8:    uval = t.item<uint8_t>(); is_uint = true; break;
                    case tenzor::DType::UInt16:   uval = t.item<uint16_t>(); is_uint = true; break;
                    case tenzor::DType::UInt32:   uval = t.item<uint32_t>(); is_uint = true; break;
                    case tenzor::DType::UInt64:   uval = t.item<uint64_t>(); is_uint = true; break;
                    case tenzor::DType::Bool:     bval = t.item<bool>(); is_bool = true; break;
                    case tenzor::DType::Complex64: {
                        auto c = t.item<std::complex<float>>();
                        cval = {c.real(), c.imag()};
                        is_complex = true; break;
                    }
                    case tenzor::DType::Complex128: cval = t.item<std::complex<double>>(); is_complex = true; break;
                    default:
                        throw std::runtime_error("Unsupported dtype for item()");
                }
            }
            if (is_float)   return py::cast(dval);
            if (is_int)     return py::cast(ival);
            if (is_uint)    return py::cast(uval);
            if (is_bool)    return py::cast(bval);
            if (is_complex) return py::cast(cval);
            throw std::runtime_error("Unsupported dtype for item()");
        }, "Extract scalar value from single-element variable")
        // Arithmetic operators - Variable-Variable (GIL released for compute)
        .def("__add__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return a + b;
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__sub__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return a - b;
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__mul__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return a * b;
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__truediv__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return a / b;
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        // Arithmetic operators - Variable-Scalar (double to match Python float precision)
        .def("__add__", [](const tenzor::Variable& a, double b) {
            return a + b;
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__radd__", [](const tenzor::Variable& a, double b) {
            return a + b;
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__mul__", [](const tenzor::Variable& a, double b) {
            return a * b;
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__rmul__", [](const tenzor::Variable& a, double b) {
            return a * b;
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__sub__", [](const tenzor::Variable& a, double b) {
            return a - b;
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__rsub__", [](const tenzor::Variable& a, double b) {
            // b - a = -(a - b)
            return (a - b) * -1.0;
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__truediv__", [](const tenzor::Variable& a, double b) {
            return a / b;
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__rtruediv__", [](const tenzor::Variable& a, double b) {
            // b / a: create scalar variable with value b, divide by a
            auto b_var = tenzor::Variable(
                tenzor::full({}, b, a.dtype(), a.device()), false);
            return b_var / a;
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__neg__", [](const tenzor::Variable& a) {
            return a * -1.0;
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        // Matrix multiplication
        .def("__matmul__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return a.matmul(b);
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__rmatmul__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return b.matmul(a);
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        // Power (uses tensor-level pow, wraps back in Variable)
        .def("__pow__", [](const tenzor::Variable& a, float exp) {
            auto result = tenzor::pow(a.tensor(), exp);
            return tenzor::Variable(result, false);
        }, py::is_operator())
        // Modulo and floor division (operate on underlying tensors)
        .def("__mod__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            auto result = tenzor::fmod(a.tensor(), b.tensor());
            return tenzor::Variable(result, false);
        }, py::is_operator())
        .def("__mod__", [](const tenzor::Variable& a, float b) {
            auto b_tensor = tenzor::full(std::vector<int64_t>{}, static_cast<double>(b),
                                         a.dtype(), a.device());
            auto result = tenzor::fmod(a.tensor(), b_tensor);
            return tenzor::Variable(result, false);
        }, py::is_operator())
        .def("__rmod__", [](const tenzor::Variable& a, float b) {
            auto b_tensor = tenzor::full(std::vector<int64_t>{}, static_cast<double>(b),
                                         a.dtype(), a.device());
            auto result = tenzor::fmod(b_tensor, a.tensor());
            return tenzor::Variable(result, false);
        }, py::is_operator())
        .def("__floordiv__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            auto result = tenzor::floor(a.tensor() / b.tensor());
            return tenzor::Variable(result, false);
        }, py::is_operator())
        .def("__floordiv__", [](const tenzor::Variable& a, float b) {
            auto result = tenzor::floor(a.tensor() / static_cast<double>(b));
            return tenzor::Variable(result, false);
        }, py::is_operator())
        .def("__rfloordiv__", [](const tenzor::Variable& a, float b) {
            auto b_tensor = tenzor::full(std::vector<int64_t>{}, static_cast<double>(b),
                                         a.dtype(), a.device());
            auto result = tenzor::floor(b_tensor / a.tensor());
            return tenzor::Variable(result, false);
        }, py::is_operator())
        // Comparison operators (return Tensors, not Variables — no grad needed)
        .def("__eq__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return tenzor::eq(a.tensor(), b.tensor());
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__ne__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return tenzor::ne(a.tensor(), b.tensor());
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__lt__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return tenzor::lt(a.tensor(), b.tensor());
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__le__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return tenzor::le(a.tensor(), b.tensor());
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__gt__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return tenzor::gt(a.tensor(), b.tensor());
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__ge__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return tenzor::ge(a.tensor(), b.tensor());
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        // Scalar comparison operators
        .def("__eq__", [](const tenzor::Variable& a, double b) {
            return tenzor::eq(a.tensor(), tenzor::full(std::vector<int64_t>{}, b, a.dtype(), a.device()));
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__ne__", [](const tenzor::Variable& a, double b) {
            return tenzor::ne(a.tensor(), tenzor::full(std::vector<int64_t>{}, b, a.dtype(), a.device()));
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__lt__", [](const tenzor::Variable& a, double b) {
            return tenzor::lt(a.tensor(), tenzor::full(std::vector<int64_t>{}, b, a.dtype(), a.device()));
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__le__", [](const tenzor::Variable& a, double b) {
            return tenzor::le(a.tensor(), tenzor::full(std::vector<int64_t>{}, b, a.dtype(), a.device()));
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__gt__", [](const tenzor::Variable& a, double b) {
            return tenzor::gt(a.tensor(), tenzor::full(std::vector<int64_t>{}, b, a.dtype(), a.device()));
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        .def("__ge__", [](const tenzor::Variable& a, double b) {
            return tenzor::ge(a.tensor(), tenzor::full(std::vector<int64_t>{}, b, a.dtype(), a.device()));
        }, py::is_operator(), py::call_guard<py::gil_scoped_release>())
        // In-place operators
        .def("__iadd__", [](tenzor::Variable& a, const tenzor::Variable& b) -> tenzor::Variable& {
            tenzor::add_(a.tensor(), b.tensor());
            return a;
        }, py::is_operator())
        .def("__isub__", [](tenzor::Variable& a, const tenzor::Variable& b) -> tenzor::Variable& {
            tenzor::sub_(a.tensor(), b.tensor());
            return a;
        }, py::is_operator())
        .def("__imul__", [](tenzor::Variable& a, const tenzor::Variable& b) -> tenzor::Variable& {
            tenzor::mul_(a.tensor(), b.tensor());
            return a;
        }, py::is_operator())
        .def("__itruediv__", [](tenzor::Variable& a, const tenzor::Variable& b) -> tenzor::Variable& {
            tenzor::div_(a.tensor(), b.tensor());
            return a;
        }, py::is_operator())
        // Scalar in-place
        .def("__iadd__", [](tenzor::Variable& a, float b) -> tenzor::Variable& {
            auto scalar_t = tenzor::full({1}, static_cast<double>(b), a.dtype(), a.device());
            tenzor::add_(a.tensor(), scalar_t);
            return a;
        }, py::is_operator())
        .def("__isub__", [](tenzor::Variable& a, float b) -> tenzor::Variable& {
            auto scalar_t = tenzor::full({1}, static_cast<double>(b), a.dtype(), a.device());
            tenzor::sub_(a.tensor(), scalar_t);
            return a;
        }, py::is_operator())
        .def("__imul__", [](tenzor::Variable& a, float b) -> tenzor::Variable& {
            auto scalar_t = tenzor::full({1}, static_cast<double>(b), a.dtype(), a.device());
            tenzor::mul_(a.tensor(), scalar_t);
            return a;
        }, py::is_operator())
        .def("__itruediv__", [](tenzor::Variable& a, float b) -> tenzor::Variable& {
            auto scalar_t = tenzor::full({1}, static_cast<double>(b), a.dtype(), a.device());
            tenzor::div_(a.tensor(), scalar_t);
            return a;
        }, py::is_operator())
        // Numeric protocol
        .def("__float__", [](const tenzor::Variable& v) {
            auto t = v.tensor();
            if (t.numel() != 1) {
                throw py::value_error(
                    "only single-element tensors can be converted to Python scalars (got " +
                    std::to_string(t.numel()) + " elements)");
            }
            if (t.dtype() == tenzor::DType::Float64)
                return t.item<double>();
            return static_cast<double>(t.item<float>());
        })
        .def("__int__", [](const tenzor::Variable& v) {
            auto t = v.tensor();
            if (t.numel() != 1) {
                throw py::value_error(
                    "only single-element tensors can be converted to Python scalars (got " +
                    std::to_string(t.numel()) + " elements)");
            }
            if (t.dtype() == tenzor::DType::Int64)
                return t.item<int64_t>();
            if (t.dtype() == tenzor::DType::Int32)
                return static_cast<int64_t>(t.item<int32_t>());
            return static_cast<int64_t>(t.item<float>());
        })
        .def("__repr__", [](const tenzor::Variable& v) {
            if (!v.is_initialized()) {
                return std::string("Variable(<uninitialized>)");
            }
            // Get the Tensor repr and replace "tensor(" prefix with "Variable("
            // This reuses the multi-dimensional formatting from Tensor.__repr__
            auto t_repr = py::cast(v.tensor()).attr("__repr__")().cast<std::string>();
            // Insert requires_grad before the closing paren
            std::string result;
            if (t_repr.size() > 1 && t_repr.back() == ')') {
                result = t_repr.substr(0, t_repr.size() - 1);
                result += ", requires_grad=";
                result += v.requires_grad() ? "True" : "False";
                result += ")";
            } else {
                result = "Variable(" + t_repr + ", requires_grad=" +
                         (v.requires_grad() ? "True" : "False") + ")";
            }
            return result;
        })
        .def("dim", [](const tenzor::Variable& v) { return v.tensor().ndim(); },
             "Number of dimensions")
        .def("__index__", [](const tenzor::Variable& v) {
            return v.tensor().item<int64_t>();
        })
        .def("__len__", [](const tenzor::Variable& v) -> int64_t {
            if (v.tensor().ndim() == 0) throw py::type_error("len() of a 0-d Variable");
            return v.tensor().shape()[0];
        })
        .def_property_readonly("strides", [](const tenzor::Variable& v) {
            auto s = v.tensor().strides();
            py::tuple result(s.size());
            for (size_t i = 0; i < s.size(); ++i)
                result[i] = s[i];
            return result;
        })
        // Pickle support for model saving/loading
        .def(py::pickle(
            // __getstate__: serialize to (shape, dtype_int, device_str, requires_grad, bytes)
            [](const tenzor::Variable& v) {
                tenzor::Tensor t = v.tensor();
                tenzor::Tensor cpu_t = (t.device().type != tenzor::Device::Type::CPU)
                    ? t.to(tenzor::Device::cpu()) : t;
                if (!cpu_t.is_contiguous()) cpu_t = cpu_t.contiguous();

                auto shape = cpu_t.shape();
                std::vector<int64_t> shape_vec(shape.begin(), shape.end());

                size_t nbytes = static_cast<size_t>(cpu_t.numel()) * tenzor::dtype_size(cpu_t.dtype());
                py::bytes data(reinterpret_cast<const char*>(cpu_t.data_ptr()), nbytes);

                return py::make_tuple(
                    shape_vec,
                    static_cast<int>(cpu_t.dtype()),
                    t.device().to_string(),
                    v.requires_grad(),
                    data
                );
            },
            // __setstate__: deserialize from (shape, dtype_int, device_str, requires_grad, bytes)
            [](py::tuple state) {
                if (state.size() != 5) throw std::runtime_error("Invalid pickle state");

                auto shape = state[0].cast<std::vector<int64_t>>();
                auto dtype_int = state[1].cast<int>();
                if (dtype_int < 0 || dtype_int > static_cast<int>(tenzor::DType::Complex128))
                    throw std::runtime_error("Invalid dtype in pickle state: " + std::to_string(dtype_int));
                for (auto d : shape)
                    if (d < 0) throw std::runtime_error("Negative dimension in pickle state");
                auto dtype = static_cast<tenzor::DType>(dtype_int);
                auto device_str = state[2].cast<std::string>();
                bool requires_grad = state[3].cast<bool>();
                auto data = state[4].cast<std::string>();

                tenzor::Tensor t(shape, dtype, tenzor::Device::cpu());
                size_t nbytes = static_cast<size_t>(t.numel()) * tenzor::dtype_size(dtype);
                if (data.size() != nbytes) {
                    throw std::runtime_error("Pickle data size mismatch");
                }
                std::memcpy(t.data_ptr(), data.data(), nbytes);

                auto target_device = tenzor::Device::from_string(device_str);
                if (target_device.type != tenzor::Device::Type::CPU) {
                    t = t.to(target_device);
                }
                return std::make_shared<tenzor::Variable>(t, requires_grad);
            }
        ));

    // Gradient control functions
    m.def("is_grad_enabled", &tenzor::is_grad_enabled,
          "Check if gradient computation is globally enabled");

    m.def("set_grad_enabled", &tenzor::set_grad_enabled,
          py::arg("enabled"),
          "Set global gradient computation state");

    // Python-friendly context manager wrapper for no_grad
    // NoGradGuard is not movable/copyable, so we wrap it in a class that manages its lifetime
    struct PyNoGradContext {
        std::unique_ptr<tenzor::NoGradGuard> guard_;

        void enter() {
            guard_ = std::make_unique<tenzor::NoGradGuard>();
        }

        void exit() {
            guard_.reset();
        }
    };

    py::class_<PyNoGradContext>(m, "no_grad",
        "Context manager and decorator for disabling gradient computation.\n\n"
        "Usage as context manager:\n"
        "    with tz.no_grad():\n"
        "        y = model(x)\n\n"
        "Usage as decorator:\n"
        "    @tz.no_grad()\n"
        "    def inference(x):\n"
        "        return model(x)")
        .def(py::init<>())
        .def("__enter__", [](PyNoGradContext& self) -> PyNoGradContext& {
            self.enter();
            return self;
        })
        .def("__exit__", [](PyNoGradContext& self, py::object, py::object, py::object) {
            self.exit();
            return false;
        })
        .def("__call__", [](PyNoGradContext&, py::function func) -> py::object {
            // When used as @no_grad() decorator, wrap the function
            // so that grad is disabled during its execution
            auto wrapper = py::cpp_function([func](py::args args, py::kwargs kwargs) -> py::object {
                tenzor::NoGradGuard guard;
                return func(*args, **kwargs);
            });
            // Preserve original function metadata (__name__, __doc__, etc.)
            try {
                py::module_ functools = py::module_::import("functools");
                functools.attr("update_wrapper")(wrapper, func);
            } catch (py::error_already_set&) {
                // functools.update_wrapper failed — not critical, continue without metadata
                PyErr_Clear();
            }
            return wrapper;
        }, py::arg("func"));

    // Python-friendly context manager for enable_grad
    struct PyEnableGradContext {
        bool prev_state_ = false;

        void enter() {
            prev_state_ = tenzor::is_grad_enabled();
            tenzor::set_grad_enabled(true);
        }

        void exit() {
            tenzor::set_grad_enabled(prev_state_);
        }
    };

    py::class_<PyEnableGradContext>(m, "enable_grad",
        "Context manager and decorator for enabling gradient computation")
        .def(py::init<>())
        .def("__enter__", [](PyEnableGradContext& self) -> PyEnableGradContext& {
            self.enter();
            return self;
        })
        .def("__exit__", [](PyEnableGradContext& self, py::object, py::object, py::object) {
            self.exit();
            return false;
        })
        .def("__call__", [](PyEnableGradContext&, py::function func) -> py::object {
            auto wrapper = py::cpp_function([func](py::args args, py::kwargs kwargs) -> py::object {
                bool prev = tenzor::is_grad_enabled();
                tenzor::set_grad_enabled(true);
                try {
                    py::object result = func(*args, **kwargs);
                    tenzor::set_grad_enabled(prev);
                    return result;
                } catch (...) {
                    tenzor::set_grad_enabled(prev);
                    throw;
                }
            });
            try {
                py::module_ functools = py::module_::import("functools");
                functools.attr("update_wrapper")(wrapper, func);
            } catch (...) {}
            return wrapper;
        }, py::arg("func"));

    // set_grad_enabled as context manager too
    struct PySetGradEnabledContext {
        bool mode_;
        bool prev_state_ = false;

        PySetGradEnabledContext(bool mode) : mode_(mode) {}

        void enter() {
            prev_state_ = tenzor::is_grad_enabled();
            tenzor::set_grad_enabled(mode_);
        }

        void exit() {
            tenzor::set_grad_enabled(prev_state_);
        }
    };

    // Keep the existing set_grad_enabled function, but also allow context manager usage

    // ========================================================================
    // Inference mode (stronger than no_grad — also skips version counters)
    // ========================================================================
    struct PyInferenceModeContext {
        std::unique_ptr<tenzor::InferenceModeGuard> guard_;

        void enter() {
            guard_ = std::make_unique<tenzor::InferenceModeGuard>();
        }

        void exit() {
            guard_.reset();
        }
    };

    py::class_<PyInferenceModeContext>(m, "inference_mode",
        "Context manager and decorator for inference mode.\n\n"
        "Stronger than no_grad(): disables gradient computation AND\n"
        "skips version counter increments for in-place ops.\n\n"
        "Usage as context manager:\n"
        "    with tz.inference_mode():\n"
        "        y = model(x)\n\n"
        "Usage as decorator:\n"
        "    @tz.inference_mode()\n"
        "    def predict(x):\n"
        "        return model(x)")
        .def(py::init<>())
        .def("__enter__", [](PyInferenceModeContext& self) -> PyInferenceModeContext& {
            self.enter();
            return self;
        })
        .def("__exit__", [](PyInferenceModeContext& self, py::object, py::object, py::object) {
            self.exit();
            return false;
        })
        .def("__call__", [](PyInferenceModeContext&, py::function func) -> py::object {
            auto wrapper = py::cpp_function([func](py::args args, py::kwargs kwargs) -> py::object {
                tenzor::InferenceModeGuard guard;
                return func(*args, **kwargs);
            });
            try {
                py::module_ functools = py::module_::import("functools");
                functools.attr("update_wrapper")(wrapper, func);
            } catch (py::error_already_set&) {
                PyErr_Clear();
            }
            return wrapper;
        }, py::arg("func"));

    m.def("is_inference_mode", &tenzor::is_inference_mode_enabled,
          "Check if inference mode is currently active");

    // ========================================================================
    // Anomaly detection (NaN/Inf checking in backward)
    // ========================================================================
    m.def("set_anomaly_detection", &tenzor::set_anomaly_detection,
          py::arg("enabled"),
          "Enable or disable anomaly detection (NaN/Inf checking) in backward passes");

    m.def("is_anomaly_detection_enabled", &tenzor::is_anomaly_detection_enabled,
          "Check if anomaly detection is enabled");

    struct PyAnomalyModeContext {
        std::unique_ptr<tenzor::AnomalyMode> guard_;
        bool enabled_;

        PyAnomalyModeContext(bool enabled = true) : enabled_(enabled) {}

        void enter() {
            guard_ = std::make_unique<tenzor::AnomalyMode>(enabled_);
        }

        void exit() {
            guard_.reset();
        }
    };

    py::class_<PyAnomalyModeContext>(m, "detect_anomaly",
        "Context manager for anomaly detection in backward passes.\n"
        "When enabled, checks all computed gradients for NaN/Inf values\n"
        "and throws an error identifying the responsible autograd function.")
        .def(py::init<bool>(), py::arg("enabled") = true)
        .def("__enter__", [](PyAnomalyModeContext& self) -> PyAnomalyModeContext& {
            self.enter();
            return self;
        })
        .def("__exit__", [](PyAnomalyModeContext& self, py::object, py::object, py::object) {
            self.exit();
        });

    // ========================================================================
    // Automatic Gradient Checkpointing
    // ========================================================================

    py::enum_<tenzor::autograd::CheckpointStrategy>(m, "CheckpointStrategy",
        "Strategy for automatic gradient checkpoint placement")
        .value("none", tenzor::autograd::CheckpointStrategy::None,
               "No automatic checkpointing")
        .value("every_n", tenzor::autograd::CheckpointStrategy::EveryN,
               "Checkpoint every N layers")
        .value("sqrt_n", tenzor::autograd::CheckpointStrategy::SqrtN,
               "Checkpoint every sqrt(N) layers (optimal for sequential models)")
        .value("memory_budget", tenzor::autograd::CheckpointStrategy::MemoryBudget,
               "Checkpoint to stay within a parameter memory budget per segment");

    py::class_<tenzor::autograd::AutoCheckpointPolicy,
               std::shared_ptr<tenzor::autograd::AutoCheckpointPolicy>>(
        m, "AutoCheckpointPolicy",
        "Automatic gradient checkpointing policy for memory-efficient training")
        .def(py::init<tenzor::autograd::CheckpointStrategy, int, size_t>(),
             py::arg("strategy") = tenzor::autograd::CheckpointStrategy::SqrtN,
             py::arg("every_n") = 0,
             py::arg("memory_budget_bytes") = 0)
        .def("apply", &tenzor::autograd::AutoCheckpointPolicy::apply,
             py::arg("module"), "Apply checkpointing policy to a module")
        .def("remove", &tenzor::autograd::AutoCheckpointPolicy::remove,
             py::arg("module"), "Remove checkpointing policy from a module")
        .def_property_readonly("strategy",
             &tenzor::autograd::AutoCheckpointPolicy::strategy);

    m.def("enable_auto_checkpoint",
        [](tenzor::nn::Module& module, const std::string& strategy,
           int every_n, size_t memory_budget_bytes) {
            tenzor::autograd::CheckpointStrategy strat;
            if (strategy == "sqrt_n" || strategy == "sqrtn")
                strat = tenzor::autograd::CheckpointStrategy::SqrtN;
            else if (strategy == "every_n" || strategy == "everyn")
                strat = tenzor::autograd::CheckpointStrategy::EveryN;
            else if (strategy == "memory_budget")
                strat = tenzor::autograd::CheckpointStrategy::MemoryBudget;
            else if (strategy == "none")
                strat = tenzor::autograd::CheckpointStrategy::None;
            else
                throw std::invalid_argument("Unknown strategy: " + strategy +
                    ". Use 'sqrt_n', 'every_n', 'memory_budget', or 'none'");
            return tenzor::autograd::enable_auto_checkpoint(
                module, strat, every_n, memory_budget_bytes);
        },
        py::arg("module"),
        py::arg("strategy") = "sqrt_n",
        py::arg("every_n") = 0,
        py::arg("memory_budget_bytes") = 0,
        "Enable automatic gradient checkpointing on a model.\n\n"
        "Args:\n"
        "    module: Model to checkpoint\n"
        "    strategy: 'sqrt_n', 'every_n', 'memory_budget', or 'none'\n"
        "    every_n: Interval for every_n strategy\n"
        "    memory_budget_bytes: Budget for memory_budget strategy\n"
        "Returns:\n"
        "    AutoCheckpointPolicy handle (keep reference alive to maintain hooks)");

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

    // PyCustomFunction bridges Python custom Functions to C++ autograd graph
    struct PyCustomFunction : public tenzor::Function {
        py::object py_forward_fn_;  // Python static forward function
        py::object py_backward_fn_; // Python static backward function
        std::shared_ptr<PyFunctionCtx> ctx_;

        PyCustomFunction(py::object forward_fn, py::object backward_fn)
            : py_forward_fn_(std::move(forward_fn)),
              py_backward_fn_(std::move(backward_fn)),
              ctx_(std::make_shared<PyFunctionCtx>()) {}

        auto forward(std::vector<tenzor::Variable> inputs) -> std::vector<tenzor::Variable> override {
            py::gil_scoped_acquire acquire;
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
            } catch (py::error_already_set&) {
                throw;  // Preserves Python traceback
            }
        }

        auto backward(std::vector<tenzor::Tensor> grad_outputs) -> std::vector<tenzor::Tensor> override {
            py::gil_scoped_acquire acquire;
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
            } catch (py::error_already_set&) {
                throw;  // Preserves Python traceback
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

    // The Python-facing autograd.Function class
    // Usage: class MyFunc(tenzor.autograd.Function):
    //     @staticmethod
    //     def forward(ctx, input): ...
    //     @staticmethod
    //     def backward(ctx, grad_output): ...
    //
    // result = MyFunc.apply(input_var)
    auto autograd_mod = m.def_submodule("autograd", "Autograd components");

    auto custom_func_cls = py::class_<PyCustomFunction, std::shared_ptr<PyCustomFunction>>(
        autograd_mod, "_CustomFunctionImpl");
    (void)custom_func_cls;  // Registration side-effect only

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

    // autograd.grad() - functional gradient computation
    autograd_mod.def("grad", [](
        py::object outputs_obj,
        py::object inputs_obj,
        py::object grad_outputs_obj,
        bool retain_graph,
        bool create_graph
    ) -> py::tuple {
        // Normalize outputs to a list
        std::vector<tenzor::Variable> outputs;
        if (py::isinstance<tenzor::Variable>(outputs_obj)) {
            outputs.push_back(outputs_obj.cast<tenzor::Variable>());
        } else {
            for (auto o : outputs_obj.cast<py::sequence>()) {
                outputs.push_back(o.cast<tenzor::Variable>());
            }
        }

        // Normalize inputs to a list
        std::vector<tenzor::Variable> inputs;
        if (py::isinstance<tenzor::Variable>(inputs_obj)) {
            inputs.push_back(inputs_obj.cast<tenzor::Variable>());
        } else {
            for (auto i : inputs_obj.cast<py::sequence>()) {
                inputs.push_back(i.cast<tenzor::Variable>());
            }
        }

        // Zero gradients on inputs first
        for (auto& inp : inputs) {
            inp.zero_grad();
            inp.retain_grad();
        }

        // Run backward for each output
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

        // Collect gradients
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

    // autograd.make_dot() - computation graph visualization (Graphviz DOT format)
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

    // autograd.optimize_graph() - graph optimization
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
    "Optimize the computation graph of a Variable. Returns optimization stats dict.\n"
    "This is opt-in and does NOT run automatically during backward().");

    // ========================================================================
    // Composable function transforms (torch.func equivalent)
    // ========================================================================
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
    "Return a function that computes the gradient of f.\n"
    "f must be a scalar-valued function of a single Variable.");

    func_mod.def("vmap", [](py::function f, int64_t in_dim, int64_t out_dim) {
        return py::cpp_function([f, in_dim](const tenzor::Variable& batched_input) -> tenzor::Variable {
            py::gil_scoped_acquire gil;
            auto cpp_fn = [&f](const tenzor::Variable& x) -> tenzor::Variable {
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
                py::object result = f(input);
                return result.cast<tenzor::Variable>();
            };
            tenzor::Tensor H = tenzor::hessian(cpp_fn, x);
            return tenzor::Variable(H, false);
        });
    }, py::arg("f"),
    "Return a function that computes the Hessian of a scalar-valued f.");

    // Neural network
    auto nn = m.def_submodule("nn", "Neural network components");

    // ========================================================================
    // Module base class with full Python subclassing support
    // ========================================================================
    // Uses PyModule trampoline to enable Python users to create custom modules.
    // Python users can:
    //   1. Subclass tz.nn.Module
    //   2. Call super().__init__() in __init__
    //   3. Override forward_impl() OR forward() method
    //   4. Use register_parameter/buffer/module for explicit registration
    //   5. Store submodules as attributes for automatic discovery

    py::class_<tenzor::nn::Module, PyModule, std::shared_ptr<tenzor::nn::Module>>(nn, "Module")
        // Default constructor for Python subclasses
        .def(py::init<>())

        // ====================================================================
        // Forward pass methods
        // ====================================================================
        .def("forward", &tenzor::nn::Module::forward,
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Perform forward pass (calls forward_impl with hooks)")
        .def("forward_impl", &tenzor::nn::Module::forward_impl,
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Implementation of forward pass - override this in subclasses")
        .def("extra_repr", &tenzor::nn::Module::extra_repr,
             "Extra representation string for __repr__ — override in subclasses")
        .def("__call__", &tenzor::nn::Module::operator(),
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Callable interface - equivalent to forward()")

        // ====================================================================
        // Parameter and buffer management
        // ====================================================================
        .def("parameters", &tenzor::nn::Module::parameters,
             "Get all parameters (recursive through submodules)")
        .def("own_parameters", &tenzor::nn::Module::own_parameters,
             "Get only this module's direct parameters (not submodules')")
        .def("named_parameters", &tenzor::nn::Module::named_parameters,
             "Get all parameters with their names")
        .def("buffers", &tenzor::nn::Module::buffers,
             "Get all buffers (non-trainable tensors, recursive)")
        .def("own_buffers", &tenzor::nn::Module::own_buffers,
             "Get only this module's direct buffers")
        .def("named_buffers", &tenzor::nn::Module::named_buffers,
             "Get all buffers with their names")
        .def("get_submodules", &tenzor::nn::Module::get_submodules,
             "Get direct submodules as a dictionary")

        // ====================================================================
        // Parameter/buffer/module registration (exposed from protected via lambda)
        // ====================================================================
        .def("register_parameter", [](tenzor::nn::Module& self, const std::string& name, tenzor::Variable param) {
            auto* pymod = dynamic_cast<PyModule*>(&self);
            if (!pymod) throw std::runtime_error("register_parameter requires a Python-subclassed Module");
            pymod->py_register_parameter(name, std::move(param));
        }, py::arg("name"), py::arg("param"),
             "Register a trainable parameter")
        .def("register_buffer", [](tenzor::nn::Module& self, const std::string& name, tenzor::Variable buffer) {
            auto* pymod = dynamic_cast<PyModule*>(&self);
            if (!pymod) throw std::runtime_error("register_buffer requires a Python-subclassed Module");
            pymod->py_register_buffer(name, std::move(buffer));
        }, py::arg("name"), py::arg("buffer"),
             "Register a non-trainable buffer (e.g., running stats)")
        .def("register_module", [](tenzor::nn::Module& self, const std::string& name, std::shared_ptr<tenzor::nn::Module> module) {
            auto* pymod = dynamic_cast<PyModule*>(&self);
            if (!pymod) throw std::runtime_error("register_module requires a Python-subclassed Module");
            pymod->py_register_module(name, std::move(module));
        }, py::arg("name"), py::arg("module"),
             "Register a child module")

        // ====================================================================
        // Individual parameter/buffer access and unregistration
        // ====================================================================
        .def("get_parameter", &tenzor::nn::Module::get_parameter,
             py::arg("name"),
             "Get a single parameter by name")
        .def("get_buffer", &tenzor::nn::Module::get_buffer,
             py::arg("name"),
             "Get a single buffer by name")
        .def("unregister_parameter", &tenzor::nn::Module::unregister_parameter,
             py::arg("name"),
             "Remove a registered parameter by name")
        .def("unregister_buffer", &tenzor::nn::Module::unregister_buffer,
             py::arg("name"),
             "Remove a registered buffer by name")
        .def("unregister_module", &tenzor::nn::Module::unregister_module,
             py::arg("name"),
             "Remove a registered submodule by name")

        // ====================================================================
        // Training mode
        // ====================================================================
        .def("train", &tenzor::nn::Module::train,
             py::arg("mode") = true,
             "Set training mode (affects dropout, batchnorm, etc.)")
        .def("eval", &tenzor::nn::Module::eval,
             "Set evaluation mode - equivalent to train(False)")
        .def("is_training", &tenzor::nn::Module::is_training,
             "Check if module is in training mode")
        .def_property_readonly("training", &tenzor::nn::Module::is_training,
             "Training mode flag (read-only property)")

        // ====================================================================
        // Device management
        // ====================================================================
        .def("to", py::overload_cast<tenzor::Device>(&tenzor::nn::Module::to),
             py::arg("device"),
             "Move module to specified device",
             py::call_guard<py::gil_scoped_release>())
        .def("to", py::overload_cast<tenzor::DType>(&tenzor::nn::Module::to),
             py::arg("dtype"),
             "Convert module parameters to specified dtype",
             py::call_guard<py::gil_scoped_release>())
        // String device overload for PyTorch compatibility
        .def("to", [](tenzor::nn::Module& self, const std::string& device) {
            if (device == "cpu") {
                self.cpu();
            } else if (device == "cuda" || device.rfind("cuda:", 0) == 0) {
                int device_id = 0;
                if (device.size() > 5) {
                    try {
                        device_id = std::stoi(device.substr(5));
                    } catch (const std::exception&) {
                        throw std::runtime_error("Invalid CUDA device ID in: " + device);
                    }
                }
                self.cuda(device_id);
            } else {
                throw std::runtime_error("Unknown device: " + device);
            }
        }, py::arg("device"),
             "Move module to device specified by string ('cpu', 'cuda', 'cuda:0')",
             py::call_guard<py::gil_scoped_release>())
        .def("cuda", &tenzor::nn::Module::cuda,
             py::arg("device_id") = 0,
             "Move module to CUDA device",
             py::call_guard<py::gil_scoped_release>())
        .def("cpu", &tenzor::nn::Module::cpu,
             "Move module to CPU",
             py::call_guard<py::gil_scoped_release>())

        // ====================================================================
        // Gradient management
        // ====================================================================
        .def("zero_grad", &tenzor::nn::Module::zero_grad,
             "Zero all parameter gradients")
        .def("requires_grad_", [](tenzor::nn::Module& self, bool requires_grad) {
            for (auto& param : self.parameters()) {
                param->set_requires_grad(requires_grad);
            }
            return &self;  // Return self for chaining
        }, py::arg("requires_grad") = true, py::return_value_policy::reference,
           "Set requires_grad for all parameters in-place")

        // ====================================================================
        // Serialization
        // ====================================================================
        .def("state_dict", &tenzor::nn::Module::state_dict,
             "Get module state as dictionary (parameters + buffers)")
        .def("load_state_dict", py::overload_cast<const std::unordered_map<std::string, tenzor::Tensor>&, bool>(
             &tenzor::nn::Module::load_state_dict),
             py::arg("state"), py::arg("strict") = true,
             "Load module state from dictionary. If strict=True (default), throws on missing/unexpected keys.")
        .def("save", &tenzor::nn::Module::save,
             py::arg("path"),
             "Save module to file")
        .def("load", &tenzor::nn::Module::load,
             py::arg("path"),
             "Load module from file")

        // ====================================================================
        // Hook system (PyTorch-compatible naming)
        // ====================================================================
        // Forward hooks - called after forward pass with input and output
        .def("register_forward_hook", [](tenzor::nn::Module& self, py::object hook) {
            py::object hook_ref = hook;
            return self.register_forward_post_hook([hook_ref](tenzor::nn::Module* m, const tenzor::Variable& input, const tenzor::Variable& output) {
                py::gil_scoped_acquire acquire;
                try {
                    hook_ref(m, input, output);
                } catch (py::error_already_set&) {
                    throw;
                }
            });
        }, py::arg("hook"),
           "Register hook called after forward pass (PyTorch-compatible)")
        .def("register_forward_pre_hook", [](tenzor::nn::Module& self, py::object hook) {
            py::object hook_ref = hook;
            return self.register_forward_pre_hook([hook_ref](tenzor::nn::Module* m, const tenzor::Variable& input) {
                py::gil_scoped_acquire acquire;
                try {
                    hook_ref(m, input);
                } catch (py::error_already_set&) {
                    throw;
                }
            });
        }, py::arg("hook"),
           "Register hook called before forward pass")
        .def("register_backward_hook", [](tenzor::nn::Module& self, py::object hook) {
            py::object hook_ref = hook;
            return self.register_backward_post_hook([hook_ref](tenzor::nn::Module* m, const tenzor::Variable& grad_input, const tenzor::Variable& grad_output) {
                py::gil_scoped_acquire acquire;
                try {
                    hook_ref(m, grad_input, grad_output);
                } catch (py::error_already_set&) {
                    throw;
                }
            });
        }, py::arg("hook"),
           "Register hook called after backward pass (PyTorch-compatible)")
        .def("register_full_backward_hook", [](tenzor::nn::Module& self, py::object hook) {
            py::object hook_ref = hook;
            return self.register_backward_post_hook([hook_ref](tenzor::nn::Module* m, const tenzor::Variable& grad_input, const tenzor::Variable& grad_output) {
                py::gil_scoped_acquire acquire;
                try {
                    hook_ref(m, grad_input, grad_output);
                } catch (py::error_already_set&) {
                    throw;
                }
            });
        }, py::arg("hook"),
           "Register hook called after backward pass with full gradients")
        .def("register_full_backward_pre_hook", [](tenzor::nn::Module& self, py::object hook) {
            py::object hook_ref = hook;
            return self.register_backward_pre_hook([hook_ref](tenzor::nn::Module* m, const tenzor::Variable& grad_output) {
                py::gil_scoped_acquire acquire;
                try {
                    hook_ref(m, grad_output);
                } catch (py::error_already_set&) {
                    throw;
                }
            });
        }, py::arg("hook"),
           "Register hook called before backward pass")
        .def("register_forward_post_hook", [](tenzor::nn::Module& self, py::object hook) {
            py::object hook_ref = hook;
            return self.register_forward_post_hook([hook_ref](tenzor::nn::Module* m, const tenzor::Variable& input, const tenzor::Variable& output) {
                py::gil_scoped_acquire acquire;
                try {
                    hook_ref(m, input, output);
                } catch (py::error_already_set&) {
                    throw;
                }
            });
        }, py::arg("hook"),
           "Register hook called after forward pass")
        .def("remove_hook", &tenzor::nn::Module::remove_hook,
             py::arg("hook_id"),
             "Remove a registered hook by ID")

        // ====================================================================
        // Python-friendly utilities
        // ====================================================================
        .def("__repr__", [](const tenzor::nn::Module& self) {
            auto params = const_cast<tenzor::nn::Module&>(self).parameters();
            size_t total_params = 0;
            for (const auto& p : params) {
                total_params += p->tensor().numel();
            }
            return "Module(parameters=" + std::to_string(total_params) + ")";
        })
        .def("num_parameters", [](tenzor::nn::Module& self) {
            size_t total = 0;
            for (const auto& p : self.parameters()) {
                total += p->tensor().numel();
            }
            return total;
        }, "Count total number of parameters")
        .def("num_trainable_parameters", [](tenzor::nn::Module& self) {
            size_t total = 0;
            for (const auto& p : self.parameters()) {
                if (p->requires_grad()) {
                    total += p->tensor().numel();
                }
            }
            return total;
        }, "Count number of trainable parameters")

        // ====================================================================
        // Attribute-based submodule discovery (PyTorch-like behavior)
        // ====================================================================
        // This allows Python users to store modules as attributes and have
        // them automatically discovered by parameters() and state_dict()
        .def("_register_submodule_from_attr", [](tenzor::nn::Module& self,
                                                  const std::string& name,
                                                  std::shared_ptr<tenzor::nn::Module> module) {
            // Helper for __setattr__ to auto-register modules
            // Use dynamic_cast for safety in case of non-PyModule instances
            auto* pymod = dynamic_cast<PyModule*>(&self);
            if (pymod) {
                pymod->py_register_module(name, std::move(module));
            }
        }, py::arg("name"), py::arg("module"));

    py::class_<tenzor::nn::Linear, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Linear>>(nn, "Linear")
        .def(py::init<int64_t, int64_t, bool>(),
             py::arg("in_features"), py::arg("out_features"),
             py::arg("bias") = true)
        .def("weight", [](const tenzor::nn::Linear& self) {
            return self.weight();
        }, py::return_value_policy::reference_internal)
        .def("bias", [](const tenzor::nn::Linear& self) {
            return self.bias();
        })
        .def_property_readonly("has_bias", &tenzor::nn::Linear::has_bias)
        .def("__repr__", [](const tenzor::nn::Linear& self) {
            auto params = const_cast<tenzor::nn::Linear&>(self).own_parameters();
            int64_t in_f = 0, out_f = 0;
            if (!params.empty()) {
                auto shape = params[0]->tensor().shape();
                if (shape.size() >= 2) {
                    out_f = shape[0];
                    in_f = shape[1];
                }
            }
            bool has_bias = params.size() > 1;
            return "Linear(in_features=" + std::to_string(in_f) +
                   ", out_features=" + std::to_string(out_f) +
                   ", bias=" + (has_bias ? "True" : "False") + ")";
        });

    py::class_<tenzor::nn::LazyLinear, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::LazyLinear>>(nn, "LazyLinear")
        .def(py::init<int64_t, bool>(),
             py::arg("out_features"),
             py::arg("bias") = true)
        .def("is_materialized", &tenzor::nn::LazyLinear::is_materialized)
        .def("__repr__", [](const tenzor::nn::LazyLinear& self) {
            auto& mut_self = const_cast<tenzor::nn::LazyLinear&>(self);
            if (!self.is_materialized()) {
                auto params = mut_self.own_parameters();
                return std::string("LazyLinear(in_features=<not materialized>, out_features=?, bias=?)");
            }
            auto params = mut_self.own_parameters();
            int64_t in_f = 0, out_f = 0;
            if (!params.empty()) {
                auto shape = params[0]->tensor().shape();
                if (shape.size() >= 2) {
                    out_f = shape[0];
                    in_f = shape[1];
                }
            }
            bool has_bias = params.size() > 1;
            return "LazyLinear(in_features=" + std::to_string(in_f) +
                   ", out_features=" + std::to_string(out_f) +
                   ", bias=" + (has_bias ? "True" : "False") + ")";
        });

    // Convolution layers
    py::class_<tenzor::nn::Conv2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Conv2d>>(nn, "Conv2d")
        // Square kernel constructor (int args)
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size"),
             py::arg("stride") = 1,
             py::arg("padding") = 0,
             py::arg("dilation") = 1,
             py::arg("groups") = 1,
             py::arg("bias") = true)
        // Non-square kernel constructor (tuple args)
        .def(py::init<int64_t, int64_t,
                       std::pair<int64_t, int64_t>,
                       std::pair<int64_t, int64_t>,
                       std::pair<int64_t, int64_t>,
                       std::pair<int64_t, int64_t>,
                       int64_t, bool>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size"),
             py::arg("stride") = std::make_pair<int64_t,int64_t>(1, 1),
             py::arg("padding") = std::make_pair<int64_t,int64_t>(0, 0),
             py::arg("dilation") = std::make_pair<int64_t,int64_t>(1, 1),
             py::arg("groups") = 1,
             py::arg("bias") = true)
        .def_property_readonly("stride_h", &tenzor::nn::Conv2d::stride_h)
        .def_property_readonly("stride_w", &tenzor::nn::Conv2d::stride_w)
        .def_property_readonly("padding_h", &tenzor::nn::Conv2d::padding_h)
        .def_property_readonly("padding_w", &tenzor::nn::Conv2d::padding_w)
        .def_property_readonly("dilation_h", &tenzor::nn::Conv2d::dilation_h)
        .def_property_readonly("dilation_w", &tenzor::nn::Conv2d::dilation_w)
        .def_property_readonly("groups", &tenzor::nn::Conv2d::groups)
        .def("__repr__", [](const tenzor::nn::Conv2d& self) {
            auto params = const_cast<tenzor::nn::Conv2d&>(self).own_parameters();
            int64_t in_c = 0, out_c = 0, kh = 0, kw = 0;
            if (!params.empty()) {
                auto shape = params[0]->tensor().shape();
                if (shape.size() >= 4) {
                    out_c = shape[0];
                    in_c = shape[1];
                    kh = shape[2];
                    kw = shape[3];
                }
            }
            bool has_bias = params.size() > 1;
            return "Conv2d(" + std::to_string(in_c) + ", " + std::to_string(out_c) +
                   ", kernel_size=(" + std::to_string(kh) + ", " + std::to_string(kw) + ")" +
                   ", bias=" + (has_bias ? "True" : "False") + ")";
        });

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
             py::arg("bias") = true)
        .def("__repr__", [](const tenzor::nn::Conv1d& self) {
            auto params = const_cast<tenzor::nn::Conv1d&>(self).own_parameters();
            int64_t in_c = 0, out_c = 0, k = 0;
            if (!params.empty()) {
                auto shape = params[0]->tensor().shape();
                if (shape.size() >= 3) { out_c = shape[0]; in_c = shape[1]; k = shape[2]; }
            }
            return "Conv1d(" + std::to_string(in_c) + ", " + std::to_string(out_c) +
                   ", kernel_size=" + std::to_string(k) + ")";
        });

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

    py::class_<tenzor::nn::Conv3d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Conv3d>>(nn, "Conv3d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size"),
             py::arg("stride") = 1,
             py::arg("padding") = 0,
             py::arg("dilation") = 1,
             py::arg("groups") = 1,
             py::arg("bias") = true);

    py::class_<tenzor::nn::ConvTranspose3d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ConvTranspose3d>>(nn, "ConvTranspose3d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size"),
             py::arg("stride") = 1,
             py::arg("padding") = 0,
             py::arg("output_padding") = 0,
             py::arg("dilation") = 1,
             py::arg("groups") = 1,
             py::arg("bias") = true);

    py::class_<tenzor::nn::ConvTranspose1d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ConvTranspose1d>>(nn, "ConvTranspose1d")
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
             py::arg("track_running_stats") = true)
        .def("__repr__", [](const tenzor::nn::BatchNorm2d& self) {
            auto params = const_cast<tenzor::nn::BatchNorm2d&>(self).own_parameters();
            int64_t num_f = params.empty() ? 0 : params[0]->tensor().numel();
            return "BatchNorm2d(" + std::to_string(num_f) + ")";
        });

    py::class_<tenzor::nn::BatchNorm3d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::BatchNorm3d>>(nn, "BatchNorm3d")
        .def(py::init<int64_t, double, double, bool, bool>(),
             py::arg("num_features"),
             py::arg("eps") = 1e-5,
             py::arg("momentum") = 0.1,
             py::arg("affine") = true,
             py::arg("track_running_stats") = true)
        .def("__repr__", [](const tenzor::nn::BatchNorm3d& self) {
            auto params = const_cast<tenzor::nn::BatchNorm3d&>(self).own_parameters();
            int64_t num_f = params.empty() ? 0 : params[0]->tensor().numel();
            return "BatchNorm3d(" + std::to_string(num_f) + ")";
        });

    py::class_<tenzor::nn::SyncBatchNorm, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::SyncBatchNorm>>(nn, "SyncBatchNorm",
               "Synchronized Batch Normalization across distributed processes.\n"
               "Synchronizes mean/variance via an all-reduce callback.")
        .def(py::init<int64_t, tenzor::nn::AllReduceFn, int, double, double, bool, bool>(),
             py::arg("num_features"),
             py::arg("all_reduce_fn"),
             py::arg("world_size") = 1,
             py::arg("eps") = 1e-5,
             py::arg("momentum") = 0.1,
             py::arg("affine") = true,
             py::arg("track_running_stats") = true)
        .def("__repr__", [](const tenzor::nn::SyncBatchNorm& self) {
            return "SyncBatchNorm(" + self.extra_repr() + ")";
        });

    py::class_<tenzor::nn::BatchNorm1d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::BatchNorm1d>>(nn, "BatchNorm1d")
        .def(py::init<int64_t, double, double, bool, bool>(),
             py::arg("num_features"),
             py::arg("eps") = 1e-5,
             py::arg("momentum") = 0.1,
             py::arg("affine") = true,
             py::arg("track_running_stats") = true)
        .def("__repr__", [](const tenzor::nn::BatchNorm1d& self) {
            auto params = const_cast<tenzor::nn::BatchNorm1d&>(self).own_parameters();
            int64_t num_f = params.empty() ? 0 : params[0]->tensor().numel();
            return "BatchNorm1d(" + std::to_string(num_f) + ")";
        });

    py::class_<tenzor::nn::LayerNorm, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::LayerNorm>>(nn, "LayerNorm")
        .def(py::init<std::vector<int64_t>, double, bool>(),
             py::arg("normalized_shape"),
             py::arg("eps") = 1e-5,
             py::arg("elementwise_affine") = true)
        .def("__repr__", [](const tenzor::nn::LayerNorm& self) {
            auto params = const_cast<tenzor::nn::LayerNorm&>(self).own_parameters();
            int64_t size = params.empty() ? 0 : params[0]->tensor().numel();
            return "LayerNorm([" + std::to_string(size) + "])";
        });

    py::class_<tenzor::nn::GroupNorm, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::GroupNorm>>(nn, "GroupNorm")
        .def(py::init<int64_t, int64_t, double, bool>(),
             py::arg("num_groups"),
             py::arg("num_channels"),
             py::arg("eps") = 1e-5,
             py::arg("affine") = true);

    py::class_<tenzor::nn::InstanceNorm2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::InstanceNorm2d>>(nn, "InstanceNorm2d")
        .def(py::init<int64_t, double, bool>(),
             py::arg("num_features"),
             py::arg("eps") = 1e-5,
             py::arg("affine") = true)
        .def("__repr__", [](const tenzor::nn::InstanceNorm2d& self) {
            auto params = const_cast<tenzor::nn::InstanceNorm2d&>(self).own_parameters();
            int64_t size = params.empty() ? 0 : params[0]->tensor().numel();
            return "InstanceNorm2d(" + std::to_string(size) + ")";
        });

    py::class_<tenzor::nn::InstanceNorm3d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::InstanceNorm3d>>(nn, "InstanceNorm3d")
        .def(py::init<int64_t, double, bool>(),
             py::arg("num_features"),
             py::arg("eps") = 1e-5,
             py::arg("affine") = true)
        .def("__repr__", [](const tenzor::nn::InstanceNorm3d& self) {
            auto params = const_cast<tenzor::nn::InstanceNorm3d&>(self).own_parameters();
            int64_t size = params.empty() ? 0 : params[0]->tensor().numel();
            return "InstanceNorm3d(" + std::to_string(size) + ")";
        });

    py::class_<tenzor::nn::InstanceNorm1d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::InstanceNorm1d>>(nn, "InstanceNorm1d")
        .def(py::init<int64_t, double, bool>(),
             py::arg("num_features"),
             py::arg("eps") = 1e-5,
             py::arg("affine") = true)
        .def("__repr__", [](const tenzor::nn::InstanceNorm1d& self) {
            auto params = const_cast<tenzor::nn::InstanceNorm1d&>(self).own_parameters();
            int64_t size = params.empty() ? 0 : params[0]->tensor().numel();
            return "InstanceNorm1d(" + std::to_string(size) + ")";
        });

    py::class_<tenzor::nn::RMSNorm, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::RMSNorm>>(nn, "RMSNorm")
        .def(py::init<int64_t, double>(),
             py::arg("normalized_shape"),
             py::arg("eps") = 1e-6)
        .def("__repr__", [](const tenzor::nn::RMSNorm& self) {
            auto params = const_cast<tenzor::nn::RMSNorm&>(self).own_parameters();
            int64_t size = params.empty() ? 0 : params[0]->tensor().numel();
            return "RMSNorm(" + std::to_string(size) + ")";
        });

    // Regularization layers
    py::class_<tenzor::nn::Dropout, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Dropout>>(nn, "Dropout")
        .def(py::init<double>(),
             py::arg("p") = 0.5)
        .def("__repr__", [](const tenzor::nn::Dropout&) {
            return "Dropout()";
        });

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
        .def(py::init<int64_t, int64_t, int64_t, bool, bool>(),
             py::arg("kernel_size"),
             py::arg("stride") = -1,
             py::arg("padding") = 0,
             py::arg("ceil_mode") = false,
             py::arg("return_indices") = false);

    py::class_<tenzor::nn::AvgPool2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::AvgPool2d>>(nn, "AvgPool2d")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("kernel_size"),
             py::arg("stride") = -1,
             py::arg("padding") = 0);

    py::class_<tenzor::nn::MaxPool3d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::MaxPool3d>>(nn, "MaxPool3d")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool>(),
             py::arg("kernel_size"),
             py::arg("stride") = -1,
             py::arg("padding") = 0,
             py::arg("ceil_mode") = false,
             py::arg("return_indices") = false);

    py::class_<tenzor::nn::AvgPool3d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::AvgPool3d>>(nn, "AvgPool3d")
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

    py::class_<tenzor::nn::MaxPool1d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::MaxPool1d>>(nn, "MaxPool1d")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool>(),
             py::arg("kernel_size"),
             py::arg("stride") = -1,
             py::arg("padding") = 0,
             py::arg("ceil_mode") = false,
             py::arg("return_indices") = false);

    py::class_<tenzor::nn::AvgPool1d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::AvgPool1d>>(nn, "AvgPool1d")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("kernel_size"),
             py::arg("stride") = -1,
             py::arg("padding") = 0);

    py::class_<tenzor::nn::AdaptiveAvgPool1d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::AdaptiveAvgPool1d>>(nn, "AdaptiveAvgPool1d")
        .def(py::init<int64_t>(), py::arg("output_size"));

    py::class_<tenzor::nn::AdaptiveMaxPool2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::AdaptiveMaxPool2d>>(nn, "AdaptiveMaxPool2d")
        .def(py::init<int64_t, int64_t>(),
             py::arg("output_h"), py::arg("output_w"))
        .def(py::init<int64_t>(),
             py::arg("output_size"));

    py::class_<tenzor::nn::AdaptiveMaxPool1d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::AdaptiveMaxPool1d>>(nn, "AdaptiveMaxPool1d")
        .def(py::init<int64_t>(), py::arg("output_size"));

    py::class_<tenzor::nn::AdaptiveMaxPool3d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::AdaptiveMaxPool3d>>(nn, "AdaptiveMaxPool3d")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("output_d"), py::arg("output_h"), py::arg("output_w"))
        .def(py::init<int64_t>(),
             py::arg("output_size"));

    py::class_<tenzor::nn::AdaptiveAvgPool3d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::AdaptiveAvgPool3d>>(nn, "AdaptiveAvgPool3d")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("output_d"), py::arg("output_h"), py::arg("output_w"))
        .def(py::init<int64_t>(),
             py::arg("output_size"));

    // Utility layers
    py::class_<tenzor::nn::Flatten, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Flatten>>(nn, "Flatten")
        .def(py::init<int64_t, int64_t>(),
             py::arg("start_dim") = 1,
             py::arg("end_dim") = -1);

    py::class_<tenzor::nn::Identity, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Identity>>(nn, "Identity",
               "Identity layer — passes input through unchanged")
        .def(py::init<>());

    // Sequential container
    // NOTE: To support Python-defined modules, we can't use constructor with variadic
    // args because temporary Python objects may be GC'd. Use append() method instead
    // for inline module creation, or store references before passing.
    py::class_<tenzor::nn::Sequential, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Sequential>>(nn, "Sequential")
        .def(py::init<>(),
             "Create an empty Sequential container")
        .def(py::init([](py::list modules) {
            auto seq = std::make_shared<tenzor::nn::Sequential>();
            // Store list in the Sequential for reference counting
            for (size_t i = 0; i < modules.size(); ++i) {
                py::object mod_obj = modules[i];
                auto mod_ptr = py::cast<std::shared_ptr<tenzor::nn::Module>>(mod_obj);
                seq->add_module(mod_ptr);
            }
            return seq;
        }), py::arg("modules"),
           "Create Sequential container from list of modules")
        .def("add_module", &tenzor::nn::Sequential::add_module,
             py::return_value_policy::reference_internal,
             py::arg("module"),
             "Add a module to the sequential container")
        .def("__len__", [](const tenzor::nn::Sequential& self) {
            return self.get_submodules().size();
        }, "Return number of modules in sequence")
        .def("__getitem__", [](tenzor::nn::Sequential& self, size_t idx) {
            auto& submodules = self.get_submodules();
            std::string key = "module_" + std::to_string(idx);
            auto it = submodules.find(key);
            if (it == submodules.end()) {
                throw py::index_error("Index out of range");
            }
            return it->second;
        }, py::arg("index"), "Get module at index")
        .def("append", [](tenzor::nn::Sequential& self, std::shared_ptr<tenzor::nn::Module> module) {
            self.add_module(module);
            return; // No chaining for append (PyTorch-compatible)
        }, py::arg("module"), "Append a module to the sequence");

    // ModuleList - a list container for modules (like PyTorch's nn.ModuleList)
    py::class_<tenzor::nn::ModuleList, tenzor::nn::Module, std::shared_ptr<tenzor::nn::ModuleList>>(nn, "ModuleList")
        .def(py::init<>(), "Create an empty ModuleList")
        .def(py::init([](py::list modules) {
            auto ml = std::make_shared<tenzor::nn::ModuleList>();
            for (auto module : modules) {
                ml->append(module.cast<std::shared_ptr<tenzor::nn::Module>>());
            }
            return ml;
        }), py::arg("modules"), "Create ModuleList from list of modules")
        .def("append", [](tenzor::nn::ModuleList& self, std::shared_ptr<tenzor::nn::Module> module) {
            self.append(module);
        }, py::arg("module"), "Append a module to the list")
        .def("extend", [](tenzor::nn::ModuleList& self, py::list modules) {
            for (auto module : modules) {
                self.append(module.cast<std::shared_ptr<tenzor::nn::Module>>());
            }
        }, py::arg("modules"), "Extend with a list of modules")
        .def("__len__", &tenzor::nn::ModuleList::size, "Return number of modules")
        .def("__getitem__", &tenzor::nn::ModuleList::at, py::arg("index"), "Get module at index")
        .def("__iter__", [](tenzor::nn::ModuleList& self) {
            return py::make_iterator(self.begin(), self.end());
        }, py::keep_alive<0, 1>(), "Iterate over modules");

    // ModuleDict - a dictionary container for modules (like PyTorch's nn.ModuleDict)
    py::class_<tenzor::nn::ModuleDict, tenzor::nn::Module, std::shared_ptr<tenzor::nn::ModuleDict>>(nn, "ModuleDict")
        .def(py::init<>(), "Create an empty ModuleDict")
        .def(py::init([](py::dict modules) {
            auto md = std::make_shared<tenzor::nn::ModuleDict>();
            for (auto item : modules) {
                md->insert(item.first.cast<std::string>(),
                          item.second.cast<std::shared_ptr<tenzor::nn::Module>>());
            }
            return md;
        }), py::arg("modules"), "Create ModuleDict from dictionary of modules")
        .def("__getitem__", &tenzor::nn::ModuleDict::at, py::arg("key"), "Get module by key")
        .def("__setitem__", [](tenzor::nn::ModuleDict& self, const std::string& key,
                               std::shared_ptr<tenzor::nn::Module> module) {
            self.insert(key, module);
        }, py::arg("key"), py::arg("module"), "Set module at key")
        .def("__delitem__", &tenzor::nn::ModuleDict::erase, py::arg("key"), "Delete module at key")
        .def("__len__", &tenzor::nn::ModuleDict::size, "Return number of modules")
        .def("__contains__", &tenzor::nn::ModuleDict::contains, py::arg("key"), "Check if key exists")
        .def("__iter__", [](tenzor::nn::ModuleDict& self) {
            auto keys = self.keys();
            return py::make_iterator(keys.begin(), keys.end());
        }, py::keep_alive<0, 1>(), "Iterate over keys")
        .def("keys", &tenzor::nn::ModuleDict::keys, "Get all keys in insertion order")
        .def("values", &tenzor::nn::ModuleDict::values, "Get all modules in insertion order")
        .def("items", &tenzor::nn::ModuleDict::items, "Get all (key, module) pairs in insertion order");

    // Activation function classes
    py::class_<tenzor::nn::ReLU, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ReLU>>(nn, "ReLU")
        .def(py::init<>())
        .def("__repr__", [](const tenzor::nn::ReLU&) { return "ReLU()"; });

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

    auto swish_class = py::class_<tenzor::nn::Swish, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Swish>>(nn, "Swish")
        .def(py::init<>());

    // SiLU is an alias for Swish (same activation function)
    nn.attr("SiLU") = swish_class;

    py::class_<tenzor::nn::Mish, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Mish>>(nn, "Mish")
        .def(py::init<>());

    py::class_<tenzor::nn::ReLU6, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ReLU6>>(nn, "ReLU6")
        .def(py::init<>())
        .def("__repr__", [](const tenzor::nn::ReLU6&) { return "ReLU6()"; });

    py::class_<tenzor::nn::PReLU, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::PReLU>>(nn, "PReLU")
        .def(py::init<int64_t, double>(),
             py::arg("num_parameters") = 1, py::arg("init") = 0.25)
        .def("__repr__", [](const tenzor::nn::PReLU& self) {
            return "PReLU()";
        });

    py::class_<tenzor::nn::Hardswish, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Hardswish>>(nn, "Hardswish")
        .def(py::init<>())
        .def("__repr__", [](const tenzor::nn::Hardswish&) { return "Hardswish()"; });

    py::class_<tenzor::nn::Hardsigmoid, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Hardsigmoid>>(nn, "Hardsigmoid")
        .def(py::init<>())
        .def("__repr__", [](const tenzor::nn::Hardsigmoid&) { return "Hardsigmoid()"; });

    // Functional activations
    nn.def("hardswish", [](const tenzor::Variable& input) {
        return tenzor::nn::hardswish(input);
    }, "Functional Hardswish activation", py::arg("input"));

    nn.def("hardsigmoid", [](const tenzor::Variable& input) {
        return tenzor::nn::hardsigmoid(input);
    }, "Functional Hardsigmoid activation", py::arg("input"));

    // GLU
    py::class_<tenzor::nn::GLU, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::GLU>>(nn, "GLU")
        .def(py::init<int64_t>(), py::arg("dim") = -1)
        .def("__repr__", [](const tenzor::nn::GLU&) { return "GLU()"; });

    nn.def("glu", [](const tenzor::Variable& input, int64_t dim) {
        return tenzor::nn::glu(input, dim);
    }, "Functional GLU activation", py::arg("input"), py::arg("dim") = -1);

    // Unflatten
    py::class_<tenzor::nn::Unflatten, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Unflatten>>(nn, "Unflatten")
        .def(py::init<int64_t, std::vector<int64_t>>(),
             py::arg("dim"), py::arg("unflattened_size"))
        .def("__repr__", [](const tenzor::nn::Unflatten&) { return "Unflatten()"; });

    // PixelShuffle / PixelUnshuffle
    py::class_<tenzor::nn::PixelShuffle, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::PixelShuffle>>(nn, "PixelShuffle")
        .def(py::init<int64_t>(), py::arg("upscale_factor"))
        .def("__repr__", [](const tenzor::nn::PixelShuffle&) { return "PixelShuffle()"; });

    py::class_<tenzor::nn::PixelUnshuffle, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::PixelUnshuffle>>(nn, "PixelUnshuffle")
        .def(py::init<int64_t>(), py::arg("downscale_factor"))
        .def("__repr__", [](const tenzor::nn::PixelUnshuffle&) { return "PixelUnshuffle()"; });

    py::class_<tenzor::nn::ChannelShuffle, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ChannelShuffle>>(nn, "ChannelShuffle",
        "Rearranges channels by splitting into groups, transposing, and\n"
        "flattening back. Used in ShuffleNet architectures.")
        .def(py::init<int64_t>(), py::arg("groups"))
        .def("__repr__", [](const tenzor::nn::ChannelShuffle& self) {
            return "ChannelShuffle()";
        });

    // Padding layers
    py::class_<tenzor::nn::ConstantPad1d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ConstantPad1d>>(nn, "ConstantPad1d")
        .def(py::init<int64_t, int64_t, double>(),
             py::arg("padding_left"), py::arg("padding_right"), py::arg("value") = 0.0)
        .def(py::init<int64_t, double>(),
             py::arg("padding"), py::arg("value") = 0.0)
        .def("__repr__", [](const tenzor::nn::ConstantPad1d&) { return "ConstantPad1d()"; });

    py::class_<tenzor::nn::ConstantPad2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ConstantPad2d>>(nn, "ConstantPad2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, double>(),
             py::arg("padding_left"), py::arg("padding_right"),
             py::arg("padding_top"), py::arg("padding_bottom"),
             py::arg("value") = 0.0)
        .def(py::init<int64_t, double>(),
             py::arg("padding"), py::arg("value") = 0.0)
        .def("__repr__", [](const tenzor::nn::ConstantPad2d&) { return "ConstantPad2d()"; });

    py::class_<tenzor::nn::ConstantPad3d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ConstantPad3d>>(nn, "ConstantPad3d")
        .def(py::init<std::vector<int64_t>, double>(),
             py::arg("padding"), py::arg("value") = 0.0)
        .def(py::init<int64_t, double>(),
             py::arg("padding"), py::arg("value") = 0.0)
        .def("__repr__", [](const tenzor::nn::ConstantPad3d&) { return "ConstantPad3d()"; });

    py::class_<tenzor::nn::ReflectionPad1d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ReflectionPad1d>>(nn, "ReflectionPad1d")
        .def(py::init<int64_t, int64_t>(),
             py::arg("padding_left"), py::arg("padding_right"))
        .def(py::init<int64_t>(), py::arg("padding"))
        .def("__repr__", [](const tenzor::nn::ReflectionPad1d&) { return "ReflectionPad1d()"; });

    py::class_<tenzor::nn::ReflectionPad2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ReflectionPad2d>>(nn, "ReflectionPad2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>(),
             py::arg("padding_left"), py::arg("padding_right"),
             py::arg("padding_top"), py::arg("padding_bottom"))
        .def(py::init<int64_t>(), py::arg("padding"))
        .def("__repr__", [](const tenzor::nn::ReflectionPad2d&) { return "ReflectionPad2d()"; });

    py::class_<tenzor::nn::ReplicationPad1d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ReplicationPad1d>>(nn, "ReplicationPad1d")
        .def(py::init<int64_t, int64_t>(),
             py::arg("padding_left"), py::arg("padding_right"))
        .def(py::init<int64_t>(), py::arg("padding"))
        .def("__repr__", [](const tenzor::nn::ReplicationPad1d&) { return "ReplicationPad1d()"; });

    py::class_<tenzor::nn::ReplicationPad2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ReplicationPad2d>>(nn, "ReplicationPad2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>(),
             py::arg("padding_left"), py::arg("padding_right"),
             py::arg("padding_top"), py::arg("padding_bottom"))
        .def(py::init<int64_t>(), py::arg("padding"))
        .def("__repr__", [](const tenzor::nn::ReplicationPad2d&) { return "ReplicationPad2d()"; });

    py::class_<tenzor::nn::ReplicationPad3d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ReplicationPad3d>>(nn, "ReplicationPad3d")
        .def(py::init<std::vector<int64_t>>(), py::arg("padding"))
        .def(py::init<int64_t>(), py::arg("padding"))
        .def("__repr__", [](const tenzor::nn::ReplicationPad3d&) { return "ReplicationPad3d()"; });

    py::class_<tenzor::nn::ZeroPad2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ZeroPad2d>>(nn, "ZeroPad2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>(),
             py::arg("padding_left"), py::arg("padding_right"),
             py::arg("padding_top"), py::arg("padding_bottom"))
        .def(py::init<int64_t>(), py::arg("padding"))
        .def("__repr__", [](const tenzor::nn::ZeroPad2d&) { return "ZeroPad2d()"; });

    // Upsample layer
    py::class_<tenzor::nn::Upsample, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Upsample>>(nn, "Upsample")
        .def(py::init<std::optional<std::vector<int64_t>>, std::optional<double>,
                       const std::string&, bool>(),
             py::arg("size") = py::none(),
             py::arg("scale_factor") = py::none(),
             py::arg("mode") = "nearest",
             py::arg("align_corners") = false)
        .def("__repr__", [](const tenzor::nn::Upsample&) { return "Upsample()"; });

    // ParameterList / ParameterDict
    py::class_<tenzor::nn::ParameterList, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ParameterList>>(nn, "ParameterList")
        .def(py::init<>())
        .def("append", &tenzor::nn::ParameterList::append, py::arg("param"))
        .def("__getitem__", [](const tenzor::nn::ParameterList& self, size_t idx) {
            return *self.at(idx);
        })
        .def("__len__", &tenzor::nn::ParameterList::size)
        .def("__repr__", [](const tenzor::nn::ParameterList& self) {
            return "ParameterList(size=" + std::to_string(self.size()) + ")";
        });

    py::class_<tenzor::nn::ParameterDict, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ParameterDict>>(nn, "ParameterDict")
        .def(py::init<>())
        .def("insert", &tenzor::nn::ParameterDict::insert, py::arg("key"), py::arg("param"))
        .def("__getitem__", [](const tenzor::nn::ParameterDict& self, const std::string& key) {
            return *self.at(key);
        })
        .def("__contains__", &tenzor::nn::ParameterDict::contains)
        .def("__len__", &tenzor::nn::ParameterDict::size)
        .def("keys", &tenzor::nn::ParameterDict::keys)
        .def("__repr__", [](const tenzor::nn::ParameterDict& self) {
            return "ParameterDict(size=" + std::to_string(self.size()) + ")";
        });

    // RNN layers
    py::class_<tenzor::nn::RNNCell, tenzor::nn::Module, std::shared_ptr<tenzor::nn::RNNCell>>(nn, "RNNCell")
        .def(py::init<int64_t, int64_t, const std::string&, bool>(),
             py::arg("input_size"), py::arg("hidden_size"),
             py::arg("nonlinearity") = "tanh", py::arg("bias") = true)
        .def("forward", [](tenzor::nn::RNNCell& self, const tenzor::Variable& input, const tenzor::Variable& hx) {
            return self.forward(input, hx);
        }, py::arg("input"), py::arg("hx") = tenzor::Variable{},
           py::call_guard<py::gil_scoped_release>());

    py::class_<tenzor::nn::RNN, tenzor::nn::Module, std::shared_ptr<tenzor::nn::RNN>>(nn, "RNN")
        .def(py::init<int64_t, int64_t, int64_t, const std::string&, bool, bool, double, bool>(),
             py::arg("input_size"), py::arg("hidden_size"), py::arg("num_layers") = 1,
             py::arg("nonlinearity") = "tanh", py::arg("bias") = true,
             py::arg("batch_first") = false, py::arg("dropout") = 0.0,
             py::arg("bidirectional") = false)
        .def("forward", [](tenzor::nn::RNN& self, const tenzor::Variable& input, const tenzor::Variable& hx) {
            return self.forward(input, hx);
        }, py::arg("input"), py::arg("hx") = tenzor::Variable{},
           py::call_guard<py::gil_scoped_release>());

    py::class_<tenzor::nn::LSTMCell, tenzor::nn::Module, std::shared_ptr<tenzor::nn::LSTMCell>>(nn, "LSTMCell")
        .def(py::init<int64_t, int64_t, bool>(),
             py::arg("input_size"), py::arg("hidden_size"), py::arg("bias") = true)
        .def("forward", [](tenzor::nn::LSTMCell& self, const tenzor::Variable& input,
                           const tenzor::Variable& hx, const tenzor::Variable& cx) {
            return self.forward(input, hx, cx);
        }, py::arg("input"), py::arg("hx") = tenzor::Variable{},
           py::arg("cx") = tenzor::Variable{},
           py::call_guard<py::gil_scoped_release>());

    py::class_<tenzor::nn::LSTM, tenzor::nn::Module, std::shared_ptr<tenzor::nn::LSTM>>(nn, "LSTM")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool, double, bool, int64_t>(),
             py::arg("input_size"), py::arg("hidden_size"), py::arg("num_layers") = 1,
             py::arg("bias") = true, py::arg("batch_first") = false,
             py::arg("dropout") = 0.0, py::arg("bidirectional") = false,
             py::arg("proj_size") = 0)
        .def("forward", [](tenzor::nn::LSTM& self, const tenzor::Variable& input,
                           const std::pair<tenzor::Variable, tenzor::Variable>& hx) {
            return self.forward(input, hx);
        }, py::arg("input"), py::arg("hx") = std::pair<tenzor::Variable, tenzor::Variable>{},
           py::call_guard<py::gil_scoped_release>())
        .def("__repr__", [](const tenzor::nn::LSTM&) { return "LSTM()"; });

    py::class_<tenzor::nn::GRUCell, tenzor::nn::Module, std::shared_ptr<tenzor::nn::GRUCell>>(nn, "GRUCell")
        .def(py::init<int64_t, int64_t, bool>(),
             py::arg("input_size"), py::arg("hidden_size"), py::arg("bias") = true)
        .def("forward", [](tenzor::nn::GRUCell& self, const tenzor::Variable& input, const tenzor::Variable& hx) {
            return self.forward(input, hx);
        }, py::arg("input"), py::arg("hx") = tenzor::Variable{},
           py::call_guard<py::gil_scoped_release>());

    py::class_<tenzor::nn::GRU, tenzor::nn::Module, std::shared_ptr<tenzor::nn::GRU>>(nn, "GRU")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool, double, bool>(),
             py::arg("input_size"), py::arg("hidden_size"), py::arg("num_layers") = 1,
             py::arg("bias") = true, py::arg("batch_first") = false,
             py::arg("dropout") = 0.0, py::arg("bidirectional") = false)
        .def("forward", [](tenzor::nn::GRU& self, const tenzor::Variable& input, const tenzor::Variable& hx) {
            return self.forward(input, hx);
        }, py::arg("input"), py::arg("hx") = tenzor::Variable{},
           py::call_guard<py::gil_scoped_release>())
        .def("__repr__", [](const tenzor::nn::GRU&) { return "GRU()"; });

    // Attention and Transformer
    py::class_<tenzor::nn::MultiheadAttention, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::MultiheadAttention>>(nn, "MultiheadAttention")
        .def(py::init<int64_t, int64_t, double, bool, bool, bool, int64_t, int64_t, bool, bool>(),
             py::arg("embed_dim"), py::arg("num_heads"), py::arg("dropout") = 0.0,
             py::arg("bias") = true, py::arg("add_bias_kv") = false,
             py::arg("add_zero_attn") = false, py::arg("kdim") = 0,
             py::arg("vdim") = 0, py::arg("batch_first") = false,
             py::arg("is_causal") = false)
        .def("forward", [](tenzor::nn::MultiheadAttention& self, const tenzor::Variable& query,
                           const tenzor::Variable& key, const tenzor::Variable& value,
                           const tenzor::Tensor& key_padding_mask, const tenzor::Tensor& attn_mask,
                           bool need_weights, const tenzor::Tensor& position_bias) {
            return self.forward(query, key, value, key_padding_mask, attn_mask, need_weights, position_bias);
        }, py::arg("query"), py::arg("key"), py::arg("value"),
           py::arg("key_padding_mask") = tenzor::Tensor{},
           py::arg("attn_mask") = tenzor::Tensor{},
           py::arg("need_weights") = true,
           py::arg("position_bias") = tenzor::Tensor{},
           py::call_guard<py::gil_scoped_release>())
        .def("__repr__", [](const tenzor::nn::MultiheadAttention&) { return "MultiheadAttention()"; });

    py::class_<tenzor::nn::PositionalEncoding, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::PositionalEncoding>>(nn, "PositionalEncoding")
        .def(py::init<int64_t, int64_t, double>(),
             py::arg("d_model"), py::arg("max_len") = 5000, py::arg("dropout") = 0.0)
        .def("forward", &tenzor::nn::PositionalEncoding::forward,
             py::call_guard<py::gil_scoped_release>());

    py::class_<tenzor::nn::TransformerEncoderLayer, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::TransformerEncoderLayer>>(nn, "TransformerEncoderLayer")
        .def(py::init<int64_t, int64_t, int64_t, double, const std::string&, bool, bool>(),
             py::arg("d_model"), py::arg("nhead"), py::arg("dim_feedforward") = 2048,
             py::arg("dropout") = 0.1, py::arg("activation") = "relu",
             py::arg("batch_first") = false, py::arg("norm_first") = false)
        .def("forward", [](tenzor::nn::TransformerEncoderLayer& self, const tenzor::Variable& src,
                           const tenzor::Tensor& src_mask, const tenzor::Tensor& src_key_padding_mask) {
            return self.forward(src, src_mask, src_key_padding_mask);
        }, py::arg("src"), py::arg("src_mask") = tenzor::Tensor{},
           py::arg("src_key_padding_mask") = tenzor::Tensor{},
           py::call_guard<py::gil_scoped_release>());

    py::class_<tenzor::nn::TransformerEncoder, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::TransformerEncoder>>(nn, "TransformerEncoder")
        .def(py::init<std::shared_ptr<tenzor::nn::TransformerEncoderLayer>, int64_t,
                     std::shared_ptr<tenzor::nn::LayerNorm>>(),
             py::arg("encoder_layer"), py::arg("num_layers"), py::arg("norm") = nullptr)
        .def("forward", [](tenzor::nn::TransformerEncoder& self, const tenzor::Variable& src,
                           const tenzor::Tensor& mask, const tenzor::Tensor& src_key_padding_mask) {
            return self.forward(src, mask, src_key_padding_mask);
        }, py::arg("src"), py::arg("mask") = tenzor::Tensor{},
           py::arg("src_key_padding_mask") = tenzor::Tensor{},
           py::call_guard<py::gil_scoped_release>());

    py::class_<tenzor::nn::TransformerDecoderLayer, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::TransformerDecoderLayer>>(nn, "TransformerDecoderLayer")
        .def(py::init<int64_t, int64_t, int64_t, double, const std::string&, bool, bool>(),
             py::arg("d_model"), py::arg("nhead"), py::arg("dim_feedforward") = 2048,
             py::arg("dropout") = 0.1, py::arg("activation") = "relu",
             py::arg("batch_first") = false, py::arg("norm_first") = false)
        .def("forward", [](tenzor::nn::TransformerDecoderLayer& self, const tenzor::Variable& tgt,
                           const tenzor::Variable& memory, const tenzor::Tensor& tgt_mask,
                           const tenzor::Tensor& memory_mask, const tenzor::Tensor& tgt_key_padding_mask,
                           const tenzor::Tensor& memory_key_padding_mask) {
            return self.forward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask);
        }, py::arg("tgt"), py::arg("memory"),
           py::arg("tgt_mask") = tenzor::Tensor{},
           py::arg("memory_mask") = tenzor::Tensor{},
           py::arg("tgt_key_padding_mask") = tenzor::Tensor{},
           py::arg("memory_key_padding_mask") = tenzor::Tensor{},
           py::call_guard<py::gil_scoped_release>());

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
           py::arg("memory_key_padding_mask") = tenzor::Tensor{},
           py::call_guard<py::gil_scoped_release>());

    py::class_<tenzor::nn::Transformer, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Transformer>>(nn, "Transformer")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, double, const std::string&, bool, bool>(),
             py::arg("d_model") = 512, py::arg("nhead") = 8,
             py::arg("num_encoder_layers") = 6, py::arg("num_decoder_layers") = 6,
             py::arg("dim_feedforward") = 2048, py::arg("dropout") = 0.1,
             py::arg("activation") = "relu", py::arg("batch_first") = false,
             py::arg("norm_first") = false)
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
           py::arg("memory_key_padding_mask") = tenzor::Tensor{},
           py::call_guard<py::gil_scoped_release>());

    // Embedding layers
    py::class_<tenzor::nn::Embedding, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Embedding>>(nn, "Embedding")
        .def(py::init<int64_t, int64_t, int64_t, double, double, bool, bool>(),
             py::arg("num_embeddings"), py::arg("embedding_dim"),
             py::arg("padding_idx") = -1, py::arg("max_norm") = 0.0,
             py::arg("norm_type") = 2.0, py::arg("scale_grad_by_freq") = false,
             py::arg("sparse") = false)
        .def("forward", &tenzor::nn::Embedding::forward,
             py::call_guard<py::gil_scoped_release>())
        .def("weight", py::overload_cast<>(&tenzor::nn::Embedding::weight))
        .def_static("from_pretrained", &tenzor::nn::Embedding::from_pretrained,
             py::arg("embeddings"), py::arg("freeze") = true,
             py::arg("padding_idx") = -1,
             "Create Embedding from pretrained weight tensor")
        .def("__repr__", [](const tenzor::nn::Embedding& self) {
            auto params = const_cast<tenzor::nn::Embedding&>(self).own_parameters();
            int64_t num_emb = 0, emb_dim = 0;
            if (!params.empty()) {
                auto shape = params[0]->tensor().shape();
                if (shape.size() >= 2) { num_emb = shape[0]; emb_dim = shape[1]; }
            }
            return "Embedding(" + std::to_string(num_emb) + ", " + std::to_string(emb_dim) + ")";
        });

    py::class_<tenzor::nn::EmbeddingBag, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::EmbeddingBag>>(nn, "EmbeddingBag")
        .def(py::init<int64_t, int64_t, double, double, bool, const std::string&, bool, bool>(),
             py::arg("num_embeddings"), py::arg("embedding_dim"),
             py::arg("max_norm") = 0.0, py::arg("norm_type") = 2.0,
             py::arg("scale_grad_by_freq") = false, py::arg("mode") = "mean",
             py::arg("sparse") = false, py::arg("include_last_offset") = false)
        .def("forward", [](tenzor::nn::EmbeddingBag& self, const tenzor::Variable& input, const tenzor::Variable& offsets) {
            return self.forward(input, offsets);
        }, py::arg("input"), py::arg("offsets") = tenzor::Variable{},
           py::call_guard<py::gil_scoped_release>());

    // =========================================================================
    // Priority 1: LLM-critical layers
    // =========================================================================

    py::class_<tenzor::nn::ALiBi, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ALiBi>>(nn, "ALiBi",
        "Attention with Linear Biases (ALiBi) positional encoding")
        .def(py::init<int64_t>(),
             py::arg("num_heads"))
        .def("forward", &tenzor::nn::ALiBi::forward_impl,
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Forward pass (identity — use get_bias() for the bias tensor)")
        .def("get_bias", &tenzor::nn::ALiBi::get_bias,
             py::arg("seq_q"), py::arg("seq_k"),
             py::arg("device") = tenzor::Device::cpu(),
             py::arg("dtype") = tenzor::DType::Float32,
             py::call_guard<py::gil_scoped_release>(),
             "Get ALiBi bias tensor of shape (1, num_heads, seq_q, seq_k)");

    py::class_<tenzor::nn::GroupedQueryAttention, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::GroupedQueryAttention>>(nn, "GroupedQueryAttention",
        "Grouped Query Attention (GQA / MQA) layer")
        .def(py::init<int64_t, int64_t, int64_t, double, bool, bool>(),
             py::arg("embed_dim"),
             py::arg("num_heads"),
             py::arg("num_kv_heads"),
             py::arg("dropout") = 0.0,
             py::arg("bias") = true,
             py::arg("is_causal") = false)
        .def("forward", [](tenzor::nn::GroupedQueryAttention& self,
                           const tenzor::Variable& query,
                           const tenzor::Variable& key,
                           const tenzor::Variable& value,
                           const tenzor::Tensor& attn_mask,
                           bool need_weights) {
            return self.forward(query, key, value, attn_mask, need_weights);
        }, py::arg("query"), py::arg("key"), py::arg("value"),
           py::arg("attn_mask") = tenzor::Tensor{},
           py::arg("need_weights") = false,
           py::call_guard<py::gil_scoped_release>(),
           "Forward pass through grouped query attention")
        .def_property_readonly("embed_dim", &tenzor::nn::GroupedQueryAttention::embed_dim)
        .def_property_readonly("num_heads", &tenzor::nn::GroupedQueryAttention::num_heads)
        .def_property_readonly("num_kv_heads", &tenzor::nn::GroupedQueryAttention::num_kv_heads)
        .def_property_readonly("num_heads_per_group", &tenzor::nn::GroupedQueryAttention::num_heads_per_group)
        .def_property_readonly("head_dim", &tenzor::nn::GroupedQueryAttention::head_dim)
        .def_property_readonly("is_causal", &tenzor::nn::GroupedQueryAttention::is_causal);

    // =========================================================================
    // Priority 2: ViT-critical layers
    // =========================================================================

    py::class_<tenzor::nn::DropPath, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::DropPath>>(nn, "DropPath",
        "DropPath (Stochastic Depth) regularization — drops entire samples")
        .def(py::init<double>(),
             py::arg("p") = 0.0)
        .def("forward", &tenzor::nn::DropPath::forward_impl,
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Forward pass through DropPath");

    py::class_<tenzor::nn::PatchEmbedding, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::PatchEmbedding>>(nn, "PatchEmbedding",
        "Patch embedding layer for Vision Transformers")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>(),
             py::arg("in_channels"),
             py::arg("embed_dim"),
             py::arg("patch_size"),
             py::arg("img_size") = 224)
        .def("forward", &tenzor::nn::PatchEmbedding::forward_impl,
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Convert image to patch embeddings: (N,C,H,W) -> (N,num_patches,embed_dim)")
        .def_property_readonly("num_patches", &tenzor::nn::PatchEmbedding::num_patches)
        .def_property_readonly("patch_size", &tenzor::nn::PatchEmbedding::patch_size)
        .def_property_readonly("embed_dim", &tenzor::nn::PatchEmbedding::embed_dim);

    py::class_<tenzor::nn::WindowAttention, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::WindowAttention>>(nn, "WindowAttention",
        "Window-based Multi-Head Self-Attention for Swin Transformer")
        .def(py::init<int64_t, int64_t, int64_t, bool, double, double, double>(),
             py::arg("dim"),
             py::arg("window_size") = 7,
             py::arg("num_heads") = 3,
             py::arg("qkv_bias") = true,
             py::arg("qk_scale") = 0.0,
             py::arg("attn_drop") = 0.0,
             py::arg("proj_drop") = 0.0)
        .def("forward", [](tenzor::nn::WindowAttention& self,
                           const tenzor::Variable& input,
                           const py::object& mask_obj) {
            tenzor::Tensor mask;
            if (!mask_obj.is_none()) {
                mask = mask_obj.cast<tenzor::Tensor>();
            }
            py::gil_scoped_release release;
            return self.forward(input, mask);
        }, py::arg("input"), py::arg("mask") = py::none(),
           "Forward pass through window attention")
        .def_property_readonly("dim", &tenzor::nn::WindowAttention::dim)
        .def_property_readonly("window_size", &tenzor::nn::WindowAttention::window_size)
        .def_property_readonly("num_heads", &tenzor::nn::WindowAttention::num_heads);

    // Vision helper functions
    nn.def("window_partition", &tenzor::nn::window_partition,
           py::arg("input"), py::arg("window_size"),
           py::call_guard<py::gil_scoped_release>(),
           "Partition tensor into non-overlapping windows: (B,H,W,C) -> (B*nW,M*M,C)");

    nn.def("window_reverse", &tenzor::nn::window_reverse,
           py::arg("windows"), py::arg("window_size"),
           py::arg("H"), py::arg("W"),
           py::call_guard<py::gil_scoped_release>(),
           "Reverse window partition: (B*nW,M*M,C) -> (B,H,W,C)");

    nn.def("create_shifted_window_mask", &tenzor::nn::create_shifted_window_mask,
           py::arg("H"), py::arg("W"),
           py::arg("window_size"), py::arg("shift_size"),
           py::arg("device") = tenzor::Device::cpu(),
           py::arg("dtype") = tenzor::DType::Float32,
           py::call_guard<py::gil_scoped_release>(),
           "Create attention mask for shifted window attention");

    // =========================================================================
    // Priority 3: Mobile/Segmentation layers
    // =========================================================================

    py::class_<tenzor::nn::SqueezeExcitation, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::SqueezeExcitation>>(nn, "SqueezeExcitation",
        "Squeeze-and-Excitation channel attention block")
        .def(py::init<int64_t, int64_t, std::string>(),
             py::arg("channels"),
             py::arg("reduction") = 16,
             py::arg("activation") = "relu")
        .def("forward", &tenzor::nn::SqueezeExcitation::forward_impl,
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Forward pass through SE block: (N,C,H,W) -> (N,C,H,W)");

    py::class_<tenzor::nn::InvertedResidual, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::InvertedResidual>>(nn, "InvertedResidual",
        "Inverted residual block (MBConv) for MobileNet/EfficientNet")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, bool, int64_t, std::string>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("expand_ratio") = 6,
             py::arg("stride") = 1,
             py::arg("use_se") = false,
             py::arg("kernel_size") = 3,
             py::arg("activation") = "relu6")
        .def("forward", &tenzor::nn::InvertedResidual::forward_impl,
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Forward pass through inverted residual block");

    py::class_<tenzor::nn::FusedMBConv, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::FusedMBConv>>(nn, "FusedMBConv",
        "Fused Mobile Inverted Bottleneck for EfficientNetV2")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, bool, std::string>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("expand_ratio") = 4,
             py::arg("stride") = 1,
             py::arg("use_se") = false,
             py::arg("activation") = "swish")
        .def("forward", &tenzor::nn::FusedMBConv::forward_impl,
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Forward pass through fused MBConv block");

    py::class_<tenzor::nn::AtrousSeparableConv2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::AtrousSeparableConv2d>>(nn, "AtrousSeparableConv2d",
        "Atrous (Dilated) Separable Convolution for DeepLab")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, bool>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size") = 3,
             py::arg("dilation") = 1,
             py::arg("bias") = false)
        .def("forward", &tenzor::nn::AtrousSeparableConv2d::forward_impl,
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Forward pass through atrous separable convolution");

    py::class_<tenzor::nn::ASPP, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::ASPP>>(nn, "ASPP",
        "Atrous Spatial Pyramid Pooling for semantic segmentation")
        .def(py::init<int64_t, int64_t, std::vector<int64_t>, bool, float>(),
             py::arg("in_channels"),
             py::arg("out_channels") = 256,
             py::arg("atrous_rates") = std::vector<int64_t>{6, 12, 18},
             py::arg("use_separable") = true,
             py::arg("dropout_rate") = 0.5f)
        .def("forward", &tenzor::nn::ASPP::forward_impl,
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Forward pass through ASPP: (N,C_in,H,W) -> (N,C_out,H,W)");

    // =========================================================================
    // Priority 4: Sparse layers
    // =========================================================================

    py::class_<tenzor::nn::SparseLinear, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::SparseLinear>>(nn, "SparseLinear",
        "Linear layer with sparse weight matrix (CSR format)")
        .def(py::init<int64_t, int64_t, double, bool>(),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("density") = 0.1,
             py::arg("bias") = true)
        .def(py::init<const tenzor::SparseTensor&, bool>(),
             py::arg("sparse_weight"),
             py::arg("bias") = true)
        .def("forward", &tenzor::nn::SparseLinear::forward_impl,
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Forward pass: y = spmm(sparse_weight, x^T)^T + bias")
        .def_property_readonly("has_bias", &tenzor::nn::SparseLinear::has_bias)
        .def_property_readonly("density", &tenzor::nn::SparseLinear::density);

    py::class_<tenzor::nn::SparseEmbedding, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::SparseEmbedding>>(nn, "SparseEmbedding",
        "Embedding layer with sparse gradient accumulation")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("num_embeddings"),
             py::arg("embedding_dim"),
             py::arg("padding_idx") = -1)
        .def("forward", &tenzor::nn::SparseEmbedding::forward_impl,
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>(),
             "Lookup embeddings with sparse gradient support")
        .def_property_readonly("num_embeddings", &tenzor::nn::SparseEmbedding::num_embeddings)
        .def_property_readonly("embedding_dim", &tenzor::nn::SparseEmbedding::embedding_dim);

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

    nn.def("softplus", [](const tenzor::Variable& input, float beta) -> tenzor::Variable {
        return tenzor::softplus(input, beta);
    }, py::arg("input"), py::arg("beta") = 1.0f,
       "Softplus activation: log(1 + exp(beta*x))/beta",
       py::call_guard<py::gil_scoped_release>());

    // Reduction enum for loss functions
    py::enum_<tenzor::nn::Reduction>(nn, "Reduction",
        "Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'")
        .value("NONE", tenzor::nn::Reduction::None, "No reduction will be applied")
        .value("MEAN", tenzor::nn::Reduction::Mean, "The output will be averaged")
        .value("SUM", tenzor::nn::Reduction::Sum, "The output will be summed")
        .value("none", tenzor::nn::Reduction::None)  // Lowercase alias
        .value("mean", tenzor::nn::Reduction::Mean)  // Lowercase alias
        .value("sum", tenzor::nn::Reduction::Sum)    // Lowercase alias
        .export_values();

    // Loss function classes
    py::class_<tenzor::nn::MSELoss>(nn, "MSELoss",
        "Mean Squared Error loss for regression tasks")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean,
             "Create MSELoss with specified reduction mode")
        .def("forward", &tenzor::nn::MSELoss::forward,
             py::arg("input"), py::arg("target"),
             "Compute MSE loss between input and target",
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::MSELoss::operator(),
             py::arg("input"), py::arg("target"),
             "Compute MSE loss between input and target");

    py::class_<tenzor::nn::L1Loss>(nn, "L1Loss",
        "L1 Loss (Mean Absolute Error) for robust regression")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean,
             "Create L1Loss with specified reduction mode")
        .def("forward", &tenzor::nn::L1Loss::forward,
             py::arg("input"), py::arg("target"),
             "Compute L1 loss between input and target",
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::L1Loss::operator(),
             py::arg("input"), py::arg("target"),
             "Compute L1 loss between input and target");

    py::class_<tenzor::nn::SmoothL1Loss>(nn, "SmoothL1Loss",
        "Smooth L1 Loss (Huber Loss) combining L1 and L2 loss properties")
        .def(py::init<tenzor::nn::Reduction, double>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean,
             py::arg("beta") = 1.0,
             "Create SmoothL1Loss with reduction mode and beta threshold")
        .def("forward", &tenzor::nn::SmoothL1Loss::forward,
             py::arg("input"), py::arg("target"),
             "Compute Smooth L1 loss between input and target",
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::SmoothL1Loss::operator(),
             py::arg("input"), py::arg("target"),
             "Compute Smooth L1 loss between input and target");

    py::class_<tenzor::nn::CrossEntropyLoss>(nn, "CrossEntropyLoss",
        "Cross Entropy Loss for multi-class classification (combines LogSoftmax and NLLLoss)")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean,
             "Create CrossEntropyLoss with specified reduction mode")
        .def("forward", &tenzor::nn::CrossEntropyLoss::forward,
             py::arg("input"), py::arg("target"),
             "Compute cross entropy loss between input logits and target class indices",
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::CrossEntropyLoss::operator(),
             py::arg("input"), py::arg("target"),
             "Compute cross entropy loss between input logits and target class indices");

    py::class_<tenzor::nn::NLLLoss>(nn, "NLLLoss",
        "Negative Log Likelihood Loss for classification with log-probabilities")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean,
             "Create NLLLoss with specified reduction mode")
        .def("forward", &tenzor::nn::NLLLoss::forward,
             py::arg("input"), py::arg("target"),
             "Compute NLL loss between input log-probabilities and target class indices",
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::NLLLoss::operator(),
             py::arg("input"), py::arg("target"),
             "Compute NLL loss between input log-probabilities and target class indices");

    py::class_<tenzor::nn::BCELoss>(nn, "BCELoss",
        "Binary Cross Entropy Loss for binary classification with probabilities")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean,
             "Create BCELoss with specified reduction mode")
        .def("forward", &tenzor::nn::BCELoss::forward,
             py::arg("input"), py::arg("target"),
             "Compute BCE loss between input probabilities and binary targets",
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::BCELoss::operator(),
             py::arg("input"), py::arg("target"),
             "Compute BCE loss between input probabilities and binary targets");

    py::class_<tenzor::nn::BCEWithLogitsLoss>(nn, "BCEWithLogitsLoss",
        "Binary Cross Entropy with Logits Loss (numerically stable version for binary classification)")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean,
             "Create BCEWithLogitsLoss with specified reduction mode")
        .def("forward", &tenzor::nn::BCEWithLogitsLoss::forward,
             py::arg("input"), py::arg("target"),
             "Compute BCE with logits loss between input logits and binary targets",
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::BCEWithLogitsLoss::operator(),
             py::arg("input"), py::arg("target"),
             "Compute BCE with logits loss between input logits and binary targets");

    // Advanced loss functions
    py::class_<tenzor::nn::KLDivLoss>(nn, "KLDivLoss")
        .def(py::init<const std::string&, bool>(),
             py::arg("reduction") = "mean", py::arg("log_target") = false)
        .def("forward", &tenzor::nn::KLDivLoss::forward,
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::KLDivLoss::operator());

    py::class_<tenzor::nn::FocalLoss>(nn, "FocalLoss")
        .def(py::init<double, double, const std::string&>(),
             py::arg("alpha") = 1.0, py::arg("gamma") = 2.0,
             py::arg("reduction") = "mean")
        .def("forward", &tenzor::nn::FocalLoss::forward,
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::FocalLoss::operator());

    py::class_<tenzor::nn::DiceLoss>(nn, "DiceLoss")
        .def(py::init<double, const std::string&>(),
             py::arg("smooth") = 1.0, py::arg("reduction") = "mean")
        .def("forward", &tenzor::nn::DiceLoss::forward,
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::DiceLoss::operator());

    py::class_<tenzor::nn::HuberLoss>(nn, "HuberLoss")
        .def(py::init<double, const std::string&>(),
             py::arg("delta") = 1.0, py::arg("reduction") = "mean")
        .def("forward", &tenzor::nn::HuberLoss::forward,
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::HuberLoss::operator());

    py::class_<tenzor::nn::CTCLoss>(nn, "CTCLoss",
        "Connectionist Temporal Classification loss for sequence-to-sequence tasks")
        .def(py::init<const std::string&, int64_t, bool>(),
             py::arg("reduction") = "mean",
             py::arg("blank") = 0,
             py::arg("zero_infinity") = false)
        .def("forward", &tenzor::nn::CTCLoss::forward,
             py::arg("log_probs"), py::arg("targets"),
             py::arg("input_lengths"), py::arg("target_lengths"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::CTCLoss::operator(),
             py::arg("log_probs"), py::arg("targets"),
             py::arg("input_lengths"), py::arg("target_lengths"));

    // Contrastive loss functions
    py::class_<tenzor::nn::InfoNCELoss>(nn, "InfoNCELoss",
        "InfoNCE loss for contrastive learning (SimCLR, CLIP, MoCo)")
        .def(py::init<double, tenzor::nn::Reduction>(),
             py::arg("temperature") = 0.07,
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::InfoNCELoss::forward,
             py::arg("queries"), py::arg("keys"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::InfoNCELoss::operator(),
             py::arg("queries"), py::arg("keys"));

    py::class_<tenzor::nn::NTXentLoss>(nn, "NTXentLoss",
        "NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR")
        .def(py::init<double, tenzor::nn::Reduction>(),
             py::arg("temperature") = 0.5,
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::NTXentLoss::forward,
             py::arg("z_i"), py::arg("z_j"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::NTXentLoss::operator(),
             py::arg("z_i"), py::arg("z_j"));

    py::class_<tenzor::nn::TripletLoss>(nn, "TripletLoss",
        "Triplet margin loss for metric learning")
        .def(py::init<double, double, bool, tenzor::nn::Reduction>(),
             py::arg("margin") = 1.0, py::arg("p") = 2.0,
             py::arg("swap") = false,
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::TripletLoss::forward,
             py::arg("anchor"), py::arg("positive"), py::arg("negative"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::TripletLoss::operator(),
             py::arg("anchor"), py::arg("positive"), py::arg("negative"));

    py::class_<tenzor::nn::MarginRankingLoss>(nn, "MarginRankingLoss",
        "Margin ranking loss for ranking tasks")
        .def(py::init<double, tenzor::nn::Reduction>(),
             py::arg("margin") = 0.0,
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::MarginRankingLoss::forward,
             py::arg("input1"), py::arg("input2"), py::arg("target"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::MarginRankingLoss::operator(),
             py::arg("input1"), py::arg("input2"), py::arg("target"));

    py::class_<tenzor::nn::SoftMarginLoss>(nn, "SoftMarginLoss",
        "Two-class soft margin loss: log(1 + exp(-y * x))")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::SoftMarginLoss::forward,
             py::arg("input"), py::arg("target"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::SoftMarginLoss::operator(),
             py::arg("input"), py::arg("target"));

    py::class_<tenzor::nn::HingeEmbeddingLoss>(nn, "HingeEmbeddingLoss",
        "Hinge embedding loss for similarity/dissimilarity measurement")
        .def(py::init<double, tenzor::nn::Reduction>(),
             py::arg("margin") = 1.0,
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::HingeEmbeddingLoss::forward,
             py::arg("input"), py::arg("target"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::HingeEmbeddingLoss::operator(),
             py::arg("input"), py::arg("target"));

    py::class_<tenzor::nn::PoissonNLLLoss>(nn, "PoissonNLLLoss",
        "Poisson negative log-likelihood loss for count data")
        .def(py::init<bool, bool, double, tenzor::nn::Reduction>(),
             py::arg("log_input") = true,
             py::arg("full") = false,
             py::arg("eps") = 1e-8,
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::PoissonNLLLoss::forward,
             py::arg("input"), py::arg("target"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::PoissonNLLLoss::operator(),
             py::arg("input"), py::arg("target"));

    py::class_<tenzor::nn::CosineEmbeddingLoss>(nn, "CosineEmbeddingLoss",
        "Cosine embedding loss using cosine similarity")
        .def(py::init<double, tenzor::nn::Reduction>(),
             py::arg("margin") = 0.0,
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::CosineEmbeddingLoss::forward,
             py::arg("input1"), py::arg("input2"), py::arg("target"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::CosineEmbeddingLoss::operator(),
             py::arg("input1"), py::arg("input2"), py::arg("target"));

    py::class_<tenzor::nn::TripletMarginLoss>(nn, "TripletMarginLoss",
        "Triplet margin loss with configurable distance norm")
        .def(py::init<double, double, bool, tenzor::nn::Reduction>(),
             py::arg("margin") = 1.0,
             py::arg("p") = 2.0,
             py::arg("swap") = false,
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::TripletMarginLoss::forward,
             py::arg("anchor"), py::arg("positive"), py::arg("negative"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::TripletMarginLoss::operator(),
             py::arg("anchor"), py::arg("positive"), py::arg("negative"));

    py::class_<tenzor::nn::MultiLabelSoftMarginLoss>(nn, "MultiLabelSoftMarginLoss",
        "Multi-label one-versus-all loss based on max-entropy")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::MultiLabelSoftMarginLoss::forward,
             py::arg("input"), py::arg("target"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::MultiLabelSoftMarginLoss::operator(),
             py::arg("input"), py::arg("target"));

    py::class_<tenzor::nn::MultiMarginLoss>(nn, "MultiMarginLoss",
        "Multi-class classification hinge loss")
        .def(py::init<int, double, tenzor::nn::Reduction>(),
             py::arg("p") = 1,
             py::arg("margin") = 1.0,
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::MultiMarginLoss::forward,
             py::arg("input"), py::arg("target"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::MultiMarginLoss::operator(),
             py::arg("input"), py::arg("target"));

    py::class_<tenzor::nn::GaussianNLLLoss>(nn, "GaussianNLLLoss",
        "Gaussian negative log-likelihood loss for regression with uncertainty")
        .def(py::init<bool, double, tenzor::nn::Reduction>(),
             py::arg("full") = false,
             py::arg("eps") = 1e-6,
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::GaussianNLLLoss::forward,
             py::arg("input"), py::arg("target"), py::arg("var"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::GaussianNLLLoss::operator(),
             py::arg("input"), py::arg("target"), py::arg("var"));

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

    nn.def("kl_div_loss", &tenzor::nn::kl_div_loss,
          "KL divergence loss function",
          py::arg("input"), py::arg("target"),
          py::arg("reduction") = "mean", py::arg("log_target") = false,
          py::call_guard<py::gil_scoped_release>());

    nn.def("huber_loss", &tenzor::nn::huber_loss,
          "Huber loss function",
          py::arg("input"), py::arg("target"),
          py::arg("delta") = 1.0, py::arg("reduction") = "mean",
          py::call_guard<py::gil_scoped_release>());

    nn.def("smooth_l1_loss",
          [](const tenzor::Variable& input, const tenzor::Variable& target,
             tenzor::nn::Reduction reduction, double beta) -> tenzor::Variable {
        tenzor::nn::SmoothL1Loss loss(reduction, beta);
        return loss.forward(input, target);
    }, py::arg("input"), py::arg("target"),
       py::arg("reduction") = tenzor::nn::Reduction::Mean,
       py::arg("beta") = 1.0,
       "Smooth L1 loss function",
       py::call_guard<py::gil_scoped_release>());

    nn.def("info_nce_loss", &tenzor::nn::info_nce_loss,
          py::arg("queries"), py::arg("keys"),
          py::arg("temperature") = 0.07,
          py::arg("reduction") = tenzor::nn::Reduction::Mean,
          py::call_guard<py::gil_scoped_release>());
    nn.def("nt_xent_loss", &tenzor::nn::nt_xent_loss,
          py::arg("z_i"), py::arg("z_j"),
          py::arg("temperature") = 0.5,
          py::arg("reduction") = tenzor::nn::Reduction::Mean,
          py::call_guard<py::gil_scoped_release>());
    nn.def("triplet_loss", &tenzor::nn::triplet_loss,
          py::arg("anchor"), py::arg("positive"), py::arg("negative"),
          py::arg("margin") = 1.0, py::arg("p") = 2.0, py::arg("swap") = false,
          py::arg("reduction") = tenzor::nn::Reduction::Mean,
          py::call_guard<py::gil_scoped_release>());
    nn.def("margin_ranking_loss", &tenzor::nn::margin_ranking_loss,
          py::arg("input1"), py::arg("input2"), py::arg("target"),
          py::arg("margin") = 0.0,
          py::arg("reduction") = tenzor::nn::Reduction::Mean,
          py::call_guard<py::gil_scoped_release>());

    nn.def("soft_margin_loss", &tenzor::nn::soft_margin_loss,
          py::arg("input"), py::arg("target"),
          py::arg("reduction") = tenzor::nn::Reduction::Mean,
          py::call_guard<py::gil_scoped_release>());
    nn.def("hinge_embedding_loss", &tenzor::nn::hinge_embedding_loss,
          py::arg("input"), py::arg("target"),
          py::arg("margin") = 1.0,
          py::arg("reduction") = tenzor::nn::Reduction::Mean,
          py::call_guard<py::gil_scoped_release>());
    nn.def("poisson_nll_loss", &tenzor::nn::poisson_nll_loss,
          py::arg("input"), py::arg("target"),
          py::arg("log_input") = true, py::arg("full") = false,
          py::arg("eps") = 1e-8,
          py::arg("reduction") = tenzor::nn::Reduction::Mean,
          py::call_guard<py::gil_scoped_release>());
    nn.def("cosine_embedding_loss", &tenzor::nn::cosine_embedding_loss,
          py::arg("input1"), py::arg("input2"), py::arg("target"),
          py::arg("margin") = 0.0,
          py::arg("reduction") = tenzor::nn::Reduction::Mean,
          py::call_guard<py::gil_scoped_release>());
    nn.def("triplet_margin_loss", &tenzor::nn::triplet_margin_loss,
          py::arg("anchor"), py::arg("positive"), py::arg("negative"),
          py::arg("margin") = 1.0, py::arg("p") = 2.0, py::arg("swap") = false,
          py::arg("reduction") = tenzor::nn::Reduction::Mean,
          py::call_guard<py::gil_scoped_release>());
    nn.def("multi_label_soft_margin_loss", &tenzor::nn::multi_label_soft_margin_loss,
          py::arg("input"), py::arg("target"),
          py::arg("reduction") = tenzor::nn::Reduction::Mean,
          py::call_guard<py::gil_scoped_release>());
    nn.def("multi_margin_loss", &tenzor::nn::multi_margin_loss,
          py::arg("input"), py::arg("target"),
          py::arg("p") = 1, py::arg("margin") = 1.0,
          py::arg("reduction") = tenzor::nn::Reduction::Mean,
          py::call_guard<py::gil_scoped_release>());
    nn.def("gaussian_nll_loss", &tenzor::nn::gaussian_nll_loss,
          py::arg("input"), py::arg("target"), py::arg("var"),
          py::arg("full") = false, py::arg("eps") = 1e-6,
          py::arg("reduction") = tenzor::nn::Reduction::Mean,
          py::call_guard<py::gil_scoped_release>());

    // =========================================================================
    // Functional API — stateless operation wrappers (F.dropout, F.linear, etc.)
    // =========================================================================

    nn.def("functional_dropout", [](const tenzor::Variable& input, double p, bool training) -> tenzor::Variable {
        if (!training || p == 0.0) return input;
        tenzor::nn::Dropout layer(p);
        layer.train(true);
        return layer.forward_impl(input);
    }, py::arg("input"), py::arg("p") = 0.5, py::arg("training") = true,
       "Apply dropout to input tensor");

    nn.def("functional_linear", [](const tenzor::Variable& input, const tenzor::Variable& weight,
                                    const py::object& bias) -> tenzor::Variable {
        if (bias.is_none()) {
            // y = x @ W^T (no bias)
            return tenzor::matmul(input, tenzor::transpose(weight, 0, 1));
        }
        return tenzor::linear(input, weight, bias.cast<tenzor::Variable>());
    }, py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
       "Apply linear transformation: y = xW^T + b");

    nn.def("functional_max_pool2d", [](const tenzor::Variable& input, int64_t kernel_size,
                                        int64_t stride, int64_t padding) -> tenzor::Variable {
        tenzor::nn::MaxPool2d layer(kernel_size, stride, padding);
        return layer.forward_impl(input);
    }, py::arg("input"), py::arg("kernel_size"), py::arg("stride") = -1, py::arg("padding") = 0,
       "Apply 2D max pooling");

    nn.def("functional_avg_pool2d", [](const tenzor::Variable& input, int64_t kernel_size,
                                        int64_t stride, int64_t padding) -> tenzor::Variable {
        tenzor::nn::AvgPool2d layer(kernel_size, stride, padding);
        return layer.forward_impl(input);
    }, py::arg("input"), py::arg("kernel_size"), py::arg("stride") = -1, py::arg("padding") = 0,
       "Apply 2D average pooling");

    nn.def("functional_adaptive_avg_pool2d",
          [](const tenzor::Variable& input, int64_t output_h, int64_t output_w) -> tenzor::Variable {
        tenzor::nn::AdaptiveAvgPool2d layer(output_h, output_w);
        return layer.forward_impl(input);
    }, py::arg("input"), py::arg("output_h"), py::arg("output_w"),
       "Apply 2D adaptive average pooling",
       py::call_guard<py::gil_scoped_release>());

    nn.def("functional_adaptive_max_pool2d",
          [](const tenzor::Variable& input, int64_t output_h, int64_t output_w) -> tenzor::Variable {
        tenzor::nn::AdaptiveMaxPool2d layer(output_h, output_w);
        return layer.forward_impl(input);
    }, py::arg("input"), py::arg("output_h"), py::arg("output_w"),
       "Apply 2D adaptive max pooling",
       py::call_guard<py::gil_scoped_release>());

    nn.def("functional_batch_norm", [](const tenzor::Variable& input, int64_t num_features,
                                        bool training, double momentum, double eps) -> tenzor::Variable {
        tenzor::nn::BatchNorm2d layer(num_features, eps, momentum, true, true);
        layer.train(training);
        return layer.forward_impl(input);
    }, py::arg("input"), py::arg("num_features"),
       py::arg("training") = true, py::arg("momentum") = 0.1, py::arg("eps") = 1e-5,
       "Apply batch normalization (creates fresh running stats)");

    nn.def("functional_layer_norm", [](const tenzor::Variable& input,
                                        std::vector<int64_t> normalized_shape,
                                        double eps) -> tenzor::Variable {
        tenzor::nn::LayerNorm layer(std::move(normalized_shape), eps, true);
        return layer.forward_impl(input);
    }, py::arg("input"), py::arg("normalized_shape"), py::arg("eps") = 1e-5,
       "Apply layer normalization");

    nn.def("functional_group_norm", [](const tenzor::Variable& input,
                                        int64_t num_groups, int64_t num_channels,
                                        double eps) -> tenzor::Variable {
        tenzor::nn::GroupNorm layer(num_groups, num_channels, eps, true);
        return layer.forward_impl(input);
    }, py::arg("input"), py::arg("num_groups"), py::arg("num_channels"),
       py::arg("eps") = 1e-5,
       "Apply group normalization");

    nn.def("functional_instance_norm",
          [](const tenzor::Variable& input, int64_t num_features,
             double eps, bool affine) -> tenzor::Variable {
        tenzor::nn::InstanceNorm2d layer(num_features, eps, affine);
        return layer.forward_impl(input);
    }, py::arg("input"), py::arg("num_features"),
       py::arg("eps") = 1e-5, py::arg("affine") = false,
       "Apply instance normalization",
       py::call_guard<py::gil_scoped_release>());

    nn.def("functional_rms_norm",
          [](const tenzor::Variable& input, int64_t normalized_shape,
             double eps) -> tenzor::Variable {
        tenzor::nn::RMSNorm layer(normalized_shape, eps);
        return layer.forward_impl(input);
    }, py::arg("input"), py::arg("normalized_shape"),
       py::arg("eps") = 1e-6,
       "Apply RMS normalization",
       py::call_guard<py::gil_scoped_release>());

    nn.def("functional_interpolate", [](const tenzor::Variable& input,
                                         std::vector<int64_t> size,
                                         const std::string& mode,
                                         bool align_corners) -> tenzor::Variable {
        auto result = tenzor::ops::interpolate(input.tensor(), size, mode, align_corners);
        return tenzor::Variable(result, input.requires_grad());
    }, py::arg("input"), py::arg("size"),
       py::arg("mode") = "bilinear", py::arg("align_corners") = false,
       "Interpolate/resize tensor to given size");

    nn.def("functional_embedding", [](const tenzor::Variable& input,
                                       const tenzor::Variable& weight,
                                       int64_t padding_idx) -> tenzor::Variable {
        // Create embedding layer with weight's dimensions
        auto weight_shape = weight.shape();
        tenzor::nn::Embedding layer(weight_shape[0], weight_shape[1], padding_idx);
        // Copy weight into embedding
        // Note: functional embedding creates a new Embedding layer each call
        return layer.forward_impl(input);
    }, py::arg("input"), py::arg("weight"), py::arg("padding_idx") = -1,
       "Lookup embeddings from weight matrix");

    nn.def("functional_binary_cross_entropy_with_logits",
           [](const tenzor::Variable& input, const tenzor::Variable& target,
              tenzor::nn::Reduction reduction) -> tenzor::Variable {
        tenzor::nn::BCEWithLogitsLoss loss(reduction);
        return loss.forward(input, target);
    }, py::arg("input"), py::arg("target"),
       py::arg("reduction") = tenzor::nn::Reduction::Mean,
       "Binary cross entropy with logits loss");

    // New nn.functional wrappers using the C++ functional namespace
    nn.def("functional_nll_loss", &tenzor::nn::functional::nll_loss,
           py::arg("input"), py::arg("target"),
           py::arg("reduction") = tenzor::nn::Reduction::Mean,
           "Negative log likelihood loss",
           py::call_guard<py::gil_scoped_release>());

    nn.def("functional_smooth_l1_loss", &tenzor::nn::functional::smooth_l1_loss,
           py::arg("input"), py::arg("target"),
           py::arg("reduction") = tenzor::nn::Reduction::Mean,
           py::arg("beta") = 1.0,
           "Smooth L1 (Huber) loss",
           py::call_guard<py::gil_scoped_release>());

    nn.def("functional_cosine_similarity", &tenzor::nn::functional::cosine_similarity,
           py::arg("x1"), py::arg("x2"),
           py::arg("dim") = 1, py::arg("eps") = 1e-8,
           "Cosine similarity between tensors",
           py::call_guard<py::gil_scoped_release>());

    nn.def("functional_conv2d", &tenzor::nn::functional::conv2d,
           py::arg("input"), py::arg("weight"),
           py::arg("bias") = std::nullopt,
           py::arg("stride") = std::make_pair(1LL, 1LL),
           py::arg("padding") = std::make_pair(0LL, 0LL),
           py::arg("dilation") = std::make_pair(1LL, 1LL),
           py::arg("groups") = 1,
           "Functional 2D convolution",
           py::call_guard<py::gil_scoped_release>());

    nn.def("functional_normalize", &tenzor::nn::functional::normalize,
           py::arg("input"), py::arg("p") = 2.0, py::arg("dim") = 1,
           py::arg("eps") = 1e-12,
           "L_p normalize input along a dimension",
           py::call_guard<py::gil_scoped_release>());

    nn.def("functional_pad", &tenzor::nn::functional::pad,
           py::arg("input"), py::arg("pad"), py::arg("mode") = "constant",
           py::arg("value") = 0.0,
           "Pad a tensor (constant mode)",
           py::call_guard<py::gil_scoped_release>());

    nn.def("functional_scaled_dot_product_attention",
           [](const tenzor::Variable& query, const tenzor::Variable& key,
              const tenzor::Variable& value,
              std::optional<tenzor::Variable> attn_mask,
              double dropout_p, bool is_causal) -> tenzor::Variable {
               tenzor::nn::functional::SDPAOptions opts;
               opts.attn_mask = attn_mask;
               opts.dropout_p = dropout_p;
               opts.is_causal = is_causal;
               return tenzor::nn::functional::scaled_dot_product_attention(
                   query, key, value, opts);
           },
           py::arg("query"), py::arg("key"), py::arg("value"),
           py::arg("attn_mask") = std::nullopt,
           py::arg("dropout_p") = 0.0, py::arg("is_causal") = false,
           "Scaled dot-product attention",
           py::call_guard<py::gil_scoped_release>());

    // Gradient clipping utilities
    nn.def("clip_grad_norm_", &tenzor::nn::utils::clip_grad_norm_,
           py::arg("parameters"), py::arg("max_norm"), py::arg("norm_type") = 2.0,
           "Clip gradients by global norm, returns total norm before clipping");

    nn.def("clip_grad_value_", &tenzor::nn::utils::clip_grad_value_,
           py::arg("parameters"), py::arg("clip_value"),
           "Clip gradient values to [-clip_value, clip_value]");

    // PackedSequence and RNN utilities
    py::class_<tenzor::nn::PackedSequence>(nn, "PackedSequence")
        .def_readwrite("data", &tenzor::nn::PackedSequence::data)
        .def_readwrite("batch_sizes", &tenzor::nn::PackedSequence::batch_sizes)
        .def_readwrite("sorted_indices", &tenzor::nn::PackedSequence::sorted_indices)
        .def_readwrite("unsorted_indices", &tenzor::nn::PackedSequence::unsorted_indices);

    nn.def("pack_padded_sequence", &tenzor::nn::pack_padded_sequence,
           py::arg("input"), py::arg("lengths"),
           py::arg("batch_first") = false, py::arg("enforce_sorted") = true,
           "Pack a padded batch of variable-length sequences");

    nn.def("pad_packed_sequence", &tenzor::nn::pad_packed_sequence,
           py::arg("packed"), py::arg("batch_first") = false,
           py::arg("padding_value") = 0.0f, py::arg("total_length") = -1,
           "Unpack a PackedSequence back to padded tensor");

    nn.def("pack_sequence", &tenzor::nn::pack_sequence,
           py::arg("sequences"), py::arg("enforce_sorted") = true,
           "Pack a list of variable-length tensors");

    // Optimizers
    auto optim = m.def_submodule("optim", "Optimization algorithms");

    // ClipMode enum for gradient clipping
    py::enum_<tenzor::optim::ClipMode>(optim, "ClipMode",
        "Gradient clipping mode")
        .value("NONE", tenzor::optim::ClipMode::None, "No gradient clipping")
        .value("NORM", tenzor::optim::ClipMode::Norm, "Clip by global norm (L2)")
        .value("VALUE", tenzor::optim::ClipMode::Value, "Clip by value (element-wise clamping)")
        .export_values();

    // ClipConfig struct for gradient clipping configuration
    py::class_<tenzor::optim::ClipConfig>(optim, "ClipConfig",
        "Configuration for automatic gradient clipping in optimizers")
        .def(py::init<>())
        .def(py::init([](tenzor::optim::ClipMode mode, double max_norm, double norm_type) {
            return tenzor::optim::ClipConfig{mode, max_norm, norm_type};
        }), py::arg("mode"), py::arg("max_norm") = 1.0, py::arg("norm_type") = 2.0)
        .def_readwrite("mode", &tenzor::optim::ClipConfig::mode,
             "Clipping mode")
        .def_readwrite("max_norm", &tenzor::optim::ClipConfig::max_norm,
             "Maximum norm for Norm mode, or max absolute value for Value mode")
        .def_readwrite("norm_type", &tenzor::optim::ClipConfig::norm_type,
             "Norm type for Norm mode (default: L2)")
        .def("__repr__", [](const tenzor::optim::ClipConfig& self) {
            std::string mode_str = "None";
            if (self.mode == tenzor::optim::ClipMode::Norm) mode_str = "Norm";
            else if (self.mode == tenzor::optim::ClipMode::Value) mode_str = "Value";
            return "ClipConfig(mode=" + mode_str +
                   ", max_norm=" + std::to_string(self.max_norm) +
                   ", norm_type=" + std::to_string(self.norm_type) + ")";
        });

    // ParamGroup struct for per-group hyperparameters
    py::class_<tenzor::optim::ParamGroup>(optim, "ParamGroup",
        "Parameter group with individual learning rate and weight decay")
        .def(py::init([](std::vector<std::shared_ptr<tenzor::Variable>> params, double lr, double weight_decay) {
            return tenzor::optim::ParamGroup{std::move(params), lr, weight_decay};
        }), py::arg("params"), py::arg("lr"), py::arg("weight_decay") = 0.0)
        .def_readwrite("params", &tenzor::optim::ParamGroup::params,
             "Parameters in this group")
        .def_readwrite("lr", &tenzor::optim::ParamGroup::lr,
             "Learning rate for this group")
        .def_readwrite("weight_decay", &tenzor::optim::ParamGroup::weight_decay,
             "Weight decay (L2 regularization) for this group")
        .def("__repr__", [](const tenzor::optim::ParamGroup& self) {
            return "ParamGroup(params=" + std::to_string(self.params.size()) +
                   ", lr=" + std::to_string(self.lr) +
                   ", weight_decay=" + std::to_string(self.weight_decay) + ")";
        });

    // Optimizer base class - needed for functions that accept any optimizer
    py::class_<tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::Optimizer>>(optim, "Optimizer",
        "Base class for all optimizers")
        .def("zero_grad", &tenzor::optim::Optimizer::zero_grad, "Zero out all parameter gradients")
        .def("state_dict", &tenzor::optim::Optimizer::state_dict, "Get optimizer state dictionary")
        .def("load_state_dict", &tenzor::optim::Optimizer::load_state_dict, py::arg("state"),
             "Load optimizer state dictionary")
        .def("add_param_group", &tenzor::optim::Optimizer::add_param_group,
             py::arg("group"), "Add a parameter group with custom hyperparameters")
        .def("param_groups", static_cast<std::vector<tenzor::optim::ParamGroup>& (tenzor::optim::Optimizer::*)()>(
             &tenzor::optim::Optimizer::param_groups),
             py::return_value_policy::reference_internal,
             "Get all parameter groups")
        .def("set_clip_config", &tenzor::optim::Optimizer::set_clip_config,
             py::arg("config"), "Set gradient clipping configuration")
        .def("clip_config", &tenzor::optim::Optimizer::clip_config,
             py::return_value_policy::reference_internal,
             "Get current gradient clipping configuration");

    py::class_<tenzor::optim::SGD, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::SGD>>(optim, "SGD")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, bool>(),
             py::arg("params"), py::arg("lr"),
             py::arg("momentum") = 0.0, py::arg("dampening") = 0.0,
             py::arg("weight_decay") = 0.0, py::arg("nesterov") = false)
        .def("step", [](tenzor::optim::SGD& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) {
                return py::cast(self.step(*closure));
            }
            self.step();
            return py::none();
        }, py::arg("closure") = py::none(),
           "Perform optimization step. Optionally takes a closure that recomputes the loss.")
        .def("zero_grad", &tenzor::optim::SGD::zero_grad)
        .def("set_lr", &tenzor::optim::SGD::set_lr,
             py::arg("lr"), "Set learning rate")
        .def("get_lr", &tenzor::optim::SGD::get_lr,
             "Get current learning rate")
        .def("state_dict", &tenzor::optim::SGD::state_dict,
             "Get optimizer state dictionary")
        .def("load_state_dict", &tenzor::optim::SGD::load_state_dict,
             py::arg("state"), "Load optimizer state dictionary");

    py::class_<tenzor::optim::Adam, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::Adam>>(optim, "Adam")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double, bool>(),
             py::arg("params"), py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             py::arg("amsgrad") = false)
        .def("step", [](tenzor::optim::Adam& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            self.step(); return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::Adam::zero_grad)
        .def("set_lr", &tenzor::optim::Adam::set_lr,
             py::arg("lr"), "Set learning rate")
        .def("get_lr", &tenzor::optim::Adam::get_lr,
             "Get current learning rate")
        .def("state_dict", &tenzor::optim::Adam::state_dict,
             "Get optimizer state dictionary")
        .def("load_state_dict", &tenzor::optim::Adam::load_state_dict,
             py::arg("state"), "Load optimizer state dictionary");

    py::class_<tenzor::optim::AdamW, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::AdamW>>(optim, "AdamW")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double, bool>(),
             py::arg("params"), py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.01,
             py::arg("amsgrad") = false)
        .def("step", [](tenzor::optim::AdamW& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            self.step(); return py::none();
        }, py::arg("closure") = py::none())
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
        .def("step", [](tenzor::optim::RMSprop& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            self.step(); return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::RMSprop::zero_grad)
        .def("state_dict", &tenzor::optim::RMSprop::state_dict)
        .def("load_state_dict", &tenzor::optim::RMSprop::load_state_dict);

    py::class_<tenzor::optim::Adagrad>(optim, "Adagrad")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 0.01, py::arg("lr_decay") = 0.0,
             py::arg("weight_decay") = 0.0, py::arg("initial_accumulator_value") = 0.0,
             py::arg("eps") = 1e-10)
        .def("step", [](tenzor::optim::Adagrad& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            self.step(); return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::Adagrad::zero_grad);

    py::class_<tenzor::optim::Adadelta>(optim, "Adadelta")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 1.0, py::arg("rho") = 0.9,
             py::arg("eps") = 1e-6, py::arg("weight_decay") = 0.0)
        .def("step", [](tenzor::optim::Adadelta& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            self.step(); return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::Adadelta::zero_grad);

    py::class_<tenzor::optim::RAdam, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::RAdam>>(optim, "RAdam",
        "Rectified Adam optimizer (no warmup needed)")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0)
        .def("step", [](tenzor::optim::RAdam& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            self.step(); return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::RAdam::zero_grad)
        .def("set_lr", &tenzor::optim::RAdam::set_lr, py::arg("lr"))
        .def("get_lr", &tenzor::optim::RAdam::get_lr)
        .def("state_dict", &tenzor::optim::RAdam::state_dict)
        .def("load_state_dict", &tenzor::optim::RAdam::load_state_dict, py::arg("state"));

    py::class_<tenzor::optim::LAMB, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::LAMB>>(optim, "LAMB",
        "LAMB optimizer for large-batch training")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-6, py::arg("weight_decay") = 0.01)
        .def("step", [](tenzor::optim::LAMB& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            self.step(); return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::LAMB::zero_grad)
        .def("set_lr", &tenzor::optim::LAMB::set_lr, py::arg("lr"))
        .def("get_lr", &tenzor::optim::LAMB::get_lr)
        .def("state_dict", &tenzor::optim::LAMB::state_dict)
        .def("load_state_dict", &tenzor::optim::LAMB::load_state_dict, py::arg("state"));

    py::class_<tenzor::optim::SparseAdam, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::SparseAdam>>(optim, "SparseAdam",
        "SparseAdam optimizer for efficient embedding training with sparse gradients")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8)
        .def("step", [](tenzor::optim::SparseAdam& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            self.step(); return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::SparseAdam::zero_grad)
        .def("set_lr", &tenzor::optim::SparseAdam::set_lr, py::arg("lr"))
        .def("get_lr", &tenzor::optim::SparseAdam::get_lr)
        .def("state_dict", &tenzor::optim::SparseAdam::state_dict)
        .def("load_state_dict", &tenzor::optim::SparseAdam::load_state_dict, py::arg("state"));

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
        .def(py::init<tenzor::optim::RMSprop&, int, double>(),
             py::arg("optimizer"),
             py::arg("step_size"),
             py::arg("gamma") = 0.1)
        .def(py::init<tenzor::optim::Adagrad&, int, double>(),
             py::arg("optimizer"),
             py::arg("step_size"),
             py::arg("gamma") = 0.1)
        .def(py::init<tenzor::optim::Adadelta&, int, double>(),
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
        .def(py::init<tenzor::optim::RMSprop&, double>(),
             py::arg("optimizer"),
             py::arg("gamma"))
        .def(py::init<tenzor::optim::Adagrad&, double>(),
             py::arg("optimizer"),
             py::arg("gamma"))
        .def(py::init<tenzor::optim::Adadelta&, double>(),
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
        .def(py::init<tenzor::optim::RMSprop&, int, double>(),
             py::arg("optimizer"),
             py::arg("T_max"),
             py::arg("eta_min") = 0.0)
        .def(py::init<tenzor::optim::Adagrad&, int, double>(),
             py::arg("optimizer"),
             py::arg("T_max"),
             py::arg("eta_min") = 0.0)
        .def(py::init<tenzor::optim::Adadelta&, int, double>(),
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

    // ========================================================================
    // Training Callbacks
    // ========================================================================

    // Base Callback class
    py::class_<tenzor::nn::Callback, std::shared_ptr<tenzor::nn::Callback>>(nn, "Callback",
        "Base callback interface for training loop hooks")
        .def(py::init<>())
        .def("on_epoch_begin", &tenzor::nn::Callback::on_epoch_begin,
             py::arg("epoch"),
             "Called at the beginning of each epoch")
        .def("on_epoch_end", &tenzor::nn::Callback::on_epoch_end,
             py::arg("epoch"), py::arg("train_loss"), py::arg("val_loss"),
             "Called at the end of each epoch")
        .def("on_batch_begin", &tenzor::nn::Callback::on_batch_begin,
             py::arg("batch_idx"),
             "Called at the beginning of each batch")
        .def("on_batch_end", &tenzor::nn::Callback::on_batch_end,
             py::arg("batch_idx"), py::arg("loss"),
             "Called at the end of each batch")
        .def("on_train_begin", &tenzor::nn::Callback::on_train_begin,
             "Called at the beginning of training")
        .def("on_train_end", &tenzor::nn::Callback::on_train_end,
             "Called at the end of training");

    // ProgressCallback
    py::class_<tenzor::nn::ProgressCallback, tenzor::nn::Callback,
               std::shared_ptr<tenzor::nn::ProgressCallback>>(nn, "ProgressCallback",
        "Callback for printing training progress with progress bars and loss summaries")
        .def(py::init<int>(),
             py::arg("print_every") = 1,
             "Create ProgressCallback that prints every N batches")
        .def("set_total_batches", &tenzor::nn::ProgressCallback::set_total_batches,
             py::arg("total"),
             "Set total number of batches per epoch for progress display")
        .def("set_total_epochs", &tenzor::nn::ProgressCallback::set_total_epochs,
             py::arg("total"),
             "Set total number of epochs for progress display");

    // EarlyStoppingCallback
    py::class_<tenzor::nn::EarlyStoppingCallback, tenzor::nn::Callback,
               std::shared_ptr<tenzor::nn::EarlyStoppingCallback>>(nn, "EarlyStoppingCallback",
        "Callback for early stopping based on validation loss")
        .def(py::init<int, float, const std::string&>(),
             py::arg("patience") = 5,
             py::arg("min_delta") = 0.0f,
             py::arg("monitor") = "val_loss",
             "Create EarlyStoppingCallback\n\n"
             "Args:\n"
             "    patience: Number of epochs with no improvement before stopping\n"
             "    min_delta: Minimum change to qualify as improvement\n"
             "    monitor: Metric to monitor ('val_loss' or 'train_loss')")
        .def("should_stop", &tenzor::nn::EarlyStoppingCallback::should_stop,
             "Check if training should stop")
        .def("best_loss", &tenzor::nn::EarlyStoppingCallback::best_loss,
             "Get best loss value seen so far")
        .def("wait_count", &tenzor::nn::EarlyStoppingCallback::wait_count,
             "Get number of epochs since last improvement");

    // ModelCheckpointCallback
    py::class_<tenzor::nn::ModelCheckpointCallback, tenzor::nn::Callback,
               std::shared_ptr<tenzor::nn::ModelCheckpointCallback>>(nn, "ModelCheckpointCallback",
        "Callback for saving model checkpoints during training")
        .def(py::init<const std::string&, std::shared_ptr<tenzor::nn::Module>, bool, const std::string&>(),
             py::arg("filepath"),
             py::arg("model"),
             py::arg("save_best_only") = true,
             py::arg("monitor") = "val_loss",
             "Create ModelCheckpointCallback\n\n"
             "Args:\n"
             "    filepath: Path template for checkpoint files (can include {epoch} or {epoch:03d})\n"
             "    model: Model to save\n"
             "    save_best_only: If True, only save when validation loss improves\n"
             "    monitor: Metric to monitor for best model ('val_loss' or 'train_loss')")
        .def("best_loss", &tenzor::nn::ModelCheckpointCallback::best_loss,
             "Get best loss value for saved model")
        .def("last_checkpoint", &tenzor::nn::ModelCheckpointCallback::last_checkpoint,
             "Get path of last saved checkpoint");

    // LRSchedulerCallback
    py::class_<tenzor::nn::LRSchedulerCallback, tenzor::nn::Callback,
               std::shared_ptr<tenzor::nn::LRSchedulerCallback>>(nn, "LRSchedulerCallback",
        "Callback for adjusting learning rate during training")
        .def(py::init<std::shared_ptr<tenzor::optim::Optimizer>, const std::string&, float, int, float, int>(),
             py::arg("optimizer"),
             py::arg("schedule_type") = "step",
             py::arg("decay_factor") = 0.1f,
             py::arg("decay_epochs") = 10,
             py::arg("min_lr") = 0.0f,
             py::arg("patience") = 5,
             "Create LRSchedulerCallback\n\n"
             "Args:\n"
             "    optimizer: Optimizer to adjust learning rate for\n"
             "    schedule_type: Type of schedule ('step', 'exponential', 'cosine', 'plateau')\n"
             "    decay_factor: Factor to multiply learning rate by\n"
             "    decay_epochs: For 'step': decay every N epochs. For 'cosine': total epochs\n"
             "    min_lr: Minimum learning rate\n"
             "    patience: For 'plateau': epochs to wait before reducing LR")
        .def("current_lr", &tenzor::nn::LRSchedulerCallback::current_lr,
             "Get current learning rate");

    // CallbackList
    py::class_<tenzor::nn::CallbackList>(nn, "CallbackList",
        "Collection of callbacks for training")
        .def(py::init<>())
        .def("add", &tenzor::nn::CallbackList::add,
             py::arg("callback"),
             "Add a callback to the list")
        .def("on_epoch_begin", &tenzor::nn::CallbackList::on_epoch_begin,
             py::arg("epoch"))
        .def("on_epoch_end", &tenzor::nn::CallbackList::on_epoch_end,
             py::arg("epoch"), py::arg("train_loss"), py::arg("val_loss"))
        .def("on_batch_begin", &tenzor::nn::CallbackList::on_batch_begin,
             py::arg("batch_idx"))
        .def("on_batch_end", &tenzor::nn::CallbackList::on_batch_end,
             py::arg("batch_idx"), py::arg("loss"))
        .def("on_train_begin", &tenzor::nn::CallbackList::on_train_begin)
        .def("on_train_end", &tenzor::nn::CallbackList::on_train_end)
        .def("callbacks", &tenzor::nn::CallbackList::callbacks,
             "Get all callbacks");

    // ========================================================================
    // Model Checkpointing
    // ========================================================================

    // TrainingMetadata
    py::class_<tenzor::nn::TrainingMetadata>(nn, "TrainingMetadata",
        "Training metadata stored with checkpoints")
        .def(py::init<>())
        .def_readwrite("epoch", &tenzor::nn::TrainingMetadata::epoch,
                      "Current training epoch")
        .def_readwrite("global_step", &tenzor::nn::TrainingMetadata::global_step,
                      "Total training steps")
        .def_readwrite("learning_rate", &tenzor::nn::TrainingMetadata::learning_rate,
                      "Current learning rate")
        .def_readwrite("train_loss", &tenzor::nn::TrainingMetadata::train_loss,
                      "Last training loss")
        .def_readwrite("val_loss", &tenzor::nn::TrainingMetadata::val_loss,
                      "Last validation loss")
        .def_readwrite("train_accuracy", &tenzor::nn::TrainingMetadata::train_accuracy,
                      "Last training accuracy")
        .def_readwrite("val_accuracy", &tenzor::nn::TrainingMetadata::val_accuracy,
                      "Last validation accuracy")
        .def_readwrite("best_val_loss", &tenzor::nn::TrainingMetadata::best_val_loss,
                      "Best validation loss")
        .def_readwrite("best_val_accuracy", &tenzor::nn::TrainingMetadata::best_val_accuracy,
                      "Best validation accuracy")
        .def_readwrite("timestamp", &tenzor::nn::TrainingMetadata::timestamp,
                      "Checkpoint creation time")
        .def_readwrite("custom_metrics", &tenzor::nn::TrainingMetadata::custom_metrics,
                      "User-defined metrics")
        .def("to_dict", &tenzor::nn::TrainingMetadata::to_dict,
             "Serialize metadata to dictionary")
        .def("from_dict", &tenzor::nn::TrainingMetadata::from_dict,
             py::arg("dict"),
             "Deserialize metadata from dictionary");

    // CheckpointConfig
    py::class_<tenzor::nn::CheckpointConfig>(nn, "CheckpointConfig",
        "Checkpoint configuration")
        .def(py::init<>())
        .def_readwrite("save_optimizer", &tenzor::nn::CheckpointConfig::save_optimizer,
                      "Include optimizer state")
        .def_readwrite("save_scheduler", &tenzor::nn::CheckpointConfig::save_scheduler,
                      "Include scheduler state")
        .def_readwrite("verify_checksum", &tenzor::nn::CheckpointConfig::verify_checksum,
                      "Verify data integrity")
        .def_readwrite("atomic_save", &tenzor::nn::CheckpointConfig::atomic_save,
                      "Use atomic writes");

    // Checkpoint
    py::class_<tenzor::nn::Checkpoint>(nn, "Checkpoint",
        "Complete checkpoint data structure")
        .def(py::init<>())
        .def_readwrite("version", &tenzor::nn::Checkpoint::version,
                      "Format version")
        .def_readwrite("model_state", &tenzor::nn::Checkpoint::model_state,
                      "Model parameters and buffers")
        .def_readwrite("optimizer_state", &tenzor::nn::Checkpoint::optimizer_state,
                      "Optimizer state")
        .def_readwrite("scheduler_state", &tenzor::nn::Checkpoint::scheduler_state,
                      "Scheduler state")
        .def_readwrite("metadata", &tenzor::nn::Checkpoint::metadata,
                      "Training metadata")
        .def_readwrite("config", &tenzor::nn::Checkpoint::config,
                      "Checkpoint configuration")
        .def("size_bytes", &tenzor::nn::Checkpoint::size_bytes,
             "Get total size of checkpoint in bytes")
        .def("is_valid", &tenzor::nn::Checkpoint::is_valid,
             "Check if checkpoint is valid");

    // ModelCheckpoint
    py::class_<tenzor::nn::ModelCheckpoint>(nn, "ModelCheckpoint",
        R"pbdoc(
            Model checkpoint manager.

            Handles saving and loading of model checkpoints with support for:
            - Model state (parameters, buffers)
            - Optimizer state (momentum, adaptive learning rates, etc.)
            - Scheduler state (step counts, learning rate history)
            - Training metadata (epoch, loss, metrics)
            - Versioning and backward compatibility
            - Optional compression
            - Atomic writes for crash safety

            Example:
                >>> checkpoint_manager = tenzor.nn.ModelCheckpoint()
                >>> # Save model
                >>> metadata = tenzor.nn.TrainingMetadata()
                >>> metadata.epoch = 10
                >>> metadata.train_loss = 0.25
                >>> checkpoint_manager.save_model("model.pt", model, metadata)
                >>>
                >>> # Load model
                >>> state_dict = checkpoint_manager.load_model("model.pt")
                >>> model.load_state_dict(state_dict)
        )pbdoc")
        .def(py::init<>())
        .def(py::init<tenzor::nn::CheckpointConfig>(),
             py::arg("config"),
             "Create ModelCheckpoint with custom configuration")
        .def("save", &tenzor::nn::ModelCheckpoint::save,
             py::arg("path"),
             py::arg("module"),
             py::arg("optimizer") = nullptr,
             py::arg("scheduler") = nullptr,
             py::arg("metadata") = tenzor::nn::TrainingMetadata{},
             R"pbdoc(
                 Save complete checkpoint to file.

                 Args:
                     path: File path for checkpoint
                     module: Model to save
                     optimizer: Optimizer to save (optional)
                     scheduler: Learning rate scheduler to save (optional)
                     metadata: Training metadata (optional)

                 Example:
                     >>> checkpoint_manager.save(
                     ...     "checkpoint.pt",
                     ...     model,
                     ...     optimizer,
                     ...     scheduler,
                     ...     metadata
                     ... )
             )pbdoc")
        .def("load", &tenzor::nn::ModelCheckpoint::load,
             py::arg("path"),
             "Load complete checkpoint from file")
        .def("save_model", &tenzor::nn::ModelCheckpoint::save_model,
             py::arg("path"),
             py::arg("module"),
             py::arg("metadata") = tenzor::nn::TrainingMetadata{},
             "Save only model state (no optimizer/scheduler)")
        .def("load_model", &tenzor::nn::ModelCheckpoint::load_model,
             py::arg("path"),
             "Load only model state")
        .def("verify_checkpoint", &tenzor::nn::ModelCheckpoint::verify_checkpoint,
             py::arg("path"),
             "Verify checkpoint file integrity")
        .def("get_metadata", &tenzor::nn::ModelCheckpoint::get_metadata,
             py::arg("path"),
             "Get checkpoint metadata without loading full checkpoint")
        .def("get_version", &tenzor::nn::ModelCheckpoint::get_version,
             py::arg("path"),
             "Get checkpoint version")
        .def("is_compatible", &tenzor::nn::ModelCheckpoint::is_compatible,
             py::arg("path"),
             "Check if checkpoint is compatible with current version")
        .def("config", &tenzor::nn::ModelCheckpoint::config,
             "Get current configuration")
        .def("set_config", &tenzor::nn::ModelCheckpoint::set_config,
             py::arg("config"),
             "Set configuration");

    // AutoCheckpoint
    py::class_<tenzor::nn::AutoCheckpoint>(nn, "AutoCheckpoint",
        R"pbdoc(
            Automatic checkpoint manager for training loops.

            Automatically saves checkpoints at specified intervals and
            keeps only the best N checkpoints based on a metric.

            Features:
            - Save every N epochs
            - Save every N steps
            - Keep top K checkpoints by metric
            - Early stopping integration
            - Automatic cleanup of old checkpoints

            Example:
                >>> auto_checkpoint = tenzor.nn.AutoCheckpoint("./checkpoints", max_checkpoints=5)
                >>> auto_checkpoint.set_metric_mode("min")  # Lower is better
                >>>
                >>> for epoch in range(num_epochs):
                ...     # Training code...
                ...     val_loss = validate(model)
                ...
                ...     # Automatically saves and manages checkpoints
                ...     auto_checkpoint.step(
                ...         model,
                ...         optimizer,
                ...         epoch,
                ...         val_loss,
                ...         "val_loss",
                ...         scheduler
                ...     )
                >>>
                >>> # Get path to best checkpoint
                >>> best_path = auto_checkpoint.best_checkpoint_path()
        )pbdoc")
        .def(py::init<std::string, int, int>(),
             py::arg("directory"),
             py::arg("max_checkpoints") = 3,
             py::arg("save_frequency") = 1,
             R"pbdoc(
                 Create auto checkpoint manager.

                 Args:
                     directory: Directory to save checkpoints
                     max_checkpoints: Maximum number of checkpoints to keep (default: 3)
                     save_frequency: Save every N epochs (default: 1)
             )pbdoc")
        .def("step", &tenzor::nn::AutoCheckpoint::step,
             py::arg("module"),
             py::arg("optimizer"),
             py::arg("epoch"),
             py::arg("metric_value"),
             py::arg("metric_name"),
             py::arg("scheduler") = nullptr,
             R"pbdoc(
                 Step function to call after each epoch/step.

                 Args:
                     module: Model to save
                     optimizer: Optimizer to save
                     epoch: Current epoch number
                     metric_value: Current metric value
                     metric_name: Metric name for tracking
                     scheduler: Optional scheduler to save

                 Returns:
                     True if checkpoint was saved
             )pbdoc")
        .def("set_metric_mode", &tenzor::nn::AutoCheckpoint::set_metric_mode,
             py::arg("mode"),
             R"pbdoc(
                 Set metric optimization mode.

                 Args:
                     mode: "min" or "max" (default: "min")

                 Example:
                     >>> auto_checkpoint.set_metric_mode("min")  # For loss
                     >>> auto_checkpoint.set_metric_mode("max")  # For accuracy
             )pbdoc")
        .def("best_checkpoint_path", &tenzor::nn::AutoCheckpoint::best_checkpoint_path,
             "Get path to best checkpoint")
        .def("best_metric_value", &tenzor::nn::AutoCheckpoint::best_metric_value,
             "Get best metric value")
        .def("checkpoint_paths", &tenzor::nn::AutoCheckpoint::checkpoint_paths,
             "Get list of all checkpoint paths")
        .def("cleanup", &tenzor::nn::AutoCheckpoint::cleanup,
             "Clean up old checkpoints (keep only top K)");

    // ========================================================================
    // High-Level Training API
    // ========================================================================

    // DataLoader class for simple batch iteration
    py::class_<tenzor::nn::DataLoader>(nn, "SimpleDataLoader",
        R"pbdoc(
            Simple DataLoader for iterating over batches of data.

            Provides basic iterator interface for training/validation data batches.
            For more advanced features (shuffling, multi-threading, etc.), use
            tenzor.data.DataLoader instead.

            Args:
                data: List of (input, target) tensor pairs
                batch_size: Number of samples per batch

            Example:
                >>> data = [(input1, target1), (input2, target2), ...]
                >>> loader = tenzor.nn.SimpleDataLoader(data, batch_size=32)
                >>> for inputs, targets in loader:
                ...     loss = model.train_step(inputs, targets)
        )pbdoc")
        .def(py::init<std::vector<std::pair<tenzor::Tensor, tenzor::Tensor>>, size_t>(),
             py::arg("data"), py::arg("batch_size"),
             "Create DataLoader with data and batch size")
        .def("__iter__", [](tenzor::nn::DataLoader& loader) {
            return py::make_iterator(loader.begin(), loader.end());
        }, py::keep_alive<0, 1>())
        .def("size", &tenzor::nn::DataLoader::size,
             "Get number of batches");

    // NeuralNetwork high-level training wrapper
    py::class_<tenzor::nn::NeuralNetwork, std::shared_ptr<tenzor::nn::NeuralNetwork>>(nn, "NeuralNetwork",
        R"pbdoc(
            High-level neural network training wrapper.

            NeuralNetwork provides a complete training API that wraps a model, optimizer,
            and loss function. It handles the standard training loop pattern automatically.

            Features:
            - Single-call training step with train_step()
            - Evaluation without gradients via eval_step()
            - Complete training loop with fit()
            - Automatic mode switching (train/eval)
            - Validation support
            - Callback system for monitoring

            Args:
                model: Neural network model (any Module subclass)
                optimizer: Optimization algorithm (SGD, Adam, etc.)
                loss_fn: Loss function module (MSELoss, CrossEntropyLoss, etc.)

            Example:
                >>> # Create model, optimizer, and loss
                >>> model = tenzor.nn.Sequential(
                ...     tenzor.nn.Linear(784, 128),
                ...     tenzor.nn.ReLU(),
                ...     tenzor.nn.Linear(128, 10)
                ... )
                >>> optimizer = tenzor.optim.Adam(model.parameters(), lr=0.001)
                >>> loss_fn = tenzor.nn.CrossEntropyLoss()
                >>>
                >>> # Wrap in NeuralNetwork
                >>> nn_wrapper = tenzor.nn.NeuralNetwork(model, optimizer, loss_fn)
                >>>
                >>> # Train for 10 epochs
                >>> train_loader = tenzor.nn.SimpleDataLoader(train_data, batch_size=32)
                >>> val_loader = tenzor.nn.SimpleDataLoader(val_data, batch_size=32)
                >>> nn_wrapper.fit(train_loader, epochs=10, val_loader=val_loader)
        )pbdoc")
        .def(py::init([](std::shared_ptr<tenzor::nn::Module> model,
                        std::shared_ptr<tenzor::optim::Optimizer> optimizer,
                        py::object loss_fn_obj) {
            // Create a lambda that wraps the Python loss function
            auto loss_fn = [loss_fn_obj](const tenzor::Variable& pred, const tenzor::Variable& target) -> tenzor::Variable {
                // Call the Python loss function
                py::object result = loss_fn_obj(pred, target);
                return py::cast<tenzor::Variable>(result);
            };
            return std::make_shared<tenzor::nn::NeuralNetwork>(model, optimizer, loss_fn);
        }),
             py::arg("model"), py::arg("optimizer"), py::arg("loss_fn"),
             "Create NeuralNetwork with model, optimizer, and loss function")
        .def("train_step", &tenzor::nn::NeuralNetwork::train_step,
             py::arg("input"), py::arg("target"),
             R"pbdoc(
                Perform single training step.

                Executes complete training iteration:
                1. Forward pass through model
                2. Loss computation
                3. Backward pass (gradient computation)
                4. Parameter update

                Args:
                    input: Input batch variable
                    target: Target batch variable

                Returns:
                    Loss value as float

                Example:
                    >>> loss = nn_wrapper.train_step(input_var, target_var)
                    >>> print(f"Loss: {loss:.4f}")
             )pbdoc")
        .def("eval_step", &tenzor::nn::NeuralNetwork::eval_step,
             py::arg("input"), py::arg("target"),
             R"pbdoc(
                Perform single evaluation step.

                Executes evaluation without gradient computation:
                1. Set model to evaluation mode
                2. Disable gradients (more efficient)
                3. Forward pass through model
                4. Loss computation
                5. Return loss value

                Args:
                    input: Input batch variable
                    target: Target batch variable

                Returns:
                    Loss value as float

                Example:
                    >>> val_loss = nn_wrapper.eval_step(val_input, val_target)
                    >>> print(f"Validation Loss: {val_loss:.4f}")
             )pbdoc")
        .def("fit", &tenzor::nn::NeuralNetwork::fit,
             py::arg("train_loader"), py::arg("epochs"),
             py::arg("val_loader") = nullptr,
             py::arg("callbacks") = std::vector<std::shared_ptr<tenzor::nn::Callback>>{},
             R"pbdoc(
                Train model for multiple epochs.

                Complete training loop with:
                - Epoch iteration
                - Training batch processing
                - Optional validation after each epoch
                - Callback invocation for monitoring
                - Automatic mode switching

                Args:
                    train_loader: DataLoader for training data
                    epochs: Number of epochs to train
                    val_loader: Optional DataLoader for validation (default: None)
                    callbacks: Optional list of callbacks for monitoring (default: [])

                Example:
                    >>> # Basic training
                    >>> nn_wrapper.fit(train_loader, epochs=10)
                    >>>
                    >>> # With validation
                    >>> nn_wrapper.fit(train_loader, epochs=10, val_loader=val_loader)
                    >>>
                    >>> # With callbacks
                    >>> progress = tenzor.nn.ProgressCallback()
                    >>> nn_wrapper.fit(train_loader, epochs=10, val_loader=val_loader, callbacks=[progress])
             )pbdoc")
        .def("train", &tenzor::nn::NeuralNetwork::train,
             "Set model to training mode")
        .def("eval", &tenzor::nn::NeuralNetwork::eval,
             "Set model to evaluation mode")
        .def("is_training", &tenzor::nn::NeuralNetwork::is_training,
             "Check if model is in training mode")
        .def_property_readonly("model", &tenzor::nn::NeuralNetwork::model,
             "Get underlying model")
        .def_property_readonly("optimizer", &tenzor::nn::NeuralNetwork::optimizer,
             "Get optimizer");

    // ========================================================================
    // Distributed training
    // ========================================================================
    auto distributed = m.def_submodule("distributed", "Distributed training");

    py::enum_<tenzor::distributed::ReduceOp>(distributed, "ReduceOp")
        .value("SUM", tenzor::distributed::ReduceOp::SUM)
        .value("PRODUCT", tenzor::distributed::ReduceOp::PRODUCT)
        .value("MIN", tenzor::distributed::ReduceOp::MIN)
        .value("MAX", tenzor::distributed::ReduceOp::MAX)
        .value("AVG", tenzor::distributed::ReduceOp::AVG)
        .export_values();

    distributed.def("init_process_group", &tenzor::distributed::init_process_group,
        "Initialize distributed process group",
        py::arg("backend") = "nccl",
        py::arg("rank") = -1,
        py::arg("world_size") = -1,
        py::arg("master_addr") = "localhost",
        py::arg("master_port") = 29500);

    distributed.def("destroy_process_group", &tenzor::distributed::destroy_process_group,
        "Destroy process group and cleanup resources");

    distributed.def("get_rank", &tenzor::distributed::get_rank,
        "Get current process rank");

    distributed.def("get_world_size", &tenzor::distributed::get_world_size,
        "Get total number of processes");

    distributed.def("is_initialized", &tenzor::distributed::is_initialized,
        "Check if distributed training is initialized");

    distributed.def("barrier", &tenzor::distributed::barrier,
        "Barrier synchronization across all processes");

    distributed.def("all_reduce", &tenzor::distributed::all_reduce,
        "All-reduce operation on tensor",
        py::arg("tensor"), py::arg("op") = tenzor::distributed::ReduceOp::SUM);

    distributed.def("broadcast", &tenzor::distributed::broadcast,
        "Broadcast tensor from source rank",
        py::arg("tensor"), py::arg("src_rank") = 0);

    py::class_<tenzor::distributed::ProcessGroup, std::shared_ptr<tenzor::distributed::ProcessGroup>>(
        distributed, "ProcessGroup")
        .def_property_readonly("rank", &tenzor::distributed::ProcessGroup::rank,
            "Get process rank")
        .def_property_readonly("world_size", &tenzor::distributed::ProcessGroup::world_size,
            "Get world size")
        .def("broadcast", &tenzor::distributed::ProcessGroup::broadcast,
            "Broadcast tensor from source rank",
            py::arg("tensor"), py::arg("src_rank") = 0)
        .def("all_reduce", &tenzor::distributed::ProcessGroup::all_reduce,
            "All-reduce operation",
            py::arg("tensor"), py::arg("op") = tenzor::distributed::ReduceOp::SUM)
        .def("barrier", &tenzor::distributed::ProcessGroup::barrier,
            "Barrier synchronization");

    py::class_<tenzor::distributed::DistributedDataParallel>(distributed, "DistributedDataParallel")
        .def(py::init<tenzor::nn::Module&, tenzor::distributed::ProcessGroup&, size_t>(),
            "Construct DDP wrapper",
            py::arg("module"), py::arg("process_group"),
            py::arg("bucket_size_bytes") = tenzor::distributed::DistributedDataParallel::DEFAULT_BUCKET_SIZE)
        .def("forward", &tenzor::distributed::DistributedDataParallel::forward,
            "Forward pass through wrapped module",
            py::arg("input"),
            py::call_guard<py::gil_scoped_release>())
        .def("synchronize_gradients", &tenzor::distributed::DistributedDataParallel::synchronize_gradients,
            "Synchronize gradients across all processes")
        .def("sync_comm", &tenzor::distributed::DistributedDataParallel::sync_comm,
            "Wait for pending async all-reduce operations")
        .def("auto_sync_gradients", &tenzor::distributed::DistributedDataParallel::auto_sync_gradients,
            "Enable or disable automatic gradient synchronization",
            py::arg("enabled"))
        .def("reset_buckets", &tenzor::distributed::DistributedDataParallel::reset_buckets,
            "Reset bucket ready states for next iteration");

    // =========================================================================
    // FSDP (Fully Sharded Data Parallel)
    // =========================================================================
    py::enum_<tenzor::distributed::ShardingStrategy>(distributed, "ShardingStrategy")
        .value("FULL_SHARD", tenzor::distributed::ShardingStrategy::FULL_SHARD)
        .value("SHARD_GRAD_OP", tenzor::distributed::ShardingStrategy::SHARD_GRAD_OP)
        .value("NO_SHARD", tenzor::distributed::ShardingStrategy::NO_SHARD);

    py::class_<tenzor::distributed::FSDPConfig>(distributed, "FSDPConfig")
        .def(py::init<>())
        .def_readwrite("strategy", &tenzor::distributed::FSDPConfig::strategy)
        .def_readwrite("cpu_offload", &tenzor::distributed::FSDPConfig::cpu_offload)
        .def_readwrite("auto_wrap_min_params", &tenzor::distributed::FSDPConfig::auto_wrap_min_params)
        .def_readwrite("mixed_precision", &tenzor::distributed::FSDPConfig::mixed_precision)
        .def_readwrite("forward_prefetch", &tenzor::distributed::FSDPConfig::forward_prefetch)
        .def_readwrite("backward_prefetch", &tenzor::distributed::FSDPConfig::backward_prefetch);

    py::class_<tenzor::distributed::FullyShardedDataParallel>(distributed, "FullyShardedDataParallel")
        .def(py::init<tenzor::nn::Module&, tenzor::distributed::ProcessGroup&,
                       const tenzor::distributed::FSDPConfig&>(),
             py::arg("module"), py::arg("process_group"),
             py::arg("config") = tenzor::distributed::FSDPConfig{})
        .def("forward", &tenzor::distributed::FullyShardedDataParallel::forward,
             py::arg("input"))
        .def("finalize_backward", &tenzor::distributed::FullyShardedDataParallel::finalize_backward)
        .def("summon_full_params", &tenzor::distributed::FullyShardedDataParallel::summon_full_params)
        .def("release_full_params", &tenzor::distributed::FullyShardedDataParallel::release_full_params)
        .def("total_params", &tenzor::distributed::FullyShardedDataParallel::total_params)
        .def("sharded_param_bytes", &tenzor::distributed::FullyShardedDataParallel::sharded_param_bytes);

    // =========================================================================
    // Gradient Compression
    // =========================================================================
    py::class_<tenzor::distributed::CompressedGradient>(distributed, "CompressedGradient")
        .def_readonly("data", &tenzor::distributed::CompressedGradient::data)
        .def_readonly("original_shape", &tenzor::distributed::CompressedGradient::original_shape)
        .def_readonly("compression_ratio", &tenzor::distributed::CompressedGradient::compression_ratio);

    py::class_<tenzor::distributed::FP16Compressor>(distributed, "FP16Compressor")
        .def(py::init<>())
        .def("compress", &tenzor::distributed::FP16Compressor::compress, py::arg("gradient"))
        .def("decompress", &tenzor::distributed::FP16Compressor::decompress, py::arg("compressed"))
        .def("name", &tenzor::distributed::FP16Compressor::name)
        .def("reset", &tenzor::distributed::FP16Compressor::reset);

    py::class_<tenzor::distributed::TopKCompressor>(distributed, "TopKCompressor")
        .def(py::init<double>(), py::arg("ratio") = 0.01)
        .def("compress", &tenzor::distributed::TopKCompressor::compress, py::arg("gradient"))
        .def("decompress", &tenzor::distributed::TopKCompressor::decompress, py::arg("compressed"))
        .def("name", &tenzor::distributed::TopKCompressor::name)
        .def("reset", &tenzor::distributed::TopKCompressor::reset);

    // =========================================================================
    // RPC submodule
    // =========================================================================
    auto rpc = distributed.def_submodule("rpc", "Remote Procedure Call framework");

    // Register RpcAgentConfig BEFORE init_rpc (pybind11 requires types to be
    // registered before they are used as default arguments)
    py::class_<tenzor::distributed::rpc::RpcAgentConfig>(rpc, "RpcAgentConfig",
        "Configuration for the RPC agent")
        .def(py::init<>())
        .def_readwrite("num_io_threads",
            &tenzor::distributed::rpc::RpcAgentConfig::num_io_threads,
            "I/O threads for socket operations (default: 2)")
        .def_readwrite("num_worker_threads",
            &tenzor::distributed::rpc::RpcAgentConfig::num_worker_threads,
            "Worker threads for RPC execution (default: 4)")
        .def_readwrite("timeout_ms",
            &tenzor::distributed::rpc::RpcAgentConfig::timeout_ms,
            "RPC timeout in milliseconds (default: 60000)")
        .def_readwrite("heartbeat_interval_ms",
            &tenzor::distributed::rpc::RpcAgentConfig::heartbeat_interval_ms,
            "Heartbeat interval in milliseconds (default: 5000)")
        .def_readwrite("enable_heartbeat",
            &tenzor::distributed::rpc::RpcAgentConfig::enable_heartbeat,
            "Enable health monitoring (default: true)");

    rpc.def("init_rpc", &tenzor::distributed::rpc::init_rpc,
        py::arg("name"), py::arg("rank"), py::arg("world_size"),
        py::arg("config") = tenzor::distributed::rpc::RpcAgentConfig{},
        "Initialize the RPC framework");

    rpc.def("shutdown_rpc", &tenzor::distributed::rpc::shutdown_rpc,
        "Shut down the RPC framework");

    rpc.def("rpc_sync", &tenzor::distributed::rpc::rpc_sync,
        py::arg("dst"), py::arg("func_name"), py::arg("args"),
        "Synchronous RPC call to a remote worker");

    rpc.def("rpc_async", [](int32_t dst, const std::string& func_name,
                             const std::vector<tenzor::Tensor>& args) {
        auto future = tenzor::distributed::rpc::rpc_async(dst, func_name, args);
        return future.get();  // Block in Python for simplicity
    },
    py::arg("dst"), py::arg("func_name"), py::arg("args"),
    "Asynchronous RPC call (blocks until result available in Python)");

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
    auto init = nn.def_submodule("init", "Weight initialization utilities");

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

    // Data loading utilities
    auto data_mod = m.def_submodule("data", "Data loading and dataset utilities");

    // Dataset abstract base class
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
        .def("scale", &tenzor::nn::amp::GradScaler::scale,
             py::arg("loss"),
             "Scale loss by current scale factor")
        .def("unscale_", &tenzor::nn::amp::GradScaler::unscale_,
             py::arg("optimizer"),
             "Unscale gradients in optimizer parameters")
        .def("step", &tenzor::nn::amp::GradScaler::step,
             py::arg("optimizer"),
             "Execute optimizer step with overflow detection")
        .def("update", &tenzor::nn::amp::GradScaler::update,
             "Update scale factor based on overflow history")
        .def("get_scale", &tenzor::nn::amp::GradScaler::get_scale,
             "Get current scale factor")
        .def("get_growth_tracker", &tenzor::nn::amp::GradScaler::get_growth_tracker,
             "Get number of consecutive successful iterations")
        .def("found_inf_nan", &tenzor::nn::amp::GradScaler::found_inf_nan,
             "Check if overflow was detected in last step")
        .def("reset", &tenzor::nn::amp::GradScaler::reset,
             "Reset scaler to initial state")
        .def("state_dict", &tenzor::nn::amp::GradScaler::state_dict,
             "Get scaler state for serialization")
        .def("load_state_dict", &tenzor::nn::amp::GradScaler::load_state_dict,
             py::arg("state"),
             "Load scaler state from dictionary");

    // Autocast context manager — uses a wrapper to defer activation to __enter__
    struct PyAutocastContext {
        bool enabled_;
        tenzor::DType dtype_;
        tenzor::Device::Type device_type_;
        std::unique_ptr<tenzor::nn::amp::Autocast> guard_;

        PyAutocastContext(bool enabled, tenzor::DType dtype, tenzor::Device::Type device_type)
            : enabled_(enabled), dtype_(dtype), device_type_(device_type) {}

        void enter() {
            guard_ = std::make_unique<tenzor::nn::amp::Autocast>(enabled_, dtype_, device_type_);
        }

        void exit() {
            guard_.reset();  // Destructor restores previous state
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
    auto linalg_mod = m.def_submodule("linalg", "Linear algebra operations");

    linalg_mod.def("det", &tenzor::linalg::det, "Compute matrix determinant",
                   py::arg("A"),
                   py::call_guard<py::gil_scoped_release>());
    linalg_mod.def("inv", &tenzor::linalg::inv, "Compute matrix inverse",
                   py::arg("A"),
                   py::call_guard<py::gil_scoped_release>());
    linalg_mod.def("solve", &tenzor::linalg::solve, "Solve linear system AX = B",
                   py::arg("A"), py::arg("B"),
                   py::call_guard<py::gil_scoped_release>());
    linalg_mod.def("cholesky", &tenzor::linalg::cholesky, "Cholesky decomposition",
                   py::arg("A"), py::arg("upper") = false,
                   py::call_guard<py::gil_scoped_release>());
    linalg_mod.def("norm", &tenzor::linalg::norm, "Matrix norm",
                   py::arg("A"), py::arg("ord") = "fro",
                   py::call_guard<py::gil_scoped_release>());
    linalg_mod.def("slogdet", [](const tenzor::Tensor& A) {
        auto [sign, logabsdet] = tenzor::linalg::slogdet(A);
        return py::make_tuple(sign, logabsdet);
    }, "Sign and log of absolute determinant", py::arg("A"),
       py::call_guard<py::gil_scoped_release>());
    linalg_mod.def("svd", [](const tenzor::Tensor& A, bool full_matrices) {
        auto [U, S, Vh] = tenzor::linalg::svd(A, full_matrices);
        return py::make_tuple(U, S, Vh);
    }, "Singular Value Decomposition", py::arg("A"), py::arg("full_matrices") = true,
       py::call_guard<py::gil_scoped_release>());
    linalg_mod.def("qr", [](const tenzor::Tensor& A) {
        auto [Q, R] = tenzor::linalg::qr(A);
        return py::make_tuple(Q, R);
    }, "QR decomposition", py::arg("A"),
       py::call_guard<py::gil_scoped_release>());
    linalg_mod.def("eigh", [](const tenzor::Tensor& A) {
        auto [eigenvalues, eigenvectors] = tenzor::linalg::eigh(A);
        return py::make_tuple(eigenvalues, eigenvectors);
    }, "Eigendecomposition of symmetric matrix", py::arg("A"),
       py::call_guard<py::gil_scoped_release>());
    linalg_mod.def("eigvalsh", &tenzor::linalg::eigvalsh,
                   "Eigenvalues of symmetric matrix", py::arg("A"),
                   py::call_guard<py::gil_scoped_release>());

    linalg_mod.def("matrix_power", &tenzor::linalg::matrix_power,
                   "Matrix power via binary exponentiation",
                   py::arg("A"), py::arg("n"),
                   py::call_guard<py::gil_scoped_release>());

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
    // FFT submodule
    // =========================================================================
    auto fft_mod = m.def_submodule("fft", "Fast Fourier Transform operations");

    fft_mod.def("fft", [](const tenzor::Tensor& input, std::optional<int64_t> n,
                           int64_t dim, const std::string& norm) {
        return tenzor::fft::fft(input, n, dim, norm);
    }, "1-D complex-to-complex FFT",
       py::arg("input"), py::arg("n") = py::none(), py::arg("dim") = -1,
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("ifft", [](const tenzor::Tensor& input, std::optional<int64_t> n,
                            int64_t dim, const std::string& norm) {
        return tenzor::fft::ifft(input, n, dim, norm);
    }, "1-D inverse complex-to-complex FFT",
       py::arg("input"), py::arg("n") = py::none(), py::arg("dim") = -1,
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("rfft", [](const tenzor::Tensor& input, std::optional<int64_t> n,
                            int64_t dim, const std::string& norm) {
        return tenzor::fft::rfft(input, n, dim, norm);
    }, "1-D real-to-complex FFT",
       py::arg("input"), py::arg("n") = py::none(), py::arg("dim") = -1,
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("irfft", [](const tenzor::Tensor& input, std::optional<int64_t> n,
                             int64_t dim, const std::string& norm) {
        return tenzor::fft::irfft(input, n, dim, norm);
    }, "1-D complex-to-real inverse FFT",
       py::arg("input"), py::arg("n") = py::none(), py::arg("dim") = -1,
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("fft2", [](const tenzor::Tensor& input,
                            std::optional<std::vector<int64_t>> s,
                            std::vector<int64_t> dim, const std::string& norm) {
        return tenzor::fft::fft2(input, s, dim, norm);
    }, "2-D complex-to-complex FFT",
       py::arg("input"), py::arg("s") = py::none(),
       py::arg("dim") = std::vector<int64_t>{-2, -1},
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("ifft2", [](const tenzor::Tensor& input,
                             std::optional<std::vector<int64_t>> s,
                             std::vector<int64_t> dim, const std::string& norm) {
        return tenzor::fft::ifft2(input, s, dim, norm);
    }, "2-D inverse complex-to-complex FFT",
       py::arg("input"), py::arg("s") = py::none(),
       py::arg("dim") = std::vector<int64_t>{-2, -1},
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("fftn", [](const tenzor::Tensor& input,
                            std::optional<std::vector<int64_t>> s,
                            std::optional<std::vector<int64_t>> dim, const std::string& norm) {
        return tenzor::fft::fftn(input, s, dim, norm);
    }, "N-D complex-to-complex FFT",
       py::arg("input"), py::arg("s") = py::none(), py::arg("dim") = py::none(),
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    fft_mod.def("ifftn", [](const tenzor::Tensor& input,
                             std::optional<std::vector<int64_t>> s,
                             std::optional<std::vector<int64_t>> dim, const std::string& norm) {
        return tenzor::fft::ifftn(input, s, dim, norm);
    }, "N-D inverse complex-to-complex FFT",
       py::arg("input"), py::arg("s") = py::none(), py::arg("dim") = py::none(),
       py::arg("norm") = "backward",
       py::call_guard<py::gil_scoped_release>());

    // =========================================================================
    // JIT Module - Tracing, Compilation, and Graph Optimization
    // =========================================================================
    auto jit = m.def_submodule("jit", "JIT compilation and tracing");

    // OpType enum
    py::enum_<tenzor::jit::OpType>(jit, "OpType")
        .value("Add", tenzor::jit::OpType::Add)
        .value("Sub", tenzor::jit::OpType::Sub)
        .value("Mul", tenzor::jit::OpType::Mul)
        .value("Div", tenzor::jit::OpType::Div)
        .value("MatMul", tenzor::jit::OpType::MatMul)
        .value("ReLU", tenzor::jit::OpType::ReLU)
        .value("Sigmoid", tenzor::jit::OpType::Sigmoid)
        .value("Tanh", tenzor::jit::OpType::Tanh)
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
        .value("Output", tenzor::jit::OpType::Output);

    // Value class
    py::class_<tenzor::jit::Value, std::shared_ptr<tenzor::jit::Value>>(jit, "Value",
        "Represents a tensor value in the IR graph")
        .def_property_readonly("id", &tenzor::jit::Value::id)
        .def_property_readonly("shape", &tenzor::jit::Value::shape)
        .def_property_readonly("dtype", &tenzor::jit::Value::dtype)
        .def_property_readonly("device", &tenzor::jit::Value::device);

    // Node class
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

    // Graph class
    py::class_<tenzor::jit::Graph, std::shared_ptr<tenzor::jit::Graph>>(jit, "Graph",
        "IR graph representing a complete computation")
        .def(py::init<>())
        .def("num_nodes", &tenzor::jit::Graph::num_nodes)
        .def("num_values", &tenzor::jit::Graph::num_values)
        .def("forward", &tenzor::jit::Graph::forward,
             py::arg("inputs"),
             "Execute graph with runtime inputs",
             py::call_guard<py::gil_scoped_release>())
        .def("save", &tenzor::jit::Graph::save,
             py::arg("path"),
             "Save graph to file")
        .def_static("load", &tenzor::jit::Graph::load,
             py::arg("path"),
             "Load graph from file")
        .def("to_string", &tenzor::jit::Graph::to_string,
             "Get string representation of graph")
        .def("topological_sort", &tenzor::jit::Graph::topological_sort)
        .def("infer_types", &tenzor::jit::Graph::infer_types)
        .def("__repr__", &tenzor::jit::Graph::to_string);

    // Tracer class
    py::class_<tenzor::jit::Tracer>(jit, "Tracer",
        "Tracing context for recording operations")
        .def(py::init<>())
        .def("start_trace", &tenzor::jit::Tracer::start_trace,
             "Start recording operations")
        .def("end_trace", &tenzor::jit::Tracer::end_trace,
             py::arg("inputs"), py::arg("outputs"),
             "Stop recording and build IR graph")
        .def("is_tracing", &tenzor::jit::Tracer::is_tracing,
             "Check if tracing is active")
        .def("clear", &tenzor::jit::Tracer::clear,
             "Clear all recorded operations")
        .def_static("get_instance", &tenzor::jit::Tracer::get_instance,
             py::return_value_policy::reference,
             "Get thread-local tracer instance");

    // TracingGuard RAII class
    py::class_<tenzor::jit::TracingGuard>(jit, "TracingGuard",
        "RAII guard for tracing scope")
        .def(py::init<>())
        .def("get_graph", &tenzor::jit::TracingGuard::get_graph,
             py::arg("inputs"), py::arg("outputs"),
             "Get traced graph");

    // Compiler class
    py::class_<tenzor::jit::Compiler>(jit, "Compiler",
        "Graph optimization compiler")
        .def(py::init<bool>(),
             py::arg("enable_default_passes") = true,
             "Create compiler with optional default passes")
        .def("optimize", &tenzor::jit::Compiler::optimize,
             py::arg("graph"), py::arg("max_iterations") = 10,
             "Optimize graph with all passes")
        .def("set_verbose", &tenzor::jit::Compiler::set_verbose,
             py::arg("enable"),
             "Enable verbose logging")
        .def("clear_stats", &tenzor::jit::Compiler::clear_stats);

    // JIT Free functions
    jit.def("trace", py::overload_cast<std::shared_ptr<tenzor::nn::Module>,
            const tenzor::Variable&>(&tenzor::jit::trace),
            py::arg("module"), py::arg("dummy_input"),
            "Trace a module's forward pass");

    jit.def("optimize_graph", &tenzor::jit::optimize_graph,
            py::arg("graph"),
            "Apply standard optimizations to graph");

    jit.def("save_graph", &tenzor::jit::save_graph,
            py::arg("graph"), py::arg("path"),
            "Save graph to file");

    jit.def("load_graph", &tenzor::jit::load_graph,
            py::arg("path"),
            "Load graph from file");

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
            "Verify graph integrity, returns list of errors");

    // =========================================================================
    // Compile API (torch.compile equivalent)
    // =========================================================================
    jit.def("compile", [](py::function fn, bool fullgraph, std::string mode) {
        // Wrap Python function in a C++ callable
        auto cpp_fn = [fn](const tenzor::Variable& input) -> tenzor::Variable {
            py::gil_scoped_acquire acquire;
            auto result = fn(input);
            return result.cast<tenzor::Variable>();
        };

        tenzor::jit::CompileConfig config;
        config.fullgraph = fullgraph;
        config.mode = std::move(mode);

        auto compiled = std::make_shared<tenzor::jit::CompiledFunction>(
            std::move(cpp_fn), std::move(config));

        // Return a callable Python object
        return py::cpp_function([compiled](const tenzor::Variable& input) {
            py::gil_scoped_release release;
            return (*compiled)(input);
        });
    },
    py::arg("fn"),
    py::arg("fullgraph") = false,
    py::arg("mode") = "default",
    "Compile a function for automatic graph capture and optimization.\n"
    "First call traces and compiles; subsequent calls use cached compiled graph.\n"
    "Shape mismatches trigger recompilation (up to 8 shapes cached).");

    // Also expose compile at module level
    m.def("compile", [&jit](py::function fn, bool fullgraph, std::string mode) {
        return jit.attr("compile")(fn, fullgraph, mode);
    },
    py::arg("fn"),
    py::arg("fullgraph") = false,
    py::arg("mode") = "default",
    "Compile a function for automatic graph capture (alias for jit.compile).");

    // =========================================================================
    // Vision Operations
    // =========================================================================
    auto vision = m.def_submodule("vision", "Vision operations");

    vision.def("unfold", &tenzor::ops::unfold,
               py::arg("input"),
               py::arg("kernel_size"),
               py::arg("stride") = 1,
               py::arg("padding") = 0,
               py::arg("dilation") = 1,
               "Extract sliding local blocks (im2col)");

    vision.def("fold", &tenzor::ops::fold,
               py::arg("input"),
               py::arg("output_size"),
               py::arg("kernel_size"),
               py::arg("stride") = 1,
               py::arg("padding") = 0,
               py::arg("dilation") = 1,
               "Fold tensor back to spatial dimensions (col2im)");

    vision.def("interpolate", &tenzor::ops::interpolate,
               py::arg("input"),
               py::arg("size"),
               py::arg("mode") = "bilinear",
               py::arg("align_corners") = false,
               "Resize tensor using interpolation");

    vision.def("grid_sample", &tenzor::ops::grid_sample,
               py::arg("input"),
               py::arg("grid"),
               py::arg("mode") = "bilinear",
               py::arg("padding_mode") = "zeros",
               py::arg("align_corners") = false,
               "Sample from input using grid coordinates (spatial transformer)");

    vision.def("affine_grid", &tenzor::ops::affine_grid,
               py::arg("theta"),
               py::arg("size"),
               py::arg("align_corners") = false,
               "Generate 2D affine grid for grid_sample");

    // =========================================================================
    // Detection Operations
    // =========================================================================
    auto detection = m.def_submodule("detection", "Object detection operations");

    py::enum_<tenzor::ops::IoUType>(detection, "IoUType")
        .value("IoU", tenzor::ops::IoUType::IoU)
        .value("GIoU", tenzor::ops::IoUType::GIoU)
        .value("DIoU", tenzor::ops::IoUType::DIoU)
        .value("CIoU", tenzor::ops::IoUType::CIoU);

    detection.def("box_iou", &tenzor::ops::box_iou,
                  py::arg("boxes1"), py::arg("boxes2"),
                  py::arg("iou_type") = tenzor::ops::IoUType::IoU,
                  "Compute IoU between box sets");

    detection.def("nms", &tenzor::ops::nms,
                  py::arg("boxes"), py::arg("scores"),
                  py::arg("iou_threshold") = 0.5,
                  "Non-Maximum Suppression");

    detection.def("batched_nms", &tenzor::ops::batched_nms,
                  py::arg("boxes"), py::arg("scores"),
                  py::arg("iou_threshold") = 0.5,
                  py::arg("score_threshold") = 0.05,
                  py::arg("max_output_boxes") = 100,
                  "Batched NMS for multiple classes");

    detection.def("encode_boxes", &tenzor::ops::encode_boxes,
                  py::arg("boxes"), py::arg("anchors"),
                  py::arg("weights") = std::vector<double>{1.0, 1.0, 1.0, 1.0},
                  "Encode boxes relative to anchors");

    detection.def("decode_boxes", &tenzor::ops::decode_boxes,
                  py::arg("deltas"), py::arg("anchors"),
                  py::arg("weights") = std::vector<double>{1.0, 1.0, 1.0, 1.0},
                  "Decode boxes from deltas and anchors");

    detection.def("clip_boxes_to_image", &tenzor::ops::clip_boxes_to_image,
                  py::arg("boxes"), py::arg("height"), py::arg("width"),
                  "Clip boxes to image boundaries");

    detection.def("remove_small_boxes", &tenzor::ops::remove_small_boxes,
                  py::arg("boxes"), py::arg("scores"), py::arg("min_size"),
                  "Remove boxes smaller than min_size");

    // =========================================================================
    // Async Operations
    // =========================================================================
    auto async_ops = m.def_submodule("async_ops", "Asynchronous tensor operations");

    async_ops.def("async_matmul", &tenzor::async_matmul,
                  py::arg("a"), py::arg("b"),
                  "Asynchronous matrix multiplication");

    async_ops.def("async_add", &tenzor::async_add,
                  py::arg("a"), py::arg("b"),
                  "Asynchronous element-wise addition");

    async_ops.def("async_mul", &tenzor::async_mul,
                  py::arg("a"), py::arg("b"),
                  "Asynchronous element-wise multiplication");

    async_ops.def("async_sub", &tenzor::async_sub,
                  py::arg("a"), py::arg("b"),
                  "Asynchronous element-wise subtraction");

    async_ops.def("async_div", &tenzor::async_div,
                  py::arg("a"), py::arg("b"),
                  "Asynchronous element-wise division");

    async_ops.def("async_relu", &tenzor::async_relu,
                  py::arg("input"),
                  "Asynchronous ReLU activation");

    async_ops.def("async_sigmoid", &tenzor::async_sigmoid,
                  py::arg("input"),
                  "Asynchronous sigmoid activation");

    async_ops.def("async_tanh", &tenzor::async_tanh,
                  py::arg("input"),
                  "Asynchronous tanh activation");

    async_ops.def("async_softmax", &tenzor::async_softmax,
                  py::arg("input"), py::arg("dim") = -1,
                  "Asynchronous softmax");

    // =========================================================================
    // Fused Operations
    // =========================================================================
    auto fused = m.def_submodule("fused", "Fused kernel operations");

    fused.def("fused_linear_relu", &tenzor::ops::fused_linear_relu,
              py::arg("input"), py::arg("weight"), py::arg("bias") = nullptr,
              "Fused linear + ReLU (1.5-2x faster)");

    fused.def("fused_conv2d_relu", &tenzor::ops::fused_conv2d_relu,
              py::arg("input"), py::arg("weight"), py::arg("bias") = nullptr,
              py::arg("stride") = 1, py::arg("padding") = 0,
              "Fused conv2d + ReLU (1.8-2.5x faster)");

    fused.def("fused_batchnorm_relu", &tenzor::ops::fused_batchnorm_relu,
              py::arg("input"), py::arg("running_mean"), py::arg("running_var"),
              py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5f,
              "Fused batchnorm + ReLU (1.6-2.2x faster)");

    fused.def("fused_softmax_cross_entropy", &tenzor::ops::fused_softmax_cross_entropy,
              py::arg("logits"), py::arg("targets"), py::arg("reduction") = "mean",
              "Fused softmax + cross-entropy (2-3x faster, 50% less memory)");

    fused.def("fused_add_relu", &tenzor::ops::fused_add_relu,
              py::arg("a"), py::arg("b"),
              "Fused add + ReLU for residual connections");

    fused.def("fused_gelu", &tenzor::ops::fused_gelu,
              py::arg("input"),
              "Fused GELU activation (1.5x faster)");

    fused.def("fused_layer_norm", &tenzor::ops::fused_layer_norm,
              py::arg("input"), py::arg("normalized_shape"),
              py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5f,
              "Fused layer normalization (1.4-2x faster)");

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
             py::arg("model_name"), py::arg("input_shape"),
             "Log computation graph structure")
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
    // Adam-atan2 Optimizer
    // =========================================================================
    py::class_<tenzor::optim::AdamAtan2, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::AdamAtan2>>(optim, "AdamAtan2",
        "Adam-atan2 optimizer for HRM training with bounded updates")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double, bool>(),
             py::arg("params"),
             py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8,
             py::arg("weight_decay") = 0.01,
             py::arg("amsgrad") = false)
        .def("step", [](tenzor::optim::AdamAtan2& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            self.step(); return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::AdamAtan2::zero_grad)
        .def("set_lr", &tenzor::optim::AdamAtan2::set_lr)
        .def("get_lr", &tenzor::optim::AdamAtan2::get_lr)
        .def("state_dict", &tenzor::optim::AdamAtan2::state_dict)
        .def("load_state_dict", &tenzor::optim::AdamAtan2::load_state_dict);

    // =========================================================================
    // HRM (Hierarchical Reasoning Model) Layers
    // =========================================================================
    auto hrm = nn.def_submodule("hrm", "Hierarchical Reasoning Model components");

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

    // GatedLinearUnit
    py::class_<tenzor::nn::GatedLinearUnit, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::GatedLinearUnit>>(hrm, "GatedLinearUnit",
        "Gated Linear Unit (GLU/SwiGLU)")
        .def(py::init<int64_t, int64_t, bool, bool>(),
             py::arg("in_features"), py::arg("hidden_features"),
             py::arg("use_silu") = true, py::arg("bias") = false);

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

    // Convenience functions for standard ResNet variants
    models.def("resnet18", [](int64_t num_classes) {
        return std::make_shared<tenzor::models::ResNet>(
            std::vector<int64_t>{2, 2, 2, 2}, num_classes, true);
    }, py::arg("num_classes") = 1000, "Create ResNet-18");

    models.def("resnet34", [](int64_t num_classes) {
        return std::make_shared<tenzor::models::ResNet>(
            std::vector<int64_t>{3, 4, 6, 3}, num_classes, true);
    }, py::arg("num_classes") = 1000, "Create ResNet-34");

    models.def("resnet50", [](int64_t num_classes) {
        return std::make_shared<tenzor::models::ResNet>(
            std::vector<int64_t>{3, 4, 6, 3}, num_classes, false);
    }, py::arg("num_classes") = 1000, "Create ResNet-50");

    models.def("resnet101", [](int64_t num_classes) {
        return std::make_shared<tenzor::models::ResNet>(
            std::vector<int64_t>{3, 4, 23, 3}, num_classes, false);
    }, py::arg("num_classes") = 1000, "Create ResNet-101");

    models.def("resnet152", [](int64_t num_classes) {
        return std::make_shared<tenzor::models::ResNet>(
            std::vector<int64_t>{3, 8, 36, 3}, num_classes, false);
    }, py::arg("num_classes") = 1000, "Create ResNet-152");

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

    models.def("vgg11", [](int64_t num_classes, bool bn) {
        return tenzor::models::vgg11(num_classes, bn);
    }, py::arg("num_classes") = 1000, py::arg("batch_norm") = true, "Create VGG-11");
    models.def("vgg13", [](int64_t num_classes, bool bn) {
        return tenzor::models::vgg13(num_classes, bn);
    }, py::arg("num_classes") = 1000, py::arg("batch_norm") = true, "Create VGG-13");
    models.def("vgg16", [](int64_t num_classes, bool bn) {
        return tenzor::models::vgg16(num_classes, bn);
    }, py::arg("num_classes") = 1000, py::arg("batch_norm") = true, "Create VGG-16");
    models.def("vgg19", [](int64_t num_classes, bool bn) {
        return tenzor::models::vgg19(num_classes, bn);
    }, py::arg("num_classes") = 1000, py::arg("batch_norm") = true, "Create VGG-19");

    // AlexNet
    py::class_<tenzor::models::AlexNet, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::AlexNet>>(models, "AlexNet",
        "AlexNet architecture")
        .def(py::init<int64_t, double>(),
             py::arg("num_classes") = 1000, py::arg("dropout") = 0.5);
    models.def("alexnet", [](int64_t num_classes) {
        return tenzor::models::alexnet(num_classes);
    }, py::arg("num_classes") = 1000, "Create AlexNet");

    // MobileNet
    py::class_<tenzor::models::MobileNetV2, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::MobileNetV2>>(models, "MobileNetV2",
        "MobileNetV2 architecture")
        .def(py::init<int64_t, double, double>(),
             py::arg("num_classes") = 1000, py::arg("width_mult") = 1.0,
             py::arg("dropout") = 0.2);
    models.def("mobilenet_v2", [](int64_t num_classes) {
        return tenzor::models::mobilenet_v2(num_classes);
    }, py::arg("num_classes") = 1000, "Create MobileNetV2");

    py::class_<tenzor::models::MobileNetV3, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::MobileNetV3>>(models, "MobileNetV3",
        "MobileNetV3 architecture")
        .def(py::init<int64_t, std::string, double, double>(),
             py::arg("num_classes") = 1000, py::arg("mode") = "large",
             py::arg("width_mult") = 1.0, py::arg("dropout") = 0.2);
    models.def("mobilenet_v3_large", [](int64_t num_classes) {
        return tenzor::models::mobilenet_v3_large(num_classes);
    }, py::arg("num_classes") = 1000, "Create MobileNetV3-Large");
    models.def("mobilenet_v3_small", [](int64_t num_classes) {
        return tenzor::models::mobilenet_v3_small(num_classes);
    }, py::arg("num_classes") = 1000, "Create MobileNetV3-Small");

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

    models.def("efficientnet_b0", [](int64_t num_classes) {
        return tenzor::models::efficientnet_b0(num_classes);
    }, py::arg("num_classes") = 1000, "Create EfficientNet-B0");
    models.def("efficientnet_b1", [](int64_t num_classes) {
        return tenzor::models::efficientnet_b1(num_classes);
    }, py::arg("num_classes") = 1000, "Create EfficientNet-B1");
    models.def("efficientnet_b2", [](int64_t num_classes) {
        return tenzor::models::efficientnet_b2(num_classes);
    }, py::arg("num_classes") = 1000, "Create EfficientNet-B2");
    models.def("efficientnet_b3", [](int64_t num_classes) {
        return tenzor::models::efficientnet_b3(num_classes);
    }, py::arg("num_classes") = 1000, "Create EfficientNet-B3");
    models.def("efficientnet_b4", [](int64_t num_classes) {
        return tenzor::models::efficientnet_b4(num_classes);
    }, py::arg("num_classes") = 1000, "Create EfficientNet-B4");
    models.def("efficientnet_b5", [](int64_t num_classes) {
        return tenzor::models::efficientnet_b5(num_classes);
    }, py::arg("num_classes") = 1000, "Create EfficientNet-B5");
    models.def("efficientnet_b6", [](int64_t num_classes) {
        return tenzor::models::efficientnet_b6(num_classes);
    }, py::arg("num_classes") = 1000, "Create EfficientNet-B6");
    models.def("efficientnet_b7", [](int64_t num_classes) {
        return tenzor::models::efficientnet_b7(num_classes);
    }, py::arg("num_classes") = 1000, "Create EfficientNet-B7");

    // GoogLeNet
    py::class_<tenzor::models::GoogLeNet, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::GoogLeNet>>(models, "GoogLeNet",
        "GoogLeNet/Inception architecture")
        .def(py::init<int64_t, bool, double, bool>(),
             py::arg("num_classes") = 1000, py::arg("aux_logits") = true,
             py::arg("dropout") = 0.4, py::arg("init_weights") = true);
    models.def("googlenet", [](int64_t num_classes) {
        return tenzor::models::googlenet(num_classes);
    }, py::arg("num_classes") = 1000, "Create GoogLeNet");

    // ConvNeXt
    py::class_<tenzor::models::ConvNeXt, tenzor::nn::Module,
               std::shared_ptr<tenzor::models::ConvNeXt>>(models, "ConvNeXt",
        "ConvNeXt architecture")
        .def(py::init<int64_t, int64_t, std::vector<int64_t>, std::vector<int64_t>, double, double>(),
             py::arg("in_channels") = 3, py::arg("num_classes") = 1000,
             py::arg("depths") = std::vector<int64_t>{3,3,9,3},
             py::arg("dims") = std::vector<int64_t>{96,192,384,768},
             py::arg("drop_path_rate") = 0.0, py::arg("layer_scale_init_value") = 1e-6);
    models.def("convnext_tiny", [](int64_t num_classes) {
        return tenzor::models::convnext_tiny(num_classes);
    }, py::arg("num_classes") = 1000, "Create ConvNeXt-Tiny");
    models.def("convnext_small", [](int64_t num_classes) {
        return tenzor::models::convnext_small(num_classes);
    }, py::arg("num_classes") = 1000, "Create ConvNeXt-Small");
    models.def("convnext_base", [](int64_t num_classes) {
        return tenzor::models::convnext_base(num_classes);
    }, py::arg("num_classes") = 1000, "Create ConvNeXt-Base");
    models.def("convnext_large", [](int64_t num_classes) {
        return tenzor::models::convnext_large(num_classes);
    }, py::arg("num_classes") = 1000, "Create ConvNeXt-Large");

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
    models.def("swin_tiny", [](int64_t num_classes) {
        return tenzor::models::swin_tiny(num_classes);
    }, py::arg("num_classes") = 1000, "Create Swin-Tiny");
    models.def("swin_small", [](int64_t num_classes) {
        return tenzor::models::swin_small(num_classes);
    }, py::arg("num_classes") = 1000, "Create Swin-Small");
    models.def("swin_base", [](int64_t num_classes) {
        return tenzor::models::swin_base(num_classes);
    }, py::arg("num_classes") = 1000, "Create Swin-Base");

    // BERT
    py::class_<tenzor::models::BertConfig>(models, "BertConfig", "BERT configuration")
        .def(py::init<>())
        .def_readwrite("hidden_size", &tenzor::models::BertConfig::hidden_size)
        .def_readwrite("num_hidden_layers", &tenzor::models::BertConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &tenzor::models::BertConfig::num_attention_heads)
        .def_readwrite("vocab_size", &tenzor::models::BertConfig::vocab_size)
        .def_readwrite("max_position_embeddings", &tenzor::models::BertConfig::max_position_embeddings)
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

    // =============================================================================
    // Quantization Enums and Classes
    // =============================================================================

    auto quant = m.def_submodule("quantization", "Neural network quantization");

    py::enum_<QuantizationScheme>(quant, "QuantizationScheme")
        .value("PerTensorSymmetric", QuantizationScheme::PerTensorSymmetric)
        .value("PerTensorAsymmetric", QuantizationScheme::PerTensorAsymmetric)
        .value("PerChannelSymmetric", QuantizationScheme::PerChannelSymmetric)
        .value("PerChannelAsymmetric", QuantizationScheme::PerChannelAsymmetric);

    py::enum_<QuantDType>(quant, "QuantDType")
        .value("INT8", QuantDType::INT8, "Signed 8-bit integer [-128, 127]")
        .value("UINT8", QuantDType::UINT8, "Unsigned 8-bit integer [0, 255]")
        .value("INT4", QuantDType::INT4, "Signed 4-bit integer [-8, 7], packed 2 per byte")
        .value("UINT4", QuantDType::UINT4, "Unsigned 4-bit integer [0, 15], packed 2 per byte");

    py::class_<QuantizationParams>(quant, "QuantizationParams")
        .def(py::init<Tensor, Tensor, QuantDType, QuantizationScheme, int64_t>(),
             py::arg("scale"), py::arg("zero_point"), py::arg("dtype"),
             py::arg("scheme"), py::arg("axis") = -1)
        .def_readonly("scale", &QuantizationParams::scale)
        .def_readonly("zero_point", &QuantizationParams::zero_point)
        .def_readonly("dtype", &QuantizationParams::dtype)
        .def_readonly("scheme", &QuantizationParams::scheme)
        .def_readonly("axis", &QuantizationParams::axis);

    py::class_<QuantizedTensor>(quant, "QuantizedTensor")
        .def("dequantize", &QuantizedTensor::dequantize,
             "Dequantize tensor back to floating point")
        .def("data", &QuantizedTensor::data, "Get quantized integer data")
        .def("params", &QuantizedTensor::params, "Get quantization parameters")
        .def("shape", &QuantizedTensor::shape)
        .def("device", &QuantizedTensor::device);

    // =============================================================================
    // Quantization Functions
    // =============================================================================

    quant.def("compute_quantization_params", &compute_quantization_params,
        py::arg("min"), py::arg("max"), py::arg("dtype"), py::arg("scheme"),
        "Compute quantization parameters from min/max values");

    quant.def("quantize_tensor", &quantize_tensor,
        py::arg("input"), py::arg("params"),
        "Quantize tensor using specified parameters");

    quant.def("quantize_per_tensor_symmetric", &quantize_per_tensor_symmetric,
        py::arg("input"), py::arg("dtype") = QuantDType::INT8,
        R"doc(
        Symmetric per-tensor quantization (zero-point = 0).

        Args:
            input: Input tensor to quantize
            dtype: Target quantized data type

        Returns:
            QuantizedTensor with INT8/UINT8 data

        Example:
            >>> x = Tensor([...])
            >>> q = quantize_per_tensor_symmetric(x, QuantDType.INT8)
            >>> dequant = q.dequantize()
        )doc");

    quant.def("quantize_per_tensor_asymmetric", &quantize_per_tensor_asymmetric,
        py::arg("input"), py::arg("dtype") = QuantDType::INT8,
        "Asymmetric per-tensor quantization (learnable zero-point)");

    quant.def("quantize_per_channel_symmetric", &quantize_per_channel_symmetric,
        py::arg("input"), py::arg("axis"), py::arg("dtype") = QuantDType::INT8,
        R"doc(
        Symmetric per-channel quantization.

        Better accuracy than per-tensor for convolutional layers.

        Args:
            input: Input tensor
            axis: Axis for per-channel quantization (usually 0 for weights)
            dtype: Target dtype

        Returns:
            QuantizedTensor

        Example:
            >>> weight = conv.weight  # [out_channels, in_channels, H, W]
            >>> q = quantize_per_channel_symmetric(weight, axis=0)
        )doc");

    quant.def("quantize_per_channel_asymmetric", &quantize_per_channel_asymmetric,
        py::arg("input"), py::arg("axis"), py::arg("dtype") = QuantDType::INT8,
        "Asymmetric per-channel quantization");

    quant.def("dequantize_tensor", &dequantize_tensor,
        py::arg("quantized"),
        "Dequantize tensor back to floating point");

    quant.def("compute_quantization_error", &compute_quantization_error,
        py::arg("original"), py::arg("quantized"),
        "Compute quantization error (MSE) between original and quantized tensors");

    quant.def("calibrate_quantization_params", &calibrate_quantization_params,
        py::arg("activations"), py::arg("scheme"),
        py::arg("dtype") = QuantDType::INT8, py::arg("axis") = -1,
        "Calibrate quantization parameters from activation statistics");

    // =============================================================================
    // Observers
    // =============================================================================

    py::class_<Observer, std::shared_ptr<Observer>>(quant, "Observer")
        .def("observe", &Observer::observe, "Observe tensor statistics")
        .def("calculate_qparams", &Observer::calculate_qparams,
             py::arg("dtype"), py::arg("scheme"))
        .def("reset", &Observer::reset);

    py::class_<MinMaxObserver, Observer, std::shared_ptr<MinMaxObserver>>(quant, "MinMaxObserver")
        .def(py::init<>());

    py::class_<MovingAverageMinMaxObserver, Observer,
                std::shared_ptr<MovingAverageMinMaxObserver>>(quant, "MovingAverageMinMaxObserver")
        .def(py::init<float>(), py::arg("averaging_constant") = 0.01f);

    py::class_<HistogramObserver, Observer, std::shared_ptr<HistogramObserver>>(quant, "HistogramObserver")
        .def(py::init<int>(), py::arg("bins") = 2048);

    // PerChannelHistogramObserver instead of PerChannelMinMaxObserver
    py::class_<PerChannelHistogramObserver, Observer,
                std::shared_ptr<PerChannelHistogramObserver>>(quant, "PerChannelHistogramObserver")
        .def(py::init<int64_t, int>(), py::arg("axis") = 0, py::arg("bins") = 2048);

    quant.def("make_observer", &make_observer,
        py::arg("scheme"), py::arg("use_histogram") = false, py::arg("axis") = 0,
        "Create observer instance");

    // =============================================================================
    // Fake Quantization (for QAT)
    // =============================================================================

    py::class_<FakeQuantize, Module, std::shared_ptr<FakeQuantize>>(quant, "FakeQuantize")
        .def(py::init<QuantDType, QuantizationScheme, bool, bool, int64_t>(),
             py::arg("dtype") = QuantDType::INT8,
             py::arg("scheme") = QuantizationScheme::PerTensorSymmetric,
             py::arg("learnable") = false,
             py::arg("observer_enabled") = true,
             py::arg("axis") = -1,
             "Fake quantization module for QAT")
        .def("enable_observer", &FakeQuantize::enable_observer, py::arg("enabled") = true)
        .def("disable_observer", &FakeQuantize::disable_observer)
        .def("enable_fake_quant", &FakeQuantize::enable_fake_quant, py::arg("enabled") = true)
        .def("disable_fake_quant", &FakeQuantize::disable_fake_quant)
        .def("set_qparams", &FakeQuantize::set_qparams, py::arg("params"));

    py::class_<LearnableFakeQuantize, FakeQuantize,
                std::shared_ptr<LearnableFakeQuantize>>(quant, "LearnableFakeQuantize")
        .def(py::init<QuantDType, QuantizationScheme, int64_t>(),
             py::arg("dtype") = QuantDType::INT8,
             py::arg("scheme") = QuantizationScheme::PerTensorSymmetric,
             py::arg("axis") = -1);

    // =============================================================================
    // QConfig
    // =============================================================================

    // QConfig uses factory functions internally, expose DefaultQConfigs instead
    py::class_<QConfig>(quant, "QConfig");

    // Expose DefaultQConfigs static methods
    py::class_<DefaultQConfigs>(quant, "DefaultQConfigs")
        .def_static("default_qconfig", &DefaultQConfigs::default_qconfig, "Default quantization config")
        .def_static("high_accuracy_qconfig", &DefaultQConfigs::high_accuracy_qconfig, "High accuracy QConfig")
        .def_static("fast_qconfig", &DefaultQConfigs::fast_qconfig, "Fast calibration QConfig")
        .def_static("qat_qconfig", &DefaultQConfigs::qat_qconfig, "QAT QConfig")
        .def_static("uint8_activation_qconfig", &DefaultQConfigs::uint8_activation_qconfig, "UINT8 activation config");

    py::class_<QConfigMapping>(quant, "QConfigMapping")
        .def(py::init<>())
        .def("set_global", &QConfigMapping::set_global)
        .def("set_layer_qconfig", &QConfigMapping::set_layer_qconfig)
        .def("set_type_qconfig", &QConfigMapping::set_type_qconfig)
        .def("get_qconfig", &QConfigMapping::get_qconfig);

    // =============================================================================
    // Quantized Layers
    // =============================================================================

    py::class_<QuantizedLinear, Module, std::shared_ptr<QuantizedLinear>>(quant, "QuantizedLinear")
        .def(py::init<int64_t, int64_t, QuantizationParams, float>(),
             py::arg("in_features"), py::arg("out_features"),
             py::arg("weight_qparams"), py::arg("bias_scale") = 1.0f,
             "INT8 quantized linear layer")
        .def("forward_quantized", &QuantizedLinear::forward_quantized,
             py::arg("input"), "Forward pass with quantized input")
        .def("set_weight", &QuantizedLinear::set_weight, py::arg("weights"))
        .def("set_bias", &QuantizedLinear::set_bias, py::arg("bias"))
        .def_static("from_float", &QuantizedLinear::from_float,
             py::arg("fp_linear"), py::arg("qconfig"),
             "Create quantized linear from floating-point layer");

    py::class_<QuantizedConv2d, Module, std::shared_ptr<QuantizedConv2d>>(quant, "QuantizedConv2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                       QuantizationParams, float>(),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("kernel_size"), py::arg("stride") = 1,
             py::arg("padding") = 0, py::arg("dilation") = 1,
             py::arg("groups") = 1, py::arg("weight_qparams"),
             py::arg("bias_scale") = 1.0f,
             "INT8 quantized 2D convolution layer")
        .def("forward_quantized", &QuantizedConv2d::forward_quantized,
             py::arg("input"), "Forward pass with quantized input")
        .def("set_weight", &QuantizedConv2d::set_weight, py::arg("weights"))
        .def("set_bias", &QuantizedConv2d::set_bias, py::arg("bias"))
        .def_static("from_float", &QuantizedConv2d::from_float,
             py::arg("fp_conv"), py::arg("qconfig"),
             "Create quantized conv2d from floating-point layer");

    py::class_<QuantizedConv1d, Module, std::shared_ptr<QuantizedConv1d>>(quant, "QuantizedConv1d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                       QuantizationParams, float>(),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("kernel_size"), py::arg("stride") = 1,
             py::arg("padding") = 0, py::arg("dilation") = 1,
             py::arg("groups") = 1, py::arg("weight_qparams"),
             py::arg("bias_scale") = 1.0f,
             "INT8 quantized 1D convolution layer")
        .def("forward_quantized", &QuantizedConv1d::forward_quantized,
             py::arg("input"), "Forward pass with quantized input")
        .def("set_weight", &QuantizedConv1d::set_weight, py::arg("weights"))
        .def("set_bias", &QuantizedConv1d::set_bias, py::arg("bias"))
        .def_static("from_float", &QuantizedConv1d::from_float,
             py::arg("fp_conv"), py::arg("qconfig"),
             "Create quantized conv1d from floating-point layer");

    py::class_<QuantizedConvTranspose2d, Module,
               std::shared_ptr<QuantizedConvTranspose2d>>(quant, "QuantizedConvTranspose2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                       QuantizationParams, float>(),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("kernel_size"), py::arg("stride") = 1,
             py::arg("padding") = 0, py::arg("output_padding") = 0,
             py::arg("groups") = 1, py::arg("weight_qparams"),
             py::arg("bias_scale") = 1.0f,
             "INT8 quantized transposed 2D convolution layer")
        .def("forward_quantized", &QuantizedConvTranspose2d::forward_quantized,
             py::arg("input"), "Forward pass with quantized input")
        .def("set_weight", &QuantizedConvTranspose2d::set_weight, py::arg("weights"))
        .def("set_bias", &QuantizedConvTranspose2d::set_bias, py::arg("bias"))
        .def_static("from_float", &QuantizedConvTranspose2d::from_float,
             py::arg("fp_conv"), py::arg("qconfig"),
             "Create quantized transposed conv2d from floating-point layer");

    py::class_<QuantizedBatchNorm2d, Module,
               std::shared_ptr<QuantizedBatchNorm2d>>(quant, "QuantizedBatchNorm2d")
        .def(py::init<int64_t, Tensor, Tensor>(),
             py::arg("num_features"), py::arg("scale"), py::arg("bias"),
             "Quantized batch normalization with folded parameters")
        .def("forward_quantized", &QuantizedBatchNorm2d::forward_quantized,
             py::arg("input"), "Forward pass with quantized input")
        .def_static("from_float", &QuantizedBatchNorm2d::from_float,
             py::arg("fp_bn"), py::arg("qconfig"),
             "Create quantized batchnorm from floating-point layer");

    py::class_<QuantizedEmbedding, Module,
               std::shared_ptr<QuantizedEmbedding>>(quant, "QuantizedEmbedding")
        .def(py::init<int64_t, int64_t, QuantizationParams, int64_t>(),
             py::arg("num_embeddings"), py::arg("embedding_dim"),
             py::arg("weight_qparams"), py::arg("padding_idx") = -1,
             "INT8 quantized embedding table")
        .def("forward_quantized", &QuantizedEmbedding::forward_quantized,
             py::arg("indices"), "Look up and dequantize embeddings")
        .def("set_weight", &QuantizedEmbedding::set_weight, py::arg("weights"))
        .def_property_readonly("num_embeddings", &QuantizedEmbedding::num_embeddings)
        .def_property_readonly("embedding_dim", &QuantizedEmbedding::embedding_dim)
        .def_static("from_float", &QuantizedEmbedding::from_float,
             py::arg("fp_embedding"), py::arg("qconfig"),
             "Create quantized embedding from floating-point layer");

    py::class_<QuantizedLSTMCell, Module,
               std::shared_ptr<QuantizedLSTMCell>>(quant, "QuantizedLSTMCell")
        .def(py::init<int64_t, int64_t, bool, QuantizationParams>(),
             py::arg("input_size"), py::arg("hidden_size"),
             py::arg("bias") = true,
             py::arg("weight_qparams") = QuantizationParams(
                 Tensor(), Tensor(), QuantDType::INT8, QuantizationScheme::PerTensorSymmetric),
             "INT8 quantized LSTM cell")
        .def("forward_cell", &QuantizedLSTMCell::forward_cell,
             py::arg("input"), py::arg("hx"), py::arg("cx"),
             "Single-step LSTM cell forward with explicit states")
        .def_property_readonly("input_size", &QuantizedLSTMCell::input_size)
        .def_property_readonly("hidden_size", &QuantizedLSTMCell::hidden_size)
        .def_static("from_float", &QuantizedLSTMCell::from_float,
             py::arg("fp_lstm_cell"), py::arg("qconfig"),
             "Create quantized LSTM cell from floating-point layer");

    py::class_<QuantizedLSTM, Module,
               std::shared_ptr<QuantizedLSTM>>(quant, "QuantizedLSTM")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool, bool, QuantizationParams>(),
             py::arg("input_size"), py::arg("hidden_size"),
             py::arg("num_layers") = 1, py::arg("bias") = true,
             py::arg("batch_first") = true, py::arg("bidirectional") = false,
             py::arg("weight_qparams") = QuantizationParams(
                 Tensor(), Tensor(), QuantDType::INT8, QuantizationScheme::PerTensorSymmetric),
             "INT8 quantized LSTM")
        .def("forward_with_state", &QuantizedLSTM::forward_with_state,
             py::arg("input"), py::arg("h0"), py::arg("c0"),
             "Forward with explicit initial states")
        .def_static("from_float", &QuantizedLSTM::from_float,
             py::arg("fp_lstm"), py::arg("qconfig"),
             "Create quantized LSTM from floating-point layer");

    py::class_<QuantizedGRU, Module,
               std::shared_ptr<QuantizedGRU>>(quant, "QuantizedGRU")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool, bool, QuantizationParams>(),
             py::arg("input_size"), py::arg("hidden_size"),
             py::arg("num_layers") = 1, py::arg("bias") = true,
             py::arg("batch_first") = true, py::arg("bidirectional") = false,
             py::arg("weight_qparams") = QuantizationParams(
                 Tensor(), Tensor(), QuantDType::INT8, QuantizationScheme::PerTensorSymmetric),
             "INT8 quantized GRU")
        .def("forward_with_state", &QuantizedGRU::forward_with_state,
             py::arg("input"), py::arg("h0"),
             "Forward with explicit initial hidden state")
        .def_static("from_float", &QuantizedGRU::from_float,
             py::arg("fp_gru"), py::arg("qconfig"),
             "Create quantized GRU from floating-point layer");

    // QAT utilities
    py::class_<QATHelper>(quant, "QATHelper",
        "Quantization-aware training helper for model preparation and conversion")
        .def(py::init<>())
        .def("prepare_qat", &QATHelper::prepare_qat,
             py::arg("model"),
             py::arg("dtype") = QuantDType::INT8,
             py::arg("scheme") = QuantizationScheme::PerTensorSymmetric,
             py::arg("learnable") = false,
             "Prepare model for quantization-aware training")
        .def("enable_observer", &QATHelper::enable_observer,
             "Enable observers for all fake quantize modules")
        .def("disable_observer", &QATHelper::disable_observer,
             "Disable observers and fix quantization parameters")
        .def("freeze_bn_stats", &QATHelper::freeze_bn_stats,
             "Freeze BN statistics and calculate final qparams")
        .def("convert_to_quantized", &QATHelper::convert_to_quantized,
             py::arg("model"),
             "Convert QAT model to quantized inference model");

    quant.def("fake_quantize_with_grad", &fake_quantize_with_grad,
        py::arg("input"), py::arg("scale"), py::arg("zero_point"),
        py::arg("quant_min"), py::arg("quant_max"),
        R"doc(
        Apply fake quantization with autograd support (STE backward).

        Quantizes then dequantizes the input, with straight-through estimator
        for the backward pass. Gradients pass through for values within the
        quantizable range, and are zeroed for out-of-range values.

        Args:
            input: Input variable
            scale: Quantization scale factor
            zero_point: Quantization zero point
            quant_min: Minimum quantized value (e.g. -128 for INT8)
            quant_max: Maximum quantized value (e.g. 127 for INT8)

        Returns:
            Fake-quantized variable with STE gradient
        )doc");

    quant.def("fold_bn", &fold_bn,
        py::arg("model"),
        R"doc(
        Fold BatchNorm2d into preceding Conv2d layers.

        Walks the model for Conv2d -> BatchNorm2d patterns and folds the BN
        parameters into the Conv2d weights using:
            w_folded = gamma / sqrt(var + eps) * w_conv
            b_folded = gamma / sqrt(var + eps) * (b_conv - mean) + beta

        The BN layers are neutralized (scale=1, bias=0, mean=0, var=1) so
        they act as identity. This eliminates BN computation at inference.

        Args:
            model: Sequential model to fold (modified in-place)
        )doc");

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

    // =========================================================================
    // Forward-mode AD and composable transforms
    // =========================================================================

    m.def("jvp", [](py::function py_func,
                     const tenzor::Variable& input,
                     const tenzor::Tensor& tangent) {
        auto func = [&py_func](const tenzor::Variable& x) -> tenzor::Variable {
            py::gil_scoped_acquire gil;
            py::object result = py_func(x);
            return result.cast<tenzor::Variable>();
        };
        auto [output, tangent_out] = tenzor::jvp(func, input, tangent);
        return py::make_tuple(output, tangent_out);
    }, py::arg("func"), py::arg("input"), py::arg("tangent"),
    "Compute Jacobian-Vector Product (forward-mode AD).\n"
    "Returns (output, tangent_output) where tangent_output = J @ tangent.");

    m.def("jacobian", [](py::function py_func,
                          const tenzor::Variable& input) {
        auto func = [&py_func](const tenzor::Variable& x) -> tenzor::Variable {
            py::gil_scoped_acquire gil;
            py::object result = py_func(x);
            return result.cast<tenzor::Variable>();
        };
        return tenzor::jacobian(func, input);
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
}
