#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <iostream>
#include <sstream>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/indexing.hpp>
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/linalg.hpp>
#include <tenzor/ops/transform.hpp>
#include <tenzor/backend/loader.hpp>
#include <tenzor/backend/backend.hpp>
#include <tenzor/nn/optim/scheduler.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include <tenzor/nn/layers/attention.hpp>
#include <tenzor/nn/layers/transformer.hpp>
#include <tenzor/nn/layers/embedding.hpp>
#include <tenzor/nn/layers/lazy_linear.hpp>
#include <tenzor/nn/optim/rmsprop.hpp>
#include <tenzor/nn/optim/adagrad.hpp>
#include <tenzor/nn/optim/adadelta.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <tenzor/nn/callbacks.hpp>
#include <tenzor/nn/training.hpp>
#include <tenzor/nn/checkpoint.hpp>
#include <tenzor/backend/cuda_graph.hpp>
#include <tenzor/nn/mixed_precision.hpp>
#include <tenzor/nn/amp/grad_scaler.hpp>
#include <tenzor/nn/amp/autocast.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/distributed/ddp.hpp>
#include <tenzor/models/hub.hpp>
#include <tenzor/onnx/exporter.hpp>
#include <tenzor/data/dataset.hpp>
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
#include <tenzor/nn/serialize.hpp>
#include <tenzor/models/resnet.hpp>
#include <tenzor/onnx/importer.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/fft.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>
#include <tenzor/sparse/sparse_ops.hpp>
#include <tenzor/autograd/anomaly_mode.hpp>
#include <tenzor/backend/cpu_caching_allocator.hpp>
#include <tenzor/nn/utils/clip_grad.hpp>
#include <tenzor/utils/error.hpp>
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
            return forward_override(input).cast<tenzor::Variable>();
        }

        // Fall back to forward_impl if no forward override found
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

    // Library initialization
    m.def("initialize", &tenzor::initialize,
          "Initialize the Tenzor library (registers backends and operations)");

    m.def("mkl_cleanup", &tenzor::mkl_cleanup,
          "Free MKL internal buffers (call before using PyTorch/NumPy with MKL)");

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

    // Device::Type enum (must be defined before Device class for default args)
    py::enum_<tenzor::Device::Type>(m, "DeviceType")
        .value("CPU", tenzor::Device::Type::CPU)
        .value("CUDA", tenzor::Device::Type::CUDA)
        .value("ROCm", tenzor::Device::Type::ROCm)
        .value("OneAPI", tenzor::Device::Type::OneAPI)
        .value("Vulkan", tenzor::Device::Type::Vulkan)
        .value("Metal", tenzor::Device::Type::Metal)
        .value("WebGPU", tenzor::Device::Type::WebGPU);

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
        .def_property_readonly("is_cuda", [](const tenzor::Tensor& self) {
            return self.device().type == tenzor::Device::Type::CUDA;
        })
        .def_property_readonly("is_cpu", [](const tenzor::Tensor& self) {
            return self.device().type == tenzor::Device::Type::CPU;
        })
        .def_property_readonly("numel", &tenzor::Tensor::numel)
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
        .def("to", py::overload_cast<tenzor::Device>(&tenzor::Tensor::to, py::const_),
             py::arg("device"),
             "Move tensor to specified device",
             py::call_guard<py::gil_scoped_release>())
        .def("to", py::overload_cast<tenzor::MemoryFormat>(&tenzor::Tensor::to, py::const_),
             py::arg("memory_format"),
             "Convert tensor to specified memory format (e.g., channels_last for NHWC)",
             py::call_guard<py::gil_scoped_release>())
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
        .def("clone", &tenzor::Tensor::clone)
        .def("detach", &tenzor::Tensor::detach)
        .def("contiguous", &tenzor::Tensor::contiguous)
        .def("fill_", &tenzor::Tensor::fill_, py::arg("value"),
             "Fill tensor with scalar value in-place")
        .def("zero_", &tenzor::Tensor::zero_,
             "Fill tensor with zeros in-place")
        // Buffer protocol support (enables memoryview, numpy.asarray, etc.)
        .def_buffer([](tenzor::Tensor& t) -> py::buffer_info {
            if (t.device().type != tenzor::Device::Type::CPU) {
                throw std::runtime_error("Buffer protocol only supported for CPU tensors");
            }
            if (!t.is_contiguous()) {
                throw std::runtime_error("Buffer protocol only supported for contiguous tensors");
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
                    return py::cast(static_cast<float>(t.data<tenzor::Float16>()[0]));
                case tenzor::DType::BFloat16:
                    return py::cast(static_cast<float>(t.data<tenzor::BFloat16>()[0]));
                default:
                    throw std::runtime_error("Unsupported dtype for item()");
            }
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
        // Arithmetic operators - Tensor-Scalar (use Tensor scalar operators directly)
        .def("__add__", [](const tenzor::Tensor& a, float b) -> tenzor::Tensor {
             return a + static_cast<double>(b);
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__radd__", [](const tenzor::Tensor& a, float b) -> tenzor::Tensor {
             return a + static_cast<double>(b);  // addition is commutative
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__sub__", [](const tenzor::Tensor& a, float b) -> tenzor::Tensor {
             return a - static_cast<double>(b);
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__rsub__", [](const tenzor::Tensor& a, float b) -> tenzor::Tensor {
             return tenzor::neg(a - static_cast<double>(b));  // b - a = -(a - b)
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__mul__", [](const tenzor::Tensor& a, float b) -> tenzor::Tensor {
             return a * static_cast<double>(b);
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__rmul__", [](const tenzor::Tensor& a, float b) -> tenzor::Tensor {
             return a * static_cast<double>(b);  // multiplication is commutative
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__truediv__", [](const tenzor::Tensor& a, float b) -> tenzor::Tensor {
             return a / static_cast<double>(b);
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__rtruediv__", [](const tenzor::Tensor& a, float b) -> tenzor::Tensor {
             // b / a = b * reciprocal(a)
             return tenzor::reciprocal(a) * static_cast<double>(b);
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__pow__", [](const tenzor::Tensor& a, float exponent) -> tenzor::Tensor {
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
        .def("__mod__", [](const tenzor::Tensor& a, float b) -> tenzor::Tensor {
             auto b_tensor = tenzor::full(std::vector<int64_t>{}, static_cast<double>(b),
                                          a.dtype(), a.device());
             return tenzor::fmod(a, b_tensor);
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__rmod__", [](const tenzor::Tensor& a, float b) -> tenzor::Tensor {
             auto b_tensor = tenzor::full(std::vector<int64_t>{}, static_cast<double>(b),
                                          a.dtype(), a.device());
             return tenzor::fmod(b_tensor, a);
             }, py::is_operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("__floordiv__", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
             return tenzor::floor(a / b);
             }, py::is_operator(), "Element-wise floor division",
             py::call_guard<py::gil_scoped_release>())
        .def("__floordiv__", [](const tenzor::Tensor& a, float b) -> tenzor::Tensor {
             return tenzor::floor(a / static_cast<double>(b));
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
             if (t.dtype() == tenzor::DType::Float64) return t.item<double>();
             if (t.dtype() == tenzor::DType::Int32) return static_cast<double>(t.item<int32_t>());
             if (t.dtype() == tenzor::DType::Int64) return static_cast<double>(t.item<int64_t>());
             return static_cast<double>(t.item<float>());
             })
        .def("__int__", [](const tenzor::Tensor& t) -> int64_t {
             if (t.numel() != 1) throw py::value_error("only one element tensors can be converted to Python scalars");
             if (t.dtype() == tenzor::DType::Int64) return t.item<int64_t>();
             if (t.dtype() == tenzor::DType::Int32) return static_cast<int64_t>(t.item<int32_t>());
             if (t.dtype() == tenzor::DType::Float64) return static_cast<int64_t>(t.item<double>());
             return static_cast<int64_t>(t.item<float>());
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
        // Iteration: yields t[0], t[1], ..., t[n-1]
        .def("__iter__", [](const tenzor::Tensor& t) {
             if (t.ndim() == 0) throw py::type_error("iteration over a 0-d tensor");
             int64_t n = t.shape()[0];
             py::list items;
             for (int64_t i = 0; i < n; ++i) {
                 items.append(py::cast(t[i]));
             }
             return py::iter(items);
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
                if (state.size() != 4) throw std::runtime_error("Invalid pickle state");

                auto shape = state[0].cast<std::vector<int64_t>>();
                auto dtype = static_cast<tenzor::DType>(state[1].cast<int>());
                auto device_str = state[2].cast<std::string>();
                auto data = state[3].cast<std::string>();

                // Create tensor on CPU
                tenzor::Tensor t(shape, dtype, tenzor::Device::cpu());
                size_t nbytes = static_cast<size_t>(t.numel()) * tenzor::dtype_size(dtype);
                if (data.size() != nbytes) {
                    throw std::runtime_error("Pickle data size mismatch");
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
            enum class IndexKind { Int, Slice, Tuple, TensorMask };
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
            } else if (py::isinstance<py::tuple>(key)) {
                kind = IndexKind::Tuple;
                py::tuple indices = py::cast<py::tuple>(key);
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
            } else if (py::isinstance<tenzor::Tensor>(key)) {
                kind = IndexKind::TensorMask;
                mask_tensor = key.cast<tenzor::Tensor>();
            } else {
                throw std::runtime_error("Unsupported index type: expected int, slice, or Tensor");
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
            } else if (py::isinstance<py::int_>(value) && !py::isinstance<py::bool_>(value)) {
                is_scalar_value = true;
                is_integer_scalar = true;
                int_scalar_value = py::cast<int64_t>(value);
                scalar_value = static_cast<double>(int_scalar_value);
            } else if (py::isinstance<py::float_>(value) || py::isinstance<py::int_>(value)) {
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
         }, "Matrix multiplication",
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
    m.def("scatter_add", &tenzor::scatter_add, "Scatter-add elements along dimension",
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
        // Arithmetic operators - Variable-Variable
        .def("__add__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return a + b;
        }, py::is_operator())
        .def("__sub__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return a - b;
        }, py::is_operator())
        .def("__mul__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return a * b;
        }, py::is_operator())
        .def("__truediv__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return a / b;
        }, py::is_operator())
        // Arithmetic operators - Variable-Scalar
        .def("__add__", [](const tenzor::Variable& a, float b) {
            return a + b;
        }, py::is_operator())
        .def("__radd__", [](const tenzor::Variable& a, float b) {
            return a + b;
        }, py::is_operator())
        .def("__mul__", [](const tenzor::Variable& a, float b) {
            return a * b;
        }, py::is_operator())
        .def("__rmul__", [](const tenzor::Variable& a, float b) {
            return a * b;
        }, py::is_operator())
        .def("__sub__", [](const tenzor::Variable& a, float b) {
            return a - b;
        }, py::is_operator())
        .def("__rsub__", [](const tenzor::Variable& a, float b) {
            // b - a = -(a - b)
            return (a - b) * -1.0f;
        }, py::is_operator())
        .def("__truediv__", [](const tenzor::Variable& a, float b) {
            return a / b;
        }, py::is_operator())
        .def("__rtruediv__", [](const tenzor::Variable& a, float b) {
            // b / a: create scalar variable with value b, divide by a
            auto b_var = tenzor::Variable(
                tenzor::full({}, static_cast<double>(b), a.dtype(), a.device()), false);
            return b_var / a;
        }, py::is_operator())
        .def("__neg__", [](const tenzor::Variable& a) {
            return a * -1.0f;
        }, py::is_operator())
        // Matrix multiplication
        .def("__matmul__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return a.matmul(b);
        }, py::is_operator())
        .def("__rmatmul__", [](const tenzor::Variable& a, const tenzor::Variable& b) {
            return b.matmul(a);
        }, py::is_operator())
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
            a.tensor().bump_version();
            return a;
        }, py::is_operator())
        .def("__isub__", [](tenzor::Variable& a, const tenzor::Variable& b) -> tenzor::Variable& {
            tenzor::sub_(a.tensor(), b.tensor());
            a.tensor().bump_version();
            return a;
        }, py::is_operator())
        .def("__imul__", [](tenzor::Variable& a, const tenzor::Variable& b) -> tenzor::Variable& {
            tenzor::mul_(a.tensor(), b.tensor());
            a.tensor().bump_version();
            return a;
        }, py::is_operator())
        .def("__itruediv__", [](tenzor::Variable& a, const tenzor::Variable& b) -> tenzor::Variable& {
            tenzor::div_(a.tensor(), b.tensor());
            a.tensor().bump_version();
            return a;
        }, py::is_operator())
        // Scalar in-place
        .def("__iadd__", [](tenzor::Variable& a, float b) -> tenzor::Variable& {
            auto scalar_t = tenzor::full({1}, static_cast<double>(b), a.dtype(), a.device());
            tenzor::add_(a.tensor(), scalar_t);
            a.tensor().bump_version();
            return a;
        }, py::is_operator())
        .def("__isub__", [](tenzor::Variable& a, float b) -> tenzor::Variable& {
            auto scalar_t = tenzor::full({1}, static_cast<double>(b), a.dtype(), a.device());
            tenzor::sub_(a.tensor(), scalar_t);
            a.tensor().bump_version();
            return a;
        }, py::is_operator())
        .def("__imul__", [](tenzor::Variable& a, float b) -> tenzor::Variable& {
            auto scalar_t = tenzor::full({1}, static_cast<double>(b), a.dtype(), a.device());
            tenzor::mul_(a.tensor(), scalar_t);
            a.tensor().bump_version();
            return a;
        }, py::is_operator())
        .def("__itruediv__", [](tenzor::Variable& a, float b) -> tenzor::Variable& {
            auto scalar_t = tenzor::full({1}, static_cast<double>(b), a.dtype(), a.device());
            tenzor::div_(a.tensor(), scalar_t);
            a.tensor().bump_version();
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
        .def("dim", [](const tenzor::Variable& v) { return v.tensor().ndim(); })
        .def("__index__", [](const tenzor::Variable& v) {
            return v.tensor().item<int64_t>();
        })
        .def("__iter__", [](const tenzor::Variable& v) {
            auto t = v.tensor();
            if (t.ndim() == 0) throw py::type_error("iteration over a 0-d Variable");
            int64_t n = t.shape()[0];
            py::list result;
            for (int64_t i = 0; i < n; ++i) {
                result.append(tenzor::Variable(t[i].contiguous().clone(), v.requires_grad()));
            }
            return py::iter(result);
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
                auto dtype = static_cast<tenzor::DType>(state[1].cast<int>());
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

    py::class_<PyFunctionCtx>(m, "FunctionCtx",
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
                saved_tensors_ = ctx_->saved_tensors_;
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
             "Perform forward pass (calls forward_impl with hooks)")
        .def("forward_impl", &tenzor::nn::Module::forward_impl,
             py::arg("input"),
             "Implementation of forward pass - override this in subclasses")
        .def("__call__", &tenzor::nn::Module::operator(),
             py::arg("input"),
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
        }, py::arg("input"), py::arg("hx") = std::pair<tenzor::Variable, tenzor::Variable>{})
        .def("__repr__", [](const tenzor::nn::LSTM&) { return "LSTM()"; });

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
        }, py::arg("input"), py::arg("hx") = tenzor::Variable{})
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
                           bool need_weights) {
            return self.forward(query, key, value, key_padding_mask, attn_mask, need_weights);
        }, py::arg("query"), py::arg("key"), py::arg("value"),
           py::arg("key_padding_mask") = tenzor::Tensor{},
           py::arg("attn_mask") = tenzor::Tensor{},
           py::arg("need_weights") = true)
        .def("__repr__", [](const tenzor::nn::MultiheadAttention&) { return "MultiheadAttention()"; });

    py::class_<tenzor::nn::PositionalEncoding, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::PositionalEncoding>>(nn, "PositionalEncoding")
        .def(py::init<int64_t, int64_t, double>(),
             py::arg("d_model"), py::arg("max_len") = 5000, py::arg("dropout") = 0.0)
        .def("forward", &tenzor::nn::PositionalEncoding::forward);

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
             "Compute MSE loss between input and target")
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
             "Compute L1 loss between input and target")
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
             "Compute Smooth L1 loss between input and target")
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
             "Compute cross entropy loss between input logits and target class indices")
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
             "Compute NLL loss between input log-probabilities and target class indices")
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
             "Compute BCE loss between input probabilities and binary targets")
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
             "Compute BCE with logits loss between input logits and binary targets")
        .def("__call__", &tenzor::nn::BCEWithLogitsLoss::operator(),
             py::arg("input"), py::arg("target"),
             "Compute BCE with logits loss between input logits and binary targets");

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

    py::class_<tenzor::nn::CTCLoss>(nn, "CTCLoss",
        "Connectionist Temporal Classification loss for sequence-to-sequence tasks")
        .def(py::init<const std::string&, int64_t, bool>(),
             py::arg("reduction") = "mean",
             py::arg("blank") = 0,
             py::arg("zero_infinity") = false)
        .def("forward", &tenzor::nn::CTCLoss::forward,
             py::arg("log_probs"), py::arg("targets"),
             py::arg("input_lengths"), py::arg("target_lengths"))
        .def("__call__", &tenzor::nn::CTCLoss::operator(),
             py::arg("log_probs"), py::arg("targets"),
             py::arg("input_lengths"), py::arg("target_lengths"));

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

    // Gradient clipping utilities
    nn.def("clip_grad_norm_", &tenzor::nn::utils::clip_grad_norm_,
           py::arg("parameters"), py::arg("max_norm"), py::arg("norm_type") = 2.0,
           "Clip gradients by global norm, returns total norm before clipping");

    nn.def("clip_grad_value_", &tenzor::nn::utils::clip_grad_value_,
           py::arg("parameters"), py::arg("clip_value"),
           "Clip gradient values to [-clip_value, clip_value]");

    // Optimizers
    auto optim = m.def_submodule("optim", "Optimization algorithms");

    // Optimizer base class - needed for functions that accept any optimizer
    py::class_<tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::Optimizer>>(optim, "Optimizer",
        "Base class for all optimizers")
        .def("zero_grad", &tenzor::optim::Optimizer::zero_grad, "Zero out all parameter gradients")
        .def("state_dict", &tenzor::optim::Optimizer::state_dict, "Get optimizer state dictionary")
        .def("load_state_dict", &tenzor::optim::Optimizer::load_state_dict, py::arg("state"),
             "Load optimizer state dictionary");

    py::class_<tenzor::optim::SGD, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::SGD>>(optim, "SGD")
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

    py::class_<tenzor::optim::Adam, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::Adam>>(optim, "Adam")
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

    py::class_<tenzor::optim::AdamW, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::AdamW>>(optim, "AdamW")
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
            py::arg("input"))
        .def("synchronize_gradients", &tenzor::distributed::DistributedDataParallel::synchronize_gradients,
            "Synchronize gradients across all processes")
        .def("sync_comm", &tenzor::distributed::DistributedDataParallel::sync_comm,
            "Wait for pending async all-reduce operations")
        .def("auto_sync_gradients", &tenzor::distributed::DistributedDataParallel::auto_sync_gradients,
            "Enable or disable automatic gradient synchronization",
            py::arg("enabled"))
        .def("reset_buckets", &tenzor::distributed::DistributedDataParallel::reset_buckets,
            "Reset bucket ready states for next iteration");

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
    #endif

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
                   py::arg("A"));
    linalg_mod.def("inv", &tenzor::linalg::inv, "Compute matrix inverse",
                   py::arg("A"));
    linalg_mod.def("solve", &tenzor::linalg::solve, "Solve linear system AX = B",
                   py::arg("A"), py::arg("B"));
    linalg_mod.def("cholesky", &tenzor::linalg::cholesky, "Cholesky decomposition",
                   py::arg("A"), py::arg("upper") = false);
    linalg_mod.def("norm", &tenzor::linalg::norm, "Matrix norm",
                   py::arg("A"), py::arg("ord") = "fro");
    linalg_mod.def("slogdet", [](const tenzor::Tensor& A) {
        auto [sign, logabsdet] = tenzor::linalg::slogdet(A);
        return py::make_tuple(sign, logabsdet);
    }, "Sign and log of absolute determinant", py::arg("A"));
    linalg_mod.def("svd", [](const tenzor::Tensor& A, bool full_matrices) {
        auto [U, S, Vh] = tenzor::linalg::svd(A, full_matrices);
        return py::make_tuple(U, S, Vh);
    }, "Singular Value Decomposition", py::arg("A"), py::arg("full_matrices") = true);
    linalg_mod.def("qr", [](const tenzor::Tensor& A) {
        auto [Q, R] = tenzor::linalg::qr(A);
        return py::make_tuple(Q, R);
    }, "QR decomposition", py::arg("A"));
    linalg_mod.def("eigh", [](const tenzor::Tensor& A) {
        auto [eigenvalues, eigenvectors] = tenzor::linalg::eigh(A);
        return py::make_tuple(eigenvalues, eigenvectors);
    }, "Eigendecomposition of symmetric matrix", py::arg("A"));
    linalg_mod.def("eigvalsh", &tenzor::linalg::eigvalsh,
                   "Eigenvalues of symmetric matrix", py::arg("A"));

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
             "Execute graph with runtime inputs")
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
        .def("step", &tenzor::optim::AdamAtan2::step)
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
        .value("UINT8", QuantDType::UINT8, "Unsigned 8-bit integer [0, 255]");

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
             "INT8 quantized linear layer");

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
        .value("CSR", tenzor::SparseLayout::CSR);

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
        .def("to_dense", &tenzor::SparseTensor::to_dense, "Convert to dense tensor",
             py::call_guard<py::gil_scoped_release>())
        .def("to_coo", &tenzor::SparseTensor::to_coo, "Convert to COO format",
             py::call_guard<py::gil_scoped_release>())
        .def("to_csr", &tenzor::SparseTensor::to_csr, "Convert to CSR format",
             py::call_guard<py::gil_scoped_release>())
        .def("coalesce", &tenzor::SparseTensor::coalesce, "Coalesce COO tensor",
             py::call_guard<py::gil_scoped_release>())
        .def("to", &tenzor::SparseTensor::to, "Transfer to device",
             py::arg("device"),
             py::call_guard<py::gil_scoped_release>())
        .def("__repr__", [](const tenzor::SparseTensor& s) {
            std::ostringstream ss;
            ss << "SparseTensor(layout="
               << (s.layout() == tenzor::SparseLayout::COO ? "COO" : "CSR")
               << ", shape=[";
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
}
