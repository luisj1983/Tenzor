// Core Python bindings: DType, Device, Tensor, Variable, context managers.
// Extracted from python/bindings.cpp as the final major extraction in the
// bindings split effort (P3.4).

#include "register.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <iostream>
#include <sstream>
#include <cstring>

#include <tenzor/io/image.hpp>

#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/autograd/anomaly_mode.hpp>
#include <tenzor/autograd/checkpoint.hpp>
#include <tenzor/core/device_guard.hpp>
#include <tenzor/core/dlpack.hpp>
#include <tenzor/ops/custom_op.hpp>
#include <tenzor/ops/indexing.hpp>
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/transform.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/windows.hpp>
#include <tenzor/ops/fp8_scaling.hpp>
#include <tenzor/backend/loader.hpp>
#include <tenzor/backend/backend.hpp>
#include <tenzor/backend/cuda_config.hpp>
#include <tenzor/backend/cuda_graph.hpp>
#include <tenzor/backend/dispatch_table.hpp>
#include <tenzor/backend/cpu_caching_allocator.hpp>
#include <tenzor/nn/serialize.hpp>
#include <tenzor/nn/layers/sync_batchnorm.hpp>
#include <tenzor/distributions/distribution.hpp>
#include <tenzor/distributions/transforms.hpp>
#include <tenzor/distributions/transformed.hpp>
#include <tenzor/distributions/independent.hpp>
#include <tenzor/distributions/mixture.hpp>
#include <tenzor/ops/foreach.hpp>
#include "../numpy_interop.hpp"

namespace py = pybind11;

namespace tenzor::python {

void register_core(py::module_& m) {
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
        // String constructor: accepts "cpu", "cuda", "cuda:1", "rocm", etc.
        // Paired with py::implicitly_convertible below so every pybind
        // function that takes `const Device&` can be called with a Python
        // string transparently — matches PyTorch's Python API ergonomics.
        .def(py::init([](const std::string& s) {
            return tenzor::Device::from_string(s);
        }), py::arg("spec"))
        .def_static("cpu", &tenzor::Device::cpu)
        .def_static("cuda", &tenzor::Device::cuda, py::arg("index") = 0)
        .def_static("rocm", &tenzor::Device::rocm, py::arg("index") = 0)
        .def_static("oneapi", &tenzor::Device::oneapi, py::arg("index") = 0)
        .def_static("vulkan", &tenzor::Device::vulkan, py::arg("index") = 0)
        .def_readonly("type", &tenzor::Device::type)
        .def_readonly("index", &tenzor::Device::index)
        .def("__repr__", [](const tenzor::Device& d) {
            return d.to_string();
        })
        .def("__str__", [](const tenzor::Device& d) {
            return d.to_string();
        })
        .def("__eq__", [](const tenzor::Device& a, const tenzor::Device& b) {
            return a.type == b.type && a.index == b.index;
        }, py::is_operator())
        .def("__eq__", [](const tenzor::Device& a, const std::string& s) {
            try {
                auto b = tenzor::Device::from_string(s);
                return a.type == b.type && a.index == b.index;
            } catch (...) { return false; }
        }, py::is_operator())
        .def("__ne__", [](const tenzor::Device& a, const tenzor::Device& b) {
            return !(a.type == b.type && a.index == b.index);
        }, py::is_operator())
        .def("__hash__", [](const tenzor::Device& d) {
            return std::hash<int>{}(static_cast<int>(d.type)) ^
                   (std::hash<int>{}(d.index) << 1);
        });
    // Accept string wherever the Python API expects a Device. Parses
    // through Device::from_string, so "cpu" / "cuda:0" / "rocm" / etc.
    // all work. Must be declared after the class binding.
    py::implicitly_convertible<std::string, tenzor::Device>();

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
        .value("qint4x2", tenzor::DType::QInt4x2)
        .value("fp8_e4m3", tenzor::DType::FP8_E4M3)
        .value("fp8_e5m2", tenzor::DType::FP8_E5M2);

    // Quantization functions
    m.def("quantize_per_tensor", &tenzor::quantize_per_tensor,
          py::arg("input"), py::arg("scale"), py::arg("zero_point"),
          py::arg("dtype") = tenzor::DType::QInt8,
          "Quantize a float tensor to int8/uint8 with scale and zero_point");

    // FP8 scaling utilities
    // The per-tensor scaling params returned by quantize_to_fp8. Expose its
    // three fields so round-trip tests can read `params.scale`.
    py::class_<tenzor::FP8ScalingParams>(m, "FP8ScalingParams",
        "Per-tensor FP8 quantization parameters")
        .def_readwrite("scale", &tenzor::FP8ScalingParams::scale)
        .def_readwrite("amax", &tenzor::FP8ScalingParams::amax)
        .def_readwrite("fp8_dtype", &tenzor::FP8ScalingParams::fp8_dtype);

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
        // `device_type` returns a short string like "cpu"/"cuda"/"rocm"
        // so tests and user code can compare directly against the string
        // parameter names they already use in parametrized fixtures.
        .def_property_readonly("device_type", [](const tenzor::Tensor& self) {
            switch (self.device().type) {
                case tenzor::Device::Type::CPU:    return "cpu";
                case tenzor::Device::Type::CUDA:   return "cuda";
                case tenzor::Device::Type::ROCm:   return "rocm";
                case tenzor::Device::Type::OneAPI: return "oneapi";
                case tenzor::Device::Type::Vulkan: return "vulkan";
                case tenzor::Device::Type::MPS:    return "mps";
                case tenzor::Device::Type::COUNT:
                    throw std::runtime_error("Device::Type::COUNT is a sentinel value, not a real device type");
            }
            return "unknown";
        })
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
        .def("repeat_interleave", [](const tenzor::Tensor& self, int64_t repeats, std::optional<int64_t> dim) {
             return tenzor::repeat_interleave(self, repeats, dim);
             }, py::arg("repeats"), py::arg("dim") = py::none(),
             "Repeat each element of tensor a given number of times",
             py::call_guard<py::gil_scoped_release>())
        .def("repeat_interleave", [](const tenzor::Tensor& self, const tenzor::Tensor& repeats, std::optional<int64_t> dim) {
             return tenzor::repeat_interleave(self, repeats, dim);
             }, py::arg("repeats"), py::arg("dim") = py::none(),
             "Repeat each element by per-element counts",
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
        .def("unfold", &tenzor::Tensor::unfold,
             py::arg("dim"), py::arg("size"), py::arg("step"),
             "Extract sliding windows along a dimension",
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
        // ---------------------------------------------------------------
        // Named in-place methods (Phase 2.1 — match torch.Tensor API).
        //
        // Arithmetic variants forward to the existing tenzor::add_/sub_/mul_/div_
        // C++ free functions which actually perform an in-place kernel
        // call. The unary math and activation variants take a "compute
        // out-of-place then assign" shortcut: pybind11 passes `Tensor&
        // self` as a reference to the C++ object backing the Python
        // instance, and Tensor's copy-assign is pImpl-based, so writing
        // `self = ...` replaces the impl and the Python-visible tensor
        // picks up the new value. Semantically this matches what the
        // user expects from `.abs_()` etc.; the efficiency gap only
        // matters at very large tensor sizes and a future native
        // in-place kernel can slot in without API changes.
        // ---------------------------------------------------------------
        .def("add_", [](tenzor::Tensor& self, const tenzor::Tensor& other) -> tenzor::Tensor& {
                tenzor::add_(self, other); return self;
            }, py::arg("other"),
            "In-place addition: self += other. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("sub_", [](tenzor::Tensor& self, const tenzor::Tensor& other) -> tenzor::Tensor& {
                tenzor::sub_(self, other); return self;
            }, py::arg("other"),
            "In-place subtraction: self -= other. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("mul_", [](tenzor::Tensor& self, const tenzor::Tensor& other) -> tenzor::Tensor& {
                tenzor::mul_(self, other); return self;
            }, py::arg("other"),
            "In-place multiplication: self *= other. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("div_", [](tenzor::Tensor& self, const tenzor::Tensor& other) -> tenzor::Tensor& {
                tenzor::div_(self, other); return self;
            }, py::arg("other"),
            "In-place division: self /= other. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("pow_", [](tenzor::Tensor& self, float exponent) -> tenzor::Tensor& {
                self = tenzor::pow(self, exponent); return self;
            }, py::arg("exponent"),
            "In-place power: self = self ** exponent. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("neg_", [](tenzor::Tensor& self) -> tenzor::Tensor& {
                self = tenzor::neg(self); return self;
            },
            "In-place negation: self = -self. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("abs_", [](tenzor::Tensor& self) -> tenzor::Tensor& {
                self = tenzor::abs(self); return self;
            },
            "In-place absolute value. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("sqrt_", [](tenzor::Tensor& self) -> tenzor::Tensor& {
                self = tenzor::sqrt(self); return self;
            },
            "In-place square root. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("exp_", [](tenzor::Tensor& self) -> tenzor::Tensor& {
                self = tenzor::exp(self); return self;
            },
            "In-place exponential. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("log_", [](tenzor::Tensor& self) -> tenzor::Tensor& {
                self = tenzor::log(self); return self;
            },
            "In-place natural log. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("reciprocal_", [](tenzor::Tensor& self) -> tenzor::Tensor& {
                self = tenzor::reciprocal(self); return self;
            },
            "In-place reciprocal (1/x). Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("sign_", [](tenzor::Tensor& self) -> tenzor::Tensor& {
                self = tenzor::sign(self); return self;
            },
            "In-place sign. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        // Activations — these have real in-place kernels registered
        // under OpId::ReLUInplace / SigmoidInplace / TanhInplace, so we
        // dispatch directly through the table rather than computing
        // out-of-place.
        .def("relu_", [](tenzor::Tensor& self) -> tenzor::Tensor& {
                auto& table = tenzor::DispatchTableRegistry::get_table(self.device().type);
                table.dispatch_inplace(tenzor::OpId::ReLUInplace, self, {}, {});
                return self;
            },
            "In-place ReLU. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("sigmoid_", [](tenzor::Tensor& self) -> tenzor::Tensor& {
                auto& table = tenzor::DispatchTableRegistry::get_table(self.device().type);
                table.dispatch_inplace(tenzor::OpId::SigmoidInplace, self, {}, {});
                return self;
            },
            "In-place sigmoid. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("tanh_", [](tenzor::Tensor& self) -> tenzor::Tensor& {
                auto& table = tenzor::DispatchTableRegistry::get_table(self.device().type);
                table.dispatch_inplace(tenzor::OpId::TanhInplace, self, {}, {});
                return self;
            },
            "In-place tanh. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        // Clamps.
        .def("clamp_", [](tenzor::Tensor& self, float min_val, float max_val) -> tenzor::Tensor& {
                self = tenzor::clamp(self, min_val, max_val); return self;
            }, py::arg("min"), py::arg("max"),
            "In-place clamp to [min, max]. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("clamp_min_", [](tenzor::Tensor& self, float min_val) -> tenzor::Tensor& {
                self = tenzor::clamp_min(self, min_val); return self;
            }, py::arg("min"),
            "In-place lower clamp: self = max(self, min). Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("clamp_max_", [](tenzor::Tensor& self, float max_val) -> tenzor::Tensor& {
                self = tenzor::clamp_max(self, max_val); return self;
            }, py::arg("max"),
            "In-place upper clamp: self = min(self, max). Returns self.",
            py::call_guard<py::gil_scoped_release>())
        // Copy / randomization.
        .def("copy_", [](tenzor::Tensor& self, const tenzor::Tensor& src) -> tenzor::Tensor& {
                // In-place copy: src data is written into self's existing
                // storage so other aliases of self (e.g. a Parameter's
                // stored tensor) observe the new values. The previous
                // implementation did `self = converted` which reassigned
                // the receiver's impl pointer while leaving aliased
                // tensors pointing at the original buffer — silently
                // broke `parameter.tensor().copy_(...)` used throughout
                // the parity tests and the ONNX import weight-copy path.
                // Audit J10: real broadcasting. Previously copy_ required
                // src.shape() == self.shape() exactly; now we broadcast `src`
                // up to self's shape (the destination is the broadcast target,
                // PyTorch convention) before doing the byte copy. `src` shapes
                // that don't broadcast-compatible with self still throw.
                auto ss = self.shape();
                auto xs = src.shape();
                tenzor::Tensor converted = src;
                if (ss.size() != xs.size() ||
                    !std::equal(ss.begin(), ss.end(), xs.begin())) {
                    // Try broadcasting src to self's shape. broadcast_shapes
                    // returns the result of standard NumPy-style broadcasting;
                    // we require the result equals self's shape (src can't
                    // override dimensions where self already has size > 1).
                    std::vector<int64_t> ss_vec(ss.begin(), ss.end());
                    std::vector<int64_t> xs_vec(xs.begin(), xs.end());
                    auto bs = tenzor::broadcast_shapes(
                        std::span<const int64_t>(ss_vec),
                        std::span<const int64_t>(xs_vec));
                    if (bs.size() != ss_vec.size() ||
                        !std::equal(bs.begin(), bs.end(), ss_vec.begin())) {
                        throw py::value_error(
                            "Tensor.copy_: src shape is not broadcast-compatible "
                            "with self's shape");
                    }
                    converted = tenzor::broadcast_to(converted, ss_vec);
                }
                if (converted.dtype() != self.dtype()) {
                    converted = converted.to(self.dtype());
                }
                if (converted.device() != self.device()) {
                    converted = converted.to(self.device());
                }
                if (!converted.is_contiguous()) {
                    converted = converted.contiguous();
                }
                size_t nbytes = static_cast<size_t>(self.numel()) * self.dtype_size();
                if (nbytes == 0) return self;

                if (self.is_contiguous()) {
                    // Fast path: direct device-local byte copy.
                    if (self.device().type == tenzor::Device::Type::CPU) {
                        std::memcpy(self.data_ptr(), converted.data_ptr(), nbytes);
                    } else {
                        auto* backend = tenzor::backend_registry().get_backend(self.device().type);
                        if (!backend) {
                            throw std::runtime_error(
                                "Tensor.copy_: no backend registered for device");
                        }
                        backend->copy(self.data_ptr(), converted.data_ptr(), nbytes,
                                      tenzor::CopyKind::DeviceToDevice);
                    }
                } else {
                    // Non-contiguous destination: fall back to elementwise
                    // via fill-style dispatch. We materialize converted
                    // into a contiguous buffer, then write element-by-
                    // element respecting self's strides. The existing
                    // tensor assignment op is not strided-aware either,
                    // so we do this via a scalar loop on CPU only.
                    if (self.device().type != tenzor::Device::Type::CPU) {
                        throw std::runtime_error(
                            "Tensor.copy_: non-contiguous destination only "
                            "supported on CPU; call .contiguous() first");
                    }
                    auto iter_shape = std::vector<int64_t>(ss.begin(), ss.end());
                    int64_t n = self.numel();
                    int64_t ndim = static_cast<int64_t>(iter_shape.size());
                    auto strides_self = self.strides();
                    const uint8_t* src_bytes = static_cast<const uint8_t*>(converted.data_ptr());
                    uint8_t* dst_bytes = static_cast<uint8_t*>(self.data_ptr());
                    size_t elt = self.dtype_size();
                    for (int64_t lin = 0; lin < n; ++lin) {
                        int64_t rem = lin;
                        int64_t dst_off = 0;
                        for (int64_t d = ndim - 1; d >= 0; --d) {
                            int64_t idx = rem % iter_shape[d];
                            rem /= iter_shape[d];
                            dst_off += idx * strides_self[d];
                        }
                        std::memcpy(dst_bytes + dst_off * elt,
                                    src_bytes + lin * elt, elt);
                    }
                }
                return self;
            }, py::arg("src"),
            "Copy src into self in-place (with dtype/device conversion). Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("normal_", [](tenzor::Tensor& self, double mean, double std) -> tenzor::Tensor& {
                auto shape_vec = std::vector<int64_t>(self.shape().begin(), self.shape().end());
                auto sampled = tenzor::randn(shape_vec, self.dtype(), self.device());
                if (std != 1.0 || mean != 0.0) {
                    auto scale = tenzor::full({1}, std, self.dtype(), self.device());
                    auto shift = tenzor::full({1}, mean, self.dtype(), self.device());
                    sampled = sampled * scale + shift;
                }
                self = sampled; return self;
            }, py::arg("mean") = 0.0, py::arg("std") = 1.0,
            "Fill self with samples from N(mean, std^2) in-place. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        .def("uniform_", [](tenzor::Tensor& self, double low, double high) -> tenzor::Tensor& {
                auto shape_vec = std::vector<int64_t>(self.shape().begin(), self.shape().end());
                auto sampled = tenzor::rand(shape_vec, self.dtype(), self.device());
                if (low != 0.0 || high != 1.0) {
                    auto scale = tenzor::full({1}, high - low, self.dtype(), self.device());
                    auto shift = tenzor::full({1}, low, self.dtype(), self.device());
                    sampled = sampled * scale + shift;
                }
                self = sampled; return self;
            }, py::arg("low") = 0.0, py::arg("high") = 1.0,
            "Fill self with samples from U[low, high) in-place. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        // ---------------------------------------------------------------
        // Phase 2.2 — Tensor accessors matching torch.Tensor shape API.
        // ---------------------------------------------------------------
        .def_property_readonly("T", [](const tenzor::Tensor& self) -> tenzor::Tensor {
                // torch.Tensor.T: 2-D transpose only. For anything else
                // torch raises (and has soft-deprecated the ND variant);
                // we match that here.
                if (self.ndim() != 2) {
                    throw py::value_error(
                        "tensor.T expects a 2-D tensor, got ndim=" +
                        std::to_string(self.ndim()));
                }
                return tenzor::transpose(self, 0, 1);
            },
            "2-D transpose (equivalent to t.transpose(0, 1)).")
        .def_property_readonly("mT", [](const tenzor::Tensor& self) -> tenzor::Tensor {
                // torch.Tensor.mT: swap the last two dims. Requires ndim>=2.
                if (self.ndim() < 2) {
                    throw py::value_error(
                        "tensor.mT expects ndim>=2, got ndim=" +
                        std::to_string(self.ndim()));
                }
                int64_t nd = static_cast<int64_t>(self.ndim());
                return tenzor::transpose(self, nd - 2, nd - 1);
            },
            "Batched transpose: swap the last two dimensions. "
            "Equivalent to t.transpose(-2, -1).")
        .def_property_readonly("H", [](const tenzor::Tensor& self) -> tenzor::Tensor {
                // torch.Tensor.H: conjugate transpose of last two dims.
                if (self.ndim() < 2) {
                    throw py::value_error(
                        "tensor.H expects ndim>=2, got ndim=" +
                        std::to_string(self.ndim()));
                }
                int64_t nd = static_cast<int64_t>(self.ndim());
                return tenzor::conj(tenzor::transpose(self, nd - 2, nd - 1));
            },
            "Conjugate transpose of the last two dimensions. "
            "Equivalent to t.mT.conj().")
        .def("stride", [](const tenzor::Tensor& self) -> py::tuple {
                // stride() with no argument returns a tuple of all strides,
                // matching torch.Tensor.stride().
                auto s = self.strides();
                py::tuple out(s.size());
                for (size_t i = 0; i < s.size(); ++i) {
                    out[i] = static_cast<int64_t>(s[i]);
                }
                return out;
            },
            "Return the stride of every dimension as a tuple of ints.")
        .def("stride", [](const tenzor::Tensor& self, int64_t dim) -> int64_t {
                // stride(dim) returns the stride of a single dimension.
                auto s = self.strides();
                int64_t nd = static_cast<int64_t>(s.size());
                if (dim < 0) dim += nd;
                if (dim < 0 || dim >= nd) {
                    throw py::index_error(
                        "stride: dim " + std::to_string(dim) +
                        " out of range for tensor with ndim=" + std::to_string(nd));
                }
                return s[dim];
            },
            py::arg("dim"),
            "Return the stride of the given dimension (supports negative indices).")
        .def("storage_offset", [](const tenzor::Tensor& self) -> int64_t {
                return self.offset();
            },
            "Return the offset (in elements) into the underlying storage.")
        .def("data_ptr", [](const tenzor::Tensor& self) -> std::intptr_t {
                return reinterpret_cast<std::intptr_t>(self.data_ptr());
            },
            "Return the address of the first element as a Python int. "
            "Matches torch.Tensor.data_ptr().")
        .def_property_readonly("is_pinned", [](const tenzor::Tensor& self) -> bool {
                return self.is_pinned();
            },
            "Whether the underlying CPU storage is page-locked. GPU "
            "tensors and non-pinned CPU tensors return False.")
        .def("pin_memory", [](tenzor::Tensor& self) -> tenzor::Tensor& {
                self.pin_memory();
                return self;
            },
            "Page-lock the underlying CPU storage for fast GPU transfers. "
            "No-op on GPU tensors, on non-CUDA builds, or if the "
            "underlying buffer cannot be registered. Returns self.",
            py::call_guard<py::gil_scoped_release>())
        // ---------------------------------------------------------------
        // Phase 2.3 — DLPack protocol.
        //
        // `__dlpack__(stream=None)` returns a PyCapsule named "dltensor"
        // wrapping a DLManagedTensor*. Consumers (np.from_dlpack,
        // torch.from_dlpack, cupy.from_dlpack, jax.dlpack.from_dlpack,
        // tvm.contrib.dlpack) rename the capsule to "used_dltensor" when
        // they take ownership so our destructor doesn't double-free.
        //
        // `__dlpack_device__()` returns (device_type_code, device_id)
        // per the DLPack protocol so consumers can make stream/device
        // decisions without touching the payload.
        // ---------------------------------------------------------------
        .def("__dlpack__", [](const tenzor::Tensor& self, py::object /*stream*/) -> py::capsule {
                // We accept the `stream` argument for protocol compatibility
                // but don't currently synchronize on it — producers that
                // need stream ordering should synchronize their own work
                // before handing out the tensor. A future cuda_stream_wait
                // hook can thread through here without changing the API.
                DLManagedTensor* managed = tenzor::to_dlpack(self);
                // Capsule destructor: if the consumer never renamed the
                // capsule to "used_dltensor", call the DLPack deleter
                // exactly once to release our storage reference.
                auto destructor = [](PyObject* capsule) {
                    // Check the capsule name: if the consumer renamed it
                    // to "used_dltensor", they now own the payload and
                    // we must not delete it.
                    if (PyCapsule_IsValid(capsule, "used_dltensor")) {
                        return;
                    }
                    auto* m = static_cast<DLManagedTensor*>(
                        PyCapsule_GetPointer(capsule, "dltensor"));
                    if (m && m->deleter) {
                        m->deleter(m);
                    }
                    // Clear the Python error state in case GetPointer failed
                    // with the wrong name — we've already handled both
                    // legitimate cases above.
                    if (PyErr_Occurred()) PyErr_Clear();
                };
                return py::capsule(managed, "dltensor", destructor);
            },
            py::arg("stream") = py::none(),
            "DLPack producer hook: return a PyCapsule wrapping a "
            "DLManagedTensor. Enables zero-copy interop with NumPy, JAX, "
            "CuPy, PyTorch, and TVM via their respective from_dlpack() "
            "entry points.")
        .def("__dlpack_device__", [](const tenzor::Tensor& self) -> py::tuple {
                // Return (device_type_code, device_id) as int pair.
                // device_type_code matches the DLDeviceType enum values:
                //   kDLCPU=1, kDLCUDA=2, kDLROCM=10, kDLVulkan=7,
                //   kDLOneAPI=14, kDLMetal=8. See dlpack.h for full list.
                int type_code = 0;
                switch (self.device().type) {
                    case tenzor::Device::Type::CPU:    type_code = kDLCPU;    break;
                    case tenzor::Device::Type::CUDA:   type_code = kDLCUDA;   break;
                    case tenzor::Device::Type::ROCm:   type_code = kDLROCM;   break;
                    case tenzor::Device::Type::Vulkan: type_code = kDLVulkan; break;
                    case tenzor::Device::Type::OneAPI: type_code = kDLOneAPI; break;
                    case tenzor::Device::Type::MPS:    type_code = kDLMetal;  break;
                    case tenzor::Device::Type::COUNT:
                        throw std::runtime_error("__dlpack_device__: COUNT is not a real device");
                }
                return py::make_tuple(type_code, self.device().index);
            },
            "DLPack device hook: return the (device_type_code, device_id) "
            "tuple for this tensor. Part of the DLPack consumer protocol.")
        // Buffer protocol support (enables memoryview, numpy.asarray, etc.)
        .def_buffer([](tenzor::Tensor& t) -> py::buffer_info {
            if (t.device().type != tenzor::Device::Type::CPU) {
                throw std::runtime_error("Buffer protocol only supported for CPU tensors");
            }
            if (!t.is_contiguous()) {
                throw std::runtime_error("Buffer protocol requires a contiguous tensor. Call .contiguous() first.");
            }

            // Map DType to Python struct format string. Audit J9 added the
            // Float16 / Complex64 / Complex128 cases. BF16 has no Python
            // `struct` format code (and no native NumPy type without the
            // optional ml_dtypes package) — throw with a clear, actionable
            // error message rather than letting it silently fall through.
            std::string format;
            switch (t.dtype()) {
                case tenzor::DType::Float32: format = py::format_descriptor<float>::format(); break;
                case tenzor::DType::Float64: format = py::format_descriptor<double>::format(); break;
                case tenzor::DType::Int32:   format = py::format_descriptor<int32_t>::format(); break;
                case tenzor::DType::Int64:   format = py::format_descriptor<int64_t>::format(); break;
                case tenzor::DType::Int16:   format = py::format_descriptor<int16_t>::format(); break;
                case tenzor::DType::Int8:    format = py::format_descriptor<int8_t>::format(); break;
                case tenzor::DType::UInt8:   format = py::format_descriptor<uint8_t>::format(); break;
                case tenzor::DType::Bool:    format = py::format_descriptor<bool>::format(); break;
                // Audit J9: Float16 / BFloat16 / Complex support.
                case tenzor::DType::Float16:    format = "e"; break;   // half-precision (Python 3.6+, NumPy ≥1.21)
                case tenzor::DType::Complex64:  format = "Zf"; break;  // 2 × float32 (NumPy convention)
                case tenzor::DType::Complex128: format = "Zd"; break;  // 2 × float64
                case tenzor::DType::BFloat16:
                    throw std::runtime_error(
                        "Buffer protocol: BFloat16 has no Python struct format "
                        "code and no native NumPy dtype. Convert with "
                        ".to(Float32).numpy() or install ml_dtypes for "
                        "lossless interop.");
                default:
                    throw std::runtime_error("Buffer protocol not supported for dtype");
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
        // Phase 2.4 — NumPy __array__ protocol.
        //
        // Enables `np.asarray(tensor)`, `np.array(tensor)`, NumPy ufunc
        // dispatch, and matplotlib / pandas interop to work on Tenzor
        // tensors transparently. NumPy calls `tensor.__array__()` (with
        // an optional dtype argument for the "astype" case) whenever it
        // needs to coerce an unknown object into an ndarray.
        //
        // We delegate to the existing `.numpy()` path to produce a
        // contiguous host copy, then apply `astype(dtype)` at the end
        // if the caller requested a specific output dtype. Copy=None
        // is accepted for NumPy 2.0 protocol compatibility but we
        // always copy (our storage is not owned by NumPy).
        .def("__array__", [](const tenzor::Tensor& t,
                             py::object dtype,
                             py::object /*copy*/) -> py::object {
             tenzor::Tensor cpu_tensor;
             {
                 py::gil_scoped_release release;
                 cpu_tensor = tenzor::numpy::prepare_tensor_for_numpy(t);
             }
             py::object arr = tenzor::numpy::create_numpy_array(cpu_tensor, t.dtype());
             if (!dtype.is_none()) {
                 // Delegate the dtype cast to NumPy itself so we pick up
                 // every NumPy-recognized type specifier (str, dtype
                 // object, Python scalar type, etc.) without having to
                 // enumerate them here.
                 arr = arr.attr("astype")(dtype, py::arg("copy") = false);
             }
             return arr;
         },
         py::arg("dtype") = py::none(),
         py::arg("copy") = py::none(),
         "NumPy array protocol: called by np.asarray / np.array when "
         "they need to coerce this Tensor into an ndarray. Accepts an "
         "optional dtype for type promotion. Always materializes a "
         "host copy — Tenzor storage is not NumPy-owned.")
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
            int64_t slice_start = 0, slice_stop = 0, slice_step = 1;  // Audit J11: stepped slice for __setitem__

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
                // Audit J11: stepped slice is supported via Tensor::slice(dim,start,stop,step).
                slice_start = start; slice_stop = stop; slice_step = step;
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
                // 5th-audit B'3: Python ints are unbounded. py::cast<int64_t>
                // raises py::error_already_set on PyLong_AsLongLong overflow,
                // but the message ("Python int too large to convert to C long")
                // gives no scalar-assignment context. Wrap with an explicit
                // PyLong_AsLongLongAndOverflow check for a clearer diagnostic.
                {
                    int overflow = 0;
                    long long v = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
                    if (overflow != 0) {
                        throw std::overflow_error(
                            "Tensor scalar assignment: Python int does not fit "
                            "in int64 (overflow). Cast to a wider/narrower dtype "
                            "or use a tenzor.Tensor for the scalar value.");
                    }
                    if (v == -1 && PyErr_Occurred()) {
                        throw py::error_already_set();
                    }
                    int_scalar_value = static_cast<int64_t>(v);
                }
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
                    // Audit J11: pass slice_step (was hard-coded to 1).
                    target = self.slice(0, slice_start, slice_stop, slice_step);
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
                            // Audit J11: stepped slice supported via Tensor::slice's step arg.
                            target = target.slice(adjusted_dim, start, stop, entry.step);
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

    // DLPack consumer hook — accepts anything with `__dlpack__` (the
    // modern protocol) or a raw capsule named "dltensor" (legacy /
    // direct path). Matches the torch.from_dlpack / numpy.from_dlpack
    // signature.
    m.def("from_dlpack", [](py::object obj) -> tenzor::Tensor {
            py::capsule capsule;
            if (py::hasattr(obj, "__dlpack__")) {
                // Modern protocol: call the producer to obtain a capsule.
                // Pass stream=None — we don't have a compute stream to
                // synchronize on at the module boundary.
                py::object raw = obj.attr("__dlpack__")();
                capsule = raw.cast<py::capsule>();
            } else {
                // Legacy: the caller handed us the capsule directly.
                capsule = obj.cast<py::capsule>();
            }
            // Fetch the DLManagedTensor* before renaming so from_dlpack
            // can transfer ownership; from_dlpack's internal deleter
            // will call the producer's deleter.
            auto* managed = static_cast<DLManagedTensor*>(
                PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
            if (!managed) {
                throw std::runtime_error(
                    "from_dlpack: invalid capsule (expected name 'dltensor')");
            }
            // Rename the capsule so the original producer's destructor
            // won't call the DLPack deleter a second time when the
            // capsule is garbage-collected.
            PyCapsule_SetName(capsule.ptr(), "used_dltensor");
            return tenzor::from_dlpack(managed);
        },
        py::arg("obj"),
        "Zero-copy import from a DLPack producer (NumPy 2.0+, PyTorch, "
        "JAX, CuPy, TVM, or any object exposing __dlpack__). Returns a "
        "Tenzor tensor that shares memory with the original — the "
        "producer's storage is kept alive until the returned tensor is "
        "destroyed.");

    // Operations
    m.def("zeros", &tenzor::zeros, "Create tensor filled with zeros",
         py::arg("shape"),
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("ones", &tenzor::ones, "Create tensor filled with ones",
         py::arg("shape"),
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("randn",
         static_cast<tenzor::Tensor(*)(std::vector<int64_t>, tenzor::DType, tenzor::Device)>(&tenzor::randn),
         "Create tensor with random normal values",
         py::arg("shape"),
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("randint",
         static_cast<tenzor::Tensor(*)(int64_t, int64_t, std::vector<int64_t>, tenzor::DType, tenzor::Device)>(&tenzor::randint),
         "Create tensor with random integers",
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

    m.def("normal", [](const tenzor::Tensor& mean, const tenzor::Tensor& std) {
         return tenzor::normal(mean, std);
         }, "Sample from normal distribution N(mean, std^2)",
         py::arg("mean"), py::arg("std"),
         py::call_guard<py::gil_scoped_release>());

    m.def("poisson", [](const tenzor::Tensor& rates) {
         return tenzor::poisson(rates);
         }, "Sample from Poisson distribution with given rates",
         py::arg("rates"),
         py::call_guard<py::gil_scoped_release>());

    m.def("exponential", [](const tenzor::Tensor& rate) {
         return tenzor::exponential(rate);
         }, "Sample from exponential distribution with given rate",
         py::arg("rate"),
         py::call_guard<py::gil_scoped_release>());

    m.def("bernoulli", [](const tenzor::Tensor& probs) {
         return tenzor::bernoulli(probs);
         }, "Sample from Bernoulli distribution",
         py::arg("probs"),
         py::call_guard<py::gil_scoped_release>());

    m.def("multinomial", [](const tenzor::Tensor& input, int64_t num_samples, bool replacement) {
         return tenzor::multinomial(input, num_samples, replacement);
         }, "Weighted random sampling",
         py::arg("input"), py::arg("num_samples"), py::arg("replacement") = false,
         py::call_guard<py::gil_scoped_release>());

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

    m.def("randperm",
         static_cast<tenzor::Tensor(*)(int64_t, tenzor::Device)>(&tenzor::randperm),
         "Create random permutation of integers [0, n)",
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

        // Build the tensor on CPU first — writing to t.data<T>() directly
        // on a GPU device would dereference a device pointer from host code
        // and crash. After filling, transfer to the requested device.
        auto t = tenzor::empty(shape, actual_dtype, tenzor::Device::cpu());
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
            auto ft = tenzor::empty(shape, tenzor::DType::Float32, tenzor::Device::cpu());
            auto* ptr = ft.data<float>();
            for (size_t i = 0; i < values.size(); ++i)
                ptr[i] = static_cast<float>(values[i]);
            t = ft.to(actual_dtype);
        }
        if (device.type != tenzor::Device::Type::CPU) {
            t = t.to(device);
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

    // Generator class
    py::class_<tenzor::Generator>(m, "Generator")
        .def(py::init<tenzor::Device>(), py::arg("device") = tenzor::Device::cpu())
        .def("manual_seed", &tenzor::Generator::manual_seed, py::arg("seed"),
             py::return_value_policy::reference_internal)
        .def("seed", &tenzor::Generator::seed)
        .def("initial_seed", &tenzor::Generator::initial_seed)
        .def("device", &tenzor::Generator::device)
        .def("next_seed", &tenzor::Generator::next_seed);

    // Generator-aware random ops
    m.def("rand_with_generator",
         static_cast<tenzor::Tensor(*)(std::vector<int64_t>, tenzor::DType, tenzor::Device, tenzor::Generator&)>(&tenzor::rand),
         "Create tensor with random uniform values using a Generator",
         py::arg("shape"), py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu(), py::arg("generator"));
    m.def("randn_with_generator",
         static_cast<tenzor::Tensor(*)(std::vector<int64_t>, tenzor::DType, tenzor::Device, tenzor::Generator&)>(&tenzor::randn),
         "Create tensor with random normal values using a Generator",
         py::arg("shape"), py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu(), py::arg("generator"));

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

    m.def("addmm", [](const tenzor::Tensor& input, const tenzor::Tensor& mat1,
                       const tenzor::Tensor& mat2, double beta, double alpha) {
         return tenzor::addmm(input, mat1, mat2, beta, alpha);
         },
         R"doc(Fused multiply-add: beta * input + alpha * (mat1 @ mat2).

Uses a single BLAS GEMM call with alpha/beta for optimal performance.

Args:
    input: Bias tensor (M, N) or broadcastable
    mat1: Left matrix (M, K)
    mat2: Right matrix (K, N)
    beta: Scalar multiplier for input (default: 1.0)
    alpha: Scalar multiplier for mat1 @ mat2 (default: 1.0)

Returns:
    Result tensor (M, N)

Example::

    out = tz.addmm(bias, weight, x)  # bias + weight @ x
)doc",
         py::arg("input"), py::arg("mat1"), py::arg("mat2"),
         py::arg("beta") = 1.0, py::arg("alpha") = 1.0,
         py::call_guard<py::gil_scoped_release>());

    m.def("addmv", [](const tenzor::Tensor& input, const tenzor::Tensor& mat,
                       const tenzor::Tensor& vec, double beta, double alpha) {
         return tenzor::addmv(input, mat, vec, beta, alpha);
         },
         R"doc(Fused matrix-vector multiply-add: beta * input + alpha * (mat @ vec).

Uses a single BLAS GEMV call with alpha/beta for optimal performance.

Args:
    input: Bias vector (M,) or broadcastable
    mat: Matrix (M, K)
    vec: Vector (K,)
    beta: Scalar multiplier for input (default: 1.0)
    alpha: Scalar multiplier for mat @ vec (default: 1.0)

Returns:
    Result vector (M,)
)doc",
         py::arg("input"), py::arg("mat"), py::arg("vec"),
         py::arg("beta") = 1.0, py::arg("alpha") = 1.0,
         py::call_guard<py::gil_scoped_release>());

    m.def("baddbmm", [](const tenzor::Tensor& input, const tenzor::Tensor& batch1,
                         const tenzor::Tensor& batch2, double beta, double alpha) {
         return tenzor::baddbmm(input, batch1, batch2, beta, alpha);
         },
         R"doc(Batched fused multiply-add: beta * input + alpha * (batch1 @ batch2).

Uses batched BLAS GEMM with alpha/beta for optimal performance.

Args:
    input: Bias tensor (B, M, N) or broadcastable
    batch1: Left batch of matrices (B, M, K)
    batch2: Right batch of matrices (B, K, N)
    beta: Scalar multiplier for input (default: 1.0)
    alpha: Scalar multiplier for batch1 @ batch2 (default: 1.0)

Returns:
    Result tensor (B, M, N)
)doc",
         py::arg("input"), py::arg("batch1"), py::arg("batch2"),
         py::arg("beta") = 1.0, py::arg("alpha") = 1.0,
         py::call_guard<py::gil_scoped_release>());

    // Math operations - using lambda wrappers for overloaded functions
    // GIL released for compute-heavy operations
    m.def("exp", [](const tenzor::Tensor& t) { return tenzor::exp(t); },
         "Element-wise exponential", py::call_guard<py::gil_scoped_release>());
    m.def("log", [](const tenzor::Tensor& t) { return tenzor::log(t); },
         "Element-wise natural logarithm", py::call_guard<py::gil_scoped_release>());
    // Audit J.3: Variable overload — preserves autograd graph so callers
    // can compose log() into a loss without manually wiring a Function.
    m.def("log", [](const tenzor::Variable& v) { return tenzor::log(v); },
         "Element-wise natural logarithm (autograd-aware Variable overload)",
         py::arg("input"));
    m.def("sqrt", [](const tenzor::Tensor& t) { return tenzor::sqrt(t); },
         "Element-wise square root", py::call_guard<py::gil_scoped_release>());
    m.def("abs", [](const tenzor::Tensor& t) { return tenzor::abs(t); },
         "Element-wise absolute value", py::call_guard<py::gil_scoped_release>());
    m.def("pow", [](const tenzor::Tensor& input, double exponent) {
         return tenzor::pow(input, exponent);
         }, "Element-wise power",
         py::arg("input"), py::arg("exponent"), py::call_guard<py::gil_scoped_release>());
    // Audit J.3: Variable overload — autograd-aware pow(v, scalar). The
    // C++ overload lives in include/tenzor/autograd/ops.hpp.
    m.def("pow", [](const tenzor::Variable& input, double exponent) {
         return tenzor::pow(input, static_cast<float>(exponent));
         }, "Element-wise power (autograd-aware Variable overload)",
         py::arg("input"), py::arg("exponent"));
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
    m.def("ndtri", [](const tenzor::Tensor& p) { return tenzor::ndtri(p); },
         "Inverse normal CDF (probit function)", py::call_guard<py::gil_scoped_release>());
    m.def("gammainc", [](const tenzor::Tensor& a, const tenzor::Tensor& x) { return tenzor::gammainc(a, x); },
         "Lower incomplete gamma function (non-regularized)", py::call_guard<py::gil_scoped_release>());
    m.def("gammaincc", [](const tenzor::Tensor& a, const tenzor::Tensor& x) { return tenzor::gammaincc(a, x); },
         "Upper incomplete gamma function (non-regularized)", py::call_guard<py::gil_scoped_release>());

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
    // New Phase 4 ops
    m.def("frac", [](const tenzor::Tensor& t) { return tenzor::frac(t); },
         "Fractional part: x - floor(x)", py::call_guard<py::gil_scoped_release>());
    m.def("heaviside", [](const tenzor::Tensor& input, const tenzor::Tensor& values) {
         return tenzor::heaviside(input, values);
         }, "Heaviside step function", py::arg("input"), py::arg("values"),
         py::call_guard<py::gil_scoped_release>());
    m.def("nan_to_num", [](const tenzor::Tensor& t, double nan, double posinf, double neginf) {
         return tenzor::nan_to_num(t, nan, posinf, neginf);
         }, "Replace NaN/Inf with specified values",
         py::arg("input"), py::arg("nan") = 0.0,
         py::arg("posinf") = std::numeric_limits<double>::max(),
         py::arg("neginf") = std::numeric_limits<double>::lowest(),
         py::call_guard<py::gil_scoped_release>());
    m.def("isclose", [](const tenzor::Tensor& a, const tenzor::Tensor& b, double rtol, double atol) {
         return tenzor::isclose(a, b, rtol, atol);
         }, "Element-wise closeness check",
         py::arg("a"), py::arg("b"), py::arg("rtol") = 1e-5, py::arg("atol") = 1e-8,
         py::call_guard<py::gil_scoped_release>());
    m.def("allclose", [](const tenzor::Tensor& a, const tenzor::Tensor& b, double rtol, double atol) {
         return tenzor::allclose(a, b, rtol, atol);
         }, "Check if all elements are close",
         py::arg("a"), py::arg("b"), py::arg("rtol") = 1e-5, py::arg("atol") = 1e-8,
         py::call_guard<py::gil_scoped_release>());

    // Composed math operations
    m.def("diff", [](const tenzor::Tensor& input, int64_t n, int64_t dim) {
         return tenzor::diff(input, n, dim);
         }, "Finite differences along a dimension",
         py::arg("input"), py::arg("n") = 1, py::arg("dim") = -1,
         py::call_guard<py::gil_scoped_release>());
    m.def("logaddexp", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
         return tenzor::logaddexp(a, b);
         }, "Numerically stable log(exp(a) + exp(b))",
         py::arg("a"), py::arg("b"),
         py::call_guard<py::gil_scoped_release>());
    m.def("logaddexp2", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
         return tenzor::logaddexp2(a, b);
         }, "Numerically stable log2(2^a + 2^b)",
         py::arg("a"), py::arg("b"),
         py::call_guard<py::gil_scoped_release>());
    m.def("xlogy", [](const tenzor::Tensor& x, const tenzor::Tensor& y) {
         return tenzor::xlogy(x, y);
         }, "x * log(y) with 0 * log(y) = 0",
         py::arg("x"), py::arg("y"),
         py::call_guard<py::gil_scoped_release>());

    // Statistical operations (cov, corrcoef)
    m.def("cov", [](const tenzor::Tensor& input, int64_t correction) {
         return tenzor::cov(input, correction);
         }, "Compute sample covariance matrix",
         py::arg("input"), py::arg("correction") = 1,
         py::call_guard<py::gil_scoped_release>());
    m.def("corrcoef", [](const tenzor::Tensor& input) {
         return tenzor::corrcoef(input);
         }, "Compute Pearson correlation coefficient matrix",
         py::arg("input"),
         py::call_guard<py::gil_scoped_release>());

    // Phase 5: Extended math ops
    m.def("deg2rad", [](const tenzor::Tensor& t) { return tenzor::deg2rad(t); },
         "Convert degrees to radians", py::arg("input"), py::call_guard<py::gil_scoped_release>());
    m.def("rad2deg", [](const tenzor::Tensor& t) { return tenzor::rad2deg(t); },
         "Convert radians to degrees", py::arg("input"), py::call_guard<py::gil_scoped_release>());
    m.def("logit", [](const tenzor::Tensor& t, double eps) { return tenzor::logit(t, eps); },
         "Logit function: log(x / (1-x))", py::arg("input"), py::arg("eps") = -1.0,
         py::call_guard<py::gil_scoped_release>());
    m.def("signbit", [](const tenzor::Tensor& t) { return tenzor::signbit(t); },
         "Test sign bit", py::arg("input"), py::call_guard<py::gil_scoped_release>());
    m.def("float_power", [](const tenzor::Tensor& a, const tenzor::Tensor& b) {
         return tenzor::float_power(a, b); },
         "Power with Float64 promotion", py::arg("base"), py::arg("exponent"),
         py::call_guard<py::gil_scoped_release>());
    m.def("xlog1py", [](const tenzor::Tensor& x, const tenzor::Tensor& y) {
         return tenzor::xlog1py(x, y); },
         "x * log1p(y) with 0 * log1p(y) = 0", py::arg("x"), py::arg("y"),
         py::call_guard<py::gil_scoped_release>());
    m.def("ldexp", [](const tenzor::Tensor& x, const tenzor::Tensor& n) {
         return tenzor::ldexp(x, n); },
         "ldexp(x, n) = x * 2^n", py::arg("x"), py::arg("n"),
         py::call_guard<py::gil_scoped_release>());
    m.def("isreal", [](const tenzor::Tensor& t) { return tenzor::isreal(t); },
         "Test if elements are real", py::arg("input"), py::call_guard<py::gil_scoped_release>());
    m.def("isposinf", [](const tenzor::Tensor& t) { return tenzor::isposinf(t); },
         "Test for positive infinity", py::arg("input"), py::call_guard<py::gil_scoped_release>());
    m.def("isneginf", [](const tenzor::Tensor& t) { return tenzor::isneginf(t); },
         "Test for negative infinity", py::arg("input"), py::call_guard<py::gil_scoped_release>());

    m.def("frexp", [](const tenzor::Tensor& t) {
         auto [m, e] = tenzor::frexp(t);
         return py::make_tuple(m, e);
         }, "Decompose into mantissa and exponent", py::arg("input"),
         py::call_guard<py::gil_scoped_release>());

    // Phase 4: Missing ops bindings
    m.def("nanvar", [](const tenzor::Tensor& t, std::optional<int64_t> dim, bool keepdim, int64_t correction) {
         return tenzor::nanvar(t, dim, keepdim, correction); },
         "Variance ignoring NaN", py::arg("input"), py::arg("dim") = py::none(),
         py::arg("keepdim") = false, py::arg("correction") = 1,
         py::call_guard<py::gil_scoped_release>());
    m.def("nanstd", [](const tenzor::Tensor& t, std::optional<int64_t> dim, bool keepdim, int64_t correction) {
         return tenzor::nanstd(t, dim, keepdim, correction); },
         "Standard deviation ignoring NaN", py::arg("input"), py::arg("dim") = py::none(),
         py::arg("keepdim") = false, py::arg("correction") = 1,
         py::call_guard<py::gil_scoped_release>());
    m.def("pixel_shuffle", [](const tenzor::Tensor& t, int64_t r) {
         return tenzor::pixel_shuffle(t, r); },
         "Rearrange (C*r^2, H, W) to (C, H*r, W*r)", py::arg("input"), py::arg("upscale_factor"),
         py::call_guard<py::gil_scoped_release>());
    m.def("pixel_unshuffle", [](const tenzor::Tensor& t, int64_t r) {
         return tenzor::pixel_unshuffle(t, r); },
         "Rearrange (C, H*r, W*r) to (C*r^2, H, W)", py::arg("input"), py::arg("downscale_factor"),
         py::call_guard<py::gil_scoped_release>());

    m.def("channel_shuffle", [](const tenzor::Tensor& t, int64_t groups) {
         return tenzor::channel_shuffle(t, groups); },
         "Shuffle channels across groups for ShuffleNet", py::arg("input"), py::arg("groups"),
         py::call_guard<py::gil_scoped_release>());

    // --- Gap-fill: distance functions ---

    m.def("pairwise_distance", [](const tenzor::Tensor& x1, const tenzor::Tensor& x2, double p) {
         return tenzor::pairwise_distance(x1, x2, p); },
         "Pairwise p-norm distance between corresponding rows",
         py::arg("x1"), py::arg("x2"), py::arg("p") = 2.0,
         py::call_guard<py::gil_scoped_release>());
    m.def("pdist", [](const tenzor::Tensor& input, double p) {
         return tenzor::pdist(input, p); },
         "All-pairs p-norm distances between rows of a matrix",
         py::arg("input"), py::arg("p") = 2.0,
         py::call_guard<py::gil_scoped_release>());

    m.def("cov", [](const tenzor::Tensor& t, int64_t correction) {
         return tenzor::cov(t, correction); },
         "Covariance matrix", py::arg("input"), py::arg("correction") = 1,
         py::call_guard<py::gil_scoped_release>());

    m.def("corrcoef", [](const tenzor::Tensor& t) {
         return tenzor::corrcoef(t); },
         "Pearson correlation coefficient matrix", py::arg("input"),
         py::call_guard<py::gil_scoped_release>());

    m.def("tensordot", [](const tenzor::Tensor& a, const tenzor::Tensor& b,
                          std::vector<int64_t> dims_a, std::vector<int64_t> dims_b) {
         return tenzor::tensordot(a, b, std::move(dims_a), std::move(dims_b));
         }, "Generalized tensor contraction with explicit dim lists",
         py::arg("a"), py::arg("b"), py::arg("dims_a"), py::arg("dims_b"),
         py::call_guard<py::gil_scoped_release>());
    m.def("tensordot", [](const tenzor::Tensor& a, const tenzor::Tensor& b, int64_t dims) {
         return tenzor::tensordot(a, b, dims);
         }, "Generalized tensor contraction (contract last N of a with first N of b)",
         py::arg("a"), py::arg("b"), py::arg("dims") = 2,
         py::call_guard<py::gil_scoped_release>());

    m.def("movedim", [](const tenzor::Tensor& t, std::vector<int64_t> src, std::vector<int64_t> dst) {
         return tenzor::movedim(t, src, dst);
         }, "Move dimensions to new positions",
         py::arg("input"), py::arg("source"), py::arg("destination"),
         py::call_guard<py::gil_scoped_release>());
    m.def("swapaxes", [](const tenzor::Tensor& t, int64_t dim0, int64_t dim1) {
         return tenzor::swapaxes(t, dim0, dim1);
         }, "Swap two dimensions", py::arg("input"), py::arg("dim0"), py::arg("dim1"),
         py::call_guard<py::gil_scoped_release>());
    m.def("moveaxis", [](const tenzor::Tensor& t, std::vector<int64_t> src, std::vector<int64_t> dst) {
         return tenzor::moveaxis(t, src, dst);
         }, "Move dimensions to new positions (alias for movedim)",
         py::arg("input"), py::arg("source"), py::arg("destination"),
         py::call_guard<py::gil_scoped_release>());
    m.def("narrow_copy", [](const tenzor::Tensor& t, int64_t dim, int64_t start, int64_t length) {
         return tenzor::narrow_copy(t, dim, start, length);
         }, "Narrow with copy semantics",
         py::arg("input"), py::arg("dim"), py::arg("start"), py::arg("length"),
         py::call_guard<py::gil_scoped_release>());
    m.def("column_stack", [](const std::vector<tenzor::Tensor>& tensors) {
         return tenzor::column_stack(tensors);
         }, "Stack 1D tensors as columns, 2D+ cat along dim 1",
         py::arg("tensors"), py::call_guard<py::gil_scoped_release>());
    m.def("row_stack", [](const std::vector<tenzor::Tensor>& tensors) {
         return tenzor::row_stack(tensors);
         }, "Alias for vstack",
         py::arg("tensors"), py::call_guard<py::gil_scoped_release>());

    // --- Phase 1 gap-fill: stacking / splitting ops ---

    m.def("hstack", [](const std::vector<tenzor::Tensor>& tensors) {
         return tenzor::hstack(tensors);
         }, "Horizontal stack: cat along dim 1 (dim 0 for 1D)",
         py::arg("tensors"), py::call_guard<py::gil_scoped_release>());
    m.def("vstack", [](const std::vector<tenzor::Tensor>& tensors) {
         return tenzor::vstack(tensors);
         }, "Vertical stack: cat along dim 0",
         py::arg("tensors"), py::call_guard<py::gil_scoped_release>());
    m.def("dstack", [](const std::vector<tenzor::Tensor>& tensors) {
         return tenzor::dstack(tensors);
         }, "Depth stack: cat along dim 2",
         py::arg("tensors"), py::call_guard<py::gil_scoped_release>());

    m.def("tensor_split", [](const tenzor::Tensor& t, int64_t sections, int64_t dim) {
         return tenzor::tensor_split(t, sections, dim);
         }, "Split tensor into equal sections along dim",
         py::arg("input"), py::arg("sections"), py::arg("dim") = 0,
         py::call_guard<py::gil_scoped_release>());
    m.def("tensor_split", [](const tenzor::Tensor& t, std::vector<int64_t> indices, int64_t dim) {
         return tenzor::tensor_split(t, indices, dim);
         }, "Split tensor at given indices along dim",
         py::arg("input"), py::arg("indices"), py::arg("dim") = 0,
         py::call_guard<py::gil_scoped_release>());

    m.def("hsplit", [](const tenzor::Tensor& t, int64_t sections) {
         return tenzor::hsplit(t, sections);
         }, "Horizontal split",
         py::arg("input"), py::arg("sections"),
         py::call_guard<py::gil_scoped_release>());
    m.def("vsplit", [](const tenzor::Tensor& t, int64_t sections) {
         return tenzor::vsplit(t, sections);
         }, "Vertical split",
         py::arg("input"), py::arg("sections"),
         py::call_guard<py::gil_scoped_release>());
    m.def("dsplit", [](const tenzor::Tensor& t, int64_t sections) {
         return tenzor::dsplit(t, sections);
         }, "Depth split (requires ndim >= 3)",
         py::arg("input"), py::arg("sections"),
         py::call_guard<py::gil_scoped_release>());
    m.def("broadcast_tensors", [](const std::vector<tenzor::Tensor>& tensors) {
         return tenzor::broadcast_tensors(tensors);
         }, "Broadcast all tensors to a common shape",
         py::arg("tensors"), py::call_guard<py::gil_scoped_release>());
    m.def("logspace", [](float start, float end, int64_t steps, double base,
                          tenzor::DType dtype, tenzor::Device device) {
         return tenzor::logspace(start, end, steps, base, dtype, device);
         }, "Logarithmically spaced values",
         py::arg("start"), py::arg("end"), py::arg("steps"), py::arg("base") = 10.0,
         py::arg("dtype") = tenzor::DType::Float32, py::arg("device") = tenzor::Device::cpu(),
         py::call_guard<py::gil_scoped_release>());
    m.def("count_nonzero", [](const tenzor::Tensor& t, std::optional<int64_t> dim) {
         return tenzor::count_nonzero(t, dim);
         }, "Count nonzero elements", py::arg("input"), py::arg("dim") = std::nullopt,
         py::call_guard<py::gil_scoped_release>());
    m.def("nansum", [](const tenzor::Tensor& t, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::nansum(t, dim, keepdim);
         }, "Sum ignoring NaN", py::arg("input"), py::arg("dim") = std::nullopt,
         py::arg("keepdim") = false, py::call_guard<py::gil_scoped_release>());
    m.def("nanmean", [](const tenzor::Tensor& t, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::nanmean(t, dim, keepdim);
         }, "Mean ignoring NaN", py::arg("input"), py::arg("dim") = std::nullopt,
         py::arg("keepdim") = false, py::call_guard<py::gil_scoped_release>());
    m.def("aminmax", [](const tenzor::Tensor& t, std::optional<int64_t> dim, bool keepdim) {
         return tenzor::aminmax(t, dim, keepdim);
         }, "Simultaneous min and max", py::arg("input"), py::arg("dim") = std::nullopt,
         py::arg("keepdim") = false, py::call_guard<py::gil_scoped_release>());
    m.def("histogramdd", [](const tenzor::Tensor& t, std::vector<int64_t> bins,
                             std::optional<std::vector<std::pair<double,double>>> ranges,
                             bool density) {
         return tenzor::histogramdd(t, bins, ranges, density);
         }, "Multi-dimensional histogram",
         py::arg("input"), py::arg("bins"),
         py::arg("ranges") = std::nullopt, py::arg("density") = false,
         py::call_guard<py::gil_scoped_release>());
    m.def("index_add", [](const tenzor::Tensor& t, int64_t dim, const tenzor::Tensor& idx, const tenzor::Tensor& src) {
         return tenzor::index_add(t, dim, idx, src);
         }, "Accumulate at index positions", py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("source"),
         py::call_guard<py::gil_scoped_release>());
    m.def("index_copy", [](const tenzor::Tensor& t, int64_t dim, const tenzor::Tensor& idx, const tenzor::Tensor& src) {
         return tenzor::index_copy(t, dim, idx, src);
         }, "Copy at index positions", py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("source"),
         py::call_guard<py::gil_scoped_release>());
    m.def("index_fill", [](const tenzor::Tensor& t, int64_t dim, const tenzor::Tensor& idx, float value) {
         return tenzor::index_fill(t, dim, idx, value);
         }, "Fill at index positions", py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("value"),
         py::call_guard<py::gil_scoped_release>());
    m.def("select_scatter", [](const tenzor::Tensor& input, const tenzor::Tensor& src, int64_t dim, int64_t index) {
         return tenzor::select_scatter(input, src, dim, index);
         }, "Return copy of input with src at select(dim, index)",
         py::arg("input"), py::arg("src"), py::arg("dim"), py::arg("index"),
         py::call_guard<py::gil_scoped_release>());
    m.def("slice_scatter", [](const tenzor::Tensor& input, const tenzor::Tensor& src, int64_t dim,
                               int64_t start, int64_t end, int64_t step) {
         return tenzor::slice_scatter(input, src, dim, start, end, step);
         }, "Return copy of input with src at slice(dim, start, end, step)",
         py::arg("input"), py::arg("src"), py::arg("dim") = 0,
         py::arg("start") = 0, py::arg("end") = -1, py::arg("step") = 1,
         py::call_guard<py::gil_scoped_release>());
    m.def("diagonal_scatter", [](const tenzor::Tensor& input, const tenzor::Tensor& src,
                                  int64_t offset, int64_t dim1, int64_t dim2) {
         return tenzor::diagonal_scatter(input, src, offset, dim1, dim2);
         }, "Return copy of input with src placed along the diagonal",
         py::arg("input"), py::arg("src"), py::arg("offset") = 0,
         py::arg("dim1") = 0, py::arg("dim2") = 1,
         py::call_guard<py::gil_scoped_release>());
    // Bitwise ops
    m.def("bitwise_and", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::bitwise_and(a, b); },
         "Bitwise AND", py::arg("a"), py::arg("b"), py::call_guard<py::gil_scoped_release>());
    m.def("bitwise_or", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::bitwise_or(a, b); },
         "Bitwise OR", py::arg("a"), py::arg("b"), py::call_guard<py::gil_scoped_release>());
    m.def("bitwise_xor", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::bitwise_xor(a, b); },
         "Bitwise XOR", py::arg("a"), py::arg("b"), py::call_guard<py::gil_scoped_release>());
    m.def("bitwise_not", [](const tenzor::Tensor& t) { return tenzor::bitwise_not(t); },
         "Bitwise NOT", py::call_guard<py::gil_scoped_release>());
    m.def("bitwise_left_shift", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::bitwise_left_shift(a, b); },
         "Bitwise left shift", py::arg("input"), py::arg("shift"), py::call_guard<py::gil_scoped_release>());
    m.def("bitwise_right_shift", [](const tenzor::Tensor& a, const tenzor::Tensor& b) { return tenzor::bitwise_right_shift(a, b); },
         "Bitwise right shift", py::arg("input"), py::arg("shift"), py::call_guard<py::gil_scoped_release>());

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
    // Window functions
    m.def("hann_window", [](int64_t size, bool periodic, tenzor::DType dtype, tenzor::Device device) {
         return tenzor::hann_window(size, periodic, dtype, device);
         }, "Hann (raised cosine) window",
         py::arg("size"), py::arg("periodic") = true,
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("hamming_window", [](int64_t size, bool periodic, double alpha, double beta,
                                tenzor::DType dtype, tenzor::Device device) {
         return tenzor::hamming_window(size, periodic, alpha, beta, dtype, device);
         }, "Hamming window",
         py::arg("size"), py::arg("periodic") = true,
         py::arg("alpha") = 0.54, py::arg("beta") = 0.46,
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    m.def("blackman_window", [](int64_t size, bool periodic, tenzor::DType dtype, tenzor::Device device) {
         return tenzor::blackman_window(size, periodic, dtype, device);
         }, "Blackman window",
         py::arg("size"), py::arg("periodic") = true,
         py::arg("dtype") = tenzor::DType::Float32,
         py::arg("device") = tenzor::Device::cpu());

    // Unique consecutive
    m.def("unique_consecutive", [](const tenzor::Tensor& input, bool return_inverse, bool return_counts) {
         return tenzor::unique_consecutive(input, return_inverse, return_counts);
         }, "Deduplicate consecutive equal elements",
         py::arg("input"), py::arg("return_inverse") = false, py::arg("return_counts") = false,
         py::call_guard<py::gil_scoped_release>());

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

    m.def("view_as_real",
         static_cast<tenzor::Tensor(*)(const tenzor::Tensor&)>(&tenzor::view_as_real),
         "View a complex tensor as real with trailing dim 2",
         py::arg("input"),
         py::call_guard<py::gil_scoped_release>());
    m.def("view_as_complex",
         static_cast<tenzor::Tensor(*)(const tenzor::Tensor&)>(&tenzor::view_as_complex),
         "View a real tensor with trailing dim 2 as complex",
         py::arg("input"),
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
    m.def("scatter_reduce", [](const tenzor::Tensor& input, int64_t dim, const tenzor::Tensor& index,
                                const tenzor::Tensor& src, const std::string& reduce, bool include_self) {
         return tenzor::scatter_reduce(input, dim, index, src, reduce, include_self);
         }, "Scatter with reduction (sum/prod/mean/amax/amin)",
         py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("src"),
         py::arg("reduce"), py::arg("include_self") = true,
         py::call_guard<py::gil_scoped_release>());
    m.def("masked_select",
         static_cast<tenzor::Tensor (*)(const tenzor::Tensor&, const tenzor::Tensor&)>(
             &tenzor::masked_select),
         "Select elements where mask is true",
         py::arg("input"), py::arg("mask"));
    m.def("masked_fill",
         static_cast<tenzor::Tensor (*)(const tenzor::Tensor&, const tenzor::Tensor&, float)>(
             &tenzor::masked_fill),
         "Fill elements with value where mask is true",
         py::arg("input"), py::arg("mask"), py::arg("value"));
    m.def("where", [](const tenzor::Tensor& condition, const tenzor::Tensor& x, const tenzor::Tensor& y) {
         return tenzor::where(condition, x, y);
         }, "Conditional element selection",
         py::arg("condition"), py::arg("x"), py::arg("y"));
    m.def("take", &tenzor::take, "Take elements from flattened tensor",
         py::arg("input"), py::arg("index"));
    m.def("put", &tenzor::put, "Put elements into flattened tensor",
         py::arg("input"), py::arg("index"), py::arg("source"));
    m.def("one_hot",
         static_cast<tenzor::Tensor (*)(const tenzor::Tensor&, int64_t)>(
             &tenzor::one_hot),
         "One-hot encode class indices",
         py::arg("input"), py::arg("num_classes") = -1,
         py::call_guard<py::gil_scoped_release>());

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
    m.def("repeat_interleave", [](const tenzor::Tensor& input, int64_t repeats, std::optional<int64_t> dim) {
        return tenzor::repeat_interleave(input, repeats, dim);
    }, "Repeat each element of tensor a given number of times along a dimension",
         py::arg("input"), py::arg("repeats"), py::arg("dim") = py::none(),
         py::call_guard<py::gil_scoped_release>());
    m.def("repeat_interleave", [](const tenzor::Tensor& input, const tenzor::Tensor& repeats, std::optional<int64_t> dim) {
        return tenzor::repeat_interleave(input, repeats, dim);
    }, "Repeat each element of tensor by per-element counts along a dimension",
         py::arg("input"), py::arg("repeats"), py::arg("dim") = py::none(),
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
        // Shape / numel exposed as properties to match Tensor's API surface;
        // callers can treat Variable.shape like Tensor.shape (a list).
        .def_property_readonly("shape", [](const tenzor::Variable& self) {
            auto s = self.tensor().shape();
            return std::vector<int64_t>(s.begin(), s.end());
        }, "Shape of the underlying tensor")
        .def_property_readonly("numel", [](const tenzor::Variable& self) {
            return self.tensor().numel();
        }, "Number of elements in the underlying tensor")
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
        // Power — uses the autograd-aware ``tenzor::pow(Variable, float)``
        // overload from include/tenzor/autograd/ops.hpp so the resulting
        // Variable retains its grad_fn (audit J.3: previously dropped the
        // graph by re-wrapping a Tensor with requires_grad=false).
        .def("__pow__", [](const tenzor::Variable& a, float exp) {
            return tenzor::pow(a, exp);
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
    // Activation offloading (Phase 1.6) — when enabled, save_for_backward
    // moves saved GPU tensors to CPU so they don't sit in device memory
    // for the duration of the forward pass. The autograd engine reloads
    // them to the original device before invoking each backward kernel.
    // ========================================================================
    m.def("set_activation_offload", &tenzor::set_activation_offload,
          "Enable/disable activation offloading for saved tensors.\n\n"
          "When enabled, Function::save_for_backward() moves GPU tensors\n"
          "to CPU, reducing peak device memory usage during the forward\n"
          "pass. The autograd engine reloads them to the source device\n"
          "before invoking each backward kernel.",
          py::arg("enabled"));

    m.def("is_activation_offload_enabled", &tenzor::activation_offload_enabled,
          "Return whether activation offloading is currently enabled.");

    struct PyOffloadActivationsContext {
        bool prev_state_ = false;

        void enter() {
            prev_state_ = tenzor::activation_offload_enabled();
            tenzor::set_activation_offload(true);
        }

        void exit() {
            tenzor::set_activation_offload(prev_state_);
        }
    };

    py::class_<PyOffloadActivationsContext>(m, "offload_activations",
        "Context manager and decorator that enables activation offloading\n"
        "for the duration of a forward + backward pass.\n\n"
        "Usage as context manager:\n"
        "    with tz.offload_activations():\n"
        "        y = model(x)\n"
        "        loss = y.sum()\n"
        "        loss.backward()\n\n"
        "Usage as decorator:\n"
        "    @tz.offload_activations()\n"
        "    def train_step(x):\n"
        "        ...")
        .def(py::init<>())
        .def("__enter__", [](PyOffloadActivationsContext& self) -> PyOffloadActivationsContext& {
            self.enter();
            return self;
        })
        .def("__exit__", [](PyOffloadActivationsContext& self, py::object, py::object, py::object) {
            self.exit();
            return false;
        })
        .def("__call__", [](PyOffloadActivationsContext&, py::function func) -> py::object {
            auto wrapper = py::cpp_function([func](py::args args, py::kwargs kwargs) -> py::object {
                const bool prev = tenzor::activation_offload_enabled();
                tenzor::set_activation_offload(true);
                try {
                    py::object result = func(*args, **kwargs);
                    tenzor::set_activation_offload(prev);
                    return result;
                } catch (...) {
                    tenzor::set_activation_offload(prev);
                    throw;
                }
            });
            try {
                py::module_ functools = py::module_::import("functools");
                functools.attr("update_wrapper")(wrapper, func);
            } catch (py::error_already_set&) {
                PyErr_Clear();
            }
            return wrapper;
        }, py::arg("func"));

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
    // Distributions submodule
    // ========================================================================
    auto dist_m = m.def_submodule("distributions", "Probability distributions");

    py::class_<tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Distribution>>(dist_m, "Distribution")
        .def("sample", [](tenzor::distributions::Distribution& self,
                           std::vector<int64_t> sample_shape) {
            return self.sample(std::move(sample_shape));
        }, py::arg("sample_shape") = std::vector<int64_t>{},
             "Draw samples from the distribution")
        .def("rsample", [](tenzor::distributions::Distribution& self,
                            std::vector<int64_t> sample_shape) {
            return self.rsample(std::move(sample_shape));
        }, py::arg("sample_shape") = std::vector<int64_t>{},
             "Draw reparameterized samples")
        .def("log_prob", &tenzor::distributions::Distribution::log_prob,
             py::arg("value"), "Compute log probability of a value")
        .def("entropy", &tenzor::distributions::Distribution::entropy,
             "Compute entropy of the distribution")
        .def("mean", &tenzor::distributions::Distribution::mean,
             "Distribution mean")
        .def("variance", &tenzor::distributions::Distribution::variance,
             "Distribution variance");

    py::class_<tenzor::distributions::Normal,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Normal>>(dist_m, "Normal")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("loc"), py::arg("scale"),
             "Normal distribution parameterized by mean (loc) and std (scale)");

    py::class_<tenzor::distributions::Uniform,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Uniform>>(dist_m, "Uniform")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("low"), py::arg("high"),
             "Uniform distribution on [low, high)");

    py::class_<tenzor::distributions::Categorical,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Categorical>>(dist_m, "Categorical")
        .def(py::init<tenzor::Tensor>(), py::arg("probs"),
             "Categorical distribution parameterized by probabilities")
        .def_static("from_logits", &tenzor::distributions::Categorical::from_logits,
                     py::arg("logits"),
                     "Create Categorical from unnormalized log-probabilities");

    py::class_<tenzor::distributions::Exponential,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Exponential>>(dist_m, "Exponential")
        .def(py::init<tenzor::Tensor>(), py::arg("rate"),
             "Exponential distribution parameterized by rate (lambda)");

    py::class_<tenzor::distributions::Laplace,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Laplace>>(dist_m, "Laplace")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("loc"), py::arg("scale"),
             "Laplace distribution parameterized by location and scale");

    py::class_<tenzor::distributions::BernoulliDist,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::BernoulliDist>>(dist_m, "Bernoulli")
        .def(py::init<tenzor::Tensor>(), py::arg("probs"),
             "Bernoulli distribution parameterized by probability");

    py::class_<tenzor::distributions::Gamma,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Gamma>>(dist_m, "Gamma")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("concentration"), py::arg("rate"),
             "Gamma distribution (shape-rate parameterization)");

    py::class_<tenzor::distributions::Beta,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Beta>>(dist_m, "Beta")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("concentration1"), py::arg("concentration0"),
             "Beta distribution on (0, 1)");

    py::class_<tenzor::distributions::Dirichlet,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Dirichlet>>(dist_m, "Dirichlet")
        .def(py::init<tenzor::Tensor>(), py::arg("concentration"),
             "Dirichlet distribution over the simplex");

    py::class_<tenzor::distributions::StudentT,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::StudentT>>(dist_m, "StudentT")
        .def(py::init<tenzor::Tensor, tenzor::Tensor, tenzor::Tensor>(),
             py::arg("df"), py::arg("loc"), py::arg("scale"),
             "Student-t distribution")
        .def(py::init<tenzor::Tensor>(), py::arg("df"),
             "Student-t distribution with loc=0, scale=1");

    py::class_<tenzor::distributions::Poisson,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Poisson>>(dist_m, "Poisson")
        .def(py::init<tenzor::Tensor>(), py::arg("rate"),
             "Poisson distribution parameterized by rate (lambda)");

    py::class_<tenzor::distributions::MultivariateNormal,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::MultivariateNormal>>(dist_m, "MultivariateNormal")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("loc"), py::arg("covariance_matrix"),
             "Multivariate Normal distribution");

    // New distributions (Phase 7)

    py::class_<tenzor::distributions::Binomial,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Binomial>>(dist_m, "Binomial")
        .def(py::init<int64_t, tenzor::Tensor>(),
             py::arg("total_count"), py::arg("probs"),
             "Binomial distribution: number of successes in n independent Bernoulli trials");

    py::class_<tenzor::distributions::LogNormal,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::LogNormal>>(dist_m, "LogNormal")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("loc"), py::arg("scale"),
             "Log-Normal distribution: exp(Normal(loc, scale))");

    py::class_<tenzor::distributions::Cauchy,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Cauchy>>(dist_m, "Cauchy")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("loc"), py::arg("scale"),
             "Cauchy distribution (heavy-tailed, no defined mean/variance)");

    py::class_<tenzor::distributions::Chi2,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Chi2>>(dist_m, "Chi2")
        .def(py::init<tenzor::Tensor>(), py::arg("df"),
             "Chi-squared distribution with df degrees of freedom");

    py::class_<tenzor::distributions::Geometric,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Geometric>>(dist_m, "Geometric")
        .def(py::init<tenzor::Tensor>(), py::arg("probs"),
             "Geometric distribution: trials until first success (1-indexed)");

    py::class_<tenzor::distributions::Gumbel,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Gumbel>>(dist_m, "Gumbel")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("loc"), py::arg("scale"),
             "Gumbel (Type-I extreme value) distribution");

    // ---- Audit E.4: 16 additional distributions ------------------------------

    py::class_<tenzor::distributions::HalfNormal,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::HalfNormal>>(dist_m, "HalfNormal")
        .def(py::init<tenzor::Tensor>(), py::arg("scale"),
             "HalfNormal distribution: |Normal(0, scale)|");

    py::class_<tenzor::distributions::HalfCauchy,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::HalfCauchy>>(dist_m, "HalfCauchy")
        .def(py::init<tenzor::Tensor>(), py::arg("scale"),
             "HalfCauchy distribution: |Cauchy(0, scale)|");

    py::class_<tenzor::distributions::FisherSnedecor,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::FisherSnedecor>>(dist_m, "FisherSnedecor")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("df1"), py::arg("df2"),
             "Fisher-Snedecor (F) distribution with df1 and df2 degrees of freedom");

    py::class_<tenzor::distributions::NegativeBinomial,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::NegativeBinomial>>(dist_m, "NegativeBinomial")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("total_count"), py::arg("probs"),
             "Negative Binomial distribution parameterized by total_count and probs")
        .def("cdf", &tenzor::distributions::NegativeBinomial::cdf, py::arg("value"),
             "Cumulative distribution function")
        .def("icdf", &tenzor::distributions::NegativeBinomial::icdf, py::arg("q"),
             "Inverse CDF (quantile function)");

    py::class_<tenzor::distributions::VonMises,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::VonMises>>(dist_m, "VonMises")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("loc"), py::arg("concentration"),
             "Von Mises distribution on the circle, parameterized by loc and concentration")
        .def("cdf", &tenzor::distributions::VonMises::cdf, py::arg("value"),
             "Cumulative distribution function")
        .def("icdf", &tenzor::distributions::VonMises::icdf, py::arg("q"),
             "Inverse CDF (quantile function)");

    py::class_<tenzor::distributions::RelaxedBernoulli,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::RelaxedBernoulli>>(dist_m, "RelaxedBernoulli")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("temperature"), py::arg("probs"),
             "Relaxed Bernoulli (Gumbel-Sigmoid) distribution")
        .def_static("from_logits", &tenzor::distributions::RelaxedBernoulli::from_logits,
                     py::arg("temperature"), py::arg("logits"),
                     "Create RelaxedBernoulli from unnormalized log-probabilities");

    py::class_<tenzor::distributions::RelaxedOneHotCategorical,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::RelaxedOneHotCategorical>>(
                   dist_m, "RelaxedOneHotCategorical")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("temperature"), py::arg("probs"),
             "Relaxed one-hot Categorical (Gumbel-Softmax) distribution")
        .def_static("from_logits",
                     &tenzor::distributions::RelaxedOneHotCategorical::from_logits,
                     py::arg("temperature"), py::arg("logits"),
                     "Create RelaxedOneHotCategorical from unnormalized log-probabilities");

    py::class_<tenzor::distributions::Wishart,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Wishart>>(dist_m, "Wishart")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("df"), py::arg("scale_tril"),
             "Wishart distribution over positive-definite matrices");

    py::class_<tenzor::distributions::Pareto,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Pareto>>(dist_m, "Pareto")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("scale"), py::arg("alpha"),
             "Pareto distribution (Type I) with scale and shape parameter alpha")
        .def("cdf", &tenzor::distributions::Pareto::cdf, py::arg("value"),
             "Cumulative distribution function");

    py::class_<tenzor::distributions::Weibull,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Weibull>>(dist_m, "Weibull")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("scale"), py::arg("concentration"),
             "Weibull distribution with scale and concentration (shape) parameters")
        .def("cdf", &tenzor::distributions::Weibull::cdf, py::arg("value"),
             "Cumulative distribution function");

    py::class_<tenzor::distributions::Kumaraswamy,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Kumaraswamy>>(dist_m, "Kumaraswamy")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("concentration1"), py::arg("concentration0"),
             "Kumaraswamy distribution on (0, 1)")
        .def("cdf", &tenzor::distributions::Kumaraswamy::cdf, py::arg("value"),
             "Cumulative distribution function");

    py::class_<tenzor::distributions::ContinuousBernoulli,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::ContinuousBernoulli>>(
                   dist_m, "ContinuousBernoulli")
        .def(py::init<tenzor::Tensor>(), py::arg("probs"),
             "Continuous Bernoulli distribution on [0, 1]");

    py::class_<tenzor::distributions::OneHotCategorical,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::OneHotCategorical>>(
                   dist_m, "OneHotCategorical")
        .def(py::init<tenzor::Tensor>(), py::arg("probs"),
             "One-hot Categorical distribution parameterized by probabilities");

    py::class_<tenzor::distributions::LogisticNormal,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::LogisticNormal>>(
                   dist_m, "LogisticNormal")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("loc"), py::arg("scale"),
             "Logistic-Normal distribution over the simplex");

    py::class_<tenzor::distributions::LowRankMultivariateNormal,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::LowRankMultivariateNormal>>(
                   dist_m, "LowRankMultivariateNormal")
        .def(py::init<tenzor::Tensor, tenzor::Tensor, tenzor::Tensor>(),
             py::arg("loc"), py::arg("cov_factor"), py::arg("cov_diag"),
             "Multivariate Normal with low-rank-plus-diagonal covariance");

    py::class_<tenzor::distributions::LKJCholesky,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::LKJCholesky>>(dist_m, "LKJCholesky")
        .def(py::init<int64_t, tenzor::Tensor>(),
             py::arg("dim"), py::arg("concentration"),
             "LKJ distribution over Cholesky factors of correlation matrices");

    dist_m.def("kl_divergence", &tenzor::distributions::kl_divergence,
               py::arg("p"), py::arg("q"),
               "Compute KL(p || q) for supported distribution pairs");

    // --- Transform base class ---
    py::class_<tenzor::distributions::Transform,
               std::shared_ptr<tenzor::distributions::Transform>>(dist_m, "Transform",
        "Abstract base class for invertible transforms")
        .def("__call__", &tenzor::distributions::Transform::call, py::arg("x"),
             "Apply the transform: y = f(x)")
        .def("inv", &tenzor::distributions::Transform::inv, py::arg("y"),
             "Apply the inverse: x = f^{-1}(y)")
        .def("log_abs_det_jacobian", &tenzor::distributions::Transform::log_abs_det_jacobian,
             py::arg("x"), py::arg("y"),
             "Log absolute determinant of the Jacobian");

    // --- Concrete transforms ---
    py::class_<tenzor::distributions::ExpTransform,
               tenzor::distributions::Transform,
               std::shared_ptr<tenzor::distributions::ExpTransform>>(dist_m, "ExpTransform",
        "Exponential transform: y = exp(x)")
        .def(py::init<>());

    py::class_<tenzor::distributions::AffineTransform,
               tenzor::distributions::Transform,
               std::shared_ptr<tenzor::distributions::AffineTransform>>(dist_m, "AffineTransform",
        "Affine transform: y = loc + scale * x")
        .def(py::init<tenzor::Tensor, tenzor::Tensor>(),
             py::arg("loc"), py::arg("scale"));

    py::class_<tenzor::distributions::SigmoidTransform,
               tenzor::distributions::Transform,
               std::shared_ptr<tenzor::distributions::SigmoidTransform>>(dist_m, "SigmoidTransform",
        "Sigmoid transform: y = 1 / (1 + exp(-x))")
        .def(py::init<>());

    py::class_<tenzor::distributions::TanhTransform,
               tenzor::distributions::Transform,
               std::shared_ptr<tenzor::distributions::TanhTransform>>(dist_m, "TanhTransform",
        "Tanh transform: y = tanh(x)")
        .def(py::init<>());

    py::class_<tenzor::distributions::SoftmaxTransform,
               tenzor::distributions::Transform,
               std::shared_ptr<tenzor::distributions::SoftmaxTransform>>(dist_m, "SoftmaxTransform",
        "Softmax transform along a dimension")
        .def(py::init<int64_t>(), py::arg("dim") = -1);

    py::class_<tenzor::distributions::ComposeTransform,
               tenzor::distributions::Transform,
               std::shared_ptr<tenzor::distributions::ComposeTransform>>(dist_m, "ComposeTransform",
        "Composition of transforms: y = f_n(f_{n-1}(...f_1(x)))")
        .def(py::init<std::vector<std::shared_ptr<tenzor::distributions::Transform>>>(),
             py::arg("transforms"));

    // --- TransformedDistribution ---
    py::class_<tenzor::distributions::TransformedDistribution,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::TransformedDistribution>>(dist_m, "TransformedDistribution",
        "Distribution formed by applying transforms to a base distribution")
        .def(py::init<std::shared_ptr<tenzor::distributions::Distribution>,
                       std::vector<std::shared_ptr<tenzor::distributions::Transform>>>(),
             py::arg("base_distribution"), py::arg("transforms"),
             "Create a transformed distribution from a base and a list of transforms");

    // --- Independent ---
    py::class_<tenzor::distributions::Independent,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::Independent>>(dist_m, "Independent",
        "Reinterprets trailing batch dimensions as event dimensions")
        .def(py::init<std::shared_ptr<tenzor::distributions::Distribution>, int64_t>(),
             py::arg("base_distribution"), py::arg("reinterpreted_batch_ndims"),
             "Create Independent wrapper around a base distribution")
        .def_property_readonly("reinterpreted_batch_ndims",
             &tenzor::distributions::Independent::reinterpreted_batch_ndims);

    // --- MixtureSameFamily ---
    py::class_<tenzor::distributions::MixtureSameFamily,
               tenzor::distributions::Distribution,
               std::shared_ptr<tenzor::distributions::MixtureSameFamily>>(dist_m, "MixtureSameFamily",
        "Mixture model where all components are from the same distribution family")
        .def(py::init<std::shared_ptr<tenzor::distributions::Distribution>,
                       std::shared_ptr<tenzor::distributions::Distribution>,
                       tenzor::Tensor>(),
             py::arg("mixture_distribution"), py::arg("component_distribution"),
             py::arg("mixture_logits"),
             "Create mixture from Categorical, component distribution, and log-weights")
        .def(py::init<const tenzor::Tensor&,
                       std::shared_ptr<tenzor::distributions::Distribution>>(),
             py::arg("weights"), py::arg("component_distribution"),
             "Create mixture from unnormalized weights and component distribution");


    // =========================================================================
    // Image I/O — tenzor.io submodule
    // =========================================================================
    auto io_mod = m.def_submodule("io", "Image I/O operations");

    py::enum_<tenzor::io::ImageMode>(io_mod, "ImageMode")
        .value("RGB", tenzor::io::ImageMode::RGB)
        .value("RGBA", tenzor::io::ImageMode::RGBA)
        .value("GRAYSCALE", tenzor::io::ImageMode::GRAYSCALE)
        .value("UNCHANGED", tenzor::io::ImageMode::UNCHANGED);

    io_mod.def("read_image", &tenzor::io::read_image,
        py::arg("path"), py::arg("mode") = tenzor::io::ImageMode::RGB,
        "Read an image file into a uint8 CHW tensor",
        py::call_guard<py::gil_scoped_release>());
    io_mod.def("write_image", &tenzor::io::write_image,
        py::arg("tensor"), py::arg("path"), py::arg("quality") = 95,
        "Write a uint8 CHW tensor to an image file",
        py::call_guard<py::gil_scoped_release>());
    io_mod.def("decode_jpeg", &tenzor::io::decode_jpeg,
        py::arg("data"), py::arg("mode") = tenzor::io::ImageMode::RGB,
        "Decode JPEG bytes into a uint8 CHW tensor");
    io_mod.def("decode_png", &tenzor::io::decode_png,
        py::arg("data"), py::arg("mode") = tenzor::io::ImageMode::RGB,
        "Decode PNG bytes into a uint8 CHW tensor");
    io_mod.def("encode_jpeg", &tenzor::io::encode_jpeg,
        py::arg("tensor"), py::arg("quality") = 95,
        "Encode a uint8 CHW tensor to JPEG bytes");
    io_mod.def("encode_png", &tenzor::io::encode_png,
        py::arg("tensor"),
        "Encode a uint8 CHW tensor to PNG bytes");

    // -------------------------------------------------------------------------
    // _foreach_* family — multi-tensor optimizer primitives (Phase 9-W2)
    // -------------------------------------------------------------------------
    m.def("_foreach_add",
          static_cast<std::vector<Tensor>(*)(const std::vector<Tensor>&, const std::vector<Tensor>&)>(&tenzor::foreach_add),
          py::arg("a"), py::arg("b"),
          "Element-wise add over a list of tensor pairs.");
    m.def("_foreach_add_",
          [](std::vector<Tensor> a, const std::vector<Tensor>& b) {
              tenzor::foreach_add_(a, b);
              return a;
          },
          py::arg("a"), py::arg("b"),
          "In-place element-wise add over a list of tensor pairs.");
    m.def("_foreach_sub",
          &tenzor::foreach_sub,
          py::arg("a"), py::arg("b"),
          "Element-wise sub over a list of tensor pairs.");
    m.def("_foreach_sub_",
          [](std::vector<Tensor> a, const std::vector<Tensor>& b) {
              tenzor::foreach_sub_(a, b);
              return a;
          },
          py::arg("a"), py::arg("b"),
          "In-place element-wise sub over a list of tensor pairs.");
    m.def("_foreach_mul",
          &tenzor::foreach_mul,
          py::arg("a"), py::arg("b"),
          "Element-wise mul over a list of tensor pairs.");
    m.def("_foreach_mul_",
          [](std::vector<Tensor> a, const std::vector<Tensor>& b) {
              tenzor::foreach_mul_(a, b);
              return a;
          },
          py::arg("a"), py::arg("b"),
          "In-place element-wise mul over a list of tensor pairs.");
    m.def("_foreach_div",
          &tenzor::foreach_div,
          py::arg("a"), py::arg("b"),
          "Element-wise div over a list of tensor pairs.");
    m.def("_foreach_div_",
          [](std::vector<Tensor> a, const std::vector<Tensor>& b) {
              tenzor::foreach_div_(a, b);
              return a;
          },
          py::arg("a"), py::arg("b"),
          "In-place element-wise div over a list of tensor pairs.");
    m.def("_foreach_neg",
          &tenzor::foreach_neg,
          py::arg("a"),
          "Element-wise negation over a list of tensors.");
    m.def("_foreach_neg_",
          [](std::vector<Tensor> a) {
              tenzor::foreach_neg_(a);
              return a;
          },
          py::arg("a"),
          "In-place element-wise negation over a list of tensors.");
    m.def("_foreach_abs",
          &tenzor::foreach_abs,
          py::arg("a"),
          "Element-wise absolute value over a list of tensors.");
    m.def("_foreach_abs_",
          [](std::vector<Tensor> a) {
              tenzor::foreach_abs_(a);
              return a;
          },
          py::arg("a"),
          "In-place element-wise absolute value over a list of tensors.");
    m.def("_foreach_sqrt",
          &tenzor::foreach_sqrt,
          py::arg("a"),
          "Element-wise sqrt over a list of tensors.");
    m.def("_foreach_sqrt_",
          [](std::vector<Tensor> a) {
              tenzor::foreach_sqrt_(a);
              return a;
          },
          py::arg("a"),
          "In-place element-wise sqrt over a list of tensors.");
    m.def("_foreach_zero_",
          [](std::vector<Tensor> a) {
              tenzor::foreach_zero_(a);
              return a;
          },
          py::arg("a"),
          "Zero-fill each tensor in the list in-place.");
    m.def("_foreach_copy",
          &tenzor::foreach_copy,
          py::arg("src"),
          "Return deep copies of all tensors in the list.");
    m.def("_foreach_addcdiv_",
          [](std::vector<Tensor> self,
             const std::vector<Tensor>& a,
             const std::vector<Tensor>& b,
             double scalar) {
              tenzor::foreach_addcdiv_(self, a, b, scalar);
              return self;
          },
          py::arg("self"), py::arg("a"), py::arg("b"), py::arg("scalar"),
          "self[i] += scalar * a[i] / b[i] for each tensor.");
    m.def("_foreach_addcmul_",
          [](std::vector<Tensor> self,
             const std::vector<Tensor>& a,
             const std::vector<Tensor>& b,
             double scalar) {
              tenzor::foreach_addcmul_(self, a, b, scalar);
              return self;
          },
          py::arg("self"), py::arg("a"), py::arg("b"), py::arg("scalar"),
          "self[i] += scalar * a[i] * b[i] for each tensor.");
    m.def("_foreach_lerp_",
          [](std::vector<Tensor> self,
             const std::vector<Tensor>& b,
             double scalar) {
              tenzor::foreach_lerp_(self, b, scalar);
              return self;
          },
          py::arg("self"), py::arg("b"), py::arg("scalar"),
          "self[i] = lerp(self[i], b[i], scalar) for each tensor.");
    m.def("_foreach_norm",
          &tenzor::foreach_norm,
          py::arg("a"), py::arg("p") = 2.0,
          "Compute p-norm of each tensor in the list.");

} // register_core

} // namespace tenzor::python
