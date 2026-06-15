// tenzor.lite Python bindings.
//
// Exposes:
//   tz.lite.Runtime(path: str)     — load a .tzlite, callable like a Module
//   tz.lite.export(module, path, *, input_shape, input_dtype='float32')
//                                  — export an nn::Module to .tzlite
//
// Tensor crossing: Python sees `tz.Tensor` objects. Internally the bindings
// convert each Tensor to a LiteTensor view (copy on entry to forward()),
// invoke the runtime, and copy each output LiteTensor back to a Tensor.
// Phase 5 polish can swap to zero-copy via the same view_as_tensor /
// to_lite_tensor helpers used elsewhere.

#include "register.hpp"

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <tenzor/core/tensor.hpp>
#include <tenzor/lite/exporter.hpp>
#include <tenzor/lite/lite_graph.hpp>
#include <tenzor/lite/runtime.hpp>
#include <tenzor/lite/tensor_bridge.hpp>
#include <tenzor/nn/module.hpp>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace tenzor::python {

namespace {

// Build an owning LiteTensor from a CPU-contiguous Tenzor Tensor (copy).
auto tensor_to_lite_tensor(const Tensor& t) -> lite::LiteTensor {
    Tensor src = t;
    if (src.device().type != Device::Type::CPU) src = src.to(Device::cpu());
    if (!src.is_contiguous()) src = src.contiguous();

    if (src.ndim() > lite::kMaxDims) {
        throw std::invalid_argument(
            "tensor_to_lite_tensor: tensor rank " +
            std::to_string(src.ndim()) + " exceeds lite::kMaxDims (" +
            std::to_string(lite::kMaxDims) + ")");
    }

    lite::LiteTensor lt;
    lt.ndim = static_cast<int32_t>(src.ndim());
    lt.dtype = src.dtype();
    lt.owns_data = true;
    int64_t numel = 1;
    for (int32_t i = 0; i < lt.ndim; ++i) {
        lt.shape[i] = src.size(i);
        numel *= lt.shape[i];
    }
    for (int32_t i = lt.ndim - 1; i >= 0; --i) {
        lt.strides[i] = (i == lt.ndim - 1) ? 1
                        : lt.strides[i + 1] * lt.shape[i + 1];
    }
    const auto nbytes = static_cast<size_t>(numel * dtype_size(lt.dtype));
    lt.data = (numel > 0) ? std::malloc(nbytes) : nullptr;
    if (numel > 0 && lt.data == nullptr) throw std::bad_alloc{};
    if (numel > 0) std::memcpy(lt.data, src.data_ptr(), nbytes);
    return lt;
}

// Build an owning Tensor from a LiteTensor (deep copy via the bridge's
// non-owning view + Tensor::clone).
auto lite_tensor_to_tensor(const lite::LiteTensor& lt) -> Tensor {
    return lite::view_as_tensor(lt).clone();
}

// Convert a Python DType string to DType (subset relevant to inference).
auto parse_dtype(const std::string& s) -> DType {
    if (s == "float32" || s == "f32") return DType::Float32;
    if (s == "float64" || s == "f64") return DType::Float64;
    if (s == "float16" || s == "f16") return DType::Float16;
    if (s == "bfloat16" || s == "bf16") return DType::BFloat16;
    if (s == "int8")  return DType::Int8;
    if (s == "int16") return DType::Int16;
    if (s == "int32") return DType::Int32;
    if (s == "int64") return DType::Int64;
    throw std::invalid_argument("tenzor.lite: unsupported dtype string '" + s + "'");
}

}  // anonymous namespace

void register_lite(py::module_& m) {
    auto lite_mod = m.def_submodule(
        "lite",
        "Lite inference runtime — load and execute .tzlite serialised models");

    py::class_<lite::LiteRuntime, std::unique_ptr<lite::LiteRuntime>>(lite_mod, "Runtime")
        .def(py::init([](const std::string& path) {
                 return lite::LiteRuntime::load(path);
             }),
             py::arg("path"),
             R"doc(
Load a `.tzlite` file as an inference runtime.

The file is parsed at construction; the resulting runtime is reusable across
forward() calls. Weights are held by the runtime; per-call memory is the
output buffer only.
             )doc")
        .def("__call__",
             [](lite::LiteRuntime& self, const Tensor& x) {
                 auto lt_in = tensor_to_lite_tensor(x);
                 auto out = self.forward(lt_in);
                 return lite_tensor_to_tensor(out);
             },
             py::arg("input"),
             "Single-input forward — returns a single Tensor.",
             // S.20: release the GIL across forward() so concurrent
             // Python threads (DataLoader workers, host-side decode) make
             // progress while the runtime runs its C++ inference loop.
             py::call_guard<py::gil_scoped_release>())
        .def("forward",
             [](lite::LiteRuntime& self, const std::vector<Tensor>& xs) {
                 std::vector<lite::LiteTensor> lts;
                 lts.reserve(xs.size());
                 for (const auto& x : xs) lts.push_back(tensor_to_lite_tensor(x));
                 auto outs = self.forward(lts);
                 std::vector<Tensor> result;
                 result.reserve(outs.size());
                 for (const auto& o : outs) result.push_back(lite_tensor_to_tensor(o));
                 return result;
             },
             py::arg("inputs"),
             "Multi-input forward — accepts and returns a list of Tensors.",
             // S.20: see __call__ above.
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly(
            "input_shapes",
            [](const lite::LiteRuntime& self) { return self.input_shapes(); })
        .def_property_readonly(
            "output_shapes",
            [](const lite::LiteRuntime& self) { return self.output_shapes(); })
        .def("metadata",
             [](const lite::LiteRuntime& self, const std::string& key) {
                 return self.model_metadata(key);
             },
             py::arg("key"),
             "Look up a free-form metadata value stored in the file's META section.");

    lite_mod.def(
        "export",
        [](std::shared_ptr<nn::Module> module,
           const std::string& path,
           std::vector<int64_t> input_shape,
           const std::string& input_dtype) {
            if (module == nullptr) {
                throw std::invalid_argument("tenzor.lite.export: module is None");
            }
            lite::ExportOptions opts;
            opts.input_shape  = std::move(input_shape);
            opts.input_dtype  = parse_dtype(input_dtype);
            lite::export_to_tzlite(*module, path, opts);
        },
        py::arg("module"),
        py::arg("path"),
        py::kw_only(),
        py::arg("input_shape"),
        py::arg("input_dtype") = "float32",
        R"doc(
Export an `nn.Module` to a `.tzlite` file.

The module is put into eval mode and walked recursively. Phase 3 supports
nn.Linear, nn.Sequential, nn.ReLU, nn.Sigmoid, nn.Tanh, and nn.GELU.
Unsupported layer types raise RuntimeError with the offending class name.

The exported file is portable: load it on any host with `tz.lite.Runtime`
and the same backend support compiled in.
        )doc");
}

}  // namespace tenzor::python
