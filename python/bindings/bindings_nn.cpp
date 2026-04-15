// tenzor.nn Python bindings.  Extracted from python/bindings.cpp as part
// of the ongoing P3.4 monolith split.
//
// Covers:  Module base class (with PyModule trampoline), ModuleList,
// ModuleDict, Sequential, all layers (Linear, Conv*, BatchNorm*, LayerNorm,
// GroupNorm, RNN/LSTM/GRU, Attention, Transformer, Embedding, etc.),
// activation modules, loss functions, functional API, gradient clipping,
// PackedSequence / RNN utilities, training callbacks, checkpointing,
// and the NeuralNetwork high-level wrapper.

#include "register.hpp"

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "py_module_trampoline.hpp"

#include <tenzor/tenzor.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include <tenzor/nn/layers/attention.hpp>
#include <tenzor/nn/layers/transformer.hpp>
#include <tenzor/nn/layers/embedding.hpp>
#include <tenzor/nn/layers/lazy_linear.hpp>
#include <tenzor/nn/layers/sync_batchnorm.hpp>
#include <tenzor/nn/layers/alibi.hpp>
#include <tenzor/nn/layers/gqa_attention.hpp>
#include <tenzor/nn/layers/drop_path.hpp>
#include <tenzor/nn/layers/vision.hpp>
#include <tenzor/nn/layers/mobilenet.hpp>
#include <tenzor/nn/layers/segmentation.hpp>
#include <tenzor/nn/layers/hrm.hpp>
#include <tenzor/nn/layers/sparse_linear.hpp>
#include <tenzor/nn/layers/sparse_embedding.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <tenzor/nn/metrics.hpp>
#include <tenzor/nn/loss/contrastive.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/nn/utils/clip_grad.hpp>
#include <tenzor/nn/utils/rnn_utils.hpp>
#include <tenzor/nn/callbacks.hpp>
#include <tenzor/nn/training.hpp>
#include <tenzor/nn/checkpoint.hpp>
#include <tenzor/nn/init.hpp>
#include <tenzor/nn/serialize.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/transform.hpp>
#include <tenzor/ops/vision.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>

namespace py = pybind11;

namespace tenzor::python {

void register_nn(py::module_& m) {
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
        //
        // Each register_*_hook returns a RemovableHandle (defined a few
        // lines above where tenzor::nn::Module itself is bound) instead
        // of a raw integer hook_id. RemovableHandle supports:
        //   - handle.remove()     — PyTorch-style removal
        //   - handle.id           — raw int for legacy callers
        //   - int(handle)         — implicit conversion to int
        //   - model.remove_hook(handle) — legacy path via __index__
        // ====================================================================
        .def("register_forward_hook", [](py::object self, py::object hook) -> py::object {
            auto& mod = self.cast<tenzor::nn::Module&>();
            py::object hook_ref = hook;
            size_t hook_id = mod.register_forward_post_hook(
                [hook_ref](tenzor::nn::Module* m, const tenzor::Variable& input, const tenzor::Variable& output) {
                    py::gil_scoped_acquire acquire;
                    hook_ref(m, input, output);
                });
            return py::module_::import("tenzor.nn").attr("RemovableHandle")(self, hook_id);
        }, py::arg("hook"),
           "Register hook called after forward pass (PyTorch-compatible). "
           "Returns a RemovableHandle; call handle.remove() to detach.")
        .def("register_forward_pre_hook", [](py::object self, py::object hook) -> py::object {
            auto& mod = self.cast<tenzor::nn::Module&>();
            py::object hook_ref = hook;
            size_t hook_id = mod.register_forward_pre_hook(
                [hook_ref](tenzor::nn::Module* m, const tenzor::Variable& input) {
                    py::gil_scoped_acquire acquire;
                    hook_ref(m, input);
                });
            return py::module_::import("tenzor.nn").attr("RemovableHandle")(self, hook_id);
        }, py::arg("hook"),
           "Register hook called before forward pass. Returns a "
           "RemovableHandle.")
        .def("register_backward_hook", [](py::object self, py::object hook) -> py::object {
            auto& mod = self.cast<tenzor::nn::Module&>();
            py::object hook_ref = hook;
            size_t hook_id = mod.register_backward_post_hook(
                [hook_ref](tenzor::nn::Module* m, const tenzor::Variable& grad_input, const tenzor::Variable& grad_output) {
                    py::gil_scoped_acquire acquire;
                    hook_ref(m, grad_input, grad_output);
                });
            return py::module_::import("tenzor.nn").attr("RemovableHandle")(self, hook_id);
        }, py::arg("hook"),
           "Register hook called after backward pass (PyTorch-compatible). "
           "Returns a RemovableHandle.")
        .def("register_full_backward_hook", [](py::object self, py::object hook) -> py::object {
            auto& mod = self.cast<tenzor::nn::Module&>();
            py::object hook_ref = hook;
            size_t hook_id = mod.register_backward_post_hook(
                [hook_ref](tenzor::nn::Module* m, const tenzor::Variable& grad_input, const tenzor::Variable& grad_output) {
                    py::gil_scoped_acquire acquire;
                    hook_ref(m, grad_input, grad_output);
                });
            return py::module_::import("tenzor.nn").attr("RemovableHandle")(self, hook_id);
        }, py::arg("hook"),
           "Register hook called after backward pass with full gradients. "
           "Returns a RemovableHandle.")
        .def("register_full_backward_pre_hook", [](py::object self, py::object hook) -> py::object {
            auto& mod = self.cast<tenzor::nn::Module&>();
            py::object hook_ref = hook;
            size_t hook_id = mod.register_backward_pre_hook(
                [hook_ref](tenzor::nn::Module* m, const tenzor::Variable& grad_output) {
                    py::gil_scoped_acquire acquire;
                    hook_ref(m, grad_output);
                });
            return py::module_::import("tenzor.nn").attr("RemovableHandle")(self, hook_id);
        }, py::arg("hook"),
           "Register hook called before backward pass. Returns a "
           "RemovableHandle.")
        .def("register_forward_post_hook", [](py::object self, py::object hook) -> py::object {
            auto& mod = self.cast<tenzor::nn::Module&>();
            py::object hook_ref = hook;
            size_t hook_id = mod.register_forward_post_hook(
                [hook_ref](tenzor::nn::Module* m, const tenzor::Variable& input, const tenzor::Variable& output) {
                    py::gil_scoped_acquire acquire;
                    hook_ref(m, input, output);
                });
            return py::module_::import("tenzor.nn").attr("RemovableHandle")(self, hook_id);
        }, py::arg("hook"),
           "Register hook called after forward pass. Returns a "
           "RemovableHandle.")
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

    py::class_<tenzor::nn::LocalResponseNorm, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::LocalResponseNorm>>(nn, "LocalResponseNorm",
               "Local Response Normalization across channels")
        .def(py::init<int64_t, double, double, double>(),
             py::arg("size"),
             py::arg("alpha") = 1e-4,
             py::arg("beta") = 0.75,
             py::arg("k") = 1.0);

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

    py::class_<tenzor::nn::LPPool1d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::LPPool1d>>(nn, "LPPool1d",
               "1D power-average pooling: (avg_pool(|x|^p))^(1/p)")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("norm_type"),
             py::arg("kernel_size"),
             py::arg("stride") = -1);

    py::class_<tenzor::nn::LPPool2d, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::LPPool2d>>(nn, "LPPool2d",
               "2D power-average pooling: (avg_pool(|x|^p))^(1/p)")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("norm_type"),
             py::arg("kernel_size"),
             py::arg("stride") = -1)
        .def(py::init<int64_t, std::pair<int64_t, int64_t>, std::pair<int64_t, int64_t>>(),
             py::arg("norm_type"),
             py::arg("kernel_size"),
             py::arg("stride") = std::pair<int64_t, int64_t>{-1, -1});

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
        .def("__repr__", []([[maybe_unused]] const tenzor::nn::PReLU& self) {
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

    py::class_<tenzor::nn::Hardtanh, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Hardtanh>>(nn, "Hardtanh")
        .def(py::init<double, double>(),
             py::arg("min_val") = -1.0, py::arg("max_val") = 1.0)
        .def("__repr__", [](const tenzor::nn::Hardtanh&) { return "Hardtanh()"; });

    // Functional activations
    nn.def("hardswish", [](const tenzor::Variable& input) {
        return tenzor::nn::hardswish(input);
    }, "Functional Hardswish activation", py::arg("input"));

    nn.def("hardsigmoid", [](const tenzor::Variable& input) {
        return tenzor::nn::hardsigmoid(input);
    }, "Functional Hardsigmoid activation", py::arg("input"));

    nn.def("hardtanh", [](const tenzor::Variable& input, double min_val, double max_val) {
        return tenzor::nn::hardtanh(input, min_val, max_val);
    }, "Functional Hardtanh activation", py::arg("input"),
       py::arg("min_val") = -1.0, py::arg("max_val") = 1.0);

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
        .def("__repr__", []([[maybe_unused]] const tenzor::nn::ChannelShuffle& self) {
            return "ChannelShuffle()";
        });

    // Unfold / Fold (im2col / col2im)
    py::class_<tenzor::nn::Unfold, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Unfold>>(nn, "Unfold",
        "Extracts sliding local blocks from a batched input tensor (im2col).")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>(),
             py::arg("kernel_size"), py::arg("dilation") = 1,
             py::arg("padding") = 0, py::arg("stride") = 1)
        .def("__repr__", [](const tenzor::nn::Unfold&) { return "Unfold()"; });

    py::class_<tenzor::nn::Fold, tenzor::nn::Module,
               std::shared_ptr<tenzor::nn::Fold>>(nn, "Fold",
        "Combines sliding local blocks into a tensor (col2im).")
        .def(py::init<std::vector<int64_t>, int64_t, int64_t, int64_t, int64_t>(),
             py::arg("output_size"), py::arg("kernel_size"),
             py::arg("dilation") = 1, py::arg("padding") = 0, py::arg("stride") = 1)
        .def("__repr__", [](const tenzor::nn::Fold&) { return "Fold()"; });

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
        .def(py::init([](int64_t embed_dim, int64_t num_heads, int64_t num_kv_heads,
                         double dropout, bool bias, bool is_causal, int64_t window_size) {
                return std::make_shared<tenzor::nn::GroupedQueryAttention>(
                    embed_dim, num_heads, num_kv_heads, dropout, bias, is_causal,
                    nullptr, window_size);
             }),
             py::arg("embed_dim"),
             py::arg("num_heads"),
             py::arg("num_kv_heads"),
             py::arg("dropout") = 0.0,
             py::arg("bias") = true,
             py::arg("is_causal") = false,
             py::arg("window_size") = -1)
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
        .def_property_readonly("is_causal", &tenzor::nn::GroupedQueryAttention::is_causal)
        .def_property_readonly("window_size", &tenzor::nn::GroupedQueryAttention::window_size);

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
    nn.def("gelu", &tenzor::nn::gelu,
           py::arg("input"), py::arg("approximate") = "none",
           "GELU activation function");
    nn.def("sigmoid", &tenzor::nn::sigmoid, "Sigmoid activation function");
    nn.def("tanh", &tenzor::nn::tanh, "Tanh activation function");
    nn.def("softmax", &tenzor::nn::softmax, "Softmax activation function",
          py::arg("input"), py::arg("dim") = -1);
    nn.def("log_softmax", &tenzor::nn::log_softmax, "Log-Softmax activation function",
          py::arg("input"), py::arg("dim") = -1);
    nn.def("selu", &tenzor::nn::selu, "SELU activation function");
    nn.def("swish", &tenzor::nn::swish, "Swish activation function");
    nn.def("mish", &tenzor::nn::mish, "Mish activation function");
    nn.def("rrelu", &tenzor::nn::rrelu, "Randomized ReLU activation function",
          py::arg("input"), py::arg("lower") = 1.0 / 8.0,
          py::arg("upper") = 1.0 / 3.0, py::arg("training") = false);
    nn.def("log_sigmoid", &tenzor::nn::log_sigmoid, "Log-Sigmoid activation function");

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

    py::class_<tenzor::nn::MultiLabelMarginLoss>(nn, "MultiLabelMarginLoss",
        "Multi-label classification hinge loss")
        .def(py::init<tenzor::nn::Reduction>(),
             py::arg("reduction") = tenzor::nn::Reduction::Mean)
        .def("forward", &tenzor::nn::MultiLabelMarginLoss::forward,
             py::arg("input"), py::arg("target"),
             py::call_guard<py::gil_scoped_release>())
        .def("__call__", &tenzor::nn::MultiLabelMarginLoss::operator(),
             py::arg("input"), py::arg("target"));

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
             "Get optimizer")
        .def("add_metric", &tenzor::nn::NeuralNetwork::add_metric,
             py::arg("metric"),
             R"pbdoc(
                Register a metric to be evaluated during training.

                Metrics are updated with (predictions, targets) after each batch
                during fit(), computed and logged at each epoch end, then reset.

                Args:
                    metric: A Metric instance (e.g., Accuracy, F1Score, MeanSquaredError)

                Example:
                    >>> nn_wrapper.add_metric(tenzor.nn.Accuracy(num_classes=10))
                    >>> nn_wrapper.add_metric(tenzor.nn.F1Score(num_classes=10))
                    >>> nn_wrapper.fit(train_loader, epochs=10)
             )pbdoc")
        .def_property_readonly("metrics", &tenzor::nn::NeuralNetwork::metrics,
             "Get registered metrics");


    // =========================================================================
    // Training Metrics
    // =========================================================================

    py::enum_<tenzor::nn::AverageMode>(nn, "AverageMode",
        "Averaging mode for multi-class classification metrics")
        .value("Micro", tenzor::nn::AverageMode::Micro)
        .value("Macro", tenzor::nn::AverageMode::Macro)
        .value("Weighted", tenzor::nn::AverageMode::Weighted);

    // Metric base class (needed for shared_ptr<Metric> in add_metric)
    py::class_<tenzor::nn::Metric, std::shared_ptr<tenzor::nn::Metric>>(nn, "Metric",
        "Abstract base class for training metrics")
        .def("update", &tenzor::nn::Metric::update,
             py::arg("preds"), py::arg("targets"),
             "Accumulate a batch of predictions and targets")
        .def("compute", &tenzor::nn::Metric::compute,
             "Compute the metric from accumulated state")
        .def("reset", &tenzor::nn::Metric::reset,
             "Reset internal state for a new epoch")
        .def("name", &tenzor::nn::Metric::name,
             "Get the metric name");

    py::class_<tenzor::nn::Accuracy, tenzor::nn::Metric, std::shared_ptr<tenzor::nn::Accuracy>>(nn, "Accuracy",
        "Classification accuracy metric (stateful, accumulates across batches)")
        .def(py::init<int64_t>(), py::arg("num_classes") = 2)
        .def("update", &tenzor::nn::Accuracy::update,
             py::arg("preds"), py::arg("targets"))
        .def("compute", &tenzor::nn::Accuracy::compute)
        .def("reset", &tenzor::nn::Accuracy::reset);

    py::class_<tenzor::nn::Precision, tenzor::nn::Metric, std::shared_ptr<tenzor::nn::Precision>>(nn, "Precision",
        "Precision metric: TP / (TP + FP)")
        .def(py::init<int64_t, tenzor::nn::AverageMode>(),
             py::arg("num_classes") = 2,
             py::arg("average") = tenzor::nn::AverageMode::Macro)
        .def("update", &tenzor::nn::Precision::update,
             py::arg("preds"), py::arg("targets"))
        .def("compute", &tenzor::nn::Precision::compute)
        .def("reset", &tenzor::nn::Precision::reset);

    py::class_<tenzor::nn::Recall, tenzor::nn::Metric, std::shared_ptr<tenzor::nn::Recall>>(nn, "Recall",
        "Recall metric: TP / (TP + FN)")
        .def(py::init<int64_t, tenzor::nn::AverageMode>(),
             py::arg("num_classes") = 2,
             py::arg("average") = tenzor::nn::AverageMode::Macro)
        .def("update", &tenzor::nn::Recall::update,
             py::arg("preds"), py::arg("targets"))
        .def("compute", &tenzor::nn::Recall::compute)
        .def("reset", &tenzor::nn::Recall::reset);

    py::class_<tenzor::nn::F1Score, tenzor::nn::Metric, std::shared_ptr<tenzor::nn::F1Score>>(nn, "F1Score",
        "F1 Score: harmonic mean of precision and recall")
        .def(py::init<int64_t, tenzor::nn::AverageMode>(),
             py::arg("num_classes") = 2,
             py::arg("average") = tenzor::nn::AverageMode::Macro)
        .def("update", &tenzor::nn::F1Score::update,
             py::arg("preds"), py::arg("targets"))
        .def("compute", &tenzor::nn::F1Score::compute)
        .def("reset", &tenzor::nn::F1Score::reset);

    py::class_<tenzor::nn::AUROC, tenzor::nn::Metric, std::shared_ptr<tenzor::nn::AUROC>>(nn, "AUROC",
        "Area Under the ROC Curve (binary classification)")
        .def(py::init<>())
        .def("update", &tenzor::nn::AUROC::update,
             py::arg("preds"), py::arg("targets"))
        .def("compute", &tenzor::nn::AUROC::compute)
        .def("reset", &tenzor::nn::AUROC::reset);

    py::class_<tenzor::nn::ConfusionMatrix, tenzor::nn::Metric, std::shared_ptr<tenzor::nn::ConfusionMatrix>>(nn, "ConfusionMatrix",
        "NxN confusion matrix for classification")
        .def(py::init<int64_t>(), py::arg("num_classes"))
        .def("update", &tenzor::nn::ConfusionMatrix::update,
             py::arg("preds"), py::arg("targets"))
        .def("compute", &tenzor::nn::ConfusionMatrix::compute)
        .def("reset", &tenzor::nn::ConfusionMatrix::reset);

    py::class_<tenzor::nn::MeanAbsoluteError, tenzor::nn::Metric, std::shared_ptr<tenzor::nn::MeanAbsoluteError>>(nn, "MeanAbsoluteError",
        "Mean Absolute Error metric (running average)")
        .def(py::init<>())
        .def("update", &tenzor::nn::MeanAbsoluteError::update,
             py::arg("preds"), py::arg("targets"))
        .def("compute", &tenzor::nn::MeanAbsoluteError::compute)
        .def("reset", &tenzor::nn::MeanAbsoluteError::reset);

    py::class_<tenzor::nn::MeanSquaredError, tenzor::nn::Metric, std::shared_ptr<tenzor::nn::MeanSquaredError>>(nn, "MeanSquaredError",
        "Mean Squared Error metric (running average)")
        .def(py::init<>())
        .def("update", &tenzor::nn::MeanSquaredError::update,
             py::arg("preds"), py::arg("targets"))
        .def("compute", &tenzor::nn::MeanSquaredError::compute)
        .def("reset", &tenzor::nn::MeanSquaredError::reset);


} // register_nn

} // namespace tenzor::python
