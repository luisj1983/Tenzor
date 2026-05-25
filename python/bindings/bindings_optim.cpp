// tenzor.optim Python bindings. Extracted from python/bindings.cpp
// as part of the incremental split of the monolith.
//
// Covers: ClipMode, ClipConfig, ParamGroup, Optimizer base, SGD, Adam, AdamW,
// RMSprop, Adagrad, Adadelta, RAdam, NAdam, Adamax, LAMB, SparseAdam,
// and lr_scheduler (StepLR, ExponentialLR, CosineAnnealingLR,
// ReduceLROnPlateau, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts).

#include "register.hpp"

#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/radam.hpp>
#include <tenzor/nn/optim/nadam.hpp>
#include <tenzor/nn/optim/adamax.hpp>
#include <tenzor/nn/optim/lamb.hpp>
#include <tenzor/nn/optim/sparse_adam.hpp>
#include <tenzor/nn/optim/adam_atan2.hpp>
#include <tenzor/nn/optim/rprop.hpp>
#include <tenzor/nn/optim/lbfgs.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/asgd.hpp>
#include <tenzor/nn/optim/sam.hpp>
#include <tenzor/nn/optim/swa.hpp>

namespace py = pybind11;

namespace tenzor::python {

void register_optim(py::module_& m) {
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
    // W.20: zero_grad walks every parameter and dispatches a fill kernel;
    // add_param_group / load_state_dict allocate state buffers. None of
    // these need the GIL during the C++ work — release across the call so
    // DataLoader workers / DDP comm hooks make progress in parallel.
    py::class_<tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::Optimizer>>(optim, "Optimizer",
        "Base class for all optimizers")
        .def("zero_grad", &tenzor::optim::Optimizer::zero_grad, "Zero out all parameter gradients",
             py::call_guard<py::gil_scoped_release>())
        .def("state_dict", &tenzor::optim::Optimizer::state_dict, "Get optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::Optimizer::load_state_dict, py::arg("state"),
             "Load optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>())
        .def("add_param_group", &tenzor::optim::Optimizer::add_param_group,
             py::arg("group"), "Add a parameter group with custom hyperparameters",
             py::call_guard<py::gil_scoped_release>())
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
            // Q.15: release the GIL so other Python threads (DataLoader
            // workers, DDP comm hooks) make progress while the step runs.
            // X.8: scope the release so it ends before `py::none()` is built.
            { py::gil_scoped_release release; self.step(); }
            return py::none();
        }, py::arg("closure") = py::none(),
           "Perform optimization step. Optionally takes a closure that recomputes the loss.")
        .def("zero_grad", &tenzor::optim::SGD::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("set_lr", &tenzor::optim::SGD::set_lr,
             py::arg("lr"), "Set learning rate")
        .def("get_lr", &tenzor::optim::SGD::get_lr,
             "Get current learning rate")
        .def("state_dict", &tenzor::optim::SGD::state_dict,
             "Get optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::SGD::load_state_dict,
             py::arg("state"), "Load optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>());  // W.20

    py::class_<tenzor::optim::ASGD, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::ASGD>>(optim, "ASGD",
        "Averaged Stochastic Gradient Descent optimizer")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 0.01,
             py::arg("lambd") = 1e-4, py::arg("alpha") = 0.75,
             py::arg("t0") = 1e6, py::arg("weight_decay") = 0.0)
        .def("step", [](tenzor::optim::ASGD& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            // Q.15: release GIL for the closure-less hot path.
            // X.8: scope so release ends before py::none() is built.
            { py::gil_scoped_release release; self.step(); }
            return py::none();
        }, py::arg("closure") = py::none(),
           "Perform optimization step. Optionally takes a closure that recomputes the loss.")
        .def("zero_grad", &tenzor::optim::ASGD::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("set_lr", &tenzor::optim::ASGD::set_lr,
             py::arg("lr"), "Set learning rate")
        .def("get_lr", &tenzor::optim::ASGD::get_lr,
             "Get current learning rate")
        .def("state_dict", &tenzor::optim::ASGD::state_dict,
             "Get optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::ASGD::load_state_dict,
             py::arg("state"), "Load optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>());  // W.20

    py::class_<tenzor::optim::Adam, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::Adam>>(optim, "Adam")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double, bool>(),
             py::arg("params"), py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             py::arg("amsgrad") = false)
        .def("step", [](tenzor::optim::Adam& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            // X.8: scope the GIL release so `release` is destroyed (re-acquires
            // GIL) BEFORE `py::none()` is constructed in the return statement.
            // Constructing a py::object while the GIL is released triggers a
            // pybind11 inc_ref() assertion failure on CPython 3.14+.
            { py::gil_scoped_release release; self.step(); }
            return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::Adam::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("set_lr", &tenzor::optim::Adam::set_lr,
             py::arg("lr"), "Set learning rate")
        .def("get_lr", &tenzor::optim::Adam::get_lr,
             "Get current learning rate")
        .def("state_dict", &tenzor::optim::Adam::state_dict,
             "Get optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::Adam::load_state_dict,
             py::arg("state"), "Load optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>());  // W.20

    py::class_<tenzor::optim::AdamW, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::AdamW>>(optim, "AdamW")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double, bool>(),
             py::arg("params"), py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.01,
             py::arg("amsgrad") = false)
        .def("step", [](tenzor::optim::AdamW& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            // X.8: scope the GIL release so `release` is destroyed (re-acquires
            // GIL) BEFORE `py::none()` is constructed in the return statement.
            // Constructing a py::object while the GIL is released triggers a
            // pybind11 inc_ref() assertion failure on CPython 3.14+.
            { py::gil_scoped_release release; self.step(); }
            return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::AdamW::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("set_lr", &tenzor::optim::AdamW::set_lr,
             py::arg("lr"), "Set learning rate")
        .def("get_lr", &tenzor::optim::AdamW::get_lr,
             "Get current learning rate")
        .def("state_dict", &tenzor::optim::AdamW::state_dict,
             "Get optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::AdamW::load_state_dict,
             py::arg("state"), "Load optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>());  // W.20

    // Additional optimizers
    // Audit-4 U.14: thread the Optimizer base + shared_ptr through the
    // pybind11 class template so isinstance(rmsprop, Optimizer) returns
    // True and the base class' param_groups / add_param_group / clip
    // config methods are visible on the Python instance (matches the
    // optim.pyi declaration ``class RMSprop(Optimizer)``).
    py::class_<tenzor::optim::RMSprop, tenzor::optim::Optimizer,
               std::shared_ptr<tenzor::optim::RMSprop>>(optim, "RMSprop")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double, bool>(),
             py::arg("params"), py::arg("lr") = 0.01, py::arg("alpha") = 0.99,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             py::arg("momentum") = 0.0, py::arg("centered") = false)
        .def("step", [](tenzor::optim::RMSprop& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            // X.8: scope the GIL release so `release` is destroyed (re-acquires
            // GIL) BEFORE `py::none()` is constructed in the return statement.
            // Constructing a py::object while the GIL is released triggers a
            // pybind11 inc_ref() assertion failure on CPython 3.14+.
            { py::gil_scoped_release release; self.step(); }
            return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::RMSprop::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("state_dict", &tenzor::optim::RMSprop::state_dict,
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::RMSprop::load_state_dict,
             py::call_guard<py::gil_scoped_release>());  // W.20

    // Audit-4 U.14: Adagrad inherits from Optimizer in the .pyi stub; thread
    // the base + shared_ptr through pybind11 so that contract is honoured.
    py::class_<tenzor::optim::Adagrad, tenzor::optim::Optimizer,
               std::shared_ptr<tenzor::optim::Adagrad>>(optim, "Adagrad")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 0.01, py::arg("lr_decay") = 0.0,
             py::arg("weight_decay") = 0.0, py::arg("initial_accumulator_value") = 0.0,
             py::arg("eps") = 1e-10)
        .def("step", [](tenzor::optim::Adagrad& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            // X.8: scope the GIL release so `release` is destroyed (re-acquires
            // GIL) BEFORE `py::none()` is constructed in the return statement.
            // Constructing a py::object while the GIL is released triggers a
            // pybind11 inc_ref() assertion failure on CPython 3.14+.
            { py::gil_scoped_release release; self.step(); }
            return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::Adagrad::zero_grad,
             py::call_guard<py::gil_scoped_release>());  // W.20

    // Audit-4 U.14: Adadelta — same parent + shared_ptr fix as RMSprop / Adagrad.
    py::class_<tenzor::optim::Adadelta, tenzor::optim::Optimizer,
               std::shared_ptr<tenzor::optim::Adadelta>>(optim, "Adadelta")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 1.0, py::arg("rho") = 0.9,
             py::arg("eps") = 1e-6, py::arg("weight_decay") = 0.0)
        .def("step", [](tenzor::optim::Adadelta& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            // X.8: scope the GIL release so `release` is destroyed (re-acquires
            // GIL) BEFORE `py::none()` is constructed in the return statement.
            // Constructing a py::object while the GIL is released triggers a
            // pybind11 inc_ref() assertion failure on CPython 3.14+.
            { py::gil_scoped_release release; self.step(); }
            return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::Adadelta::zero_grad,
             py::call_guard<py::gil_scoped_release>());  // W.20

    py::class_<tenzor::optim::RAdam, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::RAdam>>(optim, "RAdam",
        "Rectified Adam optimizer (no warmup needed)")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0)
        .def("step", [](tenzor::optim::RAdam& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            // X.8: scope the GIL release so `release` is destroyed (re-acquires
            // GIL) BEFORE `py::none()` is constructed in the return statement.
            // Constructing a py::object while the GIL is released triggers a
            // pybind11 inc_ref() assertion failure on CPython 3.14+.
            { py::gil_scoped_release release; self.step(); }
            return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::RAdam::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("set_lr", &tenzor::optim::RAdam::set_lr, py::arg("lr"))
        .def("get_lr", &tenzor::optim::RAdam::get_lr)
        .def("state_dict", &tenzor::optim::RAdam::state_dict,
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::RAdam::load_state_dict, py::arg("state"),
             py::call_guard<py::gil_scoped_release>());  // W.20

    py::class_<tenzor::optim::NAdam, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::NAdam>>(optim, "NAdam",
        "NAdam (Nesterov-accelerated Adam) optimizer")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 2e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             py::arg("momentum_decay") = 4e-3)
        .def("step", [](tenzor::optim::NAdam& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            // X.8: scope the GIL release so `release` is destroyed (re-acquires
            // GIL) BEFORE `py::none()` is constructed in the return statement.
            // Constructing a py::object while the GIL is released triggers a
            // pybind11 inc_ref() assertion failure on CPython 3.14+.
            { py::gil_scoped_release release; self.step(); }
            return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::NAdam::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("set_lr", &tenzor::optim::NAdam::set_lr, py::arg("lr"))
        .def("get_lr", &tenzor::optim::NAdam::get_lr)
        .def("state_dict", &tenzor::optim::NAdam::state_dict,
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::NAdam::load_state_dict, py::arg("state"),
             py::call_guard<py::gil_scoped_release>());  // W.20

    py::class_<tenzor::optim::Adamax, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::Adamax>>(optim, "Adamax",
        "Adamax optimizer (Adam variant based on infinity norm)")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 2e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0)
        .def("step", [](tenzor::optim::Adamax& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            // X.8: scope the GIL release so `release` is destroyed (re-acquires
            // GIL) BEFORE `py::none()` is constructed in the return statement.
            // Constructing a py::object while the GIL is released triggers a
            // pybind11 inc_ref() assertion failure on CPython 3.14+.
            { py::gil_scoped_release release; self.step(); }
            return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::Adamax::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("set_lr", &tenzor::optim::Adamax::set_lr, py::arg("lr"))
        .def("get_lr", &tenzor::optim::Adamax::get_lr)
        .def("state_dict", &tenzor::optim::Adamax::state_dict,
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::Adamax::load_state_dict, py::arg("state"),
             py::call_guard<py::gil_scoped_release>());  // W.20

    py::class_<tenzor::optim::LAMB, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::LAMB>>(optim, "LAMB",
        "LAMB optimizer for large-batch training")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-6, py::arg("weight_decay") = 0.01)
        .def("step", [](tenzor::optim::LAMB& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            // X.8: scope the GIL release so `release` is destroyed (re-acquires
            // GIL) BEFORE `py::none()` is constructed in the return statement.
            // Constructing a py::object while the GIL is released triggers a
            // pybind11 inc_ref() assertion failure on CPython 3.14+.
            { py::gil_scoped_release release; self.step(); }
            return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::LAMB::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("set_lr", &tenzor::optim::LAMB::set_lr, py::arg("lr"))
        .def("get_lr", &tenzor::optim::LAMB::get_lr)
        .def("state_dict", &tenzor::optim::LAMB::state_dict,
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::LAMB::load_state_dict, py::arg("state"),
             py::call_guard<py::gil_scoped_release>());  // W.20

    py::class_<tenzor::optim::SparseAdam, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::SparseAdam>>(optim, "SparseAdam",
        "SparseAdam optimizer for efficient embedding training with sparse gradients")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 1e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8)
        .def("step", [](tenzor::optim::SparseAdam& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            // X.8: scope the GIL release so `release` is destroyed (re-acquires
            // GIL) BEFORE `py::none()` is constructed in the return statement.
            // Constructing a py::object while the GIL is released triggers a
            // pybind11 inc_ref() assertion failure on CPython 3.14+.
            { py::gil_scoped_release release; self.step(); }
            return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::SparseAdam::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("set_lr", &tenzor::optim::SparseAdam::set_lr, py::arg("lr"))
        .def("get_lr", &tenzor::optim::SparseAdam::get_lr)
        .def("state_dict", &tenzor::optim::SparseAdam::state_dict,
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::SparseAdam::load_state_dict, py::arg("state"),
             py::call_guard<py::gil_scoped_release>());  // W.20

    py::class_<tenzor::optim::Rprop, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::Rprop>>(optim, "Rprop",
        "Resilient Propagation optimizer with per-parameter adaptive step sizes")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 0.01,
             py::arg("eta_minus") = 0.5, py::arg("eta_plus") = 1.2,
             py::arg("step_min") = 1e-6, py::arg("step_max") = 50.0)
        .def("step", [](tenzor::optim::Rprop& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            // X.8: scope the GIL release so `release` is destroyed (re-acquires
            // GIL) BEFORE `py::none()` is constructed in the return statement.
            // Constructing a py::object while the GIL is released triggers a
            // pybind11 inc_ref() assertion failure on CPython 3.14+.
            { py::gil_scoped_release release; self.step(); }
            return py::none();
        }, py::arg("closure") = py::none(),
           "Perform optimization step. Optionally takes a closure that recomputes the loss.")
        .def("zero_grad", &tenzor::optim::Rprop::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("set_lr", &tenzor::optim::Rprop::set_lr,
             py::arg("lr"), "Set learning rate (initial step size)")
        .def("get_lr", &tenzor::optim::Rprop::get_lr,
             "Get current learning rate")
        .def("state_dict", &tenzor::optim::Rprop::state_dict,
             "Get optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::Rprop::load_state_dict,
             py::arg("state"), "Load optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>());  // W.20

    // LBFGS: quasi-Newton optimizer for small, precisely-convergent problems.
    // Requires a closure because line search needs to re-evaluate f + grad.
    py::enum_<tenzor::optim::LBFGSLineSearch>(optim, "LBFGSLineSearch",
        "Line-search strategy for LBFGS optimizer")
        .value("Armijo", tenzor::optim::LBFGSLineSearch::Armijo)
        .value("StrongWolfe", tenzor::optim::LBFGSLineSearch::StrongWolfe)
        .export_values();

    py::class_<tenzor::optim::LBFGS, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::LBFGS>>(optim, "LBFGS",
        "Limited-memory BFGS optimizer (quasi-Newton with history size)")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, int, int, double, double, int, tenzor::optim::LBFGSLineSearch>(),
             py::arg("params"),
             py::arg("lr") = 1.0,
             py::arg("max_iter") = 20,
             py::arg("max_eval") = -1,
             py::arg("tolerance_grad") = 1e-7,
             py::arg("tolerance_change") = 1e-9,
             py::arg("history_size") = 100,
             py::arg("line_search") = tenzor::optim::LBFGSLineSearch::StrongWolfe)
        .def("step", [](tenzor::optim::LBFGS& self, std::function<tenzor::Variable()> closure) {
            // Q.15: release GIL across the step. The closure callback (which
            // runs Python) re-acquires it automatically when pybind11
            // dispatches the std::function call.
            py::gil_scoped_release release;
            return self.step(closure);
        }, py::arg("closure"),
           "Perform an L-BFGS step. Closure must recompute loss and call loss.backward().")
        .def("zero_grad", &tenzor::optim::LBFGS::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("set_lr", &tenzor::optim::LBFGS::set_lr, py::arg("lr"))
        .def("get_lr", &tenzor::optim::LBFGS::get_lr)
        .def("state_dict", &tenzor::optim::LBFGS::state_dict,
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::LBFGS::load_state_dict, py::arg("state"),
             py::call_guard<py::gil_scoped_release>());  // W.20

    // --- ZeRO optimizers (stages 1, 2, 3) ---
    // C++ now exposes a `std::shared_ptr<Optimizer>` overload on each ZeRO
    // constructor (alongside the original unique_ptr one). The Python binding
    // routes through the shared_ptr form, which lets pybind11 keep its own
    // shared_ptr reference alive concurrently with the ZeRO wrapper's. No
    // ownership transfer required; no risk of double-free.
    py::class_<tenzor::optim::ZeROStage1Config>(optim, "ZeROStage1Config",
        "Configuration for ZeROStage1Optimizer")
        .def(py::init<>())
        .def_readwrite("world_size", &tenzor::optim::ZeROStage1Config::world_size)
        .def_readwrite("rank", &tenzor::optim::ZeROStage1Config::rank)
        .def_readwrite("offload_to_cpu", &tenzor::optim::ZeROStage1Config::offload_to_cpu)
        .def_readwrite("cpu_offload_threshold", &tenzor::optim::ZeROStage1Config::cpu_offload_threshold)
        .def_readwrite("overlap_comm", &tenzor::optim::ZeROStage1Config::overlap_comm)
        .def_readwrite("pin_memory", &tenzor::optim::ZeROStage1Config::pin_memory);

    py::class_<tenzor::optim::ZeROStage2Config, tenzor::optim::ZeROStage1Config>(
        optim, "ZeROStage2Config", "Configuration for ZeROStage2Optimizer")
        .def(py::init<>())
        .def_readwrite("gradient_bucket_size", &tenzor::optim::ZeROStage2Config::gradient_bucket_size)
        .def_readwrite("reduce_scatter_in_backward", &tenzor::optim::ZeROStage2Config::reduce_scatter_in_backward)
        .def_readwrite("gradient_bucketing", &tenzor::optim::ZeROStage2Config::gradient_bucketing);

    py::class_<tenzor::optim::Stage3Config, tenzor::optim::ZeROStage2Config>(
        optim, "ZeROStage3Config", "Configuration for ZeROStage3Optimizer")
        .def(py::init<>());

    // ZeRO Stage 1 — safe shared_ptr path. Python can still hold a separate
    // reference to the base optimizer after wrapping (e.g. to read its state
    // dict); both references share ownership via the C++ shared_ptr field.
    py::class_<tenzor::optim::ZeROStage1Optimizer, std::shared_ptr<tenzor::optim::ZeROStage1Optimizer>>(
        optim, "ZeROStage1Optimizer", "ZeRO Stage 1: optimizer state partitioning")
        .def(py::init<std::shared_ptr<tenzor::optim::Optimizer>,
                      const tenzor::optim::ZeROStage1Config&>(),
             py::arg("base_optimizer"), py::arg("config"))
        .def("step", [](tenzor::optim::ZeROStage1Optimizer& self) {
            py::gil_scoped_release release;  // Q.15
            self.step();
        })
        .def("zero_grad", &tenzor::optim::ZeROStage1Optimizer::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("state_dict", &tenzor::optim::ZeROStage1Optimizer::state_dict,
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::ZeROStage1Optimizer::load_state_dict,
             py::arg("state"),
             py::call_guard<py::gil_scoped_release>());  // W.20

    // Stage 2 and 3 are aliased to Stage 1's binding for construction — they
    // share the same `base_optimizer + config` signature, only the config
    // type differs. The class is exposed so Python can type-check the handle.
    py::class_<tenzor::optim::ZeROStage2Optimizer, tenzor::optim::ZeROStage1Optimizer,
               std::shared_ptr<tenzor::optim::ZeROStage2Optimizer>>(
        optim, "ZeROStage2Optimizer", "ZeRO Stage 2: gradient + optimizer state partitioning")
        .def(py::init<std::shared_ptr<tenzor::optim::Optimizer>,
                      const tenzor::optim::ZeROStage2Config&>(),
             py::arg("base_optimizer"), py::arg("config"));

    py::class_<tenzor::optim::ZeROStage3Optimizer, tenzor::optim::ZeROStage2Optimizer,
               std::shared_ptr<tenzor::optim::ZeROStage3Optimizer>>(
        optim, "ZeROStage3Optimizer", "ZeRO Stage 3: parameter + gradient + optimizer state partitioning")
        .def(py::init<std::shared_ptr<tenzor::optim::Optimizer>,
                      const tenzor::optim::Stage3Config&>(),
             py::arg("base_optimizer"), py::arg("config"));

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

    // MultiStepLR + LambdaLR — audit item E.11.
    // Both classes already exist in C++ (scheduler.hpp + scheduler_advanced.cpp);
    // only the Python bindings were missing.
    py::class_<tenzor::optim::MultiStepLR, tenzor::optim::LRScheduler>(lr_scheduler, "MultiStepLR")
        .def(py::init<tenzor::optim::SGD&, std::vector<int>, double>(),
             py::arg("optimizer"), py::arg("milestones"), py::arg("gamma") = 0.1,
             "Decays learning rate by gamma at each epoch in `milestones`")
        .def(py::init<tenzor::optim::Adam&, std::vector<int>, double>(),
             py::arg("optimizer"), py::arg("milestones"), py::arg("gamma") = 0.1)
        .def(py::init<tenzor::optim::AdamW&, std::vector<int>, double>(),
             py::arg("optimizer"), py::arg("milestones"), py::arg("gamma") = 0.1)
        .def(py::init<tenzor::optim::RMSprop&, std::vector<int>, double>(),
             py::arg("optimizer"), py::arg("milestones"), py::arg("gamma") = 0.1)
        .def(py::init<tenzor::optim::Adagrad&, std::vector<int>, double>(),
             py::arg("optimizer"), py::arg("milestones"), py::arg("gamma") = 0.1)
        .def(py::init<tenzor::optim::Adadelta&, std::vector<int>, double>(),
             py::arg("optimizer"), py::arg("milestones"), py::arg("gamma") = 0.1)
        .def("get_epoch", &tenzor::optim::MultiStepLR::get_epoch,
             "Get current epoch number");

    py::class_<tenzor::optim::LambdaLR, tenzor::optim::LRScheduler>(lr_scheduler, "LambdaLR")
        .def(py::init([](tenzor::optim::SGD& opt, py::function lr_lambda, std::string name) {
                 return new tenzor::optim::LambdaLR(opt,
                     [lr_lambda](int epoch) -> double {
                         py::gil_scoped_acquire gil;
                         return lr_lambda(epoch).cast<double>();
                     }, std::move(name));
             }), py::arg("optimizer"), py::arg("lr_lambda"), py::arg("name") = std::string{},
             "Sets lr_t = base_lr * lr_lambda(epoch). Audit-4 W.12: pass name to "
             "guard against loading a checkpoint produced by a different lambda.")
        .def(py::init([](tenzor::optim::Adam& opt, py::function lr_lambda, std::string name) {
                 return new tenzor::optim::LambdaLR(opt,
                     [lr_lambda](int epoch) -> double {
                         py::gil_scoped_acquire gil;
                         return lr_lambda(epoch).cast<double>();
                     }, std::move(name));
             }), py::arg("optimizer"), py::arg("lr_lambda"), py::arg("name") = std::string{})
        .def(py::init([](tenzor::optim::AdamW& opt, py::function lr_lambda, std::string name) {
                 return new tenzor::optim::LambdaLR(opt,
                     [lr_lambda](int epoch) -> double {
                         py::gil_scoped_acquire gil;
                         return lr_lambda(epoch).cast<double>();
                     }, std::move(name));
             }), py::arg("optimizer"), py::arg("lr_lambda"), py::arg("name") = std::string{})
        .def(py::init([](tenzor::optim::RMSprop& opt, py::function lr_lambda, std::string name) {
                 return new tenzor::optim::LambdaLR(opt,
                     [lr_lambda](int epoch) -> double {
                         py::gil_scoped_acquire gil;
                         return lr_lambda(epoch).cast<double>();
                     }, std::move(name));
             }), py::arg("optimizer"), py::arg("lr_lambda"), py::arg("name") = std::string{})
        .def("get_epoch", &tenzor::optim::LambdaLR::get_epoch,
             "Get current epoch number")
        // Audit-4 W.12: expose load_state_dict(force=...) for callers who
        // want to bypass the saved lambda-name guard.
        .def("load_state_dict",
             [](tenzor::optim::LambdaLR& self,
                const std::unordered_map<std::string, tenzor::Tensor>& state,
                bool force) {
                 self.load_state_dict(state, force);
             },
             py::arg("state"), py::arg("force") = false,
             "Restore scheduler state. With force=True the saved "
             "lambda-name guard is skipped (audit-4 W.12).")
        .def("name", &tenzor::optim::LambdaLR::name,
             "Lambda identifier set at construction (empty if not configured).");

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

    // ConstantLR scheduler
    py::class_<tenzor::optim::ConstantLR, tenzor::optim::LRScheduler>(lr_scheduler, "ConstantLR")
        .def(py::init<tenzor::optim::Optimizer&, double, int>(),
             py::arg("optimizer"), py::arg("factor") = 1.0 / 3.0,
             py::arg("total_iters") = 5)
        .def("step", &tenzor::optim::ConstantLR::step)
        .def("get_last_lr", &tenzor::optim::ConstantLR::get_last_lr);

    // LinearLR scheduler
    py::class_<tenzor::optim::LinearLR, tenzor::optim::LRScheduler>(lr_scheduler, "LinearLR")
        .def(py::init<tenzor::optim::Optimizer&, double, double, int>(),
             py::arg("optimizer"), py::arg("start_factor") = 1.0 / 3.0,
             py::arg("end_factor") = 1.0, py::arg("total_iters") = 5)
        .def("step", &tenzor::optim::LinearLR::step)
        .def("get_last_lr", &tenzor::optim::LinearLR::get_last_lr);

    // MultiplicativeLR scheduler
    py::class_<tenzor::optim::MultiplicativeLR, tenzor::optim::LRScheduler>(lr_scheduler, "MultiplicativeLR")
        .def(py::init<tenzor::optim::Optimizer&, std::function<double(int)>, std::string>(),
             py::arg("optimizer"), py::arg("lr_lambda"),
             py::arg("name") = std::string{},
             "Audit-4 W.12: pass name to guard against loading a checkpoint "
             "produced by a different lambda.")
        .def("step", &tenzor::optim::MultiplicativeLR::step)
        .def("get_last_lr", &tenzor::optim::MultiplicativeLR::get_last_lr)
        // Audit-4 W.12: expose load_state_dict(force=...).
        .def("load_state_dict",
             [](tenzor::optim::MultiplicativeLR& self,
                const std::unordered_map<std::string, tenzor::Tensor>& state,
                bool force) {
                 self.load_state_dict(state, force);
             },
             py::arg("state"), py::arg("force") = false,
             "Restore scheduler state. With force=True the saved "
             "lambda-name guard is skipped.")
        .def("name", &tenzor::optim::MultiplicativeLR::name,
             "Lambda identifier set at construction.");

    // SequentialLR scheduler
    py::class_<tenzor::optim::SequentialLR, tenzor::optim::LRScheduler>(lr_scheduler, "SequentialLR")
        .def(py::init<tenzor::optim::Optimizer&,
                       std::vector<std::shared_ptr<tenzor::optim::LRScheduler>>,
                       std::vector<int>>(),
             py::arg("optimizer"), py::arg("schedulers"), py::arg("milestones"))
        .def("step", &tenzor::optim::SequentialLR::step)
        .def("get_last_lr", &tenzor::optim::SequentialLR::get_last_lr);

    // ChainedScheduler
    py::class_<tenzor::optim::ChainedScheduler, tenzor::optim::LRScheduler>(lr_scheduler, "ChainedScheduler")
        .def(py::init<std::vector<std::shared_ptr<tenzor::optim::LRScheduler>>>(),
             py::arg("schedulers"))
        .def("step", &tenzor::optim::ChainedScheduler::step)
        .def("get_last_lr", &tenzor::optim::ChainedScheduler::get_last_lr);

    // SAM (Sharpness-Aware Minimization) Optimizer
    py::class_<tenzor::optim::SAM, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::SAM>>(optim, "SAM",
        "Sharpness-Aware Minimization optimizer wrapper")
        .def(py::init<std::shared_ptr<tenzor::optim::Optimizer>, double>(),
             py::arg("base_optimizer"), py::arg("rho") = 0.05,
             "Construct SAM wrapping a base optimizer")
        .def("first_step", &tenzor::optim::SAM::first_step,
             "Compute perturbation and apply to weights")
        .def("second_step", &tenzor::optim::SAM::second_step,
             "Restore original weights and step the base optimizer")
        .def("zero_grad", &tenzor::optim::SAM::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("set_lr", &tenzor::optim::SAM::set_lr,
             py::arg("lr"), "Set learning rate on the base optimizer")
        .def("get_lr", &tenzor::optim::SAM::get_lr,
             "Get current learning rate from the base optimizer")
        .def("get_rho", &tenzor::optim::SAM::get_rho,
             "Get perturbation radius")
        .def("set_rho", &tenzor::optim::SAM::set_rho,
             py::arg("rho"), "Set perturbation radius")
        .def("state_dict", &tenzor::optim::SAM::state_dict,
             "Get optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::SAM::load_state_dict,
             py::arg("state"), "Load optimizer state dictionary",
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("base_optimizer",
             static_cast<tenzor::optim::Optimizer& (tenzor::optim::SAM::*)()>(
                 &tenzor::optim::SAM::base_optimizer),
             py::return_value_policy::reference_internal,
             "Get reference to the base optimizer");

    // SWA (Stochastic Weight Averaging) - AveragedModel
    py::class_<tenzor::optim::AveragedModel>(optim, "AveragedModel",
        "Maintains a running average of model parameters for SWA")
        .def(py::init<const std::vector<std::shared_ptr<tenzor::Variable>>&>(),
             py::arg("params"),
             "Initialize averaged model from current parameters")
        .def("update_parameters", &tenzor::optim::AveragedModel::update_parameters,
             py::arg("params"),
             "Update running average with current model parameters")
        .def("apply_to", &tenzor::optim::AveragedModel::apply_to,
             py::arg("params"),
             "Copy averaged parameters back to model parameters")
        .def("n_averaged", &tenzor::optim::AveragedModel::n_averaged,
             "Get number of models averaged so far");

    // SWALR scheduler (in the lr_scheduler submodule)
    py::class_<tenzor::optim::SWALR, tenzor::optim::LRScheduler>(lr_scheduler, "SWALR",
        "SWA learning rate scheduler - anneals to constant swa_lr")
        .def(py::init<tenzor::optim::Optimizer&, double, int, const std::string&>(),
             py::arg("optimizer"), py::arg("swa_lr"),
             py::arg("anneal_epochs") = 10,
             py::arg("anneal_strategy") = "linear",
             "Anneal learning rate to swa_lr over anneal_epochs, then hold constant")
        .def("step", &tenzor::optim::SWALR::step)
        .def("get_last_lr", &tenzor::optim::SWALR::get_last_lr);

    // Adam-atan2 Optimizer
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
        .def("zero_grad", &tenzor::optim::AdamAtan2::zero_grad,
             py::call_guard<py::gil_scoped_release>())  // W.20
        .def("set_lr", &tenzor::optim::AdamAtan2::set_lr)
        .def("get_lr", &tenzor::optim::AdamAtan2::get_lr)
        .def("state_dict", &tenzor::optim::AdamAtan2::state_dict,
             py::call_guard<py::gil_scoped_release>())  // CC.24
        .def("load_state_dict", &tenzor::optim::AdamAtan2::load_state_dict,
             py::call_guard<py::gil_scoped_release>());  // W.20
}

} // namespace tenzor::python
