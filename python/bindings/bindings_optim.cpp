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
#include <tenzor/nn/optim/asgd.hpp>

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

    py::class_<tenzor::optim::ASGD, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::ASGD>>(optim, "ASGD",
        "Averaged Stochastic Gradient Descent optimizer")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 0.01,
             py::arg("lambd") = 1e-4, py::arg("alpha") = 0.75,
             py::arg("t0") = 1e6, py::arg("weight_decay") = 0.0)
        .def("step", [](tenzor::optim::ASGD& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            self.step(); return py::none();
        }, py::arg("closure") = py::none(),
           "Perform optimization step. Optionally takes a closure that recomputes the loss.")
        .def("zero_grad", &tenzor::optim::ASGD::zero_grad)
        .def("set_lr", &tenzor::optim::ASGD::set_lr,
             py::arg("lr"), "Set learning rate")
        .def("get_lr", &tenzor::optim::ASGD::get_lr,
             "Get current learning rate")
        .def("state_dict", &tenzor::optim::ASGD::state_dict,
             "Get optimizer state dictionary")
        .def("load_state_dict", &tenzor::optim::ASGD::load_state_dict,
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

    py::class_<tenzor::optim::NAdam, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::NAdam>>(optim, "NAdam",
        "NAdam (Nesterov-accelerated Adam) optimizer")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 2e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
             py::arg("momentum_decay") = 4e-3)
        .def("step", [](tenzor::optim::NAdam& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            self.step(); return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::NAdam::zero_grad)
        .def("set_lr", &tenzor::optim::NAdam::set_lr, py::arg("lr"))
        .def("get_lr", &tenzor::optim::NAdam::get_lr)
        .def("state_dict", &tenzor::optim::NAdam::state_dict)
        .def("load_state_dict", &tenzor::optim::NAdam::load_state_dict, py::arg("state"));

    py::class_<tenzor::optim::Adamax, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::Adamax>>(optim, "Adamax",
        "Adamax optimizer (Adam variant based on infinity norm)")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 2e-3,
             py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
             py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0)
        .def("step", [](tenzor::optim::Adamax& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            self.step(); return py::none();
        }, py::arg("closure") = py::none())
        .def("zero_grad", &tenzor::optim::Adamax::zero_grad)
        .def("set_lr", &tenzor::optim::Adamax::set_lr, py::arg("lr"))
        .def("get_lr", &tenzor::optim::Adamax::get_lr)
        .def("state_dict", &tenzor::optim::Adamax::state_dict)
        .def("load_state_dict", &tenzor::optim::Adamax::load_state_dict, py::arg("state"));

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

    py::class_<tenzor::optim::Rprop, tenzor::optim::Optimizer, std::shared_ptr<tenzor::optim::Rprop>>(optim, "Rprop",
        "Resilient Propagation optimizer with per-parameter adaptive step sizes")
        .def(py::init<std::vector<std::shared_ptr<tenzor::Variable>>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 0.01,
             py::arg("eta_minus") = 0.5, py::arg("eta_plus") = 1.2,
             py::arg("step_min") = 1e-6, py::arg("step_max") = 50.0)
        .def("step", [](tenzor::optim::Rprop& self, std::optional<std::function<tenzor::Variable()>> closure) -> py::object {
            if (closure) return py::cast(self.step(*closure));
            self.step(); return py::none();
        }, py::arg("closure") = py::none(),
           "Perform optimization step. Optionally takes a closure that recomputes the loss.")
        .def("zero_grad", &tenzor::optim::Rprop::zero_grad)
        .def("set_lr", &tenzor::optim::Rprop::set_lr,
             py::arg("lr"), "Set learning rate (initial step size)")
        .def("get_lr", &tenzor::optim::Rprop::get_lr,
             "Get current learning rate")
        .def("state_dict", &tenzor::optim::Rprop::state_dict,
             "Get optimizer state dictionary")
        .def("load_state_dict", &tenzor::optim::Rprop::load_state_dict,
             py::arg("state"), "Load optimizer state dictionary");

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
        .def(py::init<tenzor::optim::Optimizer&, std::function<double(int)>>(),
             py::arg("optimizer"), py::arg("lr_lambda"))
        .def("step", &tenzor::optim::MultiplicativeLR::step)
        .def("get_last_lr", &tenzor::optim::MultiplicativeLR::get_last_lr);

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
        .def("zero_grad", &tenzor::optim::AdamAtan2::zero_grad)
        .def("set_lr", &tenzor::optim::AdamAtan2::set_lr)
        .def("get_lr", &tenzor::optim::AdamAtan2::get_lr)
        .def("state_dict", &tenzor::optim::AdamAtan2::state_dict)
        .def("load_state_dict", &tenzor::optim::AdamAtan2::load_state_dict);
}

} // namespace tenzor::python
