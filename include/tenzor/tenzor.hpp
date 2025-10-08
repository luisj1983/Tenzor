#pragma once

// Main header file that includes all tenzor components

// Core
#include "core/tensor.hpp"
#include "core/dtype.hpp"
#include "core/device.hpp"
#include "core/storage.hpp"
#include "core/shape.hpp"

// Operations
#include "ops/creation.hpp"
#include "ops/math.hpp"
#include "ops/reduction.hpp"
#include "ops/transform.hpp"
#include "ops/indexing.hpp"

// Autograd
#include "autograd/variable.hpp"
#include "autograd/function.hpp"
#include "autograd/graph.hpp"
#include "autograd/engine.hpp"

// Neural Network
#include "nn/module.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/conv.hpp"
#include "nn/layers/batchnorm.hpp"
#include "nn/layers/dropout.hpp"
#include "nn/activations/activations.hpp"
#include "nn/loss/losses.hpp"
#include "nn/optim/optimizer.hpp"
#include "nn/optim/sgd.hpp"
#include "nn/optim/adam.hpp"

// Backend
#include "backend/backend.hpp"
#include "backend/loader.hpp"
#include "backend/registry.hpp"
#include "backend/dispatch.hpp"

// Parallel
#include "parallel/threadpool.hpp"
#include "parallel/parallel_for.hpp"
#include "parallel/atomic.hpp"

// Utils
#include "utils/logging.hpp"
#include "utils/error.hpp"
#include "utils/config.hpp"

namespace tenzor {

// Version information
constexpr const char* VERSION = "1.0.0";
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

// Initialize Tenzor library
auto initialize() -> void;

// Cleanup Tenzor library
auto finalize() -> void;

} // namespace tenzor
