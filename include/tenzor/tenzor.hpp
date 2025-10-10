#pragma once

// Tenzor - Neural Network Library
// Convenience header including all public APIs

// Core
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/core/dtype.hpp"

// Operations
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"

// Autograd
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/engine.hpp"
#include "tenzor/autograd/ops.hpp"

// Neural Network Modules
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/nn/layers/dropout.hpp"
#include "tenzor/nn/layers/pooling.hpp"
#include "tenzor/nn/layers/flatten.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/adam.hpp"

// Initialization functions (from backend loader)
namespace tenzor {
    void initialize();
    void finalize();
}
