/**
 * @file tenzor.hpp
 * @brief Main header - includes all public Tenzor APIs
 *
 * **Tenzor** is a modern C++ deep learning library providing:
 * - Efficient tensor operations (CPU and CUDA)
 * - Automatic differentiation (autograd)
 * - Neural network building blocks (layers, activations, losses)
 * - Optimizers (SGD, Adam, AdamW)
 * - Model serialization
 * - Parallel computation
 *
 * ## Quick Start
 *
 * @code
 * #include <tenzor/tenzor.hpp>
 * using namespace tenzor;
 *
 * // Initialize library
 * initialize();
 *
 * // Create tensor
 * auto x = randn({32, 784});
 *
 * // Build model
 * class MLP : public nn::Module {
 *     nn::Linear fc1, fc2;
 * public:
 *     MLP() : fc1(784, 128), fc2(128, 10) {
 *         register_module("fc1", fc1);
 *         register_module("fc2", fc2);
 *     }
 *     auto forward(const Variable& x) -> Variable override {
 *         return fc2.forward(nn::relu(fc1.forward(x)));
 *     }
 * };
 *
 * // Train
 * auto model = MLP();
 * auto optimizer = optim::Adam(model.parameters(), 1e-3);
 * auto criterion = nn::CrossEntropyLoss();
 *
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     optimizer.zero_grad();
 *     auto output = model.forward(x);
 *     auto loss = criterion(output, targets);
 *     loss.backward();
 *     optimizer.step();
 * }
 *
 * // Cleanup
 * finalize();
 * @endcode
 *
 * ## Documentation Structure
 *
 * - **Core**: Tensor, Device, DType, Shape, Storage
 * - **Operations**: Creation, Math, Reduction, Transform, Indexing
 * - **Autograd**: Variable, Function, Engine, Computational Graph
 * - **Neural Networks**:
 *   - Layers: Linear, Conv2D, BatchNorm, Dropout, Pooling
 *   - Activations: ReLU, GELU, Sigmoid, Softmax, etc.
 *   - Losses: CrossEntropy, MSE, BCE, etc.
 *   - Optimizers: SGD, Adam, AdamW
 *   - Schedulers: StepLR, ExponentialLR, CosineAnnealingLR
 * - **Parallel**: Thread pool, parallel loops, atomic operations
 * - **Utilities**: Configuration, Logging, Error handling
 *
 * @see https://github.com/skreamz/Tenzor for full documentation
 *
 * @version 0.1.0
 * @author Lee Morton
 */

#pragma once

// Core tensor functionality
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/core/dtype.hpp"

// Tensor operations
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"

// Automatic differentiation
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/engine.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/gradcheck.hpp"

// Neural network modules
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/nn/layers/dropout.hpp"
#include "tenzor/nn/layers/pooling.hpp"
#include "tenzor/nn/layers/flatten.hpp"
#include "tenzor/nn/layers/padding.hpp"
#include "tenzor/nn/layers/upsample.hpp"
#include "tenzor/nn/layers/identity.hpp"
#include "tenzor/nn/layers/rnn.hpp"
#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/nn/layers/attention.hpp"
#include "tenzor/nn/layers/transformer.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/rprop.hpp"
#include "tenzor/nn/optim/asgd.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/optim/rmsprop.hpp"
#include "tenzor/nn/optim/adagrad.hpp"
#include "tenzor/nn/optim/adadelta.hpp"
#include "tenzor/nn/optim/sam.hpp"
#include "tenzor/nn/optim/swa.hpp"
#include "tenzor/nn/optim/scheduler.hpp"

// Pre-trained models
#include "tenzor/models/bert.hpp"
#include "tenzor/models/gpt.hpp"
#include "tenzor/models/roberta.hpp"
#include "tenzor/models/electra.hpp"
#include "tenzor/models/resnet.hpp"
#include "tenzor/models/convnext.hpp"
#include "tenzor/models/mobilenet.hpp"

// Detection and segmentation models
#include "tenzor/nn/detection/anchors.hpp"
#include "tenzor/nn/detection/roi_ops.hpp"
#include "tenzor/nn/detection/rpn.hpp"
#include "tenzor/nn/detection/roi_head.hpp"
#include "tenzor/models/faster_rcnn.hpp"

/**
 * @brief Main Tenzor namespace
 *
 * All library functionality resides in this namespace or sub-namespaces:
 * - tenzor::nn - Neural network components
 * - tenzor::optim - Optimization algorithms
 * - tenzor::autograd - Automatic differentiation
 */
namespace tenzor {
    /**
     * @brief Initialize Tenzor library
     *
     * Must be called before using library functions.
     * Initializes backends (CPU/CUDA), thread pools, and internal state.
     *
     * @code
     * int main() {
     *     tenzor::initialize();
     *     // ... use library ...
     *     tenzor::finalize();
     * }
     * @endcode
     */
    void initialize();

    /**
     * @brief Free MKL internal buffers to prevent conflicts with other MKL users
     *
     * Call this after a batch of Tenzor operations before using another library
     * that uses MKL (e.g., PyTorch, NumPy). This is only necessary when using
     * Tenzor alongside other MKL-based libraries in the same process.
     *
     * @code{.cpp}
     * // After Tenzor operations, before PyTorch
     * tenzor::mkl_cleanup();
     * @endcode
     */
    void mkl_cleanup();

    /**
     * @brief Finalize and cleanup Tenzor library
     *
     * Should be called before program termination.
     * Releases resources, shuts down thread pools, and cleanup backends.
     */
    void finalize();
}
