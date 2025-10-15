/**
 * @file data_parallel_example.cpp
 * @brief Example demonstrating DataParallel multi-GPU gradient synchronization
 *
 * This example shows how to use DataParallel for multi-GPU training with
 * proper gradient synchronization.
 */

#include <iostream>
#include <memory>
#include <vector>
#include "tenzor/nn/parallel/data_parallel.hpp"
#include "tenzor/nn/modules/linear.hpp"
#include "tenzor/nn/modules/sequential.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"

using namespace tenzor;
using namespace tenzor::nn;

/**
 * @brief Simple two-layer neural network
 */
class SimpleNet : public Module {
public:
    SimpleNet(int input_size, int hidden_size, int output_size) {
        fc1_ = std::make_shared<Linear>(input_size, hidden_size);
        fc2_ = std::make_shared<Linear>(hidden_size, output_size);

        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward(const Variable& input) -> Variable override {
        auto h = fc1_->forward(input);

        // ReLU activation
        auto h_tensor = h.tensor();
        auto zero = Tensor::zeros_like(h_tensor);
        auto h_relu_tensor = Tensor::where(h_tensor > zero, h_tensor, zero);
        Variable h_relu(h_relu_tensor, h.requires_grad());

        return fc2_->forward(h_relu);
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
};

int main() {
    std::cout << "DataParallel Multi-GPU Training Example\n";
    std::cout << "========================================\n\n";

#ifndef TENZOR_USE_CUDA
    std::cout << "Error: CUDA support not enabled. Please compile with -DTENZOR_USE_CUDA=ON\n";
    return 1;
#else

    // Check GPU availability
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cout << "Error: No CUDA devices found\n";
        return 1;
    }

    std::cout << "Found " << device_count << " CUDA device(s)\n\n";

    // Configuration
    const int input_size = 784;    // e.g., MNIST flattened 28x28
    const int hidden_size = 256;
    const int output_size = 10;    // 10 classes
    const int batch_size = 64;     // Total batch size
    const int num_epochs = 5;
    const float learning_rate = 0.01f;

    // Determine which GPUs to use
    std::vector<int> device_ids;
    int num_gpus = std::min(device_count, 4);  // Use up to 4 GPUs
    for (int i = 0; i < num_gpus; ++i) {
        device_ids.push_back(i);
    }

    std::cout << "Using " << num_gpus << " GPU(s): ";
    for (int id : device_ids) {
        std::cout << id << " ";
    }
    std::cout << "\n\n";

    // Create model
    auto model = std::make_shared<SimpleNet>(input_size, hidden_size, output_size);

    // Wrap with DataParallel
    auto parallel_model = std::make_shared<DataParallel>(
        model,
        device_ids,
        0  // Master GPU
    );

    std::cout << "Model wrapped with DataParallel\n";
    std::cout << "Master device: GPU " << parallel_model->output_device() << "\n";
    std::cout << "Batch dimension: " << parallel_model->batch_dim() << "\n\n";

    // Create optimizer
    auto params = parallel_model->parameters();
    auto optimizer = std::make_shared<SGD>(params, learning_rate);

    std::cout << "Starting training...\n\n";

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << "\n";

        // Simulate training on a few batches
        const int num_batches = 10;
        float epoch_loss = 0.0f;

        for (int batch = 0; batch < num_batches; ++batch) {
            // Create random input batch on master GPU
            Tensor input_tensor({batch_size, input_size}, DType::Float32, Device::cuda(0));
            input_tensor.fill_(0.5f);  // Dummy data
            Variable input(input_tensor, true);

            // Create random target batch on master GPU
            Tensor target_tensor({batch_size, output_size}, DType::Float32, Device::cuda(0));
            target_tensor.fill_(0.1f);  // Dummy labels
            Variable target(target_tensor, false);

            // Zero gradients
            optimizer->zero_grad();

            // Forward pass
            // DataParallel automatically:
            // 1. Splits batch across GPUs (batch_size / num_gpus per GPU)
            // 2. Replicates model to each GPU
            // 3. Runs forward pass in parallel
            // 4. Gathers outputs back to master GPU
            auto output = parallel_model->forward(input);

            // Compute loss (simple MSE)
            auto diff = output.tensor() - target.tensor();
            auto squared = diff * diff;
            auto loss_tensor = squared.sum() / static_cast<float>(batch_size);
            Variable loss(loss_tensor, true);

            // Get loss value
            float loss_value = loss.tensor().item<float>();
            epoch_loss += loss_value;

            // Backward pass
            // Computes gradients on each GPU independently
            loss.backward();

            // Synchronize gradients across GPUs
            // This performs all-reduce:
            // 1. Gathers gradients from all GPUs to master
            // 2. Averages them: grad = sum(grads) / num_gpus
            // 3. Broadcasts averaged gradient back to all GPUs
            parallel_model->synchronize_gradients();

            // Update parameters
            // Optimizer updates master model's parameters
            optimizer->step();

            if (batch % 5 == 0) {
                std::cout << "  Batch " << batch << ", Loss: " << loss_value << "\n";
            }
        }

        float avg_loss = epoch_loss / num_batches;
        std::cout << "  Average loss: " << avg_loss << "\n\n";
    }

    std::cout << "Training completed successfully!\n\n";

    // Demonstrate model evaluation
    std::cout << "Switching to evaluation mode...\n";
    parallel_model->eval();

    // Create test batch
    Tensor test_input({batch_size, input_size}, DType::Float32, Device::cuda(0));
    test_input.fill_(0.5f);
    Variable test_var(test_input, false);  // No gradients needed

    // Forward pass (no gradient tracking)
    auto test_output = parallel_model->forward(test_var);

    std::cout << "Inference output shape: [";
    auto out_shape = test_output.shape();
    for (size_t i = 0; i < out_shape.size(); ++i) {
        std::cout << out_shape[i];
        if (i < out_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "\nExample completed!\n";

    return 0;
#endif
}
