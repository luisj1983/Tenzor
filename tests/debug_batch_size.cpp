#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <iostream>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::optim;

// Minimal test to debug batch size scaling issue
TEST(DebugTest, BatchSizeScalingMinimal) {
    auto device = Device::cuda();

    // Simple 2-layer network
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(10, 5));
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<Linear>(5, 2));
    model->to(device);

    auto params = model->parameters();
    auto optimizer = Adam(params, 0.001);

    std::vector<int> batch_sizes = {16, 32};

    for (int batch_size : batch_sizes) {
        std::cout << "\n=== Testing batch size " << batch_size << " ===" << std::endl;

        // Create input and target
        auto input = Variable(randn({batch_size, 10}, DType::Float32, device), true);
        auto target = Variable(zeros({batch_size, 2}, DType::Float32, device), false);

        std::cout << "Input shape: [" << input.shape()[0] << ", " << input.shape()[1] << "]" << std::endl;
        std::cout << "Target shape: [" << target.shape()[0] << ", " << target.shape()[1] << "]" << std::endl;

        // Forward pass
        std::cout << "Forward pass..." << std::endl;
        auto output = model->forward(input);
        std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]" << std::endl;

        // Loss
        std::cout << "Computing loss..." << std::endl;
        auto loss = mse_loss(output, target, Reduction::Mean);
        std::cout << "Loss shape: " << loss.shape().size() << "D" << std::endl;

        // Backward
        std::cout << "Zero grad..." << std::endl;
        optimizer.zero_grad();

        std::cout << "Backward pass..." << std::endl;
        loss.backward(ones({1}, DType::Float32, device));

        // Check gradient shapes before optimizer step
        std::cout << "Checking gradients..." << std::endl;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->has_grad()) {
                auto grad_shape = params[i]->grad()->shape();
                auto param_shape = params[i]->shape();
                std::cout << "Param " << i << " shape: [";
                for (size_t j = 0; j < param_shape.size(); ++j) {
                    std::cout << param_shape[j];
                    if (j < param_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "], grad shape: [";
                for (size_t j = 0; j < grad_shape.size(); ++j) {
                    std::cout << grad_shape[j];
                    if (j < grad_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
        }

        // Optimizer step
        std::cout << "Optimizer step..." << std::endl;
        try {
            optimizer.step();
            std::cout << "✓ Batch size " << batch_size << " completed successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "✗ Error during optimizer step: " << e.what() << std::endl;
            throw;
        }
    }
}

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
