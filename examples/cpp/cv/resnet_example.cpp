/**
 * @file resnet_example.cpp
 * @brief Example usage of ResNet models for image classification
 *
 * Demonstrates:
 * - Creating different ResNet variants
 * - Training on custom datasets
 * - Transfer learning with pretrained weights
 * - Inference and evaluation
 */

#include <iostream>
#include <memory>
#include <vector>
#include "../../include/tenzor/models/resnet.hpp"
#include "../../include/tenzor/nn/loss/losses.hpp"
#include "../../include/tenzor/nn/optim/sgd.hpp"
#include "../../include/tenzor/nn/optim/adam.hpp"
#include "../../include/tenzor/core/tensor.hpp"
#include "../../include/tenzor/autograd/variable.hpp"

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::nn;

/**
 * @brief Example 1: Create and inspect ResNet models
 */
void example_create_models() {
    std::cout << "=== Example 1: Creating ResNet Models ===\n\n";

    // Create ResNet-18
    auto resnet18_model = resnet18(1000, false);
    std::cout << "Created ResNet-18 for ImageNet (1000 classes)\n";

    auto params18 = resnet18_model->parameters();
    int64_t total_params18 = 0;
    for (const auto& p : params18) {
        total_params18 += p->data().numel();
    }
    std::cout << "  Total parameters: " << total_params18 / 1e6 << "M\n\n";

    // Create ResNet-50
    auto resnet50_model = resnet50(1000, false);
    std::cout << "Created ResNet-50 for ImageNet (1000 classes)\n";

    auto params50 = resnet50_model->parameters();
    int64_t total_params50 = 0;
    for (const auto& p : params50) {
        total_params50 += p->data().numel();
    }
    std::cout << "  Total parameters: " << total_params50 / 1e6 << "M\n\n";

    // Create ResNeXt-50
    auto resnext50_model = resnext50_32x4d(1000, false);
    std::cout << "Created ResNeXt-50 (32x4d) for ImageNet\n\n";

    // Create Wide ResNet-50
    auto wide_resnet_model = wide_resnet50_2(1000, false);
    std::cout << "Created Wide ResNet-50-2 for ImageNet\n\n";
}

/**
 * @brief Example 2: Forward pass and inference
 */
void example_inference() {
    std::cout << "=== Example 2: Inference with ResNet ===\n\n";

    // Create model for 10 classes (e.g., CIFAR-10)
    auto model = resnet18(10, false);
    model->eval();  // Set to evaluation mode

    // Create dummy input batch (4 images, 3 channels, 224x224)
    Variable input(Tensor({4, 3, 224, 224}, DType::Float32, Device::cpu()), false);

    std::cout << "Input shape: [" << input.data().shape()[0] << ", "
              << input.data().shape()[1] << ", "
              << input.data().shape()[2] << ", "
              << input.data().shape()[3] << "]\n";

    // Forward pass
    Variable output = model->forward(input);

    std::cout << "Output shape: [" << output.data().shape()[0] << ", "
              << output.data().shape()[1] << "]\n";
    std::cout << "Output logits for first image (class 0-9):\n";

    // Apply softmax to get probabilities
    Softmax softmax_layer(-1);
    Variable probs = softmax_layer.forward(output);

    std::cout << "  (Softmax applied for probabilities)\n\n";
}

/**
 * @brief Example 3: Training ResNet on a dataset
 */
void example_training() {
    std::cout << "=== Example 3: Training ResNet ===\n\n";

    // Create model for custom dataset (100 classes)
    auto model = resnet18(100, false);
    model->train();  // Set to training mode

    // Create optimizer (SGD with momentum)
    auto params = model->parameters();
    optim::SGD optimizer(params, 0.01, 0.9);  // lr=0.01, momentum=0.9

    // Create loss function
    loss::CrossEntropyLoss criterion;

    std::cout << "Training configuration:\n";
    std::cout << "  Model: ResNet-18\n";
    std::cout << "  Classes: 100\n";
    std::cout << "  Optimizer: SGD (lr=0.01, momentum=0.9)\n";
    std::cout << "  Loss: CrossEntropyLoss\n\n";

    // Simulate training for a few iterations
    int num_epochs = 2;
    int batch_size = 8;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << "\n";

        // Simulate 3 batches
        for (int batch = 0; batch < 3; ++batch) {
            // Create dummy batch (images and labels)
            Variable images(Tensor({batch_size, 3, 224, 224}, DType::Float32, Device::cpu()), true);
            Variable labels(Tensor({batch_size}, DType::Int64, Device::cpu()), false);

            // Zero gradients
            optimizer.zero_grad();

            // Forward pass
            Variable outputs = model->forward(images);

            // Compute loss
            Variable loss = criterion.forward(outputs, labels);

            // Backward pass
            loss.backward();

            // Update weights
            optimizer.step();

            std::cout << "  Batch " << (batch + 1) << "/3 - Loss: "
                      << loss.data().item<float>() << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "Training complete!\n\n";
}

/**
 * @brief Example 4: Transfer learning with pretrained weights
 */
void example_transfer_learning() {
    std::cout << "=== Example 4: Transfer Learning ===\n\n";

    // Create ResNet-50 with pretrained ImageNet weights
    auto model = resnet50(1000, false);  // Set to true when weights are available

    std::cout << "Loaded ResNet-50 (pretrained weights would be loaded if available)\n";
    std::cout << "Original model: 1000 classes (ImageNet)\n\n";

    // Replace final layer for new task (20 classes)
    std::cout << "Replacing final layer for 20-class classification task\n";

    // Note: In actual implementation, you would need to access and replace
    // the fc layer. This is simplified for the example.
    std::cout << "  New FC layer: 2048 -> 20\n\n";

    // Freeze early layers (feature extraction)
    std::cout << "Freezing early layers (only train final layer)\n";
    auto all_params = model->parameters();
    std::cout << "  Total parameters: " << all_params.size() << "\n";
    std::cout << "  (In practice, would set requires_grad=False for early layers)\n\n";

    // Fine-tune with lower learning rate
    std::cout << "Fine-tuning with lr=0.001\n";
    optim::Adam optimizer(all_params, 0.001);

    std::cout << "Transfer learning setup complete!\n\n";
}

/**
 * @brief Example 5: Different ResNet architectures comparison
 */
void example_architecture_comparison() {
    std::cout << "=== Example 5: Architecture Comparison ===\n\n";

    struct ModelInfo {
        std::string name;
        std::shared_ptr<ResNet> model;
    };

    std::vector<ModelInfo> models = {
        {"ResNet-18", resnet18(1000, false)},
        {"ResNet-34", resnet34(1000, false)},
        {"ResNet-50", resnet50(1000, false)},
        {"ResNet-101", resnet101(1000, false)},
        {"ResNeXt-50", resnext50_32x4d(1000, false)},
        {"Wide ResNet-50", wide_resnet50_2(1000, false)}
    };

    std::cout << "Model Architecture Comparison:\n\n";
    std::cout << "Model              | Parameters | Depth\n";
    std::cout << "-------------------|------------|------\n";

    for (const auto& info : models) {
        auto params = info.model->parameters();
        int64_t total_params = 0;
        for (const auto& p : params) {
            total_params += p->data().numel();
        }

        std::cout << info.name;
        for (size_t i = info.name.length(); i < 19; ++i) std::cout << " ";
        std::cout << "| " << (total_params / 1000000) << "M";

        // Depth varies by architecture
        if (info.name == "ResNet-18") std::cout << "     | 18\n";
        else if (info.name == "ResNet-34") std::cout << "    | 34\n";
        else if (info.name == "ResNet-50") std::cout << "    | 50\n";
        else if (info.name == "ResNet-101") std::cout << "   | 101\n";
        else std::cout << "  | 50\n";
    }
    std::cout << "\n";
}

/**
 * @brief Example 6: Handling different input sizes
 */
void example_different_input_sizes() {
    std::cout << "=== Example 6: Different Input Sizes ===\n\n";

    auto model = resnet18(10, false);
    model->eval();

    std::vector<std::vector<int64_t>> input_sizes = {
        {2, 3, 224, 224},  // Standard ImageNet size
        {2, 3, 128, 128},  // Smaller images
        {2, 3, 320, 320},  // Larger images
        {2, 3, 256, 256}   // Square images
    };

    std::cout << "Testing ResNet-18 with different input sizes:\n\n";

    for (const auto& size : input_sizes) {
        Variable input(Tensor(size, DType::Float32, Device::cpu()), false);
        Variable output = model->forward(input);

        std::cout << "Input: [" << size[0] << ", " << size[1] << ", "
                  << size[2] << ", " << size[3] << "] -> Output: ["
                  << output.data().shape()[0] << ", "
                  << output.data().shape()[1] << "]\n";
    }
    std::cout << "\n";
}

/**
 * @brief Example 7: Model evaluation
 */
void example_evaluation() {
    std::cout << "=== Example 7: Model Evaluation ===\n\n";

    auto model = resnet18(10, false);
    model->eval();  // Important: set to eval mode for inference

    // Simulate evaluation on test set
    int num_test_batches = 5;
    int batch_size = 16;
    int correct = 0;
    int total = 0;

    std::cout << "Evaluating on test set...\n";

    for (int i = 0; i < num_test_batches; ++i) {
        // Create dummy test batch
        Variable images(Tensor({batch_size, 3, 224, 224}, DType::Float32, Device::cpu()), false);
        Variable labels(Tensor({batch_size}, DType::Int64, Device::cpu()), false);

        // Forward pass (no gradient computation needed)
        Variable outputs = model->forward(images);

        // Get predictions (argmax)
        // In practice, you would compare with actual labels
        total += batch_size;
        correct += batch_size / 2;  // Dummy accuracy
    }

    float accuracy = static_cast<float>(correct) / total * 100;
    std::cout << "Test Accuracy: " << accuracy << "%\n";
    std::cout << "Correct: " << correct << "/" << total << "\n\n";
}

/**
 * @brief Main function running all examples
 */
int main() {
    std::cout << "========================================\n";
    std::cout << "ResNet Model Examples\n";
    std::cout << "========================================\n\n";

    try {
        example_create_models();
        example_inference();
        example_training();
        example_transfer_learning();
        example_architecture_comparison();
        example_different_input_sizes();
        example_evaluation();

        std::cout << "========================================\n";
        std::cout << "All examples completed successfully!\n";
        std::cout << "========================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
