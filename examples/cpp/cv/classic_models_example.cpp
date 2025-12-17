/**
 * @file classic_models_example.cpp
 * @brief Comprehensive example demonstrating VGG, AlexNet, and GoogLeNet usage
 *
 * This example shows:
 * - Creating classic CNN models
 * - Forward inference
 * - Training with auxiliary classifiers (GoogLeNet)
 * - Model comparison
 * - Parameter inspection
 */

#include <tenzor/models/vgg.hpp>
#include <tenzor/models/alexnet.hpp>
#include <tenzor/models/googlenet.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <iostream>
#include <iomanip>
#include <memory>

using namespace tenzor;
using namespace tenzor::models;

// Utility function to print model info
void print_model_info(const std::string& name, std::shared_ptr<nn::Module> model) {
    std::cout << "\n========================================\n";
    std::cout << name << " Model Information\n";
    std::cout << "========================================\n";

    auto params = model->parameters();
    int64_t total_params = 0;

    for (const auto& param : params) {
        int64_t param_count = 1;
        for (auto dim : param->tensor().shape()) {
            param_count *= dim;
        }
        total_params += param_count;
    }

    std::cout << "Total parameters: " << total_params << "\n";
    std::cout << "Parameter tensors: " << params.size() << "\n";
    std::cout << "Training mode: " << (model->is_training() ? "Yes" : "No") << "\n";
}

// Example 1: Basic inference with all models
void example_basic_inference() {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║  Example 1: Basic Inference           ║\n";
    std::cout << "╚════════════════════════════════════════╝\n";

    Device device = Device::cpu();

    // Create models for ImageNet (1000 classes)
    std::cout << "\nCreating models...\n";
    auto vgg16_model = vgg16(1000, true, false);
    auto alexnet_model = alexnet(1000, false);
    auto googlenet_model = googlenet(1000, false, false);

    // Print model information
    print_model_info("VGG-16", vgg16_model);
    print_model_info("AlexNet", alexnet_model);
    print_model_info("GoogLeNet", googlenet_model);

    // Set models to evaluation mode
    vgg16_model->eval();
    alexnet_model->eval();
    googlenet_model->eval();

    // Create dummy input (224x224 RGB image)
    std::cout << "\nCreating input tensor (1, 3, 224, 224)...\n";
    Tensor input({1, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);  // Normalized pixel values
    Variable x(input, false);  // No gradients needed for inference

    // Run inference
    std::cout << "\nRunning inference...\n";

    std::cout << "  VGG-16 forward pass...";
    auto vgg_output = vgg16_model->forward(x);
    std::cout << " Output shape: (" << vgg_output.tensor().shape()[0]
              << ", " << vgg_output.tensor().shape()[1] << ")\n";

    std::cout << "  AlexNet forward pass...";
    auto alexnet_output = alexnet_model->forward(x);
    std::cout << " Output shape: (" << alexnet_output.tensor().shape()[0]
              << ", " << alexnet_output.tensor().shape()[1] << ")\n";

    std::cout << "  GoogLeNet forward pass...";
    auto googlenet_output = googlenet_model->forward(x);
    std::cout << " Output shape: (" << googlenet_output.tensor().shape()[0]
              << ", " << googlenet_output.tensor().shape()[1] << ")\n";

    std::cout << "\n✓ All models successfully performed inference!\n";
}

// Example 2: Training with GoogLeNet auxiliary classifiers
void example_googlenet_training() {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║  Example 2: GoogLeNet Training         ║\n";
    std::cout << "╚════════════════════════════════════════╝\n";

    Device device = Device::cpu();

    // Create GoogLeNet with auxiliary classifiers
    std::cout << "\nCreating GoogLeNet with auxiliary classifiers...\n";
    auto model = googlenet(10, false, true);  // 10 classes, aux_logits=true
    model->train();  // Training mode

    print_model_info("GoogLeNet (with aux)", model);

    // Create dummy batch
    std::cout << "\nCreating training batch (batch=4, 3, 224, 224)...\n";
    Tensor input({4, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);  // Requires gradients

    // Create dummy targets
    Tensor targets({4}, DType::Int64, device);
    for (int i = 0; i < 4; ++i) {
        targets.data<int64_t>()[i] = i % 10;  // Classes 0-9
    }

    // Forward pass with auxiliary outputs
    std::cout << "\nForward pass with auxiliary classifiers...\n";
    auto [main_out, aux1_out, aux2_out] = model->forward_with_aux(x);

    std::cout << "  Main output shape: (" << main_out.tensor().shape()[0]
              << ", " << main_out.tensor().shape()[1] << ")\n";
    std::cout << "  Aux1 output shape: (" << aux1_out.tensor().shape()[0]
              << ", " << aux1_out.tensor().shape()[1] << ")\n";
    std::cout << "  Aux2 output shape: (" << aux2_out.tensor().shape()[0]
              << ", " << aux2_out.tensor().shape()[1] << ")\n";

    std::cout << "\nDuring training, the total loss would be:\n";
    std::cout << "  total_loss = main_loss + 0.3 * aux1_loss + 0.3 * aux2_loss\n";
    std::cout << "\nThis helps train deeper layers by providing gradient signal!\n";
}

// Example 3: Comparing model variants
void example_model_variants() {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║  Example 3: Model Variants             ║\n";
    std::cout << "╚════════════════════════════════════════╝\n";

    Device device = Device::cpu();

    // Create all VGG variants
    std::cout << "\nCreating VGG variants...\n";
    auto vgg11_model = vgg11(10, true, false);
    auto vgg13_model = vgg13(10, true, false);
    auto vgg16_model = vgg16(10, true, false);
    auto vgg19_model = vgg19(10, true, false);

    // Print comparison
    std::cout << "\n" << std::setw(15) << "Model"
              << std::setw(20) << "Parameters"
              << std::setw(25) << "Parameter Tensors" << "\n";
    std::cout << std::string(60, '-') << "\n";

    auto print_stats = [](const std::string& name, std::shared_ptr<nn::Module> model) {
        auto params = model->parameters();
        int64_t total = 0;
        for (const auto& p : params) {
            int64_t count = 1;
            for (auto dim : p->tensor().shape()) count *= dim;
            total += count;
        }
        std::cout << std::setw(15) << name
                  << std::setw(20) << total
                  << std::setw(25) << params.size() << "\n";
    };

    print_stats("VGG-11", vgg11_model);
    print_stats("VGG-13", vgg13_model);
    print_stats("VGG-16", vgg16_model);
    print_stats("VGG-19", vgg19_model);

    std::cout << "\nAs expected, deeper models have more parameters!\n";
}

// Example 4: Batch processing
void example_batch_processing() {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║  Example 4: Batch Processing           ║\n";
    std::cout << "╚════════════════════════════════════════╝\n";

    Device device = Device::cpu();

    // Create model
    auto model = alexnet(1000, false);
    model->eval();

    // Test different batch sizes
    std::vector<int64_t> batch_sizes = {1, 2, 4, 8, 16};

    std::cout << "\nProcessing different batch sizes with AlexNet:\n";
    std::cout << std::setw(15) << "Batch Size" << std::setw(25) << "Output Shape" << "\n";
    std::cout << std::string(40, '-') << "\n";

    for (auto batch_size : batch_sizes) {
        Tensor input({batch_size, 3, 224, 224}, DType::Float32, device);
        input.fill_(0.5f);
        Variable x(input, false);

        auto output = model->forward(x);

        std::cout << std::setw(15) << batch_size
                  << std::setw(25) << ("(" + std::to_string(output.tensor().shape()[0]) +
                                      ", " + std::to_string(output.tensor().shape()[1]) + ")")
                  << "\n";
    }

    std::cout << "\n✓ All batch sizes processed successfully!\n";
}

// Example 5: Custom configurations
void example_custom_configurations() {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║  Example 5: Custom Configurations      ║\n";
    std::cout << "╚════════════════════════════════════════╝\n";

    // VGG without batch normalization
    std::cout << "\n1. VGG-16 without Batch Normalization:\n";
    auto vgg_no_bn = vgg16(100, false, false);  // batch_norm=false
    std::cout << "   Created for 100 classes without BN\n";

    // AlexNet with custom dropout
    std::cout << "\n2. AlexNet with custom dropout (0.3):\n";
    auto alexnet_custom = std::make_shared<AlexNet>(1000, 0.3);
    std::cout << "   Created with 30% dropout rate\n";

    // GoogLeNet without auxiliary classifiers
    std::cout << "\n3. GoogLeNet for inference (no aux classifiers):\n";
    auto googlenet_inference = googlenet(1000, false, false);  // aux_logits=false
    std::cout << "   Created without auxiliary classifiers (faster inference)\n";

    // VGG with custom dropout
    std::cout << "\n4. VGG-19 with high dropout (0.7):\n";
    auto vgg_high_dropout = std::make_shared<VGG>(VGGConfig::vgg19(), 10, true, 0.7);
    std::cout << "   Created with 70% dropout for heavy regularization\n";

    std::cout << "\n✓ All custom configurations created successfully!\n";
}

// Example 6: Model serialization (state dict)
void example_model_state() {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║  Example 6: Model State Management     ║\n";
    std::cout << "╚════════════════════════════════════════╝\n";

    // Create model
    auto model = vgg11(10, true, false);

    std::cout << "\nGetting model state dictionary...\n";
    auto state = model->state_dict();

    std::cout << "State contains " << state.size() << " tensors:\n";
    int count = 0;
    for (const auto& [name, tensor] : state) {
        if (count < 5) {  // Show first 5
            std::cout << "  - " << name << ": shape [";
            for (size_t i = 0; i < tensor.shape().size(); ++i) {
                std::cout << tensor.shape()[i];
                if (i < tensor.shape().size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        count++;
    }
    if (count > 5) {
        std::cout << "  ... and " << (count - 5) << " more\n";
    }

    std::cout << "\n✓ State dictionary can be used for model checkpointing!\n";
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════╗\n";
    std::cout << "║  Classic CNN Models Example (Phase 9)             ║\n";
    std::cout << "║  VGG, AlexNet, GoogLeNet                           ║\n";
    std::cout << "╔════════════════════════════════════════════════════╝\n";

    try {
        // Run all examples
        example_basic_inference();
        example_googlenet_training();
        example_model_variants();
        example_batch_processing();
        example_custom_configurations();
        example_model_state();

        std::cout << "\n╔════════════════════════════════════════════════════╗\n";
        std::cout << "║  All examples completed successfully! ✓            ║\n";
        std::cout << "╚════════════════════════════════════════════════════╝\n";

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
