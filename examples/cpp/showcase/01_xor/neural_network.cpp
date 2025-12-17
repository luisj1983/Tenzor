/**
 * @file neural_network.cpp
 * @brief XOR problem solved using Tenzor's high-level Neural Network API
 *
 * This example demonstrates the highest-level approach: using nn::Module,
 * nn::Sequential, built-in layers, loss functions, and optimizers for a
 * clean, PyTorch-like interface.
 *
 * Usage: ./01_xor_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

/**
 * @brief Custom XOR network module
 *
 * A simple 2-layer MLP for solving the XOR problem.
 */
class XORNet : public nn::Module {
public:
    XORNet() {
        // Create layers
        fc1 = std::make_shared<nn::Linear>(2, 4);   // Input -> Hidden
        fc2 = std::make_shared<nn::Linear>(4, 1);   // Hidden -> Output

        // Register modules for parameter tracking
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        // Hidden layer with sigmoid activation
        auto h = nn::sigmoid(fc1->forward(input));
        // Output layer with sigmoid activation
        auto out = nn::sigmoid(fc2->forward(h));
        return out;
    }

private:
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::Linear> fc2;
};

int main(int argc, char* argv[]) {
    // Parse backend from command line
    Device device = showcase::get_device_from_args(argc, argv);

    // Initialize Tenzor
    initialize();

    showcase::print_header("XOR - Neural Network API (High-Level)", device);

    // Set seed for reproducibility
    manual_seed(42);

    // XOR dataset
    float input_data[] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    float target_data[] = {0.0f, 1.0f, 1.0f, 0.0f};

    auto X = from_data(input_data, {4, 2}, device);
    auto y = from_data(target_data, {4, 1}, device);

    showcase::print_tensor_info("Input X", X);
    showcase::print_tensor_info("Target y", y);

    // Create model
    auto model = std::make_shared<XORNet>();

    // Move model to device
    model->to(device);

    // Create optimizer (SGD with momentum)
    auto params = model->parameters();
    optim::SGD optimizer(params, 1.0f);  // learning_rate = 1.0

    // Create loss function
    nn::MSELoss criterion;

    showcase::print_section("Model Architecture");
    std::cout << "XORNet:\n";
    std::cout << "  fc1: Linear(2, 4)\n";
    std::cout << "  fc2: Linear(4, 1)\n";
    std::cout << "  Activation: Sigmoid\n";
    std::cout << "\nTotal parameters: " << params.size() << "\n";

    // Training parameters
    int num_epochs = 10000;
    int print_every = 1000;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Set model to training mode
        model->train();

        // Zero gradients
        optimizer.zero_grad();

        // Forward pass
        Variable input(X, false);
        auto output = model->forward(input);

        // Compute loss
        Variable target(y, false);
        auto loss = criterion(output, target);

        // Backward pass
        loss.backward();

        // Update weights
        optimizer.step();

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();

            // Calculate accuracy
            auto pred_cpu = output.tensor().cpu();
            auto target_cpu = y.cpu();
            int correct = 0;
            const float* pred = pred_cpu.data<float>();
            const float* tgt = target_cpu.data<float>();
            for (int i = 0; i < 4; ++i) {
                float predicted = (pred[i] > 0.5f) ? 1.0f : 0.0f;
                if (predicted == tgt[i]) correct++;
            }
            float accuracy = correct / 4.0f;

            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    // ============ Final Evaluation ============
    showcase::print_section("Final Results");

    // Set model to evaluation mode
    model->eval();

    // Forward pass
    Variable X_eval(X, false);
    auto predictions = model->forward(X_eval);

    // Move to CPU for printing
    auto pred_cpu = predictions.tensor().cpu();
    auto target_cpu = y.cpu();

    std::cout << "Input\t\tTarget\tPrediction\tRounded\n";
    std::cout << "----------------------------------------------\n";

    auto input_cpu = X.cpu();
    const float* input_ptr = input_cpu.data<float>();
    const float* target_ptr = target_cpu.data<float>();
    const float* pred_ptr = pred_cpu.data<float>();

    for (int i = 0; i < 4; ++i) {
        float x0 = input_ptr[i * 2];
        float x1 = input_ptr[i * 2 + 1];
        float target = target_ptr[i];
        float pred = pred_ptr[i];
        float rounded = (pred > 0.5f) ? 1.0f : 0.0f;

        std::cout << "(" << x0 << ", " << x1 << ")\t\t"
                  << target << "\t"
                  << pred << "\t"
                  << rounded << "\n";
    }

    std::cout << "\nXOR problem solved using Neural Network API!\n";
    std::cout << "This is the cleanest approach with nn::Module, Optimizer, and Loss.\n";
    std::cout << "\nComparison:\n";
    std::cout << "  - tensor_only.cpp:    Manual forward & backward (educational)\n";
    std::cout << "  - autograd.cpp:       Manual forward, auto backward (middle ground)\n";
    std::cout << "  - neural_network.cpp: Full high-level API (production)\n";

    finalize();
    return 0;
}
