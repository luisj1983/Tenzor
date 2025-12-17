/**
 * @file autograd.cpp
 * @brief XOR problem solved using Tenzor's automatic differentiation
 *
 * This example demonstrates the mid-level approach: using Variable and
 * autograd for automatic gradient computation, while still manually
 * defining the network structure and training loop.
 *
 * Usage: ./01_xor_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

int main(int argc, char* argv[]) {
    // Parse backend from command line
    Device device = showcase::get_device_from_args(argc, argv);

    // Initialize Tenzor
    initialize();

    showcase::print_header("XOR - Autograd (Automatic Differentiation)", device);

    // Set seed for reproducibility
    manual_seed(42);

    // XOR dataset
    float input_data[] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    float target_data[] = {0.0f, 1.0f, 1.0f, 0.0f};

    // Create tensors (no gradient tracking for data)
    auto X_tensor = from_data(input_data, {4, 2}, device);
    auto y_tensor = from_data(target_data, {4, 1}, device);

    showcase::print_tensor_info("Input X", X_tensor);
    showcase::print_tensor_info("Target y", y_tensor);

    // Network architecture: 2 -> 4 -> 1
    // Initialize weights as Variables with requires_grad=true

    // Hidden layer weights
    auto W1_tensor = randn({2, 4}, DType::Float32, device) * 0.5f;
    auto b1_tensor = zeros({1, 4}, DType::Float32, device);

    // Output layer weights
    auto W2_tensor = randn({4, 1}, DType::Float32, device) * 0.5f;
    auto b2_tensor = zeros({1, 1}, DType::Float32, device);

    // Wrap in Variables with gradient tracking
    Variable W1(W1_tensor, true);  // requires_grad = true
    Variable b1(b1_tensor, true);
    Variable W2(W2_tensor, true);
    Variable b2(b2_tensor, true);

    showcase::print_section("Initial Parameters");
    std::cout << "W1: requires_grad=" << W1.requires_grad() << "\n";
    std::cout << "b1: requires_grad=" << b1.requires_grad() << "\n";
    std::cout << "W2: requires_grad=" << W2.requires_grad() << "\n";
    std::cout << "b2: requires_grad=" << b2.requires_grad() << "\n";

    // Training parameters
    float learning_rate = 1.0f;
    int num_epochs = 10000;
    int print_every = 1000;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Wrap input and target as Variables (no grad needed for data)
        Variable X(X_tensor, false);
        Variable y(y_tensor, false);

        // ============ Forward Pass ============
        // Hidden layer: z1 = X @ W1 + b1
        auto z1 = X.matmul(W1) + b1;

        // Apply sigmoid activation
        auto a1 = nn::sigmoid(z1);

        // Output layer: z2 = a1 @ W2 + b2
        auto z2 = a1.matmul(W2) + b2;

        // Output activation
        auto a2 = nn::sigmoid(z2);

        // ============ Compute Loss (MSE) ============
        auto error = a2 - y;
        auto squared_error = error * error;
        auto loss = mean(squared_error);

        // ============ Backward Pass (Automatic!) ============
        // Zero gradients before backward
        W1.zero_grad();
        b1.zero_grad();
        W2.zero_grad();
        b2.zero_grad();

        // Compute gradients automatically
        loss.backward();

        // ============ Update Weights (Manual SGD) ============
        // We need to detach and update the underlying tensors
        {
            NoGradGuard no_grad;  // Disable gradient tracking for updates

            // Update W1
            auto W1_new = W1.tensor() - (*W1.grad() * learning_rate);
            W1 = Variable(W1_new, true);

            // Update b1
            auto b1_new = b1.tensor() - (*b1.grad() * learning_rate);
            b1 = Variable(b1_new, true);

            // Update W2
            auto W2_new = W2.tensor() - (*W2.grad() * learning_rate);
            W2 = Variable(W2_new, true);

            // Update b2
            auto b2_new = b2.tensor() - (*b2.grad() * learning_rate);
            b2 = Variable(b2_new, true);
        }

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();

            // Calculate accuracy
            auto pred_cpu = a2.tensor().cpu();
            auto target_cpu = y_tensor.cpu();
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

    // Forward pass for final predictions
    Variable X_final(X_tensor, false);
    auto z1_final = X_final.matmul(W1) + b1;
    auto a1_final = nn::sigmoid(z1_final);
    auto z2_final = a1_final.matmul(W2) + b2;
    auto predictions = nn::sigmoid(z2_final);

    // Move to CPU for printing
    auto pred_cpu = predictions.tensor().cpu();
    auto target_cpu = y_tensor.cpu();

    std::cout << "Input\t\tTarget\tPrediction\tRounded\n";
    std::cout << "----------------------------------------------\n";

    auto input_cpu = X_tensor.cpu();
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

    std::cout << "\nXOR problem solved using autograd!\n";
    std::cout << "Automatic differentiation computed all gradients for us.\n";
    std::cout << "Compare this to tensor_only.cpp - much simpler backward pass!\n";

    finalize();
    return 0;
}
