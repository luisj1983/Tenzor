/**
 * @file tensor_only.cpp
 * @brief XOR problem solved using raw Tensor operations only
 *
 * This example demonstrates the lowest-level approach to neural networks
 * in Tenzor: manually implementing forward and backward passes using
 * only Tensor operations. No automatic differentiation is used.
 *
 * The XOR problem is a classic non-linearly separable problem that requires
 * at least one hidden layer to solve.
 *
 * Usage: ./01_xor_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>

using namespace tenzor;

// Manual sigmoid implementation
Tensor manual_sigmoid(const Tensor& x) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    auto neg_x = x * -1.0f;
    auto exp_neg_x = tenzor::exp(neg_x);
    auto one_plus_exp = exp_neg_x + 1.0f;
    return ones_like(x) / one_plus_exp;
}

// Manual sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
Tensor sigmoid_derivative(const Tensor& sigmoid_output) {
    return sigmoid_output * (ones_like(sigmoid_output) - sigmoid_output);
}

int main(int argc, char* argv[]) {
    // Parse backend from command line
    Device device = showcase::get_device_from_args(argc, argv);

    // Initialize Tenzor
    initialize();

    showcase::print_header("XOR - Tensor Only (Manual Backprop)", device);

    // Set seed for reproducibility
    manual_seed(42);

    // XOR dataset
    // Inputs: (0,0), (0,1), (1,0), (1,1)
    // Outputs:  0,    1,     1,     0
    float input_data[] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    float target_data[] = {0.0f, 1.0f, 1.0f, 0.0f};

    auto X = from_data(input_data, {4, 2}, device);
    auto y = from_data(target_data, {4, 1}, device);

    showcase::print_tensor_info("Input X", X);
    showcase::print_tensor_info("Target y", y);

    // Network architecture: 2 -> 4 -> 1
    // Hidden layer: 2 inputs, 4 neurons
    // Output layer: 4 inputs, 1 neuron

    // Initialize weights with Xavier initialization
    auto W1 = randn({2, 4}, DType::Float32, device) * 0.5f;  // (2, 4)
    auto b1 = zeros({1, 4}, DType::Float32, device);          // (1, 4)
    auto W2 = randn({4, 1}, DType::Float32, device) * 0.5f;  // (4, 1)
    auto b2 = zeros({1, 1}, DType::Float32, device);          // (1, 1)

    showcase::print_section("Initial Weights");
    showcase::print_tensor_info("W1", W1);
    showcase::print_tensor_info("b1", b1);
    showcase::print_tensor_info("W2", W2);
    showcase::print_tensor_info("b2", b2);

    // Training parameters
    float learning_rate = 1.0f;
    int num_epochs = 10000;
    int print_every = 1000;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ============ Forward Pass ============
        // Hidden layer: z1 = X @ W1 + b1, a1 = sigmoid(z1)
        auto z1 = matmul(X, W1) + b1;    // (4, 4)
        auto a1 = manual_sigmoid(z1);     // (4, 4)

        // Output layer: z2 = a1 @ W2 + b2, a2 = sigmoid(z2)
        auto z2 = matmul(a1, W2) + b2;   // (4, 1)
        auto a2 = manual_sigmoid(z2);     // (4, 1) - predictions

        // ============ Compute Loss (MSE) ============
        auto error = a2 - y;
        auto squared_error = error * error;
        float loss = tenzor::mean(squared_error).item<float>();

        // ============ Backward Pass (Manual Gradient Computation) ============
        // Output layer gradients
        // dL/da2 = 2 * (a2 - y) / n
        auto dL_da2 = error * (2.0f / 4.0f);  // (4, 1)

        // da2/dz2 = sigmoid'(z2) = a2 * (1 - a2)
        auto da2_dz2 = sigmoid_derivative(a2);  // (4, 1)

        // dL/dz2 = dL/da2 * da2/dz2
        auto dL_dz2 = dL_da2 * da2_dz2;  // (4, 1)

        // dL/dW2 = a1^T @ dL/dz2
        auto dL_dW2 = matmul(a1.transpose(0, 1), dL_dz2);  // (4, 1)

        // dL/db2 = sum(dL/dz2, dim=0)
        auto dL_db2 = tenzor::sum(dL_dz2, 0, true);  // (1, 1)

        // Hidden layer gradients
        // dL/da1 = dL/dz2 @ W2^T
        auto dL_da1 = matmul(dL_dz2, W2.transpose(0, 1));  // (4, 4)

        // da1/dz1 = sigmoid'(z1) = a1 * (1 - a1)
        auto da1_dz1 = sigmoid_derivative(a1);  // (4, 4)

        // dL/dz1 = dL/da1 * da1/dz1
        auto dL_dz1 = dL_da1 * da1_dz1;  // (4, 4)

        // dL/dW1 = X^T @ dL/dz1
        auto dL_dW1 = matmul(X.transpose(0, 1), dL_dz1);  // (2, 4)

        // dL/db1 = sum(dL/dz1, dim=0)
        auto dL_db1 = tenzor::sum(dL_dz1, 0, true);  // (1, 4)

        // ============ Update Weights (SGD) ============
        W2 = W2 - dL_dW2 * learning_rate;
        b2 = b2 - dL_db2 * learning_rate;
        W1 = W1 - dL_dW1 * learning_rate;
        b1 = b1 - dL_db1 * learning_rate;

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            // Calculate accuracy
            int correct = 0;
            const float* pred = a2.cpu().data<float>();
            const float* tgt = y.cpu().data<float>();
            for (int i = 0; i < 4; ++i) {
                float predicted = (pred[i] > 0.5f) ? 1.0f : 0.0f;
                if (predicted == tgt[i]) correct++;
            }
            float accuracy = correct / 4.0f;
            showcase::print_progress(epoch, num_epochs, loss, accuracy);
        }
    }

    // ============ Final Evaluation ============
    showcase::print_section("Final Results");

    // Forward pass for final predictions
    auto z1_final = matmul(X, W1) + b1;
    auto a1_final = manual_sigmoid(z1_final);
    auto z2_final = matmul(a1_final, W2) + b2;
    auto predictions = manual_sigmoid(z2_final);

    // Move to CPU for printing
    auto pred_cpu = predictions.cpu();
    auto target_cpu = y.cpu();

    std::cout << "Input\t\tTarget\tPrediction\tRounded\n";
    std::cout << "----------------------------------------------\n";

    const float* input_ptr = X.cpu().data<float>();
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

    std::cout << "\nXOR problem solved using raw tensors!\n";
    std::cout << "This demonstrates manual forward and backward propagation.\n";

    finalize();
    return 0;
}
