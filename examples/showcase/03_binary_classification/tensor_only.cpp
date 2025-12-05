/**
 * @file tensor_only.cpp
 * @brief Binary Classification using raw Tensor operations only
 *
 * This example demonstrates binary classification (logistic regression)
 * using only tensor operations with manual gradient computation.
 *
 * Binary classification predicts one of two classes using sigmoid activation
 * and binary cross-entropy loss.
 *
 * Usage: ./03_binary_classification_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>

using namespace tenzor;

// Manual sigmoid
Tensor manual_sigmoid(const Tensor& x) {
    auto neg_x = x * -1.0f;
    auto exp_neg_x = tenzor::exp(neg_x);
    auto one_plus_exp = exp_neg_x + 1.0f;
    return ones_like(x) / one_plus_exp;
}

// Manual binary cross-entropy loss
// BCE = -mean(y * log(p) + (1-y) * log(1-p))
float bce_loss(const Tensor& predictions, const Tensor& targets) {
    const float eps = 1e-7f;  // Numerical stability
    auto p_clipped = clamp(predictions, eps, 1.0f - eps);
    auto log_p = tenzor::log(p_clipped);
    auto log_1_minus_p = tenzor::log(ones_like(p_clipped) - p_clipped + eps);

    auto loss = targets * log_p + (ones_like(targets) - targets) * log_1_minus_p;
    return -tenzor::mean(loss).item<float>();
}

int main(int argc, char* argv[]) {
    // Parse backend from command line
    Device device = showcase::get_device_from_args(argc, argv);

    // Initialize Tenzor
    initialize();

    showcase::print_header("Binary Classification - Tensor Only (Manual Backprop)", device);

    // Set seed for reproducibility
    manual_seed(42);

    // Generate synthetic binary classification data
    // Two clusters in 2D space
    int samples_per_class = 50;
    int num_samples = samples_per_class * 2;

    // Class 0: centered around (-2, -2)
    auto class0_x = randn({samples_per_class, 2}, DType::Float32, device) + (-2.0f);

    // Class 1: centered around (2, 2)
    auto class1_x = randn({samples_per_class, 2}, DType::Float32, device) + 2.0f;

    // Combine into dataset
    auto X = cat({class0_x, class1_x}, 0);  // (100, 2)

    // Create labels: 0 for first half, 1 for second half
    auto labels0 = zeros({samples_per_class, 1}, DType::Float32, device);
    auto labels1 = ones({samples_per_class, 1}, DType::Float32, device);
    auto y = cat({labels0, labels1}, 0);  // (100, 1)

    showcase::print_tensor_info("Input X", X);
    showcase::print_tensor_info("Labels y", y);

    // Initialize weights for logistic regression: y = sigmoid(X @ W + b)
    auto W = randn({2, 1}, DType::Float32, device) * 0.1f;  // (2, 1)
    auto b = zeros({1, 1}, DType::Float32, device);          // (1, 1)

    showcase::print_section("Initial Parameters");
    showcase::print_tensor_info("W", W);
    showcase::print_tensor_info("b", b);

    // Training parameters
    float learning_rate = 0.1f;
    int num_epochs = 500;
    int print_every = 50;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ============ Forward Pass ============
        auto z = matmul(X, W) + b;      // (100, 1) - logits
        auto p = manual_sigmoid(z);      // (100, 1) - probabilities

        // ============ Compute Loss (BCE) ============
        float loss = bce_loss(p, y);

        // ============ Backward Pass (Manual Gradient Computation) ============
        // BCE gradient w.r.t. logits (after simplification):
        // dL/dz = p - y (this is a well-known result)
        float n = static_cast<float>(num_samples);
        auto dL_dz = (p - y) / n;  // (100, 1)

        // dL/dW = X^T @ dL/dz
        auto dL_dW = matmul(X.transpose(0, 1), dL_dz);  // (2, 1)

        // dL/db = sum(dL/dz)
        auto dL_db = tenzor::sum(dL_dz, 0, true);  // (1, 1)

        // ============ Update Weights (SGD) ============
        W = W - dL_dW * learning_rate;
        b = b - dL_db * learning_rate;

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            // Calculate accuracy
            auto pred_cpu = p.cpu();
            auto target_cpu = y.cpu();
            int correct = 0;
            const float* pred = pred_cpu.data<float>();
            const float* tgt = target_cpu.data<float>();
            for (int i = 0; i < num_samples; ++i) {
                float predicted = (pred[i] > 0.5f) ? 1.0f : 0.0f;
                if (predicted == tgt[i]) correct++;
            }
            float accuracy = static_cast<float>(correct) / num_samples;

            showcase::print_progress(epoch, num_epochs, loss, accuracy);
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    // Final predictions
    auto z_final = matmul(X, W) + b;
    auto p_final = manual_sigmoid(z_final);

    auto pred_cpu = p_final.cpu();
    auto target_cpu = y.cpu();
    int correct = 0;
    const float* pred = pred_cpu.data<float>();
    const float* tgt = target_cpu.data<float>();
    for (int i = 0; i < num_samples; ++i) {
        float predicted = (pred[i] > 0.5f) ? 1.0f : 0.0f;
        if (predicted == tgt[i]) correct++;
    }
    float final_accuracy = static_cast<float>(correct) / num_samples;

    std::cout << "Final Accuracy: " << (final_accuracy * 100.0f) << "%\n\n";

    // Print learned parameters
    auto W_cpu = W.cpu();
    auto b_cpu = b.cpu();
    std::cout << "Learned parameters:\n";
    std::cout << "  W = [" << W_cpu.data<float>()[0] << ", "
              << W_cpu.data<float>()[1] << "]\n";
    std::cout << "  b = " << b_cpu.data<float>()[0] << "\n";

    // Decision boundary: W[0]*x + W[1]*y + b = 0
    std::cout << "\nDecision boundary equation:\n";
    std::cout << "  " << W_cpu.data<float>()[0] << "*x1 + "
              << W_cpu.data<float>()[1] << "*x2 + "
              << b_cpu.data<float>()[0] << " = 0\n";

    std::cout << "\nBinary classification solved using raw tensors!\n";
    std::cout << "This demonstrates logistic regression with manual backprop.\n";

    finalize();
    return 0;
}
