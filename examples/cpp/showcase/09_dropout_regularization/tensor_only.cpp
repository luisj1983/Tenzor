/**
 * @file tensor_only.cpp
 * @brief Dropout Regularization using raw Tensor operations only
 *
 * This example demonstrates dropout regularization using only tensor operations.
 * Dropout randomly zeros out activations during training to prevent overfitting.
 *
 * Usage: ./09_dropout_regularization_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <random>

using namespace tenzor;

// Manual dropout implementation
Tensor manual_dropout(const Tensor& x, float drop_prob, bool training, std::mt19937& gen) {
    if (!training || drop_prob == 0.0f) {
        return x;
    }

    // Generate random mask
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> mask_data(x.numel());
    float scale = 1.0f / (1.0f - drop_prob);  // Inverted dropout scaling

    for (size_t i = 0; i < mask_data.size(); ++i) {
        mask_data[i] = (dist(gen) > drop_prob) ? scale : 0.0f;
    }

    // Convert span to vector for from_data
    std::vector<int64_t> shape_vec(x.shape().begin(), x.shape().end());
    auto mask = from_data(mask_data.data(), shape_vec, x.device());
    return x * mask;
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Dropout Regularization - Tensor Only (Manual)", device);

    manual_seed(42);
    std::mt19937 gen(42);

    // Generate synthetic data with potential for overfitting
    int batch_size = 32;
    int input_features = 64;
    int hidden_features = 128;
    int num_classes = 4;
    float dropout_prob = 0.5f;

    // Create data
    std::vector<float> X_data(batch_size * input_features);
    std::vector<int64_t> y_data(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        int label = b % num_classes;
        y_data[b] = label;
        for (int f = 0; f < input_features; ++f) {
            X_data[b * input_features + f] = static_cast<float>(label) * 0.5f +
                                              randn({1}, DType::Float32, device).cpu().data<float>()[0] * 0.3f;
        }
    }

    auto X = from_data(X_data.data(), {batch_size, input_features}, device);
    auto y = from_data(y_data.data(), {batch_size}, device);

    showcase::print_tensor_info("Input X", X);

    // Initialize network weights
    auto W1 = randn({input_features, hidden_features}, DType::Float32, device) * 0.1f;
    auto b1 = zeros({1, hidden_features}, DType::Float32, device);
    auto W2 = randn({hidden_features, hidden_features}, DType::Float32, device) * 0.1f;
    auto b2 = zeros({1, hidden_features}, DType::Float32, device);
    auto W3 = randn({hidden_features, num_classes}, DType::Float32, device) * 0.1f;
    auto b3 = zeros({1, num_classes}, DType::Float32, device);

    showcase::print_section("Network Architecture");
    std::cout << "FC1: " << input_features << " -> " << hidden_features << "\n";
    std::cout << "Dropout(p=" << dropout_prob << ")\n";
    std::cout << "ReLU\n";
    std::cout << "FC2: " << hidden_features << " -> " << hidden_features << "\n";
    std::cout << "Dropout(p=" << dropout_prob << ")\n";
    std::cout << "ReLU\n";
    std::cout << "FC3: " << hidden_features << " -> " << num_classes << "\n";

    // Training parameters
    float learning_rate = 0.01f;
    int num_epochs = 300;
    int print_every = 30;

    showcase::print_section("Training (with Dropout)");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        bool training = true;

        // ============ Forward Pass ============
        // FC1
        auto z1 = matmul(X, W1) + b1;
        auto a1_pre = clamp_min(z1, 0.0f);  // ReLU
        auto a1 = manual_dropout(a1_pre, dropout_prob, training, gen);

        // FC2
        auto z2 = matmul(a1, W2) + b2;
        auto a2_pre = clamp_min(z2, 0.0f);  // ReLU
        auto a2 = manual_dropout(a2_pre, dropout_prob, training, gen);

        // FC3 (output)
        auto logits = matmul(a2, W3) + b3;

        // Softmax
        auto exp_logits = tenzor::exp(logits - tenzor::max(logits, 1, true));
        auto probs = exp_logits / tenzor::sum(exp_logits, 1, true);

        // ============ Compute Loss ============
        auto y_cpu = y.cpu();
        auto probs_cpu = probs.cpu();
        const int64_t* target_data = y_cpu.data<int64_t>();
        const float* prob_data = probs_cpu.data<float>();

        float loss = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            loss -= std::log(prob_data[b * num_classes + target_data[b]] + 1e-7f);
        }
        loss /= batch_size;

        // ============ Backward Pass (Simplified) ============
        std::vector<float> grad_data(batch_size * num_classes);
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_classes; ++c) {
                grad_data[b * num_classes + c] = prob_data[b * num_classes + c];
                if (c == target_data[b]) {
                    grad_data[b * num_classes + c] -= 1.0f;
                }
                grad_data[b * num_classes + c] /= batch_size;
            }
        }
        auto dL_dlogits = from_data(grad_data.data(), {batch_size, num_classes}, device);

        // Backprop through FC3
        auto dL_dW3 = matmul(a2.transpose(0, 1), dL_dlogits);
        auto dL_db3 = tenzor::sum(dL_dlogits, 0, true);

        // Backprop through FC2
        auto dL_da2 = matmul(dL_dlogits, W3.transpose(0, 1));
        auto dL_dz2 = dL_da2 * (z2 > zeros_like(z2)).to(DType::Float32);
        auto dL_dW2 = matmul(a1.transpose(0, 1), dL_dz2);
        auto dL_db2 = tenzor::sum(dL_dz2, 0, true);

        // Backprop through FC1
        auto dL_da1 = matmul(dL_dz2, W2.transpose(0, 1));
        auto dL_dz1 = dL_da1 * (z1 > zeros_like(z1)).to(DType::Float32);
        auto dL_dW1 = matmul(X.transpose(0, 1), dL_dz1);
        auto dL_db1 = tenzor::sum(dL_dz1, 0, true);

        // Update weights
        W3 = W3 - dL_dW3 * learning_rate;
        b3 = b3 - dL_db3 * learning_rate;
        W2 = W2 - dL_dW2 * learning_rate;
        b2 = b2 - dL_db2 * learning_rate;
        W1 = W1 - dL_dW1 * learning_rate;
        b1 = b1 - dL_db1 * learning_rate;

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float accuracy = showcase::multiclass_accuracy(logits.cpu(), y.cpu());
            showcase::print_progress(epoch, num_epochs, loss, accuracy);
        }
    }

    // ============ Final Results (Inference without Dropout) ============
    showcase::print_section("Final Results (Inference Mode - No Dropout)");

    // Inference without dropout
    auto z1_eval = matmul(X, W1) + b1;
    auto a1_eval = clamp_min(z1_eval, 0.0f);

    auto z2_eval = matmul(a1_eval, W2) + b2;
    auto a2_eval = clamp_min(z2_eval, 0.0f);

    auto logits_eval = matmul(a2_eval, W3) + b3;

    float final_accuracy = showcase::multiclass_accuracy(logits_eval.cpu(), y.cpu());
    std::cout << "Final Accuracy: " << (final_accuracy * 100.0f) << "%\n\n";

    std::cout << "Key Dropout behaviors:\n";
    std::cout << "  - Training: randomly zeros activations with prob " << dropout_prob << "\n";
    std::cout << "  - Training: scales remaining by 1/(1-p) for expectation matching\n";
    std::cout << "  - Inference: no dropout applied, use full network\n";
    std::cout << "  - Purpose: prevents overfitting by reducing co-adaptation\n";

    std::cout << "\nDropout demonstrated with manual tensors!\n";

    finalize();
    return 0;
}
