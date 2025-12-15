/**
 * @file tensor_only.cpp
 * @brief Batch Normalization using raw Tensor operations only
 *
 * This example demonstrates batch normalization using only tensor operations.
 * BatchNorm normalizes activations to have zero mean and unit variance,
 * then applies a learnable scale (gamma) and shift (beta).
 *
 * BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * Usage: ./08_batch_normalization_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>

using namespace tenzor;

// Manual batch normalization
struct BatchNormResult {
    Tensor output;
    Tensor mean;
    Tensor var;
};

BatchNormResult batch_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                           float eps = 1e-5f) {
    // Compute mean and variance over batch dimension (dim=0)
    auto mean = tenzor::mean(x, 0, true);    // (1, features)
    auto x_centered = x - mean;               // (batch, features)

    auto var = tenzor::mean(x_centered * x_centered, 0, true);  // (1, features)

    // Normalize
    auto std = tenzor::sqrt(var + eps);
    auto x_norm = x_centered / std;

    // Scale and shift
    auto output = x_norm * gamma + beta;

    return {output, mean, var};
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Batch Normalization - Tensor Only (Manual)", device);

    manual_seed(42);

    // Generate synthetic data with different statistics per feature
    int batch_size = 32;
    int input_features = 8;
    int num_classes = 3;

    // Features with different means and variances
    std::vector<float> X_data(batch_size * input_features);
    for (int b = 0; b < batch_size; ++b) {
        for (int f = 0; f < input_features; ++f) {
            // Each feature has different mean (f*2) and variance
            float mean = f * 2.0f;
            float std = 0.5f + f * 0.2f;
            X_data[b * input_features + f] = mean + randn({1}, DType::Float32, device).cpu().data<float>()[0] * std;
        }
    }
    auto X = from_data(X_data.data(), {batch_size, input_features}, device);

    // Create labels
    std::vector<int64_t> y_data(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        y_data[i] = i % num_classes;
    }
    auto y = from_data(y_data.data(), {batch_size}, device);

    showcase::print_tensor_info("Input X", X);

    // Show input statistics before normalization
    std::cout << "\nInput statistics (per feature):\n";
    auto X_mean = tenzor::mean(X, 0, true).cpu();
    auto X_var = tenzor::mean((X - tenzor::mean(X, 0, true)) * (X - tenzor::mean(X, 0, true)), 0, true).cpu();
    for (int f = 0; f < std::min(4, input_features); ++f) {
        std::cout << "  Feature " << f << ": mean=" << X_mean.data<float>()[f]
                  << ", var=" << X_var.data<float>()[f] << "\n";
    }

    // Initialize network weights
    // Layer 1: input -> hidden (with BatchNorm)
    auto W1 = randn({input_features, 16}, DType::Float32, device) * 0.1f;
    auto b1 = zeros({1, 16}, DType::Float32, device);
    auto gamma1 = ones({1, 16}, DType::Float32, device);  // BatchNorm scale
    auto beta1 = zeros({1, 16}, DType::Float32, device);  // BatchNorm shift

    // Layer 2: hidden -> output
    auto W2 = randn({16, num_classes}, DType::Float32, device) * 0.1f;
    auto b2 = zeros({1, num_classes}, DType::Float32, device);

    showcase::print_section("Network Architecture");
    std::cout << "FC1: " << input_features << " -> 16\n";
    std::cout << "BatchNorm1: normalized, then scale (gamma) and shift (beta)\n";
    std::cout << "ReLU\n";
    std::cout << "FC2: 16 -> " << num_classes << "\n";

    // Training parameters
    float learning_rate = 0.1f;
    int num_epochs = 300;
    int print_every = 30;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ============ Forward Pass ============
        // FC1
        auto z1 = matmul(X, W1) + b1;

        // BatchNorm
        auto bn_result = batch_norm(z1, gamma1, beta1);
        auto z1_norm = bn_result.output;

        // ReLU
        auto a1 = clamp_min(z1_norm, 0.0f);

        // FC2
        auto logits = matmul(a1, W2) + b2;

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
        // Only update FC2 and FC1 for simplicity
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

        auto dL_dW2 = matmul(a1.transpose(0, 1), dL_dlogits);
        auto dL_db2 = tenzor::sum(dL_dlogits, 0, true);

        // Backprop through FC1 (simplified, ignoring BN backward for now)
        auto dL_da1 = matmul(dL_dlogits, W2.transpose(0, 1));
        auto dL_dz1_norm = dL_da1 * (z1_norm > zeros_like(z1_norm)).to(DType::Float32);
        auto dL_dW1 = matmul(X.transpose(0, 1), dL_dz1_norm);
        auto dL_db1 = tenzor::sum(dL_dz1_norm, 0, true);

        // Update weights
        W2 = W2 - dL_dW2 * learning_rate;
        b2 = b2 - dL_db2 * learning_rate;
        W1 = W1 - dL_dW1 * learning_rate;
        b1 = b1 - dL_db1 * learning_rate;

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float accuracy = showcase::multiclass_accuracy(logits, y);
            showcase::print_progress(epoch, num_epochs, loss, accuracy);
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    // Show normalized statistics
    auto z1_final = matmul(X, W1) + b1;
    auto bn_final = batch_norm(z1_final, gamma1, beta1);

    std::cout << "After BatchNorm (first 4 features):\n";
    auto bn_mean = tenzor::mean(bn_final.output, 0, true).cpu();
    auto bn_centered = bn_final.output - tenzor::mean(bn_final.output, 0, true);
    auto bn_var = tenzor::mean(bn_centered * bn_centered, 0, true).cpu();
    for (int f = 0; f < 4; ++f) {
        std::cout << "  Feature " << f << ": mean=" << bn_mean.data<float>()[f]
                  << ", var=" << bn_var.data<float>()[f] << "\n";
    }

    std::cout << "\nBatch Normalization demonstrated with manual tensors!\n";
    std::cout << "Normalizes activations to zero mean and unit variance.\n";

    finalize();
    return 0;
}
