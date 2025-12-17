/**
 * @file tensor_only.cpp
 * @brief Multi-class Classification using raw Tensor operations only
 *
 * This example demonstrates multi-class classification (softmax regression)
 * using only tensor operations with manual gradient computation.
 *
 * Multi-class classification uses softmax activation and cross-entropy loss.
 *
 * Usage: ./04_multiclass_classification_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>

using namespace tenzor;

// Manual softmax implementation
Tensor manual_softmax(const Tensor& x) {
    // Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    // Subtract max for numerical stability
    auto max_x = tenzor::max(x, 1, true);  // Max along class dimension

    // Broadcast max to same shape as x
    auto x_stable = x - max_x;
    auto exp_x = tenzor::exp(x_stable);
    auto sum_exp = tenzor::sum(exp_x, 1, true);

    return exp_x / sum_exp;
}

// Manual cross-entropy loss
float cross_entropy_loss(const Tensor& probs, const Tensor& targets) {
    // targets: (batch_size,) containing class indices
    // probs: (batch_size, num_classes)
    int batch_size = probs.shape()[0];
    int num_classes = probs.shape()[1];

    // Copy tensors to CPU first to avoid dangling pointer from temporary
    auto probs_cpu = probs.cpu();
    auto targets_cpu = targets.cpu();
    const float* prob_data = probs_cpu.data<float>();
    const int64_t* target_data = targets_cpu.data<int64_t>();

    float total_loss = 0.0f;
    const float eps = 1e-7f;

    for (int i = 0; i < batch_size; ++i) {
        int target_class = target_data[i];
        float prob = prob_data[i * num_classes + target_class];
        total_loss -= std::log(prob + eps);
    }

    return total_loss / batch_size;
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Multi-class Classification - Tensor Only (Manual Backprop)", device);

    manual_seed(42);

    // Generate synthetic multi-class data: 3 classes in 2D
    int samples_per_class = 50;
    int num_classes = 3;
    int num_samples = samples_per_class * num_classes;
    int num_features = 2;

    // Class 0: centered around (0, 3)
    auto class0_x = randn({samples_per_class, 2}, DType::Float32, device);
    auto class0_offset = from_data(std::vector<float>{0.0f, 3.0f}.data(), {1, 2}, device);
    class0_x = class0_x + class0_offset;

    // Class 1: centered around (-3, -1)
    auto class1_x = randn({samples_per_class, 2}, DType::Float32, device);
    auto class1_offset = from_data(std::vector<float>{-3.0f, -1.0f}.data(), {1, 2}, device);
    class1_x = class1_x + class1_offset;

    // Class 2: centered around (3, -1)
    auto class2_x = randn({samples_per_class, 2}, DType::Float32, device);
    auto class2_offset = from_data(std::vector<float>{3.0f, -1.0f}.data(), {1, 2}, device);
    class2_x = class2_x + class2_offset;

    // Combine into dataset
    auto X = cat({class0_x, class1_x, class2_x}, 0);  // (150, 2)

    // Create labels: 0, 1, 2
    std::vector<int64_t> label_data;
    for (int c = 0; c < num_classes; ++c) {
        for (int i = 0; i < samples_per_class; ++i) {
            label_data.push_back(c);
        }
    }
    auto y = from_data(label_data.data(), {num_samples}, device);

    showcase::print_tensor_info("Input X", X);
    showcase::print_tensor_info("Labels y", y);

    // Initialize weights: y = softmax(X @ W + b)
    auto W = randn({num_features, num_classes}, DType::Float32, device) * 0.1f;  // (2, 3)
    auto b = zeros({1, num_classes}, DType::Float32, device);  // (1, 3)

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
        auto z = matmul(X, W) + b;      // (150, 3) - logits
        auto p = manual_softmax(z);      // (150, 3) - probabilities

        // ============ Compute Loss ============
        float loss = cross_entropy_loss(p, y);

        // ============ Backward Pass (Manual Gradient Computation) ============
        // For cross-entropy + softmax, gradient simplifies to:
        // dL/dz = p - one_hot(y)

        // Create one-hot encoding of targets
        auto y_cpu = y.cpu();
        const int64_t* target_data = y_cpu.data<int64_t>();
        std::vector<float> one_hot_data(num_samples * num_classes, 0.0f);
        for (int i = 0; i < num_samples; ++i) {
            one_hot_data[i * num_classes + target_data[i]] = 1.0f;
        }
        auto one_hot = from_data(one_hot_data.data(), {num_samples, num_classes}, device);

        // dL/dz = (p - one_hot) / batch_size
        auto dL_dz = (p - one_hot) / static_cast<float>(num_samples);

        // dL/dW = X^T @ dL/dz
        auto dL_dW = matmul(X.transpose(0, 1), dL_dz);  // (2, 3)

        // dL/db = sum(dL/dz, dim=0)
        auto dL_db = tenzor::sum(dL_dz, 0, true);  // (1, 3)

        // ============ Update Weights ============
        W = W - dL_dW * learning_rate;
        b = b - dL_db * learning_rate;

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            // Calculate accuracy
            float accuracy = showcase::multiclass_accuracy(p, y);
            showcase::print_progress(epoch, num_epochs, loss, accuracy);
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    auto z_final = matmul(X, W) + b;
    auto p_final = manual_softmax(z_final);

    float final_accuracy = showcase::multiclass_accuracy(p_final, y);
    std::cout << "Final Accuracy: " << (final_accuracy * 100.0f) << "%\n\n";

    // Show confusion matrix
    auto p_cpu = p_final.cpu();
    auto y_cpu = y.cpu();
    const float* prob_data = p_cpu.data<float>();
    const int64_t* target_data = y_cpu.data<int64_t>();

    int confusion[3][3] = {{0}};
    for (int i = 0; i < num_samples; ++i) {
        int true_class = target_data[i];
        int pred_class = 0;
        float max_prob = prob_data[i * num_classes];
        for (int c = 1; c < num_classes; ++c) {
            if (prob_data[i * num_classes + c] > max_prob) {
                max_prob = prob_data[i * num_classes + c];
                pred_class = c;
            }
        }
        confusion[true_class][pred_class]++;
    }

    std::cout << "Confusion Matrix:\n";
    std::cout << "\t\tPredicted\n";
    std::cout << "\t\t0\t1\t2\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "True " << i << "\t\t";
        for (int j = 0; j < 3; ++j) {
            std::cout << confusion[i][j] << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "\nMulti-class classification solved using raw tensors!\n";

    finalize();
    return 0;
}
