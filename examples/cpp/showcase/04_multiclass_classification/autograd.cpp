/**
 * @file autograd.cpp
 * @brief Multi-class Classification using Tenzor's automatic differentiation
 *
 * This example demonstrates multi-class classification using Variable, autograd,
 * and the built-in softmax/log_softmax operations.
 *
 * Usage: ./04_multiclass_classification_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Multi-class Classification - Autograd (Automatic Differentiation)", device);

    manual_seed(42);

    // Generate synthetic multi-class data: 3 classes in 2D
    int samples_per_class = 50;
    int num_classes = 3;
    int num_samples = samples_per_class * num_classes;
    int num_features = 2;

    // Class 0: centered around (0, 3)
    auto class0_x = randn({samples_per_class, 2}, DType::Float32, device);
    class0_x = class0_x + from_data(std::vector<float>{0.0f, 3.0f}.data(), {1, 2}, device);

    // Class 1: centered around (-3, -1)
    auto class1_x = randn({samples_per_class, 2}, DType::Float32, device);
    class1_x = class1_x + from_data(std::vector<float>{-3.0f, -1.0f}.data(), {1, 2}, device);

    // Class 2: centered around (3, -1)
    auto class2_x = randn({samples_per_class, 2}, DType::Float32, device);
    class2_x = class2_x + from_data(std::vector<float>{3.0f, -1.0f}.data(), {1, 2}, device);

    auto X_tensor = cat({class0_x, class1_x, class2_x}, 0);  // (150, 2)

    // Create labels
    std::vector<int64_t> label_data;
    for (int c = 0; c < num_classes; ++c) {
        for (int i = 0; i < samples_per_class; ++i) {
            label_data.push_back(c);
        }
    }
    auto y_tensor = from_data(label_data.data(), {num_samples}, device);

    showcase::print_tensor_info("Input X", X_tensor);
    showcase::print_tensor_info("Labels y", y_tensor);

    // Initialize parameters with gradient tracking
    auto W_tensor = randn({num_features, num_classes}, DType::Float32, device) * 0.1f;
    auto b_tensor = zeros({1, num_classes}, DType::Float32, device);

    Variable W(W_tensor, true);
    Variable b(b_tensor, true);

    showcase::print_section("Initial Parameters");
    std::cout << "W: shape=(" << W.shape()[0] << ", " << W.shape()[1]
              << "), requires_grad=" << W.requires_grad() << "\n";

    // Training parameters
    float learning_rate = 0.1f;
    int num_epochs = 500;
    int print_every = 50;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable X(X_tensor, false);

        // ============ Forward Pass ============
        auto z = matmul(X, W) + b;                    // Logits
        auto log_probs = log_softmax(z, 1);          // Log-softmax for stability

        // ============ Compute Loss (NLL Loss) ============
        // Manual NLL: -mean(log_probs[i, target[i]])
        auto log_probs_cpu = log_probs.tensor().cpu();
        auto y_cpu = y_tensor.cpu();
        const float* lp_data = log_probs_cpu.data<float>();
        const int64_t* target_data = y_cpu.data<int64_t>();

        float nll_loss_val = 0.0f;
        for (int i = 0; i < num_samples; ++i) {
            nll_loss_val -= lp_data[i * num_classes + target_data[i]];
        }
        nll_loss_val /= num_samples;

        // For autograd, we need to compute the loss as a Variable
        // Use cross-entropy directly: -sum(one_hot * log_probs) / batch_size
        std::vector<float> one_hot_data(num_samples * num_classes, 0.0f);
        for (int i = 0; i < num_samples; ++i) {
            one_hot_data[i * num_classes + target_data[i]] = 1.0f;
        }
        auto one_hot_tensor = from_data(one_hot_data.data(), {num_samples, num_classes}, device);
        Variable one_hot(one_hot_tensor, false);

        auto loss_per_sample = one_hot * log_probs;
        auto loss = mean(sum(loss_per_sample, 1)) * (-1.0f);

        // ============ Backward Pass ============
        W.zero_grad();
        b.zero_grad();
        loss.backward();

        // ============ Update Weights ============
        {
            NoGradGuard no_grad;

            auto W_new = W.tensor() - (*W.grad() * learning_rate);
            W = Variable(W_new, true);

            auto b_new = b.tensor() - (*b.grad() * learning_rate);
            b = Variable(b_new, true);
        }

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            // Compute probabilities for accuracy
            auto probs = softmax(z, 1);
            float accuracy = showcase::multiclass_accuracy(probs.tensor(), y_tensor);
            showcase::print_progress(epoch, num_epochs, nll_loss_val, accuracy);
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    Variable X_final(X_tensor, false);
    auto z_final = matmul(X_final, W) + b;
    auto probs_final = softmax(z_final, 1);

    float final_accuracy = showcase::multiclass_accuracy(probs_final.tensor(), y_tensor);
    std::cout << "Final Accuracy: " << (final_accuracy * 100.0f) << "%\n";

    std::cout << "\nMulti-class classification solved using autograd!\n";
    std::cout << "Log-softmax provides numerical stability for classification.\n";

    finalize();
    return 0;
}
