/**
 * @file autograd.cpp
 * @brief Binary Classification using Tenzor's automatic differentiation
 *
 * This example demonstrates binary classification using Variable and autograd
 * for automatic gradient computation.
 *
 * Usage: ./03_binary_classification_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

int main(int argc, char* argv[]) {
    // Parse backend from command line
    Device device = showcase::get_device_from_args(argc, argv);

    // Initialize Tenzor
    initialize();

    showcase::print_header("Binary Classification - Autograd (Automatic Differentiation)", device);

    // Set seed for reproducibility
    manual_seed(42);

    // Generate synthetic binary classification data
    int samples_per_class = 50;
    int num_samples = samples_per_class * 2;

    // Class 0: centered around (-2, -2)
    auto class0_x = randn({samples_per_class, 2}, DType::Float32, device) + (-2.0f);

    // Class 1: centered around (2, 2)
    auto class1_x = randn({samples_per_class, 2}, DType::Float32, device) + 2.0f;

    // Combine into dataset
    auto X_tensor = cat({class0_x, class1_x}, 0);  // (100, 2)

    // Create labels
    auto labels0 = zeros({samples_per_class, 1}, DType::Float32, device);
    auto labels1 = ones({samples_per_class, 1}, DType::Float32, device);
    auto y_tensor = cat({labels0, labels1}, 0);  // (100, 1)

    showcase::print_tensor_info("Input X", X_tensor);
    showcase::print_tensor_info("Labels y", y_tensor);

    // Initialize parameters as Variables with gradient tracking
    auto W_tensor = randn({2, 1}, DType::Float32, device) * 0.1f;
    auto b_tensor = zeros({1, 1}, DType::Float32, device);

    Variable W(W_tensor, true);
    Variable b(b_tensor, true);

    showcase::print_section("Initial Parameters");
    std::cout << "W: requires_grad=" << W.requires_grad() << "\n";
    std::cout << "b: requires_grad=" << b.requires_grad() << "\n";

    // Training parameters
    float learning_rate = 0.1f;
    int num_epochs = 500;
    int print_every = 50;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Wrap data as Variables
        Variable X(X_tensor, false);
        Variable y(y_tensor, false);

        // ============ Forward Pass ============
        auto z = matmul(X, W) + b;        // Logits
        auto p = nn::sigmoid(z);           // Probabilities

        // ============ Compute Loss (BCE) ============
        // BCE = -mean(y * log(p) + (1-y) * log(1-p))
        std::vector<int64_t> p_shape(p.shape().begin(), p.shape().end());
        auto eps_tensor = full(p_shape, 1e-7f, DType::Float32, device);
        Variable eps(eps_tensor, false);

        auto log_p = log(p + eps);
        auto log_1_minus_p = log(Variable(ones_like(p.tensor()), false) - p + eps);

        auto loss_per_sample = y * log_p + (Variable(ones_like(y.tensor()), false) - y) * log_1_minus_p;
        auto loss = mean(loss_per_sample) * (-1.0f);

        // ============ Backward Pass (Automatic!) ============
        W.zero_grad();
        b.zero_grad();
        loss.backward();

        // ============ Update Weights (Manual SGD) ============
        {
            NoGradGuard no_grad;

            auto W_new = W.tensor() - (*W.grad() * learning_rate);
            W = Variable(W_new, true);

            auto b_new = b.tensor() - (*b.grad() * learning_rate);
            b = Variable(b_new, true);
        }

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();

            // Calculate accuracy
            auto pred_cpu = p.tensor().cpu();
            auto target_cpu = y_tensor.cpu();
            int correct = 0;
            const float* pred = pred_cpu.data<float>();
            const float* tgt = target_cpu.data<float>();
            for (int i = 0; i < num_samples; ++i) {
                float predicted = (pred[i] > 0.5f) ? 1.0f : 0.0f;
                if (predicted == tgt[i]) correct++;
            }
            float accuracy = static_cast<float>(correct) / num_samples;

            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    // Final predictions
    Variable X_final(X_tensor, false);
    auto z_final = matmul(X_final, W) + b;
    auto p_final = nn::sigmoid(z_final);

    auto pred_cpu = p_final.tensor().cpu();
    auto target_cpu = y_tensor.cpu();
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
    auto W_cpu = W.tensor().cpu();
    auto b_cpu = b.tensor().cpu();
    std::cout << "Learned parameters:\n";
    std::cout << "  W = [" << W_cpu.data<float>()[0] << ", "
              << W_cpu.data<float>()[1] << "]\n";
    std::cout << "  b = " << b_cpu.data<float>()[0] << "\n";

    std::cout << "\nBinary classification solved using autograd!\n";
    std::cout << "Gradients computed automatically through the BCE loss.\n";

    finalize();
    return 0;
}
