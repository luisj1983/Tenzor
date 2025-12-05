/**
 * @file autograd.cpp
 * @brief Dropout Regularization using Tenzor's automatic differentiation
 *
 * This example demonstrates dropout regularization using Variable and autograd.
 *
 * Usage: ./09_dropout_regularization_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <random>

using namespace tenzor;

// Dropout with Variable
Variable dropout_var(const Variable& x, float drop_prob, bool training, std::mt19937& gen) {
    if (!training || drop_prob == 0.0f) {
        return x;
    }

    // Generate dropout mask
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    int64_t numel = x.tensor().numel();
    std::vector<float> mask_data(numel);
    float scale = 1.0f / (1.0f - drop_prob);

    for (int64_t i = 0; i < numel; ++i) {
        mask_data[i] = (dist(gen) > drop_prob) ? scale : 0.0f;
    }

    // Convert shape span to vector
    std::vector<int64_t> shape_vec(x.shape().begin(), x.shape().end());
    auto mask_tensor = from_data(mask_data.data(), shape_vec, x.device());
    Variable mask(mask_tensor, false);  // Mask doesn't need gradients

    return x * mask;
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Dropout Regularization - Autograd", device);

    manual_seed(42);
    std::mt19937 gen(42);

    // Generate synthetic data
    int batch_size = 32;
    int input_features = 64;
    int hidden_features = 128;
    int num_classes = 4;
    float dropout_prob = 0.5f;

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

    auto X_tensor = from_data(X_data.data(), {batch_size, input_features}, device);
    auto y_tensor = from_data(y_data.data(), {batch_size}, device);

    showcase::print_tensor_info("Input X", X_tensor);

    // Initialize weights as Variables
    Variable W1(randn({input_features, hidden_features}, DType::Float32, device) * 0.1f, true);
    Variable b1(zeros({1, hidden_features}, DType::Float32, device), true);
    Variable W2(randn({hidden_features, hidden_features}, DType::Float32, device) * 0.1f, true);
    Variable b2(zeros({1, hidden_features}, DType::Float32, device), true);
    Variable W3(randn({hidden_features, num_classes}, DType::Float32, device) * 0.1f, true);
    Variable b3(zeros({1, num_classes}, DType::Float32, device), true);

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

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        bool training = true;

        Variable X(X_tensor, false);

        // ============ Forward Pass with Dropout ============
        // FC1 -> ReLU -> Dropout
        auto z1 = matmul(X, W1) + b1;
        auto a1_pre = nn::relu(z1);
        auto a1 = dropout_var(a1_pre, dropout_prob, training, gen);

        // FC2 -> ReLU -> Dropout
        auto z2 = matmul(a1, W2) + b2;
        auto a2_pre = nn::relu(z2);
        auto a2 = dropout_var(a2_pre, dropout_prob, training, gen);

        // FC3 (output)
        auto logits = matmul(a2, W3) + b3;

        // ============ Compute Loss ============
        auto log_probs = log_softmax(logits, 1);

        auto y_cpu = y_tensor.cpu();
        const int64_t* target_data = y_cpu.data<int64_t>();
        std::vector<float> one_hot_data(batch_size * num_classes, 0.0f);
        for (int b = 0; b < batch_size; ++b) {
            one_hot_data[b * num_classes + target_data[b]] = 1.0f;
        }
        auto one_hot_tensor = from_data(one_hot_data.data(), {batch_size, num_classes}, device);
        Variable one_hot(one_hot_tensor, false);

        auto loss = mean(sum(one_hot * log_probs, 1)) * (-1.0f);

        // ============ Backward Pass ============
        W1.zero_grad(); b1.zero_grad();
        W2.zero_grad(); b2.zero_grad();
        W3.zero_grad(); b3.zero_grad();

        loss.backward();

        // ============ Update Weights ============
        {
            NoGradGuard no_grad;
            W1 = Variable(W1.tensor() - (*W1.grad() * learning_rate), true);
            b1 = Variable(b1.tensor() - (*b1.grad() * learning_rate), true);
            W2 = Variable(W2.tensor() - (*W2.grad() * learning_rate), true);
            b2 = Variable(b2.tensor() - (*b2.grad() * learning_rate), true);
            W3 = Variable(W3.tensor() - (*W3.grad() * learning_rate), true);
            b3 = Variable(b3.tensor() - (*b3.grad() * learning_rate), true);
        }

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().cpu().item<float>();
            float accuracy = showcase::multiclass_accuracy(logits.tensor().cpu(), y_tensor.cpu());
            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    // ============ Final Results (Inference Mode) ============
    showcase::print_section("Final Results (Inference Mode)");

    // Inference without dropout
    Variable X_eval(X_tensor, false);
    auto z1_eval = matmul(X_eval, W1) + b1;
    auto a1_eval = nn::relu(z1_eval);
    // No dropout in inference

    auto z2_eval = matmul(a1_eval, W2) + b2;
    auto a2_eval = nn::relu(z2_eval);
    // No dropout in inference

    auto logits_eval = matmul(a2_eval, W3) + b3;

    float final_accuracy = showcase::multiclass_accuracy(logits_eval.tensor().cpu(), y_tensor.cpu());
    std::cout << "Final Accuracy: " << (final_accuracy * 100.0f) << "%\n\n";

    std::cout << "Dropout with autograd:\n";
    std::cout << "  - Gradients flow through non-dropped units\n";
    std::cout << "  - Dropped units contribute zero gradient\n";
    std::cout << "  - Scale factor ensures expected value matches\n";

    std::cout << "\nDropout demonstrated with autograd!\n";

    finalize();
    return 0;
}
