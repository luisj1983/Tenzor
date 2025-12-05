/**
 * @file autograd.cpp
 * @brief Batch Normalization using Tenzor's automatic differentiation
 *
 * This example demonstrates batch normalization using Variable and autograd.
 *
 * Usage: ./08_batch_normalization_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

// Batch normalization with autograd
Variable batch_norm_var(const Variable& x, const Variable& gamma, const Variable& beta,
                        float eps = 1e-5f) {
    auto mean_val = mean(x, 0, true);
    auto x_centered = x - mean_val;
    auto var_val = mean(x_centered * x_centered, 0, true);

    std::vector<int64_t> var_shape(var_val.shape().begin(), var_val.shape().end());
    auto eps_tensor = full(var_shape, eps, DType::Float32, x.device());
    Variable eps_var(eps_tensor, false);

    auto std_val = Variable(tenzor::sqrt(var_val.tensor() + eps_tensor), var_val.requires_grad());
    auto x_norm = x_centered / std_val;

    return x_norm * gamma + beta;
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Batch Normalization - Autograd", device);

    manual_seed(42);

    // Generate synthetic data
    int batch_size = 32;
    int input_features = 8;
    int num_classes = 3;

    std::vector<float> X_data(batch_size * input_features);
    for (int b = 0; b < batch_size; ++b) {
        for (int f = 0; f < input_features; ++f) {
            float feat_mean = f * 2.0f;
            float feat_std = 0.5f + f * 0.2f;
            X_data[b * input_features + f] = feat_mean + randn({1}, DType::Float32, device).cpu().data<float>()[0] * feat_std;
        }
    }
    auto X_tensor = from_data(X_data.data(), {batch_size, input_features}, device);

    std::vector<int64_t> y_data(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        y_data[i] = i % num_classes;
    }
    auto y_tensor = from_data(y_data.data(), {batch_size}, device);

    // Initialize weights as Variables
    Variable W1(randn({input_features, 16}, DType::Float32, device) * 0.1f, true);
    Variable b1(zeros({1, 16}, DType::Float32, device), true);
    Variable gamma1(ones({1, 16}, DType::Float32, device), true);
    Variable beta1(zeros({1, 16}, DType::Float32, device), true);

    Variable W2(randn({16, num_classes}, DType::Float32, device) * 0.1f, true);
    Variable b2(zeros({1, num_classes}, DType::Float32, device), true);

    // Training parameters
    float learning_rate = 0.1f;
    int num_epochs = 300;
    int print_every = 30;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable X(X_tensor, false);

        // ============ Forward Pass ============
        auto z1 = matmul(X, W1) + b1;
        auto z1_norm = batch_norm_var(z1, gamma1, beta1);
        auto a1 = nn::relu(z1_norm);
        auto logits = matmul(a1, W2) + b2;

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
        gamma1.zero_grad(); beta1.zero_grad();
        W2.zero_grad(); b2.zero_grad();

        loss.backward();

        // ============ Update Weights ============
        {
            NoGradGuard no_grad;
            W1 = Variable(W1.tensor() - (*W1.grad() * learning_rate), true);
            b1 = Variable(b1.tensor() - (*b1.grad() * learning_rate), true);
            gamma1 = Variable(gamma1.tensor() - (*gamma1.grad() * learning_rate), true);
            beta1 = Variable(beta1.tensor() - (*beta1.grad() * learning_rate), true);
            W2 = Variable(W2.tensor() - (*W2.grad() * learning_rate), true);
            b2 = Variable(b2.tensor() - (*b2.grad() * learning_rate), true);
        }

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();
            float accuracy = showcase::multiclass_accuracy(logits.tensor(), y_tensor);
            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    showcase::print_section("Final Results");
    std::cout << "BatchNorm demonstrated with autograd!\n";
    std::cout << "Gradients computed automatically through normalization.\n";

    finalize();
    return 0;
}
