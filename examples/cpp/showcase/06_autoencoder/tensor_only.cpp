/**
 * @file tensor_only.cpp
 * @brief Autoencoder using raw Tensor operations only
 *
 * This example demonstrates an autoencoder for dimensionality reduction
 * and feature learning using only tensor operations with manual backprop.
 *
 * Autoencoders learn to compress data into a lower-dimensional representation
 * and then reconstruct it.
 *
 * Usage: ./06_autoencoder_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

// Sigmoid activation
Tensor sigmoid_tensor(const Tensor& x) {
    auto neg_x = x * -1.0f;
    auto exp_neg_x = tenzor::exp(neg_x);
    return ones_like(x) / (exp_neg_x + 1.0f);
}

// Sigmoid derivative, given sigmoid(x)
Tensor sigmoid_deriv(const Tensor& sigmoid_out) {
    return sigmoid_out * (ones_like(sigmoid_out) - sigmoid_out);
}

// ReLU activation
Tensor relu_tensor(const Tensor& x) {
    return maximum(x, zeros_like(x));
}

// ReLU derivative mask, given the pre-activation z
Tensor relu_deriv(const Tensor& z) {
    // 1 where z > 0, 0 otherwise
    return (z > zeros_like(z)).to(DType::Float32);
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Autoencoder - Tensor Only (Manual Backprop)", device);

    manual_seed(42);

    // Generate synthetic data with real low-dimensional structure so that
    // a latent_dim=4 bottleneck is actually able to reconstruct it.
    // Each 16-d sample is a mixture of 4 sinusoidal basis functions,
    // shifted to [0, 1] so a sigmoid decoder can fit it.
    int batch_size = 32;
    int input_dim = 16;
    int hidden_dim = 12;
    int latent_dim = 4;

    std::vector<float> X_data(batch_size * input_dim);
    for (int b = 0; b < batch_size; ++b) {
        float c[4];
        for (int k = 0; k < 4; ++k) {
            c[k] = ((rand() % 1000) / 1000.0f) - 0.5f;  // uniform in [-0.5, 0.5]
        }
        for (int j = 0; j < input_dim; ++j) {
            float t = static_cast<float>(j) / input_dim;
            float signal = c[0] * std::sin(2.0f * 3.14159f * t) +
                           c[1] * std::cos(2.0f * 3.14159f * t) +
                           c[2] * std::sin(4.0f * 3.14159f * t) +
                           c[3] * std::cos(4.0f * 3.14159f * t);
            // rescale to ~[0, 1]
            X_data[b * input_dim + j] = 0.5f + 0.25f * signal;
        }
    }
    auto X = from_data(X_data.data(), {batch_size, input_dim}, device);

    showcase::print_tensor_info("Input X", X);
    std::cout << "Architecture: " << input_dim << " -> " << hidden_dim
              << " -> " << latent_dim << " -> " << hidden_dim << " -> " << input_dim << "\n";

    // He-style init for ReLU path, Xavier-ish for the sigmoid output layer.
    auto W1 = randn({input_dim, hidden_dim}, DType::Float32, device)
            * std::sqrt(2.0f / input_dim);
    auto b1 = zeros({1, hidden_dim}, DType::Float32, device);
    auto W2 = randn({hidden_dim, latent_dim}, DType::Float32, device)
            * std::sqrt(2.0f / hidden_dim);
    auto b2 = zeros({1, latent_dim}, DType::Float32, device);

    auto W3 = randn({latent_dim, hidden_dim}, DType::Float32, device)
            * std::sqrt(2.0f / latent_dim);
    auto b3 = zeros({1, hidden_dim}, DType::Float32, device);
    auto W4 = randn({hidden_dim, input_dim}, DType::Float32, device)
            * std::sqrt(1.0f / hidden_dim);
    auto b4 = zeros({1, input_dim}, DType::Float32, device);

    // Training parameters
    float learning_rate = 0.05f;
    int num_epochs = 1000;
    int print_every = 100;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Encoder: Linear -> ReLU -> Linear (linear latent)
        auto z1 = matmul(X, W1) + b1;
        auto a1 = relu_tensor(z1);
        auto latent = matmul(a1, W2) + b2;

        // Decoder: Linear -> ReLU -> Linear -> Sigmoid
        auto z3 = matmul(latent, W3) + b3;
        auto a3 = relu_tensor(z3);
        auto z4 = matmul(a3, W4) + b4;
        auto reconstruction = sigmoid_tensor(z4);

        // Loss (MSE)
        auto error = reconstruction - X;
        auto loss = tenzor::mean(error * error);
        float loss_val = loss.item<float>();

        // Backward
        float n = static_cast<float>(batch_size * input_dim);  // matches reduction=mean
        auto dL_drecon = error * (2.0f / n);
        auto dL_dz4 = dL_drecon * sigmoid_deriv(reconstruction);
        auto dL_dW4 = matmul(a3.transpose(0, 1), dL_dz4);
        auto dL_db4 = tenzor::sum(dL_dz4, 0, true);

        auto dL_da3 = matmul(dL_dz4, W4.transpose(0, 1));
        auto dL_dz3 = dL_da3 * relu_deriv(z3);
        auto dL_dW3 = matmul(latent.transpose(0, 1), dL_dz3);
        auto dL_db3 = tenzor::sum(dL_dz3, 0, true);

        // Latent is pre-activation (linear) - gradient passes through
        auto dL_dlatent = matmul(dL_dz3, W3.transpose(0, 1));
        auto dL_dW2 = matmul(a1.transpose(0, 1), dL_dlatent);
        auto dL_db2 = tenzor::sum(dL_dlatent, 0, true);

        auto dL_da1 = matmul(dL_dlatent, W2.transpose(0, 1));
        auto dL_dz1 = dL_da1 * relu_deriv(z1);
        auto dL_dW1 = matmul(X.transpose(0, 1), dL_dz1);
        auto dL_db1 = tenzor::sum(dL_dz1, 0, true);

        // ============ Update Weights ============
        W4 = W4 - dL_dW4 * learning_rate;
        b4 = b4 - dL_db4 * learning_rate;
        W3 = W3 - dL_dW3 * learning_rate;
        b3 = b3 - dL_db3 * learning_rate;
        W2 = W2 - dL_dW2 * learning_rate;
        b2 = b2 - dL_db2 * learning_rate;
        W1 = W1 - dL_dW1 * learning_rate;
        b1 = b1 - dL_db1 * learning_rate;

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                      << "Reconstruction Loss: " << loss_val << "\n";
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    // Final forward pass
    auto z1 = matmul(X, W1) + b1;
    auto a1 = relu_tensor(z1);
    auto latent = matmul(a1, W2) + b2;
    auto z3 = matmul(latent, W3) + b3;
    auto a3 = relu_tensor(z3);
    auto z4 = matmul(a3, W4) + b4;
    auto reconstruction = sigmoid_tensor(z4);

    auto final_error = reconstruction - X;
    float final_loss = tenzor::mean(final_error * final_error).item<float>();

    std::cout << "Final Reconstruction Loss: " << final_loss << "\n\n";

    // Show sample reconstructions
    std::cout << "Sample Input vs Reconstruction (first 4 features of first 3 samples):\n";
    auto X_cpu = X.cpu();
    auto recon_cpu = reconstruction.cpu();
    auto latent_cpu = latent.cpu();

    for (int i = 0; i < 3; ++i) {
        std::cout << "Sample " << i << ":\n";
        std::cout << "  Input:   [";
        for (int j = 0; j < 4; ++j) {
            std::cout << X_cpu.data<float>()[i * input_dim + j];
            if (j < 3) std::cout << ", ";
        }
        std::cout << ", ...]\n";

        std::cout << "  Latent:  [";
        for (int j = 0; j < latent_dim; ++j) {
            std::cout << latent_cpu.data<float>()[i * latent_dim + j];
            if (j < latent_dim - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "  Recon:   [";
        for (int j = 0; j < 4; ++j) {
            std::cout << recon_cpu.data<float>()[i * input_dim + j];
            if (j < 3) std::cout << ", ";
        }
        std::cout << ", ...]\n\n";
    }

    std::cout << "Autoencoder demonstrated with manual tensor operations!\n";
    std::cout << "Data compressed from " << input_dim << " dims to " << latent_dim << " dims.\n";

    finalize();
    return 0;
}
