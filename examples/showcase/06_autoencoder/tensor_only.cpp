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

using namespace tenzor;

// Sigmoid activation
Tensor sigmoid_tensor(const Tensor& x) {
    auto neg_x = x * -1.0f;
    auto exp_neg_x = tenzor::exp(neg_x);
    return ones_like(x) / (exp_neg_x + 1.0f);
}

// Sigmoid derivative
Tensor sigmoid_deriv(const Tensor& sigmoid_out) {
    return sigmoid_out * (ones_like(sigmoid_out) - sigmoid_out);
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Autoencoder - Tensor Only (Manual Backprop)", device);

    manual_seed(42);

    // Generate synthetic data
    // We'll use random "images" (vectors) and try to reconstruct them
    int batch_size = 32;
    int input_dim = 16;
    int hidden_dim = 8;
    int latent_dim = 4;

    // Random input data
    auto X = rand({batch_size, input_dim}, DType::Float32, device);

    showcase::print_tensor_info("Input X", X);
    std::cout << "Architecture: " << input_dim << " -> " << hidden_dim
              << " -> " << latent_dim << " -> " << hidden_dim << " -> " << input_dim << "\n";

    // Initialize weights
    // Encoder: input -> hidden -> latent
    auto W1 = randn({input_dim, hidden_dim}, DType::Float32, device) * 0.1f;
    auto b1 = zeros({1, hidden_dim}, DType::Float32, device);
    auto W2 = randn({hidden_dim, latent_dim}, DType::Float32, device) * 0.1f;
    auto b2 = zeros({1, latent_dim}, DType::Float32, device);

    // Decoder: latent -> hidden -> output
    auto W3 = randn({latent_dim, hidden_dim}, DType::Float32, device) * 0.1f;
    auto b3 = zeros({1, hidden_dim}, DType::Float32, device);
    auto W4 = randn({hidden_dim, input_dim}, DType::Float32, device) * 0.1f;
    auto b4 = zeros({1, input_dim}, DType::Float32, device);

    // Training parameters
    float learning_rate = 0.5f;
    int num_epochs = 1000;
    int print_every = 100;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ============ Forward Pass (Encoder) ============
        auto z1 = matmul(X, W1) + b1;
        auto a1 = sigmoid_tensor(z1);  // Hidden

        auto z2 = matmul(a1, W2) + b2;
        auto a2 = sigmoid_tensor(z2);  // Latent

        // ============ Forward Pass (Decoder) ============
        auto z3 = matmul(a2, W3) + b3;
        auto a3 = sigmoid_tensor(z3);  // Hidden

        auto z4 = matmul(a3, W4) + b4;
        auto reconstruction = sigmoid_tensor(z4);  // Output

        // ============ Compute Loss (MSE) ============
        auto error = reconstruction - X;
        auto loss = tenzor::mean(error * error);
        float loss_val = loss.item<float>();

        // ============ Backward Pass ============
        float n = static_cast<float>(batch_size);

        // Output layer gradients
        auto dL_drecon = error * (2.0f / n);
        auto dL_dz4 = dL_drecon * sigmoid_deriv(reconstruction);
        auto dL_dW4 = matmul(a3.transpose(0, 1), dL_dz4);
        auto dL_db4 = tenzor::sum(dL_dz4, 0, true);

        // Decoder hidden layer
        auto dL_da3 = matmul(dL_dz4, W4.transpose(0, 1));
        auto dL_dz3 = dL_da3 * sigmoid_deriv(a3);
        auto dL_dW3 = matmul(a2.transpose(0, 1), dL_dz3);
        auto dL_db3 = tenzor::sum(dL_dz3, 0, true);

        // Latent layer
        auto dL_da2 = matmul(dL_dz3, W3.transpose(0, 1));
        auto dL_dz2 = dL_da2 * sigmoid_deriv(a2);
        auto dL_dW2 = matmul(a1.transpose(0, 1), dL_dz2);
        auto dL_db2 = tenzor::sum(dL_dz2, 0, true);

        // Encoder hidden layer
        auto dL_da1 = matmul(dL_dz2, W2.transpose(0, 1));
        auto dL_dz1 = dL_da1 * sigmoid_deriv(a1);
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
    auto a1 = sigmoid_tensor(z1);
    auto z2 = matmul(a1, W2) + b2;
    auto latent = sigmoid_tensor(z2);
    auto z3 = matmul(latent, W3) + b3;
    auto a3 = sigmoid_tensor(z3);
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
