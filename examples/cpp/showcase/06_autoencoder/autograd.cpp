/**
 * @file autograd.cpp
 * @brief Autoencoder using Tenzor's automatic differentiation
 *
 * This example demonstrates an autoencoder using Variable and autograd
 * for automatic gradient computation.
 *
 * Usage: ./06_autoencoder_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Autoencoder - Autograd (Automatic Differentiation)", device);

    manual_seed(42);

    // Generate synthetic data with real low-dimensional structure so that
    // a latent_dim=4 bottleneck is actually able to reconstruct it.
    int batch_size = 32;
    int input_dim = 16;
    int hidden_dim = 12;
    int latent_dim = 4;

    std::vector<float> X_data(batch_size * input_dim);
    for (int b = 0; b < batch_size; ++b) {
        float c[4];
        for (int k = 0; k < 4; ++k) {
            c[k] = ((rand() % 1000) / 1000.0f) - 0.5f;
        }
        for (int j = 0; j < input_dim; ++j) {
            float t = static_cast<float>(j) / input_dim;
            float signal = c[0] * std::sin(2.0f * 3.14159f * t) +
                           c[1] * std::cos(2.0f * 3.14159f * t) +
                           c[2] * std::sin(4.0f * 3.14159f * t) +
                           c[3] * std::cos(4.0f * 3.14159f * t);
            X_data[b * input_dim + j] = 0.5f + 0.25f * signal;
        }
    }
    auto X_tensor = from_data(X_data.data(), {batch_size, input_dim}, device);

    showcase::print_tensor_info("Input X", X_tensor);
    std::cout << "Architecture: " << input_dim << " -> " << hidden_dim
              << " -> " << latent_dim << " -> " << hidden_dim << " -> " << input_dim << "\n";

    // He-style init for the ReLU path, Xavier for the sigmoid output
    Variable W1(randn({input_dim, hidden_dim}, DType::Float32, device)
              * std::sqrt(2.0f / input_dim), true);
    Variable b1(zeros({1, hidden_dim}, DType::Float32, device), true);
    Variable W2(randn({hidden_dim, latent_dim}, DType::Float32, device)
              * std::sqrt(2.0f / hidden_dim), true);
    Variable b2(zeros({1, latent_dim}, DType::Float32, device), true);

    Variable W3(randn({latent_dim, hidden_dim}, DType::Float32, device)
              * std::sqrt(2.0f / latent_dim), true);
    Variable b3(zeros({1, hidden_dim}, DType::Float32, device), true);
    Variable W4(randn({hidden_dim, input_dim}, DType::Float32, device)
              * std::sqrt(1.0f / hidden_dim), true);
    Variable b4(zeros({1, input_dim}, DType::Float32, device), true);

    // Training parameters
    float learning_rate = 0.05f;
    int num_epochs = 1000;
    int print_every = 100;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable X(X_tensor, false);

        // Encoder: Linear -> ReLU -> Linear (linear latent)
        auto h1 = nn::relu(matmul(X, W1) + b1);
        auto latent = matmul(h1, W2) + b2;

        // Decoder: Linear -> ReLU -> Linear -> Sigmoid
        auto h2 = nn::relu(matmul(latent, W3) + b3);
        auto reconstruction = nn::sigmoid(matmul(h2, W4) + b4);

        // ============ Compute Loss (MSE) ============
        auto error = reconstruction - X;
        auto loss = mean(error * error);

        // ============ Backward Pass (Automatic!) ============
        W1.zero_grad(); b1.zero_grad();
        W2.zero_grad(); b2.zero_grad();
        W3.zero_grad(); b3.zero_grad();
        W4.zero_grad(); b4.zero_grad();

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
            W4 = Variable(W4.tensor() - (*W4.grad() * learning_rate), true);
            b4 = Variable(b4.tensor() - (*b4.grad() * learning_rate), true);
        }

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                      << "Reconstruction Loss: " << loss_val << "\n";
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    Variable X_final(X_tensor, false);
    auto h1 = nn::relu(matmul(X_final, W1) + b1);
    auto latent = matmul(h1, W2) + b2;
    auto h2 = nn::relu(matmul(latent, W3) + b3);
    auto reconstruction = nn::sigmoid(matmul(h2, W4) + b4);

    auto error = reconstruction - X_final;
    float final_loss = mean(error * error).tensor().item<float>();

    std::cout << "Final Reconstruction Loss: " << final_loss << "\n\n";

    std::cout << "Autoencoder demonstrated with autograd!\n";
    std::cout << "Gradients computed automatically through entire network.\n";

    finalize();
    return 0;
}
