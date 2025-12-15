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

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Autoencoder - Autograd (Automatic Differentiation)", device);

    manual_seed(42);

    // Generate synthetic data
    int batch_size = 32;
    int input_dim = 16;
    int hidden_dim = 8;
    int latent_dim = 4;

    auto X_tensor = rand({batch_size, input_dim}, DType::Float32, device);

    showcase::print_tensor_info("Input X", X_tensor);
    std::cout << "Architecture: " << input_dim << " -> " << hidden_dim
              << " -> " << latent_dim << " -> " << hidden_dim << " -> " << input_dim << "\n";

    // Initialize weights as Variables with gradient tracking
    // Encoder
    Variable W1(randn({input_dim, hidden_dim}, DType::Float32, device) * 0.1f, true);
    Variable b1(zeros({1, hidden_dim}, DType::Float32, device), true);
    Variable W2(randn({hidden_dim, latent_dim}, DType::Float32, device) * 0.1f, true);
    Variable b2(zeros({1, latent_dim}, DType::Float32, device), true);

    // Decoder
    Variable W3(randn({latent_dim, hidden_dim}, DType::Float32, device) * 0.1f, true);
    Variable b3(zeros({1, hidden_dim}, DType::Float32, device), true);
    Variable W4(randn({hidden_dim, input_dim}, DType::Float32, device) * 0.1f, true);
    Variable b4(zeros({1, input_dim}, DType::Float32, device), true);

    // Training parameters
    float learning_rate = 0.5f;
    int num_epochs = 1000;
    int print_every = 100;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable X(X_tensor, false);

        // ============ Forward Pass (Encoder) ============
        auto h1 = nn::sigmoid(matmul(X, W1) + b1);     // Hidden
        auto latent = nn::sigmoid(matmul(h1, W2) + b2); // Latent

        // ============ Forward Pass (Decoder) ============
        auto h2 = nn::sigmoid(matmul(latent, W3) + b3); // Hidden
        auto reconstruction = nn::sigmoid(matmul(h2, W4) + b4); // Output

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
    auto h1 = nn::sigmoid(matmul(X_final, W1) + b1);
    auto latent = nn::sigmoid(matmul(h1, W2) + b2);
    auto h2 = nn::sigmoid(matmul(latent, W3) + b3);
    auto reconstruction = nn::sigmoid(matmul(h2, W4) + b4);

    auto error = reconstruction - X_final;
    float final_loss = mean(error * error).tensor().item<float>();

    std::cout << "Final Reconstruction Loss: " << final_loss << "\n\n";

    std::cout << "Autoencoder demonstrated with autograd!\n";
    std::cout << "Gradients computed automatically through entire network.\n";

    finalize();
    return 0;
}
