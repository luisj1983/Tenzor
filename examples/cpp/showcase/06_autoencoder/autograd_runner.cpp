/**
 * @file autograd_runner.cpp
 * @brief Implementation of the autoencoder-autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace tenzor::examples::showcase06 {

int run_autoencoder_training(int epochs,
                             double* out_initial,
                             double* out_final,
                             ::tenzor::Device device,
                             bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    int batch_size = 32;
    int input_dim = 16;
    int hidden_dim = 12;
    int latent_dim = 4;

    // Use a local RNG so we don't depend on global rand() state.
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> X_data(batch_size * input_dim);
    for (int b = 0; b < batch_size; ++b) {
        float c[4];
        for (int k = 0; k < 4; ++k) c[k] = dist(gen);
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

    float learning_rate = 0.05f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable X(X_tensor, false);

        auto h1 = nn::relu(matmul(X, W1) + b1);
        auto latent = matmul(h1, W2) + b2;
        auto h2 = nn::relu(matmul(latent, W3) + b3);
        auto reconstruction = nn::sigmoid(matmul(h2, W4) + b4);

        auto error = reconstruction - X;
        auto loss = mean(error * error);

        W1.zero_grad(); b1.zero_grad();
        W2.zero_grad(); b2.zero_grad();
        W3.zero_grad(); b3.zero_grad();
        W4.zero_grad(); b4.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W1 = Variable(W1.tensor() - (*W1.grad() * learning_rate), true);
            b1 = Variable(b1.tensor() - (*b1.grad() * learning_rate), true);
            W2 = Variable(W2.tensor() - (*W2.grad() * learning_rate), true);
            b2 = Variable(b2.tensor() - (*b2.grad() * learning_rate), true);
            W3 = Variable(W3.tensor() - (*W3.grad() * learning_rate), true);
            b3 = Variable(b3.tensor() - (*b3.grad() * learning_rate), true);
            W4 = Variable(W4.tensor() - (*W4.grad() * learning_rate), true);
            b4 = Variable(b4.tensor() - (*b4.grad() * learning_rate), true);
        }

        double loss_val = static_cast<double>(loss.tensor().item<float>());
        if (epoch == 0 && out_initial) *out_initial = loss_val;
        final_loss = loss_val;

        if (verbose && ((epoch + 1) % print_every == 0 || epoch == 0)) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << epochs
                      << "] loss=" << loss_val << "\n";
        }
    }
    if (out_final) *out_final = final_loss;
    return 0;
}

}  // namespace tenzor::examples::showcase06
