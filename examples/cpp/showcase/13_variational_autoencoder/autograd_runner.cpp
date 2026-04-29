/**
 * @file autograd_runner.cpp
 * @brief Implementation of the VAE autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace tenzor::examples::showcase13 {

int run_vae_training(int epochs,
                     double* out_initial,
                     double* out_final,
                     ::tenzor::Device device,
                     bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    int batch_size = 32;
    int input_dim  = 16;
    int hidden_dim = 16;
    int latent_dim = 2;

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

    auto he  = [&](int64_t fin) { return std::sqrt(2.0f / fin); };
    auto xav = [&](int64_t fin) { return std::sqrt(1.0f / fin); };

    Variable W1 (randn({input_dim, hidden_dim},  DType::Float32, device) * he(input_dim),  true);
    Variable b1 (zeros({1, hidden_dim},          DType::Float32, device), true);
    Variable Wmu(randn({hidden_dim, latent_dim}, DType::Float32, device) * he(hidden_dim), true);
    Variable bmu(zeros({1, latent_dim},          DType::Float32, device), true);
    Variable Wlv(randn({hidden_dim, latent_dim}, DType::Float32, device) * he(hidden_dim), true);
    Variable blv(zeros({1, latent_dim},          DType::Float32, device), true);
    Variable W3 (randn({latent_dim, hidden_dim}, DType::Float32, device) * he(latent_dim), true);
    Variable b3 (zeros({1, hidden_dim},          DType::Float32, device), true);
    Variable W4 (randn({hidden_dim, input_dim},  DType::Float32, device) * xav(hidden_dim),true);
    Variable b4 (zeros({1, input_dim},           DType::Float32, device), true);

    float lr = 0.03f;
    float kl_weight = 0.001f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable X(X_tensor, false);

        auto h = nn::relu(matmul(X, W1) + b1);
        auto mu     = matmul(h, Wmu) + bmu;
        auto logvar = matmul(h, Wlv) + blv;

        auto std_dev = ::tenzor::exp(logvar * 0.5f);
        auto eps = Variable(randn({batch_size, latent_dim}, DType::Float32, device), false);
        auto z = mu + std_dev * eps;

        auto x_hat = nn::sigmoid(matmul(nn::relu(matmul(z, W3) + b3), W4) + b4);

        auto err = x_hat - X;
        auto recon_loss = mean(err * err);

        auto kl_per = (logvar + 1.0f) - mu * mu - ::tenzor::exp(logvar);
        auto kl_loss = sum(kl_per) * (-0.5f / static_cast<float>(batch_size));

        auto loss = recon_loss + kl_loss * kl_weight;

        W1.zero_grad();  b1.zero_grad();
        Wmu.zero_grad(); bmu.zero_grad();
        Wlv.zero_grad(); blv.zero_grad();
        W3.zero_grad();  b3.zero_grad();
        W4.zero_grad();  b4.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W1  = Variable(W1.tensor()  - (*W1.grad()  * lr), true);
            b1  = Variable(b1.tensor()  - (*b1.grad()  * lr), true);
            Wmu = Variable(Wmu.tensor() - (*Wmu.grad() * lr), true);
            bmu = Variable(bmu.tensor() - (*bmu.grad() * lr), true);
            Wlv = Variable(Wlv.tensor() - (*Wlv.grad() * lr), true);
            blv = Variable(blv.tensor() - (*blv.grad() * lr), true);
            W3  = Variable(W3.tensor()  - (*W3.grad()  * lr), true);
            b3  = Variable(b3.tensor()  - (*b3.grad()  * lr), true);
            W4  = Variable(W4.tensor()  - (*W4.grad()  * lr), true);
            b4  = Variable(b4.tensor()  - (*b4.grad()  * lr), true);
        }

        double loss_val = static_cast<double>(loss.tensor().item<float>());
        if (epoch == 0 && out_initial) *out_initial = loss_val;
        final_loss = loss_val;

        if (verbose && ((epoch + 1) % print_every == 0 || epoch == 0)) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << epochs
                      << "] total=" << loss_val << "\n";
        }
    }
    if (out_final) *out_final = final_loss;
    return 0;
}

}  // namespace tenzor::examples::showcase13
