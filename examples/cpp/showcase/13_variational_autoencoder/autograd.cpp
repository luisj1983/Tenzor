/**
 * @file autograd.cpp
 * @brief Variational Autoencoder (VAE) using Tenzor's autograd
 *
 * VAEs are the canonical showcase of the reparameterization trick:
 * sampling z = mu + sigma * eps lets gradient flow from the decoder
 * back through z into (mu, log_var) and the encoder parameters.
 *
 * With autograd, the whole loss - reconstruction + KL divergence -
 * composes in one expression and backward() just works.
 *
 * Usage: ./13_variational_autoencoder_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Variational Autoencoder - Autograd", device);

    manual_seed(42);

    int batch_size = 32;
    int input_dim  = 16;
    int hidden_dim = 16;
    int latent_dim = 2;

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
              << " -> (mu, log_var) @ " << latent_dim
              << " -> " << hidden_dim << " -> " << input_dim << "\n";

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
    int num_epochs = 1500;
    int print_every = 150;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable X(X_tensor, false);

        // Encoder
        auto h = nn::relu(matmul(X, W1) + b1);
        auto mu     = matmul(h, Wmu) + bmu;
        auto logvar = matmul(h, Wlv) + blv;

        // Reparameterization: z = mu + exp(0.5 * logvar) * eps
        auto std_dev = tenzor::exp(logvar * 0.5f);
        auto eps = Variable(randn({batch_size, latent_dim}, DType::Float32, device), false);
        auto z = mu + std_dev * eps;

        // Decoder
        auto x_hat = nn::sigmoid(matmul(nn::relu(matmul(z, W3) + b3), W4) + b4);

        // Loss
        auto err = x_hat - X;
        auto recon_loss = mean(err * err);

        // KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / B
        auto kl_per = (logvar + 1.0f) - mu * mu - tenzor::exp(logvar);
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

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] Total: " << loss.tensor().item<float>()
                      << "  Recon: " << recon_loss.tensor().item<float>()
                      << "  KL: "    << kl_loss.tensor().item<float>() << "\n";
        }
    }

    showcase::print_section("Final Results");

    // Deterministic recon
    Variable X(X_tensor, false);
    auto h = nn::relu(matmul(X, W1) + b1);
    auto mu = matmul(h, Wmu) + bmu;
    auto x_hat = nn::sigmoid(matmul(nn::relu(matmul(mu, W3) + b3), W4) + b4);
    auto err = x_hat.tensor() - X_tensor;
    std::cout << "Final deterministic recon MSE (z = mu): "
              << tenzor::mean(err * err).item<float>() << "\n\n";

    // Draw from prior
    std::cout << "Generating 3 new samples from z ~ N(0, I)...\n";
    Variable z_sample(randn({3, latent_dim}, DType::Float32, device), false);
    auto gen = nn::sigmoid(matmul(nn::relu(matmul(z_sample, W3) + b3), W4) + b4);

    auto z_cpu = z_sample.tensor().cpu();
    auto gen_cpu = gen.tensor().cpu();
    for (int i = 0; i < 3; ++i) {
        std::cout << "  z = [";
        for (int j = 0; j < latent_dim; ++j) {
            std::cout << z_cpu.data<float>()[i * latent_dim + j];
            if (j < latent_dim - 1) std::cout << ", ";
        }
        std::cout << "] -> generated[0..3] = [";
        for (int j = 0; j < 4; ++j) {
            std::cout << gen_cpu.data<float>()[i * input_dim + j];
            if (j < 3) std::cout << ", ";
        }
        std::cout << ", ...]\n";
    }

    std::cout << "\nVAE demonstrated with autograd!\n";
    std::cout << "Autograd traces gradient through the reparameterized sample automatically.\n";

    finalize();
    return 0;
}
