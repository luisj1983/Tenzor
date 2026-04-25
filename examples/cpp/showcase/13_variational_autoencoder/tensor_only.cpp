/**
 * @file tensor_only.cpp
 * @brief Variational Autoencoder (VAE) using raw Tensor operations only
 *
 * A VAE maps x -> (mu, log_var), samples z = mu + exp(0.5 * log_var) * eps,
 * and decodes z back to x. Loss has two terms:
 *   - reconstruction: MSE(x_hat, x)
 *   - KL divergence:  -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
 *
 * This version is hand-rolled: manual forward, manual sampling, manual
 * backward through the reparameterization trick (the only reason VAEs
 * are trainable end-to-end).
 *
 * Usage: ./13_variational_autoencoder_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

static Tensor relu_tensor(const Tensor& x) {
    return maximum(x, zeros_like(x));
}

static Tensor relu_deriv(const Tensor& z) {
    return (z > zeros_like(z)).to(DType::Float32);
}

static Tensor sigmoid_tensor(const Tensor& x) {
    auto neg_x = x * -1.0f;
    return ones_like(x) / (tenzor::exp(neg_x) + 1.0f);
}

static Tensor sigmoid_deriv(const Tensor& sig_out) {
    return sig_out * (ones_like(sig_out) - sig_out);
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Variational Autoencoder - Tensor Only (Manual Backprop)", device);

    manual_seed(42);

    // Same structured data as the plain autoencoder so results are comparable.
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
    auto X = from_data(X_data.data(), {batch_size, input_dim}, device);

    showcase::print_tensor_info("Input X", X);
    std::cout << "Architecture: " << input_dim << " -> " << hidden_dim
              << " -> (mu, log_var) @ " << latent_dim
              << " -> " << hidden_dim << " -> " << input_dim << "\n";

    // Encoder: input -> hidden -> (mu, log_var)
    auto W1  = randn({input_dim, hidden_dim}, DType::Float32, device) * std::sqrt(2.0f / input_dim);
    auto b1  = zeros({1, hidden_dim},          DType::Float32, device);
    auto Wmu = randn({hidden_dim, latent_dim}, DType::Float32, device) * std::sqrt(2.0f / hidden_dim);
    auto bmu = zeros({1, latent_dim},          DType::Float32, device);
    auto Wlv = randn({hidden_dim, latent_dim}, DType::Float32, device) * std::sqrt(2.0f / hidden_dim);
    auto blv = zeros({1, latent_dim},          DType::Float32, device);

    // Decoder: latent -> hidden -> input
    auto W3 = randn({latent_dim, hidden_dim}, DType::Float32, device) * std::sqrt(2.0f / latent_dim);
    auto b3 = zeros({1, hidden_dim},           DType::Float32, device);
    auto W4 = randn({hidden_dim, input_dim},   DType::Float32, device) * std::sqrt(1.0f / hidden_dim);
    auto b4 = zeros({1, input_dim},            DType::Float32, device);

    float lr = 0.03f;
    float kl_weight = 0.001f;  // small weight - data is simple, avoid KL swamping recon
    int num_epochs = 1500;
    int print_every = 150;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ---- Encoder ----
        auto z1 = matmul(X, W1) + b1;
        auto a1 = relu_tensor(z1);
        auto mu     = matmul(a1, Wmu) + bmu;
        auto logvar = matmul(a1, Wlv) + blv;

        // ---- Reparameterization: z = mu + exp(0.5 * logvar) * eps ----
        auto std_dev = tenzor::exp(logvar * 0.5f);
        auto eps = randn({batch_size, latent_dim}, DType::Float32, device);
        auto z = mu + std_dev * eps;

        // ---- Decoder ----
        auto z3 = matmul(z, W3) + b3;
        auto a3 = relu_tensor(z3);
        auto z4 = matmul(a3, W4) + b4;
        auto x_hat = sigmoid_tensor(z4);

        // ---- Loss: recon (MSE) + KL ----
        auto recon_err = x_hat - X;
        float recon_loss = tenzor::mean(recon_err * recon_err).item<float>();

        // KL( N(mu, sigma^2) || N(0, 1) ) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        auto kl_per_elem = (logvar + 1.0f) - mu * mu - tenzor::exp(logvar);  // per-element
        float kl_loss = tenzor::sum(kl_per_elem).item<float>() * (-0.5f)
                       / static_cast<float>(batch_size);

        float loss_val = recon_loss + kl_weight * kl_loss;

        // ---- Backward ----
        float n = static_cast<float>(batch_size * input_dim);
        auto dL_dxhat = recon_err * (2.0f / n);

        auto dL_dz4 = dL_dxhat * sigmoid_deriv(x_hat);
        auto dL_dW4 = matmul(a3.transpose(0, 1), dL_dz4);
        auto dL_db4 = tenzor::sum(dL_dz4, 0, true);

        auto dL_da3 = matmul(dL_dz4, W4.transpose(0, 1));
        auto dL_dz3 = dL_da3 * relu_deriv(z3);
        auto dL_dW3 = matmul(z.transpose(0, 1), dL_dz3);
        auto dL_db3 = tenzor::sum(dL_dz3, 0, true);

        // dL/dz flows to both mu and logvar via the reparameterization
        auto dL_dz = matmul(dL_dz3, W3.transpose(0, 1));

        // z = mu + exp(0.5*logvar) * eps
        //   => dz/dmu = 1
        //   => dz/dlogvar = 0.5 * exp(0.5*logvar) * eps = 0.5 * std_dev * eps
        auto dRecon_dmu     = dL_dz;
        auto dRecon_dlogvar = dL_dz * (std_dev * eps * 0.5f);

        // KL contributes its own gradients w.r.t. mu, logvar
        // KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / B
        // dKL/dmu     =  mu / B
        // dKL/dlogvar = -0.5 * (1 - exp(logvar)) / B  = 0.5 * (exp(logvar) - 1) / B
        float inv_B = 1.0f / static_cast<float>(batch_size);
        auto dKL_dmu     = mu * inv_B;
        auto dKL_dlogvar = (tenzor::exp(logvar) - 1.0f) * (0.5f * inv_B);

        auto dL_dmu     = dRecon_dmu     + dKL_dmu     * kl_weight;
        auto dL_dlogvar = dRecon_dlogvar + dKL_dlogvar * kl_weight;

        auto dL_dWmu = matmul(a1.transpose(0, 1), dL_dmu);
        auto dL_dbmu = tenzor::sum(dL_dmu, 0, true);
        auto dL_dWlv = matmul(a1.transpose(0, 1), dL_dlogvar);
        auto dL_dblv = tenzor::sum(dL_dlogvar, 0, true);

        auto dL_da1 = matmul(dL_dmu, Wmu.transpose(0, 1))
                    + matmul(dL_dlogvar, Wlv.transpose(0, 1));
        auto dL_dz1 = dL_da1 * relu_deriv(z1);
        auto dL_dW1 = matmul(X.transpose(0, 1), dL_dz1);
        auto dL_db1 = tenzor::sum(dL_dz1, 0, true);

        // ---- Update ----
        W4  = W4  - dL_dW4  * lr;  b4  = b4  - dL_db4  * lr;
        W3  = W3  - dL_dW3  * lr;  b3  = b3  - dL_db3  * lr;
        Wmu = Wmu - dL_dWmu * lr;  bmu = bmu - dL_dbmu * lr;
        Wlv = Wlv - dL_dWlv * lr;  blv = blv - dL_dblv * lr;
        W1  = W1  - dL_dW1  * lr;  b1  = b1  - dL_db1  * lr;

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] Total: " << loss_val
                      << "  Recon: " << recon_loss
                      << "  KL: " << kl_loss << "\n";
        }
    }

    showcase::print_section("Final Results");

    // Deterministic pass (use mu, not a sample, for reconstruction)
    auto a1 = relu_tensor(matmul(X, W1) + b1);
    auto mu = matmul(a1, Wmu) + bmu;
    auto x_hat = sigmoid_tensor(matmul(relu_tensor(matmul(mu, W3) + b3), W4) + b4);
    auto recon_err = x_hat - X;
    std::cout << "Final deterministic recon MSE (z = mu): "
              << tenzor::mean(recon_err * recon_err).item<float>() << "\n\n";

    // Generate new samples by drawing z ~ N(0, I)
    std::cout << "Generating 3 new samples from z ~ N(0, I)...\n";
    auto z_sample = randn({3, latent_dim}, DType::Float32, device);
    auto gen = sigmoid_tensor(matmul(relu_tensor(matmul(z_sample, W3) + b3), W4) + b4);
    auto z_cpu = z_sample.cpu();
    auto gen_cpu = gen.cpu();
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

    std::cout << "\nVAE demonstrated with raw tensors!\n";
    std::cout << "Reparameterization trick lets gradient flow through the random sample.\n";
    std::cout << "KL term regularizes the latent towards N(0, I), enabling generation.\n";

    finalize();
    return 0;
}
