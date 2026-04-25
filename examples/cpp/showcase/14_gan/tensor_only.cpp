/**
 * @file tensor_only.cpp
 * @brief Generative Adversarial Network using raw Tensor operations
 *
 * A tiny 1-D GAN: generator learns to produce samples from a target
 * Gaussian N(3, 0.5) given N(0, 1) noise, while a discriminator learns
 * to distinguish real from generated samples. Adversarial training
 * alternates between the two - both use manual forward/backward.
 *
 * Usage: ./14_gan_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

static Tensor relu_t(const Tensor& x)   { return maximum(x, zeros_like(x)); }
static Tensor relu_d(const Tensor& z)   { return (z > zeros_like(z)).to(DType::Float32); }
static Tensor sigmoid_t(const Tensor& x){ return ones_like(x) / (tenzor::exp(x * -1.0f) + 1.0f); }

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("GAN - Tensor Only (Manual Backprop)", device);
    manual_seed(42);

    int batch_size = 64;
    int noise_dim  = 4;
    int hidden     = 16;

    // Target distribution: N(3, 0.5)
    auto sample_real = [&](int n) {
        auto z = randn({n, 1}, DType::Float32, device);
        return z * 0.5f + 3.0f;
    };

    // Generator: noise_dim -> hidden -> 1
    auto G_W1 = randn({noise_dim, hidden}, DType::Float32, device) * std::sqrt(2.0f / noise_dim);
    auto G_b1 = zeros({1, hidden}, DType::Float32, device);
    auto G_W2 = randn({hidden, 1},      DType::Float32, device) * std::sqrt(1.0f / hidden);
    auto G_b2 = zeros({1, 1},           DType::Float32, device);

    // Discriminator: 1 -> hidden -> 1
    auto D_W1 = randn({1, hidden}, DType::Float32, device) * std::sqrt(2.0f / 1);
    auto D_b1 = zeros({1, hidden}, DType::Float32, device);
    auto D_W2 = randn({hidden, 1}, DType::Float32, device) * std::sqrt(1.0f / hidden);
    auto D_b2 = zeros({1, 1},      DType::Float32, device);

    float lr = 0.02f;
    int num_epochs = 3000;
    int print_every = 300;

    showcase::print_section("Training");

    auto forward_G = [&](const Tensor& z) {
        auto h_pre = matmul(z, G_W1) + G_b1;
        auto h = relu_t(h_pre);
        auto out = matmul(h, G_W2) + G_b2;
        return std::make_tuple(h_pre, h, out);
    };
    auto forward_D = [&](const Tensor& x) {
        auto h_pre = matmul(x, D_W1) + D_b1;
        auto h = relu_t(h_pre);
        auto logit = matmul(h, D_W2) + D_b2;
        auto prob = sigmoid_t(logit);
        return std::make_tuple(h_pre, h, logit, prob);
    };

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ---------- Train Discriminator ----------
        // D wants to output 1 for real, 0 for fake (BCE)
        auto real = sample_real(batch_size);
        auto noise = randn({batch_size, noise_dim}, DType::Float32, device);
        auto [G_h1_pre, G_h1, fake] = forward_G(noise);

        auto [D_h1r_pre, D_h1r, D_logit_r, D_prob_r] = forward_D(real);
        auto [D_h1f_pre, D_h1f, D_logit_f, D_prob_f] = forward_D(fake);

        // BCE loss = -mean(log(D(real)) + log(1 - D(fake)))
        float eps = 1e-8f;
        float d_loss = -(tenzor::mean(tenzor::log(D_prob_r + eps)).item<float>()
                       + tenzor::mean(tenzor::log(ones_like(D_prob_f) - D_prob_f + eps)).item<float>());

        float n = static_cast<float>(batch_size);
        // d logit for real: (prob_r - 1) / n   (from BCE gradient w.r.t. logit)
        auto dLd_logit_r = (D_prob_r - 1.0f) * (1.0f / n);
        auto dLd_logit_f = (D_prob_f)        * (1.0f / n);

        // Backprop D for both batches (accumulate gradients)
        auto dL_dD_W2 = matmul(D_h1r.transpose(0, 1), dLd_logit_r)
                      + matmul(D_h1f.transpose(0, 1), dLd_logit_f);
        auto dL_dD_b2 = tenzor::sum(dLd_logit_r, 0, true) + tenzor::sum(dLd_logit_f, 0, true);

        auto dL_dD_h1r = matmul(dLd_logit_r, D_W2.transpose(0, 1)) * relu_d(D_h1r_pre);
        auto dL_dD_h1f = matmul(dLd_logit_f, D_W2.transpose(0, 1)) * relu_d(D_h1f_pre);

        auto dL_dD_W1 = matmul(real.transpose(0, 1), dL_dD_h1r)
                      + matmul(fake.transpose(0, 1), dL_dD_h1f);
        auto dL_dD_b1 = tenzor::sum(dL_dD_h1r, 0, true) + tenzor::sum(dL_dD_h1f, 0, true);

        D_W1 = D_W1 - dL_dD_W1 * lr;   D_b1 = D_b1 - dL_dD_b1 * lr;
        D_W2 = D_W2 - dL_dD_W2 * lr;   D_b2 = D_b2 - dL_dD_b2 * lr;

        // ---------- Train Generator ----------
        // G wants D(G(z)) ~ 1 -> non-saturating loss -log(D(G(z)))
        auto noise2 = randn({batch_size, noise_dim}, DType::Float32, device);
        auto [G_h2_pre, G_h2, fake2] = forward_G(noise2);
        auto [Df_h1_pre, Df_h1, Df_logit, Df_prob] = forward_D(fake2);

        float g_loss = -tenzor::mean(tenzor::log(Df_prob + eps)).item<float>();

        // Gradient of -log(D(fake)) w.r.t. logit = -(1 - D) / n = (D - 1) / n   ... same as "fake was real"
        auto dGd_logit = (Df_prob - 1.0f) * (1.0f / n);

        // Backprop through D (fixed) to get grad on fake samples (no param update for D here)
        auto dG_dDh1 = matmul(dGd_logit, D_W2.transpose(0, 1)) * relu_d(Df_h1_pre);
        auto dG_dfake = matmul(dG_dDh1, D_W1.transpose(0, 1));

        // Backprop through G
        auto dG_dG_W2 = matmul(G_h2.transpose(0, 1), dG_dfake);
        auto dG_dG_b2 = tenzor::sum(dG_dfake, 0, true);
        auto dG_dG_h1 = matmul(dG_dfake, G_W2.transpose(0, 1)) * relu_d(G_h2_pre);
        auto dG_dG_W1 = matmul(noise2.transpose(0, 1), dG_dG_h1);
        auto dG_dG_b1 = tenzor::sum(dG_dG_h1, 0, true);

        G_W1 = G_W1 - dG_dG_W1 * lr;   G_b1 = G_b1 - dG_dG_b1 * lr;
        G_W2 = G_W2 - dG_dG_W2 * lr;   G_b2 = G_b2 - dG_dG_b2 * lr;

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] D_loss: " << d_loss << "  G_loss: " << g_loss << "\n";
        }
    }

    // ---------- Evaluation ----------
    showcase::print_section("Final Results");
    auto noise_eval = randn({256, noise_dim}, DType::Float32, device);
    auto [_1, _2, fake_eval] = forward_G(noise_eval);
    auto real_eval = sample_real(256);

    float real_mean = tenzor::mean(real_eval).item<float>();
    float fake_mean = tenzor::mean(fake_eval).item<float>();
    auto real_ctr = real_eval - real_mean;
    auto fake_ctr = fake_eval - fake_mean;
    float real_std = std::sqrt(tenzor::mean(real_ctr * real_ctr).item<float>());
    float fake_std = std::sqrt(tenzor::mean(fake_ctr * fake_ctr).item<float>());

    std::cout << "Target distribution:     N(3.0, 0.5)\n";
    std::cout << "Real sample stats:       mean=" << real_mean << ", std=" << real_std << "\n";
    std::cout << "Generator sample stats:  mean=" << fake_mean << ", std=" << fake_std << "\n";
    std::cout << "\nGAN demonstrated with raw tensors!\n";
    std::cout << "Alternating D and G updates let the generator match the target N(3, 0.5).\n";

    finalize();
    return 0;
}
