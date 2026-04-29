/**
 * @file autograd_runner.cpp
 * @brief Implementation of the GAN autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <cmath>
#include <vector>

namespace tenzor::examples::showcase14 {

int run_gan_training(int epochs,
                     double* out_initial,
                     double* out_final,
                     ::tenzor::Device device,
                     bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    int batch_size = 64;
    int noise_dim  = 4;
    int hidden     = 16;

    auto sample_real = [&](int n) {
        auto z = randn({n, 1}, DType::Float32, device);
        return z * 0.5f + 3.0f;
    };

    auto he = [&](int64_t fin) { return std::sqrt(2.0f / fin); };
    auto xav= [&](int64_t fin) { return std::sqrt(1.0f / fin); };

    Variable G_W1(randn({noise_dim, hidden}, DType::Float32, device) * he(noise_dim), true);
    Variable G_b1(zeros({1, hidden}, DType::Float32, device), true);
    Variable G_W2(randn({hidden, 1}, DType::Float32, device) * xav(hidden), true);
    Variable G_b2(zeros({1, 1}, DType::Float32, device), true);
    std::vector<Variable*> G_params = {&G_W1, &G_b1, &G_W2, &G_b2};

    Variable D_W1(randn({1, hidden}, DType::Float32, device) * he(1), true);
    Variable D_b1(zeros({1, hidden}, DType::Float32, device), true);
    Variable D_W2(randn({hidden, 1}, DType::Float32, device) * xav(hidden), true);
    Variable D_b2(zeros({1, 1}, DType::Float32, device), true);
    std::vector<Variable*> D_params = {&D_W1, &D_b1, &D_W2, &D_b2};

    auto G_forward = [&](const Variable& z) {
        return matmul(nn::relu(matmul(z, G_W1) + G_b1), G_W2) + G_b2;
    };
    auto D_forward = [&](const Variable& x) {
        return matmul(nn::relu(matmul(x, D_W1) + D_b1), D_W2) + D_b2;
    };

    auto bce_with_logits = [&](const Variable& logit, float target) {
        if (target > 0.5f) {
            return mean(::tenzor::log(::tenzor::exp(logit * -1.0f) + 1.0f));
        } else {
            return mean(::tenzor::log(::tenzor::exp(logit) + 1.0f));
        }
    };

    auto zero_grads = [](std::vector<Variable*>& v) {
        for (auto* p : v) p->zero_grad();
    };
    float lr = 0.02f;
    auto sgd_step = [&](std::vector<Variable*>& v) {
        NoGradGuard ng;
        for (auto* p : v) {
            *p = Variable(p->tensor() - (*p->grad() * lr), true);
        }
    };

    int print_every = std::max(1, epochs / 10);

    double final_d_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // D step
        Variable real(sample_real(batch_size), false);
        Variable noise(randn({batch_size, noise_dim}, DType::Float32, device), false);
        auto fake = G_forward(noise);
        Variable fake_det(fake.tensor(), false);

        auto d_real = D_forward(real);
        auto d_fake = D_forward(fake_det);
        auto d_loss = bce_with_logits(d_real, 1.0f) + bce_with_logits(d_fake, 0.0f);

        zero_grads(D_params);
        d_loss.backward();
        sgd_step(D_params);

        // G step
        Variable noise2(randn({batch_size, noise_dim}, DType::Float32, device), false);
        auto fake2 = G_forward(noise2);
        auto d_fake2 = D_forward(fake2);
        auto g_loss = bce_with_logits(d_fake2, 1.0f);

        zero_grads(G_params);
        g_loss.backward();
        sgd_step(G_params);

        double d_loss_val = static_cast<double>(d_loss.tensor().item<float>());
        if (epoch == 0 && out_initial) *out_initial = d_loss_val;
        final_d_loss = d_loss_val;

        if (verbose && ((epoch + 1) % print_every == 0 || epoch == 0)) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << epochs
                      << "] D_loss=" << d_loss_val
                      << "  G_loss=" << g_loss.tensor().item<float>() << "\n";
        }
    }
    if (out_final) *out_final = final_d_loss;
    return 0;
}

}  // namespace tenzor::examples::showcase14
