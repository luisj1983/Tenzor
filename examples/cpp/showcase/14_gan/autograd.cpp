/**
 * @file autograd.cpp
 * @brief Generative Adversarial Network using Tenzor's autograd
 *
 * Alternating updates to a generator (learns to fake N(3, 0.5) samples from
 * N(0, 1) noise) and a discriminator (learns to tell real from fake).
 * Autograd removes every manual chain-rule step from the tensor_only version.
 *
 * Usage: ./14_gan_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;

static Variable bce_logit(const Variable& logit, float target_value) {
    // sum_reduction = -target * log(sigmoid(logit)) - (1 - target) * log(1 - sigmoid(logit))
    // with the numerically stable log_sigmoid form:
    //   log(sigmoid(x))      = -softplus(-x)
    //   log(1 - sigmoid(x))  = -softplus(x)
    // softplus(y) = log(1 + exp(y))
    auto neg = logit * -1.0f;
    auto sp_neg = tenzor::log(tenzor::exp(neg) + 1.0f);      // softplus(-logit)
    auto sp_pos = tenzor::log(tenzor::exp(logit) + 1.0f);    // softplus( logit)
    if (target_value > 0.5f) {
        return mean(Variable(sp_neg.tensor(), false));
    } else {
        return mean(Variable(sp_pos.tensor(), false));
    }
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("GAN - Autograd", device);
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

    // Generator params
    Variable G_W1(randn({noise_dim, hidden}, DType::Float32, device) * he(noise_dim), true);
    Variable G_b1(zeros({1, hidden}, DType::Float32, device), true);
    Variable G_W2(randn({hidden, 1}, DType::Float32, device) * xav(hidden), true);
    Variable G_b2(zeros({1, 1}, DType::Float32, device), true);
    std::vector<Variable*> G_params = {&G_W1, &G_b1, &G_W2, &G_b2};

    // Discriminator params
    Variable D_W1(randn({1, hidden}, DType::Float32, device) * he(1), true);
    Variable D_b1(zeros({1, hidden}, DType::Float32, device), true);
    Variable D_W2(randn({hidden, 1}, DType::Float32, device) * xav(hidden), true);
    Variable D_b2(zeros({1, 1}, DType::Float32, device), true);
    std::vector<Variable*> D_params = {&D_W1, &D_b1, &D_W2, &D_b2};

    auto G_forward = [&](const Variable& z) {
        return matmul(nn::relu(matmul(z, G_W1) + G_b1), G_W2) + G_b2;
    };
    auto D_forward = [&](const Variable& x) {
        return matmul(nn::relu(matmul(x, D_W1) + D_b1), D_W2) + D_b2;  // logit
    };

    // Numerically stable BCE via softplus. target in {0,1}.
    auto bce_with_logits = [&](const Variable& logit, float target) {
        if (target > 0.5f) {
            // -log sigmoid(logit) = softplus(-logit)
            return mean(tenzor::log(tenzor::exp(logit * -1.0f) + 1.0f));
        } else {
            // -log(1 - sigmoid(logit)) = softplus(logit)
            return mean(tenzor::log(tenzor::exp(logit) + 1.0f));
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

    int num_epochs = 3000;
    int print_every = 300;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ---- D step ----
        Variable real(sample_real(batch_size), false);
        Variable noise(randn({batch_size, noise_dim}, DType::Float32, device), false);
        auto fake = G_forward(noise);
        // Detach fake from G's graph so D-step doesn't update G
        Variable fake_det(fake.tensor(), false);

        auto d_real = D_forward(real);
        auto d_fake = D_forward(fake_det);
        auto d_loss = bce_with_logits(d_real, 1.0f) + bce_with_logits(d_fake, 0.0f);

        zero_grads(D_params);
        d_loss.backward();
        sgd_step(D_params);

        // ---- G step ----
        Variable noise2(randn({batch_size, noise_dim}, DType::Float32, device), false);
        auto fake2 = G_forward(noise2);
        auto d_fake2 = D_forward(fake2);
        // Non-saturating G loss: maximize log D(fake) -> minimize -log D(fake)
        auto g_loss = bce_with_logits(d_fake2, 1.0f);

        zero_grads(G_params);
        g_loss.backward();
        sgd_step(G_params);

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] D_loss: " << d_loss.tensor().item<float>()
                      << "  G_loss: " << g_loss.tensor().item<float>() << "\n";
        }
    }

    showcase::print_section("Final Results");
    Variable noise_eval(randn({256, noise_dim}, DType::Float32, device), false);
    auto fake_eval = G_forward(noise_eval).tensor();
    auto real_eval = sample_real(256);

    float real_mean = tenzor::mean(real_eval).item<float>();
    float fake_mean = tenzor::mean(fake_eval).item<float>();
    auto real_ctr = real_eval - real_mean;
    auto fake_ctr = fake_eval - fake_mean;
    float real_std = std::sqrt(tenzor::mean(real_ctr * real_ctr).item<float>());
    float fake_std = std::sqrt(tenzor::mean(fake_ctr * fake_ctr).item<float>());

    std::cout << "Target distribution:    N(3.0, 0.5)\n";
    std::cout << "Real stats:             mean=" << real_mean << ", std=" << real_std << "\n";
    std::cout << "Generator stats:        mean=" << fake_mean << ", std=" << fake_std << "\n";
    std::cout << "\nGAN demonstrated with autograd!\n";
    std::cout << "Detaching the fake samples during the D step keeps G out of that update.\n";

    finalize();
    return 0;
}
