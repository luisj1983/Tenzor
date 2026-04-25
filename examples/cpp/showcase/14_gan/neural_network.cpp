/**
 * @file neural_network.cpp
 * @brief GAN using Tenzor's high-level Neural Network API
 *
 * Two nn::Module classes (GANGenerator and GANDiscriminator), two Adam optimizers,
 * and an alternating training loop. Standard GAN setup, minimal boilerplate.
 *
 * Usage: ./14_gan_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;

class GANGenerator : public nn::Module {
public:
    GANGenerator(int64_t noise_dim, int64_t hidden, int64_t out_dim) {
        fc1 = std::make_shared<nn::Linear>(noise_dim, hidden);
        fc2 = std::make_shared<nn::Linear>(hidden, out_dim);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }
    auto forward_impl(const Variable& input) -> Variable override {
        return fc2->forward(nn::relu(fc1->forward(input)));
    }
private:
    std::shared_ptr<nn::Linear> fc1, fc2;
};

class GANDiscriminator : public nn::Module {
public:
    GANDiscriminator(int64_t in_dim, int64_t hidden) {
        fc1 = std::make_shared<nn::Linear>(in_dim, hidden);
        fc2 = std::make_shared<nn::Linear>(hidden, 1);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }
    // Returns raw logits (use BCEWithLogitsLoss)
    auto forward_impl(const Variable& input) -> Variable override {
        return fc2->forward(nn::relu(fc1->forward(input)));
    }
private:
    std::shared_ptr<nn::Linear> fc1, fc2;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("GAN - Neural Network API (High-Level)", device);
    manual_seed(42);

    int batch_size = 64;
    int noise_dim = 4;
    int hidden = 16;

    auto sample_real = [&](int n) {
        auto z = randn({n, 1}, DType::Float32, device);
        return z * 0.5f + 3.0f;
    };

    auto G = std::make_shared<GANGenerator>(noise_dim, hidden, 1);
    auto D = std::make_shared<GANDiscriminator>(1, hidden);
    G->to(device);
    D->to(device);

    optim::Adam g_opt(G->parameters(), 0.002f);
    optim::Adam d_opt(D->parameters(), 0.002f);
    nn::BCEWithLogitsLoss bce;

    showcase::print_section("Model Architecture");
    std::cout << "GANGenerator:     Linear(" << noise_dim << " -> " << hidden << ") -> ReLU -> Linear(" << hidden << " -> 1)\n";
    std::cout << "GANDiscriminator: Linear(1 -> " << hidden << ") -> ReLU -> Linear(" << hidden << " -> 1 logit)\n";

    int num_epochs = 3000;
    int print_every = 300;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // -------- D step --------
        D->train();
        d_opt.zero_grad();

        Variable real(sample_real(batch_size), false);
        Variable noise(randn({batch_size, noise_dim}, DType::Float32, device), false);
        auto fake_v = G->forward(noise);
        Variable fake_det(fake_v.tensor(), false);  // detach

        auto d_logit_real = D->forward(real);
        auto d_logit_fake = D->forward(fake_det);
        auto ones_t  = ones({batch_size, 1}, DType::Float32, device);
        auto zeros_t = zeros({batch_size, 1}, DType::Float32, device);
        Variable y_real(ones_t, false), y_fake(zeros_t, false);

        auto d_loss = bce(d_logit_real, y_real) + bce(d_logit_fake, y_fake);
        d_loss.backward();
        d_opt.step();

        // -------- G step --------
        G->train();
        g_opt.zero_grad();
        Variable noise2(randn({batch_size, noise_dim}, DType::Float32, device), false);
        auto fake2 = G->forward(noise2);
        auto d_logit_fake2 = D->forward(fake2);
        auto g_loss = bce(d_logit_fake2, y_real);  // non-saturating: label fakes as real

        g_loss.backward();
        g_opt.step();

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] D_loss: " << d_loss.tensor().item<float>()
                      << "  G_loss: " << g_loss.tensor().item<float>() << "\n";
        }
    }

    showcase::print_section("Final Results");
    G->eval();
    Variable noise_eval(randn({512, noise_dim}, DType::Float32, device), false);
    auto fake_eval = G->forward(noise_eval).tensor();
    auto real_eval = sample_real(512);

    float real_mean = tenzor::mean(real_eval).item<float>();
    float fake_mean = tenzor::mean(fake_eval).item<float>();
    auto rc = real_eval - real_mean, fc = fake_eval - fake_mean;
    float real_std = std::sqrt(tenzor::mean(rc * rc).item<float>());
    float fake_std = std::sqrt(tenzor::mean(fc * fc).item<float>());

    std::cout << "Target distribution:    N(3.0, 0.5)\n";
    std::cout << "Real stats:             mean=" << real_mean << ", std=" << real_std << "\n";
    std::cout << "Generator stats:        mean=" << fake_mean << ", std=" << fake_std << "\n";
    std::cout << "\nGAN solved using Neural Network API!\n";

    finalize();
    return 0;
}
