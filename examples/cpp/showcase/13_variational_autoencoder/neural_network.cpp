/**
 * @file neural_network.cpp
 * @brief Variational Autoencoder using Tenzor's high-level Neural Network API
 *
 * A VAE built from nn::Linear layers with an Adam optimizer. Reconstruction
 * plus KL divergence loss shows how custom training loops compose cleanly
 * with the NN API - the reparameterization is just arithmetic on Variables.
 *
 * Usage: ./13_variational_autoencoder_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

class VAE : public nn::Module {
public:
    VAE(int64_t input_dim, int64_t hidden_dim, int64_t latent_dim)
        : latent_dim_(latent_dim) {
        enc_fc  = std::make_shared<nn::Linear>(input_dim, hidden_dim);
        enc_mu  = std::make_shared<nn::Linear>(hidden_dim, latent_dim);
        enc_lv  = std::make_shared<nn::Linear>(hidden_dim, latent_dim);
        dec_fc1 = std::make_shared<nn::Linear>(latent_dim, hidden_dim);
        dec_fc2 = std::make_shared<nn::Linear>(hidden_dim, input_dim);
        register_module("enc_fc",  enc_fc);
        register_module("enc_mu",  enc_mu);
        register_module("enc_lv",  enc_lv);
        register_module("dec_fc1", dec_fc1);
        register_module("dec_fc2", dec_fc2);
    }

    struct EncOut { Variable mu, logvar; };

    EncOut encode(const Variable& x) {
        auto h = nn::relu(enc_fc->forward(x));
        return {enc_mu->forward(h), enc_lv->forward(h)};
    }

    Variable reparameterize(const Variable& mu, const Variable& logvar, const Device& device) {
        auto std_dev = tenzor::exp(logvar * 0.5f);
        auto mu_shape = mu.shape();
        std::vector<int64_t> eps_shape(mu_shape.begin(), mu_shape.end());
        auto eps = Variable(randn(eps_shape, DType::Float32, device), false);
        return mu + std_dev * eps;
    }

    Variable decode(const Variable& z) {
        return nn::sigmoid(dec_fc2->forward(nn::relu(dec_fc1->forward(z))));
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto e = encode(input);
        auto z = reparameterize(e.mu, e.logvar, input.device());
        return decode(z);
    }

    int64_t latent_dim() const { return latent_dim_; }

private:
    std::shared_ptr<nn::Linear> enc_fc, enc_mu, enc_lv, dec_fc1, dec_fc2;
    int64_t latent_dim_;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();
    showcase::print_header("Variational Autoencoder - Neural Network API", device);

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
    auto X = from_data(X_data.data(), {batch_size, input_dim}, device);
    showcase::print_tensor_info("Input X", X);

    auto model = std::make_shared<VAE>(input_dim, hidden_dim, latent_dim);
    model->to(device);

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.005f);
    nn::MSELoss recon_criterion;

    float kl_weight = 0.001f;
    int num_epochs = 500;
    int print_every = 50;

    showcase::print_section("Model Architecture");
    std::cout << "Encoder:  Linear(" << input_dim << " -> " << hidden_dim << ") -> ReLU\n";
    std::cout << "          Linear(" << hidden_dim << " -> " << latent_dim << ")  (mu)\n";
    std::cout << "          Linear(" << hidden_dim << " -> " << latent_dim << ")  (log_var)\n";
    std::cout << "Decoder:  Linear(" << latent_dim << " -> " << hidden_dim << ") -> ReLU\n";
    std::cout << "          Linear(" << hidden_dim << " -> " << input_dim << ") -> Sigmoid\n";

    showcase::print_section("Training");
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        optimizer.zero_grad();

        Variable input(X, false);
        auto e = model->encode(input);
        auto z = model->reparameterize(e.mu, e.logvar, device);
        auto x_hat = model->decode(z);

        auto recon = recon_criterion(x_hat, input);
        auto kl_per = (e.logvar + 1.0f) - e.mu * e.mu - tenzor::exp(e.logvar);
        auto kl_loss = sum(kl_per) * (-0.5f / static_cast<float>(batch_size));
        auto loss = recon + kl_loss * kl_weight;

        loss.backward();
        optimizer.step();

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] Total: " << loss.tensor().item<float>()
                      << "  Recon: " << recon.tensor().item<float>()
                      << "  KL: " << kl_loss.tensor().item<float>() << "\n";
        }
    }

    showcase::print_section("Final Results");
    model->eval();

    Variable x_eval(X, false);
    auto e = model->encode(x_eval);
    auto x_hat = model->decode(e.mu);  // use mu directly for deterministic recon
    auto err = x_hat.tensor() - X;
    std::cout << "Final deterministic recon MSE (z = mu): "
              << tenzor::mean(err * err).item<float>() << "\n\n";

    // Sample from prior
    std::cout << "Generating 3 new samples from z ~ N(0, I)...\n";
    Variable z_prior(randn({3, latent_dim}, DType::Float32, device), false);
    auto gen = model->decode(z_prior);
    auto z_cpu = z_prior.tensor().cpu();
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

    std::cout << "\nVAE solved using Neural Network API!\n";
    std::cout << "The model exposes encode()/decode()/reparameterize() for generation.\n";

    finalize();
    return 0;
}
