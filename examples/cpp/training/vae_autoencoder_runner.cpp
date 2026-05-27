/**
 * @file vae_autoencoder_runner.cpp
 * @brief Implementation of the VAE autograd runner.
 *
 * RR.18 (audit-11): exposes the SimpleVAE forward + reconstruction loop
 * from vae_autoencoder.cpp so the regression test can assert that
 * backward propagates through the Linear + ReLU + Sigmoid + MSELoss
 * pipeline and the loss actually decreases.
 *
 * Kept small (batch=2, input_dim=16, latent_dim=4) so the regression
 * runs in well under a second on CPU.
 */

#include "vae_autoencoder_runner.hpp"

#include <algorithm>
#include <iostream>
#include <memory>

#include "tenzor/tenzor.hpp"

namespace tenzor::examples::vae_autoencoder {

namespace {

using ::tenzor::Variable;
using ::tenzor::Tensor;
using ::tenzor::nn::Module;
using ::tenzor::nn::Linear;
using ::tenzor::nn::ReLU;
using ::tenzor::nn::Sigmoid;
using ::tenzor::nn::MSELoss;

class SimpleVAE : public Module {
public:
    SimpleVAE(int64_t input_dim, int64_t hidden_dim, int64_t latent_dim)
        : input_dim_(input_dim), latent_dim_(latent_dim) {
        enc_fc1_       = std::make_shared<Linear>(input_dim, hidden_dim);
        enc_fc_mu_     = std::make_shared<Linear>(hidden_dim, latent_dim);
        enc_fc_logvar_ = std::make_shared<Linear>(hidden_dim, latent_dim);
        dec_fc1_       = std::make_shared<Linear>(latent_dim, hidden_dim);
        dec_fc2_       = std::make_shared<Linear>(hidden_dim, input_dim);
        relu_          = std::make_shared<ReLU>();
        sigmoid_       = std::make_shared<Sigmoid>();
        register_module("enc_fc1", enc_fc1_);
        register_module("enc_fc_mu", enc_fc_mu_);
        register_module("enc_fc_logvar", enc_fc_logvar_);
        register_module("dec_fc1", dec_fc1_);
        register_module("dec_fc2", dec_fc2_);
        register_module("relu", relu_);
        register_module("sigmoid", sigmoid_);
    }

    std::tuple<Variable, Variable, Variable> encode_decode(const Variable& x) {
        using namespace ::tenzor;
        auto h = relu_->forward(enc_fc1_->forward(x));
        auto mu = enc_fc_mu_->forward(h);
        auto logvar = enc_fc_logvar_->forward(h);
        // Reparameterization: z = mu + std * epsilon
        auto std_tensor = tenzor::exp(logvar.tensor() * 0.5f);
        auto shape = std_tensor.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto eps = randn(shape_vec, DType::Float32, std_tensor.device());
        auto z_tensor = mu.tensor() + std_tensor * eps;
        Variable z(z_tensor, mu.requires_grad());
        auto dec_h = relu_->forward(dec_fc1_->forward(z));
        auto recon = sigmoid_->forward(dec_fc2_->forward(dec_h));
        return {recon, mu, logvar};
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto [recon, mu, logvar] = encode_decode(x);
        return recon;
    }

private:
    int64_t input_dim_;
    int64_t latent_dim_;
    std::shared_ptr<Linear> enc_fc1_, enc_fc_mu_, enc_fc_logvar_;
    std::shared_ptr<Linear> dec_fc1_, dec_fc2_;
    std::shared_ptr<ReLU> relu_;
    std::shared_ptr<Sigmoid> sigmoid_;
};

}  // namespace

int run_vae_training(int num_steps,
                     double* out_initial,
                     double* out_final,
                     ::tenzor::Device device,
                     bool verbose) {
    using namespace ::tenzor;

    const int64_t input_dim  = 16;
    const int64_t hidden_dim = 16;
    const int64_t latent_dim = 4;
    const int batch_size = 2;

    manual_seed(42);

    auto model = std::make_shared<SimpleVAE>(input_dim, hidden_dim, latent_dim);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.01f);
    MSELoss criterion;

    // Fixed input batch so loss-decrease is deterministic.
    auto x_data = randn({batch_size, input_dim}, DType::Float32, device);
    auto x_norm = tenzor::clamp(x_data, 0.0f, 1.0f);

    double initial_loss = 0.0;
    double final_loss   = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        Variable x(x_norm, true);
        optimizer.zero_grad();
        auto [recon, mu, logvar] = model->encode_decode(x);
        Variable target_var(x.tensor(), false);
        auto loss = criterion(recon, target_var);
        loss.backward();
        optimizer.step();

        const double loss_v =
            static_cast<double>(loss.tensor().cpu().data<float>()[0]);
        if (step == 0) initial_loss = loss_v;
        final_loss = loss_v;
        if (verbose) {
            std::cout << "step " << (step + 1) << "/" << num_steps
                      << " loss=" << loss_v << "\n";
        }
    }

    if (out_initial) *out_initial = initial_loss;
    if (out_final)   *out_final   = final_loss;
    return 0;
}

}  // namespace tenzor::examples::vae_autoencoder
