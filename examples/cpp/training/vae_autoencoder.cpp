/**
 * @file vae_autoencoder.cpp
 * @brief Variational Autoencoder with Activation Functions Demo
 *
 * This example demonstrates:
 * - VAE encoder-decoder architecture (simplified)
 * - Activation functions: ReLU, ReLU6, LeakyReLU, ELU, SELU, GELU, Swish/SiLU,
 *   Mish, Sigmoid, Tanh, Softmax, LogSoftmax
 * - Linear layers for VAE
 * - MSELoss for reconstruction
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <memory>

#include "tenzor/tenzor.hpp"

using namespace tenzor;
using namespace tenzor::nn;

// ============================================================================
// Activation Function Demo Module
// ============================================================================

class ActivationDemo : public Module {
public:
    ActivationDemo() {
        relu_ = std::make_shared<ReLU>();
        relu6_ = std::make_shared<ReLU6>();
        leaky_relu_ = std::make_shared<LeakyReLU>(0.2f);
        elu_ = std::make_shared<ELU>(1.0f);
        selu_ = std::make_shared<SELU>();
        gelu_ = std::make_shared<GELU>();
        swish_ = std::make_shared<Swish>();
        mish_ = std::make_shared<Mish>();
        sigmoid_ = std::make_shared<Sigmoid>();
        tanh_ = std::make_shared<Tanh>();
        softmax_ = std::make_shared<Softmax>(-1);
        log_softmax_ = std::make_shared<LogSoftmax>(-1);

        register_module("relu", relu_);
        register_module("relu6", relu6_);
        register_module("leaky_relu", leaky_relu_);
        register_module("elu", elu_);
        register_module("selu", selu_);
        register_module("gelu", gelu_);
        register_module("swish", swish_);
        register_module("mish", mish_);
        register_module("sigmoid", sigmoid_);
        register_module("tanh", tanh_);
        register_module("softmax", softmax_);
        register_module("log_softmax", log_softmax_);
    }

    void demo_all_activations(Device device) {
        std::cout << "\n=== Activation Functions Demo ===\n\n";

        std::vector<float> test_values = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
        auto input = from_data(test_values.data(), {1, static_cast<int64_t>(test_values.size())}, device);
        Variable x(input, false);

        std::cout << "Input: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]\n\n";
        std::cout << "Activation        | Output values\n";
        std::cout << "------------------|--------------------------------------------------\n";

        auto print_output = [&](const std::string& name, const Variable& y) {
            auto y_cpu = y.tensor().cpu();
            const float* y_data = y_cpu.data<float>();
            std::cout << std::setw(18) << name << " | ";
            for (size_t i = 0; i < test_values.size(); ++i) {
                std::cout << std::setw(7) << std::fixed << std::setprecision(3) << y_data[i] << " ";
            }
            std::cout << "\n";
        };

        print_output("ReLU", relu_->forward(x));
        print_output("ReLU6", relu6_->forward(x));
        print_output("LeakyReLU(0.2)", leaky_relu_->forward(x));
        print_output("ELU(1.0)", elu_->forward(x));
        print_output("SELU", selu_->forward(x));
        print_output("GELU", gelu_->forward(x));
        print_output("Swish/SiLU", swish_->forward(x));
        print_output("Mish", mish_->forward(x));
        print_output("Sigmoid", sigmoid_->forward(x));
        print_output("Tanh", tanh_->forward(x));

        std::cout << "Softmax           | [sum = 1.0, probability distribution]\n";
        std::cout << "LogSoftmax        | [log probabilities, for NLLLoss]\n";

        std::cout << "\nActivation properties:\n";
        std::cout << "  ReLU:      max(0, x) - simple, efficient, widely used\n";
        std::cout << "  ReLU6:     min(max(0, x), 6) - mobile/quantization friendly\n";
        std::cout << "  LeakyReLU: x if x>0 else alpha*x - avoids dead neurons\n";
        std::cout << "  ELU:       x if x>0 else alpha*(e^x-1) - smooth, self-normalizing\n";
        std::cout << "  SELU:      scale*(x if x>0 else alpha*(e^x-1)) - self-normalizing\n";
        std::cout << "  GELU:      x*Phi(x) - smooth, used in transformers\n";
        std::cout << "  Swish:     x*sigmoid(x) - smooth, self-gated\n";
        std::cout << "  Mish:      x*tanh(softplus(x)) - smooth, non-monotonic\n";
        std::cout << "  Sigmoid:   1/(1+e^-x) - output in (0,1)\n";
        std::cout << "  Tanh:      (e^x-e^-x)/(e^x+e^-x) - output in (-1,1)\n";
    }

    auto forward_impl(const Variable& x) -> Variable override {
        return x;
    }

private:
    std::shared_ptr<ReLU> relu_;
    std::shared_ptr<ReLU6> relu6_;
    std::shared_ptr<LeakyReLU> leaky_relu_;
    std::shared_ptr<ELU> elu_;
    std::shared_ptr<SELU> selu_;
    std::shared_ptr<GELU> gelu_;
    std::shared_ptr<Swish> swish_;
    std::shared_ptr<Mish> mish_;
    std::shared_ptr<Sigmoid> sigmoid_;
    std::shared_ptr<Tanh> tanh_;
    std::shared_ptr<Softmax> softmax_;
    std::shared_ptr<LogSoftmax> log_softmax_;
};

// ============================================================================
// Simple VAE (Fully Connected)
// ============================================================================

class SimpleVAE : public Module {
public:
    SimpleVAE(int64_t input_dim, int64_t hidden_dim, int64_t latent_dim)
        : input_dim_(input_dim), latent_dim_(latent_dim) {

        enc_fc1_ = std::make_shared<Linear>(input_dim, hidden_dim);
        enc_fc_mu_ = std::make_shared<Linear>(hidden_dim, latent_dim);
        enc_fc_logvar_ = std::make_shared<Linear>(hidden_dim, latent_dim);

        dec_fc1_ = std::make_shared<Linear>(latent_dim, hidden_dim);
        dec_fc2_ = std::make_shared<Linear>(hidden_dim, input_dim);

        relu_ = std::make_shared<ReLU>();
        sigmoid_ = std::make_shared<Sigmoid>();

        register_module("enc_fc1", enc_fc1_);
        register_module("enc_fc_mu", enc_fc_mu_);
        register_module("enc_fc_logvar", enc_fc_logvar_);
        register_module("dec_fc1", dec_fc1_);
        register_module("dec_fc2", dec_fc2_);
        register_module("relu", relu_);
        register_module("sigmoid", sigmoid_);
    }

    std::tuple<Variable, Variable, Variable> encode_decode(const Variable& x) {
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

    Variable sample(int batch_size, Device device) {
        auto z = randn({batch_size, latent_dim_}, DType::Float32, device);
        Variable z_var(z, false);
        auto dec_h = relu_->forward(dec_fc1_->forward(z_var));
        return sigmoid_->forward(dec_fc2_->forward(dec_h));
    }

private:
    int64_t input_dim_;
    int64_t latent_dim_;
    std::shared_ptr<Linear> enc_fc1_, enc_fc_mu_, enc_fc_logvar_;
    std::shared_ptr<Linear> dec_fc1_, dec_fc2_;
    std::shared_ptr<ReLU> relu_;
    std::shared_ptr<Sigmoid> sigmoid_;
};

// ============================================================================
// VAE with Different Activations
// ============================================================================

class GELUEncoder : public Module {
public:
    GELUEncoder(int64_t input_dim, int64_t hidden_dim, int64_t latent_dim) {
        fc1_ = std::make_shared<Linear>(input_dim, hidden_dim);
        fc2_ = std::make_shared<Linear>(hidden_dim, hidden_dim);
        fc_mu_ = std::make_shared<Linear>(hidden_dim, latent_dim);
        fc_logvar_ = std::make_shared<Linear>(hidden_dim, latent_dim);
        gelu_ = std::make_shared<GELU>();

        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("fc_mu", fc_mu_);
        register_module("fc_logvar", fc_logvar_);
        register_module("gelu", gelu_);
    }

    std::pair<Variable, Variable> forward_impl_pair(const Variable& x) {
        auto h = gelu_->forward(fc1_->forward(x));
        h = gelu_->forward(fc2_->forward(h));
        auto mu = fc_mu_->forward(h);
        auto logvar = fc_logvar_->forward(h);
        return {mu, logvar};
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto [mu, logvar] = forward_impl_pair(x);
        return mu;
    }

private:
    std::shared_ptr<Linear> fc1_, fc2_, fc_mu_, fc_logvar_;
    std::shared_ptr<GELU> gelu_;
};

class SwishDecoder : public Module {
public:
    SwishDecoder(int64_t latent_dim, int64_t hidden_dim, int64_t output_dim) {
        fc1_ = std::make_shared<Linear>(latent_dim, hidden_dim);
        fc2_ = std::make_shared<Linear>(hidden_dim, hidden_dim);
        fc_out_ = std::make_shared<Linear>(hidden_dim, output_dim);
        swish_ = std::make_shared<Swish>();
        sigmoid_ = std::make_shared<Sigmoid>();

        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("fc_out", fc_out_);
        register_module("swish", swish_);
        register_module("sigmoid", sigmoid_);
    }

    auto forward_impl(const Variable& z) -> Variable override {
        auto h = swish_->forward(fc1_->forward(z));
        h = swish_->forward(fc2_->forward(h));
        return sigmoid_->forward(fc_out_->forward(h));
    }

private:
    std::shared_ptr<Linear> fc1_, fc2_, fc_out_;
    std::shared_ptr<Swish> swish_;
    std::shared_ptr<Sigmoid> sigmoid_;
};

// ============================================================================
// Training Functions
// ============================================================================

void train_simple_vae(Device device) {
    std::cout << "\n=== Training Simple VAE (ReLU) ===\n\n";

    int64_t input_dim = 784;
    int64_t hidden_dim = 256;
    int64_t latent_dim = 32;
    int batch_size = 32;
    int num_train = 500;
    int num_epochs = 10;

    auto model = std::make_shared<SimpleVAE>(input_dim, hidden_dim, latent_dim);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);

    MSELoss recon_criterion;

    std::cout << "Configuration:\n";
    std::cout << "  Model: Simple VAE (fully connected)\n";
    std::cout << "  Input dim: " << input_dim << "\n";
    std::cout << "  Latent dim: " << latent_dim << "\n";
    std::cout << "  Activation: ReLU (encoder), Sigmoid (output)\n";
    std::cout << "  Loss: MSE (reconstruction)\n";
    std::cout << "  Optimizer: Adam (lr=0.001)\n\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;

        for (int i = 0; i < num_train; i += batch_size) {
            int end = std::min(i + batch_size, num_train);
            int actual_batch = end - i;

            auto x_data = randn({actual_batch, input_dim}, DType::Float32, device);
            auto x_norm = tenzor::clamp(x_data, 0.0f, 1.0f);
            Variable x(x_norm, true);

            optimizer.zero_grad();

            auto [recon, mu, logvar] = model->encode_decode(x);

            Variable target_var(x.tensor(), false);
            auto recon_loss = recon_criterion(recon, target_var);

            recon_loss.backward();
            optimizer.step();

            auto loss_cpu = recon_loss.tensor().cpu();
            epoch_loss += loss_cpu.data<float>()[0];
            num_batches++;
        }

        if ((epoch + 1) % 2 == 0 || epoch == 0) {
            std::cout << "Epoch " << std::setw(2) << (epoch + 1) << "/" << num_epochs
                      << " | Recon Loss: " << std::fixed << std::setprecision(4)
                      << (epoch_loss / num_batches) << "\n";
        }
    }

    std::cout << "\nGenerating samples from learned latent space...\n";
    model->eval();
    auto samples = model->sample(4, device);
    std::cout << "Generated " << samples.shape()[0] << " samples of dim " << samples.shape()[1] << "\n";
}

void demo_vae_with_gelu_swish(Device device) {
    std::cout << "\n=== VAE with GELU Encoder + Swish Decoder ===\n\n";

    int64_t input_dim = 256;
    int64_t hidden_dim = 128;
    int64_t latent_dim = 16;
    int batch_size = 16;
    int num_epochs = 5;

    auto encoder = std::make_shared<GELUEncoder>(input_dim, hidden_dim, latent_dim);
    auto decoder = std::make_shared<SwishDecoder>(latent_dim, hidden_dim, input_dim);
    encoder->to(device);
    decoder->to(device);

    // Combine parameters from both modules
    std::vector<std::shared_ptr<Variable>> all_params;
    auto enc_params = encoder->parameters();
    auto dec_params = decoder->parameters();
    all_params.insert(all_params.end(), enc_params.begin(), enc_params.end());
    all_params.insert(all_params.end(), dec_params.begin(), dec_params.end());

    optim::Adam optimizer(all_params, 0.001f);
    MSELoss criterion;

    std::cout << "Configuration:\n";
    std::cout << "  Encoder: 2-layer MLP with GELU\n";
    std::cout << "  Decoder: 2-layer MLP with Swish\n";
    std::cout << "  Input dim: " << input_dim << "\n";
    std::cout << "  Latent dim: " << latent_dim << "\n\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;

        for (int i = 0; i < 100; i += batch_size) {
            auto x_data = randn({batch_size, input_dim}, DType::Float32, device);
            auto x_norm = tenzor::clamp(x_data, 0.0f, 1.0f);
            Variable x(x_norm, true);

            optimizer.zero_grad();

            auto [mu, logvar] = encoder->forward_impl_pair(x);

            auto std_tensor = tenzor::exp(logvar.tensor() * 0.5f);
            auto shape = std_tensor.shape();
            std::vector<int64_t> shape_vec(shape.begin(), shape.end());
            auto eps = randn(shape_vec, DType::Float32, device);
            auto z_tensor = mu.tensor() + std_tensor * eps;
            Variable z(z_tensor, true);

            auto recon = decoder->forward(z);

            Variable target_var(x.tensor(), false);
            auto loss = criterion(recon, target_var);
            loss.backward();
            optimizer.step();

            epoch_loss += loss.tensor().cpu().data<float>()[0];
        }

        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs
                  << " | Loss: " << std::fixed << std::setprecision(4)
                  << (epoch_loss / (100 / batch_size)) << "\n";
    }
}

void demo_leaky_relu_variants(Device device) {
    std::cout << "\n=== LeakyReLU Variants Demo ===\n\n";

    std::vector<float> test_values = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto input = from_data(test_values.data(), {1, 5}, device);
    Variable x(input, false);

    std::vector<float> alphas = {0.01f, 0.1f, 0.2f, 0.3f};

    std::cout << "LeakyReLU with different negative slopes:\n\n";
    std::cout << "  Input: [-2.0, -1.0, 0.0, 1.0, 2.0]\n\n";

    for (float alpha : alphas) {
        auto leaky = std::make_shared<LeakyReLU>(alpha);
        auto output = leaky->forward(x);
        auto out_cpu = output.tensor().cpu();
        const float* out_data = out_cpu.data<float>();

        std::cout << "  alpha=" << std::fixed << std::setprecision(2) << alpha << ": [";
        for (int i = 0; i < 5; ++i) {
            std::cout << std::setprecision(2) << out_data[i];
            if (i < 4) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    std::cout << "\n  Lower alpha -> more like ReLU (sparse gradients)\n";
    std::cout << "  Higher alpha -> more linear (preserves gradients)\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    tenzor::initialize();

    Device device = Device::cpu();
    if (argc > 1) {
        std::string backend = argv[1];
        if (backend == "cuda") device = Device::cuda();
        else if (backend == "vulkan") device = Device::vulkan();
    }

    std::cout << "======================================================\n";
    std::cout << "   VAE Autoencoder - Activations & Components         \n";
    std::cout << "   Backend: " << device.to_string() << "\n";
    std::cout << "======================================================\n";

    std::cout << "\nComponents demonstrated:\n";
    std::cout << "  Activations: ReLU, ReLU6, LeakyReLU, ELU, SELU, GELU,\n";
    std::cout << "               Swish/SiLU, Mish, Sigmoid, Tanh, Softmax\n";
    std::cout << "  Layers: Linear\n";
    std::cout << "  Losses: MSELoss\n";
    std::cout << "  Models: Simple VAE, GELU+Swish VAE\n";

    try {
        ActivationDemo demo;
        demo.demo_all_activations(device);

        train_simple_vae(device);
        demo_vae_with_gelu_swish(device);
        demo_leaky_relu_variants(device);

        std::cout << "\n======================================================\n";
        std::cout << "   All VAE/activation examples completed!             \n";
        std::cout << "======================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    tenzor::finalize();
    return 0;
}
