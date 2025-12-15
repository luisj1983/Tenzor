/**
 * @file neural_network.cpp
 * @brief Autoencoder using Tenzor's high-level Neural Network API
 *
 * This example demonstrates an autoencoder using nn::Module and nn::Sequential
 * for clean, modular architecture.
 *
 * Usage: ./06_autoencoder_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

/**
 * @brief Encoder network
 */
class Encoder : public nn::Module {
public:
    Encoder(int64_t input_dim, int64_t hidden_dim, int64_t latent_dim) {
        fc1 = std::make_shared<nn::Linear>(input_dim, hidden_dim);
        fc2 = std::make_shared<nn::Linear>(hidden_dim, latent_dim);

        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto h = nn::relu(fc1->forward(input));
        auto latent = fc2->forward(h);  // No activation on latent
        return latent;
    }

private:
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::Linear> fc2;
};

/**
 * @brief Decoder network
 */
class Decoder : public nn::Module {
public:
    Decoder(int64_t latent_dim, int64_t hidden_dim, int64_t output_dim) {
        fc1 = std::make_shared<nn::Linear>(latent_dim, hidden_dim);
        fc2 = std::make_shared<nn::Linear>(hidden_dim, output_dim);

        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto h = nn::relu(fc1->forward(input));
        auto out = nn::sigmoid(fc2->forward(h));  // Sigmoid for [0,1] output
        return out;
    }

private:
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::Linear> fc2;
};

/**
 * @brief Full Autoencoder
 */
class Autoencoder : public nn::Module {
public:
    Autoencoder(int64_t input_dim, int64_t hidden_dim, int64_t latent_dim) {
        encoder = std::make_shared<Encoder>(input_dim, hidden_dim, latent_dim);
        decoder = std::make_shared<Decoder>(latent_dim, hidden_dim, input_dim);

        register_module("encoder", encoder);
        register_module("decoder", decoder);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto latent = encoder->forward(input);
        auto reconstruction = decoder->forward(latent);
        return reconstruction;
    }

    auto encode(const Variable& input) -> Variable {
        return encoder->forward(input);
    }

    auto decode(const Variable& latent) -> Variable {
        return decoder->forward(latent);
    }

private:
    std::shared_ptr<Encoder> encoder;
    std::shared_ptr<Decoder> decoder;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Autoencoder - Neural Network API (High-Level)", device);

    manual_seed(42);

    // Generate synthetic data
    int batch_size = 32;
    int input_dim = 16;
    int hidden_dim = 8;
    int latent_dim = 4;

    auto X = rand({batch_size, input_dim}, DType::Float32, device);

    showcase::print_tensor_info("Input X", X);

    // Create model
    auto model = std::make_shared<Autoencoder>(input_dim, hidden_dim, latent_dim);
    model->to(device);

    // Create optimizer
    auto params = model->parameters();
    optim::Adam optimizer(params, 0.01f);

    // Create loss function
    nn::MSELoss criterion;

    showcase::print_section("Model Architecture");
    std::cout << "Autoencoder:\n";
    std::cout << "  Encoder:\n";
    std::cout << "    fc1: Linear(" << input_dim << ", " << hidden_dim << ") + ReLU\n";
    std::cout << "    fc2: Linear(" << hidden_dim << ", " << latent_dim << ")\n";
    std::cout << "  Decoder:\n";
    std::cout << "    fc1: Linear(" << latent_dim << ", " << hidden_dim << ") + ReLU\n";
    std::cout << "    fc2: Linear(" << hidden_dim << ", " << input_dim << ") + Sigmoid\n";
    std::cout << "\nTotal parameters: " << params.size() << "\n";

    // Training parameters
    int num_epochs = 500;
    int print_every = 50;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();

        optimizer.zero_grad();

        // Forward pass
        Variable input(X, false);
        auto reconstruction = model->forward(input);

        // Compute loss (reconstruction error)
        auto loss = criterion(reconstruction, input);

        // Backward pass
        loss.backward();

        // Update weights
        optimizer.step();

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                      << "Reconstruction Loss: " << loss_val << "\n";
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    model->eval();

    Variable X_eval(X, false);
    auto reconstruction = model->forward(X_eval);
    auto latent = model->encode(X_eval);

    auto loss_final = criterion(reconstruction, X_eval);
    float final_loss = loss_final.tensor().item<float>();

    std::cout << "Final Reconstruction Loss: " << final_loss << "\n\n";

    // Show sample reconstructions
    showcase::print_section("Sample Reconstructions");

    auto X_cpu = X.cpu();
    auto recon_cpu = reconstruction.tensor().cpu();
    auto latent_cpu = latent.tensor().cpu();

    std::cout << "Input vs Reconstruction (first 4 features):\n\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "Sample " << i << ":\n";
        std::cout << "  Input:  [";
        for (int j = 0; j < 4; ++j) {
            std::cout << X_cpu.data<float>()[i * input_dim + j];
            if (j < 3) std::cout << ", ";
        }
        std::cout << ", ...]\n";

        std::cout << "  Latent: [";
        for (int j = 0; j < latent_dim; ++j) {
            std::cout << latent_cpu.data<float>()[i * latent_dim + j];
            if (j < latent_dim - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "  Recon:  [";
        for (int j = 0; j < 4; ++j) {
            std::cout << recon_cpu.data<float>()[i * input_dim + j];
            if (j < 3) std::cout << ", ";
        }
        std::cout << ", ...]\n\n";
    }

    std::cout << "Autoencoder solved using Neural Network API!\n";
    std::cout << "Dimensionality reduced from " << input_dim << " to " << latent_dim << ".\n";

    finalize();
    return 0;
}
