/**
 * @file gru_time_series.cpp
 * @brief GRU-based Time Series Forecasting with Advanced Training Techniques
 *
 * This comprehensive example demonstrates:
 * - GRU and GRUCell layers for sequence modeling
 * - LayerNorm for stable training
 * - Multiple optimizers: Adam, SGD, AdamW
 * - Additional optimizers: RMSprop, Adagrad, Adadelta
 * - Learning rate schedulers: StepLR, CosineAnnealingLR, ExponentialLR
 * - Loss functions: MSELoss, SmoothL1Loss, L1Loss
 * - Dropout for regularization
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>

#include "tenzor/tenzor.hpp"

using namespace tenzor;
using namespace tenzor::nn;

// ============================================================================
// GRU Time Series Model
// ============================================================================

class GRUPredictor : public Module {
public:
    GRUPredictor(int64_t input_size, int64_t hidden_size, int64_t output_size)
        : hidden_size_(hidden_size) {

        // GRU layer
        gru_ = std::make_shared<GRU>(input_size, hidden_size);

        // Layer norm for stability
        norm_ = std::make_shared<LayerNorm>(std::vector<int64_t>{hidden_size});

        // Dropout for regularization
        dropout_ = std::make_shared<Dropout>(0.1f);

        // Output projection
        fc_ = std::make_shared<Linear>(hidden_size, output_size);

        register_module("gru", gru_);
        register_module("norm", norm_);
        register_module("dropout", dropout_);
        register_module("fc", fc_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        // input: (batch, seq_len, input_size)
        auto gru_out = gru_->forward(input);  // (batch, seq_len, hidden)

        // Get last timestep
        auto batch_size = input.shape()[0];
        auto seq_len = input.shape()[1];

        auto last_hidden = gru_out.tensor()
            .slice(1, seq_len - 1, seq_len)
            .reshape({batch_size, hidden_size_});

        Variable last_h(last_hidden, gru_out.requires_grad());

        // Normalize, dropout, project
        auto normed = norm_->forward(last_h);
        auto dropped = dropout_->forward(normed);
        return fc_->forward(dropped);
    }

private:
    int64_t hidden_size_;
    std::shared_ptr<GRU> gru_;
    std::shared_ptr<LayerNorm> norm_;
    std::shared_ptr<Dropout> dropout_;
    std::shared_ptr<Linear> fc_;
};

// ============================================================================
// Data Generation
// ============================================================================

std::pair<Tensor, Tensor> generate_sine_data(int num_samples, int seq_len, Device device) {
    std::vector<float> X_data(num_samples * seq_len);
    std::vector<float> y_data(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        float phase = static_cast<float>(i) * 0.1f;
        for (int t = 0; t < seq_len; ++t) {
            X_data[i * seq_len + t] = std::sin(phase + t * 0.1f);
        }
        y_data[i] = std::sin(phase + seq_len * 0.1f);  // Predict next value
    }

    auto X = from_data(X_data.data(), {num_samples, seq_len, 1}, device);
    auto y = from_data(y_data.data(), {num_samples, 1}, device);

    return {X, y};
}

// ============================================================================
// Training Functions
// ============================================================================

void train_with_adam_steplr(Device device) {
    std::cout << "\n=== Training with Adam + StepLR + MSELoss ===\n\n" << std::flush;

    int num_samples = 100;
    int seq_len = 10;
    int hidden_size = 32;

    auto [X, y] = generate_sine_data(num_samples, seq_len, device);

    auto model = std::make_shared<GRUPredictor>(1, hidden_size, 1);
    model->to(device);

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.01f);
    optim::StepLR scheduler(optimizer, 50, 0.5);

    MSELoss criterion;

    std::cout << "Config: Adam(lr=0.01), StepLR(step=50, gamma=0.5), MSELoss\n\n" << std::flush;

    int num_epochs = 200;
    int print_every = 40;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        optimizer.zero_grad();

        Variable input(X, false);
        auto pred = model->forward(input);

        Variable target(y, false);
        auto loss = criterion(pred, target);

        loss.backward();
        optimizer.step();
        scheduler.step();

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch " << std::setw(3) << (epoch + 1) << "/" << num_epochs
                      << " | Loss: " << std::fixed << std::setprecision(6)
                      << loss.tensor().item<float>()
                      << " | LR: " << scheduler.get_last_lr() << "\n" << std::flush;
        }
    }
}

void train_with_sgd_exponential(Device device) {
    std::cout << "\n=== Training with SGD + ExponentialLR + SmoothL1Loss ===\n\n" << std::flush;

    int num_samples = 100;
    int seq_len = 10;
    int hidden_size = 32;

    auto [X, y] = generate_sine_data(num_samples, seq_len, device);

    auto model = std::make_shared<GRUPredictor>(1, hidden_size, 1);
    model->to(device);

    auto params = model->parameters();
    optim::SGD optimizer(params, 0.1f, 0.9f);
    optim::ExponentialLR scheduler(optimizer, 0.99);

    SmoothL1Loss criterion(Reduction::Mean, 1.0);

    std::cout << "Config: SGD(lr=0.1, momentum=0.9), ExponentialLR(gamma=0.99), SmoothL1Loss\n\n" << std::flush;

    int num_epochs = 200;
    int print_every = 40;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        optimizer.zero_grad();

        Variable input(X, false);
        auto pred = model->forward(input);

        Variable target(y, false);
        auto loss = criterion(pred, target);

        loss.backward();
        optimizer.step();
        scheduler.step();

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch " << std::setw(3) << (epoch + 1) << "/" << num_epochs
                      << " | Loss: " << std::fixed << std::setprecision(6)
                      << loss.tensor().item<float>()
                      << " | LR: " << scheduler.get_last_lr() << "\n" << std::flush;
        }
    }
}

void train_with_adamw_cosine(Device device) {
    std::cout << "\n=== Training with AdamW + CosineAnnealingLR + L1Loss ===\n\n" << std::flush;

    int num_samples = 100;
    int seq_len = 10;
    int hidden_size = 32;

    auto [X, y] = generate_sine_data(num_samples, seq_len, device);

    auto model = std::make_shared<GRUPredictor>(1, hidden_size, 1);
    model->to(device);

    auto params = model->parameters();
    optim::AdamW optimizer(params, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);

    int T_max = 200;
    optim::CosineAnnealingLR scheduler(optimizer, T_max, 0.0001);

    L1Loss criterion;

    std::cout << "Config: AdamW(lr=0.01, wd=0.01), CosineAnnealing(T=" << T_max << "), L1Loss\n\n" << std::flush;

    int print_every = 40;

    for (int epoch = 0; epoch < T_max; ++epoch) {
        model->train();
        optimizer.zero_grad();

        Variable input(X, false);
        auto pred = model->forward(input);

        Variable target(y, false);
        auto loss = criterion(pred, target);

        loss.backward();
        optimizer.step();
        scheduler.step();

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch " << std::setw(3) << (epoch + 1) << "/" << T_max
                      << " | Loss: " << std::fixed << std::setprecision(6)
                      << loss.tensor().item<float>()
                      << " | LR: " << scheduler.get_last_lr() << "\n" << std::flush;
        }
    }
}

void demo_additional_optimizers(Device device) {
    std::cout << "\n=== Additional Optimizers Demo ===\n\n" << std::flush;

    int num_samples = 50;
    int seq_len = 10;
    int hidden_size = 16;
    int num_epochs = 50;

    auto [X, y] = generate_sine_data(num_samples, seq_len, device);
    MSELoss criterion;

    // RMSprop
    std::cout << "RMSprop (lr=0.01, alpha=0.99):\n" << std::flush;
    {
        auto model = std::make_shared<GRUPredictor>(1, hidden_size, 1);
        model->to(device);
        auto params = model->parameters();
        optim::RMSprop optimizer(params, 0.01f, 0.99f);

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            optimizer.zero_grad();
            Variable input(X, false);
            auto pred = model->forward(input);
            Variable target(y, false);
            auto loss = criterion(pred, target);
            loss.backward();
            optimizer.step();

            if ((epoch + 1) % 25 == 0) {
                std::cout << "  Epoch " << (epoch + 1) << " Loss: "
                          << loss.tensor().item<float>() << "\n" << std::flush;
            }
        }
    }

    // Adagrad
    std::cout << "\nAdagrad (lr=0.1):\n" << std::flush;
    {
        auto model = std::make_shared<GRUPredictor>(1, hidden_size, 1);
        model->to(device);
        auto params = model->parameters();
        optim::Adagrad optimizer(params, 0.1f);

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            optimizer.zero_grad();
            Variable input(X, false);
            auto pred = model->forward(input);
            Variable target(y, false);
            auto loss = criterion(pred, target);
            loss.backward();
            optimizer.step();

            if ((epoch + 1) % 25 == 0) {
                std::cout << "  Epoch " << (epoch + 1) << " Loss: "
                          << loss.tensor().item<float>() << "\n" << std::flush;
            }
        }
    }

    // Adadelta
    std::cout << "\nAdadelta (rho=0.9):\n" << std::flush;
    {
        auto model = std::make_shared<GRUPredictor>(1, hidden_size, 1);
        model->to(device);
        auto params = model->parameters();
        optim::Adadelta optimizer(params, 1.0f, 0.9f);

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            optimizer.zero_grad();
            Variable input(X, false);
            auto pred = model->forward(input);
            Variable target(y, false);
            auto loss = criterion(pred, target);
            loss.backward();
            optimizer.step();

            if ((epoch + 1) % 25 == 0) {
                std::cout << "  Epoch " << (epoch + 1) << " Loss: "
                          << loss.tensor().item<float>() << "\n" << std::flush;
            }
        }
    }
}

void demo_gru_cell(Device device) {
    std::cout << "\n=== GRUCell Manual Processing ===\n\n" << std::flush;

    int input_size = 1;
    int hidden_size = 16;
    int seq_len = 5;
    int batch_size = 2;

    auto gru_cell = std::make_shared<GRUCell>(input_size, hidden_size, true);
    auto fc = std::make_shared<Linear>(hidden_size, 1);
    gru_cell->to(device);
    fc->to(device);

    std::cout << "GRUCell: input=" << input_size << ", hidden=" << hidden_size << "\n" << std::flush;
    std::cout << "Processing " << seq_len << " timesteps manually...\n\n" << std::flush;

    // Initialize hidden state
    auto h = zeros({batch_size, hidden_size}, DType::Float32, device);
    Variable hidden(h, true);

    // Process sequence step by step
    for (int t = 0; t < seq_len; ++t) {
        auto x_t = randn({batch_size, input_size}, DType::Float32, device);
        Variable x_var(x_t, true);
        hidden = gru_cell->forward(x_var, hidden);
        std::cout << "  Step " << t << ": hidden norm = "
                  << tenzor::mean(tenzor::abs(hidden.tensor())).item<float>() << "\n" << std::flush;
    }

    auto output = fc->forward(hidden);
    std::cout << "\nFinal output shape: [" << output.shape()[0] << ", "
              << output.shape()[1] << "]\n" << std::flush;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    initialize();

    Device device = Device::cpu();
    if (argc > 1) {
        std::string backend = argv[1];
        if (backend == "cuda") device = Device::cuda();
        else if (backend == "vulkan") device = Device::vulkan();
    }

    std::cout << "============================================================\n" << std::flush;
    std::cout << "   GRU Time Series Forecasting - Component Coverage\n" << std::flush;
    std::cout << "   Backend: " << device.to_string() << "\n" << std::flush;
    std::cout << "============================================================\n" << std::flush;

    std::cout << "\nComponents demonstrated:\n" << std::flush;
    std::cout << "  Layers: GRU, GRUCell, LayerNorm, Linear, Dropout\n" << std::flush;
    std::cout << "  Optimizers: Adam, SGD, AdamW, RMSprop, Adagrad, Adadelta\n" << std::flush;
    std::cout << "  Schedulers: StepLR, ExponentialLR, CosineAnnealingLR\n" << std::flush;
    std::cout << "  Losses: MSELoss, SmoothL1Loss, L1Loss\n" << std::flush;

    try {
        train_with_adam_steplr(device);
        train_with_sgd_exponential(device);
        train_with_adamw_cosine(device);
        demo_additional_optimizers(device);
        demo_gru_cell(device);

        std::cout << "\n============================================================\n" << std::flush;
        std::cout << "   All GRU examples completed successfully!\n" << std::flush;
        std::cout << "============================================================\n" << std::flush;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    finalize();
    return 0;
}
