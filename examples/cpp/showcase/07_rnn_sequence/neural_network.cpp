/**
 * @file neural_network.cpp
 * @brief RNN/Sequence model using Tenzor's high-level Neural Network API
 *
 * This example demonstrates sequence prediction using nn::RNN module
 * for a clean, high-level interface.
 *
 * Usage: ./07_rnn_sequence_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>

using namespace tenzor;

/**
 * @brief Sequence predictor using RNN
 */
class SequencePredictor : public nn::Module {
public:
    SequencePredictor(int64_t input_size, int64_t hidden_size, int64_t output_size) {
        rnn = std::make_shared<nn::RNN>(input_size, hidden_size);
        fc = std::make_shared<nn::Linear>(hidden_size, output_size);

        register_module("rnn", rnn);
        register_module("fc", fc);

        hidden_size_ = hidden_size;
    }

    auto forward_impl(const Variable& input) -> Variable override {
        // input shape: (batch, seq_len, input_size)
        auto rnn_out = rnn->forward(input);

        // Take final timestep: rnn_out[:, -1, :] -> (batch, hidden_size)
        // Using Variable-level slice/squeeze keeps the autograd graph intact.
        auto seq_len = input.shape()[1];
        auto last_step = slice(rnn_out, 1, seq_len - 1, seq_len);  // (batch, 1, hidden)
        auto last_h = squeeze(last_step, 1);                        // (batch, hidden)

        return fc->forward(last_h);
    }

private:
    std::shared_ptr<nn::RNN> rnn;
    std::shared_ptr<nn::Linear> fc;
    int64_t hidden_size_;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("RNN Sequence - Neural Network API (High-Level)", device);

    manual_seed(42);

    // Generate sine wave data
    int num_samples = 100;
    int seq_length = 10;
    int input_size = 1;
    int hidden_size = 32;

    std::vector<float> X_data(num_samples * seq_length);
    std::vector<float> y_data(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        float phase = static_cast<float>(i) * 0.1f;
        for (int t = 0; t < seq_length; ++t) {
            X_data[i * seq_length + t] = std::sin(phase + t * 0.1f);
        }
        y_data[i] = std::sin(phase + seq_length * 0.1f);
    }

    auto X = from_data(X_data.data(), {num_samples, seq_length, input_size}, device);
    auto y = from_data(y_data.data(), {num_samples, 1}, device);

    showcase::print_tensor_info("Input sequences X", X);
    showcase::print_tensor_info("Target y", y);

    // Create model
    auto model = std::make_shared<SequencePredictor>(input_size, hidden_size, 1);
    model->to(device);

    // Create optimizer
    auto params = model->parameters();
    optim::Adam optimizer(params, 0.01f);

    // Create loss function
    nn::MSELoss criterion;

    showcase::print_section("Model Architecture");
    std::cout << "SequencePredictor:\n";
    std::cout << "  RNN: input=" << input_size << ", hidden=" << hidden_size << "\n";
    std::cout << "  FC: Linear(" << hidden_size << ", 1)\n";
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
        auto y_pred = model->forward(input);

        // Compute loss
        Variable target(y, false);
        auto loss = criterion(y_pred, target);

        // Backward pass
        loss.backward();

        // Update weights
        optimizer.step();

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                      << "MSE Loss: " << loss_val << "\n";
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    model->eval();

    Variable X_eval(X, false);
    auto predictions = model->forward(X_eval);

    Variable target(y, false);
    auto final_loss = criterion(predictions, target);
    std::cout << "Final MSE Loss: " << final_loss.tensor().item<float>() << "\n\n";

    // Show predictions
    auto pred_cpu = predictions.tensor().cpu();
    auto target_cpu = y.cpu();

    std::cout << "Sample predictions:\n";
    std::cout << "Target\t\tPredicted\tError\n";
    std::cout << "-----------------------------------\n";
    for (int i = 0; i < 10; ++i) {
        float tgt = target_cpu.data<float>()[i];
        float pred = pred_cpu.data<float>()[i];
        float err = std::abs(tgt - pred);
        std::cout << tgt << "\t\t" << pred << "\t\t" << err << "\n";
    }

    std::cout << "\nRNN solved using Neural Network API!\n";
    std::cout << "Uses nn::RNN for clean sequence processing.\n";

    finalize();
    return 0;
}
