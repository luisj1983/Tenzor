/**
 * @file autograd.cpp
 * @brief RNN/Sequence model using Tenzor's automatic differentiation
 *
 * This example demonstrates an RNN using Variable and autograd
 * for automatic backpropagation through time.
 *
 * Usage: ./07_rnn_sequence_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("RNN Sequence - Autograd (Automatic BPTT)", device);

    manual_seed(42);

    // Generate sine wave data
    int num_samples = 100;
    int seq_length = 10;
    int input_size = 1;
    int hidden_size = 16;

    std::vector<float> X_data(num_samples * seq_length);
    std::vector<float> y_data(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        float phase = static_cast<float>(i) * 0.1f;
        for (int t = 0; t < seq_length; ++t) {
            X_data[i * seq_length + t] = std::sin(phase + t * 0.1f);
        }
        y_data[i] = std::sin(phase + seq_length * 0.1f);
    }

    auto X_tensor = from_data(X_data.data(), {num_samples, seq_length, input_size}, device);
    auto y_tensor = from_data(y_data.data(), {num_samples, 1}, device);

    showcase::print_tensor_info("Input sequences X", X_tensor);
    showcase::print_tensor_info("Target y", y_tensor);

    // Initialize RNN weights as Variables
    Variable W_ih(randn({input_size, hidden_size}, DType::Float32, device) * 0.1f, true);
    Variable W_hh(randn({hidden_size, hidden_size}, DType::Float32, device) * 0.1f, true);
    Variable b_h(zeros({1, hidden_size}, DType::Float32, device), true);

    Variable W_ho(randn({hidden_size, 1}, DType::Float32, device) * 0.1f, true);
    Variable b_o(zeros({1, 1}, DType::Float32, device), true);

    // Training parameters
    float learning_rate = 0.01f;
    int num_epochs = 15000;
    int print_every = 50;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Initialize hidden state
        Variable h(zeros({num_samples, hidden_size}, DType::Float32, device), false);

        // ============ Forward Pass through time ============
        for (int t = 0; t < seq_length; ++t) {
            // Get input at timestep t
            auto x_t_tensor = X_tensor.slice(1, t, t + 1).reshape({num_samples, input_size});
            Variable x_t(x_t_tensor, false);

            // h = tanh(x_t @ W_ih + h @ W_hh + b_h)
            h = nn::tanh(matmul(x_t, W_ih) + matmul(h, W_hh) + b_h);
        }

        // Output from final hidden state
        auto y_pred = matmul(h, W_ho) + b_o;

        // ============ Compute Loss (MSE) ============
        Variable y_target(y_tensor, false);
        auto error = y_pred - y_target;
        auto loss = mean(error * error);

        // ============ Backward Pass (Automatic BPTT!) ============
        W_ih.zero_grad();
        W_hh.zero_grad();
        b_h.zero_grad();
        W_ho.zero_grad();
        b_o.zero_grad();

        loss.backward();

        // ============ Update Weights ============
        {
            NoGradGuard no_grad;

            W_ih = Variable(W_ih.tensor() - (*W_ih.grad() * learning_rate), true);
            W_hh = Variable(W_hh.tensor() - (*W_hh.grad() * learning_rate), true);
            b_h = Variable(b_h.tensor() - (*b_h.grad() * learning_rate), true);
            W_ho = Variable(W_ho.tensor() - (*W_ho.grad() * learning_rate), true);
            b_o = Variable(b_o.tensor() - (*b_o.grad() * learning_rate), true);
        }

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                      << "MSE Loss: " << loss_val << "\n";
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    // Final predictions
    Variable h_final(zeros({num_samples, hidden_size}, DType::Float32, device), false);
    for (int t = 0; t < seq_length; ++t) {
        auto x_t_tensor = X_tensor.slice(1, t, t + 1).reshape({num_samples, input_size});
        Variable x_t(x_t_tensor, false);
        h_final = nn::tanh(matmul(x_t, W_ih) + matmul(h_final, W_hh) + b_h);
    }
    auto predictions = matmul(h_final, W_ho) + b_o;

    auto pred_cpu = predictions.tensor().cpu();
    auto target_cpu = y_tensor.cpu();

    std::cout << "Sample predictions:\n";
    for (int i = 0; i < 5; ++i) {
        float target = target_cpu.data<float>()[i];
        float pred = pred_cpu.data<float>()[i];
        std::cout << "Target: " << target << ", Predicted: " << pred << "\n";
    }

    std::cout << "\nRNN demonstrated with autograd!\n";
    std::cout << "BPTT computed automatically through the computation graph.\n";

    finalize();
    return 0;
}
