/**
 * @file tensor_only.cpp
 * @brief RNN/Sequence model using raw Tensor operations only
 *
 * This example demonstrates a simple RNN for sequence prediction
 * using only tensor operations with manual backpropagation through time.
 *
 * We'll predict the next value in a sine wave sequence.
 *
 * Usage: ./07_rnn_sequence_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>

using namespace tenzor;

// Tanh activation
Tensor tanh_tensor(const Tensor& x) {
    auto exp_x = tenzor::exp(x);
    auto exp_neg_x = tenzor::exp(x * -1.0f);
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
}

// Tanh derivative: 1 - tanh(x)^2
Tensor tanh_deriv(const Tensor& tanh_out) {
    return ones_like(tanh_out) - tanh_out * tanh_out;
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("RNN Sequence - Tensor Only (Manual BPTT)", device);

    manual_seed(42);

    // Generate sine wave data
    int num_samples = 100;
    int seq_length = 10;
    int input_size = 1;
    int hidden_size = 16;

    // Create sine wave sequences
    std::vector<float> X_data(num_samples * seq_length);
    std::vector<float> y_data(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        float phase = static_cast<float>(i) * 0.1f;
        for (int t = 0; t < seq_length; ++t) {
            X_data[i * seq_length + t] = std::sin(phase + t * 0.1f);
        }
        // Target: next value in sequence
        y_data[i] = std::sin(phase + seq_length * 0.1f);
    }

    auto X = from_data(X_data.data(), {num_samples, seq_length, input_size}, device);
    auto y = from_data(y_data.data(), {num_samples, 1}, device);

    showcase::print_tensor_info("Input sequences X", X);
    showcase::print_tensor_info("Target y", y);

    // Initialize RNN weights
    // h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
    auto W_ih = randn({input_size, hidden_size}, DType::Float32, device) * 0.1f;  // Input to hidden
    auto W_hh = randn({hidden_size, hidden_size}, DType::Float32, device) * 0.1f; // Hidden to hidden
    auto b_h = zeros({1, hidden_size}, DType::Float32, device);

    // Output layer: y = W_ho * h_T + b_o
    auto W_ho = randn({hidden_size, 1}, DType::Float32, device) * 0.1f;
    auto b_o = zeros({1, 1}, DType::Float32, device);

    showcase::print_section("RNN Architecture");
    std::cout << "Input size: " << input_size << "\n";
    std::cout << "Hidden size: " << hidden_size << "\n";
    std::cout << "Sequence length: " << seq_length << "\n";

    // Training parameters
    float learning_rate = 0.01f;
    int num_epochs = 500;
    int print_every = 50;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ============ Forward Pass through time ============
        // Store all hidden states for BPTT
        std::vector<Tensor> h_states(seq_length + 1);
        h_states[0] = zeros({num_samples, hidden_size}, DType::Float32, device);

        // Process each timestep
        for (int t = 0; t < seq_length; ++t) {
            // Get input at timestep t: X[:, t, :]
            auto x_t = X.slice(1, t, t + 1).reshape({num_samples, input_size});

            // h_t = tanh(x_t @ W_ih + h_{t-1} @ W_hh + b_h)
            auto pre_h = matmul(x_t, W_ih) + matmul(h_states[t], W_hh) + b_h;
            h_states[t + 1] = tanh_tensor(pre_h);
        }

        // Final hidden state -> output
        auto h_final = h_states[seq_length];
        auto y_pred = matmul(h_final, W_ho) + b_o;

        // ============ Compute Loss (MSE) ============
        auto error = y_pred - y;
        auto loss = tenzor::mean(error * error);
        float loss_val = loss.item<float>();

        // ============ Backward Pass (Truncated BPTT) ============
        float n = static_cast<float>(num_samples);

        // Output layer gradients
        auto dL_dy = error * (2.0f / n);
        auto dL_dW_ho = matmul(h_final.transpose(0, 1), dL_dy);
        auto dL_db_o = tenzor::sum(dL_dy, 0, true);

        // Gradient w.r.t. final hidden state
        auto dL_dh = matmul(dL_dy, W_ho.transpose(0, 1));

        // Backprop through time (simplified - only last few steps)
        auto dL_dW_ih = zeros_like(W_ih);
        auto dL_dW_hh = zeros_like(W_hh);
        auto dL_db_h = zeros_like(b_h);

        // Truncated to last 5 steps for efficiency
        int bptt_steps = std::min(5, seq_length);
        for (int t = seq_length - 1; t >= seq_length - bptt_steps && t >= 0; --t) {
            // Gradient through tanh
            auto dL_dpre_h = dL_dh * tanh_deriv(h_states[t + 1]);

            // Get input at timestep t
            auto x_t = X.slice(1, t, t + 1).reshape({num_samples, input_size});

            // Accumulate gradients
            dL_dW_ih = dL_dW_ih + matmul(x_t.transpose(0, 1), dL_dpre_h);
            dL_dW_hh = dL_dW_hh + matmul(h_states[t].transpose(0, 1), dL_dpre_h);
            dL_db_h = dL_db_h + tenzor::sum(dL_dpre_h, 0, true);

            // Propagate gradient to previous hidden state
            dL_dh = matmul(dL_dpre_h, W_hh.transpose(0, 1));
        }

        // ============ Update Weights ============
        W_ih = W_ih - dL_dW_ih * learning_rate;
        W_hh = W_hh - dL_dW_hh * learning_rate;
        b_h = b_h - dL_db_h * learning_rate;
        W_ho = W_ho - dL_dW_ho * learning_rate;
        b_o = b_o - dL_db_o * learning_rate;

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                      << "MSE Loss: " << loss_val << "\n";
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    // Final forward pass
    std::vector<Tensor> h_final_states(seq_length + 1);
    h_final_states[0] = zeros({num_samples, hidden_size}, DType::Float32, device);

    for (int t = 0; t < seq_length; ++t) {
        auto x_t = X.slice(1, t, t + 1).reshape({num_samples, input_size});
        auto pre_h = matmul(x_t, W_ih) + matmul(h_final_states[t], W_hh) + b_h;
        h_final_states[t + 1] = tanh_tensor(pre_h);
    }

    auto predictions = matmul(h_final_states[seq_length], W_ho) + b_o;

    auto pred_cpu = predictions.cpu();
    auto target_cpu = y.cpu();

    std::cout << "Sample predictions vs targets:\n";
    std::cout << "Target\t\tPredicted\tError\n";
    std::cout << "-----------------------------------\n";
    for (int i = 0; i < 10; ++i) {
        float target = target_cpu.data<float>()[i];
        float pred = pred_cpu.data<float>()[i];
        float err = std::abs(target - pred);
        std::cout << target << "\t\t" << pred << "\t\t" << err << "\n";
    }

    std::cout << "\nRNN demonstrated with manual tensor operations!\n";
    std::cout << "Uses truncated BPTT for gradient computation.\n";

    finalize();
    return 0;
}
