/**
 * @file autograd_runner.cpp
 * @brief Implementation of the RNN-sequence autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <cmath>
#include <vector>

namespace tenzor::examples::showcase07 {

int run_rnn_training(int epochs,
                     double* out_initial,
                     double* out_final,
                     ::tenzor::Device device,
                     bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

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

    Variable W_ih(randn({input_size, hidden_size}, DType::Float32, device) * 0.1f, true);
    Variable W_hh(randn({hidden_size, hidden_size}, DType::Float32, device) * 0.1f, true);
    Variable b_h(zeros({1, hidden_size}, DType::Float32, device), true);
    Variable W_ho(randn({hidden_size, 1}, DType::Float32, device) * 0.1f, true);
    Variable b_o(zeros({1, 1}, DType::Float32, device), true);

    float learning_rate = 0.01f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable h(zeros({num_samples, hidden_size}, DType::Float32, device), false);

        for (int t = 0; t < seq_length; ++t) {
            auto x_t_tensor = X_tensor.slice(1, t, t + 1).reshape({num_samples, input_size});
            Variable x_t(x_t_tensor, false);
            h = nn::tanh(matmul(x_t, W_ih) + matmul(h, W_hh) + b_h);
        }

        auto y_pred = matmul(h, W_ho) + b_o;

        Variable y_target(y_tensor, false);
        auto error = y_pred - y_target;
        auto loss = mean(error * error);

        W_ih.zero_grad(); W_hh.zero_grad(); b_h.zero_grad();
        W_ho.zero_grad(); b_o.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W_ih = Variable(W_ih.tensor() - (*W_ih.grad() * learning_rate), true);
            W_hh = Variable(W_hh.tensor() - (*W_hh.grad() * learning_rate), true);
            b_h  = Variable(b_h.tensor()  - (*b_h.grad()  * learning_rate), true);
            W_ho = Variable(W_ho.tensor() - (*W_ho.grad() * learning_rate), true);
            b_o  = Variable(b_o.tensor()  - (*b_o.grad()  * learning_rate), true);
        }

        double loss_val = static_cast<double>(loss.tensor().item<float>());
        if (epoch == 0 && out_initial) *out_initial = loss_val;
        final_loss = loss_val;

        if (verbose && ((epoch + 1) % print_every == 0 || epoch == 0)) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << epochs
                      << "] loss=" << loss_val << "\n";
        }
    }
    if (out_final) *out_final = final_loss;
    return 0;
}

}  // namespace tenzor::examples::showcase07
