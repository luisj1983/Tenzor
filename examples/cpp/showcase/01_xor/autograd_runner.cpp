/**
 * @file autograd_runner.cpp
 * @brief Implementation of the XOR-autograd showcase training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

namespace tenzor::examples::showcase01 {

int run_xor_training(int epochs,
                     double* out_initial,
                     double* out_final,
                     ::tenzor::Device device,
                     bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    float input_data[]  = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    float target_data[] = {0.0f, 1.0f, 1.0f, 0.0f};

    auto X_tensor = from_data(input_data, {4, 2}, device);
    auto y_tensor = from_data(target_data, {4, 1}, device);

    Variable W1(randn({2, 4}, DType::Float32, device) * 0.5f, true);
    Variable b1(zeros({1, 4}, DType::Float32, device), true);
    Variable W2(randn({4, 1}, DType::Float32, device) * 0.5f, true);
    Variable b2(zeros({1, 1}, DType::Float32, device), true);

    float learning_rate = 1.0f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable X(X_tensor, false);
        Variable y(y_tensor, false);

        auto z1 = X.matmul(W1) + b1;
        auto a1 = nn::sigmoid(z1);
        auto z2 = a1.matmul(W2) + b2;
        auto a2 = nn::sigmoid(z2);

        auto error = a2 - y;
        auto loss = mean(error * error);

        W1.zero_grad(); b1.zero_grad();
        W2.zero_grad(); b2.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W1 = Variable(W1.tensor() - (*W1.grad() * learning_rate), true);
            b1 = Variable(b1.tensor() - (*b1.grad() * learning_rate), true);
            W2 = Variable(W2.tensor() - (*W2.grad() * learning_rate), true);
            b2 = Variable(b2.tensor() - (*b2.grad() * learning_rate), true);
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

}  // namespace tenzor::examples::showcase01
