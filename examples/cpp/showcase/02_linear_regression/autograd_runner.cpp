/**
 * @file autograd_runner.cpp
 * @brief Implementation of the linear-regression autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

namespace tenzor::examples::showcase02 {

int run_linear_regression_training(int epochs,
                                   double* out_initial,
                                   double* out_final,
                                   ::tenzor::Device device,
                                   bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    int num_samples = 100;
    float true_weight = 2.0f;
    float true_bias = 1.0f;

    auto X_tensor = rand({num_samples, 1}, DType::Float32, device) * 10.0f;
    auto noise = randn({num_samples, 1}, DType::Float32, device) * 0.5f;
    auto y_tensor = X_tensor * true_weight + true_bias + noise;

    Variable W(randn({1, 1}, DType::Float32, device), true);
    Variable b(zeros({1, 1}, DType::Float32, device), true);

    float learning_rate = 0.01f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable X(X_tensor, false);
        Variable y(y_tensor, false);

        auto y_pred = matmul(X, W) + b;
        auto error = y_pred - y;
        auto loss = mean(error * error);

        W.zero_grad();
        b.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W = Variable(W.tensor() - (*W.grad() * learning_rate), true);
            b = Variable(b.tensor() - (*b.grad() * learning_rate), true);
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

}  // namespace tenzor::examples::showcase02
