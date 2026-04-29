/**
 * @file autograd_runner.cpp
 * @brief Implementation of the binary-classification autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

namespace tenzor::examples::showcase03 {

int run_binary_classification_training(int epochs,
                                       double* out_initial,
                                       double* out_final,
                                       ::tenzor::Device device,
                                       bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    int samples_per_class = 50;
    int num_samples = samples_per_class * 2;

    auto class0_x = randn({samples_per_class, 2}, DType::Float32, device) + (-2.0f);
    auto class1_x = randn({samples_per_class, 2}, DType::Float32, device) + 2.0f;
    auto X_tensor = cat({class0_x, class1_x}, 0);

    auto labels0 = zeros({samples_per_class, 1}, DType::Float32, device);
    auto labels1 = ones({samples_per_class, 1}, DType::Float32, device);
    auto y_tensor = cat({labels0, labels1}, 0);

    Variable W(randn({2, 1}, DType::Float32, device) * 0.1f, true);
    Variable b(zeros({1, 1}, DType::Float32, device), true);

    float learning_rate = 0.1f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable X(X_tensor, false);
        Variable y(y_tensor, false);

        auto z = matmul(X, W) + b;
        auto p = nn::sigmoid(z);

        std::vector<int64_t> p_shape(p.shape().begin(), p.shape().end());
        auto eps_tensor = full(p_shape, 1e-7f, DType::Float32, device);
        Variable eps(eps_tensor, false);

        auto log_p = log(p + eps);
        auto log_1_minus_p = log(Variable(ones_like(p.tensor()), false) - p + eps);

        auto loss_per_sample = y * log_p + (Variable(ones_like(y.tensor()), false) - y) * log_1_minus_p;
        auto loss = mean(loss_per_sample) * (-1.0f);

        W.zero_grad(); b.zero_grad();
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
    (void)num_samples;
    return 0;
}

}  // namespace tenzor::examples::showcase03
