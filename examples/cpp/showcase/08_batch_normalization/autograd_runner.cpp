/**
 * @file autograd_runner.cpp
 * @brief Implementation of the batch-normalization autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <random>
#include <vector>

namespace tenzor::examples::showcase08 {

namespace {

::tenzor::Variable batch_norm_var(const ::tenzor::Variable& x,
                                  const ::tenzor::Variable& gamma,
                                  const ::tenzor::Variable& beta,
                                  float eps = 1e-5f) {
    using namespace ::tenzor;
    auto mean_val = mean(x, 0, true);
    auto x_centered = x - mean_val;
    auto var_val = mean(x_centered * x_centered, 0, true);

    auto std_val = Variable(::tenzor::sqrt(var_val.tensor() + eps),
                            var_val.requires_grad());
    auto x_norm = x_centered / std_val;
    return x_norm * gamma + beta;
}

}  // namespace

int run_batchnorm_training(int epochs,
                           double* out_initial,
                           double* out_final,
                           ::tenzor::Device device,
                           bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    int batch_size = 32;
    int input_features = 8;
    int num_classes = 3;

    std::mt19937 gen(123);
    std::normal_distribution<float> ndist(0.0f, 1.0f);

    std::vector<float> X_data(batch_size * input_features);
    for (int b = 0; b < batch_size; ++b) {
        for (int f = 0; f < input_features; ++f) {
            float feat_mean = f * 2.0f;
            float feat_std = 0.5f + f * 0.2f;
            X_data[b * input_features + f] = feat_mean + ndist(gen) * feat_std;
        }
    }
    auto X_tensor = from_data(X_data.data(), {batch_size, input_features}, device);

    std::vector<int64_t> y_data(batch_size);
    for (int i = 0; i < batch_size; ++i) y_data[i] = i % num_classes;
    auto y_tensor = from_data(y_data.data(), {batch_size}, device);

    Variable W1(randn({input_features, 16}, DType::Float32, device) * 0.1f, true);
    Variable b1(zeros({1, 16}, DType::Float32, device), true);
    Variable gamma1(ones({1, 16}, DType::Float32, device), true);
    Variable beta1(zeros({1, 16}, DType::Float32, device), true);
    Variable W2(randn({16, num_classes}, DType::Float32, device) * 0.1f, true);
    Variable b2(zeros({1, num_classes}, DType::Float32, device), true);

    std::vector<float> one_hot_data(batch_size * num_classes, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        one_hot_data[b * num_classes + y_data[b]] = 1.0f;
    }
    auto one_hot_tensor = from_data(one_hot_data.data(), {batch_size, num_classes}, device);

    float learning_rate = 0.1f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable X(X_tensor, false);

        auto z1 = matmul(X, W1) + b1;
        auto z1_norm = batch_norm_var(z1, gamma1, beta1);
        auto a1 = nn::relu(z1_norm);
        auto logits = matmul(a1, W2) + b2;

        auto log_probs = log_softmax(logits, 1);
        Variable one_hot(one_hot_tensor, false);
        auto loss = mean(sum(one_hot * log_probs, 1)) * (-1.0f);

        W1.zero_grad(); b1.zero_grad();
        gamma1.zero_grad(); beta1.zero_grad();
        W2.zero_grad(); b2.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W1 = Variable(W1.tensor() - (*W1.grad() * learning_rate), true);
            b1 = Variable(b1.tensor() - (*b1.grad() * learning_rate), true);
            gamma1 = Variable(gamma1.tensor() - (*gamma1.grad() * learning_rate), true);
            beta1  = Variable(beta1.tensor()  - (*beta1.grad()  * learning_rate), true);
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

}  // namespace tenzor::examples::showcase08
