/**
 * @file autograd_runner.cpp
 * @brief Implementation of the layer-normalization autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace tenzor::examples::showcase19 {

int run_layernorm_training(int epochs,
                           double* out_initial,
                           double* out_final,
                           ::tenzor::Device device,
                           bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    int N = 128;
    int in_dim = 4;
    int hidden = 16;

    std::vector<float> X_data(N * in_dim);
    std::vector<int64_t> y_data(N);
    for (int i = 0; i < N; ++i) {
        float sign = (i % 2) ? 1.0f : -1.0f;
        y_data[i] = (i % 2);
        for (int d = 0; d < in_dim; ++d) {
            float scale = std::pow(10.0f, static_cast<float>(d));
            X_data[i * in_dim + d] = scale * (sign * 0.3f + (dist(gen) * 0.4f - 0.2f));
        }
    }
    auto X = from_data(X_data.data(), {N, in_dim}, device);
    auto y = from_data(y_data.data(), {N}, device);

    auto he  = [&](int64_t fin) { return std::sqrt(2.0f / fin); };
    auto xav = [&](int64_t fin) { return std::sqrt(1.0f / fin); };

    Variable W1(randn({in_dim, hidden}, DType::Float32, device) * he(in_dim), true);
    Variable b1(zeros({1, hidden}, DType::Float32, device), true);
    Variable gamma(ones({1, hidden}, DType::Float32, device), true);
    Variable beta(zeros({1, hidden}, DType::Float32, device), true);
    Variable W2(randn({hidden, 2}, DType::Float32, device) * xav(hidden), true);
    Variable b2(zeros({1, 2}, DType::Float32, device), true);

    auto layer_norm = [&](const Variable& x) -> Variable {
        auto mu = mean(x, 1, true);
        auto ctr = x - mu;
        auto var = mean(ctr * ctr, 1, true);
        auto xn = ctr / ::tenzor::sqrt(var + 1e-5f);
        return xn * gamma + beta;
    };

    std::vector<float> onehot(N * 2, 0.0f);
    for (int i = 0; i < N; ++i) onehot[i * 2 + y_data[i]] = 1.0f;
    auto onehot_t = from_data(onehot.data(), {N, 2}, device);

    float lr = 0.05f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable x(X, false);
        auto h = matmul(x, W1) + b1;
        auto h_ln = layer_norm(h);
        auto a = nn::relu(h_ln);
        auto logits = matmul(a, W2) + b2;
        auto log_p = log_softmax(logits, 1);

        Variable ohv(onehot_t, false);
        auto loss = mean(sum(ohv * log_p, 1)) * (-1.0f);

        W1.zero_grad(); b1.zero_grad();
        gamma.zero_grad(); beta.zero_grad();
        W2.zero_grad(); b2.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W1 = Variable(W1.tensor() - (*W1.grad() * lr), true);
            b1 = Variable(b1.tensor() - (*b1.grad() * lr), true);
            gamma = Variable(gamma.tensor() - (*gamma.grad() * lr), true);
            beta  = Variable(beta.tensor()  - (*beta.grad()  * lr), true);
            W2 = Variable(W2.tensor() - (*W2.grad() * lr), true);
            b2 = Variable(b2.tensor() - (*b2.grad() * lr), true);
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

}  // namespace tenzor::examples::showcase19
