/**
 * @file autograd_runner.cpp
 * @brief Implementation of the multitask-learning autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace tenzor::examples::showcase20 {

int run_multitask_training(int epochs,
                           double* out_initial,
                           double* out_final,
                           ::tenzor::Device device,
                           bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    int N = 128, in_dim = 4, hidden = 16;

    std::vector<float> X_data(N * in_dim);
    std::vector<float> yA_data(N), yB_data(N);
    for (int i = 0; i < N; ++i) {
        float sum = 0;
        for (int d = 0; d < in_dim; ++d) {
            float v = dist(gen);
            X_data[i * in_dim + d] = v;
            sum += v;
        }
        yA_data[i] = X_data[i * in_dim + 0] > 0 ? 1.0f : 0.0f;
        yB_data[i] = sum;
    }
    auto X_t  = from_data(X_data.data(), {N, in_dim}, device);
    auto yA_t = from_data(yA_data.data(),{N, 1},      device);
    auto yB_t = from_data(yB_data.data(),{N, 1},      device);

    auto he  = [&](int64_t fin) { return std::sqrt(2.0f / fin); };
    auto xav = [&](int64_t fin) { return std::sqrt(1.0f / fin); };

    Variable W1(randn({in_dim, hidden}, DType::Float32, device) * he(in_dim), true);
    Variable b1(zeros({1, hidden}, DType::Float32, device), true);
    Variable WA(randn({hidden, 1}, DType::Float32, device) * xav(hidden), true);
    Variable bA(zeros({1, 1}, DType::Float32, device), true);
    Variable WB(randn({hidden, 1}, DType::Float32, device) * xav(hidden), true);
    Variable bB(zeros({1, 1}, DType::Float32, device), true);

    float lr = 0.03f;
    int print_every = std::max(1, epochs / 10);

    auto bce_with_logits = [&](const Variable& logit, const Variable& tgt) {
        auto softplus = ::tenzor::log(::tenzor::exp(logit) + 1.0f);
        return mean(softplus - tgt * logit);
    };

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable X(X_t, false), yA(yA_t, false), yB(yB_t, false);
        auto feat = nn::relu(matmul(X, W1) + b1);

        auto logitA = matmul(feat, WA) + bA;
        auto predB  = matmul(feat, WB) + bB;

        auto lossA = bce_with_logits(logitA, yA);
        auto diff = predB - yB;
        auto lossB = mean(diff * diff);
        auto loss = lossA + lossB;

        W1.zero_grad(); b1.zero_grad();
        WA.zero_grad(); bA.zero_grad();
        WB.zero_grad(); bB.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W1 = Variable(W1.tensor() - (*W1.grad() * lr), true);
            b1 = Variable(b1.tensor() - (*b1.grad() * lr), true);
            WA = Variable(WA.tensor() - (*WA.grad() * lr), true);
            bA = Variable(bA.tensor() - (*bA.grad() * lr), true);
            WB = Variable(WB.tensor() - (*WB.grad() * lr), true);
            bB = Variable(bB.tensor() - (*bB.grad() * lr), true);
        }

        double loss_val = static_cast<double>(loss.tensor().item<float>());
        if (epoch == 0 && out_initial) *out_initial = loss_val;
        final_loss = loss_val;

        if (verbose && ((epoch + 1) % print_every == 0 || epoch == 0)) {
            std::cout << "Epoch " << (epoch+1)
                      << "  total=" << loss_val
                      << "  A=" << lossA.tensor().item<float>()
                      << "  B=" << lossB.tensor().item<float>() << "\n";
        }
    }
    if (out_final) *out_final = final_loss;
    return 0;
}

}  // namespace tenzor::examples::showcase20
