/**
 * @file autograd_runner.cpp
 * @brief Implementation of the transfer-learning autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace tenzor::examples::showcase18 {

int run_transfer_learning_training(int epochs_a,
                                   int epochs_b,
                                   double* out_initial,
                                   double* out_final,
                                   ::tenzor::Device device,
                                   bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    int N = 128;
    int in_dim = 4;
    int hidden = 16;

    std::vector<float> X_data(N * in_dim);
    std::vector<float> yA_data(N), yB_data(N);
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < in_dim; ++d) {
            X_data[i * in_dim + d] = dist(gen);
        }
        yA_data[i] = X_data[i * in_dim + 0] > 0 ? 1.0f : 0.0f;
        yB_data[i] = std::abs(X_data[i * in_dim + 0]) > 0.5f ? 1.0f : 0.0f;
    }
    auto X_t  = from_data(X_data.data(), {N, in_dim}, device);
    auto yA_t = from_data(yA_data.data(),{N, 1}, device);
    auto yB_t = from_data(yB_data.data(),{N, 1}, device);

    auto he  = [&](int64_t fin) { return std::sqrt(2.0f / fin); };
    auto xav = [&](int64_t fin) { return std::sqrt(1.0f / fin); };

    Variable Wb1(randn({in_dim, hidden}, DType::Float32, device) * he(in_dim), true);
    Variable bb1(zeros({1, hidden}, DType::Float32, device), true);
    Variable Wb2(randn({hidden, hidden}, DType::Float32, device) * he(hidden), true);
    Variable bb2(zeros({1, hidden}, DType::Float32, device), true);

    Variable WA(randn({hidden, 1}, DType::Float32, device) * xav(hidden), true);
    Variable bA(zeros({1, 1}, DType::Float32, device), true);

    auto backbone = [&](const Variable& x) {
        auto a1 = nn::relu(matmul(x, Wb1) + bb1);
        return nn::relu(matmul(a1, Wb2) + bb2);
    };

    auto bce_with_logits = [&](const Variable& logit, const Variable& tgt) {
        auto softplus = ::tenzor::log(::tenzor::exp(logit) + 1.0f);
        return mean(softplus - tgt * logit);
    };

    float lr = 0.05f;
    int print_every_a = std::max(1, epochs_a / 10);

    double final_loss_a = 0.0;
    for (int epoch = 0; epoch < epochs_a; ++epoch) {
        Variable x(X_t, false), y(yA_t, false);
        auto feat = backbone(x);
        auto logit = matmul(feat, WA) + bA;
        auto loss = bce_with_logits(logit, y);

        Wb1.zero_grad(); bb1.zero_grad();
        Wb2.zero_grad(); bb2.zero_grad();
        WA.zero_grad();  bA.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            Wb1 = Variable(Wb1.tensor() - (*Wb1.grad() * lr), true);
            bb1 = Variable(bb1.tensor() - (*bb1.grad() * lr), true);
            Wb2 = Variable(Wb2.tensor() - (*Wb2.grad() * lr), true);
            bb2 = Variable(bb2.tensor() - (*bb2.grad() * lr), true);
            WA  = Variable(WA.tensor()  - (*WA.grad()  * lr), true);
            bA  = Variable(bA.tensor()  - (*bA.grad()  * lr), true);
        }

        double loss_val = static_cast<double>(loss.tensor().item<float>());
        if (epoch == 0 && out_initial) *out_initial = loss_val;
        final_loss_a = loss_val;

        if (verbose && ((epoch + 1) % print_every_a == 0 || epoch == 0)) {
            std::cout << "[TaskA] epoch " << (epoch+1) << "  loss=" << loss_val << "\n";
        }
    }
    if (out_final) *out_final = final_loss_a;

    // Stage B: freeze backbone, fine-tune head B.
    Variable WB(randn({hidden, 1}, DType::Float32, device) * xav(hidden), true);
    Variable bB(zeros({1, 1}, DType::Float32, device), true);

    int print_every_b = std::max(1, epochs_b / 10);
    for (int epoch = 0; epoch < epochs_b; ++epoch) {
        Variable x(X_t, false), y(yB_t, false);
        auto feat = backbone(x);
        auto logit = matmul(feat, WB) + bB;
        auto loss = bce_with_logits(logit, y);

        WB.zero_grad(); bB.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            WB = Variable(WB.tensor() - (*WB.grad() * lr), true);
            bB = Variable(bB.tensor() - (*bB.grad() * lr), true);
        }

        if (verbose && ((epoch + 1) % print_every_b == 0 || epoch == 0)) {
            std::cout << "[TaskB] epoch " << (epoch+1) << "  loss="
                      << loss.tensor().item<float>() << "\n";
        }
    }
    return 0;
}

}  // namespace tenzor::examples::showcase18
