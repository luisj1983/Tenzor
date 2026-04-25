/**
 * @file autograd.cpp
 * @brief Multi-task learning with autograd
 *
 * Shared backbone + two task heads. The combined loss is summed:
 *     loss = BCE(task A) + MSE(task B)
 * Autograd handles gradient accumulation into the backbone from both heads.
 *
 * Usage: ./20_multitask_learning_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Multi-Task Learning - Autograd", device);
    manual_seed(42);

    int N = 128, in_dim = 4, hidden = 16;

    std::vector<float> X_data(N * in_dim);
    std::vector<float> yA_data(N), yB_data(N);
    for (int i = 0; i < N; ++i) {
        float sum = 0;
        for (int d = 0; d < in_dim; ++d) {
            float v = ((rand() % 2000) / 1000.0f) - 1.0f;
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
    int num_epochs = 300;
    int print_every = 30;

    auto bce_with_logits = [&](const Variable& logit, const Variable& tgt) {
        auto softplus = tenzor::log(tenzor::exp(logit) + 1.0f);
        return mean(softplus - tgt * logit);
    };

    showcase::print_section("Training");
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
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

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            auto prob = ones_like(logitA.tensor())
                      / (tenzor::exp(logitA.tensor() * -1.0f) + 1.0f);
            auto p_cpu = prob.cpu(); auto y_cpu = yA_t.cpu();
            int c = 0;
            for (int i = 0; i < N; ++i)
                if ((p_cpu.data<float>()[i] > 0.5f) == (y_cpu.data<float>()[i] > 0.5f)) c++;
            std::cout << "Epoch " << (epoch+1)
                      << "  LossA=" << lossA.tensor().item<float>()
                      << "  AccA=" << (100.0f * c / N) << "%"
                      << "  LossB(MSE)=" << lossB.tensor().item<float>() << "\n";
        }
    }

    std::cout << "\nMulti-task learning demonstrated with autograd!\n";
    std::cout << "loss = lossA + lossB; one backward() updates both heads and the shared backbone.\n";

    finalize();
    return 0;
}
