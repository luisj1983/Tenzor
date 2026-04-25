/**
 * @file tensor_only.cpp
 * @brief Multi-task learning with a shared backbone, raw tensors
 *
 * One backbone feeds two task heads:
 *   Head A: binary classification (x[0] > 0)
 *   Head B: regression (predict sum of features)
 * Combined loss = BCE_A + MSE_B, updated jointly.
 *
 * Usage: ./20_multitask_learning_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

static Tensor relu_t(const Tensor& x) { return maximum(x, zeros_like(x)); }
static Tensor relu_d(const Tensor& z) { return (z > zeros_like(z)).to(DType::Float32); }
static Tensor sigmoid_t(const Tensor& x) { return ones_like(x) / (tenzor::exp(x * -1.0f) + 1.0f); }

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Multi-Task Learning - Tensor Only", device);
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
    auto X  = from_data(X_data.data(), {N, in_dim}, device);
    auto yA = from_data(yA_data.data(),{N, 1},      device);
    auto yB = from_data(yB_data.data(),{N, 1},      device);

    // Shared backbone
    auto W1 = randn({in_dim, hidden}, DType::Float32, device) * std::sqrt(2.0f / in_dim);
    auto b1 = zeros({1, hidden}, DType::Float32, device);
    // Head A: classification
    auto WA = randn({hidden, 1}, DType::Float32, device) * std::sqrt(1.0f / hidden);
    auto bA = zeros({1, 1}, DType::Float32, device);
    // Head B: regression
    auto WB = randn({hidden, 1}, DType::Float32, device) * std::sqrt(1.0f / hidden);
    auto bB = zeros({1, 1}, DType::Float32, device);

    float lr = 0.03f;
    int num_epochs = 300;
    int print_every = 30;

    showcase::print_section("Training");
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Forward: backbone
        auto z1 = matmul(X, W1) + b1;
        auto a1 = relu_t(z1);

        // Head A
        auto logitA = matmul(a1, WA) + bA;
        auto probA = sigmoid_t(logitA);

        // Head B
        auto predB = matmul(a1, WB) + bB;

        // Losses (taken separately for logging)
        float nA = static_cast<float>(N);
        auto errB = predB - yB;
        float lossA = tenzor::mean((yA * -1.0f) * tenzor::log(probA + 1e-8f)
                    + (ones_like(yA) - yA) * (tenzor::log(ones_like(probA) - probA + 1e-8f) * -1.0f)).item<float>();
        float lossB = tenzor::mean(errB * errB).item<float>();

        // Backward - combined grad on shared backbone
        auto dlogitA = (probA - yA) * (1.0f / nA);
        auto dWA = matmul(a1.transpose(0, 1), dlogitA);
        auto dbA_ = tenzor::sum(dlogitA, 0, true);
        auto da1_fromA = matmul(dlogitA, WA.transpose(0, 1));

        auto dpredB = errB * (2.0f / (nA * 1.0f));
        auto dWB = matmul(a1.transpose(0, 1), dpredB);
        auto dbB_ = tenzor::sum(dpredB, 0, true);
        auto da1_fromB = matmul(dpredB, WB.transpose(0, 1));

        auto da1 = da1_fromA + da1_fromB;                // sum grads from both heads
        auto dz1 = da1 * relu_d(z1);
        auto dW1 = matmul(X.transpose(0, 1), dz1);
        auto db1 = tenzor::sum(dz1, 0, true);

        W1 = W1 - dW1 * lr;  b1 = b1 - db1 * lr;
        WA = WA - dWA * lr;  bA = bA - dbA_ * lr;
        WB = WB - dWB * lr;  bB = bB - dbB_ * lr;

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            auto pA_cpu = probA.cpu(); auto yA_cpu = yA.cpu();
            int c = 0;
            for (int i = 0; i < N; ++i)
                if ((pA_cpu.data<float>()[i] > 0.5f) == (yA_cpu.data<float>()[i] > 0.5f)) c++;
            std::cout << "Epoch " << (epoch+1) << "  LossA=" << lossA
                      << "  AccA=" << (100.0f * c / N) << "%"
                      << "  LossB(MSE)=" << lossB << "\n";
        }
    }

    std::cout << "\nMulti-task learning demonstrated with raw tensors!\n";
    std::cout << "Two heads share one backbone - gradients from both losses add into the backbone.\n";

    finalize();
    return 0;
}
