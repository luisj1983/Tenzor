/**
 * @file tensor_only.cpp
 * @brief Transfer learning concept with raw tensors
 *
 * Two-stage training on synthetic data:
 *   Stage 1: train a small MLP on task A (classify +x vs -x with one extra feature)
 *   Stage 2: freeze the backbone, retrain only the output head for task B
 *           (classify |x| small vs large)
 *
 * Demonstrates the core idea of transfer learning: reuse a shared
 * representation for a new downstream task by freezing feature layers.
 *
 * Usage: ./18_transfer_learning_tensor_only --backend cpu|cuda|vulkan
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
    showcase::print_header("Transfer Learning - Tensor Only", device);
    manual_seed(42);

    int N = 128;
    int in_dim = 4;
    int hidden = 16;

    // Task A labels: x[0] > 0
    std::vector<float> X_data(N * in_dim);
    std::vector<float> yA_data(N);
    std::vector<float> yB_data(N);
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < in_dim; ++d) {
            X_data[i * in_dim + d] = ((rand() % 2000) / 1000.0f) - 1.0f;
        }
        yA_data[i] = X_data[i * in_dim + 0] > 0 ? 1.0f : 0.0f;
        yB_data[i] = std::abs(X_data[i * in_dim + 0]) > 0.5f ? 1.0f : 0.0f;
    }
    auto X  = from_data(X_data.data(),  {N, in_dim}, device);
    auto yA = from_data(yA_data.data(), {N, 1},      device);
    auto yB = from_data(yB_data.data(), {N, 1},      device);

    // Backbone: in_dim -> hidden -> hidden  (shared for both tasks)
    auto Wb1 = randn({in_dim, hidden}, DType::Float32, device) * std::sqrt(2.0f / in_dim);
    auto bb1 = zeros({1, hidden},      DType::Float32, device);
    auto Wb2 = randn({hidden, hidden}, DType::Float32, device) * std::sqrt(2.0f / hidden);
    auto bb2 = zeros({1, hidden},      DType::Float32, device);

    auto forward_backbone = [&](const Tensor& x) {
        auto z1 = matmul(x, Wb1) + bb1; auto a1 = relu_t(z1);
        auto z2 = matmul(a1, Wb2) + bb2; auto a2 = relu_t(z2);
        return std::make_tuple(z1, a1, z2, a2);
    };

    // Head A: hidden -> 1
    auto WA = randn({hidden, 1}, DType::Float32, device) * std::sqrt(1.0f / hidden);
    auto bA = zeros({1, 1}, DType::Float32, device);

    float lr = 0.05f;

    // ===== Stage 1: train backbone + head A on task A =====
    showcase::print_section("Stage 1: pretrain backbone on task A");
    for (int epoch = 0; epoch < 300; ++epoch) {
        auto [z1, a1, z2, a2] = forward_backbone(X);
        auto logit = matmul(a2, WA) + bA;
        auto prob = sigmoid_t(logit);

        // BCE gradient at logit
        float n = static_cast<float>(N);
        auto dlogit = (prob - yA) * (1.0f / n);

        auto dWA = matmul(a2.transpose(0, 1), dlogit);
        auto dbA = tenzor::sum(dlogit, 0, true);
        auto da2 = matmul(dlogit, WA.transpose(0, 1)) * relu_d(z2);
        auto dWb2 = matmul(a1.transpose(0, 1), da2);
        auto dbb2 = tenzor::sum(da2, 0, true);
        auto da1 = matmul(da2, Wb2.transpose(0, 1)) * relu_d(z1);
        auto dWb1 = matmul(X.transpose(0, 1), da1);
        auto dbb1 = tenzor::sum(da1, 0, true);

        WA  = WA  - dWA  * lr; bA  = bA  - dbA  * lr;
        Wb2 = Wb2 - dWb2 * lr; bb2 = bb2 - dbb2 * lr;
        Wb1 = Wb1 - dWb1 * lr; bb1 = bb1 - dbb1 * lr;

        if ((epoch + 1) % 50 == 0 || epoch == 0) {
            int correct = 0;
            auto p_cpu = prob.cpu();
            auto y_cpu = yA.cpu();
            for (int i = 0; i < N; ++i)
                if ((p_cpu.data<float>()[i] > 0.5f) == (y_cpu.data<float>()[i] > 0.5f)) correct++;
            std::cout << "[TaskA] epoch " << (epoch + 1) << "  acc=" << (100.0f * correct / N) << "%\n";
        }
    }

    // ===== Stage 2: freeze backbone, train only head B =====
    showcase::print_section("Stage 2: freeze backbone, fine-tune head B on task B");
    auto WB = randn({hidden, 1}, DType::Float32, device) * std::sqrt(1.0f / hidden);
    auto bB = zeros({1, 1}, DType::Float32, device);

    for (int epoch = 0; epoch < 300; ++epoch) {
        // Backbone forward - but we won't compute its gradients below
        auto [z1, a1, z2, a2] = forward_backbone(X);
        auto logit = matmul(a2, WB) + bB;
        auto prob = sigmoid_t(logit);

        float n = static_cast<float>(N);
        auto dlogit = (prob - yB) * (1.0f / n);
        auto dWB = matmul(a2.transpose(0, 1), dlogit);
        auto dbB = tenzor::sum(dlogit, 0, true);
        // Intentionally skip backbone grad -> "frozen"

        WB = WB - dWB * lr; bB = bB - dbB * lr;

        if ((epoch + 1) % 50 == 0 || epoch == 0) {
            int correct = 0;
            auto p_cpu = prob.cpu(); auto y_cpu = yB.cpu();
            for (int i = 0; i < N; ++i)
                if ((p_cpu.data<float>()[i] > 0.5f) == (y_cpu.data<float>()[i] > 0.5f)) correct++;
            std::cout << "[TaskB] epoch " << (epoch + 1) << "  acc=" << (100.0f * correct / N) << "%\n";
        }
    }

    std::cout << "\nTransfer learning demonstrated with raw tensors!\n";
    std::cout << "Stage 1 trains the backbone; Stage 2 freezes it and retrains just the head.\n";

    finalize();
    return 0;
}
