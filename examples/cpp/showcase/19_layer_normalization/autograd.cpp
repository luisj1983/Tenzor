/**
 * @file autograd.cpp
 * @brief Layer Normalization learned end-to-end with autograd
 *
 * Trains a small MLP with a manual LayerNorm block between the two
 * linear layers. The gamma/beta scale+shift are learnable Variables,
 * and autograd handles backprop through the mean/var reduction.
 *
 * Usage: ./19_layer_normalization_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Layer Normalization - Autograd", device);
    manual_seed(42);

    // 2-class classification with features of wildly different scales
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
            X_data[i * in_dim + d] = scale * (sign * 0.3f + ((rand() % 200) / 500.0f - 0.2f));
        }
    }
    auto X = from_data(X_data.data(), {N, in_dim}, device);
    auto y = from_data(y_data.data(), {N}, device);

    auto he  = [&](int64_t fin) { return std::sqrt(2.0f / fin); };
    auto xav = [&](int64_t fin) { return std::sqrt(1.0f / fin); };

    Variable W1(randn({in_dim, hidden}, DType::Float32, device) * he(in_dim), true);
    Variable b1(zeros({1, hidden},      DType::Float32, device), true);
    Variable gamma(ones({1, hidden},    DType::Float32, device), true);
    Variable beta(zeros({1, hidden},    DType::Float32, device), true);
    Variable W2(randn({hidden, 2},      DType::Float32, device) * xav(hidden), true);
    Variable b2(zeros({1, 2},           DType::Float32, device), true);

    auto layer_norm = [&](const Variable& x) -> Variable {
        // x is (N, hidden) - reduce along dim=1 (the feature axis).
        auto mu = mean(x, 1, true);
        auto ctr = x - mu;
        auto var = mean(ctr * ctr, 1, true);
        auto xn = ctr / tenzor::sqrt(var + 1e-5f);
        return xn * gamma + beta;
    };

    float lr = 0.05f;
    int num_epochs = 300;
    int print_every = 30;

    showcase::print_section("Training");
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable x(X, false), tgt(y, false);
        auto h = matmul(x, W1) + b1;
        auto h_ln = layer_norm(h);
        auto a = nn::relu(h_ln);
        auto logits = matmul(a, W2) + b2;
        auto log_p = log_softmax(logits, 1);

        // One-hot for targets
        std::vector<float> onehot(N * 2, 0.0f);
        for (int i = 0; i < N; ++i) onehot[i * 2 + y_data[i]] = 1.0f;
        Variable ohv(from_data(onehot.data(), {N, 2}, device), false);
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

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float acc = showcase::multiclass_accuracy(logits.tensor(), y);
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] Loss: " << loss.tensor().item<float>()
                      << "  Acc: " << (acc * 100) << "%\n";
        }
    }

    showcase::print_section("Final Results");
    Variable x(X, false);
    auto h = layer_norm(matmul(x, W1) + b1);
    auto logits = matmul(nn::relu(h), W2) + b2;
    float acc = showcase::multiclass_accuracy(logits.tensor(), y);
    std::cout << "Final accuracy: " << (acc * 100) << "%\n";

    std::cout << "\nLayer normalization trained with autograd!\n";
    std::cout << "Despite wildly varying feature scales, the network converges cleanly.\n";

    finalize();
    return 0;
}
