/**
 * @file autograd_runner.cpp
 * @brief Implementation of the siamese-network autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace tenzor::examples::showcase21 {

int run_siamese_training(int epochs,
                         double* out_initial,
                         double* out_final,
                         ::tenzor::Device device,
                         bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    int N = 128, in_dim = 4, hidden = 16, emb = 4;

    std::vector<float> X1(N * in_dim), X2(N * in_dim);
    std::vector<float> y_data(N);

    auto sample_class = [&](int cls) {
        std::vector<float> v(in_dim);
        for (int d = 0; d < in_dim; ++d) {
            v[d] = (cls - 1) * 0.8f + 0.1f * dist(gen);
        }
        return v;
    };

    std::uniform_int_distribution<int> cls_dist(0, 2);
    std::uniform_int_distribution<int> shift_dist(0, 1);
    for (int i = 0; i < N; ++i) {
        int cls1 = cls_dist(gen);
        int cls2 = (i < N / 2) ? cls1 : ((cls1 + 1 + shift_dist(gen)) % 3);
        auto v1 = sample_class(cls1), v2 = sample_class(cls2);
        for (int d = 0; d < in_dim; ++d) {
            X1[i * in_dim + d] = v1[d];
            X2[i * in_dim + d] = v2[d];
        }
        y_data[i] = (cls1 == cls2) ? 1.0f : 0.0f;
    }
    auto X1_t = from_data(X1.data(), {N, in_dim}, device);
    auto X2_t = from_data(X2.data(), {N, in_dim}, device);
    auto y_t  = from_data(y_data.data(), {N, 1}, device);

    auto he  = [&](int64_t fin) { return std::sqrt(2.0f / fin); };
    auto xav = [&](int64_t fin) { return std::sqrt(1.0f / fin); };

    Variable W1(randn({in_dim, hidden}, DType::Float32, device) * he(in_dim), true);
    Variable b1(zeros({1, hidden}, DType::Float32, device), true);
    Variable W2(randn({hidden, emb}, DType::Float32, device) * xav(hidden), true);
    Variable b2(zeros({1, emb}, DType::Float32, device), true);

    auto encode = [&](const Variable& x) {
        return matmul(nn::relu(matmul(x, W1) + b1), W2) + b2;
    };

    float margin = 1.0f;
    float lr = 0.03f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable x1(X1_t, false), x2(X2_t, false), y(y_t, false);
        auto ea = encode(x1);
        auto eb = encode(x2);
        auto diff = ea - eb;
        auto d2 = sum(diff * diff, 1, true);
        auto d = ::tenzor::sqrt(d2 + 1e-8f);

        auto gap = Variable(ones_like(d.tensor()) * margin, false) - d;
        auto gap_pos = nn::relu(gap);

        auto loss_pos = y * d2;
        auto ones = Variable(ones_like(y.tensor()), false);
        auto loss_neg = (ones - y) * gap_pos * gap_pos;
        auto loss = mean(loss_pos + loss_neg);

        W1.zero_grad(); b1.zero_grad();
        W2.zero_grad(); b2.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W1 = Variable(W1.tensor() - (*W1.grad() * lr), true);
            b1 = Variable(b1.tensor() - (*b1.grad() * lr), true);
            W2 = Variable(W2.tensor() - (*W2.grad() * lr), true);
            b2 = Variable(b2.tensor() - (*b2.grad() * lr), true);
        }

        double loss_val = static_cast<double>(loss.tensor().item<float>());
        if (epoch == 0 && out_initial) *out_initial = loss_val;
        final_loss = loss_val;

        if (verbose && ((epoch + 1) % print_every == 0 || epoch == 0)) {
            std::cout << "Epoch " << (epoch+1) << "  loss=" << loss_val << "\n";
        }
    }
    if (out_final) *out_final = final_loss;
    return 0;
}

}  // namespace tenzor::examples::showcase21
