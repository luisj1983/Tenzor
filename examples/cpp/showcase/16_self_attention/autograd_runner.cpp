/**
 * @file autograd_runner.cpp
 * @brief Implementation of the self-attention autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <cmath>
#include <vector>

namespace tenzor::examples::showcase16 {

int run_self_attention_training(int epochs,
                                double* out_initial,
                                double* out_final,
                                ::tenzor::Device device,
                                bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    int batch = 4;
    int seq   = 6;
    int d_model = 8;

    std::vector<float> X_data(batch * seq * d_model);
    std::vector<float> y_data(batch * seq * d_model);
    for (int i = 0; i < batch * seq * d_model; ++i) {
        X_data[i] = ((i * 7919) % 997) / 497.0f - 1.0f;
    }
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seq; ++t) {
            int src = (t + 1) % seq;
            for (int d = 0; d < d_model; ++d) {
                y_data[(b * seq + t) * d_model + d] = X_data[(b * seq + src) * d_model + d];
            }
        }
    }
    auto X_tensor = from_data(X_data.data(), {batch, seq, d_model}, device);
    auto y_tensor = from_data(y_data.data(), {batch, seq, d_model}, device);

    auto xav = [&](int64_t fin) { return std::sqrt(1.0f / fin); };
    Variable W_q(randn({d_model, d_model}, DType::Float32, device) * xav(d_model), true);
    Variable W_k(randn({d_model, d_model}, DType::Float32, device) * xav(d_model), true);
    Variable W_v(randn({d_model, d_model}, DType::Float32, device) * xav(d_model), true);
    Variable W_o(randn({d_model, d_model}, DType::Float32, device) * xav(d_model), true);

    float lr = 0.05f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable X(X_tensor, false);
        Variable Y(y_tensor, false);

        auto X_flat = reshape(X, {batch * seq, d_model});
        auto Q = reshape(matmul(X_flat, W_q), {batch, seq, d_model});
        auto K = reshape(matmul(X_flat, W_k), {batch, seq, d_model});
        auto V = reshape(matmul(X_flat, W_v), {batch, seq, d_model});

        auto Kt = transpose(K, 1, 2);
        auto scores = bmm(Q, Kt) * (1.0f / std::sqrt(static_cast<float>(d_model)));
        auto attn = softmax(scores, 2);
        auto attended = bmm(attn, V);

        auto attended_flat = reshape(attended, {batch * seq, d_model});
        auto pred_flat = matmul(attended_flat, W_o);
        auto pred = reshape(pred_flat, {batch, seq, d_model});

        auto err = pred - Y;
        auto loss = mean(err * err);

        W_q.zero_grad(); W_k.zero_grad(); W_v.zero_grad(); W_o.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W_q = Variable(W_q.tensor() - (*W_q.grad() * lr), true);
            W_k = Variable(W_k.tensor() - (*W_k.grad() * lr), true);
            W_v = Variable(W_v.tensor() - (*W_v.grad() * lr), true);
            W_o = Variable(W_o.tensor() - (*W_o.grad() * lr), true);
        }

        double loss_val = static_cast<double>(loss.tensor().item<float>());
        if (epoch == 0 && out_initial) *out_initial = loss_val;
        final_loss = loss_val;

        if (verbose && ((epoch + 1) % print_every == 0 || epoch == 0)) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << epochs
                      << "] mse=" << loss_val << "\n";
        }
    }
    if (out_final) *out_final = final_loss;
    return 0;
}

}  // namespace tenzor::examples::showcase16
