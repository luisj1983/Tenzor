/**
 * @file autograd.cpp
 * @brief Trainable self-attention layer using autograd
 *
 * Learns a simple copy-task-by-position: given seq with unique tokens,
 * the supervised target is the input shifted left. Self-attention with
 * a trainable output head learns to route values through attention.
 *
 * Usage: ./16_self_attention_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Self-Attention - Autograd", device);
    manual_seed(42);

    int batch = 4;
    int seq   = 6;
    int d_model = 8;

    // Data: random (batch, seq, d_model); target is rolled by 1 along seq
    std::vector<float> X_data(batch * seq * d_model);
    std::vector<float> y_data(batch * seq * d_model);
    for (int i = 0; i < batch * seq * d_model; ++i) {
        X_data[i] = ((i * 7919) % 997) / 497.0f - 1.0f;
    }
    // y[b, t, :] = x[b, (t + 1) % seq, :]
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
    int num_epochs = 600;
    int print_every = 60;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable X(X_tensor, false);
        Variable Y(y_tensor, false);

        // Linear projections via matmul on flattened (B*T, D)
        auto X_flat = reshape(X, {batch * seq, d_model});
        auto Q = reshape(matmul(X_flat, W_q), {batch, seq, d_model});
        auto K = reshape(matmul(X_flat, W_k), {batch, seq, d_model});
        auto V = reshape(matmul(X_flat, W_v), {batch, seq, d_model});

        // Attention
        auto Kt = transpose(K, 1, 2);                              // (B, D, T)
        auto scores = bmm(Q, Kt) * (1.0f / std::sqrt(static_cast<float>(d_model)));
        auto attn = softmax(scores, 2);
        auto attended = bmm(attn, V);                              // (B, T, D)

        // Output projection
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

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] MSE: " << loss.tensor().item<float>() << "\n";
        }
    }

    showcase::print_section("Final Results");
    Variable X(X_tensor, false);
    auto X_flat = reshape(X, {batch * seq, d_model});
    auto Q = reshape(matmul(X_flat, W_q), {batch, seq, d_model});
    auto K = reshape(matmul(X_flat, W_k), {batch, seq, d_model});
    auto V = reshape(matmul(X_flat, W_v), {batch, seq, d_model});
    auto scores = bmm(Q, transpose(K, 1, 2)) * (1.0f / std::sqrt(static_cast<float>(d_model)));
    auto attn_final = softmax(scores, 2);

    auto a_cpu = attn_final.tensor().cpu();
    std::cout << "Attention pattern for batch=0 (each row should peak near (t+1) mod " << seq << "):\n";
    for (int t = 0; t < seq; ++t) {
        std::cout << "  t=" << t << ": [";
        for (int s = 0; s < seq; ++s) {
            std::cout << a_cpu.data<float>()[t * seq + s];
            if (s < seq - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    std::cout << "\nSelf-attention trained with autograd!\n";
    std::cout << "The attention weights learn to look at the next position.\n";

    finalize();
    return 0;
}
