/**
 * @file autograd.cpp
 * @brief Mini hierarchical reasoning network trained with autograd
 *
 * Builds an HRM-style two-timescale recurrence from primitives (matmul,
 * tanh) and trains it on a tiny modular-arithmetic task:
 *
 *   given a length-T sequence of digits (0..9),
 *   predict (sum of digits) mod 7
 *
 * The architecture:
 *   E  = one_hot(x) @ W_emb              // embedding (B, T, D)
 *   H, L initialised from E
 *   for n in 1..N_high:
 *       for t in 1..T_low:
 *           L = tanh(L @ W_L + H @ U_L + b_L)
 *       H = tanh(H @ W_H + L @ U_H + b_H)
 *   logits = mean_t(H) @ W_out + b_out   // (B, 7)
 *
 * The autograd graph spans every cycle, so backprop trains all
 * weights jointly. The full nn::HRM (neural_network.cpp) replaces
 * each block with a transformer + ACT halting.
 *
 * Usage: ./22_hierarchical_reasoning_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <random>
#include <vector>

using namespace tenzor;

static int correct_count(const Tensor& logits, const Tensor& target) {
    auto lp = logits.cpu();
    auto tp = target.cpu();
    int B = static_cast<int>(lp.shape()[0]);
    int C = static_cast<int>(lp.shape()[1]);
    const float* l = lp.data<float>();
    const int64_t* t = tp.data<int64_t>();
    int hits = 0;
    for (int b = 0; b < B; ++b) {
        int best = 0;
        float bv = l[b * C];
        for (int c = 1; c < C; ++c) {
            if (l[b * C + c] > bv) { bv = l[b * C + c]; best = c; }
        }
        if (best == t[b]) ++hits;
    }
    return hits;
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Hierarchical Reasoning Model - Autograd", device);
    manual_seed(42);

    const int batch       = 64;
    const int seq         = 6;
    const int vocab       = 10;   // digits 0..9
    const int d_model     = 16;
    const int n_classes   = 7;    // sum mod 7
    const int N_high      = 2;
    const int T_low       = 2;

    // ---- Synthesise the dataset (digits + their mod-7 sums) -----------
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> digit(0, vocab - 1);

    std::vector<float> X_oh(batch * seq * vocab, 0.0f);  // one-hot tokens
    std::vector<int64_t> Y_idx(batch);
    for (int b = 0; b < batch; ++b) {
        int s = 0;
        for (int t = 0; t < seq; ++t) {
            int d = digit(rng);
            s += d;
            X_oh[(b * seq + t) * vocab + d] = 1.0f;
        }
        Y_idx[b] = s % n_classes;
    }
    auto X_tensor = from_data(X_oh.data(), {batch, seq, vocab}, device);
    auto Y_tensor = from_data(Y_idx.data(), {batch}, device);

    // One-hot targets for the autograd cross-entropy
    std::vector<float> Y_oh(batch * n_classes, 0.0f);
    for (int b = 0; b < batch; ++b) Y_oh[b * n_classes + Y_idx[b]] = 1.0f;
    auto Y_oh_tensor = from_data(Y_oh.data(), {batch, n_classes}, device);

    // ---- Trainable weights --------------------------------------------
    auto sc_emb = std::sqrt(1.0f / vocab);
    auto sc_d   = std::sqrt(1.0f / d_model);

    Variable W_emb(randn({vocab,   d_model}, DType::Float32, device) * sc_emb, true);
    Variable W_L  (randn({d_model, d_model}, DType::Float32, device) * sc_d,   true);
    Variable U_L  (randn({d_model, d_model}, DType::Float32, device) * sc_d,   true);
    Variable b_L  (zeros({d_model}, DType::Float32, device),                   true);
    Variable W_H  (randn({d_model, d_model}, DType::Float32, device) * sc_d,   true);
    Variable U_H  (randn({d_model, d_model}, DType::Float32, device) * sc_d,   true);
    Variable b_H  (zeros({d_model}, DType::Float32, device),                   true);
    Variable W_out(randn({d_model, n_classes}, DType::Float32, device) * sc_d, true);
    Variable b_out(zeros({n_classes}, DType::Float32, device),                 true);

    auto params = std::vector<Variable*>{
        &W_emb, &W_L, &U_L, &b_L, &W_H, &U_H, &b_H, &W_out, &b_out
    };

    showcase::print_section("Architecture");
    std::cout << "OneHot(" << vocab << ") -> Linear(" << vocab << "->" << d_model
              << ") -> HRM(N=" << N_high << ", T=" << T_low
              << ") -> mean_t -> Linear(" << d_model << "->" << n_classes << ")\n";
    std::cout << "Task: predict (sum of " << seq << " digits) mod " << n_classes << "\n";

    float lr = 0.05f;
    int num_epochs = 300;
    int print_every = 30;

    showcase::print_section("Training");
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable X(X_tensor, false);
        Variable Y_oh_v(Y_oh_tensor, false);

        // Embed: (B*T, vocab) @ (vocab, D) -> (B, T, D)
        auto X_flat = reshape(X, {batch * seq, vocab});
        auto E_flat = matmul(X_flat, W_emb);
        auto E      = reshape(E_flat, {batch, seq, d_model});

        // Initial states: H starts at the embedding, L starts small
        Variable H = E;
        Variable L = E * 0.1f;

        // Hierarchical cycling
        for (int n = 0; n < N_high; ++n) {
            for (int t = 0; t < T_low; ++t) {
                auto L_flat = reshape(L, {batch * seq, d_model});
                auto H_flat = reshape(H, {batch * seq, d_model});
                auto z = matmul(L_flat, W_L) + matmul(H_flat, U_L) + b_L;
                L = reshape(tanh(z), {batch, seq, d_model});
            }
            auto H_flat = reshape(H, {batch * seq, d_model});
            auto L_flat = reshape(L, {batch * seq, d_model});
            auto z = matmul(H_flat, W_H) + matmul(L_flat, U_H) + b_H;
            H = reshape(tanh(z), {batch, seq, d_model});
        }

        // Pool over time, project to logits
        auto pooled = mean(H, {1}, false);                         // (B, D)
        auto logits = matmul(pooled, W_out) + b_out;               // (B, C)

        // Cross-entropy: -mean(sum_c one_hot * log_softmax(logits))
        auto log_probs = log_softmax(logits, 1);
        auto loss = mean(sum(Y_oh_v * log_probs, 1)) * -1.0f;

        for (auto* p : params) p->zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            for (auto* p : params) {
                *p = Variable(p->tensor() - (*p->grad() * lr), true);
            }
        }

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            int hits = correct_count(logits.tensor(), Y_tensor);
            float acc = static_cast<float>(hits) / batch;
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] loss=" << loss.tensor().item<float>()
                      << "  acc=" << (acc * 100.0f) << "%\n";
        }
    }

    showcase::print_section("Final Evaluation");
    {
        Variable X(X_tensor, false);
        auto X_flat = reshape(X, {batch * seq, vocab});
        auto E      = reshape(matmul(X_flat, W_emb), {batch, seq, d_model});
        Variable H = E;
        Variable L = E * 0.1f;
        for (int n = 0; n < N_high; ++n) {
            for (int t = 0; t < T_low; ++t) {
                auto z = matmul(reshape(L, {batch * seq, d_model}), W_L)
                       + matmul(reshape(H, {batch * seq, d_model}), U_L) + b_L;
                L = reshape(tanh(z), {batch, seq, d_model});
            }
            auto z = matmul(reshape(H, {batch * seq, d_model}), W_H)
                   + matmul(reshape(L, {batch * seq, d_model}), U_H) + b_H;
            H = reshape(tanh(z), {batch, seq, d_model});
        }
        auto logits = matmul(mean(H, {1}, false), W_out) + b_out;
        int hits = correct_count(logits.tensor(), Y_tensor);
        std::cout << "Train accuracy: "
                  << (100.0f * hits / batch) << "%  (chance = "
                  << (100.0f / n_classes) << "%)\n";
    }

    std::cout << "\nMini hierarchical recurrence trained end-to-end with autograd.\n"
                 "The H/L cycling lets the network combine partial sums across\n"
                 "the sequence before projecting to a mod-" << n_classes << " logit.\n";

    finalize();
    return 0;
}
