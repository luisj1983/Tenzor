/**
 * @file autograd_runner.cpp
 * @brief Implementation of the word-embedding autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <vector>

namespace tenzor::examples::showcase17 {

int run_word_embedding_training(int epochs,
                                double* out_initial,
                                double* out_final,
                                ::tenzor::Device device,
                                bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    int V = 8;
    int D = 8;

    std::vector<std::vector<int64_t>> sents = {
        {0, 1, 4, 5}, {0, 2, 3, 6}, {0, 1, 3, 7}, {0, 2, 4, 5},
    };

    std::vector<int64_t> centers, contexts;
    for (auto& s : sents) {
        for (size_t i = 0; i < s.size(); ++i) {
            for (int d = -1; d <= 1; ++d) {
                if (d == 0) continue;
                int j = static_cast<int>(i) + d;
                if (j < 0 || j >= (int)s.size()) continue;
                centers.push_back(s[i]);
                contexts.push_back(s[j]);
            }
        }
    }
    int N = static_cast<int>(centers.size());

    Variable W_emb(randn({V, D}, DType::Float32, device) * 0.1f, true);
    Variable W_ctx(randn({V, D}, DType::Float32, device) * 0.1f, true);

    auto center_t  = from_data(centers.data(),  {N}, device);
    auto context_t = from_data(contexts.data(), {N}, device);

    std::vector<float> one_hot(N * V, 0.0f);
    for (int i = 0; i < N; ++i) one_hot[i * V + contexts[i]] = 1.0f;
    auto target = from_data(one_hot.data(), {N, V}, device);

    float lr = 0.5f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto center_emb = index_select(W_emb, 0, center_t);

        auto scores = matmul(center_emb, W_ctx.transpose(0, 1));
        auto log_p = log_softmax(scores, 1);

        Variable target_v(target, false);
        auto loss = mean(sum(target_v * log_p, 1)) * (-1.0f);

        W_emb.zero_grad(); W_ctx.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W_emb = Variable(W_emb.tensor() - (*W_emb.grad() * lr), true);
            W_ctx = Variable(W_ctx.tensor() - (*W_ctx.grad() * lr), true);
        }

        double loss_val = static_cast<double>(loss.tensor().item<float>());
        if (epoch == 0 && out_initial) *out_initial = loss_val;
        final_loss = loss_val;

        if (verbose && ((epoch + 1) % print_every == 0 || epoch == 0)) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << epochs
                      << "] nll=" << loss_val << "\n";
        }
    }
    if (out_final) *out_final = final_loss;
    return 0;
}

}  // namespace tenzor::examples::showcase17
