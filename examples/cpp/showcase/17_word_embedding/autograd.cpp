/**
 * @file autograd.cpp
 * @brief Skip-gram-style embedding training with autograd
 *
 * Toy skip-gram: for each (center, context) pair in a small corpus,
 * predict the context from the center. Gradient flows back through
 * index_select into the embedding matrix, so the learned rows
 * land closer to their contexts in the embedding space.
 *
 * Usage: ./17_word_embedding_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <string>
#include <vector>

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Word Embedding - Autograd (Skip-gram)", device);
    manual_seed(42);

    std::vector<std::string> vocab = {
        "the", "cat", "dog", "ran", "sat", "house", "fast", "slow"
    };
    int V = static_cast<int>(vocab.size());
    int D = 8;

    // Toy sentences (word IDs). cat/dog appear with ran/sat/fast/slow; house never does.
    std::vector<std::vector<int64_t>> sents = {
        {0, 1, 4, 5}, // the cat sat house
        {0, 2, 3, 6}, // the dog ran fast
        {0, 1, 3, 7}, // the cat ran slow
        {0, 2, 4, 5}, // the dog sat house
    };

    // Build (center, context) pairs within a window of 1
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

    float lr = 0.5f;
    int num_epochs = 2000;
    int print_every = 200;

    showcase::print_section("Training");
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto center_emb  = index_select(W_emb, 0, center_t);          // (N, D)
        auto context_emb = index_select(W_ctx, 0, context_t);         // (N, D)

        // Softmax-over-vocab of (center . all_ctx_rows) would be standard,
        // but with tiny vocab we can use a direct dot-product + cross-entropy
        // by computing scores over the full vocabulary:
        auto scores = matmul(center_emb, W_ctx.transpose(0, 1));      // (N, V)
        auto log_p = log_softmax(scores, 1);

        // One-hot for contexts
        std::vector<float> one_hot(N * V, 0.0f);
        for (int i = 0; i < N; ++i) one_hot[i * V + contexts[i]] = 1.0f;
        auto target = from_data(one_hot.data(), {N, V}, device);
        Variable target_v(target, false);

        auto loss = mean(sum(target_v * log_p, 1)) * (-1.0f);

        W_emb.zero_grad(); W_ctx.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W_emb = Variable(W_emb.tensor() - (*W_emb.grad() * lr), true);
            W_ctx = Variable(W_ctx.tensor() - (*W_ctx.grad() * lr), true);
        }

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] NLL: " << loss.tensor().item<float>() << "\n";
        }
    }

    showcase::print_section("Cosine similarity to 'cat'");
    auto cat_emb = W_emb.tensor().cpu();
    auto cos_sim = [&](int a, int b) {
        float dot = 0, na = 0, nb = 0;
        for (int d = 0; d < D; ++d) {
            float va = cat_emb.data<float>()[a * D + d];
            float vb = cat_emb.data<float>()[b * D + d];
            dot += va * vb;  na += va * va;  nb += vb * vb;
        }
        return dot / (std::sqrt(na * nb) + 1e-8f);
    };
    for (int i = 0; i < V; ++i) {
        std::cout << "  sim(cat, " << vocab[i] << ") = " << cos_sim(1, i) << "\n";
    }

    std::cout << "\nEmbedding trained with autograd!\n";
    std::cout << "Gradient flows through index_select into the embedding matrix.\n";

    finalize();
    return 0;
}
