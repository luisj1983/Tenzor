/**
 * @file neural_network.cpp
 * @brief Skip-gram with nn::Embedding and Adam
 *
 * Same task as the autograd tier, now using nn::Embedding (which is
 * just a learnable lookup table) and the high-level Adam optimizer.
 *
 * Usage: ./17_word_embedding_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <string>
#include <vector>

using namespace tenzor;

class SkipGram : public nn::Module {
public:
    SkipGram(int64_t V, int64_t D) {
        emb_in  = std::make_shared<nn::Embedding>(V, D);
        emb_out = std::make_shared<nn::Embedding>(V, D);
        register_module("emb_in",  emb_in);
        register_module("emb_out", emb_out);
    }

    // center: (N,) int64 -> scores (N, V)
    Variable score(const Variable& centers) {
        auto c = emb_in->forward(centers);            // (N, D)
        // Use all out-embeddings as keys: scores = c @ W_out^T
        auto W_out = emb_out->weight();               // (V, D)
        return matmul(c, transpose(W_out, 0, 1));     // (N, V)
    }

    auto forward_impl(const Variable& centers) -> Variable override {
        return score(centers);
    }

    std::shared_ptr<nn::Embedding> emb_in, emb_out;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Word Embedding - Neural Network API (Skip-gram)", device);
    manual_seed(42);

    std::vector<std::string> vocab = {
        "the", "cat", "dog", "ran", "sat", "house", "fast", "slow"
    };
    int V = static_cast<int>(vocab.size());
    int D = 8;

    std::vector<std::vector<int64_t>> sents = {
        {0, 1, 4, 5}, {0, 2, 3, 6}, {0, 1, 3, 7}, {0, 2, 4, 5}
    };
    std::vector<int64_t> centers, contexts;
    for (auto& s : sents) {
        for (size_t i = 0; i < s.size(); ++i) {
            for (int d = -1; d <= 1; ++d) {
                if (d == 0) continue;
                int j = (int)i + d;
                if (j < 0 || j >= (int)s.size()) continue;
                centers.push_back(s[i]);
                contexts.push_back(s[j]);
            }
        }
    }
    int N = static_cast<int>(centers.size());

    auto X = from_data(centers.data(),  {N}, device);
    auto y = from_data(contexts.data(), {N}, device);

    auto model = std::make_shared<SkipGram>(V, D);
    model->to(device);

    auto params = model->parameters();
    optim::Adam opt(params, 0.05f);
    nn::CrossEntropyLoss criterion;

    showcase::print_section("Architecture");
    std::cout << "Embedding(" << V << " -> " << D << ")  [center]\n";
    std::cout << "Embedding(" << V << " -> " << D << ")  [context]\n";

    int num_epochs = 800;
    int print_every = 80;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        opt.zero_grad();
        Variable x(X, false);
        auto logits = model->forward(x);
        auto loss = criterion(logits, y);  // CrossEntropyLoss target is raw Tensor
        loss.backward();
        opt.step();
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] CrossEntropy: " << loss.tensor().item<float>() << "\n";
        }
    }

    showcase::print_section("Cosine similarity to 'cat' (after training)");
    auto W = model->emb_in->weight().tensor().cpu();
    auto cos_sim = [&](int a, int b) {
        float dot = 0, na = 0, nb = 0;
        for (int d = 0; d < D; ++d) {
            float va = W.data<float>()[a * D + d];
            float vb = W.data<float>()[b * D + d];
            dot += va * vb; na += va * va; nb += vb * vb;
        }
        return dot / (std::sqrt(na * nb) + 1e-8f);
    };
    for (int i = 0; i < V; ++i) {
        std::cout << "  sim(cat, " << vocab[i] << ") = " << cos_sim(1, i) << "\n";
    }

    std::cout << "\nSkip-gram trained using Neural Network API!\n";

    finalize();
    return 0;
}
