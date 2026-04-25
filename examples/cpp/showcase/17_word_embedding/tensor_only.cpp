/**
 * @file tensor_only.cpp
 * @brief Word embedding table lookup and similarity with raw tensors
 *
 * Shows the fundamental embedding operation: indexing rows of a
 * [vocab, dim] matrix by integer token IDs. No training here -
 * the tensor_only tier just demonstrates the lookup mechanics
 * and cosine similarity between two random word vectors.
 *
 * Usage: ./17_word_embedding_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <string>
#include <vector>

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Word Embedding - Tensor Only (Lookup Demo)", device);
    manual_seed(42);

    std::vector<std::string> vocab = {
        "the", "cat", "dog", "ran", "sat", "house", "fast", "slow"
    };
    int vocab_size = static_cast<int>(vocab.size());
    int embed_dim = 8;

    // Embedding matrix [vocab, dim]
    auto W = randn({vocab_size, embed_dim}, DType::Float32, device) * 0.1f;

    // Lookup: select rows
    std::vector<int64_t> ids = {1, 2};  // "cat", "dog"
    auto idx_tensor = from_data(ids.data(), {2}, device);
    auto embedded = index_select(W, 0, idx_tensor);   // (2, embed_dim)

    showcase::print_section("Embedding Lookup");
    std::cout << "Vocabulary: [";
    for (size_t i = 0; i < vocab.size(); ++i) {
        std::cout << "\"" << vocab[i] << "\"" << (i + 1 < vocab.size() ? ", " : "");
    }
    std::cout << "]\n";
    std::cout << "Embedding dim: " << embed_dim << "\n\n";

    auto e_cpu = embedded.cpu();
    for (size_t i = 0; i < ids.size(); ++i) {
        std::cout << vocab[ids[i]] << ": [";
        for (int d = 0; d < embed_dim; ++d) {
            std::cout << e_cpu.data<float>()[i * embed_dim + d];
            if (d < embed_dim - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // Cosine similarity between two embeddings
    showcase::print_section("Cosine similarity (cat vs every other word)");
    auto cat = index_select(W, 0, idx_tensor.slice(0, 0, 1));  // (1, D)
    std::vector<int64_t> all_ids(vocab_size);
    for (int i = 0; i < vocab_size; ++i) all_ids[i] = i;
    auto all_idx = from_data(all_ids.data(), {vocab_size}, device);
    auto others = index_select(W, 0, all_idx);                 // (V, D)

    // sim = (a . b) / (|a||b|)
    auto dot = matmul(others, cat.transpose(0, 1));            // (V, 1)
    auto cat_norm = tenzor::sqrt(tenzor::sum(cat * cat, 1, true));      // (1, 1)
    auto ot_norm  = tenzor::sqrt(tenzor::sum(others * others, 1, true));// (V, 1)
    auto sim = dot / (cat_norm * ot_norm + 1e-8f);

    auto s_cpu = sim.cpu();
    for (int i = 0; i < vocab_size; ++i) {
        std::cout << "  sim(cat, " << vocab[i] << ") = " << s_cpu.data<float>()[i] << "\n";
    }

    std::cout << "\nWord embedding lookup demonstrated with raw tensors!\n";
    std::cout << "index_select() pulls rows out of the embedding matrix.\n";

    finalize();
    return 0;
}
