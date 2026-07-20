/**
 * @file gpt_text_generation.cpp
 * @brief GPT-style Language Model Demo
 *
 * This example demonstrates:
 * - GPT-like decoder-only transformer
 * - Embedding layers for token encoding
 * - Multi-head self-attention with causal masking
 * - GELU activation
 * - LayerNorm for pre-norm architecture
 * - CrossEntropyLoss for language modeling
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <algorithm>

#include "tenzor/tenzor.hpp"
#include "gpt_text_generation_runner.hpp"

using namespace tenzor;
using namespace tenzor::nn;

// ============================================================================
// GPT Building Blocks
// ============================================================================

class GPTBlock : public Module {
public:
    GPTBlock(int64_t embed_dim, int64_t num_heads) {
        attention_ = std::make_shared<MultiheadAttention>(embed_dim, num_heads);
        ln1_ = std::make_shared<LayerNorm>(std::vector<int64_t>{embed_dim});
        ln2_ = std::make_shared<LayerNorm>(std::vector<int64_t>{embed_dim});
        fc1_ = std::make_shared<Linear>(embed_dim, 4 * embed_dim);
        fc2_ = std::make_shared<Linear>(4 * embed_dim, embed_dim);
        gelu_ = std::make_shared<GELU>();
        dropout_ = std::make_shared<Dropout>(0.1f);

        register_module("attention", attention_);
        register_module("ln1", ln1_);
        register_module("ln2", ln2_);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("gelu", gelu_);
        register_module("dropout", dropout_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        // Pre-norm attention
        auto normed = ln1_->forward(x);
        auto [attn_out, _weights] = attention_->forward(normed, normed, normed);
        attn_out = dropout_->forward(attn_out);
        auto h = x + attn_out;

        // Pre-norm FFN
        auto ff_in = ln2_->forward(h);
        auto ff = gelu_->forward(fc1_->forward(ff_in));
        ff = dropout_->forward(ff);
        ff = fc2_->forward(ff);

        return h + ff;
    }

private:
    std::shared_ptr<MultiheadAttention> attention_;
    std::shared_ptr<LayerNorm> ln1_, ln2_;
    std::shared_ptr<Linear> fc1_, fc2_;
    std::shared_ptr<GELU> gelu_;
    std::shared_ptr<Dropout> dropout_;
};

// ============================================================================
// Simple GPT Model
// ============================================================================

class SimpleGPT : public Module {
public:
    SimpleGPT(int64_t vocab_size, int64_t embed_dim, int64_t num_heads,
              int64_t num_layers, int64_t max_seq_len)
        : vocab_size_(vocab_size), embed_dim_(embed_dim), max_seq_len_(max_seq_len) {

        token_emb_ = std::make_shared<Embedding>(vocab_size, embed_dim);
        pos_emb_ = std::make_shared<Embedding>(max_seq_len, embed_dim);
        dropout_ = std::make_shared<Dropout>(0.1f);

        for (int64_t i = 0; i < num_layers; ++i) {
            auto block = std::make_shared<GPTBlock>(embed_dim, num_heads);
            blocks_.push_back(block);
            register_module("block_" + std::to_string(i), block);
        }

        ln_f_ = std::make_shared<LayerNorm>(std::vector<int64_t>{embed_dim});
        lm_head_ = std::make_shared<Linear>(embed_dim, vocab_size);

        register_module("token_emb", token_emb_);
        register_module("pos_emb", pos_emb_);
        register_module("ln_f", ln_f_);
        register_module("lm_head", lm_head_);
    }

    auto forward_impl(const Variable& input_ids) -> Variable override {
        auto shape = input_ids.shape();
        int64_t batch_size = shape[0];
        int64_t seq_len = shape[1];

        // Create position indices
        std::vector<int64_t> pos_data(seq_len);
        for (int64_t i = 0; i < seq_len; ++i) {
            pos_data[i] = i;
        }
        auto positions = from_data(pos_data.data(), {1, seq_len}, input_ids.tensor().device());

        // Embeddings
        auto tok_emb = token_emb_->forward(input_ids);
        Variable pos_var(positions, false);
        auto pos_emb = pos_emb_->forward(pos_var);

        // Broadcast position embeddings
        auto h = tok_emb + pos_emb;
        h = dropout_->forward(h);

        // Transformer blocks
        for (auto& block : blocks_) {
            h = block->forward(h);
        }

        // Final layer norm and LM head
        h = ln_f_->forward(h);
        return lm_head_->forward(h);
    }

private:
    int64_t vocab_size_;
    int64_t embed_dim_;
    int64_t max_seq_len_;
    std::shared_ptr<Embedding> token_emb_;
    std::shared_ptr<Embedding> pos_emb_;
    std::shared_ptr<Dropout> dropout_;
    std::vector<std::shared_ptr<GPTBlock>> blocks_;
    std::shared_ptr<LayerNorm> ln_f_;
    std::shared_ptr<Linear> lm_head_;
};

// ============================================================================
// Demo Functions
// ============================================================================

void demo_causal_attention() {
    std::cout << "\n=== Causal Attention Demo ===\n\n";

    std::cout << "Causal Mask (5x5):\n";
    std::cout << "Each position can only attend to itself and previous positions\n\n";

    int seq_len = 5;
    std::cout << "       ";
    for (int j = 0; j < seq_len; ++j) std::cout << "K" << j << " ";
    std::cout << "\n";

    for (int i = 0; i < seq_len; ++i) {
        std::cout << "  Q" << i << ":  ";
        for (int j = 0; j < seq_len; ++j) {
            std::cout << (j <= i ? "1  " : "0  ");
        }
        std::cout << "\n";
    }

    std::cout << "\nThis prevents information leakage from future tokens.\n";
}

void demo_generation_strategies() {
    std::cout << "\n=== Generation Strategies ===\n\n";

    std::cout << "1. Greedy Decoding:\n";
    std::cout << "   - Always select highest probability token\n";
    std::cout << "   - Deterministic but may be repetitive\n\n";

    std::cout << "2. Temperature Sampling:\n";
    std::cout << "   - Divide logits by temperature before softmax\n";
    std::cout << "   - T < 1: sharper distribution (more confident)\n";
    std::cout << "   - T > 1: flatter distribution (more random)\n\n";

    std::cout << "3. Top-k Sampling:\n";
    std::cout << "   - Keep only top k tokens, zero out rest\n";
    std::cout << "   - Prevents very unlikely tokens\n\n";

    std::cout << "4. Top-p (Nucleus) Sampling:\n";
    std::cout << "   - Keep smallest set with cumulative prob >= p\n";
    std::cout << "   - Adaptive to confidence level\n";
}

void demo_embedding_layer(Device device) {
    std::cout << "\n=== Embedding Layer Demo ===\n\n";

    int64_t vocab_size = 1000;
    int64_t embed_dim = 128;

    auto embedding = std::make_shared<Embedding>(vocab_size, embed_dim);
    embedding->to(device);

    std::vector<int64_t> tokens = {1, 50, 100, 500, 999};
    auto input = from_data(tokens.data(), {1, 5}, device);
    Variable input_var(input, false);

    auto output = embedding->forward(input_var);

    std::cout << "Embedding(vocab=" << vocab_size << ", dim=" << embed_dim << "):\n";
    std::cout << "  Input: [1, 5] (token indices)\n";
    std::cout << "  Output: [" << output.shape()[0] << ", "
              << output.shape()[1] << ", " << output.shape()[2] << "]\n";
    std::cout << "  Each token mapped to " << embed_dim << "-dim vector\n";
}

// ============================================================================
// Training
// ============================================================================

void train_gpt(Device device) {
    // NN.24: the tiny-GPT training body is shared with the regression test
    // via examples::gpt_text_generation::run_gpt_text_generation_training.
    // Run it first so the exe exercises the same code path the test does,
    // then continue with the original larger demo loop.
    std::cout << "\n=== Training Simple GPT Model ===\n\n";

    {
        double init_loss = 0.0;
        double final_loss = 0.0;
        const int runner_epochs = 5;
        examples::gpt_text_generation::run_gpt_text_generation_training(
            runner_epochs, &init_loss, &final_loss, device, /*verbose=*/false);
        std::cout << "[runner] short-loop training: initial=" << init_loss
                  << " final=" << final_loss << " over " << runner_epochs
                  << " epochs\n\n";
    }

    int64_t vocab_size = 1000;
    int64_t embed_dim = 128;
    int64_t num_heads = 4;
    int64_t num_layers = 2;
    int64_t max_seq_len = 32;
    int batch_size = 8;
    int num_train = 200;
    int num_epochs = 5;

    auto model = std::make_shared<SimpleGPT>(vocab_size, embed_dim, num_heads,
                                              num_layers, max_seq_len);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);

    CrossEntropyLoss criterion;

    std::cout << "Configuration:\n";
    std::cout << "  Model: Simple GPT (decoder-only transformer)\n";
    std::cout << "  Vocab size: " << vocab_size << "\n";
    std::cout << "  Embedding dim: " << embed_dim << "\n";
    std::cout << "  Attention heads: " << num_heads << "\n";
    std::cout << "  Layers: " << num_layers << "\n";
    std::cout << "  Max sequence: " << max_seq_len << "\n";
    std::cout << "  Loss: CrossEntropyLoss\n";
    std::cout << "  Optimizer: Adam (lr=0.001)\n\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> token_dist(0, vocab_size - 1);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;

        for (int i = 0; i < num_train; i += batch_size) {
            int end = std::min(i + batch_size, num_train);
            int actual_batch = end - i;
            int seq_len = max_seq_len;

            // Generate random token sequences
            std::vector<int64_t> input_data(actual_batch * seq_len);
            std::vector<int64_t> target_data(actual_batch * seq_len);

            for (int b = 0; b < actual_batch; ++b) {
                for (int t = 0; t < seq_len; ++t) {
                    int64_t token = token_dist(rng);
                    input_data[b * seq_len + t] = token;
                    // Target is next token (shifted by 1)
                    if (t < seq_len - 1) {
                        target_data[b * seq_len + t] = token_dist(rng);
                    } else {
                        target_data[b * seq_len + t] = 0;  // EOS
                    }
                }
            }

            auto inputs = from_data(input_data.data(), {actual_batch, seq_len}, device);
            auto targets = from_data(target_data.data(), {actual_batch, seq_len}, device);

            optimizer.zero_grad();

            Variable input_var(inputs, true);
            auto logits = model->forward(input_var);

            // Reshape for cross entropy: [batch * seq, vocab]
            auto logits_flat = logits.tensor().reshape({actual_batch * seq_len, vocab_size});
            auto targets_flat = targets.reshape({actual_batch * seq_len});

            Variable logits_var(logits_flat, true);
            auto loss = criterion(logits_var, targets_flat);

            loss.backward();
            optimizer.step();

            auto loss_cpu = loss.tensor().cpu();
            epoch_loss += loss_cpu.data<float>()[0];
            num_batches++;
        }

        float avg_loss = epoch_loss / num_batches;
        float perplexity = std::exp(avg_loss);

        std::cout << "Epoch " << std::setw(2) << (epoch + 1) << "/" << num_epochs
                  << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | Perplexity: " << std::setprecision(2) << perplexity << "\n";
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    tenzor::initialize();

    Device device = Device::cpu();
    if (argc > 1) {
        std::string backend = argv[1];
        if (backend == "cuda") device = Device::cuda();
        else if (backend == "rocm") device = Device::rocm();
        else if (backend == "vulkan") device = Device::vulkan();
        else if (backend == "oneapi") device = Device::oneapi();
        else if (backend == "mps") device = Device::mps();
    }

    std::cout << "======================================================\n";
    std::cout << "   GPT Text Generation - Components Demo              \n";
    std::cout << "   Backend: " << device.to_string() << "\n";
    std::cout << "======================================================\n";

    std::cout << "\nComponents demonstrated:\n";
    std::cout << "  Layers: Embedding, Linear, LayerNorm, Dropout\n";
    std::cout << "  Attention: MultiheadAttention (causal)\n";
    std::cout << "  Activations: GELU\n";
    std::cout << "  Loss: CrossEntropyLoss\n";
    std::cout << "  Model: GPT-style decoder-only transformer\n";

    try {
        demo_causal_attention();
        demo_generation_strategies();
        demo_embedding_layer(device);
        train_gpt(device);

        std::cout << "\n======================================================\n";
        std::cout << "   All GPT examples completed successfully!          \n";
        std::cout << "======================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    tenzor::finalize();
    return 0;
}
