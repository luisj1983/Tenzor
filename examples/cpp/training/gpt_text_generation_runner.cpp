/**
 * @file gpt_text_generation_runner.cpp
 * @brief Implementation of the GPT-style LM autograd runner.
 *
 * NN.24: extracted from gpt_text_generation.cpp's train_gpt() training
 * body so the regression test in tests/examples/test_all_autograd_examples.cpp
 * can drive the SimpleGPT (decoder-only transformer) +
 * CrossEntropyLoss + Adam pipeline end-to-end and assert that backward
 * actually moves the weights.
 *
 * Uses a deliberately tiny config (vocab=64, embed=32, 2 heads, 2 layers,
 * seq_len=16, batch=2, 8 samples) so the test wall-time stays in the
 * 1-2 second range.  The standalone exe keeps the original
 * vocab=1000 / embed=128 / 8-batch / 200-sample loop.
 */

#include "gpt_text_generation_runner.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "tenzor/tenzor.hpp"

namespace tenzor::examples::gpt_text_generation {

namespace {

using ::tenzor::Variable;
using ::tenzor::Tensor;
using ::tenzor::nn::Module;
using ::tenzor::nn::Embedding;
using ::tenzor::nn::MultiheadAttention;
using ::tenzor::nn::LayerNorm;
using ::tenzor::nn::Linear;
using ::tenzor::nn::GELU;
using ::tenzor::nn::Dropout;
using ::tenzor::nn::CrossEntropyLoss;

class GPTBlock : public Module {
public:
    GPTBlock(int64_t embed_dim, int64_t num_heads) {
        attention_ = std::make_shared<MultiheadAttention>(embed_dim, num_heads);
        ln1_  = std::make_shared<LayerNorm>(std::vector<int64_t>{embed_dim});
        ln2_  = std::make_shared<LayerNorm>(std::vector<int64_t>{embed_dim});
        fc1_  = std::make_shared<Linear>(embed_dim, 4 * embed_dim);
        fc2_  = std::make_shared<Linear>(4 * embed_dim, embed_dim);
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
        auto normed = ln1_->forward(x);
        auto [attn_out, _w] =
            attention_->forward(normed, normed, normed);
        attn_out = dropout_->forward(attn_out);
        auto h = x + attn_out;
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

class SimpleGPT : public Module {
public:
    SimpleGPT(int64_t vocab_size, int64_t embed_dim, int64_t num_heads,
              int64_t num_layers, int64_t max_seq_len)
        : vocab_size_(vocab_size), embed_dim_(embed_dim),
          max_seq_len_(max_seq_len) {
        token_emb_ = std::make_shared<Embedding>(vocab_size, embed_dim);
        pos_emb_   = std::make_shared<Embedding>(max_seq_len, embed_dim);
        dropout_   = std::make_shared<Dropout>(0.1f);
        for (int64_t i = 0; i < num_layers; ++i) {
            auto block = std::make_shared<GPTBlock>(embed_dim, num_heads);
            blocks_.push_back(block);
            register_module("block_" + std::to_string(i), block);
        }
        ln_f_    = std::make_shared<LayerNorm>(
            std::vector<int64_t>{embed_dim});
        lm_head_ = std::make_shared<Linear>(embed_dim, vocab_size);
        register_module("token_emb", token_emb_);
        register_module("pos_emb", pos_emb_);
        register_module("ln_f", ln_f_);
        register_module("lm_head", lm_head_);
    }

    auto forward_impl(const Variable& input_ids) -> Variable override {
        auto shape = input_ids.shape();
        const int64_t seq_len = shape[1];
        std::vector<int64_t> pos_data(seq_len);
        for (int64_t i = 0; i < seq_len; ++i) pos_data[i] = i;
        auto positions = ::tenzor::from_data(
            pos_data.data(), {1, seq_len}, input_ids.tensor().device());
        auto tok_emb = token_emb_->forward(input_ids);
        Variable pos_var(positions, false);
        auto pos_emb = pos_emb_->forward(pos_var);
        auto h = tok_emb + pos_emb;
        h = dropout_->forward(h);
        for (auto& block : blocks_) {
            h = block->forward(h);
        }
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

}  // namespace

int run_gpt_text_generation_training(int epochs,
                                      double* out_initial,
                                      double* out_final,
                                      ::tenzor::Device device,
                                      bool verbose) {
    using namespace ::tenzor;

    // Tiny workload — the standalone exe still runs the original
    // vocab=1000 / embed=128 / 200-sample / 5-epoch loop in main().
    const int64_t vocab_size  = 64;
    const int64_t embed_dim   = 32;
    const int64_t num_heads   = 2;
    const int64_t num_layers  = 2;
    const int64_t max_seq_len = 16;
    const int batch_size      = 2;
    const int num_train       = 8;
    const int seq_len         = static_cast<int>(max_seq_len);

    manual_seed(42);

    auto model = std::make_shared<SimpleGPT>(
        vocab_size, embed_dim, num_heads, num_layers, max_seq_len);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);
    CrossEntropyLoss criterion;

    // Reuse a single fixed (input, target) pair across epochs so the
    // loss signal is comparable between epoch 0 and epoch N. The
    // standalone exe resamples every step, but a fixed batch gives the
    // regression test a clean monotonic signal that backward is moving
    // the weights.
    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> token_dist(0, vocab_size - 1);
    std::vector<int64_t> input_data(num_train * seq_len);
    std::vector<int64_t> target_data(num_train * seq_len);
    for (int b = 0; b < num_train; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            const int64_t tok = token_dist(rng);
            input_data[b * seq_len + t] = tok;
            target_data[b * seq_len + t] =
                (t < seq_len - 1) ? token_dist(rng) : 0;
        }
    }
    auto inputs_all = from_data(
        input_data.data(), {num_train, seq_len}, device);
    auto targets_all = from_data(
        target_data.data(), {num_train, seq_len}, device);

    double initial_loss = 0.0;
    double final_loss   = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        for (int i = 0; i < num_train; i += batch_size) {
            int end = std::min(i + batch_size, num_train);
            int actual_batch = end - i;

            auto inputs  = inputs_all.slice(0, i, end);
            auto targets = targets_all.slice(0, i, end);

            optimizer.zero_grad();
            Variable input_var(inputs, true);
            auto logits = model->forward(input_var);

            // Reshape logits as a Variable op (tenzor::reshape) so the grad_fn
            // chain back to the model parameters is preserved. Using
            // logits.tensor().reshape(...) then re-wrapping the result in a
            // fresh Variable(..., true) severs the graph -- the re-wrap is a
            // leaf with no upstream grad_fn, so loss.backward() populates only
            // that leaf's grad and the model parameters receive ZERO grads,
            // leaving the loss flat across epochs (GptTextGenerationTrains).
            auto logits_flat = reshape(
                logits, {actual_batch * seq_len, vocab_size});
            auto targets_flat = targets.reshape({actual_batch * seq_len});

            auto loss = criterion(logits_flat, targets_flat);
            loss.backward();
            optimizer.step();

            epoch_loss += loss.tensor().item<float>();
            num_batches++;
        }
        const double avg_loss =
            static_cast<double>(epoch_loss) / std::max(1, num_batches);
        if (epoch == 0) initial_loss = avg_loss;
        final_loss = avg_loss;
        if (verbose) {
            std::cout << "epoch " << (epoch + 1) << "/" << epochs
                      << " loss=" << avg_loss << "\n";
        }
    }

    if (out_initial) *out_initial = initial_loss;
    if (out_final)   *out_final   = final_loss;
    return 0;
}

}  // namespace tenzor::examples::gpt_text_generation
