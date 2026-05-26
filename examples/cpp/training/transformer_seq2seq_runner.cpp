/**
 * @file transformer_seq2seq_runner.cpp
 * @brief Implementation of the transformer-seq2seq autograd runner.
 *
 * NN.24: extracted from transformer_seq2seq.cpp's
 * train_sequence_classifier() training body so the regression test in
 * tests/examples/test_all_autograd_examples.cpp can drive the
 * Embedding + MultiheadAttention + LayerNorm + GELU + Adam pipeline
 * end-to-end and assert that backward actually moves the weights.
 */

#include "transformer_seq2seq_runner.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "tenzor/tenzor.hpp"

namespace tenzor::examples::transformer_seq2seq {

namespace {

using ::tenzor::Variable;
using ::tenzor::Tensor;
using ::tenzor::nn::Module;
using ::tenzor::nn::Embedding;
using ::tenzor::nn::MultiheadAttention;
using ::tenzor::nn::Linear;
using ::tenzor::nn::LayerNorm;
using ::tenzor::nn::GELU;
using ::tenzor::nn::Dropout;
using ::tenzor::nn::CrossEntropyLoss;

class AttentionSequenceClassifier : public Module {
public:
    AttentionSequenceClassifier(int64_t vocab_size, int64_t embed_dim,
                                 int64_t hidden_dim, int64_t num_classes,
                                 int64_t num_heads)
        : vocab_size_(vocab_size), embed_dim_(embed_dim) {
        embedding_ = std::make_shared<Embedding>(vocab_size, embed_dim);
        attention_ = std::make_shared<MultiheadAttention>(embed_dim, num_heads);
        fc1_       = std::make_shared<Linear>(embed_dim, hidden_dim);
        fc2_       = std::make_shared<Linear>(hidden_dim, num_classes);
        norm_      = std::make_shared<LayerNorm>(
            std::vector<int64_t>{embed_dim});
        gelu_      = std::make_shared<GELU>();
        dropout_   = std::make_shared<Dropout>(0.1f);
        register_module("embedding", embedding_);
        register_module("attention", attention_);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("norm", norm_);
        register_module("gelu", gelu_);
        register_module("dropout", dropout_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto embedded = embedding_->forward(x);
        auto [attn_out, _w] =
            attention_->forward(embedded, embedded, embedded);
        auto h = embedded + attn_out;
        h = norm_->forward(h);
        auto pooled = ::tenzor::mean(h.tensor(), 1);
        Variable pooled_var(pooled, h.requires_grad());
        auto ff = fc1_->forward(pooled_var);
        ff = gelu_->forward(ff);
        ff = dropout_->forward(ff);
        return fc2_->forward(ff);
    }

private:
    int64_t vocab_size_;
    int64_t embed_dim_;
    std::shared_ptr<Embedding> embedding_;
    std::shared_ptr<MultiheadAttention> attention_;
    std::shared_ptr<Linear> fc1_, fc2_;
    std::shared_ptr<LayerNorm> norm_;
    std::shared_ptr<GELU> gelu_;
    std::shared_ptr<Dropout> dropout_;
};

std::pair<Tensor, Tensor> generate_sequence_data(int num_samples, int seq_len,
                                                  int vocab_size,
                                                  int num_classes,
                                                  ::tenzor::Device device) {
    using namespace ::tenzor;
    std::vector<int64_t> X_data(num_samples * seq_len);
    std::vector<int64_t> y_data(num_samples);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> token_dist(1, vocab_size - 1);
    std::uniform_int_distribution<int> class_dist(0, num_classes - 1);
    for (int i = 0; i < num_samples; ++i) {
        const int label = class_dist(rng);
        y_data[i] = label;
        for (int j = 0; j < seq_len; ++j) {
            if (j % num_classes == label) {
                X_data[i * seq_len + j] = label + 1;
            } else {
                X_data[i * seq_len + j] = token_dist(rng);
            }
        }
    }
    auto X = from_data(X_data.data(), {num_samples, seq_len}, device);
    auto y = from_data(y_data.data(), {num_samples}, device);
    return {X, y};
}

}  // namespace

int run_transformer_seq2seq_training(int epochs,
                                      double* out_initial,
                                      double* out_final,
                                      ::tenzor::Device device,
                                      bool verbose) {
    using namespace ::tenzor;

    // Small workload so the regression test stays under ~1-2s on CPU.
    // The standalone exe still runs the bigger 500-sample / 20-epoch loop
    // in main().
    const int vocab_size  = 32;
    const int embed_dim   = 16;
    const int hidden_dim  = 32;
    const int num_classes = 4;
    const int seq_len     = 16;
    const int num_train   = 16;
    const int batch_size  = 2;
    const int num_heads   = 4;

    manual_seed(42);

    auto [X_train, y_train] = generate_sequence_data(
        num_train, seq_len, vocab_size, num_classes, device);

    auto model = std::make_shared<AttentionSequenceClassifier>(
        vocab_size, embed_dim, hidden_dim, num_classes, num_heads);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);
    CrossEntropyLoss criterion;

    double initial_loss = 0.0;
    double final_loss   = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        for (int i = 0; i < num_train; i += batch_size) {
            int end = std::min(i + batch_size, num_train);
            auto batch_X = X_train.slice(0, i, end);
            auto batch_y = y_train.slice(0, i, end);

            optimizer.zero_grad();
            Variable input(batch_X, true);
            auto output = model->forward(input);
            auto loss = criterion(output, batch_y);
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

}  // namespace tenzor::examples::transformer_seq2seq
