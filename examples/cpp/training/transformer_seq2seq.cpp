/**
 * @file transformer_seq2seq.cpp
 * @brief Transformer Components for Sequence Modeling
 *
 * This example demonstrates:
 * - Embedding layers for token encoding
 * - MultiheadAttention for sequence modeling
 * - NLLLoss for sequence prediction
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>

#include "tenzor/tenzor.hpp"
#include "transformer_seq2seq_runner.hpp"

using namespace tenzor;
using namespace tenzor::nn;

// ============================================================================
// Simple Sequence Model with Attention
// ============================================================================

class AttentionSequenceClassifier : public Module {
public:
    AttentionSequenceClassifier(int64_t vocab_size, int64_t embed_dim, int64_t hidden_dim,
                                 int64_t num_classes, int64_t num_heads = 4)
        : vocab_size_(vocab_size), embed_dim_(embed_dim) {

        // Embedding layer
        embedding_ = std::make_shared<Embedding>(vocab_size, embed_dim);

        // Self-attention
        attention_ = std::make_shared<MultiheadAttention>(embed_dim, num_heads);

        // Feed-forward
        fc1_ = std::make_shared<Linear>(embed_dim, hidden_dim);
        fc2_ = std::make_shared<Linear>(hidden_dim, num_classes);

        // Layer norm
        norm_ = std::make_shared<LayerNorm>(std::vector<int64_t>{embed_dim});

        // GELU
        gelu_ = std::make_shared<GELU>();

        // Dropout
        dropout_ = std::make_shared<Dropout>(0.1f);

        register_module("embedding", embedding_);
        register_module("attention", attention_);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("norm", norm_);
        register_module("gelu", gelu_);
        register_module("dropout", dropout_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        // x: [batch, seq_len] - token indices
        auto embedded = embedding_->forward(x);  // [batch, seq_len, embed_dim]

        // Self-attention (returns pair<output, weights>)
        auto [attn_out, _weights] = attention_->forward(embedded, embedded, embedded);

        // Residual + norm
        auto h = embedded + attn_out;
        h = norm_->forward(h);

        // Mean pooling over sequence
        auto pooled = tenzor::mean(h.tensor(), 1);  // [batch, embed_dim]
        Variable pooled_var(pooled, h.requires_grad());

        // Feed-forward
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
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
    std::shared_ptr<LayerNorm> norm_;
    std::shared_ptr<GELU> gelu_;
    std::shared_ptr<Dropout> dropout_;
};

// ============================================================================
// Data Generation
// ============================================================================

std::pair<Tensor, Tensor> generate_sequence_data(int num_samples, int seq_len,
                                                  int vocab_size, int num_classes,
                                                  Device device) {
    std::vector<int64_t> X_data(num_samples * seq_len);
    std::vector<int64_t> y_data(num_samples);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> token_dist(1, vocab_size - 1);
    std::uniform_int_distribution<int> class_dist(0, num_classes - 1);

    for (int i = 0; i < num_samples; ++i) {
        int label = class_dist(rng);
        y_data[i] = label;

        // Create sequences with class-specific patterns
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

// ============================================================================
// Training
// ============================================================================

void train_sequence_classifier(Device device) {
    // NN.24: the inner training body is shared with the regression test in
    // tests/examples/test_all_autograd_examples.cpp via the runner exposed
    // by transformer_seq2seq_runner.hpp. The standalone exe additionally
    // runs the longer 20-epoch / 500-sample loop below + a validation loop;
    // both paths drive the same model + optimizer code, so a backward
    // regression caught by the test also breaks this exe.
    std::cout << "\n=== Training Attention-based Sequence Classifier ===\n\n";

    {
        double init_loss = 0.0;
        double final_loss = 0.0;
        const int runner_epochs = 5;
        examples::transformer_seq2seq::run_transformer_seq2seq_training(
            runner_epochs, &init_loss, &final_loss, device, /*verbose=*/false);
        std::cout << "[runner] short-loop training: initial=" << init_loss
                  << " final=" << final_loss << " over " << runner_epochs
                  << " epochs\n\n";
    }

    int vocab_size = 100;
    int embed_dim = 64;
    int hidden_dim = 128;
    int num_classes = 5;
    int seq_len = 16;
    int num_train = 500;
    int num_val = 100;
    int batch_size = 32;
    int num_epochs = 20;

    auto [X_train, y_train] = generate_sequence_data(num_train, seq_len, vocab_size,
                                                      num_classes, device);
    auto [X_val, y_val] = generate_sequence_data(num_val, seq_len, vocab_size,
                                                  num_classes, device);

    auto model = std::make_shared<AttentionSequenceClassifier>(vocab_size, embed_dim,
                                                                hidden_dim, num_classes, 4);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);
    optim::StepLR scheduler(optimizer, 10, 0.1);

    CrossEntropyLoss criterion;

    std::cout << "Configuration:\n";
    std::cout << "  Model: Embedding + MultiheadAttention + MLP\n";
    std::cout << "  Vocab size: " << vocab_size << "\n";
    std::cout << "  Embed dim: " << embed_dim << "\n";
    std::cout << "  Classes: " << num_classes << "\n";
    std::cout << "  Seq len: " << seq_len << "\n";
    std::cout << "  Optimizer: Adam (lr=0.001)\n";
    std::cout << "  Scheduler: StepLR (step=10, gamma=0.1)\n\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
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

        scheduler.step();

        // Validation
        model->eval();
        int correct = 0;
        int total = 0;

        for (int i = 0; i < num_val; i += batch_size) {
            int end = std::min(i + batch_size, num_val);
            auto batch_X = X_val.slice(0, i, end);
            auto batch_y = y_val.slice(0, i, end);

            Variable input(batch_X, false);
            auto output = model->forward(input);

            auto preds = tenzor::argmax(output.tensor(), 1);
            auto preds_cpu = preds.cpu();
            auto labels_cpu = batch_y.cpu();

            const int64_t* pred_data = preds_cpu.data<int64_t>();
            const int64_t* label_data = labels_cpu.data<int64_t>();

            for (int64_t j = 0; j < preds_cpu.shape()[0]; ++j) {
                if (pred_data[j] == label_data[j]) correct++;
                total++;
            }
        }

        float val_acc = 100.0f * correct / total;

        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            std::cout << "Epoch " << std::setw(2) << (epoch + 1) << "/" << num_epochs
                      << " | Loss: " << std::fixed << std::setprecision(4)
                      << (epoch_loss / num_batches)
                      << " | Val Acc: " << std::setprecision(2) << val_acc << "%"
                      << " | LR: " << std::setprecision(6) << scheduler.get_last_lr()
                      << "\n";
        }
    }
}

void demo_embedding(Device device) {
    std::cout << "\n=== Embedding Layer Demo ===\n\n";

    int vocab_size = 1000;
    int embed_dim = 128;

    auto embedding = std::make_shared<Embedding>(vocab_size, embed_dim);
    embedding->to(device);

    std::vector<int64_t> tokens = {1, 50, 100, 500, 999};
    auto input = from_data(tokens.data(), {1, 5}, device);
    Variable input_var(input, false);

    auto output = embedding->forward(input_var);

    std::cout << "Embedding(vocab_size=" << vocab_size << ", embed_dim=" << embed_dim << "):\n";
    std::cout << "  Input shape: [1, 5]\n";
    std::cout << "  Output shape: [" << output.shape()[0] << ", "
              << output.shape()[1] << ", " << output.shape()[2] << "]\n";
    std::cout << "  Each token mapped to " << embed_dim << "-dim vector\n";
}

void demo_cross_entropy(Device device) {
    std::cout << "\n=== CrossEntropyLoss Demo ===\n\n";

    int batch_size = 4;
    int num_classes = 5;

    // Create logits
    auto logits = randn({batch_size, num_classes}, DType::Float32, device);
    Variable logits_var(logits, true);

    // Create targets
    std::vector<int64_t> targets_data = {0, 1, 2, 3};
    auto targets = from_data(targets_data.data(), {batch_size}, device);

    CrossEntropyLoss criterion;
    auto loss = criterion(logits_var, targets);

    std::cout << "CrossEntropyLoss:\n";
    std::cout << "  Input (logits): [" << batch_size << ", " << num_classes << "]\n";
    std::cout << "  Targets: [" << batch_size << "]\n";
    std::cout << "  Loss: " << loss.tensor().item<float>() << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    initialize();

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
    std::cout << "   Transformer Sequence Modeling - Components         \n";
    std::cout << "   Backend: " << device.to_string() << "\n";
    std::cout << "======================================================\n";

    std::cout << "\nComponents demonstrated:\n";
    std::cout << "  Layers: Embedding, MultiheadAttention, LayerNorm, Linear\n";
    std::cout << "  Activations: GELU\n";
    std::cout << "  Losses: CrossEntropyLoss\n";
    std::cout << "  Optimizers: Adam\n";
    std::cout << "  Schedulers: StepLR\n";

    try {
        train_sequence_classifier(device);
        demo_embedding(device);
        demo_cross_entropy(device);

        std::cout << "\n======================================================\n";
        std::cout << "   All transformer examples completed successfully!   \n";
        std::cout << "======================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    finalize();
    return 0;
}
