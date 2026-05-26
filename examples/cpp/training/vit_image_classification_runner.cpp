/**
 * @file vit_image_classification_runner.cpp
 * @brief Implementation of the ViT-classification autograd runner.
 *
 * KK.27: extracted from vit_image_classification.cpp's training body so
 * the regression test in tests/examples/test_all_autograd_examples.cpp
 * can drive the MultiheadAttention + LayerNorm + GELU + AdamW pipeline
 * end-to-end and assert that backward actually moves the weights.
 */

#include "vit_image_classification_runner.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "tenzor/tenzor.hpp"

namespace tenzor::examples::vit_image_classification {

namespace {

using ::tenzor::Variable;
using ::tenzor::Tensor;
using ::tenzor::nn::Module;
using ::tenzor::nn::Linear;
using ::tenzor::nn::MultiheadAttention;
using ::tenzor::nn::LayerNorm;
using ::tenzor::nn::GELU;
using ::tenzor::nn::Dropout;
using ::tenzor::nn::CrossEntropyLoss;

class TransformerClassifier : public Module {
public:
    TransformerClassifier(int64_t input_dim, int64_t hidden_dim,
                          int64_t num_classes, int64_t num_heads) {
        fc_in_     = std::make_shared<Linear>(input_dim, hidden_dim);
        attention_ = std::make_shared<MultiheadAttention>(hidden_dim, num_heads);
        norm1_     = std::make_shared<LayerNorm>(
            std::vector<int64_t>{hidden_dim});
        norm2_     = std::make_shared<LayerNorm>(
            std::vector<int64_t>{hidden_dim});
        gelu_      = std::make_shared<GELU>();
        fc1_       = std::make_shared<Linear>(hidden_dim, hidden_dim * 4);
        fc2_       = std::make_shared<Linear>(hidden_dim * 4, hidden_dim);
        fc_out_    = std::make_shared<Linear>(hidden_dim, num_classes);
        dropout_   = std::make_shared<Dropout>(0.1f);
        register_module("fc_in", fc_in_);
        register_module("attention", attention_);
        register_module("norm1", norm1_);
        register_module("norm2", norm2_);
        register_module("gelu", gelu_);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("fc_out", fc_out_);
        register_module("dropout", dropout_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = fc_in_->forward(x);
        auto h_tensor = h.tensor();
        const int64_t batch_size = h_tensor.shape()[0];
        const int64_t hidden_dim = h_tensor.shape()[1];

        auto h_3d = h_tensor.reshape({batch_size, 1, hidden_dim});
        Variable h_seq(h_3d, h.requires_grad());

        auto [attn_out, _w] =
            attention_->forward(h_seq, h_seq, h_seq);
        attn_out = dropout_->forward(attn_out);

        auto attn_tensor = attn_out.tensor().reshape({batch_size, hidden_dim});
        Variable attn_2d(attn_tensor, attn_out.requires_grad());

        auto h_res = h + attn_2d;
        h_res = norm1_->forward(h_res);

        auto ff = fc1_->forward(h_res);
        ff = gelu_->forward(ff);
        ff = dropout_->forward(ff);
        ff = fc2_->forward(ff);

        auto out = h_res + ff;
        out = norm2_->forward(out);
        return fc_out_->forward(out);
    }

private:
    std::shared_ptr<Linear> fc_in_;
    std::shared_ptr<MultiheadAttention> attention_;
    std::shared_ptr<LayerNorm> norm1_, norm2_;
    std::shared_ptr<GELU> gelu_;
    std::shared_ptr<Linear> fc1_, fc2_, fc_out_;
    std::shared_ptr<Dropout> dropout_;
};

std::pair<Tensor, Tensor> generate_classification_data(
    int num_samples, int input_dim, int num_classes, ::tenzor::Device device) {
    using namespace ::tenzor;
    std::vector<float> X_data(num_samples * input_dim);
    std::vector<int64_t> y_data(num_samples);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> class_dist(0, num_classes - 1);
    for (int i = 0; i < num_samples; ++i) {
        const int label = class_dist(rng);
        y_data[i] = label;
        const float bias = static_cast<float>(label) * 0.5f;
        for (int j = 0; j < input_dim; ++j) {
            X_data[i * input_dim + j] =
                dist(rng) + bias * (j % 10 == label ? 1.0f : 0.0f);
        }
    }
    auto X = from_data(X_data.data(), {num_samples, input_dim}, device);
    auto y = from_data(y_data.data(), {num_samples}, device);
    return {X, y};
}

}  // namespace

int run_vit_classification_training(int epochs,
                                     double* out_initial,
                                     double* out_final,
                                     ::tenzor::Device device,
                                     bool verbose) {
    using namespace ::tenzor;

    const int num_classes = 4;
    const int input_dim   = 16;
    const int hidden_dim  = 32;
    const int num_train   = 32;
    const int batch_size  = 8;

    manual_seed(42);

    auto [X_train, y_train] = generate_classification_data(
        num_train, input_dim, num_classes, device);

    auto model = std::make_shared<TransformerClassifier>(
        input_dim, hidden_dim, num_classes, /*num_heads=*/2);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::AdamW optimizer(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
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

}  // namespace tenzor::examples::vit_image_classification
