/**
 * @file autograd_runner.cpp
 * @brief Implementation of the dropout-regularization autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <random>
#include <vector>

namespace tenzor::examples::showcase09 {

namespace {

::tenzor::Variable dropout_var(const ::tenzor::Variable& x, float drop_prob,
                                bool training, std::mt19937& gen) {
    using namespace ::tenzor;
    if (!training || drop_prob == 0.0f) return x;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    int64_t numel = x.tensor().numel();
    std::vector<float> mask_data(numel);
    float scale = 1.0f / (1.0f - drop_prob);
    for (int64_t i = 0; i < numel; ++i) {
        mask_data[i] = (dist(gen) > drop_prob) ? scale : 0.0f;
    }

    std::vector<int64_t> shape_vec(x.shape().begin(), x.shape().end());
    auto mask_tensor = from_data(mask_data.data(), shape_vec, x.device());
    Variable mask(mask_tensor, false);
    return x * mask;
}

}  // namespace

int run_dropout_training(int epochs,
                         double* out_initial,
                         double* out_final,
                         ::tenzor::Device device,
                         bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);
    std::mt19937 gen(42);
    std::normal_distribution<float> ndist(0.0f, 1.0f);

    int batch_size = 32;
    int input_features = 64;
    int hidden_features = 128;
    int num_classes = 4;
    float dropout_prob = 0.5f;

    std::vector<float> X_data(batch_size * input_features);
    std::vector<int64_t> y_data(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        int label = b % num_classes;
        y_data[b] = label;
        for (int f = 0; f < input_features; ++f) {
            X_data[b * input_features + f] =
                static_cast<float>(label) * 0.5f + ndist(gen) * 0.3f;
        }
    }

    auto X_tensor = from_data(X_data.data(), {batch_size, input_features}, device);
    auto y_tensor = from_data(y_data.data(), {batch_size}, device);

    Variable W1(randn({input_features, hidden_features}, DType::Float32, device) * 0.1f, true);
    Variable b1(zeros({1, hidden_features}, DType::Float32, device), true);
    Variable W2(randn({hidden_features, hidden_features}, DType::Float32, device) * 0.1f, true);
    Variable b2(zeros({1, hidden_features}, DType::Float32, device), true);
    Variable W3(randn({hidden_features, num_classes}, DType::Float32, device) * 0.1f, true);
    Variable b3(zeros({1, num_classes}, DType::Float32, device), true);

    std::vector<float> one_hot_data(batch_size * num_classes, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        one_hot_data[b * num_classes + y_data[b]] = 1.0f;
    }
    auto one_hot_tensor = from_data(one_hot_data.data(), {batch_size, num_classes}, device);

    float learning_rate = 0.01f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        bool training = true;
        Variable X(X_tensor, false);

        auto z1 = matmul(X, W1) + b1;
        auto a1_pre = nn::relu(z1);
        auto a1 = dropout_var(a1_pre, dropout_prob, training, gen);

        auto z2 = matmul(a1, W2) + b2;
        auto a2_pre = nn::relu(z2);
        auto a2 = dropout_var(a2_pre, dropout_prob, training, gen);

        auto logits = matmul(a2, W3) + b3;
        auto log_probs = log_softmax(logits, 1);

        Variable one_hot(one_hot_tensor, false);
        auto loss = mean(sum(one_hot * log_probs, 1)) * (-1.0f);

        W1.zero_grad(); b1.zero_grad();
        W2.zero_grad(); b2.zero_grad();
        W3.zero_grad(); b3.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W1 = Variable(W1.tensor() - (*W1.grad() * learning_rate), true);
            b1 = Variable(b1.tensor() - (*b1.grad() * learning_rate), true);
            W2 = Variable(W2.tensor() - (*W2.grad() * learning_rate), true);
            b2 = Variable(b2.tensor() - (*b2.grad() * learning_rate), true);
            W3 = Variable(W3.tensor() - (*W3.grad() * learning_rate), true);
            b3 = Variable(b3.tensor() - (*b3.grad() * learning_rate), true);
        }

        double loss_val = static_cast<double>(loss.tensor().cpu().item<float>());
        if (epoch == 0 && out_initial) *out_initial = loss_val;
        final_loss = loss_val;

        if (verbose && ((epoch + 1) % print_every == 0 || epoch == 0)) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << epochs
                      << "] loss=" << loss_val << "\n";
        }
    }
    if (out_final) *out_final = final_loss;
    return 0;
}

}  // namespace tenzor::examples::showcase09
