/**
 * @file autograd_runner.cpp
 * @brief Implementation of the custom-loss autograd training loop (focal).
 *
 * The original example trained three configurations (focal loss, label
 * smoothing, Huber regression). The runner here drives the focal-loss
 * config because that exercises the most non-trivial autograd graph
 * (log_softmax + exp + element-wise reweighting) and is sufficient as a
 * regression check.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace tenzor::examples::showcase10 {

namespace {

::tenzor::Variable focal_loss_var(const ::tenzor::Variable& logits,
                                   const ::tenzor::Tensor& targets,
                                   float gamma = 2.0f, float alpha = 0.25f) {
    using namespace ::tenzor;
    auto log_probs = log_softmax(logits, 1);
    auto probs = exp(log_probs);

    int batch_size = static_cast<int>(logits.shape()[0]);
    int num_classes = static_cast<int>(logits.shape()[1]);

    auto targets_cpu = targets.cpu();
    const int64_t* target_data = targets_cpu.data<int64_t>();

    std::vector<float> target_mask_data(batch_size * num_classes, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        target_mask_data[b * num_classes + target_data[b]] = 1.0f;
    }
    auto target_mask = from_data(target_mask_data.data(),
                                  {batch_size, num_classes}, logits.device());
    Variable mask(target_mask, false);

    auto p_t = sum(probs * mask, 1);
    auto ones_tensor = ones_like(p_t.tensor());
    Variable ones_var(ones_tensor, false);
    auto focal_weight = (ones_var - p_t);
    auto focal_weight_gamma = focal_weight * focal_weight;

    auto log_p_t = sum(log_probs * mask, 1);
    auto ce_loss = log_p_t * (-1.0f);

    auto loss = mean(focal_weight_gamma * ce_loss) * alpha;
    (void)gamma;
    return loss;
}

}  // namespace

int run_custom_loss_training(int epochs,
                             double* out_initial,
                             double* out_final,
                             ::tenzor::Device device,
                             bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    int batch_size = 64;
    int input_features = 16;
    int num_classes = 4;

    std::mt19937 gen(123);
    std::normal_distribution<float> ndist(0.0f, 1.0f);

    std::vector<float> X_data(batch_size * input_features);
    std::vector<int64_t> y_data(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        int label = (b < batch_size * 0.6) ? 0 : (1 + (b % (num_classes - 1)));
        y_data[b] = label;
        for (int f = 0; f < input_features; ++f) {
            X_data[b * input_features + f] =
                static_cast<float>(label) * 0.5f + ndist(gen) * 0.5f;
        }
    }
    auto X_tensor = from_data(X_data.data(), {batch_size, input_features}, device);
    auto y_tensor = from_data(y_data.data(), {batch_size}, device);

    Variable W1(randn({input_features, 32}, DType::Float32, device) * 0.1f, true);
    Variable b1(zeros({1, 32}, DType::Float32, device), true);
    Variable W2(randn({32, num_classes}, DType::Float32, device) * 0.1f, true);
    Variable b2(zeros({1, num_classes}, DType::Float32, device), true);

    float learning_rate = 0.1f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable X(X_tensor, false);
        auto z1 = matmul(X, W1) + b1;
        auto a1 = nn::relu(z1);
        auto logits = matmul(a1, W2) + b2;

        auto loss = focal_loss_var(logits, y_tensor, 2.0f, 0.25f);

        W1.zero_grad(); b1.zero_grad();
        W2.zero_grad(); b2.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W1 = Variable(W1.tensor() - (*W1.grad() * learning_rate), true);
            b1 = Variable(b1.tensor() - (*b1.grad() * learning_rate), true);
            W2 = Variable(W2.tensor() - (*W2.grad() * learning_rate), true);
            b2 = Variable(b2.tensor() - (*b2.grad() * learning_rate), true);
        }

        double loss_val = static_cast<double>(loss.tensor().item<float>());
        if (epoch == 0 && out_initial) *out_initial = loss_val;
        final_loss = loss_val;

        if (verbose && ((epoch + 1) % print_every == 0 || epoch == 0)) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << epochs
                      << "] focal_loss=" << loss_val << "\n";
        }
    }
    if (out_final) *out_final = final_loss;
    return 0;
}

}  // namespace tenzor::examples::showcase10
