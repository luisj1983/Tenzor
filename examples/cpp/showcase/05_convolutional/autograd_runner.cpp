/**
 * @file autograd_runner.cpp
 * @brief Implementation of the CNN-autograd showcase training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <tenzor/nn/functional.hpp>

#include <cmath>
#include <cstdlib>
#include <vector>

namespace tenzor::examples::showcase05 {

int run_convolutional_training(int epochs,
                               double* out_initial,
                               double* out_final,
                               ::tenzor::Device device,
                               bool verbose) {
    using namespace ::tenzor;
    namespace F = ::tenzor::nn::functional;

    manual_seed(42);

    int batch_size = 16;
    int in_channels = 1;
    int height = 8;
    int width = 8;
    int num_classes = 2;

    std::vector<float> X_data(batch_size * in_channels * height * width);
    std::vector<int64_t> y_data(batch_size);

    // Use a deterministic per-element noise so the runner's behavior doesn't
    // drift with the global rand() state set by other tests in the binary.
    auto pseudo_rand = [](int seed) {
        return ((seed * 1103515245u + 12345u) % 100u) / 500.0f - 0.1f;
    };

    for (int b = 0; b < batch_size; ++b) {
        bool is_vertical = (b >= batch_size / 2);
        y_data[b] = is_vertical ? 1 : 0;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = b * (in_channels * height * width) + h * width + w;
                X_data[idx] = is_vertical ? ((w % 2 == 0) ? 1.0f : 0.0f)
                                          : ((h % 2 == 0) ? 1.0f : 0.0f);
                X_data[idx] += pseudo_rand(idx);
            }
        }
    }

    auto X_tensor = from_data(X_data.data(), {batch_size, in_channels, height, width}, device);
    auto y_tensor = from_data(y_data.data(), {batch_size}, device);

    float conv_scale = std::sqrt(2.0f / (1 * 3 * 3));
    Variable conv_w(randn({4, 1, 3, 3}, DType::Float32, device) * conv_scale, true);
    Variable conv_b(zeros({4}, DType::Float32, device), true);

    int fc_in = 4 * 4 * 4;
    float fc_scale = std::sqrt(2.0f / fc_in);
    Variable fc_w(randn({fc_in, num_classes}, DType::Float32, device) * fc_scale, true);
    Variable fc_b(zeros({num_classes}, DType::Float32, device), true);

    auto y_cpu = y_tensor.cpu();
    const int64_t* target_data = y_cpu.data<int64_t>();
    std::vector<float> one_hot_data(batch_size * num_classes, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        one_hot_data[b * num_classes + target_data[b]] = 1.0f;
    }
    auto one_hot_tensor = from_data(one_hot_data.data(), {batch_size, num_classes}, device);

    float learning_rate = 0.05f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable X(X_tensor, false);

        auto conv_out = F::conv2d(X, conv_w, conv_b, {1, 1}, {1, 1});
        auto relu_out = nn::relu(conv_out);
        auto pool_out = F::max_pool2d(relu_out, {2, 2});
        auto flat = reshape(pool_out, {batch_size, fc_in});
        auto logits = matmul(flat, fc_w) + fc_b;

        auto log_probs = log_softmax(logits, 1);
        Variable one_hot(one_hot_tensor, false);
        auto loss = mean(sum(one_hot * log_probs, 1)) * (-1.0f);

        conv_w.zero_grad(); conv_b.zero_grad();
        fc_w.zero_grad();   fc_b.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            conv_w = Variable(conv_w.tensor() - (*conv_w.grad() * learning_rate), true);
            conv_b = Variable(conv_b.tensor() - (*conv_b.grad() * learning_rate), true);
            fc_w   = Variable(fc_w.tensor()   - (*fc_w.grad()   * learning_rate), true);
            fc_b   = Variable(fc_b.tensor()   - (*fc_b.grad()   * learning_rate), true);
        }

        double loss_val = static_cast<double>(loss.tensor().item<float>());
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

}  // namespace tenzor::examples::showcase05
