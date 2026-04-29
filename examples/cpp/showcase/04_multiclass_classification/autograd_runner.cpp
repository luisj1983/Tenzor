/**
 * @file autograd_runner.cpp
 * @brief Implementation of the multiclass-classification autograd training.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <vector>

namespace tenzor::examples::showcase04 {

int run_multiclass_training(int epochs,
                            double* out_initial,
                            double* out_final,
                            ::tenzor::Device device,
                            bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    int samples_per_class = 50;
    int num_classes = 3;
    int num_samples = samples_per_class * num_classes;
    int num_features = 2;

    auto class0_x = randn({samples_per_class, 2}, DType::Float32, device);
    class0_x = class0_x + from_data(std::vector<float>{0.0f, 3.0f}.data(), {1, 2}, device);
    auto class1_x = randn({samples_per_class, 2}, DType::Float32, device);
    class1_x = class1_x + from_data(std::vector<float>{-3.0f, -1.0f}.data(), {1, 2}, device);
    auto class2_x = randn({samples_per_class, 2}, DType::Float32, device);
    class2_x = class2_x + from_data(std::vector<float>{3.0f, -1.0f}.data(), {1, 2}, device);

    auto X_tensor = cat({class0_x, class1_x, class2_x}, 0);

    std::vector<int64_t> label_data;
    for (int c = 0; c < num_classes; ++c) {
        for (int i = 0; i < samples_per_class; ++i) label_data.push_back(c);
    }
    auto y_tensor = from_data(label_data.data(), {num_samples}, device);

    Variable W(randn({num_features, num_classes}, DType::Float32, device) * 0.1f, true);
    Variable b(zeros({1, num_classes}, DType::Float32, device), true);

    float learning_rate = 0.1f;
    int print_every = std::max(1, epochs / 10);

    // Pre-build one-hot tensor (constant across epochs)
    std::vector<float> one_hot_data(num_samples * num_classes, 0.0f);
    for (int i = 0; i < num_samples; ++i) {
        one_hot_data[i * num_classes + label_data[i]] = 1.0f;
    }
    auto one_hot_tensor = from_data(one_hot_data.data(), {num_samples, num_classes}, device);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable X(X_tensor, false);

        auto z = matmul(X, W) + b;
        auto log_probs = log_softmax(z, 1);

        Variable one_hot(one_hot_tensor, false);
        auto loss_per_sample = one_hot * log_probs;
        auto loss = mean(sum(loss_per_sample, 1)) * (-1.0f);

        W.zero_grad(); b.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W = Variable(W.tensor() - (*W.grad() * learning_rate), true);
            b = Variable(b.tensor() - (*b.grad() * learning_rate), true);
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

}  // namespace tenzor::examples::showcase04
