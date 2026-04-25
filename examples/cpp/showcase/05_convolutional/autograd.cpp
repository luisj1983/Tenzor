/**
 * @file autograd.cpp
 * @brief Convolutional Neural Network using Tenzor's autograd
 *
 * This example demonstrates a CNN using Variable and autograd
 * with F::conv2d and F::max_pool2d - the low-level functional API
 * that wires raw kernels straight into the autograd graph.
 *
 * Usage: ./05_convolutional_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <tenzor/nn/functional.hpp>
#include <cstdlib>

using namespace tenzor;
namespace F = tenzor::nn::functional;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Convolutional NN - Autograd (Auto-Differentiation)", device);

    manual_seed(42);

    // Synthetic 8x8 image classification: vertical vs. horizontal stripes
    int batch_size = 16;
    int in_channels = 1;
    int height = 8;
    int width = 8;
    int num_classes = 2;

    std::vector<float> X_data(batch_size * in_channels * height * width);
    std::vector<int64_t> y_data(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        bool is_vertical = (b >= batch_size / 2);
        y_data[b] = is_vertical ? 1 : 0;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = b * (in_channels * height * width) + h * width + w;
                X_data[idx] = is_vertical ? ((w % 2 == 0) ? 1.0f : 0.0f)
                                          : ((h % 2 == 0) ? 1.0f : 0.0f);
                X_data[idx] += (rand() % 100) / 500.0f - 0.1f;
            }
        }
    }

    auto X_tensor = from_data(X_data.data(), {batch_size, in_channels, height, width}, device);
    auto y_tensor = from_data(y_data.data(), {batch_size}, device);

    showcase::print_tensor_info("Input X", X_tensor);
    showcase::print_tensor_info("Labels y", y_tensor);

    // Conv weights: 1 -> 4 channels, 3x3, padding=1 -> preserves H,W
    // He-style init: scale by sqrt(2 / fan_in)
    float conv_scale = std::sqrt(2.0f / (1 * 3 * 3));
    Variable conv_w(randn({4, 1, 3, 3}, DType::Float32, device) * conv_scale, true);
    Variable conv_b(zeros({4}, DType::Float32, device), true);

    // After conv + relu + max_pool2d(2): (B, 4, 4, 4) -> flatten to 64
    int fc_in = 4 * 4 * 4;
    float fc_scale = std::sqrt(2.0f / fc_in);
    Variable fc_w(randn({fc_in, num_classes}, DType::Float32, device) * fc_scale, true);
    Variable fc_b(zeros({num_classes}, DType::Float32, device), true);

    showcase::print_section("Network Architecture (Autograd)");
    std::cout << "Conv2d(1 -> 4, 3x3, pad=1) -> ReLU -> MaxPool2d(2x2)\n";
    std::cout << "Flatten -> Linear(64 -> " << num_classes << ")\n";
    std::cout << "\nAll operations tracked by autograd via F::conv2d / F::max_pool2d.\n";

    // Pre-build targets (stable across epochs)
    auto y_cpu = y_tensor.cpu();
    const int64_t* target_data = y_cpu.data<int64_t>();
    std::vector<float> one_hot_data(batch_size * num_classes, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        one_hot_data[b * num_classes + target_data[b]] = 1.0f;
    }
    auto one_hot_tensor = from_data(one_hot_data.data(), {batch_size, num_classes}, device);

    float learning_rate = 0.05f;
    int num_epochs = 50;
    int print_every = 5;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable X(X_tensor, false);

        // Forward: conv -> relu -> pool -> flatten -> fc
        auto conv_out = F::conv2d(X, conv_w, conv_b, {1, 1}, {1, 1});
        auto relu_out = nn::relu(conv_out);
        auto pool_out = F::max_pool2d(relu_out, {2, 2});
        auto flat = reshape(pool_out, {batch_size, fc_in});
        auto logits = matmul(flat, fc_w) + fc_b;

        // Cross-entropy via log_softmax + one_hot
        auto log_probs = log_softmax(logits, 1);
        Variable one_hot(one_hot_tensor, false);
        auto loss = mean(sum(one_hot * log_probs, 1)) * (-1.0f);

        conv_w.zero_grad(); conv_b.zero_grad();
        fc_w.zero_grad();   fc_b.zero_grad();
        loss.backward();

        {
            NoGradGuard no_grad;
            conv_w = Variable(conv_w.tensor() - (*conv_w.grad() * learning_rate), true);
            conv_b = Variable(conv_b.tensor() - (*conv_b.grad() * learning_rate), true);
            fc_w   = Variable(fc_w.tensor()   - (*fc_w.grad()   * learning_rate), true);
            fc_b   = Variable(fc_b.tensor()   - (*fc_b.grad()   * learning_rate), true);
        }

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();
            float accuracy = showcase::multiclass_accuracy(logits.tensor(), y_tensor);
            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    showcase::print_section("Final Results");

    // Inference pass to report final accuracy
    {
        Variable X(X_tensor, false);
        auto conv_out = F::conv2d(X, conv_w, conv_b, {1, 1}, {1, 1});
        auto relu_out = nn::relu(conv_out);
        auto pool_out = F::max_pool2d(relu_out, {2, 2});
        auto flat = reshape(pool_out, {batch_size, fc_in});
        auto logits = matmul(flat, fc_w) + fc_b;
        float accuracy = showcase::multiclass_accuracy(logits.tensor(), y_tensor);
        std::cout << "Final Accuracy: " << (accuracy * 100.0f) << "%\n\n";
    }

    std::cout << "CNN demonstrated with autograd!\n";
    std::cout << "F::conv2d and F::max_pool2d plug straight into the autograd graph -\n";
    std::cout << "no manual im2col, no manual pooling backward.\n";

    finalize();
    return 0;
}
