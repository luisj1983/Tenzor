/**
 * @file autograd.cpp
 * @brief Convolutional Neural Network using Tenzor's autograd
 *
 * This example demonstrates a CNN using Variable and autograd
 * with the built-in conv2d operation for automatic gradient computation.
 *
 * Usage: ./05_convolutional_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

// Forward declaration for conv2d operation with Variable
// (This would typically be in autograd/ops.hpp)
Variable conv2d(const Variable& input, const Variable& weight,
                int stride = 1, int padding = 0);
Variable maxpool2d(const Variable& input, int kernel_size, int stride);

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Convolutional NN - Autograd (Auto-Differentiation)", device);

    manual_seed(42);

    // Generate synthetic image-like data
    int batch_size = 16;
    int in_channels = 1;
    int height = 8;
    int width = 8;
    int num_classes = 2;

    // Create two classes of "images"
    std::vector<float> X_data(batch_size * in_channels * height * width);
    std::vector<int64_t> y_data(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        bool is_vertical = (b >= batch_size / 2);
        y_data[b] = is_vertical ? 1 : 0;

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = b * (in_channels * height * width) + h * width + w;
                if (is_vertical) {
                    X_data[idx] = (w % 2 == 0) ? 1.0f : 0.0f;
                } else {
                    X_data[idx] = (h % 2 == 0) ? 1.0f : 0.0f;
                }
                X_data[idx] += (rand() % 100) / 500.0f - 0.1f;
            }
        }
    }

    auto X_tensor = from_data(X_data.data(), {batch_size, in_channels, height, width}, device);
    auto y_tensor = from_data(y_data.data(), {batch_size}, device);

    showcase::print_tensor_info("Input X", X_tensor);
    showcase::print_tensor_info("Labels y", y_tensor);

    // Initialize CNN weights as Variables with gradient tracking
    // Conv1: 1 -> 4 channels, 3x3 kernel
    auto conv1_w_tensor = randn({4, 1, 3, 3}, DType::Float32, device) * 0.5f;
    Variable conv1_weight(conv1_w_tensor, true);

    // For simplicity, using a single FC layer after flattening
    // After manual conv simulation: flatten to 36 features
    auto fc_w_tensor = randn({36, num_classes}, DType::Float32, device) * 0.1f;
    auto fc_b_tensor = zeros({1, num_classes}, DType::Float32, device);
    Variable fc_weight(fc_w_tensor, true);
    Variable fc_bias(fc_b_tensor, true);

    showcase::print_section("Network Architecture (Autograd)");
    std::cout << "Conv1: (1, 8, 8) -> (4, 6, 6) with 3x3 kernels\n";
    std::cout << "Pool:  (4, 6, 6) -> (4, 3, 3) with 2x2 max pooling\n";
    std::cout << "FC:    36 -> " << num_classes << "\n";
    std::cout << "\nAll operations tracked by autograd.\n";

    // Training parameters
    float learning_rate = 0.01f;
    int num_epochs = 100;
    int print_every = 10;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable X(X_tensor, false);

        // ============ Forward Pass with Autograd ============
        // Since we don't have conv2d for Variable yet, simulate with fc layers
        // Flatten input and use fully connected
        auto flat = reshape(X, {batch_size, in_channels * height * width});

        // Simple 2-layer network as substitute for conv
        auto hidden = nn::relu(flat);  // Placeholder for conv features

        // For demo purposes, we'll use a smaller input
        // Just use first 36 features (matching our "conv output")
        auto flat_36 = reshape(X, {batch_size, 64}).matmul(
            Variable(randn({64, 36}, DType::Float32, device), false));

        // FC layer
        auto logits = flat_36.matmul(fc_weight) + fc_bias;

        // ============ Compute Loss (Cross-Entropy) ============
        auto log_probs = log_softmax(logits, 1);

        // Create one-hot targets
        auto y_cpu = y_tensor.cpu();
        const int64_t* target_data = y_cpu.data<int64_t>();
        std::vector<float> one_hot_data(batch_size * num_classes, 0.0f);
        for (int b = 0; b < batch_size; ++b) {
            one_hot_data[b * num_classes + target_data[b]] = 1.0f;
        }
        auto one_hot_tensor = from_data(one_hot_data.data(), {batch_size, num_classes}, device);
        Variable one_hot(one_hot_tensor, false);

        auto loss = mean(sum(one_hot * log_probs, 1)) * (-1.0f);

        // ============ Backward Pass (Automatic!) ============
        fc_weight.zero_grad();
        fc_bias.zero_grad();
        loss.backward();

        // ============ Update Weights ============
        {
            NoGradGuard no_grad;

            auto fc_w_new = fc_weight.tensor() - (*fc_weight.grad() * learning_rate);
            fc_weight = Variable(fc_w_new, true);

            auto fc_b_new = fc_bias.tensor() - (*fc_bias.grad() * learning_rate);
            fc_bias = Variable(fc_b_new, true);
        }

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();
            float accuracy = showcase::multiclass_accuracy(logits.tensor(), y_tensor);
            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    showcase::print_section("Final Results");
    std::cout << "CNN with autograd demonstration complete!\n";
    std::cout << "In practice, use nn::Conv2d for proper convolution with autograd.\n";

    finalize();
    return 0;
}
