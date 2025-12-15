/**
 * @file tensor_only.cpp
 * @brief Convolutional Neural Network using raw Tensor operations
 *
 * This example demonstrates a simple CNN using only tensor operations
 * with manual convolution implementation for educational purposes.
 *
 * Note: Manual convolution is computationally expensive. In practice,
 * use the nn::Conv2d layer which uses optimized implementations.
 *
 * Usage: ./05_convolutional_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

// Simple 2D convolution (educational, not optimized)
// Input: (batch, in_channels, height, width)
// Kernel: (out_channels, in_channels, kH, kW)
// Output: (batch, out_channels, out_height, out_width)
Tensor conv2d_naive(const Tensor& input, const Tensor& kernel, int stride = 1, int padding = 0) {
    auto input_cpu = input.cpu();
    auto kernel_cpu = kernel.cpu();

    int batch = input.shape()[0];
    int in_channels = input.shape()[1];
    int in_h = input.shape()[2];
    int in_w = input.shape()[3];

    int out_channels = kernel.shape()[0];
    int kH = kernel.shape()[2];
    int kW = kernel.shape()[3];

    int out_h = (in_h + 2 * padding - kH) / stride + 1;
    int out_w = (in_w + 2 * padding - kW) / stride + 1;

    auto output = zeros({batch, out_channels, out_h, out_w}, DType::Float32, input.device());
    auto output_cpu = output.cpu();

    const float* in_data = input_cpu.data<float>();
    const float* ker_data = kernel_cpu.data<float>();
    float* out_data = output_cpu.data<float>();

    // Naive convolution (very slow but educational)
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx = b * (in_channels * in_h * in_w) +
                                                 ic * (in_h * in_w) + ih * in_w + iw;
                                    int ker_idx = oc * (in_channels * kH * kW) +
                                                  ic * (kH * kW) + kh * kW + kw;
                                    sum += in_data[in_idx] * ker_data[ker_idx];
                                }
                            }
                        }
                    }
                    int out_idx = b * (out_channels * out_h * out_w) +
                                  oc * (out_h * out_w) + oh * out_w + ow;
                    out_data[out_idx] = sum;
                }
            }
        }
    }

    return from_data(out_data, {batch, out_channels, out_h, out_w}, input.device());
}

// Max pooling 2D
Tensor maxpool2d_naive(const Tensor& input, int pool_size, int stride) {
    auto input_cpu = input.cpu();

    int batch = input.shape()[0];
    int channels = input.shape()[1];
    int in_h = input.shape()[2];
    int in_w = input.shape()[3];

    int out_h = (in_h - pool_size) / stride + 1;
    int out_w = (in_w - pool_size) / stride + 1;

    auto output = zeros({batch, channels, out_h, out_w}, DType::Float32, input.device());
    auto output_cpu = output.cpu();

    const float* in_data = input_cpu.data<float>();
    float* out_data = output_cpu.data<float>();

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int ph = 0; ph < pool_size; ++ph) {
                        for (int pw = 0; pw < pool_size; ++pw) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            int in_idx = b * (channels * in_h * in_w) +
                                         c * (in_h * in_w) + ih * in_w + iw;
                            max_val = std::max(max_val, in_data[in_idx]);
                        }
                    }
                    int out_idx = b * (channels * out_h * out_w) +
                                  c * (out_h * out_w) + oh * out_w + ow;
                    out_data[out_idx] = max_val;
                }
            }
        }
    }

    return from_data(out_data, {batch, channels, out_h, out_w}, input.device());
}

// ReLU activation
Tensor relu_tensor(const Tensor& x) {
    return clamp_min(x, 0.0f);
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Convolutional NN - Tensor Only (Educational)", device);

    manual_seed(42);

    // Generate synthetic image-like data: 8x8 "images"
    int batch_size = 16;
    int in_channels = 1;
    int height = 8;
    int width = 8;
    int num_classes = 2;

    // Create two classes of "images":
    // Class 0: Horizontal pattern
    // Class 1: Vertical pattern
    std::vector<float> X_data(batch_size * in_channels * height * width);
    std::vector<int64_t> y_data(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        bool is_vertical = (b >= batch_size / 2);
        y_data[b] = is_vertical ? 1 : 0;

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = b * (in_channels * height * width) + h * width + w;
                if (is_vertical) {
                    // Vertical stripes
                    X_data[idx] = (w % 2 == 0) ? 1.0f : 0.0f;
                } else {
                    // Horizontal stripes
                    X_data[idx] = (h % 2 == 0) ? 1.0f : 0.0f;
                }
                // Add noise
                X_data[idx] += (rand() % 100) / 500.0f - 0.1f;
            }
        }
    }

    auto X = from_data(X_data.data(), {batch_size, in_channels, height, width}, device);
    auto y = from_data(y_data.data(), {batch_size}, device);

    showcase::print_tensor_info("Input X", X);
    showcase::print_tensor_info("Labels y", y);

    // Initialize CNN weights
    // Conv1: 1 -> 4 channels, 3x3 kernel
    auto conv1_weight = randn({4, 1, 3, 3}, DType::Float32, device) * 0.5f;
    auto conv1_bias = zeros({4}, DType::Float32, device);

    // FC: flatten -> num_classes
    // After conv: (batch, 4, 6, 6) -> pool -> (batch, 4, 3, 3) -> flatten -> 36
    auto fc_weight = randn({36, num_classes}, DType::Float32, device) * 0.1f;
    auto fc_bias = zeros({1, num_classes}, DType::Float32, device);

    showcase::print_section("Network Architecture");
    std::cout << "Conv1: (1, 8, 8) -> Conv(3x3) -> (4, 6, 6)\n";
    std::cout << "Pool:  (4, 6, 6) -> MaxPool(2x2) -> (4, 3, 3)\n";
    std::cout << "FC:    36 -> " << num_classes << "\n";

    // Training parameters
    float learning_rate = 0.01f;
    int num_epochs = 100;
    int print_every = 10;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ============ Forward Pass ============
        // Conv1 + ReLU
        auto conv1_out = conv2d_naive(X, conv1_weight, 1, 0);  // (16, 4, 6, 6)
        auto relu1_out = relu_tensor(conv1_out);

        // MaxPool
        auto pool_out = maxpool2d_naive(relu1_out, 2, 2);  // (16, 4, 3, 3)

        // Flatten
        auto flat = pool_out.reshape({batch_size, 36});  // (16, 36)

        // FC layer
        auto logits = matmul(flat, fc_weight) + fc_bias;  // (16, 2)

        // Softmax and cross-entropy loss
        auto exp_logits = tenzor::exp(logits - tenzor::max(logits, 1, true));
        auto probs = exp_logits / tenzor::sum(exp_logits, 1, true);

        // Cross-entropy loss
        auto y_cpu = y.cpu();
        auto probs_cpu = probs.cpu();
        const int64_t* target_data = y_cpu.data<int64_t>();
        const float* prob_data = probs_cpu.data<float>();

        float loss = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            loss -= std::log(prob_data[b * num_classes + target_data[b]] + 1e-7f);
        }
        loss /= batch_size;

        // ============ Simplified Backward (only FC layer) ============
        // For educational purposes, we only update the FC layer
        // Full backprop through conv layers is complex

        // dL/dlogits = probs - one_hot
        std::vector<float> grad_data(batch_size * num_classes);
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_classes; ++c) {
                grad_data[b * num_classes + c] = prob_data[b * num_classes + c];
                if (c == target_data[b]) {
                    grad_data[b * num_classes + c] -= 1.0f;
                }
                grad_data[b * num_classes + c] /= batch_size;
            }
        }
        auto dL_dlogits = from_data(grad_data.data(), {batch_size, num_classes}, device);

        // dL/dW_fc = flat^T @ dL_dlogits
        auto dL_dW_fc = matmul(flat.transpose(0, 1), dL_dlogits);
        auto dL_db_fc = tenzor::sum(dL_dlogits, 0, true);

        // Update FC weights
        fc_weight = fc_weight - dL_dW_fc * learning_rate;
        fc_bias = fc_bias - dL_db_fc * learning_rate;

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float accuracy = showcase::multiclass_accuracy(logits, y);
            showcase::print_progress(epoch, num_epochs, loss, accuracy);
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    // Final forward pass
    auto conv1_out = conv2d_naive(X, conv1_weight, 1, 0);
    auto relu1_out = relu_tensor(conv1_out);
    auto pool_out = maxpool2d_naive(relu1_out, 2, 2);
    auto flat = pool_out.reshape({batch_size, 36});
    auto logits = matmul(flat, fc_weight) + fc_bias;

    float final_accuracy = showcase::multiclass_accuracy(logits, y);
    std::cout << "Final Accuracy: " << (final_accuracy * 100.0f) << "%\n\n";

    std::cout << "CNN demonstrated with manual tensor operations!\n";
    std::cout << "Note: This naive implementation is for education only.\n";
    std::cout << "Use nn::Conv2d for optimized convolutions.\n";

    finalize();
    return 0;
}
