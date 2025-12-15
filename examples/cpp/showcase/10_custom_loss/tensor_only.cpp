/**
 * @file tensor_only.cpp
 * @brief Custom Loss Functions using raw Tensor operations only
 *
 * This example demonstrates creating custom loss functions using only tensor operations.
 * We implement several custom losses: Focal Loss, Huber Loss, and Contrastive Loss.
 *
 * Usage: ./10_custom_loss_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <algorithm>

using namespace tenzor;

// ============ Custom Loss Functions ============

/**
 * @brief Focal Loss - addresses class imbalance by down-weighting easy examples
 * FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
 */
float focal_loss_tensor(const Tensor& logits, const Tensor& targets,
                        float gamma = 2.0f, float alpha = 0.25f) {
    // Softmax
    auto exp_logits = tenzor::exp(logits - tenzor::max(logits, 1, true));
    auto probs = exp_logits / tenzor::sum(exp_logits, 1, true);

    auto probs_cpu = probs.cpu();
    auto targets_cpu = targets.cpu();
    const float* prob_data = probs_cpu.data<float>();
    const int64_t* target_data = targets_cpu.data<int64_t>();

    int batch_size = static_cast<int>(logits.shape()[0]);
    int num_classes = static_cast<int>(logits.shape()[1]);

    float loss = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        float p_t = prob_data[b * num_classes + target_data[b]];
        p_t = std::max(p_t, 1e-7f);  // Clamp for numerical stability
        float focal_weight = alpha * std::pow(1.0f - p_t, gamma);
        loss -= focal_weight * std::log(p_t);
    }
    return loss / batch_size;
}

/**
 * @brief Huber Loss (Smooth L1) - robust to outliers
 * L = 0.5 * x^2        if |x| <= delta
 * L = delta * |x| - 0.5 * delta^2  otherwise
 */
float huber_loss_tensor(const Tensor& predictions, const Tensor& targets, float delta = 1.0f) {
    auto diff = predictions - targets;
    auto diff_cpu = diff.cpu();
    const float* diff_data = diff_cpu.data<float>();

    float loss = 0.0f;
    int n = static_cast<int>(diff.numel());

    for (int i = 0; i < n; ++i) {
        float abs_diff = std::abs(diff_data[i]);
        if (abs_diff <= delta) {
            loss += 0.5f * diff_data[i] * diff_data[i];
        } else {
            loss += delta * abs_diff - 0.5f * delta * delta;
        }
    }
    return loss / n;
}

/**
 * @brief Label Smoothing Loss - prevents overconfident predictions
 */
float label_smoothing_loss_tensor(const Tensor& logits, const Tensor& targets, float smoothing = 0.1f) {
    auto exp_logits = tenzor::exp(logits - tenzor::max(logits, 1, true));
    auto probs = exp_logits / tenzor::sum(exp_logits, 1, true);
    auto log_probs_tensor = tenzor::log(probs + 1e-7f);

    auto log_probs_cpu = log_probs_tensor.cpu();
    auto targets_cpu = targets.cpu();
    const float* log_prob_data = log_probs_cpu.data<float>();
    const int64_t* target_data = targets_cpu.data<int64_t>();

    int batch_size = static_cast<int>(logits.shape()[0]);
    int num_classes = static_cast<int>(logits.shape()[1]);

    float smooth_target = smoothing / num_classes;
    float confident_target = 1.0f - smoothing + smooth_target;

    float loss = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_classes; ++c) {
            float target_prob = (c == target_data[b]) ? confident_target : smooth_target;
            loss -= target_prob * log_prob_data[b * num_classes + c];
        }
    }
    return loss / batch_size;
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Custom Loss Functions - Tensor Only (Manual)", device);

    manual_seed(42);

    // Generate imbalanced classification data
    int batch_size = 64;
    int input_features = 16;
    int num_classes = 4;

    // Create imbalanced data (class 0 has more samples)
    std::vector<float> X_data(batch_size * input_features);
    std::vector<int64_t> y_data(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        int label;
        if (b < batch_size * 0.6) {
            label = 0;  // 60% class 0 (majority)
        } else {
            label = 1 + (b % (num_classes - 1));  // Rest spread among other classes
        }
        y_data[b] = label;

        for (int f = 0; f < input_features; ++f) {
            X_data[b * input_features + f] = static_cast<float>(label) * 0.5f +
                randn({1}, DType::Float32, device).cpu().data<float>()[0] * 0.5f;
        }
    }

    auto X = from_data(X_data.data(), {batch_size, input_features}, device);
    auto y = from_data(y_data.data(), {batch_size}, device);

    showcase::print_tensor_info("Input X", X);
    std::cout << "Data is imbalanced: 60% class 0, 40% other classes\n";

    // Initialize network weights
    auto W1 = randn({input_features, 32}, DType::Float32, device) * 0.1f;
    auto b1 = zeros({1, 32}, DType::Float32, device);
    auto W2 = randn({32, num_classes}, DType::Float32, device) * 0.1f;
    auto b2 = zeros({1, num_classes}, DType::Float32, device);

    // Clone weights for each loss comparison
    auto W1_focal = W1.clone();
    auto b1_focal = b1.clone();
    auto W2_focal = W2.clone();
    auto b2_focal = b2.clone();

    auto W1_smooth = W1.clone();
    auto b1_smooth = b1.clone();
    auto W2_smooth = W2.clone();
    auto b2_smooth = b2.clone();

    float learning_rate = 0.1f;
    int num_epochs = 200;
    int print_every = 40;

    // ============ Training with Focal Loss ============
    showcase::print_section("Training with Focal Loss (gamma=2, alpha=0.25)");
    std::cout << "Focal loss helps with class imbalance by focusing on hard examples\n\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Forward
        auto z1 = matmul(X, W1_focal) + b1_focal;
        auto a1 = clamp_min(z1, 0.0f);
        auto logits = matmul(a1, W2_focal) + b2_focal;

        float loss = focal_loss_tensor(logits, y, 2.0f, 0.25f);

        // Backward (simplified)
        auto exp_logits = tenzor::exp(logits - tenzor::max(logits, 1, true));
        auto probs = exp_logits / tenzor::sum(exp_logits, 1, true);
        auto probs_cpu = probs.cpu();
        auto y_cpu = y.cpu();
        const float* prob_data = probs_cpu.data<float>();
        const int64_t* target_data = y_cpu.data<int64_t>();

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

        auto dL_dW2 = matmul(a1.transpose(0, 1), dL_dlogits);
        auto dL_db2 = tenzor::sum(dL_dlogits, 0, true);
        auto dL_da1 = matmul(dL_dlogits, W2_focal.transpose(0, 1));
        auto dL_dz1 = dL_da1 * (z1 > zeros_like(z1)).to(DType::Float32);
        auto dL_dW1 = matmul(X.transpose(0, 1), dL_dz1);
        auto dL_db1 = tenzor::sum(dL_dz1, 0, true);

        W2_focal = W2_focal - dL_dW2 * learning_rate;
        b2_focal = b2_focal - dL_db2 * learning_rate;
        W1_focal = W1_focal - dL_dW1 * learning_rate;
        b1_focal = b1_focal - dL_db1 * learning_rate;

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float accuracy = showcase::multiclass_accuracy(logits, y);
            showcase::print_progress(epoch, num_epochs, loss, accuracy);
        }
    }

    // ============ Training with Label Smoothing ============
    showcase::print_section("Training with Label Smoothing (smoothing=0.1)");
    std::cout << "Label smoothing prevents overconfident predictions\n\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto z1 = matmul(X, W1_smooth) + b1_smooth;
        auto a1 = clamp_min(z1, 0.0f);
        auto logits = matmul(a1, W2_smooth) + b2_smooth;

        float loss = label_smoothing_loss_tensor(logits, y, 0.1f);

        // Backward
        auto exp_logits = tenzor::exp(logits - tenzor::max(logits, 1, true));
        auto probs = exp_logits / tenzor::sum(exp_logits, 1, true);
        auto probs_cpu = probs.cpu();
        auto y_cpu = y.cpu();
        const float* prob_data = probs_cpu.data<float>();
        const int64_t* target_data = y_cpu.data<int64_t>();

        float smoothing = 0.1f;
        float smooth_target = smoothing / num_classes;
        float confident_target = 1.0f - smoothing + smooth_target;

        std::vector<float> grad_data(batch_size * num_classes);
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_classes; ++c) {
                float target_prob = (c == target_data[b]) ? confident_target : smooth_target;
                grad_data[b * num_classes + c] = (prob_data[b * num_classes + c] - target_prob) / batch_size;
            }
        }
        auto dL_dlogits = from_data(grad_data.data(), {batch_size, num_classes}, device);

        auto dL_dW2 = matmul(a1.transpose(0, 1), dL_dlogits);
        auto dL_db2 = tenzor::sum(dL_dlogits, 0, true);
        auto dL_da1 = matmul(dL_dlogits, W2_smooth.transpose(0, 1));
        auto dL_dz1 = dL_da1 * (z1 > zeros_like(z1)).to(DType::Float32);
        auto dL_dW1 = matmul(X.transpose(0, 1), dL_dz1);
        auto dL_db1 = tenzor::sum(dL_dz1, 0, true);

        W2_smooth = W2_smooth - dL_dW2 * learning_rate;
        b2_smooth = b2_smooth - dL_db2 * learning_rate;
        W1_smooth = W1_smooth - dL_dW1 * learning_rate;
        b1_smooth = b1_smooth - dL_db1 * learning_rate;

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float accuracy = showcase::multiclass_accuracy(logits, y);
            showcase::print_progress(epoch, num_epochs, loss, accuracy);
        }
    }

    // ============ Huber Loss Demo ============
    showcase::print_section("Huber Loss Demo (for regression)");

    // Generate regression data with outliers
    std::vector<float> reg_X_data(32 * 4);
    std::vector<float> reg_y_data(32);
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 4; ++j) {
            reg_X_data[i * 4 + j] = randn({1}, DType::Float32, device).cpu().data<float>()[0];
        }
        // y = sum(X) + noise, with some outliers
        reg_y_data[i] = reg_X_data[i * 4] + reg_X_data[i * 4 + 1] +
                        reg_X_data[i * 4 + 2] + reg_X_data[i * 4 + 3];
        if (i < 3) {
            reg_y_data[i] += 10.0f;  // Add outliers
        }
    }
    auto reg_X = from_data(reg_X_data.data(), {32, 4}, device);
    auto reg_y = from_data(reg_y_data.data(), {32, 1}, device);

    auto W_reg = randn({4, 1}, DType::Float32, device) * 0.1f;
    auto predictions = matmul(reg_X, W_reg);

    float mse_loss = tenzor::mean((predictions - reg_y) * (predictions - reg_y)).cpu().item<float>();
    float huber = huber_loss_tensor(predictions, reg_y, 1.0f);

    std::cout << "With outliers in data:\n";
    std::cout << "  MSE Loss: " << mse_loss << " (sensitive to outliers)\n";
    std::cout << "  Huber Loss: " << huber << " (robust to outliers)\n";

    // ============ Summary ============
    showcase::print_section("Custom Loss Summary");

    std::cout << "Loss functions demonstrated:\n";
    std::cout << "1. Focal Loss: FL = -α(1-p)^γ log(p)\n";
    std::cout << "   - Addresses class imbalance\n";
    std::cout << "   - Down-weights easy examples\n\n";

    std::cout << "2. Huber Loss: L = 0.5x² if |x|≤δ, else δ|x|-0.5δ²\n";
    std::cout << "   - Robust to outliers\n";
    std::cout << "   - Combines MSE and MAE benefits\n\n";

    std::cout << "3. Label Smoothing: soft targets = (1-ε)·one_hot + ε/K\n";
    std::cout << "   - Prevents overconfident predictions\n";
    std::cout << "   - Improves generalization\n";

    std::cout << "\nCustom loss functions demonstrated with manual tensors!\n";

    finalize();
    return 0;
}
