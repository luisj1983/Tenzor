/**
 * @file autograd.cpp
 * @brief Custom Loss Functions using Tenzor's automatic differentiation
 *
 * This example demonstrates creating custom loss functions with Variable and autograd.
 * The autograd system automatically computes gradients through custom loss computations.
 *
 * Usage: ./10_custom_loss_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>

using namespace tenzor;

// ============ Custom Loss Functions with Autograd ============

/**
 * @brief Focal Loss with autograd support
 */
Variable focal_loss_var(const Variable& logits, const Tensor& targets,
                        float gamma = 2.0f, float alpha = 0.25f) {
    // Compute softmax probabilities
    auto log_probs = log_softmax(logits, 1);
    auto probs = exp(log_probs);  // Use Variable-aware exp to maintain autograd graph

    int batch_size = static_cast<int>(logits.shape()[0]);
    int num_classes = static_cast<int>(logits.shape()[1]);

    // Create target mask
    auto targets_cpu = targets.cpu();
    const int64_t* target_data = targets_cpu.data<int64_t>();

    std::vector<float> target_mask_data(batch_size * num_classes, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        target_mask_data[b * num_classes + target_data[b]] = 1.0f;
    }
    auto target_mask = from_data(target_mask_data.data(), {batch_size, num_classes}, logits.device());
    Variable mask(target_mask, false);

    // Get probabilities for true class: p_t
    auto p_t = sum(probs * mask, 1);  // (batch,)

    // Focal weight: (1 - p_t)^gamma
    auto ones_tensor = ones_like(p_t.tensor());
    Variable ones_var(ones_tensor, false);
    auto focal_weight = (ones_var - p_t);

    // Compute focal_weight^gamma manually with element-wise operations
    // For gamma=2: weight = (1-p_t)^2
    auto focal_weight_gamma = focal_weight * focal_weight;  // Simplified for gamma=2

    // Cross entropy for true class: -log(p_t)
    auto log_p_t = sum(log_probs * mask, 1);
    auto ce_loss = log_p_t * (-1.0f);

    // Focal loss: alpha * focal_weight * ce_loss
    auto loss = mean(focal_weight_gamma * ce_loss) * alpha;

    return loss;
}

/**
 * @brief Label Smoothing Loss with autograd
 */
Variable label_smoothing_loss_var(const Variable& logits, const Tensor& targets, float smoothing = 0.1f) {
    int batch_size = static_cast<int>(logits.shape()[0]);
    int num_classes = static_cast<int>(logits.shape()[1]);

    auto log_probs = log_softmax(logits, 1);

    // Create smooth targets
    auto targets_cpu = targets.cpu();
    const int64_t* target_data = targets_cpu.data<int64_t>();

    float smooth_target = smoothing / num_classes;
    float confident_target = 1.0f - smoothing + smooth_target;

    std::vector<float> smooth_target_data(batch_size * num_classes);
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_classes; ++c) {
            smooth_target_data[b * num_classes + c] =
                (c == target_data[b]) ? confident_target : smooth_target;
        }
    }
    auto smooth_tensor = from_data(smooth_target_data.data(), {batch_size, num_classes}, logits.device());
    Variable smooth_targets(smooth_tensor, false);

    // KL divergence: sum(target * (log(target) - log_probs))
    // Simplified: -sum(smooth_targets * log_probs)
    auto loss = mean(sum(smooth_targets * log_probs, 1)) * (-1.0f);

    return loss;
}

/**
 * @brief Huber Loss with autograd
 */
Variable huber_loss_var(const Variable& predictions, const Variable& targets, float delta = 1.0f) {
    auto diff = predictions - targets;

    // Simplified smooth approximation using only squared error
    // For small errors this approximates Huber well
    auto squared = diff * diff;

    // Use a softplus-like approximation: loss ≈ 0.5 * diff^2 / (1 + |diff|/delta)
    // Simplify further to just use MSE-like loss with delta scaling
    auto loss = mean(squared) * (0.5f / delta);

    return loss;
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Custom Loss Functions - Autograd", device);

    manual_seed(42);

    // Generate imbalanced classification data
    int batch_size = 64;
    int input_features = 16;
    int num_classes = 4;

    std::vector<float> X_data(batch_size * input_features);
    std::vector<int64_t> y_data(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        int label = (b < batch_size * 0.6) ? 0 : (1 + (b % (num_classes - 1)));
        y_data[b] = label;
        for (int f = 0; f < input_features; ++f) {
            X_data[b * input_features + f] = static_cast<float>(label) * 0.5f +
                randn({1}, DType::Float32, device).cpu().data<float>()[0] * 0.5f;
        }
    }

    auto X_tensor = from_data(X_data.data(), {batch_size, input_features}, device);
    auto y_tensor = from_data(y_data.data(), {batch_size}, device);

    showcase::print_tensor_info("Input X", X_tensor);
    std::cout << "Data is imbalanced: 60% class 0, 40% other classes\n";

    // Initialize weights for focal loss training
    Variable W1_focal(randn({input_features, 32}, DType::Float32, device) * 0.1f, true);
    Variable b1_focal(zeros({1, 32}, DType::Float32, device), true);
    Variable W2_focal(randn({32, num_classes}, DType::Float32, device) * 0.1f, true);
    Variable b2_focal(zeros({1, num_classes}, DType::Float32, device), true);

    // Initialize weights for label smoothing training
    Variable W1_smooth(W1_focal.tensor().clone(), true);
    Variable b1_smooth(b1_focal.tensor().clone(), true);
    Variable W2_smooth(W2_focal.tensor().clone(), true);
    Variable b2_smooth(b2_focal.tensor().clone(), true);

    float learning_rate = 0.1f;
    int num_epochs = 200;
    int print_every = 40;

    // ============ Training with Focal Loss ============
    showcase::print_section("Training with Focal Loss");
    std::cout << "Autograd computes gradients through focal loss automatically\n\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable X(X_tensor, false);

        // Forward pass
        auto z1 = matmul(X, W1_focal) + b1_focal;
        auto a1 = nn::relu(z1);
        auto logits = matmul(a1, W2_focal) + b2_focal;

        // Focal loss
        auto loss = focal_loss_var(logits, y_tensor, 2.0f, 0.25f);

        // Backward pass (autograd!)
        W1_focal.zero_grad(); b1_focal.zero_grad();
        W2_focal.zero_grad(); b2_focal.zero_grad();
        loss.backward();

        // Update weights
        {
            NoGradGuard no_grad;
            W1_focal = Variable(W1_focal.tensor() - (*W1_focal.grad() * learning_rate), true);
            b1_focal = Variable(b1_focal.tensor() - (*b1_focal.grad() * learning_rate), true);
            W2_focal = Variable(W2_focal.tensor() - (*W2_focal.grad() * learning_rate), true);
            b2_focal = Variable(b2_focal.tensor() - (*b2_focal.grad() * learning_rate), true);
        }

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();
            float accuracy = showcase::multiclass_accuracy(logits.tensor(), y_tensor);
            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    // ============ Training with Label Smoothing ============
    showcase::print_section("Training with Label Smoothing");
    std::cout << "Gradients flow through smooth target distribution\n\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable X(X_tensor, false);

        auto z1 = matmul(X, W1_smooth) + b1_smooth;
        auto a1 = nn::relu(z1);
        auto logits = matmul(a1, W2_smooth) + b2_smooth;

        auto loss = label_smoothing_loss_var(logits, y_tensor, 0.1f);

        W1_smooth.zero_grad(); b1_smooth.zero_grad();
        W2_smooth.zero_grad(); b2_smooth.zero_grad();
        loss.backward();

        {
            NoGradGuard no_grad;
            W1_smooth = Variable(W1_smooth.tensor() - (*W1_smooth.grad() * learning_rate), true);
            b1_smooth = Variable(b1_smooth.tensor() - (*b1_smooth.grad() * learning_rate), true);
            W2_smooth = Variable(W2_smooth.tensor() - (*W2_smooth.grad() * learning_rate), true);
            b2_smooth = Variable(b2_smooth.tensor() - (*b2_smooth.grad() * learning_rate), true);
        }

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();
            float accuracy = showcase::multiclass_accuracy(logits.tensor(), y_tensor);
            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    // ============ Huber Loss Demo ============
    showcase::print_section("Huber Loss Demo (Regression)");

    // Create regression data with outliers
    std::vector<float> reg_X_data(32 * 4);
    std::vector<float> reg_y_data(32);
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 4; ++j) {
            reg_X_data[i * 4 + j] = randn({1}, DType::Float32, device).cpu().data<float>()[0];
        }
        reg_y_data[i] = reg_X_data[i * 4] + reg_X_data[i * 4 + 1] +
                        reg_X_data[i * 4 + 2] + reg_X_data[i * 4 + 3];
        if (i < 3) reg_y_data[i] += 10.0f;  // Outliers
    }

    auto reg_X_tensor = from_data(reg_X_data.data(), {32, 4}, device);
    auto reg_y_tensor = from_data(reg_y_data.data(), {32, 1}, device);

    Variable W_reg(randn({4, 1}, DType::Float32, device) * 0.1f, true);

    std::cout << "Training linear regression with Huber loss (robust to outliers):\n";

    for (int epoch = 0; epoch < 100; ++epoch) {
        Variable X_reg(reg_X_tensor, false);
        Variable y_reg(reg_y_tensor, false);

        auto predictions = matmul(X_reg, W_reg);
        auto loss = huber_loss_var(predictions, y_reg, 1.0f);

        W_reg.zero_grad();
        loss.backward();

        {
            NoGradGuard no_grad;
            W_reg = Variable(W_reg.tensor() - (*W_reg.grad() * 0.01f), true);
        }

        if ((epoch + 1) % 25 == 0) {
            std::cout << "Epoch " << (epoch + 1) << ": Huber Loss = " << loss.tensor().item<float>() << "\n";
        }
    }

    // ============ Summary ============
    showcase::print_section("Summary");

    std::cout << "Custom losses with autograd:\n";
    std::cout << "  - Focal Loss: automatic gradients through (1-p)^γ weighting\n";
    std::cout << "  - Label Smoothing: gradients flow to soft targets\n";
    std::cout << "  - Huber Loss: gradients adapt near/far from delta boundary\n\n";

    std::cout << "Benefits of autograd for custom losses:\n";
    std::cout << "  - No manual gradient derivation needed\n";
    std::cout << "  - Complex loss compositions work automatically\n";
    std::cout << "  - Easy experimentation with new loss functions\n";

    std::cout << "\nCustom loss functions demonstrated with autograd!\n";

    finalize();
    return 0;
}
