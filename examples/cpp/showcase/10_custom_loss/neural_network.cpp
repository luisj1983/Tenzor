/**
 * @file neural_network.cpp
 * @brief Custom Loss Functions using Tenzor's high-level Neural Network API
 *
 * This example demonstrates using custom loss functions alongside the built-in
 * losses (FocalLoss, HuberLoss) with nn::Module for production use.
 *
 * Note: Tenzor already includes FocalLoss and HuberLoss in nn::losses.
 * This example shows both using the built-in ones and creating custom losses.
 *
 * Usage: ./10_custom_loss_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>

using namespace tenzor;

// ============ Custom Loss Class (standalone, like nn::CrossEntropyLoss) ============

/**
 * @brief Label Smoothing Cross Entropy Loss
 * Prevents overconfident predictions by softening the target distribution.
 */
class LabelSmoothingCrossEntropy {
public:
    explicit LabelSmoothingCrossEntropy(float smoothing = 0.1f) : smoothing_(smoothing) {}

    auto forward(const Variable& input, const Tensor& target) -> Variable {
        int batch_size = static_cast<int>(input.shape()[0]);
        int num_classes = static_cast<int>(input.shape()[1]);

        auto log_probs = log_softmax(input, 1);

        // Create smooth targets
        auto target_cpu = target.cpu();
        const int64_t* target_data = target_cpu.data<int64_t>();

        float smooth = smoothing_ / num_classes;
        float confident = 1.0f - smoothing_ + smooth;

        std::vector<float> smooth_data(batch_size * num_classes);
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_classes; ++c) {
                smooth_data[b * num_classes + c] = (c == target_data[b]) ? confident : smooth;
            }
        }
        auto smooth_tensor = from_data(smooth_data.data(), {batch_size, num_classes}, input.device());
        Variable smooth_targets(smooth_tensor, false);

        return mean(sum(smooth_targets * log_probs, 1)) * (-1.0f);
    }

    auto operator()(const Variable& input, const Tensor& target) -> Variable {
        return forward(input, target);
    }

private:
    float smoothing_;
};

/**
 * @brief Simple MLP for classification
 */
class SimpleMLP : public nn::Module {
public:
    SimpleMLP(int64_t input_size, int64_t hidden_size, int64_t output_size) {
        fc1 = std::make_shared<nn::Linear>(input_size, hidden_size);
        fc2 = std::make_shared<nn::Linear>(hidden_size, output_size);

        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto x = fc1->forward(input);
        x = nn::relu(x);
        x = fc2->forward(x);
        return x;
    }

private:
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::Linear> fc2;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Custom Loss Functions - Neural Network API (High-Level)", device);

    manual_seed(42);

    // Generate imbalanced classification data
    int batch_size = 64;
    int input_features = 16;
    int hidden_features = 32;
    int num_classes = 4;

    std::vector<float> X_data(batch_size * input_features);
    for (int b = 0; b < batch_size; ++b) {
        int label = (b < batch_size * 0.6) ? 0 : (1 + (b % (num_classes - 1)));
        for (int f = 0; f < input_features; ++f) {
            X_data[b * input_features + f] = static_cast<float>(label) * 0.5f +
                randn({1}, DType::Float32, device).cpu().data<float>()[0] * 0.5f;
        }
    }
    auto X = from_data(X_data.data(), {batch_size, input_features}, device);

    std::vector<int64_t> y_data(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        y_data[i] = (i < batch_size * 0.6) ? 0 : (1 + (i % (num_classes - 1)));
    }
    auto y = from_data(y_data.data(), {batch_size}, device);

    // Create one-hot encoded targets for FocalLoss (expects float targets)
    std::vector<float> y_onehot_data(batch_size * num_classes, 0.0f);
    for (int i = 0; i < batch_size; ++i) {
        y_onehot_data[i * num_classes + y_data[i]] = 1.0f;
    }
    auto y_onehot = from_data(y_onehot_data.data(), {batch_size, num_classes}, device);

    showcase::print_tensor_info("Input X", X);
    std::cout << "Imbalanced data: 60% class 0, 40% distributed among other classes\n";

    // Training parameters
    int num_epochs = 200;
    int print_every = 40;

    // ============ Training with Built-in FocalLoss ============
    showcase::print_section("Training with nn::FocalLoss (built-in, gamma=2, alpha=0.25)");

    auto model_focal = std::make_shared<SimpleMLP>(input_features, hidden_features, num_classes);
    model_focal->to(device);
    auto params_focal = model_focal->parameters();
    optim::Adam optimizer_focal(params_focal, 0.01f);
    nn::FocalLoss focal_criterion(0.25f, 2.0f);  // alpha, gamma

    std::cout << "FocalLoss focuses learning on hard-to-classify examples\n\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model_focal->train();
        optimizer_focal.zero_grad();

        Variable input(X, false);
        auto logits = model_focal->forward(input);
        auto loss = focal_criterion(logits, Variable(y_onehot, false));

        loss.backward();
        optimizer_focal.step();

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().cpu().item<float>();
            float accuracy = showcase::multiclass_accuracy(logits.tensor().cpu(), y.cpu());
            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    // ============ Training with Custom LabelSmoothingCrossEntropy ============
    showcase::print_section("Training with LabelSmoothingCrossEntropy (custom, smoothing=0.1)");

    auto model_smooth = std::make_shared<SimpleMLP>(input_features, hidden_features, num_classes);
    model_smooth->to(device);
    auto params_smooth = model_smooth->parameters();
    optim::Adam optimizer_smooth(params_smooth, 0.01f);
    LabelSmoothingCrossEntropy smooth_criterion(0.1f);

    std::cout << "Label smoothing prevents overconfident predictions\n\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model_smooth->train();
        optimizer_smooth.zero_grad();

        Variable input(X, false);
        auto logits = model_smooth->forward(input);
        auto loss = smooth_criterion(logits, y);

        loss.backward();
        optimizer_smooth.step();

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().cpu().item<float>();
            float accuracy = showcase::multiclass_accuracy(logits.tensor().cpu(), y.cpu());
            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    // ============ Training with Standard CrossEntropy (Comparison) ============
    showcase::print_section("Training with Standard CrossEntropyLoss (baseline)");

    auto model_ce = std::make_shared<SimpleMLP>(input_features, hidden_features, num_classes);
    model_ce->to(device);
    auto params_ce = model_ce->parameters();
    optim::Adam optimizer_ce(params_ce, 0.01f);
    nn::CrossEntropyLoss ce_criterion;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model_ce->train();
        optimizer_ce.zero_grad();

        Variable input(X, false);
        auto logits = model_ce->forward(input);
        auto loss = ce_criterion(logits, y);

        loss.backward();
        optimizer_ce.step();

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().cpu().item<float>();
            float accuracy = showcase::multiclass_accuracy(logits.tensor().cpu(), y.cpu());
            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    // ============ HuberLoss Demo for Regression ============
    showcase::print_section("nn::HuberLoss Demo (Regression with Outliers)");

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
    auto reg_X = from_data(reg_X_data.data(), {32, 4}, device);
    auto reg_y = from_data(reg_y_data.data(), {32, 1}, device);

    auto linear = std::make_shared<nn::Linear>(4, 1);
    linear->to(device);
    auto params_reg = linear->parameters();
    optim::SGD optimizer_reg(params_reg, 0.01f);
    nn::HuberLoss huber_criterion(1.0f);  // delta=1.0

    std::cout << "Training linear regression with outliers using built-in HuberLoss:\n";

    for (int epoch = 0; epoch < 100; ++epoch) {
        optimizer_reg.zero_grad();

        Variable input(reg_X, false);
        Variable target(reg_y, false);
        auto predictions = linear->forward(input);
        auto loss = huber_criterion(predictions, target);

        loss.backward();
        optimizer_reg.step();

        if ((epoch + 1) % 25 == 0) {
            std::cout << "Epoch " << (epoch + 1) << ": Huber Loss = " << loss.tensor().cpu().item<float>() << "\n";
        }
    }

    // ============ Final Comparison ============
    showcase::print_section("Final Comparison");

    model_focal->eval();
    model_smooth->eval();
    model_ce->eval();

    Variable X_eval(X, false);

    auto logits_focal = model_focal->forward(X_eval);
    auto logits_smooth = model_smooth->forward(X_eval);
    auto logits_ce = model_ce->forward(X_eval);

    float acc_focal = showcase::multiclass_accuracy(logits_focal.tensor().cpu(), y.cpu());
    float acc_smooth = showcase::multiclass_accuracy(logits_smooth.tensor().cpu(), y.cpu());
    float acc_ce = showcase::multiclass_accuracy(logits_ce.tensor().cpu(), y.cpu());

    std::cout << "Final Accuracies on imbalanced data:\n";
    std::cout << "  FocalLoss (built-in):         " << (acc_focal * 100.0f) << "%\n";
    std::cout << "  LabelSmoothingCE (custom):    " << (acc_smooth * 100.0f) << "%\n";
    std::cout << "  CrossEntropyLoss (built-in):  " << (acc_ce * 100.0f) << "%\n\n";

    std::cout << "Built-in loss functions in Tenzor:\n";
    std::cout << "  - nn::CrossEntropyLoss: Standard multi-class classification\n";
    std::cout << "  - nn::FocalLoss: Class imbalance, object detection\n";
    std::cout << "  - nn::HuberLoss: Regression with outliers\n";
    std::cout << "  - nn::MSELoss, nn::L1Loss: Standard regression\n";
    std::cout << "  - nn::BCELoss, nn::BCEWithLogitsLoss: Binary classification\n";
    std::cout << "  - nn::NLLLoss, nn::KLDivLoss, nn::DiceLoss: Specialized uses\n";

    std::cout << "\nCustom Loss Functions demonstrated with Neural Network API!\n";

    finalize();
    return 0;
}
