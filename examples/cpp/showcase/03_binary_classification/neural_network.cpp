/**
 * @file neural_network.cpp
 * @brief Binary Classification using Tenzor's high-level Neural Network API
 *
 * This example demonstrates binary classification using nn::Module,
 * BCEWithLogitsLoss, and optimizers for a clean, production-ready interface.
 *
 * Usage: ./03_binary_classification_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

/**
 * @brief Binary classifier with a hidden layer
 *
 * A simple 2-layer MLP for binary classification.
 */
class BinaryClassifier : public nn::Module {
public:
    BinaryClassifier(int64_t input_features, int64_t hidden_features) {
        fc1 = std::make_shared<nn::Linear>(input_features, hidden_features);
        fc2 = std::make_shared<nn::Linear>(hidden_features, 1);

        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        // Hidden layer with ReLU activation
        auto h = nn::relu(fc1->forward(input));
        // Output layer (logits, no sigmoid - used with BCEWithLogitsLoss)
        auto out = fc2->forward(h);
        return out;
    }

private:
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::Linear> fc2;
};

int main(int argc, char* argv[]) {
    // Parse backend from command line
    Device device = showcase::get_device_from_args(argc, argv);

    // Initialize Tenzor
    initialize();

    showcase::print_header("Binary Classification - Neural Network API (High-Level)", device);

    // Set seed for reproducibility
    manual_seed(42);

    // Generate synthetic binary classification data
    int samples_per_class = 50;
    int num_samples = samples_per_class * 2;

    // Class 0: centered around (-2, -2)
    auto class0_x = randn({samples_per_class, 2}, DType::Float32, device) + (-2.0f);

    // Class 1: centered around (2, 2)
    auto class1_x = randn({samples_per_class, 2}, DType::Float32, device) + 2.0f;

    // Combine into dataset
    auto X = cat({class0_x, class1_x}, 0);  // (100, 2)

    // Create labels
    auto labels0 = zeros({samples_per_class, 1}, DType::Float32, device);
    auto labels1 = ones({samples_per_class, 1}, DType::Float32, device);
    auto y = cat({labels0, labels1}, 0);  // (100, 1)

    showcase::print_tensor_info("Input X", X);
    showcase::print_tensor_info("Labels y", y);

    // Create model
    auto model = std::make_shared<BinaryClassifier>(2, 8);  // 2 inputs, 8 hidden
    model->to(device);

    // Create optimizer (Adam for better convergence)
    auto params = model->parameters();
    optim::Adam optimizer(params, 0.01f);  // learning_rate = 0.01

    // Create loss function (BCEWithLogitsLoss for numerical stability)
    nn::BCEWithLogitsLoss criterion;

    showcase::print_section("Model Architecture");
    std::cout << "BinaryClassifier:\n";
    std::cout << "  fc1: Linear(2, 8)\n";
    std::cout << "  ReLU activation\n";
    std::cout << "  fc2: Linear(8, 1)\n";
    std::cout << "  BCEWithLogitsLoss (sigmoid inside)\n";
    std::cout << "\nTotal parameters: " << params.size() << "\n";

    // Training parameters
    int num_epochs = 200;
    int print_every = 20;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();

        // Zero gradients
        optimizer.zero_grad();

        // Forward pass
        Variable input(X, false);
        auto logits = model->forward(input);

        // Compute loss
        Variable target(y, false);
        auto loss = criterion(logits, target);

        // Backward pass
        loss.backward();

        // Update weights
        optimizer.step();

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();

            // Calculate accuracy (need to apply sigmoid to logits for predictions)
            auto pred_tensor = nn::sigmoid(logits).tensor().cpu();
            auto target_cpu = y.cpu();
            int correct = 0;
            const float* pred = pred_tensor.data<float>();
            const float* tgt = target_cpu.data<float>();
            for (int i = 0; i < num_samples; ++i) {
                float predicted = (pred[i] > 0.5f) ? 1.0f : 0.0f;
                if (predicted == tgt[i]) correct++;
            }
            float accuracy = static_cast<float>(correct) / num_samples;

            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    model->eval();

    // Final predictions
    Variable X_eval(X, false);
    auto logits_final = model->forward(X_eval);
    auto probs_final = nn::sigmoid(logits_final);

    auto pred_cpu = probs_final.tensor().cpu();
    auto target_cpu = y.cpu();
    int correct = 0;
    const float* pred = pred_cpu.data<float>();
    const float* tgt = target_cpu.data<float>();
    for (int i = 0; i < num_samples; ++i) {
        float predicted = (pred[i] > 0.5f) ? 1.0f : 0.0f;
        if (predicted == tgt[i]) correct++;
    }
    float final_accuracy = static_cast<float>(correct) / num_samples;

    std::cout << "Final Accuracy: " << (final_accuracy * 100.0f) << "%\n\n";

    // Show some predictions
    showcase::print_section("Sample Predictions");

    std::cout << "Sample\t\tTrue Label\tPredicted\tProbability\n";
    std::cout << "-----------------------------------------------------\n";

    auto X_cpu = X.cpu();
    for (int i = 0; i < 10; ++i) {
        float x0 = X_cpu.data<float>()[i * 2];
        float x1 = X_cpu.data<float>()[i * 2 + 1];
        float true_label = tgt[i];
        float prob = pred[i];
        float predicted = (prob > 0.5f) ? 1.0f : 0.0f;

        std::cout << "(" << x0 << ", " << x1 << ")\t"
                  << true_label << "\t\t" << predicted << "\t\t"
                  << prob << "\n";
    }

    std::cout << "\nBinary classification solved using Neural Network API!\n";
    std::cout << "Uses BCEWithLogitsLoss for numerical stability.\n";

    finalize();
    return 0;
}
