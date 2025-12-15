/**
 * @file neural_network.cpp
 * @brief Multi-class Classification using Tenzor's high-level Neural Network API
 *
 * This example demonstrates multi-class classification using nn::Module,
 * CrossEntropyLoss, and optimizers for a clean, production-ready interface.
 *
 * Usage: ./04_multiclass_classification_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

/**
 * @brief Multi-class classifier with hidden layers
 *
 * A 3-layer MLP for multi-class classification.
 */
class MultiClassClassifier : public nn::Module {
public:
    MultiClassClassifier(int64_t input_features, int64_t hidden_features, int64_t num_classes) {
        fc1 = std::make_shared<nn::Linear>(input_features, hidden_features);
        fc2 = std::make_shared<nn::Linear>(hidden_features, hidden_features);
        fc3 = std::make_shared<nn::Linear>(hidden_features, num_classes);

        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        // Hidden layers with ReLU activation
        auto h1 = nn::relu(fc1->forward(input));
        auto h2 = nn::relu(fc2->forward(h1));
        // Output layer (logits - no softmax, used with CrossEntropyLoss)
        auto out = fc3->forward(h2);
        return out;
    }

private:
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::Linear> fc2;
    std::shared_ptr<nn::Linear> fc3;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Multi-class Classification - Neural Network API (High-Level)", device);

    manual_seed(42);

    // Generate synthetic multi-class data: 3 classes in 2D
    int samples_per_class = 50;
    int num_classes = 3;
    int num_samples = samples_per_class * num_classes;
    int num_features = 2;

    // Class 0: centered around (0, 3)
    auto class0_x = randn({samples_per_class, 2}, DType::Float32, device);
    class0_x = class0_x + from_data(std::vector<float>{0.0f, 3.0f}.data(), {1, 2}, device);

    // Class 1: centered around (-3, -1)
    auto class1_x = randn({samples_per_class, 2}, DType::Float32, device);
    class1_x = class1_x + from_data(std::vector<float>{-3.0f, -1.0f}.data(), {1, 2}, device);

    // Class 2: centered around (3, -1)
    auto class2_x = randn({samples_per_class, 2}, DType::Float32, device);
    class2_x = class2_x + from_data(std::vector<float>{3.0f, -1.0f}.data(), {1, 2}, device);

    auto X = cat({class0_x, class1_x, class2_x}, 0);  // (150, 2)

    // Create labels
    std::vector<int64_t> label_data;
    for (int c = 0; c < num_classes; ++c) {
        for (int i = 0; i < samples_per_class; ++i) {
            label_data.push_back(c);
        }
    }
    auto y = from_data(label_data.data(), {num_samples}, device);

    showcase::print_tensor_info("Input X", X);
    showcase::print_tensor_info("Labels y", y);

    // Create model
    auto model = std::make_shared<MultiClassClassifier>(num_features, 16, num_classes);
    model->to(device);

    // Create optimizer
    auto params = model->parameters();
    optim::Adam optimizer(params, 0.01f);

    // Create loss function
    nn::CrossEntropyLoss criterion;

    showcase::print_section("Model Architecture");
    std::cout << "MultiClassClassifier:\n";
    std::cout << "  fc1: Linear(2, 16) + ReLU\n";
    std::cout << "  fc2: Linear(16, 16) + ReLU\n";
    std::cout << "  fc3: Linear(16, 3)\n";
    std::cout << "  CrossEntropyLoss (softmax inside)\n";
    std::cout << "\nTotal parameters: " << params.size() << "\n";

    // Training parameters
    int num_epochs = 300;
    int print_every = 30;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();

        optimizer.zero_grad();

        // Forward pass
        Variable input(X, false);
        auto logits = model->forward(input);

        // Compute loss
        auto loss = criterion(logits, y);

        // Backward pass
        loss.backward();

        // Update weights
        optimizer.step();

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();
            float accuracy = showcase::multiclass_accuracy(logits.tensor(), y);
            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    model->eval();

    Variable X_eval(X, false);
    auto logits_final = model->forward(X_eval);

    float final_accuracy = showcase::multiclass_accuracy(logits_final.tensor(), y);
    std::cout << "Final Accuracy: " << (final_accuracy * 100.0f) << "%\n\n";

    // Show per-class accuracy
    auto logits_cpu = logits_final.tensor().cpu();
    auto y_cpu = y.cpu();
    const float* logit_data = logits_cpu.data<float>();
    const int64_t* target_data = y_cpu.data<int64_t>();

    int class_correct[3] = {0};
    int class_total[3] = {0};

    for (int i = 0; i < num_samples; ++i) {
        int true_class = target_data[i];
        int pred_class = 0;
        float max_val = logit_data[i * num_classes];
        for (int c = 1; c < num_classes; ++c) {
            if (logit_data[i * num_classes + c] > max_val) {
                max_val = logit_data[i * num_classes + c];
                pred_class = c;
            }
        }
        class_total[true_class]++;
        if (pred_class == true_class) {
            class_correct[true_class]++;
        }
    }

    std::cout << "Per-class Accuracy:\n";
    for (int c = 0; c < num_classes; ++c) {
        float acc = static_cast<float>(class_correct[c]) / class_total[c];
        std::cout << "  Class " << c << ": " << (acc * 100.0f) << "% ("
                  << class_correct[c] << "/" << class_total[c] << ")\n";
    }

    std::cout << "\nMulti-class classification solved using Neural Network API!\n";
    std::cout << "CrossEntropyLoss combines log-softmax and NLL loss for efficiency.\n";

    finalize();
    return 0;
}
