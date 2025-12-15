/**
 * @file neural_network.cpp
 * @brief Dropout Regularization using Tenzor's high-level Neural Network API
 *
 * This example demonstrates dropout regularization using nn::Dropout
 * for a clean, production-ready interface.
 *
 * Usage: ./09_dropout_regularization_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

/**
 * @brief MLP with Dropout Regularization
 */
class MLPWithDropout : public nn::Module {
public:
    MLPWithDropout(int64_t input_features, int64_t hidden_features, int64_t num_classes,
                   float dropout_prob = 0.5f) {
        fc1 = std::make_shared<nn::Linear>(input_features, hidden_features);
        dropout1 = std::make_shared<nn::Dropout>(dropout_prob);
        fc2 = std::make_shared<nn::Linear>(hidden_features, hidden_features);
        dropout2 = std::make_shared<nn::Dropout>(dropout_prob);
        fc3 = std::make_shared<nn::Linear>(hidden_features, num_classes);

        register_module("fc1", fc1);
        register_module("dropout1", dropout1);
        register_module("fc2", fc2);
        register_module("dropout2", dropout2);
        register_module("fc3", fc3);

        dropout_prob_ = dropout_prob;
    }

    auto forward_impl(const Variable& input) -> Variable override {
        // FC1 -> ReLU -> Dropout
        auto x = fc1->forward(input);
        x = nn::relu(x);
        x = dropout1->forward(x);

        // FC2 -> ReLU -> Dropout
        x = fc2->forward(x);
        x = nn::relu(x);
        x = dropout2->forward(x);

        // FC3 (output)
        x = fc3->forward(x);
        return x;
    }

    float dropout_prob() const { return dropout_prob_; }

private:
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::Dropout> dropout1;
    std::shared_ptr<nn::Linear> fc2;
    std::shared_ptr<nn::Dropout> dropout2;
    std::shared_ptr<nn::Linear> fc3;
    float dropout_prob_;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Dropout Regularization - Neural Network API (High-Level)", device);

    manual_seed(42);

    // Generate synthetic data
    int batch_size = 32;
    int input_features = 64;
    int hidden_features = 128;
    int num_classes = 4;
    float dropout_prob = 0.5f;

    std::vector<float> X_data(batch_size * input_features);
    for (int b = 0; b < batch_size; ++b) {
        int label = b % num_classes;
        for (int f = 0; f < input_features; ++f) {
            X_data[b * input_features + f] = static_cast<float>(label) * 0.5f +
                                              randn({1}, DType::Float32, device).cpu().data<float>()[0] * 0.3f;
        }
    }
    auto X = from_data(X_data.data(), {batch_size, input_features}, device);

    std::vector<int64_t> y_data(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        y_data[i] = i % num_classes;
    }
    auto y = from_data(y_data.data(), {batch_size}, device);

    showcase::print_tensor_info("Input X", X);

    // Create model
    auto model = std::make_shared<MLPWithDropout>(input_features, hidden_features, num_classes, dropout_prob);
    model->to(device);

    // Create optimizer
    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);

    // Create loss function
    nn::CrossEntropyLoss criterion;

    showcase::print_section("Model Architecture");
    std::cout << "MLPWithDropout:\n";
    std::cout << "  fc1: Linear(" << input_features << ", " << hidden_features << ")\n";
    std::cout << "  ReLU\n";
    std::cout << "  dropout1: Dropout(p=" << dropout_prob << ")\n";
    std::cout << "  fc2: Linear(" << hidden_features << ", " << hidden_features << ")\n";
    std::cout << "  ReLU\n";
    std::cout << "  dropout2: Dropout(p=" << dropout_prob << ")\n";
    std::cout << "  fc3: Linear(" << hidden_features << ", " << num_classes << ")\n";
    std::cout << "\nTotal parameters: " << params.size() << "\n";

    // Training
    int num_epochs = 300;
    int print_every = 30;

    showcase::print_section("Training (with Dropout)");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();  // Enable training mode (Dropout active)

        optimizer.zero_grad();

        Variable input(X, false);
        auto logits = model->forward(input);

        auto loss = criterion(logits, y);
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();
            float accuracy = showcase::multiclass_accuracy(logits.tensor(), y);
            showcase::print_progress(epoch, num_epochs, loss_val, accuracy);
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    // Compare training vs eval mode
    std::cout << "Training mode (with dropout):\n";
    model->train();
    Variable X_train(X, false);
    auto logits_train = model->forward(X_train);
    float train_accuracy = showcase::multiclass_accuracy(logits_train.tensor(), y);
    std::cout << "  Accuracy: " << (train_accuracy * 100.0f) << "%\n";

    std::cout << "\nEval mode (no dropout):\n";
    model->eval();  // Disable dropout for inference
    Variable X_eval(X, false);
    auto logits_eval = model->forward(X_eval);
    float eval_accuracy = showcase::multiclass_accuracy(logits_eval.tensor(), y);
    std::cout << "  Accuracy: " << (eval_accuracy * 100.0f) << "%\n\n";

    std::cout << "Key Dropout behaviors:\n";
    std::cout << "  - Training mode (model->train()): dropout active, random zeroing\n";
    std::cout << "  - Eval mode (model->eval()): dropout disabled, full network\n";
    std::cout << "  - Inverted dropout: scales by 1/(1-p) during training\n";
    std::cout << "  - Regularization: reduces overfitting on training data\n";

    // Show dropout impact
    std::cout << "\nDropout Impact:\n";
    std::cout << "  - Training accuracy may fluctuate due to random dropout\n";
    std::cout << "  - Eval accuracy is deterministic (no randomness)\n";
    std::cout << "  - Dropout helps generalization to unseen data\n";

    std::cout << "\nDropout Regularization demonstrated with Neural Network API!\n";

    finalize();
    return 0;
}
