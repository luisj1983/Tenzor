/**
 * @file neural_network.cpp
 * @brief Batch Normalization using Tenzor's high-level Neural Network API
 *
 * This example demonstrates batch normalization using nn::BatchNorm1d
 * for a clean, production-ready interface.
 *
 * Usage: ./08_batch_normalization_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

/**
 * @brief MLP with Batch Normalization
 */
class MLPWithBatchNorm : public nn::Module {
public:
    MLPWithBatchNorm(int64_t input_features, int64_t hidden_features, int64_t num_classes) {
        fc1 = std::make_shared<nn::Linear>(input_features, hidden_features);
        bn1 = std::make_shared<nn::BatchNorm1d>(hidden_features);
        fc2 = std::make_shared<nn::Linear>(hidden_features, hidden_features);
        bn2 = std::make_shared<nn::BatchNorm1d>(hidden_features);
        fc3 = std::make_shared<nn::Linear>(hidden_features, num_classes);

        register_module("fc1", fc1);
        register_module("bn1", bn1);
        register_module("fc2", fc2);
        register_module("bn2", bn2);
        register_module("fc3", fc3);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        // FC1 -> BN1 -> ReLU
        auto x = fc1->forward(input);
        x = bn1->forward(x);
        x = nn::relu(x);

        // FC2 -> BN2 -> ReLU
        x = fc2->forward(x);
        x = bn2->forward(x);
        x = nn::relu(x);

        // FC3 (output)
        x = fc3->forward(x);
        return x;
    }

private:
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::BatchNorm1d> bn1;
    std::shared_ptr<nn::Linear> fc2;
    std::shared_ptr<nn::BatchNorm1d> bn2;
    std::shared_ptr<nn::Linear> fc3;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Batch Normalization - Neural Network API (High-Level)", device);

    manual_seed(42);

    // Generate synthetic data with varying statistics per feature
    int batch_size = 64;
    int input_features = 16;
    int hidden_features = 32;
    int num_classes = 5;

    std::vector<float> X_data(batch_size * input_features);
    for (int b = 0; b < batch_size; ++b) {
        for (int f = 0; f < input_features; ++f) {
            float feat_mean = f * 3.0f - 20.0f;  // Widely varying means
            float feat_std = 0.5f + f * 0.5f;    // Varying variances
            X_data[b * input_features + f] = feat_mean + randn({1}, DType::Float32, device).cpu().data<float>()[0] * feat_std;
        }
    }
    auto X = from_data(X_data.data(), {batch_size, input_features}, device);

    std::vector<int64_t> y_data(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        y_data[i] = i % num_classes;
    }
    auto y = from_data(y_data.data(), {batch_size}, device);

    showcase::print_tensor_info("Input X", X);

    // Show input statistics
    std::cout << "\nInput statistics (varying means and variances):\n";
    auto X_mean = tenzor::mean(X, 0, true).cpu();
    for (int f = 0; f < 4; ++f) {
        std::cout << "  Feature " << f << ": mean ~ " << X_mean.data<float>()[f] << "\n";
    }

    // Create model
    auto model = std::make_shared<MLPWithBatchNorm>(input_features, hidden_features, num_classes);
    model->to(device);

    // Create optimizer
    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);

    // Create loss function
    nn::CrossEntropyLoss criterion;

    showcase::print_section("Model Architecture");
    std::cout << "MLPWithBatchNorm:\n";
    std::cout << "  fc1: Linear(" << input_features << ", " << hidden_features << ")\n";
    std::cout << "  bn1: BatchNorm1d(" << hidden_features << ")\n";
    std::cout << "  ReLU\n";
    std::cout << "  fc2: Linear(" << hidden_features << ", " << hidden_features << ")\n";
    std::cout << "  bn2: BatchNorm1d(" << hidden_features << ")\n";
    std::cout << "  ReLU\n";
    std::cout << "  fc3: Linear(" << hidden_features << ", " << num_classes << ")\n";
    std::cout << "\nTotal parameters: " << params.size() << "\n";

    // Training
    int num_epochs = 200;
    int print_every = 20;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();  // Enable training mode (BatchNorm uses batch stats)

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

    model->eval();  // Switch to eval mode (BatchNorm uses running stats)

    Variable X_eval(X, false);
    auto logits_final = model->forward(X_eval);

    float final_accuracy = showcase::multiclass_accuracy(logits_final.tensor(), y);
    std::cout << "Final Accuracy: " << (final_accuracy * 100.0f) << "%\n\n";

    std::cout << "Key BatchNorm behaviors:\n";
    std::cout << "  - Training mode: uses batch statistics (mean, var)\n";
    std::cout << "  - Eval mode: uses running statistics (accumulated during training)\n";
    std::cout << "  - Normalizes to zero mean and unit variance\n";
    std::cout << "  - Learnable scale (gamma) and shift (beta) parameters\n";

    std::cout << "\nBatch Normalization demonstrated with Neural Network API!\n";

    finalize();
    return 0;
}
