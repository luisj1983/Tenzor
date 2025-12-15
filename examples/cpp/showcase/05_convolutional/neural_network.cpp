/**
 * @file neural_network.cpp
 * @brief Convolutional Neural Network using Tenzor's high-level Neural Network API
 *
 * This example demonstrates a proper CNN using nn::Conv2d, nn::MaxPool2d,
 * and other high-level layers for image classification.
 *
 * Usage: ./05_convolutional_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

/**
 * @brief Simple CNN for image classification
 *
 * Architecture:
 * - Conv1: 1->8 channels, 3x3 kernel
 * - ReLU + MaxPool2d(2x2)
 * - Conv2: 8->16 channels, 3x3 kernel
 * - ReLU + MaxPool2d(2x2)
 * - Flatten + FC
 */
class SimpleCNN : public nn::Module {
public:
    SimpleCNN(int64_t num_classes) {
        // Convolutional layers
        conv1 = std::make_shared<nn::Conv2d>(1, 8, 3, 1, 1);   // (1, H, W) -> (8, H, W)
        conv2 = std::make_shared<nn::Conv2d>(8, 16, 3, 1, 1);  // (8, H, W) -> (16, H, W)

        // Pooling layer
        pool = std::make_shared<nn::MaxPool2d>(2, 2);

        // Flatten layer
        flatten = std::make_shared<nn::Flatten>();

        // FC layer (input size depends on image dimensions after pooling)
        // For 8x8 input: after two pool(2) -> 8/2/2 = 2
        // Features: 16 * 2 * 2 = 64
        fc = std::make_shared<nn::Linear>(64, num_classes);

        // Register modules
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("pool", pool);
        register_module("flatten", flatten);
        register_module("fc", fc);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        // Conv1 + ReLU + Pool
        auto x = pool->forward(nn::relu(conv1->forward(input)));

        // Conv2 + ReLU + Pool
        x = pool->forward(nn::relu(conv2->forward(x)));

        // Flatten
        x = flatten->forward(x);

        // FC layer (outputs logits)
        x = fc->forward(x);

        return x;
    }

private:
    std::shared_ptr<nn::Conv2d> conv1;
    std::shared_ptr<nn::Conv2d> conv2;
    std::shared_ptr<nn::MaxPool2d> pool;
    std::shared_ptr<nn::Flatten> flatten;
    std::shared_ptr<nn::Linear> fc;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Convolutional NN - Neural Network API (High-Level)", device);

    manual_seed(42);

    // Generate synthetic image-like data
    int batch_size = 32;
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

    // Create model
    auto model = std::make_shared<SimpleCNN>(num_classes);
    model->to(device);

    // Create optimizer
    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);

    // Create loss function
    nn::CrossEntropyLoss criterion;

    showcase::print_section("Model Architecture");
    std::cout << "SimpleCNN:\n";
    std::cout << "  conv1: Conv2d(1, 8, kernel=3, padding=1)\n";
    std::cout << "  ReLU + MaxPool2d(2)\n";
    std::cout << "  conv2: Conv2d(8, 16, kernel=3, padding=1)\n";
    std::cout << "  ReLU + MaxPool2d(2)\n";
    std::cout << "  Flatten\n";
    std::cout << "  fc: Linear(64, " << num_classes << ")\n";
    std::cout << "\nTotal parameters: " << params.size() << "\n";

    // Training parameters
    int num_epochs = 100;
    int print_every = 10;

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

    // Visualize first few predictions
    showcase::print_section("Sample Predictions");

    auto logits_cpu = logits_final.tensor().cpu();
    auto y_cpu = y.cpu();
    const float* logit_data = logits_cpu.data<float>();
    const int64_t* target_data = y_cpu.data<int64_t>();

    std::cout << "Sample\tTrue\tPredicted\tCorrect\n";
    std::cout << "------------------------------------\n";
    for (int i = 0; i < std::min(10, batch_size); ++i) {
        int true_class = target_data[i];
        int pred_class = (logit_data[i * num_classes + 1] > logit_data[i * num_classes]) ? 1 : 0;
        bool correct = (true_class == pred_class);

        std::cout << i << "\t" << true_class << "\t" << pred_class << "\t\t"
                  << (correct ? "Yes" : "No") << "\n";
    }

    std::cout << "\nCNN solved using Neural Network API!\n";
    std::cout << "Uses nn::Conv2d with optimized convolution kernels.\n";

    finalize();
    return 0;
}
