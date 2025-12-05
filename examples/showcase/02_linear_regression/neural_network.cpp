/**
 * @file neural_network.cpp
 * @brief Linear Regression using Tenzor's high-level Neural Network API
 *
 * This example demonstrates linear regression using nn::Module,
 * nn::Linear layer, MSELoss, and optimizers for a clean interface.
 *
 * Usage: ./02_linear_regression_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

/**
 * @brief Simple linear regression model
 *
 * Just a single Linear layer with no activation.
 */
class LinearRegressor : public nn::Module {
public:
    LinearRegressor(int64_t input_features = 1, int64_t output_features = 1) {
        linear = std::make_shared<nn::Linear>(input_features, output_features);
        register_module("linear", linear);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        return linear->forward(input);
    }

    // Access learned parameters
    float get_weight() const {
        return linear->weight()->tensor().cpu().data<float>()[0];
    }

    float get_bias() const {
        if (linear->has_bias()) {
            return linear->bias()->tensor().cpu().data<float>()[0];
        }
        return 0.0f;
    }

private:
    std::shared_ptr<nn::Linear> linear;
};

int main(int argc, char* argv[]) {
    // Parse backend from command line
    Device device = showcase::get_device_from_args(argc, argv);

    // Initialize Tenzor
    initialize();

    showcase::print_header("Linear Regression - Neural Network API (High-Level)", device);

    // Set seed for reproducibility
    manual_seed(42);

    // Generate synthetic dataset: y = 2x + 1 + noise
    int num_samples = 100;
    float true_weight = 2.0f;
    float true_bias = 1.0f;

    auto X = rand({num_samples, 1}, DType::Float32, device) * 10.0f;
    auto noise = randn({num_samples, 1}, DType::Float32, device) * 0.5f;
    auto y = X * true_weight + true_bias + noise;

    showcase::print_tensor_info("Input X", X);
    showcase::print_tensor_info("Target y", y);

    std::cout << "True relationship: y = " << true_weight << "x + " << true_bias << "\n";

    // Create model
    auto model = std::make_shared<LinearRegressor>(1, 1);
    model->to(device);

    // Create optimizer (SGD)
    auto params = model->parameters();
    optim::SGD optimizer(params, 0.01f);  // learning_rate = 0.01

    // Create loss function
    nn::MSELoss criterion;

    showcase::print_section("Model Architecture");
    std::cout << "LinearRegressor:\n";
    std::cout << "  linear: Linear(1, 1)\n";
    std::cout << "  No activation (pure linear transformation)\n";
    std::cout << "\nTotal parameters: " << params.size() << " (weight + bias)\n";

    // Training parameters
    int num_epochs = 1000;
    int print_every = 100;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();

        // Zero gradients
        optimizer.zero_grad();

        // Forward pass
        Variable input(X, false);
        auto output = model->forward(input);

        // Compute loss
        Variable target(y, false);
        auto loss = criterion(output, target);

        // Backward pass
        loss.backward();

        // Update weights
        optimizer.step();

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();
            float w_val = model->get_weight();
            float b_val = model->get_bias();

            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                      << "Loss: " << loss_val
                      << ", W: " << w_val
                      << ", b: " << b_val << "\n";
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    model->eval();

    float final_w = model->get_weight();
    float final_b = model->get_bias();

    std::cout << "Learned parameters:\n";
    std::cout << "  W = " << final_w << " (true: " << true_weight << ")\n";
    std::cout << "  b = " << final_b << " (true: " << true_bias << ")\n";

    // Calculate R-squared
    Variable X_eval(X, false);
    auto y_pred = model->forward(X_eval);

    auto residuals = y - y_pred.tensor();
    auto ss_res = tenzor::sum(residuals * residuals);

    auto y_mean_val = tenzor::mean(y).item<float>();
    auto y_mean_tensor = full({num_samples, 1}, y_mean_val, DType::Float32, device);
    auto ss_tot = tenzor::sum((y - y_mean_tensor) * (y - y_mean_tensor));
    float r_squared = 1.0f - (ss_res.item<float>() / ss_tot.item<float>());

    std::cout << "\nR-squared: " << r_squared << "\n";

    // ============ Making Predictions ============
    showcase::print_section("Sample Predictions");

    // Predict for some test values
    float test_values[] = {0.0f, 2.5f, 5.0f, 7.5f, 10.0f};
    auto test_input = from_data(test_values, {5, 1}, device);
    Variable test_var(test_input, false);
    auto predictions = model->forward(test_var);

    std::cout << "X\t\tPredicted\tExpected (y=2x+1)\n";
    std::cout << "-------------------------------------------\n";

    auto pred_cpu = predictions.tensor().cpu();
    for (int i = 0; i < 5; ++i) {
        float x = test_values[i];
        float pred = pred_cpu.data<float>()[i];
        float expected = true_weight * x + true_bias;
        std::cout << x << "\t\t" << pred << "\t\t" << expected << "\n";
    }

    std::cout << "\nLinear regression solved using Neural Network API!\n";
    std::cout << "Clean interface with nn::Module, SGD optimizer, and MSELoss.\n";

    finalize();
    return 0;
}
