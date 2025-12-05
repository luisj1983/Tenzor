/**
 * @file tensor_only.cpp
 * @brief Linear Regression using raw Tensor operations only
 *
 * This example demonstrates linear regression (y = wx + b) using
 * only tensor operations with manual gradient computation.
 *
 * Linear regression is the simplest machine learning model:
 * - Single layer with no activation function
 * - MSE loss for continuous output
 *
 * Usage: ./02_linear_regression_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

int main(int argc, char* argv[]) {
    // Parse backend from command line
    Device device = showcase::get_device_from_args(argc, argv);

    // Initialize Tenzor
    initialize();

    showcase::print_header("Linear Regression - Tensor Only (Manual Backprop)", device);

    // Set seed for reproducibility
    manual_seed(42);

    // Generate synthetic dataset: y = 2x + 1 + noise
    int num_samples = 100;
    float true_weight = 2.0f;
    float true_bias = 1.0f;

    // Generate X values uniformly from 0 to 10
    auto X = rand({num_samples, 1}, DType::Float32, device) * 10.0f;

    // Generate y = 2x + 1 + noise
    auto noise = randn({num_samples, 1}, DType::Float32, device) * 0.5f;
    auto y = X * true_weight + true_bias + noise;

    showcase::print_tensor_info("Input X", X);
    showcase::print_tensor_info("Target y", y);

    std::cout << "True relationship: y = " << true_weight << "x + " << true_bias << "\n";

    // Initialize weights randomly
    auto W = randn({1, 1}, DType::Float32, device);  // Weight
    auto b = zeros({1, 1}, DType::Float32, device);   // Bias

    showcase::print_section("Initial Parameters");
    std::cout << "W (initial): " << W.cpu().data<float>()[0] << "\n";
    std::cout << "b (initial): " << b.cpu().data<float>()[0] << "\n";

    // Training parameters
    float learning_rate = 0.01f;
    int num_epochs = 1000;
    int print_every = 100;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ============ Forward Pass ============
        // y_pred = X @ W + b
        auto y_pred = matmul(X, W) + b;  // Broadcasting b

        // ============ Compute Loss (MSE) ============
        auto error = y_pred - y;
        auto squared_error = error * error;
        auto loss = tenzor::mean(squared_error);
        float loss_val = loss.item<float>();

        // ============ Backward Pass (Manual Gradient Computation) ============
        // MSE Loss: L = (1/n) * sum((y_pred - y)^2)
        // dL/dy_pred = (2/n) * (y_pred - y)
        float n = static_cast<float>(num_samples);
        auto dL_dy_pred = error * (2.0f / n);  // (num_samples, 1)

        // y_pred = X @ W + b
        // dy_pred/dW = X^T
        // dL/dW = X^T @ dL/dy_pred
        auto dL_dW = matmul(X.transpose(0, 1), dL_dy_pred);  // (1, 1)

        // dy_pred/db = 1
        // dL/db = sum(dL/dy_pred)
        auto dL_db = tenzor::sum(dL_dy_pred, 0, true);  // (1, 1)

        // ============ Update Weights (SGD) ============
        W = W - dL_dW * learning_rate;
        b = b - dL_db * learning_rate;

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float w_val = W.cpu().data<float>()[0];
            float b_val = b.cpu().data<float>()[0];
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                      << "Loss: " << loss_val
                      << ", W: " << w_val
                      << ", b: " << b_val << "\n";
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    float final_w = W.cpu().data<float>()[0];
    float final_b = b.cpu().data<float>()[0];

    std::cout << "Learned parameters:\n";
    std::cout << "  W = " << final_w << " (true: " << true_weight << ")\n";
    std::cout << "  b = " << final_b << " (true: " << true_bias << ")\n";

    // Calculate R-squared
    auto y_pred_final = matmul(X, W) + b;
    auto ss_res = tenzor::sum((y - y_pred_final) * (y - y_pred_final));
    auto y_mean = tenzor::mean(y);
    auto y_mean_tensor = full({num_samples, 1}, y_mean.item<float>(), DType::Float32, device);
    auto ss_tot = tenzor::sum((y - y_mean_tensor) * (y - y_mean_tensor));
    float r_squared = 1.0f - (ss_res.item<float>() / ss_tot.item<float>());

    std::cout << "\nR-squared: " << r_squared << "\n";

    std::cout << "\nLinear regression solved using raw tensors!\n";
    std::cout << "This demonstrates manual gradient computation for the simplest model.\n";

    finalize();
    return 0;
}
