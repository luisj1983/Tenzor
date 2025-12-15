/**
 * @file autograd.cpp
 * @brief Linear Regression using Tenzor's automatic differentiation
 *
 * This example demonstrates linear regression using Variable and autograd
 * for automatic gradient computation.
 *
 * Usage: ./02_linear_regression_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"

using namespace tenzor;

int main(int argc, char* argv[]) {
    // Parse backend from command line
    Device device = showcase::get_device_from_args(argc, argv);

    // Initialize Tenzor
    initialize();

    showcase::print_header("Linear Regression - Autograd (Automatic Differentiation)", device);

    // Set seed for reproducibility
    manual_seed(42);

    // Generate synthetic dataset: y = 2x + 1 + noise
    int num_samples = 100;
    float true_weight = 2.0f;
    float true_bias = 1.0f;

    // Generate X and y tensors
    auto X_tensor = rand({num_samples, 1}, DType::Float32, device) * 10.0f;
    auto noise = randn({num_samples, 1}, DType::Float32, device) * 0.5f;
    auto y_tensor = X_tensor * true_weight + true_bias + noise;

    showcase::print_tensor_info("Input X", X_tensor);
    showcase::print_tensor_info("Target y", y_tensor);

    std::cout << "True relationship: y = " << true_weight << "x + " << true_bias << "\n";

    // Initialize parameters as Variables with gradient tracking
    auto W_tensor = randn({1, 1}, DType::Float32, device);
    auto b_tensor = zeros({1, 1}, DType::Float32, device);

    Variable W(W_tensor, true);  // requires_grad = true
    Variable b(b_tensor, true);

    showcase::print_section("Initial Parameters");
    std::cout << "W (initial): " << W.tensor().cpu().data<float>()[0]
              << ", requires_grad=" << W.requires_grad() << "\n";
    std::cout << "b (initial): " << b.tensor().cpu().data<float>()[0]
              << ", requires_grad=" << b.requires_grad() << "\n";

    // Training parameters
    float learning_rate = 0.01f;
    int num_epochs = 1000;
    int print_every = 100;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Wrap data as Variables (no grad needed)
        Variable X(X_tensor, false);
        Variable y(y_tensor, false);

        // ============ Forward Pass ============
        // y_pred = X @ W + b
        auto y_pred = matmul(X, W) + b;

        // ============ Compute Loss (MSE) ============
        auto error = y_pred - y;
        auto squared_error = error * error;
        auto loss = mean(squared_error);

        // ============ Backward Pass (Automatic!) ============
        W.zero_grad();
        b.zero_grad();
        loss.backward();

        // ============ Update Weights (Manual SGD) ============
        {
            NoGradGuard no_grad;  // Disable gradient tracking for updates

            auto W_new = W.tensor() - (*W.grad() * learning_rate);
            W = Variable(W_new, true);

            auto b_new = b.tensor() - (*b.grad() * learning_rate);
            b = Variable(b_new, true);
        }

        // Print progress
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float loss_val = loss.tensor().item<float>();
            float w_val = W.tensor().cpu().data<float>()[0];
            float b_val = b.tensor().cpu().data<float>()[0];

            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                      << "Loss: " << loss_val
                      << ", W: " << w_val
                      << ", b: " << b_val << "\n";
        }
    }

    // ============ Final Results ============
    showcase::print_section("Final Results");

    float final_w = W.tensor().cpu().data<float>()[0];
    float final_b = b.tensor().cpu().data<float>()[0];

    std::cout << "Learned parameters:\n";
    std::cout << "  W = " << final_w << " (true: " << true_weight << ")\n";
    std::cout << "  b = " << final_b << " (true: " << true_bias << ")\n";

    // Calculate R-squared
    Variable X_final(X_tensor, false);
    auto y_pred_final = matmul(X_final, W) + b;
    auto ss_res_var = (Variable(y_tensor, false) - y_pred_final);
    auto ss_res = tenzor::sum(ss_res_var * ss_res_var);

    auto y_mean_val = tenzor::mean(y_tensor).item<float>();
    auto y_mean_tensor = full({num_samples, 1}, y_mean_val, DType::Float32, device);
    auto ss_tot = tenzor::sum((y_tensor - y_mean_tensor) * (y_tensor - y_mean_tensor));
    float r_squared = 1.0f - (ss_res.tensor().item<float>() / ss_tot.item<float>());

    std::cout << "\nR-squared: " << r_squared << "\n";

    std::cout << "\nLinear regression solved using autograd!\n";
    std::cout << "Gradients computed automatically - no manual derivation needed.\n";

    finalize();
    return 0;
}
