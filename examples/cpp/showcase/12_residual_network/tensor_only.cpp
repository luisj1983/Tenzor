/**
 * @file tensor_only.cpp
 * @brief Residual Network using raw Tensor operations only
 *
 * Demonstrates the defining feature of a residual block - an identity
 * skip connection that bypasses a stack of transformations:
 *
 *     y = F(x) + x
 *
 * This example does everything by hand: forward pass, loss, and a
 * full manual backward pass with the skip connection's gradient
 * splitting correctly into both branches.
 *
 * Task: regress a multi-target sinusoidal signal.
 *
 * Usage: ./12_residual_network_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

static Tensor relu_tensor(const Tensor& x) {
    return maximum(x, zeros_like(x));
}

static Tensor relu_deriv(const Tensor& z) {
    return (z > zeros_like(z)).to(DType::Float32);
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Residual Network - Tensor Only (Manual Backprop)", device);

    manual_seed(42);

    // Synthetic regression: map x -> [sin(x), cos(x), sin(2x)].
    int batch_size = 64;
    int input_dim  = 1;
    int hidden_dim = 16;
    int output_dim = 3;

    std::vector<float> X_data(batch_size);
    std::vector<float> y_data(batch_size * output_dim);
    for (int i = 0; i < batch_size; ++i) {
        float x = (static_cast<float>(i) / batch_size) * 6.0f - 3.0f;  // [-3, 3]
        X_data[i] = x;
        y_data[i * output_dim + 0] = std::sin(x);
        y_data[i * output_dim + 1] = std::cos(x);
        y_data[i * output_dim + 2] = std::sin(2.0f * x);
    }

    auto X = from_data(X_data.data(), {batch_size, input_dim},  device);
    auto y = from_data(y_data.data(), {batch_size, output_dim}, device);

    showcase::print_tensor_info("Input X", X);
    showcase::print_tensor_info("Target y", y);

    // Weights
    // Stem: input_dim -> hidden_dim
    auto W_in = randn({input_dim, hidden_dim}, DType::Float32, device)
              * std::sqrt(2.0f / input_dim);
    auto b_in = zeros({1, hidden_dim}, DType::Float32, device);

    // Residual block: hidden -> hidden -> hidden, with skip x + F(x)
    auto W1 = randn({hidden_dim, hidden_dim}, DType::Float32, device)
            * std::sqrt(2.0f / hidden_dim);
    auto b1 = zeros({1, hidden_dim}, DType::Float32, device);
    auto W2 = randn({hidden_dim, hidden_dim}, DType::Float32, device)
            * std::sqrt(2.0f / hidden_dim);
    auto b2 = zeros({1, hidden_dim}, DType::Float32, device);

    // Head: hidden -> output
    auto W_out = randn({hidden_dim, output_dim}, DType::Float32, device)
               * std::sqrt(1.0f / hidden_dim);
    auto b_out = zeros({1, output_dim}, DType::Float32, device);

    showcase::print_section("Architecture");
    std::cout << "Linear(" << input_dim << " -> " << hidden_dim << ") -> ReLU\n";
    std::cout << "ResBlock: Linear+ReLU -> Linear, skip connection y = F(x) + x\n";
    std::cout << "ReLU\n";
    std::cout << "Linear(" << hidden_dim << " -> " << output_dim << ")\n";
    std::cout << "\nManual backprop. Note the skip connection contributes to gradient twice:\n";
    std::cout << "once through F(x) and once through the identity path.\n";

    float lr = 0.02f;
    int num_epochs = 2000;
    int print_every = 200;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ---- Forward ----
        auto z_in = matmul(X, W_in) + b_in;
        auto a_in = relu_tensor(z_in);                       // stem output, also residual input

        auto z1 = matmul(a_in, W1) + b1;
        auto a1 = relu_tensor(z1);
        auto z2 = matmul(a1, W2) + b2;                       // F(x)
        auto a_res = a_in + z2;                              // skip: y = F(x) + x
        auto a_res_relu = relu_tensor(a_res);

        auto y_pred = matmul(a_res_relu, W_out) + b_out;     // regression output

        // Loss (MSE)
        auto err = y_pred - y;
        auto loss = tenzor::mean(err * err);
        float loss_val = loss.item<float>();

        // ---- Backward ----
        float n = static_cast<float>(batch_size * output_dim);
        auto dL_dypred = err * (2.0f / n);

        auto dL_dW_out = matmul(a_res_relu.transpose(0, 1), dL_dypred);
        auto dL_db_out = tenzor::sum(dL_dypred, 0, true);

        auto dL_da_res_relu = matmul(dL_dypred, W_out.transpose(0, 1));
        auto dL_da_res = dL_da_res_relu * relu_deriv(a_res);

        // Skip: gradient flows to both F(x) branch and identity (a_in) branch
        auto dL_dz2    = dL_da_res;              // F(x) branch: z2 is just the last linear
        auto dL_da_in_skip = dL_da_res;          // identity branch passes grad straight to a_in

        auto dL_dW2 = matmul(a1.transpose(0, 1), dL_dz2);
        auto dL_db2 = tenzor::sum(dL_dz2, 0, true);

        auto dL_da1 = matmul(dL_dz2, W2.transpose(0, 1));
        auto dL_dz1 = dL_da1 * relu_deriv(z1);
        auto dL_dW1 = matmul(a_in.transpose(0, 1), dL_dz1);
        auto dL_db1 = tenzor::sum(dL_dz1, 0, true);

        // a_in gets both the identity and the through-F(x) gradient
        auto dL_da_in = matmul(dL_dz1, W1.transpose(0, 1)) + dL_da_in_skip;
        auto dL_dz_in = dL_da_in * relu_deriv(z_in);
        auto dL_dW_in = matmul(X.transpose(0, 1), dL_dz_in);
        auto dL_db_in = tenzor::sum(dL_dz_in, 0, true);

        // ---- Update ----
        W_out = W_out - dL_dW_out * lr;  b_out = b_out - dL_db_out * lr;
        W2    = W2    - dL_dW2    * lr;  b2    = b2    - dL_db2    * lr;
        W1    = W1    - dL_dW1    * lr;  b1    = b1    - dL_db1    * lr;
        W_in  = W_in  - dL_dW_in  * lr;  b_in  = b_in  - dL_db_in  * lr;

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] MSE Loss: " << loss_val << "\n";
        }
    }

    showcase::print_section("Final Results");
    // Show a few predictions
    auto z_in = matmul(X, W_in) + b_in;
    auto a_in = relu_tensor(z_in);
    auto z1 = matmul(a_in, W1) + b1;
    auto a1 = relu_tensor(z1);
    auto z2 = matmul(a1, W2) + b2;
    auto a_res_relu = relu_tensor(a_in + z2);
    auto y_pred = matmul(a_res_relu, W_out) + b_out;
    auto err = y_pred - y;
    std::cout << "Final MSE: " << tenzor::mean(err * err).item<float>() << "\n\n";

    auto y_cpu = y.cpu();
    auto p_cpu = y_pred.cpu();
    auto X_cpu = X.cpu();
    std::cout << "x\ttarget (sin, cos, sin2x)\t\tpredicted\n";
    for (int i = 0; i < batch_size; i += 8) {
        std::cout << X_cpu.data<float>()[i] << "\t("
                  << y_cpu.data<float>()[i * output_dim + 0] << ", "
                  << y_cpu.data<float>()[i * output_dim + 1] << ", "
                  << y_cpu.data<float>()[i * output_dim + 2] << ")\t("
                  << p_cpu.data<float>()[i * output_dim + 0] << ", "
                  << p_cpu.data<float>()[i * output_dim + 1] << ", "
                  << p_cpu.data<float>()[i * output_dim + 2] << ")\n";
    }

    std::cout << "\nResidual block demonstrated with raw tensors!\n";
    std::cout << "Skip connection: grad flows through both F(x) and the identity path.\n";

    finalize();
    return 0;
}
