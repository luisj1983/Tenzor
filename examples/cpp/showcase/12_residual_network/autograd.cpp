/**
 * @file autograd.cpp
 * @brief Residual Network using Tenzor's autograd
 *
 * Demonstrates a residual block (y = F(x) + x) using Variable and autograd.
 * With the computation graph, the skip connection's gradient duplication
 * happens for free: the backward pass routes gradient through both the F(x)
 * branch and the identity branch automatically.
 *
 * Task: regress a multi-target sinusoidal signal.
 *
 * Usage: ./12_residual_network_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Residual Network - Autograd (Auto-Differentiation)", device);

    manual_seed(42);

    int batch_size = 64;
    int input_dim  = 1;
    int hidden_dim = 16;
    int output_dim = 3;

    std::vector<float> X_data(batch_size);
    std::vector<float> y_data(batch_size * output_dim);
    for (int i = 0; i < batch_size; ++i) {
        float x = (static_cast<float>(i) / batch_size) * 6.0f - 3.0f;
        X_data[i] = x;
        y_data[i * output_dim + 0] = std::sin(x);
        y_data[i * output_dim + 1] = std::cos(x);
        y_data[i * output_dim + 2] = std::sin(2.0f * x);
    }

    auto X_tensor = from_data(X_data.data(), {batch_size, input_dim},  device);
    auto y_tensor = from_data(y_data.data(), {batch_size, output_dim}, device);

    showcase::print_tensor_info("Input X", X_tensor);
    showcase::print_tensor_info("Target y", y_tensor);

    auto he  = [&](int64_t fan_in) { return std::sqrt(2.0f / fan_in); };
    auto xav = [&](int64_t fan_in) { return std::sqrt(1.0f / fan_in); };

    Variable W_in(randn({input_dim, hidden_dim},  DType::Float32, device) * he(input_dim),   true);
    Variable b_in(zeros({1, hidden_dim},           DType::Float32, device), true);

    Variable W1(randn({hidden_dim, hidden_dim},   DType::Float32, device) * he(hidden_dim),  true);
    Variable b1(zeros({1, hidden_dim},            DType::Float32, device), true);
    Variable W2(randn({hidden_dim, hidden_dim},   DType::Float32, device) * he(hidden_dim),  true);
    Variable b2(zeros({1, hidden_dim},            DType::Float32, device), true);

    Variable W_out(randn({hidden_dim, output_dim},DType::Float32, device) * xav(hidden_dim), true);
    Variable b_out(zeros({1, output_dim},         DType::Float32, device), true);

    showcase::print_section("Architecture");
    std::cout << "Linear(" << input_dim << " -> " << hidden_dim << ") -> ReLU\n";
    std::cout << "ResBlock: Linear -> ReLU -> Linear, skip y = F(x) + x\n";
    std::cout << "ReLU -> Linear(" << hidden_dim << " -> " << output_dim << ")\n";
    std::cout << "\nAutograd handles the skip-connection gradient duplication automatically.\n";

    float lr = 0.02f;
    int num_epochs = 2000;
    int print_every = 200;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable X(X_tensor, false);
        Variable y(y_tensor, false);

        // Stem
        auto a_in = nn::relu(matmul(X, W_in) + b_in);

        // Residual block: F(x) + x
        auto f_x = matmul(nn::relu(matmul(a_in, W1) + b1), W2) + b2;
        auto a_res = nn::relu(a_in + f_x);

        // Head
        auto y_pred = matmul(a_res, W_out) + b_out;

        auto err = y_pred - y;
        auto loss = mean(err * err);

        W_in.zero_grad();  b_in.zero_grad();
        W1.zero_grad();    b1.zero_grad();
        W2.zero_grad();    b2.zero_grad();
        W_out.zero_grad(); b_out.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W_in  = Variable(W_in.tensor()  - (*W_in.grad()  * lr), true);
            b_in  = Variable(b_in.tensor()  - (*b_in.grad()  * lr), true);
            W1    = Variable(W1.tensor()    - (*W1.grad()    * lr), true);
            b1    = Variable(b1.tensor()    - (*b1.grad()    * lr), true);
            W2    = Variable(W2.tensor()    - (*W2.grad()    * lr), true);
            b2    = Variable(b2.tensor()    - (*b2.grad()    * lr), true);
            W_out = Variable(W_out.tensor() - (*W_out.grad() * lr), true);
            b_out = Variable(b_out.tensor() - (*b_out.grad() * lr), true);
        }

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] MSE Loss: " << loss.tensor().item<float>() << "\n";
        }
    }

    showcase::print_section("Final Results");

    Variable X(X_tensor, false);
    auto a_in = nn::relu(matmul(X, W_in) + b_in);
    auto f_x = matmul(nn::relu(matmul(a_in, W1) + b1), W2) + b2;
    auto y_pred = matmul(nn::relu(a_in + f_x), W_out) + b_out;

    auto err = y_pred.tensor() - y_tensor;
    std::cout << "Final MSE: " << tenzor::mean(err * err).item<float>() << "\n\n";

    auto y_cpu = y_tensor.cpu();
    auto p_cpu = y_pred.tensor().cpu();
    auto X_cpu = X_tensor.cpu();
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

    std::cout << "\nResidual block demonstrated with autograd!\n";
    std::cout << "The a_in + f_x add node routes backward grad to both branches automatically.\n";

    finalize();
    return 0;
}
