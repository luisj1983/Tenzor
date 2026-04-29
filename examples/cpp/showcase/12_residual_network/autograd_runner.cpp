/**
 * @file autograd_runner.cpp
 * @brief Implementation of the residual-network autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <cmath>
#include <vector>

namespace tenzor::examples::showcase12 {

int run_resnet_training(int epochs,
                        double* out_initial,
                        double* out_final,
                        ::tenzor::Device device,
                        bool verbose) {
    using namespace ::tenzor;

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

    auto X_tensor = from_data(X_data.data(), {batch_size, input_dim}, device);
    auto y_tensor = from_data(y_data.data(), {batch_size, output_dim}, device);

    auto he  = [&](int64_t fan_in) { return std::sqrt(2.0f / fan_in); };
    auto xav = [&](int64_t fan_in) { return std::sqrt(1.0f / fan_in); };

    Variable W_in(randn({input_dim, hidden_dim}, DType::Float32, device) * he(input_dim), true);
    Variable b_in(zeros({1, hidden_dim}, DType::Float32, device), true);
    Variable W1(randn({hidden_dim, hidden_dim}, DType::Float32, device) * he(hidden_dim), true);
    Variable b1(zeros({1, hidden_dim}, DType::Float32, device), true);
    Variable W2(randn({hidden_dim, hidden_dim}, DType::Float32, device) * he(hidden_dim), true);
    Variable b2(zeros({1, hidden_dim}, DType::Float32, device), true);
    Variable W_out(randn({hidden_dim, output_dim}, DType::Float32, device) * xav(hidden_dim), true);
    Variable b_out(zeros({1, output_dim}, DType::Float32, device), true);

    float lr = 0.02f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable X(X_tensor, false);
        Variable y(y_tensor, false);

        auto a_in = nn::relu(matmul(X, W_in) + b_in);
        auto f_x = matmul(nn::relu(matmul(a_in, W1) + b1), W2) + b2;
        auto a_res = nn::relu(a_in + f_x);
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

        double loss_val = static_cast<double>(loss.tensor().item<float>());
        if (epoch == 0 && out_initial) *out_initial = loss_val;
        final_loss = loss_val;

        if (verbose && ((epoch + 1) % print_every == 0 || epoch == 0)) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << epochs
                      << "] mse=" << loss_val << "\n";
        }
    }
    if (out_final) *out_final = final_loss;
    return 0;
}

}  // namespace tenzor::examples::showcase12
