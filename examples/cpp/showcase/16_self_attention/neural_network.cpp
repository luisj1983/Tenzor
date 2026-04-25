/**
 * @file neural_network.cpp
 * @brief Multi-head self-attention using nn::MultiheadAttention
 *
 * Shows the high-level API for attention. Tensor parts come from the
 * same copy-shift task as the autograd version, so you can compare
 * the training curves.
 *
 * Usage: ./16_self_attention_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;

class SelfAttnBlock : public nn::Module {
public:
    SelfAttnBlock(int64_t d_model, int64_t num_heads) {
        // batch_first = true: inputs are (B, T, D)
        attn = std::make_shared<nn::MultiheadAttention>(
            d_model, num_heads, /*dropout*/ 0.0, /*bias*/ true,
            /*add_bias_kv*/ false, /*add_zero_attn*/ false,
            /*kdim*/ 0, /*vdim*/ 0, /*batch_first*/ true);
        out = std::make_shared<nn::Linear>(d_model, d_model);
        register_module("attn", attn);
        register_module("out", out);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        // Self-attention: Q = K = V = x
        auto attn_out = attn->forward(x, x, x).first;   // (B, T, D)
        return out->forward(attn_out);
    }

private:
    std::shared_ptr<nn::MultiheadAttention> attn;
    std::shared_ptr<nn::Linear> out;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Self-Attention - Neural Network API", device);
    manual_seed(42);

    int batch = 4;
    int seq   = 6;
    int d_model = 8;
    int num_heads = 2;

    std::vector<float> X_data(batch * seq * d_model);
    std::vector<float> y_data(batch * seq * d_model);
    for (int i = 0; i < batch * seq * d_model; ++i) {
        X_data[i] = ((i * 7919) % 997) / 497.0f - 1.0f;
    }
    for (int b = 0; b < batch; ++b)
    for (int t = 0; t < seq;   ++t)
    for (int d = 0; d < d_model; ++d) {
        y_data[(b * seq + t) * d_model + d] = X_data[(b * seq + (t + 1) % seq) * d_model + d];
    }

    auto X = from_data(X_data.data(), {batch, seq, d_model}, device);
    auto y = from_data(y_data.data(), {batch, seq, d_model}, device);

    auto model = std::make_shared<SelfAttnBlock>(d_model, num_heads);
    model->to(device);

    auto params = model->parameters();
    optim::Adam opt(params, 0.01f);
    nn::MSELoss criterion;

    showcase::print_section("Model Architecture");
    std::cout << "MultiheadAttention(d_model=" << d_model << ", heads=" << num_heads
              << ") -> Linear(" << d_model << " -> " << d_model << ")\n";

    int num_epochs = 400;
    int print_every = 40;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        opt.zero_grad();
        Variable x(X, false), t(y, false);
        auto p = model->forward(x);
        auto loss = criterion(p, t);
        loss.backward();
        opt.step();
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] MSE: " << loss.tensor().item<float>() << "\n";
        }
    }

    showcase::print_section("Final Results");
    model->eval();
    auto pred = model->forward(Variable(X, false));
    auto err = pred.tensor() - y;
    std::cout << "Final MSE: " << tenzor::mean(err * err).item<float>() << "\n";
    std::cout << "\nMulti-head self-attention solved with the Neural Network API!\n";

    finalize();
    return 0;
}
