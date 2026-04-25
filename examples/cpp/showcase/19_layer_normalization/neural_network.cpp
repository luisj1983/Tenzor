/**
 * @file neural_network.cpp
 * @brief Layer Normalization with the nn::LayerNorm module
 *
 * Swaps the hand-rolled layer_norm() from the autograd tier for
 * nn::LayerNorm - one line, same behavior.
 *
 * Usage: ./19_layer_normalization_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

class LNNet : public nn::Module {
public:
    LNNet(int64_t in_dim, int64_t hidden, int64_t num_classes) {
        fc1 = std::make_shared<nn::Linear>(in_dim, hidden);
        ln  = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{hidden});
        fc2 = std::make_shared<nn::Linear>(hidden, num_classes);
        register_module("fc1", fc1);
        register_module("ln",  ln);
        register_module("fc2", fc2);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        return fc2->forward(nn::relu(ln->forward(fc1->forward(x))));
    }

private:
    std::shared_ptr<nn::Linear> fc1, fc2;
    std::shared_ptr<nn::LayerNorm> ln;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Layer Normalization - Neural Network API", device);
    manual_seed(42);

    int N = 128;
    int in_dim = 4;
    int hidden = 16;
    int num_classes = 2;

    std::vector<float> X_data(N * in_dim);
    std::vector<int64_t> y_data(N);
    for (int i = 0; i < N; ++i) {
        float sign = (i % 2) ? 1.0f : -1.0f;
        y_data[i] = (i % 2);
        for (int d = 0; d < in_dim; ++d) {
            float scale = std::pow(10.0f, static_cast<float>(d));
            X_data[i * in_dim + d] = scale * (sign * 0.3f + ((rand() % 200) / 500.0f - 0.2f));
        }
    }
    auto X = from_data(X_data.data(), {N, in_dim}, device);
    auto y = from_data(y_data.data(), {N}, device);

    auto model = std::make_shared<LNNet>(in_dim, hidden, num_classes);
    model->to(device);

    optim::Adam opt(model->parameters(), 0.02f);
    nn::CrossEntropyLoss criterion;

    showcase::print_section("Architecture");
    std::cout << "Linear(" << in_dim << " -> " << hidden << ") -> LayerNorm(" << hidden
              << ") -> ReLU -> Linear(" << hidden << " -> " << num_classes << ")\n";

    int num_epochs = 200;
    int print_every = 20;

    showcase::print_section("Training");
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        opt.zero_grad();
        Variable x(X, false);
        auto logits = model->forward(x);
        auto loss = criterion(logits, y);  // CrossEntropyLoss target is raw Tensor
        loss.backward();
        opt.step();
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float acc = showcase::multiclass_accuracy(logits.tensor(), y);
            std::cout << "Epoch [" << (epoch+1) << "/" << num_epochs
                      << "] Loss: " << loss.tensor().item<float>()
                      << "  Acc: " << (acc * 100) << "%\n";
        }
    }

    showcase::print_section("Final Results");
    model->eval();
    Variable xe(X, false);
    auto logits = model->forward(xe);
    float acc = showcase::multiclass_accuracy(logits.tensor(), y);
    std::cout << "Final accuracy: " << (acc * 100) << "%\n";
    std::cout << "\nLayer normalization solved using Neural Network API!\n";

    finalize();
    return 0;
}
