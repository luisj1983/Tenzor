/**
 * @file neural_network.cpp
 * @brief Residual Network using Tenzor's high-level Neural Network API
 *
 * Demonstrates how easy non-sequential architectures (with skip
 * connections) become with nn::Module: the forward_impl just writes
 * out x + F(x), and everything else is handled by the framework.
 *
 * Task: regress a multi-target sinusoidal signal.
 *
 * Usage: ./12_residual_network_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;

/**
 * @brief Residual block: y = ReLU(x + Linear(ReLU(Linear(x))))
 *
 * Skip connection bypasses the two-layer MLP and is added to its output
 * before the final activation.
 */
class ResidualBlock : public nn::Module {
public:
    ResidualBlock(int64_t dim) {
        fc1 = std::make_shared<nn::Linear>(dim, dim);
        fc2 = std::make_shared<nn::Linear>(dim, dim);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto h = nn::relu(fc1->forward(input));
        auto f = fc2->forward(h);
        return nn::relu(input + f);  // the skip-connection add
    }

private:
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::Linear> fc2;
};

/**
 * @brief ResNet-style regressor: stem -> N residual blocks -> head.
 */
class ResNetRegressor : public nn::Module {
public:
    ResNetRegressor(int64_t input_dim, int64_t hidden_dim,
                    int64_t output_dim, int64_t num_blocks) {
        stem = std::make_shared<nn::Linear>(input_dim, hidden_dim);
        register_module("stem", stem);
        for (int64_t i = 0; i < num_blocks; ++i) {
            auto block = std::make_shared<ResidualBlock>(hidden_dim);
            register_module("block" + std::to_string(i), block);
            blocks.push_back(block);
        }
        head = std::make_shared<nn::Linear>(hidden_dim, output_dim);
        register_module("head", head);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto x = nn::relu(stem->forward(input));
        for (auto& b : blocks) {
            x = b->forward(x);
        }
        return head->forward(x);
    }

private:
    std::shared_ptr<nn::Linear> stem;
    std::vector<std::shared_ptr<ResidualBlock>> blocks;
    std::shared_ptr<nn::Linear> head;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Residual Network - Neural Network API (High-Level)", device);

    manual_seed(42);

    int batch_size = 64;
    int input_dim  = 1;
    int hidden_dim = 32;
    int output_dim = 3;
    int num_blocks = 3;

    std::vector<float> X_data(batch_size);
    std::vector<float> y_data(batch_size * output_dim);
    for (int i = 0; i < batch_size; ++i) {
        float x = (static_cast<float>(i) / batch_size) * 6.0f - 3.0f;
        X_data[i] = x;
        y_data[i * output_dim + 0] = std::sin(x);
        y_data[i * output_dim + 1] = std::cos(x);
        y_data[i * output_dim + 2] = std::sin(2.0f * x);
    }

    auto X = from_data(X_data.data(), {batch_size, input_dim},  device);
    auto y = from_data(y_data.data(), {batch_size, output_dim}, device);

    showcase::print_tensor_info("Input X", X);
    showcase::print_tensor_info("Target y", y);

    auto model = std::make_shared<ResNetRegressor>(
        input_dim, hidden_dim, output_dim, num_blocks);
    model->to(device);

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.005f);
    nn::MSELoss criterion;

    showcase::print_section("Model Architecture");
    std::cout << "Stem:   Linear(" << input_dim << " -> " << hidden_dim << ") + ReLU\n";
    std::cout << "Blocks: " << num_blocks << " x ResidualBlock(" << hidden_dim
              << "), each: ReLU(x + Linear(ReLU(Linear(x))))\n";
    std::cout << "Head:   Linear(" << hidden_dim << " -> " << output_dim << ")\n";
    std::cout << "\nTotal parameters tensors: " << params.size() << "\n";

    int num_epochs = 1000;
    int print_every = 100;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        optimizer.zero_grad();

        Variable x(X, false);
        Variable t(y, false);
        auto y_pred = model->forward(x);
        auto loss = criterion(y_pred, t);

        loss.backward();
        optimizer.step();

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] MSE Loss: " << loss.tensor().item<float>() << "\n";
        }
    }

    showcase::print_section("Final Results");
    model->eval();
    Variable x_eval(X, false);
    auto y_pred = model->forward(x_eval);
    auto final_loss = criterion(y_pred, Variable(y, false));
    std::cout << "Final MSE: " << final_loss.tensor().item<float>() << "\n\n";

    auto X_cpu = X.cpu();
    auto y_cpu = y.cpu();
    auto p_cpu = y_pred.tensor().cpu();
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

    std::cout << "\nResidual network solved using Neural Network API!\n";
    std::cout << "Non-sequential architectures fall out of nn::Module composition.\n";

    finalize();
    return 0;
}
