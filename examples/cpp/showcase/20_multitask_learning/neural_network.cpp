/**
 * @file neural_network.cpp
 * @brief Multi-task learning using nn::Module (shared backbone, two heads)
 *
 * Shows a typical multi-task setup: a Backbone nn::Module feeds both a
 * classification head and a regression head. Combined loss is
 *   alpha * CE + beta * MSE.
 *
 * Usage: ./20_multitask_learning_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

class MultiTaskNet : public nn::Module {
public:
    MultiTaskNet(int64_t in_dim, int64_t hidden) {
        fc1 = std::make_shared<nn::Linear>(in_dim, hidden);
        head_cls = std::make_shared<nn::Linear>(hidden, 1);
        head_reg = std::make_shared<nn::Linear>(hidden, 1);
        register_module("fc1", fc1);
        register_module("head_cls", head_cls);
        register_module("head_reg", head_reg);
    }
    struct Out { Variable logitA, predB; };

    Out forward_both(const Variable& x) {
        auto feat = nn::relu(fc1->forward(x));
        return {head_cls->forward(feat), head_reg->forward(feat)};
    }

    auto forward_impl(const Variable& x) -> Variable override {
        return forward_both(x).logitA;   // default returns classification logits
    }

    std::shared_ptr<nn::Linear> fc1, head_cls, head_reg;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Multi-Task Learning - Neural Network API", device);
    manual_seed(42);

    int N = 128, in_dim = 4, hidden = 16;

    std::vector<float> X_data(N * in_dim);
    std::vector<float> yA_data(N), yB_data(N);
    for (int i = 0; i < N; ++i) {
        float sum = 0;
        for (int d = 0; d < in_dim; ++d) {
            float v = ((rand() % 2000) / 1000.0f) - 1.0f;
            X_data[i * in_dim + d] = v;
            sum += v;
        }
        yA_data[i] = X_data[i * in_dim + 0] > 0 ? 1.0f : 0.0f;
        yB_data[i] = sum;
    }
    auto X  = from_data(X_data.data(), {N, in_dim}, device);
    auto yA = from_data(yA_data.data(),{N, 1},      device);
    auto yB = from_data(yB_data.data(),{N, 1},      device);

    auto model = std::make_shared<MultiTaskNet>(in_dim, hidden);
    model->to(device);

    optim::Adam opt(model->parameters(), 0.02f);
    nn::BCEWithLogitsLoss bce;
    nn::MSELoss mse;

    float alpha = 1.0f, beta = 1.0f;

    int num_epochs = 200;
    int print_every = 20;

    showcase::print_section("Architecture");
    std::cout << "Backbone: Linear(" << in_dim << " -> " << hidden << ") -> ReLU\n";
    std::cout << "Head A:   Linear(" << hidden << " -> 1) [classification logit]\n";
    std::cout << "Head B:   Linear(" << hidden << " -> 1) [regression]\n";
    std::cout << "Combined loss = alpha * BCE + beta * MSE\n";

    showcase::print_section("Training");
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        opt.zero_grad();

        Variable x(X, false), tA(yA, false), tB(yB, false);
        auto out = model->forward_both(x);
        auto lossA = bce(out.logitA, tA);
        auto lossB = mse(out.predB,  tB);
        auto loss = lossA * alpha + lossB * beta;

        loss.backward();
        opt.step();

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            auto l_cpu = out.logitA.tensor().cpu();
            auto y_cpu = yA.cpu();
            int c = 0;
            for (int i = 0; i < N; ++i)
                if ((l_cpu.data<float>()[i] > 0) == (y_cpu.data<float>()[i] > 0.5f)) c++;
            std::cout << "Epoch " << (epoch+1)
                      << "  LossA=" << lossA.tensor().item<float>()
                      << "  AccA=" << (100.0f * c / N) << "%"
                      << "  LossB(MSE)=" << lossB.tensor().item<float>() << "\n";
        }
    }

    std::cout << "\nMulti-task learning solved using Neural Network API!\n";

    finalize();
    return 0;
}
