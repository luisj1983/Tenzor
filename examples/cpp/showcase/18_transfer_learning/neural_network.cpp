/**
 * @file neural_network.cpp
 * @brief Transfer learning with the nn::Module API
 *
 * Stage 1: train a backbone + head A on task A.
 * Stage 2: freeze the backbone via requires_grad(false), train only
 *          the new head B on task B.
 *
 * Shows how to compose models, save/re-use a trained backbone, and
 * control which parameters receive gradient updates.
 *
 * Usage: ./18_transfer_learning_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

class Backbone : public nn::Module {
public:
    Backbone(int64_t in_dim, int64_t hidden) {
        fc1 = std::make_shared<nn::Linear>(in_dim, hidden);
        fc2 = std::make_shared<nn::Linear>(hidden, hidden);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }
    auto forward_impl(const Variable& x) -> Variable override {
        return nn::relu(fc2->forward(nn::relu(fc1->forward(x))));
    }
    std::shared_ptr<nn::Linear> fc1, fc2;
};

class FullModel : public nn::Module {
public:
    FullModel(std::shared_ptr<Backbone> bb, int64_t hidden) : bb_(bb) {
        head = std::make_shared<nn::Linear>(hidden, 1);
        register_module("backbone", bb_);
        register_module("head", head);
    }
    auto forward_impl(const Variable& x) -> Variable override {
        return head->forward(bb_->forward(x));
    }
    std::shared_ptr<Backbone> bb_;
    std::shared_ptr<nn::Linear> head;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Transfer Learning - Neural Network API", device);
    manual_seed(42);

    int N = 128;
    int in_dim = 4;
    int hidden = 16;

    std::vector<float> X_data(N * in_dim);
    std::vector<float> yA_data(N), yB_data(N);
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < in_dim; ++d) {
            X_data[i * in_dim + d] = ((rand() % 2000) / 1000.0f) - 1.0f;
        }
        yA_data[i] = X_data[i * in_dim + 0] > 0 ? 1.0f : 0.0f;
        yB_data[i] = std::abs(X_data[i * in_dim + 0]) > 0.5f ? 1.0f : 0.0f;
    }
    auto X  = from_data(X_data.data(), {N, in_dim}, device);
    auto yA = from_data(yA_data.data(),{N, 1},      device);
    auto yB = from_data(yB_data.data(),{N, 1},      device);

    auto backbone = std::make_shared<Backbone>(in_dim, hidden);
    auto model_A = std::make_shared<FullModel>(backbone, hidden);
    model_A->to(device);

    nn::BCEWithLogitsLoss bce;
    optim::Adam optA(model_A->parameters(), 0.02f);

    // -------- Stage 1: train full model_A on task A --------
    showcase::print_section("Stage 1: pretrain on task A (all params trainable)");
    for (int epoch = 0; epoch < 200; ++epoch) {
        model_A->train();
        optA.zero_grad();
        Variable x(X, false), y(yA, false);
        auto logit = model_A->forward(x);
        auto loss = bce(logit, y);
        loss.backward();
        optA.step();

        if ((epoch + 1) % 40 == 0 || epoch == 0) {
            auto l_cpu = logit.tensor().cpu();
            auto y_cpu = yA.cpu();
            int c = 0;
            for (int i = 0; i < N; ++i)
                if ((l_cpu.data<float>()[i] > 0) == (y_cpu.data<float>()[i] > 0.5f)) c++;
            std::cout << "[TaskA] epoch " << (epoch+1) << "  loss=" << loss.tensor().item<float>()
                      << "  acc=" << (100.0f * c / N) << "%\n";
        }
    }

    // -------- Stage 2: freeze backbone, train a new head on task B --------
    showcase::print_section("Stage 2: freeze backbone, train head B on task B");
    // Build a fresh FullModel that shares the trained backbone.
    auto model_B = std::make_shared<FullModel>(backbone, hidden);
    model_B->to(device);

    // Freeze backbone params: mark them as not requiring grad.
    for (auto& p : backbone->parameters()) {
        p->set_requires_grad(false);
    }

    // Only train head B - pass only the head's parameters to the optimizer.
    optim::Adam optB(model_B->head->parameters(), 0.02f);

    for (int epoch = 0; epoch < 200; ++epoch) {
        model_B->train();
        optB.zero_grad();
        Variable x(X, false), y(yB, false);
        auto logit = model_B->forward(x);
        auto loss = bce(logit, y);
        loss.backward();
        optB.step();

        if ((epoch + 1) % 40 == 0 || epoch == 0) {
            auto l_cpu = logit.tensor().cpu();
            auto y_cpu = yB.cpu();
            int c = 0;
            for (int i = 0; i < N; ++i)
                if ((l_cpu.data<float>()[i] > 0) == (y_cpu.data<float>()[i] > 0.5f)) c++;
            std::cout << "[TaskB] epoch " << (epoch+1) << "  loss=" << loss.tensor().item<float>()
                      << "  acc=" << (100.0f * c / N) << "%\n";
        }
    }

    std::cout << "\nTransfer learning solved using Neural Network API!\n";
    std::cout << "set_requires_grad(false) on backbone params freezes them for stage 2.\n";

    finalize();
    return 0;
}
