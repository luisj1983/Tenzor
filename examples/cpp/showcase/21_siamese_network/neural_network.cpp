/**
 * @file neural_network.cpp
 * @brief Siamese network using nn::Module with contrastive loss
 *
 * A single encoder module is called on two inputs; the contrastive
 * loss is computed over their distance. Identical task to the
 * autograd tier, now with the nn::Module API and Adam.
 *
 * Usage: ./21_siamese_network_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

class Encoder : public nn::Module {
public:
    Encoder(int64_t in_dim, int64_t hidden, int64_t emb) {
        fc1 = std::make_shared<nn::Linear>(in_dim, hidden);
        fc2 = std::make_shared<nn::Linear>(hidden, emb);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }
    auto forward_impl(const Variable& x) -> Variable override {
        return fc2->forward(nn::relu(fc1->forward(x)));
    }
private:
    std::shared_ptr<nn::Linear> fc1, fc2;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Siamese Network - Neural Network API", device);
    manual_seed(42);

    int N = 128, in_dim = 4, hidden = 16, emb = 4;

    std::vector<float> X1(N * in_dim), X2(N * in_dim);
    std::vector<float> y_data(N);
    auto gen = [&](int cls) {
        std::vector<float> v(in_dim);
        for (int d = 0; d < in_dim; ++d) {
            v[d] = (cls - 1) * 0.8f + 0.1f * (((rand() % 2000) / 1000.0f) - 1.0f);
        }
        return v;
    };
    for (int i = 0; i < N; ++i) {
        int cls1 = rand() % 3;
        int cls2 = (i < N / 2) ? cls1 : ((cls1 + 1 + rand() % 2) % 3);
        auto v1 = gen(cls1), v2 = gen(cls2);
        for (int d = 0; d < in_dim; ++d) { X1[i * in_dim + d] = v1[d]; X2[i * in_dim + d] = v2[d]; }
        y_data[i] = (cls1 == cls2) ? 1.0f : 0.0f;
    }
    auto X1_t = from_data(X1.data(), {N, in_dim}, device);
    auto X2_t = from_data(X2.data(), {N, in_dim}, device);
    auto y_t  = from_data(y_data.data(), {N, 1}, device);

    auto encoder = std::make_shared<Encoder>(in_dim, hidden, emb);
    encoder->to(device);

    optim::Adam opt(encoder->parameters(), 0.01f);

    float margin = 1.0f;
    int num_epochs = 300;
    int print_every = 30;

    showcase::print_section("Architecture");
    std::cout << "Shared encoder: Linear(" << in_dim << " -> " << hidden << ") -> ReLU -> Linear("
              << hidden << " -> " << emb << ")\n";
    std::cout << "Contrastive loss with margin=" << margin << "\n";

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        encoder->train();
        opt.zero_grad();
        Variable x1(X1_t, false), x2(X2_t, false), y(y_t, false);
        auto ea = encoder->forward(x1);
        auto eb = encoder->forward(x2);
        auto diff = ea - eb;
        auto d2 = sum(diff * diff, 1, true);
        auto d  = tenzor::sqrt(d2 + 1e-8f);

        auto margin_t = Variable(ones_like(d.tensor()) * margin, false);
        auto gap_pos = nn::relu(margin_t - d);

        auto ones = Variable(ones_like(y.tensor()), false);
        auto loss = mean(y * d2 + (ones - y) * gap_pos * gap_pos);

        loss.backward();
        opt.step();

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            auto d_cpu = d.tensor().cpu(); auto y_cpu = y_t.cpu();
            float pos_d = 0, neg_d = 0; int np = 0, nn2 = 0;
            for (int i = 0; i < N; ++i) {
                float di = d_cpu.data<float>()[i];
                if (y_cpu.data<float>()[i] > 0.5f) { pos_d += di; np++; }
                else                                { neg_d += di; nn2++; }
            }
            std::cout << "Epoch " << (epoch+1)
                      << "  loss=" << loss.tensor().item<float>()
                      << "  <d|pos>=" << (np ? pos_d/np : 0)
                      << "  <d|neg>=" << (nn2 ? neg_d/nn2 : 0) << "\n";
        }
    }

    std::cout << "\nSiamese network solved using Neural Network API!\n";

    finalize();
    return 0;
}
