/**
 * @file autograd.cpp
 * @brief Siamese network with contrastive loss, autograd version
 *
 * Same task as tensor_only, but with Variables. Shared encoder
 * parameters are used by both twin branches - autograd handles
 * gradient accumulation from the two passes automatically.
 *
 * Usage: ./21_siamese_network_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Siamese Network - Autograd", device);
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
        for (int d = 0; d < in_dim; ++d) {
            X1[i * in_dim + d] = v1[d];
            X2[i * in_dim + d] = v2[d];
        }
        y_data[i] = (cls1 == cls2) ? 1.0f : 0.0f;
    }
    auto X1_t = from_data(X1.data(), {N, in_dim}, device);
    auto X2_t = from_data(X2.data(), {N, in_dim}, device);
    auto y_t  = from_data(y_data.data(), {N, 1}, device);

    auto he  = [&](int64_t fin) { return std::sqrt(2.0f / fin); };
    auto xav = [&](int64_t fin) { return std::sqrt(1.0f / fin); };

    // Shared encoder
    Variable W1(randn({in_dim, hidden}, DType::Float32, device) * he(in_dim), true);
    Variable b1(zeros({1, hidden},      DType::Float32, device), true);
    Variable W2(randn({hidden, emb},    DType::Float32, device) * xav(hidden), true);
    Variable b2(zeros({1, emb},         DType::Float32, device), true);

    auto encode = [&](const Variable& x) {
        return matmul(nn::relu(matmul(x, W1) + b1), W2) + b2;
    };

    float margin = 1.0f;
    float lr = 0.03f;
    int num_epochs = 300;
    int print_every = 30;

    showcase::print_section("Training");
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable x1(X1_t, false), x2(X2_t, false), y(y_t, false);
        auto ea = encode(x1);
        auto eb = encode(x2);
        auto diff = ea - eb;
        auto d2 = sum(diff * diff, 1, true);              // (N, 1)
        auto d = tenzor::sqrt(d2 + 1e-8f);

        auto gap = Variable(ones_like(d.tensor()) * margin, false) - d;
        auto gap_pos = nn::relu(gap);                      // max(margin - d, 0)

        auto loss_pos = y * d2;
        auto ones = Variable(ones_like(y.tensor()), false);
        auto loss_neg = (ones - y) * gap_pos * gap_pos;
        auto loss = mean(loss_pos + loss_neg);

        W1.zero_grad(); b1.zero_grad();
        W2.zero_grad(); b2.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W1 = Variable(W1.tensor() - (*W1.grad() * lr), true);
            b1 = Variable(b1.tensor() - (*b1.grad() * lr), true);
            W2 = Variable(W2.tensor() - (*W2.grad() * lr), true);
            b2 = Variable(b2.tensor() - (*b2.grad() * lr), true);
        }

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

    std::cout << "\nSiamese network trained with autograd!\n";
    std::cout << "The same encoder is called twice - autograd accumulates gradients into shared params.\n";

    finalize();
    return 0;
}
