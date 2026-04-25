/**
 * @file autograd.cpp
 * @brief Transfer learning with autograd
 *
 * Same two-stage setup as tensor_only, but with Variables. Freezing
 * the backbone is just a matter of not updating those Variables in
 * the optimizer step - autograd will still compute their gradients,
 * but they are ignored.
 *
 * Usage: ./18_transfer_learning_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Transfer Learning - Autograd", device);
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
    auto X_t  = from_data(X_data.data(), {N, in_dim}, device);
    auto yA_t = from_data(yA_data.data(),{N, 1},      device);
    auto yB_t = from_data(yB_data.data(),{N, 1},      device);

    auto he  = [&](int64_t fin) { return std::sqrt(2.0f / fin); };
    auto xav = [&](int64_t fin) { return std::sqrt(1.0f / fin); };

    // Backbone
    Variable Wb1(randn({in_dim, hidden}, DType::Float32, device) * he(in_dim), true);
    Variable bb1(zeros({1, hidden},      DType::Float32, device), true);
    Variable Wb2(randn({hidden, hidden}, DType::Float32, device) * he(hidden), true);
    Variable bb2(zeros({1, hidden},      DType::Float32, device), true);

    // Head A
    Variable WA(randn({hidden, 1}, DType::Float32, device) * xav(hidden), true);
    Variable bA(zeros({1, 1}, DType::Float32, device), true);

    auto backbone = [&](const Variable& x) {
        auto a1 = nn::relu(matmul(x, Wb1) + bb1);
        return nn::relu(matmul(a1, Wb2) + bb2);
    };

    auto bce_with_logits = [&](const Variable& logit, const Variable& tgt) {
        // -y*log(sig(x)) - (1-y)*log(1-sig(x)) via softplus for stability:
        // = -y*(x - softplus(x)) - (1-y)*(-softplus(x)) = softplus(x) - y*x
        auto softplus = tenzor::log(tenzor::exp(logit) + 1.0f);
        return mean(softplus - tgt * logit);
    };

    float lr = 0.05f;

    showcase::print_section("Stage 1: pretrain backbone on task A");
    for (int epoch = 0; epoch < 300; ++epoch) {
        Variable x(X_t, false), y(yA_t, false);
        auto feat = backbone(x);
        auto logit = matmul(feat, WA) + bA;
        auto loss = bce_with_logits(logit, y);

        Wb1.zero_grad(); bb1.zero_grad();
        Wb2.zero_grad(); bb2.zero_grad();
        WA.zero_grad();  bA.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            Wb1 = Variable(Wb1.tensor() - (*Wb1.grad() * lr), true);
            bb1 = Variable(bb1.tensor() - (*bb1.grad() * lr), true);
            Wb2 = Variable(Wb2.tensor() - (*Wb2.grad() * lr), true);
            bb2 = Variable(bb2.tensor() - (*bb2.grad() * lr), true);
            WA  = Variable(WA.tensor()  - (*WA.grad()  * lr), true);
            bA  = Variable(bA.tensor()  - (*bA.grad()  * lr), true);
        }

        if ((epoch + 1) % 50 == 0 || epoch == 0) {
            auto prob = (ones_like(logit.tensor())
                       / (tenzor::exp(logit.tensor() * -1.0f) + 1.0f));
            auto pc = prob.cpu(); auto yc = yA_t.cpu();
            int c = 0;
            for (int i = 0; i < N; ++i)
                if ((pc.data<float>()[i] > 0.5f) == (yc.data<float>()[i] > 0.5f)) c++;
            std::cout << "[TaskA] epoch " << (epoch+1) << "  loss=" << loss.tensor().item<float>()
                      << "  acc=" << (100.0f * c / N) << "%\n";
        }
    }

    showcase::print_section("Stage 2: freeze backbone, fine-tune head B");
    Variable WB(randn({hidden, 1}, DType::Float32, device) * xav(hidden), true);
    Variable bB(zeros({1, 1}, DType::Float32, device), true);

    for (int epoch = 0; epoch < 300; ++epoch) {
        Variable x(X_t, false), y(yB_t, false);
        auto feat = backbone(x);
        auto logit = matmul(feat, WB) + bB;
        auto loss = bce_with_logits(logit, y);

        WB.zero_grad(); bB.zero_grad();
        // Intentionally skip zero_grad/update for backbone - "frozen".
        loss.backward();

        {
            NoGradGuard ng;
            WB = Variable(WB.tensor() - (*WB.grad() * lr), true);
            bB = Variable(bB.tensor() - (*bB.grad() * lr), true);
        }

        if ((epoch + 1) % 50 == 0 || epoch == 0) {
            auto prob = (ones_like(logit.tensor())
                       / (tenzor::exp(logit.tensor() * -1.0f) + 1.0f));
            auto pc = prob.cpu(); auto yc = yB_t.cpu();
            int c = 0;
            for (int i = 0; i < N; ++i)
                if ((pc.data<float>()[i] > 0.5f) == (yc.data<float>()[i] > 0.5f)) c++;
            std::cout << "[TaskB] epoch " << (epoch+1) << "  loss=" << loss.tensor().item<float>()
                      << "  acc=" << (100.0f * c / N) << "%\n";
        }
    }

    std::cout << "\nTransfer learning demonstrated with autograd!\n";
    std::cout << "Backbone is frozen in stage 2 - only head B is updated.\n";

    finalize();
    return 0;
}
