/**
 * @file tensor_only.cpp
 * @brief Siamese/contrastive learning with raw tensors
 *
 * Twin encoders (shared weights) map two inputs to an embedding.
 * Contrastive loss pulls embeddings of the "same class" together,
 * pushes embeddings of different classes apart beyond a margin:
 *
 *   L = y * d^2 + (1 - y) * max(0, margin - d)^2
 *
 * where d = ||emb(x1) - emb(x2)||_2, y = 1 if same class, 0 otherwise.
 *
 * Usage: ./21_siamese_network_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace tenzor;

static Tensor relu_t(const Tensor& x) { return maximum(x, zeros_like(x)); }
static Tensor relu_d(const Tensor& z) { return (z > zeros_like(z)).to(DType::Float32); }

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Siamese Network - Tensor Only (Contrastive Loss)", device);
    manual_seed(42);

    int N = 128, in_dim = 4, hidden = 16, emb = 4;

    // Generate 3 classes; positive pairs within a class, negative pairs across.
    std::vector<float> X1(N * in_dim), X2(N * in_dim);
    std::vector<float> y_data(N);  // 1 = same, 0 = diff

    auto gen_class_vec = [&](int cls) {
        std::vector<float> v(in_dim);
        for (int d = 0; d < in_dim; ++d) {
            float base = (cls - 1) * 0.8f;                              // class center per dim
            v[d] = base + 0.1f * (((rand() % 2000) / 1000.0f) - 1.0f);   // small noise
        }
        return v;
    };
    for (int i = 0; i < N; ++i) {
        int cls1 = rand() % 3;
        int cls2 = (i < N / 2) ? cls1 : ((cls1 + 1 + rand() % 2) % 3);
        auto v1 = gen_class_vec(cls1), v2 = gen_class_vec(cls2);
        for (int d = 0; d < in_dim; ++d) {
            X1[i * in_dim + d] = v1[d];
            X2[i * in_dim + d] = v2[d];
        }
        y_data[i] = (cls1 == cls2) ? 1.0f : 0.0f;
    }
    auto X1_t = from_data(X1.data(), {N, in_dim}, device);
    auto X2_t = from_data(X2.data(), {N, in_dim}, device);
    auto y_t  = from_data(y_data.data(), {N, 1}, device);

    // Shared encoder
    auto W1 = randn({in_dim, hidden}, DType::Float32, device) * std::sqrt(2.0f / in_dim);
    auto b1 = zeros({1, hidden},      DType::Float32, device);
    auto W2 = randn({hidden, emb},    DType::Float32, device) * std::sqrt(1.0f / hidden);
    auto b2 = zeros({1, emb},         DType::Float32, device);

    auto encode = [&](const Tensor& x) {
        auto z1 = matmul(x, W1) + b1; auto a1 = relu_t(z1);
        auto z2 = matmul(a1, W2) + b2;  // linear embedding
        return std::make_tuple(z1, a1, z2);
    };

    float margin = 1.0f;
    float lr = 0.03f;
    int num_epochs = 300;
    int print_every = 30;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto [z1a, a1a, ea] = encode(X1_t);
        auto [z1b, a1b, eb] = encode(X2_t);

        // Distance d and d^2
        auto diff = ea - eb;
        auto d2 = tenzor::sum(diff * diff, 1, true);           // (N, 1)
        auto d  = tenzor::sqrt(d2 + 1e-8f);                    // (N, 1)

        // Contrastive loss
        auto margin_gap = maximum(margin - d, zeros_like(d));  // (N, 1)
        auto loss_pos = y_t * d2;
        auto loss_neg = (ones_like(y_t) - y_t) * margin_gap * margin_gap;
        auto loss_per = loss_pos + loss_neg;
        float loss_val = tenzor::mean(loss_per).item<float>();

        // Backward (manual)
        float n = static_cast<float>(N);

        // dL/dd:
        //   for positives: 2 * y * d
        //   for negatives: 2 * (1 - y) * -margin_gap   (active only when margin > d)
        // dL/d(diff) = dL/dd * (diff / d) for positives, similar chain for negatives
        // To avoid branching manually, compute via d2 gradient:
        //   d(d2) = 2 diff, d(margin_gap^2) where margin_gap = margin - d, gap > 0
        auto dL_dd2_pos = y_t;                                               // (N, 1) -> mul by 1/n below
        // For negatives: L_neg = (margin - d)^2 when gap>0; else 0.
        // d/dd L_neg = -2 * gap when gap > 0 and zero otherwise.
        // d/dd2 = d/dd * 1/(2d). So d/dd2 L_neg = -gap / d when gap > 0.
        auto active_neg = (margin_gap > zeros_like(margin_gap)).to(DType::Float32);
        auto dL_dd2_neg = (ones_like(y_t) - y_t) * (margin_gap / (d + 1e-8f)) * active_neg * -1.0f;

        auto dL_dd2 = (dL_dd2_pos + dL_dd2_neg) * (1.0f / n);
        auto dL_ddiff = diff * 2.0f * dL_dd2;                                // (N, emb)
        auto dL_dea =  dL_ddiff;
        auto dL_deb = dL_ddiff * -1.0f;

        // Backprop for encoder (twin branches accumulate into shared weights)
        auto back_enc = [&](const Tensor& x, const Tensor& z1, const Tensor& a1,
                            const Tensor& dL_de,
                            Tensor& dW1, Tensor& db1, Tensor& dW2, Tensor& db2) {
            auto dW2_local = matmul(a1.transpose(0, 1), dL_de);
            auto db2_local = tenzor::sum(dL_de, 0, true);
            auto da1 = matmul(dL_de, W2.transpose(0, 1));
            auto dz1 = da1 * relu_d(z1);
            auto dW1_local = matmul(x.transpose(0, 1), dz1);
            auto db1_local = tenzor::sum(dz1, 0, true);
            dW1 = dW1 + dW1_local; db1 = db1 + db1_local;
            dW2 = dW2 + dW2_local; db2 = db2 + db2_local;
        };

        auto dW1 = zeros_like(W1), db1 = zeros_like(b1);
        auto dW2 = zeros_like(W2), db2 = zeros_like(b2);
        back_enc(X1_t, z1a, a1a, dL_dea, dW1, db1, dW2, db2);
        back_enc(X2_t, z1b, a1b, dL_deb, dW1, db1, dW2, db2);

        W1 = W1 - dW1 * lr;  b1 = b1 - db1 * lr;
        W2 = W2 - dW2 * lr;  b2 = b2 - db2 * lr;

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            auto d_cpu = d.cpu(); auto y_cpu = y_t.cpu();
            float pos_d = 0, neg_d = 0; int np = 0, nn2 = 0;
            for (int i = 0; i < N; ++i) {
                float di = d_cpu.data<float>()[i];
                if (y_cpu.data<float>()[i] > 0.5f) { pos_d += di; np++; }
                else                                { neg_d += di; nn2++; }
            }
            std::cout << "Epoch " << (epoch+1)
                      << "  loss=" << loss_val
                      << "  <d|pos>=" << (np ? pos_d/np : 0)
                      << "  <d|neg>=" << (nn2 ? neg_d/nn2 : 0) << "\n";
        }
    }

    std::cout << "\nSiamese network trained with raw tensors!\n";
    std::cout << "Contrastive loss: same-class pair distances collapse, diff-class > margin.\n";

    finalize();
    return 0;
}
