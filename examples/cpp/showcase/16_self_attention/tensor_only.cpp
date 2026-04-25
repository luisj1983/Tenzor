/**
 * @file tensor_only.cpp
 * @brief Scaled dot-product self-attention with raw tensors
 *
 * Implements attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V from scratch,
 * purely with tensor ops. No training - the point is to show the
 * mechanism and its output. Autograd/NN tiers show learning.
 *
 * Usage: ./16_self_attention_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;

// Row-wise softmax on last dim, with subtract-max for stability.
static Tensor softmax_t(const Tensor& x, int64_t dim) {
    auto m = tenzor::max(x, dim, true);
    auto e = tenzor::exp(x - m);
    auto s = tenzor::sum(e, dim, true);
    return e / s;
}

// Scaled dot-product attention on batched (B, T, D) tensors.
// Returns (output, attention weights) where weights have shape (B, T, T).
static std::pair<Tensor, Tensor>
scaled_dp_attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
    auto d_k = Q.shape().back();
    auto Kt = K.transpose(1, 2);                               // (B, D, T)
    auto scores = bmm(Q, Kt) * (1.0f / std::sqrt(static_cast<float>(d_k)));
    auto attn = softmax_t(scores, 2);                          // row-wise softmax over keys
    auto out = bmm(attn, V);                                   // (B, T, D)
    return {out, attn};
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Self-Attention - Tensor Only (Mechanism Demo)", device);
    manual_seed(42);

    int batch = 1;
    int seq   = 5;
    int d_model = 8;

    // Random "token embeddings"
    auto X = randn({batch, seq, d_model}, DType::Float32, device);

    // Linear projections: W_q, W_k, W_v of shape (d_model, d_model)
    auto W_q = randn({d_model, d_model}, DType::Float32, device) * std::sqrt(1.0f / d_model);
    auto W_k = randn({d_model, d_model}, DType::Float32, device) * std::sqrt(1.0f / d_model);
    auto W_v = randn({d_model, d_model}, DType::Float32, device) * std::sqrt(1.0f / d_model);

    // Flatten batch+seq for 2D matmul against W_*
    auto X_flat = X.reshape({batch * seq, d_model});
    auto Q = matmul(X_flat, W_q).reshape({batch, seq, d_model});
    auto K = matmul(X_flat, W_k).reshape({batch, seq, d_model});
    auto V = matmul(X_flat, W_v).reshape({batch, seq, d_model});

    showcase::print_section("Shapes");
    std::cout << "X: (" << batch << ", " << seq << ", " << d_model << ")\n";
    std::cout << "Q, K, V: same as X after linear projection\n";

    auto [output, attn] = scaled_dp_attention(Q, K, V);

    showcase::print_section("Attention weights (row-wise softmax)");
    auto a_cpu = attn.cpu();
    for (int t = 0; t < seq; ++t) {
        std::cout << "token " << t << ": [";
        for (int s = 0; s < seq; ++s) {
            std::cout << a_cpu.data<float>()[t * seq + s];
            if (s < seq - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    showcase::print_section("Output (first 4 dims per token)");
    auto o_cpu = output.cpu();
    for (int t = 0; t < seq; ++t) {
        std::cout << "token " << t << ": ";
        for (int d = 0; d < 4; ++d) {
            std::cout << o_cpu.data<float>()[t * d_model + d] << " ";
        }
        std::cout << "...\n";
    }

    std::cout << "\nSelf-attention demonstrated with raw tensors!\n";
    std::cout << "attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V\n";

    finalize();
    return 0;
}
