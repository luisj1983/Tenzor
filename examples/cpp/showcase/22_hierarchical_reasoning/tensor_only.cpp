/**
 * @file tensor_only.cpp
 * @brief HRM hierarchical (H/L) update mechanism with raw tensors
 *
 * The Hierarchical Reasoning Model alternates two recurrent modules at
 * different timescales:
 *
 *   for n in 1..N:                  // high-level cycles (slow)
 *       for t in 1..T:              // low-level steps (fast)
 *           L <- f_L(L, H)          // L converges with H as context
 *       H <- f_H(H, L)              // H integrates the converged L
 *
 * Here f_L and f_H are stand-in linear+tanh maps so the cycling
 * structure is visible without the transformer-block machinery. We
 * print the L2 norm of each state as the loops run so you can see L
 * settling between H updates and H jumping when a high cycle closes.
 *
 * Usage: ./22_hierarchical_reasoning_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <iomanip>

using namespace tenzor;

// One linear+tanh "block": out = tanh(x @ W + ctx @ U + b).
// Models L reading H (or H reading L) as a learnable context source.
static Tensor h_l_block(const Tensor& x, const Tensor& ctx,
                        const Tensor& W, const Tensor& U, const Tensor& b) {
    auto B = x.shape()[0], T = x.shape()[1], D = x.shape()[2];
    auto x_flat = x.reshape({B * T, D});
    auto c_flat = ctx.reshape({B * T, D});
    auto y = matmul(x_flat, W) + matmul(c_flat, U) + b;
    return tanh(y).reshape({B, T, D});
}

static float l2(const Tensor& t) {
    return std::sqrt(tenzor::sum(t * t).item<float>());
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Hierarchical Reasoning Model - Tensor Only", device);
    manual_seed(42);

    int batch = 1;
    int seq   = 4;
    int d_model = 8;
    int N = 3;     // high-level cycles
    int T = 3;     // low-level steps per high cycle

    // Pretend "input encoding" — the same input drives both initial states
    auto x = randn({batch, seq, d_model}, DType::Float32, device);
    auto h_state = x;                                  // H starts at input
    auto l_state = randn({batch, seq, d_model}, DType::Float32, device) * 0.1f;

    // Two sets of weights: one for the L update, one for the H update.
    auto sc = std::sqrt(1.0f / d_model);
    auto W_L = randn({d_model, d_model}, DType::Float32, device) * sc;
    auto U_L = randn({d_model, d_model}, DType::Float32, device) * sc;  // L reads H
    auto b_L = zeros({d_model}, DType::Float32, device);

    auto W_H = randn({d_model, d_model}, DType::Float32, device) * sc;
    auto U_H = randn({d_model, d_model}, DType::Float32, device) * sc;  // H reads L
    auto b_H = zeros({d_model}, DType::Float32, device);

    showcase::print_section("Cycling H/L states");
    std::cout << "Config: N=" << N << " high cycles, T=" << T
              << " low steps/cycle, d_model=" << d_model << "\n\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  init                       ||H||=" << l2(h_state)
              << "  ||L||=" << l2(l_state) << "\n";

    for (int n = 0; n < N; ++n) {
        // L runs T times against the current (frozen-ish) H
        for (int t = 0; t < T; ++t) {
            l_state = h_l_block(l_state, h_state, W_L, U_L, b_L);
            std::cout << "  H-cycle " << n << "  L-step " << t
                      << "       ||H||=" << l2(h_state)
                      << "  ||L||=" << l2(l_state) << "\n";
        }
        // H integrates the converged L
        h_state = h_l_block(h_state, l_state, W_H, U_H, b_H);
        std::cout << "  H-cycle " << n << "  H update    "
                  << "  ||H||=" << l2(h_state)
                  << "  ||L||=" << l2(l_state) << "\n";
    }

    showcase::print_section("Final H state (first 4 dims per token)");
    auto h_cpu = h_state.cpu();
    for (int t = 0; t < seq; ++t) {
        std::cout << "  token " << t << ": ";
        for (int d = 0; d < 4; ++d) {
            std::cout << h_cpu.data<float>()[t * d_model + d] << " ";
        }
        std::cout << "...\n";
    }

    std::cout << "\nH/L cycling demonstrated: L settles between H updates,\n"
                 "then H integrates the converged L into the slower stream.\n"
                 "The full HRM wraps this with attention blocks, RoPE, and ACT.\n";

    finalize();
    return 0;
}
