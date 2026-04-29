/**
 * @file autograd.cpp
 * @brief Mini hierarchical reasoning network trained with autograd
 *
 * Builds an HRM-style two-timescale recurrence from primitives (matmul,
 * tanh) and trains it on a tiny modular-arithmetic task:
 *
 *   given a length-T sequence of digits (0..9),
 *   predict (sum of digits) mod 7
 *
 * The architecture:
 *   E  = one_hot(x) @ W_emb              // embedding (B, T, D)
 *   H, L initialised from E
 *   for n in 1..N_high:
 *       for t in 1..T_low:
 *           L = tanh(L @ W_L + H @ U_L + b_L)
 *       H = tanh(H @ W_H + L @ U_H + b_H)
 *   logits = mean_t(H) @ W_out + b_out   // (B, 7)
 *
 * The autograd graph spans every cycle, so backprop trains all
 * weights jointly. The full nn::HRM (neural_network.cpp) replaces
 * each block with a transformer + ACT halting.
 *
 * The training body itself lives in autograd_runner.cpp so
 * tests/examples/test_hrm_example.cpp can drive it as a regression check
 * (catches the IndexSelect/Narrow negative-dim bug fixed in 6cffd0a7).
 *
 * Usage: ./22_hierarchical_reasoning_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Hierarchical Reasoning Model - Autograd", device);

    int rc = tenzor::examples::showcase22::run_hrm_training(
        /*epochs=*/300,
        /*out_initial=*/nullptr,
        /*out_final=*/nullptr,
        device,
        /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nMini hierarchical recurrence trained end-to-end with autograd.\n"
                     "The H/L cycling lets the network combine partial sums across\n"
                     "the sequence before projecting to a mod-7 logit.\n";
    }

    tenzor::finalize();
    return rc;
}
