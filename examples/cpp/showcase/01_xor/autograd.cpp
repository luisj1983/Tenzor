/**
 * @file autograd.cpp
 * @brief XOR problem solved using Tenzor's automatic differentiation
 *
 * The training body lives in autograd_runner.cpp so tests/examples/
 * test_all_autograd_examples.cpp can drive it as a regression check.
 *
 * Usage: ./01_xor_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("XOR - Autograd (Automatic Differentiation)", device);

    int rc = tenzor::examples::showcase01::run_xor_training(
        /*epochs=*/10000,
        /*out_initial=*/nullptr,
        /*out_final=*/nullptr,
        device,
        /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nXOR problem solved using autograd!\n"
                     "Automatic differentiation computed all gradients for us.\n";
    }

    tenzor::finalize();
    return rc;
}
