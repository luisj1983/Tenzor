/**
 * @file autograd.cpp
 * @brief Binary Classification using Tenzor's automatic differentiation
 *
 * The training body lives in autograd_runner.cpp so the regression test
 * suite can drive it.
 *
 * Usage: ./03_binary_classification_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Binary Classification - Autograd", device);

    int rc = tenzor::examples::showcase03::run_binary_classification_training(
        /*epochs=*/500, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nBinary classification solved using autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
