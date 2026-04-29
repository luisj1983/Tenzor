/**
 * @file autograd.cpp
 * @brief Dropout Regularization using Tenzor's automatic differentiation
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./09_dropout_regularization_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Dropout Regularization - Autograd", device);

    int rc = tenzor::examples::showcase09::run_dropout_training(
        /*epochs=*/300, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nDropout demonstrated with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
