/**
 * @file autograd.cpp
 * @brief Linear Regression using Tenzor's automatic differentiation
 *
 * The training body lives in autograd_runner.cpp so tests/examples/
 * test_all_autograd_examples.cpp can drive it as a regression check.
 *
 * Usage: ./02_linear_regression_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Linear Regression - Autograd", device);

    int rc = tenzor::examples::showcase02::run_linear_regression_training(
        /*epochs=*/1000,
        /*out_initial=*/nullptr,
        /*out_final=*/nullptr,
        device,
        /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nLinear regression solved using autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
