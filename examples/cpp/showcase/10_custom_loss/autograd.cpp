/**
 * @file autograd.cpp
 * @brief Custom Loss Functions using Tenzor's automatic differentiation
 *
 * The training body lives in autograd_runner.cpp. The runner exercises
 * the focal-loss configuration; the standalone exe additionally
 * demonstrates label smoothing and Huber loss.
 *
 * Usage: ./10_custom_loss_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Custom Loss Functions - Autograd", device);

    int rc = tenzor::examples::showcase10::run_custom_loss_training(
        /*epochs=*/200, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nCustom focal-loss training demonstrated with autograd.\n";
    }

    tenzor::finalize();
    return rc;
}
