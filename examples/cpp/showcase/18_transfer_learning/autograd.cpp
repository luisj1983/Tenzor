/**
 * @file autograd.cpp
 * @brief Transfer learning with autograd
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./18_transfer_learning_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Transfer Learning - Autograd", device);

    int rc = tenzor::examples::showcase18::run_transfer_learning_training(
        /*epochs_a=*/300, /*epochs_b=*/300, nullptr, nullptr, device,
        /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nTransfer learning demonstrated with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
