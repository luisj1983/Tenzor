/**
 * @file autograd.cpp
 * @brief Convolutional Neural Network using Tenzor's autograd
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./05_convolutional_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Convolutional NN - Autograd", device);

    int rc = tenzor::examples::showcase05::run_convolutional_training(
        /*epochs=*/50, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nCNN demonstrated with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
