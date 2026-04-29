/**
 * @file autograd.cpp
 * @brief Autoencoder using Tenzor's automatic differentiation
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./06_autoencoder_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Autoencoder - Autograd", device);

    int rc = tenzor::examples::showcase06::run_autoencoder_training(
        /*epochs=*/1000, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nAutoencoder demonstrated with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
