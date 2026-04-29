/**
 * @file autograd.cpp
 * @brief Variational Autoencoder (VAE) using Tenzor's autograd
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./13_variational_autoencoder_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Variational Autoencoder - Autograd", device);

    int rc = tenzor::examples::showcase13::run_vae_training(
        /*epochs=*/1500, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nVAE demonstrated with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
