/**
 * @file autograd.cpp
 * @brief Generative Adversarial Network using Tenzor's autograd
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./14_gan_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("GAN - Autograd", device);

    int rc = tenzor::examples::showcase14::run_gan_training(
        /*epochs=*/3000, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nGAN demonstrated with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
