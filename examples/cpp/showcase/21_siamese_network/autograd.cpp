/**
 * @file autograd.cpp
 * @brief Siamese network with contrastive loss, autograd version
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./21_siamese_network_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Siamese Network - Autograd", device);

    int rc = tenzor::examples::showcase21::run_siamese_training(
        /*epochs=*/300, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nSiamese network trained with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
