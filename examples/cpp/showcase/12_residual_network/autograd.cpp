/**
 * @file autograd.cpp
 * @brief Residual Network using Tenzor's autograd
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./12_residual_network_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Residual Network - Autograd", device);

    int rc = tenzor::examples::showcase12::run_resnet_training(
        /*epochs=*/2000, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nResidual block demonstrated with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
