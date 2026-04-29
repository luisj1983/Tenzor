/**
 * @file autograd.cpp
 * @brief Layer Normalization learned end-to-end with autograd
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./19_layer_normalization_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Layer Normalization - Autograd", device);

    int rc = tenzor::examples::showcase19::run_layernorm_training(
        /*epochs=*/300, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nLayer normalization trained with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
