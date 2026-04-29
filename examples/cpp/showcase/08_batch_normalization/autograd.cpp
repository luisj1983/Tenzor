/**
 * @file autograd.cpp
 * @brief Batch Normalization using Tenzor's automatic differentiation
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./08_batch_normalization_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Batch Normalization - Autograd", device);

    int rc = tenzor::examples::showcase08::run_batchnorm_training(
        /*epochs=*/300, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nBatchNorm demonstrated with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
