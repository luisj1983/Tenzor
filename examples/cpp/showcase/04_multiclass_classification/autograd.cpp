/**
 * @file autograd.cpp
 * @brief Multi-class Classification using Tenzor's automatic differentiation
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./04_multiclass_classification_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Multi-class Classification - Autograd", device);

    int rc = tenzor::examples::showcase04::run_multiclass_training(
        /*epochs=*/500, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nMulti-class classification solved using autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
