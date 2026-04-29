/**
 * @file autograd.cpp
 * @brief Multi-task learning with autograd
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./20_multitask_learning_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Multi-Task Learning - Autograd", device);

    int rc = tenzor::examples::showcase20::run_multitask_training(
        /*epochs=*/300, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nMulti-task learning demonstrated with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
