/**
 * @file autograd.cpp
 * @brief Trainable self-attention layer using autograd
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./16_self_attention_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("Self-Attention - Autograd", device);

    int rc = tenzor::examples::showcase16::run_self_attention_training(
        /*epochs=*/600, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nSelf-attention trained with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
