/**
 * @file autograd.cpp
 * @brief RNN/Sequence model using Tenzor's automatic differentiation
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./07_rnn_sequence_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("RNN Sequence - Autograd (Automatic BPTT)", device);

    int rc = tenzor::examples::showcase07::run_rnn_training(
        /*epochs=*/15000, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nRNN demonstrated with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
