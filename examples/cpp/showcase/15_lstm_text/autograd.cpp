/**
 * @file autograd.cpp
 * @brief LSTM next-character prediction with autograd
 *
 * The training body lives in autograd_runner.cpp.
 *
 * Usage: ./15_lstm_text_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header("LSTM Text - Autograd", device);

    int rc = tenzor::examples::showcase15::run_lstm_training(
        /*epochs=*/400, nullptr, nullptr, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nLSTM demonstrated with autograd!\n";
    }

    tenzor::finalize();
    return rc;
}
