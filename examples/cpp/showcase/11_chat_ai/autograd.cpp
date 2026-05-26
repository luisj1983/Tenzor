/**
 * @file autograd.cpp
 * @brief Chat AI using Tenzor's autograd — GRU seq2seq with Bahdanau attention.
 *
 * The training body lives in autograd_runner.cpp so the
 * test_all_autograd_examples regression test can drive the same code
 * path. This standalone exe runs the same fixed-corpus training for a
 * larger number of epochs and reports progress; the interactive REPL of
 * earlier versions was dropped — see autograd_runner.cpp for the model
 * architecture (encoder-decoder GRU + Bahdanau attention).
 *
 * Usage: ./11_chat_ai_autograd [--backend cpu|cuda|vulkan|rocm|oneapi]
 *                              [--epochs N]
 */

#include "../common.hpp"
#include "autograd_runner.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    tenzor::Device device = showcase::get_device_from_args(argc, argv);
    tenzor::initialize();
    showcase::print_header(
        "Chat AI — Autograd (GRU + Bahdanau Attention)", device);

    int epochs = 12;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--epochs" && i + 1 < argc) {
            epochs = std::atoi(argv[++i]);
        }
    }

    double initial = 0.0, final_loss = 0.0;
    int rc = tenzor::examples::showcase11::run_chat_ai_training(
        epochs, &initial, &final_loss, device, /*verbose=*/true);

    if (rc == 0) {
        std::cout << "\nTrained an encoder-decoder GRU with Bahdanau attention "
                     "using Tenzor's autograd.\n";
        std::cout << "Initial loss: " << initial
                  << "  Final loss: " << final_loss << "\n";
    }

    tenzor::finalize();
    return rc;
}
