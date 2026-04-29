/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the RNN-sequence autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase07 {

int run_rnn_training(int epochs,
                     double* out_initial,
                     double* out_final,
                     ::tenzor::Device device,
                     bool verbose);

}  // namespace tenzor::examples::showcase07
