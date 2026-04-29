/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the LSTM-text autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase15 {

int run_lstm_training(int epochs,
                      double* out_initial,
                      double* out_final,
                      ::tenzor::Device device,
                      bool verbose);

}  // namespace tenzor::examples::showcase15
