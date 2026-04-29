/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the autoencoder autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase06 {

int run_autoencoder_training(int epochs,
                             double* out_initial,
                             double* out_final,
                             ::tenzor::Device device,
                             bool verbose);

}  // namespace tenzor::examples::showcase06
