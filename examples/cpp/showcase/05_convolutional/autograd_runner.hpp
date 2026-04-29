/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the CNN autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase05 {

int run_convolutional_training(int epochs,
                               double* out_initial,
                               double* out_final,
                               ::tenzor::Device device,
                               bool verbose);

}  // namespace tenzor::examples::showcase05
