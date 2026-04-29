/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the binary-classification autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase03 {

int run_binary_classification_training(int epochs,
                                       double* out_initial,
                                       double* out_final,
                                       ::tenzor::Device device,
                                       bool verbose);

}  // namespace tenzor::examples::showcase03
