/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the multiclass-classification autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase04 {

int run_multiclass_training(int epochs,
                            double* out_initial,
                            double* out_final,
                            ::tenzor::Device device,
                            bool verbose);

}  // namespace tenzor::examples::showcase04
