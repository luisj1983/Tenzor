/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the linear-regression autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase02 {

int run_linear_regression_training(int epochs,
                                   double* out_initial,
                                   double* out_final,
                                   ::tenzor::Device device,
                                   bool verbose);

}  // namespace tenzor::examples::showcase02
