/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the multitask-learning autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase20 {

int run_multitask_training(int epochs,
                           double* out_initial,
                           double* out_final,
                           ::tenzor::Device device,
                           bool verbose);

}  // namespace tenzor::examples::showcase20
