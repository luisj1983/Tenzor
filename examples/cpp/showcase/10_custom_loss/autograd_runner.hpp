/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the custom-loss autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase10 {

// Returns initial / final loss for the *focal-loss* training run.
int run_custom_loss_training(int epochs,
                             double* out_initial,
                             double* out_final,
                             ::tenzor::Device device,
                             bool verbose);

}  // namespace tenzor::examples::showcase10
