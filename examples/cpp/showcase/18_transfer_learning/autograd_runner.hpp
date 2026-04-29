/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the transfer-learning autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase18 {

// Runs both stage A (pretrain backbone on task A) and stage B
// (freeze backbone, fit head B). Returns initial / final loss for
// stage A so the regression test sees a clear monotonic decrease;
// stage B is run for completeness but its loss is not exposed
// because it converges nearly immediately on this toy task.
int run_transfer_learning_training(int epochs_a,
                                   int epochs_b,
                                   double* out_initial,
                                   double* out_final,
                                   ::tenzor::Device device,
                                   bool verbose);

}  // namespace tenzor::examples::showcase18
