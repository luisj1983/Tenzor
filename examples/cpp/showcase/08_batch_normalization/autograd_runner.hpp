/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the batch-normalization autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase08 {

int run_batchnorm_training(int epochs,
                           double* out_initial,
                           double* out_final,
                           ::tenzor::Device device,
                           bool verbose);

}  // namespace tenzor::examples::showcase08
