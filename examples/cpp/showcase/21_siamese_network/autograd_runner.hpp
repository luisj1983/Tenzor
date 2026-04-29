/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the siamese-network autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase21 {

int run_siamese_training(int epochs,
                         double* out_initial,
                         double* out_final,
                         ::tenzor::Device device,
                         bool verbose);

}  // namespace tenzor::examples::showcase21
