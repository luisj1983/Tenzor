/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the XOR autograd showcase training loop.
 *
 * Exposed so tests/examples/test_all_autograd_examples.cpp can drive the
 * same training logic the showcase exe runs in main(), and assert that
 * loss actually decreases.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase01 {

int run_xor_training(int epochs,
                     double* out_initial,
                     double* out_final,
                     ::tenzor::Device device,
                     bool verbose);

}  // namespace tenzor::examples::showcase01
