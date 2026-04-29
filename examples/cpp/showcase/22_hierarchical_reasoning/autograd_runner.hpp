/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the HRM-autograd example training loop.
 *
 * Exposed so tests/examples/test_hrm_example.cpp can drive the same
 * training logic the showcase exe runs in main(), and assert that loss
 * actually decreases. This is the regression harness for the HRM grad
 * bug fixed in commit 6cffd0a7 (negative-dim normalisation in
 * IndexSelect/Narrow backwards).
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase22 {

/**
 * Run the mini-HRM training loop end to end and report the loss measured
 * at epoch 0 and at the final epoch.
 *
 * @param epochs       Number of epochs (300 for the showcase, 50 for the
 *                     regression test — both produce a clear decrease).
 * @param out_initial  Receives the loss at epoch 0.
 * @param out_final    Receives the loss at the final epoch.
 * @param device       Backend device for tensor allocation.
 * @param verbose      If true, prints per-epoch loss/accuracy to stdout.
 * @return 0 on success, non-zero on failure.
 */
int run_hrm_training(int epochs,
                     double* out_initial,
                     double* out_final,
                     ::tenzor::Device device,
                     bool verbose);

}  // namespace tenzor::examples::showcase22
