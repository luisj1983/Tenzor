/**
 * @file gradient_checkpointing_runner.hpp
 * @brief Public entry point for the gradient-checkpointing training loop.
 *
 * KK.27: exposes the training body of gradient_checkpointing.cpp so
 * tests/examples/test_all_autograd_examples.cpp can drive the same code
 * path the standalone exe runs in main() and assert that loss decreases
 * (guards against severed grad_fn chains and zero-gradient regressions
 * in deep-ResNet / BatchNorm2d / Adam paths).
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::gradient_checkpointing {

/// Run a short ResNet training loop on ``device`` and report the loss at
/// the first and last iteration through ``out_initial`` / ``out_final``.
/// ``num_iterations`` controls the inner-loop length so the test target
/// can run a tiny version (e.g. 4 iterations) while the standalone exe
/// keeps the original 100-sample, 3-epoch shape.
///
/// Returns 0 on success, non-zero on failure (matches showcase contract).
int run_gradient_checkpointing_training(int num_iterations,
                                         double* out_initial,
                                         double* out_final,
                                         ::tenzor::Device device,
                                         bool verbose);

}  // namespace tenzor::examples::gradient_checkpointing
