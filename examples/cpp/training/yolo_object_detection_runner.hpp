/**
 * @file yolo_object_detection_runner.hpp
 * @brief Public entry point for the YOLO-style detection training loop.
 *
 * NN.24: exposes the training body of yolo_object_detection.cpp so the
 * regression test in tests/examples/test_all_autograd_examples.cpp can
 * drive the Conv2d + GroupNorm + Mish + LeakyReLU + ResidualBlock +
 * Adam pipeline and assert that training moved the weights.  Mirrors
 * the audit-9 KK.27 pattern from vit_image_classification_runner.
 *
 * Loss-decrease is asserted via std::abs(initial - final) > threshold
 * because the simplified MSE-against-zeros surrogate isn't guaranteed
 * to be monotonically decreasing on a few random batches.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::yolo_object_detection {

/// Run a short YOLO-style backbone training loop on ``device`` and
/// report the loss at the first and last iteration through
/// ``out_initial`` / ``out_final``.  Returns 0 on success.
int run_yolo_object_detection_training(int num_iterations,
                                        double* out_initial,
                                        double* out_final,
                                        ::tenzor::Device device,
                                        bool verbose);

}  // namespace tenzor::examples::yolo_object_detection
