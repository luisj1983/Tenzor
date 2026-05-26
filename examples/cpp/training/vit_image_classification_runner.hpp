/**
 * @file vit_image_classification_runner.hpp
 * @brief Public entry point for the ViT-classification training loop.
 *
 * KK.27: exposes the training body of vit_image_classification.cpp so the
 * regression test in tests/examples/test_all_autograd_examples.cpp can
 * drive the same MultiheadAttention + LayerNorm + GELU + AdamW pipeline
 * and assert that loss decreases over a small number of epochs.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::vit_image_classification {

int run_vit_classification_training(int epochs,
                                     double* out_initial,
                                     double* out_final,
                                     ::tenzor::Device device,
                                     bool verbose);

}  // namespace tenzor::examples::vit_image_classification
