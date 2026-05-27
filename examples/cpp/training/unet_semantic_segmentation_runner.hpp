/**
 * @file unet_semantic_segmentation_runner.hpp
 * @brief Public entry point for the U-Net segmentation autograd training loop.
 *
 * RR.18 (audit-11): extracted from unet_semantic_segmentation.cpp's
 * train_unet() so the regression test can drive the
 * Conv2d + BatchNorm2d + ReLU + MaxPool2d + ConvTranspose2d + cat +
 * CrossEntropyLoss + Adam pipeline and assert the loss moves.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::unet_semantic_segmentation {

int run_unet_training(int num_steps,
                      double* out_initial,
                      double* out_final,
                      ::tenzor::Device device,
                      bool verbose);

}  // namespace tenzor::examples::unet_semantic_segmentation
