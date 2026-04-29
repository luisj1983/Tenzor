/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the GAN autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase14 {

// Returns initial / final discriminator loss. We use D-loss as the
// "training is moving" signal because GAN dynamics make G-loss
// non-monotonic; the test only requires that *something* is moving
// (i.e., backward isn't returning zero).
int run_gan_training(int epochs,
                     double* out_initial,
                     double* out_final,
                     ::tenzor::Device device,
                     bool verbose);

}  // namespace tenzor::examples::showcase14
