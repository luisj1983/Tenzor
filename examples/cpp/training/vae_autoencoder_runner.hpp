/**
 * @file vae_autoencoder_runner.hpp
 * @brief Public entry point for the SimpleVAE autograd training loop.
 *
 * RR.18 (audit-11): extracted from vae_autoencoder.cpp's
 * train_simple_vae() so the regression test in
 * tests/examples/test_all_autograd_examples.cpp can drive the
 * Linear + ReLU + Sigmoid + Adam + MSELoss pipeline and assert that the
 * VAE reconstruction loss decreases over a small number of steps.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::vae_autoencoder {

int run_vae_training(int num_steps,
                     double* out_initial,
                     double* out_final,
                     ::tenzor::Device device,
                     bool verbose);

}  // namespace tenzor::examples::vae_autoencoder
