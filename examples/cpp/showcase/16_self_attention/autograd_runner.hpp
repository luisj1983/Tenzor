/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the self-attention autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase16 {

int run_self_attention_training(int epochs,
                                double* out_initial,
                                double* out_final,
                                ::tenzor::Device device,
                                bool verbose);

}  // namespace tenzor::examples::showcase16
