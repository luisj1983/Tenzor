/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the word-embedding autograd showcase.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase17 {

int run_word_embedding_training(int epochs,
                                double* out_initial,
                                double* out_final,
                                ::tenzor::Device device,
                                bool verbose);

}  // namespace tenzor::examples::showcase17
