/**
 * @file autograd_runner.hpp
 * @brief Public entry point for the chat-AI autograd showcase.
 *
 * The training body of examples/cpp/showcase/11_chat_ai/autograd.cpp is
 * extracted here so the example_11_chat_ai_autograd_runner OBJECT target
 * (defined in examples/cpp/showcase/CMakeLists.txt) can be linked into the
 * cross-cutting test_all_autograd_examples regression test. This protects
 * the encoder-decoder GRU + Bahdanau-attention path from silent
 * grad_fn-severance regressions.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::showcase11 {

// Returns initial / final loss for a short fixed-data training run of the
// GRU encoder-decoder with Bahdanau attention. Uses the same tiny
// hardcoded chat-pair corpus the standalone exe falls back to when no
// --data file is given.
int run_chat_ai_training(int epochs,
                         double* out_initial,
                         double* out_final,
                         ::tenzor::Device device,
                         bool verbose);

}  // namespace tenzor::examples::showcase11
