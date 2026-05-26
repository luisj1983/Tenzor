/**
 * @file gpt_text_generation_runner.hpp
 * @brief Public entry point for the GPT-style language-model training loop.
 *
 * NN.24: exposes the training body of examples/cpp/training/gpt_text_generation.cpp
 * so the regression test in tests/examples/test_all_autograd_examples.cpp
 * can drive the Embedding + MultiheadAttention (causal pre-norm) +
 * LayerNorm + GELU + CrossEntropyLoss + Adam pipeline and assert that
 * loss decreases over a small number of epochs.  Mirrors the
 * audit-9 KK.27 pattern used by vit_image_classification_runner.{cpp,hpp}.
 *
 * Note: the NN.24 plan referenced examples/cpp/nlp/gpt_text_generation.cpp,
 * which is a pure-inference example (no training loop).  The training
 * counterpart lives at examples/cpp/training/gpt_text_generation.cpp
 * (executable target ``gpt_text_generation_training``) — that is the
 * file this runner extracts from.  See tests/examples/SKIP_NOTES.md.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::gpt_text_generation {

/// Run a short SimpleGPT training loop on ``device`` and report the loss
/// at the first and last epoch through ``out_initial`` / ``out_final``.
/// Returns 0 on success.
int run_gpt_text_generation_training(int epochs,
                                      double* out_initial,
                                      double* out_final,
                                      ::tenzor::Device device,
                                      bool verbose);

}  // namespace tenzor::examples::gpt_text_generation
