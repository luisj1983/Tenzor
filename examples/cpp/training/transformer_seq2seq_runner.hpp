/**
 * @file transformer_seq2seq_runner.hpp
 * @brief Public entry point for the transformer-seq2seq training loop.
 *
 * NN.24: exposes the training body of transformer_seq2seq.cpp so the
 * regression test in tests/examples/test_all_autograd_examples.cpp can
 * drive the same Embedding + MultiheadAttention + LayerNorm + GELU +
 * Adam pipeline and assert that loss decreases over a small number of
 * epochs.  Mirrors the audit-9 KK.27 pattern used by
 * vit_image_classification_runner.{cpp,hpp}.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::transformer_seq2seq {

/// Run a short attention-classifier training loop on ``device`` and
/// report the loss at the first and last epoch through ``out_initial`` /
/// ``out_final``. ``epochs`` controls the loop length so the regression
/// test can run a tiny version while the standalone exe keeps its
/// 20-epoch shape.
///
/// Returns 0 on success, non-zero on failure (matches showcase contract).
int run_transformer_seq2seq_training(int epochs,
                                      double* out_initial,
                                      double* out_final,
                                      ::tenzor::Device device,
                                      bool verbose);

}  // namespace tenzor::examples::transformer_seq2seq
