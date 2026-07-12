/**
 * @file jit_mlp_training_runner.hpp
 * @brief Public entry point for the JIT-compiled MLP training/inference example.
 *
 * R2-02: the JIT subsystem (src/jit/**) had zero coverage in the
 * "examples-as-tests" regression suite (tests/examples/), unlike every
 * eager/autograd showcase. This example trains a small MLP eagerly (so the
 * standard loss-decrease assertion applies, matching every other runner in
 * this suite), then JIT-compiles the TRAINED model's inference forward pass
 * and asserts the compiled output matches eager to tight tolerance — an
 * end-to-end trace -> compile -> execute -> verify regression, so a
 * cross-backend JIT divergence in a real (if small) model shows up here even
 * when the lower-level kernel/op tests still pass.
 */

#pragma once

#include <cstddef>

#include <tenzor/core/device.hpp>

namespace tenzor::examples::jit_mlp {

/// Train a small 2-layer MLP (Linear->ReLU->Linear) eagerly via SGD on a
/// synthetic regression task for `epochs` iterations, reporting the loss at
/// the first and last iteration through `out_initial` / `out_final`. Then
/// JIT-compiles the trained model's forward pass (default nvrtc backend,
/// portable regardless of whether TENZOR_USE_MLIR_JIT is enabled) and
/// compares it against the eager forward pass on a fresh batch, reporting
/// the max absolute difference through `out_jit_vs_eager_diff`.
///
/// JIT-R122: `out_num_cached` (optional, defaults to nullptr — existing
/// callers that don't need it are unaffected) reports
/// CompiledFunction::num_cached() after the comparison, so a caller can
/// assert the JIT path was actually taken (traced/compiled/cached) rather
/// than a silent full-eager fallback masquerading as a passing diff of 0.
/// This is the same signal tests/jit/test_jit_backend_parity.cpp's
/// LinearChain_JitVsEagerAllBackends already asserts on for every backend.
///
/// Returns 0 on success, non-zero on failure (matches showcase contract).
int run_jit_mlp_training(int epochs,
                         double* out_initial,
                         double* out_final,
                         double* out_jit_vs_eager_diff,
                         ::tenzor::Device device,
                         bool verbose,
                         std::size_t* out_num_cached = nullptr);

}  // namespace tenzor::examples::jit_mlp
