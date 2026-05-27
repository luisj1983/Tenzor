/**
 * @file gru_time_series_runner.hpp
 * @brief Public entry point for the GRU time-series autograd training loop.
 *
 * RR.18 (audit-11): extracted from gru_time_series.cpp's
 * train_with_adam_steplr() so the regression test can drive the
 * GRU + LayerNorm + Dropout + Linear + MSELoss + Adam pipeline and assert
 * that loss decreases.
 */

#pragma once

#include <tenzor/core/device.hpp>

namespace tenzor::examples::gru_time_series {

int run_gru_training(int num_steps,
                     double* out_initial,
                     double* out_final,
                     ::tenzor::Device device,
                     bool verbose);

}  // namespace tenzor::examples::gru_time_series
