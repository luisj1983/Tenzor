/**
 * @file grad_accumulation.hpp
 * @brief Gradient accumulation utilities for large effective batch sizes
 *
 * Provides a helper class for accumulating gradients over multiple mini-batches
 * before performing an optimizer step, effectively increasing the batch size
 * without increasing memory usage.
 */

#pragma once

#include <cstdint>
#include "../optim/optimizer.hpp"

namespace tenzor {
namespace nn {
namespace utils {

/**
 * @brief Manages gradient accumulation over multiple mini-batches.
 *
 * Accumulates gradients for a configurable number of steps before
 * triggering the optimizer step. This simulates a larger batch size
 * without requiring additional memory.
 *
 * Features:
 * - Automatic gradient scaling by 1/accumulation_steps
 * - Automatic zero_grad after optimizer step
 * - DDP sync control via should_sync()
 *
 * @code
 * auto optimizer = SGD(model.parameters(), 0.01);
 * GradientAccumulator accum(optimizer, 4);  // accumulate 4 mini-batches
 *
 * for (auto& batch : dataloader) {
 *     auto loss = model.forward(batch) / accum.accumulation_steps();
 *     loss.backward();
 *
 *     if (accum.step()) {
 *         // Optimizer stepped and gradients zeroed
 *     }
 * }
 * // Handle any remaining accumulated gradients
 * accum.flush();
 * @endcode
 */
class GradientAccumulator {
public:
    /**
     * @brief Construct gradient accumulator.
     *
     * @param optimizer Optimizer to step
     * @param accumulation_steps Number of mini-batches to accumulate before stepping
     */
    GradientAccumulator(optim::Optimizer& optimizer, int64_t accumulation_steps);

    /**
     * @brief Advance one mini-batch step.
     *
     * Increments internal counter. When accumulation_steps mini-batches
     * have been accumulated, calls optimizer.step() and optimizer.zero_grad().
     *
     * @return true if optimizer step was performed this call
     */
    auto step() -> bool;

    /**
     * @brief Force optimizer step with any accumulated gradients.
     *
     * Useful at the end of an epoch when the number of batches may not
     * be evenly divisible by accumulation_steps.
     *
     * @return true if there were accumulated gradients to step with
     */
    auto flush() -> bool;

    /**
     * @brief Check if DDP gradient sync should be enabled.
     *
     * Returns true only on the step where the optimizer will actually update.
     * In DDP, gradient all-reduce should only happen on the accumulation boundary
     * to avoid unnecessary communication.
     *
     * @return true if this is the accumulation boundary step
     */
    auto should_sync() const -> bool;

    /**
     * @brief Get the accumulation steps count.
     * @return Number of mini-batches per optimizer step
     */
    auto accumulation_steps() const -> int64_t;

    /**
     * @brief Get current step within the accumulation window.
     * @return Current step (0 to accumulation_steps-1)
     */
    auto current_step() const -> int64_t;

    /**
     * @brief Reset the accumulator counter.
     *
     * Resets the step counter without performing an optimizer step.
     * Also zeros gradients.
     */
    auto reset() -> void;

private:
    optim::Optimizer& optimizer_;
    int64_t accumulation_steps_;
    int64_t current_step_{0};
};

} // namespace utils
} // namespace nn
} // namespace tenzor
