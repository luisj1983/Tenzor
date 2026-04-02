/**
 * @file pipeline_parallel.cpp
 * @brief Implementation of pipeline parallelism schedules
 *
 * Implements GPipe and 1F1B (one-forward-one-backward) pipeline schedules
 * for distributed model-parallel training.
 */

#include "tenzor/distributed/pipeline_parallel.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/autograd/ops.hpp"
#include <stdexcept>
#include <deque>

namespace tenzor {
namespace distributed {

// ============================================================================
// PipelineStage
// ============================================================================

PipelineStage::PipelineStage(std::shared_ptr<nn::Module> module,
                             int stage_id, int num_stages)
    : module_(std::move(module))
    , stage_id_(stage_id)
    , num_stages_(num_stages)
{
    if (!module_) {
        throw std::invalid_argument("PipelineStage: module must not be null");
    }
    if (stage_id < 0 || stage_id >= num_stages) {
        throw std::invalid_argument(
            "PipelineStage: stage_id must be in [0, num_stages), got " +
            std::to_string(stage_id) + " with num_stages=" +
            std::to_string(num_stages)
        );
    }
    if (num_stages < 1) {
        throw std::invalid_argument(
            "PipelineStage: num_stages must be >= 1, got " +
            std::to_string(num_stages)
        );
    }
}

auto PipelineStage::forward(const Variable& input) -> Variable {
    return module_->forward(input);
}

// ============================================================================
// Helper: split / concatenate micro-batches along dim 0
// ============================================================================

namespace {

/**
 * @brief Split a Variable into num_microbatches chunks along dimension 0.
 *
 * If the batch size is not evenly divisible, the last chunk is smaller.
 */
auto split_microbatches(const Variable& input, int num_microbatches)
    -> std::vector<Variable>
{
    const auto& tensor = input.tensor();
    int64_t batch_size = tensor.shape()[0];

    if (num_microbatches <= 0) {
        throw std::invalid_argument(
            "split_microbatches: num_microbatches must be > 0"
        );
    }
    if (batch_size < num_microbatches) {
        throw std::invalid_argument(
            "split_microbatches: batch size (" + std::to_string(batch_size) +
            ") must be >= num_microbatches (" + std::to_string(num_microbatches) + ")"
        );
    }

    int64_t chunk_size = (batch_size + num_microbatches - 1) / num_microbatches;
    auto chunks = split(tensor, chunk_size, /*dim=*/0);

    std::vector<Variable> result;
    result.reserve(chunks.size());
    for (auto& c : chunks) {
        result.emplace_back(std::move(c), input.requires_grad());
    }
    return result;
}

/**
 * @brief Concatenate micro-batch outputs along dimension 0.
 */
auto concat_microbatches(const std::vector<Variable>& microbatches) -> Variable {
    if (microbatches.empty()) {
        throw std::invalid_argument("concat_microbatches: empty input");
    }
    if (microbatches.size() == 1) {
        return microbatches[0];
    }
    return autograd::cat(microbatches, /*dim=*/0);
}

/**
 * @brief Send a Variable's tensor to dst_rank via point-to-point.
 */
auto send_activation(const Variable& var, int dst_rank, ProcessGroup& pg) -> void {
    Tensor t = var.tensor();  // copy handle (shallow)
    pg.send(t, dst_rank);
}

/**
 * @brief Receive a tensor from src_rank and wrap as Variable.
 *
 * The caller must provide a template tensor with the expected shape/dtype/device
 * so we can pre-allocate the receive buffer.
 */
auto recv_activation(const Tensor& shape_template, int src_rank,
                     ProcessGroup& pg, bool requires_grad) -> Variable
{
    auto shape = shape_template.shape();
    Tensor buf = empty({shape.begin(), shape.end()}, shape_template.dtype(),
                       shape_template.device());
    pg.recv(buf, src_rank);
    return Variable(std::move(buf), requires_grad);
}

} // anonymous namespace

// ============================================================================
// GPipeSchedule
// ============================================================================

auto GPipeSchedule::execute(PipelineStage& stage, const Variable& input,
                            int num_microbatches, ProcessGroup& pg) -> Variable
{
    const int prev_rank = stage.stage_id() - 1;
    const int next_rank = stage.stage_id() + 1;

    // ---- Phase 1: Split input into micro-batches (first stage only) ----
    std::vector<Variable> micro_inputs;
    if (stage.is_first()) {
        micro_inputs = split_microbatches(input, num_microbatches);
    }

    // ---- Phase 2: Forward all micro-batches ----
    std::vector<Variable> micro_outputs;
    micro_outputs.reserve(num_microbatches);

    // Stash inputs for backward (needed to compute gradients)
    std::vector<Variable> stashed_inputs;
    stashed_inputs.reserve(num_microbatches);

    for (int mb = 0; mb < num_microbatches; ++mb) {
        Variable mb_input;

        if (stage.is_first()) {
            mb_input = micro_inputs[mb];
        } else {
            // Receive activation from previous stage.
            // For the shape template we need the first micro-batch's expected
            // shape. On non-first stages we receive from upstream, so we must
            // do the first recv to discover the shape, then reuse it.
            if (mb == 0) {
                // We don't know the shape a priori on intermediate stages.
                // Use a blocking recv that allocates based on the incoming data.
                // The ProcessGroup::recv() is expected to handle this.
                Tensor buf = input.tensor();  // use input as shape hint
                pg.recv(buf, prev_rank);
                mb_input = Variable(std::move(buf), /*requires_grad=*/true);
            } else {
                mb_input = recv_activation(
                    stashed_inputs[0].tensor(), prev_rank, pg,
                    /*requires_grad=*/true
                );
            }
        }

        stashed_inputs.push_back(mb_input);

        // Forward through local stage
        auto mb_output = stage.forward(mb_input);
        micro_outputs.push_back(mb_output);

        // Send activation to next stage (if not last)
        if (!stage.is_last()) {
            send_activation(mb_output, next_rank, pg);
        }
    }

    // ---- Phase 3: Backward all micro-batches (reverse order) ----
    for (int mb = num_microbatches - 1; mb >= 0; --mb) {
        if (stage.is_last()) {
            // Last stage: backward starts from the output.
            // The loss computation and initial backward() call is the
            // caller's responsibility. Here we propagate gradients backward
            // through the stage. The caller must have called backward() on
            // the loss which populates grad on micro_outputs[mb].
            // For pipeline purposes, we call backward on each micro-batch output
            // with a unit gradient to propagate through the stage graph.
            micro_outputs[mb].backward();
        } else {
            // Receive gradient from next stage
            auto out_shape = micro_outputs[mb].tensor().shape();
            Tensor grad_buf = empty(
                {out_shape.begin(), out_shape.end()},
                micro_outputs[mb].tensor().dtype(),
                micro_outputs[mb].tensor().device()
            );
            pg.recv(grad_buf, next_rank);

            // Backward with received gradient
            micro_outputs[mb].backward(std::move(grad_buf));
        }

        // Send gradient to previous stage (if not first)
        if (!stage.is_first()) {
            // The gradient w.r.t. the input of this stage is stored in
            // stashed_inputs[mb].grad()
            const auto& input_grad_opt = stashed_inputs[mb].grad();
            if (!input_grad_opt.has_value()) {
                throw std::runtime_error(
                    "GPipeSchedule: no gradient computed for micro-batch " +
                    std::to_string(mb) + " input"
                );
            }
            Tensor input_grad = input_grad_opt.value();
            pg.send(input_grad, prev_rank);
        }
    }

    // ---- Return concatenated outputs (meaningful only on last stage) ----
    if (stage.is_last()) {
        return concat_microbatches(micro_outputs);
    }

    // Non-last stages return the concatenated outputs for API consistency,
    // though callers typically only use the last stage's output.
    return concat_microbatches(micro_outputs);
}

// ============================================================================
// OneFOneBSchedule
// ============================================================================

auto OneFOneBSchedule::execute(PipelineStage& stage, const Variable& input,
                               int num_microbatches, ProcessGroup& pg) -> Variable
{
    const int prev_rank = stage.stage_id() - 1;
    const int next_rank = stage.stage_id() + 1;

    // Number of warmup forward passes before we start interleaving backward.
    // Earlier stages do more warmup, later stages do less.
    const int num_warmup = stage.num_stages() - 1 - stage.stage_id();
    const int num_steady = num_microbatches - num_warmup;

    // Split input on first stage
    std::vector<Variable> micro_inputs;
    if (stage.is_first()) {
        micro_inputs = split_microbatches(input, num_microbatches);
    }

    // Activation stash: saves intermediate inputs/outputs for backward.
    // Index by micro-batch id.
    std::vector<Variable> stashed_inputs;
    stashed_inputs.resize(num_microbatches);

    std::vector<Variable> micro_outputs;
    micro_outputs.resize(num_microbatches);

    // Track which micro-batch to run backward on next
    std::deque<int> backward_queue;

    int fwd_mb = 0;  // next micro-batch to forward

    // Helper: run one forward micro-batch
    auto do_forward = [&](int mb) {
        Variable mb_input;

        if (stage.is_first()) {
            mb_input = micro_inputs[mb];
        } else {
            if (mb == 0) {
                // First recv: use input tensor as shape hint
                Tensor buf = input.tensor();
                pg.recv(buf, prev_rank);
                mb_input = Variable(std::move(buf), /*requires_grad=*/true);
            } else {
                mb_input = recv_activation(
                    stashed_inputs[0].tensor(), prev_rank, pg,
                    /*requires_grad=*/true
                );
            }
        }

        stashed_inputs[mb] = mb_input;
        micro_outputs[mb] = stage.forward(mb_input);

        if (!stage.is_last()) {
            send_activation(micro_outputs[mb], next_rank, pg);
        }
    };

    // Helper: run one backward micro-batch
    auto do_backward = [&](int mb) {
        if (stage.is_last()) {
            micro_outputs[mb].backward();
        } else {
            auto out_shape = micro_outputs[mb].tensor().shape();
            Tensor grad_buf = empty(
                {out_shape.begin(), out_shape.end()},
                micro_outputs[mb].tensor().dtype(),
                micro_outputs[mb].tensor().device()
            );
            pg.recv(grad_buf, next_rank);
            micro_outputs[mb].backward(std::move(grad_buf));
        }

        if (!stage.is_first()) {
            const auto& input_grad_opt = stashed_inputs[mb].grad();
            if (!input_grad_opt.has_value()) {
                throw std::runtime_error(
                    "OneFOneBSchedule: no gradient computed for micro-batch " +
                    std::to_string(mb) + " input"
                );
            }
            Tensor input_grad = input_grad_opt.value();
            pg.send(input_grad, prev_rank);
        }
    };

    // ---- Phase 1: Warmup (forward only) ----
    for (int i = 0; i < num_warmup && fwd_mb < num_microbatches; ++i) {
        do_forward(fwd_mb);
        backward_queue.push_back(fwd_mb);
        ++fwd_mb;
    }

    // ---- Phase 2: Steady state (1 forward + 1 backward) ----
    for (int i = 0; i < num_steady && fwd_mb < num_microbatches; ++i) {
        // Forward one micro-batch
        do_forward(fwd_mb);
        backward_queue.push_back(fwd_mb);
        ++fwd_mb;

        // Backward one micro-batch (oldest in queue)
        int bw_mb = backward_queue.front();
        backward_queue.pop_front();
        do_backward(bw_mb);
    }

    // ---- Phase 3: Cooldown (drain remaining backward passes) ----
    while (!backward_queue.empty()) {
        int bw_mb = backward_queue.front();
        backward_queue.pop_front();
        do_backward(bw_mb);
    }

    // ---- Return concatenated outputs ----
    std::vector<Variable> valid_outputs;
    valid_outputs.reserve(num_microbatches);
    for (int mb = 0; mb < num_microbatches; ++mb) {
        valid_outputs.push_back(micro_outputs[mb]);
    }
    return concat_microbatches(valid_outputs);
}

} // namespace distributed
} // namespace tenzor
