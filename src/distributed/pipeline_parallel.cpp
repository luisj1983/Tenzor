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
 * @brief Split a Variable into exactly num_microbatches chunks along dim 0.
 *
 * Uses a balanced partition: when the batch size is not evenly divisible the
 * first (batch_size % num_microbatches) chunks carry one extra row. Always
 * returns exactly num_microbatches non-empty chunks (requires
 * batch_size >= num_microbatches).
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

    // Build a balanced partition that yields EXACTLY num_microbatches
    // non-empty chunks. A fixed-size split (split / tensor_split with a
    // ceil chunk_size) produces ceil(batch_size / chunk_size) chunks, which
    // is fewer than num_microbatches whenever the batch is not a near
    // multiple (e.g. batch=5, mb=4 -> chunk_size=2 -> only 3 chunks). The
    // GPipe / 1F1B loops index micro_inputs[mb] for mb in [0,
    // num_microbatches), so a short vector is an out-of-bounds read.
    //
    // base = floor(batch / mb) >= 1 (guaranteed by batch_size >=
    // num_microbatches above); the first `remainder` chunks get one extra
    // row. Sizes sum to batch_size and count is exactly num_microbatches.
    int64_t base = batch_size / num_microbatches;
    int64_t remainder = batch_size % num_microbatches;
    std::vector<int64_t> split_sizes;
    split_sizes.reserve(static_cast<size_t>(num_microbatches));
    for (int64_t i = 0; i < num_microbatches; ++i) {
        split_sizes.push_back(base + (i < remainder ? 1 : 0));
    }

    auto chunks = split_with_sizes(tensor, split_sizes, /*dim=*/0);

    if (static_cast<int64_t>(chunks.size()) != num_microbatches) {
        throw std::runtime_error(
            "split_microbatches: produced " + std::to_string(chunks.size()) +
            " chunks, expected " + std::to_string(num_microbatches));
    }

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
    return tenzor::cat(microbatches, /*dim=*/0);
}

// Fixed-size activation metadata header exchanged before each activation so the
// receiver can allocate a correctly-shaped/typed buffer without assuming the
// upstream activation matches the schedule's input argument. The header is an
// Int64 tensor of kMetaHeaderLen elements: [ndim, dtype, dim0, dim1, ...].
// A fixed length keeps recv() able to pre-size its buffer (recv needs to know
// the byte count in advance).
constexpr int64_t kMetaHeaderLen = 16;          // 2 slots + up to 14 dims
constexpr int64_t kMetaMaxDims   = kMetaHeaderLen - 2;

auto encode_activation_meta(const Tensor& t) -> Tensor {
    const auto& shape = t.shape();
    if (static_cast<int64_t>(shape.size()) > kMetaMaxDims) {
        throw std::runtime_error(
            "pipeline_parallel: activation rank " + std::to_string(shape.size()) +
            " exceeds metadata header capacity (" + std::to_string(kMetaMaxDims) + ")");
    }
    Tensor header = empty({kMetaHeaderLen}, DType::Int64, Device::cpu());
    auto* h = static_cast<int64_t*>(header.data_ptr());
    for (int64_t i = 0; i < kMetaHeaderLen; ++i) h[i] = 0;
    h[0] = static_cast<int64_t>(shape.size());
    h[1] = static_cast<int64_t>(t.dtype());
    for (size_t d = 0; d < shape.size(); ++d) h[2 + d] = shape[d];
    return header;
}

/**
 * @brief Send a Variable's tensor to dst_rank via point-to-point.
 *
 * Sends a fixed-size metadata header first (shape/dtype) so the receiver can
 * allocate the matching buffer, then the activation payload itself.
 */
auto send_activation(const Variable& var, int dst_rank, ProcessGroup& pg) -> void {
    Tensor t = var.tensor();  // copy handle (shallow)
    Tensor header = encode_activation_meta(t);
    pg.send(header, dst_rank);
    pg.send(t, dst_rank);
}

/**
 * @brief Receive a tensor from src_rank using the wire-supplied metadata.
 *
 * Reads the fixed-size header (shape/dtype) sent by send_activation(), allocates
 * a buffer of exactly that shape/dtype on `device`, receives the payload, and
 * wraps it as a Variable. Does NOT assume the activation matches any caller-side
 * shape hint.
 */
auto recv_activation(int src_rank, ProcessGroup& pg, const Device& device,
                     bool requires_grad) -> Variable
{
    Tensor header = empty({kMetaHeaderLen}, DType::Int64, Device::cpu());
    pg.recv(header, src_rank);
    const auto* h = static_cast<const int64_t*>(header.data_ptr());
    int64_t ndim = h[0];
    if (ndim < 0 || ndim > kMetaMaxDims) {
        throw std::runtime_error(
            "pipeline_parallel: received activation metadata with invalid rank " +
            std::to_string(ndim));
    }
    // Validate the peer-supplied dtype: dtype_size() returns 0 only for an
    // out-of-range DType enum. Without this, a garbage h[1] reaches empty()/
    // dtype_size() and yields a divide-by-zero (SIGFPE) or a Tensor with a
    // bogus enum that later hits UB in dispatch. Mirror rpc_agent.cpp's check.
    auto dtype = static_cast<DType>(h[1]);
    if (dtype_size(dtype) == 0) {
        throw std::runtime_error(
            "pipeline_parallel: received activation metadata with invalid dtype " +
            std::to_string(h[1]));
    }

    // Copy dims with validation: reject negative dims and detect numel overflow
    // via checked multiplication before allocating, so a crafted header can't
    // produce a negative-extent or wrapped-size buffer.
    std::vector<int64_t> shape(static_cast<size_t>(ndim));
    int64_t numel = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t dim = h[2 + d];
        if (dim < 0) {
            throw std::runtime_error(
                "pipeline_parallel: received activation metadata with negative "
                "dim at index " + std::to_string(d));
        }
        if (__builtin_mul_overflow(numel, dim, &numel)) {
            throw std::runtime_error(
                "pipeline_parallel: received activation metadata with numel "
                "overflow");
        }
        shape[static_cast<size_t>(d)] = dim;
    }

    Tensor buf = empty(shape, dtype, device);
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
            // Receive activation from previous stage. Each recv reads a metadata
            // header (shape/dtype) off the wire, so per-micro-batch shapes may
            // differ and we never reuse the unrelated execute() input as a buffer
            // or shape hint. The device is taken from the schedule's input arg.
            mb_input = recv_activation(prev_rank, pg, input.tensor().device(),
                                       /*requires_grad=*/true);
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
            // with a unit gradient to propagate through the stage graph. A bare
            // backward() only works for a SCALAR output; the pipeline output is
            // generally non-scalar, so pass an explicit all-ones seed of the
            // output shape (which is what "unit gradient" means here).
            {
                auto os = micro_outputs[mb].tensor().shape();
                Tensor seed = ones(std::vector<int64_t>(os.begin(), os.end()),
                                   micro_outputs[mb].tensor().dtype(),
                                   micro_outputs[mb].tensor().device());
                micro_outputs[mb].backward(std::move(seed));
            }
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
            // Receive activation using the wire metadata header (shape/dtype),
            // so each micro-batch is sized correctly and the execute() input arg
            // is never reused as a recv buffer.
            mb_input = recv_activation(prev_rank, pg, input.tensor().device(),
                                       /*requires_grad=*/true);
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
