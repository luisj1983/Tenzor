/**
 * @file pipeline_parallel.hpp
 * @brief Pipeline parallelism for distributed model training
 *
 * Provides pipeline stages and scheduling strategies (GPipe, 1F1B) that
 * partition a model across multiple ranks and overlap computation with
 * inter-stage communication via point-to-point send/recv.
 */

#pragma once

#include "distributed.hpp"
#include "../nn/module.hpp"
#include "../autograd/variable.hpp"
#include <memory>
#include <vector>

namespace tenzor {
namespace distributed {

/**
 * @brief A pipeline stage wrapping a subset of layers assigned to one rank.
 *
 * Each stage owns a sub-module that processes its portion of the model.
 * Stages communicate via point-to-point operations on a ProcessGroup:
 * activations flow forward (stage i -> stage i+1), gradients flow
 * backward (stage i+1 -> stage i).
 */
class PipelineStage {
public:
    /**
     * @brief Construct a pipeline stage.
     *
     * @param module The sub-module (layers) assigned to this stage
     * @param stage_id Zero-based index of this stage
     * @param num_stages Total number of pipeline stages
     */
    PipelineStage(std::shared_ptr<nn::Module> module, int stage_id, int num_stages);

    /**
     * @brief Run the sub-module's forward pass on the given input.
     *
     * @param input Activation tensor from the previous stage (or original input)
     * @return Output activation tensor
     */
    auto forward(const Variable& input) -> Variable;

    /**
     * @brief Access the underlying module.
     */
    auto module() -> nn::Module& { return *module_; }

    /**
     * @brief Access the underlying module (const).
     */
    auto module() const -> const nn::Module& { return *module_; }

    /**
     * @brief Get this stage's zero-based index.
     */
    auto stage_id() const -> int { return stage_id_; }

    /**
     * @brief Get total number of pipeline stages.
     */
    auto num_stages() const -> int { return num_stages_; }

    /**
     * @brief Check if this is the first stage in the pipeline.
     */
    auto is_first() const -> bool { return stage_id_ == 0; }

    /**
     * @brief Check if this is the last stage in the pipeline.
     */
    auto is_last() const -> bool { return stage_id_ == num_stages_ - 1; }

private:
    std::shared_ptr<nn::Module> module_;
    int stage_id_;
    int num_stages_;
};

/**
 * @brief Base class for pipeline execution schedules.
 *
 * A schedule determines the order in which micro-batch forward and backward
 * passes are interleaved to minimize pipeline bubble time.
 */
class PipelineSchedule {
public:
    virtual ~PipelineSchedule() = default;

    /**
     * @brief Execute the pipeline schedule for one training iteration.
     *
     * Splits the input into micro-batches, orchestrates forward/backward
     * passes through the stage, and communicates activations/gradients
     * with neighboring stages via point-to-point ops on the ProcessGroup.
     *
     * @param stage The local pipeline stage
     * @param input Full-batch input (only meaningful on the first stage)
     * @param num_microbatches Number of micro-batches to split the input into
     * @param pg ProcessGroup for inter-stage send/recv
     * @return Concatenated output from all micro-batches (only meaningful on the last stage)
     */
    virtual auto execute(PipelineStage& stage, const Variable& input,
                         int num_microbatches, ProcessGroup& pg) -> Variable = 0;
};

/**
 * @brief GPipe schedule: all-forward then all-backward.
 *
 * Executes all micro-batch forward passes sequentially, then all backward
 * passes in reverse order. Simple but has a pipeline bubble of size
 * (p-1)/p where p is the number of stages.
 *
 * Reference: Huang et al., "GPipe: Efficient Training of Giant Neural
 * Networks using Pipeline Parallelism", NeurIPS 2019.
 */
class GPipeSchedule : public PipelineSchedule {
public:
    /**
     * @brief Execute the GPipe schedule.
     *
     * 1. Split input into num_microbatches micro-batches along dim 0
     * 2. Forward all micro-batches through the local stage
     *    - If not first stage: recv activations from previous rank
     *    - If not last stage: send activations to next rank
     * 3. Backward all micro-batches in reverse order
     *    - If not last stage: recv gradients from next rank
     *    - If not first stage: send gradients to previous rank
     * 4. Concatenate outputs and return
     */
    auto execute(PipelineStage& stage, const Variable& input,
                 int num_microbatches, ProcessGroup& pg) -> Variable override;
};

/**
 * @brief 1F1B (one-forward-one-backward) interleaved schedule.
 *
 * Interleaves forward and backward passes to reduce peak activation
 * memory and pipeline bubble time. The bubble ratio is
 * (p-1)/(m+p-1) where p=num_stages and m=num_microbatches.
 *
 * Phases:
 * 1. Warmup: p-1-stage_id forward passes
 * 2. Steady state: alternate 1 forward + 1 backward
 * 3. Cooldown: drain remaining backward passes
 *
 * Reference: Narayanan et al., "Efficient Large-Scale Language Model
 * Training on GPU Clusters Using Megatron-LM", SC 2021.
 */
class OneFOneBSchedule : public PipelineSchedule {
public:
    /**
     * @brief Execute the 1F1B schedule.
     *
     * Maintains a queue of stashed activations for each micro-batch
     * to enable interleaved forward/backward execution.
     */
    auto execute(PipelineStage& stage, const Variable& input,
                 int num_microbatches, ProcessGroup& pg) -> Variable override;
};

} // namespace distributed
} // namespace tenzor
