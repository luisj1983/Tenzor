#pragma once

#include "sampler.hpp"

namespace tenzor {
namespace data {

/**
 * @brief Distributed sampler that partitions indices across replicas
 *
 * Each replica gets a disjoint subset of the dataset indices, ensuring
 * no overlap during distributed training. Supports optional shuffling
 * with epoch-based reseeding for reproducibility.
 *
 * Index assignment follows the pattern: indices[rank::num_replicas],
 * i.e., rank 0 gets indices {0, N, 2N, ...}, rank 1 gets {1, N+1, 2N+1, ...}.
 */
class DistributedSampler : public Sampler {
public:
    /**
     * @brief Construct a distributed sampler
     *
     * @param dataset_size Total number of samples in the dataset
     * @param num_replicas Number of distributed replicas (world_size)
     * @param rank Rank of the current replica
     * @param shuffle Whether to shuffle indices each epoch
     * @param seed Base random seed for shuffling
     * @param drop_last Drop samples to make dataset evenly divisible
     */
    DistributedSampler(size_t dataset_size, int num_replicas, int rank,
                       bool shuffle = true, int64_t seed = 0, bool drop_last = false);

    auto indices() const -> const std::vector<size_t>& override { return indices_; }
    auto size() const -> size_t override { return indices_.size(); }
    auto reset(int64_t epoch = 0) -> void override;

private:
    size_t dataset_size_;
    int num_replicas_, rank_;
    bool shuffle_;
    int64_t seed_;
    bool drop_last_;
};

} // namespace data
} // namespace tenzor
