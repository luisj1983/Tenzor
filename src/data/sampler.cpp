#include "tenzor/data/sampler.hpp"
#include "tenzor/data/distributed_sampler.hpp"
#include <numeric>
#include <stdexcept>

namespace tenzor {
namespace data {

// ============================================================================
// SequentialSampler
// ============================================================================

SequentialSampler::SequentialSampler(size_t dataset_size)
    : dataset_size_(dataset_size) {
    reset();
}

auto SequentialSampler::reset(int64_t /*epoch*/) -> void {
    indices_.resize(dataset_size_);
    std::iota(indices_.begin(), indices_.end(), size_t{0});
}

// ============================================================================
// RandomSampler
// ============================================================================

RandomSampler::RandomSampler(size_t dataset_size, int64_t seed)
    : dataset_size_(dataset_size), seed_(seed) {
    reset();
}

auto RandomSampler::reset(int64_t epoch) -> void {
    indices_.resize(dataset_size_);
    std::iota(indices_.begin(), indices_.end(), size_t{0});

    std::mt19937 rng(static_cast<unsigned>(seed_ + epoch));
    std::shuffle(indices_.begin(), indices_.end(), rng);
}

// ============================================================================
// DistributedSampler
// ============================================================================

DistributedSampler::DistributedSampler(
    size_t dataset_size, int num_replicas, int rank,
    bool shuffle, int64_t seed, bool drop_last)
    : dataset_size_(dataset_size),
      num_replicas_(num_replicas),
      rank_(rank),
      shuffle_(shuffle),
      seed_(seed),
      drop_last_(drop_last) {

    if (num_replicas <= 0) {
        throw std::invalid_argument("num_replicas must be positive");
    }
    if (rank < 0 || rank >= num_replicas) {
        throw std::invalid_argument("rank must be in [0, num_replicas)");
    }

    reset();
}

auto DistributedSampler::reset(int64_t epoch) -> void {
    // Generate full index range
    std::vector<size_t> all_indices(dataset_size_);
    std::iota(all_indices.begin(), all_indices.end(), size_t{0});

    // Optionally shuffle with deterministic seed based on epoch
    if (shuffle_) {
        std::mt19937 rng(static_cast<unsigned>(seed_ + epoch));
        std::shuffle(all_indices.begin(), all_indices.end(), rng);
    }

    // Pad to make evenly divisible if not dropping last
    if (!drop_last_) {
        size_t remainder = all_indices.size() % static_cast<size_t>(num_replicas_);
        if (remainder != 0) {
            size_t padding = static_cast<size_t>(num_replicas_) - remainder;
            for (size_t i = 0; i < padding; ++i) {
                all_indices.push_back(all_indices[i % dataset_size_]);
            }
        }
    }

    // Slice: take every num_replicas-th element starting at rank
    indices_.clear();
    for (size_t i = static_cast<size_t>(rank_); i < all_indices.size();
         i += static_cast<size_t>(num_replicas_)) {
        indices_.push_back(all_indices[i]);
    }
}

} // namespace data
} // namespace tenzor
