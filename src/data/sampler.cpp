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
// WeightedRandomSampler
// ============================================================================

WeightedRandomSampler::WeightedRandomSampler(
    std::vector<double> weights, int64_t num_samples, bool replacement, int64_t seed)
    : weights_(std::move(weights)),
      num_samples_(num_samples),
      replacement_(replacement),
      seed_(seed) {

    if (weights_.empty()) {
        throw std::invalid_argument("weights must not be empty");
    }
    for (auto w : weights_) {
        if (w < 0.0) {
            throw std::invalid_argument("weights must be non-negative");
        }
    }
    if (!replacement_) {
        // Without replacement, each draw permanently zeroes the selected
        // index's weight. Once the strictly-positive weights are exhausted the
        // remaining distribution has zero total weight, which std::discrete_-
        // distribution treats as uniform (returning duplicate / zero-weight
        // indices and violating the without-replacement contract). Require that
        // there are at least num_samples_ strictly-positive weights.
        int64_t positive_count = 0;
        for (auto w : weights_) {
            if (w > 0.0) {
                ++positive_count;
            }
        }
        if (num_samples_ > positive_count) {
            throw std::invalid_argument(
                "num_samples must be <= number of strictly-positive weights "
                "when sampling without replacement");
        }
    }

    reset();
}

auto WeightedRandomSampler::reset(int64_t epoch) -> void {
    indices_.clear();
    indices_.reserve(static_cast<size_t>(num_samples_));

    std::mt19937 rng(static_cast<unsigned>(seed_ + epoch));
    std::discrete_distribution<size_t> dist(weights_.begin(), weights_.end());

    if (replacement_) {
        for (int64_t i = 0; i < num_samples_; ++i) {
            indices_.push_back(dist(rng));
        }
    } else {
        // Sample without replacement: rebuild distribution after each selection
        // to avoid rejection-loop stalls when high-weight indices dominate
        std::vector<double> remaining_weights(weights_);
        for (int64_t i = 0; i < num_samples_; ++i) {
            std::discrete_distribution<size_t> d(remaining_weights.begin(),
                                                  remaining_weights.end());
            size_t idx = d(rng);
            indices_.push_back(idx);
            remaining_weights[idx] = 0.0;  // Remove selected index
        }
    }
}

// ============================================================================
// SubsetRandomSampler
// ============================================================================

SubsetRandomSampler::SubsetRandomSampler(std::vector<size_t> indices, int64_t seed)
    : subset_indices_(std::move(indices)), seed_(seed) {
    reset();
}

auto SubsetRandomSampler::reset(int64_t epoch) -> void {
    indices_ = subset_indices_;

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
