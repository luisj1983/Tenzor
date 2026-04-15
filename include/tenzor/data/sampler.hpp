#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tenzor {
namespace data {

/**
 * @brief Abstract base class for dataset samplers
 *
 * Samplers control the order in which dataset indices are visited
 * during training. They support epoch-based reseeding for reproducible
 * shuffling across distributed workers.
 */
class Sampler {
public:
    virtual ~Sampler() = default;

    /** @brief Get the current index ordering */
    virtual auto indices() const -> const std::vector<size_t>& = 0;

    /** @brief Get the number of indices */
    virtual auto size() const -> size_t = 0;

    /** @brief Reset sampler state for a new epoch */
    virtual auto reset(int64_t epoch = 0) -> void = 0;

protected:
    std::vector<size_t> indices_;
};

/**
 * @brief Sequential sampler that visits indices in order 0, 1, ..., N-1
 */
class SequentialSampler : public Sampler {
public:
    explicit SequentialSampler(size_t dataset_size);

    auto indices() const -> const std::vector<size_t>& override { return indices_; }
    auto size() const -> size_t override { return indices_.size(); }
    auto reset(int64_t epoch = 0) -> void override;

private:
    size_t dataset_size_;
};

/**
 * @brief Random sampler that shuffles indices each epoch
 *
 * Uses mt19937 seeded with (seed + epoch) for reproducible shuffling.
 */
class RandomSampler : public Sampler {
public:
    explicit RandomSampler(size_t dataset_size, int64_t seed = 0);

    auto indices() const -> const std::vector<size_t>& override { return indices_; }
    auto size() const -> size_t override { return indices_.size(); }
    auto reset(int64_t epoch = 0) -> void override;

private:
    size_t dataset_size_;
    int64_t seed_;
};

/**
 * @brief Weighted random sampler that draws indices according to given probabilities
 *
 * Samples elements from [0, len(weights)) with probability proportional to
 * the given weights. Supports sampling with or without replacement.
 * Uses mt19937 seeded with (seed + epoch) for reproducible sampling.
 */
class WeightedRandomSampler : public Sampler {
public:
    /**
     * @brief Construct a weighted random sampler.
     *
     * @param weights Sampling weights (must be non-negative, at least one positive)
     * @param num_samples Number of samples to draw
     * @param replacement If true, sample with replacement; otherwise without
     * @param seed Random seed for reproducibility
     *
     * @throws std::invalid_argument if weights is empty or contains negative values
     * @throws std::invalid_argument if num_samples > weights.size() and replacement is false
     */
    WeightedRandomSampler(std::vector<double> weights, int64_t num_samples, bool replacement = true, int64_t seed = 0);

    auto indices() const -> const std::vector<size_t>& override { return indices_; }
    auto size() const -> size_t override { return indices_.size(); }
    auto reset(int64_t epoch = 0) -> void override;

private:
    std::vector<double> weights_;
    int64_t num_samples_;
    bool replacement_;
    int64_t seed_;
};

/**
 * @brief Sampler that randomly shuffles a given list of indices
 *
 * Useful for sampling from a predefined subset of a dataset. Each epoch,
 * the subset indices are reshuffled using mt19937 seeded with (seed + epoch).
 */
class SubsetRandomSampler : public Sampler {
public:
    /**
     * @brief Construct a subset random sampler.
     *
     * @param indices Indices to sample from
     * @param seed Random seed for reproducibility
     */
    explicit SubsetRandomSampler(std::vector<size_t> indices, int64_t seed = 0);

    auto indices() const -> const std::vector<size_t>& override { return indices_; }
    auto size() const -> size_t override { return indices_.size(); }
    auto reset(int64_t epoch = 0) -> void override;

private:
    std::vector<size_t> subset_indices_;
    int64_t seed_;
};

} // namespace data
} // namespace tenzor
