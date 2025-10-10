#pragma once

#include "../module.hpp"
#include <vector>

namespace tenzor {
namespace nn {

// Layer Normalization
// Normalizes over the last D dimensions specified by normalized_shape
// For input [N, C, H, W] with normalized_shape=[C, H, W], normalizes over all except batch dimension
class LayerNorm : public Module {
public:
    // normalized_shape: dimensions to normalize over (from the end)
    // eps: small constant for numerical stability
    // elementwise_affine: whether to learn affine parameters (gamma, beta)
    LayerNorm(std::vector<int64_t> normalized_shape,
              double eps = 1e-5,
              bool elementwise_affine = true);

    auto forward(const Variable& input) -> Variable override;

private:
    std::vector<int64_t> normalized_shape_;
    double eps_;
    bool elementwise_affine_;
    int64_t num_features_;  // Product of normalized_shape

    Variable weight_;  // gamma
    Variable bias_;    // beta

    auto reset_parameters() -> void;
};

// Group Normalization
// Divides channels into groups and normalizes each group independently
// For input [N, C, H, W], divides C into num_groups groups
class GroupNorm : public Module {
public:
    // num_groups: number of groups to divide channels into
    // num_channels: number of channels (C dimension)
    // eps: small constant for numerical stability
    // affine: whether to learn affine parameters (gamma, beta)
    GroupNorm(int64_t num_groups,
              int64_t num_channels,
              double eps = 1e-5,
              bool affine = true);

    auto forward(const Variable& input) -> Variable override;

private:
    int64_t num_groups_;
    int64_t num_channels_;
    double eps_;
    bool affine_;

    Variable weight_;  // gamma [num_channels]
    Variable bias_;    // beta [num_channels]

    auto reset_parameters() -> void;
};

} // namespace nn
} // namespace tenzor
