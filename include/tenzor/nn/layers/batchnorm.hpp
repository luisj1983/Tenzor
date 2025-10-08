#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

// Batch Normalization 2D
class BatchNorm2d : public Module {
public:
    BatchNorm2d(int64_t num_features,
                double eps = 1e-5,
                double momentum = 0.1,
                bool affine = true,
                bool track_running_stats = true);

    auto forward(const Variable& input) -> Variable override;

private:
    int64_t num_features_;
    double eps_;
    double momentum_;
    bool affine_;
    bool track_running_stats_;

    Variable weight_;  // [num_features]
    Variable bias_;    // [num_features]
    Variable running_mean_;  // [num_features]
    Variable running_var_;   // [num_features]
    int64_t num_batches_tracked_{0};

    auto reset_parameters() -> void;
};

// Batch Normalization 1D
class BatchNorm1d : public Module {
public:
    BatchNorm1d(int64_t num_features,
                double eps = 1e-5,
                double momentum = 0.1,
                bool affine = true,
                bool track_running_stats = true);

    auto forward(const Variable& input) -> Variable override;

private:
    int64_t num_features_;
    double eps_;
    double momentum_;
    bool affine_;
    bool track_running_stats_;

    Variable weight_;
    Variable bias_;
    Variable running_mean_;
    Variable running_var_;
    int64_t num_batches_tracked_{0};

    auto reset_parameters() -> void;
};

// Layer Normalization
class LayerNorm : public Module {
public:
    LayerNorm(std::vector<int64_t> normalized_shape,
              double eps = 1e-5,
              bool elementwise_affine = true);

    auto forward(const Variable& input) -> Variable override;

private:
    std::vector<int64_t> normalized_shape_;
    double eps_;
    bool elementwise_affine_;

    Variable weight_;
    Variable bias_;

    auto reset_parameters() -> void;
};

} // namespace nn
} // namespace tenzor
