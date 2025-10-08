#pragma once

#include <optional>
#include "../module.hpp"

namespace tenzor {
namespace nn {

// 2D Convolution layer
class Conv2d : public Module {
public:
    Conv2d(int64_t in_channels,
           int64_t out_channels,
           int64_t kernel_size,
           int64_t stride = 1,
           int64_t padding = 0,
           int64_t dilation = 1,
           int64_t groups = 1,
           bool bias = true);

    auto forward(const Variable& input) -> Variable override;

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t groups_;

    Variable weight_;  // [out_channels, in_channels/groups, kernel_size, kernel_size]
    std::optional<Variable> bias_;  // [out_channels]

    auto reset_parameters() -> void;
};

// 1D Convolution layer
class Conv1d : public Module {
public:
    Conv1d(int64_t in_channels,
           int64_t out_channels,
           int64_t kernel_size,
           int64_t stride = 1,
           int64_t padding = 0,
           int64_t dilation = 1,
           int64_t groups = 1,
           bool bias = true);

    auto forward(const Variable& input) -> Variable override;

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t groups_;

    Variable weight_;
    std::optional<Variable> bias_;

    auto reset_parameters() -> void;
};

// Transposed convolution (deconvolution)
class ConvTranspose2d : public Module {
public:
    ConvTranspose2d(int64_t in_channels,
                    int64_t out_channels,
                    int64_t kernel_size,
                    int64_t stride = 1,
                    int64_t padding = 0,
                    int64_t output_padding = 0,
                    int64_t groups = 1,
                    bool bias = true);

    auto forward(const Variable& input) -> Variable override;

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t output_padding_;
    int64_t groups_;

    Variable weight_;
    std::optional<Variable> bias_;

    auto reset_parameters() -> void;
};

} // namespace nn
} // namespace tenzor
