#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

// 2D Max Pooling layer
class MaxPool2d : public Module {
public:
    MaxPool2d(int64_t kernel_size,
             int64_t stride = -1,  // Default: same as kernel_size
             int64_t padding = 0);

    auto forward(const Variable& input) -> Variable override;

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};

// 2D Average Pooling layer
class AvgPool2d : public Module {
public:
    AvgPool2d(int64_t kernel_size,
             int64_t stride = -1,  // Default: same as kernel_size
             int64_t padding = 0);

    auto forward(const Variable& input) -> Variable override;

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};

// 2D Adaptive Average Pooling layer
class AdaptiveAvgPool2d : public Module {
public:
    AdaptiveAvgPool2d(int64_t output_h, int64_t output_w);

    // Convenience constructor for square output
    explicit AdaptiveAvgPool2d(int64_t output_size)
        : AdaptiveAvgPool2d(output_size, output_size) {}

    auto forward(const Variable& input) -> Variable override;

private:
    int64_t output_h_;
    int64_t output_w_;
};

} // namespace nn
} // namespace tenzor
