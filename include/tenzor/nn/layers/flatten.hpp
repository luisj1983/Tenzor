#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

// Flatten layer - flattens input from start_dim onwards
class Flatten : public Module {
public:
    explicit Flatten(int64_t start_dim = 1, int64_t end_dim = -1);

    auto forward(const Variable& input) -> Variable override;

private:
    int64_t start_dim_;
    int64_t end_dim_;
};

} // namespace nn
} // namespace tenzor
