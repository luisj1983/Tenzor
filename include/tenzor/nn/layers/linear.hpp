#pragma once

#include <optional>
#include "../module.hpp"

namespace tenzor {
namespace nn {

// Linear (fully connected) layer
class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, bool bias = true);

    auto forward(const Variable& input) -> Variable override;

    // Access weights
    auto weight() const -> const Variable& { return weight_; }
    auto bias() const -> const std::optional<Variable>& { return bias_; }

private:
    int64_t in_features_;
    int64_t out_features_;
    Variable weight_;  // [out_features, in_features]
    std::optional<Variable> bias_;  // [out_features]

    auto reset_parameters() -> void;
};

} // namespace nn
} // namespace tenzor
