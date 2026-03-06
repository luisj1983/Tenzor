/**
 * @file identity.hpp
 * @brief Identity layer that passes input through unchanged.
 */

#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Identity layer — passes input through unchanged.
 *
 * Useful as a placeholder in architectures where a layer may be
 * optionally replaced (e.g., skip connections, conditional layers).
 *
 * @code
 * Identity id;
 * Variable x(Tensor({2, 3}, DType::Float32, Device::cpu()), true);
 * Variable y = id.forward(x);  // y is identical to x
 * @endcode
 */
class Identity : public Module {
public:
    Identity() = default;
    auto forward_impl(const Variable& input) -> Variable override { return input; }
};

} // namespace nn
} // namespace tenzor
