/**
 * @file autocast_guard_test_support.hpp
 * @brief Test-only RAII autocast guard.
 *
 * AutocastGuard was previously declared in include/tenzor/nn/amp/autocast.hpp
 * but its only users are the autocast tests (test_autocast.cpp and
 * test_autocast_multidtype.cpp). The production Autocast / AutocastDisabled
 * classes it wraps remain in the production header. Relocated here.
 *
 * Header-only so the test executables can include it without CMake changes.
 * Lives in namespace tenzor::nn::amp so the test call sites (which do
 * `using namespace tenzor::nn::amp;`) remain unchanged.
 */

#pragma once

#include "tenzor/nn/amp/autocast.hpp"
#include "tenzor/core/dtype.hpp"

namespace tenzor {
namespace nn {
namespace amp {

/**
 * @brief RAII helper for autocast context
 *
 * Usage:
 * @code
 * {
 *     AutocastGuard guard(true, DType::Float16);
 *     auto output = model.forward(input);  // Operations auto-cast to FP16
 * }
 * // Autocast disabled after scope
 * @endcode
 */
class AutocastGuard {
public:
    explicit AutocastGuard(bool enabled, DType dtype = DType::Float16)
        : autocast_(enabled, dtype) {}

private:
    Autocast autocast_;
};

} // namespace amp
} // namespace nn
} // namespace tenzor
