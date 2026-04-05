/**
 * @file neon_kernels.hpp
 * @brief ARM NEON optimized kernels for the lite runtime
 */

#pragma once

#include "../lite_ops.hpp"

namespace tenzor::lite::platform {

/** Register ARM NEON-optimized kernels into the given registry. */
void register_neon_kernels(LiteOpsRegistry& registry);

}  // namespace tenzor::lite::platform
