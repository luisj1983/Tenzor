/**
 * @file cpu_kernels.hpp
 * @brief Portable C++ reference kernels for the lite runtime
 */

#pragma once

#include "../lite_ops.hpp"

namespace tenzor::lite::platform {

/** Register portable CPU reference kernels into the given registry. */
void register_cpu_kernels(LiteOpsRegistry& registry);

}  // namespace tenzor::lite::platform
