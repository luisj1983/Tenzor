/**
 * @file kernel_registry.hpp
 * @brief CPU kernel registration function declaration
 */

#pragma once

#include "tenzor/backend/dispatch_table.hpp"

namespace tenzor {

/**
 * @brief Register all CPU kernels with the dispatch table.
 *
 * This function populates the CPU dispatch table with direct function
 * pointers to all CPU kernel implementations.
 *
 * @param table The CPU backend dispatch table to populate
 */
void register_cpu_kernels(BackendDispatchTable& table);

} // namespace tenzor
