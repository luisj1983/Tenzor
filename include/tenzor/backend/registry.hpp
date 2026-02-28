/**
 * @file registry.hpp
 * @brief Backend registry support header
 *
 * Previously contained OperationRegistry (string-based dispatch).
 * That system has been fully replaced by OpId-based dispatch via
 * BackendDispatchTable in kernel_registry.hpp.
 *
 * This header is retained for transitive includes of backend.hpp.
 */

#pragma once

#include "backend.hpp"

namespace tenzor {

} // namespace tenzor
