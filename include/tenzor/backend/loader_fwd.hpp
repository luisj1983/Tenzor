/**
 * @file loader_fwd.hpp
 * @brief Forward declarations for BackendLoader and backend_registry()
 *
 * Lightweight header that declares the backend registry interface without
 * pulling in <expected> (C++23). Use this instead of loader.hpp in headers
 * that only need backend_registry() / get_backend() — particularly in files
 * compiled by nvcc or hipcc which may not support C++23.
 *
 * @see loader.hpp for full BackendLoader definition
 */

#pragma once

#include "../core/device.hpp"

namespace tenzor {

class Backend;
class BackendLoader;

/**
 * @brief Get global backend registry singleton.
 * @return Reference to global BackendLoader instance
 * @see loader.hpp for full documentation
 */
auto backend_registry() -> BackendLoader&;

/**
 * @brief Check if the backend registry is still alive.
 * @return true if registry is alive and safe to use
 */
auto is_backend_registry_alive() -> bool;

/**
 * @brief Atomically check if registry is alive and return backend.
 * @param type Device type to look up
 * @return Backend pointer, or nullptr if registry is dead or backend not found
 */
auto try_get_backend(Device::Type type) -> Backend*;

} // namespace tenzor
