/**
 * @file loader.hpp
 * @brief Dynamic backend loading and management system
 *
 * Provides facilities for loading backend implementations from shared
 * libraries at runtime, enabling extensible hardware support without
 * recompilation. Supports plugin-style backend architecture.
 */

#pragma once

#include <memory>
#include <filesystem>
#include <expected>
#include <string>
#include <vector>
#include <shared_mutex>
#include <unordered_map>
#include "backend.hpp"

namespace tenzor {

/**
 * @brief Dynamic backend loader and registry.
 *
 * Manages loading and unloading of backend implementations from shared
 * libraries (.so, .dll, .dylib). Maintains a registry of available backends
 * and provides lookup by name or device type.
 *
 * Backends can be:
 * - Loaded dynamically from shared libraries via load_backend()
 * - Registered directly via register_backend() (for statically linked backends)
 *
 * The loader handles platform-specific library loading (dlopen on Unix,
 * LoadLibrary on Windows) and symbol resolution.
 *
 * @code
 * BackendLoader loader;
 *
 * // Load CUDA backend from shared library
 * auto result = loader.load_backend("./libtenzor_cuda.so");
 * if (result) {
 *     Backend* cuda = loader.get_backend("cuda");
 *     std::cout << "CUDA devices: " << cuda->device_count() << std::endl;
 * }
 * @endcode
 *
 * @note This class is non-copyable to prevent duplicate library handles.
 * @see Backend for backend interface
 * @see backend_registry() for global singleton access
 */
class BackendLoader {
public:
    BackendLoader() = default;

    /**
     * @brief Destructor unloads all loaded libraries.
     */
    ~BackendLoader();

    BackendLoader(const BackendLoader&) = delete;
    BackendLoader& operator=(const BackendLoader&) = delete;

    /**
     * @brief Load backend from shared library.
     *
     * Loads a backend plugin from a shared library file. The library must
     * export a "create_backend" symbol returning BackendFactory.
     *
     * @param library_path Path to shared library (.so/.dll/.dylib)
     * @return Backend on success, error message on failure
     *
     * @code
     * auto result = loader.load_backend("./libtenzor_cuda.so");
     * if (result) {
     *     std::cout << "Loaded backend: " << (*result)->name() << std::endl;
     * } else {
     *     std::cerr << "Error: " << result.error() << std::endl;
     * }
     * @endcode
     *
     * @note The library is kept open until unload_backend() or destruction.
     */
    auto load_backend(const std::filesystem::path& library_path)
        -> std::expected<std::unique_ptr<Backend>, std::string>;

    /**
     * @brief Register backend directly without library loading.
     *
     * Used for statically linked backends or backends created at runtime.
     *
     * @param name Backend name identifier (e.g., "cpu", "cuda")
     * @param backend Backend instance to register
     *
     * @code
     * auto cpu_backend = std::make_unique<CPUBackend>();
     * loader.register_backend("cpu", std::move(cpu_backend));
     * @endcode
     */
    auto register_backend(std::string_view name,
                         std::unique_ptr<Backend> backend) -> void;

    /**
     * @brief Get backend by name.
     *
     * @param name Backend name (e.g., "cpu", "cuda", "rocm")
     * @return Backend pointer or nullptr if not found
     *
     * @code
     * Backend* cuda = loader.get_backend("cuda");
     * if (cuda && cuda->is_available()) {
     *     // Use CUDA backend
     * }
     * @endcode
     */
    auto get_backend(std::string_view name) -> Backend*;

    /**
     * @brief Get backend by device type.
     *
     * @param type Device type (CPU, CUDA, ROCm, OneAPI)
     * @return Backend pointer or nullptr if not registered
     *
     * @code
     * Backend* cuda = loader.get_backend(Device::Type::CUDA);
     * @endcode
     */
    auto get_backend(Device::Type type) -> Backend*;

    /**
     * @brief Check if backend is registered.
     *
     * @param name Backend name to check
     * @return true if backend is registered and available
     */
    auto has_backend(std::string_view name) const -> bool;

    /**
     * @brief Get list of all registered backend names.
     *
     * @return Vector of backend name strings
     *
     * @code
     * for (const auto& name : loader.available_backends()) {
     *     std::cout << "Backend: " << name << std::endl;
     * }
     * @endcode
     */
    auto available_backends() const -> std::vector<std::string>;

    /**
     * @brief Unload and unregister backend.
     *
     * Removes backend from registry and unloads shared library if applicable.
     *
     * @param name Backend name to unload
     * @return true if backend was found and unloaded
     *
     * @warning Unloading a backend invalidates all pointers to it.
     */
    auto unload_backend(std::string_view name) -> bool;

    /**
     * @brief Get the dlopen handle for the most recently loaded library.
     *
     * Returns the handle from the last load_backend() call, which can be
     * used for dlsym() without needing a second RTLD_NOLOAD dlopen.
     *
     * @return Library handle or nullptr if no libraries loaded
     */
    auto last_library_handle() const -> void* {
        return loaded_libraries_.empty() ? nullptr : loaded_libraries_.back();
    }

    /**
     * @brief Perform ordered shutdown of all backends.
     *
     * Destroys backends in reverse dependency order and dlcloses all loaded
     * libraries. The destructor becomes a no-op after shutdown() has been called.
     * This should be called from finalize() before static destructors run.
     */
    auto shutdown() -> void;

private:
    bool shutdown_called_{false};                                            ///< Whether shutdown() has been called
    mutable std::shared_mutex registry_mutex_;                               ///< Guards backends_ and device_to_backend_ for thread-safe access
    std::unordered_map<std::string, std::unique_ptr<Backend>> backends_;     ///< Registered backends
    std::unordered_map<Device::Type, Backend*> device_to_backend_;           ///< Device type mapping

    // Platform-specific library handle
#ifdef _WIN32
    using LibHandle = void*; ///< HMODULE on Windows
#else
    using LibHandle = void*; ///< void* (dlopen handle) on Unix
#endif

    std::vector<LibHandle> loaded_libraries_;  ///< Loaded library handles

    /**
     * @brief Load shared library.
     * @param path Library file path
     * @return Library handle or nullptr on failure
     */
    auto load_library(const std::filesystem::path& path) -> LibHandle;

    /**
     * @brief Unload shared library.
     * @param handle Library handle to unload
     */
    auto unload_library(LibHandle handle) -> void;

    /**
     * @brief Resolve symbol from library.
     * @param handle Library handle
     * @param name Symbol name to resolve
     * @return Symbol address or nullptr if not found
     */
    auto get_symbol(LibHandle handle, const char* name) -> void*;
};

/**
 * @brief Get global backend registry singleton.
 *
 * Thread-safe singleton providing access to the global backend loader.
 * Use this to access backends from anywhere in the application.
 *
 * @return Reference to global BackendLoader instance
 *
 * @code
 * Backend* cpu = backend_registry().get_backend("cpu");
 * auto backends = backend_registry().available_backends();
 * @endcode
 *
 * @note This function is thread-safe (uses static local initialization).
 */
auto backend_registry() -> BackendLoader&;

/**
 * @brief Check if the backend registry is still alive.
 *
 * Returns false during static destruction when the BackendLoader is being
 * or has been destroyed. Use this before calling backend_registry() from
 * destructors to avoid accessing destroyed memory.
 *
 * @return true if registry is alive and safe to use, false during destruction
 *
 * @note Thread-safe using atomic operations.
 */
auto is_backend_registry_alive() -> bool;

/**
 * @brief Atomically check if registry is alive and return backend.
 *
 * Eliminates the TOCTOU race between is_backend_registry_alive() and
 * backend_registry().get_backend(). Safe to call from destructors
 * during static destruction.
 *
 * @param type Device type to look up
 * @return Backend pointer, or nullptr if registry is dead or backend not found
 */
auto try_get_backend(Device::Type type) -> Backend*;

} // namespace tenzor
