#include "tenzor/backend/loader.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include <mutex>
#include <shared_mutex>
#include <cstdlib>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tenzor {

// Forward declarations
auto backend_registry() -> BackendLoader&;

// Global registry destruction flag - defined at the end of this file
extern std::atomic<bool> g_registry_destroying;

auto BackendLoader::shutdown() -> void {
    if (shutdown_called_) {
        return;
    }
    shutdown_called_ = true;

    // Mark that we're destroying the registry - this prevents DeviceStorage
    // from trying to access backends during static destruction
    g_registry_destroying.store(true, std::memory_order_release);

    // Exclusive lock: shutdown mutates both maps
    std::unique_lock lock(registry_mutex_);

    // Clear device type mapping first
    device_to_backend_.clear();

    // Destroy backends in reverse dependency order to avoid cleanup issues
    const std::vector<std::string> destruction_order = {
        "oneapi",       // Independent SYCL runtime
        "vulkan",       // Independent GPU API
        "webgpu",       // Independent GPU API
        "rocm",         // AMD runtime
        "cuda",         // NVIDIA runtime
        "cpu"           // Always safe last (destructor calls mkl_cleanup)
    };

    for (const auto& name : destruction_order) {
        auto it = backends_.find(name);
        if (it != backends_.end()) {
            it->second.reset();
            backends_.erase(it);
        }
    }

    // Destroy any remaining backends not in the explicit order
    backends_.clear();

    // Do NOT dlclose backend libraries. Backend .so files pull in transitive
    // dependencies (MKL → TBB/tbbmalloc) that register their own static
    // destructors. Unloading these libraries removes code that those
    // destructors will call, causing segfaults during exit().
    // The OS reclaims all memory at process exit anyway.
    loaded_libraries_.clear();
}

BackendLoader::~BackendLoader() {
    // Fallback: if shutdown() was not called (e.g. finalize() was never invoked),
    // perform cleanup. During static destruction we skip dlclose since other
    // statics may still hold function pointers into backend libraries.
    if (!shutdown_called_) {
        g_registry_destroying.store(true, std::memory_order_release);
        device_to_backend_.clear();
        backends_.clear();
        // Skip dlclose — OS reclaims at exit
        loaded_libraries_.clear();
    }
}

auto BackendLoader::load_backend(const std::filesystem::path& library_path)
    -> std::expected<std::unique_ptr<Backend>, std::string> {

    auto handle = load_library(library_path);
    if (!handle) {
        #ifndef _WIN32
        const char* error = dlerror();
        std::string error_msg = error ? error : "unknown error";
        return std::unexpected("Failed to load library: " + library_path.string() + " - " + error_msg);
        #else
        return std::unexpected("Failed to load library: " + library_path.string());
        #endif
    }

    // Get factory function
    auto factory = reinterpret_cast<BackendFactory>(
        get_symbol(handle, "create_backend")
    );

    if (!factory) {
        unload_library(handle);
        return std::unexpected("Failed to find create_backend symbol");
    }

    // Create backend
    auto backend = factory();
    if (!backend) {
        unload_library(handle);
        return std::unexpected("Backend factory returned null");
    }

    auto name = std::string(backend->name());
    loaded_libraries_.push_back(handle);

    return backend;
}

auto BackendLoader::register_backend(std::string_view name,
                                     std::unique_ptr<Backend> backend) -> void {
    auto backend_name = std::string(name);

    // Determine device type from backend name
    Device::Type device_type;
    if (backend_name == "cpu") {
        device_type = Device::Type::CPU;
    } else if (backend_name == "cuda") {
        device_type = Device::Type::CUDA;
    } else if (backend_name == "rocm") {
        device_type = Device::Type::ROCm;
    } else if (backend_name == "oneapi") {
        device_type = Device::Type::OneAPI;
    } else if (backend_name == "vulkan") {
        device_type = Device::Type::Vulkan;
    } else if (backend_name == "webgpu") {
        device_type = Device::Type::WebGPU;
    } else {
        device_type = Device::Type::CPU; // Default fallback
    }

    // Store backend pointer before moving
    auto* backend_ptr = backend.get();

    // Exclusive lock: registration mutates both maps
    std::unique_lock lock(registry_mutex_);

    // Register by name
    backends_[backend_name] = std::move(backend);

    // Register by device type
    device_to_backend_[device_type] = backend_ptr;
}

auto BackendLoader::get_backend(std::string_view name) -> Backend* {
    std::shared_lock lock(registry_mutex_);
    auto it = backends_.find(std::string(name));
    return it != backends_.end() ? it->second.get() : nullptr;
}

auto BackendLoader::get_backend(Device::Type type) -> Backend* {
    std::shared_lock lock(registry_mutex_);
    auto it = device_to_backend_.find(type);
    return it != device_to_backend_.end() ? it->second : nullptr;
}

auto BackendLoader::has_backend(std::string_view name) const -> bool {
    std::shared_lock lock(registry_mutex_);
    return backends_.contains(std::string(name));
}

auto BackendLoader::available_backends() const -> std::vector<std::string> {
    std::shared_lock lock(registry_mutex_);
    std::vector<std::string> names;
    names.reserve(backends_.size());
    for (const auto& [name, _] : backends_) {
        names.push_back(name);
    }
    return names;
}

auto BackendLoader::unload_backend(std::string_view name) -> bool {
    std::unique_lock lock(registry_mutex_);

    auto it = backends_.find(std::string(name));
    if (it == backends_.end()) return false;

    Backend* ptr = it->second.get();

    // Remove from device_to_backend_ mapping to avoid dangling pointer
    for (auto dit = device_to_backend_.begin(); dit != device_to_backend_.end(); ) {
        if (dit->second == ptr) {
            // Clear the dispatch table for this device type
            DispatchTableRegistry::clear_backend(dit->first);
            dit = device_to_backend_.erase(dit);
        } else {
            ++dit;
        }
    }

    backends_.erase(it);
    return true;
}

auto BackendLoader::load_library(const std::filesystem::path& path) -> LibHandle {
    #ifdef _WIN32
        return LoadLibraryA(path.string().c_str());
    #else
        // Clear any existing error
        dlerror();
        // Use RTLD_LOCAL to prevent symbol pollution between backends.
        // TBB/tbbmalloc are pre-loaded with RTLD_GLOBAL in initialize()
        // so TBB's internal dlopen of tbbmalloc works even under RTLD_LOCAL.
        return dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    #endif
}

auto BackendLoader::unload_library(LibHandle handle) -> void {
    if (!handle) return;

    #ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(handle));
    #else
        dlclose(handle);
    #endif
}

auto BackendLoader::get_symbol(LibHandle handle, const char* name) -> void* {
    if (!handle) return nullptr;

    #ifdef _WIN32
        return GetProcAddress(static_cast<HMODULE>(handle), name);
    #else
        return dlsym(handle, name);
    #endif
}

// Global registry destruction flag - set when BackendLoader destructor starts
std::atomic<bool> g_registry_destroying{false};

// Global registry
auto backend_registry() -> BackendLoader& {
    static BackendLoader registry;
    return registry;
}

// Check if the backend registry is still alive and not being destroyed
// Safe to call from any thread during static destruction
auto is_backend_registry_alive() -> bool {
    return !g_registry_destroying.load(std::memory_order_acquire);
}

auto try_get_backend(Device::Type type) -> Backend* {
    // Single atomic check — if destroying, return nullptr immediately
    if (g_registry_destroying.load(std::memory_order_acquire)) {
        return nullptr;
    }
    return backend_registry().get_backend(type);
}

} // namespace tenzor
