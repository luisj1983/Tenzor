#include "tenzor/backend/loader.hpp"
#include <mutex>
#include <cstdlib>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace tenzor {

// Forward declaration
auto backend_registry() -> BackendLoader&;

/**
 * @brief Early cleanup guard to ensure AdaptiveCpp cleanup happens before CUDA driver unloads.
 *
 * This uses std::atexit to register a cleanup handler. Since atexit handlers run in
 * LIFO order (last registered runs first), and the CUDA runtime registers its cleanup
 * early during library load, we need to register AFTER the static BackendLoader is
 * initialized but we want our cleanup to run BEFORE static destruction begins.
 *
 * The key insight is that atexit handlers run BEFORE static destructors. So by
 * registering an atexit handler that explicitly cleans up the AdaptiveCpp backend,
 * we ensure it's destroyed while the CUDA runtime is still available.
 *
 * See: https://github.com/AdaptiveCpp/AdaptiveCpp/issues/817
 */
static bool g_atexit_registered = false;

static void cleanup_adaptivecpp_before_cuda() {
    // Explicitly destroy AdaptiveCpp backend before CUDA driver unloads
    auto& loader = backend_registry();
    if (loader.has_backend("adaptivecpp")) {
        loader.unload_backend("adaptivecpp");
    }
}

BackendLoader::~BackendLoader() {
    // IMPORTANT: Destroy backends in reverse dependency order to avoid cleanup issues.
    // AdaptiveCpp (when using CUDA target) depends on CUDA runtime, so it must be
    // destroyed BEFORE CUDA backend. The order matters because AdaptiveCpp creates
    // CUDA streams that become invalid if CUDA runtime unloads first.
    //
    // Destruction order (reverse of typical dependency):
    // 1. AdaptiveCpp (depends on CUDA/ROCm runtimes)
    // 2. OneAPI (independent SYCL runtime)
    // 3. Vulkan, Metal, WebGPU (independent GPU APIs)
    // 4. ROCm (AMD GPU runtime)
    // 5. CUDA (NVIDIA GPU runtime)
    // 6. CPU (always safe, no GPU dependencies)

    // Clear device type mapping first
    device_to_backend_.clear();

    // Destroy backends in specific order to avoid runtime cleanup races
    const std::vector<std::string> destruction_order = {
        "adaptivecpp",  // Must be destroyed first - uses CUDA/ROCm internally
        "oneapi",       // Independent SYCL runtime
        "vulkan",       // Independent GPU API
        "metal",        // Independent GPU API
        "webgpu",       // Independent GPU API
        "rocm",         // AMD runtime (AdaptiveCpp may use this)
        "cuda",         // NVIDIA runtime (AdaptiveCpp may use this)
        "cpu"           // Always safe last
    };

    for (const auto& name : destruction_order) {
        auto it = backends_.find(name);
        if (it != backends_.end()) {
            // Explicitly reset the unique_ptr to destroy the backend
            it->second.reset();
            backends_.erase(it);
        }
    }

    // Destroy any remaining backends not in the explicit order
    backends_.clear();

    // Then unload libraries
    for (auto handle : loaded_libraries_) {
        unload_library(handle);
    }
    loaded_libraries_.clear();
}

auto BackendLoader::load_backend(const std::filesystem::path& library_path)
    -> std::expected<std::unique_ptr<Backend>, std::string> {

    auto handle = load_library(library_path);
    if (!handle) {
        return std::unexpected("Failed to load library: " + library_path.string());
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
    } else if (backend_name == "metal") {
        device_type = Device::Type::Metal;
    } else if (backend_name == "webgpu") {
        device_type = Device::Type::WebGPU;
    } else if (backend_name == "adaptivecpp") {
        device_type = Device::Type::AdaptiveCpp;
        // Register atexit handler to cleanup AdaptiveCpp before CUDA driver unloads.
        // This must be done here (when AdaptiveCpp is registered) so that the atexit
        // handler runs AFTER CUDA's atexit cleanup but BEFORE static destructors.
        // See: https://github.com/AdaptiveCpp/AdaptiveCpp/issues/817
        if (!g_atexit_registered) {
            std::atexit(cleanup_adaptivecpp_before_cuda);
            g_atexit_registered = true;
        }
    } else {
        device_type = Device::Type::CPU; // Default fallback
    }

    // Store backend pointer before moving
    auto* backend_ptr = backend.get();

    // Register by name
    backends_[backend_name] = std::move(backend);

    // Register by device type
    device_to_backend_[device_type] = backend_ptr;
}

auto BackendLoader::get_backend(std::string_view name) -> Backend* {
    auto it = backends_.find(std::string(name));
    return it != backends_.end() ? it->second.get() : nullptr;
}

auto BackendLoader::get_backend(Device::Type type) -> Backend* {
    auto it = device_to_backend_.find(type);
    return it != device_to_backend_.end() ? it->second : nullptr;
}

auto BackendLoader::has_backend(std::string_view name) const -> bool {
    return backends_.contains(std::string(name));
}

auto BackendLoader::available_backends() const -> std::vector<std::string> {
    std::vector<std::string> names;
    names.reserve(backends_.size());
    for (const auto& [name, _] : backends_) {
        names.push_back(name);
    }
    return names;
}

auto BackendLoader::unload_backend(std::string_view name) -> bool {
    return backends_.erase(std::string(name)) > 0;
}

auto BackendLoader::load_library(const std::filesystem::path& path) -> LibHandle {
    #ifdef _WIN32
        return LoadLibraryA(path.string().c_str());
    #else
        return dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
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

// Global registry
auto backend_registry() -> BackendLoader& {
    static BackendLoader registry;
    return registry;
}

} // namespace tenzor
