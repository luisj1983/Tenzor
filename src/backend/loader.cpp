#include "tenzor/backend/loader.hpp"
#include <mutex>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace tenzor {

BackendLoader::~BackendLoader() {
    // Unload all libraries
    for (auto handle : loaded_libraries_) {
        unload_library(handle);
    }
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
    backends_[std::string(name)] = std::move(backend);
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

// Global registry
auto backend_registry() -> BackendLoader& {
    static BackendLoader registry;
    return registry;
}

} // namespace tenzor
