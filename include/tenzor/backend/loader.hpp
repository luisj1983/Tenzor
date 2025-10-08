#pragma once

#include <memory>
#include <filesystem>
#include <expected>
#include <string>
#include <vector>
#include <unordered_map>
#include "backend.hpp"

namespace tenzor {

// Backend loader for dynamic loading
class BackendLoader {
public:
    BackendLoader() = default;
    ~BackendLoader();

    BackendLoader(const BackendLoader&) = delete;
    BackendLoader& operator=(const BackendLoader&) = delete;

    // Load backend from shared library
    auto load_backend(const std::filesystem::path& library_path)
        -> std::expected<std::unique_ptr<Backend>, std::string>;

    // Register backend directly
    auto register_backend(std::string_view name,
                         std::unique_ptr<Backend> backend) -> void;

    // Get backend by name or device type
    auto get_backend(std::string_view name) -> Backend*;
    auto get_backend(Device::Type type) -> Backend*;

    // Query backends
    auto has_backend(std::string_view name) const -> bool;
    auto available_backends() const -> std::vector<std::string>;

    // Unload backend
    auto unload_backend(std::string_view name) -> bool;

private:
    std::unordered_map<std::string, std::unique_ptr<Backend>> backends_;
    std::unordered_map<Device::Type, Backend*> device_to_backend_;

    // Platform-specific library handle
#ifdef _WIN32
    using LibHandle = void*; // HMODULE
#else
    using LibHandle = void*;
#endif

    std::vector<LibHandle> loaded_libraries_;

    auto load_library(const std::filesystem::path& path) -> LibHandle;
    auto unload_library(LibHandle handle) -> void;
    auto get_symbol(LibHandle handle, const char* name) -> void*;
};

// Global backend registry (thread-safe singleton)
auto backend_registry() -> BackendLoader&;

} // namespace tenzor
