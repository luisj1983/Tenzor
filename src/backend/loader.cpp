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
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
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
        "mps",          // Apple Metal (independent)
        "oneapi",       // Independent SYCL runtime
        "vulkan",       // Independent GPU API
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

auto BackendLoader::load_backend(const std::filesystem::path& library_path,
                                 bool skip_probe)
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

#ifndef _WIN32
  if (!skip_probe) {
    // Probe: fork a child process to test if the factory hangs.
    // Some backend constructors (ROCm hipGetDeviceCount, OneAPI sycl::platform::get_platforms)
    // can enter uninterruptible kernel sleep (D state) on misconfigured/unsupported GPU drivers.
    // A hung thread can't be killed, making the entire process unkillable.
    // Fork-based probing isolates this risk to a disposable child process.
    //
    // Results are cached in /tmp/tenzor_probe_<filename>_<mtime> to avoid
    // repeated 5s probes when many test binaries initialize concurrently.
    int probe_timeout = 5;
    if (const char* env = std::getenv("TENZOR_BACKEND_INIT_TIMEOUT")) {
        probe_timeout = std::atoi(env);
        if (probe_timeout <= 0) probe_timeout = 5;
    }

    // Build cache key from library filename + modification time
    auto lib_filename = library_path.filename().string();
    auto lib_mtime = std::filesystem::last_write_time(library_path);
    auto mtime_val = lib_mtime.time_since_epoch().count();

    // Per-user, owner-only probe-cache directory. A predictable world-writable
    // /tmp path let any local user pre-create the file to suppress the safety
    // probe ('0') or fake a failure ('1'/'2') for a healthy backend.
    std::string cache_dir;
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0]) {
        cache_dir = std::string(xdg) + "/tenzor";
    } else if (const char* home = std::getenv("HOME"); home && home[0]) {
        cache_dir = std::string(home) + "/.cache/tenzor";
    } else {
        cache_dir = "/tmp/tenzor-probe-" + std::to_string(getuid());
    }
    {
        std::error_code mkec;
        std::filesystem::create_directories(cache_dir, mkec);
        ::chmod(cache_dir.c_str(), 0700);  // best-effort owner-only
    }
    std::string cache_file = cache_dir + "/probe_" + lib_filename + "_" +
                             std::to_string(mtime_val);

    // Check cache first
    int cached_result = -1;  // -1 = no cache, 0 = ok, 1 = failed, 2 = hung
    {
        // O_NOFOLLOW: refuse to follow a symlink another user may have planted.
        int cache_fd = open(cache_file.c_str(), O_RDONLY | O_NOFOLLOW);
        if (cache_fd >= 0) {
            struct stat st{};
            // Only trust a regular file owned by us that is not group/world
            // writable — otherwise treat as no-cache and re-probe.
            if (fstat(cache_fd, &st) == 0 && S_ISREG(st.st_mode) &&
                st.st_uid == getuid() &&
                (st.st_mode & (S_IWGRP | S_IWOTH)) == 0) {
                char buf[2] = {};
                if (read(cache_fd, buf, 1) == 1) {
                    cached_result = buf[0] - '0';
                }
            }
            close(cache_fd);
        }
    }

    if (cached_result == 1) {
        unload_library(handle);
        return std::unexpected("Backend probe failed (cached)");
    }
    if (cached_result == 2) {
        unload_library(handle);
        return std::unexpected("Backend initialization hangs (cached)");
    }

    if (cached_result != 0) {
        // No cache — run fork probe.
        //
        // fork() in a multithreaded process is an async-signal-safety hazard:
        // only the calling thread survives in the child, and any mutex another
        // thread held at the fork instant (notably the libc/TBB allocator lock)
        // remains locked forever in the child. The child then calls factory(),
        // which allocates/locks, and would deadlock — the parent would SIGKILL
        // it after probe_timeout and mis-cache a healthy backend as "hung".
        //
        // Serialize all probing through a process-wide mutex so that no *other*
        // tenzor thread is executing the probe path (and thus mid-allocation
        // under an internal lock) while this thread forks. This directly
        // addresses the documented concern of many test binaries / threads
        // initializing backends concurrently.
        static std::mutex probe_fork_mutex;
        std::lock_guard<std::mutex> probe_lock(probe_fork_mutex);

        pid_t pid = fork();
        if (pid == 0) {
            // Child: try calling the factory. Exit 0 on success, 1 on failure.
            try {
                Backend* test_backend = factory();
                bool ok = test_backend != nullptr;
                delete test_backend;
                _exit(ok ? 0 : 1);
            } catch (...) {
                _exit(1);
            }
        } else if (pid > 0) {
            // Parent: wait for child with timeout
            int status = 0;
            bool child_exited = false;
            for (int i = 0; i < probe_timeout * 20; ++i) {
                pid_t result = waitpid(pid, &status, WNOHANG);
                if (result == pid) {
                    child_exited = true;
                    break;
                }
                usleep(50000);  // 50ms
            }

            // Write cache result
            auto write_cache = [&](char result) {
                int fd = open(cache_file.c_str(),
                              O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
                if (fd >= 0) {
                    [[maybe_unused]] auto _ = write(fd, &result, 1);
                    close(fd);
                }
            };

            if (!child_exited) {
                kill(pid, SIGKILL);
                // Reap the killed child with a bounded polling loop instead of a
                // single non-blocking waitpid. A child stuck in uninterruptible
                // (D-state) kernel sleep — the exact scenario this probe targets,
                // e.g. a wedged hipGetDeviceCount / sycl::platform::get_platforms —
                // cannot process SIGKILL until it leaves D state, so one WNOHANG
                // after 100ms returns 0 and leaks the child as a future zombie.
                // Poll for up to ~2s so the common case (child leaves D state
                // shortly after the wedged driver call returns) is reaped here.
                bool reaped = false;
                for (int i = 0; i < 40; ++i) {        // 40 * 50ms = 2s budget
                    pid_t r = waitpid(pid, &status, WNOHANG);
                    if (r == pid) {
                        reaped = true;
                        break;
                    }
                    if (r < 0) {
                        // ECHILD: already reaped (e.g. by a SIGCHLD handler).
                        reaped = true;
                        break;
                    }
                    usleep(50000);  // 50ms
                }
                // If still not reaped, the child is in unkillable D state and is
                // intentionally abandoned: blocking here would defeat the whole
                // purpose of fork-isolating an unkillable hang. It will become an
                // unreaped zombie only if/when the kernel eventually releases it,
                // which is an unavoidable consequence of an unkillable driver call.
                (void)reaped;
                write_cache('2');  // hung
                unload_library(handle);
                return std::unexpected(
                    "Backend initialization hung (probe timed out after " +
                    std::to_string(probe_timeout) + "s)");
            }

            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                write_cache('1');  // failed
                unload_library(handle);
                return std::unexpected("Backend probe process failed");
            }

            write_cache('0');  // success
        }
        // pid < 0: fork failed, proceed without probe
    }
  } // !skip_probe
#endif

    // Create backend (safe — probe confirmed it doesn't hang)
    std::unique_ptr<Backend> backend;
    try {
        backend.reset(factory());
    } catch (const std::exception& e) {
        unload_library(handle);
        return std::unexpected(std::string("Backend initialization failed: ") + e.what());
    }
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
    } else if (backend_name == "mps") {
        device_type = Device::Type::MPS;
    } else {
        // No silent CPU alias for unknown names — registering an unrecognised
        // backend name under Device::Type::CPU would let third-party plugins
        // shadow the real CPU backend and silently steer GPU-intent ops onto
        // the host. Force the caller to add a Device::Type for new backends.
        throw std::invalid_argument(
            "BackendLoader::register_backend: unknown backend name '" + backend_name +
            "' — no Device::Type mapping. Add the new backend to Device::Type and "
            "register_backend() before loading.");
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
