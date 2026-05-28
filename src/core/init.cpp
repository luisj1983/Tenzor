#include "tenzor/tenzor.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/utils/log.hpp"  // TENZOR_LOG_INFO / WARN / ERROR (audit I.4)
#include "tenzor/utils/logging.hpp"  // TENZOR_WARN_ONCE (S26 OCL-ICD workaround)
#include <iostream>
#include <fstream>
#include <filesystem>
#include <dlfcn.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>
#include <unistd.h>  // getpid(), for per-process OCL ICD vendor dir

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __x86_64__
#include <cpuid.h>
#endif

namespace tenzor {

// Flag to track initialization state (used by finalize() guard)
static std::atomic<bool> g_initialized{false};
static std::mutex g_init_mutex;

// Per-process OCL ICD vendor dir created by the AMD-OCL probe workaround.
// Tracked here so finalize() can clean it up. Empty if workaround was skipped.
static std::string g_ocl_icd_vendor_dir;

auto initialize() -> void {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_initialized.load(std::memory_order_acquire)) {
        return;  // Already initialized
    }

    // Pre-load TBB and tbbmalloc into the global symbol table. The CPU backend
    // is loaded with RTLD_LOCAL (to prevent MKL symbol pollution), but TBB
    // internally uses dlopen("libtbbmalloc.so.2") which only searches the
    // global scope. Without this pre-load, TBB's cache_aligned_allocate fails
    // with a null function pointer, causing segfaults in __TBB_InitOnce
    // destructors during process exit.
#ifndef _WIN32
    const char* tbb_libs[] = {
        "libtbbmalloc.so.2",
        "libtbbmalloc_debug.so.2",
        "libtbb.so.12",
        "libtbb_debug.so.12",
    };
    for (const char* lib : tbb_libs) {
        void* handle = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
        if (!handle) {
            // Debug-level log: TBB may not be needed if not using CPU backend with TBB
            TENZOR_LOG_WARN("[tenzor] Note: Could not preload {}: {}", lib, dlerror());
        }
    }
#endif

    // Configure OpenMP to use optimal thread count
    // Use physical cores (not logical/hyperthreaded) to avoid contention
    // Users can override with OMP_NUM_THREADS environment variable
#ifdef _OPENMP
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        unsigned int logical_cores = std::thread::hardware_concurrency();
        unsigned int physical_cores = logical_cores;

        // Try to detect physical cores on Linux via sysfs
        // thread_siblings_list contains comma-separated list of sibling CPUs
        // e.g., "0,12" means CPU 0 and 12 are hyperthreaded pairs (2 threads/core)
        std::ifstream siblings("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
        if (siblings.good()) {
            std::string line;
            if (std::getline(siblings, line)) {
                // Count entries in comma-separated list
                int threads_per_core = 1;
                for (char c : line) {
                    if (c == ',') threads_per_core++;
                }
                if (threads_per_core > 1) {
                    physical_cores = logical_cores / threads_per_core;
                }
            }
        }

        // Use physical cores for optimal performance (avoids HT contention)
        int num_threads = std::max(1u, physical_cores);
        omp_set_num_threads(num_threads);

        // Set environment variables for thread-aware libraries
        // These affect libraries loaded later and ensure consistent threading
        std::string threads_str = std::to_string(num_threads);

        // Set OMP_NUM_THREADS for libraries that check env var during static init
        setenv("OMP_NUM_THREADS", threads_str.c_str(), 0);

        if (std::getenv("MKL_NUM_THREADS") == nullptr) {
            setenv("MKL_NUM_THREADS", threads_str.c_str(), 0);
        }
        if (std::getenv("DNNL_CPU_RUNTIME") == nullptr) {
            setenv("DNNL_CPU_RUNTIME", "OMP", 0);
        }
    }
#endif

    TENZOR_LOG_INFO("Initializing Tenzor library v0.1.0");

    // Load CPU backend dynamically
    auto& loader = backend_registry();

    // Locate backend directory using dynamic search strategy
    auto find_backend_dir = []() -> std::filesystem::path {
        // 1. Environment variable override
        if (auto* env = std::getenv("TENZOR_BACKEND_DIR"))
            return env;

        // 2. Same directory as libtenzor.so (via dladdr)
        Dl_info info;
        if (dladdr(reinterpret_cast<void*>(&initialize), &info) && info.dli_fname) {
            auto lib_dir = std::filesystem::path(info.dli_fname).parent_path();
            if (std::filesystem::exists(lib_dir / "tenzor_backend_cpu.so"))
                return lib_dir;
        }

        // 3. Relative to executable
        std::error_code ec;
        auto exe_dir = std::filesystem::read_symlink("/proc/self/exe", ec).parent_path();
        if (!ec && std::filesystem::exists(exe_dir / "tenzor_backend_cpu.so"))
            return exe_dir;

        // 4. Current directory
        return ".";
    };

    std::filesystem::path bin_path = find_backend_dir();
    std::filesystem::path cpu_backend_path = bin_path / "tenzor_backend_cpu.so";

    TENZOR_LOG_INFO("Loading CPU backend from: {}", cpu_backend_path.string());

    auto result = loader.load_backend(cpu_backend_path);
    if (!result) {
        TENZOR_LOG_ERROR("Error: Failed to load CPU backend: {}", result.error());
        throw std::runtime_error("Failed to initialize Tenzor: CPU backend not available");
    }

    // Register the loaded backend
    auto cpu_backend_unique = std::move(result.value());
    auto* cpu_backend_ptr = cpu_backend_unique.get();

    // Register by name
    loader.register_backend(cpu_backend_ptr->name(), std::move(cpu_backend_unique));

    TENZOR_LOG_INFO("CPU backend registered: {}", cpu_backend_ptr->name());

    // Now cpu_backend_ptr points to the registered backend
    auto* cpu_backend = cpu_backend_ptr;

    // =========================================================================
    // NEW O(1) DISPATCH TABLE REGISTRATION
    // =========================================================================
    // Register CPU backend with the dispatch table registry
    DispatchTableRegistry::register_backend(Device::Type::CPU, cpu_backend);

    // Populate dispatch table with direct kernel function pointers
    auto& cpu_table = DispatchTableRegistry::get_table(Device::Type::CPU);

    // Use existing library handle from load_backend() (no second dlopen needed)
    void* handle = loader.last_library_handle();
    if (handle) {
        using RegisterFn = void(*)(BackendDispatchTable*);
        auto register_fn = reinterpret_cast<RegisterFn>(dlsym(handle, "register_kernels"));
        if (register_fn) {
            register_fn(&cpu_table);
            DispatchTableRegistry::mark_ready(Device::Type::CPU);
            auto cpu_op_count = cpu_table.op_count();
            TENZOR_LOG_INFO("CPU dispatch table initialized ({} operations registered)", cpu_op_count);
            if (cpu_op_count == 0) {
                TENZOR_LOG_WARN("Warning: CPU backend registered 0 operations");
            }
        } else {
            TENZOR_LOG_WARN("Warning: Could not find register_kernels in CPU backend");
        }
    } else {
        TENZOR_LOG_WARN("Warning: No library handle for CPU backend kernel registration");
    }

    // Try to load CUDA backend if available
    std::filesystem::path cuda_backend_path = bin_path / "tenzor_backend_cuda.so";

    if (std::filesystem::exists(cuda_backend_path)) {
        TENZOR_LOG_INFO("Loading CUDA backend from: {}", cuda_backend_path.string());

        auto cuda_result = loader.load_backend(cuda_backend_path);
        if (cuda_result) {
            auto cuda_backend_unique = std::move(cuda_result.value());
            auto* cuda_backend_ptr = cuda_backend_unique.get();

            // Check if CUDA is actually available
            if (cuda_backend_ptr->is_available()) {
                loader.register_backend(cuda_backend_ptr->name(), std::move(cuda_backend_unique));
                TENZOR_LOG_INFO("CUDA backend registered: {}", cuda_backend_ptr->name());
                TENZOR_LOG_INFO("Found {} CUDA device(s)", cuda_backend_ptr->device_count());

                auto* cuda_backend = cuda_backend_ptr;

                // =========================================================================
                // O(1) DISPATCH TABLE REGISTRATION FOR CUDA
                // =========================================================================
                DispatchTableRegistry::register_backend(Device::Type::CUDA, cuda_backend);
                auto& cuda_table = DispatchTableRegistry::get_table(Device::Type::CUDA);

                // Use existing library handle (no second dlopen needed)
                void* cuda_handle = loader.last_library_handle();
                if (cuda_handle) {
                    using RegisterFn = void(*)(BackendDispatchTable*);
                    auto register_fn = reinterpret_cast<RegisterFn>(dlsym(cuda_handle, "register_kernels"));
                    if (register_fn) {
                        register_fn(&cuda_table);
                        DispatchTableRegistry::mark_ready(Device::Type::CUDA);
                        auto cuda_op_count = cuda_table.op_count();
                        TENZOR_LOG_INFO("CUDA dispatch table initialized ({} operations registered)", cuda_op_count);
                        if (cuda_op_count == 0) {
                            TENZOR_LOG_WARN("Warning: CUDA backend registered 0 operations");
                        }
                    } else {
                        TENZOR_LOG_WARN("Warning: Could not find register_kernels in CUDA backend");
                    }
                } else {
                    TENZOR_LOG_WARN("Warning: No library handle for CUDA backend kernel registration");
                }

            } else {
                TENZOR_LOG_INFO("CUDA backend loaded but no CUDA devices available");
            }
        } else {
            TENZOR_LOG_WARN("Warning: Failed to load CUDA backend: {}", cuda_result.error());
        }
    } else {
        TENZOR_LOG_INFO("CUDA backend not found at: {}", cuda_backend_path.string());
    }

    // Try to load ROCm backend if available
    std::filesystem::path rocm_backend_path = bin_path / "tenzor_backend_rocm.so";

    // Allow disabling ROCm loading via environment variable
    bool skip_rocm = (std::getenv("TENZOR_DISABLE_ROCM") != nullptr);
    if (skip_rocm) {
        TENZOR_LOG_INFO("ROCm backend skipped (TENZOR_DISABLE_ROCM set)");
    }

    if (!skip_rocm && std::filesystem::exists(rocm_backend_path)) {
        TENZOR_LOG_INFO("Loading ROCm backend from: {}", rocm_backend_path.string());

        auto rocm_result = loader.load_backend(rocm_backend_path);
        if (rocm_result) {
            auto rocm_backend_unique = std::move(rocm_result.value());
            auto* rocm_backend_ptr = rocm_backend_unique.get();

            // Check if ROCm is actually available
            if (rocm_backend_ptr->is_available()) {
                loader.register_backend(rocm_backend_ptr->name(), std::move(rocm_backend_unique));
                TENZOR_LOG_INFO("ROCm backend registered: {}", rocm_backend_ptr->name());
                TENZOR_LOG_INFO("Found {} ROCm device(s)", rocm_backend_ptr->device_count());

                auto* rocm_backend = rocm_backend_ptr;

                // =========================================================================
                // O(1) DISPATCH TABLE REGISTRATION FOR ROCm
                // =========================================================================
                DispatchTableRegistry::register_backend(Device::Type::ROCm, rocm_backend);
                auto& rocm_table = DispatchTableRegistry::get_table(Device::Type::ROCm);

                // Use existing library handle (no second dlopen needed)
                void* rocm_handle = loader.last_library_handle();
                if (rocm_handle) {
                    using RegisterFn = void(*)(BackendDispatchTable*);
                    auto register_fn = reinterpret_cast<RegisterFn>(dlsym(rocm_handle, "register_kernels"));
                    if (register_fn) {
                        register_fn(&rocm_table);
                        DispatchTableRegistry::mark_ready(Device::Type::ROCm);
                        auto rocm_op_count = rocm_table.op_count();
                        TENZOR_LOG_INFO("ROCm dispatch table initialized ({} operations registered)", rocm_op_count);
                        if (rocm_op_count == 0) {
                            TENZOR_LOG_WARN("Warning: ROCm backend registered 0 operations");
                        }
                    } else {
                        TENZOR_LOG_WARN("Warning: Could not find register_kernels in ROCm backend");
                    }
                } else {
                    TENZOR_LOG_WARN("Warning: No library handle for ROCm backend kernel registration");
                }

            } else {
                TENZOR_LOG_INFO("ROCm backend loaded but no ROCm devices available");
            }
        } else {
            TENZOR_LOG_WARN("Warning: Failed to load ROCm backend: {}", rocm_result.error());
        }
    } else if (!skip_rocm) {
        TENZOR_LOG_INFO("ROCm backend not found at: {}", rocm_backend_path.string());
    }

    // Try to load OneAPI backend if available
    std::filesystem::path oneapi_backend_path = bin_path / "tenzor_backend_oneapi.so";

    // SYCL's platform::get_platforms() probes ALL OpenCL ICDs, including AMD's
    // which can hang if the ROCm/HSA runtime is broken (same bug as hipGetDeviceCount).
    // Create a per-process filtered ICD directory containing only Intel's OpenCL ICD
    // entry to prevent the hang.
    //
    // Discovery of libintelocl.so (in priority order):
    //   1. TENZOR_OCL_ICD_PATH         — explicit user override, full path to libintelocl.so
    //   2. INTEL_OPENCL_ICD_PATH       — Intel-conventional env var
    //   3. ${ONEAPI_ROOT}/compiler/<version>/lib/libintelocl.so (newest version)
    //   4. /opt/intel/oneapi/compiler/<version>/lib/libintelocl.so (newest version,
    //      skipping the `latest` symlink so we resolve to a real versioned dir)
    //
    // Failure paths (all silent skip + TENZOR_WARN_ONCE):
    //   - No libintelocl.so found in any of the locations above
    //   - TENZOR_OCL_ICD_PATH / INTEL_OPENCL_ICD_PATH points at a nonexistent file
    //   - Per-process temp directory cannot be created or written
    //
    // The workaround is a no-op when OCL_ICD_VENDORS is already set by the user.
    // Upstream bug tracking: the underlying AMD-OCL ICD probe hang has no public
    // fix yet; once it's resolved we can drop this entirely.
    bool oneapi_skip_probe = false;
    if (!std::getenv("OCL_ICD_VENDORS")) {
        auto find_intel_ocl_icd = []() -> std::filesystem::path {
            namespace fs = std::filesystem;
            std::error_code ec;

            // Priority 1 & 2: explicit env var overrides.
            for (const char* var : {"TENZOR_OCL_ICD_PATH", "INTEL_OPENCL_ICD_PATH"}) {
                const char* val = std::getenv(var);
                if (val && *val) {
                    fs::path p(val);
                    if (fs::exists(p, ec) && !ec) {
                        return p;
                    }
                    TENZOR_WARN_ONCE(
                        "tenzor: OpenCL ICD override env var points at nonexistent file; "
                        "skipping AMD-OCL probe workaround. See src/core/init.cpp.");
                    return {};
                }
            }

            // Build candidate compiler root directories.
            std::vector<fs::path> roots;
            if (const char* oneapi_root = std::getenv("ONEAPI_ROOT")) {
                if (*oneapi_root) {
                    roots.emplace_back(fs::path(oneapi_root) / "compiler");
                }
            }
            roots.emplace_back("/opt/intel/oneapi/compiler");

            // Pick the newest *versioned* subdirectory under each root.
            // We intentionally skip the "latest" symlink so the resolution is
            // stable across oneAPI upgrades and doesn't depend on packaging.
            fs::path best;
            for (const auto& root : roots) {
                if (!fs::is_directory(root, ec) || ec) {
                    continue;
                }
                fs::path best_in_root;
                for (auto it = fs::directory_iterator(root, ec);
                     !ec && it != fs::directory_iterator();
                     it.increment(ec)) {
                    if (!it->is_directory(ec) || ec) {
                        continue;
                    }
                    const auto name = it->path().filename().string();
                    if (name == "latest" || name.empty() || name[0] == '.') {
                        continue;
                    }
                    fs::path candidate = it->path() / "lib" / "libintelocl.so";
                    if (!fs::exists(candidate, ec) || ec) {
                        continue;
                    }
                    if (best_in_root.empty() || name > best_in_root.filename().string()) {
                        best_in_root = it->path();
                    }
                }
                if (!best_in_root.empty()) {
                    best = best_in_root / "lib" / "libintelocl.so";
                    break;
                }
            }
            return best;
        };

        std::filesystem::path intel_icd_so = find_intel_ocl_icd();
        if (intel_icd_so.empty()) {
            // No Intel OpenCL ICD on this system — workaround is impossible.
            // Silently skip; oneAPI backend will probably also fail to find a
            // device, which is fine on non-Intel systems.
            TENZOR_WARN_ONCE(
                "tenzor: libintelocl.so not found via TENZOR_OCL_ICD_PATH, "
                "INTEL_OPENCL_ICD_PATH, ONEAPI_ROOT, or /opt/intel/oneapi; "
                "skipping AMD-OCL ICD probe workaround. Set TENZOR_OCL_ICD_PATH "
                "if your Intel OpenCL ICD lives in a non-standard location.");
        } else {
            // Per-process temp dir keeps users from clobbering each other and
            // sidesteps stale content from prior runs.
            std::error_code ec;
            const char* tmpdir_env = std::getenv("TMPDIR");
            std::filesystem::path tmp_root =
                (tmpdir_env && *tmpdir_env) ? std::filesystem::path(tmpdir_env)
                                            : std::filesystem::path("/tmp");
            std::filesystem::path icd_dir =
                tmp_root / ("tenzor_ocl_vendors_" + std::to_string(::getpid()));
            std::filesystem::create_directories(icd_dir, ec);
            if (ec || !std::filesystem::is_directory(icd_dir)) {
                TENZOR_WARN_ONCE(
                    "tenzor: could not create per-process OCL ICD vendor dir "
                    "(TMPDIR not writable?); skipping AMD-OCL probe workaround.");
            } else {
                std::filesystem::path intel_icd = icd_dir / "intel64.icd";
                std::ofstream icd_file(intel_icd);
                if (!icd_file) {
                    TENZOR_WARN_ONCE(
                        "tenzor: could not write OCL ICD vendor file in TMPDIR; "
                        "skipping AMD-OCL probe workaround.");
                } else {
                    icd_file << intel_icd_so.string() << "\n";
                    icd_file.close();
                    if (icd_file.fail()) {
                        TENZOR_WARN_ONCE(
                            "tenzor: failed flushing OCL ICD vendor file; "
                            "skipping AMD-OCL probe workaround.");
                    } else {
                        setenv("OCL_ICD_VENDORS", icd_dir.c_str(), 0);
                        // Record path so finalize() can clean it up.
                        g_ocl_icd_vendor_dir = icd_dir.string();
                        oneapi_skip_probe = true;
                    }
                }
            }
        }
    }
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "*:cpu", 0);
    }

    // Configure Intel OpenCL CPU runtime target architecture BEFORE loading
    // the OneAPI backend.  The runtime reads CL_CONFIG_CPU_TARGET_ARCH during
    // static initialisation of libintelocl.so, which happens inside dlopen().
    // Setting it after dlopen() is too late.
#ifdef __x86_64__
    if (!std::getenv("CL_CONFIG_CPU_TARGET_ARCH")) {
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        __cpuid(0, eax, ebx, ecx, edx);
        if (eax >= 7) {
            __cpuid(1, eax, ebx, ecx, edx);
            bool has_avx  = (ecx >> 28) & 1;
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
            bool has_avx2    = (ebx >> 5) & 1;
            bool has_avx512f = (ebx >> 16) & 1;
            const char* arch = "corei7";
            if (has_avx512f)     arch = "skx";
            else if (has_avx2)   arch = "core-avx2";
            else if (has_avx)    arch = "corei7-avx";
            setenv("CL_CONFIG_CPU_TARGET_ARCH", arch, 1);
            setenv("CL_CONFIG_CPU_FORCE_TARGET_ARCH", arch, 1);
        }
    }
#endif

    if (std::filesystem::exists(oneapi_backend_path)) {
        TENZOR_LOG_INFO("Loading OneAPI backend from: {}", oneapi_backend_path.string());

        auto oneapi_result = loader.load_backend(oneapi_backend_path, oneapi_skip_probe);
        if (oneapi_result) {
            auto oneapi_backend_unique = std::move(oneapi_result.value());
            auto* oneapi_backend_ptr = oneapi_backend_unique.get();

            // Check if OneAPI is actually available
            if (oneapi_backend_ptr->is_available()) {
                loader.register_backend(oneapi_backend_ptr->name(), std::move(oneapi_backend_unique));
                TENZOR_LOG_INFO("OneAPI backend registered: {}", oneapi_backend_ptr->name());
                TENZOR_LOG_INFO("Found {} OneAPI device(s)", oneapi_backend_ptr->device_count());

                auto* oneapi_backend = oneapi_backend_ptr;

                // =========================================================================
                // O(1) DISPATCH TABLE REGISTRATION FOR ONEAPI
                // =========================================================================
                DispatchTableRegistry::register_backend(Device::Type::OneAPI, oneapi_backend);
                auto& oneapi_table = DispatchTableRegistry::get_table(Device::Type::OneAPI);

                // Use existing library handle (no second dlopen needed)
                void* oneapi_handle = loader.last_library_handle();
                if (oneapi_handle) {
                    using RegisterFn = void(*)(BackendDispatchTable*);
                    auto register_fn = reinterpret_cast<RegisterFn>(dlsym(oneapi_handle, "register_kernels"));
                    if (register_fn) {
                        register_fn(&oneapi_table);
                        DispatchTableRegistry::mark_ready(Device::Type::OneAPI);
                        auto oneapi_op_count = oneapi_table.op_count();
                        TENZOR_LOG_INFO("OneAPI dispatch table initialized ({} operations registered)", oneapi_op_count);
                        if (oneapi_op_count == 0) {
                            TENZOR_LOG_WARN("Warning: OneAPI backend registered 0 operations");
                        }
                    } else {
                        TENZOR_LOG_WARN("Warning: Could not find register_kernels in OneAPI backend");
                    }
                } else {
                    TENZOR_LOG_WARN("Warning: No library handle for OneAPI backend kernel registration");
                }

            } else {
                TENZOR_LOG_INFO("OneAPI backend loaded but no OneAPI devices available");
            }
        } else {
            TENZOR_LOG_WARN("Warning: Failed to load OneAPI backend: {}", oneapi_result.error());
        }
    } else {
        TENZOR_LOG_INFO("OneAPI backend not found at: {}", oneapi_backend_path.string());
    }

    // Try to load Vulkan backend if available
    std::filesystem::path vulkan_backend_path = bin_path / "tenzor_backend_vulkan.so";

    if (std::filesystem::exists(vulkan_backend_path)) {
        TENZOR_LOG_INFO("Loading Vulkan backend from: {}", vulkan_backend_path.string());

        auto vulkan_result = loader.load_backend(vulkan_backend_path);
        if (vulkan_result) {
            auto vulkan_backend_unique = std::move(vulkan_result.value());
            auto* vulkan_backend_ptr = vulkan_backend_unique.get();

            // Check if Vulkan is actually available
            if (vulkan_backend_ptr->is_available()) {
                loader.register_backend(vulkan_backend_ptr->name(), std::move(vulkan_backend_unique));
                TENZOR_LOG_INFO("Vulkan backend registered: {}", vulkan_backend_ptr->name());
                TENZOR_LOG_INFO("Found {} Vulkan device(s)", vulkan_backend_ptr->device_count());

                auto* vulkan_backend = vulkan_backend_ptr;

                // =========================================================================
                // O(1) DISPATCH TABLE REGISTRATION FOR VULKAN
                // =========================================================================
                DispatchTableRegistry::register_backend(Device::Type::Vulkan, vulkan_backend);
                auto& vulkan_table = DispatchTableRegistry::get_table(Device::Type::Vulkan);

                // Use existing library handle (no second dlopen needed)
                void* vulkan_handle = loader.last_library_handle();
                if (vulkan_handle) {
                    using RegisterFn = void(*)(BackendDispatchTable*);
                    auto register_fn = reinterpret_cast<RegisterFn>(dlsym(vulkan_handle, "register_kernels"));
                    if (register_fn) {
                        register_fn(&vulkan_table);
                        DispatchTableRegistry::mark_ready(Device::Type::Vulkan);
                        auto vulkan_op_count = vulkan_table.op_count();
                        TENZOR_LOG_INFO("Vulkan dispatch table initialized ({} operations registered)", vulkan_op_count);
                        if (vulkan_op_count == 0) {
                            TENZOR_LOG_WARN("Warning: Vulkan backend registered 0 operations");
                        }


                    } else {
                        TENZOR_LOG_WARN("Warning: Could not find register_kernels in Vulkan backend");
                    }
                } else {
                    TENZOR_LOG_WARN("Warning: No library handle for Vulkan backend kernel registration");
                }

            } else {
                TENZOR_LOG_INFO("Vulkan backend loaded but no Vulkan devices available");
            }
        } else {
            TENZOR_LOG_WARN("Warning: Failed to load Vulkan backend: {}", vulkan_result.error());
        }
    } else {
        TENZOR_LOG_INFO("Vulkan backend not found at: {}", vulkan_backend_path.string());
    }

    // Now that every backend that intends to register has done so, scan
    // the dispatch tables and report any OpId with zero coverage. This
    // converts late "operation not supported" exceptions into a single
    // startup-time report; in strict mode (TENZOR_DISPATCH_STRICT=1 env
    // var) it throws so CI can catch coverage regressions.
    DispatchTableRegistry::validate_coverage(/*strict=*/false);

    TENZOR_LOG_INFO("Tenzor initialization complete");

    g_initialized.store(true, std::memory_order_release);

    // Register finalize() to run before static destructors.
    // atexit handlers registered after a static's construction run before
    // that static's destructor, ensuring proper ordered cleanup.
    std::atexit([]() { finalize(); });

}

auto finalize() -> void {
    if (!g_initialized.load(std::memory_order_acquire)) {
        return;
    }

    // 1. Clear dispatch tables — removes all function pointers into backend .so files
    DispatchTableRegistry::clear();

    // 2. Ordered backend shutdown — destroys backends, dlcloses libraries
    backend_registry().shutdown();

    // 3. Best-effort cleanup of the per-process OCL ICD vendor dir created by
    //    the AMD-OCL probe workaround. Failures are silent — /tmp gets reaped
    //    on reboot and the path has the PID baked in so collisions are unlikely.
    if (!g_ocl_icd_vendor_dir.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(g_ocl_icd_vendor_dir, ec);
        g_ocl_icd_vendor_dir.clear();
    }

    g_initialized.store(false, std::memory_order_release);
}

} // namespace tenzor
