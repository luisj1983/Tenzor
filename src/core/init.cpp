#include "tenzor/tenzor.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <dlfcn.h>
#include <thread>
#include <atomic>
#include <mutex>

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
            std::cerr << "[tenzor] Note: Could not preload " << lib << ": " << dlerror() << std::endl;
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

    std::cout << "Initializing Tenzor library v1.0.0" << std::endl;

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

    std::cout << "Loading CPU backend from: " << cpu_backend_path << std::endl;

    auto result = loader.load_backend(cpu_backend_path);
    if (!result) {
        std::cerr << "Error: Failed to load CPU backend: " << result.error() << std::endl;
        throw std::runtime_error("Failed to initialize Tenzor: CPU backend not available");
    }

    // Register the loaded backend
    auto cpu_backend_unique = std::move(result.value());
    auto* cpu_backend_ptr = cpu_backend_unique.get();

    // Register by name
    loader.register_backend(cpu_backend_ptr->name(), std::move(cpu_backend_unique));

    std::cout << "CPU backend registered: " << cpu_backend_ptr->name() << std::endl;

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
            std::cout << "CPU dispatch table initialized (" << cpu_op_count << " operations registered)" << std::endl;
            if (cpu_op_count == 0) {
                std::cerr << "Warning: CPU backend registered 0 operations" << std::endl;
            }
        } else {
            std::cerr << "Warning: Could not find register_kernels in CPU backend" << std::endl;
        }
    } else {
        std::cerr << "Warning: No library handle for CPU backend kernel registration" << std::endl;
    }

    // Try to load CUDA backend if available
    std::filesystem::path cuda_backend_path = bin_path / "tenzor_backend_cuda.so";

    if (std::filesystem::exists(cuda_backend_path)) {
        std::cout << "Loading CUDA backend from: " << cuda_backend_path << std::endl;

        auto cuda_result = loader.load_backend(cuda_backend_path);
        if (cuda_result) {
            auto cuda_backend_unique = std::move(cuda_result.value());
            auto* cuda_backend_ptr = cuda_backend_unique.get();

            // Check if CUDA is actually available
            if (cuda_backend_ptr->is_available()) {
                loader.register_backend(cuda_backend_ptr->name(), std::move(cuda_backend_unique));
                std::cout << "CUDA backend registered: " << cuda_backend_ptr->name() << std::endl;
                std::cout << "Found " << cuda_backend_ptr->device_count() << " CUDA device(s)" << std::endl;

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
                        std::cout << "CUDA dispatch table initialized (" << cuda_op_count << " operations registered)" << std::endl;
                        if (cuda_op_count == 0) {
                            std::cerr << "Warning: CUDA backend registered 0 operations" << std::endl;
                        }
                    } else {
                        std::cerr << "Warning: Could not find register_kernels in CUDA backend" << std::endl;
                    }
                } else {
                    std::cerr << "Warning: No library handle for CUDA backend kernel registration" << std::endl;
                }

            } else {
                std::cout << "CUDA backend loaded but no CUDA devices available" << std::endl;
            }
        } else {
            std::cout << "Warning: Failed to load CUDA backend: " << cuda_result.error() << std::endl;
        }
    } else {
        std::cout << "CUDA backend not found at: " << cuda_backend_path << std::endl;
    }

    // Try to load ROCm backend if available
    std::filesystem::path rocm_backend_path = bin_path / "tenzor_backend_rocm.so";

    // Allow disabling ROCm loading via environment variable
    bool skip_rocm = (std::getenv("TENZOR_DISABLE_ROCM") != nullptr);
    if (skip_rocm) {
        std::cout << "ROCm backend skipped (TENZOR_DISABLE_ROCM set)" << std::endl;
    }

    if (!skip_rocm && std::filesystem::exists(rocm_backend_path)) {
        std::cout << "Loading ROCm backend from: " << rocm_backend_path << std::endl;

        auto rocm_result = loader.load_backend(rocm_backend_path);
        if (rocm_result) {
            auto rocm_backend_unique = std::move(rocm_result.value());
            auto* rocm_backend_ptr = rocm_backend_unique.get();

            // Check if ROCm is actually available
            if (rocm_backend_ptr->is_available()) {
                loader.register_backend(rocm_backend_ptr->name(), std::move(rocm_backend_unique));
                std::cout << "ROCm backend registered: " << rocm_backend_ptr->name() << std::endl;
                std::cout << "Found " << rocm_backend_ptr->device_count() << " ROCm device(s)" << std::endl;

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
                        std::cout << "ROCm dispatch table initialized (" << rocm_op_count << " operations registered)" << std::endl;
                        if (rocm_op_count == 0) {
                            std::cerr << "Warning: ROCm backend registered 0 operations" << std::endl;
                        }
                    } else {
                        std::cerr << "Warning: Could not find register_kernels in ROCm backend" << std::endl;
                    }
                } else {
                    std::cerr << "Warning: No library handle for ROCm backend kernel registration" << std::endl;
                }

            } else {
                std::cout << "ROCm backend loaded but no ROCm devices available" << std::endl;
            }
        } else {
            std::cout << "Warning: Failed to load ROCm backend: " << rocm_result.error() << std::endl;
        }
    } else if (!skip_rocm) {
        std::cout << "ROCm backend not found at: " << rocm_backend_path << std::endl;
    }

    // Try to load OneAPI backend if available
    std::filesystem::path oneapi_backend_path = bin_path / "tenzor_backend_oneapi.so";

    // SYCL's platform::get_platforms() probes ALL OpenCL ICDs, including AMD's
    // which can hang if the ROCm/HSA runtime is broken (same bug as hipGetDeviceCount).
    // Create a filtered ICD directory with only Intel's OpenCL to prevent the hang.
    bool oneapi_skip_probe = false;
    if (!std::getenv("OCL_ICD_VENDORS")) {
        // Create a temp directory with only Intel's ICD to avoid loading amdocl64
        std::string icd_dir = "/tmp/tenzor_ocl_vendors";
        std::filesystem::create_directories(icd_dir);
        std::string intel_icd = icd_dir + "/intel64.icd";
        if (!std::filesystem::exists(intel_icd)) {
            std::ofstream(intel_icd) << "/opt/intel/oneapi/compiler/latest/lib/libintelocl.so\n";
        }
        setenv("OCL_ICD_VENDORS", icd_dir.c_str(), 0);
        oneapi_skip_probe = true;
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
        std::cout << "Loading OneAPI backend from: " << oneapi_backend_path << std::endl;

        auto oneapi_result = loader.load_backend(oneapi_backend_path, oneapi_skip_probe);
        if (oneapi_result) {
            auto oneapi_backend_unique = std::move(oneapi_result.value());
            auto* oneapi_backend_ptr = oneapi_backend_unique.get();

            // Check if OneAPI is actually available
            if (oneapi_backend_ptr->is_available()) {
                loader.register_backend(oneapi_backend_ptr->name(), std::move(oneapi_backend_unique));
                std::cout << "OneAPI backend registered: " << oneapi_backend_ptr->name() << std::endl;
                std::cout << "Found " << oneapi_backend_ptr->device_count() << " OneAPI device(s)" << std::endl;

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
                        std::cout << "OneAPI dispatch table initialized (" << oneapi_op_count << " operations registered)" << std::endl;
                        if (oneapi_op_count == 0) {
                            std::cerr << "Warning: OneAPI backend registered 0 operations" << std::endl;
                        }
                    } else {
                        std::cerr << "Warning: Could not find register_kernels in OneAPI backend" << std::endl;
                    }
                } else {
                    std::cerr << "Warning: No library handle for OneAPI backend kernel registration" << std::endl;
                }

            } else {
                std::cout << "OneAPI backend loaded but no OneAPI devices available" << std::endl;
            }
        } else {
            std::cout << "Warning: Failed to load OneAPI backend: " << oneapi_result.error() << std::endl;
        }
    } else {
        std::cout << "OneAPI backend not found at: " << oneapi_backend_path << std::endl;
    }

    // Try to load Vulkan backend if available
    std::filesystem::path vulkan_backend_path = bin_path / "tenzor_backend_vulkan.so";

    if (std::filesystem::exists(vulkan_backend_path)) {
        std::cout << "Loading Vulkan backend from: " << vulkan_backend_path << std::endl;

        auto vulkan_result = loader.load_backend(vulkan_backend_path);
        if (vulkan_result) {
            auto vulkan_backend_unique = std::move(vulkan_result.value());
            auto* vulkan_backend_ptr = vulkan_backend_unique.get();

            // Check if Vulkan is actually available
            if (vulkan_backend_ptr->is_available()) {
                loader.register_backend(vulkan_backend_ptr->name(), std::move(vulkan_backend_unique));
                std::cout << "Vulkan backend registered: " << vulkan_backend_ptr->name() << std::endl;
                std::cout << "Found " << vulkan_backend_ptr->device_count() << " Vulkan device(s)" << std::endl;

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
                        std::cout << "Vulkan dispatch table initialized (" << vulkan_op_count << " operations registered)" << std::endl;
                        if (vulkan_op_count == 0) {
                            std::cerr << "Warning: Vulkan backend registered 0 operations" << std::endl;
                        }
                    } else {
                        std::cerr << "Warning: Could not find register_kernels in Vulkan backend" << std::endl;
                    }
                } else {
                    std::cerr << "Warning: No library handle for Vulkan backend kernel registration" << std::endl;
                }

            } else {
                std::cout << "Vulkan backend loaded but no Vulkan devices available" << std::endl;
            }
        } else {
            std::cout << "Warning: Failed to load Vulkan backend: " << vulkan_result.error() << std::endl;
        }
    } else {
        std::cout << "Vulkan backend not found at: " << vulkan_backend_path << std::endl;
    }

    std::cout << "Tenzor initialization complete" << std::endl;

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

    g_initialized.store(false, std::memory_order_release);
}

} // namespace tenzor
