#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/backend/cpu_caching_allocator.hpp"
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif
#include "cpu_thread_config.hpp"

#ifndef _WIN32
#include <dlfcn.h>
#endif

// ============================================================================
// Pin TBB malloc to prevent static destruction crash (Static Constructor)
// ============================================================================
// The CPU backend transitively links libtbb.so via oneDNN. During process
// exit, libtbb's __TBB_InitOnce destructor calls cache_aligned_deallocate
// which forwards to libtbbmalloc's scalable_free. If libtbbmalloc's static
// destructors run first, the function pointer is NULL → segfault.
//
// Fix: re-open tbbmalloc with RTLD_NODELETE at load time. This prevents
// its static destructors from ever running, so scalable_free remains valid
// when libtbb's destructor calls it. The OS reclaims all memory at exit.
//
// __attribute__((constructor)) is a GCC/Clang extension; MSVC compiles this
// file unguarded only as a stub (TBB is not used on Windows), so we wrap.
#if !defined(_WIN32) && (defined(__GNUC__) || defined(__clang__))
__attribute__((constructor(101)))
static void pin_tbb_libs() {
    // Pin all TBB libraries to prevent their static destructors from running
    // during __cxa_finalize. Without this, libtbb's __TBB_InitOnce destructor
    // calls cache_aligned_deallocate through a scalable_free weak symbol that
    // becomes NULL after tbbmalloc cleanup.
    // Primary (release) libs come first; the *_debug variants are legitimately
    // absent on most systems, so their dlopen failure is not noteworthy. We only
    // warn if NEITHER primary lib could be pinned — in that case the exit-time
    // scalable_free segfault this pin exists to prevent can still occur.
    struct LibSpec { const char* name; bool primary; };
    const LibSpec libs[] = {
        {"libtbbmalloc.so.2", true},  {"libtbbmalloc_debug.so.2", false},
        {"libtbb.so.12", true},       {"libtbb_debug.so.12", false},
    };
    bool any_primary_pinned = false;
    bool any_primary_attempted = false;
    for (const auto& lib : libs) {
        void* h = dlopen(lib.name, RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
        if (lib.primary) {
            any_primary_attempted = true;
            if (h != nullptr) {
                any_primary_pinned = true;
            }
        }
    }
    if (any_primary_attempted && !any_primary_pinned) {
        // dlopen returned NULL for every primary TBB lib. Surface it: if TBB is
        // actually the active allocator, the exit-time segfault is not prevented.
        std::fprintf(stderr,
            "[tenzor] warning: could not pin any TBB runtime library "
            "(libtbbmalloc.so.2 / libtbb.so.12); if TBB is the active allocator, "
            "an exit-time segfault in scalable_free may occur.\n");
    }
}
#endif

// Audit-11 / Stream 17: the previous `configure_openmp_early` static
// constructor was removed. There is now a single source of truth for OMP
// thread configuration — tenzor::backends::cpu::configure_omp_threads() —
// invoked from create_backend(). The competing static-constructor default
// has been deleted so behaviour is deterministic regardless of load order.

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace tenzor {

// Audit-12 / Stream 25: the long block of `namespace cpu { auto *_kernel(...); }`
// forward declarations that lived here was dead code. The CPU kernel
// registrations have lived in `cpu_kernel_registry.cpp` for many releases,
// and that file declares the kernels it needs locally. Nothing in this file
// calls any of those kernels directly (the only CPU symbol referenced from
// `CPUBackend` is `cpu::CPUCachingAllocator`, declared via
// `cpu_caching_allocator.hpp`). The forward declarations have been removed.

// Forward declaration — defined in kernels/math.cpp
void mkl_cleanup();

class CPUBackend : public Backend {
public:
    ~CPUBackend() override {
        // Release MKL thread buffers before the backend library is unloaded.
        mkl_cleanup();
    }

    auto name() const -> std::string_view override {
        return "cpu";
    }

    auto device_count() const -> int32_t override {
        return 1;
    }

    auto is_available() const -> bool override {
        return true;
    }

    auto get_device_info(int32_t device_id) const -> DeviceInfo override {
        if (device_id != 0) {
            throw std::out_of_range("CPU backend only has device 0");
        }

        DeviceInfo info;
        info.name = "CPU";
        info.vendor = "System";

        // Report the configured OMP thread count (honors TENZOR_NUM_THREADS /
        // OMP_NUM_THREADS), not raw hardware_concurrency, so compute_units
        // matches the parallelism the backend actually uses. get_configured_threads()
        // returns >=1 even before configure_omp_threads() has run.
        info.compute_units = tenzor::backends::cpu::get_configured_threads();

        // CPU always supports FP64 and usually FP16 via software
        info.supports_fp64 = true;
        info.supports_fp16 = true;
        info.is_integrated = true;

        // Try to get system memory info
        #ifdef __linux__
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                size_t kb = 0;
                if (sscanf(line.c_str(), "MemTotal: %zu kB", &kb) == 1) {
                    info.total_memory = kb * 1024;
                }
            } else if (line.find("MemAvailable:") == 0) {
                size_t kb = 0;
                if (sscanf(line.c_str(), "MemAvailable: %zu kB", &kb) == 1) {
                    info.available_memory = kb * 1024;
                }
            }
        }
        #elif defined(_WIN32)
        MEMORYSTATUSEX memStatus;
        memStatus.dwLength = sizeof(memStatus);
        if (GlobalMemoryStatusEx(&memStatus)) {
            info.total_memory = memStatus.ullTotalPhys;
            info.available_memory = memStatus.ullAvailPhys;
        }
        #elif defined(__APPLE__)
        int64_t memsize;
        size_t len = sizeof(memsize);
        if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0) {
            info.total_memory = static_cast<size_t>(memsize);
        }
        #endif

        return info;
    }

    auto allocate(size_t bytes, [[maybe_unused]] int32_t device_id) -> void* override {
        // Use caching allocator for efficient memory reuse
        return cpu::CPUCachingAllocator::instance().allocate(bytes);
    }

    auto deallocate(void* ptr) -> void override {
        // Return to cache for reuse instead of freeing immediately
        cpu::CPUCachingAllocator::instance().deallocate(ptr);
    }

    auto copy(void* dst, const void* src, size_t bytes, [[maybe_unused]] CopyKind kind) -> void override {
        std::memcpy(dst, src, bytes);
    }

    // Audit item F.7 — Backend::memset's default throws.  CPU has a
    // trivial implementation that just forwards to std::memset; the GPU
    // backends override with cudaMemset / hipMemset / etc.
    auto memset(void* ptr, int value, size_t bytes,
                [[maybe_unused]] int32_t device_id) -> void override {
        std::memset(ptr, value, bytes);
    }

    auto synchronize([[maybe_unused]] int32_t device_id) -> void override {
        // CPU is always synchronized
    }

    auto create_stream([[maybe_unused]] int32_t device_id) -> StreamHandle override {
        return nullptr;
    }

    auto destroy_stream([[maybe_unused]] StreamHandle stream) -> void override {
        // No-op for CPU
    }

    auto synchronize_stream([[maybe_unused]] StreamHandle stream) -> void override {
        // No-op for CPU
    }

    // ---- Event API (Stream 17 audit-11) -----------------------------------
    // CPU "events" are wall-clock timestamps captured at record_event().
    // EventHandle is `void*`; we heap-allocate a steady_clock::time_point so
    // the handle remains valid across record/wait/elapsed calls. record_event
    // is idempotent — re-recording overwrites the timestamp, matching the
    // CUDA/HIP semantics.
    using CPUEventClock = std::chrono::steady_clock;

    auto create_event([[maybe_unused]] int32_t device_id,
                      [[maybe_unused]] bool enable_timing = true) -> EventHandle override {
        // Allocate a time_point in the "not yet recorded" state (epoch).
        return new CPUEventClock::time_point{};
    }

    auto destroy_event(EventHandle event) -> void override {
        delete static_cast<CPUEventClock::time_point*>(event);
    }

    auto record_event(EventHandle event,
                      [[maybe_unused]] StreamHandle stream = nullptr) -> void override {
        if (event == nullptr) {
            return;
        }
        *static_cast<CPUEventClock::time_point*>(event) = CPUEventClock::now();
    }

    auto wait_event([[maybe_unused]] EventHandle event,
                    [[maybe_unused]] StreamHandle stream = nullptr) -> void override {
        // CPU work is synchronous — by the time control returns from a kernel,
        // the work is done. Nothing to wait for.
    }

    auto event_elapsed_ms(EventHandle start_event,
                          EventHandle end_event) -> float override {
        if (start_event == nullptr || end_event == nullptr) {
            return 0.0f;
        }
        auto* start = static_cast<CPUEventClock::time_point*>(start_event);
        auto* end   = static_cast<CPUEventClock::time_point*>(end_event);
        const auto delta = *end - *start;
        // duration<float, milli> yields the elapsed time in fractional ms.
        return std::chrono::duration<float, std::milli>(delta).count();
    }

    auto synchronize_event([[maybe_unused]] EventHandle event) -> void override {
        // CPU work is synchronous: a recorded event is, by construction,
        // already "complete" the moment it returns from record_event.
    }
};

// Forward declaration of kernel registration function
void register_cpu_kernels(BackendDispatchTable& table);

// Export factory function
extern "C" {
    Backend* create_backend() {
        // Single source of truth for OMP thread count: idempotent, once_flag guarded.
        tenzor::backends::cpu::configure_omp_threads();
        return new CPUBackend();
    }

    // Export kernel registration function for dispatch table initialization
    void register_kernels(BackendDispatchTable* table) {
        if (table) {
            tenzor::register_cpu_kernels(*table);
        }
    }
}

} // namespace tenzor
