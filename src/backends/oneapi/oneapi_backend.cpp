#include "oneapi_internal.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/backend/oneapi_caching_allocator.hpp"
#include "tenzor/utils/logging.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <typeinfo>

#ifdef __x86_64__
#include <cpuid.h>
#endif

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#endif

namespace tenzor {

// NOTE: The per-kernel forward declarations that previously lived here were
// removed. They were never called from this translation unit and duplicated
// (and had already drifted from) the authoritative declarations in
// oneapi_kernel_registry.cpp. Anything that genuinely needs them should include
// the shared declaration header rather than re-typing signatures here.

// ============================================================================
// oneapi_internal: Queue provider for the kernel registry
// ============================================================================
namespace {
    static void* g_backend_ptr = nullptr;
    static oneapi_internal::QueueGetter g_queue_getter = nullptr;
}

namespace oneapi_internal {
    void set_backend_queue_provider(void* backend) {
        g_backend_ptr = backend;
    }
    void set_queue_getter(QueueGetter fn) {
        g_queue_getter = fn;
    }
    sycl::queue& get_queue(int32_t device_id) {
        if (g_queue_getter == nullptr || g_backend_ptr == nullptr) {
            throw std::runtime_error(
                "OneAPI queue provider is not installed (backend not initialized "
                "or already destroyed); cannot obtain SYCL queue");
        }
        return g_queue_getter(g_backend_ptr, device_id);
    }
} // namespace oneapi_internal

// ============================================================================
// Intel OpenCL CPU Runtime: CPU architecture auto-detection
// ============================================================================
// The Intel OpenCL CPU runtime JIT-compiles SYCL/SPIR-V kernels to native
// code. It recognises Intel CPUs automatically, but does not recognise AMD
// or other x86-64 CPUs, emitting "Unknown host CPU" and failing to vectorise
// certain kernels ("Do not know how to split the result of this operator!").
//
// Fix: detect the host CPU feature set via CPUID and set the environment
// variable CL_CONFIG_CPU_TARGET_ARCH to a compatible Intel code-name that
// the runtime *does* know, before any SYCL platform/device enumeration
// triggers JIT compilation.
// ============================================================================
static void configure_opencl_cpu_target_arch() {
#ifdef __x86_64__
    // If the user already set it, respect their choice.
    if (std::getenv("CL_CONFIG_CPU_TARGET_ARCH")) return;

    // Use CPUID to detect the actual feature set of the host CPU.
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    // Check max basic CPUID leaf
    __cpuid(0, eax, ebx, ecx, edx);
    unsigned int max_leaf = eax;
    if (max_leaf < 7) {
        // Very old CPU — use the safest baseline
        setenv("CL_CONFIG_CPU_TARGET_ARCH", "corei7", /*overwrite=*/0);
        return;
    }

    // Leaf 1: detect SSE4.2 and AVX
    __cpuid(1, eax, ebx, ecx, edx);
    bool has_sse42  = (ecx >> 20) & 1;
    bool has_avx    = (ecx >> 28) & 1;

    // Leaf 7, sub-leaf 0: detect AVX2, AVX-512F
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    bool has_avx2   = (ebx >> 5) & 1;
    bool has_avx512f = (ebx >> 16) & 1;

    // Map features to an Intel code-name the OpenCL CPU runtime understands.
    // We intentionally pick conservative targets to maximise compatibility.
    const char* arch = "corei7";           // SSE4.2 baseline
    if (has_avx512f) {
        arch = "skx";                      // Skylake-X: AVX-512F
    } else if (has_avx2) {
        arch = "core-avx2";               // AVX2 (Haswell-class)
    } else if (has_avx) {
        arch = "corei7-avx";              // AVX (Sandy Bridge-class)
    } else if (has_sse42) {
        arch = "corei7";                   // SSE4.2 (Nehalem-class)
    }

    setenv("CL_CONFIG_CPU_TARGET_ARCH", arch, /*overwrite=*/0);
#endif  // __x86_64__
}

// ============================================================================
// SYCL persistent kernel-binary cache
// ============================================================================
// The Intel OpenCL CPU runtime JIT-compiles every kernel bundle on first use
// in each process. On this runtime a single cold elementwise kernel bundle
// can take 15-20 s to compile (measured: full()+first add() = 19.7 s cold vs
// 0.13 s warm), which made per-process test runs absurdly slow (e.g.
// ForeachOps.PerfManyTensors: 17.6 s for 1000 tiny adds, all of it one JIT)
// and starved heavy-model tests into 1200 s timeouts. SYCL ships an official
// on-disk binary cache for exactly this, but it is OFF by default. Enable it
// (respecting any explicit user setting, including an explicit "0").
static void configure_sycl_persistent_cache() {
    setenv("SYCL_CACHE_PERSISTENT", "1", /*overwrite=*/0);
}

/**
 * @brief OneAPI/SYCL backend implementation for Intel GPUs and CPUs.
 *
 * Supports Intel Data Center GPU Max Series, Intel Arc graphics, and Intel CPUs.
 * Uses SYCL for portable acceleration and optionally oneMKL/oneDNN for optimized operations.
 */
class OneAPIBackend : public Backend {
public:
    OneAPIBackend() {
        // Wire up queue provider BEFORE device enumeration, so kernel registry
        // callbacks can safely access queues if triggered during device init.
        oneapi_internal::set_backend_queue_provider(this);
        oneapi_internal::set_queue_getter([](void* backend, int32_t device_id) -> sycl::queue& {
            return static_cast<OneAPIBackend*>(backend)->get_queue(device_id);
        });

        // Configure the Intel OpenCL CPU runtime's JIT target architecture.
        // Must happen before any SYCL platform/device enumeration so the
        // runtime picks up the setting before it JIT-compiles SPIR-V kernels.
        configure_opencl_cpu_target_arch();
        configure_sycl_persistent_cache();

        // Policy: prefer Intel GPUs. If none are available, fall back to a
        // SYCL CPU device so the OneAPI code path remains exercisable on
        // hosts without Intel GPU hardware (this matches how every other
        // GPU backend in the project handles "no device" — by gracefully
        // falling back rather than refusing to load).
        //
        // TENZOR_ONEAPI_ALLOW_CPU controls behaviour when both a GPU and a
        // CPU SYCL device exist:
        //   unset / "1" / non-zero  → register every Intel device (GPU+CPU)
        //   "0"                     → register Intel GPUs only; never CPU
        // Regardless of this var, if no Intel GPU is found we will use the
        // CPU SYCL device as a fallback (unless the user set the var to "0").
        const char* allow_cpu_env = std::getenv("TENZOR_ONEAPI_ALLOW_CPU");
        const bool cpu_explicitly_forbidden =
            allow_cpu_env != nullptr && allow_cpu_env[0] == '0';
        const bool register_cpu_alongside_gpu =
            allow_cpu_env != nullptr && allow_cpu_env[0] != '\0' && allow_cpu_env[0] != '0';

        auto try_register_device = [this](const sycl::device& device) -> bool {
            try {
                auto queue = std::make_shared<sycl::queue>(device,
                    [this](sycl::exception_list elist) {
                        std::lock_guard<std::mutex> lock(async_errors_mutex_);
                        for (auto& e : elist) {
                            async_errors_.push_back(e);
                            try { std::rethrow_exception(e); }
                            catch (const sycl::exception& se) {
                                fprintf(stderr, "SYCL async error: %s\n", se.what());
                            }
                        }
                    },
                    sycl::property_list{sycl::property::queue::in_order{},
                                        sycl::property::queue::enable_profiling{}});

                OneAPIDeviceData dev_data;
                dev_data.queue = queue;
                dev_data.device = device;
                dev_data.name = device.get_info<sycl::info::device::name>();
                dev_data.type = device.is_gpu() ? "gpu" :
                           device.is_cpu() ? "cpu" : "accelerator";
                dev_data.max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
                dev_data.max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
                dev_data.global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
                dev_data.local_mem_size = device.get_info<sycl::info::device::local_mem_size>();

                backend::OneAPICachingAllocator::get().initialize(
                    dev_data.queue.get(), static_cast<int>(devices_.size()));

                devices_.push_back(dev_data);
                return true;
            } catch (const sycl::exception& e) {
                TENZOR_LOG_WARNING(
                    std::string("Skipping SYCL device: ") + e.what());
                return false;
            }
        };

        auto is_supported_vendor = [](const sycl::device& device) {
            // Skip NVIDIA devices - kernels are compiled for spir64 (Intel CPU/GPU)
            std::string vendor = device.get_info<sycl::info::device::vendor>();
            return vendor.find("NVIDIA") == std::string::npos &&
                   vendor.find("nvidia") == std::string::npos;
        };

        try {
            auto platforms = sycl::platform::get_platforms();

            // Pass 1: register Intel GPUs (preferred).
            for (const auto& platform : platforms) {
                for (const auto& device : platform.get_devices()) {
                    if (!device.is_gpu()) continue;
                    if (!is_supported_vendor(device)) continue;
                    try_register_device(device);
                }
            }

            // Pass 2: register CPU SYCL devices.
            //   - If a GPU was already registered, only register the CPU when
            //     TENZOR_ONEAPI_ALLOW_CPU is set to a non-zero value (legacy
            //     opt-in for parity runs).
            //   - If no GPU was registered, fall back to the CPU SYCL device
            //     so the backend is usable on Intel-less hosts. The user can
            //     suppress this fallback with TENZOR_ONEAPI_ALLOW_CPU=0.
            const bool need_cpu_fallback = devices_.empty() && !cpu_explicitly_forbidden;
            if (register_cpu_alongside_gpu || need_cpu_fallback) {
                for (const auto& platform : platforms) {
                    for (const auto& device : platform.get_devices()) {
                        if (!device.is_cpu()) continue;
                        if (!is_supported_vendor(device)) continue;
                        try_register_device(device);
                    }
                }
            }
        } catch (const sycl::exception& e) {
            // No SYCL devices available
        }
    }

    ~OneAPIBackend() override {
        // Clear the global queue provider FIRST so any kernel dispatched after
        // (or during) teardown sees a null provider and throws a clear error
        // via oneapi_internal::get_queue() instead of static_cast-ing and
        // dereferencing this soon-to-be-freed object (use-after-free).
        oneapi_internal::set_backend_queue_provider(nullptr);
        oneapi_internal::set_queue_getter(nullptr);

        // Wait for all in-flight work to finish. A pending async SYCL error
        // would be rethrown by wait_and_throw(); since destructors are
        // implicitly noexcept, an escaping throw would call std::terminate()
        // AND skip the USM cleanup below, leaking every allocation. Swallow any
        // teardown exception (using the non-throwing wait()) and always run the
        // release_all()/devices_.clear() cleanup.
        for (size_t i = 0; i < devices_.size(); ++i) {
            try {
                devices_[i].queue->wait();
            } catch (...) {
                // Best-effort: a hung/errored kernel at shutdown must not
                // prevent the remaining queues from draining or the USM from
                // being released.
            }
        }

        // Release ALL USM allocations (cached and in-use) while SYCL
        // queues are still alive. Guarded so a failure here cannot escape the
        // destructor either.
        try {
            backend::OneAPICachingAllocator::get().release_all();
        } catch (...) {
        }

        // release_all() has sycl::free'd and forgotten every USM block, so our
        // own tracking now points at freed memory. Clear it: any Storage that
        // outlives the backend will hit deallocate()'s untracked-pointer no-op
        // path instead of double-freeing.
        {
            std::lock_guard<std::mutex> lock(allocations_mutex_);
            allocations_.clear();
        }

        devices_.clear();
    }

    auto name() const -> std::string_view override {
        return "oneapi";
    }

    auto device_count() const -> int32_t override {
        return static_cast<int32_t>(devices_.size());
    }

    auto is_available() const -> bool override {
        return !devices_.empty();
    }

    auto get_device_info(int32_t device_id) const -> tenzor::DeviceInfo override {
        if (device_id < 0 || device_id >= static_cast<int32_t>(devices_.size())) {
            std::string range = devices_.empty()
                ? "(no OneAPI devices available)"
                : "(available: 0-" + std::to_string(devices_.size() - 1) + ")";
            throw std::out_of_range("Invalid OneAPI device ID: " + std::to_string(device_id) +
                                    " " + range);
        }

        const auto& dev = devices_[device_id];
        tenzor::DeviceInfo info;

        info.name = dev.name;

        // Determine vendor from device name or platform
        auto platform_name = dev.device.get_platform().get_info<sycl::info::platform::name>();
        if (platform_name.find("Intel") != std::string::npos) {
            info.vendor = "Intel";
        } else if (platform_name.find("AMD") != std::string::npos) {
            info.vendor = "AMD";
        } else if (platform_name.find("NVIDIA") != std::string::npos) {
            info.vendor = "NVIDIA";
        } else {
            info.vendor = platform_name;
        }

        // Driver version from platform
        info.driver_version = dev.device.get_platform().get_info<sycl::info::platform::version>();

        // Memory info
        info.total_memory = dev.global_mem_size;
        info.available_memory = dev.global_mem_size;  // SYCL doesn't easily provide free memory

        // Compute info
        info.compute_units = dev.max_compute_units;
        info.max_threads_per_block = static_cast<int>(dev.max_work_group_size);
        info.max_shared_memory = static_cast<int>(dev.local_mem_size);

        // SYCL sub-group size is like warp size
        // Z.5: Pick the largest sub-group size ≤ 64 so AMD (via Codeplay) reports 64,
        // NVIDIA reports 32, Intel iGPU reports 8 or 16. Falling back to front()
        // silently picked the smallest, breaking reduction-tile sizing on AMD/Intel.
        try {
            auto sub_group_sizes = dev.device.get_info<sycl::info::device::sub_group_sizes>();
            if (!sub_group_sizes.empty()) {
                size_t best = 0;
                for (size_t s : sub_group_sizes) {
                    if (s <= 64 && s > best) {
                        best = s;
                    }
                }
                // If all sizes exceed 64 (extremely unlikely), pick the smallest.
                if (best == 0) {
                    best = *std::min_element(sub_group_sizes.begin(), sub_group_sizes.end());
                }
                info.warp_size = static_cast<int>(best);
            } else {
                // Query succeeded but returned no sizes: keep a sane default so
                // downstream reduction/tile sizing never divides by zero.
                info.warp_size = 32;
            }
        }
#ifdef TENZOR_HAS_ONEMKL
        catch (const ::oneapi::mkl::exception& e) {
            // Audit L.4: surface specific MKL error info instead of folding into
            // the generic catch-all. The oneMKL exception type currently only
            // exposes what(); the encoded domain::function::info is part of the
            // message string.
            TENZOR_LOG_WARNING(
                std::string("[OneAPI get_device_info] oneMKL exception querying sub-group sizes: ")
                + e.what());
            info.warp_size = 32;  // Default
        }
#endif
        catch (const sycl::exception& e) {
            // Audit L.4: name the SYCL exception type so the actual error code
            // is preserved in logs rather than mapped to a generic string.
            TENZOR_LOG_WARNING(
                std::string("[OneAPI get_device_info] SYCL exception querying sub-group sizes: ")
                + e.what());
            info.warp_size = 32;  // Default
        }
        catch (const std::exception& e) {
            // Audit L.4: any other std-derived exception still gets its type/msg
            // logged rather than being silently mapped to the default.
            TENZOR_LOG_WARNING(
                std::string("[OneAPI get_device_info] non-SYCL exception (type=")
                + typeid(e).name() + ") querying sub-group sizes: " + e.what());
            info.warp_size = 32;  // Default
        }
        catch (...) {
            // Audit L.4: unknown exception type — keep default but log loudly.
            TENZOR_LOG_WARNING(
                "[OneAPI get_device_info] unknown exception type querying sub-group sizes; "
                "using default warp_size=32");
            info.warp_size = 32;  // Default
        }

        // Feature support
        info.supports_fp16 = dev.device.has(sycl::aspect::fp16);
        info.supports_fp64 = dev.device.has(sycl::aspect::fp64);
        // SYCL exposes no portable aspect for 8-bit integer compute. Every
        // conformant SYCL device supports sycl::char/int8 arithmetic as part of
        // the core data types, and Tenzor's int8 kernels widen to int32 for the
        // accumulation anyway, so this is an unconditional (and safe) assumption
        // rather than a queried capability like fp16/fp64 above.
        info.supports_int8 = true;

        // Device type
        info.is_integrated = !dev.device.is_gpu() ||
            (dev.device.has(sycl::aspect::usm_system_allocations));
        info.is_discrete = dev.device.is_gpu() && !info.is_integrated;

        return info;
    }

    auto allocate(size_t bytes, int32_t device_id) -> void* override {
        if (bytes == 0) {
            return nullptr;
        }

        validate_device_id(device_id);

        try {
            // Use caching allocator for efficient memory reuse
            // Uses USM (Unified Shared Memory) shared allocation under the hood
            auto& allocator = backend::OneAPICachingAllocator::get();
            void* ptr = allocator.allocate_shared(bytes, device_id);

            if (ptr == nullptr) {
                throw std::runtime_error("OneAPI caching allocator allocation failed");
            }

            // Track allocation for proper deallocation
            {
                std::lock_guard<std::mutex> lock(allocations_mutex_);
                allocations_[ptr] = device_id;
            }

            return ptr;
        } catch (const sycl::exception& e) {
            throw std::runtime_error(
                std::string("OneAPI allocation failed: ") + e.what()
            );
        }
    }

    auto deallocate(void* ptr) -> void override {
        if (ptr == nullptr) {
            return;
        }

        int32_t device_id;
        {
            std::lock_guard<std::mutex> lock(allocations_mutex_);
            auto it = allocations_.find(ptr);
            if (it == allocations_.end()) {
                // Untracked pointer. This is reachable during/after shutdown:
                // ~OneAPIBackend calls release_all() which sycl::free's and
                // forgets every USM block, but a Storage/Tensor that outlives the
                // backend (a static/global, or cross-TU teardown ordering) will
                // still call deallocate() on its now-freed pointer. Throwing here
                // would propagate out of a noexcept Storage destructor and call
                // std::terminate(). The memory was already reclaimed by
                // release_all() (and the OS reclaims everything at exit), so treat
                // an untracked free as a no-op rather than a hard error.
                return;
            }
            device_id = it->second;
            allocations_.erase(it);
        }

        // Return memory to caching allocator for reuse (outside the map lock;
        // the caching allocator has its own synchronization).
        backend::OneAPICachingAllocator::get().free(ptr, device_id);
    }

    auto copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void override {
        if (bytes == 0) {
            return;
        }

        // Determine which queue to use based on copy kind
        sycl::queue* queue_ptr = nullptr;

        switch (kind) {
            case CopyKind::HostToHost:
                // Direct memcpy for host-to-host
                std::memcpy(dst, src, bytes);
                return;

            case CopyKind::HostToDevice: {
                // Use destination device's queue for H2D
                if (devices_.empty()) {
                    throw std::runtime_error("No SYCL devices available for copy");
                }
                // Resolve the owning device of the destination USM pointer.
                // Falls back to sycl::get_pointer_device when dst is not a
                // tracked base pointer (sub-buffer/offset/external USM) so the
                // copy is submitted on the queue for the device that actually
                // backs the memory, not silently device 0.
                int32_t dev_id = resolve_device_for_ptr(dst);
                if (dev_id < 0) {
                    throw std::runtime_error(
                        "SYCL H2D copy: destination pointer is not owned by any "
                        "registered OneAPI device");
                }
                queue_ptr = devices_[dev_id].queue.get();
                break;
            }
            case CopyKind::DeviceToHost: {
                // Use source device's queue for D2H
                if (devices_.empty()) {
                    throw std::runtime_error("No SYCL devices available for copy");
                }
                int32_t dev_id = resolve_device_for_ptr(src);
                if (dev_id < 0) {
                    throw std::runtime_error(
                        "SYCL D2H copy: source pointer is not owned by any "
                        "registered OneAPI device");
                }
                queue_ptr = devices_[dev_id].queue.get();
                break;
            }
            case CopyKind::DeviceToDevice: {
                // Use destination device's queue for D2D, falling back to the
                // source device if the destination pointer cannot be resolved.
                if (devices_.empty()) {
                    throw std::runtime_error("No SYCL devices available for copy");
                }
                int32_t dev_id = resolve_device_for_ptr(dst);
                if (dev_id < 0) {
                    dev_id = resolve_device_for_ptr(src);
                }
                if (dev_id < 0) {
                    throw std::runtime_error(
                        "SYCL D2D copy: neither source nor destination pointer "
                        "is owned by any registered OneAPI device");
                }
                queue_ptr = devices_[dev_id].queue.get();
                break;
            }
        }

        if (queue_ptr) {
            try {
                auto event = queue_ptr->memcpy(dst, src, bytes);
                // Always wait for the copy to complete. For H2D/D2D the
                // source memory may be freed by the caller immediately
                // after this function returns, so the async copy must
                // finish before that happens.
                event.wait();
            } catch (const sycl::exception& e) {
                throw std::runtime_error(
                    std::string("SYCL copy failed: ") + e.what()
                );
            }
        }
    }

    // Blocks until all operations on the specified device queue complete.
    // SYCL has no timeout API — wait_and_throw() blocks indefinitely.
    // A hung kernel will cause this call to never return.
    auto synchronize(int32_t device_id) -> void override {
        validate_device_id(device_id);
        get_queue(device_id).wait_and_throw();
        check_async_errors();
    }

    auto create_stream(int32_t device_id) -> StreamHandle override {
        validate_device_id(device_id);

        try {
            auto& device = devices_[device_id].device;
            // Share the device's existing SYCL context so USM pointers allocated
            // by OneAPICachingAllocator (bound to the main queue's context) are
            // valid on this stream queue. Constructing without the context would
            // create a fresh one, making backend-allocated tensors invalid here.
            auto context = devices_[device_id].queue->get_context();
            auto* queue = new sycl::queue(context, device,
                [this](sycl::exception_list elist) {
                    std::lock_guard<std::mutex> lock(async_errors_mutex_);
                    for (auto& e : elist) {
                        async_errors_.push_back(e);
                        try { std::rethrow_exception(e); }
                        catch (const sycl::exception& se) {
                            fprintf(stderr, "SYCL async error: %s\n", se.what());
                        }
                    }
                },
                sycl::property_list{sycl::property::queue::in_order{},
                                    sycl::property::queue::enable_profiling{}});
            return static_cast<StreamHandle>(queue);
        } catch (const sycl::exception& e) {
            throw std::runtime_error(
                std::string("Failed to create SYCL queue: ") + e.what()
            );
        }
    }

    auto destroy_stream(StreamHandle stream) -> void override {
        if (stream == nullptr) {
            return;
        }

        auto* queue = static_cast<sycl::queue*>(stream);
        try {
            queue->wait();
            delete queue;
        } catch (const sycl::exception& e) {
            delete queue;
            throw std::runtime_error(
                std::string("Failed to destroy SYCL queue: ") + e.what()
            );
        }
    }

    auto synchronize_stream(StreamHandle stream) -> void override {
        if (stream == nullptr) {
            throw std::invalid_argument("Cannot synchronize null stream");
        }

        auto* queue = static_cast<sycl::queue*>(stream);
        try {
            queue->wait_and_throw();
        } catch (const sycl::exception& e) {
            throw std::runtime_error(
                std::string("SYCL stream synchronization failed: ") + e.what()
            );
        }
    }

    auto create_event(int32_t device_id, bool enable_timing = true) -> EventHandle override {
        validate_device_id(device_id);
        // SYCL events are created when operations are submitted.
        // We use a sycl::event pointer as the opaque handle.
        // A "blank" event is created via default construction.
        (void)enable_timing;  // SYCL events always support profiling if queue has it
        auto* event = new sycl::event();
        return static_cast<EventHandle>(event);
    }

    auto destroy_event(EventHandle event) -> void override {
        if (event) {
            delete static_cast<sycl::event*>(event);
        }
    }

    auto record_event(EventHandle event, StreamHandle stream = nullptr) -> void override {
        if (!event) return;
        if (!stream) {
            throw std::invalid_argument("OneAPI record_event requires a non-null stream (SYCL queue)");
        }
        auto* queue = static_cast<sycl::queue*>(stream);
        // Submit a marker event on the queue
        auto* ev = static_cast<sycl::event*>(event);
        try {
            *ev = queue->submit([](sycl::handler& h) {
                // Empty kernel acts as a synchronization marker
                h.host_task([]() {});
            });
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL record_event failed: ") + e.what());
        }
    }

    auto wait_event(EventHandle event, StreamHandle stream = nullptr) -> void override {
        if (!event) return;
        auto* ev = static_cast<sycl::event*>(event);
        // Block until the event completes
        try {
            ev->wait_and_throw();
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL wait_event failed: ") + e.what());
        }
    }

    auto event_elapsed_ms(EventHandle start_event, EventHandle end_event) -> float override {
        if (!start_event || !end_event) return 0.0f;
        auto* start_ev = static_cast<sycl::event*>(start_event);
        auto* end_ev = static_cast<sycl::event*>(end_event);
        try {
            end_ev->wait();
            start_ev->wait();
            auto start_time = start_ev->get_profiling_info<sycl::info::event_profiling::command_end>();
            auto end_time = end_ev->get_profiling_info<sycl::info::event_profiling::command_end>();
            // Profiling returns nanoseconds
            return static_cast<float>(end_time - start_time) / 1e6f;
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL event_elapsed_ms failed: ") + e.what());
        }
    }

    auto synchronize_event(EventHandle event) -> void override {
        if (!event) return;
        auto* ev = static_cast<sycl::event*>(event);
        try {
            ev->wait_and_throw();
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL synchronize_event failed: ") + e.what());
        }
    }

    auto memset(void* ptr, int value, size_t bytes, int32_t device_id) -> void override {
        validate_device_id(device_id);
        try {
            get_queue(device_id).memset(ptr, value, bytes).wait();
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL memset failed: ") + e.what());
        }
    }

    // Legacy string-keyed dispatch removed (audit Phase C).

private:
    struct OneAPIDeviceData {
        std::shared_ptr<sycl::queue> queue;
        sycl::device device;
        std::string name;
        std::string type;
        uint32_t max_compute_units;
        size_t max_work_group_size;
        uint64_t global_mem_size;
        uint64_t local_mem_size;
    };

    std::vector<OneAPIDeviceData> devices_;
    std::unordered_map<void*, int32_t> allocations_;
    // Guards allocations_. allocate()/deallocate()/copy() are called
    // concurrently from worker threads (e.g. parallel backward), and
    // std::unordered_map is not safe for concurrent insert+find/erase — a
    // rehash during another thread's lookup corrupts the table and surfaces
    // as "Attempt to free untracked pointer".
    std::mutex allocations_mutex_;
    std::mutex async_errors_mutex_;
    std::vector<std::exception_ptr> async_errors_;

    void check_async_errors() {
        std::lock_guard<std::mutex> lock(async_errors_mutex_);
        if (!async_errors_.empty()) {
            auto e = async_errors_.front();
            async_errors_.clear();
            std::rethrow_exception(e);
        }
    }

    auto get_queue(int32_t device_id) -> sycl::queue& {
        // Out-of-range indices previously did raw vector indexing (UB) — an
        // invalid Device::oneapi(N) must throw, not fall through (see
        // OneAPIBackendTest.InvalidDeviceIndex).
        validate_device_id(device_id);
        return *devices_[device_id].queue;
    }

    // Resolve the device index that owns a (possibly untracked) USM device
    // pointer. First consult allocations_ for the exact base pointer; if that
    // misses (sub-buffer/offset pointer, or externally allocated USM), query
    // SYCL via sycl::get_pointer_device against each registered device's
    // context and match the owning sycl::device. Returns -1 if the pointer is
    // not device memory owned by any registered device (e.g. a host pointer).
    auto resolve_device_for_ptr(const void* ptr) -> int32_t {
        {
            std::lock_guard<std::mutex> lock(allocations_mutex_);
            auto it = allocations_.find(const_cast<void*>(ptr));
            if (it != allocations_.end()) {
                return it->second;
            }
        }
        // Not a tracked base pointer — ask SYCL which device backs this USM
        // allocation and map it back to a registered device index.
        for (size_t i = 0; i < devices_.size(); ++i) {
            try {
                const auto& ctx = devices_[i].queue->get_context();
                auto alloc_kind = sycl::get_pointer_type(ptr, ctx);
                if (alloc_kind == sycl::usm::alloc::unknown ||
                    alloc_kind == sycl::usm::alloc::host) {
                    continue;
                }
                sycl::device owner = sycl::get_pointer_device(ptr, ctx);
                if (owner == devices_[i].device) {
                    return static_cast<int32_t>(i);
                }
            } catch (const sycl::exception&) {
                // Pointer not associated with this context; try the next.
                continue;
            }
        }
        return -1;
    }

    auto validate_device_id(int32_t device_id) const -> void {
        if (device_id < 0 || device_id >= static_cast<int32_t>(devices_.size())) {
            std::string range = devices_.empty()
                ? "(no OneAPI devices available)"
                : "(available: 0-" + std::to_string(devices_.size() - 1) + ")";
            throw std::invalid_argument(
                "Invalid device ID " + std::to_string(device_id) + " " + range
            );
        }
    }

};

// Library-level constructor: runs at dlopen() time, BEFORE create_backend().
// The Intel OpenCL CPU runtime reads CL_CONFIG_CPU_TARGET_ARCH during its own
// static initialisation which is triggered by the first SYCL platform/device
// enumeration.  Setting the env-var inside the OneAPIBackend constructor is too
// late — by then libintelocl.so has already been loaded and its JIT target has
// been locked in.  A __attribute__((constructor)) function runs early enough.
__attribute__((constructor))
static void early_configure_opencl_cpu_target() {
    configure_opencl_cpu_target_arch();
    configure_sycl_persistent_cache();
}

extern "C" {
    Backend* create_backend() {
        return new OneAPIBackend();
    }
}

} // namespace tenzor
