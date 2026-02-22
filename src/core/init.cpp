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
            std::cout << "CPU dispatch table initialized with O(1) lookup" << std::endl;
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
                        std::cout << "CUDA dispatch table initialized with O(1) lookup" << std::endl;
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

    if (std::filesystem::exists(rocm_backend_path)) {
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
                        std::cout << "ROCm dispatch table initialized with O(1) lookup" << std::endl;
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
    } else {
        std::cout << "ROCm backend not found at: " << rocm_backend_path << std::endl;
    }

    // Try to load OneAPI backend if available
    std::filesystem::path oneapi_backend_path = bin_path / "tenzor_backend_oneapi.so";

    if (std::filesystem::exists(oneapi_backend_path)) {
        std::cout << "Loading OneAPI backend from: " << oneapi_backend_path << std::endl;

        auto oneapi_result = loader.load_backend(oneapi_backend_path);
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

                // Helper macro to register OneAPI kernels with non-capturing lambdas.
                // NOTE: These lambdas delegate to backend->dispatch(string, ...) which uses
                // O(n) string lookup internally. This is a known performance limitation —
                // long-term the OneAPI backend should register proper kernel functions for
                // O(1) OpId-based dispatch like the CPU/CUDA/Vulkan backends.
                #define ONEAPI_REGISTER(op_id, op_name) \
                    oneapi_table.register_kernel(OpId::op_id, \
                        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> { \
                            return DispatchTableRegistry::get_backend(Device::Type::OneAPI)->dispatch(op_name, inputs, attrs); \
                        })

                // Arithmetic operations
                ONEAPI_REGISTER(Add, "add");
                ONEAPI_REGISTER(Sub, "sub");
                ONEAPI_REGISTER(Mul, "mul");
                ONEAPI_REGISTER(Div, "div");
                ONEAPI_REGISTER(MatMul, "matmul");
                ONEAPI_REGISTER(Bmm, "bmm");
                ONEAPI_REGISTER(Dot, "dot");
                ONEAPI_REGISTER(Pow, "pow");

                // Inplace operations
                ONEAPI_REGISTER(AddInplace, "add_inplace");
                ONEAPI_REGISTER(SubInplace, "sub_inplace");
                ONEAPI_REGISTER(MulInplace, "mul_inplace");
                ONEAPI_REGISTER(DivInplace, "div_inplace");

                // Unary math operations
                ONEAPI_REGISTER(Sqrt, "sqrt");
                ONEAPI_REGISTER(Neg, "neg");
                ONEAPI_REGISTER(Abs, "abs");
                ONEAPI_REGISTER(Sign, "sign");
                ONEAPI_REGISTER(Log, "log");
                ONEAPI_REGISTER(Exp, "exp");
                ONEAPI_REGISTER(Reciprocal, "reciprocal");
                ONEAPI_REGISTER(Floor, "floor");
                ONEAPI_REGISTER(Ceil, "ceil");
                ONEAPI_REGISTER(Round, "round");

                // Trigonometric operations
                ONEAPI_REGISTER(Sin, "sin");
                ONEAPI_REGISTER(Cos, "cos");
                ONEAPI_REGISTER(Tan, "tan");
                ONEAPI_REGISTER(Asin, "asin");
                ONEAPI_REGISTER(Acos, "acos");
                ONEAPI_REGISTER(Atan, "atan");
                ONEAPI_REGISTER(Sinh, "sinh");
                ONEAPI_REGISTER(Cosh, "cosh");
                ONEAPI_REGISTER(Tanh, "tanh");

                // Clamp operations
                ONEAPI_REGISTER(Clamp, "clamp");
                ONEAPI_REGISTER(ClampMin, "clamp_min");
                ONEAPI_REGISTER(ClampMax, "clamp_max");

                // Comparison operations
                ONEAPI_REGISTER(Eq, "eq");
                ONEAPI_REGISTER(Ne, "ne");
                ONEAPI_REGISTER(Lt, "lt");
                ONEAPI_REGISTER(Le, "le");
                ONEAPI_REGISTER(Gt, "gt");
                ONEAPI_REGISTER(Ge, "ge");

                // Reduction operations
                ONEAPI_REGISTER(Sum, "sum");
                ONEAPI_REGISTER(Mean, "mean");
                ONEAPI_REGISTER(Max, "max");
                ONEAPI_REGISTER(Min, "min");
                ONEAPI_REGISTER(ArgMax, "argmax");
                ONEAPI_REGISTER(ArgMin, "argmin");
                ONEAPI_REGISTER(Prod, "prod");
                ONEAPI_REGISTER(Var, "var");
                ONEAPI_REGISTER(Std, "std");
                ONEAPI_REGISTER(Norm, "norm");

                // Activation functions
                ONEAPI_REGISTER(ReLU, "relu");
                ONEAPI_REGISTER(ReLUBackward, "relu_backward");
                ONEAPI_REGISTER(Sigmoid, "sigmoid");
                ONEAPI_REGISTER(SigmoidBackward, "sigmoid_backward");
                ONEAPI_REGISTER(TanhBackward, "tanh_backward");
                ONEAPI_REGISTER(Gelu, "gelu");
                ONEAPI_REGISTER(GeluBackward, "gelu_backward");
                ONEAPI_REGISTER(Swish, "swish");
                ONEAPI_REGISTER(SwishBackward, "swish_backward");
                ONEAPI_REGISTER(LeakyReLU, "leaky_relu");
                ONEAPI_REGISTER(LeakyReLUBackward, "leaky_relu_backward");
                ONEAPI_REGISTER(Softmax, "softmax");
                ONEAPI_REGISTER(SoftmaxBackward, "softmax_backward");
                ONEAPI_REGISTER(LogSoftmax, "log_softmax");
                ONEAPI_REGISTER(LogSoftmaxBackward, "log_softmax_backward");

                // Shape operations
                ONEAPI_REGISTER(Reshape, "reshape");
                ONEAPI_REGISTER(Transpose, "transpose");
                ONEAPI_REGISTER(Permute, "permute");
                ONEAPI_REGISTER(Squeeze, "squeeze");
                ONEAPI_REGISTER(Unsqueeze, "unsqueeze");
                ONEAPI_REGISTER(Contiguous, "contiguous");
                ONEAPI_REGISTER(Clone, "clone");
                ONEAPI_REGISTER(Fill, "fill");
                ONEAPI_REGISTER(Repeat, "repeat");
                ONEAPI_REGISTER(Expand, "expand");
                ONEAPI_REGISTER(Cat, "cat");

                // Indexing operations
                ONEAPI_REGISTER(IndexSelect, "index_select");
                ONEAPI_REGISTER(Gather, "gather");
                ONEAPI_REGISTER(Scatter, "scatter");
                ONEAPI_REGISTER(MaskedSelect, "masked_select");
                ONEAPI_REGISTER(MaskedFill, "masked_fill");
                ONEAPI_REGISTER(Where, "where");

                // Convolution operations
                ONEAPI_REGISTER(Conv2dForward, "conv2d_forward");
                ONEAPI_REGISTER(Conv2dBackwardInput, "conv2d_backward_input");
                ONEAPI_REGISTER(Conv2dBackwardWeight, "conv2d_backward_weight");
                ONEAPI_REGISTER(Conv2dBackwardBias, "conv2d_backward_bias");

                // Pooling operations
                ONEAPI_REGISTER(AvgPool2dForward, "avg_pool2d");
                ONEAPI_REGISTER(AvgPool2dBackward, "avg_pool2d_backward");
                ONEAPI_REGISTER(MaxPool2dForward, "max_pool2d");
                ONEAPI_REGISTER(MaxPool2dBackward, "max_pool2d_backward");
                ONEAPI_REGISTER(AdaptiveAvgPool2d, "adaptive_avg_pool2d");
                ONEAPI_REGISTER(AdaptiveAvgPool2dBackward, "adaptive_avg_pool2d_backward");
                ONEAPI_REGISTER(AdaptiveMaxPool2d, "adaptive_max_pool2d");

                // Batch normalization operations
                ONEAPI_REGISTER(BatchNorm2dMeanVar, "batchnorm2d_mean_var");
                ONEAPI_REGISTER(BatchNorm2dForward, "batchnorm2d_forward");
                ONEAPI_REGISTER(BatchNorm2dForwardAffine, "batchnorm2d_forward_affine");
                ONEAPI_REGISTER(BatchNorm2dUpdateRunningStats, "batchnorm2d_update_running_stats");
                ONEAPI_REGISTER(BatchNorm2dBackward, "batchnorm2d_backward");

                // Creation operations
                ONEAPI_REGISTER(Zeros, "zeros");
                ONEAPI_REGISTER(Ones, "ones");
                ONEAPI_REGISTER(Full, "full");
                ONEAPI_REGISTER(Rand, "rand");
                ONEAPI_REGISTER(Randn, "randn");

                // Embedding operations
                ONEAPI_REGISTER(Embedding, "embedding_lookup");
                ONEAPI_REGISTER(EmbeddingBackward, "embedding_backward");

                // RNN operations
                ONEAPI_REGISTER(LSTMCellForward, "lstm_cell_forward");
                ONEAPI_REGISTER(LSTMCellBackward, "lstm_cell_backward");
                ONEAPI_REGISTER(GRUCellForward, "gru_cell_forward");
                ONEAPI_REGISTER(GRUCellBackward, "gru_cell_backward");

                // Fused operations
                ONEAPI_REGISTER(FusedAddReLU, "fused_add_relu");
                ONEAPI_REGISTER(FusedGelu, "fused_gelu");
                ONEAPI_REGISTER(FusedLayerNorm, "fused_layer_norm");
                ONEAPI_REGISTER(LayerNormBackward, "fused_layer_norm_backward");
                ONEAPI_REGISTER(FusedLinearReLU, "fused_linear_relu");
                ONEAPI_REGISTER(FusedBatchNormReLU, "fused_batchnorm_relu");
                ONEAPI_REGISTER(FusedSoftmaxCrossEntropy, "fused_softmax_cross_entropy");

                // Vision operations
                ONEAPI_REGISTER(ROIAlignForward, "roi_align");
                ONEAPI_REGISTER(ROIAlignBackward, "roi_align_backward");
                ONEAPI_REGISTER(Interpolate, "interpolate");

                // In-place activation operations
                ONEAPI_REGISTER(ReLUInplace, "relu_inplace");
                ONEAPI_REGISTER(SigmoidInplace, "sigmoid_inplace");
                ONEAPI_REGISTER(TanhInplace, "tanh_inplace");
                ONEAPI_REGISTER(LeakyReLUInplace, "leaky_relu_inplace");
                ONEAPI_REGISTER(GeluInplace, "gelu_inplace");

                // Indexing operations
                ONEAPI_REGISTER(Nonzero, "nonzero");
                ONEAPI_REGISTER(OneHot, "one_hot");
                ONEAPI_REGISTER(ArgSort, "argsort");

                // Transposed convolution
                ONEAPI_REGISTER(ConvTranspose2dForward, "conv_transpose2d_forward");

                #undef ONEAPI_REGISTER

                DispatchTableRegistry::mark_ready(Device::Type::OneAPI);
                std::cout << "OneAPI dispatch table initialized with O(1) lookup" << std::endl;

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
                        std::cout << "Vulkan dispatch table initialized with O(1) lookup" << std::endl;
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

    std::cout << "Tenzor initialization complete - 51 CPU operations registered" << std::endl;

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
