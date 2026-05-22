#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/backend/cpu_caching_allocator.hpp"
#include <cstring>
#include <stdexcept>
#include <thread>
#include <fstream>
#include <string>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#include "cpu_thread_config.hpp"
#endif

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
#ifndef _WIN32
__attribute__((constructor(101)))
static void pin_tbb_libs() {
    // Pin all TBB libraries to prevent their static destructors from running
    // during __cxa_finalize. Without this, libtbb's __TBB_InitOnce destructor
    // calls cache_aligned_deallocate through a scalable_free weak symbol that
    // becomes NULL after tbbmalloc cleanup.
    const char* libs[] = {
        "libtbbmalloc.so.2", "libtbbmalloc_debug.so.2",
        "libtbb.so.12", "libtbb_debug.so.12",
    };
    for (const char* lib : libs) {
        dlopen(lib, RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    }
}
#endif

// ============================================================================
// Early OpenMP Configuration (Static Constructor)
// ============================================================================
// This runs when the CPU backend shared library is loaded, setting
// OMP_NUM_THREADS before any OpenMP runtime initialization.
// Note: This may not help if PyTorch/MKL is loaded first, but it ensures
// our library uses all threads when loaded independently.
__attribute__((constructor(102)))
static void configure_openmp_early() {
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
        std::string env_value = std::to_string(num_threads);
        setenv("OMP_NUM_THREADS", env_value.c_str(), 0);
    }
}

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace tenzor {

// Forward declarations for CPU kernels
namespace cpu {
    auto add_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto sub_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto mul_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto div_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto matmul_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto sqrt_kernel(const Tensor& input) -> Tensor;
    auto neg_kernel(const Tensor& input) -> Tensor;
    auto abs_kernel(const Tensor& input) -> Tensor;
    auto sign_kernel(const Tensor& input) -> Tensor;
    auto clamp_kernel(const Tensor& input, double min_val, double max_val) -> Tensor;
    auto log_kernel(const Tensor& input) -> Tensor;
    auto exp_kernel(const Tensor& input) -> Tensor;
    auto pow_kernel(const Tensor& input, double exponent) -> Tensor;
    auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto max_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto min_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto argsort_kernel(const Tensor& input, int64_t dim, bool descending) -> Tensor;
    auto prod_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto var_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor;
    auto std_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor;
    auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim) -> Tensor;
    auto any_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto all_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;

    // Comparison operations
    auto eq_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto ne_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto lt_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto le_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto gt_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto ge_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto dot_kernel(const Tensor& a, const Tensor& b) -> Tensor;

    // Trigonometric operations
    auto sin_kernel(const Tensor& input) -> Tensor;
    auto cos_kernel(const Tensor& input) -> Tensor;
    auto tan_kernel(const Tensor& input) -> Tensor;
    auto asin_kernel(const Tensor& input) -> Tensor;
    auto acos_kernel(const Tensor& input) -> Tensor;
    auto atan_kernel(const Tensor& input) -> Tensor;
    auto sinh_kernel(const Tensor& input) -> Tensor;
    auto cosh_kernel(const Tensor& input) -> Tensor;

    // Rounding operations
    auto round_kernel(const Tensor& input) -> Tensor;
    auto floor_kernel(const Tensor& input) -> Tensor;
    auto ceil_kernel(const Tensor& input) -> Tensor;

    // Other math operations
    auto reciprocal_kernel(const Tensor& input) -> Tensor;

    // In-place operations
    auto add_inplace_kernel(Tensor& a, const Tensor& b) -> void;
    auto mul_inplace_kernel(Tensor& a, const Tensor& b) -> void;
    auto sub_inplace_kernel(Tensor& a, const Tensor& b) -> void;
    auto div_inplace_kernel(Tensor& a, const Tensor& b) -> void;

    // Activation kernels
    auto relu_kernel(const Tensor& input) -> Tensor;
    auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto sigmoid_kernel(const Tensor& input) -> Tensor;
    auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto tanh_kernel(const Tensor& input) -> Tensor;
    auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto gelu_kernel(const Tensor& input) -> Tensor;
    auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto swish_kernel(const Tensor& input) -> Tensor;
    auto swish_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto leaky_relu_kernel(const Tensor& input, double alpha) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, double alpha) -> Tensor;
    auto elu_kernel(const Tensor& input, float alpha) -> Tensor;
    auto elu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha) -> Tensor;
    auto selu_kernel(const Tensor& input) -> Tensor;
    auto selu_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto mish_kernel(const Tensor& input) -> Tensor;
    auto mish_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto hardswish_kernel(const Tensor& input) -> Tensor;
    auto hardsigmoid_kernel(const Tensor& input) -> Tensor;
    auto softplus_kernel(const Tensor& input, float beta, float threshold) -> Tensor;
    auto softplus_backward_kernel(const Tensor& grad_output, const Tensor& input, float beta, float threshold) -> Tensor;
    auto softmax_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor;
    auto log_softmax_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor;

    // Transform kernels
    auto contiguous_kernel(const Tensor& input) -> Tensor;
    auto fill_kernel(const Tensor& input, float value) -> Tensor;
    auto clone_kernel(const Tensor& input) -> Tensor;
    auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape) -> Tensor;
    auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor;
    auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor;
    auto squeeze_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto unsqueeze_kernel(const Tensor& input, int64_t dim) -> Tensor;

    // BatchNorm kernels
    auto batchnorm2d_mean_var_kernel(const Tensor& input) -> std::vector<Tensor>;
    auto batchnorm2d_forward_kernel(const Tensor& input, const Tensor& mean, const Tensor& variance, float epsilon) -> Tensor;
    auto batchnorm2d_forward_affine_kernel(const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, const Tensor& beta, float epsilon) -> Tensor;
    auto batchnorm2d_update_running_stats_kernel(Tensor& running_mean, Tensor& running_var, const Tensor& batch_mean, const Tensor& batch_var, float momentum) -> void;
    auto batchnorm2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, float epsilon) -> std::vector<Tensor>;

    // Creation kernels
    auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;

    // Conv2d kernels
    auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_input_kernel(const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& input_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& weight_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_bias_kernel(const Tensor& grad_output) -> Tensor;
    auto conv_transpose2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups) -> Tensor;

    // Fused operation kernels
    auto fused_linear_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor;
    auto fused_conv2d_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto fused_batchnorm_relu_kernel(const Tensor& input, const Tensor& running_mean, const Tensor& running_var, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto fused_softmax_cross_entropy_kernel(const Tensor& logits, const Tensor& targets, bool compute_grad, const std::string& reduction = "mean") -> std::vector<Tensor>;
    auto fused_add_relu_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto fused_gelu_kernel(const Tensor& input) -> Tensor;
    auto fused_layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;

    // Indexing kernels
    auto index_select_kernel(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor;
    auto gather_kernel(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor;
    auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor;
    auto scatter_add_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor;
    auto masked_select_kernel(const Tensor& input, const Tensor& mask) -> Tensor;
    auto masked_fill_kernel(const Tensor& input, const Tensor& mask, float value) -> Tensor;
    auto where_kernel(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor;
} // namespace cpu

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

        // Get number of hardware threads
        info.compute_units = std::thread::hardware_concurrency();
        if (info.compute_units == 0) {
            info.compute_units = 1;  // Fallback
        }

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
                sscanf(line.c_str(), "MemTotal: %zu kB", &kb);
                info.total_memory = kb * 1024;
            } else if (line.find("MemAvailable:") == 0) {
                size_t kb = 0;
                sscanf(line.c_str(), "MemAvailable: %zu kB", &kb);
                info.available_memory = kb * 1024;
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
