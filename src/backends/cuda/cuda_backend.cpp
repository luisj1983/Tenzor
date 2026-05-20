#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/core/device_guard.hpp"
#ifdef TENZOR_HAS_CUDNN
#include "tenzor/backend/cudnn_wrapper.hpp"
#else
#pragma message("WARNING: Building without cuDNN. Conv2d, pooling, softmax, batchnorm, and layernorm will use custom CUDA kernels with reduced performance.")
#endif
#include <cuda_runtime.h>
#include "cuda_stream_pool.hpp"
#include <array>
#include <stdexcept>
#include <limits>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <atomic>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace tenzor {

// ============================================================================
// Runtime cuDNN Availability Detection
// ============================================================================
namespace cuda {

// Cached cuDNN availability (computed once on first call)
static std::atomic<int> g_cudnn_available{-1};       // -1 = unknown, 0 = no, 1 = yes
static std::atomic<int> g_cudnn_frontend_available{-1};

bool is_cudnn_available() noexcept {
    int cached = g_cudnn_available.load(std::memory_order_acquire);
    if (cached >= 0) return cached == 1;

#ifdef TENZOR_HAS_CUDNN
    // cuDNN is linked at compile time
    g_cudnn_available.store(1, std::memory_order_release);
    return true;
#else
    // cuDNN not available
    g_cudnn_available.store(0, std::memory_order_release);
    return false;
#endif
}

bool is_cudnn_frontend_available() noexcept {
    int cached = g_cudnn_frontend_available.load(std::memory_order_acquire);
    if (cached >= 0) return cached == 1;

#ifdef TENZOR_HAS_CUDNN_FRONTEND
    // cuDNN Frontend is available
    g_cudnn_frontend_available.store(1, std::memory_order_release);
    return true;
#else
    // cuDNN Frontend not available
    g_cudnn_frontend_available.store(0, std::memory_order_release);
    return false;
#endif
}

} // namespace cuda

// Forward declarations for CUDA kernels
// These will be implemented by kernel developers in separate .cu files
namespace cuda {
    // Binary operations
    auto add_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto sub_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto mul_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto div_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto matmul_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;

    // In-place operations
    auto add_inplace_kernel(Tensor& inout, const Tensor& other, cudaStream_t stream) -> Tensor;
    auto sub_inplace_kernel(Tensor& inout, const Tensor& other, cudaStream_t stream) -> Tensor;
    auto mul_inplace_kernel(Tensor& inout, const Tensor& other, cudaStream_t stream) -> Tensor;
    auto div_inplace_kernel(Tensor& inout, const Tensor& other, cudaStream_t stream) -> Tensor;

    // Unary operations
    auto sqrt_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto neg_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto abs_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto sign_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto log_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto exp_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;

    // Trigonometric functions
    auto sin_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto cos_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto tan_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto asin_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto acos_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto atan_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto sinh_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto cosh_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;

    // Rounding functions
    auto ceil_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto floor_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto round_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto trunc_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto reciprocal_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;

    // Operations with parameters
    auto clamp_kernel(const Tensor& input, float min_val, float max_val, cudaStream_t stream) -> Tensor;
    auto clamp_min_kernel(const Tensor& input, float min_val, cudaStream_t stream) -> Tensor;
    auto clamp_max_kernel(const Tensor& input, float max_val, cudaStream_t stream) -> Tensor;
    auto pow_kernel(const Tensor& input, float exponent, cudaStream_t stream) -> Tensor;

    // Reduction operations
    auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto max_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto min_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto prod_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto var_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction, cudaStream_t stream) -> Tensor;
    auto std_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction, cudaStream_t stream) -> Tensor;
    auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;

    // Activation functions
    auto relu_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto sigmoid_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto swish_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto tanh_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto gelu_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto leaky_relu_kernel(const Tensor& input, float alpha, cudaStream_t stream) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, cudaStream_t stream) -> Tensor;
    auto elu_kernel(const Tensor& input, float alpha, cudaStream_t stream) -> Tensor;
    auto elu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, cudaStream_t stream) -> Tensor;
    auto selu_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto selu_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto mish_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto mish_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto softplus_kernel(const Tensor& input, float beta, float threshold, cudaStream_t stream) -> Tensor;
    auto softplus_backward_kernel(const Tensor& grad_output, const Tensor& input, float beta, float threshold, cudaStream_t stream) -> Tensor;

    // Softmax operations
    auto softmax_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, cudaStream_t stream) -> Tensor;
    auto log_softmax_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, cudaStream_t stream) -> Tensor;

    // Transform operations
    auto contiguous_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto clone_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, cudaStream_t stream) -> Tensor;
    auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, cudaStream_t stream) -> Tensor;
    auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, cudaStream_t stream) -> Tensor;
    auto squeeze_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto unsqueeze_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto expand_kernel(const Tensor& input, const std::vector<int64_t>& shape, void* stream) -> Tensor;
    auto repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats, cudaStream_t stream) -> Tensor;
    auto cat_kernel(std::span<const Tensor> tensors, int64_t dim, cudaStream_t stream) -> Tensor;

    // Fill operations
    auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto fill_kernel(const Tensor& tensor, double value, cudaStream_t stream) -> Tensor;

    // Random operations
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;

    // Sequence/identity creation operations
    auto arange_kernel(float start, float end, float step, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto linspace_kernel(float start, float end, int64_t steps, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto eye_kernel(int64_t n, int64_t m, DType dtype, Device device, cudaStream_t stream) -> Tensor;

    // Comparison operations
    auto eq_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto ne_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto lt_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto le_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto gt_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto ge_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto dot_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;

    // Indexing operations (native CUDA implementations - no CPU fallback)
    auto index_select_kernel(const Tensor& input, int64_t dim, const Tensor& index, cudaStream_t stream) -> Tensor;
    auto gather_kernel(const Tensor& input, int64_t dim, const Tensor& index, cudaStream_t stream) -> Tensor;
    auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src, cudaStream_t stream) -> Tensor;
    auto masked_select_kernel(const Tensor& input, const Tensor& mask, cudaStream_t stream) -> Tensor;
    auto masked_fill_kernel(const Tensor& input, const Tensor& mask, double value, cudaStream_t stream) -> Tensor;
    auto where_kernel(const Tensor& condition, const Tensor& x, const Tensor& y, cudaStream_t stream) -> Tensor;
    auto nonzero_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto one_hot_kernel(const Tensor& indices, int64_t num_classes, cudaStream_t stream) -> Tensor;

    // BatchNorm2d operations
    auto batchnorm2d_mean_var(const Tensor& input, Tensor& mean, Tensor& variance, cudaStream_t stream) -> void;
    auto batchnorm2d_forward(const Tensor& input, const Tensor& mean, const Tensor& variance, float epsilon, cudaStream_t stream) -> Tensor;
    auto batchnorm2d_forward_affine(const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, const Tensor& beta, float epsilon, cudaStream_t stream) -> Tensor;
    auto batchnorm2d_update_running_stats(Tensor& running_mean, Tensor& running_var, const Tensor& batch_mean, const Tensor& batch_var, float momentum, cudaStream_t stream) -> void;
    auto batchnorm2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, float epsilon, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;

    // Conv2d operations (custom kernels - fallback)
    auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto conv2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto conv_transpose2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    // M12 fix: per-axis overload.
    auto conv_transpose2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w, int64_t output_padding_h, int64_t output_padding_w, int64_t dilation_h, int64_t dilation_w, int64_t groups, cudaStream_t stream) -> Tensor;

    // Conv3d / ConvTranspose3d operations (custom kernels - fallback when cuDNN unavailable)
    auto conv3d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding, std::array<int64_t, 3> dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto conv3d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding, std::array<int64_t, 3> dilation, int64_t groups, bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto conv_transpose3d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto conv_transpose3d_backward_input_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto conv_transpose3d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;

    // cuDNN Conv2d operations (NHWC optimized for Tensor Cores)
    auto cudnn_conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto cudnn_conv2d_forward_nhwc(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, void* stream) -> Tensor;
    auto cudnn_conv2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto cudnn_conv2d_backward_nhwc(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias, void* stream) -> std::tuple<Tensor, Tensor, Tensor>;

    // LSTM operations (custom kernels - fallback)
    auto lstm_cell_forward_kernel(const Tensor& gates, const Tensor& c_prev, int64_t batch_size, int64_t hidden_size, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto lstm_cell_backward_kernel(const Tensor& grad_h, const Tensor& grad_c, const Tensor& gates, const Tensor& c_prev, const Tensor& c_out, int64_t batch_size, int64_t hidden_size, cudaStream_t stream) -> std::pair<Tensor, Tensor>;

    // Pooling operations
    auto adaptive_avg_pool2d_forward(const Tensor& input, int64_t output_h, int64_t output_w) -> Tensor;
    auto adaptive_avg_pool2d_backward(const Tensor& grad_output, int64_t H_in, int64_t W_in) -> Tensor;
    auto max_pool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding) -> std::pair<Tensor, Tensor>;
    auto max_pool2d_backward(const Tensor& grad_output, const Tensor& indices, int64_t H_in, int64_t W_in) -> Tensor;
    auto avg_pool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding) -> Tensor;
    auto avg_pool2d_backward(const Tensor& grad_output, int64_t H_in, int64_t W_in, int64_t kernel_size, int64_t stride, int64_t padding) -> Tensor;

    // Vision operations
    auto gather_relative_position_bias(const Tensor& table, const Tensor& indices, int64_t num_positions, int64_t num_heads) -> Tensor;

    // Fused operations
    auto fused_linear_relu_cuda(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor;
    auto fused_batchnorm_relu_cuda(const Tensor& input, const Tensor& running_mean, const Tensor& running_var, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto fused_softmax_cross_entropy_cuda(const Tensor& logits, const Tensor& targets, const std::string& reduction) -> Tensor;
    auto fused_add_relu_cuda(const Tensor& a, const Tensor& b) -> Tensor;
    auto fused_gelu_cuda(const Tensor& input) -> Tensor;
    auto fused_layer_norm_cuda(const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& weight, const Tensor& bias, float eps) -> std::tuple<Tensor, Tensor, Tensor>;
} // namespace cuda

class CUDABackend : public Backend {
public:
    CUDABackend() {
        // Caching allocator provides significant performance improvement (up to 2x faster for BMM)
        // Enabled by default. Disable with TENZOR_DISABLE_CACHING_ALLOCATOR=1 if issues arise.
        const char* disable_caching = std::getenv("TENZOR_DISABLE_CACHING_ALLOCATOR");
        use_caching_allocator_ = (disable_caching == nullptr || std::string(disable_caching) != "1");

        // Enable peer access between GPU pairs for fast D2D transfers
        init_peer_access();
    }

    auto name() const -> std::string_view override {
        return "cuda";
    }

    auto device_count() const -> int32_t override {
        int count = 0;
        cudaGetDeviceCount(&count);
        return count;
    }

    auto is_available() const -> bool override {
        return device_count() > 0;
    }

    auto set_device(int32_t device_id) -> void override {
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaSetDevice failed for device ") +
                std::to_string(device_id) + ": " + cudaGetErrorString(err));
        }
    }

    auto get_current_device() const -> int32_t override {
        int device_id = 0;
        cudaGetDevice(&device_id);
        return device_id;
    }

    auto get_device_info(int32_t device_id) const -> DeviceInfo override {
        int count = device_count();
        if (device_id < 0 || device_id >= count) {
            throw std::out_of_range("Invalid CUDA device ID: " + std::to_string(device_id) +
                                    " (available: 0-" + std::to_string(count - 1) + ")");
        }

        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);

        DeviceInfo info;
        info.name = props.name;
        info.vendor = "NVIDIA";

        // Get driver version
        int driver_version = 0;
        cudaDriverGetVersion(&driver_version);
        info.driver_version = std::to_string(driver_version / 1000) + "." +
                              std::to_string((driver_version % 1000) / 10);

        // Memory info — use DeviceGuard for exception-safe device switching
        info.total_memory = props.totalGlobalMem;
        size_t free_mem = 0, total_mem = 0;
        {
            DeviceGuard guard(Device::cuda(device_id));
            cudaMemGetInfo(&free_mem, &total_mem);
        }
        info.available_memory = free_mem;

        // Compute info
        info.compute_units = props.multiProcessorCount;
        info.max_threads_per_block = props.maxThreadsPerBlock;
        info.max_shared_memory = static_cast<int>(props.sharedMemPerBlock);
        info.warp_size = props.warpSize;

        // Compute capability
        info.major_version = props.major;
        info.minor_version = props.minor;

        // Feature support
        info.supports_fp16 = (props.major >= 6);  // Pascal+
        info.supports_fp64 = (props.major >= 2);  // Fermi+
        info.supports_int8 = (props.major >= 6 && props.minor >= 1);  // Tensor cores on Volta+

        // Device type
        info.is_integrated = (props.integrated != 0);
        info.is_discrete = !info.is_integrated;

        // PCI info
        info.pci_bus_id = props.pciBusID;
        info.pci_device_id = props.pciDeviceID;

        return info;
    }

    auto allocate(size_t bytes, int32_t device_id) -> void* override {
        // Handle empty tensors - CUDA doesn't like 0-byte allocations
        if (bytes == 0) {
            return nullptr;
        }

        // Unified (managed) memory path - accessible from both host and device
        if (use_unified_memory_) {
            void* ptr = nullptr;
            cudaError_t err = cudaMallocManaged(&ptr, bytes);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Failed to allocate managed memory: ") + cudaGetErrorString(err)
                );
            }
            // Set preferred location to the target device for better performance
            cudaMemLocation location{};
            location.type = cudaMemLocationTypeDevice;
            location.id = device_id;
            cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, location);
            // Prefetch to device for immediate use
            cudaMemPrefetchAsync(ptr, bytes, location, 0);

            {
                std::lock_guard<std::mutex> lock(ptr_device_mutex_);
                ptr_device_map_[ptr] = device_id;
                managed_ptrs_.insert(ptr);
            }
            return ptr;
        }

        if (use_caching_allocator_) {
            void* ptr = backend::CachingAllocator::get().allocate(bytes, device_id);
            {
                std::lock_guard<std::mutex> lock(ptr_device_mutex_);
                ptr_device_map_[ptr] = device_id;
            }
            return ptr;
        }

        void* ptr = nullptr;
        cudaSetDevice(device_id);
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to allocate device memory: ") + cudaGetErrorString(err)
            );
        }

        {
            std::lock_guard<std::mutex> lock(ptr_device_mutex_);
            ptr_device_map_[ptr] = device_id;
        }
        return ptr;
    }

    auto deallocate(void* ptr) -> void override {
        // Handle nullptr from empty tensor allocations
        if (ptr == nullptr) {
            return;
        }

        // Look up cached device_id and check if managed
        int device_id = 0;
        bool is_managed = false;
        {
            std::lock_guard<std::mutex> lock(ptr_device_mutex_);
            auto it = ptr_device_map_.find(ptr);
            if (it != ptr_device_map_.end()) {
                device_id = it->second;
                ptr_device_map_.erase(it);
            }
            auto mit = managed_ptrs_.find(ptr);
            if (mit != managed_ptrs_.end()) {
                is_managed = true;
                managed_ptrs_.erase(mit);
            }
        }

        // Managed memory uses cudaFree (same as device memory)
        if (is_managed) {
            cudaFree(ptr);
            return;
        }

        if (use_caching_allocator_) {
            backend::CachingAllocator::get().free(ptr, device_id);
            return;
        }

        cudaFree(ptr);
    }

    auto copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void override {
        // Handle empty tensors
        if (bytes == 0) {
            return;
        }

        cudaMemcpyKind cuda_kind;
        switch (kind) {
            case CopyKind::HostToHost: cuda_kind = cudaMemcpyHostToHost; break;
            case CopyKind::HostToDevice: cuda_kind = cudaMemcpyHostToDevice; break;
            case CopyKind::DeviceToHost: cuda_kind = cudaMemcpyDeviceToHost; break;
            case CopyKind::DeviceToDevice: cuda_kind = cudaMemcpyDeviceToDevice; break;
            default:
                throw std::runtime_error("Invalid CopyKind value");
        }

        cudaError_t err;
        if (kind == CopyKind::HostToDevice || kind == CopyKind::DeviceToDevice) {
            // Ensure the correct device is selected before async transfer.
            // Without this, multi-GPU setups may use the wrong device's default stream.
            cudaPointerAttributes dst_attrs;
            cudaPointerAttributes src_attrs;
            int dst_device = -1, src_device = -1;
            if (cudaPointerGetAttributes(&dst_attrs, dst) == cudaSuccess && dst_attrs.device >= 0) {
                dst_device = dst_attrs.device;
                cudaSetDevice(dst_device);
            }
            if (kind == CopyKind::DeviceToDevice) {
                if (cudaPointerGetAttributes(&src_attrs, src) == cudaSuccess && src_attrs.device >= 0) {
                    src_device = src_attrs.device;
                }
                // Use cudaMemcpyPeerAsync when devices differ and P2P is enabled
                if (src_device >= 0 && dst_device >= 0 && src_device != dst_device &&
                    has_peer_access(src_device, dst_device)) {
                    err = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, cudaStreamPerThread);
                    if (err != cudaSuccess) {
                        throw std::runtime_error(
                            std::string("CUDA peer copy failed: ") + cudaGetErrorString(err));
                    }
                    return;
                }
            }
            // Use async transfer for H2D and D2D on the default (legacy) stream.
            // This ensures ordering with kernel launches which also default to
            // stream 0, preventing races where a kernel reads input data before
            // the H2D copy completes.
            err = cudaMemcpyAsync(dst, src, bytes, cuda_kind, nullptr);
        } else {
            // D2H and H2H: use synchronous copy since host needs data immediately
            err = cudaMemcpy(dst, src, bytes, cuda_kind);
        }
        if (err != cudaSuccess) {
            // Debug: check pointer attributes
            cudaPointerAttributes dst_attrs, src_attrs;
            cudaPointerGetAttributes(&dst_attrs, dst);
            cudaPointerGetAttributes(&src_attrs, src);
            std::cerr << "[COPY ERROR] dst=" << dst << " src=" << src << " bytes=" << bytes
                      << " kind=" << static_cast<int>(kind)
                      << " dst_type=" << dst_attrs.type << " src_type=" << src_attrs.type
                      << std::endl;
            throw std::runtime_error(
                std::string("CUDA copy failed: ") + cudaGetErrorString(err)
            );
        }
    }

    auto synchronize(int32_t device_id) -> void override {
        cudaSetDevice(device_id);
        cudaDeviceSynchronize();
    }

    auto create_stream(int32_t device_id) -> StreamHandle override {
        auto stream = cuda::CUDAStreamPool::instance().acquire(device_id);
        return static_cast<StreamHandle>(stream);
    }

    auto destroy_stream(StreamHandle stream) -> void override {
        // Return stream to pool instead of destroying it
        int device_id = 0;
        cudaGetDevice(&device_id);
        cuda::CUDAStreamPool::instance().release(device_id, static_cast<cudaStream_t>(stream));
    }

    auto synchronize_stream(StreamHandle stream) -> void override {
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    }

    auto create_event(int32_t device_id, bool enable_timing = true) -> EventHandle override {
        cudaSetDevice(device_id);
        cudaEvent_t event;
        unsigned flags = enable_timing ? cudaEventDefault : cudaEventDisableTiming;
        cudaError_t err = cudaEventCreateWithFlags(&event, flags);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaEventCreate failed: ") + cudaGetErrorString(err));
        }
        return static_cast<EventHandle>(event);
    }

    auto destroy_event(EventHandle event) -> void override {
        if (event) {
            cudaEventDestroy(static_cast<cudaEvent_t>(event));
        }
    }

    auto record_event(EventHandle event, StreamHandle stream = nullptr) -> void override {
        cudaError_t err = cudaEventRecord(
            static_cast<cudaEvent_t>(event),
            static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaEventRecord failed: ") + cudaGetErrorString(err));
        }
    }

    auto wait_event(EventHandle event, StreamHandle stream = nullptr) -> void override {
        cudaError_t err = cudaStreamWaitEvent(
            static_cast<cudaStream_t>(stream),
            static_cast<cudaEvent_t>(event),
            0);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamWaitEvent failed: ") + cudaGetErrorString(err));
        }
    }

    auto event_elapsed_ms(EventHandle start_event, EventHandle end_event) -> float override {
        // Ensure both events have completed
        cudaEventSynchronize(static_cast<cudaEvent_t>(end_event));
        float ms = 0.0f;
        cudaError_t err = cudaEventElapsedTime(
            &ms,
            static_cast<cudaEvent_t>(start_event),
            static_cast<cudaEvent_t>(end_event));
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaEventElapsedTime failed: ") + cudaGetErrorString(err));
        }
        return ms;
    }

    auto memset(void* ptr, int value, size_t bytes, int32_t device_id) -> void override {
        cudaSetDevice(device_id);
        cudaError_t err = cudaMemset(ptr, value, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemset failed: ") + cudaGetErrorString(err));
        }
    }

    // Legacy string-keyed dispatch removed (audit Phase C).

public:
    /// Enable/disable unified (managed) memory allocation
    void set_unified_memory(bool enabled) { use_unified_memory_ = enabled; }
    bool unified_memory_enabled() const { return use_unified_memory_; }

    /// Prefetch a managed allocation to the specified device
    void prefetch_to_device(void* ptr, size_t bytes, int device_id) {
        if (ptr && bytes > 0) {
            cudaMemLocation location;
            location.type = cudaMemLocationTypeDevice;
            location.id = device_id;
            cudaMemPrefetchAsync(ptr, bytes, location, 0);
        }
    }

private:
    bool use_caching_allocator_{false};
    bool use_unified_memory_{false};
    // Cache pointer→device_id mapping to avoid cudaPointerGetAttributes() in deallocate()
    std::mutex ptr_device_mutex_;
    std::unordered_map<void*, int> ptr_device_map_;
    // Track which pointers are managed memory (for deallocate routing)
    std::unordered_set<void*> managed_ptrs_;

    // Peer access matrix: peer_access_[i][j] = true means GPU i can access GPU j directly
    std::vector<std::vector<bool>> peer_access_;

    void init_peer_access() {
        int count = 0;
        cudaGetDeviceCount(&count);
        if (count <= 1) return;

        peer_access_.resize(count, std::vector<bool>(count, false));

        int saved_device = 0;
        cudaGetDevice(&saved_device);

        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < count; ++j) {
                if (i == j) continue;
                int can_access = 0;
                cudaDeviceCanAccessPeer(&can_access, i, j);
                if (can_access) {
                    cudaSetDevice(i);
                    cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                    if (err == cudaSuccess) {
                        peer_access_[i][j] = true;
                    } else if (err == cudaErrorPeerAccessAlreadyEnabled) {
                        peer_access_[i][j] = true;
                        cudaGetLastError();  // Clear the sticky error
                    }
                }
            }
        }

        cudaSetDevice(saved_device);
    }

    bool has_peer_access(int src_device, int dst_device) const {
        if (src_device < 0 || dst_device < 0) return false;
        if (static_cast<size_t>(src_device) >= peer_access_.size() ||
            static_cast<size_t>(dst_device) >= peer_access_.size()) return false;
        return peer_access_[src_device][dst_device];
    }
};

extern "C" {
    Backend* create_backend() {
        return new CUDABackend();
    }
}

} // namespace tenzor
