#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/ops/creation.hpp"
#ifdef TENZOR_HAS_CUDNN
#include "tenzor/backend/cudnn_wrapper.hpp"
#endif
#include <cuda_runtime.h>
#include <stdexcept>
#include <limits>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <atomic>

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
    auto fill_kernel(const Tensor& tensor, float value, cudaStream_t stream) -> Tensor;

    // Random operations
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;

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

    // Fused operations (disabled - CUDA kernels not yet implemented)
    // auto fused_linear_relu_cuda(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor;
    // auto fused_batchnorm_relu_cuda(const Tensor& input, const Tensor& running_mean, const Tensor& running_var, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    // auto fused_softmax_cross_entropy_cuda(const Tensor& logits, const Tensor& targets, const std::string& reduction) -> Tensor;
    // auto fused_add_relu_cuda(const Tensor& a, const Tensor& b) -> Tensor;
    // auto fused_gelu_cuda(const Tensor& input) -> Tensor;
    // auto fused_layer_norm_cuda(const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
} // namespace cuda

class CUDABackend : public Backend {
public:
    CUDABackend() {
        // Caching allocator provides significant performance improvement (up to 2x faster for BMM)
        // Enabled by default. Disable with TENZOR_DISABLE_CACHING_ALLOCATOR=1 if issues arise.
        const char* disable_caching = std::getenv("TENZOR_DISABLE_CACHING_ALLOCATOR");
        use_caching_allocator_ = (disable_caching == nullptr || std::string(disable_caching) != "1");
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

        // Memory info
        info.total_memory = props.totalGlobalMem;
        size_t free_mem = 0, total_mem = 0;
        int current_device;
        cudaGetDevice(&current_device);
        cudaSetDevice(device_id);
        cudaMemGetInfo(&free_mem, &total_mem);
        cudaSetDevice(current_device);
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

        if (use_caching_allocator_) {
            return backend::CachingAllocator::get().allocate(bytes, device_id);
        }

        void* ptr = nullptr;
        cudaSetDevice(device_id);
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to allocate device memory: ") + cudaGetErrorString(err)
            );
        }

        return ptr;
    }

    auto deallocate(void* ptr) -> void override {
        // Handle nullptr from empty tensor allocations
        if (ptr == nullptr) {
            return;
        }

        if (use_caching_allocator_) {
            // Note: we don't know the device_id here, but CachingAllocator tracks it
            // For proper integration, we'd need to look up the device from the pointer
            int device_id = 0;
            cudaPointerAttributes attrs;
            if (cudaPointerGetAttributes(&attrs, ptr) == cudaSuccess) {
                device_id = attrs.device;
            }
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
        }

        cudaError_t err = cudaMemcpy(dst, src, bytes, cuda_kind);
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
        cudaStream_t stream;
        cudaSetDevice(device_id);
        cudaStreamCreate(&stream);
        return static_cast<StreamHandle>(stream);
    }

    auto destroy_stream(StreamHandle stream) -> void override {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    }

    auto synchronize_stream(StreamHandle stream) -> void override {
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    }

    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor> override {
        // Allow empty inputs for creation operations
        bool is_creation_op = (op_name == "zeros" || op_name == "ones" || op_name == "full" ||
                               op_name == "rand" || op_name == "randn");

        // Validate we have inputs (except for creation operations)
        if (inputs.empty() && !is_creation_op) {
            throw std::invalid_argument("dispatch requires at least one input tensor");
        }

        // Validate all inputs are on CUDA device (if any)
        for (const auto& tensor : inputs) {
            if (tensor.device().type != Device::Type::CUDA) {
                throw std::runtime_error(
                    "CUDABackend: All input tensors must be on CUDA device, got: " +
                    tensor.device().to_string()
                );
            }
        }

        // Set CUDA device - use first tensor's device or device from attrs
        int32_t device_id = 0;
        if (!inputs.empty()) {
            device_id = inputs[0].device().index;
        } else if (attrs.contains("device_id")) {
            device_id = std::stoi(attrs.at("device_id"));
        }
        cudaSetDevice(device_id);

        // Get or create stream (nullptr means default stream)
        cudaStream_t stream = nullptr;
        if (attrs.contains("stream")) {
            stream = static_cast<cudaStream_t>(
                reinterpret_cast<void*>(std::stoull(attrs.at("stream")))
            );
        }

        // Dispatch to appropriate CUDA kernel based on operation name
        try {
            if (op_name == "add") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("add operation requires exactly 2 inputs");
                }
                return {cuda::add_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "add_inplace") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("add_inplace operation requires exactly 2 inputs");
                }
                // Make a mutable copy of the first input
                Tensor result = inputs[0];
                return {cuda::add_inplace_kernel(result, inputs[1], stream)};
            }
            else if (op_name == "sub") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("sub operation requires exactly 2 inputs");
                }
                return {cuda::sub_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "sub_inplace") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("sub_inplace operation requires exactly 2 inputs");
                }
                Tensor result = inputs[0];
                return {cuda::sub_inplace_kernel(result, inputs[1], stream)};
            }
            else if (op_name == "mul") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("mul operation requires exactly 2 inputs");
                }
                return {cuda::mul_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "mul_inplace") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("mul_inplace operation requires exactly 2 inputs");
                }
                Tensor result = inputs[0];
                return {cuda::mul_inplace_kernel(result, inputs[1], stream)};
            }
            else if (op_name == "div") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("div operation requires exactly 2 inputs");
                }
                return {cuda::div_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "div_inplace") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("div_inplace operation requires exactly 2 inputs");
                }
                Tensor result = inputs[0];
                return {cuda::div_inplace_kernel(result, inputs[1], stream)};
            }
            else if (op_name == "matmul") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("matmul operation requires exactly 2 inputs");
                }
                return {cuda::matmul_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "sum") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("sum operation requires exactly 1 input");
                }
                // Parse attributes
                int64_t dim = INT64_MIN;  // INT64_MIN means full reduction (all dimensions)
                bool keepdim = false;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                if (attrs.contains("keepdim")) {
                    keepdim = (attrs.at("keepdim") == "1");
                }
                return {cuda::sum_kernel(inputs[0], dim, keepdim, stream)};
            }
            else if (op_name == "mean") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("mean operation requires exactly 1 input");
                }
                int64_t dim = INT64_MIN;  // INT64_MIN means full reduction (all dimensions)
                bool keepdim = false;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                if (attrs.contains("keepdim")) {
                    keepdim = (attrs.at("keepdim") == "1");
                }
                return {cuda::mean_kernel(inputs[0], dim, keepdim, stream)};
            }
            else if (op_name == "max") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("max operation requires exactly 1 input");
                }
                int64_t dim = -1;
                bool keepdim = false;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                if (attrs.contains("keepdim")) {
                    keepdim = (attrs.at("keepdim") == "1");
                }
                return {cuda::max_kernel(inputs[0], dim, keepdim, stream)};
            }
            else if (op_name == "min") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("min operation requires exactly 1 input");
                }
                int64_t dim = -1;
                bool keepdim = false;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                if (attrs.contains("keepdim")) {
                    keepdim = (attrs.at("keepdim") == "1");
                }
                return {cuda::min_kernel(inputs[0], dim, keepdim, stream)};
            }
            else if (op_name == "argmax") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("argmax operation requires exactly 1 input");
                }
                int64_t dim = -1;
                bool keepdim = false;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                if (attrs.contains("keepdim")) {
                    keepdim = (attrs.at("keepdim") == "1");
                }
                return {cuda::argmax_kernel(inputs[0], dim, keepdim, stream)};
            }
            else if (op_name == "argmin") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("argmin operation requires exactly 1 input");
                }
                int64_t dim = -1;
                bool keepdim = false;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                if (attrs.contains("keepdim")) {
                    keepdim = (attrs.at("keepdim") == "1");
                }
                return {cuda::argmin_kernel(inputs[0], dim, keepdim, stream)};
            }
            else if (op_name == "prod") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("prod operation requires exactly 1 input");
                }
                int64_t dim = INT64_MIN;  // INT64_MIN means full reduction (all dimensions)
                bool keepdim = false;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                if (attrs.contains("keepdim")) {
                    keepdim = (attrs.at("keepdim") == "1");
                }
                return {cuda::prod_kernel(inputs[0], dim, keepdim, stream)};
            }
            else if (op_name == "var") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("var operation requires exactly 1 input");
                }
                int64_t dim = -1;
                bool keepdim = false;
                int64_t correction = 1;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                if (attrs.contains("keepdim")) {
                    keepdim = (attrs.at("keepdim") == "1");
                }
                if (attrs.contains("correction")) {
                    correction = std::stoll(attrs.at("correction"));
                }
                return {cuda::var_kernel(inputs[0], dim, keepdim, correction, stream)};
            }
            else if (op_name == "std") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("std operation requires exactly 1 input");
                }
                int64_t dim = -1;
                bool keepdim = false;
                int64_t correction = 1;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                if (attrs.contains("keepdim")) {
                    keepdim = (attrs.at("keepdim") == "1");
                }
                if (attrs.contains("correction")) {
                    correction = std::stoll(attrs.at("correction"));
                }
                return {cuda::std_kernel(inputs[0], dim, keepdim, correction, stream)};
            }
            else if (op_name == "norm") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("norm operation requires exactly 1 input");
                }
                float p = 2.0f;
                int64_t dim = -1;
                bool keepdim = false;
                if (attrs.contains("p")) {
                    p = std::stof(attrs.at("p"));
                }
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                if (attrs.contains("keepdim")) {
                    keepdim = (attrs.at("keepdim") == "1");
                }
                return {cuda::norm_kernel(inputs[0], p, dim, keepdim, stream)};
            }
            else if (op_name == "sqrt") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("sqrt operation requires exactly 1 input");
                }
                return {cuda::sqrt_kernel(inputs[0], stream)};
            }
            else if (op_name == "neg") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("neg operation requires exactly 1 input");
                }
                return {cuda::neg_kernel(inputs[0], stream)};
            }
            else if (op_name == "abs") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("abs operation requires exactly 1 input");
                }
                return {cuda::abs_kernel(inputs[0], stream)};
            }
            else if (op_name == "sign") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("sign operation requires exactly 1 input");
                }
                return {cuda::sign_kernel(inputs[0], stream)};
            }
            else if (op_name == "clamp") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("clamp operation requires exactly 1 input");
                }
                // Parse min and max from attributes
                float min_val = -std::numeric_limits<float>::infinity();
                float max_val = std::numeric_limits<float>::infinity();
                if (attrs.contains("min")) {
                    min_val = std::stof(attrs.at("min"));
                }
                if (attrs.contains("max")) {
                    max_val = std::stof(attrs.at("max"));
                }
                return {cuda::clamp_kernel(inputs[0], min_val, max_val, stream)};
            }
            else if (op_name == "log") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("log operation requires exactly 1 input");
                }
                return {cuda::log_kernel(inputs[0], stream)};
            }
            else if (op_name == "exp") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("exp operation requires exactly 1 input");
                }
                return {cuda::exp_kernel(inputs[0], stream)};
            }
            // Trigonometric functions
            else if (op_name == "sin") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("sin operation requires exactly 1 input");
                }
                return {cuda::sin_kernel(inputs[0], stream)};
            }
            else if (op_name == "cos") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("cos operation requires exactly 1 input");
                }
                return {cuda::cos_kernel(inputs[0], stream)};
            }
            else if (op_name == "tan") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("tan operation requires exactly 1 input");
                }
                return {cuda::tan_kernel(inputs[0], stream)};
            }
            else if (op_name == "asin") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("asin operation requires exactly 1 input");
                }
                return {cuda::asin_kernel(inputs[0], stream)};
            }
            else if (op_name == "acos") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("acos operation requires exactly 1 input");
                }
                return {cuda::acos_kernel(inputs[0], stream)};
            }
            else if (op_name == "atan") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("atan operation requires exactly 1 input");
                }
                return {cuda::atan_kernel(inputs[0], stream)};
            }
            else if (op_name == "sinh") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("sinh operation requires exactly 1 input");
                }
                return {cuda::sinh_kernel(inputs[0], stream)};
            }
            else if (op_name == "cosh") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("cosh operation requires exactly 1 input");
                }
                return {cuda::cosh_kernel(inputs[0], stream)};
            }
            // Rounding functions
            else if (op_name == "ceil") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("ceil operation requires exactly 1 input");
                }
                return {cuda::ceil_kernel(inputs[0], stream)};
            }
            else if (op_name == "floor") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("floor operation requires exactly 1 input");
                }
                return {cuda::floor_kernel(inputs[0], stream)};
            }
            else if (op_name == "round") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("round operation requires exactly 1 input");
                }
                return {cuda::round_kernel(inputs[0], stream)};
            }
            else if (op_name == "trunc") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("trunc operation requires exactly 1 input");
                }
                return {cuda::trunc_kernel(inputs[0], stream)};
            }
            else if (op_name == "reciprocal") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("reciprocal operation requires exactly 1 input");
                }
                return {cuda::reciprocal_kernel(inputs[0], stream)};
            }
            else if (op_name == "clamp_min") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("clamp_min operation requires exactly 1 input");
                }
                float min_val = -std::numeric_limits<float>::infinity();
                if (attrs.contains("min")) {
                    min_val = std::stof(attrs.at("min"));
                }
                return {cuda::clamp_min_kernel(inputs[0], min_val, stream)};
            }
            else if (op_name == "clamp_max") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("clamp_max operation requires exactly 1 input");
                }
                float max_val = std::numeric_limits<float>::infinity();
                if (attrs.contains("max")) {
                    max_val = std::stof(attrs.at("max"));
                }
                return {cuda::clamp_max_kernel(inputs[0], max_val, stream)};
            }
            else if (op_name == "pow") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("pow operation requires exactly 1 input");
                }
                float exponent = 2.0f;
                if (attrs.contains("exponent")) {
                    exponent = std::stof(attrs.at("exponent"));
                }
                return {cuda::pow_kernel(inputs[0], exponent, stream)};
            }
            else if (op_name == "relu") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("relu operation requires exactly 1 input");
                }
                return {cuda::relu_kernel(inputs[0], stream)};
            }
            else if (op_name == "relu_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("relu_backward operation requires exactly 2 inputs");
                }
                return {cuda::relu_backward_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "sigmoid") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("sigmoid operation requires exactly 1 input");
                }
                return {cuda::sigmoid_kernel(inputs[0], stream)};
            }
            else if (op_name == "sigmoid_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("sigmoid_backward operation requires exactly 2 inputs");
                }
                return {cuda::sigmoid_backward_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "swish") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("swish operation requires exactly 1 input");
                }
                return {cuda::swish_kernel(inputs[0], stream)};
            }
            else if (op_name == "tanh") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("tanh operation requires exactly 1 input");
                }
                return {cuda::tanh_kernel(inputs[0], stream)};
            }
            else if (op_name == "tanh_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("tanh_backward operation requires exactly 2 inputs");
                }
                return {cuda::tanh_backward_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "gelu") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("gelu operation requires exactly 1 input");
                }
                return {cuda::gelu_kernel(inputs[0], stream)};
            }
            else if (op_name == "gelu_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("gelu_backward operation requires exactly 2 inputs");
                }
                return {cuda::gelu_backward_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "leaky_relu") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("leaky_relu operation requires exactly 1 input");
                }
                float alpha = 0.01f;
                if (attrs.contains("alpha")) {
                    alpha = std::stof(attrs.at("alpha"));
                }
                return {cuda::leaky_relu_kernel(inputs[0], alpha, stream)};
            }
            else if (op_name == "leaky_relu_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("leaky_relu_backward operation requires exactly 2 inputs");
                }
                float alpha = 0.01f;
                if (attrs.contains("alpha")) {
                    alpha = std::stof(attrs.at("alpha"));
                }
                return {cuda::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha, stream)};
            }
            else if (op_name == "elu") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("elu operation requires exactly 1 input");
                }
                float alpha = 1.0f;
                if (attrs.contains("alpha")) {
                    alpha = std::stof(attrs.at("alpha"));
                }
                return {cuda::elu_kernel(inputs[0], alpha, stream)};
            }
            else if (op_name == "elu_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("elu_backward operation requires exactly 2 inputs");
                }
                float alpha = 1.0f;
                if (attrs.contains("alpha")) {
                    alpha = std::stof(attrs.at("alpha"));
                }
                return {cuda::elu_backward_kernel(inputs[0], inputs[1], alpha, stream)};
            }
            else if (op_name == "selu") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("selu operation requires exactly 1 input");
                }
                return {cuda::selu_kernel(inputs[0], stream)};
            }
            else if (op_name == "selu_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("selu_backward operation requires exactly 2 inputs");
                }
                return {cuda::selu_backward_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "mish") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("mish operation requires exactly 1 input");
                }
                return {cuda::mish_kernel(inputs[0], stream)};
            }
            else if (op_name == "mish_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("mish_backward operation requires exactly 2 inputs");
                }
                return {cuda::mish_backward_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "softplus") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("softplus operation requires exactly 1 input");
                }
                float beta = 1.0f;
                float threshold = 20.0f;
                if (attrs.contains("beta")) {
                    beta = std::stof(attrs.at("beta"));
                }
                if (attrs.contains("threshold")) {
                    threshold = std::stof(attrs.at("threshold"));
                }
                return {cuda::softplus_kernel(inputs[0], beta, threshold, stream)};
            }
            else if (op_name == "softplus_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("softplus_backward operation requires exactly 2 inputs");
                }
                float beta = 1.0f;
                float threshold = 20.0f;
                if (attrs.contains("beta")) {
                    beta = std::stof(attrs.at("beta"));
                }
                if (attrs.contains("threshold")) {
                    threshold = std::stof(attrs.at("threshold"));
                }
                return {cuda::softplus_backward_kernel(inputs[0], inputs[1], beta, threshold, stream)};
            }
            else if (op_name == "softmax") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("softmax operation requires exactly 1 input");
                }
                int64_t dim = -1;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::softmax_kernel(inputs[0], dim, stream)};
            }
            else if (op_name == "softmax_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("softmax_backward operation requires exactly 2 inputs");
                }
                int64_t dim = -1;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::softmax_backward_kernel(inputs[0], inputs[1], dim, stream)};
            }
            else if (op_name == "log_softmax") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("log_softmax operation requires exactly 1 input");
                }
                int64_t dim = -1;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::log_softmax_kernel(inputs[0], dim, stream)};
            }
            else if (op_name == "log_softmax_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("log_softmax_backward operation requires exactly 2 inputs");
                }
                int64_t dim = -1;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::log_softmax_backward_kernel(inputs[0], inputs[1], dim, stream)};
            }
            else if (op_name == "zeros") {
                // Parse shape from attributes
                if (!attrs.contains("shape")) {
                    throw std::invalid_argument("zeros operation requires 'shape' attribute");
                }
                // Shape is stored as comma-separated values
                std::vector<int64_t> shape;
                std::string shape_str = attrs.at("shape");
                size_t pos = 0;
                while ((pos = shape_str.find(',')) != std::string::npos) {
                    shape.push_back(std::stoll(shape_str.substr(0, pos)));
                    shape_str.erase(0, pos + 1);
                }
                if (!shape_str.empty()) {
                    shape.push_back(std::stoll(shape_str));
                }

                DType dtype = DType::Float32;
                if (attrs.contains("dtype")) {
                    // Parse dtype from string
                    auto dtype_str = attrs.at("dtype");
                    if (dtype_str == "float32") dtype = DType::Float32;
                    else if (dtype_str == "float64") dtype = DType::Float64;
                    else if (dtype_str == "float16") dtype = DType::Float16;
                    else if (dtype_str == "bfloat16") dtype = DType::BFloat16;
                    else if (dtype_str == "int8") dtype = DType::Int8;
                    else if (dtype_str == "int16") dtype = DType::Int16;
                    else if (dtype_str == "int32") dtype = DType::Int32;
                    else if (dtype_str == "int64") dtype = DType::Int64;
                    else if (dtype_str == "uint8") dtype = DType::UInt8;
                    else if (dtype_str == "uint16") dtype = DType::UInt16;
                    else if (dtype_str == "uint32") dtype = DType::UInt32;
                    else if (dtype_str == "uint64") dtype = DType::UInt64;
                    else if (dtype_str == "bool") dtype = DType::Bool;
                }

                Device device = inputs.empty() ? Device::cuda(0) : inputs[0].device();
                return {cuda::zeros_kernel(shape, dtype, device, stream)};
            }
            else if (op_name == "ones") {
                if (!attrs.contains("shape")) {
                    throw std::invalid_argument("ones operation requires 'shape' attribute");
                }
                std::vector<int64_t> shape;
                std::string shape_str = attrs.at("shape");
                size_t pos = 0;
                while ((pos = shape_str.find(',')) != std::string::npos) {
                    shape.push_back(std::stoll(shape_str.substr(0, pos)));
                    shape_str.erase(0, pos + 1);
                }
                if (!shape_str.empty()) {
                    shape.push_back(std::stoll(shape_str));
                }

                DType dtype = DType::Float32;
                if (attrs.contains("dtype")) {
                    auto dtype_str = attrs.at("dtype");
                    if (dtype_str == "float32") dtype = DType::Float32;
                    else if (dtype_str == "float64") dtype = DType::Float64;
                    else if (dtype_str == "float16") dtype = DType::Float16;
                    else if (dtype_str == "bfloat16") dtype = DType::BFloat16;
                    else if (dtype_str == "int8") dtype = DType::Int8;
                    else if (dtype_str == "int16") dtype = DType::Int16;
                    else if (dtype_str == "int32") dtype = DType::Int32;
                    else if (dtype_str == "int64") dtype = DType::Int64;
                    else if (dtype_str == "uint8") dtype = DType::UInt8;
                    else if (dtype_str == "uint16") dtype = DType::UInt16;
                    else if (dtype_str == "uint32") dtype = DType::UInt32;
                    else if (dtype_str == "uint64") dtype = DType::UInt64;
                    else if (dtype_str == "bool") dtype = DType::Bool;
                }

                Device device = inputs.empty() ? Device::cuda(0) : inputs[0].device();
                return {cuda::ones_kernel(shape, dtype, device, stream)};
            }
            else if (op_name == "full") {
                if (!attrs.contains("shape")) {
                    throw std::invalid_argument("full operation requires 'shape' attribute");
                }
                if (!attrs.contains("value")) {
                    throw std::invalid_argument("full operation requires 'value' attribute");
                }

                std::vector<int64_t> shape;
                std::string shape_str = attrs.at("shape");
                size_t pos = 0;
                while ((pos = shape_str.find(',')) != std::string::npos) {
                    shape.push_back(std::stoll(shape_str.substr(0, pos)));
                    shape_str.erase(0, pos + 1);
                }
                if (!shape_str.empty()) {
                    shape.push_back(std::stoll(shape_str));
                }

                DType dtype = DType::Float32;
                if (attrs.contains("dtype")) {
                    auto dtype_str = attrs.at("dtype");
                    if (dtype_str == "float32") dtype = DType::Float32;
                    else if (dtype_str == "float64") dtype = DType::Float64;
                    else if (dtype_str == "float16") dtype = DType::Float16;
                    else if (dtype_str == "bfloat16") dtype = DType::BFloat16;
                    else if (dtype_str == "int32") dtype = DType::Int32;
                    else if (dtype_str == "int64") dtype = DType::Int64;
                    else if (dtype_str == "int8") dtype = DType::Int8;
                    else if (dtype_str == "int16") dtype = DType::Int16;
                    else if (dtype_str == "uint8") dtype = DType::UInt8;
                    else if (dtype_str == "uint16") dtype = DType::UInt16;
                    else if (dtype_str == "uint32") dtype = DType::UInt32;
                    else if (dtype_str == "uint64") dtype = DType::UInt64;
                    else if (dtype_str == "bool") dtype = DType::Bool;
                }

                // Parse value as double for full precision, then convert based on target dtype
                // For integer types, parse through double and cast directly to avoid float precision loss
                double value_d = std::stod(attrs.at("value"));
                float value;
                if (dtype == DType::Int32 || dtype == DType::Int64 || dtype == DType::Int8 ||
                    dtype == DType::Int16 || dtype == DType::UInt8 || dtype == DType::UInt16 ||
                    dtype == DType::UInt32 || dtype == DType::UInt64) {
                    // For integer types, round to nearest integer to preserve exact values
                    value = static_cast<float>(std::llround(value_d));
                } else {
                    value = static_cast<float>(value_d);
                }

                Device device = inputs.empty() ? Device::cuda(0) : inputs[0].device();
                return {cuda::full_kernel(shape, value, dtype, device, stream)};
            }
            else if (op_name == "fill") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("fill operation requires exactly 1 input");
                }
                if (!attrs.contains("value")) {
                    throw std::invalid_argument("fill operation requires 'value' attribute");
                }
                float value = std::stof(attrs.at("value"));
                return {cuda::fill_kernel(inputs[0], value, stream)};
            }
            else if (op_name == "expand") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("expand operation requires exactly 1 input");
                }
                if (!attrs.contains("shape")) {
                    throw std::invalid_argument("expand operation requires 'shape' attribute");
                }

                std::vector<int64_t> shape;
                std::string shape_str = attrs.at("shape");
                size_t pos = 0;
                while ((pos = shape_str.find(',')) != std::string::npos) {
                    shape.push_back(std::stoll(shape_str.substr(0, pos)));
                    shape_str.erase(0, pos + 1);
                }
                if (!shape_str.empty()) {
                    shape.push_back(std::stoll(shape_str));
                }

                return {cuda::expand_kernel(inputs[0], shape, static_cast<void*>(stream))};
            }
            else if (op_name == "repeat") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("repeat operation requires exactly 1 input");
                }
                if (!attrs.contains("repeats")) {
                    throw std::invalid_argument("repeat operation requires 'repeats' attribute");
                }

                std::vector<int64_t> repeats;
                std::string repeats_str = attrs.at("repeats");
                size_t pos = 0;
                while ((pos = repeats_str.find(',')) != std::string::npos) {
                    repeats.push_back(std::stoll(repeats_str.substr(0, pos)));
                    repeats_str.erase(0, pos + 1);
                }
                if (!repeats_str.empty()) {
                    repeats.push_back(std::stoll(repeats_str));
                }
                return {cuda::repeat_kernel(inputs[0], repeats, stream)};
            }
            else if (op_name == "rand") {
                if (!attrs.contains("shape")) {
                    throw std::invalid_argument("rand operation requires 'shape' attribute");
                }

                std::vector<int64_t> shape;
                std::string shape_str = attrs.at("shape");
                size_t pos = 0;
                while ((pos = shape_str.find(',')) != std::string::npos) {
                    shape.push_back(std::stoll(shape_str.substr(0, pos)));
                    shape_str.erase(0, pos + 1);
                }
                if (!shape_str.empty()) {
                    shape.push_back(std::stoll(shape_str));
                }

                DType dtype = DType::Float32;
                if (attrs.contains("dtype")) {
                    auto dtype_str = attrs.at("dtype");
                    if (dtype_str == "float32") dtype = DType::Float32;
                    else if (dtype_str == "float64") dtype = DType::Float64;
                    else if (dtype_str == "float16") dtype = DType::Float16;
                    else if (dtype_str == "bfloat16") dtype = DType::BFloat16;
                    else if (dtype_str == "int32") dtype = DType::Int32;
                    else if (dtype_str == "int64") dtype = DType::Int64;
                }

                Device device = inputs.empty() ? Device::cuda(0) : inputs[0].device();
                return {cuda::rand_kernel(shape, dtype, device, stream)};
            }
            else if (op_name == "randn") {
                if (!attrs.contains("shape")) {
                    throw std::invalid_argument("randn operation requires 'shape' attribute");
                }

                std::vector<int64_t> shape;
                std::string shape_str = attrs.at("shape");
                size_t pos = 0;
                while ((pos = shape_str.find(',')) != std::string::npos) {
                    shape.push_back(std::stoll(shape_str.substr(0, pos)));
                    shape_str.erase(0, pos + 1);
                }
                if (!shape_str.empty()) {
                    shape.push_back(std::stoll(shape_str));
                }

                DType dtype = DType::Float32;
                if (attrs.contains("dtype")) {
                    auto dtype_str = attrs.at("dtype");
                    if (dtype_str == "float32") dtype = DType::Float32;
                    else if (dtype_str == "float64") dtype = DType::Float64;
                    else if (dtype_str == "float16") dtype = DType::Float16;
                    else if (dtype_str == "bfloat16") dtype = DType::BFloat16;
                    else if (dtype_str == "int32") dtype = DType::Int32;
                    else if (dtype_str == "int64") dtype = DType::Int64;
                }

                Device device = inputs.empty() ? Device::cuda(0) : inputs[0].device();
                return {cuda::randn_kernel(shape, dtype, device, stream)};
            }
            else if (op_name == "contiguous") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("contiguous operation requires exactly 1 input");
                }
                return {cuda::contiguous_kernel(inputs[0], stream)};
            }
            else if (op_name == "clone") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("clone operation requires exactly 1 input");
                }
                return {cuda::clone_kernel(inputs[0], stream)};
            }
            else if (op_name == "reshape") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("reshape operation requires exactly 1 input");
                }
                // Parse shape from comma-separated string
                std::vector<int64_t> shape;
                if (attrs.contains("shape")) {
                    std::string shape_str = attrs.at("shape");
                    size_t pos = 0;
                    while (pos < shape_str.size()) {
                        size_t comma = shape_str.find(',', pos);
                        if (comma == std::string::npos) {
                            shape.push_back(std::stoll(shape_str.substr(pos)));
                            break;
                        }
                        shape.push_back(std::stoll(shape_str.substr(pos, comma - pos)));
                        pos = comma + 1;
                    }
                }
                return {cuda::reshape_kernel(inputs[0], shape, stream)};
            }
            else if (op_name == "transpose") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("transpose operation requires exactly 1 input");
                }
                int64_t dim0 = 0;
                int64_t dim1 = 1;
                if (attrs.contains("dim0")) {
                    dim0 = std::stoll(attrs.at("dim0"));
                }
                if (attrs.contains("dim1")) {
                    dim1 = std::stoll(attrs.at("dim1"));
                }
                return {cuda::transpose_kernel(inputs[0], dim0, dim1, stream)};
            }
            else if (op_name == "permute") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("permute operation requires exactly 1 input");
                }
                // Parse dims from comma-separated string
                std::vector<int64_t> dims;
                if (attrs.contains("dims")) {
                    std::string dims_str = attrs.at("dims");
                    size_t pos = 0;
                    while (pos < dims_str.size()) {
                        size_t comma = dims_str.find(',', pos);
                        if (comma == std::string::npos) {
                            dims.push_back(std::stoll(dims_str.substr(pos)));
                            break;
                        }
                        dims.push_back(std::stoll(dims_str.substr(pos, comma - pos)));
                        pos = comma + 1;
                    }
                }
                return {cuda::permute_kernel(inputs[0], dims, stream)};
            }
            else if (op_name == "squeeze") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("squeeze operation requires exactly 1 input");
                }
                int64_t dim = -1;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::squeeze_kernel(inputs[0], dim, stream)};
            }
            else if (op_name == "unsqueeze") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("unsqueeze operation requires exactly 1 input");
                }
                int64_t dim = 0;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::unsqueeze_kernel(inputs[0], dim, stream)};
            }
            else if (op_name == "index_select") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("index_select operation requires exactly 2 inputs");
                }
                int64_t dim = 0;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::index_select_kernel(inputs[0], dim, inputs[1], stream)};
            }
            else if (op_name == "gather") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("gather operation requires exactly 2 inputs");
                }
                int64_t dim = 0;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::gather_kernel(inputs[0], dim, inputs[1], stream)};
            }
            else if (op_name == "scatter") {
                if (inputs.size() != 3) {
                    throw std::invalid_argument("scatter operation requires exactly 3 inputs");
                }
                int64_t dim = 0;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::scatter_kernel(inputs[0], dim, inputs[1], inputs[2], stream)};
            }
            else if (op_name == "masked_select") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("masked_select operation requires exactly 2 inputs");
                }
                return {cuda::masked_select_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "masked_fill") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("masked_fill operation requires exactly 2 inputs");
                }
                double value = 0.0;
                if (attrs.contains("value")) {
                    value = std::stod(attrs.at("value"));
                }
                return {cuda::masked_fill_kernel(inputs[0], inputs[1], value, stream)};
            }
            else if (op_name == "where") {
                if (inputs.size() != 3) {
                    throw std::invalid_argument("where operation requires exactly 3 inputs");
                }
                return {cuda::where_kernel(inputs[0], inputs[1], inputs[2], stream)};
            }
            else if (op_name == "cat") {
                if (inputs.empty()) {
                    throw std::invalid_argument("cat operation requires at least 1 input tensor");
                }
                int64_t dim = 0;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::cat_kernel(inputs, dim, stream)};
            }
            else if (op_name == "batchnorm2d_mean_var") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("batchnorm2d_mean_var operation requires exactly 1 input");
                }
                // mean and variance tensors are passed in attrs as references
                // For now, we'll return them as outputs
                auto shape = inputs[0].shape();
                int64_t C = shape[1];
                Tensor mean({C}, inputs[0].dtype(), inputs[0].device());
                Tensor variance({C}, inputs[0].dtype(), inputs[0].device());
                cuda::batchnorm2d_mean_var(inputs[0], mean, variance, stream);
                return {mean, variance};
            }
            else if (op_name == "batchnorm2d_forward") {
                if (inputs.size() != 3) {
                    throw std::invalid_argument("batchnorm2d_forward operation requires exactly 3 inputs (input, mean, variance)");
                }
                float epsilon = 1e-5f;
                if (attrs.contains("epsilon")) {
                    epsilon = std::stof(attrs.at("epsilon"));
                }
                return {cuda::batchnorm2d_forward(inputs[0], inputs[1], inputs[2], epsilon, stream)};
            }
            else if (op_name == "batchnorm2d_forward_affine") {
                if (inputs.size() != 5) {
                    throw std::invalid_argument("batchnorm2d_forward_affine operation requires exactly 5 inputs (input, mean, variance, gamma, beta)");
                }
                float epsilon = 1e-5f;
                if (attrs.contains("epsilon")) {
                    epsilon = std::stof(attrs.at("epsilon"));
                }
                return {cuda::batchnorm2d_forward_affine(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, stream)};
            }
            else if (op_name == "batchnorm2d_update_running_stats") {
                if (inputs.size() != 4) {
                    throw std::invalid_argument("batchnorm2d_update_running_stats operation requires exactly 4 inputs (running_mean, running_var, batch_mean, batch_var)");
                }
                float momentum = 0.1f;
                if (attrs.contains("momentum")) {
                    momentum = std::stof(attrs.at("momentum"));
                }
                // Note: running_mean and running_var are modified in-place
                // This is a special case - we need to get mutable references
                // For now, we'll copy back the results
                Tensor running_mean = inputs[0];
                Tensor running_var = inputs[1];
                cuda::batchnorm2d_update_running_stats(running_mean, running_var, inputs[2], inputs[3], momentum, stream);
                return {running_mean, running_var};
            }
            else if (op_name == "batchnorm2d_backward") {
                if (inputs.size() != 5) {
                    throw std::invalid_argument("batchnorm2d_backward operation requires exactly 5 inputs (grad_output, input, mean, variance, gamma)");
                }
                float epsilon = 1e-5f;
                if (attrs.contains("epsilon")) {
                    epsilon = std::stof(attrs.at("epsilon"));
                }
                auto [grad_input, grad_gamma, grad_beta] = cuda::batchnorm2d_backward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, stream);
                return {grad_input, grad_gamma, grad_beta};
            }
            else if (op_name == "eq") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("eq operation requires exactly 2 inputs");
                }
                return {cuda::eq_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "ne") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("ne operation requires exactly 2 inputs");
                }
                return {cuda::ne_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "lt") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("lt operation requires exactly 2 inputs");
                }
                return {cuda::lt_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "le") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("le operation requires exactly 2 inputs");
                }
                return {cuda::le_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "gt") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("gt operation requires exactly 2 inputs");
                }
                return {cuda::gt_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "ge") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("ge operation requires exactly 2 inputs");
                }
                return {cuda::ge_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "dot") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("dot operation requires exactly 2 inputs");
                }
                return {cuda::dot_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "conv2d_forward") {
                // Parse conv2d parameters
                int64_t stride = 1, padding = 0, dilation = 1, groups = 1;
                if (attrs.contains("stride")) stride = std::stoll(attrs.at("stride"));
                if (attrs.contains("padding")) padding = std::stoll(attrs.at("padding"));
                if (attrs.contains("dilation")) dilation = std::stoll(attrs.at("dilation"));
                if (attrs.contains("groups")) groups = std::stoll(attrs.at("groups"));

                const Tensor* bias_ptr = nullptr;
                if (inputs.size() == 3) {
                    bias_ptr = &inputs[2];
                } else if (inputs.size() != 2) {
                    throw std::invalid_argument("conv2d_forward requires 2 or 3 inputs (input, weight, optional bias)");
                }

                #ifdef TENZOR_HAS_CUDNN
                // Detect memory format from input strides
                // For ChannelsLast tensors, use native NHWC cuDNN path - no conversion overhead!
                // Note: cuDNN NHWC format doesn't support Float64, so force NCHW path for Float64
                bool use_nhwc = (inputs[0].memory_format() == MemoryFormat::ChannelsLast) &&
                                (inputs[0].dtype() != DType::Float64);

                try {
                    if (use_nhwc) {
                        // Native NHWC path: input already in NHWC layout via strides
                        // cuDNN uses NHWC tensor descriptors, output preserves NHWC format
                        return {cuda::cudnn_conv2d_forward_nhwc(inputs[0], inputs[1], bias_ptr, stride, padding, dilation, groups, stream)};
                    } else {
                        // Standard NCHW path
                        return {cuda::cudnn_conv2d_forward(inputs[0], inputs[1], bias_ptr, stride, padding, dilation, groups, stream)};
                    }
                } catch (const std::exception& e) {
                    // Fall back to custom kernel if cuDNN fails (but not for Float64 since custom kernel doesn't support it)
                    if (inputs[0].dtype() == DType::Float64) {
                        throw std::runtime_error(std::string("conv2d_forward cuDNN failed for Float64: ") + e.what());
                    }
                    return {cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias_ptr, stride, padding, dilation, groups, stream)};
                }
                #else
                // Use custom kernel
                return {cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias_ptr, stride, padding, dilation, groups, stream)};
                #endif
            }
            else if (op_name == "conv2d_backward") {
                // Parse conv2d parameters
                int64_t stride = 1, padding = 0, dilation = 1, groups = 1;
                if (attrs.contains("stride")) stride = std::stoll(attrs.at("stride"));
                if (attrs.contains("padding")) padding = std::stoll(attrs.at("padding"));
                if (attrs.contains("dilation")) dilation = std::stoll(attrs.at("dilation"));
                if (attrs.contains("groups")) groups = std::stoll(attrs.at("groups"));

                bool compute_grad_input = true, compute_grad_weight = true, compute_grad_bias = true;
                if (attrs.contains("compute_grad_input")) compute_grad_input = (attrs.at("compute_grad_input") == "1");
                if (attrs.contains("compute_grad_weight")) compute_grad_weight = (attrs.at("compute_grad_weight") == "1");
                if (attrs.contains("compute_grad_bias")) compute_grad_bias = (attrs.at("compute_grad_bias") == "1");

                if (inputs.size() != 3) {
                    throw std::invalid_argument("conv2d_backward requires 3 inputs (grad_output, input, weight)");
                }

                #ifdef TENZOR_HAS_CUDNN
                // Detect memory format from input strides (inputs[1] is the original input)
                // Note: cuDNN NHWC format doesn't support Float64, so force NCHW path for Float64
                bool use_nhwc = (inputs[1].memory_format() == MemoryFormat::ChannelsLast) &&
                                (inputs[0].dtype() != DType::Float64);

                try {
                    if (use_nhwc) {
                        // Native NHWC backward path
                        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv2d_backward_nhwc(
                            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups,
                            compute_grad_input, compute_grad_weight, compute_grad_bias, stream
                        );
                        return {grad_input, grad_weight, grad_bias};
                    } else {
                        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv2d_backward(
                            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups,
                            compute_grad_input, compute_grad_weight, compute_grad_bias, stream
                        );
                        return {grad_input, grad_weight, grad_bias};
                    }
                } catch (const std::exception& e) {
                    // For Float64, re-throw cuDNN error since custom kernel doesn't support it
                    if (inputs[0].dtype() == DType::Float64) {
                        throw std::runtime_error(
                            std::string("conv2d_backward cuDNN failed for Float64: ") + e.what()
                        );
                    }
                    // Fall back to custom kernel if cuDNN fails for other dtypes
                    auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
                        inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups,
                        compute_grad_input, compute_grad_weight, compute_grad_bias, stream
                    );
                    return {grad_input, grad_weight, grad_bias};
                }
                #else
                // Use custom kernel
                auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
                    inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups,
                    compute_grad_input, compute_grad_weight, compute_grad_bias, stream
                );
                return {grad_input, grad_weight, grad_bias};
                #endif
            }
            else if (op_name == "conv2d_backward_input") {
                // Parse conv2d parameters
                int64_t stride = 1, padding = 0, dilation = 1, groups = 1;
                if (attrs.contains("stride")) stride = std::stoll(attrs.at("stride"));
                if (attrs.contains("padding")) padding = std::stoll(attrs.at("padding"));
                if (attrs.contains("dilation")) dilation = std::stoll(attrs.at("dilation"));
                if (attrs.contains("groups")) groups = std::stoll(attrs.at("groups"));

                if (inputs.size() != 2) {
                    throw std::invalid_argument("conv2d_backward_input requires 2 inputs (grad_output, weight)");
                }

                // Parse input_shape from comma-separated string
                std::vector<int64_t> input_shape;
                if (attrs.contains("input_shape")) {
                    std::string shape_str = attrs.at("input_shape");
                    std::stringstream ss(shape_str);
                    std::string item;
                    while (std::getline(ss, item, ',')) {
                        input_shape.push_back(std::stoll(item));
                    }
                }

                // Create a dummy input tensor with the correct shape for the backward kernel
                auto dummy_input = zeros(input_shape, inputs[0].dtype(), inputs[0].device());

                #ifdef TENZOR_HAS_CUDNN
                try {
                    auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv2d_backward(
                        inputs[0], dummy_input, inputs[1], stride, padding, dilation, groups,
                        true, false, false, stream
                    );
                    return std::vector<Tensor>{grad_input};
                } catch (const std::exception& e) {
                    // Re-throw for Float64 since custom kernel doesn't support it
                    if (inputs[0].dtype() == DType::Float64) {
                        throw std::runtime_error(std::string("conv2d_backward_input cuDNN failed for Float64: ") + e.what());
                    }
                    auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
                        inputs[0], dummy_input, inputs[1], stride, padding, dilation, groups,
                        true, false, false, stream
                    );
                    return std::vector<Tensor>{grad_input};
                }
                #else
                auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
                    inputs[0], dummy_input, inputs[1], stride, padding, dilation, groups,
                    true, false, false, stream
                );
                return std::vector<Tensor>{grad_input};
                #endif
            }
            else if (op_name == "conv2d_backward_weight") {
                // Parse conv2d parameters
                int64_t stride = 1, padding = 0, dilation = 1, groups = 1;
                if (attrs.contains("stride")) stride = std::stoll(attrs.at("stride"));
                if (attrs.contains("padding")) padding = std::stoll(attrs.at("padding"));
                if (attrs.contains("dilation")) dilation = std::stoll(attrs.at("dilation"));
                if (attrs.contains("groups")) groups = std::stoll(attrs.at("groups"));

                if (inputs.size() != 2) {
                    throw std::invalid_argument("conv2d_backward_weight requires 2 inputs (grad_output, input)");
                }

                // Parse weight_shape from comma-separated string
                std::vector<int64_t> weight_shape;
                if (attrs.contains("weight_shape")) {
                    std::string shape_str = attrs.at("weight_shape");
                    std::stringstream ss(shape_str);
                    std::string item;
                    while (std::getline(ss, item, ',')) {
                        weight_shape.push_back(std::stoll(item));
                    }
                }

                // Create a dummy weight tensor with the correct shape for the backward kernel
                auto dummy_weight = zeros(weight_shape, inputs[0].dtype(), inputs[0].device());

                #ifdef TENZOR_HAS_CUDNN
                try {
                    auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv2d_backward(
                        inputs[0], inputs[1], dummy_weight, stride, padding, dilation, groups,
                        false, true, false, stream
                    );
                    return std::vector<Tensor>{grad_weight};
                } catch (const std::exception& e) {
                    // Re-throw for Float64 since custom kernel doesn't support it
                    if (inputs[0].dtype() == DType::Float64) {
                        throw std::runtime_error(std::string("conv2d_backward_weight cuDNN failed for Float64: ") + e.what());
                    }
                    auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
                        inputs[0], inputs[1], dummy_weight, stride, padding, dilation, groups,
                        false, true, false, stream
                    );
                    return std::vector<Tensor>{grad_weight};
                }
                #else
                auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
                    inputs[0], inputs[1], dummy_weight, stride, padding, dilation, groups,
                    false, true, false, stream
                );
                return std::vector<Tensor>{grad_weight};
                #endif
            }
            else if (op_name == "conv2d_backward_bias") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("conv2d_backward_bias requires 1 input (grad_output)");
                }

                // Gradient w.r.t. bias is just sum over batch and spatial dimensions
                // Sum over dimensions: (batch, out_channels, out_h, out_w) -> (out_channels)
                auto grad_output_shape = inputs[0].shape();
                int64_t out_channels = grad_output_shape[1];

                // Create output tensor for bias gradient
                auto grad_bias = zeros({out_channels}, inputs[0].dtype(), inputs[0].device());

                // Use a simple sum reduction - sum over batch (0), height (2), width (3)
                // For now, use the unified backward kernel with dummy tensors
                // Create minimal dummy tensors
                auto dummy_input = zeros({1, 1, 1, 1}, inputs[0].dtype(), inputs[0].device());
                auto dummy_weight = zeros({static_cast<int64_t>(out_channels), 1, 1, 1}, inputs[0].dtype(), inputs[0].device());

                #ifdef TENZOR_HAS_CUDNN
                try {
                    auto [gi, gw, gb] = cuda::cudnn_conv2d_backward(
                        inputs[0], dummy_input, dummy_weight, 1, 0, 1, 1,
                        false, false, true, stream
                    );
                    return std::vector<Tensor>{gb};
                } catch (const std::exception& e) {
                    // Re-throw for Float64 since custom kernel doesn't support it
                    if (inputs[0].dtype() == DType::Float64) {
                        throw std::runtime_error(std::string("conv2d_backward_bias cuDNN failed for Float64: ") + e.what());
                    }
                    auto [gi, gw, gb] = cuda::conv2d_backward_kernel(
                        inputs[0], dummy_input, dummy_weight, 1, 0, 1, 1,
                        false, false, true, stream
                    );
                    return std::vector<Tensor>{gb};
                }
                #else
                auto [gi, gw, gb] = cuda::conv2d_backward_kernel(
                    inputs[0], dummy_input, dummy_weight, 1, 0, 1, 1,
                    false, false, true, stream
                );
                return std::vector<Tensor>{gb};
                #endif
            }
            else if (op_name == "conv_transpose2d_forward") {
                // Parse conv_transpose2d parameters
                int64_t stride = 1, padding = 0, output_padding = 0, dilation = 1, groups = 1;
                if (attrs.contains("stride")) stride = std::stoll(attrs.at("stride"));
                if (attrs.contains("padding")) padding = std::stoll(attrs.at("padding"));
                if (attrs.contains("output_padding")) output_padding = std::stoll(attrs.at("output_padding"));
                if (attrs.contains("dilation")) dilation = std::stoll(attrs.at("dilation"));
                if (attrs.contains("groups")) groups = std::stoll(attrs.at("groups"));

                const Tensor* bias_ptr = nullptr;
                if (inputs.size() == 3) {
                    bias_ptr = &inputs[2];
                } else if (inputs.size() != 2) {
                    throw std::invalid_argument("conv_transpose2d_forward requires 2 or 3 inputs (input, weight, optional bias)");
                }

                return {cuda::conv_transpose2d_forward_kernel(inputs[0], inputs[1], bias_ptr, stride, padding, output_padding, dilation, groups, stream)};
            }
            else if (op_name == "lstm_cell_forward") {
                // Parse LSTM parameters
                int64_t batch_size = 1, hidden_size = 1;
                if (attrs.contains("batch_size")) batch_size = std::stoll(attrs.at("batch_size"));
                if (attrs.contains("hidden_size")) hidden_size = std::stoll(attrs.at("hidden_size"));

                if (inputs.size() != 2) {
                    throw std::invalid_argument("lstm_cell_forward requires 2 inputs (gates, c_prev)");
                }

                // For LSTM cell, custom kernel is often competitive with cuDNN for single cells
                // Use custom kernel for simplicity
                auto [h_out, c_out] = cuda::lstm_cell_forward_kernel(inputs[0], inputs[1], batch_size, hidden_size, stream);
                return {h_out, c_out};
            }
            else if (op_name == "lstm_cell_backward") {
                // Parse LSTM parameters
                int64_t batch_size = 1, hidden_size = 1;
                if (attrs.contains("batch_size")) batch_size = std::stoll(attrs.at("batch_size"));
                if (attrs.contains("hidden_size")) hidden_size = std::stoll(attrs.at("hidden_size"));

                if (inputs.size() != 5) {
                    throw std::invalid_argument("lstm_cell_backward requires 5 inputs (grad_h, grad_c, gates, c_prev, c_out)");
                }

                // Use custom kernel
                auto [grad_gates, grad_c_prev] = cuda::lstm_cell_backward_kernel(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], batch_size, hidden_size, stream
                );
                return {grad_gates, grad_c_prev};
            }
            // Pooling operations
            else if (op_name == "adaptive_avg_pool2d") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("adaptive_avg_pool2d operation requires exactly 1 input");
                }
                int64_t output_h = 1, output_w = 1;
                if (attrs.contains("output_h")) {
                    output_h = std::stoll(attrs.at("output_h"));
                }
                if (attrs.contains("output_w")) {
                    output_w = std::stoll(attrs.at("output_w"));
                }
                return {cuda::adaptive_avg_pool2d_forward(inputs[0], output_h, output_w)};
            }
            else if (op_name == "adaptive_avg_pool2d_backward") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("adaptive_avg_pool2d_backward operation requires exactly 1 input");
                }
                int64_t H_in = 0, W_in = 0;
                if (attrs.contains("H_in")) {
                    H_in = std::stoll(attrs.at("H_in"));
                }
                if (attrs.contains("W_in")) {
                    W_in = std::stoll(attrs.at("W_in"));
                }
                return {cuda::adaptive_avg_pool2d_backward(inputs[0], H_in, W_in)};
            }
            else if (op_name == "max_pool2d") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("max_pool2d operation requires exactly 1 input");
                }
                int64_t kernel_size = 2, stride = 2, padding = 0;
                if (attrs.contains("kernel_size")) {
                    kernel_size = std::stoll(attrs.at("kernel_size"));
                }
                if (attrs.contains("stride")) {
                    stride = std::stoll(attrs.at("stride"));
                }
                if (attrs.contains("padding")) {
                    padding = std::stoll(attrs.at("padding"));
                }
                auto [output, indices] = cuda::max_pool2d_forward(inputs[0], kernel_size, stride, padding);
                return {output, indices};
            }
            else if (op_name == "max_pool2d_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("max_pool2d_backward operation requires exactly 2 inputs (grad_output, indices)");
                }
                int64_t H_in = 0, W_in = 0;
                if (attrs.contains("H_in")) {
                    H_in = std::stoll(attrs.at("H_in"));
                }
                if (attrs.contains("W_in")) {
                    W_in = std::stoll(attrs.at("W_in"));
                }
                return {cuda::max_pool2d_backward(inputs[0], inputs[1], H_in, W_in)};
            }
            else if (op_name == "avg_pool2d") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("avg_pool2d operation requires exactly 1 input");
                }
                int64_t kernel_size = 2, stride = 2, padding = 0;
                if (attrs.contains("kernel_size")) {
                    kernel_size = std::stoll(attrs.at("kernel_size"));
                }
                if (attrs.contains("stride")) {
                    stride = std::stoll(attrs.at("stride"));
                }
                if (attrs.contains("padding")) {
                    padding = std::stoll(attrs.at("padding"));
                }
                return {cuda::avg_pool2d_forward(inputs[0], kernel_size, stride, padding)};
            }
            else if (op_name == "avg_pool2d_backward") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("avg_pool2d_backward operation requires exactly 1 input (grad_output)");
                }
                int64_t H_in = 0, W_in = 0, kernel_size = 2, stride = 2, padding = 0;
                if (attrs.contains("H_in")) {
                    H_in = std::stoll(attrs.at("H_in"));
                }
                if (attrs.contains("W_in")) {
                    W_in = std::stoll(attrs.at("W_in"));
                }
                if (attrs.contains("kernel_size")) {
                    kernel_size = std::stoll(attrs.at("kernel_size"));
                }
                if (attrs.contains("stride")) {
                    stride = std::stoll(attrs.at("stride"));
                }
                if (attrs.contains("padding")) {
                    padding = std::stoll(attrs.at("padding"));
                }
                return {cuda::avg_pool2d_backward(inputs[0], H_in, W_in, kernel_size, stride, padding)};
            }
            // Vision operations
            else if (op_name == "gather_relative_position_bias") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("gather_relative_position_bias operation requires exactly 2 inputs");
                }
                int64_t num_positions = 0, num_heads = 0;
                if (attrs.contains("num_positions")) {
                    num_positions = std::stoll(attrs.at("num_positions"));
                }
                if (attrs.contains("num_heads")) {
                    num_heads = std::stoll(attrs.at("num_heads"));
                }
                return {cuda::gather_relative_position_bias(inputs[0], inputs[1], num_positions, num_heads)};
            }
            // Fused operations disabled - CUDA kernels not yet implemented
            // Fall back to CPU fused operations for now
            else if (op_name == "fused_linear_relu" || op_name == "fused_batchnorm_relu" ||
                     op_name == "fused_softmax_cross_entropy" || op_name == "fused_add_relu" ||
                     op_name == "fused_gelu" || op_name == "fused_layer_norm") {
                throw std::runtime_error("CUDABackend: Fused operation '" + op_name + "' not yet implemented for CUDA. Use CPU backend for fused ops.");
            }
            else {
                throw std::runtime_error("CUDABackend: Unknown operation '" + op_name + "'");
            }
        }
        catch (const std::exception& e) {
            // Check for CUDA errors and provide more context
            cudaError_t cuda_error = cudaGetLastError();
            if (cuda_error != cudaSuccess) {
                throw std::runtime_error(
                    "CUDABackend: Operation '" + op_name + "' failed with CUDA error: " +
                    cudaGetErrorString(cuda_error) + " (Original exception: " + e.what() + ")"
                );
            }
            throw;
        }
    }

private:
    bool use_caching_allocator_{false};
};

extern "C" {
    auto create_backend() -> std::unique_ptr<Backend> {
        return std::make_unique<CUDABackend>();
    }
}

} // namespace tenzor
