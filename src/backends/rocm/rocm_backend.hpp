/**
 * @file rocm_backend.hpp
 * @brief ROCm/HIP backend implementation for AMD GPUs
 *
 * Provides backend implementation using AMD's ROCm platform and HIP API.
 * This backend converts CUDA-style operations to HIP equivalents for AMD hardware.
 */

#pragma once

#include "tenzor/backend/backend.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"
#include "hip_graph.hpp"
#include <hip/hip_runtime.h>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <span>

namespace tenzor {

/**
 * @brief Forward declarations for HIP kernels
 *
 * These kernels are implemented in separate .cpp files using HIP.
 * All operations mirror the CUDA backend but use HIP API.
 */
namespace rocm {
    // Data layout for convolution operations
    enum class DataLayout {
        NCHW,  // Batch, Channels, Height, Width (default)
        NHWC   // Batch, Height, Width, Channels (TensorFlow style)
    };

    // Binary operations
    auto add_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto sub_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto mul_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto div_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto matmul_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;

    // Unary operations
    auto sqrt_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto neg_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto abs_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto sign_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto log_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto exp_kernel(const Tensor& input, hipStream_t stream) -> Tensor;

    // Operations with parameters
    auto clamp_kernel(const Tensor& input, float min_val, float max_val, hipStream_t stream) -> Tensor;
    auto pow_kernel(const Tensor& input, float exponent, hipStream_t stream) -> Tensor;

    // Reduction operations
    auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto max_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto min_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto logsumexp_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;

    // Activation functions
    auto relu_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto sigmoid_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto tanh_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto leaky_relu_kernel(const Tensor& input, float alpha, hipStream_t stream) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, hipStream_t stream) -> Tensor;

    // Softmax operations
    auto softmax_kernel(const Tensor& input, int64_t dim, hipStream_t stream, float temperature = 1.0f) -> Tensor;
    auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, hipStream_t stream) -> Tensor;
    auto log_softmax_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;
    auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, hipStream_t stream) -> Tensor;

    // Transform operations
    auto contiguous_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto clone_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, hipStream_t stream) -> Tensor;
    auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, hipStream_t stream) -> Tensor;
    auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, hipStream_t stream) -> Tensor;
    auto squeeze_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;
    auto unsqueeze_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;
    auto expand_kernel(const Tensor& input, const std::vector<int64_t>& shape, void* stream) -> Tensor;

    // Fill operations
    auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor;
    auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor;
    auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, Device device, hipStream_t stream) -> Tensor;
    auto fill_kernel(const Tensor& tensor, float value, hipStream_t stream) -> Tensor;

    // Random operations
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor;
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor;

    // BatchNorm2d operations
    auto batchnorm2d_mean_var(const Tensor& input, Tensor& mean, Tensor& variance, hipStream_t stream) -> void;
    auto batchnorm2d_forward(const Tensor& input, const Tensor& mean, const Tensor& variance, float epsilon, hipStream_t stream) -> Tensor;
    auto batchnorm2d_forward_affine(const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, const Tensor& beta, float epsilon, hipStream_t stream) -> Tensor;
    auto batchnorm2d_update_running_stats(Tensor& running_mean, Tensor& running_var, const Tensor& batch_mean, const Tensor& batch_var, float momentum, hipStream_t stream) -> void;
    auto batchnorm2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, float epsilon, hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;

    // Conv2d operations
    auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, hipStream_t stream, DataLayout layout = DataLayout::NCHW) -> Tensor;
    auto conv2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias, hipStream_t stream, DataLayout layout = DataLayout::NCHW) -> std::tuple<Tensor, Tensor, Tensor>;

    // Pooling operations
    auto maxpool2d_forward_hip(const Tensor& input, int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, bool return_indices) -> std::pair<Tensor, Tensor>;
    auto maxpool2d_backward_hip(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto avgpool2d_forward_hip(const Tensor& input, int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, bool count_include_pad) -> Tensor;

    // Swin Transformer operations
    auto gather_relative_position_bias_kernel(const Tensor& table, const Tensor& indices, int64_t num_positions, int64_t num_heads, hipStream_t stream) -> Tensor;

    // Adaptive Average Pooling operations
    auto adaptive_avgpool2d_forward(const Tensor& input, int64_t output_h, int64_t output_w, hipStream_t stream) -> Tensor;
    auto adaptive_avgpool2d_backward(const Tensor& grad_output, int64_t H_in, int64_t W_in, hipStream_t stream) -> Tensor;
} // namespace rocm

/**
 * @brief ROCm backend implementation using HIP
 *
 * Provides full backend functionality for AMD GPUs via ROCm/HIP.
 * Supports:
 * - Memory management with optional caching allocator
 * - Asynchronous operations via HIP streams
 * - Comprehensive kernel dispatch for all operations
 * - Multi-GPU support
 * - Error handling and device properties
 */
class ROCmBackend : public Backend {
public:
    ROCmBackend();
    ~ROCmBackend() override = default;

    // Backend identification
    auto name() const -> std::string_view override;
    auto device_count() const -> int32_t override;
    auto is_available() const -> bool override;
    auto get_device_info(int32_t device_id) const -> DeviceInfo override;

    // Device context management
    auto set_device(int32_t device_id) -> void override;
    auto get_current_device() const -> int32_t override;

    // Memory management
    auto allocate(size_t bytes, int32_t device_id) -> void* override;
    auto deallocate(void* ptr) -> void override;
    auto copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void override;

    // Synchronization
    auto synchronize(int32_t device_id) -> void override;

    // Stream management
    auto create_stream(int32_t device_id) -> StreamHandle override;
    auto destroy_stream(StreamHandle stream) -> void override;
    auto synchronize_stream(StreamHandle stream) -> void override;

    // Event management
    auto create_event(int32_t device_id, bool enable_timing = true) -> EventHandle override;
    auto destroy_event(EventHandle event) -> void override;
    auto record_event(EventHandle event, StreamHandle stream = nullptr) -> void override;
    auto wait_event(EventHandle event, StreamHandle stream = nullptr) -> void override;
    auto event_elapsed_ms(EventHandle start_event, EventHandle end_event) -> float override;

    // Memory fill
    auto memset(void* ptr, int value, size_t bytes, int32_t device_id) -> void override;

    // Kernel dispatch
    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor> override;

    // ROCm-specific utilities

    /**
     * @brief Get device properties for a specific device
     * @param device_id Device index
     * @return hipDeviceProp_t structure with device information
     */
    auto get_device_properties(int32_t device_id) const -> hipDeviceProp_t;

    /**
     * @brief Check if caching allocator is enabled
     * @return true if using caching allocator
     */
    auto is_using_caching_allocator() const -> bool { return use_caching_allocator_; }

    /**
     * @brief Create a HIP graph capture object for the given device.
     *
     * The returned HIPGraph can be used to capture and replay sequences
     * of HIP kernel launches for reduced dispatch overhead.
     *
     * @param device_id Device index (default: 0)
     * @return Unique pointer to a HIPGraph instance
     */
    auto create_hip_graph(int32_t device_id = 0) -> std::unique_ptr<rocm::HIPGraph>;

private:
    bool use_caching_allocator_{false};

    /// Tracks device_id for each allocation (used by deallocate to find correct device)
    mutable std::mutex alloc_map_mutex_;
    std::unordered_map<void*, int32_t> alloc_device_map_;

    /**
     * @brief Helper to check and throw on HIP errors
     * @param err HIP error code
     * @param operation Description of the operation for error messages
     */
    void check_hip_error(hipError_t err, const char* operation) const;
};

} // namespace tenzor
