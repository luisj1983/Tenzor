#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include <cstring>
#include <cstdint>

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#endif

namespace tenzor {

// Forward declarations for OneAPI/SYCL kernels
namespace oneapi {
    // Math operations
    auto add_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto sub_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto mul_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto div_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto matmul_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;

    // In-place operations
    auto add_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor;
    auto sub_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor;
    auto mul_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor;
    auto div_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor;

    // Unary operations
    auto sqrt_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto neg_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto abs_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto log_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto exp_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto pow_kernel(const Tensor& input, float exponent, sycl::queue& queue) -> Tensor;
    auto dot_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;

    // Activation functions
    auto relu_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor;
    auto sigmoid_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& output, sycl::queue& queue) -> Tensor;
    auto tanh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& output, sycl::queue& queue) -> Tensor;
    auto gelu_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor;
    auto softmax_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, sycl::queue& queue) -> Tensor;
    auto log_softmax_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, sycl::queue& queue) -> Tensor;
    auto leaky_relu_kernel(const Tensor& input, float alpha, sycl::queue& queue) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, sycl::queue& queue) -> Tensor;
    auto swish_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto swish_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor;

    // Reduction operations
    auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto max_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto min_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;

    // Transform operations
    auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, sycl::queue& queue) -> Tensor;
    auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, sycl::queue& queue) -> Tensor;
    auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, sycl::queue& queue) -> Tensor;
    auto squeeze_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto unsqueeze_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto index_select_kernel(const Tensor& input, int64_t dim, const Tensor& index, sycl::queue& queue) -> Tensor;
    auto gather_kernel(const Tensor& input, int64_t dim, const Tensor& index, sycl::queue& queue) -> Tensor;
    auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src, sycl::queue& queue) -> Tensor;
    auto masked_select_kernel(const Tensor& input, const Tensor& mask, sycl::queue& queue) -> Tensor;
    auto masked_fill_kernel(const Tensor& input, const Tensor& mask, float value, sycl::queue& queue) -> Tensor;
    auto contiguous_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto clone_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;

    // Fill operations
    auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto fill_kernel(const Tensor& tensor, float value, sycl::queue& queue) -> Tensor;

    // Random number generation
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor;

    // Batch normalization
    auto batchnorm2d_mean_var(const Tensor& input, sycl::queue& queue) -> std::vector<Tensor>;
    auto batchnorm2d_update_running_stats(Tensor& running_mean, Tensor& running_var,
                                         const Tensor& batch_mean, const Tensor& batch_var,
                                         float momentum, sycl::queue& queue) -> void;
    auto batchnorm2d_forward(const Tensor& input, const Tensor& mean, const Tensor& variance,
                            float epsilon, sycl::queue& queue) -> Tensor;
    auto batchnorm2d_forward_affine(const Tensor& input, const Tensor& mean, const Tensor& variance,
                                   const Tensor& gamma, const Tensor& beta, float epsilon, sycl::queue& queue) -> Tensor;
    auto batchnorm2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& mean,
                             const Tensor& variance, const Tensor& gamma, float epsilon, sycl::queue& queue)
                             -> std::tuple<Tensor, Tensor, Tensor>;

    // Conv2d operations
    auto conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                       int64_t stride, int64_t padding, int64_t dilation, int64_t groups, sycl::queue& queue) -> Tensor;
    auto conv2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                        int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                        bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias,
                        sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
    // Separate backward operations (matching CPU API)
    auto conv2d_backward_input(const Tensor& grad_output, const Tensor& weight,
                               const std::vector<int64_t>& input_shape,
                               int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                               sycl::queue& queue) -> Tensor;
    auto conv2d_backward_weight(const Tensor& grad_output, const Tensor& input,
                                const std::vector<int64_t>& weight_shape,
                                int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                                sycl::queue& queue) -> Tensor;
    auto conv2d_backward_bias(const Tensor& grad_output, sycl::queue& queue) -> Tensor;

    // Embedding operations
    auto embedding_lookup_kernel(const Tensor& indices, const Tensor& weights,
                                int64_t padding_idx, sycl::queue& queue) -> Tensor;
    auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                  int64_t vocab_size, int64_t embedding_dim,
                                  sycl::queue& queue) -> Tensor;
    auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets,
                                     const std::string& mode, bool include_last_offset,
                                     sycl::queue& queue) -> Tensor;
    auto embedding_renorm_kernel(Tensor& weights, const Tensor& indices,
                                double max_norm, double norm_type,
                                sycl::queue& queue) -> void;

    // Im2col/Col2im operations
    auto im2col_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto col2im_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;

    // Expand operation
    auto expand_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;

    // Pooling operations
    auto avg_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto max_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto adaptive_avg_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto adaptive_max_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto avg_pool2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto max_pool2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto max_pool2d_backward_with_indices(const Tensor& grad_output, const Tensor& indices, int64_t H_in, int64_t W_in, sycl::queue& queue) -> Tensor;
    auto adaptive_avg_pool2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto adaptive_avgpool2d_backward(const Tensor& grad_output, int64_t H_in, int64_t W_in, sycl::queue& queue) -> Tensor;

    // Statistical operations
    auto std_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto var_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto prod_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto norm_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;

    // Argmax/Argmin operations
    auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;

    // Comparison operations
    auto eq_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto ne_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto lt_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto le_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto gt_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto ge_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;

    // Utility operations
    auto cat_kernel(std::span<const Tensor> tensors, int64_t dim, sycl::queue& queue) -> Tensor;
    auto clamp_kernel(const Tensor& input, float min_val, float max_val, sycl::queue& queue) -> Tensor;
    auto sign_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;

    // Trigonometric operations
    auto sin_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto cos_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto tan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto asin_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto acos_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto atan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto sinh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto cosh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto atan2_kernel(const Tensor& y, const Tensor& x, sycl::queue& queue) -> Tensor;

    // Rounding operations
    auto round_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto floor_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto ceil_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto trunc_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto reciprocal_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;

    // Additional utility operations
    auto clamp_min_kernel(const Tensor& input, float min_val, sycl::queue& queue) -> Tensor;
    auto clamp_max_kernel(const Tensor& input, float max_val, sycl::queue& queue) -> Tensor;
    auto where_kernel(const Tensor& condition, const Tensor& x, const Tensor& y, sycl::queue& queue) -> Tensor;
    auto repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats, sycl::queue& queue) -> Tensor;

    // Fused operations
    auto fused_add_relu_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto fused_gelu_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto fused_layer_norm_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                                 const std::vector<int64_t>& normalized_shape, float epsilon,
                                 sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
    auto fused_layer_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                          const Tensor& mean, const Tensor& inv_std, const Tensor& weight,
                                          const std::vector<int64_t>& normalized_shape,
                                          sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
    auto fused_linear_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                  sycl::queue& queue) -> Tensor;
    auto fused_batchnorm_relu_kernel(const Tensor& input, const Tensor& running_mean,
                                     const Tensor& running_var, const Tensor& weight,
                                     const Tensor& bias, float epsilon, sycl::queue& queue) -> Tensor;
    auto fused_matmul_add_kernel(const Tensor& a, const Tensor& b, const Tensor& bias,
                                 sycl::queue& queue) -> Tensor;
    auto fused_softmax_cross_entropy_kernel(const Tensor& logits, const Tensor& targets,
                                            const std::string& reduction, sycl::queue& queue) -> Tensor;

    // LSTM operations
    auto lstm_cell_forward_kernel(const Tensor& gates, const Tensor& c_prev,
                                  int64_t batch_size, int64_t hidden_size,
                                  sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto lstm_cell_backward_kernel(const Tensor& grad_h, const Tensor& grad_c, const Tensor& gates,
                                   const Tensor& c_prev, const Tensor& c_out,
                                   int64_t batch_size, int64_t hidden_size,
                                   sycl::queue& queue) -> std::pair<Tensor, Tensor>;

    // GRU operations
    auto gru_cell_forward_kernel(const Tensor& reset_gates, const Tensor& update_gates,
                                 const Tensor& new_gates_input, const Tensor& new_gates_hidden,
                                 const Tensor& h_prev, int64_t batch_size, int64_t hidden_size,
                                 sycl::queue& queue) -> Tensor;
    // GRU backward outputs struct
    struct GRUBackwardOutputs {
        Tensor grad_reset;
        Tensor grad_update;
        Tensor grad_new_input;
        Tensor grad_new_hidden;
        Tensor grad_h_prev;
    };
    auto gru_cell_backward_kernel(const Tensor& grad_h, const Tensor& reset_gates,
                                  const Tensor& update_gates, const Tensor& new_gates_input,
                                  const Tensor& new_gates_hidden, const Tensor& h_prev,
                                  int64_t batch_size, int64_t hidden_size,
                                  sycl::queue& queue) -> GRUBackwardOutputs;

    // Vision operations
    auto nms_kernel(const Tensor& boxes, const Tensor& scores, float iou_threshold,
                    sycl::queue& queue) -> Tensor;
    auto roi_align_kernel(const Tensor& features, const Tensor& rois,
                          int64_t output_height, int64_t output_width,
                          float spatial_scale, int64_t sampling_ratio, bool aligned,
                          sycl::queue& queue) -> Tensor;
    auto gather_relative_position_bias_kernel(const Tensor& table, const Tensor& indices,
                                              int64_t num_positions, int64_t num_heads,
                                              sycl::queue& queue) -> Tensor;

    // Quantization operations
    auto quantize_kernel(const Tensor& input, float scale, int32_t zero_point,
                         sycl::queue& queue) -> Tensor;
    auto dequantize_kernel(const Tensor& input, float scale, int32_t zero_point,
                           sycl::queue& queue) -> Tensor;
    auto quantized_linear_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                 float input_scale, int32_t input_zero_point,
                                 float weight_scale, int32_t weight_zero_point,
                                 float output_scale, int32_t output_zero_point,
                                 sycl::queue& queue) -> Tensor;
    auto quantized_conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                 int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                                 float input_scale, int32_t input_zero_point,
                                 float weight_scale, int32_t weight_zero_point,
                                 sycl::queue& queue) -> Tensor;
} // namespace oneapi

/**
 * @brief OneAPI/SYCL backend implementation for Intel GPUs and CPUs.
 *
 * Supports Intel Data Center GPU Max Series, Intel Arc graphics, and Intel CPUs.
 * Uses SYCL for portable acceleration and optionally oneMKL/oneDNN for optimized operations.
 */
class OneAPIBackend : public Backend {
public:
    OneAPIBackend() {
        try {
            // Enumerate all available SYCL devices
            auto platforms = sycl::platform::get_platforms();

            for (const auto& platform : platforms) {
                auto devices = platform.get_devices();
                for (const auto& device : devices) {
                    // IMPORTANT: Only include devices our kernels were compiled for (spir64 target)
                    // Skip NVIDIA GPUs since kernels are compiled for CPU/Intel GPU only
                    std::string vendor = device.get_info<sycl::info::device::vendor>();
                    if (vendor.find("NVIDIA") != std::string::npos ||
                        vendor.find("nvidia") != std::string::npos) {
                        // Skip NVIDIA devices - kernels compiled for spir64, not nvptx64
                        continue;
                    }

                    // Create queue for each device
                    try {
                        auto queue = std::make_shared<sycl::queue>(device,
                            sycl::property_list{sycl::property::queue::in_order{}});

                        // Store device info
                        DeviceInfo info;
                        info.queue = queue;
                        info.device = device;
                        info.name = device.get_info<sycl::info::device::name>();
                        info.type = device.is_gpu() ? "gpu" :
                                   device.is_cpu() ? "cpu" : "accelerator";
                        info.max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
                        info.max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
                        info.global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
                        info.local_mem_size = device.get_info<sycl::info::device::local_mem_size>();

                        devices_.push_back(info);
                    } catch (const sycl::exception& e) {
                        // Skip devices that can't create queues
                        continue;
                    }
                }
            }
        } catch (const sycl::exception& e) {
            // No SYCL devices available
        }
    }

    ~OneAPIBackend() override {
        // Clean up queues
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

    auto allocate(size_t bytes, int32_t device_id) -> void* override {
        if (bytes == 0) {
            return nullptr;
        }

        validate_device_id(device_id);

        try {
            // Use USM (Unified Shared Memory) shared allocation
            // IMPORTANT: We use malloc_shared instead of malloc_device because:
            // 1. SYCL kernels need to directly access tensor memory via raw pointers
            // 2. Shared memory is accessible from both host and device
            // 3. This avoids complex buffer/accessor patterns while maintaining correctness
            // 4. Performance is comparable for most workloads on modern hardware
            auto& queue = get_queue(device_id);
            void* ptr = sycl::malloc_shared(bytes, queue);

            if (ptr == nullptr) {
                throw std::runtime_error("SYCL malloc_shared failed");
            }

            // Track allocation for proper deallocation
            allocations_[ptr] = device_id;

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

        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            throw std::runtime_error("Attempt to free untracked pointer");
        }

        int32_t device_id = it->second;
        auto& queue = get_queue(device_id);

        sycl::free(ptr, queue);
        allocations_.erase(it);
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

            case CopyKind::HostToDevice:
            case CopyKind::DeviceToHost:
            case CopyKind::DeviceToDevice: {
                // Use first available queue (device 0)
                if (devices_.empty()) {
                    throw std::runtime_error("No SYCL devices available for copy");
                }
                queue_ptr = devices_[0].queue.get();
                break;
            }
        }

        if (queue_ptr) {
            try {
                queue_ptr->memcpy(dst, src, bytes).wait();
            } catch (const sycl::exception& e) {
                throw std::runtime_error(
                    std::string("SYCL copy failed: ") + e.what()
                );
            }
        }
    }

    auto synchronize(int32_t device_id) -> void override {
        validate_device_id(device_id);
        get_queue(device_id).wait_and_throw();
    }

    auto create_stream(int32_t device_id) -> StreamHandle override {
        validate_device_id(device_id);

        try {
            auto& device = devices_[device_id].device;
            auto* queue = new sycl::queue(device,
                sycl::property_list{sycl::property::queue::in_order{}});
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

    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor> override {
        // Validate inputs
        bool is_creation_op = (op_name == "zeros" || op_name == "ones" ||
                               op_name == "full" || op_name == "rand" || op_name == "randn");

        if (inputs.empty() && !is_creation_op) {
            throw std::invalid_argument("dispatch requires at least one input tensor");
        }

        // Validate device
        for (const auto& tensor : inputs) {
            if (tensor.device().type != Device::Type::OneAPI) {
                throw std::runtime_error(
                    "OneAPIBackend: All input tensors must be on OneAPI device, got: " +
                    tensor.device().to_string()
                );
            }
        }

        // Get device and queue
        int32_t device_id = 0;
        if (!inputs.empty()) {
            device_id = inputs[0].device().index;
        } else if (attrs.contains("device_id")) {
            device_id = std::stoi(attrs.at("device_id"));
        }

        validate_device_id(device_id);
        auto& queue = get_queue(device_id);

        // Dispatch to appropriate kernel
        try {
            // Binary operations
            if (op_name == "add") {
                if (inputs.size() != 2) throw std::invalid_argument("add requires 2 inputs");
                return {oneapi::add_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "sub") {
                if (inputs.size() != 2) throw std::invalid_argument("sub requires 2 inputs");
                return {oneapi::sub_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "mul") {
                if (inputs.size() != 2) throw std::invalid_argument("mul requires 2 inputs");
                return {oneapi::mul_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "div") {
                if (inputs.size() != 2) throw std::invalid_argument("div requires 2 inputs");
                return {oneapi::div_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "matmul") {
                if (inputs.size() != 2) throw std::invalid_argument("matmul requires 2 inputs");
                return {oneapi::matmul_kernel(inputs[0], inputs[1], queue)};
            }

            // In-place operations
            else if (op_name == "add_inplace") {
                if (inputs.size() != 2) throw std::invalid_argument("add_inplace requires 2 inputs");
                Tensor result = inputs[0];
                return {oneapi::add_inplace_kernel(result, inputs[1], queue)};
            }
            else if (op_name == "sub_inplace") {
                if (inputs.size() != 2) throw std::invalid_argument("sub_inplace requires 2 inputs");
                Tensor result = inputs[0];
                return {oneapi::sub_inplace_kernel(result, inputs[1], queue)};
            }
            else if (op_name == "mul_inplace") {
                if (inputs.size() != 2) throw std::invalid_argument("mul_inplace requires 2 inputs");
                Tensor result = inputs[0];
                return {oneapi::mul_inplace_kernel(result, inputs[1], queue)};
            }
            else if (op_name == "div_inplace") {
                if (inputs.size() != 2) throw std::invalid_argument("div_inplace requires 2 inputs");
                Tensor result = inputs[0];
                return {oneapi::div_inplace_kernel(result, inputs[1], queue)};
            }

            // Unary operations
            else if (op_name == "sqrt") {
                if (inputs.size() != 1) throw std::invalid_argument("sqrt requires 1 input");
                return {oneapi::sqrt_kernel(inputs[0], queue)};
            }
            else if (op_name == "neg") {
                if (inputs.size() != 1) throw std::invalid_argument("neg requires 1 input");
                return {oneapi::neg_kernel(inputs[0], queue)};
            }
            else if (op_name == "abs") {
                if (inputs.size() != 1) throw std::invalid_argument("abs requires 1 input");
                return {oneapi::abs_kernel(inputs[0], queue)};
            }
            else if (op_name == "log") {
                if (inputs.size() != 1) throw std::invalid_argument("log requires 1 input");
                return {oneapi::log_kernel(inputs[0], queue)};
            }
            else if (op_name == "exp") {
                if (inputs.size() != 1) throw std::invalid_argument("exp requires 1 input");
                return {oneapi::exp_kernel(inputs[0], queue)};
            }
            else if (op_name == "pow") {
                if (inputs.size() != 1) throw std::invalid_argument("pow requires 1 input");
                float exponent = attrs.contains("exponent") ? std::stof(attrs.at("exponent")) : 2.0f;
                return {oneapi::pow_kernel(inputs[0], exponent, queue)};
            }
            else if (op_name == "dot") {
                if (inputs.size() != 2) throw std::invalid_argument("dot requires 2 inputs");
                return {oneapi::dot_kernel(inputs[0], inputs[1], queue)};
            }

            // Activation functions
            else if (op_name == "relu") {
                if (inputs.size() != 1) throw std::invalid_argument("relu requires 1 input");
                return {oneapi::relu_kernel(inputs[0], queue)};
            }
            else if (op_name == "relu_backward") {
                if (inputs.size() != 2) throw std::invalid_argument("relu_backward requires 2 inputs");
                return {oneapi::relu_backward_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "sigmoid") {
                if (inputs.size() != 1) throw std::invalid_argument("sigmoid requires 1 input");
                return {oneapi::sigmoid_kernel(inputs[0], queue)};
            }
            else if (op_name == "sigmoid_backward") {
                if (inputs.size() != 2) throw std::invalid_argument("sigmoid_backward requires 2 inputs");
                return {oneapi::sigmoid_backward_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "tanh") {
                if (inputs.size() != 1) throw std::invalid_argument("tanh requires 1 input");
                return {oneapi::tanh_kernel(inputs[0], queue)};
            }
            else if (op_name == "tanh_backward") {
                if (inputs.size() != 2) throw std::invalid_argument("tanh_backward requires 2 inputs");
                return {oneapi::tanh_backward_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "gelu") {
                if (inputs.size() != 1) throw std::invalid_argument("gelu requires 1 input");
                return {oneapi::gelu_kernel(inputs[0], queue)};
            }
            else if (op_name == "gelu_backward") {
                if (inputs.size() != 2) throw std::invalid_argument("gelu_backward requires 2 inputs");
                return {oneapi::gelu_backward_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "softmax") {
                if (inputs.size() != 1) throw std::invalid_argument("softmax requires 1 input");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
                return {oneapi::softmax_kernel(inputs[0], dim, queue)};
            }
            else if (op_name == "softmax_backward") {
                if (inputs.size() != 2) throw std::invalid_argument("softmax_backward requires 2 inputs");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
                return {oneapi::softmax_backward_kernel(inputs[0], inputs[1], dim, queue)};
            }
            else if (op_name == "log_softmax") {
                if (inputs.size() != 1) throw std::invalid_argument("log_softmax requires 1 input");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
                return {oneapi::log_softmax_kernel(inputs[0], dim, queue)};
            }
            else if (op_name == "log_softmax_backward") {
                if (inputs.size() != 2) throw std::invalid_argument("log_softmax_backward requires 2 inputs");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
                return {oneapi::log_softmax_backward_kernel(inputs[0], inputs[1], dim, queue)};
            }
            else if (op_name == "leaky_relu") {
                if (inputs.size() != 1) throw std::invalid_argument("leaky_relu requires 1 input");
                float alpha = attrs.contains("alpha") ? std::stof(attrs.at("alpha")) : 0.01f;
                return {oneapi::leaky_relu_kernel(inputs[0], alpha, queue)};
            }
            else if (op_name == "leaky_relu_backward") {
                if (inputs.size() != 2) throw std::invalid_argument("leaky_relu_backward requires 2 inputs");
                float alpha = attrs.contains("alpha") ? std::stof(attrs.at("alpha")) : 0.01f;
                return {oneapi::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha, queue)};
            }
            else if (op_name == "swish") {
                if (inputs.size() != 1) throw std::invalid_argument("swish requires 1 input");
                return {oneapi::swish_kernel(inputs[0], queue)};
            }
            else if (op_name == "swish_backward") {
                if (inputs.size() != 2) throw std::invalid_argument("swish_backward requires 2 inputs");
                return {oneapi::swish_backward_kernel(inputs[0], inputs[1], queue)};
            }

            // Reduction operations
            // Use INT64_MIN as sentinel for "full reduction" when dim is not specified
            else if (op_name == "sum") {
                if (inputs.size() != 1) throw std::invalid_argument("sum requires 1 input");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : INT64_MIN;
                bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
                return {oneapi::sum_kernel(inputs[0], dim, keepdim, queue)};
            }
            else if (op_name == "mean") {
                if (inputs.size() != 1) throw std::invalid_argument("mean requires 1 input");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : INT64_MIN;
                bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
                return {oneapi::mean_kernel(inputs[0], dim, keepdim, queue)};
            }
            else if (op_name == "max") {
                if (inputs.size() != 1) throw std::invalid_argument("max requires 1 input");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : INT64_MIN;
                bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
                return {oneapi::max_kernel(inputs[0], dim, keepdim, queue)};
            }
            else if (op_name == "min") {
                if (inputs.size() != 1) throw std::invalid_argument("min requires 1 input");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : INT64_MIN;
                bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
                return {oneapi::min_kernel(inputs[0], dim, keepdim, queue)};
            }

            // Transform operations
            else if (op_name == "reshape") {
                if (inputs.size() != 1) throw std::invalid_argument("reshape requires 1 input");
                std::vector<int64_t> shape = parse_shape(attrs.at("shape"));
                return {oneapi::reshape_kernel(inputs[0], shape, queue)};
            }
            else if (op_name == "transpose") {
                if (inputs.size() != 1) throw std::invalid_argument("transpose requires 1 input");
                int64_t dim0 = attrs.contains("dim0") ? std::stoll(attrs.at("dim0")) : 0;
                int64_t dim1 = attrs.contains("dim1") ? std::stoll(attrs.at("dim1")) : 1;
                return {oneapi::transpose_kernel(inputs[0], dim0, dim1, queue)};
            }
            else if (op_name == "permute") {
                if (inputs.size() != 1) throw std::invalid_argument("permute requires 1 input");
                std::vector<int64_t> dims = parse_shape(attrs.at("dims"));
                return {oneapi::permute_kernel(inputs[0], dims, queue)};
            }
            else if (op_name == "squeeze") {
                if (inputs.size() != 1) throw std::invalid_argument("squeeze requires 1 input");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
                return {oneapi::squeeze_kernel(inputs[0], dim, queue)};
            }
            else if (op_name == "unsqueeze") {
                if (inputs.size() != 1) throw std::invalid_argument("unsqueeze requires 1 input");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : 0;
                return {oneapi::unsqueeze_kernel(inputs[0], dim, queue)};
            }
            else if (op_name == "index_select") {
                if (inputs.size() != 2) throw std::invalid_argument("index_select requires 2 inputs");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : 0;
                return {oneapi::index_select_kernel(inputs[0], dim, inputs[1], queue)};
            }
            else if (op_name == "gather") {
                if (inputs.size() != 2) throw std::invalid_argument("gather requires 2 inputs");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : 0;
                return {oneapi::gather_kernel(inputs[0], dim, inputs[1], queue)};
            }
            else if (op_name == "scatter") {
                if (inputs.size() != 3) throw std::invalid_argument("scatter requires 3 inputs");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : 0;
                return {oneapi::scatter_kernel(inputs[0], dim, inputs[1], inputs[2], queue)};
            }
            else if (op_name == "masked_select") {
                if (inputs.size() != 2) throw std::invalid_argument("masked_select requires 2 inputs");
                return {oneapi::masked_select_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "masked_fill") {
                if (inputs.size() != 2) throw std::invalid_argument("masked_fill requires 2 inputs");
                if (!attrs.contains("value")) throw std::invalid_argument("masked_fill requires value attribute");
                float value = std::stof(attrs.at("value"));
                return {oneapi::masked_fill_kernel(inputs[0], inputs[1], value, queue)};
            }
            else if (op_name == "contiguous") {
                if (inputs.size() != 1) throw std::invalid_argument("contiguous requires 1 input");
                return {oneapi::contiguous_kernel(inputs[0], queue)};
            }
            else if (op_name == "clone") {
                if (inputs.size() != 1) throw std::invalid_argument("clone requires 1 input");
                return {oneapi::clone_kernel(inputs[0], queue)};
            }

            // Fill operations
            else if (op_name == "zeros") {
                std::vector<int64_t> shape = parse_shape(attrs.at("shape"));
                DType dtype = parse_dtype(attrs);
                Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
                return {oneapi::zeros_kernel(shape, dtype, device, queue)};
            }
            else if (op_name == "ones") {
                std::vector<int64_t> shape = parse_shape(attrs.at("shape"));
                DType dtype = parse_dtype(attrs);
                Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
                return {oneapi::ones_kernel(shape, dtype, device, queue)};
            }
            else if (op_name == "full") {
                std::vector<int64_t> shape = parse_shape(attrs.at("shape"));
                float value = std::stof(attrs.at("value"));
                DType dtype = parse_dtype(attrs);
                Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
                return {oneapi::full_kernel(shape, value, dtype, device, queue)};
            }
            else if (op_name == "fill") {
                if (inputs.size() != 1) throw std::invalid_argument("fill requires 1 input");
                float value = std::stof(attrs.at("value"));
                return {oneapi::fill_kernel(inputs[0], value, queue)};
            }

            // Random number generation
            else if (op_name == "randn") {
                std::vector<int64_t> shape = parse_shape(attrs.at("shape"));
                DType dtype = parse_dtype(attrs);
                Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
                return {oneapi::randn_kernel(shape, dtype, device, queue)};
            }
            else if (op_name == "rand") {
                std::vector<int64_t> shape = parse_shape(attrs.at("shape"));
                DType dtype = parse_dtype(attrs);
                Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
                return {oneapi::rand_kernel(shape, dtype, device, queue)};
            }

            // Batch normalization
            else if (op_name == "batchnorm2d_mean_var") {
                if (inputs.size() != 1) throw std::invalid_argument("batchnorm2d_mean_var requires 1 input");
                return oneapi::batchnorm2d_mean_var(inputs[0], queue);
            }
            else if (op_name == "batchnorm2d_update_running_stats") {
                if (inputs.size() != 4) throw std::invalid_argument("batchnorm2d_update_running_stats requires 4 inputs");
                float momentum = attrs.contains("momentum") ? std::stof(attrs.at("momentum")) : 0.1f;
                // Clone the running stats as we'll modify them
                Tensor updated_mean = inputs[0].clone();
                Tensor updated_var = inputs[1].clone();
                oneapi::batchnorm2d_update_running_stats(updated_mean, updated_var, inputs[2], inputs[3], momentum, queue);
                return {updated_mean, updated_var};
            }
            else if (op_name == "batchnorm2d_forward") {
                if (inputs.size() != 3) throw std::invalid_argument("batchnorm2d_forward requires 3 inputs");
                float epsilon = attrs.contains("epsilon") ? std::stof(attrs.at("epsilon")) : 1e-5f;
                return {oneapi::batchnorm2d_forward(inputs[0], inputs[1], inputs[2], epsilon, queue)};
            }
            else if (op_name == "batchnorm2d_forward_affine") {
                if (inputs.size() != 5) throw std::invalid_argument("batchnorm2d_forward_affine requires 5 inputs");
                float epsilon = attrs.contains("epsilon") ? std::stof(attrs.at("epsilon")) : 1e-5f;
                return {oneapi::batchnorm2d_forward_affine(inputs[0], inputs[1], inputs[2],
                                                           inputs[3], inputs[4], epsilon, queue)};
            }
            else if (op_name == "batchnorm2d_backward") {
                if (inputs.size() != 5) throw std::invalid_argument("batchnorm2d_backward requires 5 inputs");
                float epsilon = attrs.contains("epsilon") ? std::stof(attrs.at("epsilon")) : 1e-5f;
                auto [grad_input, grad_gamma, grad_beta] = oneapi::batchnorm2d_backward(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, queue);
                return {grad_input, grad_gamma, grad_beta};
            }

            // Conv2d operations
            else if (op_name == "conv2d_forward") {
                if (inputs.size() < 2 || inputs.size() > 3) {
                    throw std::invalid_argument("conv2d_forward requires 2 or 3 inputs");
                }
                const Tensor* bias = inputs.size() == 3 ? &inputs[2] : nullptr;
                int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
                int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
                int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;
                int64_t groups = attrs.contains("groups") ? std::stoll(attrs.at("groups")) : 1;
                return {oneapi::conv2d_forward(inputs[0], inputs[1], bias, stride, padding,
                                               dilation, groups, queue)};
            }
            else if (op_name == "conv2d_backward") {
                if (inputs.size() != 3) throw std::invalid_argument("conv2d_backward requires 3 inputs");
                int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
                int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
                int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;
                int64_t groups = attrs.contains("groups") ? std::stoll(attrs.at("groups")) : 1;
                bool compute_grad_input = !attrs.contains("compute_grad_input") || attrs.at("compute_grad_input") == "1";
                bool compute_grad_weight = !attrs.contains("compute_grad_weight") || attrs.at("compute_grad_weight") == "1";
                bool compute_grad_bias = !attrs.contains("compute_grad_bias") || attrs.at("compute_grad_bias") == "1";

                auto [grad_input, grad_weight, grad_bias] = oneapi::conv2d_backward(
                    inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups,
                    compute_grad_input, compute_grad_weight, compute_grad_bias, queue);
                return {grad_input, grad_weight, grad_bias};
            }
            else if (op_name == "conv2d_backward_input") {
                // Computes gradient with respect to input
                // API: 2 inputs (grad_output, weight) + input_shape from attrs
                if (inputs.size() != 2) throw std::invalid_argument("conv2d_backward_input requires 2 inputs: grad_output, weight");

                // Parse input_shape from comma-separated string
                std::vector<int64_t> input_shape;
                if (attrs.contains("input_shape")) {
                    std::string shape_str = attrs.at("input_shape");
                    size_t pos = 0;
                    while (pos < shape_str.size()) {
                        size_t comma = shape_str.find(',', pos);
                        if (comma == std::string::npos) {
                            input_shape.push_back(std::stoll(shape_str.substr(pos)));
                            break;
                        }
                        input_shape.push_back(std::stoll(shape_str.substr(pos, comma - pos)));
                        pos = comma + 1;
                    }
                } else {
                    throw std::invalid_argument("conv2d_backward_input requires input_shape attribute");
                }

                // Extract attributes
                int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
                int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
                int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;
                int64_t groups = attrs.contains("groups") ? std::stoll(attrs.at("groups")) : 1;

                return {oneapi::conv2d_backward_input(inputs[0], inputs[1], input_shape,
                                                      stride, padding, dilation, groups, queue)};
            }
            else if (op_name == "conv2d_backward_weight") {
                // Computes gradient with respect to weight
                // API: 2 inputs (grad_output, input) + weight_shape from attrs
                if (inputs.size() != 2) throw std::invalid_argument("conv2d_backward_weight requires 2 inputs: grad_output, input");

                // Parse weight_shape from comma-separated string
                std::vector<int64_t> weight_shape;
                if (attrs.contains("weight_shape")) {
                    std::string shape_str = attrs.at("weight_shape");
                    size_t pos = 0;
                    while (pos < shape_str.size()) {
                        size_t comma = shape_str.find(',', pos);
                        if (comma == std::string::npos) {
                            weight_shape.push_back(std::stoll(shape_str.substr(pos)));
                            break;
                        }
                        weight_shape.push_back(std::stoll(shape_str.substr(pos, comma - pos)));
                        pos = comma + 1;
                    }
                } else {
                    throw std::invalid_argument("conv2d_backward_weight requires weight_shape attribute");
                }

                // Extract attributes
                int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
                int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
                int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;
                int64_t groups = attrs.contains("groups") ? std::stoll(attrs.at("groups")) : 1;

                return {oneapi::conv2d_backward_weight(inputs[0], inputs[1], weight_shape,
                                                       stride, padding, dilation, groups, queue)};
            }
            else if (op_name == "conv2d_backward_bias") {
                // Computes gradient with respect to bias
                // API: 1 input (grad_output)
                if (inputs.size() < 1) throw std::invalid_argument("conv2d_backward_bias requires at least 1 input: grad_output");

                return {oneapi::conv2d_backward_bias(inputs[0], queue)};
            }

            // Embedding operations
            else if (op_name == "embedding_lookup") {
                if (inputs.size() != 2) throw std::invalid_argument("embedding_lookup requires 2 inputs");
                int64_t padding_idx = attrs.contains("padding_idx") ? std::stoll(attrs.at("padding_idx")) : -1;
                return {oneapi::embedding_lookup_kernel(inputs[0], inputs[1], padding_idx, queue)};
            }
            else if (op_name == "embedding_backward") {
                if (inputs.size() != 2) throw std::invalid_argument("embedding_backward requires 2 inputs");
                int64_t vocab_size = std::stoll(attrs.at("vocab_size"));
                int64_t embedding_dim = std::stoll(attrs.at("embedding_dim"));
                return {oneapi::embedding_backward_kernel(inputs[0], inputs[1], vocab_size, embedding_dim, queue)};
            }
            else if (op_name == "embedding_bag_forward") {
                if (inputs.size() != 2) throw std::invalid_argument("embedding_bag_forward requires 2 inputs");
                std::string mode = attrs.contains("mode") ? attrs.at("mode") : "mean";
                bool include_last_offset = attrs.contains("include_last_offset") && attrs.at("include_last_offset") == "1";
                return {oneapi::embedding_bag_forward_kernel(inputs[0], inputs[1], mode, include_last_offset, queue)};
            }

            // Comparison operations
            else if (op_name == "eq") {
                if (inputs.size() != 2) throw std::invalid_argument("eq requires 2 inputs");
                return {oneapi::eq_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "ne") {
                if (inputs.size() != 2) throw std::invalid_argument("ne requires 2 inputs");
                return {oneapi::ne_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "lt") {
                if (inputs.size() != 2) throw std::invalid_argument("lt requires 2 inputs");
                return {oneapi::lt_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "le") {
                if (inputs.size() != 2) throw std::invalid_argument("le requires 2 inputs");
                return {oneapi::le_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "gt") {
                if (inputs.size() != 2) throw std::invalid_argument("gt requires 2 inputs");
                return {oneapi::gt_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "ge") {
                if (inputs.size() != 2) throw std::invalid_argument("ge requires 2 inputs");
                return {oneapi::ge_kernel(inputs[0], inputs[1], queue)};
            }

            // Utility operations
            else if (op_name == "cat") {
                if (inputs.empty()) throw std::invalid_argument("cat requires at least one input");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : 0;
                return {oneapi::cat_kernel(inputs, dim, queue)};
            }
            else if (op_name == "clamp") {
                if (inputs.size() != 1) throw std::invalid_argument("clamp requires 1 input");
                if (!attrs.contains("min") || !attrs.contains("max")) {
                    throw std::invalid_argument("clamp requires 'min' and 'max' attributes");
                }
                float min_val = std::stof(attrs.at("min"));
                float max_val = std::stof(attrs.at("max"));
                return {oneapi::clamp_kernel(inputs[0], min_val, max_val, queue)};
            }
            else if (op_name == "sign") {
                if (inputs.size() != 1) throw std::invalid_argument("sign requires 1 input");
                return {oneapi::sign_kernel(inputs[0], queue)};
            }

            // Im2col/Col2im operations
            else if (op_name == "im2col") {
                if (inputs.size() != 1) throw std::invalid_argument("im2col requires 1 input");
                return {oneapi::im2col_kernel(inputs[0], attrs, queue)};
            }
            else if (op_name == "col2im") {
                if (inputs.size() != 1) throw std::invalid_argument("col2im requires 1 input");
                return {oneapi::col2im_kernel(inputs[0], attrs, queue)};
            }

            // Expand operation
            else if (op_name == "expand") {
                if (inputs.size() != 1) throw std::invalid_argument("expand requires 1 input");
                return {oneapi::expand_kernel(inputs[0], attrs, queue)};
            }

            // Pooling operations
            else if (op_name == "avg_pool2d") {
                if (inputs.size() != 1) throw std::invalid_argument("avg_pool2d requires 1 input");
                return {oneapi::avg_pool2d_kernel(inputs[0], attrs, queue)};
            }
            else if (op_name == "max_pool2d") {
                if (inputs.size() != 1) throw std::invalid_argument("max_pool2d requires 1 input");
                auto [output, indices] = oneapi::max_pool2d_kernel(inputs[0], attrs, queue);
                return {output, indices};
            }
            else if (op_name == "adaptive_avg_pool2d") {
                if (inputs.size() != 1) throw std::invalid_argument("adaptive_avg_pool2d requires 1 input");
                return {oneapi::adaptive_avg_pool2d_kernel(inputs[0], attrs, queue)};
            }
            else if (op_name == "adaptive_max_pool2d") {
                if (inputs.size() != 1) throw std::invalid_argument("adaptive_max_pool2d requires 1 input");
                return {oneapi::adaptive_max_pool2d_kernel(inputs[0], attrs, queue)};
            }
            else if (op_name == "avg_pool2d_backward") {
                if (inputs.size() != 2) throw std::invalid_argument("avg_pool2d_backward requires 2 inputs: grad_output, input");
                return {oneapi::avg_pool2d_backward_kernel(inputs[0], inputs[1], attrs, queue)};
            }
            else if (op_name == "max_pool2d_backward") {
                if (inputs.size() != 2) throw std::invalid_argument("max_pool2d_backward requires 2 inputs: grad_output, indices");
                int64_t H_in = 0, W_in = 0;
                if (attrs.contains("H_in")) {
                    H_in = std::stoll(attrs.at("H_in"));
                }
                if (attrs.contains("W_in")) {
                    W_in = std::stoll(attrs.at("W_in"));
                }
                return {oneapi::max_pool2d_backward_with_indices(inputs[0], inputs[1], H_in, W_in, queue)};
            }
            else if (op_name == "adaptive_avg_pool2d_backward") {
                if (inputs.size() != 1) throw std::invalid_argument("adaptive_avg_pool2d_backward requires 1 input: grad_output");
                int64_t H_in = 0, W_in = 0;
                if (attrs.contains("H_in")) {
                    H_in = std::stoll(attrs.at("H_in"));
                }
                if (attrs.contains("W_in")) {
                    W_in = std::stoll(attrs.at("W_in"));
                }
                return {oneapi::adaptive_avgpool2d_backward(inputs[0], H_in, W_in, queue)};
            }

            // Statistical operations
            else if (op_name == "std") {
                if (inputs.size() != 1) throw std::invalid_argument("std requires 1 input");
                return {oneapi::std_kernel(inputs[0], attrs, queue)};
            }
            else if (op_name == "norm") {
                if (inputs.size() != 1) throw std::invalid_argument("norm requires 1 input");
                return {oneapi::norm_kernel(inputs[0], attrs, queue)};
            }
            else if (op_name == "var") {
                if (inputs.size() != 1) throw std::invalid_argument("var requires 1 input");
                return {oneapi::var_kernel(inputs[0], attrs, queue)};
            }
            else if (op_name == "prod") {
                if (inputs.size() != 1) throw std::invalid_argument("prod requires 1 input");
                return {oneapi::prod_kernel(inputs[0], attrs, queue)};
            }

            // Argmax/Argmin operations
            else if (op_name == "argmax") {
                if (inputs.size() != 1) throw std::invalid_argument("argmax requires 1 input");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
                bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
                return {oneapi::argmax_kernel(inputs[0], dim, keepdim, queue)};
            }
            else if (op_name == "argmin") {
                if (inputs.size() != 1) throw std::invalid_argument("argmin requires 1 input");
                int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
                bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
                return {oneapi::argmin_kernel(inputs[0], dim, keepdim, queue)};
            }

            // ================================================================
            // Trigonometric operations
            // ================================================================
            else if (op_name == "sin") {
                if (inputs.size() != 1) throw std::invalid_argument("sin requires 1 input");
                return {oneapi::sin_kernel(inputs[0], queue)};
            }
            else if (op_name == "cos") {
                if (inputs.size() != 1) throw std::invalid_argument("cos requires 1 input");
                return {oneapi::cos_kernel(inputs[0], queue)};
            }
            else if (op_name == "tan") {
                if (inputs.size() != 1) throw std::invalid_argument("tan requires 1 input");
                return {oneapi::tan_kernel(inputs[0], queue)};
            }
            else if (op_name == "asin") {
                if (inputs.size() != 1) throw std::invalid_argument("asin requires 1 input");
                return {oneapi::asin_kernel(inputs[0], queue)};
            }
            else if (op_name == "acos") {
                if (inputs.size() != 1) throw std::invalid_argument("acos requires 1 input");
                return {oneapi::acos_kernel(inputs[0], queue)};
            }
            else if (op_name == "atan") {
                if (inputs.size() != 1) throw std::invalid_argument("atan requires 1 input");
                return {oneapi::atan_kernel(inputs[0], queue)};
            }
            else if (op_name == "sinh") {
                if (inputs.size() != 1) throw std::invalid_argument("sinh requires 1 input");
                return {oneapi::sinh_kernel(inputs[0], queue)};
            }
            else if (op_name == "cosh") {
                if (inputs.size() != 1) throw std::invalid_argument("cosh requires 1 input");
                return {oneapi::cosh_kernel(inputs[0], queue)};
            }
            else if (op_name == "atan2") {
                if (inputs.size() != 2) throw std::invalid_argument("atan2 requires 2 inputs");
                return {oneapi::atan2_kernel(inputs[0], inputs[1], queue)};
            }

            // ================================================================
            // Rounding operations
            // ================================================================
            else if (op_name == "round") {
                if (inputs.size() != 1) throw std::invalid_argument("round requires 1 input");
                return {oneapi::round_kernel(inputs[0], queue)};
            }
            else if (op_name == "floor") {
                if (inputs.size() != 1) throw std::invalid_argument("floor requires 1 input");
                return {oneapi::floor_kernel(inputs[0], queue)};
            }
            else if (op_name == "ceil") {
                if (inputs.size() != 1) throw std::invalid_argument("ceil requires 1 input");
                return {oneapi::ceil_kernel(inputs[0], queue)};
            }
            else if (op_name == "trunc") {
                if (inputs.size() != 1) throw std::invalid_argument("trunc requires 1 input");
                return {oneapi::trunc_kernel(inputs[0], queue)};
            }
            else if (op_name == "reciprocal") {
                if (inputs.size() != 1) throw std::invalid_argument("reciprocal requires 1 input");
                return {oneapi::reciprocal_kernel(inputs[0], queue)};
            }

            // ================================================================
            // Additional utility operations
            // ================================================================
            else if (op_name == "clamp_min") {
                if (inputs.size() != 1) throw std::invalid_argument("clamp_min requires 1 input");
                float min_val = attrs.contains("min") ? std::stof(attrs.at("min")) : 0.0f;
                return {oneapi::clamp_min_kernel(inputs[0], min_val, queue)};
            }
            else if (op_name == "clamp_max") {
                if (inputs.size() != 1) throw std::invalid_argument("clamp_max requires 1 input");
                float max_val = attrs.contains("max") ? std::stof(attrs.at("max")) : 1.0f;
                return {oneapi::clamp_max_kernel(inputs[0], max_val, queue)};
            }
            else if (op_name == "where") {
                if (inputs.size() != 3) throw std::invalid_argument("where requires 3 inputs (condition, x, y)");
                return {oneapi::where_kernel(inputs[0], inputs[1], inputs[2], queue)};
            }
            else if (op_name == "repeat") {
                if (inputs.size() != 1) throw std::invalid_argument("repeat requires 1 input");
                auto repeats = parse_shape(attrs.at("repeats"));
                return {oneapi::repeat_kernel(inputs[0], repeats, queue)};
            }

            // ================================================================
            // Fused operations
            // ================================================================
            else if (op_name == "fused_add_relu") {
                if (inputs.size() != 2) throw std::invalid_argument("fused_add_relu requires 2 inputs");
                return {oneapi::fused_add_relu_kernel(inputs[0], inputs[1], queue)};
            }
            else if (op_name == "fused_gelu") {
                if (inputs.size() != 1) throw std::invalid_argument("fused_gelu requires 1 input");
                return {oneapi::fused_gelu_kernel(inputs[0], queue)};
            }
            else if (op_name == "fused_layer_norm") {
                if (inputs.size() != 3) throw std::invalid_argument("fused_layer_norm requires 3 inputs");
                auto normalized_shape = parse_shape(attrs.at("normalized_shape"));
                float epsilon = attrs.contains("epsilon") ? std::stof(attrs.at("epsilon")) : 1e-5f;
                auto [output, mean, inv_std] = oneapi::fused_layer_norm_kernel(
                    inputs[0], inputs[1], inputs[2], normalized_shape, epsilon, queue);
                return {output, mean, inv_std};
            }
            else if (op_name == "fused_layer_norm_backward") {
                if (inputs.size() != 5) throw std::invalid_argument("fused_layer_norm_backward requires 5 inputs");
                auto normalized_shape = parse_shape(attrs.at("normalized_shape"));
                auto [grad_input, grad_weight, grad_bias] = oneapi::fused_layer_norm_backward_kernel(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], normalized_shape, queue);
                return {grad_input, grad_weight, grad_bias};
            }
            else if (op_name == "fused_linear_relu") {
                if (inputs.size() < 2) throw std::invalid_argument("fused_linear_relu requires 2-3 inputs");
                const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
                return {oneapi::fused_linear_relu_kernel(inputs[0], inputs[1], bias, queue)};
            }
            else if (op_name == "fused_batchnorm_relu") {
                if (inputs.size() != 5) throw std::invalid_argument("fused_batchnorm_relu requires 5 inputs");
                float epsilon = attrs.contains("epsilon") ? std::stof(attrs.at("epsilon")) : 1e-5f;
                return {oneapi::fused_batchnorm_relu_kernel(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, queue)};
            }
            else if (op_name == "fused_matmul_add") {
                if (inputs.size() != 3) throw std::invalid_argument("fused_matmul_add requires 3 inputs");
                return {oneapi::fused_matmul_add_kernel(inputs[0], inputs[1], inputs[2], queue)};
            }
            else if (op_name == "fused_softmax_cross_entropy") {
                if (inputs.size() != 2) throw std::invalid_argument("fused_softmax_cross_entropy requires 2 inputs");
                std::string reduction = attrs.contains("reduction") ? attrs.at("reduction") : "mean";
                return {oneapi::fused_softmax_cross_entropy_kernel(inputs[0], inputs[1], reduction, queue)};
            }

            // ================================================================
            // LSTM operations
            // ================================================================
            else if (op_name == "lstm_cell_forward") {
                if (inputs.size() != 2) throw std::invalid_argument("lstm_cell_forward requires 2 inputs");
                int64_t batch_size = std::stoll(attrs.at("batch_size"));
                int64_t hidden_size = std::stoll(attrs.at("hidden_size"));
                auto [h_out, c_out] = oneapi::lstm_cell_forward_kernel(
                    inputs[0], inputs[1], batch_size, hidden_size, queue);
                return {h_out, c_out};
            }
            else if (op_name == "lstm_cell_backward") {
                if (inputs.size() != 5) throw std::invalid_argument("lstm_cell_backward requires 5 inputs");
                int64_t batch_size = std::stoll(attrs.at("batch_size"));
                int64_t hidden_size = std::stoll(attrs.at("hidden_size"));
                auto [grad_gates, grad_c_prev] = oneapi::lstm_cell_backward_kernel(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                    batch_size, hidden_size, queue);
                return {grad_gates, grad_c_prev};
            }

            // ================================================================
            // GRU operations
            // ================================================================
            else if (op_name == "gru_cell_forward") {
                if (inputs.size() != 5) throw std::invalid_argument("gru_cell_forward requires 5 inputs");
                int64_t batch_size = std::stoll(attrs.at("batch_size"));
                int64_t hidden_size = std::stoll(attrs.at("hidden_size"));
                return {oneapi::gru_cell_forward_kernel(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                    batch_size, hidden_size, queue)};
            }
            else if (op_name == "gru_cell_backward") {
                if (inputs.size() != 6) throw std::invalid_argument("gru_cell_backward requires 6 inputs");
                int64_t batch_size = std::stoll(attrs.at("batch_size"));
                int64_t hidden_size = std::stoll(attrs.at("hidden_size"));
                auto outputs = oneapi::gru_cell_backward_kernel(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                    batch_size, hidden_size, queue);
                return {outputs.grad_reset, outputs.grad_update, outputs.grad_new_input,
                        outputs.grad_new_hidden, outputs.grad_h_prev};
            }

            // ================================================================
            // Vision operations
            // ================================================================
            else if (op_name == "nms") {
                if (inputs.size() != 2) throw std::invalid_argument("nms requires 2 inputs (boxes, scores)");
                float iou_threshold = attrs.contains("iou_threshold") ? std::stof(attrs.at("iou_threshold")) : 0.5f;
                return {oneapi::nms_kernel(inputs[0], inputs[1], iou_threshold, queue)};
            }
            else if (op_name == "roi_align") {
                if (inputs.size() != 2) throw std::invalid_argument("roi_align requires 2 inputs");
                int64_t output_height = std::stoll(attrs.at("output_height"));
                int64_t output_width = std::stoll(attrs.at("output_width"));
                float spatial_scale = attrs.contains("spatial_scale") ? std::stof(attrs.at("spatial_scale")) : 1.0f;
                int64_t sampling_ratio = attrs.contains("sampling_ratio") ? std::stoll(attrs.at("sampling_ratio")) : 0;
                bool aligned = attrs.contains("aligned") && attrs.at("aligned") == "1";
                return {oneapi::roi_align_kernel(inputs[0], inputs[1], output_height, output_width,
                                                 spatial_scale, sampling_ratio, aligned, queue)};
            }
            else if (op_name == "gather_relative_position_bias") {
                if (inputs.size() != 2) throw std::invalid_argument("gather_relative_position_bias requires 2 inputs (table, indices)");
                int64_t num_positions = std::stoll(attrs.at("num_positions"));
                int64_t num_heads = std::stoll(attrs.at("num_heads"));
                return {oneapi::gather_relative_position_bias_kernel(inputs[0], inputs[1], num_positions, num_heads, queue)};
            }

            // ================================================================
            // Quantization operations
            // ================================================================
            else if (op_name == "quantize") {
                if (inputs.size() != 1) throw std::invalid_argument("quantize requires 1 input");
                float scale = std::stof(attrs.at("scale"));
                int32_t zero_point = std::stoi(attrs.at("zero_point"));
                return {oneapi::quantize_kernel(inputs[0], scale, zero_point, queue)};
            }
            else if (op_name == "dequantize") {
                if (inputs.size() != 1) throw std::invalid_argument("dequantize requires 1 input");
                float scale = std::stof(attrs.at("scale"));
                int32_t zero_point = std::stoi(attrs.at("zero_point"));
                return {oneapi::dequantize_kernel(inputs[0], scale, zero_point, queue)};
            }
            else if (op_name == "quantized_linear") {
                if (inputs.size() < 2) throw std::invalid_argument("quantized_linear requires 2-3 inputs");
                const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
                float input_scale = std::stof(attrs.at("input_scale"));
                int32_t input_zero_point = std::stoi(attrs.at("input_zero_point"));
                float weight_scale = std::stof(attrs.at("weight_scale"));
                int32_t weight_zero_point = std::stoi(attrs.at("weight_zero_point"));
                float output_scale = attrs.contains("output_scale") ? std::stof(attrs.at("output_scale")) : 1.0f;
                int32_t output_zero_point = attrs.contains("output_zero_point") ? std::stoi(attrs.at("output_zero_point")) : 0;
                return {oneapi::quantized_linear_kernel(inputs[0], inputs[1], bias,
                                                        input_scale, input_zero_point,
                                                        weight_scale, weight_zero_point,
                                                        output_scale, output_zero_point, queue)};
            }
            else if (op_name == "quantized_conv2d") {
                if (inputs.size() < 2) throw std::invalid_argument("quantized_conv2d requires 2-3 inputs");
                const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
                int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
                int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
                int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;
                int64_t groups = attrs.contains("groups") ? std::stoll(attrs.at("groups")) : 1;
                float input_scale = std::stof(attrs.at("input_scale"));
                int32_t input_zero_point = std::stoi(attrs.at("input_zero_point"));
                float weight_scale = std::stof(attrs.at("weight_scale"));
                int32_t weight_zero_point = std::stoi(attrs.at("weight_zero_point"));
                return {oneapi::quantized_conv2d_kernel(inputs[0], inputs[1], bias,
                                                        stride, padding, dilation, groups,
                                                        input_scale, input_zero_point,
                                                        weight_scale, weight_zero_point, queue)};
            }

            else {
                throw std::runtime_error("OneAPIBackend: Unknown operation '" + op_name + "'");
            }
        }
        catch (const sycl::exception& e) {
            throw std::runtime_error(
                "OneAPIBackend: Operation '" + op_name + "' failed with SYCL error: " + e.what()
            );
        }
    }

private:
    struct DeviceInfo {
        std::shared_ptr<sycl::queue> queue;
        sycl::device device;
        std::string name;
        std::string type;
        uint32_t max_compute_units;
        size_t max_work_group_size;
        uint64_t global_mem_size;
        uint64_t local_mem_size;
    };

    std::vector<DeviceInfo> devices_;
    std::unordered_map<void*, int32_t> allocations_;

    auto get_queue(int32_t device_id) -> sycl::queue& {
        return *devices_[device_id].queue;
    }

    auto validate_device_id(int32_t device_id) const -> void {
        if (device_id < 0 || device_id >= static_cast<int32_t>(devices_.size())) {
            throw std::invalid_argument(
                "Invalid device ID " + std::to_string(device_id) +
                " (available: 0-" + std::to_string(devices_.size() - 1) + ")"
            );
        }
    }

    auto parse_shape(const std::string& shape_str) const -> std::vector<int64_t> {
        std::vector<int64_t> shape;
        std::string str = shape_str;
        size_t pos = 0;
        while ((pos = str.find(',')) != std::string::npos) {
            shape.push_back(std::stoll(str.substr(0, pos)));
            str.erase(0, pos + 1);
        }
        if (!str.empty()) {
            shape.push_back(std::stoll(str));
        }
        return shape;
    }

    auto parse_dtype(const OpAttributes& attrs) const -> DType {
        if (!attrs.contains("dtype")) {
            return DType::Float32;
        }

        const auto& dtype_str = attrs.at("dtype");
        if (dtype_str == "float32") return DType::Float32;
        if (dtype_str == "float64") return DType::Float64;
        if (dtype_str == "float16") return DType::Float16;
        if (dtype_str == "bfloat16") return DType::BFloat16;
        if (dtype_str == "int8") return DType::Int8;
        if (dtype_str == "int16") return DType::Int16;
        if (dtype_str == "int32") return DType::Int32;
        if (dtype_str == "int64") return DType::Int64;
        if (dtype_str == "uint8") return DType::UInt8;
        if (dtype_str == "uint16") return DType::UInt16;
        if (dtype_str == "uint32") return DType::UInt32;
        if (dtype_str == "uint64") return DType::UInt64;
        if (dtype_str == "bool") return DType::Bool;

        return DType::Float32;
    }
};

extern "C" {
    auto create_backend() -> std::unique_ptr<Backend> {
        return std::make_unique<OneAPIBackend>();
    }
}

} // namespace tenzor
