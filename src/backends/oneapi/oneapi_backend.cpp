#include "oneapi_internal.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/backend/oneapi_caching_allocator.hpp"
#include "tenzor/utils/logging.hpp"
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
    auto bmm_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;

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

    // Creation operations
    auto arange_kernel(double start, double end, double step, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto linspace_kernel(double start, double end, int64_t steps, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto eye_kernel(int64_t n, int64_t m, DType dtype, Device device, sycl::queue& queue) -> Tensor;

    // Group normalization
    auto group_norm_kernel(const Tensor& input, int64_t num_groups,
                           const Tensor* weight, const Tensor* bias,
                           float eps, sycl::queue& queue) -> std::vector<Tensor>;
    auto group_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                    const Tensor& mean, const Tensor& rstd,
                                    const Tensor& weight, int64_t num_groups,
                                    sycl::queue& queue) -> std::vector<Tensor>;

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

    // ConvTranspose2d operations
    auto conv_transpose2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                   int64_t stride, int64_t padding, int64_t output_padding,
                                   int64_t dilation, int64_t groups, sycl::queue& queue) -> Tensor;

    // Argsort operation
    auto argsort_kernel(const Tensor& input, int64_t dim, bool descending, sycl::queue& queue) -> Tensor;

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
    auto adaptive_maxpool2d_backward(const Tensor& grad_output, const Tensor& indices, int64_t H_in, int64_t W_in, sycl::queue& queue) -> Tensor;

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
    auto fused_rms_norm_kernel(const Tensor& input, const Tensor& weight, float eps,
                                sycl::queue& queue) -> std::tuple<Tensor, Tensor>;
    auto rms_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                   const Tensor& weight, const Tensor& rrms,
                                   sycl::queue& queue) -> std::tuple<Tensor, Tensor>;

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
    auto roi_align_backward_kernel(const Tensor& grad_output, const Tensor& rois,
                                   int64_t batch_size, int64_t channels,
                                   int64_t feat_height, int64_t feat_width,
                                   float spatial_scale, int64_t sampling_ratio, bool aligned,
                                   sycl::queue& queue) -> Tensor;
    auto interpolate_kernel(const Tensor& input, const std::vector<int64_t>& size,
                            const std::string& mode, bool align_corners,
                            sycl::queue& queue) -> Tensor;

    // In-place activation operations
    auto relu_inplace_kernel(Tensor& input, sycl::queue& queue) -> void;
    auto sigmoid_inplace_kernel(Tensor& input, sycl::queue& queue) -> void;
    auto tanh_inplace_kernel(Tensor& input, sycl::queue& queue) -> void;
    auto leaky_relu_inplace_kernel(Tensor& input, float alpha, sycl::queue& queue) -> void;
    auto gelu_inplace_kernel(Tensor& input, sycl::queue& queue) -> void;

    // Indexing operations
    auto nonzero_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto one_hot_kernel(const Tensor& indices, int64_t num_classes, DType output_dtype,
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
        return g_queue_getter(g_backend_ptr, device_id);
    }
} // namespace oneapi_internal

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

        try {
            // Enumerate all available SYCL devices
            auto platforms = sycl::platform::get_platforms();

            for (const auto& platform : platforms) {
                auto devices = platform.get_devices();
                for (const auto& device : devices) {
                    // Skip NVIDIA GPUs - kernels are compiled for spir64 (Intel CPU/GPU)
                    std::string vendor = device.get_info<sycl::info::device::vendor>();
                    if (vendor.find("NVIDIA") != std::string::npos ||
                        vendor.find("nvidia") != std::string::npos) {
                        continue;
                    }

                    // Create queue for each device
                    try {
                        auto queue = std::make_shared<sycl::queue>(device,
                            sycl::property_list{sycl::property::queue::in_order{}});

                        // Store device info
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

                        // Initialize caching allocator for this device
                        backend::OneAPICachingAllocator::get().initialize(
                            dev_data.queue.get(), static_cast<int>(devices_.size()));

                        devices_.push_back(dev_data);
                    } catch (const sycl::exception& e) {
                        TENZOR_LOG_WARNING(
                            std::string("Skipping SYCL device: ") + e.what());
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

    auto get_device_info(int32_t device_id) const -> tenzor::DeviceInfo override {
        if (device_id < 0 || device_id >= static_cast<int32_t>(devices_.size())) {
            throw std::out_of_range("Invalid OneAPI device ID: " + std::to_string(device_id) +
                                    " (available: 0-" + std::to_string(devices_.size() - 1) + ")");
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
        try {
            auto sub_group_sizes = dev.device.get_info<sycl::info::device::sub_group_sizes>();
            if (!sub_group_sizes.empty()) {
                info.warp_size = static_cast<int>(sub_group_sizes.front());
            }
        } catch (...) {
            info.warp_size = 32;  // Default
        }

        // Feature support
        info.supports_fp16 = dev.device.has(sycl::aspect::fp16);
        info.supports_fp64 = dev.device.has(sycl::aspect::fp64);
        info.supports_int8 = true;  // Generally supported

        // Device type
        info.is_integrated = !dev.device.is_gpu() ||
            (dev.device.get_info<sycl::info::device::host_unified_memory>());
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

        // Return memory to caching allocator for reuse
        backend::OneAPICachingAllocator::get().free(ptr, device_id);
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

            case CopyKind::HostToDevice: {
                // Use destination device's queue for H2D
                if (devices_.empty()) {
                    throw std::runtime_error("No SYCL devices available for copy");
                }
                auto dst_it = allocations_.find(dst);
                int32_t dev_id = (dst_it != allocations_.end()) ? dst_it->second : 0;
                if (dev_id < 0 || dev_id >= static_cast<int32_t>(devices_.size())) dev_id = 0;
                queue_ptr = devices_[dev_id].queue.get();
                break;
            }
            case CopyKind::DeviceToHost: {
                // Use source device's queue for D2H
                if (devices_.empty()) {
                    throw std::runtime_error("No SYCL devices available for copy");
                }
                auto src_it = allocations_.find(const_cast<void*>(src));
                int32_t dev_id = (src_it != allocations_.end()) ? src_it->second : 0;
                if (dev_id < 0 || dev_id >= static_cast<int32_t>(devices_.size())) dev_id = 0;
                queue_ptr = devices_[dev_id].queue.get();
                break;
            }
            case CopyKind::DeviceToDevice: {
                // Use destination device's queue for D2D
                if (devices_.empty()) {
                    throw std::runtime_error("No SYCL devices available for copy");
                }
                auto dst_it = allocations_.find(dst);
                int32_t dev_id = (dst_it != allocations_.end()) ? dst_it->second : 0;
                if (dev_id < 0 || dev_id >= static_cast<int32_t>(devices_.size())) dev_id = 0;
                queue_ptr = devices_[dev_id].queue.get();
                break;
            }
        }

        if (queue_ptr) {
            try {
                auto event = queue_ptr->memcpy(dst, src, bytes);
                // Only block for D2H where caller expects data immediately
                if (kind == CopyKind::DeviceToHost) {
                    event.wait();
                }
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

    auto memset(void* ptr, int value, size_t bytes, int32_t device_id) -> void override {
        validate_device_id(device_id);
        try {
            get_queue(device_id).memset(ptr, value, bytes).wait();
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL memset failed: ") + e.what());
        }
    }

    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor> override {
        throw std::runtime_error("OneAPIBackend::dispatch(string): operation '" + op_name +
            "' not available via legacy string dispatch. Use OpId-based dispatch instead.");
    }

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
        if (!attrs.has(AttrKey::Dtype)) {
            return DType::Float32;
        }

        const auto& dtype_str = std::string(attrs.get_string(AttrKey::Dtype));
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
