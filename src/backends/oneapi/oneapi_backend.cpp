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

    // Conv2d operations (Audit J.2: per-axis stride/padding/dilation)
    auto conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                       int64_t stride_h, int64_t stride_w,
                       int64_t padding_h, int64_t padding_w,
                       int64_t dilation_h, int64_t dilation_w,
                       int64_t groups, sycl::queue& queue) -> Tensor;
    auto conv2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                        int64_t stride_h, int64_t stride_w,
                        int64_t padding_h, int64_t padding_w,
                        int64_t dilation_h, int64_t dilation_w,
                        int64_t groups,
                        bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias,
                        sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
    // Separate backward operations (matching CPU API)
    auto conv2d_backward_input(const Tensor& grad_output, const Tensor& weight,
                               const std::vector<int64_t>& input_shape,
                               int64_t stride_h, int64_t stride_w,
                               int64_t padding_h, int64_t padding_w,
                               int64_t dilation_h, int64_t dilation_w,
                               int64_t groups,
                               sycl::queue& queue) -> Tensor;
    auto conv2d_backward_weight(const Tensor& grad_output, const Tensor& input,
                                const std::vector<int64_t>& weight_shape,
                                int64_t stride_h, int64_t stride_w,
                                int64_t padding_h, int64_t padding_w,
                                int64_t dilation_h, int64_t dilation_w,
                                int64_t groups,
                                sycl::queue& queue) -> Tensor;
    auto conv2d_backward_bias(const Tensor& grad_output, sycl::queue& queue) -> Tensor;

    // ConvTranspose2d operations
    auto conv_transpose2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                   int64_t sH, int64_t sW, int64_t pH, int64_t pW,
                                   int64_t opH, int64_t opW, int64_t dH, int64_t dW,
                                   int64_t groups, sycl::queue& queue) -> Tensor;

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
                                     sycl::queue& queue) -> std::vector<Tensor>;
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
    auto flash_attention_kernel(const Tensor& Q, const Tensor& K, const Tensor& V,
                                const Tensor* mask, float scale, bool is_causal,
                                sycl::queue& queue) -> Tensor;

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
                    sycl::property_list{sycl::property::queue::in_order{}});

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
        // Wait for all in-flight work to finish.
        for (size_t i = 0; i < devices_.size(); ++i) {
            devices_[i].queue->wait_and_throw();
        }

        // Release ALL USM allocations (cached and in-use) while SYCL
        // queues are still alive.
        backend::OneAPICachingAllocator::get().release_all();

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
        info.supports_int8 = true;  // Generally supported

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
                throw std::runtime_error("Attempt to free untracked pointer");
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
                int32_t dev_id;
                {
                    std::lock_guard<std::mutex> lock(allocations_mutex_);
                    auto dst_it = allocations_.find(dst);
                    dev_id = (dst_it != allocations_.end()) ? dst_it->second : 0;
                }
                if (dev_id < 0 || dev_id >= static_cast<int32_t>(devices_.size())) dev_id = 0;
                queue_ptr = devices_[dev_id].queue.get();
                break;
            }
            case CopyKind::DeviceToHost: {
                // Use source device's queue for D2H
                if (devices_.empty()) {
                    throw std::runtime_error("No SYCL devices available for copy");
                }
                int32_t dev_id;
                {
                    std::lock_guard<std::mutex> lock(allocations_mutex_);
                    auto src_it = allocations_.find(const_cast<void*>(src));
                    dev_id = (src_it != allocations_.end()) ? src_it->second : 0;
                }
                if (dev_id < 0 || dev_id >= static_cast<int32_t>(devices_.size())) dev_id = 0;
                queue_ptr = devices_[dev_id].queue.get();
                break;
            }
            case CopyKind::DeviceToDevice: {
                // Use destination device's queue for D2D
                if (devices_.empty()) {
                    throw std::runtime_error("No SYCL devices available for copy");
                }
                int32_t dev_id;
                {
                    std::lock_guard<std::mutex> lock(allocations_mutex_);
                    auto dst_it = allocations_.find(dst);
                    dev_id = (dst_it != allocations_.end()) ? dst_it->second : 0;
                }
                if (dev_id < 0 || dev_id >= static_cast<int32_t>(devices_.size())) dev_id = 0;
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
            auto* queue = new sycl::queue(device,
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
