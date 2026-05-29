#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>

#ifdef TENZOR_HAS_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#endif

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#endif

namespace tenzor {
namespace oneapi {

// Saturating float-to-half conversion: clamps to Float16 representable range
// instead of producing Infinity on overflow (matches Vulkan/GPU hardware behavior)
constexpr float HALF_MAX = 65504.0f;
inline sycl::half saturate_to_half(float val) {
    return sycl::half(sycl::fmin(sycl::fmax(val, -HALF_MAX), HALF_MAX));
}

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

#ifdef TENZOR_HAS_ONEDNN

// Map Tenzor DType to oneDNN memory data type
static auto to_dnnl_dtype(DType dt) -> dnnl::memory::data_type {
    switch (dt) {
        case DType::Float32:  return dnnl::memory::data_type::f32;
        case DType::Float64:  return dnnl::memory::data_type::f64;
        case DType::BFloat16: return dnnl::memory::data_type::bf16;
        case DType::Float16:  return dnnl::memory::data_type::f16;
        default: throw std::runtime_error("Unsupported dtype for oneDNN conv: " + dtype_name(dt));
    }
}

// Conv2d forward using oneDNN.
// Audit J.2: per-axis stride/padding/dilation — oneDNN's memory::dims
// natively supports asymmetric H/W values; the previous scalar signature
// silently dropped the W-axis value.
auto conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                    int64_t stride_h, int64_t stride_w,
                    int64_t padding_h, int64_t padding_w,
                    int64_t dilation_h, int64_t dilation_w,
                    int64_t groups,
                    sycl::queue& queue) -> Tensor {
    using namespace dnnl;

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    if (input_shape.size() != 4 || weight_shape.size() != 4) {
        throw std::invalid_argument("Conv2d requires 4D tensors");
    }

    const int64_t N = input_shape[0];   // Batch
    const int64_t C_in = input_shape[1]; // Input channels
    const int64_t H_in = input_shape[2]; // Input height
    const int64_t W_in = input_shape[3]; // Input width

    const int64_t C_out = weight_shape[0]; // Output channels
    const int64_t K_h = weight_shape[2];   // Kernel height
    const int64_t K_w = weight_shape[3];   // Kernel width

    // Calculate output dimensions (per-axis)
    const int64_t H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    const int64_t W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    Tensor output({N, C_out, H_out, W_out}, input.dtype(), input.device());

    // Create oneDNN engine and stream
    auto dnnl_engine = sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto dnnl_stream = sycl_interop::make_stream(dnnl_engine, queue);

    // Create memory descriptors
    memory::dims src_dims = {N, C_in, H_in, W_in};
    memory::dims weights_dims = {C_out, C_in / groups, K_h, K_w};
    memory::dims dst_dims = {N, C_out, H_out, W_out};
    memory::dims strides_dims = {stride_h, stride_w};
    memory::dims padding_dims = {padding_h, padding_w};
    memory::dims dilation_dims = {dilation_h - 1, dilation_w - 1};

    auto dt = to_dnnl_dtype(input.dtype());
    auto src_md = memory::desc(src_dims, dt, memory::format_tag::nchw);
    auto weights_md = memory::desc(weights_dims, dt,
                                   groups == 1 ? memory::format_tag::oihw : memory::format_tag::goihw);
    auto dst_md = memory::desc(dst_dims, dt, memory::format_tag::nchw);

    // Create convolution descriptor
    auto conv_desc = (bias != nullptr) ?
        convolution_forward::desc(
            prop_kind::forward_inference,
            algorithm::convolution_direct,
            src_md, weights_md,
            memory::desc({C_out}, dt, memory::format_tag::x),
            dst_md,
            strides_dims, dilation_dims, padding_dims, padding_dims
        ) :
        convolution_forward::desc(
            prop_kind::forward_inference,
            algorithm::convolution_direct,
            src_md, weights_md, dst_md,
            strides_dims, dilation_dims, padding_dims, padding_dims
        );

    auto conv_pd = convolution_forward::primitive_desc(conv_desc, dnnl_engine);

    // Wrap tensors as oneDNN memory
    auto src_mem = sycl_interop::make_memory(conv_pd.src_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(input.data_ptr()));

    auto weights_mem = sycl_interop::make_memory(conv_pd.weights_desc(), dnnl_engine,
                                                  sycl_interop::memory_kind::usm,
                                                  const_cast<void*>(weight.data_ptr()));

    auto dst_mem = sycl_interop::make_memory(conv_pd.dst_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(output.data_ptr()));

    // Execute convolution
    auto conv_prim = convolution_forward(conv_pd);

    try {
        if (bias != nullptr) {
            auto bias_mem = sycl_interop::make_memory(
                memory::desc({C_out}, dt, memory::format_tag::x),
                dnnl_engine,
                sycl_interop::memory_kind::usm,
                const_cast<void*>(bias->data_ptr())
            );

            conv_prim.execute(dnnl_stream, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_WEIGHTS, weights_mem},
                {DNNL_ARG_BIAS, bias_mem},
                {DNNL_ARG_DST, dst_mem}
            });
        } else {
            conv_prim.execute(dnnl_stream, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_WEIGHTS, weights_mem},
                {DNNL_ARG_DST, dst_mem}
            });
        }

        dnnl_stream.wait();
    } catch (const dnnl::error& e) {
        throw std::runtime_error(std::string("oneDNN Conv2d forward failed: ") + e.what());
    } catch (const sycl::exception& e) {
        throw std::runtime_error(std::string("SYCL error in Conv2d forward: ") + e.what());
    }

    return output;
}

// Conv2d backward using oneDNN. Audit J.2: per-axis stride/padding/dilation.
auto conv2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                     int64_t stride_h, int64_t stride_w,
                     int64_t padding_h, int64_t padding_w,
                     int64_t dilation_h, int64_t dilation_w,
                     int64_t groups,
                     bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias,
                     sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor> {
    using namespace dnnl;

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    auto grad_output_shape = grad_output.shape();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t H_in = input_shape[2];
    const int64_t W_in = input_shape[3];

    const int64_t C_out = weight_shape[0];
    const int64_t K_h = weight_shape[2];
    const int64_t K_w = weight_shape[3];

    // Create oneDNN engine and stream
    auto dnnl_engine = sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto dnnl_stream = sycl_interop::make_stream(dnnl_engine, queue);

    // Memory descriptors
    memory::dims src_dims = {N, C_in, H_in, W_in};
    memory::dims weights_dims = {C_out, C_in / groups, K_h, K_w};
    memory::dims dst_dims = {grad_output_shape[0], grad_output_shape[1],
                             grad_output_shape[2], grad_output_shape[3]};
    memory::dims strides_dims = {stride_h, stride_w};
    memory::dims padding_dims = {padding_h, padding_w};
    memory::dims dilation_dims = {dilation_h - 1, dilation_w - 1};

    auto dt = to_dnnl_dtype(input.dtype());
    auto src_md = memory::desc(src_dims, dt, memory::format_tag::nchw);
    auto weights_md = memory::desc(weights_dims, dt,
                                   groups == 1 ? memory::format_tag::oihw : memory::format_tag::goihw);
    auto dst_md = memory::desc(dst_dims, dt, memory::format_tag::nchw);

    // Forward descriptor (needed for backward)
    auto conv_fwd_desc = convolution_forward::desc(
        prop_kind::forward_training,
        algorithm::convolution_direct,
        src_md, weights_md, dst_md,
        strides_dims, dilation_dims, padding_dims, padding_dims
    );
    auto conv_fwd_pd = convolution_forward::primitive_desc(conv_fwd_desc, dnnl_engine);

    Tensor grad_input, grad_weight, grad_bias;

    // Compute grad_input
    if (compute_grad_input) {
        grad_input = Tensor(std::vector<int64_t>(input_shape.begin(), input_shape.end()), input.dtype(), input.device());

        auto conv_bwd_data_desc = convolution_backward_data::desc(
            algorithm::convolution_direct,
            src_md, weights_md, dst_md,
            strides_dims, dilation_dims, padding_dims, padding_dims
        );
        auto conv_bwd_data_pd = convolution_backward_data::primitive_desc(
            conv_bwd_data_desc, dnnl_engine, conv_fwd_pd
        );

        auto diff_src_mem = sycl_interop::make_memory(conv_bwd_data_pd.diff_src_desc(), dnnl_engine,
                                                       sycl_interop::memory_kind::usm,
                                                       const_cast<void*>(grad_input.data_ptr()));

        auto weights_mem = sycl_interop::make_memory(conv_bwd_data_pd.weights_desc(), dnnl_engine,
                                                      sycl_interop::memory_kind::usm,
                                                      const_cast<void*>(weight.data_ptr()));

        auto diff_dst_mem = sycl_interop::make_memory(conv_bwd_data_pd.diff_dst_desc(), dnnl_engine,
                                                       sycl_interop::memory_kind::usm,
                                                       const_cast<void*>(grad_output.data_ptr()));

        auto conv_bwd_data_prim = convolution_backward_data(conv_bwd_data_pd);
        try {
            conv_bwd_data_prim.execute(dnnl_stream, {
                {DNNL_ARG_DIFF_DST, diff_dst_mem},
                {DNNL_ARG_WEIGHTS, weights_mem},
                {DNNL_ARG_DIFF_SRC, diff_src_mem}
            });
        } catch (const dnnl::error& e) {
            throw std::runtime_error(std::string("oneDNN Conv2d backward (input grad) failed: ") + e.what());
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL error in Conv2d backward (input grad): ") + e.what());
        }
    }

    // Compute grad_weight and grad_bias
    if (compute_grad_weight || compute_grad_bias) {
        grad_weight = Tensor(std::vector<int64_t>(weight_shape.begin(), weight_shape.end()), weight.dtype(), weight.device());

        auto conv_bwd_weights_desc = compute_grad_bias ?
            convolution_backward_weights::desc(
                algorithm::convolution_direct,
                src_md, weights_md,
                memory::desc({C_out}, dt, memory::format_tag::x),
                dst_md,
                strides_dims, dilation_dims, padding_dims, padding_dims
            ) :
            convolution_backward_weights::desc(
                algorithm::convolution_direct,
                src_md, weights_md, dst_md,
                strides_dims, dilation_dims, padding_dims, padding_dims
            );

        auto conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(
            conv_bwd_weights_desc, dnnl_engine, conv_fwd_pd
        );

        auto src_mem = sycl_interop::make_memory(conv_bwd_weights_pd.src_desc(), dnnl_engine,
                                                  sycl_interop::memory_kind::usm,
                                                  const_cast<void*>(input.data_ptr()));

        auto diff_weights_mem = sycl_interop::make_memory(conv_bwd_weights_pd.diff_weights_desc(), dnnl_engine,
                                                           sycl_interop::memory_kind::usm,
                                                           const_cast<void*>(grad_weight.data_ptr()));

        auto diff_dst_mem = sycl_interop::make_memory(conv_bwd_weights_pd.diff_dst_desc(), dnnl_engine,
                                                       sycl_interop::memory_kind::usm,
                                                       const_cast<void*>(grad_output.data_ptr()));

        auto conv_bwd_weights_prim = convolution_backward_weights(conv_bwd_weights_pd);

        if (compute_grad_bias) {
            grad_bias = Tensor({C_out}, weight.dtype(), weight.device());

            auto diff_bias_mem = sycl_interop::make_memory(
                memory::desc({C_out}, dt, memory::format_tag::x),
                dnnl_engine,
                sycl_interop::memory_kind::usm,
                const_cast<void*>(grad_bias.data_ptr())
            );

            try {
                conv_bwd_weights_prim.execute(dnnl_stream, {
                    {DNNL_ARG_SRC, src_mem},
                    {DNNL_ARG_DIFF_DST, diff_dst_mem},
                    {DNNL_ARG_DIFF_WEIGHTS, diff_weights_mem},
                    {DNNL_ARG_DIFF_BIAS, diff_bias_mem}
                });
            } catch (const dnnl::error& e) {
                throw std::runtime_error(std::string("oneDNN Conv2d backward (weight grad) failed: ") + e.what());
            } catch (const sycl::exception& e) {
                throw std::runtime_error(std::string("SYCL error in Conv2d backward (weight grad): ") + e.what());
            }
        } else {
            try {
                conv_bwd_weights_prim.execute(dnnl_stream, {
                    {DNNL_ARG_SRC, src_mem},
                    {DNNL_ARG_DIFF_DST, diff_dst_mem},
                    {DNNL_ARG_DIFF_WEIGHTS, diff_weights_mem}
                });
            } catch (const dnnl::error& e) {
                throw std::runtime_error(std::string("oneDNN Conv2d backward (weight grad) failed: ") + e.what());
            } catch (const sycl::exception& e) {
                throw std::runtime_error(std::string("SYCL error in Conv2d backward (weight grad): ") + e.what());
            }
        }
    }

    try {
        dnnl_stream.wait();
    } catch (const dnnl::error& e) {
        throw std::runtime_error(std::string("oneDNN Conv2d backward failed: ") + e.what());
    } catch (const sycl::exception& e) {
        throw std::runtime_error(std::string("SYCL error in Conv2d backward: ") + e.what());
    }

    return {grad_input, grad_weight, grad_bias};
}

#else // !TENZOR_HAS_ONEDNN - Fallback implementation using im2col + GEMM

// Kernel class declarations for conv2d operations - Float32
class Conv2dIm2colKernel;
class Conv2dCol2imKernel;
class Conv2dForwardGemmFallback;
class Conv2dForwardBiasAdd;
class Conv2dBackwardInputGemmFallback;
class Conv2dBackwardWeightGemmFallback;
class Conv2dBackwardBiasReduction;
// Grouped convolution kernel classes - Float32
class Conv2dGroupedIm2colKernel;
class Conv2dGroupedCol2imKernel;
class Conv2dGroupedForwardGemmFallback;
class Conv2dGroupedBackwardInputGemmFallback;
class Conv2dGroupedBackwardWeightGemmFallback;

// Float64 kernel classes
class Conv2dIm2colKernelFloat64;
class Conv2dCol2imKernelFloat64;
class Conv2dForwardGemmFallbackFloat64;
class Conv2dForwardBiasAddFloat64;
class Conv2dBackwardInputGemmFallbackFloat64;
class Conv2dBackwardWeightGemmFallbackFloat64;
class Conv2dBackwardBiasReductionFloat64;
class Conv2dGroupedIm2colKernelFloat64;
class Conv2dGroupedCol2imKernelFloat64;
class Conv2dGroupedForwardGemmFallbackFloat64;
class Conv2dGroupedBackwardInputGemmFallbackFloat64;
class Conv2dGroupedBackwardWeightGemmFallbackFloat64;

// Float16 kernel classes
class Conv2dIm2colKernelFloat16;
class Conv2dCol2imKernelFloat16;
class Conv2dForwardGemmFallbackFloat16;
class Conv2dForwardBiasAddFloat16;
class Conv2dBackwardInputGemmFallbackFloat16;
class Conv2dBackwardWeightGemmFallbackFloat16;
class Conv2dBackwardBiasReductionFloat16;
class Conv2dGroupedIm2colKernelFloat16;
class Conv2dGroupedCol2imKernelFloat16;
class Conv2dGroupedForwardGemmFallbackFloat16;
class Conv2dGroupedBackwardInputGemmFallbackFloat16;
class Conv2dGroupedBackwardWeightGemmFallbackFloat16;

// Im2col transformation for convolution (per-axis pad/stride/dilation)
template<typename T>
void im2col_kernel(const T* data_im, int64_t channels, int64_t height, int64_t width,
                   int64_t kernel_h, int64_t kernel_w,
                   int64_t pad_h, int64_t pad_w,
                   int64_t stride_h, int64_t stride_w,
                   int64_t dilation_h, int64_t dilation_w,
                   T* data_col, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t output_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    const int64_t col_size = channels * kernel_h * kernel_w * output_h * output_w;

    queue.parallel_for<Conv2dIm2colKernel>(sycl::range<1>(col_size), [=](sycl::id<1> index) {
        int64_t w_out = index % output_w;
        int64_t idx = index / output_w;
        int64_t h_out = idx % output_h;
        idx /= output_h;
        int64_t kw = idx % kernel_w;
        idx /= kernel_w;
        int64_t kh = idx % kernel_h;
        int64_t c = idx / kernel_h;

        int64_t h_in = h_out * stride_h - pad_h + kh * dilation_h;
        int64_t w_in = w_out * stride_w - pad_w + kw * dilation_w;

        data_col[index] = (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) ?
            data_im[(c * height + h_in) * width + w_in] : T(0);
    });
}

// Col2im transformation (inverse of im2col, per-axis pad/stride/dilation)
template<typename T>
void col2im_kernel(const T* data_col, int64_t channels, int64_t height, int64_t width,
                   int64_t kernel_h, int64_t kernel_w,
                   int64_t pad_h, int64_t pad_w,
                   int64_t stride_h, int64_t stride_w,
                   int64_t dilation_h, int64_t dilation_w,
                   T* data_im, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t output_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    const int64_t im_size = channels * height * width;

    // Initialize grad_input to zero
    queue.fill(data_im, T(0), im_size);

    // Accumulate gradients from col buffer
    queue.parallel_for<Conv2dCol2imKernel>(sycl::range<1>(channels * kernel_h * kernel_w * output_h * output_w),
                      [=](sycl::id<1> index) {
        int64_t w_out = index % output_w;
        int64_t idx = index / output_w;
        int64_t h_out = idx % output_h;
        idx /= output_h;
        int64_t kw = idx % kernel_w;
        idx /= kernel_w;
        int64_t kh = idx % kernel_h;
        int64_t c = idx / kernel_h;

        int64_t h_in = h_out * stride_h - pad_h + kh * dilation_h;
        int64_t w_in = w_out * stride_w - pad_w + kw * dilation_w;

        if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
            int64_t im_idx = (c * height + h_in) * width + w_in;
            // Atomic add for thread safety
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
                atomic_val(data_im[im_idx]);
            atomic_val.fetch_add(data_col[index]);
        }
    });
}

// Grouped im2col - type-specific overloads for SYCL kernel naming.
// Audit J.2: per-axis pad/stride/dilation so asymmetric Conv2d
// (e.g., stride={2,1}) lowers to the correct sliding window.
// Float32 version
void im2col_grouped_kernel(const float* data_im, int64_t total_channels, int64_t channels_per_group,
                           int64_t channel_offset, int64_t height, int64_t width,
                           int64_t kernel_h, int64_t kernel_w,
                           int64_t pad_h, int64_t pad_w,
                           int64_t stride_h, int64_t stride_w,
                           int64_t dilation_h, int64_t dilation_w,
                           float* data_col, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t output_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    const int64_t col_size = channels_per_group * kernel_h * kernel_w * output_h * output_w;

    queue.parallel_for<Conv2dGroupedIm2colKernel>(sycl::range<1>(col_size), [=](sycl::id<1> index) {
        int64_t w_out = index % output_w;
        int64_t idx = index / output_w;
        int64_t h_out = idx % output_h;
        idx /= output_h;
        int64_t kw = idx % kernel_w;
        idx /= kernel_w;
        int64_t kh = idx % kernel_h;
        int64_t c_local = idx / kernel_h;
        int64_t c_global = channel_offset + c_local;
        int64_t h_in = h_out * stride_h - pad_h + kh * dilation_h;
        int64_t w_in = w_out * stride_w - pad_w + kw * dilation_w;
        data_col[index] = (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) ?
            data_im[(c_global * height + h_in) * width + w_in] : 0.0f;
    });
}

// Float64 version
void im2col_grouped_kernel(const double* data_im, int64_t total_channels, int64_t channels_per_group,
                           int64_t channel_offset, int64_t height, int64_t width,
                           int64_t kernel_h, int64_t kernel_w,
                           int64_t pad_h, int64_t pad_w,
                           int64_t stride_h, int64_t stride_w,
                           int64_t dilation_h, int64_t dilation_w,
                           double* data_col, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t output_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    const int64_t col_size = channels_per_group * kernel_h * kernel_w * output_h * output_w;

    queue.parallel_for<Conv2dGroupedIm2colKernelFloat64>(sycl::range<1>(col_size), [=](sycl::id<1> index) {
        int64_t w_out = index % output_w;
        int64_t idx = index / output_w;
        int64_t h_out = idx % output_h;
        idx /= output_h;
        int64_t kw = idx % kernel_w;
        idx /= kernel_w;
        int64_t kh = idx % kernel_h;
        int64_t c_local = idx / kernel_h;
        int64_t c_global = channel_offset + c_local;
        int64_t h_in = h_out * stride_h - pad_h + kh * dilation_h;
        int64_t w_in = w_out * stride_w - pad_w + kw * dilation_w;
        data_col[index] = (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) ?
            data_im[(c_global * height + h_in) * width + w_in] : 0.0;
    });
}

// Float16 version
void im2col_grouped_kernel(const sycl::half* data_im, int64_t total_channels, int64_t channels_per_group,
                           int64_t channel_offset, int64_t height, int64_t width,
                           int64_t kernel_h, int64_t kernel_w,
                           int64_t pad_h, int64_t pad_w,
                           int64_t stride_h, int64_t stride_w,
                           int64_t dilation_h, int64_t dilation_w,
                           sycl::half* data_col, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t output_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    const int64_t col_size = channels_per_group * kernel_h * kernel_w * output_h * output_w;

    queue.parallel_for<Conv2dGroupedIm2colKernelFloat16>(sycl::range<1>(col_size), [=](sycl::id<1> index) {
        int64_t w_out = index % output_w;
        int64_t idx = index / output_w;
        int64_t h_out = idx % output_h;
        idx /= output_h;
        int64_t kw = idx % kernel_w;
        idx /= kernel_w;
        int64_t kh = idx % kernel_h;
        int64_t c_local = idx / kernel_h;
        int64_t c_global = channel_offset + c_local;
        int64_t h_in = h_out * stride_h - pad_h + kh * dilation_h;
        int64_t w_in = w_out * stride_w - pad_w + kw * dilation_w;
        data_col[index] = (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) ?
            data_im[(c_global * height + h_in) * width + w_in] : sycl::half(0.0f);
    });
}

// Grouped col2im - type-specific overloads. Audit J.2: per-axis
// pad/stride/dilation so asymmetric Conv2d backward maps gradients back
// to the correct input locations.
// Float32 version
void col2im_grouped_kernel(const float* data_col, int64_t total_channels, int64_t channels_per_group,
                           int64_t channel_offset, int64_t height, int64_t width,
                           int64_t kernel_h, int64_t kernel_w,
                           int64_t pad_h, int64_t pad_w,
                           int64_t stride_h, int64_t stride_w,
                           int64_t dilation_h, int64_t dilation_w,
                           float* data_im, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t output_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    queue.parallel_for<Conv2dGroupedCol2imKernel>(
        sycl::range<1>(channels_per_group * kernel_h * kernel_w * output_h * output_w),
        [=](sycl::id<1> index) {
            int64_t w_out = index % output_w;
            int64_t idx = index / output_w;
            int64_t h_out = idx % output_h;
            idx /= output_h;
            int64_t kw = idx % kernel_w;
            idx /= kernel_w;
            int64_t kh = idx % kernel_h;
            int64_t c_local = idx / kernel_h;
            int64_t c_global = channel_offset + c_local;
            int64_t h_in = h_out * stride_h - pad_h + kh * dilation_h;
            int64_t w_in = w_out * stride_w - pad_w + kw * dilation_w;
            if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
                int64_t im_idx = (c_global * height + h_in) * width + w_in;
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                    atomic_val(data_im[im_idx]);
                atomic_val.fetch_add(data_col[index]);
            }
        });
}

// Float64 version
void col2im_grouped_kernel(const double* data_col, int64_t total_channels, int64_t channels_per_group,
                           int64_t channel_offset, int64_t height, int64_t width,
                           int64_t kernel_h, int64_t kernel_w,
                           int64_t pad_h, int64_t pad_w,
                           int64_t stride_h, int64_t stride_w,
                           int64_t dilation_h, int64_t dilation_w,
                           double* data_im, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t output_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    queue.parallel_for<Conv2dGroupedCol2imKernelFloat64>(
        sycl::range<1>(channels_per_group * kernel_h * kernel_w * output_h * output_w),
        [=](sycl::id<1> index) {
            int64_t w_out = index % output_w;
            int64_t idx = index / output_w;
            int64_t h_out = idx % output_h;
            idx /= output_h;
            int64_t kw = idx % kernel_w;
            idx /= kernel_w;
            int64_t kh = idx % kernel_h;
            int64_t c_local = idx / kernel_h;
            int64_t c_global = channel_offset + c_local;
            int64_t h_in = h_out * stride_h - pad_h + kh * dilation_h;
            int64_t w_in = w_out * stride_w - pad_w + kw * dilation_w;
            if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
                int64_t im_idx = (c_global * height + h_in) * width + w_in;
                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>
                    atomic_val(data_im[im_idx]);
                atomic_val.fetch_add(data_col[index]);
            }
        });
}

// Float16 version - uses float accumulation for atomics (half atomics may not be supported)
void col2im_grouped_kernel(const sycl::half* data_col, int64_t total_channels, int64_t channels_per_group,
                           int64_t channel_offset, int64_t height, int64_t width,
                           int64_t kernel_h, int64_t kernel_w,
                           int64_t pad_h, int64_t pad_w,
                           int64_t stride_h, int64_t stride_w,
                           int64_t dilation_h, int64_t dilation_w,
                           sycl::half* data_im, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t output_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Audit F10: atomic accumulation into a Float32 USM scratch buffer, then
    // narrowed back into the existing Float16 grad-input buffer at the end.
    // Previously this kernel did a non-atomic read-modify-write of `data_im`
    // (`data_im[im_idx] = sycl::half(... + val)`) under the assumption that
    // atomic_ref<sycl::half> isn't supported. Multiple work-items hitting
    // the same im_idx (the common case: any non-1x1 conv) raced and
    // silently corrupted F16 grouped-conv backward gradients.
    //
    // We can't use atomic_ref<sycl::half> portably, but atomic_ref<float>
    // is universal — so accumulate per-element in a float scratch and
    // narrow once at the end. The scratch is sized to the full data_im
    // image (total_channels * height * width); since the caller adds into
    // `data_im` across batches/groups, we initialise scratch from the
    // current `data_im` contents and write back the merged result.
    const int64_t im_total = total_channels * height * width;
    Tensor scratch_tensor({im_total}, DType::Float32,
                          Device{Device::Type::OneAPI, 0});  // queue's device — single-device OneAPI build
    float* scratch = get_data_ptr<float>(scratch_tensor);

    // Initialise scratch from current data_im (widen). One thread per pixel.
    sycl::half* data_im_ptr = data_im;  // capture by value into the lambda
    queue.parallel_for(sycl::range<1>(im_total), [=](sycl::id<1> i) {
        scratch[i] = static_cast<float>(data_im_ptr[i]);
    }).wait();

    // Accumulate atomically into scratch.
    queue.parallel_for<Conv2dGroupedCol2imKernelFloat16>(
        sycl::range<1>(channels_per_group * kernel_h * kernel_w * output_h * output_w),
        [=](sycl::id<1> index) {
            int64_t w_out = index % output_w;
            int64_t idx = index / output_w;
            int64_t h_out = idx % output_h;
            idx /= output_h;
            int64_t kw = idx % kernel_w;
            idx /= kernel_w;
            int64_t kh = idx % kernel_h;
            int64_t c_local = idx / kernel_h;
            int64_t c_global = channel_offset + c_local;
            int64_t h_in = h_out * stride_h - pad_h + kh * dilation_h;
            int64_t w_in = w_out * stride_w - pad_w + kw * dilation_w;
            if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
                int64_t im_idx = (c_global * height + h_in) * width + w_in;
                float val = static_cast<float>(data_col[index]);
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                    atomic_val(scratch[im_idx]);
                atomic_val.fetch_add(val);
            }
        }).wait();

    // Narrow scratch back into data_im.
    queue.parallel_for(sycl::range<1>(im_total), [=](sycl::id<1> i) {
        data_im_ptr[i] = sycl::half(scratch[i]);
    }).wait();
}

// Audit J.2: per-axis stride/padding/dilation for the im2col+GEMM fallback path.
auto conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                    int64_t stride_h, int64_t stride_w,
                    int64_t padding_h, int64_t padding_w,
                    int64_t dilation_h, int64_t dilation_w,
                    int64_t groups,
                    sycl::queue& queue) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t H_in = input_shape[2];
    const int64_t W_in = input_shape[3];

    const int64_t C_out = weight_shape[0];
    const int64_t C_in_per_group = weight_shape[1];  // This is C_in / groups
    const int64_t K_h = weight_shape[2];
    const int64_t K_w = weight_shape[3];

    const int64_t C_out_per_group = C_out / groups;

    const int64_t H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    const int64_t W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    Tensor output({N, C_out, H_out, W_out}, input.dtype(), input.device());

    // Common dimensions for GEMM
    const int64_t col_size = C_in_per_group * K_h * K_w * H_out * W_out;
    const int64_t M = C_out_per_group;
    const int64_t N_gemm = H_out * W_out;
    const int64_t K = C_in_per_group * K_h * K_w;

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr<const float>(input);
        const float* weight_ptr = get_data_ptr<const float>(weight);
        float* output_ptr = get_data_ptr<float>(output);

        Tensor col_buffer({col_size}, input.dtype(), input.device());
        float* col_ptr = get_data_ptr<float>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * C_in_per_group * K_h * K_w;

                im2col_grouped_kernel(
                    input_ptr + n * C_in * H_in * W_in,
                    C_in, C_in_per_group, in_channel_offset,
                    H_in, W_in, K_h, K_w,
                    padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                    col_ptr, queue
                );

                const float* weight_group_ptr = weight_ptr + weight_offset;
                float* output_group_ptr = output_ptr + n * C_out * H_out * W_out + out_channel_offset * H_out * W_out;

                queue.parallel_for<Conv2dGroupedForwardGemmFallback>(sycl::range<2>(M, N_gemm), [=](sycl::id<2> idx) {
                    const int64_t oc_local = idx[0];
                    const int64_t hw = idx[1];
                    float sum = 0.0f;
                    for (int64_t k = 0; k < K; ++k) {
                        sum += weight_group_ptr[oc_local * K + k] * col_ptr[k * N_gemm + hw];
                    }
                    output_group_ptr[oc_local * N_gemm + hw] = sum;
                });
            }

            if (bias != nullptr) {
                const float* bias_ptr = get_data_ptr<const float>(*bias);
                queue.parallel_for<Conv2dForwardBiasAdd>(sycl::range<2>(C_out, H_out * W_out), [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];
                    const int64_t hw = idx[1];
                    const int64_t out_idx = n * C_out * H_out * W_out + oc * H_out * W_out + hw;
                    output_ptr[out_idx] += bias_ptr[oc];
                });
            }
        }
    }
    else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr<const double>(input);
        const double* weight_ptr = get_data_ptr<const double>(weight);
        double* output_ptr = get_data_ptr<double>(output);

        Tensor col_buffer({col_size}, input.dtype(), input.device());
        double* col_ptr = get_data_ptr<double>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * C_in_per_group * K_h * K_w;

                im2col_grouped_kernel(
                    input_ptr + n * C_in * H_in * W_in,
                    C_in, C_in_per_group, in_channel_offset,
                    H_in, W_in, K_h, K_w,
                    padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                    col_ptr, queue
                );

                const double* weight_group_ptr = weight_ptr + weight_offset;
                double* output_group_ptr = output_ptr + n * C_out * H_out * W_out + out_channel_offset * H_out * W_out;

                queue.parallel_for<Conv2dGroupedForwardGemmFallbackFloat64>(sycl::range<2>(M, N_gemm), [=](sycl::id<2> idx) {
                    const int64_t oc_local = idx[0];
                    const int64_t hw = idx[1];
                    double sum = 0.0;
                    for (int64_t k = 0; k < K; ++k) {
                        sum += weight_group_ptr[oc_local * K + k] * col_ptr[k * N_gemm + hw];
                    }
                    output_group_ptr[oc_local * N_gemm + hw] = sum;
                });
            }

            if (bias != nullptr) {
                const double* bias_ptr = get_data_ptr<const double>(*bias);
                queue.parallel_for<Conv2dForwardBiasAddFloat64>(sycl::range<2>(C_out, H_out * W_out), [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];
                    const int64_t hw = idx[1];
                    const int64_t out_idx = n * C_out * H_out * W_out + oc * H_out * W_out + hw;
                    output_ptr[out_idx] += bias_ptr[oc];
                });
            }
        }
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* input_ptr = get_data_ptr<const sycl::half>(input);
        const sycl::half* weight_ptr = get_data_ptr<const sycl::half>(weight);
        sycl::half* output_ptr = get_data_ptr<sycl::half>(output);

        Tensor col_buffer({col_size}, input.dtype(), input.device());
        sycl::half* col_ptr = get_data_ptr<sycl::half>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * C_in_per_group * K_h * K_w;

                im2col_grouped_kernel(
                    input_ptr + n * C_in * H_in * W_in,
                    C_in, C_in_per_group, in_channel_offset,
                    H_in, W_in, K_h, K_w,
                    padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                    col_ptr, queue
                );

                const sycl::half* weight_group_ptr = weight_ptr + weight_offset;
                sycl::half* output_group_ptr = output_ptr + n * C_out * H_out * W_out + out_channel_offset * H_out * W_out;

                // For Float16, use float accumulation for better precision
                queue.parallel_for<Conv2dGroupedForwardGemmFallbackFloat16>(sycl::range<2>(M, N_gemm), [=](sycl::id<2> idx) {
                    const int64_t oc_local = idx[0];
                    const int64_t hw = idx[1];
                    float sum = 0.0f;  // Accumulate in float for precision
                    for (int64_t k = 0; k < K; ++k) {
                        sum += static_cast<float>(weight_group_ptr[oc_local * K + k]) *
                               static_cast<float>(col_ptr[k * N_gemm + hw]);
                    }
                    // Saturate to Float16 range to prevent Inf/NaN propagation
                    output_group_ptr[oc_local * N_gemm + hw] = saturate_to_half(sum);
                });
            }

            if (bias != nullptr) {
                const sycl::half* bias_ptr = get_data_ptr<const sycl::half>(*bias);
                queue.parallel_for<Conv2dForwardBiasAddFloat16>(sycl::range<2>(C_out, H_out * W_out), [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];
                    const int64_t hw = idx[1];
                    const int64_t out_idx = n * C_out * H_out * W_out + oc * H_out * W_out + hw;
                    output_ptr[out_idx] = saturate_to_half(static_cast<float>(output_ptr[out_idx]) +
                                                           static_cast<float>(bias_ptr[oc]));
                });
            }
        }
    }
    else {
        throw std::runtime_error("Unsupported dtype for conv2d_forward (fallback)");
    }

    return output;
}

// Audit J.2: per-axis stride/padding/dilation for the im2col+GEMM fallback path.
auto conv2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                     int64_t stride_h, int64_t stride_w,
                     int64_t padding_h, int64_t padding_w,
                     int64_t dilation_h, int64_t dilation_w,
                     int64_t groups,
                     bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias,
                     sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor> {
    Tensor grad_input, grad_weight, grad_bias;

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    auto grad_output_shape = grad_output.shape();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t H_in = input_shape[2];
    const int64_t W_in = input_shape[3];

    const int64_t C_out = weight_shape[0];
    const int64_t C_in_per_group = weight_shape[1];  // C_in / groups
    const int64_t K_h = weight_shape[2];
    const int64_t K_w = weight_shape[3];

    const int64_t C_out_per_group = C_out / groups;

    const int64_t H_out = grad_output_shape[2];
    const int64_t W_out = grad_output_shape[3];

    if (input.dtype() != DType::Float32) {
        throw std::runtime_error("Unsupported dtype for conv2d_backward (fallback)");
    }

    const float* input_ptr = get_data_ptr<const float>(input);
    const float* weight_ptr = get_data_ptr<const float>(weight);
    const float* grad_output_ptr = get_data_ptr<const float>(grad_output);

    // Compute grad_input: transpose convolution of grad_output with weight
    if (compute_grad_input) {
        grad_input = Tensor(std::vector<int64_t>(input_shape.begin(), input_shape.end()), input.dtype(), input.device());
        float* grad_input_ptr = get_data_ptr<float>(grad_input);

        // Initialize grad_input to zero (needed for grouped col2im accumulation)
        queue.fill(grad_input_ptr, 0.0f, N * C_in * H_in * W_in);

        // Col buffer for one group
        const int64_t col_size = C_in_per_group * K_h * K_w * H_out * W_out;
        Tensor col_buffer({col_size}, input.dtype(), input.device());
        float* col_ptr = get_data_ptr<float>(col_buffer);

        // Process each batch
        for (int64_t n = 0; n < N; ++n) {
            float* grad_in_batch = grad_input_ptr + n * C_in * H_in * W_in;

            // Process each group
            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * C_in_per_group * K_h * K_w;

                const float* grad_out_group = grad_output_ptr + n * C_out * H_out * W_out + out_channel_offset * H_out * W_out;
                const float* weight_group = weight_ptr + weight_offset;

                // GEMM: col = weight_group^T * grad_output_group
                // weight_group: [C_out_per_group, C_in_per_group * K_h * K_w]
                // grad_output_group: [C_out_per_group, H_out * W_out]
                // col: [C_in_per_group * K_h * K_w, H_out * W_out]
                const int64_t M_group = C_in_per_group * K_h * K_w;
                const int64_t N_gemm = H_out * W_out;
                const int64_t K_group = C_out_per_group;

                queue.parallel_for<Conv2dGroupedBackwardInputGemmFallback>(sycl::range<2>(M_group, N_gemm),
                                 [=](sycl::id<2> idx) {
                    const int64_t k = idx[0];  // position in weight column
                    const int64_t hw = idx[1]; // spatial position

                    float sum = 0.0f;
                    for (int64_t oc = 0; oc < K_group; ++oc) {
                        sum += weight_group[oc * M_group + k] * grad_out_group[oc * N_gemm + hw];
                    }
                    col_ptr[k * N_gemm + hw] = sum;
                });

                // Col2im for this group - accumulates into grad_input
                col2im_grouped_kernel(
                    col_ptr, C_in, C_in_per_group, in_channel_offset,
                    H_in, W_in, K_h, K_w,
                    padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                    grad_in_batch, queue
                );
            }
        }
    }

    // Compute grad_weight: convolution of input with grad_output
    if (compute_grad_weight) {
        grad_weight = Tensor(std::vector<int64_t>(weight_shape.begin(), weight_shape.end()), weight.dtype(), weight.device());
        float* grad_weight_ptr = get_data_ptr<float>(grad_weight);

        // Initialize grad_weight to zero
        const int64_t total_weight_size = C_out * C_in_per_group * K_h * K_w;
        queue.fill(grad_weight_ptr, 0.0f, total_weight_size);

        // Col buffer for one group
        const int64_t col_size = C_in_per_group * K_h * K_w * H_out * W_out;
        Tensor col_buffer({col_size}, input.dtype(), input.device());
        float* col_ptr = get_data_ptr<float>(col_buffer);

        // Process each batch and accumulate gradients
        for (int64_t n = 0; n < N; ++n) {
            const float* input_batch = input_ptr + n * C_in * H_in * W_in;

            // Process each group
            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * C_in_per_group * K_h * K_w;

                const float* grad_out_group = grad_output_ptr + n * C_out * H_out * W_out + out_channel_offset * H_out * W_out;
                float* grad_weight_group = grad_weight_ptr + weight_offset;

                // Im2col for this group's input channels
                im2col_grouped_kernel(
                    input_batch, C_in, C_in_per_group, in_channel_offset,
                    H_in, W_in, K_h, K_w,
                    padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                    col_ptr, queue
                );

                // GEMM: grad_weight_group += grad_output_group * col^T
                // grad_output_group: [C_out_per_group, H_out * W_out]
                // col: [C_in_per_group * K_h * K_w, H_out * W_out]
                // grad_weight_group: [C_out_per_group, C_in_per_group * K_h * K_w]
                const int64_t M_group = C_out_per_group;
                const int64_t N_weight = C_in_per_group * K_h * K_w;
                const int64_t K_spatial = H_out * W_out;

                queue.parallel_for<Conv2dGroupedBackwardWeightGemmFallback>(sycl::range<2>(M_group, N_weight),
                                 [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];  // output channel within group
                    const int64_t k = idx[1];   // weight position

                    float sum = 0.0f;
                    for (int64_t hw = 0; hw < K_spatial; ++hw) {
                        sum += grad_out_group[oc * K_spatial + hw] * col_ptr[k * K_spatial + hw];
                    }

                    // Atomic add for accumulation across batches
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                        atomic_val(grad_weight_group[oc * N_weight + k]);
                    atomic_val.fetch_add(sum);
                });
            }
        }
    }

    // Compute grad_bias: sum grad_output over batch, height, width dimensions
    // (same for all groups, bias is per output channel)
    if (compute_grad_bias) {
        grad_bias = Tensor({weight_shape[0]}, weight.dtype(), weight.device());
        const int64_t N_batch = grad_output.shape()[0];
        const int64_t C = grad_output.shape()[1];
        const int64_t H = grad_output.shape()[2];
        const int64_t W = grad_output.shape()[3];

        float* grad_bias_ptr = get_data_ptr<float>(grad_bias);

        queue.parallel_for<Conv2dBackwardBiasReduction>(sycl::range<1>(C), [=](sycl::id<1> c) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N_batch; ++n) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        sum += grad_output_ptr[((n * C + c) * H + h) * W + w];
                    }
                }
            }
            grad_bias_ptr[c] = sum;
        });
    }

    return {grad_input, grad_weight, grad_bias};
}

// Kernel classes for separate backward operations
class Conv2dBackwardInputSeparateGemm;
class Conv2dBackwardInputSeparateCol2im;
class Conv2dBackwardWeightSeparateGemm;
class Conv2dBackwardBiasSeparateReduction;

// Float64 kernel classes for separate backward operations
class Conv2dBackwardInputSeparateGemmFloat64;
class Conv2dBackwardWeightSeparateGemmFloat64;
class Conv2dBackwardBiasSeparateReductionFloat64;

// Float16 kernel classes for separate backward operations
class Conv2dBackwardInputSeparateGemmFloat16;
class Conv2dBackwardWeightSeparateGemmFloat16;
class Conv2dBackwardBiasSeparateReductionFloat16;

// Separate conv2d_backward_input that takes input_shape instead of input tensor.
// Audit J.2: per-axis stride/padding/dilation, matches CPU backend API.
auto conv2d_backward_input(const Tensor& grad_output, const Tensor& weight,
                           const std::vector<int64_t>& input_shape,
                           int64_t stride_h, int64_t stride_w,
                           int64_t padding_h, int64_t padding_w,
                           int64_t dilation_h, int64_t dilation_w,
                           int64_t groups,
                           sycl::queue& queue) -> Tensor {
    auto weight_shape = weight.shape();
    auto grad_output_shape = grad_output.shape();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t H_in = input_shape[2];
    const int64_t W_in = input_shape[3];

    const int64_t C_out = weight_shape[0];
    const int64_t C_in_per_group = weight_shape[1];
    const int64_t K_h = weight_shape[2];
    const int64_t K_w = weight_shape[3];

    const int64_t C_out_per_group = C_out / groups;

    const int64_t H_out = grad_output_shape[2];
    const int64_t W_out = grad_output_shape[3];

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
    const int64_t col_size = C_in_per_group * K_h * K_w * H_out * W_out;

    if (grad_output.dtype() == DType::Float32) {
        float* grad_input_ptr = get_data_ptr<float>(grad_input);
        const float* weight_ptr = get_data_ptr<const float>(weight);
        const float* grad_output_ptr = get_data_ptr<const float>(grad_output);

        queue.fill(grad_input_ptr, 0.0f, N * C_in * H_in * W_in);

        Tensor col_buffer({col_size}, grad_output.dtype(), grad_output.device());
        float* col_ptr = get_data_ptr<float>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            float* grad_in_batch = grad_input_ptr + n * C_in * H_in * W_in;

            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * C_in_per_group * K_h * K_w;

                const float* grad_out_group = grad_output_ptr + n * C_out * H_out * W_out + out_channel_offset * H_out * W_out;
                const float* weight_group = weight_ptr + weight_offset;

                const int64_t M_group = C_in_per_group * K_h * K_w;
                const int64_t N_gemm = H_out * W_out;
                const int64_t K_group = C_out_per_group;

                queue.parallel_for<Conv2dBackwardInputSeparateGemm>(sycl::range<2>(M_group, N_gemm),
                                 [=](sycl::id<2> idx) {
                    const int64_t k = idx[0];
                    const int64_t hw = idx[1];

                    float sum = 0.0f;
                    for (int64_t oc = 0; oc < K_group; ++oc) {
                        sum += weight_group[oc * M_group + k] * grad_out_group[oc * N_gemm + hw];
                    }
                    col_ptr[k * N_gemm + hw] = sum;
                });

                col2im_grouped_kernel(
                    col_ptr, C_in, C_in_per_group, in_channel_offset,
                    H_in, W_in, K_h, K_w,
                    padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                    grad_in_batch, queue
                );
            }
        }
    }
    else if (grad_output.dtype() == DType::Float64) {
        double* grad_input_ptr = get_data_ptr<double>(grad_input);
        const double* weight_ptr = get_data_ptr<const double>(weight);
        const double* grad_output_ptr = get_data_ptr<const double>(grad_output);

        queue.fill(grad_input_ptr, 0.0, N * C_in * H_in * W_in);

        Tensor col_buffer({col_size}, grad_output.dtype(), grad_output.device());
        double* col_ptr = get_data_ptr<double>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            double* grad_in_batch = grad_input_ptr + n * C_in * H_in * W_in;

            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * C_in_per_group * K_h * K_w;

                const double* grad_out_group = grad_output_ptr + n * C_out * H_out * W_out + out_channel_offset * H_out * W_out;
                const double* weight_group = weight_ptr + weight_offset;

                const int64_t M_group = C_in_per_group * K_h * K_w;
                const int64_t N_gemm = H_out * W_out;
                const int64_t K_group = C_out_per_group;

                queue.parallel_for<Conv2dBackwardInputSeparateGemmFloat64>(sycl::range<2>(M_group, N_gemm),
                                 [=](sycl::id<2> idx) {
                    const int64_t k = idx[0];
                    const int64_t hw = idx[1];

                    double sum = 0.0;
                    for (int64_t oc = 0; oc < K_group; ++oc) {
                        sum += weight_group[oc * M_group + k] * grad_out_group[oc * N_gemm + hw];
                    }
                    col_ptr[k * N_gemm + hw] = sum;
                });

                col2im_grouped_kernel(
                    col_ptr, C_in, C_in_per_group, in_channel_offset,
                    H_in, W_in, K_h, K_w,
                    padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                    grad_in_batch, queue
                );
            }
        }
    }
    else if (grad_output.dtype() == DType::Float16) {
        sycl::half* grad_input_ptr = get_data_ptr<sycl::half>(grad_input);
        const sycl::half* weight_ptr = get_data_ptr<const sycl::half>(weight);
        const sycl::half* grad_output_ptr = get_data_ptr<const sycl::half>(grad_output);

        // Initialize to zero
        queue.parallel_for(sycl::range<1>(N * C_in * H_in * W_in), [=](sycl::id<1> i) {
            grad_input_ptr[i] = sycl::half(0.0f);
        });

        Tensor col_buffer({col_size}, grad_output.dtype(), grad_output.device());
        sycl::half* col_ptr = get_data_ptr<sycl::half>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            sycl::half* grad_in_batch = grad_input_ptr + n * C_in * H_in * W_in;

            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * C_in_per_group * K_h * K_w;

                const sycl::half* grad_out_group = grad_output_ptr + n * C_out * H_out * W_out + out_channel_offset * H_out * W_out;
                const sycl::half* weight_group = weight_ptr + weight_offset;

                const int64_t M_group = C_in_per_group * K_h * K_w;
                const int64_t N_gemm = H_out * W_out;
                const int64_t K_group = C_out_per_group;

                // Use float accumulation for precision
                queue.parallel_for<Conv2dBackwardInputSeparateGemmFloat16>(sycl::range<2>(M_group, N_gemm),
                                 [=](sycl::id<2> idx) {
                    const int64_t k = idx[0];
                    const int64_t hw = idx[1];

                    float sum = 0.0f;
                    for (int64_t oc = 0; oc < K_group; ++oc) {
                        sum += static_cast<float>(weight_group[oc * M_group + k]) *
                               static_cast<float>(grad_out_group[oc * N_gemm + hw]);
                    }
                    col_ptr[k * N_gemm + hw] = saturate_to_half(sum);
                });

                col2im_grouped_kernel(
                    col_ptr, C_in, C_in_per_group, in_channel_offset,
                    H_in, W_in, K_h, K_w,
                    padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                    grad_in_batch, queue
                );
            }
        }
    }
    else {
        throw std::runtime_error("Unsupported dtype for conv2d_backward_input");
    }

    return grad_input;
}

// Separate conv2d_backward_weight that takes weight_shape instead of weight tensor.
// Audit J.2: per-axis stride/padding/dilation, matches CPU backend API.
auto conv2d_backward_weight(const Tensor& grad_output, const Tensor& input,
                            const std::vector<int64_t>& weight_shape,
                            int64_t stride_h, int64_t stride_w,
                            int64_t padding_h, int64_t padding_w,
                            int64_t dilation_h, int64_t dilation_w,
                            int64_t groups,
                            sycl::queue& queue) -> Tensor {
    auto input_shape = input.shape();
    auto grad_output_shape = grad_output.shape();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t H_in = input_shape[2];
    const int64_t W_in = input_shape[3];

    const int64_t C_out = weight_shape[0];
    const int64_t C_in_per_group = weight_shape[1];
    const int64_t K_h = weight_shape[2];
    const int64_t K_w = weight_shape[3];

    const int64_t C_out_per_group = C_out / groups;

    const int64_t H_out = grad_output_shape[2];
    const int64_t W_out = grad_output_shape[3];

    Tensor grad_weight(weight_shape, input.dtype(), input.device());
    const int64_t total_weight_size = C_out * C_in_per_group * K_h * K_w;
    const int64_t col_size = C_in_per_group * K_h * K_w * H_out * W_out;

    if (input.dtype() == DType::Float32) {
        float* grad_weight_ptr = get_data_ptr<float>(grad_weight);
        const float* input_ptr = get_data_ptr<const float>(input);
        const float* grad_output_ptr = get_data_ptr<const float>(grad_output);

        queue.fill(grad_weight_ptr, 0.0f, total_weight_size);

        Tensor col_buffer({col_size}, input.dtype(), input.device());
        float* col_ptr = get_data_ptr<float>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            const float* input_batch = input_ptr + n * C_in * H_in * W_in;

            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * C_in_per_group * K_h * K_w;

                const float* grad_out_group = grad_output_ptr + n * C_out * H_out * W_out + out_channel_offset * H_out * W_out;
                float* grad_weight_group = grad_weight_ptr + weight_offset;

                im2col_grouped_kernel(
                    input_batch, C_in, C_in_per_group, in_channel_offset,
                    H_in, W_in, K_h, K_w,
                    padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                    col_ptr, queue
                );

                const int64_t M_group = C_out_per_group;
                const int64_t N_weight = C_in_per_group * K_h * K_w;
                const int64_t K_spatial = H_out * W_out;

                queue.parallel_for<Conv2dBackwardWeightSeparateGemm>(sycl::range<2>(M_group, N_weight),
                                 [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];
                    const int64_t k = idx[1];

                    float sum = 0.0f;
                    for (int64_t hw = 0; hw < K_spatial; ++hw) {
                        sum += grad_out_group[oc * K_spatial + hw] * col_ptr[k * K_spatial + hw];
                    }

                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                        atomic_val(grad_weight_group[oc * N_weight + k]);
                    atomic_val.fetch_add(sum);
                });
            }
        }
    }
    else if (input.dtype() == DType::Float64) {
        double* grad_weight_ptr = get_data_ptr<double>(grad_weight);
        const double* input_ptr = get_data_ptr<const double>(input);
        const double* grad_output_ptr = get_data_ptr<const double>(grad_output);

        queue.fill(grad_weight_ptr, 0.0, total_weight_size);

        Tensor col_buffer({col_size}, input.dtype(), input.device());
        double* col_ptr = get_data_ptr<double>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            const double* input_batch = input_ptr + n * C_in * H_in * W_in;

            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * C_in_per_group * K_h * K_w;

                const double* grad_out_group = grad_output_ptr + n * C_out * H_out * W_out + out_channel_offset * H_out * W_out;
                double* grad_weight_group = grad_weight_ptr + weight_offset;

                im2col_grouped_kernel(
                    input_batch, C_in, C_in_per_group, in_channel_offset,
                    H_in, W_in, K_h, K_w,
                    padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                    col_ptr, queue
                );

                const int64_t M_group = C_out_per_group;
                const int64_t N_weight = C_in_per_group * K_h * K_w;
                const int64_t K_spatial = H_out * W_out;

                queue.parallel_for<Conv2dBackwardWeightSeparateGemmFloat64>(sycl::range<2>(M_group, N_weight),
                                 [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];
                    const int64_t k = idx[1];

                    double sum = 0.0;
                    for (int64_t hw = 0; hw < K_spatial; ++hw) {
                        sum += grad_out_group[oc * K_spatial + hw] * col_ptr[k * K_spatial + hw];
                    }

                    sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>
                        atomic_val(grad_weight_group[oc * N_weight + k]);
                    atomic_val.fetch_add(sum);
                });
            }
        }
    }
    else if (input.dtype() == DType::Float16) {
        sycl::half* grad_weight_ptr = get_data_ptr<sycl::half>(grad_weight);
        const sycl::half* input_ptr = get_data_ptr<const sycl::half>(input);
        const sycl::half* grad_output_ptr = get_data_ptr<const sycl::half>(grad_output);

        // Initialize to zero
        queue.parallel_for(sycl::range<1>(total_weight_size), [=](sycl::id<1> i) {
            grad_weight_ptr[i] = sycl::half(0.0f);
        });

        Tensor col_buffer({col_size}, input.dtype(), input.device());
        sycl::half* col_ptr = get_data_ptr<sycl::half>(col_buffer);

        // Use float accumulation buffer for better precision
        Tensor grad_weight_float({total_weight_size}, DType::Float32, input.device());
        float* grad_weight_float_ptr = get_data_ptr<float>(grad_weight_float);
        queue.fill(grad_weight_float_ptr, 0.0f, total_weight_size);

        for (int64_t n = 0; n < N; ++n) {
            const sycl::half* input_batch = input_ptr + n * C_in * H_in * W_in;

            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * C_in_per_group * K_h * K_w;

                const sycl::half* grad_out_group = grad_output_ptr + n * C_out * H_out * W_out + out_channel_offset * H_out * W_out;
                float* grad_weight_group = grad_weight_float_ptr + weight_offset;

                im2col_grouped_kernel(
                    input_batch, C_in, C_in_per_group, in_channel_offset,
                    H_in, W_in, K_h, K_w,
                    padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                    col_ptr, queue
                );

                const int64_t M_group = C_out_per_group;
                const int64_t N_weight = C_in_per_group * K_h * K_w;
                const int64_t K_spatial = H_out * W_out;

                queue.parallel_for<Conv2dBackwardWeightSeparateGemmFloat16>(sycl::range<2>(M_group, N_weight),
                                 [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];
                    const int64_t k = idx[1];

                    float sum = 0.0f;
                    for (int64_t hw = 0; hw < K_spatial; ++hw) {
                        sum += static_cast<float>(grad_out_group[oc * K_spatial + hw]) *
                               static_cast<float>(col_ptr[k * K_spatial + hw]);
                    }

                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                        atomic_val(grad_weight_group[oc * N_weight + k]);
                    atomic_val.fetch_add(sum);
                });
            }
        }

        // Convert accumulated float values back to half
        queue.parallel_for(sycl::range<1>(total_weight_size), [=](sycl::id<1> i) {
            grad_weight_ptr[i] = sycl::half(grad_weight_float_ptr[i]);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for conv2d_backward_weight");
    }

    return grad_weight;
}

// Separate conv2d_backward_bias
auto conv2d_backward_bias(const Tensor& grad_output, sycl::queue& queue) -> Tensor {
    auto grad_output_shape = grad_output.shape();
    const int64_t N = grad_output_shape[0];
    const int64_t C = grad_output_shape[1];
    const int64_t H = grad_output_shape[2];
    const int64_t W = grad_output_shape[3];

    Tensor grad_bias({C}, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        float* grad_bias_ptr = get_data_ptr<float>(grad_bias);
        const float* grad_output_ptr = get_data_ptr<const float>(grad_output);

        queue.parallel_for<Conv2dBackwardBiasSeparateReduction>(sycl::range<1>(C), [=](sycl::id<1> c) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        sum += grad_output_ptr[((n * C + c) * H + h) * W + w];
                    }
                }
            }
            grad_bias_ptr[c] = sum;
        });
    }
    else if (grad_output.dtype() == DType::Float64) {
        double* grad_bias_ptr = get_data_ptr<double>(grad_bias);
        const double* grad_output_ptr = get_data_ptr<const double>(grad_output);

        queue.parallel_for<Conv2dBackwardBiasSeparateReductionFloat64>(sycl::range<1>(C), [=](sycl::id<1> c) {
            double sum = 0.0;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        sum += grad_output_ptr[((n * C + c) * H + h) * W + w];
                    }
                }
            }
            grad_bias_ptr[c] = sum;
        });
    }
    else if (grad_output.dtype() == DType::Float16) {
        sycl::half* grad_bias_ptr = get_data_ptr<sycl::half>(grad_bias);
        const sycl::half* grad_output_ptr = get_data_ptr<const sycl::half>(grad_output);

        // Use float accumulation for precision
        queue.parallel_for<Conv2dBackwardBiasSeparateReductionFloat16>(sycl::range<1>(C), [=](sycl::id<1> c) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        sum += static_cast<float>(grad_output_ptr[((n * C + c) * H + h) * W + w]);
                    }
                }
            }
            grad_bias_ptr[c] = saturate_to_half(sum);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for conv2d_backward_bias");
    }

    return grad_bias;
}

// ============================================================================
// Transposed Convolution (ConvTranspose2d) Forward
// ============================================================================

// SYCL Kernel name classes for conv_transpose2d
struct ConvTranspose2dForwardKernelFloat32 {};
struct ConvTranspose2dForwardKernelFloat64 {};
struct ConvTranspose2dForwardKernelFloat16 {};
struct ConvTranspose2dBiasKernelFloat32 {};
struct ConvTranspose2dBiasKernelFloat64 {};
struct ConvTranspose2dBiasKernelFloat16 {};

/**
 * @brief Transposed 2D convolution (deconvolution) forward pass
 *
 * Uses gather approach: each output position independently gathers contributions
 * from input positions through the transposed kernel relationship.
 *
 * Output size: (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
 *
 * @param input Input tensor (N, C_in, H_in, W_in)
 * @param weight Weight tensor (C_in, C_out/groups, kH, kW)
 * @param bias Optional bias tensor (C_out) or nullptr
 * @param stride Stride of transposed convolution
 * @param padding Padding applied to input
 * @param output_padding Additional size added to output
 * @param dilation Spacing between kernel elements
 * @param groups Number of groups for grouped convolution
 * @param queue SYCL queue for execution
 * @return Output tensor (N, C_out, H_out, W_out)
 */
auto conv_transpose2d_forward(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    int64_t stride, int64_t padding, int64_t output_padding,
    int64_t dilation, int64_t groups,
    sycl::queue& queue
) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int64_t in_channels_per_group = weight_shape[0] / groups;
    int64_t out_channels_per_group = weight_shape[1];
    int64_t out_channels = out_channels_per_group * groups;
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions
    int64_t out_h = (in_h - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    int64_t out_w = (in_w - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument(
            "Invalid ConvTranspose2d configuration: output dimensions are non-positive (out_h=" +
            std::to_string(out_h) + ", out_w=" + std::to_string(out_w) + ")");
    }

    Tensor output({batch, out_channels, out_h, out_w}, input.dtype(), input.device());

    int64_t total_output = batch * out_channels * out_h * out_w;

    if (input.dtype() == DType::Float32) {
        const float* input_data = get_data_ptr<const float>(input);
        const float* weight_data = get_data_ptr<const float>(weight);
        float* output_data = get_data_ptr<float>(output);

        queue.parallel_for<ConvTranspose2dForwardKernelFloat32>(
            sycl::range<1>(total_output),
            [=](sycl::id<1> idx) {
                int64_t w = idx % out_w;
                int64_t h = (idx / out_w) % out_h;
                int64_t c = (idx / (out_w * out_h)) % out_channels;
                int64_t b = idx / (out_w * out_h * out_channels);

                int64_t g = c / out_channels_per_group;
                int64_t oc = c % out_channels_per_group;
                int64_t in_start = g * in_channels_per_group;

                float sum = 0.0f;

                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t h_shifted = h + padding - kh * dilation;
                            int64_t w_shifted = w + padding - kw * dilation;

                            if (h_shifted >= 0 && h_shifted % stride == 0 &&
                                w_shifted >= 0 && w_shifted % stride == 0) {
                                int64_t ih = h_shifted / stride;
                                int64_t iw = w_shifted / stride;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int64_t input_idx = b * (in_channels * in_h * in_w) +
                                                       (in_start + ic) * (in_h * in_w) +
                                                       ih * in_w + iw;
                                    int64_t weight_idx = (in_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                                        oc * (kernel_h * kernel_w) +
                                                        kh * kernel_w + kw;
                                    sum += input_data[input_idx] * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                }

                output_data[idx] = sum;
            }
        );

        // Add bias if present
        if (bias != nullptr) {
            const float* bias_data = get_data_ptr<const float>(*bias);
            queue.parallel_for<ConvTranspose2dBiasKernelFloat32>(
                sycl::range<1>(total_output),
                [=](sycl::id<1> idx) {
                    int64_t c = (idx / (out_w * out_h)) % out_channels;
                    output_data[idx] += bias_data[c];
                }
            );
        }
    }
    else if (input.dtype() == DType::Float64) {
        const double* input_data = get_data_ptr<const double>(input);
        const double* weight_data = get_data_ptr<const double>(weight);
        double* output_data = get_data_ptr<double>(output);

        queue.parallel_for<ConvTranspose2dForwardKernelFloat64>(
            sycl::range<1>(total_output),
            [=](sycl::id<1> idx) {
                int64_t w = idx % out_w;
                int64_t h = (idx / out_w) % out_h;
                int64_t c = (idx / (out_w * out_h)) % out_channels;
                int64_t b = idx / (out_w * out_h * out_channels);

                int64_t g = c / out_channels_per_group;
                int64_t oc = c % out_channels_per_group;
                int64_t in_start = g * in_channels_per_group;

                double sum = 0.0;

                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t h_shifted = h + padding - kh * dilation;
                            int64_t w_shifted = w + padding - kw * dilation;

                            if (h_shifted >= 0 && h_shifted % stride == 0 &&
                                w_shifted >= 0 && w_shifted % stride == 0) {
                                int64_t ih = h_shifted / stride;
                                int64_t iw = w_shifted / stride;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int64_t input_idx = b * (in_channels * in_h * in_w) +
                                                       (in_start + ic) * (in_h * in_w) +
                                                       ih * in_w + iw;
                                    int64_t weight_idx = (in_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                                        oc * (kernel_h * kernel_w) +
                                                        kh * kernel_w + kw;
                                    sum += input_data[input_idx] * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                }

                output_data[idx] = sum;
            }
        );

        if (bias != nullptr) {
            const double* bias_data = get_data_ptr<const double>(*bias);
            queue.parallel_for<ConvTranspose2dBiasKernelFloat64>(
                sycl::range<1>(total_output),
                [=](sycl::id<1> idx) {
                    int64_t c = (idx / (out_w * out_h)) % out_channels;
                    output_data[idx] += bias_data[c];
                }
            );
        }
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* input_data = get_data_ptr<const sycl::half>(input);
        const sycl::half* weight_data = get_data_ptr<const sycl::half>(weight);
        sycl::half* output_data = get_data_ptr<sycl::half>(output);

        queue.parallel_for<ConvTranspose2dForwardKernelFloat16>(
            sycl::range<1>(total_output),
            [=](sycl::id<1> idx) {
                int64_t w = idx % out_w;
                int64_t h = (idx / out_w) % out_h;
                int64_t c = (idx / (out_w * out_h)) % out_channels;
                int64_t b = idx / (out_w * out_h * out_channels);

                int64_t g = c / out_channels_per_group;
                int64_t oc = c % out_channels_per_group;
                int64_t in_start = g * in_channels_per_group;

                float sum = 0.0f;

                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t h_shifted = h + padding - kh * dilation;
                            int64_t w_shifted = w + padding - kw * dilation;

                            if (h_shifted >= 0 && h_shifted % stride == 0 &&
                                w_shifted >= 0 && w_shifted % stride == 0) {
                                int64_t ih = h_shifted / stride;
                                int64_t iw = w_shifted / stride;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int64_t input_idx = b * (in_channels * in_h * in_w) +
                                                       (in_start + ic) * (in_h * in_w) +
                                                       ih * in_w + iw;
                                    int64_t weight_idx = (in_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                                        oc * (kernel_h * kernel_w) +
                                                        kh * kernel_w + kw;
                                    sum += float(input_data[input_idx]) * float(weight_data[weight_idx]);
                                }
                            }
                        }
                    }
                }

                output_data[idx] = saturate_to_half(sum);
            }
        );

        if (bias != nullptr) {
            const sycl::half* bias_data = get_data_ptr<const sycl::half>(*bias);
            queue.parallel_for<ConvTranspose2dBiasKernelFloat16>(
                sycl::range<1>(total_output),
                [=](sycl::id<1> idx) {
                    int64_t c = (idx / (out_w * out_h)) % out_channels;
                    output_data[idx] = saturate_to_half(float(output_data[idx]) + float(bias_data[c]));
                }
            );
        }
    }
    else {
        throw std::runtime_error("Unsupported dtype for conv_transpose2d_forward");
    }

    return output;
}

#endif // TENZOR_HAS_ONEDNN

// ============================================================================
// DepthwiseConv2d kernel
// ============================================================================
class DepthwiseConv2dKernelF32;
class DepthwiseConv2dKernelF64;

auto depthwise_conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                              int64_t stride, int64_t padding, int64_t dilation,
                              sycl::queue& queue) -> Tensor {
    auto in_shape = input.shape();
    auto w_shape = weight.shape();

    int64_t N = in_shape[0];
    int64_t C = in_shape[1];
    int64_t H_in = in_shape[2];
    int64_t W_in = in_shape[3];
    int64_t kH = w_shape[2];
    int64_t kW = w_shape[3];

    int64_t H_out = (H_in + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
    int64_t W_out = (W_in + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());
    int64_t total = N * C * H_out * W_out;

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = static_cast<const float*>(input.data_ptr());
        const float* w_ptr = static_cast<const float*>(weight.data_ptr());
        float* out_ptr = static_cast<float*>(const_cast<void*>(output.data_ptr()));
        bool has_bias = (bias != nullptr && bias->numel() > 0);
        const float* b_ptr = has_bias ? static_cast<const float*>(bias->data_ptr()) : nullptr;

        queue.parallel_for<DepthwiseConv2dKernelF32>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            int64_t w_out = idx % W_out;
            int64_t h_out = (idx / W_out) % H_out;
            int64_t c = (idx / (W_out * H_out)) % C;
            int64_t n = idx / (W_out * H_out * C);

            float sum = 0.0f;
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh * dilation;
                    int64_t w_in = w_out * stride - padding + kw * dilation;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        float in_val = in_ptr[n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in];
                        float w_val = w_ptr[c * kH * kW + kh * kW + kw];
                        sum += in_val * w_val;
                    }
                }
            }
            if (has_bias) sum += b_ptr[c];
            out_ptr[idx] = sum;
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = static_cast<const double*>(input.data_ptr());
        const double* w_ptr = static_cast<const double*>(weight.data_ptr());
        double* out_ptr = static_cast<double*>(const_cast<void*>(output.data_ptr()));
        bool has_bias = (bias != nullptr && bias->numel() > 0);
        const double* b_ptr = has_bias ? static_cast<const double*>(bias->data_ptr()) : nullptr;

        queue.parallel_for<DepthwiseConv2dKernelF64>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            int64_t w_out = idx % W_out;
            int64_t h_out = (idx / W_out) % H_out;
            int64_t c = (idx / (W_out * H_out)) % C;
            int64_t n = idx / (W_out * H_out * C);

            double sum = 0.0;
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh * dilation;
                    int64_t w_in = w_out * stride - padding + kw * dilation;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        double in_val = in_ptr[n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in];
                        double w_val = w_ptr[c * kH * kW + kh * kW + kw];
                        sum += in_val * w_val;
                    }
                }
            }
            if (has_bias) sum += b_ptr[c];
            out_ptr[idx] = sum;
        });
    } else {
        throw std::runtime_error("depthwise_conv2d: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Deformable Conv2d (DCNv2) — SYCL device kernels
// ============================================================================

// SYCL kernel name classes
class DeformableConv2dForwardF32;
class DeformableConv2dForwardF64;
class DeformableConv2dBackwardInputF32;
class DeformableConv2dBackwardInputF64;
class DeformableConv2dBackwardWeightF32;
class DeformableConv2dBackwardWeightF64;

// Device-side bilinear interpolation at fractional (h, w) in a single channel plane.
// Returns 0 for out-of-bounds.
template <typename T>
inline T dcn_bilinear_sample(const T* data, int64_t H, int64_t W,
                             float h, float w) {
    if (h <= -1.0f || h >= static_cast<float>(H) ||
        w <= -1.0f || w >= static_cast<float>(W)) {
        return T(0);
    }

    int64_t h_low = static_cast<int64_t>(sycl::floor(h));
    int64_t w_low = static_cast<int64_t>(sycl::floor(w));
    int64_t h_high = h_low + 1;
    int64_t w_high = w_low + 1;

    float lh = h - static_cast<float>(h_low);
    float lw = w - static_cast<float>(w_low);
    float hh = 1.0f - lh;
    float hw = 1.0f - lw;

    float v1 = (h_low >= 0 && w_low >= 0)   ? static_cast<float>(data[h_low * W + w_low])   : 0.0f;
    float v2 = (h_low >= 0 && w_high < W)    ? static_cast<float>(data[h_low * W + w_high])  : 0.0f;
    float v3 = (h_high < H && w_low >= 0)    ? static_cast<float>(data[h_high * W + w_low])  : 0.0f;
    float v4 = (h_high < H && w_high < W)    ? static_cast<float>(data[h_high * W + w_high]) : 0.0f;

    return static_cast<T>(hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4);
}

// Device-side: scatter gradient back through bilinear interpolation via sycl::atomic_ref.
template <typename T>
inline void dcn_bilinear_scatter(T* grad_data, int64_t H, int64_t W,
                                  float h, float w, float top_grad) {
    if (h <= -1.0f || h >= static_cast<float>(H) ||
        w <= -1.0f || w >= static_cast<float>(W)) {
        return;
    }

    int64_t h_low = static_cast<int64_t>(sycl::floor(h));
    int64_t w_low = static_cast<int64_t>(sycl::floor(w));
    int64_t h_high = h_low + 1;
    int64_t w_high = w_low + 1;

    float lh = h - static_cast<float>(h_low);
    float lw = w - static_cast<float>(w_low);
    float hh = 1.0f - lh;
    float hw = 1.0f - lw;

    if (h_low >= 0 && w_low >= 0) {
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(grad_data[h_low * W + w_low]);
        ref.fetch_add(static_cast<T>(hh * hw * top_grad));
    }
    if (h_low >= 0 && w_high < W) {
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(grad_data[h_low * W + w_high]);
        ref.fetch_add(static_cast<T>(hh * lw * top_grad));
    }
    if (h_high < H && w_low >= 0) {
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(grad_data[h_high * W + w_low]);
        ref.fetch_add(static_cast<T>(lh * hw * top_grad));
    }
    if (h_high < H && w_high < W) {
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(grad_data[h_high * W + w_high]);
        ref.fetch_add(static_cast<T>(lh * lw * top_grad));
    }
}

// Device-side: compute dval/dh and dval/dw for bilinear interpolation (offset gradients).
template <typename T>
inline void dcn_bilinear_offset_grad(const T* data, int64_t H, int64_t W,
                                      float h, float w,
                                      float& grad_h, float& grad_w) {
    grad_h = 0.0f;
    grad_w = 0.0f;
    if (h <= -1.0f || h >= static_cast<float>(H) ||
        w <= -1.0f || w >= static_cast<float>(W)) {
        return;
    }

    int64_t h_low = static_cast<int64_t>(sycl::floor(h));
    int64_t w_low = static_cast<int64_t>(sycl::floor(w));
    int64_t h_high = h_low + 1;
    int64_t w_high = w_low + 1;

    float lh = h - static_cast<float>(h_low);
    float lw = w - static_cast<float>(w_low);
    float hh = 1.0f - lh;
    float hw = 1.0f - lw;

    float v1 = (h_low >= 0 && w_low >= 0)   ? static_cast<float>(data[h_low * W + w_low])   : 0.0f;
    float v2 = (h_low >= 0 && w_high < W)    ? static_cast<float>(data[h_low * W + w_high])  : 0.0f;
    float v3 = (h_high < H && w_low >= 0)    ? static_cast<float>(data[h_high * W + w_low])  : 0.0f;
    float v4 = (h_high < H && w_high < W)    ? static_cast<float>(data[h_high * W + w_high]) : 0.0f;

    // d/dh: derivative of bilinear w.r.t. h
    grad_h = -hw * v1 - lw * v2 + hw * v3 + lw * v4;
    // d/dw: derivative of bilinear w.r.t. w
    grad_w = -hh * v1 + hh * v2 - lh * v3 + lh * v4;
}

auto deformable_conv2d_forward_kernel(
    const Tensor& input, const Tensor& offset, const Tensor& weight,
    const Tensor& bias, const Tensor& mask,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    sycl::queue& queue) -> Tensor {

    // Float16 / BFloat16: widen to Float32, compute, narrow back.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig_dtype = input.dtype();
        auto in32 = input.to(DType::Float32);
        auto off32 = offset.to(DType::Float32);
        auto w32 = weight.to(DType::Float32);
        auto b32 = (bias.numel() > 0) ? bias.to(DType::Float32) : bias;
        auto m32 = (mask.numel() > 0) ? mask.to(DType::Float32) : mask;
        Tensor result = deformable_conv2d_forward_kernel(
            in32, off32, w32, b32, m32,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, queue);
        return result.to(orig_dtype);
    }

    auto ishape = input.shape();
    auto wshape = weight.shape();
    const int64_t N = ishape[0], C_in = ishape[1], H = ishape[2], W = ishape[3];
    const int64_t C_out = wshape[0], kH = wshape[2], kW = wshape[3];
    const int64_t H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int64_t W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;

    const bool use_mask = mask.numel() > 0;
    const bool use_bias = bias.numel() > 0;
    const int64_t channels_per_group = C_in / groups;
    const int64_t out_channels_per_group = C_out / groups;
    const int64_t channels_per_offset_group = C_in / offset_groups;
    const int64_t total = N * C_out * H_out * W_out;

    Tensor output({N, C_out, H_out, W_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* off_ptr = get_data_ptr<const float>(offset);
        const float* w_ptr = get_data_ptr<const float>(weight);
        const float* b_ptr = use_bias ? get_data_ptr<const float>(bias) : nullptr;
        const float* m_ptr = use_mask ? get_data_ptr<const float>(mask) : nullptr;
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<DeformableConv2dForwardF32>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t ow = idx % W_out;
            const int64_t oh = (idx / W_out) % H_out;
            const int64_t oc = (idx / (W_out * H_out)) % C_out;
            const int64_t n  = idx / (W_out * H_out * C_out);

            const int64_t g = oc / out_channels_per_group;
            float sum = 0.0f;

            for (int64_t ic_local = 0; ic_local < channels_per_group; ++ic_local) {
                const int64_t ic = g * channels_per_group + ic_local;
                const int64_t og = ic / channels_per_offset_group;

                const float* input_plane = in_ptr + (n * C_in + ic) * H * W;
                const float* weight_plane = w_ptr + (oc * channels_per_group + ic_local) * kH * kW;

                for (int64_t kh = 0; kh < kH; ++kh) {
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        const int64_t k_linear = kh * kW + kw;
                        const int64_t offset_base = og * 2 * kH * kW;

                        const float* off_h_plane = off_ptr +
                            (n * offset_groups * 2 * kH * kW + offset_base + 2 * k_linear) * H_out * W_out;
                        const float* off_w_plane = off_ptr +
                            (n * offset_groups * 2 * kH * kW + offset_base + 2 * k_linear + 1) * H_out * W_out;

                        float h_base = oh * stride_h - pad_h + kh * dil_h;
                        float w_base = ow * stride_w - pad_w + kw * dil_w;
                        float h_loc = h_base + off_h_plane[oh * W_out + ow];
                        float w_loc = w_base + off_w_plane[oh * W_out + ow];

                        float val = dcn_bilinear_sample(input_plane, H, W, h_loc, w_loc);

                        if (use_mask) {
                            const int64_t mask_base = og * kH * kW;
                            const float* mask_plane = m_ptr +
                                (n * offset_groups * kH * kW + mask_base + k_linear) * H_out * W_out;
                            val *= mask_plane[oh * W_out + ow];
                        }

                        sum += val * weight_plane[k_linear];
                    }
                }
            }

            if (use_bias) {
                sum += b_ptr[oc];
            }
            out_ptr[idx] = sum;
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* off_ptr = get_data_ptr<const double>(offset);
        const double* w_ptr = get_data_ptr<const double>(weight);
        const double* b_ptr = use_bias ? get_data_ptr<const double>(bias) : nullptr;
        const double* m_ptr = use_mask ? get_data_ptr<const double>(mask) : nullptr;
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<DeformableConv2dForwardF64>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t ow = idx % W_out;
            const int64_t oh = (idx / W_out) % H_out;
            const int64_t oc = (idx / (W_out * H_out)) % C_out;
            const int64_t n  = idx / (W_out * H_out * C_out);

            const int64_t g = oc / out_channels_per_group;
            double sum = 0.0;

            for (int64_t ic_local = 0; ic_local < channels_per_group; ++ic_local) {
                const int64_t ic = g * channels_per_group + ic_local;
                const int64_t og = ic / channels_per_offset_group;

                const double* input_plane = in_ptr + (n * C_in + ic) * H * W;
                const double* weight_plane = w_ptr + (oc * channels_per_group + ic_local) * kH * kW;

                for (int64_t kh = 0; kh < kH; ++kh) {
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        const int64_t k_linear = kh * kW + kw;
                        const int64_t offset_base = og * 2 * kH * kW;

                        const double* off_h_plane = off_ptr +
                            (n * offset_groups * 2 * kH * kW + offset_base + 2 * k_linear) * H_out * W_out;
                        const double* off_w_plane = off_ptr +
                            (n * offset_groups * 2 * kH * kW + offset_base + 2 * k_linear + 1) * H_out * W_out;

                        float h_base = static_cast<float>(oh * stride_h - pad_h + kh * dil_h);
                        float w_base = static_cast<float>(ow * stride_w - pad_w + kw * dil_w);
                        float h_loc = h_base + static_cast<float>(off_h_plane[oh * W_out + ow]);
                        float w_loc = w_base + static_cast<float>(off_w_plane[oh * W_out + ow]);

                        double val = dcn_bilinear_sample(input_plane, H, W, h_loc, w_loc);

                        if (use_mask) {
                            const int64_t mask_base = og * kH * kW;
                            const double* mask_plane = m_ptr +
                                (n * offset_groups * kH * kW + mask_base + k_linear) * H_out * W_out;
                            val *= mask_plane[oh * W_out + ow];
                        }

                        sum += val * weight_plane[k_linear];
                    }
                }
            }

            if (use_bias) {
                sum += b_ptr[oc];
            }
            out_ptr[idx] = sum;
        });
    } else {
        throw std::runtime_error("deformable_conv2d_forward: unsupported dtype (requires Float32 or Float64)");
    }

    return output;
}

auto deformable_conv2d_backward_input_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& offset,
    const Tensor& weight, const Tensor& mask,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    sycl::queue& queue) -> std::vector<Tensor> {

    // Float16 / BFloat16: widen to Float32, compute, narrow back.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig_dtype = input.dtype();
        auto g32 = grad_output.to(DType::Float32);
        auto i32 = input.to(DType::Float32);
        auto o32 = offset.to(DType::Float32);
        auto w32 = weight.to(DType::Float32);
        auto m32 = (mask.numel() > 0) ? mask.to(DType::Float32) : mask;
        auto results = deformable_conv2d_backward_input_kernel(
            g32, i32, o32, w32, m32,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, queue);
        std::vector<Tensor> narrowed;
        narrowed.reserve(results.size());
        for (auto& r : results) {
            narrowed.push_back(r.numel() > 0 ? r.to(orig_dtype) : r);
        }
        return narrowed;
    }

    auto ishape = input.shape();
    auto wshape = weight.shape();
    auto oshape = offset.shape();
    const int64_t N = ishape[0], C_in = ishape[1], H = ishape[2], W = ishape[3];
    const int64_t C_out = wshape[0], kH = wshape[2], kW = wshape[3];
    const int64_t H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];

    const bool use_mask = mask.numel() > 0;
    const int64_t channels_per_group = C_in / groups;
    const int64_t out_channels_per_group = C_out / groups;
    const int64_t channels_per_offset_group = C_in / offset_groups;

    // Total work items: one per (n, oc, oh, ow) — mirrors the forward kernel decomposition
    const int64_t total = N * C_out * H_out * W_out;

    Tensor grad_input(std::vector<int64_t>(ishape.begin(), ishape.end()), input.dtype(), input.device());
    Tensor grad_offset(std::vector<int64_t>(oshape.begin(), oshape.end()), input.dtype(), input.device());
    Tensor grad_mask;
    if (use_mask) {
        auto mshape = mask.shape();
        grad_mask = Tensor(std::vector<int64_t>(mshape.begin(), mshape.end()), input.dtype(), input.device());
    }

    if (input.dtype() == DType::Float32) {
        float* gi_ptr = get_data_ptr<float>(grad_input);
        float* go_off_ptr = get_data_ptr<float>(grad_offset);
        float* gm_ptr = use_mask ? get_data_ptr<float>(grad_mask) : nullptr;

        queue.fill(gi_ptr, 0.0f, static_cast<size_t>(N * C_in * H * W));
        queue.fill(go_off_ptr, 0.0f, static_cast<size_t>(offset.numel()));
        if (use_mask) queue.fill(gm_ptr, 0.0f, static_cast<size_t>(mask.numel()));
        queue.wait_and_throw();

        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* off_ptr = get_data_ptr<const float>(offset);
        const float* w_ptr = get_data_ptr<const float>(weight);
        const float* m_ptr = use_mask ? get_data_ptr<const float>(mask) : nullptr;

        queue.parallel_for<DeformableConv2dBackwardInputF32>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t ow = idx % W_out;
            const int64_t oh = (idx / W_out) % H_out;
            const int64_t oc = (idx / (W_out * H_out)) % C_out;
            const int64_t n  = idx / (W_out * H_out * C_out);

            const int64_t g = oc / out_channels_per_group;
            const float grad_out_val = grad_out_ptr[(n * C_out + oc) * H_out * W_out + oh * W_out + ow];

            for (int64_t ic_local = 0; ic_local < channels_per_group; ++ic_local) {
                const int64_t ic = g * channels_per_group + ic_local;
                const int64_t og = ic / channels_per_offset_group;

                const float* input_plane = in_ptr + (n * C_in + ic) * H * W;
                float* grad_input_plane = gi_ptr + (n * C_in + ic) * H * W;
                const float* weight_plane = w_ptr + (oc * channels_per_group + ic_local) * kH * kW;

                for (int64_t kh = 0; kh < kH; ++kh) {
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        const int64_t k_linear = kh * kW + kw;
                        const int64_t offset_base = og * 2 * kH * kW;
                        const int64_t mask_base = og * kH * kW;

                        const int64_t off_h_idx = offset_base + 2 * k_linear;
                        const int64_t off_w_idx = offset_base + 2 * k_linear + 1;

                        const float* off_h_plane = off_ptr +
                            (n * offset_groups * 2 * kH * kW + off_h_idx) * H_out * W_out;
                        const float* off_w_plane = off_ptr +
                            (n * offset_groups * 2 * kH * kW + off_w_idx) * H_out * W_out;

                        float h_base = oh * stride_h - pad_h + kh * dil_h;
                        float w_base = ow * stride_w - pad_w + kw * dil_w;
                        float h_loc = h_base + off_h_plane[oh * W_out + ow];
                        float w_loc = w_base + off_w_plane[oh * W_out + ow];

                        float w_val = weight_plane[k_linear];
                        float m_val = 1.0f;
                        if (use_mask) {
                            const float* mask_plane = m_ptr +
                                (n * offset_groups * kH * kW + mask_base + k_linear) * H_out * W_out;
                            m_val = mask_plane[oh * W_out + ow];
                        }

                        float top_grad = grad_out_val * w_val * m_val;

                        // grad_input: scatter through bilinear interpolation
                        dcn_bilinear_scatter(grad_input_plane, H, W, h_loc, w_loc, top_grad);

                        // grad_offset: d(bilinear)/d(h,w)
                        float dh, dw;
                        dcn_bilinear_offset_grad(input_plane, H, W, h_loc, w_loc, dh, dw);

                        float* grad_off_h = go_off_ptr +
                            (n * offset_groups * 2 * kH * kW + off_h_idx) * H_out * W_out;
                        float* grad_off_w = go_off_ptr +
                            (n * offset_groups * 2 * kH * kW + off_w_idx) * H_out * W_out;

                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            ref_h(grad_off_h[oh * W_out + ow]);
                        ref_h.fetch_add(grad_out_val * w_val * m_val * dh);

                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            ref_w(grad_off_w[oh * W_out + ow]);
                        ref_w.fetch_add(grad_out_val * w_val * m_val * dw);

                        // grad_mask
                        if (use_mask && gm_ptr) {
                            float interp_val = dcn_bilinear_sample(input_plane, H, W, h_loc, w_loc);
                            float* grad_mask_plane = gm_ptr +
                                (n * offset_groups * kH * kW + mask_base + k_linear) * H_out * W_out;
                            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                ref_m(grad_mask_plane[oh * W_out + ow]);
                            ref_m.fetch_add(grad_out_val * w_val * interp_val);
                        }
                    }
                }
            }
        });
    } else if (input.dtype() == DType::Float64) {
        double* gi_ptr = get_data_ptr<double>(grad_input);
        double* go_off_ptr = get_data_ptr<double>(grad_offset);
        double* gm_ptr = use_mask ? get_data_ptr<double>(grad_mask) : nullptr;

        queue.fill(gi_ptr, 0.0, static_cast<size_t>(N * C_in * H * W));
        queue.fill(go_off_ptr, 0.0, static_cast<size_t>(offset.numel()));
        if (use_mask) queue.fill(gm_ptr, 0.0, static_cast<size_t>(mask.numel()));
        queue.wait_and_throw();

        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* off_ptr = get_data_ptr<const double>(offset);
        const double* w_ptr = get_data_ptr<const double>(weight);
        const double* m_ptr = use_mask ? get_data_ptr<const double>(mask) : nullptr;

        queue.parallel_for<DeformableConv2dBackwardInputF64>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t ow = idx % W_out;
            const int64_t oh = (idx / W_out) % H_out;
            const int64_t oc = (idx / (W_out * H_out)) % C_out;
            const int64_t n  = idx / (W_out * H_out * C_out);

            const int64_t g = oc / out_channels_per_group;
            const double grad_out_val = grad_out_ptr[(n * C_out + oc) * H_out * W_out + oh * W_out + ow];

            for (int64_t ic_local = 0; ic_local < channels_per_group; ++ic_local) {
                const int64_t ic = g * channels_per_group + ic_local;
                const int64_t og = ic / channels_per_offset_group;

                const double* input_plane = in_ptr + (n * C_in + ic) * H * W;
                double* grad_input_plane = gi_ptr + (n * C_in + ic) * H * W;
                const double* weight_plane = w_ptr + (oc * channels_per_group + ic_local) * kH * kW;

                for (int64_t kh = 0; kh < kH; ++kh) {
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        const int64_t k_linear = kh * kW + kw;
                        const int64_t offset_base = og * 2 * kH * kW;
                        const int64_t mask_base = og * kH * kW;

                        const int64_t off_h_idx = offset_base + 2 * k_linear;
                        const int64_t off_w_idx = offset_base + 2 * k_linear + 1;

                        const double* off_h_plane = off_ptr +
                            (n * offset_groups * 2 * kH * kW + off_h_idx) * H_out * W_out;
                        const double* off_w_plane = off_ptr +
                            (n * offset_groups * 2 * kH * kW + off_w_idx) * H_out * W_out;

                        float h_base = static_cast<float>(oh * stride_h - pad_h + kh * dil_h);
                        float w_base = static_cast<float>(ow * stride_w - pad_w + kw * dil_w);
                        float h_loc = h_base + static_cast<float>(off_h_plane[oh * W_out + ow]);
                        float w_loc = w_base + static_cast<float>(off_w_plane[oh * W_out + ow]);

                        double w_val = weight_plane[k_linear];
                        double m_val = 1.0;
                        if (use_mask) {
                            const double* mask_plane = m_ptr +
                                (n * offset_groups * kH * kW + mask_base + k_linear) * H_out * W_out;
                            m_val = mask_plane[oh * W_out + ow];
                        }

                        double top_grad = grad_out_val * w_val * m_val;

                        dcn_bilinear_scatter(grad_input_plane, H, W, h_loc, w_loc, static_cast<float>(top_grad));

                        float dh, dw_grad;
                        dcn_bilinear_offset_grad(input_plane, H, W, h_loc, w_loc, dh, dw_grad);

                        double* grad_off_h = go_off_ptr +
                            (n * offset_groups * 2 * kH * kW + off_h_idx) * H_out * W_out;
                        double* grad_off_w = go_off_ptr +
                            (n * offset_groups * 2 * kH * kW + off_w_idx) * H_out * W_out;

                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            ref_h(grad_off_h[oh * W_out + ow]);
                        ref_h.fetch_add(static_cast<double>(grad_out_val * w_val * m_val * dh));

                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            ref_w(grad_off_w[oh * W_out + ow]);
                        ref_w.fetch_add(static_cast<double>(grad_out_val * w_val * m_val * dw_grad));

                        if (use_mask && gm_ptr) {
                            double interp_val = dcn_bilinear_sample(input_plane, H, W, h_loc, w_loc);
                            double* grad_mask_plane = gm_ptr +
                                (n * offset_groups * kH * kW + mask_base + k_linear) * H_out * W_out;
                            sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                ref_m(grad_mask_plane[oh * W_out + ow]);
                            ref_m.fetch_add(static_cast<double>(grad_out_val * w_val * interp_val));
                        }
                    }
                }
            }
        });
    } else {
        throw std::runtime_error("deformable_conv2d_backward_input: unsupported dtype (requires Float32 or Float64)");
    }

    if (use_mask) {
        return {grad_input, grad_offset, grad_mask};
    }
    return {grad_input, grad_offset};
}

auto deformable_conv2d_backward_weight_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& offset,
    const Tensor& mask,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    const std::vector<int64_t>& weight_shape,
    sycl::queue& queue) -> Tensor {

    // Float16 / BFloat16: widen to Float32, compute, narrow back.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig_dtype = input.dtype();
        auto g32 = grad_output.to(DType::Float32);
        auto i32 = input.to(DType::Float32);
        auto o32 = offset.to(DType::Float32);
        auto m32 = (mask.numel() > 0) ? mask.to(DType::Float32) : mask;
        Tensor result = deformable_conv2d_backward_weight_kernel(
            g32, i32, o32, m32,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, weight_shape, queue);
        return result.to(orig_dtype);
    }

    auto ishape = input.shape();
    const int64_t N = ishape[0], C_in = ishape[1], H = ishape[2], W = ishape[3];
    const int64_t C_out = weight_shape[0], kH = weight_shape[2], kW = weight_shape[3];
    const int64_t H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];

    const bool use_mask = mask.numel() > 0;
    const int64_t channels_per_group = C_in / groups;
    const int64_t out_channels_per_group = C_out / groups;
    const int64_t channels_per_offset_group = C_in / offset_groups;

    // One work item per (oc, ic_local, kh, kw) — each accumulates over N * H_out * W_out
    const int64_t total = C_out * channels_per_group * kH * kW;

    Tensor grad_weight(weight_shape, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        float* gw_ptr = get_data_ptr<float>(grad_weight);
        queue.fill(gw_ptr, 0.0f, static_cast<size_t>(C_out * channels_per_group * kH * kW));
        queue.wait_and_throw();

        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* off_ptr = get_data_ptr<const float>(offset);
        const float* m_ptr = use_mask ? get_data_ptr<const float>(mask) : nullptr;

        queue.parallel_for<DeformableConv2dBackwardWeightF32>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t kw_idx = idx % kW;
            const int64_t kh_idx = (idx / kW) % kH;
            const int64_t ic_local = (idx / (kW * kH)) % channels_per_group;
            const int64_t oc = idx / (kW * kH * channels_per_group);

            const int64_t g = oc / out_channels_per_group;
            const int64_t ic = g * channels_per_group + ic_local;
            const int64_t og = ic / channels_per_offset_group;

            const int64_t k_linear = kh_idx * kW + kw_idx;
            const int64_t offset_base = og * 2 * kH * kW;
            const int64_t mask_base = og * kH * kW;

            float sum = 0.0f;

            for (int64_t n = 0; n < N; ++n) {
                const float* input_plane = in_ptr + (n * C_in + ic) * H * W;

                for (int64_t oh = 0; oh < H_out; ++oh) {
                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        float h_base = oh * stride_h - pad_h + kh_idx * dil_h;
                        float w_base = ow * stride_w - pad_w + kw_idx * dil_w;

                        const float* off_h_plane = off_ptr +
                            (n * offset_groups * 2 * kH * kW + offset_base + 2 * k_linear) * H_out * W_out;
                        const float* off_w_plane = off_ptr +
                            (n * offset_groups * 2 * kH * kW + offset_base + 2 * k_linear + 1) * H_out * W_out;

                        float h_loc = h_base + off_h_plane[oh * W_out + ow];
                        float w_loc = w_base + off_w_plane[oh * W_out + ow];

                        float val = dcn_bilinear_sample(input_plane, H, W, h_loc, w_loc);

                        if (use_mask) {
                            const float* mask_plane = m_ptr +
                                (n * offset_groups * kH * kW + mask_base + k_linear) * H_out * W_out;
                            val *= mask_plane[oh * W_out + ow];
                        }

                        float go = grad_out_ptr[(n * C_out + oc) * H_out * W_out + oh * W_out + ow];
                        sum += go * val;
                    }
                }
            }

            gw_ptr[(oc * channels_per_group + ic_local) * kH * kW + k_linear] = sum;
        });
    } else if (input.dtype() == DType::Float64) {
        double* gw_ptr = get_data_ptr<double>(grad_weight);
        queue.fill(gw_ptr, 0.0, static_cast<size_t>(C_out * channels_per_group * kH * kW));
        queue.wait_and_throw();

        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* off_ptr = get_data_ptr<const double>(offset);
        const double* m_ptr = use_mask ? get_data_ptr<const double>(mask) : nullptr;

        queue.parallel_for<DeformableConv2dBackwardWeightF64>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t kw_idx = idx % kW;
            const int64_t kh_idx = (idx / kW) % kH;
            const int64_t ic_local = (idx / (kW * kH)) % channels_per_group;
            const int64_t oc = idx / (kW * kH * channels_per_group);

            const int64_t g = oc / out_channels_per_group;
            const int64_t ic = g * channels_per_group + ic_local;
            const int64_t og = ic / channels_per_offset_group;

            const int64_t k_linear = kh_idx * kW + kw_idx;
            const int64_t offset_base = og * 2 * kH * kW;
            const int64_t mask_base = og * kH * kW;

            double sum = 0.0;

            for (int64_t n = 0; n < N; ++n) {
                const double* input_plane = in_ptr + (n * C_in + ic) * H * W;

                for (int64_t oh = 0; oh < H_out; ++oh) {
                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        float h_base = static_cast<float>(oh * stride_h - pad_h + kh_idx * dil_h);
                        float w_base = static_cast<float>(ow * stride_w - pad_w + kw_idx * dil_w);

                        const double* off_h_plane = off_ptr +
                            (n * offset_groups * 2 * kH * kW + offset_base + 2 * k_linear) * H_out * W_out;
                        const double* off_w_plane = off_ptr +
                            (n * offset_groups * 2 * kH * kW + offset_base + 2 * k_linear + 1) * H_out * W_out;

                        float h_loc = h_base + static_cast<float>(off_h_plane[oh * W_out + ow]);
                        float w_loc = w_base + static_cast<float>(off_w_plane[oh * W_out + ow]);

                        double val = dcn_bilinear_sample(input_plane, H, W, h_loc, w_loc);

                        if (use_mask) {
                            const double* mask_plane = m_ptr +
                                (n * offset_groups * kH * kW + mask_base + k_linear) * H_out * W_out;
                            val *= mask_plane[oh * W_out + ow];
                        }

                        double go = grad_out_ptr[(n * C_out + oc) * H_out * W_out + oh * W_out + ow];
                        sum += go * val;
                    }
                }
            }

            gw_ptr[(oc * channels_per_group + ic_local) * kH * kW + k_linear] = sum;
        });
    } else {
        throw std::runtime_error("deformable_conv2d_backward_weight: unsupported dtype (requires Float32 or Float64)");
    }

    return grad_weight;
}

// ============================================================================
// Depthwise Conv1d / Conv3d (forward, groups == channels).
//   Conv1d: input [N,C,1,L], weight [C,1,1,kL], output [N,C,1,L_out].
//   Conv3d: input [N,C,D,H,W], weight [C,1,kD,kH,kW], output [N,C,Do,Ho,Wo].
// Float32/Float64 native; Float16/BFloat16 widen to Float32. Backward is
// autograd-composed in the NN layer.
// ============================================================================
namespace {
template <typename T> struct DWConv1dK {};
template <typename T> struct DWConv3dK {};

template <typename T>
void run_dwconv1d(sycl::queue& q, const T* in, const T* w, const T* bias, T* out,
                  int N, int C, int L, int kL, int Lo, int S, int P, int D) {
    int total = N * C * Lo;
    if (total == 0) return;
    q.parallel_for<DWConv1dK<T>>(sycl::range<1>(total), [=](sycl::id<1> i_) {
        int idx = (int)i_;
        int ol = idx % Lo;
        int c  = (idx / Lo) % C;
        int n  = idx / (C * Lo);
        const T* in_nc = in + (n * C + c) * L;
        const T* w_c   = w + c * kL;
        T acc = bias ? bias[c] : T(0);
        for (int k = 0; k < kL; ++k) {
            int il = ol * S - P + k * D;
            if (il >= 0 && il < L) acc += in_nc[il] * w_c[k];
        }
        out[(n * C + c) * Lo + ol] = acc;
    });
    q.wait_and_throw();
}

template <typename T>
void run_dwconv3d(sycl::queue& q, const T* in, const T* w, const T* bias, T* out,
                  int N, int C, int Di, int Hi, int Wi, int kD, int kH, int kW,
                  int Do, int Ho, int Wo,
                  int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW) {
    int total = N * C * Do * Ho * Wo;
    if (total == 0) return;
    q.parallel_for<DWConv3dK<T>>(sycl::range<1>(total), [=](sycl::id<1> i_) {
        int idx = (int)i_;
        int ow = idx % Wo;
        int oh = (idx / Wo) % Ho;
        int od = (idx / (Wo * Ho)) % Do;
        int c  = (idx / (Wo * Ho * Do)) % C;
        int n  = idx / (C * Do * Ho * Wo);
        const T* in_nc = in + (n * C + c) * Di * Hi * Wi;
        const T* w_c   = w + c * kD * kH * kW;
        T acc = bias ? bias[c] : T(0);
        for (int kd = 0; kd < kD; ++kd) {
            int id = od * sD - pD + kd * dD;
            if (id < 0 || id >= Di) continue;
            for (int kh = 0; kh < kH; ++kh) {
                int ih = oh * sH - pH + kh * dH;
                if (ih < 0 || ih >= Hi) continue;
                for (int kw = 0; kw < kW; ++kw) {
                    int iw = ow * sW - pW + kw * dW;
                    if (iw < 0 || iw >= Wi) continue;
                    acc += in_nc[(id * Hi + ih) * Wi + iw] * w_c[(kd * kH + kh) * kW + kw];
                }
            }
        }
        out[(((n * C + c) * Do + od) * Ho + oh) * Wo + ow] = acc;
    });
    q.wait_and_throw();
}
}  // namespace

auto depthwise_conv1d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                             int64_t stride, int64_t padding, int64_t dilation,
                             sycl::queue& queue) -> Tensor {
    auto is = input.shape();
    auto ws = weight.shape();
    int N = (int)is[0], C = (int)is[1], L = (int)is[3], kL = (int)ws[3];
    int Lo = (int)((L + 2 * padding - dilation * (kL - 1) - 1) / stride + 1);
    if (Lo <= 0) throw std::runtime_error("depthwise_conv1d (OneAPI): non-positive output length");

    auto run = [&](DType dt) {
        Tensor in = input.dtype() == dt ? input.contiguous() : input.to(dt);
        Tensor w  = weight.dtype() == dt ? weight.contiguous() : weight.to(dt);
        Tensor b; const void* bptr = nullptr;
        if (bias) { b = bias->dtype() == dt ? bias->contiguous() : bias->to(dt); bptr = b.data_ptr(); }
        Tensor out(std::vector<int64_t>{(int64_t)N, (int64_t)C, 1, (int64_t)Lo}, dt, input.device());
        if (dt == DType::Float64) {
            run_dwconv1d<double>(queue, get_data_ptr<const double>(in), get_data_ptr<const double>(w),
                (const double*)bptr, get_data_ptr<double>(out), N, C, L, kL, Lo,
                (int)stride, (int)padding, (int)dilation);
        } else {
            run_dwconv1d<float>(queue, get_data_ptr<const float>(in), get_data_ptr<const float>(w),
                (const float*)bptr, get_data_ptr<float>(out), N, C, L, kL, Lo,
                (int)stride, (int)padding, (int)dilation);
        }
        return out;
    };

    DType in_dt = input.dtype();
    if (in_dt == DType::Float64) return run(DType::Float64);
    if (in_dt == DType::Float32) return run(DType::Float32);
    if (in_dt == DType::Float16 || in_dt == DType::BFloat16) return run(DType::Float32).to(in_dt);
    throw std::runtime_error("depthwise_conv1d (OneAPI): unsupported dtype");
}

auto depthwise_conv3d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                             int64_t sD, int64_t sH, int64_t sW,
                             int64_t pD, int64_t pH, int64_t pW,
                             int64_t dD, int64_t dH, int64_t dW,
                             sycl::queue& queue) -> Tensor {
    auto is = input.shape();
    auto ws = weight.shape();
    int N = (int)is[0], C = (int)is[1], Di = (int)is[2], Hi = (int)is[3], Wi = (int)is[4];
    int kD = (int)ws[2], kH = (int)ws[3], kW = (int)ws[4];
    int Do = (int)((Di + 2 * pD - dD * (kD - 1) - 1) / sD + 1);
    int Ho = (int)((Hi + 2 * pH - dH * (kH - 1) - 1) / sH + 1);
    int Wo = (int)((Wi + 2 * pW - dW * (kW - 1) - 1) / sW + 1);
    if (Do <= 0 || Ho <= 0 || Wo <= 0) throw std::runtime_error("depthwise_conv3d (OneAPI): non-positive output size");

    auto run = [&](DType dt) {
        Tensor in = input.dtype() == dt ? input.contiguous() : input.to(dt);
        Tensor w  = weight.dtype() == dt ? weight.contiguous() : weight.to(dt);
        Tensor b; const void* bptr = nullptr;
        if (bias) { b = bias->dtype() == dt ? bias->contiguous() : bias->to(dt); bptr = b.data_ptr(); }
        Tensor out(std::vector<int64_t>{(int64_t)N, (int64_t)C, (int64_t)Do, (int64_t)Ho, (int64_t)Wo}, dt, input.device());
        if (dt == DType::Float64) {
            run_dwconv3d<double>(queue, get_data_ptr<const double>(in), get_data_ptr<const double>(w),
                (const double*)bptr, get_data_ptr<double>(out), N, C, Di, Hi, Wi, kD, kH, kW, Do, Ho, Wo,
                (int)sD,(int)sH,(int)sW,(int)pD,(int)pH,(int)pW,(int)dD,(int)dH,(int)dW);
        } else {
            run_dwconv3d<float>(queue, get_data_ptr<const float>(in), get_data_ptr<const float>(w),
                (const float*)bptr, get_data_ptr<float>(out), N, C, Di, Hi, Wi, kD, kH, kW, Do, Ho, Wo,
                (int)sD,(int)sH,(int)sW,(int)pD,(int)pH,(int)pW,(int)dD,(int)dH,(int)dW);
        }
        return out;
    };

    DType in_dt = input.dtype();
    if (in_dt == DType::Float64) return run(DType::Float64);
    if (in_dt == DType::Float32) return run(DType::Float32);
    if (in_dt == DType::Float16 || in_dt == DType::BFloat16) return run(DType::Float32).to(in_dt);
    throw std::runtime_error("depthwise_conv3d (OneAPI): unsupported dtype");
}

} // namespace oneapi
} // namespace tenzor
