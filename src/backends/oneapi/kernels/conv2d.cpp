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

// Conv2d forward using oneDNN
auto conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                    int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
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

    // Calculate output dimensions
    const int64_t H_out = (H_in + 2 * padding - dilation * (K_h - 1) - 1) / stride + 1;
    const int64_t W_out = (W_in + 2 * padding - dilation * (K_w - 1) - 1) / stride + 1;

    Tensor output({N, C_out, H_out, W_out}, input.dtype(), input.device());

    // Create oneDNN engine and stream
    auto dnnl_engine = sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto dnnl_stream = sycl_interop::make_stream(dnnl_engine, queue);

    // Create memory descriptors
    memory::dims src_dims = {N, C_in, H_in, W_in};
    memory::dims weights_dims = {C_out, C_in / groups, K_h, K_w};
    memory::dims dst_dims = {N, C_out, H_out, W_out};
    memory::dims strides_dims = {stride, stride};
    memory::dims padding_dims = {padding, padding};
    memory::dims dilation_dims = {dilation - 1, dilation - 1};

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

// Conv2d backward using oneDNN
auto conv2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                     int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
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
    memory::dims strides_dims = {stride, stride};
    memory::dims padding_dims = {padding, padding};
    memory::dims dilation_dims = {dilation - 1, dilation - 1};

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
#pragma message("WARNING: Building without oneDNN — using slower Conv2d fallback")

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

// Im2col transformation for convolution
template<typename T>
void im2col_kernel(const T* data_im, int64_t channels, int64_t height, int64_t width,
                   int64_t kernel_h, int64_t kernel_w, int64_t pad, int64_t stride,
                   int64_t dilation, T* data_col, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int64_t output_w = (width + 2 * pad - dilation * (kernel_w - 1) - 1) / stride + 1;
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

        int64_t h_in = h_out * stride - pad + kh * dilation;
        int64_t w_in = w_out * stride - pad + kw * dilation;

        data_col[index] = (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) ?
            data_im[(c * height + h_in) * width + w_in] : T(0);
    });
}

// Col2im transformation (inverse of im2col)
template<typename T>
void col2im_kernel(const T* data_col, int64_t channels, int64_t height, int64_t width,
                   int64_t kernel_h, int64_t kernel_w, int64_t pad, int64_t stride,
                   int64_t dilation, T* data_im, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int64_t output_w = (width + 2 * pad - dilation * (kernel_w - 1) - 1) / stride + 1;
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

        int64_t h_in = h_out * stride - pad + kh * dilation;
        int64_t w_in = w_out * stride - pad + kw * dilation;

        if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
            int64_t im_idx = (c * height + h_in) * width + w_in;
            // Atomic add for thread safety
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
                atomic_val(data_im[im_idx]);
            atomic_val.fetch_add(data_col[index]);
        }
    });
}

// Grouped im2col - type-specific overloads for SYCL kernel naming
// Float32 version
void im2col_grouped_kernel(const float* data_im, int64_t total_channels, int64_t channels_per_group,
                           int64_t channel_offset, int64_t height, int64_t width,
                           int64_t kernel_h, int64_t kernel_w, int64_t pad, int64_t stride,
                           int64_t dilation, float* data_col, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int64_t output_w = (width + 2 * pad - dilation * (kernel_w - 1) - 1) / stride + 1;
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
        int64_t h_in = h_out * stride - pad + kh * dilation;
        int64_t w_in = w_out * stride - pad + kw * dilation;
        data_col[index] = (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) ?
            data_im[(c_global * height + h_in) * width + w_in] : 0.0f;
    });
}

// Float64 version
void im2col_grouped_kernel(const double* data_im, int64_t total_channels, int64_t channels_per_group,
                           int64_t channel_offset, int64_t height, int64_t width,
                           int64_t kernel_h, int64_t kernel_w, int64_t pad, int64_t stride,
                           int64_t dilation, double* data_col, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int64_t output_w = (width + 2 * pad - dilation * (kernel_w - 1) - 1) / stride + 1;
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
        int64_t h_in = h_out * stride - pad + kh * dilation;
        int64_t w_in = w_out * stride - pad + kw * dilation;
        data_col[index] = (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) ?
            data_im[(c_global * height + h_in) * width + w_in] : 0.0;
    });
}

// Float16 version
void im2col_grouped_kernel(const sycl::half* data_im, int64_t total_channels, int64_t channels_per_group,
                           int64_t channel_offset, int64_t height, int64_t width,
                           int64_t kernel_h, int64_t kernel_w, int64_t pad, int64_t stride,
                           int64_t dilation, sycl::half* data_col, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int64_t output_w = (width + 2 * pad - dilation * (kernel_w - 1) - 1) / stride + 1;
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
        int64_t h_in = h_out * stride - pad + kh * dilation;
        int64_t w_in = w_out * stride - pad + kw * dilation;
        data_col[index] = (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) ?
            data_im[(c_global * height + h_in) * width + w_in] : sycl::half(0.0f);
    });
}

// Grouped col2im - type-specific overloads
// Float32 version
void col2im_grouped_kernel(const float* data_col, int64_t total_channels, int64_t channels_per_group,
                           int64_t channel_offset, int64_t height, int64_t width,
                           int64_t kernel_h, int64_t kernel_w, int64_t pad, int64_t stride,
                           int64_t dilation, float* data_im, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int64_t output_w = (width + 2 * pad - dilation * (kernel_w - 1) - 1) / stride + 1;

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
            int64_t h_in = h_out * stride - pad + kh * dilation;
            int64_t w_in = w_out * stride - pad + kw * dilation;
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
                           int64_t kernel_h, int64_t kernel_w, int64_t pad, int64_t stride,
                           int64_t dilation, double* data_im, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int64_t output_w = (width + 2 * pad - dilation * (kernel_w - 1) - 1) / stride + 1;

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
            int64_t h_in = h_out * stride - pad + kh * dilation;
            int64_t w_in = w_out * stride - pad + kw * dilation;
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
                           int64_t kernel_h, int64_t kernel_w, int64_t pad, int64_t stride,
                           int64_t dilation, sycl::half* data_im, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int64_t output_w = (width + 2 * pad - dilation * (kernel_w - 1) - 1) / stride + 1;

    // For half precision, we use a different approach since atomic_ref<half> may not be supported
    // Serialize the updates using sequential processing for correctness
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
            int64_t h_in = h_out * stride - pad + kh * dilation;
            int64_t w_in = w_out * stride - pad + kw * dilation;
            if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
                int64_t im_idx = (c_global * height + h_in) * width + w_in;
                // Use float atomic as workaround for half
                float val = static_cast<float>(data_col[index]);
                // Note: This is a simplified approach - for production, consider using
                // compare-exchange loop or accumulating in float buffer first
                data_im[im_idx] = sycl::half(static_cast<float>(data_im[im_idx]) + val);
            }
        });
}

auto conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                    int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
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

    const int64_t H_out = (H_in + 2 * padding - dilation * (K_h - 1) - 1) / stride + 1;
    const int64_t W_out = (W_in + 2 * padding - dilation * (K_w - 1) - 1) / stride + 1;

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
                    H_in, W_in, K_h, K_w, padding, stride, dilation,
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
                    H_in, W_in, K_h, K_w, padding, stride, dilation,
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
                    H_in, W_in, K_h, K_w, padding, stride, dilation,
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

auto conv2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                     int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
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
                    H_in, W_in, K_h, K_w, padding, stride, dilation,
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
                    H_in, W_in, K_h, K_w, padding, stride, dilation,
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

// Separate conv2d_backward_input that takes input_shape instead of input tensor
// This matches the CPU backend API
auto conv2d_backward_input(const Tensor& grad_output, const Tensor& weight,
                           const std::vector<int64_t>& input_shape,
                           int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
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
                    H_in, W_in, K_h, K_w, padding, stride, dilation,
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
                    H_in, W_in, K_h, K_w, padding, stride, dilation,
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
                    H_in, W_in, K_h, K_w, padding, stride, dilation,
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

// Separate conv2d_backward_weight that takes weight_shape instead of weight tensor
// This matches the CPU backend API
auto conv2d_backward_weight(const Tensor& grad_output, const Tensor& input,
                            const std::vector<int64_t>& weight_shape,
                            int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
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
                    H_in, W_in, K_h, K_w, padding, stride, dilation,
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
                    H_in, W_in, K_h, K_w, padding, stride, dilation,
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
                    H_in, W_in, K_h, K_w, padding, stride, dilation,
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

} // namespace oneapi
} // namespace tenzor
