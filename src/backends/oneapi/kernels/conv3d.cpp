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

// ============================================================================
// Helpers
// ============================================================================

constexpr float HALF_MAX_CONV3D = 65504.0f;
inline sycl::half saturate_to_half_conv3d(float val) {
    return sycl::half(sycl::fmin(sycl::fmax(val, -HALF_MAX_CONV3D), HALF_MAX_CONV3D));
}

template<typename T>
inline auto get_data_ptr_conv3d(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

static inline int64_t calc_out_size_3d(int64_t in, int64_t kernel, int64_t stride,
                                        int64_t padding, int64_t dilation) {
    return (in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

// ============================================================================
// SYCL kernel class declarations for Conv3d
// ============================================================================

// Float32
class Conv3dIm3colKernel;
class Conv3dCol3imKernel;
class Conv3dForwardGemmKernel;
class Conv3dForwardBiasKernel;
class Conv3dBackwardInputGemmKernel;
class Conv3dBackwardWeightGemmKernel;
class Conv3dBackwardBiasKernel;
class Conv3dGroupedIm3colKernel;
class Conv3dGroupedCol3imKernel;
class Conv3dGroupedForwardGemmKernel;
class Conv3dGroupedBackwardInputGemmKernel;
class Conv3dGroupedBackwardWeightGemmKernel;

// Float64
class Conv3dGroupedIm3colKernelFloat64;
class Conv3dGroupedCol3imKernelFloat64;
class Conv3dGroupedForwardGemmKernelFloat64;
class Conv3dForwardBiasKernelFloat64;
class Conv3dBackwardInputGemmKernelFloat64;
class Conv3dBackwardWeightGemmKernelFloat64;
class Conv3dBackwardBiasKernelFloat64;

// Float16
class Conv3dGroupedIm3colKernelFloat16;
class Conv3dGroupedCol3imKernelFloat16;
class Conv3dGroupedForwardGemmKernelFloat16;
class Conv3dForwardBiasKernelFloat16;
class Conv3dBackwardInputGemmKernelFloat16;
class Conv3dBackwardWeightGemmKernelFloat16;
class Conv3dBackwardBiasKernelFloat16;

// ConvTranspose3d
class ConvTranspose3dForwardKernelFloat32;
class ConvTranspose3dForwardKernelFloat64;
class ConvTranspose3dForwardKernelFloat16;
class ConvTranspose3dBiasKernelFloat32;
class ConvTranspose3dBiasKernelFloat64;
class ConvTranspose3dBiasKernelFloat16;

// ============================================================================
// im3col grouped kernel - type-specific overloads
// ============================================================================

// Float32
void im3col_grouped_3d_kernel(const float* data_im, int64_t total_channels, int64_t channels_per_group,
                               int64_t channel_offset, int64_t depth, int64_t height, int64_t width,
                               int64_t kD, int64_t kH, int64_t kW,
                               int64_t stride_d, int64_t stride_h, int64_t stride_w,
                               int64_t pad_d, int64_t pad_h, int64_t pad_w,
                               int64_t dil_d, int64_t dil_h, int64_t dil_w,
                               int64_t out_d, int64_t out_h, int64_t out_w,
                               float* data_col, sycl::queue& queue) {
    const int64_t col_size = channels_per_group * kD * kH * kW * out_d * out_h * out_w;

    queue.parallel_for<Conv3dGroupedIm3colKernel>(sycl::range<1>(col_size), [=](sycl::id<1> index) {
        int64_t tmp = index;
        int64_t ow = tmp % out_w; tmp /= out_w;
        int64_t oh = tmp % out_h; tmp /= out_h;
        int64_t od = tmp % out_d; tmp /= out_d;
        int64_t kw = tmp % kW; tmp /= kW;
        int64_t kh = tmp % kH; tmp /= kH;
        int64_t kd = tmp % kD; tmp /= kD;
        int64_t c_local = tmp;
        int64_t c_global = channel_offset + c_local;

        int64_t id = od * stride_d - pad_d + kd * dil_d;
        int64_t ih = oh * stride_h - pad_h + kh * dil_h;
        int64_t iw = ow * stride_w - pad_w + kw * dil_w;

        data_col[index] = (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width)
            ? data_im[(c_global * depth + id) * height * width + ih * width + iw]
            : 0.0f;
    }).wait();
}

// Float64
void im3col_grouped_3d_kernel(const double* data_im, int64_t total_channels, int64_t channels_per_group,
                               int64_t channel_offset, int64_t depth, int64_t height, int64_t width,
                               int64_t kD, int64_t kH, int64_t kW,
                               int64_t stride_d, int64_t stride_h, int64_t stride_w,
                               int64_t pad_d, int64_t pad_h, int64_t pad_w,
                               int64_t dil_d, int64_t dil_h, int64_t dil_w,
                               int64_t out_d, int64_t out_h, int64_t out_w,
                               double* data_col, sycl::queue& queue) {
    const int64_t col_size = channels_per_group * kD * kH * kW * out_d * out_h * out_w;

    queue.parallel_for<Conv3dGroupedIm3colKernelFloat64>(sycl::range<1>(col_size), [=](sycl::id<1> index) {
        int64_t tmp = index;
        int64_t ow = tmp % out_w; tmp /= out_w;
        int64_t oh = tmp % out_h; tmp /= out_h;
        int64_t od = tmp % out_d; tmp /= out_d;
        int64_t kw = tmp % kW; tmp /= kW;
        int64_t kh = tmp % kH; tmp /= kH;
        int64_t kd = tmp % kD; tmp /= kD;
        int64_t c_local = tmp;
        int64_t c_global = channel_offset + c_local;

        int64_t id = od * stride_d - pad_d + kd * dil_d;
        int64_t ih = oh * stride_h - pad_h + kh * dil_h;
        int64_t iw = ow * stride_w - pad_w + kw * dil_w;

        data_col[index] = (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width)
            ? data_im[(c_global * depth + id) * height * width + ih * width + iw]
            : 0.0;
    }).wait();
}

// Float16
void im3col_grouped_3d_kernel(const sycl::half* data_im, int64_t total_channels, int64_t channels_per_group,
                               int64_t channel_offset, int64_t depth, int64_t height, int64_t width,
                               int64_t kD, int64_t kH, int64_t kW,
                               int64_t stride_d, int64_t stride_h, int64_t stride_w,
                               int64_t pad_d, int64_t pad_h, int64_t pad_w,
                               int64_t dil_d, int64_t dil_h, int64_t dil_w,
                               int64_t out_d, int64_t out_h, int64_t out_w,
                               sycl::half* data_col, sycl::queue& queue) {
    const int64_t col_size = channels_per_group * kD * kH * kW * out_d * out_h * out_w;

    queue.parallel_for<Conv3dGroupedIm3colKernelFloat16>(sycl::range<1>(col_size), [=](sycl::id<1> index) {
        int64_t tmp = index;
        int64_t ow = tmp % out_w; tmp /= out_w;
        int64_t oh = tmp % out_h; tmp /= out_h;
        int64_t od = tmp % out_d; tmp /= out_d;
        int64_t kw = tmp % kW; tmp /= kW;
        int64_t kh = tmp % kH; tmp /= kH;
        int64_t kd = tmp % kD; tmp /= kD;
        int64_t c_local = tmp;
        int64_t c_global = channel_offset + c_local;

        int64_t id = od * stride_d - pad_d + kd * dil_d;
        int64_t ih = oh * stride_h - pad_h + kh * dil_h;
        int64_t iw = ow * stride_w - pad_w + kw * dil_w;

        data_col[index] = (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width)
            ? data_im[(c_global * depth + id) * height * width + ih * width + iw]
            : sycl::half(0.0f);
    }).wait();
}

// ============================================================================
// col3im grouped kernel - type-specific overloads
// ============================================================================

// Float32
void col3im_grouped_3d_kernel(const float* data_col, int64_t total_channels, int64_t channels_per_group,
                               int64_t channel_offset, int64_t depth, int64_t height, int64_t width,
                               int64_t kD, int64_t kH, int64_t kW,
                               int64_t stride_d, int64_t stride_h, int64_t stride_w,
                               int64_t pad_d, int64_t pad_h, int64_t pad_w,
                               int64_t dil_d, int64_t dil_h, int64_t dil_w,
                               int64_t out_d, int64_t out_h, int64_t out_w,
                               float* data_im, sycl::queue& queue) {
    const int64_t col_size = channels_per_group * kD * kH * kW * out_d * out_h * out_w;

    queue.parallel_for<Conv3dGroupedCol3imKernel>(sycl::range<1>(col_size), [=](sycl::id<1> index) {
        int64_t tmp = index;
        int64_t ow = tmp % out_w; tmp /= out_w;
        int64_t oh = tmp % out_h; tmp /= out_h;
        int64_t od = tmp % out_d; tmp /= out_d;
        int64_t kw = tmp % kW; tmp /= kW;
        int64_t kh = tmp % kH; tmp /= kH;
        int64_t kd = tmp % kD; tmp /= kD;
        int64_t c_local = tmp;
        int64_t c_global = channel_offset + c_local;

        int64_t id = od * stride_d - pad_d + kd * dil_d;
        int64_t ih = oh * stride_h - pad_h + kh * dil_h;
        int64_t iw = ow * stride_w - pad_w + kw * dil_w;

        if (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width) {
            int64_t im_idx = (c_global * depth + id) * height * width + ih * width + iw;
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                atomic_val(data_im[im_idx]);
            atomic_val.fetch_add(data_col[index]);
        }
    }).wait();
}

// Float64
void col3im_grouped_3d_kernel(const double* data_col, int64_t total_channels, int64_t channels_per_group,
                               int64_t channel_offset, int64_t depth, int64_t height, int64_t width,
                               int64_t kD, int64_t kH, int64_t kW,
                               int64_t stride_d, int64_t stride_h, int64_t stride_w,
                               int64_t pad_d, int64_t pad_h, int64_t pad_w,
                               int64_t dil_d, int64_t dil_h, int64_t dil_w,
                               int64_t out_d, int64_t out_h, int64_t out_w,
                               double* data_im, sycl::queue& queue) {
    const int64_t col_size = channels_per_group * kD * kH * kW * out_d * out_h * out_w;

    queue.parallel_for<Conv3dGroupedCol3imKernelFloat64>(sycl::range<1>(col_size), [=](sycl::id<1> index) {
        int64_t tmp = index;
        int64_t ow = tmp % out_w; tmp /= out_w;
        int64_t oh = tmp % out_h; tmp /= out_h;
        int64_t od = tmp % out_d; tmp /= out_d;
        int64_t kw = tmp % kW; tmp /= kW;
        int64_t kh = tmp % kH; tmp /= kH;
        int64_t kd = tmp % kD; tmp /= kD;
        int64_t c_local = tmp;
        int64_t c_global = channel_offset + c_local;

        int64_t id = od * stride_d - pad_d + kd * dil_d;
        int64_t ih = oh * stride_h - pad_h + kh * dil_h;
        int64_t iw = ow * stride_w - pad_w + kw * dil_w;

        if (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width) {
            int64_t im_idx = (c_global * depth + id) * height * width + ih * width + iw;
            sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>
                atomic_val(data_im[im_idx]);
            atomic_val.fetch_add(data_col[index]);
        }
    }).wait();
}

// Float16 (uses float cast for atomics)
void col3im_grouped_3d_kernel(const sycl::half* data_col, int64_t total_channels, int64_t channels_per_group,
                               int64_t channel_offset, int64_t depth, int64_t height, int64_t width,
                               int64_t kD, int64_t kH, int64_t kW,
                               int64_t stride_d, int64_t stride_h, int64_t stride_w,
                               int64_t pad_d, int64_t pad_h, int64_t pad_w,
                               int64_t dil_d, int64_t dil_h, int64_t dil_w,
                               int64_t out_d, int64_t out_h, int64_t out_w,
                               sycl::half* data_im, sycl::queue& queue) {
    const int64_t col_size = channels_per_group * kD * kH * kW * out_d * out_h * out_w;

    queue.parallel_for<Conv3dGroupedCol3imKernelFloat16>(sycl::range<1>(col_size), [=](sycl::id<1> index) {
        int64_t tmp = index;
        int64_t ow = tmp % out_w; tmp /= out_w;
        int64_t oh = tmp % out_h; tmp /= out_h;
        int64_t od = tmp % out_d; tmp /= out_d;
        int64_t kw = tmp % kW; tmp /= kW;
        int64_t kh = tmp % kH; tmp /= kH;
        int64_t kd = tmp % kD; tmp /= kD;
        int64_t c_local = tmp;
        int64_t c_global = channel_offset + c_local;

        int64_t id = od * stride_d - pad_d + kd * dil_d;
        int64_t ih = oh * stride_h - pad_h + kh * dil_h;
        int64_t iw = ow * stride_w - pad_w + kw * dil_w;

        if (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width) {
            int64_t im_idx = (c_global * depth + id) * height * width + ih * width + iw;
            float val = static_cast<float>(data_col[index]);
            data_im[im_idx] = sycl::half(static_cast<float>(data_im[im_idx]) + val);
        }
    }).wait();
}

#ifdef TENZOR_HAS_ONEDNN

// ============================================================================
// Conv3d Forward using oneDNN (supports 3D convolution natively)
// ============================================================================
auto conv3d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                    const std::vector<int64_t>& dilation, int64_t groups,
                    sycl::queue& queue) -> Tensor {
    using namespace dnnl;

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    if (input_shape.size() != 5 || weight_shape.size() != 5) {
        throw std::invalid_argument("Conv3d requires 5D tensors (N,C,D,H,W)");
    }

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t D_in = input_shape[2];
    const int64_t H_in = input_shape[3];
    const int64_t W_in = input_shape[4];

    const int64_t C_out = weight_shape[0];
    const int64_t kD = weight_shape[2];
    const int64_t kH = weight_shape[3];
    const int64_t kW = weight_shape[4];

    const int64_t stride_d = stride.size() > 0 ? stride[0] : 1;
    const int64_t stride_h = stride.size() > 1 ? stride[1] : stride_d;
    const int64_t stride_w = stride.size() > 2 ? stride[2] : stride_h;

    const int64_t pad_d = padding.size() > 0 ? padding[0] : 0;
    const int64_t pad_h = padding.size() > 1 ? padding[1] : pad_d;
    const int64_t pad_w = padding.size() > 2 ? padding[2] : pad_h;

    const int64_t dil_d = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dil_h = dilation.size() > 1 ? dilation[1] : dil_d;
    const int64_t dil_w = dilation.size() > 2 ? dilation[2] : dil_h;

    const int64_t D_out = calc_out_size_3d(D_in, kD, stride_d, pad_d, dil_d);
    const int64_t H_out = calc_out_size_3d(H_in, kH, stride_h, pad_h, dil_h);
    const int64_t W_out = calc_out_size_3d(W_in, kW, stride_w, pad_w, dil_w);

    Tensor output({N, C_out, D_out, H_out, W_out}, input.dtype(), input.device());

    auto dnnl_engine = sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto dnnl_stream = sycl_interop::make_stream(dnnl_engine, queue);

    memory::dims src_dims = {N, C_in, D_in, H_in, W_in};
    memory::dims weights_dims = {C_out, C_in / groups, kD, kH, kW};
    memory::dims dst_dims = {N, C_out, D_out, H_out, W_out};
    memory::dims strides_dims = {stride_d, stride_h, stride_w};
    memory::dims padding_dims = {pad_d, pad_h, pad_w};
    memory::dims dilation_dims = {dil_d - 1, dil_h - 1, dil_w - 1};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::ncdhw);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32,
                                   groups == 1 ? memory::format_tag::oidhw : memory::format_tag::goidhw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::ncdhw);

    auto conv_desc = (bias != nullptr) ?
        convolution_forward::desc(
            prop_kind::forward_inference,
            algorithm::convolution_direct,
            src_md, weights_md,
            memory::desc({C_out}, memory::data_type::f32, memory::format_tag::x),
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

    auto src_mem = sycl_interop::make_memory(conv_pd.src_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(input.data_ptr()));
    auto weights_mem = sycl_interop::make_memory(conv_pd.weights_desc(), dnnl_engine,
                                                  sycl_interop::memory_kind::usm,
                                                  const_cast<void*>(weight.data_ptr()));
    auto dst_mem = sycl_interop::make_memory(conv_pd.dst_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(output.data_ptr()));

    auto conv_prim = convolution_forward(conv_pd);

    if (bias != nullptr) {
        auto bias_mem = sycl_interop::make_memory(
            memory::desc({C_out}, memory::data_type::f32, memory::format_tag::x),
            dnnl_engine, sycl_interop::memory_kind::usm,
            const_cast<void*>(bias->data_ptr()));

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
    return output;
}

// ============================================================================
// Conv3d Backward using oneDNN
// ============================================================================
auto conv3d_backward_input(const Tensor& grad_output, const Tensor& weight,
                            const std::vector<int64_t>& input_shape,
                            const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                            const std::vector<int64_t>& dilation, int64_t groups,
                            sycl::queue& queue) -> Tensor {
    using namespace dnnl;

    auto weight_shape = weight.shape();
    auto grad_shape = grad_output.shape();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t D_in = input_shape[2];
    const int64_t H_in = input_shape[3];
    const int64_t W_in = input_shape[4];

    const int64_t C_out = weight_shape[0];
    const int64_t kD = weight_shape[2];
    const int64_t kH = weight_shape[3];
    const int64_t kW = weight_shape[4];

    const int64_t stride_d = stride.size() > 0 ? stride[0] : 1;
    const int64_t stride_h = stride.size() > 1 ? stride[1] : stride_d;
    const int64_t stride_w = stride.size() > 2 ? stride[2] : stride_h;

    const int64_t pad_d = padding.size() > 0 ? padding[0] : 0;
    const int64_t pad_h = padding.size() > 1 ? padding[1] : pad_d;
    const int64_t pad_w = padding.size() > 2 ? padding[2] : pad_h;

    const int64_t dil_d = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dil_h = dilation.size() > 1 ? dilation[1] : dil_d;
    const int64_t dil_w = dilation.size() > 2 ? dilation[2] : dil_h;

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    auto dnnl_engine = sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto dnnl_stream = sycl_interop::make_stream(dnnl_engine, queue);

    memory::dims src_dims = {N, C_in, D_in, H_in, W_in};
    memory::dims weights_dims = {C_out, C_in / groups, kD, kH, kW};
    memory::dims dst_dims = {grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3], grad_shape[4]};
    memory::dims strides_dims = {stride_d, stride_h, stride_w};
    memory::dims padding_dims = {pad_d, pad_h, pad_w};
    memory::dims dilation_dims = {dil_d - 1, dil_h - 1, dil_w - 1};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::ncdhw);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32,
                                   groups == 1 ? memory::format_tag::oidhw : memory::format_tag::goidhw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::ncdhw);

    // Forward descriptor (needed for backward)
    auto conv_fwd_desc = convolution_forward::desc(
        prop_kind::forward_training,
        algorithm::convolution_direct,
        src_md, weights_md, dst_md,
        strides_dims, dilation_dims, padding_dims, padding_dims);
    auto conv_fwd_pd = convolution_forward::primitive_desc(conv_fwd_desc, dnnl_engine);

    auto conv_bwd_data_desc = convolution_backward_data::desc(
        algorithm::convolution_direct,
        src_md, weights_md, dst_md,
        strides_dims, dilation_dims, padding_dims, padding_dims);
    auto conv_bwd_data_pd = convolution_backward_data::primitive_desc(
        conv_bwd_data_desc, dnnl_engine, conv_fwd_pd);

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
    conv_bwd_data_prim.execute(dnnl_stream, {
        {DNNL_ARG_DIFF_DST, diff_dst_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DIFF_SRC, diff_src_mem}
    });

    dnnl_stream.wait();
    return grad_input;
}

auto conv3d_backward_weight(const Tensor& grad_output, const Tensor& input,
                             const std::vector<int64_t>& weight_shape,
                             const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& dilation, int64_t groups,
                             sycl::queue& queue) -> Tensor {
    using namespace dnnl;

    auto input_shape = input.shape();
    auto grad_shape = grad_output.shape();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t D_in = input_shape[2];
    const int64_t H_in = input_shape[3];
    const int64_t W_in = input_shape[4];

    const int64_t C_out = weight_shape[0];
    const int64_t kD = weight_shape[2];
    const int64_t kH = weight_shape[3];
    const int64_t kW = weight_shape[4];

    const int64_t stride_d = stride.size() > 0 ? stride[0] : 1;
    const int64_t stride_h = stride.size() > 1 ? stride[1] : stride_d;
    const int64_t stride_w = stride.size() > 2 ? stride[2] : stride_h;

    const int64_t pad_d = padding.size() > 0 ? padding[0] : 0;
    const int64_t pad_h = padding.size() > 1 ? padding[1] : pad_d;
    const int64_t pad_w = padding.size() > 2 ? padding[2] : pad_h;

    const int64_t dil_d = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dil_h = dilation.size() > 1 ? dilation[1] : dil_d;
    const int64_t dil_w = dilation.size() > 2 ? dilation[2] : dil_h;

    Tensor grad_weight(weight_shape, input.dtype(), input.device());

    auto dnnl_engine = sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto dnnl_stream = sycl_interop::make_stream(dnnl_engine, queue);

    memory::dims src_dims = {N, C_in, D_in, H_in, W_in};
    memory::dims weights_dims = {C_out, C_in / groups, kD, kH, kW};
    memory::dims dst_dims = {grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3], grad_shape[4]};
    memory::dims strides_dims = {stride_d, stride_h, stride_w};
    memory::dims padding_dims = {pad_d, pad_h, pad_w};
    memory::dims dilation_dims = {dil_d - 1, dil_h - 1, dil_w - 1};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::ncdhw);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32,
                                   groups == 1 ? memory::format_tag::oidhw : memory::format_tag::goidhw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::ncdhw);

    auto conv_fwd_desc = convolution_forward::desc(
        prop_kind::forward_training,
        algorithm::convolution_direct,
        src_md, weights_md, dst_md,
        strides_dims, dilation_dims, padding_dims, padding_dims);
    auto conv_fwd_pd = convolution_forward::primitive_desc(conv_fwd_desc, dnnl_engine);

    auto conv_bwd_weights_desc = convolution_backward_weights::desc(
        algorithm::convolution_direct,
        src_md, weights_md, dst_md,
        strides_dims, dilation_dims, padding_dims, padding_dims);
    auto conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(
        conv_bwd_weights_desc, dnnl_engine, conv_fwd_pd);

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
    conv_bwd_weights_prim.execute(dnnl_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DIFF_DST, diff_dst_mem},
        {DNNL_ARG_DIFF_WEIGHTS, diff_weights_mem}
    });

    dnnl_stream.wait();
    return grad_weight;
}

auto conv3d_backward_bias(const Tensor& grad_output, sycl::queue& queue) -> Tensor {
    auto grad_shape = grad_output.shape();
    const int64_t N = grad_shape[0];
    const int64_t C = grad_shape[1];
    const int64_t D = grad_shape[2];
    const int64_t H = grad_shape[3];
    const int64_t W = grad_shape[4];

    Tensor grad_bias({C}, grad_output.dtype(), grad_output.device());

    const float* grad_output_ptr = get_data_ptr_conv3d<const float>(grad_output);
    float* grad_bias_ptr = get_data_ptr_conv3d<float>(grad_bias);

    queue.parallel_for<Conv3dBackwardBiasKernel>(sycl::range<1>(C), [=](sycl::id<1> c) {
        float sum = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t d = 0; d < D; ++d) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        sum += grad_output_ptr[(((n * C + c) * D + d) * H + h) * W + w];
                    }
                }
            }
        }
        grad_bias_ptr[c] = sum;
    }).wait();

    return grad_bias;
}

#else // !TENZOR_HAS_ONEDNN - Fallback implementation using im3col + SYCL GEMM

// ============================================================================
// Conv3d Forward (im3col + GEMM fallback)
// ============================================================================
auto conv3d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                    const std::vector<int64_t>& dilation, int64_t groups,
                    sycl::queue& queue) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    if (input_shape.size() != 5 || weight_shape.size() != 5) {
        throw std::invalid_argument("Conv3d requires 5D tensors (N,C,D,H,W)");
    }

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t D_in = input_shape[2];
    const int64_t H_in = input_shape[3];
    const int64_t W_in = input_shape[4];

    const int64_t C_out = weight_shape[0];
    const int64_t C_in_per_group = weight_shape[1];
    const int64_t kD = weight_shape[2];
    const int64_t kH = weight_shape[3];
    const int64_t kW = weight_shape[4];

    const int64_t C_out_per_group = C_out / groups;

    const int64_t stride_d = stride.size() > 0 ? stride[0] : 1;
    const int64_t stride_h = stride.size() > 1 ? stride[1] : stride_d;
    const int64_t stride_w = stride.size() > 2 ? stride[2] : stride_h;

    const int64_t pad_d = padding.size() > 0 ? padding[0] : 0;
    const int64_t pad_h = padding.size() > 1 ? padding[1] : pad_d;
    const int64_t pad_w = padding.size() > 2 ? padding[2] : pad_h;

    const int64_t dil_d = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dil_h = dilation.size() > 1 ? dilation[1] : dil_d;
    const int64_t dil_w = dilation.size() > 2 ? dilation[2] : dil_h;

    const int64_t D_out = calc_out_size_3d(D_in, kD, stride_d, pad_d, dil_d);
    const int64_t H_out = calc_out_size_3d(H_in, kH, stride_h, pad_h, dil_h);
    const int64_t W_out = calc_out_size_3d(W_in, kW, stride_w, pad_w, dil_w);

    Tensor output({N, C_out, D_out, H_out, W_out}, input.dtype(), input.device());

    const int64_t K = C_in_per_group * kD * kH * kW;
    const int64_t N_gemm = D_out * H_out * W_out;
    const int64_t M = C_out_per_group;
    const int64_t col_size = K * N_gemm;

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr_conv3d<const float>(input);
        const float* weight_ptr = get_data_ptr_conv3d<const float>(weight);
        float* output_ptr = get_data_ptr_conv3d<float>(output);

        Tensor col_buffer({col_size}, input.dtype(), input.device());
        float* col_ptr = get_data_ptr_conv3d<float>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * K;

                im3col_grouped_3d_kernel(
                    input_ptr + n * C_in * D_in * H_in * W_in,
                    C_in, C_in_per_group, in_channel_offset,
                    D_in, H_in, W_in, kD, kH, kW,
                    stride_d, stride_h, stride_w,
                    pad_d, pad_h, pad_w,
                    dil_d, dil_h, dil_w,
                    D_out, H_out, W_out,
                    col_ptr, queue
                );

                const float* weight_group_ptr = weight_ptr + weight_offset;
                float* output_group_ptr = output_ptr + n * C_out * N_gemm + out_channel_offset * N_gemm;

                queue.parallel_for<Conv3dGroupedForwardGemmKernel>(sycl::range<2>(M, N_gemm), [=](sycl::id<2> idx) {
                    const int64_t oc_local = idx[0];
                    const int64_t hw = idx[1];
                    float sum = 0.0f;
                    for (int64_t k = 0; k < K; ++k) {
                        sum += weight_group_ptr[oc_local * K + k] * col_ptr[k * N_gemm + hw];
                    }
                    output_group_ptr[oc_local * N_gemm + hw] = sum;
                }).wait();
            }

            if (bias != nullptr) {
                const float* bias_ptr = get_data_ptr_conv3d<const float>(*bias);
                queue.parallel_for<Conv3dForwardBiasKernel>(sycl::range<2>(C_out, N_gemm), [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];
                    const int64_t dhw = idx[1];
                    const int64_t out_idx = n * C_out * N_gemm + oc * N_gemm + dhw;
                    output_ptr[out_idx] += bias_ptr[oc];
                }).wait();
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr_conv3d<const double>(input);
        const double* weight_ptr = get_data_ptr_conv3d<const double>(weight);
        double* output_ptr = get_data_ptr_conv3d<double>(output);

        Tensor col_buffer({col_size}, input.dtype(), input.device());
        double* col_ptr = get_data_ptr_conv3d<double>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * K;

                im3col_grouped_3d_kernel(
                    input_ptr + n * C_in * D_in * H_in * W_in,
                    C_in, C_in_per_group, in_channel_offset,
                    D_in, H_in, W_in, kD, kH, kW,
                    stride_d, stride_h, stride_w,
                    pad_d, pad_h, pad_w,
                    dil_d, dil_h, dil_w,
                    D_out, H_out, W_out,
                    col_ptr, queue
                );

                const double* weight_group_ptr = weight_ptr + weight_offset;
                double* output_group_ptr = output_ptr + n * C_out * N_gemm + out_channel_offset * N_gemm;

                queue.parallel_for<Conv3dGroupedForwardGemmKernelFloat64>(sycl::range<2>(M, N_gemm), [=](sycl::id<2> idx) {
                    const int64_t oc_local = idx[0];
                    const int64_t hw = idx[1];
                    double sum = 0.0;
                    for (int64_t k = 0; k < K; ++k) {
                        sum += weight_group_ptr[oc_local * K + k] * col_ptr[k * N_gemm + hw];
                    }
                    output_group_ptr[oc_local * N_gemm + hw] = sum;
                }).wait();
            }

            if (bias != nullptr) {
                const double* bias_ptr = get_data_ptr_conv3d<const double>(*bias);
                queue.parallel_for<Conv3dForwardBiasKernelFloat64>(sycl::range<2>(C_out, N_gemm), [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];
                    const int64_t dhw = idx[1];
                    const int64_t out_idx = n * C_out * N_gemm + oc * N_gemm + dhw;
                    output_ptr[out_idx] += bias_ptr[oc];
                }).wait();
            }
        }
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* input_ptr = get_data_ptr_conv3d<const sycl::half>(input);
        const sycl::half* weight_ptr = get_data_ptr_conv3d<const sycl::half>(weight);
        sycl::half* output_ptr = get_data_ptr_conv3d<sycl::half>(output);

        Tensor col_buffer({col_size}, input.dtype(), input.device());
        sycl::half* col_ptr = get_data_ptr_conv3d<sycl::half>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * K;

                im3col_grouped_3d_kernel(
                    input_ptr + n * C_in * D_in * H_in * W_in,
                    C_in, C_in_per_group, in_channel_offset,
                    D_in, H_in, W_in, kD, kH, kW,
                    stride_d, stride_h, stride_w,
                    pad_d, pad_h, pad_w,
                    dil_d, dil_h, dil_w,
                    D_out, H_out, W_out,
                    col_ptr, queue
                );

                const sycl::half* weight_group_ptr = weight_ptr + weight_offset;
                sycl::half* output_group_ptr = output_ptr + n * C_out * N_gemm + out_channel_offset * N_gemm;

                queue.parallel_for<Conv3dGroupedForwardGemmKernelFloat16>(sycl::range<2>(M, N_gemm), [=](sycl::id<2> idx) {
                    const int64_t oc_local = idx[0];
                    const int64_t hw = idx[1];
                    float sum = 0.0f;
                    for (int64_t k = 0; k < K; ++k) {
                        sum += static_cast<float>(weight_group_ptr[oc_local * K + k]) *
                               static_cast<float>(col_ptr[k * N_gemm + hw]);
                    }
                    output_group_ptr[oc_local * N_gemm + hw] = saturate_to_half_conv3d(sum);
                }).wait();
            }

            if (bias != nullptr) {
                const sycl::half* bias_ptr = get_data_ptr_conv3d<const sycl::half>(*bias);
                queue.parallel_for<Conv3dForwardBiasKernelFloat16>(sycl::range<2>(C_out, N_gemm), [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];
                    const int64_t dhw = idx[1];
                    const int64_t out_idx = n * C_out * N_gemm + oc * N_gemm + dhw;
                    output_ptr[out_idx] = saturate_to_half_conv3d(
                        static_cast<float>(output_ptr[out_idx]) + static_cast<float>(bias_ptr[oc]));
                }).wait();
            }
        }
    } else {
        throw std::runtime_error("Unsupported dtype for conv3d_forward (fallback)");
    }

    return output;
}

// ============================================================================
// Conv3d Backward Input (im3col-based fallback)
// ============================================================================
auto conv3d_backward_input(const Tensor& grad_output, const Tensor& weight,
                            const std::vector<int64_t>& input_shape,
                            const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                            const std::vector<int64_t>& dilation, int64_t groups,
                            sycl::queue& queue) -> Tensor {
    auto weight_shape = weight.shape();
    auto grad_shape = grad_output.shape();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t D_in = input_shape[2];
    const int64_t H_in = input_shape[3];
    const int64_t W_in = input_shape[4];

    const int64_t C_out = weight_shape[0];
    const int64_t C_in_per_group = weight_shape[1];
    const int64_t kD = weight_shape[2];
    const int64_t kH = weight_shape[3];
    const int64_t kW = weight_shape[4];

    const int64_t C_out_per_group = C_out / groups;

    const int64_t D_out = grad_shape[2];
    const int64_t H_out = grad_shape[3];
    const int64_t W_out = grad_shape[4];

    const int64_t stride_d = stride.size() > 0 ? stride[0] : 1;
    const int64_t stride_h = stride.size() > 1 ? stride[1] : stride_d;
    const int64_t stride_w = stride.size() > 2 ? stride[2] : stride_h;

    const int64_t pad_d = padding.size() > 0 ? padding[0] : 0;
    const int64_t pad_h = padding.size() > 1 ? padding[1] : pad_d;
    const int64_t pad_w = padding.size() > 2 ? padding[2] : pad_h;

    const int64_t dil_d = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dil_h = dilation.size() > 1 ? dilation[1] : dil_d;
    const int64_t dil_w = dilation.size() > 2 ? dilation[2] : dil_h;

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    const int64_t K = C_in_per_group * kD * kH * kW;
    const int64_t N_gemm = D_out * H_out * W_out;
    const int64_t col_size = K * N_gemm;

    if (grad_output.dtype() == DType::Float32) {
        float* grad_input_ptr = get_data_ptr_conv3d<float>(grad_input);
        const float* weight_ptr = get_data_ptr_conv3d<const float>(weight);
        const float* grad_output_ptr = get_data_ptr_conv3d<const float>(grad_output);

        queue.fill(grad_input_ptr, 0.0f, N * C_in * D_in * H_in * W_in).wait();

        Tensor col_buffer({col_size}, grad_output.dtype(), grad_output.device());
        float* col_ptr = get_data_ptr_conv3d<float>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            float* grad_in_batch = grad_input_ptr + n * C_in * D_in * H_in * W_in;

            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * K;

                const float* grad_out_group = grad_output_ptr + n * C_out * N_gemm + out_channel_offset * N_gemm;
                const float* weight_group = weight_ptr + weight_offset;

                // GEMM: col = weight^T @ grad_output
                const int64_t M_group = K;
                queue.parallel_for<Conv3dGroupedBackwardInputGemmKernel>(sycl::range<2>(M_group, N_gemm),
                                 [=](sycl::id<2> idx) {
                    const int64_t k = idx[0];
                    const int64_t hw = idx[1];
                    float sum = 0.0f;
                    for (int64_t oc = 0; oc < C_out_per_group; ++oc) {
                        sum += weight_group[oc * M_group + k] * grad_out_group[oc * N_gemm + hw];
                    }
                    col_ptr[k * N_gemm + hw] = sum;
                }).wait();

                col3im_grouped_3d_kernel(
                    col_ptr, C_in, C_in_per_group, in_channel_offset,
                    D_in, H_in, W_in, kD, kH, kW,
                    stride_d, stride_h, stride_w,
                    pad_d, pad_h, pad_w,
                    dil_d, dil_h, dil_w,
                    D_out, H_out, W_out,
                    grad_in_batch, queue
                );
            }
        }
    } else if (grad_output.dtype() == DType::Float64) {
        double* grad_input_ptr = get_data_ptr_conv3d<double>(grad_input);
        const double* weight_ptr = get_data_ptr_conv3d<const double>(weight);
        const double* grad_output_ptr = get_data_ptr_conv3d<const double>(grad_output);

        queue.fill(grad_input_ptr, 0.0, N * C_in * D_in * H_in * W_in).wait();

        Tensor col_buffer({col_size}, grad_output.dtype(), grad_output.device());
        double* col_ptr = get_data_ptr_conv3d<double>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            double* grad_in_batch = grad_input_ptr + n * C_in * D_in * H_in * W_in;

            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * K;

                const double* grad_out_group = grad_output_ptr + n * C_out * N_gemm + out_channel_offset * N_gemm;
                const double* weight_group = weight_ptr + weight_offset;

                const int64_t M_group = K;
                queue.parallel_for<Conv3dBackwardInputGemmKernelFloat64>(sycl::range<2>(M_group, N_gemm),
                                 [=](sycl::id<2> idx) {
                    const int64_t k = idx[0];
                    const int64_t hw = idx[1];
                    double sum = 0.0;
                    for (int64_t oc = 0; oc < C_out_per_group; ++oc) {
                        sum += weight_group[oc * M_group + k] * grad_out_group[oc * N_gemm + hw];
                    }
                    col_ptr[k * N_gemm + hw] = sum;
                }).wait();

                col3im_grouped_3d_kernel(
                    col_ptr, C_in, C_in_per_group, in_channel_offset,
                    D_in, H_in, W_in, kD, kH, kW,
                    stride_d, stride_h, stride_w,
                    pad_d, pad_h, pad_w,
                    dil_d, dil_h, dil_w,
                    D_out, H_out, W_out,
                    grad_in_batch, queue
                );
            }
        }
    } else if (grad_output.dtype() == DType::Float16) {
        sycl::half* grad_input_ptr = get_data_ptr_conv3d<sycl::half>(grad_input);
        const sycl::half* weight_ptr = get_data_ptr_conv3d<const sycl::half>(weight);
        const sycl::half* grad_output_ptr = get_data_ptr_conv3d<const sycl::half>(grad_output);

        queue.parallel_for(sycl::range<1>(N * C_in * D_in * H_in * W_in), [=](sycl::id<1> i) {
            grad_input_ptr[i] = sycl::half(0.0f);
        }).wait();

        Tensor col_buffer({col_size}, grad_output.dtype(), grad_output.device());
        sycl::half* col_ptr = get_data_ptr_conv3d<sycl::half>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            sycl::half* grad_in_batch = grad_input_ptr + n * C_in * D_in * H_in * W_in;

            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * K;

                const sycl::half* grad_out_group = grad_output_ptr + n * C_out * N_gemm + out_channel_offset * N_gemm;
                const sycl::half* weight_group = weight_ptr + weight_offset;

                const int64_t M_group = K;
                queue.parallel_for<Conv3dBackwardInputGemmKernelFloat16>(sycl::range<2>(M_group, N_gemm),
                                 [=](sycl::id<2> idx) {
                    const int64_t k = idx[0];
                    const int64_t hw = idx[1];
                    float sum = 0.0f;
                    for (int64_t oc = 0; oc < C_out_per_group; ++oc) {
                        sum += static_cast<float>(weight_group[oc * M_group + k]) *
                               static_cast<float>(grad_out_group[oc * N_gemm + hw]);
                    }
                    col_ptr[k * N_gemm + hw] = saturate_to_half_conv3d(sum);
                }).wait();

                col3im_grouped_3d_kernel(
                    col_ptr, C_in, C_in_per_group, in_channel_offset,
                    D_in, H_in, W_in, kD, kH, kW,
                    stride_d, stride_h, stride_w,
                    pad_d, pad_h, pad_w,
                    dil_d, dil_h, dil_w,
                    D_out, H_out, W_out,
                    grad_in_batch, queue
                );
            }
        }
    } else {
        throw std::runtime_error("Unsupported dtype for conv3d_backward_input");
    }

    return grad_input;
}

// ============================================================================
// Conv3d Backward Weight (im3col-based fallback)
// ============================================================================
auto conv3d_backward_weight(const Tensor& grad_output, const Tensor& input,
                             const std::vector<int64_t>& weight_shape,
                             const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& dilation, int64_t groups,
                             sycl::queue& queue) -> Tensor {
    auto input_shape = input.shape();
    auto grad_shape = grad_output.shape();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t D_in = input_shape[2];
    const int64_t H_in = input_shape[3];
    const int64_t W_in = input_shape[4];

    const int64_t C_out = weight_shape[0];
    const int64_t C_in_per_group = weight_shape[1];
    const int64_t kD = weight_shape[2];
    const int64_t kH = weight_shape[3];
    const int64_t kW = weight_shape[4];

    const int64_t C_out_per_group = C_out / groups;

    const int64_t D_out = grad_shape[2];
    const int64_t H_out = grad_shape[3];
    const int64_t W_out = grad_shape[4];

    const int64_t stride_d = stride.size() > 0 ? stride[0] : 1;
    const int64_t stride_h = stride.size() > 1 ? stride[1] : stride_d;
    const int64_t stride_w = stride.size() > 2 ? stride[2] : stride_h;

    const int64_t pad_d = padding.size() > 0 ? padding[0] : 0;
    const int64_t pad_h = padding.size() > 1 ? padding[1] : pad_d;
    const int64_t pad_w = padding.size() > 2 ? padding[2] : pad_h;

    const int64_t dil_d = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dil_h = dilation.size() > 1 ? dilation[1] : dil_d;
    const int64_t dil_w = dilation.size() > 2 ? dilation[2] : dil_h;

    Tensor grad_weight(weight_shape, input.dtype(), input.device());

    const int64_t K = C_in_per_group * kD * kH * kW;
    const int64_t N_spatial = D_out * H_out * W_out;
    const int64_t total_weight_size = C_out * K;
    const int64_t col_size = K * N_spatial;

    if (input.dtype() == DType::Float32) {
        float* grad_weight_ptr = get_data_ptr_conv3d<float>(grad_weight);
        const float* input_ptr = get_data_ptr_conv3d<const float>(input);
        const float* grad_output_ptr = get_data_ptr_conv3d<const float>(grad_output);

        queue.fill(grad_weight_ptr, 0.0f, total_weight_size).wait();

        Tensor col_buffer({col_size}, input.dtype(), input.device());
        float* col_ptr = get_data_ptr_conv3d<float>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            const float* input_batch = input_ptr + n * C_in * D_in * H_in * W_in;

            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * K;

                const float* grad_out_group = grad_output_ptr + n * C_out * N_spatial + out_channel_offset * N_spatial;
                float* grad_weight_group = grad_weight_ptr + weight_offset;

                im3col_grouped_3d_kernel(
                    input_batch, C_in, C_in_per_group, in_channel_offset,
                    D_in, H_in, W_in, kD, kH, kW,
                    stride_d, stride_h, stride_w,
                    pad_d, pad_h, pad_w,
                    dil_d, dil_h, dil_w,
                    D_out, H_out, W_out,
                    col_ptr, queue
                );

                queue.parallel_for<Conv3dGroupedBackwardWeightGemmKernel>(sycl::range<2>(C_out_per_group, K),
                                 [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];
                    const int64_t k = idx[1];
                    float sum = 0.0f;
                    for (int64_t hw = 0; hw < N_spatial; ++hw) {
                        sum += grad_out_group[oc * N_spatial + hw] * col_ptr[k * N_spatial + hw];
                    }
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                        atomic_val(grad_weight_group[oc * K + k]);
                    atomic_val.fetch_add(sum);
                }).wait();
            }
        }
    } else if (input.dtype() == DType::Float64) {
        double* grad_weight_ptr = get_data_ptr_conv3d<double>(grad_weight);
        const double* input_ptr = get_data_ptr_conv3d<const double>(input);
        const double* grad_output_ptr = get_data_ptr_conv3d<const double>(grad_output);

        queue.fill(grad_weight_ptr, 0.0, total_weight_size).wait();

        Tensor col_buffer({col_size}, input.dtype(), input.device());
        double* col_ptr = get_data_ptr_conv3d<double>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            const double* input_batch = input_ptr + n * C_in * D_in * H_in * W_in;

            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * K;

                const double* grad_out_group = grad_output_ptr + n * C_out * N_spatial + out_channel_offset * N_spatial;
                double* grad_weight_group = grad_weight_ptr + weight_offset;

                im3col_grouped_3d_kernel(
                    input_batch, C_in, C_in_per_group, in_channel_offset,
                    D_in, H_in, W_in, kD, kH, kW,
                    stride_d, stride_h, stride_w,
                    pad_d, pad_h, pad_w,
                    dil_d, dil_h, dil_w,
                    D_out, H_out, W_out,
                    col_ptr, queue
                );

                queue.parallel_for<Conv3dBackwardWeightGemmKernelFloat64>(sycl::range<2>(C_out_per_group, K),
                                 [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];
                    const int64_t k = idx[1];
                    double sum = 0.0;
                    for (int64_t hw = 0; hw < N_spatial; ++hw) {
                        sum += grad_out_group[oc * N_spatial + hw] * col_ptr[k * N_spatial + hw];
                    }
                    sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>
                        atomic_val(grad_weight_group[oc * K + k]);
                    atomic_val.fetch_add(sum);
                }).wait();
            }
        }
    } else if (input.dtype() == DType::Float16) {
        sycl::half* grad_weight_ptr = get_data_ptr_conv3d<sycl::half>(grad_weight);
        const sycl::half* input_ptr = get_data_ptr_conv3d<const sycl::half>(input);
        const sycl::half* grad_output_ptr = get_data_ptr_conv3d<const sycl::half>(grad_output);

        // Use float accumulation buffer
        Tensor grad_weight_float({total_weight_size}, DType::Float32, input.device());
        float* grad_weight_float_ptr = get_data_ptr_conv3d<float>(grad_weight_float);
        queue.fill(grad_weight_float_ptr, 0.0f, total_weight_size).wait();

        Tensor col_buffer({col_size}, input.dtype(), input.device());
        sycl::half* col_ptr = get_data_ptr_conv3d<sycl::half>(col_buffer);

        for (int64_t n = 0; n < N; ++n) {
            const sycl::half* input_batch = input_ptr + n * C_in * D_in * H_in * W_in;

            for (int64_t g = 0; g < groups; ++g) {
                const int64_t in_channel_offset = g * C_in_per_group;
                const int64_t out_channel_offset = g * C_out_per_group;
                const int64_t weight_offset = g * C_out_per_group * K;

                const sycl::half* grad_out_group = grad_output_ptr + n * C_out * N_spatial + out_channel_offset * N_spatial;
                float* grad_weight_group = grad_weight_float_ptr + weight_offset;

                im3col_grouped_3d_kernel(
                    input_batch, C_in, C_in_per_group, in_channel_offset,
                    D_in, H_in, W_in, kD, kH, kW,
                    stride_d, stride_h, stride_w,
                    pad_d, pad_h, pad_w,
                    dil_d, dil_h, dil_w,
                    D_out, H_out, W_out,
                    col_ptr, queue
                );

                queue.parallel_for<Conv3dBackwardWeightGemmKernelFloat16>(sycl::range<2>(C_out_per_group, K),
                                 [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];
                    const int64_t k = idx[1];
                    float sum = 0.0f;
                    for (int64_t hw = 0; hw < N_spatial; ++hw) {
                        sum += static_cast<float>(grad_out_group[oc * N_spatial + hw]) *
                               static_cast<float>(col_ptr[k * N_spatial + hw]);
                    }
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                        atomic_val(grad_weight_group[oc * K + k]);
                    atomic_val.fetch_add(sum);
                }).wait();
            }
        }

        // Convert accumulated float values back to half
        queue.parallel_for(sycl::range<1>(total_weight_size), [=](sycl::id<1> i) {
            grad_weight_ptr[i] = sycl::half(grad_weight_float_ptr[i]);
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for conv3d_backward_weight");
    }

    return grad_weight;
}

// ============================================================================
// Conv3d Backward Bias (fallback)
// ============================================================================
auto conv3d_backward_bias(const Tensor& grad_output, sycl::queue& queue) -> Tensor {
    auto grad_shape = grad_output.shape();
    const int64_t N = grad_shape[0];
    const int64_t C = grad_shape[1];
    const int64_t D = grad_shape[2];
    const int64_t H = grad_shape[3];
    const int64_t W = grad_shape[4];

    Tensor grad_bias({C}, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        float* grad_bias_ptr = get_data_ptr_conv3d<float>(grad_bias);
        const float* grad_output_ptr = get_data_ptr_conv3d<const float>(grad_output);

        queue.parallel_for<Conv3dBackwardBiasKernel>(sycl::range<1>(C), [=](sycl::id<1> c) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t d = 0; d < D; ++d) {
                    for (int64_t h = 0; h < H; ++h) {
                        for (int64_t w = 0; w < W; ++w) {
                            sum += grad_output_ptr[(((n * C + c) * D + d) * H + h) * W + w];
                        }
                    }
                }
            }
            grad_bias_ptr[c] = sum;
        }).wait();
    } else if (grad_output.dtype() == DType::Float64) {
        double* grad_bias_ptr = get_data_ptr_conv3d<double>(grad_bias);
        const double* grad_output_ptr = get_data_ptr_conv3d<const double>(grad_output);

        queue.parallel_for<Conv3dBackwardBiasKernelFloat64>(sycl::range<1>(C), [=](sycl::id<1> c) {
            double sum = 0.0;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t d = 0; d < D; ++d) {
                    for (int64_t h = 0; h < H; ++h) {
                        for (int64_t w = 0; w < W; ++w) {
                            sum += grad_output_ptr[(((n * C + c) * D + d) * H + h) * W + w];
                        }
                    }
                }
            }
            grad_bias_ptr[c] = sum;
        }).wait();
    } else if (grad_output.dtype() == DType::Float16) {
        sycl::half* grad_bias_ptr = get_data_ptr_conv3d<sycl::half>(grad_bias);
        const sycl::half* grad_output_ptr = get_data_ptr_conv3d<const sycl::half>(grad_output);

        queue.parallel_for<Conv3dBackwardBiasKernelFloat16>(sycl::range<1>(C), [=](sycl::id<1> c) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t d = 0; d < D; ++d) {
                    for (int64_t h = 0; h < H; ++h) {
                        for (int64_t w = 0; w < W; ++w) {
                            sum += static_cast<float>(grad_output_ptr[(((n * C + c) * D + d) * H + h) * W + w]);
                        }
                    }
                }
            }
            grad_bias_ptr[c] = saturate_to_half_conv3d(sum);
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for conv3d_backward_bias");
    }

    return grad_bias;
}

#endif // TENZOR_HAS_ONEDNN

// ============================================================================
// ConvTranspose3d Forward (gather approach, no oneDNN dependency)
// ============================================================================
auto conv_transpose3d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                               const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& output_padding,
                               const std::vector<int64_t>& dilation, int64_t groups,
                               sycl::queue& queue) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t D_in = input_shape[2];
    const int64_t H_in = input_shape[3];
    const int64_t W_in = input_shape[4];

    const int64_t C_out = weight_shape[1];
    const int64_t kD = weight_shape[2];
    const int64_t kH = weight_shape[3];
    const int64_t kW = weight_shape[4];

    const int64_t stride_d = stride.size() > 0 ? stride[0] : 1;
    const int64_t stride_h = stride.size() > 1 ? stride[1] : stride_d;
    const int64_t stride_w = stride.size() > 2 ? stride[2] : stride_h;

    const int64_t pad_d = padding.size() > 0 ? padding[0] : 0;
    const int64_t pad_h = padding.size() > 1 ? padding[1] : pad_d;
    const int64_t pad_w = padding.size() > 2 ? padding[2] : pad_h;

    const int64_t out_pad_d = output_padding.size() > 0 ? output_padding[0] : 0;
    const int64_t out_pad_h = output_padding.size() > 1 ? output_padding[1] : out_pad_d;
    const int64_t out_pad_w = output_padding.size() > 2 ? output_padding[2] : out_pad_h;

    const int64_t dil_d = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dil_h = dilation.size() > 1 ? dilation[1] : dil_d;
    const int64_t dil_w = dilation.size() > 2 ? dilation[2] : dil_h;

    const int64_t D_out = (D_in - 1) * stride_d - 2 * pad_d + dil_d * (kD - 1) + out_pad_d + 1;
    const int64_t H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + out_pad_h + 1;
    const int64_t W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + out_pad_w + 1;

    Tensor output({N, C_out, D_out, H_out, W_out}, input.dtype(), input.device());
    const int64_t total_elements = output.numel();

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr_conv3d<const float>(input);
        const float* weight_ptr = get_data_ptr_conv3d<const float>(weight);
        const float* bias_ptr = (bias != nullptr) ? get_data_ptr_conv3d<const float>(*bias) : nullptr;
        float* output_ptr = get_data_ptr_conv3d<float>(output);

        queue.parallel_for<ConvTranspose3dForwardKernelFloat32>(sycl::range<1>(total_elements), [=](sycl::id<1> idx) {
            int64_t tmp = idx;
            int64_t ow = tmp % W_out; tmp /= W_out;
            int64_t oh = tmp % H_out; tmp /= H_out;
            int64_t od = tmp % D_out; tmp /= D_out;
            int64_t oc = tmp % C_out; tmp /= C_out;
            int64_t n = tmp;

            float sum = bias_ptr ? bias_ptr[oc] : 0.0f;

            for (int64_t ic = 0; ic < C_in; ++ic) {
                for (int64_t kd = 0; kd < kD; ++kd) {
                    for (int64_t kh = 0; kh < kH; ++kh) {
                        for (int64_t kw_iter = 0; kw_iter < kW; ++kw_iter) {
                            int64_t d_off = od + pad_d - kd;
                            int64_t h_off = oh + pad_h - kh;
                            int64_t w_off = ow + pad_w - kw_iter;
                            if (d_off % stride_d != 0 || h_off % stride_h != 0 || w_off % stride_w != 0) continue;
                            int64_t id = d_off / stride_d;
                            int64_t ih = h_off / stride_h;
                            int64_t iw = w_off / stride_w;
                            if (id >= 0 && id < D_in && ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                int64_t input_idx = n * (C_in * D_in * H_in * W_in) +
                                                   ic * (D_in * H_in * W_in) + id * (H_in * W_in) + ih * W_in + iw;
                                int64_t weight_idx = ic * (C_out * kD * kH * kW) +
                                                    oc * (kD * kH * kW) + kd * (kH * kW) + kh * kW + kw_iter;
                                sum += input_ptr[input_idx] * weight_ptr[weight_idx];
                            }
                        }
                    }
                }
            }
            output_ptr[idx] = sum;
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr_conv3d<const double>(input);
        const double* weight_ptr = get_data_ptr_conv3d<const double>(weight);
        const double* bias_ptr = (bias != nullptr) ? get_data_ptr_conv3d<const double>(*bias) : nullptr;
        double* output_ptr = get_data_ptr_conv3d<double>(output);

        queue.parallel_for<ConvTranspose3dForwardKernelFloat64>(sycl::range<1>(total_elements), [=](sycl::id<1> idx) {
            int64_t tmp = idx;
            int64_t ow = tmp % W_out; tmp /= W_out;
            int64_t oh = tmp % H_out; tmp /= H_out;
            int64_t od = tmp % D_out; tmp /= D_out;
            int64_t oc = tmp % C_out; tmp /= C_out;
            int64_t n = tmp;

            double sum = bias_ptr ? bias_ptr[oc] : 0.0;

            for (int64_t ic = 0; ic < C_in; ++ic) {
                for (int64_t kd = 0; kd < kD; ++kd) {
                    for (int64_t kh = 0; kh < kH; ++kh) {
                        for (int64_t kw_iter = 0; kw_iter < kW; ++kw_iter) {
                            int64_t d_off = od + pad_d - kd;
                            int64_t h_off = oh + pad_h - kh;
                            int64_t w_off = ow + pad_w - kw_iter;
                            if (d_off % stride_d != 0 || h_off % stride_h != 0 || w_off % stride_w != 0) continue;
                            int64_t id = d_off / stride_d;
                            int64_t ih = h_off / stride_h;
                            int64_t iw = w_off / stride_w;
                            if (id >= 0 && id < D_in && ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                int64_t input_idx = n * (C_in * D_in * H_in * W_in) +
                                                   ic * (D_in * H_in * W_in) + id * (H_in * W_in) + ih * W_in + iw;
                                int64_t weight_idx = ic * (C_out * kD * kH * kW) +
                                                    oc * (kD * kH * kW) + kd * (kH * kW) + kh * kW + kw_iter;
                                sum += input_ptr[input_idx] * weight_ptr[weight_idx];
                            }
                        }
                    }
                }
            }
            output_ptr[idx] = sum;
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* input_ptr = get_data_ptr_conv3d<const sycl::half>(input);
        const sycl::half* weight_ptr = get_data_ptr_conv3d<const sycl::half>(weight);
        const sycl::half* bias_ptr = (bias != nullptr) ? get_data_ptr_conv3d<const sycl::half>(*bias) : nullptr;
        sycl::half* output_ptr = get_data_ptr_conv3d<sycl::half>(output);

        queue.parallel_for<ConvTranspose3dForwardKernelFloat16>(sycl::range<1>(total_elements), [=](sycl::id<1> idx) {
            int64_t tmp = idx;
            int64_t ow = tmp % W_out; tmp /= W_out;
            int64_t oh = tmp % H_out; tmp /= H_out;
            int64_t od = tmp % D_out; tmp /= D_out;
            int64_t oc = tmp % C_out; tmp /= C_out;
            int64_t n = tmp;

            float sum = bias_ptr ? static_cast<float>(bias_ptr[oc]) : 0.0f;

            for (int64_t ic = 0; ic < C_in; ++ic) {
                for (int64_t kd = 0; kd < kD; ++kd) {
                    for (int64_t kh = 0; kh < kH; ++kh) {
                        for (int64_t kw_iter = 0; kw_iter < kW; ++kw_iter) {
                            int64_t d_off = od + pad_d - kd;
                            int64_t h_off = oh + pad_h - kh;
                            int64_t w_off = ow + pad_w - kw_iter;
                            if (d_off % stride_d != 0 || h_off % stride_h != 0 || w_off % stride_w != 0) continue;
                            int64_t id = d_off / stride_d;
                            int64_t ih = h_off / stride_h;
                            int64_t iw = w_off / stride_w;
                            if (id >= 0 && id < D_in && ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                int64_t input_idx = n * (C_in * D_in * H_in * W_in) +
                                                   ic * (D_in * H_in * W_in) + id * (H_in * W_in) + ih * W_in + iw;
                                int64_t weight_idx = ic * (C_out * kD * kH * kW) +
                                                    oc * (kD * kH * kW) + kd * (kH * kW) + kh * kW + kw_iter;
                                sum += static_cast<float>(input_ptr[input_idx]) * static_cast<float>(weight_ptr[weight_idx]);
                            }
                        }
                    }
                }
            }
            output_ptr[idx] = saturate_to_half_conv3d(sum);
        }).wait();
    } else {
        throw std::runtime_error("ConvTranspose3d forward: unsupported dtype");
    }

    return output;
}

// =========================================================================
// ConvTranspose3d backward operations
// =========================================================================

// ConvTranspose3d backward input: this is a regular conv3d forward
auto conv_transpose3d_backward_input(const Tensor& grad_output, const Tensor& weight,
                                      const std::vector<int64_t>& input_shape,
                                      const std::vector<int64_t>& stride,
                                      const std::vector<int64_t>& padding,
                                      const std::vector<int64_t>& dilation,
                                      int64_t groups, sycl::queue& queue) -> Tensor {
    // ConvTranspose3d backward w.r.t. input is a regular Conv3d forward
    // with weight treating C_in as output channels
    return conv3d_forward(grad_output, weight, nullptr, stride, padding, dilation, groups, queue);
}

// ConvTranspose3d backward weight: delegates to conv3d backward weight
auto conv_transpose3d_backward_weight(const Tensor& grad_output, const Tensor& input,
                                       const std::vector<int64_t>& weight_shape,
                                       const std::vector<int64_t>& stride,
                                       const std::vector<int64_t>& padding,
                                       const std::vector<int64_t>& dilation,
                                       int64_t groups, sycl::queue& queue) -> Tensor {
    return conv3d_backward_weight(grad_output, input, weight_shape, stride, padding, dilation, groups, queue);
}

// ConvTranspose3d backward bias: same as conv3d backward bias
auto conv_transpose3d_backward_bias(const Tensor& grad_output, sycl::queue& queue) -> Tensor {
    return conv3d_backward_bias(grad_output, queue);
}

} // namespace oneapi
} // namespace tenzor
