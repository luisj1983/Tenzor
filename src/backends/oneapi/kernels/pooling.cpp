#include "tenzor/core/tensor.hpp"
#include "oneapi_kernel_utils.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <sycl/sycl.hpp>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef TENZOR_HAS_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#endif

namespace tenzor {
namespace oneapi {

// Kernel class declarations for SYCL
class MaxPool2dKernelFloat32;
class MaxPool2dKernelFloat64;
class MaxPool2dWithIndicesKernelFloat32;
class MaxPool2dWithIndicesKernelFloat64;
class MaxPool2dWithIndicesKernelFloat16;
class AvgPool2dKernelFloat32;
class AvgPool2dKernelFloat64;
class AvgPool2dKernelFloat16;
class AdaptiveAvgPool2dKernelFloat32;
class AdaptiveAvgPool2dKernelFloat64;
class AdaptiveAvgPool2dKernelFloat16;
class AdaptiveMaxPool2dKernelFloat32;
class AdaptiveMaxPool2dKernelFloat64;
class AdaptiveMaxPool2dKernelFloat16;
class AdaptiveAvgPool2dBackwardKernelFloat32;
class AdaptiveAvgPool2dBackwardKernelFloat64;
class AdaptiveAvgPool2dBackwardKernelFloat16;
class AvgPool2dBackwardKernelFloat32;
class AvgPool2dBackwardKernelFloat64;
class AvgPool2dBackwardKernelFloat16;
class MaxPool2dBackwardWithIndicesKernelFloat32;
class MaxPool2dBackwardWithIndicesKernelFloat64;
class MaxPool2dBackwardWithIndicesKernelFloat16;
class MaxPool2dWithIndicesKernelBFloat16;
class AvgPool2dKernelBFloat16;
class AdaptiveAvgPool2dKernelBFloat16;
class AdaptiveMaxPool2dKernelBFloat16;
class AdaptiveAvgPool2dBackwardKernelBFloat16;
class AvgPool2dBackwardKernelBFloat16;
class MaxPool2dBackwardWithIndicesKernelBFloat16;
class AdaptiveMaxPool2dBackwardKernelFloat32;
class AdaptiveMaxPool2dBackwardKernelFloat64;
class AdaptiveMaxPool2dBackwardKernelFloat16;
class AdaptiveMaxPool2dBackwardKernelBFloat16;

// 1D pooling kernel name classes
class MaxPool1dForwardFloat32 {};
class MaxPool1dForwardFloat64 {};
class MaxPool1dForwardFloat16 {};
class MaxPool1dForwardBFloat16 {};
class MaxPool1dBackwardFloat32 {};
class MaxPool1dBackwardFloat64 {};
class MaxPool1dBackwardFloat16 {};
class MaxPool1dBackwardBFloat16 {};
class AvgPool1dForwardFloat32 {};
class AvgPool1dForwardFloat64 {};
class AvgPool1dForwardFloat16 {};
class AvgPool1dForwardBFloat16 {};
class AvgPool1dBackwardFloat32 {};
class AvgPool1dBackwardFloat64 {};
class AvgPool1dBackwardFloat16 {};
class AvgPool1dBackwardBFloat16 {};
class AdaptiveMaxPool1dForwardFloat32 {};
class AdaptiveMaxPool1dForwardFloat64 {};
class AdaptiveMaxPool1dForwardFloat16 {};
class AdaptiveMaxPool1dForwardBFloat16 {};
class AdaptiveMaxPool1dBackwardFloat32 {};
class AdaptiveMaxPool1dBackwardFloat64 {};
class AdaptiveMaxPool1dBackwardFloat16 {};
class AdaptiveMaxPool1dBackwardBFloat16 {};
class AdaptiveAvgPool1dForwardFloat32 {};
class AdaptiveAvgPool1dForwardFloat64 {};
class AdaptiveAvgPool1dForwardFloat16 {};
class AdaptiveAvgPool1dForwardBFloat16 {};
class AdaptiveAvgPool1dBackwardFloat32 {};
class AdaptiveAvgPool1dBackwardFloat64 {};
class AdaptiveAvgPool1dBackwardFloat16 {};
class AdaptiveAvgPool1dBackwardBFloat16 {};

// 3D pooling kernel name classes
class MaxPool3dForwardFloat32 {};
class MaxPool3dForwardFloat64 {};
class MaxPool3dForwardFloat16 {};
class MaxPool3dForwardBFloat16 {};
class MaxPool3dBackwardFloat32 {};
class MaxPool3dBackwardFloat64 {};
class MaxPool3dBackwardFloat16 {};
class MaxPool3dBackwardBFloat16 {};
class AvgPool3dForwardFloat32 {};
class AvgPool3dForwardFloat64 {};
class AvgPool3dForwardFloat16 {};
class AvgPool3dForwardBFloat16 {};
class AvgPool3dBackwardFloat32 {};
class AvgPool3dBackwardFloat64 {};
class AvgPool3dBackwardFloat16 {};
class AvgPool3dBackwardBFloat16 {};
class AdaptiveMaxPool3dForwardFloat32 {};
class AdaptiveMaxPool3dForwardFloat64 {};
class AdaptiveMaxPool3dForwardFloat16 {};
class AdaptiveMaxPool3dForwardBFloat16 {};
class AdaptiveMaxPool3dBackwardFloat32 {};
class AdaptiveMaxPool3dBackwardFloat64 {};
class AdaptiveMaxPool3dBackwardFloat16 {};
class AdaptiveMaxPool3dBackwardBFloat16 {};
class AdaptiveAvgPool3dForwardFloat32 {};
class AdaptiveAvgPool3dForwardFloat64 {};
class AdaptiveAvgPool3dForwardFloat16 {};
class AdaptiveAvgPool3dForwardBFloat16 {};
class AdaptiveAvgPool3dBackwardFloat32 {};
class AdaptiveAvgPool3dBackwardFloat64 {};
class AdaptiveAvgPool3dBackwardFloat16 {};
class AdaptiveAvgPool3dBackwardBFloat16 {};



#ifdef TENZOR_HAS_ONEDNN

// Helper: map Tenzor DType to oneDNN memory data type (pooling-local)
static auto to_dnnl_dtype(DType dt) -> dnnl::memory::data_type {
    switch (dt) {
        case DType::Float32:  return dnnl::memory::data_type::f32;
        case DType::Float64:  return dnnl::memory::data_type::f64;
        case DType::BFloat16: return dnnl::memory::data_type::bf16;
        case DType::Float16:  return dnnl::memory::data_type::f16;
        default: throw std::runtime_error("Unsupported dtype for oneDNN pooling");
    }
}

// Forward declaration: pure SYCL maxpool implementation for non-Float32 dtypes
static auto maxpool2d_forward_with_indices_sycl(const Tensor& input,
                                                 int64_t kernel_h, int64_t kernel_w,
                                                 int64_t stride_h, int64_t stride_w,
                                                 int64_t padding_h, int64_t padding_w,
                                                 int64_t dilation_h, int64_t dilation_w,
                                                 sycl::queue& queue) -> std::pair<Tensor, Tensor>;

// MaxPool2d forward with indices using oneDNN - returns both output and indices for backward pass
auto maxpool2d_forward_with_indices(const Tensor& input,
                                    int64_t kernel_h, int64_t kernel_w,
                                    int64_t stride_h, int64_t stride_w,
                                    int64_t padding_h, int64_t padding_w,
                                    int64_t dilation_h, int64_t dilation_w,
                                    sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    using namespace dnnl;

    // oneDNN pooling only reliably supports f32 on all devices;
    // for other dtypes use the pure SYCL implementation to avoid
    // buffer-size mismatches between the tensor and the descriptor.
    if (input.dtype() != DType::Float32) {
        return maxpool2d_forward_with_indices_sycl(input,
            kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, queue);
    }

    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("MaxPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];

    const int64_t H_out = (H_in + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t W_out = (W_in + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());
    // Indices must be Int64 to correctly represent element positions for all tensor sizes.
    Tensor indices({N, C, H_out, W_out}, DType::Int64, input.device());

    // Create oneDNN engine and stream
    auto dnnl_engine = sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto dnnl_stream = sycl_interop::make_stream(dnnl_engine, queue);

    // Memory descriptors
    memory::dims src_dims = {N, C, H_in, W_in};
    memory::dims dst_dims = {N, C, H_out, W_out};
    memory::dims kernel_dims = {kernel_h, kernel_w};
    memory::dims strides_dims = {stride_h, stride_w};
    memory::dims padding_dims = {padding_h, padding_w};
    memory::dims dilation_dims = {dilation_h - 1, dilation_w - 1};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    // oneDNN v3 API: primitive_desc built directly from engine + parameters
    // (the legacy ::desc step was removed in oneDNN 3.x). forward_training keeps
    // the workspace (max indices).
    auto pool_pd = pooling_forward::primitive_desc(
        dnnl_engine,
        prop_kind::forward_training,
        algorithm::pooling_max,
        src_md, dst_md,
        strides_dims, kernel_dims,
        dilation_dims,
        padding_dims, padding_dims
    );

    // Wrap tensors
    auto src_mem = sycl_interop::make_memory(pool_pd.src_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(input.data_ptr()));

    auto dst_mem = sycl_interop::make_memory(pool_pd.dst_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(output.data_ptr()));

    // Create workspace for indices (oneDNN stores max indices in workspace)
    auto workspace_mem = memory(pool_pd.workspace_desc(), dnnl_engine);

    // Execute
    auto pool_prim = pooling_forward(pool_pd);
    pool_prim.execute(dnnl_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem},
        {DNNL_ARG_WORKSPACE, workspace_mem}
    });

    dnnl_stream.wait();

    // oneDNN workspace format is internal, so we need to compute indices manually
    // for compatibility with our backward pass that expects linear indices
    const float* in_ptr = get_data_ptr<const float>(input);
    int64_t* idx_ptr = get_data_ptr<int64_t>(indices);

    const int64_t total_size = N * C * H_out * W_out;
    queue.parallel_for(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
        int64_t temp = flat_idx;
        const int64_t w_out = temp % W_out;
        temp /= W_out;
        const int64_t h_out = temp % H_out;
        temp /= H_out;
        const int64_t c = temp % C;
        const int64_t n = temp / C;

        float max_val = -3.4028235e+38f;
        int64_t max_idx = 0;

        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                int64_t h_in = h_out * stride_h - padding_h + kh * dilation_h;
                int64_t w_in = w_out * stride_w - padding_w + kw * dilation_w;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                    float val = in_ptr[input_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = input_idx;
                    }
                }
            }
        }

        idx_ptr[((n * C + c) * H_out + h_out) * W_out + w_out] = max_idx;
    });

    return {output, indices};
}

// MaxPool2d forward using oneDNN - returns only output
auto maxpool2d_forward(const Tensor& input,
                       int64_t kernel_h, int64_t kernel_w,
                       int64_t stride_h, int64_t stride_w,
                       int64_t padding_h, int64_t padding_w,
                       int64_t dilation_h, int64_t dilation_w,
                       sycl::queue& queue) -> Tensor {
    auto [output, indices] = maxpool2d_forward_with_indices(input,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, queue);
    return output;
}

// AvgPool2d forward using oneDNN
auto avgpool2d_forward(const Tensor& input,
                       int64_t kernel_h, int64_t kernel_w,
                       int64_t stride_h, int64_t stride_w,
                       int64_t padding_h, int64_t padding_w,
                       bool count_include_pad, sycl::queue& queue) -> Tensor {
    using namespace dnnl;

    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AvgPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];

    const int64_t H_out = (H_in + 2 * padding_h - kernel_h) / stride_h + 1;
    const int64_t W_out = (W_in + 2 * padding_w - kernel_w) / stride_w + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());

    // Create oneDNN engine and stream
    auto dnnl_engine = sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto dnnl_stream = sycl_interop::make_stream(dnnl_engine, queue);

    // Memory descriptors
    memory::dims src_dims = {N, C, H_in, W_in};
    memory::dims dst_dims = {N, C, H_out, W_out};
    memory::dims kernel_dims = {kernel_h, kernel_w};
    memory::dims strides_dims = {stride_h, stride_w};
    memory::dims padding_dims = {padding_h, padding_w};

    auto dt = to_dnnl_dtype(input.dtype());
    auto src_md = memory::desc(src_dims, dt, memory::format_tag::nchw);
    auto dst_md = memory::desc(dst_dims, dt, memory::format_tag::nchw);

    // Choose algorithm based on count_include_pad
    auto algo = count_include_pad ?
        algorithm::pooling_avg_include_padding :
        algorithm::pooling_avg_exclude_padding;

    // oneDNN v3 API: primitive_desc built directly from engine + parameters.
    // v3 requires an explicit dilation argument; average pooling is undilated
    // (oneDNN encodes dilation as value-1, so 0 == standard pooling).
    memory::dims pool_dilation = {0, 0};
    auto pool_pd = pooling_forward::primitive_desc(
        dnnl_engine,
        prop_kind::forward_inference,
        algo,
        src_md, dst_md,
        strides_dims, kernel_dims,
        pool_dilation,
        padding_dims, padding_dims
    );

    // Wrap tensors
    auto src_mem = sycl_interop::make_memory(pool_pd.src_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(input.data_ptr()));

    auto dst_mem = sycl_interop::make_memory(pool_pd.dst_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(output.data_ptr()));

    // Execute
    auto pool_prim = pooling_forward(pool_pd);
    pool_prim.execute(dnnl_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem}
    });

    dnnl_stream.wait();

    return output;
}

// Native SYCL FP64 average-pool forward. oneDNN has no FP64 pooling primitive,
// and widening to FP32 introduces rounding that breaks FP64 finite-difference
// gradcheck. Computed directly in double.
static auto avgpool2d_forward_f64_sycl(const Tensor& input,
                                       int64_t kernel_h, int64_t kernel_w,
                                       int64_t stride_h, int64_t stride_w,
                                       int64_t padding_h, int64_t padding_w,
                                       bool count_include_pad, sycl::queue& queue) -> Tensor {
    const auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AvgPool2d requires 4D input (N, C, H, W)");
    }
    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];
    const int64_t H_out = (H_in + 2 * padding_h - kernel_h) / stride_h + 1;
    const int64_t W_out = (W_in + 2 * padding_w - kernel_w) / stride_w + 1;

    Tensor output({N, C, H_out, W_out}, DType::Float64, input.device());
    const double* in_ptr = get_data_ptr<const double>(input);
    double* out_ptr = get_data_ptr<double>(output);

    const int64_t total_size = N * C * H_out * W_out;
    queue.parallel_for<AvgPool2dKernelFloat64>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
        int64_t temp = flat_idx;
        const int64_t w_out = temp % W_out;
        temp /= W_out;
        const int64_t h_out = temp % H_out;
        temp /= H_out;
        const int64_t c = temp % C;
        const int64_t n = temp / C;

        double sum = 0.0;
        int64_t count = 0;
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                int64_t h_in = h_out * stride_h - padding_h + kh;
                int64_t w_in = w_out * stride_w - padding_w + kw;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    sum += in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in];
                    count++;
                } else if (count_include_pad) {
                    count++;
                }
            }
        }
        out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out] =
            count > 0 ? sum / static_cast<double>(count) : 0.0;
    });
    return output;
}

// Pure SYCL maxpool implementation used for non-Float32 dtypes when oneDNN is available
static auto maxpool2d_forward_with_indices_sycl(const Tensor& input,
                                                 int64_t kernel_h, int64_t kernel_w,
                                                 int64_t stride_h, int64_t stride_w,
                                                 int64_t padding_h, int64_t padding_w,
                                                 int64_t dilation_h, int64_t dilation_w,
                                                 sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("MaxPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];

    const int64_t H_out = (H_in + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t W_out = (W_in + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());
    // Indices must be Int64 to correctly represent element positions for all tensor sizes.
    // Using the input's dtype (e.g. Float16) would silently truncate large indices,
    // causing out-of-bounds access in the backward pass.
    Tensor indices({N, C, H_out, W_out}, DType::Int64, input.device());

    if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dWithIndicesKernelFloat64>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            double max_val = -1.7976931348623157e+308;
            int64_t max_idx = 0;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride_h - padding_h + kh * dilation_h;
                    int64_t w_in = w_out * stride_w - padding_w + kw * dilation_w;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                        double val = in_ptr[input_idx];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = input_idx;
                        }
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
            out_ptr[out_idx] = max_val;
            idx_ptr[out_idx] = max_idx;
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dWithIndicesKernelFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            float max_val = -3.4028235e+38f;
            int64_t max_idx = 0;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride_h - padding_h + kh * dilation_h;
                    int64_t w_in = w_out * stride_w - padding_w + kw * dilation_w;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                        float val = static_cast<float>(in_ptr[input_idx]);
                        if (val > max_val) {
                            max_val = val;
                            max_idx = input_idx;
                        }
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
            out_ptr[out_idx] = sycl::half(max_val);
            idx_ptr[out_idx] = max_idx;
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dWithIndicesKernelBFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            float max_val = -3.4028235e+38f;
            int64_t max_idx = 0;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride_h - padding_h + kh * dilation_h;
                    int64_t w_in = w_out * stride_w - padding_w + kw * dilation_w;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                        float val = bf16_to_f32(in_ptr[input_idx]);
                        if (val > max_val) {
                            max_val = val;
                            max_idx = input_idx;
                        }
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
            out_ptr[out_idx] = f32_to_bf16(max_val);
            idx_ptr[out_idx] = max_idx;
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for maxpool2d_forward_with_indices");
    }

    return {output, indices};
}

#else // !TENZOR_HAS_ONEDNN - Pure SYCL implementation (slower fallback)

// MaxPool2d forward with indices (pure SYCL) - returns both output and indices for backward pass
auto maxpool2d_forward_with_indices(const Tensor& input,
                                    int64_t kernel_h, int64_t kernel_w,
                                    int64_t stride_h, int64_t stride_w,
                                    int64_t padding_h, int64_t padding_w,
                                    int64_t dilation_h, int64_t dilation_w,
                                    sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("MaxPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];

    const int64_t H_out = (H_in + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t W_out = (W_in + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());
    // Indices must be Int64 to correctly represent element positions for all tensor sizes.
    Tensor indices({N, C, H_out, W_out}, DType::Int64, input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dWithIndicesKernelFloat32>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            float max_val = -3.4028235e+38f;
            int64_t max_idx = 0;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride_h - padding_h + kh * dilation_h;
                    int64_t w_in = w_out * stride_w - padding_w + kw * dilation_w;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                        float val = in_ptr[input_idx];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = input_idx;
                        }
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
            out_ptr[out_idx] = max_val;
            idx_ptr[out_idx] = max_idx;
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dWithIndicesKernelFloat64>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            double max_val = -1.7976931348623157e+308;
            int64_t max_idx = 0;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride_h - padding_h + kh * dilation_h;
                    int64_t w_in = w_out * stride_w - padding_w + kw * dilation_w;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                        double val = in_ptr[input_idx];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = input_idx;
                        }
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
            out_ptr[out_idx] = max_val;
            idx_ptr[out_idx] = max_idx;
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dWithIndicesKernelFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Use float for max computation for numerical stability
            float max_val = -3.4028235e+38f;
            int64_t max_idx = 0;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride_h - padding_h + kh * dilation_h;
                    int64_t w_in = w_out * stride_w - padding_w + kw * dilation_w;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                        float val = static_cast<float>(in_ptr[input_idx]);
                        if (val > max_val) {
                            max_val = val;
                            max_idx = input_idx;
                        }
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
            out_ptr[out_idx] = sycl::half(max_val);
            idx_ptr[out_idx] = max_idx;
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dWithIndicesKernelBFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Use float for max computation for numerical stability
            float max_val = -3.4028235e+38f;
            int64_t max_idx = 0;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride_h - padding_h + kh * dilation_h;
                    int64_t w_in = w_out * stride_w - padding_w + kw * dilation_w;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                        float val = bf16_to_f32(in_ptr[input_idx]);
                        if (val > max_val) {
                            max_val = val;
                            max_idx = input_idx;
                        }
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
            out_ptr[out_idx] = f32_to_bf16(max_val);
            idx_ptr[out_idx] = max_idx;
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for maxpool2d_forward_with_indices");
    }

    return {output, indices};
}

// MaxPool2d forward (pure SYCL) - returns only output (for operations that don't need indices)
auto maxpool2d_forward(const Tensor& input,
                       int64_t kernel_h, int64_t kernel_w,
                       int64_t stride_h, int64_t stride_w,
                       int64_t padding_h, int64_t padding_w,
                       int64_t dilation_h, int64_t dilation_w,
                       sycl::queue& queue) -> Tensor {
    auto [output, indices] = maxpool2d_forward_with_indices(input,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, queue);
    return output;
}

// AvgPool2d forward (pure SYCL)
auto avgpool2d_forward(const Tensor& input,
                       int64_t kernel_h, int64_t kernel_w,
                       int64_t stride_h, int64_t stride_w,
                       int64_t padding_h, int64_t padding_w,
                       bool count_include_pad, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AvgPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];

    const int64_t H_out = (H_in + 2 * padding_h - kernel_h) / stride_h + 1;
    const int64_t W_out = (W_in + 2 * padding_w - kernel_w) / stride_w + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dKernelFloat32>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            float sum = 0.0f;
            int64_t count = 0;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride_h - padding_h + kh;
                    int64_t w_in = w_out * stride_w - padding_w + kw;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        sum += in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in];
                        count++;
                    } else if (count_include_pad) {
                        count++;
                    }
                }
            }

            out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out] =
                count > 0 ? sum / static_cast<float>(count) : 0.0f;
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dKernelFloat64>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            double sum = 0.0;
            int64_t count = 0;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride_h - padding_h + kh;
                    int64_t w_in = w_out * stride_w - padding_w + kw;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        sum += in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in];
                        count++;
                    } else if (count_include_pad) {
                        count++;
                    }
                }
            }

            out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out] =
                count > 0 ? sum / static_cast<double>(count) : 0.0;
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dKernelFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Use float for accumulation for numerical stability
            float sum = 0.0f;
            int64_t count = 0;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride_h - padding_h + kh;
                    int64_t w_in = w_out * stride_w - padding_w + kw;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        sum += static_cast<float>(in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in]);
                        count++;
                    } else if (count_include_pad) {
                        count++;
                    }
                }
            }

            out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out] =
                sycl::half(count > 0 ? sum / static_cast<float>(count) : 0.0f);
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dKernelBFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Use float for accumulation for numerical stability
            float sum = 0.0f;
            int64_t count = 0;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride_h - padding_h + kh;
                    int64_t w_in = w_out * stride_w - padding_w + kw;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        sum += bf16_to_f32(in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in]);
                        count++;
                    } else if (count_include_pad) {
                        count++;
                    }
                }
            }

            out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out] =
                f32_to_bf16(count > 0 ? sum / static_cast<float>(count) : 0.0f);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for avgpool2d_forward");
    }

    return output;
}

#endif // TENZOR_HAS_ONEDNN

// AdaptiveAvgPool2d - always pure SYCL
auto adaptive_avgpool2d_forward(const Tensor& input, int64_t output_h, int64_t output_w,
                                 sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AdaptiveAvgPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];

    Tensor output({N, C, output_h, output_w}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        const int64_t total_size = N * C * output_h * output_w;
        queue.parallel_for<AdaptiveAvgPool2dKernelFloat32>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Calculate input region
            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            float sum = 0.0f;
            int64_t count = 0;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    sum += in_ptr[((n * C + c) * H_in + h) * W_in + w];
                    count++;
                }
            }

            out_ptr[((n * C + c) * output_h + h_out) * output_w + w_out] =
                count > 0 ? sum / static_cast<float>(count) : 0.0f;
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        const int64_t total_size = N * C * output_h * output_w;
        queue.parallel_for<AdaptiveAvgPool2dKernelFloat64>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            double sum = 0.0;
            int64_t count = 0;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    sum += in_ptr[((n * C + c) * H_in + h) * W_in + w];
                    count++;
                }
            }

            out_ptr[((n * C + c) * output_h + h_out) * output_w + w_out] =
                count > 0 ? sum / static_cast<double>(count) : 0.0;
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        const int64_t total_size = N * C * output_h * output_w;
        queue.parallel_for<AdaptiveAvgPool2dKernelFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Calculate input region
            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            // Use float accumulation for numerical stability
            float sum = 0.0f;
            int64_t count = 0;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    sum += static_cast<float>(in_ptr[((n * C + c) * H_in + h) * W_in + w]);
                    count++;
                }
            }

            out_ptr[((n * C + c) * output_h + h_out) * output_w + w_out] =
                sycl::half(count > 0 ? sum / static_cast<float>(count) : 0.0f);
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        const int64_t total_size = N * C * output_h * output_w;
        queue.parallel_for<AdaptiveAvgPool2dKernelBFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Calculate input region
            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            // Use float accumulation for numerical stability
            float sum = 0.0f;
            int64_t count = 0;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    sum += bf16_to_f32(in_ptr[((n * C + c) * H_in + h) * W_in + w]);
                    count++;
                }
            }

            out_ptr[((n * C + c) * output_h + h_out) * output_w + w_out] =
                f32_to_bf16(count > 0 ? sum / static_cast<float>(count) : 0.0f);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool2d_forward");
    }

    return output;
}

// AdaptiveAvgPool2d backward - always pure SYCL
auto adaptive_avgpool2d_backward(const Tensor& grad_output, int64_t H_in, int64_t W_in,
                                  sycl::queue& queue) -> Tensor {
    auto shape = grad_output.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AdaptiveAvgPool2d backward requires 4D grad_output (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_out = shape[2];
    const int64_t W_out = shape[3];

    Tensor grad_input({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());

    // Initialize grad_input to zeros
    const size_t bytes = grad_input.numel() * grad_input.dtype_size();
    queue.memset(const_cast<void*>(grad_input.data_ptr()), 0, bytes);

    if (grad_output.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AdaptiveAvgPool2dBackwardKernelFloat32>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Calculate input region
            const int64_t h_start = (h_out * H_in) / H_out;
            const int64_t h_end = ((h_out + 1) * H_in) / H_out;
            const int64_t w_start = (w_out * W_in) / W_out;
            const int64_t w_end = ((w_out + 1) * W_in) / W_out;

            const int64_t count = (h_end - h_start) * (w_end - w_start);
            const float grad_val = grad_out_ptr[flat_idx] / static_cast<float>(count);

            // Distribute gradient to input positions using atomic add
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    const int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> atomic_val(grad_in_ptr[input_idx]);
                    atomic_val.fetch_add(grad_val);
                }
            }
        });
    }
    else if (grad_output.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AdaptiveAvgPool2dBackwardKernelFloat64>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            const int64_t h_start = (h_out * H_in) / H_out;
            const int64_t h_end = ((h_out + 1) * H_in) / H_out;
            const int64_t w_start = (w_out * W_in) / W_out;
            const int64_t w_end = ((w_out + 1) * W_in) / W_out;

            const int64_t count = (h_end - h_start) * (w_end - w_start);
            const double grad_val = grad_out_ptr[flat_idx] / static_cast<double>(count);

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    const int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                    sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> atomic_val(grad_in_ptr[input_idx]);
                    atomic_val.fetch_add(grad_val);
                }
            }
        });
    }
    else if (grad_output.dtype() == DType::Float16) {
        // Float16 backward uses float accumulation for numerical stability
        // We need to first accumulate in float, then convert
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        // Use a temporary float buffer for atomic accumulation
        Tensor grad_input_f32({N, C, H_in, W_in}, DType::Float32, grad_output.device());
        queue.memset(const_cast<void*>(grad_input_f32.data_ptr()), 0, grad_input_f32.numel() * sizeof(float));
        float* grad_in_f32_ptr = get_data_ptr<float>(grad_input_f32);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AdaptiveAvgPool2dBackwardKernelFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            const int64_t h_start = (h_out * H_in) / H_out;
            const int64_t h_end = ((h_out + 1) * H_in) / H_out;
            const int64_t w_start = (w_out * W_in) / W_out;
            const int64_t w_end = ((w_out + 1) * W_in) / W_out;

            const int64_t count = (h_end - h_start) * (w_end - w_start);
            const float grad_val = static_cast<float>(grad_out_ptr[flat_idx]) / static_cast<float>(count);

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    const int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> atomic_val(grad_in_f32_ptr[input_idx]);
                    atomic_val.fetch_add(grad_val);
                }
            }
        });

        // Convert float result back to half
        const int64_t total_input = N * C * H_in * W_in;
        queue.parallel_for(sycl::range<1>(total_input), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = sycl::half(grad_in_f32_ptr[idx]);
        });
    }
    else if (grad_output.dtype() == DType::BFloat16) {
        // BFloat16 backward uses float accumulation for numerical stability
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        // Use a temporary float buffer for atomic accumulation
        Tensor grad_input_f32({N, C, H_in, W_in}, DType::Float32, grad_output.device());
        queue.memset(const_cast<void*>(grad_input_f32.data_ptr()), 0, grad_input_f32.numel() * sizeof(float));
        float* grad_in_f32_ptr = get_data_ptr<float>(grad_input_f32);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AdaptiveAvgPool2dBackwardKernelBFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            const int64_t h_start = (h_out * H_in) / H_out;
            const int64_t h_end = ((h_out + 1) * H_in) / H_out;
            const int64_t w_start = (w_out * W_in) / W_out;
            const int64_t w_end = ((w_out + 1) * W_in) / W_out;

            const int64_t count = (h_end - h_start) * (w_end - w_start);
            const float grad_val = bf16_to_f32(grad_out_ptr[flat_idx]) / static_cast<float>(count);

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    const int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> atomic_val(grad_in_f32_ptr[input_idx]);
                    atomic_val.fetch_add(grad_val);
                }
            }
        });

        // Convert float result back to BFloat16
        const int64_t total_input = N * C * H_in * W_in;
        queue.parallel_for(sycl::range<1>(total_input), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = f32_to_bf16(grad_in_f32_ptr[idx]);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool2d_backward");
    }

    return grad_input;
}

// AdaptiveMaxPool2d - always pure SYCL
auto adaptive_maxpool2d_forward(const Tensor& input, int64_t output_h, int64_t output_w,
                                 sycl::queue& queue) -> std::vector<Tensor> {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AdaptiveMaxPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];

    Tensor output({N, C, output_h, output_w}, input.dtype(), input.device());
    Tensor indices({N, C, output_h, output_w}, DType::Int64, input.device());
    const int64_t total_size = N * C * output_h * output_w;

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);

        queue.parallel_for<AdaptiveMaxPool2dKernelFloat32>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            float max_val = -3.4028235e+38f;
            int64_t max_idx = 0;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t idx = ((n * C + c) * H_in + h) * W_in + w;
                    float val = in_ptr[idx];
                    if (val > max_val) { max_val = val; max_idx = idx; }
                }
            }

            int64_t oi = ((n * C + c) * output_h + h_out) * output_w + w_out;
            out_ptr[oi] = max_val;
            idx_ptr[oi] = max_idx;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);

        queue.parallel_for<AdaptiveMaxPool2dKernelFloat64>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            double max_val = -1.7976931348623157e+308;
            int64_t max_idx = 0;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t idx = ((n * C + c) * H_in + h) * W_in + w;
                    double val = in_ptr[idx];
                    if (val > max_val) { max_val = val; max_idx = idx; }
                }
            }

            int64_t oi = ((n * C + c) * output_h + h_out) * output_w + w_out;
            out_ptr[oi] = max_val;
            idx_ptr[oi] = max_idx;
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);

        queue.parallel_for<AdaptiveMaxPool2dKernelFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            float max_val = -3.4028235e+38f;
            int64_t max_idx = 0;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t idx = ((n * C + c) * H_in + h) * W_in + w;
                    float val = static_cast<float>(in_ptr[idx]);
                    if (val > max_val) { max_val = val; max_idx = idx; }
                }
            }

            int64_t oi = ((n * C + c) * output_h + h_out) * output_w + w_out;
            out_ptr[oi] = sycl::half(max_val);
            idx_ptr[oi] = max_idx;
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);

        queue.parallel_for<AdaptiveMaxPool2dKernelBFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            float max_val = -3.4028235e+38f;
            int64_t max_idx = 0;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t idx = ((n * C + c) * H_in + h) * W_in + w;
                    float val = bf16_to_f32(in_ptr[idx]);
                    if (val > max_val) { max_val = val; max_idx = idx; }
                }
            }

            int64_t oi = ((n * C + c) * output_h + h_out) * output_w + w_out;
            out_ptr[oi] = f32_to_bf16(max_val);
            idx_ptr[oi] = max_idx;
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for adaptive_maxpool2d_forward");
    }

    return {output, indices};
}

// AdaptiveMaxPool2d backward - routes gradients to max positions using indices
auto adaptive_maxpool2d_backward(const Tensor& grad_output, const Tensor& indices,
                                   int64_t H_in, int64_t W_in,
                                   sycl::queue& queue) -> Tensor {
    auto shape = grad_output.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AdaptiveMaxPool2d backward requires 4D grad_output (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_out = shape[2];
    const int64_t W_out = shape[3];

    Tensor grad_input({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());

    // Initialize grad_input to zeros
    const size_t bytes = grad_input.numel() * grad_input.dtype_size();
    queue.memset(const_cast<void*>(grad_input.data_ptr()), 0, bytes);

    const int64_t total_size = N * C * H_out * W_out;

    if (grad_output.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        queue.parallel_for<AdaptiveMaxPool2dBackwardKernelFloat32>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t input_idx = idx_ptr[flat_idx];
            if (input_idx >= 0 && input_idx < N * C * H_in * W_in) {
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> atomic_val(grad_in_ptr[input_idx]);
                atomic_val.fetch_add(grad_out_ptr[flat_idx]);
            }
        });
    }
    else if (grad_output.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.parallel_for<AdaptiveMaxPool2dBackwardKernelFloat64>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t input_idx = idx_ptr[flat_idx];
            if (input_idx >= 0 && input_idx < N * C * H_in * W_in) {
                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> atomic_val(grad_in_ptr[input_idx]);
                atomic_val.fetch_add(grad_out_ptr[flat_idx]);
            }
        });
    }
    else if (grad_output.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        // Use temporary float buffer for atomic accumulation
        Tensor grad_input_f32({N, C, H_in, W_in}, DType::Float32, grad_output.device());
        queue.memset(const_cast<void*>(grad_input_f32.data_ptr()), 0, grad_input_f32.numel() * sizeof(float));
        float* grad_in_f32_ptr = get_data_ptr<float>(grad_input_f32);

        queue.parallel_for<AdaptiveMaxPool2dBackwardKernelFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t input_idx = idx_ptr[flat_idx];
            if (input_idx >= 0 && input_idx < N * C * H_in * W_in) {
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> atomic_val(grad_in_f32_ptr[input_idx]);
                atomic_val.fetch_add(static_cast<float>(grad_out_ptr[flat_idx]));
            }
        });

        // Convert float result back to half
        const int64_t total_input = N * C * H_in * W_in;
        queue.parallel_for(sycl::range<1>(total_input), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = sycl::half(grad_in_f32_ptr[idx]);
        });
    }
    else if (grad_output.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        // Use temporary float buffer for atomic accumulation
        Tensor grad_input_f32({N, C, H_in, W_in}, DType::Float32, grad_output.device());
        queue.memset(const_cast<void*>(grad_input_f32.data_ptr()), 0, grad_input_f32.numel() * sizeof(float));
        float* grad_in_f32_ptr = get_data_ptr<float>(grad_input_f32);

        queue.parallel_for<AdaptiveMaxPool2dBackwardKernelBFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t input_idx = idx_ptr[flat_idx];
            if (input_idx >= 0 && input_idx < N * C * H_in * W_in) {
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> atomic_val(grad_in_f32_ptr[input_idx]);
                atomic_val.fetch_add(bf16_to_f32(grad_out_ptr[flat_idx]));
            }
        });

        // Convert float result back to BFloat16
        const int64_t total_input = N * C * H_in * W_in;
        queue.parallel_for(sycl::range<1>(total_input), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = f32_to_bf16(grad_in_f32_ptr[idx]);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for adaptive_maxpool2d_backward");
    }

    return grad_input;
}

/**
 * @brief Average pooling 2D operation wrapper.
 *
 * Applies 2D average pooling over an input signal.
 * OpAttributes wrapper for avg_pool2d operation.
 *
 * @param input Input tensor (4D: batch, channels, height, width)
 * @param attrs Operation attributes containing kernel_size, stride, padding
 * @param queue SYCL queue for execution
 * @return Tensor Pooled output tensor
 */
auto avg_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    // FP16/BF16 pooling is unsupported on some SYCL oneDNN devices ("could not
    // create a descriptor for a pooling primitive"). Widen those to FP32.
    // FP64 is computed natively (oneDNN has no FP64 pooling primitive, and
    // widening it would break FP64 finite-difference gradcheck precision).
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig = input.dtype();
        return avg_pool2d_kernel(input.to(DType::Float32), attrs, queue).to(orig);
    }
    if (!attrs.has(AttrKey::KernelSize) && !attrs.has(AttrKey::KernelSizeH) && !attrs.has(AttrKey::KernelSizeW)) {
        throw std::invalid_argument("avg_pool2d: 'kernel_size' attribute is required");
    }

    // Per-axis with scalar fallback. Matches the canonical pattern in
    // include/tenzor/backend/attr_macros.hpp.
    const int64_t k_scalar = attrs.get_int(AttrKey::KernelSize, 1);
    const int64_t kernel_h = attrs.get_int(AttrKey::KernelSizeH, k_scalar);
    const int64_t kernel_w = attrs.get_int(AttrKey::KernelSizeW, k_scalar);
    const int64_t s_scalar = attrs.get_int(AttrKey::Stride, k_scalar);
    const int64_t stride_h = attrs.get_int(AttrKey::StrideH, s_scalar);
    const int64_t stride_w = attrs.get_int(AttrKey::StrideW, s_scalar);
    const int64_t p_scalar = attrs.get_int(AttrKey::Padding, 0);
    const int64_t padding_h = attrs.get_int(AttrKey::PaddingH, p_scalar);
    const int64_t padding_w = attrs.get_int(AttrKey::PaddingW, p_scalar);
    const bool count_include_pad = attrs.get_bool(AttrKey::CountIncludePad, false);

#ifdef TENZOR_HAS_ONEDNN
    // oneDNN avgpool2d_forward has no FP64 primitive; route FP64 to native SYCL.
    if (input.dtype() == DType::Float64) {
        return avgpool2d_forward_f64_sycl(input, kernel_h, kernel_w, stride_h, stride_w,
                                          padding_h, padding_w, count_include_pad, queue);
    }
#endif
    return avgpool2d_forward(input, kernel_h, kernel_w, stride_h, stride_w,
                              padding_h, padding_w, count_include_pad, queue);
}

/**
 * @brief Max pooling 2D operation wrapper.
 *
 * Applies 2D max pooling over an input signal.
 * OpAttributes wrapper for max_pool2d operation.
 * Returns both output and indices tensors for backward pass support.
 *
 * @param input Input tensor (4D: batch, channels, height, width)
 * @param attrs Operation attributes containing kernel_size, stride, padding
 * @param queue SYCL queue for execution
 * @return std::pair<Tensor, Tensor> Pooled output tensor and indices tensor
 */
auto max_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    // FP16/BF16 widen to FP32 (oneDNN lacks those primitives). FP64 must NOT
    // widen: max-pool is a pure element selection with no arithmetic, so the
    // native double sycl path (maxpool2d_forward_with_indices -> *_sycl)
    // reproduces the input element bit-exactly. Widening FP64 to FP32 truncated
    // the selected element by ~1e-7 and broke exact cross-backend parity (and
    // FP64 finite-difference gradcheck) — same rationale as adaptive_avg_pool2d.
    // Indices are dtype-independent — only narrow the output.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig = input.dtype();
        auto [out, idx] = max_pool2d_kernel(input.to(DType::Float32), attrs, queue);
        return {out.to(orig), idx};
    }
    if (!attrs.has(AttrKey::KernelSize) && !attrs.has(AttrKey::KernelSizeH) && !attrs.has(AttrKey::KernelSizeW)) {
        throw std::invalid_argument("max_pool2d: 'kernel_size' attribute is required");
    }

    // Per-axis with scalar fallback.
    const int64_t k_scalar = attrs.get_int(AttrKey::KernelSize, 1);
    const int64_t kernel_h = attrs.get_int(AttrKey::KernelSizeH, k_scalar);
    const int64_t kernel_w = attrs.get_int(AttrKey::KernelSizeW, k_scalar);
    const int64_t s_scalar = attrs.get_int(AttrKey::Stride, k_scalar);
    const int64_t stride_h = attrs.get_int(AttrKey::StrideH, s_scalar);
    const int64_t stride_w = attrs.get_int(AttrKey::StrideW, s_scalar);
    const int64_t p_scalar = attrs.get_int(AttrKey::Padding, 0);
    const int64_t padding_h = attrs.get_int(AttrKey::PaddingH, p_scalar);
    const int64_t padding_w = attrs.get_int(AttrKey::PaddingW, p_scalar);
    const int64_t d_scalar = attrs.get_int(AttrKey::Dilation, 1);
    const int64_t dilation_h = attrs.get_int(AttrKey::DilationH, d_scalar);
    const int64_t dilation_w = attrs.get_int(AttrKey::DilationW, d_scalar);

    return maxpool2d_forward_with_indices(input,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, queue);
}

/**
 * @brief Adaptive average pooling 2D operation wrapper.
 *
 * Applies 2D adaptive average pooling over an input signal.
 * OpAttributes wrapper for adaptive_avg_pool2d operation.
 *
 * @param input Input tensor (4D: batch, channels, height, width)
 * @param attrs Operation attributes containing output_size
 * @param queue SYCL queue for execution
 * @return Tensor Pooled output tensor
 */
auto adaptive_avg_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    // FP16/BF16 widen to FP32; FP64 runs natively (widening breaks FP64 gradcheck).
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig = input.dtype();
        return adaptive_avg_pool2d_kernel(input.to(DType::Float32), attrs, queue).to(orig);
    }
    int64_t output_h = 1, output_w = 1;

    // Support both formats: "output_size" (H,W string) and "output_h"/"output_w" (separate integers)
    if (attrs.has(AttrKey::OutputSize)) {
        // Parse output_size (format: "H,W")
        auto sizes = attrs.get_int_list(AttrKey::OutputSize);
        if (sizes.size() != 2) {
            throw std::invalid_argument("adaptive_avg_pool2d: output_size must have 2 values");
        }
        output_h = sizes[0];
        output_w = sizes[1];
    } else if (attrs.has(AttrKey::OutputSizeH) && attrs.has(AttrKey::OutputSizeW)) {
        output_h = attrs.get_int(AttrKey::OutputSizeH);
        output_w = attrs.get_int(AttrKey::OutputSizeW);
    } else if (attrs.has(AttrKey::OutputSizeH)) {
        // Square output if only output_h is provided
        output_h = attrs.get_int(AttrKey::OutputSizeH);
        output_w = output_h;
    } else {
        throw std::invalid_argument("adaptive_avg_pool2d: 'output_size' or 'output_h'/'output_w' attributes are required");
    }

    return adaptive_avgpool2d_forward(input, output_h, output_w, queue);
}

/**
 * @brief Adaptive max pooling 2D operation wrapper.
 *
 * Applies 2D adaptive max pooling over an input signal.
 * OpAttributes wrapper for adaptive_max_pool2d operation.
 *
 * @param input Input tensor (4D: batch, channels, height, width)
 * @param attrs Operation attributes containing output_size
 * @param queue SYCL queue for execution
 * @return Tensor Pooled output tensor
 */
auto adaptive_max_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> std::vector<Tensor> {
    if (input.dtype() == DType::Float64 || input.dtype() == DType::Float16 ||
        input.dtype() == DType::BFloat16) {
        const DType orig = input.dtype();
        auto out = adaptive_max_pool2d_kernel(input.to(DType::Float32), attrs, queue);
        if (!out.empty()) out[0] = out[0].to(orig);  // out[1] = indices (dtype-independent)
        return out;
    }
    int64_t output_h = 1, output_w = 1;

    // Support both formats: "output_size" (H,W string) and "output_h"/"output_w" (separate integers)
    if (attrs.has(AttrKey::OutputSize)) {
        // Parse output_size (format: "H,W")
        auto sizes = attrs.get_int_list(AttrKey::OutputSize);
        if (sizes.size() != 2) {
            throw std::invalid_argument("adaptive_max_pool2d: output_size must have 2 values");
        }
        output_h = sizes[0];
        output_w = sizes[1];
    } else if (attrs.has(AttrKey::OutputSizeH) && attrs.has(AttrKey::OutputSizeW)) {
        output_h = attrs.get_int(AttrKey::OutputSizeH);
        output_w = attrs.get_int(AttrKey::OutputSizeW);
    } else if (attrs.has(AttrKey::OutputSizeH)) {
        // Square output if only output_h is provided
        output_h = attrs.get_int(AttrKey::OutputSizeH);
        output_w = output_h;
    } else {
        throw std::invalid_argument("adaptive_max_pool2d: 'output_size' or 'output_h'/'output_w' attributes are required");
    }

    return adaptive_maxpool2d_forward(input, output_h, output_w, queue);
}

/**
 * @brief Average pooling 2D backward operation.
 *
 * Computes gradients for average pooling by distributing the gradient
 * evenly across the pooling window for each output position.
 *
 * @param grad_output Gradient from next layer (4D: batch, channels, height_out, width_out)
 * @param input Original input tensor (used for shape information)
 * @param attrs Operation attributes containing kernel_size, stride, padding
 * @param queue SYCL queue for execution
 * @return Tensor Gradient with respect to input
 */
auto avg_pool2d_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    // FP16/BF16 widen to FP32 (no native atomic accumulation for those types).
    // FP64 is handled by the native FP64 branch below — widening it loses the
    // precision FP64 finite-difference gradcheck requires.
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        const DType orig = grad_output.dtype();
        return avg_pool2d_backward_kernel(grad_output.to(DType::Float32),
                                          input.to(DType::Float32), attrs, queue).to(orig);
    }
    if (!attrs.has(AttrKey::KernelSize) && !attrs.has(AttrKey::KernelSizeH) && !attrs.has(AttrKey::KernelSizeW)) {
        throw std::invalid_argument("avg_pool2d_backward: 'kernel_size' attribute is required");
    }

    auto grad_shape = grad_output.shape();
    auto input_shape = input.shape();

    if (grad_shape.size() != 4 || input_shape.size() != 4) {
        throw std::invalid_argument("avg_pool2d_backward requires 4D inputs (N, C, H, W)");
    }

    // Per-axis with scalar fallback.
    const int64_t k_scalar = attrs.get_int(AttrKey::KernelSize, 1);
    const int64_t kernel_h = attrs.get_int(AttrKey::KernelSizeH, k_scalar);
    const int64_t kernel_w = attrs.get_int(AttrKey::KernelSizeW, k_scalar);
    const int64_t s_scalar = attrs.get_int(AttrKey::Stride, k_scalar);
    const int64_t stride_h = attrs.get_int(AttrKey::StrideH, s_scalar);
    const int64_t stride_w = attrs.get_int(AttrKey::StrideW, s_scalar);
    const int64_t p_scalar = attrs.get_int(AttrKey::Padding, 0);
    const int64_t padding_h = attrs.get_int(AttrKey::PaddingH, p_scalar);
    const int64_t padding_w = attrs.get_int(AttrKey::PaddingW, p_scalar);
    const bool count_include_pad = attrs.get_bool(AttrKey::CountIncludePad, false);

    const int64_t N = input_shape[0];
    const int64_t C = input_shape[1];
    const int64_t H_in = input_shape[2];
    const int64_t W_in = input_shape[3];

    const int64_t H_out = grad_shape[2];
    const int64_t W_out = grad_shape[3];

    // Create gradient input tensor (same shape as original input)
    Tensor grad_input({N, C, H_in, W_in}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        // Initialize grad_input to zero
        const int64_t input_size = N * C * H_in * W_in;
        queue.fill(grad_in_ptr, 0.0f, input_size).wait();

        // For each output position, distribute gradient to input positions
        const int64_t total_output_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dBackwardKernelFloat32>(sycl::range<1>(total_output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const float grad_val = grad_out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out];

                // Count valid positions in pooling window
                int64_t count = 0;
                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                    for (int64_t kw = 0; kw < kernel_w; ++kw) {
                        int64_t h_in = h_out * stride_h - padding_h + kh;
                        int64_t w_in = w_out * stride_w - padding_w + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            count++;
                        } else if (count_include_pad) {
                            count++;
                        }
                    }
                }

                // Distribute gradient evenly
                const float grad_per_input = count > 0 ? grad_val / static_cast<float>(count) : 0.0f;

                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                    for (int64_t kw = 0; kw < kernel_w; ++kw) {
                        int64_t h_in = h_out * stride_h - padding_h + kh;
                        int64_t w_in = w_out * stride_w - padding_w + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                           sycl::memory_scope::device> atomic_grad(grad_in_ptr[input_idx]);
                            atomic_grad += grad_per_input;
                        }
                    }
                }
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        const int64_t input_size = N * C * H_in * W_in;
        queue.fill(grad_in_ptr, 0.0, input_size).wait();

        const int64_t total_output_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dBackwardKernelFloat64>(sycl::range<1>(total_output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const double grad_val = grad_out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out];

                int64_t count = 0;
                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                    for (int64_t kw = 0; kw < kernel_w; ++kw) {
                        int64_t h_in = h_out * stride_h - padding_h + kh;
                        int64_t w_in = w_out * stride_w - padding_w + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            count++;
                        } else if (count_include_pad) {
                            count++;
                        }
                    }
                }

                const double grad_per_input = count > 0 ? grad_val / static_cast<double>(count) : 0.0;

                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                    for (int64_t kw = 0; kw < kernel_w; ++kw) {
                        int64_t h_in = h_out * stride_h - padding_h + kh;
                        int64_t w_in = w_out * stride_w - padding_w + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                           sycl::memory_scope::device> atomic_grad(grad_in_ptr[input_idx]);
                            atomic_grad += grad_per_input;
                        }
                    }
                }
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        // Use float32 intermediate buffer since atomic_ref<sycl::half> is not widely supported
        const int64_t input_size = N * C * H_in * W_in;
        float* acc_ptr = sycl::malloc_device<float>(input_size, queue);
        queue.fill(acc_ptr, 0.0f, input_size).wait();

        const int64_t total_output_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dBackwardKernelFloat16>(sycl::range<1>(total_output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const float grad_val = static_cast<float>(grad_out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out]);

                int64_t count = 0;
                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                    for (int64_t kw = 0; kw < kernel_w; ++kw) {
                        int64_t h_in = h_out * stride_h - padding_h + kh;
                        int64_t w_in = w_out * stride_w - padding_w + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            count++;
                        } else if (count_include_pad) {
                            count++;
                        }
                    }
                }

                const float grad_per_input = count > 0 ? grad_val / static_cast<float>(count) : 0.0f;

                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                    for (int64_t kw = 0; kw < kernel_w; ++kw) {
                        int64_t h_in = h_out * stride_h - padding_h + kh;
                        int64_t w_in = w_out * stride_w - padding_w + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                           sycl::memory_scope::device> atomic_grad(acc_ptr[input_idx]);
                            atomic_grad += grad_per_input;
                        }
                    }
                }
        });

        // Convert float32 accumulator back to Float16
        queue.parallel_for(sycl::range<1>(input_size), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = sycl::half(acc_ptr[idx]);
        });

        sycl::free(acc_ptr, queue);
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        // Use float32 intermediate buffer since atomic_ref<uint16_t> is not suitable for BFloat16
        const int64_t input_size = N * C * H_in * W_in;
        float* acc_ptr = sycl::malloc_device<float>(input_size, queue);
        queue.fill(acc_ptr, 0.0f, input_size).wait();

        const int64_t total_output_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dBackwardKernelBFloat16>(sycl::range<1>(total_output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const float grad_val = bf16_to_f32(grad_out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out]);

                int64_t count = 0;
                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                    for (int64_t kw = 0; kw < kernel_w; ++kw) {
                        int64_t h_in = h_out * stride_h - padding_h + kh;
                        int64_t w_in = w_out * stride_w - padding_w + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            count++;
                        } else if (count_include_pad) {
                            count++;
                        }
                    }
                }

                const float grad_per_input = count > 0 ? grad_val / static_cast<float>(count) : 0.0f;

                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                    for (int64_t kw = 0; kw < kernel_w; ++kw) {
                        int64_t h_in = h_out * stride_h - padding_h + kh;
                        int64_t w_in = w_out * stride_w - padding_w + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                           sycl::memory_scope::device> atomic_grad(acc_ptr[input_idx]);
                            atomic_grad += grad_per_input;
                        }
                    }
                }
        });

        // Convert float32 accumulator back to BFloat16
        queue.parallel_for(sycl::range<1>(input_size), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = f32_to_bf16(acc_ptr[idx]);
        });

        sycl::free(acc_ptr, queue);
    }
    else {
        throw std::runtime_error("Unsupported dtype for avg_pool2d_backward");
    }

    return grad_input;
}

/**
 * @brief Adaptive average pooling 2D backward operation wrapper.
 *
 * Computes gradients for adaptive average pooling.
 * OpAttributes wrapper for adaptive_avg_pool2d_backward operation.
 *
 * @param grad_output Gradient from next layer (4D: batch, channels, height_out, width_out)
 * @param input Original input tensor (used for shape information)
 * @param attrs Operation attributes (optional, H_in and W_in can be derived from input)
 * @param queue SYCL queue for execution
 * @return Tensor Gradient with respect to input
 */
auto adaptive_avg_pool2d_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                          const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    // FP16/BF16 widen to FP32; FP64 runs natively (widening breaks FP64 gradcheck).
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        const DType orig = grad_output.dtype();
        return adaptive_avg_pool2d_backward_kernel(grad_output.to(DType::Float32),
                                                   input.to(DType::Float32), attrs, queue).to(orig);
    }
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("adaptive_avg_pool2d_backward requires 4D input (N, C, H, W)");
    }

    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];

    // Allow override from attrs if provided
    if (attrs.has(AttrKey::InputH)) {
        H_in = attrs.get_int(AttrKey::InputH);
    }
    if (attrs.has(AttrKey::InputW)) {
        W_in = attrs.get_int(AttrKey::InputW);
    }

    return adaptive_avgpool2d_backward(grad_output, H_in, W_in, queue);
}

/**
 * @brief Max pooling 2D backward operation using stored indices.
 *
 * Computes gradients for max pooling by routing gradient to the
 * max element positions stored in the indices tensor.
 *
 * @param grad_output Gradient from next layer (4D: batch, channels, height_out, width_out)
 * @param indices Indices tensor from forward pass (same shape as grad_output)
 * @param H_in Original input height
 * @param W_in Original input width
 * @param queue SYCL queue for execution
 * @return Tensor Gradient with respect to input
 */
auto max_pool2d_backward_with_indices(const Tensor& grad_output, const Tensor& indices,
                                       int64_t H_in, int64_t W_in, sycl::queue& queue) -> Tensor {
    auto grad_shape = grad_output.shape();
    if (grad_shape.size() != 4) {
        throw std::invalid_argument("max_pool2d_backward_with_indices requires 4D grad_output (N, C, H, W)");
    }

    const int64_t N = grad_shape[0];
    const int64_t C = grad_shape[1];
    const int64_t H_out = grad_shape[2];
    const int64_t W_out = grad_shape[3];

    // Create gradient input tensor
    Tensor grad_input({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());

    const int64_t input_size = N * C * H_in * W_in;
    const int64_t output_size = N * C * H_out * W_out;

    // Indices are always stored as Int64 by the forward pass
    const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

    if (grad_output.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        queue.fill(grad_in_ptr, 0.0f, input_size).wait();

        // The forward stores max_idx as a full N*C*H*W flat index (see
        // pooling.cpp:277: `max_idx = ((n*C+c)*H_in+h_in)*W_in+w_in`).
        // The previous backward reinterpreted it per-channel via
        // `h_in = max_idx / W_in` and then bounds-checked `h_in < H_in`,
        // which rejected every idx for n>0 or c>0 — so gradients never
        // reached the later batches/channels. Scatter directly with the
        // full flat index instead.
        queue.parallel_for<MaxPool2dBackwardWithIndicesKernelFloat32>(sycl::range<1>(output_size),
            [=](sycl::id<1> flat_idx) {
                const float grad_val = grad_out_ptr[flat_idx];
                const int64_t max_idx = idx_ptr[flat_idx];

                if (max_idx >= 0 && max_idx < input_size) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space> atomic_grad(grad_in_ptr[max_idx]);
                    atomic_grad.fetch_add(grad_val);
                }
        });
    }
    else if (grad_output.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.fill(grad_in_ptr, 0.0, input_size).wait();

        queue.parallel_for<MaxPool2dBackwardWithIndicesKernelFloat64>(sycl::range<1>(output_size),
            [=](sycl::id<1> flat_idx) {
                const double grad_val = grad_out_ptr[flat_idx];
                const int64_t max_idx = idx_ptr[flat_idx];

                if (max_idx >= 0 && max_idx < input_size) {
                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space> atomic_grad(grad_in_ptr[max_idx]);
                    atomic_grad.fetch_add(grad_val);
                }
        });
    }
    else if (grad_output.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        // Use float accumulation buffer for Float16
        Tensor grad_input_float({N, C, H_in, W_in}, DType::Float32, grad_output.device());
        float* grad_in_float_ptr = get_data_ptr<float>(grad_input_float);

        queue.fill(grad_in_float_ptr, 0.0f, input_size).wait();

        queue.parallel_for<MaxPool2dBackwardWithIndicesKernelFloat16>(sycl::range<1>(output_size),
            [=](sycl::id<1> flat_idx) {
                const float grad_val = static_cast<float>(grad_out_ptr[flat_idx]);
                const int64_t max_idx = idx_ptr[flat_idx];
                if (max_idx >= 0 && max_idx < input_size) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space> atomic_grad(grad_in_float_ptr[max_idx]);
                    atomic_grad.fetch_add(grad_val);
                }
        });

        // Convert back to Float16
        queue.parallel_for(sycl::range<1>(input_size), [=](sycl::id<1> i) {
            grad_in_ptr[i] = sycl::half(grad_in_float_ptr[i]);
        });
    }
    else if (grad_output.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        // Use float accumulation buffer for BFloat16
        Tensor grad_input_float({N, C, H_in, W_in}, DType::Float32, grad_output.device());
        float* grad_in_float_ptr = get_data_ptr<float>(grad_input_float);

        queue.fill(grad_in_float_ptr, 0.0f, input_size).wait();

        queue.parallel_for<MaxPool2dBackwardWithIndicesKernelBFloat16>(sycl::range<1>(output_size),
            [=](sycl::id<1> flat_idx) {
                const float grad_val = bf16_to_f32(grad_out_ptr[flat_idx]);
                const int64_t max_idx = idx_ptr[flat_idx];
                if (max_idx >= 0 && max_idx < input_size) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space> atomic_grad(grad_in_float_ptr[max_idx]);
                    atomic_grad.fetch_add(grad_val);
                }
        });

        // Convert back to BFloat16
        queue.parallel_for(sycl::range<1>(input_size), [=](sycl::id<1> i) {
            grad_in_ptr[i] = f32_to_bf16(grad_in_float_ptr[i]);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for max_pool2d_backward_with_indices");
    }

    return grad_input;
}

// =============================================================================
// 1D Pooling Operations
// =============================================================================

auto maxpool1d_forward(const Tensor& input, std::array<int64_t, 1> kernel_size_a, std::array<int64_t, 1> stride_a,
                       std::array<int64_t, 1> padding_a, std::array<int64_t, 1> dilation_a, sycl::queue& queue) -> std::vector<Tensor> {
    // Q.7: per-axis std::array<int64_t, 1> signature + dilation parameter
    // (previously missing — PyTorch supports dilation > 1).
    const int64_t kernel_size = kernel_size_a[0];
    const int64_t stride      = stride_a[0];
    const int64_t padding     = padding_a[0];
    const int64_t dilation    = dilation_a[0];
    auto shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("MaxPool1d requires 3D input (N, C, L)");
    }
    const int64_t N = shape[0], C = shape[1], L_in = shape[2];
    const int64_t L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    Tensor output({N, C, L_out}, input.dtype(), input.device());
    Tensor indices({N, C, L_out}, DType::Int64, input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const int64_t total = N * C * L_out;
        queue.parallel_for<MaxPool1dForwardFloat32>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            float mx = -3.4028235e+38f; int64_t mi = 0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k * dilation;
                if (li >= 0 && li < L_in) {
                    int64_t idx = (n * C + c) * L_in + li;
                    float v = in_ptr[idx];
                    if (v > mx) { mx = v; mi = idx; }
                }
            }
            int64_t oi = (n * C + c) * L_out + l;
            out_ptr[oi] = mx; idx_ptr[oi] = mi;
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const int64_t total = N * C * L_out;
        queue.parallel_for<MaxPool1dForwardFloat64>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            double mx = -1.7976931348623157e+308; int64_t mi = 0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k * dilation;
                if (li >= 0 && li < L_in) {
                    int64_t idx = (n * C + c) * L_in + li;
                    double v = in_ptr[idx];
                    if (v > mx) { mx = v; mi = idx; }
                }
            }
            int64_t oi = (n * C + c) * L_out + l;
            out_ptr[oi] = mx; idx_ptr[oi] = mi;
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const int64_t total = N * C * L_out;
        queue.parallel_for<MaxPool1dForwardFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            float mx = -3.4028235e+38f; int64_t mi = 0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k * dilation;
                if (li >= 0 && li < L_in) {
                    int64_t idx = (n * C + c) * L_in + li;
                    float v = static_cast<float>(in_ptr[idx]);
                    if (v > mx) { mx = v; mi = idx; }
                }
            }
            int64_t oi = (n * C + c) * L_out + l;
            out_ptr[oi] = sycl::half(mx); idx_ptr[oi] = mi;
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const int64_t total = N * C * L_out;
        queue.parallel_for<MaxPool1dForwardBFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            float mx = -3.4028235e+38f; int64_t mi = 0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k * dilation;
                if (li >= 0 && li < L_in) {
                    int64_t idx = (n * C + c) * L_in + li;
                    float v = bf16_to_f32(in_ptr[idx]);
                    if (v > mx) { mx = v; mi = idx; }
                }
            }
            int64_t oi = (n * C + c) * L_out + l;
            out_ptr[oi] = f32_to_bf16(mx); idx_ptr[oi] = mi;
        });
    } else {
        throw std::runtime_error("Unsupported dtype for maxpool1d_forward");
    }
    return {output, indices};
}

auto maxpool1d_backward(const Tensor& grad_output, const Tensor& indices,
                         const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor {
    if (input_shape.size() != 3) {
        throw std::invalid_argument("MaxPool1d backward requires 3D input_shape (N, C, L)");
    }
    const int64_t N = input_shape[0], C = input_shape[1], L_in = input_shape[2];
    auto grad_shape = grad_output.shape();
    const int64_t L_out = grad_shape[2];

    Tensor grad_input({N, C, L_in}, grad_output.dtype(), grad_output.device());
    const int64_t in_size = N * C * L_in;
    const int64_t out_size = N * C * L_out;

    // W.5: explicitly chain fill → scatter via depends_on instead of
    // relying on the in_order queue contract; safer if a future change
    // switches to an out-of-order queue.
    if (grad_output.dtype() == DType::Float32) {
        const float* go = get_data_ptr<const float>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);
        float* gi = get_data_ptr<float>(grad_input);
        auto fill_evt = queue.fill(gi, 0.0f, in_size);
        queue.submit([&](sycl::handler& h) {
            h.depends_on(fill_evt);
            h.parallel_for<MaxPool1dBackwardFloat32>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
                float gv = go[gid];
                int64_t mi = idx[gid];
                if (mi >= 0 && mi < in_size) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> ar(gi[mi]);
                    ar.fetch_add(gv);
                }
            });
        });
    } else if (grad_output.dtype() == DType::Float64) {
        const double* go = get_data_ptr<const double>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);
        double* gi = get_data_ptr<double>(grad_input);
        auto fill_evt = queue.fill(gi, 0.0, in_size);
        queue.submit([&](sycl::handler& h) {
            h.depends_on(fill_evt);
            h.parallel_for<MaxPool1dBackwardFloat64>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
                double gv = go[gid];
                int64_t mi = idx[gid];
                if (mi >= 0 && mi < in_size) {
                    sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> ar(gi[mi]);
                    ar.fetch_add(gv);
                }
            });
        });
    } else if (grad_output.dtype() == DType::Float16) {
        const sycl::half* go = get_data_ptr<const sycl::half>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);
        sycl::half* gi = get_data_ptr<sycl::half>(grad_input);
        float* acc = sycl::malloc_device<float>(in_size, queue);
        auto fill_evt = queue.fill(acc, 0.0f, in_size);
        auto scatter_evt = queue.submit([&](sycl::handler& h) {
            h.depends_on(fill_evt);
            h.parallel_for<MaxPool1dBackwardFloat16>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
                float gv = static_cast<float>(go[gid]);
                int64_t mi = idx[gid];
                if (mi >= 0 && mi < in_size) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> ar(acc[mi]);
                    ar.fetch_add(gv);
                }
            });
        });
        queue.submit([&](sycl::handler& h) {
            h.depends_on(scatter_evt);
            h.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) {
                gi[i] = sycl::half(acc[i]);
            });
        }).wait();
        sycl::free(acc, queue);
    } else if (grad_output.dtype() == DType::BFloat16) {
        const uint16_t* go = get_data_ptr<const uint16_t>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);
        uint16_t* gi = get_data_ptr<uint16_t>(grad_input);
        float* acc = sycl::malloc_device<float>(in_size, queue);
        auto fill_evt = queue.fill(acc, 0.0f, in_size);
        auto scatter_evt = queue.submit([&](sycl::handler& h) {
            h.depends_on(fill_evt);
            h.parallel_for<MaxPool1dBackwardBFloat16>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
                float gv = bf16_to_f32(go[gid]);
                int64_t mi = idx[gid];
                if (mi >= 0 && mi < in_size) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> ar(acc[mi]);
                    ar.fetch_add(gv);
                }
            });
        });
        queue.submit([&](sycl::handler& h) {
            h.depends_on(scatter_evt);
            h.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) {
                gi[i] = f32_to_bf16(acc[i]);
            });
        }).wait();
        sycl::free(acc, queue);
    } else {
        throw std::runtime_error("Unsupported dtype for maxpool1d_backward");
    }
    return grad_input;
}

auto avgpool1d_forward(const Tensor& input, std::array<int64_t, 1> kernel_size_a, std::array<int64_t, 1> stride_a,
                       std::array<int64_t, 1> padding_a, sycl::queue& queue) -> Tensor {
    // Q.7: per-axis std::array<int64_t, 1> signature.
    const int64_t kernel_size = kernel_size_a[0];
    const int64_t stride      = stride_a[0];
    const int64_t padding     = padding_a[0];
    auto shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("AvgPool1d requires 3D input (N, C, L)");
    }
    const int64_t N = shape[0], C = shape[1], L_in = shape[2];
    const int64_t L_out = (L_in + 2 * padding - kernel_size) / stride + 1;

    Tensor output({N, C, L_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        const int64_t total = N * C * L_out;
        queue.parallel_for<AvgPool1dForwardFloat32>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            float sum = 0.0f; int64_t count = 0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k;
                if (li >= 0 && li < L_in) { sum += in_ptr[(n * C + c) * L_in + li]; count++; }
            }
            out_ptr[(n * C + c) * L_out + l] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const int64_t total = N * C * L_out;
        queue.parallel_for<AvgPool1dForwardFloat64>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            double sum = 0.0; int64_t count = 0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k;
                if (li >= 0 && li < L_in) { sum += in_ptr[(n * C + c) * L_in + li]; count++; }
            }
            out_ptr[(n * C + c) * L_out + l] = count > 0 ? sum / static_cast<double>(count) : 0.0;
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        const int64_t total = N * C * L_out;
        queue.parallel_for<AvgPool1dForwardFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            float sum = 0.0f; int64_t count = 0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k;
                if (li >= 0 && li < L_in) { sum += static_cast<float>(in_ptr[(n * C + c) * L_in + li]); count++; }
            }
            out_ptr[(n * C + c) * L_out + l] = sycl::half(count > 0 ? sum / static_cast<float>(count) : 0.0f);
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        const int64_t total = N * C * L_out;
        queue.parallel_for<AvgPool1dForwardBFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            float sum = 0.0f; int64_t count = 0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k;
                if (li >= 0 && li < L_in) { sum += bf16_to_f32(in_ptr[(n * C + c) * L_in + li]); count++; }
            }
            out_ptr[(n * C + c) * L_out + l] = f32_to_bf16(count > 0 ? sum / static_cast<float>(count) : 0.0f);
        });
    } else {
        throw std::runtime_error("Unsupported dtype for avgpool1d_forward");
    }
    return output;
}

auto avgpool1d_backward(const Tensor& grad_output, std::array<int64_t, 1> kernel_size_a, std::array<int64_t, 1> stride_a,
                         std::array<int64_t, 1> padding_a, const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor {
    // Q.7: per-axis std::array<int64_t, 1> signature.
    const int64_t kernel_size = kernel_size_a[0];
    const int64_t stride      = stride_a[0];
    const int64_t padding     = padding_a[0];
    if (input_shape.size() != 3) {
        throw std::invalid_argument("AvgPool1d backward requires 3D input_shape (N, C, L)");
    }
    const int64_t N = input_shape[0], C = input_shape[1], L_in = input_shape[2];
    auto gs = grad_output.shape();
    const int64_t L_out = gs[2];

    Tensor grad_input({N, C, L_in}, grad_output.dtype(), grad_output.device());
    const int64_t in_size = N * C * L_in;
    const int64_t out_size = N * C * L_out;

    if (grad_output.dtype() == DType::Float32) {
        const float* go = get_data_ptr<const float>(grad_output);
        float* gi = get_data_ptr<float>(grad_input);
        queue.fill(gi, 0.0f, in_size).wait();
        queue.parallel_for<AvgPool1dBackwardFloat32>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t count = 0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k;
                if (li >= 0 && li < L_in) count++;
            }
            float gv = count > 0 ? go[gid] / static_cast<float>(count) : 0.0f;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k;
                if (li >= 0 && li < L_in) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> ar(gi[(n * C + c) * L_in + li]);
                    ar.fetch_add(gv);
                }
            }
        });
    } else if (grad_output.dtype() == DType::Float64) {
        const double* go = get_data_ptr<const double>(grad_output);
        double* gi = get_data_ptr<double>(grad_input);
        queue.fill(gi, 0.0, in_size).wait();
        queue.parallel_for<AvgPool1dBackwardFloat64>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t count = 0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k;
                if (li >= 0 && li < L_in) count++;
            }
            double gv = count > 0 ? go[gid] / static_cast<double>(count) : 0.0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k;
                if (li >= 0 && li < L_in) {
                    sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> ar(gi[(n * C + c) * L_in + li]);
                    ar.fetch_add(gv);
                }
            }
        });
    } else if (grad_output.dtype() == DType::Float16) {
        const sycl::half* go = get_data_ptr<const sycl::half>(grad_output);
        sycl::half* gi = get_data_ptr<sycl::half>(grad_input);
        float* acc = sycl::malloc_device<float>(in_size, queue);
        queue.fill(acc, 0.0f, in_size).wait();
        queue.parallel_for<AvgPool1dBackwardFloat16>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t count = 0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k;
                if (li >= 0 && li < L_in) count++;
            }
            float gv = count > 0 ? static_cast<float>(go[gid]) / static_cast<float>(count) : 0.0f;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k;
                if (li >= 0 && li < L_in) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> ar(acc[(n * C + c) * L_in + li]);
                    ar.fetch_add(gv);
                }
            }
        });
        queue.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) { gi[i] = sycl::half(acc[i]); });
        sycl::free(acc, queue);
    } else if (grad_output.dtype() == DType::BFloat16) {
        const uint16_t* go = get_data_ptr<const uint16_t>(grad_output);
        uint16_t* gi = get_data_ptr<uint16_t>(grad_input);
        float* acc = sycl::malloc_device<float>(in_size, queue);
        queue.fill(acc, 0.0f, in_size).wait();
        queue.parallel_for<AvgPool1dBackwardBFloat16>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t count = 0;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k;
                if (li >= 0 && li < L_in) count++;
            }
            float gv = count > 0 ? bf16_to_f32(go[gid]) / static_cast<float>(count) : 0.0f;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t li = l * stride - padding + k;
                if (li >= 0 && li < L_in) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> ar(acc[(n * C + c) * L_in + li]);
                    ar.fetch_add(gv);
                }
            }
        });
        queue.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) { gi[i] = f32_to_bf16(acc[i]); });
        sycl::free(acc, queue);
    } else {
        throw std::runtime_error("Unsupported dtype for avgpool1d_backward");
    }
    return grad_input;
}

auto adaptive_maxpool1d_forward(const Tensor& input, int64_t output_size,
                                 sycl::queue& queue) -> std::vector<Tensor> {
    auto shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("AdaptiveMaxPool1d requires 3D input (N, C, L)");
    }
    const int64_t N = shape[0], C = shape[1], L_in = shape[2];
    const int64_t L_out = output_size;

    Tensor output({N, C, L_out}, input.dtype(), input.device());
    Tensor indices({N, C, L_out}, DType::Int64, input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const int64_t total = N * C * L_out;
        queue.parallel_for<AdaptiveMaxPool1dForwardFloat32>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t start = (l * L_in) / L_out;
            int64_t end = ((l + 1) * L_in) / L_out;
            float mx = -3.4028235e+38f; int64_t mi = start;
            for (int64_t i = start; i < end; ++i) {
                int64_t idx = (n * C + c) * L_in + i;
                float v = in_ptr[idx];
                if (v > mx) { mx = v; mi = idx; }
            }
            int64_t oi = (n * C + c) * L_out + l;
            out_ptr[oi] = mx; idx_ptr[oi] = mi;
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const int64_t total = N * C * L_out;
        queue.parallel_for<AdaptiveMaxPool1dForwardFloat64>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t start = (l * L_in) / L_out;
            int64_t end = ((l + 1) * L_in) / L_out;
            double mx = -1.7976931348623157e+308; int64_t mi = start;
            for (int64_t i = start; i < end; ++i) {
                int64_t idx = (n * C + c) * L_in + i;
                double v = in_ptr[idx];
                if (v > mx) { mx = v; mi = idx; }
            }
            int64_t oi = (n * C + c) * L_out + l;
            out_ptr[oi] = mx; idx_ptr[oi] = mi;
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const int64_t total = N * C * L_out;
        queue.parallel_for<AdaptiveMaxPool1dForwardFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t start = (l * L_in) / L_out;
            int64_t end = ((l + 1) * L_in) / L_out;
            float mx = -3.4028235e+38f; int64_t mi = start;
            for (int64_t i = start; i < end; ++i) {
                int64_t idx = (n * C + c) * L_in + i;
                float v = static_cast<float>(in_ptr[idx]);
                if (v > mx) { mx = v; mi = idx; }
            }
            int64_t oi = (n * C + c) * L_out + l;
            out_ptr[oi] = sycl::half(mx); idx_ptr[oi] = mi;
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const int64_t total = N * C * L_out;
        queue.parallel_for<AdaptiveMaxPool1dForwardBFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t start = (l * L_in) / L_out;
            int64_t end = ((l + 1) * L_in) / L_out;
            float mx = -3.4028235e+38f; int64_t mi = start;
            for (int64_t i = start; i < end; ++i) {
                int64_t idx = (n * C + c) * L_in + i;
                float v = bf16_to_f32(in_ptr[idx]);
                if (v > mx) { mx = v; mi = idx; }
            }
            int64_t oi = (n * C + c) * L_out + l;
            out_ptr[oi] = f32_to_bf16(mx); idx_ptr[oi] = mi;
        });
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_maxpool1d_forward");
    }
    return {output, indices};
}

auto adaptive_maxpool1d_backward(const Tensor& grad_output, const Tensor& indices,
                                  const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor {
    // Same as maxpool1d_backward - scatter grad using indices
    return maxpool1d_backward(grad_output, indices, input_shape, queue);
}

auto adaptive_avgpool1d_forward(const Tensor& input, int64_t output_size,
                                 sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("AdaptiveAvgPool1d requires 3D input (N, C, L)");
    }
    const int64_t N = shape[0], C = shape[1], L_in = shape[2];
    const int64_t L_out = output_size;

    Tensor output({N, C, L_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        const int64_t total = N * C * L_out;
        queue.parallel_for<AdaptiveAvgPool1dForwardFloat32>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t start = (l * L_in) / L_out;
            int64_t end = ((l + 1) * L_in) / L_out;
            float sum = 0.0f;
            for (int64_t i = start; i < end; ++i)
                sum += in_ptr[(n * C + c) * L_in + i];
            int64_t cnt = end - start;
            out_ptr[(n * C + c) * L_out + l] = cnt > 0 ? sum / static_cast<float>(cnt) : 0.0f;
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const int64_t total = N * C * L_out;
        queue.parallel_for<AdaptiveAvgPool1dForwardFloat64>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t start = (l * L_in) / L_out;
            int64_t end = ((l + 1) * L_in) / L_out;
            double sum = 0.0;
            for (int64_t i = start; i < end; ++i)
                sum += in_ptr[(n * C + c) * L_in + i];
            int64_t cnt = end - start;
            out_ptr[(n * C + c) * L_out + l] = cnt > 0 ? sum / static_cast<double>(cnt) : 0.0;
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        const int64_t total = N * C * L_out;
        queue.parallel_for<AdaptiveAvgPool1dForwardFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t start = (l * L_in) / L_out;
            int64_t end = ((l + 1) * L_in) / L_out;
            float sum = 0.0f;
            for (int64_t i = start; i < end; ++i)
                sum += static_cast<float>(in_ptr[(n * C + c) * L_in + i]);
            int64_t cnt = end - start;
            out_ptr[(n * C + c) * L_out + l] = sycl::half(cnt > 0 ? sum / static_cast<float>(cnt) : 0.0f);
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        const int64_t total = N * C * L_out;
        queue.parallel_for<AdaptiveAvgPool1dForwardBFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t start = (l * L_in) / L_out;
            int64_t end = ((l + 1) * L_in) / L_out;
            float sum = 0.0f;
            for (int64_t i = start; i < end; ++i)
                sum += bf16_to_f32(in_ptr[(n * C + c) * L_in + i]);
            int64_t cnt = end - start;
            out_ptr[(n * C + c) * L_out + l] = f32_to_bf16(cnt > 0 ? sum / static_cast<float>(cnt) : 0.0f);
        });
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool1d_forward");
    }
    return output;
}

auto adaptive_avgpool1d_backward(const Tensor& grad_output, const std::vector<int64_t>& input_shape,
                                  sycl::queue& queue) -> Tensor {
    if (input_shape.size() != 3) {
        throw std::invalid_argument("AdaptiveAvgPool1d backward requires 3D input_shape");
    }
    const int64_t N = input_shape[0], C = input_shape[1], L_in = input_shape[2];
    auto gs = grad_output.shape();
    const int64_t L_out = gs[2];

    Tensor grad_input({N, C, L_in}, grad_output.dtype(), grad_output.device());
    const int64_t in_size = N * C * L_in;
    const int64_t out_size = N * C * L_out;

    if (grad_output.dtype() == DType::Float32) {
        const float* go = get_data_ptr<const float>(grad_output);
        float* gi = get_data_ptr<float>(grad_input);
        queue.fill(gi, 0.0f, in_size).wait();
        queue.parallel_for<AdaptiveAvgPool1dBackwardFloat32>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t start = (l * L_in) / L_out;
            int64_t end = ((l + 1) * L_in) / L_out;
            int64_t cnt = end - start;
            float gv = cnt > 0 ? go[gid] / static_cast<float>(cnt) : 0.0f;
            for (int64_t i = start; i < end; ++i) {
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> ar(gi[(n * C + c) * L_in + i]);
                ar.fetch_add(gv);
            }
        });
    } else if (grad_output.dtype() == DType::Float64) {
        const double* go = get_data_ptr<const double>(grad_output);
        double* gi = get_data_ptr<double>(grad_input);
        queue.fill(gi, 0.0, in_size).wait();
        queue.parallel_for<AdaptiveAvgPool1dBackwardFloat64>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t start = (l * L_in) / L_out;
            int64_t end = ((l + 1) * L_in) / L_out;
            int64_t cnt = end - start;
            double gv = cnt > 0 ? go[gid] / static_cast<double>(cnt) : 0.0;
            for (int64_t i = start; i < end; ++i) {
                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> ar(gi[(n * C + c) * L_in + i]);
                ar.fetch_add(gv);
            }
        });
    } else if (grad_output.dtype() == DType::Float16) {
        const sycl::half* go = get_data_ptr<const sycl::half>(grad_output);
        sycl::half* gi = get_data_ptr<sycl::half>(grad_input);
        float* acc = sycl::malloc_device<float>(in_size, queue);
        queue.fill(acc, 0.0f, in_size).wait();
        queue.parallel_for<AdaptiveAvgPool1dBackwardFloat16>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t start = (l * L_in) / L_out;
            int64_t end = ((l + 1) * L_in) / L_out;
            int64_t cnt = end - start;
            float gv = cnt > 0 ? static_cast<float>(go[gid]) / static_cast<float>(cnt) : 0.0f;
            for (int64_t i = start; i < end; ++i) {
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> ar(acc[(n * C + c) * L_in + i]);
                ar.fetch_add(gv);
            }
        });
        queue.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) { gi[i] = sycl::half(acc[i]); });
        sycl::free(acc, queue);
    } else if (grad_output.dtype() == DType::BFloat16) {
        const uint16_t* go = get_data_ptr<const uint16_t>(grad_output);
        uint16_t* gi = get_data_ptr<uint16_t>(grad_input);
        float* acc = sycl::malloc_device<float>(in_size, queue);
        queue.fill(acc, 0.0f, in_size).wait();
        queue.parallel_for<AdaptiveAvgPool1dBackwardBFloat16>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid; const int64_t l = tmp % L_out; tmp /= L_out;
            const int64_t c = tmp % C; const int64_t n = tmp / C;
            int64_t start = (l * L_in) / L_out;
            int64_t end = ((l + 1) * L_in) / L_out;
            int64_t cnt = end - start;
            float gv = cnt > 0 ? bf16_to_f32(go[gid]) / static_cast<float>(cnt) : 0.0f;
            for (int64_t i = start; i < end; ++i) {
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> ar(acc[(n * C + c) * L_in + i]);
                ar.fetch_add(gv);
            }
        });
        queue.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) { gi[i] = f32_to_bf16(acc[i]); });
        sycl::free(acc, queue);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool1d_backward");
    }
    return grad_input;
}

// =============================================================================
// 3D Pooling Operations
// =============================================================================

auto maxpool3d_forward(const Tensor& input, const std::vector<int64_t>& kernel_size,
                       const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                       sycl::queue& queue) -> std::vector<Tensor> {
    auto shape = input.shape();
    if (shape.size() != 5) {
        throw std::invalid_argument("MaxPool3d requires 5D input (N, C, D, H, W)");
    }
    const int64_t N = shape[0], Ch = shape[1], D_in = shape[2], H_in = shape[3], W_in = shape[4];
    const int64_t kD = kernel_size[0], kH = kernel_size[1], kW = kernel_size[2];
    const int64_t sD = stride[0], sH = stride[1], sW = stride[2];
    const int64_t pD = padding[0], pH = padding[1], pW = padding[2];
    const int64_t D_out = (D_in + 2*pD - kD) / sD + 1;
    const int64_t H_out = (H_in + 2*pH - kH) / sH + 1;
    const int64_t W_out = (W_in + 2*pW - kW) / sW + 1;

    Tensor output({N, Ch, D_out, H_out, W_out}, input.dtype(), input.device());
    Tensor indices({N, Ch, D_out, H_out, W_out}, DType::Int64, input.device());
    const int64_t total = N * Ch * D_out * H_out * W_out;

    if (input.dtype() == DType::Float32) {
        const float* in_p = get_data_ptr<const float>(input);
        float* out_p = get_data_ptr<float>(output);
        int64_t* idx_p = get_data_ptr<int64_t>(indices);
        queue.parallel_for<MaxPool3dForwardFloat32>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            float mx = -3.4028235e+38f; int64_t mi = 0;
            for (int64_t kd = 0; kd < kD; ++kd) {
                int64_t di = d * sD - pD + kd;
                if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) {
                    int64_t hi = h * sH - pH + kh;
                    if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        int64_t wi = w * sW - pW + kw;
                        if (wi < 0 || wi >= W_in) continue;
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        float v = in_p[idx];
                        if (v > mx) { mx = v; mi = idx; }
                    }
                }
            }
            int64_t oi = ((n * Ch + c) * D_out + d) * H_out * W_out + h * W_out + w;
            out_p[oi] = mx; idx_p[oi] = mi;
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_p = get_data_ptr<const double>(input);
        double* out_p = get_data_ptr<double>(output);
        int64_t* idx_p = get_data_ptr<int64_t>(indices);
        queue.parallel_for<MaxPool3dForwardFloat64>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            double mx = -1.7976931348623157e+308; int64_t mi = 0;
            for (int64_t kd = 0; kd < kD; ++kd) {
                int64_t di = d * sD - pD + kd;
                if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) {
                    int64_t hi = h * sH - pH + kh;
                    if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        int64_t wi = w * sW - pW + kw;
                        if (wi < 0 || wi >= W_in) continue;
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        double v = in_p[idx];
                        if (v > mx) { mx = v; mi = idx; }
                    }
                }
            }
            int64_t oi = ((n * Ch + c) * D_out + d) * H_out * W_out + h * W_out + w;
            out_p[oi] = mx; idx_p[oi] = mi;
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_p = get_data_ptr<const sycl::half>(input);
        sycl::half* out_p = get_data_ptr<sycl::half>(output);
        int64_t* idx_p = get_data_ptr<int64_t>(indices);
        queue.parallel_for<MaxPool3dForwardFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            float mx = -3.4028235e+38f; int64_t mi = 0;
            for (int64_t kd = 0; kd < kD; ++kd) {
                int64_t di = d * sD - pD + kd;
                if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) {
                    int64_t hi = h * sH - pH + kh;
                    if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        int64_t wi = w * sW - pW + kw;
                        if (wi < 0 || wi >= W_in) continue;
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        float v = static_cast<float>(in_p[idx]);
                        if (v > mx) { mx = v; mi = idx; }
                    }
                }
            }
            int64_t oi = ((n * Ch + c) * D_out + d) * H_out * W_out + h * W_out + w;
            out_p[oi] = sycl::half(mx); idx_p[oi] = mi;
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_p = get_data_ptr<const uint16_t>(input);
        uint16_t* out_p = get_data_ptr<uint16_t>(output);
        int64_t* idx_p = get_data_ptr<int64_t>(indices);
        queue.parallel_for<MaxPool3dForwardBFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            float mx = -3.4028235e+38f; int64_t mi = 0;
            for (int64_t kd = 0; kd < kD; ++kd) {
                int64_t di = d * sD - pD + kd;
                if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) {
                    int64_t hi = h * sH - pH + kh;
                    if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        int64_t wi = w * sW - pW + kw;
                        if (wi < 0 || wi >= W_in) continue;
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        float v = bf16_to_f32(in_p[idx]);
                        if (v > mx) { mx = v; mi = idx; }
                    }
                }
            }
            int64_t oi = ((n * Ch + c) * D_out + d) * H_out * W_out + h * W_out + w;
            out_p[oi] = f32_to_bf16(mx); idx_p[oi] = mi;
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for maxpool3d_forward");
    }
    return {output, indices};
}

auto maxpool3d_backward(const Tensor& grad_output, const Tensor& indices,
                         const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor {
    if (input_shape.size() != 5) {
        throw std::invalid_argument("MaxPool3d backward requires 5D input_shape");
    }
    const int64_t in_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4];
    const int64_t out_size = grad_output.numel();

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    // Audit II.5: chain sequential kernels through event deps so we
    // only block once at the function epilogue instead of three round
    // trips per backward call.
    if (grad_output.dtype() == DType::Float32) {
        const float* go = get_data_ptr<const float>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);
        float* gi = get_data_ptr<float>(grad_input);
        sycl::event fill_ev = queue.fill(gi, 0.0f, in_size);
        sycl::event scatter_ev = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(fill_ev);
            cgh.parallel_for<MaxPool3dBackwardFloat32>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
                float gv = go[gid]; int64_t mi = idx[gid];
                if (mi >= 0 && mi < in_size) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> ar(gi[mi]);
                    ar.fetch_add(gv);
                }
            });
        });
        scatter_ev.wait();
    } else if (grad_output.dtype() == DType::Float64) {
        const double* go = get_data_ptr<const double>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);
        double* gi = get_data_ptr<double>(grad_input);
        sycl::event fill_ev = queue.fill(gi, 0.0, in_size);
        sycl::event scatter_ev = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(fill_ev);
            cgh.parallel_for<MaxPool3dBackwardFloat64>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
                double gv = go[gid]; int64_t mi = idx[gid];
                if (mi >= 0 && mi < in_size) {
                    sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> ar(gi[mi]);
                    ar.fetch_add(gv);
                }
            });
        });
        scatter_ev.wait();
    } else if (grad_output.dtype() == DType::Float16) {
        const sycl::half* go = get_data_ptr<const sycl::half>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);
        sycl::half* gi = get_data_ptr<sycl::half>(grad_input);
        float* acc = sycl::malloc_device<float>(in_size, queue);
        sycl::event fill_ev = queue.fill(acc, 0.0f, in_size);
        sycl::event scatter_ev = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(fill_ev);
            cgh.parallel_for<MaxPool3dBackwardFloat16>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
                float gv = static_cast<float>(go[gid]); int64_t mi = idx[gid];
                if (mi >= 0 && mi < in_size) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> ar(acc[mi]);
                    ar.fetch_add(gv);
                }
            });
        });
        sycl::event narrow_ev = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(scatter_ev);
            cgh.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) { gi[i] = sycl::half(acc[i]); });
        });
        narrow_ev.wait();
        sycl::free(acc, queue);
    } else if (grad_output.dtype() == DType::BFloat16) {
        const uint16_t* go = get_data_ptr<const uint16_t>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);
        uint16_t* gi = get_data_ptr<uint16_t>(grad_input);
        float* acc = sycl::malloc_device<float>(in_size, queue);
        sycl::event fill_ev = queue.fill(acc, 0.0f, in_size);
        sycl::event scatter_ev = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(fill_ev);
            cgh.parallel_for<MaxPool3dBackwardBFloat16>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
                float gv = bf16_to_f32(go[gid]); int64_t mi = idx[gid];
                if (mi >= 0 && mi < in_size) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> ar(acc[mi]);
                    ar.fetch_add(gv);
                }
            });
        });
        sycl::event narrow_ev = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(scatter_ev);
            cgh.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) { gi[i] = f32_to_bf16(acc[i]); });
        });
        narrow_ev.wait();
        sycl::free(acc, queue);
    } else {
        throw std::runtime_error("Unsupported dtype for maxpool3d_backward");
    }
    return grad_input;
}

auto avgpool3d_forward(const Tensor& input, const std::vector<int64_t>& kernel_size,
                       const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                       sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 5) {
        throw std::invalid_argument("AvgPool3d requires 5D input (N, C, D, H, W)");
    }
    const int64_t N = shape[0], Ch = shape[1], D_in = shape[2], H_in = shape[3], W_in = shape[4];
    const int64_t kD = kernel_size[0], kH = kernel_size[1], kW = kernel_size[2];
    const int64_t sD = stride[0], sH = stride[1], sW = stride[2];
    const int64_t pD = padding[0], pH = padding[1], pW = padding[2];
    const int64_t D_out = (D_in + 2*pD - kD) / sD + 1;
    const int64_t H_out = (H_in + 2*pH - kH) / sH + 1;
    const int64_t W_out = (W_in + 2*pW - kW) / sW + 1;

    Tensor output({N, Ch, D_out, H_out, W_out}, input.dtype(), input.device());
    const int64_t total = N * Ch * D_out * H_out * W_out;

    if (input.dtype() == DType::Float32) {
        const float* in_p = get_data_ptr<const float>(input);
        float* out_p = get_data_ptr<float>(output);
        queue.parallel_for<AvgPool3dForwardFloat32>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            float sum = 0.0f; int64_t count = 0;
            for (int64_t kd = 0; kd < kD; ++kd) {
                int64_t di = d * sD - pD + kd; if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) {
                    int64_t hi = h * sH - pH + kh; if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        int64_t wi = w * sW - pW + kw; if (wi < 0 || wi >= W_in) continue;
                        sum += in_p[((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi];
                        count++;
                    }
                }
            }
            out_p[gid] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_p = get_data_ptr<const double>(input);
        double* out_p = get_data_ptr<double>(output);
        queue.parallel_for<AvgPool3dForwardFloat64>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            double sum = 0.0; int64_t count = 0;
            for (int64_t kd = 0; kd < kD; ++kd) {
                int64_t di = d * sD - pD + kd; if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) {
                    int64_t hi = h * sH - pH + kh; if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        int64_t wi = w * sW - pW + kw; if (wi < 0 || wi >= W_in) continue;
                        sum += in_p[((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi];
                        count++;
                    }
                }
            }
            out_p[gid] = count > 0 ? sum / static_cast<double>(count) : 0.0;
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_p = get_data_ptr<const sycl::half>(input);
        sycl::half* out_p = get_data_ptr<sycl::half>(output);
        queue.parallel_for<AvgPool3dForwardFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            float sum = 0.0f; int64_t count = 0;
            for (int64_t kd = 0; kd < kD; ++kd) {
                int64_t di = d * sD - pD + kd; if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) {
                    int64_t hi = h * sH - pH + kh; if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        int64_t wi = w * sW - pW + kw; if (wi < 0 || wi >= W_in) continue;
                        sum += static_cast<float>(in_p[((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi]);
                        count++;
                    }
                }
            }
            out_p[gid] = sycl::half(count > 0 ? sum / static_cast<float>(count) : 0.0f);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_p = get_data_ptr<const uint16_t>(input);
        uint16_t* out_p = get_data_ptr<uint16_t>(output);
        queue.parallel_for<AvgPool3dForwardBFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            float sum = 0.0f; int64_t count = 0;
            for (int64_t kd = 0; kd < kD; ++kd) {
                int64_t di = d * sD - pD + kd; if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) {
                    int64_t hi = h * sH - pH + kh; if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        int64_t wi = w * sW - pW + kw; if (wi < 0 || wi >= W_in) continue;
                        sum += bf16_to_f32(in_p[((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi]);
                        count++;
                    }
                }
            }
            out_p[gid] = f32_to_bf16(count > 0 ? sum / static_cast<float>(count) : 0.0f);
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for avgpool3d_forward");
    }
    return output;
}

auto avgpool3d_backward(const Tensor& grad_output, const std::vector<int64_t>& kernel_size,
                         const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                         const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor {
    if (input_shape.size() != 5) {
        throw std::invalid_argument("AvgPool3d backward requires 5D input_shape");
    }
    const int64_t N = input_shape[0], Ch = input_shape[1], D_in = input_shape[2], H_in = input_shape[3], W_in = input_shape[4];
    const int64_t kD = kernel_size[0], kH = kernel_size[1], kW = kernel_size[2];
    const int64_t sD = stride[0], sH = stride[1], sW = stride[2];
    const int64_t pD = padding[0], pH = padding[1], pW = padding[2];
    auto gs = grad_output.shape();
    const int64_t D_out = gs[2], H_out = gs[3], W_out = gs[4];

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
    const int64_t in_size = N * Ch * D_in * H_in * W_in;
    const int64_t out_size = N * Ch * D_out * H_out * W_out;

    if (grad_output.dtype() == DType::Float32) {
        const float* go = get_data_ptr<const float>(grad_output);
        float* gi = get_data_ptr<float>(grad_input);
        queue.fill(gi, 0.0f, in_size).wait();
        queue.parallel_for<AvgPool3dBackwardFloat32>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t count = 0;
            for (int64_t kd = 0; kd < kD; ++kd) { int64_t di = d*sD - pD + kd; if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) { int64_t hi = h*sH - pH + kh; if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) { int64_t wi = w*sW - pW + kw; if (wi < 0 || wi >= W_in) continue; count++; }}}
            float gv = count > 0 ? go[gid] / static_cast<float>(count) : 0.0f;
            for (int64_t kd = 0; kd < kD; ++kd) { int64_t di = d*sD - pD + kd; if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) { int64_t hi = h*sH - pH + kh; if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) { int64_t wi = w*sW - pW + kw; if (wi < 0 || wi >= W_in) continue;
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space> ar(gi[idx]);
                        ar.fetch_add(gv);
                    }}}
        }).wait();
    } else if (grad_output.dtype() == DType::Float64) {
        const double* go = get_data_ptr<const double>(grad_output);
        double* gi = get_data_ptr<double>(grad_input);
        queue.fill(gi, 0.0, in_size).wait();
        queue.parallel_for<AvgPool3dBackwardFloat64>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t count = 0;
            for (int64_t kd = 0; kd < kD; ++kd) { int64_t di = d*sD - pD + kd; if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) { int64_t hi = h*sH - pH + kh; if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) { int64_t wi = w*sW - pW + kw; if (wi < 0 || wi >= W_in) continue; count++; }}}
            double gv = count > 0 ? go[gid] / static_cast<double>(count) : 0.0;
            for (int64_t kd = 0; kd < kD; ++kd) { int64_t di = d*sD - pD + kd; if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) { int64_t hi = h*sH - pH + kh; if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) { int64_t wi = w*sW - pW + kw; if (wi < 0 || wi >= W_in) continue;
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space> ar(gi[idx]);
                        ar.fetch_add(gv);
                    }}}
        }).wait();
    } else if (grad_output.dtype() == DType::Float16) {
        const sycl::half* go = get_data_ptr<const sycl::half>(grad_output);
        sycl::half* gi = get_data_ptr<sycl::half>(grad_input);
        float* acc = sycl::malloc_device<float>(in_size, queue);
        queue.fill(acc, 0.0f, in_size).wait();
        queue.parallel_for<AvgPool3dBackwardFloat16>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t count = 0;
            for (int64_t kd = 0; kd < kD; ++kd) { int64_t di = d*sD - pD + kd; if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) { int64_t hi = h*sH - pH + kh; if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) { int64_t wi = w*sW - pW + kw; if (wi < 0 || wi >= W_in) continue; count++; }}}
            float gv = count > 0 ? static_cast<float>(go[gid]) / static_cast<float>(count) : 0.0f;
            for (int64_t kd = 0; kd < kD; ++kd) { int64_t di = d*sD - pD + kd; if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) { int64_t hi = h*sH - pH + kh; if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) { int64_t wi = w*sW - pW + kw; if (wi < 0 || wi >= W_in) continue;
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space> ar(acc[idx]);
                        ar.fetch_add(gv);
                    }}}
        }).wait();
        queue.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) { gi[i] = sycl::half(acc[i]); }).wait();
        sycl::free(acc, queue);
    } else if (grad_output.dtype() == DType::BFloat16) {
        const uint16_t* go = get_data_ptr<const uint16_t>(grad_output);
        uint16_t* gi = get_data_ptr<uint16_t>(grad_input);
        float* acc = sycl::malloc_device<float>(in_size, queue);
        queue.fill(acc, 0.0f, in_size).wait();
        queue.parallel_for<AvgPool3dBackwardBFloat16>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t count = 0;
            for (int64_t kd = 0; kd < kD; ++kd) { int64_t di = d*sD - pD + kd; if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) { int64_t hi = h*sH - pH + kh; if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) { int64_t wi = w*sW - pW + kw; if (wi < 0 || wi >= W_in) continue; count++; }}}
            float gv = count > 0 ? bf16_to_f32(go[gid]) / static_cast<float>(count) : 0.0f;
            for (int64_t kd = 0; kd < kD; ++kd) { int64_t di = d*sD - pD + kd; if (di < 0 || di >= D_in) continue;
                for (int64_t kh = 0; kh < kH; ++kh) { int64_t hi = h*sH - pH + kh; if (hi < 0 || hi >= H_in) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) { int64_t wi = w*sW - pW + kw; if (wi < 0 || wi >= W_in) continue;
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space> ar(acc[idx]);
                        ar.fetch_add(gv);
                    }}}
        }).wait();
        queue.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) { gi[i] = f32_to_bf16(acc[i]); }).wait();
        sycl::free(acc, queue);
    } else {
        throw std::runtime_error("Unsupported dtype for avgpool3d_backward");
    }
    return grad_input;
}

auto adaptive_maxpool3d_forward(const Tensor& input, const std::vector<int64_t>& output_size,
                                 sycl::queue& queue) -> std::vector<Tensor> {
    auto shape = input.shape();
    if (shape.size() != 5) {
        throw std::invalid_argument("AdaptiveMaxPool3d requires 5D input (N, C, D, H, W)");
    }
    const int64_t N = shape[0], Ch = shape[1], D_in = shape[2], H_in = shape[3], W_in = shape[4];
    const int64_t D_out = output_size[0], H_out = output_size[1], W_out = output_size[2];

    Tensor output({N, Ch, D_out, H_out, W_out}, input.dtype(), input.device());
    Tensor indices({N, Ch, D_out, H_out, W_out}, DType::Int64, input.device());
    const int64_t total = N * Ch * D_out * H_out * W_out;

    if (input.dtype() == DType::Float32) {
        const float* in_p = get_data_ptr<const float>(input);
        float* out_p = get_data_ptr<float>(output);
        int64_t* idx_p = get_data_ptr<int64_t>(indices);
        queue.parallel_for<AdaptiveMaxPool3dForwardFloat32>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t ds = (d * D_in) / D_out, de = ((d+1) * D_in) / D_out;
            int64_t hs = (h * H_in) / H_out, he = ((h+1) * H_in) / H_out;
            int64_t ws = (w * W_in) / W_out, we = ((w+1) * W_in) / W_out;
            float mx = -3.4028235e+38f; int64_t mi = 0;
            for (int64_t di = ds; di < de; ++di)
                for (int64_t hi = hs; hi < he; ++hi)
                    for (int64_t wi = ws; wi < we; ++wi) {
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        float v = in_p[idx];
                        if (v > mx) { mx = v; mi = idx; }
                    }
            out_p[gid] = mx; idx_p[gid] = mi;
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_p = get_data_ptr<const double>(input);
        double* out_p = get_data_ptr<double>(output);
        int64_t* idx_p = get_data_ptr<int64_t>(indices);
        queue.parallel_for<AdaptiveMaxPool3dForwardFloat64>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t ds = (d * D_in) / D_out, de = ((d+1) * D_in) / D_out;
            int64_t hs = (h * H_in) / H_out, he = ((h+1) * H_in) / H_out;
            int64_t ws = (w * W_in) / W_out, we = ((w+1) * W_in) / W_out;
            double mx = -1.7976931348623157e+308; int64_t mi = 0;
            for (int64_t di = ds; di < de; ++di)
                for (int64_t hi = hs; hi < he; ++hi)
                    for (int64_t wi = ws; wi < we; ++wi) {
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        double v = in_p[idx];
                        if (v > mx) { mx = v; mi = idx; }
                    }
            out_p[gid] = mx; idx_p[gid] = mi;
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_p = get_data_ptr<const sycl::half>(input);
        sycl::half* out_p = get_data_ptr<sycl::half>(output);
        int64_t* idx_p = get_data_ptr<int64_t>(indices);
        queue.parallel_for<AdaptiveMaxPool3dForwardFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t ds = (d * D_in) / D_out, de = ((d+1) * D_in) / D_out;
            int64_t hs = (h * H_in) / H_out, he = ((h+1) * H_in) / H_out;
            int64_t ws = (w * W_in) / W_out, we = ((w+1) * W_in) / W_out;
            float mx = -3.4028235e+38f; int64_t mi = 0;
            for (int64_t di = ds; di < de; ++di)
                for (int64_t hi = hs; hi < he; ++hi)
                    for (int64_t wi = ws; wi < we; ++wi) {
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        float v = static_cast<float>(in_p[idx]);
                        if (v > mx) { mx = v; mi = idx; }
                    }
            out_p[gid] = sycl::half(mx); idx_p[gid] = mi;
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_p = get_data_ptr<const uint16_t>(input);
        uint16_t* out_p = get_data_ptr<uint16_t>(output);
        int64_t* idx_p = get_data_ptr<int64_t>(indices);
        queue.parallel_for<AdaptiveMaxPool3dForwardBFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t ds = (d * D_in) / D_out, de = ((d+1) * D_in) / D_out;
            int64_t hs = (h * H_in) / H_out, he = ((h+1) * H_in) / H_out;
            int64_t ws = (w * W_in) / W_out, we = ((w+1) * W_in) / W_out;
            float mx = -3.4028235e+38f; int64_t mi = 0;
            for (int64_t di = ds; di < de; ++di)
                for (int64_t hi = hs; hi < he; ++hi)
                    for (int64_t wi = ws; wi < we; ++wi) {
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        float v = bf16_to_f32(in_p[idx]);
                        if (v > mx) { mx = v; mi = idx; }
                    }
            out_p[gid] = f32_to_bf16(mx); idx_p[gid] = mi;
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_maxpool3d_forward");
    }
    return {output, indices};
}

auto adaptive_maxpool3d_backward(const Tensor& grad_output, const Tensor& indices,
                                  const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor {
    return maxpool3d_backward(grad_output, indices, input_shape, queue);
}

auto adaptive_avgpool3d_forward(const Tensor& input, const std::vector<int64_t>& output_size,
                                 sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 5) {
        throw std::invalid_argument("AdaptiveAvgPool3d requires 5D input (N, C, D, H, W)");
    }
    const int64_t N = shape[0], Ch = shape[1], D_in = shape[2], H_in = shape[3], W_in = shape[4];
    const int64_t D_out = output_size[0], H_out = output_size[1], W_out = output_size[2];

    Tensor output({N, Ch, D_out, H_out, W_out}, input.dtype(), input.device());
    const int64_t total = N * Ch * D_out * H_out * W_out;

    if (input.dtype() == DType::Float32) {
        const float* in_p = get_data_ptr<const float>(input);
        float* out_p = get_data_ptr<float>(output);
        queue.parallel_for<AdaptiveAvgPool3dForwardFloat32>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t ds = (d * D_in) / D_out, de = ((d+1) * D_in) / D_out;
            int64_t hs = (h * H_in) / H_out, he = ((h+1) * H_in) / H_out;
            int64_t ws = (w * W_in) / W_out, we = ((w+1) * W_in) / W_out;
            float sum = 0.0f; int64_t cnt = 0;
            for (int64_t di = ds; di < de; ++di)
                for (int64_t hi = hs; hi < he; ++hi)
                    for (int64_t wi = ws; wi < we; ++wi) {
                        sum += in_p[((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi]; cnt++;
                    }
            out_p[gid] = cnt > 0 ? sum / static_cast<float>(cnt) : 0.0f;
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_p = get_data_ptr<const double>(input);
        double* out_p = get_data_ptr<double>(output);
        queue.parallel_for<AdaptiveAvgPool3dForwardFloat64>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t ds = (d * D_in) / D_out, de = ((d+1) * D_in) / D_out;
            int64_t hs = (h * H_in) / H_out, he = ((h+1) * H_in) / H_out;
            int64_t ws = (w * W_in) / W_out, we = ((w+1) * W_in) / W_out;
            double sum = 0.0; int64_t cnt = 0;
            for (int64_t di = ds; di < de; ++di)
                for (int64_t hi = hs; hi < he; ++hi)
                    for (int64_t wi = ws; wi < we; ++wi) {
                        sum += in_p[((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi]; cnt++;
                    }
            out_p[gid] = cnt > 0 ? sum / static_cast<double>(cnt) : 0.0;
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_p = get_data_ptr<const sycl::half>(input);
        sycl::half* out_p = get_data_ptr<sycl::half>(output);
        queue.parallel_for<AdaptiveAvgPool3dForwardFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t ds = (d * D_in) / D_out, de = ((d+1) * D_in) / D_out;
            int64_t hs = (h * H_in) / H_out, he = ((h+1) * H_in) / H_out;
            int64_t ws = (w * W_in) / W_out, we = ((w+1) * W_in) / W_out;
            float sum = 0.0f; int64_t cnt = 0;
            for (int64_t di = ds; di < de; ++di)
                for (int64_t hi = hs; hi < he; ++hi)
                    for (int64_t wi = ws; wi < we; ++wi) {
                        sum += static_cast<float>(in_p[((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi]); cnt++;
                    }
            out_p[gid] = sycl::half(cnt > 0 ? sum / static_cast<float>(cnt) : 0.0f);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_p = get_data_ptr<const uint16_t>(input);
        uint16_t* out_p = get_data_ptr<uint16_t>(output);
        queue.parallel_for<AdaptiveAvgPool3dForwardBFloat16>(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t ds = (d * D_in) / D_out, de = ((d+1) * D_in) / D_out;
            int64_t hs = (h * H_in) / H_out, he = ((h+1) * H_in) / H_out;
            int64_t ws = (w * W_in) / W_out, we = ((w+1) * W_in) / W_out;
            float sum = 0.0f; int64_t cnt = 0;
            for (int64_t di = ds; di < de; ++di)
                for (int64_t hi = hs; hi < he; ++hi)
                    for (int64_t wi = ws; wi < we; ++wi) {
                        sum += bf16_to_f32(in_p[((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi]); cnt++;
                    }
            out_p[gid] = f32_to_bf16(cnt > 0 ? sum / static_cast<float>(cnt) : 0.0f);
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool3d_forward");
    }
    return output;
}

auto adaptive_avgpool3d_backward(const Tensor& grad_output, const std::vector<int64_t>& input_shape,
                                  sycl::queue& queue) -> Tensor {
    if (input_shape.size() != 5) {
        throw std::invalid_argument("AdaptiveAvgPool3d backward requires 5D input_shape");
    }
    const int64_t N = input_shape[0], Ch = input_shape[1], D_in = input_shape[2], H_in = input_shape[3], W_in = input_shape[4];
    auto gs = grad_output.shape();
    const int64_t D_out = gs[2], H_out = gs[3], W_out = gs[4];

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
    const int64_t in_size = N * Ch * D_in * H_in * W_in;
    const int64_t out_size = N * Ch * D_out * H_out * W_out;

    if (grad_output.dtype() == DType::Float32) {
        const float* go = get_data_ptr<const float>(grad_output);
        float* gi = get_data_ptr<float>(grad_input);
        queue.fill(gi, 0.0f, in_size).wait();
        queue.parallel_for<AdaptiveAvgPool3dBackwardFloat32>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t ds = (d * D_in) / D_out, de = ((d+1) * D_in) / D_out;
            int64_t hs = (h * H_in) / H_out, he = ((h+1) * H_in) / H_out;
            int64_t ws = (w * W_in) / W_out, we = ((w+1) * W_in) / W_out;
            int64_t cnt = (de - ds) * (he - hs) * (we - ws);
            float gv = cnt > 0 ? go[gid] / static_cast<float>(cnt) : 0.0f;
            for (int64_t di = ds; di < de; ++di)
                for (int64_t hi = hs; hi < he; ++hi)
                    for (int64_t wi = ws; wi < we; ++wi) {
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space> ar(gi[idx]);
                        ar.fetch_add(gv);
                    }
        }).wait();
    } else if (grad_output.dtype() == DType::Float64) {
        const double* go = get_data_ptr<const double>(grad_output);
        double* gi = get_data_ptr<double>(grad_input);
        queue.fill(gi, 0.0, in_size).wait();
        queue.parallel_for<AdaptiveAvgPool3dBackwardFloat64>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t ds = (d * D_in) / D_out, de = ((d+1) * D_in) / D_out;
            int64_t hs = (h * H_in) / H_out, he = ((h+1) * H_in) / H_out;
            int64_t ws = (w * W_in) / W_out, we = ((w+1) * W_in) / W_out;
            int64_t cnt = (de - ds) * (he - hs) * (we - ws);
            double gv = cnt > 0 ? go[gid] / static_cast<double>(cnt) : 0.0;
            for (int64_t di = ds; di < de; ++di)
                for (int64_t hi = hs; hi < he; ++hi)
                    for (int64_t wi = ws; wi < we; ++wi) {
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space> ar(gi[idx]);
                        ar.fetch_add(gv);
                    }
        }).wait();
    } else if (grad_output.dtype() == DType::Float16) {
        const sycl::half* go = get_data_ptr<const sycl::half>(grad_output);
        sycl::half* gi = get_data_ptr<sycl::half>(grad_input);
        float* acc = sycl::malloc_device<float>(in_size, queue);
        queue.fill(acc, 0.0f, in_size).wait();
        queue.parallel_for<AdaptiveAvgPool3dBackwardFloat16>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t ds = (d * D_in) / D_out, de = ((d+1) * D_in) / D_out;
            int64_t hs = (h * H_in) / H_out, he = ((h+1) * H_in) / H_out;
            int64_t ws = (w * W_in) / W_out, we = ((w+1) * W_in) / W_out;
            int64_t cnt = (de - ds) * (he - hs) * (we - ws);
            float gv = cnt > 0 ? static_cast<float>(go[gid]) / static_cast<float>(cnt) : 0.0f;
            for (int64_t di = ds; di < de; ++di)
                for (int64_t hi = hs; hi < he; ++hi)
                    for (int64_t wi = ws; wi < we; ++wi) {
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space> ar(acc[idx]);
                        ar.fetch_add(gv);
                    }
        }).wait();
        queue.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) { gi[i] = sycl::half(acc[i]); }).wait();
        sycl::free(acc, queue);
    } else if (grad_output.dtype() == DType::BFloat16) {
        const uint16_t* go = get_data_ptr<const uint16_t>(grad_output);
        uint16_t* gi = get_data_ptr<uint16_t>(grad_input);
        float* acc = sycl::malloc_device<float>(in_size, queue);
        queue.fill(acc, 0.0f, in_size).wait();
        queue.parallel_for<AdaptiveAvgPool3dBackwardBFloat16>(sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t tmp = gid;
            const int64_t w = tmp % W_out; tmp /= W_out;
            const int64_t h = tmp % H_out; tmp /= H_out;
            const int64_t d = tmp % D_out; tmp /= D_out;
            const int64_t c = tmp % Ch; const int64_t n = tmp / Ch;
            int64_t ds = (d * D_in) / D_out, de = ((d+1) * D_in) / D_out;
            int64_t hs = (h * H_in) / H_out, he = ((h+1) * H_in) / H_out;
            int64_t ws = (w * W_in) / W_out, we = ((w+1) * W_in) / W_out;
            int64_t cnt = (de - ds) * (he - hs) * (we - ws);
            float gv = cnt > 0 ? bf16_to_f32(go[gid]) / static_cast<float>(cnt) : 0.0f;
            for (int64_t di = ds; di < de; ++di)
                for (int64_t hi = hs; hi < he; ++hi)
                    for (int64_t wi = ws; wi < we; ++wi) {
                        int64_t idx = ((n * Ch + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space> ar(acc[idx]);
                        ar.fetch_add(gv);
                    }
        }).wait();
        queue.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) { gi[i] = f32_to_bf16(acc[i]); }).wait();
        sycl::free(acc, queue);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool3d_backward");
    }
    return grad_input;
}

// ============================================================================
// Fractional Max Pool 2D — SYCL kernel name classes
// ============================================================================
class FractionalMaxPool2dForwardFloat32 {};
class FractionalMaxPool2dForwardFloat64 {};
class FractionalMaxPool2dBackwardFloat32 {};
class FractionalMaxPool2dBackwardFloat64 {};

// ============================================================================
// Fractional Max Pool 3D — SYCL kernel name classes
// ============================================================================
class FractionalMaxPool3dForwardFloat32 {};
class FractionalMaxPool3dForwardFloat64 {};
class FractionalMaxPool3dBackwardFloat32 {};
class FractionalMaxPool3dBackwardFloat64 {};

// ============================================================================
// Max Unpool 2D/3D — SYCL kernel name classes
// ============================================================================
class MaxUnpool2dForwardFloat32 {};
class MaxUnpool2dForwardFloat64 {};
class MaxUnpool2dBackwardFloat32 {};
class MaxUnpool2dBackwardFloat64 {};
class MaxUnpool3dForwardFloat32 {};
class MaxUnpool3dForwardFloat64 {};
class MaxUnpool3dBackwardFloat32 {};
class MaxUnpool3dBackwardFloat64 {};
// Zero-fill kernel names
class ZeroFillFloat32 {};
class ZeroFillFloat64 {};

// ============================================================================
// Fractional Max Pool 2D Forward
// ============================================================================

auto fractional_maxpool2d_forward_kernel(const Tensor& input,
                                         int64_t out_h, int64_t out_w,
                                         const Tensor* random_samples,
                                         sycl::queue& queue)
    -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("FractionalMaxPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H = shape[2];
    const int64_t W = shape[3];

    Tensor output({N, C, out_h, out_w}, input.dtype(), input.device());
    Tensor indices({N, C, out_h, out_w}, DType::Int64, input.device());

    const int64_t total_out = N * C * out_h * out_w;
    if (total_out == 0) return {output, indices};

    // Copy random samples to device if provided, otherwise use default 0.5
    float* dev_samples = nullptr;
    const int64_t num_samples = N * C * 2;
    if (random_samples && random_samples->numel() > 0) {
        dev_samples = get_data_ptr<float>(*random_samples);
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const float* sp = dev_samples;

        queue.parallel_for<FractionalMaxPool2dForwardFloat32>(
            sycl::range<1>(total_out), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t ow_idx = linear % out_w; linear /= out_w;
            int64_t oh_idx = linear % out_h; linear /= out_h;
            int64_t c = linear % C;
            int64_t n = linear / C;

            float sample_h = sp ? sp[(n * C + c) * 2 + 0] : 0.5f;
            float sample_w = sp ? sp[(n * C + c) * 2 + 1] : 0.5f;

            float ratio_h = static_cast<float>(H) / static_cast<float>(out_h);
            float ratio_w = static_cast<float>(W) / static_cast<float>(out_w);

            int64_t h_start = static_cast<int64_t>(sycl::floor(
                (oh_idx + sample_h) * ratio_h - sample_h));
            int64_t h_end = static_cast<int64_t>(sycl::floor(
                (oh_idx + 1 + sample_h) * ratio_h - sample_h));
            int64_t w_start = static_cast<int64_t>(sycl::floor(
                (ow_idx + sample_w) * ratio_w - sample_w));
            int64_t w_end = static_cast<int64_t>(sycl::floor(
                (ow_idx + 1 + sample_w) * ratio_w - sample_w));

            h_start = sycl::max(h_start, int64_t{0});
            h_end = sycl::min(h_end, H);
            w_start = sycl::max(w_start, int64_t{0});
            w_end = sycl::min(w_end, W);
            if (h_end <= h_start) h_end = sycl::min(h_start + 1, H);
            if (w_end <= w_start) w_end = sycl::min(w_start + 1, W);

            float max_val = -3.4028235e+38f;
            int64_t max_idx = h_start * W + w_start;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    float val = in_ptr[in_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = h * W + w;
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * out_h + oh_idx) * out_w + ow_idx;
            out_ptr[out_idx] = max_val;
            idx_ptr[out_idx] = max_idx;
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const float* sp = dev_samples;

        queue.parallel_for<FractionalMaxPool2dForwardFloat64>(
            sycl::range<1>(total_out), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t ow_idx = linear % out_w; linear /= out_w;
            int64_t oh_idx = linear % out_h; linear /= out_h;
            int64_t c = linear % C;
            int64_t n = linear / C;

            float sample_h = sp ? sp[(n * C + c) * 2 + 0] : 0.5f;
            float sample_w = sp ? sp[(n * C + c) * 2 + 1] : 0.5f;

            float ratio_h = static_cast<float>(H) / static_cast<float>(out_h);
            float ratio_w = static_cast<float>(W) / static_cast<float>(out_w);

            int64_t h_start = static_cast<int64_t>(sycl::floor(
                (oh_idx + sample_h) * ratio_h - sample_h));
            int64_t h_end = static_cast<int64_t>(sycl::floor(
                (oh_idx + 1 + sample_h) * ratio_h - sample_h));
            int64_t w_start = static_cast<int64_t>(sycl::floor(
                (ow_idx + sample_w) * ratio_w - sample_w));
            int64_t w_end = static_cast<int64_t>(sycl::floor(
                (ow_idx + 1 + sample_w) * ratio_w - sample_w));

            h_start = sycl::max(h_start, int64_t{0});
            h_end = sycl::min(h_end, H);
            w_start = sycl::max(w_start, int64_t{0});
            w_end = sycl::min(w_end, W);
            if (h_end <= h_start) h_end = sycl::min(h_start + 1, H);
            if (w_end <= w_start) w_end = sycl::min(w_start + 1, W);

            double max_val = -1.7976931348623157e+308;
            int64_t max_idx = h_start * W + w_start;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    double val = in_ptr[in_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = h * W + w;
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * out_h + oh_idx) * out_w + ow_idx;
            out_ptr[out_idx] = max_val;
            idx_ptr[out_idx] = max_idx;
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const float* sp = dev_samples;

        queue.parallel_for<class FractionalMaxPool2dForwardFloat16>(
            sycl::range<1>(total_out), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t ow_idx = linear % out_w; linear /= out_w;
            int64_t oh_idx = linear % out_h; linear /= out_h;
            int64_t c = linear % C;
            int64_t n = linear / C;

            float sample_h = sp ? sp[(n * C + c) * 2 + 0] : 0.5f;
            float sample_w = sp ? sp[(n * C + c) * 2 + 1] : 0.5f;
            float ratio_h = static_cast<float>(H) / static_cast<float>(out_h);
            float ratio_w = static_cast<float>(W) / static_cast<float>(out_w);

            int64_t h_start = static_cast<int64_t>(sycl::floor(
                (oh_idx + sample_h) * ratio_h - sample_h));
            int64_t h_end = static_cast<int64_t>(sycl::floor(
                (oh_idx + 1 + sample_h) * ratio_h - sample_h));
            int64_t w_start = static_cast<int64_t>(sycl::floor(
                (ow_idx + sample_w) * ratio_w - sample_w));
            int64_t w_end = static_cast<int64_t>(sycl::floor(
                (ow_idx + 1 + sample_w) * ratio_w - sample_w));

            h_start = sycl::max(h_start, int64_t{0});
            h_end = sycl::min(h_end, H);
            w_start = sycl::max(w_start, int64_t{0});
            w_end = sycl::min(w_end, W);
            if (h_end <= h_start) h_end = sycl::min(h_start + 1, H);
            if (w_end <= w_start) w_end = sycl::min(w_start + 1, W);

            float max_val = -3.4028235e+38f;
            int64_t max_idx = h_start * W + w_start;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    float val = static_cast<float>(in_ptr[in_idx]);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = h * W + w;
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * out_h + oh_idx) * out_w + ow_idx;
            out_ptr[out_idx] = sycl::half(max_val);
            idx_ptr[out_idx] = max_idx;
        }).wait();
    } else {
        throw std::runtime_error("FractionalMaxPool2d forward OneAPI: unsupported dtype (Float32/Float64/Float16 supported)");
    }

    return {output, indices};
}

// ============================================================================
// Fractional Max Pool 2D Backward
// ============================================================================

auto fractional_maxpool2d_backward_kernel(const Tensor& grad_output,
                                          const Tensor& indices,
                                          const std::vector<int64_t>& input_shape,
                                          sycl::queue& queue) -> Tensor {
    const int64_t N = input_shape[0];
    const int64_t C = input_shape[1];
    const int64_t H = input_shape[2];
    const int64_t W = input_shape[3];
    auto grad_shape = grad_output.shape();
    const int64_t out_h = grad_shape[2];
    const int64_t out_w = grad_shape[3];

    const int64_t in_spatial = H * W;
    const int64_t out_spatial = out_h * out_w;
    const int64_t in_size = N * C * in_spatial;
    const int64_t out_size = N * C * out_spatial;

    Tensor grad_input({N, C, H, W}, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        float* gi = get_data_ptr<float>(grad_input);
        const float* go = get_data_ptr<const float>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);

        // Zero grad_input
        queue.fill(gi, 0.0f, in_size).wait();

        // Scatter gradients using atomicAdd
        queue.parallel_for<FractionalMaxPool2dBackwardFloat32>(
            sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / out_spatial;
            int64_t max_idx = idx[linear];
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ar(gi[nc * in_spatial + max_idx]);
            ar.fetch_add(go[linear]);
        }).wait();
    } else if (grad_output.dtype() == DType::Float64) {
        double* gi = get_data_ptr<double>(grad_input);
        const double* go = get_data_ptr<const double>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);

        // For Float64, use float accumulator + atomic since some devices lack f64 atomics.
        // No need to pre-zero `gi`: the acc->gi copy below writes every element.
        float* acc = sycl::malloc_device<float>(in_size, queue);
        queue.fill(acc, 0.0f, in_size).wait();

        queue.parallel_for<FractionalMaxPool2dBackwardFloat64>(
            sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / out_spatial;
            int64_t max_idx = idx[linear];
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ar(acc[nc * in_spatial + max_idx]);
            ar.fetch_add(static_cast<float>(go[linear]));
        }).wait();

        // Copy accumulator back to double
        queue.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) {
            gi[i] = static_cast<double>(acc[i]);
        }).wait();
        sycl::free(acc, queue);
    } else if (grad_output.dtype() == DType::Float16) {
        sycl::half* gi = get_data_ptr<sycl::half>(grad_input);
        const sycl::half* go = get_data_ptr<const sycl::half>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);

        // No need to pre-zero `gi`: the acc->gi copy below writes every element.
        float* acc = sycl::malloc_device<float>(in_size, queue);
        queue.fill(acc, 0.0f, in_size).wait();

        queue.parallel_for<class FractionalMaxPool2dBackwardFloat16>(
            sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / out_spatial;
            int64_t max_idx = idx[linear];
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ar(acc[nc * in_spatial + max_idx]);
            ar.fetch_add(static_cast<float>(go[linear]));
        }).wait();

        queue.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) {
            gi[i] = sycl::half(acc[i]);
        }).wait();
        sycl::free(acc, queue);
    } else {
        throw std::runtime_error("FractionalMaxPool2d backward OneAPI: unsupported dtype");
    }

    return grad_input;
}

// ============================================================================
// Fractional Max Pool 3D Forward
// ============================================================================

auto fractional_maxpool3d_forward_kernel(const Tensor& input,
                                         int64_t out_d, int64_t out_h, int64_t out_w,
                                         const Tensor* random_samples,
                                         sycl::queue& queue)
    -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    if (shape.size() != 5) {
        throw std::invalid_argument("FractionalMaxPool3d requires 5D input (N, C, D, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t D = shape[2];
    const int64_t H = shape[3];
    const int64_t W = shape[4];

    Tensor output({N, C, out_d, out_h, out_w}, input.dtype(), input.device());
    Tensor indices({N, C, out_d, out_h, out_w}, DType::Int64, input.device());

    const int64_t total_out = N * C * out_d * out_h * out_w;
    if (total_out == 0) return {output, indices};

    float* dev_samples = nullptr;
    if (random_samples && random_samples->numel() > 0) {
        dev_samples = get_data_ptr<float>(*random_samples);
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const float* sp = dev_samples;

        queue.parallel_for<FractionalMaxPool3dForwardFloat32>(
            sycl::range<1>(total_out), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t ow_idx = linear % out_w; linear /= out_w;
            int64_t oh_idx = linear % out_h; linear /= out_h;
            int64_t od_idx = linear % out_d; linear /= out_d;
            int64_t c = linear % C;
            int64_t n = linear / C;

            float sample_d = sp ? sp[(n * C + c) * 3 + 0] : 0.5f;
            float sample_h = sp ? sp[(n * C + c) * 3 + 1] : 0.5f;
            float sample_w = sp ? sp[(n * C + c) * 3 + 2] : 0.5f;

            float ratio_d = static_cast<float>(D) / static_cast<float>(out_d);
            float ratio_h = static_cast<float>(H) / static_cast<float>(out_h);
            float ratio_w = static_cast<float>(W) / static_cast<float>(out_w);

            int64_t d_start = static_cast<int64_t>(sycl::floor((od_idx + sample_d) * ratio_d - sample_d));
            int64_t d_end   = static_cast<int64_t>(sycl::floor((od_idx + 1 + sample_d) * ratio_d - sample_d));
            int64_t h_start = static_cast<int64_t>(sycl::floor((oh_idx + sample_h) * ratio_h - sample_h));
            int64_t h_end   = static_cast<int64_t>(sycl::floor((oh_idx + 1 + sample_h) * ratio_h - sample_h));
            int64_t w_start = static_cast<int64_t>(sycl::floor((ow_idx + sample_w) * ratio_w - sample_w));
            int64_t w_end   = static_cast<int64_t>(sycl::floor((ow_idx + 1 + sample_w) * ratio_w - sample_w));

            d_start = sycl::max(d_start, int64_t{0}); d_end = sycl::min(d_end, D);
            h_start = sycl::max(h_start, int64_t{0}); h_end = sycl::min(h_end, H);
            w_start = sycl::max(w_start, int64_t{0}); w_end = sycl::min(w_end, W);
            if (d_end <= d_start) d_end = sycl::min(d_start + 1, D);
            if (h_end <= h_start) h_end = sycl::min(h_start + 1, H);
            if (w_end <= w_start) w_end = sycl::min(w_start + 1, W);

            float max_val = -3.4028235e+38f;
            int64_t max_idx = (d_start * H + h_start) * W + w_start;

            for (int64_t d = d_start; d < d_end; ++d) {
                for (int64_t h = h_start; h < h_end; ++h) {
                    for (int64_t w = w_start; w < w_end; ++w) {
                        int64_t in_idx = (((n * C + c) * D + d) * H + h) * W + w;
                        float val = in_ptr[in_idx];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = (d * H + h) * W + w;
                        }
                    }
                }
            }

            int64_t out_idx = (((n * C + c) * out_d + od_idx) * out_h + oh_idx) * out_w + ow_idx;
            out_ptr[out_idx] = max_val;
            idx_ptr[out_idx] = max_idx;
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const float* sp = dev_samples;

        queue.parallel_for<FractionalMaxPool3dForwardFloat64>(
            sycl::range<1>(total_out), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t ow_idx = linear % out_w; linear /= out_w;
            int64_t oh_idx = linear % out_h; linear /= out_h;
            int64_t od_idx = linear % out_d; linear /= out_d;
            int64_t c = linear % C;
            int64_t n = linear / C;

            float sample_d = sp ? sp[(n * C + c) * 3 + 0] : 0.5f;
            float sample_h = sp ? sp[(n * C + c) * 3 + 1] : 0.5f;
            float sample_w = sp ? sp[(n * C + c) * 3 + 2] : 0.5f;

            float ratio_d = static_cast<float>(D) / static_cast<float>(out_d);
            float ratio_h = static_cast<float>(H) / static_cast<float>(out_h);
            float ratio_w = static_cast<float>(W) / static_cast<float>(out_w);

            int64_t d_start = static_cast<int64_t>(sycl::floor((od_idx + sample_d) * ratio_d - sample_d));
            int64_t d_end   = static_cast<int64_t>(sycl::floor((od_idx + 1 + sample_d) * ratio_d - sample_d));
            int64_t h_start = static_cast<int64_t>(sycl::floor((oh_idx + sample_h) * ratio_h - sample_h));
            int64_t h_end   = static_cast<int64_t>(sycl::floor((oh_idx + 1 + sample_h) * ratio_h - sample_h));
            int64_t w_start = static_cast<int64_t>(sycl::floor((ow_idx + sample_w) * ratio_w - sample_w));
            int64_t w_end   = static_cast<int64_t>(sycl::floor((ow_idx + 1 + sample_w) * ratio_w - sample_w));

            d_start = sycl::max(d_start, int64_t{0}); d_end = sycl::min(d_end, D);
            h_start = sycl::max(h_start, int64_t{0}); h_end = sycl::min(h_end, H);
            w_start = sycl::max(w_start, int64_t{0}); w_end = sycl::min(w_end, W);
            if (d_end <= d_start) d_end = sycl::min(d_start + 1, D);
            if (h_end <= h_start) h_end = sycl::min(h_start + 1, H);
            if (w_end <= w_start) w_end = sycl::min(w_start + 1, W);

            double max_val = -1.7976931348623157e+308;
            int64_t max_idx = (d_start * H + h_start) * W + w_start;

            for (int64_t d = d_start; d < d_end; ++d) {
                for (int64_t h = h_start; h < h_end; ++h) {
                    for (int64_t w = w_start; w < w_end; ++w) {
                        int64_t in_idx = (((n * C + c) * D + d) * H + h) * W + w;
                        double val = in_ptr[in_idx];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = (d * H + h) * W + w;
                        }
                    }
                }
            }

            int64_t out_idx = (((n * C + c) * out_d + od_idx) * out_h + oh_idx) * out_w + ow_idx;
            out_ptr[out_idx] = max_val;
            idx_ptr[out_idx] = max_idx;
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices);
        const float* sp = dev_samples;

        queue.parallel_for<class FractionalMaxPool3dForwardFloat16>(
            sycl::range<1>(total_out), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t ow_idx = linear % out_w; linear /= out_w;
            int64_t oh_idx = linear % out_h; linear /= out_h;
            int64_t od_idx = linear % out_d; linear /= out_d;
            int64_t c = linear % C;
            int64_t n = linear / C;

            float sample_d = sp ? sp[(n * C + c) * 3 + 0] : 0.5f;
            float sample_h = sp ? sp[(n * C + c) * 3 + 1] : 0.5f;
            float sample_w = sp ? sp[(n * C + c) * 3 + 2] : 0.5f;

            float ratio_d = static_cast<float>(D) / static_cast<float>(out_d);
            float ratio_h = static_cast<float>(H) / static_cast<float>(out_h);
            float ratio_w = static_cast<float>(W) / static_cast<float>(out_w);

            int64_t d_start = static_cast<int64_t>(sycl::floor((od_idx + sample_d) * ratio_d - sample_d));
            int64_t d_end   = static_cast<int64_t>(sycl::floor((od_idx + 1 + sample_d) * ratio_d - sample_d));
            int64_t h_start = static_cast<int64_t>(sycl::floor((oh_idx + sample_h) * ratio_h - sample_h));
            int64_t h_end   = static_cast<int64_t>(sycl::floor((oh_idx + 1 + sample_h) * ratio_h - sample_h));
            int64_t w_start = static_cast<int64_t>(sycl::floor((ow_idx + sample_w) * ratio_w - sample_w));
            int64_t w_end   = static_cast<int64_t>(sycl::floor((ow_idx + 1 + sample_w) * ratio_w - sample_w));

            d_start = sycl::max(d_start, int64_t{0}); d_end = sycl::min(d_end, D);
            h_start = sycl::max(h_start, int64_t{0}); h_end = sycl::min(h_end, H);
            w_start = sycl::max(w_start, int64_t{0}); w_end = sycl::min(w_end, W);
            if (d_end <= d_start) d_end = sycl::min(d_start + 1, D);
            if (h_end <= h_start) h_end = sycl::min(h_start + 1, H);
            if (w_end <= w_start) w_end = sycl::min(w_start + 1, W);

            float max_val = -3.4028235e+38f;
            int64_t max_idx = (d_start * H + h_start) * W + w_start;

            for (int64_t d = d_start; d < d_end; ++d) {
                for (int64_t h = h_start; h < h_end; ++h) {
                    for (int64_t w = w_start; w < w_end; ++w) {
                        int64_t in_idx = (((n * C + c) * D + d) * H + h) * W + w;
                        float val = static_cast<float>(in_ptr[in_idx]);
                        if (val > max_val) {
                            max_val = val;
                            max_idx = (d * H + h) * W + w;
                        }
                    }
                }
            }

            int64_t out_idx = (((n * C + c) * out_d + od_idx) * out_h + oh_idx) * out_w + ow_idx;
            out_ptr[out_idx] = sycl::half(max_val);
            idx_ptr[out_idx] = max_idx;
        }).wait();
    } else {
        throw std::runtime_error("FractionalMaxPool3d forward OneAPI: unsupported dtype (Float32/Float64/Float16 supported)");
    }

    return {output, indices};
}

// ============================================================================
// Fractional Max Pool 3D Backward
// ============================================================================

auto fractional_maxpool3d_backward_kernel(const Tensor& grad_output,
                                          const Tensor& indices,
                                          const std::vector<int64_t>& input_shape,
                                          sycl::queue& queue) -> Tensor {
    const int64_t N = input_shape[0];
    const int64_t C = input_shape[1];
    const int64_t D = input_shape[2];
    const int64_t H = input_shape[3];
    const int64_t W = input_shape[4];
    auto grad_shape = grad_output.shape();
    const int64_t out_d = grad_shape[2];
    const int64_t out_h = grad_shape[3];
    const int64_t out_w = grad_shape[4];

    const int64_t in_spatial = D * H * W;
    const int64_t out_spatial = out_d * out_h * out_w;
    const int64_t in_size = N * C * in_spatial;
    const int64_t out_size = N * C * out_spatial;

    Tensor grad_input({N, C, D, H, W}, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        float* gi = get_data_ptr<float>(grad_input);
        const float* go = get_data_ptr<const float>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);

        queue.fill(gi, 0.0f, in_size).wait();

        queue.parallel_for<FractionalMaxPool3dBackwardFloat32>(
            sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / out_spatial;
            int64_t max_idx = idx[linear];
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ar(gi[nc * in_spatial + max_idx]);
            ar.fetch_add(go[linear]);
        }).wait();
    } else if (grad_output.dtype() == DType::Float64) {
        double* gi = get_data_ptr<double>(grad_input);
        const double* go = get_data_ptr<const double>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);

        // No need to pre-zero `gi`: the acc->gi copy below writes every element.
        float* acc = sycl::malloc_device<float>(in_size, queue);
        queue.fill(acc, 0.0f, in_size).wait();

        queue.parallel_for<FractionalMaxPool3dBackwardFloat64>(
            sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / out_spatial;
            int64_t max_idx = idx[linear];
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ar(acc[nc * in_spatial + max_idx]);
            ar.fetch_add(static_cast<float>(go[linear]));
        }).wait();

        queue.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) {
            gi[i] = static_cast<double>(acc[i]);
        }).wait();
        sycl::free(acc, queue);
    } else if (grad_output.dtype() == DType::Float16) {
        sycl::half* gi = get_data_ptr<sycl::half>(grad_input);
        const sycl::half* go = get_data_ptr<const sycl::half>(grad_output);
        const int64_t* idx = get_data_ptr<const int64_t>(indices);

        // No need to pre-zero `gi`: the acc->gi copy below writes every element.
        float* acc = sycl::malloc_device<float>(in_size, queue);
        queue.fill(acc, 0.0f, in_size).wait();

        queue.parallel_for<class FractionalMaxPool3dBackwardFloat16>(
            sycl::range<1>(out_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / out_spatial;
            int64_t max_idx = idx[linear];
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ar(acc[nc * in_spatial + max_idx]);
            ar.fetch_add(static_cast<float>(go[linear]));
        }).wait();

        queue.parallel_for(sycl::range<1>(in_size), [=](sycl::id<1> i) {
            gi[i] = sycl::half(acc[i]);
        }).wait();
        sycl::free(acc, queue);
    } else {
        throw std::runtime_error("FractionalMaxPool3d backward OneAPI: unsupported dtype");
    }

    return grad_input;
}

// ============================================================================
// Max Unpool 2D Forward — scatter: output[indices[i]] = input[i]
// ============================================================================

auto max_unpool2d_forward_kernel(const Tensor& input, const Tensor& indices,
                                 int64_t out_h, int64_t out_w,
                                 sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t in_h = shape[2];
    const int64_t in_w = shape[3];

    const int64_t in_spatial = in_h * in_w;
    const int64_t out_spatial = out_h * out_w;
    const int64_t in_size = N * C * in_spatial;
    const int64_t out_size = N * C * out_spatial;

    Tensor output({N, C, out_h, out_w}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        float* out_ptr = get_data_ptr<float>(output);
        const float* in_ptr = get_data_ptr<const float>(input);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        // Zero the output
        queue.fill(out_ptr, 0.0f, out_size).wait();

        // Scatter input values to output at index positions
        queue.parallel_for<MaxUnpool2dForwardFloat32>(
            sycl::range<1>(in_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / in_spatial;
            int64_t idx = idx_ptr[linear];
            if (idx >= 0 && idx < out_spatial) {
                out_ptr[nc * out_spatial + idx] = in_ptr[linear];
            }
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        double* out_ptr = get_data_ptr<double>(output);
        const double* in_ptr = get_data_ptr<const double>(input);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.fill(out_ptr, 0.0, out_size).wait();

        queue.parallel_for<MaxUnpool2dForwardFloat64>(
            sycl::range<1>(in_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / in_spatial;
            int64_t idx = idx_ptr[linear];
            if (idx >= 0 && idx < out_spatial) {
                out_ptr[nc * out_spatial + idx] = in_ptr[linear];
            }
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.fill(out_ptr, sycl::half(0.0f), out_size).wait();

        queue.parallel_for<class MaxUnpool2dForwardFloat16>(
            sycl::range<1>(in_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / in_spatial;
            int64_t idx = idx_ptr[linear];
            if (idx >= 0 && idx < out_spatial) {
                out_ptr[nc * out_spatial + idx] = in_ptr[linear];
            }
        }).wait();
    } else {
        throw std::runtime_error("MaxUnpool2d forward OneAPI: unsupported dtype (Float32/Float64/Float16 supported)");
    }

    return output;
}

// ============================================================================
// Max Unpool 2D Backward — gather: grad_input[i] = grad_output[indices[i]]
// ============================================================================

auto max_unpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                  const std::vector<int64_t>& input_shape,
                                  sycl::queue& queue) -> Tensor {
    const int64_t N = input_shape[0];
    const int64_t C = input_shape[1];
    const int64_t in_h = input_shape[2];
    const int64_t in_w = input_shape[3];
    auto grad_shape = grad_output.shape();
    const int64_t out_h = grad_shape[2];
    const int64_t out_w = grad_shape[3];

    const int64_t in_spatial = in_h * in_w;
    const int64_t out_spatial = out_h * out_w;
    const int64_t in_size = N * C * in_spatial;

    Tensor grad_input({N, C, in_h, in_w}, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        float* gi = get_data_ptr<float>(grad_input);
        const float* go = get_data_ptr<const float>(grad_output);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.parallel_for<MaxUnpool2dBackwardFloat32>(
            sycl::range<1>(in_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / in_spatial;
            int64_t idx = idx_ptr[linear];
            if (idx >= 0 && idx < out_spatial) {
                gi[linear] = go[nc * out_spatial + idx];
            } else {
                gi[linear] = 0.0f;
            }
        }).wait();
    } else if (grad_output.dtype() == DType::Float64) {
        double* gi = get_data_ptr<double>(grad_input);
        const double* go = get_data_ptr<const double>(grad_output);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.parallel_for<MaxUnpool2dBackwardFloat64>(
            sycl::range<1>(in_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / in_spatial;
            int64_t idx = idx_ptr[linear];
            if (idx >= 0 && idx < out_spatial) {
                gi[linear] = go[nc * out_spatial + idx];
            } else {
                gi[linear] = 0.0;
            }
        }).wait();
    } else if (grad_output.dtype() == DType::Float16) {
        sycl::half* gi = get_data_ptr<sycl::half>(grad_input);
        const sycl::half* go = get_data_ptr<const sycl::half>(grad_output);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.parallel_for<class MaxUnpool2dBackwardFloat16>(
            sycl::range<1>(in_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / in_spatial;
            int64_t idx = idx_ptr[linear];
            if (idx >= 0 && idx < out_spatial) {
                gi[linear] = go[nc * out_spatial + idx];
            } else {
                gi[linear] = sycl::half(0.0f);
            }
        }).wait();
    } else {
        throw std::runtime_error("MaxUnpool2d backward OneAPI: unsupported dtype");
    }

    return grad_input;
}

// ============================================================================
// Max Unpool 3D Forward — scatter: output[indices[i]] = input[i]
// ============================================================================

auto max_unpool3d_forward_kernel(const Tensor& input, const Tensor& indices,
                                 int64_t out_d, int64_t out_h, int64_t out_w,
                                 sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t in_d = shape[2];
    const int64_t in_h = shape[3];
    const int64_t in_w = shape[4];

    const int64_t in_spatial = in_d * in_h * in_w;
    const int64_t out_spatial = out_d * out_h * out_w;
    const int64_t in_size = N * C * in_spatial;
    const int64_t out_size = N * C * out_spatial;

    Tensor output({N, C, out_d, out_h, out_w}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        float* out_ptr = get_data_ptr<float>(output);
        const float* in_ptr = get_data_ptr<const float>(input);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.fill(out_ptr, 0.0f, out_size).wait();

        queue.parallel_for<MaxUnpool3dForwardFloat32>(
            sycl::range<1>(in_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / in_spatial;
            int64_t idx = idx_ptr[linear];
            if (idx >= 0 && idx < out_spatial) {
                out_ptr[nc * out_spatial + idx] = in_ptr[linear];
            }
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        double* out_ptr = get_data_ptr<double>(output);
        const double* in_ptr = get_data_ptr<const double>(input);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.fill(out_ptr, 0.0, out_size).wait();

        queue.parallel_for<MaxUnpool3dForwardFloat64>(
            sycl::range<1>(in_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / in_spatial;
            int64_t idx = idx_ptr[linear];
            if (idx >= 0 && idx < out_spatial) {
                out_ptr[nc * out_spatial + idx] = in_ptr[linear];
            }
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.fill(out_ptr, sycl::half(0.0f), out_size).wait();

        queue.parallel_for<class MaxUnpool3dForwardFloat16>(
            sycl::range<1>(in_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / in_spatial;
            int64_t idx = idx_ptr[linear];
            if (idx >= 0 && idx < out_spatial) {
                out_ptr[nc * out_spatial + idx] = in_ptr[linear];
            }
        }).wait();
    } else {
        throw std::runtime_error("MaxUnpool3d forward OneAPI: unsupported dtype (Float32/Float64/Float16 supported)");
    }

    return output;
}

// ============================================================================
// Max Unpool 3D Backward — gather: grad_input[i] = grad_output[indices[i]]
// ============================================================================

auto max_unpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                  const std::vector<int64_t>& input_shape,
                                  sycl::queue& queue) -> Tensor {
    const int64_t N = input_shape[0];
    const int64_t C = input_shape[1];
    const int64_t in_d = input_shape[2];
    const int64_t in_h = input_shape[3];
    const int64_t in_w = input_shape[4];
    auto grad_shape = grad_output.shape();
    const int64_t out_d = grad_shape[2];
    const int64_t out_h = grad_shape[3];
    const int64_t out_w = grad_shape[4];

    const int64_t in_spatial = in_d * in_h * in_w;
    const int64_t out_spatial = out_d * out_h * out_w;
    const int64_t in_size = N * C * in_spatial;

    Tensor grad_input({N, C, in_d, in_h, in_w}, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        float* gi = get_data_ptr<float>(grad_input);
        const float* go = get_data_ptr<const float>(grad_output);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.parallel_for<MaxUnpool3dBackwardFloat32>(
            sycl::range<1>(in_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / in_spatial;
            int64_t idx = idx_ptr[linear];
            if (idx >= 0 && idx < out_spatial) {
                gi[linear] = go[nc * out_spatial + idx];
            } else {
                gi[linear] = 0.0f;
            }
        }).wait();
    } else if (grad_output.dtype() == DType::Float64) {
        double* gi = get_data_ptr<double>(grad_input);
        const double* go = get_data_ptr<const double>(grad_output);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.parallel_for<MaxUnpool3dBackwardFloat64>(
            sycl::range<1>(in_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / in_spatial;
            int64_t idx = idx_ptr[linear];
            if (idx >= 0 && idx < out_spatial) {
                gi[linear] = go[nc * out_spatial + idx];
            } else {
                gi[linear] = 0.0;
            }
        }).wait();
    } else if (grad_output.dtype() == DType::Float16) {
        sycl::half* gi = get_data_ptr<sycl::half>(grad_input);
        const sycl::half* go = get_data_ptr<const sycl::half>(grad_output);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.parallel_for<class MaxUnpool3dBackwardFloat16>(
            sycl::range<1>(in_size), [=](sycl::id<1> gid) {
            int64_t linear = gid;
            int64_t nc = linear / in_spatial;
            int64_t idx = idx_ptr[linear];
            if (idx >= 0 && idx < out_spatial) {
                gi[linear] = go[nc * out_spatial + idx];
            } else {
                gi[linear] = sycl::half(0.0f);
            }
        }).wait();
    } else {
        throw std::runtime_error("MaxUnpool3d backward OneAPI: unsupported dtype");
    }

    return grad_input;
}

// ============================================================================
// Phase A.1 — Max Unpool 1D (OneAPI). Reshape (N, C, L) → (N, C, L, 1) and
// reuse the existing 2D kernel.
// ============================================================================

auto max_unpool1d_forward_kernel(const Tensor& input, const Tensor& indices,
                                 int64_t out_l, sycl::queue& queue) -> Tensor
{
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], in_l = shape[2];
    auto input_4d = input.contiguous().reshape({N, C, in_l, 1});
    auto indices_4d = indices.contiguous().reshape({N, C, in_l, 1});
    auto out_4d = max_unpool2d_forward_kernel(input_4d, indices_4d, out_l, /*out_w=*/1, queue);
    return out_4d.reshape({N, C, out_l});
}

auto max_unpool1d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                   const std::vector<int64_t>& input_shape,
                                   sycl::queue& queue) -> Tensor
{
    int64_t N = input_shape[0], C = input_shape[1], in_l = input_shape[2];
    int64_t out_l = grad_output.shape()[2];
    std::vector<int64_t> input_shape_4d = {N, C, in_l, 1};
    auto grad_4d = grad_output.contiguous().reshape({N, C, out_l, 1});
    auto indices_4d = indices.contiguous().reshape({N, C, in_l, 1});
    auto grad_in_4d = max_unpool2d_backward_kernel(grad_4d, indices_4d, input_shape_4d, queue);
    return grad_in_4d.reshape({N, C, in_l});
}

} // namespace oneapi
} // namespace tenzor
