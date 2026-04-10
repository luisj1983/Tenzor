#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>

#ifdef TENZOR_HAS_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#endif

namespace tenzor {
namespace oneapi {

// Kernel class declarations for SYCL
class BatchNorm2dForwardKernelFloat32;
class BatchNorm2dForwardKernelFloat64;
class BatchNorm2dForwardKernelFloat16;
class BatchNorm2dForwardAffineKernelFloat32;
class BatchNorm2dForwardAffineKernelFloat64;
class BatchNorm2dForwardAffineKernelFloat16;
class BatchNorm2dBackwardGammaKernelFloat32;
class BatchNorm2dBackwardGammaKernelFloat64;
class BatchNorm2dBackwardGammaKernelFloat16;
class BatchNorm2dBackwardInputKernelFloat32;
class BatchNorm2dBackwardInputKernelFloat64;
class BatchNorm2dBackwardInputKernelFloat16;
class BatchNormScaleShiftKernelFloat32;
class BatchNormExtractGradKernelFloat32;
class BatchNorm2dMeanKernelFloat32;
class BatchNorm2dMeanKernelFloat64;
class BatchNorm2dMeanKernelFloat16;
class BatchNorm2dVarianceKernelFloat32;
class BatchNorm2dVarianceKernelFloat64;
class BatchNorm2dVarianceKernelFloat16;
class BatchNorm2dUpdateRunningStatsKernelFloat32;
class BatchNorm2dUpdateRunningStatsKernelFloat64;
class BatchNorm2dUpdateRunningStatsKernelFloat16;
class BatchNorm2dForwardKernelBFloat16;
class BatchNorm2dForwardAffineKernelBFloat16;
class BatchNorm2dBackwardGammaKernelBFloat16;
class BatchNorm2dBackwardInputKernelBFloat16;
class BatchNorm2dMeanKernelBFloat16;
class BatchNorm2dVarianceKernelBFloat16;
class BatchNorm2dUpdateRunningStatsKernelBFloat16;

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// BFloat16 <-> Float32 conversion helpers (device-compatible)
inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

inline uint16_t f32_to_bf16(float f32) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f32, sizeof(uint32_t));
    // Round to nearest even (banker's rounding) for BFloat16
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

#ifdef TENZOR_HAS_ONEDNN

// Forward declarations: pure SYCL implementations for non-Float32 dtypes
static auto batchnorm2d_forward_sycl(const Tensor& input, const Tensor& mean, const Tensor& variance,
                                      float epsilon, sycl::queue& queue) -> Tensor;
static auto batchnorm2d_forward_affine_sycl(const Tensor& input, const Tensor& mean, const Tensor& variance,
                                             const Tensor& gamma, const Tensor& beta, float epsilon, sycl::queue& queue) -> Tensor;
static auto batchnorm2d_backward_sycl(const Tensor& grad_output, const Tensor& input, const Tensor& mean,
                                       const Tensor& variance, const Tensor& gamma, float epsilon, sycl::queue& queue)
                                       -> std::tuple<Tensor, Tensor, Tensor>;

// Batch normalization forward using oneDNN
auto batchnorm2d_forward(const Tensor& input, const Tensor& mean, const Tensor& variance,
                         float epsilon, sycl::queue& queue) -> Tensor {
    // oneDNN BatchNorm descriptors are f32-only; delegate non-f32 to SYCL
    if (input.dtype() != DType::Float32) {
        return batchnorm2d_forward_sycl(input, mean, variance, epsilon, queue);
    }

    using namespace dnnl;

    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("BatchNorm2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H = shape[2];
    const int64_t W = shape[3];

    Tensor output(std::vector<int64_t>{N, C, H, W}, input.dtype(), input.device());

    // Create oneDNN engine and stream from SYCL queue
    auto dnnl_engine = sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto dnnl_stream = sycl_interop::make_stream(dnnl_engine, queue);

    // Create memory descriptors
    memory::dims src_dims = {N, C, H, W};
    memory::dims scale_shift_dims = {C};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto dst_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto scale_shift_md = memory::desc(scale_shift_dims, memory::data_type::f32, memory::format_tag::x);

    // Create primitive descriptor
    auto bn_desc = batch_normalization_forward::desc(
        prop_kind::forward_inference,
        src_md,
        epsilon,
        normalization_flags::use_global_stats
    );

    auto bn_pd = batch_normalization_forward::primitive_desc(bn_desc, dnnl_engine);

    // Wrap tensors as oneDNN memory objects
    auto src_mem = sycl_interop::make_memory(bn_pd.src_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(input.data_ptr()));

    auto dst_mem = sycl_interop::make_memory(bn_pd.dst_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(output.data_ptr()));

    auto mean_mem = sycl_interop::make_memory(scale_shift_md, dnnl_engine,
                                               sycl_interop::memory_kind::usm,
                                               const_cast<void*>(mean.data_ptr()));

    auto var_mem = sycl_interop::make_memory(scale_shift_md, dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(variance.data_ptr()));

    // Create and execute primitive
    auto bn_prim = batch_normalization_forward(bn_pd);
    bn_prim.execute(dnnl_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_MEAN, mean_mem},
        {DNNL_ARG_VARIANCE, var_mem},
        {DNNL_ARG_DST, dst_mem}
    });

    dnnl_stream.wait();

    return output;
}

// Batch normalization forward with affine parameters
auto batchnorm2d_forward_affine(const Tensor& input, const Tensor& mean, const Tensor& variance,
                                const Tensor& gamma, const Tensor& beta, float epsilon, sycl::queue& queue) -> Tensor {
    // oneDNN BatchNorm descriptors are f32-only; delegate non-f32 to SYCL
    if (input.dtype() != DType::Float32) {
        return batchnorm2d_forward_affine_sycl(input, mean, variance, gamma, beta, epsilon, queue);
    }

    using namespace dnnl;

    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("BatchNorm2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H = shape[2];
    const int64_t W = shape[3];

    Tensor output(std::vector<int64_t>{N, C, H, W}, input.dtype(), input.device());

    // Create oneDNN engine and stream
    auto dnnl_engine = sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto dnnl_stream = sycl_interop::make_stream(dnnl_engine, queue);

    // Create memory descriptors
    memory::dims src_dims = {N, C, H, W};
    memory::dims scale_shift_dims = {C};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto dst_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto scale_shift_md = memory::desc(scale_shift_dims, memory::data_type::f32, memory::format_tag::x);

    // Create primitive descriptor with scale and shift
    auto bn_desc = batch_normalization_forward::desc(
        prop_kind::forward_inference,
        src_md,
        epsilon,
        normalization_flags::use_global_stats | normalization_flags::use_scale_shift
    );

    auto bn_pd = batch_normalization_forward::primitive_desc(bn_desc, dnnl_engine);

    // Wrap tensors
    auto src_mem = sycl_interop::make_memory(bn_pd.src_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(input.data_ptr()));

    auto dst_mem = sycl_interop::make_memory(bn_pd.dst_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(output.data_ptr()));

    auto mean_mem = sycl_interop::make_memory(scale_shift_md, dnnl_engine,
                                               sycl_interop::memory_kind::usm,
                                               const_cast<void*>(mean.data_ptr()));

    auto var_mem = sycl_interop::make_memory(scale_shift_md, dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(variance.data_ptr()));

    // Create scale-shift tensor (gamma and beta concatenated)
    Tensor scale_shift({2, C}, DType::Float32, input.device());
    float* ss_ptr = get_data_ptr<float>(scale_shift);
    const float* gamma_ptr = get_data_ptr<const float>(gamma);
    const float* beta_ptr = get_data_ptr<const float>(beta);

    queue.parallel_for<BatchNormScaleShiftKernelFloat32>(sycl::range<1>(C), [=](sycl::id<1> i) {
        ss_ptr[i] = gamma_ptr[i];           // Scale
        ss_ptr[C + i] = beta_ptr[i];        // Shift
    });

    auto ss_mem = sycl_interop::make_memory(
        memory::desc({2, C}, memory::data_type::f32, memory::format_tag::nc),
        dnnl_engine,
        sycl_interop::memory_kind::usm,
        const_cast<void*>(scale_shift.data_ptr())
    );

    // Execute
    auto bn_prim = batch_normalization_forward(bn_pd);
    bn_prim.execute(dnnl_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_MEAN, mean_mem},
        {DNNL_ARG_VARIANCE, var_mem},
        {DNNL_ARG_SCALE_SHIFT, ss_mem},
        {DNNL_ARG_DST, dst_mem}
    });

    dnnl_stream.wait();

    return output;
}

// Batch normalization backward
auto batchnorm2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& mean,
                          const Tensor& variance, const Tensor& gamma, float epsilon, sycl::queue& queue)
                          -> std::tuple<Tensor, Tensor, Tensor> {
    // oneDNN BatchNorm backward descriptors are f32-only; delegate non-f32 to SYCL
    if (input.dtype() != DType::Float32) {
        return batchnorm2d_backward_sycl(grad_output, input, mean, variance, gamma, epsilon, queue);
    }

    using namespace dnnl;

    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("BatchNorm2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H = shape[2];
    const int64_t W = shape[3];

    Tensor grad_input(std::vector<int64_t>{N, C, H, W}, input.dtype(), input.device());
    Tensor grad_gamma({C}, input.dtype(), input.device());
    Tensor grad_beta({C}, input.dtype(), input.device());

    // Create oneDNN engine and stream
    auto dnnl_engine = sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto dnnl_stream = sycl_interop::make_stream(dnnl_engine, queue);

    // Memory descriptors
    memory::dims src_dims = {N, C, H, W};
    memory::dims scale_shift_dims = {C};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto diff_dst_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto diff_src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto scale_shift_md = memory::desc(scale_shift_dims, memory::data_type::f32, memory::format_tag::x);

    // Forward descriptor (needed for backward)
    auto bn_fwd_desc = batch_normalization_forward::desc(
        prop_kind::forward_training,
        src_md,
        epsilon,
        normalization_flags::use_scale_shift
    );
    auto bn_fwd_pd = batch_normalization_forward::primitive_desc(bn_fwd_desc, dnnl_engine);

    // Backward descriptor
    auto bn_bwd_desc = batch_normalization_backward::desc(
        prop_kind::backward,
        diff_dst_md,
        src_md,
        epsilon,
        normalization_flags::use_scale_shift
    );
    auto bn_bwd_pd = batch_normalization_backward::primitive_desc(bn_bwd_desc, dnnl_engine, bn_fwd_pd);

    // Wrap tensors
    auto src_mem = sycl_interop::make_memory(bn_bwd_pd.src_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(input.data_ptr()));

    auto diff_dst_mem = sycl_interop::make_memory(bn_bwd_pd.diff_dst_desc(), dnnl_engine,
                                                   sycl_interop::memory_kind::usm,
                                                   const_cast<void*>(grad_output.data_ptr()));

    auto diff_src_mem = sycl_interop::make_memory(bn_bwd_pd.diff_src_desc(), dnnl_engine,
                                                   sycl_interop::memory_kind::usm,
                                                   const_cast<void*>(grad_input.data_ptr()));

    auto mean_mem = sycl_interop::make_memory(scale_shift_md, dnnl_engine,
                                               sycl_interop::memory_kind::usm,
                                               const_cast<void*>(mean.data_ptr()));

    auto var_mem = sycl_interop::make_memory(scale_shift_md, dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(variance.data_ptr()));

    // Scale-shift for forward (always f32 for oneDNN)
    Tensor scale_shift({2, C}, DType::Float32, input.device());
    float* ss_ptr = get_data_ptr<float>(scale_shift);
    const float* gamma_ptr = get_data_ptr<const float>(gamma);

    queue.parallel_for<BatchNormScaleShiftKernelFloat32>(sycl::range<1>(C), [=](sycl::id<1> i) {
        ss_ptr[i] = gamma_ptr[i];
        ss_ptr[C + i] = 0.0f;  // Beta not used in backward
    });

    auto ss_mem = sycl_interop::make_memory(
        memory::desc({2, C}, memory::data_type::f32, memory::format_tag::nc),
        dnnl_engine,
        sycl_interop::memory_kind::usm,
        const_cast<void*>(scale_shift.data_ptr())
    );

    // Diff scale-shift for gradients (always f32 for oneDNN)
    Tensor diff_scale_shift({2, C}, DType::Float32, input.device());
    auto diff_ss_mem = sycl_interop::make_memory(
        memory::desc({2, C}, memory::data_type::f32, memory::format_tag::nc),
        dnnl_engine,
        sycl_interop::memory_kind::usm,
        const_cast<void*>(diff_scale_shift.data_ptr())
    );

    // Execute backward
    auto bn_bwd_prim = batch_normalization_backward(bn_bwd_pd);
    bn_bwd_prim.execute(dnnl_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_MEAN, mean_mem},
        {DNNL_ARG_VARIANCE, var_mem},
        {DNNL_ARG_DIFF_DST, diff_dst_mem},
        {DNNL_ARG_SCALE_SHIFT, ss_mem},
        {DNNL_ARG_DIFF_SRC, diff_src_mem},
        {DNNL_ARG_DIFF_SCALE_SHIFT, diff_ss_mem}
    });

    dnnl_stream.wait();

    // Extract grad_gamma and grad_beta
    float* diff_ss_ptr = get_data_ptr<float>(diff_scale_shift);
    float* grad_gamma_ptr = get_data_ptr<float>(grad_gamma);
    float* grad_beta_ptr = get_data_ptr<float>(grad_beta);

    queue.parallel_for<BatchNormExtractGradKernelFloat32>(sycl::range<1>(C), [=](sycl::id<1> i) {
        grad_gamma_ptr[i] = diff_ss_ptr[i];
        grad_beta_ptr[i] = diff_ss_ptr[C + i];
    });

    return {grad_input, grad_gamma, grad_beta};
}

// ============================================================================
// Pure SYCL fallback implementations for non-Float32 dtypes (used when oneDNN is available
// but the dtype is not f32, since oneDNN descriptors are hardcoded to f32).
// ============================================================================

static auto batchnorm2d_forward_sycl(const Tensor& input, const Tensor& mean, const Tensor& variance,
                                      float epsilon, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    const int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    Tensor output(std::vector<int64_t>{N, C, H, W}, input.dtype(), input.device());

    if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* mean_ptr = get_data_ptr<const double>(mean);
        const double* var_ptr = get_data_ptr<const double>(variance);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<BatchNorm2dForwardKernelFloat64>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0], c = idx[1], hw = idx[2];
            const double std_inv = 1.0 / sycl::sqrt(var_ptr[c] + static_cast<double>(epsilon));
            const int64_t i = ((n * C + c) * H * W) + hw;
            out_ptr[i] = (in_ptr[i] - mean_ptr[c]) * std_inv;
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const bool stats_f32 = (mean.dtype() == DType::Float32);
        const float* mean_f32 = stats_f32 ? get_data_ptr<const float>(mean) : nullptr;
        const float* var_f32 = stats_f32 ? get_data_ptr<const float>(variance) : nullptr;
        const sycl::half* mean_f16 = !stats_f32 ? get_data_ptr<const sycl::half>(mean) : nullptr;
        const sycl::half* var_f16 = !stats_f32 ? get_data_ptr<const sycl::half>(variance) : nullptr;
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<BatchNorm2dForwardKernelFloat16>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0], c = idx[1], hw = idx[2];
            const float m = stats_f32 ? mean_f32[c] : static_cast<float>(mean_f16[c]);
            const float v = stats_f32 ? var_f32[c] : static_cast<float>(var_f16[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);
            const int64_t i = ((n * C + c) * H * W) + hw;
            out_ptr[i] = sycl::half((static_cast<float>(in_ptr[i]) - m) * std_inv);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        const bool stats_f32 = (mean.dtype() == DType::Float32);
        const float* mean_f32 = stats_f32 ? get_data_ptr<const float>(mean) : nullptr;
        const float* var_f32 = stats_f32 ? get_data_ptr<const float>(variance) : nullptr;
        const uint16_t* mean_bf16 = !stats_f32 ? get_data_ptr<const uint16_t>(mean) : nullptr;
        const uint16_t* var_bf16 = !stats_f32 ? get_data_ptr<const uint16_t>(variance) : nullptr;
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<BatchNorm2dForwardKernelBFloat16>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0], c = idx[1], hw = idx[2];
            const float m = stats_f32 ? mean_f32[c] : bf16_to_f32(mean_bf16[c]);
            const float v = stats_f32 ? var_f32[c] : bf16_to_f32(var_bf16[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);
            const int64_t i = ((n * C + c) * H * W) + hw;
            out_ptr[i] = f32_to_bf16((bf16_to_f32(in_ptr[i]) - m) * std_inv);
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for batchnorm2d_forward");
    }
    return output;
}

static auto batchnorm2d_forward_affine_sycl(const Tensor& input, const Tensor& mean, const Tensor& variance,
                                             const Tensor& gamma, const Tensor& beta, float epsilon, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    const int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    Tensor output(std::vector<int64_t>{N, C, H, W}, input.dtype(), input.device());

    if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* mean_ptr = get_data_ptr<const double>(mean);
        const double* var_ptr = get_data_ptr<const double>(variance);
        const double* g_ptr = get_data_ptr<const double>(gamma);
        const double* b_ptr = get_data_ptr<const double>(beta);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<BatchNorm2dForwardAffineKernelFloat64>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0], c = idx[1], hw = idx[2];
            const double std_inv = 1.0 / sycl::sqrt(var_ptr[c] + static_cast<double>(epsilon));
            const int64_t i = ((n * C + c) * H * W) + hw;
            out_ptr[i] = g_ptr[c] * (in_ptr[i] - mean_ptr[c]) * std_inv + b_ptr[c];
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const bool stats_f32 = (mean.dtype() == DType::Float32);
        const float* mean_f32 = stats_f32 ? get_data_ptr<const float>(mean) : nullptr;
        const float* var_f32 = stats_f32 ? get_data_ptr<const float>(variance) : nullptr;
        const sycl::half* mean_f16 = !stats_f32 ? get_data_ptr<const sycl::half>(mean) : nullptr;
        const sycl::half* var_f16 = !stats_f32 ? get_data_ptr<const sycl::half>(variance) : nullptr;
        const sycl::half* g_ptr = get_data_ptr<const sycl::half>(gamma);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(beta);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<BatchNorm2dForwardAffineKernelFloat16>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0], c = idx[1], hw = idx[2];
            const float m = stats_f32 ? mean_f32[c] : static_cast<float>(mean_f16[c]);
            const float v = stats_f32 ? var_f32[c] : static_cast<float>(var_f16[c]);
            const float g = static_cast<float>(g_ptr[c]);
            const float b = static_cast<float>(b_ptr[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);
            const int64_t i = ((n * C + c) * H * W) + hw;
            out_ptr[i] = sycl::half(g * (static_cast<float>(in_ptr[i]) - m) * std_inv + b);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        const bool stats_f32 = (mean.dtype() == DType::Float32);
        const float* mean_f32 = stats_f32 ? get_data_ptr<const float>(mean) : nullptr;
        const float* var_f32 = stats_f32 ? get_data_ptr<const float>(variance) : nullptr;
        const uint16_t* mean_bf16 = !stats_f32 ? get_data_ptr<const uint16_t>(mean) : nullptr;
        const uint16_t* var_bf16 = !stats_f32 ? get_data_ptr<const uint16_t>(variance) : nullptr;
        const uint16_t* g_ptr = get_data_ptr<const uint16_t>(gamma);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(beta);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<BatchNorm2dForwardAffineKernelBFloat16>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0], c = idx[1], hw = idx[2];
            const float m = stats_f32 ? mean_f32[c] : bf16_to_f32(mean_bf16[c]);
            const float v = stats_f32 ? var_f32[c] : bf16_to_f32(var_bf16[c]);
            const float g = bf16_to_f32(g_ptr[c]);
            const float b = bf16_to_f32(b_ptr[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);
            const int64_t i = ((n * C + c) * H * W) + hw;
            out_ptr[i] = f32_to_bf16(g * (bf16_to_f32(in_ptr[i]) - m) * std_inv + b);
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for batchnorm2d_forward_affine");
    }
    return output;
}

static auto batchnorm2d_backward_sycl(const Tensor& grad_output, const Tensor& input, const Tensor& mean,
                                       const Tensor& variance, const Tensor& gamma, float epsilon, sycl::queue& queue)
                                       -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape = input.shape();
    const int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    const int64_t spatial = H * W;

    Tensor grad_input(std::vector<int64_t>{N, C, H, W}, input.dtype(), input.device());
    Tensor grad_gamma({C}, input.dtype(), input.device());
    Tensor grad_beta({C}, input.dtype(), input.device());

    if (input.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* mean_ptr = get_data_ptr<const double>(mean);
        const double* var_ptr = get_data_ptr<const double>(variance);
        const double* gamma_ptr = get_data_ptr<const double>(gamma);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);
        double* grad_gamma_ptr = get_data_ptr<double>(grad_gamma);
        double* grad_beta_ptr = get_data_ptr<double>(grad_beta);

        queue.parallel_for<BatchNorm2dBackwardGammaKernelFloat64>(sycl::range<1>(C), [=](sycl::id<1> c) {
            double sum_go = 0.0, sum_go_norm = 0.0;
            const double m = mean_ptr[c], std_inv = 1.0 / sycl::sqrt(var_ptr[c] + static_cast<double>(epsilon));
            for (int64_t n = 0; n < N; ++n)
                for (int64_t hw = 0; hw < spatial; ++hw) {
                    const int64_t idx = ((n * C + c) * spatial) + hw;
                    sum_go += grad_out_ptr[idx];
                    sum_go_norm += grad_out_ptr[idx] * (in_ptr[idx] - m) * std_inv;
                }
            grad_gamma_ptr[c] = sum_go_norm;
            grad_beta_ptr[c] = sum_go;
        }).wait();

        // The input gradient kernel reads grad_gamma and grad_beta written above
        queue.wait();

        const double scale = 1.0 / static_cast<double>(N * spatial);
        queue.parallel_for<BatchNorm2dBackwardInputKernelFloat64>(sycl::range<3>(N, C, spatial), [=](sycl::id<3> idx) {
            const int64_t n = idx[0], c = idx[1], hw = idx[2];
            const double std_inv = 1.0 / sycl::sqrt(var_ptr[c] + static_cast<double>(epsilon));
            const int64_t i = ((n * C + c) * spatial) + hw;
            const double x_norm = (in_ptr[i] - mean_ptr[c]) * std_inv;
            grad_in_ptr[i] = gamma_ptr[c] * std_inv * (grad_out_ptr[i] -
                grad_beta_ptr[c] * scale - x_norm * grad_gamma_ptr[c] * scale);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const bool stats_f32 = (mean.dtype() == DType::Float32);
        const float* mean_f32 = stats_f32 ? get_data_ptr<const float>(mean) : nullptr;
        const float* var_f32 = stats_f32 ? get_data_ptr<const float>(variance) : nullptr;
        const sycl::half* mean_f16 = !stats_f32 ? get_data_ptr<const sycl::half>(mean) : nullptr;
        const sycl::half* var_f16 = !stats_f32 ? get_data_ptr<const sycl::half>(variance) : nullptr;
        const sycl::half* gamma_ptr = get_data_ptr<const sycl::half>(gamma);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);
        sycl::half* grad_gamma_ptr = get_data_ptr<sycl::half>(grad_gamma);
        sycl::half* grad_beta_ptr = get_data_ptr<sycl::half>(grad_beta);

        queue.parallel_for<BatchNorm2dBackwardGammaKernelFloat16>(sycl::range<1>(C), [=](sycl::id<1> c) {
            float sum_go = 0.0f, sum_go_norm = 0.0f;
            const float m = stats_f32 ? mean_f32[c] : static_cast<float>(mean_f16[c]);
            const float v = stats_f32 ? var_f32[c] : static_cast<float>(var_f16[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);
            for (int64_t n = 0; n < N; ++n)
                for (int64_t hw = 0; hw < spatial; ++hw) {
                    const int64_t idx = ((n * C + c) * spatial) + hw;
                    float go = static_cast<float>(grad_out_ptr[idx]);
                    sum_go += go;
                    sum_go_norm += go * (static_cast<float>(in_ptr[idx]) - m) * std_inv;
                }
            grad_gamma_ptr[c] = sycl::half(sum_go_norm);
            grad_beta_ptr[c] = sycl::half(sum_go);
        }).wait();

        // The input gradient kernel reads grad_gamma and grad_beta written above
        queue.wait();

        const float scale = 1.0f / static_cast<float>(N * spatial);
        queue.parallel_for<BatchNorm2dBackwardInputKernelFloat16>(sycl::range<3>(N, C, spatial), [=](sycl::id<3> idx) {
            const int64_t n = idx[0], c = idx[1], hw = idx[2];
            const float m = stats_f32 ? mean_f32[c] : static_cast<float>(mean_f16[c]);
            const float v = stats_f32 ? var_f32[c] : static_cast<float>(var_f16[c]);
            const float g = static_cast<float>(gamma_ptr[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);
            const int64_t i = ((n * C + c) * spatial) + hw;
            const float x_norm = (static_cast<float>(in_ptr[i]) - m) * std_inv;
            grad_in_ptr[i] = sycl::half(g * std_inv * (static_cast<float>(grad_out_ptr[i]) -
                static_cast<float>(grad_beta_ptr[c]) * scale -
                x_norm * static_cast<float>(grad_gamma_ptr[c]) * scale));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        const bool stats_f32 = (mean.dtype() == DType::Float32);
        const float* mean_f32 = stats_f32 ? get_data_ptr<const float>(mean) : nullptr;
        const float* var_f32 = stats_f32 ? get_data_ptr<const float>(variance) : nullptr;
        const uint16_t* mean_bf16 = !stats_f32 ? get_data_ptr<const uint16_t>(mean) : nullptr;
        const uint16_t* var_bf16 = !stats_f32 ? get_data_ptr<const uint16_t>(variance) : nullptr;
        const uint16_t* gamma_ptr = get_data_ptr<const uint16_t>(gamma);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);
        uint16_t* grad_gamma_ptr = get_data_ptr<uint16_t>(grad_gamma);
        uint16_t* grad_beta_ptr = get_data_ptr<uint16_t>(grad_beta);

        queue.parallel_for<BatchNorm2dBackwardGammaKernelBFloat16>(sycl::range<1>(C), [=](sycl::id<1> c) {
            float sum_go = 0.0f, sum_go_norm = 0.0f;
            const float m = stats_f32 ? mean_f32[c] : bf16_to_f32(mean_bf16[c]);
            const float v = stats_f32 ? var_f32[c] : bf16_to_f32(var_bf16[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);
            for (int64_t n = 0; n < N; ++n)
                for (int64_t hw = 0; hw < spatial; ++hw) {
                    const int64_t idx = ((n * C + c) * spatial) + hw;
                    float go = bf16_to_f32(grad_out_ptr[idx]);
                    sum_go += go;
                    sum_go_norm += go * (bf16_to_f32(in_ptr[idx]) - m) * std_inv;
                }
            grad_gamma_ptr[c] = f32_to_bf16(sum_go_norm);
            grad_beta_ptr[c] = f32_to_bf16(sum_go);
        }).wait();

        // The input gradient kernel reads grad_gamma and grad_beta written above
        queue.wait();

        const float scale = 1.0f / static_cast<float>(N * spatial);
        queue.parallel_for<BatchNorm2dBackwardInputKernelBFloat16>(sycl::range<3>(N, C, spatial), [=](sycl::id<3> idx) {
            const int64_t n = idx[0], c = idx[1], hw = idx[2];
            const float m = stats_f32 ? mean_f32[c] : bf16_to_f32(mean_bf16[c]);
            const float v = stats_f32 ? var_f32[c] : bf16_to_f32(var_bf16[c]);
            const float g = bf16_to_f32(gamma_ptr[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);
            const int64_t i = ((n * C + c) * spatial) + hw;
            const float x_norm = (bf16_to_f32(in_ptr[i]) - m) * std_inv;
            grad_in_ptr[i] = f32_to_bf16(g * std_inv * (bf16_to_f32(grad_out_ptr[i]) -
                bf16_to_f32(grad_beta_ptr[c]) * scale -
                x_norm * bf16_to_f32(grad_gamma_ptr[c]) * scale));
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for batchnorm2d_backward");
    }
    return {grad_input, grad_gamma, grad_beta};
}

#else // !TENZOR_HAS_ONEDNN
#pragma message("WARNING: Building without oneDNN — using slower BatchNorm fallback")

// Pure SYCL implementation (fallback)

auto batchnorm2d_forward(const Tensor& input, const Tensor& mean, const Tensor& variance,
                         float epsilon, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("BatchNorm2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H = shape[2];
    const int64_t W = shape[3];

    Tensor output(std::vector<int64_t>{N, C, H, W}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* mean_ptr = get_data_ptr<const float>(mean);
        const float* var_ptr = get_data_ptr<const float>(variance);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<BatchNorm2dForwardKernelFloat32>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            const float m = mean_ptr[c];
            const float v = var_ptr[c];
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            const int64_t input_idx = ((n * C + c) * H * W) + hw;
            out_ptr[input_idx] = (in_ptr[input_idx] - m) * std_inv;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* mean_ptr = get_data_ptr<const double>(mean);
        const double* var_ptr = get_data_ptr<const double>(variance);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<BatchNorm2dForwardKernelFloat64>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            const double m = mean_ptr[c];
            const double v = var_ptr[c];
            const double std_inv = 1.0 / sycl::sqrt(v + static_cast<double>(epsilon));

            const int64_t input_idx = ((n * C + c) * H * W) + hw;
            out_ptr[input_idx] = (in_ptr[input_idx] - m) * std_inv;
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        // Mean/variance may be Float32 (from mean_var) or Float16 (running stats)
        const bool stats_f32 = (mean.dtype() == DType::Float32);
        const float* mean_f32 = stats_f32 ? get_data_ptr<const float>(mean) : nullptr;
        const float* var_f32 = stats_f32 ? get_data_ptr<const float>(variance) : nullptr;
        const sycl::half* mean_f16 = !stats_f32 ? get_data_ptr<const sycl::half>(mean) : nullptr;
        const sycl::half* var_f16 = !stats_f32 ? get_data_ptr<const sycl::half>(variance) : nullptr;
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<BatchNorm2dForwardKernelFloat16>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            // Use float for intermediate calculations for numerical stability
            const float m = stats_f32 ? mean_f32[c] : static_cast<float>(mean_f16[c]);
            const float v = stats_f32 ? var_f32[c] : static_cast<float>(var_f16[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            const int64_t input_idx = ((n * C + c) * H * W) + hw;
            const float val = static_cast<float>(in_ptr[input_idx]);
            out_ptr[input_idx] = sycl::half((val - m) * std_inv);
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        // Mean/variance may be Float32 (from mean_var) or BFloat16 (running stats)
        const bool stats_f32 = (mean.dtype() == DType::Float32);
        const float* mean_f32 = stats_f32 ? get_data_ptr<const float>(mean) : nullptr;
        const float* var_f32 = stats_f32 ? get_data_ptr<const float>(variance) : nullptr;
        const uint16_t* mean_bf16 = !stats_f32 ? get_data_ptr<const uint16_t>(mean) : nullptr;
        const uint16_t* var_bf16 = !stats_f32 ? get_data_ptr<const uint16_t>(variance) : nullptr;
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<BatchNorm2dForwardKernelBFloat16>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            // Use float for intermediate calculations for numerical stability
            const float m = stats_f32 ? mean_f32[c] : bf16_to_f32(mean_bf16[c]);
            const float v = stats_f32 ? var_f32[c] : bf16_to_f32(var_bf16[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            const int64_t input_idx = ((n * C + c) * H * W) + hw;
            const float val = bf16_to_f32(in_ptr[input_idx]);
            out_ptr[input_idx] = f32_to_bf16((val - m) * std_inv);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for batchnorm2d_forward (pure SYCL)");
    }

    return output;
}

auto batchnorm2d_forward_affine(const Tensor& input, const Tensor& mean, const Tensor& variance,
                                const Tensor& gamma, const Tensor& beta, float epsilon, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("BatchNorm2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H = shape[2];
    const int64_t W = shape[3];

    Tensor output(std::vector<int64_t>{N, C, H, W}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* mean_ptr = get_data_ptr<const float>(mean);
        const float* var_ptr = get_data_ptr<const float>(variance);
        const float* gamma_ptr = get_data_ptr<const float>(gamma);
        const float* beta_ptr = get_data_ptr<const float>(beta);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<BatchNorm2dForwardAffineKernelFloat32>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            const float m = mean_ptr[c];
            const float v = var_ptr[c];
            const float g = gamma_ptr[c];
            const float b = beta_ptr[c];
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            const int64_t input_idx = ((n * C + c) * H * W) + hw;
            out_ptr[input_idx] = g * (in_ptr[input_idx] - m) * std_inv + b;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* mean_ptr = get_data_ptr<const double>(mean);
        const double* var_ptr = get_data_ptr<const double>(variance);
        const double* gamma_ptr = get_data_ptr<const double>(gamma);
        const double* beta_ptr = get_data_ptr<const double>(beta);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<BatchNorm2dForwardAffineKernelFloat64>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            const double m = mean_ptr[c];
            const double v = var_ptr[c];
            const double g = gamma_ptr[c];
            const double b = beta_ptr[c];
            const double std_inv = 1.0 / sycl::sqrt(v + static_cast<double>(epsilon));

            const int64_t input_idx = ((n * C + c) * H * W) + hw;
            out_ptr[input_idx] = g * (in_ptr[input_idx] - m) * std_inv + b;
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        // Mean/variance may be Float32 (from mean_var) or Float16 (running stats)
        const bool stats_f32 = (mean.dtype() == DType::Float32);
        const float* mean_f32 = stats_f32 ? get_data_ptr<const float>(mean) : nullptr;
        const float* var_f32 = stats_f32 ? get_data_ptr<const float>(variance) : nullptr;
        const sycl::half* mean_f16 = !stats_f32 ? get_data_ptr<const sycl::half>(mean) : nullptr;
        const sycl::half* var_f16 = !stats_f32 ? get_data_ptr<const sycl::half>(variance) : nullptr;
        const sycl::half* gamma_ptr = get_data_ptr<const sycl::half>(gamma);
        const sycl::half* beta_ptr = get_data_ptr<const sycl::half>(beta);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<BatchNorm2dForwardAffineKernelFloat16>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            // Use float for intermediate calculations for numerical stability
            const float m = stats_f32 ? mean_f32[c] : static_cast<float>(mean_f16[c]);
            const float v = stats_f32 ? var_f32[c] : static_cast<float>(var_f16[c]);
            const float g = static_cast<float>(gamma_ptr[c]);
            const float b = static_cast<float>(beta_ptr[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            const int64_t input_idx = ((n * C + c) * H * W) + hw;
            const float val = static_cast<float>(in_ptr[input_idx]);
            out_ptr[input_idx] = sycl::half(g * (val - m) * std_inv + b);
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        // Mean/variance may be Float32 (from mean_var) or BFloat16 (running stats)
        const bool stats_f32 = (mean.dtype() == DType::Float32);
        const float* mean_f32 = stats_f32 ? get_data_ptr<const float>(mean) : nullptr;
        const float* var_f32 = stats_f32 ? get_data_ptr<const float>(variance) : nullptr;
        const uint16_t* mean_bf16 = !stats_f32 ? get_data_ptr<const uint16_t>(mean) : nullptr;
        const uint16_t* var_bf16 = !stats_f32 ? get_data_ptr<const uint16_t>(variance) : nullptr;
        const uint16_t* gamma_ptr = get_data_ptr<const uint16_t>(gamma);
        const uint16_t* beta_ptr = get_data_ptr<const uint16_t>(beta);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<BatchNorm2dForwardAffineKernelBFloat16>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            // Use float for intermediate calculations for numerical stability
            const float m = stats_f32 ? mean_f32[c] : bf16_to_f32(mean_bf16[c]);
            const float v = stats_f32 ? var_f32[c] : bf16_to_f32(var_bf16[c]);
            const float g = bf16_to_f32(gamma_ptr[c]);
            const float b = bf16_to_f32(beta_ptr[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            const int64_t input_idx = ((n * C + c) * H * W) + hw;
            const float val = bf16_to_f32(in_ptr[input_idx]);
            out_ptr[input_idx] = f32_to_bf16(g * (val - m) * std_inv + b);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for batchnorm2d_forward_affine (pure SYCL)");
    }

    return output;
}

auto batchnorm2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& mean,
                          const Tensor& variance, const Tensor& gamma, float epsilon, sycl::queue& queue)
                          -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape = input.shape();
    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H = shape[2];
    const int64_t W = shape[3];
    const int64_t spatial = H * W;

    Tensor grad_input(std::vector<int64_t>{N, C, H, W}, input.dtype(), input.device());
    Tensor grad_gamma({C}, input.dtype(), input.device());
    Tensor grad_beta({C}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* mean_ptr = get_data_ptr<const float>(mean);
        const float* var_ptr = get_data_ptr<const float>(variance);
        const float* gamma_ptr = get_data_ptr<const float>(gamma);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);
        float* grad_gamma_ptr = get_data_ptr<float>(grad_gamma);
        float* grad_beta_ptr = get_data_ptr<float>(grad_beta);

        // Compute grad_gamma and grad_beta
        queue.parallel_for<BatchNorm2dBackwardGammaKernelFloat32>(sycl::range<1>(C), [=](sycl::id<1> c) {
            float sum_grad_out = 0.0f;
            float sum_grad_out_norm = 0.0f;
            const float m = mean_ptr[c];
            const float v = var_ptr[c];
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t hw = 0; hw < spatial; ++hw) {
                    const int64_t idx = ((n * C + c) * spatial) + hw;
                    sum_grad_out += grad_out_ptr[idx];
                    sum_grad_out_norm += grad_out_ptr[idx] * (in_ptr[idx] - m) * std_inv;
                }
            }

            grad_gamma_ptr[c] = sum_grad_out_norm;
            grad_beta_ptr[c] = sum_grad_out;
        }).wait();

        // Compute grad_input
        const float scale = 1.0f / static_cast<float>(N * spatial);
        queue.parallel_for<BatchNorm2dBackwardInputKernelFloat32>(sycl::range<3>(N, C, spatial), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            const float m = mean_ptr[c];
            const float v = var_ptr[c];
            const float g = gamma_ptr[c];
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            const int64_t input_idx = ((n * C + c) * spatial) + hw;
            const float x_norm = (in_ptr[input_idx] - m) * std_inv;

            grad_in_ptr[input_idx] = g * std_inv * (grad_out_ptr[input_idx] -
                                                      grad_beta_ptr[c] * scale -
                                                      x_norm * grad_gamma_ptr[c] * scale);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* mean_ptr = get_data_ptr<const double>(mean);
        const double* var_ptr = get_data_ptr<const double>(variance);
        const double* gamma_ptr = get_data_ptr<const double>(gamma);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);
        double* grad_gamma_ptr = get_data_ptr<double>(grad_gamma);
        double* grad_beta_ptr = get_data_ptr<double>(grad_beta);

        // Compute grad_gamma and grad_beta
        queue.parallel_for<BatchNorm2dBackwardGammaKernelFloat64>(sycl::range<1>(C), [=](sycl::id<1> c) {
            double sum_grad_out = 0.0;
            double sum_grad_out_norm = 0.0;
            const double m = mean_ptr[c];
            const double v = var_ptr[c];
            const double std_inv = 1.0 / sycl::sqrt(v + static_cast<double>(epsilon));

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t hw = 0; hw < spatial; ++hw) {
                    const int64_t idx = ((n * C + c) * spatial) + hw;
                    sum_grad_out += grad_out_ptr[idx];
                    sum_grad_out_norm += grad_out_ptr[idx] * (in_ptr[idx] - m) * std_inv;
                }
            }

            grad_gamma_ptr[c] = sum_grad_out_norm;
            grad_beta_ptr[c] = sum_grad_out;
        }).wait();

        // Compute grad_input
        const double scale = 1.0 / static_cast<double>(N * spatial);
        queue.parallel_for<BatchNorm2dBackwardInputKernelFloat64>(sycl::range<3>(N, C, spatial), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            const double m = mean_ptr[c];
            const double v = var_ptr[c];
            const double g = gamma_ptr[c];
            const double std_inv = 1.0 / sycl::sqrt(v + static_cast<double>(epsilon));

            const int64_t input_idx = ((n * C + c) * spatial) + hw;
            const double x_norm = (in_ptr[input_idx] - m) * std_inv;

            grad_in_ptr[input_idx] = g * std_inv * (grad_out_ptr[input_idx] -
                                                      grad_beta_ptr[c] * scale -
                                                      x_norm * grad_gamma_ptr[c] * scale);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        // Mean/variance may be Float32 (from mean_var) or Float16 (running stats)
        const bool stats_f32 = (mean.dtype() == DType::Float32);
        const float* mean_f32 = stats_f32 ? get_data_ptr<const float>(mean) : nullptr;
        const float* var_f32 = stats_f32 ? get_data_ptr<const float>(variance) : nullptr;
        const sycl::half* mean_f16 = !stats_f32 ? get_data_ptr<const sycl::half>(mean) : nullptr;
        const sycl::half* var_f16 = !stats_f32 ? get_data_ptr<const sycl::half>(variance) : nullptr;
        const sycl::half* gamma_ptr = get_data_ptr<const sycl::half>(gamma);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);
        sycl::half* grad_gamma_ptr = get_data_ptr<sycl::half>(grad_gamma);
        sycl::half* grad_beta_ptr = get_data_ptr<sycl::half>(grad_beta);

        // Compute grad_gamma and grad_beta using float accumulation
        queue.parallel_for<BatchNorm2dBackwardGammaKernelFloat16>(sycl::range<1>(C), [=](sycl::id<1> c) {
            float sum_grad_out = 0.0f;
            float sum_grad_out_norm = 0.0f;
            const float m = stats_f32 ? mean_f32[c] : static_cast<float>(mean_f16[c]);
            const float v = stats_f32 ? var_f32[c] : static_cast<float>(var_f16[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t hw = 0; hw < spatial; ++hw) {
                    const int64_t idx = ((n * C + c) * spatial) + hw;
                    float grad_out_val = static_cast<float>(grad_out_ptr[idx]);
                    float in_val = static_cast<float>(in_ptr[idx]);
                    sum_grad_out += grad_out_val;
                    sum_grad_out_norm += grad_out_val * (in_val - m) * std_inv;
                }
            }

            grad_gamma_ptr[c] = sycl::half(sum_grad_out_norm);
            grad_beta_ptr[c] = sycl::half(sum_grad_out);
        }).wait();

        // Compute grad_input
        const float scale = 1.0f / static_cast<float>(N * spatial);
        queue.parallel_for<BatchNorm2dBackwardInputKernelFloat16>(sycl::range<3>(N, C, spatial), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            const float m = stats_f32 ? mean_f32[c] : static_cast<float>(mean_f16[c]);
            const float v = stats_f32 ? var_f32[c] : static_cast<float>(var_f16[c]);
            const float g = static_cast<float>(gamma_ptr[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            const int64_t input_idx = ((n * C + c) * spatial) + hw;
            const float in_val = static_cast<float>(in_ptr[input_idx]);
            const float x_norm = (in_val - m) * std_inv;
            const float grad_out_val = static_cast<float>(grad_out_ptr[input_idx]);
            const float grad_beta_val = static_cast<float>(grad_beta_ptr[c]);
            const float grad_gamma_val = static_cast<float>(grad_gamma_ptr[c]);

            grad_in_ptr[input_idx] = sycl::half(g * std_inv * (grad_out_val -
                                                      grad_beta_val * scale -
                                                      x_norm * grad_gamma_val * scale));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        // Mean/variance may be Float32 (from mean_var) or BFloat16 (running stats)
        const bool stats_f32 = (mean.dtype() == DType::Float32);
        const float* mean_f32 = stats_f32 ? get_data_ptr<const float>(mean) : nullptr;
        const float* var_f32 = stats_f32 ? get_data_ptr<const float>(variance) : nullptr;
        const uint16_t* mean_bf16 = !stats_f32 ? get_data_ptr<const uint16_t>(mean) : nullptr;
        const uint16_t* var_bf16 = !stats_f32 ? get_data_ptr<const uint16_t>(variance) : nullptr;
        const uint16_t* gamma_ptr = get_data_ptr<const uint16_t>(gamma);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);
        uint16_t* grad_gamma_ptr = get_data_ptr<uint16_t>(grad_gamma);
        uint16_t* grad_beta_ptr = get_data_ptr<uint16_t>(grad_beta);

        // Compute grad_gamma and grad_beta using float accumulation
        queue.parallel_for<BatchNorm2dBackwardGammaKernelBFloat16>(sycl::range<1>(C), [=](sycl::id<1> c) {
            float sum_grad_out = 0.0f;
            float sum_grad_out_norm = 0.0f;
            const float m = stats_f32 ? mean_f32[c] : bf16_to_f32(mean_bf16[c]);
            const float v = stats_f32 ? var_f32[c] : bf16_to_f32(var_bf16[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t hw = 0; hw < spatial; ++hw) {
                    const int64_t idx = ((n * C + c) * spatial) + hw;
                    float grad_out_val = bf16_to_f32(grad_out_ptr[idx]);
                    float in_val = bf16_to_f32(in_ptr[idx]);
                    sum_grad_out += grad_out_val;
                    sum_grad_out_norm += grad_out_val * (in_val - m) * std_inv;
                }
            }

            grad_gamma_ptr[c] = f32_to_bf16(sum_grad_out_norm);
            grad_beta_ptr[c] = f32_to_bf16(sum_grad_out);
        }).wait();

        // Compute grad_input
        const float scale = 1.0f / static_cast<float>(N * spatial);
        queue.parallel_for<BatchNorm2dBackwardInputKernelBFloat16>(sycl::range<3>(N, C, spatial), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            const float m = stats_f32 ? mean_f32[c] : bf16_to_f32(mean_bf16[c]);
            const float v = stats_f32 ? var_f32[c] : bf16_to_f32(var_bf16[c]);
            const float g = bf16_to_f32(gamma_ptr[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            const int64_t input_idx = ((n * C + c) * spatial) + hw;
            const float in_val = bf16_to_f32(in_ptr[input_idx]);
            const float x_norm = (in_val - m) * std_inv;
            const float grad_out_val = bf16_to_f32(grad_out_ptr[input_idx]);
            const float grad_beta_val = bf16_to_f32(grad_beta_ptr[c]);
            const float grad_gamma_val = bf16_to_f32(grad_gamma_ptr[c]);

            grad_in_ptr[input_idx] = f32_to_bf16(g * std_inv * (grad_out_val -
                                                      grad_beta_val * scale -
                                                      x_norm * grad_gamma_val * scale));
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for batchnorm2d_backward (pure SYCL)");
    }

    return {grad_input, grad_gamma, grad_beta};
}

#endif // TENZOR_HAS_ONEDNN

// Update running statistics: running = (1 - momentum) * running + momentum * batch
auto batchnorm2d_update_running_stats(Tensor& running_mean, Tensor& running_var,
                                      const Tensor& batch_mean, const Tensor& batch_var,
                                      float momentum, sycl::queue& queue) -> void {
    const int64_t C = batch_mean.shape()[0];

    if (running_mean.dtype() == DType::Float32) {
        float* run_mean_ptr = get_data_ptr<float>(running_mean);
        float* run_var_ptr = get_data_ptr<float>(running_var);
        const float* batch_mean_ptr = get_data_ptr<const float>(batch_mean);
        const float* batch_var_ptr = get_data_ptr<const float>(batch_var);

        queue.parallel_for<BatchNorm2dUpdateRunningStatsKernelFloat32>(sycl::range<1>(C), [=](sycl::id<1> c_id) {
            const int64_t c = c_id[0];
            run_mean_ptr[c] = (1.0f - momentum) * run_mean_ptr[c] + momentum * batch_mean_ptr[c];
            run_var_ptr[c] = (1.0f - momentum) * run_var_ptr[c] + momentum * batch_var_ptr[c];
        }).wait();
    }
    else if (running_mean.dtype() == DType::Float64) {
        double* run_mean_ptr = get_data_ptr<double>(running_mean);
        double* run_var_ptr = get_data_ptr<double>(running_var);
        const double* batch_mean_ptr = get_data_ptr<const double>(batch_mean);
        const double* batch_var_ptr = get_data_ptr<const double>(batch_var);

        const double momentum_d = static_cast<double>(momentum);
        queue.parallel_for<BatchNorm2dUpdateRunningStatsKernelFloat64>(sycl::range<1>(C), [=](sycl::id<1> c_id) {
            const int64_t c = c_id[0];
            run_mean_ptr[c] = (1.0 - momentum_d) * run_mean_ptr[c] + momentum_d * batch_mean_ptr[c];
            run_var_ptr[c] = (1.0 - momentum_d) * run_var_ptr[c] + momentum_d * batch_var_ptr[c];
        }).wait();
    }
    else if (running_mean.dtype() == DType::Float16) {
        sycl::half* run_mean_ptr = get_data_ptr<sycl::half>(running_mean);
        sycl::half* run_var_ptr = get_data_ptr<sycl::half>(running_var);
        const sycl::half* batch_mean_ptr = get_data_ptr<const sycl::half>(batch_mean);
        const sycl::half* batch_var_ptr = get_data_ptr<const sycl::half>(batch_var);

        queue.parallel_for<BatchNorm2dUpdateRunningStatsKernelFloat16>(sycl::range<1>(C), [=](sycl::id<1> c_id) {
            const int64_t c = c_id[0];
            // Use float for intermediate calculations for numerical stability
            float run_mean = static_cast<float>(run_mean_ptr[c]);
            float run_var = static_cast<float>(run_var_ptr[c]);
            float batch_mean = static_cast<float>(batch_mean_ptr[c]);
            float batch_var = static_cast<float>(batch_var_ptr[c]);

            run_mean_ptr[c] = sycl::half((1.0f - momentum) * run_mean + momentum * batch_mean);
            run_var_ptr[c] = sycl::half((1.0f - momentum) * run_var + momentum * batch_var);
        }).wait();
    }
    else if (running_mean.dtype() == DType::BFloat16) {
        uint16_t* run_mean_ptr = get_data_ptr<uint16_t>(running_mean);
        uint16_t* run_var_ptr = get_data_ptr<uint16_t>(running_var);
        const uint16_t* batch_mean_ptr = get_data_ptr<const uint16_t>(batch_mean);
        const uint16_t* batch_var_ptr = get_data_ptr<const uint16_t>(batch_var);

        queue.parallel_for<BatchNorm2dUpdateRunningStatsKernelBFloat16>(sycl::range<1>(C), [=](sycl::id<1> c_id) {
            const int64_t c = c_id[0];
            // Use float for intermediate calculations for numerical stability
            float run_mean = bf16_to_f32(run_mean_ptr[c]);
            float run_var = bf16_to_f32(run_var_ptr[c]);
            float batch_mean = bf16_to_f32(batch_mean_ptr[c]);
            float batch_var = bf16_to_f32(batch_var_ptr[c]);

            run_mean_ptr[c] = f32_to_bf16((1.0f - momentum) * run_mean + momentum * batch_mean);
            run_var_ptr[c] = f32_to_bf16((1.0f - momentum) * run_var + momentum * batch_var);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for batchnorm2d_update_running_stats");
    }
}

// Compute per-channel mean and variance
auto batchnorm2d_mean_var(const Tensor& input, sycl::queue& queue) -> std::vector<Tensor> {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d_mean_var expects 4D input (NCHW)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H = shape[2];
    const int64_t W = shape[3];
    const int64_t spatial = H * W;
    const int64_t total_elements = N * spatial;

    if (total_elements == 0) {
        throw std::runtime_error("BatchNorm2d: Cannot compute mean/variance for empty tensor");
    }

    // For Float16/BFloat16 inputs, store mean/variance in Float32 to avoid overflow
    DType stats_dtype = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) ? DType::Float32 : input.dtype();
    Tensor mean({C}, stats_dtype, input.device());
    Tensor variance({C}, stats_dtype, input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* mean_ptr = get_data_ptr<float>(mean);
        float* var_ptr = get_data_ptr<float>(variance);

        // Compute mean for each channel
        queue.parallel_for<BatchNorm2dMeanKernelFloat32>(sycl::range<1>(C), [=](sycl::id<1> c_id) {
            const int64_t c = c_id[0];
            float sum = 0.0f;

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        const int64_t idx = ((n * C + c) * H + h) * W + w;
                        sum += in_ptr[idx];
                    }
                }
            }

            mean_ptr[c] = sum / static_cast<float>(total_elements);
        }).wait();

        // Compute variance for each channel
        queue.parallel_for<BatchNorm2dVarianceKernelFloat32>(sycl::range<1>(C), [=](sycl::id<1> c_id) {
            const int64_t c = c_id[0];
            const float channel_mean = mean_ptr[c];
            float sum_sq_diff = 0.0f;

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        const int64_t idx = ((n * C + c) * H + h) * W + w;
                        const float diff = in_ptr[idx] - channel_mean;
                        sum_sq_diff += diff * diff;
                    }
                }
            }

            var_ptr[c] = sum_sq_diff / static_cast<float>(total_elements);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* mean_ptr = get_data_ptr<double>(mean);
        double* var_ptr = get_data_ptr<double>(variance);

        // Compute mean for each channel
        queue.parallel_for<BatchNorm2dMeanKernelFloat64>(sycl::range<1>(C), [=](sycl::id<1> c_id) {
            const int64_t c = c_id[0];
            double sum = 0.0;

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        const int64_t idx = ((n * C + c) * H + h) * W + w;
                        sum += in_ptr[idx];
                    }
                }
            }

            mean_ptr[c] = sum / static_cast<double>(total_elements);
        }).wait();

        // Compute variance for each channel
        queue.parallel_for<BatchNorm2dVarianceKernelFloat64>(sycl::range<1>(C), [=](sycl::id<1> c_id) {
            const int64_t c = c_id[0];
            const double channel_mean = mean_ptr[c];
            double sum_sq_diff = 0.0;

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        const int64_t idx = ((n * C + c) * H + h) * W + w;
                        const double diff = in_ptr[idx] - channel_mean;
                        sum_sq_diff += diff * diff;
                    }
                }
            }

            var_ptr[c] = sum_sq_diff / static_cast<double>(total_elements);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        // Mean/variance stored as Float32 to avoid Float16 overflow
        float* mean_ptr = get_data_ptr<float>(mean);
        float* var_ptr = get_data_ptr<float>(variance);

        // Compute mean for each channel using float accumulation
        queue.parallel_for<BatchNorm2dMeanKernelFloat16>(sycl::range<1>(C), [=](sycl::id<1> c_id) {
            const int64_t c = c_id[0];
            float sum = 0.0f;

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        const int64_t idx = ((n * C + c) * H + h) * W + w;
                        sum += static_cast<float>(in_ptr[idx]);
                    }
                }
            }

            mean_ptr[c] = sum / static_cast<float>(total_elements);
        }).wait();

        // Compute variance for each channel using float accumulation
        queue.parallel_for<BatchNorm2dVarianceKernelFloat16>(sycl::range<1>(C), [=](sycl::id<1> c_id) {
            const int64_t c = c_id[0];
            const float channel_mean = mean_ptr[c];
            float sum_sq_diff = 0.0f;

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        const int64_t idx = ((n * C + c) * H + h) * W + w;
                        const float diff = static_cast<float>(in_ptr[idx]) - channel_mean;
                        sum_sq_diff += diff * diff;
                    }
                }
            }

            var_ptr[c] = sum_sq_diff / static_cast<float>(total_elements);
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        // Mean/variance stored as Float32 to avoid BFloat16 overflow
        float* mean_ptr = get_data_ptr<float>(mean);
        float* var_ptr = get_data_ptr<float>(variance);

        // Compute mean for each channel using float accumulation
        queue.parallel_for<BatchNorm2dMeanKernelBFloat16>(sycl::range<1>(C), [=](sycl::id<1> c_id) {
            const int64_t c = c_id[0];
            float sum = 0.0f;

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        const int64_t idx = ((n * C + c) * H + h) * W + w;
                        sum += bf16_to_f32(in_ptr[idx]);
                    }
                }
            }

            mean_ptr[c] = sum / static_cast<float>(total_elements);
        }).wait();

        // Compute variance for each channel using float accumulation
        queue.parallel_for<BatchNorm2dVarianceKernelBFloat16>(sycl::range<1>(C), [=](sycl::id<1> c_id) {
            const int64_t c = c_id[0];
            const float channel_mean = mean_ptr[c];
            float sum_sq_diff = 0.0f;

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        const int64_t idx = ((n * C + c) * H + h) * W + w;
                        const float diff = bf16_to_f32(in_ptr[idx]) - channel_mean;
                        sum_sq_diff += diff * diff;
                    }
                }
            }

            var_ptr[c] = sum_sq_diff / static_cast<float>(total_elements);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for batchnorm2d_mean_var (pure SYCL)");
    }

    return {mean, variance};
}

// ============================================================================
// Group Normalization
// ============================================================================

// SYCL kernel class declarations
class GroupNormMeanKernelFloat32;
class GroupNormMeanKernelFloat64;
class GroupNormMeanKernelFloat16;
class GroupNormVarianceKernelFloat32;
class GroupNormVarianceKernelFloat64;
class GroupNormVarianceKernelFloat16;
class GroupNormNormalizeKernelFloat32;
class GroupNormNormalizeKernelFloat64;
class GroupNormNormalizeKernelFloat16;
class GroupNormMeanKernelBFloat16;
class GroupNormVarianceKernelBFloat16;
class GroupNormNormalizeKernelBFloat16;

auto group_norm_kernel(const Tensor& input, int64_t num_groups,
                       const Tensor* weight, const Tensor* bias,
                       float eps, sycl::queue& queue) -> std::vector<Tensor> {
    auto shape = input.shape();
    if (shape.size() < 2) {
        throw std::invalid_argument("group_norm requires at least 2D input");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        spatial_size *= shape[i];
    }

    const int64_t channels_per_group = C / num_groups;
    const int64_t group_size = channels_per_group * spatial_size;

    // Output same shape/dtype as input; mean/inv_std are [N, num_groups] Float32
    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());
    Tensor mean_out({N, num_groups}, DType::Float32, input.device());
    Tensor inv_std_out({N, num_groups}, DType::Float32, input.device());

    float* mean_ptr = get_data_ptr<float>(mean_out);
    float* inv_std_ptr = get_data_ptr<float>(inv_std_out);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        const float* w_ptr = weight ? get_data_ptr<const float>(*weight) : nullptr;
        const float* b_ptr = bias ? get_data_ptr<const float>(*bias) : nullptr;

        // Pass 1: compute per-group mean
        queue.parallel_for<GroupNormMeanKernelFloat32>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;

                float sum = 0.0f;
                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        sum += in_ptr[(n * C + c) * spatial_size + s];
                    }
                }
                mean_ptr[n * num_groups + g] = sum / static_cast<float>(group_size);
            }
        );
        queue.wait();

        // Pass 2: compute per-group variance and inv_std
        queue.parallel_for<GroupNormVarianceKernelFloat32>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;
                const float m = mean_ptr[n * num_groups + g];

                float var = 0.0f;
                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        float diff = in_ptr[(n * C + c) * spatial_size + s] - m;
                        var += diff * diff;
                    }
                }
                var /= static_cast<float>(group_size);
                inv_std_ptr[n * num_groups + g] = 1.0f / sycl::sqrt(var + eps);
            }
        );
        queue.wait();

        // Pass 3: normalize + affine
        queue.parallel_for<GroupNormNormalizeKernelFloat32>(
            sycl::range<3>(N, C, spatial_size),
            [=](sycl::id<3> idx) {
                const int64_t n = idx[0];
                const int64_t c = idx[1];
                const int64_t s = idx[2];
                const int64_t g = c / channels_per_group;

                const float m = mean_ptr[n * num_groups + g];
                const float istd = inv_std_ptr[n * num_groups + g];
                const float w = w_ptr ? w_ptr[c] : 1.0f;
                const float b = b_ptr ? b_ptr[c] : 0.0f;

                const int64_t i = (n * C + c) * spatial_size + s;
                out_ptr[i] = (in_ptr[i] - m) * istd * w + b;
            }
        );
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double* w_ptr = weight ? get_data_ptr<const double>(*weight) : nullptr;
        const double* b_ptr = bias ? get_data_ptr<const double>(*bias) : nullptr;
        const double eps_d = static_cast<double>(eps);

        queue.parallel_for<GroupNormMeanKernelFloat64>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;

                double sum = 0.0;
                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        sum += in_ptr[(n * C + c) * spatial_size + s];
                    }
                }
                // Store mean as float (stats are always Float32)
                mean_ptr[n * num_groups + g] = static_cast<float>(sum / static_cast<double>(group_size));
            }
        );
        queue.wait();

        queue.parallel_for<GroupNormVarianceKernelFloat64>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;
                const double m = static_cast<double>(mean_ptr[n * num_groups + g]);

                double var = 0.0;
                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        double diff = in_ptr[(n * C + c) * spatial_size + s] - m;
                        var += diff * diff;
                    }
                }
                var /= static_cast<double>(group_size);
                inv_std_ptr[n * num_groups + g] = static_cast<float>(1.0 / sycl::sqrt(var + eps_d));
            }
        );
        queue.wait();

        queue.parallel_for<GroupNormNormalizeKernelFloat64>(
            sycl::range<3>(N, C, spatial_size),
            [=](sycl::id<3> idx) {
                const int64_t n = idx[0];
                const int64_t c = idx[1];
                const int64_t s = idx[2];
                const int64_t g = c / channels_per_group;

                const double m = static_cast<double>(mean_ptr[n * num_groups + g]);
                const double istd = static_cast<double>(inv_std_ptr[n * num_groups + g]);
                const double w = w_ptr ? w_ptr[c] : 1.0;
                const double b = b_ptr ? b_ptr[c] : 0.0;

                const int64_t i = (n * C + c) * spatial_size + s;
                out_ptr[i] = (in_ptr[i] - m) * istd * w + b;
            }
        );
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        const sycl::half* w_ptr = weight ? get_data_ptr<const sycl::half>(*weight) : nullptr;
        const sycl::half* b_ptr = bias ? get_data_ptr<const sycl::half>(*bias) : nullptr;

        // Use float accumulation for numerical stability
        queue.parallel_for<GroupNormMeanKernelFloat16>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;

                float sum = 0.0f;
                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        sum += static_cast<float>(in_ptr[(n * C + c) * spatial_size + s]);
                    }
                }
                mean_ptr[n * num_groups + g] = sum / static_cast<float>(group_size);
            }
        );
        queue.wait();

        queue.parallel_for<GroupNormVarianceKernelFloat16>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;
                const float m = mean_ptr[n * num_groups + g];

                float var = 0.0f;
                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        float diff = static_cast<float>(in_ptr[(n * C + c) * spatial_size + s]) - m;
                        var += diff * diff;
                    }
                }
                var /= static_cast<float>(group_size);
                inv_std_ptr[n * num_groups + g] = 1.0f / sycl::sqrt(var + eps);
            }
        );
        queue.wait();

        queue.parallel_for<GroupNormNormalizeKernelFloat16>(
            sycl::range<3>(N, C, spatial_size),
            [=](sycl::id<3> idx) {
                const int64_t n = idx[0];
                const int64_t c = idx[1];
                const int64_t s = idx[2];
                const int64_t g = c / channels_per_group;

                const float m = mean_ptr[n * num_groups + g];
                const float istd = inv_std_ptr[n * num_groups + g];
                const float w = w_ptr ? static_cast<float>(w_ptr[c]) : 1.0f;
                const float b = b_ptr ? static_cast<float>(b_ptr[c]) : 0.0f;

                const int64_t i = (n * C + c) * spatial_size + s;
                float val = static_cast<float>(in_ptr[i]);
                out_ptr[i] = sycl::half((val - m) * istd * w + b);
            }
        );
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        const uint16_t* w_ptr = weight ? get_data_ptr<const uint16_t>(*weight) : nullptr;
        const uint16_t* b_ptr = bias ? get_data_ptr<const uint16_t>(*bias) : nullptr;

        // Use float accumulation for numerical stability
        queue.parallel_for<GroupNormMeanKernelBFloat16>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;

                float sum = 0.0f;
                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        sum += bf16_to_f32(in_ptr[(n * C + c) * spatial_size + s]);
                    }
                }
                mean_ptr[n * num_groups + g] = sum / static_cast<float>(group_size);
            }
        );
        queue.wait();

        queue.parallel_for<GroupNormVarianceKernelBFloat16>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;
                const float m = mean_ptr[n * num_groups + g];

                float var = 0.0f;
                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        float diff = bf16_to_f32(in_ptr[(n * C + c) * spatial_size + s]) - m;
                        var += diff * diff;
                    }
                }
                var /= static_cast<float>(group_size);
                inv_std_ptr[n * num_groups + g] = 1.0f / sycl::sqrt(var + eps);
            }
        );
        queue.wait();

        queue.parallel_for<GroupNormNormalizeKernelBFloat16>(
            sycl::range<3>(N, C, spatial_size),
            [=](sycl::id<3> idx) {
                const int64_t n = idx[0];
                const int64_t c = idx[1];
                const int64_t s = idx[2];
                const int64_t g = c / channels_per_group;

                const float m = mean_ptr[n * num_groups + g];
                const float istd = inv_std_ptr[n * num_groups + g];
                const float w = w_ptr ? bf16_to_f32(w_ptr[c]) : 1.0f;
                const float b = b_ptr ? bf16_to_f32(b_ptr[c]) : 0.0f;

                const int64_t i = (n * C + c) * spatial_size + s;
                float val = bf16_to_f32(in_ptr[i]);
                out_ptr[i] = f32_to_bf16((val - m) * istd * w + b);
            }
        );
    }
    else {
        throw std::runtime_error("Unsupported dtype for group_norm_kernel");
    }

    return {output, mean_out, inv_std_out};
}

// ============================================================================
// Group Normalization Backward
// ============================================================================

// SYCL kernel class declarations
class GroupNormBackwardPassOneFloat32;
class GroupNormBackwardPassOneFloat64;
class GroupNormBackwardPassOneFloat16;
class GroupNormBackwardPassTwoFloat32;
class GroupNormBackwardPassTwoFloat64;
class GroupNormBackwardPassTwoFloat16;
class GroupNormBackwardGradWBFloat32;
class GroupNormBackwardGradWBFloat64;
class GroupNormBackwardGradWBFloat16;
class GroupNormBackwardPassOneBFloat16;
class GroupNormBackwardPassTwoBFloat16;
class GroupNormBackwardGradWBBFloat16;

auto group_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                const Tensor& mean, const Tensor& rstd,
                                const Tensor& weight, int64_t num_groups,
                                sycl::queue& queue) -> std::vector<Tensor> {
    auto shape = input.shape();
    const int64_t N = shape[0];
    const int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        spatial_size *= shape[i];
    }
    const int64_t channels_per_group = C / num_groups;
    const int64_t group_size = channels_per_group * spatial_size;

    Tensor grad_input(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());
    Tensor grad_weight({C}, weight.dtype(), weight.device());
    Tensor grad_bias({C}, weight.dtype(), weight.device());

    // mean and rstd are Float32, shape [N, num_groups]
    const float* mean_ptr = get_data_ptr<const float>(mean);
    const float* rstd_ptr = get_data_ptr<const float>(rstd);

    if (input.dtype() == DType::Float32) {
        const float* go_ptr = get_data_ptr<const float>(grad_output);
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* w_ptr = get_data_ptr<const float>(weight);
        float* gi_ptr = get_data_ptr<float>(grad_input);
        float* gw_ptr = get_data_ptr<float>(grad_weight);
        float* gb_ptr = get_data_ptr<float>(grad_bias);

        // Zero grad_weight and grad_bias
        queue.memset(gw_ptr, 0, C * sizeof(float));
        queue.memset(gb_ptr, 0, C * sizeof(float));

        // Temp buffer for per-(n,g) ds and db
        Tensor ds_buf({N, num_groups}, DType::Float32, input.device());
        Tensor db_buf({N, num_groups}, DType::Float32, input.device());
        float* ds_ptr = get_data_ptr<float>(ds_buf);
        float* db_ptr = get_data_ptr<float>(db_buf);

        // Pass 1: compute ds, db per (n, g)
        queue.parallel_for<GroupNormBackwardPassOneFloat32>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;
                const float m = mean_ptr[n * num_groups + g];
                const float r = rstd_ptr[n * num_groups + g];

                float ds_val = 0.0f;
                float db_val = 0.0f;
                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        const int64_t i = (n * C + c) * spatial_size + s;
                        float dy = go_ptr[i];
                        float x_hat = (in_ptr[i] - m) * r;
                        float dy_w = dy * w_ptr[c];
                        ds_val += dy_w * x_hat;
                        db_val += dy_w;
                    }
                }
                ds_ptr[n * num_groups + g] = ds_val;
                db_ptr[n * num_groups + g] = db_val;
            }
        );

        // Pass 2: compute grad_input and accumulate grad_weight/grad_bias
        queue.parallel_for<GroupNormBackwardPassTwoFloat32>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;
                const float m = mean_ptr[n * num_groups + g];
                const float r = rstd_ptr[n * num_groups + g];
                const float ds_val = ds_ptr[n * num_groups + g];
                const float db_val = db_ptr[n * num_groups + g];
                const float inv_gs = 1.0f / static_cast<float>(group_size);

                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        const int64_t i = (n * C + c) * spatial_size + s;
                        float dy = go_ptr[i];
                        float x_hat = (in_ptr[i] - m) * r;
                        gi_ptr[i] = r * (dy * w_ptr[c] - inv_gs * (db_val + x_hat * ds_val));
                    }
                }
            }
        );

        // Pass 3: accumulate grad_weight/grad_bias across batch
        queue.parallel_for<GroupNormBackwardGradWBFloat32>(
            sycl::range<1>(C),
            [=](sycl::id<1> c_id) {
                const int64_t c = c_id[0];
                const int64_t g = c / channels_per_group;

                float gw_val = 0.0f;
                float gb_val = 0.0f;
                for (int64_t n = 0; n < N; ++n) {
                    const float m = mean_ptr[n * num_groups + g];
                    const float r = rstd_ptr[n * num_groups + g];
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        const int64_t i = (n * C + c) * spatial_size + s;
                        float x_hat = (in_ptr[i] - m) * r;
                        gw_val += go_ptr[i] * x_hat;
                        gb_val += go_ptr[i];
                    }
                }
                gw_ptr[c] = gw_val;
                gb_ptr[c] = gb_val;
            }
        );
    }
    else if (input.dtype() == DType::Float64) {
        const double* go_ptr = get_data_ptr<const double>(grad_output);
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* w_ptr = get_data_ptr<const double>(weight);
        double* gi_ptr = get_data_ptr<double>(grad_input);
        double* gw_ptr = get_data_ptr<double>(grad_weight);
        double* gb_ptr = get_data_ptr<double>(grad_bias);

        queue.memset(gw_ptr, 0, C * sizeof(double));
        queue.memset(gb_ptr, 0, C * sizeof(double));

        Tensor ds_buf({N, num_groups}, DType::Float32, input.device());
        Tensor db_buf({N, num_groups}, DType::Float32, input.device());
        float* ds_ptr = get_data_ptr<float>(ds_buf);
        float* db_ptr = get_data_ptr<float>(db_buf);

        queue.parallel_for<GroupNormBackwardPassOneFloat64>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;
                const double m = static_cast<double>(mean_ptr[n * num_groups + g]);
                const double r = static_cast<double>(rstd_ptr[n * num_groups + g]);

                double ds_val = 0.0;
                double db_val = 0.0;
                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        const int64_t i = (n * C + c) * spatial_size + s;
                        double dy = go_ptr[i];
                        double x_hat = (in_ptr[i] - m) * r;
                        double dy_w = dy * w_ptr[c];
                        ds_val += dy_w * x_hat;
                        db_val += dy_w;
                    }
                }
                ds_ptr[n * num_groups + g] = static_cast<float>(ds_val);
                db_ptr[n * num_groups + g] = static_cast<float>(db_val);
            }
        );

        queue.parallel_for<GroupNormBackwardPassTwoFloat64>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;
                const double m = static_cast<double>(mean_ptr[n * num_groups + g]);
                const double r = static_cast<double>(rstd_ptr[n * num_groups + g]);
                const double ds_val = static_cast<double>(ds_ptr[n * num_groups + g]);
                const double db_val = static_cast<double>(db_ptr[n * num_groups + g]);
                const double inv_gs = 1.0 / static_cast<double>(group_size);

                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        const int64_t i = (n * C + c) * spatial_size + s;
                        double dy = go_ptr[i];
                        double x_hat = (in_ptr[i] - m) * r;
                        gi_ptr[i] = r * (dy * w_ptr[c] - inv_gs * (db_val + x_hat * ds_val));
                    }
                }
            }
        );

        queue.parallel_for<GroupNormBackwardGradWBFloat64>(
            sycl::range<1>(C),
            [=](sycl::id<1> c_id) {
                const int64_t c = c_id[0];
                const int64_t g = c / channels_per_group;

                double gw_val = 0.0;
                double gb_val = 0.0;
                for (int64_t n = 0; n < N; ++n) {
                    const double m = static_cast<double>(mean_ptr[n * num_groups + g]);
                    const double r = static_cast<double>(rstd_ptr[n * num_groups + g]);
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        const int64_t i = (n * C + c) * spatial_size + s;
                        double x_hat = (in_ptr[i] - m) * r;
                        gw_val += go_ptr[i] * x_hat;
                        gb_val += go_ptr[i];
                    }
                }
                gw_ptr[c] = gw_val;
                gb_ptr[c] = gb_val;
            }
        );
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* go_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const sycl::half* w_ptr = get_data_ptr<const sycl::half>(weight);
        sycl::half* gi_ptr = get_data_ptr<sycl::half>(grad_input);
        sycl::half* gw_ptr = get_data_ptr<sycl::half>(grad_weight);
        sycl::half* gb_ptr = get_data_ptr<sycl::half>(grad_bias);

        // Use float intermediates
        Tensor ds_buf({N, num_groups}, DType::Float32, input.device());
        Tensor db_buf({N, num_groups}, DType::Float32, input.device());
        float* ds_ptr = get_data_ptr<float>(ds_buf);
        float* db_ptr = get_data_ptr<float>(db_buf);

        queue.parallel_for<GroupNormBackwardPassOneFloat16>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;
                const float m = mean_ptr[n * num_groups + g];
                const float r = rstd_ptr[n * num_groups + g];

                float ds_val = 0.0f;
                float db_val = 0.0f;
                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        const int64_t i = (n * C + c) * spatial_size + s;
                        float dy = static_cast<float>(go_ptr[i]);
                        float x_hat = (static_cast<float>(in_ptr[i]) - m) * r;
                        float dy_w = dy * static_cast<float>(w_ptr[c]);
                        ds_val += dy_w * x_hat;
                        db_val += dy_w;
                    }
                }
                ds_ptr[n * num_groups + g] = ds_val;
                db_ptr[n * num_groups + g] = db_val;
            }
        );

        queue.parallel_for<GroupNormBackwardPassTwoFloat16>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;
                const float m = mean_ptr[n * num_groups + g];
                const float r = rstd_ptr[n * num_groups + g];
                const float ds_val = ds_ptr[n * num_groups + g];
                const float db_val = db_ptr[n * num_groups + g];
                const float inv_gs = 1.0f / static_cast<float>(group_size);

                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        const int64_t i = (n * C + c) * spatial_size + s;
                        float dy = static_cast<float>(go_ptr[i]);
                        float x_hat = (static_cast<float>(in_ptr[i]) - m) * r;
                        float w = static_cast<float>(w_ptr[c]);
                        gi_ptr[i] = sycl::half(r * (dy * w - inv_gs * (db_val + x_hat * ds_val)));
                    }
                }
            }
        );

        queue.parallel_for<GroupNormBackwardGradWBFloat16>(
            sycl::range<1>(C),
            [=](sycl::id<1> c_id) {
                const int64_t c = c_id[0];
                const int64_t g = c / channels_per_group;

                float gw_val = 0.0f;
                float gb_val = 0.0f;
                for (int64_t n = 0; n < N; ++n) {
                    const float m = mean_ptr[n * num_groups + g];
                    const float r = rstd_ptr[n * num_groups + g];
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        const int64_t i = (n * C + c) * spatial_size + s;
                        float x_hat = (static_cast<float>(in_ptr[i]) - m) * r;
                        gw_val += static_cast<float>(go_ptr[i]) * x_hat;
                        gb_val += static_cast<float>(go_ptr[i]);
                    }
                }
                gw_ptr[c] = sycl::half(gw_val);
                gb_ptr[c] = sycl::half(gb_val);
            }
        );
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* go_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        const uint16_t* w_ptr = get_data_ptr<const uint16_t>(weight);
        uint16_t* gi_ptr = get_data_ptr<uint16_t>(grad_input);
        uint16_t* gw_ptr = get_data_ptr<uint16_t>(grad_weight);
        uint16_t* gb_ptr = get_data_ptr<uint16_t>(grad_bias);

        // Use float intermediates
        Tensor ds_buf({N, num_groups}, DType::Float32, input.device());
        Tensor db_buf({N, num_groups}, DType::Float32, input.device());
        float* ds_ptr = get_data_ptr<float>(ds_buf);
        float* db_ptr = get_data_ptr<float>(db_buf);

        queue.parallel_for<GroupNormBackwardPassOneBFloat16>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;
                const float m = mean_ptr[n * num_groups + g];
                const float r = rstd_ptr[n * num_groups + g];

                float ds_val = 0.0f;
                float db_val = 0.0f;
                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        const int64_t i = (n * C + c) * spatial_size + s;
                        float dy = bf16_to_f32(go_ptr[i]);
                        float x_hat = (bf16_to_f32(in_ptr[i]) - m) * r;
                        float dy_w = dy * bf16_to_f32(w_ptr[c]);
                        ds_val += dy_w * x_hat;
                        db_val += dy_w;
                    }
                }
                ds_ptr[n * num_groups + g] = ds_val;
                db_ptr[n * num_groups + g] = db_val;
            }
        );

        queue.parallel_for<GroupNormBackwardPassTwoBFloat16>(
            sycl::range<2>(N, num_groups),
            [=](sycl::id<2> idx) {
                const int64_t n = idx[0];
                const int64_t g = idx[1];
                const int64_t c_start = g * channels_per_group;
                const float m = mean_ptr[n * num_groups + g];
                const float r = rstd_ptr[n * num_groups + g];
                const float ds_val = ds_ptr[n * num_groups + g];
                const float db_val = db_ptr[n * num_groups + g];
                const float inv_gs = 1.0f / static_cast<float>(group_size);

                for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        const int64_t i = (n * C + c) * spatial_size + s;
                        float dy = bf16_to_f32(go_ptr[i]);
                        float x_hat = (bf16_to_f32(in_ptr[i]) - m) * r;
                        float w = bf16_to_f32(w_ptr[c]);
                        gi_ptr[i] = f32_to_bf16(r * (dy * w - inv_gs * (db_val + x_hat * ds_val)));
                    }
                }
            }
        );

        queue.parallel_for<GroupNormBackwardGradWBBFloat16>(
            sycl::range<1>(C),
            [=](sycl::id<1> c_id) {
                const int64_t c = c_id[0];
                const int64_t g = c / channels_per_group;

                float gw_val = 0.0f;
                float gb_val = 0.0f;
                for (int64_t n = 0; n < N; ++n) {
                    const float m = mean_ptr[n * num_groups + g];
                    const float r = rstd_ptr[n * num_groups + g];
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        const int64_t i = (n * C + c) * spatial_size + s;
                        float x_hat = (bf16_to_f32(in_ptr[i]) - m) * r;
                        gw_val += bf16_to_f32(go_ptr[i]) * x_hat;
                        gb_val += bf16_to_f32(go_ptr[i]);
                    }
                }
                gw_ptr[c] = f32_to_bf16(gw_val);
                gb_ptr[c] = f32_to_bf16(gb_val);
            }
        );
    }
    else {
        throw std::runtime_error("Unsupported dtype for group_norm_backward_kernel");
    }

    return {grad_input, grad_weight, grad_bias};
}

} // namespace oneapi
} // namespace tenzor
