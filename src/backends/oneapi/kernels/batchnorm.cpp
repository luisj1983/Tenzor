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
class BatchNorm2dVarianceKernelFloat32;
class BatchNorm2dUpdateRunningStatsKernelFloat32;

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

#ifdef TENZOR_HAS_ONEDNN

// Batch normalization forward using oneDNN
auto batchnorm2d_forward(const Tensor& input, const Tensor& mean, const Tensor& variance,
                         float epsilon, sycl::queue& queue) -> Tensor {
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
    Tensor scale_shift({2, C}, input.dtype(), input.device());
    float* ss_ptr = get_data_ptr<float>(scale_shift);
    const float* gamma_ptr = get_data_ptr<const float>(gamma);
    const float* beta_ptr = get_data_ptr<const float>(beta);

    queue.parallel_for<BatchNormScaleShiftKernelFloat32>(sycl::range<1>(C), [=](sycl::id<1> i) {
        ss_ptr[i] = gamma_ptr[i];           // Scale
        ss_ptr[C + i] = beta_ptr[i];        // Shift
    }).wait();

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

    // Scale-shift for forward
    Tensor scale_shift({2, C}, input.dtype(), input.device());
    float* ss_ptr = get_data_ptr<float>(scale_shift);
    const float* gamma_ptr = get_data_ptr<const float>(gamma);

    queue.parallel_for<BatchNormScaleShiftKernelFloat32>(sycl::range<1>(C), [=](sycl::id<1> i) {
        ss_ptr[i] = gamma_ptr[i];
        ss_ptr[C + i] = 0.0f;  // Beta not used in backward
    }).wait();

    auto ss_mem = sycl_interop::make_memory(
        memory::desc({2, C}, memory::data_type::f32, memory::format_tag::nc),
        dnnl_engine,
        sycl_interop::memory_kind::usm,
        const_cast<void*>(scale_shift.data_ptr())
    );

    // Diff scale-shift for gradients
    Tensor diff_scale_shift({2, C}, input.dtype(), input.device());
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
    }).wait();

    return {grad_input, grad_gamma, grad_beta};
}

#else // !TENZOR_HAS_ONEDNN

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
        const sycl::half* mean_ptr = get_data_ptr<const sycl::half>(mean);
        const sycl::half* var_ptr = get_data_ptr<const sycl::half>(variance);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<BatchNorm2dForwardKernelFloat16>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            // Use float for intermediate calculations for numerical stability
            const float m = static_cast<float>(mean_ptr[c]);
            const float v = static_cast<float>(var_ptr[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            const int64_t input_idx = ((n * C + c) * H * W) + hw;
            const float val = static_cast<float>(in_ptr[input_idx]);
            out_ptr[input_idx] = sycl::half((val - m) * std_inv);
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
        const sycl::half* mean_ptr = get_data_ptr<const sycl::half>(mean);
        const sycl::half* var_ptr = get_data_ptr<const sycl::half>(variance);
        const sycl::half* gamma_ptr = get_data_ptr<const sycl::half>(gamma);
        const sycl::half* beta_ptr = get_data_ptr<const sycl::half>(beta);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<BatchNorm2dForwardAffineKernelFloat16>(sycl::range<3>(N, C, H * W), [=](sycl::id<3> idx) {
            const int64_t n = idx[0];
            const int64_t c = idx[1];
            const int64_t hw = idx[2];

            // Use float for intermediate calculations for numerical stability
            const float m = static_cast<float>(mean_ptr[c]);
            const float v = static_cast<float>(var_ptr[c]);
            const float g = static_cast<float>(gamma_ptr[c]);
            const float b = static_cast<float>(beta_ptr[c]);
            const float std_inv = 1.0f / sycl::sqrt(v + epsilon);

            const int64_t input_idx = ((n * C + c) * H * W) + hw;
            const float val = static_cast<float>(in_ptr[input_idx]);
            out_ptr[input_idx] = sycl::half(g * (val - m) * std_inv + b);
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
        const sycl::half* mean_ptr = get_data_ptr<const sycl::half>(mean);
        const sycl::half* var_ptr = get_data_ptr<const sycl::half>(variance);
        const sycl::half* gamma_ptr = get_data_ptr<const sycl::half>(gamma);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);
        sycl::half* grad_gamma_ptr = get_data_ptr<sycl::half>(grad_gamma);
        sycl::half* grad_beta_ptr = get_data_ptr<sycl::half>(grad_beta);

        // Compute grad_gamma and grad_beta using float accumulation
        queue.parallel_for<BatchNorm2dBackwardGammaKernelFloat16>(sycl::range<1>(C), [=](sycl::id<1> c) {
            float sum_grad_out = 0.0f;
            float sum_grad_out_norm = 0.0f;
            const float m = static_cast<float>(mean_ptr[c]);
            const float v = static_cast<float>(var_ptr[c]);
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

            const float m = static_cast<float>(mean_ptr[c]);
            const float v = static_cast<float>(var_ptr[c]);
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

    Tensor mean({C}, input.dtype(), input.device());
    Tensor variance({C}, input.dtype(), input.device());

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
    else {
        throw std::runtime_error("Unsupported dtype for batchnorm2d_mean_var (pure SYCL)");
    }

    return {mean, variance};
}

} // namespace oneapi
} // namespace tenzor
