#include "tenzor/core/tensor.hpp"
#include <CL/sycl.hpp>
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
class BatchNorm2dForwardAffineKernelFloat32;
class BatchNorm2dBackwardGammaKernelFloat32;
class BatchNorm2dBackwardInputKernelFloat32;
class BatchNormScaleShiftKernelFloat32;
class BatchNormExtractGradKernelFloat32;

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
    else {
        throw std::runtime_error("Unsupported dtype for batchnorm2d_backward (pure SYCL)");
    }

    return {grad_input, grad_gamma, grad_beta};
}

#endif // TENZOR_HAS_ONEDNN

} // namespace oneapi
} // namespace tenzor
