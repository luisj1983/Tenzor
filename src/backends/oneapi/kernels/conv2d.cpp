#include "tenzor/core/tensor.hpp"
#include <CL/sycl.hpp>
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

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

#ifdef TENZOR_HAS_ONEDNN

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

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32,
                                   groups == 1 ? memory::format_tag::oihw : memory::format_tag::goihw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    // Create convolution descriptor
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

    if (bias != nullptr) {
        auto bias_mem = sycl_interop::make_memory(
            memory::desc({C_out}, memory::data_type::f32, memory::format_tag::x),
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

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32,
                                   groups == 1 ? memory::format_tag::oihw : memory::format_tag::goihw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

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
        conv_bwd_data_prim.execute(dnnl_stream, {
            {DNNL_ARG_DIFF_DST, diff_dst_mem},
            {DNNL_ARG_WEIGHTS, weights_mem},
            {DNNL_ARG_DIFF_SRC, diff_src_mem}
        });
    }

    // Compute grad_weight and grad_bias
    if (compute_grad_weight || compute_grad_bias) {
        grad_weight = Tensor(std::vector<int64_t>(weight_shape.begin(), weight_shape.end()), weight.dtype(), weight.device());

        auto conv_bwd_weights_desc = compute_grad_bias ?
            convolution_backward_weights::desc(
                algorithm::convolution_direct,
                src_md, weights_md,
                memory::desc({C_out}, memory::data_type::f32, memory::format_tag::x),
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
                memory::desc({C_out}, memory::data_type::f32, memory::format_tag::x),
                dnnl_engine,
                sycl_interop::memory_kind::usm,
                const_cast<void*>(grad_bias.data_ptr())
            );

            conv_bwd_weights_prim.execute(dnnl_stream, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_DIFF_DST, diff_dst_mem},
                {DNNL_ARG_DIFF_WEIGHTS, diff_weights_mem},
                {DNNL_ARG_DIFF_BIAS, diff_bias_mem}
            });
        } else {
            conv_bwd_weights_prim.execute(dnnl_stream, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_DIFF_DST, diff_dst_mem},
                {DNNL_ARG_DIFF_WEIGHTS, diff_weights_mem}
            });
        }
    }

    dnnl_stream.wait();

    return {grad_input, grad_weight, grad_bias};
}

#else // !TENZOR_HAS_ONEDNN - Fallback implementation using im2col + GEMM

// Im2col transformation for convolution
template<typename T>
void im2col_kernel(const T* data_im, int64_t channels, int64_t height, int64_t width,
                   int64_t kernel_h, int64_t kernel_w, int64_t pad, int64_t stride,
                   int64_t dilation, T* data_col, sycl::queue& queue) {
    const int64_t output_h = (height + 2 * pad - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int64_t output_w = (width + 2 * pad - dilation * (kernel_w - 1) - 1) / stride + 1;
    const int64_t col_size = channels * kernel_h * kernel_w * output_h * output_w;

    queue.parallel_for(sycl::range<1>(col_size), [=](sycl::id<1> index) {
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
    }).wait();
}

auto conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                    int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                    sycl::queue& queue) -> Tensor {
    if (groups != 1) {
        throw std::runtime_error("Conv2d fallback: grouped convolutions not yet supported");
    }

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t H_in = input_shape[2];
    const int64_t W_in = input_shape[3];

    const int64_t C_out = weight_shape[0];
    const int64_t K_h = weight_shape[2];
    const int64_t K_w = weight_shape[3];

    const int64_t H_out = (H_in + 2 * padding - dilation * (K_h - 1) - 1) / stride + 1;
    const int64_t W_out = (W_in + 2 * padding - dilation * (K_w - 1) - 1) / stride + 1;

    Tensor output({N, C_out, H_out, W_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr<const float>(input);
        const float* weight_ptr = get_data_ptr<const float>(weight);
        float* output_ptr = get_data_ptr<float>(output);

        // Im2col buffer
        const int64_t col_size = C_in * K_h * K_w * H_out * W_out;
        Tensor col_buffer({col_size}, input.dtype(), input.device());
        float* col_ptr = get_data_ptr<float>(col_buffer);

        // Process each batch
        for (int64_t n = 0; n < N; ++n) {
            // Im2col
            im2col_kernel<float>(
                input_ptr + n * C_in * H_in * W_in,
                C_in, H_in, W_in, K_h, K_w, padding, stride, dilation,
                col_ptr, queue
            );

#ifdef TENZOR_HAS_ONEMKL
            // GEMM: output = weight * col
            const int64_t M = C_out;
            const int64_t N_gemm = H_out * W_out;
            const int64_t K = C_in * K_h * K_w;

            oneapi::mkl::blas::gemm(
                queue,
                oneapi::mkl::transpose::nontrans,
                oneapi::mkl::transpose::nontrans,
                N_gemm, M, K,
                1.0f,
                col_ptr, N_gemm,
                weight_ptr, K,
                0.0f,
                output_ptr + n * C_out * H_out * W_out, N_gemm
            );
            queue.wait();
#else
            // Naive GEMM fallback
            queue.parallel_for(sycl::range<2>(C_out, H_out * W_out), [=](sycl::id<2> idx) {
                const int64_t oc = idx[0];
                const int64_t hw = idx[1];

                float sum = 0.0f;
                for (int64_t k = 0; k < C_in * K_h * K_w; ++k) {
                    sum += weight_ptr[oc * C_in * K_h * K_w + k] * col_ptr[k * H_out * W_out + hw];
                }
                output_ptr[n * C_out * H_out * W_out + oc * H_out * W_out + hw] = sum;
            }).wait();
#endif

            // Add bias if present
            if (bias != nullptr) {
                const float* bias_ptr = get_data_ptr<const float>(*bias);
                queue.parallel_for(sycl::range<2>(C_out, H_out * W_out), [=](sycl::id<2> idx) {
                    const int64_t oc = idx[0];
                    const int64_t hw = idx[1];
                    const int64_t out_idx = n * C_out * H_out * W_out + oc * H_out * W_out + hw;
                    output_ptr[out_idx] += bias_ptr[oc];
                }).wait();
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
    // Simplified backward implementation
    Tensor grad_input, grad_weight, grad_bias;

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    if (compute_grad_input) {
        grad_input = Tensor(std::vector<int64_t>(input_shape.begin(), input_shape.end()), input.dtype(), input.device());
        // TODO: Implement grad_input computation
    }

    if (compute_grad_weight) {
        grad_weight = Tensor(std::vector<int64_t>(weight_shape.begin(), weight_shape.end()), weight.dtype(), weight.device());
        // TODO: Implement grad_weight computation
    }

    if (compute_grad_bias) {
        grad_bias = Tensor({weight_shape[0]}, weight.dtype(), weight.device());
        // Sum grad_output over batch, height, width dimensions
        const int64_t N = grad_output.shape()[0];
        const int64_t C = grad_output.shape()[1];
        const int64_t H = grad_output.shape()[2];
        const int64_t W = grad_output.shape()[3];

        if (grad_output.dtype() == DType::Float32) {
            const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
            float* grad_bias_ptr = get_data_ptr<float>(grad_bias);

            queue.parallel_for(sycl::range<1>(C), [=](sycl::id<1> c) {
                float sum = 0.0f;
                for (int64_t n = 0; n < N; ++n) {
                    for (int64_t h = 0; h < H; ++h) {
                        for (int64_t w = 0; w < W; ++w) {
                            sum += grad_out_ptr[((n * C + c) * H + h) * W + w];
                        }
                    }
                }
                grad_bias_ptr[c] = sum;
            }).wait();
        }
    }

    return {grad_input, grad_weight, grad_bias};
}

#endif // TENZOR_HAS_ONEDNN

} // namespace oneapi
} // namespace tenzor
