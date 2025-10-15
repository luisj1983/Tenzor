#ifdef TENZOR_HAS_CUDNN

#include "tenzor/backend/cudnn_wrapper.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace tenzor {
namespace cuda {

// ============================================================================
// cuDNN Conv2d Forward Implementation
// ============================================================================

auto cudnn_conv2d_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions
    int64_t out_h = (height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_w = (width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_h, out_w};
    Tensor output(output_shape, input.dtype(), input.device());

    // Create cuDNN handle
    CuDNNHandle handle;
    handle.set_stream(stream);

    // Convert dtype
    cudnnDataType_t cudnn_dtype;
    switch (input.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        default:
            throw std::runtime_error("cuDNN Conv2d: unsupported dtype");
    }

    // Create descriptors
    TensorDescriptor input_desc, output_desc;
    FilterDescriptor filter_desc;
    ConvolutionDescriptor conv_desc;

    input_desc.set(cudnn_dtype, batch, in_channels, height, width);
    output_desc.set(cudnn_dtype, batch, out_channels, out_h, out_w);
    filter_desc.set(cudnn_dtype, out_channels, in_channels / groups, kernel_h, kernel_w);
    conv_desc.set(padding, padding, stride, stride, dilation, dilation, cudnn_dtype);

    if (groups > 1) {
        conv_desc.set_group_count(groups);
    }

    // Find best convolution algorithm
    int requested_algo_count = 10;
    int returned_algo_count = 0;
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(requested_algo_count);

    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        handle.get(),
        input_desc.get(),
        filter_desc.get(),
        conv_desc.get(),
        output_desc.get(),
        requested_algo_count,
        &returned_algo_count,
        perf_results.data()
    ));

    // Use the fastest algorithm
    cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;
    size_t workspace_size = perf_results[0].memory;

    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Set alpha and beta for the operation: output = alpha * conv(input) + beta * output
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform convolution
    if (input.dtype() == DType::Float32) {
        CUDNN_CHECK(cudnnConvolutionForward(
            handle.get(),
            &alpha,
            input_desc.get(),
            input.data<float>(),
            filter_desc.get(),
            weight.data<float>(),
            conv_desc.get(),
            algo,
            workspace,
            workspace_size,
            &beta,
            output_desc.get(),
            output.data<float>()
        ));
    } else if (input.dtype() == DType::Float64) {
        const double alpha_d = 1.0;
        const double beta_d = 0.0;
        CUDNN_CHECK(cudnnConvolutionForward(
            handle.get(),
            &alpha_d,
            input_desc.get(),
            input.data<double>(),
            filter_desc.get(),
            weight.data<double>(),
            conv_desc.get(),
            algo,
            workspace,
            workspace_size,
            &beta_d,
            output_desc.get(),
            output.data<double>()
        ));
    } else if (input.dtype() == DType::Float16) {
        CUDNN_CHECK(cudnnConvolutionForward(
            handle.get(),
            &alpha,
            input_desc.get(),
            input.data<Float16>(),
            filter_desc.get(),
            weight.data<Float16>(),
            conv_desc.get(),
            algo,
            workspace,
            workspace_size,
            &beta,
            output_desc.get(),
            output.data<Float16>()
        ));
    }

    // Add bias if present
    if (bias != nullptr) {
        TensorDescriptor bias_desc;
        bias_desc.set(cudnn_dtype, 1, out_channels, 1, 1);

        const float alpha_bias = 1.0f;
        const float beta_bias = 1.0f;

        if (input.dtype() == DType::Float32) {
            CUDNN_CHECK(cudnnAddTensor(
                handle.get(),
                &alpha_bias,
                bias_desc.get(),
                bias->data<float>(),
                &beta_bias,
                output_desc.get(),
                output.data<float>()
            ));
        } else if (input.dtype() == DType::Float64) {
            const double alpha_bias_d = 1.0;
            const double beta_bias_d = 1.0;
            CUDNN_CHECK(cudnnAddTensor(
                handle.get(),
                &alpha_bias_d,
                bias_desc.get(),
                bias->data<double>(),
                &beta_bias_d,
                output_desc.get(),
                output.data<double>()
            ));
        } else if (input.dtype() == DType::Float16) {
            CUDNN_CHECK(cudnnAddTensor(
                handle.get(),
                &alpha_bias,
                bias_desc.get(),
                bias->data<Float16>(),
                &beta_bias,
                output_desc.get(),
                output.data<Float16>()
            ));
        }
    }

    // Cleanup workspace
    if (workspace) {
        cudaFree(workspace);
    }

    return output;
}

// ============================================================================
// cuDNN Conv2d Backward Implementation
// ============================================================================

auto cudnn_conv2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool compute_grad_input,
    bool compute_grad_weight,
    bool compute_grad_bias,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    auto grad_shape = grad_output.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h = grad_shape[2];
    int64_t out_w = grad_shape[3];

    // Initialize gradients
    Tensor grad_input({batch, in_channels, height, width}, input.dtype(), input.device());
    Tensor grad_weight({out_channels, in_channels / groups, kernel_h, kernel_w}, weight.dtype(), weight.device());
    Tensor grad_bias({out_channels}, weight.dtype(), weight.device());

    // Create cuDNN handle
    CuDNNHandle handle;
    handle.set_stream(stream);

    // Convert dtype
    cudnnDataType_t cudnn_dtype;
    switch (input.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        default:
            throw std::runtime_error("cuDNN Conv2d backward: unsupported dtype");
    }

    // Create descriptors
    TensorDescriptor input_desc, grad_output_desc;
    FilterDescriptor filter_desc;
    ConvolutionDescriptor conv_desc;

    input_desc.set(cudnn_dtype, batch, in_channels, height, width);
    grad_output_desc.set(cudnn_dtype, batch, out_channels, out_h, out_w);
    filter_desc.set(cudnn_dtype, out_channels, in_channels / groups, kernel_h, kernel_w);
    conv_desc.set(padding, padding, stride, stride, dilation, dilation, cudnn_dtype);

    if (groups > 1) {
        conv_desc.set_group_count(groups);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute gradient w.r.t. input
    if (compute_grad_input) {
        // Find best algorithm
        int requested_algo_count = 10;
        int returned_algo_count = 0;
        std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(requested_algo_count);

        CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
            handle.get(),
            filter_desc.get(),
            grad_output_desc.get(),
            conv_desc.get(),
            input_desc.get(),
            requested_algo_count,
            &returned_algo_count,
            perf_results.data()
        ));

        cudnnConvolutionBwdDataAlgo_t algo = perf_results[0].algo;
        size_t workspace_size = perf_results[0].memory;

        void* workspace = nullptr;
        if (workspace_size > 0) {
            cudaMalloc(&workspace, workspace_size);
        }

        if (input.dtype() == DType::Float32) {
            CUDNN_CHECK(cudnnConvolutionBackwardData(
                handle.get(),
                &alpha,
                filter_desc.get(),
                weight.data<float>(),
                grad_output_desc.get(),
                grad_output.data<float>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta,
                input_desc.get(),
                grad_input.data<float>()
            ));
        } else if (input.dtype() == DType::Float64) {
            const double alpha_d = 1.0;
            const double beta_d = 0.0;
            CUDNN_CHECK(cudnnConvolutionBackwardData(
                handle.get(),
                &alpha_d,
                filter_desc.get(),
                weight.data<double>(),
                grad_output_desc.get(),
                grad_output.data<double>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta_d,
                input_desc.get(),
                grad_input.data<double>()
            ));
        } else if (input.dtype() == DType::Float16) {
            CUDNN_CHECK(cudnnConvolutionBackwardData(
                handle.get(),
                &alpha,
                filter_desc.get(),
                weight.data<Float16>(),
                grad_output_desc.get(),
                grad_output.data<Float16>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta,
                input_desc.get(),
                grad_input.data<Float16>()
            ));
        }

        if (workspace) {
            cudaFree(workspace);
        }
    }

    // Compute gradient w.r.t. weight
    if (compute_grad_weight) {
        // Find best algorithm
        int requested_algo_count = 10;
        int returned_algo_count = 0;
        std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(requested_algo_count);

        CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(
            handle.get(),
            input_desc.get(),
            grad_output_desc.get(),
            conv_desc.get(),
            filter_desc.get(),
            requested_algo_count,
            &returned_algo_count,
            perf_results.data()
        ));

        cudnnConvolutionBwdFilterAlgo_t algo = perf_results[0].algo;
        size_t workspace_size = perf_results[0].memory;

        void* workspace = nullptr;
        if (workspace_size > 0) {
            cudaMalloc(&workspace, workspace_size);
        }

        if (input.dtype() == DType::Float32) {
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle.get(),
                &alpha,
                input_desc.get(),
                input.data<float>(),
                grad_output_desc.get(),
                grad_output.data<float>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta,
                filter_desc.get(),
                grad_weight.data<float>()
            ));
        } else if (input.dtype() == DType::Float64) {
            const double alpha_d = 1.0;
            const double beta_d = 0.0;
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle.get(),
                &alpha_d,
                input_desc.get(),
                input.data<double>(),
                grad_output_desc.get(),
                grad_output.data<double>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta_d,
                filter_desc.get(),
                grad_weight.data<double>()
            ));
        } else if (input.dtype() == DType::Float16) {
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle.get(),
                &alpha,
                input_desc.get(),
                input.data<Float16>(),
                grad_output_desc.get(),
                grad_output.data<Float16>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta,
                filter_desc.get(),
                grad_weight.data<Float16>()
            ));
        }

        if (workspace) {
            cudaFree(workspace);
        }
    }

    // Compute gradient w.r.t. bias
    if (compute_grad_bias) {
        TensorDescriptor bias_desc;
        bias_desc.set(cudnn_dtype, 1, out_channels, 1, 1);

        if (input.dtype() == DType::Float32) {
            CUDNN_CHECK(cudnnConvolutionBackwardBias(
                handle.get(),
                &alpha,
                grad_output_desc.get(),
                grad_output.data<float>(),
                &beta,
                bias_desc.get(),
                grad_bias.data<float>()
            ));
        } else if (input.dtype() == DType::Float64) {
            const double alpha_d = 1.0;
            const double beta_d = 0.0;
            CUDNN_CHECK(cudnnConvolutionBackwardBias(
                handle.get(),
                &alpha_d,
                grad_output_desc.get(),
                grad_output.data<double>(),
                &beta_d,
                bias_desc.get(),
                grad_bias.data<double>()
            ));
        } else if (input.dtype() == DType::Float16) {
            CUDNN_CHECK(cudnnConvolutionBackwardBias(
                handle.get(),
                &alpha,
                grad_output_desc.get(),
                grad_output.data<Float16>(),
                &beta,
                bias_desc.get(),
                grad_bias.data<Float16>()
            ));
        }
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// ============================================================================
// cuDNN LSTM Forward Implementation
// NOTE: Deprecated cuDNN RNN v7 APIs - disabled for cuDNN >= 8.9
// Use custom CUDA LSTM kernels instead (see cuda/kernels/lstm.cu)
// ============================================================================

#if 0  // Disabled due to deprecated cuDNN RNN API
auto cudnn_lstm_forward(
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& weights,
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t input_size = input_shape[2];

    int num_directions = bidirectional ? 2 : 1;

    // Create output tensors
    Tensor output({seq_len, batch, hidden_size * num_directions}, input.dtype(), input.device());
    Tensor hy({num_layers * num_directions, batch, hidden_size}, input.dtype(), input.device());
    Tensor cy({num_layers * num_directions, batch, hidden_size}, input.dtype(), input.device());

    // Create cuDNN handle
    CuDNNHandle handle;
    handle.set_stream(stream);

    // Convert dtype
    cudnnDataType_t cudnn_dtype;
    switch (input.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        default:
            throw std::runtime_error("cuDNN LSTM: unsupported dtype");
    }

    // Create dropout descriptor
    DropoutDescriptor dropout_desc;
    size_t dropout_state_size;
    CUDNN_CHECK(cudnnDropoutGetStatesSize(handle.get(), &dropout_state_size));

    void* dropout_states = nullptr;
    if (dropout > 0.0f && dropout_state_size > 0) {
        cudaMalloc(&dropout_states, dropout_state_size);
        dropout_desc.set(handle.get(), dropout, dropout_states, dropout_state_size, 0);
    } else {
        dropout_desc.set(handle.get(), 0.0f, nullptr, 0, 0);
    }

    // Create RNN descriptor
    RNNDescriptor rnn_desc;
    cudnnDirectionMode_t direction = bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    rnn_desc.set_lstm(handle.get(), hidden_size, num_layers, dropout_desc.get(),
                      CUDNN_LINEAR_INPUT, direction, cudnn_dtype);

    // Create tensor descriptors for sequence
    std::vector<TensorDescriptor> x_descs(seq_len);
    std::vector<TensorDescriptor> y_descs(seq_len);
    std::vector<cudnnTensorDescriptor_t> x_desc_array(seq_len);
    std::vector<cudnnTensorDescriptor_t> y_desc_array(seq_len);

    for (int64_t i = 0; i < seq_len; ++i) {
        x_descs[i].set(cudnn_dtype, batch, input_size, 1, 1);
        y_descs[i].set(cudnn_dtype, batch, hidden_size * num_directions, 1, 1);
        x_desc_array[i] = x_descs[i].get();
        y_desc_array[i] = y_descs[i].get();
    }

    // Create h and c descriptors
    TensorDescriptor hx_desc, cx_desc, hy_desc, cy_desc;
    hx_desc.set(cudnn_dtype, num_layers * num_directions, batch, hidden_size, 1);
    cx_desc.set(cudnn_dtype, num_layers * num_directions, batch, hidden_size, 1);
    hy_desc.set(cudnn_dtype, num_layers * num_directions, batch, hidden_size, 1);
    cy_desc.set(cudnn_dtype, num_layers * num_directions, batch, hidden_size, 1);

    // Get workspace and reserve space sizes
    size_t workspace_size;
    size_t reserve_size;

    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle.get(),
        rnn_desc.get(),
        seq_len,
        x_desc_array.data(),
        &workspace_size
    ));

    CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
        handle.get(),
        rnn_desc.get(),
        seq_len,
        x_desc_array.data(),
        &reserve_size
    ));

    // Allocate workspace and reserve space
    void* workspace = nullptr;
    void* reserve_space = nullptr;

    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    if (reserve_size > 0) {
        cudaMalloc(&reserve_space, reserve_size);
    }

    // Forward pass
    if (input.dtype() == DType::Float32) {
        CUDNN_CHECK(cudnnRNNForwardTraining(
            handle.get(),
            rnn_desc.get(),
            seq_len,
            x_desc_array.data(),
            input.data<float>(),
            hx_desc.get(),
            hx.data<float>(),
            cx_desc.get(),
            cx.data<float>(),
            rnn_desc.get(), // Weight descriptor (embedded in RNN descriptor for cuDNN >= 8.0)
            weights.data<float>(),
            y_desc_array.data(),
            output.data<float>(),
            hy_desc.get(),
            hy.data<float>(),
            cy_desc.get(),
            cy.data<float>(),
            workspace,
            workspace_size,
            reserve_space,
            reserve_size
        ));
    } else if (input.dtype() == DType::Float64) {
        CUDNN_CHECK(cudnnRNNForwardTraining(
            handle.get(),
            rnn_desc.get(),
            seq_len,
            x_desc_array.data(),
            input.data<double>(),
            hx_desc.get(),
            hx.data<double>(),
            cx_desc.get(),
            cx.data<double>(),
            rnn_desc.get(),
            weights.data<double>(),
            y_desc_array.data(),
            output.data<double>(),
            hy_desc.get(),
            hy.data<double>(),
            cy_desc.get(),
            cy.data<double>(),
            workspace,
            workspace_size,
            reserve_space,
            reserve_size
        ));
    } else if (input.dtype() == DType::Float16) {
        CUDNN_CHECK(cudnnRNNForwardTraining(
            handle.get(),
            rnn_desc.get(),
            seq_len,
            x_desc_array.data(),
            input.data<Float16>(),
            hx_desc.get(),
            hx.data<Float16>(),
            cx_desc.get(),
            cx.data<Float16>(),
            rnn_desc.get(),
            weights.data<Float16>(),
            y_desc_array.data(),
            output.data<Float16>(),
            hy_desc.get(),
            hy.data<Float16>(),
            cy_desc.get(),
            cy.data<Float16>(),
            workspace,
            workspace_size,
            reserve_space,
            reserve_size
        ));
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    if (reserve_space) cudaFree(reserve_space);
    if (dropout_states) cudaFree(dropout_states);

    return std::make_tuple(output, hy, cy);
}

// ============================================================================
// cuDNN LSTM Backward Implementation (Simplified - Full version omitted for brevity)
// ============================================================================

auto cudnn_lstm_backward(
    const Tensor& grad_output,
    const Tensor& grad_hy,
    const Tensor& grad_cy,
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& output,
    const Tensor& hy,
    const Tensor& cy,
    const Tensor& weights,
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor, Tensor> {
    // Similar structure to forward but using cudnnRNNBackwardData and cudnnRNNBackwardWeights
    // Implementation follows the same pattern with appropriate descriptor setup

    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t input_size = input_shape[2];

    int num_directions = bidirectional ? 2 : 1;

    // Create gradient tensors
    Tensor grad_input({seq_len, batch, input_size}, input.dtype(), input.device());
    Tensor grad_hx({num_layers * num_directions, batch, hidden_size}, input.dtype(), input.device());
    Tensor grad_cx({num_layers * num_directions, batch, hidden_size}, input.dtype(), input.device());
    Tensor grad_weights = Tensor::zeros_like(weights);

    // Note: Full backward implementation would follow similar pattern to forward
    // with cudnnRNNBackwardData and cudnnRNNBackwardWeights calls
    // For brevity, returning initialized tensors

    return std::make_tuple(grad_input, grad_hx, grad_cx, grad_weights);
}
#endif // Disabled cuDNN LSTM - using custom CUDA kernels

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_HAS_CUDNN
