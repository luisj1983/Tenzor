/**
 * @file rnn_kernels.cpp
 * @brief CPU RNN kernel implementations (LSTM, GRU)
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>

namespace tenzor {
namespace cpu {

// Sigmoid activation
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

auto lstm_cell_forward_kernel(const Tensor& input, const Tensor& hx, const Tensor& cx,
                               const Tensor& weight_ih, const Tensor& weight_hh,
                               const Tensor& bias_ih, const Tensor& bias_hh)
    -> std::vector<Tensor> {
    // input: [batch, input_size]
    // hx: [batch, hidden_size]
    // cx: [batch, hidden_size]
    // weight_ih: [4 * hidden_size, input_size]
    // weight_hh: [4 * hidden_size, hidden_size]
    // bias_ih, bias_hh: [4 * hidden_size]

    auto in_shape = input.shape();
    auto hx_shape = hx.shape();

    int64_t batch_size = in_shape[0];
    int64_t input_size = in_shape[1];
    int64_t hidden_size = hx_shape[1];

    // Output: hy, cy
    auto hy = Tensor::empty_uninitialized({batch_size, hidden_size}, input.dtype(), input.device());
    auto cy = Tensor::empty_uninitialized({batch_size, hidden_size}, input.dtype(), input.device());

    const float* in_data = input.data<float>();
    const float* hx_data = hx.data<float>();
    const float* cx_data = cx.data<float>();
    const float* w_ih_data = weight_ih.data<float>();
    const float* w_hh_data = weight_hh.data<float>();
    const float* b_ih_data = bias_ih.data<float>();
    const float* b_hh_data = bias_hh.data<float>();
    float* hy_data = hy.data<float>();
    float* cy_data = cy.data<float>();

    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        // Compute gates: i, f, g, o (4 * hidden_size total)
        std::vector<float> gates(4 * hidden_size);

        // gates = input @ weight_ih^T + hx @ weight_hh^T + bias_ih + bias_hh
        for (int64_t g = 0; g < 4 * hidden_size; ++g) {
            float sum = b_ih_data[g] + b_hh_data[g];

            for (int64_t i = 0; i < input_size; ++i) {
                sum += in_data[b * input_size + i] * w_ih_data[g * input_size + i];
            }
            for (int64_t h = 0; h < hidden_size; ++h) {
                sum += hx_data[b * hidden_size + h] * w_hh_data[g * hidden_size + h];
            }
            gates[g] = sum;
        }

        // Apply activations and compute outputs
        for (int64_t h = 0; h < hidden_size; ++h) {
            float i_gate = sigmoid(gates[h]);
            float f_gate = sigmoid(gates[hidden_size + h]);
            float g_gate = std::tanh(gates[2 * hidden_size + h]);
            float o_gate = sigmoid(gates[3 * hidden_size + h]);

            float c_new = f_gate * cx_data[b * hidden_size + h] + i_gate * g_gate;
            float h_new = o_gate * std::tanh(c_new);

            cy_data[b * hidden_size + h] = c_new;
            hy_data[b * hidden_size + h] = h_new;
        }
    }

    return {hy, cy};
}

auto lstm_cell_backward_kernel(const Tensor& grad_hy, const Tensor& grad_cy,
                                const Tensor& input, const Tensor& hx, const Tensor& cx,
                                const Tensor& hy, const Tensor& cy,
                                const Tensor& weight_ih, const Tensor& weight_hh)
    -> std::vector<Tensor> {
    auto shape = grad_hy.shape();
    int64_t batch_size = shape[0];
    int64_t hidden_size = shape[1];
    int64_t input_size = input.shape()[1];

    // Gradients
    auto grad_input = zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                            input.dtype(), input.device());
    auto grad_hx = zeros(std::vector<int64_t>(hx.shape().begin(), hx.shape().end()),
                         hx.dtype(), hx.device());
    auto grad_cx = zeros(std::vector<int64_t>(cx.shape().begin(), cx.shape().end()),
                         cx.dtype(), cx.device());
    auto grad_weight_ih = zeros(std::vector<int64_t>(weight_ih.shape().begin(), weight_ih.shape().end()),
                                weight_ih.dtype(), weight_ih.device());
    auto grad_weight_hh = zeros(std::vector<int64_t>(weight_hh.shape().begin(), weight_hh.shape().end()),
                                weight_hh.dtype(), weight_hh.device());
    auto grad_bias_ih = zeros({4 * hidden_size}, input.dtype(), input.device());
    auto grad_bias_hh = zeros({4 * hidden_size}, input.dtype(), input.device());

    // Simplified backward pass (would need gate caching for full implementation)
    // For now, return zero gradients as placeholder
    return {grad_input, grad_hx, grad_cx, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh};
}

auto gru_cell_forward_kernel(const Tensor& input, const Tensor& hx,
                              const Tensor& weight_ih, const Tensor& weight_hh,
                              const Tensor& bias_ih, const Tensor& bias_hh) -> Tensor {
    // input: [batch, input_size]
    // hx: [batch, hidden_size]
    // weight_ih: [3 * hidden_size, input_size] (r, z, n gates)
    // weight_hh: [3 * hidden_size, hidden_size]

    auto in_shape = input.shape();
    auto hx_shape = hx.shape();

    int64_t batch_size = in_shape[0];
    int64_t input_size = in_shape[1];
    int64_t hidden_size = hx_shape[1];

    auto hy = Tensor::empty_uninitialized({batch_size, hidden_size}, input.dtype(), input.device());

    const float* in_data = input.data<float>();
    const float* hx_data = hx.data<float>();
    const float* w_ih_data = weight_ih.data<float>();
    const float* w_hh_data = weight_hh.data<float>();
    const float* b_ih_data = bias_ih.data<float>();
    const float* b_hh_data = bias_hh.data<float>();
    float* hy_data = hy.data<float>();

    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        // Compute r, z gates (2 * hidden_size)
        std::vector<float> gates_rz(2 * hidden_size);
        for (int64_t g = 0; g < 2 * hidden_size; ++g) {
            float sum = b_ih_data[g] + b_hh_data[g];
            for (int64_t i = 0; i < input_size; ++i) {
                sum += in_data[b * input_size + i] * w_ih_data[g * input_size + i];
            }
            for (int64_t h = 0; h < hidden_size; ++h) {
                sum += hx_data[b * hidden_size + h] * w_hh_data[g * hidden_size + h];
            }
            gates_rz[g] = sigmoid(sum);
        }

        // Compute n gate with reset gate applied
        for (int64_t h = 0; h < hidden_size; ++h) {
            float r = gates_rz[h];
            float z = gates_rz[hidden_size + h];

            float n_ih = b_ih_data[2 * hidden_size + h];
            float n_hh = b_hh_data[2 * hidden_size + h];

            for (int64_t i = 0; i < input_size; ++i) {
                n_ih += in_data[b * input_size + i] * w_ih_data[(2 * hidden_size + h) * input_size + i];
            }
            for (int64_t hh = 0; hh < hidden_size; ++hh) {
                n_hh += (r * hx_data[b * hidden_size + hh]) *
                        w_hh_data[(2 * hidden_size + h) * hidden_size + hh];
            }

            float n = std::tanh(n_ih + n_hh);
            hy_data[b * hidden_size + h] = (1.0f - z) * n + z * hx_data[b * hidden_size + h];
        }
    }

    return hy;
}

auto gru_cell_backward_kernel(const Tensor& grad_hy, const Tensor& input, const Tensor& hx,
                               const Tensor& weight_ih, const Tensor& weight_hh)
    -> std::vector<Tensor> {
    auto shape = grad_hy.shape();
    int64_t hidden_size = shape[1];
    int64_t input_size = input.shape()[1];

    // Gradients (placeholder implementation)
    auto grad_input = zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                            input.dtype(), input.device());
    auto grad_hx = zeros(std::vector<int64_t>(hx.shape().begin(), hx.shape().end()),
                         hx.dtype(), hx.device());
    auto grad_weight_ih = zeros(std::vector<int64_t>(weight_ih.shape().begin(), weight_ih.shape().end()),
                                weight_ih.dtype(), weight_ih.device());
    auto grad_weight_hh = zeros(std::vector<int64_t>(weight_hh.shape().begin(), weight_hh.shape().end()),
                                weight_hh.dtype(), weight_hh.device());
    auto grad_bias_ih = zeros({3 * hidden_size}, input.dtype(), input.device());
    auto grad_bias_hh = zeros({3 * hidden_size}, input.dtype(), input.device());

    return {grad_input, grad_hx, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh};
}

} // namespace cpu
} // namespace tenzor
