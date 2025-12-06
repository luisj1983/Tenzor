#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace tenzor {
namespace oneapi {

// SYCL Kernel name classes
struct LSTMCellForwardFloat32 {};
struct LSTMCellForwardFloat64 {};
struct LSTMCellBackwardFloat32 {};
struct LSTMCellBackwardFloat64 {};

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// ============================================================================
// LSTM Cell Forward Kernel
// ============================================================================
/**
 * @brief Fused LSTM cell forward kernel
 *
 * Computes all 4 gates (input, forget, cell, output) in a single kernel.
 * This reduces memory bandwidth and improves performance.
 *
 * LSTM equations:
 *   i_t = σ(gates[i])                    // Input gate
 *   f_t = σ(gates[f])                    // Forget gate
 *   g_t = tanh(gates[g])                 // Cell gate
 *   o_t = σ(gates[o])                    // Output gate
 *   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t     // New cell state
 *   h_t = o_t ⊙ tanh(c_t)               // New hidden state
 *
 * @param gates Pre-computed gate values: (batch, 4 * hidden_size)
 *              Layout: [input_gate | forget_gate | cell_gate | output_gate]
 * @param c_prev Previous cell state: (batch, hidden_size)
 * @param batch_size Number of sequences
 * @param hidden_size Hidden dimension
 * @return Pair of (h_out, c_out) tensors
 */
auto lstm_cell_forward_kernel(
    const Tensor& gates,
    const Tensor& c_prev,
    int64_t batch_size,
    int64_t hidden_size,
    sycl::queue& queue
) -> std::pair<Tensor, Tensor> {
    std::vector<int64_t> output_shape = {batch_size, hidden_size};
    Tensor h_out(output_shape, gates.dtype(), gates.device());
    Tensor c_out(output_shape, gates.dtype(), gates.device());

    int64_t total_elements = batch_size * hidden_size;

    if (gates.dtype() == DType::Float32) {
        const float* gates_ptr = get_data_ptr<const float>(gates);
        const float* c_prev_ptr = get_data_ptr<const float>(c_prev);
        float* h_out_ptr = get_data_ptr<float>(h_out);
        float* c_out_ptr = get_data_ptr<float>(c_out);

        queue.parallel_for<LSTMCellForwardFloat32>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t batch_idx = idx / hidden_size;
                int64_t hidden_idx = idx % hidden_size;

                // Offsets for each gate
                int64_t gate_offset = batch_idx * (4 * hidden_size) + hidden_idx;
                int64_t i_offset = gate_offset;                      // Input gate
                int64_t f_offset = gate_offset + hidden_size;        // Forget gate
                int64_t g_offset = gate_offset + 2 * hidden_size;    // Cell gate
                int64_t o_offset = gate_offset + 3 * hidden_size;    // Output gate

                // Load pre-computed gate values
                float i_gate = gates_ptr[i_offset];
                float f_gate = gates_ptr[f_offset];
                float g_gate = gates_ptr[g_offset];
                float o_gate = gates_ptr[o_offset];

                // Apply activations
                // Sigmoid for i, f, o gates
                float i_t = 1.0f / (1.0f + sycl::exp(-i_gate));
                float f_t = 1.0f / (1.0f + sycl::exp(-f_gate));
                float o_t = 1.0f / (1.0f + sycl::exp(-o_gate));

                // Tanh for cell gate
                float g_t = sycl::tanh(g_gate);

                // Load previous cell state
                float c_prev_val = c_prev_ptr[idx];

                // Update cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
                float c_t = f_t * c_prev_val + i_t * g_t;

                // Update hidden state: h_t = o_t ⊙ tanh(c_t)
                float h_t = o_t * sycl::tanh(c_t);

                // Store outputs
                c_out_ptr[idx] = c_t;
                h_out_ptr[idx] = h_t;
            }
        ).wait();
    }
    else if (gates.dtype() == DType::Float64) {
        const double* gates_ptr = get_data_ptr<const double>(gates);
        const double* c_prev_ptr = get_data_ptr<const double>(c_prev);
        double* h_out_ptr = get_data_ptr<double>(h_out);
        double* c_out_ptr = get_data_ptr<double>(c_out);

        queue.parallel_for<LSTMCellForwardFloat64>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t batch_idx = idx / hidden_size;
                int64_t hidden_idx = idx % hidden_size;

                int64_t gate_offset = batch_idx * (4 * hidden_size) + hidden_idx;
                int64_t i_offset = gate_offset;
                int64_t f_offset = gate_offset + hidden_size;
                int64_t g_offset = gate_offset + 2 * hidden_size;
                int64_t o_offset = gate_offset + 3 * hidden_size;

                double i_gate = gates_ptr[i_offset];
                double f_gate = gates_ptr[f_offset];
                double g_gate = gates_ptr[g_offset];
                double o_gate = gates_ptr[o_offset];

                double i_t = 1.0 / (1.0 + sycl::exp(-i_gate));
                double f_t = 1.0 / (1.0 + sycl::exp(-f_gate));
                double o_t = 1.0 / (1.0 + sycl::exp(-o_gate));
                double g_t = sycl::tanh(g_gate);

                double c_prev_val = c_prev_ptr[idx];
                double c_t = f_t * c_prev_val + i_t * g_t;
                double h_t = o_t * sycl::tanh(c_t);

                c_out_ptr[idx] = c_t;
                h_out_ptr[idx] = h_t;
            }
        ).wait();
    }
    else {
        throw std::runtime_error("lstm_cell_forward: only Float32 and Float64 supported");
    }

    return {h_out, c_out};
}

// ============================================================================
// LSTM Cell Backward Kernel
// ============================================================================
/**
 * @brief Fused LSTM cell backward kernel
 *
 * Computes gradients for all gates in a single kernel.
 *
 * @param grad_h Gradient w.r.t. hidden output: (batch, hidden_size)
 * @param grad_c Gradient w.r.t. cell output: (batch, hidden_size)
 * @param gates Gate values from forward: (batch, 4 * hidden_size)
 * @param c_prev Previous cell state: (batch, hidden_size)
 * @param c_out Current cell state: (batch, hidden_size)
 * @return Pair of (grad_gates, grad_c_prev)
 */
auto lstm_cell_backward_kernel(
    const Tensor& grad_h,
    const Tensor& grad_c,
    const Tensor& gates,
    const Tensor& c_prev,
    const Tensor& c_out,
    int64_t batch_size,
    int64_t hidden_size,
    sycl::queue& queue
) -> std::pair<Tensor, Tensor> {
    std::vector<int64_t> gate_shape = {batch_size, 4 * hidden_size};
    std::vector<int64_t> state_shape = {batch_size, hidden_size};

    Tensor grad_gates(gate_shape, gates.dtype(), gates.device());
    Tensor grad_c_prev(state_shape, gates.dtype(), gates.device());

    int64_t total_elements = batch_size * hidden_size;

    if (gates.dtype() == DType::Float32) {
        const float* grad_h_ptr = get_data_ptr<const float>(grad_h);
        const float* grad_c_ptr = get_data_ptr<const float>(grad_c);
        const float* gates_ptr = get_data_ptr<const float>(gates);
        const float* c_prev_ptr = get_data_ptr<const float>(c_prev);
        const float* c_out_ptr = get_data_ptr<const float>(c_out);
        float* grad_gates_ptr = get_data_ptr<float>(grad_gates);
        float* grad_c_prev_ptr = get_data_ptr<float>(grad_c_prev);

        queue.parallel_for<LSTMCellBackwardFloat32>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t batch_idx = idx / hidden_size;
                int64_t hidden_idx = idx % hidden_size;

                // Offsets
                int64_t gate_offset = batch_idx * (4 * hidden_size) + hidden_idx;
                int64_t i_offset = gate_offset;
                int64_t f_offset = gate_offset + hidden_size;
                int64_t g_offset = gate_offset + 2 * hidden_size;
                int64_t o_offset = gate_offset + 3 * hidden_size;

                // Load gate values and recompute activations
                float i_gate = gates_ptr[i_offset];
                float f_gate = gates_ptr[f_offset];
                float g_gate = gates_ptr[g_offset];
                float o_gate = gates_ptr[o_offset];

                float i_t = 1.0f / (1.0f + sycl::exp(-i_gate));
                float f_t = 1.0f / (1.0f + sycl::exp(-f_gate));
                float g_t = sycl::tanh(g_gate);
                float o_t = 1.0f / (1.0f + sycl::exp(-o_gate));

                // Load states
                float c_prev_val = c_prev_ptr[idx];
                float c_t = c_out_ptr[idx];
                float tanh_c_t = sycl::tanh(c_t);

                // Load incoming gradients
                float dh = grad_h_ptr[idx];
                float dc = grad_c_ptr[idx];

                // Gradient through h = o * tanh(c)
                dc += dh * o_t * (1.0f - tanh_c_t * tanh_c_t);  // tanh derivative
                float do_t = dh * tanh_c_t;

                // Gradient through c = f * c_prev + i * g
                float df_t = dc * c_prev_val;
                float di_t = dc * g_t;
                float dg_t = dc * i_t;
                float dc_prev = dc * f_t;

                // Gradient through gate activations
                // Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
                float di_gate = di_t * i_t * (1.0f - i_t);
                float df_gate = df_t * f_t * (1.0f - f_t);
                float do_gate = do_t * o_t * (1.0f - o_t);

                // Tanh derivative: tanh'(x) = 1 - tanh^2(x)
                float dg_gate = dg_t * (1.0f - g_t * g_t);

                // Store gradients
                grad_gates_ptr[i_offset] = di_gate;
                grad_gates_ptr[f_offset] = df_gate;
                grad_gates_ptr[g_offset] = dg_gate;
                grad_gates_ptr[o_offset] = do_gate;
                grad_c_prev_ptr[idx] = dc_prev;
            }
        ).wait();
    }
    else if (gates.dtype() == DType::Float64) {
        const double* grad_h_ptr = get_data_ptr<const double>(grad_h);
        const double* grad_c_ptr = get_data_ptr<const double>(grad_c);
        const double* gates_ptr = get_data_ptr<const double>(gates);
        const double* c_prev_ptr = get_data_ptr<const double>(c_prev);
        const double* c_out_ptr = get_data_ptr<const double>(c_out);
        double* grad_gates_ptr = get_data_ptr<double>(grad_gates);
        double* grad_c_prev_ptr = get_data_ptr<double>(grad_c_prev);

        queue.parallel_for<LSTMCellBackwardFloat64>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t batch_idx = idx / hidden_size;
                int64_t hidden_idx = idx % hidden_size;

                int64_t gate_offset = batch_idx * (4 * hidden_size) + hidden_idx;
                int64_t i_offset = gate_offset;
                int64_t f_offset = gate_offset + hidden_size;
                int64_t g_offset = gate_offset + 2 * hidden_size;
                int64_t o_offset = gate_offset + 3 * hidden_size;

                double i_gate = gates_ptr[i_offset];
                double f_gate = gates_ptr[f_offset];
                double g_gate = gates_ptr[g_offset];
                double o_gate = gates_ptr[o_offset];

                double i_t = 1.0 / (1.0 + sycl::exp(-i_gate));
                double f_t = 1.0 / (1.0 + sycl::exp(-f_gate));
                double g_t = sycl::tanh(g_gate);
                double o_t = 1.0 / (1.0 + sycl::exp(-o_gate));

                double c_prev_val = c_prev_ptr[idx];
                double c_t = c_out_ptr[idx];
                double tanh_c_t = sycl::tanh(c_t);

                double dh = grad_h_ptr[idx];
                double dc = grad_c_ptr[idx];

                dc += dh * o_t * (1.0 - tanh_c_t * tanh_c_t);
                double do_t = dh * tanh_c_t;

                double df_t = dc * c_prev_val;
                double di_t = dc * g_t;
                double dg_t = dc * i_t;
                double dc_prev = dc * f_t;

                double di_gate = di_t * i_t * (1.0 - i_t);
                double df_gate = df_t * f_t * (1.0 - f_t);
                double do_gate = do_t * o_t * (1.0 - o_t);
                double dg_gate = dg_t * (1.0 - g_t * g_t);

                grad_gates_ptr[i_offset] = di_gate;
                grad_gates_ptr[f_offset] = df_gate;
                grad_gates_ptr[g_offset] = dg_gate;
                grad_gates_ptr[o_offset] = do_gate;
                grad_c_prev_ptr[idx] = dc_prev;
            }
        ).wait();
    }
    else {
        throw std::runtime_error("lstm_cell_backward: only Float32 and Float64 supported");
    }

    return {grad_gates, grad_c_prev};
}

} // namespace oneapi
} // namespace tenzor
