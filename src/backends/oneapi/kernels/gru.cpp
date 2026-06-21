#include "tenzor/core/tensor.hpp"
#include "oneapi_kernel_utils.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>

namespace tenzor {
namespace oneapi {

// SYCL Kernel name classes
struct GRUCellForwardFloat32 {};
struct GRUCellForwardFloat64 {};
struct GRUCellBackwardFloat32 {};
struct GRUCellBackwardFloat64 {};
struct GRUCellForwardBFloat16 {};
struct GRUCellBackwardBFloat16 {};



// ============================================================================
// GRU Cell Forward Kernel
// ============================================================================
/**
 * @brief Fused GRU cell forward kernel
 *
 * Computes all 3 gates (reset, update, new) in a single kernel.
 *
 * GRU equations:
 *   r_t = σ(reset_gates)                              // Reset gate
 *   z_t = σ(update_gates)                             // Update gate
 *   n_t = tanh(new_gates_input + r_t ⊙ new_gates_hidden)  // New gate
 *   h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}            // New hidden state
 *
 * @param reset_gates Pre-computed reset gate values: (batch, hidden_size)
 * @param update_gates Pre-computed update gate values: (batch, hidden_size)
 * @param new_gates_input New gate input part: (batch, hidden_size)
 * @param new_gates_hidden New gate hidden part: (batch, hidden_size)
 * @param h_prev Previous hidden state: (batch, hidden_size)
 * @return Output hidden state tensor
 */
auto gru_cell_forward_kernel(
    const Tensor& reset_gates,
    const Tensor& update_gates,
    const Tensor& new_gates_input,
    const Tensor& new_gates_hidden,
    const Tensor& h_prev,
    int64_t batch_size,
    int64_t hidden_size,
    sycl::queue& queue
) -> Tensor {
    // Float16: widen all gates to Float32 on device, compute, narrow back.
    // (BFloat16 has a native SYCL kernel below; Float16 previously fell through
    // to the throw, so a fp16 GRU cell crashed on OneAPI while ROCm/CPU handle it.)
    if (reset_gates.dtype() == DType::Float16) {
        Tensor h = gru_cell_forward_kernel(
            reset_gates.to(DType::Float32), update_gates.to(DType::Float32),
            new_gates_input.to(DType::Float32), new_gates_hidden.to(DType::Float32),
            h_prev.to(DType::Float32), batch_size, hidden_size, queue);
        return h.to(DType::Float16);
    }

    std::vector<int64_t> output_shape = {batch_size, hidden_size};
    Tensor h_out(output_shape, reset_gates.dtype(), reset_gates.device());

    int64_t total_elements = batch_size * hidden_size;

    if (reset_gates.dtype() == DType::Float32) {
        const float* reset_ptr = get_data_ptr<const float>(reset_gates);
        const float* update_ptr = get_data_ptr<const float>(update_gates);
        const float* new_input_ptr = get_data_ptr<const float>(new_gates_input);
        const float* new_hidden_ptr = get_data_ptr<const float>(new_gates_hidden);
        const float* h_prev_ptr = get_data_ptr<const float>(h_prev);
        float* h_out_ptr = get_data_ptr<float>(h_out);

        queue.parallel_for<GRUCellForwardFloat32>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                // Load pre-computed gate values
                float r_gate = reset_ptr[idx];
                float z_gate = update_ptr[idx];
                float n_input = new_input_ptr[idx];
                float n_hidden = new_hidden_ptr[idx];
                float h_prev_val = h_prev_ptr[idx];

                // Apply sigmoid activation to reset and update gates.
                // Clamp pre-activations to [-20, 20] for parity with the
                // OneAPI LSTM cell, preventing exp() overflow on extreme inputs.
                float r_clamped = sycl::fmax(-20.0f, sycl::fmin(20.0f, r_gate));
                float z_clamped = sycl::fmax(-20.0f, sycl::fmin(20.0f, z_gate));
                float r_t = 1.0f / (1.0f + sycl::exp(-r_clamped));
                float z_t = 1.0f / (1.0f + sycl::exp(-z_clamped));

                // Apply reset gate to hidden part and compute new gate
                float n_combined = n_input + r_t * n_hidden;
                float n_t = sycl::tanh(n_combined);

                // Compute new hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
                float h_t = (1.0f - z_t) * n_t + z_t * h_prev_val;

                // Store output
                h_out_ptr[idx] = h_t;
            }
        );
    }
    else if (reset_gates.dtype() == DType::Float64) {
        const double* reset_ptr = get_data_ptr<const double>(reset_gates);
        const double* update_ptr = get_data_ptr<const double>(update_gates);
        const double* new_input_ptr = get_data_ptr<const double>(new_gates_input);
        const double* new_hidden_ptr = get_data_ptr<const double>(new_gates_hidden);
        const double* h_prev_ptr = get_data_ptr<const double>(h_prev);
        double* h_out_ptr = get_data_ptr<double>(h_out);

        queue.parallel_for<GRUCellForwardFloat64>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                double r_gate = reset_ptr[idx];
                double z_gate = update_ptr[idx];
                double n_input = new_input_ptr[idx];
                double n_hidden = new_hidden_ptr[idx];
                double h_prev_val = h_prev_ptr[idx];

                double r_clamped = sycl::fmax(-20.0, sycl::fmin(20.0, r_gate));
                double z_clamped = sycl::fmax(-20.0, sycl::fmin(20.0, z_gate));
                double r_t = 1.0 / (1.0 + sycl::exp(-r_clamped));
                double z_t = 1.0 / (1.0 + sycl::exp(-z_clamped));

                double n_combined = n_input + r_t * n_hidden;
                double n_t = sycl::tanh(n_combined);

                double h_t = (1.0 - z_t) * n_t + z_t * h_prev_val;

                h_out_ptr[idx] = h_t;
            }
        );
    }
    else if (reset_gates.dtype() == DType::BFloat16) {
        const uint16_t* reset_ptr = get_data_ptr<const uint16_t>(reset_gates);
        const uint16_t* update_ptr = get_data_ptr<const uint16_t>(update_gates);
        const uint16_t* new_input_ptr = get_data_ptr<const uint16_t>(new_gates_input);
        const uint16_t* new_hidden_ptr = get_data_ptr<const uint16_t>(new_gates_hidden);
        const uint16_t* h_prev_ptr = get_data_ptr<const uint16_t>(h_prev);
        uint16_t* h_out_ptr = get_data_ptr<uint16_t>(h_out);

        queue.parallel_for<GRUCellForwardBFloat16>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                float r_gate = bf16_to_f32(reset_ptr[idx]);
                float z_gate = bf16_to_f32(update_ptr[idx]);
                float n_input = bf16_to_f32(new_input_ptr[idx]);
                float n_hidden = bf16_to_f32(new_hidden_ptr[idx]);
                float h_prev_val = bf16_to_f32(h_prev_ptr[idx]);

                float r_clamped = sycl::fmax(-20.0f, sycl::fmin(20.0f, r_gate));
                float z_clamped = sycl::fmax(-20.0f, sycl::fmin(20.0f, z_gate));
                float r_t = 1.0f / (1.0f + sycl::exp(-r_clamped));
                float z_t = 1.0f / (1.0f + sycl::exp(-z_clamped));

                float n_combined = n_input + r_t * n_hidden;
                float n_t = sycl::tanh(n_combined);

                float h_t = (1.0f - z_t) * n_t + z_t * h_prev_val;

                h_out_ptr[idx] = f32_to_bf16(h_t);
            }
        );
    }
    else {
        throw std::runtime_error("gru_cell_forward: only Float32, Float64, and BFloat16 supported");
    }

    // Drain before the host reads the USM-shared h_out.
    queue.wait_and_throw();
    return h_out;
}

// Output struct for GRU backward
struct GRUBackwardOutputs {
    Tensor grad_reset;
    Tensor grad_update;
    Tensor grad_new_input;
    Tensor grad_new_hidden;
    Tensor grad_h_prev;
};

// ============================================================================
// GRU Cell Backward Kernel
// ============================================================================
/**
 * @brief Fused GRU cell backward kernel
 *
 * Computes gradients for all 3 gates in a single kernel.
 *
 * @param grad_h Gradient w.r.t. hidden output: (batch, hidden_size)
 * @param reset_gates Reset gate values from forward: (batch, hidden_size)
 * @param update_gates Update gate values from forward: (batch, hidden_size)
 * @param new_gates_input New gate input part from forward: (batch, hidden_size)
 * @param new_gates_hidden New gate hidden part from forward: (batch, hidden_size)
 * @param h_prev Previous hidden state: (batch, hidden_size)
 * @return GRUBackwardOutputs struct with all gradients
 */
auto gru_cell_backward_kernel(
    const Tensor& grad_h,
    const Tensor& reset_gates,
    const Tensor& update_gates,
    const Tensor& new_gates_input,
    const Tensor& new_gates_hidden,
    const Tensor& h_prev,
    int64_t batch_size,
    int64_t hidden_size,
    sycl::queue& queue
) -> GRUBackwardOutputs {
    // Float16: widen to Float32 on device, compute, narrow each gradient back.
    if (grad_h.dtype() == DType::Float16) {
        auto out = gru_cell_backward_kernel(
            grad_h.to(DType::Float32), reset_gates.to(DType::Float32),
            update_gates.to(DType::Float32), new_gates_input.to(DType::Float32),
            new_gates_hidden.to(DType::Float32), h_prev.to(DType::Float32),
            batch_size, hidden_size, queue);
        out.grad_reset = out.grad_reset.to(DType::Float16);
        out.grad_update = out.grad_update.to(DType::Float16);
        out.grad_new_input = out.grad_new_input.to(DType::Float16);
        out.grad_new_hidden = out.grad_new_hidden.to(DType::Float16);
        out.grad_h_prev = out.grad_h_prev.to(DType::Float16);
        return out;
    }

    std::vector<int64_t> state_shape = {batch_size, hidden_size};

    GRUBackwardOutputs outputs;
    outputs.grad_reset = Tensor(state_shape, grad_h.dtype(), grad_h.device());
    outputs.grad_update = Tensor(state_shape, grad_h.dtype(), grad_h.device());
    outputs.grad_new_input = Tensor(state_shape, grad_h.dtype(), grad_h.device());
    outputs.grad_new_hidden = Tensor(state_shape, grad_h.dtype(), grad_h.device());
    outputs.grad_h_prev = Tensor(state_shape, grad_h.dtype(), grad_h.device());

    int64_t total_elements = batch_size * hidden_size;

    if (grad_h.dtype() == DType::Float32) {
        const float* grad_h_ptr = get_data_ptr<const float>(grad_h);
        const float* reset_ptr = get_data_ptr<const float>(reset_gates);
        const float* update_ptr = get_data_ptr<const float>(update_gates);
        const float* new_input_ptr = get_data_ptr<const float>(new_gates_input);
        const float* new_hidden_ptr = get_data_ptr<const float>(new_gates_hidden);
        const float* h_prev_ptr = get_data_ptr<const float>(h_prev);

        float* grad_reset_ptr = get_data_ptr<float>(outputs.grad_reset);
        float* grad_update_ptr = get_data_ptr<float>(outputs.grad_update);
        float* grad_new_input_ptr = get_data_ptr<float>(outputs.grad_new_input);
        float* grad_new_hidden_ptr = get_data_ptr<float>(outputs.grad_new_hidden);
        float* grad_h_prev_ptr = get_data_ptr<float>(outputs.grad_h_prev);

        queue.parallel_for<GRUCellBackwardFloat32>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                // Recompute forward pass values
                float r_gate = reset_ptr[idx];
                float z_gate = update_ptr[idx];
                float n_input = new_input_ptr[idx];
                float n_hidden = new_hidden_ptr[idx];
                float h_prev_val = h_prev_ptr[idx];

                // Recompute activations with the SAME ±20 clamp the forward
                // applies to the sigmoid gates, so derivatives match the forward.
                float r_clamped = sycl::fmax(-20.0f, sycl::fmin(20.0f, r_gate));
                float z_clamped = sycl::fmax(-20.0f, sycl::fmin(20.0f, z_gate));
                float r_t = 1.0f / (1.0f + sycl::exp(-r_clamped));
                float z_t = 1.0f / (1.0f + sycl::exp(-z_clamped));

                float n_combined = n_input + r_t * n_hidden;
                float n_t = sycl::tanh(n_combined);

                // Load incoming gradient
                float dh = grad_h_ptr[idx];

                // Gradient through h_t = (1 - z_t) * n_t + z_t * h_prev
                float dn_t = dh * (1.0f - z_t);
                float dz_t = dh * (h_prev_val - n_t);
                float dh_prev = dh * z_t;

                // Gradient through n_t = tanh(n_combined)
                // tanh'(x) = 1 - tanh^2(x)
                float dn_combined = dn_t * (1.0f - n_t * n_t);

                // Gradient through n_combined = n_input + r_t * n_hidden
                float dn_input = dn_combined;
                float dr_t = dn_combined * n_hidden;
                float dn_hidden = dn_combined * r_t;

                // Gradient through sigmoid activations
                // σ'(x) = σ(x) * (1 - σ(x))
                float dr_gate = dr_t * r_t * (1.0f - r_t);
                float dz_gate = dz_t * z_t * (1.0f - z_t);

                // Store gradients
                grad_reset_ptr[idx] = dr_gate;
                grad_update_ptr[idx] = dz_gate;
                grad_new_input_ptr[idx] = dn_input;
                grad_new_hidden_ptr[idx] = dn_hidden;

                // Gradient for previous hidden also comes from new gate hidden part
                grad_h_prev_ptr[idx] = dh_prev;
            }
        );
    }
    else if (grad_h.dtype() == DType::Float64) {
        const double* grad_h_ptr = get_data_ptr<const double>(grad_h);
        const double* reset_ptr = get_data_ptr<const double>(reset_gates);
        const double* update_ptr = get_data_ptr<const double>(update_gates);
        const double* new_input_ptr = get_data_ptr<const double>(new_gates_input);
        const double* new_hidden_ptr = get_data_ptr<const double>(new_gates_hidden);
        const double* h_prev_ptr = get_data_ptr<const double>(h_prev);

        double* grad_reset_ptr = get_data_ptr<double>(outputs.grad_reset);
        double* grad_update_ptr = get_data_ptr<double>(outputs.grad_update);
        double* grad_new_input_ptr = get_data_ptr<double>(outputs.grad_new_input);
        double* grad_new_hidden_ptr = get_data_ptr<double>(outputs.grad_new_hidden);
        double* grad_h_prev_ptr = get_data_ptr<double>(outputs.grad_h_prev);

        queue.parallel_for<GRUCellBackwardFloat64>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                double r_gate = reset_ptr[idx];
                double z_gate = update_ptr[idx];
                double n_input = new_input_ptr[idx];
                double n_hidden = new_hidden_ptr[idx];
                double h_prev_val = h_prev_ptr[idx];

                // Recompute activations with the SAME ±20 clamp the forward
                // applies to the sigmoid gates, so derivatives match the forward.
                double r_clamped = sycl::fmax(-20.0, sycl::fmin(20.0, r_gate));
                double z_clamped = sycl::fmax(-20.0, sycl::fmin(20.0, z_gate));
                double r_t = 1.0 / (1.0 + sycl::exp(-r_clamped));
                double z_t = 1.0 / (1.0 + sycl::exp(-z_clamped));

                double n_combined = n_input + r_t * n_hidden;
                double n_t = sycl::tanh(n_combined);

                double dh = grad_h_ptr[idx];

                double dn_t = dh * (1.0 - z_t);
                double dz_t = dh * (h_prev_val - n_t);
                double dh_prev = dh * z_t;

                double dn_combined = dn_t * (1.0 - n_t * n_t);

                double dn_input = dn_combined;
                double dr_t = dn_combined * n_hidden;
                double dn_hidden = dn_combined * r_t;

                double dr_gate = dr_t * r_t * (1.0 - r_t);
                double dz_gate = dz_t * z_t * (1.0 - z_t);

                grad_reset_ptr[idx] = dr_gate;
                grad_update_ptr[idx] = dz_gate;
                grad_new_input_ptr[idx] = dn_input;
                grad_new_hidden_ptr[idx] = dn_hidden;
                grad_h_prev_ptr[idx] = dh_prev;
            }
        );
    }
    else if (grad_h.dtype() == DType::BFloat16) {
        const uint16_t* grad_h_ptr = get_data_ptr<const uint16_t>(grad_h);
        const uint16_t* reset_ptr = get_data_ptr<const uint16_t>(reset_gates);
        const uint16_t* update_ptr = get_data_ptr<const uint16_t>(update_gates);
        const uint16_t* new_input_ptr = get_data_ptr<const uint16_t>(new_gates_input);
        const uint16_t* new_hidden_ptr = get_data_ptr<const uint16_t>(new_gates_hidden);
        const uint16_t* h_prev_ptr = get_data_ptr<const uint16_t>(h_prev);

        uint16_t* grad_reset_ptr = get_data_ptr<uint16_t>(outputs.grad_reset);
        uint16_t* grad_update_ptr = get_data_ptr<uint16_t>(outputs.grad_update);
        uint16_t* grad_new_input_ptr = get_data_ptr<uint16_t>(outputs.grad_new_input);
        uint16_t* grad_new_hidden_ptr = get_data_ptr<uint16_t>(outputs.grad_new_hidden);
        uint16_t* grad_h_prev_ptr = get_data_ptr<uint16_t>(outputs.grad_h_prev);

        queue.parallel_for<GRUCellBackwardBFloat16>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                float r_gate = bf16_to_f32(reset_ptr[idx]);
                float z_gate = bf16_to_f32(update_ptr[idx]);
                float n_input = bf16_to_f32(new_input_ptr[idx]);
                float n_hidden = bf16_to_f32(new_hidden_ptr[idx]);
                float h_prev_val = bf16_to_f32(h_prev_ptr[idx]);

                // Recompute activations with the SAME ±20 clamp the forward
                // applies to the sigmoid gates, so derivatives match the forward.
                float r_clamped = sycl::fmax(-20.0f, sycl::fmin(20.0f, r_gate));
                float z_clamped = sycl::fmax(-20.0f, sycl::fmin(20.0f, z_gate));
                float r_t = 1.0f / (1.0f + sycl::exp(-r_clamped));
                float z_t = 1.0f / (1.0f + sycl::exp(-z_clamped));

                float n_combined = n_input + r_t * n_hidden;
                float n_t = sycl::tanh(n_combined);

                float dh = bf16_to_f32(grad_h_ptr[idx]);

                float dn_t = dh * (1.0f - z_t);
                float dz_t = dh * (h_prev_val - n_t);
                float dh_prev = dh * z_t;

                float dn_combined = dn_t * (1.0f - n_t * n_t);

                float dn_input = dn_combined;
                float dr_t = dn_combined * n_hidden;
                float dn_hidden = dn_combined * r_t;

                float dr_gate = dr_t * r_t * (1.0f - r_t);
                float dz_gate = dz_t * z_t * (1.0f - z_t);

                grad_reset_ptr[idx] = f32_to_bf16(dr_gate);
                grad_update_ptr[idx] = f32_to_bf16(dz_gate);
                grad_new_input_ptr[idx] = f32_to_bf16(dn_input);
                grad_new_hidden_ptr[idx] = f32_to_bf16(dn_hidden);
                grad_h_prev_ptr[idx] = f32_to_bf16(dh_prev);
            }
        );
    }
    else {
        throw std::runtime_error("gru_cell_backward: only Float32, Float64, and BFloat16 supported");
    }

    // Drain before the host reads the USM-shared gradient outputs.
    queue.wait_and_throw();
    return outputs;
}

} // namespace oneapi
} // namespace tenzor
