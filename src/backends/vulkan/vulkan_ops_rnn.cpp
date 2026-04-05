#include "vulkan_ops_common.hpp"

namespace tenzor {

// ============================================================================
// Phase 11.3: RNN Operations
// ============================================================================

/**
 * @brief LSTM forward pass — loops over timesteps using lstm_cell compute shader.
 *
 * Input: [seq_len, batch_size, input_size]
 * Returns: [output, h_n, c_n]
 *   output: [seq_len, batch_size, hidden_size]
 *   h_n: [1, batch_size, hidden_size]
 *   c_n: [1, batch_size, hidden_size]
 */
auto VulkanBackend::dispatchLSTMForward(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                                         const Tensor& bias_ih, const Tensor& bias_hh,
                                         const Tensor& h0, const Tensor& c0) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch_size = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t hidden_size = W_hh.shape()[1];  // W_hh: [4*hidden_size, hidden_size]

    bool is_f64 = (input.dtype() == DType::Float64);
    std::string cell_shader = is_f64 ? "lstm_cell_f64" : "lstm_cell";
    int32_t device_id = input.device().index;

    // Output tensor: [seq_len, batch_size, hidden_size]
    Tensor output({seq_len, batch_size, hidden_size}, input.dtype(), input.device());

    // h, c state — initially h0, c0 (squeezed to [batch_size, hidden_size])
    // h0/c0 shape: [1, batch_size, hidden_size] or [batch_size, hidden_size]
    Tensor h_state = h0.numel() > 0 ? dispatchContiguous(h0).reshape({batch_size, hidden_size})
                                     : dispatchZeros({batch_size, hidden_size}, input.dtype(), input.device());
    Tensor c_state = c0.numel() > 0 ? dispatchContiguous(c0).reshape({batch_size, hidden_size})
                                     : dispatchZeros({batch_size, hidden_size}, input.dtype(), input.device());

    // Transpose weights for x*W_ih^T: W_ih is [4H, I], we need [I, 4H]
    Tensor W_ih_t = dispatchTranspose(W_ih, 0, 1);
    Tensor W_hh_t = dispatchTranspose(W_hh, 0, 1);

    auto* pipeline = getPipeline(cell_shader, device_id);

    for (int64_t t = 0; t < seq_len; ++t) {
        // x_t: [batch_size, input_size] — slice from input
        // We can use pointer arithmetic since input is contiguous along dim 0
        Tensor x_t = input.slice(0, t, t + 1).reshape({batch_size, input_size});

        // Compute gates = x_t * W_ih^T + h * W_hh^T
        Tensor gates = dispatchMatmul(x_t, W_ih_t);  // [batch, 4*hidden]
        Tensor h_gates = dispatchMatmul(h_state, W_hh_t);  // [batch, 4*hidden]
        gates = dispatchBinaryOp("add", gates, h_gates);

        // Add biases if non-empty
        if (bias_ih.numel() > 0) {
            gates = dispatchBinaryOp("add", gates, bias_ih);
        }
        if (bias_hh.numel() > 0) {
            gates = dispatchBinaryOp("add", gates, bias_hh);
        }

        // Allocate new h and c
        Tensor h_new({batch_size, hidden_size}, input.dtype(), input.device());
        Tensor c_new({batch_size, hidden_size}, input.dtype(), input.device());

        // Dispatch LSTM cell shader
        uint32_t total = static_cast<uint32_t>(batch_size * hidden_size);
        size_t elem_size = input.dtype_size();
        size_t gate_bytes = batch_size * 4 * hidden_size * elem_size;
        size_t state_bytes = batch_size * hidden_size * elem_size;

        struct { uint32_t batch_size; uint32_t hidden_size; } pc;
        pc.batch_size = static_cast<uint32_t>(batch_size);
        pc.hidden_size = static_cast<uint32_t>(hidden_size);

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, gates.data_ptr()}, {1, c_state.data_ptr()},
            {2, h_new.data_ptr()}, {3, c_new.data_ptr()}
        };
        std::vector<size_t> sizes = {gate_bytes, state_bytes, state_bytes, state_bytes};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        // Copy h_new into output[t]
        copy(static_cast<char*>(output.data_ptr()) + t * batch_size * hidden_size * elem_size,
             h_new.data_ptr(), state_bytes, CopyKind::DeviceToDevice);

        h_state = h_new;
        c_state = c_new;
    }

    synchronize(device_id);

    // h_n, c_n: [1, batch_size, hidden_size]
    Tensor h_n = h_state.reshape({1, batch_size, hidden_size});
    Tensor c_n = c_state.reshape({1, batch_size, hidden_size});

    return {output, h_n, c_n};
}

/**
 * @brief GRU forward pass — loops over timesteps using gru_cell compute shader.
 */
auto VulkanBackend::dispatchGRUForward(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                                        const Tensor& bias, const Tensor& h0) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch_size = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t hidden_size = W_hh.shape()[1];  // W_hh: [3*hidden_size, hidden_size]

    bool is_f64 = (input.dtype() == DType::Float64);
    std::string cell_shader = is_f64 ? "gru_cell_f64" : "gru_cell";
    int32_t device_id = input.device().index;

    Tensor output({seq_len, batch_size, hidden_size}, input.dtype(), input.device());

    Tensor h_state = h0.numel() > 0 ? dispatchContiguous(h0).reshape({batch_size, hidden_size})
                                     : dispatchZeros({batch_size, hidden_size}, input.dtype(), input.device());

    Tensor W_ih_t = dispatchTranspose(W_ih, 0, 1);
    Tensor W_hh_t = dispatchTranspose(W_hh, 0, 1);

    // Split bias into bias_ih and bias_hh if provided
    // GRU bias: [6*hidden_size] = [bias_ih(3H) | bias_hh(3H)]
    // Or the kernel registry passes just [bias] as a single combined bias
    Tensor bias_ih, bias_hh;
    if (bias.numel() >= 6 * hidden_size) {
        // Combined bias: first 3H = bias_ih, next 3H = bias_hh
        bias_ih = bias.slice(0, 0, 3 * hidden_size);
        bias_hh = bias.slice(0, 3 * hidden_size, 6 * hidden_size);
    } else if (bias.numel() >= 3 * hidden_size) {
        bias_ih = bias;
    }

    auto* pipeline = getPipeline(cell_shader, device_id);

    for (int64_t t = 0; t < seq_len; ++t) {
        Tensor x_t = input.slice(0, t, t + 1).reshape({batch_size, input_size});

        // gates_x = x_t * W_ih^T + bias_ih  [batch, 3*hidden]
        Tensor gates_x = dispatchMatmul(x_t, W_ih_t);
        if (bias_ih.numel() > 0) {
            gates_x = dispatchBinaryOp("add", gates_x, bias_ih);
        }

        // gates_h = h * W_hh^T + bias_hh  [batch, 3*hidden]
        Tensor gates_h = dispatchMatmul(h_state, W_hh_t);
        if (bias_hh.numel() > 0) {
            gates_h = dispatchBinaryOp("add", gates_h, bias_hh);
        }

        // Allocate new h
        Tensor h_new({batch_size, hidden_size}, input.dtype(), input.device());

        uint32_t total = static_cast<uint32_t>(batch_size * hidden_size);
        size_t elem_size = input.dtype_size();
        size_t gate_bytes = batch_size * 3 * hidden_size * elem_size;
        size_t state_bytes = batch_size * hidden_size * elem_size;

        struct { uint32_t batch_size; uint32_t hidden_size; } pc;
        pc.batch_size = static_cast<uint32_t>(batch_size);
        pc.hidden_size = static_cast<uint32_t>(hidden_size);

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, gates_x.data_ptr()}, {1, gates_h.data_ptr()},
            {2, h_state.data_ptr()}, {3, h_new.data_ptr()}
        };
        std::vector<size_t> sizes = {gate_bytes, gate_bytes, state_bytes, state_bytes};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        // Copy h_new into output[t]
        copy(static_cast<char*>(output.data_ptr()) + t * batch_size * hidden_size * elem_size,
             h_new.data_ptr(), state_bytes, CopyKind::DeviceToDevice);

        h_state = h_new;
    }

    synchronize(device_id);

    Tensor h_n = h_state.reshape({1, batch_size, hidden_size});
    return {output, h_n};
}

/**
 * @brief LSTM cell backward — computes gate and cell state gradients.
 *
 * Inputs: grad_h [batch, hidden], grad_c_next [batch, hidden],
 *         gates [batch, 4*hidden], c_prev [batch, hidden], c_out [batch, hidden]
 * Returns: [grad_gates, grad_c_prev]
 */
auto VulkanBackend::dispatchLSTMCellBackward(const Tensor& grad_h, const Tensor& grad_c_next,
                                              const Tensor& gates, const Tensor& c_prev,
                                              const Tensor& c_out,
                                              int64_t batch_size, int64_t hidden_size) -> std::vector<Tensor> {
    // Float16: upcast to Float32 for numerical stability
    if (grad_h.dtype() == DType::Float16) {
        DType orig = grad_h.dtype();
        auto results = dispatchLSTMCellBackward(
            grad_h.to(DType::Float32), grad_c_next.to(DType::Float32),
            gates.to(DType::Float32), c_prev.to(DType::Float32),
            c_out.to(DType::Float32), batch_size, hidden_size);
        for (auto& r : results) r = r.to(orig);
        return results;
    }

    int32_t device_id = grad_h.device().index;
    bool is_f64 = (grad_h.dtype() == DType::Float64);
    bool is_bf16 = (grad_h.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "lstm_cell_backward_f64" : is_bf16 ? "lstm_cell_backward_bf16" : "lstm_cell_backward";

    size_t elem_size = grad_h.dtype_size();
    size_t state_bytes = batch_size * hidden_size * elem_size;
    size_t gate_bytes = batch_size * 4 * hidden_size * elem_size;

    // Allocate outputs
    Tensor grad_gates({batch_size, 4 * hidden_size}, grad_h.dtype(), grad_h.device());
    Tensor grad_c_prev({batch_size, hidden_size}, grad_h.dtype(), grad_h.device());

    struct { uint32_t batch_size; uint32_t hidden_size; } pc;
    pc.batch_size = static_cast<uint32_t>(batch_size);
    pc.hidden_size = static_cast<uint32_t>(hidden_size);

    auto* pipeline = getPipeline(shader, device_id);
    uint32_t total = static_cast<uint32_t>(batch_size * hidden_size);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_h.data_ptr()},      {1, grad_c_next.data_ptr()},
        {2, gates.data_ptr()},        {3, c_prev.data_ptr()},
        {4, c_out.data_ptr()},        {5, grad_gates.data_ptr()},
        {6, grad_c_prev.data_ptr()}
    };
    std::vector<size_t> sizes = {
        state_bytes, state_bytes, gate_bytes, state_bytes,
        state_bytes, gate_bytes, state_bytes
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    synchronize(device_id);

    return {grad_gates, grad_c_prev};
}

/**
 * @brief GRU cell backward — computes gate and hidden state gradients.
 *
 * Inputs: grad_h [batch, hidden], gates_x [batch, 3*hidden],
 *         gates_h [batch, 3*hidden], h_prev [batch, hidden]
 * Returns: [grad_gates_x, grad_gates_h, grad_h_prev]
 */
auto VulkanBackend::dispatchGRUCellBackward(const Tensor& grad_h, const Tensor& gates_x,
                                             const Tensor& gates_h, const Tensor& h_prev,
                                             int64_t batch_size, int64_t hidden_size) -> std::vector<Tensor> {
    // Float16: upcast to Float32 for numerical stability
    if (grad_h.dtype() == DType::Float16) {
        DType orig = grad_h.dtype();
        auto results = dispatchGRUCellBackward(
            grad_h.to(DType::Float32), gates_x.to(DType::Float32),
            gates_h.to(DType::Float32), h_prev.to(DType::Float32),
            batch_size, hidden_size);
        for (auto& r : results) r = r.to(orig);
        return results;
    }

    int32_t device_id = grad_h.device().index;
    bool is_f64 = (grad_h.dtype() == DType::Float64);
    bool is_bf16 = (grad_h.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "gru_cell_backward_f64" : is_bf16 ? "gru_cell_backward_bf16" : "gru_cell_backward";

    size_t elem_size = grad_h.dtype_size();
    size_t state_bytes = batch_size * hidden_size * elem_size;
    size_t gate_bytes = batch_size * 3 * hidden_size * elem_size;

    // Allocate outputs
    Tensor grad_gates_x({batch_size, 3 * hidden_size}, grad_h.dtype(), grad_h.device());
    Tensor grad_gates_h({batch_size, 3 * hidden_size}, grad_h.dtype(), grad_h.device());
    Tensor grad_h_prev({batch_size, hidden_size}, grad_h.dtype(), grad_h.device());

    struct { uint32_t batch_size; uint32_t hidden_size; } pc;
    pc.batch_size = static_cast<uint32_t>(batch_size);
    pc.hidden_size = static_cast<uint32_t>(hidden_size);

    auto* pipeline = getPipeline(shader, device_id);
    uint32_t total = static_cast<uint32_t>(batch_size * hidden_size);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_h.data_ptr()},       {1, gates_x.data_ptr()},
        {2, gates_h.data_ptr()},      {3, h_prev.data_ptr()},
        {4, grad_gates_x.data_ptr()}, {5, grad_gates_h.data_ptr()},
        {6, grad_h_prev.data_ptr()}
    };
    std::vector<size_t> sizes = {
        state_bytes, gate_bytes, gate_bytes, state_bytes,
        gate_bytes, gate_bytes, state_bytes
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    synchronize(device_id);

    return {grad_gates_x, grad_gates_h, grad_h_prev};
}

/**
 * @brief Multi-layer LSTM forward — chains single-layer LSTM calls.
 */
auto VulkanBackend::dispatchLSTMMultiLayerForward(const Tensor& input,
                                                    const std::vector<Tensor>& W_ih_list,
                                                    const std::vector<Tensor>& W_hh_list,
                                                    const std::vector<Tensor>& bias_list,
                                                    const Tensor& h0, const Tensor& c0) -> std::vector<Tensor> {
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    auto shape = input.shape();
    int64_t batch_size = shape[1];
    int64_t hidden_size = W_hh_list[0].shape()[1];

    // h0, c0: [num_layers, batch_size, hidden_size]
    Tensor current_input = input;

    std::vector<Tensor> h_n_layers, c_n_layers;

    for (int64_t l = 0; l < num_layers; ++l) {
        // Extract per-layer initial states
        Tensor h0_l = h0.numel() > 0 ? h0.slice(0, l, l + 1).reshape({1, batch_size, hidden_size})
                                      : Tensor({0}, input.dtype(), input.device());
        Tensor c0_l = c0.numel() > 0 ? c0.slice(0, l, l + 1).reshape({1, batch_size, hidden_size})
                                      : Tensor({0}, input.dtype(), input.device());

        // Split combined bias into bias_ih and bias_hh
        Tensor bias_ih, bias_hh;
        if (bias_list[l].numel() >= 8 * hidden_size) {
            // Combined: [bias_ih(4H) | bias_hh(4H)]
            bias_ih = bias_list[l].slice(0, 0, 4 * hidden_size);
            bias_hh = bias_list[l].slice(0, 4 * hidden_size, 8 * hidden_size);
        } else if (bias_list[l].numel() > 0) {
            bias_ih = bias_list[l];
            bias_hh = Tensor({0}, input.dtype(), input.device());
        } else {
            bias_ih = Tensor({0}, input.dtype(), input.device());
            bias_hh = Tensor({0}, input.dtype(), input.device());
        }

        auto result = dispatchLSTMForward(current_input, W_ih_list[l], W_hh_list[l],
                                           bias_ih, bias_hh, h0_l, c0_l);
        current_input = result[0];  // output becomes input for next layer
        h_n_layers.push_back(result[1]);
        c_n_layers.push_back(result[2]);
    }

    // Stack h_n and c_n: [num_layers, batch, hidden]
    Tensor h_n = dispatchCat(h_n_layers, 0);
    Tensor c_n = dispatchCat(c_n_layers, 0);

    return {current_input, h_n, c_n};
}

/**
 * @brief Multi-layer GRU forward — chains single-layer GRU calls.
 */
auto VulkanBackend::dispatchGRUMultiLayerForward(const Tensor& input,
                                                   const std::vector<Tensor>& W_ih_list,
                                                   const std::vector<Tensor>& W_hh_list,
                                                   const std::vector<Tensor>& bias_list,
                                                   const Tensor& h0) -> std::vector<Tensor> {
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    auto shape = input.shape();
    int64_t batch_size = shape[1];
    int64_t hidden_size = W_hh_list[0].shape()[1];

    Tensor current_input = input;
    std::vector<Tensor> h_n_layers;

    for (int64_t l = 0; l < num_layers; ++l) {
        Tensor h0_l = h0.numel() > 0 ? h0.slice(0, l, l + 1).reshape({1, batch_size, hidden_size})
                                      : Tensor({0}, input.dtype(), input.device());

        auto result = dispatchGRUForward(current_input, W_ih_list[l], W_hh_list[l],
                                          bias_list[l], h0_l);
        current_input = result[0];
        h_n_layers.push_back(result[1]);
    }

    Tensor h_n = dispatchCat(h_n_layers, 0);
    return {current_input, h_n};
}

/**
 * @brief Bidirectional LSTM forward.
 * Runs forward LSTM and backward (reverse) LSTM, concatenates outputs.
 */
auto VulkanBackend::dispatchBiLSTMForward(const Tensor& input, const Tensor& h0, const Tensor& c0,
                                            const Tensor& W_ih_fwd, const Tensor& W_hh_fwd,
                                            const Tensor& bias_ih_fwd, const Tensor& bias_hh_fwd,
                                            const Tensor& W_ih_bwd, const Tensor& W_hh_bwd,
                                            const Tensor& bias_ih_bwd, const Tensor& bias_hh_bwd) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch_size = input_shape[1];
    int64_t hidden_size = W_hh_fwd.shape()[1];

    // Split h0, c0 into forward/backward: [2, batch, hidden]
    Tensor h0_fwd = h0.numel() > 0 ? h0.slice(0, 0, 1) : Tensor({0}, input.dtype(), input.device());
    Tensor c0_fwd = c0.numel() > 0 ? c0.slice(0, 0, 1) : Tensor({0}, input.dtype(), input.device());
    Tensor h0_bwd = h0.numel() > 0 ? h0.slice(0, 1, 2) : Tensor({0}, input.dtype(), input.device());
    Tensor c0_bwd = c0.numel() > 0 ? c0.slice(0, 1, 2) : Tensor({0}, input.dtype(), input.device());

    // Forward direction
    auto fwd_result = dispatchLSTMForward(input, W_ih_fwd, W_hh_fwd,
                                           bias_ih_fwd, bias_hh_fwd, h0_fwd, c0_fwd);

    // Reverse input along time dimension for backward direction
    // Create reversed input by copying slices in reverse order
    Tensor rev_input({seq_len, batch_size, input_shape[2]}, input.dtype(), input.device());
    size_t slice_bytes = batch_size * input_shape[2] * input.dtype_size();
    int32_t device_id = input.device().index;
    for (int64_t t = 0; t < seq_len; ++t) {
        copy(static_cast<char*>(rev_input.data_ptr()) + t * slice_bytes,
             static_cast<const char*>(input.data_ptr()) + (seq_len - 1 - t) * slice_bytes,
             slice_bytes, CopyKind::DeviceToDevice);
    }
    synchronize(device_id);

    // Backward direction
    auto bwd_result = dispatchLSTMForward(rev_input, W_ih_bwd, W_hh_bwd,
                                           bias_ih_bwd, bias_hh_bwd, h0_bwd, c0_bwd);

    // Reverse backward output back to original time order
    Tensor bwd_output({seq_len, batch_size, hidden_size}, input.dtype(), input.device());
    size_t h_slice_bytes = batch_size * hidden_size * input.dtype_size();
    for (int64_t t = 0; t < seq_len; ++t) {
        copy(static_cast<char*>(bwd_output.data_ptr()) + t * h_slice_bytes,
             static_cast<const char*>(bwd_result[0].data_ptr()) + (seq_len - 1 - t) * h_slice_bytes,
             h_slice_bytes, CopyKind::DeviceToDevice);
    }
    synchronize(device_id);

    // Concatenate forward and backward outputs along last dim: [seq, batch, 2*hidden]
    Tensor output = dispatchCat({fwd_result[0], bwd_output}, 2);

    // Stack h_n, c_n: [2, batch, hidden]
    Tensor h_n = dispatchCat({fwd_result[1], bwd_result[1]}, 0);
    Tensor c_n = dispatchCat({fwd_result[2], bwd_result[2]}, 0);

    return {output, h_n, c_n};
}

// ============================================================================

} // namespace tenzor
