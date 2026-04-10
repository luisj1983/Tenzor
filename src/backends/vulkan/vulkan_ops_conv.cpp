#include "vulkan_ops_common.hpp"

namespace tenzor {

// ============================================================================
// Conv2d Forward Operation (OpAttributes version)
// ============================================================================

auto VulkanBackend::dispatchConv2dForward(const Tensor& input, const Tensor& weight, const Tensor* bias, const OpAttributes& attrs) -> Tensor {
    // Phase 2.1: Float16 path now runs natively via `conv2d_forward_f16.comp`,
    // which accumulates in float32 internally (see that shader's `float sum`
    // accumulator at line 45). The previous host-side upcast to Float32
    // doubled memory bandwidth and VRAM for zero numerical benefit. The
    // shader selection below at the `input.dtype() == DType::Float16` branch
    // used to be unreachable because of the upcast — now it lights up.

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    if (input_shape.size() != 4) {
        throw std::invalid_argument("conv2d_forward requires 4D input (N, C, H, W)");
    }
    if (weight_shape.size() != 4) {
        throw std::invalid_argument("conv2d_forward requires 4D weight (out_channels, in_channels, kH, kW)");
    }

    // Extract attributes
    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
    int64_t groups = attrs.get_int(AttrKey::Groups, 1);
    bool has_bias = (bias != nullptr);

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Route to Winograd F(2,3) for 3x3 convolutions with stride=1, dilation=1, Float32
    // Winograd is significantly faster for these common configurations.
    if (kernel_h == 3 && kernel_w == 3 && stride == 1 && dilation == 1
        && input.dtype() == DType::Float32) {
        return dispatchConv2dWinograd(input, weight, bias, padding, groups);
    }

    // Calculate output dimensions
    int64_t out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "conv2d_forward";
    if (input.dtype() == DType::Float64) {
        shader_name = "conv2d_forward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "conv2d_forward_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_weight = weight.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_bias = has_bias ? bias->data_ptr() : buffer_output;

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_weight = weight.numel() * weight.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();
    size_t buffer_size_bias = has_bias ? (bias->numel() * bias->dtype_size()) : 4;

    // Setup descriptor set bindings (input, weight, bias, output)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_weight},
        {2, buffer_bias},
        {3, buffer_output}
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_weight,
        buffer_size_bias,
        buffer_size_output
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t in_channels;
        uint32_t out_channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
        uint32_t groups;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t has_bias;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.in_channels = static_cast<uint32_t>(in_channels);
    push_constants.out_channels = static_cast<uint32_t>(out_channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);
    push_constants.groups = static_cast<uint32_t>(groups);
    push_constants.out_h = static_cast<uint32_t>(out_height);
    push_constants.out_w = static_cast<uint32_t>(out_width);
    push_constants.has_bias = has_bias ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    // Saturate FP16 output: clamp ±Inf to ±65504 to prevent NaN propagation
    return fp16_saturate_if_needed(*this, output);
}

// ============================================================================
// Conv2d Winograd F(2,3) Forward — 3x3 kernel, stride=1, dilation=1
// ============================================================================

auto VulkanBackend::dispatchConv2dWinograd(const Tensor& input, const Tensor& weight,
                                            const Tensor* bias,
                                            int64_t padding, int64_t groups) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];
    int64_t out_channels = weight_shape[0];

    // Output size for stride=1, dilation=1, 3x3 kernel
    int64_t out_height = in_height + 2 * padding - 2;  // in_h + 2p - (3-1)
    int64_t out_width = in_width + 2 * padding - 2;

    // Number of 2x2 output tiles
    int64_t tiles_h = (out_height + 1) / 2;
    int64_t tiles_w = (out_width + 1) / 2;

    int32_t device_id = input.device().index;
    bool has_bias = (bias != nullptr);

    // For grouped convolution, fall back to standard path for now
    // (Winograd with groups requires per-group transforms)
    if (groups > 1) {
        // Use the standard conv2d_forward shader directly (no Winograd for grouped conv)
        std::string shader_name = "conv2d_forward";
        auto* pipeline = getPipeline(shader_name, device_id);

        std::vector<int64_t> output_shape = {batch, out_channels, out_height, out_width};
        Tensor output(output_shape, input.dtype(), input.device());

        const void* buffer_bias = has_bias ? bias->data_ptr() : output.data_ptr();
        size_t buffer_size_bias = has_bias ? (bias->numel() * bias->dtype_size()) : 4;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()}, {1, weight.data_ptr()},
            {2, buffer_bias}, {3, output.data_ptr()}
        };
        std::vector<size_t> sizes = {
            static_cast<size_t>(input.numel() * input.dtype_size()),
            static_cast<size_t>(weight.numel() * weight.dtype_size()),
            buffer_size_bias,
            static_cast<size_t>(output.numel() * output.dtype_size())
        };

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);

        struct {
            uint32_t n_elements, batch, in_channels, out_channels;
            uint32_t in_height, in_width, kernel_h, kernel_w;
            uint32_t stride, padding, dilation, groups;
            uint32_t out_h, out_w, has_bias;
        } pc;
        pc.n_elements = static_cast<uint32_t>(output.numel());
        pc.batch = static_cast<uint32_t>(batch);
        pc.in_channels = static_cast<uint32_t>(in_channels);
        pc.out_channels = static_cast<uint32_t>(out_channels);
        pc.in_height = static_cast<uint32_t>(in_height);
        pc.in_width = static_cast<uint32_t>(in_width);
        pc.kernel_h = 3; pc.kernel_w = 3;
        pc.stride = 1; pc.padding = static_cast<uint32_t>(padding);
        pc.dilation = 1; pc.groups = static_cast<uint32_t>(groups);
        pc.out_h = static_cast<uint32_t>(out_height);
        pc.out_w = static_cast<uint32_t>(out_width);
        pc.has_bias = has_bias ? 1u : 0u;

        vkCmdPushConstants(cmd, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
        vkCmdDispatch(cmd, workgroups, 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    // ---- Winograd F(2,3) path for groups=1 ----

    // Step 1: Transform filters  G * g * G^T -> U [16][K][C]
    uint32_t total_filters = static_cast<uint32_t>(out_channels * in_channels);
    // U is stored as [16][K*C] (16 = 4x4 Winograd domain elements)
    Tensor U({16 * static_cast<int64_t>(out_channels) * in_channels}, input.dtype(), input.device());

    {
        auto* filter_pipeline = getPipeline("winograd_filter_transform", device_id);
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, weight.data_ptr()}, {1, U.data_ptr()}
        };
        std::vector<size_t> sizes = {
            static_cast<size_t>(weight.numel() * weight.dtype_size()),
            static_cast<size_t>(U.numel() * U.dtype_size())
        };
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, filter_pipeline, bindings, sizes);

        struct { uint32_t out_channels; uint32_t in_channels; } fpc;
        fpc.out_channels = static_cast<uint32_t>(out_channels);
        fpc.in_channels = static_cast<uint32_t>(in_channels);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, filter_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               filter_pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, filter_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(fpc), &fpc);
        uint32_t wg = static_cast<uint32_t>(div_wg(total_filters, devices_[device_id].workgroupSize));
        vkCmdDispatch(cmd, wg, 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    // Step 2: Transform input tiles  B^T * d * B -> V [16][N*C*tiles_h*tiles_w]
    uint32_t total_tiles = static_cast<uint32_t>(batch * in_channels * tiles_h * tiles_w);
    Tensor V({16 * static_cast<int64_t>(total_tiles)}, input.dtype(), input.device());

    {
        auto* input_pipeline = getPipeline("winograd_input_transform", device_id);
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()}, {1, V.data_ptr()}
        };
        std::vector<size_t> sizes = {
            static_cast<size_t>(input.numel() * input.dtype_size()),
            static_cast<size_t>(V.numel() * V.dtype_size())
        };
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, input_pipeline, bindings, sizes);

        struct {
            uint32_t batch, channels, in_height, in_width;
            uint32_t tiles_h, tiles_w, pad;
        } ipc;
        ipc.batch = static_cast<uint32_t>(batch);
        ipc.channels = static_cast<uint32_t>(in_channels);
        ipc.in_height = static_cast<uint32_t>(in_height);
        ipc.in_width = static_cast<uint32_t>(in_width);
        ipc.tiles_h = static_cast<uint32_t>(tiles_h);
        ipc.tiles_w = static_cast<uint32_t>(tiles_w);
        ipc.pad = static_cast<uint32_t>(padding);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, input_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               input_pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, input_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ipc), &ipc);
        uint32_t wg = static_cast<uint32_t>(div_wg(total_tiles, devices_[device_id].workgroupSize));
        vkCmdDispatch(cmd, wg, 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    // Step 3: Element-wise matmul in Winograd domain
    // For each of the 16 (alpha,beta) pairs:
    //   M[alpha][beta] = U[alpha][beta] (K x C) * V[alpha][beta] (C x tiles)^T
    // We reshape and use batched matmul: 16 batches of (K x C) @ (C x N*tiles_h*tiles_w)
    // U layout: [16][K*C], V layout: [16][N*C*tiles_h*tiles_w]
    // We need M: [16][N*K*tiles_h*tiles_w]
    //
    // For each Winograd point: M(k, n*th*tw) = sum_c U(k,c) * V(n*c*th*tw... )
    // This is a batched matmul: for each of the 16 points,
    //   U_slice: (K, C), V_slice: (C, N*tiles_h*tiles_w) -> M_slice: (K, N*tiles_h*tiles_w)
    // But V is stored as [16][N][C][tiles_h][tiles_w], so we need V transposed per-point
    // to get (C, N*tiles_h*tiles_w). Actually V is stored [16][tile_idx] where
    // tile_idx = (n*C + c)*tiles_h*tiles_w + th*tiles_w + tw, so within each Winograd point
    // the data is (N*C*TH*TW) interleaved.
    //
    // We'll use a simpler approach: for each Winograd point, dispatch a matmul
    // U_point (K, C) x V_point_reshaped (C, N*TH*TW) = M_point (K, N*TH*TW)
    // The tile layout from the shaders stores V as [16][tile_idx] where
    // tile_idx indexes (n, c, th, tw) -- so we reshape V_point to (N*TH*TW, C) and
    // compute M = U_point @ V_point^T... This is getting complex.
    //
    // Simpler: use element-wise multiplication. The standard Winograd approach does
    // pointwise multiplies M[i][j][k][n_tile] = sum_c U[i][j][k][c] * V[i][j][c][n_tile]
    // which is a batched matmul. We can use the existing matmul dispatch for each point.
    uint32_t N_tiles = static_cast<uint32_t>(batch * tiles_h * tiles_w);
    Tensor M({16 * static_cast<int64_t>(out_channels) * static_cast<int64_t>(N_tiles)},
             input.dtype(), input.device());

    {
        // For each of the 16 Winograd domain points, compute:
        //   M_p = U_p (K, C) @ V_p (C, N*TH*TW) = (K, N*TH*TW)
        // The input transform stores V as [16][tile_stride] where
        // tile_stride = N*C*TH*TW indexed as (n*C+c)*TH*TW + th*TW + tw.
        // So V_p naturally has layout (N, C, TH, TW) which we permute to (C, N*TH*TW).

        std::vector<Tensor> M_parts;
        M_parts.reserve(16);

        for (int p = 0; p < 16; p++) {
            int64_t u_offset = static_cast<int64_t>(p) * out_channels * in_channels;
            int64_t v_offset = static_cast<int64_t>(p) * batch * in_channels * tiles_h * tiles_w;

            Tensor U_p = U.slice(0, u_offset, u_offset + out_channels * in_channels)
                           .reshape({out_channels, in_channels});

            // V_p: (N, C, TH*TW) -> permute -> (C, N, TH*TW) -> reshape -> (C, N_tiles)
            Tensor V_p = V.slice(0, v_offset, v_offset + batch * in_channels * tiles_h * tiles_w)
                           .reshape({batch, in_channels, tiles_h * tiles_w})
                           .permute({1, 0, 2})
                           .contiguous()
                           .reshape({in_channels, static_cast<int64_t>(N_tiles)});

            // M_p = U_p @ V_p -> (K, N_tiles)
            M_parts.push_back(dispatchMatmul(U_p, V_p));
        }

        // Stack into M: [16, K, N_tiles]
        std::span<const Tensor> parts_span(M_parts);
        M = dispatchStack(parts_span, 0);
    }

    // Step 4: Output transform  A^T * M * A -> Y [N, K, out_h, out_w]
    std::vector<int64_t> output_shape = {batch, out_channels, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    {
        // M from stack is [16, K, N_tiles] where N_tiles = N*TH*TW.
        // The output transform shader expects [16][N][K][tiles_h][tiles_w].
        // Rearrange: [16, K, N*TH*TW] -> [16, K, N, TH, TW] -> [16, N, K, TH, TW]
        Tensor M_reordered = M.reshape({16, out_channels, batch, tiles_h, tiles_w})
                               .permute({0, 2, 1, 3, 4})
                               .contiguous();

        auto* output_pipeline = getPipeline("winograd_output_transform", device_id);
        uint32_t total_out_tiles = static_cast<uint32_t>(batch * out_channels * tiles_h * tiles_w);

        const void* buffer_bias_ptr = has_bias ? bias->data_ptr() : output.data_ptr();
        size_t buffer_bias_size = has_bias ? (bias->numel() * bias->dtype_size()) : 4;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, M_reordered.data_ptr()}, {1, buffer_bias_ptr}, {2, output.data_ptr()}
        };
        std::vector<size_t> sizes_out = {
            static_cast<size_t>(M_reordered.numel() * M_reordered.dtype_size()),
            buffer_bias_size,
            static_cast<size_t>(output.numel() * output.dtype_size())
        };
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, output_pipeline, bindings, sizes_out);

        struct {
            uint32_t batch, out_channels, out_height, out_width;
            uint32_t tiles_h, tiles_w, has_bias;
        } opc;
        opc.batch = static_cast<uint32_t>(batch);
        opc.out_channels = static_cast<uint32_t>(out_channels);
        opc.out_height = static_cast<uint32_t>(out_height);
        opc.out_width = static_cast<uint32_t>(out_width);
        opc.tiles_h = static_cast<uint32_t>(tiles_h);
        opc.tiles_w = static_cast<uint32_t>(tiles_w);
        opc.has_bias = has_bias ? 1u : 0u;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, output_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               output_pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, output_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(opc), &opc);
        uint32_t wg = static_cast<uint32_t>(div_wg(total_out_tiles, devices_[device_id].workgroupSize));
        vkCmdDispatch(cmd, wg, 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    return output;
}

// ============================================================================
// ConvTranspose2d Forward Operation
// ============================================================================

auto VulkanBackend::dispatchConvTranspose2dForward(const Tensor& input, const Tensor& weight, const Tensor* bias, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    if (input_shape.size() != 4) {
        throw std::invalid_argument("conv_transpose2d_forward requires 4D input (N, C, H, W)");
    }
    if (weight_shape.size() != 4) {
        throw std::invalid_argument("conv_transpose2d_forward requires 4D weight (in_channels, out_channels/groups, kH, kW)");
    }

    // Extract attributes
    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
    int64_t groups = attrs.get_int(AttrKey::Groups, 1);
    bool has_bias = (bias != nullptr);

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    // Weight shape for transposed conv: [in_channels, out_channels/groups, kH, kW]
    int64_t out_channels_per_group = weight_shape[1];
    int64_t out_channels = out_channels_per_group * groups;
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions for transposed convolution
    // out = (in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    int64_t out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    int64_t out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument("Invalid conv_transpose2d configuration: output dimensions are non-positive");
    }

    int32_t device_id = input.device().index;

    // Select pipeline based on dtype
    std::string shader_name = "conv_transpose2d_forward";
    if (input.dtype() == DType::Float64) shader_name = "conv_transpose2d_forward_f64";
    else if (input.dtype() == DType::Float16) shader_name = "conv_transpose2d_forward_f16";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_weight = weight.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_bias = has_bias ? bias->data_ptr() : buffer_output;

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_weight = weight.numel() * weight.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();
    size_t buffer_size_bias = has_bias ? (bias->numel() * bias->dtype_size()) : 4;

    // Setup descriptor set bindings (input, weight, bias, output)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_weight},
        {2, buffer_bias},
        {3, buffer_output}
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_weight,
        buffer_size_bias,
        buffer_size_output
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t in_channels;
        uint32_t out_channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride;
        uint32_t padding;
        uint32_t output_padding;
        uint32_t dilation;
        uint32_t groups;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t has_bias;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.in_channels = static_cast<uint32_t>(in_channels);
    push_constants.out_channels = static_cast<uint32_t>(out_channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.output_padding = static_cast<uint32_t>(output_padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);
    push_constants.groups = static_cast<uint32_t>(groups);
    push_constants.out_h = static_cast<uint32_t>(out_height);
    push_constants.out_w = static_cast<uint32_t>(out_width);
    push_constants.has_bias = has_bias ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return fp16_saturate_if_needed(*this, output);
}

// ============================================================================
// Conv3d Forward Operation
// ============================================================================

auto VulkanBackend::dispatchConv3dForward(const Tensor& input, const Tensor& weight, const Tensor* bias, const OpAttributes& attrs) -> Tensor {
    // For Float16, upcast to Float32 to avoid overflow in accumulation
    if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        std::optional<Tensor> bias_f32;
        const Tensor* bias_f32_ptr = nullptr;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
            bias_f32_ptr = &*bias_f32;
        }
        auto result_f32 = dispatchConv3dForward(input_f32, weight_f32, bias_f32_ptr, attrs);
        result_f32 = dispatchClamp(result_f32, -65504.0f, 65504.0f);
        return result_f32.to(DType::Float16);
    }

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    if (input_shape.size() != 5) {
        throw std::invalid_argument("conv3d_forward requires 5D input (N, C, D, H, W)");
    }
    if (weight_shape.size() != 5) {
        throw std::invalid_argument("conv3d_forward requires 5D weight (out_channels, in_channels/groups, kD, kH, kW)");
    }

    // Extract attributes
    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
    int64_t groups = attrs.get_int(AttrKey::Groups, 1);
    bool has_bias = (bias != nullptr);

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_depth = input_shape[2];
    int64_t in_height = input_shape[3];
    int64_t in_width = input_shape[4];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_d = weight_shape[2];
    int64_t kernel_h = weight_shape[3];
    int64_t kernel_w = weight_shape[4];

    // Calculate output dimensions
    int64_t out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int64_t out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "conv3d_forward";
    if (input.dtype() == DType::Float64) {
        shader_name = "conv3d_forward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "conv3d_forward_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_depth, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_weight = weight.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_bias = has_bias ? bias->data_ptr() : buffer_output;

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_weight = weight.numel() * weight.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();
    size_t buffer_size_bias = has_bias ? (bias->numel() * bias->dtype_size()) : 4;

    // Setup descriptor set bindings (input, weight, bias, output)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_weight},
        {2, buffer_bias},
        {3, buffer_output}
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_weight,
        buffer_size_bias,
        buffer_size_output
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t in_channels;
        uint32_t out_channels;
        uint32_t in_depth;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t kernel_d;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_d;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_d;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t dilation_d;
        uint32_t dilation_h;
        uint32_t dilation_w;
        uint32_t groups;
        uint32_t out_d;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t has_bias;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.in_channels = static_cast<uint32_t>(in_channels);
    push_constants.out_channels = static_cast<uint32_t>(out_channels);
    push_constants.in_depth = static_cast<uint32_t>(in_depth);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.kernel_d = static_cast<uint32_t>(kernel_d);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_d = static_cast<uint32_t>(stride);
    push_constants.stride_h = static_cast<uint32_t>(stride);
    push_constants.stride_w = static_cast<uint32_t>(stride);
    push_constants.padding_d = static_cast<uint32_t>(padding);
    push_constants.padding_h = static_cast<uint32_t>(padding);
    push_constants.padding_w = static_cast<uint32_t>(padding);
    push_constants.dilation_d = static_cast<uint32_t>(dilation);
    push_constants.dilation_h = static_cast<uint32_t>(dilation);
    push_constants.dilation_w = static_cast<uint32_t>(dilation);
    push_constants.groups = static_cast<uint32_t>(groups);
    push_constants.out_d = static_cast<uint32_t>(out_depth);
    push_constants.out_h = static_cast<uint32_t>(out_height);
    push_constants.out_w = static_cast<uint32_t>(out_width);
    push_constants.has_bias = has_bias ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return fp16_saturate_if_needed(*this, output);
}

// ============================================================================
// Conv3d Backward Input
// ============================================================================

auto VulkanBackend::dispatchConv3dBackwardInput(
    const Tensor& grad_output,
    const Tensor& weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    const std::vector<int64_t>& input_shape,
    int64_t groups) -> Tensor {

    // Extract dimensions
    auto grad_shape = grad_output.shape();
    auto weight_shape = weight.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t depth_out = grad_shape[2];
    int64_t height_out = grad_shape[3];
    int64_t width_out = grad_shape[4];

    int64_t channels_in = input_shape[1];
    int64_t depth_in = input_shape[2];
    int64_t height_in = input_shape[3];
    int64_t width_in = input_shape[4];

    int64_t kernel_d = weight_shape[2];
    int64_t kernel_h = weight_shape[3];
    int64_t kernel_w = weight_shape[4];

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype (F16 uses native shader with F32 accumulation)
    std::string shader_name = "conv3d_backward_input";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv3d_backward_input_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "conv3d_backward_input_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient input tensor
    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_weight = weight.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    // Calculate buffer sizes (round up to 4-byte boundary for F16 packed uint32 access)
    size_t buffer_size_grad_out, buffer_size_weight, buffer_size_grad_in;
    if (grad_output.dtype() == DType::Float16) {
        buffer_size_grad_out = ((grad_output.numel() + 1) / 2) * 4;
        buffer_size_weight = ((weight.numel() + 1) / 2) * 4;
        buffer_size_grad_in = ((grad_input.numel() + 1) / 2) * 4;
    } else {
        buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
        buffer_size_weight = weight.numel() * weight.dtype_size();
        buffer_size_grad_in = grad_input.numel() * grad_input.dtype_size();
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_weight},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_weight, buffer_size_grad_in};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels_in;
        uint32_t channels_out;
        uint32_t depth_in;
        uint32_t height_in;
        uint32_t width_in;
        uint32_t depth_out;
        uint32_t height_out;
        uint32_t width_out;
        uint32_t kernel_d;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_d;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_d;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t dilation_d;
        uint32_t dilation_h;
        uint32_t dilation_w;
        uint32_t groups;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels_in = static_cast<uint32_t>(channels_in);
    push_constants.channels_out = static_cast<uint32_t>(channels_out);
    push_constants.depth_in = static_cast<uint32_t>(depth_in);
    push_constants.height_in = static_cast<uint32_t>(height_in);
    push_constants.width_in = static_cast<uint32_t>(width_in);
    push_constants.depth_out = static_cast<uint32_t>(depth_out);
    push_constants.height_out = static_cast<uint32_t>(height_out);
    push_constants.width_out = static_cast<uint32_t>(width_out);
    push_constants.kernel_d = static_cast<uint32_t>(kernel_d);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_d = static_cast<uint32_t>(stride);
    push_constants.stride_h = static_cast<uint32_t>(stride);
    push_constants.stride_w = static_cast<uint32_t>(stride);
    push_constants.padding_d = static_cast<uint32_t>(padding);
    push_constants.padding_h = static_cast<uint32_t>(padding);
    push_constants.padding_w = static_cast<uint32_t>(padding);
    push_constants.dilation_d = static_cast<uint32_t>(dilation);
    push_constants.dilation_h = static_cast<uint32_t>(dilation);
    push_constants.dilation_w = static_cast<uint32_t>(dilation);
    push_constants.groups = static_cast<uint32_t>(groups);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    int64_t total_elements = batch * channels_in * depth_in * height_in * width_in;
    uint32_t workgroups = static_cast<uint32_t>(div_wg(total_elements, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// ============================================================================
// Conv3d Backward Weight
// ============================================================================

auto VulkanBackend::dispatchConv3dBackwardWeight(
    const Tensor& grad_output,
    const Tensor& input,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    const std::vector<int64_t>& weight_shape,
    int64_t groups) -> Tensor {

    // Extract dimensions
    auto grad_shape = grad_output.shape();
    auto input_shape = input.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t depth_out = grad_shape[2];
    int64_t height_out = grad_shape[3];
    int64_t width_out = grad_shape[4];

    int64_t channels_in = input_shape[1];
    int64_t depth_in = input_shape[2];
    int64_t height_in = input_shape[3];
    int64_t width_in = input_shape[4];

    int64_t kernel_d = weight_shape[2];
    int64_t kernel_h = weight_shape[3];
    int64_t kernel_w = weight_shape[4];

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype (F16 uses native shader with F32 accumulation)
    std::string shader_name = "conv3d_backward_weight";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv3d_backward_weight_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "conv3d_backward_weight_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient weight tensor
    Tensor grad_weight(weight_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_input = input.data_ptr();
    const void* buffer_grad_weight = grad_weight.data_ptr();

    // Calculate buffer sizes (round up to 4-byte boundary for F16 packed uint32 access)
    size_t buffer_size_grad_out, buffer_size_input, buffer_size_grad_weight;
    if (grad_output.dtype() == DType::Float16) {
        buffer_size_grad_out = ((grad_output.numel() + 1) / 2) * 4;
        buffer_size_input = ((input.numel() + 1) / 2) * 4;
        buffer_size_grad_weight = ((grad_weight.numel() + 1) / 2) * 4;
    } else {
        buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
        buffer_size_input = input.numel() * input.dtype_size();
        buffer_size_grad_weight = grad_weight.numel() * grad_weight.dtype_size();
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_input},
        {2, buffer_grad_weight}
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_input, buffer_size_grad_weight};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels_in;
        uint32_t channels_out;
        uint32_t depth_in;
        uint32_t height_in;
        uint32_t width_in;
        uint32_t depth_out;
        uint32_t height_out;
        uint32_t width_out;
        uint32_t kernel_d;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_d;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_d;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t dilation_d;
        uint32_t dilation_h;
        uint32_t dilation_w;
        uint32_t groups;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels_in = static_cast<uint32_t>(channels_in);
    push_constants.channels_out = static_cast<uint32_t>(channels_out);
    push_constants.depth_in = static_cast<uint32_t>(depth_in);
    push_constants.height_in = static_cast<uint32_t>(height_in);
    push_constants.width_in = static_cast<uint32_t>(width_in);
    push_constants.depth_out = static_cast<uint32_t>(depth_out);
    push_constants.height_out = static_cast<uint32_t>(height_out);
    push_constants.width_out = static_cast<uint32_t>(width_out);
    push_constants.kernel_d = static_cast<uint32_t>(kernel_d);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_d = static_cast<uint32_t>(stride);
    push_constants.stride_h = static_cast<uint32_t>(stride);
    push_constants.stride_w = static_cast<uint32_t>(stride);
    push_constants.padding_d = static_cast<uint32_t>(padding);
    push_constants.padding_h = static_cast<uint32_t>(padding);
    push_constants.padding_w = static_cast<uint32_t>(padding);
    push_constants.dilation_d = static_cast<uint32_t>(dilation);
    push_constants.dilation_h = static_cast<uint32_t>(dilation);
    push_constants.dilation_w = static_cast<uint32_t>(dilation);
    push_constants.groups = static_cast<uint32_t>(groups);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    int64_t in_channels_per_group = channels_in / groups;
    int64_t total_weight_elements = channels_out * in_channels_per_group * kernel_d * kernel_h * kernel_w;
    uint32_t workgroups = static_cast<uint32_t>(div_wg(total_weight_elements, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_weight;
}

// ============================================================================
// Conv3d Backward Bias
// ============================================================================

auto VulkanBackend::dispatchConv3dBackwardBias(const Tensor& grad_output) -> Tensor {
    // Extract dimensions
    auto grad_shape = grad_output.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t depth_out = grad_shape[2];
    int64_t height_out = grad_shape[3];
    int64_t width_out = grad_shape[4];

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype (F16 uses native shader with F32 accumulation)
    std::string shader_name = "conv3d_backward_bias";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv3d_backward_bias_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "conv3d_backward_bias_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient bias tensor
    std::vector<int64_t> bias_shape = {channels_out};
    Tensor grad_bias(bias_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_grad_bias = grad_bias.data_ptr();

    // Calculate buffer sizes (round up to 4-byte boundary for F16 packed uint32 access)
    size_t buffer_size_grad_out, buffer_size_grad_bias;
    if (grad_output.dtype() == DType::Float16) {
        buffer_size_grad_out = ((grad_output.numel() + 1) / 2) * 4;
        buffer_size_grad_bias = ((grad_bias.numel() + 1) / 2) * 4;
    } else {
        buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
        buffer_size_grad_bias = grad_bias.numel() * grad_bias.dtype_size();
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_grad_bias}
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_grad_bias};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels_out;
        uint32_t depth_out;
        uint32_t height_out;
        uint32_t width_out;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels_out = static_cast<uint32_t>(channels_out);
    push_constants.depth_out = static_cast<uint32_t>(depth_out);
    push_constants.height_out = static_cast<uint32_t>(height_out);
    push_constants.width_out = static_cast<uint32_t>(width_out);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (one thread per output channel)
    uint32_t workgroups = static_cast<uint32_t>(div_wg(channels_out, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_bias;
}

// ============================================================================
// ConvTranspose3d Forward (uses conv3d_backward_input shader via duality)
// ============================================================================

auto VulkanBackend::dispatchConvTranspose3dForward(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    const OpAttributes& attrs) -> Tensor {

    // ConvTranspose3d forward is Conv3d backward-input with swapped roles:
    // output_shape = computed from input shape + kernel + stride + padding + output_padding

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
    int64_t groups = attrs.get_int(AttrKey::Groups, 1);

    int64_t batch = input_shape[0];
    int64_t in_depth = input_shape[2];
    int64_t in_height = input_shape[3];
    int64_t in_width = input_shape[4];

    // For ConvTranspose3d: weight is (in_channels, out_channels/groups, kD, kH, kW)
    int64_t out_channels = weight_shape[1] * groups;
    int64_t kernel_d = weight_shape[2];
    int64_t kernel_h = weight_shape[3];
    int64_t kernel_w = weight_shape[4];

    // Output dimensions for transposed conv
    int64_t out_depth = (in_depth - 1) * stride - 2 * padding + dilation * (kernel_d - 1) + output_padding + 1;
    int64_t out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    int64_t out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    std::vector<int64_t> output_shape = {batch, out_channels, out_depth, out_height, out_width};

    // Use conv3d_backward_input shader: grad_output=input, weight=weight^T, output=result
    // The backward_input shader computes: for each output element, sum over input * weight
    // which is exactly the transposed convolution operation
    auto result = dispatchConv3dBackwardInput(input, weight, stride, padding, dilation, output_shape, groups);

    // Add bias if present — reshape bias to (1, C, 1, 1, 1) and broadcast-add
    if (bias) {
        auto bias_5d = bias->reshape({1, out_channels, 1, 1, 1});
        result = dispatchBinaryOp("add", result, bias_5d);
    }

    return result;
}

// ConvTranspose3d Backward Input (uses conv3d_forward shader via duality)
auto VulkanBackend::dispatchConvTranspose3dBackwardInput(
    const Tensor& grad_output, const Tensor& weight, const OpAttributes& attrs) -> Tensor {

    // ConvTranspose3d backward w.r.t. input = regular Conv3d forward
    return dispatchConv3dForward(grad_output, weight, nullptr, attrs);
}

// ConvTranspose3d Backward Weight (same computation as conv3d backward weight but with swapped roles)
auto VulkanBackend::dispatchConvTranspose3dBackwardWeight(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& weight_shape, const OpAttributes& attrs) -> Tensor {

    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
    int64_t groups = attrs.get_int(AttrKey::Groups, 1);

    // For ConvTranspose3d backward weight, roles are swapped:
    // input plays the role of grad_output, grad_output plays the role of input
    return dispatchConv3dBackwardWeight(input, grad_output, stride, padding, dilation, weight_shape, groups);
}

// ConvTranspose3d Backward Bias (same as conv3d backward bias)
auto VulkanBackend::dispatchConvTranspose3dBackwardBias(const Tensor& grad_output) -> Tensor {
    return dispatchConv3dBackwardBias(grad_output);
}


} // namespace tenzor
