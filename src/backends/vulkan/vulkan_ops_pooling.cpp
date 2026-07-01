#include "vulkan_ops_common.hpp"
#include "tenzor/backend/attr_macros.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"

namespace tenzor {

// Pooling operations implementation
auto VulkanBackend::dispatchMaxPool2d(const Tensor& input_orig, int64_t kernel_h, int64_t kernel_w,
                                      int64_t stride_h, int64_t stride_w,
                                      int64_t padding_h, int64_t padding_w,
                                      int64_t dilation_h, int64_t dilation_w) -> std::pair<Tensor, Tensor> {
    // The shader gathers input via logical N/C/H/W offsets assuming a contiguous
    // offset-0 layout; materialize first so a permuted/sliced view reads the
    // right elements (mirrors the adaptive-pool entry points).
    const Tensor input = input_orig.contiguous();
    auto input_shape = input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int64_t out_height = (in_height + 2*padding_h - dilation_h*(kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_width = (in_width + 2*padding_w - dilation_w*(kernel_w - 1) - 1) / stride_w + 1;

    int32_t device_id = input.device().index;
    // Select shader based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Float16) {
        shader_name = "pooling_forward_with_indices_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "pooling_forward_with_indices_bf16";
    } else if (input.dtype() == DType::Float64) {
        shader_name = "pooling_forward_with_indices_f64";
    } else if (input.dtype() == DType::Float32) {
        shader_name = "pooling_forward_with_indices";
    } else {
        vulkan_assert_dtype_supported("MaxPool2d", input.dtype(),
            {DType::Float32, DType::Float64, DType::Float16, DType::BFloat16});
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_height, out_width};
    // For Float16 odd-numel outputs, allocate an extra padding half so the
    // pair-packed shader's final whole-word CAS store stays in-bounds; slice back
    // at return. Indices are Int32 (4-byte, unpacked) and need no padding.
    const int64_t logical_numel = batch * channels * out_height * out_width;
    const bool need_fp16_pad = (input.dtype() == DType::Float16) && ((logical_numel & 1) == 1);
    Tensor output = need_fp16_pad
        ? Tensor({logical_numel + 1}, input.dtype(), input.device())
        : Tensor(out_shape, input.dtype(), input.device());
    Tensor indices(out_shape, DType::Int32, input.device());  // Use Int32 for Vulkan

    // Push constants matching pooling_forward_with_indices.comp
    struct PushConstants {
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t dilation_h;
        uint32_t dilation_w;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.dilation_h = static_cast<uint32_t>(dilation_h);
    push_constants.dilation_w = static_cast<uint32_t>(dilation_w);

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_indices = indices.data_ptr();

    // Calculate buffer sizes
    size_t input_size = input.numel() * input.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();
    size_t indices_size = indices.numel() * indices.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output},
        {2, buffer_indices}
    };
    std::vector<size_t> sizes = {input_size, output_size, indices_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch with (16, 16) local size shader.
    //   x = out_width chunks
    //   y = out_height chunks
    //   z = batch * channels (the shader decodes z back into (b, c))
    // The previous code looped over batch on the host and issued
    // `vkCmdDispatch` per batch, but every iteration used the SAME
    // descriptor set bindings (no offset), so iterations 1..N-1 overwrote
    // batch 0's output region and batches 1..N-1 were left uninitialized
    // — silent batch>1 correctness bug. A single packed-z dispatch with
    // the shader doing the (b, c) split is the right model.
    uint32_t workgroups_x = static_cast<uint32_t>((out_width + 15) / 16);
    uint32_t workgroups_y = static_cast<uint32_t>((out_height + 15) / 16);
    uint32_t workgroups_z = static_cast<uint32_t>(batch * channels);
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    if (need_fp16_pad) {
        Tensor sliced = tenzor::slice(output, 0, 0, logical_numel);
        return {sliced.reshape(out_shape).contiguous(), indices};
    }
    return {output, indices};
}

auto VulkanBackend::dispatchAdaptiveMaxPool2d(const Tensor& input, int64_t out_h, int64_t out_w) -> std::pair<Tensor, Tensor> {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int32_t device_id = cont_input.device().index;

    std::string shader_name = "adaptive_pooling";
    if (cont_input.dtype() == DType::Float64) shader_name = "adaptive_pooling_f64";
    else if (cont_input.dtype() == DType::Float16) shader_name = "adaptive_pooling_f16";
    else if (cont_input.dtype() == DType::BFloat16) shader_name = "adaptive_pooling_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_h, out_w};
    // Float16 odd-numel: allocate a padding half so the pair-packed CAS store of
    // the final word stays in-bounds; slice back at return. Indices are Int32.
    const int64_t logical_numel = batch * channels * out_h * out_w;
    const bool need_fp16_pad = (cont_input.dtype() == DType::Float16) && ((logical_numel & 1) == 1);
    Tensor output = need_fp16_pad
        ? Tensor({logical_numel + 1}, cont_input.dtype(), cont_input.device())
        : Tensor(out_shape, cont_input.dtype(), cont_input.device());
    Tensor indices(out_shape, DType::Int32, cont_input.device());

    // Push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t pool_type;  // 0=max, 1=avg
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_h);
    push_constants.in_width = static_cast<uint32_t>(in_w);
    push_constants.out_height = static_cast<uint32_t>(out_h);
    push_constants.out_width = static_cast<uint32_t>(out_w);
    push_constants.pool_type = 0;  // 0 = max pooling

    // Get VkBuffer handles
    const void* buffer_input = cont_input.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_indices = indices.data_ptr();

    // Calculate buffer sizes
    size_t input_size = cont_input.numel() * cont_input.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();
    size_t indices_size = indices.numel() * indices.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output},
        {2, buffer_indices}
    };
    std::vector<size_t> sizes = {input_size, output_size, indices_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch with (16, 16) local size shader.
    //   x = out_width chunks
    //   y = out_height chunks
    //   z = batch * channels (the shader decodes z back into (b, c))
    // The previous code looped over batch on the host but every iteration
    // issued the SAME dispatch with the SAME push constants/bindings, and
    // the shader's index math used only the channel index `c` (no batch
    // stride), so every batch aliased batch 0 and batches 1..N-1 were never
    // computed — a silent batch>1 correctness bug. Packing the batch into z
    // and letting the shader split (b, c) is the correct model (mirrors
    // dispatchMaxPool2d above).
    uint32_t workgroups_x = static_cast<uint32_t>((out_w + 15) / 16);
    uint32_t workgroups_y = static_cast<uint32_t>((out_h + 15) / 16);
    uint32_t workgroups_z = static_cast<uint32_t>(batch * channels);
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    if (need_fp16_pad) {
        Tensor sliced = tenzor::slice(output, 0, 0, logical_numel);
        return {sliced.reshape(out_shape).contiguous(), indices};
    }
    return {output, indices};
}

auto VulkanBackend::dispatchAdaptiveAvgPool2d(const Tensor& input, int64_t out_h, int64_t out_w) -> Tensor {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int32_t device_id = cont_input.device().index;

    // Select shader based on dtype
    std::string shader_name;
    if (cont_input.dtype() == DType::Float64) {
        shader_name = "adaptive_pooling_f64";
    } else if (cont_input.dtype() == DType::Float16) {
        shader_name = "adaptive_pooling_f16";
    } else if (cont_input.dtype() == DType::BFloat16) {
        shader_name = "adaptive_pooling_bf16";
    } else {
        shader_name = "adaptive_pooling";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_h, out_w};
    // Float16 odd-numel: pad with one half so the pair-packed CAS store of the
    // final word stays in-bounds; slice back at return.
    const int64_t logical_numel = batch * channels * out_h * out_w;
    const bool need_fp16_pad = (cont_input.dtype() == DType::Float16) && ((logical_numel & 1) == 1);
    Tensor output = need_fp16_pad
        ? Tensor({logical_numel + 1}, cont_input.dtype(), cont_input.device())
        : Tensor(out_shape, cont_input.dtype(), cont_input.device());

    // Create dummy indices buffer for avg pool (shader requires it)
    Tensor dummy_indices(out_shape, DType::Int32, cont_input.device());

    // Push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t pool_type;  // 0=max, 1=avg
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_h);
    push_constants.in_width = static_cast<uint32_t>(in_w);
    push_constants.out_height = static_cast<uint32_t>(out_h);
    push_constants.out_width = static_cast<uint32_t>(out_w);
    push_constants.pool_type = 1;  // 1 = avg pooling

    // Get VkBuffer handles
    const void* buffer_input = cont_input.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_indices = dummy_indices.data_ptr();

    // Calculate buffer sizes
    size_t input_size = cont_input.numel() * cont_input.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();
    size_t indices_size = dummy_indices.numel() * dummy_indices.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output},
        {2, buffer_indices}
    };
    std::vector<size_t> sizes = {input_size, output_size, indices_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch with (16, 16) local size shader.
    //   x = out_width chunks
    //   y = out_height chunks
    //   z = batch * channels (the shader decodes z back into (b, c))
    // The previous code looped over batch on the host but every iteration
    // issued the SAME dispatch with the SAME push constants/bindings, and
    // the shader's index math used only the channel index `c` (no batch
    // stride), so every batch aliased batch 0 and batches 1..N-1 were never
    // computed — a silent batch>1 correctness bug. Packing the batch into z
    // and letting the shader split (b, c) is the correct model.
    uint32_t workgroups_x = static_cast<uint32_t>((out_w + 15) / 16);
    uint32_t workgroups_y = static_cast<uint32_t>((out_h + 15) / 16);
    uint32_t workgroups_z = static_cast<uint32_t>(batch * channels);
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    if (need_fp16_pad) {
        Tensor sliced = tenzor::slice(output, 0, 0, logical_numel);
        return sliced.reshape(out_shape).contiguous();
    }
    return output;
}

auto VulkanBackend::dispatchAdaptiveAvgPool2dBackward(const Tensor& grad_output, int64_t H_in, int64_t W_in) -> Tensor {
    auto cont_grad = grad_output.contiguous();
    auto grad_shape = cont_grad.shape();
    int64_t batch = grad_shape[0];
    int64_t channels = grad_shape[1];
    int64_t H_out = grad_shape[2];
    int64_t W_out = grad_shape[3];

    int32_t device_id = cont_grad.device().index;

    // Select shader based on dtype
    // Float64, Float16, and BFloat16 versions don't use atomics - they iterate over input positions
    std::string shader_name;
    bool is_float64 = (cont_grad.dtype() == DType::Float64);
    bool is_float16 = (cont_grad.dtype() == DType::Float16);
    bool is_bfloat16 = (cont_grad.dtype() == DType::BFloat16);
    bool needs_input_iteration = is_float64 || is_float16 || is_bfloat16;  // Non-atomic versions
    if (is_float64) {
        shader_name = "adaptive_avg_pool2d_backward_f64";
    } else if (is_float16) {
        shader_name = "adaptive_avg_pool2d_backward_f16";
    } else if (is_bfloat16) {
        shader_name = "adaptive_avg_pool2d_backward_bf16";
    } else {
        shader_name = "adaptive_avg_pool2d_backward";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Output shape: [batch, channels, H_in, W_in]
    std::vector<int64_t> out_shape = {batch, channels, H_in, W_in};
    Tensor grad_input(out_shape, cont_grad.dtype(), cont_grad.device());

    // Zero initialize grad_input (only for Float32 atomic version)
    // Float64 and Float16 versions write directly without atomics, so no need for zero init
    if (!needs_input_iteration) {
        auto* fill_pipeline = getPipeline("fill", device_id);

        struct FillPushConstants {
            uint32_t n_elements;
            uint32_t value_bits;  // float bit representation
        } fill_push_constants;

        fill_push_constants.n_elements = static_cast<uint32_t>(grad_input.numel());
        fill_push_constants.value_bits = 0;  // 0.0f in bits

        const void* buffer_fill = grad_input.data_ptr();
        size_t fill_size = grad_input.numel() * grad_input.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> fill_bindings = {{0, buffer_fill}};
        std::vector<size_t> fill_sizes = {fill_size};

        VkDescriptorSet fillDescSet = allocateAndWriteDescriptorSet(
            device_id, fill_pipeline, fill_bindings, fill_sizes);

        VkCommandBuffer fillCmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(fillCmd, VK_PIPELINE_BIND_POINT_COMPUTE, fill_pipeline->pipeline());
        vkCmdBindDescriptorSets(fillCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               fill_pipeline->layout(), 0, 1, &fillDescSet, 0, nullptr);
        vkCmdPushConstants(fillCmd, fill_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(FillPushConstants), &fill_push_constants);
        uint32_t fill_workgroups = div_wg(fill_push_constants.n_elements, devices_[device_id].workgroupSize);
        vkCmdDispatch(fillCmd, fill_workgroups, 1, 1);

        VkMemoryBarrier fillBarrier{};
        fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        fillBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(fillCmd,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &fillBarrier, 0, nullptr, 0, nullptr);
        endSingleTimeCommands(fillCmd, device_id);
    }

    // Push constants for backward
    struct PushConstants {
        uint32_t batch;
        uint32_t channels;
        uint32_t H_in;
        uint32_t W_in;
        uint32_t H_out;
        uint32_t W_out;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.H_in = static_cast<uint32_t>(H_in);
    push_constants.W_in = static_cast<uint32_t>(W_in);
    push_constants.H_out = static_cast<uint32_t>(H_out);
    push_constants.W_out = static_cast<uint32_t>(W_out);

    // Get VkBuffer handles
    const void* buffer_grad_output = cont_grad.data_ptr();
    const void* buffer_grad_input = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t grad_output_size = cont_grad.numel() * cont_grad.dtype_size();
    size_t grad_input_size = grad_input.numel() * grad_input.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_output},
        {1, buffer_grad_input}
    };
    std::vector<size_t> sizes = {grad_output_size, grad_input_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch
    // Float64/Float16 versions iterate over INPUT elements (each thread handles one input position)
    // Float32 version iterates over OUTPUT elements (uses atomics to accumulate)
    uint32_t total_elements;
    if (needs_input_iteration) {
        total_elements = static_cast<uint32_t>(batch * channels * H_in * W_in);
    } else {
        total_elements = static_cast<uint32_t>(batch * channels * H_out * W_out);
    }
    uint32_t workgroups = div_wg(total_elements, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// NOTE: the old 9-argument `dispatchMaxPool2dBackward` overload (which
// allocated grad_input without zeroing, bound no descriptor set / push
// constants, and dispatched a pipeline with no bound buffers — returning
// uninitialized gradients) has been removed. The registry uses the working
// OpAttributes-based overload (dispatchMaxPool2dBackward(grad, input, attrs)).

// ============================================================================
// Pooling Operations Implementation (OpAttributes versions)
// ============================================================================

auto VulkanBackend::dispatchAvgPool2dForward(const Tensor& input_orig, const OpAttributes& attrs) -> Tensor {
    // Materialize to a contiguous offset-0 layout; the shader gathers input via
    // logical N/C/H/W offsets (mirrors the adaptive-pool entry points).
    const Tensor input = input_orig.contiguous();
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("avg_pool2d requires 4D input (N, C, H, W)");
    }

    // Extract attributes (support both split H/W and single-value forms)
    int64_t kernel_h = attrs.has(AttrKey::KernelSizeH) ? attrs.get_int(AttrKey::KernelSizeH) : attrs.get_int(AttrKey::KernelSize, 2);
    int64_t kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : kernel_h;
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : attrs.get_int(AttrKey::Stride, kernel_h);
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : stride_h;
    int64_t padding_h = attrs.has(AttrKey::PaddingH) ? attrs.get_int(AttrKey::PaddingH) : attrs.get_int(AttrKey::Padding, 0);
    int64_t padding_w = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : padding_h;
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1);  // include-pad default (matches CPU/other backends + AvgPool1d/3d)

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    // Calculate output dimensions
    int64_t out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "avg_pool2d";
    if (input.dtype() == DType::Float64) {
        shader_name = "avg_pool2d_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "avg_pool2d_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "avg_pool2d_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor. For Float16 odd-numel outputs, allocate an
    // extra padding F16 so the pair-packed shader's final whole-word store
    // is in-bounds; slice back to the logical shape at return.
    std::vector<int64_t> output_shape = {batch, channels, out_height, out_width};
    const int64_t logical_numel = batch * channels * out_height * out_width;
    const bool need_fp16_pad = (input.dtype() == DType::Float16) && ((logical_numel & 1) == 1);
    Tensor output = need_fp16_pad
        ? Tensor({logical_numel + 1}, input.dtype(), input.device())
        : Tensor(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output}
    };
    std::vector<size_t> sizes = {buffer_size_input, buffer_size_output};

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
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t count_include_pad;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(logical_numel);
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups. Float16 shader packs two outputs per thread,
    // so the thread count is halved (ceil so the final odd element still
    // gets a thread).
    uint32_t threads = static_cast<uint32_t>(logical_numel);
    if (input.dtype() == DType::Float16) {
        threads = (static_cast<uint32_t>(logical_numel) + 1) / 2;
    }
    uint32_t workgroups = static_cast<uint32_t>(div_wg(threads, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    // Slice off the padding element for odd-numel F16 outputs and reshape
    // back to the 4D logical shape.
    if (need_fp16_pad) {
        Tensor sliced = tenzor::slice(output, 0, 0, logical_numel);
        return sliced.reshape(output_shape).contiguous();
    }
    return output;
}

auto VulkanBackend::dispatchMaxPool2dForward(const Tensor& input_orig, const OpAttributes& attrs) -> Tensor {
    // The shader gathers input via logical N/C/H/W offsets assuming a contiguous
    // offset-0 layout; materialize so a permuted/sliced view reads the right
    // elements (mirrors the adaptive-pool entry points).
    const Tensor input = input_orig.contiguous();
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("max_pool2d requires 4D input (N, C, H, W)");
    }

    // Extract attributes. Accept scalar KernelSize/Stride/Padding fallbacks
    // set by the MaxPool2d Module.
    int64_t ks = attrs.get_int(AttrKey::KernelSize, 0);
    int64_t st = attrs.get_int(AttrKey::Stride, ks);
    int64_t pd = attrs.get_int(AttrKey::Padding, 0);
    int64_t dl = attrs.get_int(AttrKey::Dilation, 1);
    int64_t kernel_h = attrs.has(AttrKey::KernelSizeH) ? attrs.get_int(AttrKey::KernelSizeH) : ks;
    int64_t kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : ks;
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : (st > 0 ? st : kernel_h);
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : (st > 0 ? st : kernel_w);
    int64_t padding_h = attrs.has(AttrKey::PaddingH) ? attrs.get_int(AttrKey::PaddingH) : pd;
    int64_t padding_w = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : pd;
    int64_t dilation_h = attrs.has(AttrKey::DilationH) ? attrs.get_int(AttrKey::DilationH) : dl;
    int64_t dilation_w = attrs.has(AttrKey::DilationW) ? attrs.get_int(AttrKey::DilationW) : dl;
    if (kernel_h <= 0 || kernel_w <= 0 || stride_h <= 0 || stride_w <= 0) {
        throw std::invalid_argument("Vulkan max_pool2d: kernel_size and stride must be positive");
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    // Calculate output dimensions with dilation
    int64_t effective_kernel_h = (kernel_h - 1) * dilation_h + 1;
    int64_t effective_kernel_w = (kernel_w - 1) * dilation_w + 1;
    int64_t out_height = (in_height + 2 * padding_h - effective_kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2 * padding_w - effective_kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "max_pool2d";
    if (input.dtype() == DType::Float64) {
        shader_name = "max_pool2d_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "max_pool2d_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "max_pool2d_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor. For Float16 odd-numel outputs, allocate an extra
    // padding half so the pair-packed shader's final whole-word CAS store stays
    // in-bounds; slice back at return.
    std::vector<int64_t> output_shape = {batch, channels, out_height, out_width};
    const int64_t logical_numel = batch * channels * out_height * out_width;
    const bool need_fp16_pad = (input.dtype() == DType::Float16) && ((logical_numel & 1) == 1);
    Tensor output = need_fp16_pad
        ? Tensor({logical_numel + 1}, input.dtype(), input.device())
        : Tensor(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output}
    };
    std::vector<size_t> sizes = {buffer_size_input, buffer_size_output};

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
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t dilation_h;
        uint32_t dilation_w;
    } push_constants;

    // Use the logical element count, not the (possibly +1 padded) allocation, so
    // the shader doesn't process a phantom output position.
    push_constants.n_elements = static_cast<uint32_t>(logical_numel);
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.dilation_h = static_cast<uint32_t>(dilation_h);
    push_constants.dilation_w = static_cast<uint32_t>(dilation_w);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>(div_wg(logical_numel, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    if (need_fp16_pad) {
        Tensor sliced = tenzor::slice(output, 0, 0, logical_numel);
        return sliced.reshape(output_shape).contiguous();
    }
    return output;
}

auto VulkanBackend::dispatchAvgPool2dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("avg_pool2d_backward requires 4D input (N, C, H, W)");
    }

    // Extract attributes (support both split H/W and single-value forms)
    int64_t kernel_h = attrs.has(AttrKey::KernelSizeH) ? attrs.get_int(AttrKey::KernelSizeH) : attrs.get_int(AttrKey::KernelSize, 2);
    int64_t kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : kernel_h;
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : attrs.get_int(AttrKey::Stride, kernel_h);
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : stride_h;
    int64_t padding_h = attrs.has(AttrKey::PaddingH) ? attrs.get_int(AttrKey::PaddingH) : attrs.get_int(AttrKey::Padding, 0);
    int64_t padding_w = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : padding_h;
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1);  // include-pad default (matches CPU/other backends + AvgPool1d/3d)

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    auto grad_out_shape = grad_output.shape();
    int64_t out_height = grad_out_shape[2];
    int64_t out_width = grad_out_shape[3];

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "avg_pool2d_backward";
    if (input.dtype() == DType::Float64) {
        shader_name = "avg_pool2d_backward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "avg_pool2d_backward_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "avg_pool2d_backward_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient input tensor (same shape as input), initialized to zero
    std::vector<int64_t> grad_input_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_input_shape, input.dtype(), input.device());

    // Zero initialize grad_input using fill operation
    grad_input = dispatchFill(grad_input, 0.0f);

    // Get VkBuffer handles
    const void* buffer_grad_output = grad_output.data_ptr();
    const void* buffer_grad_input = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_grad_output = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_grad_input = grad_input.numel() * grad_input.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_output},
        {1, buffer_grad_input}
    };
    std::vector<size_t> sizes = {buffer_size_grad_output, buffer_size_grad_input};

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
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t count_include_pad;
    } push_constants;

    // For f64 shader: input-centric gather (one thread per input element)
    // For other shaders: output-centric scatter (one thread per output element)
    bool is_input_centric = (input.dtype() == DType::Float64);
    push_constants.n_elements = static_cast<uint32_t>(is_input_centric ? input.numel() : grad_output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch: f64 uses input-centric gather (over input elements), others use output-centric scatter
    int64_t dispatch_count = is_input_centric ? input.numel() : grad_output.numel();
    uint32_t workgroups = static_cast<uint32_t>(div_wg(dispatch_count, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchMaxPool2dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("max_pool2d_backward requires 4D input (N, C, H, W)");
    }

    // Extract attributes (support both split H/W and single-value forms)
    int64_t kernel_h = attrs.has(AttrKey::KernelSizeH) ? attrs.get_int(AttrKey::KernelSizeH) : attrs.get_int(AttrKey::KernelSize, 2);
    int64_t kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : kernel_h;
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : attrs.get_int(AttrKey::Stride, kernel_h);
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : stride_h;
    int64_t padding_h = attrs.has(AttrKey::PaddingH) ? attrs.get_int(AttrKey::PaddingH) : attrs.get_int(AttrKey::Padding, 0);
    int64_t padding_w = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : padding_h;
    int64_t dilation_h = attrs.get_int(AttrKey::DilationH, 1);
    int64_t dilation_w = attrs.get_int(AttrKey::DilationW, 1);

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    auto grad_out_shape = grad_output.shape();
    int64_t out_height = grad_out_shape[2];
    int64_t out_width = grad_out_shape[3];

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    if (is_f64) {
        // Y.10: max_pool2d_backward_f64 uses GL_EXT_shader_atomic_int64; gate.
        vulkan::ensure_atomic_int64_supported(device_id, "MaxPool2dBackward");
    }
    std::string mp_shader = is_f64 ? "max_pool2d_backward_f64" :
                            (is_f16 ? "max_pool2d_backward_f16" :
                            (is_bf16 ? "max_pool2d_backward_bf16" : "max_pool2d_backward"));
    auto* pipeline = getPipeline(mp_shader, device_id);

    // Create gradient input tensor (same shape as input), initialized to zero
    std::vector<int64_t> grad_input_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_input_shape, input.dtype(), input.device());

    // Zero initialize grad_input using fill operation
    grad_input = dispatchFill(grad_input, 0.0f);

    // Get VkBuffer handles
    const void* buffer_grad_output = grad_output.data_ptr();
    const void* buffer_input = input.data_ptr();
    const void* buffer_grad_input = grad_input.data_ptr();

    // Calculate buffer sizes. The F16 AND BF16 shaders bind packed buffers (2
    // halves per 32-bit word) for grad_output, input AND grad_input
    // (CAS-accumulated), reading/writing whole words up to index (numel-1)/2.
    // Round each range up to ((numel+1)/2)*4 so an odd-numel tail word stays
    // in-bounds.
    const bool packed16 = is_f16 || is_bf16;
    auto packed_bytes = [packed16](int64_t numel, size_t dtype_size) -> size_t {
        if (packed16) return ((static_cast<size_t>(numel) + 1) / 2) * sizeof(uint32_t);
        return static_cast<size_t>(numel) * dtype_size;
    };
    size_t buffer_size_grad_output = packed_bytes(grad_output.numel(), grad_output.dtype_size());
    size_t buffer_size_input = packed_bytes(input.numel(), input.dtype_size());
    size_t buffer_size_grad_input = packed_bytes(grad_input.numel(), grad_input.dtype_size());

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_output},
        {1, buffer_input},
        {2, buffer_grad_input}
    };
    std::vector<size_t> sizes = {buffer_size_grad_output, buffer_size_input, buffer_size_grad_input};

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
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t dilation_h;
        uint32_t dilation_w;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(grad_output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.dilation_h = static_cast<uint32_t>(dilation_h);
    push_constants.dilation_w = static_cast<uint32_t>(dilation_w);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (iterate over grad_output elements)
    uint32_t workgroups = static_cast<uint32_t>(div_wg(grad_output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// ============================================================================
// MaxPool2d Backward with Indices (scatter-based with GPU atomicAdd)
// ============================================================================

auto VulkanBackend::dispatchMaxPool2dBackwardWithIndices(const Tensor& grad_output, const Tensor& indices,
                                                          int64_t H_in, int64_t W_in) -> Tensor {
    // Use GPU kernel with atomicAdd for scatter operation
    auto grad_out_shape = grad_output.shape();
    if (grad_out_shape.size() != 4) {
        throw std::invalid_argument("max_pool2d_backward requires 4D grad_output (N, C, H_out, W_out)");
    }

    int64_t N = grad_out_shape[0];
    int64_t C = grad_out_shape[1];
    int64_t grad_out_numel = grad_output.numel();
    int64_t grad_in_numel = N * C * H_in * W_in;

    if (grad_out_numel == 0) {
        return Tensor({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());
    }

    int32_t device_id = grad_output.device().index;

    // Create grad_input tensor initialized to zeros
    Tensor grad_input = dispatchZeros({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());

    // Select shader based on dtype
    std::string shader_name = "max_pool2d_backward_indices";
    if (grad_output.dtype() == DType::Float64) {
        // The F64 shader accumulates via uint64 CAS (GL_EXT_shader_atomic_int64);
        // fail fast with a clear error on devices lacking the extension, matching
        // the non-indices F64 maxpool path.
        vulkan::ensure_atomic_int64_supported(device_id, "MaxPool2dBackward");
        shader_name = "max_pool2d_backward_indices_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "max_pool2d_backward_indices_f16";
    } else if (grad_output.dtype() != DType::Float32) {
        throw std::runtime_error("Unsupported dtype for max_pool2d_backward_with_indices");
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    const void* buffer_grad_out = const_cast<void*>(grad_output.data_ptr());
    const void* buffer_indices = const_cast<void*>(indices.data_ptr());
    const void* buffer_grad_in = grad_input.data_ptr();

    // F16 binds packed-F16 buffers (grad_out read, grad_in CAS-accumulated);
    // round those ranges up to whole 32-bit words so an odd-numel tail word stays
    // in-bounds. Indices are Int32, unaffected.
    const bool idx_is_f16 = (grad_output.dtype() == DType::Float16);
    size_t grad_out_size = idx_is_f16
        ? ((static_cast<size_t>(grad_out_numel) + 1) / 2) * sizeof(uint32_t)
        : grad_out_numel * grad_output.dtype_size();
    size_t indices_size = indices.numel() * indices.dtype_size();
    size_t grad_in_size = idx_is_f16
        ? ((static_cast<size_t>(grad_in_numel) + 1) / 2) * sizeof(uint32_t)
        : grad_in_numel * grad_input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_indices},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {grad_out_size, indices_size, grad_in_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t n_elements;
        uint32_t grad_input_size;
        uint32_t out_plane;   // H_out * W_out
        uint32_t in_plane;    // H_in * W_in
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(grad_out_numel);
    push_constants.grad_input_size = static_cast<uint32_t>(grad_in_numel);
    push_constants.out_plane = static_cast<uint32_t>(grad_out_shape[2] * grad_out_shape[3]);
    push_constants.in_plane = static_cast<uint32_t>(H_in * W_in);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(grad_out_numel, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// ============================================================================
// 1D Pooling Operations
// ============================================================================

auto VulkanBackend::dispatchMaxPool1dForward(const Tensor& input, const OpAttributes& attrs) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("max_pool1d requires 3D input (N, C, L)");
    }

    int64_t kernel_size = attrs.get_int(AttrKey::KernelSize);
    int64_t stride = attrs.has(AttrKey::Stride) ? attrs.get_int(AttrKey::Stride) : kernel_size;
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);

    // PyTorch-style validation. Reject configurations that could otherwise
    // produce an all-padding pooling window (shader writes -inf and index 0).
    if (kernel_size <= 0 || stride <= 0 || dilation <= 0) {
        throw std::invalid_argument("Vulkan max_pool1d: kernel_size, stride and dilation must be positive");
    }
    if (padding < 0 || padding > kernel_size / 2) {
        throw std::invalid_argument("max_pool1d: padding must be <= kernel_size/2");
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_length = input_shape[2];

    int64_t effective_kernel = (kernel_size - 1) * dilation + 1;
    int64_t out_length = (in_length + 2 * padding - effective_kernel) / stride + 1;

    int32_t device_id = input.device().index;

    std::string shader_name = "max_pool1d";
    if (input.dtype() == DType::Float64) {
        shader_name = "max_pool1d_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "max_pool1d_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "max_pool1d_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> output_shape = {batch, channels, out_length};
    Tensor output(output_shape, input.dtype(), input.device());
    Tensor indices(output_shape, DType::Int32, input.device());

    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_indices = indices.data_ptr();

    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();
    size_t buffer_size_indices = indices.numel() * indices.dtype_size();

    if (input.dtype() == DType::Float16) {
        buffer_size_input = ((input.numel() + 1) / 2) * 4;
        buffer_size_output = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input}, {1, buffer_output}, {2, buffer_indices}
    };
    std::vector<size_t> sizes = {buffer_size_input, buffer_size_output, buffer_size_indices};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch, channels, in_length, out_length;
        uint32_t kernel_size, stride, padding, dilation;
    } pc;

    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch = static_cast<uint32_t>(batch);
    pc.channels = static_cast<uint32_t>(channels);
    pc.in_length = static_cast<uint32_t>(in_length);
    pc.out_length = static_cast<uint32_t>(out_length);
    pc.kernel_size = static_cast<uint32_t>(kernel_size);
    pc.stride = static_cast<uint32_t>(stride);
    pc.padding = static_cast<uint32_t>(padding);
    pc.dilation = static_cast<uint32_t>(dilation);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchMaxPool1dBackward(const Tensor& grad_output, const Tensor& indices,
                                               int64_t L_in) -> Tensor {
    // F16/BF16: the packed-word backward shaders bind half-width (numel*2)
    // buffers and overrun by a 32-bit word / drop the tail gradient for odd
    // element counts. Compute the scatter in Float32 on the GPU and narrow back
    // (stays on the device — not a CPU fallback).
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        const DType orig = grad_output.dtype();
        return dispatchMaxPool1dBackward(grad_output.to(DType::Float32), indices, L_in)
            .to(orig);
    }
    auto grad_out_shape = grad_output.shape();
    int64_t N = grad_out_shape[0];
    int64_t C = grad_out_shape[1];
    int64_t grad_out_numel = grad_output.numel();
    int64_t grad_in_numel = N * C * L_in;

    if (grad_out_numel == 0) {
        return Tensor({N, C, L_in}, grad_output.dtype(), grad_output.device());
    }

    int32_t device_id = grad_output.device().index;
    if (grad_output.dtype() == DType::Float64) {
        // Y.10: max_pool1d_backward_f64 uses GL_EXT_shader_atomic_int64; gate.
        vulkan::ensure_atomic_int64_supported(device_id, "MaxPool1dBackward");
    }
    Tensor grad_input = dispatchZeros({N, C, L_in}, grad_output.dtype(), grad_output.device());

    std::string shader_name = "max_pool1d_backward";
    if (grad_output.dtype() == DType::Float64) shader_name = "max_pool1d_backward_f64";
    else if (grad_output.dtype() == DType::Float16) shader_name = "max_pool1d_backward_f16";
    else if (grad_output.dtype() == DType::BFloat16) shader_name = "max_pool1d_backward_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()}, {1, indices.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(grad_out_numel * grad_output.dtype_size()),
        static_cast<size_t>(indices.numel() * indices.dtype_size()),
        static_cast<size_t>(grad_in_numel * grad_input.dtype_size())
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants { uint32_t n_elements, grad_input_size; } pc;
    pc.n_elements = static_cast<uint32_t>(grad_out_numel);
    pc.grad_input_size = static_cast<uint32_t>(grad_in_numel);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmdBuffer, div_wg(grad_out_numel, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchAvgPool1dForward(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("avg_pool1d requires 3D input (N, C, L)");
    }

    int64_t kernel_size = attrs.get_int(AttrKey::KernelSize);
    int64_t stride = attrs.has(AttrKey::Stride) ? attrs.get_int(AttrKey::Stride) : kernel_size;
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1);

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_length = input_shape[2];
    int64_t out_length = (in_length + 2 * padding - kernel_size) / stride + 1;

    int32_t device_id = input.device().index;

    std::string shader_name = "avg_pool1d";
    if (input.dtype() == DType::Float64) shader_name = "avg_pool1d_f64";
    else if (input.dtype() == DType::Float16) shader_name = "avg_pool1d_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "avg_pool1d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> output_shape = {batch, channels, out_length};
    Tensor output(output_shape, input.dtype(), input.device());

    size_t buf_in = input.numel() * input.dtype_size();
    size_t buf_out = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16) {
        buf_in = ((input.numel() + 1) / 2) * 4;
        buf_out = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, input.data_ptr()}, {1, output.data_ptr()}};
    std::vector<size_t> sizes = {buf_in, buf_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch, channels, in_length, out_length;
        uint32_t kernel_size, stride, padding, count_include_pad;
    } pc;

    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch = static_cast<uint32_t>(batch);
    pc.channels = static_cast<uint32_t>(channels);
    pc.in_length = static_cast<uint32_t>(in_length);
    pc.out_length = static_cast<uint32_t>(out_length);
    pc.kernel_size = static_cast<uint32_t>(kernel_size);
    pc.stride = static_cast<uint32_t>(stride);
    pc.padding = static_cast<uint32_t>(padding);
    pc.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAvgPool1dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    int64_t kernel_size = attrs.get_int(AttrKey::KernelSize);
    int64_t stride = attrs.has(AttrKey::Stride) ? attrs.get_int(AttrKey::Stride) : kernel_size;
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1);

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_length = input_shape[2];
    int64_t out_length = grad_output.shape()[2];

    int32_t device_id = input.device().index;

    std::string shader_name = "avg_pool1d_backward";
    if (input.dtype() == DType::Float64) {
        shader_name = "avg_pool1d_backward_f64";
    } else if (input.dtype() == DType::Float16) shader_name = "avg_pool1d_backward_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "avg_pool1d_backward_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> grad_input_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_input_shape, input.dtype(), input.device());
    grad_input = dispatchFill(grad_input, 0.0f);

    size_t buf_go = grad_output.numel() * grad_output.dtype_size();
    size_t buf_gi = grad_input.numel() * grad_input.dtype_size();
    if (input.dtype() == DType::Float16) {
        buf_go = ((grad_output.numel() + 1) / 2) * 4;
        buf_gi = ((grad_input.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, grad_output.data_ptr()}, {1, grad_input.data_ptr()}};
    std::vector<size_t> sizes = {buf_go, buf_gi};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch, channels, in_length, out_length;
        uint32_t kernel_size, stride, padding, count_include_pad;
    } pc;

    // For f64 shader: input-centric gather (one thread per input element)
    // For other shaders: output-centric scatter (one thread per output element)
    bool is_input_centric_1d = (input.dtype() == DType::Float64);
    pc.n_elements = static_cast<uint32_t>(is_input_centric_1d ? input.numel() : grad_output.numel());
    pc.batch = static_cast<uint32_t>(batch);
    pc.channels = static_cast<uint32_t>(channels);
    pc.in_length = static_cast<uint32_t>(in_length);
    pc.out_length = static_cast<uint32_t>(out_length);
    pc.kernel_size = static_cast<uint32_t>(kernel_size);
    pc.stride = static_cast<uint32_t>(stride);
    pc.padding = static_cast<uint32_t>(padding);
    pc.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    int64_t dispatch_count_1d = is_input_centric_1d ? input.numel() : grad_output.numel();
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(dispatch_count_1d, devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchAdaptiveMaxPool1d(const Tensor& input, int64_t output_size) -> std::pair<Tensor, Tensor> {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0], channels = input_shape[1], in_length = input_shape[2];
    int32_t device_id = cont_input.device().index;

    std::string shader_name = "adaptive_pooling_1d";
    if (cont_input.dtype() == DType::Float64) shader_name = "adaptive_pooling_1d_f64";
    else if (cont_input.dtype() == DType::Float16) shader_name = "adaptive_pooling_1d_f16";
    else if (cont_input.dtype() == DType::BFloat16) shader_name = "adaptive_pooling_1d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, output_size};
    Tensor output(out_shape, cont_input.dtype(), cont_input.device());
    Tensor indices(out_shape, DType::Int32, cont_input.device());

    size_t in_sz = cont_input.numel() * cont_input.dtype_size();
    size_t out_sz = output.numel() * output.dtype_size();
    size_t idx_sz = indices.numel() * indices.dtype_size();
    if (cont_input.dtype() == DType::Float16) {
        in_sz = ((cont_input.numel() + 1) / 2) * 4;
        out_sz = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, cont_input.data_ptr()}, {1, output.data_ptr()}, {2, indices.data_ptr()}
    };
    std::vector<size_t> sizes = {in_sz, out_sz, idx_sz};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch, channels, in_length, out_length, pool_type;
    } pc;
    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch = static_cast<uint32_t>(batch);
    pc.channels = static_cast<uint32_t>(channels);
    pc.in_length = static_cast<uint32_t>(in_length);
    pc.out_length = static_cast<uint32_t>(output_size);
    pc.pool_type = 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchAdaptiveAvgPool1d(const Tensor& input, int64_t output_size) -> Tensor {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0], channels = input_shape[1], in_length = input_shape[2];
    int32_t device_id = cont_input.device().index;

    std::string shader_name = "adaptive_pooling_1d";
    if (cont_input.dtype() == DType::Float64) shader_name = "adaptive_pooling_1d_f64";
    else if (cont_input.dtype() == DType::Float16) shader_name = "adaptive_pooling_1d_f16";
    else if (cont_input.dtype() == DType::BFloat16) shader_name = "adaptive_pooling_1d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, output_size};
    Tensor output(out_shape, cont_input.dtype(), cont_input.device());
    Tensor dummy_indices(out_shape, DType::Int32, cont_input.device());

    size_t in_sz = cont_input.numel() * cont_input.dtype_size();
    size_t out_sz = output.numel() * output.dtype_size();
    size_t idx_sz = dummy_indices.numel() * dummy_indices.dtype_size();
    if (cont_input.dtype() == DType::Float16) {
        in_sz = ((cont_input.numel() + 1) / 2) * 4;
        out_sz = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, cont_input.data_ptr()}, {1, output.data_ptr()}, {2, dummy_indices.data_ptr()}
    };
    std::vector<size_t> sizes = {in_sz, out_sz, idx_sz};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch, channels, in_length, out_length, pool_type;
    } pc;
    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch = static_cast<uint32_t>(batch);
    pc.channels = static_cast<uint32_t>(channels);
    pc.in_length = static_cast<uint32_t>(in_length);
    pc.out_length = static_cast<uint32_t>(output_size);
    pc.pool_type = 1;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAdaptiveMaxPool1dBackward(const Tensor& grad_output, const Tensor& indices,
                                                        const std::vector<int64_t>& input_shape) -> Tensor {
    // F16/BF16: the packed-word backward shaders bind half-width buffers and
    // overrun / drop the tail gradient for odd element counts. Compute in
    // Float32 on the GPU and narrow back (stays on the device).
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        const DType orig = grad_output.dtype();
        return dispatchAdaptiveMaxPool1dBackward(grad_output.to(DType::Float32), indices,
                                                 input_shape).to(orig);
    }
    int64_t grad_out_numel = grad_output.numel();
    int64_t grad_in_numel = 1;
    for (auto s : input_shape) grad_in_numel *= s;
    if (grad_out_numel == 0) return Tensor(input_shape, grad_output.dtype(), grad_output.device());

    int32_t device_id = grad_output.device().index;
    if (grad_output.dtype() == DType::Float64) {
        // Y.10: max_pool1d_backward_f64 uses GL_EXT_shader_atomic_int64; gate.
        vulkan::ensure_atomic_int64_supported(device_id, "AdaptiveMaxPool1dBackward");
    }
    Tensor grad_input = dispatchZeros(input_shape, grad_output.dtype(), grad_output.device());

    std::string shader_name = "max_pool1d_backward";
    if (grad_output.dtype() == DType::Float64) shader_name = "max_pool1d_backward_f64";
    else if (grad_output.dtype() == DType::Float16) shader_name = "max_pool1d_backward_f16";
    else if (grad_output.dtype() == DType::BFloat16) shader_name = "max_pool1d_backward_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()}, {1, indices.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(grad_out_numel * grad_output.dtype_size()),
        static_cast<size_t>(indices.numel() * indices.dtype_size()),
        static_cast<size_t>(grad_in_numel * grad_input.dtype_size())
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants { uint32_t n_elements, grad_input_size; } pc;
    pc.n_elements = static_cast<uint32_t>(grad_out_numel);
    pc.grad_input_size = static_cast<uint32_t>(grad_in_numel);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, div_wg(grad_out_numel, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchAdaptiveAvgPool1dBackward(const Tensor& grad_output, int64_t L_in) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t batch = grad_shape[0], channels = grad_shape[1], L_out = grad_shape[2];
    std::vector<int64_t> out_shape = {batch, channels, L_in};
    Tensor grad_input = dispatchZeros(out_shape, grad_output.dtype(), grad_output.device());

    if (L_in % L_out == 0) {
        int64_t k = L_in / L_out;
        OpAttributes backward_attrs;
        backward_attrs.set(AttrKey::KernelSize, k);
        backward_attrs.set(AttrKey::Stride, k);
        backward_attrs.set(AttrKey::Padding, static_cast<int64_t>(0));
        backward_attrs.set(AttrKey::CountIncludePad, static_cast<int64_t>(1));
        Tensor dummy_input(out_shape, grad_output.dtype(), grad_output.device());
        return dispatchAvgPool1dBackward(grad_output, dummy_input, backward_attrs);
    }

    // Non-divisible case: delegate to adaptive_avg_pool2d_backward by treating
    // the 1D problem as 2D with H=1. The 2D shader computes per-element adaptive
    // window boundaries (start = j * L_in / L_out, end = (j+1) * L_in / L_out)
    // and distributes grad_output[j] / window_size to each input in the window.
    Tensor grad_output_4d = grad_output.reshape({batch, channels, 1, L_out});
    Tensor grad_input_4d = dispatchAdaptiveAvgPool2dBackward(grad_output_4d, /*H_in=*/1, /*W_in=*/L_in);
    return grad_input_4d.reshape({batch, channels, L_in});
}

// ============================================================================
// 3D Pooling Operations
// ============================================================================

auto VulkanBackend::dispatchMaxPool3dForward(const Tensor& input, const OpAttributes& attrs) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("max_pool3d requires 5D input (N, C, D, H, W)");
    }

    // Module code (src/nn/layers/pooling.cpp) sets scalar AttrKey::KernelSize /
    // Stride / Padding. Accept those as fallbacks if per-dim keys are absent,
    // otherwise kernel_d/h/w default to 0 and the output-shape math below
    // does a division by zero — which was causing GPU hangs.
    int64_t ks = attrs.get_int(AttrKey::KernelSize, 0);
    int64_t st = attrs.get_int(AttrKey::Stride, ks);
    int64_t pd = attrs.get_int(AttrKey::Padding, 0);
    int64_t kernel_d = attrs.has(AttrKey::KernelSizeD) ? attrs.get_int(AttrKey::KernelSizeD) : ks;
    int64_t kernel_h = attrs.has(AttrKey::KernelSizeH) ? attrs.get_int(AttrKey::KernelSizeH) : ks;
    int64_t kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : ks;
    int64_t stride_d = attrs.has(AttrKey::StrideD) ? attrs.get_int(AttrKey::StrideD) : (st > 0 ? st : kernel_d);
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : (st > 0 ? st : kernel_h);
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : (st > 0 ? st : kernel_w);
    int64_t padding_d = attrs.has(AttrKey::PaddingD) ? attrs.get_int(AttrKey::PaddingD) : pd;
    int64_t padding_h = attrs.has(AttrKey::PaddingH) ? attrs.get_int(AttrKey::PaddingH) : pd;
    int64_t padding_w = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : pd;
    int64_t dil = attrs.get_int(AttrKey::Dilation, 1);
    int64_t dilation_d = attrs.has(AttrKey::DilationD) ? attrs.get_int(AttrKey::DilationD) : dil;
    int64_t dilation_h = attrs.has(AttrKey::DilationH) ? attrs.get_int(AttrKey::DilationH) : dil;
    int64_t dilation_w = attrs.has(AttrKey::DilationW) ? attrs.get_int(AttrKey::DilationW) : dil;
    bool ceil_mode = attrs.get_int(AttrKey::CeilMode, 0) != 0;
    if (kernel_d <= 0 || kernel_h <= 0 || kernel_w <= 0 ||
        stride_d <= 0 || stride_h <= 0 || stride_w <= 0 ||
        dilation_d <= 0 || dilation_h <= 0 || dilation_w <= 0) {
        throw std::invalid_argument(
            "Vulkan max_pool3d: kernel_size, stride, and dilation must be positive "
            "(got kernel=" + std::to_string(kernel_d) + "x" +
            std::to_string(kernel_h) + "x" + std::to_string(kernel_w) +
            ", stride=" + std::to_string(stride_d) + "x" +
            std::to_string(stride_h) + "x" + std::to_string(stride_w) +
            ", dilation=" + std::to_string(dilation_d) + "x" +
            std::to_string(dilation_h) + "x" + std::to_string(dilation_w) + ")");
    }

    int64_t batch = input_shape[0], channels = input_shape[1];
    int64_t in_depth = input_shape[2], in_height = input_shape[3], in_width = input_shape[4];

    // PyTorch out shape: floor((in + 2*pad - dilation*(kernel-1) - 1) / stride) + 1
    // ceil_mode bumps to ceil(...).
    auto compute_out = [&](int64_t in, int64_t pad, int64_t dil_v, int64_t k, int64_t s) -> int64_t {
        int64_t num = in + 2 * pad - dil_v * (k - 1) - 1;
        if (ceil_mode) {
            // ceil(num / s) = (num + s - 1) / s for positive num, but PyTorch
            // also guards the last window to ensure it starts within input.
            int64_t out = (num + s - 1) / s + 1;
            if ((out - 1) * s >= in + pad) out -= 1;
            return out;
        }
        return num / s + 1;
    };
    int64_t out_depth  = compute_out(in_depth,  padding_d, dilation_d, kernel_d, stride_d);
    int64_t out_height = compute_out(in_height, padding_h, dilation_h, kernel_h, stride_h);
    int64_t out_width  = compute_out(in_width,  padding_w, dilation_w, kernel_w, stride_w);

    int32_t device_id = input.device().index;

    std::string shader_name = "max_pool3d";
    if (input.dtype() == DType::Float64) shader_name = "max_pool3d_f64";
    else if (input.dtype() == DType::Float16) shader_name = "max_pool3d_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "max_pool3d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> output_shape = {batch, channels, out_depth, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());
    Tensor indices(output_shape, DType::Int32, input.device());

    size_t buf_in = input.numel() * input.dtype_size();
    size_t buf_out = output.numel() * output.dtype_size();
    size_t buf_idx = indices.numel() * indices.dtype_size();
    if (input.dtype() == DType::Float16) {
        buf_in = ((input.numel() + 1) / 2) * 4;
        buf_out = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}, {2, indices.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_in, buf_out, buf_idx};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch_channels;
        uint32_t in_depth, in_height, in_width;
        uint32_t out_depth, out_height, out_width;
        uint32_t kernel_d, kernel_h, kernel_w;
        uint32_t stride_d, stride_h, stride_w;
        uint32_t padding_d, padding_h, padding_w;
        uint32_t dilation_d, dilation_h, dilation_w;
        uint32_t ceil_mode;
    } pc;

    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch_channels = static_cast<uint32_t>(batch * channels);
    pc.in_depth = static_cast<uint32_t>(in_depth);
    pc.in_height = static_cast<uint32_t>(in_height);
    pc.in_width = static_cast<uint32_t>(in_width);
    pc.out_depth = static_cast<uint32_t>(out_depth);
    pc.out_height = static_cast<uint32_t>(out_height);
    pc.out_width = static_cast<uint32_t>(out_width);
    pc.kernel_d = static_cast<uint32_t>(kernel_d);
    pc.kernel_h = static_cast<uint32_t>(kernel_h);
    pc.kernel_w = static_cast<uint32_t>(kernel_w);
    pc.stride_d = static_cast<uint32_t>(stride_d);
    pc.stride_h = static_cast<uint32_t>(stride_h);
    pc.stride_w = static_cast<uint32_t>(stride_w);
    pc.padding_d = static_cast<uint32_t>(padding_d);
    pc.padding_h = static_cast<uint32_t>(padding_h);
    pc.padding_w = static_cast<uint32_t>(padding_w);
    pc.dilation_d = static_cast<uint32_t>(dilation_d);
    pc.dilation_h = static_cast<uint32_t>(dilation_h);
    pc.dilation_w = static_cast<uint32_t>(dilation_w);
    pc.ceil_mode = ceil_mode ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchMaxPool3dBackward(const Tensor& grad_output, const Tensor& indices,
                                               int64_t D_in, int64_t H_in, int64_t W_in) -> Tensor {
    auto grad_out_shape = grad_output.shape();
    int64_t N = grad_out_shape[0], C = grad_out_shape[1];
    int64_t grad_out_numel = grad_output.numel();
    int64_t grad_in_numel = N * C * D_in * H_in * W_in;

    if (grad_out_numel == 0) return Tensor({N, C, D_in, H_in, W_in}, grad_output.dtype(), grad_output.device());

    // Float16/BFloat16 scatter-backward produce zero/garbage grads — the f32
    // shader would write 4-byte floats into a 2-byte buffer (OOB) and there is
    // no reliable 16-bit atomic accumulation. Widen to Float32, compute, narrow.
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        const DType orig = grad_output.dtype();
        auto gi = dispatchMaxPool3dBackward(grad_output.to(DType::Float32), indices, D_in, H_in, W_in);
        return gi.to(orig);
    }

    int32_t device_id = grad_output.device().index;
    Tensor grad_input = dispatchZeros({N, C, D_in, H_in, W_in}, grad_output.dtype(), grad_output.device());

    std::string shader_name = "max_pool3d_backward";
    if (grad_output.dtype() == DType::Float64) shader_name = "max_pool3d_backward_f64";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()}, {1, indices.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(grad_out_numel * grad_output.dtype_size()),
        static_cast<size_t>(indices.numel() * indices.dtype_size()),
        static_cast<size_t>(grad_in_numel * grad_input.dtype_size())
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants { uint32_t n_elements, grad_input_size; } pc;
    pc.n_elements = static_cast<uint32_t>(grad_out_numel);
    pc.grad_input_size = static_cast<uint32_t>(grad_in_numel);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, div_wg(grad_out_numel, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchAvgPool3dForward(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("avg_pool3d requires 5D input (N, C, D, H, W)");
    }

    // Accept scalar KernelSize/Stride/Padding fallbacks (set by AvgPool3d
    // Module); otherwise the output-shape math below divides by zero.
    int64_t ks = attrs.get_int(AttrKey::KernelSize, 0);
    int64_t st = attrs.get_int(AttrKey::Stride, ks);
    int64_t pd = attrs.get_int(AttrKey::Padding, 0);
    int64_t kernel_d = attrs.has(AttrKey::KernelSizeD) ? attrs.get_int(AttrKey::KernelSizeD) : ks;
    int64_t kernel_h = attrs.has(AttrKey::KernelSizeH) ? attrs.get_int(AttrKey::KernelSizeH) : ks;
    int64_t kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : ks;
    int64_t stride_d = attrs.has(AttrKey::StrideD) ? attrs.get_int(AttrKey::StrideD) : (st > 0 ? st : kernel_d);
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : (st > 0 ? st : kernel_h);
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : (st > 0 ? st : kernel_w);
    int64_t padding_d = attrs.has(AttrKey::PaddingD) ? attrs.get_int(AttrKey::PaddingD) : pd;
    int64_t padding_h = attrs.has(AttrKey::PaddingH) ? attrs.get_int(AttrKey::PaddingH) : pd;
    int64_t padding_w = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : pd;
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1);
    if (kernel_d <= 0 || kernel_h <= 0 || kernel_w <= 0 ||
        stride_d <= 0 || stride_h <= 0 || stride_w <= 0) {
        throw std::invalid_argument("Vulkan avg_pool3d: kernel_size and stride must be positive");
    }

    int64_t batch = input_shape[0], channels = input_shape[1];
    int64_t in_depth = input_shape[2], in_height = input_shape[3], in_width = input_shape[4];

    int64_t out_depth = (in_depth + 2 * padding_d - kernel_d) / stride_d + 1;
    int64_t out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;

    std::string shader_name = "avg_pool3d";
    if (input.dtype() == DType::Float64) shader_name = "avg_pool3d_f64";
    else if (input.dtype() == DType::Float16) shader_name = "avg_pool3d_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "avg_pool3d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> output_shape = {batch, channels, out_depth, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    size_t buf_in = input.numel() * input.dtype_size();
    size_t buf_out = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16) {
        buf_in = ((input.numel() + 1) / 2) * 4;
        buf_out = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, input.data_ptr()}, {1, output.data_ptr()}};
    std::vector<size_t> sizes = {buf_in, buf_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch_channels;
        uint32_t in_depth, in_height, in_width;
        uint32_t out_depth, out_height, out_width;
        uint32_t kernel_d, kernel_h, kernel_w;
        uint32_t stride_d, stride_h, stride_w;
        uint32_t padding_d, padding_h, padding_w;
        uint32_t count_include_pad;
    } pc;

    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch_channels = static_cast<uint32_t>(batch * channels);
    pc.in_depth = static_cast<uint32_t>(in_depth);
    pc.in_height = static_cast<uint32_t>(in_height);
    pc.in_width = static_cast<uint32_t>(in_width);
    pc.out_depth = static_cast<uint32_t>(out_depth);
    pc.out_height = static_cast<uint32_t>(out_height);
    pc.out_width = static_cast<uint32_t>(out_width);
    pc.kernel_d = static_cast<uint32_t>(kernel_d);
    pc.kernel_h = static_cast<uint32_t>(kernel_h);
    pc.kernel_w = static_cast<uint32_t>(kernel_w);
    pc.stride_d = static_cast<uint32_t>(stride_d);
    pc.stride_h = static_cast<uint32_t>(stride_h);
    pc.stride_w = static_cast<uint32_t>(stride_w);
    pc.padding_d = static_cast<uint32_t>(padding_d);
    pc.padding_h = static_cast<uint32_t>(padding_h);
    pc.padding_w = static_cast<uint32_t>(padding_w);
    pc.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAvgPool3dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    // Accept scalar KernelSize/Stride/Padding fallbacks.
    int64_t ks = attrs.get_int(AttrKey::KernelSize, 0);
    int64_t st = attrs.get_int(AttrKey::Stride, ks);
    int64_t pd = attrs.get_int(AttrKey::Padding, 0);
    int64_t kernel_d = attrs.has(AttrKey::KernelSizeD) ? attrs.get_int(AttrKey::KernelSizeD) : ks;
    int64_t kernel_h = attrs.has(AttrKey::KernelSizeH) ? attrs.get_int(AttrKey::KernelSizeH) : ks;
    int64_t kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : ks;
    int64_t stride_d = attrs.has(AttrKey::StrideD) ? attrs.get_int(AttrKey::StrideD) : (st > 0 ? st : kernel_d);
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : (st > 0 ? st : kernel_h);
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : (st > 0 ? st : kernel_w);
    int64_t padding_d = attrs.has(AttrKey::PaddingD) ? attrs.get_int(AttrKey::PaddingD) : pd;
    int64_t padding_h = attrs.has(AttrKey::PaddingH) ? attrs.get_int(AttrKey::PaddingH) : pd;
    int64_t padding_w = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : pd;
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1);

    int64_t batch = input_shape[0], channels = input_shape[1];
    int64_t in_depth = input_shape[2], in_height = input_shape[3], in_width = input_shape[4];
    int64_t out_depth = grad_output.shape()[2], out_height = grad_output.shape()[3], out_width = grad_output.shape()[4];

    int32_t device_id = input.device().index;

    std::string shader_name = "avg_pool3d_backward";
    if (input.dtype() == DType::Float64) {
        shader_name = "avg_pool3d_backward_f64";
    } else if (input.dtype() == DType::Float16) shader_name = "avg_pool3d_backward_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "avg_pool3d_backward_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> grad_input_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_input_shape, input.dtype(), input.device());
    grad_input = dispatchFill(grad_input, 0.0f);

    size_t buf_go = grad_output.numel() * grad_output.dtype_size();
    size_t buf_gi = grad_input.numel() * grad_input.dtype_size();
    if (input.dtype() == DType::Float16) {
        buf_go = ((grad_output.numel() + 1) / 2) * 4;
        buf_gi = ((grad_input.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, grad_output.data_ptr()}, {1, grad_input.data_ptr()}};
    std::vector<size_t> sizes = {buf_go, buf_gi};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch_channels;
        uint32_t in_depth, in_height, in_width;
        uint32_t out_depth, out_height, out_width;
        uint32_t kernel_d, kernel_h, kernel_w;
        uint32_t stride_d, stride_h, stride_w;
        uint32_t padding_d, padding_h, padding_w;
        uint32_t count_include_pad;
    } pc;

    // For f64 shader: input-centric gather (one thread per input element)
    // For other shaders: output-centric scatter (one thread per output element)
    bool is_input_centric_3d = (input.dtype() == DType::Float64);
    pc.n_elements = static_cast<uint32_t>(is_input_centric_3d ? input.numel() : grad_output.numel());
    pc.batch_channels = static_cast<uint32_t>(batch * channels);
    pc.in_depth = static_cast<uint32_t>(in_depth);
    pc.in_height = static_cast<uint32_t>(in_height);
    pc.in_width = static_cast<uint32_t>(in_width);
    pc.out_depth = static_cast<uint32_t>(out_depth);
    pc.out_height = static_cast<uint32_t>(out_height);
    pc.out_width = static_cast<uint32_t>(out_width);
    pc.kernel_d = static_cast<uint32_t>(kernel_d);
    pc.kernel_h = static_cast<uint32_t>(kernel_h);
    pc.kernel_w = static_cast<uint32_t>(kernel_w);
    pc.stride_d = static_cast<uint32_t>(stride_d);
    pc.stride_h = static_cast<uint32_t>(stride_h);
    pc.stride_w = static_cast<uint32_t>(stride_w);
    pc.padding_d = static_cast<uint32_t>(padding_d);
    pc.padding_h = static_cast<uint32_t>(padding_h);
    pc.padding_w = static_cast<uint32_t>(padding_w);
    pc.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    int64_t dispatch_count_3d = is_input_centric_3d ? input.numel() : grad_output.numel();
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(dispatch_count_3d, devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchAdaptiveMaxPool3d(const Tensor& input, int64_t out_d, int64_t out_h, int64_t out_w) -> std::pair<Tensor, Tensor> {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0], channels = input_shape[1];
    int64_t in_depth = input_shape[2], in_height = input_shape[3], in_width = input_shape[4];
    int32_t device_id = cont_input.device().index;

    std::string shader_name = "adaptive_pooling_3d";
    if (cont_input.dtype() == DType::Float64) shader_name = "adaptive_pooling_3d_f64";
    else if (cont_input.dtype() == DType::Float16) shader_name = "adaptive_pooling_3d_f16";
    else if (cont_input.dtype() == DType::BFloat16) shader_name = "adaptive_pooling_3d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_d, out_h, out_w};
    Tensor output(out_shape, cont_input.dtype(), cont_input.device());
    Tensor indices(out_shape, DType::Int32, cont_input.device());

    size_t in_sz = cont_input.numel() * cont_input.dtype_size();
    size_t out_sz = output.numel() * output.dtype_size();
    size_t idx_sz = indices.numel() * indices.dtype_size();
    if (cont_input.dtype() == DType::Float16) {
        in_sz = ((cont_input.numel() + 1) / 2) * 4;
        out_sz = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, cont_input.data_ptr()}, {1, output.data_ptr()}, {2, indices.data_ptr()}
    };
    std::vector<size_t> sizes = {in_sz, out_sz, idx_sz};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch_channels;
        uint32_t in_depth, in_height, in_width;
        uint32_t out_depth, out_height, out_width;
        uint32_t pool_type;
    } pc;
    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch_channels = static_cast<uint32_t>(batch * channels);
    pc.in_depth = static_cast<uint32_t>(in_depth);
    pc.in_height = static_cast<uint32_t>(in_height);
    pc.in_width = static_cast<uint32_t>(in_width);
    pc.out_depth = static_cast<uint32_t>(out_d);
    pc.out_height = static_cast<uint32_t>(out_h);
    pc.out_width = static_cast<uint32_t>(out_w);
    pc.pool_type = 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchAdaptiveAvgPool3d(const Tensor& input, int64_t out_d, int64_t out_h, int64_t out_w) -> Tensor {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0], channels = input_shape[1];
    int64_t in_depth = input_shape[2], in_height = input_shape[3], in_width = input_shape[4];
    int32_t device_id = cont_input.device().index;

    std::string shader_name = "adaptive_pooling_3d";
    if (cont_input.dtype() == DType::Float64) shader_name = "adaptive_pooling_3d_f64";
    else if (cont_input.dtype() == DType::Float16) shader_name = "adaptive_pooling_3d_f16";
    else if (cont_input.dtype() == DType::BFloat16) shader_name = "adaptive_pooling_3d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_d, out_h, out_w};
    Tensor output(out_shape, cont_input.dtype(), cont_input.device());
    Tensor dummy_indices(out_shape, DType::Int32, cont_input.device());

    size_t in_sz = cont_input.numel() * cont_input.dtype_size();
    size_t out_sz = output.numel() * output.dtype_size();
    size_t idx_sz = dummy_indices.numel() * dummy_indices.dtype_size();
    if (cont_input.dtype() == DType::Float16) {
        in_sz = ((cont_input.numel() + 1) / 2) * 4;
        out_sz = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, cont_input.data_ptr()}, {1, output.data_ptr()}, {2, dummy_indices.data_ptr()}
    };
    std::vector<size_t> sizes = {in_sz, out_sz, idx_sz};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch_channels;
        uint32_t in_depth, in_height, in_width;
        uint32_t out_depth, out_height, out_width;
        uint32_t pool_type;
    } pc;
    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch_channels = static_cast<uint32_t>(batch * channels);
    pc.in_depth = static_cast<uint32_t>(in_depth);
    pc.in_height = static_cast<uint32_t>(in_height);
    pc.in_width = static_cast<uint32_t>(in_width);
    pc.out_depth = static_cast<uint32_t>(out_d);
    pc.out_height = static_cast<uint32_t>(out_h);
    pc.out_width = static_cast<uint32_t>(out_w);
    pc.pool_type = 1;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAdaptiveMaxPool3dBackward(const Tensor& grad_output, const Tensor& indices,
                                                        const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t grad_out_numel = grad_output.numel();
    int64_t grad_in_numel = 1;
    for (auto s : input_shape) grad_in_numel *= s;
    if (grad_out_numel == 0) return Tensor(input_shape, grad_output.dtype(), grad_output.device());

    // Float16/BFloat16 scatter-backward produce zero/garbage grads — the f32
    // shader would write 4-byte floats into a 2-byte buffer (OOB) and there is
    // no reliable 16-bit atomic accumulation. Widen to Float32, compute, narrow.
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        const DType orig = grad_output.dtype();
        auto gi = dispatchAdaptiveMaxPool3dBackward(grad_output.to(DType::Float32), indices, input_shape);
        return gi.to(orig);
    }

    int32_t device_id = grad_output.device().index;
    Tensor grad_input = dispatchZeros(input_shape, grad_output.dtype(), grad_output.device());

    std::string shader_name = "max_pool3d_backward";
    if (grad_output.dtype() == DType::Float64) shader_name = "max_pool3d_backward_f64";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()}, {1, indices.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(grad_out_numel * grad_output.dtype_size()),
        static_cast<size_t>(indices.numel() * indices.dtype_size()),
        static_cast<size_t>(grad_in_numel * grad_input.dtype_size())
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants { uint32_t n_elements, grad_input_size; } pc;
    pc.n_elements = static_cast<uint32_t>(grad_out_numel);
    pc.grad_input_size = static_cast<uint32_t>(grad_in_numel);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, div_wg(grad_out_numel, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchAdaptiveAvgPool3dBackward(const Tensor& grad_output, int64_t D_in, int64_t H_in, int64_t W_in) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t batch = grad_shape[0], channels = grad_shape[1];
    int64_t D_out = grad_shape[2], H_out = grad_shape[3], W_out = grad_shape[4];

    std::vector<int64_t> out_shape = {batch, channels, D_in, H_in, W_in};
    Tensor grad_input = dispatchZeros(out_shape, grad_output.dtype(), grad_output.device());

    if (D_in % D_out == 0 && H_in % H_out == 0 && W_in % W_out == 0) {
        OpAttributes backward_attrs;
        backward_attrs.set(AttrKey::KernelSizeD, D_in / D_out);
        backward_attrs.set(AttrKey::KernelSizeH, H_in / H_out);
        backward_attrs.set(AttrKey::KernelSizeW, W_in / W_out);
        backward_attrs.set(AttrKey::StrideD, D_in / D_out);
        backward_attrs.set(AttrKey::StrideH, H_in / H_out);
        backward_attrs.set(AttrKey::StrideW, W_in / W_out);
        backward_attrs.set(AttrKey::PaddingD, static_cast<int64_t>(0));
        backward_attrs.set(AttrKey::PaddingH, static_cast<int64_t>(0));
        backward_attrs.set(AttrKey::PaddingW, static_cast<int64_t>(0));
        backward_attrs.set(AttrKey::CountIncludePad, static_cast<int64_t>(1));
        Tensor dummy_input(out_shape, grad_output.dtype(), grad_output.device());
        return dispatchAvgPool3dBackward(grad_output, dummy_input, backward_attrs);
    }

    // Non-divisible case: previously fell through and returned the all-zeros
    // grad_input, silently producing zero gradients (broken training). Distribute
    // grad_output / window_size over each adaptive window via a dedicated
    // adaptive_avg_pool3d_backward shader, mirroring the 2D path.
    auto cont_grad = grad_output.contiguous();
    int32_t device_id = cont_grad.device().index;

    // Float32 uses atomic float add and iterates over OUTPUT elements; Float64 /
    // Float16 / BFloat16 lack atomic add, so they iterate over INPUT elements and
    // accumulate per-input (race-free). grad_input is already zero-initialized
    // above via dispatchZeros, which the Float32 atomic accumulation requires; the
    // input-iteration variants overwrite each input position exactly once.
    const bool is_float64 = (cont_grad.dtype() == DType::Float64);
    const bool is_float16 = (cont_grad.dtype() == DType::Float16);
    const bool is_bfloat16 = (cont_grad.dtype() == DType::BFloat16);
    const bool needs_input_iteration = is_float64 || is_float16 || is_bfloat16;

    std::string shader_name = "adaptive_avg_pool3d_backward";
    if (is_float64) shader_name = "adaptive_avg_pool3d_backward_f64";
    else if (is_float16) shader_name = "adaptive_avg_pool3d_backward_f16";
    else if (is_bfloat16) shader_name = "adaptive_avg_pool3d_backward_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto packed_half_words = [](int64_t numel) -> size_t {
        return static_cast<size_t>((numel + 1) / 2) * 4;
    };
    const bool is_packed_half = is_float16 || is_bfloat16;
    size_t grad_output_size = is_packed_half
        ? packed_half_words(cont_grad.numel())
        : static_cast<size_t>(cont_grad.numel()) * cont_grad.dtype_size();
    size_t grad_input_size = is_packed_half
        ? packed_half_words(grad_input.numel())
        : static_cast<size_t>(grad_input.numel()) * grad_input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, cont_grad.data_ptr()},
        {1, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {grad_output_size, grad_input_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch;
        uint32_t channels;
        uint32_t D_in;
        uint32_t H_in;
        uint32_t W_in;
        uint32_t D_out;
        uint32_t H_out;
        uint32_t W_out;
    } pc;
    pc.batch = static_cast<uint32_t>(batch);
    pc.channels = static_cast<uint32_t>(channels);
    pc.D_in = static_cast<uint32_t>(D_in);
    pc.H_in = static_cast<uint32_t>(H_in);
    pc.W_in = static_cast<uint32_t>(W_in);
    pc.D_out = static_cast<uint32_t>(D_out);
    pc.H_out = static_cast<uint32_t>(H_out);
    pc.W_out = static_cast<uint32_t>(W_out);

    uint32_t total_elements = needs_input_iteration
        ? static_cast<uint32_t>(batch * channels * D_in * H_in * W_in)
        : static_cast<uint32_t>(batch * channels * D_out * H_out * W_out);
    uint32_t workgroups = div_wg(total_elements, devices_[device_id].workgroupSize);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

} // namespace tenzor
