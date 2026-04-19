#include "vulkan_ops_common.hpp"

namespace tenzor {

// Softmax and loss operations implementation
auto VulkanBackend::dispatchSoftmax(const Tensor& input_orig, int64_t dim) -> Tensor {
    // The shader's batch_size = product(shape[0..dim)) and num_classes =
    // shape[dim] assume softmax is applied to the LAST dimension (the
    // innermost contiguous axis). For dim != last_dim the flat-buffer
    // dispatch reads down the wrong axis and produces wrong results.
    // Normalize: if dim isn't the last axis, transpose to put it last,
    // recurse with dim=-1, then transpose back. Also materialize a
    // contiguous copy so non-contiguous strided views work.
    auto orig_shape_span = input_orig.shape();
    int64_t ndim = static_cast<int64_t>(orig_shape_span.size());
    int64_t norm_dim = (dim < 0) ? ndim + dim : dim;
    int64_t last = ndim - 1;
    if (norm_dim != last) {
        Tensor transposed = input_orig.transpose(norm_dim, last).contiguous();
        Tensor result = dispatchSoftmax(transposed, /*dim=*/last);
        return result.transpose(norm_dim, last).contiguous();
    }

    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // For Float16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchSoftmax(input_f32, dim);
        return result_f32.to(orig_dtype);
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "softmax_f64";
    } else if (is_bfloat16) {
        shader_name = "softmax_bf16";
    } else {
        shader_name = "softmax";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Handle negative dimension
    if (dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate batch size and number of classes
    uint32_t batch_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        batch_size *= static_cast<uint32_t>(input_shape[i]);
    }
    uint32_t num_classes = static_cast<uint32_t>(input_shape[dim]);

    // NOTE: max_vals and sum_vals are computed using shared memory within the shader.
    // No separate device allocations needed - the backward pass computes from output only.

    // Get VkBuffer handles
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Bind buffers (binding 0: input, 1: output)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t batch_size;
        uint32_t num_classes;
        uint32_t dim;
        uint32_t mode;  // 0=forward, 1=backward
    } pushConstants;

    pushConstants.batch_size = batch_size;
    pushConstants.num_classes = num_classes;
    pushConstants.dim = static_cast<uint32_t>(dim);
    pushConstants.mode = 0;  // forward

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per batch element
    vkCmdDispatch(cmdBuffer, batch_size, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return fp16_saturate_if_needed(*this, output);
}

auto VulkanBackend::dispatchLogSoftmax(const Tensor& input_orig, int64_t dim) -> Tensor {
    // Same dim-handling concern as dispatchSoftmax — the shader's
    // batch_size/num_classes assume softmax over the last axis. Normalize
    // by transposing the requested dim to the last position when needed.
    auto orig_shape_span = input_orig.shape();
    int64_t ndim = static_cast<int64_t>(orig_shape_span.size());
    int64_t norm_dim = (dim < 0) ? ndim + dim : dim;
    int64_t last = ndim - 1;
    if (norm_dim != last) {
        Tensor transposed = input_orig.transpose(norm_dim, last).contiguous();
        Tensor result = dispatchLogSoftmax(transposed, /*dim=*/last);
        return result.transpose(norm_dim, last).contiguous();
    }

    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "log_softmax";
    if (input.dtype() == DType::Float64) shader_name = "log_softmax_f64";
    else if (input.dtype() == DType::Float16) shader_name = "log_softmax_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "log_softmax_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Handle negative dimension
    if (dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate batch size and number of classes
    uint32_t batch_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        batch_size *= static_cast<uint32_t>(input_shape[i]);
    }
    uint32_t num_classes = static_cast<uint32_t>(input_shape[dim]);

    // Get VkBuffer handles
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Bind buffers (binding 0: input, 1: output)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t batch_size;
        uint32_t num_classes;
        uint32_t dim;
    } pushConstants;

    pushConstants.batch_size = batch_size;
    pushConstants.num_classes = num_classes;
    pushConstants.dim = static_cast<uint32_t>(dim);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per batch element
    vkCmdDispatch(cmdBuffer, batch_size, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return fp16_saturate_if_needed(*this, output);
}

auto VulkanBackend::dispatchNestedLogSoftmax(const Tensor& values, const Tensor& offsets,
                                              int64_t /*dim*/) -> Tensor {
    int32_t device_id = values.device().index;
    auto shape = values.shape();
    int64_t total_len = shape[0];
    uint32_t D = (shape.size() > 1) ? static_cast<uint32_t>(shape[1]) : 1;
    uint32_t B = static_cast<uint32_t>(offsets.numel() - 1);

    auto* pipeline = getPipeline("nested_log_softmax", device_id);

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), values.dtype(), values.device());

    // Offsets must be Int32 for the shader
    Tensor offsets_i32 = (offsets.dtype() == DType::Int32)
                         ? offsets : offsets.to(DType::Int32);

    const void* buf_values  = values.data_ptr();
    const void* buf_offsets = offsets_i32.data_ptr();
    const void* buf_output  = output.data_ptr();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_values},
        {1, buf_offsets},
        {2, buf_output},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(values.numel()) * values.dtype_size(),
        static_cast<size_t>(offsets_i32.numel()) * offsets_i32.dtype_size(),
        static_cast<size_t>(output.numel()) * output.dtype_size(),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);

    struct { uint32_t D; uint32_t B; } pc{D, B};
    vkCmdPushConstants(cmd, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    // One workgroup per batch element
    vkCmdDispatch(cmd, B, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchNestedSoftmax(const Tensor& values, const Tensor& offsets,
                                           int64_t /*dim*/) -> Tensor {
    int32_t device_id = values.device().index;
    auto shape = values.shape();
    uint32_t D = (shape.size() > 1) ? static_cast<uint32_t>(shape[1]) : 1;
    uint32_t B = static_cast<uint32_t>(offsets.numel() - 1);

    auto* pipeline = getPipeline("nested_softmax", device_id);

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), values.dtype(), values.device());

    Tensor offsets_i32 = (offsets.dtype() == DType::Int32)
                         ? offsets : offsets.to(DType::Int32);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, values.data_ptr()},
        {1, offsets_i32.data_ptr()},
        {2, output.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(values.numel()) * values.dtype_size(),
        static_cast<size_t>(offsets_i32.numel()) * offsets_i32.dtype_size(),
        static_cast<size_t>(output.numel()) * output.dtype_size(),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);

    struct { uint32_t D; uint32_t B; } pc{D, B};
    vkCmdPushConstants(cmd, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, B, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchNestedSum(const Tensor& values, const Tensor& offsets) -> Tensor {
    int32_t device_id = values.device().index;
    auto shape = values.shape();
    uint32_t D = (shape.size() > 1) ? static_cast<uint32_t>(shape[1]) : 1;
    uint32_t B = static_cast<uint32_t>(offsets.numel() - 1);

    auto* pipeline = getPipeline("nested_sum", device_id);

    // Output is [B, D] (one row per segment, keepdim=true style)
    Tensor output({static_cast<int64_t>(B), static_cast<int64_t>(D)}, values.dtype(), values.device());

    Tensor offsets_i32 = (offsets.dtype() == DType::Int32)
                         ? offsets : offsets.to(DType::Int32);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, values.data_ptr()},
        {1, offsets_i32.data_ptr()},
        {2, output.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(values.numel()) * values.dtype_size(),
        static_cast<size_t>(offsets_i32.numel()) * offsets_i32.dtype_size(),
        static_cast<size_t>(output.numel()) * output.dtype_size(),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);

    struct { uint32_t D; uint32_t B; } pc{D, B};
    vkCmdPushConstants(cmd, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, B, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchNestedMean(const Tensor& values, const Tensor& offsets) -> Tensor {
    int32_t device_id = values.device().index;
    auto shape = values.shape();
    uint32_t D = (shape.size() > 1) ? static_cast<uint32_t>(shape[1]) : 1;
    uint32_t B = static_cast<uint32_t>(offsets.numel() - 1);

    auto* pipeline = getPipeline("nested_mean", device_id);

    Tensor output({static_cast<int64_t>(B), static_cast<int64_t>(D)}, values.dtype(), values.device());

    Tensor offsets_i32 = (offsets.dtype() == DType::Int32)
                         ? offsets : offsets.to(DType::Int32);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, values.data_ptr()},
        {1, offsets_i32.data_ptr()},
        {2, output.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(values.numel()) * values.dtype_size(),
        static_cast<size_t>(offsets_i32.numel()) * offsets_i32.dtype_size(),
        static_cast<size_t>(output.numel()) * output.dtype_size(),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);

    struct { uint32_t D; uint32_t B; } pc{D, B};
    vkCmdPushConstants(cmd, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, B, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchNestedToPadded(const Tensor& values, const Tensor& offsets,
                                            int64_t max_len, float padding_value) -> Tensor {
    int32_t device_id = values.device().index;
    auto shape = values.shape();
    uint32_t D = (shape.size() > 1) ? static_cast<uint32_t>(shape[1]) : 1;
    uint32_t B = static_cast<uint32_t>(offsets.numel() - 1);

    auto* pipeline = getPipeline("nested_to_padded", device_id);

    Tensor output({static_cast<int64_t>(B), max_len, static_cast<int64_t>(D)},
                  values.dtype(), values.device());

    Tensor offsets_i32 = (offsets.dtype() == DType::Int32)
                         ? offsets : offsets.to(DType::Int32);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, values.data_ptr()},
        {1, offsets_i32.data_ptr()},
        {2, output.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(values.numel()) * values.dtype_size(),
        static_cast<size_t>(offsets_i32.numel()) * offsets_i32.dtype_size(),
        static_cast<size_t>(output.numel()) * output.dtype_size(),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);

    struct { uint32_t D; uint32_t B; uint32_t max_len; float pad_value; } pc{
        D, B, static_cast<uint32_t>(max_len), padding_value};
    vkCmdPushConstants(cmd, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, B, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchNestedFromPadded(const Tensor& padded, const Tensor& offsets) -> Tensor {
    int32_t device_id = padded.device().index;
    auto shape = padded.shape();
    uint32_t B = static_cast<uint32_t>(offsets.numel() - 1);
    uint32_t max_len = static_cast<uint32_t>(shape[1]);
    uint32_t D = (shape.size() > 2) ? static_cast<uint32_t>(shape[2]) : 1;

    // Read only the last element of offsets to get total_len (avoids full GPU->CPU copy)
    Tensor total_len_scalar = offsets.slice(0, static_cast<int64_t>(B),
                                            static_cast<int64_t>(B) + 1).to(Device::cpu());
    int64_t total_len = total_len_scalar.data<int64_t>()[0];

    auto* pipeline = getPipeline("nested_from_padded", device_id);

    Tensor output({total_len, static_cast<int64_t>(D)}, padded.dtype(), padded.device());

    Tensor offsets_i32 = (offsets.dtype() == DType::Int32)
                         ? offsets : offsets.to(DType::Int32);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, padded.data_ptr()},
        {1, offsets_i32.data_ptr()},
        {2, output.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(padded.numel()) * padded.dtype_size(),
        static_cast<size_t>(offsets_i32.numel()) * offsets_i32.dtype_size(),
        static_cast<size_t>(output.numel()) * output.dtype_size(),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);

    struct { uint32_t D; uint32_t B; uint32_t max_len; } pc{D, B, max_len};
    vkCmdPushConstants(cmd, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, B, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchCrossEntropy(const Tensor& log_probs, const Tensor& targets,
                                         int64_t reduction) -> Tensor {
    int32_t device_id = log_probs.device().index;

    // Select shader based on dtype
    bool is_float64 = (log_probs.dtype() == DType::Float64);
    bool is_float16 = (log_probs.dtype() == DType::Float16);
    std::string shader_name = is_float64 ? "cross_entropy_f64" : (is_float16 ? "cross_entropy_f16" : "cross_entropy");
    auto* pipeline = getPipeline(shader_name, device_id);

    auto log_probs_shape = log_probs.shape();
    int64_t batch_size = log_probs_shape[0];
    int64_t num_classes = log_probs_shape[1];

    std::vector<int64_t> out_shape;
    if (reduction == 0) { // none
        out_shape = {batch_size};
    } else { // mean or sum
        out_shape = {1};
    }

    Tensor output(out_shape, log_probs.dtype(), log_probs.device());

    // Get VkBuffer handles
    const void* buffer_log_probs = log_probs.data_ptr();
    const void* buffer_targets = targets.data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes
    size_t size_log_probs = log_probs.numel() * log_probs.dtype_size();
    size_t size_targets = targets.numel() * dtype_size(targets.dtype());
    size_t size_output = output.numel() * output.dtype_size();

    // Bind buffers (binding 0: log_probs, 1: targets, 2: output)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_log_probs},
        {1, buffer_targets},
        {2, buffer_output}
    };
    std::vector<size_t> sizes = {size_log_probs, size_targets, size_output};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t batch_size;
        uint32_t num_classes;
        uint32_t reduction;  // 0=none, 1=mean, 2=sum
    } pushConstants;

    pushConstants.batch_size = static_cast<uint32_t>(batch_size);
    pushConstants.num_classes = static_cast<uint32_t>(num_classes);
    pushConstants.reduction = static_cast<uint32_t>(reduction);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    if (workgroups == 0) workgroups = 1;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

} // namespace tenzor
