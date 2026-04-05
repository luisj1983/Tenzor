#include "vulkan_ops_common.hpp"

namespace tenzor {

// Softmax and loss operations implementation
auto VulkanBackend::dispatchSoftmax(const Tensor& input, int64_t dim) -> Tensor {
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

auto VulkanBackend::dispatchLogSoftmax(const Tensor& input, int64_t dim) -> Tensor {
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
