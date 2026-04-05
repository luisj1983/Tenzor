#include "vulkan_ops_common.hpp"

namespace tenzor {

// ============================================================================
// Complex Number Operations
// ============================================================================

auto VulkanBackend::dispatchConj(const Tensor& input) -> Tensor {
    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Float16: use native F16 shader (operates on packed complex elements)
    if (input.dtype() == DType::Float16) {
        int32_t device_id = input.device().index;
        auto* pipeline = getPipeline("conj_f16", device_id);

        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        Tensor output(out_shape, input.dtype(), input.device());

        // For F16 conj, num_complex = numel / 2 (each complex = 2 float16 = 1 uint32)
        uint32_t num_complex = static_cast<uint32_t>(input.numel() / 2);
        struct { uint32_t num_complex; } pc;
        pc.num_complex = num_complex;

        size_t buf_size = ((input.numel() + 1) / 2) * 4;  // packed F16 buffer

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {buf_size, buf_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_complex, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16_conj = (input.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "conj_f64" : is_bfloat16_conj ? "conj_bf16" : "conj";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    Tensor output(out_shape, input.dtype(), input.device());

    uint32_t num_elements = static_cast<uint32_t>(input.numel());

    struct { uint32_t num_elements; } pushConstants;
    pushConstants.num_elements = num_elements;

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size = input.numel() * input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchReal(const Tensor& input) -> Tensor {
    if (input.numel() == 0) {
        // Output has half the elements (real parts only)
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        if (!out_shape.empty()) out_shape.back() /= 2;
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Float16: use native F16 shader (packed complex -> packed real)
    if (input.dtype() == DType::Float16) {
        int32_t device_id = input.device().index;
        auto* pipeline = getPipeline("real_f16", device_id);

        int64_t num_complex = input.numel() / 2;
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        if (!out_shape.empty()) out_shape.back() /= 2;
        Tensor output(out_shape, input.dtype(), input.device());

        struct { uint32_t num_complex; } pc;
        pc.num_complex = static_cast<uint32_t>(num_complex);

        // F16: input is num_complex packed complex uint32 words, output is (num_complex+1)/2 packed real words
        size_t in_size = num_complex * 4;  // 1 uint32 per complex element
        size_t out_size = ((num_complex + 1) / 2) * 4;  // packed real F16

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        uint32_t num_words = static_cast<uint32_t>((num_complex + 1) / 2);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_words, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16_real = (input.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "real_f64" : is_bfloat16_real ? "real_bf16" : "real";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Complex buffer has pairs: num_complex = numel / 2
    int64_t num_complex = input.numel() / 2;
    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    if (!out_shape.empty()) out_shape.back() /= 2;
    Tensor output(out_shape, input.dtype(), input.device());

    struct { uint32_t num_complex; } pushConstants;
    pushConstants.num_complex = static_cast<uint32_t>(num_complex);

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t in_size = input.numel() * input.dtype_size();
    size_t out_size = num_complex * input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {in_size, out_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(num_complex, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchImag(const Tensor& input) -> Tensor {
    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        if (!out_shape.empty()) out_shape.back() /= 2;
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Float16: use native F16 shader
    if (input.dtype() == DType::Float16) {
        int32_t device_id = input.device().index;
        auto* pipeline = getPipeline("imag_f16", device_id);

        int64_t num_complex = input.numel() / 2;
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        if (!out_shape.empty()) out_shape.back() /= 2;
        Tensor output(out_shape, input.dtype(), input.device());

        struct { uint32_t num_complex; } pc;
        pc.num_complex = static_cast<uint32_t>(num_complex);

        size_t in_size = num_complex * 4;
        size_t out_size = ((num_complex + 1) / 2) * 4;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        uint32_t num_words = static_cast<uint32_t>((num_complex + 1) / 2);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_words, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16_imag = (input.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "imag_f64" : is_bfloat16_imag ? "imag_bf16" : "imag";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t num_complex = input.numel() / 2;
    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    if (!out_shape.empty()) out_shape.back() /= 2;
    Tensor output(out_shape, input.dtype(), input.device());

    struct { uint32_t num_complex; } pushConstants;
    pushConstants.num_complex = static_cast<uint32_t>(num_complex);

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t in_size = input.numel() * input.dtype_size();
    size_t out_size = num_complex * input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {in_size, out_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(num_complex, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAngle(const Tensor& input) -> Tensor {
    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        if (!out_shape.empty()) out_shape.back() /= 2;
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Float16/BFloat16: use native packed shader
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        bool is_bf16_angle = (input.dtype() == DType::BFloat16);
        int32_t device_id = input.device().index;
        auto* pipeline = getPipeline(is_bf16_angle ? "angle_bf16" : "angle_f16", device_id);

        int64_t num_complex = input.numel() / 2;
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        if (!out_shape.empty()) out_shape.back() /= 2;
        Tensor output(out_shape, input.dtype(), input.device());

        struct { uint32_t num_complex; } pc;
        pc.num_complex = static_cast<uint32_t>(num_complex);

        size_t in_size = num_complex * 4;
        size_t out_size = ((num_complex + 1) / 2) * 4;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        uint32_t num_words = static_cast<uint32_t>((num_complex + 1) / 2);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_words, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);

    std::string shader_name = is_float64 ? "angle_f64" : "angle";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t num_complex = input.numel() / 2;
    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    if (!out_shape.empty()) out_shape.back() /= 2;
    Tensor output(out_shape, input.dtype(), input.device());

    struct { uint32_t num_complex; } pushConstants;
    pushConstants.num_complex = static_cast<uint32_t>(num_complex);

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t in_size = input.numel() * input.dtype_size();
    size_t out_size = num_complex * input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {in_size, out_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(num_complex, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchPolar(const Tensor& abs, const Tensor& angle) -> Tensor {
    if (abs.numel() == 0) {
        // Output shape: same as input but last dim doubled (interleaved real, imag)
        std::vector<int64_t> out_shape(abs.shape().begin(), abs.shape().end());
        if (!out_shape.empty()) out_shape.back() *= 2;
        return Tensor(out_shape, abs.dtype(), abs.device());
    }

    // Float16/BFloat16: use native packed shader
    if (abs.dtype() == DType::Float16 || abs.dtype() == DType::BFloat16) {
        bool is_bf16_polar = (abs.dtype() == DType::BFloat16);
        int32_t device_id = abs.device().index;
        auto* pipeline = getPipeline(is_bf16_polar ? "polar_bf16" : "polar_f16", device_id);

        int64_t num_complex = abs.numel();
        std::vector<int64_t> out_shape(abs.shape().begin(), abs.shape().end());
        if (!out_shape.empty()) out_shape.back() *= 2;
        Tensor output(out_shape, abs.dtype(), abs.device());

        struct { uint32_t num_complex; } pc;
        pc.num_complex = static_cast<uint32_t>(num_complex);

        // abs and angle are packed F16 real buffers
        size_t real_buf_size = ((num_complex + 1) / 2) * 4;
        // output is packed complex: num_complex uint32 words
        size_t out_size = num_complex * 4;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, abs.data_ptr()}, {1, angle.data_ptr()}, {2, output.data_ptr()}
        };
        std::vector<size_t> sizes = {real_buf_size, real_buf_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_complex, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    int32_t device_id = abs.device().index;
    bool is_float64 = (abs.dtype() == DType::Float64);

    std::string shader_name = is_float64 ? "polar_f64" : "polar";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t num_complex = abs.numel();
    std::vector<int64_t> out_shape(abs.shape().begin(), abs.shape().end());
    if (!out_shape.empty()) out_shape.back() *= 2;
    Tensor output(out_shape, abs.dtype(), abs.device());

    struct { uint32_t num_complex; } pushConstants;
    pushConstants.num_complex = static_cast<uint32_t>(num_complex);

    const void* buffer_abs = abs.data_ptr();
    const void* buffer_angle = angle.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t in_size = abs.numel() * abs.dtype_size();
    size_t out_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_abs},
        {1, buffer_angle},
        {2, buffer_out}
    };
    std::vector<size_t> sizes = {in_size, in_size, out_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(num_complex, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}


} // namespace tenzor
