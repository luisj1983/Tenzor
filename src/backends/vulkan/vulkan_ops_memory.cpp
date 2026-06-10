#include "vulkan_ops_common.hpp"
#include <cmath>
#include <cstdint>

namespace tenzor {

// ============================================================================
// Memory Operations Implementation
// ============================================================================

/**
 * @brief Create tensor filled with zeros
 */
auto VulkanBackend::dispatchZeros(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor {
    // For Float64, Int64, UInt8, or Bool, use full() with 0.0 since the basic fill shader only handles 32-bit values
    // This is consistent with how dispatchOnes handles these types
    if (dtype == DType::Float64 || dtype == DType::Int64 || dtype == DType::UInt8 || dtype == DType::Bool) {
        return dispatchFull(shape, 0.0, dtype);
    }

    // Complex dtypes: zero bits represent zero value in both Complex64 (2x f32) and
    // Complex128 (2x f64). Compute numel*dtype_size then fill the underlying bytes with zeros.
    if (dtype == DType::Complex64 || dtype == DType::Complex128) {
        Tensor output(shape, dtype, device);
        int64_t numel = 1;
        for (auto d : shape) numel *= d;
        if (numel == 0) return output;

        int32_t device_id = device.index;
        auto* pipeline = getPipeline("fill", device_id);

        const void* buffer_out = output.data_ptr();
        // One uint32 per real-or-imag-half (two halves per complex element for Complex64,
        // four halves per complex element for Complex128 when viewed as u32 chunks).
        size_t buffer_size_out = output.numel() * output.dtype_size();
        uint32_t n_u32 = static_cast<uint32_t>(buffer_size_out / 4);

        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buffer_out}};
        std::vector<size_t> sizes = {buffer_size_out};
        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct PushConstants { uint32_t n; float value; } push_constants{n_u32, 0.0f};

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);
        uint32_t workgroups = div_wg(n_u32, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // Create tensor with given shape
    Tensor output(shape, dtype, device);

    // Special case: empty tensors don't need GPU operations
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    if (numel == 0) {
        return output;  // Empty tensor, nothing to fill
    }

    // Fill with zeros using fill shader (writes uint32 zeros).
    // This is safe for Float32, Int32, Float16, BFloat16 since zero bits == zero value
    // for all these types, and 16-bit types are packed 2-per-uint32.
    int32_t device_id = device.index;
    auto* pipeline = getPipeline("fill", device_id);

    const void* buffer_out = output.data_ptr();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t n;
        float value;
    } push_constants;

    push_constants.n = static_cast<uint32_t>(output.numel());
    push_constants.value = 0.0f;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Arange - generate sequential values on GPU
 */
auto VulkanBackend::dispatchArange(double start, double end, double step, DType dtype, const Device& device) -> Tensor {
    if (step == 0.0f) {
        throw std::runtime_error("arange: step must be non-zero");
    }
    if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
        return Tensor({0}, dtype, device);
    }

    int64_t numel = static_cast<int64_t>(std::ceil((end - start) / step));
    if (numel <= 0) {
        return Tensor({0}, dtype, device);
    }

    // Float64 uses dedicated arange_f64 shader for full precision
    if (dtype == DType::Float64) {
        Tensor output({numel}, DType::Float64, device);

        int32_t device_id = device.index;
        auto* pipeline = getPipeline("arange_f64", device_id);

        const void* buf_out = output.data_ptr();
        size_t buf_size = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsF64 {
            uint32_t num_elements;
            uint32_t _pad;
            double start;
            double step;
        } push_constants;

        push_constants.num_elements = static_cast<uint32_t>(numel);
        push_constants._pad = 0;
        push_constants.start = static_cast<double>(start);
        push_constants.step = static_cast<double>(step);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants);

        uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // Float16: use native arange_f16 shader
    if (dtype == DType::Float16) {
        Tensor output({numel}, DType::Float16, device);
        int32_t device_id = device.index;
        auto* pipeline = getPipeline("arange_f16", device_id);

        size_t buf_size = ((numel + 1) / 2) * sizeof(uint32_t);
        const void* buf_out = output.data_ptr();
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            float start;
            float step;
            uint32_t num_elements;
        } push_constants;
        push_constants.start = start;
        push_constants.step = step;
        push_constants.num_elements = static_cast<uint32_t>(numel);

        uint32_t num_pairs = static_cast<uint32_t>((numel + 1) / 2);
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);
        vkCmdDispatch(cmdBuffer, div_wg(num_pairs, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // Int64 / UInt64: dedicated 64-bit shader. Going through a Float32
    // intermediate would truncate large integers (>2^24), so compute the
    // sequence directly from the double-precision start/step.
    if (dtype == DType::Int64 || dtype == DType::UInt64) {
        Tensor output({numel}, dtype, device);
        int32_t device_id = device.index;
        auto* pipeline = getPipeline("arange_i64", device_id);

        const void* buf_out = output.data_ptr();
        size_t buf_size = output.numel() * output.dtype_size();
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};
        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsI64 {
            uint32_t num_elements;
            uint32_t _pad;
            double start;
            double step;
        } push_constants;
        push_constants.num_elements = static_cast<uint32_t>(numel);
        push_constants._pad = 0;
        push_constants.start = start;
        push_constants.step = step;

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsI64), &push_constants);
        uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // For other non-Float32 dtypes, compute in Float32 then convert
    if (dtype != DType::Float32) {
        auto result_f32 = dispatchArange(start, end, step, DType::Float32, device);
        return result_f32.to(dtype);
    }

    Tensor output({numel}, DType::Float32, device);

    int32_t device_id = device.index;
    auto* pipeline = getPipeline("arange", device_id);

    const void* buf_out = output.data_ptr();
    size_t buf_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
    std::vector<size_t> sizes = {buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        float start;
        float step;
        uint32_t num_elements;
    } push_constants;

    push_constants.start = start;
    push_constants.step = step;
    push_constants.num_elements = static_cast<uint32_t>(numel);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Linspace - generate linearly-spaced values on GPU
 */
auto VulkanBackend::dispatchLinspace(double start, double end, int64_t steps, DType dtype, const Device& device) -> Tensor {
    if (steps < 0) {
        throw std::runtime_error("linspace: number of steps must be non-negative");
    }
    if (steps == 0) {
        return Tensor({0}, dtype, device);
    }

    // Float64 uses dedicated linspace_f64 shader for full precision
    if (dtype == DType::Float64) {
        Tensor output({steps}, DType::Float64, device);

        int32_t device_id = device.index;
        auto* pipeline = getPipeline("linspace_f64", device_id);

        const void* buf_out = output.data_ptr();
        size_t buf_size = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsF64 {
            uint32_t num_steps;
            uint32_t _pad;
            double start;
            double end_val;
        } push_constants;

        push_constants.num_steps = static_cast<uint32_t>(steps);
        push_constants._pad = 0;
        push_constants.start = static_cast<double>(start);
        push_constants.end_val = static_cast<double>(end);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants);

        uint32_t workgroups = div_wg(steps, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // Float16: use native linspace_f16 shader
    if (dtype == DType::Float16) {
        Tensor output({steps}, DType::Float16, device);
        int32_t device_id = device.index;
        auto* pipeline = getPipeline("linspace_f16", device_id);

        size_t buf_size = ((steps + 1) / 2) * sizeof(uint32_t);
        const void* buf_out = output.data_ptr();
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};
        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            float start;
            float end_val;
            uint32_t num_steps;
        } push_constants;
        push_constants.start = start;
        push_constants.end_val = end;
        push_constants.num_steps = static_cast<uint32_t>(steps);

        uint32_t num_pairs = static_cast<uint32_t>((steps + 1) / 2);
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);
        vkCmdDispatch(cmdBuffer, div_wg(num_pairs, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // Int64 / UInt64: dedicated 64-bit shader to avoid Float32-intermediate
    // truncation of large integer endpoints.
    if (dtype == DType::Int64 || dtype == DType::UInt64) {
        Tensor output({steps}, dtype, device);
        int32_t device_id = device.index;
        auto* pipeline = getPipeline("linspace_i64", device_id);

        const void* buf_out = output.data_ptr();
        size_t buf_size = output.numel() * output.dtype_size();
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};
        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsI64 {
            uint32_t num_steps;
            uint32_t _pad;
            double start;
            double end_val;
        } push_constants;
        push_constants.num_steps = static_cast<uint32_t>(steps);
        push_constants._pad = 0;
        push_constants.start = start;
        push_constants.end_val = end;

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsI64), &push_constants);
        uint32_t workgroups = div_wg(steps, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // For other non-Float32 dtypes, compute in Float32 then convert
    if (dtype != DType::Float32) {
        auto result_f32 = dispatchLinspace(start, end, steps, DType::Float32, device);
        return result_f32.to(dtype);
    }

    Tensor output({steps}, DType::Float32, device);

    int32_t device_id = device.index;
    auto* pipeline = getPipeline("linspace", device_id);

    const void* buf_out = output.data_ptr();
    size_t buf_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
    std::vector<size_t> sizes = {buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        float start;
        float end_val;
        uint32_t num_steps;
    } push_constants;

    push_constants.start = start;
    push_constants.end_val = end;
    push_constants.num_steps = static_cast<uint32_t>(steps);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(steps, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Eye - generate identity matrix on GPU
 */
auto VulkanBackend::dispatchEye(int64_t n, int64_t m, DType dtype, const Device& device) -> Tensor {
    if (m < 0) m = n;

    if (n == 0 || m == 0) {
        return Tensor({n, m}, dtype, device);
    }

    // Float64 uses dedicated eye_f64 shader for proper double-precision output
    if (dtype == DType::Float64) {
        Tensor output({n, m}, DType::Float64, device);

        int32_t device_id = device.index;
        auto* pipeline = getPipeline("eye_f64", device_id);

        const void* buf_out = output.data_ptr();
        size_t buf_size = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t rows;
            uint32_t cols;
        } push_constants;

        push_constants.rows = static_cast<uint32_t>(n);
        push_constants.cols = static_cast<uint32_t>(m);

        int64_t total = n * m;
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = div_wg(total, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // For other non-Float32 dtypes, compute in Float32 then convert
    // Float16: use native eye_f16 shader
    if (dtype == DType::Float16) {
        Tensor output({n, m}, DType::Float16, device);
        int32_t device_id = device.index;
        auto* pipeline = getPipeline("eye_f16", device_id);

        int64_t total = n * m;
        size_t buf_size = ((total + 1) / 2) * sizeof(uint32_t);
        const void* buf_out = output.data_ptr();
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};
        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t rows;
            uint32_t cols;
        } push_constants;
        push_constants.rows = static_cast<uint32_t>(n);
        push_constants.cols = static_cast<uint32_t>(m);

        uint32_t num_pairs = static_cast<uint32_t>((total + 1) / 2);
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);
        vkCmdDispatch(cmdBuffer, div_wg(num_pairs, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    if (dtype != DType::Float32) {
        auto result_f32 = dispatchEye(n, m, DType::Float32, device);
        return result_f32.to(dtype);
    }

    Tensor output({n, m}, DType::Float32, device);

    int32_t device_id = device.index;
    auto* pipeline = getPipeline("eye", device_id);

    const void* buf_out = output.data_ptr();
    size_t buf_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
    std::vector<size_t> sizes = {buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t rows;
        uint32_t cols;
    } push_constants;

    push_constants.rows = static_cast<uint32_t>(n);
    push_constants.cols = static_cast<uint32_t>(m);

    int64_t total = n * m;
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(total, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Fill tensor with scalar value using compute shader
 */
auto VulkanBackend::dispatchFill(const Tensor& input, double value) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    const void* buffer_out = output.data_ptr();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Float64 requires special handling: the generic fill shader is 32-bit only,
    // so use the full_f64 pipeline which properly writes double-precision values
    if (input.dtype() == DType::Float64) {
        auto* pipeline = getPipeline("full_f64", device_id);

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_out}
        };
        std::vector<size_t> sizes = {buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsF64 {
            uint32_t n_elements;
            uint32_t padding;
            double fill_value;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());
        push_constants.padding = 0;
        push_constants.fill_value = value;

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants);

        uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // Float16: use dedicated F16 fill shader with packed pairs
    if (input.dtype() == DType::Float16) {
        auto* f16_pipeline = getPipeline("fill_f16", device_id);
        size_t f16_buf_size = ((output.numel() + 1) / 2) * sizeof(uint32_t);
        std::vector<std::pair<uint32_t, const void*>> f16_bindings = {{0, buffer_out}};
        std::vector<size_t> f16_sizes = {f16_buf_size};
        VkDescriptorSet f16_ds = allocateAndWriteDescriptorSet(
            device_id, f16_pipeline, f16_bindings, f16_sizes);

        struct PushConstantsF16 {
            uint32_t n;
            uint32_t value_bits;
        } push_f16;
        push_f16.n = static_cast<uint32_t>(output.numel());
        // Convert float32 to float16 bits on CPU.
        //
        // Narrow value (double) to float FIRST before pulling bits — the
        // previous direct memcpy of 4 bytes from a double pulled the low
        // half of the IEEE-754 double layout, which is zero for typical
        // values and silently produced f16 = 0.0 for every non-special
        // fill value.
        float float_value = static_cast<float>(value);
        uint32_t f32_bits;
        std::memcpy(&f32_bits, &float_value, sizeof(uint32_t));
        uint32_t sign = (f32_bits >> 16) & 0x8000u;
        int32_t exponent = static_cast<int32_t>((f32_bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = (f32_bits >> 13) & 0x3FFu;
        uint16_t f16_val;
        if (exponent <= 0) f16_val = static_cast<uint16_t>(sign);
        else if (exponent >= 31) f16_val = static_cast<uint16_t>(sign | 0x7C00u);
        else f16_val = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
        push_f16.value_bits = f16_val;

        uint32_t num_pairs = static_cast<uint32_t>((output.numel() + 1) / 2);
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, f16_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               f16_pipeline->layout(), 0, 1, &f16_ds, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, f16_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsF16), &push_f16);
        vkCmdDispatch(cmdBuffer, div_wg(num_pairs, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    auto* pipeline = getPipeline("fill", device_id);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t n;
        uint32_t value_bits;
    } push_constants;

    push_constants.n = static_cast<uint32_t>(output.numel());

    // Convert value to bits based on dtype.
    //
    // `value` is a double (widened from float so Float64 subnormals survive
    // — same rationale as dispatchFull). The push-constant slot is 32 bits;
    // we must narrow to the target type FIRST, then memcpy. Directly
    // memcpy'ing 4 bytes of the double would copy the low half of the
    // IEEE-754 double bit pattern, which for most values (0.0, 1.0, 256.0,
    // 60000.0, …) is zero. That bug previously turned every Vulkan
    // dispatchFill(<float-tensor>, <non-trivial-value>) into a fill with
    // 0.0f and produced NaN downstream (e.g. BatchNorm2dMeanVar fills
    // `normalizer_tensor = batch * H * W = 256` and then divides
    // temp_sum / normalizer — with the bug normalizer was 0 so the whole
    // BatchNorm2d output became NaN).
    if (input.dtype() == DType::Int32) {
        int32_t int_value = static_cast<int32_t>(value);
        std::memcpy(&push_constants.value_bits, &int_value, sizeof(uint32_t));
    } else {
        // For float types, narrow to Float32 before copying the bits.
        float float_value = static_cast<float>(value);
        std::memcpy(&push_constants.value_bits, &float_value, sizeof(uint32_t));
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Full barrier: the filled buffer may be subsequently overwritten by a
    // transfer (e.g. sort's DeviceToDevice copy of slice data into the padded
    // work buffer).  insertComputeOnlyBarrier lacks TRANSFER_WRITE coverage.
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Clone tensor - deep copy via device-to-device buffer copy
 */
auto VulkanBackend::dispatchClone(const Tensor& input) -> Tensor {
    auto input_shape = input.shape();
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        return output;
    }

    // Use Vulkan's vkCmdCopyBuffer for efficient device-to-device copy
    size_t bytes = input.numel() * input.dtype_size();
    copy(output.data_ptr(), input.data_ptr(), bytes, CopyKind::DeviceToDevice);

    return output;
}

/**
 * @brief Expand tensor to larger size using broadcasting
 */
auto VulkanBackend::dispatchExpand(const Tensor& input_in, const std::vector<int64_t>& shape) -> Tensor {
    // Validate broadcast-compatibility up-front, matching cpu::expand_kernel.
    // Without this, expand(t({2,3}), {4,3}) silently produced a (4,3) tensor
    // full of garbage because the shader assumed the caller had already
    // checked shape compatibility.
    {
        auto in_shape = input_in.shape();
        int64_t ndim_out = static_cast<int64_t>(shape.size());
        int64_t ndim_in  = static_cast<int64_t>(in_shape.size());
        if (ndim_out < ndim_in) {
            throw std::runtime_error("expand: target has fewer dimensions than input");
        }
        int64_t dim_diff = ndim_out - ndim_in;
        for (int64_t i = 0; i < ndim_out; ++i) {
            int64_t in_idx = i - dim_diff;
            if (in_idx < 0) continue;  // New leading dim
            int64_t in_dim = in_shape[in_idx];
            int64_t tgt_dim = shape[i];
            if (in_dim != tgt_dim && in_dim != 1 && tgt_dim != -1) {
                throw std::runtime_error("expand: incompatible shapes — cannot expand dim "
                    + std::to_string(in_idx) + " of size " + std::to_string(in_dim)
                    + " to size " + std::to_string(tgt_dim));
            }
        }
    }

    // The shader below receives input strides computed from the input shape
    // (contiguous assumption). If the caller passed a non-contiguous view
    // (e.g. permute + unsqueeze) those computed strides would not match the
    // actual memory layout and the shader would read wrong elements.
    // Materialize to contiguous first, matching CPU semantics.
    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
    int32_t device_id = input.device().index;

    // Complex dtypes: expand by treating the complex element as two adjacent
    // real values. We recurse into the real-expand path on a "view" where each
    // complex element is a {…, 2} real pair. The underlying storage layout is
    // interleaved so expanding the real expanded tensor with an appended 2-dim
    // broadcasts both real and imaginary parts in lock-step.
    if (input.dtype() == DType::Complex64 || input.dtype() == DType::Complex128) {
        auto real_dtype = (input.dtype() == DType::Complex64) ? DType::Float32 : DType::Float64;
        auto in_shape_v = std::vector<int64_t>(input.shape().begin(), input.shape().end());
        in_shape_v.push_back(2);
        Tensor input_real(in_shape_v, real_dtype, input.device());
        size_t bytes = input.numel() * input.dtype_size();
        this->copy(input_real.data_ptr(), input.data_ptr(), bytes, CopyKind::DeviceToDevice);

        std::vector<int64_t> target_real_shape(shape.begin(), shape.end());
        target_real_shape.push_back(2);

        Tensor output_real = dispatchExpand(input_real, target_real_shape);

        Tensor output(shape, input.dtype(), input.device());
        size_t out_bytes = output.numel() * output.dtype_size();
        this->copy(output.data_ptr(), output_real.data_ptr(), out_bytes, CopyKind::DeviceToDevice);
        return output;
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    bool is_int64   = (input.dtype() == DType::Int64);
    bool is_u8      = (input.dtype() == DType::Bool ||
                       input.dtype() == DType::Int8 ||
                       input.dtype() == DType::UInt8);
    std::string shader_name;
    if (is_float64) {
        shader_name = "expand_f64";
    } else if (is_float16) {
        shader_name = "expand_f16";
    } else if (is_bfloat16) {
        shader_name = "expand_bf16";
    } else if (is_int64) {
        shader_name = "expand_i64";
    } else if (is_u8) {
        shader_name = "expand_u8";
    } else {
        shader_name = "expand";
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor with new shape
    Tensor output(shape, input.dtype(), input.device());

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();
    // For Float16 / BFloat16, the shader works with uint32 (packed pairs),
    // so descriptor size needs to cover the full uint32 reads/writes.
    const bool is_packed_half = (is_float16 || is_bfloat16);
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_packed_half) {
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }
    if (is_u8) {
        // 1-byte storage must round up to 4-byte boundary so robust buffer
        // access doesn't zero out the trailing bytes of the last word.
        buffer_size_in  = (buffer_size_in  + 3u) & ~size_t(3);
        buffer_size_out = (buffer_size_out + 3u) & ~size_t(3);
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate strides for input tensor
    auto input_shape = input.shape();
    std::vector<uint32_t> input_strides(input_shape.size());
    uint32_t stride = 1;
    for (int i = static_cast<int>(input_shape.size()) - 1; i >= 0; i--) {
        input_strides[i] = stride;
        stride *= static_cast<uint32_t>(input_shape[i]);
    }

    // Prepare push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t input_ndim;
        uint32_t output_ndim;
        uint32_t input_shape[8];
        uint32_t output_shape[8];
        uint32_t input_strides[8];
    } push_constants = {}; // Zero-initialize all fields including arrays

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.input_ndim = static_cast<uint32_t>(input_shape.size());
    push_constants.output_ndim = static_cast<uint32_t>(shape.size());

    for (size_t i = 0; i < input_shape.size() && i < 8; i++) {
        push_constants.input_shape[i] = static_cast<uint32_t>(input_shape[i]);
        push_constants.input_strides[i] = input_strides[i];
    }
    for (size_t i = 0; i < shape.size() && i < 8; i++) {
        push_constants.output_shape[i] = static_cast<uint32_t>(shape[i]);
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Packed-pair shaders process 2 elements per thread
    uint32_t num_items = is_packed_half
        ? static_cast<uint32_t>((output.numel() + 1) / 2)
        : static_cast<uint32_t>(output.numel());
    uint32_t workgroups = div_wg(num_items, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Concatenate N tensors along a dimension
 *
 * This implementation uses Vulkan buffer copy operations to concatenate
 * tensors without requiring dynamic descriptor sets. Each input tensor
 * is copied to its appropriate region in the output buffer.
 */
auto VulkanBackend::dispatchCat(const std::vector<Tensor>& inputs, int64_t dim) -> Tensor {
    if (inputs.empty()) {
        throw std::invalid_argument("VulkanBackend::dispatchCat requires at least 1 input tensor");
    }

    // Special case: single tensor just clone it
    if (inputs.size() == 1) {
        return dispatchClone(inputs[0]);
    }

    // IMPORTANT: Make all inputs contiguous
    // Sliced tensors (like those from roll operation) are not contiguous
    // and have different strides/offsets that don't work with simple buffer copying
    std::vector<Tensor> contiguous_inputs;
    contiguous_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        if (!input.is_contiguous()) {
            contiguous_inputs.push_back(dispatchContiguous(input));
        } else {
            contiguous_inputs.push_back(input);
        }
    }

    const Tensor& first_input = contiguous_inputs[0];
    int32_t device_id = first_input.device().index;
    auto first_shape = first_input.shape();
    size_t ndim = first_shape.size();

    // Normalize dimension
    if (dim < 0) {
        dim += static_cast<int64_t>(ndim);
    }
    if (dim < 0 || dim >= static_cast<int64_t>(ndim)) {
        throw std::invalid_argument("Invalid concatenation dimension");
    }

    // Validate all tensors have compatible shapes and calculate output shape
    std::vector<int64_t> output_shape(first_shape.begin(), first_shape.end());
    int64_t total_dim_size = first_shape[dim];

    for (size_t i = 1; i < contiguous_inputs.size(); ++i) {
        auto shape = contiguous_inputs[i].shape();
        if (shape.size() != ndim) {
            throw std::invalid_argument("All input tensors must have the same number of dimensions");
        }
        for (size_t j = 0; j < ndim; ++j) {
            if (j != static_cast<size_t>(dim) && shape[j] != first_shape[j]) {
                throw std::invalid_argument("All input tensors must have the same shape except along concatenation dimension");
            }
        }
        total_dim_size += shape[dim];
    }

    output_shape[dim] = total_dim_size;

    // Create output tensor
    Tensor output(output_shape, first_input.dtype(), first_input.device());

    // Handle empty output tensor - no GPU work needed
    int64_t out_numel = 1;
    for (auto d : output_shape) out_numel *= d;
    if (out_numel == 0) {
        return output;
    }

    auto [vk_buffer_out, buffer_out_offset] = getVulkanBufferAndOffset(output.data_ptr());
    size_t element_size = first_input.dtype_size();

    // Calculate strides for copying
    // outer_size: number of "blocks" before the cat dimension
    // inner_size: size of each contiguous chunk within the cat dimension
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= first_shape[i];
    }

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < ndim; i++) {
        inner_size *= first_shape[i];
    }

    // Begin command buffer for all copy operations
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);

    // Copy each input tensor to the appropriate location in output
    int64_t current_offset_in_cat_dim = 0;

    for (const auto& input : contiguous_inputs) {
        int64_t input_dim_size = input.shape()[dim];

        // Skip empty inputs (0-element tensors have null data_ptr)
        if (input.numel() == 0) {
            current_offset_in_cat_dim += input_dim_size;
            continue;
        }

        auto [buffer_in, buffer_in_base_offset] = getVulkanBufferAndOffset(input.data_ptr());

        // For each outer block, copy the input data to the correct position
        for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
            // Calculate source and destination offsets
            // buffer_in_base_offset accounts for slice view offset into the storage buffer
            int64_t src_offset = outer_idx * input_dim_size * inner_size * element_size +
                                static_cast<int64_t>(buffer_in_base_offset);
            int64_t dst_offset = outer_idx * total_dim_size * inner_size * element_size +
                                current_offset_in_cat_dim * inner_size * element_size +
                                static_cast<int64_t>(buffer_out_offset);

            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = static_cast<VkDeviceSize>(src_offset);
            copyRegion.dstOffset = static_cast<VkDeviceSize>(dst_offset);
            copyRegion.size = static_cast<VkDeviceSize>(input_dim_size * inner_size * element_size);

            vkCmdCopyBuffer(cmdBuffer, buffer_in, vk_buffer_out, 1, &copyRegion);
        }

        current_offset_in_cat_dim += input_dim_size;
    }

    // Add a memory barrier to ensure all copies complete before any subsequent operations
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Clamp tensor values to [min, max] range
 */
auto VulkanBackend::dispatchClamp(const Tensor& input, double min_value, double max_value) -> Tensor {
    // Handle empty tensors - no work to do
    if (input.numel() == 0) {
        auto input_shape = input.shape();
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
        return Tensor(output_shape, input.dtype(), input.device());
    }

    // Integer clamp: dedicated by-byte-width shaders (the float clamp shaders
    // would reinterpret integer storage as float and corrupt it). ±inf bounds
    // (from clamp_min / clamp_max) map to the dtype's representable extremes.
    {
        const DType dt = input.dtype();
        int elem_bytes = 0;
        bool is_signed = false;
        int64_t tmin = 0, tmax = 0;
        bool is_int = true;
        switch (dt) {
            case DType::Int8:   elem_bytes=1; is_signed=true;  tmin=-128; tmax=127; break;
            case DType::UInt8:  elem_bytes=1; is_signed=false; tmin=0; tmax=255; break;
            case DType::Int16:  elem_bytes=2; is_signed=true;  tmin=-32768; tmax=32767; break;
            case DType::UInt16: elem_bytes=2; is_signed=false; tmin=0; tmax=65535; break;
            case DType::Int32:  elem_bytes=4; is_signed=true;  tmin=INT32_MIN; tmax=INT32_MAX; break;
            case DType::UInt32: elem_bytes=4; is_signed=false; tmin=0; tmax=UINT32_MAX; break;
            case DType::Int64:  elem_bytes=8; is_signed=true;  tmin=INT64_MIN; tmax=INT64_MAX; break;
            case DType::UInt64: elem_bytes=8; is_signed=false;
                                tmin=0; tmax=static_cast<int64_t>(UINT64_MAX); break;
            default: is_int = false; break;
        }
        if (is_int) {
            auto cast_bound = [&](double v) -> int64_t {
                switch (dt) {
                    case DType::Int8:   return static_cast<int64_t>(static_cast<int8_t>(v));
                    case DType::UInt8:  return static_cast<int64_t>(static_cast<uint8_t>(v));
                    case DType::Int16:  return static_cast<int64_t>(static_cast<int16_t>(v));
                    case DType::UInt16: return static_cast<int64_t>(static_cast<uint16_t>(v));
                    case DType::Int32:  return static_cast<int64_t>(static_cast<int32_t>(v));
                    case DType::UInt32: return static_cast<int64_t>(static_cast<uint32_t>(v));
                    case DType::Int64:  return static_cast<int64_t>(v);
                    default:            return static_cast<int64_t>(static_cast<uint64_t>(v));
                }
            };
            const int64_t lo64 = std::isinf(min_value) ? (min_value < 0 ? tmin : tmax)
                                                       : cast_bound(min_value);
            const int64_t hi64 = std::isinf(max_value) ? (max_value > 0 ? tmax : tmin)
                                                       : cast_bound(max_value);

            const int32_t device_id = input.device().index;
            const std::string shader = elem_bytes == 1 ? "clamp_int8"
                                     : elem_bytes == 2 ? "clamp_int16"
                                     : elem_bytes == 4 ? "clamp_int32"
                                                       : "clamp_int64";
            auto* pipeline = getPipeline(shader, device_id);

            std::vector<int64_t> oshape(input.shape().begin(), input.shape().end());
            Tensor output(oshape, input.dtype(), input.device());
            const int64_t numel = input.numel();
            const size_t buf = ((static_cast<size_t>(numel) * elem_bytes + 3) / 4) * 4;

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, input.data_ptr()}, {1, output.data_ptr()}};
            std::vector<size_t> sizes = {buf, buf};
            VkDescriptorSet descriptorSet =
                allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
            if (elem_bytes == 8) {
                struct PC64 { uint32_t n; uint32_t is_signed; int64_t lo; int64_t hi; } pc;
                pc.n = static_cast<uint32_t>(numel);
                pc.is_signed = is_signed ? 1u : 0u;
                pc.lo = lo64; pc.hi = hi64;
                vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            } else {
                struct PC32 { uint32_t n; uint32_t is_signed; int32_t lo; int32_t hi; } pc;
                pc.n = static_cast<uint32_t>(numel);
                pc.is_signed = is_signed ? 1u : 0u;
                pc.lo = static_cast<int32_t>(lo64);
                pc.hi = static_cast<int32_t>(hi64);
                vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            }
            const int64_t units = elem_bytes == 1 ? (numel + 3) / 4
                                : elem_bytes == 2 ? (numel + 1) / 2
                                                  : numel;
            uint32_t workgroups = div_wg(static_cast<uint32_t>(units),
                                         devices_[device_id].workgroupSize);
            vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
            insertComputeOnlyBarrier(cmdBuffer);
            endSingleTimeCommands(cmdBuffer, device_id);
            return output;
        }
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    // BF16 has its own dedicated shader — without this branch BF16 buffers
    // would fall through to the generic 32-bit `clamp` shader which smears
    // two BF16 slots per store and returns garbage like [0, v, 0, v].
    std::string shader_name =
        is_float64 ? "clamp_f64"
        : is_float16 ? "clamp_f16"
        : is_bfloat16 ? "clamp_bf16"
        : "clamp";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    auto input_shape = input.shape();
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();
    size_t buffer_size_in, buffer_size_out;
    if (is_float16 || is_bfloat16) {
        // F16 / BF16 are packed 2 elements per uint32 — round up to 4-byte boundary
        size_t num_pairs_in = (input.numel() + 1) / 2;
        size_t num_pairs_out = (output.numel() + 1) / 2;
        buffer_size_in = num_pairs_in * 4;
        buffer_size_out = num_pairs_out * 4;
    } else {
        buffer_size_in = input.numel() * input.dtype_size();
        buffer_size_out = output.numel() * output.dtype_size();
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    if (is_float64) {
        // Float64 push constants with double min/max values
        struct PushConstantsF64 {
            uint32_t n_elements;
            uint32_t padding;
            double min_value;
            double max_value;
        } push_constants_f64;

        push_constants_f64.n_elements = static_cast<uint32_t>(output.numel());
        push_constants_f64.padding = 0;
        push_constants_f64.min_value = static_cast<double>(min_value);
        push_constants_f64.max_value = static_cast<double>(max_value);

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants_f64);
    } else {
        // Float32 and Float16 push constants (same layout: n, min, max as float)
        struct PushConstants {
            uint32_t n_elements;
            float min_value;
            float max_value;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());
        push_constants.min_value = min_value;
        push_constants.max_value = max_value;

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);
    }

    // Float16 shader processes pairs of elements
    uint32_t workgroups;
    if (is_float16) {
        uint32_t num_pairs = (static_cast<uint32_t>(output.numel()) + 1) / 2;
        workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);
    } else {
        workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Dispatch forward activation operations (element-wise)
 * Operations: 0=relu, 1=sigmoid, 2=tanh, 3=gelu, 4=leaky_relu, 5=swish
 */
auto VulkanBackend::dispatchActivation([[maybe_unused]] const std::string& op_name,
                                        const Tensor& input_raw,
                                        uint32_t opcode,
                                        double param) -> Tensor {
    // audit-2026-05-03 bug #15 mirror: ensure contiguous input.
    auto input = input_raw.contiguous();
    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        auto input_shape = input.shape();
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
        return Tensor(output_shape, input.dtype(), input.device());
    }

    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "activations_f64";
    } else if (is_float16) {
        shader_name = "activations_f16";
    } else if (is_bfloat16) {
        shader_name = "activations_bf16";
    } else {
        shader_name = "activations";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    auto shape = input.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Prepare push constants - use different structure for Float32 vs Float64
    struct PushConstantsF32 {
        uint32_t n;
        uint32_t activation;
        float alpha;
    };
    struct PushConstantsF64 {
        uint32_t n;
        uint32_t activation;
        double alpha;
    };

    PushConstantsF32 push_constants_f32;
    PushConstantsF64 push_constants_f64;
    void* push_constants_ptr;
    size_t push_constants_size;

    if (is_float64) {
        push_constants_f64.n = static_cast<uint32_t>(input.numel());
        push_constants_f64.activation = opcode;
        push_constants_f64.alpha = static_cast<double>(param);
        push_constants_ptr = &push_constants_f64;
        push_constants_size = sizeof(PushConstantsF64);
    } else {
        push_constants_f32.n = static_cast<uint32_t>(input.numel());
        push_constants_f32.activation = opcode;
        push_constants_f32.alpha = param;
        push_constants_ptr = &push_constants_f32;
        push_constants_size = sizeof(PushConstantsF32);
    }

    // Get VkBuffer handles
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    // Setup descriptor set
    // Binding 0: input, Binding 1: output
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, push_constants_size, push_constants_ptr);

    // Dispatch compute workgroups
    // For Float16, shader processes 2 elements per thread (packed pairs)
    uint32_t num_elements = static_cast<uint32_t>(input.numel());
    uint32_t workgroups;
    if (is_float16) {
        // Each thread handles 2 elements (pair), 256 threads per workgroup
        uint32_t num_pairs = (num_elements + 1) / 2;
        workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);
    } else {
        workgroups = div_wg(num_elements, devices_[device_id].workgroupSize);
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Dispatch backward activation operations (element-wise)
 * Operations: 0=relu, 1=sigmoid, 2=tanh, 3=leaky_relu, 4=gelu
 */
auto VulkanBackend::dispatchActivationBackward([[maybe_unused]] const std::string& op_name,
                                                const Tensor& grad_output_raw,
                                                const Tensor& input_or_output_raw,
                                                uint32_t opcode,
                                                double param) -> Tensor {
    // audit-2026-05-03 bug #15 mirror: ensure contiguous inputs.
    auto grad_output = grad_output_raw.contiguous();
    auto input_or_output = input_or_output_raw.contiguous();
    int32_t device_id = grad_output.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    bool is_float16 = (grad_output.dtype() == DType::Float16);
    bool is_bfloat16 = (grad_output.dtype() == DType::BFloat16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "activations_backward_f64";
    } else if (is_float16) {
        shader_name = "activations_backward_f16";
    } else if (is_bfloat16) {
        shader_name = "activations_backward_bf16";
    } else {
        shader_name = "activations_backward";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    auto shape = grad_output.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor grad_input(output_shape, grad_output.dtype(), grad_output.device());

    // Prepare push constants — must match the shader's layout exactly.
    // The f64 backward shader declares `double alpha`, the others declare
    // `float alpha`. Picking the wrong struct silently truncates alpha to
    // a Float32 value (visible as ~1e-9 error on Float64 LeakyReLU/ELU
    // backward at alpha=0.1) or worse, misaligns the read.
    struct PushConstantsF32 {
        uint32_t n;
        uint32_t op;
        float alpha;
    };
    struct PushConstantsF64 {
        uint32_t n;
        uint32_t op;
        double alpha;
    };

    PushConstantsF32 push_constants_f32;
    PushConstantsF64 push_constants_f64;
    void* push_constants_ptr;
    size_t push_constants_size;
    if (is_float64) {
        push_constants_f64.n = static_cast<uint32_t>(grad_output.numel());
        push_constants_f64.op = opcode;
        push_constants_f64.alpha = param;
        push_constants_ptr = &push_constants_f64;
        push_constants_size = sizeof(PushConstantsF64);
    } else {
        push_constants_f32.n = static_cast<uint32_t>(grad_output.numel());
        push_constants_f32.op = opcode;
        push_constants_f32.alpha = static_cast<float>(param);
        push_constants_ptr = &push_constants_f32;
        push_constants_size = sizeof(PushConstantsF32);
    }

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_input_or_output = input_or_output.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_input_or_output = input_or_output.numel() * input_or_output.dtype_size();
    size_t buffer_size_grad_in = grad_input.numel() * grad_input.dtype_size();
    if (is_float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t go_pairs = (grad_output.numel() + 1) / 2;
        size_t io_pairs = (input_or_output.numel() + 1) / 2;
        size_t gi_pairs = (grad_input.numel() + 1) / 2;
        buffer_size_grad_out = go_pairs * 4;
        buffer_size_input_or_output = io_pairs * 4;
        buffer_size_grad_in = gi_pairs * 4;
    }

    // Setup descriptor set
    // Binding 0: grad_output, Binding 1: input_or_output, Binding 2: grad_input
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_input_or_output},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_input_or_output, buffer_size_grad_in};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, push_constants_size, push_constants_ptr);

    // Dispatch compute workgroups
    // For Float16, each thread processes 2 elements (pairs)
    uint32_t num_threads = is_float16 ? ((grad_output.numel() + 1) / 2) : grad_output.numel();
    uint32_t workgroups = div_wg(num_threads, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

/**
 * @brief Dispatch swish backward operation
 * Formula: swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
 * where swish(x) = x * sigmoid(x)
 */
auto VulkanBackend::dispatchSwishBackward(const Tensor& grad_output,
                                           const Tensor& input) -> Tensor {
    int32_t device_id = grad_output.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    bool is_float16 = (grad_output.dtype() == DType::Float16);
    std::string shader_name = is_float64 ? "swish_backward_f64" : (is_float16 ? "swish_backward_f16" : "swish_backward");
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    auto shape = grad_output.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor grad_input(output_shape, grad_output.dtype(), grad_output.device());

    // Prepare push constants
    struct PushConstants {
        uint32_t n;  // Number of elements
    } push_constants;

    push_constants.n = static_cast<uint32_t>(grad_output.numel());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_input = input.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_grad_in = grad_input.numel() * grad_input.dtype_size();

    // Setup descriptor set
    // Binding 0: grad_output, Binding 1: input, Binding 2: grad_input
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_input},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_input, buffer_size_grad_in};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch compute workgroups
    uint32_t workgroups = div_wg(grad_output.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

/**
 * @brief Dispatch RReLU (Randomized Leaky ReLU) forward
 * Eval:  output = (x >= 0) ? x : slope * x  where slope = (lower + upper) / 2
 * Train: output = (x >= 0) ? x : random_slope * x  with PCG random per element
 */
auto VulkanBackend::dispatchRReLU(const Tensor& input,
                                   float lower, float upper,
                                   bool training) -> Tensor {
    // Handle empty tensors
    if (input.numel() == 0) {
        auto s = input.shape();
        return Tensor(std::vector<int64_t>(s.begin(), s.end()), input.dtype(), input.device());
    }

    // Non-Float32: upcast, compute, downcast
    if (input.dtype() != DType::Float32) {
        auto f32 = input.to(DType::Float32);
        auto result = dispatchRReLU(f32, lower, upper, training);
        return result.to(input.dtype());
    }

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("rrelu", device_id);

    auto shape = input.shape();
    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

    struct PushConstants {
        uint32_t num_elements;
        float lower;
        float upper;
        uint32_t training;
        uint32_t seed;
    } pc;

    pc.num_elements = static_cast<uint32_t>(input.numel());
    pc.lower = lower;
    pc.upper = upper;
    pc.training = training ? 1u : 0u;
    pc.seed = training ? static_cast<uint32_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() & 0xFFFFFFFF) : 0u;

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &pc);

    uint32_t workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Dispatch RReLU backward
 * output = (input >= 0) ? grad : slope * grad
 */
auto VulkanBackend::dispatchRReLUBackward(const Tensor& grad_output,
                                           const Tensor& input,
                                           float slope) -> Tensor {
    // Handle empty tensors
    if (grad_output.numel() == 0) {
        auto s = grad_output.shape();
        return Tensor(std::vector<int64_t>(s.begin(), s.end()), grad_output.dtype(), grad_output.device());
    }

    // Non-Float32: upcast, compute, downcast
    if (grad_output.dtype() != DType::Float32) {
        auto f32_g = grad_output.to(DType::Float32);
        auto f32_in = input.to(DType::Float32);
        auto result = dispatchRReLUBackward(f32_g, f32_in, slope);
        return result.to(grad_output.dtype());
    }

    int32_t device_id = grad_output.device().index;
    auto* pipeline = getPipeline("rrelu_backward", device_id);

    auto shape = grad_output.shape();
    Tensor grad_input(std::vector<int64_t>(shape.begin(), shape.end()), grad_output.dtype(), grad_output.device());

    struct PushConstants {
        uint32_t num_elements;
        float slope;
    } pc;

    pc.num_elements = static_cast<uint32_t>(grad_output.numel());
    pc.slope = slope;

    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_input = input.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_grad_in = grad_input.numel() * grad_input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_input},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_input, buffer_size_grad_in};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &pc);

    uint32_t workgroups = div_wg(grad_output.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

/**
 * @brief Dispatch LogSigmoid backward
 * output = grad * sigmoid(-x)
 * where sigmoid(-x) = (x >= 0) ? exp(-x)/(1+exp(-x)) : 1/(1+exp(x))
 */
auto VulkanBackend::dispatchLogSigmoidBackward(const Tensor& grad_output,
                                                const Tensor& input) -> Tensor {
    // Handle empty tensors
    if (grad_output.numel() == 0) {
        auto s = grad_output.shape();
        return Tensor(std::vector<int64_t>(s.begin(), s.end()), grad_output.dtype(), grad_output.device());
    }

    // Non-Float32: upcast, compute, downcast
    if (grad_output.dtype() != DType::Float32) {
        auto f32_g = grad_output.to(DType::Float32);
        auto f32_in = input.to(DType::Float32);
        auto result = dispatchLogSigmoidBackward(f32_g, f32_in);
        return result.to(grad_output.dtype());
    }

    int32_t device_id = grad_output.device().index;
    auto* pipeline = getPipeline("log_sigmoid_backward", device_id);

    auto shape = grad_output.shape();
    Tensor grad_input(std::vector<int64_t>(shape.begin(), shape.end()), grad_output.dtype(), grad_output.device());

    struct PushConstants {
        uint32_t num_elements;
    } pc;

    pc.num_elements = static_cast<uint32_t>(grad_output.numel());

    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_input = input.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_grad_in = grad_input.numel() * grad_input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_input},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_input, buffer_size_grad_in};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &pc);

    uint32_t workgroups = div_wg(grad_output.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

/**
 * @brief Dispatch softmax backward operation
 * Formula: grad_input = output * (grad_output - dot(grad_output, output))
 */
auto VulkanBackend::dispatchSoftmaxBackward(const Tensor& grad_output,
                                             const Tensor& output,
                                             int64_t dim) -> Tensor {
    int32_t device_id = grad_output.device().index;

    // For Float16, upcast to Float32 for numerical stability
    if (grad_output.dtype() == DType::Float16) {
        DType orig_dtype = grad_output.dtype();
        auto grad_f32 = grad_output.to(DType::Float32);
        auto out_f32 = output.to(DType::Float32);
        auto result_f32 = dispatchSoftmaxBackward(grad_f32, out_f32, dim);
        return result_f32.to(orig_dtype);
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    bool is_bfloat16 = (grad_output.dtype() == DType::BFloat16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "softmax_backward_f64";
    } else if (is_bfloat16) {
        shader_name = "softmax_backward_bf16";
    } else {
        shader_name = "softmax_backward";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape = output.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    // Create output tensor
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor grad_input(output_shape, grad_output.dtype(), grad_output.device());

    // Calculate dimension strides
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }
    int64_t dim_size = shape[dim];
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        inner_size *= shape[i];
    }

    // Prepare push constants
    struct PushConstants {
        uint32_t outer_size;
        uint32_t dim_size;
        uint32_t inner_size;
    } push_constants;

    push_constants.outer_size = static_cast<uint32_t>(outer_size);
    push_constants.dim_size = static_cast<uint32_t>(dim_size);
    push_constants.inner_size = static_cast<uint32_t>(inner_size);

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size = grad_output.numel() * grad_output.dtype_size();

    // Setup descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_output},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch: one thread per (outer, inner) position
    uint32_t total_threads = static_cast<uint32_t>(outer_size * inner_size);
    uint32_t workgroups = div_wg(total_threads, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

/**
 * @brief Dispatch log_softmax backward operation
 * Formula: grad_input = grad_output - exp(output) * sum(grad_output)
 */
auto VulkanBackend::dispatchLogSoftmaxBackward(const Tensor& grad_output,
                                                const Tensor& output,
                                                int64_t dim) -> Tensor {
    int32_t device_id = grad_output.device().index;

    // Select correct pipeline based on dtype
    std::string shader_name = "log_softmax_backward";
    if (grad_output.dtype() == DType::Float64) shader_name = "log_softmax_backward_f64";
    else if (grad_output.dtype() == DType::Float16) shader_name = "log_softmax_backward_f16";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape = output.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    // Create output tensor
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor grad_input(output_shape, grad_output.dtype(), grad_output.device());

    // Calculate dimension strides
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }
    int64_t dim_size = shape[dim];
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        inner_size *= shape[i];
    }

    // Prepare push constants
    struct PushConstants {
        uint32_t outer_size;
        uint32_t dim_size;
        uint32_t inner_size;
    } push_constants;

    push_constants.outer_size = static_cast<uint32_t>(outer_size);
    push_constants.dim_size = static_cast<uint32_t>(dim_size);
    push_constants.inner_size = static_cast<uint32_t>(inner_size);

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size = grad_output.numel() * grad_output.dtype_size();

    // Setup descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_output},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch: one thread per (outer, inner) position
    uint32_t total_threads = static_cast<uint32_t>(outer_size * inner_size);
    uint32_t workgroups = div_wg(total_threads, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

} // namespace tenzor
