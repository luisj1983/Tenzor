/**
 * @file vulkan_ops_misc.cpp
 * @brief Vulkan backend miscellaneous operations (Full, Ones, Rand,
 *        Randint, Repeat, MaskedSelect, MaskedFill, Where, Cast, Slice, Split, Chunk,
 *        Flatten, typed dispatch wrappers, Phase 11.5 misc, ScatterAdd,
 *        IndexAdd, IndexCopy, IndexFill)
 */

#include "vulkan_ops_common.hpp"
#include "tenzor/ops/creation.hpp"  // for tenzor::get_global_seed

namespace tenzor {
// ============================================================================

auto VulkanBackend::dispatchFull(const std::vector<int64_t>& shape, float value, DType dtype) -> Tensor {
    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    // Handle empty tensors - no GPU work needed
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    if (numel == 0) {
        return output;
    }

    int32_t device_id = device.index;

    // Select shader based on dtype
    bool is_float64 = (dtype == DType::Float64);
    bool is_float16 = (dtype == DType::Float16);
    bool is_bfloat16 = (dtype == DType::BFloat16);
    bool is_int8 = (dtype == DType::Int8);
    bool is_uint8 = (dtype == DType::UInt8);
    bool is_int64 = (dtype == DType::Int64);
    bool is_bool = (dtype == DType::Bool);
    std::string shader_name;
    if (is_float64) {
        shader_name = "full_f64";
    } else if (is_float16) {
        shader_name = "full_f16";
    } else if (is_bfloat16) {
        shader_name = "full_bf16";
    } else if (is_int8) {
        shader_name = "full_i8";
    } else if (is_uint8 || is_bool) {
        // Bool is stored as uint8_t, so use the same shader
        shader_name = "full_uint8";
    } else if (is_int64) {
        shader_name = "full_i64";
    } else {
        shader_name = "full";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    const void* buffer_out = output.data_ptr();
    // For Float16, the shader works with uint32 (packed pairs), so descriptor size needs
    // to cover the full uint32 writes. Even for 1 element, shader writes a full uint32.
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        // Round up to 4-byte boundary (minimum uint32 size for shader access)
        size_t num_pairs = (output.numel() + 1) / 2;
        buffer_size_out = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Use different push constants structure based on dtype
    if (is_float64) {
        struct PushConstantsF64 {
            uint32_t n_elements;
            uint32_t padding;  // Alignment padding
            double fill_value;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());
        push_constants.padding = 0;
        push_constants.fill_value = static_cast<double>(value);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants);

        uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // Float16 uses a dedicated shader with half-precision packing
    if (is_float16) {
        struct PushConstantsF16 {
            uint32_t n_elements;
            uint32_t fill_value_f16;  // Float16 bits in lower 16 bits
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());

        // Convert float to Float16 bits
        Float16 f16_value(value);
        push_constants.fill_value_f16 = static_cast<uint32_t>(f16_value.bits);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF16), &push_constants);

        // Each thread handles 2 float16 elements, so we need half the workgroups
        uint32_t num_pairs = (output.numel() + 1) / 2;
        uint32_t workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // Int8 uses a dedicated shader that packs 4 elements per uint32
    if (is_int8) {
        struct PushConstantsI8 {
            uint32_t n_elements;
            uint32_t fill_value_i8;  // Int8 bits in lower 8 bits
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());
        // Convert float value to int8 and store in lower 8 bits
        int8_t i8_value = static_cast<int8_t>(value);
        push_constants.fill_value_i8 = static_cast<uint32_t>(static_cast<uint8_t>(i8_value));

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsI8), &push_constants);

        // Each thread handles 4 int8 elements, so we need 1/4 the workgroups
        uint32_t num_quads = (output.numel() + 3) / 4;
        uint32_t workgroups = div_wg(num_quads, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // UInt8 and Bool use a dedicated shader with direct byte access via 8-bit storage extension
    // Bool is stored as uint8_t (0 = false, non-zero = true)
    if (is_uint8 || is_bool) {
        struct PushConstantsUInt8 {
            uint32_t n_elements;
            uint32_t fill_value_uint8;  // UInt8 bits in lower 8 bits
        } push_constants_u8;

        push_constants_u8.n_elements = static_cast<uint32_t>(output.numel());
        // Convert float value to uint8 and store in lower 8 bits
        // For Bool: any non-zero value becomes 1 (true)
        uint8_t u8_value = is_bool ? (value != 0.0f ? 1 : 0) : static_cast<uint8_t>(value);
        push_constants_u8.fill_value_uint8 = static_cast<uint32_t>(u8_value);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsUInt8), &push_constants_u8);

        // Each thread handles 1 uint8 element
        uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // Int64 uses a dedicated shader with 64-bit value split into two 32-bit parts
    if (is_int64) {
        struct PushConstantsI64 {
            uint32_t n_elements;
            uint32_t value_low;   // Low 32 bits of int64
            uint32_t value_high;  // High 32 bits of int64
        } push_constants_i64;

        push_constants_i64.n_elements = static_cast<uint32_t>(output.numel());
        // Convert float value to int64 and split into two uint32
        int64_t i64_value = static_cast<int64_t>(value);
        push_constants_i64.value_low = static_cast<uint32_t>(i64_value & 0xFFFFFFFF);
        push_constants_i64.value_high = static_cast<uint32_t>((static_cast<uint64_t>(i64_value) >> 32) & 0xFFFFFFFF);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsI64), &push_constants_i64);

        uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    struct PushConstants {
        uint32_t n_elements;
        uint32_t fill_value_bits;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());

    // Convert value to bits based on dtype
    if (dtype == DType::Int32) {
        int32_t int_value = static_cast<int32_t>(value);
        std::memcpy(&push_constants.fill_value_bits, &int_value, sizeof(uint32_t));
    } else if (dtype == DType::BFloat16) {
        // BFloat16 is the upper 16 bits of float32 with round-to-nearest-even
        uint32_t float_bits;
        std::memcpy(&float_bits, &value, sizeof(uint32_t));
        uint32_t bf16_bits = (float_bits + 0x7FFFu + ((float_bits >> 16) & 1u)) >> 16;
        push_constants.fill_value_bits = bf16_bits;
    } else {
        // For float types, use the float bits directly
        std::memcpy(&push_constants.fill_value_bits, &value, sizeof(uint32_t));
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

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Ones Operation - Create tensor filled with 1.0
// ============================================================================

auto VulkanBackend::dispatchOnes(const std::vector<int64_t>& shape, DType dtype) -> Tensor {
    // For Int64, UInt8, Bool, or BFloat16, use full() instead since ones shader only supports 32-bit values
    // (BFloat16 now has a native full_bf16 shader, but we still route through full() for consistency)
    if (dtype == DType::Int64 || dtype == DType::UInt8 || dtype == DType::Bool || dtype == DType::BFloat16) {
        return dispatchFull(shape, 1.0, dtype);
    }

    // Float64 uses dedicated ones_f64 shader
    if (dtype == DType::Float64) {
        Device device(Device::Type::Vulkan, 0);
        Tensor output(shape, dtype, device);
        int32_t device_id = device.index;

        auto* pipeline = getPipeline("ones_f64", device_id);

        const void* buffer_out = output.data_ptr();
        size_t buffer_size_out = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buffer_out}};
        std::vector<size_t> sizes = {buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsF64 {
            uint32_t n_elements;
        } push_constants;
        push_constants.n_elements = static_cast<uint32_t>(output.numel());

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

    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    int32_t device_id = device.index;

    // Select shader based on dtype
    bool is_float16 = (dtype == DType::Float16);
    bool is_int8 = (dtype == DType::Int8);
    std::string shader_name = is_float16 ? "ones_f16" : (is_int8 ? "ones_i8" : "ones");

    auto* pipeline = getPipeline(shader_name, device_id);

    const void* buffer_out = output.data_ptr();
    // For Float16, the shader works with uint32 (packed pairs), so descriptor size needs
    // to cover the full uint32 writes. Even for 1 element, shader writes a full uint32.
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        // Round up to 4-byte boundary (minimum uint32 size for shader access)
        size_t num_pairs = (output.numel() + 1) / 2;
        buffer_size_out = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Float16 uses a different shader that only needs n_elements
    if (is_float16) {
        struct PushConstantsF16 {
            uint32_t n_elements;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF16), &push_constants);

        // Each thread handles 2 float16 elements, so we need half the workgroups
        uint32_t num_pairs = (output.numel() + 1) / 2;
        uint32_t workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);

        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // Int8 uses a different shader that packs 4 elements per uint32
    if (is_int8) {
        struct PushConstantsI8 {
            uint32_t n_elements;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsI8), &push_constants);

        // Each thread handles 4 int8 elements, so we need 1/4 the workgroups
        uint32_t num_quads = (output.numel() + 3) / 4;
        uint32_t workgroups = div_wg(num_quads, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    struct PushConstants {
        uint32_t n_elements;
        uint32_t one_value_bits;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());

    // Convert value 1.0 or 1 to bits based on dtype
    if (dtype == DType::Int32) {
        int32_t int_value = 1;
        std::memcpy(&push_constants.one_value_bits, &int_value, sizeof(uint32_t));
    } else if (dtype == DType::Int64) {
        int32_t int_value = 1;
        std::memcpy(&push_constants.one_value_bits, &int_value, sizeof(uint32_t));
    } else {
        // For float types, use 1.0f
        float float_value = 1.0f;
        std::memcpy(&push_constants.one_value_bits, &float_value, sizeof(uint32_t));
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

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchRand(const std::vector<int64_t>& shape, DType dtype) -> Tensor {
    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    size_t numel = output.numel();
    if (numel == 0) {
        return output;
    }

    // Use GPU Philox RNG for random number generation
    int32_t device_id = device.index;

    // Select shader based on dtype
    std::string shader_name = "random";
    if (dtype == DType::Float64) {
        shader_name = "random_f64";
    } else if (dtype == DType::Float16) {
        shader_name = "random_f16";
    } else if (dtype == DType::BFloat16) {
        shader_name = "random_bf16";
    } else if (dtype != DType::Float32) {
        throw std::runtime_error("Unsupported dtype for rand: only Float32, Float64, Float16, and BFloat16 are supported");
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    const void* buffer_out = output.data_ptr();
    size_t buffer_size = numel * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buffer_out}};
    std::vector<size_t> sizes = {buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Honor `tenzor::manual_seed`; falls back to time-based when unset.
    static std::atomic<uint32_t> offset_counter{0};

    struct PushConstants {
        uint32_t n_elements;
        uint32_t seed;
        uint32_t offset;
        uint32_t distribution;  // 0 = uniform
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(numel);
    push_constants.seed = static_cast<uint32_t>(::tenzor::get_global_seed());
    push_constants.offset = offset_counter.fetch_add(static_cast<uint32_t>(numel));
    push_constants.distribution = 0;  // uniform

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchRandn(const std::vector<int64_t>& shape, DType dtype) -> Tensor {
    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    size_t numel = output.numel();
    if (numel == 0) {
        return output;
    }

    // Use GPU Philox RNG with Box-Muller transform for normal distribution
    int32_t device_id = device.index;

    // Select shader based on dtype
    std::string shader_name = "random";
    if (dtype == DType::Float64) {
        shader_name = "random_f64";
    } else if (dtype == DType::Float16) {
        shader_name = "random_f16";
    } else if (dtype == DType::BFloat16) {
        shader_name = "random_bf16";
    } else if (dtype != DType::Float32) {
        throw std::runtime_error("Unsupported dtype for randn: only Float32, Float64, Float16, and BFloat16 are supported");
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    const void* buffer_out = output.data_ptr();
    size_t buffer_size = numel * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buffer_out}};
    std::vector<size_t> sizes = {buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Honor `tenzor::manual_seed`; falls back to time-based when unset.
    static std::atomic<uint32_t> offset_counter{0};

    struct PushConstants {
        uint32_t n_elements;
        uint32_t seed;
        uint32_t offset;
        uint32_t distribution;  // 1 = normal
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(numel);
    push_constants.seed = static_cast<uint32_t>(::tenzor::get_global_seed());
    push_constants.offset = offset_counter.fetch_add(static_cast<uint32_t>(numel));
    push_constants.distribution = 1;  // normal

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Randint (OpId::Randint)
// ============================================================================

auto VulkanBackend::dispatchRandint(int64_t low, int64_t high,
                                     const std::vector<int64_t>& shape,
                                     DType dtype, const Device& device) -> Tensor {
    Tensor output(shape, dtype, device);

    size_t numel = output.numel();
    if (numel == 0) {
        return output;
    }

    int32_t device_id = device.index;

    // Select shader based on dtype
    std::string shader_name = "randint";
    if (dtype == DType::Int64) {
        shader_name = "randint_i64";
    } else if (dtype != DType::Int32) {
        throw std::runtime_error("Unsupported dtype for randint: only Int32 and Int64 are supported");
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    const void* buffer_out = output.data_ptr();
    size_t buffer_size = numel * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buffer_out}};
    std::vector<size_t> sizes = {buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Honor `tenzor::manual_seed`; falls back to time-based when unset.
    static std::atomic<uint32_t> offset_counter{0};

    struct RandintPC {
        uint32_t n;
        int32_t low;
        int32_t high;
        uint32_t seed;
        uint32_t offset;
    } push_constants;

    push_constants.n = static_cast<uint32_t>(numel);
    push_constants.low = static_cast<int32_t>(low);
    push_constants.high = static_cast<int32_t>(high);
    push_constants.seed = static_cast<uint32_t>(::tenzor::get_global_seed());
    push_constants.offset = offset_counter.fetch_add(static_cast<uint32_t>(numel));

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(RandintPC), &push_constants);

    uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}
// New operations for Vulkan backend - to be appended to vulkan_backend.cpp before closing namespace

// Repeat operation - repeats elements along dimensions
auto VulkanBackend::dispatchRepeat(const Tensor& input, const std::vector<int64_t>& repeats) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = input_shape.size();

    // Pad repeats with 1s at the front if needed
    std::vector<int64_t> padded_repeats = repeats;
    while (padded_repeats.size() < static_cast<size_t>(ndim)) {
        padded_repeats.insert(padded_repeats.begin(), 1);
    }

    // Calculate output shape
    std::vector<int64_t> output_shape(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        output_shape[i] = input_shape[i] * padded_repeats[i];
    }

    // Use expand-based implementation on GPU
    // First unsqueeze to add repeat dimensions, then use expand
    Tensor current = input;
    for (int64_t dim = ndim - 1; dim >= 0; --dim) {
        if (padded_repeats[dim] > 1) {
            // Unsqueeze at dim+1, expand, then flatten back
            current = current.unsqueeze(dim + 1);
            auto curr_shape = current.shape();
            std::vector<int64_t> expand_shape(curr_shape.begin(), curr_shape.end());
            expand_shape[dim + 1] = padded_repeats[dim];
            current = dispatchExpand(current, expand_shape);

            // Flatten the two dimensions back together
            auto curr_shape2 = current.shape();
            std::vector<int64_t> flatten_shape(curr_shape2.begin(), curr_shape2.end());
            flatten_shape[dim] = flatten_shape[dim] * flatten_shape[dim + 1];
            flatten_shape.erase(flatten_shape.begin() + dim + 1);
            current = dispatchReshape(current, flatten_shape);
        }
    }

    return current;
}

/**
 * @brief Dispatch masked_select operation using GPU prefix-sum and gather shaders.
 */
auto VulkanBackend::dispatchMaskedSelect(const Tensor& input, const Tensor& mask) -> Tensor {
    // Validate shapes match
    auto input_shape = input.shape();
    auto mask_shape = mask.shape();
    if (!std::equal(input_shape.begin(), input_shape.end(), mask_shape.begin(), mask_shape.end())) {
        throw std::invalid_argument("masked_select: input and mask must have same shape");
    }

    if (mask.dtype() != DType::Bool && mask.dtype() != DType::Float32) {
        throw std::invalid_argument("masked_select: mask tensor must have dtype Bool or Float32");
    }

    const int64_t numel = input.numel();
    if (numel == 0) {
        return Tensor({0}, input.dtype(), input.device());
    }

    // Determine gather shader based on dtype
    std::string gather_shader = "masked_select_gather";
    if (input.dtype() == DType::Float64) {
        gather_shader = "masked_select_gather_f64";
    } else if (input.dtype() == DType::Float16) {
        gather_shader = "masked_select_gather_f16";
    } else if (input.dtype() == DType::Int32) {
        gather_shader = "masked_select_gather_i32";
    } else if (input.dtype() == DType::Int64) {
        gather_shader = "masked_select_gather_i64";
    } else if (input.dtype() == DType::Bool) {
        gather_shader = "masked_select_gather_bool";
    } else if (input.dtype() == DType::BFloat16) {
        gather_shader = "masked_select_gather_bf16";
    } else if (input.dtype() == DType::Int8 || input.dtype() == DType::UInt8) {
        // Cast to Int32, do masked_select, cast back
        DType orig_dtype = input.dtype();
        auto input_cast = input.to(DType::Int32);
        auto result_cast = dispatchMaskedSelect(input_cast, mask);
        return result_cast.to(orig_dtype);
    } else if (input.dtype() != DType::Float32) {
        // Unsupported dtype: cast to Float32, run GPU MaskedSelect, cast back (no CPU fallback)
        DType orig_dtype = input.dtype();
        auto input_cast = input.to(DType::Float32);
        auto result_cast = dispatchMaskedSelect(input_cast, mask);
        return result_cast.to(orig_dtype);
    }

    int32_t device_id = input.device().index;
    uint32_t n = static_cast<uint32_t>(numel);
    uint32_t mask_is_float = (mask.dtype() == DType::Float32) ? 1 : 0;
    uint32_t n_workgroups = div_wg(n, devices_[device_id].workgroupSize);

    // ---- Pass 1: Count true elements (1a + 1b in single command buffer) ----
    Tensor count_buf({static_cast<int64_t>(n_workgroups + 1)}, DType::Int32, input.device());
    count_buf = dispatchFill(count_buf, 0.0f);

    const void* buffer_mask = mask.data_ptr();
    const void* buffer_count = count_buf.data_ptr();
    size_t mask_bytes = mask.numel() * mask.dtype_size();
    size_t count_bytes = count_buf.numel() * count_buf.dtype_size();

    {
        auto* pipeline = getPipeline("masked_select_count", device_id);
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buffer_mask}, {1, buffer_count}};
        std::vector<size_t> sizes = {mask_bytes, count_bytes};
        VkDescriptorSet ds_1a = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        VkDescriptorSet ds_1b = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t pass; } pc;
        pc.n_elements = n; pc.mask_is_float = mask_is_float;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);

        // Pass 1a: Per-workgroup count
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds_1a, 0, nullptr);
        pc.pass = 0;
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertBarrier(cmd, BarrierType::ComputeToCompute);

        // Pass 1b: Sum workgroup counts (single workgroup)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds_1b, 0, nullptr);
        pc.pass = 1;
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, 1, 1, 1);
        insertComputeBarrier(cmd);

        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);  // Single sync for CPU readback of count
    }

    // Minimal scalar readback (single int32, 4 bytes) — NOT a CPU computation fallback.
    // This is the minimum GPU->CPU sync required for variable-size output allocation in Vulkan.
    Tensor count_cpu = count_buf.slice(0, 0, 1).to(Device::cpu());
    int64_t total_count = static_cast<int64_t>(count_cpu.data<int32_t>()[0]);

    if (total_count == 0) {
        return Tensor({0}, input.dtype(), input.device());
    }

    // ---- Passes 2+3: Prefix sum + gather in single command buffer ----
    Tensor prefix_sums({static_cast<int64_t>(n)}, DType::Int32, input.device());
    prefix_sums = dispatchFill(prefix_sums, 0.0f);
    Tensor block_sums({static_cast<int64_t>(n_workgroups)}, DType::Int32, input.device());
    block_sums = dispatchFill(block_sums, 0.0f);

    const void* buffer_prefix = prefix_sums.data_ptr();
    const void* buffer_blocks = block_sums.data_ptr();
    size_t prefix_bytes = prefix_sums.numel() * prefix_sums.dtype_size();
    size_t blocks_bytes = block_sums.numel() * block_sums.dtype_size();

    Tensor output({total_count}, input.dtype(), input.device());
    if (input.dtype() == DType::Float16) {
        output = dispatchFill(output, 0.0f);
    }

    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();
    size_t input_bytes = input.numel() * input.dtype_size();
    size_t output_bytes = output.numel() * output.dtype_size();

    {
        auto* prefix_pipeline = getPipeline("prefix_sum", device_id);
        auto* gather_pipeline = getPipeline(gather_shader, device_id);

        // Pre-allocate all descriptor sets
        std::vector<std::pair<uint32_t, const void*>> prefix_bindings = {{0, buffer_mask}, {1, buffer_prefix}, {2, buffer_blocks}};
        std::vector<size_t> prefix_sizes = {mask_bytes, prefix_bytes, blocks_bytes};
        VkDescriptorSet ds_2a = allocateAndWriteDescriptorSet(device_id, prefix_pipeline, prefix_bindings, prefix_sizes);
        VkDescriptorSet ds_2b = (n_workgroups > 1)
            ? allocateAndWriteDescriptorSet(device_id, prefix_pipeline, prefix_bindings, prefix_sizes)
            : VK_NULL_HANDLE;

        std::vector<std::pair<uint32_t, const void*>> gather_bindings = {
            {0, buffer_input}, {1, buffer_mask}, {2, buffer_output}, {3, buffer_prefix}
        };
        std::vector<size_t> gather_sizes = {input_bytes, mask_bytes, output_bytes, prefix_bytes};
        VkDescriptorSet ds_3 = allocateAndWriteDescriptorSet(device_id, gather_pipeline, gather_bindings, gather_sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);

        // Pass 2a: Local scan
        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t pass; } pc;
        pc.n_elements = n; pc.mask_is_float = mask_is_float;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefix_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefix_pipeline->layout(), 0, 1, &ds_2a, 0, nullptr);
        pc.pass = 0;
        vkCmdPushConstants(cmd, prefix_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertBarrier(cmd, BarrierType::ComputeToCompute);

        // Pass 2b: Add block offsets
        if (n_workgroups > 1) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefix_pipeline->layout(), 0, 1, &ds_2b, 0, nullptr);
            pc.pass = 1;
            vkCmdPushConstants(cmd, prefix_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, n_workgroups, 1, 1);
            insertBarrier(cmd, BarrierType::ComputeToCompute);
        }

        // Pass 3: Gather selected elements
        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t output_size; } gather_pc;
        gather_pc.n_elements = n; gather_pc.mask_is_float = mask_is_float;
        gather_pc.output_size = static_cast<uint32_t>(total_count);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gather_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gather_pipeline->layout(), 0, 1, &ds_3, 0, nullptr);
        vkCmdPushConstants(cmd, gather_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gather_pc), &gather_pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);

        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);  // Single sync for entire prefix-sum + gather
    }

    return output;
}

// Masked fill - fill elements where mask is true with value
auto VulkanBackend::dispatchMaskedFill(const Tensor& input, const Tensor& mask, float value) -> Tensor {
    // Create value tensor
    auto input_shape = input.shape();
    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    auto value_tensor = dispatchFull(shape_vec, value, input.dtype());

    // Use where: where(mask, value, input)
    return dispatchWhere(mask, value_tensor, input);
}

// Where operation - select from x where condition is true, else from y
auto VulkanBackend::dispatchWhere(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor {
    // Ensure all tensors are on the same device
    if (condition.device().type != Device::Type::Vulkan ||
        x.device().type != Device::Type::Vulkan ||
        y.device().type != Device::Type::Vulkan) {
        throw std::runtime_error("All tensors must be on Vulkan device for where operation");
    }

    auto cond_shape = condition.shape();
    auto x_shape = x.shape();
    auto y_shape = y.shape();

    // Validate shapes match (caller is expected to broadcast beforehand)
    if (cond_shape.size() != x_shape.size() || cond_shape.size() != y_shape.size()) {
        throw std::invalid_argument("where: all tensors must have same number of dimensions");
    }
    for (size_t i = 0; i < cond_shape.size(); ++i) {
        if (cond_shape[i] != x_shape[i] || cond_shape[i] != y_shape[i]) {
            throw std::invalid_argument("where: all tensors must have same shape");
        }
    }

    // Native shader paths. Float64 has a dedicated where_f64 shader so we
    // don't round-trip through Float32 (which would lose ~16 mantissa bits
    // and break Float64 gradcheck on Where). All other dtypes still promote
    // to Float32 since their relative error falls within test tolerances.
    DType target_dtype = x.dtype();
    bool is_f64 = (target_dtype == DType::Float64);
    Tensor cond_u8 = (condition.dtype() == DType::Bool) ? condition : condition.to(DType::Bool);

    Tensor x_work, y_work;
    if (is_f64) {
        x_work = (x.dtype() == DType::Float64) ? x : x.to(DType::Float64);
        y_work = (y.dtype() == DType::Float64) ? y : y.to(DType::Float64);
    } else {
        x_work = (x.dtype() == DType::Float32) ? x : x.to(DType::Float32);
        y_work = (y.dtype() == DType::Float32) ? y : y.to(DType::Float32);
    }

    if (!cond_u8.is_contiguous()) cond_u8 = cond_u8.contiguous();
    if (!x_work.is_contiguous())  x_work  = x_work.contiguous();
    if (!y_work.is_contiguous())  y_work  = y_work.contiguous();

    int32_t device_id = x.device().index;
    auto* pipeline = getPipeline(is_f64 ? "where_f64" : "where", device_id);

    std::vector<int64_t> out_shape(cond_shape.begin(), cond_shape.end());
    Tensor out_work(out_shape, is_f64 ? DType::Float64 : DType::Float32, x.device());

    uint32_t n = static_cast<uint32_t>(out_work.numel());
    struct { uint32_t num_elements; } pc;
    pc.num_elements = n;

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
    size_t data_size = static_cast<size_t>(n) * elem_size;
    size_t bool_size = static_cast<size_t>(n) * sizeof(uint8_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, cond_u8.data_ptr()},
        {1, x_work.data_ptr()},
        {2, y_work.data_ptr()},
        {3, out_work.data_ptr()},
    };
    std::vector<size_t> sizes = {bool_size, data_size, data_size, data_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(n, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return (out_work.dtype() == target_dtype) ? out_work : out_work.to(target_dtype);
}

// ============================================================================
// Interpolation Operation Implementation
// ============================================================================

auto VulkanBackend::dispatchInterpolate(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("interpolate requires 4D input (N, C, H, W), got " +
                                    std::to_string(input_shape.size()) + "D");
    }

    // Extract attributes
    std::string mode(attrs.get_string(AttrKey::Mode));
    bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);

    // Parse output size from comma-separated string
    std::string size_str(attrs.get_string(AttrKey::OutputSize));
    auto comma_pos = size_str.find(',');
    int64_t out_height = std::stoll(size_str.substr(0, comma_pos));
    int64_t out_width = std::stoll(size_str.substr(comma_pos + 1));

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);

    // Select shader based on mode and dtype
    std::string shader_name;
    if (mode == "bilinear" || mode == "bicubic") {
        shader_name = is_float64 ? "bilinear_interpolate_f64" :
                      is_float16 ? "bilinear_interpolate_f16" : "bilinear_interpolate";
    } else {
        shader_name = is_float64 ? "nearest_interpolate_f64" :
                      is_float16 ? "nearest_interpolate_f16" : "nearest_interpolate";
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, channels, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

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
        uint32_t align_corners;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.align_corners = align_corners ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchROIAlignForward(const Tensor& features, const Tensor& rois, const OpAttributes& attrs) -> Tensor {
    auto feat_shape = features.shape();
    if (feat_shape.size() != 4) {
        throw std::invalid_argument("roi_align_forward requires 4D features (N, C, H, W), got " +
                                    std::to_string(feat_shape.size()) + "D");
    }
    if (rois.ndim() != 2 || rois.shape()[1] != 5) {
        throw std::invalid_argument("roi_align_forward requires rois of shape (num_rois, 5)");
    }

    int64_t channels = feat_shape[1];
    int64_t feat_height = feat_shape[2];
    int64_t feat_width = feat_shape[3];
    int64_t num_rois = rois.shape()[0];

    int64_t output_h = attrs.get_int(AttrKey::OutputSizeH);
    int64_t output_w = attrs.get_int(AttrKey::OutputSizeW);
    float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale));
    int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
    bool aligned = attrs.get_bool(AttrKey::Aligned, false);

    int32_t device_id = features.device().index;

    // Select shader based on dtype
    std::string shader_name = "roi_align";
    if (features.dtype() == DType::Float64) {
        shader_name = "roi_align_f64";
    } else if (features.dtype() == DType::Float16) {
        shader_name = "roi_align_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {num_rois, channels, output_h, output_w};
    Tensor output(output_shape, features.dtype(), features.device());

    // Get VkBuffer handles
    const void* buffer_features = features.data_ptr();
    const void* buffer_rois = rois.data_ptr();
    const void* buffer_output = output.data_ptr();

    size_t buffer_size_features = features.numel() * features.dtype_size();
    size_t buffer_size_rois = rois.numel() * rois.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Allocate and write descriptor set (3 bindings)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_features},
        {1, buffer_rois},
        {2, buffer_output}
    };
    std::vector<size_t> sizes = {buffer_size_features, buffer_size_rois, buffer_size_output};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Push constants: 10 uint32_t values = 40 bytes
    struct PushConstants {
        uint32_t n_elements;
        uint32_t num_rois;
        uint32_t channels;
        uint32_t feat_height;
        uint32_t feat_width;
        uint32_t output_h;
        uint32_t output_w;
        uint32_t spatial_scale_bits;
        uint32_t sampling_ratio;
        uint32_t aligned;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.num_rois = static_cast<uint32_t>(num_rois);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.feat_height = static_cast<uint32_t>(feat_height);
    push_constants.feat_width = static_cast<uint32_t>(feat_width);
    push_constants.output_h = static_cast<uint32_t>(output_h);
    push_constants.output_w = static_cast<uint32_t>(output_w);
    // Pass float as uint bits
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &spatial_scale, sizeof(float));
    push_constants.spatial_scale_bits = scale_bits;
    push_constants.sampling_ratio = static_cast<uint32_t>(sampling_ratio);
    push_constants.aligned = aligned ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchROIAlignBackward(const Tensor& grad_output, const Tensor& rois, const OpAttributes& attrs) -> Tensor {
    auto grad_shape = grad_output.shape();
    if (grad_shape.size() != 4) {
        throw std::invalid_argument("roi_align_backward requires 4D grad_output (num_rois, C, output_h, output_w)");
    }

    int64_t num_rois = grad_shape[0];
    int64_t channels = grad_shape[1];
    int64_t output_h = grad_shape[2];
    int64_t output_w = grad_shape[3];

    // Extract feature map dimensions from attributes
    int64_t batch_size = attrs.get_int(AttrKey::BatchSize);
    int64_t feat_height = attrs.get_int(AttrKey::FeatHeight);
    int64_t feat_width = attrs.get_int(AttrKey::FeatWidth);
    float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale));
    int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
    bool aligned = attrs.get_bool(AttrKey::Aligned, false);

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype
    std::string shader_name = "roi_align_backward";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "roi_align_backward_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "roi_align_backward_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create grad_features output tensor (same shape as original features)
    // For f16 backward, we accumulate in f32 then convert
    DType accum_dtype = (grad_output.dtype() == DType::Float16) ? DType::Float32 : grad_output.dtype();
    std::vector<int64_t> grad_features_shape = {batch_size, channels, feat_height, feat_width};
    Tensor grad_features(grad_features_shape, accum_dtype, grad_output.device());

    // Zero-initialize grad_features (atomicAdd accumulates into it)
    grad_features = dispatchFill(grad_features, 0.0f);

    const void* buffer_grad_output = grad_output.data_ptr();
    const void* buffer_rois = rois.data_ptr();
    const void* buffer_grad_features = grad_features.data_ptr();

    size_t buffer_size_grad_output = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_rois = rois.numel() * rois.dtype_size();
    size_t buffer_size_grad_features = grad_features.numel() * grad_features.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_output},
        {1, buffer_rois},
        {2, buffer_grad_features}
    };
    std::vector<size_t> sizes = {buffer_size_grad_output, buffer_size_rois, buffer_size_grad_features};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Push constants: 11 uint32_t values = 44 bytes
    struct PushConstants {
        uint32_t n_elements;
        uint32_t num_rois;
        uint32_t channels;
        uint32_t feat_height;
        uint32_t feat_width;
        uint32_t output_h;
        uint32_t output_w;
        uint32_t spatial_scale_bits;
        uint32_t sampling_ratio;
        uint32_t aligned;
        uint32_t batch_size;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(grad_output.numel());
    push_constants.num_rois = static_cast<uint32_t>(num_rois);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.feat_height = static_cast<uint32_t>(feat_height);
    push_constants.feat_width = static_cast<uint32_t>(feat_width);
    push_constants.output_h = static_cast<uint32_t>(output_h);
    push_constants.output_w = static_cast<uint32_t>(output_w);
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &spatial_scale, sizeof(float));
    push_constants.spatial_scale_bits = scale_bits;
    push_constants.sampling_ratio = static_cast<uint32_t>(sampling_ratio);
    push_constants.aligned = aligned ? 1u : 0u;
    push_constants.batch_size = static_cast<uint32_t>(batch_size);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = static_cast<uint32_t>(div_wg(grad_output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // Convert back from f32 accumulation buffer to f16 if needed
    if (grad_output.dtype() == DType::Float16) {
        grad_features = grad_features.to(DType::Float16);
    }

    return grad_features;
}

auto VulkanBackend::dispatchArgSort(const Tensor& input, int64_t dim, bool descending) -> Tensor {
    auto input_shape = input.shape();
    const int ndim = static_cast<int>(input_shape.size());

    // Normalize dim
    if (dim < 0) dim += ndim;

    const int64_t sort_size = input_shape[dim];

    // Determine shader based on dtype
    std::string sort_shader;
    DType work_dtype = DType::Float32;
    size_t elem_size = sizeof(float);
    if (input.dtype() == DType::Float32) {
        sort_shader = "bitonic_sort";
        work_dtype = DType::Float32;
        elem_size = sizeof(float);
    } else if (input.dtype() == DType::Float64) {
        sort_shader = "bitonic_sort_f64";
        work_dtype = DType::Float64;
        elem_size = sizeof(double);
    } else if (input.dtype() == DType::Int32) {
        sort_shader = "bitonic_sort_i32";
        work_dtype = DType::Int32;
        elem_size = sizeof(int32_t);
    } else if (input.dtype() == DType::Int64) {
        sort_shader = "bitonic_sort_i64";
        work_dtype = DType::Int64;
        elem_size = sizeof(int64_t);
    } else if (input.dtype() == DType::Float16) {
        sort_shader = "bitonic_sort_f16";
        work_dtype = DType::Float16;
        elem_size = sizeof(uint16_t);
    } else if (input.dtype() == DType::Int8 || input.dtype() == DType::UInt8 || input.dtype() == DType::Bool) {
        // Cast to Int32, sort using i32 shader, indices are dtype-independent
        Tensor int32_input = input.to(DType::Int32);
        return dispatchArgSort(int32_input, dim, descending);
    } else if (input.dtype() == DType::BFloat16) {
        // Cast to Float32, argsort (indices are dtype-independent)
        Tensor f32_input = input.to(DType::Float32);
        return dispatchArgSort(f32_input, dim, descending);
    } else {
        sort_shader = "";
    }

    if (sort_shader.empty()) {
        throw std::runtime_error(std::string("Vulkan: ArgSort not supported for dtype ") +
                                 std::string(dtype_name(input.dtype())));
    }

    // Non-last-dim: transpose so sort dim is last, argsort, transpose back
    if (dim != ndim - 1) {
        std::vector<int64_t> perm(ndim);
        std::iota(perm.begin(), perm.end(), int64_t(0));
        std::swap(perm[dim], perm[ndim - 1]);

        std::vector<int64_t> inv_perm(ndim);
        for (int i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;

        Tensor transposed = dispatchContiguous(dispatchPermute(input, perm));
        Tensor indices_t = dispatchArgSort(transposed, ndim - 1, descending);

        return dispatchContiguous(dispatchPermute(indices_t, inv_perm));
    }

    // For large arrays (> 65K), use GPU radix sort instead of bitonic
    if (sort_size > 65536) {
        // Flatten last dim, radix sort, unflatten
        auto flat = input.contiguous();
        auto [sorted_vals, sorted_indices] = dispatchRadixSort(flat, descending);
        // The radix sort returns Int64 indices already
        return sorted_indices;
    }

    if (sort_size <= 1) {
        std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
        Tensor result(shape_vec, DType::Int64, input.device());
        result = dispatchFill(result, 0.0f);
        return result;
    }

    int32_t device_id = input.device().index;

    // Compute padded size (next power of 2)
    uint32_t n = static_cast<uint32_t>(sort_size);
    uint32_t padded_n = 1;
    while (padded_n < n) padded_n <<= 1;

    // Number of bitonic sort stages = log2(padded_n)
    uint32_t num_stages = 0;
    { uint32_t tmp = padded_n; while (tmp > 1) { num_stages++; tmp >>= 1; } }

    // Number of independent sort slices (sort along last dim)
    int64_t num_slices = 1;
    for (int i = 0; i < ndim - 1; ++i) num_slices *= input_shape[i];

    // Output tensor (Int64 indices)
    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    Tensor output(shape_vec, DType::Int64, input.device());

    // Allocate working buffers for bitonic sort (padded to power-of-2)
    Tensor work_values({static_cast<int64_t>(padded_n)}, work_dtype, input.device());
    Tensor work_indices({static_cast<int64_t>(padded_n)}, DType::Int32, input.device());

    float pad_value = descending ? -std::numeric_limits<float>::infinity()
                                 : std::numeric_limits<float>::infinity();
    // For integer types, use max/min as pad value (float approximation is fine
    // for padding — padded elements just need to sort to the end)
    if (work_dtype == DType::Int32) {
        pad_value = descending ? static_cast<float>(std::numeric_limits<int32_t>::min())
                               : static_cast<float>(std::numeric_limits<int32_t>::max());
    } else if (work_dtype == DType::Int64) {
        pad_value = descending ? static_cast<float>(std::numeric_limits<int64_t>::min())
                               : static_cast<float>(std::numeric_limits<int64_t>::max());
    }

    size_t values_bytes = padded_n * elem_size;
    size_t indices_bytes = padded_n * sizeof(int32_t);

    auto* pipeline = getPipeline(sort_shader, device_id);
    uint32_t workgroups = div_wg(padded_n / 2, devices_[device_id].workgroupSize);

    // Pre-build initial index array on CPU (reused for each slice)
    std::vector<int32_t> init_indices(padded_n);
    for (uint32_t i = 0; i < padded_n; ++i) {
        init_indices[i] = (i < n) ? static_cast<int32_t>(i) : static_cast<int32_t>(n);
    }

    for (int64_t slice = 0; slice < num_slices; ++slice) {
        // Step 1: Initialize working buffers (batched into single sync)
        work_values = dispatchFill(work_values, pad_value);

        size_t slice_bytes = sort_size * elem_size;
        copy(work_values.data_ptr(),
             static_cast<const char*>(input.data_ptr()) + slice * slice_bytes,
             slice_bytes, CopyKind::DeviceToDevice);

        copy(work_indices.data_ptr(), init_indices.data(),
             padded_n * sizeof(int32_t), CopyKind::HostToDevice);
        synchronize(device_id);  // Single sync for all 3 init operations

        // Step 2: Run all bitonic sort passes in a single command buffer
        {
            const void* buffer_values = work_values.data_ptr();
            const void* buffer_indices = work_indices.data_ptr();
            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, buffer_values}, {1, buffer_indices}
            };
            std::vector<size_t> sizes = {values_bytes, indices_bytes};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(
                device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);

            for (uint32_t stage = 0; stage < num_stages; ++stage) {
                for (int32_t substage = static_cast<int32_t>(stage); substage >= 0; --substage) {
                    struct {
                        uint32_t n;
                        uint32_t padded_n;
                        uint32_t stage;
                        uint32_t substage;
                        uint32_t descending;
                    } pc;
                    pc.n = n;
                    pc.padded_n = padded_n;
                    pc.stage = stage;
                    pc.substage = static_cast<uint32_t>(substage);
                    pc.descending = descending ? 1 : 0;

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
                    vkCmdPushConstants(cmd, pipeline->layout(),
                                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    vkCmdDispatch(cmd, workgroups, 1, 1);
                    insertComputeBarrier(cmd);
                }
            }

            endSingleTimeCommands(cmd, device_id);
            synchronize(device_id);
        }

        // Step 3: Convert Int32 indices to Int64 on GPU using cast shader
        {
            auto* cast_pipeline = getPipeline("cast_i32_i64", device_id);
            Tensor int64_chunk({sort_size}, DType::Int64, input.device());

            size_t in_bytes = sort_size * sizeof(int32_t);
            size_t out_bytes = sort_size * sizeof(int64_t);

            std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
                {0, work_indices.data_ptr()},
                {1, int64_chunk.data_ptr()}
            };
            std::vector<size_t> cast_sizes = {in_bytes, out_bytes};

            VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(
                device_id, cast_pipeline, cast_bindings, cast_sizes);

            struct { uint32_t n; } cast_pc;
            cast_pc.n = static_cast<uint32_t>(sort_size);

            VkCommandBuffer cast_cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cast_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
            vkCmdBindDescriptorSets(cast_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
            vkCmdPushConstants(cast_cmd, cast_pipeline->layout(),
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
            vkCmdDispatch(cast_cmd, div_wg(sort_size, devices_[device_id].workgroupSize), 1, 1);
            insertComputeBarrier(cast_cmd);
            endSingleTimeCommands(cast_cmd, device_id);
            synchronize(device_id);

            // Copy to output slice
            void* dst_ptr = static_cast<char*>(output.data_ptr()) +
                            slice * sort_size * sizeof(int64_t);
            copy(dst_ptr, int64_chunk.data_ptr(), out_bytes, CopyKind::DeviceToDevice);
            synchronize(device_id);
        }
    }

    return output;
}



// ============================================================================
// Type Cast Operations
// ============================================================================

/**
 * @brief Cast tensor to a different dtype using dedicated GPU shaders.
 *
 * Supports the following conversion paths:
 *   Float32 <-> Float16  (via packed uint32 f16 pair shaders)
 *   Float32 <-> Float64  (via float64_t shaders)
 *   Float16 -> Float64   (two-step: f16->f32->f64)
 *   Float64 -> Float16   (two-step: f64->f32->f16)
 *
 * For unsupported conversion paths, a two-step GPU cast through Float32
 * is used, keeping all computation on the Vulkan device.
 */
auto VulkanBackend::dispatchCast(const Tensor& input, DType target_dtype) -> Tensor {
    if (input.dtype() == target_dtype) {
        return input;  // No-op
    }

    DType src_dtype = input.dtype();
    int32_t device_id = input.device().index;
    int64_t numel = input.numel();

    // Complex casts: real↔complex64 and real↔complex128 use native Vulkan shaders.
    // Complex-to-complex (Complex64 ↔ Complex128) also uses native shaders.
    auto is_complex_of = [](DType d) {
        return d == DType::Complex64 || d == DType::Complex128;
    };
    if (is_complex_of(src_dtype) != is_complex_of(target_dtype)) {
        int32_t dev_id = input.device().index;
        int64_t n = input.numel();

        if (!is_complex_of(src_dtype) && target_dtype == DType::Complex64) {
            // Real -> Complex64: zero-fill imaginary part
            Tensor input_f32 = (src_dtype != DType::Float32) ? input.to(DType::Float32) : input;
            auto* pipeline_rc = getPipeline("cast_real_complex", dev_id);
            auto input_shape = input.shape();
            Tensor output(std::vector<int64_t>(input_shape.begin(), input_shape.end()), DType::Complex64, input.device());
            struct { uint32_t num_elements; } pc_rc;
            pc_rc.num_elements = static_cast<uint32_t>(n);
            size_t in_buf_rc = static_cast<size_t>(n) * sizeof(float);
            size_t out_buf_rc = static_cast<size_t>(n) * 2 * sizeof(float);
            std::vector<std::pair<uint32_t, const void*>> bindings_rc = {
                {0, input_f32.data_ptr()}, {1, output.data_ptr()}
            };
            std::vector<size_t> sizes_rc = {in_buf_rc, out_buf_rc};
            VkDescriptorSet ds_rc = allocateAndWriteDescriptorSet(dev_id, pipeline_rc, bindings_rc, sizes_rc);
            VkCommandBuffer cmd_rc = beginSingleTimeCommands(dev_id);
            vkCmdBindPipeline(cmd_rc, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_rc->pipeline());
            vkCmdBindDescriptorSets(cmd_rc, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline_rc->layout(), 0, 1, &ds_rc, 0, nullptr);
            vkCmdPushConstants(cmd_rc, pipeline_rc->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_rc), &pc_rc);
            vkCmdDispatch(cmd_rc, div_wg(n, devices_[dev_id].workgroupSize), 1, 1);
            insertComputeOnlyBarrier(cmd_rc);
            endSingleTimeCommands(cmd_rc, dev_id);
            return output;
        } else if (src_dtype == DType::Complex64 && !is_complex_of(target_dtype)) {
            // Complex64 -> Real: extract real part
            auto* pipeline_cr = getPipeline("cast_complex_real", dev_id);
            auto input_shape = input.shape();
            Tensor output_f32(std::vector<int64_t>(input_shape.begin(), input_shape.end()), DType::Float32, input.device());
            struct { uint32_t num_elements; } pc_cr;
            pc_cr.num_elements = static_cast<uint32_t>(n);
            size_t in_buf_cr = static_cast<size_t>(n) * 2 * sizeof(float);
            size_t out_buf_cr = static_cast<size_t>(n) * sizeof(float);
            std::vector<std::pair<uint32_t, const void*>> bindings_cr = {
                {0, input.data_ptr()}, {1, output_f32.data_ptr()}
            };
            std::vector<size_t> sizes_cr = {in_buf_cr, out_buf_cr};
            VkDescriptorSet ds_cr = allocateAndWriteDescriptorSet(dev_id, pipeline_cr, bindings_cr, sizes_cr);
            VkCommandBuffer cmd_cr = beginSingleTimeCommands(dev_id);
            vkCmdBindPipeline(cmd_cr, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_cr->pipeline());
            vkCmdBindDescriptorSets(cmd_cr, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline_cr->layout(), 0, 1, &ds_cr, 0, nullptr);
            vkCmdPushConstants(cmd_cr, pipeline_cr->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_cr), &pc_cr);
            vkCmdDispatch(cmd_cr, div_wg(n, devices_[dev_id].workgroupSize), 1, 1);
            insertComputeOnlyBarrier(cmd_cr);
            endSingleTimeCommands(cmd_cr, dev_id);
            return (target_dtype != DType::Float32) ? output_f32.to(target_dtype) : output_f32;
        } else if (!is_complex_of(src_dtype) && target_dtype == DType::Complex128) {
            // Real -> Complex128: zero-fill imaginary part
            Tensor input_f64 = (src_dtype != DType::Float64) ? input.to(DType::Float64) : input;
            auto* pipeline_rc128 = getPipeline("cast_real_complex128", dev_id);
            auto input_shape = input.shape();
            Tensor output(std::vector<int64_t>(input_shape.begin(), input_shape.end()), DType::Complex128, input.device());
            struct { uint32_t num_elements; } pc_rc128;
            pc_rc128.num_elements = static_cast<uint32_t>(n);
            size_t in_buf_rc128 = static_cast<size_t>(n) * sizeof(double);
            size_t out_buf_rc128 = static_cast<size_t>(n) * 2 * sizeof(double);
            std::vector<std::pair<uint32_t, const void*>> bindings_rc128 = {
                {0, input_f64.data_ptr()}, {1, output.data_ptr()}
            };
            std::vector<size_t> sizes_rc128 = {in_buf_rc128, out_buf_rc128};
            VkDescriptorSet ds_rc128 = allocateAndWriteDescriptorSet(dev_id, pipeline_rc128, bindings_rc128, sizes_rc128);
            VkCommandBuffer cmd_rc128 = beginSingleTimeCommands(dev_id);
            vkCmdBindPipeline(cmd_rc128, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_rc128->pipeline());
            vkCmdBindDescriptorSets(cmd_rc128, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline_rc128->layout(), 0, 1, &ds_rc128, 0, nullptr);
            vkCmdPushConstants(cmd_rc128, pipeline_rc128->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_rc128), &pc_rc128);
            vkCmdDispatch(cmd_rc128, div_wg(n, devices_[dev_id].workgroupSize), 1, 1);
            insertComputeOnlyBarrier(cmd_rc128);
            endSingleTimeCommands(cmd_rc128, dev_id);
            return output;
        } else if (src_dtype == DType::Complex128 && !is_complex_of(target_dtype)) {
            // Complex128 -> Real: extract real part
            auto* pipeline_cr128 = getPipeline("cast_complex128_real", dev_id);
            auto input_shape = input.shape();
            Tensor output_f64(std::vector<int64_t>(input_shape.begin(), input_shape.end()), DType::Float64, input.device());
            struct { uint32_t num_elements; } pc_cr128;
            pc_cr128.num_elements = static_cast<uint32_t>(n);
            size_t in_buf_cr128 = static_cast<size_t>(n) * 2 * sizeof(double);
            size_t out_buf_cr128 = static_cast<size_t>(n) * sizeof(double);
            std::vector<std::pair<uint32_t, const void*>> bindings_cr128 = {
                {0, input.data_ptr()}, {1, output_f64.data_ptr()}
            };
            std::vector<size_t> sizes_cr128 = {in_buf_cr128, out_buf_cr128};
            VkDescriptorSet ds_cr128 = allocateAndWriteDescriptorSet(dev_id, pipeline_cr128, bindings_cr128, sizes_cr128);
            VkCommandBuffer cmd_cr128 = beginSingleTimeCommands(dev_id);
            vkCmdBindPipeline(cmd_cr128, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_cr128->pipeline());
            vkCmdBindDescriptorSets(cmd_cr128, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline_cr128->layout(), 0, 1, &ds_cr128, 0, nullptr);
            vkCmdPushConstants(cmd_cr128, pipeline_cr128->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_cr128), &pc_cr128);
            vkCmdDispatch(cmd_cr128, div_wg(n, devices_[dev_id].workgroupSize), 1, 1);
            insertComputeOnlyBarrier(cmd_cr128);
            endSingleTimeCommands(cmd_cr128, dev_id);
            return (target_dtype != DType::Float64) ? output_f64.to(target_dtype) : output_f64;
        } else {
            // Defensive path: convert via on-device intermediate to avoid CPU round-trip.
            // This branch is technically unreachable since all real↔complex combinations
            // are handled above, but we keep it as a safety net.
            if (is_complex_of(target_dtype)) {
                // Unknown real type → complex: cast real to Float32/Float64 first on device
                DType intermediate = (target_dtype == DType::Complex128) ? DType::Float64 : DType::Float32;
                Tensor intermediate_tensor = (src_dtype != intermediate) ? dispatchCast(input, intermediate) : input;
                return dispatchCast(intermediate_tensor, target_dtype);
            } else {
                // Complex → unknown real type: extract real part to Float32/Float64, then cast
                DType intermediate = (src_dtype == DType::Complex128) ? DType::Float64 : DType::Float32;
                Tensor intermediate_tensor = dispatchCast(input, intermediate);
                return (intermediate != target_dtype) ? dispatchCast(intermediate_tensor, target_dtype) : intermediate_tensor;
            }
        }
    }
    if (is_complex_of(src_dtype) && is_complex_of(target_dtype) && src_dtype != target_dtype) {
        // Complex-to-complex: native Vulkan shader
        std::string complex_shader = (src_dtype == DType::Complex64)
            ? "cast_complex64_complex128" : "cast_complex128_complex64";
        int32_t dev_id = input.device().index;
        auto* cpipeline = getPipeline(complex_shader, dev_id);

        int64_t num_complex = input.numel();  // Number of complex elements
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        Tensor output(out_shape, target_dtype, input.device());

        // Buffer sizes: Complex64 = 8 bytes/elem, Complex128 = 16 bytes/elem
        size_t in_buf_size = static_cast<size_t>(num_complex) * dtype_size(src_dtype);
        size_t out_buf_size = static_cast<size_t>(num_complex) * dtype_size(target_dtype);

        struct { uint32_t num_elements; } cpc;
        cpc.num_elements = static_cast<uint32_t>(num_complex);

        std::vector<std::pair<uint32_t, const void*>> cbindings = {
            {0, input.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> csizes = {in_buf_size, out_buf_size};

        VkDescriptorSet cds = allocateAndWriteDescriptorSet(dev_id, cpipeline, cbindings, csizes);

        VkCommandBuffer ccmd = beginSingleTimeCommands(dev_id);
        vkCmdBindPipeline(ccmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpipeline->pipeline());
        vkCmdBindDescriptorSets(ccmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               cpipeline->layout(), 0, 1, &cds, 0, nullptr);
        vkCmdPushConstants(ccmd, cpipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cpc), &cpc);
        vkCmdDispatch(ccmd, div_wg(static_cast<uint32_t>(num_complex), devices_[dev_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(ccmd);
        endSingleTimeCommands(ccmd, dev_id);

        return output;
    }

    // Determine shader name based on source/target dtype pair
    std::string shader_name;
    bool two_step = false;

    if (src_dtype == DType::Float32 && target_dtype == DType::Float16) {
        shader_name = "cast_f32_f16";
    } else if (src_dtype == DType::Float16 && target_dtype == DType::Float32) {
        shader_name = "cast_f16_f32";
    } else if (src_dtype == DType::Float32 && target_dtype == DType::Float64) {
        shader_name = "cast_f32_f64";
    } else if (src_dtype == DType::Float64 && target_dtype == DType::Float32) {
        shader_name = "cast_f64_f32";
    } else if (src_dtype == DType::Float16 && target_dtype == DType::Float64) {
        // Two-step: f16 -> f32 -> f64
        two_step = true;
    } else if (src_dtype == DType::Float64 && target_dtype == DType::Float16) {
        // Two-step: f64 -> f32 -> f16
        two_step = true;
    } else if (src_dtype == DType::Int32 && target_dtype == DType::Float32) {
        shader_name = "cast_i32_f32";
    } else if (src_dtype == DType::Float32 && target_dtype == DType::Int32) {
        shader_name = "cast_f32_i32";
    } else if (src_dtype == DType::Int8 && target_dtype == DType::Float32) {
        shader_name = "cast_i8_f32";
    } else if (src_dtype == DType::Float32 && target_dtype == DType::Int8) {
        shader_name = "cast_f32_i8";
    } else if (src_dtype == DType::UInt8 && target_dtype == DType::Float32) {
        shader_name = "cast_u8_f32";
    } else if (src_dtype == DType::Float32 && target_dtype == DType::UInt8) {
        shader_name = "cast_f32_u8";
    } else if (src_dtype == DType::Bool && target_dtype == DType::Float32) {
        shader_name = "cast_bool_f32";
    } else if (src_dtype == DType::Float32 && target_dtype == DType::Bool) {
        shader_name = "cast_f32_bool";
    } else if (src_dtype == DType::Bool && target_dtype == DType::Int64) {
        shader_name = "cast_bool_i64";
    } else if (src_dtype == DType::BFloat16 && target_dtype == DType::Float32) {
        shader_name = "cast_bf16_f32";
    } else if (src_dtype == DType::Float32 && target_dtype == DType::BFloat16) {
        shader_name = "cast_f32_bf16";
    } else if (src_dtype == DType::Int32 && target_dtype == DType::Int64) {
        shader_name = "cast_i32_i64";
    } else if (src_dtype == DType::Float64 && target_dtype == DType::Int32) {
        shader_name = "cast_f64_i32";
    } else if (src_dtype == DType::Int32 && target_dtype == DType::Float64) {
        shader_name = "cast_i32_f64";
    } else if (src_dtype == DType::Float64 && target_dtype == DType::Int64) {
        shader_name = "cast_f64_i64";
    } else if (src_dtype == DType::Int64 && target_dtype == DType::Float64) {
        shader_name = "cast_i64_f64";
    } else if (src_dtype == DType::BFloat16 &&
               (target_dtype == DType::Float64 || target_dtype == DType::Float16 ||
                target_dtype == DType::Int32 || target_dtype == DType::Int64)) {
        // Two-step via Float32: BFloat16 -> Float32 -> target
        two_step = true;
    } else if ((src_dtype == DType::Float64 || src_dtype == DType::Float16 ||
                src_dtype == DType::Int32 || src_dtype == DType::Int64) &&
               target_dtype == DType::BFloat16) {
        // Two-step via Float32: source -> Float32 -> BFloat16
        two_step = true;
    } else if ((src_dtype == DType::Int8 || src_dtype == DType::Bool ||
                src_dtype == DType::UInt8) &&
               (target_dtype == DType::Float64 || target_dtype == DType::Float16 ||
                target_dtype == DType::BFloat16 ||
                target_dtype == DType::Int32 || target_dtype == DType::Int64)) {
        // Two-step via Float32: Int8/UInt8/Bool -> Float32 -> target
        two_step = true;
    } else if (src_dtype == DType::Float32 && target_dtype == DType::UInt8) {
        // Already handled above; left for completeness
        shader_name = "cast_f32_u8";
    } else if ((target_dtype == DType::UInt8) &&
               (src_dtype == DType::Float64 || src_dtype == DType::Float16 ||
                src_dtype == DType::BFloat16 ||
                src_dtype == DType::Int32 || src_dtype == DType::Int64)) {
        // Two-step via Float32: source -> Float32 -> UInt8
        two_step = true;
    } else if ((src_dtype == DType::Int32 || src_dtype == DType::Int64) &&
               (target_dtype == DType::Float64 || target_dtype == DType::Float16)) {
        // Two-step via Float32: Int32/Int64 -> Float32 -> target
        two_step = true;
    } else if ((src_dtype == DType::Float64 || src_dtype == DType::Float16) &&
               (target_dtype == DType::Int32 || target_dtype == DType::Int8 || target_dtype == DType::Bool)) {
        // Two-step via Float32: Float64/Float16 -> Float32 -> target
        two_step = true;
    } else if (src_dtype == DType::Float32 && target_dtype == DType::Int64) {
        shader_name = "cast_f32_i64";
    } else if (src_dtype == DType::Int64 && target_dtype == DType::Float32) {
        shader_name = "cast_i64_f32";
    } else if (src_dtype == DType::Float32 && target_dtype == DType::FP8_E4M3) {
        shader_name = "cast_f32_fp8e4m3";
    } else if (src_dtype == DType::FP8_E4M3 && target_dtype == DType::Float32) {
        shader_name = "cast_fp8e4m3_f32";
    } else if (src_dtype == DType::Float32 && target_dtype == DType::FP8_E5M2) {
        shader_name = "cast_f32_fp8e5m2";
    } else if (src_dtype == DType::FP8_E5M2 && target_dtype == DType::Float32) {
        shader_name = "cast_fp8e5m2_f32";
    } else if ((src_dtype == DType::FP8_E4M3 || src_dtype == DType::FP8_E5M2) &&
               target_dtype != DType::Float32) {
        // FP8 -> non-Float32: two-step via Float32
        two_step = true;
    } else if (src_dtype != DType::Float32 &&
               (target_dtype == DType::FP8_E4M3 || target_dtype == DType::FP8_E5M2)) {
        // non-Float32 -> FP8: two-step via Float32
        two_step = true;
    } else if (src_dtype != DType::Float32 && target_dtype != DType::Float32) {
        // Generic two-step via Float32 for any remaining dtype pair
        two_step = true;
    } else if (src_dtype == DType::Int16 || target_dtype == DType::Int16) {
        // No native Int16 SPIR-V cast shaders. Round-trip through CPU; this
        // matches the existing BFloat16/FP8 fallback pattern in this file
        // when no direct GPU path exists. Used by the Int32-promoted bitwise
        // path in vulkan_kernel_registry.cpp for Int16 inputs.
        Tensor cpu_input = input.to(Device::cpu());
        Tensor cpu_output = cpu_input.to(target_dtype);
        return cpu_output.to(input.device());
    } else {
        // Should not reach here; all paths covered. Safety fallback.
        throw std::runtime_error("Vulkan cast: unsupported dtype pair (" +
                                 std::string(dtype_name(src_dtype)) + " -> " + std::string(dtype_name(target_dtype)) + ")");
    }

    // Two-step casts via Float32 intermediate
    if (two_step) {
        Tensor intermediate = dispatchCast(input, DType::Float32);
        return dispatchCast(intermediate, target_dtype);
    }

    // Single-step GPU cast using compute shader
    auto* pipeline = getPipeline(shader_name, device_id);

    // Determine input/output buffer sizes
    // For Float16 (packed): buffer size = ceil(numel / 2) * 4 bytes
    // For byte types (Bool, Int8, UInt8): round up to 4-byte boundary
    auto buffer_size_for_dtype = [&](DType dtype) -> size_t {
        if (dtype == DType::Float16 || dtype == DType::BFloat16) {
            size_t num_pairs = (static_cast<size_t>(numel) + 1) / 2;
            return num_pairs * 4;  // 4 bytes per packed uint32
        }
        size_t raw = static_cast<size_t>(numel) * dtype_size(dtype);
        if (dtype == DType::Bool || dtype == DType::Int8 || dtype == DType::UInt8 ||
            dtype == DType::FP8_E4M3 || dtype == DType::FP8_E5M2) {
            return (raw + 3) & ~size_t(3);  // Round up to 4-byte boundary
        }
        return raw;
    };

    size_t input_buf_size = buffer_size_for_dtype(src_dtype);
    size_t output_buf_size = buffer_size_for_dtype(target_dtype);

    // Allocate output tensor
    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    Tensor output(out_shape, target_dtype, input.device());

    const void* buf_in = input.data_ptr();
    const void* buf_out = output.data_ptr();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_in},
        {1, buf_out}
    };
    std::vector<size_t> sizes = {input_buf_size, output_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t num_elements;
    } push_constants;

    push_constants.num_elements = static_cast<uint32_t>(numel);

    // For packed f16 shaders, workgroups process pairs of elements
    uint32_t dispatch_count;
    if ((src_dtype == DType::Float16 && target_dtype == DType::Float32) ||
        (src_dtype == DType::Float32 && target_dtype == DType::Float16) ||
        (src_dtype == DType::BFloat16 && target_dtype == DType::Float32) ||
        (src_dtype == DType::Float32 && target_dtype == DType::BFloat16)) {
        // Each invocation handles 2 elements (one packed pair)
        uint32_t num_pairs = (static_cast<uint32_t>(numel) + 1) / 2;
        dispatch_count = div_wg(num_pairs, devices_[device_id].workgroupSize);
    } else {
        // Each invocation handles 1 element
        dispatch_count = div_wg(static_cast<uint32_t>(numel), devices_[device_id].workgroupSize);
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    vkCmdDispatch(cmdBuffer, dispatch_count, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================

// ============================================================================
// Slice/Split/Chunk/Flatten Operations
// ============================================================================

auto VulkanBackend::dispatchSlice(const Tensor& input,
                                   const std::vector<int64_t>& starts,
                                   const std::vector<int64_t>& ends,
                                   const std::vector<int64_t>& steps) -> Tensor {
    // Multi-dimensional slice using strided_copy shader
    // For each dimension, compute new shape, stride, and offset
    auto input_shape = input.shape();
    auto input_strides = input.strides();
    int64_t ndim = input_shape.size();

    // Normalize and compute output shape
    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);
    int64_t offset = input.offset();

    for (int64_t d = 0; d < ndim; ++d) {
        int64_t dim_size = input_shape[d];
        int64_t start = (d < static_cast<int64_t>(starts.size())) ? starts[d] : 0;
        int64_t end = (d < static_cast<int64_t>(ends.size())) ? ends[d] : dim_size;
        int64_t step = (d < static_cast<int64_t>(steps.size())) ? steps[d] : 1;

        // Normalize negative indices
        if (start < 0) start += dim_size;
        if (end < 0) end += dim_size;

        // Clamp to valid range
        start = std::clamp(start, int64_t(0), dim_size);
        end = std::clamp(end, int64_t(0), dim_size);

        if (step <= 0) {
            throw std::invalid_argument("Slice: step must be positive");
        }

        // Compute output dim size
        int64_t length = (end > start) ? ((end - start + step - 1) / step) : 0;

        new_shape[d] = length;
        new_strides[d] = input_strides[d] * step;
        offset += start * input_strides[d];
    }

    // Check if result is empty
    int64_t out_numel = 1;
    for (auto s : new_shape) out_numel *= s;
    if (out_numel == 0) {
        return Tensor(new_shape, input.dtype(), input.device());
    }

    // Create a view if all steps are 1 (simple sub-tensor)
    bool all_step_one = true;
    for (size_t d = 0; d < steps.size(); ++d) {
        if (steps[d] != 1) { all_step_one = false; break; }
    }

    if (all_step_one) {
        // Can create a view - shares storage with offset
        Tensor result;
        result.impl_ = make_intrusive<TensorImpl>(*input.impl_);
        result.mutable_shape() = new_shape;
        result.mutable_strides() = new_strides;
        result.set_offset(offset);
        return result;
    }

    // Non-unit steps require copying to a contiguous tensor via strided_copy
    // Create a view first, then make it contiguous
    Tensor view;
    view.impl_ = make_intrusive<TensorImpl>(*input.impl_);
    view.mutable_shape() = new_shape;
    view.mutable_strides() = new_strides;
    view.set_offset(offset);
    return dispatchContiguous(view);
}

auto VulkanBackend::dispatchSplit(const Tensor& input, int64_t split_size, int64_t dim) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    // Normalize dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Split: invalid dimension " + std::to_string(dim));
    }

    int64_t dim_size = shape[dim];
    std::vector<Tensor> results;

    for (int64_t start = 0; start < dim_size; start += split_size) {
        int64_t end = std::min(start + split_size, dim_size);

        // Create starts/ends/steps for slice
        std::vector<int64_t> starts(ndim, 0);
        std::vector<int64_t> ends(shape.begin(), shape.end());
        std::vector<int64_t> steps(ndim, 1);

        starts[dim] = start;
        ends[dim] = end;

        results.push_back(dispatchSlice(input, starts, ends, steps));
    }

    return results;
}

auto VulkanBackend::dispatchChunk(const Tensor& input, int64_t chunks, int64_t dim) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    // Normalize dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Chunk: invalid dimension " + std::to_string(dim));
    }

    int64_t dim_size = shape[dim];
    // Each chunk gets ceil(dim_size / chunks) elements (except possibly the last)
    int64_t chunk_size = (dim_size + chunks - 1) / chunks;

    return dispatchSplit(input, chunk_size, dim);
}

auto VulkanBackend::dispatchFlatten(const Tensor& input, int64_t start_dim, int64_t end_dim) -> Tensor {
    // Flatten is metadata-only (reshape)
    auto shape = input.shape();
    int64_t ndim = shape.size();

    // Normalize dims
    if (start_dim < 0) start_dim += ndim;
    if (end_dim < 0) end_dim += ndim;

    if (start_dim < 0 || start_dim >= ndim || end_dim < 0 || end_dim >= ndim || start_dim > end_dim) {
        throw std::invalid_argument("Flatten: invalid dimensions");
    }

    // Compute new shape
    std::vector<int64_t> new_shape;
    for (int64_t d = 0; d < start_dim; ++d) {
        new_shape.push_back(shape[d]);
    }

    int64_t flat_size = 1;
    for (int64_t d = start_dim; d <= end_dim; ++d) {
        flat_size *= shape[d];
    }
    new_shape.push_back(flat_size);

    for (int64_t d = end_dim + 1; d < ndim; ++d) {
        new_shape.push_back(shape[d]);
    }

    return dispatchReshape(input, new_shape);
}

// ============================================================================
// Typed dispatch wrappers for formerly string-dispatched operations
// ============================================================================

auto VulkanBackend::dispatchBatchNorm2dUpdateRunningStats(
    std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
    // Previously this delegated to the dispatch table, but the dispatch
    // table's entry for OpId::BatchNorm2dUpdateRunningStats calls this
    // same function — infinite recursion that left BN-train hanging
    // until stack overflow. Implement the EMA update directly via the
    // existing elementwise ops:
    //
    //     new_running = (1 - momentum) * running + momentum * batch
    //
    // on both running_mean and running_var.
    //
    // inputs: [running_mean, running_var, batch_mean, batch_var]
    if (inputs.size() < 4) {
        throw std::runtime_error(
            "BatchNorm2dUpdateRunningStats expects 4 tensors "
            "(running_mean, running_var, batch_mean, batch_var)");
    }

    const Tensor& running_mean = inputs[0];
    const Tensor& running_var  = inputs[1];
    const Tensor& batch_mean   = inputs[2];
    const Tensor& batch_var    = inputs[3];

    float momentum =
        static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));
    float one_minus_m = 1.0f - momentum;

    auto mix = [&](const Tensor& running, const Tensor& batch) -> Tensor {
        // Scale running by (1 - momentum) using a broadcast with a
        // filled 1-element tensor. dispatchBinaryOp handles broadcast
        // across a scalar shape.
        auto scale_r = dispatchFill(
            Tensor({1}, running.dtype(), running.device()), one_minus_m);
        auto scale_b = dispatchFill(
            Tensor({1}, batch.dtype(), batch.device()), momentum);

        auto r_scaled = dispatchBinaryOp("mul", running, scale_r);
        auto b_scaled = dispatchBinaryOp("mul", batch,   scale_b);
        return dispatchBinaryOp("add", r_scaled, b_scaled);
    };

    auto new_running_mean = mix(running_mean, batch_mean);
    auto new_running_var  = mix(running_var,  batch_var);

    return {new_running_mean, new_running_var};
}

auto VulkanBackend::dispatchFusedRMSPropStep(
    std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
    // inputs (from registry): [param, grad, square_avg, grad_avg?, momentum?]
    // shader bindings:         0=grad, 1=param, 2=square_avg, 3=momentum, 4=grad_avg
    //
    // Direct pipeline invocation — the previous dispatch("fused_rmsprop_step",
    // inputs, attrs) recursed back through the op-id table and blew the
    // stack. Same bug class as dispatchFusedAdadeltaStep.
    if (inputs.size() < 3) {
        throw std::invalid_argument(
            "FusedRMSPropStep requires at least 3 inputs (param, grad, square_avg)");
    }

    double lr = attrs.get_float(AttrKey::Lr, 0.01);
    double alpha = attrs.get_float(AttrKey::Alpha, 0.99);
    double eps = attrs.get_float(AttrKey::Eps, 1e-8);
    double weight_decay = attrs.get_float(AttrKey::WeightDecay, 0.0);
    double momentum = attrs.get_float(AttrKey::Momentum, 0.0);
    bool centered = attrs.get_bool(AttrKey::Centered, false);
    bool has_momentum = (momentum > 0.0);

    int64_t numel = inputs[0].numel();
    int32_t device_id = inputs[0].device().index;
    bool is_f64 = (inputs[0].dtype() == DType::Float64);
    std::string shader = is_f64 ? "fused_rmsprop_step_f64" : "fused_rmsprop_step";
    auto* pipeline = getPipeline(shader, device_id);

    size_t buf_size = numel * inputs[0].dtype_size();

    // grad_avg and momentum_buffer are optional. If the caller didn't
    // provide them but centered/has_momentum is on, bail with a clear
    // message rather than crashing. If centered/has_momentum is off, we
    // still need SOMETHING to bind at slots 3 and 4 (shader unconditionally
    // declares them) — reuse square_avg as a harmless placeholder.
    const Tensor* momentum_src = &inputs[2];  // placeholder
    const Tensor* gradavg_src  = &inputs[2];  // placeholder
    if (has_momentum) {
        if (inputs.size() <= 4) {
            throw std::invalid_argument(
                "FusedRMSPropStep: momentum>0 requires inputs[4] momentum buffer");
        }
        momentum_src = &inputs[4];
    }
    if (centered) {
        if (inputs.size() <= 3) {
            throw std::invalid_argument(
                "FusedRMSPropStep: centered=true requires inputs[3] grad_avg buffer");
        }
        gradavg_src = &inputs[3];
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, inputs[1].data_ptr()},       // grad
        {1, inputs[0].data_ptr()},       // param
        {2, inputs[2].data_ptr()},       // square_avg
        {3, momentum_src->data_ptr()},   // momentum_buffer (or placeholder)
        {4, gradavg_src->data_ptr()},    // grad_avg (or placeholder)
    };
    std::vector<size_t> sizes = {buf_size, buf_size, buf_size, buf_size, buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t numel;
        float lr;
        float alpha;
        float eps;
        float weight_decay;
        float momentum;
        uint32_t centered;
        uint32_t has_momentum;
    } pc = {};
    pc.numel = static_cast<uint32_t>(numel);
    pc.lr = static_cast<float>(lr);
    pc.alpha = static_cast<float>(alpha);
    pc.eps = static_cast<float>(eps);
    pc.weight_decay = static_cast<float>(weight_decay);
    pc.momentum = static_cast<float>(momentum);
    pc.centered = centered ? 1u : 0u;
    pc.has_momentum = has_momentum ? 1u : 0u;

    uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {inputs[0]};
}

auto VulkanBackend::dispatchFusedAdadeltaStep(
    std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
    // inputs (from registry): [param, grad, square_avg, acc_delta]
    // shader bindings:         0=grad, 1=param, 2=square_avg, 3=acc_delta
    //
    // Previously this called the generic dispatch(shader, inputs, attrs) helper,
    // which loops back through the op-id dispatch table and re-enters *this*
    // function — infinite recursion → stack overflow → SIGSEGV. Match the
    // direct-pipeline pattern used by FusedAdamStep instead.
    if (inputs.size() < 4) {
        throw std::invalid_argument(
            "FusedAdadeltaStep requires 4 inputs (param, grad, square_avg, acc_delta)");
    }

    double rho = attrs.get_float(AttrKey::Rho, 0.9);
    double eps = attrs.get_float(AttrKey::Eps, 1e-6);
    double lr  = attrs.get_float(AttrKey::Lr, 1.0);
    double weight_decay = attrs.get_float(AttrKey::WeightDecay, 0.0);

    int64_t numel = inputs[0].numel();
    int32_t device_id = inputs[0].device().index;
    DType dt = inputs[0].dtype();
    std::string shader =
        (dt == DType::Float64)  ? "fused_adadelta_step_f64" :
        (dt == DType::Float16)  ? "fused_adadelta_step_f16" :
        (dt == DType::BFloat16) ? "fused_adadelta_step_bf16" :
                                  "fused_adadelta_step";
    auto* pipeline = getPipeline(shader, device_id);

    // Half-precision buffers are packed: 2 FP16/BF16 elements per uint32.
    size_t elem_bytes = inputs[0].dtype_size();
    bool is_half = (dt == DType::Float16 || dt == DType::BFloat16);
    size_t buf_size = is_half ? ((numel + 1) / 2) * 4
                              : numel * elem_bytes;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, inputs[1].data_ptr()},  // grad     -> binding 0
        {1, inputs[0].data_ptr()},  // param    -> binding 1
        {2, inputs[2].data_ptr()},  // squareAvg-> binding 2
        {3, inputs[3].data_ptr()},  // accDelta -> binding 3
    };
    std::vector<size_t> sizes = {buf_size, buf_size, buf_size, buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t numel;
        float lr;
        float rho;
        float eps;
        float weight_decay;
        uint32_t padding0, padding1, padding2;
    } pc = {};
    pc.numel = static_cast<uint32_t>(numel);
    pc.lr = static_cast<float>(lr);
    pc.rho = static_cast<float>(rho);
    pc.eps = static_cast<float>(eps);
    pc.weight_decay = static_cast<float>(weight_decay);

    // Half shaders process 2 elements per invocation (packed uint32 words);
    // scale the dispatch count accordingly.
    int64_t dispatch_count = is_half ? (numel + 1) / 2 : numel;
    uint32_t workgroups = div_wg(dispatch_count, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {inputs[0]};  // param is written in-place
}

auto VulkanBackend::dispatchFusedAdagradStep(
    std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
    // inputs (from registry): [param, grad, sum_sq]
    // shader bindings:         0=grad, 1=param, 2=sum_sq
    // Direct pipeline invocation — same infinite-recursion fix as Adadelta/RMSProp.
    if (inputs.size() < 3) {
        throw std::invalid_argument(
            "FusedAdagradStep requires 3 inputs (param, grad, sum_sq)");
    }

    double lr = attrs.get_float(AttrKey::Lr, 0.01);
    double eps = attrs.get_float(AttrKey::Eps, 1e-10);
    double weight_decay = attrs.get_float(AttrKey::WeightDecay, 0.0);

    int64_t numel = inputs[0].numel();
    int32_t device_id = inputs[0].device().index;
    bool is_f64 = (inputs[0].dtype() == DType::Float64);
    std::string shader = is_f64 ? "fused_adagrad_step_f64" : "fused_adagrad_step";
    auto* pipeline = getPipeline(shader, device_id);

    size_t buf_size = numel * inputs[0].dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, inputs[1].data_ptr()},   // grad
        {1, inputs[0].data_ptr()},   // param
        {2, inputs[2].data_ptr()},   // sum_sq
    };
    std::vector<size_t> sizes = {buf_size, buf_size, buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t numel;
        float lr;
        float eps;
        float weight_decay;
    } pc = {};
    pc.numel = static_cast<uint32_t>(numel);
    pc.lr = static_cast<float>(lr);
    pc.eps = static_cast<float>(eps);
    pc.weight_decay = static_cast<float>(weight_decay);

    uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {inputs[0]};
}


// Phase 11.5: Misc Operations
// ============================================================================

/**
 * @brief Strided fill — fill non-contiguous tensor with a value.
 */
auto VulkanBackend::dispatchStridedFill(Tensor& input, float value) -> void {
    // Always use the strided fill shader to write into the existing buffer.
    // The previous contiguous fast-path called dispatchFill() which creates a
    // NEW tensor, replacing input's impl_ and losing the version counter.
    // For in-place fill, we must write to the existing allocation.

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    bool is_i64 = (input.dtype() == DType::Int64);
    // The generic `strided_fill` shader writes 32-bit floats; for BFloat16
    // buffers that smears two BF16 slots per store and produces [0, v, 0, v]
    // patterns. Dispatch to the dedicated BF16 shader instead. Int64 needs
    // full 8-byte writes or fill_(v) leaves the high half undefined.
    std::string shader =
        is_f64 ? "strided_fill_f64"
        : is_i64 ? "strided_fill_i64"
        : is_f16 ? "strided_fill_f16"
        : is_bf16 ? "strided_fill_bf16"
        : "strided_fill";
    auto* pipeline = getPipeline(shader, device_id);

    int64_t numel = input.numel();
    auto shape = input.shape();
    auto strides = input.strides();
    int ndim = static_cast<int>(shape.size());

    // Compute physical storage extent: max_offset + 1 element
    size_t max_offset = 0;
    for (int d = 0; d < ndim; d++) {
        if (shape[d] > 0) {
            max_offset += static_cast<size_t>(shape[d] - 1) * static_cast<size_t>(strides[d]);
        }
    }
    size_t buffer_size = (max_offset + 1) * input.dtype_size();

    if (is_f64 || is_i64) {
        // 8-byte variants (F64 / I64): pass fill value as two uint32 halves so
        // the shader can reassemble a full 64-bit word per slot.
        struct {
            uint32_t n_elements;
            uint32_t ndim;
            uint32_t fill_value_lo;
            uint32_t fill_value_hi;
            uint32_t shape_stride[16];
        } pc = {};
        pc.n_elements = static_cast<uint32_t>(numel);
        pc.ndim = static_cast<uint32_t>(ndim);

        if (is_f64) {
            double dval = static_cast<double>(value);
            uint64_t bits;
            std::memcpy(&bits, &dval, sizeof(bits));
            pc.fill_value_lo = static_cast<uint32_t>(bits & 0xFFFFFFFF);
            pc.fill_value_hi = static_cast<uint32_t>(bits >> 32);
        } else {
            int64_t ival = static_cast<int64_t>(value);
            uint64_t bits;
            std::memcpy(&bits, &ival, sizeof(bits));
            pc.fill_value_lo = static_cast<uint32_t>(bits & 0xFFFFFFFF);
            pc.fill_value_hi = static_cast<uint32_t>(bits >> 32);
        }

        for (int d = 0; d < std::min(ndim, 8); d++) {
            pc.shape_stride[2 * d] = static_cast<uint32_t>(shape[d]);
            pc.shape_stride[2 * d + 1] = static_cast<uint32_t>(strides[d]);
        }

        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, input.data_ptr()}};
        std::vector<size_t> sizes = {buffer_size};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(numel, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    } else {
        struct {
            uint32_t n_elements;
            uint32_t ndim;
            float fill_value;
            uint32_t shape_stride[16];
        } pc = {};
        pc.n_elements = static_cast<uint32_t>(numel);
        pc.ndim = static_cast<uint32_t>(ndim);

        // The shader declares the buffer as `float[]` and stores `fill_value`
        // directly. For Int32 storage (same 4-byte slot as Float32) we must
        // pass the exact bit pattern so that Int32(v) reads back correctly,
        // not float(v)'s IEEE bits. Reinterpret the target bit pattern as a
        // float so the shader writes those bits verbatim.
        if (input.dtype() == DType::Int32) {
            int32_t ival = static_cast<int32_t>(value);
            uint32_t bits;
            std::memcpy(&bits, &ival, sizeof(uint32_t));
            std::memcpy(&pc.fill_value, &bits, sizeof(float));
        } else {
            pc.fill_value = value;
        }

        for (int d = 0; d < std::min(ndim, 8); d++) {
            pc.shape_stride[2 * d] = static_cast<uint32_t>(shape[d]);
            pc.shape_stride[2 * d + 1] = static_cast<uint32_t>(strides[d]);
        }

        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, input.data_ptr()}};
        std::vector<size_t> sizes = {buffer_size};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(numel, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }
}

/**
 * @brief Memory format conversion: NCHW <-> NHWC.
 * format: 0 = NCHW->NHWC, 1 = NHWC->NCHW
 */
auto VulkanBackend::dispatchToMemoryFormat(const Tensor& input, int format) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("vulkan to_memory_format: requires 4D tensor (NCHW/NHWC)");
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    std::string shader = is_f64 ? "to_memory_format_f64" : (is_f16 ? "to_memory_format_f16" : "to_memory_format");
    auto* pipeline = getPipeline(shader, device_id);

    int64_t N = shape[0], C, H, W;
    if (format == 0) {
        // NCHW -> NHWC: input is NCHW
        C = shape[1]; H = shape[2]; W = shape[3];
    } else {
        // NHWC -> NCHW: input is NHWC
        H = shape[1]; W = shape[2]; C = shape[3];
    }

    // Output shape depends on format
    std::vector<int64_t> output_shape;
    if (format == 0) {
        output_shape = {N, H, W, C};  // NHWC
    } else {
        output_shape = {N, C, H, W};  // NCHW
    }

    Tensor output(output_shape, input.dtype(), input.device());
    int64_t numel = input.numel();

    struct {
        uint32_t n_elements;
        uint32_t N, C, H, W;
        uint32_t format;
    } pc;
    pc.n_elements = static_cast<uint32_t>(numel);
    pc.N = static_cast<uint32_t>(N);
    pc.C = static_cast<uint32_t>(C);
    pc.H = static_cast<uint32_t>(H);
    pc.W = static_cast<uint32_t>(W);
    pc.format = static_cast<uint32_t>(format);

    size_t buffer_bytes = numel * input.dtype_size();
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buffer_bytes, buffer_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(numel, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

/**
 * @brief Check tensor for inf/nan values. Returns 1-element Bool tensor.
 */
auto VulkanBackend::dispatchHasInfNan(const Tensor& input) -> Tensor {
    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    std::string shader = is_f64 ? "has_inf_nan_f64" : "has_inf_nan";

    // For non-float types, no inf/nan possible
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64 &&
        input.dtype() != DType::Float16 && input.dtype() != DType::BFloat16) {
        Tensor result({1}, DType::Bool, input.device());
        result = dispatchFill(result, 0.0f);
        return result;
    }

    // For Float16/BFloat16, convert to Float32 first
    Tensor work_input = input;
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        work_input = dispatchCast(input, DType::Float32);
        is_f64 = false;
        shader = "has_inf_nan";
    }

    auto* pipeline = getPipeline(shader, device_id);

    // Result buffer — single uint32 initialized to 0
    Tensor result_buf({1}, DType::Int32, input.device());
    result_buf = dispatchFill(result_buf, 0.0f);

    int64_t numel = work_input.numel();
    struct { uint32_t n; } pc;
    pc.n = static_cast<uint32_t>(numel);

    size_t input_bytes = numel * work_input.dtype_size();
    size_t result_bytes = sizeof(uint32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, work_input.data_ptr()}, {1, result_buf.data_ptr()}
    };
    std::vector<size_t> sizes = {input_bytes, result_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(numel, devices_[device_id].workgroupSize), 1, 1);
    insertComputeBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    // Convert Int32 result to Bool
    return dispatchCast(result_buf, DType::Bool);
}

/**
 * @brief Depthwise convolution 2D forward.
 */
auto VulkanBackend::dispatchDepthwiseConv2d(const Tensor& input, const Tensor& weight,
                                              const Tensor* bias, int64_t stride,
                                              int64_t padding, int64_t dilation) -> Tensor {
    auto in_shape = input.shape();  // [N, C, H, W]
    int64_t N = in_shape[0];
    int64_t C = in_shape[1];
    int64_t H = in_shape[2];
    int64_t W = in_shape[3];

    auto w_shape = weight.shape();  // [C, 1, kH, kW]
    int64_t kH = w_shape[2];
    int64_t kW = w_shape[3];

    int64_t out_h = (H + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
    int64_t out_w = (W + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;
    int64_t out_elements = N * C * out_h * out_w;

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "depthwise_conv2d_f64" : is_f16 ? "depthwise_conv2d_f16" : is_bf16 ? "depthwise_conv2d_bf16" : "depthwise_conv2d";
    auto* pipeline = getPipeline(shader, device_id);

    Tensor output({N, C, out_h, out_w}, input.dtype(), input.device());

    // Create dummy bias if not provided
    Tensor bias_tensor;
    if (bias && bias->numel() > 0) {
        bias_tensor = *bias;
    } else {
        bias_tensor = Tensor({1}, input.dtype(), input.device());
        bias_tensor = dispatchFill(bias_tensor, 0.0f);
    }

    struct {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t has_bias;
    } pc;
    pc.n_elements = static_cast<uint32_t>(out_elements);
    pc.batch = static_cast<uint32_t>(N);
    pc.channels = static_cast<uint32_t>(C);
    pc.in_height = static_cast<uint32_t>(H);
    pc.in_width = static_cast<uint32_t>(W);
    pc.kernel_h = static_cast<uint32_t>(kH);
    pc.kernel_w = static_cast<uint32_t>(kW);
    pc.stride = static_cast<uint32_t>(stride);
    pc.padding = static_cast<uint32_t>(padding);
    pc.dilation = static_cast<uint32_t>(dilation);
    pc.out_h = static_cast<uint32_t>(out_h);
    pc.out_w = static_cast<uint32_t>(out_w);
    pc.has_bias = (bias && bias->numel() > 0) ? 1 : 0;

    size_t elem = input.dtype_size();

    // For Float16, descriptor buffer ranges must be rounded up to 4-byte boundaries
    auto f16_buf_size = [&](int64_t numel) -> size_t {
        if (is_f16) {
            size_t num_pairs = (static_cast<size_t>(numel) + 1) / 2;
            return num_pairs * 4;
        }
        return static_cast<size_t>(numel) * elem;
    };

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, weight.data_ptr()},
        {2, bias_tensor.data_ptr()}, {3, output.data_ptr()}
    };
    std::vector<size_t> sizes = {
        f16_buf_size(N * C * H * W),
        f16_buf_size(C * kH * kW),
        f16_buf_size(bias_tensor.numel()),
        f16_buf_size(out_elements)
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(out_elements, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

/**
 * @brief Adaptive max pool 2D backward — routes gradients via saved indices.
 */
auto VulkanBackend::dispatchAdaptiveMaxPool2dBackward(const Tensor& grad_output,
                                                        const Tensor& indices,
                                                        const std::vector<int64_t>& input_shape) -> Tensor {
    // input_shape: [N, C, H_in, W_in]
    int64_t N = input_shape.size() > 0 ? input_shape[0] : 1;
    int64_t C = input_shape.size() > 1 ? input_shape[1] : 1;
    int64_t H_in = input_shape.size() > 2 ? input_shape[2] : 1;
    int64_t W_in = input_shape.size() > 3 ? input_shape[3] : 1;

    int32_t device_id = grad_output.device().index;
    bool is_f64 = (grad_output.dtype() == DType::Float64);
    bool is_f16 = (grad_output.dtype() == DType::Float16);
    bool is_bf16 = (grad_output.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "adaptive_max_pool2d_backward_f64" :
                          (is_f16 ? "adaptive_max_pool2d_backward_f16" :
                          (is_bf16 ? "adaptive_max_pool2d_backward_bf16" : "adaptive_max_pool2d_backward"));
    auto* pipeline = getPipeline(shader, device_id);

    int64_t grad_input_size = N * C * H_in * W_in;
    Tensor grad_input({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());
    // Zero-initialize grad_input
    grad_input = dispatchFill(grad_input, 0.0f);

    int64_t n_elements = grad_output.numel();

    struct {
        uint32_t n_elements;
        uint32_t grad_input_size;
        uint32_t channels;
        uint32_t in_h;
        uint32_t in_w;
    } pc;
    pc.n_elements = static_cast<uint32_t>(n_elements);
    pc.grad_input_size = static_cast<uint32_t>(grad_input_size);
    pc.channels = static_cast<uint32_t>(C);
    pc.in_h = static_cast<uint32_t>(H_in);
    pc.in_w = static_cast<uint32_t>(W_in);

    size_t elem = grad_output.dtype_size();
    // For F16: round buffer sizes up to 4-byte boundary for packed uint32 access
    size_t go_buf_size, gi_buf_size;
    if (is_f16) {
        go_buf_size = ((static_cast<size_t>(n_elements) + 1) / 2) * 4;
        gi_buf_size = ((static_cast<size_t>(grad_input_size) + 1) / 2) * 4;
    } else {
        go_buf_size = static_cast<size_t>(n_elements) * elem;
        gi_buf_size = static_cast<size_t>(grad_input_size) * elem;
    }
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()}, {1, indices.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        go_buf_size,
        static_cast<size_t>(n_elements) * sizeof(int32_t),
        gi_buf_size
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(n_elements, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return grad_input;
}

/**
 * @brief Cumulative sum (prefix sum) along a dimension.
 */
auto VulkanBackend::dispatchCumSum(const Tensor& input, int64_t dim) -> Tensor {
    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());

    // Normalize negative dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("cumsum: dimension out of range (got " +
                                 std::to_string(dim) + " for tensor with " +
                                 std::to_string(ndim) + " dimensions)");
    }

    // No native int shaders — the default path reads int bytes as float and
    // silently produces zero (int 1 reinterpreted as float = denormal ~0).
    // Promote to Float32, cumsum, cast back. NMS relied on this path for
    // its prefix-sum compaction; empty output = int cumsum returning 0s.
    DType src_dtype = input.dtype();
    if (src_dtype == DType::Int32 || src_dtype == DType::Int64 ||
        src_dtype == DType::Int16 || src_dtype == DType::Int8 ||
        src_dtype == DType::UInt8 || src_dtype == DType::Bool) {
        auto f32 = input.to(DType::Float32);
        auto out_f32 = dispatchCumSum(f32, dim);
        return out_f32.to(src_dtype);
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "cumsum_f64" : is_f16 ? "cumsum_f16" : is_bf16 ? "cumsum_bf16" : "cumsum";
    auto* pipeline = getPipeline(shader, device_id);

    Tensor output(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                  input.dtype(), input.device());

    // Compute outer_size (product of dims before dim) and inner_size (product of dims after dim)
    uint32_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= static_cast<uint32_t>(in_shape[i]);
    }
    uint32_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) {
        inner_size *= static_cast<uint32_t>(in_shape[i]);
    }
    uint32_t reduce_size = static_cast<uint32_t>(in_shape[dim]);
    uint32_t total_lines = outer_size * inner_size;

    struct {
        uint32_t total_lines;
        uint32_t reduce_size;
        uint32_t inner_size;
    } pc;
    pc.total_lines = total_lines;
    pc.reduce_size = reduce_size;
    pc.inner_size = inner_size;

    size_t elem = input.dtype_size();
    size_t buf_size = static_cast<size_t>(input.numel()) * elem;
    if (is_f16) {
        size_t num_pairs = (input.numel() + 1) / 2;
        buf_size = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_size, buf_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_lines, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

/**
 * @brief Cumulative product along a dimension.
 */
auto VulkanBackend::dispatchCumProd(const Tensor& input, int64_t dim) -> Tensor {
    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());

    // Normalize negative dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("cumprod: dimension out of range (got " +
                                 std::to_string(dim) + " for tensor with " +
                                 std::to_string(ndim) + " dimensions)");
    }

    // Same int-dtype fall-through issue as dispatchCumSum: the default float
    // shader reads int bytes as float and yields denormal/NaN garbage.
    // Promote to Float32, compute, cast back.
    DType src_dtype = input.dtype();
    if (src_dtype == DType::Int32 || src_dtype == DType::Int64 ||
        src_dtype == DType::Int16 || src_dtype == DType::Int8 ||
        src_dtype == DType::UInt8 || src_dtype == DType::Bool) {
        auto f32 = input.to(DType::Float32);
        auto out_f32 = dispatchCumProd(f32, dim);
        return out_f32.to(src_dtype);
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "cumprod_f64" : is_f16 ? "cumprod_f16" : is_bf16 ? "cumprod_bf16" : "cumprod";
    auto* pipeline = getPipeline(shader, device_id);

    Tensor output(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                  input.dtype(), input.device());

    uint32_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= static_cast<uint32_t>(in_shape[i]);
    }
    uint32_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) {
        inner_size *= static_cast<uint32_t>(in_shape[i]);
    }
    uint32_t reduce_size = static_cast<uint32_t>(in_shape[dim]);
    uint32_t total_lines = outer_size * inner_size;

    struct {
        uint32_t total_lines;
        uint32_t reduce_size;
        uint32_t inner_size;
    } pc;
    pc.total_lines = total_lines;
    pc.reduce_size = reduce_size;
    pc.inner_size = inner_size;

    size_t elem = input.dtype_size();
    size_t buf_size = static_cast<size_t>(input.numel()) * elem;
    if (is_f16) {
        size_t num_pairs = (input.numel() + 1) / 2;
        buf_size = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_size, buf_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_lines, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ============================================================================
// Log-Cumulative-Sum-Exp (logcumsumexp) along a dimension
// ============================================================================

auto VulkanBackend::dispatchLogcumsumexp(const Tensor& input, int64_t dim) -> Tensor {
    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());

    // Normalize negative dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("logcumsumexp: dimension out of range (got " +
                                 std::to_string(dim) + " for tensor with " +
                                 std::to_string(ndim) + " dimensions)");
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "logcumsumexp_f64" : is_f16 ? "logcumsumexp_f16"
                       : is_bf16 ? "logcumsumexp_bf16" : "logcumsumexp";
    auto* pipeline = getPipeline(shader, device_id);

    Tensor output(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                  input.dtype(), input.device());

    uint32_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= static_cast<uint32_t>(in_shape[i]);
    }
    uint32_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) {
        inner_size *= static_cast<uint32_t>(in_shape[i]);
    }
    uint32_t reduce_size = static_cast<uint32_t>(in_shape[dim]);
    uint32_t total_lines = outer_size * inner_size;

    struct {
        uint32_t total_lines;
        uint32_t reduce_size;
        uint32_t inner_size;
    } pc;
    pc.total_lines = total_lines;
    pc.reduce_size = reduce_size;
    pc.inner_size = inner_size;

    size_t elem = input.dtype_size();
    size_t buf_size = static_cast<size_t>(input.numel()) * elem;
    if (is_f16 || is_bf16) {
        size_t num_pairs = (input.numel() + 1) / 2;
        buf_size = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_size, buf_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_lines, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ============================================================================
// Bincount — count occurrences of integer values, optionally with weights
// ============================================================================

auto VulkanBackend::dispatchBincount(const Tensor& input, const std::optional<Tensor>& weights, int64_t minlength) -> Tensor {
    int32_t device_id = input.device().index;

    // Input must be Int32 for the shader
    Tensor int_input = (input.dtype() != DType::Int32) ? dispatchCast(input, DType::Int32) : input;

    // Determine number of bins: max(max_val + 1, minlength)
    // We need the max value from the input. Use a reduction.
    int64_t num_bins = minlength;
    if (input.numel() > 0) {
        // Get max value by moving a small result to CPU
        Tensor max_val = dispatchReduction("max", int_input.to(DType::Float32), -1, false);
        Tensor max_cpu = max_val.to(Device::cpu());
        int64_t max_v = static_cast<int64_t>(static_cast<const float*>(max_cpu.data_ptr())[0]);
        num_bins = std::max(num_bins, max_v + 1);
    }

    if (num_bins == 0) {
        return Tensor({0}, DType::Float32, input.device());
    }

    // Allocate output and zero-initialize
    Tensor output = dispatchFull({num_bins}, 0.0f, DType::Float32);
    // Move output to the correct device if needed
    if (output.device() != input.device()) {
        output = output.to(input.device());
    }

    if (input.numel() == 0) {
        return output;
    }

    bool has_weights = weights.has_value();
    Tensor w_tensor = has_weights ? weights.value() : dispatchFull({1}, 1.0f, DType::Float32);
    if (has_weights && w_tensor.dtype() != DType::Float32) {
        w_tensor = dispatchCast(w_tensor, DType::Float32);
    }

    auto* pipeline = getPipeline("bincount", device_id);

    uint32_t n = static_cast<uint32_t>(input.numel());

    struct {
        uint32_t n;
        uint32_t num_bins;
        uint32_t has_weights;
    } pc;
    pc.n = n;
    pc.num_bins = static_cast<uint32_t>(num_bins);
    pc.has_weights = has_weights ? 1 : 0;

    size_t input_buf_size = static_cast<size_t>(n) * sizeof(int32_t);
    size_t weights_buf_size = has_weights ? static_cast<size_t>(n) * sizeof(float) : sizeof(float);
    size_t output_buf_size = static_cast<size_t>(num_bins) * sizeof(float);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, int_input.data_ptr()}, {1, w_tensor.data_ptr()}, {2, output.data_ptr()}
    };
    std::vector<size_t> sizes_vec = {input_buf_size, weights_buf_size, output_buf_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes_vec);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(n, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ============================================================================
// Bool Predicate Operations (isnan, isinf, isfinite)
// ============================================================================

auto VulkanBackend::dispatchBoolPredicateOp(const std::string& op_name,
                                             const Tensor& input) -> Tensor {
    // Handle empty tensors
    if (input.numel() == 0) {
        auto input_shape = input.shape();
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
        return Tensor(output_shape, DType::Bool, input.device());
    }

    // BFloat16: upcast to Float32
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        return dispatchBoolPredicateOp(op_name, input_f32);
    }

    // Float16: upcast to Float32 (no packed-pair bool_predicates shader)
    if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        return dispatchBoolPredicateOp(op_name, input_f32);
    }

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = (input.dtype() == DType::Float64) ? "bool_predicates_f64" : "bool_predicates";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create Bool output tensor
    auto input_shape = input.shape();
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, DType::Bool, input.device());

    // Map operation name to opcode
    uint32_t opcode = 0;
    if (op_name == "isnan") opcode = 0;
    else if (op_name == "isinf") opcode = 1;
    else if (op_name == "isfinite") opcode = 2;
    else if (op_name == "signbit") opcode = 3;
    else if (op_name == "isposinf") opcode = 4;
    else if (op_name == "isneginf") opcode = 5;
    else throw std::runtime_error("Unknown bool predicate operation: " + op_name);

    struct PushConstants {
        uint32_t n;
        uint32_t op;
    } push_constants;
    push_constants.n = static_cast<uint32_t>(input.numel());
    push_constants.op = opcode;

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in}, {1, buffer_out}
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
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Logical Operations (and, or, not, xor)
// ============================================================================

auto VulkanBackend::dispatchLogicalOp(const std::string& op_name,
                                       const Tensor& a, const Tensor& b) -> Tensor {
    // Handle empty tensors
    if (a.numel() == 0) {
        auto a_shape = a.shape();
        std::vector<int64_t> output_shape(a_shape.begin(), a_shape.end());
        return Tensor(output_shape, DType::Bool, a.device());
    }

    int32_t device_id = a.device().index;

    // Convert inputs to Bool if not already
    Tensor a_bool = (a.dtype() == DType::Bool) ? a : a.to(DType::Bool);
    Tensor b_bool = (b.dtype() == DType::Bool) ? b : b.to(DType::Bool);

    auto* pipeline = getPipeline("logical", device_id);

    auto a_shape = a_bool.shape();
    std::vector<int64_t> output_shape(a_shape.begin(), a_shape.end());
    Tensor output(output_shape, DType::Bool, a.device());

    // Map operation name to opcode
    uint32_t opcode = 0;
    if (op_name == "logical_and") opcode = 0;
    else if (op_name == "logical_or") opcode = 1;
    else if (op_name == "logical_not") opcode = 2;
    else if (op_name == "logical_xor") opcode = 3;
    else throw std::runtime_error("Unknown logical operation: " + op_name);

    struct PushConstants {
        uint32_t n;
        uint32_t op;
    } push_constants;
    push_constants.n = static_cast<uint32_t>(a_bool.numel());
    push_constants.op = opcode;

    const void* buffer_a = a_bool.data_ptr();
    const void* buffer_b = b_bool.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size = a_bool.numel() * a_bool.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_a}, {1, buffer_b}, {2, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(a_bool.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Lerp (Linear Interpolation)
// ============================================================================

auto VulkanBackend::dispatchLerp(const Tensor& start, const Tensor& end,
                                  const Tensor& weight) -> Tensor {
    // Handle empty tensors
    if (start.numel() == 0) {
        auto start_shape = start.shape();
        std::vector<int64_t> output_shape(start_shape.begin(), start_shape.end());
        return Tensor(output_shape, start.dtype(), start.device());
    }

    int32_t device_id = start.device().index;
    bool is_float64 = (start.dtype() == DType::Float64);
    bool is_float16 = (start.dtype() == DType::Float16);
    bool is_bfloat16 = (start.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "lerp_f64" : is_float16 ? "lerp_f16" : is_bfloat16 ? "lerp_bf16" : "lerp";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto start_shape = start.shape();
    std::vector<int64_t> output_shape(start_shape.begin(), start_shape.end());
    Tensor output(output_shape, start.dtype(), start.device());

    // Broadcast scalar or partial-shape weight tensor up to start's shape.
    // The lerp shaders read weight_buf[idx] per-element, so passing a
    // shape-{1} scalar made every thread past idx=0 read out of the
    // descriptor range and get 0 (which collapsed the output to `start`).
    Tensor weight_broadcast = weight;
    bool shapes_match = (weight.shape().size() == start.shape().size());
    if (shapes_match) {
        for (size_t i = 0; i < start.shape().size(); ++i) {
            if (weight.shape()[i] != start.shape()[i]) { shapes_match = false; break; }
        }
    }
    if (weight.numel() != start.numel() || !shapes_match) {
        std::vector<int64_t> target_shape(start_shape.begin(), start_shape.end());
        // Expand requires ndim matching; left-pad weight's shape with 1s.
        if (static_cast<int64_t>(weight.shape().size()) < static_cast<int64_t>(target_shape.size())) {
            std::vector<int64_t> padded(target_shape.size(), 1);
            int64_t offset = static_cast<int64_t>(target_shape.size()) - static_cast<int64_t>(weight.shape().size());
            for (size_t i = 0; i < weight.shape().size(); ++i) {
                padded[offset + i] = weight.shape()[i];
            }
            weight_broadcast = weight.reshape(padded);
        }
        weight_broadcast = dispatchExpand(weight_broadcast, target_shape);
        weight_broadcast = weight_broadcast.is_contiguous() ? weight_broadcast : weight_broadcast.contiguous();
    }

    struct PushConstants {
        uint32_t n;
    } push_constants;
    push_constants.n = static_cast<uint32_t>(start.numel());

    const void* buffer_start = start.data_ptr();
    const void* buffer_end = end.data_ptr();
    const void* buffer_weight = weight_broadcast.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t elem_size = start.dtype_size();
    size_t buffer_size = start.numel() * elem_size;
    if (is_float16) {
        size_t num_pairs = (start.numel() + 1) / 2;
        buffer_size = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_start}, {1, buffer_end}, {2, buffer_weight}, {3, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size, buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t num_work_items;
    if (is_float16) {
        num_work_items = static_cast<uint32_t>((start.numel() + 1) / 2);
    } else {
        num_work_items = static_cast<uint32_t>(start.numel());
    }
    uint32_t workgroups = div_wg(num_work_items, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}


// Addcmul: result = input + value * tensor1 * tensor2
//
// fp32 path uses the addcmul.comp shader; fp64 path uses addcmul_f64.comp
// which has a doubled push-constant layout (8-byte scalar multiplier).
// Half precision (Float16/BFloat16) is promoted via the generic Float32
// path by the caller contract — the Tenzor autograd functions cast before
// calling here.
auto VulkanBackend::dispatchAddcmul(const Tensor& input, const Tensor& tensor1,
                                     const Tensor& tensor2, float value) -> Tensor {
    if (input.numel() == 0) {
        auto inp_shape = input.shape();
        std::vector<int64_t> output_shape(inp_shape.begin(), inp_shape.end());
        return Tensor(output_shape, input.dtype(), input.device());
    }

    // Float16 / BFloat16 have no dedicated shader — promote to Float32,
    // compute, then narrow back. Keeps the fp32 shader as the single
    // implementation for every non-fp64 dtype.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto out_f32 = dispatchAddcmul(input.to(DType::Float32),
                                        tensor1.to(DType::Float32),
                                        tensor2.to(DType::Float32), value);
        return out_f32.to(input.dtype());
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    auto* pipeline = getPipeline(is_f64 ? "addcmul_f64" : "addcmul", device_id);

    auto inp_shape = input.shape();
    std::vector<int64_t> output_shape(inp_shape.begin(), inp_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // fp32 and fp64 variants share the (uint32 n, scalar value) layout but
    // with different scalar widths. We pack a 16-byte struct either way to
    // satisfy alignment (the fp64 shader aligns its `double` to 8).
    struct PushConstantsF32 { uint32_t n; float value; uint32_t pad0; uint32_t pad1; };
    struct PushConstantsF64 { uint32_t n; uint32_t pad; double value; };

    size_t pc_size = is_f64 ? sizeof(PushConstantsF64) : sizeof(PushConstantsF32);
    union {
        PushConstantsF32 f32;
        PushConstantsF64 f64;
    } push_constants = {};
    if (is_f64) {
        push_constants.f64.n = static_cast<uint32_t>(input.numel());
        push_constants.f64.value = static_cast<double>(value);
    } else {
        push_constants.f32.n = static_cast<uint32_t>(input.numel());
        push_constants.f32.value = value;
    }

    size_t buffer_size = input.numel() * input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, tensor1.data_ptr()}, {2, tensor2.data_ptr()}, {3, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size, buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, static_cast<uint32_t>(pc_size), &push_constants);

    uint32_t workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// Addcdiv: result = input + value * tensor1 / tensor2
//
// See dispatchAddcmul for the fp32/fp64 push-constant layout rationale.
auto VulkanBackend::dispatchAddcdiv(const Tensor& input, const Tensor& tensor1,
                                     const Tensor& tensor2, float value) -> Tensor {
    if (input.numel() == 0) {
        auto inp_shape = input.shape();
        std::vector<int64_t> output_shape(inp_shape.begin(), inp_shape.end());
        return Tensor(output_shape, input.dtype(), input.device());
    }

    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto out_f32 = dispatchAddcdiv(input.to(DType::Float32),
                                        tensor1.to(DType::Float32),
                                        tensor2.to(DType::Float32), value);
        return out_f32.to(input.dtype());
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    auto* pipeline = getPipeline(is_f64 ? "addcdiv_f64" : "addcdiv", device_id);

    auto inp_shape = input.shape();
    std::vector<int64_t> output_shape(inp_shape.begin(), inp_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    struct PushConstantsF32 { uint32_t n; float value; uint32_t pad0; uint32_t pad1; };
    struct PushConstantsF64 { uint32_t n; uint32_t pad; double value; };
    size_t pc_size = is_f64 ? sizeof(PushConstantsF64) : sizeof(PushConstantsF32);
    union { PushConstantsF32 f32; PushConstantsF64 f64; } push_constants = {};
    if (is_f64) {
        push_constants.f64.n = static_cast<uint32_t>(input.numel());
        push_constants.f64.value = static_cast<double>(value);
    } else {
        push_constants.f32.n = static_cast<uint32_t>(input.numel());
        push_constants.f32.value = value;
    }

    size_t buffer_size = input.numel() * input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, tensor1.data_ptr()}, {2, tensor2.data_ptr()}, {3, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size, buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, static_cast<uint32_t>(pc_size), &push_constants);

    uint32_t workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}


// ScatterAdd (OpId 410)
// ============================================================================

auto VulkanBackend::dispatchScatterAdd(const Tensor& self, int64_t dim,
                                        const Tensor& index, const Tensor& src) -> Tensor {
    auto self_shape = self.shape();

    // Handle empty tensors
    if (self.numel() == 0 || index.numel() == 0) {
        // Phase 7.6 EmptyIndex fix: when there's nothing to scatter, the
        // output equals `self`. Must clone its data — allocating a fresh
        // tensor of the same shape leaves it uninitialised, which on
        // Vulkan defaults to zeros and silently corrupts the result.
        return self.clone();
    }

    int32_t device_id = self.device().index;

    // Float16: upcast to Float32 for scatter_add (atomic accumulation in F32)
    if (self.dtype() == DType::Float16) {
        DType orig_dtype = self.dtype();
        auto self_f32 = self.to(DType::Float32);
        auto src_f32 = src.to(DType::Float32);
        auto result_f32 = dispatchScatterAdd(self_f32, dim, index, src_f32);
        return result_f32.to(orig_dtype);
    }

    // Float64 requires atomic int64 for CAS-loop atomics
    if (self.dtype() == DType::Float64 && !devices_[device_id].hasAtomicInt64) {
        throw std::runtime_error(
            "ScatterAdd with Float64 requires VK_KHR_shader_atomic_int64 support. "
            "Use CPU backend or Float32 for this device.");
    }

    // Select shader based on dtype
    const char* shader_name = (self.dtype() == DType::Float64) ? "scatter_add_f64"
                            : (self.dtype() == DType::Float16) ? "scatter_add_f16"
                            : (self.dtype() == DType::BFloat16) ? "scatter_add_bf16"
                            : "scatter_add";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Normalize dimension
    int64_t ndim = self.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("ScatterAdd: dimension out of range");
    }

    // Create output as copy of self
    std::vector<int64_t> out_shape(self_shape.begin(), self_shape.end());
    Tensor output(out_shape, self.dtype(), self.device());
    size_t bytes = self.numel() * self.dtype_size();
    copy(output.data_ptr(), self.data_ptr(), bytes, CopyKind::DeviceToDevice);

    // Convert Int64 indices to Int32 for shader compatibility
    Tensor index_int32 = index;
    if (index.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(index.shape().begin(), index.shape().end());
        index_int32 = Tensor(idx_shape, DType::Int32, index.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = index.data_ptr();
        const void* buf_out = index_int32.data_ptr();
        size_t size_in = index.numel() * sizeof(int64_t);
        size_t size_out = index_int32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(index.numel());

        uint32_t cast_groups = div_wg(cast_pc.n_elements, devices_[device_id].workgroupSize);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, cast_groups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Compute scatter parameters
    uint32_t dim_size = static_cast<uint32_t>(self_shape[dim]);
    uint32_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= static_cast<uint32_t>(self_shape[d]);
    }

    auto src_shape = src.shape();
    uint32_t src_dim_size = static_cast<uint32_t>(src_shape[dim]);

    // Buffers
    const void* buf_src = src.data_ptr();
    const void* buf_idx = index_int32.data_ptr();
    const void* buf_out = output.data_ptr();

    size_t buf_src_size = src.numel() * src.dtype_size();
    size_t buf_idx_size = index_int32.numel() * sizeof(int32_t);
    size_t buf_out_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_src}, {1, buf_idx}, {2, buf_out}
    };
    std::vector<size_t> sizes = {buf_src_size, buf_idx_size, buf_out_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t n;
        uint32_t dim_size;
        uint32_t inner_size;
        uint32_t src_dim_size;
    } push_constants;

    push_constants.n = static_cast<uint32_t>(index.numel());
    push_constants.dim_size = dim_size;
    push_constants.inner_size = inner_size;
    push_constants.src_dim_size = src_dim_size;

    uint32_t workgroups = div_wg(index.numel(), devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push_constants), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// ScatterReduce (native Vulkan)
// ============================================================================

auto VulkanBackend::dispatchScatterReduce(const Tensor& self, int64_t dim,
                                           const Tensor& index, const Tensor& src,
                                           const std::string& reduce, bool include_self) -> Tensor {
    auto self_shape = self.shape();

    if (self.numel() == 0 || index.numel() == 0) {
        // Phase 7.6 EmptyIndex fix: when there's nothing to scatter, the
        // output equals `self`. Must clone its data — allocating a fresh
        // tensor of the same shape leaves it uninitialised, which on
        // Vulkan defaults to zeros and silently corrupts the result.
        return self.clone();
    }

    int32_t device_id = self.device().index;

    // Non-Float32: upcast, compute, downcast
    if (self.dtype() != DType::Float32) {
        DType orig_dtype = self.dtype();
        auto self_f32 = self.to(DType::Float32);
        auto src_f32 = src.to(DType::Float32);
        auto result_f32 = dispatchScatterReduce(self_f32, dim, index, src_f32, reduce, include_self);
        return result_f32.to(orig_dtype);
    }

    // Determine mode
    uint32_t mode;
    if (reduce == "sum") mode = 0;
    else if (reduce == "prod") mode = 1;
    else if (reduce == "mean") mode = 2;
    else if (reduce == "amax") mode = 3;
    else if (reduce == "amin") mode = 4;
    else throw std::invalid_argument("scatter_reduce: unknown reduce mode '" + reduce + "'");

    auto* pipeline = getPipeline("scatter_reduce", device_id);

    // Normalize dimension
    int64_t ndim = self.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("ScatterReduce: dimension out of range");
    }

    // Create output as copy of self
    std::vector<int64_t> out_shape(self_shape.begin(), self_shape.end());
    Tensor output(out_shape, self.dtype(), self.device());
    size_t bytes = self.numel() * self.dtype_size();
    copy(output.data_ptr(), self.data_ptr(), bytes, CopyKind::DeviceToDevice);

    // If !include_self, initialize touched positions to identity
    // For simplicity, we re-use the index_add pattern: we initialize using a separate
    // dispatch with the init shader. For the MVP, we handle this by falling back to CPU
    // for !include_self. This is uncommon for GNN use cases.
    // Phase 7.6 SumNoIncludeSelf fix: when include_self=false, every output
    // position that the subsequent scatter touches must be reset to the
    // mode's identity value first. Untouched positions keep their input
    // value. The init pass below uses the same per-mode identity logic as
    // the CUDA / ROCm init kernels.
    //
    // We dispatch the init shader BEFORE the index-cast block so it can run
    // on the original Int64 index buffer — but the init shader expects an
    // Int32 index buffer for shader compatibility. Defer the dispatch
    // until after the index-cast block has produced `index_int32`.

    // Convert Int64 indices to Int32 for shader compatibility
    Tensor index_int32 = index;
    if (index.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(index.shape().begin(), index.shape().end());
        index_int32 = Tensor(idx_shape, DType::Int32, index.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = index.data_ptr();
        const void* buf_out = index_int32.data_ptr();
        size_t size_in = index.numel() * sizeof(int64_t);
        size_t size_out = index_int32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(index.numel());

        uint32_t cast_groups = div_wg(cast_pc.n_elements, devices_[device_id].workgroupSize);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, cast_groups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Compute index op parameters.
    // Phase 7.6 2D scatter fix: idx_n must be the index tensor's size along
    // the scatter dim, NOT the total numel. For 1D scatter these are equal.
    uint32_t dim_size = static_cast<uint32_t>(self_shape[dim]);
    uint32_t idx_n = (index.shape().size() > static_cast<size_t>(dim))
                   ? static_cast<uint32_t>(index.shape()[dim])
                   : static_cast<uint32_t>(index.numel());
    uint32_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= static_cast<uint32_t>(self_shape[d]);
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= static_cast<uint32_t>(self_shape[d]);

    uint32_t total = outer * idx_n * inner;
    if (total == 0) return output;

    // Phase 7.6 SumNoIncludeSelf fix: for !include_self, run the init
    // shader on every position that the subsequent scatter will touch,
    // setting it to the mode's identity. Uses the just-built `index_int32`
    // (cast above).
    if (!include_self) {
        auto* init_pipeline = getPipeline("scatter_reduce_init", device_id);
        std::vector<std::pair<uint32_t, const void*>> init_bindings = {
            {0, output.data_ptr()},
            {1, index_int32.data_ptr()},
        };
        std::vector<size_t> init_sizes = {
            static_cast<size_t>(output.numel()) * sizeof(float),
            static_cast<size_t>(index_int32.numel()) * sizeof(int32_t),
        };
        VkDescriptorSet init_ds = allocateAndWriteDescriptorSet(
            device_id, init_pipeline, init_bindings, init_sizes);

        struct InitPC {
            uint32_t outer;
            uint32_t dim_size;
            uint32_t idx_n;
            uint32_t inner;
            uint32_t mode;
        } init_pc;
        init_pc.outer = outer;
        init_pc.dim_size = dim_size;
        init_pc.idx_n = idx_n;
        init_pc.inner = inner;
        init_pc.mode = mode;

        uint32_t init_groups = div_wg(total, devices_[device_id].workgroupSize);
        VkCommandBuffer init_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(init_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, init_pipeline->pipeline());
        vkCmdBindDescriptorSets(init_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               init_pipeline->layout(), 0, 1, &init_ds, 0, nullptr);
        vkCmdPushConstants(init_cmd, init_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(init_pc), &init_pc);
        vkCmdDispatch(init_cmd, init_groups, 1, 1);
        insertComputeOnlyBarrier(init_cmd);
        endSingleTimeCommands(init_cmd, device_id);
    }

    // Allocate count buffer for mean mode (Phase 7.6 fix — was previously
    // allocated but never zeroed, leaving stale memory in the count slots
    // and breaking the mean divisor pass).
    int64_t out_numel = output.numel();
    Tensor count_tensor;
    if (mode == 2) {
        count_tensor = Tensor({out_numel}, DType::Int32, output.device());
        size_t count_bytes = static_cast<size_t>(out_numel) * sizeof(int32_t);
        memset(count_tensor.data_ptr(), 0, count_bytes, device_id);
    }

    // Buffers: binding 0 = output (uint for atomics), 1 = source, 2 = index, 3 = counts
    const void* buf_out = output.data_ptr();
    const void* buf_src = src.data_ptr();
    const void* buf_idx = index_int32.data_ptr();
    const void* buf_cnt = (mode == 2) ? count_tensor.data_ptr() : buf_out; // dummy if not mean

    size_t buf_out_size = output.numel() * output.dtype_size();
    size_t buf_src_size = src.numel() * src.dtype_size();
    size_t buf_idx_size = index_int32.numel() * sizeof(int32_t);
    size_t buf_cnt_size = (mode == 2) ? out_numel * sizeof(int32_t) : sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_out}, {1, buf_src}, {2, buf_idx}, {3, buf_cnt}
    };
    std::vector<size_t> sizes = {buf_out_size, buf_src_size, buf_idx_size, buf_cnt_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t outer;
        uint32_t dim_size;
        uint32_t idx_n;
        uint32_t inner;
        uint32_t mode;
    } push_constants;

    push_constants.outer = outer;
    push_constants.dim_size = dim_size;
    push_constants.idx_n = idx_n;
    push_constants.inner = inner;
    push_constants.mode = mode;

    uint32_t workgroups = div_wg(total, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push_constants), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // Phase 7.6 mean fix: second dispatch to divide each output element by
    // its scatter count (+1 if include_self). Mirrors the CUDA / ROCm /
    // OneAPI post-pass.
    if (mode == 2) {
        auto* div_pipeline = getPipeline("scatter_reduce_mean_div", device_id);
        std::vector<std::pair<uint32_t, const void*>> div_bindings = {
            {0, output.data_ptr()},
            {1, count_tensor.data_ptr()},
        };
        std::vector<size_t> div_sizes = {
            static_cast<size_t>(out_numel) * sizeof(float),
            static_cast<size_t>(out_numel) * sizeof(int32_t),
        };
        VkDescriptorSet div_ds = allocateAndWriteDescriptorSet(
            device_id, div_pipeline, div_bindings, div_sizes);

        struct MeanDivPC {
            uint32_t numel;
            uint32_t include_self;
        } div_pc;
        div_pc.numel = static_cast<uint32_t>(out_numel);
        div_pc.include_self = include_self ? 1u : 0u;

        uint32_t div_groups = div_wg(div_pc.numel, devices_[device_id].workgroupSize);
        VkCommandBuffer div_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(div_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, div_pipeline->pipeline());
        vkCmdBindDescriptorSets(div_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               div_pipeline->layout(), 0, 1, &div_ds, 0, nullptr);
        vkCmdPushConstants(div_cmd, div_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(div_pc), &div_pc);
        vkCmdDispatch(div_cmd, div_groups, 1, 1);
        insertComputeOnlyBarrier(div_cmd);
        endSingleTimeCommands(div_cmd, device_id);
    }

    return output;
}

// ============================================================================
// IndexAdd (native Vulkan)
// ============================================================================

auto VulkanBackend::dispatchIndexAdd(const Tensor& self, int64_t dim,
                                      const Tensor& index, const Tensor& src) -> Tensor {
    auto self_shape = self.shape();

    // Handle empty tensors
    if (self.numel() == 0 || index.numel() == 0) {
        // Phase 7.6 EmptyIndex fix: when there's nothing to scatter, the
        // output equals `self`. Must clone its data — allocating a fresh
        // tensor of the same shape leaves it uninitialised, which on
        // Vulkan defaults to zeros and silently corrupts the result.
        return self.clone();
    }

    int32_t device_id = self.device().index;

    // Non-Float32: upcast, compute, downcast
    if (self.dtype() != DType::Float32) {
        DType orig_dtype = self.dtype();
        auto self_f32 = self.to(DType::Float32);
        auto src_f32 = src.to(DType::Float32);
        auto result_f32 = dispatchIndexAdd(self_f32, dim, index, src_f32);
        return result_f32.to(orig_dtype);
    }

    auto* pipeline = getPipeline("index_add", device_id);

    // Normalize dimension
    int64_t ndim = self.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("IndexAdd: dimension out of range");
    }

    // Create output as copy of self
    std::vector<int64_t> out_shape(self_shape.begin(), self_shape.end());
    Tensor output(out_shape, self.dtype(), self.device());
    size_t bytes = self.numel() * self.dtype_size();
    copy(output.data_ptr(), self.data_ptr(), bytes, CopyKind::DeviceToDevice);

    // Convert Int64 indices to Int32 for shader compatibility
    Tensor index_int32 = index;
    if (index.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(index.shape().begin(), index.shape().end());
        index_int32 = Tensor(idx_shape, DType::Int32, index.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = index.data_ptr();
        const void* buf_out = index_int32.data_ptr();
        size_t size_in = index.numel() * sizeof(int64_t);
        size_t size_out = index_int32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(index.numel());

        uint32_t cast_groups = div_wg(cast_pc.n_elements, devices_[device_id].workgroupSize);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, cast_groups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Compute index op parameters
    uint32_t dim_size = static_cast<uint32_t>(self_shape[dim]);
    uint32_t idx_n = static_cast<uint32_t>(index.numel());
    uint32_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= static_cast<uint32_t>(self_shape[d]);
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= static_cast<uint32_t>(self_shape[d]);

    uint32_t total = outer * idx_n * inner;
    if (total == 0) return output;

    // Buffers: binding 0 = output (uint for atomics), 1 = source, 2 = index
    const void* buf_out = output.data_ptr();
    const void* buf_src = src.data_ptr();
    const void* buf_idx = index_int32.data_ptr();

    size_t buf_out_size = output.numel() * output.dtype_size();
    size_t buf_src_size = src.numel() * src.dtype_size();
    size_t buf_idx_size = index_int32.numel() * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_out}, {1, buf_src}, {2, buf_idx}
    };
    std::vector<size_t> sizes = {buf_out_size, buf_src_size, buf_idx_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t outer;
        uint32_t dim_size;
        uint32_t idx_n;
        uint32_t inner;
    } push_constants;

    push_constants.outer = outer;
    push_constants.dim_size = dim_size;
    push_constants.idx_n = idx_n;
    push_constants.inner = inner;

    uint32_t workgroups = div_wg(total, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push_constants), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// IndexCopy (native Vulkan)
// ============================================================================

auto VulkanBackend::dispatchIndexCopy(const Tensor& self, int64_t dim,
                                       const Tensor& index, const Tensor& src) -> Tensor {
    auto self_shape = self.shape();

    if (self.numel() == 0 || index.numel() == 0) {
        // Phase 7.6 EmptyIndex fix: when there's nothing to scatter, the
        // output equals `self`. Must clone its data — allocating a fresh
        // tensor of the same shape leaves it uninitialised, which on
        // Vulkan defaults to zeros and silently corrupts the result.
        return self.clone();
    }

    int32_t device_id = self.device().index;

    // Non-Float32: upcast, compute, downcast
    if (self.dtype() != DType::Float32) {
        DType orig_dtype = self.dtype();
        auto self_f32 = self.to(DType::Float32);
        auto src_f32 = src.to(DType::Float32);
        auto result_f32 = dispatchIndexCopy(self_f32, dim, index, src_f32);
        return result_f32.to(orig_dtype);
    }

    auto* pipeline = getPipeline("index_copy", device_id);

    int64_t ndim = self.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("IndexCopy: dimension out of range");
    }

    // Create output as copy of self
    std::vector<int64_t> out_shape(self_shape.begin(), self_shape.end());
    Tensor output(out_shape, self.dtype(), self.device());
    size_t bytes = self.numel() * self.dtype_size();
    copy(output.data_ptr(), self.data_ptr(), bytes, CopyKind::DeviceToDevice);

    // Convert Int64 indices to Int32
    Tensor index_int32 = index;
    if (index.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(index.shape().begin(), index.shape().end());
        index_int32 = Tensor(idx_shape, DType::Int32, index.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = index.data_ptr();
        const void* buf_out = index_int32.data_ptr();
        size_t size_in = index.numel() * sizeof(int64_t);
        size_t size_out = index_int32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(index.numel());

        uint32_t cast_groups = div_wg(cast_pc.n_elements, devices_[device_id].workgroupSize);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, cast_groups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    uint32_t dim_size = static_cast<uint32_t>(self_shape[dim]);
    uint32_t idx_n = static_cast<uint32_t>(index.numel());
    uint32_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= static_cast<uint32_t>(self_shape[d]);
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= static_cast<uint32_t>(self_shape[d]);

    uint32_t total = outer * idx_n * inner;
    if (total == 0) return output;

    // Buffers: binding 0 = output, 1 = source, 2 = index
    const void* buf_out = output.data_ptr();
    const void* buf_src = src.data_ptr();
    const void* buf_idx = index_int32.data_ptr();

    size_t buf_out_size = output.numel() * output.dtype_size();
    size_t buf_src_size = src.numel() * src.dtype_size();
    size_t buf_idx_size = index_int32.numel() * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_out}, {1, buf_src}, {2, buf_idx}
    };
    std::vector<size_t> sizes = {buf_out_size, buf_src_size, buf_idx_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t outer;
        uint32_t dim_size;
        uint32_t idx_n;
        uint32_t inner;
    } push_constants;

    push_constants.outer = outer;
    push_constants.dim_size = dim_size;
    push_constants.idx_n = idx_n;
    push_constants.inner = inner;

    uint32_t workgroups = div_wg(total, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push_constants), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// IndexFill (native Vulkan)
// ============================================================================

auto VulkanBackend::dispatchIndexFill(const Tensor& self, int64_t dim,
                                       const Tensor& index, float value) -> Tensor {
    auto self_shape = self.shape();

    if (self.numel() == 0 || index.numel() == 0) {
        // Phase 7.6 EmptyIndex fix: when there's nothing to scatter, the
        // output equals `self`. Must clone its data — allocating a fresh
        // tensor of the same shape leaves it uninitialised, which on
        // Vulkan defaults to zeros and silently corrupts the result.
        return self.clone();
    }

    int32_t device_id = self.device().index;

    // Non-Float32: upcast, compute, downcast
    if (self.dtype() != DType::Float32) {
        DType orig_dtype = self.dtype();
        auto self_f32 = self.to(DType::Float32);
        auto result_f32 = dispatchIndexFill(self_f32, dim, index, value);
        return result_f32.to(orig_dtype);
    }

    auto* pipeline = getPipeline("index_fill", device_id);

    int64_t ndim = self.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("IndexFill: dimension out of range");
    }

    // Create output as copy of self
    std::vector<int64_t> out_shape(self_shape.begin(), self_shape.end());
    Tensor output(out_shape, self.dtype(), self.device());
    size_t bytes = self.numel() * self.dtype_size();
    copy(output.data_ptr(), self.data_ptr(), bytes, CopyKind::DeviceToDevice);

    // Convert Int64 indices to Int32
    Tensor index_int32 = index;
    if (index.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(index.shape().begin(), index.shape().end());
        index_int32 = Tensor(idx_shape, DType::Int32, index.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = index.data_ptr();
        const void* buf_out = index_int32.data_ptr();
        size_t size_in = index.numel() * sizeof(int64_t);
        size_t size_out = index_int32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(index.numel());

        uint32_t cast_groups = div_wg(cast_pc.n_elements, devices_[device_id].workgroupSize);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, cast_groups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    uint32_t dim_size = static_cast<uint32_t>(self_shape[dim]);
    uint32_t idx_n = static_cast<uint32_t>(index.numel());
    uint32_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= static_cast<uint32_t>(self_shape[d]);
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= static_cast<uint32_t>(self_shape[d]);

    uint32_t total = outer * idx_n * inner;
    if (total == 0) return output;

    // Buffers: binding 0 = output, 1 = index
    const void* buf_out = output.data_ptr();
    const void* buf_idx = index_int32.data_ptr();

    size_t buf_out_size = output.numel() * output.dtype_size();
    size_t buf_idx_size = index_int32.numel() * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_out}, {1, buf_idx}
    };
    std::vector<size_t> sizes = {buf_out_size, buf_idx_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t outer;
        uint32_t dim_size;
        uint32_t idx_n;
        uint32_t inner;
        float value;
    } push_constants;

    push_constants.outer = outer;
    push_constants.dim_size = dim_size;
    push_constants.idx_n = idx_n;
    push_constants.inner = inner;
    push_constants.value = value;

    uint32_t workgroups = div_wg(total, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push_constants), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// NanToNum: native Vulkan dispatch using nan_to_num.comp shader
// ============================================================================
auto VulkanBackend::dispatchNanToNum(const Tensor& input,
                                     float nan_val, float posinf_val,
                                     float neginf_val) -> Tensor {
    if (input.numel() == 0) {
        auto s = input.shape();
        return Tensor(std::vector<int64_t>(s.begin(), s.end()), input.dtype(), input.device());
    }

    int32_t device_id = input.device().index;
    auto s = input.shape();
    std::vector<int64_t> output_shape(s.begin(), s.end());
    Tensor output(output_shape, input.dtype(), input.device());

    bool is_f64 = (input.dtype() == DType::Float64);
    auto* pipeline = getPipeline(is_f64 ? "nan_to_num_f64" : "nan_to_num", device_id);

    // fp32 and fp64 variants share (uint32 num_elements, N*scalar) push-
    // constants. 16-byte alignment for the fp64 struct comes from the
    // doubles; we reserve enough space and write the proper bit pattern.
    struct PushConstantsF32 {
        uint32_t num_elements;
        float nan_val;
        float posinf_val;
        float neginf_val;
    };
    struct PushConstantsF64 {
        uint32_t num_elements;
        uint32_t pad;
        double nan_val;
        double posinf_val;
        double neginf_val;
    };
    size_t pc_size = is_f64 ? sizeof(PushConstantsF64) : sizeof(PushConstantsF32);
    union { PushConstantsF32 f32; PushConstantsF64 f64; } pc = {};
    if (is_f64) {
        pc.f64.num_elements = static_cast<uint32_t>(input.numel());
        pc.f64.nan_val      = static_cast<double>(nan_val);
        pc.f64.posinf_val   = static_cast<double>(posinf_val);
        pc.f64.neginf_val   = static_cast<double>(neginf_val);
    } else {
        pc.f32.num_elements = static_cast<uint32_t>(input.numel());
        pc.f32.nan_val      = nan_val;
        pc.f32.posinf_val   = posinf_val;
        pc.f32.neginf_val   = neginf_val;
    }

    size_t buf_size = input.numel() * input.dtype_size();
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_size, buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, static_cast<uint32_t>(pc_size), &pc);

    uint32_t workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Bitwise binary ops: dispatch standalone int32 shaders (bitwise_and, bitwise_or,
// bitwise_xor, bitwise_left_shift, bitwise_right_shift)
// ============================================================================
auto VulkanBackend::dispatchBitwiseBinaryOp(const std::string& shader_name,
                                             const Tensor& a,
                                             const Tensor& b) -> Tensor {
    if (a.numel() == 0) {
        auto s = a.shape();
        return Tensor(std::vector<int64_t>(s.begin(), s.end()), a.dtype(), a.device());
    }

    int32_t device_id = a.device().index;
    auto s = a.shape();
    std::vector<int64_t> output_shape(s.begin(), s.end());
    Tensor output(output_shape, a.dtype(), a.device());

    auto* pipeline = getPipeline(shader_name, device_id);

    struct PushConstants { uint32_t num_elements; } pc;
    pc.num_elements = static_cast<uint32_t>(a.numel());

    size_t buf_size = a.numel() * a.dtype_size();
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, a.data_ptr()}, {1, b.data_ptr()}, {2, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_size, buf_size, buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);

    uint32_t workgroups = div_wg(a.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Heaviside — uses the dedicated heaviside.comp shader with 3 buffers
// (input, values, output) + num_elements push constant. Wired separately
// from dispatchBinaryOp because that routes through math.comp's opcode
// switch which doesn't have a heaviside case. Float32 only at the shader
// layer; Float16/BFloat16 are promoted by the caller.
// ============================================================================
auto VulkanBackend::dispatchHeaviside(const Tensor& input, const Tensor& values) -> Tensor {
    if (input.numel() == 0) {
        auto s = input.shape();
        return Tensor(std::vector<int64_t>(s.begin(), s.end()),
                      input.dtype(), input.device());
    }

    // The shader operates on Float32. Promote half types and cast the
    // result back on the caller's behalf.
    Tensor input_f32 = input.dtype() == DType::Float32
        ? input : input.to(DType::Float32);
    Tensor values_f32 = values.dtype() == DType::Float32
        ? values : values.to(DType::Float32);

    int32_t device_id = input_f32.device().index;
    auto s = input_f32.shape();
    std::vector<int64_t> output_shape(s.begin(), s.end());
    Tensor output(output_shape, DType::Float32, input_f32.device());

    auto* pipeline = getPipeline("heaviside", device_id);

    struct PushConstants { uint32_t num_elements; } pc;
    pc.num_elements = static_cast<uint32_t>(input_f32.numel());

    size_t buf_size = input_f32.numel() * sizeof(float);
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_f32.data_ptr()},
        {1, values_f32.data_ptr()},
        {2, output.data_ptr()},
    };
    std::vector<size_t> sizes = {buf_size, buf_size, buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);

    uint32_t workgroups = div_wg(input_f32.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // Narrow back to the caller's dtype if we promoted.
    return input.dtype() == DType::Float32 ? output : output.to(input.dtype());
}

// ============================================================================
// CumMax — cumulative maximum along a dimension (native Vulkan compute shader)
// ============================================================================

auto VulkanBackend::dispatchCumMax(const Tensor& input, int64_t dim) -> std::pair<Tensor, Tensor> {
    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (dim < 0) dim += ndim;

    // Same int-dtype fall-through as cumsum/cumprod: only Float32/Float64
    // shaders exist. Promote integer inputs so bytes aren't reinterpreted.
    DType src_dtype = input.dtype();
    if (src_dtype == DType::Int32 || src_dtype == DType::Int64 ||
        src_dtype == DType::Int16 || src_dtype == DType::Int8 ||
        src_dtype == DType::UInt8 || src_dtype == DType::Bool) {
        auto f32 = input.to(DType::Float32);
        auto [v_f32, idx] = dispatchCumMax(f32, dim);
        return {v_f32.to(src_dtype), idx};
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    std::string shader = is_f64 ? "cummax_f64" : "cummax";
    auto* pipeline = getPipeline(shader, device_id);

    Tensor values(std::vector<int64_t>(in_shape.begin(), in_shape.end()), input.dtype(), input.device());
    Tensor indices(std::vector<int64_t>(in_shape.begin(), in_shape.end()), DType::Int32, input.device());

    uint32_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) outer_size *= static_cast<uint32_t>(in_shape[i]);
    uint32_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) inner_size *= static_cast<uint32_t>(in_shape[i]);
    uint32_t reduce_size = static_cast<uint32_t>(in_shape[dim]);
    uint32_t total_lines = outer_size * inner_size;

    struct { uint32_t total_lines; uint32_t reduce_size; uint32_t inner_size; } pc;
    pc.total_lines = total_lines;
    pc.reduce_size = reduce_size;
    pc.inner_size = inner_size;

    size_t elem = input.dtype_size();
    size_t buf_size = static_cast<size_t>(input.numel()) * elem;
    size_t idx_size = static_cast<size_t>(input.numel()) * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, values.data_ptr()}, {2, indices.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_size, buf_size, idx_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_lines, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    // Convert indices to Int64 to match API contract
    Tensor indices_i64 = indices.to(DType::Int64);
    return {values, indices_i64};
}

// ============================================================================
// CumMin — cumulative minimum along a dimension (native Vulkan compute shader)
// ============================================================================

auto VulkanBackend::dispatchCumMin(const Tensor& input, int64_t dim) -> std::pair<Tensor, Tensor> {
    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (dim < 0) dim += ndim;

    // Same int-dtype fall-through — only Float32/Float64 shaders exist.
    DType src_dtype = input.dtype();
    if (src_dtype == DType::Int32 || src_dtype == DType::Int64 ||
        src_dtype == DType::Int16 || src_dtype == DType::Int8 ||
        src_dtype == DType::UInt8 || src_dtype == DType::Bool) {
        auto f32 = input.to(DType::Float32);
        auto [v_f32, idx] = dispatchCumMin(f32, dim);
        return {v_f32.to(src_dtype), idx};
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    std::string shader = is_f64 ? "cummin_f64" : "cummin";
    auto* pipeline = getPipeline(shader, device_id);

    Tensor values(std::vector<int64_t>(in_shape.begin(), in_shape.end()), input.dtype(), input.device());
    Tensor indices(std::vector<int64_t>(in_shape.begin(), in_shape.end()), DType::Int32, input.device());

    uint32_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) outer_size *= static_cast<uint32_t>(in_shape[i]);
    uint32_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) inner_size *= static_cast<uint32_t>(in_shape[i]);
    uint32_t reduce_size = static_cast<uint32_t>(in_shape[dim]);
    uint32_t total_lines = outer_size * inner_size;

    struct { uint32_t total_lines; uint32_t reduce_size; uint32_t inner_size; } pc;
    pc.total_lines = total_lines;
    pc.reduce_size = reduce_size;
    pc.inner_size = inner_size;

    size_t elem = input.dtype_size();
    size_t buf_size = static_cast<size_t>(input.numel()) * elem;
    size_t idx_size = static_cast<size_t>(input.numel()) * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, values.data_ptr()}, {2, indices.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_size, buf_size, idx_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_lines, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    Tensor indices_i64 = indices.to(DType::Int64);
    return {values, indices_i64};
}

// ============================================================================
// Fmax — NaN-aware element-wise max (native Vulkan compute shader)
// ============================================================================

auto VulkanBackend::dispatchFmax(const Tensor& a, const Tensor& b) -> Tensor {
    return dispatchBinaryOp("fmax", a, b);
}

// ============================================================================
// Fmin — NaN-aware element-wise min (native Vulkan compute shader)
// ============================================================================

auto VulkanBackend::dispatchFmin(const Tensor& a, const Tensor& b) -> Tensor {
    return dispatchBinaryOp("fmin", a, b);
}

// ============================================================================
// Isin — set membership test (native Vulkan compute shader)
// ============================================================================

auto VulkanBackend::dispatchIsin(const Tensor& elements, const Tensor& test_elements) -> Tensor {
    // For non-Float32 dtypes, cast to Float32 for the shader
    if (elements.dtype() != DType::Float32) {
        auto elem_f32 = elements.to(DType::Float32);
        auto test_f32 = test_elements.to(DType::Float32);
        return dispatchIsin(elem_f32, test_f32);
    }

    int32_t device_id = elements.device().index;
    auto* pipeline = getPipeline("isin", device_id);

    auto elem_shape = elements.shape();
    Tensor output(std::vector<int64_t>(elem_shape.begin(), elem_shape.end()), DType::Bool, elements.device());

    struct { uint32_t num_elements; uint32_t num_test_elements; } pc;
    pc.num_elements = static_cast<uint32_t>(elements.numel());
    pc.num_test_elements = static_cast<uint32_t>(test_elements.numel());

    size_t elem_buf = static_cast<size_t>(elements.numel()) * elements.dtype_size();
    size_t test_buf = static_cast<size_t>(test_elements.numel()) * test_elements.dtype_size();
    size_t out_buf = static_cast<size_t>(output.numel()) * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, elements.data_ptr()}, {1, test_elements.data_ptr()}, {2, output.data_ptr()}
    };
    std::vector<size_t> sizes = {elem_buf, test_buf, out_buf};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(elements.numel(), devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ============================================================================
// Kthvalue — k-th smallest (native Vulkan compute shader)
// ============================================================================

auto VulkanBackend::dispatchKthvalue(const Tensor& input, int64_t k, int64_t dim, bool keepdim) -> std::pair<Tensor, Tensor> {
    // Cast non-Float32 to Float32 for the shader
    if (input.dtype() != DType::Float32) {
        auto input_f32 = input.to(DType::Float32);
        auto [vals, idxs] = dispatchKthvalue(input_f32, k, dim, keepdim);
        return {vals.to(input.dtype()), idxs};
    }

    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (dim < 0) dim += ndim;
    int64_t dim_size = in_shape[dim];

    uint32_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= static_cast<uint32_t>(in_shape[i]);
    uint32_t inner_sz = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_sz *= static_cast<uint32_t>(in_shape[i]);
    uint32_t total_slices = outer_size * inner_sz;

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(in_shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("kthvalue", device_id);

    Tensor values(out_shape, input.dtype(), input.device());
    Tensor indices(out_shape, DType::Int32, input.device());

    struct { uint32_t total_slices; uint32_t dim_size; uint32_t inner_size; uint32_t k; } pc;
    pc.total_slices = total_slices;
    pc.dim_size = static_cast<uint32_t>(dim_size);
    pc.inner_size = inner_sz;
    pc.k = static_cast<uint32_t>(k);

    size_t in_buf = static_cast<size_t>(input.numel()) * input.dtype_size();
    size_t out_buf = static_cast<size_t>(total_slices) * sizeof(float);
    size_t idx_buf = static_cast<size_t>(total_slices) * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, values.data_ptr()}, {2, indices.data_ptr()}
    };
    std::vector<size_t> sizes = {in_buf, out_buf, idx_buf};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_slices, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return {values, indices.to(DType::Int64)};
}

// ============================================================================
// Quantile — interpolated quantile (native Vulkan compute shader)
// ============================================================================

auto VulkanBackend::dispatchQuantile(const Tensor& input, double q, int64_t dim, bool keepdim) -> Tensor {
    if (input.dtype() != DType::Float32) {
        auto input_f32 = input.to(DType::Float32);
        return dispatchQuantile(input_f32, q, dim, keepdim).to(input.dtype());
    }

    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (dim < 0) dim += ndim;
    int64_t dim_size = in_shape[dim];

    uint32_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= static_cast<uint32_t>(in_shape[i]);
    uint32_t inner_sz = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_sz *= static_cast<uint32_t>(in_shape[i]);
    uint32_t total_slices = outer_size * inner_sz;

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(in_shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("quantile", device_id);

    Tensor output(out_shape, input.dtype(), input.device());
    // Scratch buffer for per-slice sorting
    Tensor scratch({static_cast<int64_t>(total_slices) * dim_size}, DType::Float32, input.device());

    struct { uint32_t total_slices; uint32_t dim_size; uint32_t inner_size; float q; } pc;
    pc.total_slices = total_slices;
    pc.dim_size = static_cast<uint32_t>(dim_size);
    pc.inner_size = inner_sz;
    pc.q = static_cast<float>(q);

    size_t in_buf = static_cast<size_t>(input.numel()) * sizeof(float);
    size_t out_buf = static_cast<size_t>(total_slices) * sizeof(float);
    size_t scratch_buf = static_cast<size_t>(total_slices) * dim_size * sizeof(float);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}, {2, scratch.data_ptr()}
    };
    std::vector<size_t> sizes = {in_buf, out_buf, scratch_buf};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_slices, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ============================================================================
// Nanquantile — NaN-ignoring quantile (native Vulkan compute shader)
// ============================================================================

auto VulkanBackend::dispatchNanquantile(const Tensor& input, double q, int64_t dim, bool keepdim) -> Tensor {
    if (input.dtype() != DType::Float32) {
        auto input_f32 = input.to(DType::Float32);
        return dispatchNanquantile(input_f32, q, dim, keepdim).to(input.dtype());
    }

    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (dim < 0) dim += ndim;
    int64_t dim_size = in_shape[dim];

    uint32_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= static_cast<uint32_t>(in_shape[i]);
    uint32_t inner_sz = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_sz *= static_cast<uint32_t>(in_shape[i]);
    uint32_t total_slices = outer_size * inner_sz;

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(in_shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("nanquantile", device_id);

    Tensor output(out_shape, input.dtype(), input.device());
    Tensor scratch({static_cast<int64_t>(total_slices) * dim_size}, DType::Float32, input.device());

    struct { uint32_t total_slices; uint32_t dim_size; uint32_t inner_size; float q; } pc;
    pc.total_slices = total_slices;
    pc.dim_size = static_cast<uint32_t>(dim_size);
    pc.inner_size = inner_sz;
    pc.q = static_cast<float>(q);

    size_t in_buf = static_cast<size_t>(input.numel()) * sizeof(float);
    size_t out_buf = static_cast<size_t>(total_slices) * sizeof(float);
    size_t scratch_buf = static_cast<size_t>(total_slices) * dim_size * sizeof(float);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}, {2, scratch.data_ptr()}
    };
    std::vector<size_t> sizes = {in_buf, out_buf, scratch_buf};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_slices, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ============================================================================
// Nanmedian — NaN-ignoring median (delegates to nanquantile with q=0.5)
// ============================================================================

auto VulkanBackend::dispatchNanmedian(const Tensor& input, int64_t dim) -> Tensor {
    return dispatchNanquantile(input, 0.5, dim, false);
}

// ============================================================================
// Histc — fixed-bin histogram (native Vulkan compute shader with atomics)
// ============================================================================

auto VulkanBackend::dispatchHistc(const Tensor& input, int64_t bins, double min_val, double max_val) -> Tensor {
    Tensor input_f32 = (input.dtype() != DType::Float32) ? input.to(DType::Float32) : input;
    int32_t device_id = input.device().index;

    // Output is uint32 histogram, zero-initialized
    Tensor output_u32 = dispatchFull({bins}, 0.0f, DType::Int32);

    const bool auto_range = (min_val >= max_val);
    size_t in_buf = static_cast<size_t>(input_f32.numel()) * sizeof(float);
    size_t out_buf = static_cast<size_t>(bins) * sizeof(int32_t);

    if (auto_range) {
        // Phase 8.4: auto-range path runs entirely on device. The pack shader
        // writes (min, max, bin_width) into range_buf; histc_dyn reads it.
        Tensor flat = input_f32.contiguous().reshape({input_f32.numel()});
        Tensor mn_gpu = dispatchReduction("min", flat, 0, false).contiguous();
        Tensor mx_gpu = dispatchReduction("max", flat, 0, false).contiguous();
        Tensor range_buf({3}, DType::Float32, input.device());

        {
            auto* pack_pipeline = getPipeline("histogram_pack_range", device_id);
            struct { uint32_t num_bins; } pack_pc{static_cast<uint32_t>(bins)};
            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, mn_gpu.data_ptr()},
                {1, mx_gpu.data_ptr()},
                {2, range_buf.data_ptr()},
            };
            std::vector<size_t> sizes = {sizeof(float), sizeof(float), 3 * sizeof(float)};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pack_pipeline, bindings, sizes);
            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pack_pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pack_pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pack_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pack_pc), &pack_pc);
            vkCmdDispatch(cmd, 1, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        auto* hist_pipeline = getPipeline("histc_dyn", device_id);
        struct { uint32_t num_elements; uint32_t num_bins; } pc{
            static_cast<uint32_t>(input_f32.numel()), static_cast<uint32_t>(bins)};
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input_f32.data_ptr()},
            {1, output_u32.data_ptr()},
            {2, range_buf.data_ptr()},
        };
        std::vector<size_t> sizes = {in_buf, out_buf, 3 * sizeof(float)};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, hist_pipeline, bindings, sizes);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hist_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               hist_pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, hist_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(input_f32.numel(), devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    } else {
        auto* pipeline = getPipeline("histc", device_id);
        struct { uint32_t num_elements; uint32_t num_bins; float min_val; float max_val; } pc;
        pc.num_elements = static_cast<uint32_t>(input_f32.numel());
        pc.num_bins = static_cast<uint32_t>(bins);
        pc.min_val = static_cast<float>(min_val);
        pc.max_val = static_cast<float>(max_val);

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input_f32.data_ptr()}, {1, output_u32.data_ptr()}
        };
        std::vector<size_t> sizes = {in_buf, out_buf};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(input_f32.numel(), devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    return output_u32.to(DType::Float32);
}

// ============================================================================
// MaskedScatter with precomputed prefix sum — native Vulkan compute shader
// ============================================================================

auto VulkanBackend::dispatchMaskedScatterWithPrefix(const Tensor& input, const Tensor& mask,
                                                     const Tensor& source, const Tensor& prefix_sum) -> Tensor {
    Tensor in_f32 = (input.dtype() != DType::Float32) ? input.to(DType::Float32) : input;
    Tensor src_f32 = (source.dtype() != DType::Float32) ? source.to(DType::Float32) : source;
    Tensor mask_u8 = mask.to(DType::Bool);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("masked_scatter", device_id);

    Tensor output = in_f32.clone();
    uint32_t n = static_cast<uint32_t>(input.numel());

    struct { uint32_t num_elements; } pc;
    pc.num_elements = n;

    size_t f32_size = static_cast<size_t>(n) * sizeof(float);
    size_t bool_size = static_cast<size_t>(n) * sizeof(uint8_t);
    size_t i32_size = static_cast<size_t>(n) * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, output.data_ptr()},       // input (will be overwritten at mask positions)
        {1, mask_u8.data_ptr()},      // mask
        {2, src_f32.data_ptr()},      // source values
        {3, prefix_sum.data_ptr()},   // prefix sum of mask
        {4, output.data_ptr()}        // output (same as input, in-place)
    };
    std::vector<size_t> sizes = {f32_size, bool_size, f32_size, i32_size, f32_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(n, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return (input.dtype() != DType::Float32) ? output.to(input.dtype()) : output;
}

// ============================================================================
// UniqueConsecutive — consecutive dedup
// Uses GPU boundary-marking shader + minimal host readback for variable-size output.
// TODO: fully native multi-dispatch version (mark -> prefix_sum -> scatter)
// ============================================================================

auto VulkanBackend::dispatchUniqueConsecutive(const Tensor& input, bool return_inverse) -> std::tuple<Tensor, Tensor, Tensor> {
    int64_t n = input.numel();
    auto dev = input.device();

    if (n == 0) {
        return {Tensor({0}, input.dtype(), dev), Tensor({0}, DType::Int64, dev),
                Tensor({0}, DType::Int64, dev)};
    }

    // Step 1: Mark boundaries on GPU using unique_consecutive_mark shader
    // marks[i] = 1 if input[i] != input[i-1], else 0; marks[0] = 1
    Tensor input_f32 = (input.dtype() != DType::Float32) ? input.to(DType::Float32) : input;

    int32_t device_id = dev.index;
    auto* mark_pipeline = getPipeline("unique_consecutive_mark", device_id);

    Tensor marks({n}, DType::Int32, dev);
    struct { uint32_t num_elements; } pc;
    pc.num_elements = static_cast<uint32_t>(n);

    size_t f32_buf = static_cast<size_t>(n) * sizeof(float);
    size_t i32_buf = static_cast<size_t>(n) * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_f32.data_ptr()}, {1, marks.data_ptr()}
    };
    std::vector<size_t> sizes = {f32_buf, i32_buf};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, mark_pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mark_pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           mark_pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, mark_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(n, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    // Step 2: Prefix sum of marks on GPU (gives inverse mapping)
    Tensor marks_f32 = marks.to(DType::Float32);
    Tensor inverse = dispatchCumSum(marks_f32, 0);
    Tensor inverse_i64 = inverse.to(DType::Int64);
    // Adjust: cumsum is 1-based (first element = 1), we need 0-based
    Tensor one_tensor = dispatchFull({n}, 1.0f, DType::Int64);
    inverse_i64 = inverse_i64 - one_tensor;

    // Step 3: Scalar readback of unique count (mandatory for variable-size output allocation)
    Tensor total_marks = dispatchReduction("sum", marks_f32, 0, false);
    Tensor total_cpu = total_marks.to(Device::cpu());
    int64_t nu = static_cast<int64_t>(total_cpu.data<float>()[0]);

    // Step 4: Gather unique values on GPU using index_select-like pattern
    // Find positions where marks == 1 using nonzero-like logic
    // For simplicity, we do the compact step by gathering from the inverse map
    // unique_output[k] = input[first position where inverse == k]
    // Since consecutive, the first occurrence of each unique value is at marks == 1 positions

    // Use masked_select to get unique values (marks acts as mask)
    Tensor marks_bool = marks.to(DType::Bool);
    Tensor unique_vals = dispatchMaskedSelect(input_f32, marks_bool);
    if (input.dtype() != DType::Float32) {
        unique_vals = unique_vals.to(input.dtype());
    }

    // Step 5: Compute counts on GPU: counts[k] = number of elements with inverse == k
    // counts = diff of positions where marks == 1, plus trailing count
    // Simpler: counts[k] = (position of (k+1)-th mark) - (position of k-th mark)
    // For the last group: n - position_of_last_mark
    // We can compute this from the prefix sum differences
    Tensor counts_result({nu}, DType::Int64, dev);
    if (nu > 0) {
        // counts[i] for i < nu-1: number of elements between consecutive marks
        // Extract mark positions using masked_select on arange
        Tensor arange_t = dispatchArange(0, static_cast<float>(n), 1, DType::Int64, dev);
        Tensor mark_positions = dispatchMaskedSelect(arange_t.to(DType::Float32), marks_bool).to(DType::Int64);

        if (nu == 1) {
            // Only one unique value: count = n
            counts_result = dispatchFull({1}, static_cast<float>(n), DType::Int64);
        } else {
            // counts[i] = mark_positions[i+1] - mark_positions[i] for i < nu-1
            // counts[nu-1] = n - mark_positions[nu-1]
            Tensor pos_shifted = mark_positions.slice(0, 1, nu);
            Tensor pos_base = mark_positions.slice(0, 0, nu - 1);
            Tensor counts_head = pos_shifted - pos_base;

            // Last count
            Tensor last_pos = mark_positions.slice(0, nu - 1, nu);
            Tensor n_tensor = dispatchFull({1}, static_cast<float>(n), DType::Int64);
            Tensor counts_tail = n_tensor - last_pos;

            // Concatenate via direct dispatch
            std::vector<Tensor> count_parts = {counts_head, counts_tail};
            NewOpAttributes cat_attrs;
            cat_attrs.set(AttrKey::Dim, int64_t{0});
            counts_result = ::tenzor::dispatch<OpId::Cat>(
                std::span<const Tensor>(count_parts), cat_attrs)[0];
        }
    }

    return {unique_vals, inverse_i64, counts_result};
}

// ============================================================================
// SegmentReduce — native Vulkan compute shader (one workgroup per segment)
// ============================================================================

auto VulkanBackend::dispatchSegmentReduce(const Tensor& data, const Tensor& offsets,
                                          const std::string& reduce, int64_t axis) -> Tensor {
    Tensor data_f32 = (data.dtype() != DType::Float32) ? data.to(DType::Float32) : data;
    Tensor offs_i32 = offsets.to(DType::Int32);  // shader uses int offsets

    int64_t ndim = data_f32.ndim();
    if (axis < 0) axis += ndim;

    const auto& shape = data_f32.shape();
    int64_t axis_size = shape[axis];
    int64_t num_segments = offs_i32.numel() - 1;

    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < ndim; ++i) inner_size *= shape[i];

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        out_shape.push_back(i == axis ? num_segments : shape[i]);
    }

    uint32_t mode = 0;
    if (reduce == "sum") mode = 0;
    else if (reduce == "mean") mode = 1;
    else if (reduce == "max") mode = 2;
    else if (reduce == "min") mode = 3;
    else if (reduce == "prod") mode = 4;

    int32_t device_id = data.device().index;
    auto* pipeline = getPipeline("segment_reduce", device_id);

    Tensor output(out_shape, DType::Float32, data.device());

    struct {
        uint32_t num_segments;
        uint32_t reduce_mode;
        uint32_t outer_size;
        uint32_t axis_size;
        uint32_t inner_size;
    } pc;
    pc.num_segments = static_cast<uint32_t>(num_segments);
    pc.reduce_mode = mode;
    pc.outer_size = static_cast<uint32_t>(outer_size);
    pc.axis_size = static_cast<uint32_t>(axis_size);
    pc.inner_size = static_cast<uint32_t>(inner_size);

    size_t data_buf_size = static_cast<size_t>(data_f32.numel()) * sizeof(float);
    size_t offsets_buf_size = static_cast<size_t>(offs_i32.numel()) * sizeof(int32_t);
    size_t output_buf_size = static_cast<size_t>(output.numel()) * sizeof(float);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, data_f32.data_ptr()}, {1, offs_i32.data_ptr()}, {2, output.data_ptr()}
    };
    std::vector<size_t> sizes = {data_buf_size, offsets_buf_size, output_buf_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    // One workgroup per (outer, segment, inner) triple
    uint32_t total_workgroups = static_cast<uint32_t>(outer_size * num_segments * inner_size);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, total_workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    if (data.dtype() != DType::Float32) {
        return output.to(data.dtype());
    }
    return output;
}

// ============================================================================
// Fractional Max Pool 2D Forward — native Vulkan compute shader
// ============================================================================

auto VulkanBackend::dispatchFractionalMaxPool2dForward(const Tensor& input, int64_t out_h, int64_t out_w,
                                                        const Tensor* random_samples) -> std::pair<Tensor, Tensor> {
    bool is_f64 = (input.dtype() == DType::Float64);
    DType compute_dtype = is_f64 ? DType::Float64 : DType::Float32;
    Tensor input_compute = (input.dtype() != compute_dtype) ? input.to(compute_dtype) : input;
    auto shape = input_compute.shape();
    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int64_t total = N * C * out_h * out_w;
    bool has_samples = (random_samples && random_samples->numel() > 0);
    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline(is_f64 ? "fractional_maxpool2d_f64" : "fractional_maxpool2d", device_id);

    Tensor output({N, C, out_h, out_w}, compute_dtype, input.device());
    Tensor indices({N, C, out_h, out_w}, DType::Int32, input.device());
    Tensor samples_buf = has_samples ? *random_samples : Tensor({1}, DType::Float32, input.device());

    struct { uint32_t N, C, H, W, out_h, out_w, total, has_samples; } pc;
    pc.N = N; pc.C = C; pc.H = H; pc.W = W;
    pc.out_h = out_h; pc.out_w = out_w;
    pc.total = total; pc.has_samples = has_samples ? 1 : 0;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_compute.data_ptr()}, {1, output.data_ptr()},
        {2, indices.data_ptr()}, {3, samples_buf.data_ptr()}
    };
    std::vector<size_t> sizes = {
        size_t(input_compute.numel() * elem_size), size_t(total * elem_size),
        size_t(total * sizeof(int32_t)), size_t(samples_buf.numel() * sizeof(float))
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

    Tensor out_final = (input.dtype() != compute_dtype) ? output.to(input.dtype()) : output;
    return {out_final, indices.to(DType::Int64)};
}

auto VulkanBackend::dispatchFractionalMaxPool2dBackward(const Tensor& grad_output, const Tensor& indices,
                                                         const std::vector<int64_t>& input_shape) -> Tensor {
    bool is_f64 = (grad_output.dtype() == DType::Float64);
    DType compute_dtype = is_f64 ? DType::Float64 : DType::Float32;
    Tensor go = (grad_output.dtype() != compute_dtype) ? grad_output.to(compute_dtype) : grad_output;
    Tensor idx = indices.to(DType::Int32);
    auto grad_shape = go.shape();
    int64_t N = input_shape[0], C = input_shape[1], H = input_shape[2], W = input_shape[3];
    int64_t out_h = grad_shape[2], out_w = grad_shape[3];
    int64_t total = N * C * out_h * out_w;
    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);

    int32_t device_id = go.device().index;
    auto* pipeline = getPipeline(is_f64 ? "fractional_maxpool2d_backward_f64" : "fractional_maxpool2d_backward", device_id);
    Tensor grad_input = is_f64
        ? dispatchFull(input_shape, 0.0, DType::Float64)
        : dispatchFull(input_shape, 0.0f, DType::Float32);

    struct { uint32_t N, C, H_W, out_spatial, total; } pc;
    pc.N = N; pc.C = C; pc.H_W = H * W; pc.out_spatial = out_h * out_w; pc.total = total;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, go.data_ptr()}, {1, idx.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        size_t(total * elem_size), size_t(total * sizeof(int32_t)),
        size_t(N * C * H * W * elem_size)
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

    return (grad_output.dtype() != compute_dtype) ? grad_input.to(grad_output.dtype()) : grad_input;
}

auto VulkanBackend::dispatchFractionalMaxPool3dForward(const Tensor& input, int64_t out_d, int64_t out_h, int64_t out_w,
                                                        const Tensor* random_samples) -> std::pair<Tensor, Tensor> {
    bool is_f64 = (input.dtype() == DType::Float64);
    DType compute_dtype = is_f64 ? DType::Float64 : DType::Float32;
    Tensor input_compute = (input.dtype() != compute_dtype) ? input.to(compute_dtype) : input;
    auto shape = input_compute.shape();
    int64_t N = shape[0], C = shape[1], D = shape[2], H = shape[3], W = shape[4];
    int64_t total = N * C * out_d * out_h * out_w;
    bool has_samples = (random_samples && random_samples->numel() > 0);
    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline(is_f64 ? "fractional_maxpool3d_f64" : "fractional_maxpool3d", device_id);

    Tensor output({N, C, out_d, out_h, out_w}, compute_dtype, input.device());
    Tensor indices({N, C, out_d, out_h, out_w}, DType::Int32, input.device());
    Tensor samples_buf = has_samples ? *random_samples : Tensor({1}, DType::Float32, input.device());

    struct { uint32_t N, C, D, H, W, out_d, out_h, out_w, total, has_samples; } pc;
    pc.N = N; pc.C = C; pc.D = D; pc.H = H; pc.W = W;
    pc.out_d = out_d; pc.out_h = out_h; pc.out_w = out_w;
    pc.total = total; pc.has_samples = has_samples ? 1 : 0;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_compute.data_ptr()}, {1, output.data_ptr()},
        {2, indices.data_ptr()}, {3, samples_buf.data_ptr()}
    };
    std::vector<size_t> sizes = {
        size_t(input_compute.numel() * elem_size), size_t(total * elem_size),
        size_t(total * sizeof(int32_t)), size_t(samples_buf.numel() * sizeof(float))
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

    Tensor out_final = (input.dtype() != compute_dtype) ? output.to(input.dtype()) : output;
    return {out_final, indices.to(DType::Int64)};
}

auto VulkanBackend::dispatchFractionalMaxPool3dBackward(const Tensor& grad_output, const Tensor& indices,
                                                         const std::vector<int64_t>& input_shape) -> Tensor {
    bool is_f64 = (grad_output.dtype() == DType::Float64);
    DType compute_dtype = is_f64 ? DType::Float64 : DType::Float32;
    Tensor go = (grad_output.dtype() != compute_dtype) ? grad_output.to(compute_dtype) : grad_output;
    Tensor idx = indices.to(DType::Int32);
    auto grad_shape = go.shape();
    int64_t N = input_shape[0], C = input_shape[1], D = input_shape[2], H = input_shape[3], W = input_shape[4];
    int64_t out_d = grad_shape[2], out_h = grad_shape[3], out_w = grad_shape[4];
    int64_t total = N * C * out_d * out_h * out_w;
    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);

    int32_t device_id = go.device().index;
    auto* pipeline = getPipeline(is_f64 ? "fractional_maxpool3d_backward_f64" : "fractional_maxpool3d_backward", device_id);
    Tensor grad_input = is_f64
        ? dispatchFull(input_shape, 0.0, DType::Float64)
        : dispatchFull(input_shape, 0.0f, DType::Float32);

    struct { uint32_t total, D_H_W, out_spatial; } pc;
    pc.total = total; pc.D_H_W = D * H * W; pc.out_spatial = out_d * out_h * out_w;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, go.data_ptr()}, {1, idx.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        size_t(total * elem_size), size_t(total * sizeof(int32_t)),
        size_t(N * C * D * H * W * elem_size)
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

    return (grad_output.dtype() != compute_dtype) ? grad_input.to(grad_output.dtype()) : grad_input;
}

auto VulkanBackend::dispatchMaxUnpool2dForward(const Tensor& input, const Tensor& indices,
                                                int64_t out_h, int64_t out_w) -> Tensor {
    // audit-2026-05-03 — Float64 input previously detoured through Float32,
    // dropping ~30 mantissa bits and breaking gradcheck. Native Float64 path
    // uses max_unpool2d_f64.comp (which requires
    // GL_EXT_shader_explicit_arithmetic_types_float64).
    bool is_f64 = (input.dtype() == DType::Float64);
    DType compute_dtype = is_f64 ? DType::Float64 : DType::Float32;
    Tensor in_compute = (input.dtype() != compute_dtype) ? input.to(compute_dtype) : input;
    Tensor idx = indices.to(DType::Int32);
    auto shape = in_compute.shape();
    int64_t N = shape[0], C = shape[1];
    int64_t in_spatial = shape[2] * shape[3];
    int64_t out_spatial = out_h * out_w;
    int64_t total_input = in_compute.numel();
    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline(is_f64 ? "max_unpool2d_f64" : "max_unpool2d", device_id);
    Tensor output = is_f64
        ? dispatchFull({N, C, out_h, out_w}, 0.0, DType::Float64)
        : dispatchFull({N, C, out_h, out_w}, 0.0f, DType::Float32);

    struct { uint32_t total_input, in_spatial, out_spatial; } pc;
    pc.total_input = total_input; pc.in_spatial = in_spatial; pc.out_spatial = out_spatial;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, in_compute.data_ptr()}, {1, idx.data_ptr()}, {2, output.data_ptr()}
    };
    std::vector<size_t> sizes = {
        size_t(total_input * elem_size), size_t(total_input * sizeof(int32_t)),
        size_t(N * C * out_spatial * elem_size)
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_input, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return (input.dtype() != compute_dtype) ? output.to(input.dtype()) : output;
}

auto VulkanBackend::dispatchMaxUnpool2dBackward(const Tensor& grad_output, const Tensor& indices,
                                                 const std::vector<int64_t>& input_shape) -> Tensor {
    bool is_f64 = (grad_output.dtype() == DType::Float64);
    DType compute_dtype = is_f64 ? DType::Float64 : DType::Float32;
    Tensor go = (grad_output.dtype() != compute_dtype) ? grad_output.to(compute_dtype) : grad_output;
    Tensor idx = indices.to(DType::Int32);
    int64_t in_spatial = input_shape[2] * input_shape[3];
    auto go_shape = go.shape();
    int64_t out_spatial = go_shape[2] * go_shape[3];
    int64_t total_input = 1;
    for (auto d : input_shape) total_input *= d;
    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);

    int32_t device_id = go.device().index;
    auto* pipeline = getPipeline(is_f64 ? "max_unpool2d_backward_f64" : "max_unpool2d_backward", device_id);
    Tensor grad_input(input_shape, compute_dtype, go.device());

    struct { uint32_t total_input, in_spatial, out_spatial; } pc;
    pc.total_input = total_input; pc.in_spatial = in_spatial; pc.out_spatial = out_spatial;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, go.data_ptr()}, {1, idx.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        size_t(go.numel() * elem_size), size_t(total_input * sizeof(int32_t)),
        size_t(total_input * elem_size)
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_input, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return (grad_output.dtype() != compute_dtype) ? grad_input.to(grad_output.dtype()) : grad_input;
}

auto VulkanBackend::dispatchMaxUnpool3dForward(const Tensor& input, const Tensor& indices,
                                                int64_t out_d, int64_t out_h, int64_t out_w) -> Tensor {
    bool is_f64 = (input.dtype() == DType::Float64);
    DType compute_dtype = is_f64 ? DType::Float64 : DType::Float32;
    Tensor in_compute = (input.dtype() != compute_dtype) ? input.to(compute_dtype) : input;
    Tensor idx = indices.to(DType::Int32);
    auto shape = in_compute.shape();
    int64_t N = shape[0], C = shape[1];
    int64_t in_spatial = shape[2] * shape[3] * shape[4];
    int64_t out_spatial = out_d * out_h * out_w;
    int64_t total_input = in_compute.numel();
    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline(is_f64 ? "max_unpool3d_f64" : "max_unpool3d", device_id);
    Tensor output = is_f64
        ? dispatchFull({N, C, out_d, out_h, out_w}, 0.0, DType::Float64)
        : dispatchFull({N, C, out_d, out_h, out_w}, 0.0f, DType::Float32);

    struct { uint32_t total_input, in_spatial, out_spatial; } pc;
    pc.total_input = total_input; pc.in_spatial = in_spatial; pc.out_spatial = out_spatial;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, in_compute.data_ptr()}, {1, idx.data_ptr()}, {2, output.data_ptr()}
    };
    std::vector<size_t> sizes = {
        size_t(total_input * elem_size), size_t(total_input * sizeof(int32_t)),
        size_t(N * C * out_spatial * elem_size)
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_input, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return (input.dtype() != compute_dtype) ? output.to(input.dtype()) : output;
}

auto VulkanBackend::dispatchMaxUnpool3dBackward(const Tensor& grad_output, const Tensor& indices,
                                                 const std::vector<int64_t>& input_shape) -> Tensor {
    bool is_f64 = (grad_output.dtype() == DType::Float64);
    DType compute_dtype = is_f64 ? DType::Float64 : DType::Float32;
    Tensor go = (grad_output.dtype() != compute_dtype) ? grad_output.to(compute_dtype) : grad_output;
    Tensor idx = indices.to(DType::Int32);
    int64_t in_spatial = input_shape[2] * input_shape[3] * input_shape[4];
    auto go_shape = go.shape();
    int64_t out_spatial = go_shape[2] * go_shape[3] * go_shape[4];
    int64_t total_input = 1;
    for (auto d : input_shape) total_input *= d;
    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);

    int32_t device_id = go.device().index;
    auto* pipeline = getPipeline(is_f64 ? "max_unpool3d_backward_f64" : "max_unpool3d_backward", device_id);
    Tensor grad_input(input_shape, compute_dtype, go.device());

    struct { uint32_t total_input, in_spatial, out_spatial; } pc;
    pc.total_input = total_input; pc.in_spatial = in_spatial; pc.out_spatial = out_spatial;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, go.data_ptr()}, {1, idx.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        size_t(go.numel() * elem_size), size_t(total_input * sizeof(int32_t)),
        size_t(total_input * elem_size)
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_input, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return (grad_output.dtype() != compute_dtype) ? grad_input.to(grad_output.dtype()) : grad_input;
}

// ============================================================================
// Frexp — decompose float into mantissa and exponent
// ============================================================================

auto VulkanBackend::dispatchFrexp(const Tensor& input)
    -> std::pair<Tensor, Tensor> {
    // Empty tensor fast path
    if (input.numel() == 0) {
        std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
        return {Tensor(shape, input.dtype(), input.device()),
                Tensor(shape, DType::Int32, input.device())};
    }

    DType orig_dtype = input.dtype();
    Tensor work_input = input;
    if (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16) {
        work_input = input.to(DType::Float32);
    }

    // For Float64: promote to f32, frexp, then cast mantissa back
    if (work_input.dtype() == DType::Float64) {
        Tensor f32_input = work_input.to(DType::Float32);
        auto [m32, e32] = dispatchFrexp(f32_input);
        return {m32.to(DType::Float64), e32};
    }

    std::vector<int64_t> shape(work_input.shape().begin(), work_input.shape().end());
    Tensor mantissa(shape, DType::Float32, work_input.device());
    Tensor exponent(shape, DType::Int32, work_input.device());

    int32_t device_id = work_input.device().index;
    auto* pipeline = getPipeline("frexp", device_id);

    struct { uint32_t n; } pc;
    pc.n = static_cast<uint32_t>(work_input.numel());

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, work_input.data_ptr()},
        {1, mantissa.data_ptr()},
        {2, exponent.data_ptr()},
    };
    size_t buf_in   = work_input.numel() * work_input.dtype_size();
    size_t buf_mant = mantissa.numel() * mantissa.dtype_size();
    size_t buf_exp  = exponent.numel() * exponent.dtype_size();
    std::vector<size_t> sizes = {buf_in, buf_mant, buf_exp};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(pc.n, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    if (orig_dtype != DType::Float32) {
        mantissa = mantissa.to(orig_dtype);
    }
    return {mantissa, exponent};
}

// ============================================================================
// DiagEmbed — embed a vector as a diagonal of a matrix
// ============================================================================

auto VulkanBackend::dispatchDiagEmbed(const Tensor& input, int64_t offset,
                                       int64_t dim1, int64_t dim2)
    -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize dims to output dimensionality (output has one more dim)
    int64_t out_ndim = ndim + 1;
    if (dim1 < 0) dim1 += out_ndim;
    if (dim2 < 0) dim2 += out_ndim;

    int64_t vec_len = input_shape[ndim - 1];
    int64_t abs_offset = offset >= 0 ? offset : -offset;
    int64_t mat_size = vec_len + abs_offset;

    // Build output shape: batch dims + (mat_size, mat_size)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim - 1; ++i) {
        output_shape.push_back(input_shape[i]);
    }
    output_shape.push_back(mat_size);
    output_shape.push_back(mat_size);

    if (input.numel() == 0) {
        return Tensor(output_shape, input.dtype(), input.device());
    }

    // Compute batch size
    int64_t batch = 1;
    for (int64_t i = 0; i < ndim - 1; ++i) batch *= input_shape[i];

    // Zero the output
    Tensor zero_out = dispatchFull(output_shape, 0.0f, input.dtype());

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "diag_f64" : is_float16 ? "diag_f16" : is_bfloat16 ? "diag_bf16" : "diag";
    auto* pipeline = getPipeline(shader_name, device_id);

    struct {
        uint32_t n;
        uint32_t rows;
        uint32_t cols;
        int32_t diagonal;
        uint32_t op;
    } pushConstants;

    if (batch == 1) {
        pushConstants.n = static_cast<uint32_t>(mat_size * mat_size);
        pushConstants.rows = static_cast<uint32_t>(mat_size);
        pushConstants.cols = static_cast<uint32_t>(mat_size);
        pushConstants.diagonal = static_cast<int32_t>(offset);
        pushConstants.op = 1; // construct

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()},
            {1, zero_out.data_ptr()}
        };
        std::vector<size_t> sizes_vec = {
            static_cast<size_t>(input.numel() * input.dtype_size()),
            static_cast<size_t>(zero_out.numel() * zero_out.dtype_size())
        };

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes_vec);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(cmd, div_wg(mat_size * mat_size, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        return zero_out;
    }

    // Batched case: iterate per batch
    Tensor flat_input = input.reshape({batch, vec_len});
    Tensor result = zero_out.reshape({batch, mat_size, mat_size});

    for (int64_t b = 0; b < batch; ++b) {
        Tensor in_slice = flat_input.select(0, b);
        Tensor out_slice = result.select(0, b);

        pushConstants.n = static_cast<uint32_t>(out_slice.numel());
        pushConstants.rows = static_cast<uint32_t>(mat_size);
        pushConstants.cols = static_cast<uint32_t>(mat_size);
        pushConstants.diagonal = static_cast<int32_t>(offset);
        pushConstants.op = 1; // construct

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, in_slice.data_ptr()},
            {1, out_slice.data_ptr()}
        };
        std::vector<size_t> sizes_vec = {
            static_cast<size_t>(in_slice.numel() * in_slice.dtype_size()),
            static_cast<size_t>(out_slice.numel() * out_slice.dtype_size())
        };

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes_vec);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(cmd, div_wg(out_slice.numel(), devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    return result.reshape(output_shape);
}

// ============================================================================
// Diagflat — flatten input, then create diagonal matrix
// ============================================================================

auto VulkanBackend::dispatchDiagflat(const Tensor& input, int64_t offset)
    -> Tensor {
    int64_t numel = input.numel();
    Tensor flat = input.reshape({numel});
    return dispatchDiag(flat, offset);
}

// ============================================================================
// NanVar — NaN-ignoring variance
// ============================================================================

auto VulkanBackend::dispatchNanVar(const Tensor& input, int64_t correction)
    -> Tensor {
    if (input.numel() == 0) {
        return dispatchFull({1}, std::numeric_limits<float>::quiet_NaN(), input.dtype());
    }

    DType orig_dtype = input.dtype();
    Tensor work_input = input;
    if (orig_dtype == DType::BFloat16 || orig_dtype == DType::Float16) {
        work_input = input.to(DType::Float32);
    }

    int32_t device_id = work_input.device().index;

    std::string shader_name = (work_input.dtype() == DType::Float64)
        ? "nanvar_f64" : "nanvar";
    auto* pipeline = getPipeline(shader_name, device_id);

    Tensor output({1}, work_input.dtype(), work_input.device());

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, work_input.data_ptr()},
        {1, output.data_ptr()}
    };
    std::vector<size_t> sizes = {
        work_input.numel() * work_input.dtype_size(),
        output.numel() * output.dtype_size()
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct {
        uint32_t num_elements;
        uint32_t correction;
    } pushConstants;
    pushConstants.num_elements = static_cast<uint32_t>(work_input.numel());
    pushConstants.correction = static_cast<uint32_t>(correction);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(cmd, 1, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    if (orig_dtype != work_input.dtype()) {
        return output.to(orig_dtype);
    }
    return output;
}

// ============================================================================
// NanStd — NaN-ignoring standard deviation = sqrt(nanvar)
// ============================================================================

auto VulkanBackend::dispatchNanStd(const Tensor& input, int64_t correction)
    -> Tensor {
    Tensor var = dispatchNanVar(input, correction);
    return dispatchUnaryOp("sqrt", var);
}

// ============================================================================
// TrilIndices — native Vulkan compute shader (tril_indices.comp)
// ============================================================================

auto VulkanBackend::dispatchTrilIndices(int64_t row, int64_t col, int64_t offset) -> Tensor {
    // Compute total element count on host using O(1) closed-form formula
    int64_t first = std::max(static_cast<int64_t>(0), -offset);
    int64_t n = 0;
    if (first < row) {
        int64_t full_from = col - offset - 1;
        if (full_from <= first) {
            n = (row - first) * col;
        } else if (full_from >= row) {
            int64_t count = row - first;
            int64_t first_val = first + offset + 1;
            int64_t last_val  = row - 1 + offset + 1;
            n = count * (first_val + last_val) / 2;
        } else {
            int64_t partial_count = full_from - first;
            int64_t first_val = first + offset + 1;
            int64_t last_val  = full_from - 1 + offset + 1;
            int64_t partial_sum = partial_count * (first_val + last_val) / 2;
            int64_t full_sum = (row - full_from) * col;
            n = partial_sum + full_sum;
        }
    }

    Device device(Device::Type::Vulkan, 0);
    if (n == 0) {
        return Tensor({2, static_cast<int64_t>(0)}, DType::Int64, device);
    }

    int32_t device_id = 0;

    // Allocate output as Int32 (Vulkan shaders use 32-bit int), shape {2, n}
    // Two separate buffers: row_indices[n] and col_indices[n]
    Tensor row_out({n}, DType::Int32, device);
    Tensor col_out({n}, DType::Int32, device);

    auto* pipeline = getPipeline("tril_indices", device_id);

    struct { int32_t n; int32_t row; int32_t col; int32_t offset; } pc;
    pc.n = static_cast<int32_t>(n);
    pc.row = static_cast<int32_t>(row);
    pc.col = static_cast<int32_t>(col);
    pc.offset = static_cast<int32_t>(offset);

    size_t buf_size = static_cast<size_t>(n) * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, row_out.data_ptr()},
        {1, col_out.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_size, buf_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(n, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    // Concatenate into {2, n} Int64 output
    Tensor row_i64 = row_out.to(DType::Int64);
    Tensor col_i64 = col_out.to(DType::Int64);
    return dispatchCat({row_i64.reshape({1, n}), col_i64.reshape({1, n})}, 0);
}

// ============================================================================
// TriuIndices — native Vulkan compute shader (triu_indices.comp)
// ============================================================================

auto VulkanBackend::dispatchTriuIndices(int64_t row, int64_t col, int64_t offset) -> Tensor {
    // Compute total element count on host using O(1) closed-form formula
    int64_t n = 0;
    int64_t last_full = std::min(row - 1, -offset);
    if (last_full >= 0) {
        n += (last_full + 1) * col;
    }
    int64_t first_partial = std::max(static_cast<int64_t>(0), last_full + 1);
    int64_t last_nonempty = std::min(row - 1, col - offset - 1);
    if (first_partial <= last_nonempty) {
        int64_t count = last_nonempty - first_partial + 1;
        int64_t first_val = col - (first_partial + offset);
        int64_t last_val  = col - (last_nonempty + offset);
        n += count * (first_val + last_val) / 2;
    }

    Device device(Device::Type::Vulkan, 0);
    if (n == 0) {
        return Tensor({2, static_cast<int64_t>(0)}, DType::Int64, device);
    }

    int32_t device_id = 0;

    // Allocate output as Int32 (Vulkan shaders use 32-bit int)
    Tensor row_out({n}, DType::Int32, device);
    Tensor col_out({n}, DType::Int32, device);

    auto* pipeline = getPipeline("triu_indices", device_id);

    struct { int32_t n; int32_t row; int32_t col; int32_t offset; } pc;
    pc.n = static_cast<int32_t>(n);
    pc.row = static_cast<int32_t>(row);
    pc.col = static_cast<int32_t>(col);
    pc.offset = static_cast<int32_t>(offset);

    size_t buf_size = static_cast<size_t>(n) * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, row_out.data_ptr()},
        {1, col_out.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_size, buf_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(n, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    // Concatenate into {2, n} Int64 output
    Tensor row_i64 = row_out.to(DType::Int64);
    Tensor col_i64 = col_out.to(DType::Int64);
    return dispatchCat({row_i64.reshape({1, n}), col_i64.reshape({1, n})}, 0);
}

} // namespace tenzor
