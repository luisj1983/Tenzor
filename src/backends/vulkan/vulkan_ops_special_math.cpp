/**
 * @file vulkan_ops_special_math.cpp
 * @brief Native Vulkan dispatch for special-math functions.
 *
 * Replaces the previous CPU-roundtrip fallbacks. The shaders
 * (special_math_unary.comp, special_math_binary.comp, special_math_ternary.comp)
 * implement the same Cephes / Abramowitz polynomial approximations as the CPU
 * backend in float32. For non-Float32 input dtypes the host dispatcher promotes
 * via the existing dispatchCast pipeline (which is itself an on-device Vulkan
 * compute shader — never a CPU fallback) and casts back at the end.
 */

#include "vulkan_ops_common.hpp"

namespace tenzor {

namespace {

// Push-constant layout matching special_math_unary.comp
struct UnaryPushConstants {
    uint32_t n;
    uint32_t op;
    int32_t  param_int;  // Polygamma order (0 if unused)
};

// Push-constant layout matching special_math_binary.comp
struct BinaryPushConstants {
    uint32_t n;
    uint32_t op;
};

// Promote input to Float32 if necessary; returns the promoted tensor and a
// flag indicating whether the caller should cast the result back.
inline Tensor maybe_promote(const Tensor& input, DType original_dtype, VulkanBackend* backend) {
    if (original_dtype == DType::Float32) return input;
    return backend->dispatchCast(input, DType::Float32);
}

inline Tensor maybe_demote(const Tensor& f32_result, DType target_dtype, VulkanBackend* backend) {
    if (target_dtype == DType::Float32) return f32_result;
    return backend->dispatchCast(f32_result, target_dtype);
}

}  // namespace

auto VulkanBackend::dispatchSpecialMathUnary(const Tensor& input, uint32_t opcode, int32_t param_int) -> Tensor {
    DType orig_dtype = input.dtype();
    if (orig_dtype != DType::Float32 && orig_dtype != DType::Float64
        && orig_dtype != DType::Float16 && orig_dtype != DType::BFloat16) {
        throw std::runtime_error("dispatchSpecialMathUnary: dtype must be a floating-point type");
    }

    // Empty-tensor fast path
    if (input.numel() == 0) {
        std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
        return Tensor(shape, orig_dtype, input.device());
    }

    // Select native dtype shader to avoid host-side promote/demote round-trips.
    // Float16 and BFloat16 shaders unpack pairs, compute in f32, and repack.
    // Float64 shader uses double-precision polynomial approximations.
    std::string shader_name = "special_math_unary";
    if (orig_dtype == DType::Float16) {
        shader_name = "special_math_unary_f16";
    } else if (orig_dtype == DType::BFloat16) {
        shader_name = "special_math_unary_bf16";
    } else if (orig_dtype == DType::Float64) {
        shader_name = "special_math_unary_f64";
    }

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor output(shape, orig_dtype, input.device());

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline(shader_name, device_id);

    UnaryPushConstants pc{};
    pc.n = static_cast<uint32_t>(input.numel());
    pc.op = opcode;
    pc.param_int = param_int;

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();
    size_t buffer_size_in  = input.numel()  * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out},
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UnaryPushConstants), &pc);

    // For packed f16/bf16 shaders, each invocation processes 2 elements
    uint32_t num_work_items = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16)
        ? (static_cast<uint32_t>(input.numel()) + 1) / 2
        : static_cast<uint32_t>(input.numel());
    uint32_t workgroups = div_wg(num_work_items, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);
    synchronize(device_id);

    return output;
}

auto VulkanBackend::dispatchSpecialMathBinary(const Tensor& a, const Tensor& b, uint32_t opcode) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("dispatchSpecialMathBinary: input dtypes must match");
    }
    DType orig_dtype = a.dtype();
    if (orig_dtype != DType::Float32 && orig_dtype != DType::Float64
        && orig_dtype != DType::Float16 && orig_dtype != DType::BFloat16) {
        throw std::runtime_error("dispatchSpecialMathBinary: dtype must be a floating-point type");
    }

    if (a.numel() == 0) {
        std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
        return Tensor(shape, orig_dtype, a.device());
    }

    Tensor f32_a = maybe_promote(a, orig_dtype, this);
    Tensor f32_b = maybe_promote(b, orig_dtype, this);

    std::vector<int64_t> shape(f32_a.shape().begin(), f32_a.shape().end());
    Tensor f32_output(shape, DType::Float32, f32_a.device());

    int32_t device_id = f32_a.device().index;
    auto* pipeline = getPipeline("special_math_binary", device_id);

    BinaryPushConstants pc{};
    pc.n = static_cast<uint32_t>(f32_a.numel());
    pc.op = opcode;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, f32_a.data_ptr()},
        {1, f32_b.data_ptr()},
        {2, f32_output.data_ptr()},
    };
    size_t buf_size = f32_a.numel() * sizeof(float);
    std::vector<size_t> sizes = {buf_size, buf_size, buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BinaryPushConstants), &pc);

    uint32_t workgroups = div_wg(static_cast<uint32_t>(f32_a.numel()),
                                  devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);
    synchronize(device_id);

    return maybe_demote(f32_output, orig_dtype, this);
}

auto VulkanBackend::dispatchSpecialMathTernary(const Tensor& a, const Tensor& b, const Tensor& x) -> Tensor {
    if (a.dtype() != b.dtype() || a.dtype() != x.dtype()) {
        throw std::runtime_error("dispatchSpecialMathTernary: input dtypes must match");
    }
    DType orig_dtype = a.dtype();
    if (orig_dtype != DType::Float32 && orig_dtype != DType::Float64
        && orig_dtype != DType::Float16 && orig_dtype != DType::BFloat16) {
        throw std::runtime_error("dispatchSpecialMathTernary: dtype must be a floating-point type");
    }

    if (a.numel() == 0) {
        std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
        return Tensor(shape, orig_dtype, a.device());
    }

    Tensor f32_a = maybe_promote(a, orig_dtype, this);
    Tensor f32_b = maybe_promote(b, orig_dtype, this);
    Tensor f32_x = maybe_promote(x, orig_dtype, this);

    std::vector<int64_t> shape(f32_a.shape().begin(), f32_a.shape().end());
    Tensor f32_output(shape, DType::Float32, f32_a.device());

    int32_t device_id = f32_a.device().index;
    auto* pipeline = getPipeline("special_math_ternary", device_id);

    BinaryPushConstants pc{};
    pc.n = static_cast<uint32_t>(f32_a.numel());
    pc.op = 0;  // betainc is the only ternary op

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, f32_a.data_ptr()},
        {1, f32_b.data_ptr()},
        {2, f32_x.data_ptr()},
        {3, f32_output.data_ptr()},
    };
    size_t buf_size = f32_a.numel() * sizeof(float);
    std::vector<size_t> sizes = {buf_size, buf_size, buf_size, buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BinaryPushConstants), &pc);

    uint32_t workgroups = div_wg(static_cast<uint32_t>(f32_a.numel()),
                                  devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);
    synchronize(device_id);

    return maybe_demote(f32_output, orig_dtype, this);
}

}  // namespace tenzor
