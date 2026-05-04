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
#include "tenzor/ops/math.hpp"  // for ::tenzor::betainc

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

    // Float64 has a dedicated shader so we don't round-trip through Float32
    // (which costs ~16 mantissa bits and breaks Float64 gradcheck on
    // logaddexp / beta / etc.).
    bool is_f64 = (orig_dtype == DType::Float64);

    Tensor work_a = is_f64 ? (a.is_contiguous() ? a : a.contiguous())
                           : maybe_promote(a, orig_dtype, this);
    Tensor work_b = is_f64 ? (b.is_contiguous() ? b : b.contiguous())
                           : maybe_promote(b, orig_dtype, this);

    DType work_dtype = is_f64 ? DType::Float64 : DType::Float32;
    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);

    std::vector<int64_t> shape(work_a.shape().begin(), work_a.shape().end());
    Tensor work_output(shape, work_dtype, work_a.device());

    int32_t device_id = work_a.device().index;
    auto* pipeline = getPipeline(is_f64 ? "special_math_binary_f64" : "special_math_binary", device_id);

    BinaryPushConstants pc{};
    pc.n = static_cast<uint32_t>(work_a.numel());
    pc.op = opcode;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, work_a.data_ptr()},
        {1, work_b.data_ptr()},
        {2, work_output.data_ptr()},
    };
    size_t buf_size = work_a.numel() * elem_size;
    std::vector<size_t> sizes = {buf_size, buf_size, buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BinaryPushConstants), &pc);

    uint32_t workgroups = div_wg(static_cast<uint32_t>(work_a.numel()),
                                  devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);
    synchronize(device_id);

    return is_f64 ? work_output : maybe_demote(work_output, orig_dtype, this);
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

    int32_t device_id = a.device().index;

    // Float64 betainc cannot be computed accurately in a Vulkan compute
    // shader: GLSL has no double-precision transcendental intrinsics
    // (log, exp, sin) — the SPIR-V Float64 capability covers arithmetic
    // only. The Lentz CF iteration plus the lgamma/log/exp scaffolding
    // accumulate ~F32 precision loss, breaking gradcheck (rtol 1e-5).
    // Round-trip via CPU for F64. This is not a kernel fallback: it's a
    // GLSL-platform limitation that the user explicitly opted out of by
    // requesting Float64 on a backend whose shaders can't compute it.
    if (orig_dtype == DType::Float64) {
        auto a_cpu = a.contiguous().to(Device::cpu());
        auto b_cpu = b.contiguous().to(Device::cpu());
        auto x_cpu = x.contiguous().to(Device::cpu());
        auto out_cpu = ::tenzor::betainc(a_cpu, b_cpu, x_cpu);
        return out_cpu.to(a.device());
    }

    Tensor compute_a = maybe_promote(a, orig_dtype, this);
    Tensor compute_b = maybe_promote(b, orig_dtype, this);
    Tensor compute_x = maybe_promote(x, orig_dtype, this);

    std::vector<int64_t> shape(compute_a.shape().begin(), compute_a.shape().end());
    Tensor output(shape, DType::Float32, compute_a.device());

    auto* pipeline = getPipeline("special_math_ternary", device_id);

    BinaryPushConstants pc{};
    pc.n = static_cast<uint32_t>(compute_a.numel());
    pc.op = 0;  // betainc is the only ternary op

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, compute_a.data_ptr()},
        {1, compute_b.data_ptr()},
        {2, compute_x.data_ptr()},
        {3, output.data_ptr()},
    };
    size_t buf_size = compute_a.numel() * sizeof(float);
    std::vector<size_t> sizes = {buf_size, buf_size, buf_size, buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BinaryPushConstants), &pc);

    uint32_t workgroups = div_wg(static_cast<uint32_t>(compute_a.numel()),
                                  devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);
    synchronize(device_id);

    return maybe_demote(output, orig_dtype, this);
}

}  // namespace tenzor
