/**
 * @file vulkan_ops_fused.cpp
 * @brief Vulkan backend fused optimizer step operations
 */

#include "vulkan_ops_common.hpp"

namespace tenzor {

namespace {

// Fused optimizer steps mutate param / momentum / exp_avg / exp_avg_sq IN PLACE
// and return the same param storage. Such operands MUST NOT be replaced with a
// throwaway dispatchContiguous() copy — that would write the update into a
// discarded buffer and silently no-op the optimizer. Instead we require them to
// already be packed at offset 0 (the optimizer owns these buffers, so they
// normally are). A misaligned view here would otherwise trip the descriptor
// guard with an opaque message; this gives a precise one.
inline void require_inplace_contiguous(const Tensor& t, const char* op,
                                       const char* name) {
    if (!(t.is_contiguous() && t.offset() == 0)) {
        throw std::runtime_error(
            std::string("VulkanBackend::") + op + ": in-place operand '" + name +
            "' must be contiguous at storage offset 0 (got offset " +
            std::to_string(t.offset()) + "); an unmaterialized view cannot be "
            "updated in place — materialize it in the caller before the step");
    }
}

}  // namespace

// ============================================================================
// Fused SGD Step (OpId::FusedSGDStep, 219)
// ============================================================================

auto VulkanBackend::dispatchFusedSGDStep(std::span<const Tensor> inputs,
                                          const OpAttributes& attrs) -> std::vector<Tensor> {
    if (inputs.size() < 2) {
        throw std::invalid_argument("FusedSGDStep requires at least 2 inputs (param, grad)");
    }

    float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
    float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.0));
    float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
    float dampening = static_cast<float>(attrs.get_float(AttrKey::Dampening, 0.0));
    bool nesterov = attrs.get_bool(AttrKey::Nesterov, false);

    int64_t numel = inputs[0].numel();
    int32_t device_id = inputs[0].device().index;
    bool is_float16 = (inputs[0].dtype() == DType::Float16);
    bool is_float64 = (inputs[0].dtype() == DType::Float64);
    bool is_bfloat16 = (inputs[0].dtype() == DType::BFloat16);
    std::string sgd_shader = is_float16 ? "fused_sgd_step_f16"
                            : is_float64 ? "fused_sgd_step_f64"
                            : is_bfloat16 ? "fused_sgd_step_bf16"
                            : "fused_sgd_step";
    auto* pipeline = getPipeline(sgd_shader, device_id);

    bool has_momentum = (inputs.size() > 2 && momentum > 0.0f);

    // In-place operands: param (and momentum buffer when used) are updated in
    // place and must stay on their own storage — require offset-0 contiguous.
    require_inplace_contiguous(inputs[0], "FusedSGDStep", "param");
    if (has_momentum) {
        require_inplace_contiguous(inputs[2], "FusedSGDStep", "momentum_buf");
    }
    // grad is read-only; materialize it to a packed offset-0 buffer for binding.
    const Tensor grad_c = dispatchContiguous(inputs[1]);

    // For F16/BF16: param/grad are packed uint32, momentum is Float32.
    // audit-10 MM.2: state_buf_size must be Float32-sized for BOTH F16 and
    // BF16 paths.  Previously is_bfloat16 fell into the `else` branch and
    // used the half-packed param size (~2 bytes/elem) — but fused_sgd_
    // step_bf16.comp declares MomentumBuf as `float momentum_buf[]`
    // (F32 master weights, 4 bytes/elem).  Descriptor write under-reported
    // range → Vulkan validation error / OOB writes.  Mirrors fused_adam_
    // atan2_step's correct logic.
    size_t buf_size = (is_float16 || is_bfloat16) ? ((numel + 1) / 2) * 4 : numel * inputs[0].dtype_size();
    size_t state_buf_size = (is_float16 || is_bfloat16) ? numel * sizeof(float) : buf_size;

    // Bindings: grad(0), param(1), momentum_buf(2)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_c.data_ptr()},     // grad (read-only, materialized)
        {1, inputs[0].data_ptr()},  // param (modified in-place)
    };
    std::vector<size_t> sizes = {buf_size, buf_size};

    // Kept alive across the dispatch below; used only for the no-momentum
    // F16/BF16 dummy binding.
    Tensor momentum_scratch;
    if (has_momentum) {
        bindings.push_back({2, inputs[2].data_ptr()});
        sizes.push_back(state_buf_size);
    } else if (is_float16 || is_bfloat16) {
        // audit-10 OO.5: the dummy binding must declare the SHADER's expected
        // F32 momentum_buf layout (numel*4). Binding the packed-F16 param
        // (~numel*2 bytes) under a numel*4 range declares bytes past the
        // allocation end (validation error / MoltenVK fail). Allocate a small F32
        // scratch that actually spans state_buf_size; the shader never reads
        // binding 2 when has_momentum=0.
        momentum_scratch = Tensor({static_cast<int64_t>(numel)}, DType::Float32, inputs[0].device());
        bindings.push_back({2, momentum_scratch.data_ptr()});
        sizes.push_back(state_buf_size);
    } else {
        // F32/F64 param: state_buf_size == buf_size == the param buffer span, so
        // reusing the param buffer for the inert binding is in-bounds.
        bindings.push_back({2, inputs[0].data_ptr()});
        sizes.push_back(state_buf_size);
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t numel;
        float lr;
        float momentum;
        float weight_decay;
        float dampening;
        uint32_t nesterov;
        uint32_t has_momentum;
        uint32_t padding;
    } pc;

    pc.numel = static_cast<uint32_t>(numel);
    pc.lr = lr;
    pc.momentum = momentum;
    pc.weight_decay = weight_decay;
    pc.dampening = dampening;
    pc.nesterov = nesterov ? 1u : 0u;
    pc.has_momentum = has_momentum ? 1u : 0u;
    pc.padding = 0;

    // F16 shader processes pairs (1 thread per word), F32 processes 1 element per thread
    int64_t dispatch_count = (is_float16 || is_bfloat16) ? (numel + 1) / 2 : numel;  // BF16 also packs pairs (audit C9)
    uint32_t workgroups = div_wg(dispatch_count, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &pc);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {inputs[0]};  // Return param (modified in-place)
}

// ============================================================================
// Fused Adam Step (OpId::FusedAdamStep, 220)
// ============================================================================

auto VulkanBackend::dispatchFusedAdamStep(std::span<const Tensor> inputs,
                                           const OpAttributes& attrs) -> std::vector<Tensor> {
    // inputs: [param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq(optional)]
    if (inputs.size() < 4) {
        throw std::invalid_argument("FusedAdamStep requires at least 4 inputs (param, grad, exp_avg, exp_avg_sq)");
    }

    double lr = attrs.get_float(AttrKey::Lr, 0.001);
    double beta1 = attrs.get_float(AttrKey::Beta1, 0.9);
    double beta2 = attrs.get_float(AttrKey::Beta2, 0.999);
    double eps = attrs.get_float(AttrKey::Eps, 1e-8);
    double weight_decay = attrs.get_float(AttrKey::WeightDecay, 0.0);
    int64_t step = attrs.get_int(AttrKey::Step, 1);
    bool decoupled = attrs.get_bool(AttrKey::Decoupled, false);
    bool amsgrad = attrs.get_bool(AttrKey::Amsgrad, false);

    // Precompute bias corrections on host
    float bias_correction1 = static_cast<float>(1.0 - std::pow(beta1, step));
    float bias_correction2_sqrt = static_cast<float>(std::sqrt(1.0 - std::pow(beta2, step)));

    int64_t numel = inputs[0].numel();
    int32_t device_id = inputs[0].device().index;
    bool is_float64 = (inputs[0].dtype() == DType::Float64);
    bool is_float16 = (inputs[0].dtype() == DType::Float16);
    bool is_bfloat16 = (inputs[0].dtype() == DType::BFloat16);
    // Half precision: param/grad are packed F16/BF16 while the moment state
    // buffers (exp_avg, exp_avg_sq, max) are Float32 master copies — mirrors
    // fused_adam_atan2_step and fused_sgd_step. The optimizer allocates Float32
    // moment buffers for half-precision params (see vulkan fused_adam_step_f16/
    // _bf16 shaders). This is full on-device half-precision Adam, not a fallback.
    std::string adam_shader = is_float16 ? "fused_adam_step_f16"
                            : is_bfloat16 ? "fused_adam_step_bf16"
                            : is_float64 ? "fused_adam_step_f64"
                            : "fused_adam_step";
    auto* pipeline = getPipeline(adam_shader, device_id);

    // In-place operands: param / exp_avg / exp_avg_sq (and max_exp_avg_sq when
    // amsgrad) are updated in place and must stay on their own storage.
    require_inplace_contiguous(inputs[0], "FusedAdamStep", "param");
    require_inplace_contiguous(inputs[2], "FusedAdamStep", "exp_avg");
    require_inplace_contiguous(inputs[3], "FusedAdamStep", "exp_avg_sq");
    if (amsgrad && inputs.size() > 4) {
        require_inplace_contiguous(inputs[4], "FusedAdamStep", "max_exp_avg_sq");
    }
    // grad is read-only; materialize it to a packed offset-0 buffer for binding.
    const Tensor grad_c = dispatchContiguous(inputs[1]);

    size_t f16_buf_size = ((numel + 1) / 2) * 4;
    size_t param_buf_size = (is_float16 || is_bfloat16) ? f16_buf_size
                                                        : numel * inputs[0].dtype_size();
    size_t state_buf_size = (is_float16 || is_bfloat16) ? numel * sizeof(float)
                                                        : param_buf_size;

    // Bindings: grad(0), param(1), exp_avg(2), exp_avg_sq(3), max_exp_avg_sq(4)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_c.data_ptr()},     // grad (read-only, materialized)
        {1, inputs[0].data_ptr()},  // param
        {2, inputs[2].data_ptr()},  // exp_avg
        {3, inputs[3].data_ptr()},  // exp_avg_sq
    };
    std::vector<size_t> sizes = {param_buf_size, param_buf_size, state_buf_size, state_buf_size};

    if (amsgrad && inputs.size() > 4) {
        bindings.push_back({4, inputs[4].data_ptr()});
        sizes.push_back(state_buf_size);
    } else {
        bindings.push_back({4, inputs[0].data_ptr()});
        sizes.push_back(param_buf_size);
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t numel;
        float lr;
        float beta1;
        float beta2;
        float eps;
        float weight_decay;
        float bias_correction1;
        float bias_correction2_sqrt;
        uint32_t decoupled;
        uint32_t amsgrad;
        uint32_t padding0;
        uint32_t padding1;
    } pc;

    pc.numel = static_cast<uint32_t>(numel);
    pc.lr = static_cast<float>(lr);
    pc.beta1 = static_cast<float>(beta1);
    pc.beta2 = static_cast<float>(beta2);
    pc.eps = static_cast<float>(eps);
    pc.weight_decay = static_cast<float>(weight_decay);
    pc.bias_correction1 = bias_correction1;
    pc.bias_correction2_sqrt = bias_correction2_sqrt;
    pc.decoupled = decoupled ? 1u : 0u;
    pc.amsgrad = amsgrad ? 1u : 0u;
    pc.padding0 = 0;
    pc.padding1 = 0;

    // Half precision packs two elements per uint word; F32/F64 are per-element.
    int64_t dispatch_count = (is_float16 || is_bfloat16) ? (numel + 1) / 2 : numel;
    uint32_t workgroups = div_wg(dispatch_count, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &pc);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {inputs[0]};  // Return param (modified in-place)
}

// ============================================================================
// Fused Adam-Atan2 Optimizer Step
// ============================================================================

auto VulkanBackend::dispatchFusedAdamAtan2Step(std::span<const Tensor> inputs,
                                                const OpAttributes& attrs) -> std::vector<Tensor> {
    // inputs: [param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq(optional)]
    if (inputs.size() < 4) {
        throw std::invalid_argument("FusedAdamAtan2Step requires at least 4 inputs");
    }

    double lr = attrs.get_float(AttrKey::Lr, 0.001);
    double beta1 = attrs.get_float(AttrKey::Beta1, 0.9);
    double beta2 = attrs.get_float(AttrKey::Beta2, 0.999);
    double eps = attrs.get_float(AttrKey::Eps, 1e-8);
    double weight_decay = attrs.get_float(AttrKey::WeightDecay, 0.0);
    int64_t step = attrs.get_int(AttrKey::Step, 1);
    bool decoupled = attrs.get_bool(AttrKey::Decoupled, false);
    bool amsgrad = attrs.get_bool(AttrKey::Amsgrad, false);

    float bias_correction1 = static_cast<float>(1.0 - std::pow(beta1, step));
    float bias_correction2_sqrt = static_cast<float>(std::sqrt(1.0 - std::pow(beta2, step)));

    int64_t numel = inputs[0].numel();
    int32_t device_id = inputs[0].device().index;
    bool is_float16 = (inputs[0].dtype() == DType::Float16);
    bool is_float64 = (inputs[0].dtype() == DType::Float64);
    bool is_bfloat16 = (inputs[0].dtype() == DType::BFloat16);
    std::string shader = is_float16 ? "fused_adam_atan2_step_f16"
                       : is_float64 ? "fused_adam_atan2_step_f64"
                       : is_bfloat16 ? "fused_adam_atan2_step_bf16"
                       : "fused_adam_atan2_step";
    auto* pipeline = getPipeline(shader, device_id);

    // In-place operands: param / exp_avg / exp_avg_sq (and max_exp_avg_sq when
    // amsgrad) are updated in place and must stay on their own storage.
    require_inplace_contiguous(inputs[0], "FusedAdamAtan2Step", "param");
    require_inplace_contiguous(inputs[2], "FusedAdamAtan2Step", "exp_avg");
    require_inplace_contiguous(inputs[3], "FusedAdamAtan2Step", "exp_avg_sq");
    if (amsgrad && inputs.size() > 4) {
        require_inplace_contiguous(inputs[4], "FusedAdamAtan2Step", "max_exp_avg_sq");
    }
    // grad is read-only; materialize it to a packed offset-0 buffer for binding.
    const Tensor grad_c = dispatchContiguous(inputs[1]);

    size_t f16_buf_size = ((numel + 1) / 2) * 4;
    size_t param_buf_size = (is_float16 || is_bfloat16) ? f16_buf_size : numel * inputs[0].dtype_size();
    size_t state_buf_size = (is_float16 || is_bfloat16) ? numel * sizeof(float) : param_buf_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_c.data_ptr()},     // grad (read-only, materialized)
        {1, inputs[0].data_ptr()},  // param
        {2, inputs[2].data_ptr()},  // exp_avg
        {3, inputs[3].data_ptr()},  // exp_avg_sq
    };
    std::vector<size_t> sizes = {param_buf_size, param_buf_size, state_buf_size, state_buf_size};

    if (amsgrad && inputs.size() > 4) {
        bindings.push_back({4, inputs[4].data_ptr()});
        sizes.push_back(state_buf_size);
    } else {
        bindings.push_back({4, inputs[0].data_ptr()});
        sizes.push_back(param_buf_size);
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t numel;
        float lr;
        float beta1;
        float beta2;
        float eps;
        float weight_decay;
        float bias_correction1;
        float bias_correction2_sqrt;
        uint32_t decoupled;
        uint32_t amsgrad;
        uint32_t padding0;
        uint32_t padding1;
    } pc;

    pc.numel = static_cast<uint32_t>(numel);
    pc.lr = static_cast<float>(lr);
    pc.beta1 = static_cast<float>(beta1);
    pc.beta2 = static_cast<float>(beta2);
    pc.eps = static_cast<float>(eps);
    pc.weight_decay = static_cast<float>(weight_decay);
    pc.bias_correction1 = bias_correction1;
    pc.bias_correction2_sqrt = bias_correction2_sqrt;
    pc.decoupled = decoupled ? 1u : 0u;
    pc.amsgrad = amsgrad ? 1u : 0u;
    pc.padding0 = 0;
    pc.padding1 = 0;

    int64_t dispatch_count = (is_float16 || is_bfloat16) ? (numel + 1) / 2 : numel;  // BF16 also packs pairs (audit C9)
    uint32_t workgroups = div_wg(dispatch_count, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &pc);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {inputs[0]};
}

} // namespace tenzor
