/**
 * @file vulkan_ops_fused.cpp
 * @brief Vulkan backend fused optimizer step operations
 */

#include "vulkan_ops_common.hpp"

namespace tenzor {

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

    // For F16: param/grad are packed uint32, momentum is Float32
    size_t buf_size = (is_float16 || is_bfloat16) ? ((numel + 1) / 2) * 4 : numel * inputs[0].dtype_size();
    size_t state_buf_size = is_float16 ? numel * sizeof(float) : buf_size;

    // Bindings: grad(0), param(1), momentum_buf(2)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, inputs[1].data_ptr()},  // grad
        {1, inputs[0].data_ptr()},  // param (modified in-place)
    };
    std::vector<size_t> sizes = {buf_size, buf_size};

    if (has_momentum) {
        bindings.push_back({2, inputs[2].data_ptr()});
        sizes.push_back(state_buf_size);
    } else {
        // Dummy binding
        bindings.push_back({2, inputs[0].data_ptr()});
        sizes.push_back(buf_size);
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
    bool is_float16 = (inputs[0].dtype() == DType::Float16);
    bool is_float64 = (inputs[0].dtype() == DType::Float64);
    bool is_bfloat16 = (inputs[0].dtype() == DType::BFloat16);
    std::string adam_shader = is_float16 ? "fused_adam_step_f16"
                            : is_float64 ? "fused_adam_step_f64"
                            : is_bfloat16 ? "fused_adam_step_bf16"
                            : "fused_adam_step";
    auto* pipeline = getPipeline(adam_shader, device_id);

    // For F16/BF16: param/grad are packed uint32, state buffers are Float32
    size_t f16_buf_size = ((numel + 1) / 2) * 4;
    size_t param_buf_size = (is_float16 || is_bfloat16) ? f16_buf_size : numel * inputs[0].dtype_size();
    size_t state_buf_size = (is_float16 || is_bfloat16) ? numel * sizeof(float) : param_buf_size;

    // Bindings: grad(0), param(1), exp_avg(2), exp_avg_sq(3), max_exp_avg_sq(4)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, inputs[1].data_ptr()},  // grad
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

    size_t f16_buf_size = ((numel + 1) / 2) * 4;
    size_t param_buf_size = (is_float16 || is_bfloat16) ? f16_buf_size : numel * inputs[0].dtype_size();
    size_t state_buf_size = (is_float16 || is_bfloat16) ? numel * sizeof(float) : param_buf_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, inputs[1].data_ptr()},  // grad
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
