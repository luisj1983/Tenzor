#include "vulkan_ops_common.hpp"

#include <cstddef>  // offsetof — for partial push-constant ranges

namespace tenzor {

// Normalization operations implementation
//
// NOTE: the old non-functional `dispatchBatchNorm2d` (which dispatched a
// pipeline with no descriptor set / push constants / bindings and returned
// uninitialized memory) has been removed. The working forward path is
// dispatchBatchNorm2dForward below; backward is dispatchBatchNorm2dBackward.

auto VulkanBackend::dispatchBatchNorm2dBackward(const Tensor& grad_out, const Tensor& input,
                                                 const Tensor& mean, const Tensor& var,
                                                 const Tensor* gamma, float epsilon)
                                                 -> std::tuple<Tensor, Tensor, Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d_backward requires 4D input (N, C, H, W)");
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t spatial_size = height * width;
    int64_t n_elements = input.numel();

    int32_t device_id = input.device().index;

    // Materialize all read operands into packed, offset-0 buffers before they
    // reach descriptor writes. A stride-contiguous outer-dim view (offset != 0)
    // would otherwise trip the misaligned-descriptor guard. dispatchContiguous
    // is a no-op when the tensor is already contiguous at offset 0.
    const Tensor grad_out_c = dispatchContiguous(grad_out);
    const Tensor input_c = dispatchContiguous(input);
    const Tensor mean_c = dispatchContiguous(mean);
    const Tensor var_c = dispatchContiguous(var);
    const Tensor gamma_c = gamma ? dispatchContiguous(*gamma) : Tensor();

    // Float16 / BFloat16 use native packed-half backward shaders implementing the
    // correct training-mode two-pass algorithm (grad_gamma/grad_beta via global
    // atomicAdd in pass 1, then grad_input from the completed per-channel sums in
    // pass 2). The F16 shader (batchnorm2d_backward_f16) is a single pipeline that
    // selects the pass via the `pass` push-constant; the BF16 path uses two
    // separate pipelines (batchnorm2d_backward_bf16 then
    // batchnorm2d_backward_grad_input_bf16). Both are wired below.

    // Select shader based on dtype
    std::string shader_name = "batchnorm2d_backward";
    bool is_f64_bn = (input.dtype() == DType::Float64);
    if (is_f64_bn) {
        shader_name = "batchnorm2d_backward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "batchnorm2d_backward_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "batchnorm2d_backward_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    const bool bn_is_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);

    // Create output tensors
    std::vector<int64_t> grad_in_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_in_shape, input.dtype(), input.device());

    // For Float16, the backward shader accumulates grad_gamma/grad_beta in Float32
    // (mean/var are also Float32 for F16 input)
    DType stats_dtype = bn_is_half ? DType::Float32 : input.dtype();
    std::vector<int64_t> param_shape = {channels};
    // Initialize grad_gamma and grad_beta to zeros (non-f64 shaders use atomicAdd; f64 uses reduction pass)
    Tensor grad_gamma = dispatchZeros(param_shape, stats_dtype, input.device());
    Tensor grad_beta = dispatchZeros(param_shape, stats_dtype, input.device());

    // For Float16 input, cast gamma to Float32 if needed (shader expects Float32 stats)
    Tensor gamma_f32;
    const Tensor* gamma_effective = gamma ? &gamma_c : nullptr;
    if (gamma && bn_is_half && gamma_c.dtype() != DType::Float32) {
        gamma_f32 = dispatchContiguous(gamma_c.to(DType::Float32));
        gamma_effective = &gamma_f32;
    }

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_out_c.data_ptr();
    const void* buffer_input = input_c.data_ptr();
    const void* buffer_mean = mean_c.data_ptr();
    const void* buffer_var = var_c.data_ptr();
    const void* buffer_grad_input = grad_input.data_ptr();
    const void* buffer_grad_gamma = grad_gamma.data_ptr();
    const void* buffer_grad_beta = grad_beta.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_input = n_elements * input.dtype_size();
    if (bn_is_half) {
        // Round up to 4-byte boundary for uint32 shader access (2 half per uint32)
        size_t input_pairs = (n_elements + 1) / 2;
        buffer_size_input = input_pairs * 4;
    }
    // Statistics (mean, var, gamma, grad_gamma, grad_beta) use stats_dtype
    size_t buffer_size_channel = channels * mean.dtype_size();

    // For Float64: two-pass approach with partial buffers (no atomics needed)
    // Partial buffers hold per-(batch, channel) partial sums: shape [N * C]
    Tensor partial_grad_gamma, partial_grad_beta;
    if (is_f64_bn) {
        int64_t num_partials = batch * channels;
        partial_grad_gamma = Tensor({num_partials}, DType::Float64, input.device());
        partial_grad_beta = Tensor({num_partials}, DType::Float64, input.device());
    }

    // Set up descriptor bindings
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},    // grad_output
        {1, buffer_input},       // input
        {2, buffer_mean},        // mean
        {3, buffer_var},         // variance
    };
    std::vector<size_t> sizes = {
        buffer_size_input,  // grad_output
        buffer_size_input,  // input
        buffer_size_channel, // mean
        buffer_size_channel, // variance
    };

    // Handle optional gamma
    if (gamma_effective) {
        const void* buffer_gamma = gamma_effective->data_ptr();
        bindings.push_back({4, buffer_gamma});
        sizes.push_back(gamma_effective->numel() * gamma_effective->dtype_size());
    } else {
        // Use dummy buffer for gamma binding
        bindings.push_back({4, buffer_mean});
        sizes.push_back(buffer_size_channel);
    }

    // Add output buffers
    bindings.push_back({5, buffer_grad_input});

    if (is_f64_bn) {
        // f64: bindings 6/7 are partial buffers (not final grad_gamma/grad_beta)
        size_t partial_buf_size = batch * channels * sizeof(double);
        bindings.push_back({6, partial_grad_gamma.data_ptr()});
        bindings.push_back({7, partial_grad_beta.data_ptr()});
        sizes.push_back(buffer_size_input);   // grad_input
        sizes.push_back(partial_buf_size);     // partial_grad_gamma
        sizes.push_back(partial_buf_size);     // partial_grad_beta
    } else {
        bindings.push_back({6, buffer_grad_gamma});
        bindings.push_back({7, buffer_grad_beta});
        sizes.push_back(buffer_size_input);   // grad_input
        sizes.push_back(buffer_size_channel); // grad_gamma
        sizes.push_back(buffer_size_channel); // grad_beta
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants. The trailing `pass` field is only consumed by the F16
    // shader (batchnorm2d_backward_f16), which is a single pipeline that runs
    // both passes selected by `pass`. The F32 / BF16 / F64 shaders reflect a
    // 6-field (24-byte) push range, so for those we push only the first 24 bytes;
    // for F16 we push the full 28-byte struct.
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t spatial_size;
        float eps;
        uint32_t has_gamma;
        uint32_t pass;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(n_elements);
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
    push_constants.eps = epsilon;
    push_constants.has_gamma = (gamma != nullptr) ? 1 : 0;
    push_constants.pass = 0;  // pass 0 = grad_gamma/grad_beta reduction (F16)

    // Only the F16 shader declares the 7th `pass` field; everyone else has a
    // 24-byte (6-field) push range.
    const bool bn_uses_pass = (input.dtype() == DType::Float16);
    const uint32_t bn_push_size = bn_uses_pass
        ? static_cast<uint32_t>(sizeof(PushConstants))
        : static_cast<uint32_t>(offsetof(PushConstants, pass));

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, bn_push_size, &push_constants);

    if (is_f64_bn) {
        // f64: one workgroup per (batch, channel) pair
        uint32_t workgroups_bn = static_cast<uint32_t>(batch * channels);
        vkCmdDispatch(cmdBuffer, workgroups_bn, 1, 1);
    } else {
        // For packed half (F16/BF16), each thread processes a word (2 elements),
        // so dispatch half as many threads (word count).
        uint64_t dispatch_count = n_elements;
        if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
            dispatch_count = (n_elements + 1) / 2;  // number of words
        }
        uint32_t workgroups = div_wg(dispatch_count, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    }

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // Float16: pass 2 — re-run the SAME f16 pipeline with pass=1 to compute
    // grad_input from the completed per-channel grad_gamma/grad_beta atomics.
    if (input.dtype() == DType::Float16) {
        PushConstants pc_pass1 = push_constants;
        pc_pass1.pass = 1;

        VkDescriptorSet f16_gi_ds = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer f16_gi_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(f16_gi_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(f16_gi_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline->layout(), 0, 1, &f16_gi_ds, 0, nullptr);
        vkCmdPushConstants(f16_gi_cmd, pipeline->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, static_cast<uint32_t>(sizeof(PushConstants)), &pc_pass1);
        uint32_t f16_words = div_wg((n_elements + 1) / 2, devices_[device_id].workgroupSize);
        vkCmdDispatch(f16_gi_cmd, f16_words, 1, 1);
        insertComputeOnlyBarrier(f16_gi_cmd);
        endSingleTimeCommands(f16_gi_cmd, device_id);
    }

    // BFloat16: pass 2 — dedicated grad_input pipeline consuming the per-channel
    // grad_gamma/grad_beta accumulated by the bf16 pass-1 shader. Each thread
    // handles one packed word (2 BF16 elements), so dispatch by word count.
    if (input.dtype() == DType::BFloat16) {
        auto* bn_input_pipeline_bf16 =
            getPipeline("batchnorm2d_backward_grad_input_bf16", device_id);
        VkDescriptorSet bf16_gi_ds = allocateAndWriteDescriptorSet(
            device_id, bn_input_pipeline_bf16, bindings, sizes);

        VkCommandBuffer bf16_gi_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(bf16_gi_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          bn_input_pipeline_bf16->pipeline());
        vkCmdBindDescriptorSets(bf16_gi_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                bn_input_pipeline_bf16->layout(), 0, 1, &bf16_gi_ds, 0, nullptr);
        // bf16 grad_input shader reflects a 6-field (24-byte) push range.
        vkCmdPushConstants(bf16_gi_cmd, bn_input_pipeline_bf16->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, static_cast<uint32_t>(offsetof(PushConstants, pass)), &push_constants);
        uint32_t bf16_words = div_wg((n_elements + 1) / 2, devices_[device_id].workgroupSize);
        vkCmdDispatch(bf16_gi_cmd, bf16_words, 1, 1);
        insertComputeOnlyBarrier(bf16_gi_cmd);
        endSingleTimeCommands(bf16_gi_cmd, device_id);
    }

    // Float32 backward is now two passes: pass 1 accumulated grad_gamma/
    // grad_beta; pass 2 uses those per-channel sums to compute grad_input
    // with the correct batchnorm training-mode formula.
    // (Float64 path handles grad_input in its own dedicated pass-3 shader.)
    if (input.dtype() == DType::Float32) {
        auto* bn_input_pipeline = getPipeline("batchnorm2d_backward_grad_input", device_id);
        VkDescriptorSet bn_input_ds = allocateAndWriteDescriptorSet(
            device_id, bn_input_pipeline, bindings, sizes);

        VkCommandBuffer bn_input_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(bn_input_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          bn_input_pipeline->pipeline());
        vkCmdBindDescriptorSets(bn_input_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                bn_input_pipeline->layout(), 0, 1, &bn_input_ds, 0, nullptr);
        vkCmdPushConstants(bn_input_cmd, bn_input_pipeline->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, static_cast<uint32_t>(offsetof(PushConstants, pass)), &push_constants);
        uint32_t bn_input_wg = div_wg(n_elements, devices_[device_id].workgroupSize);
        vkCmdDispatch(bn_input_cmd, bn_input_wg, 1, 1);
        insertComputeOnlyBarrier(bn_input_cmd);
        endSingleTimeCommands(bn_input_cmd, device_id);
    }

    // Float64: pass 2 — reduce partial buffers into final grad_gamma/grad_beta
    if (is_f64_bn) {
        auto* reduce_pipeline = getPipeline("reduce_partial_sums_f64", device_id);

        struct ReducePC {
            uint32_t num_partials;
            uint32_t channels;
        } reduce_pc;
        reduce_pc.num_partials = static_cast<uint32_t>(batch);
        reduce_pc.channels = static_cast<uint32_t>(channels);

        size_t partial_buf_size = batch * channels * sizeof(double);
        std::vector<std::pair<uint32_t, const void*>> reduce_bindings = {
            {0, partial_grad_gamma.data_ptr()},
            {1, partial_grad_beta.data_ptr()},
            {2, grad_gamma.data_ptr()},
            {3, grad_beta.data_ptr()}
        };
        std::vector<size_t> reduce_sizes = {
            partial_buf_size, partial_buf_size, buffer_size_channel, buffer_size_channel
        };

        VkDescriptorSet reduce_ds = allocateAndWriteDescriptorSet(
            device_id, reduce_pipeline, reduce_bindings, reduce_sizes);

        VkCommandBuffer reduce_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(reduce_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reduce_pipeline->pipeline());
        vkCmdBindDescriptorSets(reduce_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               reduce_pipeline->layout(), 0, 1, &reduce_ds, 0, nullptr);
        vkCmdPushConstants(reduce_cmd, reduce_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(ReducePC), &reduce_pc);
        vkCmdDispatch(reduce_cmd, static_cast<uint32_t>(channels), 1, 1);
        insertComputeOnlyBarrier(reduce_cmd);
        endSingleTimeCommands(reduce_cmd, device_id);

        // Float64: pass 3 — compute grad_input from the final per-channel
        // grad_gamma/grad_beta sums using the training-mode formula. Pass 1 only
        // wrote the partial grad_gamma/grad_beta; grad_input was deliberately
        // left unwritten because it needs the full per-channel reductions.
        // Bindings 0-5 mirror the main pass; bindings 6/7 must point at the FINAL
        // (reduced) channel-sized grad_gamma/grad_beta, NOT the partial buffers.
        auto* bn_input_pipeline_f64 =
            getPipeline("batchnorm2d_backward_grad_input_f64", device_id);

        std::vector<std::pair<uint32_t, const void*>> gi_bindings = {
            {0, buffer_grad_out},
            {1, buffer_input},
            {2, buffer_mean},
            {3, buffer_var},
            {4, gamma_effective ? gamma_effective->data_ptr() : buffer_mean},
            {5, buffer_grad_input},
            {6, grad_gamma.data_ptr()},
            {7, grad_beta.data_ptr()},
        };
        std::vector<size_t> gi_sizes = {
            buffer_size_input,
            buffer_size_input,
            buffer_size_channel,
            buffer_size_channel,
            gamma_effective ? gamma_effective->numel() * gamma_effective->dtype_size()
                            : buffer_size_channel,
            buffer_size_input,
            buffer_size_channel,
            buffer_size_channel,
        };

        VkDescriptorSet gi_ds = allocateAndWriteDescriptorSet(
            device_id, bn_input_pipeline_f64, gi_bindings, gi_sizes);

        VkCommandBuffer gi_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(gi_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          bn_input_pipeline_f64->pipeline());
        vkCmdBindDescriptorSets(gi_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                bn_input_pipeline_f64->layout(), 0, 1, &gi_ds, 0, nullptr);
        vkCmdPushConstants(gi_cmd, bn_input_pipeline_f64->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, static_cast<uint32_t>(offsetof(PushConstants, pass)), &push_constants);
        uint32_t gi_wg = div_wg(n_elements, devices_[device_id].workgroupSize);
        vkCmdDispatch(gi_cmd, gi_wg, 1, 1);
        insertComputeOnlyBarrier(gi_cmd);
        endSingleTimeCommands(gi_cmd, device_id);
    }

    // For Float16 input, the shader accumulates grad_gamma/grad_beta in Float32
    // for precision; convert back to match the input dtype.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        grad_gamma = grad_gamma.to(input.dtype());
        grad_beta = grad_beta.to(input.dtype());
    }

    return {grad_input, grad_gamma, grad_beta};
}

// BatchNorm2d Forward - New implementation with proper buffer management
auto VulkanBackend::dispatchBatchNorm2dForward(const Tensor& input, const Tensor& mean, const Tensor& var,
                                               const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d_forward requires 4D input (N, C, H, W)");
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t spatial_size = height * width;

    int32_t device_id = input.device().index;

    // Materialize read operands to packed offset-0 buffers (see backward note).
    const Tensor input_c = dispatchContiguous(input);
    const Tensor mean_c = dispatchContiguous(mean);
    const Tensor var_c = dispatchContiguous(var);
    const Tensor gamma_c = gamma ? dispatchContiguous(*gamma) : Tensor();
    const Tensor beta_c = beta ? dispatchContiguous(*beta) : Tensor();

    // Select shader based on dtype
    std::string shader_name = "batchnorm2d_forward";
    if (input.dtype() == DType::Float64) {
        shader_name = "batchnorm2d_forward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "batchnorm2d_forward_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "batchnorm2d_forward_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // For Float16/BFloat16 input, the shader expects mean/var as Float32 for numerical stability
    // Keep converted tensors alive in this scope so their buffers remain valid
    Tensor mean_f32, var_f32;
    const Tensor* mean_ptr = &mean_c;
    const Tensor* var_ptr = &var_c;
    if ((input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) && mean_c.dtype() != DType::Float32) {
        mean_f32 = dispatchContiguous(mean_c.to(DType::Float32));
        var_f32 = dispatchContiguous(var_c.to(DType::Float32));
        mean_ptr = &mean_f32;
        var_ptr = &var_f32;
    }

    // Get VkBuffer handles
    const void* buffer_input = input_c.data_ptr();
    const void* buffer_mean = mean_ptr->data_ptr();
    const void* buffer_var = var_ptr->data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_mean = mean_ptr->numel() * mean_ptr->dtype_size();
    size_t buffer_size_var = var_ptr->numel() * var_ptr->dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16) {
        // Round up input/output to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t input_pairs = (input.numel() + 1) / 2;
        size_t output_pairs = (output.numel() + 1) / 2;
        buffer_size_input = input_pairs * 4;
        buffer_size_output = output_pairs * 4;
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},   // input
        {1, buffer_mean},    // mean
        {2, buffer_var},     // variance
    };
    std::vector<size_t> sizes = {buffer_size_input, buffer_size_mean, buffer_size_var};

    // Add optional gamma and beta buffers
    // Keep cast tensors alive in this scope so their buffers remain valid
    Tensor gamma_f32, beta_f32;
    if (gamma && beta) {
        const Tensor* gamma_ptr = &gamma_c;
        const Tensor* beta_ptr = &beta_c;

        // For Float16 input, the shader expects gamma/beta as Float32 for numerical stability
        if (input.dtype() == DType::Float16 && gamma_c.dtype() == DType::Float16) {
            gamma_f32 = dispatchContiguous(gamma_c.to(DType::Float32));
            beta_f32 = dispatchContiguous(beta_c.to(DType::Float32));
            gamma_ptr = &gamma_f32;
            beta_ptr = &beta_f32;
        }

        const void* buffer_gamma = gamma_ptr->data_ptr();
        const void* buffer_beta = beta_ptr->data_ptr();
        size_t buffer_size_gamma = gamma_ptr->numel() * gamma_ptr->dtype_size();
        size_t buffer_size_beta = beta_ptr->numel() * beta_ptr->dtype_size();

        bindings.push_back({3, buffer_gamma});
        bindings.push_back({4, buffer_beta});
        bindings.push_back({5, buffer_output});

        sizes.push_back(buffer_size_gamma);
        sizes.push_back(buffer_size_beta);
        sizes.push_back(buffer_size_output);
    } else {
        // Create dummy buffers for bindings 3 and 4
        bindings.push_back({3, buffer_mean});  // dummy
        bindings.push_back({4, buffer_mean});  // dummy
        bindings.push_back({5, buffer_output});

        sizes.push_back(buffer_size_mean);
        sizes.push_back(buffer_size_mean);
        sizes.push_back(buffer_size_output);
    }

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
        uint32_t spatial_size;
        float eps;
        uint32_t has_affine;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(input.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
    push_constants.eps = epsilon;
    push_constants.has_affine = (gamma && beta) ? 1 : 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    // For Float16, each thread processes a word (2 elements), so dispatch half as many threads
    uint64_t dispatch_count = input.numel();
    if (input.dtype() == DType::Float16) {
        dispatch_count = (input.numel() + 1) / 2;  // number of words
    }
    uint32_t workgroups = static_cast<uint32_t>(div_wg(dispatch_count, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return fp16_saturate_if_needed(*this, output);
}

// BatchNorm2d Mean and Variance computation
auto VulkanBackend::dispatchBatchNorm2dMeanVar(const Tensor& input) -> std::pair<Tensor, Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d_mean_var requires 4D input (N, C, H, W)");
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t spatial_size = height * width;
    int64_t normalizer = batch * spatial_size;

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "batchnorm2d_mean_var";
    if (input.dtype() == DType::Float64) {
        // batchnorm2d_mean_var_f64.comp is now a deterministic
        // one-workgroup-per-channel tree reduction (mirroring the F32 path),
        // using a plain shared double array instead of float64 buffer
        // atomicAdd. No shaderBufferFloat64AtomicAdd gate is required.
        shader_name = "batchnorm2d_mean_var_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "batchnorm2d_mean_var_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Materialize input to a packed offset-0 buffer before binding.
    const Tensor input_c = dispatchContiguous(input);

    // Create output tensors - statistics are always Float32 (for F16 inputs) or same dtype
    // For F16: accumulation in Float32 for numerical stability, output as Float32
    DType stats_dtype = (input.dtype() == DType::Float16) ? DType::Float32 : input.dtype();
    std::vector<int64_t> stats_shape = {channels};
    Tensor mean(stats_shape, stats_dtype, input.device());
    Tensor variance(stats_shape, stats_dtype, input.device());
    Tensor temp_sum(stats_shape, stats_dtype, input.device());

    // Initialize outputs to zero using fill operation
    mean = dispatchFill(mean, 0.0f);
    variance = dispatchFill(variance, 0.0f);
    temp_sum = dispatchFill(temp_sum, 0.0f);

    // Get VkBuffer handles
    const void* buffer_input = input_c.data_ptr();
    const void* buffer_mean = mean.data_ptr();
    const void* buffer_var = variance.data_ptr();
    const void* buffer_temp = temp_sum.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    if (input.dtype() == DType::Float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t input_pairs = (input.numel() + 1) / 2;
        buffer_size_input = input_pairs * 4;
    }
    size_t buffer_size_stats = channels * mean.dtype_size();

    // First pass: compute mean
    {
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_input},
            {1, buffer_mean},
            {2, buffer_var},
            {3, buffer_temp}
        };
        std::vector<size_t> sizes = {buffer_size_input, buffer_size_stats, buffer_size_stats, buffer_size_stats};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        struct PushConstants {
            uint32_t n_elements;
            uint32_t batch;
            uint32_t channels;
            uint32_t spatial_size;
            uint32_t pass_id;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(input.numel());
        push_constants.batch = static_cast<uint32_t>(batch);
        push_constants.channels = static_cast<uint32_t>(channels);
        push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
        push_constants.pass_id = 0;  // First pass: mean

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        // The F32 and F64 shaders are deterministic one-workgroup-per-channel
        // tree reductions (see batchnorm2d_mean_var.comp / *_f64.comp); the f16
        // variant still uses one-thread-per-element atomics.
        uint32_t workgroups = (shader_name == "batchnorm2d_mean_var" ||
                               shader_name == "batchnorm2d_mean_var_f64")
            ? static_cast<uint32_t>(channels)
            : static_cast<uint32_t>(div_wg(input.numel(), devices_[device_id].workgroupSize));
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
    }

    // Normalize mean: temp_sum / normalizer -> mean
    {
        Tensor normalizer_tensor(stats_shape, stats_dtype, input.device());
        normalizer_tensor = dispatchFill(normalizer_tensor, static_cast<float>(normalizer));
        mean = dispatchBinaryOp("div", temp_sum, normalizer_tensor);
    }

    // Update buffer_mean to point to the newly computed mean tensor
    buffer_mean = mean.data_ptr();
    buffer_size_stats = channels * mean.dtype_size();

    // Second pass: compute variance
    {
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_input},
            {1, buffer_mean},
            {2, buffer_var},
            {3, buffer_temp}
        };
        std::vector<size_t> sizes = {buffer_size_input, buffer_size_stats, buffer_size_stats, buffer_size_stats};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        struct PushConstants {
            uint32_t n_elements;
            uint32_t batch;
            uint32_t channels;
            uint32_t spatial_size;
            uint32_t pass_id;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(input.numel());
        push_constants.batch = static_cast<uint32_t>(batch);
        push_constants.channels = static_cast<uint32_t>(channels);
        push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
        push_constants.pass_id = 1;  // Second pass: variance

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        // Same deterministic per-channel dispatch as the mean pass.
        uint32_t workgroups = (shader_name == "batchnorm2d_mean_var" ||
                               shader_name == "batchnorm2d_mean_var_f64")
            ? static_cast<uint32_t>(channels)
            : static_cast<uint32_t>(div_wg(input.numel(), devices_[device_id].workgroupSize));
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
    }

    // Normalize variance: var_data / normalizer -> variance
    {
        Tensor normalizer_tensor(stats_shape, stats_dtype, input.device());
        normalizer_tensor = dispatchFill(normalizer_tensor, static_cast<float>(normalizer));
        variance = dispatchBinaryOp("div", variance, normalizer_tensor);
    }

    return {mean, variance};
}

auto VulkanBackend::dispatchLayerNorm(const Tensor& input, int64_t normalized_shape,
                                      const Tensor* gamma, const Tensor* beta, float epsilon)
                                      -> std::tuple<Tensor, Tensor, Tensor> {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // For Float16/BFloat16, upcast to Float32 for numerical stability
    // (BFloat16 has only 7 mantissa bits, gamma/beta dtype must also match shader expectations).
    // The mean/rstd tensors stay Float32 as in CPU/CUDA — they're a saved-stats
    // contract, not user-facing data, so dtype is fixed.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        Tensor gamma_f32, beta_f32;
        const Tensor* gamma_ptr = nullptr;
        const Tensor* beta_ptr = nullptr;
        if (gamma && beta) {
            gamma_f32 = gamma->to(DType::Float32);
            beta_f32 = beta->to(DType::Float32);
            gamma_ptr = &gamma_f32;
            beta_ptr = &beta_f32;
        }
        auto [out_f32, mean_f32, rstd_f32] =
            dispatchLayerNorm(input_f32, normalized_shape, gamma_ptr, beta_ptr, epsilon);
        return {out_f32.to(orig_dtype), mean_f32, rstd_f32};
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "layer_norm_f64";
    } else if (is_bfloat16) {
        shader_name = "layer_norm_bf16";
    } else {
        shader_name = "layer_norm";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    int64_t batch_size = input.numel() / normalized_shape;
    bool has_affine = (gamma != nullptr && beta != nullptr);

    // Phase P0 / Fix 4: persist mean and rstd buffers — one Float32 scalar
    // per batch element. Vulkan's contract now matches CPU/CUDA/OneAPI:
    // forward returns {output, mean, rstd} so backward can read them
    // without re-deriving from input.
    // For Float64 input, keep the saved stats in Float64 so the f64 backward
    // reads full-precision mean/rstd. The f64 forward shader writes `double`
    // mean_out/rstd_out; storing them as Float32 truncated to ~1e-7 relative
    // error (visible as f64 gradcheck failures). Other dtypes keep Float32 (the
    // saved-stats contract shared with CPU/CUDA).
    DType stat_dtype = is_float64 ? DType::Float64 : DType::Float32;
    Tensor mean({batch_size}, stat_dtype, input.device());
    Tensor rstd({batch_size}, stat_dtype, input.device());
    size_t stat_buffer_size = batch_size * (is_float64 ? sizeof(double) : sizeof(float));

    // Materialize read operands to packed offset-0 buffers before binding.
    const Tensor input_c = dispatchContiguous(input);
    const Tensor gamma_c = gamma ? dispatchContiguous(*gamma) : Tensor();
    const Tensor beta_c = beta ? dispatchContiguous(*beta) : Tensor();

    // Get VkBuffer handles
    const void* buffer_input = input_c.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_mean = mean.data_ptr();
    const void* buffer_rstd = rstd.data_ptr();

    size_t elem_size = input.dtype_size();
    size_t input_buffer_size = input.numel() * elem_size;
    size_t output_buffer_size = output.numel() * elem_size;
    size_t norm_buffer_size = normalized_shape * elem_size;

    // Build buffer bindings: input(0), output(1), gamma(2), beta(3), mean(4), rstd(5)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output}
    };
    std::vector<size_t> sizes = {input_buffer_size, output_buffer_size};

    if (has_affine) {
        const void* buffer_gamma = gamma_c.data_ptr();
        const void* buffer_beta = beta_c.data_ptr();
        bindings.push_back({2, buffer_gamma});
        bindings.push_back({3, buffer_beta});
        sizes.push_back(norm_buffer_size);
        sizes.push_back(norm_buffer_size);
    } else {
        // Bind dummy buffers (use output buffer as placeholder for unused bindings)
        bindings.push_back({2, buffer_output});
        bindings.push_back({3, buffer_output});
        sizes.push_back(output_buffer_size);
        sizes.push_back(output_buffer_size);
    }

    // Saved-stats buffers (always bound — see binding 4/5 in layer_norm.comp).
    bindings.push_back({4, buffer_mean});
    bindings.push_back({5, buffer_rstd});
    sizes.push_back(stat_buffer_size);
    sizes.push_back(stat_buffer_size);

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants: batch_size, normalized_shape, epsilon, affine
    struct PushConstants {
        uint32_t batch_size;
        uint32_t normalized_shape;
        float epsilon;
        uint32_t affine;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.normalized_shape = static_cast<uint32_t>(normalized_shape);
    push_constants.epsilon = epsilon;
    push_constants.affine = has_affine ? 1u : 0u;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return {fp16_saturate_if_needed(*this, output), mean, rstd};
}

auto VulkanBackend::dispatchGroupNorm(const Tensor& input, int64_t num_groups,
                                      const Tensor* gamma, const Tensor* beta, float epsilon) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() < 2) {
        throw std::invalid_argument("group_norm requires at least 2D input");
    }

    // Unsupported types convert to Float32. BFloat16 has a native packed shader
    // (group_norm_bf16) selected below, so it is intentionally excluded here.
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float16 &&
        input.dtype() != DType::Float64 && input.dtype() != DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        Tensor gamma_f32, beta_f32;
        const Tensor* gamma_ptr = nullptr;
        const Tensor* beta_ptr = nullptr;
        if (gamma && beta) {
            gamma_f32 = gamma->to(DType::Float32);
            beta_f32 = beta->to(DType::Float32);
            gamma_ptr = &gamma_f32;
            beta_ptr = &beta_f32;
        }
        auto results = dispatchGroupNorm(input_f32, num_groups, gamma_ptr, beta_ptr, epsilon);
        return {results[0].to(orig_dtype), results[1], results[2]};
    }

    int32_t device_id = input.device().index;

    // Select shader variant by dtype
    std::string shader_name = "group_norm";
    if (input.dtype() == DType::Float64) {
        shader_name = "group_norm_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "group_norm_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "group_norm_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = (input_shape.size() > 2) ? input_shape[2] : 1;
    int64_t W = (input_shape.size() > 3) ? input_shape[3] : 1;
    // For >4D, fold extra spatial dims into W
    for (size_t i = 4; i < input_shape.size(); ++i) {
        W *= input_shape[i];
    }

    bool has_affine = (gamma != nullptr && beta != nullptr);

    // Create output tensors
    // audit-2026-05-03 — stats dtype must match shader binding type. The F64
    // shader declares `double mean_out[]`; allocating Float32 here meant the
    // shader wrote 8 bytes per element into a buffer sized for 4 bytes,
    // corrupting adjacent memory.
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());
    DType stats_dtype = (input.dtype() == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor mean_out({N, num_groups}, stats_dtype, input.device());
    Tensor inv_std_out({N, num_groups}, stats_dtype, input.device());

    size_t elem_size = input.dtype_size();
    size_t stats_elem_size = (stats_dtype == DType::Float64) ? sizeof(double) : sizeof(float);

    // Materialize read operands to packed offset-0 buffers before binding.
    const Tensor input_c = dispatchContiguous(input);
    const Tensor gamma_c = gamma ? dispatchContiguous(*gamma) : Tensor();
    const Tensor beta_c = beta ? dispatchContiguous(*beta) : Tensor();

    // Get VkBuffer handles
    const void* buf_input = input_c.data_ptr();
    const void* buf_output = output.data_ptr();
    const void* buf_mean = mean_out.data_ptr();
    const void* buf_inv_std = inv_std_out.data_ptr();

    // F16 and BF16 both pack 2 elements per uint32 word (4-byte aligned) and use
    // Float32 channel statistics / affine params.
    bool is_packed_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
    // For packed half: packed uint32 words, 4-byte aligned
    size_t input_buf_size = is_packed_half ? ((input.numel() + 1) / 2) * 4 : input.numel() * elem_size;
    size_t output_buf_size = is_packed_half ? ((output.numel() + 1) / 2) * 4 : output.numel() * elem_size;
    size_t stats_buf_size = N * num_groups * stats_elem_size;

    // Bindings: input(0), output(1), gamma(2), beta(3), mean(4), inv_std(5)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_input},
        {1, buf_output},
    };
    std::vector<size_t> sizes = {input_buf_size, output_buf_size};

    if (has_affine) {
        const void* buf_gamma = gamma_c.data_ptr();
        const void* buf_beta = beta_c.data_ptr();
        bindings.push_back({2, buf_gamma});
        bindings.push_back({3, buf_beta});
        // Gamma/beta are always Float32 for the packed-half (F16/BF16) shaders,
        // elem_size for F32/F64
        size_t affine_elem_size = is_packed_half ? sizeof(float) : elem_size;
        sizes.push_back(C * affine_elem_size);
        sizes.push_back(C * affine_elem_size);
    } else {
        // Bind dummy buffers for unused bindings
        bindings.push_back({2, buf_output});
        bindings.push_back({3, buf_output});
        sizes.push_back(output_buf_size);
        sizes.push_back(output_buf_size);
    }

    bindings.push_back({4, buf_mean});
    bindings.push_back({5, buf_inv_std});
    sizes.push_back(stats_buf_size);
    sizes.push_back(stats_buf_size);

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t num_channels;
        uint32_t height;
        uint32_t width;
        uint32_t num_groups;
        float epsilon;
        uint32_t affine;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(N);
    push_constants.num_channels = static_cast<uint32_t>(C);
    push_constants.height = static_cast<uint32_t>(H);
    push_constants.width = static_cast<uint32_t>(W);
    push_constants.num_groups = static_cast<uint32_t>(num_groups);
    push_constants.epsilon = epsilon;
    push_constants.affine = has_affine ? 1u : 0u;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per (batch, group) pair
    uint32_t workgroups = static_cast<uint32_t>(N * num_groups);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {fp16_saturate_if_needed(*this, output), mean_out, inv_std_out};
}

// LayerNorm Backward - GPU implementation
auto VulkanBackend::dispatchLayerNormBackward(const Tensor& grad_output, const Tensor& input,
                                               const Tensor& mean, const Tensor& rstd,
                                               const Tensor* weight, int64_t normalized_shape)
                                               -> std::tuple<Tensor, Tensor, Tensor> {
    int32_t device_id = input.device().index;

    // For Float16/BFloat16, upcast to Float32. Besides numerical stability, this
    // avoids the bf16 backward shader reading the Float32 saved mean/rstd (the
    // forward saved-stats contract) as packed BF16, which corrupts grad_input.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto rstd_f32 = rstd.to(DType::Float32);
        Tensor w_f32;
        const Tensor* w_ptr = nullptr;
        if (weight) {
            w_f32 = weight->to(DType::Float32);
            w_ptr = &w_f32;
        }
        auto [gi, gw, gb] = dispatchLayerNormBackward(go_f32, in_f32, mean_f32, rstd_f32, w_ptr, normalized_shape);
        return {gi.to(orig_dtype), gw.to(orig_dtype), gb.to(orig_dtype)};
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "layer_norm_backward_f64" : is_bfloat16 ? "layer_norm_backward_bf16" : "layer_norm_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Saved mean/rstd are Float32 (the saved-stats contract). The f64 backward
    // shader declares them as double[]; reading Float32 bytes as double yields
    // garbage rstd, which trips the (rstd > 100) zero-variance guard and zeros
    // grad_input. Widen the stats to Float64 so the shader reads them correctly.
    Tensor mean_use = mean;
    Tensor rstd_use = rstd;
    if (is_float64) {
        if (mean_use.dtype() != DType::Float64) mean_use = mean_use.to(DType::Float64);
        if (rstd_use.dtype() != DType::Float64) rstd_use = rstd_use.to(DType::Float64);
    }

    // Materialize read operands to packed offset-0 buffers before binding.
    const Tensor grad_output_c = dispatchContiguous(grad_output);
    const Tensor input_c = dispatchContiguous(input);
    mean_use = dispatchContiguous(mean_use);
    rstd_use = dispatchContiguous(rstd_use);
    const Tensor weight_c = weight ? dispatchContiguous(*weight) : Tensor();

    int64_t batch_size = input.numel() / normalized_shape;
    bool has_affine = (weight != nullptr);

    // Create output tensors
    Tensor grad_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
    Tensor grad_weight = dispatchZeros({normalized_shape}, input.dtype(), input.device());
    Tensor grad_bias = dispatchZeros({normalized_shape}, input.dtype(), input.device());

    size_t elem_size = input.dtype_size();

    // For Float64: allocate partial buffers for two-pass reduction (no atomics)
    Tensor partial_grad_weight_ln, partial_grad_bias_ln;
    if (is_float64) {
        partial_grad_weight_ln = Tensor({batch_size * normalized_shape}, DType::Float64, input.device());
        partial_grad_bias_ln = Tensor({batch_size * normalized_shape}, DType::Float64, input.device());
    }

    // Get VkBuffer handles
    const void* buf_grad_out = grad_output_c.data_ptr();
    const void* buf_input = input_c.data_ptr();
    const void* buf_mean = mean_use.data_ptr();
    const void* buf_rstd = rstd_use.data_ptr();
    const void* buf_grad_input = grad_input.data_ptr();
    const void* buf_grad_weight = grad_weight.data_ptr();
    const void* buf_grad_bias = grad_bias.data_ptr();

    size_t input_buf_size = input.numel() * elem_size;
    size_t stats_buf_size = batch_size * elem_size;
    size_t norm_buf_size = normalized_shape * elem_size;

    // Bindings: grad_output(0), input(1), mean(2), rstd(3), weight(4), grad_input(5), grad_weight/partial(6), grad_bias/partial(7)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_grad_out},
        {1, buf_input},
        {2, buf_mean},
        {3, buf_rstd},
    };
    std::vector<size_t> sizes = {input_buf_size, input_buf_size, stats_buf_size, stats_buf_size};

    if (has_affine) {
        const void* buf_weight = weight_c.data_ptr();
        bindings.push_back({4, buf_weight});
        sizes.push_back(norm_buf_size);
    } else {
        bindings.push_back({4, buf_grad_input});  // dummy
        sizes.push_back(input_buf_size);
    }

    bindings.push_back({5, buf_grad_input});
    sizes.push_back(input_buf_size);

    if (is_float64) {
        // f64: bindings 6/7 are partial buffers sized [batch_size * normalized_shape]
        size_t partial_buf_size = batch_size * normalized_shape * sizeof(double);
        bindings.push_back({6, partial_grad_weight_ln.data_ptr()});
        bindings.push_back({7, partial_grad_bias_ln.data_ptr()});
        sizes.push_back(partial_buf_size);
        sizes.push_back(partial_buf_size);
    } else {
        bindings.push_back({6, buf_grad_weight});
        bindings.push_back({7, buf_grad_bias});
        sizes.push_back(norm_buf_size);
        sizes.push_back(norm_buf_size);
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t normalized_shape;
        uint32_t affine;
        uint32_t padding;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.normalized_shape = static_cast<uint32_t>(normalized_shape);
    push_constants.affine = has_affine ? 1u : 0u;
    push_constants.padding = 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // Float64: pass 2 — reduce partial buffers into final grad_weight/grad_bias
    if (is_float64) {
        auto* reduce_pipeline = getPipeline("reduce_partial_sums_f64", device_id);

        struct ReducePC {
            uint32_t num_partials;
            uint32_t channels;
        } reduce_pc;
        reduce_pc.num_partials = static_cast<uint32_t>(batch_size);
        reduce_pc.channels = static_cast<uint32_t>(normalized_shape);

        size_t partial_buf_size = batch_size * normalized_shape * sizeof(double);
        std::vector<std::pair<uint32_t, const void*>> reduce_bindings = {
            {0, partial_grad_weight_ln.data_ptr()},
            {1, partial_grad_bias_ln.data_ptr()},
            {2, grad_weight.data_ptr()},
            {3, grad_bias.data_ptr()}
        };
        std::vector<size_t> reduce_sizes = {
            partial_buf_size, partial_buf_size, norm_buf_size, norm_buf_size
        };

        VkDescriptorSet reduce_ds = allocateAndWriteDescriptorSet(
            device_id, reduce_pipeline, reduce_bindings, reduce_sizes);

        VkCommandBuffer reduce_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(reduce_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reduce_pipeline->pipeline());
        vkCmdBindDescriptorSets(reduce_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               reduce_pipeline->layout(), 0, 1, &reduce_ds, 0, nullptr);
        vkCmdPushConstants(reduce_cmd, reduce_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(ReducePC), &reduce_pc);
        vkCmdDispatch(reduce_cmd, static_cast<uint32_t>(normalized_shape), 1, 1);
        insertComputeOnlyBarrier(reduce_cmd);
        endSingleTimeCommands(reduce_cmd, device_id);
    }

    return {grad_input, grad_weight, grad_bias};
}

// GroupNorm Backward - GPU implementation
auto VulkanBackend::dispatchGroupNormBackward(const Tensor& grad_output, const Tensor& input,
                                               const Tensor& mean, const Tensor& rstd,
                                               const Tensor* weight, int64_t num_groups)
                                               -> std::tuple<Tensor, Tensor, Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("group_norm_backward requires 4D input (N, C, H, W)");
    }

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];
    int32_t device_id = input.device().index;

    // For Float16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16) {
        DType orig_dtype = input.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto rstd_f32 = rstd.to(DType::Float32);
        Tensor w_f32;
        const Tensor* w_ptr = nullptr;
        if (weight) {
            w_f32 = weight->to(DType::Float32);
            w_ptr = &w_f32;
        }
        auto [gi, gw, gb] = dispatchGroupNormBackward(go_f32, in_f32, mean_f32, rstd_f32, w_ptr, num_groups);
        return {gi.to(orig_dtype), gw.to(orig_dtype), gb.to(orig_dtype)};
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "group_norm_backward_f64" : is_bfloat16 ? "group_norm_backward_bf16" : "group_norm_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    bool has_affine = (weight != nullptr);

    // Create output tensors
    Tensor grad_input(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                      input.dtype(), input.device());
    Tensor grad_weight = dispatchZeros({C}, input.dtype(), input.device());
    Tensor grad_bias = dispatchZeros({C}, input.dtype(), input.device());

    size_t elem_size = input.dtype_size();
    size_t input_buf_size = input.numel() * elem_size;
    size_t stats_buf_size = N * num_groups * elem_size;
    size_t channel_buf_size = C * elem_size;

    // For Float64: allocate partial buffers for two-pass reduction (no atomics)
    // Partials: one per (batch, group) workgroup, sized [N * num_groups * C]
    int64_t num_wg_gn = N * num_groups;
    Tensor partial_grad_weight_gn, partial_grad_bias_gn;
    if (is_float64) {
        partial_grad_weight_gn = Tensor({num_wg_gn * C}, DType::Float64, input.device());
        partial_grad_bias_gn = Tensor({num_wg_gn * C}, DType::Float64, input.device());
    }

    // Materialize read operands to packed offset-0 buffers before binding.
    const Tensor grad_output_c = dispatchContiguous(grad_output);
    const Tensor input_c = dispatchContiguous(input);
    const Tensor mean_c = dispatchContiguous(mean);
    const Tensor rstd_c = dispatchContiguous(rstd);
    const Tensor weight_c = weight ? dispatchContiguous(*weight) : Tensor();

    const void* buf_grad_out = grad_output_c.data_ptr();
    const void* buf_input = input_c.data_ptr();
    const void* buf_mean = mean_c.data_ptr();
    const void* buf_rstd = rstd_c.data_ptr();
    const void* buf_grad_input = grad_input.data_ptr();
    const void* buf_grad_weight = grad_weight.data_ptr();
    const void* buf_grad_bias = grad_bias.data_ptr();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_grad_out},
        {1, buf_input},
        {2, buf_mean},
        {3, buf_rstd},
    };
    std::vector<size_t> sizes = {input_buf_size, input_buf_size, stats_buf_size, stats_buf_size};

    if (has_affine) {
        const void* buf_weight = weight_c.data_ptr();
        bindings.push_back({4, buf_weight});
        sizes.push_back(channel_buf_size);
    } else {
        bindings.push_back({4, buf_grad_input});  // dummy
        sizes.push_back(input_buf_size);
    }

    bindings.push_back({5, buf_grad_input});
    sizes.push_back(input_buf_size);

    if (is_float64) {
        // f64: bindings 6/7 are partial buffers sized [N * num_groups * C]
        size_t partial_buf_size = num_wg_gn * C * sizeof(double);
        bindings.push_back({6, partial_grad_weight_gn.data_ptr()});
        bindings.push_back({7, partial_grad_bias_gn.data_ptr()});
        sizes.push_back(partial_buf_size);
        sizes.push_back(partial_buf_size);
    } else {
        bindings.push_back({6, buf_grad_weight});
        bindings.push_back({7, buf_grad_bias});
        sizes.push_back(channel_buf_size);
        sizes.push_back(channel_buf_size);
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t num_channels;
        uint32_t height;
        uint32_t width;
        uint32_t num_groups;
        float epsilon;
        uint32_t affine;
        uint32_t padding;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(N);
    push_constants.num_channels = static_cast<uint32_t>(C);
    push_constants.height = static_cast<uint32_t>(H);
    push_constants.width = static_cast<uint32_t>(W);
    push_constants.num_groups = static_cast<uint32_t>(num_groups);
    push_constants.epsilon = 0.0f;  // not used in backward, but part of push constant layout
    push_constants.affine = has_affine ? 1u : 0u;
    push_constants.padding = 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per (batch, group) pair
    uint32_t workgroups = static_cast<uint32_t>(num_wg_gn);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // Float64: pass 2 — reduce partial buffers into final grad_weight/grad_bias
    if (is_float64) {
        auto* reduce_pipeline = getPipeline("reduce_partial_sums_f64", device_id);

        struct ReducePC {
            uint32_t num_partials;
            uint32_t channels;
        } reduce_pc;
        reduce_pc.num_partials = static_cast<uint32_t>(num_wg_gn);
        reduce_pc.channels = static_cast<uint32_t>(C);

        size_t partial_buf_size = num_wg_gn * C * sizeof(double);
        std::vector<std::pair<uint32_t, const void*>> reduce_bindings = {
            {0, partial_grad_weight_gn.data_ptr()},
            {1, partial_grad_bias_gn.data_ptr()},
            {2, grad_weight.data_ptr()},
            {3, grad_bias.data_ptr()}
        };
        std::vector<size_t> reduce_sizes = {
            partial_buf_size, partial_buf_size, channel_buf_size, channel_buf_size
        };

        VkDescriptorSet reduce_ds = allocateAndWriteDescriptorSet(
            device_id, reduce_pipeline, reduce_bindings, reduce_sizes);

        VkCommandBuffer reduce_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(reduce_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reduce_pipeline->pipeline());
        vkCmdBindDescriptorSets(reduce_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               reduce_pipeline->layout(), 0, 1, &reduce_ds, 0, nullptr);
        vkCmdPushConstants(reduce_cmd, reduce_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(ReducePC), &reduce_pc);
        vkCmdDispatch(reduce_cmd, static_cast<uint32_t>(C), 1, 1);
        insertComputeOnlyBarrier(reduce_cmd);
        endSingleTimeCommands(reduce_cmd, device_id);
    }

    return {grad_input, grad_weight, grad_bias};
}

// Embedding Backward - GPU implementation
auto VulkanBackend::dispatchEmbeddingBackward(const Tensor& grad_output, const Tensor& indices_in,
                                                int64_t num_embeddings, int64_t embedding_dim) -> Tensor {
    int32_t device_id = grad_output.device().index;
    // All embedding_backward shaders read `int indices[]` (32-bit). If the
    // caller passed Int64 the 4-byte stride halves the effective indices and
    // scatter accumulations land in the wrong rows of grad_weight.
    Tensor indices = (indices_in.dtype() == DType::Int64)
        ? indices_in.to(DType::Int32)
        : indices_in;
    int64_t num_indices = indices.numel();

    // For Float16, upcast to Float32 for numerical stability
    if (grad_output.dtype() == DType::Float16) {
        DType orig_dtype = grad_output.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto result_f32 = dispatchEmbeddingBackward(go_f32, indices, num_embeddings, embedding_dim);
        return result_f32.to(orig_dtype);
    }

    // Materialize read operands to packed offset-0 buffers before binding. All
    // branches below bind grad_output / indices into descriptors.
    const Tensor grad_output_c = dispatchContiguous(grad_output);
    indices = dispatchContiguous(indices);

    // BFloat16: use native shader
    if (grad_output.dtype() == DType::BFloat16) {
        std::string shader_name = "embedding_backward_bf16";
        auto* pipeline = getPipeline(shader_name, device_id);

        Tensor grad_weight = dispatchZeros({num_embeddings, embedding_dim}, DType::BFloat16, grad_output.device());

        const void* buf_grad_out = grad_output_c.data_ptr();
        const void* buf_indices = indices.data_ptr();
        const void* buf_grad_weight = grad_weight.data_ptr();

        size_t grad_out_pairs = (grad_output.numel() + 1) / 2;
        size_t grad_weight_pairs = (grad_weight.numel() + 1) / 2;
        size_t grad_out_size = grad_out_pairs * 4;
        size_t indices_size = num_indices * sizeof(int32_t);
        size_t grad_weight_size = grad_weight_pairs * 4;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buf_grad_out}, {1, buf_indices}, {2, buf_grad_weight},
        };
        std::vector<size_t> sizes = {grad_out_size, indices_size, grad_weight_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t num_indices;
            uint32_t embedding_dim;
            uint32_t num_embeddings;
        } push_constants;

        push_constants.num_indices = static_cast<uint32_t>(num_indices);
        push_constants.embedding_dim = static_cast<uint32_t>(embedding_dim);
        push_constants.num_embeddings = static_cast<uint32_t>(num_embeddings);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(push_constants), &push_constants);

        uint32_t total_elements = static_cast<uint32_t>(num_indices * embedding_dim);
        uint32_t workgroups = div_wg(total_elements, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return grad_weight;
    }

    // For Float64 with atomic int64 support, use CAS-based GPU shader
    if (grad_output.dtype() == DType::Float64 && devices_[device_id].hasAtomicInt64) {
        std::string shader_name = "embedding_backward_f64_atomic";
        auto* pipeline = getPipeline(shader_name, device_id);

        // Output: grad_weight as uint64_t buffer (for CAS atomics), initialized to zero
        Tensor grad_weight = dispatchZeros({num_embeddings, embedding_dim}, DType::Float64, grad_output.device());

        const void* buf_grad_out = grad_output_c.data_ptr();
        const void* buf_indices = indices.data_ptr();
        const void* buf_grad_weight = grad_weight.data_ptr();

        size_t grad_out_size = grad_output.numel() * sizeof(double);
        size_t indices_size = num_indices * sizeof(int32_t);
        size_t grad_weight_size = num_embeddings * embedding_dim * sizeof(double);

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buf_grad_out}, {1, buf_indices}, {2, buf_grad_weight},
        };
        std::vector<size_t> sizes = {grad_out_size, indices_size, grad_weight_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t num_indices;
            uint32_t embedding_dim;
            uint32_t num_embeddings;
        } push_constants;

        push_constants.num_indices = static_cast<uint32_t>(num_indices);
        push_constants.embedding_dim = static_cast<uint32_t>(embedding_dim);
        push_constants.num_embeddings = static_cast<uint32_t>(num_embeddings);

        uint64_t total_threads = static_cast<uint64_t>(num_indices) * embedding_dim;
        uint32_t workgroups = static_cast<uint32_t>(div_wg(total_threads, devices_[device_id].workgroupSize));

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return grad_weight;
    }

    // Fallback path (reached when the device lacks atomic-int64 support, so the
    // CAS-based "embedding_backward_f64_atomic" shader cannot be used because it
    // hard-requires GL_EXT_shader_atomic_int64). For Float64 use the dedicated
    // non-atomic shader "embedding_backward_f64", which writes a real float64_t
    // grad_weight buffer (matching the Float64 grad_weight allocated below).
    std::string shader_name = (grad_output.dtype() == DType::Float64)
        ? "embedding_backward_f64" : "embedding_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Output: grad_weight of shape [num_embeddings, embedding_dim], initialized to zero
    Tensor grad_weight = dispatchZeros({num_embeddings, embedding_dim}, grad_output.dtype(), grad_output.device());

    size_t elem_size = grad_output.dtype_size();

    const void* buf_grad_out = grad_output_c.data_ptr();
    const void* buf_indices = indices.data_ptr();
    const void* buf_grad_weight = grad_weight.data_ptr();

    size_t grad_out_size = grad_output.numel() * elem_size;
    size_t indices_size = num_indices * sizeof(int32_t);  // shader uses int (32-bit)
    size_t grad_weight_size = num_embeddings * embedding_dim * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_grad_out},
        {1, buf_indices},
        {2, buf_grad_weight},
    };
    std::vector<size_t> sizes = {grad_out_size, indices_size, grad_weight_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t num_indices;
        uint32_t embedding_dim;
        uint32_t num_embeddings;
    } push_constants;

    push_constants.num_indices = static_cast<uint32_t>(num_indices);
    push_constants.embedding_dim = static_cast<uint32_t>(embedding_dim);
    push_constants.num_embeddings = static_cast<uint32_t>(num_embeddings);

    // Total threads = num_indices * embedding_dim (one per element)
    uint64_t total_threads = static_cast<uint64_t>(num_indices) * embedding_dim;
    uint32_t workgroups = static_cast<uint32_t>(div_wg(total_threads, devices_[device_id].workgroupSize));

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_weight;
}

// RMSNorm Forward - GPU implementation
auto VulkanBackend::dispatchRMSNorm(const Tensor& input, const Tensor& weight,
                                     int64_t normalized_shape, float epsilon) -> std::pair<Tensor, Tensor> {
    int32_t device_id = input.device().index;

    // For Float16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16) {
        DType orig_dtype = input.dtype();
        auto in_f32 = input.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto [out_f32, rrms_f32] = dispatchRMSNorm(in_f32, w_f32, normalized_shape, epsilon);
        return {out_f32.to(orig_dtype), rrms_f32};  // rrms stays F32 for backward
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16_rms = (input.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "rms_norm_f64" : is_bfloat16_rms ? "rms_norm_bf16" : "rms_norm";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t batch_size = input.numel() / normalized_shape;

    // Output tensor same shape as input
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    // rrms tensor: one value per batch element (always same dtype as input for F64, F32 otherwise)
    DType rrms_dtype = is_float64 ? DType::Float64 : DType::Float32;
    Tensor rrms({batch_size}, rrms_dtype, input.device());

    size_t elem_size = input.dtype_size();
    size_t rrms_elem_size = rrms.dtype_size();

    // Materialize read operands to packed offset-0 buffers before binding.
    const Tensor input_c = dispatchContiguous(input);
    const Tensor weight_c = dispatchContiguous(weight);

    const void* buf_input = input_c.data_ptr();
    const void* buf_output = output.data_ptr();
    const void* buf_weight = weight_c.data_ptr();
    const void* buf_rrms = rrms.data_ptr();

    size_t input_buf_size = input.numel() * elem_size;
    size_t output_buf_size = output.numel() * elem_size;
    size_t weight_buf_size = normalized_shape * elem_size;
    size_t rrms_buf_size = batch_size * rrms_elem_size;

    // Bindings: input(0), output(1), weight(2), rrms_out(3)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_input},
        {1, buf_output},
        {2, buf_weight},
        {3, buf_rrms},
    };
    std::vector<size_t> sizes = {input_buf_size, output_buf_size, weight_buf_size, rrms_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t normalized_shape;
        float epsilon;
        uint32_t padding;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.normalized_shape = static_cast<uint32_t>(normalized_shape);
    push_constants.epsilon = epsilon;
    push_constants.padding = 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {fp16_saturate_if_needed(*this, output), rrms};
}

// RMSNorm Backward - GPU implementation
auto VulkanBackend::dispatchRMSNormBackward(const Tensor& grad_output, const Tensor& input,
                                              const Tensor& rrms, const Tensor& weight,
                                              int64_t normalized_shape)
                                              -> std::pair<Tensor, Tensor> {
    int32_t device_id = input.device().index;

    // For Float16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16) {
        DType orig_dtype = input.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        auto rrms_f32 = rrms.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto [gi, gw] = dispatchRMSNormBackward(go_f32, in_f32, rrms_f32, w_f32, normalized_shape);
        return {gi.to(orig_dtype), gw.to(orig_dtype)};
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16_rmsb = (input.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "rms_norm_backward_f64" : is_bfloat16_rmsb ? "rms_norm_backward_bf16" : "rms_norm_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t batch_size = input.numel() / normalized_shape;

    // Create output tensors
    Tensor grad_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
    Tensor grad_weight = dispatchZeros({normalized_shape}, input.dtype(), input.device());

    size_t elem_size = input.dtype_size();
    size_t rrms_elem_size = rrms.dtype_size();

    // For Float64: rms_norm_backward_f64.comp is a two-pass (no-atomics) shader
    // mirroring layer_norm_backward_f64. Binding 5 is a PARTIAL grad_weight of
    // size [batch_size * normalized_shape] (one row per batch element), reduced
    // into the final [normalized_shape] grad_weight by reduce_partial_sums_f64.
    Tensor partial_grad_weight_rms;
    if (is_float64) {
        partial_grad_weight_rms = Tensor({batch_size * normalized_shape}, DType::Float64, input.device());
    }

    // Materialize read operands to packed offset-0 buffers before binding.
    const Tensor grad_output_c = dispatchContiguous(grad_output);
    const Tensor input_c = dispatchContiguous(input);
    const Tensor rrms_c = dispatchContiguous(rrms);
    const Tensor weight_c = dispatchContiguous(weight);

    const void* buf_grad_out = grad_output_c.data_ptr();
    const void* buf_input = input_c.data_ptr();
    const void* buf_rrms = rrms_c.data_ptr();
    const void* buf_weight = weight_c.data_ptr();
    const void* buf_grad_input = grad_input.data_ptr();
    const void* buf_grad_weight = grad_weight.data_ptr();

    size_t input_buf_size = input.numel() * elem_size;
    size_t rrms_buf_size = batch_size * rrms_elem_size;
    size_t norm_buf_size = normalized_shape * elem_size;

    // Bindings: grad_output(0), input(1), rrms(2), weight(3), grad_input(4),
    // grad_weight(5). For F64, binding 5 is the partial grad_weight buffer.
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_grad_out},
        {1, buf_input},
        {2, buf_rrms},
        {3, buf_weight},
        {4, buf_grad_input},
    };
    std::vector<size_t> sizes = {input_buf_size, input_buf_size, rrms_buf_size, norm_buf_size,
                                  input_buf_size};

    if (is_float64) {
        size_t partial_buf_size = batch_size * normalized_shape * sizeof(double);
        bindings.push_back({5, partial_grad_weight_rms.data_ptr()});
        sizes.push_back(partial_buf_size);
    } else {
        bindings.push_back({5, buf_grad_weight});
        sizes.push_back(norm_buf_size);
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t normalized_shape;
        uint32_t padding0;
        uint32_t padding1;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.normalized_shape = static_cast<uint32_t>(normalized_shape);
    push_constants.padding0 = 0;
    push_constants.padding1 = 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // Float64: pass 2 — reduce the per-batch-element partial grad_weight rows
    // into the final [normalized_shape] grad_weight. reduce_partial_sums_f64
    // reduces two partial inputs (bindings 0,1) into two outputs (bindings 2,3);
    // RMS has a single grad to reduce, so we bind the partial to BOTH inputs and
    // a throwaway scratch to the second output. The "channels" axis here is the
    // normalized_shape (one column per normalized element), num_partials is the
    // batch_size (rows summed together).
    if (is_float64) {
        auto* reduce_pipeline = getPipeline("reduce_partial_sums_f64", device_id);

        struct ReducePC {
            uint32_t num_partials;
            uint32_t channels;
        } reduce_pc;
        reduce_pc.num_partials = static_cast<uint32_t>(batch_size);
        reduce_pc.channels = static_cast<uint32_t>(normalized_shape);

        // Scratch output for the unused second reduction result. Must not alias
        // the partial input (it is written), so allocate a dedicated buffer.
        Tensor grad_weight_dummy = dispatchZeros({normalized_shape}, DType::Float64, input.device());

        size_t partial_buf_size = batch_size * normalized_shape * sizeof(double);
        std::vector<std::pair<uint32_t, const void*>> reduce_bindings = {
            {0, partial_grad_weight_rms.data_ptr()},
            {1, partial_grad_weight_rms.data_ptr()},
            {2, grad_weight.data_ptr()},
            {3, grad_weight_dummy.data_ptr()}
        };
        std::vector<size_t> reduce_sizes = {
            partial_buf_size, partial_buf_size, norm_buf_size, norm_buf_size
        };

        VkDescriptorSet reduce_ds = allocateAndWriteDescriptorSet(
            device_id, reduce_pipeline, reduce_bindings, reduce_sizes);

        VkCommandBuffer reduce_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(reduce_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reduce_pipeline->pipeline());
        vkCmdBindDescriptorSets(reduce_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               reduce_pipeline->layout(), 0, 1, &reduce_ds, 0, nullptr);
        vkCmdPushConstants(reduce_cmd, reduce_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(ReducePC), &reduce_pc);
        vkCmdDispatch(reduce_cmd, static_cast<uint32_t>(normalized_shape), 1, 1);
        insertComputeOnlyBarrier(reduce_cmd);
        endSingleTimeCommands(reduce_cmd, device_id);
    }

    return {grad_input, grad_weight};
}


} // namespace tenzor
