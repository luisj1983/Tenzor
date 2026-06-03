/**
 * @file vulkan_ops_embed.cpp
 * @brief Vulkan backend InstanceNorm and EmbeddingBag operations
 */

#include "vulkan_ops_common.hpp"

namespace tenzor {

// ============================================================================
// InstanceNorm Forward (OpId::InstanceNorm)
// ============================================================================

auto VulkanBackend::dispatchInstanceNorm(const Tensor& input, const Tensor& weight,
                                          const Tensor& bias, float epsilon)
    -> std::vector<Tensor> {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // Require 4D input (N, C, H, W)
    if (input.ndim() < 3) {
        throw std::invalid_argument("InstanceNorm requires at least 3D input, got " +
                                    std::to_string(input.ndim()) + "D");
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "instance_norm_f64" : is_float16 ? "instance_norm_f16" : is_bfloat16 ? "instance_norm_bf16" : "instance_norm";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t spatial_size = 1;
    for (size_t d = 2; d < input_shape.size(); ++d) {
        spatial_size *= input_shape[d];
    }
    int64_t total_nc = N * C;

    bool has_affine = (weight.numel() > 0 && bias.numel() > 0);

    // The _f16/_bf16 shader declares weight/bias as Float32 buffers. Always convert the
    // weight/bias to Float32 before dispatch so the binding stride matches the shader.
    Tensor weight_f32 = weight;
    Tensor bias_f32 = bias;
    if ((is_float16 || is_bfloat16) && has_affine) {
        if (weight_f32.dtype() != DType::Float32) weight_f32 = weight_f32.to(DType::Float32);
        if (bias_f32.dtype() != DType::Float32) bias_f32 = bias_f32.to(DType::Float32);
    }

    // Output tensors — mean/rstd are always Float32 for F16 shader
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());
    DType stats_dtype = (is_float16 || is_bfloat16) ? DType::Float32 : input.dtype();
    Tensor mean_out({total_nc}, stats_dtype, input.device());
    Tensor rstd_out({total_nc}, stats_dtype, input.device());

    size_t elem_size = input.dtype_size();
    // For F16/BF16: packed uint32 words (4-byte aligned); stats and weight/bias are Float32.
    const bool is_packed_half = (is_float16 || is_bfloat16);
    size_t input_buf_size = is_packed_half ? ((input.numel() + 1) / 2) * 4 : input.numel() * elem_size;
    size_t output_buf_size = is_packed_half ? ((output.numel() + 1) / 2) * 4 : output.numel() * elem_size;
    size_t stats_elem_size = is_packed_half ? sizeof(float) : elem_size;
    size_t nc_buf_size = total_nc * stats_elem_size;
    // Weight/bias are Float32 for the F16/BF16 shader
    size_t channel_buf_size = C * (is_packed_half ? sizeof(float) : elem_size);

    // Build bindings: input(0), output(1), weight(2), bias(3), mean(4), var(5)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()},
        {1, output.data_ptr()}
    };
    std::vector<size_t> sizes = {input_buf_size, output_buf_size};

    if (has_affine) {
        bindings.push_back({2, weight_f32.data_ptr()});
        bindings.push_back({3, bias_f32.data_ptr()});
        sizes.push_back(channel_buf_size);
        sizes.push_back(channel_buf_size);
    } else {
        bindings.push_back({2, output.data_ptr()});
        bindings.push_back({3, output.data_ptr()});
        sizes.push_back(output_buf_size);
        sizes.push_back(output_buf_size);
    }

    bindings.push_back({4, mean_out.data_ptr()});
    bindings.push_back({5, rstd_out.data_ptr()});
    sizes.push_back(nc_buf_size);
    sizes.push_back(nc_buf_size);

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t channels;
        uint32_t spatial_size;
        float epsilon;
        uint32_t affine;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(N);
    push_constants.channels = static_cast<uint32_t>(C);
    push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
    push_constants.epsilon = epsilon;
    push_constants.affine = has_affine ? 1u : 0u;

    // One workgroup per (N, C) pair
    uint32_t workgroups = static_cast<uint32_t>(total_nc);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {fp16_saturate_if_needed(*this, output), mean_out, rstd_out};
}

// ============================================================================
// InstanceNorm Backward (OpId::InstanceNormBackward)
// ============================================================================

auto VulkanBackend::dispatchInstanceNormBackward(const Tensor& grad_output, const Tensor& input,
                                                  const Tensor& mean, const Tensor& rstd,
                                                  const Tensor& weight)
    -> std::tuple<Tensor, Tensor, Tensor> {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "instance_norm_backward_f64" : is_float16 ? "instance_norm_backward_f16" : is_bfloat16 ? "instance_norm_backward_bf16" : "instance_norm_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t spatial_size = 1;
    for (size_t d = 2; d < input_shape.size(); ++d) {
        spatial_size *= input_shape[d];
    }
    int64_t total_nc = N * C;

    bool has_affine = (weight.numel() > 0);

    // Align weight dtype with shader expectation: F16/BF16 backward shader reads
    // weight as Float32. Convert if needed so binding strides match.
    Tensor weight_f32 = weight;
    if ((is_float16 || is_bfloat16) && has_affine && weight_f32.dtype() != DType::Float32) {
        weight_f32 = weight_f32.to(DType::Float32);
    }

    size_t elem_size = input.dtype_size();

    // Allocate outputs — grad_weight/grad_bias always Float32 for F16 shader
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(out_shape, input.dtype(), input.device());
    const bool is_packed_half = (is_float16 || is_bfloat16);
    DType accum_dtype = is_packed_half ? DType::Float32 : input.dtype();
    size_t accum_elem_size = is_packed_half ? sizeof(float) : elem_size;
    Tensor grad_weight({C}, accum_dtype, input.device());
    Tensor grad_bias({C}, accum_dtype, input.device());

    // Zero-init grad_weight and grad_bias for atomic accumulation
    memset(grad_weight.data_ptr(), 0, C * accum_elem_size, device_id);
    memset(grad_bias.data_ptr(), 0, C * accum_elem_size, device_id);

    // For F16/BF16: packed uint32 words
    size_t input_buf_size = is_packed_half ? ((input.numel() + 1) / 2) * 4 : input.numel() * elem_size;
    // mean/rstd are Float32 for F16/BF16
    size_t nc_buf_size = total_nc * (is_packed_half ? sizeof(float) : elem_size);
    size_t channel_buf_size = C * accum_elem_size;

    // Bindings: grad_output(0), input(1), mean(2), rstd(3), weight(4),
    //           grad_input(5), grad_weight(6), grad_bias(7)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()},
        {1, input.data_ptr()},
        {2, mean.data_ptr()},
        {3, rstd.data_ptr()},
    };
    std::vector<size_t> sizes = {input_buf_size, input_buf_size, nc_buf_size, nc_buf_size};

    if (has_affine) {
        bindings.push_back({4, weight_f32.data_ptr()});
        sizes.push_back(channel_buf_size);
    } else {
        bindings.push_back({4, grad_input.data_ptr()});
        sizes.push_back(input_buf_size);
    }

    bindings.push_back({5, grad_input.data_ptr()});
    bindings.push_back({6, grad_weight.data_ptr()});
    bindings.push_back({7, grad_bias.data_ptr()});
    sizes.push_back(input_buf_size);
    sizes.push_back(channel_buf_size);
    sizes.push_back(channel_buf_size);

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t channels;
        uint32_t spatial_size;
        uint32_t affine;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(N);
    push_constants.channels = static_cast<uint32_t>(C);
    push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
    push_constants.affine = has_affine ? 1u : 0u;

    uint32_t workgroups = static_cast<uint32_t>(total_nc);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// EmbeddingBag Forward (OpId::EmbeddingBagForward, 434)
// ============================================================================

auto VulkanBackend::dispatchEmbeddingBag(const Tensor& embeddings, const Tensor& offsets,
                                          int64_t embedding_dim, const std::string& mode,
                                          bool include_last_offset) -> std::vector<Tensor> {
    int32_t device_id = embeddings.device().index;

    // Select shader based on dtype
    std::string shader_name = "embedding_bag";
    if (embeddings.dtype() == DType::Float64) {
        shader_name = "embedding_bag_f64";
    } else if (embeddings.dtype() == DType::Float16) {
        shader_name = "embedding_bag_f16";
    } else if (embeddings.dtype() == DType::BFloat16) {
        shader_name = "embedding_bag_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t total_rows = embeddings.shape()[0];
    int64_t num_offsets_raw = offsets.numel();
    int64_t num_bags = include_last_offset ? (num_offsets_raw - 1) : num_offsets_raw;

    // Convert mode string to int
    uint32_t mode_int = 0;  // sum
    if (mode == "mean") mode_int = 1;
    else if (mode == "max") mode_int = 2;

    if (num_bags <= 0) {
        return {Tensor({0, embedding_dim}, embeddings.dtype(), embeddings.device()),
                Tensor({0}, DType::Int64, embeddings.device())};
    }

    // Offsets must be Int32 for the shader
    Tensor offsets_i32 = offsets;
    if (offsets.dtype() == DType::Int64) {
        std::vector<int64_t> offs_shape(offsets.shape().begin(), offsets.shape().end());
        offsets_i32 = Tensor(offs_shape, DType::Int32, offsets.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = offsets.data_ptr();
        const void* buf_out = offsets_i32.data_ptr();
        size_t size_in = offsets.numel() * sizeof(int64_t);
        size_t size_out = offsets_i32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(offsets.numel());

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, div_wg(offsets.numel(), devices_[device_id].workgroupSize), 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    Tensor output({num_bags, embedding_dim}, embeddings.dtype(), embeddings.device());

    // Always bound at binding 3 (the shader declares it for every mode). Written
    // with the per-(bag,feature) global argmax element index for max mode; for
    // sum/mean it is left undefined and discarded below.
    Tensor max_indices({num_bags, embedding_dim}, DType::Int64, embeddings.device());

    size_t elem_size = embeddings.dtype_size();
    size_t emb_buf_size = embeddings.numel() * elem_size;
    size_t offs_buf_size = offsets_i32.numel() * sizeof(int32_t);
    size_t out_buf_size = output.numel() * elem_size;
    size_t idx_buf_size = max_indices.numel() * sizeof(int64_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, embeddings.data_ptr()},
        {1, offsets_i32.data_ptr()},
        {2, output.data_ptr()},
        {3, max_indices.data_ptr()},
    };
    std::vector<size_t> sizes = {emb_buf_size, offs_buf_size, out_buf_size, idx_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t num_bags;
        uint32_t embedding_dim;
        uint32_t total_rows;
        uint32_t mode;
        uint32_t include_last_offset;
        uint32_t num_offsets;
    } push_constants;

    push_constants.num_bags = static_cast<uint32_t>(num_bags);
    push_constants.embedding_dim = static_cast<uint32_t>(embedding_dim);
    push_constants.total_rows = static_cast<uint32_t>(total_rows);
    push_constants.mode = mode_int;
    push_constants.include_last_offset = include_last_offset ? 1u : 0u;
    push_constants.num_offsets = static_cast<uint32_t>(num_offsets_raw);

    uint32_t total_output = static_cast<uint32_t>(num_bags * embedding_dim);
    uint32_t workgroups = div_wg(total_output, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    if (mode_int == 2) {
        return {output, max_indices};
    }
    return {output, Tensor({0}, DType::Int64, embeddings.device())};
}

// ============================================================================
// EmbeddingBag Backward (OpId::EmbeddingBagBackward, 435)
// ============================================================================

auto VulkanBackend::dispatchEmbeddingBagBackward(const Tensor& grad_output,
                                                   const Tensor& indices,
                                                   const Tensor& offsets,
                                                   int64_t num_embeddings,
                                                   int64_t embedding_dim,
                                                   const std::string& mode,
                                                   bool include_last_offset) -> Tensor {
    int32_t device_id = grad_output.device().index;

    // For Float16, accumulate in Float32 and convert back
    bool is_f16 = (grad_output.dtype() == DType::Float16);
    bool is_bf16 = (grad_output.dtype() == DType::BFloat16);
    DType weight_dtype = is_f16 ? DType::Float32 : grad_output.dtype();

    // Select shader based on dtype
    std::string shader_name = "embedding_bag_backward";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "embedding_bag_backward_f64";
    } else if (is_f16) {
        shader_name = "embedding_bag_backward_f16";
    } else if (is_bf16) {
        shader_name = "embedding_bag_backward_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t num_offsets_raw = offsets.numel();
    int64_t num_bags = include_last_offset ? (num_offsets_raw - 1) : num_offsets_raw;

    if (num_bags <= 0) {
        return Tensor({num_embeddings, embedding_dim}, grad_output.dtype(), grad_output.device());
    }

    // Convert mode string to int
    uint32_t mode_int = 0;  // sum
    if (mode == "mean") mode_int = 1;
    else if (mode == "max") mode_int = 2;

    // Indices must be Int32 for the shader
    Tensor indices_i32 = indices;
    if (indices.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(indices.shape().begin(), indices.shape().end());
        indices_i32 = Tensor(idx_shape, DType::Int32, indices.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = indices.data_ptr();
        const void* buf_out = indices_i32.data_ptr();
        size_t size_in = indices.numel() * sizeof(int64_t);
        size_t size_out = indices_i32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(indices.numel());

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, div_wg(indices.numel(), devices_[device_id].workgroupSize), 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Offsets must be Int32 for the shader
    Tensor offsets_i32 = offsets;
    if (offsets.dtype() == DType::Int64) {
        std::vector<int64_t> offs_shape(offsets.shape().begin(), offsets.shape().end());
        offsets_i32 = Tensor(offs_shape, DType::Int32, offsets.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = offsets.data_ptr();
        const void* buf_out = offsets_i32.data_ptr();
        size_t size_in = offsets.numel() * sizeof(int64_t);
        size_t size_out = offsets_i32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(offsets.numel());

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, div_wg(offsets.numel(), devices_[device_id].workgroupSize), 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Create zero-initialized grad_weight tensor
    Tensor grad_weight({num_embeddings, embedding_dim}, weight_dtype, grad_output.device());
    grad_weight = dispatchFill(grad_weight, 0.0f);

    size_t go_elem_size = grad_output.dtype_size();
    size_t gw_elem_size = grad_weight.dtype_size();
    size_t go_buf_size = grad_output.numel() * go_elem_size;
    size_t idx_buf_size = indices_i32.numel() * sizeof(int32_t);
    size_t offs_buf_size = offsets_i32.numel() * sizeof(int32_t);
    size_t gw_buf_size = grad_weight.numel() * gw_elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()},
        {1, indices_i32.data_ptr()},
        {2, offsets_i32.data_ptr()},
        {3, grad_weight.data_ptr()},
    };
    std::vector<size_t> sizes = {go_buf_size, idx_buf_size, offs_buf_size, gw_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct EmbBagBwdPC {
        uint32_t num_bags;
        uint32_t embedding_dim;
        uint32_t num_embeddings;
        uint32_t mode;
        uint32_t include_last_offset;
        uint32_t num_offsets;
    } push_constants;

    push_constants.num_bags = static_cast<uint32_t>(num_bags);
    push_constants.embedding_dim = static_cast<uint32_t>(embedding_dim);
    push_constants.num_embeddings = static_cast<uint32_t>(num_embeddings);
    push_constants.mode = mode_int;
    push_constants.include_last_offset = include_last_offset ? 1u : 0u;
    push_constants.num_offsets = static_cast<uint32_t>(num_offsets_raw);

    uint32_t total_output = static_cast<uint32_t>(num_bags * embedding_dim);
    uint32_t workgroups = div_wg(total_output, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(EmbBagBwdPC), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // For Float16, convert accumulated Float32 result back
    if (is_f16) {
        return grad_weight.to(DType::Float16);
    }

    return grad_weight;
}

} // namespace tenzor
