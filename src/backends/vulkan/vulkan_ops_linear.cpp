#include "vulkan_ops_common.hpp"
#include "tenzor/ops/creation.hpp"  // for tenzor::get_global_seed

namespace tenzor {

// ============================================================================
// Linear/FC Operations
// ============================================================================

auto VulkanBackend::dispatchLinear(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor {
    // Linear: output = input @ weight^T + bias
    // input: (*, K), weight: (N, K) -> output: (*, N)
    //
    // FP16 / BF16 note: both linear_f16.comp and linear_bf16.comp already
    // accumulate their dot products in float32 while keeping FP16/BF16 I/O
    // (see e.g. linear_f16.comp lines 58-96). There is no need to upcast on
    // the host before dispatch — doing so doubles the bandwidth and VRAM
    // footprint for no numerical win. Shader selection happens below via
    // `shader_name` so each dtype lands on its native-I/O variant.

    Tensor input_contig = (input.is_contiguous() && input.offset() == 0) ? input : dispatchContiguous(input);
    Tensor weight_contig = (weight.is_contiguous() && weight.offset() == 0) ? weight : dispatchContiguous(weight);

    auto input_shape = input_contig.shape();
    auto weight_shape = weight_contig.shape();

    if (weight_shape.size() != 2) {
        throw std::invalid_argument("Linear: weight must be 2D");
    }

    int64_t N = weight_shape[0];  // output features
    int64_t K = weight_shape[1];  // input features

    // Flatten input to 2D: (M, K) where M = product of batch dims
    int64_t M = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
        M *= input_shape[i];
    }

    if (input_shape.back() != K) {
        throw std::invalid_argument("Linear: input features (" +
            std::to_string(input_shape.back()) + ") != weight features (" +
            std::to_string(K) + ")");
    }

    int32_t device_id = input_contig.device().index;

    // Select shader by dtype
    bool is_float64 = (input_contig.dtype() == DType::Float64);
    bool is_float16_lin = (input_contig.dtype() == DType::Float16);
    bool is_bfloat16_lin = (input_contig.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "linear_f64" : is_float16_lin ? "linear_f16" : is_bfloat16_lin ? "linear_bf16" : "linear";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Output shape: (batch_dims..., N)
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end() - 1);
    out_shape.push_back(N);
    Tensor output(out_shape, input_contig.dtype(), input_contig.device());

    // Create a dummy bias buffer if no bias (we still need 4 bindings)
    Tensor dummy_bias;
    const void* bias_ptr_data;
    size_t bias_size;
    if (bias) {
        Tensor bias_contig = (bias->is_contiguous() && bias->offset() == 0) ? *bias : dispatchContiguous(*bias);
        bias_ptr_data = bias_contig.data_ptr();
        bias_size = bias_contig.numel() * bias_contig.dtype_size();
    } else {
        // Create a minimal buffer for the unused binding
        dummy_bias = Tensor({1}, input_contig.dtype(), input_contig.device());
        bias_ptr_data = dummy_bias.data_ptr();
        bias_size = dummy_bias.dtype_size();
    }

    size_t input_size = input_contig.numel() * input_contig.dtype_size();
    size_t weight_size = weight_contig.numel() * weight_contig.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_contig.data_ptr()},
        {1, weight_contig.data_ptr()},
        {2, bias_ptr_data},
        {3, output.data_ptr()}
    };
    std::vector<size_t> sizes = {input_size, weight_size, bias_size, output_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint32_t has_bias;
    } push_constants;

    push_constants.M = static_cast<uint32_t>(M);
    push_constants.N = static_cast<uint32_t>(N);
    push_constants.K = static_cast<uint32_t>(K);
    push_constants.has_bias = bias ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups_x = (push_constants.N + 15) / 16;
    uint32_t workgroups_y = (push_constants.M + 15) / 16;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return fp16_saturate_if_needed(*this, output);
}

auto VulkanBackend::dispatchLinearBackward(const Tensor& grad_output, const Tensor& input, const Tensor& weight) -> std::vector<Tensor> {
    // grad_input  = grad_output @ weight       (M,N) x (N,K) -> (M,K)
    // grad_weight = grad_output^T @ input      (N,M) x (M,K) -> (N,K)
    // grad_bias   = sum(grad_output, dim=0)    (M,N) -> (N,)

    Tensor go_contig = (grad_output.is_contiguous() && grad_output.offset() == 0) ? grad_output : dispatchContiguous(grad_output);
    Tensor in_contig = (input.is_contiguous() && input.offset() == 0) ? input : dispatchContiguous(input);
    Tensor w_contig = (weight.is_contiguous() && weight.offset() == 0) ? weight : dispatchContiguous(weight);

    auto go_shape = go_contig.shape();
    auto w_shape = w_contig.shape();
    std::vector<int64_t> in_shape(in_contig.shape().begin(), in_contig.shape().end());

    int64_t N = w_shape[0];
    int64_t K = w_shape[1];
    int64_t M = 1;
    for (size_t i = 0; i < go_shape.size() - 1; ++i) {
        M *= go_shape[i];
    }

    int32_t device_id = go_contig.device().index;

    // 1. Compute grad_input using dedicated shader (F16/BF16 use native shader with F32 accumulation)
    bool is_float64 = (go_contig.dtype() == DType::Float64);
    bool is_float16 = (go_contig.dtype() == DType::Float16);
    bool is_bfloat16 = (go_contig.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "linear_backward_f64" : is_float16 ? "linear_backward_f16" : is_bfloat16 ? "linear_backward_bf16" : "linear_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    Tensor grad_input(in_shape, go_contig.dtype(), go_contig.device());

    // Buffer sizes (round up to 4-byte boundary for F16 packed uint32 access)
    size_t go_size, w_size, gi_size;
    if (is_float16 || is_bfloat16) {
        go_size = ((go_contig.numel() + 1) / 2) * 4;
        w_size = ((w_contig.numel() + 1) / 2) * 4;
        gi_size = ((grad_input.numel() + 1) / 2) * 4;
    } else {
        go_size = go_contig.numel() * go_contig.dtype_size();
        w_size = w_contig.numel() * w_contig.dtype_size();
        gi_size = grad_input.numel() * grad_input.dtype_size();
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, go_contig.data_ptr()},
        {1, w_contig.data_ptr()},
        {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {go_size, w_size, gi_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t M;
        uint32_t N;
        uint32_t K;
    } push_constants;

    push_constants.M = static_cast<uint32_t>(M);
    push_constants.N = static_cast<uint32_t>(N);
    push_constants.K = static_cast<uint32_t>(K);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups_x = (push_constants.K + 15) / 16;
    uint32_t workgroups_y = (push_constants.M + 15) / 16;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // 2. grad_weight = grad_output^T @ input  via matmul
    // Reshape to 2D for matmul: go (M,N) -> transpose -> (N,M), in (M,K) -> gw (N,K)
    Tensor go_2d = go_contig.reshape({M, N});
    Tensor in_2d = in_contig.reshape({M, K});
    Tensor go_t = dispatchTranspose(go_2d, 0, 1);  // (N, M)
    Tensor go_t_contig = (go_t.is_contiguous() && go_t.offset() == 0) ? go_t : dispatchContiguous(go_t);
    Tensor grad_weight = dispatchMatmul(go_t_contig, in_2d);  // (N, K)

    // 3. grad_bias = sum(grad_output, dim=0) -> (N,)
    Tensor go_for_sum = go_contig.reshape({M, N});
    Tensor grad_bias = dispatchReduction("sum", go_for_sum, 0, false);

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// Dropout Operations
// ============================================================================

auto VulkanBackend::dispatchDropout(const Tensor& input, float p, bool training) -> std::pair<Tensor, Tensor> {
    // If not training or p == 0, return input unchanged with all-ones mask
    if (!training || p == 0.0f) {
        // Allocate the ones mask on the input's device (dispatchFull hardcodes
        // Vulkan device 0, which would straddle devices on a multi-GPU setup;
        // the p>=1 branch below already passes input.device() via dispatchZeros).
        Tensor mask = tenzor::full(
            std::vector<int64_t>(input.shape().begin(), input.shape().end()),
            1.0, input.dtype(), input.device());
        return {input, mask};
    }

    // p == 1 means drop everything
    if (p >= 1.0f) {
        Tensor output = dispatchZeros(
            std::vector<int64_t>(input.shape().begin(), input.shape().end()),
            input.dtype(), input.device());
        Tensor mask = dispatchZeros(
            std::vector<int64_t>(input.shape().begin(), input.shape().end()),
            input.dtype(), input.device());
        return {output, mask};
    }

    Tensor input_contig = (input.is_contiguous() && input.offset() == 0) ? input : dispatchContiguous(input);

    size_t numel = input_contig.numel();
    if (numel == 0) {
        Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
        Tensor mask(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                    input.dtype(), input.device());
        return {output, mask};
    }

    int32_t device_id = input_contig.device().index;

    bool is_float64 = (input_contig.dtype() == DType::Float64);
    bool is_float16 = (input_contig.dtype() == DType::Float16);
    bool is_bfloat16 = (input_contig.dtype() == DType::BFloat16);

    std::string shader_name = "dropout";
    if (is_float64) shader_name = "dropout_f64";
    else if (is_float16) shader_name = "dropout_f16";
    else if (is_bfloat16) shader_name = "dropout_bf16";

    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor output(shape_vec, input_contig.dtype(), input_contig.device());
    Tensor mask(shape_vec, input_contig.dtype(), input_contig.device());

    size_t elem_size = input_contig.dtype_size();
    size_t input_buf_size = numel * elem_size;
    size_t output_buf_size = numel * elem_size;
    size_t mask_buf_size = numel * elem_size;

    // For Float16/BFloat16, round up buffer sizes to 4-byte boundaries
    if (is_float16 || is_bfloat16) {
        size_t num_pairs = (numel + 1) / 2;
        input_buf_size = num_pairs * 4;
        output_buf_size = num_pairs * 4;
        mask_buf_size = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_contig.data_ptr()},
        {1, output.data_ptr()},
        {2, mask.data_ptr()}
    };
    std::vector<size_t> sizes = {input_buf_size, output_buf_size, mask_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Honor `tenzor::manual_seed`; falls back to time-based when unset. The
    // Philox offset is derived deterministically from the seed (which already
    // increments per call via get_global_seed()), NOT from a process-global
    // static counter — a static counter past manual_seed() with no reset hook
    // made `manual_seed(s); dropout(...)` non-reproducible. offset==0 makes the
    // sequence fully determined by the seed, matching the manual_seed contract
    // and the other backends (see vulkan_ops_misc.cpp rand path).
    float scale = 1.0f / (1.0f - p);

    struct PushConstants {
        uint32_t n_elements;
        uint32_t seed_lo;
        uint32_t seed_hi;
        uint32_t offset;
        float p;
        float scale;
    } push_constants;

    // Thread the full 64-bit global seed as two 32-bit words (lo/hi) into the
    // Philox key. Previously only the low 32 bits were passed, dropping the high
    // half of the seed and weakly seeding the counter.
    uint64_t global_seed = static_cast<uint64_t>(::tenzor::get_global_seed());
    push_constants.n_elements = static_cast<uint32_t>(numel);
    push_constants.seed_lo = static_cast<uint32_t>(global_seed & 0xFFFFFFFFull);
    push_constants.seed_hi = static_cast<uint32_t>(global_seed >> 32);
    push_constants.offset = 0;
    push_constants.p = p;
    push_constants.scale = scale;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups;
    if (is_float16 || is_bfloat16) {
        uint32_t num_pairs = (static_cast<uint32_t>(numel) + 1) / 2;
        workgroups = div_wg_checked(num_pairs, devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch");
    } else {
        workgroups = div_wg_checked(numel, devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch");
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, mask};
}

auto VulkanBackend::dispatchDropoutBackward(const Tensor& grad_output, const Tensor& mask, float p) -> Tensor {
    // grad_input = grad_output * mask * scale
    if (p == 0.0f) {
        return grad_output;
    }
    // p==1.0 drops every element (forward's mask is 0 everywhere, output is a
    // clean zero tensor). scale=1/(1-p) is +inf at p=1.0, so mask(0)*scale(inf)
    // is NaN/inf under IEEE-754 float arithmetic, not 0 -- unlike forward's
    // direct zero write, this would poison every gradient element instead of
    // the mathematically correct zero. Short-circuit before the shader ever
    // computes with scale.
    if (p >= 1.0f) {
        return dispatchFull(std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end()),
                            0.0, grad_output.dtype(), grad_output.device());
    }

    Tensor go_contig = (grad_output.is_contiguous() && grad_output.offset() == 0) ? grad_output : dispatchContiguous(grad_output);
    Tensor mask_contig = (mask.is_contiguous() && mask.offset() == 0) ? mask : dispatchContiguous(mask);

    size_t numel = go_contig.numel();
    if (numel == 0) {
        return Tensor(std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end()),
                      grad_output.dtype(), grad_output.device());
    }

    int32_t device_id = go_contig.device().index;

    bool is_float64 = (go_contig.dtype() == DType::Float64);
    bool is_float16 = (go_contig.dtype() == DType::Float16);
    bool is_bfloat16 = (go_contig.dtype() == DType::BFloat16);

    std::string shader_name = "dropout_backward";
    if (is_float64) shader_name = "dropout_backward_f64";
    else if (is_float16) shader_name = "dropout_backward_f16";
    else if (is_bfloat16) shader_name = "dropout_backward_bf16";

    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape_vec = std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end());
    Tensor grad_input(shape_vec, go_contig.dtype(), go_contig.device());

    size_t elem_size = go_contig.dtype_size();
    size_t go_buf_size = numel * elem_size;
    size_t mask_buf_size = numel * elem_size;
    size_t gi_buf_size = numel * elem_size;

    if (is_float16 || is_bfloat16) {
        size_t num_pairs = (numel + 1) / 2;
        go_buf_size = num_pairs * 4;
        mask_buf_size = num_pairs * 4;
        gi_buf_size = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, go_contig.data_ptr()},
        {1, mask_contig.data_ptr()},
        {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {go_buf_size, mask_buf_size, gi_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    float scale = 1.0f / (1.0f - p);

    struct PushConstants {
        uint32_t n_elements;
        float scale;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(numel);
    push_constants.scale = scale;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups;
    if (is_float16 || is_bfloat16) {
        uint32_t num_pairs = (static_cast<uint32_t>(numel) + 1) / 2;
        workgroups = div_wg_checked(num_pairs, devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch");
    } else {
        workgroups = div_wg_checked(numel, devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch");
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}


} // namespace tenzor
