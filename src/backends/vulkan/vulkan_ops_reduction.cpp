#include "vulkan_ops_common.hpp"

namespace tenzor {

// Advanced reduction operations implementation
auto VulkanBackend::dispatchArgmax(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Compute output shape first
    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    if (dim < 0) {
        out_shape = {1};
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        return Tensor(out_shape, DType::Int64, input.device());
    }

    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Int32) shader_name = "argmax_argmin_i32";
    else if (input.dtype() == DType::Float64) shader_name = "argmax_argmin_f64";
    else if (input.dtype() == DType::Float16) shader_name = "argmax_argmin_f16";
    else shader_name = "argmax_argmin";
    auto* pipeline = getPipeline(shader_name, device_id);

    Tensor output(out_shape, DType::Int64, input.device());  // Use Int64 for consistency with other backends

    // Get VkBuffer handles from tensor data pointers
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16) {
        // Round up input buffer to 4-byte boundary for uint32 shader access
        size_t in_pairs = (input.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate inner_size (product of dimensions after reduction dimension)
    uint32_t inner_size = 1;
    if (dim >= 0) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t n;
        uint32_t reduce_size;
        uint32_t outer_size;
        uint32_t inner_size;
        uint32_t op;
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.reduce_size = (dim >= 0) ? static_cast<uint32_t>(input_shape[dim]) : pushConstants.n;
    pushConstants.outer_size = (dim >= 0) ? static_cast<uint32_t>(output.numel()) : 1;
    pushConstants.inner_size = inner_size;
    pushConstants.op = 0; // 0 = argmax

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per output element
    uint32_t workgroups = pushConstants.outer_size;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchArgmin(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Compute output shape first
    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    if (dim < 0) {
        out_shape = {1};
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        return Tensor(out_shape, DType::Int64, input.device());
    }

    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Int32) shader_name = "argmax_argmin_i32";
    else if (input.dtype() == DType::Float64) shader_name = "argmax_argmin_f64";
    else if (input.dtype() == DType::Float16) shader_name = "argmax_argmin_f16";
    else shader_name = "argmax_argmin";
    auto* pipeline = getPipeline(shader_name, device_id);

    Tensor output(out_shape, DType::Int64, input.device());  // Use Int64 for consistency with other backends

    // Get VkBuffer handles from tensor data pointers
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16) {
        // Round up input buffer to 4-byte boundary for uint32 shader access
        size_t in_pairs = (input.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate inner_size (product of dimensions after reduction dimension)
    uint32_t inner_size = 1;
    if (dim >= 0) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t n;
        uint32_t reduce_size;
        uint32_t outer_size;
        uint32_t inner_size;
        uint32_t op;
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.reduce_size = (dim >= 0) ? static_cast<uint32_t>(input_shape[dim]) : pushConstants.n;
    pushConstants.outer_size = (dim >= 0) ? static_cast<uint32_t>(output.numel()) : 1;
    pushConstants.inner_size = inner_size;
    pushConstants.op = 1; // 1 = argmin

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per output element
    uint32_t workgroups = pushConstants.outer_size;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchVariance(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor {
    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    // Check if this is a full reduction (dim < 0 means reduce all elements)
    bool full_reduction = (dim < 0);

    // For dispatchReduction, use INT64_MIN to signal full reduction
    int64_t reduction_dim = full_reduction ? INT64_MIN : dim;

    if (full_reduction) {
        // Full reduction: output is scalar (empty shape) or [1,1,...] if keepdim
        if (keepdim) {
            out_shape.assign(input_shape.size(), 1);
        } else {
            out_shape = {};  // Scalar
        }
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    // Compute variance using the formula: Var(X) = E[(X - mean)^2]
    // 1. Compute mean (with keepdim=true for broadcasting)
    Tensor mean = dispatchReduction("mean", input, reduction_dim, true);

    // 2. Compute (X - mean)
    Tensor diff = dispatchBinaryOp("sub", input, mean);

    // 3. Square the differences
    Tensor squared_diff = dispatchBinaryOp("mul", diff, diff);

    // 4. Compute sum of squared differences
    Tensor sum_squared = dispatchReduction("sum", squared_diff, reduction_dim, keepdim);

    // 5. Divide by N or N-1
    uint32_t reduce_size = full_reduction ? static_cast<uint32_t>(input.numel()) : static_cast<uint32_t>(input_shape[dim]);
    double divisor = unbiased ? static_cast<double>(reduce_size - 1) : static_cast<double>(reduce_size);

    // Create a scalar tensor with the divisor matching the output shape
    // This ensures broadcasting preserves the correct output shape
    auto sum_shape = sum_squared.shape();
    std::vector<int64_t> divisor_shape(sum_shape.begin(), sum_shape.end());
    if (divisor_shape.empty()) {
        divisor_shape = {1};
    }
    Tensor divisor_tensor = dispatchFull(divisor_shape, static_cast<float>(divisor), input.dtype());

    // Divide variance by divisor and reshape to match expected output
    Tensor result = dispatchBinaryOp("div", sum_squared, divisor_tensor);

    // If the output should be a scalar but broadcast made it [1], reshape to scalar
    if (full_reduction && !keepdim && result.shape().size() > 0) {
        return result.reshape({});
    }
    return result;
}

auto VulkanBackend::dispatchStd(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor {
    // Standard deviation is just sqrt of variance
    Tensor variance = dispatchVariance(input, dim, unbiased, keepdim);
    return dispatchUnaryOp("sqrt", variance);
}

auto VulkanBackend::dispatchNorm(const Tensor& input, float p, int64_t dim, bool keepdim) -> Tensor {
    // Compute p-norm: (sum(|x|^p))^(1/p)
    // For p=2 (L2 norm): sqrt(sum(x^2))
    // For p=1 (L1 norm): sum(|x|)

    Tensor abs_input = dispatchUnaryOp("abs", input);

    if (p == 1.0f) {
        // L1 norm: sum of absolute values
        return dispatchReduction("sum", abs_input, dim, keepdim);
    } else if (p == 2.0f) {
        // L2 norm: sqrt(sum(x^2))
        Tensor squared = dispatchBinaryOp("mul", abs_input, abs_input);
        Tensor sum = dispatchReduction("sum", squared, dim, keepdim);
        return dispatchUnaryOp("sqrt", sum);
    } else {
        // General p-norm: (sum(|x|^p))^(1/p)
        // Create a tensor filled with p for the power operation
        std::vector<int64_t> scalar_shape = {1};
        Tensor p_tensor(scalar_shape, input.dtype(), input.device());
        float* p_data = static_cast<float*>(p_tensor.data_ptr());
        *p_data = p;

        Tensor powered = dispatchBinaryOp("pow", abs_input, p_tensor);
        Tensor sum = dispatchReduction("sum", powered, dim, keepdim);

        // Compute 1/p root
        Tensor inv_p_tensor(scalar_shape, input.dtype(), input.device());
        float* inv_p_data = static_cast<float*>(inv_p_tensor.data_ptr());
        *inv_p_data = 1.0f / p;

        return dispatchBinaryOp("pow", sum, inv_p_tensor);
    }
}

auto VulkanBackend::dispatchProd(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;

    // Int64 has no native prod shader; convert to Float64, compute, convert back
    DType orig_dtype = input.dtype();
    Tensor prod_input = input;
    if (orig_dtype == DType::Int64) {
        prod_input = input.to(DType::Float64);
    }

    // Select correct pipeline based on dtype
    std::string shader_name;
    if (prod_input.dtype() == DType::Int32) {
        shader_name = "prod_reduction_i32";
    } else if (prod_input.dtype() == DType::Float64) {
        shader_name = "prod_reduction_f64";
    } else {
        shader_name = "prod_reduction";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    // Check if this is a full reduction (dim < 0 means reduce all elements)
    bool full_reduction = (dim < 0);

    if (full_reduction) {
        // Full reduction: output is scalar (empty shape) or [1,1,...] if keepdim
        if (keepdim) {
            out_shape.assign(input_shape.size(), 1);
        } else {
            out_shape = {};  // Scalar - but we need at least 1 element for the output
        }
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    // For full reduction to scalar, we still need buffer space, so use [1] internally
    std::vector<int64_t> buffer_shape = out_shape.empty() ? std::vector<int64_t>{1} : out_shape;
    Tensor output(buffer_shape, prod_input.dtype(), prod_input.device());

    // Get VkBuffer handles from tensor data pointers
    const void* buffer_in = prod_input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = prod_input.numel() * prod_input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate inner_size (product of dimensions after reduction dimension)
    uint32_t inner_size = 1;
    if (dim >= 0) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t n;
        uint32_t reduce_size;
        uint32_t outer_size;
        uint32_t inner_size;
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(prod_input.numel());
    pushConstants.reduce_size = (dim >= 0) ? static_cast<uint32_t>(input_shape[dim]) : pushConstants.n;
    pushConstants.outer_size = (dim >= 0) ? static_cast<uint32_t>(output.numel()) : 1;
    pushConstants.inner_size = inner_size;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per output element
    uint32_t workgroups = pushConstants.outer_size;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    // Synchronize to ensure GPU has completed before reading results
    synchronize(device_id);

    // Convert back to original dtype if we did an Int64->Float64 conversion
    Tensor result = output;
    if (orig_dtype != result.dtype()) {
        result = result.to(orig_dtype);
    }

    // If output should be scalar but we used [1] internally, reshape to scalar
    if (full_reduction && !keepdim) {
        return result.reshape({});
    }
    return result;
}

auto VulkanBackend::dispatchBooleanReduction(const std::string& op_name,
                                              const Tensor& input,
                                              int64_t dim, bool keepdim) -> Tensor {
    // Handle empty tensors
    if (input.numel() == 0) {
        // any of empty = false, all of empty = true
        bool identity = (op_name == "all");
        std::vector<int64_t> out_shape;
        if (dim == INT64_MIN) {
            out_shape = keepdim ? std::vector<int64_t>(input.shape().size(), 1) : std::vector<int64_t>{};
        } else {
            auto input_shape = input.shape();
            out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
            if (keepdim) out_shape[dim < 0 ? dim + static_cast<int64_t>(input_shape.size()) : dim] = 1;
            else out_shape.erase(out_shape.begin() + (dim < 0 ? dim + static_cast<int64_t>(input_shape.size()) : dim));
        }
        return dispatchFull(out_shape, identity ? 1.0f : 0.0f, DType::Bool);
    }

    // BFloat16: upcast to Float32
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        return dispatchBooleanReduction(op_name, input_f32, dim, keepdim);
    }

    int32_t device_id = input.device().index;
    auto input_shape = input.shape();

    // Handle dimension specification
    bool full_reduction = (dim == INT64_MIN);
    if (!full_reduction && dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }

    // Select shader based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "boolean_reduction_f64" : "boolean_reduction";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Map operation name to opcode: 0=any, 1=all
    uint32_t op_code = (op_name == "all") ? 1 : 0;

    // Calculate output shape
    std::vector<int64_t> out_shape;
    if (full_reduction) {
        if (keepdim) {
            out_shape.assign(input_shape.size(), 1);
        } else {
            out_shape = {};
        }
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    // Output uses DType::Bool (uint8_t per element), but shader writes int (4 bytes per element)
    // We use an intermediate Int32 buffer, then cast to Bool
    Tensor int_output(out_shape.empty() ? std::vector<int64_t>{1} : out_shape, DType::Int32, input.device());

    // Get buffer handles
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = int_output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = int_output.numel() * int_output.dtype_size();

    // Allocate descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate reduction parameters
    uint32_t inner_size = 1;
    if (!full_reduction) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    // Push constants
    struct {
        uint32_t n;
        uint32_t reduce_size;
        uint32_t outer_size;
        uint32_t inner_size;
        uint32_t op;
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.reduce_size = full_reduction ? pushConstants.n : static_cast<uint32_t>(input_shape[dim]);
    pushConstants.outer_size = static_cast<uint32_t>(int_output.numel());
    pushConstants.inner_size = inner_size;
    pushConstants.op = op_code;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // One workgroup per output element
    uint32_t workgroups = pushConstants.outer_size;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    synchronize(device_id);

    // Cast Int32 result to Bool
    Tensor output = dispatchCast(int_output, DType::Bool);

    // If full reduction with no keepdim, reshape to scalar
    if (full_reduction && !keepdim) {
        return output.reshape({});
    }
    return output;
}

auto VulkanBackend::dispatchAll(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    return dispatchBooleanReduction("all", input, dim, keepdim);
}

auto VulkanBackend::dispatchAny(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    return dispatchBooleanReduction("any", input, dim, keepdim);
}

auto VulkanBackend::dispatchLogSumExp(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Empty tensor: return -inf (log of 0)
    if (input.numel() == 0) {
        std::vector<int64_t> out_shape;
        if (dim == INT64_MIN) {
            out_shape = keepdim ? std::vector<int64_t>(input.shape().size(), 1) : std::vector<int64_t>{};
        } else {
            auto input_shape = input.shape();
            out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
            int64_t d = dim < 0 ? static_cast<int64_t>(input_shape.size()) + dim : dim;
            if (keepdim) {
                out_shape[d] = 1;
            } else {
                out_shape.erase(out_shape.begin() + d);
            }
        }
        return dispatchFull(out_shape, -std::numeric_limits<float>::infinity(), input.dtype());
    }

    int32_t device_id = input.device().index;
    auto input_shape = input.shape();

    bool full_reduction = (dim == INT64_MIN);
    if (!full_reduction && dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }

    // Select shader by dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "logsumexp_f64";
    } else if (is_float16) {
        shader_name = "logsumexp_f16";
    } else if (is_bfloat16) {
        shader_name = "logsumexp_bf16";
    } else {
        shader_name = "logsumexp";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Calculate output shape
    std::vector<int64_t> out_shape;
    if (full_reduction) {
        if (keepdim) {
            out_shape.assign(input_shape.size(), 1);
        } else {
            out_shape = {};
        }
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, input.dtype(), input.device());

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate inner_size
    uint32_t inner_size = 1;
    if (!full_reduction) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct {
        uint32_t n;
        uint32_t reduce_size;
        uint32_t outer_size;
        uint32_t inner_size;
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.reduce_size = full_reduction ? pushConstants.n : static_cast<uint32_t>(input_shape[dim]);
    pushConstants.outer_size = full_reduction ? 1 : static_cast<uint32_t>(output.numel());
    pushConstants.inner_size = inner_size;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups;
    if (is_float16) {
        workgroups = (pushConstants.outer_size + 1) / 2;
    } else {
        workgroups = pushConstants.outer_size;
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);
    synchronize(device_id);

    return output;
}

auto VulkanBackend::dispatchTriuTril(const std::string& op_name,
                                      const Tensor& input, int64_t diagonal) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() < 2) {
        throw std::invalid_argument("triu/tril requires at least a 2D tensor");
    }

    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "triu_tril_f64" : is_float16 ? "triu_tril_f16" : is_bfloat16 ? "triu_tril_bf16" : "triu_tril";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    uint32_t rows = static_cast<uint32_t>(input_shape[input_shape.size() - 2]);
    uint32_t cols = static_cast<uint32_t>(input_shape[input_shape.size() - 1]);
    uint32_t op_code = (op_name == "tril") ? 1 : 0;

    struct {
        uint32_t n;
        uint32_t rows;
        uint32_t cols;
        int32_t diagonal;
        uint32_t op;
    } pushConstants;
    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.rows = rows;
    pushConstants.cols = cols;
    pushConstants.diagonal = static_cast<int32_t>(diagonal);
    pushConstants.op = op_code;

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
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

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups;
    if (is_float16) {
        workgroups = div_wg((input.numel() + 1) / 2, devices_[device_id].workgroupSize);
    } else {
        workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchDiag(const Tensor& input, int64_t diagonal) -> Tensor {
    auto input_shape = input.shape();

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "diag_f64" : is_float16 ? "diag_f16" : is_bfloat16 ? "diag_bf16" : "diag";
    auto* pipeline = getPipeline(shader_name, device_id);

    if (input_shape.size() == 1) {
        // 1D -> 2D: construct diagonal matrix
        int64_t vec_len = input_shape[0];
        int64_t abs_diag = diagonal >= 0 ? diagonal : -diagonal;
        int64_t mat_size = vec_len + abs_diag;

        std::vector<int64_t> out_shape = {mat_size, mat_size};
        Tensor output(out_shape, input.dtype(), input.device());

        struct {
            uint32_t n;
            uint32_t rows;
            uint32_t cols;
            int32_t diagonal;
            uint32_t op;
        } pushConstants;
        pushConstants.n = static_cast<uint32_t>(output.numel());
        pushConstants.rows = static_cast<uint32_t>(mat_size);
        pushConstants.cols = static_cast<uint32_t>(mat_size);
        pushConstants.diagonal = static_cast<int32_t>(diagonal);
        pushConstants.op = 1; // construct

        const void* buffer_in = input.data_ptr();
        const void* buffer_out = output.data_ptr();

        size_t buffer_size_in = input.numel() * input.dtype_size();
        size_t buffer_size_out = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_in},
            {1, buffer_out}
        };
        std::vector<size_t> sizes_vec = {buffer_size_in, buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes_vec);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

        uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    } else if (input_shape.size() == 2) {
        // 2D -> 1D: extract diagonal
        int64_t rows = input_shape[0];
        int64_t cols = input_shape[1];

        int64_t diag_len;
        if (diagonal >= 0) {
            diag_len = std::min(rows, cols - diagonal);
        } else {
            diag_len = std::min(rows + diagonal, cols);
        }
        if (diag_len <= 0) {
            return Tensor({0}, input.dtype(), input.device());
        }

        std::vector<int64_t> out_shape = {diag_len};
        Tensor output(out_shape, input.dtype(), input.device());

        struct {
            uint32_t n;
            uint32_t rows;
            uint32_t cols;
            int32_t diagonal;
            uint32_t op;
        } pushConstants;
        pushConstants.n = static_cast<uint32_t>(diag_len);
        pushConstants.rows = static_cast<uint32_t>(rows);
        pushConstants.cols = static_cast<uint32_t>(cols);
        pushConstants.diagonal = static_cast<int32_t>(diagonal);
        pushConstants.op = 0; // extract

        const void* buffer_in = input.data_ptr();
        const void* buffer_out = output.data_ptr();

        size_t buffer_size_in = input.numel() * input.dtype_size();
        size_t buffer_size_out = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_in},
            {1, buffer_out}
        };
        std::vector<size_t> sizes_vec = {buffer_size_in, buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes_vec);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

        uint32_t workgroups = div_wg(diag_len, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    } else {
        throw std::invalid_argument("diag requires a 1D or 2D tensor, got " +
                                    std::to_string(input_shape.size()) + "D");
    }
}

auto VulkanBackend::dispatchFlip(const Tensor& input, int64_t dim) -> Tensor {
    auto input_shape = input.shape();

    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Handle negative dim
    if (dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("flip dim out of range");
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "flip_f64" : is_float16 ? "flip_f16" : is_bfloat16 ? "flip_bf16" : "flip";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate inner_size: product of dims after flip dim
    uint32_t inner_size = 1;
    for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
        inner_size *= static_cast<uint32_t>(input_shape[i]);
    }

    struct {
        uint32_t n;
        uint32_t dim_size;
        uint32_t inner_size;
    } pushConstants;
    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.dim_size = static_cast<uint32_t>(input_shape[dim]);
    pushConstants.inner_size = inner_size;

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
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

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups;
    if (is_float16) {
        workgroups = div_wg((input.numel() + 1) / 2, devices_[device_id].workgroupSize);
    } else {
        workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchRoll(const Tensor& input, int64_t shift, int64_t dim) -> Tensor {
    auto input_shape = input.shape();

    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Handle negative dim
    if (dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("roll dim out of range");
    }

    // Float16: native F16 roll shader
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);

    std::string shader_name = is_float64 ? "roll_f64" : is_float16 ? "roll_f16" : is_bfloat16 ? "roll_bf16" : "roll";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate inner_size: product of dims after roll dim
    uint32_t inner_size = 1;
    for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
        inner_size *= static_cast<uint32_t>(input_shape[i]);
    }

    // Normalize shift to [0, dim_size)
    int64_t dim_size = input_shape[dim];
    int64_t normalized_shift = ((shift % dim_size) + dim_size) % dim_size;

    struct {
        uint32_t n;
        uint32_t dim_size;
        uint32_t shift;
        uint32_t inner_size;
    } pushConstants;
    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.dim_size = static_cast<uint32_t>(dim_size);
    pushConstants.shift = static_cast<uint32_t>(normalized_shift);
    pushConstants.inner_size = inner_size;

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
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchRepeatInterleave(const Tensor& input, int64_t repeats, int64_t dim) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dim
    if (dim < 0) dim += ndim;

    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
        out_shape[dim] = 0;
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Build output shape
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    out_shape[dim] = input_shape[dim] * repeats;

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "repeat_interleave_f64"
                       : is_f16 ? "repeat_interleave_f16"
                       : is_bf16 ? "repeat_interleave_bf16"
                       : "repeat_interleave";

    // Int types reinterpret as float (same 4-byte width)
    Tensor work_input = input;
    DType orig_dtype = input.dtype();
    if (orig_dtype == DType::Int32) {
        // Int32 and Float32 are both 4 bytes; reinterpret
        shader = "repeat_interleave";
    } else if (orig_dtype == DType::Int64) {
        // Int64 is 8 bytes like Float64
        shader = "repeat_interleave_f64";
    } else if (orig_dtype == DType::Int8 || orig_dtype == DType::Bool || orig_dtype == DType::UInt8) {
        // Upcast to Float32, repeat, then cast back
        Tensor f32_input = dispatchCast(input, DType::Float32);
        Tensor f32_result = dispatchRepeatInterleave(f32_input, repeats, dim);
        return dispatchCast(f32_result, orig_dtype);
    }

    auto* pipeline = getPipeline(shader, device_id);

    Tensor output(out_shape, orig_dtype, input.device());

    // Compute dim decomposition
    uint32_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) {
        inner_size *= static_cast<uint32_t>(input_shape[i]);
    }
    uint32_t dim_size = static_cast<uint32_t>(input_shape[dim]);
    uint32_t total_elements = static_cast<uint32_t>(output.numel());

    struct {
        uint32_t total_elements;
        uint32_t repeats;
        uint32_t dim_size;
        uint32_t inner_size;
    } pc;
    pc.total_elements = total_elements;
    pc.repeats = static_cast<uint32_t>(repeats);
    pc.dim_size = dim_size;
    pc.inner_size = inner_size;

    size_t elem = input.dtype_size();
    size_t in_buf_size = static_cast<size_t>(input.numel()) * elem;
    size_t out_buf_size = static_cast<size_t>(output.numel()) * elem;
    if (is_f16 || is_bf16) {
        in_buf_size = ((input.numel() + 1) / 2) * 4;
        out_buf_size = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}
    };
    std::vector<size_t> sizes = {in_buf_size, out_buf_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_elements, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchRepeatInterleaveTensor(const Tensor& input, const Tensor& repeats, int64_t dim) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dim
    if (dim < 0) dim += ndim;

    int32_t device_id = input.device().index;

    // Cast repeats to Int32 if needed (shader expects int buffer)
    Tensor int_repeats = (repeats.dtype() != DType::Int32)
        ? dispatchCast(repeats, DType::Int32) : repeats;

    // Compute exclusive prefix sum of repeats on GPU using cumsum
    // Then read total from the last element to determine output size
    Tensor prefix_sum = dispatchCumSum(int_repeats.to(DType::Float32), 0);
    // We need an offsets array of size (dim_size + 1) with offsets[0] = 0
    // The prefix sum gives us offsets[1..dim_size]
    // Minimal scalar readback: only the total (last element) for output allocation
    int64_t dim_size = input_shape[dim];
    Tensor total_scalar = prefix_sum.slice(0, dim_size - 1, dim_size).to(Device::cpu());
    float total_f = static_cast<const float*>(total_scalar.data_ptr())[0];
    int64_t total_repeats = static_cast<int64_t>(total_f);

    if (total_repeats == 0 || input.numel() == 0) {
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
        out_shape[dim] = 0;
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Build offsets on GPU: offsets[0] = 0, offsets[i+1] = int(cumsum[i])
    Tensor offsets = dispatchZeros({dim_size + 1}, DType::Int32, input.device());
    {
        auto* ofs_pipeline = getPipeline("build_offsets_from_cumsum", device_id);
        size_t cs_size = prefix_sum.numel() * sizeof(float);
        size_t of_size = offsets.numel() * sizeof(int32_t);
        std::vector<std::pair<uint32_t, const void*>> ofs_bindings = {
            {0, prefix_sum.data_ptr()}, {1, offsets.data_ptr()},
        };
        std::vector<size_t> ofs_sizes = {cs_size, of_size};
        VkDescriptorSet ofs_ds = allocateAndWriteDescriptorSet(
            device_id, ofs_pipeline, ofs_bindings, ofs_sizes);

        struct { uint32_t n; } ofs_pc;
        ofs_pc.n = static_cast<uint32_t>(dim_size);

        VkCommandBuffer ofs_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(ofs_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ofs_pipeline->pipeline());
        vkCmdBindDescriptorSets(ofs_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               ofs_pipeline->layout(), 0, 1, &ofs_ds, 0, nullptr);
        vkCmdPushConstants(ofs_cmd, ofs_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(ofs_pc), &ofs_pc);
        vkCmdDispatch(ofs_cmd, 1, 1, 1);
        insertComputeOnlyBarrier(ofs_cmd);
        endSingleTimeCommands(ofs_cmd, device_id);
    }

    // Build output shape
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    out_shape[dim] = total_repeats;

    // For non-float types, handle via cast round-trip
    DType orig_dtype = input.dtype();
    if (orig_dtype == DType::Int8 || orig_dtype == DType::Bool || orig_dtype == DType::UInt8) {
        Tensor f32_input = dispatchCast(input, DType::Float32);
        Tensor f32_result = dispatchRepeatInterleaveTensor(f32_input, repeats, dim);
        return dispatchCast(f32_result, orig_dtype);
    }

    // Select shader (float32 only for now; int32/int64 reinterpret as float/double)
    std::string shader = "repeat_interleave_tensor";
    auto* pipeline = getPipeline(shader, device_id);

    Tensor output(out_shape, orig_dtype, input.device());

    uint32_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) {
        inner_size *= static_cast<uint32_t>(input_shape[i]);
    }

    uint32_t total_output = static_cast<uint32_t>(output.numel());

    struct {
        uint32_t total_output;
        uint32_t dim_size;
        uint32_t inner_size;
    } pc;
    pc.total_output = total_output;
    pc.dim_size = static_cast<uint32_t>(dim_size);
    pc.inner_size = inner_size;

    size_t elem = input.dtype_size();
    size_t in_buf_size = static_cast<size_t>(input.numel()) * elem;
    size_t out_buf_size = static_cast<size_t>(output.numel()) * elem;
    size_t offsets_buf_size = static_cast<size_t>(dim_size + 1) * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}, {2, offsets.data_ptr()}
    };
    std::vector<size_t> sizes_vec = {in_buf_size, out_buf_size, offsets_buf_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes_vec);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_output, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchCountNonzero(const Tensor& input) -> Tensor {
    // Handle empty tensors
    if (input.numel() == 0) {
        return dispatchFull({1}, 0.0f, DType::Int64);
    }

    // BFloat16/Float16: upcast to Float32 (shader uses float buffer)
    if (input.dtype() == DType::BFloat16 || input.dtype() == DType::Float16) {
        return dispatchCountNonzero(input.to(DType::Float32));
    }

    int32_t device_id = input.device().index;

    // Shader outputs the count as a float (or float64), which we then convert to Int64
    bool is_f64 = (input.dtype() == DType::Float64);
    std::string shader_name = is_f64 ? "count_nonzero_f64" : "count_nonzero";
    DType out_dtype = is_f64 ? DType::Float64 : DType::Float32;
    auto* pipeline = getPipeline(shader_name, device_id);

    Tensor output({1}, out_dtype, input.device());

    const void* buffer_in  = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size_in  = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct {
        uint32_t num_elements;
    } pushConstants;
    pushConstants.num_elements = static_cast<uint32_t>(input.numel());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Single workgroup dispatch
    vkCmdDispatch(cmdBuffer, 1, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);
    synchronize(device_id);

    // Convert float -> Int64; keep shape {1} to match CPU backend
    return output.to(DType::Int64);
}

auto VulkanBackend::dispatchNansum(const Tensor& input) -> Tensor {
    // Handle empty tensors
    if (input.numel() == 0) {
        return dispatchFull({1}, 0.0f, input.dtype());
    }

    // BFloat16/Float16: upcast to Float32
    DType orig_dtype = input.dtype();
    Tensor work_input = input;
    if (orig_dtype == DType::BFloat16 || orig_dtype == DType::Float16) {
        work_input = input.to(DType::Float32);
    }

    int32_t device_id = work_input.device().index;

    std::string shader_name = (work_input.dtype() == DType::Float64)
        ? "nansum_f64" : "nansum";
    auto* pipeline = getPipeline(shader_name, device_id);

    Tensor output({1}, work_input.dtype(), work_input.device());

    const void* buffer_in  = work_input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size_in  = work_input.numel() * work_input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct {
        uint32_t num_elements;
    } pushConstants;
    pushConstants.num_elements = static_cast<uint32_t>(work_input.numel());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    vkCmdDispatch(cmdBuffer, 1, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);
    synchronize(device_id);

    // Convert back to original dtype if needed; keep shape {1}
    if (orig_dtype != work_input.dtype()) {
        return output.to(orig_dtype);
    }
    return output;
}

auto VulkanBackend::dispatchNanmean(const Tensor& input) -> Tensor {
    // Handle empty tensors: NaN for empty (no non-NaN elements)
    if (input.numel() == 0) {
        return dispatchFull({1}, std::numeric_limits<float>::quiet_NaN(), input.dtype());
    }

    // BFloat16/Float16: upcast to Float32
    DType orig_dtype = input.dtype();
    Tensor work_input = input;
    if (orig_dtype == DType::BFloat16 || orig_dtype == DType::Float16) {
        work_input = input.to(DType::Float32);
    }

    int32_t device_id = work_input.device().index;

    std::string shader_name = (work_input.dtype() == DType::Float64)
        ? "nanmean_f64" : "nanmean";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Output: single element (the shader computes mean directly)
    Tensor output({1}, work_input.dtype(), work_input.device());

    const void* buffer_in  = work_input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size_in  = work_input.numel() * work_input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct {
        uint32_t num_elements;
    } pushConstants;
    pushConstants.num_elements = static_cast<uint32_t>(work_input.numel());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    vkCmdDispatch(cmdBuffer, 1, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);
    synchronize(device_id);

    // Convert back to original dtype if needed; keep shape {1}
    if (orig_dtype != work_input.dtype()) {
        return output.to(orig_dtype);
    }
    return output;
}

auto VulkanBackend::dispatchAminmax(const Tensor& input) -> std::pair<Tensor, Tensor> {
    // Handle empty tensors
    if (input.numel() == 0) {
        throw std::invalid_argument("aminmax: cannot compute min/max of empty tensor");
    }

    // BFloat16/Float16: upcast to Float32
    DType orig_dtype = input.dtype();
    Tensor work_input = input;
    if (orig_dtype == DType::BFloat16 || orig_dtype == DType::Float16) {
        work_input = input.to(DType::Float32);
    }

    int32_t device_id = work_input.device().index;

    std::string shader_name = (work_input.dtype() == DType::Float64)
        ? "aminmax_f64" : "aminmax";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Output: 2 elements [min, max]
    Tensor output({2}, work_input.dtype(), work_input.device());

    const void* buffer_in  = work_input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size_in  = work_input.numel() * work_input.dtype_size();
    size_t buffer_size_out = 2 * work_input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct {
        uint32_t num_elements;
    } pushConstants;
    pushConstants.num_elements = static_cast<uint32_t>(work_input.numel());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    vkCmdDispatch(cmdBuffer, 1, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);
    synchronize(device_id);

    // Split the 2-element output [min, max] into two {1}-shaped tensors.
    // Move to CPU for the split, then back to the original device.
    Device orig_device = work_input.device();
    Tensor output_cpu = output.to(Device::cpu());

    Tensor min_tensor({1}, work_input.dtype(), Device::cpu());
    Tensor max_tensor({1}, work_input.dtype(), Device::cpu());

    size_t elem_size = work_input.dtype_size();
    std::memcpy(min_tensor.data_ptr(),
                static_cast<const char*>(output_cpu.data_ptr()),
                elem_size);
    std::memcpy(max_tensor.data_ptr(),
                static_cast<const char*>(output_cpu.data_ptr()) + elem_size,
                elem_size);

    // Move back to original device
    min_tensor = min_tensor.to(orig_device);
    max_tensor = max_tensor.to(orig_device);

    // Convert back to original dtype if needed
    if (orig_dtype != work_input.dtype()) {
        min_tensor = min_tensor.to(orig_dtype);
        max_tensor = max_tensor.to(orig_dtype);
    }

    return {min_tensor, max_tensor};
}

auto VulkanBackend::dispatchTrace(const Tensor& input) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 2) {
        throw std::invalid_argument("trace requires a 2D tensor, got " +
                                    std::to_string(input_shape.size()) + "D");
    }

    // Extract diagonal, then sum
    Tensor diag = dispatchDiag(input, 0);
    return dispatchReduction("sum", diag, INT64_MIN, false);
}

} // namespace tenzor
