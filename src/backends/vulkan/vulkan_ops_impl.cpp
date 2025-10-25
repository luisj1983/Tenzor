/**
 * @file vulkan_ops_impl.cpp
 * @brief Implementation of Vulkan backend operations
 */

#include "vulkan_backend.hpp"
#include <algorithm>
#include <cmath>

namespace tenzor {

// ============================================================================
// Pooling Operations
// ============================================================================

auto VulkanBackend::dispatchMaxPool2d(const Tensor& input,
                                      int64_t kernel_h, int64_t kernel_w,
                                      int64_t stride_h, int64_t stride_w,
                                      int64_t padding_h, int64_t padding_w)
                                      -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t in_height = shape[2];
    int64_t in_width = shape[3];

    int64_t out_height = (in_height + 2*padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2*padding_w - kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("pooling_forward_with_indices", device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_height, out_width};
    Tensor output(out_shape, input.dtype(), input.device());
    Tensor indices(out_shape, DType::Int32, input.device());

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Push constants
    struct PushConstants {
        uint32_t batch, channels, in_height, in_width;
        uint32_t out_height, out_width, kernel_h, kernel_w;
        uint32_t stride_h, stride_w, padding_h, padding_w;
    } push;

    push.batch = batch;
    push.channels = channels;
    push.in_height = in_height;
    push.in_width = in_width;
    push.out_height = out_height;
    push.out_width = out_width;
    push.kernel_h = kernel_h;
    push.kernel_w = kernel_w;
    push.stride_h = stride_h;
    push.stride_w = stride_w;
    push.padding_h = padding_h;
    push.padding_w = padding_w;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups_x = (out_width + 15) / 16;
    uint32_t workgroups_y = (out_height + 15) / 16;
    uint32_t workgroups_z = channels;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchAvgPool2d(const Tensor& input,
                                      int64_t kernel_h, int64_t kernel_w,
                                      int64_t stride_h, int64_t stride_w,
                                      int64_t padding_h, int64_t padding_w) -> Tensor {
    auto shape = input.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t in_height = shape[2];
    int64_t in_width = shape[3];

    int64_t out_height = (in_height + 2*padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2*padding_w - kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("pooling", device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_height, out_width};
    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t batch, channels, in_height, in_width;
        uint32_t out_height, out_width, kernel_h, kernel_w;
        uint32_t stride_h, stride_w, padding_h, padding_w;
        uint32_t pool_type;
    } push;

    push.batch = batch;
    push.channels = channels;
    push.in_height = in_height;
    push.in_width = in_width;
    push.out_height = out_height;
    push.out_width = out_width;
    push.kernel_h = kernel_h;
    push.kernel_w = kernel_w;
    push.stride_h = stride_h;
    push.stride_w = stride_w;
    push.padding_h = padding_h;
    push.padding_w = padding_w;
    push.pool_type = 1; // avg pool

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups_x = (out_width + 15) / 16;
    uint32_t workgroups_y = (out_height + 15) / 16;
    uint32_t workgroups_z = channels;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAdaptiveMaxPool2d(const Tensor& input, int64_t out_h, int64_t out_w)
                                              -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t in_height = shape[2];
    int64_t in_width = shape[3];

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("adaptive_pooling", device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_h, out_w};
    Tensor output(out_shape, input.dtype(), input.device());
    Tensor indices(out_shape, DType::Int32, input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t batch, channels, in_height, in_width;
        uint32_t out_height, out_width, pool_type;
    } push;

    push.batch = batch;
    push.channels = channels;
    push.in_height = in_height;
    push.in_width = in_width;
    push.out_height = out_h;
    push.out_width = out_w;
    push.pool_type = 0; // max

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups_x = (out_w + 15) / 16;
    uint32_t workgroups_y = (out_h + 15) / 16;
    uint32_t workgroups_z = channels;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchAdaptiveAvgPool2d(const Tensor& input, int64_t out_h, int64_t out_w) -> Tensor {
    auto shape = input.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t in_height = shape[2];
    int64_t in_width = shape[3];

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("adaptive_pooling", device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_h, out_w};
    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t batch, channels, in_height, in_width;
        uint32_t out_height, out_width, pool_type;
    } push;

    push.batch = batch;
    push.channels = channels;
    push.in_height = in_height;
    push.in_width = in_width;
    push.out_height = out_h;
    push.out_width = out_w;
    push.pool_type = 1; // avg

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups_x = (out_w + 15) / 16;
    uint32_t workgroups_y = (out_h + 15) / 16;
    uint32_t workgroups_z = channels;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchMaxPool2dBackward(const Tensor& grad_out, const Tensor& input,
                                               const Tensor& indices,
                                               int64_t kernel_h, int64_t kernel_w,
                                               int64_t stride_h, int64_t stride_w,
                                               int64_t padding_h, int64_t padding_w) -> Tensor {
    auto in_shape = input.shape();
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("pooling_backward", device_id);

    Tensor grad_input(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                     input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t batch, channels, in_height, in_width;
        uint32_t out_height, out_width, kernel_h, kernel_w;
        uint32_t stride_h, stride_w, padding_h, padding_w;
        uint32_t pool_type;
    } push;

    auto grad_shape = grad_out.shape();
    push.batch = in_shape[0];
    push.channels = in_shape[1];
    push.in_height = in_shape[2];
    push.in_width = in_shape[3];
    push.out_height = grad_shape[2];
    push.out_width = grad_shape[3];
    push.kernel_h = kernel_h;
    push.kernel_w = kernel_w;
    push.stride_h = stride_h;
    push.stride_w = stride_w;
    push.padding_h = padding_h;
    push.padding_w = padding_w;
    push.pool_type = 0; // max

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups_x = (in_shape[3] + 15) / 16;
    uint32_t workgroups_y = (in_shape[2] + 15) / 16;
    uint32_t workgroups_z = in_shape[1];
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// ============================================================================
// Normalization Operations
// ============================================================================

auto VulkanBackend::dispatchBatchNorm2d(const Tensor& input, const Tensor& mean, const Tensor& var,
                                        const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("batchnorm", device_id);

    auto in_shape = input.shape();
    Tensor output(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                 input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t batch, channels, height, width;
        float epsilon;
        uint32_t affine;
    } push;

    push.batch = in_shape[0];
    push.channels = in_shape[1];
    push.height = in_shape[2];
    push.width = in_shape[3];
    push.epsilon = epsilon;
    push.affine = (gamma && beta) ? 1 : 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t total = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3];
    uint32_t workgroups = (total + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchBatchNorm2dBackward(const Tensor& grad_out, const Tensor& input,
                                                const Tensor& mean, const Tensor& var,
                                                const Tensor* gamma, float epsilon)
                                                -> std::tuple<Tensor, Tensor, Tensor> {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("batchnorm_backward", device_id);

    auto in_shape = input.shape();
    Tensor grad_input(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                     input.dtype(), input.device());
    Tensor grad_gamma({in_shape[1]}, input.dtype(), input.device());
    Tensor grad_beta({in_shape[1]}, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t batch, channels, height, width;
        float epsilon;
        uint32_t affine;
    } push;

    push.batch = in_shape[0];
    push.channels = in_shape[1];
    push.height = in_shape[2];
    push.width = in_shape[3];
    push.epsilon = epsilon;
    push.affine = gamma ? 1 : 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t total = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3];
    uint32_t workgroups = (total + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return {grad_input, grad_gamma, grad_beta};
}

auto VulkanBackend::dispatchLayerNorm(const Tensor& input, int64_t normalized_shape,
                                      const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("layer_norm", device_id);

    auto in_shape = input.shape();
    Tensor output(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                 input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t batch_size, normalized_shape;
        float epsilon;
        uint32_t affine;
    } push;

    push.batch_size = input.numel() / normalized_shape;
    push.normalized_shape = normalized_shape;
    push.epsilon = epsilon;
    push.affine = (gamma && beta) ? 1 : 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = push.batch_size;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchGroupNorm(const Tensor& input, int64_t num_groups,
                                      const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("group_norm", device_id);

    auto in_shape = input.shape();
    Tensor output(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                 input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t batch_size, num_channels, height, width;
        uint32_t num_groups;
        float epsilon;
        uint32_t affine;
    } push;

    push.batch_size = in_shape[0];
    push.num_channels = in_shape[1];
    push.height = in_shape[2];
    push.width = in_shape[3];
    push.num_groups = num_groups;
    push.epsilon = epsilon;
    push.affine = (gamma && beta) ? 1 : 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = in_shape[0] * num_groups;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Softmax and Loss Operations
// ============================================================================

auto VulkanBackend::dispatchSoftmax(const Tensor& input, int64_t dim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("softmax", device_id);

    auto in_shape = input.shape();
    Tensor output(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                 input.dtype(), input.device());

    // Assuming last dimension for now
    int64_t batch_size = input.numel() / in_shape.back();
    int64_t num_classes = in_shape.back();

    // Auxiliary buffers for max and sum
    Tensor max_vals({batch_size}, input.dtype(), input.device());
    Tensor sum_vals({batch_size}, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t batch_size, num_classes, dim, mode;
    } push;

    push.batch_size = batch_size;
    push.num_classes = num_classes;
    push.dim = (dim < 0) ? in_shape.size() + dim : dim;
    push.mode = 0; // forward

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    vkCmdDispatch(cmdBuffer, batch_size, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchLogSoftmax(const Tensor& input, int64_t dim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("log_softmax", device_id);

    auto in_shape = input.shape();
    Tensor output(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                 input.dtype(), input.device());

    int64_t batch_size = input.numel() / in_shape.back();
    int64_t num_classes = in_shape.back();

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t batch_size, num_classes, dim;
    } push;

    push.batch_size = batch_size;
    push.num_classes = num_classes;
    push.dim = (dim < 0) ? in_shape.size() + dim : dim;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    vkCmdDispatch(cmdBuffer, batch_size, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchCrossEntropy(const Tensor& log_probs, const Tensor& targets,
                                         int64_t reduction) -> Tensor {
    int32_t device_id = log_probs.device().index;
    auto* pipeline = getPipeline("cross_entropy", device_id);

    auto lp_shape = log_probs.shape();
    int64_t batch_size = lp_shape[0];
    int64_t num_classes = lp_shape[1];

    std::vector<int64_t> out_shape = (reduction == 0) ? std::vector<int64_t>{batch_size} : std::vector<int64_t>{1};
    Tensor output(out_shape, log_probs.dtype(), log_probs.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t batch_size, num_classes, reduction;
    } push;

    push.batch_size = batch_size;
    push.num_classes = num_classes;
    push.reduction = reduction;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = (batch_size + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Advanced Reduction Operations
// ============================================================================

auto VulkanBackend::dispatchArgmax(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("argmax_argmin", device_id);

    auto in_shape = input.shape();
    std::vector<int64_t> out_shape;
    int64_t reduce_size = 1;

    if (dim < 0) {
        out_shape = keepdim ? std::vector<int64_t>(in_shape.size(), 1) : std::vector<int64_t>{1};
        reduce_size = input.numel();
    } else {
        out_shape = std::vector<int64_t>(in_shape.begin(), in_shape.end());
        reduce_size = out_shape[dim];
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, DType::Int32, input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t n, reduce_size, outer_size, op;
    } push;

    push.n = input.numel();
    push.reduce_size = reduce_size;
    push.outer_size = input.numel() / reduce_size;
    push.op = 0; // argmax

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchArgmin(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("argmax_argmin", device_id);

    auto in_shape = input.shape();
    std::vector<int64_t> out_shape;
    int64_t reduce_size = 1;

    if (dim < 0) {
        out_shape = keepdim ? std::vector<int64_t>(in_shape.size(), 1) : std::vector<int64_t>{1};
        reduce_size = input.numel();
    } else {
        out_shape = std::vector<int64_t>(in_shape.begin(), in_shape.end());
        reduce_size = out_shape[dim];
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, DType::Int32, input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t n, reduce_size, outer_size, op;
    } push;

    push.n = input.numel();
    push.reduce_size = reduce_size;
    push.outer_size = input.numel() / reduce_size;
    push.op = 1; // argmin

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchVariance(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor {
    // First compute mean
    auto mean = dispatchReduction("mean", input, dim, true);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("variance_std", device_id);

    auto in_shape = input.shape();
    std::vector<int64_t> out_shape;
    int64_t reduce_size = 1;

    if (dim < 0) {
        out_shape = keepdim ? std::vector<int64_t>(in_shape.size(), 1) : std::vector<int64_t>{1};
        reduce_size = input.numel();
    } else {
        out_shape = std::vector<int64_t>(in_shape.begin(), in_shape.end());
        reduce_size = out_shape[dim];
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t n, reduce_size, outer_size, op, unbiased;
    } push;

    push.n = input.numel();
    push.reduce_size = reduce_size;
    push.outer_size = input.numel() / reduce_size;
    push.op = 0; // variance
    push.unbiased = unbiased ? 1 : 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchStd(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor {
    auto mean = dispatchReduction("mean", input, dim, true);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("variance_std", device_id);

    auto in_shape = input.shape();
    std::vector<int64_t> out_shape;
    int64_t reduce_size = 1;

    if (dim < 0) {
        out_shape = keepdim ? std::vector<int64_t>(in_shape.size(), 1) : std::vector<int64_t>{1};
        reduce_size = input.numel();
    } else {
        out_shape = std::vector<int64_t>(in_shape.begin(), in_shape.end());
        reduce_size = out_shape[dim];
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t n, reduce_size, outer_size, op, unbiased;
    } push;

    push.n = input.numel();
    push.reduce_size = reduce_size;
    push.outer_size = input.numel() / reduce_size;
    push.op = 1; // std
    push.unbiased = unbiased ? 1 : 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchProd(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("prod_reduction", device_id);

    auto in_shape = input.shape();
    std::vector<int64_t> out_shape;
    int64_t reduce_size = 1;

    if (dim < 0) {
        out_shape = keepdim ? std::vector<int64_t>(in_shape.size(), 1) : std::vector<int64_t>{1};
        reduce_size = input.numel();
    } else {
        out_shape = std::vector<int64_t>(in_shape.begin(), in_shape.end());
        reduce_size = out_shape[dim];
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t n, reduce_size, outer_size;
    } push;

    push.n = input.numel();
    push.reduce_size = reduce_size;
    push.outer_size = input.numel() / reduce_size;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAll(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("boolean_reduction", device_id);

    auto in_shape = input.shape();
    std::vector<int64_t> out_shape;
    int64_t reduce_size = 1;

    if (dim < 0) {
        out_shape = keepdim ? std::vector<int64_t>(in_shape.size(), 1) : std::vector<int64_t>{1};
        reduce_size = input.numel();
    } else {
        out_shape = std::vector<int64_t>(in_shape.begin(), in_shape.end());
        reduce_size = out_shape[dim];
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, DType::Int32, input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t n, reduce_size, outer_size, op;
    } push;

    push.n = input.numel();
    push.reduce_size = reduce_size;
    push.outer_size = input.numel() / reduce_size;
    push.op = 0; // all

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAny(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("boolean_reduction", device_id);

    auto in_shape = input.shape();
    std::vector<int64_t> out_shape;
    int64_t reduce_size = 1;

    if (dim < 0) {
        out_shape = keepdim ? std::vector<int64_t>(in_shape.size(), 1) : std::vector<int64_t>{1};
        reduce_size = input.numel();
    } else {
        out_shape = std::vector<int64_t>(in_shape.begin(), in_shape.end());
        reduce_size = out_shape[dim];
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, DType::Int32, input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t n, reduce_size, outer_size, op;
    } push;

    push.n = input.numel();
    push.reduce_size = reduce_size;
    push.outer_size = input.numel() / reduce_size;
    push.op = 1; // any

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Indexing Operations
// ============================================================================

auto VulkanBackend::dispatchEmbedding(const Tensor& weight, const Tensor& indices,
                                      int64_t padding_idx) -> Tensor {
    int32_t device_id = weight.device().index;
    auto* pipeline = getPipeline("embedding", device_id);

    auto weight_shape = weight.shape();
    auto indices_shape = indices.shape();

    int64_t num_embeddings = weight_shape[0];
    int64_t embedding_dim = weight_shape[1];
    int64_t num_indices = indices.numel();

    std::vector<int64_t> out_shape(indices_shape.begin(), indices_shape.end());
    out_shape.push_back(embedding_dim);

    Tensor output(out_shape, weight.dtype(), weight.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t num_embeddings, embedding_dim, num_indices, padding_idx;
    } push;

    push.num_embeddings = num_embeddings;
    push.embedding_dim = embedding_dim;
    push.num_indices = num_indices;
    push.padding_idx = (padding_idx >= 0) ? padding_idx : UINT32_MAX;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = (num_indices + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchGather(const Tensor& input, int64_t dim, const Tensor& indices) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("gather", device_id);

    auto in_shape = input.shape();
    auto indices_shape = indices.shape();

    std::vector<int64_t> out_shape(indices_shape.begin(), indices_shape.end());

    int64_t dim_size = in_shape[dim];
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < in_shape.size(); i++) {
        inner_size *= in_shape[i];
    }
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= in_shape[i];
    }

    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t input_size, output_size, dim, dim_size, inner_size, outer_size;
    } push;

    push.input_size = input.numel();
    push.output_size = output.numel();
    push.dim = dim;
    push.dim_size = dim_size;
    push.inner_size = inner_size;
    push.outer_size = outer_size;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchScatter(const Tensor& input, int64_t dim, const Tensor& indices,
                                    const Tensor& values, int64_t reduction) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("scatter", device_id);

    auto in_shape = input.shape();
    Tensor output(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                 input.dtype(), input.device());

    int64_t dim_size = in_shape[dim];
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < in_shape.size(); i++) {
        inner_size *= in_shape[i];
    }
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= in_shape[i];
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t input_size, output_size, dim, dim_size, inner_size, outer_size;
        uint32_t reduction, use_values;
    } push;

    push.input_size = indices.numel();
    push.output_size = output.numel();
    push.dim = dim;
    push.dim_size = dim_size;
    push.inner_size = inner_size;
    push.outer_size = outer_size;
    push.reduction = reduction;
    push.use_values = 1;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = (indices.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchIndexSelect(const Tensor& input, int64_t dim, const Tensor& indices) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("index_select", device_id);

    auto in_shape = input.shape();
    auto num_indices = indices.numel();

    std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end());
    out_shape[dim] = num_indices;

    int64_t dim_size = in_shape[dim];
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < in_shape.size(); i++) {
        inner_size *= in_shape[i];
    }
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= in_shape[i];
    }

    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    struct PushConstants {
        uint32_t num_indices, dim, dim_size, inner_size, outer_size;
    } push;

    push.num_indices = num_indices;
    push.dim = dim;
    push.dim_size = dim_size;
    push.inner_size = inner_size;
    push.outer_size = outer_size;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

} // namespace tenzor
