/**
 * @file vulkan_ops_grid_sample.cpp
 * @brief Native Vulkan dispatch for grid_sample and affine_grid.
 *
 * Replaces the previous CPU-roundtrip fallbacks. Float32-only on the GPU;
 * other dtypes are promoted via dispatchCast (an on-device Vulkan compute
 * pipeline) before invocation and demoted afterwards.
 */

#include "vulkan_ops_common.hpp"

namespace tenzor {

namespace {

struct GridSamplePushConstants {
    int32_t N;
    int32_t C;
    int32_t H_in;
    int32_t W_in;
    int32_t H_out;
    int32_t W_out;
    int32_t padding_mode;
    int32_t align_corners;
    int32_t mode;
    int32_t total;
};

struct AffineGridPushConstants {
    int32_t N;
    int32_t H;
    int32_t W;
    int32_t align_corners;
    int32_t total;
};

struct GridSampleBackwardPushConstants {
    int32_t N;
    int32_t C;
    int32_t H_in;
    int32_t W_in;
    int32_t H_out;
    int32_t W_out;
    int32_t padding_mode;
    int32_t align_corners;
    int32_t mode;
    int32_t total;
};

struct AffineGridBackwardPushConstants {
    int32_t N;
    int32_t H;
    int32_t W;
    int32_t align_corners;
    int32_t total;
};

}  // namespace

auto VulkanBackend::dispatchGridSample(const Tensor& input, const Tensor& grid,
                                       const std::string& mode_str,
                                       const std::string& padding_mode_str,
                                       bool align_corners) -> Tensor {
    auto in_shape = input.shape();
    auto grid_shape = grid.shape();
    int32_t N = static_cast<int32_t>(in_shape[0]);
    int32_t C = static_cast<int32_t>(in_shape[1]);
    int32_t H_in = static_cast<int32_t>(in_shape[2]);
    int32_t W_in = static_cast<int32_t>(in_shape[3]);
    int32_t H_out = static_cast<int32_t>(grid_shape[1]);
    int32_t W_out = static_cast<int32_t>(grid_shape[2]);

    DType orig_dtype = input.dtype();
    Tensor input_f32 = (orig_dtype == DType::Float32) ? input : dispatchCast(input, DType::Float32);
    Tensor grid_f32  = (grid.dtype() == DType::Float32) ? grid  : dispatchCast(grid,  DType::Float32);

    Tensor output_f32(std::vector<int64_t>{N, C, H_out, W_out},
                      DType::Float32, input.device());

    int32_t total = N * C * H_out * W_out;
    if (total == 0) return (orig_dtype == DType::Float32) ? output_f32 : dispatchCast(output_f32, orig_dtype);

    int32_t pad_mode = 0;
    if (padding_mode_str == "border") pad_mode = 1;
    else if (padding_mode_str == "reflection") pad_mode = 2;
    int32_t mode_int = (mode_str == "nearest") ? 1 : (mode_str == "bicubic") ? 2 : 0;

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("grid_sample", device_id);

    GridSamplePushConstants pc{N, C, H_in, W_in, H_out, W_out,
                               pad_mode, align_corners ? 1 : 0, mode_int, total};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_f32.data_ptr()},
        {1, grid_f32.data_ptr()},
        {2, output_f32.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(input_f32.numel())  * sizeof(float),
        static_cast<size_t>(grid_f32.numel())   * sizeof(float),
        static_cast<size_t>(output_f32.numel()) * sizeof(float),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(GridSamplePushConstants), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(total), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return (orig_dtype == DType::Float32) ? output_f32 : dispatchCast(output_f32, orig_dtype);
}

// audit Q.4: grid_sample backward. F32 only on GPU; promote/demote via
// dispatchCast as in the forward. Modes: bilinear, nearest, bicubic.
auto VulkanBackend::dispatchGridSampleBackward(const Tensor& grad_output,
                                               const Tensor& input, const Tensor& grid,
                                               const std::string& mode_str,
                                               const std::string& padding_mode_str,
                                               bool align_corners)
    -> std::pair<Tensor, Tensor>
{
    if (mode_str != "bilinear" && mode_str != "nearest" && mode_str != "bicubic") {
        throw std::runtime_error(
            "dispatchGridSampleBackward (Vulkan): mode '" + mode_str +
            "' not supported. Supported: 'bilinear', 'nearest', 'bicubic'.");
    }
    // U.5: grid_sample_backward.comp uses atomicAdd(float) on grad_input
    // (VK_EXT_shader_atomic_float). Fail fast on devices that don't advertise it.
    vulkan::ensure_atomic_float_supported(input.device().index, "grid_sample_backward");
    auto in_shape = input.shape();
    auto grid_shape = grid.shape();
    int32_t N = static_cast<int32_t>(in_shape[0]);
    int32_t C = static_cast<int32_t>(in_shape[1]);
    int32_t H_in = static_cast<int32_t>(in_shape[2]);
    int32_t W_in = static_cast<int32_t>(in_shape[3]);
    int32_t H_out = static_cast<int32_t>(grid_shape[1]);
    int32_t W_out = static_cast<int32_t>(grid_shape[2]);

    DType in_dt = input.dtype();
    DType gr_dt = grid.dtype();

    Tensor input_f32 = (in_dt == DType::Float32) ? input : dispatchCast(input, DType::Float32);
    Tensor grid_f32  = (gr_dt == DType::Float32) ? grid  : dispatchCast(grid,  DType::Float32);
    Tensor go_f32    = (grad_output.dtype() == DType::Float32)
                       ? grad_output : dispatchCast(grad_output, DType::Float32);

    Tensor gi_f32(std::vector<int64_t>{N, C, H_in, W_in},
                  DType::Float32, input.device());
    Tensor gg_f32(std::vector<int64_t>{N, H_out, W_out, 2},
                  DType::Float32, grid.device());

    int32_t total = N * H_out * W_out;
    int32_t device_id = input.device().index;

    // Zero grad_input — scatter is via atomicAdd on this buffer.
    // grad_grid is overwritten by direct store, no zeroing needed.
    {
        auto* fill_pipeline = getPipeline("fill", device_id);
        struct FillPushConstants {
            uint32_t n_elements;
            uint32_t value_bits;
        } fpc;
        fpc.n_elements = static_cast<uint32_t>(gi_f32.numel());
        fpc.value_bits = 0;

        std::vector<std::pair<uint32_t, const void*>> fb = {{0, gi_f32.data_ptr()}};
        std::vector<size_t> fs = {static_cast<size_t>(gi_f32.numel()) * sizeof(float)};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, fill_pipeline, fb, fs);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fill_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               fill_pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, fill_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(FillPushConstants), &fpc);
        uint32_t wg = div_wg(fpc.n_elements, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmd, wg, 1, 1);

        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &mb, 0, nullptr, 0, nullptr);
        endSingleTimeCommands(cmd, device_id);
        // W.9: force the fill submission to finish before the backward kernel
        // is recorded into a fresh command buffer.  Under USE_COMMAND_BATCHING
        // the two endSingleTimeCommands() invocations only enqueue work; the
        // pipeline barrier above is intra-command-buffer and does not span
        // separate submissions.  Without this synchronize, the scatter
        // atomicAdd may race against an unflushed fill.
        synchronize(device_id);
    }

    if (total == 0) {
        return {(in_dt == DType::Float32) ? gi_f32 : dispatchCast(gi_f32, in_dt),
                (gr_dt == DType::Float32) ? gg_f32 : dispatchCast(gg_f32, gr_dt)};
    }

    int32_t pad_mode = 0;
    if (padding_mode_str == "border") pad_mode = 1;
    else if (padding_mode_str == "reflection") pad_mode = 2;
    int32_t mode_int = (mode_str == "nearest") ? 1 : (mode_str == "bicubic") ? 2 : 0;

    auto* pipeline = getPipeline("grid_sample_backward", device_id);
    GridSampleBackwardPushConstants pc{N, C, H_in, W_in, H_out, W_out,
                                        pad_mode, align_corners ? 1 : 0, mode_int, total};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_f32.data_ptr()},
        {1, grid_f32.data_ptr()},
        {2, go_f32.data_ptr()},
        {3, gi_f32.data_ptr()},
        {4, gg_f32.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(input_f32.numel()) * sizeof(float),
        static_cast<size_t>(grid_f32.numel())  * sizeof(float),
        static_cast<size_t>(go_f32.numel())    * sizeof(float),
        static_cast<size_t>(gi_f32.numel())    * sizeof(float),
        static_cast<size_t>(gg_f32.numel())    * sizeof(float),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(GridSampleBackwardPushConstants), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(total), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return {(in_dt == DType::Float32) ? gi_f32 : dispatchCast(gi_f32, in_dt),
            (gr_dt == DType::Float32) ? gg_f32 : dispatchCast(gg_f32, gr_dt)};
}

auto VulkanBackend::dispatchAffineGridBackward(const Tensor& grad_grid,
                                               const std::vector<int64_t>& size,
                                               bool align_corners) -> Tensor
{
    // U.6: affine_grid_backward.comp uses atomicAdd(float) on grad_theta
    // (VK_EXT_shader_atomic_float). Fail fast on devices that don't advertise it.
    vulkan::ensure_atomic_float_supported(grad_grid.device().index, "affine_grid_backward");
    int32_t N = static_cast<int32_t>(size[0]);
    int32_t H = static_cast<int32_t>(size[2]);
    int32_t W = static_cast<int32_t>(size[3]);
    int32_t total = N * H * W;

    DType gr_dt = grad_grid.dtype();
    Tensor gg_f32 = (gr_dt == DType::Float32) ? grad_grid : dispatchCast(grad_grid, DType::Float32);
    Tensor gt_f32(std::vector<int64_t>{N, 2, 3}, DType::Float32, grad_grid.device());

    int32_t device_id = grad_grid.device().index;

    // Zero grad_theta (atomic accumulator).
    {
        auto* fill_pipeline = getPipeline("fill", device_id);
        struct FillPushConstants {
            uint32_t n_elements;
            uint32_t value_bits;
        } fpc;
        fpc.n_elements = static_cast<uint32_t>(gt_f32.numel());
        fpc.value_bits = 0;

        std::vector<std::pair<uint32_t, const void*>> fb = {{0, gt_f32.data_ptr()}};
        std::vector<size_t> fs = {static_cast<size_t>(gt_f32.numel()) * sizeof(float)};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, fill_pipeline, fb, fs);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fill_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               fill_pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, fill_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(FillPushConstants), &fpc);
        uint32_t wg = div_wg(fpc.n_elements, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmd, wg, 1, 1);

        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &mb, 0, nullptr, 0, nullptr);
        endSingleTimeCommands(cmd, device_id);
        // W.9: same fence-gap as grid_sample_backward; force the fill to
        // finish before the backward kernel is recorded into a fresh
        // command buffer.
        synchronize(device_id);
    }

    if (total == 0) {
        return (gr_dt == DType::Float32) ? gt_f32 : dispatchCast(gt_f32, gr_dt);
    }

    auto* pipeline = getPipeline("affine_grid_backward", device_id);
    AffineGridBackwardPushConstants pc{N, H, W, align_corners ? 1 : 0, total};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, gg_f32.data_ptr()},
        {1, gt_f32.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(gg_f32.numel()) * sizeof(float),
        static_cast<size_t>(gt_f32.numel()) * sizeof(float),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(AffineGridBackwardPushConstants), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(total), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return (gr_dt == DType::Float32) ? gt_f32 : dispatchCast(gt_f32, gr_dt);
}

auto VulkanBackend::dispatchAffineGrid(const Tensor& theta, const std::vector<int64_t>& size,
                                       bool align_corners) -> Tensor {
    int32_t N = static_cast<int32_t>(size[0]);
    int32_t H = static_cast<int32_t>(size[2]);
    int32_t W = static_cast<int32_t>(size[3]);
    int32_t total = N * H * W;

    DType orig_dtype = theta.dtype();
    Tensor theta_f32 = (orig_dtype == DType::Float32) ? theta : dispatchCast(theta, DType::Float32);

    Tensor grid(std::vector<int64_t>{N, H, W, 2}, DType::Float32, theta.device());
    if (total == 0) return (orig_dtype == DType::Float32) ? grid : dispatchCast(grid, orig_dtype);

    int32_t device_id = theta.device().index;
    auto* pipeline = getPipeline("affine_grid", device_id);

    AffineGridPushConstants pc{N, H, W, align_corners ? 1 : 0, total};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, theta_f32.data_ptr()},
        {1, grid.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(theta_f32.numel()) * sizeof(float),
        static_cast<size_t>(grid.numel())      * sizeof(float),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(AffineGridPushConstants), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(total), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return (orig_dtype == DType::Float32) ? grid : dispatchCast(grid, orig_dtype);
}

}  // namespace tenzor
