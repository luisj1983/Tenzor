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
    int32_t mode_int = (mode_str == "nearest") ? 1 : 0;

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
