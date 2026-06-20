#include "vulkan_ops_common.hpp"
#include <algorithm>

namespace tenzor {

// Phase 3: BoxIoU - GPU implementation
auto VulkanBackend::dispatchBoxIoU(const Tensor& boxes1, const Tensor& boxes2, int64_t iou_type) -> Tensor {
    int32_t device_id = boxes1.device().index;
    auto* pipeline = getPipeline("box_iou", device_id);

    int64_t N = boxes1.shape()[0];
    int64_t M = boxes2.shape()[0];

    // The box_iou shader operates on float (32-bit), so convert Float64→Float32
    // Also ensure inputs are contiguous (views/slices may have offsets into parent buffers)
    Tensor b1 = boxes1.contiguous();
    Tensor b2 = boxes2.contiguous();
    if (b1.dtype() != DType::Float32) b1 = b1.to(DType::Float32);
    if (b2.dtype() != DType::Float32) b2 = b2.to(DType::Float32);

    Tensor result({N, M}, DType::Float32, boxes1.device());

    size_t elem_size = sizeof(float);
    const void* buf_boxes1 = b1.data_ptr();
    const void* buf_boxes2 = b2.data_ptr();
    const void* buf_result = result.data_ptr();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_boxes1},
        {1, buf_boxes2},
        {2, buf_result},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(N * 4) * elem_size,
        static_cast<size_t>(M * 4) * elem_size,
        static_cast<size_t>(N * M) * elem_size,
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t N;
        uint32_t M;
        uint32_t iou_type;
        uint32_t padding;
    } push_constants;

    push_constants.N = static_cast<uint32_t>(N);
    push_constants.M = static_cast<uint32_t>(M);
    push_constants.iou_type = static_cast<uint32_t>(iou_type);
    push_constants.padding = 0;

    uint64_t total = static_cast<uint64_t>(N) * M;
    uint32_t workgroups = static_cast<uint32_t>(div_wg(total, devices_[device_id].workgroupSize));

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

    // Convert back to original dtype if needed
    if (boxes1.dtype() != DType::Float32) {
        result = result.to(boxes1.dtype());
    }

    return result;
}

// NMS (Non-Maximum Suppression) - GPU implementation
auto VulkanBackend::dispatchNMS(const Tensor& boxes, const Tensor& scores, float iou_threshold) -> Tensor {
    int32_t device_id = boxes.device().index;
    int64_t N = boxes.shape()[0];

    if (N == 0) {
        return Tensor({0}, DType::Int64, boxes.device());
    }

    // Sort scores descending to get order
    auto [sorted_scores, sorted_indices] = dispatchSort(scores, 0, /*descending=*/true);

    // Reorder boxes by sorted indices
    Tensor sorted_boxes = dispatchIndexSelect(boxes, 0, sorted_indices);

    // Ensure Float32 for the shader
    Tensor boxes_f32 = sorted_boxes.contiguous();
    if (boxes_f32.dtype() != DType::Float32) boxes_f32 = boxes_f32.to(DType::Float32);

    // Create suppressed mask (uint32, zero-initialized)
    Tensor suppressed_mask({N}, DType::Int32, boxes.device());
    suppressed_mask = dispatchFill(suppressed_mask, 0.0f);

    // Run NMS shader
    auto* pipeline = getPipeline("nms", device_id);

    const void* buf_boxes = boxes_f32.data_ptr();
    const void* buf_suppressed = suppressed_mask.data_ptr();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_boxes},
        {1, buf_suppressed},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(N * 4) * sizeof(float),
        static_cast<size_t>(N) * sizeof(uint32_t),
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t N;
        float iou_threshold;
    } push_constants;

    push_constants.N = static_cast<uint32_t>(N);
    push_constants.iou_threshold = iou_threshold;

    // The NMS shader is now sequential (local_size_x = 1, single-threaded
    // loop over all boxes). Dispatch exactly one workgroup/one thread so
    // we don't spin up parallel copies that each race on `suppressed`.
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    vkCmdDispatch(cmdBuffer, 1, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);
    synchronize(device_id);

    // GPU compaction: use MaskedSelect-style prefix-sum + scatter to avoid CPU roundtrip.
    // 1. Invert suppressed mask: not_suppressed[i] = (suppressed_mask[i] == 0 ? 1 : 0)
    Tensor ones = dispatchFull({N}, 1.0f, DType::Int32);
    ones = ones.to(boxes.device());
    Tensor not_suppressed = dispatchBinaryOp("sub", ones, suppressed_mask);

    // 2. Prefix sum on not_suppressed flags (inclusive scan giving compacted positions)
    Tensor prefix = dispatchCumSum(not_suppressed, 0);

    // 3. Read back only the total count (4 bytes — single int32). Slice the
    // last element, force a fresh contiguous device copy to land it at offset
    // 0, then transfer to host. This matches the cudaMemcpyAsync 4-byte
    // pattern used by CUDA/ROCm NMS (cuda/kernels/nms.cu:259) and the slice
    // pattern at vulkan_ops_misc.cpp:804 (Unique). The earlier "materialize
    // entire prefix" workaround pre-dated the dispatchContiguous fix that
    // now correctly handles non-zero source offsets.
    Tensor prefix_last = dispatchContiguous(prefix.slice(0, N - 1, N));
    Tensor count_cpu = prefix_last.to(Device::cpu());
    int64_t num_kept = static_cast<int64_t>(count_cpu.data<int32_t>()[0]);

    if (num_kept == 0) {
        return Tensor({0}, DType::Int64, boxes.device());
    }

    // 4. Scatter kept original indices into compacted output on GPU
    //    For each i where not_suppressed[i]==1: output[prefix[i]-1] = sorted_indices[i]
    //    Use the nms_compact shader for this scatter operation.
    Tensor output({num_kept}, DType::Int64, boxes.device());
    {
        auto* compact_pipeline = getPipeline("nms_compact", device_id);
        size_t suppressed_bytes = static_cast<size_t>(N) * sizeof(int32_t);
        size_t prefix_bytes = static_cast<size_t>(N) * sizeof(int32_t);
        size_t sorted_idx_bytes = static_cast<size_t>(N) * sizeof(int64_t);
        size_t output_bytes = static_cast<size_t>(num_kept) * sizeof(int64_t);

        std::vector<std::pair<uint32_t, const void*>> compact_bindings = {
            {0, not_suppressed.data_ptr()},
            {1, prefix.data_ptr()},
            {2, sorted_indices.data_ptr()},
            {3, output.data_ptr()}
        };
        std::vector<size_t> compact_sizes = {suppressed_bytes, prefix_bytes, sorted_idx_bytes, output_bytes};
        VkDescriptorSet compact_ds = allocateAndWriteDescriptorSet(
            device_id, compact_pipeline, compact_bindings, compact_sizes);

        struct { uint32_t N; } compact_pc;
        compact_pc.N = static_cast<uint32_t>(N);

        uint32_t compact_wg = div_wg(static_cast<uint64_t>(N), devices_[device_id].workgroupSize);
        VkCommandBuffer compact_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(compact_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compact_pipeline->pipeline());
        vkCmdBindDescriptorSets(compact_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               compact_pipeline->layout(), 0, 1, &compact_ds, 0, nullptr);
        vkCmdPushConstants(compact_cmd, compact_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compact_pc), &compact_pc);
        vkCmdDispatch(compact_cmd, compact_wg, 1, 1);
        insertComputeOnlyBarrier(compact_cmd);
        endSingleTimeCommands(compact_cmd, device_id);
    }

    return output;
}

// Phase 3: OneHot - GPU implementation
auto VulkanBackend::dispatchOneHot(const Tensor& indices, int64_t num_classes) -> Tensor {
    int32_t device_id = indices.device().index;

    // Match the CPU reference (cpu/kernels/indexing.cpp:one_hot_kernel): indices
    // must be an integer type and the output is always Float32. There is no
    // Float64 one_hot at the op level — the OneHot op carries no output-dtype
    // attribute and the reference kernel produces Float32 unconditionally — so
    // emitting Float64 here would break cross-backend parity. The one_hot_f64
    // shader is intentionally left for a future dtype-propagating op variant.
    if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
        throw std::runtime_error("one_hot: indices must be integer type");
    }

    auto* pipeline = getPipeline("one_hot", device_id);
    DType out_dtype = DType::Float32;

    // Validate index values host-side so out-of-range classes raise the same
    // std::out_of_range as the CPU reference instead of the shader silently
    // writing all-zero rows (negative indices wrap to a huge uint, and indices
    // beyond INT32 range would wrap when narrowed to the 32-bit shader buffer).
    // Valid class indices satisfy 0 <= cls < num_classes, and num_classes fits
    // in int32, so this check also guarantees the Int64->Int32 narrowing below
    // is lossless.
    {
        Tensor idx_host = indices.to(Device::cpu());
        auto check_range = [&](int64_t cls) {
            if (cls < 0 || cls >= num_classes) {
                throw std::out_of_range(
                    "one_hot: class index " + std::to_string(cls) +
                    " out of range [0, " + std::to_string(num_classes) + ")");
            }
        };
        int64_t n = idx_host.numel();
        if (indices.dtype() == DType::Int64) {
            const int64_t* p = idx_host.data<int64_t>();
            for (int64_t i = 0; i < n; ++i) check_range(p[i]);
        } else {
            const int32_t* p = idx_host.data<int32_t>();
            for (int64_t i = 0; i < n; ++i) check_range(static_cast<int64_t>(p[i]));
        }
    }

    // The one_hot shader reads int indices_data[] (32-bit), so convert Int64→Int32.
    // The host-side range check above guarantees every value fits in int32.
    Tensor indices_i32 = (indices.dtype() == DType::Int32) ? indices : indices.to(DType::Int32);

    int64_t batch_size = indices_i32.numel();
    Tensor output({batch_size, num_classes}, out_dtype, indices.device());

    const void* buf_indices = indices_i32.data_ptr();
    const void* buf_output = output.data_ptr();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_indices},
        {1, buf_output},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(batch_size) * indices_i32.dtype_size(),
        static_cast<size_t>(batch_size * num_classes) * sizeof(float),
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t num_classes;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.num_classes = static_cast<uint32_t>(num_classes);

    uint64_t total = static_cast<uint64_t>(batch_size) * num_classes;
    uint32_t workgroups = static_cast<uint32_t>(div_wg(total, devices_[device_id].workgroupSize));

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

    return output;
}

// Phase 3: Nonzero - GPU implementation (multi-pass: count, prefix_sum, gather)
auto VulkanBackend::dispatchNonzero(const Tensor& input) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = shape.size();
    int64_t numel = input.numel();

    if (numel == 0) {
        return Tensor({0, ndim}, DType::Int64, input.device());
    }

    int32_t device_id = input.device().index;
    uint32_t n = static_cast<uint32_t>(numel);
    uint32_t n_workgroups = div_wg(n, devices_[device_id].workgroupSize);

    // Ensure input is Float32 for the nonzero_count shader
    Tensor input_f32 = (input.dtype() == DType::Float32) ? input : input.to(DType::Float32);

    // Allocate flags buffer (one uint per element: 1=nonzero, 0=zero)
    Tensor flags({static_cast<int64_t>(n)}, DType::Int32, input.device());
    // Allocate count buffer (one per workgroup + space for total)
    Tensor count_buf({static_cast<int64_t>(n_workgroups + 1)}, DType::Int32, input.device());
    count_buf = dispatchFill(count_buf, 0.0f);

    const void* buf_input = input_f32.data_ptr();
    const void* buf_flags = flags.data_ptr();
    const void* buf_count = count_buf.data_ptr();
    size_t input_bytes = n * sizeof(float);
    size_t flags_bytes = n * sizeof(uint32_t);
    size_t count_bytes = (n_workgroups + 1) * sizeof(uint32_t);

    // ---- Pass 1a: Per-element flags + workgroup counts ----
    {
        auto* pipeline = getPipeline("nonzero_count", device_id);
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buf_input}, {1, buf_flags}, {2, buf_count}
        };
        std::vector<size_t> sizes = {input_bytes, flags_bytes, count_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t pass; } pc;
        pc.n_elements = n; pc.pass = 0;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // ---- Pass 1b: Reduce workgroup counts to get total ----
    {
        auto* pipeline = getPipeline("nonzero_count", device_id);
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buf_input}, {1, buf_flags}, {2, buf_count}
        };
        std::vector<size_t> sizes = {input_bytes, flags_bytes, count_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t pass; } pc;
        pc.n_elements = n; pc.pass = 1;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, 1, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Read back total count (GPU→CPU sync required for variable-size output)
    Tensor count_cpu = count_buf.to(Device::cpu());
    int64_t total_count = static_cast<int64_t>(count_cpu.data<int32_t>()[0]);

    if (total_count == 0) {
        return Tensor({0, ndim}, DType::Int64, input.device());
    }

    // ---- Pass 2: Prefix sum of flags ----
    // Reuse prefix_sum shader with flags as "mask" (mask_is_float=1 since flags are uint 0/1,
    // and uintBitsToFloat(0)==0.0, uintBitsToFloat(1)!=0.0)
    Tensor prefix_sums({static_cast<int64_t>(n)}, DType::Int32, input.device());
    prefix_sums = dispatchFill(prefix_sums, 0.0f);
    Tensor block_sums({static_cast<int64_t>(n_workgroups)}, DType::Int32, input.device());
    block_sums = dispatchFill(block_sums, 0.0f);

    const void* buf_prefix = prefix_sums.data_ptr();
    const void* buf_blocks = block_sums.data_ptr();
    size_t prefix_bytes = n * sizeof(uint32_t);
    size_t blocks_bytes = n_workgroups * sizeof(uint32_t);

    // Pass 2a: Local scan
    {
        auto* pipeline = getPipeline("prefix_sum", device_id);
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buf_flags}, {1, buf_prefix}, {2, buf_blocks}
        };
        std::vector<size_t> sizes = {flags_bytes, prefix_bytes, blocks_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t pass; } pc;
        pc.n_elements = n; pc.mask_is_float = 1; pc.pass = 0;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Pass 2b: Add block offsets
    if (n_workgroups > 1) {
        auto* pipeline = getPipeline("prefix_sum", device_id);
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buf_flags}, {1, buf_prefix}, {2, buf_blocks}
        };
        std::vector<size_t> sizes = {flags_bytes, prefix_bytes, blocks_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t pass; } pc;
        pc.n_elements = n; pc.mask_is_float = 1; pc.pass = 1;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // ---- Pass 3: Gather multi-dimensional indices ----
    // Output is (num_nonzero, ndim) Int32 on GPU, converted to Int64 at the end
    Tensor output_i32({total_count * ndim}, DType::Int32, input.device());

    // Upload shape to GPU buffer
    std::vector<uint32_t> shape_u32(ndim);
    for (int64_t d = 0; d < ndim; ++d) {
        shape_u32[d] = static_cast<uint32_t>(shape[d]);
    }
    Tensor shape_buf({ndim}, DType::Int32, input.device());
    copy(shape_buf.data_ptr(), shape_u32.data(), ndim * sizeof(uint32_t), CopyKind::HostToDevice);
    synchronize(device_id);

    {
        auto* pipeline = getPipeline("nonzero_gather", device_id);
        const void* buf_output = output_i32.data_ptr();
        const void* buf_shape = shape_buf.data_ptr();
        size_t output_bytes = total_count * ndim * sizeof(int32_t);
        size_t shape_bytes = ndim * sizeof(uint32_t);

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buf_flags}, {1, buf_prefix}, {2, buf_output}, {3, buf_shape}
        };
        std::vector<size_t> sizes = {flags_bytes, prefix_bytes, output_bytes, shape_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t ndim; uint32_t output_size; } pc;
        pc.n_elements = n; pc.ndim = static_cast<uint32_t>(ndim);
        pc.output_size = static_cast<uint32_t>(total_count);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Convert Int32 output to Int64 on GPU using cast shader
    auto* cast_pipeline = getPipeline("cast_i32_i64", device_id);
    Tensor result({total_count, ndim}, DType::Int64, input.device());
    int64_t total_elements = total_count * ndim;
    size_t in_bytes = total_elements * sizeof(int32_t);
    size_t out_bytes = total_elements * sizeof(int64_t);

    std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
        {0, output_i32.data_ptr()}, {1, result.data_ptr()}
    };
    std::vector<size_t> cast_sizes = {in_bytes, out_bytes};

    VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(
        device_id, cast_pipeline, cast_bindings, cast_sizes);

    struct { uint32_t n; } cast_pc;
    cast_pc.n = static_cast<uint32_t>(total_elements);

    VkCommandBuffer cast_cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cast_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
    vkCmdBindDescriptorSets(cast_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
    vkCmdPushConstants(cast_cmd, cast_pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
    vkCmdDispatch(cast_cmd, div_wg(total_elements, devices_[device_id].workgroupSize), 1, 1);
    insertComputeBarrier(cast_cmd);
    endSingleTimeCommands(cast_cmd, device_id);

    return result;
}

} // namespace tenzor
