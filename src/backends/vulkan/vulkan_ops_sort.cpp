/**
 * @file vulkan_ops_sort.cpp
 * @brief Vulkan backend sorting operations (Phase 11.4)
 */

#include "vulkan_ops_common.hpp"

namespace tenzor {

// ============================================================================
// Phase 11.4: Sorting Operations
// ============================================================================

/**
 * @brief GPU radix sort for large arrays (> 65K elements).
 * 3-pass per 8-bit digit: histogram, prefix sum, scatter. 4 passes for 32-bit, 8 for 64-bit.
 */
auto VulkanBackend::dispatchRadixSort(const Tensor& input, bool descending) -> std::pair<Tensor, Tensor> {
    int32_t device_id = input.device().index;
    int64_t n = input.numel();
    DType dtype = input.dtype();

    // Determine number of digit passes based on key width
    int num_passes = 4;  // default for 32-bit
    if (dtype == DType::Float64 || dtype == DType::Int64) num_passes = 8;
    if (dtype == DType::Float16) num_passes = 2;

    // Select shader variants based on dtype
    std::string hist_shader = "radix_histogram";
    std::string scatter_shader = "radix_scatter";
    if (dtype == DType::Float64) { hist_shader += "_f64"; scatter_shader += "_f64"; }
    else if (dtype == DType::Int32) { hist_shader += "_i32"; scatter_shader += "_i32"; }
    else if (dtype == DType::Int64) { hist_shader += "_i64"; scatter_shader += "_i64"; }
    else if (dtype == DType::Float16) { scatter_shader += "_f16"; }
    // Float32 uses base shader names

    // Number of workgroups for histogram/scatter
    uint32_t wg_size = devices_[device_id].workgroupSize;
    uint32_t n_wgs = div_wg(static_cast<uint32_t>(n), wg_size);
    if (n_wgs > 256) n_wgs = 256;  // cap workgroups to keep histogram matrix manageable

    size_t key_size = dtype_size(dtype);
    size_t key_buf_size = n * key_size;
    size_t idx_buf_size = n * sizeof(int32_t);
    size_t histo_size = 256 * n_wgs * sizeof(uint32_t);

    // Create ping-pong key/index buffers
    Tensor keys_a = input.contiguous();
    Tensor keys_b(std::vector<int64_t>{n}, dtype, input.device());
    Tensor idx_a({n}, DType::Int32, input.device());
    Tensor idx_b({n}, DType::Int32, input.device());
    Tensor histo_buf({static_cast<int64_t>(256 * n_wgs)}, DType::Int32, input.device());

    // Initialize indices to [0, 1, 2, ...]
    idx_a = dispatchArange(0.0f, static_cast<float>(n), 1.0f, DType::Int32, input.device());

    auto* hist_pipeline = getPipeline(hist_shader, device_id);
    auto* prefix_pipeline = getPipeline("radix_prefix_sum", device_id);
    auto* scatter_pipeline = getPipeline(scatter_shader, device_id);

    for (int digit = 0; digit < num_passes; ++digit) {
        // Pass 1: Histogram
        histo_buf = dispatchFill(histo_buf, 0.0f);
        {
            struct { uint32_t n; uint32_t digit; uint32_t n_wgs; } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.digit = descending ? (num_passes - 1 - digit) : digit;
            pc.n_wgs = n_wgs;

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, keys_a.data_ptr()}, {1, histo_buf.data_ptr()}
            };
            std::vector<size_t> sizes = {key_buf_size, histo_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, hist_pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hist_pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   hist_pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, hist_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, n_wgs, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Pass 2: Prefix sum over histogram matrix
        {
            struct { uint32_t total_entries; } pc;
            pc.total_entries = 256 * n_wgs;

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, histo_buf.data_ptr()}
            };
            std::vector<size_t> sizes = {histo_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, prefix_pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefix_pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   prefix_pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, prefix_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Pass 3: Scatter
        {
            struct { uint32_t n; uint32_t digit; uint32_t n_wgs; } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.digit = descending ? (num_passes - 1 - digit) : digit;
            pc.n_wgs = n_wgs;

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, keys_a.data_ptr()}, {1, idx_a.data_ptr()},
                {2, keys_b.data_ptr()}, {3, idx_b.data_ptr()},
                {4, histo_buf.data_ptr()}
            };
            std::vector<size_t> sizes = {key_buf_size, idx_buf_size, key_buf_size, idx_buf_size, histo_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, scatter_pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scatter_pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   scatter_pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, scatter_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, n_wgs, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        std::swap(keys_a, keys_b);
        std::swap(idx_a, idx_b);
    }

    // If descending, reverse the result
    if (descending) {
        // The sign-bit flip encoding produces ascending order; reverse for descending
        keys_a = dispatchFlip(keys_a, 0);
        idx_a = dispatchFlip(idx_a, 0);
    }

    // Convert Int32 indices to Int64
    Tensor indices_i64 = idx_a.to(DType::Int64);
    return {keys_a, indices_i64};
}

/**
 * @brief Full sort — uses bitonic sort shader, returns (sorted_values, indices).
 * Delegates to dispatchArgSort for the sorting mechanism, then gathers values.
 */
auto VulkanBackend::dispatchSort(const Tensor& input, int64_t dim, bool descending) -> std::pair<Tensor, Tensor> {
    auto input_shape = input.shape();
    const int ndim = static_cast<int>(input_shape.size());

    // Normalize dim
    if (dim < 0) dim += ndim;

    const int64_t sort_size = input_shape[dim];

    // Determine shader based on dtype
    std::string sort_shader;
    DType work_dtype = DType::Float32;
    size_t elem_size = sizeof(float);
    if (input.dtype() == DType::Float32) {
        sort_shader = "bitonic_sort";
    } else if (input.dtype() == DType::Float64) {
        sort_shader = "bitonic_sort_f64";
        work_dtype = DType::Float64;
        elem_size = sizeof(double);
    } else if (input.dtype() == DType::Int32) {
        sort_shader = "bitonic_sort_i32";
        work_dtype = DType::Int32;
        elem_size = sizeof(int32_t);
    } else if (input.dtype() == DType::Int64) {
        sort_shader = "bitonic_sort_i64";
        work_dtype = DType::Int64;
        elem_size = sizeof(int64_t);
    } else if (input.dtype() == DType::Float16) {
        sort_shader = "bitonic_sort_f16";
        work_dtype = DType::Float16;
        elem_size = sizeof(uint16_t);
    } else if (input.dtype() == DType::Int8 || input.dtype() == DType::UInt8 || input.dtype() == DType::Bool) {
        // Cast to Int32, sort, then cast sorted values back to original dtype
        DType orig_dtype = input.dtype();
        Tensor int32_input = input.to(DType::Int32);
        auto [sorted_i32, indices] = dispatchSort(int32_input, dim, descending);
        return {sorted_i32.to(orig_dtype), indices};
    } else if (input.dtype() == DType::BFloat16) {
        Tensor f32_input = input.to(DType::Float32);
        auto [sorted_f32, indices] = dispatchSort(f32_input, dim, descending);
        return {sorted_f32.to(DType::BFloat16), indices};
    } else {
        sort_shader = "";
    }

    if (sort_shader.empty()) {
        throw std::runtime_error(std::string("Vulkan: Sort not supported for dtype ") +
                                 std::string(dtype_name(input.dtype())));
    }

    // For large sorts (>2^24 elements along sort dim), use GPU radix sort
    // instead of bitonic sort which has O(n log^2 n) pass count
    if (sort_size > (1 << 24)) {
        // Non-last-dim: transpose so sort dim is last, sort, transpose back
        if (dim != ndim - 1) {
            std::vector<int64_t> perm(ndim);
            std::iota(perm.begin(), perm.end(), int64_t(0));
            std::swap(perm[dim], perm[ndim - 1]);

            std::vector<int64_t> inv_perm(ndim);
            for (int i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;

            Tensor transposed = dispatchContiguous(dispatchPermute(input, perm));
            auto [sorted_t, indices_t] = dispatchSort(transposed, ndim - 1, descending);

            return {dispatchContiguous(dispatchPermute(sorted_t, inv_perm)),
                    dispatchContiguous(dispatchPermute(indices_t, inv_perm))};
        }

        int64_t num_slices = 1;
        for (int i = 0; i < ndim - 1; ++i) num_slices *= input_shape[i];

        std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
        Tensor sorted_values(shape_vec, input.dtype(), input.device());
        Tensor sorted_indices(shape_vec, DType::Int64, input.device());

        size_t slice_val_bytes = sort_size * dtype_size(input.dtype());
        size_t slice_idx_bytes = sort_size * sizeof(int64_t);
        int32_t device_id = input.device().index;

        Tensor contig_input = input.contiguous();
        for (int64_t slice = 0; slice < num_slices; ++slice) {
            // Extract slice as a contiguous 1-D tensor
            Tensor slice_data({sort_size}, input.dtype(), input.device());
            copy(slice_data.data_ptr(),
                 static_cast<const char*>(contig_input.data_ptr()) + slice * slice_val_bytes,
                 slice_val_bytes, CopyKind::DeviceToDevice);
            synchronize(device_id);

            auto [sv, si] = dispatchRadixSort(slice_data, descending);

            // Copy sorted values and indices back into output tensors
            copy(static_cast<char*>(sorted_values.data_ptr()) + slice * slice_val_bytes,
                 sv.data_ptr(), slice_val_bytes, CopyKind::DeviceToDevice);
            copy(static_cast<char*>(sorted_indices.data_ptr()) + slice * slice_idx_bytes,
                 si.data_ptr(), slice_idx_bytes, CopyKind::DeviceToDevice);
            synchronize(device_id);
        }

        return {sorted_values, sorted_indices};
    }

    // Non-last-dim: transpose so sort dim is last, sort, transpose back
    if (dim != ndim - 1) {
        std::vector<int64_t> perm(ndim);
        std::iota(perm.begin(), perm.end(), int64_t(0));
        std::swap(perm[dim], perm[ndim - 1]);

        std::vector<int64_t> inv_perm(ndim);
        for (int i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;

        Tensor transposed = dispatchContiguous(dispatchPermute(input, perm));
        auto [sorted_t, indices_t] = dispatchSort(transposed, ndim - 1, descending);

        return {dispatchContiguous(dispatchPermute(sorted_t, inv_perm)),
                dispatchContiguous(dispatchPermute(indices_t, inv_perm))};
    }

    if (sort_size <= 1) {
        std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
        Tensor indices(shape_vec, DType::Int64, input.device());
        indices = dispatchFill(indices, 0.0f);
        return {dispatchClone(input), indices};
    }

    int32_t device_id = input.device().index;

    // Padded size (power of 2)
    uint32_t n = static_cast<uint32_t>(sort_size);
    uint32_t padded_n = 1;
    while (padded_n < n) padded_n <<= 1;

    uint32_t num_stages = 0;
    { uint32_t tmp = padded_n; while (tmp > 1) { num_stages++; tmp >>= 1; } }

    int64_t num_slices = 1;
    for (int i = 0; i < ndim - 1; ++i) num_slices *= input_shape[i];

    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    Tensor sorted_values(shape_vec, input.dtype(), input.device());
    Tensor sorted_indices(shape_vec, DType::Int64, input.device());

    Tensor work_values({static_cast<int64_t>(padded_n)}, work_dtype, input.device());
    Tensor work_indices({static_cast<int64_t>(padded_n)}, DType::Int32, input.device());

    float pad_value = descending ? -std::numeric_limits<float>::infinity()
                                 : std::numeric_limits<float>::infinity();
    if (work_dtype == DType::Int32) {
        // Don't use float intermediate: INT32_MAX (2^31-1) can't be represented
        // exactly as float and static_cast<int32_t> on the overflowing float is
        // UB (typically INT32_MIN), which breaks ascending int32 sorts.
        pad_value = 0.0f;  // filled directly below via CPU → device copy
    }

    size_t values_bytes = padded_n * elem_size;
    size_t indices_bytes = padded_n * sizeof(int32_t);

    auto* pipeline = getPipeline(sort_shader, device_id);
    uint32_t workgroups = div_wg(padded_n / 2, devices_[device_id].workgroupSize);

    std::vector<int32_t> init_indices(padded_n);
    for (uint32_t i = 0; i < padded_n; ++i) {
        init_indices[i] = (i < n) ? static_cast<int32_t>(i) : static_cast<int32_t>(n);
    }

    // Ensure input is contiguous on the GPU for D2D slice copies.
    Tensor input_contig = input.is_contiguous() ? input : dispatchContiguous(input);

    // Create a GPU-side pad template: fill once, then D2D copy each iteration.
    // Allocated once outside the loop to avoid repeated alloc/dealloc which
    // would trigger forced batch submits.
    Tensor pad_template({static_cast<int64_t>(padded_n)}, work_dtype, input.device());
    if (work_dtype == DType::Int32) {
        // Fill the int32 pad with exact INT32_MAX / INT32_MIN by preparing on
        // the host and uploading — the float-based fill path can't represent
        // INT32_MAX, and dispatchFill's float→int32 cast would overflow to
        // INT32_MIN.
        std::vector<int32_t> pad_host(padded_n,
            descending ? std::numeric_limits<int32_t>::min()
                       : std::numeric_limits<int32_t>::max());
        copy(pad_template.data_ptr(), pad_host.data(),
             padded_n * sizeof(int32_t), CopyKind::HostToDevice);
    } else if (work_dtype == DType::Int64) {
        // Same rationale as Int32 plus dispatchFill's uint32 fill shader only
        // writes the low 4 bytes of each 8-byte int64 slot — the high half stays
        // whatever the allocator left behind, corrupting sort comparisons.
        std::vector<int64_t> pad_host(padded_n,
            descending ? std::numeric_limits<int64_t>::min()
                       : std::numeric_limits<int64_t>::max());
        copy(pad_template.data_ptr(), pad_host.data(),
             padded_n * sizeof(int64_t), CopyKind::HostToDevice);
    } else {
        pad_template = dispatchFill(pad_template, pad_value);
    }

    // Pre-allocate the int64 cast tensor outside the loop.  Allocating it
    // inside would destroy the old one each iteration, which calls deallocate
    // -> submitBatchIfNeeded(force=true), splitting operations across command
    // buffers.  The descriptor set must still be re-allocated per iteration
    // because synchronize() resets the descriptor pool.
    auto* cast_pipeline = getPipeline("cast_i32_i64", device_id);
    Tensor int64_chunk({sort_size}, DType::Int64, input.device());
    size_t cast_out_bytes = sort_size * sizeof(int64_t);
    struct { uint32_t n; } cast_pc;
    cast_pc.n = static_cast<uint32_t>(sort_size);

    for (int64_t slice = 0; slice < num_slices; ++slice) {
        size_t slice_bytes = sort_size * elem_size;

        // GPU-side fill: copy the pre-filled pad template into work_values
        copy(work_values.data_ptr(), pad_template.data_ptr(),
             padded_n * elem_size, CopyKind::DeviceToDevice);

        // GPU-side slice copy: overlay this slice's data into work_values
        const void* slice_src = static_cast<const char*>(input_contig.data_ptr())
                                + slice * slice_bytes;
        copy(work_values.data_ptr(), slice_src,
             slice_bytes, CopyKind::DeviceToDevice);

        copy(work_indices.data_ptr(), init_indices.data(),
             padded_n * sizeof(int32_t), CopyKind::HostToDevice);
        synchronize(device_id);

        // Run all bitonic sort passes
        {
            const void* buffer_values = work_values.data_ptr();
            const void* buffer_indices = work_indices.data_ptr();
            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, buffer_values}, {1, buffer_indices}
            };
            std::vector<size_t> sizes = {values_bytes, indices_bytes};
            VkDescriptorSet sort_ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);

            for (uint32_t stage = 0; stage < num_stages; ++stage) {
                for (int32_t substage = static_cast<int32_t>(stage); substage >= 0; --substage) {
                    struct {
                        uint32_t n;
                        uint32_t padded_n;
                        uint32_t stage;
                        uint32_t substage;
                        uint32_t descending;
                    } pc;
                    pc.n = n;
                    pc.padded_n = padded_n;
                    pc.stage = stage;
                    pc.substage = static_cast<uint32_t>(substage);
                    pc.descending = descending ? 1 : 0;

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                           pipeline->layout(), 0, 1, &sort_ds, 0, nullptr);
                    vkCmdPushConstants(cmd, pipeline->layout(),
                                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    vkCmdDispatch(cmd, workgroups, 1, 1);
                    insertComputeBarrier(cmd);
                }
            }

            endSingleTimeCommands(cmd, device_id);
            synchronize(device_id);
        }

        // Read sorted values and indices
        {
            // Copy sorted values — use vkCmdCopyBuffer directly with known
            // VkBuffer + offset to avoid getVulkanBufferAndOffset on offset
            // pointers, which can resolve to the wrong slab sub-block.
            {
                auto [sv_buf, sv_off] = getVulkanBufferAndOffset(sorted_values.data_ptr());
                auto [wv_buf, wv_off] = getVulkanBufferAndOffset(work_values.data_ptr());
                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                VkBufferCopy region{};
                region.srcOffset = wv_off;
                region.dstOffset = sv_off + slice * slice_bytes;
                region.size = slice_bytes;
                vkCmdCopyBuffer(cmd, wv_buf, sv_buf, 1, &region);
                insertTransferToComputeBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }

            // Convert Int32 indices to Int64 and copy to output
            {
                std::vector<std::pair<uint32_t, const void*>> cb = {
                    {0, work_indices.data_ptr()}, {1, int64_chunk.data_ptr()}
                };
                std::vector<size_t> cs = {sort_size * sizeof(int32_t), cast_out_bytes};
                VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cb, cs);

                VkCommandBuffer cast_cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cast_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
                vkCmdBindDescriptorSets(cast_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
                vkCmdPushConstants(cast_cmd, cast_pipeline->layout(),
                                  VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
                vkCmdDispatch(cast_cmd, div_wg(sort_size, devices_[device_id].workgroupSize), 1, 1);
                insertComputeBarrier(cast_cmd);
                endSingleTimeCommands(cast_cmd, device_id);
                synchronize(device_id);

                // Copy int64 indices to sorted_indices at the slice offset —
                // use vkCmdCopyBuffer directly to avoid offset pointer lookup.
                {
                    auto [si_buf, si_off] = getVulkanBufferAndOffset(sorted_indices.data_ptr());
                    auto [ic_buf, ic_off] = getVulkanBufferAndOffset(int64_chunk.data_ptr());
                    VkCommandBuffer icmd = beginSingleTimeCommands(device_id);
                    VkBufferCopy iregion{};
                    iregion.srcOffset = ic_off;
                    iregion.dstOffset = si_off + slice * sort_size * sizeof(int64_t);
                    iregion.size = cast_out_bytes;
                    vkCmdCopyBuffer(icmd, ic_buf, si_buf, 1, &iregion);
                    insertTransferToComputeBarrier(icmd);
                    endSingleTimeCommands(icmd, device_id);
                }
                synchronize(device_id);
            }
        }
    }

    return {sorted_values, sorted_indices};
}

/**
 * @brief TopK — sort then take first K elements.
 */
auto VulkanBackend::dispatchTopK(const Tensor& input, int64_t k, int64_t dim,
                                   bool largest, [[maybe_unused]] bool sorted) -> std::pair<Tensor, Tensor> {
    auto input_shape = input.shape();
    const int ndim = static_cast<int>(input_shape.size());
    if (dim < 0) dim += ndim;

    // Full sort descending for largest, ascending for smallest
    auto [sorted_values, sorted_indices] = dispatchSort(input, dim, largest);

    // Take first K along the sort dimension
    // Use slice along dim
    std::vector<int64_t> starts(ndim, 0);
    std::vector<int64_t> ends(input_shape.begin(), input_shape.end());
    std::vector<int64_t> steps(ndim, 1);
    ends[dim] = k;

    Tensor topk_values = dispatchSlice(sorted_values, starts, ends, steps);
    Tensor topk_indices = dispatchSlice(sorted_indices, starts, ends, steps);

    return {dispatchContiguous(topk_values), dispatchContiguous(topk_indices)};
}

/**
 * @brief Median — sort along dim, extract middle element.
 *
 * Returns {values, indices} where values[i] is the median of the i-th slice
 * and indices[i] is its position in the original slice.
 */
auto VulkanBackend::dispatchMedian(const Tensor& input, int64_t dim, bool keepdim) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    const int ndim = static_cast<int>(input_shape.size());

    // Normalize dim
    if (dim < 0) dim += ndim;

    const int64_t dim_size = input_shape[dim];

    // Edge case: empty tensor
    if (input.numel() == 0 || dim_size == 0) {
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
        out_shape[dim] = 0;
        if (!keepdim) {
            out_shape.erase(out_shape.begin() + dim);
        }
        return {Tensor(out_shape, input.dtype(), input.device()),
                Tensor(out_shape, DType::Int64, input.device())};
    }

    // Sort along dim (ascending). Float16/BFloat16 are sorted as Float32: the
    // sort comparator path does not order reduced-precision values correctly,
    // which produced a wrong sorted order and therefore the wrong median index
    // (the IndexNotPlaceholderZero regression). Widening preserves the original
    // slice positions in sorted_indices; the median value is narrowed back to
    // the input dtype below.
    const bool half_input = (input.dtype() == DType::Float16 ||
                             input.dtype() == DType::BFloat16);
    Tensor sort_input = half_input ? dispatchCast(input, DType::Float32) : input;
    auto [sorted_values, sorted_indices] = dispatchSort(sort_input, dim, false);

    // Median index: N/2 for even-length (lower median), (N-1)/2 same thing
    int64_t median_idx = (dim_size - 1) / 2;

    // Extract the median element using index_select along dim
    Tensor idx_tensor = dispatchFull({1}, static_cast<float>(median_idx), DType::Int64);

    Tensor median_values = dispatchIndexSelect(sorted_values, dim, idx_tensor);
    Tensor median_indices = dispatchIndexSelect(sorted_indices, dim, idx_tensor);
    if (half_input) {
        median_values = dispatchCast(median_values, input.dtype());
    }

    // Squeeze the dim (index_select keeps dim with size 1)
    if (!keepdim) {
        // Remove the dimension
        std::vector<int64_t> out_shape;
        auto med_shape = median_values.shape();
        for (int i = 0; i < ndim; ++i) {
            if (i != dim) out_shape.push_back(med_shape[i]);
        }
        if (out_shape.empty()) out_shape.push_back(1);  // scalar
        median_values = dispatchReshape(median_values, out_shape);
        median_indices = dispatchReshape(median_indices, out_shape);
    }

    return {dispatchContiguous(median_values), dispatchContiguous(median_indices)};
}

/**
 * @brief Mode — sort along dim, then find longest run of equal values.
 *
 * Uses dispatchSort to sort data, then launches mode.comp shader to find
 * the most frequent element in each sorted slice.
 * Returns {values, indices}.
 */
auto VulkanBackend::dispatchMode(const Tensor& input, int64_t dim, bool keepdim) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    const int ndim = static_cast<int>(input_shape.size());

    // Normalize dim
    if (dim < 0) dim += ndim;

    const int64_t dim_size = input_shape[dim];

    // Edge case: empty tensor
    if (input.numel() == 0 || dim_size == 0) {
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
        if (out_shape.empty()) out_shape.push_back(1);
        return {Tensor(out_shape, input.dtype(), input.device()),
                Tensor(out_shape, DType::Int64, input.device())};
    }

    // Size-1 dim: mode is the element itself
    if (dim_size == 1) {
        if (keepdim) {
            Tensor indices_out(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               DType::Int64, input.device());
            // Zero-fill indices
            memset(indices_out.data_ptr(), 0,
                   static_cast<size_t>(indices_out.numel()) * sizeof(int64_t),
                   input.device().index);
            return {input.contiguous(), indices_out};
        } else {
            std::vector<int64_t> out_shape;
            for (int i = 0; i < ndim; ++i) {
                if (i != dim) out_shape.push_back(input_shape[i]);
            }
            if (out_shape.empty()) out_shape.push_back(1);
            Tensor vals = dispatchReshape(input, out_shape);
            Tensor indices_out(out_shape, DType::Int64, input.device());
            memset(indices_out.data_ptr(), 0,
                   static_cast<size_t>(indices_out.numel()) * sizeof(int64_t),
                   input.device().index);
            return {dispatchContiguous(vals), indices_out};
        }
    }

    // Sort along dim (ascending)
    auto [sorted_values, sorted_indices] = dispatchSort(input, dim, false);

    // Transpose so that dim is the last dimension, then make contiguous
    Tensor sorted_contig, indices_contig;
    std::vector<int64_t> inv_perm;
    if (dim != ndim - 1) {
        std::vector<int64_t> perm(ndim);
        std::iota(perm.begin(), perm.end(), int64_t(0));
        std::swap(perm[dim], perm[ndim - 1]);

        inv_perm.resize(ndim);
        for (int i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;

        sorted_contig = dispatchContiguous(dispatchPermute(sorted_values, perm));
        indices_contig = dispatchContiguous(dispatchPermute(sorted_indices, perm));
    } else {
        sorted_contig = dispatchContiguous(sorted_values);
        indices_contig = dispatchContiguous(sorted_indices);
    }

    // Compute number of slices = product of all dims except last
    auto sc_shape = sorted_contig.shape();
    int64_t num_slices = 1;
    for (int i = 0; i < ndim - 1; ++i) num_slices *= sc_shape[i];
    int64_t slice_size = sc_shape[ndim - 1];

    // Determine shader based on dtype
    std::string shader_name;
    DType work_dtype = input.dtype();
    bool needs_cast = false;
    DType orig_dtype = input.dtype();

    if (work_dtype == DType::Float32) {
        shader_name = "mode";
    } else if (work_dtype == DType::Float64) {
        shader_name = "mode_f64";
    } else {
        // For other dtypes, cast to Float32
        needs_cast = true;
        work_dtype = DType::Float32;
        shader_name = "mode";
        sorted_contig = sorted_contig.to(DType::Float32);
    }

    int32_t device_id = input.device().index;

    // Output tensors: one value and one index per slice
    Tensor mode_values({num_slices}, work_dtype, input.device());
    Tensor mode_indices_flat({num_slices}, DType::Int32, input.device());

    // Launch mode shader
    {
        auto* pipeline = getPipeline(shader_name, device_id);
        size_t sorted_bytes = static_cast<size_t>(num_slices * slice_size) * dtype_size(work_dtype);
        size_t values_bytes = static_cast<size_t>(num_slices) * dtype_size(work_dtype);
        size_t indices_bytes = static_cast<size_t>(num_slices) * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, sorted_contig.data_ptr()},
            {1, mode_values.data_ptr()},
            {2, mode_indices_flat.data_ptr()}
        };
        std::vector<size_t> sizes = {sorted_bytes, values_bytes, indices_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t slice_size; uint32_t num_slices; } pc;
        pc.slice_size = static_cast<uint32_t>(slice_size);
        pc.num_slices = static_cast<uint32_t>(num_slices);

        uint32_t wg = static_cast<uint32_t>(div_wg(num_slices, devices_[device_id].workgroupSize));
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, wg, 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    // GPU index remapping: compute flat_idx = arange(num_slices) * slice_size + mode_indices_flat
    // then gather: orig_indices = sorted_indices.flatten().index_select(0, flat_idx)
    // No CPU roundtrip needed — all done with existing GPU dispatch ops.
    Tensor slice_offsets = dispatchArange(0.0f, static_cast<float>(num_slices), 1.0f,
                                          DType::Int32, input.device());
    Tensor stride_tensor = dispatchFull({num_slices}, static_cast<float>(slice_size), DType::Int32);
    stride_tensor = stride_tensor.to(input.device());
    Tensor flat_idx = dispatchBinaryOp("add",
                         dispatchBinaryOp("mul", slice_offsets, stride_tensor),
                         mode_indices_flat);
    // Cast to Int64 to match sorted index dtype
    flat_idx = flat_idx.to(DType::Int64);
    Tensor flat_sort_indices = dispatchReshape(indices_contig, {num_slices * slice_size});
    Tensor orig_indices = dispatchIndexSelect(flat_sort_indices, 0, flat_idx);

    // Cast values back if needed
    if (needs_cast) {
        mode_values = mode_values.to(orig_dtype);
    }

    // Reshape to proper output shape
    std::vector<int64_t> out_shape;
    if (keepdim) {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        out_shape[dim] = 1;
    } else {
        for (int i = 0; i < ndim; ++i) {
            if (i != dim) out_shape.push_back(input_shape[i]);
        }
        if (out_shape.empty()) out_shape.push_back(1);
    }

    mode_values = dispatchReshape(mode_values, out_shape);
    orig_indices = dispatchReshape(orig_indices, out_shape);

    return {dispatchContiguous(mode_values), dispatchContiguous(orig_indices)};
}

/**
 * @brief Unique — sort on GPU, compact on host, transfer back.
 *
 * Hybrid approach: uses Vulkan bitonic sort on device, then reads back sorted
 * data for O(n) dedup on host. This avoids the full input D2H + CPU sort +
 * H2D roundtrip of a pure CPU fallback.
 */
auto VulkanBackend::dispatchUnique(const Tensor& input, bool sorted,
                                     bool return_inverse, bool return_counts) -> std::vector<Tensor> {
    int64_t numel = input.numel();
    if (numel == 0) {
        Tensor empty_vals({0}, input.dtype(), input.device());
        Tensor empty_inv({0}, DType::Int64, input.device());
        Tensor empty_cnt({0}, DType::Int64, input.device());
        return {empty_vals, empty_inv, empty_cnt};
    }

    // F20: Int8/UInt8/Bool AND Int16/UInt16 — cast to Int32, run unique on
    // GPU, cast unique values back. Previously only Int8/UInt8/Bool were
    // cast; Int16/UInt16 fell through to the dtype reject below and threw
    // "Unique not supported for dtype Int16". The sort + mark/compact
    // pipeline only has Int32/Int64/F32/F64 shaders, so the int16/uint16
    // widen-narrow is the honest way to support them without a new shader
    // pair.
    if (input.dtype() == DType::Int8 || input.dtype() == DType::UInt8 ||
        input.dtype() == DType::Int16 || input.dtype() == DType::UInt16 ||
        input.dtype() == DType::Bool) {
        DType orig_dtype = input.dtype();
        Tensor int32_input = input.to(DType::Int32);
        auto results = dispatchUnique(int32_input, sorted, return_inverse, return_counts);
        // Cast unique values back to original dtype; inverse/counts stay as Int64
        results[0] = results[0].to(orig_dtype);
        return results;
    }

    // Flatten input
    Tensor flat = dispatchContiguous(dispatchReshape(input, {numel}));

    // Sort on GPU using existing bitonic sort
    auto [sorted_vals, sorted_indices] = dispatchSort(flat, 0, false);

    int32_t device_id = input.device().index;

    // Select mark/compact shaders based on dtype
    std::string mark_shader, compact_shader;
    DType sorted_dtype = sorted_vals.dtype();
    // For F16/BF16: cast sorted values to F32 so F32 mark/compact shaders work correctly
    if (sorted_dtype == DType::Float16 || sorted_dtype == DType::BFloat16) {
        sorted_vals = sorted_vals.to(DType::Float32);
        sorted_dtype = DType::Float32;
    }
    if (sorted_dtype == DType::Float64) {
        mark_shader = "unique_mark_f64";
        compact_shader = "unique_compact_f64";
    } else if (sorted_dtype == DType::Int32) {
        mark_shader = "unique_mark_i32";
        compact_shader = "unique_compact_i32";
    } else if (sorted_dtype == DType::Int64) {
        mark_shader = "unique_mark_i64";
        compact_shader = "unique_compact_i64";
    } else {
        // F32 (default; F16/BF16 are already cast to F32 above)
        mark_shader = "unique_mark";
        compact_shader = "unique_compact";
    }

    // For types without native shaders, fall back to host dedup
    bool has_gpu_shader = (sorted_dtype == DType::Float32 || sorted_dtype == DType::Float64 ||
                           sorted_dtype == DType::Int32 || sorted_dtype == DType::Int64);

    if (!has_gpu_shader) {
        throw std::runtime_error(std::string("Vulkan: Unique not supported for dtype ") +
                                 std::string(dtype_name(sorted_dtype)));
    }

    // Step 1: Mark boundaries on GPU — boundary[i] = 1 if sorted[i] != sorted[i-1]
    Tensor boundaries({numel}, DType::Int32, input.device());
    {
        auto* pipeline = getPipeline(mark_shader, device_id);
        size_t sorted_bytes = numel * dtype_size(sorted_dtype);
        size_t boundary_bytes = numel * sizeof(uint32_t);
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, sorted_vals.data_ptr()}, {1, boundaries.data_ptr()}
        };
        std::vector<size_t> sizes = {sorted_bytes, boundary_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        struct { uint32_t numel; } pc;
        pc.numel = static_cast<uint32_t>(numel);
        uint32_t wg = static_cast<uint32_t>(div_wg(numel, devices_[device_id].workgroupSize));
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, wg, 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    // Hybrid GPU+CPU: GPU sorts and detects boundaries, host handles variable-size output compaction
    // Step 2: Prefix sum on boundary flags (gives compacted positions)
    Tensor prefix_sum = dispatchCumSum(boundaries, 0);

    // Step 3: Read n_unique from last element of prefix_sum (single int32 scalar readback,
    // not a CPU computation fallback — minimum sync required for variable-size output allocation)
    synchronize(device_id);
    Tensor last_elem = prefix_sum.slice(0, numel - 1, numel).to(Device::cpu());
    int32_t n_unique_i32 = last_elem.data<int32_t>()[0];
    int64_t n_unique = static_cast<int64_t>(n_unique_i32);

    // Step 4: Compact unique values on GPU
    Tensor out_vals({n_unique}, sorted_dtype, input.device());
    {
        auto* pipeline = getPipeline(compact_shader, device_id);
        size_t sorted_bytes = numel * dtype_size(sorted_dtype);
        size_t prefix_bytes = numel * sizeof(uint32_t);
        size_t output_bytes = n_unique * dtype_size(sorted_dtype);
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, sorted_vals.data_ptr()}, {1, prefix_sum.data_ptr()}, {2, out_vals.data_ptr()}
        };
        std::vector<size_t> sizes = {sorted_bytes, prefix_bytes, output_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        struct { uint32_t numel; } pc;
        pc.numel = static_cast<uint32_t>(numel);
        uint32_t wg = static_cast<uint32_t>(div_wg(numel, devices_[device_id].workgroupSize));
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, wg, 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    // Cast back to original dtype if sorted values were upcast (F16/BF16)
    if (sorted_dtype != input.dtype()) {
        out_vals = out_vals.to(input.dtype());
    }

    // GPU inverse mapping: inverse[sorted_indices[i]] = prefix_sum[i] - 1
    // Uses unique_inverse shader to scatter group IDs to original positions.
    Tensor out_inverse({return_inverse ? numel : int64_t(0)}, DType::Int64, input.device());
    if (return_inverse) {
        auto* inv_pipeline = getPipeline("unique_inverse", device_id);
        size_t idx_bytes = numel * sizeof(int64_t);
        size_t ps_bytes = numel * sizeof(int32_t);
        size_t inv_bytes = numel * sizeof(int64_t);
        std::vector<std::pair<uint32_t, const void*>> inv_bindings = {
            {0, sorted_indices.data_ptr()}, {1, prefix_sum.data_ptr()}, {2, out_inverse.data_ptr()}
        };
        std::vector<size_t> inv_sizes = {idx_bytes, ps_bytes, inv_bytes};
        VkDescriptorSet inv_ds = allocateAndWriteDescriptorSet(device_id, inv_pipeline, inv_bindings, inv_sizes);
        struct { uint32_t numel; } inv_pc;
        inv_pc.numel = static_cast<uint32_t>(numel);
        uint32_t inv_wg = static_cast<uint32_t>(div_wg(numel, devices_[device_id].workgroupSize));
        VkCommandBuffer inv_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(inv_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, inv_pipeline->pipeline());
        vkCmdBindDescriptorSets(inv_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               inv_pipeline->layout(), 0, 1, &inv_ds, 0, nullptr);
        vkCmdPushConstants(inv_cmd, inv_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(inv_pc), &inv_pc);
        vkCmdDispatch(inv_cmd, inv_wg, 1, 1);
        insertComputeOnlyBarrier(inv_cmd);
        endSingleTimeCommands(inv_cmd, device_id);
    }

    // GPU counts: scatter boundary positions, then compute differences.
    // boundary_pos[group] = position of group start. counts[g] = boundary_pos[g+1] - boundary_pos[g],
    // with a sentinel at n_unique = numel.
    Tensor out_counts({return_counts ? n_unique : int64_t(0)}, DType::Int64, input.device());
    if (return_counts && n_unique > 0) {
        // Step 1: Scatter boundary positions into boundary_pos (size n_unique + 1, last = numel)
        Tensor boundary_pos({n_unique + 1}, DType::Int64, input.device());
        {
            auto* cnt_pipeline = getPipeline("unique_counts", device_id);
            size_t bnd_bytes = numel * sizeof(int32_t);
            size_t ps_bytes = numel * sizeof(int32_t);
            size_t bp_bytes = (n_unique + 1) * sizeof(int64_t);
            std::vector<std::pair<uint32_t, const void*>> cnt_bindings = {
                {0, boundaries.data_ptr()}, {1, prefix_sum.data_ptr()}, {2, boundary_pos.data_ptr()}
            };
            std::vector<size_t> cnt_sizes = {bnd_bytes, ps_bytes, bp_bytes};
            VkDescriptorSet cnt_ds = allocateAndWriteDescriptorSet(device_id, cnt_pipeline, cnt_bindings, cnt_sizes);
            struct { uint32_t numel; uint32_t n_unique; } cnt_pc;
            cnt_pc.numel = static_cast<uint32_t>(numel);
            cnt_pc.n_unique = static_cast<uint32_t>(n_unique);
            uint32_t cnt_wg = static_cast<uint32_t>(div_wg(numel, devices_[device_id].workgroupSize));
            VkCommandBuffer cnt_cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cnt_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cnt_pipeline->pipeline());
            vkCmdBindDescriptorSets(cnt_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   cnt_pipeline->layout(), 0, 1, &cnt_ds, 0, nullptr);
            vkCmdPushConstants(cnt_cmd, cnt_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(cnt_pc), &cnt_pc);
            vkCmdDispatch(cnt_cmd, cnt_wg, 1, 1);
            insertComputeOnlyBarrier(cnt_cmd);
            endSingleTimeCommands(cnt_cmd, device_id);
        }

        // Step 2: Set sentinel: boundary_pos[n_unique] = numel
        // Write the trailing int64 directly via a host→device copy. The older
        // dispatchScatter path misbehaved here because there's no native int64
        // scatter shader — scatter silently reinterpreted the int64 buffer
        // as float32 and clobbered the wrong word.
        {
            int64_t sentinel_val = static_cast<int64_t>(numel);
            auto* bp_base = static_cast<char*>(boundary_pos.data_ptr());
            void* slot = bp_base + n_unique * sizeof(int64_t);
            copy(slot, &sentinel_val, sizeof(int64_t), CopyKind::HostToDevice);
        }

        // Step 3: Compute counts as differences of consecutive boundary positions on GPU
        Tensor bp_starts = dispatchSlice(boundary_pos, {0}, {n_unique}, {1});
        Tensor bp_ends = dispatchSlice(boundary_pos, {1}, {n_unique + 1}, {1});
        out_counts = dispatchBinaryOp("sub", bp_ends, bp_starts);
    }

    return {out_vals, out_inverse, out_counts};
}

} // namespace tenzor
