#include "vulkan_ops_common.hpp"

namespace tenzor {

// ============================================================================
// Stack/Take/Tile/Put Operations (native Vulkan shaders)
// ============================================================================

auto VulkanBackend::dispatchStack(std::span<const Tensor> inputs, int64_t dim) -> Tensor {
    if (inputs.empty()) {
        throw std::invalid_argument("stack requires at least one tensor");
    }

    auto first_shape = inputs[0].shape();
    int64_t ndim = static_cast<int64_t>(first_shape.size());

    // Handle negative dim
    if (dim < 0) dim = ndim + 1 + dim;
    if (dim < 0 || dim > ndim) {
        throw std::invalid_argument("stack dim out of range");
    }

    // BFloat16/Float16: use native packed shader
    if (inputs[0].dtype() == DType::Float16 || inputs[0].dtype() == DType::BFloat16) {
        bool is_bf16_stack = (inputs[0].dtype() == DType::BFloat16);
        int32_t device_id = inputs[0].device().index;
        DType dtype = inputs[0].dtype();

        int64_t num_tensors_i = static_cast<int64_t>(inputs.size());
        int64_t elements_per_tensor_i = inputs[0].numel();

        std::vector<int64_t> out_shape_f16;
        for (int64_t d = 0; d < ndim; ++d) {
            if (d == dim) out_shape_f16.push_back(num_tensors_i);
            out_shape_f16.push_back(first_shape[d]);
        }
        if (dim == ndim) out_shape_f16.push_back(num_tensors_i);

        int64_t output_numel_f16 = num_tensors_i * elements_per_tensor_i;

        auto* pipeline = getPipeline(is_bf16_stack ? "stack_bf16" : "stack_f16", device_id);
        Tensor output(out_shape_f16, dtype, inputs[0].device());

        // Copy all input tensors into a single contiguous buffer
        size_t elem_bytes = inputs[0].dtype_size();
        Tensor concat_input({output_numel_f16}, dtype, inputs[0].device());

        auto [dst_vk_buffer, dst_base_offset] = getVulkanBufferAndOffset(concat_input.data_ptr());
        {
            VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
            for (int64_t t = 0; t < num_tensors_i; ++t) {
                auto src = inputs[t].is_contiguous() ? inputs[t] : inputs[t].contiguous();
                auto [src_vk_buffer, src_offset] = getVulkanBufferAndOffset(src.data_ptr());
                size_t chunk_bytes = static_cast<size_t>(elements_per_tensor_i) * elem_bytes;
                VkBufferCopy copyRegion{};
                copyRegion.srcOffset = static_cast<VkDeviceSize>(src_offset);
                copyRegion.dstOffset = static_cast<VkDeviceSize>(dst_base_offset)
                                     + static_cast<VkDeviceSize>(t * elements_per_tensor_i) * elem_bytes;
                copyRegion.size = chunk_bytes;
                vkCmdCopyBuffer(cmdBuffer, src_vk_buffer, dst_vk_buffer, 1, &copyRegion);
            }
            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &barrier, 0, nullptr, 0, nullptr);
            endSingleTimeCommands(cmdBuffer, device_id);
        }

        // Match the standard stack.comp push-constant layout — outer_size
        // and inner_size let the shader handle arbitrary `dim`, not just 0.
        uint32_t outer_size_p = 1;
        for (int64_t d = 0; d < dim; ++d) outer_size_p *= static_cast<uint32_t>(first_shape[d]);
        uint32_t inner_size_p = static_cast<uint32_t>(elements_per_tensor_i) / std::max<uint32_t>(outer_size_p, 1u);
        struct { uint32_t num_tensors; uint32_t elements_per_tensor; uint32_t output_numel; uint32_t outer_size; uint32_t inner_size; } pc;
        pc.num_tensors = static_cast<uint32_t>(num_tensors_i);
        pc.elements_per_tensor = static_cast<uint32_t>(elements_per_tensor_i);
        pc.output_numel = static_cast<uint32_t>(output_numel_f16);
        pc.outer_size = outer_size_p;
        pc.inner_size = inner_size_p;

        // F16 packed buffer sizes: round up to 4-byte boundaries
        size_t in_buf_size = (static_cast<size_t>(output_numel_f16) + 1) / 2 * 4;
        size_t out_buf_size = in_buf_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, concat_input.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_buf_size, out_buf_size};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(output_numel_f16, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        return output;
    }

    int32_t device_id = inputs[0].device().index;
    bool is_float64 = (inputs[0].dtype() == DType::Float64);
    DType dtype = inputs[0].dtype();

    if (is_float64) {
        vulkan::ensure_fp64_supported(device_id, "Stack");
    }

    int64_t num_tensors = static_cast<int64_t>(inputs.size());
    int64_t elements_per_tensor = inputs[0].numel();

    // Build output shape: insert num_tensors at dim
    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d == dim) out_shape.push_back(num_tensors);
        out_shape.push_back(first_shape[d]);
    }
    if (dim == ndim) out_shape.push_back(num_tensors);

    int64_t output_numel = num_tensors * elements_per_tensor;

    std::string shader_name = is_float64 ? "stack_f64" : "stack";
    auto* pipeline = getPipeline(shader_name, device_id);

    Tensor output(out_shape, dtype, inputs[0].device());

    // Copy all input tensors into a single contiguous buffer using vkCmdCopyBuffer
    size_t elem_bytes = inputs[0].dtype_size();
    size_t total_input_bytes = static_cast<size_t>(output_numel) * elem_bytes;
    Tensor concat_input({output_numel}, dtype, inputs[0].device());

    auto [dst_vk_buffer, dst_base_offset] = getVulkanBufferAndOffset(concat_input.data_ptr());

    {
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        for (int64_t t = 0; t < num_tensors; ++t) {
            auto src = inputs[t].is_contiguous() ? inputs[t] : inputs[t].contiguous();
            auto [src_vk_buffer, src_offset] = getVulkanBufferAndOffset(src.data_ptr());
            size_t chunk_bytes = static_cast<size_t>(elements_per_tensor) * elem_bytes;

            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = static_cast<VkDeviceSize>(src_offset);
            copyRegion.dstOffset = static_cast<VkDeviceSize>(dst_base_offset)
                                 + static_cast<VkDeviceSize>(t * elements_per_tensor) * elem_bytes;
            copyRegion.size = chunk_bytes;
            vkCmdCopyBuffer(cmdBuffer, src_vk_buffer, dst_vk_buffer, 1, &copyRegion);
        }

        // Barrier to ensure copies complete before compute shader reads
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmdBuffer,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &barrier, 0, nullptr, 0, nullptr);

        endSingleTimeCommands(cmdBuffer, device_id);
    }

    // outer_size = product of input dims before `dim`;
    // inner_size = product of input dims from `dim` onward.
    // Together with num_tensors and elements_per_tensor they let the shader
    // map any output index to the right (tensor, outer, inner) triple for
    // an arbitrary stack axis.
    uint32_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) outer_size *= static_cast<uint32_t>(first_shape[d]);
    uint32_t inner_size = static_cast<uint32_t>(elements_per_tensor) / outer_size;

    struct {
        uint32_t num_tensors;
        uint32_t elements_per_tensor;
        uint32_t output_numel;
        uint32_t outer_size;
        uint32_t inner_size;
    } pushConstants;
    pushConstants.num_tensors = static_cast<uint32_t>(num_tensors);
    pushConstants.elements_per_tensor = static_cast<uint32_t>(elements_per_tensor);
    pushConstants.output_numel = static_cast<uint32_t>(output_numel);
    pushConstants.outer_size = outer_size;
    pushConstants.inner_size = inner_size;

    const void* buffer_in = concat_input.data_ptr();
    const void* buffer_out = output.data_ptr();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {total_input_bytes, total_input_bytes};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(output_numel, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchTake(const Tensor& input, const Tensor& indices) -> Tensor {
    if (indices.numel() == 0) {
        return Tensor({0}, input.dtype(), input.device());
    }

    // All take shaders read `int indices[]` (32-bit). If we get Int64 on the
    // wire the 4-byte stride would misalign every other element, so cast
    // down to Int32 once on the device.
    Tensor idx = (indices.dtype() == DType::Int64)
        ? indices.to(DType::Int32)
        : indices;

    // BFloat16/Float16: use native packed shader (indices stay int)
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        bool is_bf16_take = (input.dtype() == DType::BFloat16);
        int32_t device_id = input.device().index;

        auto* pipeline = getPipeline(is_bf16_take ? "take_bf16" : "take_f16", device_id);

        std::vector<int64_t> out_shape(indices.shape().begin(), indices.shape().end());
        Tensor output(out_shape, input.dtype(), input.device());

        struct { uint32_t numel; uint32_t num_indices; } pc;
        pc.numel = static_cast<uint32_t>(input.numel());
        pc.num_indices = static_cast<uint32_t>(idx.numel());

        // F16 packed buffer sizes: round up to 4-byte boundaries
        size_t in_size = (static_cast<size_t>(input.numel()) + 1) / 2 * 4;
        size_t idx_size = idx.numel() * idx.dtype_size();
        size_t out_size = (static_cast<size_t>(output.numel()) + 1) / 2 * 4;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()}, {1, idx.data_ptr()}, {2, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, idx_size, out_size};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(idx.numel(), devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        return output;
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);

    if (is_float64) {
        vulkan::ensure_fp64_supported(device_id, "Take");
    }
    std::string shader_name = is_float64 ? "take_f64" : "take";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(indices.shape().begin(), indices.shape().end());
    Tensor output(out_shape, input.dtype(), input.device());

    struct {
        uint32_t numel;
        uint32_t num_indices;
    } pushConstants;
    pushConstants.numel = static_cast<uint32_t>(input.numel());
    pushConstants.num_indices = static_cast<uint32_t>(idx.numel());

    const void* buffer_in = input.data_ptr();
    const void* buffer_idx = idx.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t in_size = input.numel() * input.dtype_size();
    size_t idx_size = idx.numel() * idx.dtype_size();
    size_t out_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_idx},
        {2, buffer_out}
    };
    std::vector<size_t> sizes = {in_size, idx_size, out_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(idx.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchTile(const Tensor& input, const std::vector<int64_t>& reps) -> Tensor {
    if (input.numel() == 0) {
        return Tensor(std::vector<int64_t>(input.shape().begin(), input.shape().end()), input.dtype(), input.device());
    }

    // BFloat16/Float16: use native packed shader
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        bool is_bf16_tile = (input.dtype() == DType::BFloat16);
        int32_t dev_id = input.device().index;

        auto in_shp = input.shape();
        size_t nd = std::max(in_shp.size(), reps.size());

        std::vector<uint32_t> in_shape_pad(nd, 1);
        std::vector<uint32_t> out_shape_u(nd);
        std::vector<int64_t> out_shape_i(nd);

        size_t in_off = nd - in_shp.size();
        for (size_t i = 0; i < in_shp.size(); ++i)
            in_shape_pad[in_off + i] = static_cast<uint32_t>(in_shp[i]);

        size_t rep_off = nd - reps.size();
        for (size_t i = 0; i < nd; ++i) {
            int64_t r = (i >= rep_off) ? reps[i - rep_off] : 1;
            out_shape_u[i] = in_shape_pad[i] * static_cast<uint32_t>(r);
            out_shape_i[i] = static_cast<int64_t>(out_shape_u[i]);
        }

        Tensor out_f16(out_shape_i, input.dtype(), input.device());
        int64_t out_numel = out_f16.numel();
        if (out_numel == 0) return out_f16;

        auto* pipe = getPipeline(is_bf16_tile ? "tile_bf16" : "tile_f16", dev_id);

        size_t shp_sz = nd * sizeof(uint32_t);
        Tensor shp_in({static_cast<int64_t>(nd)}, DType::Int32, input.device());
        Tensor shp_out({static_cast<int64_t>(nd)}, DType::Int32, input.device());
        copy(shp_in.data_ptr(), in_shape_pad.data(), shp_sz, CopyKind::HostToDevice);
        copy(shp_out.data_ptr(), out_shape_u.data(), shp_sz, CopyKind::HostToDevice);

        struct { uint32_t output_numel; uint32_t ndims; } pc;
        pc.output_numel = static_cast<uint32_t>(out_numel);
        pc.ndims = static_cast<uint32_t>(nd);

        auto in_cont = input.is_contiguous() ? input : input.contiguous();

        // F16 packed buffer sizes: round up to 4-byte boundaries
        size_t in_bsz = (static_cast<size_t>(in_cont.numel()) + 1) / 2 * 4;
        size_t out_bsz = (static_cast<size_t>(out_numel) + 1) / 2 * 4;

        std::vector<std::pair<uint32_t, const void*>> binds = {
            {0, in_cont.data_ptr()}, {1, out_f16.data_ptr()},
            {2, shp_in.data_ptr()}, {3, shp_out.data_ptr()}
        };
        std::vector<size_t> szs = {in_bsz, out_bsz, shp_sz, shp_sz};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(dev_id, pipe, binds, szs);
        VkCommandBuffer cmd = beginSingleTimeCommands(dev_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipe->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipe->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(out_numel, devices_[dev_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, dev_id);

        return out_f16;
    }

    // The base tile.comp / tile_f64.comp shaders have hardcoded
    // `float input_data[]` / `double input_data[]` SSBO bindings.
    // Non-Float dtypes would silently reinterpret as float garbage. Route
    // narrow types via Float32, Int64 via Float64 (preserves all int values
    // up to 2^53). Int8 / Bool / UInt8 widen to Float32. Mirror of the
    // repeat_interleave Int64 fix.
    DType orig_dtype = input.dtype();
    if (orig_dtype == DType::Int8 || orig_dtype == DType::Bool ||
        orig_dtype == DType::UInt8 || orig_dtype == DType::Int16 ||
        orig_dtype == DType::Int32) {
        Tensor f32_input = dispatchCast(input, DType::Float32);
        Tensor f32_result = dispatchTile(f32_input, reps);
        return dispatchCast(f32_result, orig_dtype);
    }
    if (orig_dtype == DType::Int64) {
        Tensor f64_input = dispatchCast(input, DType::Float64);
        Tensor f64_result = dispatchTile(f64_input, reps);
        return dispatchCast(f64_result, DType::Int64);
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    DType dtype = input.dtype();

    if (is_float64) {
        vulkan::ensure_fp64_supported(device_id, "Tile");
    }

    auto in_shape = input.shape();
    size_t ndims = std::max(in_shape.size(), reps.size());

    // Pad input shape and reps to same number of dimensions
    std::vector<uint32_t> input_shape_padded(ndims, 1);
    std::vector<uint32_t> output_shape_u32(ndims);
    std::vector<int64_t> output_shape_i64(ndims);

    size_t in_offset = ndims - in_shape.size();
    for (size_t i = 0; i < in_shape.size(); ++i) {
        input_shape_padded[in_offset + i] = static_cast<uint32_t>(in_shape[i]);
    }

    size_t rep_offset = ndims - reps.size();
    for (size_t i = 0; i < ndims; ++i) {
        int64_t r = (i >= rep_offset) ? reps[i - rep_offset] : 1;
        uint32_t in_s = input_shape_padded[i];
        output_shape_u32[i] = in_s * static_cast<uint32_t>(r);
        output_shape_i64[i] = static_cast<int64_t>(output_shape_u32[i]);
    }

    Tensor output(output_shape_i64, dtype, input.device());
    int64_t output_numel = output.numel();

    if (output_numel == 0) return output;

    std::string shader_name = is_float64 ? "tile_f64" : "tile";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Upload input_shape and output_shape as storage buffers
    size_t shape_buf_size = ndims * sizeof(uint32_t);

    // Create temporary tensors to hold shape data on GPU
    Tensor shape_in_tensor({static_cast<int64_t>(ndims)}, DType::Int32, input.device());
    Tensor shape_out_tensor({static_cast<int64_t>(ndims)}, DType::Int32, input.device());

    // Copy shape data to GPU using the backend copy() method
    copy(shape_in_tensor.data_ptr(), input_shape_padded.data(),
         shape_buf_size, CopyKind::HostToDevice);
    copy(shape_out_tensor.data_ptr(), output_shape_u32.data(),
         shape_buf_size, CopyKind::HostToDevice);

    struct {
        uint32_t output_numel;
        uint32_t ndims;
    } pushConstants;
    pushConstants.output_numel = static_cast<uint32_t>(output_numel);
    pushConstants.ndims = static_cast<uint32_t>(ndims);

    auto input_cont = input.is_contiguous() ? input : input.contiguous();

    const void* buffer_in = input_cont.data_ptr();
    const void* buffer_out = output.data_ptr();
    const void* buffer_in_shape = shape_in_tensor.data_ptr();
    const void* buffer_out_shape = shape_out_tensor.data_ptr();

    size_t in_size = input_cont.numel() * input_cont.dtype_size();
    size_t out_size = output_numel * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out},
        {2, buffer_in_shape},
        {3, buffer_out_shape}
    };
    std::vector<size_t> sizes = {in_size, out_size, shape_buf_size, shape_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(output_numel, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchPut(const Tensor& input, const Tensor& indices_in,
                                 const Tensor& source, bool accumulate) -> Tensor {
    // Clone input to output (put modifies output in-place)
    Tensor output = dispatchClone(input);

    if (indices_in.numel() == 0) return output;

    // put shaders read `int indices[]` (32-bit). If the caller passed Int64
    // we must cast down; otherwise the 4-byte stride misaligns every other
    // element just like take did.
    Tensor indices = (indices_in.dtype() == DType::Int64)
        ? indices_in.to(DType::Int32)
        : indices_in;

    // BFloat16/Float16: use native packed shader with CAS atomics
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        bool is_bf16_put = (input.dtype() == DType::BFloat16);
        Tensor output_f16 = dispatchClone(input);
        if (indices.numel() == 0) return output_f16;

        int32_t dev_id = input.device().index;
        auto* pipe = getPipeline(is_bf16_put ? "put_bf16" : "put_f16", dev_id);

        struct { uint32_t numel; uint32_t num_indices; uint32_t accumulate; } pc;
        pc.numel = static_cast<uint32_t>(input.numel());
        pc.num_indices = static_cast<uint32_t>(indices.numel());
        pc.accumulate = accumulate ? 1u : 0u;

        // F16 packed buffer sizes: round up to 4-byte boundaries
        size_t out_bsz = (static_cast<size_t>(input.numel()) + 1) / 2 * 4;
        size_t idx_bsz = indices.numel() * indices.dtype_size();
        size_t src_bsz = (static_cast<size_t>(source.numel()) + 1) / 2 * 4;

        std::vector<std::pair<uint32_t, const void*>> binds = {
            {0, output_f16.data_ptr()}, {1, indices.data_ptr()}, {2, source.data_ptr()}
        };
        std::vector<size_t> szs = {out_bsz, idx_bsz, src_bsz};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(dev_id, pipe, binds, szs);
        VkCommandBuffer cmd = beginSingleTimeCommands(dev_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipe->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipe->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(indices.numel(), devices_[dev_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, dev_id);

        return output_f16;
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);

    if (is_float64) {
        vulkan::ensure_fp64_supported(device_id, "Put");
        // Y.10: put_f64.comp uses GL_EXT_shader_atomic_int64 for CAS-based
        // Float64 atomicAdd when accumulating; gate so unsupported devices
        // fail fast instead of hitting an opaque SPIR-V validation error.
        vulkan::ensure_atomic_int64_supported(device_id, "Put");
    }
    std::string shader_name = is_float64 ? "put_f64" : "put";
    auto* pipeline = getPipeline(shader_name, device_id);

    struct {
        uint32_t numel;
        uint32_t num_indices;
        uint32_t accumulate;
    } pushConstants;
    pushConstants.numel = static_cast<uint32_t>(input.numel());
    pushConstants.num_indices = static_cast<uint32_t>(indices.numel());
    pushConstants.accumulate = accumulate ? 1u : 0u;

    const void* buffer_out = output.data_ptr();
    const void* buffer_idx = indices.data_ptr();
    const void* buffer_src = source.data_ptr();

    size_t out_size = output.numel() * output.dtype_size();
    size_t idx_size = indices.numel() * indices.dtype_size();
    size_t src_size = source.numel() * source.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_out},
        {1, buffer_idx},
        {2, buffer_src}
    };
    std::vector<size_t> sizes = {out_size, idx_size, src_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(indices.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// =============================================================================
// FFT Operations — Native Vulkan compute shader implementation
// =============================================================================

// Helper: check if n is a power of 2
static bool is_power_of_2(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Helper: compute log2 of a power-of-2 integer
static uint32_t log2_int(uint32_t n) {
    uint32_t result = 0;
    while (n > 1) { n >>= 1; ++result; }
    return result;
}

// Helper: compute normalization scale factor for FFT
static double fft_norm_factor(int64_t n, const std::string& norm, bool is_forward) {
    if (norm == "ortho") return 1.0 / std::sqrt(static_cast<double>(n));
    if (is_forward && norm == "forward") return 1.0 / static_cast<double>(n);
    if (!is_forward && (norm == "backward" || norm.empty())) return 1.0 / static_cast<double>(n);
    return 1.0;
}

// Try to factorize N into supported radices {2, 3, 5, 7}.
// Returns list of radices in order (e.g., 60 -> {2,2,3,5}), or empty if not factorable.
static std::vector<int> factorize_fft(int64_t N) {
    std::vector<int> factors;
    const int radices[] = {7, 5, 3, 2};
    int64_t rem = N;
    for (int r : radices) {
        while (rem % r == 0) {
            factors.push_back(r);
            rem /= r;
        }
    }
    if (rem != 1) return {};  // not factorable into supported radices
    // Reverse so smallest radices come first (better cache behavior)
    std::reverse(factors.begin(), factors.end());
    return factors;
}

// MAX_VULKAN_FFT_SIZE removed: Bluestein's algorithm handles any signal length
// by converting to a power-of-2 FFT, so there is no inherent GPU size limit.

// Max matrix size for native linalg shaders (det, inv, solve, cholesky, qr).
//
// Tiled blocked algorithms use panel factorization (panel_width=32) + trailing
// matrix update shaders.  Memory usage is O(panel_width * N) per panel step,
// so arbitrarily large matrices are supported in principle.  The limit below is
// Tiers:
//   1. Single-workgroup shaders for matrices up to 128x128 (shared-memory LU).
//   2. Tiled blocked algorithms for larger matrices.
//
// SVD, Eigh, and Eig only have single-workgroup shaders — tiled
// implementations for larger matrices are pending.
static constexpr int64_t MAX_SMALL_LINALG_SIZE = 128;  // single-workgroup shader limit
static constexpr int64_t TILED_BLOCK_SIZE = 32;        // panel width for blocked algorithms

auto VulkanBackend::runFFTButterfly(const Tensor& input, uint32_t fft_size,
                                     uint32_t direction, uint32_t batch_size,
                                     uint32_t batch_stride) -> Tensor {
    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Complex128);
    bool is_f16 = (input.dtype() == DType::Float16);
    if (is_f64) {
        vulkan::ensure_fp64_supported(device_id, "FFTButterfly");
    }
    uint32_t num_stages = log2_int(fft_size);

    // We ping-pong between two buffers for each stage
    Tensor buf_a = input.contiguous();
    Tensor buf_b(std::vector<int64_t>(input.shape().begin(), input.shape().end()), input.dtype(), input.device());

    // Bit-reversal permutation first
    {
        std::string shader = is_f16 ? "fft_bit_reverse_f16"
                           : is_f64 ? "fft_bit_reverse_f64"
                           : "fft_bit_reverse";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            uint32_t log2_n;
            uint32_t batch_size;
            uint32_t batch_stride;
        } pc;
        pc.n = fft_size;
        pc.log2_n = num_stages;
        pc.batch_size = batch_size;
        pc.batch_stride = batch_stride;

        // F16: 1 uint32 per complex element; F64: 16 bytes; F32: 8 bytes
        size_t elem_size = is_f16 ? 4 : (is_f64 ? 16 : 8);
        size_t buf_size = input.numel() * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buf_a.data_ptr()}, {1, buf_b.data_ptr()}
        };
        std::vector<size_t> sizes = {buf_size, buf_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(fft_size * batch_size, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        std::swap(buf_a, buf_b);  // buf_a now has bit-reversed data
    }

    // Run butterfly stages
    for (uint32_t stage = 0; stage < num_stages; ++stage) {
        std::string shader = is_f16 ? "fft_f16"
                           : is_f64 ? "fft_f64"
                           : "fft";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            uint32_t stage;
            uint32_t direction;
            uint32_t batch_size;
            uint32_t batch_stride;
        } pc;
        pc.n = fft_size;
        pc.stage = stage;
        pc.direction = direction;
        pc.batch_size = batch_size;
        pc.batch_stride = batch_stride;

        size_t elem_size = is_f16 ? 4 : (is_f64 ? 16 : 8);
        size_t buf_size = input.numel() * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buf_a.data_ptr()}, {1, buf_b.data_ptr()}
        };
        std::vector<size_t> sizes = {buf_size, buf_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        uint32_t num_butterflies = fft_size / 2;
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_butterflies * batch_size, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        std::swap(buf_a, buf_b);  // Result is now in buf_a
    }

    return buf_a;
}

auto VulkanBackend::runMixedRadixFFT(const Tensor& input, int64_t N, uint32_t direction,
                                       uint32_t batch_size, uint32_t batch_stride) -> Tensor {
    auto factors = factorize_fft(N);
    if (factors.empty()) {
        throw std::runtime_error("runMixedRadixFFT: N not factorable into {2,3,5,7}");
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Complex128);
    bool is_f16 = (input.dtype() == DType::Float16);
    if (is_f64) {
        vulkan::ensure_fp64_supported(device_id, "MixedRadixFFT");
    }

    size_t elem_size = is_f16 ? 4 : (is_f64 ? 16 : 8);
    size_t buf_size = input.numel() * elem_size;

    // Ping-pong buffers
    Tensor buf_a = input.contiguous();
    Tensor buf_b(std::vector<int64_t>(input.shape().begin(), input.shape().end()), input.dtype(), input.device());

    // Execute one stage per factor
    int64_t stage_stride = 1;
    for (int radix : factors) {
        std::string shader = "fft_radix" + std::to_string(radix);
        if (is_f64) shader += "_f64";
        else if (is_f16) shader += "_f16";

        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t N_val;
            uint32_t stage_stride;
            uint32_t direction;
            uint32_t batch_size;
            uint32_t batch_stride;
        } pc;
        pc.N_val = static_cast<uint32_t>(N);
        pc.stage_stride = static_cast<uint32_t>(stage_stride);
        pc.direction = direction;
        pc.batch_size = batch_size;
        pc.batch_stride = batch_stride;

        uint32_t n_butterflies = static_cast<uint32_t>(N) / radix;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buf_a.data_ptr()}, {1, buf_b.data_ptr()}
        };
        std::vector<size_t> sizes = {buf_size, buf_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(n_butterflies * batch_size, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        std::swap(buf_a, buf_b);
        stage_stride *= radix;
    }

    return buf_a;
}

auto VulkanBackend::runFFTScale(Tensor& data, uint32_t n, double scale_factor) -> void {
    if (std::abs(scale_factor - 1.0) < 1e-15) return;  // No scaling needed

    int32_t device_id = data.device().index;
    bool is_f64 = (data.dtype() == DType::Complex128);
    bool is_f16 = (data.dtype() == DType::Float16);

    if (is_f64) {
        vulkan::ensure_fp64_supported(device_id, "FFTScale");
    }
    if (is_f16) {
        // F16: each complex element = 1 uint32 word
        std::string shader = "fft_scale_f16";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            float inv_n;
        } pc;
        pc.n = n;
        pc.inv_n = static_cast<float>(scale_factor);

        size_t buf_size = n * 4;  // 1 uint32 per complex element
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, data.data_ptr()}};
        std::vector<size_t> sizes = {buf_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(n, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    } else if (is_f64) {
        std::string shader = "fft_scale_f64";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            uint32_t padding0;
            double inv_n;
        } pc;
        pc.n = n;
        pc.padding0 = 0;
        pc.inv_n = scale_factor;

        size_t buf_size = n * 16;  // complex128
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, data.data_ptr()}};
        std::vector<size_t> sizes = {buf_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(n, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    } else {
        std::string shader = "fft_scale";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            float inv_n;
        } pc;
        pc.n = n;
        pc.inv_n = static_cast<float>(scale_factor);

        size_t buf_size = n * 8;  // complex64
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, data.data_ptr()}};
        std::vector<size_t> sizes = {buf_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(n, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }
}

// Helper: next power of 2 >= n
static int64_t next_power_of_2(int64_t n) {
    int64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

auto VulkanBackend::runFFTChirpMultiply(Tensor& data, const Tensor& chirp,
                                          uint32_t n, bool conjugate) -> void {
    int32_t device_id = data.device().index;
    bool is_f64 = (data.dtype() == DType::Complex128);

    if (is_f64) {
        vulkan::ensure_fp64_supported(device_id, "FFTChirpMultiply");
    }
    std::string shader = is_f64 ? "fft_bluestein_chirp_f64" : "fft_bluestein_chirp";
    auto* pipeline = getPipeline(shader, device_id);

    struct PushConstants {
        uint32_t n;
        uint32_t conjugate;
    } pc;
    pc.n = n;
    pc.conjugate = conjugate ? 1 : 0;

    size_t elem_size = is_f64 ? 16 : 8;  // complex element size
    size_t data_size = data.numel() * elem_size;
    size_t chirp_size = chirp.numel() * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, data.data_ptr()}, {1, chirp.data_ptr()}
    };
    std::vector<size_t> sizes = {data_size, chirp_size};
    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(n, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
}

auto VulkanBackend::runFFTChirpGen(Tensor& output, uint32_t N, int32_t sign) -> void {
    int32_t device_id = output.device().index;
    bool is_f64 = (output.dtype() == DType::Complex128);

    if (is_f64) {
        vulkan::ensure_fp64_supported(device_id, "FFTChirpGen");
    }
    std::string shader = is_f64 ? "fft_bluestein_chirp_gen_f64" : "fft_bluestein_chirp_gen";
    auto* pipeline = getPipeline(shader, device_id);

    struct PushConstants {
        uint32_t N;
        int32_t sign;
    } pc;
    pc.N = N;
    pc.sign = sign;

    size_t elem_size = is_f64 ? 16 : 8;
    size_t buf_size = N * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_size};
    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(N, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
}

auto VulkanBackend::runFFTConjKernelGen(Tensor& output, uint32_t N, uint32_t M,
                                          int32_t sign) -> void {
    int32_t device_id = output.device().index;
    bool is_f64 = (output.dtype() == DType::Complex128);

    if (is_f64) {
        vulkan::ensure_fp64_supported(device_id, "FFTConjKernelGen");
    }
    std::string shader = is_f64 ? "fft_bluestein_conj_kernel_f64" : "fft_bluestein_conj_kernel";
    auto* pipeline = getPipeline(shader, device_id);

    struct PushConstants {
        uint32_t N;
        uint32_t M;
        int32_t sign;
    } pc;
    pc.N = N;
    pc.M = M;
    pc.sign = sign;

    size_t elem_size = is_f64 ? 16 : 8;
    size_t buf_size = M * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_size};
    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(N, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
}

auto VulkanBackend::dispatchFFTBluestein(const Tensor& input, int64_t signal_len,
                                           uint32_t direction) -> Tensor {
    // Bluestein's algorithm converts N-point DFT to circular convolution of
    // length M >= 2N-1 where M is a power of 2, enabling use of Cooley-Tukey FFT.
    //
    // Input: 1D complex tensor of length signal_len (single batch element)
    // Returns: 1D complex tensor of length signal_len

    bool is_f64 = (input.dtype() == DType::Complex128);
    DType complex_dtype = input.dtype();
    size_t elem_size = is_f64 ? 16 : 8;  // bytes per complex element

    int64_t N = signal_len;
    int64_t M = next_power_of_2(2 * N - 1);
    int32_t device_id = input.device().index;
    Device vulkan_dev = input.device();

    // Step 1: Generate chirp sequence on GPU
    // chirp[k] = exp(sign * j * pi * k^2 / N)
    // Forward: sign = -1, Inverse: sign = +1
    int32_t sign_int = (direction == 0) ? -1 : 1;

    Tensor chirp_gpu({N}, complex_dtype, vulkan_dev);
    runFFTChirpGen(chirp_gpu, static_cast<uint32_t>(N), sign_int);

    // Step 2: Create zero-padded a[M] with a[0..N-1] = input[0..N-1] * chirp[0..N-1]
    // Tensor constructor zero-initializes, so padding is already zero
    Tensor a_padded({M}, complex_dtype, vulkan_dev);

    // Copy input data into first N elements of a_padded via vkCmdCopyBuffer
    {
        auto [src_buf, src_off] = getVulkanBufferAndOffset(input.data_ptr());
        auto [dst_buf, dst_off] = getVulkanBufferAndOffset(a_padded.data_ptr());

        VkBufferCopy region{};
        region.srcOffset = static_cast<VkDeviceSize>(src_off);
        region.dstOffset = static_cast<VkDeviceSize>(dst_off);
        region.size = static_cast<VkDeviceSize>(N * elem_size);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
        insertTransferToComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    // Multiply first N elements of a_padded by chirp
    runFFTChirpMultiply(a_padded, chirp_gpu, static_cast<uint32_t>(N), /*conjugate=*/false);

    // Step 3: Generate conjugate chirp convolution kernel of length M on GPU
    // b[0] = conj(chirp[0]), b[k] = conj(chirp[k]) for k=1..N-1,
    // b[M-k] = conj(chirp[k]) for k=1..N-1, rest = 0
    // Tensor constructor zero-initializes, so padding is already zero
    Tensor b_padded({M}, complex_dtype, vulkan_dev);
    runFFTConjKernelGen(b_padded, static_cast<uint32_t>(N), static_cast<uint32_t>(M), sign_int);

    // Step 4: FFT(a), FFT(b) using power-of-2 Cooley-Tukey
    Tensor A = runFFTButterfly(a_padded, static_cast<uint32_t>(M), 0, 1, static_cast<uint32_t>(M));
    Tensor B = runFFTButterfly(b_padded, static_cast<uint32_t>(M), 0, 1, static_cast<uint32_t>(M));

    // Step 5: Pointwise multiply A *= B
    runFFTChirpMultiply(A, B, static_cast<uint32_t>(M), /*conjugate=*/false);

    // Step 6: IFFT of the product
    Tensor conv_result = runFFTButterfly(A, static_cast<uint32_t>(M), 1, 1, static_cast<uint32_t>(M));

    // Scale by 1/M for the IFFT
    runFFTScale(conv_result, static_cast<uint32_t>(M), 1.0 / static_cast<double>(M));

    // Step 7: Extract first N elements and multiply by chirp
    Tensor result({N}, complex_dtype, vulkan_dev);
    {
        auto [src_buf, src_off] = getVulkanBufferAndOffset(conv_result.data_ptr());
        auto [dst_buf, dst_off] = getVulkanBufferAndOffset(result.data_ptr());

        VkBufferCopy region{};
        region.srcOffset = static_cast<VkDeviceSize>(src_off);
        region.dstOffset = static_cast<VkDeviceSize>(dst_off);
        region.size = static_cast<VkDeviceSize>(N * elem_size);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
        insertTransferToComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    runFFTChirpMultiply(result, chirp_gpu, static_cast<uint32_t>(N), /*conjugate=*/false);

    return result;
}

auto VulkanBackend::dispatchFFT(const Tensor& input, int64_t dim, int64_t n,
                                 const std::string& norm) -> Tensor {
    // Input is Complex64 or Complex128 (interleaved re/im)
    if (input.dtype() == DType::Complex128) {
        vulkan::ensure_fp64_supported(input.device().index, "FFT");
    }
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    // Non-last-dim: transpose target dim to last, run FFT on last dim, transpose back
    if (dim != ndim - 1 && ndim > 1) {
        auto transposed = dispatchTranspose(input, dim, ndim - 1);
        auto fft_result = dispatchFFT(transposed, ndim - 1, n, norm);
        return dispatchTranspose(fft_result, dim, ndim - 1);
    }

    int64_t signal_len = shape[dim];

    // GPU-side pad or truncate when requested FFT length differs from input length
    Tensor working_input = input;
    if (n != signal_len) {
        if (n > signal_len) {
            // Pad with zeros along the FFT dimension on GPU
            std::vector<int64_t> pad_shape(shape.begin(), shape.end());
            pad_shape[dim] = n - signal_len;
            Tensor zeros_pad = dispatchZeros(pad_shape, input.dtype(), input.device());
            working_input = dispatchCat({input, zeros_pad}, dim);
        } else {
            // Truncate: slice to first n elements along the FFT dimension
            std::vector<int64_t> starts(ndim, 0);
            std::vector<int64_t> ends(shape.begin(), shape.end());
            std::vector<int64_t> steps(ndim, 1);
            ends[dim] = n;
            working_input = dispatchSlice(input, starts, ends, steps);
        }
        signal_len = n;
        shape = working_input.shape();
    }

    // Check if we can handle this on the GPU (guaranteed last dim after transpose above)
    bool can_cooley_tukey = is_power_of_2(signal_len);
    // Mixed-radix Stockham shaders produce incorrect results (e.g. FFT of a
    // unit impulse comes out as [2,0,0,0] instead of [1,1,1,1] for N=4=2*2).
    // Prefer Bluestein's algorithm, which internally uses the working
    // Cooley-Tukey path. Leaves the fast path in place for sizes that are
    // already powers of two.
    bool can_mixed_radix = false;

    // Compute batch size (product of all dims except last)
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 1; ++i) batch_size *= shape[i];

    auto result = working_input.contiguous();

    if (can_cooley_tukey) {
        // Power-of-2: use Cooley-Tukey directly (fastest) — batched dispatch
        result = runFFTButterfly(result, static_cast<uint32_t>(signal_len), 0,
                                  static_cast<uint32_t>(batch_size),
                                  static_cast<uint32_t>(signal_len));
    } else if (can_mixed_radix && !is_power_of_2(signal_len)) {
        // Factorable into {2,3,5,7}: use mixed-radix Stockham — batched dispatch
        result = runMixedRadixFFT(result, signal_len, 0,
                                    static_cast<uint32_t>(batch_size),
                                    static_cast<uint32_t>(signal_len));
    } else {
        // Non-power-of-2: use Bluestein's algorithm per batch element
        size_t elem_size = (input.dtype() == DType::Complex128) ? 16 : 8;
        int32_t device_id = input.device().index;

        for (int64_t b = 0; b < batch_size; ++b) {
            // Extract batch slice as 1D tensor
            Tensor batch_slice({signal_len}, input.dtype(), input.device());
            {
                auto [src_buf, src_off] = getVulkanBufferAndOffset(result.data_ptr());
                auto [dst_buf, dst_off] = getVulkanBufferAndOffset(batch_slice.data_ptr());

                VkBufferCopy region{};
                region.srcOffset = static_cast<VkDeviceSize>(src_off) + b * signal_len * elem_size;
                region.dstOffset = static_cast<VkDeviceSize>(dst_off);
                region.size = signal_len * elem_size;

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
                insertTransferToComputeBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }

            Tensor bluestein_result = dispatchFFTBluestein(batch_slice, signal_len, 0);

            // Write result back to correct batch position
            {
                auto [src_buf, src_off] = getVulkanBufferAndOffset(bluestein_result.data_ptr());
                auto [dst_buf, dst_off] = getVulkanBufferAndOffset(result.data_ptr());

                VkBufferCopy region{};
                region.srcOffset = static_cast<VkDeviceSize>(src_off);
                region.dstOffset = static_cast<VkDeviceSize>(dst_off) + b * signal_len * elem_size;
                region.size = signal_len * elem_size;

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
                insertTransferToComputeBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
    }

    // Apply normalization
    double scale = fft_norm_factor(signal_len, norm, /*is_forward=*/true);
    if (std::abs(scale - 1.0) > 1e-15) {
        uint32_t total_elems = static_cast<uint32_t>(result.numel());
        runFFTScale(result, total_elems, scale);
    }

    return result;
}

auto VulkanBackend::dispatchIFFT(const Tensor& input, int64_t dim, int64_t n,
                                  const std::string& norm) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    // Non-last-dim: transpose, IFFT on last dim, transpose back
    if (dim != ndim - 1 && ndim > 1) {
        auto transposed = dispatchTranspose(input, dim, ndim - 1);
        auto ifft_result = dispatchIFFT(transposed, ndim - 1, n, norm);
        return dispatchTranspose(ifft_result, dim, ndim - 1);
    }

    int64_t signal_len = shape[dim];

    // GPU-side pad or truncate when requested IFFT length differs from input length
    Tensor working_input = input;
    if (n != signal_len) {
        if (n > signal_len) {
            std::vector<int64_t> pad_shape(shape.begin(), shape.end());
            pad_shape[dim] = n - signal_len;
            Tensor zeros_pad = dispatchZeros(pad_shape, input.dtype(), input.device());
            working_input = dispatchCat({input, zeros_pad}, dim);
        } else {
            std::vector<int64_t> starts(ndim, 0);
            std::vector<int64_t> ends(shape.begin(), shape.end());
            std::vector<int64_t> steps(ndim, 1);
            ends[dim] = n;
            working_input = dispatchSlice(input, starts, ends, steps);
        }
        signal_len = n;
        shape = working_input.shape();
    }

    bool can_cooley_tukey = is_power_of_2(signal_len);
    bool can_mixed_radix = false;  // Mixed-radix Stockham path is buggy; use Bluestein

    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 1; ++i) batch_size *= shape[i];

    auto result = working_input.contiguous();

    if (can_cooley_tukey) {
        result = runFFTButterfly(result, static_cast<uint32_t>(signal_len), 1,
                                  static_cast<uint32_t>(batch_size),
                                  static_cast<uint32_t>(signal_len));
    } else if (can_mixed_radix && !is_power_of_2(signal_len)) {
        // Factorable into {2,3,5,7}: use mixed-radix Stockham with direction=1 (inverse)
        result = runMixedRadixFFT(result, signal_len, 1,
                                    static_cast<uint32_t>(batch_size),
                                    static_cast<uint32_t>(signal_len));
    } else {
        // Non-factorable: use Bluestein's algorithm per batch element
        size_t elem_size = (input.dtype() == DType::Complex128) ? 16 : 8;
        int32_t device_id = input.device().index;

        for (int64_t b = 0; b < batch_size; ++b) {
            Tensor batch_slice({signal_len}, input.dtype(), input.device());
            {
                auto [src_buf, src_off] = getVulkanBufferAndOffset(result.data_ptr());
                auto [dst_buf, dst_off] = getVulkanBufferAndOffset(batch_slice.data_ptr());

                VkBufferCopy region{};
                region.srcOffset = static_cast<VkDeviceSize>(src_off) + b * signal_len * elem_size;
                region.dstOffset = static_cast<VkDeviceSize>(dst_off);
                region.size = signal_len * elem_size;

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
                insertTransferToComputeBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }

            Tensor bluestein_result = dispatchFFTBluestein(batch_slice, signal_len, 1);

            {
                auto [src_buf, src_off] = getVulkanBufferAndOffset(bluestein_result.data_ptr());
                auto [dst_buf, dst_off] = getVulkanBufferAndOffset(result.data_ptr());

                VkBufferCopy region{};
                region.srcOffset = static_cast<VkDeviceSize>(src_off);
                region.dstOffset = static_cast<VkDeviceSize>(dst_off) + b * signal_len * elem_size;
                region.size = signal_len * elem_size;

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
                insertTransferToComputeBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
    }

    // IFFT normalization: default "backward" norm divides by N
    double scale = fft_norm_factor(signal_len, norm, /*is_forward=*/false);
    if (std::abs(scale - 1.0) > 1e-15) {
        uint32_t total_elems = static_cast<uint32_t>(result.numel());
        runFFTScale(result, total_elems, scale);
    }

    return result;
}

auto VulkanBackend::dispatchRFFT(const Tensor& input, int64_t dim, int64_t n,
                                  const std::string& norm) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    // Non-last-dim: transpose, RFFT on last dim, transpose back
    if (dim != ndim - 1 && ndim > 1) {
        auto transposed = dispatchTranspose(input, dim, ndim - 1);
        auto rfft_result = dispatchRFFT(transposed, ndim - 1, n, norm);
        return dispatchTranspose(rfft_result, dim, ndim - 1);
    }

    int64_t signal_len = shape[dim];
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);

    if (is_f64) {
        vulkan::ensure_fp64_supported(input.device().index, "RFFT");
    }

    // GPU-side pad or truncate when requested RFFT length differs from input length
    Tensor working_input = input;
    if (n != signal_len) {
        if (n > signal_len) {
            std::vector<int64_t> pad_shape(shape.begin(), shape.end());
            pad_shape[dim] = n - signal_len;
            Tensor zeros_pad = dispatchZeros(pad_shape, input.dtype(), input.device());
            working_input = dispatchCat({input, zeros_pad}, dim);
        } else {
            std::vector<int64_t> starts(ndim, 0);
            std::vector<int64_t> ends(shape.begin(), shape.end());
            std::vector<int64_t> steps(ndim, 1);
            ends[dim] = n;
            working_input = dispatchSlice(input, starts, ends, steps);
        }
        signal_len = n;
        shape = working_input.shape();
    }

    if (signal_len < 2) {
        throw std::invalid_argument(std::format(
            "Vulkan RFFT requires signal length >= 2, got {}", signal_len));
    }

    bool can_cooley_tukey = is_power_of_2(signal_len);
    bool can_mixed_radix = false;  // Mixed-radix Stockham path is buggy; use Bluestein

    int32_t device_id = input.device().index;
    DType complex_dtype = is_f64 ? DType::Complex128 : DType::Complex64;
    int64_t half_n = signal_len / 2;
    size_t complex_elem_size = is_f64 ? 16 : 8;

    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 1; ++i) batch_size *= shape[i];

    // Output shape: [..., N/2+1] complex
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[ndim - 1] = half_n + 1;
    Tensor output(out_shape, complex_dtype, input.device());

    // Helper lambda: real-to-complex conversion on GPU via shader
    auto run_real_to_complex = [&](const Tensor& cont_input, int64_t b_idx) -> Tensor {
        Tensor complex_input({signal_len}, complex_dtype, input.device());
        std::string shader = is_f64 ? "real_to_complex_f64" : "real_to_complex";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t num_elements;
        } pc;
        pc.num_elements = static_cast<uint32_t>(signal_len);

        size_t real_elem_sz = is_f64 ? 8 : 4;
        const void* in_ptr = static_cast<const char*>(cont_input.data_ptr())
                             + b_idx * signal_len * real_elem_sz;
        size_t in_size = signal_len * real_elem_sz;
        size_t out_size = signal_len * complex_elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, in_ptr}, {1, complex_input.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(signal_len, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return complex_input;
    };

    // Helper lambda: copy first N/2+1 complex bins from full FFT to output batch slice
    auto copy_half_bins_to_output = [&](const Tensor& full_fft, int64_t b_idx) {
        auto [src_buf, src_off] = getVulkanBufferAndOffset(full_fft.data_ptr());
        auto [dst_buf, dst_off] = getVulkanBufferAndOffset(output.data_ptr());

        VkBufferCopy region{};
        region.srcOffset = static_cast<VkDeviceSize>(src_off);
        region.dstOffset = static_cast<VkDeviceSize>(dst_off)
                         + b_idx * (half_n + 1) * complex_elem_size;
        region.size = (half_n + 1) * complex_elem_size;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
        insertTransferToComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    };

    // Mixed-radix path for non-power-of-2 sizes factorable into {2,3,5,7}
    if (can_mixed_radix && !is_power_of_2(signal_len)) {
        auto cont = working_input.contiguous();

        for (int64_t b = 0; b < batch_size; ++b) {
            Tensor complex_input = run_real_to_complex(cont, b);
            Tensor full_fft = runMixedRadixFFT(complex_input, signal_len, 0, 1, static_cast<uint32_t>(signal_len));
            copy_half_bins_to_output(full_fft, b);
        }

        double scale = fft_norm_factor(signal_len, norm, /*is_forward=*/true);
        if (std::abs(scale - 1.0) > 1e-15) {
            uint32_t total_elems = static_cast<uint32_t>(output.numel());
            runFFTScale(output, total_elems, scale);
        }

        return output;
    }

    // Non-factorable non-power-of-2: convert real to complex on GPU, run full FFT via Bluestein,
    // then extract first N/2+1 bins
    if (!can_cooley_tukey) {
        auto cont = working_input.contiguous();

        for (int64_t b = 0; b < batch_size; ++b) {
            Tensor complex_input = run_real_to_complex(cont, b);

            Tensor full_fft = dispatchFFTBluestein(complex_input, signal_len, 0);
            copy_half_bins_to_output(full_fft, b);
        }

        // Apply normalization
        double scale = fft_norm_factor(signal_len, norm, /*is_forward=*/true);
        if (std::abs(scale - 1.0) > 1e-15) {
            uint32_t total_elems = static_cast<uint32_t>(output.numel());
            runFFTScale(output, total_elems, scale);
        }

        return output;
    }

    auto cont = working_input.contiguous();

    // Step 1: Pack N real values into N/2 complex — batched
    std::vector<int64_t> packed_shape = {batch_size * half_n};
    Tensor packed(packed_shape, complex_dtype, input.device());

    {
        std::string shader = is_f16 ? "rfft_pack_f16" : is_f64 ? "rfft_pack_f64" : "rfft_pack";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            uint32_t batch_size;
            uint32_t in_batch_stride;
            uint32_t out_batch_stride;
        } pc;
        pc.n = static_cast<uint32_t>(signal_len);
        pc.batch_size = static_cast<uint32_t>(batch_size);
        pc.in_batch_stride = static_cast<uint32_t>(signal_len);
        pc.out_batch_stride = static_cast<uint32_t>(half_n * 2);  // complex interleaved

        size_t in_size = cont.numel() * (is_f64 ? 8 : 4);
        size_t out_size = batch_size * half_n * complex_elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, packed.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(half_n * batch_size, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    // Step 2: Run N/2-point complex FFT on packed data — batched
    Tensor fft_result = runFFTButterfly(packed, static_cast<uint32_t>(half_n), 0,
                                        static_cast<uint32_t>(batch_size),
                                        static_cast<uint32_t>(half_n));

    // Step 3: Unpack to N/2+1 complex output — batched
    {
        std::string shader = is_f16 ? "rfft_unpack_f16" : is_f64 ? "rfft_unpack_f64" : "rfft_unpack";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            uint32_t batch_size;
            uint32_t in_batch_stride;
            uint32_t out_batch_stride;
        } pc;
        pc.n = static_cast<uint32_t>(signal_len);
        pc.batch_size = static_cast<uint32_t>(batch_size);
        // rfft_unpack.comp reads `data_in[(in_base + k) * 2]`, i.e. it
        // treats `in_base + k` as a COMPLEX-element index and multiplies
        // by 2 internally to get the float offset. So in_batch_stride
        // must be given in COMPLEX units = half_n, not half_n * 2.
        //
        // The previous value (half_n * 2) caused batch b to read batch
        // 2*b's data, producing the "every other frame" pattern that
        // surfaced as a 2/3 amplitude drop in STFT round-trip tests on
        // multi-frame inputs. For batch_size == 1 (e.g. single-signal
        // RFFT) the bug is invisible because there's only one batch.
        //
        // out_batch_stride is already in float units (the shader writes
        // `data_out[out_base + k * 2] = ...`), so (half_n + 1) * 2 is
        // correct.
        pc.in_batch_stride = static_cast<uint32_t>(half_n);
        pc.out_batch_stride = static_cast<uint32_t>((half_n + 1) * 2);

        size_t in_size = batch_size * half_n * complex_elem_size;
        // Output is a Complex64/128 tensor: `output.numel()` counts complex
        // elements, not bytes. Each complex element is `complex_elem_size`
        // bytes (8 for Complex64, 16 for Complex128). The previous code
        // multiplied by 4 in the Float32 branch — half the required size —
        // which made the descriptor-bound range smaller than the shader
        // writes and produced undefined results on permissive drivers.
        size_t out_size = static_cast<size_t>(output.numel()) * complex_elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, fft_result.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg((half_n + 1) * batch_size, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    // Apply normalization
    double scale = fft_norm_factor(signal_len, norm, /*is_forward=*/true);
    if (std::abs(scale - 1.0) > 1e-15) {
        uint32_t total_elems = static_cast<uint32_t>(output.numel());
        runFFTScale(output, total_elems, scale);
    }

    return output;
}

auto VulkanBackend::dispatchIRFFT(const Tensor& input, int64_t dim, int64_t n,
                                   const std::string& norm) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    // Non-last-dim: transpose, IRFFT on last dim, transpose back
    if (dim != ndim - 1 && ndim > 1) {
        auto transposed = dispatchTranspose(input, dim, ndim - 1);
        auto irfft_result = dispatchIRFFT(transposed, ndim - 1, n, norm);
        return dispatchTranspose(irfft_result, dim, ndim - 1);
    }

    // Promote real inputs to complex (Hermitian-symmetric reinterpretation).
    // CPU accepts Float32/Float64 half-spectrum inputs; mirror that here so the
    // rest of the pipeline sees only Complex64/Complex128.
    Tensor input_c = input;
    if (input_c.dtype() == DType::Float32) input_c = dispatchCast(input_c, DType::Complex64);
    else if (input_c.dtype() == DType::Float64) input_c = dispatchCast(input_c, DType::Complex128);
    else if (input_c.dtype() == DType::Float16) input_c = dispatchCast(input_c, DType::Complex64);
    else if (input_c.dtype() == DType::BFloat16) input_c = dispatchCast(input_c, DType::Complex64);

    int64_t freq_bins = shape[dim];  // N/2+1
    int64_t output_len = n;
    bool is_f64 = (input_c.dtype() == DType::Complex128);
    bool is_f16 = (input_c.dtype() == DType::Float16);

    if (is_f64) {
        vulkan::ensure_fp64_supported(input.device().index, "IRFFT");
    }

    if (output_len < 2) {
        throw std::invalid_argument(std::format(
            "Vulkan IRFFT requires output length >= 2, got {}", output_len));
    }

    // GPU-side pad or truncate frequency bins when they don't match expected N/2+1
    int64_t expected_bins = output_len / 2 + 1;
    Tensor working_input = input_c;
    if (freq_bins != expected_bins) {
        if (expected_bins > freq_bins) {
            // Pad frequency bins with zeros along the FFT dimension on GPU
            std::vector<int64_t> pad_shape(shape.begin(), shape.end());
            pad_shape[dim] = expected_bins - freq_bins;
            Tensor zeros_pad = dispatchZeros(pad_shape, input_c.dtype(), input_c.device());
            working_input = dispatchCat({input_c, zeros_pad}, dim);
        } else {
            // Truncate frequency bins to expected_bins
            std::vector<int64_t> starts(ndim, 0);
            std::vector<int64_t> ends(shape.begin(), shape.end());
            std::vector<int64_t> steps(ndim, 1);
            ends[dim] = expected_bins;
            working_input = dispatchSlice(input_c, starts, ends, steps);
        }
        freq_bins = expected_bins;
        shape = working_input.shape();
    }

    bool can_cooley_tukey = is_power_of_2(output_len);
    bool can_mixed_radix = false;  // Mixed-radix Stockham path is buggy; use Bluestein

    int32_t device_id = input.device().index;
    DType real_dtype = is_f64 ? DType::Float64 : DType::Float32;
    DType complex_dtype = is_f64 ? DType::Complex128 : DType::Complex64;
    int64_t half_n = output_len / 2;
    size_t complex_elem_size = is_f64 ? 16 : 8;
    size_t real_elem_size = is_f64 ? 8 : 4;

    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 1; ++i) batch_size *= shape[i];

    // Output shape: [..., N] real
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[ndim - 1] = output_len;
    Tensor output(out_shape, real_dtype, input.device());

    // Helper lambda: Hermitian mirror on GPU — reconstruct full N-point spectrum from N/2+1 bins
    auto run_hermitian_mirror = [&](const Tensor& cont_input, int64_t b_idx) -> Tensor {
        Tensor full_spectrum({output_len}, complex_dtype, input.device());
        std::string shader = is_f64 ? "hermitian_mirror_f64" : "hermitian_mirror";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t half_plus_one;
            uint32_t full_size;
        } pc;
        pc.half_plus_one = static_cast<uint32_t>(freq_bins);
        pc.full_size = static_cast<uint32_t>(output_len);

        const void* in_ptr = static_cast<const char*>(cont_input.data_ptr())
                             + b_idx * freq_bins * complex_elem_size;
        size_t in_size = freq_bins * complex_elem_size;
        size_t out_size = output_len * complex_elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, in_ptr}, {1, full_spectrum.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(output_len, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return full_spectrum;
    };

    // Helper lambda: extract real parts from complex IFFT result on GPU
    auto run_complex_to_real = [&](const Tensor& ifft_result, int64_t b_idx) {
        std::string shader = is_f64 ? "complex_to_real_f64" : "complex_to_real";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t num_elements;
        } pc;
        pc.num_elements = static_cast<uint32_t>(output_len);

        size_t in_size = output_len * complex_elem_size;
        void* out_ptr = static_cast<char*>(const_cast<void*>(output.data_ptr()))
                        + b_idx * output_len * real_elem_size;
        size_t out_size = output_len * real_elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, ifft_result.data_ptr()}, {1, out_ptr}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(output_len, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    };

    // Mixed-radix path for non-power-of-2 sizes factorable into {2,3,5,7}
    if (can_mixed_radix && !is_power_of_2(output_len)) {
        auto cont = working_input.contiguous();

        for (int64_t b = 0; b < batch_size; ++b) {
            // Reconstruct full Hermitian spectrum on GPU
            Tensor full_spectrum = run_hermitian_mirror(cont, b);

            // Run inverse mixed-radix FFT
            Tensor ifft_result = runMixedRadixFFT(full_spectrum, output_len, 1, 1, static_cast<uint32_t>(output_len));

            // Scale by 1/N for the IFFT (mixed-radix does not apply normalization)
            runFFTScale(ifft_result, static_cast<uint32_t>(output_len),
                        1.0 / static_cast<double>(output_len));

            // Extract real parts on GPU
            run_complex_to_real(ifft_result, b);
        }

        // Apply normalization correction
        double applied_scale = 1.0 / static_cast<double>(output_len);
        double target_scale = fft_norm_factor(output_len, norm, /*is_forward=*/false);
        if (std::abs(target_scale - 1.0) < 1e-15) {
            target_scale = 1.0;
        }
        double correction = target_scale / applied_scale;
        if (std::abs(correction - 1.0) > 1e-15) {
            auto scale_tensor = dispatchFull({1}, static_cast<float>(correction), real_dtype);
            output = dispatchBinaryOp("mul", output,
                dispatchExpand(scale_tensor, std::vector<int64_t>(out_shape.begin(), out_shape.end())));
        }

        return output;
    }

    // Non-factorable non-power-of-2: reconstruct Hermitian spectrum on GPU,
    // run full IFFT via Bluestein, extract real parts on GPU
    if (!can_cooley_tukey) {
        auto cont = working_input.contiguous();

        for (int64_t b = 0; b < batch_size; ++b) {
            // Reconstruct full Hermitian spectrum on GPU
            Tensor full_spectrum = run_hermitian_mirror(cont, b);

            // Run inverse FFT via Bluestein
            Tensor ifft_result = dispatchFFTBluestein(full_spectrum, output_len, 1);

            // Scale by 1/N for the IFFT (Bluestein does not apply normalization)
            runFFTScale(ifft_result, static_cast<uint32_t>(output_len),
                        1.0 / static_cast<double>(output_len));

            // Extract real parts on GPU
            run_complex_to_real(ifft_result, b);
        }

        // Apply normalization correction: Bluestein IFFT + 1/N already applied.
        // Adjust to match requested norm.
        double applied_scale = 1.0 / static_cast<double>(output_len);
        double target_scale = fft_norm_factor(output_len, norm, /*is_forward=*/false);
        if (std::abs(target_scale - 1.0) < 1e-15) {
            target_scale = 1.0;
        }
        double correction = target_scale / applied_scale;
        if (std::abs(correction - 1.0) > 1e-15) {
            auto scale_tensor = dispatchFull({1}, static_cast<float>(correction), real_dtype);
            output = dispatchBinaryOp("mul", output,
                dispatchExpand(scale_tensor, std::vector<int64_t>(out_shape.begin(), out_shape.end())));
        }

        return output;
    }

    auto cont = working_input.contiguous();

    // Step 1: Pack N/2+1 complex freq bins into N/2 complex values — batched
    std::vector<int64_t> packed_shape = {batch_size * half_n};
    Tensor packed(packed_shape, complex_dtype, input.device());

    {
        std::string shader = is_f16 ? "irfft_pack_f16" : is_f64 ? "irfft_pack_f64" : "irfft_pack";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            uint32_t batch_size;
            uint32_t in_batch_stride;
            uint32_t out_batch_stride;
        } pc;
        pc.n = static_cast<uint32_t>(output_len);
        pc.batch_size = static_cast<uint32_t>(batch_size);
        // irfft_pack.comp reads `data_in[in_base + k * 2]`, so in_base is
        // a FLOAT index — in_batch_stride must be in float units:
        // `freq_bins * 2`. ✓
        pc.in_batch_stride = static_cast<uint32_t>(freq_bins * 2);
        // irfft_pack.comp writes `data_out[(out_base + k) * 2]`, so
        // `out_base + k` is a COMPLEX index — out_batch_stride must be
        // in complex units: `half_n`. The previous value (half_n * 2)
        // incorrectly used float units, causing batch b to write to
        // batch 2*b's output slot on multi-batch inputs. Paired with
        // the same bug in rfft_unpack, this was the root cause of the
        // Vulkan STFT "every other frame" / 2/3-amplitude bug on
        // multi-frame inputs.
        pc.out_batch_stride = static_cast<uint32_t>(half_n);

        // `cont` is the complex-dtype frequency-domain input. numel() counts
        // complex elements, not bytes. Match the RFFT unpack fix: use
        // complex_elem_size, not a phantom real-element size.
        size_t in_size = static_cast<size_t>(cont.numel()) * complex_elem_size;
        size_t out_size = batch_size * half_n * complex_elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, packed.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(half_n * batch_size, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    // Step 2: Run N/2-point inverse complex FFT — batched
    Tensor ifft_result = runFFTButterfly(packed, static_cast<uint32_t>(half_n), 1,
                                          static_cast<uint32_t>(batch_size),
                                          static_cast<uint32_t>(half_n));

    // Scale by 1/(N/2) for the IFFT
    runFFTScale(ifft_result, static_cast<uint32_t>(half_n * batch_size), 1.0 / static_cast<double>(half_n));

    // Step 3: Unpack N/2 complex values to N real values — batched
    {
        std::string shader = is_f16 ? "irfft_unpack_f16" : is_f64 ? "irfft_unpack_f64" : "irfft_unpack";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            uint32_t batch_size;
            uint32_t in_batch_stride;
            uint32_t out_batch_stride;
        } pc;
        pc.n = static_cast<uint32_t>(output_len);
        pc.batch_size = static_cast<uint32_t>(batch_size);
        pc.in_batch_stride = static_cast<uint32_t>(half_n * 2);
        pc.out_batch_stride = static_cast<uint32_t>(output_len);

        size_t in_size = batch_size * half_n * complex_elem_size;
        size_t out_size = output.numel() * real_elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, ifft_result.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(half_n * batch_size, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    // Apply IRFFT normalization correction.
    // We already applied 1/(N/2) in the inner IFFT. Adjust to match requested norm.
    // Default "backward": total scale should be 1/N, we have 1/(N/2) = 2/N, so multiply by 0.5
    // "ortho": total scale should be 1/sqrt(N), we have 2/N, so multiply by sqrt(N)/2
    // "forward": no normalization (scale=1), we have 2/N, so multiply by N/2
    double applied = 1.0 / static_cast<double>(half_n);  // What we already applied
    double target = fft_norm_factor(output_len, norm, /*is_forward=*/false);
    if (std::abs(target - 1.0) < 1e-15) {
        // norm == "forward": undo the applied scale
        target = 1.0;
    }
    double correction = target / applied;
    // For norm=="backward" or empty: target = 1/N, applied = 1/(N/2) = 2/N, correction = 0.5
    // For norm=="ortho": target = 1/sqrt(N), applied = 2/N, correction = N/(2*sqrt(N)) = sqrt(N)/2
    // For norm=="forward": target = 1, applied = 2/N, correction = N/2

    if (std::abs(correction - 1.0) > 1e-15) {
        auto scale_tensor = dispatchFull({1}, correction, real_dtype);
        output = dispatchBinaryOp("mul", output,
            dispatchExpand(scale_tensor, std::vector<int64_t>(out_shape.begin(), out_shape.end())));
    }

    return output;
}

auto VulkanBackend::dispatchFFT2(const Tensor& input, const std::vector<int64_t>& dims,
                                  const std::string& norm) -> Tensor {
    // 2D FFT = 1D FFT along each of the two dims sequentially
    auto result = input;
    for (auto d : dims) {
        int64_t actual_dim = d < 0 ? d + result.ndim() : d;
        int64_t n = result.shape()[actual_dim];
        result = dispatchFFT(result, d, n, norm);
    }
    return result;
}

auto VulkanBackend::dispatchIFFT2(const Tensor& input, const std::vector<int64_t>& dims,
                                   const std::string& norm) -> Tensor {
    auto result = input;
    for (auto d : dims) {
        int64_t actual_dim = d < 0 ? d + result.ndim() : d;
        int64_t n = result.shape()[actual_dim];
        result = dispatchIFFT(result, d, n, norm);
    }
    return result;
}

auto VulkanBackend::dispatchFFTN(const Tensor& input, const std::vector<int64_t>& dims,
                                  const std::string& norm) -> Tensor {
    auto result = input;
    for (auto d : dims) {
        int64_t actual_dim = d < 0 ? d + result.ndim() : d;
        int64_t n = result.shape()[actual_dim];
        result = dispatchFFT(result, d, n, norm);
    }
    return result;
}

auto VulkanBackend::dispatchIFFTN(const Tensor& input, const std::vector<int64_t>& dims,
                                   const std::string& norm) -> Tensor {
    auto result = input;
    for (auto d : dims) {
        int64_t actual_dim = d < 0 ? d + result.ndim() : d;
        int64_t n = result.shape()[actual_dim];
        result = dispatchIFFT(result, d, n, norm);
    }
    return result;
}


} // namespace tenzor
