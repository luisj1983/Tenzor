#include "vulkan_ops_common.hpp"
#include <cstdint>

namespace tenzor {

// ============================================================================
// Vision Operations Implementation
// ============================================================================

/**
 * @brief Gather relative position bias for Swin Transformer attention
 *
 * Gathers position bias values from a table using precomputed indices.
 * table: [table_size*table_size, num_heads] - position bias table
 * indices: [num_positions, num_positions] - lookup indices (Int64)
 * output: [num_positions, num_positions, num_heads] - gathered biases
 */
auto VulkanBackend::dispatchGatherRelativePositionBias(const Tensor& table, const Tensor& indices,
                                                        int64_t num_positions, int64_t num_heads) -> Tensor {
    int32_t device_id = table.device().index;

    // BFloat16 has no dedicated, correct shader (the *_bf16.comp variant is
    // dead and declares 32-bit `float` buffers, so it cannot read/write packed
    // 2-byte bf16 storage). Routing BFloat16 through the base float32 shader
    // would also misread the 2-byte buffer. Use the established widen→compute→
    // narrow pattern (see dispatchTranspose): cast the table to Float32, run
    // the op, then cast the gathered output back to BFloat16. This matches the
    // dedicated BFloat16 path the CPU/OneAPI backends provide for this op.
    if (table.dtype() == DType::BFloat16) {
        Tensor table_f32 = table.to(DType::Float32);
        Tensor out_f32 = dispatchGatherRelativePositionBias(table_f32, indices,
                                                            num_positions, num_heads);
        return out_f32.to(DType::BFloat16);
    }

    // Materialize read operands to packed, offset-0 buffers before binding.
    // The shader indexes table/indices linearly, so a strided or non-zero
    // offset view would bind the parent storage region and read logically
    // wrong elements (matches every sibling indexing op in
    // vulkan_ops_indexing.cpp: dispatchEmbedding/Gather/Scatter/IndexSelect).
    const Tensor table_c = dispatchContiguous(table);
    const Tensor indices_c = dispatchContiguous(indices);

    // Validate indices host-side so a malformed/hostile index table raises the
    // same std::out_of_range as the CPU reference
    // (src/backends/cpu/kernels/vision.cpp gather_relative_position_bias)
    // instead of the shader performing a logically-wrong (out-of-table) read.
    // In the Vulkan table layout the shader computes
    //   table_data[uint(table_idx) * num_heads + h]
    // so each table_idx must lie in [0, table_rows) where
    //   table_rows = table.numel() / num_heads.
    // Like the CPU reference, negative indices are NOT normalized here.
    {
        int64_t table_rows = num_heads > 0 ? (table.numel() / num_heads) : 0;
        Tensor idx_host = indices_c.to(Device::cpu());
        int64_t num_index = idx_host.numel();
        if (indices_c.dtype() == DType::Int64) {
            const int64_t* p = idx_host.data<int64_t>();
            for (int64_t i = 0; i < num_index; ++i) {
                int64_t v = p[i];
                if (v < 0 || v >= table_rows) {
                    throw std::out_of_range(
                        "gather_relative_position_bias: rel_pos_index value " +
                        std::to_string(v) + " out of range [0, " +
                        std::to_string(table_rows) + ")");
                }
            }
        } else {
            const int32_t* p = idx_host.data<int32_t>();
            for (int64_t i = 0; i < num_index; ++i) {
                int64_t v = static_cast<int64_t>(p[i]);
                if (v < 0 || v >= table_rows) {
                    throw std::out_of_range(
                        "gather_relative_position_bias: rel_pos_index value " +
                        std::to_string(v) + " out of range [0, " +
                        std::to_string(table_rows) + ")");
                }
            }
        }
    }

    // Select shader based on dtype
    std::string shader_name;
    switch (table_c.dtype()) {
        case DType::Float64:
            shader_name = "gather_relative_position_bias_f64";
            break;
        case DType::Float16:
            shader_name = "gather_relative_position_bias_f16";
            break;
        default:
            shader_name = "gather_relative_position_bias";
            break;
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Output shape: [num_positions, num_positions, num_heads]
    std::vector<int64_t> out_shape = {num_positions, num_positions, num_heads};
    Tensor output(out_shape, table_c.dtype(), table_c.device());

    uint32_t total_elements = static_cast<uint32_t>(num_positions * num_positions * num_heads);

    // Push constants
    struct PushConstants {
        uint32_t num_positions;
        uint32_t num_heads;
        uint32_t total_elements;
    } push_constants;

    push_constants.num_positions = static_cast<uint32_t>(num_positions);
    push_constants.num_heads = static_cast<uint32_t>(num_heads);
    push_constants.total_elements = total_elements;

    // Get VkBuffer handles
    const void* buffer_table = table_c.data_ptr();
    const void* buffer_indices = indices_c.data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes
    size_t table_size = table_c.numel() * table_c.dtype_size();
    size_t indices_size = indices_c.numel() * indices_c.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_table},
        {1, buffer_indices},
        {2, buffer_output}
    };
    std::vector<size_t> sizes = {table_size, indices_size, output_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch compute workgroups
    uint32_t workgroups = div_wg_checked(total_elements, devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch");
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Shape Operations Implementation
// ============================================================================

/**
 * @brief Reshape tensor - metadata-only operation (no data movement)
 *
 * Reshapes the tensor to the new shape. This is a metadata-only operation
 * that doesn't move data, just creates a new view with different dimensions.
 */
auto VulkanBackend::dispatchReshape(const Tensor& input, const std::vector<int64_t>& new_shape) -> Tensor {
    // Verify total elements match
    int64_t old_numel = input.numel();
    int64_t new_numel = 1;
    for (int64_t dim : new_shape) {
        new_numel *= dim;
    }

    if (old_numel != new_numel) {
        throw std::invalid_argument(
            "Reshape: total elements must match (old=" + std::to_string(old_numel) +
            ", new=" + std::to_string(new_numel) + ")"
        );
    }

    // For contiguous tensors, reshape is just metadata manipulation
    // Create a view that shares storage with the input tensor
    if (!input.is_contiguous()) {
        // Need to make contiguous first, then reshape
        Tensor contiguous = dispatchContiguous(input);
        return dispatchReshape(contiguous, new_shape);
    }

    // Create new tensor that shares storage (view)
    Tensor result;
    result.impl_ = make_intrusive<TensorImpl>(*input.impl_);
    result.mutable_shape() = new_shape;
    result.mutable_strides() = tenzor::compute_strides(new_shape);

    return result;
}

/**
 * @brief Transpose two dimensions using compute shader
 *
 * Swaps two dimensions of the tensor, reordering data in memory according
 * to the transposed layout.
 */
auto VulkanBackend::dispatchTranspose(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor {
    auto input_shape = input.shape();
    int32_t ndim = input.ndim();

    // Normalize negative dimensions
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;

    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::invalid_argument("Transpose: dimension out of range");
    }

    // Create output shape
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    std::swap(out_shape[dim0], out_shape[dim1]);

    // Calculate output strides
    std::vector<int64_t> out_strides(ndim);
    out_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    int32_t device_id = input.device().index;

    // For simple 2D transpose or contiguous case, use optimized path.
    // Complex128 has 16 bytes per element — delegate to dispatchPermute which
    // knows how to pick permute_c128. Otherwise route 8-byte types (Float64,
    // Complex64) through the transform_f64 shader (treats each element as
    // one 8-byte slot), and everything else through the 4-byte transform
    // shader.
    if (ndim == 2 && input.is_contiguous() && input.dtype() != DType::Complex128) {
        // Use simplified transform shader for 2D case
        // For Float16/BFloat16, convert to Float32, transpose, convert back
        DType orig_dtype = input.dtype();
        Tensor transpose_input = input;
        if (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16) {
            transpose_input = input.to(DType::Float32);
        }
        std::string shader_name;
        if (transpose_input.dtype() == DType::Float64
            || transpose_input.dtype() == DType::Int64
            || transpose_input.dtype() == DType::Complex64) {
            // All 8-byte-per-element dtypes share the f64 transform shader.
            shader_name = "transform_f64";
        } else {
            shader_name = "transform";
        }
        auto* pipeline = getPipeline(shader_name, device_id);
        Tensor output(out_shape, transpose_input.dtype(), transpose_input.device());

        const void* buffer_in = transpose_input.data_ptr();
        const void* buffer_out = output.data_ptr();

        size_t buffer_size_in = transpose_input.numel() * transpose_input.dtype_size();
        size_t buffer_size_out = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_in},
            {1, buffer_out}
        };
        std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t n;
            uint32_t ndim;
            uint32_t transform;
            uint32_t rows;
            uint32_t cols;
        } push_constants;

        push_constants.n = static_cast<uint32_t>(transpose_input.numel());
        push_constants.ndim = static_cast<uint32_t>(ndim);
        push_constants.transform = 1; // transpose
        push_constants.rows = static_cast<uint32_t>(input_shape[0]);
        push_constants.cols = static_cast<uint32_t>(input_shape[1]);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);

        // Insert pre-read barrier to ensure input data from previous ops is ready
        insertComputeOnlyBarrier(cmdBuffer);

        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = div_wg_checked(transpose_input.numel(), devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch");
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);

        // Convert back to original dtype if we converted F16/BF16 to F32
        if (orig_dtype != output.dtype()) {
            return output.to(orig_dtype);
        }
        return output;
    }

    // For general N-D transpose, delegate to dispatchPermute with appropriate
    // permutation vector. Transpose(dim0, dim1) is equivalent to permute where
    // the permutation swaps dim0 and dim1 and leaves all other dims in place.
    std::vector<int64_t> perm(ndim);
    for (int32_t i = 0; i < ndim; ++i) perm[i] = i;
    std::swap(perm[dim0], perm[dim1]);
    return dispatchPermute(input, perm);
}

/**
 * @brief Permute dimensions using compute shader
 *
 * Reorders dimensions according to the specified permutation.
 */
auto VulkanBackend::dispatchPermute(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor {
    auto input_shape = input.shape();
    auto input_strides = input.strides();
    int32_t ndim = input.ndim();
    int32_t device_id = input.device().index;

    // Validate permutation
    if (static_cast<int64_t>(dims.size()) != ndim) {
        throw std::invalid_argument("Permute: number of dimensions doesn't match");
    }

    std::vector<bool> seen(ndim, false);
    for (int64_t dim : dims) {
        if (dim < 0 || dim >= ndim || seen[dim]) {
            throw std::invalid_argument("Permute: invalid permutation");
        }
        seen[dim] = true;
    }

    // Create output shape by permuting input shape
    std::vector<int64_t> out_shape;
    for (int64_t dim : dims) {
        out_shape.push_back(input_shape[dim]);
    }

    // Create output tensor
    Tensor output(out_shape, input.dtype(), input.device());

    // Get pipeline - select dtype-specific shader variant
    std::string permute_shader;
    if (input.dtype() == DType::Complex128) {
        // 16-byte-per-element shader — each output slot copies two float64 halves.
        permute_shader = "permute_c128";
    }
    else if (input.dtype() == DType::Float64 || input.dtype() == DType::Int64 ||
        input.dtype() == DType::Complex64) {
        // 8-byte types share the f64 shader (uvec2/uint64 layout)
        permute_shader = "permute_f64";
    }
    else if (input.dtype() == DType::Float16) permute_shader = "permute_f16";
    else if (input.dtype() == DType::BFloat16) permute_shader = "permute_bf16";
    else permute_shader = "permute";
    auto* pipeline = getPipeline(permute_shader, device_id);

    // Get Vulkan buffers for input and output
    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();

    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Convert int64_t to int32_t for shader compatibility.
    std::vector<int32_t> shape_i32(ndim);
    std::vector<int32_t> strides_i32(ndim);
    std::vector<int32_t> dims_i32(ndim);
    for (int32_t i = 0; i < ndim; ++i) {
        shape_i32[i] = static_cast<int32_t>(input_shape[i]);
        strides_i32[i] = static_cast<int32_t>(input_strides[i]);
        dims_i32[i] = static_cast<int32_t>(dims[i]);
    }

    size_t metadata_size = ndim * sizeof(int32_t);

    // Allocate metadata buffers through the main allocator so their pointers
    // are tracked — the previous standalone VulkanBuffer path returned raw
    // VkBuffer handles that allocateAndWriteDescriptorSet's lookup couldn't
    // resolve, producing "Invalid buffer pointer: buffer not tracked" when
    // dispatchPermute was invoked from higher-level ops like fft2.
    void* ptr_shape   = allocate(metadata_size, device_id);
    void* ptr_strides = allocate(metadata_size, device_id);
    void* ptr_perm    = allocate(metadata_size, device_id);

    copy(ptr_shape,   shape_i32.data(),   metadata_size, CopyKind::HostToDevice);
    copy(ptr_strides, strides_i32.data(), metadata_size, CopyKind::HostToDevice);
    copy(ptr_perm,    dims_i32.data(),    metadata_size, CopyKind::HostToDevice);

    // Set up descriptor set with all buffers
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output},
        {2, ptr_shape},
        {3, ptr_strides},
        {4, ptr_perm}
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_output,
        metadata_size,
        metadata_size,
        metadata_size
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants
    struct PushConstants {
        uint32_t n;
        uint32_t ndim;
    } push_constants;

    push_constants.n = static_cast<uint32_t>(output.numel());
    push_constants.ndim = static_cast<uint32_t>(ndim);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);

    // Insert pre-read barrier to ensure input data from previous ops is ready
    insertComputeOnlyBarrier(cmdBuffer);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg_checked(output.numel(), devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch");
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    // Ensure the compute dispatch is done before freeing metadata buffers.
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        submitBatchIfNeeded(device_id, true);
        ensurePendingWorkComplete(device_id);
    }

    deallocate(ptr_shape);
    deallocate(ptr_strides);
    deallocate(ptr_perm);

    return output;
}

/**
 * @brief Squeeze - remove dimensions of size 1 (metadata-only)
 */
auto VulkanBackend::dispatchSqueeze(const Tensor& input, int64_t dim) -> Tensor {
    // Make this a pure metadata-only operation like the core Tensor::squeeze()
    // This prevents potential recursion through reshape → contiguous

    auto input_shape = input.shape();
    auto input_strides = input.strides();
    int32_t ndim = input.ndim();

    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_strides;

    // AUTOGRAD-R047: the sentinel for "no dim supplied -> squeeze all" must
    // be a value that cannot also be a legitimate (possibly negative) axis
    // index. The previous `dim < 0` check treated EVERY negative dim,
    // including -1 (last axis), as squeeze-all, silently ignoring a
    // specific requested axis; it also made the negative-dim normalization
    // below dead code, since the else-branch could only ever see dim >= 0.
    // Matches cpu::squeeze_kernel's SQUEEZE_ALL contract exactly.
    if (dim == std::numeric_limits<int64_t>::min()) {
        // Squeeze all dimensions of size 1
        for (int64_t i = 0; i < ndim; i++) {
            if (input_shape[i] != 1) {
                new_shape.push_back(input_shape[i]);
                new_strides.push_back(input_strides[i]);
            }
        }
    } else {
        // Normalize negative dimension
        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) {
            throw std::invalid_argument("Squeeze: dimension out of range");
        }

        // Squeeze specific dimension. PyTorch leaves a non-size-1 axis
        // untouched rather than throwing (mirrors cpu::squeeze_kernel).
        if (input_shape[dim] != 1) {
            return input;
        }

        for (int64_t i = 0; i < ndim; i++) {
            if (i != dim) {
                new_shape.push_back(input_shape[i]);
                new_strides.push_back(input_strides[i]);
            }
        }
    }

    // If all dimensions were size 1, keep at least one
    if (new_shape.empty()) {
        new_shape.push_back(1);
        new_strides.push_back(1);
    }

    // Create result tensor sharing storage (zero-copy metadata-only operation)
    Tensor result;
    result.impl_ = make_intrusive<TensorImpl>(*(input.impl_));
    result.mutable_shape() = std::move(new_shape);
    result.mutable_strides() = std::move(new_strides);

    return result;
}

/**
 * @brief Unsqueeze - add dimension of size 1 (metadata-only)
 */
auto VulkanBackend::dispatchUnsqueeze(const Tensor& input, int64_t dim) -> Tensor {
    auto input_shape = input.shape();
    int32_t ndim = input.ndim();

    // Normalize negative dimension (allow ndim as well for appending)
    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim) {
        throw std::invalid_argument("Unsqueeze: dimension out of range");
    }

    // Create output shape with new dimension of size 1
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    out_shape.insert(out_shape.begin() + dim, 1);

    // Metadata-only operation - just reshape
    return dispatchReshape(input, out_shape);
}

/**
 * @brief Contiguous - ensure tensor is contiguous in memory
 *
 * If already contiguous with zero offset, returns the input tensor.
 * Otherwise, creates a new contiguous copy using GPU strided_copy kernel.
 */
auto VulkanBackend::dispatchContiguous(const Tensor& input) -> Tensor {
    // If already contiguous AND starts at offset 0, return as-is.
    // Views (slices) may be stride-contiguous but sit at an offset within
    // a shared parent buffer; they need to be copied to own storage.
    if (input.is_contiguous() && input.offset() == 0) {
        return input;
    }

    // For non-contiguous tensors, use GPU kernel to reorder the data
    const int64_t total_elements = input.numel();
    const int64_t ndims = input.ndim();
    const int64_t base_offset = input.is_valid() ? input.offset() : 0;

    // Create new contiguous tensor with same shape, dtype, device
    Tensor result(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    if (total_elements == 0) {
        return result;
    }

    int32_t device_id = input.device().index;

    // Select shader based on dtype (element size must match shader buffer layout)
    std::string shader_name = "strided_copy";
    if (input.dtype() == DType::Float64 || input.dtype() == DType::Int64 ||
        input.dtype() == DType::UInt64 || input.dtype() == DType::Complex64) {
        shader_name = "strided_copy_f64";  // uvec2 layout works for any 8-byte type
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16 ||
               input.dtype() == DType::Int16 || input.dtype() == DType::UInt16) {
        // strided_copy_f16 moves raw 16-bit words (bit copy), so it serves all
        // 2-byte dtypes, not just half floats.
        shader_name = "strided_copy_f16";
    } else if (input.dtype() == DType::UInt8 || input.dtype() == DType::Bool ||
               input.dtype() == DType::Int8 ||
               input.dtype() == DType::FP8_E4M3 || input.dtype() == DType::FP8_E5M2 ||
               input.dtype() == DType::FP8_E4M3FNUZ || input.dtype() == DType::FP8_E5M2FNUZ) {
        // strided_copy_u8 moves raw bytes, so it serves all 1-byte dtypes.
        shader_name = "strided_copy_u8";
    } else if (input.dtype() == DType::Complex128) {
        // F18: 16-byte element wired through new `strided_copy_c128.comp`
        // shader (a uvec4 port of strided_copy_f64.comp). Previously this
        // path threw — every Complex128 op that needed contiguous input
        // (FFT, MatMul, almost anything reshape-flavoured) failed outright.
        shader_name = "strided_copy_c128";
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Get Vulkan buffers - use base storage pointer for input
    const void* base_storage_ptr = input.is_valid() ? input.storage()->data() : input.data_ptr();
    const void* buffer_in = const_cast<void*>(base_storage_ptr);
    const void* buffer_out = result.data_ptr();

    // Calculate buffer sizes
    int64_t max_offset = base_offset;
    auto strides = input.strides();
    auto shape = input.shape();
    if (ndims > 0) {
        for (int64_t dim = 0; dim < ndims; ++dim) {
            max_offset += (shape[dim] - 1) * std::abs(strides[dim]);
        }
    }
    // The strided_copy shaders accumulate the source offset in a signed 32-bit
    // int (`int src_offset`) and address the output via a 32-bit `uint` flat
    // index. A tensor whose largest reachable source offset exceeds INT32_MAX
    // would overflow src_offset to negative and read out of bounds; an output
    // with >UINT32_MAX elements would wrap the flat index. The shader has no
    // 64-bit path (shaderInt64 is not enabled on the device), so reject such
    // tensors up front with a clear message rather than corrupt memory.
    if (max_offset > static_cast<int64_t>(INT32_MAX) ||
        total_elements > static_cast<int64_t>(UINT32_MAX)) {
        throw std::runtime_error(
            "Vulkan strided copy: tensor too large for 32-bit indexing "
            "(max source offset " + std::to_string(max_offset) +
            ", elements " + std::to_string(total_elements) +
            "); exceeds INT32_MAX/UINT32_MAX");
    }
    // The strided_copy push constants only carry shape[0..7]/strides[0..7] and
    // the shader bails out (`if (ndims > 8) return;`) for higher rank — which
    // would leave `result` Vulkan-default-zeroed and silently return an all-zeros
    // tensor. Reject >8-D up front (mirrors the AdvancedIndex ndim guard).
    if (ndims > 8) {
        throw std::runtime_error(
            "Vulkan strided copy (contiguous): ndim " + std::to_string(ndims) +
            " > 8 is unsupported");
    }

    size_t input_buffer_size = (max_offset + 1) * input.dtype_size();
    size_t output_buffer_size = total_elements * input.dtype_size();

    // The strided_copy_f16/bf16/u8/i8 shaders access the buffer as uint32
    // words (2x fp16, 4x u8). A sub-dword binding makes Vulkan robust
    // buffer access return zero for any word that isn't fully in range —
    // which silently drops the last packed element when the slice length
    // is odd or lands at an odd offset. Round both buffer sizes up to the
    // next 4-byte boundary so every word the shader touches is in-bounds.
    input_buffer_size  = (input_buffer_size  + 3u) & ~size_t(3);
    output_buffer_size = (output_buffer_size + 3u) & ~size_t(3);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {input_buffer_size, output_buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants structure matching the shader
    // IMPORTANT: The shader uses uvec4/ivec4 which require 16-byte alignment
    // We need padding after base_offset to align the first uvec4
    struct PushConstants {
        uint32_t n_elements;
        uint32_t ndims;
        uint32_t base_offset;
        uint32_t _padding;  // Padding to align shape_0_3 to 16 bytes
        uint32_t shape_0_3[4];   // shape[0..3] as uvec4
        uint32_t shape_4_7[4];   // shape[4..7] as uvec4
        int32_t strides_0_3[4];  // strides[0..3] as ivec4
        int32_t strides_4_7[4];  // strides[4..7] as ivec4
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(total_elements);
    push_constants.ndims = static_cast<uint32_t>(ndims);
    push_constants.base_offset = static_cast<uint32_t>(base_offset);
    push_constants._padding = 0;

    // Initialize all shape/stride values to defaults
    for (int i = 0; i < 4; ++i) {
        push_constants.shape_0_3[i] = 1;
        push_constants.shape_4_7[i] = 1;
        push_constants.strides_0_3[i] = 0;
        push_constants.strides_4_7[i] = 0;
    }

    // Fill in actual shape and strides
    for (int64_t i = 0; i < ndims && i < 4; ++i) {
        push_constants.shape_0_3[i] = static_cast<uint32_t>(shape[i]);
        push_constants.strides_0_3[i] = static_cast<int32_t>(strides[i]);
    }
    for (int64_t i = 4; i < ndims && i < 8; ++i) {
        push_constants.shape_4_7[i - 4] = static_cast<uint32_t>(shape[i]);
        push_constants.strides_4_7[i - 4] = static_cast<int32_t>(strides[i]);
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);

    // Insert pre-read barrier to ensure input data from previous ops is ready
    // This is critical for strided_copy which reads non-contiguous data
    insertComputeOnlyBarrier(cmdBuffer);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg_checked(total_elements, devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch");
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    // Ensure strided_copy completes before result is used.
    // Without this, command batching defers GPU submission, causing hangs
    // when the backward pass chain reads the result immediately.
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        submitBatchIfNeeded(device_id, true);   // Force immediate submit
        ensurePendingWorkComplete(device_id);   // Wait for GPU completion
    }

    return result;
}

} // namespace tenzor
