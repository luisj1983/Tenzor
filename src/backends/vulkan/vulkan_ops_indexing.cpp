#include "vulkan_ops_common.hpp"

namespace tenzor {

// Indexing operations implementation
auto VulkanBackend::dispatchEmbedding(const Tensor& weight, const Tensor& indices,
                                      int64_t padding_idx) -> Tensor {
    auto weight_shape = weight.shape();
    auto indices_shape = indices.shape();

    int32_t device_id = weight.device().index;
    bool is_float64 = (weight.dtype() == DType::Float64);
    bool is_float16 = (weight.dtype() == DType::Float16);
    std::string shader_name = is_float64 ? "embedding_f64"
                            : is_float16 ? "embedding_f16"
                            : "embedding";
    auto* pipeline = getPipeline(shader_name, device_id);

    uint32_t num_embeddings = static_cast<uint32_t>(weight_shape[0]);
    uint32_t embedding_dim = static_cast<uint32_t>(weight_shape[1]);
    uint32_t num_indices = static_cast<uint32_t>(indices.numel());

    // Output shape: indices_shape + [embedding_dim]
    std::vector<int64_t> out_shape(indices_shape.begin(), indices_shape.end());
    out_shape.push_back(weight_shape[1]);

    Tensor output(out_shape, weight.dtype(), weight.device());

    // Handle empty tensors
    if (num_indices == 0) {
        return output;
    }

    // Materialize read operands to packed offset-0 buffers before binding.
    const Tensor weight_c = dispatchContiguous(weight);
    const Tensor indices_packed = dispatchContiguous(indices);

    // Convert Int64 indices to Int32 for shader compatibility
    Tensor indices_i32 = indices_packed;
    if (indices.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(indices_shape.begin(), indices_shape.end());
        indices_i32 = Tensor(idx_shape, DType::Int32, indices.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = indices_packed.data_ptr();
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

        VkCommandBuffer cast_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cast_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cast_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cast_cmd, cast_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cast_cmd, div_wg(indices.numel(), devices_[device_id].workgroupSize), 1, 1);
        insertComputeBarrier(cast_cmd);
        endSingleTimeCommands(cast_cmd, device_id);
        // Under USE_COMMAND_BATCHING the embedding kernel records into a
        // separate command buffer, so the intra-buffer barrier above does not
        // order the cross-buffer dependency on indices_i32. Force completion of
        // the cast before the kernel reads it (matches dispatchGather /
        // dispatchIndexSelect / dispatchScatter siblings).
        synchronize(device_id);
    }

    // Get VkBuffer handles
    const void* buf_weight = weight_c.data_ptr();
    const void* buf_indices = indices_i32.data_ptr();
    const void* buf_output = output.data_ptr();

    size_t weight_buf_size = weight.numel() * weight.dtype_size();
    size_t indices_buf_size = indices_i32.numel() * sizeof(int32_t);
    size_t output_buf_size = output.numel() * output.dtype_size();

    // Bindings: embeddings(0), indices(1), output(2)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_weight},
        {1, buf_indices},
        {2, buf_output}
    };
    std::vector<size_t> sizes = {weight_buf_size, indices_buf_size, output_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants
    struct PushConstants {
        uint32_t num_embeddings;
        uint32_t embedding_dim;
        uint32_t num_indices;
        uint32_t padding_idx;
    } push_constants;

    push_constants.num_embeddings = num_embeddings;
    push_constants.embedding_dim = embedding_dim;
    push_constants.num_indices = num_indices;
    push_constants.padding_idx = (padding_idx >= 0) ? static_cast<uint32_t>(padding_idx) : 0xFFFFFFFFu;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One thread per index (each thread copies embedding_dim elements)
    uint32_t workgroups = div_wg(num_indices, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchGather(const Tensor& input, int64_t dim, const Tensor& indices) -> Tensor {
    auto indices_shape = indices.shape();
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // Handle empty tensors
    if (input.numel() == 0 || indices.numel() == 0) {
        std::vector<int64_t> out_shape(indices_shape.begin(), indices_shape.end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Dispatch by element representation. The generic "gather" shader copies
    // 32-bit slots; 8-byte/16-bit/16-byte dtypes need width-matched movers.
    // gather_f64 / gather_c128 / gather_i16 are pure bit-preserving copies, so
    // they serve any same-width dtype (e.g. Complex64 via the 8-byte mover).
    const DType gdt = input.dtype();
    std::string shader_name;
    if (gdt == DType::Float64 || gdt == DType::Int64 || gdt == DType::UInt64 ||
        gdt == DType::Complex64) {
        shader_name = "gather_f64";
    } else if (gdt == DType::Float16) {
        shader_name = "gather_f16";
    } else if (gdt == DType::BFloat16) {
        shader_name = "gather_bf16";
    } else if (gdt == DType::Int16 || gdt == DType::UInt16) {
        shader_name = "gather_i16";
    } else if (gdt == DType::Complex128) {
        shader_name = "gather_c128";
    } else {
        shader_name = "gather";  // Float32 / Int32 / UInt32
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Normalize dimension
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Gather: dimension out of range");
    }

    std::vector<int64_t> out_shape(indices_shape.begin(), indices_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Materialize read operands to packed offset-0 buffers before binding.
    const Tensor input_c = dispatchContiguous(input);
    const Tensor indices_packed = dispatchContiguous(indices);

    // Convert Int64 indices to Int32 for shader compatibility
    Tensor indices_int32 = indices_packed;
    if (indices.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(indices_shape.begin(), indices_shape.end());
        indices_int32 = Tensor(idx_shape, DType::Int32, indices.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = indices_packed.data_ptr();
        const void* buf_out = indices_int32.data_ptr();
        size_t size_in = indices.numel() * sizeof(int64_t);
        size_t size_out = indices_int32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(indices.numel());

        uint32_t cast_groups = div_wg(cast_pc.n_elements, devices_[device_id].workgroupSize);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, cast_groups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Get Vulkan buffers
    const void* buffer_input = input_c.data_ptr();
    const void* buffer_indices = indices_int32.data_ptr();
    const void* buffer_output = output.data_ptr();

    // F16/BF16 data buffers are packed two halves per 32-bit word and the shader
    // reads element e at word e/2 (4-byte read + CAS). For odd numel the last
    // word's 4-byte span extends 2 bytes past numel*2, so size the SSBO range as
    // ((numel + 1) / 2) * 4. Indices stay int32. Matches FFT/complex sizing.
    const bool gather_packed_half = (gdt == DType::Float16 || gdt == DType::BFloat16);
    size_t buffer_size_input = gather_packed_half
        ? ((input.numel() + 1) / 2) * 4
        : input.numel() * input.dtype_size();
    size_t buffer_size_indices = indices_int32.numel() * sizeof(int32_t);
    size_t buffer_size_output = gather_packed_half
        ? ((output.numel() + 1) / 2) * 4
        : output.numel() * output.dtype_size();

    // Set up descriptor set with all buffers
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_indices},
        {2, buffer_output}
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_indices,
        buffer_size_output
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate gather parameters
    // dim_size = indices_shape[dim] for output index decomposition
    // input_dim_size = input_shape[dim] for bounds check and input indexing
    uint32_t dim_size = static_cast<uint32_t>(indices_shape[dim]);
    uint32_t input_dim_size = static_cast<uint32_t>(input_shape[dim]);
    uint32_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= static_cast<uint32_t>(indices_shape[d]);
    }
    uint32_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) {
        outer_size *= static_cast<uint32_t>(indices_shape[d]);
    }

    // Push constants matching shader layout
    struct PushConstants {
        uint32_t input_size;
        uint32_t output_size;
        uint32_t dim;
        uint32_t dim_size;
        uint32_t inner_size;
        uint32_t outer_size;
        uint32_t input_dim_size;
    } push_constants;

    push_constants.input_size = static_cast<uint32_t>(input.numel());
    push_constants.output_size = static_cast<uint32_t>(output.numel());
    push_constants.dim = static_cast<uint32_t>(dim);
    push_constants.dim_size = dim_size;
    push_constants.inner_size = inner_size;
    push_constants.outer_size = outer_size;
    push_constants.input_dim_size = input_dim_size;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchScatter(const Tensor& input_raw, int64_t dim, const Tensor& indices_raw,
                                    const Tensor& values_raw, int64_t reduction) -> Tensor {
    // audit-2026-05-03 bug #15 mirror: ensure all inputs are contiguous before
    // SSBO upload. Non-contiguous slice/expand views skip logical elements.
    // Use dispatchContiguous (not .contiguous()): a stride-contiguous offset
    // view is left untouched by .contiguous() and would trip the descriptor
    // offset guard. The scattered result is written to a freshly-allocated
    // `output` below (not in place into `input`), so materializing these read
    // operands is safe and does not discard any in-place update.
    auto input = dispatchContiguous(input_raw);
    auto indices = dispatchContiguous(indices_raw);
    auto values = dispatchContiguous(values_raw);
    auto input_shape = input.shape();

    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0 || indices.numel() == 0) {
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    int32_t device_id = input.device().index;

    // Float64 scatter with reduction requires atomic int64 support for CAS-loop atomics
    if (input.dtype() == DType::Float64 && reduction != 0 &&
        !devices_[device_id].hasAtomicInt64) {
        throw std::runtime_error(
            "Scatter with Float64 reduction requires VK_KHR_shader_atomic_int64 support. "
            "Use CPU backend or reduction=0 (direct assignment) for this device.");
    }

    // Select shader by element representation. scatter_f64 (8-byte) and
    // scatter_c128 (16-byte) are bit-preserving for any same-width dtype on
    // overwrite; scatter_i16 packs 2x 16-bit. Generic scatter covers 32-bit.
    const DType sdt = input.dtype();
    const char* shader_name =
        (sdt == DType::Float64 || sdt == DType::Int64 || sdt == DType::UInt64 ||
         sdt == DType::Complex64) ? "scatter_f64"
        : (sdt == DType::Float16) ? "scatter_f16"
        : (sdt == DType::BFloat16) ? "scatter_bf16"
        : (sdt == DType::Int16 || sdt == DType::UInt16) ? "scatter_i16"
        : (sdt == DType::Complex128) ? "scatter_c128"
        : "scatter";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Normalize dimension
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Scatter: dimension out of range");
    }

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // First copy input to output
    size_t bytes = input.numel() * input.dtype_size();
    copy(output.data_ptr(), input.data_ptr(), bytes, CopyKind::DeviceToDevice);

    // Convert Int64 indices to Int32 for shader compatibility (on-device)
    Tensor indices_int32 = indices;
    if (indices.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(indices.shape().begin(), indices.shape().end());
        indices_int32 = Tensor(idx_shape, DType::Int32, indices.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = indices.data_ptr();
        const void* buf_out = indices_int32.data_ptr();
        size_t size_in = indices.numel() * sizeof(int64_t);
        size_t size_out = indices_int32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(indices.numel());

        uint32_t cast_groups = div_wg(cast_pc.n_elements, devices_[device_id].workgroupSize);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, cast_groups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Get Vulkan buffers
    const void* buffer_input = input.data_ptr();
    const void* buffer_indices = indices_int32.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_values = values.data_ptr();

    // F16/BF16 data buffers are packed two halves per 32-bit word; the shader
    // reads element e at word e/2 (4-byte read + CAS). Size SSBO ranges as
    // ((numel + 1) / 2) * 4 so an odd-numel last word stays in bounds. Indices
    // remain int32.
    const bool scatter_packed_half = (sdt == DType::Float16 || sdt == DType::BFloat16);
    size_t buffer_size_input = scatter_packed_half
        ? ((input.numel() + 1) / 2) * 4
        : input.numel() * input.dtype_size();
    size_t buffer_size_indices = indices_int32.numel() * sizeof(int32_t);
    size_t buffer_size_output = scatter_packed_half
        ? ((output.numel() + 1) / 2) * 4
        : output.numel() * output.dtype_size();
    size_t buffer_size_values = scatter_packed_half
        ? ((values.numel() + 1) / 2) * 4
        : values.numel() * values.dtype_size();

    // Set up descriptor set with all buffers
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_indices},
        {2, buffer_output},
        {3, buffer_values}
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_indices,
        buffer_size_output,
        buffer_size_values
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate scatter parameters
    // Output (destination) tensor parameters
    uint32_t output_dim_size = static_cast<uint32_t>(input_shape[dim]);
    uint32_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= static_cast<uint32_t>(input_shape[d]);
    }
    uint32_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) {
        outer_size *= static_cast<uint32_t>(input_shape[d]);
    }

    // Input (values) tensor dimension size
    auto values_shape = values.shape();
    uint32_t values_dim_size = static_cast<uint32_t>(values_shape[dim]);

    // Push constants for scatter shader
    struct PushConstants {
        uint32_t input_size;
        uint32_t output_size;
        uint32_t dim;
        uint32_t dim_size;
        uint32_t input_dim_size;
        uint32_t inner_size;
        uint32_t outer_size;
        uint32_t reduction;
        uint32_t use_values;
    } push_constants;

    push_constants.input_size = static_cast<uint32_t>(indices.numel());
    push_constants.output_size = static_cast<uint32_t>(output.numel());
    push_constants.dim = static_cast<uint32_t>(dim);
    push_constants.dim_size = output_dim_size;
    push_constants.input_dim_size = values_dim_size;
    push_constants.inner_size = inner_size;
    push_constants.outer_size = outer_size;
    push_constants.reduction = static_cast<uint32_t>(reduction);
    push_constants.use_values = 1;  // Use values buffer

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(indices.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchIndexSelect(const Tensor& input, int64_t dim, const Tensor& indices) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;
    int32_t ndim = input.ndim();

    // Normalize negative dimension
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Invalid dimension for index_select");
    }

    // Calculate output shape
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    out_shape[dim] = indices.numel();
    Tensor output(out_shape, input.dtype(), input.device());

    // Handle empty tensors - no GPU work needed
    if (output.numel() == 0 || indices.numel() == 0) {
        return output;
    }

    // Select correct shader based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Float64) shader_name = "index_select_f64";
    else if (input.dtype() == DType::Float16) shader_name = "index_select_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "index_select_bf16";
    else if (input.dtype() == DType::Int64) shader_name = "index_select_i64";
    else if (input.dtype() == DType::Int32) shader_name = "index_select_i32";
    else if (input.dtype() == DType::Bool) shader_name = "index_select_bool";
    // 8-byte (UInt64 / Complex64) and 16-byte (Complex128) move bit-exact via
    // the f64/c128 shaders — far better than widening complex through Float32
    // (which would drop the imaginary part) or losing UInt64 precision.
    else if (input.dtype() == DType::UInt64 || input.dtype() == DType::Complex64)
        shader_name = "index_select_f64";
    else if (input.dtype() == DType::Complex128) shader_name = "index_select_c128";
    else shader_name = "index_select";

    // For Int8/UInt8: cast to Int32, do index_select, cast back
    if (input.dtype() == DType::Int8 || input.dtype() == DType::UInt8) {
        DType orig_dtype = input.dtype();
        auto input_cast = input.to(DType::Int32);
        auto result_cast = dispatchIndexSelect(input_cast, dim, indices);
        return result_cast.to(orig_dtype);
    }

    // For any remaining unsupported dtypes: cast to Float32, run on GPU, cast back (no CPU fallback)
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64 &&
        input.dtype() != DType::Float16 && input.dtype() != DType::BFloat16 &&
        input.dtype() != DType::Int64 && input.dtype() != DType::UInt64 &&
        input.dtype() != DType::Complex64 && input.dtype() != DType::Complex128 &&
        input.dtype() != DType::Int32 && input.dtype() != DType::Bool) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchIndexSelect(input_f32, dim, indices);
        return result_f32.to(orig_dtype);
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Materialize read operands to packed offset-0 buffers before binding.
    const Tensor input_c = dispatchContiguous(input);
    const Tensor indices_packed = dispatchContiguous(indices);

    // Convert indices to Int32 if needed (shader expects int32, on-device)
    Tensor indices_int32 = indices_packed;
    if (indices.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(indices.shape().begin(), indices.shape().end());
        indices_int32 = Tensor(idx_shape, DType::Int32, indices.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = indices_packed.data_ptr();
        const void* buf_out = indices_int32.data_ptr();
        size_t size_in = indices.numel() * sizeof(int64_t);
        size_t size_out = indices_int32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(indices.numel());

        uint32_t cast_groups = div_wg(cast_pc.n_elements, devices_[device_id].workgroupSize);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, cast_groups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Calculate strides
    uint32_t dim_size = static_cast<uint32_t>(input_shape[dim]);
    uint32_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) {
        inner_size *= static_cast<uint32_t>(input_shape[i]);
    }
    uint32_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= static_cast<uint32_t>(input_shape[i]);
    }

    // Set up push constants
    struct PushConstants {
        uint32_t num_indices;
        uint32_t dim;
        uint32_t dim_size;
        uint32_t inner_size;
        uint32_t outer_size;
    } push_constants;

    push_constants.num_indices = static_cast<uint32_t>(indices.numel());
    push_constants.dim = static_cast<uint32_t>(dim);
    push_constants.dim_size = dim_size;
    push_constants.inner_size = inner_size;
    push_constants.outer_size = outer_size;

    // Get VkBuffer handles
    const void* buffer_input = input_c.data_ptr();
    const void* buffer_indices = indices_int32.data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes. F16/BF16 use the packed two-halves-per-word shaders
    // (index_select_f16/bf16), which read element e at word e/2; size the SSBO
    // ranges as ((numel + 1) / 2) * 4 so an odd-numel last word stays in bounds.
    // i64/c128 paths move bit-exact via native-width shaders and are unaffected.
    const bool isel_packed_half =
        (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
    size_t input_size = isel_packed_half
        ? ((input.numel() + 1) / 2) * 4
        : input.numel() * input.dtype_size();
    size_t indices_size = indices_int32.numel() * indices_int32.dtype_size();
    size_t output_size = isel_packed_half
        ? ((output.numel() + 1) / 2) * 4
        : output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_indices},
        {2, buffer_output}
    };
    std::vector<size_t> sizes = {input_size, indices_size, output_size};

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
    uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

} // namespace tenzor
