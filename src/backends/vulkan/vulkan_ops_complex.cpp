#include "vulkan_ops_common.hpp"

namespace tenzor {

// ============================================================================
// Complex Number Operations
// ============================================================================

auto VulkanBackend::dispatchConj(const Tensor& input) -> Tensor {
    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // For real (non-complex) dtypes, conjugate is the identity.
    // Matches CPU kernel semantics (see conj_kernel in cpu/kernels/math.cpp).
    if (input.dtype() != DType::Complex64 && input.dtype() != DType::Complex128) {
        return dispatchClone(input);
    }

    // Only Complex64/Complex128 reach here (all real dtypes returned above).
    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Complex128);

    if (is_float64) {
        vulkan::ensure_fp64_supported(device_id, "Conj");
    }
    std::string shader_name = is_float64 ? "conj_f64" : "conj";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    Tensor output(out_shape, input.dtype(), input.device());

    // The shader indexes by FLOAT slot, not by complex element. numel() is the
    // complex element count, so we dispatch 2x to cover both real and imag slots.
    uint32_t num_elements = static_cast<uint32_t>(input.numel() * 2);

    struct { uint32_t num_elements; } pushConstants;
    pushConstants.num_elements = num_elements;

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size = input.numel() * input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(num_elements, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchReal(const Tensor& input) -> Tensor {
    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // For real (non-complex) dtypes, real() is the identity.
    // Matches CPU kernel semantics (see real_kernel in cpu/kernels/math.cpp).
    if (input.dtype() != DType::Complex64 && input.dtype() != DType::Complex128) {
        return dispatchClone(input);
    }

    // Complex64/128 fast path: input is a true complex tensor (numel =
    // element count, dtype_size = 2 floats). Output is the same shape but
    // Float32/Float64. Extract real parts by walking the interleaved
    // storage two floats at a time.
    if (input.dtype() == DType::Complex64 || input.dtype() == DType::Complex128) {
        int32_t device_id = input.device().index;
        bool is_c128 = (input.dtype() == DType::Complex128);
        if (is_c128) {
            vulkan::ensure_fp64_supported(device_id, "Real");
        }
        std::string shader_name = is_c128 ? "complex_to_real_f64" : "complex_to_real";
        auto* pipeline = getPipeline(shader_name, device_id);

        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        DType out_dtype = is_c128 ? DType::Float64 : DType::Float32;
        Tensor output(out_shape, out_dtype, input.device());

        int64_t num_complex = input.numel();
        struct { uint32_t num_complex; } pc{static_cast<uint32_t>(num_complex)};

        const void* buffer_in  = input.data_ptr();
        const void* buffer_out = output.data_ptr();
        size_t in_size  = input.numel()  * input.dtype_size();
        size_t out_size = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_in}, {1, buffer_out}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_complex, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    // Unreachable: every dtype is either real (cloned above) or
    // Complex64/Complex128 (handled by the fast path above).
    return dispatchClone(input);
}

auto VulkanBackend::dispatchImag(const Tensor& input) -> Tensor {
    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // For real (non-complex) dtypes, imag() returns a zero tensor of same shape/dtype.
    // Matches CPU kernel semantics (see imag_kernel in cpu/kernels/math.cpp).
    if (input.dtype() != DType::Complex64 && input.dtype() != DType::Complex128) {
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        Tensor output(out_shape, input.dtype(), input.device());
        return dispatchFill(output, 0.0f);
    }

    // Complex64/128 fast path (mirrors dispatchReal — see comment there).
    // Uses the existing imag / imag_f64 shaders which read the second
    // float of each interleaved (re, im) pair.
    if (input.dtype() == DType::Complex64 || input.dtype() == DType::Complex128) {
        int32_t device_id = input.device().index;
        bool is_c128 = (input.dtype() == DType::Complex128);
        if (is_c128) {
            vulkan::ensure_fp64_supported(device_id, "Imag");
        }
        std::string shader_name = is_c128 ? "imag_f64" : "imag";
        auto* pipeline = getPipeline(shader_name, device_id);

        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        DType out_dtype = is_c128 ? DType::Float64 : DType::Float32;
        Tensor output(out_shape, out_dtype, input.device());

        int64_t num_complex = input.numel();
        struct { uint32_t num_complex; } pc{static_cast<uint32_t>(num_complex)};

        const void* buffer_in  = input.data_ptr();
        const void* buffer_out = output.data_ptr();
        size_t in_size  = input.numel()  * input.dtype_size();
        size_t out_size = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_in}, {1, buffer_out}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_complex, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    // Unreachable: every dtype is either real (zero-filled above) or
    // Complex64/Complex128 (handled by the fast path above).
    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    Tensor output(out_shape, input.dtype(), input.device());
    return dispatchFill(output, 0.0f);
}

auto VulkanBackend::dispatchAngle(const Tensor& input) -> Tensor {
    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // For real (non-complex) dtypes, angle(x) = atan2(0, x).
    // Matches CPU kernel semantics (see angle_kernel in cpu/kernels/math.cpp).
    if (input.dtype() != DType::Complex64 && input.dtype() != DType::Complex128) {
        // Float16/BFloat16: widen to Float32, compute, narrow back (many backend
        // kernels lack half-precision dispatch paths for binary ops).
        if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
            DType orig = input.dtype();
            Tensor x_f32 = input.to(DType::Float32);
            std::vector<int64_t> out_shape(x_f32.shape().begin(), x_f32.shape().end());
            Tensor zeros_t(out_shape, DType::Float32, x_f32.device());
            zeros_t = dispatchFill(zeros_t, 0.0f);
            Tensor r = dispatchBinaryOp("atan2", zeros_t, x_f32);
            return r.to(orig);
        }
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        Tensor zeros_t(out_shape, input.dtype(), input.device());
        zeros_t = dispatchFill(zeros_t, 0.0f);
        return dispatchBinaryOp("atan2", zeros_t, input);
    }

    // Complex64/128 fast path — mirrors dispatchReal. Shape preserved,
    // output dtype is the corresponding real. Reuses existing angle /
    // angle_f64 shaders which read (re, im) pairs.
    if (input.dtype() == DType::Complex64 || input.dtype() == DType::Complex128) {
        int32_t device_id = input.device().index;
        bool is_c128 = (input.dtype() == DType::Complex128);
        if (is_c128) {
            vulkan::ensure_fp64_supported(device_id, "Angle");
        }
        std::string shader_name = is_c128 ? "angle_f64" : "angle";
        auto* pipeline = getPipeline(shader_name, device_id);

        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        DType out_dtype = is_c128 ? DType::Float64 : DType::Float32;
        Tensor output(out_shape, out_dtype, input.device());

        int64_t num_complex = input.numel();
        struct { uint32_t num_complex; } pc{static_cast<uint32_t>(num_complex)};

        size_t in_size  = input.numel()  * input.dtype_size();
        size_t out_size = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_complex, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    // Unreachable: every dtype is either real (atan2 path above) or
    // Complex64/Complex128 (handled by the fast path above).
    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    Tensor zeros_t(out_shape, input.dtype(), input.device());
    zeros_t = dispatchFill(zeros_t, 0.0f);
    return dispatchBinaryOp("atan2", zeros_t, input);
}

auto VulkanBackend::dispatchPolar(const Tensor& abs, const Tensor& angle) -> Tensor {
    // Float16/BFloat16: upcast to Float32 and produce Complex64, matching
    // CPU kernel semantics (polar has no complex-half output dtype).
    if (abs.dtype() == DType::Float16 || abs.dtype() == DType::BFloat16) {
        return dispatchPolar(abs.to(DType::Float32), angle.to(DType::Float32));
    }

    // Float32/64 → Complex64/128 fast path — matches CPU semantics.
    // Shape is preserved (not doubled) and dtype promotes to the
    // corresponding complex type. Uses the existing polar / polar_f64
    // shaders which write interleaved (re, im) pairs.
    if (abs.dtype() == DType::Float32 || abs.dtype() == DType::Float64) {
        int32_t device_id = abs.device().index;
        bool is_f64 = (abs.dtype() == DType::Float64);
        if (is_f64) {
            vulkan::ensure_fp64_supported(device_id, "Polar");
        }
        std::string shader_name = is_f64 ? "polar_f64" : "polar";
        auto* pipeline = getPipeline(shader_name, device_id);

        std::vector<int64_t> out_shape(abs.shape().begin(), abs.shape().end());
        DType out_dtype = is_f64 ? DType::Complex128 : DType::Complex64;
        Tensor output(out_shape, out_dtype, abs.device());

        int64_t num_complex = abs.numel();
        struct { uint32_t num_complex; } pc{static_cast<uint32_t>(num_complex)};

        size_t real_size    = abs.numel()    * abs.dtype_size();
        size_t output_size  = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, abs.data_ptr()},
            {1, angle.data_ptr()},
            {2, output.data_ptr()},
        };
        std::vector<size_t> sizes = {real_size, real_size, output_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_complex, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    // Unreachable: Float16/BFloat16 are upcast+recursed above, and Float32/
    // Float64 are handled by the fast path above. Any other dtype is invalid
    // input for polar().
    throw std::runtime_error(
        "VulkanBackend::dispatchPolar: unsupported input dtype");
}


// ============================================================================
// ComplexTensor — create complex tensor from separate real + imag parts
// ============================================================================

auto VulkanBackend::dispatchComplexTensor(const Tensor& real, const Tensor& imag) -> Tensor {
    if (real.numel() == 0) {
        std::vector<int64_t> out_shape(real.shape().begin(), real.shape().end());
        DType out_dtype = (real.dtype() == DType::Float64) ? DType::Complex128 : DType::Complex64;
        return Tensor(out_shape, out_dtype, real.device());
    }

    // Float16/BFloat16: use native packed shaders
    if (real.dtype() == DType::Float16 || real.dtype() == DType::BFloat16) {
        bool is_bf16 = (real.dtype() == DType::BFloat16);
        int32_t device_id = real.device().index;
        auto* pipeline = getPipeline(is_bf16 ? "complex_from_parts_bf16" : "complex_from_parts_f16", device_id);

        int64_t num_complex = real.numel();
        std::vector<int64_t> out_shape(real.shape().begin(), real.shape().end());
        Tensor output(out_shape, DType::Complex64, real.device());

        struct { uint32_t num_complex; } pc;
        pc.num_complex = static_cast<uint32_t>(num_complex);

        // Packed F16/BF16 real buffers: (num_complex+1)/2 uint32 words each
        size_t real_buf_size = ((num_complex + 1) / 2) * 4;
        // Output is num_complex uint32 words (each holding one complex F16/BF16 pair)
        size_t out_size = num_complex * 4;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, real.data_ptr()}, {1, imag.data_ptr()}, {2, output.data_ptr()}
        };
        std::vector<size_t> sizes = {real_buf_size, real_buf_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_complex, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    int32_t device_id = real.device().index;
    bool is_float64 = (real.dtype() == DType::Float64);

    if (is_float64) {
        vulkan::ensure_fp64_supported(device_id, "ComplexTensor");
    }
    std::string shader_name = is_float64 ? "complex_from_parts_f64" : "complex_from_parts";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t num_elements = real.numel();
    DType out_dtype = is_float64 ? DType::Complex128 : DType::Complex64;
    std::vector<int64_t> out_shape(real.shape().begin(), real.shape().end());
    Tensor output(out_shape, out_dtype, real.device());

    struct { uint32_t num_elements; } pushConstants;
    pushConstants.num_elements = static_cast<uint32_t>(num_elements);

    size_t in_size = num_elements * real.dtype_size();
    size_t out_size = num_elements * 2 * real.dtype_size();  // complex = 2x real size

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, real.data_ptr()},
        {1, imag.data_ptr()},
        {2, output.data_ptr()}
    };
    std::vector<size_t> sizes_vec = {in_size, in_size, out_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes_vec);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(num_elements, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

} // namespace tenzor
