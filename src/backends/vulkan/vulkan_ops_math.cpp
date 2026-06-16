#include "vulkan_ops_common.hpp"
#include "tenzor/backend/attr_macros.hpp"
#include <unordered_set>
#include <cmath>
#include <cstdint>

namespace tenzor {

auto VulkanBackend::dispatchBinaryOp(const std::string& op_name,
                                     const Tensor& a_raw, const Tensor& b_raw) -> Tensor {
    // audit-2026-05-03 bug #15 mirror: ensure inputs are contiguous so that
    // the SSBO uploads see logically-contiguous data. Non-contiguous slice
    // views would otherwise produce wrong values (same root cause as the
    // CPU/CUDA/ROCm sum/sigmoid/tanh kernel fixes).
    auto a = a_raw.contiguous();
    auto b = b_raw.contiguous();
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    // Convert to vectors for easier manipulation
    std::vector<int64_t> shape_a_vec(a_shape.begin(), a_shape.end());
    std::vector<int64_t> shape_b_vec(b_shape.begin(), b_shape.end());

    // Check if shapes are broadcastable
    if (!are_broadcastable(shape_a_vec, shape_b_vec)) {
        std::string err = "Tensors shapes are not broadcastable: [";
        for (size_t i = 0; i < shape_a_vec.size(); i++) {
            if (i > 0) err += ",";
            err += std::to_string(shape_a_vec[i]);
        }
        err += "] vs [";
        for (size_t i = 0; i < shape_b_vec.size(); i++) {
            if (i > 0) err += ",";
            err += std::to_string(shape_b_vec[i]);
        }
        err += "] (op=" + op_name + ")";
        throw std::runtime_error(err);
    }

    // Compute output shape
    std::vector<int64_t> output_shape = compute_broadcast_shape(shape_a_vec, shape_b_vec);

    // Handle empty tensors - no GPU work needed
    int64_t out_numel = 1;
    for (auto d : output_shape) out_numel *= d;
    if (out_numel == 0) {
        return Tensor(output_shape, a.dtype(), a.device());
    }

    // FP8 has insufficient precision for direct math; upcast to Float32, compute,
    // then downcast back. The native shaders have no FP8 paths.
    if (a.dtype() == DType::FP8_E4M3 || a.dtype() == DType::FP8_E5M2 ||
        a.dtype() == DType::FP8_E4M3FNUZ || a.dtype() == DType::FP8_E5M2FNUZ ||
        b.dtype() == DType::FP8_E4M3 || b.dtype() == DType::FP8_E5M2 ||
        b.dtype() == DType::FP8_E4M3FNUZ || b.dtype() == DType::FP8_E5M2FNUZ) {
        DType orig = a.dtype();
        Tensor a_f32 = (a.dtype() == DType::Float32) ? a : a.to(DType::Float32);
        Tensor b_f32 = (b.dtype() == DType::Float32) ? b : b.to(DType::Float32);
        Tensor result = dispatchBinaryOp(op_name, a_f32, b_f32);
        return (orig == DType::Float32) ? result : result.to(orig);
    }

    int32_t device_id = a.device().index;

    // Map operation name to opcode
    uint32_t opcode = 0;
    if (op_name == "add") opcode = 0;
    else if (op_name == "sub") opcode = 1;
    else if (op_name == "mul") opcode = 2;
    else if (op_name == "div") opcode = 3;
    else if (op_name == "atan2") opcode = 23;
    else if (op_name == "fmod") opcode = 24;
    else if (op_name == "remainder") opcode = 25;
    else if (op_name == "minimum") opcode = 26;
    else if (op_name == "maximum") opcode = 27;
    else if (op_name == "hypot") opcode = 33;
    else if (op_name == "copysign") opcode = 34;
    else if (op_name == "nextafter") opcode = 35;
    else if (op_name == "igamma") opcode = 38;
    else if (op_name == "igammac") opcode = 39;
    else if (op_name == "gcd") opcode = 36;
    else if (op_name == "lcm") opcode = 37;
    else if (op_name == "fmax") opcode = 42;
    else if (op_name == "fmin") opcode = 43;
    else if (op_name == "float_power") opcode = 47;
    else if (op_name == "xlog1py") opcode = 48;
    else if (op_name == "ldexp") opcode = 49;
    else throw std::runtime_error("Unknown binary operation: " + op_name);

    // Check if we can use the fast path (same-shape, no broadcasting needed)
    bool same_shape = std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end());
    bool is_complex64 = (a.dtype() == DType::Complex64);
    bool is_complex128 = (a.dtype() == DType::Complex128);

    // Complex number path: use complex_math / complex_math_f64 shaders
    // Complex types support add(0), sub(1), mul(2), div(3) with same-shape or broadcast operands
    if ((is_complex64 || is_complex128) && opcode <= 3) {
        Tensor output(output_shape, a.dtype(), a.device());

        // The complex shaders (same-shape and broadcast) index from physical
        // offset 0 and, in the broadcast path, derive strides purely from the
        // logical shape — i.e. they assume a contiguous, zero-offset layout.
        // Sliced/permuted views would otherwise read the wrong elements, so
        // materialize contiguous operands before binding.
        Tensor a_cx = (a.offset() != 0 || !a.is_contiguous()) ? dispatchContiguous(a) : a;
        Tensor b_cx = (b.offset() != 0 || !b.is_contiguous()) ? dispatchContiguous(b) : b;

        // Buffer sizes: complex elements are stored as interleaved real/imag pairs
        size_t buffer_size_a = a_cx.numel() * a_cx.dtype_size();
        size_t buffer_size_b = b_cx.numel() * b_cx.dtype_size();
        size_t buffer_size_out = out_numel * output.dtype_size();

        const void* buffer_a = a_cx.data_ptr();
        const void* buffer_b = b_cx.data_ptr();
        const void* buffer_out = output.data_ptr();

        if (!same_shape) {
            // Broadcasting path: use complex_math_broadcast / complex_math_broadcast_f64 shaders
            // The broadcast push-constant block holds strides_a[8]/strides_b[8]/
            // shape_out[8]; reject >8-D inputs that would overflow those arrays.
            if (output_shape.size() > 8 || shape_a_vec.size() > 8 || shape_b_vec.size() > 8) {
                throw std::runtime_error("Vulkan broadcast supports at most 8 dimensions");
            }
            std::string shader_name = is_complex128 ? "complex_math_broadcast_f64" : "complex_math_broadcast";
            auto* pipeline = getPipeline(shader_name, device_id);

            auto strides_a = compute_broadcast_strides(shape_a_vec, output_shape);
            auto strides_b = compute_broadcast_strides(shape_b_vec, output_shape);

            struct PushConstantsBroadcast {
                uint32_t output_size;
                uint32_t op;
                uint32_t _pad0;
                uint32_t ndim_a;
                uint32_t ndim_b;
                uint32_t ndim_out;
                uint32_t strides_a[8];
                uint32_t strides_b[8];
                uint32_t shape_out[8];
            } push_constants = {};

            push_constants.output_size = static_cast<uint32_t>(out_numel);
            push_constants.op = opcode;
            push_constants.ndim_a = static_cast<uint32_t>(shape_a_vec.size());
            push_constants.ndim_b = static_cast<uint32_t>(shape_b_vec.size());
            push_constants.ndim_out = static_cast<uint32_t>(output_shape.size());

            for (size_t i = 0; i < std::min(size_t(8), strides_a.size()); ++i) {
                push_constants.strides_a[i] = strides_a[i];
            }
            for (size_t i = 0; i < std::min(size_t(8), strides_b.size()); ++i) {
                push_constants.strides_b[i] = strides_b[i];
            }
            for (size_t i = 0; i < std::min(size_t(8), output_shape.size()); ++i) {
                push_constants.shape_out[i] = static_cast<uint32_t>(output_shape[i]);
            }

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, buffer_a}, {1, buffer_b}, {2, buffer_out}
            };
            std::vector<size_t> sizes = {buffer_size_a, buffer_size_b, buffer_size_out};

            VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
                device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
            vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                              VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(PushConstantsBroadcast), &push_constants);

            uint32_t workgroups = div_wg(out_numel, devices_[device_id].workgroupSize);
            vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

            insertComputeOnlyBarrier(cmdBuffer);
            endSingleTimeCommands(cmdBuffer, device_id);

            return output;
        }

        // Same-shape fast path
        std::string shader_name = is_complex128 ? "complex_math_f64" : "complex_math";
        auto* pipeline = getPipeline(shader_name, device_id);

        struct PushConstantsComplex {
            uint32_t n;
            uint32_t op;
        };
        PushConstantsComplex push_constants;
        push_constants.n = static_cast<uint32_t>(out_numel);
        push_constants.op = opcode;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_a}, {1, buffer_b}, {2, buffer_out}
        };
        std::vector<size_t> sizes = {buffer_size_a, buffer_size_b, buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsComplex), &push_constants);

        uint32_t workgroups = div_wg(out_numel, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    bool is_float32 = (a.dtype() == DType::Float32);
    bool is_float64 = (a.dtype() == DType::Float64);
    bool is_float16 = (a.dtype() == DType::Float16);
    bool is_bfloat16 = (a.dtype() == DType::BFloat16);
    bool is_int32 = (a.dtype() == DType::Int32);
    bool is_int64 = (a.dtype() == DType::Int64);

    // Sliced views have non-zero storage offsets.  The math shader indexes
    // from 0, so copy them to dedicated buffers via dispatchContiguous.
    Tensor a_op = (a.offset() != 0 || !a.is_contiguous()) ? dispatchContiguous(a) : a;
    Tensor b_op = (b.offset() != 0 || !b.is_contiguous()) ? dispatchContiguous(b) : b;

    if (same_shape && (is_float32 || is_float64 || is_float16 || is_bfloat16 || is_int32 || is_int64)) {
        // Fast path: use math shader for same-shape operations
        // Select shader based on dtype
        std::string shader_name = is_float64 ? "math_f64" : is_float16 ? "math_f16" : is_bfloat16 ? "math_bf16" : is_int32 ? "math_i32" : is_int64 ? "math_i64" : "math";
        auto* pipeline = getPipeline(shader_name, device_id);

        Tensor output(output_shape, a_op.dtype(), a_op.device());

        // Prepare push constants - use different structure for Float32/Float16 vs Float64
        struct PushConstantsF32 {
            uint32_t n;
            uint32_t op;
            float param;
        };
        struct PushConstantsF64 {
            uint32_t n;
            uint32_t op;
            double param;
        };

        // Initialize the appropriate structure based on dtype
        PushConstantsF32 push_constants_f32;
        PushConstantsF64 push_constants_f64;
        void* push_constants_ptr;
        size_t push_constants_size;

        if (is_float64 || is_int64) {
            // Float64 and Int64 shaders use 64-bit param field
            push_constants_f64.n = static_cast<uint32_t>(a_op.numel());
            push_constants_f64.op = opcode;
            push_constants_f64.param = 0.0;
            push_constants_ptr = &push_constants_f64;
            push_constants_size = sizeof(PushConstantsF64);
        } else {
            // Float32, Float16, BFloat16, Int32 use 32-bit param field
            push_constants_f32.n = static_cast<uint32_t>(a_op.numel());
            push_constants_f32.op = opcode;
            push_constants_f32.param = 0.0f;
            push_constants_ptr = &push_constants_f32;
            push_constants_size = sizeof(PushConstantsF32);
        }

        // Get VkBuffer handles
        const void* buffer_a = a_op.data_ptr();
        const void* buffer_b = b_op.data_ptr();
        const void* buffer_out = output.data_ptr();

        // Calculate buffer sizes
        // For Float16, the shader reads uint32 words (2 elements per word),
        // so descriptor ranges must be rounded up to 4-byte boundaries
        size_t buffer_size_a = a_op.numel() * a_op.dtype_size();
        size_t buffer_size_b = b_op.numel() * b_op.dtype_size();
        size_t buffer_size_out = output.numel() * output.dtype_size();
        if (is_float16 || is_bfloat16) {
            size_t num_pairs_a = (a_op.numel() + 1) / 2;
            size_t num_pairs_b = (b_op.numel() + 1) / 2;
            size_t num_pairs_out = (output.numel() + 1) / 2;
            buffer_size_a = num_pairs_a * 4;
            buffer_size_b = num_pairs_b * 4;
            buffer_size_out = num_pairs_out * 4;
        }

        // Allocate and write descriptor set
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_a}, {1, buffer_b}, {2, buffer_out}
        };
        std::vector<size_t> sizes = {buffer_size_a, buffer_size_b, buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        // Execute compute shader
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, push_constants_size, push_constants_ptr);

        // Float16/BFloat16 shader processes pairs of elements, so we need fewer workgroups
        uint32_t workgroups;
        if (is_float16 || is_bfloat16) {
            uint32_t num_pairs = (static_cast<uint32_t>(a_op.numel()) + 1) / 2;
            workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);
        } else {
            workgroups = div_wg(a_op.numel(), devices_[device_id].workgroupSize);
        }
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    } else {
        // Broadcasting path: use math_broadcast shader
        // The broadcast push-constant block holds strides_a[8]/strides_b[8]/
        // shape_out[8]; tensors with more than 8 dimensions would set ndim>8
        // while the arrays silently truncate, producing wrong index decoding.
        if (output_shape.size() > 8 || shape_a_vec.size() > 8 || shape_b_vec.size() > 8) {
            throw std::runtime_error("Vulkan broadcast supports at most 8 dimensions");
        }
        // Select shader based on dtype
        bool is_float64 = (a.dtype() == DType::Float64);
        bool is_float16 = (a.dtype() == DType::Float16);
        bool is_bfloat16_bc = (a.dtype() == DType::BFloat16);
        bool is_int8 = (a.dtype() == DType::Int8);
        bool is_uint8 = (a.dtype() == DType::UInt8);
        bool is_int64 = (a.dtype() == DType::Int64);
        bool is_bool = (a.dtype() == DType::Bool);
        std::string shader_name;
        if (is_float64) {
            shader_name = "math_broadcast_f64";
        } else if (is_float16) {
            shader_name = "math_broadcast_f16";
        } else if (is_bfloat16_bc) {
            shader_name = "math_broadcast_bf16";
        } else if (is_int8) {
            shader_name = "math_broadcast_i8";
        } else if (is_uint8) {
            shader_name = "math_broadcast_uint8";
        } else if (is_int64) {
            shader_name = "math_broadcast_i64";
        } else if (is_bool) {
            shader_name = "math_broadcast_bool";
        } else {
            shader_name = "math_broadcast";
        }
        auto* pipeline = getPipeline(shader_name, device_id);

        Tensor output(output_shape, a.dtype(), a.device());

        // Compute broadcasting strides
        auto strides_a = compute_broadcast_strides(shape_a_vec, output_shape);
        auto strides_b = compute_broadcast_strides(shape_b_vec, output_shape);

        // Prepare push constants for broadcasting shader
        struct PushConstantsBroadcast {
            uint32_t output_size;
            uint32_t op;
            uint32_t dtype;        // 0=float32, 1=int32
            uint32_t ndim_a;
            uint32_t ndim_b;
            uint32_t ndim_out;
            uint32_t strides_a[8];
            uint32_t strides_b[8];
            uint32_t shape_out[8];
        } push_constants = {};

        int64_t output_numel = 1;
        for (auto dim : output_shape) {
            output_numel *= dim;
        }

        // Determine dtype code
        // Note: for Float64, we use separate shader (math_broadcast_f64), so dtype field is unused
        // But we set it correctly for consistency
        uint32_t dtype_code = 0;  // 0=float32, 1=float64 (for f64 shader), 1=int32 (for regular shader)
        if (a.dtype() == DType::Float64) {
            dtype_code = 1;  // Float64 uses dtype=1 in math_broadcast_f64 shader
        } else if (a.dtype() == DType::Int32) {
            dtype_code = 1;  // Int32 uses dtype=1 in math_broadcast shader
        }

        push_constants.output_size = static_cast<uint32_t>(output_numel);
        push_constants.op = opcode;
        push_constants.dtype = dtype_code;
        push_constants.ndim_a = static_cast<uint32_t>(shape_a_vec.size());
        push_constants.ndim_b = static_cast<uint32_t>(shape_b_vec.size());
        push_constants.ndim_out = static_cast<uint32_t>(output_shape.size());

        // Copy strides and output shape (up to 8 dimensions)
        for (size_t i = 0; i < std::min(size_t(8), strides_a.size()); ++i) {
            push_constants.strides_a[i] = strides_a[i];
        }
        for (size_t i = 0; i < std::min(size_t(8), strides_b.size()); ++i) {
            push_constants.strides_b[i] = strides_b[i];
        }
        for (size_t i = 0; i < std::min(size_t(8), output_shape.size()); ++i) {
            push_constants.shape_out[i] = static_cast<uint32_t>(output_shape[i]);
        }

        // Get VkBuffer handles.  The broadcast shader derives physical strides
        // purely from the logical shape (compute_broadcast_strides above), i.e.
        // it assumes a contiguous, zero-offset layout.  A sliced/permuted view
        // would be indexed at the wrong physical elements, so bind the
        // contiguous, zero-offset operands (a_op/b_op) materialized earlier,
        // mirroring the same-shape fast path.
        const void* buffer_a = a_op.data_ptr();
        const void* buffer_b = b_op.data_ptr();
        const void* buffer_out = output.data_ptr();

        // Calculate buffer sizes
        // For Float16, the shader works with uint32 (packed pairs), so descriptor size needs
        // to cover the full uint32 reads/writes
        size_t buffer_size_a = a_op.numel() * a_op.dtype_size();
        size_t buffer_size_b = b_op.numel() * b_op.dtype_size();
        size_t buffer_size_out = output_numel * output.dtype_size();
        if (is_float16 || is_bfloat16_bc) {
            // Round up to 4-byte boundary (minimum uint32 size for shader access)
            size_t a_pairs = (a_op.numel() + 1) / 2;
            size_t b_pairs = (b_op.numel() + 1) / 2;
            size_t out_pairs = (output_numel + 1) / 2;
            buffer_size_a = a_pairs * 4;
            buffer_size_b = b_pairs * 4;
            buffer_size_out = out_pairs * 4;
        }

        // Allocate and write descriptor set
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_a}, {1, buffer_b}, {2, buffer_out}
        };
        std::vector<size_t> sizes = {buffer_size_a, buffer_size_b, buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        // Execute compute shader
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsBroadcast), &push_constants);

        // Calculate workgroups - Float16/BFloat16 processes 2 elements per thread
        uint32_t workgroups;
        if (is_float16 || is_bfloat16_bc) {
            uint32_t num_pairs = (output_numel + 1) / 2;
            workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);
        } else {
            workgroups = div_wg(output_numel, devices_[device_id].workgroupSize);
        }
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }
}

auto VulkanBackend::dispatchUnaryOp(const std::string& op_name,
                                    const Tensor& input_in) -> Tensor {
    // Handle empty tensors - no GPU work needed
    if (input_in.numel() == 0) {
        auto input_shape = input_in.shape();
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
        // abs(complex) produces a real-valued output even for empty inputs.
        DType empty_out_dtype = input_in.dtype();
        if (op_name == "abs" && input_in.dtype() == DType::Complex64)  empty_out_dtype = DType::Float32;
        if (op_name == "abs" && input_in.dtype() == DType::Complex128) empty_out_dtype = DType::Float64;
        return Tensor(output_shape, empty_out_dtype, input_in.device());
    }

    // Every shader below indexes its input buffer from element 0 (no per-binding
    // byte offset is plumbed into the descriptor or push constants). A sliced /
    // narrowed view carries a non-zero storage offset() and/or non-contiguous
    // strides; binding its raw data_ptr() would (a) write a raw byte offset into
    // the STORAGE_BUFFER descriptor — which Vulkan requires be a multiple of
    // minStorageBufferOffsetAlignment — and (b) index the wrong physical
    // elements. Materialize a contiguous, zero-offset operand first, mirroring
    // the same-shape binary fast path (dispatchBinaryOp).
    Tensor input = (input_in.offset() != 0 || !input_in.is_contiguous())
                       ? dispatchContiguous(input_in)
                       : input_in;

    int32_t device_id = input.device().index;

    // Integer sign: dedicated by-byte-width shaders (the float `math` shader
    // would reinterpret integer storage as float). -1/0/1 for signed, 0/1 for
    // unsigned — matches CPU/CUDA/ROCm/OneAPI.
    if (op_name == "sign") {
        const DType dt = input.dtype();
        int elem_bytes = 0;
        bool is_signed = false;
        bool is_int = true;
        switch (dt) {
            case DType::Int8:   elem_bytes=1; is_signed=true;  break;
            case DType::UInt8:  elem_bytes=1; is_signed=false; break;
            case DType::Int16:  elem_bytes=2; is_signed=true;  break;
            case DType::UInt16: elem_bytes=2; is_signed=false; break;
            case DType::Int32:  elem_bytes=4; is_signed=true;  break;
            case DType::UInt32: elem_bytes=4; is_signed=false; break;
            case DType::Int64:  elem_bytes=8; is_signed=true;  break;
            case DType::UInt64: elem_bytes=8; is_signed=false; break;
            default: is_int = false; break;
        }
        if (is_int) {
            const std::string shader = elem_bytes == 1 ? "sign_int8"
                                     : elem_bytes == 2 ? "sign_int16"
                                     : elem_bytes == 4 ? "sign_int32"
                                                       : "sign_int64";
            auto* pipeline = getPipeline(shader, device_id);
            std::vector<int64_t> oshape(input.shape().begin(), input.shape().end());
            Tensor output(oshape, input.dtype(), input.device());
            const int64_t numel = input.numel();
            const size_t buf = ((static_cast<size_t>(numel) * elem_bytes + 3) / 4) * 4;
            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, input.data_ptr()}, {1, output.data_ptr()}};
            std::vector<size_t> sizes = {buf, buf};
            VkDescriptorSet descriptorSet =
                allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
            VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
            struct SignPC { uint32_t n; uint32_t is_signed; } pc;
            pc.n = static_cast<uint32_t>(numel);
            pc.is_signed = is_signed ? 1u : 0u;
            vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            const int64_t units = elem_bytes == 1 ? (numel + 3) / 4
                                : elem_bytes == 2 ? (numel + 1) / 2
                                                  : numel;
            uint32_t workgroups = div_wg(static_cast<uint32_t>(units),
                                         devices_[device_id].workgroupSize);
            vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
            insertComputeOnlyBarrier(cmdBuffer);
            endSingleTimeCommands(cmdBuffer, device_id);
            return output;
        }
    }

    // Complex-input path for abs/neg/exp/log/sqrt/sin/cos. The generic `math`
    // and `trigonometric` shaders interpret memory as float and would corrupt
    // interleaved complex pairs; dispatch to dedicated complex_* shaders
    // instead.
    static const std::unordered_set<std::string> kComplexOps = {
        "abs", "neg", "exp", "log", "sqrt", "sin", "cos"
    };
    if (kComplexOps.count(op_name) &&
        (input.dtype() == DType::Complex64 || input.dtype() == DType::Complex128)) {
        bool is_f64 = (input.dtype() == DType::Complex128);
        int64_t numel = input.numel();
        auto input_shape = input.shape();
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());

        // abs emits a real tensor; neg/exp/log/sqrt/sin/cos stay complex.
        DType out_dtype = input.dtype();
        if (op_name == "abs") out_dtype = is_f64 ? DType::Float64 : DType::Float32;
        Tensor output(output_shape, out_dtype, input.device());

        std::string shader_name = "complex_" + op_name;
        if (is_f64) shader_name += "_f64";

        auto* pipeline = getPipeline(shader_name, device_id);

        size_t in_bytes  = static_cast<size_t>(numel) * 2 * (is_f64 ? 8 : 4);
        size_t out_bytes = (op_name == "abs")
            ? static_cast<size_t>(numel) * (is_f64 ? 8 : 4)
            : in_bytes;

        const void* buffer_in  = input.data_ptr();
        const void* buffer_out = output.data_ptr();

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_in}, {1, buffer_out}
        };
        std::vector<size_t> sizes = {in_bytes, out_bytes};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        struct { uint32_t num_elements; } pc{ static_cast<uint32_t>(numel) };
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // Create output tensor (convert span to vector)
    auto input_shape = input.shape();
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Determine shader and opcode based on operation
    std::string shader_name;
    uint32_t opcode = 0;

    // Trigonometric operations: 0=sin, 1=cos, 2=tan, 3=asin, 4=acos, 5=atan
    if (op_name == "sin") { shader_name = "trigonometric"; opcode = 0; }
    else if (op_name == "cos") { shader_name = "trigonometric"; opcode = 1; }
    else if (op_name == "tan") { shader_name = "trigonometric"; opcode = 2; }
    else if (op_name == "asin") { shader_name = "trigonometric"; opcode = 3; }
    else if (op_name == "acos") { shader_name = "trigonometric"; opcode = 4; }
    else if (op_name == "atan") { shader_name = "trigonometric"; opcode = 5; }
    // Hyperbolic operations: 0=sinh, 1=cosh, 2=tanh
    else if (op_name == "sinh") { shader_name = "hyperbolic"; opcode = 0; }
    else if (op_name == "cosh") { shader_name = "hyperbolic"; opcode = 1; }
    else if (op_name == "tanh") { shader_name = "hyperbolic"; opcode = 2; }
    // Math operations: 4=sqrt, 5=exp, 6=log, 7=neg, 8=abs, 10=sign
    else if (op_name == "sqrt") { shader_name = "math"; opcode = 4; }
    else if (op_name == "exp") { shader_name = "math"; opcode = 5; }
    else if (op_name == "log") { shader_name = "math"; opcode = 6; }
    else if (op_name == "neg") { shader_name = "math"; opcode = 7; }
    else if (op_name == "abs") { shader_name = "math"; opcode = 8; }
    else if (op_name == "sign") { shader_name = "math"; opcode = 10; }
    else if (op_name == "floor") { shader_name = "math"; opcode = 11; }
    else if (op_name == "ceil") { shader_name = "math"; opcode = 12; }
    else if (op_name == "round") { shader_name = "math"; opcode = 13; }
    else if (op_name == "trunc") { shader_name = "math"; opcode = 14; }
    else if (op_name == "reciprocal") { shader_name = "math"; opcode = 15; }
    else if (op_name == "log2") { shader_name = "math"; opcode = 16; }
    else if (op_name == "log10") { shader_name = "math"; opcode = 17; }
    else if (op_name == "log1p") { shader_name = "math"; opcode = 18; }
    else if (op_name == "exp2") { shader_name = "math"; opcode = 19; }
    else if (op_name == "expm1") { shader_name = "math"; opcode = 20; }
    else if (op_name == "erf") { shader_name = "math"; opcode = 21; }
    else if (op_name == "erfc") { shader_name = "math"; opcode = 22; }
    else if (op_name == "rsqrt") { shader_name = "math"; opcode = 28; }
    else if (op_name == "square") { shader_name = "math"; opcode = 29; }
    else if (op_name == "asinh") { shader_name = "math"; opcode = 30; }
    else if (op_name == "acosh") { shader_name = "math"; opcode = 31; }
    else if (op_name == "atanh") { shader_name = "math"; opcode = 32; }
    else if (op_name == "deg2rad") { shader_name = "math"; opcode = 44; }
    else if (op_name == "rad2deg") { shader_name = "math"; opcode = 45; }
    else if (op_name == "logit") { shader_name = "math"; opcode = 46; }
    else if (op_name == "frac") { shader_name = "math"; opcode = 50; }
    else if (op_name == "log_sigmoid") { shader_name = "math"; opcode = 51; }
    else if (op_name == "bitwise_not") { shader_name = "math"; opcode = 52; }
    else throw std::runtime_error("Unknown unary operation: " + op_name);

    // Select correct pipeline based on dtype for math operations
    if (shader_name == "math") {
        if (input.dtype() == DType::Float64) {
            shader_name = "math_f64";
        } else if (input.dtype() == DType::Int32) {
            shader_name = "math_i32";
        } else if (input.dtype() == DType::Int64) {
            shader_name = "math_i64";
        } else if (input.dtype() == DType::Float16) {
            shader_name = "math_f16";
        } else if (input.dtype() == DType::BFloat16) {
            shader_name = "math_bf16";
        }
    } else if (shader_name == "trigonometric") {
        if (input.dtype() == DType::Float16) {
            shader_name = "trigonometric_f16";
        } else if (input.dtype() == DType::Float64) {
            shader_name = "trigonometric_f64";
        } else if (input.dtype() == DType::BFloat16) {
            shader_name = "trigonometric_bf16";
        }
    } else if (shader_name == "hyperbolic") {
        if (input.dtype() == DType::Float16) {
            shader_name = "hyperbolic_f16";
        } else if (input.dtype() == DType::Float64) {
            shader_name = "hyperbolic_f64";
        } else if (input.dtype() == DType::BFloat16) {
            shader_name = "hyperbolic_bf16";
        }
    }

    auto* pipeline = getPipeline(shader_name, device_id);
    // Prepare push constants - use different structure based on shader type
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_trig_or_hyp = (shader_name == "trigonometric" || shader_name == "trigonometric_f16" || shader_name == "trigonometric_f64" || shader_name == "trigonometric_bf16" ||
                           shader_name == "hyperbolic" || shader_name == "hyperbolic_f16" || shader_name == "hyperbolic_f64" || shader_name == "hyperbolic_bf16");
    struct PushConstantsSimple {
        uint32_t n;
        uint32_t op;
    };
    struct PushConstantsF32 {
        uint32_t n;
        uint32_t op;
        float param;
    };
    struct PushConstantsF64 {
        uint32_t n;
        uint32_t op;
        double param;
    };

    PushConstantsSimple push_constants_simple;
    PushConstantsF32 push_constants_f32;
    PushConstantsF64 push_constants_f64;
    void* push_constants_ptr;
    size_t push_constants_size;

    if (is_trig_or_hyp) {
        push_constants_simple.n = static_cast<uint32_t>(input.numel());
        push_constants_simple.op = opcode;
        push_constants_ptr = &push_constants_simple;
        push_constants_size = sizeof(PushConstantsSimple);
    } else if (is_float64) {
        push_constants_f64.n = static_cast<uint32_t>(input.numel());
        push_constants_f64.op = opcode;
        push_constants_f64.param = 0.0;
        push_constants_ptr = &push_constants_f64;
        push_constants_size = sizeof(PushConstantsF64);
    } else {
        push_constants_f32.n = static_cast<uint32_t>(input.numel());
        push_constants_f32.op = opcode;
        push_constants_f32.param = 0.0f;
        push_constants_ptr = &push_constants_f32;
        push_constants_size = sizeof(PushConstantsF32);
    }

    // Get VkBuffer handles from tensor data pointers
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16/BFloat16 per uint32)
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    // Allocate and write descriptor set
    // For trigonometric/hyperbolic shaders: Binding 0: input, Binding 1: output
    // For math/math_f64 shaders: Binding 0: input, Binding 1: unused (set to input), Binding 2: output
    std::vector<std::pair<uint32_t, const void*>> bindings;
    std::vector<size_t> sizes;

    if (shader_name == "math" || shader_name == "math_f64" || shader_name == "math_i32" || shader_name == "math_f16" || shader_name == "math_bf16") {
        bindings = {
            {0, buffer_in},
            {1, buffer_in},  // Unary ops don't use binding 1, but descriptor set expects it
            {2, buffer_out}
        };
        sizes = {buffer_size_in, buffer_size_in, buffer_size_out};
    } else {
        // trigonometric, hyperbolic, and their F16 variants use simpler layout
        bindings = {
            {0, buffer_in},
            {1, buffer_out}
        };
        sizes = {buffer_size_in, buffer_size_out};
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, push_constants_size, push_constants_ptr);

    // For F16/BF16 packed-pair shaders, each thread processes 2 elements
    bool is_f16_packed = (shader_name == "math_f16" || shader_name == "trigonometric_f16" || shader_name == "hyperbolic_f16" ||
                          shader_name == "math_bf16" || shader_name == "trigonometric_bf16" || shader_name == "hyperbolic_bf16");
    uint32_t num_work_items = is_f16_packed ? static_cast<uint32_t>((input.numel() + 1) / 2) : static_cast<uint32_t>(input.numel());
    uint32_t workgroups = div_wg(num_work_items, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // No synchronize() here — see comment in dispatchReduction. The
    // returned tensor's data is consumed by either subsequent compute
    // ops (which chain through CB barriers + cross-submission semaphore)
    // or by host readback (which calls ensurePendingWorkComplete).

    return output;
}

auto VulkanBackend::dispatchUnaryOpWithParam(const std::string& op_name,
                                              const Tensor& input,
                                              double param) -> Tensor {
    int32_t device_id = input.device().index;

    // Integer pow: dedicated by-byte-width shaders (the float "math" shader would
    // reinterpret integer storage as float and corrupt it). Non-negative integer
    // exponents use exact modular exponentiation, matching CPU/CUDA/ROCm/OneAPI.
    {
        const DType dt = input.dtype();
        int elem_bytes = 0;
        bool is_signed = false;
        bool is_int = true;
        switch (dt) {
            case DType::Int8:   elem_bytes=1; is_signed=true;  break;
            case DType::UInt8:  elem_bytes=1; is_signed=false; break;
            case DType::Int16:  elem_bytes=2; is_signed=true;  break;
            case DType::UInt16: elem_bytes=2; is_signed=false; break;
            case DType::Int32:  elem_bytes=4; is_signed=true;  break;
            case DType::UInt32: elem_bytes=4; is_signed=false; break;
            case DType::Int64:  elem_bytes=8; is_signed=true;  break;
            case DType::UInt64: elem_bytes=8; is_signed=false; break;
            default: is_int = false; break;
        }
        if (is_int) {
            if (op_name != "pow") {
                throw std::runtime_error(
                    "Vulkan " + op_name + ": integer dtypes are only supported for pow");
            }
            const double e = static_cast<double>(param);
            const bool int_exp = (e == std::floor(e)) && e >= 0.0;
            const uint32_t k = static_cast<uint32_t>(int_exp ? static_cast<int64_t>(e) : 0);

            const std::string shader = elem_bytes == 1 ? "pow_int8"
                                     : elem_bytes == 2 ? "pow_int16"
                                     : elem_bytes == 4 ? "pow_int32"
                                                       : "pow_int64";
            auto* pipeline = getPipeline(shader, device_id);

            std::vector<int64_t> oshape(input.shape().begin(), input.shape().end());
            Tensor output(oshape, input.dtype(), input.device());
            const int64_t numel = input.numel();
            const size_t buf = ((static_cast<size_t>(numel) * elem_bytes + 3) / 4) * 4;

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, input.data_ptr()}, {1, output.data_ptr()}};
            std::vector<size_t> sizes = {buf, buf};
            VkDescriptorSet descriptorSet =
                allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
            struct PowPC {
                uint32_t n; uint32_t is_signed; uint32_t int_exp; uint32_t k; float e;
            } pc;
            pc.n = static_cast<uint32_t>(numel);
            pc.is_signed = is_signed ? 1u : 0u;
            pc.int_exp = int_exp ? 1u : 0u;
            pc.k = k;
            pc.e = static_cast<float>(e);
            vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

            const int64_t units = elem_bytes == 1 ? (numel + 3) / 4
                                : elem_bytes == 2 ? (numel + 1) / 2
                                                  : numel;
            uint32_t workgroups = div_wg(static_cast<uint32_t>(units),
                                         devices_[device_id].workgroupSize);
            vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
            insertComputeOnlyBarrier(cmdBuffer);
            endSingleTimeCommands(cmdBuffer, device_id);
            return output;
        }
    }

    // Select correct pipeline based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Float64) shader_name = "math_f64";
    else if (input.dtype() == DType::Float16) shader_name = "math_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "math_bf16";
    else shader_name = "math";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor (convert span to vector)
    auto input_shape = input.shape();
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Map operation name to opcode (see math.comp shader)
    // 0=add, 1=sub, 2=mul, 3=div, 4=sqrt, 5=exp, 6=log, 7=neg, 8=abs, 9=pow, 10=sign
    uint32_t opcode = 0;
    if (op_name == "pow") opcode = 9;
    else if (op_name == "logit") opcode = 46;
    else throw std::runtime_error("Unknown parameterized unary operation: " + op_name);

    // Prepare push constants - use different structure for Float32 vs Float64
    bool is_float64 = (input.dtype() == DType::Float64);
    struct PushConstantsF32 {
        uint32_t n;
        uint32_t op;
        float param;
    };
    struct PushConstantsF64 {
        uint32_t n;
        uint32_t op;
        double param;
    };

    PushConstantsF32 push_constants_f32;
    PushConstantsF64 push_constants_f64;
    void* push_constants_ptr;
    size_t push_constants_size;

    if (is_float64) {
        push_constants_f64.n = static_cast<uint32_t>(input.numel());
        push_constants_f64.op = opcode;
        push_constants_f64.param = param;
        push_constants_ptr = &push_constants_f64;
        push_constants_size = sizeof(PushConstantsF64);
    } else {
        push_constants_f32.n = static_cast<uint32_t>(input.numel());
        push_constants_f32.op = opcode;
        push_constants_f32.param = param;
        push_constants_ptr = &push_constants_f32;
        push_constants_size = sizeof(PushConstantsF32);
    }

    // Get VkBuffer handles from tensor data pointers
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16/BFloat16 per uint32)
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    // Allocate and write descriptor set
    // Binding 0: input, Binding 1: unused (set to input), Binding 2: output
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_in},  // Unary ops don't use binding 1, but descriptor set expects it
        {2, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, push_constants_size, push_constants_ptr);

    // For F16 / BF16 packed-pair shaders, each thread processes 2 elements
    const bool packed_pair = (shader_name == "math_f16" || shader_name == "math_bf16");
    uint32_t num_work_items = packed_pair
        ? static_cast<uint32_t>((input.numel() + 1) / 2)
        : static_cast<uint32_t>(input.numel());
    uint32_t workgroups = div_wg(num_work_items, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchTrigonometricOp(const std::string& op_name,
                                             const Tensor& input) -> Tensor {
    // Complex sin/cos need dedicated shaders because the generic
    // trigonometric shader treats memory as real-valued floats.
    if ((op_name == "sin" || op_name == "cos") &&
        (input.dtype() == DType::Complex64 || input.dtype() == DType::Complex128)) {
        return dispatchUnaryOp(op_name, input);
    }

    // Map operation name to opcode (see trigonometric.comp shader)
    // 0=sin, 1=cos, 2=tan, 3=asin, 4=acos, 5=atan
    uint32_t opcode = 0;
    if (op_name == "sin") opcode = 0;
    else if (op_name == "cos") opcode = 1;
    else if (op_name == "tan") opcode = 2;
    else if (op_name == "asin") opcode = 3;
    else if (op_name == "acos") opcode = 4;
    else if (op_name == "atan") opcode = 5;
    else throw std::runtime_error("Unknown trigonometric operation: " + op_name);

    int32_t device_id = input.device().index;
    std::string shader_name = "trigonometric";
    if (input.dtype() == DType::Float16) shader_name = "trigonometric_f16";
    else if (input.dtype() == DType::Float64) shader_name = "trigonometric_f64";
    else if (input.dtype() == DType::BFloat16) shader_name = "trigonometric_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape = input.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Prepare push constants
    struct PushConstants {
        uint32_t n;   // Number of elements
        uint32_t op;  // Operation code
    } push_constants;
    push_constants.n = static_cast<uint32_t>(input.numel());
    push_constants.op = opcode;

    // Get VkBuffer handles
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    const bool is_packed_half = (input.dtype() == DType::Float16 ||
                                 input.dtype() == DType::BFloat16);
    if (is_packed_half) {
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in}, {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // For packed-pair F16 / BF16, each thread processes 2 elements
    uint32_t num_work_items = is_packed_half
        ? static_cast<uint32_t>((input.numel() + 1) / 2)
        : static_cast<uint32_t>(input.numel());
    uint32_t workgroups = div_wg(num_work_items, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);
    // No synchronize() here — see comment in dispatchReduction.

    return output;
}

auto VulkanBackend::dispatchHyperbolicOp(const std::string& op_name,
                                          const Tensor& input_raw) -> Tensor {
    // audit-2026-05-03 bug #15 mirror: the shader iterates over input as a
    // contiguous element stream — non-contiguous slice/view inputs would
    // produce only the first contiguous run correctly and garbage afterwards
    // (LSTM `tanh(slice(gates))` was the failing path).
    auto input = input_raw.contiguous();

    // Map operation name to opcode (see hyperbolic.comp shader)
    // 0=sinh, 1=cosh, 2=tanh
    uint32_t opcode = 0;
    if (op_name == "sinh") opcode = 0;
    else if (op_name == "cosh") opcode = 1;
    else if (op_name == "tanh") opcode = 2;
    else throw std::runtime_error("Unknown hyperbolic operation: " + op_name);

    int32_t device_id = input.device().index;
    std::string shader_name = "hyperbolic";
    if (input.dtype() == DType::Float16) shader_name = "hyperbolic_f16";
    else if (input.dtype() == DType::Float64) shader_name = "hyperbolic_f64";
    else if (input.dtype() == DType::BFloat16) shader_name = "hyperbolic_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape = input.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Prepare push constants
    struct PushConstants {
        uint32_t n;   // Number of elements
        uint32_t op;  // Operation code
    } push_constants;
    push_constants.n = static_cast<uint32_t>(input.numel());
    push_constants.op = opcode;

    // Get VkBuffer handles
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    const bool is_packed_half = (input.dtype() == DType::Float16 ||
                                 input.dtype() == DType::BFloat16);
    if (is_packed_half) {
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in}, {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // For packed-pair F16 / BF16, each thread processes 2 elements
    uint32_t num_work_items = is_packed_half
        ? static_cast<uint32_t>((input.numel() + 1) / 2)
        : static_cast<uint32_t>(input.numel());
    uint32_t workgroups = div_wg(num_work_items, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);
    // No synchronize() here — see comment in dispatchReduction.

    return output;
}

auto VulkanBackend::dispatchComparisonOp(const std::string& op_name,
                                          const Tensor& a_in, const Tensor& b_in) -> Tensor {
    // Broadcast inputs to a common shape per numpy rules. Previously this
    // threw on any shape mismatch, which broke callers like MoE that pass
    // e.g. eq(idx_col_shape_N, scalar_shape_1).
    auto broadcast_shape = [](std::span<const int64_t> x, std::span<const int64_t> y)
        -> std::vector<int64_t> {
        std::vector<int64_t> out;
        auto xi = x.rbegin(); auto yi = y.rbegin();
        while (xi != x.rend() || yi != y.rend()) {
            int64_t dx = (xi != x.rend()) ? *xi : 1;
            int64_t dy = (yi != y.rend()) ? *yi : 1;
            if (dx != dy && dx != 1 && dy != 1) return {};
            out.push_back(std::max(dx, dy));
            if (xi != x.rend()) ++xi;
            if (yi != y.rend()) ++yi;
        }
        std::reverse(out.begin(), out.end());
        return out;
    };
    Tensor a = a_in; Tensor b = b_in;
    if (!std::equal(a.shape().begin(), a.shape().end(),
                    b.shape().begin(), b.shape().end())) {
        auto bcast = broadcast_shape(a.shape(), b.shape());
        if (bcast.empty()) {
            throw std::invalid_argument("Tensors not broadcastable for comparison op");
        }
        bool a_matches = std::equal(a.shape().begin(), a.shape().end(),
                                     bcast.begin(), bcast.end());
        bool b_matches = std::equal(b.shape().begin(), b.shape().end(),
                                     bcast.begin(), bcast.end());
        if (!a_matches) a = a_in.expand(bcast).contiguous();
        if (!b_matches) b = b_in.expand(bcast).contiguous();
    }
    auto a_shape = a.shape();

    // Handle empty tensors - no GPU work needed
    if (a.numel() == 0) {
        std::vector<int64_t> output_shape(a_shape.begin(), a_shape.end());
        return Tensor(output_shape, DType::Bool, a.device());
    }

    int32_t device_id = a.device().index;

    // Select shader based on input dtype
    std::string shader_name;
    switch (a.dtype()) {
        case DType::Bool:
            shader_name = "comparison_bool";
            break;
        case DType::Float64:
            shader_name = "comparison_f64";
            break;
        case DType::Int32:
            shader_name = "comparison_i32";
            break;
        case DType::Int64:
            shader_name = "comparison_i64";
            break;
        case DType::Float16:
            shader_name = "comparison_f16";
            break;
        case DType::BFloat16:
            shader_name = "comparison_bf16";
            break;
        default:
            // Float32 uses the default comparison shader
            shader_name = "comparison";
            break;
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor (convert span to vector)
    // Output is boolean values (DType::Bool stored as uint8 with 0 or 1)
    std::vector<int64_t> output_shape(a_shape.begin(), a_shape.end());
    Tensor output(output_shape, DType::Bool, a.device());

    // Map operation name to opcode (see comparison.comp shader)
    // 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge
    uint32_t opcode = 0;
    if (op_name == "eq") opcode = 0;
    else if (op_name == "ne") opcode = 1;
    else if (op_name == "lt") opcode = 2;
    else if (op_name == "le") opcode = 3;
    else if (op_name == "gt") opcode = 4;
    else if (op_name == "ge") opcode = 5;
    else throw std::runtime_error("Unknown comparison operation: " + op_name);

    // Prepare push constants
    struct PushConstants {
        uint32_t n;   // Number of elements
        uint32_t op;  // Operation code
    } push_constants;
    push_constants.n = static_cast<uint32_t>(a.numel());
    push_constants.op = opcode;

    // Get VkBuffer handles from tensor data pointers
    const void* buffer_a = a.data_ptr();
    const void* buffer_b = b.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_a = a.numel() * a.dtype_size();
    size_t buffer_size_b = b.numel() * b.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        // Round up to 4-byte boundary for uint32 shader access (2 halves per uint32)
        size_t a_pairs = (a.numel() + 1) / 2;
        size_t b_pairs = (b.numel() + 1) / 2;
        buffer_size_a = a_pairs * 4;
        buffer_size_b = b_pairs * 4;
        // Output is Bool (uint8_t per element), NOT packed — keep as-is
    }

    // Allocate and write descriptor set
    // Binding 0: input A, Binding 1: input B, Binding 2: output
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_a},
        {1, buffer_b},
        {2, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_a, buffer_size_b, buffer_size_out};

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
    uint32_t workgroups = div_wg(a.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier to ensure shader writes complete
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchReduction(const std::string& op_name,
                                      const Tensor& input_orig,
                                      int64_t dim, bool keepdim) -> Tensor {
    // The reduction shaders ([outer | reduce | inner] layout via
    // inner_size) already handle arbitrary reduction axes on a
    // contiguous input. The contiguous() below ensures the shader's
    // assumption holds.
    // Complex sum/mean: reduce the real and imaginary parts independently and
    // recombine. (Other reductions — max/min/prod — are undefined on complex.)
    if ((input_orig.dtype() == DType::Complex64 || input_orig.dtype() == DType::Complex128) &&
        (op_name == "sum" || op_name == "mean")) {
        Tensor re = dispatchReduction(op_name, dispatchReal(input_orig), dim, keepdim);
        Tensor im = dispatchReduction(op_name, dispatchImag(input_orig), dim, keepdim);
        return dispatchComplexTensor(re, im);
    }

    Tensor input = (input_orig.is_contiguous() && input_orig.offset() == 0) ? input_orig : dispatchContiguous(input_orig);
    // Special case: handle empty tensors
    if (input.numel() == 0) {
        // For empty tensors, return identity value
        // sum: 0, mean: 0, max: -inf, min: +inf
        // dispatchFull narrows the value to the target dtype; casting a float
        // infinity to an integer dtype is UB (commonly INT_MIN regardless of
        // sign), so for integer dtypes use the dtype's representable extreme,
        // matching CPU semantics (empty min -> dtype max, empty max -> dtype min).
        double identity_value = 0.0;
        const DType out_dtype = input.dtype();
        const bool is_integer =
            (out_dtype == DType::Int8 || out_dtype == DType::Int16 ||
             out_dtype == DType::Int32 || out_dtype == DType::Int64 ||
             out_dtype == DType::UInt8 || out_dtype == DType::UInt16 ||
             out_dtype == DType::UInt32 || out_dtype == DType::UInt64);
        if (op_name == "max") {
            if (is_integer) {
                // Identity for max is the smallest representable value.
                switch (out_dtype) {
                    case DType::Int8:   identity_value = static_cast<double>(std::numeric_limits<int8_t>::min()); break;
                    case DType::Int16:  identity_value = static_cast<double>(std::numeric_limits<int16_t>::min()); break;
                    case DType::Int32:  identity_value = static_cast<double>(std::numeric_limits<int32_t>::min()); break;
                    case DType::Int64:  identity_value = static_cast<double>(std::numeric_limits<int64_t>::min()); break;
                    default:            identity_value = 0.0; break;  // unsigned: min is 0
                }
            } else {
                identity_value = -std::numeric_limits<double>::infinity();
            }
        } else if (op_name == "min") {
            if (is_integer) {
                // Identity for min is the largest representable value.
                switch (out_dtype) {
                    case DType::Int8:   identity_value = static_cast<double>(std::numeric_limits<int8_t>::max()); break;
                    case DType::Int16:  identity_value = static_cast<double>(std::numeric_limits<int16_t>::max()); break;
                    case DType::Int32:  identity_value = static_cast<double>(std::numeric_limits<int32_t>::max()); break;
                    case DType::Int64:  identity_value = static_cast<double>(std::numeric_limits<int64_t>::max()); break;
                    case DType::UInt8:  identity_value = static_cast<double>(std::numeric_limits<uint8_t>::max()); break;
                    case DType::UInt16: identity_value = static_cast<double>(std::numeric_limits<uint16_t>::max()); break;
                    case DType::UInt32: identity_value = static_cast<double>(std::numeric_limits<uint32_t>::max()); break;
                    case DType::UInt64: identity_value = static_cast<double>(std::numeric_limits<uint64_t>::max()); break;
                    default:            identity_value = 0.0; break;
                }
            } else {
                identity_value = std::numeric_limits<double>::infinity();
            }
        }

        // Calculate output shape
        std::vector<int64_t> out_shape;
        if (dim < 0) {
            out_shape = {1};
        } else {
            auto input_shape = input.shape();
            out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
            if (keepdim) {
                out_shape[dim] = 1;
            } else {
                out_shape.erase(out_shape.begin() + dim);
            }
        }

        return dispatchFull(out_shape, identity_value, input.dtype());
    }

    int32_t device_id = input.device().index;
    auto input_shape = input.shape();

    // Handle dimension specification:
    // - INT64_MIN means "reduce all elements" (full reduction to scalar)
    // - Negative values like -1, -2 mean indexing from the end
    bool full_reduction = (dim == INT64_MIN);
    if (!full_reduction && dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    bool is_int32 = (input.dtype() == DType::Int32);
    // Int64 and UInt64 have no native reduction shader; reduce via Float64 and
    // narrow back to the original dtype (exact for the magnitudes that fit f64's
    // 53-bit mantissa). UInt64 is not promoted at the op layer, so it arrives here.
    bool is_int64 = (input.dtype() == DType::Int64 || input.dtype() == DType::UInt64);

    // Int64 has no native shader; convert to Float64 for reduction, then convert back
    DType orig_dtype = input.dtype();
    Tensor reduction_input = input;
    if (is_int64) {
        reduction_input = input.to(DType::Float64);
        is_float64 = true;
    }

    std::string shader_name;
    if (is_float64) {
        shader_name = "reduction_f64";
    } else if (is_float16) {
        shader_name = "reduction_f16";
    } else if (is_bfloat16) {
        shader_name = "reduction_bf16";
    } else if (is_int32) {
        shader_name = "reduction_i32";
    } else {
        // Use subgroup-optimized shader when device supports subgroup arithmetic.
        // EXCEPT max/min: subgroupMax/Min drop NaN, but IEEE/PyTorch require NaN
        // propagation — so force the explicit NaN-aware `reduction` shader.
        auto& ctx = devices_[device_id];
        if (ctx.hasSubgroupArithmetic && op_name != "max" && op_name != "min") {
            shader_name = "reduction_subgroup";
        } else {
            shader_name = "reduction";
        }
    }

    // Map operation name to opcode for push constants
    // 0=sum, 1=mean, 2=max, 3=min (see reduction.comp shader)
    uint32_t op_code = 0;
    if (op_name == "sum") op_code = 0;
    else if (op_name == "mean") op_code = 1;
    else if (op_name == "max") op_code = 2;
    else if (op_name == "min") op_code = 3;
    else {
        throw std::invalid_argument("Unknown reduction operation: " + op_name);
    }

    // --- Tree reduction fast path ---
    // For large full reductions on Float32 with subgroup arithmetic, use the
    // two-pass tree reduction shader which leverages subgroupAdd/Max/Min for
    // the first level and shared-memory tree reduction for the second level.
    // The "mean" op is handled by doing sum + divide at the end.
    auto& ctx_tree = devices_[device_id];
    bool use_tree_reduction = full_reduction
        && ctx_tree.hasSubgroupArithmetic
        && !is_float64 && !is_float16 && !is_bfloat16 && !is_int32 && !is_int64
        && reduction_input.numel() > 1024
        // max/min excluded: the tree path's subgroupMax/Min drop NaN.
        && (op_name == "sum" || op_name == "mean");

    if (use_tree_reduction) {
        // Tree reduction maps reduce_op: 0=sum, 1=max, 2=min
        uint32_t tree_reduce_op = 0;
        if (op_name == "sum" || op_name == "mean") tree_reduce_op = 0;
        else if (op_name == "max") tree_reduce_op = 1;
        else tree_reduce_op = 2;

        constexpr uint32_t WG_SIZE = 256;
        uint32_t input_size = static_cast<uint32_t>(reduction_input.numel());
        uint32_t num_workgroups = (input_size + WG_SIZE - 1) / WG_SIZE;

        // Pass 1: reduce input -> partial results (one per workgroup)
        Tensor partial({static_cast<int64_t>(num_workgroups)}, reduction_input.dtype(), reduction_input.device());

        auto* tree_pipeline = getPipeline("reduction_tree", device_id);

        const void* buf_in = reduction_input.data_ptr();
        const void* buf_partial = partial.data_ptr();
        size_t sz_in = input_size * reduction_input.dtype_size();
        size_t sz_partial = num_workgroups * partial.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings1 = {{0, buf_in}, {1, buf_partial}};
        std::vector<size_t> sizes1 = {sz_in, sz_partial};
        VkDescriptorSet ds1 = allocateAndWriteDescriptorSet(device_id, tree_pipeline, bindings1, sizes1);

        struct TreePushConstants {
            uint32_t input_size;
            uint32_t output_size;
            uint32_t reduce_op;
        } treePC;
        treePC.input_size = input_size;
        treePC.output_size = num_workgroups;
        treePC.reduce_op = tree_reduce_op;

        VkCommandBuffer cmd1 = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, tree_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
                               tree_pipeline->layout(), 0, 1, &ds1, 0, nullptr);
        vkCmdPushConstants(cmd1, tree_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(treePC), &treePC);
        vkCmdDispatch(cmd1, num_workgroups, 1, 1);
        insertComputeOnlyBarrier(cmd1);

        // Pass 2: reduce partial results -> single scalar
        // Calculate output shape for full reduction
        std::vector<int64_t> out_shape;
        if (keepdim) {
            out_shape.assign(input_shape.size(), 1);
        } else {
            out_shape = {};
        }
        Tensor output(out_shape, reduction_input.dtype(), reduction_input.device());

        const void* buf_out = output.data_ptr();
        size_t sz_out = std::max<size_t>(output.numel(), 1) * output.dtype_size();

        // Reuse the same pipeline for pass 2
        std::vector<std::pair<uint32_t, const void*>> bindings2 = {{0, buf_partial}, {1, buf_out}};
        std::vector<size_t> sizes2 = {sz_partial, sz_out};
        VkDescriptorSet ds2 = allocateAndWriteDescriptorSet(device_id, tree_pipeline, bindings2, sizes2);

        treePC.input_size = num_workgroups;
        treePC.output_size = 1;
        uint32_t pass2_wg = (num_workgroups + WG_SIZE - 1) / WG_SIZE;

        vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
                               tree_pipeline->layout(), 0, 1, &ds2, 0, nullptr);
        vkCmdPushConstants(cmd1, tree_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(treePC), &treePC);
        vkCmdDispatch(cmd1, pass2_wg, 1, 1);
        insertComputeOnlyBarrier(cmd1);

        endSingleTimeCommands(cmd1, device_id);
        // No synchronize() here — see comment in dispatchReduction.

        // For mean, divide by element count
        if (op_name == "mean") {
            float inv_n = 1.0f / static_cast<float>(input_size);
            // Create a scalar-shaped tensor matching output's shape for broadcast-free multiply
            auto scale_shape = out_shape.empty() ? std::vector<int64_t>{1} : out_shape;
            Tensor scale = dispatchFull(scale_shape, inv_n, output.dtype());
            if (out_shape.empty()) {
                // Reshape output to {1} for binary op, then back to scalar
                output = dispatchBinaryOp("mul", output.reshape({1}), scale).reshape({});
            } else {
                output = dispatchBinaryOp("mul", output, scale);
            }
        }

        if (orig_dtype != output.dtype()) {
            return output.to(orig_dtype);
        }
        return output;
    }

    // --- Standard reduction path ---
    // NOTE: The tree reduction fast path above (WG_SIZE=256 with subgroup arithmetic)
    // already handles large full reductions efficiently. The standard path below
    // dispatches one workgroup per output element. For per-dim reductions on large
    // reduce dimensions (>65536), a two-pass strategy similar to the tree reduction
    // could be beneficial: pass 1 reduces chunks to partial results, pass 2 reduces
    // partials to final output. This would improve bandwidth utilization by using
    // 256-thread workgroups with shared memory tree-reduction within each group.
    auto* pipeline = getPipeline(shader_name, device_id);

    // Calculate output shape
    std::vector<int64_t> out_shape;
    if (full_reduction) {
        // Full reduction: output is a scalar (shape {1} or {} depending on keepdim)
        if (keepdim) {
            out_shape.assign(input_shape.size(), 1);
        } else {
            out_shape = {};  // Scalar
        }
    } else {
        // Dimensional reduction
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, reduction_input.dtype(), reduction_input.device());

    // Get VkBuffer handles from tensor data pointers
    const void* buffer_in = reduction_input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = reduction_input.numel() * reduction_input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16 || is_bfloat16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16/BFloat16 per uint32)
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    // Allocate and write descriptor set
    // Binding 0: input, Binding 1: output (matches reduction.comp shader layout)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate inner_size (product of dimensions after reduction dimension)
    uint32_t inner_size = 1;
    if (!full_reduction) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants for reduction operation (must match shader layout)
    struct {
        uint32_t n;             // Total number of elements
        uint32_t reduce_size;   // Size of dimension to reduce
        uint32_t outer_size;    // Number of output elements
        uint32_t inner_size;    // Product of dimensions after reduction dim
        uint32_t op;            // Operation code (0=sum, 1=mean, 2=max, 3=min)
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(reduction_input.numel());
    pushConstants.reduce_size = full_reduction ? pushConstants.n : static_cast<uint32_t>(input_shape[dim]);
    pushConstants.outer_size = full_reduction ? 1 : static_cast<uint32_t>(output.numel());
    pushConstants.inner_size = inner_size;
    pushConstants.op = op_code;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch workgroups
    // For Float16: each workgroup handles TWO adjacent output elements (one packed word)
    uint32_t workgroups;
    if (is_float16) {
        workgroups = (pushConstants.outer_size + 1) / 2;
    } else {
        // Each workgroup has 256 threads and reduces one output element
        workgroups = pushConstants.outer_size;
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    // No synchronize() here: output is returned as a Tensor wrapping its
    // pending GPU work. Subsequent compute ops chain through the active
    // command buffer's barriers and the per-frame semaphore on submit;
    // host readbacks (e.g. .to(Device::cpu())) issue their own
    // ensurePendingWorkComplete. Calling synchronize() here resets the
    // device's descriptor and command pools, which is fatal under
    // multi-thread autograd accumulation: another thread that has just
    // allocated a descriptor set (between allocateAndWriteDescriptorSet
    // and beginSingleTimeCommands) will record vkCmdBindDescriptorSets
    // against a now-invalid handle and read garbage from its dispatch.

    // Convert back to original dtype if we did an Int64->Float64 conversion
    if (orig_dtype != output.dtype()) {
        return output.to(orig_dtype);
    }
    return output;
}

// isSimpleTranspose2D is defined in vulkan_helpers.hpp

auto VulkanBackend::dispatchMatmul(const Tensor& a_raw, const Tensor& b_raw) -> Tensor {
    // audit-2026-05-03 bug #15 mirror: ensure contiguous inputs.
    auto a = a_raw.contiguous();
    auto b = b_raw.contiguous();
    // Optimized matmul with proper buffer binding and tiled execution

    // Float16: upcast to Float32 for numerical stability
    // The matmul_f16 shader uses F32 accumulation but outputs F16, which can
    // overflow the F16 range (+-65504) for large reduction dimensions (K).
    // BFloat16 has the same exponent range as Float32 so no overflow risk.
    if (a.dtype() == DType::Float16) {
        DType orig_dtype = a.dtype();
        auto a_f32 = a.to(DType::Float32);
        auto b_f32 = b.to(DType::Float32);
        auto result_f32 = dispatchMatmul(a_f32, b_f32);
        return result_f32.to(orig_dtype);
    }

    // FP8: no native FP8 matmul shader; widen to Float32 and downcast result
    if (a.dtype() == DType::FP8_E4M3 || a.dtype() == DType::FP8_E5M2 ||
        a.dtype() == DType::FP8_E4M3FNUZ || a.dtype() == DType::FP8_E5M2FNUZ) {
        DType orig_dtype = a.dtype();
        auto a_f32 = a.to(DType::Float32);
        auto b_f32 = b.to(DType::Float32);
        auto result_f32 = dispatchMatmul(a_f32, b_f32);
        return result_f32.to(orig_dtype);
    }

    // Integer dtypes (except Int32, which has a native shader): no integer
    // matmul shader, so widen to Float32, compute, narrow. Exact while the
    // accumulated products fit Float32's 24-bit mantissa — same accepted
    // pattern as the F16/FP8 widen above.
    if (a.dtype() == DType::Int8  || a.dtype() == DType::Int16  ||
        a.dtype() == DType::Int64 || a.dtype() == DType::UInt8  ||
        a.dtype() == DType::UInt16 || a.dtype() == DType::UInt32 ||
        a.dtype() == DType::UInt64) {
        DType orig_dtype = a.dtype();
        auto result_f32 = dispatchMatmul(a.to(DType::Float32), b.to(DType::Float32));
        return result_f32.to(orig_dtype);
    }

    // Complex matmul via four real matmuls:
    //   Cr = Ar@Br - Ai@Bi,  Ci = Ar@Bi + Ai@Br.
    // (No native complex matmul shader; mirrors the Bmm/Dot complex paths.)
    if (a.dtype() == DType::Complex64 || a.dtype() == DType::Complex128) {
        Tensor Ar = dispatchReal(a), Ai = dispatchImag(a);
        Tensor Br = dispatchReal(b), Bi = dispatchImag(b);
        Tensor Cr = dispatchBinaryOp("sub", dispatchMatmul(Ar, Br), dispatchMatmul(Ai, Bi));
        Tensor Ci = dispatchBinaryOp("add", dispatchMatmul(Ar, Bi), dispatchMatmul(Ai, Br));
        return dispatchComplexTensor(Cr, Ci);
    }

    // Make A contiguous if needed (A is usually already contiguous)
    Tensor a_contig = (a.is_contiguous() && a.offset() == 0) ? a : dispatchContiguous(a);

    // For B, check if it's a simple transpose - we can handle that without copying
    // This is common in linear layers: weight.transpose(0, 1)
    // Check if B is a simple transpose (common for linear layers: weight.T)
    // If so, we can use the _bt shader variant instead of making a contiguous copy
    bool b_is_transposed = !b.is_contiguous() && isSimpleTranspose2D(b);
    Tensor b_for_compute = b_is_transposed ? b : ((b.is_contiguous() && b.offset() == 0) ? b : dispatchContiguous(b));

    auto a_shape = a_contig.shape();
    auto b_shape = b_for_compute.shape();

    // Handle 1D vector x 2D matrix case
    if (a_shape.size() == 1 && b_shape.size() == 2) {
        // Validate dimensions
        if (a_shape[0] != b_shape[0]) {
            throw std::invalid_argument("Incompatible dimensions for vector-matrix matmul");
        }

        // Treat 1D vector as row vector: (N,) -> (1, N)
        // Then matmul becomes: (1, N) x (N, K) = (1, K)
        // Finally squeeze to get (K,)
        Tensor a_2d = a_contig.unsqueeze(0);  // (N,) -> (1, N)
        Tensor result_2d = dispatchMatmul(a_2d, b_for_compute);  // (1, K)

        // Return as 1D tensor
        std::vector<int64_t> result_shape = {result_2d.shape()[1]};
        return result_2d.reshape(result_shape);
    }

    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::invalid_argument("Matmul requires 2D tensors");
    }
    if (a_shape[1] != b_shape[0]) {
        throw std::invalid_argument("Incompatible dimensions for matmul");
    }

    int32_t device_id = a_contig.device().index;

    // Select correct pipeline based on dtype and whether B is transposed
    bool is_float64 = (a_contig.dtype() == DType::Float64);
    bool is_float16 = (a_contig.dtype() == DType::Float16);
    bool is_int32 = (a_contig.dtype() == DType::Int32);

    // For Float64, use optimized shader for larger matrices (>= 64x64)
    // Note: optimized shader doesn't have a _bt variant yet, so don't use it for transposed
    bool use_optimized_f64 = is_float64 && !b_is_transposed && (a_shape[0] >= 64 && b_shape[1] >= 64);

    std::string shader_name;
    if (is_float64) {
        if (use_optimized_f64) {
            shader_name = "matmul_f64_optimized";
        } else {
            shader_name = b_is_transposed ? "matmul_f64_bt" : "matmul_f64";
        }
    } else if (is_float16) {
        shader_name = b_is_transposed ? "matmul_bt_f16" : "matmul_f16";
    } else if (a_contig.dtype() == DType::BFloat16) {
        shader_name = b_is_transposed ? "matmul_bt_bf16" : "matmul_bf16";
    } else if (is_int32) {
        // No _bt variant for I32 -- force B contiguous if transposed
        if (b_is_transposed) {
            b_for_compute = dispatchContiguous(b);
            b_is_transposed = false;
        }
        shader_name = "matmul_i32";
    } else {
        // Use tiled matmul with shared memory for large Float32 matrices
        // (non-transposed only; the tiled shader assumes row-major B)
        if (!b_is_transposed && a_shape[0] > 128 && b_shape[1] > 128) {
            shader_name = "matmul_tiled";
        } else {
            shader_name = b_is_transposed ? "matmul_bt" : "matmul";
        }
    }

    // Use vendor-specific workgroup tile sizes via specialization constants.
    // The shader declares layout(local_size_x_id=0, local_size_y_id=1) and
    // constant_id 0/1 for TILE_X/TILE_Y respectively.
    auto& ctx = devices_[device_id];
    auto [tile_x, tile_y] = recommended_workgroup_2d(ctx.vendor, OpKind::Matmul);

    VkSpecializationMapEntry specEntries[2] = {
        {0, 0,                    sizeof(uint32_t)},  // constant_id=0 -> TILE_X
        {1, sizeof(uint32_t),     sizeof(uint32_t)},  // constant_id=1 -> TILE_Y
    };
    uint32_t specData[2] = {tile_x, tile_y};

    auto* pipeline = getPipelineSpecialized(
        shader_name, device_id,
        {specEntries, specEntries + 2},
        specData, sizeof(specData));

    std::vector<int64_t> out_shape = {a_shape[0], b_shape[1]};
    Tensor output(out_shape, a_contig.dtype(), a_contig.device());

    // Get VkBuffer handles - for transposed B, we need to access the underlying storage
    const void* buffer_a = a_contig.data_ptr();
    void* b_data_ptr = b_is_transposed ? const_cast<void*>(b_for_compute.storage()->data())
                                        : b_for_compute.data_ptr();
    const void* buffer_b = b_data_ptr;
    const void* buffer_c = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_a = a_contig.numel() * a_contig.dtype_size();
    size_t buffer_size_b = b_for_compute.numel() * b_for_compute.dtype_size();
    size_t buffer_size_c = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_a},
        {1, buffer_b},
        {2, buffer_c}
    };
    std::vector<size_t> sizes = {buffer_size_a, buffer_size_b, buffer_size_c};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants for matrix dimensions
    struct PushConstants {
        uint32_t M;  // rows of A
        uint32_t N;  // cols of B
        uint32_t K;  // cols of A / rows of B
    } push_constants;

    push_constants.M = static_cast<uint32_t>(a_shape[0]);
    push_constants.N = static_cast<uint32_t>(b_shape[1]);
    push_constants.K = static_cast<uint32_t>(a_shape[1]);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Tile-based dispatch using vendor-specific tile sizes.
    // - Standard shaders: output tile covers TILE_X cols x TILE_Y rows
    // - Optimized Float64: fixed 32x32 output tiles (2x2 per thread, own tiling)
    // - Non-transposed Float16/BFloat16: each thread processes 2 columns (TILE_X*2 wide)
    // - Transposed (_bt) Float16/BFloat16: one element per thread, same as standard
    uint32_t workgroups_x, workgroups_y;
    if (use_optimized_f64) {
        // Optimized F64: 32x32 output tiles (its own tiling, not vendor-specialised yet)
        workgroups_x = (push_constants.N + 31) / 32;
        workgroups_y = (push_constants.M + 31) / 32;
    } else if ((is_float16 || a_contig.dtype() == DType::BFloat16) && !b_is_transposed) {
        // Non-transposed F16/BF16: each thread handles 2 adjacent columns
        workgroups_x = ((push_constants.N + 1) / 2 + tile_x - 1) / tile_x;
        workgroups_y = (push_constants.M + tile_y - 1) / tile_y;
    } else {
        // Standard / tiled / _bt matmul: output tile is TILE_X cols x TILE_Y rows
        workgroups_x = (push_constants.N + tile_x - 1) / tile_x;
        workgroups_y = (push_constants.M + tile_y - 1) / tile_y;
    }
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // Saturate FP16 output: clamp +-Inf to +-65504 to prevent NaN propagation
    return fp16_saturate_if_needed(*this, output);
}

auto VulkanBackend::dispatchBmm(const Tensor& a, const Tensor& b) -> Tensor {
    // Batched matrix multiplication: C[b, :, :] = A[b, :, :] @ B[b, :, :]
    // A: (batch, M, K), B: (batch, K, N), C: (batch, M, N)
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    if (a_shape.size() != 3 || b_shape.size() != 3) {
        throw std::invalid_argument("Bmm requires 3D tensors, got " +
            std::to_string(a_shape.size()) + "D and " +
            std::to_string(b_shape.size()) + "D");
    }

    int64_t batch = a_shape[0];
    int64_t M = a_shape[1];
    int64_t K = a_shape[2];
    int64_t N = b_shape[2];

    if (b_shape[0] != batch) {
        throw std::invalid_argument("Bmm batch dimensions must match: " +
            std::to_string(batch) + " vs " + std::to_string(b_shape[0]));
    }
    if (b_shape[1] != K) {
        throw std::invalid_argument("Bmm inner dimensions must match: " +
            std::to_string(K) + " vs " + std::to_string(b_shape[1]));
    }

    int32_t device_id = a.device().index;
    // Integer dtypes (except Int32): widen to Float32, compute, narrow. Exact
    // while products fit Float32's 24-bit mantissa (matches the matmul/F16/FP8
    // widen pattern). No native integer batched-matmul shader exists.
    if (a.dtype() == DType::Int8  || a.dtype() == DType::Int16  ||
        a.dtype() == DType::Int64 || a.dtype() == DType::UInt8  ||
        a.dtype() == DType::UInt16 || a.dtype() == DType::UInt32 ||
        a.dtype() == DType::UInt64) {
        DType orig_dtype = a.dtype();
        auto result_f32 = dispatchBmm(a.to(DType::Float32), b.to(DType::Float32));
        return result_f32.to(orig_dtype);
    }

    // Complex batched matmul via four real bmms:
    //   Cr = Ar@Br - Ai@Bi,  Ci = Ar@Bi + Ai@Br.
    if (a.dtype() == DType::Complex64 || a.dtype() == DType::Complex128) {
        Tensor Ar = dispatchReal(a), Ai = dispatchImag(a);
        Tensor Br = dispatchReal(b), Bi = dispatchImag(b);
        Tensor Cr = dispatchBinaryOp("sub", dispatchBmm(Ar, Br), dispatchBmm(Ai, Bi));
        Tensor Ci = dispatchBinaryOp("add", dispatchBmm(Ar, Bi), dispatchBmm(Ai, Br));
        return dispatchComplexTensor(Cr, Ci);
    }

    bool is_float64 = (a.dtype() == DType::Float64);
    bool is_float16 = (a.dtype() == DType::Float16);
    bool is_bfloat16 = (a.dtype() == DType::BFloat16);
    bool a_contig = a.is_contiguous();
    bool b_contig = b.is_contiguous();

    // Use strided shader when either input is non-contiguous to avoid
    // memory-wasting contiguous copies (saves ~hundreds of MB in attention backward)
    if (!a_contig || !b_contig) {
        std::string shader_name = is_float64 ? "bmm_strided_f64" : is_float16 ? "bmm_strided_f16" : is_bfloat16 ? "bmm_strided_bf16" : "bmm_strided";
        auto* pipeline = getPipeline(shader_name, device_id);

        std::vector<int64_t> out_shape = {batch, M, N};
        Tensor output(out_shape, a.dtype(), a.device());

        // Get VkBuffer handles - for non-contiguous tensors, data_ptr() points
        // to the underlying storage which getVulkanBuffer resolves correctly
        const void* buffer_a = a.data_ptr();
        const void* buffer_b = b.data_ptr();
        const void* buffer_c = output.data_ptr();

        // Buffer sizes: for non-contiguous tensors, compute the extent
        // (max addressable byte) from strides rather than numel * dtype_size
        auto a_strides = a.strides();
        auto b_strides = b.strides();
        size_t dtype_sz = a.dtype_size();

        // Extent = (shape[i]-1)*stride[i] for each dim + 1 element
        auto compute_extent = [&](const Tensor& t) -> size_t {
            auto sh = t.shape();
            auto st = t.strides();
            size_t extent = 1;
            for (size_t i = 0; i < sh.size(); i++) {
                if (sh[i] > 1) {
                    extent += (sh[i] - 1) * std::abs(st[i]);
                }
            }
            return extent * t.dtype_size();
        };

        size_t buffer_size_a = a_contig ? a.numel() * dtype_sz : compute_extent(a);
        size_t buffer_size_b = b_contig ? b.numel() * dtype_sz : compute_extent(b);
        size_t buffer_size_c = output.numel() * dtype_sz;

        // The bmm_strided_f16/bf16 shaders treat all buffers as packed uint[]
        // (word = elem/2 loads, 32-bit atomicCompSwap RMW on output). For odd
        // element counts the final word extends 2 bytes past an un-rounded
        // descriptor range -> OOB. Round F16/BF16 sizes up to a 4-byte
        // boundary, mirroring the contiguous path.
        if (is_float16 || is_bfloat16) {
            buffer_size_a = (buffer_size_a + 3) & ~size_t{3};
            buffer_size_b = (buffer_size_b + 3) & ~size_t{3};
            buffer_size_c = (buffer_size_c + 3) & ~size_t{3};
        }

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_a}, {1, buffer_b}, {2, buffer_c}
        };
        std::vector<size_t> sizes = {buffer_size_a, buffer_size_b, buffer_size_c};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        // Push constants with stride information
        struct StridedPushConstants {
            uint32_t batch;
            uint32_t M;
            uint32_t N;
            uint32_t K;
            uint32_t a_stride0;
            uint32_t a_stride1;
            uint32_t a_stride2;
            uint32_t b_stride0;
            uint32_t b_stride1;
            uint32_t b_stride2;
        } push_constants;

        push_constants.batch = static_cast<uint32_t>(batch);
        push_constants.M = static_cast<uint32_t>(M);
        push_constants.N = static_cast<uint32_t>(N);
        push_constants.K = static_cast<uint32_t>(K);
        push_constants.a_stride0 = static_cast<uint32_t>(a_strides[0]);
        push_constants.a_stride1 = static_cast<uint32_t>(a_strides[1]);
        push_constants.a_stride2 = static_cast<uint32_t>(a_strides[2]);
        push_constants.b_stride0 = static_cast<uint32_t>(b_strides[0]);
        push_constants.b_stride1 = static_cast<uint32_t>(b_strides[1]);
        push_constants.b_stride2 = static_cast<uint32_t>(b_strides[2]);

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(StridedPushConstants), &push_constants);

        uint32_t workgroups_x = (push_constants.N + 15) / 16;
        uint32_t workgroups_y = (push_constants.M + 15) / 16;
        uint32_t workgroups_z = push_constants.batch;
        vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // Contiguous fast path - use original shaders (no stride overhead)
    std::string shader_name = is_float64 ? "bmm_f64" : is_float16 ? "bmm_f16" : is_bfloat16 ? "bmm_bf16" : "bmm";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, M, N};
    Tensor output(out_shape, a.dtype(), a.device());

    // Get VkBuffer handles
    const void* buffer_a = a.data_ptr();
    const void* buffer_b = b.data_ptr();
    const void* buffer_c = output.data_ptr();

    // Calculate buffer sizes (round up to 4-byte boundary for F16 packed uint32 access)
    size_t buffer_size_a, buffer_size_b, buffer_size_c;
    if (is_float16) {
        buffer_size_a = ((a.numel() + 1) / 2) * 4;
        buffer_size_b = ((b.numel() + 1) / 2) * 4;
        buffer_size_c = ((output.numel() + 1) / 2) * 4;
    } else {
        buffer_size_a = a.numel() * a.dtype_size();
        buffer_size_b = b.numel() * b.dtype_size();
        buffer_size_c = output.numel() * output.dtype_size();
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_a},
        {1, buffer_b},
        {2, buffer_c}
    };
    std::vector<size_t> sizes = {buffer_size_a, buffer_size_b, buffer_size_c};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants for batched matrix dimensions
    struct PushConstants {
        uint32_t batch;  // batch size
        uint32_t M;      // rows of A
        uint32_t N;      // cols of B
        uint32_t K;      // cols of A / rows of B
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.M = static_cast<uint32_t>(M);
    push_constants.N = static_cast<uint32_t>(N);
    push_constants.K = static_cast<uint32_t>(K);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch: each workgroup handles 16x16 output elements
    // Z dimension is batch to avoid cross-batch tiling errors
    uint32_t workgroups_x = (push_constants.N + 15) / 16;
    uint32_t workgroups_y = (push_constants.M + 15) / 16;
    uint32_t workgroups_z = push_constants.batch;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // Saturate FP16 output: clamp +-Inf to +-65504 to prevent NaN propagation
    return fp16_saturate_if_needed(*this, output);
}

auto VulkanBackend::dispatchDot(const Tensor& a, const Tensor& b) -> Tensor {
    // Dot product: element-wise multiply followed by sum
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    if (a_shape.size() != 1 || b_shape.size() != 1) {
        throw std::invalid_argument("Dot product requires 1D tensors");
    }
    if (a_shape[0] != b_shape[0]) {
        throw std::invalid_argument("Dot product tensors must have same size");
    }

    // Complex dot: sum(a_i * b_i) without conjugation. Decompose into real dot
    // products on the real/imag parts: re = ar·br - ai·bi, im = ar·bi + ai·br.
    if (a.dtype() == DType::Complex64 || a.dtype() == DType::Complex128) {
        Tensor ar = dispatchReal(a), ai = dispatchImag(a);
        Tensor br = dispatchReal(b), bi = dispatchImag(b);
        Tensor re = dispatchBinaryOp("sub", dispatchDot(ar, br), dispatchDot(ai, bi));
        Tensor im = dispatchBinaryOp("add", dispatchDot(ar, bi), dispatchDot(ai, br));
        return dispatchComplexTensor(re, im);
    }

    // Integer dtypes (except Int32): widen to Float32, compute, narrow. Exact
    // while products fit Float32's 24-bit mantissa. The mul/sum building blocks
    // below lack integer paths for these widths.
    if (a.dtype() == DType::Int8  || a.dtype() == DType::Int16  ||
        a.dtype() == DType::Int64 || a.dtype() == DType::UInt8  ||
        a.dtype() == DType::UInt16 || a.dtype() == DType::UInt32 ||
        a.dtype() == DType::UInt64) {
        DType orig_dtype = a.dtype();
        auto result_f32 = dispatchDot(a.to(DType::Float32), b.to(DType::Float32));
        return result_f32.to(orig_dtype);
    }

    // Element-wise multiply
    Tensor product = dispatchBinaryOp("mul", a, b);

    // Sum all elements (dim=-1 means all dimensions, keepdim=false for scalar result)
    Tensor result = dispatchReduction("sum", product, 0, false);

    return result;
}

/**
 * @brief Conv2d Backward Input - Gradient w.r.t. input
 *
 * Computes gradient of input using transposed convolution (col2im pattern).
 * For each input pixel, accumulates gradients from all output positions
 * that used it during forward pass.
 */
auto VulkanBackend::dispatchConv2dBackwardInput(
    const Tensor& grad_output,
    const Tensor& weight,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    const std::vector<int64_t>& input_shape,
    int64_t groups) -> Tensor {

    // Extract dimensions
    auto grad_shape = grad_output.shape();
    auto weight_shape = weight.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t height_out = grad_shape[2];
    int64_t width_out = grad_shape[3];

    int64_t channels_in = input_shape[1];
    int64_t height_in = input_shape[2];
    int64_t width_in = input_shape[3];

    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // The conv2d_backward_input shaders compute grad_output flat indices in
    // 32-bit uint (e.g. conv2d_backward_input_f64.comp grad_out_idx); a tensor
    // with >UINT32_MAX elements would wrap the index and read out of bounds.
    // shaderInt64 is not enabled on the device, so reject up front.
    int64_t grad_input_numel = 1;
    for (int64_t d : input_shape) grad_input_numel *= d;
    if (grad_output.numel() > static_cast<int64_t>(UINT32_MAX) ||
        grad_input_numel > static_cast<int64_t>(UINT32_MAX)) {
        throw std::runtime_error(
            "Vulkan conv2d_backward_input: tensor too large for 32-bit indexing "
            "(grad_output elements " + std::to_string(grad_output.numel()) +
            ", grad_input elements " + std::to_string(grad_input_numel) + ")");
    }

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype (F16 uses native shader with F32 accumulation)
    std::string shader_name = "conv2d_backward_input";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv2d_backward_input_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "conv2d_backward_input_f16";
    } else if (grad_output.dtype() == DType::BFloat16) {
        shader_name = "conv2d_backward_input_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient input tensor
    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_weight = weight.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    // Calculate buffer sizes (round up to 4-byte boundary for F16 packed uint32 access)
    size_t buffer_size_grad_out, buffer_size_weight, buffer_size_grad_in;
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        buffer_size_grad_out = ((grad_output.numel() + 1) / 2) * 4;
        buffer_size_weight = ((weight.numel() + 1) / 2) * 4;
        buffer_size_grad_in = ((grad_input.numel() + 1) / 2) * 4;
    } else {
        buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
        buffer_size_weight = weight.numel() * weight.dtype_size();
        buffer_size_grad_in = grad_input.numel() * grad_input.dtype_size();
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},  // grad_output
        {1, buffer_weight},    // weight
        {2, buffer_grad_in}    // grad_input (output)
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_weight, buffer_size_grad_in};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants (groups support for depthwise/grouped convolutions)
    struct PushConstants {
        uint32_t batch;
        uint32_t channels_in;
        uint32_t channels_out;
        uint32_t height_in;
        uint32_t width_in;
        uint32_t height_out;
        uint32_t width_out;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t pad_h;
        uint32_t pad_w;
        uint32_t dil_h;
        uint32_t dil_w;
        uint32_t groups;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels_in = static_cast<uint32_t>(channels_in);
    push_constants.channels_out = static_cast<uint32_t>(channels_out);
    push_constants.height_in = static_cast<uint32_t>(height_in);
    push_constants.width_in = static_cast<uint32_t>(width_in);
    push_constants.height_out = static_cast<uint32_t>(height_out);
    push_constants.width_out = static_cast<uint32_t>(width_out);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.pad_h = static_cast<uint32_t>(pad_h);
    push_constants.pad_w = static_cast<uint32_t>(pad_w);
    push_constants.dil_h = static_cast<uint32_t>(dil_h);
    push_constants.dil_w = static_cast<uint32_t>(dil_w);
    push_constants.groups = static_cast<uint32_t>(groups);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (256 threads per workgroup as defined in shader)
    int64_t total_elements = batch * channels_in * height_in * width_in;
    uint32_t workgroups = static_cast<uint32_t>(div_wg(total_elements, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return fp16_saturate_if_needed(*this, grad_input);
}

/**
 * @brief Conv2d Backward Weight - Gradient w.r.t. weights
 *
 * Computes gradient of weights by correlating input patches with grad_output
 * across all batch samples and spatial positions.
 */
auto VulkanBackend::dispatchConv2dBackwardWeight(
    const Tensor& grad_output,
    const Tensor& input,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    const std::vector<int64_t>& weight_shape,
    int64_t groups) -> Tensor {

    // Extract dimensions
    auto grad_shape = grad_output.shape();
    auto input_shape = input.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t height_out = grad_shape[2];
    int64_t width_out = grad_shape[3];

    int64_t channels_in = input_shape[1];
    int64_t height_in = input_shape[2];
    int64_t width_in = input_shape[3];

    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // The conv2d_backward_weight shaders compute grad_output/input flat indices
    // in 32-bit uint; a tensor with >UINT32_MAX elements would wrap the index
    // and read/write out of bounds. shaderInt64 is not enabled on the device, so
    // reject oversized tensors up front rather than overflow silently.
    if (grad_output.numel() > static_cast<int64_t>(UINT32_MAX) ||
        input.numel() > static_cast<int64_t>(UINT32_MAX)) {
        throw std::runtime_error(
            "Vulkan conv2d_backward_weight: tensor too large for 32-bit indexing "
            "(grad_output elements " + std::to_string(grad_output.numel()) +
            ", input elements " + std::to_string(input.numel()) + ")");
    }

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype (F16 uses native shader with F32 accumulation)
    std::string shader_name = "conv2d_backward_weight";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv2d_backward_weight_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "conv2d_backward_weight_f16";
    } else if (grad_output.dtype() == DType::BFloat16) {
        shader_name = "conv2d_backward_weight_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient weight tensor
    Tensor grad_weight(weight_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_input = input.data_ptr();
    const void* buffer_grad_weight = grad_weight.data_ptr();

    // Calculate buffer sizes (round up to 4-byte boundary for F16 packed uint32 access)
    size_t buffer_size_grad_out, buffer_size_input, buffer_size_grad_weight;
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        buffer_size_grad_out = ((grad_output.numel() + 1) / 2) * 4;
        buffer_size_input = ((input.numel() + 1) / 2) * 4;
        buffer_size_grad_weight = ((grad_weight.numel() + 1) / 2) * 4;
    } else {
        buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
        buffer_size_input = input.numel() * input.dtype_size();
        buffer_size_grad_weight = grad_weight.numel() * grad_weight.dtype_size();
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},    // grad_output
        {1, buffer_input},       // input
        {2, buffer_grad_weight}  // grad_weight (output)
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_input, buffer_size_grad_weight};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants (groups support for depthwise/grouped convolutions)
    struct PushConstants {
        uint32_t batch;
        uint32_t channels_in;
        uint32_t channels_out;
        uint32_t height_in;
        uint32_t width_in;
        uint32_t height_out;
        uint32_t width_out;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t pad_h;
        uint32_t pad_w;
        uint32_t dil_h;
        uint32_t dil_w;
        uint32_t groups;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels_in = static_cast<uint32_t>(channels_in);
    push_constants.channels_out = static_cast<uint32_t>(channels_out);
    push_constants.height_in = static_cast<uint32_t>(height_in);
    push_constants.width_in = static_cast<uint32_t>(width_in);
    push_constants.height_out = static_cast<uint32_t>(height_out);
    push_constants.width_out = static_cast<uint32_t>(width_out);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.pad_h = static_cast<uint32_t>(pad_h);
    push_constants.pad_w = static_cast<uint32_t>(pad_w);
    push_constants.dil_h = static_cast<uint32_t>(dil_h);
    push_constants.dil_w = static_cast<uint32_t>(dil_w);
    push_constants.groups = static_cast<uint32_t>(groups);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (256 threads per workgroup as defined in shader)
    // Weight shape: (C_out, C_in/groups, K_h, K_w) - total elements is the product of these
    int64_t in_channels_per_group = channels_in / groups;
    int64_t total_weight_elements = channels_out * in_channels_per_group * kernel_h * kernel_w;
    uint32_t workgroups = static_cast<uint32_t>(div_wg(total_weight_elements, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return fp16_saturate_if_needed(*this, grad_weight);
}

/**
 * @brief Conv2d Backward Bias - Gradient w.r.t. bias
 *
 * Computes gradient of bias by summing grad_output across batch,
 * height, and width dimensions for each output channel.
 */
auto VulkanBackend::dispatchConv2dBackwardBias(const Tensor& grad_output) -> Tensor {
    // Extract dimensions
    auto grad_shape = grad_output.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t height_out = grad_shape[2];
    int64_t width_out = grad_shape[3];

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype (F16 uses native shader with F32 accumulation)
    std::string shader_name = "conv2d_backward_bias";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv2d_backward_bias_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "conv2d_backward_bias_f16";
    } else if (grad_output.dtype() == DType::BFloat16) {
        shader_name = "conv2d_backward_bias_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient bias tensor
    std::vector<int64_t> bias_shape = {channels_out};
    Tensor grad_bias(bias_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_grad_bias = grad_bias.data_ptr();

    // Calculate buffer sizes (round up to 4-byte boundary for F16 packed uint32 access)
    size_t buffer_size_grad_out, buffer_size_grad_bias;
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        buffer_size_grad_out = ((grad_output.numel() + 1) / 2) * 4;
        buffer_size_grad_bias = ((grad_bias.numel() + 1) / 2) * 4;
    } else {
        buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
        buffer_size_grad_bias = grad_bias.numel() * grad_bias.dtype_size();
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},   // grad_output
        {1, buffer_grad_bias}   // grad_bias (output)
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_grad_bias};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels_out;
        uint32_t height_out;
        uint32_t width_out;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels_out = static_cast<uint32_t>(channels_out);
    push_constants.height_out = static_cast<uint32_t>(height_out);
    push_constants.width_out = static_cast<uint32_t>(width_out);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (256 threads per workgroup, one thread per output channel)
    uint32_t workgroups = static_cast<uint32_t>(div_wg(channels_out, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_bias;
}

/**
 * @brief Im2col (unfold) operation - transforms image to column format
 *
 * Transforms (N,C,H,W) -> (N, C*K*K, L) where L=out_h*out_w
 * Used for efficient convolution via matrix multiplication
 */
auto VulkanBackend::dispatchIm2Col(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("im2col requires 4D input (N,C,H,W)");
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    // HH.6: per-axis kernel/stride/padding/dilation. The shader's PushConstants
    // now carry (kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h,
    // dil_w) so asymmetric tuples like kernel_size=(3,5) are honoured.
    const auto kernel_2d = ::tenzor::backend::attrs::read_2d(attrs,
        AttrKey::KernelSize, AttrKey::KernelSizeH, AttrKey::KernelSizeW, 0);
    const auto stride_2d   = ::tenzor::backend::attrs::stride_2d(attrs);
    const auto padding_2d  = ::tenzor::backend::attrs::padding_2d(attrs);
    const auto dilation_2d = ::tenzor::backend::attrs::dilation_2d(attrs);
    int64_t kernel_h = kernel_2d[0];
    int64_t kernel_w = kernel_2d[1];
    int64_t stride_h = stride_2d[0];
    int64_t stride_w = stride_2d[1];
    int64_t pad_h = padding_2d[0];
    int64_t pad_w = padding_2d[1];
    int64_t dil_h = dilation_2d[0];
    int64_t dil_w = dilation_2d[1];

    // Calculate output dimensions
    int64_t out_h = (height + 2*pad_h - dil_h*(kernel_h-1) - 1) / stride_h + 1;
    int64_t out_w = (width + 2*pad_w - dil_w*(kernel_w-1) - 1) / stride_w + 1;
    int64_t num_blocks = out_h * out_w;

    // Output shape: (N, C*K_h*K_w, L)
    std::vector<int64_t> out_shape = {batch, channels * kernel_h * kernel_w, num_blocks};
    Tensor output(out_shape, input.dtype(), input.device());

    int32_t device_id = input.device().index;
    std::string im2col_shader = (input.dtype() == DType::Float64) ? "im2col_f64"
                              : (input.dtype() == DType::Float16) ? "im2col_f16"
                              : (input.dtype() == DType::BFloat16) ? "im2col_bf16" : "im2col";
    auto* pipeline = getPipeline(im2col_shader, device_id);

    // Total elements to process
    int64_t total_elements = batch * channels * kernel_h * kernel_w * num_blocks;

    // Prepare buffers
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

    // Push constants (HH.6: per-axis kernel/stride/padding/dilation)
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t height;
        uint32_t width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t pad_h;
        uint32_t pad_w;
        uint32_t dil_h;
        uint32_t dil_w;
        uint32_t out_h;
        uint32_t out_w;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(total_elements);
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.height = static_cast<uint32_t>(height);
    push_constants.width = static_cast<uint32_t>(width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.pad_h = static_cast<uint32_t>(pad_h);
    push_constants.pad_w = static_cast<uint32_t>(pad_w);
    push_constants.dil_h = static_cast<uint32_t>(dil_h);
    push_constants.dil_w = static_cast<uint32_t>(dil_w);
    push_constants.out_h = static_cast<uint32_t>(out_h);
    push_constants.out_w = static_cast<uint32_t>(out_w);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (256 threads per workgroup)
    uint32_t workgroups = div_wg(total_elements, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);
    return output;
}

/**
 * @brief Col2im (fold) operation - transforms column format back to image
 *
 * Transforms (N, C*K*K, L) -> (N,C,H,W) with atomic accumulation
 * Inverse operation of im2col, accumulates overlapping values
 */
auto VulkanBackend::dispatchCol2Im(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    // U.7: col2im.comp (Fold / MaxUnpool path) uses atomicAdd(float) on the
    // output buffer (VK_EXT_shader_atomic_float). Fail fast on devices that
    // don't advertise it.
    vulkan::ensure_atomic_float_supported(input.device().index, "col2im");
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("col2im requires 3D input (N, C*K*K, L)");
    }

    int64_t batch = input_shape[0];

    // Extract attributes. The public ops::fold() wrapper (src/ops/vision.cpp)
    // stores the output H/W as a comma-separated AttrKey::OutputSize string
    // and computes channels from input_shape[1] / (kernel^2); it does NOT set
    // per-dim AttrKey::Channels / Height / Width. Previously we read those
    // missing keys as 0, producing empty output tensors.
    //
    // HH.6: per-axis kernel/stride/padding/dilation. The col2im*.comp shaders
    // now take (kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h,
    // dil_w) so asymmetric tuples are honoured rather than rejected.
    const auto kernel_2d_axes  = ::tenzor::backend::attrs::read_2d(attrs,
        AttrKey::KernelSize, AttrKey::KernelSizeH, AttrKey::KernelSizeW, 0);
    const auto stride_2d_axes  = ::tenzor::backend::attrs::stride_2d(attrs);
    const auto padding_2d_axes = ::tenzor::backend::attrs::padding_2d(attrs);
    const auto dilation_2d_axes= ::tenzor::backend::attrs::dilation_2d(attrs);
    int64_t kernel_h = kernel_2d_axes[0];
    int64_t kernel_w = kernel_2d_axes[1];
    int64_t stride_h = stride_2d_axes[0];
    int64_t stride_w = stride_2d_axes[1];
    int64_t pad_h = padding_2d_axes[0];
    int64_t pad_w = padding_2d_axes[1];
    int64_t dil_h = dilation_2d_axes[0];
    int64_t dil_w = dilation_2d_axes[1];

    int64_t height = 0, width = 0, channels = 0;
    if (attrs.has(AttrKey::OutputSize)) {
        auto parsed = attrs.get_int_list(AttrKey::OutputSize);
        if (parsed.size() != 2) {
            throw std::invalid_argument(
                "col2im: OutputSize must have 2 elements (H, W); got " +
                std::to_string(parsed.size()));
        }
        height = parsed[0];
        width = parsed[1];
    } else {
        height = attrs.get_int(AttrKey::Height);
        width = attrs.get_int(AttrKey::Width);
    }
    if (attrs.has(AttrKey::Channels)) {
        channels = attrs.get_int(AttrKey::Channels);
    } else {
        // Derive channels from input: input shape (N, C*K_h*K_w, L) so
        // C = second-dim / (K_h*K_w). HH.6: per-axis kernel dims.
        if (kernel_h <= 0 || kernel_w <= 0) {
            throw std::invalid_argument("col2im: kernel_h/kernel_w must be positive");
        }
        int64_t col_dim = input_shape[1];
        int64_t k_prod = kernel_h * kernel_w;
        if (col_dim % k_prod != 0) {
            throw std::invalid_argument(
                "col2im: second input dim (" + std::to_string(col_dim) +
                ") must be divisible by kernel_h*kernel_w (" + std::to_string(k_prod) + ")");
        }
        channels = col_dim / k_prod;
    }

    // Calculate output dimensions (HH.6: per-axis)
    int64_t out_h = (height + 2*pad_h - dil_h*(kernel_h-1) - 1) / stride_h + 1;
    int64_t out_w = (width  + 2*pad_w - dil_w*(kernel_w-1) - 1) / stride_w + 1;

    // Output shape: (N, C, H, W)
    std::vector<int64_t> out_shape = {batch, channels, height, width};
    Tensor output(out_shape, input.dtype(), input.device());

    // Initialize output to zero (required for atomic accumulation)
    int32_t device_id = input.device().index;
    auto* fill_pipeline = getPipeline("fill", device_id);

    const void* buffer_out = output.data_ptr();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Zero-initialize output buffer
    {
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buffer_out}};
        std::vector<size_t> sizes = {buffer_size_out};
        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, fill_pipeline, bindings, sizes);

        struct FillPushConstants {
            uint32_t n;
            float value;
        } fill_push;
        fill_push.n = static_cast<uint32_t>(output.numel());
        fill_push.value = 0.0f;

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, fill_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               fill_pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, fill_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(FillPushConstants), &fill_push);
        uint32_t fill_workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, fill_workgroups, 1, 1);
        // Barrier between fill and col2im accumulation to prevent WAW race
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);
    }

    // Now perform col2im operation - select dtype-specific variant
    bool is_col2im_f16  = (input.dtype() == DType::Float16);
    bool is_col2im_bf16 = (input.dtype() == DType::BFloat16);
    bool is_col2im_f64  = (input.dtype() == DType::Float64);
    std::string col2im_shader = is_col2im_f16  ? "col2im_f16"
                              : is_col2im_bf16 ? "col2im_bf16"
                              : is_col2im_f64  ? "col2im_f64"
                              : "col2im";
    auto* pipeline = getPipeline(col2im_shader, device_id);

    // Total elements to process (HH.6: per-axis K_h*K_w)
    int64_t col_channels = channels * kernel_h * kernel_w;
    int64_t num_blocks = out_h * out_w;
    int64_t total_elements = batch * col_channels * num_blocks;

    // Prepare buffers
    const void* buffer_in = input.data_ptr();

    size_t buffer_size_in = input.numel() * input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants (HH.6: per-axis kernel/stride/padding/dilation)
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t height;
        uint32_t width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t pad_h;
        uint32_t pad_w;
        uint32_t dil_h;
        uint32_t dil_w;
        uint32_t out_h;
        uint32_t out_w;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(total_elements);
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.height = static_cast<uint32_t>(height);
    push_constants.width = static_cast<uint32_t>(width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.pad_h = static_cast<uint32_t>(pad_h);
    push_constants.pad_w = static_cast<uint32_t>(pad_w);
    push_constants.dil_h = static_cast<uint32_t>(dil_h);
    push_constants.dil_w = static_cast<uint32_t>(dil_w);
    push_constants.out_h = static_cast<uint32_t>(out_h);
    push_constants.out_w = static_cast<uint32_t>(out_w);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (256 threads per workgroup)
    uint32_t workgroups = div_wg(total_elements, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);
    return output;
}

} // namespace tenzor
